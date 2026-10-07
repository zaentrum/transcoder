"""The cap: a source the planner would copy (HEVC Main or Main 10, 4:2:0,
at most 10 bits) is copied only within the cap of its kind and bucket
(2160 from 2000 lines or 3200 wide), and every source-size HEVC encode
is capped there — pure logic, no ffmpeg needed (the worker's part with
a fake probe and encode). Real runs are in test_encode_real.py."""

from __future__ import annotations

from pathlib import Path

import pytest
from structlog.testing import capture_logs

from transcoder import worker
from transcoder.decision import (
    CPU_ENCODERS,
    GIB,
    NVENC_ENCODERS,
    Caps,
    Encoders,
    SourceError,
    cap_bucket,
    parse_ladder,
    plan_renditions,
    video_bit_rate,
)
from transcoder.ffmpeg import EncodeSettings, RungResult
from transcoder.katalog import ClaimedItem

MBPS = 1_000_000
HOSTS = [(NVENC_ENCODERS, "hevc_nvenc"), (CPU_ENCODERS, "libx265"),
         (Encoders(hevc="libx265", h264="h264_nvenc"), "libx265")]
WIDTH = {1080: 1920, 2160: 3840}

# (kind, bucket, cap Mbit/s) by the owner's table: movies (and extras)
# 8 / 14, episodes 6 / 8.
TABLE = [("movie", 1080, 8), ("movie", 2160, 14), ("extra", 1080, 8), ("extra", 2160, 14),
         ("episode", 1080, 6), ("episode", 2160, 8)]


def _probe(codec: str = "hevc", width: int = 1920, height: int = 1080, *,
           rate: int | None = None, size: int | None = None, duration_ms: int = 0,
           audio: list[dict] | None = None, videos: list[dict] | None = None,
           **video: object) -> dict:
    v = {"codec_name": codec, "width": width, "height": height, "pix_fmt": "yuv420p",
         "profile": "Main" if codec == "hevc" else "", "avg_frame_rate": "24000/1001",
         **video}
    if rate is not None:
        v["bit_rate"] = str(rate)
    probe = {"video": v, "audio": audio or [], "subtitles": [], "duration_ms": duration_ms,
             "size_bytes": size}
    if videos is not None:
        probe["videos"] = videos
    return probe


def _plan(probe: dict, ladder: str = "", encoders: Encoders = NVENC_ENCODERS, **kw: object):
    return plan_renditions(probe, parse_ladder(ladder), encoders, **kw)


# ------------------------------------------------------------- the table
@pytest.mark.parametrize("encoders", [e for e, _ in HOSTS])
@pytest.mark.parametrize(("item_type", "bucket", "cap"), TABLE)
@pytest.mark.parametrize("ladder", ["", "source:hevc"])
@pytest.mark.parametrize("below", [0, 1, 4_000_000])
def test_hevc_within_its_cap_is_copied(
    encoders: Encoders, item_type: str, bucket: int, cap: int, ladder: str, below: int,
) -> None:
    # At the cap exactly, or under it: copied, as before.
    probe = _probe(width=WIDTH[bucket], height=bucket, rate=cap * MBPS - below)
    plan = _plan(probe, ladder, encoders, item_type=item_type)
    [v0] = plan.rungs
    assert (v0.mode, v0.reason) == ("copy", "source_already_hevc:hevc")
    assert (plan.cap.bucket, plan.cap.rate_bps, plan.cap.over) == (bucket, cap * MBPS, None)


@pytest.mark.parametrize(("encoders", "hevc"), HOSTS)
@pytest.mark.parametrize(("item_type", "bucket", "cap"), TABLE)
@pytest.mark.parametrize("ladder", ["", "source:hevc"])
@pytest.mark.parametrize("pix_fmt", ["yuv420p", "yuv420p10le"])
def test_hevc_above_its_cap_is_one_capped_encode_at_its_size(
    encoders: Encoders, hevc: str, item_type: str, bucket: int, cap: int, ladder: str,
    pix_fmt: str,
) -> None:
    probe = _probe(width=WIDTH[bucket], height=bucket, rate=cap * MBPS + 1, pix_fmt=pix_fmt,
                   profile="Main 10" if pix_fmt.endswith("10le") else "Main")
    plan = _plan(probe, ladder, encoders, item_type=item_type)
    [v0] = plan.rungs
    assert (v0.mode, v0.encoder, v0.codec, v0.width, v0.height, v0.scaled) == (
        "encode", hevc, "hevc", WIDTH[bucket], bucket, False)
    assert v0.reason == "hevc_over_cap:bitrate"
    # The cap is the maxrate (bufsize, twice it, comes with the command).
    assert v0.maxrate_bps == cap * MBPS
    # The bit depth is kept: Main 10 for the 10-bit source, Main for 8.
    assert v0.ten_bit is pix_fmt.endswith("10le")
    assert plan.cap.over == "bitrate" and plan.keyframes == "interval"


def test_an_hdr_source_above_its_cap_stays_10bit_hdr() -> None:
    probe = _probe(width=3840, height=2160, rate=40 * MBPS, pix_fmt="yuv420p10le",
                   profile="Main 10", color_transfer="smpte2084")
    [v0] = _plan(probe, "source:hevc").rungs
    assert (v0.mode, v0.ten_bit, v0.tonemap, v0.maxrate_bps) == ("encode", True, False, 14 * MBPS)


# ------------------------------------------------------------ the bucket
# 2160 from 2000 lines or from 3200 wide; 1080 otherwise.
@pytest.mark.parametrize(("size", "bucket"), [
    ((0, 0), (1080, None)), ((1280, 720), (1080, None)), ((1920, 1080), (1080, None)),
    ((1920, 800), (1080, None)), ((2560, 1440), (1080, None)), ((3199, 1999), (1080, None)),
    ((1920, 2000), (2160, "height")), ((3840, 2160), (2160, "height")),
    ((4096, 2160), (2160, "height")), ((7680, 4320), (2160, "height")),
    ((3200, 1350), (2160, "width")), ((3840, 1600), (2160, "width")),
    ((3840, 1606), (2160, "width")), ((4096, 1716), (2160, "width")),
])
def test_cap_bucket(size: tuple[int, int], bucket: tuple[int, str | None]) -> None:
    assert cap_bucket(*size) == bucket


@pytest.mark.parametrize(("encoders", "hevc"), HOSTS)
def test_a_scope_uhd_movie_takes_the_4k_cap(encoders: Encoders, hevc: str) -> None:
    # 3840x1600: under 2000 lines but 3200 wide, so the 2160 bucket and a
    # movie's 14 Mbit/s: copied at 12, encoded at 20 with maxrate 14.
    plan = _plan(_probe(width=3840, height=1600, rate=12 * MBPS), "", encoders)
    assert (plan.cap.bucket, plan.cap.bucket_by, plan.cap.rate_bps) == (2160, "width", 14 * MBPS)
    assert (plan.rungs[0].mode, plan.cap.over) == ("copy", None)
    plan = _plan(_probe(width=3840, height=1600, rate=20 * MBPS), "", encoders)
    [v0] = plan.rungs
    assert (v0.mode, v0.encoder, v0.width, v0.height, v0.maxrate_bps) == (
        "encode", hevc, 3840, 1600, 14 * MBPS)
    assert (plan.cap.over, plan.cap.bucket_by) == ("bitrate", "width")


def test_a_dci_4k_scope_movie_takes_the_4k_cap() -> None:
    plan = _plan(_probe(width=4096, height=1716, rate=13 * MBPS))
    assert (plan.cap.bucket, plan.cap.bucket_by, plan.rungs[0].mode) == (2160, "width", "copy")


@pytest.mark.parametrize(("rate", "mode"), [(7, "copy"), (9, "encode")])
def test_a_1920x800_movie_stays_in_the_1080_bucket(rate: int, mode: str) -> None:
    plan = _plan(_probe(width=1920, height=800, rate=rate * MBPS))
    assert (plan.cap.bucket, plan.cap.bucket_by, plan.cap.rate_bps) == (1080, None, 8 * MBPS)
    assert plan.rungs[0].mode == mode
    if mode == "encode":
        assert plan.rungs[0].maxrate_bps == 8 * MBPS


@pytest.mark.parametrize(("rate", "mode"), [(7, "copy"), (9, "encode")])
def test_a_scope_uhd_episode_takes_the_episodes_4k_cap(rate: int, mode: str) -> None:
    plan = _plan(_probe(width=3840, height=1600, rate=rate * MBPS), item_type="episode")
    assert (plan.cap.bucket, plan.cap.rate_bps, plan.rungs[0].mode) == (2160, 8 * MBPS, mode)


def test_a_scope_h264_encode_takes_the_4k_maxrate() -> None:
    # The width rule picks every source-size HEVC encode's maxrate too,
    # and NVENC_MAXRATE_2160P overrides it there.
    [v0] = _plan(_probe("h264", 3840, 1600), "source:hevc").rungs
    assert v0.maxrate_bps == 14 * MBPS
    [v0] = _plan(_probe("h264", 3840, 1600), "source:hevc", nvenc_caps_mbps=(10, 20)).rungs
    assert v0.maxrate_bps == 20 * MBPS


def test_the_bucket_is_the_tallest_and_the_widest_videos() -> None:
    # The cover art is no video (ffmpeg.ffprobe leaves it out of `videos`);
    # a second picture stream, taller or wider, is.
    plan = _plan(_probe(rate=12 * MBPS, videos=[{"height": 1080}, {"coded_height": 2160}]))
    assert (plan.cap.bucket, plan.cap.bucket_by, plan.cap.over) == (2160, "height", None)
    plan = _plan(_probe(rate=12 * MBPS, videos=[{"width": 1920, "height": 1080},
                                                {"coded_width": 3840, "height": 1600}]))
    assert (plan.cap.bucket, plan.cap.bucket_by, plan.cap.over) == (2160, "width", None)


# ---------------------------------------------------------- the size rule
@pytest.mark.parametrize(("item_type", "size", "over"), [
    ("movie", 15 * GIB + 1, "size"),
    ("movie", 15 * GIB, None),           # at the limit: kept
    ("extra", 16 * GIB, "size"),         # an extra takes the movies' rules
    ("episode", 40 * GIB, None),         # episodes have no size rule
    ("movie", None, None),               # an unknown size breaks nothing
])
@pytest.mark.parametrize(("encoders", "hevc"), HOSTS)
def test_a_movie_file_above_15_gib_is_encoded_whatever_its_bitrate(
    encoders: Encoders, hevc: str, item_type: str, size: int | None, over: str | None,
) -> None:
    plan = _plan(_probe(rate=2 * MBPS, size=size), "", encoders, item_type=item_type)
    [v0] = plan.rungs
    assert plan.cap.over == over
    if over:
        assert (v0.mode, v0.encoder, v0.reason, v0.maxrate_bps) == (
            "encode", hevc, "hevc_over_cap:size", 8 * MBPS)
    else:
        assert v0.mode == "copy"


def test_the_bitrate_rule_is_named_first_when_both_break() -> None:
    plan = _plan(_probe(rate=20 * MBPS, size=20 * GIB))
    assert (plan.cap.over, plan.rungs[0].reason) == ("bitrate", "hevc_over_cap:bitrate")


# --------------------------------------------------------- rules switched off
def test_a_cap_of_0_switches_its_bitrate_rule_off() -> None:
    caps = Caps(movie_1080=0, episode_2160=0)
    for probe, kind in [(_probe(rate=90 * MBPS), "movie"),
                        (_probe(width=3840, height=2160, rate=90 * MBPS), "episode")]:
        plan = _plan(probe, item_type=kind, caps=caps)
        assert (plan.rungs[0].mode, plan.cap.rate_bps, plan.cap.over) == ("copy", 0, None)


@pytest.mark.parametrize(("height", "fallback"), [(1080, 8), (2160, 14)])
def test_an_encode_whose_cap_is_off_takes_the_movies_default_maxrate(
    height: int, fallback: int,
) -> None:
    caps = Caps(movie_1080=0, movie_2160=0, episode_1080=0, episode_2160=0)
    for kind in ("movie", "episode"):
        [v0] = _plan(_probe("h264", WIDTH[height], height), "source:hevc", item_type=kind,
                     caps=caps).rungs
        assert v0.maxrate_bps == fallback * MBPS


def test_a_size_rule_of_0_is_off() -> None:
    plan = _plan(_probe(rate=2 * MBPS, size=100 * GIB), caps=Caps(movie_max_bytes=0))
    assert (plan.rungs[0].mode, plan.cap.over) == ("copy", None)


def test_no_caps_at_all_copy_as_before() -> None:
    plan = _plan(_probe(rate=90 * MBPS, size=100 * GIB),
                 caps=Caps(movie_1080=0, movie_max_bytes=0))
    assert plan.rungs[0].mode == "copy"


# --------------------------------------- every source-size HEVC encode is capped
@pytest.mark.parametrize(("encoders", "hevc"), HOSTS)
@pytest.mark.parametrize(("item_type", "bucket", "cap"), TABLE)
@pytest.mark.parametrize(("codec", "extra"), [
    ("h264", {}), ("h264", {"pix_fmt": "yuv420p10le", "profile": "High 10"}), ("av1", {}),
    ("vp9", {}), ("mpeg2video", {}),
    ("hevc", {"profile": "Rext", "pix_fmt": "yuv422p10le"}),  # not copyable
])
def test_every_source_size_hevc_encode_takes_the_cap_as_its_maxrate(
    encoders: Encoders, hevc: str, item_type: str, bucket: int, cap: int, codec: str,
    extra: dict,
) -> None:
    probe = _probe(codec, WIDTH[bucket], bucket, rate=1 * MBPS, **extra)
    [v0] = _plan(probe, "source:hevc", encoders, item_type=item_type).rungs
    assert (v0.mode, v0.encoder, v0.maxrate_bps) == ("encode", hevc, cap * MBPS)


@pytest.mark.parametrize(("item_type", "bucket", "cap"), TABLE)
def test_nvenc_maxrate_overrides_the_maxrate_of_every_kind(
    item_type: str, bucket: int, cap: int,
) -> None:
    # NVENC_MAXRATE_1080P=10 / 2160P=20: the maxrate of every source-size
    # HEVC encode of that bucket, whatever the kind — the capped copy's
    # and an H.264 source's alike.
    override = {1080: 10, 2160: 20}[bucket]
    over_cap = _probe(width=WIDTH[bucket], height=bucket, rate=cap * MBPS + 1)
    [v0] = _plan(over_cap, item_type=item_type, nvenc_caps_mbps=(10, 20)).rungs
    assert (v0.mode, v0.maxrate_bps) == ("encode", override * MBPS)
    [h264] = _plan(_probe("h264", WIDTH[bucket], bucket), "source:hevc", item_type=item_type,
                   nvenc_caps_mbps=(10, 20)).rungs
    assert h264.maxrate_bps == override * MBPS


def test_the_gate_stays_the_caps_with_an_override() -> None:
    # An episode at 7 Mbit/s: above its 6 Mbit/s cap, below the override's
    # 10. Encoded, because of the cap, at the override's maxrate.
    plan = _plan(_probe(rate=7 * MBPS), item_type="episode", nvenc_caps_mbps=(10, None))
    assert (plan.cap.over, plan.rungs[0].maxrate_bps) == ("bitrate", 10 * MBPS)
    # And a movie at 7 Mbit/s is within its 8: copied.
    assert _plan(_probe(rate=7 * MBPS), nvenc_caps_mbps=(10, None)).rungs[0].mode == "copy"


def test_a_maxrate_the_ladder_names_wins() -> None:
    [v0] = _plan(_probe(rate=20 * MBPS), "source:hevc:5M", item_type="episode").rungs
    assert (v0.mode, v0.maxrate_bps) == ("encode", 5 * MBPS)


def test_scaled_rungs_and_h264_encodes_keep_the_table() -> None:
    plan = _plan(_probe("mpeg2video", 1920, 1080), "source:h264,720p:hevc,480p",
                 item_type="episode")
    assert [(r.codec, r.height, r.maxrate_bps) for r in plan.rungs] == [
        ("h264", 1080, 6 * MBPS), ("hevc", 720, 2_500_000), ("h264", 480, 1_500_000)]


def test_a_ladder_with_the_capped_source_has_no_copy() -> None:
    plan = _plan(_probe(rate=20 * MBPS), "source,720p", item_type="movie")
    assert [(r.mode, r.codec, r.height) for r in plan.rungs] == [
        ("encode", "hevc", 1080), ("encode", "h264", 720)]
    assert plan.keyframes == "interval"


# ----------------------------------------------- what the cap does not touch
def test_h264_passed_through_on_a_cpu_is_not_capped() -> None:
    # The cap guards HEVC copies; the CPU rule's H.264 pass-through stays.
    [v0] = _plan(_probe("h264", rate=40 * MBPS), "", CPU_ENCODERS).rungs
    assert (v0.mode, v0.reason) == ("copy", "cpu_passthrough_h264")


def test_the_extras_default_ladder_has_no_hevc_copy_to_cap() -> None:
    plan = _plan(_probe(width=1280, height=720, rate=40 * MBPS), "720p:h264,480p:h264",
                 item_type="extra")
    assert [(r.mode, r.codec) for r in plan.rungs] == [("encode", "h264"), ("encode", "h264")]


def _dv(profile: int, compat: int, rate: int) -> dict:
    record = {"side_data_type": "DOVI configuration record", "dv_profile": profile,
              "dv_bl_signal_compatibility_id": compat}
    return _probe(width=3840, height=2160, rate=rate, pix_fmt="yuv420p10le",
                  profile="Main 10", color_transfer="smpte2084", side_data_list=[record])


def test_dolby_vision_8_1_above_its_cap_is_encoded_as_its_base_layer() -> None:
    [v0] = _plan(_dv(8, 1, 30 * MBPS), "", CPU_ENCODERS).rungs
    assert (v0.mode, v0.ten_bit, v0.maxrate_bps, v0.reason) == (
        "encode", True, 14 * MBPS, "hevc_over_cap:bitrate")
    [copy] = _plan(_dv(8, 1, 10 * MBPS)).rungs
    assert copy.mode == "copy"


def test_dolby_vision_5_is_refused_within_its_cap_too() -> None:
    with pytest.raises(SourceError):
        _plan(_dv(5, 0, 2 * MBPS))


# ------------------------------------------------------ the video bit rate
@pytest.mark.parametrize(("video", "probe", "expected"), [
    ({"bit_rate": "12000000"}, {}, (12_000_000, "stream")),
    # The stream's own rate wins over a tag and over the file's.
    ({"bit_rate": "12000000", "tags": {"BPS": "9"}}, {"size_bytes": 10**9, "duration_ms": 1000},
     (12_000_000, "stream")),
    ({"tags": {"BPS": "9000000"}}, {}, (9_000_000, "tag:BPS")),
    ({"tags": {"BPS-eng": "7000000"}}, {}, (7_000_000, "tag:BPS-eng")),
    ({"tags": {"bps": "5000000"}}, {}, (5_000_000, "tag:bps")),
    # A rate of 0 or N/A is no rate.
    ({"bit_rate": "N/A", "tags": {"BPS": "0", "BPS-eng": "6000000"}}, {},
     (6_000_000, "tag:BPS-eng")),
    # 10 MB in 4 s = 20 Mbit/s, less 448k (bit_rate) and 640k (BPS tag) of
    # audio; the AAC track that says neither counts 0.
    ({}, {"size_bytes": 10_000_000, "duration_ms": 4000,
          "audio": [{"bit_rate": "448000"}, {"tags": {"BPS": "640000"}}, {"codec_name": "aac"}]},
     (18_912_000, "size-minus-audio")),
    ({}, {"size_bytes": 10_000_000, "duration_ms": 0}, (None, "unknown")),
    ({}, {"size_bytes": None, "duration_ms": 4000}, (None, "unknown")),
    ({}, {"size_bytes": 1000, "duration_ms": 4000, "audio": [{"bit_rate": "448000"}]},
     (None, "unknown")),
])
def test_video_bit_rate_and_where_it_comes_from(video: dict, probe: dict, expected: tuple) -> None:
    assert video_bit_rate({"video": video, **probe}) == expected


@pytest.mark.parametrize(("probe", "rate_from", "over"), [
    (_probe(rate=9 * MBPS), "stream", "bitrate"),
    (_probe(tags={"BPS": "9000000"}), "tag:BPS", "bitrate"),
    (_probe(tags={"BPS-eng": "7000000"}), "tag:BPS-eng", None),
    # 5 MB in 4 s: 10 Mbit/s for the video alone, 7 once 3 Mbit/s of
    # audio are taken off.
    (_probe(size=5_000_000, duration_ms=4000), "size-minus-audio", "bitrate"),
    (_probe(size=5_000_000, duration_ms=4000, audio=[{"bit_rate": "3000000"}]),
     "size-minus-audio", None),
    (_probe(), "unknown", None),
])
def test_the_gate_reads_each_source_of_the_bit_rate(probe: dict, rate_from: str,
                                                    over: str | None) -> None:
    # A movie in the 1080 bucket: 8 Mbit/s.
    plan = _plan(probe)
    assert (plan.source.video_bit_rate_from, plan.cap.over) == (rate_from, over)
    assert plan.rungs[0].mode == ("encode" if over else "copy")


# ------------------------------------------------------------ the worker
class Steps:
    def __init__(self) -> None:
        self.steps: list[tuple[str, dict]] = []

    def upsert_step(self, _id: str, status: str, **kw: object) -> None:
        self.steps.append((status, kw))


def _item(path: Path, item_type: str) -> ClaimedItem:
    return ClaimedItem(id="0f1e2d3c-0000-4000-8000-000000000001", type=item_type,
                       title="clip", year=None, duration_ms=None, path=str(path))


def _worker_probe(**kw: object) -> dict:
    return {**_probe(**kw), "video_index": 0, "start_time": 0.0, "bit_rate": None}


def test_the_worker_reports_and_logs_a_copy_within_the_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    source = tmp_path / "hevc.mkv"
    source.write_bytes(b"\0")
    monkeypatch.setattr(worker, "ffprobe", lambda _p: _worker_probe(tags={"BPS": "4000000"}))
    steps = Steps()
    with capture_logs() as logs:
        ok = worker._process_one(_item(source, "movie"), steps, tmp_path / "inbox",
                                 EncodeSettings())
    assert ok and steps.steps[-1][0] == "not_applicable"
    assert steps.steps[-1][1]["details"].endswith(
        "reason=source_already_hevc:hevc rate=4.0Mbps(tag:BPS) cap=movie-1080:8Mbps over=-")
    [cap] = [e for e in logs if e["event"] == "transcoder.item.cap"]
    assert {k: cap[k] for k in ("kind", "bucket", "video_bit_rate", "bit_rate_from", "cap_bps",
                                "over")} == {
        "kind": "movie", "bucket": 1080, "video_bit_rate": 4_000_000, "bit_rate_from": "tag:BPS",
        "cap_bps": 8_000_000, "over": None}


def test_the_worker_encodes_an_episode_above_its_cap_at_its_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    source = tmp_path / "hevc.mkv"
    source.write_bytes(b"\0")
    monkeypatch.setattr(worker, "ffprobe", lambda _p: _worker_probe(
        size=5_000_000, duration_ms=4000, audio=[{"bit_rate": "448000"}]))
    commands: list[list[str]] = []

    def encode(args: list[str], outputs: list, *, log_label: str):
        commands.append(args)
        results = []
        for out in outputs:
            out.final.parent.mkdir(parents=True, exist_ok=True)
            out.final.write_bytes(b"\0" * 1000)
            results.append(RungResult(rung=out.rung, path=out.final, size_bytes=1000,
                                             width=out.rung.width, height=out.rung.height))
        return results, 0.1

    monkeypatch.setattr(worker, "run_encode", encode)
    steps = Steps()
    with capture_logs() as logs:
        ok = worker._process_one(_item(source, "episode"), steps, tmp_path / "inbox",
                                 EncodeSettings(encoders=CPU_ENCODERS))
    assert ok and steps.steps[-1][0] == "done"
    [args] = commands
    assert args[args.index("-maxrate") + 1:args.index("-maxrate") + 4] == ["6M", "-bufsize", "12M"]
    details = steps.steps[-1][1]["details"]
    assert "maxrate=6Mbps" in details
    assert details.endswith("rate=9.6Mbps(size-minus-audio) cap=episode-1080:6Mbps over=bitrate")
    [cap] = [e for e in logs if e["event"] == "transcoder.item.cap"]
    assert (cap["bit_rate_from"], cap["over"]) == ("size-minus-audio", "bitrate")


def test_the_worker_names_what_put_a_source_in_the_4k_bucket(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    source = tmp_path / "scope.mkv"
    source.write_bytes(b"\0")
    monkeypatch.setattr(worker, "ffprobe", lambda _p: _worker_probe(
        width=3840, height=1600, tags={"BPS": "12000000"}))
    steps = Steps()
    with capture_logs() as logs:
        ok = worker._process_one(_item(source, "movie"), steps, tmp_path / "inbox",
                                 EncodeSettings())
    assert ok and steps.steps[-1][0] == "not_applicable"
    assert steps.steps[-1][1]["details"].endswith(
        "rate=12.0Mbps(tag:BPS) cap=movie-2160(width):14Mbps over=-")
    [cap] = [e for e in logs if e["event"] == "transcoder.item.cap"]
    assert (cap["bucket"], cap["bucket_by"], cap["cap_bps"]) == (2160, "width", 14_000_000)
