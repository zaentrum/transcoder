"""Real ffmpeg runs on generated lavfi clips (skipped without ffmpeg).

Drives the worker's `_process_one` end to end with the CPU encoders and
a stub katalog client, then checks the handoff on disk: which files
exist, their codecs and sizes, the keyframe positions, and
renditions.json. Fast presets keep the whole module to a few seconds.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from transcoder import extras
from transcoder.config import DEFAULT_EXTRA_LADDER
from transcoder.decision import CPU_ENCODERS, parse_ladder
from transcoder.ffmpeg import EncodeSettings
from transcoder.katalog import ClaimedExtra, ClaimedItem
from transcoder.worker import _inbox_dir, _process_one


def _ffmpeg_has(kind: str, name: str) -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    out = subprocess.run(["ffmpeg", "-hide_banner", f"-{kind}"], capture_output=True,
                         text=True, stdin=subprocess.DEVNULL).stdout
    return any(line.split()[1:2] == [name] for line in out.splitlines() if line.strip())


def _ffmpeg_major() -> int:
    """'ffmpeg version n7.1.5-12-g…' / '8.1.2' / '6.1.1-3ubuntu5' -> 7 / 8 / 6."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        return 0
    first = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True,
                           stdin=subprocess.DEVNULL).stdout.split("\n", 1)[0]
    m = re.search(r"version n?(\d+)\.", first)
    return int(m.group(1)) if m else 99  # git builds ("N-12345-g…") are new


# The image pins ffmpeg 7.1; `-enc_time_base demux` and 7.x start-time
# handling are what these runs exercise, so older distro builds skip.
pytestmark = pytest.mark.skipif(
    _ffmpeg_major() < 7, reason="needs ffmpeg/ffprobe >= 7 (the image pins 7.1)",
)

ITEM_ID = "0f1e2d3c-0000-4000-8000-000000000001"


class StubKatalog:
    def __init__(self) -> None:
        self.steps: list[tuple[str, dict]] = []

    def upsert_step(self, _item_id: str, status: str, **kw: object) -> None:
        self.steps.append((status, kw))


def _make_clip(path: Path, vcodec: list[str], *, seconds: int = 10, size: str = "1280x720",
               vf: str | None = None, extra: list[str] | None = None) -> Path:
    """10 s test clip: moving test pattern + 5.1 tone + one SRT track."""
    srt = path.with_suffix(".srt")
    srt.write_text("1\n00:00:01,000 --> 00:00:03,000\nHello\n")
    args = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=24000/1001:duration={seconds}",
        "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={seconds}",
        "-i", str(srt),
        "-map", "0:v", "-map", "1:a", "-map", "2",
        *(["-vf", vf] if vf else []),
        *vcodec,
        "-c:a", "aac", "-ac", "6", "-c:s", "srt",
        "-metadata:s:a:0", "language=eng", "-metadata:s:s:0", "language=eng",
        *(extra or []),
        str(path),
    ]
    subprocess.run(args, check=True, stdin=subprocess.DEVNULL)
    return path


def _packets(path: Path) -> list[tuple[float, bool]]:
    """(pts, is_keyframe) of every video packet, in presentation order."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "packet=pts_time,flags", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    rows = [line.split(",")[:2] for line in out.splitlines() if line]
    return sorted((round(float(t), 3), "K" in flags) for t, flags in rows)


def _keyframes(path: Path) -> list[float]:
    return [t for t, key in _packets(path) if key]


def _interval_keyframes(path: Path, seconds: int) -> list[float]:
    """Where `expr:gte(t,n_forced*S)` must put the IDRs: the first frame
    at/after every multiple of S on the file's own (shared) timeline."""
    pts = [t for t, _ in _packets(path)]
    expected, n = [], 0
    for t in pts:
        if t + 1e-6 >= n * seconds:
            expected.append(t)
            while n * seconds <= t + 1e-6:
                n += 1
    return expected


def _streams(path: Path) -> list[dict]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)["streams"]


def _run(tmp_path: Path, src: Path, ladder: str, **settings) -> tuple[bool, StubKatalog, Path]:
    cfg = EncodeSettings(
        ladder=tuple(parse_ladder(ladder)), encoders=CPU_ENCODERS,
        x264_preset="ultrafast", x265_preset="ultrafast", **settings,
    )
    client = StubKatalog()
    item = ClaimedItem(id=ITEM_ID, type="movie", title="clip", year=None,
                       duration_ms=None, path=str(src))
    ok = _process_one(item, client, _inbox_dir(tmp_path / "packages", ITEM_ID), cfg)
    return ok, client, tmp_path / "packages" / "_inbox" / ITEM_ID


@pytest.fixture(scope="module")
def h264_clip(tmp_path_factory) -> Path:
    # GOP of 100 frames (4.171 s): deliberately NOT a multiple of 6 s.
    d = tmp_path_factory.mktemp("clips")
    return _make_clip(d / "h264.mkv", ["-c:v", "libx264", "-preset", "ultrafast", "-g", "100"])


def test_cpu_default_passes_h264_through(tmp_path: Path, h264_clip: Path) -> None:
    ok, client, inbox = _run(tmp_path, h264_clip, "")
    assert ok
    status, kw = client.steps[-1]
    assert status == "not_applicable"
    assert "reason=cpu_passthrough_h264" in str(kw["details"])
    assert not inbox.exists() or not any(inbox.iterdir())


def test_cpu_ladder_aligns_lower_rungs_to_source_keyframes(tmp_path: Path, h264_clip: Path) -> None:
    ok, client, inbox = _run(tmp_path, h264_clip, "source,480p,360p")
    assert ok and client.steps[-1][0] == "done"
    assert sorted(p.name for p in inbox.iterdir()) == ["renditions.json", "v1.mkv", "v2.mkv"]

    contract = json.loads((inbox / "renditions.json").read_text())
    assert contract["version"] == 1
    assert contract["keyframes"] == "source"
    assert contract["segmentSeconds"] == 6
    # The source as ffprobe reads it: the packager forwards this to the
    # catalog as the title's source asset.
    fmt = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration,bit_rate", "-of", "json",
         str(h264_clip)], capture_output=True, text=True, check=True).stdout)["format"]
    assert contract["source"] == {
        "codec": "h264", "width": 1280, "height": 720, "frameRate": "24000/1001", "hdr": False,
        "durationMs": int(float(fmt["duration"]) * 1000), "bitRate": int(fmt["bit_rate"]),
    }
    assert [(v["id"], v["mode"], v["file"], v["width"], v["height"]) for v in contract["video"]] \
        == [("v0", "copy", None, 1280, 720), ("v1", "encode", "v1.mkv", 854, 480),
            ("v2", "encode", "v2.mkv", 640, 360)]
    assert contract["video"][0]["carries"] == ["video", "audio", "subtitles"]

    # Same frames as the source's keyframes, on the shared timeline (the
    # source shifted by timestampOffset — 0 or +0.021 s depending on how
    # the ffmpeg version reads the AAC priming of this clip).
    offset = contract["timestampOffset"]
    source_kf = [round(t + offset, 3) for t in _keyframes(h264_clip)]
    assert len(source_kf) == 3  # GOP 100 @ 23.976: 0, 4.171, 8.342
    assert contract["video"][0]["videoStart"] == pytest.approx(source_kf[0], abs=0.002)
    for rung in ("v1.mkv", "v2.mkv"):
        assert _keyframes(inbox / rung) == pytest.approx(source_kf, abs=0.002)
        kinds = [s["codec_type"] for s in _streams(inbox / rung)]
        assert kinds == ["video"]
    assert [v["videoStart"] for v in contract["video"]] == pytest.approx(
        [source_kf[0]] * 3, abs=0.002)


@pytest.mark.skipif(not _ffmpeg_has("encoders", "libx265"), reason="no libx265")
def test_cpu_exotic_source_encodes_hevc_with_fixed_interval(tmp_path: Path) -> None:
    src = _make_clip(tmp_path / "mpeg4.mkv", ["-c:v", "mpeg4", "-q:v", "4", "-g", "100"])
    ok, client, inbox = _run(tmp_path, src, "source,360p", segment_seconds=2)
    assert ok and client.steps[-1][0] == "done"
    details = str(client.steps[-1][1]["details"])
    assert "profile=x265-720p" in details and "kf=interval/2s" in details

    prepared = _streams(inbox / "prepared.mkv")
    assert [(s["codec_type"], s["codec_name"]) for s in prepared] == [
        ("video", "hevc"), ("audio", "aac"), ("subtitle", "subrip"),
    ]
    assert prepared[1]["channels"] == 6  # audio copied untouched

    expected = _interval_keyframes(inbox / "prepared.mkv", 2)
    assert len(expected) == 5  # 10 s clip, IDR every 2 s
    assert _keyframes(inbox / "prepared.mkv") == expected
    assert _keyframes(inbox / "v1.mkv") == expected
    contract = json.loads((inbox / "renditions.json").read_text())
    assert contract["keyframes"] == "interval"
    assert [v["encoder"] for v in contract["video"]] == ["libx265", "libx264"]
    # Audio was shifted with the video, not left behind: the copied track
    # starts where the source's earliest stream started on the new timeline.
    audio_start = float(prepared[1].get("start_time") or 0.0)
    assert audio_start == pytest.approx(0.0, abs=0.002)


@pytest.mark.skipif(
    not (_ffmpeg_has("filters", "zscale") and _ffmpeg_has("encoders", "libx265")),
    reason="needs zscale (libzimg) + libx265",
)
def test_hdr_source_gets_a_tonemapped_sdr_h264_rung(tmp_path: Path) -> None:
    src = _make_clip(
        tmp_path / "hdr.mkv",
        ["-c:v", "libx265", "-preset", "ultrafast",
         "-x265-params", "keyint=48:log-level=error:colorprim=bt2020:transfer=smpte2084"
                         ":colormatrix=bt2020nc"],
        seconds=4, vf="format=yuv420p10le",
        extra=["-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc"],
    )
    ok, _client, inbox = _run(tmp_path, src, "source,360p")
    assert ok
    [video] = [s for s in _streams(inbox / "v1.mkv") if s["codec_type"] == "video"]
    assert (video["codec_name"], video["pix_fmt"]) == ("h264", "yuv420p")
    assert video.get("color_transfer") == "bt709"
    contract = json.loads((inbox / "renditions.json").read_text())
    assert [v["hdr"] for v in contract["video"]] == [True, False]


def test_failed_encode_leaves_no_handoff(tmp_path: Path) -> None:
    bogus = tmp_path / "not-a-video.mkv"
    bogus.write_bytes(b"\x1a\x45\xdf\xa3 definitely not matroska")
    ok, client, inbox = _run(tmp_path, bogus, "source,360p")
    assert not ok
    assert client.steps[-1][0] == "failed"
    assert not inbox.exists()


# --------------------------------------------------------------- extras
EXTRA_ID = "1b5c2a8e-0000-4000-8000-0000000000e1"
PARENT_ID = "ea886f9b-0d06-4f0f-babb-d2a1162f9b01"
EXTRAS_TOPIC = "stube.catalog.extra.transcoded"


class StubExtrasCatalog:
    """The extras' worker protocol: one extra's record, and step writes."""

    def __init__(self, path: Path) -> None:
        self.extra = ClaimedExtra(id=EXTRA_ID, parent_id=PARENT_ID, kind="trailer",
                                  title="Trailer", path=str(path), state="queued")
        self.steps: list[tuple[str, dict]] = []

    def get_extra(self, extra_id: str) -> ClaimedExtra | None:
        return self.extra if extra_id == EXTRA_ID else None

    def upsert_extra_step(self, _extra_id: str, status: str, **kw: object) -> None:
        self.steps.append((status, kw))


class StubProducer:
    def __init__(self) -> None:
        self.produced: list[tuple[str, str, dict]] = []

    def produce(self, topic: str, key: bytes, value: bytes) -> None:
        self.produced.append((topic, key.decode(), json.loads(value)))

    def flush(self, *_args: object) -> int:
        return 0


def _run_extra(tmp_path: Path, src: Path) -> tuple[StubExtrasCatalog, StubProducer, Path]:
    """One catalog.extra.queued trigger through the extras handler, on the
    extras' default ladder and the CPU encoders."""
    cfg = EncodeSettings(
        ladder=tuple(parse_ladder(DEFAULT_EXTRA_LADDER)), encoders=CPU_ENCODERS,
        x264_preset="ultrafast", x265_preset="ultrafast",
    )
    catalog, producer = StubExtrasCatalog(src), StubProducer()
    trigger = {"eventId": "9f2b", "extraId": EXTRA_ID, "parentId": PARENT_ID, "type": "extra",
               "kind": "trailer", "step": "transcode", "status": "queued", "source": "api"}
    extras._handle_extra(EXTRA_ID, trigger, catalog, producer, EXTRAS_TOPIC,  # type: ignore[arg-type]
                         tmp_path / "packages", cfg)
    return catalog, producer, tmp_path / "packages" / "_inbox" / f"extra-{EXTRA_ID}"


def _passed_on(producer: StubProducer) -> None:
    [(topic, key, event)] = producer.produced
    assert (topic, key) == (EXTRAS_TOPIC, EXTRA_ID)
    assert "itemId" not in event
    assert {k: event[k] for k in ("extraId", "parentId", "type", "kind", "step", "status",
                                  "source")} == {
        "extraId": EXTRA_ID, "parentId": PARENT_ID, "type": "extra", "kind": "trailer",
        "step": "package", "status": "queued", "source": "transcoder"}


def test_extra_720p_h264_is_a_copy_and_a_480p_encode(tmp_path: Path, h264_clip: Path) -> None:
    catalog, producer, inbox = _run_extra(tmp_path, h264_clip)
    assert [status for status, _ in catalog.steps] == ["in_progress", "done"]
    assert "ladder=v0:copy:1280x720,v1:libx264:854x480" in str(catalog.steps[-1][1]["details"])
    # The extra's inbox only, never one named like an item's.
    assert [p.name for p in (tmp_path / "packages" / "_inbox").iterdir()] == [f"extra-{EXTRA_ID}"]
    assert sorted(p.name for p in inbox.iterdir()) == ["renditions.json", "v1.mkv"]

    # The unchanged contract, named after the extra.
    contract = json.loads((inbox / "renditions.json").read_text())
    assert (contract["version"], contract["itemId"], contract["keyframes"]) == (
        1, EXTRA_ID, "source")
    assert [(v["id"], v["mode"], v["file"], v["codec"], v["encoder"], v["width"], v["height"])
            for v in contract["video"]] == [
        ("v0", "copy", None, "h264", "copy", 1280, 720),
        ("v1", "encode", "v1.mkv", "h264", "libx264", 854, 480),
    ]
    # The 480p rung cuts where the copied source does.
    offset = contract["timestampOffset"]
    source_kf = [round(t + offset, 3) for t in _keyframes(h264_clip)]
    assert _keyframes(inbox / "v1.mkv") == pytest.approx(source_kf, abs=0.002)
    assert [s["codec_type"] for s in _streams(inbox / "v1.mkv")] == ["video"]
    _passed_on(producer)


@pytest.mark.skipif(not _ffmpeg_has("encoders", "libvpx-vp9"), reason="no libvpx-vp9")
def test_extra_480p_vp9_is_one_h264_encode_with_its_tracks(tmp_path: Path) -> None:
    src = _make_clip(tmp_path / "vp9.mkv", ["-c:v", "libvpx-vp9", "-deadline", "realtime",
                                            "-cpu-used", "8", "-b:v", "400k"], size="854x480")
    catalog, producer, inbox = _run_extra(tmp_path, src)
    assert [status for status, _ in catalog.steps] == ["in_progress", "done"]
    assert sorted(p.name for p in inbox.iterdir()) == ["prepared.mkv", "renditions.json"]

    prepared = _streams(inbox / "prepared.mkv")
    assert [(s["codec_type"], s["codec_name"]) for s in prepared] == [
        ("video", "h264"), ("audio", "aac"), ("subtitle", "subrip")]
    assert (prepared[0]["width"], prepared[0]["height"], prepared[0]["pix_fmt"]) == (
        854, 480, "yuv420p")
    contract = json.loads((inbox / "renditions.json").read_text())
    assert contract["source"]["codec"] == "vp9"
    assert contract["keyframes"] == "interval"
    assert [(v["id"], v["mode"], v["file"], v["width"], v["height"])
            for v in contract["video"]] == [("v0", "encode", "prepared.mkv", 854, 480)]
    expected = _interval_keyframes(inbox / "prepared.mkv", 6)
    assert len(expected) == 2  # 10 s clip, an IDR every 6 s
    assert _keyframes(inbox / "prepared.mkv") == expected
    _passed_on(producer)


def test_extra_small_h264_needs_no_encode(tmp_path: Path) -> None:
    # 640x360 H.264 already fits both rungs: the packager packages the
    # original, so the inbox stays empty and the step is not_applicable.
    src = _make_clip(tmp_path / "small.mp4", ["-c:v", "libx264", "-preset", "ultrafast"],
                     seconds=2, size="640x360", extra=["-sn"])
    catalog, producer, inbox = _run_extra(tmp_path, src)
    assert [status for status, _ in catalog.steps] == ["in_progress", "not_applicable"]
    assert "reason=source_already_h264" in str(catalog.steps[-1][1]["details"])
    assert not inbox.exists() or not any(inbox.iterdir())
    _passed_on(producer)
