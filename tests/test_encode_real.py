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
import struct
import subprocess
from pathlib import Path

import pytest

from transcoder import extras, worker
from transcoder.config import DEFAULT_EXTRA_LADDER
from transcoder.decision import CPU_ENCODERS, parse_ladder
from transcoder.ffmpeg import EncodeSettings
from transcoder.katalog import ClaimedExtra, ClaimedItem, LibraryRecord
from transcoder.worker import _inbox_dir, _process_one


def _ffmpeg_has(kind: str, name: str) -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    out = subprocess.run(["ffmpeg", "-hide_banner", f"-{kind}"], capture_output=True,
                         text=True, stdin=subprocess.DEVNULL).stdout
    return any(line.split()[1:2] == [name] for line in out.splitlines() if line.strip())


def _encodes(encoder: str, pix_fmt: str) -> bool:
    """`encoder` is built in and takes `pix_fmt` (x264 / x265 built for
    10-bit, say)."""
    if not _ffmpeg_has("encoders", encoder):
        return False
    out = subprocess.run(["ffmpeg", "-hide_banner", "-h", f"encoder={encoder}"],
                         capture_output=True, text=True, stdin=subprocess.DEVNULL).stdout
    return any(line.strip().startswith("Supported pixel formats:") and pix_fmt in line.split()
               for line in out.splitlines())


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
    """The extras' worker protocol: one extra's record, and step writes.
    With `inbox`, the record is the v2 layout's and names that inbox."""

    def __init__(self, path: Path, inbox: Path | None = None) -> None:
        library = None if inbox is None else LibraryRecord(contract=1, inbox_dir=str(inbox))
        self.extra = ClaimedExtra(id=EXTRA_ID, parent_id=PARENT_ID, kind="trailer",
                                  title="Trailer", path=str(path), state="queued",
                                  library=library)
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


def _run_extra(tmp_path: Path, src: Path, *, ladder: str = DEFAULT_EXTRA_LADDER,
               inbox: Path | None = None) -> tuple[StubExtrasCatalog, StubProducer, Path]:
    """One catalog.extra.queued trigger through the extras handler, on the
    extras' default ladder (or `ladder`) and the CPU encoders. With
    `inbox`, the record is the v2 layout's, naming that inbox."""
    cfg = EncodeSettings(
        ladder=tuple(parse_ladder(ladder)), encoders=CPU_ENCODERS,
        x264_preset="ultrafast", x265_preset="ultrafast",
    )
    catalog, producer = StubExtrasCatalog(src, inbox), StubProducer()
    trigger = {"eventId": "9f2b", "extraId": EXTRA_ID, "parentId": PARENT_ID, "type": "extra",
               "kind": "trailer", "step": "transcode", "status": "queued", "source": "api"}
    extras._handle_extra(EXTRA_ID, trigger, catalog, producer, EXTRAS_TOPIC,  # type: ignore[arg-type]
                         tmp_path / "packages", cfg)
    return catalog, producer, inbox or tmp_path / "packages" / "_inbox" / f"extra-{EXTRA_ID}"


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


# ------------------------------------------------------------- HEVC only
# LADDER=source:hevc (EXTRA_LADDER alike): one HEVC rendition at the
# source's own size — the source copied when it is HEVC, else one encode,
# here libx265 (the CPU encoders), where the default ladder would pass a
# browser-friendly H.264 source through.
HEVC_ONLY = "source:hevc"
ITEMS_TOPIC = "stube.catalog.item.transcoded"
needs_x265 = pytest.mark.skipif(not _ffmpeg_has("encoders", "libx265"), reason="no libx265")


# Wide-gamut SDR: BT.2020 primaries and matrix with the BT.2020 SDR
# transfer, the tags a 10-bit SDR source keeps through its encode.
BT2020_SDR = "setparams=color_primaries=bt2020:color_trc=bt2020-10:colorspace=bt2020nc"
BT2020_SDR_TAGS = {"color_primaries": "bt2020", "color_transfer": "bt2020-10",
                   "color_space": "bt2020nc"}


def _colours(path: Path) -> dict[str, str | None]:
    video = _streams(path)[0]
    return {k: video.get(k) for k in BT2020_SDR_TAGS}


def _one_hevc_encode(inbox: Path, item_id: str, *, size: tuple[int, int] = (1280, 720),
                     pix_fmt: str = "yuv420p", profile: str = "Main") -> dict:
    """The handoff of a single HEVC encode, checked; its contract."""
    assert sorted(p.name for p in inbox.iterdir()) == ["prepared.mkv", "renditions.json"]
    prepared = _streams(inbox / "prepared.mkv")
    assert [(s["codec_type"], s["codec_name"]) for s in prepared] == [
        ("video", "hevc"), ("audio", "aac"), ("subtitle", "subrip")]
    assert (prepared[0]["width"], prepared[0]["height"], prepared[0]["pix_fmt"],
            prepared[0]["profile"]) == (*size, pix_fmt, profile)
    assert prepared[1]["channels"] == 6  # audio copied untouched
    contract = json.loads((inbox / "renditions.json").read_text())
    assert (contract["version"], contract["itemId"], contract["keyframes"]) == (
        1, item_id, "interval")
    assert [(v["id"], v["mode"], v["file"], v["codec"], v["encoder"], v["width"], v["height"],
             v["carries"]) for v in contract["video"]] == [
        ("v0", "encode", "prepared.mkv", "hevc", "libx265", *size,
         ["video", "audio", "subtitles"])]
    return contract


@needs_x265
def test_hevc_only_encodes_browser_friendly_h264_once(tmp_path: Path, h264_clip: Path) -> None:
    # The same clip the default ladder passes through on a CPU
    # (test_cpu_default_passes_h264_through).
    ok, client, inbox = _run(tmp_path, h264_clip, HEVC_ONLY)
    assert ok and [status for status, _ in client.steps] == ["in_progress", "done"]
    details = str(client.steps[-1][1]["details"])
    assert "src_codec=h264" in details and "ladder=v0:libx265:1280x720" in details
    assert "kf=interval/6s" in details
    contract = _one_hevc_encode(inbox, ITEM_ID)
    assert contract["source"]["codec"] == "h264"
    # An IDR every 6 s on the file's own timeline: 0 and 6.006 in 10 s.
    expected = _interval_keyframes(inbox / "prepared.mkv", 6)
    assert len(expected) == 2
    assert _keyframes(inbox / "prepared.mkv") == expected


@pytest.mark.skipif(not (_encodes("libx264", "yuv420p10le") and _encodes("libx265", "yuv420p10le")),
                    reason="needs a 10-bit libx264 + libx265")
def test_hevc_only_encodes_hi10p_h264_once_to_main10_sdr(tmp_path: Path) -> None:
    # High 10 H.264 decodes almost nowhere in hardware: never a copy. Its
    # 10 bits are kept (Main 10), and so is its BT.2020 SDR tagging.
    src = _make_clip(tmp_path / "hi10p.mkv", ["-c:v", "libx264", "-preset", "ultrafast"],
                     seconds=2, vf=f"format=yuv420p10le,{BT2020_SDR}")
    assert _streams(src)[0]["profile"] == "High 10"
    assert _colours(src) == BT2020_SDR_TAGS
    ok, client, inbox = _run(tmp_path, src, HEVC_ONLY)
    assert ok and client.steps[-1][0] == "done"
    _one_hevc_encode(inbox, ITEM_ID, pix_fmt="yuv420p10le", profile="Main 10")
    assert _colours(inbox / "prepared.mkv") == BT2020_SDR_TAGS


@needs_x265
@pytest.mark.parametrize("pix_fmt", ["yuv420p", "yuv420p10le"])
def test_hevc_only_copies_an_hevc_source(tmp_path: Path, pix_fmt: str) -> None:
    if not _encodes("libx265", pix_fmt):
        pytest.skip(f"libx265 without {pix_fmt}")
    src = _make_clip(tmp_path / "hevc.mkv",
                     ["-c:v", "libx265", "-preset", "ultrafast", "-x265-params", "log-level=error"],
                     seconds=2, vf=f"format={pix_fmt}")
    ok, client, inbox = _run(tmp_path, src, HEVC_ONLY)
    assert ok and [status for status, _ in client.steps] == ["in_progress", "not_applicable"]
    assert "reason=source_already_hevc:hevc" in str(client.steps[-1][1]["details"])
    # The packager packages the original's HEVC: no handoff.
    assert not inbox.exists() or not any(inbox.iterdir())


class StubV2Catalog:
    """An item's worker protocol on the v2 library layout: a record whose
    library block names the inbox, no step finished yet, step writes."""

    def __init__(self, src: Path, inbox: Path) -> None:
        self.item = ClaimedItem(id=ITEM_ID, type="movie", title="clip", year=None,
                                duration_ms=None, path=str(src),
                                library=LibraryRecord(contract=1, inbox_dir=str(inbox)))
        self.steps: list[tuple[str, dict]] = []

    def get_item(self, item_id: str) -> ClaimedItem | None:
        return self.item if item_id == ITEM_ID else None

    def get_steps(self, _item_id: str) -> dict[str, str]:
        return {}

    def upsert_step(self, _item_id: str, status: str, **kw: object) -> None:
        self.steps.append((status, kw))


def _tree(root: Path) -> list[str]:
    return [p.relative_to(root).as_posix() for p in sorted(root.rglob("*"))]


@needs_x265
def test_hevc_only_item_on_the_v2_layout_hands_off_into_its_records_inbox(
    tmp_path: Path, h264_clip: Path,
) -> None:
    work_inbox = tmp_path / "katalog" / ".work" / "inbox"
    catalog, producer = StubV2Catalog(h264_clip, work_inbox / ITEM_ID), StubProducer()
    cfg = EncodeSettings(ladder=tuple(parse_ladder(HEVC_ONLY)), encoders=CPU_ENCODERS,
                         x265_preset="ultrafast")
    worker._handle_item(ITEM_ID, "movie", catalog, producer, ITEMS_TOPIC,  # type: ignore[arg-type]
                        tmp_path / "packages", cfg)
    assert [status for status, _ in catalog.steps] == ["in_progress", "done"]
    # Only the record's inbox is written; nothing under the legacy root.
    assert not (tmp_path / "packages").exists()
    assert _tree(work_inbox) == [ITEM_ID, f"{ITEM_ID}/prepared.mkv",
                                 f"{ITEM_ID}/renditions.json"]
    _one_hevc_encode(work_inbox / ITEM_ID, ITEM_ID)
    [(topic, key, event)] = producer.produced
    assert (topic, key, event["itemId"], event["step"]) == (ITEMS_TOPIC, ITEM_ID, ITEM_ID,
                                                            "package")


@needs_x265
def test_hevc_only_extra_on_the_v2_layout_hands_off_into_its_records_inbox(
    tmp_path: Path, h264_clip: Path,
) -> None:
    work_inbox = tmp_path / "katalog" / ".work" / "inbox"
    catalog, producer, inbox = _run_extra(tmp_path, h264_clip, ladder=HEVC_ONLY,
                                          inbox=work_inbox / f"extra-{EXTRA_ID}")
    assert [status for status, _ in catalog.steps] == ["in_progress", "done"]
    assert "ladder=v0:libx265:1280x720" in str(catalog.steps[-1][1]["details"])
    assert not (tmp_path / "packages").exists()
    assert _tree(work_inbox) == [f"extra-{EXTRA_ID}", f"extra-{EXTRA_ID}/prepared.mkv",
                                 f"extra-{EXTRA_ID}/renditions.json"]
    # The unchanged contract, named after the extra.
    _one_hevc_encode(inbox, EXTRA_ID)
    _passed_on(producer)


# ----------------------------------- bit depth, the HEVC copy rule, Dolby Vision
X265 = ["-c:v", "libx265", "-preset", "ultrafast", "-x265-params", "log-level=error"]
needs_10bit_x265 = pytest.mark.skipif(not _encodes("libx265", "yuv420p10le"),
                                      reason="needs a 10-bit libx265")


@pytest.mark.parametrize(("name", "vcodec", "encoder"), [
    ("av1-10.mkv", ["-c:v", "libsvtav1", "-preset", "12"], "libsvtav1"),
    ("vp9-10.mkv", ["-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8",
                    "-b:v", "300k"], "libvpx-vp9"),
])
def test_a_10bit_sdr_av1_or_vp9_becomes_main10_with_its_colour_tags(
    tmp_path: Path, name: str, vcodec: list[str], encoder: str,
) -> None:
    if not (_encodes(encoder, "yuv420p10le") and _encodes("libx265", "yuv420p10le")):
        pytest.skip(f"needs a 10-bit {encoder} + libx265")
    src = _make_clip(tmp_path / name, vcodec, seconds=2, size="320x240",
                     vf=f"format=yuv420p10le,{BT2020_SDR}")
    assert _colours(src) == BT2020_SDR_TAGS
    ok, client, inbox = _run(tmp_path, src, HEVC_ONLY)
    assert ok and client.steps[-1][0] == "done"
    _one_hevc_encode(inbox, ITEM_ID, size=(320, 240), pix_fmt="yuv420p10le", profile="Main 10")
    assert _colours(inbox / "prepared.mkv") == BT2020_SDR_TAGS


@pytest.mark.parametrize(("pix_fmt", "out_fmt", "out_profile"), [
    ("yuv422p10le", "yuv420p10le", "Main 10"),
    ("yuv420p12le", "yuv420p10le", "Main 10"),
    ("yuv444p", "yuv420p", "Main"),
])
def test_hevc_that_is_not_main_or_main10_420_is_reencoded(
    tmp_path: Path, pix_fmt: str, out_fmt: str, out_profile: str,
) -> None:
    # The default ladder, which copies HEVC Main and Main 10: a Rext 4:2:2,
    # 12-bit or 4:4:4 one is encoded to 4:2:0 at its size instead, its
    # bits kept up to 10 and its colour tags with it.
    if not (_encodes("libx265", pix_fmt) and _encodes("libx265", "yuv420p10le")):
        pytest.skip(f"needs libx265 with {pix_fmt}")
    src = _make_clip(tmp_path / "rext.mkv", X265, seconds=2, size="320x240",
                     vf=f"format={pix_fmt},{BT2020_SDR}")
    assert (_streams(src)[0]["profile"], _streams(src)[0]["pix_fmt"]) == ("Rext", pix_fmt)
    ok, client, inbox = _run(tmp_path, src, "")
    assert ok and [status for status, _ in client.steps] == ["in_progress", "done"]
    _one_hevc_encode(inbox, ITEM_ID, size=(320, 240), pix_fmt=out_fmt, profile=out_profile)
    assert _colours(inbox / "prepared.mkv") == BT2020_SDR_TAGS


def _dovi_record(profile: int, compat: int, level: int = 6) -> bytes:
    """A Dolby Vision decoder configuration record (24 bytes: version 1.0,
    the RPU and the base layer present, no enhancement layer)."""
    flags = (profile << 9) | (level << 3) | 0b101
    return bytes([1, 0]) + struct.pack(">HI", flags, compat << 28) + bytes(16)


def _with_dolby_vision(src: Path, dst: Path, *, profile: int, compat: int,
                       entry: str | None = None) -> Path:
    """`src`, an MP4 whose moov follows its mdat (as ffmpeg writes one),
    with a Dolby Vision configuration record in its video sample entry
    (dvcC up to profile 7, dvvC above), and that entry renamed to `entry`
    (dvh1 for profile 5). The boxes on the way grow by the record; mdat
    does not move, so no chunk offset changes."""
    data = bytearray(src.read_bytes())

    def boxes(start: int, end: int):
        pos = start
        while pos + 8 <= end:
            size, kind = struct.unpack(">I4s", data[pos:pos + 8])
            assert size >= 8, "no 64-bit or open-ended boxes here"
            yield kind, pos, size
            pos += size

    def child(box: tuple[int, int], kind: bytes) -> tuple[int, int]:
        return next((p, s) for k, p, s in boxes(box[0] + 8, sum(box)) if k == kind)

    top = {k: (p, s) for k, p, s in boxes(0, len(data))}
    moov = top[b"moov"]
    assert top[b"mdat"][0] < moov[0], "the moov must follow the mdat"
    for kind, start, size in boxes(moov[0] + 8, sum(moov)):
        mdia = child((start, size), b"mdia") if kind == b"trak" else None
        if mdia and data[child(mdia, b"hdlr")[0] + 16:child(mdia, b"hdlr")[0] + 20] == b"vide":
            trak = (start, size)
            break
    else:
        raise AssertionError("no video track")
    minf = child(mdia, b"minf")
    stbl = child(minf, b"stbl")
    stsd = child(stbl, b"stsd")
    sample_entry = (stsd[0] + 16, struct.unpack(">I", data[stsd[0] + 16:stsd[0] + 20])[0])
    record = _dovi_record(profile, compat)
    box = struct.pack(">I4s", 8 + len(record), b"dvcC" if profile <= 7 else b"dvvC") + record
    data[sum(sample_entry):sum(sample_entry)] = box
    for start, size in (moov, trak, mdia, minf, stbl, stsd, sample_entry):
        struct.pack_into(">I", data, start, size + len(box))
    if entry:
        data[sample_entry[0] + 4:sample_entry[0] + 8] = entry.encode()
    dst.write_bytes(data)
    return dst


def _dv_clip(tmp_path: Path, profile: int, compat: int, entry: str | None = None) -> Path:
    """A 2 s HEVC Main 10 PQ MP4 carrying a Dolby Vision record that the
    real ffprobe reads back."""
    base = _make_clip(tmp_path / "hevc10.mp4", [*X265, "-tag:v", "hvc1"], seconds=2,
                      size="320x240", extra=["-sn"],
                      vf="format=yuv420p10le,setparams=color_primaries=bt2020"
                         ":color_trc=smpte2084:colorspace=bt2020nc")
    dv = _with_dolby_vision(base, tmp_path / f"dv{profile}.{compat}.mp4", profile=profile,
                            compat=compat, entry=entry)
    video = _streams(dv)[0]
    [record] = [sd for sd in video.get("side_data_list", [])
                if sd.get("side_data_type") == "DOVI configuration record"]
    assert (record["dv_profile"], record["dv_bl_signal_compatibility_id"]) == (profile, compat)
    assert video["codec_tag_string"] == (entry or "hvc1")
    return dv


@needs_10bit_x265
@pytest.mark.parametrize(("profile", "compat", "entry"), [(5, 0, "dvh1"), (7, 6, None)])
@pytest.mark.parametrize("ladder", ["", HEVC_ONLY])
def test_dolby_vision_5_and_7_fail_the_step_and_keep_the_original(
    tmp_path: Path, profile: int, compat: int, entry: str | None, ladder: str,
) -> None:
    src = _dv_clip(tmp_path, profile, compat, entry)
    before = src.read_bytes()
    ok, client, inbox = _run(tmp_path, src, ladder)
    assert not ok
    assert [status for status, _ in client.steps] == ["in_progress", "failed"]
    assert client.steps[-1][1]["error"] == (
        f"Dolby Vision profile {profile} needs a tone-mapping encode; kept the original")
    assert not inbox.exists()
    assert src.read_bytes() == before


@needs_10bit_x265
def test_dolby_vision_8_1_is_copied(tmp_path: Path) -> None:
    src = _dv_clip(tmp_path, 8, 1)
    ok, client, inbox = _run(tmp_path, src, HEVC_ONLY)
    assert ok and [status for status, _ in client.steps] == ["in_progress", "not_applicable"]
    assert "reason=source_already_hevc:hevc" in str(client.steps[-1][1]["details"])
    assert not inbox.exists() or not any(inbox.iterdir())


@needs_10bit_x265
def test_dolby_vision_5_on_the_v2_layout_writes_nothing_and_sends_no_event(
    tmp_path: Path,
) -> None:
    src = _dv_clip(tmp_path, 5, 0, "dvh1")
    root = tmp_path / "katalog"
    catalog, producer = StubV2Catalog(src, root / ".work" / "inbox" / ITEM_ID), StubProducer()
    cfg = EncodeSettings(ladder=tuple(parse_ladder(HEVC_ONLY)), encoders=CPU_ENCODERS,
                         x265_preset="ultrafast")
    worker._handle_item(ITEM_ID, "movie", catalog, producer, ITEMS_TOPIC,  # type: ignore[arg-type]
                        tmp_path / "packages", cfg)
    assert [status for status, _ in catalog.steps] == ["in_progress", "failed"]
    assert producer.produced == []
    assert not root.exists() and not (tmp_path / "packages").exists()


# ------------------------------------------------------------ captions
def _with_captions(tmp_path: Path, *, seconds: int = 2) -> Path:
    """An H.264 clip whose every frame carries an EIA-608 caption in its
    video (an A53 "GA94" cc_data SEI before each slice, as broadcast and
    disc sources have them), in a Matroska file."""
    plain = tmp_path / "plain.h264"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", f"testsrc2=size=640x360:rate=25:duration={seconds}", "-c:v", "libx264",
         "-preset", "ultrafast", "-x264-params", "slices=1", "-f", "h264", str(plain)],
        check=True, stdin=subprocess.DEVNULL)
    # cc_data: process_cc_data_flag, two field-1 byte pairs (odd parity).
    cc = bytes([0x40 | 2, 0xFF, 0xFC, 0x94, 0x2C, 0xFC, 0xC8, 0x49, 0xFF])
    payload = bytes([0xB5, 0x00, 0x31]) + b"GA94" + bytes([0x03]) + cc
    sei = b"\x00\x00\x00\x01" + bytes([0x06, 0x04, len(payload)]) + payload + b"\x80"
    data, out, pos = plain.read_bytes(), bytearray(), 0
    for m in re.finditer(rb"\x00\x00\x01(.)", data, re.DOTALL):
        start = m.start() - 1 if m.start() and data[m.start() - 1] == 0 else m.start()
        if m.group(1)[0] & 0x1F in (1, 5):  # a slice: one per picture
            out += data[pos:start] + sei
            pos = start
    out += data[pos:]
    raw = tmp_path / "captioned.h264"
    raw.write_bytes(bytes(out))
    clip = tmp_path / "captioned.mkv"
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-framerate", "25", "-i", str(raw), "-c", "copy", str(clip)],
                   check=True, stdin=subprocess.DEVNULL)
    return clip


def _captioned_frames(path: Path) -> tuple[int, int]:
    """(video frames, those with A53 closed captions)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames", "-show_entries",
         "frame=pts:frame_side_data=side_data_type", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    frames = json.loads(out)["frames"]
    with_cc = [f for f in frames if any("A53" in sd.get("side_data_type", "")
                                        for sd in f.get("side_data_list", []))]
    return len(frames), len(with_cc)


@needs_x265
def test_every_encode_keeps_the_closed_captions_in_the_video(tmp_path: Path) -> None:
    # HEVC only plus a 240p H.264 rung, on the CPU: libx265 (whose -a53cc
    # is off unless named) and libx264, through the split and the scale.
    src = _with_captions(tmp_path)
    frames, with_cc = _captioned_frames(src)
    assert frames == with_cc == 50
    ok, client, inbox = _run(tmp_path, src, "source:hevc,240p")
    assert ok and client.steps[-1][0] == "done"
    for rung, codec in (("prepared.mkv", "hevc"), ("v1.mkv", "h264")):
        assert _streams(inbox / rung)[0]["codec_name"] == codec
        assert _captioned_frames(inbox / rung) == (50, 50), rung
