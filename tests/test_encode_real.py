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

from transcoder.decision import CPU_ENCODERS, parse_ladder
from transcoder.ffmpeg import EncodeSettings
from transcoder.katalog import ClaimedItem
from transcoder.worker import _process_one


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
    ok = _process_one(item, client, tmp_path / "packages", cfg)  # type: ignore[arg-type]
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
