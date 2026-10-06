"""The generated ffmpeg command lines + NVENC detection (no ffmpeg run)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from transcoder import ffmpeg as ff
from transcoder.decision import CPU_ENCODERS, NVENC_ENCODERS, parse_ladder, plan_renditions
from transcoder.ffmpeg import EncodeSettings, TranscodeError, build_encode_command

SRC = Path("/media/movie.mkv")
INBOX = Path("/packages/_inbox/item")
SUBS = [{"codec_name": "subrip"}, {"codec_name": "mov_text"}, {"codec_name": "hdmv_pgs_subtitle"}]


def _probe(codec: str, w: int, h: int, **extra: object) -> dict:
    video = {"codec_name": codec, "width": w, "height": h, "pix_fmt": "yuv420p",
             "avg_frame_rate": "24000/1001", **extra}
    return {"video": video, "video_index": 0, "audio": [], "subtitles": SUBS}


def _cmd(probe: dict, ladder: str, encoders=NVENC_ENCODERS, **kw):
    settings = EncodeSettings(ladder=tuple(parse_ladder(ladder)), encoders=encoders, **kw)
    plan = plan_renditions(probe, list(settings.ladder), encoders,
                           segment_seconds=settings.segment_seconds)
    return build_encode_command(SRC, INBOX, plan, settings,
                                probe_subtitles=SUBS, video_index=probe["video_index"])


def test_default_gpu_command_is_the_old_one_plus_keyframes() -> None:
    """The single-rendition GPU path must keep every flag the worker has
    always used; the only additions are -nostdin, the keyframe flags and
    the exact-timestamp encoder time base."""
    args, outputs = _cmd(_probe("h264", 1920, 1080), "")
    assert args == [
        "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "warning",
        "-fflags", "+genpts", "-avoid_negative_ts", "make_zero",
        "-i", str(SRC),
        "-map", "0:0",
        "-map", "0:a?", "-map", "0:s:0", "-map", "0:s:2",
        "-map", "-0:d", "-map", "-0:v:m:attached_pic",
        "-c:v", "hevc_nvenc", "-preset", "p5", "-profile:v", "main", "-pix_fmt", "yuv420p",
        "-rc:v", "vbr", "-cq", "23", "-maxrate", "8M", "-bufsize", "16M",
        "-b_ref_mode", "middle", "-spatial-aq", "1", "-rc-lookahead", "20",
        "-forced-idr", "1", "-force_key_frames", "expr:gte(t,n_forced*6)", "-g", "144",
        "-enc_time_base:v", "demux",
        "-c:a", "copy", "-c:s", "copy",
        "-f", "matroska", str(INBOX / "prepared.mkv.partial"),
    ]
    assert [o.final.name for o in outputs] == ["prepared.mkv"]


def test_ladder_command_decodes_once_and_splits() -> None:
    args, outputs = _cmd(_probe("h264", 1920, 1080), "source,720p,480p")
    graph = args[args.index("-filter_complex") + 1]
    assert graph == (
        "[0:0]split=3[bv0][bv1][bv2];"
        "[bv0]format=yuv420p[ov0];"
        "[bv1]scale=1280:720,setsar=1,format=yuv420p[ov1];"
        "[bv2]scale=854:480,setsar=1,format=yuv420p[ov2]"
    )
    assert args.count("-i") == 1
    assert [o.final.name for o in outputs] == ["prepared.mkv", "v1.mkv", "v2.mkv"]
    # v0 carries audio + subs; lower rungs are video only.
    v0 = args[: args.index(str(INBOX / "prepared.mkv.partial"))]
    assert "0:a?" in v0 and v0.count("-c:a") == 1
    v1 = args[args.index(str(INBOX / "prepared.mkv.partial")):args.index(
        str(INBOX / "v1.mkv.partial"))]
    assert "-an" in v1 and "-sn" in v1 and "0:a?" not in v1
    assert v1[v1.index("-c:v") + 1] == "h264_nvenc"
    # Identical keyframe flags and encoder time base on every rung.
    assert args.count("expr:gte(t,n_forced*6)") == 3
    assert args.count("-enc_time_base:v") == 3


def test_source_aligned_keyframes_when_top_rung_is_a_copy() -> None:
    args, outputs = _cmd(_probe("hevc", 1920, 1080), "source,720p", encoders=CPU_ENCODERS)
    assert [o.final.name for o in outputs] == ["v1.mkv"]
    assert "-force_key_frames" in args
    assert args[args.index("-force_key_frames") + 1] == "source"
    # x264 must not add scene-cut keyframes of its own.
    assert args[args.index("-sc_threshold") + 1] == "0"
    assert args[args.index("-g") + 1] == "1439"  # ceil(23.976 fps * 60 s)
    # No audio / subs: the packager takes them from the original source.
    assert "0:a?" not in args


def test_nvenc_source_aligned_disables_scenecut() -> None:
    args, _ = _cmd(_probe("hevc", 1920, 1080), "source,720p")
    assert args[args.index("-no-scenecut") + 1] == "1"


def test_hdr_rungs_tonemap_h264_and_keep_10bit_hevc() -> None:
    probe = _probe("hevc", 3840, 2160, pix_fmt="yuv420p10le", color_transfer="smpte2084")
    args, _ = _cmd(probe, "source,1080p:hevc,720p", encoders=CPU_ENCODERS)
    graph = args[args.index("-filter_complex") + 1]
    assert "format=yuv420p10le[ov1]" in graph
    assert "tonemap=tonemap=hable" in graph.split("[ov1];")[1]
    assert "main10" in args
    assert args[args.index("-color_trc") + 1] == "bt709"
    assert "hdr-opt=1" in args[args.index("-x265-params") + 1]


def test_cpu_encoders_args() -> None:
    args, _ = _cmd(_probe("mpeg2video", 1920, 1080), "source,720p", encoders=CPU_ENCODERS,
                   x265_preset="fast", x265_crf=22, x264_preset="slow", x264_crf=21)
    assert args[args.index("-preset") + 1] == "fast"
    assert args[args.index("-crf") + 1] == "22"
    second = args[args.index("libx264"):]
    assert second[second.index("-preset") + 1] == "slow"
    assert second[second.index("-crf") + 1] == "21"
    assert "-max_muxing_queue_size" in args


def test_segment_seconds_drives_interval_and_gop() -> None:
    args, _ = _cmd(_probe("h264", 1920, 1080, avg_frame_rate="25/1"), "", segment_seconds=4)
    assert args[args.index("-force_key_frames") + 1] == "expr:gte(t,n_forced*4)"
    assert args[args.index("-g") + 1] == "100"


def test_no_encoded_rung_is_an_error() -> None:
    with pytest.raises(TranscodeError):
        _cmd(_probe("hevc", 1920, 1080), "")


# ------------------------------------------------------------- HEVC only
def test_hevc_only_on_nvenc_is_the_default_command() -> None:
    probe = _probe("h264", 1920, 1080)
    assert _cmd(probe, "source:hevc") == _cmd(probe, "")


def test_hevc_only_on_a_cpu_is_one_x265_encode_of_the_source() -> None:
    # Browser-friendly H.264, which the default ladder passes through on a
    # CPU: one libx265 encode, the video mapped as it is (no filter graph),
    # every audio and subtitle track copied beside it.
    args, outputs = _cmd(_probe("h264", 1920, 1080), "source:hevc", encoders=CPU_ENCODERS)
    assert [o.final.name for o in outputs] == ["prepared.mkv"]
    assert "-filter_complex" not in args
    assert args[args.index("-map") + 1] == "0:0"
    assert args.count("-c:v") == 1 and args[args.index("-c:v") + 1] == "libx265"
    assert args[args.index("-profile:v") + 1] == "main"
    assert args[args.index("-pix_fmt") + 1] == "yuv420p"
    assert args[args.index("-force_key_frames") + 1] == "expr:gte(t,n_forced*6)"
    assert "scenecut=0" not in args[args.index("-x265-params") + 1]
    assert args[args.index("-c:a"):args.index("-c:a") + 4] == ["-c:a", "copy", "-c:s", "copy"]


@pytest.mark.parametrize(("encoders", "fmt", "pix_fmt"), [
    (NVENC_ENCODERS, "p010le", "p010le"), (CPU_ENCODERS, "yuv420p10le", "yuv420p10le")])
def test_hevc_only_keeps_an_hdr_source_10bit(encoders, fmt: str, pix_fmt: str) -> None:
    probe = _probe("av1", 3840, 2160, pix_fmt="yuv420p10le", color_transfer="smpte2084")
    args, _ = _cmd(probe, "source:hevc", encoders=encoders)
    assert args[args.index("-filter_complex") + 1] == f"[0:0]format={fmt}[ov0]"
    assert args[args.index("-profile:v") + 1] == "main10"
    assert args[args.index("-pix_fmt") + 1] == pix_fmt
    assert "-color_trc" not in args  # no tone-map: it stays HDR


# ---------------------------------------------------------- detection
class _Result:
    def __init__(self, rc: int, stderr: str = "") -> None:
        self.returncode, self.stderr, self.stdout = rc, stderr, ""


def test_detect_encoders_auto_falls_back_per_codec(monkeypatch) -> None:
    def fake_run(args, **_kw):
        enc = args[args.index("-c:v") + 1]
        return _Result(0 if enc == "h264_nvenc" else 255, "Cannot load libcuda.so.1")

    monkeypatch.setattr(subprocess, "run", fake_run)
    enc = ff.detect_encoders("auto")
    assert (enc.hevc, enc.h264) == ("libx265", "h264_nvenc")


def test_detect_encoders_no_gpu_is_cpu(monkeypatch) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Result(255, "No NVENC capable devices"))
    assert ff.detect_encoders("auto") == CPU_ENCODERS
    with pytest.raises(TranscodeError, match="ENCODER=nvenc"):
        ff.detect_encoders("nvenc")


def test_detect_encoders_cpu_skips_the_probe(monkeypatch) -> None:
    def boom(*_a, **_k):
        raise AssertionError("must not probe")

    monkeypatch.setattr(subprocess, "run", boom)
    assert ff.detect_encoders("cpu") == CPU_ENCODERS
    with pytest.raises(TranscodeError):
        ff.detect_encoders("quantum")
