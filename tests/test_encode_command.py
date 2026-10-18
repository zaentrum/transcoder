"""The generated ffmpeg command lines, NVENC detection and the Vulkan
preflight, and the checks around a Dolby Vision 5 encode (no ffmpeg run)."""

from __future__ import annotations

import ctypes
import json
import subprocess
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from transcoder import ffmpeg as ff
from transcoder.decision import (
    CPU_ENCODERS,
    NVENC_ENCODERS,
    VulkanDevice,
    parse_ladder,
    plan_renditions,
)
from transcoder.ffmpeg import EncodeSettings, TranscodeError, _subtitle_maps, build_encode_command

SRC = Path("/media/movie.mkv")
INBOX = Path("/packages/_inbox/item")
SUBS = [{"codec_name": "subrip"}, {"codec_name": "mov_text"}, {"codec_name": "hdmv_pgs_subtitle"}]
VULKAN = VulkanDevice(index="0", name="NVIDIA GeForce RTX 3090 (discrete)", loader="1.3.204",
                      error="")


def _probe(codec: str, w: int, h: int, **extra: object) -> dict:
    video = {"codec_name": codec, "width": w, "height": h, "pix_fmt": "yuv420p",
             "avg_frame_rate": "24000/1001", **extra}
    return {"video": video, "video_index": 0, "audio": [], "subtitles": SUBS}


def _cmd(probe: dict, ladder: str, encoders=NVENC_ENCODERS, **kw):
    settings = EncodeSettings(ladder=tuple(parse_ladder(ladder)), encoders=encoders, **kw)
    plan = plan_renditions(probe, list(settings.ladder), encoders,
                           segment_seconds=settings.segment_seconds, vulkan=settings.vulkan)
    return build_encode_command(SRC, INBOX, plan, settings,
                                probe_subtitles=SUBS, video_index=probe["video_index"])


def test_default_gpu_command_is_the_old_one_plus_keyframes() -> None:
    """The single-rendition GPU path must keep every flag the worker has
    always used; the only additions are -nostdin, the closed captions
    (-a53cc), the keyframe flags and the exact-timestamp encoder time
    base. The maxrate is a 1080 movie's cap."""
    args, outputs = _cmd(_probe("h264", 1920, 1080), "")
    assert args == [
        "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "warning",
        "-fflags", "+genpts", "-avoid_negative_ts", "make_zero",
        "-i", str(SRC),
        "-map", "0:0",
        "-map", "0:a?", "-map", "0:s:0", "-map", "0:s:1", "-map", "0:s:2",
        "-map", "-0:d", "-map", "-0:v:m:attached_pic",
        "-c:v", "hevc_nvenc", "-preset", "p5", "-profile:v", "main", "-pix_fmt", "yuv420p",
        "-rc:v", "vbr", "-cq", "23", "-maxrate", "8M", "-bufsize", "16M",
        "-b_ref_mode", "middle", "-spatial-aq", "1", "-rc-lookahead", "20",
        "-a53cc", "1",
        "-forced-idr", "1", "-force_key_frames", "expr:gte(t,n_forced*6)", "-g", "144",
        "-enc_time_base:v", "demux",
        "-c:a", "copy", "-c:s", "copy", "-c:s:1", "srt",
        "-f", "matroska", str(INBOX / "prepared.mkv.partial"),
    ]
    assert [o.final.name for o in outputs] == ["prepared.mkv"]


def test_subtitles_matroska_cannot_carry() -> None:
    """MP4's text captions are written as SubRip, each named by its place
    among the subtitle streams that are kept; a codec Matroska can neither
    copy nor take as text is left out, and the ones after it move up."""
    subs = [{"codec_name": c} for c in ("mov_text", "eia_608", "hdmv_pgs_subtitle", "mov_text")]
    maps, codecs, dropped = _subtitle_maps(subs)
    assert maps == ["-map", "0:s:0", "-map", "0:s:2", "-map", "0:s:3"]
    assert codecs == ["-c:s", "copy", "-c:s:0", "srt", "-c:s:2", "srt"]
    assert dropped == ["eia_608"]
    assert _subtitle_maps([]) == ([], ["-c:s", "copy"], [])
    assert _subtitle_maps(None) == (["-map", "0:s?"], ["-c:s", "copy"], [])


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


# ------------------------------------------------------------ captions
def _per_output(args: list[str]) -> list[list[str]]:
    """The command's options split by output (each ends with its file)."""
    chunks, start = [], args.index("-i") + 2
    for i, arg in enumerate(args):
        if arg.endswith(".partial"):
            chunks.append(args[start:i + 1])
            start = i + 1
    return chunks


@pytest.mark.parametrize("encoders", [NVENC_ENCODERS, CPU_ENCODERS,
                                      ff.Encoders(hevc="libx265", h264="h264_nvenc")])
@pytest.mark.parametrize(("probe", "ladder"), [
    (_probe("mpeg2video", 1920, 1080), ""),                              # one direct encode
    (_probe("h264", 1920, 1080), "source:hevc,720p,480p:hevc"),          # split, scaled
    (_probe("hevc", 1920, 1080, profile="Main"), "source,720p"),         # a copy beside
    (_probe("hevc", 3840, 2160, pix_fmt="yuv420p10le", profile="Main 10",
            color_transfer="smpte2084"), "source,1080p:hevc,720p"),      # tone-mapped H.264
])
def test_every_encode_keeps_the_closed_captions(encoders, probe: dict, ladder: str) -> None:
    # EIA/CEA-608/708 in the video's SEI: -a53cc 1 on every encoded rung,
    # whichever encoder it takes (libx265 has it off by default).
    args, outputs = _cmd(probe, ladder, encoders=encoders)
    chunks = _per_output(args)
    assert len(chunks) == len(outputs) > 0
    for chunk in chunks:
        assert chunk[chunk.index("-a53cc") + 1] == "1", chunk[chunk.index("-c:v") + 1]


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


@pytest.mark.parametrize(("encoders", "fmt"), [(NVENC_ENCODERS, "p010le"),
                                               (CPU_ENCODERS, "yuv420p10le")])
@pytest.mark.parametrize(("codec", "extra"), [
    ("h264", {"pix_fmt": "yuv420p10le", "profile": "High 10"}),
    ("vp9", {"pix_fmt": "yuv420p10le", "profile": "Profile 2", "color_space": "bt2020nc",
             "color_transfer": "bt2020-10", "color_primaries": "bt2020"}),
    ("av1", {"pix_fmt": "yuv420p12le"}),
])
def test_a_10bit_sdr_source_is_main10_sdr(encoders, fmt: str, codec: str, extra: dict) -> None:
    # Main 10 into a 10-bit 4:2:0 format, no HDR options, and no colour
    # flags at all: the frames' own tags (BT.2020 SDR, say) go through.
    args, _ = _cmd(_probe(codec, 1920, 1080, **extra), "source:hevc", encoders=encoders)
    assert args[args.index("-filter_complex") + 1] == f"[0:0]format={fmt}[ov0]"
    assert args[args.index("-profile:v") + 1] == "main10"
    assert args[args.index("-pix_fmt") + 1] == fmt
    assert not {"-color_primaries", "-color_trc", "-colorspace"} & set(args)
    if encoders is CPU_ENCODERS:
        assert args[args.index("-x265-params") + 1] == "log-level=error"


@pytest.mark.parametrize(("encoders", "fmt"), [(NVENC_ENCODERS, "p010le"),
                                               (CPU_ENCODERS, "yuv420p10le")])
def test_a_422_10bit_hevc_source_is_reencoded_to_420_main10(encoders, fmt: str) -> None:
    # The format filter takes the 4:2:2 down to 4:2:0, keeping 10 bits.
    probe = _probe("hevc", 1920, 1080, profile="Rext", pix_fmt="yuv422p10le")
    args, outputs = _cmd(probe, "", encoders=encoders)
    assert [o.final.name for o in outputs] == ["prepared.mkv"]
    assert args[args.index("-filter_complex") + 1] == f"[0:0]format={fmt}[ov0]"
    assert args[args.index("-profile:v") + 1] == "main10"


def test_an_8bit_444_hevc_source_is_reencoded_to_420_main() -> None:
    # 8 bits stay 8 bits: Main, the video mapped as it is and taken to
    # 4:2:0 by the encoder's pixel format.
    probe = _probe("hevc", 1920, 1080, profile="Rext", pix_fmt="yuv444p")
    args, _ = _cmd(probe, "source:hevc", encoders=CPU_ENCODERS)
    assert "-filter_complex" not in args
    assert args[args.index("-profile:v") + 1] == "main"
    assert args[args.index("-pix_fmt") + 1] == "yuv420p"


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
    def __init__(self, rc: int, stderr: str = "", stdout: str = "") -> None:
        self.returncode, self.stderr, self.stdout = rc, stderr, stdout


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


# ------------------------------------------------------- Dolby Vision 5
def _dv5(**extra: object) -> dict:
    """A Dolby Vision 5 source as ffprobe reports it from Matroska: HEVC
    Main 10, full range, no other colour tag, the record 5.6."""
    record = {"side_data_type": "DOVI configuration record", "dv_profile": 5, "dv_level": 6,
              "rpu_present_flag": 1, "dv_bl_signal_compatibility_id": 0}
    return _probe("hevc", 3840, 1920, profile="Main 10", pix_fmt="yuv420p10le",
                  color_range="pc", side_data_list=[record], **extra)


RESHAPE = ("format=yuv420p10le,hwupload,libplacebo=apply_dolbyvision=1:colorspace=bt2020nc"
           ":color_primaries=bt2020:color_trc=smpte2084:range=tv:format={fmt},"
           "hwdownload,format={fmt}")
HDR10_FLAGS = ["-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace",
               "bt2020nc", "-color_range", "tv"]
BT709_FLAGS = ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]


def test_dolby_vision_5_on_nvenc_is_reshaped_on_the_vulkan_device() -> None:
    """The whole command: the preflight's Vulkan device opened for the
    filters, the reshaping ahead of the encoder (down in NVENC's p010le,
    so nothing converts it after), Main 10 with HDR10's tags named, and
    a 4K movie's cap."""
    args, outputs = _cmd(_dv5(), "", vulkan=VULKAN)
    assert args == [
        "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "warning",
        "-init_hw_device", "vulkan=vk:0", "-filter_hw_device", "vk",
        "-fflags", "+genpts", "-avoid_negative_ts", "make_zero",
        "-i", str(SRC),
        "-filter_complex", f"[0:0]{RESHAPE.format(fmt='p010le')}[ov0]",
        "-map", "[ov0]",
        "-map", "0:a?", "-map", "0:s:0", "-map", "0:s:1", "-map", "0:s:2",
        "-c:v", "hevc_nvenc", "-preset", "p5", "-profile:v", "main10", "-pix_fmt", "p010le",
        "-rc:v", "vbr", "-cq", "23", "-maxrate", "14M", "-bufsize", "28M",
        "-b_ref_mode", "middle", "-spatial-aq", "1", "-rc-lookahead", "20",
        "-a53cc", "1", *HDR10_FLAGS,
        "-forced-idr", "1", "-force_key_frames", "expr:gte(t,n_forced*6)", "-g", "144",
        "-enc_time_base:v", "demux",
        "-c:a", "copy", "-c:s", "copy", "-c:s:1", "srt",
        "-f", "matroska", str(INBOX / "prepared.mkv.partial"),
    ]
    assert [(o.final.name, o.reshaped) for o in outputs] == [("prepared.mkv", True)]


def test_dolby_vision_5_on_a_cpu_comes_down_planar_for_x265() -> None:
    args, _ = _cmd(_dv5(), "source:hevc", encoders=CPU_ENCODERS, vulkan=VULKAN)
    assert args[args.index("-filter_complex") + 1] == (
        f"[0:0]{RESHAPE.format(fmt='yuv420p10le')}[ov0]")
    assert (args[args.index("-c:v") + 1], args[args.index("-pix_fmt") + 1]) == (
        "libx265", "yuv420p10le")
    assert args[args.index("-x265-params") + 1] == "log-level=error:hdr-opt=1:repeat-headers=1"
    start = args.index("-color_primaries")
    assert args[start:start + len(HDR10_FLAGS)] == HDR10_FLAGS


def test_dolby_vision_5_is_reshaped_once_before_the_split() -> None:
    # The H.264 rung is tone-mapped from the reshaped HDR10 picture, and
    # tagged as the BT.709 it is.
    args, outputs = _cmd(_dv5(), "source,720p", vulkan=VULKAN)
    graph = args[args.index("-filter_complex") + 1]
    assert graph == (
        f"[0:0]{RESHAPE.format(fmt='p010le')},split=2[bv0][bv1];"
        "[bv0]format=p010le[ov0];"
        f"[bv1]scale=1280:640,setsar=1,{ff.TONEMAP_CHAIN}[ov1]")
    assert graph.count("libplacebo") == 1 and args.count("-init_hw_device") == 1
    v0, v1 = _per_output(args)
    assert v0[v0.index("-color_primaries"):][:len(HDR10_FLAGS)] == HDR10_FLAGS
    assert v1[v1.index("-color_primaries"):][:len(BT709_FLAGS)] == BT709_FLAGS
    assert "-color_range" not in v1
    assert [o.reshaped for o in outputs] == [True, True]


def test_without_an_index_the_vulkan_device_is_ffmpegs_default() -> None:
    args, _ = _cmd(_dv5(), "", vulkan=VulkanDevice(index="", name="GPU", error=""))
    assert args[args.index("-init_hw_device") + 1] == "vulkan=vk"


def test_a_reshaping_command_without_a_vulkan_device_is_an_error() -> None:
    plan = plan_renditions(_dv5(), parse_ladder(""), NVENC_ENCODERS, vulkan=VULKAN)
    with pytest.raises(TranscodeError, match="profile 5 needs a Vulkan device"):
        build_encode_command(SRC, INBOX, plan, EncodeSettings(), probe_subtitles=SUBS,
                             video_index=0)


@pytest.mark.parametrize(("probe", "ladder"), [
    (_probe("h264", 1920, 1080), ""),
    (_probe("hevc", 3840, 2160, pix_fmt="yuv420p10le", color_transfer="smpte2084"),
     "source,1080p:hevc,720p"),
    (_probe("hevc", 3840, 2160, profile="Rext", pix_fmt="yuv422p10le",
            color_transfer="smpte2084"), ""),
    (_probe("hevc", 3840, 2160, profile="Main 10", pix_fmt="yuv420p10le",
            color_transfer="smpte2084", side_data_list=[
                {"side_data_type": "DOVI configuration record", "dv_profile": 8,
                 "dv_bl_signal_compatibility_id": 1}]), "source,720p"),
])
def test_no_other_source_opens_a_vulkan_device(probe: dict, ladder: str) -> None:
    # HDR10 and Dolby Vision 8.1 encode as before: no device, no libplacebo,
    # no colour flags of the reshaping, nothing checked after.
    args, outputs = _cmd(probe, ladder, vulkan=VULKAN)
    assert "-init_hw_device" not in args and "libplacebo" not in " ".join(args)
    assert "smpte2084" not in args
    assert not any(o.reshaped for o in outputs)


# ----------------------------------------------------- Vulkan preflight
LISTING = ("[AVHWDeviceContext @ 0x55d0c8e05a40] GPU listing:\n"
           "[AVHWDeviceContext @ 0x55d0c8e05a40]     0: {name} ({kind}) ({pci})\n"
           "[AVHWDeviceContext @ 0x55d0c8e05a40] Device 0 selected: {name} ({kind}) ({pci})\n"
           "[AVHWDeviceContext @ 0x55d0c8e05a40] Using device: {name}\n")
# What a GPU pod logged with the loader's messages on, before the driver's
# graphics libraries were complete.
NO_ICD = (
    "WARNING: [Loader Message] Code 0 : loader_scanned_icd_add: Could not get "
    "'vkCreateInstance' via 'vk_icdGetInstanceProcAddr' for ICD "
    "/usr/lib64/libGLX_nvidia.so.0\n"
    "[AVHWDeviceContext @ 0x5581c0a4b8c0] Instance creation failure: "
    "VK_ERROR_INCOMPATIBLE_DRIVER\n"
    "Device creation failed: -542398533.\n"
    "Failed to set value 'vulkan=vk' for option 'init_hw_device': Generic error in an "
    "external library\n"
    "Error parsing global options: Generic error in an external library\n")


@pytest.fixture
def loader(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ff, "vulkan_loader_version", lambda: "1.3.204")


def _preflight_line(logs: list[dict]) -> dict:
    [line] = [e for e in logs if e["event"] == "transcoder.vulkan.preflight"]
    return line


def test_the_vulkan_preflight_finds_the_gpu(monkeypatch, loader) -> None:
    calls = []

    def fake_run(args, **kw):
        calls.append((args, kw))
        return _Result(0, LISTING.format(name="NVIDIA GeForce RTX 3090", kind="discrete",
                                         pci="0x2204"))

    monkeypatch.setattr(subprocess, "run", fake_run)
    with capture_logs() as logs:
        device = ff.vulkan_preflight()
    assert device == VulkanDevice(index="0", name="NVIDIA GeForce RTX 3090 (discrete)",
                                  loader="1.3.204", error="")
    # One generated frame through the chain an encode runs, on ffmpeg's own
    # pick of a device, with the loader's complaints switched on.
    [(args, kw)] = calls
    assert args[args.index("-init_hw_device") + 1] == "vulkan=vk"
    assert args[args.index("-vf") + 1] == RESHAPE.format(fmt="yuv420p10le")
    assert kw["env"]["VK_LOADER_DEBUG"] == "error,warn"
    line = _preflight_line(logs)
    assert {k: line[k] for k in ("log_level", "ok", "loader", "device", "error",
                                 "dolby_vision_5")} == {
        "log_level": "info", "ok": True, "loader": "1.3.204",
        "device": "NVIDIA GeForce RTX 3090 (discrete)", "error": None,
        "dolby_vision_5": "reshaped to HDR10"}


def test_the_vulkan_preflight_says_why_there_is_no_device(monkeypatch, loader) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Result(1, NO_ICD))
    with capture_logs() as logs:
        device = ff.vulkan_preflight()
    assert not device.ok
    # The loader's own line and ffmpeg's, not ffmpeg's generic option errors.
    assert device.error == (
        "Vulkan loader 1.3.204: WARNING: [Loader Message] Code 0 : loader_scanned_icd_add: "
        "Could not get 'vkCreateInstance' via 'vk_icdGetInstanceProcAddr' for ICD "
        "/usr/lib64/libGLX_nvidia.so.0 | Instance creation failure: "
        "VK_ERROR_INCOMPATIBLE_DRIVER")
    line = _preflight_line(logs)
    assert {k: line[k] for k in ("log_level", "ok", "loader", "device", "error",
                                 "dolby_vision_5")} == {
        "log_level": "warning", "ok": False, "loader": "1.3.204", "device": None,
        "error": device.error, "dolby_vision_5": "refused: its step fails"}


def test_the_vulkan_preflight_without_a_loader(monkeypatch) -> None:
    monkeypatch.setattr(ff, "vulkan_loader_version", lambda: None)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Result(
        1, "[AVHWDeviceContext @ 0x55e1] Unable to open the libvulkan library!\n"
           "Device creation failed: -1313558101.\n"))
    assert ff.vulkan_preflight() == VulkanDevice(
        error="no Vulkan loader (libvulkan.so.1): Unable to open the libvulkan library!")


def test_the_vulkan_preflight_takes_no_software_device(monkeypatch, loader) -> None:
    lavapipe = LISTING.format(name="llvmpipe (LLVM 15.0.7, 128 bits)", kind="software",
                              pci="0x0")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Result(0, lavapipe))
    device = ff.vulkan_preflight()
    assert (device.ok, device.name) == (False, "llvmpipe (LLVM 15.0.7, 128 bits) (software)")
    assert device.error == ("Vulkan loader 1.3.204: the only Vulkan device is a software "
                            "one, llvmpipe (LLVM 15.0.7, 128 bits) (software)")
    # Taken only when asked for (the real runs of the tests).
    assert ff.vulkan_preflight(allow_software=True).index == "0"


def test_the_vulkan_preflight_without_ffmpeg(monkeypatch, loader) -> None:
    def missing(*_a, **_k):
        raise FileNotFoundError(2, "No such file or directory", "ffmpeg")

    monkeypatch.setattr(subprocess, "run", missing)
    device = ff.vulkan_preflight()
    assert not device.ok and "No such file or directory" in device.error


def test_the_loader_version_is_the_loaders_own(monkeypatch) -> None:
    class Loader:
        @staticmethod
        def vkEnumerateInstanceVersion(version) -> int:
            version._obj.value = (1 << 22) | (3 << 12) | 204  # VK_MAKE_API_VERSION
            return 0

    monkeypatch.setattr(ctypes, "CDLL", lambda _name: Loader())
    assert ff.vulkan_loader_version() == "1.3.204"
    monkeypatch.setattr(ctypes, "CDLL", lambda _name: object())  # a 1.0 loader
    assert ff.vulkan_loader_version() == "1.0"

    def none(name):
        raise OSError(f"{name}: cannot open shared object file")

    monkeypatch.setattr(ctypes, "CDLL", none)
    assert ff.vulkan_loader_version() is None


# ---------------------------------------- the checks around a reshaping
def _frames(*with_rpu: bool) -> str:
    rpu = [{"side_data_type": "Dolby Vision RPU Data"},
           {"side_data_type": "Dolby Vision Metadata"}]
    return json.dumps({"frames": [{"pts": i, **({"side_data_list": rpu} if has else {})}
                                  for i, has in enumerate(with_rpu)]})


def test_the_rpu_check_counts_the_frames_that_carry_one(monkeypatch) -> None:
    calls = []

    def fake_run(args, **_kw):
        calls.append(args)
        return _Result(0, stdout=_frames(True, True, False))

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert ff.dovi_metadata_frames(SRC, 0) == (3, 2)
    assert ff.dovi_metadata_frames(SRC, None, frames=3) == (3, 2)
    first, second = calls
    assert (first[first.index("-select_streams") + 1],
            first[first.index("-read_intervals") + 1]) == ("0", "%+#8")
    assert (second[second.index("-select_streams") + 1],
            second[second.index("-read_intervals") + 1]) == ("v:0", "%+#3")


def test_the_rpu_check_fails_with_ffprobe(monkeypatch) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Result(1, "Invalid data found"))
    with pytest.raises(TranscodeError, match="Invalid data found"):
        ff.dovi_metadata_frames(SRC, 0)


HDR10 = {"color_primaries": "bt2020", "color_transfer": "smpte2084", "color_space": "bt2020nc",
         "color_range": "tv"}
SDR = {"color_primaries": "bt709", "color_transfer": "bt709", "color_space": "bt709",
       "color_range": "tv"}


def _reshaped(tmp_path: Path, ladder: str = "source,720p"):
    settings = EncodeSettings(ladder=tuple(parse_ladder(ladder)), vulkan=VULKAN)
    plan = plan_renditions(_dv5(), list(settings.ladder), NVENC_ENCODERS, vulkan=VULKAN)
    return build_encode_command(SRC, tmp_path / "inbox", plan, settings,
                                probe_subtitles=SUBS, video_index=0)


def _encoder(stderr: str = "", streams: dict[str, dict] | None = None, probed: list | None = None):
    """subprocess.run for run_encode: the encode writes its partials and
    exits 0 with `stderr`; ffprobe answers an output's stream (by its file
    name) from `streams`."""
    def run(args, **_kw):
        if args[0] == "ffmpeg":
            for arg in args:
                if arg.endswith(".partial"):
                    Path(arg).write_bytes(b"\0")
            return _Result(0, stderr)
        name = Path(args[-1]).name.removesuffix(".partial")
        if probed is not None:
            probed.append(name)
        if "stream=width,height,start_time" in args:
            return _Result(0, stdout=json.dumps({"streams": [{"width": 3840, "height": 1920}]}))
        return _Result(0, stdout=json.dumps({"streams": [(streams or {})[name]]}))
    return run


def test_a_reshaped_encode_is_handed_off_when_its_outputs_are_what_they_must_be(
    monkeypatch, tmp_path: Path,
) -> None:
    args, outputs = _reshaped(tmp_path)
    monkeypatch.setattr(subprocess, "run", _encoder(streams={"prepared.mkv": HDR10,
                                                             "v1.mkv": SDR}))
    results, _ = ff.run_encode(args, outputs, log_label="i")
    assert sorted(p.name for p in (tmp_path / "inbox").iterdir()) == ["prepared.mkv", "v1.mkv"]
    assert [r.rung.id for r in results] == ["v0", "v1"]


@pytest.mark.parametrize(("stderr", "streams", "why"), [
    ("[hevc @ 0x55] Error parsing DOVI NAL unit.\n    Last message repeated 1 times\n",
     {"prepared.mkv": HDR10, "v1.mkv": SDR},
     "the decoder could not read the Dolby Vision RPU of some frames "
     "('Error parsing DOVI NAL unit'): those would play in the wrong colours"),
    ("", {"prepared.mkv": {**HDR10, "color_transfer": None}, "v1.mkv": SDR},
     "v0 came out as bt2020/None/bt2020nc/tv, not bt2020/smpte2084/bt2020nc/tv"),
    ("", {"prepared.mkv": {**HDR10, "color_range": "pc"}, "v1.mkv": SDR},
     "v0 came out as bt2020/smpte2084/bt2020nc/pc, not bt2020/smpte2084/bt2020nc/tv"),
    ("", {"prepared.mkv": HDR10, "v1.mkv": HDR10},
     "v1 came out as bt2020/smpte2084/bt2020nc/tv, not bt709/bt709/bt709/tv"),
    ("", {"prepared.mkv": {**HDR10, "side_data_list": [
        {"side_data_type": "DOVI configuration record"}]}, "v1.mkv": SDR},
     "v0 came out as bt2020/smpte2084/bt2020nc/tv with a DOVI record"),
])
def test_a_reshaped_encode_that_is_not_right_is_not_handed_off(
    monkeypatch, tmp_path: Path, stderr: str, streams: dict, why: str,
) -> None:
    # Fail closed: an exit 0 is not enough; nothing is renamed into place.
    args, outputs = _reshaped(tmp_path)
    monkeypatch.setattr(subprocess, "run", _encoder(stderr, streams))
    with pytest.raises(TranscodeError) as failed:
        ff.run_encode(args, outputs, log_label="i")
    assert str(failed.value).startswith(f"Dolby Vision to HDR10: {why}")
    assert str(failed.value).endswith("; nothing handed off")
    assert list((tmp_path / "inbox").iterdir()) == []


def test_any_other_encode_is_not_checked_for_its_colours(monkeypatch, tmp_path: Path) -> None:
    probed: list[str] = []
    settings = EncodeSettings()
    plan = plan_renditions(_probe("h264", 1920, 1080), list(settings.ladder), NVENC_ENCODERS)
    args, outputs = build_encode_command(SRC, tmp_path / "inbox", plan, settings,
                                         probe_subtitles=SUBS, video_index=0)
    monkeypatch.setattr(subprocess, "run", _encoder("[hevc @ 0x55] Error parsing DOVI NAL unit.",
                                                    probed=probed))
    ff.run_encode(args, outputs, log_label="i")
    assert probed == ["prepared.mkv"]  # the size probe only
