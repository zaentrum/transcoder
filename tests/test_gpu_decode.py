"""GPU decoding (README "GPU decoding"): which sources NVDEC decodes into
CUDA frames, the CUDA filter chains and commands, the startup probe, the
CPU run a failed CUDA run falls back to, and what the worker says of it —
no ffmpeg run. The GPU runs themselves need an NVIDIA GPU, which CI has
not: they were made on one (the README's numbers)."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest
from structlog.testing import capture_logs

from transcoder import ffmpeg as ff
from transcoder import worker
from transcoder.decision import (
    CPU_ENCODERS,
    NVENC_ENCODERS,
    Encoders,
    SourceInfo,
    parse_ladder,
    plan_renditions,
)
from transcoder.ffmpeg import (
    CUDA_DECODE_ARGS,
    TONEMAP_CHAIN,
    EncodeSettings,
    RungResult,
    TranscodeError,
    build_encode_command,
    cuda_decode_blocker,
)
from transcoder.katalog import ClaimedItem

SRC = Path("/media/movie.mkv")
INBOX = Path("/packages/_inbox/item")
SUBS = [{"codec_name": "subrip"}, {"codec_name": "mov_text"}]
PQ = {"color_transfer": "smpte2084", "color_primaries": "bt2020", "color_space": "bt2020nc"}


def _probe(codec: str, w: int, h: int, pix_fmt: str = "yuv420p", profile: str = "",
           **extra: object) -> dict:
    video = {"codec_name": codec, "width": w, "height": h, "pix_fmt": pix_fmt,
             "profile": profile, "avg_frame_rate": "24000/1001", **extra}
    return {"video": video, "video_index": 0, "audio": [], "subtitles": SUBS}


H264 = _probe("h264", 1920, 1080, profile="High")
UHD_HDR10 = _probe("hevc", 3840, 2160, "yuv420p10le", "Main 10", bit_rate="40000000", **PQ)
DV81 = _probe("hevc", 3840, 2160, "yuv420p10le", "Main 10", **PQ, side_data_list=[
    {"side_data_type": "DOVI configuration record", "dv_profile": 8,
     "dv_bl_signal_compatibility_id": 1}])
DV5 = _probe("hevc", 3840, 1920, "yuv420p10le", "Main 10", color_range="pc", side_data_list=[
    {"side_data_type": "DOVI configuration record", "dv_profile": 5,
     "dv_bl_signal_compatibility_id": 0}])


# ------------------------------------------------------- what NVDEC takes
@pytest.mark.parametrize(("probe", "blocker"), [
    (H264, None),
    (_probe("h264", 1920, 1080, "yuvj420p", "Main"), None),
    (_probe("h264", 1280, 720, profile="Constrained Baseline"), None),
    (_probe("hevc", 1920, 1080, profile="Main"), None),
    (UHD_HDR10, None),
    (DV81, None),                                          # its base layer
    (_probe("vp9", 1920, 1080, profile="Profile 0"), None),
    (_probe("vp9", 3840, 2160, "yuv420p10le", "Profile 2", **PQ), None),
    (_probe("av1", 3840, 2160, "yuv420p10le", "Main"), None),
    # NVDEC has no H.264 above 8 bits, nor any 4:2:2, nor HEVC 4:4:4 or
    # 12-bit as NVENC takes it: the CPU, as before.
    (_probe("h264", 1920, 1080, "yuv420p10le", "High 10"), "10-bit"),
    (_probe("h264", 1920, 1080, "yuv422p", "High 4:2:2"), "4:2:2"),
    (_probe("h264", 1920, 1080, "yuv444p", "High 4:4:4 Predictive"), "4:4:4"),
    (_probe("h264", 1920, 1080, profile="Extended"), "profile=extended"),
    (_probe("hevc", 1920, 1080, "yuv422p10le", "Rext"), "4:2:2"),
    (_probe("hevc", 1920, 1080, "yuv444p", "Rext"), "4:4:4"),
    (_probe("hevc", 1920, 1080, "yuv420p12le", "Rext"), "12-bit"),
    (_probe("hevc", 1920, 1080, "yuv420p", "Rext"), "profile=rext"),
    (_probe("vp9", 1920, 1080, "yuv444p", "Profile 1"), "4:4:4"),
    (_probe("vp9", 1920, 1080, "yuv420p12le", "Profile 2"), "12-bit"),
    (_probe("av1", 1920, 1080, "gray10le", "Main"), "4:0:0"),
    (_probe("hevc", 1920, 1080, "", "Main"), "pix_fmt=unknown"),
    # Interlaced sources, coded as fields; codecs left to the CPU.
    (_probe("h264", 1920, 1080, profile="High", field_order="tt"), "interlaced"),
    (_probe("h264", 1920, 1080, profile="High", field_order="bb"), "interlaced"),
    (_probe("h264", 1920, 1080, profile="High", field_order="progressive"), None),
    (_probe("mpeg2video", 1920, 1080, profile="Main"), "codec=mpeg2video"),
    (_probe("vc1", 1920, 1080, profile="Advanced"), "codec=vc1"),
    (_probe("", 1920, 1080), "codec=unknown"),
    # Dolby Vision 5 is reshaped from system memory (DOVI_RESHAPE_CHAIN).
    (DV5, "dolby_vision_5"),
])
def test_which_sources_nvdec_decodes(probe: dict, blocker: str | None) -> None:
    assert SourceInfo.from_probe(probe).cuda_decode_blocker == blocker


def _plan(probe: dict, ladder: str = "", encoders: Encoders = NVENC_ENCODERS):
    return plan_renditions(probe, parse_ladder(ladder), encoders)


def test_a_plan_with_a_cpu_encoder_decodes_on_the_cpu() -> None:
    # A host whose H.264 is libx264: its rung takes system memory frames,
    # and the source is decoded once for every rung.
    plan = _plan(H264, "source,720p", Encoders(hevc="hevc_nvenc", h264="libx264"))
    settings = EncodeSettings(decoder="cuda")
    assert cuda_decode_blocker(plan, settings) == "encoder=libx264"
    assert cuda_decode_blocker(_plan(H264, "source,720p"), settings) is None
    assert cuda_decode_blocker(_plan(H264), EncodeSettings()) == "cuda off"


# ----------------------------------------------------------- the commands
def _cmds(probe: dict, ladder: str = "", **kw: object):
    settings = EncodeSettings(ladder=tuple(parse_ladder(ladder)), decoder="cuda", **kw)
    plan = plan_renditions(probe, list(settings.ladder), settings.encoders)
    cpu = build_encode_command(SRC, INBOX, plan, settings, probe_subtitles=SUBS, video_index=0)
    cuda = build_encode_command(SRC, INBOX, plan, settings, probe_subtitles=SUBS,
                                video_index=0, decoder="cuda")
    return cpu, cuda


def _graph(args: list[str]) -> str:
    return args[args.index("-filter_complex") + 1]


def test_a_4k_main10_encode_decodes_on_the_gpu() -> None:
    """The case it is for: a 4K HEVC Main 10 source over its cap, one
    source-size NVENC encode. NVDEC hands its p010 frames on in GPU memory
    and NVENC takes them as they are (no -pix_fmt: it would be a conversion
    in system memory). All else after the decode is the CPU command's."""
    (cpu, outputs), (cuda, cuda_outputs) = _cmds(UHD_HDR10)
    assert cuda == [
        "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "warning",
        # The decoder's CUDA device, named, and a second beside it, so that
        # an encoder of system memory frames opens a context of its own.
        "-init_hw_device", "cuda=dec", "-init_hw_device", "cuda=spare",
        "-hwaccel", "cuda", "-hwaccel_device", "dec", "-hwaccel_output_format", "cuda",
        "-filter_hw_device", "dec",
        "-fflags", "+genpts", "-avoid_negative_ts", "make_zero",
        "-i", str(SRC),
        "-filter_complex", "[0:0]format=cuda[ov0]",
        "-map", "[ov0]", "-map", "0:a?", "-map", "0:s:0", "-map", "0:s:1",
        "-c:v", "hevc_nvenc", "-preset", "p5", "-profile:v", "main10",
        "-rc:v", "vbr", "-cq", "23", "-maxrate", "14M", "-bufsize", "28M",
        "-b_ref_mode", "middle", "-spatial-aq", "1", "-rc-lookahead", "20",
        "-a53cc", "1",
        "-forced-idr", "1", "-force_key_frames", "expr:gte(t,n_forced*6)", "-g", "144",
        "-enc_time_base:v", "demux",
        "-c:a", "copy", "-c:s", "copy", "-c:s:1", "srt",
        "-f", "matroska", str(INBOX / "prepared.mkv.partial"),
    ]
    # The CPU command: the same encode of the software decoder's frames,
    # taken to p010le on the CPU.
    assert _graph(cpu) == "[0:0]format=p010le[ov0]"
    assert cpu[cpu.index("-pix_fmt"):cpu.index("-pix_fmt") + 2] == ["-pix_fmt", "p010le"]
    assert [o.partial for o in outputs] == [o.partial for o in cuda_outputs]


def _per_output(args: list[str]) -> list[list[str]]:
    """The command's options split by output (each ends with its file),
    from after the filter graph (or the input)."""
    start = (args.index("-filter_complex") if "-filter_complex" in args
             else args.index("-i")) + 2
    chunks = []
    for i, arg in enumerate(args):
        if arg.endswith(".partial"):
            chunks.append(args[start:i + 1])
            start = i + 1
    return chunks


@pytest.mark.parametrize(("probe", "ladder"), [
    (H264, ""), (H264, "source,720p,480p"), (H264, "720p:h264,480p:h264"), (UHD_HDR10, ""),
    (UHD_HDR10, "source,720p,480p"), (DV81, "source:hevc,720p"),
    (_probe("vp9", 1920, 1080, "yuv420p10le", "Profile 2"), "source:hevc,540p"),
])
def test_the_cuda_command_is_the_cpu_command_but_for_the_decode(probe: dict, ladder: str) -> None:
    # Per output, every option the CPU command gives it, but the -pix_fmt
    # of the one encoder that takes the CUDA frames: the same maps,
    # outputs, encoders, rate control and keyframes.
    (cpu, _), (cuda, _) = _cmds(probe, ladder)
    assert cuda[6:6 + len(CUDA_DECODE_ARGS)] == CUDA_DECODE_ARGS and "-hwaccel" not in cpu
    assert cuda.count("-i") == cpu.count("-i") == 1
    fed = 0
    for c, g in zip(_per_output(cpu), _per_output(cuda), strict=True):
        if "-pix_fmt" not in g:
            fed += 1
            at = c.index("-pix_fmt")
            c = c[:at] + c[at + 2:]
        # The CPU command maps a single unfiltered rung as it is, and its
        # negative maps keep the data streams and cover art out of it.
        text = " ".join(c).replace(" -map -0:d -map -0:v:m:attached_pic", "")
        assert text.replace("-map 0:0 ", "-map [ov0] ") == " ".join(g)
    # None when every encoded rung's frames come down (Dolby Vision 8.1's
    # v0 is a copy, its 720p H.264 tone-mapped).
    plan = _plan(probe, ladder)
    encoded = [r for r in plan.rungs if r.mode == "encode"]
    assert fed == (1 if any(not ff._down_on_the_cpu(r, plan.source) for r in encoded) else 0)


def test_a_uhd_hdr10_ladder_scales_on_the_gpu_and_tone_maps_on_the_cpu() -> None:
    (_, _), (cuda, _) = _cmds(UHD_HDR10, "source,720p,480p")
    assert _graph(cuda) == (
        "[0:0]split=3[bv0][bv1][bv2];"
        "[bv0]format=cuda[ov0];"
        "[bv1]scale_cuda=1280:720:format=p010le:interp_algo=bilinear,setsar=1,"
        f"hwdownload,format=p010le,format=yuv420p10le,{TONEMAP_CHAIN}[ov1];"
        # Further down than 3x: a half first.
        "[bv2]scale_cuda=1920:1080:interp_algo=bilinear,"
        "scale_cuda=854:480:format=p010le:interp_algo=bilinear,setsar=1,"
        f"hwdownload,format=p010le,format=yuv420p10le,{TONEMAP_CHAIN}[ov2]")
    # v0's NVENC takes CUDA frames; the tone-mapped rungs come from system
    # memory, in the CPU path's format.
    pix = [c[c.index("-pix_fmt") + 1] if "-pix_fmt" in c else None for c in _per_output(cuda)]
    assert pix == [None, "yuv420p", "yuv420p"]


@pytest.mark.parametrize(("probe", "ladder", "graph"), [
    # The 8-bit ladder: v0 on CUDA frames, the H.264 rungs scaled on the GPU
    # and brought down in the CPU path's yuv420p.
    (H264, "source,720p,480p",
     "[0:0]split=3[bv0][bv1][bv2];[bv0]format=cuda[ov0];"
     "[bv1]scale_cuda=1280:720:format=nv12:interp_algo=bilinear,setsar=1,"
     "hwdownload,format=nv12,format=yuv420p[ov1];"
     "[bv2]scale_cuda=854:480:format=nv12:interp_algo=bilinear,setsar=1,"
     "hwdownload,format=nv12,format=yuv420p[ov2]"),
    # The extras' default ladder: its first rung takes the CUDA frames.
    (H264, "720p:h264,480p:h264",
     "[0:0]split=2[bv0][bv1];"
     "[bv0]scale_cuda=1280:720:format=nv12:interp_algo=bilinear,setsar=1[ov0];"
     "[bv1]scale_cuda=854:480:format=nv12:interp_algo=bilinear,setsar=1,"
     "hwdownload,format=nv12,format=yuv420p[ov1]"),
    # v0 a copy (HEVC within its cap): the 720p rung takes them.
    (_probe("hevc", 1920, 1080, profile="Main", bit_rate="4000000"), "source,720p,480p",
     "[0:0]split=2[bv1][bv2];"
     "[bv1]scale_cuda=1280:720:format=nv12:interp_algo=bilinear,setsar=1[ov1];"
     "[bv2]scale_cuda=854:480:format=nv12:interp_algo=bilinear,setsar=1,"
     "hwdownload,format=nv12,format=yuv420p[ov2]"),
    # A 10-bit SDR source's H.264 rungs: scaled on the GPU at 10 bits,
    # taken to 8 by swscale with its dither, as on the CPU path (scale_cuda
    # would cut the two bits off).
    (_probe("hevc", 3840, 2160, "yuv420p10le", "Main 10", bit_rate="40000000"),
     "source,1080p,720p", "[0:0]split=3[bv0][bv1][bv2];[bv0]format=cuda[ov0];"
                          "[bv1]scale_cuda=1920:1080:format=p010le:interp_algo=bilinear,setsar=1,"
                          "hwdownload,format=p010le,format=yuv420p[ov1];"
                          "[bv2]scale_cuda=1280:720:format=p010le:interp_algo=bilinear,setsar=1,"
                          "hwdownload,format=p010le,format=yuv420p[ov2]"),
    # The same as the extras' ladder: no rung's encoder takes CUDA frames.
    (_probe("vp9", 1920, 1080, "yuv420p10le", "Profile 2"), "720p:h264,480p:h264",
     "[0:0]split=2[bv0][bv1];"
     "[bv0]scale_cuda=1280:720:format=p010le:interp_algo=bilinear,setsar=1,"
     "hwdownload,format=p010le,format=yuv420p[ov0];"
     "[bv1]scale_cuda=854:480:format=p010le:interp_algo=bilinear,setsar=1,"
     "hwdownload,format=p010le,format=yuv420p[ov1]"),
    # An 8-bit HDR source's Main 10: NVDEC's nv12 taken to p010 on the GPU.
    (_probe("hevc", 1920, 1080, "yuv420p", "Main", bit_rate="40000000", **PQ), "",
     "[0:0]scale_cuda=format=p010le[ov0]"),
    # Its tone-mapped rung at its own size: down at once, no scale_cuda.
    (_probe("hevc", 1280, 720, "yuv420p10le", "Main 10", **PQ), "720p",
     f"[0:0]hwdownload,format=p010le,format=yuv420p10le,{TONEMAP_CHAIN}[ov0]"),
    # An anamorphic source: scaled from its stored size to square pixels.
    (_probe("h264", 720, 576, profile="High", sample_aspect_ratio="64:45"), "source:hevc,480p",
     "[0:0]split=2[bv0][bv1];[bv0]format=cuda[ov0];"
     "[bv1]scale_cuda=854:480:format=nv12:interp_algo=bilinear,setsar=1,"
     "hwdownload,format=nv12,format=yuv420p[ov1]"),
])
def test_cuda_chains(probe: dict, ladder: str, graph: str) -> None:
    (_, _), (cuda, _) = _cmds(probe, ladder)
    assert _graph(cuda) == graph
    # The one encoder named no -pix_fmt is the one the CUDA frames reach:
    # the first whose frames need not come down (ffmpeg._down_on_the_cpu).
    plan = _plan(probe, ladder)
    encoded = [r for r in plan.rungs if r.mode == "encode"]
    fed = [i for i, c in enumerate(_per_output(cuda)) if "-pix_fmt" not in c]
    assert fed == ([0] if any(not ff._down_on_the_cpu(r, plan.source) for r in encoded)
                   else [])


@pytest.mark.parametrize(("size", "to", "steps"), [
    ((3840, 2160), (1920, 1080), [(1920, 1080)]),
    ((3840, 2160), (1280, 720), [(1280, 720)]),             # 3x: one step
    ((3840, 2160), (854, 480), [(1920, 1080), (854, 480)]),
    ((3840, 2160), (640, 360), [(1920, 1080), (640, 360)]),
    ((1920, 1080), (640, 360), [(640, 360)]),
    ((7680, 4320), (854, 480), [(3840, 2160), (1920, 1080), (854, 480)]),
    ((3840, 1600), (854, 356), [(1920, 800), (854, 356)]),
    ((4096, 1716), (1280, 536), [(2048, 858), (1280, 536)]),  # 3.2x wide
])
def test_cuda_scale_steps(size: tuple[int, int], to: tuple[int, int],
                          steps: list[tuple[int, int]]) -> None:
    assert ff._cuda_scale_steps(*size, *to) == steps


@pytest.mark.parametrize(("probe", "ladder", "encoders", "why"), [
    (_probe("hevc", 1920, 1080, "yuv422p10le", "Rext"), "", NVENC_ENCODERS, "4:2:2"),
    (DV5, "", NVENC_ENCODERS, "dolby_vision_5"),
    # Two HEVC encodes opening at once: NVENC_LEVEL_RACE.
    (UHD_HDR10, "source,1080p:hevc", NVENC_ENCODERS, "hevc_rungs=2"),
    (H264, "source,720p:hevc,480p:hevc", NVENC_ENCODERS, "hevc_rungs=3"),
    (H264, "source,720p", Encoders(hevc="hevc_nvenc", h264="libx264"), "encoder=libx264"),
    (_probe("mpeg2video", 1920, 1080), "source:hevc", CPU_ENCODERS, "encoder=libx265"),
])
def test_a_plan_nvdec_does_not_serve_has_no_cuda_command(
    probe: dict, ladder: str, encoders: Encoders, why: str,
) -> None:
    settings = EncodeSettings(ladder=tuple(parse_ladder(ladder)), encoders=encoders)
    plan = plan_renditions(probe, list(settings.ladder), encoders,
                           vulkan=ff.VulkanDevice(index="0", name="GPU", error=""))
    with pytest.raises(TranscodeError, match=f"into CUDA frames: {why}$"):
        build_encode_command(SRC, INBOX, plan, settings, probe_subtitles=SUBS, decoder="cuda")
    with pytest.raises(TranscodeError, match="no decoder 'nvdec'"):
        build_encode_command(SRC, INBOX, plan, settings, probe_subtitles=SUBS, decoder="nvdec")


# -------------------------------------------------------------- the probe
class _Result:
    def __init__(self, rc: int, stderr: str = "") -> None:
        self.returncode, self.stderr, self.stdout = rc, stderr, ""


def _probe_line(logs: list[dict]) -> dict:
    [line] = [e for e in logs if e["event"] == "transcoder.decoder.probe"]
    return {k: line[k] for k in ("log_level", "decoder", "ok", "reason", "error")}


def test_the_probe_runs_the_gpu_decode_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(args, **_kw):
        calls.append(args)
        return _Result(0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with capture_logs() as logs:
        assert ff.detect_decoder("auto", NVENC_ENCODERS) == "cuda"
    # A Main 10 clip from hevc_nvenc, then NVDEC into CUDA frames, a
    # conversion on the GPU and NVENC from GPU memory.
    make, decode = calls
    assert make[make.index("-c:v") + 1:make.index("-c:v") + 4] == ["hevc_nvenc", "-profile:v",
                                                                  "main10"]
    clip = make[-1]
    assert clip.endswith("probe.mkv") and not Path(clip).exists()  # its temp dir is gone
    assert decode[decode.index("-init_hw_device"):decode.index("-i")] == CUDA_DECODE_ARGS
    assert decode[decode.index("-i") + 1] == clip
    assert "scale_cuda=format=nv12" in decode[decode.index("-filter_complex") + 1]
    assert _probe_line(logs) == {"log_level": "info", "decoder": "cuda", "ok": True,
                                 "reason": None, "error": None}


def test_a_failed_probe_decodes_on_the_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    runs = iter([_Result(0), _Result(1, "[hevc @ 0x55] Hardware is lacking required "
                                        "capabilities\nError while decoding stream")])
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: next(runs))
    with capture_logs() as logs:
        assert ff.detect_decoder("auto", NVENC_ENCODERS) == "cpu"
    assert _probe_line(logs) == {
        "log_level": "warning", "decoder": "cpu", "ok": False, "reason": "the probe failed",
        "error": "[hevc @ 0x55] Hardware is lacking required capabilities\n"
                 "Error while decoding stream"}


def test_a_probe_without_ffmpeg_decodes_on_the_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*_a, **_k):
        raise FileNotFoundError(2, "No such file or directory", "ffmpeg")

    monkeypatch.setattr(subprocess, "run", missing)
    assert ff.detect_decoder("auto", NVENC_ENCODERS) == "cpu"


@pytest.mark.parametrize(("mode", "encoders", "reason"), [
    ("cpu", NVENC_ENCODERS, "DECODER=cpu"),
    (" CPU ", NVENC_ENCODERS, "DECODER=cpu"),
    ("auto", CPU_ENCODERS, "HEVC is encoded with libx265"),
    ("", Encoders(hevc="libx265", h264="h264_nvenc"), "HEVC is encoded with libx265"),
])
def test_no_probe_is_run_for_the_cpu(monkeypatch: pytest.MonkeyPatch, mode: str,
                                     encoders: Encoders, reason: str) -> None:
    def boom(*_a, **_k):
        raise AssertionError("must not probe")

    monkeypatch.setattr(subprocess, "run", boom)
    with capture_logs() as logs:
        assert ff.detect_decoder(mode, encoders) == "cpu"
    assert _probe_line(logs) == {"log_level": "info", "decoder": "cpu", "ok": None,
                                 "reason": reason, "error": None}


def test_a_decoder_setting_it_does_not_know_fails_startup() -> None:
    with pytest.raises(TranscodeError, match="DECODER must be auto or cpu"):
        ff.detect_decoder("nvdec")


# ------------------------------------------------------- the fallback run
def _commands(tmp_path: Path, probe: dict = UHD_HDR10, ladder: str = "source,720p"):
    settings = EncodeSettings(ladder=tuple(parse_ladder(ladder)), decoder="cuda")
    plan = plan_renditions(probe, list(settings.ladder), settings.encoders)
    inbox = tmp_path / "inbox"
    cpu, outputs = build_encode_command(SRC, inbox, plan, settings, probe_subtitles=SUBS)
    cuda, _ = build_encode_command(SRC, inbox, plan, settings, probe_subtitles=SUBS,
                                   decoder="cuda")
    return cpu, cuda, outputs, inbox


# The probe of an NVENC output's level, DPB and profile (ffmpeg._stream_facts).
FACTS = "stream=codec_name,profile,level,width,height,refs,r_frame_rate,avg_frame_rate"


class Runs:
    """subprocess.run and ffmpeg._run_watched for run_encode: an ffmpeg run
    writes its partials and exits as `results` says for its decoder ("cuda",
    "cpu"); the GPU run stands still when `stall`; the probes of its outputs
    answer what NVENC makes of their rungs (its profile, DPB 6, the lowest
    level), but for what `facts` says for a decoder and an output."""

    def __init__(self, outputs, results, facts=None, stall=False) -> None:
        self.rungs = {o.partial.name.removesuffix(".partial"): o.rung for o in outputs}
        self.results, self.facts, self.stall = results, facts or {}, stall
        self.runs: list[str] = []
        # The partials each ffmpeg run found on disk when it started.
        self.found: list[list[str]] = []

    def _facts(self, name: str) -> dict:
        rung = self.rungs[name]
        hevc = rung.codec == "hevc"
        own = {"codec_name": rung.codec, "width": rung.width, "height": rung.height,
               "profile": ("Main 10" if rung.ten_bit else "Main") if hevc else "High",
               "level": ff.max_level(rung.codec, rung.width, rung.height, 24000 / 1001,
                                     rung.maxrate_bps or 8_000_000),
               "refs": 1 if hevc else 4, "dpb": 6,
               "r_frame_rate": "24000/1001", "avg_frame_rate": "24000/1001"}
        return {**own, **self.facts.get(self.runs[-1], {}).get(name, {})}

    def run(self, args, **_kw):
        if args[0] == "ffmpeg" and "trace_headers" in args:
            name = Path(args[args.index("-i") + 1]).name.removesuffix(".partial")
            facts = self._facts(name)
            vui = Fraction(facts.get("vui_frame_rate", "0"))
            if facts["codec_name"] == "hevc":
                return _Result(0, "[trace_headers @ 0x55] 230  sps_max_dec_pic_buffering_minus1[0]"
                                  f"  00110 = {facts['dpb'] - 1}\n"
                                  + (f"[trace_headers @ 0x55] 380  vui_num_units_in_tick  01 = "
                                     f"{vui.denominator}\n[trace_headers @ 0x55] 412  "
                                     f"vui_time_scale  01 = {vui.numerator}\n" if vui else ""))
            return _Result(0, f"[trace_headers @ 0x55] 64  max_num_ref_frames  00101 = "
                              f"{facts['refs']}\n"
                              + (f"[trace_headers @ 0x55] 120  num_units_in_tick  01 = "
                                 f"{vui.denominator}\n[trace_headers @ 0x55] 152  time_scale  01 = "
                                 f"{2 * vui.numerator}\n" if vui else ""))
        if args[0] == "ffmpeg":
            decoder = "cuda" if "-hwaccel" in args else "cpu"
            self.runs.append(decoder)
            partials = [Path(arg) for arg in args if arg.endswith(".partial")]
            self.found.append(sorted(p.name for p in partials if p.exists()))
            for partial in partials:
                partial.write_bytes(b"\0" * 10)
            rc, stderr = self.results[decoder]
            return _Result(rc, stderr)
        name = Path(args[-1]).name.removesuffix(".partial")
        if FACTS in args:
            # ffprobe's refs is 1, whatever the stream's reference frames.
            facts = {**{k: v for k, v in self._facts(name).items()
                        if k not in ("dpb", "vui_frame_rate")}, "refs": 1}
            result = _Result(0)
            result.stdout = json.dumps({"streams": [facts]})
            return result
        result = _Result(0)
        result.stdout = '{"streams": [{"width": 3840, "height": 2160, "start_time": "0.021"}]}'
        return result

    def watched(self, args, *_a, **_k):
        return self.run(args), self.stall


def _runs(monkeypatch, outputs, results, **kw) -> Runs:
    runs = Runs(outputs, results, **kw)
    monkeypatch.setattr(subprocess, "run", runs.run)
    monkeypatch.setattr(ff, "_run_watched", runs.watched)
    return runs


def test_a_cuda_run_that_works_is_the_encode(monkeypatch, tmp_path: Path) -> None:
    cpu, cuda, outputs, inbox = _commands(tmp_path)
    runs = _runs(monkeypatch, outputs, {"cuda": (0, "")})
    with capture_logs() as logs:
        results, _ = ff.run_encode(cuda, outputs, log_label="i", cpu_fallback=cpu)
    assert runs.runs == ["cuda"]
    assert [(r.rung.id, r.decoder) for r in results] == [("v0", "cuda"), ("v1", "cuda")]
    assert sorted(p.name for p in inbox.iterdir()) == ["prepared.mkv", "v1.mkv"]
    events = [(e["event"], e.get("decoder")) for e in logs]
    assert events == [("transcoder.encode.start", "cuda"), ("transcoder.encode.done", "cuda")]


def test_a_cuda_run_that_fails_is_made_again_on_the_cpu(monkeypatch, tmp_path: Path) -> None:
    cpu, cuda, outputs, inbox = _commands(tmp_path)
    refusal = ("[graph 0 input from stream 0:0 @ 0x55] Impossible to convert between the "
               "formats supported by the filter 'graph 0 input' and the filter 'format'")
    runs = _runs(monkeypatch, outputs, {"cuda": (218, refusal), "cpu": (0, "")})
    with capture_logs() as logs:
        results, _ = ff.run_encode(cuda, outputs, log_label="i", cpu_fallback=cpu)
    assert runs.runs == ["cuda", "cpu"]
    # The GPU run's partials are gone before the CPU run starts.
    assert runs.found == [[], []]
    assert [r.decoder for r in results] == ["cpu", "cpu"]
    assert sorted(p.name for p in inbox.iterdir()) == ["prepared.mkv", "v1.mkv"]
    [failed] = [e for e in logs if e["event"] == "transcoder.encode.cuda_failed"]
    assert (failed["log_level"], failed["why"], failed["exit_code"], failed["error"],
            failed["then"]) == ("warning", "exit 218", 218, refusal, "decoding on the CPU")
    [done] = [e for e in logs if e["event"] == "transcoder.encode.done"]
    assert done["decoder"] == "cpu"


def test_only_the_cpu_runs_failure_fails_the_encode(monkeypatch, tmp_path: Path) -> None:
    cpu, cuda, outputs, inbox = _commands(tmp_path)
    runs = _runs(monkeypatch, outputs, {"cuda": (1, "CUDA_ERROR_OUT_OF_MEMORY"),
                                        "cpu": (1, "No space left on device")})
    with pytest.raises(TranscodeError, match="No space left on device") as failed:
        ff.run_encode(cuda, outputs, log_label="i", cpu_fallback=cpu)
    assert "CUDA_ERROR" not in str(failed.value)
    assert runs.runs == ["cuda", "cpu"] and list(inbox.iterdir()) == []


@pytest.mark.parametrize(("stderr", "facts", "stall", "why"), [
    # NVENC_LEVEL_RACE: the HEVC rung's DPB one larger, its level raised.
    ("", {"prepared.mkv": {"dpb": 7, "level": 180}}, False,
     "v0 came out with a DPB of 7, NVENC's own is at most 6"),
    ("", {"v1.mkv": {"refs": 5}}, False,
     "v1 came out with a DPB of 5, NVENC's own is at most 4"),
    # A level above the lowest its size, frame rate and maxrate allow.
    ("", {"prepared.mkv": {"level": 153}}, False, "v0 came out at level 153, above 150"),
    ("", {"v1.mkv": {"level": 32}}, False, "v1 came out at level 32, above 31"),
    ("", {"prepared.mkv": {"profile": "Main"}}, False, "v0 came out Main, not Main 10"),
    # A crop NVDEC frames leave on (cuda_frame_blocker's): not the rung's size.
    ("", {"prepared.mkv": {"height": 2176}}, False, "v0 came out 3840x2176, not 3840x2160"),
    # A frame the GPU did not decode; a GPU that stood still.
    ("[hevc @ 0x55] hardware accelerator failed to decode picture\n", {}, False,
     "decoder error (hardware accelerator failed)"),
    ("[vist#0:0/hevc @ 0x55] [dec:hevc @ 0x56] Decoding error: Invalid data found\n", {}, False,
     "decoder error (decoding error)"),
    ("", {}, True, "no progress for 600 s"),
])
def test_a_gpu_run_that_exits_0_but_is_not_the_cpu_paths_is_made_again_on_the_cpu(
    monkeypatch, tmp_path: Path, stderr: str, facts: dict, stall: bool, why: str,
) -> None:
    cpu, cuda, outputs, inbox = _commands(tmp_path)
    runs = _runs(monkeypatch, outputs, {"cuda": (0, stderr), "cpu": (0, "")},
                 facts={"cuda": facts}, stall=stall)
    with capture_logs() as logs:
        results, _ = ff.run_encode(cuda, outputs, log_label="i", cpu_fallback=cpu)
    assert runs.runs == ["cuda", "cpu"] and [r.decoder for r in results] == ["cpu", "cpu"]
    [failed] = [e for e in logs if e["event"] == "transcoder.encode.cuda_failed"]
    assert (failed["why"], failed["exit_code"]) == (why, 0)


@pytest.mark.parametrize(("facts", "why"), [
    ({"prepared.mkv": {"dpb": 7, "level": 180}}, "v0 came out with a DPB of 7, NVENC's own is "
                                                 "at most 6"),
    ({"prepared.mkv": {"level": 153}}, "v0 came out at level 153, above 150"),
])
def test_an_nvenc_run_on_the_cpus_decode_whose_level_jumped_fails(
    monkeypatch, tmp_path: Path, facts: dict, why: str,
) -> None:
    # No GPU run to fall back from: nothing handed off, the step fails.
    cpu, _cuda, outputs, inbox = _commands(tmp_path)
    _runs(monkeypatch, outputs, {"cpu": (0, "")}, facts={"cpu": facts})
    with pytest.raises(TranscodeError) as failed:
        ff.run_encode(cpu, outputs, log_label="i")
    assert str(failed.value) == f"NVENC: {why}; nothing handed off"
    assert list(inbox.iterdir()) == []


def test_the_cpu_runs_sizes_and_unknown_facts_are_not_judged(monkeypatch, tmp_path: Path) -> None:
    # A CPU run's size is the encoder's (an odd source's, say); a value
    # ffprobe gives no answer for proves nothing; libx265 is not NVENC.
    cpu, _cuda, outputs, _inbox = _commands(tmp_path)
    _runs(monkeypatch, outputs, {"cpu": (0, "")},
          facts={"cpu": {"prepared.mkv": {"height": 2176, "level": None, "profile": None}}})
    ff.run_encode(cpu, outputs, log_label="i")
    settings = EncodeSettings(ladder=tuple(parse_ladder("")), encoders=CPU_ENCODERS)
    plan = plan_renditions(UHD_HDR10, list(settings.ladder), CPU_ENCODERS)
    args, x265 = build_encode_command(SRC, tmp_path / "x265", plan, settings, probe_subtitles=SUBS)
    _runs(monkeypatch, x265, {"cpu": (0, "")},
          facts={"cpu": {"prepared.mkv": {"dpb": 7, "level": 186}}})
    ff.run_encode(args, x265, log_label="i")


@pytest.mark.parametrize(("vui", "why"), [
    ("50", None), (None, "v1 came out at level 32, above 31"),
])
def test_the_level_is_judged_at_the_frame_rate_nvenc_was_given(
    monkeypatch, tmp_path: Path, vui: str | None, why: str | None,
) -> None:
    # A 720p H.264 rung at level 3.2: right for the 50 frames a second its
    # VUI says NVENC was given, though its timestamps read as 25 (3.1).
    cpu, _cuda, outputs, inbox = _commands(tmp_path)
    facts = {"level": 32, "r_frame_rate": "25/1", "avg_frame_rate": "25/1",
             **({"vui_frame_rate": vui} if vui else {})}
    _runs(monkeypatch, outputs, {"cpu": (0, "")}, facts={"cpu": {"v1.mkv": facts}})
    if why is None:
        ff.run_encode(cpu, outputs, log_label="i")
        assert sorted(p.name for p in inbox.iterdir()) == ["prepared.mkv", "v1.mkv"]
    else:
        with pytest.raises(TranscodeError, match=why):
            ff.run_encode(cpu, outputs, log_label="i")


# The level NVENC gives one encode of system memory frames (DPB 6, 4 refs),
# as the GPU host measured it: size, maxrate, codec -> ffprobe's level.
@pytest.mark.parametrize(("codec", "size", "maxrate", "level"), [
    ("hevc", (3840, 2160), 14_000_000, 150), ("hevc", (1920, 1080), 8_000_000, 120),
    ("hevc", (1280, 720), 8_000_000, 93), ("hevc", (854, 480), 8_000_000, 93),
    ("hevc", (640, 360), 8_000_000, 93), ("hevc", (854, 480), 1_200_000, 90),
    ("h264", (3840, 2160), 16_000_000, 51), ("h264", (1920, 1080), 6_000_000, 40),
    ("h264", (1280, 720), 3_000_000, 31), ("h264", (854, 480), 1_500_000, 30),
    ("h264", (640, 360), 800_000, 30),
    # About a level's MaxBR: HEVC's counted x1000 (3.1: 10 Mbit/s), H.264
    # High's x1250 at least (3.1: 17.5 Mbit/s; NVENC kept 3.1 up to 19).
    ("hevc", (1280, 720), 9_500_000, 93), ("hevc", (1280, 720), 10_500_000, 120),
    ("h264", (1280, 720), 15_000_000, 31),
])
def test_max_level_is_the_level_nvenc_picks(codec: str, size: tuple[int, int], maxrate: int,
                                            level: int) -> None:
    assert ff.max_level(codec, *size, 24000 / 1001, maxrate) == level


def test_max_level_rises_with_the_frame_rate_and_the_maxrate() -> None:
    assert ff.max_level("hevc", 3840, 2160, 60, 14_000_000) == 153
    assert ff.max_level("hevc", 1920, 1080, 24, 22_000_000) == 150
    assert ff.max_level("hevc", 1920, 1080, 24, 30_000_000) == 153
    assert ff.max_level("hevc", 16384, 16384, 24, 1) is None  # no level holds it


def test_the_facts_read_an_hevc_dpb_from_its_parameter_sets(monkeypatch) -> None:
    def run(args, **_kw):
        if args[0] == "ffprobe":
            result = _Result(0)
            result.stdout = json.dumps({"streams": [{"codec_name": "hevc", "level": 150,
                                                     "refs": 1}]})
            return result
        # Two sub-layers: the largest DPB. The VUI's timing: 24000/1001.
        trace = "[trace_headers @ 0x1] {}  sps_max_dec_pic_buffering_minus1[{}]  {} = {}\n"
        return _Result(0, trace.format(230, 0, "011", 3) + trace.format(245, 1, "00110", 5)
                       + "[trace_headers @ 0x1] 380  vui_num_units_in_tick  0001111101001 = 1001\n"
                         "[trace_headers @ 0x1] 412  vui_time_scale  0101110111000000 = 24000\n")
    monkeypatch.setattr(subprocess, "run", run)
    assert ff._stream_facts(Path("/x.mkv")) == {"codec_name": "hevc", "level": 150, "dpb": 6,
                                                "vui_frame_rate": "24000/1001"}


def _script(tmp_path: Path, body: str) -> str:
    """An executable that stands in for ffmpeg: it ignores its arguments."""
    path = tmp_path / "fake-ffmpeg"
    path.write_text(f"#!{sys.executable}\nimport sys, time\n{body}\n")
    path.chmod(0o755)
    return str(path)


def test_the_watchdog_kills_a_run_that_stands_still(tmp_path: Path) -> None:
    # Progress, then nothing: killed once it has stood still that long.
    fake = _script(tmp_path, "print('frame=1', flush=True)\nsys.stderr.write('started')\n"
                             "sys.stderr.flush()\ntime.sleep(60)")
    t0 = time.monotonic()
    result, stalled = ff._run_watched([fake, "-i", "x"], stall_seconds=1.5)
    assert stalled and result.returncode != 0 and time.monotonic() - t0 < 30
    assert result.stderr == "started"
    assert result.args[1:5] == ["-progress", "pipe:1", "-stats_period", "5"]


def test_the_watchdog_lets_a_run_that_progresses_finish(tmp_path: Path) -> None:
    fake = _script(tmp_path, "for i in range(4):\n    print(f'frame={i}', flush=True)\n"
                             "    time.sleep(0.5)\nsys.stderr.write('done')\nsys.exit(3)")
    result, stalled = ff._run_watched([fake], stall_seconds=1.5)
    assert (stalled, result.returncode, result.stderr) == (False, 3, "done")


# ------------------------------------------------------------- the worker
class Steps:
    def __init__(self) -> None:
        self.steps: list[tuple[str, dict]] = []

    def upsert_step(self, _id: str, status: str, **kw: object) -> None:
        self.steps.append((status, kw))


@pytest.fixture
def encodes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """The worker's probe answers `encodes.probe`; run_encode is faked,
    its calls kept, its results said to come from `encodes.decoder` (None:
    the run asked for)."""
    source = tmp_path / "source.mkv"
    source.write_bytes(b"\0")
    state = SimpleNamespace(path=source, probe=UHD_HDR10, decoder=None, calls=[])

    def encode(args, outputs, *, log_label, cpu_fallback=None):
        state.calls.append((args, cpu_fallback))
        decoder = state.decoder or ("cuda" if cpu_fallback is not None else "cpu")
        results = []
        for out in outputs:
            out.final.parent.mkdir(parents=True, exist_ok=True)
            out.final.write_bytes(b"\0" * 1000)
            results.append(RungResult(rung=out.rung, path=out.final, size_bytes=1000,
                                      width=out.rung.width, height=out.rung.height,
                                      decoder=decoder))
        return results, 0.1

    monkeypatch.setattr(worker, "ffprobe", lambda _p: {**state.probe, "start_time": 0.0,
                                                       "duration_ms": 10_000})
    monkeypatch.setattr(worker, "run_encode", encode)
    # NVDEC's first frame: the picture's own size, unless a test says not.
    state.frame = None
    monkeypatch.setattr(worker, "cuda_frame_blocker", lambda *_a: state.frame)
    return state


def _item(path: Path) -> ClaimedItem:
    return ClaimedItem(id="0f1e2d3c-0000-4000-8000-000000000001", type="movie", title="clip",
                       year=None, duration_ms=None, path=str(path))


def test_the_worker_decodes_on_the_gpu_with_the_cpu_run_in_reserve(encodes, tmp_path) -> None:
    steps = Steps()
    with capture_logs() as logs:
        ok = worker._process_one(_item(encodes.path), steps, tmp_path / "inbox",
                                 EncodeSettings(decoder="cuda"))
    assert ok and steps.steps[-1][0] == "done"
    [(args, fallback)] = encodes.calls
    assert args[6:6 + len(CUDA_DECODE_ARGS)] == CUDA_DECODE_ARGS
    assert fallback is not None and "-hwaccel" not in fallback
    assert " decode=cuda " in steps.steps[-1][1]["details"]
    [done] = [e for e in logs if e["event"] == "transcoder.item.done"]
    assert done["decode"] == "cuda"


def test_the_worker_says_the_cuda_run_failed(encodes, tmp_path) -> None:
    encodes.decoder = "cpu"
    steps = Steps()
    assert worker._process_one(_item(encodes.path), steps, tmp_path / "inbox",
                               EncodeSettings(decoder="cuda"))
    assert " decode=cpu(cuda failed) " in steps.steps[-1][1]["details"]


@pytest.mark.parametrize(("probe", "decoder", "said"), [
    (UHD_HDR10, "cpu", "cpu(cuda off)"),
    (_probe("hevc", 1920, 1080, "yuv422p10le", "Rext"), "cuda", "cpu(4:2:2)"),
    (_probe("h264", 1920, 1080, profile="High", field_order="tt"), "cuda", "cpu(interlaced)"),
    (_probe("mpeg2video", 720, 576, profile="Main"), "cuda", "cpu(codec=mpeg2video)"),
])
def test_the_worker_decodes_on_the_cpu_and_says_why(encodes, tmp_path, probe: dict,
                                                    decoder: str, said: str) -> None:
    encodes.probe = probe
    steps = Steps()
    assert worker._process_one(_item(encodes.path), steps, tmp_path / "inbox",
                               EncodeSettings(decoder=decoder))
    [(args, fallback)] = encodes.calls
    assert "-hwaccel" not in args and fallback is None
    assert f" decode={said} " in steps.steps[-1][1]["details"]


def test_the_worker_decodes_two_hevc_rungs_on_the_cpu(encodes, tmp_path) -> None:
    # NVENC_LEVEL_RACE: a ladder of two HEVC rungs keeps the CPU's decode.
    steps = Steps()
    settings = EncodeSettings(ladder=tuple(parse_ladder("source,1080p:hevc")), decoder="cuda")
    assert worker._process_one(_item(encodes.path), steps, tmp_path / "inbox", settings)
    [(args, fallback)] = encodes.calls
    assert "-hwaccel" not in args and fallback is None
    assert " decode=cpu(hevc_rungs=2) " in steps.steps[-1][1]["details"]


# ------------------------------------------------- NVDEC's first frame
def _pgm(width: int, height: int) -> bytes:
    return f"P5\n{width} {height}\n255\n".encode() + b"\0" * 16


@pytest.mark.parametrize(("probe", "stdout", "rc", "blocker"), [
    (H264, _pgm(1920, 1080), 0, None),
    # A top crop (16 lines): NVDEC hands the frame on uncropped there.
    (_probe("h264", 1920, 1064, profile="High"), _pgm(1920, 1080), 0, "nvdec_frame=1920x1080"),
    # A left crop (64 columns).
    (_probe("h264", 1856, 1080, profile="High"), _pgm(1920, 1080), 0, "nvdec_frame=1920x1080"),
    (UHD_HDR10, b"", 1, "nvdec_preflight_failed"),
    (UHD_HDR10, b"not a picture", 0, "nvdec_preflight_failed"),
])
def test_nvdecs_first_frame_must_be_the_picture(monkeypatch, probe: dict, stdout: bytes, rc: int,
                                                blocker: str | None) -> None:
    calls: list[list[str]] = []

    def run(args, **_kw):
        calls.append(args)
        result = _Result(rc)
        result.stdout = stdout
        return result

    monkeypatch.setattr(subprocess, "run", run)
    plan = _plan(probe)
    assert ff.cuda_frame_blocker(SRC, plan, 3) == blocker
    [args] = calls
    assert args[args.index("-init_hw_device"):args.index("-i")] == CUDA_DECODE_ARGS
    fmt = "p010le" if plan.source.bit_depth == 10 else "nv12"
    assert args[args.index("-map") + 1] == "0:3" and args[args.index("-frames:v") + 1] == "1"
    assert args[args.index("-vf") + 1] == f"hwdownload,format={fmt},format=gray"


def test_a_preflight_that_hangs_leaves_the_source_to_the_cpu(monkeypatch) -> None:
    def hang(args, **kw):
        raise subprocess.TimeoutExpired(args, kw.get("timeout"))

    monkeypatch.setattr(subprocess, "run", hang)
    assert ff.cuda_frame_blocker(SRC, _plan(H264), None) == "nvdec_preflight_failed"


def test_the_worker_decodes_a_cropped_source_on_the_cpu(encodes, tmp_path) -> None:
    encodes.frame = "nvdec_frame=1920x1080"
    steps = Steps()
    assert worker._process_one(_item(encodes.path), steps, tmp_path / "inbox",
                               EncodeSettings(decoder="cuda"))
    [(args, fallback)] = encodes.calls
    assert "-hwaccel" not in args and fallback is None
    assert " decode=cpu(nvdec_frame=1920x1080) " in steps.steps[-1][1]["details"]


def test_the_facts_read_h264_reference_frames_from_its_parameter_sets(monkeypatch) -> None:
    # ffprobe's refs says 1 for NVENC's 4: the SPS's max_num_ref_frames.
    def run(args, **_kw):
        if args[0] == "ffprobe":
            result = _Result(0)
            result.stdout = json.dumps({"streams": [{"codec_name": "h264", "level": 31,
                                                     "refs": 1}]})
            return result
        # H.264's time_scale counts fields: 48000 over 1001 ticks is 23.976 frames.
        return _Result(0, "[trace_headers @ 0x1] 64  max_num_ref_frames  00101 = 4\n"
                          "[trace_headers @ 0x1] 70  gaps_in_frame_num_allowed_flag  0 = 0\n"
                          "[trace_headers @ 0x1] 120  num_units_in_tick  0001111101001 = 1001\n"
                          "[trace_headers @ 0x1] 152  time_scale  01011101110000000 = 48000\n")
    monkeypatch.setattr(subprocess, "run", run)
    assert ff._stream_facts(Path("/x.mkv")) == {"codec_name": "h264", "level": 31, "refs": 4,
                                                "vui_frame_rate": "24000/1001"}
