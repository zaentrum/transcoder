"""ffprobe + ffmpeg invocations.

Responsibilities:
  1. `ffprobe()` — parse the source into a small dict (container,
     duration_ms, video, audio[], subtitles[]) used by the decision
     layer and by the encode pipeline.
  2. `detect_encoders()` — find out at startup whether NVENC actually
     works on this host (a GPU-less box has the encoder compiled in but
     no driver), and fall back to libx265 / libx264 per codec.
     `detect_decoder()` — find out at startup whether NVDEC decodes into
     CUDA frames that scale_cuda and NVENC take (README "GPU decoding").
  3. `vulkan_preflight()` — find out at startup whether ffmpeg can open a
     Vulkan device and run libplacebo on it, which a Dolby Vision 5
     source is reshaped with; it logs one line naming the Vulkan loader.
  4. `build_encode_command()` + `run_encode()` — ONE ffmpeg invocation
     per item: decode the source once (by NVDEC into GPU memory where it
     can, `decoder="cuda"`, else on the CPU, which a failed CUDA run falls
     back to), `split` it into one branch per encoded rung of the plan,
     and write one intermediate MKV per rung:
        - v0 (`prepared.mkv`, the top rung) carries every audio track
          and every Matroska-copyable subtitle track verbatim (the
          packager re-encodes audio downstream — doing it here would
          waste GPU time on a CPU-only operation), including image
          formats (PGS, VobSub) and rich text (ASS, SSA) that MP4 can't
          carry — the whole reason we use MKV for the intermediate —
          and MP4's own text captions (`mov_text`) written as SubRip;
        - v1..vN are video-only.
     Keyframes are forced identically on every encoded rung (see
     `_keyframe_args`) so the packager cuts all rungs at the same
     instants. A Dolby Vision 5 source is reshaped to HDR10 once, before
     the split (DOVI_RESHAPE_CHAIN).

The atomic-rename pattern matters: every output is written to
`<name>.partial` first and `os.replace`d into place only after ffmpeg
exits 0. The packager's worker treats the existence of `prepared.mkv`
(and of `renditions.json`, written last by the worker) as proof of a
complete handoff.
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import re
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import structlog

from .decision import (
    CPU_ENCODERS,
    DEFAULT_CAPS,
    NO_VULKAN,
    NVENC_ENCODERS,
    Caps,
    Encoders,
    Plan,
    RungPlan,
    RungSpec,
    SourceInfo,
    VulkanDevice,
    parse_ladder,
    plan_renditions,
)

log = structlog.get_logger(__name__)


# Subtitle codecs Matroska can carry losslessly via stream-copy. We
# stream-copy these into prepared.mkv so the packager downstream can
# extract them. The list is the intersection of "supported by ffmpeg's
# matroska muxer" and "documented in the Matroska spec" — any other
# source codec gets dropped from the intermediate, but for the text
# codecs below.
MATROSKA_SUBTITLE_COPY_CODECS = {
    "subrip", "srt", "ass", "ssa", "webvtt",
    "hdmv_pgs_subtitle", "dvb_subtitle", "dvd_subtitle",
    "microdvd",
}

# Text subtitle codecs Matroska cannot carry as they are, written into
# prepared.mkv as SubRip instead: `mov_text`, the MP4-native captions
# codec. Dropped, an MP4 whose video is encoded here was packaged without
# any of its subtitles (the packager reads them from prepared.mkv); text
# to text costs no GPU and next to no CPU, and the packager writes WebVTT
# from either.
MATROSKA_SUBTITLE_AS_SUBRIP_CODECS = {"mov_text"}

# HDR (PQ/HLG) -> SDR BT.709 for H.264 rungs: linearise, hable tone-map,
# back to BT.709 limited range. Same chain chino-stream's on-demand CPU
# path uses, so a packaged 720p looks like a live-transcoded one.
TONEMAP_CHAIN = (
    "zscale=transfer=linear:npl=100,format=gbrpf32le,"
    "tonemap=tonemap=hable:desat=0,"
    "zscale=primaries=709:transfer=709:matrix=709:range=tv,format=yuv420p"
)

# Dolby Vision 5 -> HDR10 (decision.DOVI_RESHAPED_PROFILES), once per
# source, before the split. ffmpeg's HEVC decoder hands each frame on
# with its RPU (frame side data "Dolby Vision Metadata"); the frames go up
# to the Vulkan device (in ffmpeg 7.1 hwupload takes no CUDA frames, so
# they come from system memory), libplacebo applies the RPU and renders
# PQ, BT.2020, limited range, and they come back down, still 10-bit, in
# `{fmt}`: what the top rung's encoder takes.
DOVI_RESHAPE_CHAIN = (
    "format=yuv420p10le,hwupload,"
    "libplacebo=apply_dolbyvision=1:colorspace=bt2020nc:color_primaries=bt2020"
    ":color_trc=smpte2084:range=tv:format={fmt},"
    "hwdownload,format={fmt}"
)
# The colour tags every rung of a reshaped source must come out with: the
# HDR10 rungs', and the tone-mapped H.264 rungs' (TONEMAP_CHAIN), limited
# range both (the source is full range). Checked on the outputs before
# they are handed off.
HDR10_TAGS = {"color_primaries": "bt2020", "color_transfer": "smpte2084",
              "color_space": "bt2020nc", "color_range": "tv"}
SDR_TAGS = {"color_primaries": "bt709", "color_transfer": "bt709", "color_space": "bt709",
            "color_range": "tv"}
# The frame side data an RPU is exported as, and how many of the first
# frames must carry it before a Dolby Vision 5 source is encoded.
DOVI_FRAME_METADATA = "Dolby Vision Metadata"
DOVI_CHECK_FRAMES = 8
# What ffmpeg's HEVC decoder logs when it cannot read a frame's RPU. That
# frame, and those up to the next RPU it reads, carry no Dolby Vision
# metadata: libplacebo would leave them as they are, in the wrong colours.
DOVI_RPU_ERROR = "Error parsing DOVI NAL unit"

# Dolby Vision 7 to its HDR10 base layer, on a copy the transcoder writes
# (decision.DOVI_BASE_LAYER_PROFILES): filter_units drops the enhancement
# layer's and the RPU's NAL units (types 63 and 62) and leaves every other
# unit as it is, the base layer's SEI (its HDR10 static metadata) among
# them; dovi_rpu=strip=1 drops the configuration record, so the copy
# claims no Dolby Vision.
DOVI_EL_RPU_NAL_TYPES = (62, 63)
DOVI_STRIP_BSF = "filter_units=remove_types=62|63,dovi_rpu=strip=1"

# The startup preflight: one generated frame through the reshaping chain.
VULKAN_PREFLIGHT_SOURCE = "color=c=black:s=256x144:r=25:d=0.2"
# ffmpeg's Vulkan device choice, at -v verbose: "Device 0 selected:
# NVIDIA GeForce RTX 3090 (discrete) (0x2204)".
_VULKAN_SELECTED = re.compile(r"Device (\d+) selected: (.+) \((\w+)\) \(0x[0-9a-fA-F]+\)")
# The lines of a failed preflight that say why (the loader's own, with
# VK_LOADER_DEBUG, and ffmpeg's), minus ffmpeg's generic option errors.
_VULKAN_WHY = re.compile(r"error|fail|unable|could not|not found|incompatible", re.IGNORECASE)
_VULKAN_NOISE = ("Failed to set value", "Error parsing global options", "Device creation failed")
_LOG_PREFIX = re.compile(r"^\[[^\]]+ @ 0x[0-9a-fA-F]+\] ")

# With source-aligned keyframes the encoders must not add keyframes of
# their own — one landing early in a segment window would move that
# rung's cut. The GOP ceiling only matters for sources with GOPs longer
# than this many seconds.
SOURCE_ALIGNED_MAX_GOP_SECONDS = 60

# GPU decoding (README "GPU decoding"): NVDEC, through ffmpeg's cuda
# hwaccel, hands its frames on in GPU memory, as CUDA frames, and the
# filters (scale_cuda) and NVENC take them there; no picture crosses to the
# CPU. With -hwaccel cuda ffmpeg picks the decoder that has it (its own av1
# decoder rather than libdav1d). One CUDA device: a second beside the
# decoder's, so that the encoders of system memory frames open contexts of
# their own, kept none of NVENC_LEVEL_RACE away (_nvenc_failure catches
# it) and cost every run about 0.8 GB of GPU memory and 0.2 s to start. The
# frames' layout, by the source's bits, is a 4:2:0 one NVENC takes as it is:
CUDA_DECODE_ARGS = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
CUDA_FRAME_FORMATS = {8: "nv12", 10: "p010le"}
# A rung that needs neither a scale nor another layout gets the frames
# through `format=cuda`, which converts nothing and takes CUDA frames only:
# frames the decoder handed on in system memory after all (NVDEC turned
# the source down) fail the run at its first frame, and run_encode makes
# it again on the CPU, rather than encoding them unnoticed.
CUDA_GUARD = "format=cuda"
# The planar twin of each, which the CPU's tone-mapping (zscale) is fed, as
# the CPU path feeds it straight from the software decoder.
PLANAR_FORMATS = {"nv12": "yuv420p", "p010le": "yuv420p10le"}
# How scale_cuda scales. Its bicubic and lanczos read a fixed 4x4 texels
# whatever the ratio, so a downscale skips detail and aliases; its bilinear
# widens with the ratio, up to 3x (taps {w, 1, w}). A rung further down than
# 3x is scaled in halves until it is within 3x of its size
# (_cuda_scale_steps). It stays softer than the CPU path's swscale bicubic:
# against a lanczos reference (a test pattern with grain; PSNR Y/U/V) 4K to
# 1080p 47.5/46.1/42.3 dB where the CPU path's is 59.3/55.4/51.4, 1080p to
# 720p 48.7/47.8/41.4 (CPU 56.9/55.9/50.0); bicubic, lanczos, or halves to
# 2x were no closer, and scale_cuda's lanczos shifts the chroma (U 26 dB).
CUDA_SCALE_ALGO = "bilinear"
CUDA_SCALE_MAX_RATIO = 3
# The startup probe's clip, encoded with hevc_nvenc as HEVC Main 10 (NVDEC
# decodes no HEVC under 144 lines).
CUDA_PROBE_SOURCE = "testsrc2=s=320x180:r=25:d=0.2"
# An encode whose progress (frames, bytes written, time encoded) stands
# still this long is taken for hung (in NVDEC, NVENC, CUDA or anywhere) and
# killed (_run_watched): a GPU run is made again on the CPU, any other run
# fails its step. Its progress comes every PROGRESS_SECONDS. One that has
# not died KILL_WAIT_SECONDS after its kill (a call into the driver that
# does not return) fails its step and leaves the pod unhealthy
# (unkillable(), /healthz), for the kubelet to restart it.
STALL_SECONDS = 600
PROGRESS_SECONDS = 5
KILL_WAIT_SECONDS = 30
# The ffmpeg runs that had not died after their kill (_run_watched).
_UNKILLED: list[str] = []
# What ffmpeg writes when a frame was not decoded (the CLI's "Decoding
# error", a hwaccel that failed a picture, a CUDA call that failed): a GPU
# run that writes any of it, whatever its exit code, is made again on the
# CPU, which conceals what it cannot decode as it always has. Matched
# without regard to case.
GPU_DECODE_ERRORS = ("decoding error", "error while decoding", "hardware accelerator failed",
                     "error submitting packet to decoder", "cuda_error")

# What every NVENC output is checked against (_nvenc_failure), on either
# path, before it is handed off: the profile it was asked for, a DPB no
# larger than NVENC's own for system memory frames at any preset (HEVC 6
# pictures, sps_max_dec_pic_buffering_minus1 5 (p1: 4); H.264 4 reference
# frames (p1: 3)), and a level no higher than the lowest the spec allows for
# its coded size, frame rate (the highest of its average, its base rate and
# the one its VUI names, which NVENC was given) and VBV maxrate with that
# DPB, the level NVENC picks itself (the VBV buffer, twice the maxrate,
# does not raise it). Six pictures fit any HEVC level at any size it
# allows, so an HEVC level above it is NVENC_LEVEL_RACE's.
NVENC_MAX_HEVC_DPB = 6
NVENC_MAX_H264_REFS = 4
# general_level_idc, MaxLumaPs, MaxLumaSr, MaxBR (Main tier, kbit/s): H.265
# table A.8 / A.9.
HEVC_LEVELS = (
    (30, 36_864, 552_960, 128), (60, 122_880, 3_686_400, 1_500),
    (63, 245_760, 7_372_800, 3_000), (90, 552_960, 16_588_800, 6_000),
    (93, 983_040, 33_177_600, 10_000), (120, 2_228_224, 66_846_720, 12_000),
    (123, 2_228_224, 133_693_440, 20_000), (150, 8_912_896, 267_386_880, 25_000),
    (153, 8_912_896, 534_773_760, 40_000), (156, 8_912_896, 1_069_547_520, 60_000),
    (180, 35_651_584, 1_069_547_520, 60_000), (183, 35_651_584, 2_139_095_040, 120_000),
    (186, 35_651_584, 4_278_190_080, 240_000),
)
# level_idc, MaxMBPS, MaxFS, MaxDpbMbs, MaxBR (kbit/s, x1.25 for High): H.264
# table A-1.
H264_LEVELS = (
    (10, 1_485, 99, 396, 64), (11, 3_000, 396, 900, 192), (12, 6_000, 396, 2_376, 384),
    (13, 11_880, 396, 2_376, 768), (20, 11_880, 396, 2_376, 2_000),
    (21, 19_800, 792, 4_752, 4_000), (22, 20_250, 1_620, 8_100, 4_000),
    (30, 40_500, 1_620, 8_100, 10_000), (31, 108_000, 3_600, 18_000, 14_000),
    (32, 216_000, 5_120, 20_480, 20_000), (40, 245_760, 8_192, 32_768, 20_000),
    (41, 245_760, 8_192, 32_768, 50_000), (42, 522_240, 8_704, 34_816, 50_000),
    (50, 589_824, 22_080, 110_400, 135_000), (51, 983_040, 36_864, 184_320, 240_000),
    (52, 2_073_600, 36_864, 184_320, 240_000), (60, 4_177_920, 139_264, 696_320, 240_000),
    (61, 8_355_840, 139_264, 696_320, 480_000), (62, 16_711_680, 139_264, 696_320, 800_000),
)


class TranscodeError(RuntimeError):
    """Raised when ffprobe/ffmpeg fail. Message is captured into the
    transcode step's `error` column for operator triage."""


def ffprobe(path: Path) -> dict[str, Any]:
    """Run ffprobe and return a normalised payload:

        {
          "container": "matroska,webm",
          "duration_ms": 5400000,
          "video": {"codec_name": "h264", "width": 1920, "height": 1080,
                    "bit_rate": "8000000", ...},
          "video_index": 0,
          "audio": [{"codec_name": "ac3", "channels": 6, ...}, ...],
          "subtitles": [{"codec_name": "hdmv_pgs_subtitle", ...}, ...],
        }

    `video` is the first video stream that is NOT an attached picture
    (cover art some MP4/MKV rips carry as a "video" stream), and
    `video_index` its absolute stream index, so the encode maps the
    real picture even when the poster comes first. `videos` are all the
    video streams but cover art (the cap's bucket is the tallest's and the
    widest's),
    and `size_bytes` the file's size (the cap's size rule, its bit rate
    fallback, and the check of a statistics tag against the file).

    We pass exactly the same flags as packager/_ffprobe so any future
    debugging that compares the two services' probes is reading the
    same raw payload.
    """
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        raise TranscodeError(
            f"ffprobe exited {result.returncode}: "
            f"{(result.stderr or '').strip()[:500] or '(no stderr)'}"
        )
    raw = json.loads(result.stdout or "{}")
    fmt = raw.get("format", {})
    try:
        duration_ms = int(float(fmt.get("duration", "0")) * 1000)
    except (TypeError, ValueError):
        duration_ms = 0
    streams = raw.get("streams", [])
    videos = [s for s in streams if s.get("codec_type") == "video"]
    pictures = [s for s in videos if not (s.get("disposition") or {}).get("attached_pic")]
    main_video = next(iter(pictures), videos[0] if videos else {})
    try:
        size_bytes = int(fmt.get("size")) if fmt.get("size") else None
    except (TypeError, ValueError):
        size_bytes = None
    try:
        start_time = float(fmt.get("start_time") or 0.0)
    except (TypeError, ValueError):
        start_time = 0.0
    return {
        "container": fmt.get("format_name", ""),
        "duration_ms": duration_ms,
        # Earliest timestamp over all streams. ffmpeg shifts every input
        # timestamp by -start_time before encoding; the contract records
        # that shift so the packager can put the original on the same
        # timeline (renditions.py, "timestampOffset").
        "start_time": start_time,
        "bit_rate": fmt.get("bit_rate"),
        "size_bytes": size_bytes,
        "video": main_video,
        "videos": pictures,
        "video_index": main_video.get("index") if main_video else None,
        "audio": [s for s in streams if s.get("codec_type") == "audio"],
        "subtitles": [s for s in streams if s.get("codec_type") == "subtitle"],
    }


@dataclass(frozen=True)
class EncodeProfile:
    """Resolved per-resolution rate-control numbers + identifying
    label. The label goes into the step `details` column so the
    Processing tile can show 'nvenc-1080p' vs 'nvenc-2160p' without
    parsing the maxrate string."""
    label: str
    maxrate_mbps: int


def pick_profile(
    width: int,
    height: int,
    maxrate_1080p_mbps: int,
    maxrate_2160p_mbps: int,
) -> EncodeProfile:
    """Map the source resolution onto one of two profiles. Anything
    wider than 1920 counts as UHD; anything else uses the HD/SD cap.
    This two-bucket split is a common shape for hardware-accelerated
    transcoders — narrower buckets risk visibly dropping bitrate on,
    say, 1440p uploads that aren't quite 4K.

    The worker no longer uses it: an encode's maxrate and its step label
    follow the cap's bucket, 2160 from 2000 lines or 3200 wide
    (decision.cap_bucket)."""
    if max(width, height) > 1920:
        return EncodeProfile(label="nvenc-2160p", maxrate_mbps=maxrate_2160p_mbps)
    return EncodeProfile(label="nvenc-1080p", maxrate_mbps=maxrate_1080p_mbps)


# ------------------------------------------------------------- settings
@dataclass(frozen=True)
class EncodeSettings:
    """Everything the encode path needs, resolved once at startup."""
    ladder: tuple[RungSpec, ...] = field(default_factory=lambda: tuple(parse_ladder("")))
    encoders: Encoders = NVENC_ENCODERS
    # Forced keyframe interval == the packager's HLS segment duration.
    segment_seconds: int = 6
    nvenc_preset: str = "p5"
    nvenc_cq: int = 23
    # NVENC_MAXRATE_1080P / 2160P: overrides of a source-size HEVC encode's
    # maxrate, by bucket; None leaves it to the caps.
    maxrate_1080p_mbps: int | None = None
    maxrate_2160p_mbps: int | None = None
    caps: Caps = DEFAULT_CAPS
    x264_preset: str = "medium"
    x264_crf: int = 23
    x265_preset: str = "medium"
    x265_crf: int = 24
    # The Vulkan device a Dolby Vision 5 source is reshaped on (the
    # startup preflight's); none: such a source fails its step.
    vulkan: VulkanDevice = NO_VULKAN
    # Where a source is decoded: "cuda" (NVDEC into GPU memory, for the
    # sources and rungs it serves: cuda_decode_blocker) when the startup
    # probe found it working (detect_decoder), else "cpu", as before.
    decoder: str = "cpu"


# ------------------------------------------------------- NVENC detection
def _nvenc_works(encoder: str, ffmpeg_bin: str = "ffmpeg") -> tuple[bool, str]:
    """Encode one black frame with `encoder`. The BtbN build always has
    NVENC compiled in, so `-encoders` can't tell; only opening the
    encoder does ("Cannot load libcuda.so.1" without a driver, "No
    capable devices found" without a GPU). ~1 s, startup only."""
    try:
        result = subprocess.run(
            [
                ffmpeg_bin, "-nostdin", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=c=black:s=256x144:r=25:d=0.2",
                "-frames:v", "1", "-c:v", encoder, "-f", "null", "-",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, str(e)[:300]
    return result.returncode == 0, (result.stderr or "").strip()[-300:]


def detect_encoders(mode: str = "auto", ffmpeg_bin: str = "ffmpeg") -> Encoders:
    """Resolve the ENCODER setting into concrete encoders.

      * `cpu`   — libx265 / libx264, no probe.
      * `nvenc` — NVENC for both codecs; raises if the probe fails, so a
                  host that was promised a GPU fails loudly at startup.
      * `auto`  — probe each NVENC encoder, fall back to the CPU encoder
                  per codec (the default; a GPU-less box just works).
    """
    mode = (mode or "auto").strip().lower()
    if mode == "cpu":
        return CPU_ENCODERS
    if mode not in ("auto", "nvenc"):
        raise TranscodeError(f"ENCODER must be auto, nvenc or cpu (got {mode!r})")
    probes = {enc: _nvenc_works(enc, ffmpeg_bin) for enc in ("hevc_nvenc", "h264_nvenc")}
    for enc, (ok, err) in probes.items():
        log.info("transcoder.encoder.probe", encoder=enc, ok=ok, error=None if ok else err)
    if mode == "nvenc":
        failed = [enc for enc, (ok, _) in probes.items() if not ok]
        if failed:
            raise TranscodeError(
                f"ENCODER=nvenc but {', '.join(failed)} failed to open: "
                f"{probes[failed[0]][1]}"
            )
        return NVENC_ENCODERS
    return Encoders(
        hevc="hevc_nvenc" if probes["hevc_nvenc"][0] else CPU_ENCODERS.hevc,
        h264="h264_nvenc" if probes["h264_nvenc"][0] else CPU_ENCODERS.h264,
    )


# ------------------------------------------------------- NVDEC detection
def _cuda_decode_works(ffmpeg_bin: str = "ffmpeg") -> tuple[bool, str]:
    """Decode a few frames of HEVC Main 10, which hevc_nvenc writes first,
    with NVDEC into CUDA frames, take them to 8 bits with scale_cuda and
    encode them with hevc_nvenc: the pieces a CUDA-decoded encode runs on,
    once (the driver's NVDEC, the CUDA kernels of the filters, NVENC from
    GPU memory). Frames that came out of the decoder in system memory fail
    the scale_cuda. About 1 s, startup only, and once more the 5 s in which
    the driver compiles scale_cuda's kernels from their PTX when its cache
    (CUDA_CACHE_PATH, which the image sets) has none of them yet: the first
    encode that scales then finds them there."""
    with tempfile.TemporaryDirectory(prefix="transcoder-cuda-") as tmp:
        clip = str(Path(tmp) / "probe.mkv")
        runs = [
            ["-f", "lavfi", "-i", CUDA_PROBE_SOURCE, "-c:v", "hevc_nvenc",
             "-profile:v", "main10", "-pix_fmt", "p010le", clip],
            [*CUDA_DECODE_ARGS, "-i", clip, "-filter_complex", "[0:v]scale_cuda=format=nv12[o]",
             "-map", "[o]", "-c:v", "hevc_nvenc", "-f", "null", "-"],
        ]
        for args in runs:
            try:
                result = subprocess.run(
                    [ffmpeg_bin, "-nostdin", "-hide_banner", "-loglevel", "error", *args],
                    capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL,
                )
            except (OSError, subprocess.TimeoutExpired) as e:
                return False, str(e)[:300]
            if result.returncode != 0:
                return False, ((result.stderr or "").strip()[-300:]
                               or f"ffmpeg exited {result.returncode}")
    return True, ""


def detect_decoder(mode: str = "auto", encoders: Encoders = NVENC_ENCODERS,
                   ffmpeg_bin: str = "ffmpeg") -> str:
    """Resolve the DECODER setting into where the sources are decoded, and
    log it in one line (`transcoder.decoder.probe`):

      * `cpu`  — ffmpeg's software decoders, no probe (as before).
      * `auto` — "cuda" when HEVC is encoded with NVENC and the probe
                 (_cuda_decode_works) decodes on the GPU, else "cpu" (the
                 default; a host without a GPU just works).

    "cuda" decodes only the sources and rungs NVDEC serves
    (cuda_decode_blocker), and a CUDA run that fails is run again on the
    CPU (run_encode)."""
    mode = (mode or "auto").strip().lower()
    if mode not in ("auto", "cpu"):
        raise TranscodeError(f"DECODER must be auto or cpu (got {mode!r})")
    # ok: the probe's verdict, None when none was run.
    ok: bool | None = None
    error = ""
    if mode == "cpu":
        reason: str | None = "DECODER=cpu"
    elif encoders.hevc != "hevc_nvenc":
        reason = f"HEVC is encoded with {encoders.hevc}"
    else:
        ok, error = _cuda_decode_works(ffmpeg_bin)
        reason = None if ok else "the probe failed"
    decoder = "cuda" if ok else "cpu"
    emit = log.warning if ok is False else log.info
    emit("transcoder.decoder.probe", decoder=decoder, ok=ok, reason=reason, error=error or None)
    return decoder


# ------------------------------------------------------- Vulkan preflight
def vulkan_loader_version() -> str | None:
    """The Vulkan loader's version ("1.3.204"), as the loader reports it
    (vkEnumerateInstanceVersion), or None when there is no loader. Only
    the loader is asked: it loads no driver for this call."""
    try:
        loader = ctypes.CDLL("libvulkan.so.1")
    except OSError:
        return None
    query = getattr(loader, "vkEnumerateInstanceVersion", None)
    if query is None:
        return "1.0"  # a Vulkan 1.0 loader has no version query
    version = ctypes.c_uint32(0)
    if query(ctypes.byref(version)) != 0:
        return None
    v = version.value
    return f"{v >> 22 & 0x7F}.{v >> 12 & 0x3FF}.{v & 0xFFF}"


def _vulkan_why(stderr: str) -> str:
    """Why a preflight run failed, from its stderr: the loader's messages
    and ffmpeg's errors, last four, without ffmpeg's generic option-parsing
    lines."""
    lines: list[str] = []
    for raw in (stderr or "").splitlines():
        line = _LOG_PREFIX.sub("", raw.strip())
        if (line and _VULKAN_WHY.search(line) and not line.startswith(_VULKAN_NOISE)
                and line not in lines):
            lines.append(line)
    return " | ".join(lines[-4:])[:400] or (stderr or "").strip()[-300:] or "(no stderr)"


def vulkan_preflight(ffmpeg_bin: str = "ffmpeg", *, allow_software: bool = False) -> VulkanDevice:
    """Find the Vulkan device a Dolby Vision 5 source is reshaped on, once
    at startup, and log it in one line (`transcoder.vulkan.preflight`,
    with the loader's version): one generated frame through the very
    chain an encode runs (DOVI_RESHAPE_CHAIN), so the loader, a driver,
    the device, the upload and download and libplacebo all have to work.
    The loader's own complaints (VK_LOADER_DEBUG=error,warn) end up in
    the reason when they don't.

    A software device (Mesa's lavapipe, say) is no device here unless
    `allow_software`: the reshaping runs on a GPU. Without a device, a
    Dolby Vision 5 source fails its transcode step (fail closed), with
    this reason."""
    loader = vulkan_loader_version()
    args = [
        ffmpeg_bin, "-nostdin", "-hide_banner", "-v", "verbose",
        "-init_hw_device", "vulkan=vk", "-filter_hw_device", "vk",
        "-f", "lavfi", "-i", VULKAN_PREFLIGHT_SOURCE, "-frames:v", "1",
        "-vf", DOVI_RESHAPE_CHAIN.format(fmt="yuv420p10le"), "-f", "null", "-",
    ]
    index: str | None = None
    name = why = ""
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL,
            env={**os.environ, "VK_LOADER_DEBUG": "error,warn"},
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        why = str(e)[:300]
    else:
        selected = _VULKAN_SELECTED.search(result.stderr or "")
        if selected:
            name = f"{selected.group(2)} ({selected.group(3)})"
        if result.returncode != 0:
            why = _vulkan_why(result.stderr)
        elif selected and selected.group(3) == "software" and not allow_software:
            why = f"the only Vulkan device is a software one, {name}"
        else:
            index = selected.group(1) if selected else ""
    if index is None:
        why = (f"Vulkan loader {loader}: {why}" if loader
               else f"no Vulkan loader (libvulkan.so.1): {why}")
    device = VulkanDevice(index=index, name=name, loader=loader, error=why)
    emit = log.info if device.ok else log.warning
    emit(
        "transcoder.vulkan.preflight",
        ok=device.ok,
        loader=loader,
        device=name or None,
        error=why or None,
        dolby_vision_5="reshaped to HDR10" if device.ok else "refused: its step fails",
    )
    return device


# ---------------------------------------------------------- the command
def _rate(bps: int) -> str:
    """8_000_000 -> '8M', 1_500_000 -> '1500k' (ffmpeg's suffixes)."""
    if bps % 1_000_000 == 0:
        return f"{bps // 1_000_000}M"
    return f"{max(1, bps // 1000)}k"


def _subtitle_maps(
    probe_subtitles: list[dict[str, Any]] | None,
) -> tuple[list[str], list[str], list[str]]:
    """Map the subtitle streams Matroska can carry: (maps, codecs,
    dropped). Those it can stream-copy are copied; MP4's text captions
    (MATROSKA_SUBTITLE_AS_SUBRIP_CODECS) are written as SubRip, each
    named by its place among the output's subtitle streams (the last
    matching `-c` wins); any other is left out and named in `dropped`.
    The default `-map 0:s?` would include every subtitle stream and then
    the muxer rejects one it cannot carry. Without a probe, fall back to
    the optional glob (only useful when the caller knows the source is
    sub-clean)."""
    if probe_subtitles is None:
        return ["-map", "0:s?"], ["-c:s", "copy"], []
    maps: list[str] = []
    codecs: list[str] = ["-c:s", "copy"]
    dropped: list[str] = []
    out = 0
    for i, s in enumerate(probe_subtitles):
        codec = (s.get("codec_name") or "").lower()
        if codec in MATROSKA_SUBTITLE_AS_SUBRIP_CODECS:
            codecs.extend([f"-c:s:{out}", "srt"])
        elif codec not in MATROSKA_SUBTITLE_COPY_CODECS:
            dropped.append(codec or "unknown")
            continue
        maps.extend(["-map", f"0:s:{i}"])
        out += 1
    return maps, codecs, dropped


def _filters(rung: RungPlan) -> list[str]:
    """Per-rung filter chain: fit into the rung's box, then either the
    HDR->SDR tone-map, a 10-bit 4:2:0 format (HEVC rungs of an HDR source
    or of one above 8 bits), or plain 8-bit 4:2:0. The format filters
    keep the frames' colour tags, which the encoder then writes."""
    chain: list[str] = []
    if rung.scaled:
        # The planner already fitted the source's DISPLAY size into the
        # rung's box; scale to exactly that with square pixels. (Letting
        # scale keep the aspect itself leaves a near-1 SAR such as
        # 1280:1281 on 854x480, and an anamorphic DVD's 64:45 SAR on
        # every rung.)
        chain.append(f"scale={rung.width}:{rung.height},setsar=1")
    if rung.tonemap:
        chain.append(TONEMAP_CHAIN)
    elif rung.ten_bit:
        chain.append("format=p010le" if rung.encoder.endswith("_nvenc") else "format=yuv420p10le")
    else:
        chain.append("format=yuv420p")
    return chain


def _cuda_scale_steps(width: int, height: int, to_w: int, to_h: int) -> list[tuple[int, int]]:
    """The sizes scale_cuda takes a picture through, from width x height to
    to_w x to_h: halves (even) while the picture is more than
    CUDA_SCALE_MAX_RATIO times the target either way, then the target.
    3840x2160 to 854x480: 1920x1080, 854x480; to 1280x720: one step."""
    steps: list[tuple[int, int]] = []
    while width > CUDA_SCALE_MAX_RATIO * to_w or height > CUDA_SCALE_MAX_RATIO * to_h:
        width, height = max(2, width // 4 * 2), max(2, height // 4 * 2)
        steps.append((width, height))
    return [*steps, (to_w, to_h)]


def _down_on_the_cpu(rung: RungPlan, src: SourceInfo) -> bool:
    """Whether the rung's frames must come down to the CPU for what the CPU
    path does to them: the tone-map, and an 8-bit rung of a source above 8
    bits, which swscale takes down to 8 bits with its dither, where
    scale_cuda would cut the two bits off (a ramp bands, and the picture
    darkens by half a step on average)."""
    return rung.tonemap or (not rung.ten_bit and src.bit_depth > 8)


def _cuda_filters(rung: RungPlan, src: SourceInfo, *, cuda_frames: bool) -> list[str]:
    """_filters for CUDA frames, the source's as NVDEC hands them on: the
    same steps on the GPU. scale_cuda fits the frames into the rung's box
    (square pixels, as `scale`; CUDA_SCALE_ALGO, in steps beyond 3x) and
    leaves them in the 4:2:0 layout the rung's NVENC encode takes: p010le
    for a 10-bit rung, nv12 for an 8-bit one of an 8-bit source; in the
    decoder's own layout for a rung that comes down for the CPU's part
    (_down_on_the_cpu).

    `cuda_frames`: the rung's encoder takes them there, as they are (the
    one rung of a run that does, _cuda_fed): a source-size rung in the
    decoder's own layout is CUDA_GUARD alone, one in another layout (an
    8-bit HDR source's Main 10, say) `scale_cuda=format=<it>`. Else they
    come down (hwdownload) in the format the CPU path gives the encoder
    (8-bit: yuv420p, which swscale makes, with its dither from 10 bits). A
    tone-mapped rung comes down too, in the decoder's layout, at its own
    size, and is tone-mapped on the CPU from the planar format the software
    decoder would have given it (TONEMAP_CHAIN).

    Every chain here takes CUDA frames only (the guard, scale_cuda,
    hwdownload): a source the decoder handed on in system memory after all
    fails the run, and the CPU decodes it (run_encode)."""
    frames = CUDA_FRAME_FORMATS[src.bit_depth]
    if _down_on_the_cpu(rung, src):
        fmt = frames
    else:
        fmt = "p010le" if rung.ten_bit else "nv12"
    chain: list[str] = []
    if rung.scaled:
        *halves, last = _cuda_scale_steps(src.width, src.height, rung.width, rung.height)
        chain += [f"scale_cuda={w}:{h}:interp_algo={CUDA_SCALE_ALGO}" for w, h in halves]
        chain += [f"scale_cuda={last[0]}:{last[1]}:format={fmt}:interp_algo={CUDA_SCALE_ALGO}",
                  "setsar=1"]
    elif fmt != frames:
        chain.append(f"scale_cuda=format={fmt}")
    if rung.tonemap:
        chain += ["hwdownload", f"format={fmt}", f"format={PLANAR_FORMATS[fmt]}", TONEMAP_CHAIN]
    elif not cuda_frames:
        chain += ["hwdownload", f"format={fmt}", *([] if rung.ten_bit else ["format=yuv420p"])]
    return chain or [CUDA_GUARD]


def _cuda_fed(encoded: list[RungPlan], src: SourceInfo) -> str | None:
    """The id of the one encoded rung of a run whose encoder takes the
    decoder's CUDA frames as they are: the first whose frames need not come
    down to the CPU (_down_on_the_cpu), the largest; None when every rung's
    do. The other rungs are scaled on the GPU all the same and come down
    (hwdownload) in the format the CPU path gives their encoder, which
    takes them as it does there.

    NVENC_LEVEL_RACE is why one at most, and why a plan of two HEVC rungs
    is not decoded on the GPU at all (_plan_cuda_blocker)."""
    return next((r.id for r in encoded if not _down_on_the_cpu(r, src)), None)


# What the GPU path keeps away from (_cuda_fed, _plan_cuda_blocker), and
# what _nvenc_failure catches where it happens all the same: NVENC now and
# then sizes an HEVC encode's DPB one frame larger than its own, and raises
# its level to fit it (a 720p rung to 4.0, a 1080p one to 5.0, a 4K one to
# 6.0, where the CPU path's are 3.1, 4.0 and 5.0, which a decoder for 4.1 or
# 5.1 turns down), when another encode opens beside it: every time a second
# encode took CUDA frames from the same context, and with the second
# brought down to system memory still in 1 of 6 runs of two HEVC encodes,
# where the CPU path's two (each opens later, in a context of its own) never
# did in 9. One HEVC encode beside H.264 ones that take system memory frames
# did it in 17 of 20 runs whose PTX cache (CUDA_CACHE_PATH) was empty, with
# one CUDA device or two, and in none of 52 once the cache held scale_cuda's
# kernels, as the startup probe leaves it (_cuda_decode_works). Neither
# -dpb_size nor -level holds it: NVENC ignores the one and refuses to open
# with the other.
NVENC_LEVEL_RACE = "hevc_rungs"


def _plan_cuda_blocker(plan: Plan) -> str | None:
    """Why `plan` can't be decoded into CUDA frames: an encoded rung that a
    CPU encoder takes ("encoder=libx264"), more than one HEVC rung
    ("hevc_rungs=2", NVENC_LEVEL_RACE), or the source's reason
    (SourceInfo.cuda_decode_blocker); None when it can."""
    encoded = [r for r in plan.rungs if r.mode == "encode"]
    if not encoded:
        return "nothing to decode"
    for rung in encoded:
        if not rung.encoder.endswith("_nvenc"):
            return f"encoder={rung.encoder}"
    if (hevc := sum(r.codec == "hevc" for r in encoded)) > 1:
        return f"{NVENC_LEVEL_RACE}={hevc}"
    return plan.source.cuda_decode_blocker


def cuda_decode_blocker(plan: Plan, settings: EncodeSettings) -> str | None:
    """Why `plan`'s source is decoded on the CPU, or None when NVDEC decodes
    it into CUDA frames (README "GPU decoding"): "cuda off" (DECODER=cpu, or
    a host whose startup probe found none: detect_decoder), "encoder=..."
    (a rung a CPU encoder takes), "hevc_rungs=2" (NVENC_LEVEL_RACE), or the
    source's reason ("codec=mpeg2video", "4:2:2", "12-bit", "interlaced",
    "dolby_vision_5", ...)."""
    if settings.decoder != "cuda":
        return "cuda off"
    return _plan_cuda_blocker(plan)


def cuda_frame_blocker(src: Path, plan: Plan, video_index: int | None) -> str | None:
    """Why the frames NVDEC hands on for `src` are not the source's own, or
    None when they are: its first frame, decoded into CUDA frames and
    brought down, must be as large as ffprobe says its picture is. A picture
    cropped at its top or its left (H.264's frame cropping, HEVC's
    conformance window) comes out of NVDEC uncropped there: libavutil crops
    a hardware frame at its right and bottom only, so the frame would be
    the picture shifted by the rows or columns it drops, with exit 0
    ("nvdec_frame=1920x1080" for a 1920x1064 picture). A decode that fails
    or hands on nothing ("nvdec_preflight_failed") leaves the source to the
    CPU as well. About 0.5 s, once per item that would decode on the GPU."""
    fmt = CUDA_FRAME_FORMATS.get(plan.source.bit_depth, "nv12")
    try:
        result = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", *CUDA_DECODE_ARGS, "-i", str(src),
             "-map", f"0:{video_index}" if video_index is not None else "0:v:0",
             "-frames:v", "1", "-vf", f"hwdownload,format={fmt},format=gray",
             "-f", "image2pipe", "-c:v", "pgm", "-"],
            capture_output=True, stdin=subprocess.DEVNULL, timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "nvdec_preflight_failed"
    header = re.match(rb"P5\s+(\d+)\s+(\d+)\s", result.stdout or b"")
    if result.returncode != 0 or header is None:
        return "nvdec_preflight_failed"
    width, height = int(header.group(1)), int(header.group(2))
    if (width, height) != (plan.source.width, plan.source.height):
        return f"nvdec_frame={width}x{height}"
    return None


def _reshape_format(top: RungPlan) -> str:
    """The 10-bit format a reshaped source comes down from the Vulkan
    device in: NVENC's p010le when the top rung is a 10-bit NVENC encode,
    so it needs no conversion on the CPU, else planar yuv420p10le."""
    return "p010le" if top.ten_bit and top.encoder.endswith("_nvenc") else "yuv420p10le"


def _vulkan_device_arg(vulkan: VulkanDevice) -> str:
    """`-init_hw_device`'s value for the preflight's device: its index in
    ffmpeg's listing, or ffmpeg's default device."""
    return f"vulkan=vk:{vulkan.index}" if vulkan.index else "vulkan=vk"


def _keyframe_args(rung: RungPlan, plan: Plan) -> list[str]:
    """Keyframe placement, identical for every encoded rung of a plan.

    interval: IDR at the first frame at/after every multiple of
      segment_seconds (`expr:gte(t,n_forced*S)`). The GOP length equals
      the interval too, so an encoder whose GOP counter doesn't reset on
      a forced IDR (NVENC) still lands its own keyframes on the same
      frames. Result: every segment is exactly S seconds on every rung.

    source: a stream-copied rung keeps the source's keyframes, which we
      can't move. The encoded rungs therefore put an IDR on exactly the
      frames the source has keyframes on (`-force_key_frames source`)
      and on no others (scene-cut detection off, GOP ceiling raised), so
      the packager — which cuts at the first keyframe of each S-second
      window — cuts every rung at the same instants.
    """
    seg = plan.segment_seconds
    if plan.keyframes == "source":
        gop = max(1, math.ceil(plan.source.fps * SOURCE_ALIGNED_MAX_GOP_SECONDS))
        args = ["-forced-idr", "1", "-force_key_frames", "source", "-g", str(gop)]
        if rung.encoder.endswith("_nvenc"):
            args += ["-no-scenecut", "1"]
        elif rung.encoder == "libx264":
            args += ["-sc_threshold", "0"]
        return args
    gop = max(1, math.ceil(plan.source.fps * seg))
    return [
        "-forced-idr", "1",
        "-force_key_frames", f"expr:gte(t,n_forced*{seg})",
        "-g", str(gop),
    ]


def _video_args(rung: RungPlan, plan: Plan, settings: EncodeSettings, *,
                cuda_frames: bool = False) -> list[str]:
    """The rung's encoder and its options. `cuda_frames`: the frames reach
    the encoder in GPU memory, in the layout the encode takes (the CUDA
    filters put them in it), so NVENC is named no -pix_fmt, which ffmpeg
    would meet with a conversion in system memory that CUDA frames can't
    go through."""
    maxrate = rung.maxrate_bps or 8_000_000
    rate = ["-maxrate", _rate(maxrate), "-bufsize", _rate(maxrate * 2)]
    enc = rung.encoder
    if enc == "hevc_nvenc":
        # The flag set the GPU path has always used; 10-bit for an HDR
        # source or one above 8 bits.
        args = [
            "-c:v", "hevc_nvenc",
            "-preset", settings.nvenc_preset,
            "-profile:v", "main10" if rung.ten_bit else "main",
            *([] if cuda_frames else ["-pix_fmt", "p010le" if rung.ten_bit else "yuv420p"]),
            "-rc:v", "vbr",
            "-cq", str(settings.nvenc_cq),
            *rate,
            "-b_ref_mode", "middle",
            # Spatial adaptive quantization. The hevc_nvenc AVOption is
            # spelled with a hyphen (`-spatial-aq`) in current ffmpeg
            # builds; the underscore form `-spatial_aq` is rejected as
            # "Unrecognized option" and aborts the whole encode
            # (returncode 8), so keep the hyphen.
            "-spatial-aq", "1",
            "-rc-lookahead", "20",
        ]
    elif enc == "h264_nvenc":
        args = [
            "-c:v", "h264_nvenc",
            "-preset", settings.nvenc_preset,
            "-profile:v", "high",
            *([] if cuda_frames else ["-pix_fmt", "yuv420p"]),
            "-rc:v", "vbr",
            "-cq", str(settings.nvenc_cq),
            *rate,
            "-b_ref_mode", "middle",
            "-spatial-aq", "1",
            "-rc-lookahead", "20",
        ]
    elif cuda_frames:
        raise TranscodeError(f"{enc} takes no CUDA frames")
    elif enc == "libx265":
        params = ["log-level=error"]
        if plan.keyframes == "source":
            params.append("scenecut=0")
        if rung.ten_bit and plan.source.hdr:
            # x265's HDR quantiser offsets: for PQ / HLG only. A 10-bit SDR
            # source is plain Main 10.
            params += ["hdr-opt=1", "repeat-headers=1"]
        args = [
            "-c:v", "libx265",
            "-preset", settings.x265_preset,
            "-crf", str(settings.x265_crf),
            *rate,
            "-profile:v", "main10" if rung.ten_bit else "main",
            "-pix_fmt", "yuv420p10le" if rung.ten_bit else "yuv420p",
            "-x265-params", ":".join(params),
        ]
    elif enc == "libx264":
        args = [
            "-c:v", "libx264",
            "-preset", settings.x264_preset,
            "-crf", str(settings.x264_crf),
            *rate,
            "-profile:v", "high",
            "-pix_fmt", "yuv420p",
        ]
    else:
        raise TranscodeError(f"no encoder args for {enc!r}")
    # Closed captions (EIA/CEA-608/708) that the source carries in its
    # video — A53 SEI or user data, which the decoder hands on with each
    # frame through every filter here — go into every encode. All four
    # encoders take -a53cc from ffmpeg 6.1 on; libx265 has it off by
    # default since 7.1, the others on, so it is named for each.
    args += ["-a53cc", "1"]
    if rung.tonemap:
        args += ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]
    elif plan.source.dovi_reshape:
        # HDR10, named: what libplacebo renders, while the source's own
        # tags (full range, the rest unset) say nothing of it.
        args += ["-color_primaries", "bt2020", "-color_trc", "smpte2084",
                 "-colorspace", "bt2020nc", "-color_range", "tv"]
    return args + _keyframe_args(rung, plan)


@dataclass(frozen=True)
class EncodeOutput:
    rung: RungPlan
    partial: Path
    final: Path
    # Encoded from a Dolby Vision 5 source's reshaped picture: run_encode
    # checks it before it is handed off (_reshape_failure).
    reshaped: bool = False
    # Made from a Dolby Vision 7 source's base layer, copied or encoded:
    # run_encode checks it before it is handed off (_base_layer_failure).
    base_layer: bool = False


def build_encode_command(
    src: Path,
    inbox: Path,
    plan: Plan,
    settings: EncodeSettings,
    *,
    probe_subtitles: list[dict[str, Any]] | None,
    video_index: int | None = None,
    decoder: str = "cpu",
) -> tuple[list[str], list[EncodeOutput]]:
    """Build the single ffmpeg invocation for every rung the transcoder
    writes: the encoded rungs, and a copy it writes itself (strip_dovi).

    One encoded rung that needs no filter (the default: a source-size
    HEVC encode of an SDR source) maps the video directly — the same
    command the worker has always run, plus the keyframe flags. Anything
    else goes through one filter graph: `split` the decoded source into
    one branch per rung, scale / tone-map each branch.

    Why -map -0:v:m:attached_pic: many MKV rips carry the poster as a
    "video" stream with disposition attached_pic. Without the explicit
    exclude, ffmpeg interprets `-map 0:v` as "all video streams" and
    feeds the poster image into the encoder, which then fails with
    'Width or height not supported with this codec'.

    A Dolby Vision 5 source opens the preflight's Vulkan device and is
    reshaped to HDR10 once, ahead of everything else in the graph
    (DOVI_RESHAPE_CHAIN); its rungs are then those of an HDR10 source,
    their colour tags named (_video_args).

    decoder="cuda": the same encode, its source decoded by NVDEC into GPU
    memory (CUDA_DECODE_ARGS), every branch's filters the CUDA ones
    (_cuda_filters), always through the graph, and NVENC named no pixel
    format (the frames are in the one it takes). Everything else — the
    maps, the encoders and their rate control, the keyframes, the time
    base, the audio and subtitles — is the CPU command's. TranscodeError
    when the plan can't be decoded so (_plan_cuda_blocker).

    A Dolby Vision 7 source's copy (RungPlan.strip_dovi) is its base layer:
    the source's video stream-copied through DOVI_STRIP_BSF, beside the
    encodes if there are any, no decoder involved.
    """
    written = [r for r in plan.rungs if r.file is not None]
    if not written:
        raise TranscodeError("plan has no rung to write")
    encoded = [r for r in written if r.mode == "encode"]
    if decoder not in ("cuda", "cpu"):
        raise TranscodeError(f"no decoder {decoder!r}")
    cuda = decoder == "cuda"
    if cuda and (why := _plan_cuda_blocker(plan)) is not None:
        raise TranscodeError(f"NVDEC does not decode this source into CUDA frames: {why}")
    vin = f"0:{video_index}" if video_index is not None else "0:v:0"
    reshape = plan.source.dovi_reshape
    if reshape and not settings.vulkan.ok:
        raise TranscodeError(
            f"Dolby Vision profile {plan.source.dv_profile} needs a Vulkan device: "
            f"{settings.vulkan.error}"
        )

    gpu_fed = _cuda_fed(encoded, plan.source) if cuda else None

    def filters(rung: RungPlan) -> list[str]:
        if cuda:
            return _cuda_filters(rung, plan.source, cuda_frames=rung.id == gpu_fed)
        return _filters(rung)

    need_graph = bool(encoded) and (cuda or reshape or len(encoded) > 1 or any(
        _filters(r) != ["format=yuv420p"] for r in encoded))

    args = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-hide_banner",
        "-loglevel", "warning",
    ]
    if reshape:
        args += ["-init_hw_device", _vulkan_device_arg(settings.vulkan),
                 "-filter_hw_device", "vk"]
    if cuda:
        args += CUDA_DECODE_ARGS
    args += [
        # -fflags +genpts: regenerate timestamps for sources with
        # broken DTS sequences (a lot of old DVD rips). -avoid_negative_ts
        # make_zero: pin the first PTS at 0 so downstream packagers
        # don't see negative timestamps that they then refuse.
        "-fflags", "+genpts",
        "-avoid_negative_ts", "make_zero",
        "-i", str(src),
    ]
    if need_graph:
        head = [DOVI_RESHAPE_CHAIN.format(fmt=_reshape_format(encoded[0]))] if reshape else []
        if len(encoded) > 1:
            branches = [f"[b{r.id}]" for r in encoded]
            chains = [f"[{vin}]{','.join([*head, f'split={len(encoded)}'])}{''.join(branches)}"]
            for branch, rung in zip(branches, encoded, strict=True):
                chains.append(f"{branch}{','.join(filters(rung))}[o{rung.id}]")
        else:
            [rung] = encoded
            chain = filters(rung)
            if head and chain == [f"format={_reshape_format(rung)}"]:
                chain = []  # the reshaping already ends in the rung's format
            chains = [f"[{vin}]{','.join([*head, *chain])}[o{rung.id}]"]
        args += ["-filter_complex", ";".join(chains)]

    sub_maps, sub_codecs, _dropped = _subtitle_maps(probe_subtitles)
    outputs: list[EncodeOutput] = []
    for rung in written:
        assert rung.file is not None
        final = inbox / rung.file
        partial = final.with_name(final.name + ".partial")
        carries_tracks = rung.id == "v0"
        if rung.mode == "copy":
            # The base layer, as it is but for the two layers it drops; its
            # packets keep the source's timestamps, on the shared timeline.
            args += ["-map", vin]
            if carries_tracks:
                args += ["-map", "0:a?", *sub_maps]
            args += ["-c:v", "copy", "-bsf:v", DOVI_STRIP_BSF]
        else:
            args += ["-map", f"[o{rung.id}]" if need_graph else vin]
            if carries_tracks:
                args += ["-map", "0:a?", *sub_maps]
                if not need_graph:
                    args += ["-map", "-0:d", "-map", "-0:v:m:attached_pic"]
            # One rung's encoder takes CUDA frames; the others' come down.
            args += _video_args(rung, plan, settings, cuda_frames=rung.id == gpu_fed)
            # Keep the decoder's timestamps exactly. The default encoder
            # time base is 1/framerate, which rounds ffmpeg's start-time
            # shift to a whole frame (e.g. an AAC-primed source starting at
            # -0.021 s: video lands at 0.042 instead of 0.021) — 21 ms off
            # the audio, and enough to move a keyframe across a segment
            # boundary relative to a stream-copied rung. Same encoder time
            # base on every rung.
            args += ["-enc_time_base:v", "demux"]
        if carries_tracks:
            # Audio + subs: stream-copy (MP4 text captions as SubRip).
            # The packager downstream re-encodes audio and extracts the
            # subtitles.
            args += ["-c:a", "copy", *sub_codecs]
        else:
            args += ["-an", "-sn", "-dn"]
        if len(written) > 1:
            # Several encoders of different speed feed separate muxers;
            # give the slow ones headroom before the muxer queue trips.
            args += ["-max_muxing_queue_size", "4096"]
        # Force Matroska: we write to `.partial`, so ffmpeg can't infer
        # the format from the extension.
        args += ["-f", "matroska", str(partial)]
        outputs.append(EncodeOutput(rung=rung, partial=partial, final=final, reshaped=reshape,
                                    base_layer=plan.source.dovi_base_layer))
    return args, outputs


def _probe_video(path: Path) -> tuple[int, int, float | None]:
    """Width, height and start time of the first video stream (header
    read only)."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,start_time", "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    try:
        stream = json.loads(result.stdout or "{}").get("streams", [{}])[0]
        start = stream.get("start_time")
        return (
            int(stream.get("width") or 0),
            int(stream.get("height") or 0),
            float(start) if start not in (None, "N/A") else None,
        )
    except (ValueError, IndexError, TypeError):
        return 0, 0, None


def dovi_metadata_frames(path: Path, video_index: int | None,
                         frames: int = DOVI_CHECK_FRAMES) -> tuple[int, int]:
    """(frames decoded, those that carry Dolby Vision metadata) among the
    first `frames` of the source's video: the RPUs ffmpeg's HEVC decoder
    hands on with each frame, which libplacebo reshapes the picture with.
    A frame without one would be encoded as it is, in the wrong colours.
    Raises TranscodeError when ffprobe fails."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", str(video_index) if video_index is not None else "v:0",
            "-read_intervals", f"%+#{frames}", "-show_frames",
            "-show_entries", "frame=pts:frame_side_data=side_data_type", "-of", "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        stdin=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        raise TranscodeError(
            f"ffprobe (Dolby Vision metadata) exited {result.returncode}: "
            f"{(result.stderr or '').strip()[:500] or '(no stderr)'}"
        )
    try:
        decoded = json.loads(result.stdout or "{}").get("frames") or []
    except ValueError:
        decoded = []
    with_rpu = sum(
        1 for f in decoded
        if any(sd.get("side_data_type") == DOVI_FRAME_METADATA
               for sd in f.get("side_data_list") or [] if isinstance(sd, dict))
    )
    return len(decoded), with_rpu


def _probe_colours(path: Path) -> tuple[dict[str, str | None], bool]:
    """The first video stream's colour tags (HDR10_TAGS' keys), and
    whether it carries a Dolby Vision configuration record."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
            f"stream={','.join(HDR10_TAGS)}:stream_side_data=side_data_type",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    try:
        stream = (json.loads(result.stdout or "{}").get("streams") or [{}])[0]
    except ValueError:
        stream = {}
    tags = {k: stream.get(k) for k in HDR10_TAGS}
    dovi = any("DOVI" in str(sd.get("side_data_type") or "")
               for sd in stream.get("side_data_list") or [] if isinstance(sd, dict))
    return tags, dovi


def max_level(codec: str, width: int, height: int, fps: float, maxrate_bps: int) -> int | None:
    """The lowest level the spec allows an encode of this coded size, frame
    rate and VBV maxrate with NVENC's DPB (NVENC_MAX_HEVC_DPB pictures,
    NVENC_MAX_H264_REFS reference frames), as ffprobe reports a level: HEVC's
    general_level_idc (5.0: 150), H.264's level_idc (4.0: 40); None when no
    level holds it. NVENC picks this level itself; the coded size is the
    picture's rounded up to 16 (1080 lines: 1088). HEVC's MaxBR counts
    x1000, as NVENC counts it (720p at 10.5 Mbit/s: 4.0), H.264 High's
    x1250, where NVENC counts more (it kept 720p at 3.1 up to 19 Mbit/s,
    past 17.5): this level is then at or above NVENC's, never below."""
    w, h = -(-width // 16) * 16, -(-height // 16) * 16
    if codec == "hevc":
        samples = w * h
        for idc, max_ps, max_sr, max_br in HEVC_LEVELS:
            if (samples <= max_ps and max(w, h) <= math.isqrt(8 * max_ps)
                    and samples * fps <= max_sr and maxrate_bps <= max_br * 1000):
                return idc
        return None
    mbs = (w // 16) * (h // 16)
    for idc, max_mbps, max_fs, max_dpb_mbs, max_br in H264_LEVELS:
        if (mbs <= max_fs and max(w, h) // 16 <= math.isqrt(8 * max_fs)
                and mbs * fps <= max_mbps and maxrate_bps <= max_br * 1250
                and min(max_dpb_mbs // mbs, 16) >= NVENC_MAX_H264_REFS):
            return idc
    return None


def _rate_of(raw: object) -> float:
    """An ffprobe frame rate ("24000/1001", "0/0") as a number; 0 unknown."""
    try:
        return float(Fraction(str(raw)))
    except (ValueError, ZeroDivisionError):
        return 0.0


def _stream_facts(path: Path) -> dict[str, Any]:
    """What _nvenc_failure checks of an output's video stream: its codec,
    profile, level, size and frame rates as ffprobe reports them, and, from
    its parameter sets (trace_headers on its first packet), an HEVC
    stream's DPB (sps_max_dec_pic_buffering_minus1 + 1, the largest of its
    sub-layers') or an H.264 stream's reference frames (max_num_ref_frames;
    ffprobe's `refs` says 1 whatever they are), and the frame rate its VUI's
    timing names (vui_frame_rate: the one NVENC was given, which the
    timestamps of a variable frame rate need not show). A value it cannot
    read is left out."""
    facts: dict[str, Any] = {}
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,profile,level,width,height,refs,r_frame_rate,avg_frame_rate",
         "-of", "json", str(path)],
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
    )
    try:
        facts.update((json.loads(result.stdout or "{}").get("streams") or [{}])[0])
    except ValueError:
        pass
    facts.pop("refs", None)
    if facts.get("codec_name") in ("hevc", "h264"):
        trace = subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "info", "-i", str(path),
             "-map", "0:v:0", "-frames:v", "1", "-c", "copy", "-bsf:v", "trace_headers",
             "-f", "null", "-"],
            capture_output=True, text=True, stdin=subprocess.DEVNULL,
        )
        text = trace.stderr or ""

        def field(name: str) -> int:
            found = re.search(rf"\s{name}\s+[01]+\s+=\s+(\d+)", text)
            return int(found.group(1)) if found else 0

        if facts["codec_name"] == "hevc":
            dpb = [int(v) for v in re.findall(
                r"sps_max_dec_pic_buffering_minus1\[\d+\]\s+[01]+\s+=\s+(\d+)", text)]
            if dpb:
                facts["dpb"] = max(dpb) + 1
            tick, scale = field("vui_num_units_in_tick"), field("vui_time_scale")
        else:
            if refs := re.findall(r"\smax_num_ref_frames\s+[01]+\s+=\s+(\d+)", text):
                facts["refs"] = max(int(v) for v in refs)
            # H.264's time_scale counts fields: two ticks a frame.
            tick, scale = 2 * field("num_units_in_tick"), field("time_scale")
        if tick and scale:
            facts["vui_frame_rate"] = str(Fraction(scale, tick))
    return facts


def _nvenc_failure(outputs: list[EncodeOutput], *, gpu: bool, log_label: str = "",
                   unread: list[str] | None = None) -> str | None:
    """Why an NVENC output must not be handed off, or None: its profile is
    not the one the encode was asked for, its DPB is larger than NVENC's own
    (NVENC_MAX_HEVC_DPB, NVENC_MAX_H264_REFS), or its level is above the
    lowest that holds its coded size, frame rate and maxrate (max_level),
    as NVENC_LEVEL_RACE makes them. On the GPU (`gpu`) also when its size is
    not its rung's (cuda_frame_blocker's crops), or its DPB (an H.264
    stream's reference frames) could not be read: trace_headers no longer
    says it in the words _stream_facts reads (another ffmpeg), and the run
    is made again on the CPU. Elsewhere such a DPB is logged loudly
    (transcoder.encode.nvenc_unread, an error), its rung's id added to
    `unread`, and the rest checked. Outputs of other encoders are not read;
    a profile or a level ffprobe gives no answer for proves nothing."""
    for out in outputs:
        rung = out.rung
        if not rung.encoder.endswith("_nvenc"):
            continue
        facts = _stream_facts(out.partial)
        size = (facts.get("width"), facts.get("height"))
        if gpu and None not in size and size != (rung.width, rung.height):
            return f"{rung.id} came out {size[0]}x{size[1]}, not {rung.width}x{rung.height}"
        want = ("Main 10" if rung.ten_bit else "Main") if rung.codec == "hevc" else "High"
        if facts.get("profile") and facts["profile"] != want:
            return f"{rung.id} came out {facts['profile']}, not {want}"
        dpb, most = ((facts.get("dpb"), NVENC_MAX_HEVC_DPB) if rung.codec == "hevc"
                     else (facts.get("refs"), NVENC_MAX_H264_REFS))
        if not isinstance(dpb, int):
            what = "DPB" if rung.codec == "hevc" else "reference frames"
            if gpu:
                return f"{rung.id}'s {what} could not be read"
            log.error("transcoder.encode.nvenc_unread", label=log_label, rung=rung.id, what=what,
                      then="handed off with its DPB unchecked")
            if unread is not None:
                unread.append(rung.id)
        elif dpb > most:
            return f"{rung.id} came out with a DPB of {dpb}, NVENC's own is at most {most}"
        fps = max(_rate_of(facts.get(rate))
                  for rate in ("r_frame_rate", "avg_frame_rate", "vui_frame_rate"))
        level = facts.get("level")
        if isinstance(level, int) and level > 0 and fps and None not in size:
            top = max_level(rung.codec, int(size[0]), int(size[1]), fps,
                            rung.maxrate_bps or 8_000_000)
            if top is not None and level > top:
                return f"{rung.id} came out at level {level}, above {top}"
    return None


def unkillable() -> str | None:
    """Why the pod is unhealthy, or None: an ffmpeg run had not died
    KILL_WAIT_SECONDS after its kill (_run_watched), and holds what it held
    (the GPU, the memory) until the pod is restarted."""
    return _UNKILLED[0] if _UNKILLED else None


def _run_watched(args: list[str], stall_seconds: float = STALL_SECONDS,
                 period: float = PROGRESS_SECONDS,
                 kill_wait: float = KILL_WAIT_SECONDS) -> tuple[subprocess.CompletedProcess, bool]:
    """Run an ffmpeg command with its progress on its stdout (-progress),
    and kill it when the progress stands still for `stall_seconds` (no frame
    encoded, no byte written, no time encoded; the same block every period
    is no progress): a call into NVDEC, NVENC or CUDA that never returns
    would hold the worker forever. (The result, its stderr captured as
    subprocess.run's; whether it was killed.) Raises TranscodeError when it
    has not died `kill_wait` seconds after its kill, and the pod is
    unhealthy from then on (unkillable)."""
    cmd = [args[0], "-progress", "pipe:1", "-stats_period", str(period), *args[1:]]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            stdin=subprocess.DEVNULL, text=True, errors="replace")
    last = [time.monotonic()]
    seen: dict[str, str] = {}
    stderr: list[str] = []

    def progress() -> None:
        for line in proc.stdout:  # type: ignore[union-attr]
            key, _, value = line.strip().partition("=")
            if key in ("frame", "total_size", "out_time_us") and seen.get(key) != value:
                seen[key] = value
                last[0] = time.monotonic()

    readers = [threading.Thread(target=progress, daemon=True),
               threading.Thread(target=lambda: stderr.append(proc.stderr.read()),  # type: ignore[union-attr]
                                daemon=True)]
    for reader in readers:
        reader.start()
    stalled = False
    while True:
        try:
            proc.wait(timeout=min(1.0, stall_seconds))
            break
        except subprocess.TimeoutExpired:
            if time.monotonic() - last[0] > stall_seconds:
                stalled = True
                proc.kill()
                try:
                    proc.wait(timeout=kill_wait)
                except subprocess.TimeoutExpired:
                    why = (f"ffmpeg (pid {proc.pid}) stood still for {stall_seconds:g} s and had "
                           f"not died {kill_wait:g} s after its kill")
                    _UNKILLED.append(why)
                    log.error("transcoder.encode.unkillable", pid=proc.pid, why=why,
                              then="the pod reports unhealthy")
                    raise TranscodeError(f"{why}; the pod reports unhealthy") from None
                break
    for reader in readers:
        reader.join(timeout=10)
    return subprocess.CompletedProcess(cmd, proc.returncode, "", "".join(stderr)), stalled


def _watched(args: list[str],
             outputs: list[EncodeOutput]) -> tuple[subprocess.CompletedProcess, bool]:
    """_run_watched, the partials removed when it raises (an ffmpeg that
    would not die)."""
    try:
        return _run_watched(args)
    except TranscodeError:
        for out in outputs:
            out.partial.unlink(missing_ok=True)
        raise


def _gpu_failure(result: subprocess.CompletedProcess, stalled: bool,
                 outputs: list[EncodeOutput]) -> str | None:
    """Why a GPU run must be made again on the CPU, or None: it stood still
    (_run_watched), it failed, it wrote a decoder error (GPU_DECODE_ERRORS:
    a frame not decoded), or an output is not what NVENC makes of system
    memory frames, or not its rung's size (_nvenc_failure)."""
    if stalled:
        return f"no progress for {STALL_SECONDS} s"
    if result.returncode != 0:
        return f"exit {result.returncode}"
    text = (result.stderr or "").lower()
    if (error := next((e for e in GPU_DECODE_ERRORS if e in text), None)) is not None:
        return f"decoder error ({error})"
    return _nvenc_failure(outputs, gpu=True)


def _tag_failure(out: EncodeOutput) -> str | None:
    """Why an output of a Dolby Vision source is not what it is meant to be,
    or None: it is not tagged HDR10 (BT.709 for a tone-mapped H.264 rung),
    or it claims to be Dolby Vision (a configuration record)."""
    tags, dovi = _probe_colours(out.partial)
    want = SDR_TAGS if out.rung.tonemap else HDR10_TAGS
    if tags != want or dovi:
        got = "/".join(str(tags[k]) for k in HDR10_TAGS) + (" with a DOVI record" if dovi
                                                             else "")
        return f"{out.rung.id} came out as {got}, not {'/'.join(want.values())}"
    return None


def _reshape_failure(result_stderr: str, outputs: list[EncodeOutput]) -> str | None:
    """Why a finished encode of a Dolby Vision 5 source must not be handed
    off, or None: the decoder could not read an RPU (those frames were
    not reshaped), or an output is not tagged as what it is meant to be —
    HDR10, or BT.709 for a tone-mapped H.264 rung — or claims to be Dolby
    Vision."""
    if DOVI_RPU_ERROR in (result_stderr or ""):
        # ffmpeg logs a repeated message once, so the log does not count
        # the frames.
        return (f"the decoder could not read the Dolby Vision RPU of some frames "
                f"({DOVI_RPU_ERROR!r}): those would play in the wrong colours")
    for out in outputs:
        if (why := _tag_failure(out)) is not None:
            return why
    return None


def _dovi_nal_types(path: Path, frames: int = DOVI_CHECK_FRAMES) -> list[int]:
    """The Dolby Vision NAL unit types (DOVI_EL_RPU_NAL_TYPES) among the first
    `frames` packets of an HEVC output, as ffmpeg writes them out of it in
    Annex B; [] for none, or when it could not be read (the tags say the
    rest)."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:v:0",
             "-frames:v", str(frames), "-c", "copy", "-bsf:v", "hevc_mp4toannexb",
             "-f", "hevc", "-"],
            capture_output=True, stdin=subprocess.DEVNULL, timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    units = (result.stdout or b"").split(b"\0\0\1")[1:]
    return sorted({unit[0] >> 1 & 0x3F for unit in units if unit} & set(DOVI_EL_RPU_NAL_TYPES))


def _base_layer_failure(outputs: list[EncodeOutput]) -> str | None:
    """Why the outputs of a Dolby Vision 7 source, its base layer copied or
    encoded, must not be handed off, or None: one is not tagged what it is
    meant to be or claims Dolby Vision (_tag_failure), or an HEVC one's
    first frames still carry the enhancement layer's or the RPU's NAL
    units."""
    for out in outputs:
        if (why := _tag_failure(out)) is not None:
            return why
        if out.rung.codec == "hevc" and (left := _dovi_nal_types(out.partial)):
            return (f"{out.rung.id} still carries NAL units of type "
                    f"{' and '.join(map(str, left))}")
    return None


@dataclass(frozen=True)
class RungResult:
    rung: RungPlan
    path: Path
    size_bytes: int
    width: int
    height: int
    # First video timestamp in the output file — on the shared timeline
    # (source timestamp + the input start-time shift) for every rung.
    video_start: float | None = None
    # Where the source was decoded for it: "cuda", "cpu", or "none" for a
    # run of copies alone (run_encode).
    decoder: str = "cpu"


def run_encode(
    args: list[str],
    outputs: list[EncodeOutput],
    *,
    log_label: str,
    cpu_fallback: list[str] | None = None,
) -> tuple[list[RungResult], float]:
    """Run the encode. Every output lands as `.partial` first; on
    success the lower rungs are renamed into place before v0
    (`prepared.mkv`), whose existence an older packager treats as
    "handoff complete". Raises TranscodeError on failure (partials
    removed); the caller logs + reports.

    `cpu_fallback`: `args` decode the source on the GPU (decoder="cuda"),
    and this is the same encode decoding it on the CPU. The GPU run is
    watched (_run_watched), and when it stands still, fails, writes a
    decoder error, or makes an output that is not what NVENC makes of
    system memory frames or not its rung's size (_gpu_failure), whatever
    its exit code, it is logged (`transcoder.encode.cuda_failed`, with why
    and ffmpeg's last words), its partials removed, and the CPU run made in
    its place, so that no item fails for the GPU's decoding: only the CPU
    run's failure fails it. Each result names the run it came from
    (RungResult.decoder).

    Every run is watched (_run_watched): a run on the CPU's decode that
    stands still is killed and fails, nothing renamed into place; one that
    will not die fails and leaves the pod unhealthy (unkillable). Every
    NVENC output of a run on the CPU's decode is checked as a GPU run's
    are, but for its size and a DPB it cannot read, which it logs
    (_nvenc_failure): one whose profile, DPB or level is not NVENC's own
    fails the run, nothing renamed into place.

    Outputs of a Dolby Vision 5 source (`reshaped`): a run that exits 0
    still fails, nothing renamed into place, when the decoder could not
    read an RPU or an output is not tagged as it must be
    (_reshape_failure). So do those of a Dolby Vision 7 source
    (`base_layer`) when one is not tagged as it must be, claims Dolby
    Vision, or still carries an enhancement layer or an RPU in its first
    frames (_base_layer_failure)."""
    for out in outputs:
        out.final.parent.mkdir(parents=True, exist_ok=True)
        # Clean up a stale partial from a previous crash. We never resume
        # a partial encode — the encoder state is gone with the process.
        out.partial.unlink(missing_ok=True)

    encoders = ",".join(sorted({o.rung.encoder for o in outputs}))
    decodes = any(out.rung.mode == "encode" for out in outputs)
    decoder = "cuda" if cpu_fallback is not None else ("cpu" if decodes else "none")
    log.info(
        "transcoder.encode.start",
        label=log_label,
        encoders=encoders,
        decoder=decoder,
        rungs=[f"{o.rung.id}:{o.rung.encoder}:{o.rung.width}x{o.rung.height}" for o in outputs],
    )
    t0 = time.monotonic()
    result, stalled = _watched(args, outputs)
    if cpu_fallback is not None:
        if (why := _gpu_failure(result, stalled, outputs)) is not None:
            for out in outputs:
                out.partial.unlink(missing_ok=True)
            log.warning(
                "transcoder.encode.cuda_failed",
                label=log_label,
                why=why,
                exit_code=result.returncode,
                after_s=round(time.monotonic() - t0, 1),
                error=(result.stderr or "").strip()[-600:] or None,
                then="decoding on the CPU",
            )
            decoder = "cpu"
            result, stalled = _watched(cpu_fallback, outputs)
    elapsed = round(time.monotonic() - t0, 1)
    if stalled:
        for out in outputs:
            out.partial.unlink(missing_ok=True)
        raise TranscodeError(
            f"{encoders} made no progress for {STALL_SECONDS} s and was killed after "
            f"{elapsed}s: {(result.stderr or '').strip()[-1500:] or '(no stderr)'}"
        )
    if result.returncode != 0:
        # Drop the partials so a retry isn't tempted to pick them up.
        for out in outputs:
            out.partial.unlink(missing_ok=True)
        raise TranscodeError(
            f"{encoders} exited {result.returncode} after {elapsed}s: "
            f"{(result.stderr or '').strip()[-1500:] or '(no stderr)'}"
        )
    if decoder != "cuda" and (why := _nvenc_failure(outputs, gpu=False,
                                                    log_label=log_label)) is not None:
        for out in outputs:
            out.partial.unlink(missing_ok=True)
        raise TranscodeError(f"NVENC: {why}; nothing handed off")
    reshaped = any(out.reshaped for out in outputs)
    if reshaped and (why := _reshape_failure(result.stderr, outputs)) is not None:
        for out in outputs:
            out.partial.unlink(missing_ok=True)
        raise TranscodeError(f"Dolby Vision to HDR10: {why}; nothing handed off")
    if any(out.base_layer for out in outputs) and (why := _base_layer_failure(outputs)):
        for out in outputs:
            out.partial.unlink(missing_ok=True)
        raise TranscodeError(f"Dolby Vision 7 to its HDR10 base layer: {why}; nothing handed off")
    results: list[RungResult] = []
    for out in sorted(outputs, key=lambda o: o.rung.id == "v0"):
        os.replace(out.partial, out.final)
    for out in outputs:
        width, height, start = _probe_video(out.final)
        results.append(RungResult(
            rung=out.rung,
            path=out.final,
            size_bytes=out.final.stat().st_size,
            width=width or out.rung.width,
            height=height or out.rung.height,
            video_start=start,
            decoder=decoder,
        ))
    starts = {r.rung.id: r.video_start for r in results if r.video_start is not None}
    if starts and max(starts.values()) - min(starts.values()) > 0.002:
        # Every rung comes out of one ffmpeg run, so they should all start
        # on the same frame at the same instant. If a muxer shifted one
        # file, the packager's segments would no longer line up.
        log.warning("transcoder.encode.rung_start_mismatch", label=log_label, starts=starts)
    log.info(
        "transcoder.encode.done",
        label=log_label,
        decoder=decoder,
        elapsed_s=elapsed,
        out_size_mb=round(sum(r.size_bytes for r in results) / 1_000_000, 1),
    )
    return results, elapsed


# ------------------------------------------------------- NVENC self-test
# Its clip: 2 s of 720p at 23.976 fps (past NVENC's lookahead and its
# B-frames), as MPEG-2, which ffmpeg's own encoder writes in every build
# and whose source rung is an HEVC encode; and the rungs it is encoded to.
SELFTEST_SOURCE = "testsrc2=s=1280x720:r=24000/1001:d=2"
SELFTEST_RUNGS = (("source", "hevc"), ("480p", "h264"))


@dataclass(frozen=True)
class SelfTest:
    """What nvenc_selftest found. `status`: "ok"; "skipped" (no NVENC
    encoder); "unread" (an output's DPB could not be read: every GPU run
    would fail the check and be made again on the CPU); "failed" (an output
    of this host's NVENC is not what the check holds it to, or the encode
    failed: every NVENC encode would fail its step). `why`: the check's or
    ffmpeg's words."""

    status: str
    why: str | None = None


def nvenc_selftest(settings: EncodeSettings, ffmpeg_bin: str = "ffmpeg") -> SelfTest:
    """Encode a short clip (SELFTEST_SOURCE) to a source-size HEVC rung and
    a 480p H.264 rung, those of the two whose encoder is NVENC, with the
    CPU path's command (system memory frames) and the settings every
    encode takes, and check the outputs as every encode's are
    (_nvenc_failure). Another GPU, driver or ffmpeg whose NVENC picks
    another DPB, level or profile than the check holds it to would fail
    every NVENC encode's step, again on each retry. About 1 s, at startup;
    one log line, transcoder.nvenc.selftest (an error but for "ok" and
    "skipped")."""
    codecs = {codec for codec, encoder in (("hevc", settings.encoders.hevc),
                                           ("h264", settings.encoders.h264))
              if encoder.endswith("_nvenc")}
    t0 = time.monotonic()

    def done(status: str, why: str | None = None) -> SelfTest:
        emit = log.info if status in ("ok", "skipped") else log.error
        emit("transcoder.nvenc.selftest", status=status, why=why,
             seconds=round(time.monotonic() - t0, 2))
        return SelfTest(status, why)

    if not codecs:
        return done("skipped")
    with tempfile.TemporaryDirectory(prefix="transcoder-selftest-") as tmp:
        clip = Path(tmp) / "clip.mkv"
        try:
            made = subprocess.run(
                [ffmpeg_bin, "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                 "-i", SELFTEST_SOURCE, "-c:v", "mpeg2video", "-q:v", "2", str(clip)],
                capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL,
            )
            if made.returncode != 0:
                return done("failed", f"the clip: {(made.stderr or '').strip()[-300:]}")
            probe = ffprobe(clip)
            ladder = parse_ladder(",".join(f"{name}:{codec}" for name, codec in SELFTEST_RUNGS
                                           if codec in codecs))
            plan = plan_renditions(
                probe, ladder, settings.encoders, segment_seconds=settings.segment_seconds,
                nvenc_caps_mbps=(settings.maxrate_1080p_mbps, settings.maxrate_2160p_mbps),
                caps=settings.caps)
            args, outputs = build_encode_command(clip, Path(tmp) / "out", plan, settings,
                                                 probe_subtitles=[],
                                                 video_index=probe.get("video_index"))
            for out in outputs:
                out.partial.parent.mkdir(parents=True, exist_ok=True)
            run = subprocess.run(args, capture_output=True, text=True, timeout=300,
                                 stdin=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired, TranscodeError, ValueError) as e:
            return done("failed", str(e)[-300:])
        if run.returncode != 0:
            return done("failed", f"exit {run.returncode}: {(run.stderr or '').strip()[-300:]}")
        unread: list[str] = []
        if (why := _nvenc_failure(outputs, gpu=False, log_label="selftest",
                                  unread=unread)) is not None:
            return done("failed", why)
    if unread:
        return done("unread", f"the DPB of {', '.join(unread)} could not be read")
    return done("ok")
