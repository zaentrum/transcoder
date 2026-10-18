"""ffprobe + ffmpeg invocations.

Responsibilities:
  1. `ffprobe()` — parse the source into a small dict (container,
     duration_ms, video, audio[], subtitles[]) used by the decision
     layer and by the encode pipeline.
  2. `detect_encoders()` — find out at startup whether NVENC actually
     works on this host (a GPU-less box has the encoder compiled in but
     no driver), and fall back to libx265 / libx264 per codec.
  3. `vulkan_preflight()` — find out at startup whether ffmpeg can open a
     Vulkan device and run libplacebo on it, which a Dolby Vision 5
     source is reshaped with; it logs one line naming the Vulkan loader.
  4. `build_encode_command()` + `run_encode()` — ONE ffmpeg invocation
     per item: decode the source once, `split` it into one branch per
     encoded rung of the plan, and write one intermediate MKV per rung:
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
import time
from dataclasses import dataclass, field
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
    VulkanDevice,
    parse_ladder,
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
    and `size_bytes` the file's size (the cap's size rule, and its bit
    rate fallback).

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


def _video_args(rung: RungPlan, plan: Plan, settings: EncodeSettings) -> list[str]:
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
            "-pix_fmt", "p010le" if rung.ten_bit else "yuv420p",
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
            "-pix_fmt", "yuv420p",
            "-rc:v", "vbr",
            "-cq", str(settings.nvenc_cq),
            *rate,
            "-b_ref_mode", "middle",
            "-spatial-aq", "1",
            "-rc-lookahead", "20",
        ]
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


def build_encode_command(
    src: Path,
    inbox: Path,
    plan: Plan,
    settings: EncodeSettings,
    *,
    probe_subtitles: list[dict[str, Any]] | None,
    video_index: int | None = None,
) -> tuple[list[str], list[EncodeOutput]]:
    """Build the single ffmpeg invocation for every encoded rung.

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
    """
    encoded = [r for r in plan.rungs if r.mode == "encode"]
    if not encoded:
        raise TranscodeError("plan has no encoded rung")
    vin = f"0:{video_index}" if video_index is not None else "0:v:0"
    reshape = plan.source.dovi_reshape
    if reshape and not settings.vulkan.ok:
        raise TranscodeError(
            f"Dolby Vision profile {plan.source.dv_profile} needs a Vulkan device: "
            f"{settings.vulkan.error}"
        )
    need_graph = reshape or len(encoded) > 1 or any(
        _filters(r) != ["format=yuv420p"] for r in encoded)

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
                chains.append(f"{branch}{','.join(_filters(rung))}[o{rung.id}]")
        else:
            [rung] = encoded
            filters = _filters(rung)
            if head and filters == [f"format={_reshape_format(rung)}"]:
                filters = []  # the reshaping already ends in the rung's format
            chains = [f"[{vin}]{','.join([*head, *filters])}[o{rung.id}]"]
        args += ["-filter_complex", ";".join(chains)]

    sub_maps, sub_codecs, _dropped = _subtitle_maps(probe_subtitles)
    outputs: list[EncodeOutput] = []
    for rung in encoded:
        assert rung.file is not None
        final = inbox / rung.file
        partial = final.with_name(final.name + ".partial")
        args += ["-map", f"[o{rung.id}]" if need_graph else vin]
        carries_tracks = rung.id == "v0"
        if carries_tracks:
            args += ["-map", "0:a?", *sub_maps]
            if not need_graph:
                args += ["-map", "-0:d", "-map", "-0:v:m:attached_pic"]
        args += _video_args(rung, plan, settings)
        # Keep the decoder's timestamps exactly. The default encoder time
        # base is 1/framerate, which rounds ffmpeg's start-time shift to a
        # whole frame (e.g. an AAC-primed source starting at -0.021 s:
        # video lands at 0.042 instead of 0.021) — 21 ms off the audio,
        # and enough to move a keyframe across a segment boundary relative
        # to a stream-copied rung. Same encoder time base on every rung.
        args += ["-enc_time_base:v", "demux"]
        if carries_tracks:
            # Audio + subs: stream-copy (MP4 text captions as SubRip).
            # The packager downstream re-encodes audio and extracts the
            # subtitles.
            args += ["-c:a", "copy", *sub_codecs]
        else:
            args += ["-an", "-sn", "-dn"]
        if len(encoded) > 1:
            # Several encoders of different speed feed separate muxers;
            # give the slow ones headroom before the muxer queue trips.
            args += ["-max_muxing_queue_size", "4096"]
        # Force Matroska: we write to `.partial`, so ffmpeg can't infer
        # the format from the extension.
        args += ["-f", "matroska", str(partial)]
        outputs.append(EncodeOutput(rung=rung, partial=partial, final=final, reshaped=reshape))
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
        tags, dovi = _probe_colours(out.partial)
        want = SDR_TAGS if out.rung.tonemap else HDR10_TAGS
        if tags != want or dovi:
            got = "/".join(str(tags[k]) for k in HDR10_TAGS) + (" with a DOVI record" if dovi
                                                                 else "")
            return (f"{out.rung.id} came out as {got}, not "
                    f"{'/'.join(want.values())}")
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


def run_encode(
    args: list[str],
    outputs: list[EncodeOutput],
    *,
    log_label: str,
) -> tuple[list[RungResult], float]:
    """Run the encode. Every output lands as `.partial` first; on
    success the lower rungs are renamed into place before v0
    (`prepared.mkv`), whose existence an older packager treats as
    "handoff complete". Raises TranscodeError on failure (partials
    removed); the caller logs + reports.

    Outputs of a Dolby Vision 5 source (`reshaped`): a run that exits 0
    still fails, nothing renamed into place, when the decoder could not
    read an RPU or an output is not tagged as it must be
    (_reshape_failure)."""
    for out in outputs:
        out.final.parent.mkdir(parents=True, exist_ok=True)
        # Clean up a stale partial from a previous crash. We never resume
        # a partial encode — the encoder state is gone with the process.
        out.partial.unlink(missing_ok=True)

    encoders = ",".join(sorted({o.rung.encoder for o in outputs}))
    log.info(
        "transcoder.encode.start",
        label=log_label,
        encoders=encoders,
        rungs=[f"{o.rung.id}:{o.rung.encoder}:{o.rung.width}x{o.rung.height}" for o in outputs],
    )
    t0 = time.monotonic()
    result = subprocess.run(args, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    elapsed = round(time.monotonic() - t0, 1)
    if result.returncode != 0:
        # Drop the partials so a retry isn't tempted to pick them up.
        for out in outputs:
            out.partial.unlink(missing_ok=True)
        raise TranscodeError(
            f"{encoders} exited {result.returncode} after {elapsed}s: "
            f"{(result.stderr or '').strip()[-1500:] or '(no stderr)'}"
        )
    reshaped = any(out.reshaped for out in outputs)
    if reshaped and (why := _reshape_failure(result.stderr, outputs)) is not None:
        for out in outputs:
            out.partial.unlink(missing_ok=True)
        raise TranscodeError(f"Dolby Vision to HDR10: {why}; nothing handed off")
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
        elapsed_s=elapsed,
        out_size_mb=round(sum(r.size_bytes for r in results) / 1_000_000, 1),
    )
    return results, elapsed
