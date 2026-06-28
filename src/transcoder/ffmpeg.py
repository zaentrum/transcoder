"""ffprobe + ffmpeg NVENC invocations.

Two responsibilities:
  1. `ffprobe()` — parse the source into a small dict (container,
     duration_ms, video, audio[], subtitles[]) used by the decision
     layer and by the encode pipeline to choose -maxrate.
  2. `encode_to_hevc_mkv()` — run a single hevc_nvenc ffmpeg
     invocation that writes an intermediate MKV containing:
        - HEVC video at the resolution-appropriate bitrate cap;
        - every audio track copied through verbatim (the packager
          re-encodes to AAC stereo downstream — doing it here would
          waste GPU time on a CPU-only operation);
        - every subtitle track copied through verbatim, including
          image-based formats (PGS, VobSub) and rich-text formats
          (ASS, SSA). MP4 can't carry those; MKV can, which is the
          whole reason we use MKV for the intermediate.

The atomic-rename pattern matters: write to `prepared.mkv.partial`
first, then `os.replace` to `prepared.mkv`. The packager's worker
treats the existence of `prepared.mkv` as proof of a complete file.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)


# Subtitle codecs Matroska can carry losslessly via stream-copy. We
# stream-copy these into prepared.mkv so the packager downstream can
# extract them. The list is the intersection of "supported by ffmpeg's
# matroska muxer" and "documented in the Matroska spec" — any other
# source codec gets dropped from the intermediate (the most common
# culprit is `mov_text`, the MP4-native captions codec; it'd need a
# transcode to subrip which we don't want to do on the GPU pod).
MATROSKA_SUBTITLE_COPY_CODECS = {
    "subrip", "srt", "ass", "ssa", "webvtt",
    "hdmv_pgs_subtitle", "dvb_subtitle", "dvd_subtitle",
    "microdvd",
}


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
          "audio": [{"codec_name": "ac3", "channels": 6, ...}, ...],
          "subtitles": [{"codec_name": "hdmv_pgs_subtitle", ...}, ...],
        }

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
    return {
        "container": fmt.get("format_name", ""),
        "duration_ms": duration_ms,
        "video": next((s for s in streams if s.get("codec_type") == "video"), {}),
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
    say, 1440p uploads that aren't quite 4K."""
    if max(width, height) > 1920:
        return EncodeProfile(label="nvenc-2160p", maxrate_mbps=maxrate_2160p_mbps)
    return EncodeProfile(label="nvenc-1080p", maxrate_mbps=maxrate_1080p_mbps)


def encode_to_hevc_mkv(
    src: Path,
    out_path: Path,
    *,
    profile: EncodeProfile,
    nvenc_preset: str,
    nvenc_cq: int,
    log_label: str,
    probe_subtitles: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run one hevc_nvenc ffmpeg invocation. Writes to a `.partial`
    sibling first, then atomically renames to `out_path` on success.
    Raises TranscodeError on failure; the caller logs + reports.

    Why -map -0:v:m:attached_pic: many MKV rips carry the poster as a
    "video" stream with disposition attached_pic. Without the explicit
    exclude, ffmpeg interprets `-map 0:v` as "all video streams" and
    feeds the poster image into the NVENC encoder, which then fails
    with 'Width or height not supported with this codec' (the poster
    is rarely an exact mod-2 size).

    `probe_subtitles` is the ordered list of subtitle streams from
    ffprobe (`{"codec_name": ..., ...}`). Streams whose codec is in
    MATROSKA_SUBTITLE_COPY_CODECS are stream-copied; everything else
    is dropped (most commonly `mov_text`, the MP4-native captions
    codec that Matroska can't carry). Pass None to disable per-stream
    filtering and just `-map 0:s?` (the legacy path; only useful when
    the caller already knows the source is sub-clean).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_name(out_path.name + ".partial")
    # Clean up a stale partial from a previous crash. We never resume
    # a partial encode — NVENC's frame state is in the encoder
    # context, which is gone the moment the process exited.
    partial.unlink(missing_ok=True)

    bufsize_mbps = profile.maxrate_mbps * 2

    # Build subtitle map args. The default `-map 0:s?` would include
    # every subtitle stream and then the muxer rejects mov_text etc.
    # When we have a probe we explicitly map only the supported
    # streams. When we don't, fall back to the optional-glob.
    sub_map_args: list[str] = []
    dropped_sub_codecs: list[str] = []
    kept_sub_count = 0
    if probe_subtitles is not None:
        for i, s in enumerate(probe_subtitles):
            codec = (s.get("codec_name") or "").lower()
            if codec in MATROSKA_SUBTITLE_COPY_CODECS:
                sub_map_args.extend(["-map", f"0:s:{i}"])
                kept_sub_count += 1
            else:
                dropped_sub_codecs.append(codec or "unknown")
    else:
        sub_map_args.extend(["-map", "0:s?"])

    args = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "warning",
        # -fflags +genpts: regenerate timestamps for sources with
        # broken DTS sequences (a lot of old DVD rips). -avoid_negative_ts
        # make_zero: pin the first PTS at 0 so downstream packagers
        # don't see negative timestamps that they then refuse.
        "-fflags", "+genpts",
        "-avoid_negative_ts", "make_zero",
        "-i", str(src),
        # Stream mapping — see docstring for the attached_pic exclude.
        "-map", "0:v:0",
        "-map", "0:a?",
        *sub_map_args,
        "-map", "-0:d",
        "-map", "-0:v:m:attached_pic",
        # NVENC HEVC at the resolution-appropriate cap.
        "-c:v", "hevc_nvenc",
        "-preset", nvenc_preset,
        "-profile:v", "main",
        "-pix_fmt", "yuv420p",
        "-rc:v", "vbr",
        "-cq", str(nvenc_cq),
        "-maxrate", f"{profile.maxrate_mbps}M",
        "-bufsize", f"{bufsize_mbps}M",
        "-b_ref_mode", "middle",
        "-spatial_aq", "1",
        "-rc-lookahead", "20",
        # Audio + subs: stream-copy. The packager downstream will
        # re-encode audio to AAC stereo and the subtitles get
        # extracted (text) or skipped (image) there.
        "-c:a", "copy",
        "-c:s", "copy",
        # Force Matroska container. We write to `.partial` first
        # (atomic rename to `.mkv` on success) so ffmpeg can't infer
        # the format from the extension; without -f matroska it
        # bails with "Unable to choose an output format" before the
        # encode even starts.
        "-f", "matroska",
        str(partial),
    ]
    log.info(
        "transcoder.encode.start",
        label=log_label,
        profile=profile.label,
        maxrate_mbps=profile.maxrate_mbps,
        cq=nvenc_cq,
        preset=nvenc_preset,
        src=str(src),
        out=str(out_path),
        kept_subs=kept_sub_count,
        dropped_subs=dropped_sub_codecs or None,
    )
    t0 = time.monotonic()
    result = subprocess.run(args, capture_output=True, text=True)
    elapsed = round(time.monotonic() - t0, 1)
    if result.returncode != 0:
        # Drop the partial so a retry isn't tempted to pick it up.
        partial.unlink(missing_ok=True)
        raise TranscodeError(
            f"hevc_nvenc exited {result.returncode} after {elapsed}s: "
            f"{(result.stderr or '').strip()[-1500:] or '(no stderr)'}"
        )
    # Atomic rename — packager treats prepared.mkv's existence as
    # proof of completeness.
    os.replace(partial, out_path)
    out_size_bytes = out_path.stat().st_size
    log.info(
        "transcoder.encode.done",
        label=log_label,
        elapsed_s=elapsed,
        out_size_mb=round(out_size_bytes / 1_000_000, 1),
    )
    return {
        "elapsed_s": elapsed,
        "out_size_bytes": out_size_bytes,
        "profile_label": profile.label,
        "maxrate_mbps": profile.maxrate_mbps,
    }
