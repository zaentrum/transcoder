"""Skip-vs-encode decision for one source file.

User rule (locked):
  * Source video codec is already HEVC  -> SKIP (no re-encode).
  * Anything else (H.264, AV1, MPEG-2, ...) -> ENCODE to HEVC NVENC.

Bitrate / size / hardware-tier checks are deliberately NOT here — the
goal is uniform codec across the catalog (`hev1.1.6.L120.B0`), not
re-encoding everything to a smaller file. Per-resolution bitrate caps
only kick in on the encode path so we don't ship a 50 Mbps remux of a
2160p Blu-ray straight into HLS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# Codec names ffprobe reports for HEVC content. `hevc` is the canonical
# one; older mux toolchains occasionally tag it as `h265`. Both pass
# through the packager untouched, so both count as "no re-encode".
HEVC_CODEC_NAMES = {"hevc", "h265"}


@dataclass(frozen=True)
class Decision:
    """Outcome of `_should_transcode_video(probe)`."""
    skip: bool
    reason: str
    source_codec: str
    width: int
    height: int


def decide(probe: dict[str, Any]) -> Decision:
    """Take a parsed ffprobe payload (shape from `transcoder.ffmpeg.ffprobe`)
    and return the skip-vs-encode decision plus a one-line reason for
    the audit row.

    Defensive: a missing/empty video stream -> encode (so we go through
    NVENC and produce a known-good prepared.mkv) rather than skip with
    bad data. If the file really is video-less, ffmpeg will fail
    loudly downstream.
    """
    video = probe.get("video") or {}
    codec = (video.get("codec_name") or "").lower()
    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)

    if not codec:
        return Decision(
            skip=False,
            reason="no_video_stream_in_probe",
            source_codec="",
            width=width,
            height=height,
        )

    if codec in HEVC_CODEC_NAMES:
        return Decision(
            skip=True,
            reason=f"source_already_hevc:{codec}",
            source_codec=codec,
            width=width,
            height=height,
        )

    return Decision(
        skip=False,
        reason=f"non_hevc_source:{codec}",
        source_codec=codec,
        width=width,
        height=height,
    )
