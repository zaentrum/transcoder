"""The transcoder -> packager handoff contract (`renditions.json`).

Per item the transcoder leaves, under `{PACKAGES_ROOT}/_inbox/{itemId}/`:

    prepared.mkv     v0 when v0 is encoded: video + every audio track +
                     every Matroska-copyable subtitle track (legacy name,
                     so an older packager still finds it)
    v1.mkv, v2.mkv   lower rungs, video only
    renditions.json  this manifest, written LAST (atomic rename) — its
                     presence means every file it lists is complete

A rung with `"file": null` is a stream copy of the item's original
source; the packager reads `item.path` for it. v0 always carries the
audio and subtitle tracks (`"carries"`), from `prepared.mkv` or — when
v0 is a copy — from the original source.

`segmentSeconds` + `keyframes` tell the packager how the encoded rungs
place keyframes: "interval" = an IDR every segmentSeconds on every rung;
"source" = IDRs exactly where the stream-copied rung has keyframes. The
packager must segment with the same segmentSeconds.

`source` is the source as the transcoder probed it: ffprobe's codec name,
the coded size, the frame rate, HDR, and the container's duration
(`durationMs`) and overall bit rate (`bitRate`, bit/s; null when ffprobe
has none). The packager forwards it to the catalog, which keeps it as
the title's source asset.

`timestampOffset` is the shift (seconds) ffmpeg applied to every source
timestamp before encoding (minus the source's earliest timestamp — e.g.
+0.021 for an AAC-primed source whose audio starts at -0.021 s). Every
file in the inbox is on that shifted timeline, frame for frame. The
packager keeps it (`-copyts`) and puts the ORIGINAL on it too (`-copyts
-output_ts_offset <timestampOffset>`), so a stream-copied rung, the
encoded rungs and the audio all carry identical timestamps for identical
frames regardless of which ffmpeg version the packager runs.

No renditions.json at all (the default single rendition when the source
needs no work) means: package the original source, exactly as before.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .decision import Encoders, Plan
from .ffmpeg import RungResult

CONTRACT_VERSION = 1
RENDITIONS_FILE = "renditions.json"


def build_contract(
    item_id: str,
    plan: Plan,
    results: list[RungResult],
    encoders: Encoders,
    *,
    duration_ms: int,
    source_start_time: float = 0.0,
    source_bit_rate: int | None = None,
) -> dict[str, Any]:
    by_id = {r.rung.id: r for r in results}
    seconds = duration_ms / 1000 if duration_ms > 0 else 0
    offset = -source_start_time
    video: list[dict[str, Any]] = []
    for rung in plan.rungs:
        res = by_id.get(rung.id)
        if res is not None:
            width, height, size = res.width, res.height, res.size_bytes
            bitrate = round(size * 8 / seconds) if seconds else None
            video_start = res.video_start
        else:
            width, height, size = rung.width, rung.height, None
            bitrate = plan.source.bit_rate
            video_start = (
                plan.source.start_time + offset if plan.source.start_time is not None else None
            )
        video.append({
            "id": rung.id,
            "label": rung.name,
            "file": rung.file,
            "mode": rung.mode,
            "codec": rung.codec,
            "encoder": rung.encoder,
            "width": width,
            "height": height,
            # HDR survives on copies and on 10-bit HEVC encodes; H.264
            # rungs of an HDR source are tone-mapped to SDR BT.709.
            "hdr": plan.source.hdr and not rung.tonemap,
            # File bitrate (size x 8 / duration). For v0 this includes the
            # audio + subtitle tracks it carries; the source's own figure
            # (often absent in MKV) for a stream copy. Informational — the
            # packager measures the real segment bitrates itself.
            "bitrateBps": bitrate,
            "maxrateBps": rung.maxrate_bps,
            "bytes": size,
            # First video timestamp on the shared (shifted) timeline.
            "videoStart": round(video_start, 6) if video_start is not None else None,
            "carries": ["video", "audio", "subtitles"] if rung.id == "v0" else ["video"],
        })
    return {
        "version": CONTRACT_VERSION,
        "itemId": item_id,
        "createdAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "segmentSeconds": plan.segment_seconds,
        "keyframes": plan.keyframes,
        "timestampOffset": round(offset, 6),
        "backend": encoders.backend,
        "source": {
            "codec": plan.source.codec,
            "width": plan.source.width,
            "height": plan.source.height,
            "frameRate": plan.source.frame_rate,
            "hdr": plan.source.hdr,
            "durationMs": duration_ms if duration_ms > 0 else None,
            "bitRate": source_bit_rate,
        },
        "video": video,
    }


def write_contract(inbox: Path, contract: dict[str, Any]) -> Path:
    """Atomic write: `.partial` then rename, so the packager never reads
    a half-written manifest."""
    inbox.mkdir(parents=True, exist_ok=True)
    final = inbox / RENDITIONS_FILE
    partial = final.with_name(final.name + ".partial")
    partial.write_text(json.dumps(contract, indent=2) + "\n")
    os.replace(partial, final)
    return final
