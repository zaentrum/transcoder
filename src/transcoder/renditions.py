"""The transcoder -> packager handoff contract (`renditions.json`).

Per item the transcoder leaves, in its inbox — the worker record's
`library.inboxDir` on the v2 library layout, `<work root>/inbox/{itemId}/`,
and `{PACKAGES_ROOT}/_inbox/{itemId}/` on the legacy one:

    prepared.mkv     v0 when v0 is encoded: video + every audio track +
                     every Matroska-copyable subtitle track, MP4's text
                     captions as SubRip (legacy name, so an older packager
                     still finds it)
    v1.mkv, v2.mkv   lower rungs, video only
    renditions.json  this manifest, written LAST (atomic rename) — its
                     presence means every file it lists is complete

A rung with `"file": null` is a stream copy of the item's original
source; the packager reads `item.path` for it. A copy with a file of its
own (`"mode": "copy"`, `"encoder": "copy"`) is the original's video as the
transcoder wrote it, as it is but for what it drops: a Dolby Vision 7
source's enhancement layer and RPU (RungPlan.strip_dovi). v0 always carries the
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

`scan` is what the package's video is made of as far as fields go, for
the package's record (ADR-0015: the master is progressive; no word lets an
original go, the deletion gate keeps one marked as fields or found to hold
them):

    "scan": {"found": "fields", "judgedBy": "pictures", "fieldOrder": "tb",
             "done": "deinterlaced-frame", "frameRate": "25/1",
             "order": "tff", "orderBy": "mark", "orderSeen": true,
             "filter": "bwdif=mode=send_frame:parity=tff:deint=all",
             "idet": {"frames": 1000, "tff": 980, "bff": 0, "progressive": 0,
                      "undetermined": 20, "repeatedTop": 0, "repeatedBottom": 0,
                      "matchedCombed": 560, "matchedDetermined": 1000,
                      "matchedRun": 14, "cadenceBlocks": 0, "otherBlocks": 200,
                      "blindBlocks": 0,
                      "windows": {"film": 0, "video": 5, "whole": 0, "unsure": 0,
                                  "detail": 0}},
             "verified": null}

  * `found` (decision.FIELDS_FOUND): "fields" (video), "film", "telecine",
    "mixed", "progressive", "unexamined", "unknown" (the pictures could not
    tell) or "field-pictures" (HEVC coded as field pictures), as `judgedBy`
    says:
    "pictures" (idet over a sample, decision.judge_fields), "mark" (the
    source's field order alone), "bitstream" (its parameter sets) or
    "failed" (nothing read); `fieldOrder` is that mark (null for none).
    "progressive" only where the whole title was read through and nothing
    was found (a title it found fields in is made again, found "mixed"; a
    3:2 film with parts decimate would thin, "film"); whole frames on a
    partial basis (the sample alone, a copy or a read-through that did not
    reach the end, a source not examined) are "unexamined"
    (decision.FOUND_UNEXAMINED); a source left as it is with fields in one
    order too few to make it over is "mixed".
  * `done`, what was done (decision.SCAN_DONE): "inverse-telecine" (a film
    at its own frames), "progressive" (whole frames, as found),
    "deinterlaced-field" (video at the field rate, in the order idet saw),
    "deinterlaced-frame" (video at the frame rate: every second moment
    dropped), "field-matched" (film and video mixed, or a film not
    decimated or not sure), "fields-kept" (HEVC field pictures),
    "fields-left" (fields too few to make a source marked progressive over,
    left as they are), "unexamined" (nothing told, or nothing read: the
    safe chain) or "none" (marked progressive, nothing found).
    inverse-telecine, progressive and deinterlaced-field
    (decision.SCAN_RELEASES) say the package lost nothing to its fields:
    each stands only where the whole title, read through as it was encoded,
    bears it out; elsewhere the title is made again with the safe chain,
    and `done` says that chain's word.
  * `order`: the field order the filters took ("tff" or "bff"), `orderBy`
    what named it ("pictures", "mark" or "default"), `orderSeen` whether
    idet saw it so.
  * `frameRate`: the package's video frame rate: the encoded rungs' (a
    deinterlace's output rate, 50/1 at the field rate of a 25-frame
    source, 25/1 at its frame rate), a copy's the source's.
    `filter`: the filters its pictures took, null for none (an inverse
    telecine's graph also carries the closed captions around decimate:
    ffmpeg.CARRY_CAPTIONS_GRAPH; `sidedata=mode=delete:type=A53_CC` at
    its end where they are left out: ffmpeg.captions_at_field_rate).
    `idet`: the counts judge_fields read of the sample (as the frames are,
    after fieldmatch in `order`, the 3:2 blocks, the windows), null when no
    examination read a frame.
  * `verified`: what the read-through of the whole title found, null where
    none ran: `ok`, `why` not, what the sample said (`sampleDone`,
    `sampleFound`), and its counts (`droppedDistinct`: frames decimate
    dropped that were no near-duplicate; `uncertainCycles`: cycles whose
    drop was no clear duplicate; `seen`: frames combed in runs no refusal
    took up; `blindRun`: the longest stretch, not black, idet could not
    see; `complete`: the whole title read). `ok` false beside a `done`: the
    title was made again with the chain `done` names (its `verified` is the
    first encode's, which refused it), or left as it was made (not read to
    the end, or found "mixed" or "unexamined" without another chain).

No renditions.json at all (the default single rendition when the source
needs no work) means: package the original source, exactly as before. A
source marked as fields that is copied all the same (its pictures are
whole frames) has one, every rung a copy (`"file": null`), so that its
`scan` reaches the package's record; so has a copy of HEVC field pictures
(`fields-kept`), found `field-pictures`.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .decision import Encoders, Plan
from .ffmpeg import RungResult, fields_chain

CONTRACT_VERSION = 1
RENDITIONS_FILE = "renditions.json"


def scan_record(plan: Plan) -> dict[str, Any]:
    """The contract's `scan` (module doc): what the package's video is made
    of as far as fields go, and what said so."""
    f = plan.fields
    encoded = any(r.mode == "encode" for r in plan.rungs)
    idet = None
    if f.frames:
        idet = {"frames": f.frames, "tff": f.tff, "bff": f.bff, "progressive": f.progressive,
                "undetermined": f.undetermined, "repeatedTop": f.repeated_top,
                "repeatedBottom": f.repeated_bottom,
                "matchedCombed": f.matched_combed, "matchedDetermined": f.matched_determined,
                "matchedRun": f.matched_run, "cadenceBlocks": f.cadence_blocks,
                "otherBlocks": f.other_blocks, "blindBlocks": f.blind_blocks,
                "windows": {"film": f.film_windows, "video": f.video_windows,
                            "whole": f.whole_windows, "unsure": f.unsure_windows,
                            "detail": f.detail_windows}}
    return {
        "found": plan.found_word,
        "judgedBy": f.by,
        "fieldOrder": f.mark or None,
        "done": plan.scan_done,
        "order": f.order,
        "orderBy": f.order_by,
        "orderSeen": f.order_seen,
        "frameRate": plan.frame_rate if encoded else plan.source.frame_rate,
        "filter": fields_chain(plan) if encoded else None,
        "idet": idet,
        "verified": _verified(plan),
    }


def _verified(plan: Plan) -> dict[str, Any] | None:
    """The whole title's read-through (Plan.verification), null where the
    sample's word needed none."""
    v = plan.verification
    if v is None:
        return None
    return {"ok": v.ok, "why": v.why or None, "sampleDone": v.sample_done or plan.sample_done,
            "sampleFound": v.sample_found or plan.fields.found.replace("_", "-"),
            "frames": v.frames, "chunksInFields": v.chunks_in_fields,
            "otherOrder": v.other_order, "offCadence": v.off_cadence, "repeats": v.repeats,
            "matchedUnsure": v.matched_unsure, "combedOut": v.combed_out,
            "combedOutRun": v.combed_out_run, "droppedDistinct": v.dropped_distinct,
            "uncertainCycles": v.uncertain_cycles, "seen": v.seen, "blindRun": v.blind_run,
            "complete": v.complete}


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
        "scan": scan_record(plan),
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
