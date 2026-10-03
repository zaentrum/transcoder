"""Per-item video-prep worker (NVENC, or libx265/libx264 without a GPU).

Consumes `stube.catalog.item.analyzed`, plans the video renditions the
packager needs (the `LADDER`, default: one), and encodes the ones that
can't be stream-copied in a single ffmpeg run into
`/var/lib/katalog/packages/_inbox/{itemId}/` — `prepared.mkv` (the top
rung, with every audio and subtitle track), `v1.mkv`... (lower rungs,
video only) and `renditions.json` (the handoff contract, see
`transcoder.renditions`).

Why MKV and not MP4 for the handoff: source rips routinely carry
subtitle codecs that MP4 can't contain at all (PGS bitmap, ASS rich
text). Once the original source files are pruned post-ingest, the
intermediate is the *only* place those tracks survive between the
encode and the shaka-packager run. MKV keeps every subtitle codec the
source had, and the packager remuxes it to MP4 for shaka-packager.

Outcomes reported back via `PUT /api/analyze/items/{id}/steps/transcode`:
  * status=`not_applicable` — nothing to encode (an HEVC source on the
    default ladder, or a browser-friendly H.264 source on a host without
    NVENC); no handoff written. The packager reads the original source.
  * status=`done` — rungs encoded; the packager picks up the handoff.
  * status=`failed` — probe / encode failed; package step is left
    pending so a manual retry (or operator action) can recover.

In every successful outcome (`done` or `not_applicable`) the Java side
flips `package=pending` so the packager queue picks the item up next.

Runs in its own pod (katalog-transcoder): one NVIDIA GPU per replica on
GPU hosts, or CPU-only with the same image.
"""
