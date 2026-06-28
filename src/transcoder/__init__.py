"""Per-item GPU video-prep worker (NVENC HEVC).

Claims items whose `transcode` step is pending from katalog-app's
`/api/analyze/claim?pass=transcoder` endpoint, decides whether the
source is already HEVC (skip) or needs re-encoding, and — when an
encode is needed — runs ffmpeg with `hevc_nvenc` to produce an
intermediate MKV at `/var/lib/katalog/packages/_inbox/{itemId}/prepared.mkv`.

Why MKV and not MP4 for the handoff: source rips routinely carry
subtitle codecs that MP4 can't contain at all (PGS bitmap, ASS rich
text). Once the original source files are pruned post-ingest, the
intermediate is the *only* place those tracks survive between the
GPU pass and the shaka-packager run. MKV keeps every subtitle codec
the source had, and shaka-package on the other side reads MKV fine
(it gets remuxed to MP4 inside the packager's `_prepare_source`).

Outcomes reported back via `PUT /api/analyze/items/{id}/steps/transcode`:
  * status=`not_applicable` — source video codec is already HEVC; no
    encode performed, no prepared.mkv written. The packager will read
    the original source path directly.
  * status=`done` — NVENC produced a prepared.mkv. The packager will
    pick it up via the _inbox handoff.
  * status=`failed` — probe / encode failed; package step is left
    pending so a manual retry (or operator action) can recover.

In every successful outcome (`done` or `not_applicable`) the Java side
flips `package=pending` so the packager queue picks the item up next.

Runs in its own pod (katalog-transcoder) on a GPU-enabled host: one
NVIDIA GPU per replica, schedule onto labelled nodes only.
"""
