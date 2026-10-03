# transcoder

Video-prep worker for the zaentrum platform. Consumes catalog items whose
`transcode` step is pending, decides per item which video renditions the
packager needs, and encodes the ones that can't be stream-copied — with
NVENC when the host has a working NVIDIA GPU, with libx265 / libx264
when it doesn't. The downstream packager turns the result into one
CMAF/HLS tree with shaka-packager.

## How it works

1. `ffprobe` the source (container / video / audio / subtitle streams).
2. Plan the renditions from the ladder (`LADDER`, default: one rendition).
   A rung at source size whose codec the source already has is a stream
   copy; the rest are encodes.
3. If anything needs encoding, run **one** ffmpeg: decode the source
   once, `split` it per rung, write one intermediate MKV per encoded rung
   under `{packages_root}/_inbox/{itemId}/`, then `renditions.json`.
   MKV is used for the handoff because it losslessly carries subtitle
   codecs (PGS, ASS) that MP4 cannot.
4. Report the step (`done` / `not_applicable` / `failed`) back to the
   catalog API and emit `stube.catalog.item.transcoded`.

One encode runs at a time per pod. To scale, add replicas (one GPU each
on GPU hosts); do not raise the claim batch size above 1.

## Rendition contract (transcoder → packager)

Per item, in `{packages_root}/_inbox/{itemId}/`:

| File | Content |
| --- | --- |
| `prepared.mkv` | v0 when v0 is encoded: video + every audio track + every Matroska-copyable subtitle track, copied verbatim |
| `v1.mkv`, `v2.mkv`, … | lower rungs, video only |
| `renditions.json` | the manifest below, written **last** (atomic rename) |

```json
{
  "version": 1,
  "segmentSeconds": 6,
  "keyframes": "source",
  "timestampOffset": 0.021,
  "backend": "nvenc",
  "source": {"codec": "hevc", "width": 1920, "height": 1080, "frameRate": "24000/1001", "hdr": false},
  "video": [
    {"id": "v0", "label": "source", "file": null, "mode": "copy", "codec": "hevc",
     "encoder": "copy", "width": 1920, "height": 1080, "hdr": false, "bitrateBps": null,
     "maxrateBps": null, "videoStart": 0.021, "carries": ["video", "audio", "subtitles"]},
    {"id": "v1", "label": "720p", "file": "v1.mkv", "mode": "encode", "codec": "h264",
     "encoder": "h264_nvenc", "width": 1280, "height": 720, "hdr": false,
     "bitrateBps": 2400000, "maxrateBps": 3000000, "videoStart": 0.021, "carries": ["video"]}
  ]
}
```

- Rungs are ordered largest first. **v0 carries the audio and subtitle
  tracks**, from `prepared.mkv` or, when `"file": null`, from the item's
  original source (a stream copy, nothing written).
- **Keyframes** are identical on every encoded rung. `interval`: an IDR
  every `segmentSeconds`, so every segment is exactly that long.
  `source`: a stream-copied rung keeps the source's keyframes, so the
  encoded rungs get IDRs on exactly those frames and nowhere else. Either
  way the packager, segmenting at `segmentSeconds`, cuts all rungs at the
  same instants. `SEGMENT_SECONDS` must match the packager; the packager
  reads it back from this file.
- **Timestamps**: every inbox file is on the source timeline shifted by
  `timestampOffset`, frame for frame (the shift ffmpeg applies to start
  at zero, kept exact via `-enc_time_base demux`). The packager remuxes
  with `-copyts` and shifts the original by the same offset.
- **Compatibility**: no `renditions.json` and no `prepared.mkv` means
  "package the original" (an HEVC source on the default ladder), exactly
  as before. `prepared.mkv` keeps its name, so an older packager still
  packages v0 of a single-rendition item. Turn on a ladder only once the
  packager reads `renditions.json`.

## Ladder

`LADDER` is a comma-separated list of rungs:
`<source|NNNp>[:<hevc|h264>][:<maxrate>]`, e.g. `source,720p,480p` or
`source,720p:h264:2500k`.

- Default codec: `source` → HEVC (the catalog rule: HEVC sources are never
  re-encoded), scaled rungs → H.264 (decodes everywhere).
- A rung fits the source into its 16:9 box (`720p` = 1280×720, so a
  2.39:1 film becomes 1280×536). It **never upscales**: a box the source
  already fits in collapses to source size. Duplicates (same size and
  codec) are dropped.
- Caps per rung (VBV maxrate, capped CRF/CQ): H.264 1080p 6, 720p 3,
  540p 2, 480p 1.5, 360p 0.8 Mbit/s; HEVC 720p 2.5, 480p 1.2. The source
  rung keeps `NVENC_MAXRATE_1080P_MBPS` / `NVENC_MAXRATE_2160P_MBPS`.
- HDR sources: H.264 rungs are tone-mapped to SDR BT.709; HEVC rungs stay
  10-bit HDR.

**Default: empty `LADDER` = one rendition per item**, exactly as before,
so an existing install does not grow its storage. Opt in per install. The
lean choice is `LADDER=source,720p`: one H.264 720p rung next to the HEVC
top rung, for devices that cannot decode HEVC. Players do not adapt
between codecs, so serve each client only the variants it can decode.

## Encoders

`ENCODER=auto` (default) opens `hevc_nvenc` / `h264_nvenc` once at
startup and falls back per codec to `libx265` / `libx264`. The same
image runs on GPU and CPU-only nodes. Without NVENC:

- HEVC sources: stream copy (unchanged).
- Browser-friendly H.264 (8-bit 4:2:0, ≤ High): stream copy at the source
  rung instead of hours of libx265.
- Everything else (MPEG-2, VC-1, AV1, Hi10P, …): libx265.
- Scaled H.264 rungs: libx264.

`ENCODER=nvenc` fails startup if NVENC doesn't open; `ENCODER=cpu`
never probes.

## Layout

```
src/transcoder/main.py         # entry point: encoder probe, worker thread, /healthz, /readyz
src/transcoder/config.py       # env-driven config
src/transcoder/decision.py     # ladder parsing + rendition planning
src/transcoder/ffmpeg.py       # ffprobe, NVENC detection, the encode command
src/transcoder/renditions.py   # the renditions.json handoff contract
src/transcoder/katalog.py      # HTTP client for the catalog step API
src/transcoder/worker.py       # Kafka consumer loop
tests/                         # unit tests + real ffmpeg runs on lavfi clips
k8s/                           # Deployment, Service, ServiceAccount, ServiceMonitor, GrafanaDashboard
```

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `KATALOG_API_URL` | (required) | Base URL of the catalog step API |
| `OIDC_TOKEN_URL` | (required) | OIDC token endpoint (client-credentials) |
| `OIDC_CLIENT_ID` | (required) | OIDC client id |
| `OIDC_CLIENT_SECRET` | (required) | OIDC client secret |
| `PACKAGES_ROOT` | `/var/lib/katalog/packages` | Root of the `_inbox` handoff tree |
| `LADDER` | (empty) | Rendition ladder, see above; empty = one rendition |
| `ENCODER` | `auto` | `auto`, `nvenc` or `cpu` |
| `SEGMENT_SECONDS` | `6` | Forced keyframe interval = packager segment length |
| `NVENC_PRESET` | `p5` | NVENC preset |
| `NVENC_CQ` | `23` | NVENC constant quality |
| `NVENC_MAXRATE_1080P_MBPS` | `8` | Source-rung cap for HD/SD sources |
| `NVENC_MAXRATE_2160P_MBPS` | `14` | Source-rung cap for UHD sources |
| `X264_PRESET` / `X264_CRF` | `medium` / `23` | CPU H.264 rungs |
| `X265_PRESET` / `X265_CRF` | `medium` / `24` | CPU HEVC rungs |
| `KAFKA_BROKERS` | `kafka:9092` | Bootstrap brokers |

## Local development

```bash
uv sync
uv run pytest
```

Unit tests need nothing installed. `tests/test_encode_real.py` runs real
encodes on generated clips when `ffmpeg` is on `PATH` (the HDR case also
needs the `zscale` filter).

## Build the container

The image is based on `nvidia/cuda:*-runtime` and installs a pinned,
checksummed BtbN ffmpeg 7.1 static build (NVENC, libx264, libx265, zimg).
GPU nodes need the NVIDIA device plugin (`nvidia.com/gpu: 1`); CPU-only
nodes run the same image without the GPU request.

```bash
docker build -t zaentrum/transcoder .
```

Build and push the image to your own registry, then update the image reference
in `k8s/deployment.yaml` for your environment.

## License

[MPL-2.0](LICENSE).
