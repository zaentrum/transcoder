# transcoder

GPU video-prep worker for the zaentrum platform. Claims catalog items whose
`transcode` step is pending, decides whether the source is already HEVC (skip)
or needs re-encoding, and — when an encode is needed — runs ffmpeg with
`hevc_nvenc` to produce an intermediate MKV that the downstream packager picks
up and runs through shaka-packager.

## How it works

The worker polls the catalog claim API for items in `transcode=pending`, then
for each item:

1. `ffprobe` the source to read container / video / audio / subtitle streams.
2. Decide: source video codec already HEVC → skip; anything else (H.264, AV1,
   MPEG-2, ...) → encode to HEVC NVENC.
3. On encode, run a single `hevc_nvenc` ffmpeg invocation that writes an
   intermediate MKV under `{packages_root}/_inbox/{itemId}/prepared.mkv`. MKV
   is used for the handoff because it can losslessly carry subtitle codecs
   (PGS bitmap, ASS rich text) that MP4 cannot.
4. Report the terminal step status (`done` / `not_applicable` / `failed`) back
   to the catalog API, which promotes `package=pending` so the packager queue
   picks the item up next.

One GPU encode runs at a time per pod — a single card saturates on one
`hevc_nvenc` encode. To scale, add Deployment replicas (one GPU each); do not
raise the claim batch size above 1.

## Layout

```
src/transcoder/__init__.py     # package docstring (worker overview)
src/transcoder/main.py         # entry point: worker thread + FastAPI /healthz, /readyz
src/transcoder/config.py       # env-driven config (NVENC knobs, claim tuning)
src/transcoder/decision.py     # skip-vs-encode decision from an ffprobe payload
src/transcoder/ffmpeg.py       # ffprobe + hevc_nvenc encode invocations
src/transcoder/katalog.py      # HTTP client for the catalog claim/step API
src/transcoder/worker.py       # claim-poller worker loop
tests/test_worker_shape.py     # shape tests (no ffmpeg/CUDA required)
k8s/                           # Deployment, Service, ServiceAccount, ServiceMonitor, GrafanaDashboard
Dockerfile
```

## Configuration

All configuration is read from environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `KATALOG_API_URL` | (required) | Base URL of the catalog claim/step API |
| `OIDC_TOKEN_URL` | (required) | OIDC token endpoint (client-credentials) |
| `OIDC_CLIENT_ID` | (required) | OIDC client id |
| `OIDC_CLIENT_SECRET` | (required) | OIDC client secret |
| `CLAIM_BATCH_SIZE` | `1` | Items claimed per cycle (keep at 1) |
| `IDLE_SLEEP_SECONDS` | `30` | Re-poll interval when the queue is empty |
| `ERROR_SLEEP_SECONDS` | `60` | Back-off after a claim failure |
| `PACKAGES_ROOT` | `/var/lib/katalog/packages` | Root of the `_inbox` handoff tree |
| `NVENC_PRESET` | `p5` | NVENC preset |
| `NVENC_CQ` | `23` | Target constant quality |
| `NVENC_MAXRATE_1080P_MBPS` | `8` | Rate cap for HD/SD sources |
| `NVENC_MAXRATE_2160P_MBPS` | `14` | Rate cap for UHD sources |

## Local development

```bash
uv sync
uv run pytest
```

The shape tests run without ffmpeg or CUDA installed; the full encode path is
exercised on a GPU-enabled host.

## Build the container

The image is based on `nvidia/cuda:*-runtime` and installs a prebuilt
NVENC-enabled ffmpeg static build (Ubuntu's stock ffmpeg ships without
`--enable-nvenc`). The container needs a GPU at runtime via the NVIDIA device
plugin (`nvidia.com/gpu: 1`).

```bash
docker build -t zaentrum/transcoder .
```

Build and push the image to your own registry, then update the image reference
in `k8s/deployment.yaml` for your environment.

## License

[MPL-2.0](LICENSE).
