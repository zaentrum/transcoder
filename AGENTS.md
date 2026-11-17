# transcoder

The transcoder is the second worker of the platform's media pipeline, a Python service. It
consumes `stube.catalog.item.analyzed` from Kafka (consumer group `transcoder-workers`),
probes the title's original with ffprobe, plans the video renditions from its ladder
(`LADDER`), and encodes those the packager can't copy into the item's inbox, with NVENC on
a host with an NVIDIA GPU (decoding with NVDEC where it can) and libx265 / libx264
without one. It writes `renditions.json` beside them, reports the `transcode` step to the
catalog API (katalog-manager) and sends `stube.catalog.item.transcoded` to the packager. A
second consumer does the same for a title's extras, from
`<KAFKA_TOPIC_PREFIX>catalog.extra.queued` to `<KAFKA_TOPIC_PREFIX>catalog.extra.transcoded`.
The [README](README.md) has the rendition contract, the caps and the Dolby Vision rules.

How a change is proposed and what a pull request has to pass is in the
[guide to contributing](https://github.com/zaentrum/.github/blob/main/CONTRIBUTING.md);
what may be built is in [building from the specification](https://github.com/zaentrum/zaentrum/blob/main/AGENTS.md).

## Build

```sh
python3.10 -m venv .venv && . .venv/bin/activate
pip install '.[dev]'
python -m compileall src
```

Python 3.10 or later (`requires-python` in `pyproject.toml`; `build.yml` uses 3.10). At run
time it calls `ffmpeg` and `ffprobe` from the PATH. The image:

```sh
docker build -t zaentrum/transcoder .
```

It is based on `nvidia/cuda` (12.3 runtime, Ubuntu 22.04) and carries a pinned, checksummed
BtbN ffmpeg 7.1 static build (NVENC, libx264, libx265, libplacebo, Vulkan;
`FFMPEG_BUILD_URL` and `FFMPEG_SHA256` override it) and a Vulkan loader built from Khronos's
sources in a stage of its own (`VULKAN_SDK_TAG`). `image.yml` builds and pushes it on `main`,
on a tag `vX.Y.Z` and when started by hand, with `IMAGE_VERSION` set to the tag or the commit.

## Test

`build.yml` installs the ffmpeg the image ships, BtbN's static 7.1.5 by the Dockerfile's URL
and sha256, and runs, after the build above:

```sh
python -m pytest
```

- Everything but `tests/test_encode_real.py` runs without ffmpeg: the commands, probes and
  plans are checked against fake probes, the catalog is a fake behind an httpx mock
  transport and the broker a fake.
- `tests/test_encode_real.py` encodes generated clips with libx264 and libx265 through the
  worker and checks the handoff on disk. It skips without `ffmpeg` and `ffprobe` 7 or later
  on the PATH; single cases skip without libx265, libvpx-vp9, the `zscale` filter or
  encoders built for 10 bits, and the Dolby Vision 5 reshaping case without
  `TRANSCODER_DOVI5_SAMPLE` naming a profile 5 file, `libplacebo` and a Vulkan device (a
  software one will do); the three other Dolby Vision 5 cases make their own clip and need
  only a libx265 built for 10 bits.

## Run it

The transcoder runs inside the platform: the operator's chart starts it when the `Zaentrum`
resource sets `features.pipeline`, as on the reference demo ([how it is deployed](https://github.com/zaentrum/zaentrum/blob/main/docs/reference-demo.md)).
`spec.pipeline.encoder` says where it encodes: `gpu`, the default, with NVENC, the chart
asking for `nvidia.com/gpu` and a node with an NVIDIA GPU; `cpu` with libx264 and libx265 on
any node (`ENCODER=cpu`).
It needs the catalog API, an OIDC client's id and secret (`OIDC_CLIENT_ID`,
`OIDC_CLIENT_SECRET`, for the client-credentials grant; the platform's chart gives analyzer,
packager and transcoder one shared client), a Kafka broker and the library share at
`/var/lib/katalog`; without a GPU it encodes on the CPU. Started by hand against those:

```sh
export KATALOG_API_URL=http://localhost:8081 OIDC_TOKEN_URL=https://idp.example.org/token \
  OIDC_CLIENT_ID=transcoder OIDC_CLIENT_SECRET=... KAFKA_BROKERS=localhost:9092 \
  ENCODER=cpu DECODER=cpu
python -m transcoder.main
```

`KATALOG_API_URL` and the three `OIDC_` variables are required. `ENCODER` (`auto`, `nvenc`,
`cpu`) and `DECODER` (`auto`, `cpu`) pick the encoders and the decode; the README's
"Configuration" lists the ladders, the caps and the rest, `config.py` and `kafka.py` the
Kafka settings. Port 8080 serves `/healthz` (503 once an ffmpeg would not die after its
kill) and `/readyz` (503 while the item consumer or the extras consumer is not running, as
when NVENC failed its startup self-test and neither started). Without a broker or a
catalog it starts all the same and logs each failed connection. No mock server or recorded
answers ship; the tests' fake catalog and broker stand in for them.

## What the tests leave unchecked

- No test runs on a GPU. NVENC, NVDEC, `scale_cuda`, the startup probes
  (`transcoder.decoder.probe`, `transcoder.nvenc.selftest`) and the fallback of a failed
  GPU run to the CPU are checked as commands and against fakes (`tests/test_gpu_decode.py`,
  `tests/test_encode_command.py`). The README's GPU figures were measured by hand.
- No profile 5 sample ships, so the Dolby Vision 5 reshaping on a Vulkan device is not run.
- On `build.yml`'s runner `tests/test_encode_real.py` encodes with the image's ffmpeg build,
  on no GPU.
- No test talks to a real broker or catalog, or fetches an OIDC token.
- The image (its ffmpeg 7.1, the Vulkan loader, the NVIDIA runtime's libraries) is built
  only by `image.yml`, on `main`, on tags and when started by hand; no test runs in it.
