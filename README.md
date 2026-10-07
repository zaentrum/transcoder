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
   copy (HEVC only when every HEVC device decodes it, see
   [Sources](#sources), and when it is within its [cap](#caps)); the rest
   are encodes, capped. A source no rung may copy or encode as it is
   (Dolby Vision profile 5 or 7) fails the step.
3. If anything needs encoding, run **one** ffmpeg: decode the source
   once, `split` it per rung, write one intermediate MKV per encoded rung
   into the item's [inbox](#inbox), then `renditions.json`.
   MKV is used for the handoff because it losslessly carries subtitle
   codecs (PGS, ASS) that MP4 cannot.
4. Report the step (`done` / `not_applicable` / `failed`) back to the
   catalog API and emit `stube.catalog.item.transcoded`.

One item encode runs at a time per pod, beside at most one extra's (a
trailer, see [Extras](#extras)). To scale, add replicas (one GPU each
on GPU hosts); do not raise the claim batch size above 1.

The encode runs on the Kafka poll thread, as in the analyzer, and the
consumer's `max.poll.interval.ms` is 24 h (librdkafka's maximum): an
encode takes minutes on a GPU but can take hours on a CPU, and a worker
that polls too late loses its partition — the broker would hand the
item, not committed yet, to another replica, which would encode it a
second time into the same inbox. It is the catalog's reaper, not Kafka,
that decides when a silent run is dead. A rebalance (a replica joining
or leaving) waits until every busy replica has finished its item.

## Redelivered and retried events

The transcode step is finished when it is `done`, `not_applicable` (the
source needs no encode) or `skipped`. An `analyzed` event for an item
whose step has finished — a redelivery after a crash, a duplicate — runs
nothing: no probe, no encode, no step write. It only passes the chain on
(`transcoded`), so a packager that missed the first event recovers.

The catalog retries a failed or silent transcode by sending its
`analyzed` event again, marked `"status": "retry"`. A retry whose step
has finished since — a long run the catalog's reaper took for dead that
reported done after all — is acked with one log line
(`transcoder.retry.already_finished`) and nothing else: the run that
finished the step passed the chain on itself. Any other retry runs the
transcode as usual.

## Inbox

Where the handoff goes depends on the catalog's library layout, which
the worker record says:

| Layout | Worker record | Item | Extra |
| --- | --- | --- | --- |
| v2 | carries `library` | `library.inboxDir`, `<work root>/inbox/<itemId>/` | `library.inboxDir`, `<work root>/inbox/extra-<extraId>/` |
| legacy | no `library` (or `null`) | `{PACKAGES_ROOT}/_inbox/<itemId>/` | `{PACKAGES_ROOT}/_inbox/extra-<extraId>/` |

On the v2 layout the catalog decides the path and the packager reads it
back from the same record; the transcoder computes none. It empties the
inbox before it writes, so it takes `inboxDir` only in the shape the
catalog's contract gives it: an absolute path without `..` that ends in
`inbox/<itemId>` (`inbox/extra-<extraId>` for an extra). A v2 record
without such an inbox, or whose `library.contract` is not `1`, fails the
transcode step with the reason and sends no event. It never falls back
to the legacy inbox, where a v2 packager would not look. The inbox must
be writable in the pod: mount the whole share once at `/var/lib/katalog`,
as the platform chart does.

## Rendition contract (transcoder → packager)

Per item, in its [inbox](#inbox):

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
  "source": {"codec": "hevc", "width": 1920, "height": 1080, "frameRate": "24000/1001", "hdr": false,
             "durationMs": 5400000, "bitRate": 9800000},
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
- `source` is the source as probed: ffprobe's codec name, the coded size,
  and the container's duration and overall bit rate (bit/s, `null` when
  ffprobe has none). The packager forwards it to the catalog, which keeps
  it as the title's source asset. `durationMs` and `bitRate` are new; a
  packager that doesn't know them ignores them.
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
  "package the original" (a copyable HEVC source on the default ladder),
  exactly as before. `prepared.mkv` keeps its name, so an older packager still
  packages v0 of a single-rendition item. Turn on a ladder only once the
  packager reads `renditions.json`.

## Ladder

`LADDER` is a comma-separated list of rungs:
`<source|NNNp>[:<hevc|h264>][:<maxrate>]`, e.g. `source,720p,480p` or
`source,720p:h264:2500k`.

- Default codec: `source` → HEVC (the catalog rule: HEVC sources are never
  re-encoded), scaled rungs → H.264 (decodes everywhere).
- A codec the ladder names is a requirement. A source-size HEVC rung
  whose codec is left to the default lets the CPU rule
  ([Encoders](#encoders)) pass a browser-friendly H.264 source through;
  a named one (`source:hevc`) never does.
- A rung fits the source into its 16:9 box (`720p` = 1280×720, so a
  2.39:1 film becomes 1280×536). It **never upscales**: a box the source
  already fits in collapses to source size. Duplicates (same size and
  codec) are dropped.
- Caps per rung (VBV maxrate, capped CRF/CQ): H.264 1080p 6, 720p 3,
  540p 2, 480p 1.5, 360p 0.8 Mbit/s; HEVC 720p 2.5, 480p 1.2. A
  source-size HEVC encode takes the cap of the item's kind and bucket
  ([Caps](#caps)); a maxrate the ladder names wins over both.
- HDR sources: H.264 rungs are tone-mapped to SDR BT.709; HEVC rungs stay
  10-bit HDR. Any source above 8 bits gets Main 10 HEVC rungs
  ([Sources](#sources)).

**Default: empty `LADDER` = one rendition per item**, exactly as before,
so an existing install does not grow its storage. Opt in per install. The
lean choice is `LADDER=source,720p`: one H.264 720p rung next to the HEVC
top rung, for devices that cannot decode HEVC. Players do not adapt
between codecs, so serve each client only the variants it can decode.

### HEVC only

`LADDER=source:hevc` makes every package HEVC only: one rendition at the
source's own size, and no H.264 rung.

| Source | The package's video |
| --- | --- |
| HEVC Main or Main 10, 4:2:0 (SDR, HDR, Dolby Vision 8.x), within its [cap](#caps) | the source's, copied: `not_applicable`, no handoff |
| the same, above its cap (or a movie file above 15 GiB) | one HEVC encode at its size, maxrate = the cap |
| HEVC 4:2:2, 4:4:4 or 12-bit | one HEVC encode at its size, 4:2:0 |
| H.264 of any profile, browser-friendly included | one HEVC encode at its size |
| VP9, AV1, MPEG-2, VC-1, … | one HEVC encode at its size |
| Dolby Vision 5 or 7 | none: the step fails ([Sources](#sources)) |

Every encode in this table is capped at the cap of the item's kind and
bucket, and keeps the closed captions carried in the video.

The encode is `hevc_nvenc`, or `libx265` on a host without NVENC. With
NVENC, `source:hevc` plans exactly as an empty `LADDER`; without it, the
named codec keeps the CPU rule from passing H.264 through, so a CPU host
pays a libx265 encode (hours) for every H.264 title. A source above 8
bits is encoded Main 10, HDR as HDR and SDR as SDR; an 8-bit one Main
([Sources](#sources)). A device that cannot decode HEVC relies on the
streaming side transcoding the package on the fly.

On the platform chart, `pipeline.ladder: "source:hevc"` sets it (the
chart passes `pipeline.ladder` as `LADDER`). The extras' value is the
same, `EXTRA_LADDER=source:hevc`; [Extras](#extras) says when to set it.

## Sources

What becomes of a source's video, on every ladder:

- **HEVC is copied only when every HEVC device decodes it**: ffprobe's
  `Main` or `Main 10`, 4:2:0, at most 10 bits. Any other HEVC — Rext's
  4:2:2 or 4:4:4, 12-bit, SCC — is re-encoded to 4:2:0 at its own size
  (`hevc_not_copyable:4:2:2` and the like in the plan).
- **An HEVC encode keeps the bit depth.** A source with more than 8 bits
  (High 10 H.264, 10-bit VP9 or AV1, 12-bit HEVC, ProRes, …) is encoded
  Main 10, `yuv420p10le` into libx265 and `p010le` into NVENC: HDR stays
  HDR, and SDR stays SDR with its colour tags (BT.2020 SDR stays BT.2020
  SDR: the encode writes the tags of the decoded frames). An 8-bit source
  is encoded Main. Originals are deleted once packaged, so a 10-bit source
  dropped to 8 bits would be lost for good. H.264 rungs are 8-bit; an HDR
  source's are tone-mapped.
- **Dolby Vision** is packaged as its base layer when other devices play
  that, and refused when they don't:

| Dolby Vision | Base layer | What happens |
| --- | --- | --- |
| 8.1, 8.2, 8.4 | HDR10, SDR, HLG | the base layer's rules: HEVC Main 10 is copied, as before |
| 4, 9, 10.1, 10.2, 10.4 | SDR, HDR10 or HLG | the base layer's rules |
| 5 | none (IPT-PQ-c2 pictures) | the step fails |
| 7 | HDR10, with an enhancement layer | the step fails |
| a base layer of compatibility 0 (10.0, …), or a `dvh1` / `dvhe` / `dav1` sample entry without a record | none | the step fails |

A refused source fails the transcode step with the reason, e.g. `Dolby
Vision profile 5 needs a tone-mapping encode; kept the original`.
Nothing is encoded or handed off (an older run's handoff is removed),
and no event goes. The title stays unpackaged, so its original is never
retired: it waits for a tone-mapping encode. The profile comes from the
stream's configuration record (ffprobe's side data `DOVI configuration
record`), else from its sample entry.

- **Closed captions** (EIA/CEA-608/708) carried in the video, as A53 SEI
  or user data, stay in every encode: each encoded rung is given
  `-a53cc 1`, which libx265 needs since ffmpeg 7.1 (it defaults to off).
  A copy keeps them as they are.

## Caps

A source the packager would copy, HEVC Main or Main 10 4:2:0 up to 10
bits, is copied only within its cap. Above the cap it is encoded **once
at its own size**: NVENC, else libx265, its bit depth kept (Main 10 above
8 bits, HDR as HDR), the Dolby Vision rules first. The VBV maxrate is the
cap and the bufsize twice it. Every source-size HEVC encode takes the
same cap as its maxrate (an H.264 source on `source:hevc`, a re-encoded
Rext, the capped copy).

The cap goes by the item's kind and the height of its tallest video
stream (cover art aside). The 2160 bucket starts at 2000 lines, so a
3840×1600 scope picture is in the 1080 one:

| Kind | 1080 bucket | 2160 bucket | File size |
| --- | --- | --- | --- |
| movie, extra | 8 Mbit/s | 14 Mbit/s | above 15 GiB: encoded, whatever its bit rate |
| episode | 6 Mbit/s | 8 Mbit/s | no rule |

A source is above its cap when its video's bit rate is strictly above
it. That is the video stream's own rate, read from, in order:

1. ffprobe's `bit_rate` of the video stream (MP4, and TS when known);
2. the Matroska statistics tag, `BPS` or `BPS-eng`;
3. the file's size × 8 ÷ its duration, less the audio streams' rates
   (each its `bit_rate`, else its `BPS` tag; one with neither counts 0,
   which errs high);
4. else unknown: the bit rate rule breaks nothing.

The step's details and the log line `transcoder.item.cap` say which one
it was (`rate=12.7Mbps(size-minus-audio) cap=movie-1080:8Mbps
over=bitrate`). The H.264 that a CPU host passes through, and the H.264
rungs, are not capped this way: they keep the ladder's table.

`CAP_MOVIE_1080_MBPS`, `CAP_MOVIE_2160_MBPS`, `CAP_MOVIE_MAX_GIB`,
`CAP_EPISODE_1080_MBPS` and `CAP_EPISODE_2160_MBPS` set the table; `0`
switches a rule off. An encode whose cap is off takes 8 or 14 Mbit/s by
bucket. `NVENC_MAXRATE_1080P_MBPS` / `NVENC_MAXRATE_2160P_MBPS`, when
set, override the maxrate of every source-size HEVC encode in their
bucket, whatever the kind, but not whether a copy is kept. Leave them
unset to let the caps set it.

## Encoders

`ENCODER=auto` (default) opens `hevc_nvenc` / `h264_nvenc` once at
startup and falls back per codec to `libx265` / `libx264`. The same
image runs on GPU and CPU-only nodes. Without NVENC:

- HEVC Main or Main 10, 4:2:0, up to 10-bit, within its cap: stream copy
  (unchanged); above its cap, or any other HEVC: libx265
  ([Sources](#sources), [Caps](#caps)).
- Browser-friendly H.264 (8-bit 4:2:0, ≤ High): stream copy at the source
  rung instead of hours of libx265 — the CPU rule — unless the ladder
  names the rung's codec ([HEVC only](#hevc-only)): then libx265.
- Everything else (MPEG-2, VC-1, AV1, Hi10P, …): libx265.
- Scaled H.264 rungs: libx264.

`ENCODER=nvenc` fails startup if NVENC doesn't open; `ENCODER=cpu`
never probes.

## Extras

The extras of a title (trailers, teasers, featurettes, making-ofs) are
catalog rows of their own, keyed by an extraId, and are packaged on
their own, never inside the title's package. A second consumer, on a
thread of its own, encodes them, so a feature's encode never holds a
two-minute trailer up behind it. One item encode and one extra encode
can therefore run at once per pod, sharing the GPU (every encoded rung
is an encoder session of its own) or the CPU.

| | Items | Extras |
| --- | --- | --- |
| Consumes | `CONSUME_TOPIC` | `<KAFKA_TOPIC_PREFIX>catalog.extra.queued` |
| Produces | `PRODUCE_TOPIC` | `<KAFKA_TOPIC_PREFIX>catalog.extra.transcoded` |
| Consumer group | `KAFKA_GROUP_ID` (`transcoder-workers`) | `EXTRAS_GROUP_ID` (`transcoder-extras`) |
| Ladder | `LADDER` (empty: one rendition) | `EXTRA_LADDER` (`720p:h264,480p:h264`) |
| [Inbox](#inbox) | v2: `<work root>/inbox/<itemId>/`; legacy: `_inbox/<itemId>/` | v2: `<work root>/inbox/extra-<extraId>/`; legacy: `_inbox/extra-<extraId>/` |
| Worker record | `GET /api/analyze/items/{id}` | `GET /api/analyze/extras/{id}` |
| Step | `PUT /api/analyze/items/{id}/steps/transcode` | `PUT /api/analyze/extras/{id}/steps/transcode` |

`KAFKA_TOPIC_PREFIX` is the tenant's topic prefix, `stube.` by default
(a missing trailing dot is added), so the extras' topics default to
`stube.catalog.extra.queued` and `stube.catalog.extra.transcoded`.

The extras' events carry `extraId`, `parentId` (the movie or series the
extra belongs to) and `kind` in place of `itemId`, and `"type": "extra"`:

```json
{"eventId": "9f2b…", "extraId": "1b5c2a8e-…", "parentId": "ea886f9b-…",
 "type": "extra", "kind": "trailer", "step": "transcode", "status": "queued",
 "occurredAt": "2026-10-06T08:00:00Z", "source": "api"}
```

They never carry an `itemId`, so an item worker pointed at an extras
topic skips them. The `catalog.extra.transcoded` the transcoder sends is
the same envelope with `"step": "package"`, `"status": "queued"` and
`"source": "transcoder"`. An extraId that is not a lower-case UUID makes
the event malformed: it is committed and skipped.

Per trigger:

1. `GET /api/analyze/extras/{id}`. A 404 (unknown or removed), a record
   with `removedAt`, or an extra in state `missing` (its file is gone) is
   skipped: no step write, no event.
2. An extra past its transcode (`transcoded`, `packaging` or `ready`)
   runs nothing. A redelivered or duplicate trigger sends
   `catalog.extra.transcoded` again, so a packager that missed it
   recovers; a retry (`"status": "retry"`) is acked with one log line
   (`transcoder.extra.retry.already_finished`) and nothing else, as for
   items.
3. Otherwise the transcode runs as for an item (probe, plan, one ffmpeg),
   with `EXTRA_LADDER`, into the extra's [inbox](#inbox), under the
   unchanged [rendition contract](#rendition-contract-transcoder--packager)
   (its `itemId` is the extraId; a `"file": null` rung is a copy of the
   extra's source). The step goes `in_progress`, then `done`,
   `not_applicable` or `failed`, with the body an item's step takes; on
   `done` or `not_applicable` the transcoder sends
   `catalog.extra.transcoded`. The item types gate (movie, episode) does
   not apply.

The extras' default ladder has no source rung, so whatever the source,
the package is H.264. Keep `EXTRA_LADDER` to H.264 rungs while an extra
is served without an on-the-fly fallback (HEVC only: below). A rung at
the source's own size is a stream copy when the source is
browser-friendly H.264; VP9, Theora, HEVC and any other source is
encoded. With the default ladder:

| Source | v0 | v1 |
| --- | --- | --- |
| 1080p H.264 | 720p encode | 480p encode |
| 720p H.264 | the source, copied | 480p encode, on the source's keyframes |
| 480p VP9 | 480p encode (one rung: both boxes hold the source) | |
| 4K HEVC | 720p encode | 480p encode |
| 360p H.264 | the source, copied: `not_applicable`, no handoff | |

The rungs of an HDR source are tone-mapped to SDR BT.709, as H.264
rungs always are ([Ladder](#ladder)). Unset or empty `EXTRA_LADDER` is
the default, never the single HEVC source rung an empty `LADDER` means;
a typo fails startup. The [Sources](#sources) rules hold for an extra
too: a Dolby Vision 5 or 7 trailer fails its step.

`EXTRA_LADDER=source:hevc` makes an extra's package HEVC only, as for
the items ([HEVC only](#hevc-only)): one rendition at the source's own
size, an HEVC source copied (`not_applicable`), any other encoded once.
Set it only once the streaming side transcodes extras on the fly from
their package; until then a device without HEVC cannot play them. The
platform chart does not pass `EXTRA_LADDER` (yet), so a chart install
runs the default.

## Layout

```
src/transcoder/main.py         # entry point: encoder probe, worker threads, /healthz, /readyz
src/transcoder/config.py       # env-driven config
src/transcoder/decision.py     # ladder parsing + rendition planning
src/transcoder/ffmpeg.py       # ffprobe, NVENC detection, the encode command
src/transcoder/renditions.py   # the renditions.json handoff contract
src/transcoder/katalog.py      # HTTP client for the catalog step API
src/transcoder/worker.py       # Kafka consumer loop
src/transcoder/extras.py       # the extras' Kafka consumer loop
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
| `PACKAGES_ROOT` | `/var/lib/katalog/packages` | Root of the legacy layout's `_inbox` handoff tree (a worker record without `library`, see [Inbox](#inbox)) |
| `LADDER` | (empty) | Rendition ladder, see above; empty = one rendition; `source:hevc` = [HEVC only](#hevc-only) |
| `ENCODER` | `auto` | `auto`, `nvenc` or `cpu` |
| `SEGMENT_SECONDS` | `6` | Forced keyframe interval = packager segment length |
| `NVENC_PRESET` | `p5` | NVENC preset |
| `NVENC_CQ` | `23` | NVENC constant quality |
| `NVENC_MAXRATE_1080P_MBPS` | (unset) | Override of a source-size HEVC encode's maxrate in the 1080 bucket, every kind; unset = the cap ([Caps](#caps)) |
| `NVENC_MAXRATE_2160P_MBPS` | (unset) | The same for the 2160 bucket |
| `CAP_MOVIE_1080_MBPS` | `8` | A movie's or an extra's cap below 2000 lines, Mbit/s of video; `0` = off |
| `CAP_MOVIE_2160_MBPS` | `14` | The same from 2000 lines |
| `CAP_MOVIE_MAX_GIB` | `15` | A movie or extra file above this many GiB is encoded; `0` = off |
| `CAP_EPISODE_1080_MBPS` | `6` | An episode's cap below 2000 lines; `0` = off |
| `CAP_EPISODE_2160_MBPS` | `8` | The same from 2000 lines |
| `X264_PRESET` / `X264_CRF` | `medium` / `23` | CPU H.264 rungs |
| `X265_PRESET` / `X265_CRF` | `medium` / `24` | CPU HEVC rungs |
| `KAFKA_BROKERS` | `kafka:9092` | Bootstrap brokers |
| `KAFKA_TOPIC_PREFIX` | `stube.` | Tenant topic prefix of the extras' topics |
| `EXTRA_LADDER` | `720p:h264,480p:h264` | The extras' ladder, see [Extras](#extras); empty = the default; `source:hevc` = HEVC only |
| `EXTRAS_GROUP_ID` | `transcoder-extras` | The extras' consumer group |

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
