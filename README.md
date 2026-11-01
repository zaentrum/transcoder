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
   are encodes, capped. A Dolby Vision 5 source is encoded from its
   picture reshaped to HDR10 ([Dolby Vision 5](#dolby-vision-5)), and a
   Dolby Vision 7 one is packaged as its HDR10 base layer
   ([Dolby Vision 7](#dolby-vision-7)). A source no rung may copy or
   encode as it is (Dolby Vision 5 on a host without a Vulkan device, a
   base layer of compatibility 0) fails the step.
3. If anything needs encoding, run **one** ffmpeg: decode the source
   once (on the GPU where it can, see [GPU decoding](#gpu-decoding)),
   `split` it per rung, write one intermediate MKV per encoded rung (and
   a Dolby Vision 7 source's copy of its base layer) into the item's
   [inbox](#inbox), then `renditions.json`.
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
| `prepared.mkv` | v0 when v0 is encoded: video + every audio track + every Matroska-copyable subtitle track, copied verbatim; an MP4's text captions (`mov_text`) as SubRip |
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
- A `"mode": "copy"` rung with a file of its own (`"encoder": "copy"`) is
  the original's video as the transcoder wrote it into the inbox, as it
  is but for what it drops: a Dolby Vision 7 source's enhancement layer
  and RPU ([Dolby Vision 7](#dolby-vision-7)). It keeps the original's
  frames, keyframes and timestamps on the shared timeline, like every
  file in the inbox.
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
| Dolby Vision 5 | one HEVC Main 10 encode at its size of the picture reshaped to HDR10; without a Vulkan device, none: the step fails ([Dolby Vision 5](#dolby-vision-5)) |
| Dolby Vision 7 | its HDR10 base layer: within its cap copied without the enhancement layer and the RPU (`prepared.mkv`, a copy the transcoder writes), above it one HEVC Main 10 encode of it at its size ([Dolby Vision 7](#dolby-vision-7)) |

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
  that, as its HDR10 base layer alone when it is profile 7, reshaped to
  HDR10 when it is profile 5, and refused otherwise:

| Dolby Vision | Base layer | What happens |
| --- | --- | --- |
| 8.1, 8.2, 8.4 | HDR10, SDR, HLG | the base layer's rules: HEVC Main 10 is copied, as before |
| 4, 9, 10.1, 10.2, 10.4 | SDR, HDR10 or HLG | the base layer's rules |
| 5 | none (IPT-PQ-c2 pictures) | reshaped to HDR10 and encoded ([Dolby Vision 5](#dolby-vision-5)); without a Vulkan device, the step fails |
| 7 | HDR10 (PQ), with an enhancement layer | the base layer alone, the enhancement layer and the RPU dropped: copied within its cap, encoded above it ([Dolby Vision 7](#dolby-vision-7)); a stream not tagged PQ fails the step |
| a base layer of compatibility 0 (10.0, …), or a `dvh1` / `dvhe` / `dav1` sample entry without a record | none | the step fails |

A refused source fails the transcode step with the reason, e.g. `Dolby
Vision profile 8.0 needs a tone-mapping encode; kept the original`.
Nothing is encoded or handed off (an older run's handoff is removed),
and no event goes. The title stays unpackaged, so its original is never
retired: it waits for a tone-mapping encode. The profile comes from the
stream's configuration record (ffprobe's side data `DOVI configuration
record`), else from its sample entry.

- **Closed captions** (EIA/CEA-608/708) carried in the video, as A53 SEI
  or user data, stay in every encode: each encoded rung is given
  `-a53cc 1`, which libx265 needs since ffmpeg 7.1 (it defaults to off).
  A copy keeps them as they are.
- **HDR10 static metadata** (the mastering display and the content light
  level, SMPTE ST 2086 and MaxCLL / MaxFALL): a copy keeps it as it is, in
  its container and in its SEI. Every encode's MKV carries the source's
  values in its track's `Colour` element (`MasteringMetadata`, `MaxCLL`,
  `MaxFALL`): ffmpeg hands the source's on to the encode, and its Matroska
  muxer writes them. A libx265 encode writes them into its bitstream as
  well, as the two SEI messages. An NVENC encode's bitstream carries
  neither: ffmpeg 7.1's `hevc_nvenc` writes no SEI but closed captions
  (ITU-T T.35), timecodes and, with `-udu_sei`, user data unregistered,
  and no bitstream filter of 7.1 adds one. The NVENC fields for the two
  (`outputMasteringDisplay`, `outputMaxCll`) came with NVENC SDK 13.0
  (a driver from 570 on), which ffmpeg uses from 8.0, built against it.
  The packager writes the `mdcv` and `clli` boxes of every HDR10 rendition
  from the original's values.

### Dolby Vision 7

Profile 7 is a UHD Blu-ray's dual layer in one track: an HDR10 base layer
(PQ, BT.2020, the disc's HDR10 picture), an enhancement layer in NAL units
of type 63, and the RPU in units of type 62. The base layer needs no
tone-mapping, and Dolby Vision 7 plays on no streaming device as it is, so
the package is the base layer alone, as HDR10, under the
[caps](#caps) of any HEVC source:

- **Within its cap**, a copy: not the original, which the packager would
  take with both layers, but a copy the transcoder writes into
  `prepared.mkv`, every audio and subtitle track beside it, as an encoded
  v0 has them. Its video is the original's stream copied through
  `-bsf:v filter_units=remove_types=62|63,dovi_rpu=strip=1`: every unit of
  the enhancement layer and the RPU dropped, the configuration record too,
  and everything else as it was, the base layer's SEI (its HDR10 static
  metadata) among it, its frames, keyframes and timestamps. Nothing is
  decoded. In the contract it is `"mode": "copy"`, `"encoder": "copy"`,
  with its file. A ladder's other rungs are encoded beside it in the same
  run, on its keyframes.
- **Above its cap**, one HEVC Main 10 encode of the base layer at its size,
  as for HDR10 (on the GPU where [GPU decoding](#gpu-decoding) serves it):
  the decoders read no unit of type 63, and no encoder here writes an RPU
  or a record.

The record is the one judge, whatever compatibility it names; a stream it
calls profile 7 that is not tagged PQ fails the step (`Dolby Vision
profile 7 without a PQ base layer needs a tone-mapping encode; kept the
original`). The cap is judged on the source's video as it is, the
enhancement layer included.

It fails closed as profile 5 does: after the run, before anything is
renamed into place, every output must be tagged HDR10 (BT.709 for a
tone-mapped H.264 rung), carry no Dolby Vision record, and, if HEVC,
carry no unit of type 62 or 63 in its first 8 frames; else the step fails
(`Dolby Vision 7 to its HDR10 base layer: v0 still carries NAL units of
type 62 and 63; nothing handed off`). The step's details say
`base_layer=dolby_vision_7:hdr10` (and `decode=none` for the copy alone),
and `transcoder.item.dolby_vision` logs the profile and its rungs' modes.

Profile 8 keeps its RPU, which belongs to the base layer it ships with: an
8.1 within its cap is the original's own stream, Dolby Vision and all, as
before; an encode of it carries neither the RPU nor the record.

### Dolby Vision 5

Profile 5 has no base layer other devices play: its pictures are
IPT-PQ-c2, and its colour tags say nothing (a web release's transfer is
unset). Each frame's RPU says how to reshape the picture, and ffmpeg's
HEVC decoder hands it on with the frame. A source whose configuration
record says profile 5 is therefore never copied: every rung is encoded
from the picture libplacebo reshapes to HDR10, once per source, before
the split:

```
-init_hw_device vulkan=vk:<index> -filter_hw_device vk
[0:v]format=yuv420p10le,hwupload,
     libplacebo=apply_dolbyvision=1:colorspace=bt2020nc:color_primaries=bt2020
                :color_trc=smpte2084:range=tv:format=p010le,
     hwdownload,format=p010le
```

The frames go up to the Vulkan device from system memory (ffmpeg 7.1's
`hwupload` takes no CUDA frames) and come back down 10-bit, in what the
top rung's encoder takes: `p010le` for NVENC, `yuv420p10le` for libx265.
The rungs are those of an HDR10 source: HEVC Main 10 with HDR10's tags
named on the encoder (BT.2020, PQ, BT.2020 non-constant, limited range),
H.264 rungs tone-mapped to BT.709, the caps as for any HEVC encode. The
package carries no Dolby Vision (no record, no RPU) and no static HDR10
metadata: ffmpeg 7.1's libplacebo filter writes no mastering display or
light level, and NVENC writes no such SEI (Sources, HDR10 static
metadata).

It fails closed. The step fails with the reason, nothing is handed off,
and the original is kept, when:

| When | The step's error |
| --- | --- |
| the host has no Vulkan device (the startup preflight) | `Dolby Vision profile 5 is reshaped to HDR10 on a Vulkan device, and this host has none (Vulkan loader 1.3.204: Instance creation failure: VK_ERROR_INCOMPATIBLE_DRIVER); kept the original` |
| one of the first 8 decoded frames carries no RPU | `Dolby Vision profile 5: 0 of the first 8 decoded frames carry the RPU that libplacebo reshapes them with; kept the original` |
| the decoder could not read an RPU during the encode (`Error parsing DOVI NAL unit`) | `Dolby Vision to HDR10: the decoder could not read the Dolby Vision RPU of some frames (…); nothing handed off` |
| an output is not tagged HDR10, limited range (BT.709 for an H.264 rung), or claims Dolby Vision | `Dolby Vision to HDR10: v0 came out as …, not bt2020/smpte2084/bt2020nc/tv; nothing handed off` |

At startup the preflight runs one generated frame through that chain and
logs one line, `transcoder.vulkan.preflight`: `ok`, the Vulkan loader's
version as the loader reports it (`loader`), the `device`, and when
there is none the `error`, with the loader's own messages (the preflight
runs with `VK_LOADER_DEBUG=error,warn`). A software Vulkan device
(Mesa's lavapipe) counts as none: the reshaping runs on a GPU. Each
reshaped item logs `transcoder.item.dolby_vision`, and its step details
say `reshape=dolby_vision_5:hdr10`.

The container needs a Vulkan loader new enough for the driver's ICD, the X11
libraries that `libGLX_nvidia`, the library the NVIDIA ICD names, links
against (`libx11-6`, `libxext6`), glvnd (`libglvnd0`, `libegl1`, `libgl1`,
`libglx0`), and `NVIDIA_DRIVER_CAPABILITIES=all`, for the NVIDIA container
toolkit to mount the driver's Vulkan and EGL libraries, its ICD manifest and
its EGL vendor file. Without glvnd the driver loads but declines to create an
instance. The image has all of it: its loader is built from Khronos's sources
(`VULKAN_SDK_TAG`, a 1.4 release). A host without a GPU has no device, and
its profile 5 sources fail their step.

## Caps

A source the packager would copy, HEVC Main or Main 10 4:2:0 up to 10
bits, is copied only within its cap. Above the cap it is encoded **once
at its own size**: NVENC, else libx265, its bit depth kept (Main 10 above
8 bits, HDR as HDR), the Dolby Vision rules first. The VBV maxrate is the
cap and the bufsize twice it. Every source-size HEVC encode takes the
same cap as its maxrate (an H.264 source on `source:hevc`, a re-encoded
Rext, the capped copy).

The cap goes by the item's kind and the size of its video (its tallest
and its widest stream, cover art aside). A source is in the **2160
bucket when it is 2000 lines tall or 3200 wide**, and in the 1080
bucket otherwise:

| Kind | 1080 bucket | 2160 bucket | File size |
| --- | --- | --- | --- |
| movie, extra | 8 Mbit/s | 14 Mbit/s | above 15 GiB: encoded, whatever its bit rate |
| episode | 6 Mbit/s | 8 Mbit/s | no rule |

| Picture | Bucket | Because |
| --- | --- | --- |
| 3840×2160, 4096×2160 | 2160 | its height |
| 3840×1600 scope, 4096×1716 DCI scope | 2160 | its width |
| 1920×1080, 1920×800, 2560×1440 | 1080 | |

So a 3840×1600 movie is copied at 12 Mbit/s (under 14) and encoded at
20 Mbit/s, with maxrate 14.

A source is above its cap when its video's bit rate is strictly above
it. That is the video stream's own rate, read from, in order:

1. ffprobe's `bit_rate` of the video stream (MP4, and TS when known);
2. the Matroska statistics tag, `BPS` or `BPS-eng`, unless it is stale;
3. the file's size × 8 ÷ the video's own duration (the shortest of the
   file's and the video's `DURATION` tags), less the audio streams' rates:
   each its `bit_rate`, else its `BPS` tag, and of a stream whose own
   `DURATION` ends before the video's only its share, its rate × its
   `DURATION` ÷ the video's (a 5.1 PCM track of 4.6 Mbit/s that ends half
   way through takes off 2.3). One with neither, or with a stale tag,
   counts 0, and so does every audio tag once the video's is stale;
4. else unknown: the bit rate rule breaks nothing.

A statistics tag that a re-encode or a remux kept from its source
(ffmpeg keeps `BPS` and `NUMBER_OF_BYTES` as they were, and writes only
`DURATION` anew) describes another stream. It is stale, and left out,
when it says its stream is bigger than the whole file, which no stream
is: a `NUMBER_OF_BYTES` more than 2 % above the file's size, or a `BPS`
more than 2 % above the file's size × 8 ÷ the stream's own duration. A
video's `BPS` is judged so only when the video has a `DURATION` tag of
its own: the file may run on past the video (a subtitle's last cue), and
on the file's duration a right tag would read stale (ffmpeg writing to a
pipe writes no `DURATION`). An audio stream's is judged on its own
`DURATION`, else on the file's duration: left out wrongly, it only makes
3 read high.
And once the video's tag is stale, no audio tag is taken off: the tool
that kept the video's kept theirs, a lossless track's on its lossy
re-encode, say.

So 3 errs high — the rest of the file goes with the video — as long as
the audio rates it takes off are those streams' own, no audio stream
without a `DURATION` tag ends well before the video, and the video has a
`DURATION` tag: a file that runs on past a video without one reads low. A
stream's share can only raise the estimate: it takes off at most the
stream's whole rate.

The step's details and the log line `transcoder.item.cap` say which one
it was, and for the 2160 bucket what put the source there
(`rate=12.7Mbps(size-minus-audio) cap=movie-1080:8Mbps over=bitrate`,
`rate=20.0Mbps(tag:BPS) cap=movie-2160(width):14Mbps over=bitrate`; the
log line's `bucket_by` is `height` or `width`), and the stale tags left
out (`rate=10.0Mbps(size-minus-audio, stale tag:BPS 24.0Mbps)`, an
audio stream's as `a:0 tag:BPS 30.0Mbps`; the log line's `stale_tags`).
The H.264 that a CPU host passes through, and the H.264 rungs, are not
capped this way: they keep the ladder's table.

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

## GPU decoding

On a host whose HEVC encoder is NVENC, the source of an encode is decoded
on the GPU too: NVDEC hands the frames on in GPU memory (CUDA frames), the
rungs are scaled there (`scale_cuda`), and NVENC takes them where they
are. The CPU only reads, parses and writes the files, so a 4K HEVC encode
no longer waits for the CPU's decoder.

`DECODER=auto` (the default) probes it once at startup and logs one line,
`transcoder.decoder.probe`: a few frames of HEVC Main 10, which
`hevc_nvenc` writes first, decoded by NVDEC, converted with `scale_cuda`
and encoded by NVENC from GPU memory. When the probe fails, or HEVC is
encoded with libx265, every source is decoded on the CPU. `DECODER=cpu`
never probes: every source is decoded on the CPU, as before.

| Source | Decoded |
| --- | --- |
| H.264 8-bit 4:2:0, Baseline to High; HEVC Main or Main 10, 4:2:0 (Dolby Vision 7 and 8.x: their base layer); VP9 profile 0 or 2; AV1 Main; 8 or 10 bits, progressive | on the GPU |
| 4:2:2, 4:4:4 or 12-bit; H.264 High 10; MPEG-2, VC-1, …; interlaced (ffprobe's field order `tt`, `bb`, `tb` or `bt`); Dolby Vision 5 (reshaped from system memory) | on the CPU |
| a source whose encode a CPU encoder takes part in (libx264 rungs on a host without `h264_nvenc`) | on the CPU |
| a source whose ladder has more than one HEVC rung (`source,1080p:hevc`; below) | on the CPU |
| a source NVDEC hands on at another size than its picture's: cropped at the top or the left (below) | on the CPU |

Per rung, the GPU's filters do what the CPU's do:

| Rung | CPU | GPU |
| --- | --- | --- |
| the source's size and bit depth | `format=p010le` (10-bit), `format=yuv420p` or none (8-bit) | `format=cuda` |
| the source's size, 10-bit of an 8-bit source | `format=p010le` | `scale_cuda=format=p010le` |
| scaled | `scale=W:H,setsar=1,format=…` | `scale_cuda=W:H:format=…:interp_algo=bilinear,setsar=1`, halved first while the picture is more than 3 times the rung's size (4K to 480p: 1920×1080, then 854×480) |
| H.264 of an HDR source | the scale, then the tone-map | the scale on the GPU, then `hwdownload` and the same tone-map, at the rung's size |

One encoder of a run takes the frames in GPU memory as they are: the
first rung that is not tone-mapped, the largest. Every other rung is
scaled on the GPU all the same and comes down (`hwdownload`) in the format
the CPU path gives its encoder (`p010le`, or `yuv420p` for 8 bits).

The decoder's CUDA device is named (`-init_hw_device cuda=dec
-hwaccel_device dec`) and a second is opened beside it (`cuda=spare`):
ffmpeg hands an encoder of system memory frames the one CUDA device there
is, so with one, every NVENC encode of the run would open on the
decoder's context; with two, each encoder of system memory frames opens a
context of its own, as on the CPU path.

NVENC now and then sizes the DPB of an HEVC encode one frame larger than
its own and raises its level to fit it: a 720p rung from level 3.1 to
4.0, a 1080p one to 5.0, a 4K one to 6.0, which a decoder for 4.1 or 5.1
turns down. Neither `-dpb_size` nor `-level` holds it. Two HEVC encodes
opening at once did it in 1 of 6 runs, so a ladder of more than one HEVC
rung is decoded on the CPU. One HEVC rung beside H.264 rungs did it in 17
of 20 runs whose PTX cache (`CUDA_CACHE_PATH`, below) was empty, with one
CUDA device or two, and in none of 52 runs of four ladders once the cache
held the kernels of `scale_cuda`, as the startup probe leaves it. Every
NVENC output is checked for it (below): a GPU run it happened in is made
again on the CPU.

Before a source is decoded on the GPU, its first frame is (about 0.7 s):
it must come out of NVDEC as large as ffprobe says its picture is. A
picture cropped at its top or its left (H.264's frame cropping, HEVC's
conformance window) comes out of NVDEC uncropped there — libavutil crops a
hardware frame at its right and bottom only — and would be encoded as the
picture shifted by the rows or columns it drops (a 1920×1064 picture as
1920×1080, with exit 0). Such a source is decoded on the CPU
(`decode=cpu(nvdec_frame=1920x1080)`); so is one NVDEC turns down there
(`nvdec_preflight_failed`).

`format=cuda` converts nothing and takes CUDA frames only, as every GPU
chain here does (`scale_cuda` and `hwdownload` too): a source the decoder
hands on in system memory after all fails the run at its first frame. The
GPU run is watched: when its progress (frames, bytes written, time
encoded) stands still for 10 minutes, it is taken for a call into NVDEC,
NVENC or CUDA that will not return, and killed. A GPU run that fails,
stands still, writes a decoder error (a frame it did not decode:
`Decoding error`, `hardware accelerator failed`, a CUDA error), or makes
an output that is not its rung's size or not what NVENC makes of system
memory frames (below), whatever its exit code, is logged
(`transcoder.encode.cuda_failed`, with why and ffmpeg's last words) and
made again at once with the CPU's command, the one the transcoder ran
before; only that run's failure fails the step.

Every NVENC output, of either path, is checked before it is handed off:
the profile it was asked for, a DPB no larger than NVENC's own for system
memory frames at any preset (HEVC 6 pictures, H.264 4 reference frames),
and a level no higher than the lowest the spec allows its coded size,
frame rate (its timestamps', or the one its VUI names, which NVENC was
given, when that is higher) and VBV maxrate with that DPB: the level
NVENC picks itself. On the CPU's decode an output that fails it fails the
step: `NVENC: v0 came out with a DPB of 7, NVENC's own is at most 6;
nothing handed off`.

Where the source was decoded is in the step's details, `decode=cuda`,
`decode=cpu(<why not>)` (`cpu(cuda off)`, `cpu(4:2:2)`, `cpu(interlaced)`,
`cpu(codec=mpeg2video)`, `cpu(encoder=libx264)`, `cpu(hevc_rungs=2)`,
`cpu(nvdec_frame=1920x1080)`, …) or `decode=cpu(cuda failed)`, and in the
log lines `transcoder.encode.start` and `transcoder.encode.done`
(`decoder`) and `transcoder.item.done` (`decode`).

The output is the CPU path's: the same rungs and sizes, pixel formats,
colour tags, sample aspect ratio, levels, frames, timestamps and
keyframes, rate control and closed captions. NVDEC decodes bit-exactly
and NVENC is deterministic, so a rung at the source's size is the CPU
path's bit for bit. A scaled rung differs as far as `scale_cuda`'s
bilinear differs from swscale's bicubic: its frames are at an SSIM (Y) of
0.994 to 0.999 to the CPU path's (an 8-bit rung of a 10-bit source drops
its last two bits without dither: 0.994).

On an RTX 3090, the container limited to 4 CPUs, 60 s of synthetic video
(a test pattern with grain), one source-size HEVC encode:

| Source | CPU decoding | GPU decoding |
| --- | --- | --- |
| 4K HEVC Main 10, 46 Mbit/s | 45.5 fps, 3.8 CPUs | 75.5 fps, 0.1 CPUs |
| 1080p H.264 High, 24 Mbit/s | 176 fps, 3.8 CPUs | 271 fps, 0.2 CPUs |

NVENC is then what limits an encode. The kernels of `scale_cuda` come as
PTX, which the driver compiles in about 5 s on first use, and keeps where
`CUDA_CACHE_PATH` says: the image sets it to `/tmp/cuda-cache` (the
container's home is not writable), so the startup probe pays for it once,
not every encode.

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
too: a Dolby Vision 7 trailer's rungs are tone-mapped from its base
layer, and a profile 5 one is reshaped to HDR10 first, its rungs
tone-mapped from there (on a host without a Vulkan device, it fails its
step).

`EXTRA_LADDER=source:hevc` makes an extra's package HEVC only, as for
the items ([HEVC only](#hevc-only)): one rendition at the source's own
size, an HEVC source copied (`not_applicable`), any other encoded once.
Set it only once the streaming side transcodes extras on the fly from
their package; until then a device without HEVC cannot play them. The
platform chart does not pass `EXTRA_LADDER` (yet), so a chart install
runs the default.

## Rules report

The catalog holds every movie and episode against the rules its workers
run with now, to find the packages made before a rule came (a 4K
original's video copied as it was, before the [caps](#caps)), and knows
the rules only as a worker reports them. At startup and every ten minutes
the transcoder reports its own, on a thread and a client of its own:

```
PUT /api/analyze/rules/transcoder
{"version": "3f2a9c1", "rules": {
  "caps": {"movie1080Mbps": 8, "movie2160Mbps": 14, "movieMaxGiB": 15,
           "episode1080Mbps": 6, "episode2160Mbps": 8},
  "bucket2160": {"minHeight": 2000, "minWidth": 3200},
  "encoder": {"codec": "hevc_nvenc", "preset": "p5", "cq": 23,
              "maxrate1080Mbps": null, "maxrate2160Mbps": null},
  "ladder": "source:hevc", "extraLadder": "source:hevc",
  "ladderRungs": [{"name": "source", "codec": "hevc", "codecNamed": true, "maxrateBps": null}],
  "rungMaxrateMbps": {"360": {"h264": 0.8, "hevc": 0.6}, "480": {"h264": 1.5, "hevc": 1.2},
                      "540": {"h264": 2.0, "hevc": 1.6}, "720": {"h264": 3.0, "hevc": 2.5},
                      "1080": {"h264": 6.0, "hevc": 8.0}, "1440": {"h264": 9.0, "hevc": 8.0},
                      "2160": {"h264": 16.0, "hevc": 14.0}},
  "dolbyVision": {"profile5": "reshaped", "profile7": "baseLayer"}}}
```

Every value is the one the process runs with: the caps (`0` a rule
switched off), the 2160 bucket, the HEVC encoder the startup probe found
with its knobs (NVENC's preset and `cq`; libx265's preset and `crf`, `cq`
null), the `NVENC_MAXRATE_*` overrides (null: the caps), and each ladder
as its spec reads with every rung as the planner takes it — its codec
where the spec names it, its maxrate where it sets one — so an empty
`LADDER` is `source`, the same one rendition; `ladderRungs` is the item
ladder in parts, and `rungMaxrateMbps` the maxrate of a rung that sets
none, by the height of its box and its codec (a scaled rung's, and a
source-size H.264 encode's), so that the catalog counts every rung of a
package it estimates. `dolbyVision` is what becomes of an original of each
Dolby Vision profile, as the process can do it: `reshaped` (profile 5,
reshaped to HDR10 on the Vulkan device the startup preflight found:
`transcoder.vulkan.preflight`, [Dolby Vision 5](#dolby-vision-5)),
`baseLayer` (profile 7, packaged as its HDR10 base layer on any host,
[Dolby Vision 7](#dolby-vision-7)) or `refused` (the step fails and the
original is kept: profile 5 without such a device); the catalog offers no
title of a refused profile to be processed again. `version` is the image's, `IMAGE_VERSION` (the image
build sets it to the release tag or the commit), else the package's.

A catalog older than the route answers 404 or 405: the transcoder says so
once (`transcoder.rules.unsupported`) and asks again every round. Any other
failure is logged (`transcoder.rules.refused`, `transcoder.rules.report_failed`)
and never stops the worker.

## Layout

```
src/transcoder/main.py         # entry point: encoder probe, worker threads, /healthz, /readyz
src/transcoder/config.py       # env-driven config
src/transcoder/decision.py     # ladder parsing + rendition planning
src/transcoder/ffmpeg.py       # ffprobe, NVENC / NVDEC detection, the encode command
src/transcoder/renditions.py   # the renditions.json handoff contract
src/transcoder/katalog.py      # HTTP client for the catalog step API
src/transcoder/rules.py        # the report of the rules it runs with
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
| `DECODER` | `auto` | `auto` (decode on the GPU when the startup probe finds NVDEC working and HEVC is encoded with NVENC) or `cpu`; see [GPU decoding](#gpu-decoding) |
| `SEGMENT_SECONDS` | `6` | Forced keyframe interval = packager segment length |
| `NVENC_PRESET` | `p5` | NVENC preset |
| `NVENC_CQ` | `23` | NVENC constant quality |
| `NVENC_MAXRATE_1080P_MBPS` | (unset) | Override of a source-size HEVC encode's maxrate in the 1080 bucket, every kind; unset = the cap ([Caps](#caps)) |
| `NVENC_MAXRATE_2160P_MBPS` | (unset) | The same for the 2160 bucket |
| `CAP_MOVIE_1080_MBPS` | `8` | A movie's or an extra's cap in the 1080 bucket, Mbit/s of video; `0` = off |
| `CAP_MOVIE_2160_MBPS` | `14` | The same in the 2160 bucket (2000 lines tall or 3200 wide) |
| `CAP_MOVIE_MAX_GIB` | `15` | A movie or extra file above this many GiB is encoded; `0` = off |
| `CAP_EPISODE_1080_MBPS` | `6` | An episode's cap in the 1080 bucket; `0` = off |
| `CAP_EPISODE_2160_MBPS` | `8` | The same in the 2160 bucket |
| `X264_PRESET` / `X264_CRF` | `medium` / `23` | CPU H.264 rungs |
| `X265_PRESET` / `X265_CRF` | `medium` / `24` | CPU HEVC rungs |
| `KAFKA_BROKERS` | `kafka:9092` | Bootstrap brokers |
| `KAFKA_TOPIC_PREFIX` | `stube.` | Tenant topic prefix of the extras' topics |
| `EXTRA_LADDER` | `720p:h264,480p:h264` | The extras' ladder, see [Extras](#extras); empty = the default; `source:hevc` = HEVC only |
| `EXTRAS_GROUP_ID` | `transcoder-extras` | The extras' consumer group |
| `IMAGE_VERSION` | (the image's) | The version the [rules report](#rules-report) names; the image build sets it, empty = the package's |

## Local development

```bash
uv sync
uv run pytest
```

Unit tests need nothing installed. `tests/test_encode_real.py` runs real
encodes on generated clips when `ffmpeg` is on `PATH` (the HDR case also
needs the `zscale` filter). The Dolby Vision 5 reshaping runs only with
an ffmpeg that has the `libplacebo` and `zscale` filters, a Vulkan device
(a software one will do there, slowly) and a profile 5 file, which no
test ships: `TRANSCODER_DOVI5_SAMPLE=/path/to/dv5.mkv uv run pytest`.

## Build the container

The image is based on `nvidia/cuda:*-runtime` and installs a pinned,
checksummed BtbN ffmpeg 7.1 static build (NVENC, libx264, libx265, zimg,
libplacebo and Vulkan), the Vulkan loader and the X11 libraries the
NVIDIA Vulkan driver links against; it asks the NVIDIA runtime for
`compute,video,utility,graphics` ([Dolby Vision 5](#dolby-vision-5)), and
keeps the CUDA kernels the driver compiles in `/tmp/cuda-cache`
(`CUDA_CACHE_PATH`, [GPU decoding](#gpu-decoding)).
GPU nodes need the NVIDIA device plugin (`nvidia.com/gpu: 1`); CPU-only
nodes run the same image without the GPU request.

```bash
docker build -t zaentrum/transcoder .
```

Build and push the image to your own registry, then update the image reference
in `k8s/deployment.yaml` for your environment.

## License

[MPL-2.0](LICENSE).
