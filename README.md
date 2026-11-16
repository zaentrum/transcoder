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
   ([Dolby Vision 7](#dolby-vision-7)). A source in fields, as its
   pictures show it, is never copied: it is encoded progressive
   ([Fields](#fields)). A source no rung may copy or
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
  ],
  "scan": {"found": "progressive", "judgedBy": "mark", "fieldOrder": "progressive", "done": "none",
           "order": "tff", "orderBy": "default", "orderSeen": false, "frameRate": "24000/1001",
           "filter": null, "idet": null, "verified": null}
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
- `scan` is what the package's video is as far as fields go, for the
  package's record: what the source was `found` to be (`fields`, `film`,
  `telecine`, `mixed`, `progressive`, `unknown`, `field-pictures`) and by
  what (`judgedBy`: `pictures`, `mark`, `bitstream` or `failed`, its
  `fieldOrder`), what was `done` — the field the deletion gate reads:
  `inverse-telecine`, `progressive`, `deinterlaced-field` (these three let
  an original in fields go, each only once the whole title is read through),
  `deinterlaced-frame`, `field-matched`, `fields-kept`, `unexamined`,
  `unverified` or `none` —,
  the field `order` taken, what named it (`orderBy`) and whether idet saw it
  (`orderSeen`), the package's `frameRate` (a deinterlace's output rate), the
  `filter`, idet's counts of the sample (`idet`) and what the read-through
  of the whole title found (`verified`); see [Fields](#fields). New; a packager that doesn't know it
  ignores it.
- **Timestamps**: every inbox file is on the source timeline shifted by
  `timestampOffset`, frame for frame (the shift ffmpeg applies to start
  at zero, kept exact via `-enc_time_base demux`). The packager remuxes
  with `-copyts` and shifts the original by the same offset.
- **Compatibility**: no `renditions.json` and no `prepared.mkv` means
  "package the original" (a copyable HEVC source on the default ladder),
  exactly as before. A copy of a source marked as fields whose pictures
  are whole frames, and a copy of HEVC field pictures, are the copies with
  a `renditions.json`, every rung `"file": null`, so that their `scan`
  (`progressive`, `fields-kept`) reaches the package's record. `prepared.mkv` keeps its name, so an older packager still
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
| any of these that may hold fields, as its pictures show it | one HEVC encode at its size: video deinterlaced (at its frame rate by default), a film field-matched to its own frames, film and video field-matched where it can be ([Fields](#fields)) |

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
  A copy keeps them as they are. Through the fields chains, pair for pair:
  [Closed captions](#closed-captions).
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
                :color_trc=smpte2084:range=tv
                :tonemapping=clip:peak_detect=0:contrast_recovery=0
                :extra_opts=tone_map_metadata=none:format=p010le,
     hwdownload,format=p010le
```

The frames go up to the Vulkan device from system memory (ffmpeg 7.1's
`hwupload` takes no CUDA frames) and come back down 10-bit, in what the
top rung's encoder takes: `p010le` for NVENC, `yuv420p10le` for libx265.

The picture is the one the RPU composes, as it is: libplacebo's tone
mapping is off (`tonemapping=clip`, `peak_detect=0`, `contrast_recovery=0`,
`tone_map_metadata=none`). At its defaults libplacebo maps the composed
picture once more, from the RPU's black (`source_min_pq`) and scene peak
(its L1 block) to a display of 10,000 nits and no black, although PQ holds
the picture as it is. A profile 5 test picture then came out darker than
its RPU composes it, by 11 10-bit codes at 0.1 nits, 8 at 5 nits and 5 at
90 nits, and clipped above the L1 peak; with it off, every patch came out
within 1.5 codes of the composition as ETSI GS CCM 001 defines it.
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

## Fields

The master is progressive. A source that may hold fields is never copied,
whatever its codec: it is encoded, and made progressive at the start of its
filter chain, before the split and every rung's scale and format, from
frames the CPU decodes (it is never decoded on the GPU). What it holds is
judged by its pictures, not by its mark: ffprobe's field order is the first
picture's, or what a muxer wrote.

The record's rule: a word that lets an original in fields go
(`inverse-telecine`, `progressive`, `deinterlaced-field`) is said about the
whole title as it was encoded, never about a sample. A sample of the source
chooses the chain; the encode then reads the whole title through, every
frame of the source and of its output, and the word stands only where
nothing in it says otherwise, with no part left unread and no failure
anywhere ([The read-through](#the-read-through)). Anything less gives a word
that keeps the original.

| The sample shows | The chain | Frame rate | `scan.done`, read through |
| --- | --- | --- | --- |
| video in fields throughout | `bwdif=mode=send_frame:parity=<order>:deint=all`, one frame of every two fields (`DEINTERLACE=frame`, the default) | 25 or 29.97 | `deinterlaced-frame` (keeps) |
| the same, `DEINTERLACE=field`, in the order idet saw | `bwdif=mode=send_field:parity=<order>:deint=all`, one frame a field | 50 or 59.94 | `deinterlaced-field` |
| a 3:2 film at 29.97 or 30 whose cadence holds in every block | the field-matched chain (below), then `decimate`, the closed captions carried around it ([Closed captions](#closed-captions)) | 23.976 or 24 | `inverse-telecine` |
| a film at its own rate (2:2, shifted by a field), no repeated fields, mostly determined | the field-matched chain | the source's | `inverse-telecine` |
| film and video mixed; a film whose cadence breaks, repeats fields or is mostly undetermined | the field-matched chain | the source's | `field-matched` (keeps) |
| whole frames, marked as fields | none: copied where its codec allows, else encoded with `setfield=prog` | the source's | `progressive` |
| whole frames, marked progressive or not marked | as before (`setfield=prog` where not marked) | the source's | `none` (keeps; nothing to let go) |
| fields in one order too few to make a source marked progressive (or not marked) over | as before | the source's | `unverified` (keeps) |
| HEVC field pictures (below) | none: copied as before | the field rate, half as tall | `fields-kept` (keeps) |
| nothing the pictures can tell, or nothing read | the field-matched chain | the source's | `unexamined` (keeps) |

The field-matched chain is `fieldmatch=order=<order>:combmatch=full`, then
`bwdif=mode=send_frame:parity=<order>:deint=interlaced` (the frames
fieldmatch marks still combed: a lone orphan field after a cut), then `idet`
and a second such `bwdif` (the frames idet finds still combed: fieldmatch's
own marks miss most of a soft picture's video). Whole frames pass through it
as they are.

### The record, and the gate

`scan.done` in `renditions.json` is the one field the deletion gate reads
of it ([Rendition contract](#rendition-contract-transcoder--packager)); the
packager keeps the block in the package's manifest as it is. The rule that
reads it: an original in fields goes only where its package's scan says
`inverse-telecine`, `progressive` or `deinterlaced-field`. Any other word
keeps it, and so does a package made before this, which has no `scan`.
`scan.verified` says what the read-through found (`ok`, `why`, the word the
sample said), `scan.frameRate` the package's frame rate (25/1 at the frame
rate of a 25-frame source, 50/1 at its field rate).

### The examination

Before a source is planned, a sample of its frames goes through ffmpeg's
`idet`, as the frames are and after `fieldmatch` in either field order, in
one decode on the CPU (1,000 frames). It chooses the chain. Each rule sits
where made clips put it: those of `tests/test_fields_real.py` (every moment
of some of them numbered, `tests/fieldcode.py`) and of two reviews'
adversarial sets.

| Rule | Value | Why there |
| --- | --- | --- |
| Examined | every source at most 1088 lines tall, whatever its mark or codec; no Dolby Vision | ffprobe's mark is the first picture's: an MPEG-2 stream of 2 s of 25p, then 50i, reads progressive. Woven frames coded as progressive pictures read progressive, or nothing, in H.264, HEVC, VP9, AV1 and MPEG-4 alike. No Dolby Vision profile is interlaced |
| The mark | the stream's; a stored source record's (`sourceFieldOrder` in the worker record) where it gives one; where either says fields, the one that does | a copy of a source keeps its video bit for bit, but its fMP4 reads `unknown` where the source said `tb` |
| Where | 5 windows of 200 frames, from 10, 30, 50, 70 and 90 % of the video's duration (its own, where its stream says it) | a title's start is its logos and black; five places see a programme's parts. What lies between the windows the read-through sees |
| What of the picture | its middle: 10 % of the width off each side, 15 % of the height off the top and the bottom, the offsets in multiples of 4 lines | a broadcaster's logo in fields over a dark film frame is all idet sees of it: a made MPEG-2 film transfer reads 9 TFF frames of 157 determined uncropped and 0 cropped. A 2.39:1 film's letterbox bars in a 16:9 frame are 12.8 % of its height each. An offset of an odd line would make the top field the bottom one |
| Each window | "whole" where fewer than 5 % of its determined frames (and 10) are combed; "detail" where they comb in both orders (under 80 % one way); else "film" where fieldmatch leaves at most 5 % combed, never 5 in a row; "video" where it leaves at least 30 %; "unsure" between | woven fields come in one order: every made clip in fields reads 95–100 % one way, fine detail misread as combing 43–70 %. Film after fieldmatch: a clean 3:2 film 0.1 %, light grain 3.9 %, each combed frame on its own. Video: fieldmatch makes a clean-looking frame of about half a moving video's frames (made 50i 57 %, 60i 55 %), of none of others (100 %). Unsure: a slow object over a still, 7 % |
| Progressive | every window whole (detail only, under a mark of progressive or none) | one window that is not whole frames rules it out, whatever the rest's share: a 50i part filling one window is 19 % of the sample |
| Made over, marked progressive or none | fields in one order in at least 20 % of the determined frames and 50 | the share chooses the processing, never the word: below it the source is left as it is, `unverified` |
| Video | every window that sees is video, and its repeats keep no 3:2 cadence over half the blocks that see one | a film body with credits in fields is whole windows and a video one, which would be deinterlaced whole, every film frame rebuilt from one field (about 1.1 dB): mixed instead. A telecined film under grain that fieldmatch could not put back still keeps its cadence over 75 % of its blocks: mixed |
| Film | every window that sees is film or whole, one at least film | whole windows are a film's frames whole already |
| Mixed | any other mix, a detail window among them | a film with a 60i tail or middle; film and 60i half and half |
| Telecine | a film at 29.97 or 30, compared as numbers (30000/1001, 2997/100 and an MP4's 7180812/239593 alike), whose 3:2 cadence holds in every 5-frame block idet sees a repeat in (at least 20), followed across cuts and held pictures | the cadence's phase is a block's one top and one bottom repeated field; a block repeating one of the two at its place is on it (a held picture hides the other); blocks with no repeat and no combing count neither way, nor does a window's first block (idet sees no repeat in its first frame), nor a window's only pair in its last block (a window cut short by the end). One block off it (a cut after the pulldown) and decimate would drop a film frame there: the film is field-matched instead, every frame kept. Never at 25: decimated, it would come out at 20 |
| 2:2 film released | at its own rate: no 3:2 block, repeated fields in at most 0.5 % of its frames, its sample at least 50 % determined | a euro pulldown repeats a field every 12 frames, 7 % of them; a slow object over a still in fields looks whole once matched, every second moment gone, its sample 22 % determined |
| Undecided | fewer than 50 frames determined, whatever the mark: the safe chain and `unexamined` | a still in fields determines none of the frames idet reads |
| Failed | nothing read; a window whose run timed out, did not start, exited non-zero, or read fewer than its 200 frames before the end (near the end, fewer than all but a tenth of what is left, and two frames); no duration; a probe without the picture's size; an HEVC field-pictures check that could not tell: the safe chain and `unexamined` | a failed window is no verdict on its part of the title, and without a duration only the start could be read |
| The field order | idet's, where 80 % of its TFF and BFF frames say one; else the mark's (`tt`, `tb` top first; `bb`, `bt` bottom first); else top first. Never the frames' own flag | an unmarked stream's frames say bottom first by default. A field-rate deinterlace is made only in an order idet saw: in a wrong one it costs about 7.5 dB |

### The read-through

A plan whose sample's word would let the original go is encoded on the CPU
with three more branches in the same run, each `idet` on the middle of the
picture, printed to a file and sent to a null output: every source frame as
it is, every frame after `fieldmatch` (a film's chain), and every frame the
fields chain makes, before the rungs are made of them. A copy (whole frames
marked as fields) is decoded once through for it. The word stands when:

| Word | The whole title must show |
| --- | --- |
| every word | the output read through to its last frame (as many as the top rung's packets; the source's frames as many as the rate change makes them); no more combed output frames than 0.2 % of those determined (and 2), never 2 in a row |
| `progressive` | every chunk of 200 source frames whole, and nowhere 5 frames in a row combed one way |
| `deinterlaced-field` | every chunk in fields combing in the order taken, and nowhere 5 frames in a row the other way |
| `inverse-telecine`, decimated | no 5-frame block off the 3:2 cadence, and every chunk film or whole after fieldmatch |
| `inverse-telecine` at its own rate | repeated fields in at most 0.5 % of the frames, and every chunk film or whole after fieldmatch |

Where it does not, the record says `unverified`, and the title is made
again with the chain that is safe for what it holds: video with a part in
the other order at the frame rate (`deinterlaced-frame`), anything else
with the field-matched chain, never decimated (`field-matched`). A title not
read through at all keeps `unverified` as it was made. The read-through
logs `transcoder.item.fields_verified`, a title made again
`transcoder.item.fields_remade`.

What it costs: the transcoder's own encode command for a plan, with and
without the read-through's branches, on made 30 s clips, timed at normal
priority on one shared machine, the quicker of two runs each way, back to
back. The figures are relative: what the branches add to the same encode.

| Source (made) | Encoder | Time added | CPU time added |
| --- | --- | --- | --- |
| 1080i50 to 1080p50 | libx265 ultrafast | 4 % | 4 % |
| 1080i50 to 1080p50 | libx265 medium | 3 % | 1 % |
| 576i50 to 576p50 | libx265 ultrafast | 1 % | 3 % |
| 576i50 to 576p50 | libx265 medium | none measurable | 1 % |
| 480-line 3:2 film to 23.976 | libx265 ultrafast | 18 % | 5 % |
| 480-line 3:2 film to 23.976 | libx265 medium | 1 % | 2 % |

Three runs of `idet` on the middle of the picture are little beside an
encode; where the encode itself is quick (ultrafast on a 480-line film),
the graph waits on them, and the time grows more than the work. A copy's
read-through is one decode of the whole title on the CPU, where the copy
itself decodes nothing.

Known limits, all kept and none released: a 3:2 film edited after its
pulldown (a cut at another place of the cadence), under heavy grain (its
repeats hidden) or animated on threes is field-matched at 29.97, not
decimated, every film frame kept and one in five shown twice; a euro
pulldown is field-matched at 25; a sharp progressive picture marked as
fields gets the safe chain, softened where it flags frames; a slow video over
a still is field-matched; a source marked progressive with fields in one
order under the processing share is left as it is, `unverified`.
What is released: video in fields throughout, deinterlaced at the field rate
in the order idet saw, read through; a 3:2 film whose cadence holds through
the whole title, decimated; a 2:2 film, shifted by a field or not, with no
repeated fields; whole frames marked as fields with no fields anywhere in
them.

### Closed captions

Closed captions carried in the video (A53 `cc_data`, which the decoder hands
on with each frame) come through every fields chain pair for pair: each
valid CEA-608 pair of the source is in the package once, in the order the
source shows them, or none is.

- Deinterlaced or field-matched, every frame is kept: `fieldmatch` makes one
  frame of each, with its own captions, and `bwdif` re-times them at its
  output rate through ffmpeg's caption fifo (`libavfilter/ccfifo.c`), or
  hands each frame's on as they are where that fifo has no table for the
  rate.
- Inverse telecined, `decimate` drops a frame with its captions, a fifth of
  them; in ffmpeg 7.1 the caption fifo serves `fps`, `bwdif`, `yadif`,
  `tinterlace` and `ccrepack` only, and a `ccrepack` after `decimate` finds
  them gone. So the frames are split ahead of it. Decimate's frames lose
  their own captions and are numbered on the film's grid; every frame before
  it is put a tenth of a frame ahead of the film frame at four fifths of its
  number, rounded up; `interleave` puts both in that order, and `fps` at the
  film's rate drops every one of those frames, its caption fifo keeping their
  captions for the film frame next out, and puts out each film frame once (a
  pad frame after the last keeps the last from fps's end). A pair comes out
  between one source frame (33 ms) early and a fifth of a film frame (8 ms)
  late. The pictures and their timestamps are decimate's, but where decimate
  drops the very first frame (no repeat among the first five), which fps then
  puts out in its own place ahead of them. The film's rate is 24000/1001 or
  24/1 (the rates the fifo has a table for), however the source's 29.97 or
  30 is written.
- At the field rate, bwdif re-times a frame's captions onto its two fields'
  frames only from 29.97 or 30 (to 59.94 or 60), the stream's rate and its
  average alike. At any other rate it would hand each frame's captions on
  with both, every pair twice: there they are left out
  (`sidedata=mode=delete:type=A53_CC` after bwdif, in `scan.filter`), the
  package carries none, and its original, which has them, is kept for them.

The deletion gate compares an original's valid pairs a second with its
package's, window by window (katalog-manager, through the analyzer's
read): a package's window shows the pairs its original's shows, to within
those of one source frame at each of its edges. `tests/test_fields_real.py`
counts them pair by pair on made clips whose every picture's captions are
numbered (`tests/fieldcode.py`): a 3:2 film inverse telecined, one cut at
every place of its cadence, a film with a video part field-matched, 60i at
the frame and at the field rate: every pair once and in order. 50i at the
field rate: none.
`decimate` alone, on the same 30 s film, loses a fifth of them.

### HEVC field pictures

HEVC coded as field pictures (a picture a field; ffmpeg 7.1 decodes them as
1920×540 frames at 50 a second, the stream marked progressive) is found by
its parameter sets, the VUI's `field_seq_flag` (`trace_headers` over the first
5 s of it, bounded by `-t` before `-i`, nothing decoded, for every HEVC
source), and copied as before, its record `fields-kept` whatever idet would
read in frames half as tall. The read is bounded on the input: `-frames:v 1`
on the output does not stop ffmpeg 7.1, which traced nearly every packet of a
feature-length remux through it, millions of lines. A check that times out
(120 s), fails, sees no sequence parameter set, or sees one without the VUI
that carries the flag can't tell, and fails safe: the safe chain and
`unexamined`. Its copy has a `renditions.json` all the same, every rung
`"file": null`: its source record says progressive (ffprobe's mark), so its
package's scan is what keeps its original. A stream that sets the flag on
woven frames (an encoder told so) is kept the same way.

### The rate

`DEINTERLACE` (`frame`, the default, or `field`) is the rate video in
fields is deinterlaced at. At its field rate no motion is dropped, every
field becomes a frame; at its frame rate every second field's moment is
gone, and its original is kept. A telecined film is brought back to its own
frames either way. The keyframes stay every `SEGMENT_SECONDS`, at the
encoded rate.

The default goes to `field` once a 1080p50 package plays everywhere it
must: the streaming side's live H.264 rendition at the level 50 or 59.94
frames need (it is encoded at Level 4.0, which 1080p50 is above), the
players asking for the HEVC level 1080p50 needs (4.1, where 25 frames fit
4.0; a 576-line one 3.1, where 25 fit 3.0), and NVENC's encode of 1080p50
within the caps.

### What it says

The step's details carry `scan=<found>:<by>:<done>@<frame rate>`, in the
contract's words: `scan=fields:pictures:deinterlaced-frame@25/1`,
`scan=fields:pictures:deinterlaced-field@50/1`,
`scan=telecine:pictures:inverse-telecine@24000/1001`,
`scan=film:pictures:inverse-telecine@25/1`,
`scan=mixed:pictures:field-matched@30000/1001`,
`scan=progressive:pictures:progressive@25/1`,
`scan=progressive:pictures:unverified@25/1` (the read-through, or the share),
`scan=unknown:failed:unexamined@25/1`,
`scan=field-pictures:bitstream:fields-kept@50/1`; nothing for a source
nothing was examined of. The reason of a rung that would have been a copy
is `<codec>_not_copyable:<found>`. `transcoder.item.fields` logs the mark,
the verdict, the order and idet's counts; `transcoder.fields.window_failed`
a window that could not be read.

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
| 4:2:2, 4:4:4 or 12-bit; H.264 High 10; MPEG-2, VC-1, …; interlaced (ffprobe's field order `tt`, `bb`, `tb` or `bt`, or found in fields by its pictures: [Fields](#fields)); Dolby Vision 5 (reshaped from system memory) | on the CPU |
| a source whose encode a CPU encoder takes part in (libx264 rungs on a host without `h264_nvenc`) | on the CPU |
| a source whose ladder has more than one HEVC rung (`source,1080p:hevc`; below) | on the CPU |
| a source NVDEC hands on at another size than its picture's: cropped at the top or the left (below) | on the CPU |

Per rung, the GPU's filters do what the CPU's do:

| Rung | CPU | GPU |
| --- | --- | --- |
| the source's size and bit depth | `format=p010le` (10-bit), `format=yuv420p` or none (8-bit) | `format=cuda` |
| the source's size, 10-bit of an 8-bit source | `format=p010le` | `scale_cuda=format=p010le` |
| scaled | `scale=W:H,setsar=1,format=…` | `scale_cuda=W:H:format=…:interp_algo=bilinear,setsar=1`, halved first while the picture is more than 3 times the rung's size (4K to 480p: 1920×1080, then 854×480) |
| 8-bit (H.264) of a 10-bit SDR source | the scale, then `format=yuv420p` (swscale's dither) | the scale on the GPU at 10 bits, then `hwdownload` and the same `format=yuv420p` |
| H.264 of an HDR source | the scale, then the tone-map | the scale on the GPU, then `hwdownload` and the same tone-map, at the rung's size |

One encoder of a run takes the frames in GPU memory as they are: the
first rung that needs nothing of the CPU (neither a tone-map nor a
reduction to 8 bits), the largest. Every other rung is scaled on the GPU
all the same and comes down (`hwdownload`) in the format the CPU path
gives its encoder (`p010le`, or `yuv420p` for 8 bits).

NVENC now and then sizes the DPB of an HEVC encode one frame larger than
its own and raises its level to fit it: a 720p rung from level 3.1 to
4.0, a 1080p one to 5.0, a 4K one to 6.0, which a decoder for 4.1 or 5.1
turns down. Neither `-dpb_size` nor `-level` holds it. Two HEVC encodes
opening at once did it in 1 of 6 runs, so a ladder of more than one HEVC
rung is decoded on the CPU. One HEVC rung beside H.264 rungs did it in 17
of 20 runs whose PTX cache (`CUDA_CACHE_PATH`, below) was empty, and in
none of 52 runs of four ladders once the cache held the kernels of
`scale_cuda`, as the startup probe leaves it. A second CUDA device for the
encoders of system memory frames, so that they open contexts of their
own, made no difference, and is not opened. Every NVENC output is checked
for it (below): a GPU run it happened in is made again on the CPU.

Before a source is decoded on the GPU, its first frame is (about 0.5 s):
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
hands on in system memory after all fails the run at its first frame.
Every run is watched, the GPU's, its CPU fallback and a run on the CPU's
decode: when its progress (frames, bytes written, time encoded) stands
still for 10 minutes, it is taken for a call into NVDEC, NVENC or CUDA
that will not return, and killed; a run on the CPU's decode then fails
its step. One that has not died 30 s after its kill fails its step, and
`/healthz` answers 503 from then on, for the kubelet to restart the pod.
A GPU run that fails, stands still, writes a decoder error (a frame it
did not decode: `Decoding error`, `hardware accelerator failed`, a CUDA
error), or makes an output that is not its rung's size or not what NVENC
makes of system memory frames (below), whatever its exit code, is logged
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
nothing handed off`. The DPB is read from the parameter sets
(`trace_headers`); one that cannot be read (another ffmpeg writes them
out in other words) sends a GPU run to the CPU, and on the CPU's decode
is logged as an error (`transcoder.encode.nvenc_unread`) and handed off.

At startup NVENC's own output is checked the same way (one log line,
`transcoder.nvenc.selftest`): 2 s of 720p encoded to a source-size HEVC
rung and a 480p H.264 rung, those whose encoder is NVENC, with the CPU
path's command and the settings every encode takes. When it fails the
check (another GPU or driver, whose NVENC picks another DPB, level or
profile), no item and no extra is taken and `/readyz` answers 503 with
why, rather than every NVENC encode failing its step on each retry. When
its DPB cannot be read, every source is decoded on the CPU, where the
check logs it.

Where the source was decoded is in the step's details, `decode=cuda`,
`decode=cpu(<why not>)` (`cpu(cuda off)`, `cpu(4:2:2)`, `cpu(interlaced)`,
`cpu(codec=mpeg2video)`, `cpu(encoder=libx264)`, `cpu(hevc_rungs=2)`,
`cpu(nvdec_frame=1920x1080)`, …) or `decode=cpu(cuda failed)`, and in the
log lines `transcoder.encode.start` and `transcoder.encode.done`
(`decoder`) and `transcoder.item.done` (`decode`).

A rung at the source's size is the CPU path's bit for bit: NVDEC decodes
bit-exactly and NVENC is deterministic. Every rung has the CPU path's
size, pixel format, colour tags, sample aspect ratio, level, DPB, frames,
timestamps and keyframes, rate control and closed captions. A scaled rung
is not the CPU path's: `scale_cuda`'s bilinear is softer than swscale's
bicubic. Against a lanczos reference (a test pattern with grain), PSNR
Y/U/V:

| Rung | CPU path | GPU |
| --- | --- | --- |
| 4K 10-bit to 1080p | 59.3 / 55.4 / 51.4 dB | 47.5 / 46.1 / 42.3 dB |
| 4K 10-bit to 480p | 58.3 / 49.9 / 45.8 dB | 48.1 / 42.4 / 38.9 dB |
| 1080p to 720p | 56.9 / 55.9 / 50.0 dB | 48.7 / 47.8 / 41.4 dB |
| 1080p to 360p | 57.2 / 54.8 / 49.5 dB | 46.8 / 43.6 / 39.8 dB |

(SSIM Y 0.993 to 0.999 for the GPU, 0.9995 or more for the CPU path.) An
encode at the rung's maxrate loses more than either: the GPU's and the CPU
path's encodes of a rung are at SSIM 0.988 to 0.995 to each other. No
other way of `scale_cuda` came closer: its bicubic and lanczos alias, and
its lanczos shifts the chroma. An 8-bit rung of a 10-bit source is taken
to 8 bits by swscale, with its dither, as on the CPU path: scale_cuda would
cut the two bits off, darken the picture by half a step on average, and
band a ramp.

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
  "dolbyVision": {"profile5": "reshaped", "profile7": "baseLayer"},
  "interlacing": {"deinterlace": "frame", "inverseTelecine": true, "judgedBy": "pictures"}}}
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
title of a refused profile to be processed again. `interlacing` is what
becomes of a source in fields ([Fields](#fields)): deinterlaced at the rate
`deinterlace` names (`DEINTERLACE`), a telecined film brought back to its
own frames, judged by its pictures. `version` is the image's, `IMAGE_VERSION` (the image
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
| `DEINTERLACE` | `frame` | The rate a source in fields is deinterlaced at: `frame` (25 or 29.97; its original is kept) or `field` (its field rate, 50 or 59.94); see [Fields](#fields) |
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
`tests/test_fields_real.py` makes a clip of each kind of source in fields
(true 50i, 60i bottom field first, a telecined film, whole frames marked as
fields, an MPEG-2 film transfer with a logo in fields, mixed, film with a
60i part, HEVC field pictures) and runs each
end to end; it needs ffmpeg's `idet`, `bwdif`, `fieldmatch`, `decimate`,
`telecine`, `tinterlace` and `setfield` filters.

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
