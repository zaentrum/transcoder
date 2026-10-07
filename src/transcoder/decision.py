"""Skip-vs-encode decision and rendition-ladder planning for one source.

User rule (locked) for the default single rendition:
  * Source video is HEVC every HEVC device decodes — Main or Main 10,
    4:2:0, at most 10 bits — and within its cap -> SKIP (no re-encode).
    Any other HEVC (4:2:2, 4:4:4, 12-bit) -> ENCODE to 4:2:0 HEVC at its
    own size.
  * The cap ("wasteful" sources): by the item's type and the tallest
    video's height (the 2160 bucket from 2000 lines), a movie's or an
    extra's video above 8 / 14 Mbit/s, or a movie file above 15 GiB, an
    episode's video above 6 / 8 Mbit/s -> ENCODE once at its own size,
    capped there (VBV maxrate = the cap). Every source-size HEVC encode
    takes the same cap as its maxrate (Caps).
  * Anything else (H.264, AV1, MPEG-2, ...) -> ENCODE to HEVC.
  * An HEVC encode keeps the source's bit depth: Main 10 for a source
    with more than 8 bits (HDR as HDR, SDR as SDR with its colour tags),
    Main for an 8-bit one. Originals are deleted once packaged, so
    dropping a 10-bit source to 8 bits would lose it for good.
  * Dolby Vision is packaged as its base layer when other devices play
    that (profile 8.1 / 8.2 / 8.4 and the like). Profile 5 has no such
    base layer, and profile 7 is refused as well: copied, or encoded
    without a tone-map, they play in the wrong colours. They get no
    plan at all (SourceError): the step fails, the title stays
    unpackaged, and its original is kept.

Hardware-tier checks are deliberately NOT here. The bitrate cap is the
one size rule: a copy is kept unless its video is wasteful for its kind
and resolution, so we don't ship a 50 Mbps remux of a 2160p Blu-ray
straight into HLS, and every encode is capped the same way.

On top of that rule sits the optional ladder (`LADDER` env, see the
README "Rendition contract"): extra, smaller renditions so a client that
cannot decode the top rung — or cannot sustain its bitrate — has
something to fall back to. `plan_renditions()` turns a ladder spec plus
the probed source into a concrete list of rungs: which ones are stream
copies, which ones are encodes, at what size, with which encoder.

The CPU rule: without NVENC, an HEVC encode is libx265 — hours per
title. A browser-friendly H.264 source therefore passes through
untouched at the source rung instead of being re-encoded to HEVC; only
sources nothing can play (MPEG-2, VC-1, AV1, Hi10P, ...) pay for x265.
A ladder that names the codec opts out: `source:hevc` is the HEVC-only
ladder, one rendition at the source's own size — the source's video
copied when it is HEVC, else ONE HEVC encode, NVENC or else libx265,
whatever the source.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# Codec names ffprobe reports for HEVC content. `hevc` is the canonical
# one; older mux toolchains occasionally tag it as `h265`. Both pass
# through the packager untouched, so both count as "no re-encode".
HEVC_CODEC_NAMES = {"hevc", "h265"}

# The HEVC a package carries as it is: what every HEVC decoder plays —
# ffprobe's "Main" or "Main 10", 4:2:0, at most 10 bits. The rest (Rext's
# 4:2:2 / 4:4:4 / 12-bit, SCC, ...) is re-encoded to 4:2:0.
HEVC_COPY_PROFILES = {"main", "main 10"}
HEVC_COPY_MAX_BITS = 10

# Dolby Vision. A stream's configuration record (ffprobe's stream side
# data "DOVI configuration record") names its profile and its base
# layer's signal compatibility: 1 HDR10, 2 SDR, 4 HLG, 6 Blu-ray HDR10,
# 0 none. A base layer other devices play (8.1, 8.2, 8.4, ...) is
# packaged as that base layer. Profile 5 has none (its pictures are
# IPT-PQ-c2: wrong colours on any device without Dolby Vision), profile 7
# is refused too, and so is any base layer of compatibility 0: those
# need a tone-mapping encode.
DOVI_SIDE_DATA = "DOVI configuration record"
DOVI_REFUSED_PROFILES = {5, 7}
# The sample entries a stream whose base layer no other decoder plays
# is given (dvhe / dvh1 HEVC, dav1 AV1, dvav / dva1 AVC): Dolby Vision
# of that kind even when the record itself is missing.
DOVI_ONLY_SAMPLE_ENTRIES = {"dvhe", "dvh1", "dav1", "dvav", "dva1"}

# H.264 is only "browser-friendly" in 8-bit 4:2:0. High 10 / 4:2:2 /
# 4:4:4 decode almost nowhere in hardware, so those get re-encoded like
# any other exotic source.
H264_PASSTHROUGH_PIX_FMTS = {"yuv420p", "yuvj420p"}
H264_PASSTHROUGH_PROFILES = {"baseline", "constrained baseline", "main", "high", ""}

# Transfer characteristics that mark a source as HDR (PQ / HLG).
HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}

# A pixel format's chroma subsampling and bits per sample, read off the
# name ffprobe reports: planar `yuv420p10le` (4:2:0, 10), semi-planar
# `p010le` / `nv12` (4:2:0, 10 / 8), packed `y210le` (4:2:2, 10), grey
# `gray12le` (4:0:0, 12), planar RGB `gbrp10le` (4:4:4, 10). The chroma
# is "420", "422", "444", "440", "411", "410" or "400" (grey).
_PLANAR_YUV = re.compile(r"^yuv[aj]?(4[0-4][0-4]|41[01])p(\d+)?(?:le|be)?$")
_SEMI_PLANAR = re.compile(r"^p([024])(\d\d)(?:le|be)?$")
_SEMI_PLANAR_CHROMA = {"0": "420", "2": "422", "4": "444"}
_GREY = re.compile(r"^(?:gray|grey|ya)(\d+)?(?:le|be)?$")
_PLANAR_RGB = re.compile(r"^gbra?p(\d+)?(?:le|be)?$")
_NAMED_LAYOUTS: dict[str, tuple[str, int]] = {
    "nv12": ("420", 8), "nv21": ("420", 8), "nv16": ("422", 8), "nv20le": ("422", 10),
    "nv20be": ("422", 10), "nv24": ("444", 8), "nv42": ("444", 8),
    "yuyv422": ("422", 8), "uyvy422": ("422", 8), "yvyu422": ("422", 8),
    "y210le": ("422", 10), "y212le": ("422", 12), "xv30le": ("444", 10),
    "xv36le": ("444", 12), "vuya": ("444", 8), "vuyx": ("444", 8), "ayuv64le": ("444", 16),
}


def pix_fmt_layout(pix_fmt: str) -> tuple[str | None, int | None]:
    """(chroma, bits per sample) of an ffprobe pixel format name:
    "yuv422p10le" -> ("422", 10), "p010le" -> ("420", 10), "nv16" ->
    ("422", 8); (None, None) for a name it doesn't know."""
    name = (pix_fmt or "").lower()
    if name in _NAMED_LAYOUTS:
        return _NAMED_LAYOUTS[name]
    if m := _PLANAR_YUV.match(name):
        return m.group(1), int(m.group(2) or 8)
    if m := _SEMI_PLANAR.match(name):
        return _SEMI_PLANAR_CHROMA[m.group(1)], int(m.group(2))
    if m := _GREY.match(name):
        return "400", int(m.group(1) or 8)
    if m := _PLANAR_RGB.match(name):
        return "444", int(m.group(1) or 8)
    return None, None


def _int(raw: object) -> int | None:
    """An ffprobe integer (7, "10"), or None ("N/A", absent)."""
    try:
        return int(str(raw))
    except (TypeError, ValueError):
        return None


def _positive(raw: object) -> int | None:
    """An ffprobe integer above 0, or None."""
    value = _int(raw)
    return value if value is not None and value > 0 else None


# The statistics tags a Matroska muxer writes per track (mkvmerge's BPS,
# BPS-eng from older versions): the track's average bit rate.
STATISTICS_TAGS = ("BPS", "BPS-eng")


def _statistics_rate(stream: dict[str, Any]) -> tuple[int | None, str]:
    """A stream's BPS statistics tag, and its name."""
    tags = {str(k).upper(): (k, v) for k, v in (stream.get("tags") or {}).items()}
    for name in STATISTICS_TAGS:
        if name.upper() in tags:
            key, value = tags[name.upper()]
            if (rate := _positive(value)) is not None:
                return rate, str(key)
    return None, ""


def video_bit_rate(probe: dict[str, Any]) -> tuple[int | None, str]:
    """The video stream's own average bit rate, bit/s, and where it was
    read — the cap is judged on it, not on the whole file:

      1. "stream": ffprobe's bit_rate of the video stream (MP4, TS with a
         known rate, ...);
      2. "tag:BPS" / "tag:BPS-eng": the Matroska statistics tag, which a
         Matroska muxer writes per track and ffprobe does not turn into
         a bit_rate;
      3. "size-minus-audio": the file's size x 8 / its duration, less the
         audio streams' bit rates (each its bit_rate, else its BPS tag;
         one with neither counts 0, which errs high);
      4. "unknown": none of those (no size or duration) — the bit rate
         rule then breaks nothing.
    """
    video = probe.get("video") or {}
    if (rate := _positive(video.get("bit_rate"))) is not None:
        return rate, "stream"
    rate, tag = _statistics_rate(video)
    if rate is not None:
        return rate, f"tag:{tag}"
    size = _positive(probe.get("size_bytes"))
    duration_ms = _positive(probe.get("duration_ms"))
    if size is not None and duration_ms is not None:
        audio = 0
        for stream in probe.get("audio") or []:
            audio += _positive(stream.get("bit_rate")) or _statistics_rate(stream)[0] or 0
        rate = round(size * 8 * 1000 / duration_ms) - audio
        if rate > 0:
            return rate, "size-minus-audio"
    return None, "unknown"

# The rung codec each ladder token gets when the spec doesn't name one:
# the source-resolution rung stays HEVC (the catalog rule above), every
# scaled rung is H.264 because that is what every device decodes.
DEFAULT_SOURCE_CODEC = "hevc"
DEFAULT_SCALED_CODEC = "h264"
RUNG_CODECS = {"hevc", "h264"}

# Bitrate caps (VBV maxrate, Mbps) for scaled rungs by box height. The
# encoders run capped-quality (CRF / CQ), so these bound the peaks, not
# the average — film content usually lands well below. A source-size
# HEVC rung takes its cap from Caps instead (NVENC_MAXRATE_* override).
_RUNG_MAXRATE_MBPS: dict[int, dict[str, float]] = {
    2160: {"h264": 16.0, "hevc": 14.0},
    1440: {"h264": 9.0, "hevc": 8.0},
    1080: {"h264": 6.0, "hevc": 8.0},
    720: {"h264": 3.0, "hevc": 2.5},
    540: {"h264": 2.0, "hevc": 1.6},
    480: {"h264": 1.5, "hevc": 1.2},
    360: {"h264": 0.8, "hevc": 0.6},
}

_RUNG_TOKEN = re.compile(r"^(source|\d{3,4}p)$")
_MAXRATE_TOKEN = re.compile(r"^(\d+(?:\.\d+)?)([km])$")

# ---------------------------------------------------------------- the cap
# A source's height bucket: the tallest video stream's height (not
# counting cover art), 2160 from 2000 lines up, 1080 below. A 3840x1600
# scope picture is in the 1080 bucket.
UHD_MIN_HEIGHT = 2000
GIB = 1024 ** 3

# The maxrate of a source-size HEVC encode whose cap is switched off
# (0) and that NVENC_MAXRATE_* does not set: the movies' default caps.
FALLBACK_MAXRATE_BPS = {1080: 8_000_000, 2160: 14_000_000}


def height_bucket(height: int) -> int:
    """1080 or 2160: the cap bucket of a source this many lines tall."""
    return 2160 if height >= UHD_MIN_HEIGHT else 1080


def cap_kind(item_type: str) -> str:
    """The caps an item takes: an episode's, or a movie's (a movie, an
    extra, or anything else)."""
    return "episode" if item_type == "episode" else "movie"


@dataclass(frozen=True)
class Caps:
    """The bitrate caps of a package's video, bit/s, by the item's kind
    (cap_kind) and its source's height bucket, and the movies' file-size
    rule, bytes. A source the packager would copy is copied only within
    them, and every source-size HEVC encode takes the cap as its VBV
    maxrate (bufsize twice that). 0 switches a rule off."""
    movie_1080: int = 8_000_000
    movie_2160: int = 14_000_000
    movie_max_bytes: int = 15 * GIB
    episode_1080: int = 6_000_000
    episode_2160: int = 8_000_000

    def rate(self, item_type: str, bucket: int) -> int:
        """The bitrate cap, bit/s; 0 = none."""
        if cap_kind(item_type) == "episode":
            return self.episode_2160 if bucket == 2160 else self.episode_1080
        return self.movie_2160 if bucket == 2160 else self.movie_1080

    def max_bytes(self, item_type: str) -> int:
        """The file-size rule, bytes; 0 = none (episodes never have one)."""
        return self.movie_max_bytes if cap_kind(item_type) == "movie" else 0


@dataclass(frozen=True)
class CapCheck:
    """The cap rules as they apply to one source: its kind and bucket, the
    two rules (0 = off), and which one the source breaks — "bitrate" or
    "size" — if any."""
    item_type: str
    bucket: int
    rate_bps: int
    max_bytes: int
    over: str | None


DEFAULT_CAPS = Caps()
# A plan's cap when it was made without any (rules off).
NO_CAP = CapCheck("movie", 1080, 0, 0, None)


@dataclass(frozen=True)
class Decision:
    """Outcome of `_should_transcode_video(probe)`."""
    skip: bool
    reason: str
    source_codec: str
    width: int
    height: int


def decide(probe: dict[str, Any]) -> Decision:
    """Take a parsed ffprobe payload (shape from `transcoder.ffmpeg.ffprobe`)
    and return the skip-vs-encode decision plus a one-line reason for
    the audit row.

    Defensive: a missing/empty video stream -> encode (so we go through
    NVENC and produce a known-good prepared.mkv) rather than skip with
    bad data. If the file really is video-less, ffmpeg will fail
    loudly downstream.

    The codec check only: the encode path plans with plan_renditions,
    whose copy rule is the authority (HEVC Main or Main 10, 4:2:0, at
    most 10 bits; see SourceInfo.hevc_copy_blocker).
    """
    video = probe.get("video") or {}
    codec = (video.get("codec_name") or "").lower()
    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)

    if not codec:
        return Decision(
            skip=False,
            reason="no_video_stream_in_probe",
            source_codec="",
            width=width,
            height=height,
        )

    if codec in HEVC_CODEC_NAMES:
        return Decision(
            skip=True,
            reason=f"source_already_hevc:{codec}",
            source_codec=codec,
            width=width,
            height=height,
        )

    return Decision(
        skip=False,
        reason=f"non_hevc_source:{codec}",
        source_codec=codec,
        width=width,
        height=height,
    )


# ------------------------------------------------------------------ ladder
class LadderError(ValueError):
    """Raised for an unparseable LADDER spec. Surfaces at startup so a
    typo never silently degrades to the single-rendition default."""


class SourceError(ValueError):
    """A source no ladder packages as it is: copied, or encoded without a
    tone-map, its package would play in the wrong colours (Dolby Vision
    profile 5 or 7, say). The transcode step fails with this message and
    nothing is handed off, so the title stays unpackaged and its original
    is never retired."""


@dataclass(frozen=True)
class RungSpec:
    """One token of the LADDER env var, e.g. `720p:h264:3M`."""
    name: str                   # "source" or "<height>p"
    height: int | None          # None for "source"
    codec: str                  # "hevc" | "h264"
    maxrate_bps: int | None     # explicit cap, None = defaults table
    # The token names its codec (`source:hevc`) instead of leaving it to
    # the default: a requirement, which the CPU rule never trades for an
    # H.264 pass-through (plan_renditions, rule 3).
    codec_named: bool = False


def parse_ladder(spec: str | None) -> list[RungSpec]:
    """Parse `LADDER`. Grammar: comma-separated rungs, each
    `<source|NNNp>[:<hevc|h264>][:<maxrate, e.g. 3M or 1500k>]`.

    Empty / unset -> the single source rung (today's behaviour), so an
    install that never sets LADDER keeps exactly one rendition per item.
    `source:hevc` is that rung with its codec named, which makes HEVC a
    requirement: the HEVC-only ladder, on a CPU host too.
    """
    tokens = [t.strip().lower() for t in (spec or "").split(",") if t.strip()]
    if not tokens:
        return [RungSpec("source", None, DEFAULT_SOURCE_CODEC, None)]
    rungs: list[RungSpec] = []
    for tok in tokens:
        parts = [p.strip() for p in tok.split(":")]
        name = parts[0]
        if not _RUNG_TOKEN.match(name):
            raise LadderError(f"bad ladder rung {tok!r}: expected 'source' or e.g. '720p'")
        height = None if name == "source" else int(name[:-1])
        if height is not None and not 144 <= height <= 4320:
            raise LadderError(f"bad ladder rung {tok!r}: height out of range")
        codec = DEFAULT_SOURCE_CODEC if height is None else DEFAULT_SCALED_CODEC
        codec_named = False
        maxrate: int | None = None
        for extra in parts[1:]:
            if extra in RUNG_CODECS:
                codec, codec_named = extra, True
            elif m := _MAXRATE_TOKEN.match(extra):
                scale = 1_000_000 if m.group(2) == "m" else 1_000
                maxrate = int(float(m.group(1)) * scale)
            else:
                raise LadderError(
                    f"bad ladder rung {tok!r}: {extra!r} is neither a codec "
                    f"({'/'.join(sorted(RUNG_CODECS))}) nor a maxrate like 3M"
                )
        rungs.append(RungSpec(name, height, codec, maxrate, codec_named))
    return rungs


@dataclass(frozen=True)
class SourceInfo:
    """The probe facts the planner needs, pulled out of the raw ffprobe
    stream dict once so the rest of the code doesn't re-parse it."""
    codec: str
    width: int
    height: int
    pix_fmt: str
    profile: str
    fps: float
    frame_rate: str      # ffprobe's fraction string, e.g. "24000/1001"
    hdr: bool
    bit_rate: int | None
    start_time: float | None = None  # first video timestamp in the source
    sar: float = 1.0                 # sample (pixel) aspect ratio
    # Bits per sample (the pixel format's, else ffprobe's
    # bits_per_raw_sample; 8 when neither says) and chroma subsampling
    # ("420", "422", "444", ...; None when the pixel format is unknown).
    bit_depth: int = 8
    chroma: str | None = None
    # Dolby Vision, from the stream's configuration record: its profile
    # and base-layer compatibility id (None without a record), and the
    # stream's sample entry (ffprobe's codec_tag_string, lower-case).
    dv_profile: int | None = None
    dv_compat: int | None = None
    codec_tag: str = ""
    # What the cap is judged on (cap_check): the tallest video stream's
    # height, the file's size (bytes, None unknown), and the video's own
    # bit rate (bit/s, None unknown) with where it was read (video_rate).
    max_height: int = 0
    size_bytes: int | None = None
    video_bit_rate: int | None = None
    video_bit_rate_from: str = "unknown"

    @property
    def display_width(self) -> int:
        """Width in square pixels: a 720x576 16:9 DVD (SAR 64:45) is
        1024 wide on screen. Rungs are sized from this, with square
        pixels, so a scaled rung never inherits an odd SAR."""
        if abs(self.sar - 1.0) < 1e-3:
            return self.width
        return max(2, round(self.width * self.sar / 2) * 2)

    @classmethod
    def from_probe(cls, probe: dict[str, Any]) -> SourceInfo:
        v = probe.get("video") or {}
        rate = v.get("avg_frame_rate") or ""
        if not _fraction(rate):
            rate = v.get("r_frame_rate") or ""
        sar = _fraction((v.get("sample_aspect_ratio") or "").replace(":", "/")) or 1.0
        try:
            bit_rate = int(v.get("bit_rate")) if v.get("bit_rate") else None
        except (TypeError, ValueError):
            bit_rate = None
        raw_start = v.get("start_time")
        try:
            start_time = float(raw_start) if raw_start not in (None, "N/A") else None
        except (TypeError, ValueError):
            start_time = None
        pix_fmt = (v.get("pix_fmt") or "").lower()
        chroma, depth = pix_fmt_layout(pix_fmt)
        dovi = next((sd for sd in v.get("side_data_list") or []
                     if isinstance(sd, dict) and sd.get("side_data_type") == DOVI_SIDE_DATA), {})
        videos = [s for s in probe.get("videos") or [v] if isinstance(s, dict)]
        video_rate, video_rate_from = video_bit_rate(probe)
        return cls(
            codec=(v.get("codec_name") or "").lower(),
            width=int(v.get("width") or 0),
            height=int(v.get("height") or 0),
            pix_fmt=pix_fmt,
            profile=(v.get("profile") or "").lower(),
            fps=_fraction(rate) or 24.0,
            frame_rate=rate or "24/1",
            hdr=(v.get("color_transfer") or "").lower() in HDR_TRANSFERS,
            bit_rate=bit_rate,
            start_time=start_time,
            sar=sar,
            bit_depth=depth or _int(v.get("bits_per_raw_sample")) or 8,
            chroma=chroma,
            dv_profile=_int(dovi.get("dv_profile")),
            dv_compat=_int(dovi.get("dv_bl_signal_compatibility_id")),
            codec_tag=str(v.get("codec_tag_string") or "").lower(),
            max_height=max((_int(s.get("height")) or _int(s.get("coded_height")) or 0
                            for s in videos), default=0),
            size_bytes=_positive(probe.get("size_bytes")),
            video_bit_rate=video_rate,
            video_bit_rate_from=video_rate_from,
        )

    def cap_check(self, item_type: str, caps: Caps) -> CapCheck:
        """The cap rules for this source as an item of `item_type`, and the
        one it breaks: "bitrate" when its video is above the cap of its
        kind and bucket, "size" when it is a movie (or an extra) whose
        file is above the size rule. Both strictly above; an unknown bit
        rate or size breaks nothing."""
        bucket = height_bucket(self.max_height)
        rate, max_bytes = caps.rate(item_type, bucket), caps.max_bytes(item_type)
        over = None
        if rate and self.video_bit_rate is not None and self.video_bit_rate > rate:
            over = "bitrate"
        elif max_bytes and self.size_bytes is not None and self.size_bytes > max_bytes:
            over = "size"
        return CapCheck(cap_kind(item_type), bucket, rate, max_bytes, over)

    @property
    def h264_browser_friendly(self) -> bool:
        return (
            self.codec == "h264"
            and self.pix_fmt in H264_PASSTHROUGH_PIX_FMTS
            and self.profile in H264_PASSTHROUGH_PROFILES
        )

    @property
    def hevc_copy_blocker(self) -> str | None:
        """Why this HEVC source's video can't go into a package as it is
        ("4:2:2", "12-bit", "profile=rext", ...), or None when it can:
        Main or Main 10, 4:2:0, at most 10 bits. A profile ffprobe left
        empty is judged by the pixel format alone; with neither known,
        nothing is copied blind."""
        if self.chroma is not None and self.chroma != "420":
            return ":".join(self.chroma)
        if self.bit_depth > HEVC_COPY_MAX_BITS:
            return f"{self.bit_depth}-bit"
        if self.profile and self.profile not in HEVC_COPY_PROFILES:
            return "profile=" + self.profile.replace(" ", "_")
        if self.chroma is None and not self.profile:
            return "pix_fmt=unknown"
        return None

    @property
    def dolby_vision_blocker(self) -> str | None:
        """Why this source can't be packaged without a tone-mapping encode
        — Dolby Vision whose base layer no other device plays — or None.
        The record decides when there is one; without it, a Dolby
        Vision-only sample entry (dvh1, ...) does."""
        if self.dv_profile is not None:
            if self.dv_profile in DOVI_REFUSED_PROFILES:
                what = f"profile {self.dv_profile}"
            elif self.dv_compat == 0:
                what = f"profile {self.dv_profile}.0"
            else:
                return None
        elif self.codec_tag in DOVI_ONLY_SAMPLE_ENTRIES:
            what = f"({self.codec_tag}, no compatible base layer)"
        else:
            return None
        return f"Dolby Vision {what} needs a tone-mapping encode; kept the original"


def _fraction(rate: str) -> float | None:
    """'24000/1001' -> 23.976; '0/0' / '' -> None."""
    try:
        num, _, den = rate.partition("/")
        value = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return None
    return value if value > 0 and math.isfinite(value) else None


@dataclass(frozen=True)
class Encoders:
    """Which encoder serves each rung codec on this host. NVENC when the
    startup probe found a working one, the CPU encoder otherwise."""
    hevc: str = "hevc_nvenc"
    h264: str = "h264_nvenc"

    def for_codec(self, codec: str) -> str:
        return self.hevc if codec == "hevc" else self.h264

    @property
    def backend(self) -> str:
        nv = [e.endswith("_nvenc") for e in (self.hevc, self.h264)]
        return "nvenc" if all(nv) else ("cpu" if not any(nv) else "mixed")


CPU_ENCODERS = Encoders(hevc="libx265", h264="libx264")
NVENC_ENCODERS = Encoders()


@dataclass(frozen=True)
class RungPlan:
    """One concrete rendition the transcoder will hand to the packager."""
    id: str                  # v0, v1, ... (v0 = top rung, carries audio + subs)
    name: str                # ladder token it came from ("source", "720p")
    codec: str               # output codec: "hevc" | "h264"
    mode: str                # "copy" (stream copy of the source) | "encode"
    encoder: str             # "copy" | hevc_nvenc | h264_nvenc | libx265 | libx264
    width: int
    height: int
    scaled: bool             # True when the encode resizes the frame
    box_width: int           # the fit-within box (scaled rungs only)
    box_height: int
    maxrate_bps: int | None  # VBV cap for encodes
    tonemap: bool            # HDR source -> SDR BT.709 (H.264 rungs)
    ten_bit: bool            # HEVC Main 10: an HDR source, or one above 8 bits
    reason: str              # audit trail for copy-vs-encode

    @property
    def file(self) -> str | None:
        """Inbox-relative output file name; None for a stream copy (the
        packager then reads the item's original source). v0 keeps the
        legacy `prepared.mkv` name so an older packager still finds it."""
        if self.mode == "copy":
            return None
        return "prepared.mkv" if self.id == "v0" else f"{self.id}.mkv"


@dataclass(frozen=True)
class Plan:
    source: SourceInfo
    rungs: list[RungPlan]
    segment_seconds: int
    # The cap rules as they applied to this source.
    cap: CapCheck = NO_CAP

    @property
    def all_copy(self) -> bool:
        return all(r.mode == "copy" for r in self.rungs)

    @property
    def keyframes(self) -> str:
        """How the encoded rungs place their keyframes:
          * "interval" — every rung is encoded: IDR forced every
            segment_seconds, identical across rungs -> fixed-length,
            aligned segments.
          * "source" — a rung is a stream copy whose keyframes we can't
            move: the encoded rungs copy the SOURCE keyframe positions
            (and nothing else) so the packager cuts every rung at the
            same instants.
          * "none" — nothing is encoded.
        """
        modes = {r.mode for r in self.rungs}
        if modes == {"copy"}:
            return "none"
        return "source" if "copy" in modes else "interval"


def box_for_height(height: int) -> tuple[int, int]:
    """16:9 box for a ladder height: 720 -> 1280x720, 480 -> 854x480."""
    width = int(math.ceil(height * 16 / 9 / 2)) * 2
    return width, height


def fit_within(src_w: int, src_h: int, box_w: int, box_h: int) -> tuple[int, int]:
    """Largest even size of the source's aspect ratio inside the box —
    the same numbers ffmpeg's scale=w=BW:h=BH:force_original_aspect_ratio
    =decrease:force_divisible_by=2 picks (the other side rounded to the
    NEAREST even number: 1918x802 into 854x480 -> 854x358; verified
    identical in ffmpeg 7.1 and 8.1). The encode scales to exactly this
    size, so the plan, the encode and renditions.json agree."""
    if src_w <= 0 or src_h <= 0:
        return box_w, box_h
    tmp_w = (box_h * src_w + src_h) // (2 * src_h) * 2
    tmp_h = (box_w * src_h + src_w) // (2 * src_w) * 2
    w, h = min(tmp_w, box_w), min(tmp_h, box_h)
    return max(2, w - w % 2), max(2, h - h % 2)


def _source_maxrate(height: int, codec: str, cap: CapCheck,
                    overrides_mbps: tuple[int | None, int | None]) -> int:
    """VBV maxrate of a source-size encode. HEVC: the NVENC_MAXRATE_*
    override of the source's bucket when one is set, else the cap of its
    kind and bucket, else (a cap switched off) the movies' default cap;
    the H.264 table otherwise."""
    if codec != "hevc":
        return _table_maxrate(height, codec)
    override = overrides_mbps[1] if cap.bucket == 2160 else overrides_mbps[0]
    if override:
        return int(override * 1_000_000)
    return cap.rate_bps or FALLBACK_MAXRATE_BPS[cap.bucket]


def _table_maxrate(height: int, codec: str) -> int:
    for h in sorted(_RUNG_MAXRATE_MBPS):
        if height <= h:
            return int(_RUNG_MAXRATE_MBPS[h][codec] * 1_000_000)
    return int(_RUNG_MAXRATE_MBPS[2160][codec] * 1_000_000)


def plan_renditions(
    probe: dict[str, Any],
    ladder: list[RungSpec],
    encoders: Encoders,
    *,
    segment_seconds: int = 6,
    nvenc_caps_mbps: tuple[int | None, int | None] = (None, None),
    item_type: str = "movie",
    caps: Caps = DEFAULT_CAPS,
) -> Plan:
    """Turn the ladder into concrete rungs for this source, an item of
    `item_type` ("movie", "episode", "extra"), under `caps`.
    `nvenc_caps_mbps` are the NVENC_MAXRATE_1080P / 2160P overrides of a
    source-size HEVC encode's maxrate (None: the cap's).

    Raises SourceError, before any rung, for a source no rung may copy or
    encode as it is: Dolby Vision profile 5 or 7, a base layer of
    compatibility 0, or a Dolby Vision-only sample entry (dvh1, ...).
    Dolby Vision with a compatible base layer (8.1, 8.2, 8.4) is planned
    as that base layer.

    Rules, in order:
      1. A rung never upscales: a box the source already fits in collapses
         to the source size.
      2. A rung at source size whose codec the source already has is a
         stream copy (HEVC source, HEVC rung -> the locked skip rule):
         HEVC only when every HEVC decoder plays it (Main or Main 10,
         4:2:0, at most 10 bits; else it is re-encoded to 4:2:0 at its
         own size) and the source is within its cap (SourceInfo.cap_check:
         else it is re-encoded at its own size, the cap its maxrate),
         H.264 only when it is browser-friendly.
      3. CPU rule: an HEVC source-size rung that would need libx265 on a
         browser-friendly H.264 source is a stream copy of the H.264 —
         unless the ladder names the rung's codec (`source:hevc`): then
         HEVC is a requirement and libx265 encodes it.
      4. Duplicates (same size + codec) collapse to the first one.
      5. Rungs sort largest first; v0 is the top and carries the audio
         and subtitle tracks for the packager.
      6. Bit depth: an HEVC encode is Main 10 when the source is HDR or
         has more than 8 bits — an SDR source stays SDR, its colour tags
         passed through (ffmpeg takes them from the decoded frames, and
         nothing here re-tags them) — and Main when it is 8-bit SDR.
         H.264 rungs are 8-bit; an HDR source's are tone-mapped.
      7. Maxrate: a scaled rung takes the table's; a source-size HEVC
         encode its kind's and bucket's cap (_source_maxrate); a maxrate
         the ladder names wins over both.
    """
    src = SourceInfo.from_probe(probe)
    if (refused := src.dolby_vision_blocker) is not None:
        raise SourceError(refused)
    cap = src.cap_check(item_type, caps)
    planned: list[RungPlan] = []
    seen: set[tuple[int, int, str]] = set()
    disp_w = src.display_width
    for spec in ladder:
        if spec.height is None or (src.width and src.height and src.height <= spec.height
                                   and disp_w <= box_for_height(spec.height)[0]):
            # Source-size rung: either the explicit "source" token or a box
            # the source already fits in (never upscale). Keeps the source's
            # SAR if it has one.
            width, height, scaled = src.width, src.height, False
            box_w, box_h = width, height
        else:
            box_w, box_h = box_for_height(spec.height)
            width, height = fit_within(disp_w, src.height, box_w, box_h)
            scaled = True

        codec = spec.codec
        hevc_at_source = not scaled and src.codec in HEVC_CODEC_NAMES and codec == "hevc"
        if hevc_at_source and src.hevc_copy_blocker is None and cap.over is None:
            mode, encoder, reason = "copy", "copy", f"source_already_hevc:{src.codec}"
        elif not scaled and codec == "h264" and src.h264_browser_friendly:
            mode, encoder, reason = "copy", "copy", "source_already_h264"
        elif (not scaled and codec == "hevc" and not spec.codec_named
              and encoders.hevc == "libx265" and src.h264_browser_friendly):
            # The CPU rule: keep the H.264 instead of hours of libx265.
            # Only for a codec left to the default; a named one is kept.
            codec, mode, encoder = "h264", "copy", "copy"
            reason = "cpu_passthrough_h264"
        else:
            mode, encoder = "encode", encoders.for_codec(codec)
            if scaled:
                reason = f"scale_to_{height}p"
            elif hevc_at_source and src.hevc_copy_blocker is not None:
                reason = f"hevc_not_copyable:{src.hevc_copy_blocker}"
            elif hevc_at_source:
                reason = f"hevc_over_cap:{cap.over}"
            else:
                reason = f"{src.codec or 'unknown'}_to_{codec}"

        key = (width, height, codec)
        if key in seen:
            continue
        seen.add(key)

        if mode == "encode":
            maxrate = spec.maxrate_bps or (
                _table_maxrate(height, codec) if scaled
                else _source_maxrate(height, codec, cap, nvenc_caps_mbps)
            )
        else:
            maxrate = None
        planned.append(RungPlan(
            id="", name=spec.name, codec=codec, mode=mode, encoder=encoder,
            width=width, height=height, scaled=scaled, box_width=box_w,
            box_height=box_h, maxrate_bps=maxrate,
            tonemap=mode == "encode" and src.hdr and codec == "h264",
            ten_bit=mode == "encode" and codec == "hevc" and (src.hdr or src.bit_depth > 8),
            reason=reason,
        ))

    # Largest first; at equal size HEVC before H.264 (the better codec is
    # the top rung). Stable for equal keys, so ladder order breaks ties.
    planned.sort(key=lambda r: (-(r.width * r.height), r.codec != "hevc"))
    rungs = [replace(r, id=f"v{i}") for i, r in enumerate(planned)]
    return Plan(source=src, rungs=rungs, segment_seconds=segment_seconds, cap=cap)
