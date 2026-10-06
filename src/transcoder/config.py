"""Runtime configuration. Everything from env vars; defaults are
sized for a single-replica GPU deployment doing one encode at a time
(a single 3090 saturates on a 1080p hevc_nvenc encode, two concurrent
encodes on the same card cut the per-item throughput by ~30 % with no
end-to-end win — so we consume one Kafka message at a time). The
extras (trailers and other bonus material, README "Extras") are the one
exception: a consumer of their own, so a feature's encode never holds a
two-minute trailer up behind it."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .decision import parse_ladder

# The extras' ladder when EXTRA_LADDER is unset or empty: two H.264
# rungs. An extra is served without an on-the-fly fallback, so keep every
# rung in a codec every device decodes. A rung the source already fits in
# is the source's own size, and a stream copy when the source is
# browser-friendly H.264. HEVC only is EXTRA_LADDER=source:hevc, for once
# the streaming side transcodes extras on the fly from their package.
DEFAULT_EXTRA_LADDER = "720p:h264,480p:h264"
DEFAULT_TOPIC_PREFIX = "stube."


@dataclass(frozen=True)
class Config:
    katalog_api_url: str
    oidc_token_url: str
    oidc_client_id: str
    oidc_client_secret: str
    # Kafka event-chain wiring. The transcoder consumes
    # stube.catalog.item.analyzed and produces stube.catalog.item.transcoded.
    # Broker list is comma-separated (e.g. "kafka:9092"). The bundled demo
    # broker is PLAINTEXT (no TLS); override security_protocol only for a
    # TLS/SASL cluster.
    kafka_brokers: str = "kafka:9092"
    kafka_group_id: str = "transcoder-workers"
    consume_topic: str = "stube.catalog.item.analyzed"
    produce_topic: str = "stube.catalog.item.transcoded"
    security_protocol: str = "PLAINTEXT"
    # The legacy layout's _inbox root, the tree the packager reads: a
    # worker record without `library` hands off into
    # `{packages_root}/_inbox/{itemId}/` (an extra: `_inbox/extra-<id>/`).
    # On the v2 library layout the record names the inbox
    # (`library.inboxDir`) and this is not used.
    packages_root: str = "/var/lib/katalog/packages"
    # NVENC quality / rate-control knobs, tuned for the packager:
    #   - preset p5: NVENC "slow" — best quality on the 3090 family
    #     without falling off the realtime curve. p6/p7 are higher
    #     quality but ~2x slower per frame.
    #   - cq 23: target constant quality; visually lossless at 1080p
    #     in our sample set, ~20-30 % smaller than a cq-28 baseline
    #     (we have storage; we'd rather not re-encode again to gain
    #     quality).
    #   - maxrate caps are resolution-aware so a single hevc_nvenc
    #     command line works for SD/HD/UHD. Anything >1920 wide gets
    #     the UHD band.
    nvenc_preset: str = "p5"
    nvenc_cq: int = 23
    maxrate_1080p_mbps: int = 8
    maxrate_2160p_mbps: int = 14
    # Rendition ladder, e.g. "source,720p,480p" (README "Ladder"). Empty
    # = ONE rendition per item, exactly as before — an install that never
    # sets it doesn't grow its storage. "source:hevc" = HEVC only, on a
    # host without NVENC too (README "HEVC only").
    ladder: str = ""
    # Encoder backend: auto (probe NVENC at startup, fall back to
    # libx265/libx264), nvenc (require it), cpu (never use it).
    encoder: str = "auto"
    # Forced keyframe interval. MUST equal the packager's segment
    # duration (it reads this value back from renditions.json).
    segment_seconds: int = 6
    # CPU encoders (the no-GPU path). x264 for H.264 rungs; x265 only for
    # source-size HEVC encodes of sources nothing plays as-is.
    x264_preset: str = "medium"
    x264_crf: int = 23
    x265_preset: str = "medium"
    x265_crf: int = 24
    # The extras mode. Its topics are the tenant's, named as katalog-manager
    # names them: <KAFKA_TOPIC_PREFIX>catalog.extra.queued in,
    # <KAFKA_TOPIC_PREFIX>catalog.extra.transcoded out.
    topic_prefix: str = DEFAULT_TOPIC_PREFIX
    extras_group_id: str = "transcoder-extras"
    extra_ladder: str = DEFAULT_EXTRA_LADDER

    @property
    def extras_consume_topic(self) -> str:
        return f"{self.topic_prefix}catalog.extra.queued"

    @property
    def extras_produce_topic(self) -> str:
        return f"{self.topic_prefix}catalog.extra.transcoded"

    @classmethod
    def from_env(cls) -> Config:
        cfg = cls(
            katalog_api_url=_require("KATALOG_API_URL"),
            oidc_token_url=_require("OIDC_TOKEN_URL"),
            oidc_client_id=_require("OIDC_CLIENT_ID"),
            oidc_client_secret=_require("OIDC_CLIENT_SECRET"),
            kafka_brokers=os.environ.get("KAFKA_BROKERS", "kafka:9092"),
            kafka_group_id=os.environ.get("KAFKA_GROUP_ID", "transcoder-workers"),
            consume_topic=os.environ.get("CONSUME_TOPIC", "stube.catalog.item.analyzed"),
            produce_topic=os.environ.get("PRODUCE_TOPIC", "stube.catalog.item.transcoded"),
            security_protocol=os.environ.get("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT"),
            packages_root=os.environ.get("PACKAGES_ROOT", "/var/lib/katalog/packages"),
            nvenc_preset=os.environ.get("NVENC_PRESET", "p5"),
            nvenc_cq=int(os.environ.get("NVENC_CQ", "23")),
            maxrate_1080p_mbps=int(os.environ.get("NVENC_MAXRATE_1080P_MBPS", "8")),
            maxrate_2160p_mbps=int(os.environ.get("NVENC_MAXRATE_2160P_MBPS", "14")),
            ladder=os.environ.get("LADDER", ""),
            encoder=os.environ.get("ENCODER", "auto"),
            segment_seconds=int(os.environ.get("SEGMENT_SECONDS", "6")),
            x264_preset=os.environ.get("X264_PRESET", "medium"),
            x264_crf=int(os.environ.get("X264_CRF", "23")),
            x265_preset=os.environ.get("X265_PRESET", "medium"),
            x265_crf=int(os.environ.get("X265_CRF", "24")),
            topic_prefix=normalize_topic_prefix(os.environ.get("KAFKA_TOPIC_PREFIX")),
            extras_group_id=os.environ.get("EXTRAS_GROUP_ID") or "transcoder-extras",
            extra_ladder=os.environ.get("EXTRA_LADDER", "").strip() or DEFAULT_EXTRA_LADDER,
        )
        # Fail at startup on a typo rather than silently falling back to
        # the single-rendition default.
        parse_ladder(cfg.ladder)
        parse_ladder(cfg.extra_ladder)
        if not 1 <= cfg.segment_seconds <= 30:
            raise RuntimeError(f"SEGMENT_SECONDS must be 1..30 (got {cfg.segment_seconds})")
        return cfg


def normalize_topic_prefix(raw: str | None) -> str:
    """KAFKA_TOPIC_PREFIX as katalog-manager reads it: blank is "stube.",
    and a missing trailing dot is added ("tenant" -> "tenant.")."""
    prefix = (raw or "").strip() or DEFAULT_TOPIC_PREFIX
    return prefix if prefix.endswith(".") else prefix + "."


def _require(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(f"required env var {key} is empty/unset")
    return val
