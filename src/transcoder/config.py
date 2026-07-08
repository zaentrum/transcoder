"""Runtime configuration. Everything from env vars; defaults are
sized for a single-replica GPU deployment doing one encode at a time
(a single 3090 saturates on a 1080p hevc_nvenc encode, two concurrent
encodes on the same card cut the per-item throughput by ~30 % with no
end-to-end win — so we consume one Kafka message at a time)."""

from __future__ import annotations

import os
from dataclasses import dataclass


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
    # _inbox root. Same PVC the packager reads from — the transcoder
    # only ever writes to `{packages_root}/_inbox/{itemId}/prepared.mkv`.
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

    @classmethod
    def from_env(cls) -> Config:
        return cls(
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
        )


def _require(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(f"required env var {key} is empty/unset")
    return val
