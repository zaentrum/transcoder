"""Entry point. One process runs:
  - the worker loop (thread)
  - a tiny FastAPI server for /healthz and /readyz, so kubelet probes
    work.

Same shape as packager.main and analyzer.main — intentionally — so
anyone reading all three can map them onto each other line-for-line.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
from pathlib import Path

import structlog
import uvicorn
from fastapi import FastAPI

from .config import Config
from .decision import parse_ladder
from .ffmpeg import EncodeSettings, detect_encoders
from .katalog import KatalogClient
from .worker import run_worker

# Pods run with random non-root UID in GID 0. Without this, mkdir under
# /var/lib/katalog/packages creates 0750 dirs the *other* packager pod
# (different UID, same GID 0) cannot write into. 0002 → group rwx.
os.umask(0o002)


def _configure_logging() -> None:
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=logging.INFO)
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )


def main() -> int:
    _configure_logging()
    log = structlog.get_logger("transcoder.main")
    cfg = Config.from_env()
    # Probe NVENC once: a box without a GPU (or with ENCODER=cpu) encodes
    # with libx265 / libx264 instead of never transcoding at all.
    encoders = detect_encoders(cfg.encoder)
    ladder = parse_ladder(cfg.ladder)
    settings = EncodeSettings(
        ladder=tuple(ladder),
        encoders=encoders,
        segment_seconds=cfg.segment_seconds,
        nvenc_preset=cfg.nvenc_preset,
        nvenc_cq=cfg.nvenc_cq,
        maxrate_1080p_mbps=cfg.maxrate_1080p_mbps,
        maxrate_2160p_mbps=cfg.maxrate_2160p_mbps,
        x264_preset=cfg.x264_preset,
        x264_crf=cfg.x264_crf,
        x265_preset=cfg.x265_preset,
        x265_crf=cfg.x265_crf,
    )
    log.info(
        "transcoder.start",
        katalog=cfg.katalog_api_url,
        kafka_brokers=cfg.kafka_brokers,
        kafka_group_id=cfg.kafka_group_id,
        consume_topic=cfg.consume_topic,
        produce_topic=cfg.produce_topic,
        backend=encoders.backend,
        hevc_encoder=encoders.hevc,
        h264_encoder=encoders.h264,
        ladder=[f"{r.name}:{r.codec}" for r in ladder],
        segment_seconds=cfg.segment_seconds,
        nvenc_preset=cfg.nvenc_preset,
        nvenc_cq=cfg.nvenc_cq,
        maxrate_1080p_mbps=cfg.maxrate_1080p_mbps,
        maxrate_2160p_mbps=cfg.maxrate_2160p_mbps,
    )

    client = KatalogClient(
        base_url=cfg.katalog_api_url,
        token_url=cfg.oidc_token_url,
        client_id=cfg.oidc_client_id,
        client_secret=cfg.oidc_client_secret,
    )

    stop = threading.Event()

    def _handle_sigterm(signum: int, _frame: object) -> None:
        log.info("transcoder.signal", signum=signum)
        stop.set()

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    # A SINGLE consumer thread (was the claim poll loop). One GPU encode
    # at a time per pod; scale with Deployment replicas in the same
    # consumer group, not with more threads.
    worker_thread = threading.Thread(
        target=run_worker,
        kwargs={
            "client": client,
            "packages_root": Path(cfg.packages_root),
            "kafka_brokers": cfg.kafka_brokers,
            "kafka_group_id": cfg.kafka_group_id,
            "consume_topic": cfg.consume_topic,
            "produce_topic": cfg.produce_topic,
            "security_protocol": cfg.security_protocol,
            "settings": settings,
            "stop": stop,
        },
        daemon=True,
        name="transcoder-worker",
    )
    worker_thread.start()

    app = FastAPI()

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    @app.get("/readyz")
    def readyz() -> dict:
        return {"ok": worker_thread.is_alive()}

    uvicorn.run(app, host="0.0.0.0", port=8080, log_config=None)
    stop.set()
    client.close()
    worker_thread.join(timeout=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
