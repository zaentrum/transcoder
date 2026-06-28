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
    log.info(
        "transcoder.start",
        katalog=cfg.katalog_api_url,
        batch_size=cfg.claim_batch_size,
        idle_sleep=cfg.idle_sleep_seconds,
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

    worker_thread = threading.Thread(
        target=run_worker,
        kwargs={
            "client": client,
            "packages_root": Path(cfg.packages_root),
            "batch_size": cfg.claim_batch_size,
            "idle_sleep": cfg.idle_sleep_seconds,
            "error_sleep": cfg.error_sleep_seconds,
            "nvenc_preset": cfg.nvenc_preset,
            "nvenc_cq": cfg.nvenc_cq,
            "maxrate_1080p_mbps": cfg.maxrate_1080p_mbps,
            "maxrate_2160p_mbps": cfg.maxrate_2160p_mbps,
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
