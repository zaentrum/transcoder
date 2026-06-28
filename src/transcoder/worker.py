"""Claim-poller worker loop.

Same lifecycle contract as packager.worker / analyzer.run_worker:
  * Block on stop event; exit cleanly on SIGTERM / SIGINT.
  * Each iteration: claim -> process -> mark transcode terminal -> repeat.
  * Idle sleep on empty queue; error sleep on claim failure.

Per-cycle work is fully serial inside one worker — a single hevc_nvenc
encode saturates the 3090 we schedule onto. To scale, add Deployment
replicas (one GPU each); do NOT raise the batch size above 1.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import structlog

from .decision import decide
from .ffmpeg import TranscodeError, encode_to_hevc_mkv, ffprobe, pick_profile
from .katalog import ClaimedItem, KatalogClient

log = structlog.get_logger(__name__)


def _inbox_dir(packages_root: Path, item_id: str) -> Path:
    """Per-item handoff directory the packager reads from. Flat layout
    under `_inbox/` (no category sharding) — the in-flight set is
    small (<= replicas x claim_batch) so directory size never grows."""
    return packages_root / "_inbox" / item_id


def _process_one(
    item: ClaimedItem,
    client: KatalogClient,
    packages_root: Path,
    nvenc_preset: str,
    nvenc_cq: int,
    maxrate_1080p_mbps: int,
    maxrate_2160p_mbps: int,
) -> None:
    """Run the transcode decision for one claimed item.

    Heartbeats the transcode step at start (in_progress) and end (the
    appropriate terminal status: done | not_applicable | failed). The
    Java claim endpoint already flipped it to in_progress before we
    got here; we re-upsert anyway so modifiedAt tracks our progress,
    not the dequeue time."""
    log.info(
        "transcoder.item.start",
        item_id=item.id,
        title=item.title,
        type=item.type,
        path=item.path,
    )

    if not os.path.exists(item.path):
        # Source file vanished between scan and claim. Don't keep
        # retrying — flag it so an operator can re-scan the library.
        # We DON'T push package=pending; the chain stops here.
        msg = f"source file missing: {item.path}"
        log.warning("transcoder.item.missing_file", item_id=item.id, path=item.path)
        client.upsert_step(item.id, "failed", error=msg)
        return

    client.upsert_step(item.id, "in_progress")

    t0 = time.monotonic()
    try:
        probe = ffprobe(Path(item.path))
    except TranscodeError as e:
        log.exception("transcoder.probe.failed", item_id=item.id, error=str(e)[:300])
        client.upsert_step(item.id, "failed", error=f"ffprobe: {e}"[:500])
        return

    decision = decide(probe)
    if decision.skip:
        # Already-HEVC source. Nothing to do — let the Java side
        # promote package=pending and the packager will read the
        # original `item.path` directly. We MUST NOT leave a stale
        # prepared.mkv from a prior non-skip run lying around for the
        # same item; clean the _inbox dir defensively.
        inbox = _inbox_dir(packages_root, item.id)
        if inbox.exists():
            for child in inbox.iterdir():
                try:
                    if child.is_file():
                        child.unlink()
                except OSError as e:
                    log.warning(
                        "transcoder.inbox.cleanup_skip_failed",
                        item_id=item.id,
                        file=str(child),
                        error=str(e),
                    )
        details = (
            f"skip codec={decision.source_codec} "
            f"res={decision.width}x{decision.height} "
            f"reason={decision.reason}"
        )
        client.upsert_step(item.id, "not_applicable", details=details)
        log.info(
            "transcoder.item.skipped",
            item_id=item.id,
            title=item.title,
            codec=decision.source_codec,
            seconds=round(time.monotonic() - t0, 2),
        )
        return

    # Encode path. Resolution -> profile -> ffmpeg.
    profile = pick_profile(
        decision.width,
        decision.height,
        maxrate_1080p_mbps=maxrate_1080p_mbps,
        maxrate_2160p_mbps=maxrate_2160p_mbps,
    )
    out_path = _inbox_dir(packages_root, item.id) / "prepared.mkv"

    try:
        result = encode_to_hevc_mkv(
            Path(item.path),
            out_path,
            profile=profile,
            nvenc_preset=nvenc_preset,
            nvenc_cq=nvenc_cq,
            log_label=item.id,
            probe_subtitles=probe.get("subtitles") or [],
        )
    except TranscodeError as e:
        # Drop the inbox dir entirely so the packager doesn't get half
        # a file. encode_to_hevc_mkv already nukes the .partial, but
        # the directory itself might still exist.
        try:
            if out_path.parent.exists():
                for child in out_path.parent.iterdir():
                    child.unlink()
                out_path.parent.rmdir()
        except OSError:
            # Cleanup is best-effort — the next run overwrites anyway.
            pass
        log.exception("transcoder.encode.failed", item_id=item.id, error=str(e)[:300])
        client.upsert_step(item.id, "failed", error=str(e)[:500])
        return

    seconds = round(time.monotonic() - t0, 2)
    details = (
        f"profile={result['profile_label']} "
        f"src_codec={decision.source_codec} "
        f"res={decision.width}x{decision.height} "
        f"maxrate={result['maxrate_mbps']}Mbps "
        f"out_mb={round(result['out_size_bytes'] / 1_000_000, 1)} "
        f"dur_s={seconds}"
    )
    client.upsert_step(item.id, "done", details=details)
    log.info(
        "transcoder.item.done",
        item_id=item.id,
        title=item.title,
        profile=result["profile_label"],
        seconds=seconds,
    )


def run_worker(
    client: KatalogClient,
    packages_root: Path,
    batch_size: int,
    idle_sleep: float,
    error_sleep: float,
    nvenc_preset: str,
    nvenc_cq: int,
    maxrate_1080p_mbps: int,
    maxrate_2160p_mbps: int,
    stop: threading.Event,
) -> None:
    """Blocking loop. Exits when `stop` is set (SIGTERM handler in main)."""
    while not stop.is_set():
        try:
            batch = client.claim(limit=batch_size)
        except Exception as e:
            log.exception("transcoder.claim_failed", error=str(e)[:300])
            stop.wait(error_sleep)
            continue

        if not batch:
            stop.wait(idle_sleep)
            continue

        for item in batch:
            if stop.is_set():
                # Mid-batch shutdown: hand the in-flight item back as
                # pending so the next pod can pick it up. Same trick
                # the packager uses; without it the row sits
                # in_progress until the janitor sweep rescues it.
                client.upsert_step(
                    item.id,
                    "pending",
                    error="worker shutdown before start",
                )
                break
            try:
                _process_one(
                    item,
                    client,
                    packages_root,
                    nvenc_preset=nvenc_preset,
                    nvenc_cq=nvenc_cq,
                    maxrate_1080p_mbps=maxrate_1080p_mbps,
                    maxrate_2160p_mbps=maxrate_2160p_mbps,
                )
            except Exception as e:
                # _process_one already attributed any error it owns to
                # the transcode step; anything that escapes is a bug
                # in this loop itself.
                log.exception(
                    "transcoder.process_unexpected",
                    item_id=item.id,
                    error=str(e)[:300],
                )
                try:
                    client.fail(item.id, f"worker bug: {e}"[:500])
                except Exception:
                    log.exception("transcoder.fail_report_failed", item_id=item.id)
