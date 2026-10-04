"""Kafka event-consumer worker loop.

Same lifecycle contract as packager.worker / analyzer.run_worker:
  * Block on a poll(); exit cleanly on SIGTERM / SIGINT via the stop event.
  * Each message: parse -> resolve item -> (idempotency guard) -> encode
    -> mark transcode step terminal -> produce the next event -> commit.

Per-message work is fully serial inside one worker — a single hevc_nvenc
encode saturates the 3090 we schedule onto (and a CPU encode saturates
the pod's cores), and we commit the consumed offset only after the
encode + step-write + produce all succeed. To scale, add Deployment
replicas (one GPU each) in the same consumer group; do NOT process more
than one message concurrently. All rungs of one item come out of ONE
ffmpeg run (decode once, split per rung).

Crash-safety: the offset is committed ONLY after the transcode step is
written AND the `stube.catalog.item.transcoded` event is produced +
flushed. A crash mid-encode therefore reprocesses the message; that is
safe because the katalog (item_id, step) unique index plus the pre-work
`get_steps` guard make the DB writes idempotent.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import structlog

from .decision import plan_renditions
from .ffmpeg import (
    EncodeSettings,
    TranscodeError,
    build_encode_command,
    ffprobe,
    pick_profile,
    run_encode,
)
from .kafka import (
    build_consumer,
    build_event,
    build_producer,
    is_retry,
    parse_envelope,
    parse_item_id,
    produce_event,
)
from .katalog import ClaimedItem, KatalogClient
from .renditions import build_contract, write_contract

log = structlog.get_logger(__name__)

# The step this worker owns, and the step name we stamp on the produced
# event so downstream + the Go hub read the same string.
OWNING_STEP = "transcode"
NEXT_STEP = "package"
EVENT_SOURCE = "transcoder"

# Item types the transcoder handles. Audio / non-video items have no
# video stream to re-encode; we skip them (log + commit, no event).
VIDEO_TYPES = {"movie", "episode"}

# The transcode step's statuses that need no run: `done` (encoded),
# `not_applicable` (the source needs no encode; the packager reads it as
# it is) and `skipped` (final by the catalog's word). The catalog
# retries none of them.
FINISHED_STATUSES = frozenset({"done", "not_applicable", "skipped"})

# Consumer poll timeout — how long poll() blocks before returning None so
# the loop can re-check the stop event.
POLL_TIMEOUT_SECONDS = 1.0


def _inbox_dir(packages_root: Path, item_id: str) -> Path:
    """Per-item handoff directory the packager reads from. Flat layout
    under `_inbox/` (no category sharding) — the in-flight set is small
    (<= replicas) so directory size never grows."""
    return packages_root / "_inbox" / item_id


def _clear_inbox_files(inbox: Path, item_id: str) -> None:
    """Remove every file in the item's inbox dir (stale handoff from a
    prior run — e.g. a prepared.mkv from before the source was replaced,
    or a v2.mkv from a longer ladder). Best-effort."""
    if not inbox.exists():
        return
    for child in inbox.iterdir():
        try:
            if child.is_file():
                child.unlink()
        except OSError as e:
            log.warning(
                "transcoder.inbox.cleanup_failed",
                item_id=item_id,
                file=str(child),
                error=str(e),
            )


def _process_one(
    item: ClaimedItem,
    client: KatalogClient,
    packages_root: Path,
    settings: EncodeSettings,
) -> bool:
    """Run the transcode decision for one item. Returns True when the
    chain should advance (a terminal `done` / `not_applicable`), False on
    a `failed` outcome (the caller then commits but emits no event).

    Heartbeats the transcode step at start (in_progress) and end (the
    appropriate terminal status: done | not_applicable | failed). The
    step rows are the state the Activity monitor reads — kept unchanged
    from the claim-poller version."""
    log.info(
        "transcoder.item.start",
        item_id=item.id,
        title=item.title,
        type=item.type,
        path=item.path,
    )

    if not os.path.exists(item.path):
        # Source file vanished between scan and encode. Don't keep
        # retrying — flag it so an operator can re-scan the library.
        # The chain stops here (no next event).
        msg = f"source file missing: {item.path}"
        log.warning("transcoder.item.missing_file", item_id=item.id, path=item.path)
        client.upsert_step(item.id, "failed", error=msg)
        return False

    client.upsert_step(item.id, "in_progress")

    t0 = time.monotonic()
    try:
        probe = ffprobe(Path(item.path))
    except TranscodeError as e:
        log.exception("transcoder.probe.failed", item_id=item.id, error=str(e)[:300])
        client.upsert_step(item.id, "failed", error=f"ffprobe: {e}"[:500])
        return False

    plan = plan_renditions(
        probe,
        list(settings.ladder),
        settings.encoders,
        segment_seconds=settings.segment_seconds,
        nvenc_caps_mbps=(settings.maxrate_1080p_mbps, settings.maxrate_2160p_mbps),
    )
    src = plan.source
    inbox = _inbox_dir(packages_root, item.id)
    if plan.all_copy:
        # Nothing to encode (an HEVC source on the default ladder, or a
        # browser-friendly H.264 one without a GPU). The packager will read
        # the original `item.path` directly. We MUST NOT leave a stale
        # handoff from a prior run lying around; clean the _inbox dir.
        _clear_inbox_files(inbox, item.id)
        details = (
            f"skip codec={src.codec} "
            f"res={src.width}x{src.height} "
            f"reason={plan.rungs[0].reason}"
        )
        client.upsert_step(item.id, "not_applicable", details=details)
        log.info(
            "transcoder.item.skipped",
            item_id=item.id,
            title=item.title,
            codec=src.codec,
            reason=plan.rungs[0].reason,
            seconds=round(time.monotonic() - t0, 2),
        )
        # not_applicable is still a terminal success — advance the chain
        # so the packager picks up the passthrough source.
        return True

    # Encode path: one ffmpeg run writes every encoded rung.
    _clear_inbox_files(inbox, item.id)
    try:
        args, outputs = build_encode_command(
            Path(item.path),
            inbox,
            plan,
            settings,
            probe_subtitles=probe.get("subtitles") or [],
            video_index=probe.get("video_index"),
        )
        results, _elapsed = run_encode(args, outputs, log_label=item.id)
        write_contract(
            inbox,
            build_contract(
                item.id, plan, results, settings.encoders,
                duration_ms=int(probe.get("duration_ms") or 0),
                source_start_time=float(probe.get("start_time") or 0.0),
            ),
        )
    except (TranscodeError, OSError) as e:
        # Drop the inbox dir entirely so the packager doesn't get half a
        # handoff. run_encode already nukes the .partials, but finished
        # rungs and the directory itself might still exist.
        try:
            _clear_inbox_files(inbox, item.id)
            if inbox.exists():
                inbox.rmdir()
        except OSError:
            # Cleanup is best-effort — the next run overwrites anyway.
            pass
        log.exception("transcoder.encode.failed", item_id=item.id, error=str(e)[:300])
        client.upsert_step(item.id, "failed", error=str(e)[:500])
        return False

    seconds = round(time.monotonic() - t0, 2)
    top = plan.rungs[0]
    if top.encoder == "hevc_nvenc":
        label = pick_profile(
            src.width, src.height, settings.maxrate_1080p_mbps, settings.maxrate_2160p_mbps,
        ).label
    else:
        label = f"{top.encoder.replace('lib', '')}-{top.height}p"
    ladder = ",".join(f"{r.id}:{r.encoder}:{r.width}x{r.height}" for r in plan.rungs)
    details = (
        f"profile={label} "
        f"src_codec={src.codec} "
        f"res={src.width}x{src.height} "
        + (f"maxrate={top.maxrate_bps / 1_000_000:g}Mbps " if top.maxrate_bps else "")
        + f"ladder={ladder} "
        f"kf={plan.keyframes}/{plan.segment_seconds}s "
        f"out_mb={round(sum(r.size_bytes for r in results) / 1_000_000, 1)} "
        f"dur_s={seconds}"
    )
    client.upsert_step(item.id, "done", details=details)
    log.info(
        "transcoder.item.done",
        item_id=item.id,
        title=item.title,
        profile=label,
        ladder=ladder,
        seconds=seconds,
    )
    return True


def run_worker(
    client: KatalogClient,
    packages_root: Path,
    kafka_brokers: str,
    kafka_group_id: str,
    consume_topic: str,
    produce_topic: str,
    security_protocol: str,
    settings: EncodeSettings,
    stop: threading.Event,
) -> None:
    """Blocking Kafka consume loop. Exits when `stop` is set (SIGTERM
    handler in main).

    Per-message flow (replaces the claim poll loop):
      1. Parse the JSON value -> itemId. Malformed -> warn + commit + skip.
      2. get_item(itemId). None/404/no-path -> log + commit + skip.
      3. Non-video type -> log + commit + skip (no event).
      4. Idempotency guard: if transcode is already finished (`done`,
         `not_applicable` or `skipped`), skip the encode but STILL
         produce the next event, then commit. A retry the catalog sent
         for a step that has finished since is only committed (and
         logged): the run that finished it passed the chain on.
      5. Run the encode body; keep every katalog step write.
      6. On success, produce stube.catalog.item.transcoded + flush, then
         commit. On failure, mark step failed then commit (no event) to
         avoid a poison loop.
    """
    consumer = build_consumer(
        kafka_brokers, kafka_group_id, consume_topic, security_protocol
    )
    producer = build_producer(kafka_brokers, security_protocol)
    try:
        while not stop.is_set():
            msg = consumer.poll(POLL_TIMEOUT_SECONDS)
            if msg is None:
                continue
            if msg.error():
                # Transient consumer error (rebalance, broker blip). Log
                # and keep polling — no offset to commit for an errored
                # message.
                log.warning("transcoder.consume.error", error=str(msg.error()))
                continue

            raw = msg.value()
            item_id = parse_item_id(raw)
            if item_id is None:
                log.warning(
                    "transcoder.msg.malformed",
                    partition=msg.partition(),
                    offset=msg.offset(),
                )
                consumer.commit(message=msg)
                continue

            envelope = parse_envelope(raw)
            upstream_type = envelope.get("type") or None

            try:
                _handle_item(
                    item_id=item_id,
                    upstream_type=upstream_type,
                    client=client,
                    producer=producer,
                    produce_topic=produce_topic,
                    packages_root=packages_root,
                    settings=settings,
                    retry=is_retry(envelope),
                )
            except Exception as e:
                # Anything that escapes _handle_item is a bug in this loop
                # (the encode path already attributes its own errors to
                # the step). Report a hard fail, then commit so we don't
                # poison-loop on the same message forever.
                log.exception(
                    "transcoder.process_unexpected",
                    item_id=item_id,
                    error=str(e)[:300],
                )
                try:
                    client.fail(item_id, f"worker bug: {e}"[:500])
                except Exception:
                    log.exception("transcoder.fail_report_failed", item_id=item_id)

            # Commit AFTER the item is fully processed (and the next event
            # produced, inside _handle_item). Even on failure we commit to
            # avoid a poison loop — the failed step row is the durable
            # record for operator triage.
            consumer.commit(message=msg)
    finally:
        # flush any buffered produce before we tear the consumer down.
        try:
            producer.flush(5)
        except Exception:
            log.warning("transcoder.producer.flush_failed_on_shutdown")
        consumer.close()
        log.info("transcoder.consumer.closed")


def _handle_item(
    item_id: str,
    upstream_type: str | None,
    client: KatalogClient,
    producer: object,
    produce_topic: str,
    packages_root: Path,
    settings: EncodeSettings,
    retry: bool = False,
) -> None:
    """Resolve + process a single itemId. Produces the next event on any
    terminal-success outcome (including the idempotent already-done and
    the not_applicable passthrough). Emits nothing on skip/failure, nor
    for a `retry` whose step has finished since the catalog sent it.

    The offset commit lives in the caller so this stays free to raise;
    the caller commits regardless to avoid poison loops."""
    item = client.get_item(item_id)
    if item is None:
        # Unknown item, or metadata-only row with no primary path.
        log.warning("transcoder.item.unresolved", item_id=item_id)
        return

    if item.type not in VIDEO_TYPES:
        # Audio / non-video item — nothing to encode. Skip without a
        # step write and without emitting the next event.
        log.info(
            "transcoder.item.non_video_skip",
            item_id=item_id,
            type=item.type,
        )
        return

    # Idempotency guard: if we already finished this item's transcode
    # step (a reprocessed message after a mid-flight crash, or a
    # duplicate event), don't probe or burn the GPU again — but still
    # push the chain forward so a stuck downstream recovers. A source
    # that needed no encode (not_applicable) is as finished as an encoded
    # one: running it again would only re-probe it and re-package it.
    status = client.get_steps(item_id).get(OWNING_STEP)
    if status in FINISHED_STATUSES and retry:
        # The catalog retried a failed or silent transcode, and it has
        # finished since (a run the reaper took for dead reported done
        # after all). That run passed the chain on: nothing to do here.
        log.info("transcoder.retry.already_finished", item_id=item_id, status=status)
        return
    if status in FINISHED_STATUSES:
        log.info(
            "transcoder.item.already_done",
            item_id=item_id,
            title=item.title,
            status=status,
        )
        _emit_transcoded(
            producer, produce_topic, item, upstream_type
        )
        return

    advanced = _process_one(item, client, packages_root, settings)
    if advanced:
        _emit_transcoded(producer, produce_topic, item, upstream_type)


def _emit_transcoded(
    producer: object,
    produce_topic: str,
    item: ClaimedItem,
    upstream_type: str | None,
) -> None:
    """Produce stube.catalog.item.transcoded for `item` and flush. Called
    only on a terminal-success outcome; the caller commits the consumed
    offset afterwards so a crash between step-write and produce reprocesses
    (safely, via the idempotency guard)."""
    value = build_event(
        item.id,
        step=NEXT_STEP,
        status="done",
        type_=item.type,
        event_type=upstream_type,
        source=EVENT_SOURCE,
    )
    produce_event(producer, produce_topic, item.id, value)  # type: ignore[arg-type]
    log.info(
        "transcoder.event.produced",
        item_id=item.id,
        topic=produce_topic,
        step=NEXT_STEP,
    )
