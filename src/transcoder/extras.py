"""The extras mode: the trailers, featurettes and other bonus material
of a title.

The catalog keeps an extra apart from its title, as a row of its own
keyed by its extraId, and packages it on its own, never inside the
title's package. Its chain is short, with no analyzer and no trickplay:

    <prefix>catalog.extra.queued       catalog -> transcoder
    <prefix>catalog.extra.transcoded   transcoder -> packager

This consumer runs on a thread of its own, in a consumer group of its own
(`transcoder-extras`): the item loop takes one message at a time, so a
feature's encode, hours on a CPU, would otherwise hold every trailer up
behind it. The price is up to two encodes at once per pod, one item and
one extra (NVENC sessions, or CPU cores).

Per message, as the item loop does it (worker.run_worker):
  1. The extraId from the envelope, a lower-case UUID. Malformed -> warn,
     commit, skip.
  2. `GET /api/analyze/extras/{id}`. Unknown or removed (404) -> log,
     commit, skip: no step write, no event.
  3. The guard on the extra's state. Past its transcode (`transcoded`,
     `packaging`, `ready`), nothing runs: a redelivered or duplicate
     trigger passes the chain on (catalog.extra.transcoded again, so a
     packager that missed it recovers), and a retry is only acked (the
     run that finished the step passed the chain on itself). `missing`
     (the scanner found the file gone) has nothing to encode; the state
     stays the scanner's to change.
  4. The item encode path (`worker._process_one`), with the extras'
     settings: plan with EXTRA_LADDER, write the handoff into
     `_inbox/extra-<extraId>/` (the unchanged renditions.json contract),
     report the step to `PUT /api/analyze/extras/{id}/steps/transcode`.
     No VIDEO_TYPES gate: an extra is a video by the catalog's word.
  5. On done / not_applicable: produce catalog.extra.transcoded, flush,
     then commit. On failed: commit, no event; the catalog's retry sends
     the trigger again.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import structlog

from .ffmpeg import EncodeSettings
from .kafka import (
    build_consumer,
    build_extra_event,
    build_producer,
    is_retry,
    parse_envelope,
    parse_extra_id,
    produce_event,
)
from .katalog import ClaimedExtra, KatalogClient
from .worker import EVENT_SOURCE, NEXT_STEP, POLL_TIMEOUT_SECONDS, _process_one

log = structlog.get_logger(__name__)

# An extra's states past its transcode: the handoff is in its inbox, the
# packager has taken it, or it plays. The catalog moves the state on from
# `transcoded` when the packager starts, so all three are finished here —
# and a run now would clear the inbox under the packager's feet.
FINISHED_STATES = frozenset({"transcoded", "packaging", "ready"})

# The scanner's state for an extra whose file is gone. The catalog queues
# the extra again if the file comes back.
MISSING_STATE = "missing"

# The transcoded event's status: the package step waits for its worker.
QUEUED = "queued"


def extra_inbox_dir(packages_root: Path, extra_id: str) -> Path:
    """An extra's handoff directory, beside the items' `_inbox/<itemId>/`:
    `_inbox/extra-<extraId>/`. The prefix keeps an extra's handoff apart
    from every item's and tells the two apart at a glance."""
    return packages_root / "_inbox" / f"extra-{extra_id}"


class ExtraSteps:
    """An extra's transcode step, written the way `_process_one` writes an
    item's (worker.StepWriter): `PUT /api/analyze/extras/{id}/steps/transcode`."""

    def __init__(self, client: KatalogClient) -> None:
        self._client = client

    def upsert_step(
        self, extra_id: str, status: str, /, *, error: str | None = None,
        details: str | None = None,
    ) -> None:
        self._client.upsert_extra_step(extra_id, status, error=error, details=details)


def run_extras_worker(
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
    """Blocking consume loop for the extras, the twin of the item loop:
    one message at a time, the offset committed only once the extra is
    handled (its transcoded event produced and flushed), and committed on
    a failure too, so that one bad extra never loops. Exits when `stop`
    is set."""
    consumer = build_consumer(kafka_brokers, kafka_group_id, consume_topic, security_protocol)
    producer = build_producer(kafka_brokers, security_protocol)
    try:
        while not stop.is_set():
            msg = consumer.poll(POLL_TIMEOUT_SECONDS)
            if msg is None:
                continue
            if msg.error():
                log.warning("transcoder.extra.consume.error", error=str(msg.error()))
                continue

            raw = msg.value()
            extra_id = parse_extra_id(raw)
            if extra_id is None:
                log.warning(
                    "transcoder.extra.malformed",
                    partition=msg.partition(),
                    offset=msg.offset(),
                )
                consumer.commit(message=msg)
                continue

            try:
                _handle_extra(
                    extra_id=extra_id,
                    envelope=parse_envelope(raw),
                    client=client,
                    producer=producer,
                    produce_topic=produce_topic,
                    packages_root=packages_root,
                    settings=settings,
                )
            except Exception as e:
                # A catalog that would not answer, or a bug in this loop:
                # the encode path reports its own errors on the step.
                log.exception(
                    "transcoder.extra.process_unexpected",
                    extra_id=extra_id,
                    error=str(e)[:300],
                )
                try:
                    client.upsert_extra_step(extra_id, "failed", error=f"worker bug: {e}"[:500])
                except Exception:
                    log.exception("transcoder.extra.fail_report_failed", extra_id=extra_id)

            consumer.commit(message=msg)
    finally:
        try:
            producer.flush(5)
        except Exception:
            log.warning("transcoder.extra.producer.flush_failed_on_shutdown")
        consumer.close()
        log.info("transcoder.extra.consumer.closed")


def _handle_extra(
    extra_id: str,
    envelope: dict[str, Any],
    client: KatalogClient,
    producer: object,
    produce_topic: str,
    packages_root: Path,
    settings: EncodeSettings,
) -> None:
    """Resolve, guard and encode one extra. Produces catalog.extra.transcoded
    on a terminal success, and for a redelivered trigger of an extra past
    its transcode; nothing for an extra that is gone or missing, for a
    failure, or for a retry of an extra past its transcode. The caller
    commits."""
    retry = is_retry(envelope)
    extra = client.get_extra(extra_id)
    if extra is None:
        log.info("transcoder.extra.unresolved", extra_id=extra_id, retry=retry)
        return
    if extra.removed:
        log.info("transcoder.extra.removed_skip", extra_id=extra_id)
        return
    if extra.state == MISSING_STATE:
        log.info("transcoder.extra.missing_skip", extra_id=extra_id, path=extra.path)
        return
    if extra.state in FINISHED_STATES:
        if retry:
            # The catalog took a slow run for dead and sent the trigger
            # again, and the run finished since: it passed the chain on.
            log.info("transcoder.extra.retry.already_finished", extra_id=extra_id,
                     state=extra.state)
            return
        # A redelivery (a crash between the step write and the commit) or
        # a duplicate: no second encode, but the packager may never have
        # seen the transcoded event, so it goes again.
        log.info("transcoder.extra.already_done", extra_id=extra_id, state=extra.state)
        _emit_transcoded(producer, produce_topic, extra, envelope)
        return

    inbox = extra_inbox_dir(packages_root, extra.id)
    if _process_one(extra, ExtraSteps(client), inbox, settings):
        _emit_transcoded(producer, produce_topic, extra, envelope)


def _emit_transcoded(
    producer: object,
    produce_topic: str,
    extra: ClaimedExtra,
    envelope: dict[str, Any],
) -> None:
    """Produce catalog.extra.transcoded for `extra`, keyed by its id, and
    flush. The record names the parent and the kind; the trigger's are
    the fallback."""
    parent_id = extra.parent_id or _text(envelope.get("parentId"))
    value = build_extra_event(
        extra.id,
        parent_id=parent_id,
        kind=extra.kind or _text(envelope.get("kind")),
        step=NEXT_STEP,
        status=QUEUED,
        source=EVENT_SOURCE,
    )
    produce_event(producer, produce_topic, extra.id, value)  # type: ignore[arg-type]
    log.info(
        "transcoder.extra.event.produced",
        extra_id=extra.id,
        parent_id=parent_id,
        topic=produce_topic,
        step=NEXT_STEP,
    )


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
