"""Kafka wiring shared in shape across the three pipeline workers
(analyzer -> transcoder -> packager).

The workers form an event chain over the katalog domain topics:

    stube.catalog.item.enriched
        -> analyzer   -> stube.catalog.item.analyzed
        -> transcoder -> stube.catalog.item.transcoded
        -> packager   -> (terminal)

An extra of a title (a trailer, a featurette) has a chain of its own,
keyed by extraId, on topics under the same tenant prefix:

    stube.catalog.extra.queued
        -> transcoder (extras.py) -> stube.catalog.extra.transcoded
        -> packager               -> (terminal)

This module owns the confluent-kafka Consumer/Producer construction and
the JSON envelope, so the per-worker loop only deals with domain work.

Contract (must match the analyzer, packager, and the Go hub exactly):
  * Consumer: enable.auto.commit=false, auto.offset.reset=earliest. The
    caller commits the offset ONLY after the item is fully processed AND
    (for analyzer/transcoder) the next event has been produced + flushed.
    A crash mid-work therefore reprocesses the message; reprocessing is
    safe because the katalog (item_id, step) unique index and the
    pre-work step-status guard make the DB writes idempotent. The work
    runs on the poll thread, so max.poll.interval.ms outlasts the
    longest run (MAX_POLL_INTERVAL_MS).
  * Producer: acks=all, key = itemId encoded utf-8 so all events for one
    item land on the same partition (per-item ordering). flush() before
    the caller commits the consumed offset.
  * Envelope (JSON value): consumers REQUIRE only itemId and tolerate any
    extra fields; producers emit the full shape below. The extras
    envelope carries extraId, parentId and kind instead, and no itemId:
    an item worker pointed at an extras topic skips the event.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from confluent_kafka import Consumer, Producer

log = structlog.get_logger(__name__)

# How long the worker may go between two polls before the broker takes
# its partitions away and gives them — and the item in hand, whose offset
# is not committed yet — to another member, which would run the same
# encode a second time into the same inbox. The work runs on the poll
# thread, one item at a time, as in the analyzer, and an encode takes
# minutes on a GPU but hours on a CPU: past the catalog's 6 h transcode
# timeout when it must (the reaper then retries the step, and the retry
# is acked once this run reports done). So the worker holds its partition
# for as long as librdkafka allows (24 h); the catalog's reaper, not
# Kafka, decides when a silent run is dead. The price: a rebalance (a
# replica joining or leaving) waits until every busy member has finished
# its item.
MAX_POLL_INTERVAL_MS = 24 * 60 * 60 * 1000

# An extra's id as the catalog and the library name it: a lower-case
# RFC 4122 UUID. It names a directory (`_inbox/extra-<id>/`) and a URL
# path, so an event with anything else is malformed.
_EXTRA_ID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def _rfc3339_now() -> str:
    """UTC, RFC3339 with a trailing Z (matches the Go hub's time.Format)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _security_conf(security_protocol: str) -> dict[str, str]:
    """Kafka security settings. When KAFKA_CERT_DIR points at a mounted
    mTLS secret (user.crt/user.key + the CLUSTER CA's ca.crt — the shared
    Strimzi profile), it wins over `security_protocol`: a mounted cert dir
    IS the operator's way of saying "this broker speaks mTLS"."""
    cert_dir = os.environ.get("KAFKA_CERT_DIR", "").strip()
    if cert_dir and os.path.isdir(cert_dir):
        return {
            "security.protocol": "SSL",
            "ssl.ca.location": os.path.join(cert_dir, "ca.crt"),
            "ssl.certificate.location": os.path.join(cert_dir, "user.crt"),
            "ssl.key.location": os.path.join(cert_dir, "user.key"),
        }
    return {"security.protocol": security_protocol}


def build_consumer(
    brokers: str,
    group_id: str,
    topic: str,
    security_protocol: str = "PLAINTEXT",
) -> Consumer:
    """Construct a subscribed Consumer. Manual commit; earliest so a
    fresh consumer group replays the backlog rather than skipping it;
    the partition held through the longest encode (MAX_POLL_INTERVAL_MS)."""
    consumer = Consumer(
        {
            "bootstrap.servers": brokers,
            "group.id": group_id,
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
            "max.poll.interval.ms": MAX_POLL_INTERVAL_MS,
            **_security_conf(security_protocol),
        }
    )
    consumer.subscribe([topic])
    log.info(
        "kafka.consumer.subscribed",
        brokers=brokers,
        group_id=group_id,
        topic=topic,
        security_protocol=security_protocol,
    )
    return consumer


def build_producer(
    brokers: str,
    security_protocol: str = "PLAINTEXT",
) -> Producer:
    """Construct a Producer with acks=all (a produce isn't durable until
    the full ISR acknowledges — we only commit the consumed offset after
    the produced event is safe)."""
    producer = Producer(
        {
            "bootstrap.servers": brokers,
            "acks": "all",
            **_security_conf(security_protocol),
        }
    )
    log.info(
        "kafka.producer.ready",
        brokers=brokers,
        security_protocol=security_protocol,
    )
    return producer


def parse_item_id(raw_value: bytes | str | None) -> str | None:
    """Pull itemId out of an event envelope. Returns None (caller
    logs + commits + skips) when the payload is missing, not JSON, or
    carries no itemId. Extra fields are ignored per the contract."""
    if raw_value is None:
        return None
    try:
        payload = json.loads(raw_value)
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    item_id = payload.get("itemId")
    if not item_id:
        return None
    return str(item_id)


def parse_envelope(raw_value: bytes | str | None) -> dict[str, Any]:
    """Return the decoded envelope dict (or {} when unparseable). Used to
    carry `type` through to the produced event when the upstream event
    already knew it."""
    if raw_value is None:
        return {}
    try:
        payload = json.loads(raw_value)
    except (ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def parse_extra_id(raw_value: bytes | str | None) -> str | None:
    """Pull extraId out of an extras trigger (`catalog.extra.queued`).
    None (the caller logs, commits and skips) when the payload is
    missing, not JSON, or carries no extraId, or one that is not a
    lower-case UUID."""
    extra_id = parse_envelope(raw_value).get("extraId")
    if not isinstance(extra_id, str) or not _EXTRA_ID.match(extra_id):
        return None
    return extra_id


def is_retry(envelope: dict[str, Any]) -> bool:
    """True for an event the catalog sent again to retry a failed or
    silent step (status "retry", source "retry"). Its step may have
    finished since it was sent — a run the catalog's reaper took for dead
    that reported done after all — and then there is nothing to do."""
    return envelope.get("status") == "retry" or envelope.get("source") == "retry"


def build_event(
    item_id: str,
    *,
    step: str,
    status: str = "done",
    type_: str | None = None,
    source: str,
    event_type: str | None = None,
) -> bytes:
    """Serialise the next-stage event envelope. `eventId` is a fresh
    uuid4 hex; `occurredAt` is now. `event_type` maps to the envelope's
    `type` field (carried through from the upstream event when known)."""
    envelope: dict[str, Any] = {
        "eventId": uuid.uuid4().hex,
        "itemId": item_id,
        "type": event_type or type_ or "",
        "step": step,
        "status": status,
        "occurredAt": _rfc3339_now(),
        "source": source,
    }
    return json.dumps(envelope).encode("utf-8")


def build_extra_event(
    extra_id: str,
    *,
    parent_id: str | None,
    kind: str | None,
    step: str,
    status: str,
    source: str,
) -> bytes:
    """Serialise an extras envelope (`catalog.extra.*`): the item
    envelope's shape with extraId, parentId (the movie or series the
    extra belongs to) and kind (trailer, featurette, ...) in place of
    itemId, and type "extra". Never an itemId: every item worker requires
    one, so an extras event that reaches an item topic is skipped, never
    run as an item."""
    envelope: dict[str, Any] = {
        "eventId": uuid.uuid4().hex,
        "extraId": extra_id,
        "parentId": parent_id or None,
        "type": "extra",
        "kind": kind or None,
        "step": step,
        "status": status,
        "occurredAt": _rfc3339_now(),
        "source": source,
    }
    return json.dumps(envelope).encode("utf-8")


def produce_event(
    producer: Producer,
    topic: str,
    item_id: str,
    value: bytes,
) -> None:
    """Produce keyed by itemId (utf-8) so per-item events stay ordered
    on one partition, then flush so the message is durable before the
    caller commits the consumed offset. An extra's events are keyed by
    its extraId the same way."""
    producer.produce(topic, key=item_id.encode("utf-8"), value=value)
    producer.flush()
