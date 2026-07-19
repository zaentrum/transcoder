"""Kafka wiring shared in shape across the three pipeline workers
(analyzer -> transcoder -> packager).

The workers form an event chain over the katalog domain topics:

    stube.catalog.item.enriched
        -> analyzer   -> stube.catalog.item.analyzed
        -> transcoder -> stube.catalog.item.transcoded
        -> packager   -> (terminal)

This module owns the confluent-kafka Consumer/Producer construction and
the JSON envelope, so the per-worker loop only deals with domain work.

Contract (must match the analyzer, packager, and the Go hub exactly):
  * Consumer: enable.auto.commit=false, auto.offset.reset=earliest. The
    caller commits the offset ONLY after the item is fully processed AND
    (for analyzer/transcoder) the next event has been produced + flushed.
    A crash mid-work therefore reprocesses the message; reprocessing is
    safe because the katalog (item_id, step) unique index and the
    pre-work step-status guard make the DB writes idempotent.
  * Producer: acks=all, key = itemId encoded utf-8 so all events for one
    item land on the same partition (per-item ordering). flush() before
    the caller commits the consumed offset.
  * Envelope (JSON value): consumers REQUIRE only itemId and tolerate any
    extra fields; producers emit the full shape below.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from confluent_kafka import Consumer, Producer

log = structlog.get_logger(__name__)


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
    fresh consumer group replays the backlog rather than skipping it."""
    consumer = Consumer(
        {
            "bootstrap.servers": brokers,
            "group.id": group_id,
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
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


def produce_event(
    producer: Producer,
    topic: str,
    item_id: str,
    value: bytes,
) -> None:
    """Produce keyed by itemId (utf-8) so per-item events stay ordered
    on one partition, then flush so the message is durable before the
    caller commits the consumed offset."""
    producer.produce(topic, key=item_id.encode("utf-8"), value=value)
    producer.flush()
