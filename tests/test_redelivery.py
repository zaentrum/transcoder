"""The consumer loop against a fake broker and a fake catalog API: which
`analyzed` events run the transcode, which only pass the chain on
because the item's transcode step has finished already, and which a
worker only acks (a retry of a transcode that has finished since).

The real KatalogClient talks to the fake catalog through an httpx mock
transport, so the guard reads the step statuses exactly as it does in
production; the encode itself (`_process_one`) is replaced by a recorder
— the real encodes are in test_encode_real.py.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import httpx
import pytest
from structlog.testing import capture_logs

from transcoder import kafka, worker
from transcoder.ffmpeg import EncodeSettings
from transcoder.kafka import is_retry
from transcoder.katalog import KatalogClient

ITEM = "7a1c0de0-0000-4000-8000-000000000001"
BASE = "http://catalog.test"


class Message:
    """A consumed record, as confluent-kafka hands it to the loop."""

    def __init__(self, value: dict, offset: int) -> None:
        self._value = json.dumps(value).encode()
        self._offset = offset

    def value(self) -> bytes:
        return self._value

    def error(self) -> None:
        return None

    def partition(self) -> int:
        return 0

    def offset(self) -> int:
        return self._offset


class Broker:
    """One partition for the consumer and the topic the producer writes.
    Sets `stop` once every message has been polled, which ends the loop."""

    def __init__(self, events: list[dict], stop: threading.Event) -> None:
        self.pending = [Message(e, i) for i, e in enumerate(events)]
        self.stop = stop
        self.committed: list[int] = []
        self.produced: list[tuple[str, str, dict]] = []

    def poll(self, _timeout: float) -> Message | None:
        if not self.pending:
            self.stop.set()
            return None
        return self.pending.pop(0)

    def commit(self, message: Message) -> None:
        self.committed.append(message.offset())

    def close(self) -> None:
        pass

    def produce(self, topic: str, key: bytes, value: bytes) -> None:
        self.produced.append((topic, key.decode(), json.loads(value)))

    def flush(self, *_args: object) -> int:
        return 0


class Catalog:
    """The catalog's worker protocol: item detail, step statuses, writes."""

    def __init__(self, steps: dict[str, str]) -> None:
        self.steps = steps
        self.writes: list[tuple[str, str, object]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/token":
            return httpx.Response(200, json={"access_token": "t", "expires_in": 300})
        if request.method == "GET" and path == f"/api/analyze/items/{ITEM}":
            return httpx.Response(200, json={
                "id": ITEM, "type": "movie", "title": "Clip", "year": 2020,
                "durationMs": 60_000, "path": "/media/clip.mkv",
            })
        if request.method == "GET" and path == f"/api/analyze/items/{ITEM}/steps":
            return httpx.Response(200, json={"itemId": ITEM, "steps": self.steps})
        self.writes.append((request.method, path, json.loads(request.content or b"null")))
        return httpx.Response(200, json={})


def event(**fields: str) -> dict:
    """An `analyzed` envelope as the analyzer produces it."""
    return {"eventId": "e1", "itemId": ITEM, "type": "movie", "step": "transcode",
            "status": "done", "occurredAt": "2026-10-04T00:00:00Z", "source": "analyzer",
            **fields}


def retry() -> dict:
    """The same trigger as the catalog's retry sends it again."""
    return event(status="retry", source="retry")


def run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, events: list[dict],
        steps: dict[str, str]) -> tuple[Broker, Catalog, list[str]]:
    stop = threading.Event()
    broker = Broker(events, stop)
    catalog = Catalog(steps)
    encodes: list[str] = []

    def encode(item, _client, _root, _settings) -> bool:
        encodes.append(item.id)
        return True

    monkeypatch.setattr(worker, "build_consumer", lambda *a, **k: broker)
    monkeypatch.setattr(worker, "build_producer", lambda *a, **k: broker)
    monkeypatch.setattr(worker, "_process_one", encode)
    client = KatalogClient(BASE, f"{BASE}/token", "worker", "not-a-secret")
    client._http = httpx.Client(transport=httpx.MockTransport(catalog))
    worker.run_worker(client, tmp_path, "broker.test:9092", "transcoder-workers",
                      "stube.catalog.item.analyzed", "stube.catalog.item.transcoded",
                      "PLAINTEXT", EncodeSettings(), stop)
    return broker, catalog, encodes


@pytest.mark.parametrize("status", ["done", "not_applicable", "skipped"])
def test_finished_transcode_passes_the_chain_on_without_a_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: str,
) -> None:
    # A redelivered analyzed event (a crash between the step write and the
    # commit) for an item whose source needed no encode must not probe and
    # re-package it: no run, no step write, just the transcoded event.
    broker, catalog, encodes = run(monkeypatch, tmp_path, [event()], {"transcode": status})
    assert encodes == []
    assert catalog.writes == []
    assert [(t, k, e["step"], e["source"]) for t, k, e in broker.produced] == [
        ("stube.catalog.item.transcoded", ITEM, "package", "transcoder")]
    assert broker.committed == [0]


@pytest.mark.parametrize("steps", [{}, {"transcode": "pending"}, {"transcode": "in_progress"},
                                   {"transcode": "failed"}])
def test_unfinished_transcode_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, steps: dict[str, str],
) -> None:
    broker, _catalog, encodes = run(monkeypatch, tmp_path, [event()], steps)
    assert encodes == [ITEM]
    assert len(broker.produced) == 1
    assert broker.committed == [0]


@pytest.mark.parametrize("status", ["done", "not_applicable", "skipped"])
def test_retry_of_a_finished_transcode_is_only_acked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: str,
) -> None:
    # The reaper took a long run for dead and the catalog sent the trigger
    # again; the run reported its end before the retry was consumed. The
    # retry runs nothing, writes nothing, sends nothing: one log line.
    with capture_logs() as logs:
        broker, catalog, encodes = run(monkeypatch, tmp_path, [retry()], {"transcode": status})
    assert encodes == []
    assert catalog.writes == []
    assert broker.produced == []
    assert broker.committed == [0]
    said = [e for e in logs if e.get("item_id") == ITEM]
    assert [(e["event"], e["status"]) for e in said] == [
        ("transcoder.retry.already_finished", status)]


@pytest.mark.parametrize("status", ["pending", "failed", "in_progress"])
def test_retry_of_an_unfinished_transcode_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: str,
) -> None:
    # What a retry is for: the claimed step waits (pending) for this run.
    broker, _catalog, encodes = run(monkeypatch, tmp_path, [retry()], {"transcode": status})
    assert encodes == [ITEM]
    assert [e["step"] for _t, _k, e in broker.produced] == ["package"]
    assert broker.committed == [0]


def test_consumer_keeps_its_partition_through_a_long_encode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # librdkafka's default (5 min) is shorter than a GPU encode of a film:
    # the broker would hand the uncommitted item to another member, which
    # encodes it again into the same inbox. The interval must outlast the
    # catalog's transcode timeout (6 h), up to librdkafka's ceiling (24 h).
    seen: dict = {}

    class Consumer:
        def __init__(self, conf: dict) -> None:
            seen.update(conf)

        def subscribe(self, topics: list[str]) -> None:
            seen["topics"] = topics

    monkeypatch.setattr(kafka, "Consumer", Consumer)
    kafka.build_consumer("broker.test:9092", "transcoder-workers", "analyzed")
    assert 6 * 3600 * 1000 < seen["max.poll.interval.ms"] <= 86_400_000
    assert seen["enable.auto.commit"] is False
    assert seen["topics"] == ["analyzed"]


def test_retry_marker() -> None:
    assert is_retry({"status": "retry", "source": "retry"})
    assert is_retry({"status": "retry"})
    assert not is_retry({"status": "done", "source": "analyzer"})
    # The admin's packaging action starts the transcode with its own marker.
    assert not is_retry({"status": "package", "source": "package"})
    assert not is_retry({})
