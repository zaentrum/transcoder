"""The extras mode: trailers and other bonus material of a title, on a
consumer of their own (`catalog.extra.queued` -> `catalog.extra.transcoded`).

The loop runs against a fake broker and the real KatalogClient talking to
a fake catalog API through an httpx mock transport, so the guards read
the extra's record exactly as in production; the encode (`_process_one`)
is a recorder that reports through the step writer it is handed. The real
encodes of extras are in test_encode_real.py.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import httpx
import pytest
from structlog.testing import capture_logs

from transcoder import extras, worker
from transcoder.config import DEFAULT_EXTRA_LADDER
from transcoder.decision import parse_ladder
from transcoder.ffmpeg import EncodeSettings
from transcoder.kafka import build_extra_event, is_retry, parse_extra_id, parse_item_id
from transcoder.katalog import KatalogClient

EXTRA = "1b5c2a8e-0000-4000-8000-0000000000e1"
PARENT = "ea886f9b-0d06-4f0f-babb-d2a1162f9b01"
BASE = "http://catalog.test"
CONSUME = "stube.catalog.extra.queued"
PRODUCE = "stube.catalog.extra.transcoded"
SETTINGS = EncodeSettings(ladder=tuple(parse_ladder(DEFAULT_EXTRA_LADDER)))


def trigger(**fields: object) -> dict:
    """A `catalog.extra.queued` envelope as the catalog sends it."""
    return {"eventId": "9f2b", "extraId": EXTRA, "parentId": PARENT, "type": "extra",
            "kind": "trailer", "step": "transcode", "status": "queued",
            "occurredAt": "2026-10-06T08:00:00Z", "source": "api", **fields}


# ------------------------------------------------------------- envelope
def test_extra_id_is_read_from_the_trigger() -> None:
    assert parse_extra_id(json.dumps(trigger()).encode()) == EXTRA
    assert parse_extra_id(json.dumps(trigger())) == EXTRA


@pytest.mark.parametrize("raw", [
    None, b"", b"not json", b"[]", json.dumps({"itemId": PARENT}).encode(),
    json.dumps(trigger(extraId=None)).encode(),
    # It names a directory and a URL path: a lower-case UUID or nothing.
    json.dumps(trigger(extraId="../../etc")).encode(),
    json.dumps(trigger(extraId=EXTRA.upper())).encode(),
    json.dumps(trigger(extraId="{" + EXTRA + "}")).encode(),
    json.dumps(trigger(extraId=EXTRA + "/x")).encode(),
    json.dumps(trigger(extraId=7)).encode(),
])
def test_malformed_extras_triggers_have_no_extra_id(raw) -> None:
    assert parse_extra_id(raw) is None


def test_an_item_worker_skips_an_extras_event() -> None:
    # No itemId, on purpose: an item worker pointed at an extras topic by
    # mistake finds nothing to run.
    assert parse_item_id(json.dumps(trigger()).encode()) is None
    produced = build_extra_event(EXTRA, parent_id=PARENT, kind="trailer", step="package",
                                 status="queued", source="transcoder")
    assert parse_item_id(produced) is None


def test_transcoded_envelope_shape() -> None:
    event = json.loads(build_extra_event(EXTRA, parent_id=PARENT, kind="teaser",
                                         step="package", status="queued", source="transcoder"))
    assert list(event) == ["eventId", "extraId", "parentId", "type", "kind", "step", "status",
                           "occurredAt", "source"]
    assert {k: v for k, v in event.items() if k not in ("eventId", "occurredAt")} == {
        "extraId": EXTRA, "parentId": PARENT, "type": "extra", "kind": "teaser",
        "step": "package", "status": "queued", "source": "transcoder",
    }
    assert len(event["eventId"]) == 32
    assert event["occurredAt"].endswith("Z")


def test_extras_retry_marker() -> None:
    assert is_retry(trigger(status="retry", source="retry"))
    assert not is_retry(trigger())


# ---------------------------------------------------------------- inbox
def test_an_extra_has_an_inbox_of_its_own() -> None:
    root = Path("/var/lib/katalog/packages")
    assert extras.extra_inbox_dir(root, EXTRA) == root / "_inbox" / f"extra-{EXTRA}"
    # Never an item's, even were the ids ever to meet.
    assert extras.extra_inbox_dir(root, EXTRA) != worker._inbox_dir(root, EXTRA)


# ------------------------------------------------------------ the loop
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
        self.subscribed: tuple = ()
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
    """The catalog's extras worker protocol: the record, and step writes."""

    def __init__(self, state: str | None, **record: object) -> None:
        self.record = None if state is None else {
            "id": EXTRA, "type": "extra", "parentId": PARENT, "parentType": "movie",
            "parentTitle": "Big Buck Bunny", "kind": "trailer", "title": "Trailer",
            "language": "en", "seasonNumber": None,
            "path": "/var/lib/katalog/extras/big-buck-bunny/trailer.mov", "state": state,
            **record,
        }
        self.reads: list[str] = []
        self.writes: list[tuple[str, str, object]] = []
        self.fail_reads = False

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/token":
            return httpx.Response(200, json={"access_token": "t", "expires_in": 300})
        if request.method == "GET":
            self.reads.append(path)
            if self.fail_reads:
                return httpx.Response(503)
            if path == f"/api/analyze/extras/{EXTRA}" and self.record is not None:
                return httpx.Response(200, json=self.record)
            return httpx.Response(404)
        self.writes.append((request.method, path, json.loads(request.content or b"null")))
        return httpx.Response(200, json={})


def retry_trigger() -> dict:
    """The trigger as the catalog's retry sends it again."""
    return trigger(status="retry", source="retry")


def run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, events: list[dict], catalog: Catalog,
        *, outcome: bool = True) -> tuple[Broker, list[tuple]]:
    stop = threading.Event()
    broker = Broker(events, stop)
    encodes: list[tuple] = []

    def encode(item, steps, inbox, settings) -> bool:
        # Report the way _process_one does, through the writer handed in.
        encodes.append((item, inbox, settings))
        steps.upsert_step(item.id, "in_progress")
        steps.upsert_step(item.id, "done" if outcome else "failed",
                          **({"details": "profile=x264-720p"} if outcome else {"error": "boom"}))
        return outcome

    def consumer(brokers, group, topic, protocol):
        broker.subscribed = (brokers, group, topic, protocol)
        return broker

    monkeypatch.setattr(extras, "build_consumer", consumer)
    monkeypatch.setattr(extras, "build_producer", lambda *a, **k: broker)
    monkeypatch.setattr(extras, "_process_one", encode)
    client = KatalogClient(BASE, f"{BASE}/token", "worker", "not-a-secret")
    client._http = httpx.Client(transport=httpx.MockTransport(catalog))
    extras.run_extras_worker(client, tmp_path, "broker.test:9092", "transcoder-extras",
                             CONSUME, PRODUCE, "PLAINTEXT", SETTINGS, stop)
    return broker, encodes


STEP = f"/api/analyze/extras/{EXTRA}/steps/transcode"


def transcoded(broker: Broker) -> list[tuple[str, str, dict]]:
    """The produced events, without the fields that differ every time."""
    return [(t, k, {f: v for f, v in e.items() if f not in ("eventId", "occurredAt")})
            for t, k, e in broker.produced]


PASSED_ON = [(PRODUCE, EXTRA, {
    "extraId": EXTRA, "parentId": PARENT, "type": "extra", "kind": "trailer",
    "step": "package", "status": "queued", "source": "transcoder",
})]


def test_a_queued_extra_is_encoded_into_its_own_inbox(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    catalog = Catalog("queued")
    broker, encodes = run(monkeypatch, tmp_path, [trigger()], catalog)
    assert broker.subscribed == ("broker.test:9092", "transcoder-extras", CONSUME, "PLAINTEXT")
    [(item, inbox, settings)] = encodes
    # No VIDEO_TYPES gate: the extra runs as it is, with the extras' ladder.
    assert (item.id, item.type, item.path) == (EXTRA, "extra", catalog.record["path"])
    assert inbox == tmp_path / "_inbox" / f"extra-{EXTRA}"
    assert settings is SETTINGS
    # The step is the extra's, never an item's.
    assert catalog.reads == [f"/api/analyze/extras/{EXTRA}"]
    assert catalog.writes == [("PUT", STEP, {"status": "in_progress"}),
                              ("PUT", STEP, {"status": "done", "details": "profile=x264-720p"})]
    assert transcoded(broker) == PASSED_ON
    assert broker.committed == [0]


@pytest.mark.parametrize("state", ["queued", "pending", "transcoding", "failed"])
@pytest.mark.parametrize("make", [trigger, retry_trigger])
def test_an_extra_not_past_its_transcode_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, state: str, make,
) -> None:
    # A retry is what a retry is for; a redelivery of a run that crashed
    # (transcoding) runs again, as for an item.
    broker, encodes = run(monkeypatch, tmp_path, [make()], Catalog(state))
    assert len(encodes) == 1
    assert transcoded(broker) == PASSED_ON
    assert broker.committed == [0]


@pytest.mark.parametrize("state", ["transcoded", "packaging", "ready"])
def test_a_redelivered_trigger_past_the_transcode_passes_the_chain_on(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, state: str,
) -> None:
    # A crash between the step write and the commit, or a duplicate: no
    # second encode (packaging: the packager reads the inbox right now),
    # no step write, only the transcoded event again for a stuck extra.
    catalog = Catalog(state)
    broker, encodes = run(monkeypatch, tmp_path, [trigger()], catalog)
    assert encodes == []
    assert catalog.writes == []
    assert transcoded(broker) == PASSED_ON
    assert broker.committed == [0]


@pytest.mark.parametrize("state", ["transcoded", "packaging", "ready"])
def test_a_retry_past_the_transcode_is_only_acked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, state: str,
) -> None:
    # The catalog took a slow run for dead and sent the trigger again; the
    # run finished before the retry was consumed and passed the chain on.
    catalog = Catalog(state)
    with capture_logs() as logs:
        broker, encodes = run(monkeypatch, tmp_path, [retry_trigger()], catalog)
    assert encodes == []
    assert catalog.writes == []
    assert broker.produced == []
    assert broker.committed == [0]
    said = [e for e in logs if e.get("extra_id") == EXTRA]
    assert [(e["event"], e["state"]) for e in said] == [
        ("transcoder.extra.retry.already_finished", state)]


@pytest.mark.parametrize(("state", "record", "event"), [
    (None, {}, "transcoder.extra.unresolved"),  # 404: unknown, or removed
    ("queued", {"removedAt": "2026-10-05T08:00:00Z"}, "transcoder.extra.removed_skip"),
    ("missing", {}, "transcoder.extra.missing_skip"),
])
@pytest.mark.parametrize("make", [trigger, retry_trigger])
def test_an_extra_that_is_gone_is_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, state: str | None, record: dict,
    event: str, make,
) -> None:
    catalog = Catalog(state, **record)
    with capture_logs() as logs:
        broker, encodes = run(monkeypatch, tmp_path, [make()], catalog)
    assert encodes == []
    assert catalog.writes == []
    assert broker.produced == []
    assert broker.committed == [0]
    assert event in [e["event"] for e in logs if e.get("extra_id") == EXTRA]


def test_a_failed_encode_passes_nothing_on(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    catalog = Catalog("queued")
    broker, _ = run(monkeypatch, tmp_path, [trigger()], catalog, outcome=False)
    assert catalog.writes[-1] == ("PUT", STEP, {"status": "failed", "error": "boom"})
    assert broker.produced == []
    assert broker.committed == [0]


def test_a_catalog_that_does_not_answer_fails_the_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # As for an item: the step is reported failed (the catalog retries it)
    # and the offset committed, so one extra never blocks the next.
    catalog = Catalog("queued")
    catalog.fail_reads = True
    broker, encodes = run(monkeypatch, tmp_path, [trigger(), trigger()], catalog)
    assert encodes == []
    assert [(m, p, b["status"]) for m, p, b in catalog.writes] == [("PUT", STEP, "failed")] * 2
    assert catalog.writes[0][2]["error"].startswith("worker bug: ")
    assert broker.produced == []
    assert broker.committed == [0, 1]


@pytest.mark.parametrize("event", [
    {"eventId": "e1", "itemId": PARENT, "type": "movie", "step": "transcode", "status": "done"},
    {"extraId": "../../etc", "type": "extra"},
    {},
])
def test_a_malformed_trigger_is_committed_unread(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, event: dict,
) -> None:
    catalog = Catalog("queued")
    broker, encodes = run(monkeypatch, tmp_path, [event, trigger()], catalog)
    # Only the well-formed one is read and run.
    assert catalog.reads == [f"/api/analyze/extras/{EXTRA}"]
    assert len(encodes) == 1
    assert broker.committed == [0, 1]


def test_the_item_loop_skips_an_extras_trigger(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # The trigger has no itemId: an item worker pointed at the extras
    # topic by mistake reads nothing, runs nothing, and commits.
    stop = threading.Event()
    broker = Broker([trigger()], stop)
    catalog = Catalog("queued")
    monkeypatch.setattr(worker, "build_consumer", lambda *a, **k: broker)
    monkeypatch.setattr(worker, "build_producer", lambda *a, **k: broker)
    client = KatalogClient(BASE, f"{BASE}/token", "worker", "not-a-secret")
    client._http = httpx.Client(transport=httpx.MockTransport(catalog))
    worker.run_worker(client, tmp_path, "broker.test:9092", "transcoder-workers", CONSUME,
                      PRODUCE, "PLAINTEXT", SETTINGS, stop)
    assert catalog.reads == []
    assert catalog.writes == []
    assert broker.produced == []
    assert broker.committed == [0]
