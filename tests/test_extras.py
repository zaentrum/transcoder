"""The extras mode: trailers and other bonus material of a title, on a
consumer of their own (`catalog.extra.queued` -> `catalog.extra.transcoded`).
"""

from __future__ import annotations

import json

import pytest

from transcoder.kafka import build_extra_event, is_retry, parse_extra_id, parse_item_id

EXTRA = "1b5c2a8e-0000-4000-8000-0000000000e1"
PARENT = "ea886f9b-0d06-4f0f-babb-d2a1162f9b01"


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
