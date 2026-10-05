"""The catalog client's requests, against a fake catalog API (an httpx
mock transport): what the worker sends for an item's transcode step and
for an extra's, and how it reads an extra's worker record."""

from __future__ import annotations

import json

import httpx
import pytest

from transcoder.katalog import ClaimedExtra, KatalogClient

BASE = "http://catalog.test"
ITEM = "7a1c0de0-0000-4000-8000-000000000001"
EXTRA = "1b5c2a8e-0000-4000-8000-0000000000e1"
PARENT = "ea886f9b-0d06-4f0f-babb-d2a1162f9b01"

RECORD = {
    "id": EXTRA, "type": "extra", "parentId": PARENT, "parentType": "movie",
    "parentTitle": "Big Buck Bunny", "kind": "trailer", "title": "Trailer", "language": "en",
    "seasonNumber": None, "path": "/var/lib/katalog/extras/big-buck-bunny/trailer.mov",
    "state": "queued",
}


def client_for(handler) -> tuple[KatalogClient, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def transport(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/token":
            return httpx.Response(200, json={"access_token": "t", "expires_in": 300})
        seen.append(request)
        return handler(request)

    client = KatalogClient(BASE, f"{BASE}/token", "worker", "not-a-secret")
    client._http = httpx.Client(transport=httpx.MockTransport(transport))
    return client, seen


def test_get_extra_reads_the_worker_record() -> None:
    client, seen = client_for(lambda _r: httpx.Response(200, json=RECORD))
    extra = client.get_extra(EXTRA)
    assert extra == ClaimedExtra(id=EXTRA, parent_id=PARENT, kind="trailer", title="Trailer",
                                 path=RECORD["path"], state="queued")
    assert extra.type == "extra"
    assert not extra.removed
    [request] = seen
    assert (request.method, request.url.path) == ("GET", f"/api/analyze/extras/{EXTRA}")
    assert request.headers["authorization"] == "Bearer t"


def test_get_extra_unknown_or_removed_is_none() -> None:
    client, _ = client_for(lambda _r: httpx.Response(404))
    assert client.get_extra(EXTRA) is None


def test_get_extra_that_says_it_was_removed() -> None:
    body = {**RECORD, "removedAt": "2026-10-05T08:00:00Z"}
    client, _ = client_for(lambda _r: httpx.Response(200, json=body))
    extra = client.get_extra(EXTRA)
    assert extra is not None and extra.removed


def test_get_extra_server_error_raises() -> None:
    # As get_item: the loop reports the run failed, and the catalog retries.
    client, _ = client_for(lambda _r: httpx.Response(503))
    with pytest.raises(httpx.HTTPStatusError):
        client.get_extra(EXTRA)


def test_get_extra_keeps_the_id_it_asked_for() -> None:
    # The inbox is named after the id; the one the request named is it.
    client, _ = client_for(lambda _r: httpx.Response(200, json={**RECORD, "id": "other"}))
    assert client.get_extra(EXTRA).id == EXTRA


def test_extra_step_is_put_with_the_item_step_body() -> None:
    client, seen = client_for(lambda _r: httpx.Response(200, json={}))
    client.upsert_extra_step(EXTRA, "in_progress")
    client.upsert_extra_step(EXTRA, "done", details="profile=x264-720p")
    client.upsert_extra_step(EXTRA, "failed", error="e" * 600)
    assert [(r.method, r.url.path) for r in seen] == [
        ("PUT", f"/api/analyze/extras/{EXTRA}/steps/transcode")] * 3
    assert [json.loads(r.content) for r in seen] == [
        {"status": "in_progress"},
        {"status": "done", "details": "profile=x264-720p"},
        {"status": "failed", "error": "e" * 500},
    ]


def test_item_step_is_put_where_it_always_was() -> None:
    client, seen = client_for(lambda _r: httpx.Response(200, json={}))
    client.upsert_step(ITEM, "not_applicable", details="skip codec=hevc")
    [request] = seen
    assert request.method == "PUT"
    assert request.url.path == f"/api/analyze/items/{ITEM}/steps/transcode"
    assert json.loads(request.content) == {"status": "not_applicable", "details": "skip codec=hevc"}


@pytest.mark.parametrize("answer", [httpx.Response(500, text="boom"), httpx.ConnectError("down")])
def test_step_writes_are_best_effort(answer) -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        if isinstance(answer, Exception):
            raise answer
        return answer

    client, _ = client_for(handler)
    client.upsert_extra_step(EXTRA, "done")  # logged, never raised
    client.upsert_step(ITEM, "done")
