"""The catalog client's requests, against a fake catalog API (an httpx
mock transport): what the worker sends for an item's transcode step and
for an extra's, and how it reads the worker records, their `library`
block (the v2 library layout) included."""

from __future__ import annotations

import json

import httpx
import pytest

from transcoder.katalog import ClaimedExtra, ClaimedItem, KatalogClient, LibraryRecord

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


# ------------------------------------------------------- the library block
# The worker records on the v2 library layout, as the catalog's contract
# (platform-library/1) shapes them.
ITEM_RECORD = {
    "id": ITEM, "type": "movie", "title": "Sintel", "year": 2010, "durationMs": 888000,
    "path": "/var/lib/katalog/.work/incoming/Sintel (2010).mkv", "seasonNumber": None,
    "episodeNumber": None, "seriesTmdbId": None, "movieTmdbId": 45745,
    "hasOwnPoster": False, "hasOwnBackdrop": False,
    "library": {
        "contract": 1, "root": "/var/lib/katalog",
        "itemDir": f"/var/lib/katalog/movies/7a/{ITEM}", "blocked": None,
        "source": {"sourceId": "0b6c", "recorded": False,
                   "recordDir": f"/var/lib/katalog/movies/7a/{ITEM}/sources/0b6c",
                   "libraryPath": "Sintel (2010).mkv", "sizeBytes": 1234567890,
                   "qh1": "sha256:" + "0" * 64},
        "inboxDir": f"/var/lib/katalog/.work/inbox/{ITEM}",
        "build": {"versionId": "9a2e", "stagingDir": "/var/lib/katalog/.work/staging/9a2e",
                  "versionDir": f"/var/lib/katalog/movies/7a/{ITEM}/versions/9a2e",
                  "createdBy": "katalog-manager", "chapters": None, "chaptersFrom": None,
                  "segments": []},
        "current": None,
    },
}
EXTRA_LIBRARY = {
    "contract": 1, "itemDir": f"/var/lib/katalog/movies/ea/{PARENT}",
    "inboxDir": f"/var/lib/katalog/.work/inbox/extra-{EXTRA}",
    "stagingDir": f"/var/lib/katalog/.work/staging/extra-{EXTRA}",
    "extraDir": f"/var/lib/katalog/movies/ea/{PARENT}/extras/{EXTRA}", "recorded": False,
    "record": {"kind": "trailer", "title": "Trailer", "localizedTitles": {}, "language": "zxx",
               "seasonNumber": None, "origin": None, "createdAt": "2026-10-06T08:00:00Z",
               "createdBy": "katalog-manager/api"},
    "original": {"name": "trailer.mov", "sizeBytes": 123456789, "qh1": "sha256:" + "0" * 64},
}


def test_get_item_reads_the_library_block() -> None:
    client, seen = client_for(lambda _r: httpx.Response(200, json=ITEM_RECORD))
    item = client.get_item(ITEM)
    assert item.library == LibraryRecord(contract=1,
                                         inbox_dir=f"/var/lib/katalog/.work/inbox/{ITEM}")
    assert item.path == ITEM_RECORD["path"]
    [request] = seen
    assert (request.method, request.url.path) == ("GET", f"/api/analyze/items/{ITEM}")


def test_get_extra_reads_the_library_block() -> None:
    client, _ = client_for(lambda _r: httpx.Response(200, json={**RECORD,
                                                                 "library": EXTRA_LIBRARY}))
    assert client.get_extra(EXTRA).library == LibraryRecord(
        contract=1, inbox_dir=f"/var/lib/katalog/.work/inbox/extra-{EXTRA}")


@pytest.mark.parametrize("body", [{}, {"library": None}])
def test_a_record_without_a_library_block_is_the_legacy_layout(body: dict) -> None:
    item_body = {k: v for k, v in ITEM_RECORD.items() if k != "library"}
    client, _ = client_for(lambda _r: httpx.Response(200, json={**item_body, **body}))
    assert client.get_item(ITEM).library is None
    client, _ = client_for(lambda _r: httpx.Response(200, json={**RECORD, **body}))
    assert client.get_extra(EXTRA).library is None
    assert ClaimedItem.from_json({**item_body, **body}).library is None


@pytest.mark.parametrize(("block", "expected"), [
    ({"contract": 1}, LibraryRecord(contract=1, inbox_dir=None)),
    ({"contract": 1, "inboxDir": ""}, LibraryRecord(contract=1, inbox_dir=None)),
    ({"contract": 1, "inboxDir": 7}, LibraryRecord(contract=1, inbox_dir=None)),
    ({"inboxDir": "/w/inbox/x"}, LibraryRecord(contract=None, inbox_dir="/w/inbox/x")),
    ({"contract": 2, "inboxDir": "/w/inbox/x"}, LibraryRecord(contract=2, inbox_dir="/w/inbox/x")),
    ("v2", LibraryRecord(contract=None, inbox_dir=None)),
    ([], LibraryRecord(contract=None, inbox_dir=None)),
])
def test_a_library_block_says_v2_even_when_it_is_unusable(block: object,
                                                          expected: LibraryRecord) -> None:
    # Kept as it reads, never dropped: the worker refuses it, rather than
    # falling back to the legacy inbox, where a v2 packager never looks.
    assert LibraryRecord.from_json(block) == expected


@pytest.mark.parametrize("answer", [httpx.Response(500, text="boom"), httpx.ConnectError("down")])
def test_step_writes_are_best_effort(answer) -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        if isinstance(answer, Exception):
            raise answer
        return answer

    client, _ = client_for(handler)
    client.upsert_extra_step(EXTRA, "done")  # logged, never raised
    client.upsert_step(ITEM, "done")
