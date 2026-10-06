"""Where a transcode's handoff goes: into the inbox the worker record
names (`library.inboxDir`, the v2 library layout), else into
`{packages_root}/_inbox/...` exactly as before (the legacy layout).

The handlers run with the real KatalogClient against a fake catalog API
(an httpx mock transport), so the record is read as in production; the
encode (`_process_one`) is a recorder of the inbox it is handed. Real
encodes into a v2 inbox are in test_encode_real.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from structlog.testing import capture_logs

from transcoder import extras, worker
from transcoder.ffmpeg import EncodeSettings
from transcoder.katalog import KatalogClient, LibraryRecord
from transcoder.worker import InboxError, handoff_inbox

ITEM = "f001aeff-9c18-4183-b51b-51403af2515e"
OTHER = "0b6c1e2d-0000-4000-8000-000000000002"
EXTRA = "16aa63f3-0000-4000-8000-0000000000e1"
PARENT = "ea886f9b-0d06-4f0f-babb-d2a1162f9b01"
BASE = "http://catalog.test"
LIB = "/var/lib/katalog"
WORK = f"{LIB}/.work"
PACKAGES = Path(f"{LIB}/packages")
ITEM_INBOX = f"{WORK}/inbox/{ITEM}"
EXTRA_INBOX = f"{WORK}/inbox/extra-{EXTRA}"
ITEM_TOPIC = "stube.catalog.item.transcoded"
EXTRA_TOPIC = "stube.catalog.extra.transcoded"
ITEM_STEP = f"/api/analyze/items/{ITEM}/steps/transcode"
EXTRA_STEP = f"/api/analyze/extras/{EXTRA}/steps/transcode"

# The record has no `library` key at all (as opposed to "library": null).
ABSENT = object()


def item_library(**fields: object) -> dict:
    """An item's library block, shaped as the catalog's contract has it."""
    return {
        "contract": 1, "root": LIB, "itemDir": f"{LIB}/movies/f0/{ITEM}", "blocked": None,
        "source": {"sourceId": "0b6c", "recorded": False,
                   "recordDir": f"{LIB}/movies/f0/{ITEM}/sources/0b6c",
                   "libraryPath": "Sintel (2010).mkv", "sizeBytes": 1234567890,
                   "qh1": "sha256:" + "0" * 64},
        "inboxDir": ITEM_INBOX,
        "build": {"versionId": "9a2e", "stagingDir": f"{WORK}/staging/9a2e",
                  "versionDir": f"{LIB}/movies/f0/{ITEM}/versions/9a2e",
                  "createdBy": "katalog-manager", "chapters": None, "chaptersFrom": None,
                  "segments": []},
        "current": None,
        **fields,
    }


def extra_library(**fields: object) -> dict:
    """An extra's library block, shaped as the catalog's contract has it."""
    return {
        "contract": 1, "itemDir": f"{LIB}/movies/ea/{PARENT}", "inboxDir": EXTRA_INBOX,
        "stagingDir": f"{WORK}/staging/extra-{EXTRA}",
        "extraDir": f"{LIB}/movies/ea/{PARENT}/extras/{EXTRA}", "recorded": False,
        "record": {"kind": "trailer", "title": "Trailer", "localizedTitles": {},
                   "language": "zxx", "seasonNumber": None, "origin": None,
                   "createdAt": "2026-10-06T08:00:00Z", "createdBy": "katalog-manager/api"},
        "original": {"name": "trailer.mov", "sizeBytes": 123456789,
                     "qh1": "sha256:" + "0" * 64},
        **fields,
    }


class Catalog:
    """The worker protocol for one item and one extra: their records, the
    item's step statuses, and every write."""

    def __init__(self, *, library: object = ABSENT, steps: dict | None = None,
                 state: str = "queued") -> None:
        self.item = {"id": ITEM, "type": "movie", "title": "Sintel", "year": 2010,
                     "durationMs": 888000, "path": f"{WORK}/incoming/Sintel (2010).mkv"}
        self.extra = {"id": EXTRA, "type": "extra", "parentId": PARENT, "parentType": "movie",
                      "parentTitle": "Sintel", "kind": "trailer", "title": "Trailer",
                      "language": "en", "seasonNumber": None,
                      "path": f"{WORK}/extras/sintel/trailer.mov", "state": state,
                      "removedAt": None}
        if library is not ABSENT:
            self.item["library"] = self.extra["library"] = library
        self.steps = steps or {}
        self.writes: list[tuple[str, str, object]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/token":
            return httpx.Response(200, json={"access_token": "t", "expires_in": 300})
        if request.method == "GET":
            if path == f"/api/analyze/items/{ITEM}":
                return httpx.Response(200, json=self.item)
            if path == f"/api/analyze/items/{ITEM}/steps":
                return httpx.Response(200, json={"itemId": ITEM, "steps": self.steps})
            if path == f"/api/analyze/extras/{EXTRA}":
                return httpx.Response(200, json=self.extra)
            return httpx.Response(404)
        self.writes.append((request.method, path, json.loads(request.content or b"null")))
        return httpx.Response(200, json={})


class Producer:
    def __init__(self) -> None:
        self.produced: list[tuple[str, str]] = []

    def produce(self, topic: str, key: bytes, value: bytes) -> None:
        self.produced.append((topic, key.decode()))

    def flush(self, *_args: object) -> int:
        return 0


def _client(catalog: Catalog) -> KatalogClient:
    client = KatalogClient(BASE, f"{BASE}/token", "worker", "not-a-secret")
    client._http = httpx.Client(transport=httpx.MockTransport(catalog))
    return client


def _recorder(monkeypatch: pytest.MonkeyPatch, module: object) -> list[Path]:
    inboxes: list[Path] = []

    def encode(_item, _steps, inbox: Path, _settings) -> bool:
        inboxes.append(inbox)
        return True

    monkeypatch.setattr(module, "_process_one", encode)
    return inboxes


def run_item(monkeypatch: pytest.MonkeyPatch, catalog: Catalog) -> tuple[list[Path], Producer]:
    """One `analyzed` event for ITEM through the item handler."""
    inboxes, producer = _recorder(monkeypatch, worker), Producer()
    worker._handle_item(item_id=ITEM, upstream_type="movie", client=_client(catalog),
                        producer=producer, produce_topic=ITEM_TOPIC, packages_root=PACKAGES,
                        settings=EncodeSettings())
    return inboxes, producer


def run_extra(monkeypatch: pytest.MonkeyPatch, catalog: Catalog) -> tuple[list[Path], Producer]:
    """One `catalog.extra.queued` trigger for EXTRA through the extras handler."""
    inboxes, producer = _recorder(monkeypatch, extras), Producer()
    trigger = {"eventId": "9f2b", "extraId": EXTRA, "parentId": PARENT, "type": "extra",
               "kind": "trailer", "step": "transcode", "status": "queued", "source": "api"}
    extras._handle_extra(EXTRA, trigger, _client(catalog), producer, EXTRA_TOPIC, PACKAGES,
                         EncodeSettings())
    return inboxes, producer


# --------------------------------------------------------- both layouts
def test_a_v2_item_hands_off_into_the_inbox_its_record_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inboxes, producer = run_item(monkeypatch, Catalog(library=item_library()))
    assert inboxes == [Path(f"{WORK}/inbox/{ITEM}")]
    assert producer.produced == [(ITEM_TOPIC, ITEM)]


def test_a_v2_extra_hands_off_into_the_inbox_its_record_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inboxes, producer = run_extra(monkeypatch, Catalog(library=extra_library()))
    assert inboxes == [Path(f"{WORK}/inbox/extra-{EXTRA}")]
    assert producer.produced == [(EXTRA_TOPIC, EXTRA)]


@pytest.mark.parametrize("library", [ABSENT, None])
def test_a_legacy_item_hands_off_into_packages_inbox_as_before(
    monkeypatch: pytest.MonkeyPatch, library: object,
) -> None:
    inboxes, producer = run_item(monkeypatch, Catalog(library=library))
    assert inboxes == [PACKAGES / "_inbox" / ITEM]
    assert inboxes == [worker._inbox_dir(PACKAGES, ITEM)]
    assert producer.produced == [(ITEM_TOPIC, ITEM)]


@pytest.mark.parametrize("library", [ABSENT, None])
def test_a_legacy_extra_hands_off_into_packages_inbox_as_before(
    monkeypatch: pytest.MonkeyPatch, library: object,
) -> None:
    inboxes, producer = run_extra(monkeypatch, Catalog(library=library))
    assert inboxes == [PACKAGES / "_inbox" / f"extra-{EXTRA}"]
    assert inboxes == [extras.extra_inbox_dir(PACKAGES, EXTRA)]
    assert producer.produced == [(EXTRA_TOPIC, EXTRA)]


def test_the_inbox_names_are_the_same_on_both_layouts() -> None:
    # <itemId> and extra-<extraId>: under _inbox/ (legacy), inbox/ (v2).
    assert worker._inbox_dir(PACKAGES, ITEM).name == Path(ITEM_INBOX).name == ITEM
    assert extras.extra_inbox_dir(PACKAGES, EXTRA).name == Path(EXTRA_INBOX).name
    assert extras.extra_inbox_name(EXTRA) == f"extra-{EXTRA}"


@pytest.mark.parametrize("raw", [f"{WORK}/inbox/{ITEM}/", f"{LIB}//.work/inbox/{ITEM}",
                                 f"{LIB}/./.work/inbox/{ITEM}"])
def test_a_v2_inbox_is_taken_as_a_path(raw: str) -> None:
    legacy = worker._inbox_dir(PACKAGES, ITEM)
    library = LibraryRecord(contract=1, inbox_dir=raw)
    assert handoff_inbox(library, ITEM, legacy) == Path(ITEM_INBOX)
    assert handoff_inbox(None, ITEM, legacy) is legacy


# -------------------------------------------------------------- refusals
# A v2 record names the inbox; one this worker may not write is refused —
# never swapped for the legacy inbox, where a v2 packager never looks.
ITEM_REFUSALS = [
    pytest.param(item_library(inboxDir=None), "names no inboxDir", id="no-inbox"),
    pytest.param({k: v for k, v in item_library().items() if k != "inboxDir"},
                 "names no inboxDir", id="no-key"),
    pytest.param(item_library(inboxDir=""), "names no inboxDir", id="empty"),
    pytest.param(item_library(inboxDir=f"var/lib/katalog/.work/inbox/{ITEM}"),
                 "is not <work root>/inbox/", id="relative"),
    pytest.param(item_library(inboxDir=f"{WORK}/inbox/../../movies/f0/{ITEM}"),
                 "is not <work root>/inbox/", id="dot-dot"),
    # The title's own folder ends in its id too: never emptied.
    pytest.param(item_library(inboxDir=f"{LIB}/movies/f0/{ITEM}"),
                 "is not <work root>/inbox/", id="item-dir"),
    pytest.param(item_library(inboxDir=f"{WORK}/staging/{ITEM}"),
                 "is not <work root>/inbox/", id="staging"),
    pytest.param(item_library(inboxDir=f"{WORK}/inbox/{OTHER}"),
                 "is not <work root>/inbox/", id="other-item"),
    pytest.param(item_library(inboxDir=f"{WORK}/inbox/extra-{ITEM}"),
                 "is not <work root>/inbox/", id="extra-name"),
    pytest.param(item_library(inboxDir=f"{LIB}/packages/_inbox/{ITEM}"),
                 "is not <work root>/inbox/", id="legacy-inbox"),
    pytest.param(item_library(contract=2), "library contract 2", id="contract-2"),
    pytest.param({k: v for k, v in item_library().items() if k != "contract"},
                 "library contract None", id="no-contract"),
    pytest.param("v2", "library contract None", id="not-an-object"),
]


@pytest.mark.parametrize(("library", "error"), ITEM_REFUSALS)
def test_a_v2_item_record_without_a_usable_inbox_fails_the_step(
    monkeypatch: pytest.MonkeyPatch, library: object, error: str,
) -> None:
    catalog = Catalog(library=library)
    with capture_logs() as logs:
        inboxes, producer = run_item(monkeypatch, catalog)
    assert inboxes == []
    assert producer.produced == []
    [(method, path, body)] = catalog.writes
    assert (method, path, body["status"]) == ("PUT", ITEM_STEP, "failed")
    assert error in body["error"] and body["error"].startswith("worker record: ")
    assert "transcoder.item.inbox_refused" in [e["event"] for e in logs]


@pytest.mark.parametrize(("library", "error"), [
    pytest.param(extra_library(inboxDir=None), "names no inboxDir", id="no-inbox"),
    # An item's name is no extra's, even in the right place.
    pytest.param(extra_library(inboxDir=f"{WORK}/inbox/{EXTRA}"),
                 "is not <work root>/inbox/extra-", id="item-name"),
    pytest.param(extra_library(inboxDir=f"{LIB}/movies/ea/{PARENT}/extras/{EXTRA}"),
                 "is not <work root>/inbox/extra-", id="extra-dir"),
    pytest.param(extra_library(inboxDir=f"{WORK}/staging/extra-{EXTRA}"),
                 "is not <work root>/inbox/extra-", id="staging"),
    pytest.param(extra_library(inboxDir=f"{LIB}/packages/_inbox/extra-{EXTRA}"),
                 "is not <work root>/inbox/extra-", id="legacy-inbox"),
    pytest.param(extra_library(contract=2), "library contract 2", id="contract-2"),
    pytest.param([], "library contract None", id="not-an-object"),
])
def test_a_v2_extra_record_without_a_usable_inbox_fails_the_step(
    monkeypatch: pytest.MonkeyPatch, library: object, error: str,
) -> None:
    catalog = Catalog(library=library)
    with capture_logs() as logs:
        inboxes, producer = run_extra(monkeypatch, catalog)
    assert inboxes == []
    assert producer.produced == []
    [(method, path, body)] = catalog.writes
    assert (method, path, body["status"]) == ("PUT", EXTRA_STEP, "failed")
    assert error in body["error"]
    assert "transcoder.extra.inbox_refused" in [e["event"] for e in logs]


def test_handoff_inbox_raises_inbox_error() -> None:
    with pytest.raises(InboxError, match="names no inboxDir"):
        handoff_inbox(LibraryRecord(contract=1, inbox_dir=None), ITEM, PACKAGES)


# ------------------------------------------- finished: no inbox needed
@pytest.mark.parametrize("status", ["done", "not_applicable", "skipped"])
def test_a_finished_item_passes_the_chain_on_whatever_its_library_block(
    monkeypatch: pytest.MonkeyPatch, status: str,
) -> None:
    # The guard runs first: nothing is written, so no inbox is looked at.
    catalog = Catalog(library=item_library(contract=2), steps={"transcode": status})
    inboxes, producer = run_item(monkeypatch, catalog)
    assert inboxes == []
    assert catalog.writes == []
    assert producer.produced == [(ITEM_TOPIC, ITEM)]


@pytest.mark.parametrize("state", ["transcoded", "packaging", "ready"])
def test_a_finished_extra_passes_the_chain_on_whatever_its_library_block(
    monkeypatch: pytest.MonkeyPatch, state: str,
) -> None:
    catalog = Catalog(library=extra_library(inboxDir=None), state=state)
    inboxes, producer = run_extra(monkeypatch, catalog)
    assert inboxes == []
    assert catalog.writes == []
    assert producer.produced == [(EXTRA_TOPIC, EXTRA)]
