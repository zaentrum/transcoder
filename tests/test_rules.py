"""The report of the rules this transcoder runs with (rules.py): what it
says, as the catalog's compliance check reads it, and how it goes —
against a fake catalog API (an httpx mock transport) that takes it,
refuses it, or is older than the route and answers 404 or 405."""

from __future__ import annotations

import json
import threading

import httpx
import pytest
from structlog.testing import capture_logs

from transcoder.config import Config
from transcoder.decision import CPU_ENCODERS, NVENC_ENCODERS, parse_ladder
from transcoder.katalog import KatalogClient
from transcoder.rules import RulesReporter, canonical_ladder, image_version, rules_payload

BASE = "http://catalog.test"


def config(**kw: object) -> Config:
    """The config of a deployment that sets nothing but its credentials and
    what kw names: every other value its default."""
    return Config(katalog_api_url=BASE, oidc_token_url=f"{BASE}/token", oidc_client_id="worker",
                  oidc_client_secret="not-a-secret", **kw)  # type: ignore[arg-type]


def payload(cfg: Config, encoders=NVENC_ENCODERS, version: str | None = "3f2a9c1") -> dict:
    return rules_payload(cfg, encoders, parse_ladder(cfg.ladder), parse_ladder(cfg.extra_ladder),
                         version)


def test_the_payload_is_the_catalogs_contract() -> None:
    # HEVC-only packages on a GPU host, everything else the defaults: the
    # catalog's example, to the letter, and the ladder in parts.
    got = payload(config(ladder="source:hevc", extra_ladder="source:hevc"))
    assert got == {"version": "3f2a9c1", "rules": {
        "caps": {"movie1080Mbps": 8, "movie2160Mbps": 14, "movieMaxGiB": 15,
                 "episode1080Mbps": 6, "episode2160Mbps": 8},
        "bucket2160": {"minHeight": 2000, "minWidth": 3200},
        "encoder": {"codec": "hevc_nvenc", "preset": "p5", "cq": 23,
                    "maxrate1080Mbps": None, "maxrate2160Mbps": None},
        "ladder": "source:hevc", "extraLadder": "source:hevc",
        "ladderRungs": [{"name": "source", "codec": "hevc", "codecNamed": True,
                         "maxrateBps": None}],
    }}
    # what goes on the wire: whole numbers as such, null as null
    assert json.dumps(got["rules"]["caps"]) == (
        '{"movie1080Mbps": 8, "movie2160Mbps": 14, "movieMaxGiB": 15, '
        '"episode1080Mbps": 6, "episode2160Mbps": 8}')


def test_the_payload_says_the_values_this_process_runs_with() -> None:
    # caps switched off are 0, a fraction stays one, the overrides are set;
    # without NVENC the encoder is libx265 with its preset and crf
    cfg = config(cap_movie_2160_mbps=0.0, cap_episode_1080_mbps=5.5, maxrate_2160p_mbps=20,
                 ladder="", x265_preset="slow", x265_crf=22)
    rules = payload(cfg, encoders=CPU_ENCODERS)["rules"]
    assert rules["caps"]["movie2160Mbps"] == 0
    assert rules["caps"]["episode1080Mbps"] == 5.5
    assert rules["encoder"] == {"codec": "libx265", "preset": "slow", "cq": None, "crf": 22,
                                "maxrate1080Mbps": None, "maxrate2160Mbps": 20}
    # the empty default is one source rendition, its codec left to the
    # default (the CPU rule may keep an H.264 source)
    assert rules["ladder"] == "source"
    assert rules["ladderRungs"] == [{"name": "source", "codec": "hevc", "codecNamed": False,
                                     "maxrateBps": None}]
    assert rules["extraLadder"] == "720p:h264,480p:h264"


@pytest.mark.parametrize(("spec", "canonical"), [
    ("", "source"),
    ("source", "source"),
    (" SOURCE:HEVC ", "source:hevc"),
    ("source,720p", "source,720p"),
    ("source,720p:h264:2500k", "source,720p:h264:2500k"),
    ("1080p:3M,480p:h264:1.5m", "1080p:3M,480p:h264:1500k"),
])
def test_a_ladder_is_written_as_its_spec_reads(spec: str, canonical: str) -> None:
    assert canonical_ladder(parse_ladder(spec)) == canonical
    # and reads back as the same rungs
    assert parse_ladder(canonical) == parse_ladder(spec)


def test_the_version_is_the_images(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGE_VERSION", " v1.4.0 ")
    assert image_version() == "v1.4.0"
    monkeypatch.delenv("IMAGE_VERSION")
    assert image_version() == "0.1.0"  # the installed package's


def rules_logs(logs: list[dict]) -> list[dict]:
    """The reporter's lines of logs (the client logs its token too)."""
    return [e for e in logs if e["event"].startswith("transcoder.rules.")]


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


def test_the_report_is_put_where_the_catalog_takes_it() -> None:
    body = payload(config(ladder="source:hevc"))
    client, seen = client_for(
        lambda _r: httpx.Response(200, json={"worker": "transcoder", "changed": True}))
    with capture_logs() as logs:
        assert RulesReporter(client, body).report_once()
    [request] = seen
    assert (request.method, request.url.path) == ("PUT", "/api/analyze/rules/transcoder")
    assert request.headers["authorization"] == "Bearer t"
    assert json.loads(request.content) == body
    assert [e["event"] for e in rules_logs(logs)] == ["transcoder.rules.reported"]


@pytest.mark.parametrize("status", [404, 405])
def test_an_older_catalog_is_said_once_and_asked_again(status: int) -> None:
    client, seen = client_for(lambda _r: httpx.Response(status, text="no route"))
    reporter = RulesReporter(client, payload(config()))
    with capture_logs() as logs:
        assert not reporter.report_once()
        assert not reporter.report_once()
    assert len(seen) == 2  # asked again, every round
    [said] = rules_logs(logs)
    assert (said["event"], said["log_level"]) == ("transcoder.rules.unsupported", "info")
    assert said["status"] == status


@pytest.mark.parametrize("answer", [httpx.Response(400, text="caps.movie1080Mbps is a JSON string"),
                                    httpx.Response(503, text="migration 051 is not applied"),
                                    httpx.ConnectError("down")])
def test_a_failed_report_never_stops_the_worker(answer) -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        if isinstance(answer, Exception):
            raise answer
        return answer

    client, _ = client_for(handler)
    with capture_logs() as logs:
        assert not RulesReporter(client, payload(config())).report_once()  # logged, never raised
    assert [e["log_level"] for e in rules_logs(logs)] == ["warning"]


def test_the_report_goes_at_start_and_every_interval() -> None:
    stop = threading.Event()
    calls: list[str] = []

    class Catalog:
        def put_rules(self, worker: str, _payload: dict) -> httpx.Response:
            calls.append(worker)
            if len(calls) == 3:
                stop.set()
            if len(calls) == 2:
                raise RuntimeError("a hiccup")  # the next round goes all the same
            return httpx.Response(200)

    thread = threading.Thread(target=RulesReporter(Catalog(), payload(config()), interval=0.01).run,
                              kwargs={"stop": stop})
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert calls == ["transcoder"] * 3
