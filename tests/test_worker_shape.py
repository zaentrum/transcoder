"""Shape tests for the transcoder — everything that runs without
ffmpeg/CUDA being installed in the test environment.

The full encode path is covered live in the cluster (a successful
encode lands a prepared.mkv on the packages PVC; the packager picks
it up; the audit row carries the profile + size + duration). These
tests catch the cheap stuff: config validation, dataclass shape,
decision logic, profile picking.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from transcoder.config import Config, normalize_topic_prefix
from transcoder.decision import HEVC_CODEC_NAMES, Caps, LadderError, decide, parse_ladder
from transcoder.ffmpeg import pick_profile
from transcoder.katalog import ClaimedItem


# ---------------------------------------------------------------- config
def test_config_from_env_requires_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KATALOG_API_URL", raising=False)
    with pytest.raises(RuntimeError, match="KATALOG_API_URL"):
        Config.from_env()


def test_config_from_env_full(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KATALOG_API_URL", "http://katalog-app")
    monkeypatch.setenv("OIDC_TOKEN_URL", "https://sso.example/token")
    monkeypatch.setenv("OIDC_CLIENT_ID", "katalog")
    monkeypatch.setenv("OIDC_CLIENT_SECRET", "x" * 32)
    monkeypatch.setenv("NVENC_CQ", "20")
    monkeypatch.setenv("NVENC_MAXRATE_2160P_MBPS", "20")
    cfg = Config.from_env()
    assert cfg.katalog_api_url == "http://katalog-app"
    assert cfg.nvenc_cq == 20
    assert cfg.maxrate_2160p_mbps == 20
    # Defaults retained when env vars unset. An unset maxrate overrides
    # nothing: the cap (8 Mbit/s for a 1080 movie) is the maxrate.
    assert cfg.maxrate_1080p_mbps is None
    assert cfg.caps.movie_1080 == 8_000_000
    assert cfg.nvenc_preset == "p5"
    assert cfg.packages_root == "/var/lib/katalog/packages"


def _extras_env(monkeypatch: pytest.MonkeyPatch, **env: str) -> Config:
    monkeypatch.setenv("KATALOG_API_URL", "http://katalog-app")
    monkeypatch.setenv("OIDC_TOKEN_URL", "https://sso.example/token")
    monkeypatch.setenv("OIDC_CLIENT_ID", "katalog")
    monkeypatch.setenv("OIDC_CLIENT_SECRET", "x" * 32)
    for key in ("KAFKA_TOPIC_PREFIX", "EXTRA_LADDER", "EXTRAS_GROUP_ID", "CONSUME_TOPIC",
                "PRODUCE_TOPIC", "KAFKA_GROUP_ID"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Config.from_env()


def test_config_extras_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _extras_env(monkeypatch)
    assert cfg.extra_ladder == "720p:h264,480p:h264"
    assert cfg.extras_group_id == "transcoder-extras"
    assert cfg.extras_consume_topic == "stube.catalog.extra.queued"
    assert cfg.extras_produce_topic == "stube.catalog.extra.transcoded"
    # The items keep their own topics and group.
    assert (cfg.consume_topic, cfg.produce_topic, cfg.kafka_group_id) == (
        "stube.catalog.item.analyzed", "stube.catalog.item.transcoded", "transcoder-workers")


def test_config_extras_topics_carry_the_tenant_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _extras_env(monkeypatch, KAFKA_TOPIC_PREFIX="tenant-a.",
                      EXTRAS_GROUP_ID="tenant-a-transcoder-extras")
    assert cfg.extras_consume_topic == "tenant-a.catalog.extra.queued"
    assert cfg.extras_produce_topic == "tenant-a.catalog.extra.transcoded"
    assert cfg.extras_group_id == "tenant-a-transcoder-extras"


@pytest.mark.parametrize(("raw", "prefix"), [
    (None, "stube."), ("", "stube."), ("  ", "stube."), ("tenant", "tenant."),
    ("tenant.", "tenant."), (" tenant-b. ", "tenant-b."),
])
def test_topic_prefix_reads_as_katalog_manager_reads_it(raw: str | None, prefix: str) -> None:
    assert normalize_topic_prefix(raw) == prefix


def test_config_empty_extra_ladder_is_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # Empty must not fall to parse_ladder's empty case, the single HEVC
    # source rung: an extra has no on-the-fly fallback for HEVC.
    assert _extras_env(monkeypatch, EXTRA_LADDER=" ").extra_ladder == "720p:h264,480p:h264"
    cfg = _extras_env(monkeypatch, EXTRA_LADDER="1080p:h264,720p:h264")
    assert cfg.extra_ladder == "1080p:h264,720p:h264"


def test_config_bad_extra_ladder_fails_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(LadderError):
        _extras_env(monkeypatch, EXTRA_LADDER="720p:vp9")


CAP_KEYS = ("CAP_MOVIE_1080_MBPS", "CAP_MOVIE_2160_MBPS", "CAP_MOVIE_MAX_GIB",
            "CAP_EPISODE_1080_MBPS", "CAP_EPISODE_2160_MBPS", "NVENC_MAXRATE_1080P_MBPS",
            "NVENC_MAXRATE_2160P_MBPS")


def _caps_env(monkeypatch: pytest.MonkeyPatch, **env: str) -> Config:
    for key in CAP_KEYS:
        monkeypatch.delenv(key, raising=False)
    return _extras_env(monkeypatch, **env)


def test_config_caps_default_to_the_owners_table(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _caps_env(monkeypatch)
    assert cfg.caps == Caps(movie_1080=8_000_000, movie_2160=14_000_000,
                            movie_max_bytes=15 * 1024**3, episode_1080=6_000_000,
                            episode_2160=8_000_000)
    assert (cfg.maxrate_1080p_mbps, cfg.maxrate_2160p_mbps) == (None, None)


def test_config_caps_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _caps_env(monkeypatch, CAP_MOVIE_1080_MBPS="7.5", CAP_MOVIE_2160_MBPS="0",
                    CAP_MOVIE_MAX_GIB="20", CAP_EPISODE_1080_MBPS=" 5 ",
                    CAP_EPISODE_2160_MBPS="", NVENC_MAXRATE_1080P_MBPS="10",
                    NVENC_MAXRATE_2160P_MBPS="")
    # 0 switches a rule off; empty is the default.
    assert cfg.caps == Caps(movie_1080=7_500_000, movie_2160=0, movie_max_bytes=20 * 1024**3,
                            episode_1080=5_000_000, episode_2160=8_000_000)
    assert (cfg.maxrate_1080p_mbps, cfg.maxrate_2160p_mbps) == (10, None)


@pytest.mark.parametrize(("key", "value"), [
    ("CAP_MOVIE_1080_MBPS", "8M"), ("CAP_EPISODE_2160_MBPS", "-1"), ("CAP_MOVIE_MAX_GIB", "inf"),
    ("CAP_MOVIE_MAX_GIB", "nan"), ("NVENC_MAXRATE_1080P_MBPS", "0"),
    ("NVENC_MAXRATE_2160P_MBPS", "14.5"),
])
def test_config_bad_caps_fail_startup(monkeypatch: pytest.MonkeyPatch, key: str,
                                      value: str) -> None:
    with pytest.raises(RuntimeError, match=key):
        _caps_env(monkeypatch, **{key: value})


def test_config_decoder(monkeypatch: pytest.MonkeyPatch) -> None:
    # GPU decoding is on unless switched off (detect_decoder probes it).
    monkeypatch.delenv("DECODER", raising=False)
    assert _extras_env(monkeypatch).decoder == "auto"
    assert _extras_env(monkeypatch, DECODER="cpu").decoder == "cpu"


def test_config_takes_the_hevc_only_ladders(monkeypatch: pytest.MonkeyPatch) -> None:
    # HEVC-only packages: the same value for the items and the extras.
    cfg = _extras_env(monkeypatch, LADDER="source:hevc", EXTRA_LADDER="source:hevc")
    assert (cfg.ladder, cfg.extra_ladder) == ("source:hevc", "source:hevc")
    [rung] = parse_ladder(cfg.extra_ladder)
    assert (rung.name, rung.codec, rung.codec_named) == ("source", "hevc", True)


# ----------------------------------------------------------- claimed item
def test_claimed_item_from_json_minimum() -> None:
    body = {
        "id": "00111617-5a35-4c0c-afaa-ff9aae094f86",
        "type": "movie",
        "title": "Behind the Mask",
        "year": 2006,
        "durationMs": 5460000,
        "path": "/var/lib/katalog/media/movies/Behind the Mask (2006)/Behind the Mask (2006).mkv",
    }
    item = ClaimedItem.from_json(body)
    assert item.id == body["id"]
    assert item.duration_ms == 5460000


# --------------------------------------------------------------- decision
def test_decide_skips_hevc_sources() -> None:
    for codec in HEVC_CODEC_NAMES:
        probe = {"video": {"codec_name": codec, "width": 1920, "height": 1080}}
        decision = decide(probe)
        assert decision.skip is True
        assert decision.source_codec == codec
        assert "already_hevc" in decision.reason


def test_decide_encodes_h264() -> None:
    probe = {"video": {"codec_name": "h264", "width": 1920, "height": 1080}}
    decision = decide(probe)
    assert decision.skip is False
    assert decision.source_codec == "h264"


def test_decide_encodes_av1() -> None:
    # AV1 is "not HEVC" by the user rule, so we re-encode. We don't
    # special-case it; this test exists to lock the rule in.
    probe = {"video": {"codec_name": "av1", "width": 3840, "height": 2160}}
    decision = decide(probe)
    assert decision.skip is False


def test_decide_handles_missing_video_stream() -> None:
    decision = decide({"video": {}})
    assert decision.skip is False
    assert decision.reason == "no_video_stream_in_probe"


# ---------------------------------------------------------------- profile
def test_pick_profile_1080p() -> None:
    p = pick_profile(1920, 1080, 8, 14)
    assert p.label == "nvenc-1080p"
    assert p.maxrate_mbps == 8


def test_pick_profile_2160p() -> None:
    p = pick_profile(3840, 2160, 8, 14)
    assert p.label == "nvenc-2160p"
    assert p.maxrate_mbps == 14


def test_pick_profile_portrait_1080p() -> None:
    # Tall-format source (1080 wide) — picks 1080p profile because
    # neither dim exceeds 1920.
    p = pick_profile(1080, 1920, 8, 14)
    assert p.label == "nvenc-1080p"


def test_pick_profile_wide_uhd_treated_as_2160p() -> None:
    # Cinema-scope ~2048 width treated as UHD-tier because >1920.
    p = pick_profile(2048, 858, 8, 14)
    assert p.label == "nvenc-2160p"


# ---------------------------------------------------------- module layout
def test_module_layout_importable() -> None:
    # Mirrors packager.tests.test_worker_shape: we deliberately don't
    # touch transcoder.main here because it imports uvicorn + fastapi
    # which the lightweight CI test image doesn't install. The entry
    # point is exercised in-cluster.
    import transcoder
    import transcoder.config
    import transcoder.decision
    import transcoder.extras
    import transcoder.ffmpeg
    import transcoder.katalog
    import transcoder.worker

    # Touch each so lint doesn't strip the import.
    assert transcoder.__doc__
    assert transcoder.config.Config
    assert transcoder.decision.decide
    assert transcoder.extras.run_extras_worker
    assert transcoder.ffmpeg.pick_profile
    assert transcoder.katalog.ClaimedItem
    assert transcoder.worker.run_worker


class _Thread:
    """threading.Thread for main: kept, never run."""
    started: ClassVar[list] = []

    def __init__(self, target=None, kwargs=None, daemon=None, name=None) -> None:
        self.kwargs, self.name, self.alive = kwargs or {}, name, False

    def start(self) -> None:
        self.alive = True
        _Thread.started.append(self)

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout=None) -> None: ...


def _main(monkeypatch: pytest.MonkeyPatch, **patch: object):
    """Run transcoder.main with its probes, its loops' threads, the catalog
    client and uvicorn stood in for: (the probes' calls, the threads it
    started, the app it served). `patch` overrides main's names."""
    pytest.importorskip("uvicorn")
    pytest.importorskip("fastapi")
    from transcoder import main as m
    from transcoder.decision import NO_VULKAN, NVENC_ENCODERS
    from transcoder.ffmpeg import SelfTest

    _extras_env(monkeypatch, DECODER="auto")
    probed: list[tuple[str, object]] = []
    apps: list = []
    monkeypatch.setattr(_Thread, "started", [])
    monkeypatch.setattr(m, "_configure_logging", lambda: None)
    monkeypatch.setattr(m, "detect_encoders", lambda mode: NVENC_ENCODERS)
    monkeypatch.setattr(m, "detect_decoder",
                        lambda mode, encoders: probed.append((mode, encoders)) or "cuda")
    monkeypatch.setattr(m, "vulkan_preflight", lambda: NO_VULKAN)
    monkeypatch.setattr(m, "nvenc_selftest", lambda settings: SelfTest("ok"))

    class Client:
        def __init__(self, **_kw: object) -> None: ...

        def close(self) -> None: ...

    monkeypatch.setattr(m, "KatalogClient", Client)
    monkeypatch.setattr(m.threading, "Thread", _Thread)
    monkeypatch.setattr(m.uvicorn, "run", lambda app, **_k: apps.append(app))
    monkeypatch.setattr(m.signal, "signal", lambda *_a: None)
    for name, value in patch.items():
        monkeypatch.setattr(m, name, value)
    assert m.main() == 0
    return probed, _Thread.started, apps[0]


def test_main_hands_the_probed_decoder_to_both_loops(monkeypatch: pytest.MonkeyPatch) -> None:
    # What detect_decoder found is what the item loop and the extras loop
    # decode with (a main that dropped it would decode every source on the
    # CPU, EncodeSettings' default, without a word).
    from transcoder.decision import NVENC_ENCODERS

    probed, started, _app = _main(monkeypatch)
    assert probed == [("auto", NVENC_ENCODERS)]
    settings = {t.name: t.kwargs.get("settings") for t in started}
    assert settings["transcoder-worker"].decoder == "cuda"
    assert settings["transcoder-extras"].decoder == "cuda"


def test_healthz_fails_once_an_ffmpeg_would_not_die(monkeypatch: pytest.MonkeyPatch) -> None:
    # An ffmpeg stuck in the driver after its kill: the kubelet is to
    # restart the pod.
    from fastapi.testclient import TestClient

    from transcoder import ffmpeg as ff

    _probed, _started, app = _main(monkeypatch)
    client = TestClient(app)
    monkeypatch.setattr(ff, "_UNKILLED", [])
    ok = client.get("/healthz")
    assert (ok.status_code, ok.json()) == (200, {"ok": True, "why": None})
    monkeypatch.setattr(ff, "_UNKILLED", ["ffmpeg (pid 7) stood still for 600 s and had not "
                                          "died 30 s after its kill"])
    hung = client.get("/healthz")
    assert hung.status_code == 503
    assert hung.json() == {"ok": False, "why": "ffmpeg (pid 7) stood still for 600 s and had "
                                                "not died 30 s after its kill"}


def test_main_takes_no_item_when_nvencs_own_output_fails_the_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Another GPU or driver: every NVENC encode would fail its step, again
    # on each retry. No loop starts, and /readyz says why with a 503.
    from fastapi.testclient import TestClient

    from transcoder.ffmpeg import SelfTest

    why = "v0 came out with a DPB of 7, NVENC's own is at most 6"
    _probed, started, app = _main(monkeypatch,
                                  nvenc_selftest=lambda settings: SelfTest("failed", why))
    assert [t.name for t in started] == ["transcoder-rules"]
    ready = TestClient(app).get("/readyz")
    assert ready.status_code == 503
    assert ready.json() == {"ok": False, "extras": False, "nvenc_selftest": "failed", "why": why}


@pytest.mark.parametrize(("selftest", "decoder"), [("ok", "cuda"), ("unread", "cpu")])
def test_main_is_ready_after_its_selftest(monkeypatch: pytest.MonkeyPatch, selftest: str,
                                          decoder: str) -> None:
    # A DPB the check cannot read would send every GPU run to the CPU after
    # it: the sources are decoded on the CPU from the start.
    from fastapi.testclient import TestClient

    from transcoder.ffmpeg import SelfTest

    why = None if selftest == "ok" else "the DPB of v0 could not be read"
    _probed, started, app = _main(monkeypatch,
                                  nvenc_selftest=lambda settings: SelfTest(selftest, why))
    assert {t.name: t.kwargs["settings"].decoder for t in started
            if t.name != "transcoder-rules"} == {"transcoder-worker": decoder,
                                                 "transcoder-extras": decoder}
    ready = TestClient(app).get("/readyz")
    assert ready.status_code == 200
    assert ready.json() == {"ok": True, "extras": True, "nvenc_selftest": selftest, "why": why}

