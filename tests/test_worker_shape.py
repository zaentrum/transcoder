"""Shape tests for the transcoder — everything that runs without
ffmpeg/CUDA being installed in the test environment.

The full encode path is covered live in the cluster (a successful
encode lands a prepared.mkv on the packages PVC; the packager picks
it up; the audit row carries the profile + size + duration). These
tests catch the cheap stuff: config validation, dataclass shape,
decision logic, profile picking.
"""

from __future__ import annotations

import pytest

from transcoder.config import Config
from transcoder.decision import HEVC_CODEC_NAMES, decide
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
    # Defaults retained when env vars unset.
    assert cfg.maxrate_1080p_mbps == 8
    assert cfg.nvenc_preset == "p5"
    assert cfg.packages_root == "/var/lib/katalog/packages"


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
    import transcoder.ffmpeg
    import transcoder.katalog
    import transcoder.worker

    # Touch each so lint doesn't strip the import.
    assert transcoder.__doc__
    assert transcoder.config.Config
    assert transcoder.decision.decide
    assert transcoder.ffmpeg.pick_profile
    assert transcoder.katalog.ClaimedItem
    assert transcoder.worker.run_worker
