"""A source the rules won't package as it is — Dolby Vision whose base
layer no other device plays — through the worker, with a fake probe: the
step fails with the reason, nothing is encoded or handed off (an older
run's handoff is removed too), and no event goes, so the title stays
unpackaged and its original is never retired. Real probes of such files
are in test_encode_real.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from transcoder import extras, worker
from transcoder.decision import parse_ladder
from transcoder.ffmpeg import EncodeSettings
from transcoder.katalog import ClaimedExtra, ClaimedItem

ITEM = "0f1e2d3c-0000-4000-8000-000000000001"
EXTRA = "1b5c2a8e-0000-4000-8000-0000000000e1"
PARENT = "ea886f9b-0d06-4f0f-babb-d2a1162f9b01"
REFUSAL = "Dolby Vision profile {} needs a tone-mapping encode; kept the original"


def _dv_probe(profile: int, compat: int) -> dict:
    """ffprobe's payload for a UHD Main 10 PQ stream with a Dolby Vision
    configuration record (transcoder.ffmpeg.ffprobe's shape)."""
    video = {"index": 0, "codec_type": "video", "codec_name": "hevc", "profile": "Main 10",
             "codec_tag_string": "dvh1" if profile == 5 else "hvc1", "width": 3840,
             "height": 2160, "pix_fmt": "yuv420p10le", "color_transfer": "smpte2084",
             "avg_frame_rate": "24000/1001",
             "side_data_list": [{"side_data_type": "DOVI configuration record",
                                 "dv_profile": profile, "dv_level": 6,
                                 "dv_bl_signal_compatibility_id": compat}]}
    return {"container": "mov,mp4,m4a,3gp,3g2,mj2", "duration_ms": 2000, "start_time": 0.0,
            "bit_rate": "9000000", "video": video, "video_index": 0, "audio": [],
            "subtitles": []}


class Steps:
    def __init__(self) -> None:
        self.steps: list[tuple[str, dict]] = []

    def upsert_step(self, _id: str, status: str, **kw: object) -> None:
        self.steps.append((status, kw))


class Catalog(Steps):
    """The item worker protocol: one item, no step finished yet."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.item = ClaimedItem(id=ITEM, type="movie", title="DV", year=None, duration_ms=None,
                                path=str(path))

    def get_item(self, item_id: str) -> ClaimedItem | None:
        return self.item if item_id == ITEM else None

    def get_steps(self, _item_id: str) -> dict[str, str]:
        return {}

    def get_extra(self, _extra_id: str) -> ClaimedExtra:
        return ClaimedExtra(id=EXTRA, parent_id=PARENT, kind="trailer", title="Trailer",
                            path=self.item.path, state="queued")

    def upsert_extra_step(self, extra_id: str, status: str, **kw: object) -> None:
        self.upsert_step(extra_id, status, **kw)


class Producer:
    def __init__(self) -> None:
        self.produced: list[tuple[str, str, dict]] = []

    def produce(self, topic: str, key: bytes, value: bytes) -> None:
        self.produced.append((topic, key.decode(), json.loads(value)))

    def flush(self, *_args: object) -> int:
        return 0


@pytest.fixture
def dv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A source file, and the worker's probe of it answering with the
    record of `dv.profile` / `dv.compat`; an encode would fail the test."""
    source = tmp_path / "dv.mp4"
    source.write_bytes(b"\0")

    class DV:
        path = source
        profile, compat = 5, 0

    monkeypatch.setattr(worker, "ffprobe", lambda _p: _dv_probe(DV.profile, DV.compat))

    def no_encode(*_a: object, **_k: object) -> None:
        raise AssertionError("nothing may be encoded")

    monkeypatch.setattr(worker, "run_encode", no_encode)
    return DV


@pytest.mark.parametrize(("profile", "compat"), [(5, 0), (7, 6)])
@pytest.mark.parametrize("ladder", ["", "source:hevc", "source,720p"])
def test_the_step_fails_with_the_reason_and_an_older_handoff_goes(
    dv, tmp_path: Path, profile: int, compat: int, ladder: str,
) -> None:
    dv.profile, dv.compat = profile, compat
    inbox = tmp_path / "_inbox" / ITEM
    inbox.mkdir(parents=True)
    (inbox / "prepared.mkv").write_bytes(b"an older run's")
    (inbox / "renditions.json").write_text("{}")
    steps = Steps()
    item = ClaimedItem(id=ITEM, type="movie", title="DV", year=None, duration_ms=None,
                       path=str(dv.path))
    ok = worker._process_one(item, steps, inbox, EncodeSettings(ladder=tuple(parse_ladder(ladder))))
    assert ok is False
    assert steps.steps == [("in_progress", {}), ("failed", {"error": REFUSAL.format(profile)})]
    assert not inbox.exists()


def test_an_item_that_is_refused_sends_no_event(dv, tmp_path: Path) -> None:
    catalog, producer = Catalog(dv.path), Producer()
    worker._handle_item(ITEM, "movie", catalog, producer, "stube.catalog.item.transcoded",  # type: ignore[arg-type]
                        tmp_path / "packages", EncodeSettings())
    assert [status for status, _ in catalog.steps] == ["in_progress", "failed"]
    assert producer.produced == []


def test_an_extra_that_is_refused_sends_no_event(dv, tmp_path: Path) -> None:
    catalog, producer = Catalog(dv.path), Producer()
    trigger = {"extraId": EXTRA, "parentId": PARENT, "type": "extra", "kind": "trailer",
               "step": "transcode", "status": "queued", "source": "api"}
    extras._handle_extra(EXTRA, trigger, catalog, producer,  # type: ignore[arg-type]
                         "stube.catalog.extra.transcoded", tmp_path / "packages",
                         EncodeSettings(ladder=tuple(parse_ladder("720p:h264,480p:h264"))))
    assert [(status, kw.get("error")) for status, kw in catalog.steps] == [
        ("in_progress", None), ("failed", REFUSAL.format(5))]
    assert producer.produced == []
    assert not (tmp_path / "packages").exists()


def test_dolby_vision_8_1_is_no_refusal(dv, tmp_path: Path) -> None:
    # Copied as before: not_applicable, and the chain goes on.
    dv.profile, dv.compat = 8, 1
    catalog, producer = Catalog(dv.path), Producer()
    worker._handle_item(ITEM, "movie", catalog, producer, "stube.catalog.item.transcoded",  # type: ignore[arg-type]
                        tmp_path / "packages", EncodeSettings())
    assert [status for status, _ in catalog.steps] == ["in_progress", "not_applicable"]
    assert len(producer.produced) == 1
