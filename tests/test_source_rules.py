"""A source the rules won't package as it is — Dolby Vision whose base
layer no other device plays, profile 5 on a host without a Vulkan device
— through the worker, with a fake probe: the step fails with the reason,
nothing is encoded or handed off (an older run's handoff is removed too),
and no event goes, so the title stays unpackaged and its original is
never retired. With a Vulkan device, profile 5 is encoded from its
reshaped HDR10 picture, once its first frames are seen to carry their
RPU. Real probes of such files are in test_encode_real.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from structlog.testing import capture_logs

from transcoder import extras, worker
from transcoder.decision import VulkanDevice, parse_ladder
from transcoder.ffmpeg import EncodeSettings, RungResult, TranscodeError
from transcoder.katalog import ClaimedExtra, ClaimedItem

ITEM = "0f1e2d3c-0000-4000-8000-000000000001"
EXTRA = "1b5c2a8e-0000-4000-8000-0000000000e1"
PARENT = "ea886f9b-0d06-4f0f-babb-d2a1162f9b01"
REFUSAL = {
    7: "Dolby Vision profile 7 needs a tone-mapping encode; kept the original",
    5: "Dolby Vision profile 5 is reshaped to HDR10 on a Vulkan device, and this host has "
       "none (no Vulkan preflight was run); kept the original",
}


def _dv_probe(profile: int, compat: int) -> dict:
    """ffprobe's payload for a UHD Main 10 stream with a Dolby Vision
    configuration record (transcoder.ffmpeg.ffprobe's shape): PQ, but for
    profile 5, whose colour tags say nothing but its full range."""
    tags = ({"color_range": "pc"} if profile == 5 else {"color_transfer": "smpte2084"})
    video = {"index": 0, "codec_type": "video", "codec_name": "hevc", "profile": "Main 10",
             "codec_tag_string": "dvh1" if profile == 5 else "hvc1", "width": 3840,
             "height": 2160, "pix_fmt": "yuv420p10le", "avg_frame_rate": "24000/1001",
             **tags,
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
    assert steps.steps == [("in_progress", {}), ("failed", {"error": REFUSAL[profile]})]
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
        ("in_progress", None), ("failed", REFUSAL[5])]
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


# ------------------------------------------- Dolby Vision 5, reshaped
VULKAN = VulkanDevice(index="0", name="NVIDIA GeForce RTX 3090 (discrete)", loader="1.3.204",
                      error="")
TRIGGER = {"extraId": EXTRA, "parentId": PARENT, "type": "extra", "kind": "trailer",
           "step": "transcode", "status": "queued", "source": "api"}


@pytest.fixture
def dv5(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A Dolby Vision 5 source: the worker's probe of it, its RPU check
    answering `dv5.rpu` (frames decoded, frames with an RPU; or an error
    to raise), and the encodes it runs, kept in `dv5.commands`."""
    source = tmp_path / "dv5.mkv"
    source.write_bytes(b"\0")
    dv5 = SimpleNamespace(path=source, rpu=(8, 8), commands=[])

    def rpu_check(path: Path, video_index: int | None) -> tuple[int, int]:
        assert (path, video_index) == (source, 0)
        if isinstance(dv5.rpu, Exception):
            raise dv5.rpu
        return dv5.rpu

    def encode(args: list[str], outputs: list, *, log_label: str):
        dv5.commands.append(args)
        results = []
        for out in outputs:
            out.final.parent.mkdir(parents=True, exist_ok=True)
            out.final.write_bytes(b"\0" * 1000)
            results.append(RungResult(rung=out.rung, path=out.final, size_bytes=1000,
                                      width=out.rung.width, height=out.rung.height))
        return results, 0.1

    monkeypatch.setattr(worker, "ffprobe", lambda _p: _dv_probe(5, 0))
    monkeypatch.setattr(worker, "dovi_metadata_frames", rpu_check)
    monkeypatch.setattr(worker, "run_encode", encode)
    return dv5


def _dv5_item(path: Path) -> ClaimedItem:
    return ClaimedItem(id=ITEM, type="movie", title="DV", year=None, duration_ms=None,
                       path=str(path))


@pytest.mark.parametrize("ladder", ["", "source:hevc"])
def test_with_a_vulkan_device_dolby_vision_5_is_encoded_as_hdr10(
    dv5, tmp_path: Path, ladder: str,
) -> None:
    inbox, steps = tmp_path / "_inbox" / ITEM, Steps()
    settings = EncodeSettings(ladder=tuple(parse_ladder(ladder)), vulkan=VULKAN)
    with capture_logs() as logs:
        ok = worker._process_one(_dv5_item(dv5.path), steps, inbox, settings)
    assert ok and [status for status, _ in steps.steps] == ["in_progress", "done"]
    details = steps.steps[-1][1]["details"]
    assert "res=3840x2160 reshape=dolby_vision_5:hdr10 maxrate=14Mbps" in details
    assert "ladder=v0:hevc_nvenc:3840x2160 " in details
    [args] = dv5.commands
    assert args[args.index("-init_hw_device") + 1] == "vulkan=vk:0"
    assert "libplacebo=apply_dolbyvision=1" in args[args.index("-filter_complex") + 1]
    # The handoff says HDR, of the source and of its encode.
    contract = json.loads((inbox / "renditions.json").read_text())
    assert (contract["source"]["hdr"], [v["hdr"] for v in contract["video"]]) == (True, [True])
    [line] = [e for e in logs if e["event"] == "transcoder.item.dolby_vision"]
    assert {k: line[k] for k in ("profile", "to", "rpu_frames", "vulkan_device",
                                 "vulkan_loader")} == {
        "profile": 5, "to": "hdr10", "rpu_frames": 8,
        "vulkan_device": "NVIDIA GeForce RTX 3090 (discrete)", "vulkan_loader": "1.3.204"}


@pytest.mark.parametrize(("rpu", "said"), [((8, 0), "0 of the first 8"),
                                           ((8, 7), "7 of the first 8"),
                                           ((0, 0), "0 of the first 0")])
def test_dolby_vision_5_whose_frames_lack_their_rpu_is_not_encoded(
    dv5, tmp_path: Path, rpu: tuple[int, int], said: str,
) -> None:
    # libplacebo would leave such a frame as it is, in the wrong colours.
    dv5.rpu = rpu
    inbox, steps = tmp_path / "_inbox" / ITEM, Steps()
    inbox.mkdir(parents=True)
    (inbox / "prepared.mkv").write_bytes(b"an older run's")
    ok = worker._process_one(_dv5_item(dv5.path), steps, inbox, EncodeSettings(vulkan=VULKAN))
    assert ok is False
    assert steps.steps == [("in_progress", {}), ("failed", {"error": (
        f"Dolby Vision profile 5: {said} decoded frames carry the RPU that libplacebo "
        "reshapes them with; kept the original")})]
    assert dv5.commands == [] and not inbox.exists()


def test_a_failed_rpu_check_fails_the_step(dv5, tmp_path: Path) -> None:
    dv5.rpu = TranscodeError("ffprobe (Dolby Vision metadata) exited 1: Invalid data found")
    inbox, steps = tmp_path / "_inbox" / ITEM, Steps()
    ok = worker._process_one(_dv5_item(dv5.path), steps, inbox, EncodeSettings(vulkan=VULKAN))
    assert ok is False
    assert steps.steps[-1] == ("failed", {
        "error": "ffprobe: ffprobe (Dolby Vision metadata) exited 1: Invalid data found"})
    assert dv5.commands == [] and not inbox.exists()


def test_a_reshaped_item_passes_the_chain_on(dv5, tmp_path: Path) -> None:
    catalog, producer = Catalog(dv5.path), Producer()
    worker._handle_item(ITEM, "movie", catalog, producer, "stube.catalog.item.transcoded",  # type: ignore[arg-type]
                        tmp_path / "packages", EncodeSettings(vulkan=VULKAN))
    assert [status for status, _ in catalog.steps] == ["in_progress", "done"]
    assert len(producer.produced) == 1


def test_a_dolby_vision_5_extra_is_reshaped_then_tone_mapped(dv5, tmp_path: Path) -> None:
    catalog, producer = Catalog(dv5.path), Producer()
    extras._handle_extra(EXTRA, TRIGGER, catalog, producer,  # type: ignore[arg-type]
                         "stube.catalog.extra.transcoded", tmp_path / "packages",
                         EncodeSettings(ladder=tuple(parse_ladder("720p:h264,480p:h264")),
                                        vulkan=VULKAN))
    assert [status for status, _ in catalog.steps] == ["in_progress", "done"]
    [args] = dv5.commands
    graph = args[args.index("-filter_complex") + 1]
    assert graph.startswith("[0:0]format=yuv420p10le,hwupload,libplacebo=apply_dolbyvision=1")
    assert graph.count("libplacebo") == 1 and graph.count("tonemap=tonemap=hable") == 2
    assert len(producer.produced) == 1
