"""Ladder parsing + rendition planning — pure logic, no ffmpeg needed."""

from __future__ import annotations

import pytest

from transcoder.decision import (
    CPU_ENCODERS,
    NVENC_ENCODERS,
    Encoders,
    LadderError,
    RungSpec,
    box_for_height,
    fit_within,
    parse_ladder,
    plan_renditions,
)


def _probe(codec: str, w: int, h: int, **extra: object) -> dict:
    video = {"codec_name": codec, "width": w, "height": h, "pix_fmt": "yuv420p",
             "avg_frame_rate": "24000/1001", **extra}
    return {"video": video, "audio": [], "subtitles": [], "duration_ms": 10_000}


# ------------------------------------------------------------- parsing
def test_empty_ladder_is_the_single_source_rung() -> None:
    for spec in ("", None, "  ", ","):
        assert parse_ladder(spec) == [RungSpec("source", None, "hevc", None)]


def test_parse_ladder_defaults_and_overrides() -> None:
    rungs = parse_ladder("source, 720p, 480p:hevc, 360p:h264:800k, 1080p:6M")
    assert rungs == [
        RungSpec("source", None, "hevc", None),
        RungSpec("720p", 720, "h264", None),
        RungSpec("480p", 480, "hevc", None),
        RungSpec("360p", 360, "h264", 800_000),
        RungSpec("1080p", 1080, "h264", 6_000_000),
    ]


@pytest.mark.parametrize("spec", ["720", "hd", "720p:vp9", "source:fast", "99p", "9999p"])
def test_parse_ladder_rejects_garbage(spec: str) -> None:
    with pytest.raises(LadderError):
        parse_ladder(spec)


# --------------------------------------------------------------- sizing
def test_box_for_height() -> None:
    assert box_for_height(720) == (1280, 720)
    assert box_for_height(480) == (854, 480)
    assert box_for_height(1080) == (1920, 1080)


@pytest.mark.parametrize(
    ("src", "box", "expected"),
    [
        # Measured with ffmpeg 7.1 and 8.1 (scale ... force_original_aspect_ratio
        # =decrease:force_divisible_by=2) — the planner must agree exactly.
        ((1918, 802), (854, 480), (854, 358)),
        ((1440, 1080), (1280, 720), (960, 720)),
        ((720, 576), (854, 480), (600, 480)),
        ((1998, 1080), (854, 480), (854, 462)),
        ((3840, 1606), (1920, 1080), (1920, 804)),
        ((1280, 534), (854, 480), (854, 356)),
        ((3840, 1606), (1280, 720), (1280, 536)),
    ],
)
def test_fit_within_matches_ffmpeg(src, box, expected) -> None:
    assert fit_within(*src, *box) == expected


# -------------------------------------------------------------- planning
def test_default_hevc_source_is_a_copy() -> None:
    plan = plan_renditions(_probe("hevc", 3840, 2160), parse_ladder(""), NVENC_ENCODERS)
    assert plan.all_copy
    assert plan.keyframes == "none"
    assert plan.rungs[0].reason == "source_already_hevc:hevc"
    assert plan.rungs[0].file is None


def test_default_h264_on_gpu_encodes_hevc_nvenc_like_before() -> None:
    plan = plan_renditions(_probe("h264", 1920, 1080), parse_ladder(""), NVENC_ENCODERS)
    [v0] = plan.rungs
    assert (v0.mode, v0.encoder, v0.codec) == ("encode", "hevc_nvenc", "hevc")
    assert (v0.width, v0.height, v0.scaled) == (1920, 1080, False)
    assert v0.maxrate_bps == 8_000_000
    assert v0.file == "prepared.mkv"
    assert plan.keyframes == "interval"


def test_default_uhd_h264_uses_the_uhd_cap() -> None:
    plan = plan_renditions(
        _probe("h264", 3840, 2160), parse_ladder(""), NVENC_ENCODERS, nvenc_caps_mbps=(8, 14),
    )
    assert plan.rungs[0].maxrate_bps == 14_000_000


def test_cpu_passes_browser_friendly_h264_through() -> None:
    plan = plan_renditions(_probe("h264", 1920, 1080), parse_ladder(""), CPU_ENCODERS)
    assert plan.all_copy
    assert plan.rungs[0].codec == "h264"
    assert plan.rungs[0].reason == "cpu_passthrough_h264"


def test_cpu_reencodes_hi10p_h264_with_x265() -> None:
    probe = _probe("h264", 1920, 1080, pix_fmt="yuv420p10le", profile="High 10")
    plan = plan_renditions(probe, parse_ladder(""), CPU_ENCODERS)
    assert plan.rungs[0].encoder == "libx265"


def test_cpu_encodes_exotic_sources_with_x265() -> None:
    plan = plan_renditions(_probe("mpeg2video", 720, 576), parse_ladder(""), CPU_ENCODERS)
    [v0] = plan.rungs
    assert (v0.mode, v0.encoder) == ("encode", "libx265")


def test_mixed_host_picks_per_codec() -> None:
    enc = Encoders(hevc="libx265", h264="h264_nvenc")
    assert enc.backend == "mixed"
    plan = plan_renditions(_probe("hevc", 1920, 1080), parse_ladder("source,720p"), enc)
    assert [r.encoder for r in plan.rungs] == ["copy", "h264_nvenc"]


def test_ladder_on_hevc_source_copies_top_and_aligns_to_source() -> None:
    plan = plan_renditions(
        _probe("hevc", 1920, 1080), parse_ladder("source,720p,480p"), NVENC_ENCODERS,
    )
    assert [(r.id, r.mode, r.encoder, r.width, r.height) for r in plan.rungs] == [
        ("v0", "copy", "copy", 1920, 1080),
        ("v1", "encode", "h264_nvenc", 1280, 720),
        ("v2", "encode", "h264_nvenc", 854, 480),
    ]
    assert plan.keyframes == "source"
    assert [r.file for r in plan.rungs] == [None, "v1.mkv", "v2.mkv"]
    assert plan.rungs[1].maxrate_bps == 3_000_000


def test_ladder_never_upscales_and_dedupes() -> None:
    # 720p source: the 1080p rung collapses to source size (H.264, kept —
    # it is the universally decodable copy of the HEVC top rung); the
    # 720p rung is then a duplicate of it and dropped.
    plan = plan_renditions(
        _probe("hevc", 1280, 720), parse_ladder("source,1080p,720p,480p"), NVENC_ENCODERS,
    )
    assert [(r.codec, r.width, r.height, r.scaled) for r in plan.rungs] == [
        ("hevc", 1280, 720, False),
        ("h264", 1280, 720, False),
        ("h264", 854, 480, True),
    ]
    assert all(r.width <= 1280 and r.height <= 720 for r in plan.rungs)


def test_ladder_sorts_largest_first_whatever_the_order() -> None:
    plan = plan_renditions(
        _probe("h264", 1920, 1080), parse_ladder("480p,source,720p"), NVENC_ENCODERS,
    )
    assert [r.height for r in plan.rungs] == [1080, 720, 480]
    assert [r.id for r in plan.rungs] == ["v0", "v1", "v2"]
    assert plan.rungs[0].file == "prepared.mkv"


def test_ultrawide_rungs_fit_the_16x9_box() -> None:
    plan = plan_renditions(
        _probe("hevc", 3840, 1606), parse_ladder("source,1080p,720p"), NVENC_ENCODERS,
    )
    assert [(r.width, r.height) for r in plan.rungs] == [(3840, 1606), (1920, 804), (1280, 536)]


def test_hdr_source_tonemaps_h264_and_keeps_10bit_hevc() -> None:
    probe = _probe("hevc", 3840, 2160, pix_fmt="yuv420p10le", color_transfer="smpte2084")
    plan = plan_renditions(probe, parse_ladder("source,1080p:hevc,720p"), NVENC_ENCODERS)
    top, hevc_1080, h264_720 = plan.rungs
    assert top.mode == "copy" and not top.tonemap
    assert hevc_1080.ten_bit and not hevc_1080.tonemap
    assert h264_720.tonemap and not h264_720.ten_bit


def test_explicit_rung_maxrate_wins() -> None:
    plan = plan_renditions(
        _probe("hevc", 1920, 1080), parse_ladder("source,720p:h264:2500k"), NVENC_ENCODERS,
    )
    assert plan.rungs[1].maxrate_bps == 2_500_000


def test_source_h264_rung_on_h264_source_is_a_copy() -> None:
    plan = plan_renditions(_probe("h264", 1920, 1080), parse_ladder("source:h264"), NVENC_ENCODERS)
    assert plan.all_copy
    assert plan.rungs[0].reason == "source_already_h264"


def test_frame_rate_falls_back_to_r_frame_rate() -> None:
    probe = _probe("h264", 1920, 1080, avg_frame_rate="0/0", r_frame_rate="25/1")
    plan = plan_renditions(probe, parse_ladder(""), NVENC_ENCODERS)
    assert plan.source.fps == 25.0
    assert plan.source.frame_rate == "25/1"


def test_anamorphic_source_is_sized_from_its_display_aspect() -> None:
    # 16:9 PAL DVD: 720x576 stored, SAR 64:45 -> 1024x576 on screen.
    probe = _probe("mpeg2video", 720, 576, sample_aspect_ratio="64:45")
    plan = plan_renditions(probe, parse_ladder("source,720p,480p"), NVENC_ENCODERS)
    # 720p: the 1024x576 picture already fits -> collapses to source size
    # (stored 720x576, SAR kept); 480p: scaled to square-pixel 16:9.
    assert [(r.width, r.height, r.scaled) for r in plan.rungs] == [
        (720, 576, False), (720, 576, False), (854, 480, True),
    ]
    assert [r.codec for r in plan.rungs] == ["hevc", "h264", "h264"]


def test_square_pixel_sources_are_unaffected_by_sar_handling() -> None:
    probe = _probe("hevc", 1920, 1080, sample_aspect_ratio="1:1")
    plan = plan_renditions(probe, parse_ladder("source,480p"), NVENC_ENCODERS)
    assert (plan.rungs[1].width, plan.rungs[1].height) == (854, 480)
