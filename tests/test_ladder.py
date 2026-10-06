"""Ladder parsing + rendition planning — pure logic, no ffmpeg needed."""

from __future__ import annotations

import pytest

from transcoder.config import DEFAULT_EXTRA_LADDER
from transcoder.decision import (
    CPU_ENCODERS,
    NVENC_ENCODERS,
    Encoders,
    LadderError,
    RungSpec,
    box_for_height,
    fit_within,
    parse_ladder,
    pix_fmt_layout,
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
        RungSpec("480p", 480, "hevc", None, codec_named=True),
        RungSpec("360p", 360, "h264", 800_000, codec_named=True),
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


# ------------------------------------------------------------- HEVC only
# `source:hevc`, the HEVC-only ladder (LADDER and EXTRA_LADDER alike): one
# rendition at the source's own size — the source's video copied when it
# is HEVC, else ONE HEVC encode, NVENC or else libx265. Never an H.264
# rung, and never the CPU rule's H.264 pass-through.
HEVC_ONLY = parse_ladder("source:hevc")

# (encoders, the HEVC encoder they encode with): a GPU host, a CPU host,
# and a host whose NVENC opens for H.264 only.
HOSTS = [(NVENC_ENCODERS, "hevc_nvenc"), (CPU_ENCODERS, "libx265"),
         (Encoders(hevc="libx265", h264="h264_nvenc"), "libx265")]


def test_hevc_only_is_the_source_rung_with_its_codec_named() -> None:
    assert HEVC_ONLY == [RungSpec("source", None, "hevc", None, codec_named=True)]
    assert parse_ladder(" Source:HEVC ") == HEVC_ONLY
    # The default is the same rung, its codec left to the default.
    assert parse_ladder("") == parse_ladder("source") == [RungSpec("source", None, "hevc", None)]


@pytest.mark.parametrize("encoders", [e for e, _ in HOSTS])
@pytest.mark.parametrize(("codec", "extra"), [
    ("hevc", {}),
    ("h265", {}),
    ("hevc", {"pix_fmt": "yuv420p10le", "profile": "Main 10"}),
    ("hevc", {"pix_fmt": "yuv420p10le", "profile": "Main 10", "color_transfer": "smpte2084"}),
    ("hevc", {"pix_fmt": "yuv420p10le", "profile": "Main 10", "color_transfer": "arib-std-b67"}),
])
def test_hevc_only_copies_an_hevc_source(encoders: Encoders, codec: str, extra: dict) -> None:
    plan = plan_renditions(_probe(codec, 3840, 1606, **extra), HEVC_ONLY, encoders)
    [v0] = plan.rungs
    assert (v0.id, v0.mode, v0.encoder, v0.codec, v0.width, v0.height, v0.file) == (
        "v0", "copy", "copy", "hevc", 3840, 1606, None)
    assert v0.reason == f"source_already_hevc:{codec}"
    # Nothing to encode: the packager packages the original's video.
    assert plan.all_copy and plan.keyframes == "none"
    assert not (v0.tonemap or v0.ten_bit)


@pytest.mark.parametrize(("encoders", "hevc"), HOSTS)
@pytest.mark.parametrize("extra", [
    {},                                                   # browser-friendly
    {"profile": "High"},                                  # browser-friendly
    {"profile": "Constrained Baseline"},                  # browser-friendly
    {"pix_fmt": "yuv420p10le", "profile": "High 10"},     # Hi10P
    {"pix_fmt": "yuv422p", "profile": "High 4:2:2"},
])
def test_hevc_only_encodes_h264_once_to_hevc_at_its_size(
    encoders: Encoders, hevc: str, extra: dict,
) -> None:
    plan = plan_renditions(_probe("h264", 1920, 800, **extra), HEVC_ONLY, encoders)
    [v0] = plan.rungs
    assert (v0.mode, v0.encoder, v0.codec, v0.width, v0.height, v0.scaled, v0.file) == (
        "encode", hevc, "hevc", 1920, 800, False, "prepared.mkv")
    assert v0.reason == "h264_to_hevc"
    assert v0.maxrate_bps == 8_000_000  # the source rung's NVENC_MAXRATE_1080P_MBPS
    assert plan.keyframes == "interval"


def test_the_cpu_rule_stays_for_a_source_rung_whose_codec_is_the_default() -> None:
    # An install that never sets LADDER, or sets "source", keeps the CPU
    # behaviour it has: browser-friendly H.264 passes through as H.264.
    probe = _probe("h264", 1920, 1080)
    for spec in ("", "source"):
        [v0] = plan_renditions(probe, parse_ladder(spec), CPU_ENCODERS).rungs
        assert (v0.mode, v0.codec, v0.reason) == ("copy", "h264", "cpu_passthrough_h264")
    [v0] = plan_renditions(probe, HEVC_ONLY, CPU_ENCODERS).rungs
    assert (v0.mode, v0.codec, v0.encoder, v0.reason) == (
        "encode", "hevc", "libx265", "h264_to_hevc")


@pytest.mark.parametrize("codec", ["h264", "hevc", "vp9", "av1", "mpeg2video"])
def test_with_nvenc_hevc_only_plans_as_the_default_ladder(codec: str) -> None:
    # The CPU rule never applies with NVENC, so on a GPU host the two
    # spellings are the same plan.
    probe = _probe(codec, 1920, 1080)
    assert plan_renditions(probe, HEVC_ONLY, NVENC_ENCODERS) == plan_renditions(
        probe, parse_ladder(""), NVENC_ENCODERS)


@pytest.mark.parametrize(("encoders", "hevc"), HOSTS)
@pytest.mark.parametrize("codec", ["vp9", "av1", "vp8", "mpeg2video", "vc1", "mpeg4", "prores"])
def test_hevc_only_encodes_any_other_codec_once_to_hevc(
    encoders: Encoders, hevc: str, codec: str,
) -> None:
    plan = plan_renditions(_probe(codec, 1280, 720), HEVC_ONLY, encoders)
    [v0] = plan.rungs
    assert (v0.mode, v0.encoder, v0.codec, v0.width, v0.height, v0.scaled) == (
        "encode", hevc, "hevc", 1280, 720, False)
    assert v0.reason == f"{codec}_to_hevc"


def test_hevc_only_uhd_encode_takes_the_uhd_cap() -> None:
    [v0] = plan_renditions(_probe("av1", 3840, 2160), HEVC_ONLY, NVENC_ENCODERS).rungs
    assert (v0.encoder, v0.width, v0.height, v0.maxrate_bps) == (
        "hevc_nvenc", 3840, 2160, 14_000_000)


@pytest.mark.parametrize(("encoders", "hevc"), HOSTS)
@pytest.mark.parametrize(("codec", "extra", "hdr", "main10"), [
    # HDR (PQ / HLG): Main 10 HDR, never tone-mapped (no H.264 rung).
    ("av1", {"pix_fmt": "yuv420p10le", "color_transfer": "smpte2084"}, True, True),
    ("vp9", {"pix_fmt": "yuv420p10le", "profile": "Profile 2",
             "color_transfer": "arib-std-b67"}, True, True),
    ("h264", {"pix_fmt": "yuv420p10le", "profile": "High 10",
              "color_transfer": "smpte2084"}, True, True),
    # SDR above 8 bits: Main 10 SDR, the precision kept.
    ("av1", {"pix_fmt": "yuv420p10le", "color_transfer": "bt709"}, False, True),
    ("vp9", {"pix_fmt": "yuv420p10le", "profile": "Profile 2"}, False, True),
    ("vp9", {"pix_fmt": "yuv420p12le", "profile": "Profile 2"}, False, True),
    ("h264", {"pix_fmt": "yuv420p10le", "profile": "High 10"}, False, True),
    ("h264", {"pix_fmt": "yuv422p10le", "profile": "High 4:2:2"}, False, True),
    ("prores", {"pix_fmt": "yuv422p10le", "profile": "HQ"}, False, True),
    ("ffv1", {"pix_fmt": "yuv444p16le"}, False, True),
    # 8-bit: Main.
    ("av1", {}, False, False),
    ("mpeg2video", {}, False, False),
    ("h264", {"pix_fmt": "yuv444p", "profile": "High 4:4:4 Predictive"}, False, False),
])
def test_hevc_only_keeps_the_bit_depth(
    encoders: Encoders, hevc: str, codec: str, extra: dict, hdr: bool, main10: bool,
) -> None:
    # Originals are deleted once packaged: a source above 8 bits is encoded
    # Main 10, whether HDR or SDR; an 8-bit one Main. Nothing is tone-mapped.
    plan = plan_renditions(_probe(codec, 3840, 2160, **extra), HEVC_ONLY, encoders)
    [v0] = plan.rungs
    assert (v0.mode, v0.encoder, v0.codec) == ("encode", hevc, "hevc")
    assert (plan.source.hdr, v0.ten_bit, v0.tonemap) == (hdr, main10, False)


@pytest.mark.parametrize("encoders", [NVENC_ENCODERS, CPU_ENCODERS])
def test_every_hevc_rung_of_a_10bit_source_is_main10_and_h264_rungs_8bit(
    encoders: Encoders,
) -> None:
    # A 10-bit SDR source on a ladder: its HEVC rungs, scaled or not, are
    # Main 10; its H.264 rung is 8-bit, and not tone-mapped (it is SDR).
    probe = _probe("av1", 1920, 1080, pix_fmt="yuv420p10le")
    plan = plan_renditions(probe, parse_ladder("source,720p:hevc,480p"), encoders)
    assert [(r.codec, r.height, r.ten_bit, r.tonemap) for r in plan.rungs] == [
        ("hevc", 1080, True, False), ("hevc", 720, True, False), ("h264", 480, False, False)]


def test_the_default_ladder_keeps_the_bit_depth_too() -> None:
    # Not only source:hevc: the single default rung of a Hi10P source.
    [v0] = plan_renditions(_probe("h264", 1920, 1080, pix_fmt="yuv420p10le",
                                  profile="High 10"), parse_ladder(""), NVENC_ENCODERS).rungs
    assert (v0.encoder, v0.ten_bit) == ("hevc_nvenc", True)


@pytest.mark.parametrize(("pix_fmt", "layout"), [
    ("yuv420p", ("420", 8)), ("yuvj420p", ("420", 8)), ("yuv420p10le", ("420", 10)),
    ("yuv420p12be", ("420", 12)), ("yuv422p10le", ("422", 10)), ("yuv444p", ("444", 8)),
    ("yuv444p16le", ("444", 16)), ("yuva420p10le", ("420", 10)), ("yuv440p", ("440", 8)),
    ("yuv411p", ("411", 8)), ("p010le", ("420", 10)), ("p016le", ("420", 16)),
    ("p210le", ("422", 10)), ("p410le", ("444", 10)), ("nv12", ("420", 8)),
    ("nv16", ("422", 8)), ("nv20le", ("422", 10)), ("nv24", ("444", 8)),
    ("y210le", ("422", 10)), ("gray", ("400", 8)), ("gray10le", ("400", 10)),
    ("gbrp", ("444", 8)), ("gbrp12le", ("444", 12)), ("YUV420P10LE", ("420", 10)),
    ("", (None, None)), ("rgb24", (None, None)), ("bayer_rggb8", (None, None)),
])
def test_pix_fmt_layout(pix_fmt: str, layout: tuple) -> None:
    assert pix_fmt_layout(pix_fmt) == layout


@pytest.mark.parametrize(("extra", "depth"), [
    ({"pix_fmt": "yuv420p10le"}, 10),
    ({"pix_fmt": "yuv420p10le", "bits_per_raw_sample": "8"}, 10),  # the pixel format wins
    ({"pix_fmt": "", "bits_per_raw_sample": "10"}, 10),
    ({"pix_fmt": "weird", "bits_per_raw_sample": 12}, 12),
    ({"pix_fmt": "", "bits_per_raw_sample": "N/A"}, 8),
    ({"pix_fmt": ""}, 8),
])
def test_bit_depth_from_the_probe(extra: dict, depth: int) -> None:
    assert plan_renditions(_probe("h264", 640, 360, **extra), HEVC_ONLY,
                           NVENC_ENCODERS).source.bit_depth == depth


def test_hevc_only_keeps_the_stored_size_of_an_anamorphic_source() -> None:
    # 16:9 PAL DVD (720x576, SAR 64:45): the one rendition is the source's
    # own size, its SAR kept, never scaled to square pixels.
    probe = _probe("mpeg2video", 720, 576, sample_aspect_ratio="64:45")
    [v0] = plan_renditions(probe, HEVC_ONLY, CPU_ENCODERS).rungs
    assert (v0.mode, v0.encoder, v0.width, v0.height, v0.scaled) == (
        "encode", "libx265", 720, 576, False)


# ------------------------------------------------------ the HEVC copy rule
# Copied only when every HEVC decoder plays it: Main or Main 10, 4:2:0,
# at most 10 bits. The rest is re-encoded to 4:2:0 at its own size —
# Main 10 above 8 bits, Main for 8 — on every ladder.
@pytest.mark.parametrize("encoders", [e for e, _ in HOSTS])
@pytest.mark.parametrize("spec", ["", "source:hevc", "source,720p"])
@pytest.mark.parametrize("extra", [
    {"profile": "Main"},
    {"profile": "Main 10", "pix_fmt": "yuv420p10le"},
    {"profile": "Main 10", "pix_fmt": "yuv420p10le", "color_transfer": "smpte2084"},
    {"profile": "", "pix_fmt": "yuv420p"},
    # The profile alone vouches for 4:2:0 and its bit depth.
    {"profile": "Main", "pix_fmt": ""},
    {"profile": "Main 10", "pix_fmt": ""},
])
def test_hevc_main_and_main10_420_are_copied(encoders: Encoders, spec: str, extra: dict) -> None:
    plan = plan_renditions(_probe("hevc", 1920, 1080, **extra), parse_ladder(spec), encoders)
    v0 = plan.rungs[0]
    assert (v0.mode, v0.codec, v0.width, v0.height) == ("copy", "hevc", 1920, 1080)
    assert v0.reason == "source_already_hevc:hevc"


@pytest.mark.parametrize(("encoders", "hevc"), HOSTS)
@pytest.mark.parametrize("spec", ["", "source:hevc"])
@pytest.mark.parametrize(("extra", "why", "main10"), [
    ({"profile": "Rext", "pix_fmt": "yuv422p10le"}, "4:2:2", True),
    ({"profile": "Rext", "pix_fmt": "yuv422p"}, "4:2:2", False),
    ({"profile": "Rext", "pix_fmt": "yuv444p10le"}, "4:4:4", True),
    ({"profile": "Rext", "pix_fmt": "yuv444p"}, "4:4:4", False),
    ({"profile": "Rext", "pix_fmt": "gbrp10le"}, "4:4:4", True),
    ({"profile": "Rext", "pix_fmt": "yuv420p12le"}, "12-bit", True),
    ({"profile": "Rext", "pix_fmt": "yuv422p12le"}, "4:2:2", True),
    ({"profile": "Rext", "pix_fmt": "gray10le"}, "4:0:0", True),
    # 4:2:0 10-bit, but a Rext stream (an intra profile, say): not Main 10.
    ({"profile": "Rext", "pix_fmt": "yuv420p10le"}, "profile=rext", True),
    ({"profile": "Scc", "pix_fmt": "yuv420p"}, "profile=scc", False),
    ({"profile": "Main Still Picture", "pix_fmt": "yuv420p"},
     "profile=main_still_picture", False),
    ({"profile": "", "pix_fmt": ""}, "pix_fmt=unknown", False),
])
def test_other_hevc_is_reencoded_to_420_at_its_size(
    encoders: Encoders, hevc: str, spec: str, extra: dict, why: str, main10: bool,
) -> None:
    plan = plan_renditions(_probe("hevc", 3840, 2160, **extra), parse_ladder(spec), encoders)
    [v0] = plan.rungs
    assert (v0.mode, v0.encoder, v0.codec, v0.width, v0.height, v0.scaled, v0.file) == (
        "encode", hevc, "hevc", 3840, 2160, False, "prepared.mkv")
    assert v0.reason == f"hevc_not_copyable:{why}"
    assert (v0.ten_bit, v0.tonemap) == (main10, False)
    assert v0.maxrate_bps == 14_000_000
    assert plan.keyframes == "interval"


def test_a_reencoded_hevc_source_is_no_copy_on_its_ladder_either() -> None:
    # Nothing on the ladder is a copy, so the rungs' keyframes are the
    # interval, not the source's.
    probe = _probe("hevc", 1920, 1080, profile="Rext", pix_fmt="yuv422p10le")
    plan = plan_renditions(probe, parse_ladder("source,720p"), NVENC_ENCODERS)
    assert [(r.mode, r.encoder, r.codec, r.height, r.ten_bit) for r in plan.rungs] == [
        ("encode", "hevc_nvenc", "hevc", 1080, True), ("encode", "h264_nvenc", "h264", 720, False)]
    assert plan.keyframes == "interval"


def test_a_named_hevc_rung_at_the_sources_size_is_hevc_on_a_cpu_too() -> None:
    # 720p:hevc collapses to a 720p source's size; its codec is named, so
    # it is no pass-through either.
    plan = plan_renditions(_probe("h264", 1280, 720), parse_ladder("720p:hevc,480p"),
                           CPU_ENCODERS)
    assert [(r.mode, r.encoder, r.codec, r.width, r.height) for r in plan.rungs] == [
        ("encode", "libx265", "hevc", 1280, 720), ("encode", "libx264", "h264", 854, 480)]


# --------------------------------------------------------------- extras
# The extras' ladder (EXTRA_LADDER's default): no source rung, two H.264
# rungs, so whatever the source, the package plays on every device.
EXTRAS = parse_ladder(DEFAULT_EXTRA_LADDER)


def test_extras_ladder_has_no_source_rung() -> None:
    assert EXTRAS == [RungSpec("720p", 720, "h264", None, codec_named=True),
                      RungSpec("480p", 480, "h264", None, codec_named=True)]


@pytest.mark.parametrize(("encoders", "h264"), [(NVENC_ENCODERS, "h264_nvenc"),
                                                (CPU_ENCODERS, "libx264")])
def test_extras_1080p_h264_encodes_both_rungs(encoders: Encoders, h264: str) -> None:
    plan = plan_renditions(_probe("h264", 1920, 1080), EXTRAS, encoders)
    assert [(r.id, r.mode, r.encoder, r.codec, r.width, r.height, r.file) for r in plan.rungs] == [
        ("v0", "encode", h264, "h264", 1280, 720, "prepared.mkv"),
        ("v1", "encode", h264, "h264", 854, 480, "v1.mkv"),
    ]
    assert [r.maxrate_bps for r in plan.rungs] == [3_000_000, 1_500_000]
    assert plan.keyframes == "interval"


def test_extras_720p_h264_copies_the_top_rung_and_encodes_480p() -> None:
    plan = plan_renditions(_probe("h264", 1280, 720), EXTRAS, NVENC_ENCODERS)
    assert [(r.id, r.mode, r.encoder, r.width, r.height, r.file) for r in plan.rungs] == [
        ("v0", "copy", "copy", 1280, 720, None),
        ("v1", "encode", "h264_nvenc", 854, 480, "v1.mkv"),
    ]
    assert plan.rungs[0].reason == "source_already_h264"
    # The 480p rung's keyframes land on the copied source's.
    assert plan.keyframes == "source"


def test_extras_480p_vp9_is_one_h264_encode_at_its_own_size() -> None:
    # Both boxes hold the 854x480 source, so both rungs are its own size:
    # one H.264 encode (VP9 is no copy), the second rung a duplicate.
    plan = plan_renditions(_probe("vp9", 854, 480), EXTRAS, NVENC_ENCODERS)
    [v0] = plan.rungs
    assert (v0.mode, v0.encoder, v0.codec, v0.width, v0.height, v0.scaled) == (
        "encode", "h264_nvenc", "h264", 854, 480, False)
    assert v0.reason == "vp9_to_h264"
    assert v0.file == "prepared.mkv"
    assert plan.keyframes == "interval"


def test_extras_4k_hevc_encodes_both_rungs_to_h264() -> None:
    plan = plan_renditions(_probe("hevc", 3840, 2160), EXTRAS, NVENC_ENCODERS)
    assert [(r.mode, r.encoder, r.width, r.height, r.scaled) for r in plan.rungs] == [
        ("encode", "h264_nvenc", 1280, 720, True),
        ("encode", "h264_nvenc", 854, 480, True),
    ]
    assert not any(r.tonemap for r in plan.rungs)
    assert plan.keyframes == "interval"


def test_extras_4k_hdr_hevc_is_tonemapped_on_every_rung() -> None:
    probe = _probe("hevc", 3840, 2160, pix_fmt="yuv420p10le", color_transfer="smpte2084")
    plan = plan_renditions(probe, EXTRAS, NVENC_ENCODERS)
    assert [(r.codec, r.tonemap, r.ten_bit) for r in plan.rungs] == [
        ("h264", True, False), ("h264", True, False)]


@pytest.mark.parametrize(("codec", "extra"), [
    ("vp9", {}), ("theora", {}), ("hevc", {}), ("av1", {}),
    ("h264", {"pix_fmt": "yuv420p10le", "profile": "High 10"}),
])
def test_extras_encode_every_source_that_is_not_browser_friendly_h264(
    codec: str, extra: dict,
) -> None:
    plan = plan_renditions(_probe(codec, 1280, 720, **extra), EXTRAS, NVENC_ENCODERS)
    assert [(r.mode, r.codec, r.width, r.height) for r in plan.rungs] == [
        ("encode", "h264", 1280, 720), ("encode", "h264", 854, 480)]


def test_extras_small_h264_source_needs_no_encode() -> None:
    # Both rungs are the 640x360 source's own size: a copy, nothing to do.
    plan = plan_renditions(_probe("h264", 640, 360), EXTRAS, CPU_ENCODERS)
    assert plan.all_copy
    assert [(r.width, r.height, r.reason) for r in plan.rungs] == [
        (640, 360, "source_already_h264")]


@pytest.mark.parametrize(("encoders", "hevc"), HOSTS)
@pytest.mark.parametrize(("codec", "size", "mode"), [
    ("h264", (1920, 1080), "encode"),
    ("h264", (1280, 720), "encode"),
    ("vp9", (854, 480), "encode"),
    ("hevc", (3840, 2160), "copy"),
    ("h264", (640, 360), "encode"),
])
def test_extras_hevc_only_is_one_hevc_rendition_at_the_sources_size(
    encoders: Encoders, hevc: str, codec: str, size: tuple[int, int], mode: str,
) -> None:
    # EXTRA_LADDER=source:hevc, on the sources of the default's table: no
    # 720p or 480p rung, one HEVC rendition each, the HEVC trailer copied.
    plan = plan_renditions(_probe(codec, *size), HEVC_ONLY, encoders)
    [v0] = plan.rungs
    assert (v0.mode, v0.codec, (v0.width, v0.height)) == (mode, "hevc", size)
    assert v0.encoder == (hevc if mode == "encode" else "copy")
