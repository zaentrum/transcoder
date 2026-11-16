"""Sources in fields: the verdict on idet's counts, the plan, the command
and the record (no ffmpeg run; tests/test_fields_real.py runs made clips)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from transcoder import ffmpeg as ff
from transcoder.config import Config
from transcoder.decision import (
    CPU_ENCODERS,
    DEFAULT_DEINTERLACE,
    FIELDS_MIN_DETERMINED,
    FIELDS_MIN_FRAMES,
    NVENC_ENCODERS,
    SCAN_DONE,
    SCAN_RELEASES,
    Fields,
    LadderError,
    SourceInfo,
    fields_by_mark,
    judge_fields,
    parse_ladder,
    plan_renditions,
)
from transcoder.ffmpeg import EncodeSettings, TranscodeError, build_encode_command
from transcoder.renditions import build_contract, scan_record
from transcoder.worker import _scan_details

SRC = Path("/media/title.mkv")
INBOX = Path("/packages/_inbox/item")


def _w(single: str, repeated: str | None = None) -> tuple[list[str], list[str]]:
    """A window as idet's letters: t b p u for its single-frame verdicts, n t
    b for its repeated fields (all neither when not given)."""
    words = {"t": "tff", "b": "bff", "p": "progressive", "u": "undetermined"}
    reps = {"n": "neither", "t": "top", "b": "bottom"}
    repeated = repeated if repeated is not None else "n" * len(single)
    return [words[c] for c in single], [reps[c] for c in repeated]


# ------------------------------------------------------------- the verdict
def test_true_interlacing_is_fields_in_the_order_its_mark_names() -> None:
    f = judge_fields("tb", [_w("u" + "t" * 199)] * 5)
    assert (f.found, f.by, f.order, f.order_by) == ("fields", "pictures", "tff", "mark")
    assert (f.frames, f.tff, f.bff, f.progressive, f.undetermined) == (1000, 995, 0, 0, 5)
    assert f.in_fields


def test_idet_overrules_a_mark_that_names_the_other_field_first() -> None:
    f = judge_fields("tb", [_w("b" * 200)] * 5)
    assert (f.found, f.order, f.order_by) == ("fields", "bff", "pictures")


def test_a_mark_idet_cannot_contradict_names_the_order() -> None:
    # As many TFF as BFF frames: idet names no order, the mark does.
    f = judge_fields("bb", [_w("tb" * 100)] * 5)
    assert (f.found, f.order, f.order_by) == ("fields", "bff", "mark")


def test_without_a_mark_or_an_order_from_idet_the_frames_flags_decide() -> None:
    f = judge_fields("", [_w("tb" * 100)] * 5)
    assert (f.found, f.order, f.order_by) == ("fields", "auto", "frames")
    f = judge_fields("unknown", [_w("t" * 200)] * 5)
    assert (f.found, f.order, f.order_by) == ("fields", "tff", "pictures")


def test_a_32_pulldown_is_telecine() -> None:
    # 3:2 as idet sees it: 4 of 5 frames combed, one top and one bottom field
    # repeated in every 5, at the same two places.
    f = judge_fields("tb", [_w("ptttt" * 40, "nntnb" * 40)] * 5)
    assert (f.found, f.by) == ("telecine", "pictures")
    assert (f.repeated_top, f.repeated_bottom) == (200, 200)
    assert (f.cadence, f.coverage) == (1.0, 1.0)


def test_the_cadence_may_move_from_one_window_to_the_next() -> None:
    # A cut moves the cadence: each window's own two places count.
    windows = [_w("ptttt" * 40, ("n" * k + "nntnb" * 40)[:200]) for k in range(5)]
    f = judge_fields("tb", windows)
    assert f.found == "telecine" and f.cadence == 1.0


def test_film_with_a_video_part_is_deinterlaced_not_inverse_telecined() -> None:
    # The cadence over 2 of 5 windows: a film with video-rate parts, whose
    # video fieldmatch and decimate would leave combed or drop.
    film, video = _w("ptttt" * 40, "nntnb" * 40), _w("t" * 200)
    f = judge_fields("tb", [film, film, video, video, video])
    assert (f.found, f.coverage) == ("fields", 0.4)


def test_repeated_fields_off_a_cadence_are_no_telecine() -> None:
    # As many repeats as 3:2 makes, but each cycle at other places of its 5.
    drifting = ("tnbnn" + "ntnbn" + "nntnb" + "bnntn" + "nbnnt") * 8
    f = judge_fields("tb", [_w("t" * 200, drifting)] * 5)
    assert f.cadence < 0.9
    assert f.found == "fields"


def test_one_kind_of_repeated_field_alone_is_no_telecine() -> None:
    f = judge_fields("tb", [_w("ptttt" * 40, "ntntn" * 40)] * 5)
    assert f.found == "fields"


def test_whole_frames_marked_as_fields_are_progressive() -> None:
    # A film transfer in an interlace-flagged stream: whole frames, a few
    # undetermined, none in fields.
    f = judge_fields("tb", [_w("u" + "p" * 199)] * 5)
    assert (f.found, f.by) == ("progressive", "pictures")
    assert not f.in_fields


def test_a_few_frames_idet_misreads_leave_a_film_progressive() -> None:
    # Below the share (5 % of the determined frames): a dark logo, a frame
    # of fine detail.
    f = judge_fields("tb", [_w("t" * 9 + "p" * 191)] + [_w("p" * 200)] * 4)
    assert f.found == "progressive"
    f = judge_fields("tb", [_w("t" * 40 + "p" * 160)] * 5)
    assert f.found == "fields"  # 20 % of its frames


def test_the_share_and_the_least_number_of_frames_in_fields() -> None:
    # FIELDS_MIN_FRAMES frames in fields of 100 determined: fields; one less:
    # progressive, though it is above 5 % of them.
    n = FIELDS_MIN_FRAMES
    assert judge_fields("tb", [_w("t" * n + "p" * (100 - n))]).found == "fields"
    assert judge_fields("tb", [_w("t" * (n - 1) + "p" * (101 - n))]).found == "progressive"
    # 5 % of 1000 determined frames is 50: 49 is progressive, 50 fields.
    assert judge_fields("tb", [_w("t" * 49 + "p" * 951)]).found == "progressive"
    assert judge_fields("tb", [_w("t" * 50 + "p" * 950)]).found == "fields"


def test_mixed_content_is_fields() -> None:
    f = judge_fields("tb", [_w("p" * 200), _w("p" * 100 + "t" * 100), _w("t" * 200)])
    assert f.found == "fields"


def test_a_still_picture_is_judged_by_its_mark() -> None:
    # idet determines too few frames (a still, black): the mark decides.
    few = FIELDS_MIN_DETERMINED - 1
    f = judge_fields("tb", [_w("u" * 200)] * 4 + [_w("t" * few + "u" * (200 - few))])
    assert (f.found, f.by, f.order, f.order_by) == ("fields", "mark", "tff", "mark")
    assert judge_fields("", [_w("u" * 200)]).found == "progressive"
    # Nothing read at all.
    assert judge_fields("bt", [([], [])]) == Fields(found="fields", by="mark", mark="bt",
                                                     order="bff", order_by="mark")


@pytest.mark.parametrize(("mark", "found", "order"), [
    ("tt", "fields", "tff"), ("tb", "fields", "tff"), ("bb", "fields", "bff"),
    ("bt", "fields", "bff"), ("progressive", "progressive", "auto"), ("", "progressive", "auto"),
    ("TB", "fields", "tff"),
])
def test_the_mark_alone(mark: str, found: str, order: str) -> None:
    f = fields_by_mark(mark)
    assert (f.found, f.by, f.order) == (found, "mark", order)


@pytest.mark.parametrize(("field_order", "height", "examined"), [
    ("tt", 576, True), ("tb", 1080, True), ("bt", 2160, True),
    ("", 576, True), ("unknown", 1080, True), ("", 1088, True), ("", 1440, False),
    ("", 2160, False), ("progressive", 576, False), ("progressive", 1080, False),
])
def test_which_sources_are_examined(field_order: str, height: int, examined: bool) -> None:
    src = SourceInfo.from_probe(_probe("mpeg2video", 720 if height <= 576 else 1920, height,
                                       field_order=field_order))
    assert src.fields_examined() is examined


@pytest.mark.parametrize(("codec", "examined"), [
    ("h264", True), ("vc1", True), ("mpeg4", True), ("dvvideo", True),
    ("hevc", False), ("vp9", False), ("av1", False),
])
def test_an_unmarked_source_is_examined_in_a_codec_of_woven_fields(codec: str,
                                                                     examined: bool) -> None:
    src = SourceInfo.from_probe(_probe(codec, 1920, 1080, field_order=""))
    assert src.fields_examined() is examined
    # Marked as fields, any codec is.
    assert SourceInfo.from_probe(_probe(codec, 1920, 1080, field_order="tt")).fields_examined()


def test_a_stored_mark_stands_in_for_the_streams() -> None:
    # A copy of the source lost its container's mark ("unknown"); the source
    # record's field order says what it was.
    src = SourceInfo.from_probe(_probe("hevc", 3840, 2160, field_order="progressive"))
    assert not src.fields_examined()
    assert src.fields_examined("tb")


# ---------------------------------------------------------------- the plan
def _probe(codec: str, w: int, h: int, rate: str = "25/1", **video: object) -> dict:
    v = {"codec_name": codec, "width": w, "height": h, "pix_fmt": "yuv420p",
         "avg_frame_rate": rate, "profile": {"hevc": "Main", "h264": "High"}.get(codec, ""),
         **video}
    return {"video": v, "video_index": 0, "audio": [], "subtitles": [], "duration_ms": 60_000}


FIELDS = Fields(found="fields", by="pictures", mark="tb", order="tff", order_by="mark",
                frames=1000, tff=990, undetermined=10)
TELECINE = Fields(found="telecine", by="pictures", mark="tb", order="tff", order_by="mark",
                  frames=1000, tff=800, progressive=200, repeated_top=200, repeated_bottom=200,
                  cadence=1.0, coverage=1.0)
AS_FOUND = Fields(found="progressive", by="pictures", mark="tb", order="tff", order_by="mark",
                  frames=1000, progressive=990, undetermined=10)


def test_an_hevc_source_in_fields_is_encoded_never_copied() -> None:
    probe = _probe("hevc", 1920, 1080, field_order="tb")
    # Its mark alone, no examination: in fields, deinterlaced at the
    # default rate, its frame rate.
    plan = plan_renditions(probe, parse_ladder(""), NVENC_ENCODERS)
    [rung] = plan.rungs
    assert (rung.mode, rung.encoder, rung.reason) == ("encode", "hevc_nvenc",
                                                       "hevc_not_copyable:fields")
    assert rung.maxrate_bps == 8_000_000  # its cap, as any source-size encode
    assert (plan.made_progressive, plan.scan_done, plan.frame_rate, plan.fps) == (
        "deinterlaced", "deinterlaced-frame", "25/1", 25.0)
    assert plan.fields.by == "mark"
    field = plan_renditions(probe, parse_ladder(""), NVENC_ENCODERS, deinterlace="field")
    assert (field.scan_done, field.frame_rate, field.fps) == ("deinterlaced-field", "50/1", 50.0)


def test_hevc_field_pictures_are_copied_and_keep_their_fields() -> None:
    # HEVC coded as field pictures, which ffmpeg decodes as 1920x540 at 50
    # and ffprobe marks progressive: copied as before, its record saying it
    # kept them.
    probe = _probe("hevc", 1920, 540, "50/1", field_order="progressive")
    pictures = Fields(found="field_pictures", by="bitstream", mark="progressive")
    plan = plan_renditions(probe, parse_ladder(""), NVENC_ENCODERS, fields=pictures)
    [rung] = plan.rungs
    assert (rung.mode, rung.reason) == ("copy", "source_already_hevc:hevc")
    assert (plan.made_progressive, plan.scan_done, plan.frame_rate) == (
        "fields_kept", "fields-kept", "50/1")
    assert ff.fields_chain(plan) is None
    assert plan.scan_done not in SCAN_RELEASES


def test_whole_frames_marked_as_fields_are_copied_as_today() -> None:
    probe = _probe("hevc", 1920, 1080, field_order="tb")
    plan = plan_renditions(probe, parse_ladder(""), NVENC_ENCODERS, fields=AS_FOUND)
    [rung] = plan.rungs
    assert (rung.mode, rung.reason) == ("copy", "source_already_hevc:hevc")
    assert plan.all_copy and plan.made_progressive == "as_found"
    assert plan.frame_rate == "25/1"


def test_an_interlaced_h264_is_not_passed_through_on_a_cpu_host() -> None:
    probe = _probe("h264", 1920, 1080, field_order="tt")
    plan = plan_renditions(probe, parse_ladder(""), CPU_ENCODERS, fields=FIELDS)
    [rung] = plan.rungs
    assert (rung.codec, rung.mode, rung.encoder, rung.reason) == (
        "hevc", "encode", "libx265", "h264_not_copyable:fields")
    # Nor at its own codec.
    [rung] = plan_renditions(probe, parse_ladder("source:h264"), CPU_ENCODERS,
                             fields=FIELDS).rungs
    assert (rung.codec, rung.mode, rung.encoder) == ("h264", "encode", "libx264")


def test_progressive_h264_still_passes_through_on_a_cpu_host() -> None:
    probe = _probe("h264", 1920, 1080, field_order="progressive")
    [rung] = plan_renditions(probe, parse_ladder(""), CPU_ENCODERS).rungs
    assert (rung.mode, rung.reason) == ("copy", "cpu_passthrough_h264")


@pytest.mark.parametrize(("rate", "fields", "deinterlace", "out"), [
    ("25/1", FIELDS, "field", "50/1"),
    ("30000/1001", FIELDS, "field", "60000/1001"),
    ("25/1", FIELDS, "frame", "25/1"),
    ("30000/1001", FIELDS, "frame", "30000/1001"),
    ("30000/1001", TELECINE, "field", "24000/1001"),
    ("30000/1001", TELECINE, "frame", "24000/1001"),
    ("25/1", AS_FOUND, "field", "25/1"),
])
def test_the_encoded_frame_rate(rate: str, fields: Fields, deinterlace: str, out: str) -> None:
    probe = _probe("mpeg2video", 720, 576, rate, field_order="tb")
    plan = plan_renditions(probe, parse_ladder(""), CPU_ENCODERS, fields=fields,
                           deinterlace=deinterlace)
    assert plan.frame_rate == out


def test_the_rate_is_one_of_two() -> None:
    with pytest.raises(LadderError, match="deinterlace must be one of field, frame"):
        plan_renditions(_probe("mpeg2video", 720, 576), parse_ladder(""), CPU_ENCODERS,
                        deinterlace="double")


# -------------------------------------------------------------- the command
def _cmd(probe: dict, ladder: str, fields: Fields | None, encoders=NVENC_ENCODERS,
         deinterlace: str = "field", decoder: str = "cpu"):
    settings = EncodeSettings(ladder=tuple(parse_ladder(ladder)), encoders=encoders,
                              deinterlace=deinterlace)
    plan = plan_renditions(probe, list(settings.ladder), encoders, fields=fields,
                           deinterlace=deinterlace)
    args, outputs = build_encode_command(SRC, INBOX, plan, settings, probe_subtitles=[],
                                         video_index=0, decoder=decoder)
    return plan, args, outputs


def _after(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def test_the_nvenc_command_deinterlaces_at_the_field_rate_first() -> None:
    # The GPU host's encode (hevc_nvenc), its source decoded on the CPU.
    plan, args, _ = _cmd(_probe("h264", 1920, 1080, field_order="tb"), "", FIELDS)
    assert _after(args, "-filter_complex") == (
        "[0:0]bwdif=mode=send_field:parity=tff:deint=all,format=yuv420p[ov0]")
    assert "-hwaccel" not in args
    assert _after(args, "-c:v") == "hevc_nvenc"
    # A keyframe every 6 s of 50 frames a second.
    assert _after(args, "-g") == "300"
    assert _after(args, "-force_key_frames") == "expr:gte(t,n_forced*6)"


def test_the_frame_rate_setting_names_send_frame() -> None:
    _, args, _ = _cmd(_probe("mpeg2video", 720, 576, field_order="bt"), "",
                      Fields(found="fields", by="mark", mark="bt", order="bff",
                             order_by="mark"), deinterlace="frame")
    assert _after(args, "-filter_complex").startswith(
        "[0:0]bwdif=mode=send_frame:parity=bff:deint=all,")
    assert _after(args, "-g") == "150"


def test_a_ladder_splits_after_the_deinterlace() -> None:
    _, args, outputs = _cmd(_probe("mpeg2video", 720, 576, field_order="tb"), "source,360p",
                            FIELDS, encoders=CPU_ENCODERS)
    graph = _after(args, "-filter_complex")
    assert graph.startswith("[0:0]bwdif=mode=send_field:parity=tff:deint=all,split=2[bv0][bv1];")
    assert "[bv1]scale=450:360,setsar=1,format=yuv420p[ov1]" in graph
    assert [o.rung.encoder for o in outputs] == ["libx265", "libx264"]


def test_a_telecined_film_is_brought_back_to_its_frames() -> None:
    _, args, _ = _cmd(_probe("mpeg2video", 720, 480, "30000/1001", field_order="tb"), "",
                      TELECINE, encoders=CPU_ENCODERS)
    assert _after(args, "-filter_complex") == (
        "[0:0]fieldmatch=order=tff:combmatch=full,"
        "bwdif=mode=send_frame:parity=auto:deint=interlaced,decimate,format=yuv420p[ov0]")
    assert _after(args, "-g") == "144"  # 6 s of 23.976


def test_whole_frames_marked_as_fields_are_encoded_marked_progressive() -> None:
    _, args, _ = _cmd(_probe("mpeg2video", 720, 576, field_order="tb"), "", AS_FOUND,
                      encoders=CPU_ENCODERS)
    assert _after(args, "-filter_complex") == "[0:0]setfield=prog,format=yuv420p[ov0]"
    assert "bwdif" not in " ".join(args)
    assert _after(args, "-g") == "150"


def test_a_progressive_source_has_no_fields_filter() -> None:
    _, args, _ = _cmd(_probe("mpeg2video", 720, 576, field_order="progressive"), "", None,
                      encoders=CPU_ENCODERS)
    assert "-filter_complex" not in args


def test_a_source_in_fields_is_never_decoded_on_the_gpu() -> None:
    # Unmarked, so the source alone gives the GPU no reason: the plan does.
    probe = _probe("h264", 1920, 1080, field_order="")
    unmarked = Fields(found="fields", by="pictures", mark="", order="tff",
                      order_by="pictures", frames=1000, tff=990)
    settings = EncodeSettings(ladder=tuple(parse_ladder("")), encoders=NVENC_ENCODERS,
                              decoder="cuda")
    plan = plan_renditions(probe, parse_ladder(""), NVENC_ENCODERS, fields=unmarked)
    assert plan.source.cuda_decode_blocker is None
    assert ff.cuda_decode_blocker(plan, settings) == "interlaced"
    with pytest.raises(TranscodeError, match="interlaced"):
        build_encode_command(SRC, INBOX, plan, settings, probe_subtitles=[], video_index=0,
                             decoder="cuda")
    tc = plan_renditions(probe, parse_ladder(""), NVENC_ENCODERS,
                         fields=Fields(found="telecine", by="pictures", mark=""))
    assert ff.cuda_decode_blocker(tc, settings) == "telecine"
    # A source marked as fields keeps the reason it had.
    marked = plan_renditions(_probe("h264", 1920, 1080, field_order="tt"), parse_ladder(""),
                             NVENC_ENCODERS, fields=FIELDS)
    assert ff.cuda_decode_blocker(marked, settings) == "interlaced"


# --------------------------------------------------------------- the record
def test_the_record_says_what_was_done() -> None:
    plan = plan_renditions(_probe("mpeg2video", 720, 576, field_order="tb"), parse_ladder(""),
                           CPU_ENCODERS, fields=FIELDS)
    assert scan_record(plan) == {
        "found": "fields", "judgedBy": "pictures", "fieldOrder": "tb",
        "done": "deinterlaced-frame", "order": "tff", "orderBy": "mark", "frameRate": "25/1",
        "filter": "bwdif=mode=send_frame:parity=tff:deint=all",
        "idet": {"frames": 1000, "tff": 990, "bff": 0, "progressive": 0, "undetermined": 10,
                 "repeatedTop": 0, "repeatedBottom": 0, "cadence": 0.0, "coverage": 0.0},
    }
    contract = build_contract("i", plan, [], CPU_ENCODERS, duration_ms=60_000)
    assert contract["scan"] == scan_record(plan)
    assert contract["version"] == 1
    # At the field rate: its output rate, twice the source's.
    field = plan_renditions(_probe("mpeg2video", 720, 576, field_order="tb"), parse_ladder(""),
                            CPU_ENCODERS, fields=FIELDS, deinterlace="field")
    rec = scan_record(field)
    assert (rec["done"], rec["frameRate"], rec["filter"]) == (
        "deinterlaced-field", "50/1", "bwdif=mode=send_field:parity=tff:deint=all")


def test_the_record_of_a_telecined_film_and_of_whole_frames() -> None:
    tc = plan_renditions(_probe("mpeg2video", 720, 480, "30000/1001", field_order="tb"),
                         parse_ladder(""), CPU_ENCODERS, fields=TELECINE)
    rec = scan_record(tc)
    assert (rec["found"], rec["done"], rec["frameRate"]) == (
        "telecine", "inverse-telecine", "24000/1001")
    assert rec["filter"].startswith("fieldmatch=order=tff:")
    copied = plan_renditions(_probe("hevc", 1920, 1080, field_order="tb"), parse_ladder(""),
                             NVENC_ENCODERS, fields=AS_FOUND)
    rec = scan_record(copied)
    assert (rec["found"], rec["done"], rec["frameRate"], rec["filter"]) == (
        "progressive", "progressive", "25/1", None)
    pictures = plan_renditions(_probe("hevc", 1920, 540, "50/1", field_order="progressive"),
                               parse_ladder(""), NVENC_ENCODERS,
                               fields=Fields(found="field_pictures", by="bitstream",
                                             mark="progressive"))
    rec = scan_record(pictures)
    assert (rec["found"], rec["judgedBy"], rec["done"], rec["frameRate"], rec["idet"]) == (
        "field-pictures", "bitstream", "fields-kept", "50/1", None)


def test_the_gate_lets_an_original_in_fields_go_on_three_words_only() -> None:
    assert SCAN_DONE == ("inverse-telecine", "progressive", "deinterlaced-field",
                         "deinterlaced-frame", "fields-kept", "none")
    assert SCAN_RELEASES == ("inverse-telecine", "progressive", "deinterlaced-field")


def test_the_record_of_a_source_nothing_examined() -> None:
    plan = plan_renditions(_probe("mpeg2video", 720, 576, field_order="progressive"),
                           parse_ladder(""), CPU_ENCODERS)
    assert scan_record(plan) == {
        "found": "progressive", "judgedBy": "mark", "fieldOrder": "progressive", "done": "none",
        "order": "auto", "orderBy": "frames", "frameRate": "25/1", "filter": None,
        "idet": None,
    }


def test_the_step_details_say_it() -> None:
    probe = _probe("mpeg2video", 720, 576, field_order="tb")
    cases = [
        (FIELDS, "field", "scan=fields:pictures:deinterlaced-field@50/1 "),
        (FIELDS, "frame", "scan=fields:pictures:deinterlaced-frame@25/1 "),
        (AS_FOUND, "field", "scan=progressive:pictures:progressive@25/1 "),
    ]
    for fields, rate, details in cases:
        plan = plan_renditions(probe, parse_ladder(""), CPU_ENCODERS, fields=fields,
                               deinterlace=rate)
        assert _scan_details(plan) == details
    tc = plan_renditions(_probe("mpeg2video", 720, 480, "30000/1001", field_order="tb"),
                         parse_ladder(""), CPU_ENCODERS, fields=TELECINE)
    assert _scan_details(tc) == "scan=telecine:pictures:inverse-telecine@24000/1001 "
    pictures = plan_renditions(_probe("hevc", 1920, 540, "50/1", field_order="progressive"),
                               parse_ladder(""), NVENC_ENCODERS,
                               fields=Fields(found="field_pictures", by="bitstream",
                                             mark="progressive"))
    assert _scan_details(pictures) == "scan=field-pictures:bitstream:fields-kept@50/1 "
    progressive = plan_renditions(_probe("mpeg2video", 720, 576, field_order="progressive"),
                                  parse_ladder(""), CPU_ENCODERS)
    assert _scan_details(progressive) == ""


# ---------------------------------------------------------------- the setting
def _env(monkeypatch: pytest.MonkeyPatch, **env: str) -> Config:
    for key, value in {"KATALOG_API_URL": "http://katalog", "OIDC_TOKEN_URL": "http://t",
                       "OIDC_CLIENT_ID": "c", "OIDC_CLIENT_SECRET": "s", **env}.items():
        monkeypatch.setenv(key, value)
    return Config.from_env()


def test_the_setting_defaults_to_the_frame_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEINTERLACE", raising=False)
    assert DEFAULT_DEINTERLACE == "frame"
    assert _env(monkeypatch).deinterlace == "frame"
    assert _env(monkeypatch, DEINTERLACE="").deinterlace == "frame"
    assert _env(monkeypatch, DEINTERLACE=" Field ").deinterlace == "field"
    assert EncodeSettings().deinterlace == "frame"
    with pytest.raises(RuntimeError, match="DEINTERLACE must be field or frame"):
        _env(monkeypatch, DEINTERLACE="50p")


# ------------------------------------------------------- the examination run
def test_the_examination_reads_five_windows_of_the_middle_of_the_picture(
        monkeypatch: pytest.MonkeyPatch) -> None:
    runs: list[list[str]] = []

    def run(args, **_kw):
        runs.append(args)
        frame = ("frame:0 pts:0 pts_time:0\nlavfi.idet.repeated.current_frame=neither\n"
                 "lavfi.idet.single.current_frame=tff\nlavfi.idet.multiple.current_frame=tff\n")
        return subprocess.CompletedProcess(args, 0, frame * 200, "")

    monkeypatch.setattr(ff.subprocess, "run", run)
    probe = _probe("mpeg2video", 720, 576, field_order="tb")
    probe["video_index"] = 3
    f = ff.examine_fields(SRC, probe)
    assert (f.found, f.by, f.frames, f.tff) == ("fields", "pictures", 1000, 1000)
    assert [a[a.index("-ss") + 1] for a in runs] == ["6.000", "18.000", "30.000", "42.000",
                                                     "54.000"]
    for a in runs:
        assert a[a.index("-map") + 1] == "0:3"
        assert a[a.index("-frames:v") + 1] == "200"
        assert a[a.index("-vf") + 1] == (
            "crop=w=trunc(iw*0.8/4)*4:h=trunc(ih*0.7/4)*4:x=trunc(iw*0.1/4)*4"
            ":y=trunc(ih*0.15/4)*4,idet,metadata=mode=print:file=-")
        assert "-hwaccel" not in a


def test_the_examination_without_a_duration_reads_one_window_from_the_start(
        monkeypatch: pytest.MonkeyPatch) -> None:
    runs: list[list[str]] = []
    monkeypatch.setattr(ff.subprocess, "run", lambda args, **_kw: runs.append(args) or
                        subprocess.CompletedProcess(args, 1, "", "Invalid data found"))
    probe = _probe("mpeg2video", 720, 576, field_order="bb")
    probe["duration_ms"] = 0
    with capture_logs() as logs:
        f = ff.examine_fields(SRC, probe)
    assert len(runs) == 1 and "-ss" not in runs[0]
    # Nothing read: the mark decides.
    assert (f.found, f.by, f.order) == ("fields", "mark", "bff")
    assert [e["event"] for e in logs] == ["transcoder.fields.window_failed"]


def test_a_source_not_examined_runs_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ff.subprocess, "run", lambda *a, **k: pytest.fail("ran ffmpeg"))
    f = ff.examine_fields(SRC, _probe("h264", 1920, 1080, field_order="progressive"))
    assert (f.found, f.by, f.frames) == ("progressive", "mark", 0)


@pytest.mark.parametrize(("trace", "found"), [
    ("[trace_headers @ 0x1] 265         field_seq_flag      1 = 1\n", "field_pictures"),
    ("[trace_headers @ 0x1] 259         field_seq_flag      0 = 0\n", "progressive"),
    ("", "progressive"),
])
def test_hevc_is_asked_its_parameter_sets_whether_it_is_field_pictures(
        monkeypatch: pytest.MonkeyPatch, trace: str, found: str) -> None:
    runs: list[list[str]] = []
    monkeypatch.setattr(ff.subprocess, "run", lambda args, **_kw: runs.append(args) or
                        subprocess.CompletedProcess(args, 0, "", trace))
    f = ff.examine_fields(SRC, _probe("hevc", 3840, 2160, field_order="progressive"))
    assert f.found == found
    assert f.by == ("bitstream" if found == "field_pictures" else "mark")
    # One packet's parameter sets, nothing decoded; no idet for a 2160 one.
    [args] = runs
    assert args[args.index("-bsf:v") + 1] == "trace_headers"
    assert args[args.index("-c") + 1] == "copy" and args[args.index("-frames:v") + 1] == "1"
