"""Sources that may hold fields: the verdict on what idet says of a sample as
it is and after fieldmatch, the plan, the command and the record (no ffmpeg
run; tests/test_fields_real.py runs made clips).

The principle: the record (scan.done) says inverse-telecine, progressive or
deinterlaced-field, the words that let an original in fields go, only for a
package with no combing left and no moment of its source dropped; in doubt
it says a word that keeps the original."""

from __future__ import annotations

import re
import subprocess
from dataclasses import replace
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
    FieldsVerification,
    FieldsWindow,
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
NTSC = "30000/1001"
WORDS = {"t": "tff", "b": "bff", "p": "progressive", "u": "undetermined"}
REPS = {"n": "neither", "t": "top", "b": "bottom"}


def _w(single: str, repeated: str | None = None, matched: str | None = None,
       matched_bff: str | None = None) -> FieldsWindow:
    """A window in idet's letters: t b p u its single-frame verdicts as the
    frames are, n t b its repeated fields (none when not given), and its
    single-frame verdicts after fieldmatch top field first (`matched`: as the
    frames are when not given: video, which fieldmatch cannot put back) and
    bottom first (as top first when not given)."""
    repeated = repeated if repeated is not None else "n" * len(single)
    matched = matched if matched is not None else single
    matched_bff = matched_bff if matched_bff is not None else matched
    return FieldsWindow(tuple(WORDS[c] for c in single), tuple(REPS[c] for c in repeated),
                        {"tff": tuple(WORDS[c] for c in matched),
                         "bff": tuple(WORDS[c] for c in matched_bff)})


def _telecine(blocks: int = 40, shift: int = 0) -> FieldsWindow:
    """A window of a 3:2 film: 4 of 5 frames combed, one top and one bottom
    field repeated in every 5 frames, all whole once fieldmatch puts them
    back."""
    single = ("ptttt" * (blocks + 1))[shift:shift + 5 * blocks]
    repeated = ("nntnb" * (blocks + 1))[shift:shift + 5 * blocks]
    return _w(single, repeated, "p" * len(single))


# ------------------------------------------------------------- the verdict
def test_true_interlacing_is_video_in_the_order_its_mark_names() -> None:
    f = judge_fields("tb", [_w("u" + "t" * 199)] * 5)
    assert (f.found, f.by, f.order, f.order_by, f.order_seen) == (
        "fields", "pictures", "tff", "mark", True)
    assert (f.frames, f.tff, f.undetermined, f.residue) == (1000, 995, 5, 1.0)
    assert f.in_fields


def test_idet_overrules_a_mark_that_names_the_other_field_first() -> None:
    f = judge_fields("tb", [_w("b" * 200)] * 5)
    assert (f.found, f.order, f.order_by, f.order_seen) == ("fields", "bff", "pictures", True)


def test_combing_in_both_orders_is_no_verdict_of_fields() -> None:
    # A sharp progressive picture idet misreads as combing, 43 % TFF and 57 %
    # BFF: no woven fields come in both orders. Marked as fields, the
    # pictures can't tell, and the safe chain keeps the original; marked
    # progressive, or not marked, it is whole frames.
    sharp = [_w("ttbbbp" * 33 + "pp")] * 5
    assert judge_fields("tb", sharp).found == "unknown"
    assert judge_fields("progressive", sharp).found == "progressive"
    assert judge_fields("", sharp).found == "progressive"


@pytest.mark.parametrize("mark", ["progressive", "", "unknown"])
def test_a_source_marked_progressive_or_not_marked_is_examined_all_the_same(mark: str) -> None:
    # ffprobe's mark is the first picture's: 2 s of 25p, then 50i woven.
    windows = [_w("p" * 50 + "t" * 150)] + [_w("t" * 200)] * 4
    f = judge_fields(mark, windows)
    assert (f.found, f.by, f.order, f.order_by, f.order_seen) == (
        "fields", "pictures", "tff", "pictures", True)


def test_a_source_marked_progressive_takes_more_to_be_found_in_fields() -> None:
    # 15 % of its frames in fields, in one order (video among whole frames,
    # mixed): a source marked as fields is found so; one marked progressive,
    # or not marked, is not (20 % and 50 frames).
    windows = [_w("t" * 30 + "p" * 170)] * 5
    assert judge_fields("tb", windows).found == "mixed"
    assert judge_fields("progressive", windows).found == "progressive"
    assert judge_fields("", windows).found == "progressive"
    assert judge_fields("progressive", [_w("t" * 45 + "p" * 155)] * 5).found == "mixed"


def test_a_32_pulldown_at_29_97_is_telecine() -> None:
    f = judge_fields("tb", [_telecine()] * 5, NTSC)
    assert (f.found, f.by) == ("telecine", "pictures")
    assert (f.repeated_top, f.repeated_bottom, f.cadence_blocks, f.other_blocks) == (
        200, 200, 200, 0)
    assert (f.residue, f.coverage) == (0.0, 1.0)


def test_the_same_pulldown_at_25_is_film_never_decimated() -> None:
    # Decimated, a 25-frame source would come out at 20.
    assert judge_fields("tb", [_telecine()] * 5, "25/1").found == "film"


def test_cuts_inside_a_window_keep_the_cadence() -> None:
    # A cut every 15 blocks (2.5 s), each moving the cadence by a frame or two.
    def cut_up(shifts: list[int]) -> FieldsWindow:
        parts = [_telecine(15, k) for k in shifts]
        return FieldsWindow(sum((p.single for p in parts), ()),
                            sum((p.repeated for p in parts), ()),
                            {o: sum((p.matched[o] for p in parts), ()) for o in ("tff", "bff")})
    f = judge_fields("tb", [cut_up([0, 3, 1]), cut_up([4, 2, 0]), cut_up([2, 4, 1])] * 2, NTSC)
    assert f.found == "telecine"
    assert f.coverage >= 0.9


def test_held_pictures_count_neither_way() -> None:
    # Half of the film a held picture: no repeat and no combing, blind.
    # idet leaves a held picture's frames undetermined (a still has no
    # combing and nothing to call progressive by).
    held = _w("u" * 200, None, "u" * 200)
    f = judge_fields("tb", [_telecine(), held, _telecine(), held, _telecine()], NTSC)
    assert (f.found, f.blind_blocks, f.coverage) == ("telecine", 80, 1.0)


def test_one_repeat_a_block_on_the_cadence_is_on_it() -> None:
    # A held drawing hides one of the two repeats: the other, at its place,
    # holds the cadence.
    single, repeated = "ptttt" * 40, ("nntnb" + "nntnn" + "nnnnb") * 13 + "nntnb"
    f = judge_fields("tb", [_w(single, repeated, "p" * 200)] * 5, NTSC)
    assert (f.found, f.coverage) == ("telecine", 1.0)


def test_a_window_cut_short_whose_one_pair_is_its_last_block_is_blind() -> None:
    # The last window of a short title: 11 frames, its first block one repeat
    # (blind), its second the window's only pair, nothing after it to confirm
    # it by. Blind, as a window's first block is; one more block after it
    # that does not confirm it, and it is off (a cut).
    from transcoder.decision import _blocks

    single = tuple(WORDS[c] for c in "pttttptttt" + "t")
    assert _blocks(single, tuple(REPS[c] for c in "nnntnbnntnb")) == (0, 0, 2)
    longer = tuple(WORDS[c] for c in "ptttt" * 3)
    assert _blocks(longer, tuple(REPS[c] for c in "nnntn" + "bnntn" + "ntnnb"))[1] == 2
    f = judge_fields("tb", [_telecine()] * 4 + [_w("pttttptttt" + "t", "nnntnbnntnb",
                                                    "p" * 11)], NTSC)
    assert (f.found, f.other_blocks) == ("telecine", 0)


def test_repeats_off_a_cadence_are_no_telecine() -> None:
    # As many repeats as 3:2 makes, at other places in every block.
    drifting = ("tnbnn" + "nbntn" + "nntnb" + "bnntn" + "ntnnb") * 8
    f = judge_fields("tb", [_w("ptttt" * 40, drifting, "p" * 200)] * 5, NTSC)
    assert f.coverage < 0.8
    assert f.found == "film"


def test_a_film_with_a_video_part_over_80_percent_film_is_mixed() -> None:
    # 85 % 3:2 film, 15 % video: its cadence covers what idet sees of the
    # film, but fieldmatch leaves the video combed, in a run: mixed, never
    # decimated.
    video = _w("t" * 150, None, "t" * 150)
    f = judge_fields("tb", [_telecine()] * 4 + [_telecine(10), video], NTSC)
    assert f.coverage >= 0.8
    assert (f.found, f.matched_run) == ("mixed", 150)


def test_orphan_fields_here_and_there_leave_a_film_a_film() -> None:
    # One combed frame after fieldmatch every 40 (a cut's orphan field, grain):
    # 2.5 %, never two in a row.
    window = _telecine()
    matched = tuple("tff" if i % 40 == 7 else v for i, v in enumerate(window.matched["tff"]))
    film = FieldsWindow(window.single, window.repeated, {"tff": matched, "bff": matched})
    f = judge_fields("tb", [film] * 5, NTSC)
    assert (f.found, f.matched_run) == ("telecine", 1)
    assert 0.02 < f.residue < 0.05


def test_a_film_shifted_by_a_field_is_film() -> None:
    # 2:2 PAL shifted by a field: every frame combed, bottom first, whole once
    # matched.
    f = judge_fields("bt", [_w("b" * 200, None, "p" * 200, "p" * 200)] * 5, "25/1")
    assert (f.found, f.order, f.residue) == ("film", "bff", 0.0)


def test_each_window_is_judged_on_its_own() -> None:
    # Video: fieldmatch leaves at least 30 % of a window combed (a moving
    # video's, about half: it pairs fields of two moments into a clean-looking
    # frame for the rest).
    video = _w("t" * 200, None, "tp" * 100)
    film = _w("t" * 200, None, "p" * 200)
    whole = _w("p" * 200)
    unsure = _w("t" * 200, None, "t" * 20 + "p" * 180)
    assert judge_fields("tb", [video] * 5).found == "fields"
    assert judge_fields("tb", [film] * 5).found == "film"
    # A film's whole windows are its frames, whole already.
    assert judge_fields("tb", [film, whole, film, whole, film]).found == "film"
    # Film and video; a film body in whole frames with credits in fields
    # (deinterlaced whole, every frame of the film would be rebuilt from one
    # field); a window neither film nor video: mixed.
    assert judge_fields("tb", [film] * 4 + [video]).found == "mixed"
    assert judge_fields("tb", [whole] * 4 + [video]).found == "mixed"
    assert judge_fields("tb", [video] * 4 + [unsure]).found == "mixed"
    f = judge_fields("tb", [film, film, video, whole, unsure])
    assert (f.film_windows, f.video_windows, f.whole_windows, f.unsure_windows) == (2, 1, 1, 1)


def test_a_film_fieldmatch_could_not_put_back_is_never_video() -> None:
    # Grain over a soft telecined picture: fieldmatch leaves it combed, but
    # its repeats keep a 3:2 cadence: mixed, field-matched, kept.
    grainy = _w("ptttt" * 40, "nntnb" * 40, "tp" * 100)
    assert judge_fields("tb", [grainy] * 5, NTSC).found == "mixed"


def test_whole_frames_marked_as_fields_are_progressive() -> None:
    f = judge_fields("tb", [_w("u" + "p" * 199)] * 5)
    assert (f.found, f.by) == ("progressive", "pictures")
    assert not f.in_fields


def test_the_share_and_the_least_number_of_frames_in_fields() -> None:
    n = FIELDS_MIN_FRAMES
    assert judge_fields("tb", [_w("t" * n + "p" * (100 - n))]).found == "mixed"
    assert judge_fields("tb", [_w("t" * (n - 1) + "p" * (101 - n))]).found == "progressive"
    assert judge_fields("tb", [_w("t" * 49 + "p" * 951)]).found == "progressive"
    assert judge_fields("tb", [_w("t" * 50 + "p" * 950)]).found == "mixed"


def test_a_still_can_not_tell() -> None:
    few = FIELDS_MIN_DETERMINED - 1
    windows = [_w("u" * 200)] * 4 + [_w("t" * few + "u" * (200 - few))]
    # Whatever the mark: the pictures can't tell, the safe chain, unexamined.
    for mark in ("tb", "", "progressive"):
        f = judge_fields(mark, windows)
        assert (f.found, f.by) == ("unknown", "mark")


@pytest.mark.parametrize("mark", ["", "tb", "progressive"])
def test_an_examination_that_read_nothing_fails_safe(mark: str) -> None:
    f = judge_fields(mark, [FieldsWindow()] * 5)
    assert (f.found, f.by, f.frames) == ("unknown", "failed", 0)
    assert f.in_fields


@pytest.mark.parametrize(("mark", "found", "order"), [
    ("tt", "unknown", "tff"), ("tb", "unknown", "tff"), ("bb", "unknown", "bff"),
    ("bt", "unknown", "bff"), ("progressive", "progressive", "tff"), ("", "progressive", "tff"),
    ("TB", "unknown", "tff"),
])
def test_the_mark_alone(mark: str, found: str, order: str) -> None:
    # A source marked as fields that nothing examined may hold anything.
    f = fields_by_mark(mark)
    assert (f.found, f.by, f.order) == (found, "mark", order)


def test_an_unmarked_source_without_an_order_from_idet_is_taken_top_field_first() -> None:
    f = judge_fields("", [FieldsWindow()])
    assert (f.order, f.order_by, f.order_seen) == ("tff", "default", False)


# ------------------------------------------------------------ examined
def _probe(codec: str, w: int, h: int, rate: str = "25/1", **video: object) -> dict:
    v = {"codec_name": codec, "width": w, "height": h, "pix_fmt": "yuv420p",
         "avg_frame_rate": rate, "profile": {"hevc": "Main", "h264": "High"}.get(codec, ""),
         **video}
    return {"video": v, "video_index": 0, "audio": [], "subtitles": [], "duration_ms": 60_000}


@pytest.mark.parametrize(("codec", "field_order", "height", "examined"), [
    ("mpeg2video", "tt", 576, True), ("h264", "progressive", 1080, True),
    ("hevc", "progressive", 1080, True), ("vp9", "", 576, True), ("av1", "", 1080, True),
    ("hevc", "progressive", 1088, True), ("hevc", "progressive", 1440, False),
    ("hevc", "progressive", 2160, False), ("h264", "tt", 2160, False),
])
def test_which_sources_are_examined(codec: str, field_order: str, height: int,
                                    examined: bool) -> None:
    src = SourceInfo.from_probe(_probe(codec, 1920, height, field_order=field_order))
    assert src.fields_examined() is examined


def test_dolby_vision_is_not_examined() -> None:
    # No Dolby Vision profile is interlaced.
    record = {"side_data_type": "DOVI configuration record", "dv_profile": 7,
              "dv_bl_signal_compatibility_id": 6}
    src = SourceInfo.from_probe(_probe("hevc", 1920, 1080, side_data_list=[record]))
    assert not src.fields_examined()
    assert not SourceInfo.from_probe(_probe("hevc", 1920, 1080, codec_tag_string="dvh1")
                                     ).fields_examined()


# ---------------------------------------------------------------- the plan
FIELDS = Fields(found="fields", by="pictures", mark="tb", order="tff", order_by="mark",
                order_seen=True, frames=1000, tff=990, undetermined=10, matched_combed=990,
                matched_determined=990)
FILM_22 = Fields(found="film", by="pictures", mark="bt", order="bff", order_by="mark",
                 order_seen=True, frames=1000, bff=1000, matched_determined=1000)
TELECINE = Fields(found="telecine", by="pictures", mark="tb", order="tff", order_by="mark",
                  order_seen=True, frames=1000, tff=800, progressive=200, repeated_top=200,
                  repeated_bottom=200, matched_determined=1000, cadence_blocks=200)
MIXED = Fields(found="mixed", by="pictures", mark="tb", order="tff", order_by="mark",
               order_seen=True, frames=1000, tff=850, progressive=150, matched_combed=150,
               matched_determined=1000, matched_run=150)
AS_FOUND = Fields(found="progressive", by="pictures", mark="tb", order="tff", order_by="mark",
                  frames=1000, progressive=990, undetermined=10)
UNMARKED = Fields(found="progressive", by="pictures", mark="", frames=1000, progressive=1000)


def test_an_hevc_source_in_fields_is_encoded_never_copied() -> None:
    probe = _probe("hevc", 1920, 1080, field_order="tb")
    plan = plan_renditions(probe, parse_ladder(""), NVENC_ENCODERS, fields=FIELDS)
    [rung] = plan.rungs
    assert (rung.mode, rung.encoder, rung.reason) == ("encode", "hevc_nvenc",
                                                       "hevc_not_copyable:fields")
    assert rung.maxrate_bps == 8_000_000
    assert (plan.made_progressive, plan.scan_done, plan.frame_rate) == (
        "deinterlaced", "deinterlaced-frame", "25/1")
    field = plan_renditions(probe, parse_ladder(""), NVENC_ENCODERS, fields=FIELDS,
                            deinterlace="field")
    assert (field.scan_done, field.frame_rate, field.fps) == ("deinterlaced-field", "50/1", 50.0)


def test_a_source_marked_as_fields_nothing_examined_gets_the_safe_chain() -> None:
    plan = plan_renditions(_probe("hevc", 1920, 1080, field_order="tb"), parse_ladder(""),
                           NVENC_ENCODERS, deinterlace="field")
    [rung] = plan.rungs
    assert (rung.mode, rung.reason) == ("encode", "hevc_not_copyable:unknown")
    assert (plan.made_progressive, plan.scan_done, plan.frame_rate) == (
        "field_matched", "unexamined", "25/1")
    assert ff.fields_chain(plan) == (
        "fieldmatch=order=tff:combmatch=full,bwdif=mode=send_frame:parity=tff:deint=interlaced,"
        "idet,bwdif=mode=send_frame:parity=tff:deint=interlaced")


def test_the_field_rate_is_taken_only_in_an_order_idet_saw() -> None:
    # The mark alone named the order: a field-rate deinterlace in a wrong one
    # costs 7.5 dB, so the frame rate, and its keeping word.
    unseen = Fields(**{**FIELDS.__dict__, "order_seen": False, "order_by": "mark"})
    plan = plan_renditions(_probe("mpeg2video", 720, 576, field_order="tb"), parse_ladder(""),
                           CPU_ENCODERS, fields=unseen, deinterlace="field")
    assert (plan.deinterlace_rate, plan.scan_done, plan.frame_rate) == (
        "frame", "deinterlaced-frame", "25/1")
    assert ff.fields_chain(plan) == "bwdif=mode=send_frame:parity=tff:deint=all"


def test_hevc_field_pictures_are_copied_and_keep_their_fields() -> None:
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
    assert plan.scan_done == "progressive"


@pytest.mark.parametrize("fields", [FIELDS, FILM_22, TELECINE, MIXED,
                                    fields_by_mark("tt"), judge_fields("", [FieldsWindow()])])
def test_a_source_that_may_hold_fields_is_never_passed_through(fields: Fields) -> None:
    probe = _probe("h264", 1920, 1080, field_order=fields.mark or "progressive")
    [rung] = plan_renditions(probe, parse_ladder(""), CPU_ENCODERS, fields=fields).rungs
    assert (rung.codec, rung.mode, rung.encoder) == ("hevc", "encode", "libx265")
    assert rung.reason == f"h264_not_copyable:{fields.found}"


def test_progressive_h264_still_passes_through_on_a_cpu_host() -> None:
    probe = _probe("h264", 1920, 1080, field_order="progressive")
    progressive = Fields(found="progressive", by="pictures", mark="progressive", frames=1000,
                         progressive=1000)
    [rung] = plan_renditions(probe, parse_ladder(""), CPU_ENCODERS, fields=progressive).rungs
    assert (rung.mode, rung.reason) == ("copy", "cpu_passthrough_h264")


@pytest.mark.parametrize(("rate", "fields", "deinterlace", "out", "done"), [
    ("25/1", FIELDS, "field", "50/1", "deinterlaced-field"),
    ("30000/1001", FIELDS, "field", "60000/1001", "deinterlaced-field"),
    ("25/1", FIELDS, "frame", "25/1", "deinterlaced-frame"),
    ("30000/1001", TELECINE, "field", "24000/1001", "inverse-telecine"),
    ("25/1", FILM_22, "field", "25/1", "inverse-telecine"),
    ("30000/1001", FILM_22, "field", "30000/1001", "field-matched"),
    ("30000/1001", MIXED, "field", "30000/1001", "field-matched"),
    ("25/1", AS_FOUND, "field", "25/1", "progressive"),
])
def test_the_encoded_frame_rate_and_the_word(rate: str, fields: Fields, deinterlace: str,
                                             out: str, done: str) -> None:
    probe = _probe("mpeg2video", 720, 576, rate, field_order=fields.mark)
    plan = plan_renditions(probe, parse_ladder(""), CPU_ENCODERS, fields=fields,
                           deinterlace=deinterlace)
    assert (plan.frame_rate, plan.scan_done) == (out, done)


def test_video_whose_motion_idet_barely_sees_is_not_released_as_film() -> None:
    # A slow object over a still: fieldmatch makes it look whole, every second
    # moment gone. Its sample mostly undetermined, its film keeps its original.
    slow = Fields(found="film", by="pictures", mark="tb", order="tff", order_by="mark",
                  order_seen=True, frames=1000, tff=220, undetermined=780,
                  matched_determined=700)
    plan = plan_renditions(_probe("mpeg2video", 720, 576, field_order="tb"), parse_ladder(""),
                           CPU_ENCODERS, fields=slow)
    assert plan.scan_done == "field-matched"


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
    plan, args, _ = _cmd(_probe("h264", 1920, 1080, field_order="tb"), "", FIELDS)
    # 25 frames: no rate bwdif re-times captions at, so none are kept twice.
    assert _after(args, "-filter_complex") == (
        "[0:0]bwdif=mode=send_field:parity=tff:deint=all,sidedata=mode=delete:type=A53_CC,"
        "format=yuv420p[ov0]")
    assert "-hwaccel" not in args
    assert _after(args, "-c:v") == "hevc_nvenc"
    assert _after(args, "-g") == "300"
    assert _after(args, "-force_key_frames") == "expr:gte(t,n_forced*6)"


def test_the_frame_rate_setting_names_send_frame() -> None:
    bff = Fields(**{**FIELDS.__dict__, "mark": "bt", "order": "bff"})
    _, args, _ = _cmd(_probe("mpeg2video", 720, 576, field_order="bt"), "", bff,
                      deinterlace="frame")
    assert _after(args, "-filter_complex").startswith(
        "[0:0]bwdif=mode=send_frame:parity=bff:deint=all,")
    assert _after(args, "-g") == "150"


def test_a_ladder_splits_after_the_deinterlace() -> None:
    _, args, outputs = _cmd(_probe("mpeg2video", 720, 576, field_order="tb"), "source,360p",
                            FIELDS, encoders=CPU_ENCODERS)
    graph = _after(args, "-filter_complex")
    assert graph.startswith("[0:0]bwdif=mode=send_field:parity=tff:deint=all,"
                            "sidedata=mode=delete:type=A53_CC,split=2[bv0][bv1];")
    assert "[bv1]scale=450:360,setsar=1,format=yuv420p[ov1]" in graph
    assert [o.rung.encoder for o in outputs] == ["libx265", "libx264"]


def test_a_telecined_film_is_field_matched_and_decimated() -> None:
    _, args, _ = _cmd(_probe("mpeg2video", 720, 480, NTSC, field_order="tb"), "",
                      TELECINE, encoders=CPU_ENCODERS)
    # decimate's frames lose their captions; every frame's go ahead of the
    # film frame at four fifths of its number, and fps keeps them for it; on
    # a timeline from 0, put back at the source's first picture.
    rate = "24000/1001"
    assert _after(args, "-filter_complex") == (
        "[0:0]fieldmatch=order=tff:combmatch=full,bwdif=mode=send_frame:parity=tff:deint=interlaced,"
        "idet,bwdif=mode=send_frame:parity=tff:deint=interlaced[fitm];"
        "[fitm]split=2[ccc][ccp];"
        "[ccp]decimate=dupthresh=4,sidedata=mode=delete:type=A53_CC,tpad=stop=1:stop_mode=clone,"
        f"setpts=round(N/({rate})/TB)[ccd];"
        f"[ccc]setpts=round((ceil(N*4/5)-0.1)/({rate})/TB)[cck];"
        f"[cck][ccd]interleave,fps={rate},settb=AVTB,setpts=PTS+0[fout];"
        "[fout]format=yuv420p[ov0]")
    assert _after(args, "-g") == "144"


@pytest.mark.parametrize(("rate", "stream_rate", "left_out"), [
    ("30000/1001", "30000/1001", False), ("30/1", "30/1", False), ("25/1", "25/1", True),
    ("30000/1001", "60000/1001", True), ("2997/100", "2997/100", True)])
def test_captions_at_the_field_rate_only_where_bwdif_re_times_them(
        rate: str, stream_rate: str, left_out: bool) -> None:
    # Elsewhere its fifo hands a frame's captions on with both fields'
    # frames, each pair twice: left out, the package carries none.
    plan = plan_renditions(_probe("mpeg2video", 720, 480, rate, field_order="tb",
                                  r_frame_rate=stream_rate),
                           parse_ladder(""), CPU_ENCODERS, fields=FIELDS, deinterlace="field")
    assert ff.captions_at_field_rate(plan) is not left_out
    assert ff.fields_chain(plan).endswith(",sidedata=mode=delete:type=A53_CC") is left_out
    frame = plan_renditions(_probe("mpeg2video", 720, 480, rate, field_order="tb"),
                            parse_ladder(""), CPU_ENCODERS, fields=FIELDS, deinterlace="frame")
    assert "sidedata" not in ff.fields_chain(frame)


@pytest.mark.parametrize(("rate", "film"), [
    ("30000/1001", "24000/1001"), ("2997/100", "24000/1001"), ("30/1", "24/1"),
    ("7180812/239593", "24000/1001")])
def test_an_inverse_telecine_is_at_the_film_rate_the_caption_fifo_knows(rate: str,
                                                                        film: str) -> None:
    plan = plan_renditions(_probe("mpeg2video", 720, 480, rate, field_order="tb"),
                           parse_ladder(""), CPU_ENCODERS, fields=TELECINE)
    assert (plan.made_progressive, plan.frame_rate) == ("inverse_telecine", film)
    assert ff.fields_graph(plan, "0:0", "fout")[-1] == (
        f"[cck][ccd]interleave,fps={film},settb=AVTB,setpts=PTS+0[fout]")


@pytest.mark.parametrize("fields", [FILM_22, MIXED])
def test_film_at_its_rate_and_mixed_are_field_matched_without_decimate(fields: Fields) -> None:
    _, args, _ = _cmd(_probe("mpeg2video", 720, 576, field_order=fields.mark), "", fields,
                      encoders=CPU_ENCODERS)
    graph = _after(args, "-filter_complex")
    o = fields.order
    assert graph == (f"[0:0]fieldmatch=order={o}:combmatch=full,"
                     f"bwdif=mode=send_frame:parity={o}:deint=interlaced,idet,"
                     f"bwdif=mode=send_frame:parity={o}:deint=interlaced,format=yuv420p[ov0]")
    assert "decimate" not in graph and "parity=auto" not in graph


@pytest.mark.parametrize("fields", [AS_FOUND, UNMARKED])
def test_whole_frames_marked_as_fields_or_not_marked_are_encoded_marked_progressive(
        fields: Fields) -> None:
    _, args, _ = _cmd(_probe("mpeg2video", 720, 576, field_order=fields.mark), "", fields,
                      encoders=CPU_ENCODERS)
    assert _after(args, "-filter_complex") == "[0:0]setfield=prog,format=yuv420p[ov0]"
    assert "bwdif" not in " ".join(args)


def test_a_progressive_source_has_no_fields_filter() -> None:
    progressive = Fields(found="progressive", by="pictures", mark="progressive", frames=1000,
                         progressive=1000)
    _, args, _ = _cmd(_probe("mpeg2video", 720, 576, field_order="progressive"), "",
                      progressive, encoders=CPU_ENCODERS)
    assert "-filter_complex" not in args


def test_a_source_that_may_hold_fields_is_never_decoded_on_the_gpu() -> None:
    probe = _probe("h264", 1920, 1080, field_order="")
    settings = EncodeSettings(ladder=tuple(parse_ladder("")), encoders=NVENC_ENCODERS,
                              decoder="cuda")
    unmarked = Fields(**{**FIELDS.__dict__, "mark": "", "order_by": "pictures"})
    plan = plan_renditions(probe, parse_ladder(""), NVENC_ENCODERS, fields=unmarked)
    assert plan.source.cuda_decode_blocker is None
    assert ff.cuda_decode_blocker(plan, settings) == "interlaced"
    with pytest.raises(TranscodeError, match="interlaced"):
        build_encode_command(SRC, INBOX, plan, settings, probe_subtitles=[], video_index=0,
                             decoder="cuda")
    for fields, why in ((TELECINE, "telecine"), (MIXED, "field_matched"),
                        (UNMARKED, "as_found")):
        p = plan_renditions(probe, parse_ladder(""), NVENC_ENCODERS,
                            fields=Fields(**{**fields.__dict__, "mark": ""}))
        assert ff.cuda_decode_blocker(p, settings) == why
    marked = plan_renditions(_probe("h264", 1920, 1080, field_order="tt"), parse_ladder(""),
                             NVENC_ENCODERS, fields=FIELDS)
    assert ff.cuda_decode_blocker(marked, settings) == "interlaced"


# --------------------------------------------------------------- the record
def test_the_record_says_what_was_done() -> None:
    plan = plan_renditions(_probe("mpeg2video", 720, 576, field_order="tb"), parse_ladder(""),
                           CPU_ENCODERS, fields=FIELDS)
    assert scan_record(plan) == {
        "found": "fields", "judgedBy": "pictures", "fieldOrder": "tb",
        "done": "deinterlaced-frame", "order": "tff", "orderBy": "mark", "orderSeen": True,
        "frameRate": "25/1", "filter": "bwdif=mode=send_frame:parity=tff:deint=all",
        "idet": {"frames": 1000, "tff": 990, "bff": 0, "progressive": 0, "undetermined": 10,
                 "repeatedTop": 0, "repeatedBottom": 0, "matchedCombed": 990,
                 "matchedDetermined": 990, "matchedRun": 0, "cadenceBlocks": 0,
                 "otherBlocks": 0, "blindBlocks": 0,
                 "windows": {"film": 0, "video": 0, "whole": 0, "unsure": 0, "detail": 0}},
        "verified": None,
    }
    contract = build_contract("i", plan, [], CPU_ENCODERS, duration_ms=60_000)
    assert contract["scan"] == scan_record(plan)
    assert contract["version"] == 1
    field = plan_renditions(_probe("mpeg2video", 720, 576, field_order="tb"), parse_ladder(""),
                            CPU_ENCODERS, fields=FIELDS, deinterlace="field")
    rec = scan_record(field)
    assert (rec["done"], rec["frameRate"], rec["filter"]) == (
        "deinterlaced-field", "50/1",
        "bwdif=mode=send_field:parity=tff:deint=all,sidedata=mode=delete:type=A53_CC")


def test_the_record_of_film_mixed_and_whole_frames() -> None:
    tc = plan_renditions(_probe("mpeg2video", 720, 480, NTSC, field_order="tb"),
                         parse_ladder(""), CPU_ENCODERS, fields=TELECINE)
    rec = scan_record(tc)
    assert (rec["found"], rec["done"], rec["frameRate"]) == (
        "telecine", "inverse-telecine", "24000/1001")
    assert rec["filter"].endswith(",decimate=dupthresh=4")
    mixed = scan_record(plan_renditions(_probe("mpeg2video", 720, 480, NTSC, field_order="tb"),
                                        parse_ladder(""), CPU_ENCODERS, fields=MIXED))
    assert (mixed["found"], mixed["done"], mixed["frameRate"]) == (
        "mixed", "field-matched", "30000/1001")
    copied = plan_renditions(_probe("hevc", 1920, 1080, field_order="tb"), parse_ladder(""),
                             NVENC_ENCODERS, fields=AS_FOUND)
    rec = scan_record(copied)
    # Whole frames by the sample's windows alone: not "progressive" until
    # the whole title is read through and bears it out.
    assert (rec["found"], rec["done"], rec["frameRate"], rec["filter"]) == (
        "unexamined", "progressive", "25/1", None)
    read = scan_record(replace(copied, verification=FieldsVerification(ok=True, frames=1500)))
    assert read["found"] == "progressive"
    short = scan_record(replace(copied, verification=FieldsVerification(
        ok=False, why="1200 of 1500 frames read through", frames=1200)))
    assert short["found"] == "unexamined"
    failed = scan_record(plan_renditions(_probe("mpeg4", 720, 576, field_order=""),
                                         parse_ladder(""), CPU_ENCODERS,
                                         fields=judge_fields("", [FieldsWindow()])))
    assert (failed["found"], failed["judgedBy"], failed["done"], failed["idet"]) == (
        "unknown", "failed", "unexamined", None)
    assert failed["filter"].startswith("fieldmatch=order=tff:")


def test_the_words_of_what_was_done() -> None:
    # No word lets an original go (ADR-0015); three say the package lost
    # nothing to its fields, each borne out by the whole title read through.
    assert SCAN_DONE == ("inverse-telecine", "progressive", "deinterlaced-field",
                         "deinterlaced-frame", "field-matched", "fields-kept", "fields-left",
                         "unexamined", "none")
    assert SCAN_RELEASES == ("inverse-telecine", "progressive", "deinterlaced-field")


def test_the_record_of_a_source_nothing_examined() -> None:
    plan = plan_renditions(_probe("mpeg2video", 720, 576, field_order="progressive"),
                           parse_ladder(""), CPU_ENCODERS)
    rec = scan_record(plan)
    assert (rec["found"], rec["judgedBy"], rec["done"], rec["filter"], rec["idet"]) == (
        "unexamined", "mark", "none", None, None)


def test_the_step_details_say_it() -> None:
    probe = _probe("mpeg2video", 720, 576, field_order="tb")
    cases = [
        (FIELDS, "field", "scan=fields:pictures:deinterlaced-field@50/1 "),
        (FIELDS, "frame", "scan=fields:pictures:deinterlaced-frame@25/1 "),
        (AS_FOUND, "field", "scan=unexamined:pictures:progressive@25/1 "),
        (MIXED, "field", "scan=mixed:pictures:field-matched@25/1 "),
        (FILM_22, "field", "scan=film:pictures:inverse-telecine@25/1 "),
    ]
    for fields, rate, details in cases:
        plan = plan_renditions(probe, parse_ladder(""), CPU_ENCODERS, fields=fields,
                               deinterlace=rate)
        assert _scan_details(plan) == details
    tc = plan_renditions(_probe("mpeg2video", 720, 480, NTSC, field_order="tb"),
                         parse_ladder(""), CPU_ENCODERS, fields=TELECINE)
    assert _scan_details(tc) == "scan=telecine:pictures:inverse-telecine@24000/1001 "
    pictures = plan_renditions(_probe("hevc", 1920, 540, "50/1", field_order="progressive"),
                               parse_ladder(""), NVENC_ENCODERS,
                               fields=Fields(found="field_pictures", by="bitstream",
                                             mark="progressive"))
    assert _scan_details(pictures) == "scan=field-pictures:bitstream:fields-kept@50/1 "
    failed = plan_renditions(_probe("mpeg4", 720, 576, field_order=""), parse_ladder(""),
                             CPU_ENCODERS, fields=judge_fields("", [FieldsWindow()]))
    assert _scan_details(failed) == "scan=unknown:failed:unexamined@25/1 "
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
FRAME = ("frame:0 pts:0 pts_time:0\nlavfi.idet.repeated.current_frame=neither\n"
         "lavfi.idet.single.current_frame={v}\nlavfi.idet.multiple.current_frame={v}\n")


def _fake_ffmpeg(raw: str, matched: str, runs: list[list[str]], trace: str = ""):
    """subprocess.run as ffmpeg answers the examination: idet's verdict `raw`
    of every frame as it is and `matched` after fieldmatch, written to the
    files the metadata filters name; trace_headers' `trace`."""
    def run(args, **_kw):
        runs.append(args)
        if "-filter_complex" in args:
            graph = args[args.index("-filter_complex") + 1]
            for target in re.findall(r"file=([^\[\],;]+)", graph):
                v = raw if target.endswith("raw.txt") else matched
                Path(target).write_text(FRAME.format(v=v) * 200)
            return subprocess.CompletedProcess(args, 0, "", "")
        return subprocess.CompletedProcess(args, 0, "", trace)
    return run


def test_the_examination_reads_five_windows_of_the_middle_of_the_picture(
        monkeypatch: pytest.MonkeyPatch) -> None:
    runs: list[list[str]] = []
    monkeypatch.setattr(ff.subprocess, "run", _fake_ffmpeg("tff", "tff", runs))
    probe = _probe("mpeg2video", 720, 576, field_order="tb")
    probe["video_index"] = 3
    f = ff.examine_fields(SRC, probe)
    assert (f.found, f.by, f.frames, f.tff, f.matched_combed) == (
        "fields", "pictures", 1000, 1000, 1000)
    assert [a[a.index("-ss") + 1] for a in runs] == ["6.000", "18.000", "30.000", "42.000",
                                                     "54.000"]
    for a in runs:
        graph = a[a.index("-filter_complex") + 1]
        assert graph.startswith(
            "[0:3]crop=w=trunc(iw*0.8/4)*4:h=trunc(ih*0.7/4)*4:x=trunc(iw*0.1/4)*4"
            ":y=trunc(ih*0.15/4)*4,split=3[raw][tff][bff];[raw]idet,")
        assert "[tff]fieldmatch=order=tff:combmatch=full,idet," in graph
        assert "[bff]fieldmatch=order=bff:combmatch=full,idet," in graph
        assert a.count("-frames:v") == 3 and "-hwaccel" not in a


def test_a_film_is_found_by_what_fieldmatch_puts_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ff.subprocess, "run", _fake_ffmpeg("bff", "progressive", []))
    f = ff.examine_fields(SRC, _probe("mpeg2video", 720, 576, field_order="bt"))
    assert (f.found, f.order, f.residue) == ("film", "bff", 0.0)


def test_an_examination_without_a_duration_reads_nothing_and_fails_safe(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # Without a duration only the start could be read: no verdict on the rest.
    monkeypatch.setattr(ff.subprocess, "run", lambda *a, **k: pytest.fail("read a window"))
    probe = _probe("mpeg2video", 720, 576, field_order="")
    probe["duration_ms"] = 0
    with capture_logs() as logs:
        f = ff.examine_fields(SRC, probe)
    assert (f.found, f.by, f.order) == ("unknown", "failed", "tff")
    assert [e["event"] for e in logs] == ["transcoder.fields.no_duration"]


def test_a_window_short_of_its_frames_before_the_end_fails_the_examination(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # 200 frames are there to be read from 30 s of 60: one read gives 120.
    runs: list[list[str]] = []
    full = _fake_ffmpeg("progressive", "progressive", runs)

    def run(args, **kw):
        result = full(args, **kw)
        if len(runs) == 3:
            graph = args[args.index("-filter_complex") + 1]
            for target in re.findall(r"file=([^\[\],;]+)", graph):
                Path(target).write_text(FRAME.format(v="progressive") * 120)
        return result

    monkeypatch.setattr(ff.subprocess, "run", run)
    f = ff.examine_fields(SRC, _probe("mpeg2video", 720, 576, field_order="tb"))
    assert (f.found, f.by) == ("unknown", "failed")


def test_the_windows_lie_across_the_picture_and_its_end_is_read_to_a_frame_or_two(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # The audio runs 10 s past a 60 s picture (its DURATION tag): the windows
    # start at 10 .. 90 % of the picture's 60 s, and the last, 6 s from its
    # end (150 frames at 25), may read two frames short of what is left.
    starts: list[float] = []

    def window(_path, _vin, start, frames=200):
        starts.append(start)
        left = round((60 - start) * 25)
        return _w("p" * min(frames, left - 2)), None

    monkeypatch.setattr(ff, "_idet_window", window)
    probe = _probe("mpeg2video", 720, 576, field_order="tb")
    probe["duration_ms"] = 70_000
    probe["video"]["tags"] = {"DURATION": "00:01:00.000000000"}
    f = ff.examine_fields(SRC, probe)
    assert starts == pytest.approx([6.0, 18.0, 30.0, 42.0, 54.0])
    assert (f.found, f.by) == ("progressive", "pictures")


def test_an_unreadable_picture_size_fails_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ff.subprocess, "run", lambda *a, **k: pytest.fail("ran ffmpeg"))
    f = ff.examine_fields(SRC, _probe("hevc", 0, 0, field_order="progressive"))
    assert (f.found, f.by) == ("unknown", "failed")


def test_a_source_not_examined_runs_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ff.subprocess, "run", lambda *a, **k: pytest.fail("ran ffmpeg"))
    f = ff.examine_fields(SRC, _probe("h264", 3840, 2160, field_order="progressive"))
    assert (f.found, f.by, f.frames) == ("progressive", "mark", 0)


SPS = "[trace_headers @ 0x1] 40          sps_seq_parameter_set_id    1 = 0\n"


@pytest.mark.parametrize(("trace", "raw", "found"), [
    (SPS + "[trace_headers @ 0x1] 265         field_seq_flag      1 = 1\n", "progressive",
     "field_pictures"),
    # Field pictures whatever idet reads of frames half as tall: never made
    # over and released.
    (SPS + "[trace_headers @ 0x1] 265         field_seq_flag      1 = 1\n", "tff",
     "field_pictures"),
    (SPS + "[trace_headers @ 0x1] 259         field_seq_flag      0 = 0\n", "progressive",
     "progressive"),
    # A sequence parameter set without the VUI that carries the flag: not told.
    (SPS, "progressive", "unknown"),
])
def test_hevc_field_pictures_by_their_parameter_sets(monkeypatch: pytest.MonkeyPatch, trace: str,
                                                     raw: str, found: str) -> None:
    # A stream that says field pictures but holds woven frames (idet finds
    # them in fields) is made progressive as any other.
    runs: list[list[str]] = []
    monkeypatch.setattr(ff.subprocess, "run", _fake_ffmpeg(raw, raw, runs, trace))
    f = ff.examine_fields(SRC, _probe("hevc", 720, 576, field_order="progressive"))
    assert f.found == found
    traced = [a for a in runs if "trace_headers" in a]
    assert len(traced) == 1 and traced[0][traced[0].index("-c") + 1] == "copy"


def test_the_field_pictures_read_is_bounded_on_the_input(monkeypatch: pytest.MonkeyPatch) -> None:
    # -frames:v 1 on the output did not stop ffmpeg 7.1 from tracing nearly
    # every packet of a feature-length file: the read is bounded before -i.
    runs: list[list[str]] = []
    monkeypatch.setattr(ff.subprocess, "run", _fake_ffmpeg("progressive", "progressive", runs,
                                                           SPS))
    ff.hevc_field_pictures(SRC, "0:0")
    [args] = runs
    assert args.index("-t") < args.index("-i")
    assert args[args.index("-t") + 1] == str(ff.FIELD_PICTURES_READ_SECONDS)
    assert "-frames:v" not in args


@pytest.mark.parametrize("outcome", ["timeout", "error", "no sps", "no start"])
def test_a_field_pictures_check_that_cannot_tell_fails_safe(
        monkeypatch: pytest.MonkeyPatch, outcome: str) -> None:
    # A large remux that runs past the time, a run that fails, or one that
    # sees no sequence parameter set is no "not field pictures": the source
    # gets the safe chain and a keeping word, whatever its mark or size.
    def run(args, **_kw):
        if "trace_headers" not in args:
            pytest.fail("examined after an unknown field-pictures check")
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(args, ff.FIELD_PICTURES_TIMEOUT)
        if outcome == "no start":
            raise OSError("ffmpeg not found")
        return subprocess.CompletedProcess(args, 1 if outcome == "error" else 0, "",
                                           "" if outcome == "no sps" else SPS)

    monkeypatch.setattr(ff.subprocess, "run", run)
    assert ff.hevc_field_pictures(SRC, "0:0") is None
    for height in (1080, 2160):
        probe = _probe("hevc", 1920, height, field_order="progressive")
        with capture_logs() as logs:
            f = ff.examine_fields(SRC, probe)
        assert (f.found, f.by) == ("unknown", "failed")
        assert "transcoder.fields.field_pictures_unknown" in [e["event"] for e in logs]
        plan = plan_renditions(probe, parse_ladder(""), NVENC_ENCODERS, fields=f)
        [rung] = plan.rungs
        assert (rung.mode, plan.scan_done) == ("encode", "unexamined")
        assert ff.fields_chain(plan).startswith("fieldmatch=order=tff:combmatch=full,bwdif=")


def test_a_window_that_fails_fails_the_examination_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    # Four windows read whole frames; the fifth times out. What the four say
    # is no verdict on the part the fifth would have read.
    runs: list[list[str]] = []
    whole = _fake_ffmpeg("progressive", "progressive", runs)

    def run(args, **kw):
        if len(runs) == 4:
            runs.append(args)
            raise subprocess.TimeoutExpired(args, ff.FIELDS_WINDOW_TIMEOUT)
        return whole(args, **kw)

    monkeypatch.setattr(ff.subprocess, "run", run)
    f = ff.examine_fields(SRC, _probe("mpeg2video", 720, 576, field_order="tb"))
    assert (f.found, f.by, f.frames) == ("unknown", "failed", 800)
    # A window past the end that exits 0 having read nothing is blind, no
    # failure.
    assert judge_fields("tb", [_w("p" * 200)] * 4 + [FieldsWindow()]).found == "progressive"
    assert judge_fields("tb", [_w("p" * 200)] * 4 + [FieldsWindow(failed=True)]).found == (
        "unknown")


# ------------------------------------------------------------ the stored mark
def test_the_worker_hands_the_stored_field_order_to_the_examination(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from transcoder import worker
    from transcoder.katalog import ClaimedItem

    source = tmp_path / "copy.mp4"
    source.write_bytes(b"\0")
    item = ClaimedItem.from_json({"id": "0f1e2d3c-0000-4000-8000-0000000000f2", "type": "movie",
                                  "title": "t", "path": str(source),
                                  "sourceFieldOrder": " TB "})
    assert item.field_order == "tb"
    assert ClaimedItem.from_json({"id": "i", "type": "movie", "path": "/x"}).field_order is None
    marks: list[str | None] = []

    def examine(_path, _probe, mark=None):
        marks.append(mark)
        return fields_by_mark(mark or "")

    monkeypatch.setattr(worker, "ffprobe", lambda _p: _probe("h264", 1920, 1080,
                                                             field_order="unknown"))
    monkeypatch.setattr(worker, "examine_fields", examine)
    monkeypatch.setattr(worker, "run_encode", lambda *_a, **_k: ([], 0.0))

    class Steps:
        def upsert_step(self, *_a, **_k) -> None:
            pass

    worker._process_one(item, Steps(), tmp_path / "inbox", EncodeSettings(encoders=CPU_ENCODERS))
    assert marks == ["tb"]
