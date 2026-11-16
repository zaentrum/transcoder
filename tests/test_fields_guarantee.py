"""The guarantee, stated as tests: a word that says the package lost nothing of
its source to its fields and holds none (SCAN_RELEASES) stands only where the
package has no combing left and no dropped (or doubled) moment. The sample may
choose the chain, never the word: the whole title, read through as it is
encoded, decides that; where it does not bear the word out, the title is made
again with the chain that is safe for what it holds (Plan.remade_done), and
its record keeps what the sample said (test_fields_real.py reads every moment
of made clips). No word lets an original go (ADR-0015)."""
from __future__ import annotations

from pathlib import Path

import pytest

from transcoder import ffmpeg as ff
from transcoder.decision import (
    CPU_ENCODERS,
    SCAN_RELEASES,
    FieldsWindow,
    judge_fields,
    parse_ladder,
    plan_renditions,
)

WORDS = {"t": "tff", "b": "bff", "p": "progressive", "u": "undetermined"}
REPS = {"n": "neither", "t": "top", "b": "bottom"}


def _w(single, repeated=None, matched=None):
    repeated = repeated if repeated is not None else "n" * len(single)
    matched = matched if matched is not None else single
    m = tuple(WORDS[c] for c in matched)
    return FieldsWindow(tuple(WORDS[c] for c in single), tuple(REPS[c] for c in repeated),
                        {"tff": m, "bff": m})


def _telecine(blocks=40):
    return _w("ptttt" * blocks, "nntnb" * blocks, "p" * (5 * blocks))


def _probe(codec, w, h, rate="25/1", **video):
    return {"video": {"codec_name": codec, "width": w, "height": h, "pix_fmt": "yuv420p",
                      "profile": "Main", "avg_frame_rate": rate, **video},
            "video_index": 0, "audio": [], "subtitles": [], "duration_ms": 60_000}


def _done(fields, codec="mpeg2video", rate="25/1", mark="tb"):
    plan = plan_renditions(_probe(codec, 720, 576, rate, field_order=mark), parse_ladder(""),
                           CPU_ENCODERS, fields=fields)
    return plan.scan_done


WHOLE = _w("p" * 200)


# 1. A window that is clearly video, below the whole sample's share --------------
def test_one_window_in_fields_below_the_samples_share_is_not_released():
    # 45 of its 200 frames TFF (22 %, every one top first), 4.5 % of the sample.
    f = judge_fields("tb", [WHOLE] * 4 + [_w("t" * 45 + "p" * 155)])
    assert _done(f) not in SCAN_RELEASES, (f.found, f.unsure_windows, f.video_windows)


def test_a_video_window_in_a_source_marked_progressive_is_not_released():
    # One window of woven video (190 frames TFF), 19 % of the sample.
    f = judge_fields("progressive", [WHOLE] * 4 + [_w("t" * 190 + "p" * 10, None, "tp" * 100)])
    assert _done(f, mark="progressive") not in SCAN_RELEASES, (f.found, f.video_windows)


def test_combing_in_one_order_in_every_window_marked_progressive_is_not_released():
    # 15 % of every window combed, all TFF: too few to make it over, never
    # "whole frames".
    f = judge_fields("progressive", [_w("t" * 30 + "p" * 170)] * 5)
    assert _done(f, mark="progressive") not in SCAN_RELEASES, f.found


# 2. Fail-safe: a window that failed, one window only, a stored mark ---------------
def _fake(windows_by_call):
    calls = {"n": 0}

    def window(_path, _vin, _start, frames=200):
        i = calls["n"]
        calls["n"] += 1
        w = windows_by_call[i]
        return (w, None) if w is not None else (FieldsWindow(), "timeout after 300 s")
    return window


def test_a_window_that_timed_out_is_no_verdict(monkeypatch):
    # The window that would have seen the video failed; the other four are whole.
    monkeypatch.setattr(ff, "_idet_window", _fake([WHOLE, WHOLE, None, WHOLE, WHOLE]))
    f = ff.examine_fields(Path("/x.mkv"), _probe("mpeg2video", 720, 576, field_order="tb"))
    assert _done(f) not in SCAN_RELEASES, (f.found, f.by, f.frames)


def test_four_windows_that_failed_are_no_verdict(monkeypatch):
    monkeypatch.setattr(ff, "_idet_window", _fake([None, None, None, None, WHOLE]))
    f = ff.examine_fields(Path("/x.mkv"), _probe("mpeg2video", 720, 576, field_order="tb"))
    assert _done(f) not in SCAN_RELEASES, (f.found, f.by, f.frames)


def test_one_window_from_the_start_is_no_verdict(monkeypatch):
    # No duration known: 200 frames of the start (a logo, a title card) decide.
    monkeypatch.setattr(ff, "_idet_window", _fake([WHOLE]))
    probe = _probe("mpeg2video", 720, 576, field_order="tb")
    probe["duration_ms"] = 0
    f = ff.examine_fields(Path("/x.mkv"), probe)
    assert _done(f) not in SCAN_RELEASES, (f.found, f.frames)


def test_a_stored_progressive_mark_does_not_weaken_the_streams_mark_of_fields(monkeypatch):
    # The stream says tt; the stored record says progressive. 15 % of the sample
    # combed in one order: with the stream's mark that is "mixed" (kept).
    monkeypatch.setattr(ff, "_idet_window", _fake([_w("t" * 30 + "p" * 170)] * 5))
    probe = _probe("mpeg2video", 720, 576, field_order="tt")
    f = ff.examine_fields(Path("/x.mkv"), probe, mark="progressive")
    assert _done(f, mark="tt") not in SCAN_RELEASES, (f.found, f.mark)


# 3. A 3:2 film the rate string does not name -------------------------------------
@pytest.mark.parametrize("rate", ["2997/100", "60000/1001", "1000000/33367"])
def test_a_32_film_at_a_rate_spelled_otherwise_is_not_released_undecimated(rate):
    f = judge_fields("tb", [_telecine()] * 5, rate)
    done = _done(f, rate=rate)
    plan = plan_renditions(_probe("mpeg2video", 720, 480, rate, field_order="tb"),
                           parse_ladder(""), CPU_ENCODERS, fields=f)
    chain = ff.fields_chain(plan) or ""
    # Either decimated (inverse telecine in fact) or a keeping word: never a
    # releasing word over a package with one frame of every five doubled.
    assert "decimate" in chain or done not in SCAN_RELEASES, (f.found, done, chain)


# 4. HEVC field pictures whose half-height frames idet reads as fields --------------
def test_hevc_field_pictures_are_never_deinterlaced_and_released(monkeypatch):
    monkeypatch.setattr(ff, "hevc_field_pictures", lambda *_a: True)
    monkeypatch.setattr(ff, "_idet_window", _fake([_w("t" * 200, None, "tp" * 100)] * 5))
    probe = _probe("hevc", 720, 288, "50/1", field_order="progressive")
    f = ff.examine_fields(Path("/x.mkv"), probe)
    plan = plan_renditions(probe, parse_ladder(""), CPU_ENCODERS, fields=f, deinterlace="field")
    assert plan.scan_done not in SCAN_RELEASES, (f.found, plan.scan_done)


# 5. The whole title, read through ---------------------------------------------------
def _plan(fields, rate="25/1", mark="tb", deinterlace="frame"):
    return plan_renditions(_probe("mpeg2video", 720, 576, rate, field_order=mark),
                           parse_ladder(""), CPU_ENCODERS, fields=fields, deinterlace=deinterlace)


def _verified(plan, raw, repeated=None, matched=None, out=None, kept=None):
    from dataclasses import replace

    from transcoder.decision import verify_title
    repeated = repeated if repeated is not None else "n" * len(raw)
    v = verify_title(plan, [WORDS[c] for c in raw], [REPS[c] for c in repeated],
                     None if matched is None else [WORDS[c] for c in matched],
                     None if out is None else [WORDS[c] for c in out], kept)
    return replace(plan, verification=v)


def test_whole_frames_with_fields_between_the_windows_are_not_released():
    plan = _plan(judge_fields("tb", [WHOLE] * 5))
    assert plan.sample_done == "progressive" and plan.needs_verification
    # 76 woven frames between two windows: the read-through sees them.
    title = "p" * 1400 + "t" * 76 + "p" * 600
    assert _verified(plan, title, out=title).scan_done == "field-matched"
    assert _verified(plan, "p" * 2076, out="p" * 2076).scan_done == "progressive"


def test_a_32_film_whose_cadence_breaks_between_the_windows_is_not_released():
    plan = _plan(judge_fields("tb", [_telecine()] * 5, "30000/1001"), "30000/1001")
    assert plan.sample_done == "inverse-telecine"
    film = "ptttt" * 200
    reps = "nntnb" * 200
    # A video part between the windows: combed, no repeats.
    v_raw, v_rep = film + "t" * 150 + film, reps + "n" * 150 + reps
    v_matched = "p" * 1000 + "tp" * 75 + "p" * 1000
    out = "p" * 1720
    assert _verified(plan, v_raw, v_rep, v_matched, out, 1720).scan_done == "field-matched"
    clean = _verified(plan, film * 2, reps * 2, "p" * 2000, out[:1600], 1600)
    assert clean.scan_done == "inverse-telecine"
    # Kept only where every frame decimate dropped was a near-duplicate: its
    # twin in mixed mode kept 30 frames more, or its drops were not read.
    thinned = _verified(plan, film * 2, reps * 2, "p" * 2000, out[:1600], 1630)
    assert (thinned.scan_done, thinned.verification.dropped_distinct) == ("field-matched", 30)
    assert _verified(plan, film * 2, reps * 2, "p" * 2000, out[:1600]).scan_done == (
        "field-matched")


def test_video_in_the_other_order_between_the_windows_is_not_released_at_the_field_rate():
    f = judge_fields("tb", [_w("t" * 200)] * 5)
    plan = _plan(f, deinterlace="field")
    assert plan.sample_done == "deinterlaced-field"
    raw = "t" * 1400 + "b" * 200 + "t" * 400
    assert _verified(plan, raw, out="p" * 4000).scan_done == "deinterlaced-frame"
    assert _verified(plan, "t" * 2000, out="p" * 4000).scan_done == "deinterlaced-field"


def test_a_combed_output_is_never_released():
    plan = _plan(judge_fields("tb", [_w("t" * 200)] * 5), deinterlace="field")
    out = "p" * 1990 + "tt" + "p" * 8
    assert _verified(plan, "t" * 1000, out=out).scan_done == "field-matched"


def test_a_film_at_its_own_rate_that_repeats_fields_is_not_released():
    # A euro pulldown: whole once matched, but a field repeated every 12 frames.
    f = judge_fields("tb", [_w("t" * 200, None, "p" * 200)] * 5)
    plan = _plan(f)
    assert plan.sample_done == "inverse-telecine"
    euro = ("n" * 11 + "t") * 100
    assert _verified(plan, "t" * 1200, euro, "p" * 1200, "p" * 1200).scan_done == (
        "field-matched")
    f_euro = judge_fields("tb", [_w("t" * 200, ("n" * 11 + "t") * 17 + "n" * 32, "p" * 200)] * 5)
    assert _plan(f_euro).sample_done not in SCAN_RELEASES


def test_a_title_not_read_through_says_so_and_a_decimated_one_is_made_again():
    from dataclasses import replace

    from transcoder.decision import FieldsVerification
    from transcoder.renditions import scan_record
    plan = _plan(judge_fields("tb", [WHOLE] * 5))
    missing = replace(plan, verification=FieldsVerification(ok=False, why="not read through"))
    # Whole frames as found, not read through: what was done, and that the
    # read-through did not bear it out, side by side in the record.
    rec = scan_record(missing)
    assert (rec["done"], rec["verified"]["ok"], rec["verified"]["sampleDone"]) == (
        "progressive", False, "progressive")
    # A decimate stands only where its output was read: made again.
    tc = _plan(judge_fields("tb", [_telecine()] * 5, "30000/1001"), "30000/1001")
    gone = replace(tc, verification=FieldsVerification(ok=False, why="not read through"))
    assert (gone.sample_done, gone.scan_done) == ("inverse-telecine", "field-matched")


def test_a_short_part_in_the_other_order_or_in_fields_is_seen_by_its_run():
    # 30 frames bottom first inside a chunk of top-first video (85 % top first
    # in that chunk): its run gives it away.
    plan = _plan(judge_fields("tb", [_w("t" * 200)] * 5), deinterlace="field")
    raw = "t" * 1270 + "b" * 30 + "t" * 700
    assert _verified(plan, raw, out="p" * 4000).scan_done == "deinterlaced-frame"
    # 8 woven frames in a row inside whole frames (under a chunk's share).
    whole = _plan(judge_fields("tb", [WHOLE] * 5))
    title = "p" * 1000 + "t" * 8 + "p" * 992
    assert _verified(whole, title, out=title.replace("t", "p")).scan_done == "field-matched"
    # A frame of fine detail here and there is no part in fields.
    sprinkled = ("p" * 99 + "t") * 20
    assert _verified(whole, sprinkled, out="p" * 2000).scan_done == "progressive"


def test_a_window_start_without_a_repeat_before_it_is_no_cadence_break():
    # idet sees no repeat in a window's first frame: its first block may show
    # one of its two repeats, which breaks no cadence.
    window = _w("ptttt" * 40, "nnnnb" + "nntnb" * 39, "p" * 200)
    f = judge_fields("tb", [window] * 5, "30000/1001")
    assert (f.found, f.other_blocks) == ("telecine", 0)


# ------------------------------------------------ the third review's findings
TC_SINGLE, TC_REP = "ptttt", "nntnb"     # a 3:2 film as idet reads it, block by block


def _ntsc(fields, deinterlace="frame"):
    return _plan(fields, "30000/1001", deinterlace=deinterlace)


def test_a_29_97p_part_of_a_telecined_film_is_never_decimated():
    # The last 150 frames 29.97p (credits rendered progressive): whole moving
    # frames, no repeat, which decimate would thin one in five.
    plan = _ntsc(judge_fields("tb", [_telecine()] * 5, "30000/1001"))
    assert plan.sample_done == "inverse-telecine"
    raw, rep = TC_SINGLE * 90 + "p" * 150, TC_REP * 90 + "n" * 150
    v = _verified(plan, raw, rep, "p" * 600, "p" * 480, kept=510)
    assert (v.scan_done, v.verification.dropped_distinct) == ("field-matched", 30)
    # Its blocks alone say so: whole frames with no repeat are off the cadence.
    assert _verified(plan, raw, rep, "p" * 600, "p" * 480, kept=480).scan_done == "field-matched"


def test_whole_windows_beside_a_telecined_film_are_never_decimated():
    f = judge_fields("tb", [_telecine()] * 4 + [WHOLE], "30000/1001")
    assert (f.found, _ntsc(f).sample_done) == ("film", "field-matched")


def test_a_cadence_moved_where_idet_saw_nothing_is_no_cadence():
    # Phase (2, 4) before a still stretch (idet leaves a still undetermined),
    # (0, 2) after it: at the cut decimate drops a film frame.
    plan = _ntsc(judge_fields("tb", [_telecine()] * 5, "30000/1001"))
    raw = TC_SINGLE * 36 + "u" * 70 + "tttpt" * 48
    rep = TC_REP * 36 + "n" * 70 + "tnbnn" * 48
    v = _verified(plan, raw, rep, "p" * len(raw), "p" * (len(raw) * 4 // 5),
                  kept=len(raw) * 4 // 5)
    assert v.scan_done == "field-matched" and v.verification.off_cadence >= 1


def test_whole_frames_or_repeats_in_video_are_not_kept_at_the_field_rate():
    # 75 frames of 25p between the windows of a 50i title: each of them would
    # be shown twice at 50, made from one field. And a 3:2 part in 60i.
    plan = _plan(judge_fields("tb", [_w("t" * 200)] * 5), deinterlace="field")
    assert plan.sample_done == "deinterlaced-field"
    raw = "t" * 362 + "p" * 75 + "t" * 1063
    v = _verified(plan, raw, out="p" * 3000)
    assert (v.scan_done, v.verification.why) == ("field-matched",
                                                  "whole frames at the field rate")
    tc = TC_SINGLE * 27
    v = _verified(plan, "t" * 400 + tc + "t" * 465, "n" * 400 + TC_REP * 27 + "n" * 465,
                  out="p" * 2000)
    assert (v.scan_done, v.verification.why) == ("field-matched",
                                                  "repeated fields at the field rate")


def test_a_remade_titles_record_keeps_what_the_sample_said(tmp_path):
    from transcoder.decision import CPU_ENCODERS as ENC
    from transcoder.ffmpeg import EncodeSettings
    from transcoder.katalog import ClaimedItem
    from transcoder.renditions import scan_record
    from transcoder.worker import _remade

    probe = _probe("mpeg2video", 720, 576, field_order="tb")
    settings = EncodeSettings(ladder=tuple(parse_ladder("")), encoders=ENC)
    item = ClaimedItem(id="i", type="movie", title="t", year=None, duration_ms=None,
                       path=str(tmp_path / "x.mkv"))
    plan = _plan(judge_fields("tb", [WHOLE] * 5))
    title = "p" * 1400 + "t" * 76 + "p" * 600
    refused = _verified(plan, title, out=title.replace("t", "p"))
    remade = _remade(refused, probe, item, settings)
    rec = scan_record(remade)
    assert (rec["found"], rec["done"]) == ("mixed", "field-matched")
    assert {k: rec["verified"][k] for k in ("ok", "why", "sampleDone", "sampleFound")} == {
        "ok": False, "why": "fields in the title", "sampleDone": "progressive",
        "sampleFound": "progressive"}
    # Made again once: a remade plan is not made again (the worker's second
    # look at its failed read-through).
    assert _remade(remade, probe, item, settings) is remade


def test_a_copys_read_through_counts_the_video_not_the_audio(tmp_path):
    # Audio 2 s past a 20 s picture at 25: the copy's 500 frames are all of it.
    from transcoder.ffmpeg import VERIFY_FILES, read_verification

    probe = _probe("h264", 720, 576, field_order="tb")
    probe["duration_ms"] = 22_000
    probe["video"]["tags"] = {"DURATION": "00:00:20.000000000"}
    plan = plan_renditions(probe, parse_ladder(""), CPU_ENCODERS,
                           fields=judge_fields("tb", [WHOLE] * 5))
    line = ("lavfi.idet.single.current_frame=progressive\n"
            "lavfi.idet.repeated.current_frame=neither\n")
    (tmp_path / VERIFY_FILES["raw"]).write_text(line * 500)
    v = read_verification(plan, tmp_path, None)
    assert v.ok, v.why
    (tmp_path / VERIFY_FILES["raw"]).write_text(line * 440)
    assert read_verification(plan, tmp_path, None).why == "440 of 500 frames read through"
