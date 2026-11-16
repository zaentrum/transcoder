"""Sources in fields, end to end on made clips (skipped without ffmpeg 7).

Each kind is made here with ffmpeg, never taken from a title:

  * true 50i: testsrc2 at 50 frames a second, woven into 25 frames of two
    fields each (tinterlace=mode=interleave_top), as MPEG-2 coded as fields;
  * 60i, bottom field first, as interlaced H.264 (the H.264 a CPU host used
    to pass through);
  * a telecined film: a 23.976p clip through telecine=pattern=23 (29.97);
  * whole frames marked as fields: a 25p clip through setfield=tff, as HEVC
    in Matroska marked tb, and as MPEG-2 coded as fields — a film transfer
    at 25 frames, with a dark logo of its own in fields in a corner and a
    fade to near black, where the logo is all idet sees;
  * mixed: 25p, then 50i woven; a telecined film, then 60i woven; and a
    telecined film whose last eighth is 60i video (a film with video parts,
    which a decimate would spoil and leave combed);
  * a telecined film cut every 2.5 s, each cut at another place of its
    cadence, and one half of which is a held picture;
  * a source marked progressive (MPEG-2 in MPEG-TS: its first picture is),
    2 s of 25p, then 50i woven;
  * HEVC coded as field pictures: 50i split into its fields
    (separatefields), each a picture of its own (libx265 interlace=tff);
  * clips whose every moment is numbered in a code strip, and clips whose
    every picture's closed captions are numbered pair by pair
    (tests/fieldcode.py), read back frame by frame and pair by pair.

The pictures of the progressive clips are testsrc2 drawn at twice their
height and scaled down, as a camera's picture is: testsrc2 as drawn has
detail one line high, which idet reads as combing in a progressive frame.

Every one runs through the worker's `_process_one` with the CPU encoders,
and the handoff is checked: the video's frame rate, frames and mark, idet's
verdict of the encode, and the record in renditions.json.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from transcoder.decision import CPU_ENCODERS, parse_ladder
from transcoder.ffmpeg import EncodeSettings, examine_fields, ffprobe
from transcoder.katalog import ClaimedItem
from transcoder.worker import _inbox_dir, _process_one


def _ffmpeg_major() -> int:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        return 0
    first = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True,
                           stdin=subprocess.DEVNULL).stdout.split("\n", 1)[0]
    m = re.search(r"version n?(\d+)\.", first)
    return int(m.group(1)) if m else 99


def _has(kind: str, name: str) -> bool:
    out = subprocess.run(["ffmpeg", "-hide_banner", f"-{kind}"], capture_output=True, text=True,
                         stdin=subprocess.DEVNULL).stdout
    return any(line.split()[1:2] == [name] for line in out.splitlines() if line.strip())


pytestmark = [
    pytest.mark.skipif(_ffmpeg_major() < 7,
                       reason="needs ffmpeg/ffprobe >= 7 (the image pins 7.1)"),
    pytest.mark.skipif(_ffmpeg_major() >= 7 and not all(
        _has("filters", f) for f in ("idet", "bwdif", "fieldmatch", "decimate", "telecine",
                                     "tinterlace", "setfield")), reason="needs ffmpeg's filters"),
    pytest.mark.skipif(_ffmpeg_major() >= 7 and not all(
        _has("encoders", e) for e in ("libx265", "libx264", "mpeg2video")),
        reason="needs libx265, libx264 and mpeg2video"),
]

ITEM_ID = "0f1e2d3c-0000-4000-8000-0000000000f1"
SECONDS = 4
# MPEG-2 coded as fields, top field first as setfield marks the frames
# (ffmpeg 9 takes no -top on an encoder).
MPEG2_FIELDS = ["-c:v", "mpeg2video", "-flags", "+ildct+ilme", "-q:v", "3"]


def picture(w: int, h: int, rate: str, seconds: int = SECONDS) -> str:
    """A camera's picture: testsrc2 drawn twice as tall, scaled down."""
    return f"testsrc2=s={w}x{h * 2}:r={rate}:d={seconds},scale={w}:{h}"


def _make(path: Path, inputs: list[str], vf: str, vcodec: list[str], *,
          complex_graph: bool = False, extra: list[str] | None = None,
          seconds: int = SECONDS) -> Path:
    args = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    for source in inputs:
        args += ["-f", "lavfi", "-i", source]
    args += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={seconds}"]
    audio = len(inputs)
    if complex_graph:
        args += ["-filter_complex", vf + "[v]", "-map", "[v]"]
    else:
        args += ["-map", "0:v", "-vf", vf]
    args += ["-map", f"{audio}:a", *vcodec, "-c:a", "aac", *(extra or []), str(path)]
    subprocess.run(args, check=True, stdin=subprocess.DEVNULL)
    return path


def _cut_film(path: Path) -> Path:
    """12 s of a telecined film edited after the pulldown: a shot every 2.5
    s, each telecined on its own and cut at another place of its cadence."""
    graph, names = [], []
    for n, k in enumerate((0, 3, 1, 4, 2)):
        graph.append(f"{picture(720, 480, '24000/1001', 3)},hue=h={n * 60},telecine=pattern=23,"
                     f"trim=start_frame={k}:end_frame={k + 75},setpts=N*1001/30000/TB[s{n}]")
        names.append(f"[s{n}]")
    graph.append("".join(names) + f"concat=n={len(names)}:v=1,setpts=N*1001/30000/TB,setfield=tff")
    return _make(path, [], ";".join(graph), MPEG2_FIELDS, complex_graph=True, seconds=13,
                 extra=["-r", "30000/1001"])


@pytest.fixture(scope="module")
def clips(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("fields")
    return {
        "50i": _make(d / "50i.mkv", [f"testsrc2=s=720x576:r=50:d={SECONDS}"],
                     "tinterlace=mode=interleave_top,setfield=tff", MPEG2_FIELDS),
        "60i_bff": _make(d / "60i.mkv", [picture(720, 480, "60000/1001")],
                         "tinterlace=mode=interleave_bottom,setfield=bff",
                         ["-c:v", "libx264", "-preset", "ultrafast", "-flags", "+ildct+ilme",
                          "-x264-params", "bff=1"]),
        "telecine": _make(d / "telecine.mkv", [picture(720, 480, "24000/1001")],
                          "telecine=pattern=23,setfield=tff", MPEG2_FIELDS),
        "hevc_marked": _make(d / "hevc_marked.mkv", [picture(1280, 720, "25")], "setfield=tff",
                             ["-c:v", "libx265", "-preset", "ultrafast",
                              "-x265-params", "log-level=error"], extra=["-field_order", "tb"]),
        "film_mpeg2": _make(
            d / "film.mkv",
            [picture(720, 576, "25") + f",fade=t=out:st={SECONDS - 1.5}:d=0.5:c=0x101010",
             f"testsrc2=s=96x48:r=50:d={SECONDS},hue=s=0,eq=brightness=-0.25,"
             "tinterlace=mode=interleave_top"],
            "[0][1]overlay=x=600:y=24:shortest=1,setfield=tff", MPEG2_FIELDS, complex_graph=True),
        "mixed": _make(d / "mixed.mkv",
                       [picture(720, 576, "25", SECONDS // 2),
                        picture(720, 576, "50", SECONDS // 2)],
                       "[1]tinterlace=mode=interleave_top[i];[0][i]concat=n=2:v=1,setfield=tff",
                       MPEG2_FIELDS, complex_graph=True),
        "hevc_field_pictures": _make(
            d / "hevc_fields.mkv", [picture(720, 576, "50")],
            "tinterlace=mode=interleave_top,setfield=tff,separatefields",
            ["-c:v", "libx265", "-preset", "ultrafast",
             "-x265-params", "log-level=error:interlace=tff"]),
        "film_85_video_15": _make(
            d / "film_85_video_15.mkv",
            [picture(720, 480, "24000/1001", 7), picture(720, 480, "60000/1001", 1)],
            "[0]telecine=pattern=23[f];[1]tinterlace=mode=interleave_top[i];"
            "[f][i]concat=n=2:v=1,setfield=tff", MPEG2_FIELDS, complex_graph=True,
            seconds=8),
        "telecine_cuts": _cut_film(d / "telecine_cuts.mkv"),
        "telecine_held": _make(
            d / "telecine_held.mkv",
            [picture(720, 480, "24000/1001", 2), picture(720, 480, "24000/1001", 1),
             picture(720, 480, "24000/1001", 2), picture(720, 480, "24000/1001", 1)],
            "[1]trim=end_frame=1,loop=loop=71:size=1,setpts=N/(24000/1001)/TB[h1];"
            "[3]trim=end_frame=1,loop=loop=71:size=1,setpts=N/(24000/1001)/TB[h3];"
            "[0][h1][2][h3]concat=n=4:v=1,fps=24000/1001,telecine=pattern=23,setfield=tff",
            MPEG2_FIELDS, complex_graph=True, seconds=10),
        "progressive_then_50i": _make(
            d / "progressive_then_50i.ts",
            [picture(720, 576, "25", 2), picture(720, 576, "50", 6)],
            "[1]tinterlace=mode=interleave_top[i];[0][i]concat=n=2:v=1",
            ["-c:v", "mpeg2video", "-q:v", "3"], complex_graph=True, seconds=8),
        "film_and_60i": _make(
            d / "film_and_60i.mkv",
            [picture(720, 480, "24000/1001", SECONDS // 2),
             picture(720, 480, "60000/1001", SECONDS // 2)],
            "[0]telecine=pattern=23[f];[1]tinterlace=mode=interleave_top[i];"
            "[f][i]concat=n=2:v=1,setfield=tff", MPEG2_FIELDS, complex_graph=True),
    }


class StubKatalog:
    def __init__(self) -> None:
        self.steps: list[tuple[str, dict]] = []

    def upsert_step(self, _item_id: str, status: str, **kw: object) -> None:
        self.steps.append((status, kw))


def _run(tmp_path: Path, src: Path, ladder: str = "", **settings) -> tuple[bool, StubKatalog, Path]:
    cfg = EncodeSettings(ladder=tuple(parse_ladder(ladder)), encoders=CPU_ENCODERS,
                         x264_preset="ultrafast", x265_preset="ultrafast", segment_seconds=2,
                         **settings)
    client = StubKatalog()
    item = ClaimedItem(id=ITEM_ID, type="movie", title="clip", year=None, duration_ms=None,
                       path=str(src))
    ok = _process_one(item, client, _inbox_dir(tmp_path / "packages", ITEM_ID), cfg)
    return ok, client, tmp_path / "packages" / "_inbox" / ITEM_ID


def _video(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                          "-show_entries", "stream=codec_name,avg_frame_rate,field_order,"
                          "nb_read_frames,width,height", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)["streams"][0]


def _idet(path: Path) -> dict[str, int]:
    """idet's single-frame verdicts over the whole of a file's video."""
    out = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-i", str(path),
                          "-map", "0:v:0", "-vf", "idet,metadata=mode=print:file=-",
                          "-f", "null", "-"], capture_output=True, text=True, check=True).stdout
    found = re.findall(r"lavfi\.idet\.single\.current_frame=(\w+)", out)
    return {k: found.count(k) for k in ("tff", "bff", "progressive", "undetermined")}


def _progressive_in_fact(path: Path, most: float = 0.02) -> None:
    """idet calls no more than `most` of the frames it determines TFF or
    BFF."""
    c = _idet(path)
    determined = c["tff"] + c["bff"] + c["progressive"]
    assert determined > 0
    assert c["tff"] + c["bff"] <= most * determined, c


def _contract(inbox: Path) -> dict:
    return json.loads((inbox / "renditions.json").read_text())


def test_each_kind_is_found_what_it_is(clips: dict[str, Path]) -> None:
    found = {}
    for name, path in clips.items():
        f = examine_fields(path, ffprobe(path))
        found[name] = (f.found, f.by, f.order)
    assert found == {
        "50i": ("fields", "pictures", "tff"),
        "60i_bff": ("fields", "pictures", "bff"),
        "telecine": ("telecine", "pictures", "tff"),
        "hevc_marked": ("progressive", "pictures", "tff"),
        "film_mpeg2": ("progressive", "pictures", "tff"),
        "mixed": ("fields", "pictures", "tff"),
        "film_85_video_15": ("mixed", "pictures", "tff"),
        "telecine_cuts": ("film", "pictures", "tff"),   # its cadence breaks: field-matched
        "telecine_held": ("telecine", "pictures", "tff"),
        "progressive_then_50i": ("fields", "pictures", "tff"),
        "film_and_60i": ("mixed", "pictures", "tff"),
        "hevc_field_pictures": ("field_pictures", "bitstream", "tff"),
    }


def test_50i_is_deinterlaced_at_its_field_rate(tmp_path: Path, clips: dict[str, Path]) -> None:
    src = clips["50i"]
    ok, client, inbox = _run(tmp_path, src, deinterlace="field")
    assert ok and client.steps[-1][0] == "done"
    details = str(client.steps[-1][1]["details"])
    assert "scan=fields:pictures:deinterlaced-field@50/1 " in details
    assert "src_codec=mpeg2video" in details
    out, before = _video(inbox / "prepared.mkv"), _video(src)
    assert (out["codec_name"], out["avg_frame_rate"], out["field_order"]) == (
        "hevc", "50/1", "progressive")
    assert int(out["nb_read_frames"]) == 2 * int(before["nb_read_frames"])
    _progressive_in_fact(inbox / "prepared.mkv")
    scan = _contract(inbox)["scan"]
    assert {k: scan[k] for k in ("found", "judgedBy", "fieldOrder", "done", "order",
                                 "frameRate", "filter")} == {
        "found": "fields", "judgedBy": "pictures", "fieldOrder": "tb",
        "done": "deinterlaced-field", "order": "tff", "frameRate": "50/1",
        # 25 frames: no captions handed on twice (README "Closed captions").
        "filter": "bwdif=mode=send_field:parity=tff:deint=all,sidedata=mode=delete:type=A53_CC"}
    assert scan["idet"]["tff"] > 0.9 * scan["idet"]["frames"]
    # The audio goes with it, from the same start.
    streams = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,start_time", "-of", "json",
         str(inbox / "prepared.mkv")], capture_output=True, text=True, check=True,
    ).stdout)["streams"]
    starts = {s["codec_type"]: float(s.get("start_time") or 0) for s in streams}
    assert starts["audio"] == pytest.approx(starts["video"], abs=0.03)


def test_50i_at_its_frame_rate_by_default(tmp_path: Path, clips: dict[str, Path]) -> None:
    # The default setting: one frame of every two fields.
    src = clips["50i"]
    ok, client, inbox = _run(tmp_path, src)
    assert ok and "scan=fields:pictures:deinterlaced-frame@25/1 " in str(
        client.steps[-1][1]["details"])
    out = _video(inbox / "prepared.mkv")
    assert (out["avg_frame_rate"], out["field_order"]) == ("25/1", "progressive")
    assert int(out["nb_read_frames"]) == int(_video(src)["nb_read_frames"])
    _progressive_in_fact(inbox / "prepared.mkv")
    scan = _contract(inbox)["scan"]
    assert (scan["done"], scan["frameRate"], scan["filter"]) == (
        "deinterlaced-frame", "25/1", "bwdif=mode=send_frame:parity=tff:deint=all")


def test_interlaced_h264_is_no_longer_passed_through(tmp_path: Path,
                                                      clips: dict[str, Path]) -> None:
    # Browser-friendly H.264, which a CPU host copied: in fields, it is
    # encoded, bottom field first as idet and its mark say, at 59.94.
    ok, client, inbox = _run(tmp_path, clips["60i_bff"], deinterlace="field")
    assert ok and client.steps[-1][0] == "done"
    details = str(client.steps[-1][1]["details"])
    assert "scan=fields:pictures:deinterlaced-field@60000/1001 " in details
    contract = _contract(inbox)
    [v0] = contract["video"]
    assert (v0["mode"], v0["encoder"]) == ("encode", "libx265")
    assert contract["scan"]["filter"] == "bwdif=mode=send_field:parity=bff:deint=all"
    _progressive_in_fact(inbox / "prepared.mkv")
    out = _video(inbox / "prepared.mkv")
    assert int(out["nb_read_frames"]) == 2 * int(_video(clips["60i_bff"])["nb_read_frames"])


def test_a_telecined_film_is_brought_back_to_its_frames(tmp_path: Path,
                                                         clips: dict[str, Path]) -> None:
    src = clips["telecine"]
    ok, client, inbox = _run(tmp_path, src)
    assert ok and "scan=telecine:pictures:inverse-telecine@24000/1001 " in str(
        client.steps[-1][1]["details"])
    out, before = _video(inbox / "prepared.mkv"), _video(src)
    assert (out["avg_frame_rate"], out["field_order"]) == ("24000/1001", "progressive")
    assert int(out["nb_read_frames"]) == pytest.approx(int(before["nb_read_frames"]) * 4 / 5, abs=1)
    _progressive_in_fact(inbox / "prepared.mkv")
    scan = _contract(inbox)["scan"]
    assert (scan["found"], scan["done"], scan["frameRate"]) == (
        "telecine", "inverse-telecine", "24000/1001")
    assert scan["idet"]["repeatedTop"] > 0 and scan["idet"]["repeatedBottom"] > 0


def test_whole_frames_marked_as_fields_are_copied_and_say_so(tmp_path: Path,
                                                              clips: dict[str, Path]) -> None:
    # HEVC Main marked tb, its frames whole: copied as before, and the
    # record of the copy says it was found progressive.
    ok, client, inbox = _run(tmp_path, clips["hevc_marked"])
    assert ok and client.steps[-1][0] == "not_applicable"
    details = str(client.steps[-1][1]["details"])
    assert "reason=source_already_hevc:hevc" in details
    assert "scan=progressive:pictures:progressive@25/1 " in details
    assert sorted(p.name for p in inbox.iterdir()) == ["renditions.json"]
    contract = _contract(inbox)
    [v0] = contract["video"]
    assert (v0["file"], v0["mode"], v0["carries"]) == (None, "copy",
                                                       ["video", "audio", "subtitles"])
    assert {k: contract["scan"][k] for k in ("found", "judgedBy", "fieldOrder", "done",
                                             "filter")} == {
        "found": "progressive", "judgedBy": "pictures", "fieldOrder": "tb",
        "done": "progressive", "filter": None}


def test_an_mpeg2_film_transfer_at_25_frames_is_left_as_it_is(tmp_path: Path,
                                                               clips: dict[str, Path]) -> None:
    # Progressive frames in an MPEG-2 stream coded and marked as fields, a
    # dark logo of its own in fields: encoded frame for frame, not
    # deinterlaced (bwdif would only soften it), and marked progressive.
    src = clips["film_mpeg2"]
    ok, client, inbox = _run(tmp_path, src)
    assert ok and client.steps[-1][0] == "done"
    assert "scan=progressive:pictures:progressive@25/1 " in str(
        client.steps[-1][1]["details"])
    out, before = _video(inbox / "prepared.mkv"), _video(src)
    assert (out["avg_frame_rate"], out["field_order"]) == ("25/1", "progressive")
    assert out["nb_read_frames"] == before["nb_read_frames"]
    scan = _contract(inbox)["scan"]
    assert (scan["done"], scan["filter"]) == ("progressive", "setfield=prog")
    assert scan["idet"]["tff"] + scan["idet"]["bff"] == 0


def test_a_short_whole_part_before_video_is_deinterlaced_with_it(
        tmp_path: Path, clips: dict[str, Path]) -> None:
    # 2 s of 25p, then 50i: every window of the sample sees the video, so it
    # is video, deinterlaced whole at the field rate in the order idet saw:
    # nothing combed, no moment dropped (the 25p part is shown twice a frame).
    ok, client, inbox = _run(tmp_path, clips["mixed"], deinterlace="field")
    assert ok and "scan=fields:pictures:deinterlaced-field@50/1 " in str(
        client.steps[-1][1]["details"])
    _progressive_in_fact(inbox / "prepared.mkv")
    assert _video(inbox / "prepared.mkv")["avg_frame_rate"] == "50/1"


@pytest.mark.parametrize("name", ["film_and_60i", "film_85_video_15"])
def test_a_film_with_video_parts_is_field_matched_never_decimated(
        tmp_path: Path, clips: dict[str, Path], name: str) -> None:
    # Field-matched at the source's rate, its video part deinterlaced, never
    # decimated (which would drop its motion and leave it combed), its record
    # a word that keeps the original: also where 85 % of it is film and its
    # cadence covers what idet sees of that.
    ok, client, inbox = _run(tmp_path, clips[name], deinterlace="field")
    assert ok and "scan=mixed:pictures:field-matched@30000/1001 " in str(
        client.steps[-1][1]["details"])
    scan = _contract(inbox)["scan"]
    assert (scan["done"], scan["frameRate"]) == ("field-matched", "30000/1001")
    assert "decimate" not in scan["filter"]
    assert int(_video(inbox / "prepared.mkv")["nb_read_frames"]) == int(
        _video(clips[name])["nb_read_frames"])
    # Frames idet marks only once its last few agree are left at a cut from
    # film to video (its record keeps the original).
    _progressive_in_fact(inbox / "prepared.mkv", most=0.12)


def test_a_telecined_film_with_held_pictures_is_inverse_telecined(
        tmp_path: Path, clips: dict[str, Path]) -> None:
    # A held picture hides one of a block's two repeats: its cadence holds.
    ok, client, inbox = _run(tmp_path, clips["telecine_held"])
    assert ok and "scan=telecine:pictures:inverse-telecine@24000/1001 " in str(
        client.steps[-1][1]["details"])
    out, before = _video(inbox / "prepared.mkv"), _video(clips["telecine_held"])
    assert out["avg_frame_rate"] == "24000/1001"
    assert int(out["nb_read_frames"]) == pytest.approx(int(before["nb_read_frames"]) * 4 / 5,
                                                       abs=3)
    _progressive_in_fact(inbox / "prepared.mkv")


def test_a_telecined_film_cut_after_its_pulldown_is_field_matched(
        tmp_path: Path, clips: dict[str, Path]) -> None:
    # Cut at other places of its cadence, decimate would drop a film frame at
    # a cut: field-matched at 29.97 instead, every frame kept, and kept.
    ok, client, inbox = _run(tmp_path, clips["telecine_cuts"])
    assert ok and "scan=film:pictures:field-matched@30000/1001 " in str(
        client.steps[-1][1]["details"])
    out, before = _video(inbox / "prepared.mkv"), _video(clips["telecine_cuts"])
    assert (out["avg_frame_rate"], out["nb_read_frames"]) == (
        "30000/1001", before["nb_read_frames"])
    _progressive_in_fact(inbox / "prepared.mkv")


def test_a_source_marked_progressive_that_holds_fields_is_deinterlaced(
        tmp_path: Path, clips: dict[str, Path]) -> None:
    # ffprobe's mark is its first picture's: progressive. Its pictures say
    # fields, in one order: deinterlaced, recorded so.
    src = clips["progressive_then_50i"]
    assert _video(src)["field_order"] == "progressive"
    ok, client, inbox = _run(tmp_path, src)
    assert ok and "scan=fields:pictures:deinterlaced-frame@25/1 " in str(
        client.steps[-1][1]["details"])
    _progressive_in_fact(inbox / "prepared.mkv")


def test_an_examination_that_fails_encodes_the_safe_chain_and_keeps_the_original(
        tmp_path: Path, clips: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    # No window read: an unmarked or any source fails safe, never "none".
    from transcoder import ffmpeg as ff
    from transcoder.decision import FieldsWindow
    monkeypatch.setattr(ff, "_idet_window", lambda *_a, **_k: (FieldsWindow(), "no frame read"))
    ok, client, inbox = _run(tmp_path, clips["hevc_marked"])
    assert ok and client.steps[-1][0] == "done"
    assert "scan=unknown:failed:unexamined@25/1 " in str(client.steps[-1][1]["details"])
    scan = _contract(inbox)["scan"]
    assert (scan["done"], scan["judgedBy"]) == ("unexamined", "failed")
    assert scan["filter"].startswith("fieldmatch=order=tff:combmatch=full,bwdif=")
    assert "decimate" not in scan["filter"]


def test_a_ladder_of_a_source_in_fields(tmp_path: Path, clips: dict[str, Path]) -> None:
    # Every rung of one run is made from the same deinterlaced frames, and
    # keyframes come every segment on each.
    ok, _client, inbox = _run(tmp_path, clips["50i"], "source,360p", deinterlace="field")
    assert ok
    contract = _contract(inbox)
    assert [(v["id"], v["encoder"]) for v in contract["video"]] == [("v0", "libx265"),
                                                                    ("v1", "libx264")]
    assert contract["keyframes"] == "interval"
    for name in ("prepared.mkv", "v1.mkv"):
        assert _video(inbox / name)["avg_frame_rate"] == "50/1"
        _progressive_in_fact(inbox / name)
        keys = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                               "-show_entries", "packet=pts_time,flags", "-of", "csv=p=0",
                               str(inbox / name)], capture_output=True, text=True,
                              check=True).stdout
        assert sorted(round(float(t)) for t, f in (ln.split(",")[:2] for ln in keys.splitlines())
                      if "K" in f) == [0, 2]


def test_the_bounded_field_pictures_read_still_finds_the_flag(clips: dict[str, Path]) -> None:
    # libx265 interlace=tff writes field_seq_flag 1; the read stops after
    # FIELD_PICTURES_READ_SECONDS on every ffmpeg, and still sees it.
    from transcoder import ffmpeg as ff
    assert ff.hevc_field_pictures(clips["hevc_field_pictures"], "0:0") is True
    assert ff.hevc_field_pictures(clips["hevc_marked"], "0:0") is False


def test_hevc_field_pictures_are_copied_and_keep_their_original(tmp_path: Path,
                                                                 clips: dict[str, Path]) -> None:
    # ffmpeg decodes them as frames half as tall at the field rate, the
    # stream marked progressive: copied as before, and the copy's record
    # says it kept the fields (fields-kept), which lets no original go.
    src = clips["hevc_field_pictures"]
    before = _video(src)
    assert (before["height"], before["avg_frame_rate"], before["field_order"]) == (
        288, "50/1", "progressive")
    ok, client, inbox = _run(tmp_path, src)
    assert ok and client.steps[-1][0] == "not_applicable"
    assert "scan=field-pictures:bitstream:fields-kept@50/1 " in str(
        client.steps[-1][1]["details"])
    contract = _contract(inbox)
    [v0] = contract["video"]
    assert (v0["file"], v0["mode"]) == (None, "copy")
    assert {k: contract["scan"][k] for k in ("found", "judgedBy", "done", "filter")} == {
        "found": "field-pictures", "judgedBy": "bitstream", "done": "fields-kept",
        "filter": None}
    # Not examined: its pictures are fields, half as tall, no frame to judge.
    assert contract["scan"]["idet"] is None


# ------------------------------------------------------- every moment, read back
# Made clips whose every moment carries its number (tests/fieldcode.py), long
# enough (48 s) that parts of them lie between the examination's windows. The
# guarantee: a word that lets the original go only for a package with no
# combing, no moment lost and no frame doubled; every other word keeps it.
CUTS = ",".join(f"film32k{k}:2.5" for k in (0, 3, 1, 4, 2, 0, 2, 4, 1, 3, 0, 1))


@pytest.fixture(scope="module")
def coded(tmp_path_factory) -> dict[str, Path]:
    import fieldcode as fc

    d = tmp_path_factory.mktemp("coded")
    return {
        "film32": fc.make(d / "film32.mkv", "film32:48"),
        "vid50": fc.make(d / "vid50.mkv", "vid50:48", h=576),
        "film32_video_gap": fc.make(d / "gap.mkv", "film32:11.8,vid60:2,film32:34.2"),
        "pal_film_video_gap": fc.make(d / "palgap.mkv", "film25:13,vid50:1.2,film25:33.8",
                                      h=576, mark="tff"),
        "film32_cuts": fc.make(d / "cuts.mkv", CUTS),
        "pal_shifted": fc.make(d / "shifted.mkv", "film25s:30", h=576, mark="bff"),
        "euro": fc.make(d / "euro.mkv", "film24e:30", h=576),
        "vid50_bff_gap": fc.make(d / "bffgap.mkv", "vid50:13,vid50b:1.2,vid50:33.8", h=576,
                                 mark="tff"),
    }


@pytest.mark.parametrize(("name", "setting", "word"), [
    ("film32", "frame", "inverse-telecine"),
    ("vid50", "field", "deinterlaced-field"),
    ("pal_shifted", "frame", "inverse-telecine"),
])
def test_a_released_package_has_every_moment_once_and_no_combing(
        tmp_path: Path, coded: dict[str, Path], name: str, setting: str, word: str) -> None:
    import fieldcode as fc

    ok, client, inbox = _run(tmp_path, coded[name], deinterlace=setting)
    assert ok
    scan = _contract(inbox)["scan"]
    assert scan["done"] == word and scan["verified"]["ok"], scan
    got = fc.compare(fc.read(coded[name]), fc.read(inbox / "prepared.mkv"))
    assert (got["combed"], got["whole_lost"], got["field_lost_inside"], got["back"]) == (
        0, 0, 0, 0), got
    assert got["repeats"] <= 1 and got["field_lost"] <= 1, got


@pytest.mark.parametrize(("name", "setting"), [
    ("film32_video_gap", "frame"),     # 59.94 video between two windows of a 3:2 film
    ("pal_film_video_gap", "frame"),   # 50i between two windows of a 2:2 film marked tt
    ("film32_cuts", "frame"),          # a 3:2 film cut at every place of its cadence
    ("euro", "frame"),                 # a euro pulldown: a field repeated every 12 frames
    ("vid50_bff_gap", "field"),        # bottom first between two windows of top-first 50i
])
def test_what_the_sample_misses_keeps_the_original_and_loses_no_film_frame(
        tmp_path: Path, coded: dict[str, Path], name: str, setting: str) -> None:
    import fieldcode as fc

    from transcoder.decision import SCAN_RELEASES

    ok, client, inbox = _run(tmp_path, coded[name], deinterlace=setting)
    assert ok
    scan = _contract(inbox)["scan"]
    assert scan["done"] not in SCAN_RELEASES, scan
    got = fc.compare(fc.read(coded[name]), fc.read(inbox / "prepared.mkv"))
    # Made again where the read-through said otherwise: no film frame lost,
    # nothing stepping back in time.
    assert (got["whole_lost"], got["back"]) == (0, 0), got


# ------------------------------------------------- closed captions, pair by pair
# Made clips whose every picture carries closed captions numbered pair by pair
# (tests/fieldcode.py): each chain hands on every valid CEA-608 pair of the
# source once, in order, decimate's dropped frames' among them.
@pytest.fixture(scope="module")
def captioned(tmp_path_factory) -> dict[str, Path]:
    import fieldcode as fc

    d = tmp_path_factory.mktemp("captioned")
    cuts = ",".join(f"film32k{k}:2.5" for k in (0, 3, 1, 4, 2, 0, 2, 4, 1, 3))
    return {
        "film32": fc.make(d / "film32.mkv", "film32:30", captions=True),
        "film32_cuts": fc.make(d / "cuts.mkv", cuts, captions=True),
        "film_video": fc.make(d / "film_video.mkv", "film32:20,vid60:10", captions=True),
        "vid60": fc.make(d / "vid60.mkv", "vid60:20", captions=True),
        "vid50": fc.make(d / "vid50.mkv", "vid50:20", h=576, captions=True),
    }


@pytest.mark.parametrize(("name", "setting", "chain"), [
    ("film32", "frame", "inverse-telecine"),       # decimate drops one frame in five
    ("film32_cuts", "frame", None),                # the cadence cut at every place of it
    ("film_video", "frame", "field-matched"),      # fieldmatch, every frame kept
    ("vid60", "frame", "deinterlaced-frame"),
    ("vid60", "field", "deinterlaced-field"),      # bwdif re-times them at 59.94
])
def test_every_caption_pair_is_carried_once_and_in_order(
        tmp_path: Path, captioned: dict[str, Path], name: str, setting: str,
        chain: str | None) -> None:
    import fieldcode as fc

    src = captioned[name]
    ok, _client, inbox = _run(tmp_path, src, deinterlace=setting)
    assert ok
    scan = _contract(inbox)["scan"]
    if chain is not None:
        assert scan["done"] == chain, scan
    pairs = fc.caption_pairs(src)
    assert len(pairs) == 2 * int(_video(src)["nb_read_frames"])
    assert fc.caption_pairs(inbox / "prepared.mkv") == pairs


def test_captions_bwdif_would_hand_on_twice_are_left_out(
        tmp_path: Path, captioned: dict[str, Path]) -> None:
    # 25 frames at the field rate: bwdif's fifo has no table for 50 and
    # would give both fields' frames the frame's captions. None are kept,
    # so the package claims none and its original is kept for them.
    import fieldcode as fc

    src = captioned["vid50"]
    ok, _client, inbox = _run(tmp_path, src, deinterlace="field")
    assert ok
    scan = _contract(inbox)["scan"]
    assert (scan["done"], scan["frameRate"]) == ("deinterlaced-field", "50/1")
    assert fc.caption_pairs(src) and fc.caption_pairs(inbox / "prepared.mkv") == []
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=closed_captions", "-of", "csv=p=0",
                            str(inbox / "prepared.mkv")], capture_output=True, text=True,
                           check=True).stdout.strip()
    assert probe in ("0", "")   # ffprobe 9.0 leaves the field out where 7.1 says 0
