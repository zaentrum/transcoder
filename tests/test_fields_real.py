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
  * mixed: 25p, then 50i woven; and a telecined film, then 60i woven (a
    film with video-rate parts, which inverse telecine would spoil).

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
MPEG2_FIELDS = ["-c:v", "mpeg2video", "-flags", "+ildct+ilme", "-top", "1", "-q:v", "3"]


def picture(w: int, h: int, rate: str, seconds: int = SECONDS) -> str:
    """A camera's picture: testsrc2 drawn twice as tall, scaled down."""
    return f"testsrc2=s={w}x{h * 2}:r={rate}:d={seconds},scale={w}:{h}"


def _make(path: Path, inputs: list[str], vf: str, vcodec: list[str], *,
          complex_graph: bool = False, extra: list[str] | None = None) -> Path:
    args = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    for source in inputs:
        args += ["-f", "lavfi", "-i", source]
    args += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={SECONDS}"]
    audio = len(inputs)
    if complex_graph:
        args += ["-filter_complex", vf + "[v]", "-map", "[v]"]
    else:
        args += ["-map", "0:v", "-vf", vf]
    args += ["-map", f"{audio}:a", *vcodec, "-c:a", "aac", *(extra or []), str(path)]
    subprocess.run(args, check=True, stdin=subprocess.DEVNULL)
    return path


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


def _progressive_in_fact(path: Path) -> None:
    """idet calls no more than 2 % of the frames it determines TFF or BFF."""
    c = _idet(path)
    determined = c["tff"] + c["bff"] + c["progressive"]
    assert determined > 0
    assert c["tff"] + c["bff"] <= 0.02 * determined, c


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
        "film_and_60i": ("fields", "pictures", "tff"),
    }


def test_50i_is_deinterlaced_at_its_field_rate(tmp_path: Path, clips: dict[str, Path]) -> None:
    src = clips["50i"]
    ok, client, inbox = _run(tmp_path, src)
    assert ok and client.steps[-1][0] == "done"
    details = str(client.steps[-1][1]["details"])
    assert "scan=fields:pictures:deinterlaced:field@50/1 " in details
    assert "src_codec=mpeg2video" in details
    out, before = _video(inbox / "prepared.mkv"), _video(src)
    assert (out["codec_name"], out["avg_frame_rate"], out["field_order"]) == (
        "hevc", "50/1", "progressive")
    assert int(out["nb_read_frames"]) == 2 * int(before["nb_read_frames"])
    _progressive_in_fact(inbox / "prepared.mkv")
    scan = _contract(inbox)["scan"]
    assert {k: scan[k] for k in ("found", "judgedBy", "fieldOrder", "done", "rate", "order",
                                 "frameRate", "filter")} == {
        "found": "fields", "judgedBy": "pictures", "fieldOrder": "tb", "done": "deinterlaced",
        "rate": "field", "order": "tff", "frameRate": "50/1",
        "filter": "bwdif=mode=send_field:parity=tff:deint=all"}
    assert scan["idet"]["tff"] > 0.9 * scan["idet"]["frames"]
    # The audio goes with it, from the same start.
    streams = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,start_time", "-of", "json",
         str(inbox / "prepared.mkv")], capture_output=True, text=True, check=True,
    ).stdout)["streams"]
    starts = {s["codec_type"]: float(s.get("start_time") or 0) for s in streams}
    assert starts["audio"] == pytest.approx(starts["video"], abs=0.03)


def test_50i_at_the_frame_rate_when_the_setting_says_so(tmp_path: Path,
                                                         clips: dict[str, Path]) -> None:
    src = clips["50i"]
    ok, client, inbox = _run(tmp_path, src, deinterlace="frame")
    assert ok and "scan=fields:pictures:deinterlaced:frame@25/1 " in str(
        client.steps[-1][1]["details"])
    out = _video(inbox / "prepared.mkv")
    assert (out["avg_frame_rate"], out["field_order"]) == ("25/1", "progressive")
    assert int(out["nb_read_frames"]) == int(_video(src)["nb_read_frames"])
    _progressive_in_fact(inbox / "prepared.mkv")
    assert _contract(inbox)["scan"]["rate"] == "frame"


def test_interlaced_h264_is_no_longer_passed_through(tmp_path: Path,
                                                      clips: dict[str, Path]) -> None:
    # Browser-friendly H.264, which a CPU host copied: in fields, it is
    # encoded, bottom field first as idet and its mark say, at 59.94.
    ok, client, inbox = _run(tmp_path, clips["60i_bff"])
    assert ok and client.steps[-1][0] == "done"
    details = str(client.steps[-1][1]["details"])
    assert "scan=fields:pictures:deinterlaced:field@60000/1001 " in details
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
    assert ok and "scan=telecine:pictures:inverse_telecine@24000/1001 " in str(
        client.steps[-1][1]["details"])
    out, before = _video(inbox / "prepared.mkv"), _video(src)
    assert (out["avg_frame_rate"], out["field_order"]) == ("24000/1001", "progressive")
    assert int(out["nb_read_frames"]) == pytest.approx(int(before["nb_read_frames"]) * 4 / 5, abs=1)
    _progressive_in_fact(inbox / "prepared.mkv")
    scan = _contract(inbox)["scan"]
    assert (scan["found"], scan["done"], scan["frameRate"]) == (
        "telecine", "inverseTelecine", "24000/1001")
    assert scan["idet"]["repeatedTop"] > 0 and scan["idet"]["repeatedBottom"] > 0


def test_whole_frames_marked_as_fields_are_copied_and_say_so(tmp_path: Path,
                                                              clips: dict[str, Path]) -> None:
    # HEVC Main marked tb, its frames whole: copied as before, and the
    # record of the copy says it was found progressive.
    ok, client, inbox = _run(tmp_path, clips["hevc_marked"])
    assert ok and client.steps[-1][0] == "not_applicable"
    details = str(client.steps[-1][1]["details"])
    assert "reason=source_already_hevc:hevc" in details
    assert "scan=progressive:pictures:as_found@25/1 " in details
    assert sorted(p.name for p in inbox.iterdir()) == ["renditions.json"]
    contract = _contract(inbox)
    [v0] = contract["video"]
    assert (v0["file"], v0["mode"], v0["carries"]) == (None, "copy",
                                                       ["video", "audio", "subtitles"])
    assert {k: contract["scan"][k] for k in ("found", "judgedBy", "fieldOrder", "done",
                                             "filter")} == {
        "found": "progressive", "judgedBy": "pictures", "fieldOrder": "tb",
        "done": "progressiveAsFound", "filter": None}


def test_an_mpeg2_film_transfer_at_25_frames_is_left_as_it_is(tmp_path: Path,
                                                               clips: dict[str, Path]) -> None:
    # Progressive frames in an MPEG-2 stream coded and marked as fields, a
    # dark logo of its own in fields: encoded frame for frame, not
    # deinterlaced (bwdif would only soften it), and marked progressive.
    src = clips["film_mpeg2"]
    ok, client, inbox = _run(tmp_path, src)
    assert ok and client.steps[-1][0] == "done"
    assert "scan=progressive:pictures:as_found@25/1 " in str(client.steps[-1][1]["details"])
    out, before = _video(inbox / "prepared.mkv"), _video(src)
    assert (out["avg_frame_rate"], out["field_order"]) == ("25/1", "progressive")
    assert out["nb_read_frames"] == before["nb_read_frames"]
    scan = _contract(inbox)["scan"]
    assert (scan["done"], scan["filter"]) == ("progressiveAsFound", "setfield=prog")
    assert scan["idet"]["tff"] + scan["idet"]["bff"] == 0


def test_mixed_content_is_deinterlaced(tmp_path: Path, clips: dict[str, Path]) -> None:
    ok, client, inbox = _run(tmp_path, clips["mixed"])
    assert ok and "scan=fields:pictures:deinterlaced:field@50/1 " in str(
        client.steps[-1][1]["details"])
    _progressive_in_fact(inbox / "prepared.mkv")
    assert _video(inbox / "prepared.mkv")["avg_frame_rate"] == "50/1"


def test_a_film_with_video_parts_is_deinterlaced_at_its_field_rate(
        tmp_path: Path, clips: dict[str, Path]) -> None:
    # Its cadence covers half of it: deinterlaced, every field a frame, so
    # the video part keeps its motion and nothing stays combed.
    ok, client, inbox = _run(tmp_path, clips["film_and_60i"])
    assert ok and "scan=fields:pictures:deinterlaced:field@60000/1001 " in str(
        client.steps[-1][1]["details"])
    scan = _contract(inbox)["scan"]
    assert scan["idet"]["coverage"] < 0.8 and scan["idet"]["cadence"] >= 0.9
    _progressive_in_fact(inbox / "prepared.mkv")


def test_a_ladder_of_a_source_in_fields(tmp_path: Path, clips: dict[str, Path]) -> None:
    # Every rung of one run is made from the same deinterlaced frames, and
    # keyframes come every segment on each.
    ok, _client, inbox = _run(tmp_path, clips["50i"], "source,360p")
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
