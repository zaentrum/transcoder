"""Made clips whose every moment carries its own number, and a reader of it.

Each moment of a clip's picture gets a 14-bit Gray code drawn in a strip of
16x16 blocks at the top left (x 8, y 8), outside the middle of the picture
the examination reads. Read back field by field, the strip says for every
frame of a package which moment its top field and its bottom field show: a
frame whose fields show two moments is combed; a moment of the source that
no clean frame shows is lost; the same clean frame twice in a row is a
repeat; a moment before the one before it is a step back in time.

    make(path, "film32:20,vid60:2,film32:20")    # segments KIND:SECONDS
    compare(read(source), read(package))

With `captions`, every picture also carries closed captions numbered pair
by pair (an A53 cc_data SEI: a CEA-608 field-1 and field-2 pair holding
the picture's number), which `caption_pairs` reads back.

Kinds: film32 (23.976p film, 3:2 telecined to 29.97), film32kN (film32 cut
into at cadence offset N), vid60 / vid60b (59.94 moments woven TFF / BFF into
29.97 frames), film25 (25p film, whole frames, 2:2), film25s (25p film
shifted by a field), vid50 / vid50b (50 moments woven TFF / BFF into 25
frames), film24e (24p film in a euro pulldown to 25).
"""

from __future__ import annotations

import re
import statistics
import subprocess
from pathlib import Path

BITS = 14


def _code(rate: str, seconds: float, offset: int) -> str:
    n = f"(N+{offset})"
    gray = f"bitor({n},floor({n}/2))-bitand({n},floor({n}/2))"
    return (f"color=c=black:s={16 * BITS}x16:r={rate}:d={seconds},format=yuv420p,"
            f"geq=lum='if(bitand({gray},pow(2,floor(X/16))),235,16)':cb=128:cr=128")


def _background(w: int, h: int, rate: str, seconds: float, n: int) -> str:
    return (f"testsrc2=s={w}x{h * 2}:r={rate}:d={seconds},scale={w}:{h},setsar=1,"
            f"format=yuv420p,hue=h={n * 47 % 360}")


def make(path: Path, spec: str, *, w: int = 720, h: int = 480, codec: str = "mpeg2",
         mark: str | None = None, rate: str | None = None, captions: bool = False) -> Path:
    """Make a clip of `spec`'s segments, every moment numbered in its strip,
    with a 440 Hz tone; MPEG-2 coded as fields (`codec` "mpeg2"), as
    progressive pictures ("mpeg2p"), or H.264 ("x264"). `mark`: the frames'
    last setfield (tff, bff or prog). `captions`: H.264 coded as fields,
    every picture's closed captions numbered (_with_captions)."""
    segs = [(k, float(s)) for k, s in (part.split(":") for part in spec.split(","))]
    ntsc = any(k.startswith(("film32", "vid60")) for k, _ in segs)
    out_rate = rate or ("30000/1001" if ntsc else "25")
    graph, names, off = [], [], 0
    for n, (kind, sec) in enumerate(segs):
        if kind.startswith("film32"):
            r, frames = "24000/1001", round(sec * 24000 / 1001)
            k = int(kind[7:]) if kind.startswith("film32k") else 0
            tail = (f",telecine=pattern=23,trim=start_frame={k}"
                    f":end_frame={k + round(sec * 30000 / 1001)},setpts=N*1001/30000/TB")
            moments = frames + 2
        elif kind.startswith("vid60"):
            r, frames = "60000/1001", round(sec * 60000 / 1001)
            mode = "interleave_bottom" if kind.endswith("b") else "interleave_top"
            tail, moments = f",tinterlace=mode={mode},setpts=N*1001/30000/TB", frames
        elif kind == "film25":
            r, frames = "25", round(sec * 25)
            tail, moments = ",setpts=N/25/TB", frames
        elif kind == "film25s":
            r, frames = "25", round(sec * 25)
            tail = (",setfield=tff,separatefields,trim=start_frame=1,setpts=N/50/TB,"
                    "weave=first_field=bottom,setpts=N/25/TB")
            moments = frames + 2
        elif kind == "film24e":
            r, frames = "24", round(sec * 24)
            tail = ",telecine=first_field=top:pattern=222222222223,setpts=N/25/TB"
            moments = frames + 2
        elif kind.startswith("vid50"):
            r, frames = "50", round(sec * 50)
            mode = "interleave_bottom" if kind.endswith("b") else "interleave_top"
            tail, moments = f",tinterlace=mode={mode},setpts=N/25/TB", frames
        else:
            raise ValueError(f"unknown kind {kind}")
        graph.append(f"{_background(w, h, r, sec + 0.5, n)}[b{n}];{_code(r, sec + 0.5, off)}[c{n}];"
                     f"[b{n}][c{n}]overlay=8:8,trim=end_frame={frames + 3}{tail}[s{n}]")
        names.append(f"[s{n}]")
        off += moments + 2
    vids = [k for k, _ in segs if k.startswith("vid")]
    order = mark or ("bff" if vids and all(k.endswith("b") for k in vids) else "tff")
    setpts = "N*1001/30000/TB" if out_rate == "30000/1001" else f"N/({out_rate})/TB"
    graph.append("".join(names) + f"concat=n={len(names)}:v=1,setpts={setpts},setfield={order}[v]")
    total = sum(s for _, s in segs)
    if captions:
        return _with_captions(path, ";".join(graph), out_rate, total, order)
    vcodec = {"mpeg2": ["-c:v", "mpeg2video", "-flags", "+ildct+ilme", "-q:v", "2"],
              "mpeg2p": ["-c:v", "mpeg2video", "-q:v", "2"],
              "x264": ["-c:v", "libx264", "-preset", "veryfast", "-crf", "16"]}[codec]
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-filter_complex", ";".join(graph), "-f", "lavfi", "-i",
                    f"sine=frequency=440:sample_rate=48000:duration={total}",
                    "-map", "[v]", "-map", "0:a", "-r", out_rate, *vcodec, "-c:a", "aac",
                    str(path)], check=True, stdin=subprocess.DEVNULL)
    return path


def _odd_parity(b: int) -> int:
    b &= 0x7F
    return b | (0x80 if bin(b).count("1") % 2 == 0 else 0)


def _cc_sei(n: int) -> bytes:
    """An H.264 SEI NAL unit with picture n's cc_data: a valid CEA-608
    field-1 pair (number 2n) and field-2 pair (2n + 1), 14 bits each in
    two bytes of odd parity."""
    pairs = []
    for kind, number in ((0xFC, 2 * n), (0xFD, 2 * n + 1)):
        pairs += [kind, _odd_parity(0x20 + (number >> 7) % 0x60), _odd_parity(number & 0x7F)]
    cc = bytes([0x40 | 2, 0xFF, *pairs, 0xFF])
    payload = bytes([0xB5, 0x00, 0x31]) + b"GA94" + bytes([0x03]) + cc
    return b"\x00\x00\x00\x01" + bytes([0x06, 0x04, len(payload)]) + payload + b"\x80"


def _with_captions(path: Path, graph: str, rate: str, seconds: float, order: str) -> Path:
    """H.264 coded as fields in `order` (one slice, no B-frames: decode
    order is picture order), an SEI with its numbered captions put ahead
    of every picture's slice, muxed with a tone."""
    raw = path.with_suffix(".h264")
    fields = {"tff": ":interlaced=1:tff=1", "bff": ":interlaced=1:bff=1"}.get(order, "")
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-filter_complex", graph, "-map", "[v]", "-r", rate, "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "16", "-bf", "0",
                    "-x264-params", f"slices=1{fields}", "-f", "h264", str(raw)],
                   check=True, stdin=subprocess.DEVNULL)
    data, made, pos, n = raw.read_bytes(), bytearray(), 0, 0
    for m in re.finditer(rb"\x00\x00\x01(.)", data, re.DOTALL):
        start = m.start() - 1 if m.start() and data[m.start() - 1] == 0 else m.start()
        if m.group(1)[0] & 0x1F in (1, 5):
            made += data[pos:start] + _cc_sei(n)
            pos, n = start, n + 1
    raw.write_bytes(bytes(made + data[pos:]))
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-framerate", rate, "-i", str(raw), "-f", "lavfi", "-i",
                    f"sine=frequency=440:sample_rate=48000:duration={seconds}",
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", str(path)],
                   check=True, stdin=subprocess.DEVNULL)
    raw.unlink()
    return path


def caption_pairs(path: Path) -> list[int]:
    """The number of every valid CEA-608 pair (cc_valid 1, field 1 or 2) the
    video's closed captions carry, in the order its frames show them."""
    out = subprocess.run(["ffprobe", "-v", "error", "-f", "lavfi", "-i",
                          f"movie={path}[out0+subcc]", "-select_streams", "s:0",
                          "-show_entries", "packet=data", "-show_data", "-of", "default=nw=1"],
                         capture_output=True, text=True, check=True,
                         stdin=subprocess.DEVNULL).stdout
    numbers, payload = [], bytearray()
    for line in [*out.splitlines(), "[/PACKET]"]:
        m = re.match(r"^[0-9a-f]{8}: ((?:[0-9a-f]{4} ?)+)", line)
        if m:
            payload += bytes.fromhex(m.group(1).replace(" ", ""))
        elif line.startswith(("[/PACKET]", "data=")):
            for i in range(0, len(payload) - 2, 3):
                kind, b1, b2 = payload[i:i + 3]
                if kind & 0x04 and kind & 0x03 in (0, 1):
                    numbers.append(((b1 & 0x7F) - 0x20) * 128 + (b2 & 0x7F))
            payload = bytearray()
    return numbers


def read(path: Path) -> list[tuple[int, int]]:
    """The moment each frame's top field and bottom field show."""
    w, h = 16 * BITS, 16
    # Every decoded frame once (-fps_mode passthrough): rawvideo's default
    # would repeat a first frame that starts after the audio.
    raw = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-i", str(path),
                          "-map", "0:v:0", "-vf", f"crop={w}:{h}:8:8,format=gray",
                          "-fps_mode", "passthrough", "-f", "rawvideo", "-"],
                         capture_output=True, check=True, stdin=subprocess.DEVNULL).stdout
    size = w * h
    frames = []
    for i in range(len(raw) // size):
        f = raw[i * size:(i + 1) * size]
        codes = []
        for parity in (0, 1):
            rows = [y for y in range(2, 14) if y % 2 == parity]
            v = 0
            for b in range(BITS):
                xs = range(16 * b + 5, 16 * b + 11)
                if statistics.median(f[y * w + x] for y in rows for x in xs) > 126:
                    v |= 1 << b
            n, s = v, v >> 1
            while s:
                n ^= s
                s >>= 1
            codes.append(n)
        frames.append((codes[0], codes[1]))
    return frames


def compare(src: list[tuple[int, int]], pkg: list[tuple[int, int]]) -> dict[str, int]:
    """What a package did with its source's moments: combed frames, moments
    shown whole in the source and lost from the package, field moments
    (video, orphan fields) of the source no field of the package shows (and
    of those, the ones neither its first nor its last moment: an orphan
    field at a clip's very edge has no field to be matched with), repeated
    frames after the first (and of those, the ones before the last four: the
    last cycle of fewer than five frames, which decimate passes whole, may
    show its duplicate), and steps back in time. The output check allows
    those two at a title's edges, by rule: field_lost_inside and
    repeats_inside are what it does not."""
    top, bottom = {t for t, _ in src}, {b for _, b in src}
    whole = top & bottom
    fields_only = (top | bottom) - whole
    clean = [t for t, b in pkg if t == b]
    any_field = {t for t, _ in pkg} | {b for _, b in pkg}
    lost = fields_only - any_field
    repeats = [i for i in range(1, len(pkg)) if pkg[i] == pkg[i - 1] and pkg[i][0] == pkg[i][1]]
    return {
        "frames": len(pkg),
        "combed": sum(t != b for t, b in pkg),
        "whole_lost": len(whole - set(clean)),
        "field_lost": len(lost),
        "field_lost_inside": len(lost - {min(top | bottom), max(top | bottom)}),
        "repeats": len(repeats),
        "repeats_inside": sum(i < len(pkg) - 4 for i in repeats),
        "back": sum(1 for i in range(1, len(clean)) if clean[i] < clean[i - 1]),
    }
