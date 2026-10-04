"""renditions.json's source block — pure logic, no ffmpeg needed."""

from __future__ import annotations

from pathlib import Path

from transcoder.decision import CPU_ENCODERS, parse_ladder, plan_renditions
from transcoder.ffmpeg import RungResult
from transcoder.renditions import build_contract
from transcoder.worker import _bit_rate


def _plan():
    probe = {"video": {"codec_name": "mpeg2video", "width": 720, "height": 576,
                       "pix_fmt": "yuv420p", "avg_frame_rate": "25/1"},
             "audio": [], "subtitles": [], "duration_ms": 5_400_000}
    return plan_renditions(probe, parse_ladder(""), CPU_ENCODERS)


def test_source_block_says_what_the_catalog_keeps() -> None:
    # The packager forwards this block to the catalog's packaging-complete,
    # which keeps it as the title's source asset (codec, WxH, duration,
    # bit rate) — the exact probe, not the encode's figures.
    plan = _plan()
    [rung] = plan.rungs
    result = RungResult(rung=rung, path=Path("/inbox/prepared.mkv"), size_bytes=1_000_000_000,
                        width=720, height=576, video_start=0.0)
    contract = build_contract("i", plan, [result], CPU_ENCODERS, duration_ms=5_400_000,
                              source_bit_rate=6_500_000)
    assert contract["source"] == {
        "codec": "mpeg2video", "width": 720, "height": 576, "frameRate": "25/1",
        "hdr": False, "durationMs": 5_400_000, "bitRate": 6_500_000,
    }
    # The encode's own bit rate stays on its rung.
    assert contract["video"][0]["bitrateBps"] == round(1_000_000_000 * 8 / 5400)


def test_source_block_leaves_unknowns_null() -> None:
    plan = _plan()
    contract = build_contract("i", plan, [], CPU_ENCODERS, duration_ms=0)
    assert contract["source"]["durationMs"] is None
    assert contract["source"]["bitRate"] is None


def test_bit_rate_reads_ffprobe_strings() -> None:
    assert _bit_rate("8000000") == 8_000_000
    assert _bit_rate(None) is None
    assert _bit_rate("N/A") is None
    assert _bit_rate("0") is None
