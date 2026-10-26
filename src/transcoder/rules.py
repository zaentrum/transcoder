"""The rules this transcoder runs with, as it reports them to the catalog.

The catalog holds every movie and episode against the rules its workers
run with now, to find a package made before a rule came: a 4K original's
video copied as it was, before the caps (katalog-manager's README,
"Library compliance"). It knows the rules only as a worker reports them,
and assumes none: `PUT /api/analyze/rules/transcoder` with

    {"version": "<image commit or release>", "rules": {
      "caps": {"movie1080Mbps": 8, "movie2160Mbps": 14, "movieMaxGiB": 15,
               "episode1080Mbps": 6, "episode2160Mbps": 8},
      "bucket2160": {"minHeight": 2000, "minWidth": 3200},
      "encoder": {"codec": "hevc_nvenc", "preset": "p5", "cq": 23,
                  "maxrate1080Mbps": null, "maxrate2160Mbps": null},
      "ladder": "source:hevc", "extraLadder": "source:hevc",
      "ladderRungs": [{"name": "source", "codec": "hevc", "codecNamed": true,
                       "maxrateBps": null}],
      "rungMaxrateMbps": {"360": {"h264": 0.8, "hevc": 0.6}, ...,
                          "2160": {"h264": 16.0, "hevc": 14.0}},
      "dolbyVision": {"profile5": "reshaped", "profile7": "refused"}}}

Every value is the one this process runs with (README "Caps", "Encoders",
"Ladder"): a cap of 0 is a rule switched off, a maxrate of null leaves the
maxrate to the caps. `encoder` is the HEVC encoder the startup probe found,
with its knobs: NVENC's preset and constant quality, or libx265's preset and
constant rate factor (`crf`, and `cq` null). A ladder is written as its
spec reads, every rung as the planner takes it — its codec where the spec
names one, its maxrate where it sets one — so the empty default is
`source`, the same one rendition; `ladderRungs` are the item ladder's rungs
in parts, and `rungMaxrateMbps` the maxrate of a rung that sets none, by
the height of its box and its codec (decision.py's table: a scaled rung's,
and a source-size H.264 encode's), so the catalog counts every rung of a
package it estimates. `dolbyVision` is what becomes of an original of each Dolby Vision
profile the planner names, as this process can do it: `reshaped` (profile 5,
reshaped to HDR10 on the Vulkan device the startup preflight found, the
libplacebo chain an encode runs run once on it) or `refused` (the step fails
and the original is kept: profile 7, which needs a tone-mapping encode, and
profile 5 without such a device).

The report goes at startup and every ten minutes, on a thread and a client
of its own. A catalog that does not take it (404 or 405: older than the
check) is said once, in `transcoder.rules.unsupported`, and asked again on
the next round; any other failure is logged and never stops the worker.
"""

from __future__ import annotations

import importlib.metadata
import os
import threading
from typing import Any, Protocol

import httpx
import structlog

from .config import Config
from .decision import (
    DOVI_REFUSED_PROFILES,
    DOVI_RESHAPED_PROFILES,
    UHD_MIN_HEIGHT,
    UHD_MIN_WIDTH,
    Encoders,
    RungSpec,
    VulkanDevice,
    rung_maxrates_mbps,
)

log = structlog.get_logger(__name__)

WORKER = "transcoder"

# How often the report goes, seconds.
REPORT_INTERVAL_SECONDS = 600.0

# What an older catalog answers a route it does not have.
UNSUPPORTED = frozenset({404, 405})


def image_version() -> str | None:
    """This image's version: its commit or release tag, as the image build
    bakes it in (IMAGE_VERSION), else the installed package's."""
    version = (os.environ.get("IMAGE_VERSION") or "").strip()
    if version:
        return version
    try:
        return importlib.metadata.version("transcoder")
    except importlib.metadata.PackageNotFoundError:
        return None


def _number(value: float) -> float | int:
    """A number as short as it reads: 8.0 is 8, 7.5 stays."""
    return int(value) if float(value).is_integer() else value


def _rate(bps: int) -> str:
    """A rung's maxrate as a ladder spec writes it: 3M, 2500k."""
    if bps % 1_000_000 == 0:
        return f"{bps // 1_000_000}M"
    if bps % 1000 == 0:
        return f"{bps // 1000}k"
    return f"{bps / 1_000_000:g}M"


def canonical_ladder(rungs: list[RungSpec] | tuple[RungSpec, ...]) -> str:
    """A ladder as its spec reads, each rung as the planner takes it: its
    name, its codec where the spec names it, its maxrate where it sets one
    ("source", "source:hevc", "source,720p:h264:2500k")."""
    tokens = []
    for r in rungs:
        token = r.name
        if r.codec_named:
            token += f":{r.codec}"
        if r.maxrate_bps:
            token += f":{_rate(r.maxrate_bps)}"
        tokens.append(token)
    return ",".join(tokens)


def dolby_vision(vulkan: VulkanDevice) -> dict[str, str]:
    """What becomes of an original of each Dolby Vision profile the planner
    names (decision.py), as this process can do it now: a profile it
    reshapes is `reshaped` while the startup preflight found a Vulkan device
    the reshaping runs on, and `refused` without one; a profile it never
    plans is `refused`."""
    out: dict[str, str] = {}
    for profile in sorted(DOVI_RESHAPED_PROFILES | DOVI_REFUSED_PROFILES):
        reshaped = profile in DOVI_RESHAPED_PROFILES and vulkan.ok
        out[f"profile{profile}"] = "reshaped" if reshaped else "refused"
    return out


def rules_payload(
    cfg: Config,
    encoders: Encoders,
    ladder: list[RungSpec] | tuple[RungSpec, ...],
    extra_ladder: list[RungSpec] | tuple[RungSpec, ...],
    version: str | None,
    *,
    vulkan: VulkanDevice,
) -> dict[str, Any]:
    """The report of the rules this transcoder runs with (module doc); vulkan
    is the device the startup preflight found, or none."""
    nvenc = encoders.hevc.endswith("_nvenc")
    encoder: dict[str, Any] = {
        "codec": encoders.hevc,
        "preset": cfg.nvenc_preset if nvenc else cfg.x265_preset,
        "cq": cfg.nvenc_cq if nvenc else None,
        "maxrate1080Mbps": cfg.maxrate_1080p_mbps,
        "maxrate2160Mbps": cfg.maxrate_2160p_mbps,
    }
    if not nvenc:
        encoder["crf"] = cfg.x265_crf
    return {
        "version": version,
        "rules": {
            "caps": {
                "movie1080Mbps": _number(cfg.cap_movie_1080_mbps),
                "movie2160Mbps": _number(cfg.cap_movie_2160_mbps),
                "movieMaxGiB": _number(cfg.cap_movie_max_gib),
                "episode1080Mbps": _number(cfg.cap_episode_1080_mbps),
                "episode2160Mbps": _number(cfg.cap_episode_2160_mbps),
            },
            "bucket2160": {"minHeight": UHD_MIN_HEIGHT, "minWidth": UHD_MIN_WIDTH},
            "encoder": encoder,
            "ladder": canonical_ladder(ladder),
            "extraLadder": canonical_ladder(extra_ladder),
            "ladderRungs": [
                {"name": r.name, "codec": r.codec, "codecNamed": r.codec_named,
                 "maxrateBps": r.maxrate_bps}
                for r in ladder
            ],
            "rungMaxrateMbps": rung_maxrates_mbps(),
            "dolbyVision": dolby_vision(vulkan),
        },
    }


class RulesClient(Protocol):
    def put_rules(self, worker: str, payload: dict[str, Any]) -> httpx.Response: ...


class RulesReporter:
    """Reports the rules to the catalog, at once and then every interval
    until stop is set. Nothing it meets stops the worker: a catalog that
    does not take the report is said once, any other failure logged."""

    def __init__(
        self,
        client: RulesClient,
        payload: dict[str, Any],
        interval: float = REPORT_INTERVAL_SECONDS,
    ) -> None:
        self._client = client
        self._payload = payload
        self._interval = interval
        self._unsupported_said = False
        self._reported = False

    def report_once(self) -> bool:
        """One report; True when the catalog took it."""
        try:
            resp = self._client.put_rules(WORKER, self._payload)
        except Exception as e:
            log.warning("transcoder.rules.report_failed", error=str(e)[:200])
            return False
        if resp.status_code in UNSUPPORTED:
            if not self._unsupported_said:
                log.info("transcoder.rules.unsupported", status=resp.status_code,
                         note="the catalog takes no report of rules; asked again every round")
                self._unsupported_said = True
            return False
        if resp.status_code >= 400:
            log.warning("transcoder.rules.refused", status=resp.status_code, body=resp.text[:300])
            return False
        if not self._reported:
            log.info("transcoder.rules.reported", version=self._payload.get("version"))
            self._reported = True
        return True

    def run(self, stop: threading.Event) -> None:
        """Report now and every interval until stop is set."""
        while not stop.is_set():
            try:
                self.report_once()
            except Exception:  # never let the thread die of a report
                log.exception("transcoder.rules.report_failed")
            if stop.wait(self._interval):
                return
