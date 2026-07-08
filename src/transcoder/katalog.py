"""HTTP client for the katalog Spring app.

The transcoder is a pure Kafka consumer now; the katalog HTTP calls it
still makes are all state reads + step writes (the DB step rows are the
state the Activity monitor reads — the worker never claims over HTTP):
  * `GET  /api/analyze/items/{id}` — resolve one item to its full detail
    (id, type, title, year, durationMs, path, season/episode, tmdb ids).
    Driven by the itemId in the consumed `stube.catalog.item.analyzed`
    event. Returns None on 404 / no primary path.
  * `GET  /api/analyze/items/{id}/steps` — read the current status of
    every step. The event loop uses this as the idempotency guard: if
    `transcode` is already done, skip the encode but still emit the next
    event so the chain isn't stuck.
  * `PUT  /api/analyze/items/{id}/steps/transcode` — flip the step to
    in_progress / done / skipped / not_applicable / failed as the
    worker progresses. Setting transcode to a terminal state on the
    Java side will also push the chain forward by inserting
    `package=pending` for the next worker.
  * `POST /api/analyze/items/{id}/fail` — last-resort hard fail when
    we can't even attribute the error to the transcode step (e.g.
    the source file vanished from NFS between scan and encode).

Token refresh on 401 is handled here so the worker loop stays
straightforward. Same shape as packager/katalog.py — the two clients
are intentionally parallel so anyone reading both sees the same
shape.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx
import structlog

log = structlog.get_logger(__name__)

# Keycloak default for client_credentials is 300 s; refresh 30 s ahead
# so we never send a token within seconds of expiry.
TOKEN_REFRESH_LEAD_SECONDS = 30


@dataclass
class ClaimedItem:
    id: str
    type: str
    title: str
    year: int | None
    duration_ms: int | None
    path: str

    @classmethod
    def from_json(cls, body: dict[str, Any]) -> ClaimedItem:
        return cls(
            id=body["id"],
            type=body["type"],
            title=body.get("title") or "",
            year=body.get("year"),
            duration_ms=body.get("durationMs"),
            path=body["path"],
        )


class KatalogClient:
    def __init__(
        self,
        base_url: str,
        token_url: str,
        client_id: str,
        client_secret: str,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._token_url = token_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._http = httpx.Client(timeout=timeout_seconds)
        self._token: str | None = None
        self._token_expires_at: float = 0.0

    def close(self) -> None:
        self._http.close()

    # ---------------------------------------------------------------- auth
    def _ensure_token(self) -> str:
        if self._token and time.time() < self._token_expires_at:
            return self._token
        resp = self._http.post(
            self._token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
            },
        )
        resp.raise_for_status()
        body = resp.json()
        self._token = body["access_token"]
        ttl = int(body.get("expires_in", 60))
        self._token_expires_at = time.time() + ttl - TOKEN_REFRESH_LEAD_SECONDS
        log.debug("oidc.token_refreshed", expires_in=ttl)
        return self._token

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._ensure_token()}"}

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        url = f"{self._base}{path}"
        for attempt in range(2):
            resp = self._http.request(method, url, headers=self._headers(), **kwargs)
            if resp.status_code == 401 and attempt == 0:
                # Token revoked or rotated; refresh and retry once.
                self._token = None
                self._token_expires_at = 0
                continue
            return resp
        return resp  # type: ignore[return-value]

    # -------------------------------------------------------------- reads
    def get_item(self, item_id: str) -> ClaimedItem | None:
        """Fetch one item with its full detail + primary playback path.
        Driven by the itemId carried in the consumed Kafka event. Returns
        None when the item is unknown (404) or carries no primary path
        (the event loop then logs + commits + skips — no next event).

        The katalog endpoint returns the FULL detail shape
        ({id,type,title,year,durationMs,path,seasonNumber,episodeNumber,
        seriesTitle,seriesTmdbId,movieTmdbId}); ClaimedItem.from_json
        reads only the fields the encode path needs and tolerates the
        rest."""
        resp = self._request("GET", f"/api/analyze/items/{item_id}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        body = resp.json()
        if not body.get("path"):
            # Item exists but has no primary asset (e.g. metadata-only
            # row). Nothing for the transcoder to encode.
            return None
        return ClaimedItem.from_json(body)

    def get_steps(self, item_id: str) -> dict[str, str]:
        """Return the current status of every step on `item_id`. The
        event loop uses this as the pre-work idempotency guard: if the
        `transcode` step is already `done`, skip the encode (but still
        emit the next event). Best-effort — on error return {} so the
        guard falls through to doing the work."""
        try:
            resp = self._request(
                "GET",
                f"/api/analyze/items/{item_id}/steps",
            )
            if resp.status_code >= 400:
                log.warning(
                    "steps.get_failed",
                    item_id=item_id,
                    status=resp.status_code,
                    body=resp.text[:200],
                )
                return {}
            body = resp.json()
            steps = body.get("steps") or {}
            return {str(k): str(v) for k, v in steps.items()}
        except Exception as e:
            log.warning("steps.get_exception", item_id=item_id, error=str(e)[:200])
            return {}

    # ------------------------------------------------------------- steps
    def upsert_step(
        self,
        item_id: str,
        status: str,
        *,
        error: str | None = None,
        details: str | None = None,
    ) -> None:
        """Move the transcode step to `status`. Best-effort; failures
        are logged and swallowed so a flaky bookkeeping call doesn't
        crash an otherwise-successful encode job. The endpoint is
        idempotent via ON CONFLICT (item_id, step)."""
        body: dict[str, Any] = {"status": status}
        if error is not None:
            body["error"] = error[:500]
        if details is not None:
            body["details"] = details
        try:
            resp = self._request(
                "PUT",
                f"/api/analyze/items/{item_id}/steps/transcode",
                json=body,
            )
            if resp.status_code >= 400:
                log.warning(
                    "transcode.step.upsert_failed",
                    item_id=item_id,
                    status=status,
                    http=resp.status_code,
                    body=resp.text[:300],
                )
        except Exception as e:
            log.warning(
                "transcode.step.upsert_exception",
                item_id=item_id,
                status=status,
                error=str(e)[:200],
            )

    def fail(self, item_id: str, reason: str) -> None:
        """Catastrophic-failure fallback (source file missing, etc.).
        Sets transcode=failed via the same global handler the analyzer
        uses; logs are captured in the audit row."""
        try:
            resp = self._request(
                "POST",
                f"/api/analyze/items/{item_id}/fail",
                json={"reason": reason},
            )
            if resp.status_code >= 400:
                log.warning(
                    "transcode.fail.report_failed",
                    item_id=item_id,
                    status=resp.status_code,
                    body=resp.text[:300],
                )
        except Exception as e:
            log.warning("transcode.fail.exception", item_id=item_id, error=str(e)[:200])
