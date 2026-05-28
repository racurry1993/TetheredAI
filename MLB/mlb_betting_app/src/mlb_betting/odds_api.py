from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Mapping, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests

from .db import insert_api_usage, insert_raw_payload, upsert_odds_event
from .team_mapping import normalize_team_name

LOGGER = logging.getLogger(__name__)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sanitize_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return url
    parts = urlsplit(url)
    query = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if key.lower() == "apikey":
            value = "REDACTED"
        query.append((key, value))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


class OddsApiClient:
    def __init__(self, api_key: str, base_url: str = "https://api.the-odds-api.com") -> None:
        if not api_key:
            raise ValueError("ODDS_API_KEY is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def get_sports(self, all_sports: bool = False) -> tuple[list[dict[str, Any]], Mapping[str, Any], str]:
        url = f"{self.base_url}/v4/sports/"
        params = {"apiKey": self.api_key}
        if all_sports:
            params["all"] = "true"
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json(), response.headers, response.url

    def get_odds(
        self,
        sport: str = "baseball_mlb",
        regions: str = "us",
        markets: str = "h2h,spreads,totals",
        odds_format: str = "american",
        date_format: str = "iso",
        bookmakers: Optional[str] = None,
        commence_time_from: Optional[str] = None,
        commence_time_to: Optional[str] = None,
        event_ids: Optional[str] = None,
        include_links: bool = False,
        include_sids: bool = True,
        include_bet_limits: bool = False,
    ) -> tuple[list[dict[str, Any]], Mapping[str, Any], str, Mapping[str, Any]]:
        url = f"{self.base_url}/v4/sports/{sport}/odds/"
        params: dict[str, Any] = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
            "dateFormat": date_format,
            "includeSids": str(include_sids).lower(),
            "includeLinks": str(include_links).lower(),
            "includeBetLimits": str(include_bet_limits).lower(),
        }
        if bookmakers:
            params["bookmakers"] = bookmakers
            params.pop("regions", None)
        if commence_time_from:
            params["commenceTimeFrom"] = commence_time_from
        if commence_time_to:
            params["commenceTimeTo"] = commence_time_to
        if event_ids:
            params["eventIds"] = event_ids
        response = self.session.get(url, params=params, timeout=45)
        response.raise_for_status()
        safe_params = dict(params)
        safe_params["apiKey"] = "REDACTED"
        return response.json(), response.headers, response.url, safe_params


def save_odds_payload(conn, payload: list[dict[str, Any]], fetched_at_utc: str) -> int:
    rows = 0
    for event in payload:
        upsert_odds_event(conn, event, fetched_at_utc)
        for bookmaker in event.get("bookmakers", []) or []:
            for market in bookmaker.get("markets", []) or []:
                for outcome in market.get("outcomes", []) or []:
                    point = outcome.get("point")
                    point_key = "NA" if point is None else str(point)
                    conn.execute(
                        """
                        INSERT INTO odds_snapshots (
                            fetched_at_utc, event_id, sport_key, commence_time_utc,
                            home_team, away_team, bookmaker_key, bookmaker_title,
                            bookmaker_last_update_utc, market_key, outcome_name,
                            outcome_name_norm, outcome_price, outcome_point,
                            outcome_point_key, outcome_description, outcome_link, outcome_sid
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            fetched_at_utc,
                            event.get("id"),
                            event.get("sport_key"),
                            event.get("commence_time"),
                            event.get("home_team"),
                            event.get("away_team"),
                            bookmaker.get("key"),
                            bookmaker.get("title"),
                            bookmaker.get("last_update"),
                            market.get("key"),
                            outcome.get("name"),
                            normalize_team_name(outcome.get("name")),
                            outcome.get("price"),
                            point,
                            point_key,
                            outcome.get("description"),
                            outcome.get("link"),
                            outcome.get("sid"),
                        ),
                    )
                    rows += 1
    return rows


def fetch_and_store_odds(
    conn,
    client: OddsApiClient,
    sport: str,
    regions: str,
    markets: str,
    odds_format: str = "american",
    bookmakers: Optional[str] = None,
    commence_time_from: Optional[str] = None,
    commence_time_to: Optional[str] = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    fetched_at = utc_now_iso()
    try:
        payload, headers, url, params = client.get_odds(
            sport=sport,
            regions=regions,
            markets=markets,
            odds_format=odds_format,
            bookmakers=bookmakers,
            commence_time_from=commence_time_from,
            commence_time_to=commence_time_to,
        )
        safe_url = sanitize_url(url)
        insert_api_usage(
            conn,
            source="the_odds_api",
            endpoint="/v4/sports/{sport}/odds",
            fetched_at_utc=fetched_at,
            request_url=safe_url,
            status_code=200,
            headers=headers,
        )
        if dry_run:
            conn.commit()
            return {"events": len(payload), "odds_rows": 0, "fetched_at_utc": fetched_at, "dry_run": True}
        insert_raw_payload(conn, "the_odds_api", "/v4/sports/{sport}/odds", fetched_at, params, payload)
        rows = save_odds_payload(conn, payload, fetched_at)
        conn.commit()
        return {
            "events": len(payload),
            "odds_rows": rows,
            "fetched_at_utc": fetched_at,
            "requests_remaining": headers.get("x-requests-remaining"),
            "requests_used": headers.get("x-requests-used"),
            "requests_last": headers.get("x-requests-last"),
        }
    except requests.HTTPError as exc:
        response = exc.response
        insert_api_usage(
            conn,
            source="the_odds_api",
            endpoint="/v4/sports/{sport}/odds",
            fetched_at_utc=fetched_at,
            request_url=sanitize_url(getattr(response, "url", None)),
            status_code=getattr(response, "status_code", None),
            headers=getattr(response, "headers", None),
            error_message=str(exc),
        )
        conn.commit()
        raise
