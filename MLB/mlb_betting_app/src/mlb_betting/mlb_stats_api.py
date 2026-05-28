from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterator, Mapping, Optional

import requests

from .db import insert_api_usage, insert_raw_payload
from .team_mapping import normalize_team_name

LOGGER = logging.getLogger(__name__)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def to_mlb_date(value: date | datetime | str) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


class MlbStatsClient:
    def __init__(self, base_url: str = "https://statsapi.mlb.com/api/v1") -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def get_schedule(
        self,
        start_date: date | datetime | str,
        end_date: date | datetime | str,
        sport_id: int = 1,
        hydrate: str = "team,probablePitcher,venue",
        game_type: Optional[str] = None,
    ) -> tuple[dict[str, Any], Mapping[str, Any], str, Mapping[str, Any]]:
        url = f"{self.base_url}/schedule"
        params: dict[str, Any] = {
            "sportId": sport_id,
            "startDate": to_mlb_date(start_date),
            "endDate": to_mlb_date(end_date),
            "hydrate": hydrate,
        }
        if game_type:
            params["gameType"] = game_type
        response = self.session.get(url, params=params, timeout=45)
        response.raise_for_status()
        return response.json(), response.headers, response.url, params


def iter_schedule_games(payload: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    for date_obj in payload.get("dates", []) or []:
        for game in date_obj.get("games", []) or []:
            yield game


def _get_team_side(game: Mapping[str, Any], side: str) -> Mapping[str, Any]:
    return ((game.get("teams") or {}).get(side) or {})


def _get_team_name(game: Mapping[str, Any], side: str) -> Optional[str]:
    team = (_get_team_side(game, side).get("team") or {})
    return team.get("name")


def _get_team_id(game: Mapping[str, Any], side: str) -> Optional[int]:
    team = (_get_team_side(game, side).get("team") or {})
    return team.get("id")


def _get_score(game: Mapping[str, Any], side: str) -> Optional[int]:
    value = _get_team_side(game, side).get("score")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _get_probable_pitcher(game: Mapping[str, Any], side: str) -> tuple[Optional[int], Optional[str]]:
    pitcher = _get_team_side(game, side).get("probablePitcher") or {}
    return pitcher.get("id"), pitcher.get("fullName")


def parse_schedule_game(game: Mapping[str, Any], fetched_at_utc: str) -> dict[str, Any]:
    status = game.get("status") or {}
    venue = game.get("venue") or {}
    home_name = _get_team_name(game, "home")
    away_name = _get_team_name(game, "away")
    home_score = _get_score(game, "home")
    away_score = _get_score(game, "away")
    completed = home_score is not None and away_score is not None and (status.get("abstractGameState") == "Final" or status.get("codedGameState") == "F")
    home_win = None
    home_margin = None
    total_runs = None
    if completed:
        home_margin = int(home_score) - int(away_score)
        home_win = 1 if home_margin > 0 else 0
        total_runs = int(home_score) + int(away_score)

    home_pitcher_id, home_pitcher_name = _get_probable_pitcher(game, "home")
    away_pitcher_id, away_pitcher_name = _get_probable_pitcher(game, "away")

    return {
        "game_pk": game.get("gamePk"),
        "game_guid": game.get("gameGuid"),
        "season": int(game.get("season")) if game.get("season") else None,
        "game_type": game.get("gameType"),
        "game_date": game.get("gameDate"),
        "official_date": game.get("officialDate"),
        "game_datetime_utc": game.get("gameDate"),
        "status_code": status.get("codedGameState"),
        "detailed_state": status.get("detailedState"),
        "abstract_state": status.get("abstractGameState"),
        "venue_id": venue.get("id"),
        "venue_name": venue.get("name"),
        "home_team_id": _get_team_id(game, "home"),
        "home_team_name": home_name,
        "home_team_norm": normalize_team_name(home_name),
        "away_team_id": _get_team_id(game, "away"),
        "away_team_name": away_name,
        "away_team_norm": normalize_team_name(away_name),
        "home_score": home_score,
        "away_score": away_score,
        "target_home_win": home_win,
        "home_margin": home_margin,
        "total_runs": total_runs,
        "probable_home_pitcher_id": home_pitcher_id,
        "probable_home_pitcher_name": home_pitcher_name,
        "probable_away_pitcher_id": away_pitcher_id,
        "probable_away_pitcher_name": away_pitcher_name,
        "last_updated_utc": fetched_at_utc,
    }


def upsert_mlb_game(conn, row: Mapping[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO mlb_games (
            game_pk, game_guid, season, game_type, game_date, official_date,
            game_datetime_utc, status_code, detailed_state, abstract_state,
            venue_id, venue_name, home_team_id, home_team_name, home_team_norm,
            away_team_id, away_team_name, away_team_norm, home_score, away_score,
            target_home_win, home_margin, total_runs,
            probable_home_pitcher_id, probable_home_pitcher_name,
            probable_away_pitcher_id, probable_away_pitcher_name, last_updated_utc
        ) VALUES (
            :game_pk, :game_guid, :season, :game_type, :game_date, :official_date,
            :game_datetime_utc, :status_code, :detailed_state, :abstract_state,
            :venue_id, :venue_name, :home_team_id, :home_team_name, :home_team_norm,
            :away_team_id, :away_team_name, :away_team_norm, :home_score, :away_score,
            :target_home_win, :home_margin, :total_runs,
            :probable_home_pitcher_id, :probable_home_pitcher_name,
            :probable_away_pitcher_id, :probable_away_pitcher_name, :last_updated_utc
        )
        ON CONFLICT(game_pk) DO UPDATE SET
            game_guid=excluded.game_guid,
            season=excluded.season,
            game_type=excluded.game_type,
            game_date=excluded.game_date,
            official_date=excluded.official_date,
            game_datetime_utc=excluded.game_datetime_utc,
            status_code=excluded.status_code,
            detailed_state=excluded.detailed_state,
            abstract_state=excluded.abstract_state,
            venue_id=excluded.venue_id,
            venue_name=excluded.venue_name,
            home_team_id=excluded.home_team_id,
            home_team_name=excluded.home_team_name,
            home_team_norm=excluded.home_team_norm,
            away_team_id=excluded.away_team_id,
            away_team_name=excluded.away_team_name,
            away_team_norm=excluded.away_team_norm,
            home_score=excluded.home_score,
            away_score=excluded.away_score,
            target_home_win=excluded.target_home_win,
            home_margin=excluded.home_margin,
            total_runs=excluded.total_runs,
            probable_home_pitcher_id=excluded.probable_home_pitcher_id,
            probable_home_pitcher_name=excluded.probable_home_pitcher_name,
            probable_away_pitcher_id=excluded.probable_away_pitcher_id,
            probable_away_pitcher_name=excluded.probable_away_pitcher_name,
            last_updated_utc=excluded.last_updated_utc
        """,
        dict(row),
    )


def fetch_schedule_to_db(
    conn,
    client: MlbStatsClient,
    start_date: date | datetime | str,
    end_date: date | datetime | str,
    game_type: Optional[str] = None,
) -> dict[str, Any]:
    fetched_at = utc_now_iso()
    payload, headers, url, params = client.get_schedule(start_date, end_date, game_type=game_type)
    insert_api_usage(
        conn,
        source="mlb_stats_api",
        endpoint="/api/v1/schedule",
        fetched_at_utc=fetched_at,
        request_url=url,
        status_code=200,
        headers=headers,
    )
    insert_raw_payload(conn, "mlb_stats_api", "/api/v1/schedule", fetched_at, params, payload)
    rows = 0
    for game in iter_schedule_games(payload):
        row = parse_schedule_game(game, fetched_at)
        if row.get("game_pk") is None:
            continue
        upsert_mlb_game(conn, row)
        rows += 1
    conn.commit()
    return {"games": rows, "fetched_at_utc": fetched_at, "start_date": str(start_date), "end_date": str(end_date)}


def date_range_from_days(days_back: int, days_forward: int) -> tuple[str, str]:
    today = datetime.now(timezone.utc).date()
    return (today - timedelta(days=days_back)).isoformat(), (today + timedelta(days=days_forward)).isoformat()
