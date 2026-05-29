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



def _to_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def innings_to_outs(value: Any) -> Optional[int]:
    """Convert MLB innings strings like '5.2' to outs. '5.2' means 5 innings + 2 outs."""
    if value in (None, ""):
        return None
    text = str(value)
    try:
        if "." not in text:
            return int(float(text)) * 3
        whole, frac = text.split(".", 1)
        return int(whole) * 3 + int(frac[:1] or 0)
    except (TypeError, ValueError):
        return None


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _player_id_from_key(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value)
    if text.upper().startswith("ID"):
        text = text[2:]
    return _to_int(text)


class MlbStatsClient(MlbStatsClient):  # extend the class above without changing existing callers
    def get_game_feed(self, game_pk: int) -> tuple[dict[str, Any], Mapping[str, Any], str, Mapping[str, Any]]:
        """Fetch live-feed data, which includes boxscore stats and player handedness."""
        url = f"https://statsapi.mlb.com/api/v1.1/game/{int(game_pk)}/feed/live"
        params: dict[str, Any] = {}
        response = self.session.get(url, params=params, timeout=45)
        response.raise_for_status()
        return response.json(), response.headers, response.url, params


def _person_hand(game_feed: Mapping[str, Any], player_id: int) -> Optional[str]:
    players = (((game_feed.get("gameData") or {}).get("players") or {}))
    player = players.get(f"ID{player_id}") or players.get(str(player_id)) or {}
    person = player.get("person") or player
    hand = person.get("pitchHand") or player.get("pitchHand") or {}
    code = hand.get("code") or hand.get("description")
    if code is None:
        return None
    code = str(code).upper()[:1]
    return code if code in {"L", "R", "S"} else None


def _extract_game_context_from_feed(game_feed: Mapping[str, Any]) -> dict[str, Any]:
    game_data = game_feed.get("gameData") or {}
    game = game_data.get("game") or {}
    datetime_obj = game_data.get("datetime") or {}
    teams = game_data.get("teams") or {}
    return {
        "game_pk": _to_int((game_data.get("game") or {}).get("pk") or game_feed.get("gamePk")),
        "season": _to_int(game.get("season")),
        "game_type": game.get("type"),
        "official_date": datetime_obj.get("officialDate"),
        "game_datetime_utc": datetime_obj.get("dateTime"),
        "home_team_id": _to_int(((teams.get("home") or {}).get("id"))),
        "away_team_id": _to_int(((teams.get("away") or {}).get("id"))),
    }


def _parse_team_batting_row(
    game_feed: Mapping[str, Any],
    side: str,
    fetched_at_utc: str,
) -> Optional[dict[str, Any]]:
    ctx = _extract_game_context_from_feed(game_feed)
    teams = (((game_feed.get("liveData") or {}).get("boxscore") or {}).get("teams") or {})
    side_obj = teams.get(side) or {}
    opp_side = "away" if side == "home" else "home"
    stats = ((side_obj.get("teamStats") or {}).get("batting") or {})
    team_id = _to_int(((side_obj.get("team") or {}).get("id"))) or ctx.get(f"{side}_team_id")
    opponent_team_id = ctx.get(f"{opp_side}_team_id")
    if team_id is None or not stats:
        return None
    return {
        "game_pk": ctx["game_pk"],
        "team_id": team_id,
        "opponent_team_id": opponent_team_id,
        "is_home": 1 if side == "home" else 0,
        "at_bats": _to_int(stats.get("atBats")),
        "runs": _to_int(stats.get("runs")),
        "hits": _to_int(stats.get("hits")),
        "doubles": _to_int(stats.get("doubles")),
        "triples": _to_int(stats.get("triples")),
        "home_runs": _to_int(stats.get("homeRuns")),
        "rbi": _to_int(stats.get("rbi")),
        "walks": _to_int(stats.get("baseOnBalls")),
        "strikeouts": _to_int(stats.get("strikeOuts")),
        "left_on_base": _to_int(stats.get("leftOnBase")),
        "stolen_bases": _to_int(stats.get("stolenBases")),
        "caught_stealing": _to_int(stats.get("caughtStealing")),
        "avg": _to_float(stats.get("avg")),
        "obp": _to_float(stats.get("obp")),
        "slg": _to_float(stats.get("slg")),
        "ops": _to_float(stats.get("ops")),
        "official_date": ctx["official_date"],
        "game_datetime_utc": ctx["game_datetime_utc"],
        "last_updated_utc": fetched_at_utc,
    }


def _parse_pitcher_rows(
    game_feed: Mapping[str, Any],
    side: str,
    fetched_at_utc: str,
) -> list[dict[str, Any]]:
    ctx = _extract_game_context_from_feed(game_feed)
    teams = (((game_feed.get("liveData") or {}).get("boxscore") or {}).get("teams") or {})
    side_obj = teams.get(side) or {}
    opp_side = "away" if side == "home" else "home"
    team_id = _to_int(((side_obj.get("team") or {}).get("id"))) or ctx.get(f"{side}_team_id")
    opponent_team_id = ctx.get(f"{opp_side}_team_id")
    pitcher_order = [_player_id_from_key(x) for x in (side_obj.get("pitchers") or [])]
    pitcher_order = [x for x in pitcher_order if x is not None]
    first_pitcher_id = pitcher_order[0] if pitcher_order else None
    players = side_obj.get("players") or {}
    rows: list[dict[str, Any]] = []

    for key, player in players.items():
        pitcher_id = _to_int(((player.get("person") or {}).get("id"))) or _player_id_from_key(key)
        if pitcher_id is None:
            continue
        stats = ((player.get("stats") or {}).get("pitching") or {})
        if not stats:
            continue
        outs = innings_to_outs(stats.get("inningsPitched"))
        innings = None if outs is None else outs / 3.0
        is_starter = 1 if pitcher_id == first_pitcher_id or _to_int(stats.get("gamesStarted")) == 1 else 0
        rows.append({
            "game_pk": ctx["game_pk"],
            "pitcher_id": pitcher_id,
            "team_id": team_id,
            "opponent_team_id": opponent_team_id,
            "is_home": 1 if side == "home" else 0,
            "pitcher_name": ((player.get("person") or {}).get("fullName")),
            "pitcher_hand": _person_hand(game_feed, pitcher_id),
            "is_starter": is_starter,
            "decision": stats.get("note"),
            "innings_pitched": innings,
            "outs_pitched": outs,
            "hits": _to_int(stats.get("hits")),
            "runs": _to_int(stats.get("runs")),
            "earned_runs": _to_int(stats.get("earnedRuns")),
            "walks": _to_int(stats.get("baseOnBalls")),
            "strikeouts": _to_int(stats.get("strikeOuts")),
            "home_runs": _to_int(stats.get("homeRuns")),
            "pitches_thrown": _to_int(_first_not_none(stats.get("pitchesThrown"), stats.get("numberOfPitches"))),
            "batters_faced": _to_int(stats.get("battersFaced")),
            "official_date": ctx["official_date"],
            "game_datetime_utc": ctx["game_datetime_utc"],
            "last_updated_utc": fetched_at_utc,
        })
    return rows


def parse_game_feed_stats(game_feed: Mapping[str, Any], fetched_at_utc: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pitcher_rows: list[dict[str, Any]] = []
    team_rows: list[dict[str, Any]] = []
    for side in ("home", "away"):
        pitcher_rows.extend(_parse_pitcher_rows(game_feed, side, fetched_at_utc))
        team_row = _parse_team_batting_row(game_feed, side, fetched_at_utc)
        if team_row is not None:
            team_rows.append(team_row)
    return pitcher_rows, team_rows


def fetch_game_feed_stats_to_db(conn, client: MlbStatsClient, game_pk: int) -> dict[str, Any]:
    from .db import upsert_mlb_pitcher_game_stat, upsert_mlb_team_game_stat

    fetched_at = utc_now_iso()
    payload, headers, url, params = client.get_game_feed(game_pk)
    insert_api_usage(
        conn,
        source="mlb_stats_api",
        endpoint="/api/v1.1/game/{gamePk}/feed/live",
        fetched_at_utc=fetched_at,
        request_url=url,
        status_code=200,
        headers=headers,
    )
    # Do not store the full live-feed payload by default. It is very large and will bloat SQLite/Git.
    # The parsed pitcher/team stat rows below are the durable data we need for modeling.
    pitcher_rows, team_rows = parse_game_feed_stats(payload, fetched_at)
    for row in pitcher_rows:
        if row.get("game_pk") is not None and row.get("pitcher_id") is not None and row.get("team_id") is not None:
            upsert_mlb_pitcher_game_stat(conn, row)
    for row in team_rows:
        if row.get("game_pk") is not None and row.get("team_id") is not None:
            upsert_mlb_team_game_stat(conn, row)
    conn.commit()
    return {"game_pk": game_pk, "pitcher_rows": len(pitcher_rows), "team_rows": len(team_rows), "fetched_at_utc": fetched_at}
