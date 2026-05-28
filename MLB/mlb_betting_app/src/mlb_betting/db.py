from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, Mapping, Optional

import pandas as pd

from .team_mapping import normalize_team_name

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS api_usage_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    fetched_at_utc TEXT NOT NULL,
    request_url TEXT,
    status_code INTEGER,
    requests_remaining INTEGER,
    requests_used INTEGER,
    requests_last INTEGER,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS raw_api_payloads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    fetched_at_utc TEXT NOT NULL,
    params_json TEXT,
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS odds_events (
    event_id TEXT PRIMARY KEY,
    sport_key TEXT NOT NULL,
    sport_title TEXT,
    commence_time_utc TEXT,
    home_team TEXT,
    away_team TEXT,
    home_team_norm TEXT,
    away_team_norm TEXT,
    last_seen_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS odds_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    fetched_at_utc TEXT NOT NULL,
    event_id TEXT NOT NULL,
    sport_key TEXT NOT NULL,
    commence_time_utc TEXT,
    home_team TEXT,
    away_team TEXT,
    bookmaker_key TEXT NOT NULL,
    bookmaker_title TEXT,
    bookmaker_last_update_utc TEXT,
    market_key TEXT NOT NULL,
    outcome_name TEXT NOT NULL,
    outcome_name_norm TEXT,
    outcome_price REAL,
    outcome_point REAL,
    outcome_point_key TEXT NOT NULL,
    outcome_description TEXT,
    outcome_link TEXT,
    outcome_sid TEXT,
    FOREIGN KEY(event_id) REFERENCES odds_events(event_id)
);

CREATE INDEX IF NOT EXISTS idx_odds_snapshots_event_market
    ON odds_snapshots(event_id, market_key, fetched_at_utc);
CREATE INDEX IF NOT EXISTS idx_odds_snapshots_bookmaker
    ON odds_snapshots(bookmaker_key, market_key, fetched_at_utc);
CREATE INDEX IF NOT EXISTS idx_odds_events_teams_date
    ON odds_events(home_team_norm, away_team_norm, commence_time_utc);

CREATE TABLE IF NOT EXISTS mlb_games (
    game_pk INTEGER PRIMARY KEY,
    game_guid TEXT,
    season INTEGER,
    game_type TEXT,
    game_date TEXT,
    official_date TEXT,
    game_datetime_utc TEXT,
    status_code TEXT,
    detailed_state TEXT,
    abstract_state TEXT,
    venue_id INTEGER,
    venue_name TEXT,
    home_team_id INTEGER,
    home_team_name TEXT,
    home_team_norm TEXT,
    away_team_id INTEGER,
    away_team_name TEXT,
    away_team_norm TEXT,
    home_score INTEGER,
    away_score INTEGER,
    target_home_win INTEGER,
    home_margin INTEGER,
    total_runs INTEGER,
    probable_home_pitcher_id INTEGER,
    probable_home_pitcher_name TEXT,
    probable_away_pitcher_id INTEGER,
    probable_away_pitcher_name TEXT,
    last_updated_utc TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_mlb_games_date ON mlb_games(official_date);
CREATE INDEX IF NOT EXISTS idx_mlb_games_teams ON mlb_games(home_team_norm, away_team_norm, official_date);

CREATE TABLE IF NOT EXISTS model_runs (
    run_id TEXT PRIMARY KEY,
    created_at_utc TEXT NOT NULL,
    model_name TEXT NOT NULL,
    target TEXT NOT NULL,
    train_start_date TEXT,
    train_end_date TEXT,
    test_start_date TEXT,
    test_end_date TEXT,
    n_train INTEGER,
    n_test INTEGER,
    metrics_json TEXT,
    params_json TEXT,
    feature_columns_json TEXT,
    artifact_path TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    scored_at_utc TEXT NOT NULL,
    game_pk INTEGER,
    official_date TEXT,
    game_datetime_utc TEXT,
    home_team_name TEXT,
    away_team_name TEXT,
    model_home_win_prob REAL,
    market_home_no_vig_prob REAL,
    home_moneyline_median REAL,
    away_moneyline_median REAL,
    recommended_side TEXT,
    recommended_price REAL,
    edge REAL,
    expected_value_per_unit REAL,
    feature_snapshot_json TEXT,
    FOREIGN KEY(game_pk) REFERENCES mlb_games(game_pk)
);
"""


def connect(db_path: Path | str) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | str) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.commit()


def read_sql(conn: sqlite3.Connection, query: str, params: Optional[Mapping] = None) -> pd.DataFrame:
    return pd.read_sql_query(query, conn, params=params or {})


def upsert_odds_event(conn: sqlite3.Connection, event: Mapping, fetched_at_utc: str) -> None:
    conn.execute(
        """
        INSERT INTO odds_events (
            event_id, sport_key, sport_title, commence_time_utc,
            home_team, away_team, home_team_norm, away_team_norm, last_seen_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(event_id) DO UPDATE SET
            sport_key=excluded.sport_key,
            sport_title=excluded.sport_title,
            commence_time_utc=excluded.commence_time_utc,
            home_team=excluded.home_team,
            away_team=excluded.away_team,
            home_team_norm=excluded.home_team_norm,
            away_team_norm=excluded.away_team_norm,
            last_seen_utc=excluded.last_seen_utc
        """,
        (
            event.get("id"),
            event.get("sport_key"),
            event.get("sport_title"),
            event.get("commence_time"),
            event.get("home_team"),
            event.get("away_team"),
            normalize_team_name(event.get("home_team")),
            normalize_team_name(event.get("away_team")),
            fetched_at_utc,
        ),
    )


def insert_api_usage(
    conn: sqlite3.Connection,
    source: str,
    endpoint: str,
    fetched_at_utc: str,
    request_url: Optional[str],
    status_code: Optional[int],
    headers: Optional[Mapping] = None,
    error_message: Optional[str] = None,
) -> None:
    headers = headers or {}
    def get_int(name: str) -> Optional[int]:
        value = headers.get(name) or headers.get(name.lower())
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    conn.execute(
        """
        INSERT INTO api_usage_log (
            source, endpoint, fetched_at_utc, request_url, status_code,
            requests_remaining, requests_used, requests_last, error_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            source,
            endpoint,
            fetched_at_utc,
            request_url,
            status_code,
            get_int("x-requests-remaining"),
            get_int("x-requests-used"),
            get_int("x-requests-last"),
            error_message,
        ),
    )


def insert_raw_payload(
    conn: sqlite3.Connection,
    source: str,
    endpoint: str,
    fetched_at_utc: str,
    params: Mapping,
    payload: object,
) -> None:
    conn.execute(
        """
        INSERT INTO raw_api_payloads (source, endpoint, fetched_at_utc, params_json, payload_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (source, endpoint, fetched_at_utc, json.dumps(params, sort_keys=True), json.dumps(payload)),
    )


def insert_prediction_rows(conn: sqlite3.Connection, rows: Iterable[Mapping]) -> int:
    count = 0
    for row in rows:
        conn.execute(
            """
            INSERT INTO predictions (
                run_id, scored_at_utc, game_pk, official_date, game_datetime_utc,
                home_team_name, away_team_name, model_home_win_prob,
                market_home_no_vig_prob, home_moneyline_median, away_moneyline_median,
                recommended_side, recommended_price, edge, expected_value_per_unit,
                feature_snapshot_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.get("run_id"), row.get("scored_at_utc"), row.get("game_pk"),
                row.get("official_date"), row.get("game_datetime_utc"),
                row.get("home_team_name"), row.get("away_team_name"),
                row.get("model_home_win_prob"), row.get("market_home_no_vig_prob"),
                row.get("home_moneyline_median"), row.get("away_moneyline_median"),
                row.get("recommended_side"), row.get("recommended_price"),
                row.get("edge"), row.get("expected_value_per_unit"),
                row.get("feature_snapshot_json"),
            ),
        )
        count += 1
    return count
