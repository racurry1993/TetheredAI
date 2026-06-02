from __future__ import annotations

"""Fetch and aggregate Statcast data into odds.db.

This script does not call The Odds API. It uses pybaseball's Statcast wrapper
and stores compact game-level aggregates that the feature pipeline can consume.

Recommended first test in GitHub Actions:
    python scripts/09_fetch_statcast.py --days-back 14 --days-forward 0 --chunk-days 3 --limit-chunks 1

Recommended full backfill, run sparingly:
    python scripts/09_fetch_statcast.py --days-back 730 --chunk-days 3
"""

import argparse
import math
import os
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlb_betting.config import get_settings
from mlb_betting.db import connect, init_db

STATCAST_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS mlb_statcast_team_game (
    game_pk INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    opponent_team_id INTEGER,
    is_home INTEGER,
    official_date TEXT,
    game_datetime_utc TEXT,
    sc_pa INTEGER,
    sc_pitches_seen INTEGER,
    sc_bbe INTEGER,
    sc_woba REAL,
    sc_xwoba_contact REAL,
    sc_avg_ev REAL,
    sc_max_ev REAL,
    sc_avg_la REAL,
    sc_avg_batted_ball_ev REAL,
    sc_median_batted_ball_ev REAL,
    sc_p90_batted_ball_ev REAL,
    sc_max_batted_ball_ev REAL,
    sc_avg_batted_ball_distance REAL,
    sc_median_batted_ball_distance REAL,
    sc_p90_batted_ball_distance REAL,
    sc_max_batted_ball_distance REAL,
    sc_batted_ball_distance_count INTEGER,
    sc_hard_hit_rate REAL,
    sc_barrel_rate REAL,
    sc_sweetspot_rate REAL,
    sc_whiff_rate REAL,
    sc_csw_rate REAL,
    sc_k_rate REAL,
    sc_bb_rate REAL,
    sc_hr_rate REAL,
    last_updated_utc TEXT NOT NULL,
    PRIMARY KEY(game_pk, team_id)
);

CREATE INDEX IF NOT EXISTS idx_statcast_team_game_team_date
    ON mlb_statcast_team_game(team_id, game_datetime_utc);

CREATE TABLE IF NOT EXISTS mlb_statcast_team_hand_game (
    game_pk INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    opponent_team_id INTEGER,
    is_home INTEGER,
    pitcher_hand TEXT NOT NULL,
    official_date TEXT,
    game_datetime_utc TEXT,
    sc_pa INTEGER,
    sc_pitches_seen INTEGER,
    sc_bbe INTEGER,
    sc_woba REAL,
    sc_xwoba_contact REAL,
    sc_avg_ev REAL,
    sc_max_ev REAL,
    sc_avg_la REAL,
    sc_avg_batted_ball_ev REAL,
    sc_median_batted_ball_ev REAL,
    sc_p90_batted_ball_ev REAL,
    sc_max_batted_ball_ev REAL,
    sc_avg_batted_ball_distance REAL,
    sc_median_batted_ball_distance REAL,
    sc_p90_batted_ball_distance REAL,
    sc_max_batted_ball_distance REAL,
    sc_batted_ball_distance_count INTEGER,
    sc_hard_hit_rate REAL,
    sc_barrel_rate REAL,
    sc_sweetspot_rate REAL,
    sc_whiff_rate REAL,
    sc_csw_rate REAL,
    sc_k_rate REAL,
    sc_bb_rate REAL,
    sc_hr_rate REAL,
    last_updated_utc TEXT NOT NULL,
    PRIMARY KEY(game_pk, team_id, pitcher_hand)
);

CREATE INDEX IF NOT EXISTS idx_statcast_team_hand_team_hand_date
    ON mlb_statcast_team_hand_game(team_id, pitcher_hand, game_datetime_utc);

CREATE TABLE IF NOT EXISTS mlb_statcast_pitcher_game (
    game_pk INTEGER NOT NULL,
    pitcher_id INTEGER NOT NULL,
    team_id INTEGER,
    opponent_team_id INTEGER,
    is_home INTEGER,
    pitcher_name TEXT,
    pitcher_hand TEXT,
    is_starter INTEGER,
    official_date TEXT,
    game_datetime_utc TEXT,
    sc_pitches INTEGER,
    sc_pa INTEGER,
    sc_bbe_allowed INTEGER,
    sc_release_speed_mean REAL,
    sc_release_speed_max REAL,
    sc_release_spin_mean REAL,
    sc_release_extension_mean REAL,
    sc_pitch_mix_entropy REAL,
    sc_fastball_pct REAL,
    sc_breaking_pct REAL,
    sc_offspeed_pct REAL,
    sc_zone_rate REAL,
    sc_whiff_rate REAL,
    sc_csw_rate REAL,
    sc_called_strike_rate REAL,
    sc_xwoba_allowed_contact REAL,
    sc_woba_allowed REAL,
    sc_avg_ev_allowed REAL,
    sc_max_ev_allowed REAL,
    sc_avg_la_allowed REAL,
    sc_avg_batted_ball_ev_allowed REAL,
    sc_median_batted_ball_ev_allowed REAL,
    sc_p90_batted_ball_ev_allowed REAL,
    sc_max_batted_ball_ev_allowed REAL,
    sc_avg_batted_ball_distance_allowed REAL,
    sc_median_batted_ball_distance_allowed REAL,
    sc_p90_batted_ball_distance_allowed REAL,
    sc_max_batted_ball_distance_allowed REAL,
    sc_batted_ball_distance_allowed_count INTEGER,
    sc_hard_hit_rate_allowed REAL,
    sc_barrel_rate_allowed REAL,
    sc_sweetspot_rate_allowed REAL,
    sc_k_rate REAL,
    sc_bb_rate REAL,
    sc_hr_rate REAL,
    last_updated_utc TEXT NOT NULL,
    PRIMARY KEY(game_pk, pitcher_id, team_id)
);

CREATE INDEX IF NOT EXISTS idx_statcast_pitcher_game_pitcher_date
    ON mlb_statcast_pitcher_game(pitcher_id, game_datetime_utc);
CREATE INDEX IF NOT EXISTS idx_statcast_pitcher_game_team_date
    ON mlb_statcast_pitcher_game(team_id, game_datetime_utc);
"""

SWING_DESCRIPTIONS = {
    "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play",
    "hit_into_play_no_out", "hit_into_play_score", "foul_bunt", "missed_bunt", "bunt_foul_tip",
}
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
CALLED_STRIKE_DESCRIPTIONS = {"called_strike"}
WALK_EVENTS = {"walk", "intent_walk"}
K_EVENTS = {"strikeout", "strikeout_double_play"}
HR_EVENTS = {"home_run"}
FASTBALL_TYPES = {"FF", "SI", "FT", "FC", "FA"}
BREAKING_TYPES = {"SL", "CU", "KC", "ST", "SV"}
OFFSPEED_TYPES = {"CH", "FS", "FO", "SC", "KN"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", help="YYYY-MM-DD. Overrides days-back if provided.")
    parser.add_argument("--end-date", help="YYYY-MM-DD. Defaults to today UTC.")
    parser.add_argument("--days-back", type=int, default=30)
    parser.add_argument("--days-forward", type=int, default=0, help="Usually 0; Statcast only exists for completed games.")
    parser.add_argument("--chunk-days", type=int, default=3, help="Small chunks reduce pybaseball timeout risk.")
    parser.add_argument("--limit-chunks", type=int, default=None, help="Debug option to limit API calls.")
    parser.add_argument("--sleep", type=float, default=0.5, help="Pause between chunks.")
    parser.add_argument("--parallel", action="store_true", help="Pass parallel=True to pybaseball.statcast.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip a Statcast chunk when mlb_statcast_team_game already has rows "
            "for every official_date in that chunk. Useful when rerunning failed workflows."
        ),
    )
    return parser.parse_args()


def _qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _existing_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({_qident(table)})").fetchall()}


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: Mapping[str, str]) -> None:
    existing = _existing_columns(conn, table)
    for col, col_type in columns.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE {_qident(table)} ADD COLUMN {_qident(col)} {col_type}")


def init_statcast_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(STATCAST_SCHEMA_SQL)

    # CREATE TABLE IF NOT EXISTS will not add new columns to an existing odds.db.
    # These ALTERs make the batted-ball distance/velocity upgrade backward compatible.
    team_new_cols = {
        "sc_avg_batted_ball_ev": "REAL",
        "sc_median_batted_ball_ev": "REAL",
        "sc_p90_batted_ball_ev": "REAL",
        "sc_max_batted_ball_ev": "REAL",
        "sc_avg_batted_ball_distance": "REAL",
        "sc_median_batted_ball_distance": "REAL",
        "sc_p90_batted_ball_distance": "REAL",
        "sc_max_batted_ball_distance": "REAL",
        "sc_batted_ball_distance_count": "INTEGER",
    }
    pitcher_new_cols = {
        "sc_avg_batted_ball_ev_allowed": "REAL",
        "sc_median_batted_ball_ev_allowed": "REAL",
        "sc_p90_batted_ball_ev_allowed": "REAL",
        "sc_max_batted_ball_ev_allowed": "REAL",
        "sc_avg_batted_ball_distance_allowed": "REAL",
        "sc_median_batted_ball_distance_allowed": "REAL",
        "sc_p90_batted_ball_distance_allowed": "REAL",
        "sc_max_batted_ball_distance_allowed": "REAL",
        "sc_batted_ball_distance_allowed_count": "INTEGER",
    }
    _ensure_columns(conn, "mlb_statcast_team_game", team_new_cols)
    _ensure_columns(conn, "mlb_statcast_team_hand_game", team_new_cols)
    _ensure_columns(conn, "mlb_statcast_pitcher_game", pitcher_new_cols)
    conn.commit()


def date_chunks(start: date, end: date, chunk_days: int) -> Iterable[tuple[date, date]]:
    cur = start
    while cur <= end:
        chunk_end = min(end, cur + timedelta(days=chunk_days - 1))
        yield cur, chunk_end
        cur = chunk_end + timedelta(days=1)


def chunk_has_existing_statcast(conn: sqlite3.Connection, chunk_start: date, chunk_end: date) -> bool:
    """Return True when a chunk appears already loaded in mlb_statcast_team_game.

    This is intentionally conservative: it requires at least one row for every
    date in the chunk. The underlying Statcast table is keyed by game/team, so
    reruns are safe even when we do fetch a partially loaded chunk.
    """
    expected_dates = {
        (chunk_start + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range((chunk_end - chunk_start).days + 1)
    }
    if not expected_dates:
        return False

    try:
        rows = conn.execute(
            """
            SELECT DISTINCT official_date
            FROM mlb_statcast_team_game
            WHERE official_date >= ? AND official_date <= ?
            """,
            (chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")),
        ).fetchall()
    except sqlite3.Error:
        return False

    loaded_dates = {str(row[0]) for row in rows if row[0] is not None}
    return expected_dates.issubset(loaded_dates)


def load_games(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query("SELECT * FROM mlb_games", conn)
    if df.empty:
        return df
    df["game_datetime_utc"] = pd.to_datetime(df["game_datetime_utc"], utc=True, errors="coerce")
    df["official_date"] = pd.to_datetime(df["official_date"], errors="coerce").dt.date.astype(str)
    return df


def load_starters(conn: sqlite3.Connection) -> pd.DataFrame:
    try:
        df = pd.read_sql_query(
            "SELECT game_pk, pitcher_id, team_id, is_starter, pitcher_hand FROM mlb_pitcher_game_stats", conn
        )
    except Exception:
        return pd.DataFrame()
    return df


def prepare_statcast(raw: pd.DataFrame, games: pd.DataFrame, starters: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw

    df = raw.copy()
    if "game_pk" not in df.columns:
        return pd.DataFrame()

    keep_game_cols = [
        "game_pk", "official_date", "game_datetime_utc", "home_team_id", "away_team_id",
        "home_team_name", "away_team_name",
    ]
    games_small = games[[c for c in keep_game_cols if c in games.columns]].drop_duplicates("game_pk")
    df = df.merge(games_small, on="game_pk", how="inner")

    if df.empty:
        return df

    topbot = df.get("inning_topbot", pd.Series(index=df.index, dtype="object")).astype(str).str.lower()
    is_top = topbot.str.startswith("top")
    df["batting_team_id"] = np.where(is_top, df["away_team_id"], df["home_team_id"])
    df["pitching_team_id"] = np.where(is_top, df["home_team_id"], df["away_team_id"])
    df["batting_is_home"] = np.where(is_top, 0, 1)
    df["pitching_is_home"] = np.where(is_top, 1, 0)
    df["batting_opponent_team_id"] = df["pitching_team_id"]
    df["pitching_opponent_team_id"] = df["batting_team_id"]

    # Normalize important fields.
    for col in [
        "pitcher", "batter", "release_speed", "release_spin_rate", "release_extension", "launch_speed",
        "launch_angle", "hit_distance_sc", "hit_distance", "estimated_woba_using_speedangle", "woba_value", "woba_denom", "zone", "launch_speed_angle",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["is_pa"] = df.get("events", pd.Series(index=df.index)).notna()
    df["is_bbe"] = df.get("launch_speed", pd.Series(index=df.index)).notna()
    df["is_hard_hit"] = df.get("launch_speed", pd.Series(index=df.index)).ge(95)
    df["is_barrel"] = df.get("launch_speed_angle", pd.Series(index=df.index)).eq(6)
    df["is_sweetspot"] = df.get("launch_angle", pd.Series(index=df.index)).between(8, 32)
    desc = df.get("description", pd.Series(index=df.index, dtype="object")).astype(str)
    events = df.get("events", pd.Series(index=df.index, dtype="object")).astype(str)
    df["is_swing"] = desc.isin(SWING_DESCRIPTIONS)
    df["is_whiff"] = desc.isin(WHIFF_DESCRIPTIONS)
    df["is_called_strike"] = desc.isin(CALLED_STRIKE_DESCRIPTIONS)
    df["is_csw"] = df["is_whiff"] | df["is_called_strike"]
    df["is_zone"] = df.get("zone", pd.Series(index=df.index)).between(1, 9)
    df["is_k"] = events.isin(K_EVENTS)
    df["is_bb"] = events.isin(WALK_EVENTS)
    df["is_hr"] = events.isin(HR_EVENTS)
    df["pitcher_hand"] = df.get("p_throws", pd.Series(index=df.index, dtype="object")).fillna("UNK").astype(str)

    if not starters.empty:
        st = starters[["game_pk", "pitcher_id", "team_id", "is_starter", "pitcher_hand"]].copy()
        st = st.rename(columns={"pitcher_id": "pitcher", "team_id": "pitching_team_id", "pitcher_hand": "starter_table_hand"})
        df = df.merge(st, on=["game_pk", "pitcher", "pitching_team_id"], how="left")
        df["is_starter"] = pd.to_numeric(df["is_starter"], errors="coerce").fillna(0).astype(int)
        df["pitcher_hand"] = df["pitcher_hand"].replace("nan", np.nan).fillna(df.get("starter_table_hand")).fillna("UNK")
    else:
        df["is_starter"] = 0

    return df




def _safe_float(value) -> float:
    """Convert numeric scalars to float while tolerating np.nan/pd.NA/None."""
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def _numeric_series(frame: pd.DataFrame, col: str) -> pd.Series:
    """Return a numeric Series for col, or an empty numeric Series if missing."""
    if col not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[col], errors="coerce")


def _mean(frame: pd.DataFrame, col: str) -> float:
    return _safe_float(_numeric_series(frame, col).mean(skipna=True))


def _max(frame: pd.DataFrame, col: str) -> float:
    return _safe_float(_numeric_series(frame, col).max(skipna=True))


def _median(frame: pd.DataFrame, col: str) -> float:
    return _safe_float(_numeric_series(frame, col).median(skipna=True))


def _quantile(frame: pd.DataFrame, col: str, q: float) -> float:
    s = _numeric_series(frame, col).dropna()
    if s.empty:
        return np.nan
    return _safe_float(s.quantile(q))


def _first_present_col(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in frame.columns:
            return col
    return None


def _mean_any(frame: pd.DataFrame, candidates: list[str]) -> float:
    col = _first_present_col(frame, candidates)
    return _mean(frame, col) if col else np.nan


def _median_any(frame: pd.DataFrame, candidates: list[str]) -> float:
    col = _first_present_col(frame, candidates)
    return _median(frame, col) if col else np.nan


def _quantile_any(frame: pd.DataFrame, candidates: list[str], q: float) -> float:
    col = _first_present_col(frame, candidates)
    return _quantile(frame, col, q) if col else np.nan


def _max_any(frame: pd.DataFrame, candidates: list[str]) -> float:
    col = _first_present_col(frame, candidates)
    return _max(frame, col) if col else np.nan


def _nonnull_count_any(frame: pd.DataFrame, candidates: list[str]) -> int:
    col = _first_present_col(frame, candidates)
    if not col:
        return 0
    return int(_numeric_series(frame, col).notna().sum())

def _rate(num: float, den: float) -> float:
    try:
        if den and den > 0:
            return float(num / den)
    except Exception:
        pass
    return np.nan


def _entropy(values: pd.Series) -> float:
    counts = values.dropna().value_counts()
    if counts.empty:
        return np.nan
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def _pitch_mix_pct(values: pd.Series, pitch_set: set[str]) -> float:
    valid = values.dropna().astype(str)
    if valid.empty:
        return np.nan
    return float(valid.isin(pitch_set).mean())


def aggregate_team_game(df: pd.DataFrame, fetched_at: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for (game_pk, team_id), g in df.groupby(["game_pk", "batting_team_id"], dropna=False, sort=False):
        pa = g[g["is_pa"]]
        bbe = g[g["is_bbe"]]
        row = {
            "game_pk": int(game_pk),
            "team_id": int(team_id) if pd.notna(team_id) else None,
            "opponent_team_id": int(g["batting_opponent_team_id"].iloc[0]) if pd.notna(g["batting_opponent_team_id"].iloc[0]) else None,
            "is_home": int(g["batting_is_home"].iloc[0]) if pd.notna(g["batting_is_home"].iloc[0]) else None,
            "official_date": g["official_date"].iloc[0],
            "game_datetime_utc": str(g["game_datetime_utc"].iloc[0]),
            "sc_pa": int(pa.shape[0]),
            "sc_pitches_seen": int(g.shape[0]),
            "sc_bbe": int(bbe.shape[0]),
            "sc_woba": _rate(pa.get("woba_value", pd.Series(dtype=float)).sum(skipna=True), pa.get("woba_denom", pd.Series(dtype=float)).sum(skipna=True)),
            "sc_xwoba_contact": _mean(bbe, "estimated_woba_using_speedangle") if not bbe.empty else np.nan,
            "sc_avg_ev": _mean(bbe, "launch_speed") if not bbe.empty else np.nan,
            "sc_max_ev": _max(bbe, "launch_speed") if not bbe.empty else np.nan,
            "sc_avg_la": _mean(bbe, "launch_angle") if not bbe.empty else np.nan,
            "sc_avg_batted_ball_ev": _mean(bbe, "launch_speed") if not bbe.empty else np.nan,
            "sc_median_batted_ball_ev": _median(bbe, "launch_speed") if not bbe.empty else np.nan,
            "sc_p90_batted_ball_ev": _quantile(bbe, "launch_speed", 0.90) if not bbe.empty else np.nan,
            "sc_max_batted_ball_ev": _max(bbe, "launch_speed") if not bbe.empty else np.nan,
            "sc_avg_batted_ball_distance": _mean_any(bbe, ["hit_distance_sc", "hit_distance"]) if not bbe.empty else np.nan,
            "sc_median_batted_ball_distance": _median_any(bbe, ["hit_distance_sc", "hit_distance"]) if not bbe.empty else np.nan,
            "sc_p90_batted_ball_distance": _quantile_any(bbe, ["hit_distance_sc", "hit_distance"], 0.90) if not bbe.empty else np.nan,
            "sc_max_batted_ball_distance": _max_any(bbe, ["hit_distance_sc", "hit_distance"]) if not bbe.empty else np.nan,
            "sc_batted_ball_distance_count": _nonnull_count_any(bbe, ["hit_distance_sc", "hit_distance"]) if not bbe.empty else 0,
            "sc_hard_hit_rate": _rate(g["is_hard_hit"].sum(), bbe.shape[0]),
            "sc_barrel_rate": _rate(g["is_barrel"].sum(), bbe.shape[0]),
            "sc_sweetspot_rate": _rate(g["is_sweetspot"].sum(), bbe.shape[0]),
            "sc_whiff_rate": _rate(g["is_whiff"].sum(), g["is_swing"].sum()),
            "sc_csw_rate": _rate(g["is_csw"].sum(), g.shape[0]),
            "sc_k_rate": _rate(pa["is_k"].sum(), pa.shape[0]),
            "sc_bb_rate": _rate(pa["is_bb"].sum(), pa.shape[0]),
            "sc_hr_rate": _rate(pa["is_hr"].sum(), pa.shape[0]),
            "last_updated_utc": fetched_at,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_team_hand_game(df: pd.DataFrame, fetched_at: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for (game_pk, team_id, hand), g in df.groupby(["game_pk", "batting_team_id", "pitcher_hand"], dropna=False, sort=False):
        pa = g[g["is_pa"]]
        bbe = g[g["is_bbe"]]
        row = {
            "game_pk": int(game_pk),
            "team_id": int(team_id) if pd.notna(team_id) else None,
            "opponent_team_id": int(g["batting_opponent_team_id"].iloc[0]) if pd.notna(g["batting_opponent_team_id"].iloc[0]) else None,
            "is_home": int(g["batting_is_home"].iloc[0]) if pd.notna(g["batting_is_home"].iloc[0]) else None,
            "pitcher_hand": str(hand) if pd.notna(hand) else "UNK",
            "official_date": g["official_date"].iloc[0],
            "game_datetime_utc": str(g["game_datetime_utc"].iloc[0]),
            "sc_pa": int(pa.shape[0]),
            "sc_pitches_seen": int(g.shape[0]),
            "sc_bbe": int(bbe.shape[0]),
            "sc_woba": _rate(pa.get("woba_value", pd.Series(dtype=float)).sum(skipna=True), pa.get("woba_denom", pd.Series(dtype=float)).sum(skipna=True)),
            "sc_xwoba_contact": _mean(bbe, "estimated_woba_using_speedangle") if not bbe.empty else np.nan,
            "sc_avg_ev": _mean(bbe, "launch_speed") if not bbe.empty else np.nan,
            "sc_max_ev": _max(bbe, "launch_speed") if not bbe.empty else np.nan,
            "sc_avg_la": _mean(bbe, "launch_angle") if not bbe.empty else np.nan,
            "sc_avg_batted_ball_ev": _mean(bbe, "launch_speed") if not bbe.empty else np.nan,
            "sc_median_batted_ball_ev": _median(bbe, "launch_speed") if not bbe.empty else np.nan,
            "sc_p90_batted_ball_ev": _quantile(bbe, "launch_speed", 0.90) if not bbe.empty else np.nan,
            "sc_max_batted_ball_ev": _max(bbe, "launch_speed") if not bbe.empty else np.nan,
            "sc_avg_batted_ball_distance": _mean_any(bbe, ["hit_distance_sc", "hit_distance"]) if not bbe.empty else np.nan,
            "sc_median_batted_ball_distance": _median_any(bbe, ["hit_distance_sc", "hit_distance"]) if not bbe.empty else np.nan,
            "sc_p90_batted_ball_distance": _quantile_any(bbe, ["hit_distance_sc", "hit_distance"], 0.90) if not bbe.empty else np.nan,
            "sc_max_batted_ball_distance": _max_any(bbe, ["hit_distance_sc", "hit_distance"]) if not bbe.empty else np.nan,
            "sc_batted_ball_distance_count": _nonnull_count_any(bbe, ["hit_distance_sc", "hit_distance"]) if not bbe.empty else 0,
            "sc_hard_hit_rate": _rate(g["is_hard_hit"].sum(), bbe.shape[0]),
            "sc_barrel_rate": _rate(g["is_barrel"].sum(), bbe.shape[0]),
            "sc_sweetspot_rate": _rate(g["is_sweetspot"].sum(), bbe.shape[0]),
            "sc_whiff_rate": _rate(g["is_whiff"].sum(), g["is_swing"].sum()),
            "sc_csw_rate": _rate(g["is_csw"].sum(), g.shape[0]),
            "sc_k_rate": _rate(pa["is_k"].sum(), pa.shape[0]),
            "sc_bb_rate": _rate(pa["is_bb"].sum(), pa.shape[0]),
            "sc_hr_rate": _rate(pa["is_hr"].sum(), pa.shape[0]),
            "last_updated_utc": fetched_at,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_pitcher_game(df: pd.DataFrame, fetched_at: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for (game_pk, pitcher, team_id), g in df.groupby(["game_pk", "pitcher", "pitching_team_id"], dropna=False, sort=False):
        pa = g[g["is_pa"]]
        bbe = g[g["is_bbe"]]
        pitch_type = g.get("pitch_type", pd.Series(index=g.index, dtype="object"))
        row = {
            "game_pk": int(game_pk),
            "pitcher_id": int(pitcher) if pd.notna(pitcher) else None,
            "team_id": int(team_id) if pd.notna(team_id) else None,
            "opponent_team_id": int(g["pitching_opponent_team_id"].iloc[0]) if pd.notna(g["pitching_opponent_team_id"].iloc[0]) else None,
            "is_home": int(g["pitching_is_home"].iloc[0]) if pd.notna(g["pitching_is_home"].iloc[0]) else None,
            "pitcher_name": str(g.get("player_name", pd.Series([None])).iloc[0]) if "player_name" in g else None,
            "pitcher_hand": str(g["pitcher_hand"].dropna().iloc[0]) if g["pitcher_hand"].notna().any() else "UNK",
            "is_starter": int(pd.to_numeric(g.get("is_starter", pd.Series([0])), errors="coerce").fillna(0).max()),
            "official_date": g["official_date"].iloc[0],
            "game_datetime_utc": str(g["game_datetime_utc"].iloc[0]),
            "sc_pitches": int(g.shape[0]),
            "sc_pa": int(pa.shape[0]),
            "sc_bbe_allowed": int(bbe.shape[0]),
            "sc_release_speed_mean": _mean(g, "release_speed"),
            "sc_release_speed_max": _max(g, "release_speed"),
            "sc_release_spin_mean": _mean(g, "release_spin_rate"),
            "sc_release_extension_mean": _mean(g, "release_extension"),
            "sc_pitch_mix_entropy": _entropy(pitch_type),
            "sc_fastball_pct": _pitch_mix_pct(pitch_type, FASTBALL_TYPES),
            "sc_breaking_pct": _pitch_mix_pct(pitch_type, BREAKING_TYPES),
            "sc_offspeed_pct": _pitch_mix_pct(pitch_type, OFFSPEED_TYPES),
            "sc_zone_rate": _rate(g["is_zone"].sum(), g.shape[0]),
            "sc_whiff_rate": _rate(g["is_whiff"].sum(), g["is_swing"].sum()),
            "sc_csw_rate": _rate(g["is_csw"].sum(), g.shape[0]),
            "sc_called_strike_rate": _rate(g["is_called_strike"].sum(), g.shape[0]),
            "sc_xwoba_allowed_contact": _mean(bbe, "estimated_woba_using_speedangle") if not bbe.empty else np.nan,
            "sc_woba_allowed": _rate(pa.get("woba_value", pd.Series(dtype=float)).sum(skipna=True), pa.get("woba_denom", pd.Series(dtype=float)).sum(skipna=True)),
            "sc_avg_ev_allowed": _mean(bbe, "launch_speed") if not bbe.empty else np.nan,
            "sc_max_ev_allowed": _max(bbe, "launch_speed") if not bbe.empty else np.nan,
            "sc_avg_la_allowed": _mean(bbe, "launch_angle") if not bbe.empty else np.nan,
            "sc_avg_batted_ball_ev_allowed": _mean(bbe, "launch_speed") if not bbe.empty else np.nan,
            "sc_median_batted_ball_ev_allowed": _median(bbe, "launch_speed") if not bbe.empty else np.nan,
            "sc_p90_batted_ball_ev_allowed": _quantile(bbe, "launch_speed", 0.90) if not bbe.empty else np.nan,
            "sc_max_batted_ball_ev_allowed": _max(bbe, "launch_speed") if not bbe.empty else np.nan,
            "sc_avg_batted_ball_distance_allowed": _mean_any(bbe, ["hit_distance_sc", "hit_distance"]) if not bbe.empty else np.nan,
            "sc_median_batted_ball_distance_allowed": _median_any(bbe, ["hit_distance_sc", "hit_distance"]) if not bbe.empty else np.nan,
            "sc_p90_batted_ball_distance_allowed": _quantile_any(bbe, ["hit_distance_sc", "hit_distance"], 0.90) if not bbe.empty else np.nan,
            "sc_max_batted_ball_distance_allowed": _max_any(bbe, ["hit_distance_sc", "hit_distance"]) if not bbe.empty else np.nan,
            "sc_batted_ball_distance_allowed_count": _nonnull_count_any(bbe, ["hit_distance_sc", "hit_distance"]) if not bbe.empty else 0,
            "sc_hard_hit_rate_allowed": _rate(g["is_hard_hit"].sum(), bbe.shape[0]),
            "sc_barrel_rate_allowed": _rate(g["is_barrel"].sum(), bbe.shape[0]),
            "sc_sweetspot_rate_allowed": _rate(g["is_sweetspot"].sum(), bbe.shape[0]),
            "sc_k_rate": _rate(pa["is_k"].sum(), pa.shape[0]),
            "sc_bb_rate": _rate(pa["is_bb"].sum(), pa.shape[0]),
            "sc_hr_rate": _rate(pa["is_hr"].sum(), pa.shape[0]),
            "last_updated_utc": fetched_at,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def upsert_df(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    cols = list(df.columns)
    placeholders = ", ".join([":" + c for c in cols])
    col_sql = ", ".join(cols)
    update_cols = [c for c in cols if c not in {"game_pk", "team_id", "pitcher_id", "pitcher_hand"}]
    update_sql = ", ".join([f"{c}=excluded.{c}" for c in update_cols])
    if table == "mlb_statcast_team_game":
        conflict = "game_pk, team_id"
    elif table == "mlb_statcast_team_hand_game":
        conflict = "game_pk, team_id, pitcher_hand"
    elif table == "mlb_statcast_pitcher_game":
        conflict = "game_pk, pitcher_id, team_id"
    else:
        raise ValueError(f"Unexpected table: {table}")
    sql = f"""
        INSERT INTO {table} ({col_sql}) VALUES ({placeholders})
        ON CONFLICT({conflict}) DO UPDATE SET {update_sql}
    """
    records = df.replace({pd.NA: np.nan}).where(pd.notna(df), None).to_dict("records")
    conn.executemany(sql, records)
    return len(records)


def fetch_statcast_chunk(start: date, end: date, parallel: bool) -> pd.DataFrame:
    try:
        from pybaseball import statcast
    except ImportError as exc:
        raise SystemExit(
            "pybaseball is not installed. Add 'pybaseball>=2.2.7' to requirements.txt "
            "or install it in your notebook/Codespaces environment."
        ) from exc
    return statcast(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), parallel=parallel)


def main() -> None:
    args = parse_args()
    settings = get_settings()
    init_db(settings.odds_db_path)

    end = datetime.now(timezone.utc).date() + timedelta(days=args.days_forward)
    if args.end_date:
        end = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    if args.start_date:
        start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    else:
        start = end - timedelta(days=args.days_back)

    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with connect(settings.odds_db_path) as conn:
        init_statcast_tables(conn)
        games = load_games(conn)
        if games.empty:
            raise SystemExit("mlb_games is empty. Run scripts/02_fetch_mlb_games.py first.")
        starters = load_starters(conn)

        total_team = total_hand = total_pitcher = 0
        chunks = list(date_chunks(start, end, max(1, args.chunk_days)))
        if args.limit_chunks is not None:
            chunks = chunks[: args.limit_chunks]

        skipped_chunks = 0

        for i, (chunk_start, chunk_end) in enumerate(chunks, start=1):
            if args.skip_existing and chunk_has_existing_statcast(conn, chunk_start, chunk_end):
                skipped_chunks += 1
                print(f"Skipping Statcast chunk {i}/{len(chunks)} already loaded: {chunk_start} to {chunk_end}")
                continue

            print(f"Fetching Statcast chunk {i}/{len(chunks)}: {chunk_start} to {chunk_end}")
            raw = fetch_statcast_chunk(chunk_start, chunk_end, parallel=args.parallel)
            if raw is None or raw.empty:
                print("  No Statcast rows returned.")
                time.sleep(args.sleep)
                continue
            prepared = prepare_statcast(raw, games, starters)
            print(f"  Raw rows: {len(raw):,}; matched rows: {len(prepared):,}")
            if prepared.empty:
                time.sleep(args.sleep)
                continue
            team_df = aggregate_team_game(prepared, fetched_at)
            hand_df = aggregate_team_hand_game(prepared, fetched_at)
            pitcher_df = aggregate_pitcher_game(prepared, fetched_at)
            total_team += upsert_df(conn, "mlb_statcast_team_game", team_df)
            total_hand += upsert_df(conn, "mlb_statcast_team_hand_game", hand_df)
            total_pitcher += upsert_df(conn, "mlb_statcast_pitcher_game", pitcher_df)
            conn.commit()
            print(f"  Upserted team={len(team_df):,}, team_hand={len(hand_df):,}, pitcher={len(pitcher_df):,}")
            time.sleep(args.sleep)

        print({
            "db": str(settings.odds_db_path),
            "date_start": str(start),
            "date_end": str(end),
            "team_rows_upserted": total_team,
            "team_hand_rows_upserted": total_hand,
            "pitcher_rows_upserted": total_pitcher,
            "chunks_skipped_existing": skipped_chunks,
        })


if __name__ == "__main__":
    main()
