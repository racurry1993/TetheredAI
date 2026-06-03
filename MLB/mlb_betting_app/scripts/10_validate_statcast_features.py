from __future__ import annotations

"""Lightweight validation for Statcast aggregate tables.

This script does NOT rebuild the full feature frame. It only validates the
SQLite tables produced by scripts/09_fetch_statcast.py. That keeps GitHub
Actions reruns cheap and avoids duplicating the expensive feature build.

The previous validator expected older column names such as game_date,
fetched_at_utc, and sc_batters_faced. The current Statcast fetcher writes
official_date, last_updated_utc, sc_pa, and sc_pitches_seen/sc_pitches, so this
validator uses the actual schema.
"""

import sqlite3
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlb_betting.config import get_settings
from mlb_betting.db import connect, init_db


REQUIRED_TABLES: dict[str, list[str]] = {
    "mlb_games": [
        "game_pk", "official_date", "game_datetime_utc", "home_team_id", "away_team_id",
    ],
    "mlb_statcast_team_game": [
        "game_pk", "team_id", "opponent_team_id", "is_home", "official_date", "game_datetime_utc",
        "sc_pa", "sc_pitches_seen", "sc_bbe", "sc_woba", "sc_xwoba_contact", "sc_avg_ev",
        "sc_avg_batted_ball_ev", "sc_p90_batted_ball_ev",
        "sc_avg_batted_ball_distance", "sc_p90_batted_ball_distance",
        "sc_hard_hit_rate", "sc_barrel_rate", "sc_whiff_rate", "sc_csw_rate", "sc_k_rate",
        "sc_bb_rate", "sc_hr_rate", "last_updated_utc",
    ],
    "mlb_statcast_team_hand_game": [
        "game_pk", "team_id", "opponent_team_id", "is_home", "pitcher_hand", "official_date",
        "game_datetime_utc", "sc_pa", "sc_pitches_seen", "sc_bbe", "sc_woba",
        "sc_xwoba_contact", "sc_avg_ev",
        "sc_avg_batted_ball_ev", "sc_p90_batted_ball_ev",
        "sc_avg_batted_ball_distance", "sc_p90_batted_ball_distance",
        "sc_hard_hit_rate", "sc_barrel_rate", "sc_whiff_rate",
        "sc_csw_rate", "sc_k_rate", "sc_bb_rate", "sc_hr_rate", "last_updated_utc",
    ],
    "mlb_statcast_pitcher_game": [
        "game_pk", "pitcher_id", "team_id", "opponent_team_id", "is_home", "pitcher_hand",
        "is_starter", "official_date", "game_datetime_utc", "sc_pitches", "sc_pa",
        "sc_bbe_allowed", "sc_release_speed_mean", "sc_release_spin_mean",
        "sc_release_extension_mean", "sc_pitch_mix_entropy", "sc_whiff_rate", "sc_csw_rate",
        "sc_xwoba_allowed_contact", "sc_woba_allowed",
        "sc_avg_batted_ball_ev_allowed", "sc_p90_batted_ball_ev_allowed",
        "sc_avg_batted_ball_distance_allowed", "sc_p90_batted_ball_distance_allowed",
        "sc_hard_hit_rate_allowed",
        "sc_barrel_rate_allowed", "sc_k_rate", "sc_bb_rate", "sc_hr_rate", "last_updated_utc",
    ],

    "mlb_statcast_team_pitch_type_game": [
        "game_pk", "team_id", "opponent_team_id", "is_home", "pitch_type_group",
        "official_date", "game_datetime_utc", "sc_pitch_type_pa",
        "sc_pitch_type_pitches_seen", "sc_pitch_type_bbe", "sc_pitch_type_woba",
        "sc_pitch_type_xwoba_contact", "sc_pitch_type_avg_ev",
        "sc_pitch_type_avg_batted_ball_distance", "sc_pitch_type_hard_hit_rate",
        "sc_pitch_type_whiff_rate", "sc_pitch_type_csw_rate", "last_updated_utc",
    ],
    "mlb_statcast_pitcher_pitch_type_game": [
        "game_pk", "pitcher_id", "team_id", "opponent_team_id", "is_home",
        "pitcher_hand", "is_starter", "pitch_type_group", "official_date",
        "game_datetime_utc", "sc_pitch_type_pitches", "sc_pitch_type_pct",
        "sc_pitch_type_pa", "sc_pitch_type_bbe_allowed",
        "sc_pitch_type_release_speed_mean", "sc_pitch_type_whiff_rate",
        "sc_pitch_type_csw_rate", "sc_pitch_type_woba_allowed",
        "sc_pitch_type_xwoba_allowed_contact", "sc_pitch_type_avg_ev_allowed",
        "sc_pitch_type_avg_batted_ball_distance_allowed", "last_updated_utc",
    ],
}

PRIMARY_KEY_CHECKS: dict[str, list[str]] = {
    "mlb_statcast_team_game": ["game_pk", "team_id"],
    "mlb_statcast_team_hand_game": ["game_pk", "team_id", "pitcher_hand"],
    "mlb_statcast_pitcher_game": ["game_pk", "pitcher_id", "team_id"],

    "mlb_statcast_team_pitch_type_game": ["game_pk", "team_id", "pitch_type_group"],
    "mlb_statcast_pitcher_pitch_type_game": ["game_pk", "pitcher_id", "team_id", "pitch_type_group"],
}


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def get_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f"PRAGMA table_info({qident(table)})").fetchall()]


def row_count(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {qident(table)}").fetchone()[0])


def date_coverage(conn: sqlite3.Connection, table: str) -> tuple[str | None, str | None, int]:
    cols = set(get_columns(conn, table))
    date_col = "official_date" if "official_date" in cols else None
    if date_col is None:
        return None, None, 0
    row = conn.execute(
        f"SELECT MIN({qident(date_col)}), MAX({qident(date_col)}), COUNT(DISTINCT {qident(date_col)}) FROM {qident(table)}"
    ).fetchone()
    return row[0], row[1], int(row[2] or 0)


def duplicate_key_count(conn: sqlite3.Connection, table: str, key_cols: Iterable[str]) -> int:
    cols = list(key_cols)
    existing = set(get_columns(conn, table))
    if not all(c in existing for c in cols):
        return -1
    group_cols = ", ".join(qident(c) for c in cols)
    sql = f"""
        SELECT COUNT(*)
        FROM (
            SELECT {group_cols}, COUNT(*) AS n
            FROM {qident(table)}
            GROUP BY {group_cols}
            HAVING COUNT(*) > 1
        )
    """
    return int(conn.execute(sql).fetchone()[0])


def sample_null_rate(conn: sqlite3.Connection, table: str, col: str) -> float | None:
    if col not in get_columns(conn, table):
        return None
    n = row_count(conn, table)
    if n == 0:
        return None
    nulls = int(conn.execute(f"SELECT COUNT(*) FROM {qident(table)} WHERE {qident(col)} IS NULL").fetchone()[0])
    return nulls / n


def main() -> None:
    settings = get_settings()
    init_db(settings.odds_db_path)

    print("Statcast validation: lightweight SQLite table/schema checks.")
    print("Database:", settings.odds_db_path)

    failures: list[str] = []

    with connect(settings.odds_db_path) as conn:
        for table, required_cols in REQUIRED_TABLES.items():
            print("\n" + "=" * 80)
            print(table)
            print("=" * 80)

            if not table_exists(conn, table):
                failures.append(f"Missing table: {table}")
                print("exists: False")
                continue

            cols = get_columns(conn, table)
            n = row_count(conn, table)
            min_date, max_date, distinct_dates = date_coverage(conn, table)
            print({
                "exists": True,
                "rows": n,
                "columns": len(cols),
                "min_official_date": min_date,
                "max_official_date": max_date,
                "distinct_dates": distinct_dates,
            })

            missing = [c for c in required_cols if c not in cols]
            if missing:
                failures.append(f"{table} missing required columns: {missing}")
                print({"missing_required_columns": missing})
            else:
                print("required columns: OK")

            if table != "mlb_games" and n == 0:
                failures.append(f"{table} has zero rows")

            if table in PRIMARY_KEY_CHECKS:
                dups = duplicate_key_count(conn, table, PRIMARY_KEY_CHECKS[table])
                print({"duplicate_primary_keys": dups})
                if dups > 0:
                    failures.append(f"{table} has duplicate key rows: {dups}")

            for col in [
                "sc_xwoba_contact", "sc_avg_ev", "sc_woba", "sc_release_speed_mean",
                "sc_avg_batted_ball_ev", "sc_avg_batted_ball_distance",
                "sc_avg_batted_ball_ev_allowed", "sc_avg_batted_ball_distance_allowed",
                "sc_pitch_type_woba", "sc_pitch_type_avg_ev",
                "sc_pitch_type_woba_allowed", "sc_pitch_type_pct",
            ]:
                rate = sample_null_rate(conn, table, col)
                if rate is not None:
                    print({f"{col}_null_rate": round(rate, 4)})

        print("\n" + "=" * 80)
        print("Statcast ↔ mlb_games key coverage")
        print("=" * 80)
        if table_exists(conn, "mlb_statcast_team_game") and table_exists(conn, "mlb_games"):
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS statcast_team_rows,
                    SUM(CASE WHEN g.game_pk IS NOT NULL THEN 1 ELSE 0 END) AS rows_matching_mlb_games,
                    COUNT(DISTINCT s.game_pk) AS statcast_games,
                    COUNT(DISTINCT g.game_pk) AS matched_games
                FROM mlb_statcast_team_game s
                LEFT JOIN mlb_games g ON s.game_pk = g.game_pk
                """
            ).fetchone()
            print({
                "statcast_team_rows": row[0],
                "rows_matching_mlb_games": row[1],
                "statcast_games": row[2],
                "matched_games": row[3],
            })
            if row[0] and row[1] == 0:
                failures.append("No Statcast team rows match mlb_games.game_pk")

    if failures:
        print("\nValidation failures:")
        for f in failures:
            print("-", f)
        raise SystemExit(1)

    print("\nStatcast validation completed successfully.")


if __name__ == "__main__":
    main()
