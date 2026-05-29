from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlb_betting.config import get_settings
from mlb_betting.db import connect, init_db, read_sql
from mlb_betting.logging_utils import configure_logging
from mlb_betting.mlb_stats_api import MlbStatsClient, fetch_schedule_to_db


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch MLB schedule/results from MLB Stats API.")
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--days-back", type=int, default=365)
    parser.add_argument("--days-forward", type=int, default=14)
    parser.add_argument(
        "--game-type",
        default="R",
        help="MLB game type filter. Use R for regular season. Pass empty string to omit.",
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=30,
        help="Fetch the schedule in chunks instead of one very large date range.",
    )
    return parser.parse_args()


def parse_yyyy_mm_dd(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def iter_date_chunks(start: date, end: date, chunk_days: int) -> Iterable[tuple[date, date]]:
    if chunk_days < 1:
        raise ValueError("chunk_days must be >= 1")

    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)


def summarize_games(conn: sqlite3.Connection, today: date) -> dict:
    summary = read_sql(
        conn,
        """
        SELECT
            COUNT(*) AS total_games,
            MIN(official_date) AS min_official_date,
            MAX(official_date) AS max_official_date,
            SUM(CASE WHEN target_home_win IS NOT NULL THEN 1 ELSE 0 END) AS completed_games,
            SUM(CASE WHEN official_date >= :today AND target_home_win IS NULL THEN 1 ELSE 0 END) AS future_or_unfinal_games
        FROM mlb_games
        """,
        {"today": today.isoformat()},
    ).iloc[0].to_dict()

    status_counts = read_sql(
        conn,
        """
        SELECT
            COALESCE(abstract_state, 'UNKNOWN') AS abstract_state,
            COALESCE(detailed_state, 'UNKNOWN') AS detailed_state,
            COUNT(*) AS games
        FROM mlb_games
        GROUP BY 1, 2
        ORDER BY games DESC
        """,
    )

    upcoming = read_sql(
        conn,
        """
        SELECT
            official_date,
            game_datetime_utc,
            away_team_name,
            home_team_name,
            detailed_state,
            abstract_state
        FROM mlb_games
        WHERE official_date >= :today
          AND target_home_win IS NULL
        ORDER BY official_date, game_datetime_utc
        LIMIT 20
        """,
        {"today": today.isoformat()},
    )

    print("\nMLB games summary")
    print(summary)
    print("\nStatus counts")
    print(status_counts.to_string(index=False))
    print("\nUpcoming/unfinal sample")
    if upcoming.empty:
        print("No upcoming/unfinal MLB games found in mlb_games.")
    else:
        print(upcoming.to_string(index=False))

    return summary


def main() -> None:
    configure_logging()
    args = parse_args()
    settings = get_settings()
    init_db(settings.odds_db_path)

    today = datetime.now(timezone.utc).date()
    start_date = parse_yyyy_mm_dd(args.start_date) if args.start_date else today - timedelta(days=args.days_back)
    end_date = parse_yyyy_mm_dd(args.end_date) if args.end_date else today + timedelta(days=args.days_forward)

    if end_date < start_date:
        raise SystemExit(f"end-date {end_date} is before start-date {start_date}")

    game_type = args.game_type.strip() if args.game_type is not None else None
    if game_type == "":
        game_type = None

    client = MlbStatsClient()
    total_inserted_or_updated = 0
    chunks = 0

    print({
        "message": "Fetching MLB schedule in chunks",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "game_type": game_type,
        "chunk_days": args.chunk_days,
        "db": str(settings.odds_db_path),
    })

    with connect(settings.odds_db_path) as conn:
        for chunk_start, chunk_end in iter_date_chunks(start_date, end_date, args.chunk_days):
            result = fetch_schedule_to_db(
                conn,
                client,
                chunk_start.isoformat(),
                chunk_end.isoformat(),
                game_type=game_type,
            )
            chunks += 1
            total_inserted_or_updated += int(result.get("games", 0) or 0)
            print({
                "chunk": chunks,
                "start_date": chunk_start.isoformat(),
                "end_date": chunk_end.isoformat(),
                "games_returned": result.get("games", 0),
            })

        summary = summarize_games(conn, today)

    print({
        "chunks": chunks,
        "games_returned_across_chunks": total_inserted_or_updated,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "future_or_unfinal_games_in_db": summary.get("future_or_unfinal_games"),
    })

    if int(summary.get("future_or_unfinal_games") or 0) == 0 and end_date >= today:
        raise SystemExit(
            "No future/unfinal MLB games were found after fetching the future date range. "
            "Check the schedule fetch log, game_type, and date window."
        )


if __name__ == "__main__":
    main()
