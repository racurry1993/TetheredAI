from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlb_betting.config import get_settings
from mlb_betting.db import connect, init_db, read_sql
from mlb_betting.logging_utils import configure_logging
from mlb_betting.mlb_stats_api import MlbStatsClient, fetch_game_feed_stats_to_db


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch MLB boxscore/player stats for completed games.")
    parser.add_argument("--days-back", type=int, default=730)
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of games to fetch. 0 means no limit.")
    parser.add_argument("--sleep", type=float, default=0.10, help="Seconds to sleep between MLB Stats API calls.")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch games already present in stat tables.")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    settings = get_settings()
    init_db(settings.odds_db_path)
    client = MlbStatsClient()

    with connect(settings.odds_db_path) as conn:
        query = """
            SELECT g.game_pk, g.official_date, g.home_team_name, g.away_team_name
            FROM mlb_games g
            LEFT JOIN (
                SELECT game_pk, COUNT(*) AS pitcher_rows
                FROM mlb_pitcher_game_stats
                GROUP BY game_pk
            ) p ON g.game_pk = p.game_pk
            WHERE g.target_home_win IS NOT NULL
              AND date(g.official_date) >= date('now', ?)
        """
        params = (f"-{int(args.days_back)} day",)
        if not args.refresh:
            query += " AND COALESCE(p.pitcher_rows, 0) = 0"
        query += " ORDER BY g.official_date, g.game_pk"
        games = read_sql(conn, query, params=params)

        if args.limit and args.limit > 0:
            games = games.head(args.limit)

        print({"candidate_completed_games": len(games), "refresh": args.refresh, "days_back": args.days_back})
        total_pitcher_rows = 0
        total_team_rows = 0
        failures = 0

        for i, row in games.iterrows():
            game_pk = int(row["game_pk"])
            try:
                result = fetch_game_feed_stats_to_db(conn, client, game_pk)
                total_pitcher_rows += result["pitcher_rows"]
                total_team_rows += result["team_rows"]
                if (i + 1) % 50 == 0 or i == 0:
                    print({
                        "processed": int(i + 1),
                        "game_pk": game_pk,
                        "pitcher_rows_total": total_pitcher_rows,
                        "team_rows_total": total_team_rows,
                    })
            except Exception as exc:  # keep going; one bad game should not kill the backfill
                failures += 1
                print({"game_pk": game_pk, "error": repr(exc)})
            if args.sleep > 0:
                time.sleep(args.sleep)

        print({
            "completed_game_feeds_processed": len(games),
            "pitcher_rows_inserted_or_updated": total_pitcher_rows,
            "team_rows_inserted_or_updated": total_team_rows,
            "failures": failures,
        })


if __name__ == "__main__":
    main()
