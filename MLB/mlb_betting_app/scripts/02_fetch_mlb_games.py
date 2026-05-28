from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlb_betting.config import get_settings
from mlb_betting.db import connect, init_db
from mlb_betting.logging_utils import configure_logging
from mlb_betting.mlb_stats_api import MlbStatsClient, fetch_schedule_to_db


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch MLB schedule/results from MLB Stats API.")
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--days-back", type=int, default=365)
    parser.add_argument("--days-forward", type=int, default=14)
    parser.add_argument("--game-type", default=None, help="Optional MLB game type filter, e.g. R")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    settings = get_settings()
    init_db(settings.odds_db_path)
    today = datetime.now(timezone.utc).date()
    start_date = args.start_date or (today - timedelta(days=args.days_back)).isoformat()
    end_date = args.end_date or (today + timedelta(days=args.days_forward)).isoformat()
    client = MlbStatsClient()
    with connect(settings.odds_db_path) as conn:
        result = fetch_schedule_to_db(conn, client, start_date, end_date, game_type=args.game_type)
    print(result)


if __name__ == "__main__":
    main()
