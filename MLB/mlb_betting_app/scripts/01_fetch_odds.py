from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlb_betting.config import get_settings
from mlb_betting.db import connect, init_db
from mlb_betting.logging_utils import configure_logging
from mlb_betting.odds_api import OddsApiClient, fetch_and_store_odds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch odds from The Odds API and store normalized snapshots.")
    parser.add_argument("--sport", default=None, help="Sport key, e.g. baseball_mlb")
    parser.add_argument("--regions", default=None, help="Comma-separated regions, e.g. us")
    parser.add_argument("--markets", default=None, help="Comma-separated markets, e.g. h2h,spreads,totals")
    parser.add_argument("--bookmakers", default=None, help="Optional comma-separated bookmaker keys. Overrides regions.")
    parser.add_argument("--odds-format", default=None, choices=["american", "decimal"])
    parser.add_argument("--commence-time-from", default=None)
    parser.add_argument("--commence-time-to", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    settings = get_settings()
    if not settings.odds_api_key:
        raise SystemExit("ODDS_API_KEY is missing. Put it in .env or your environment.")
    init_db(settings.odds_db_path)
    client = OddsApiClient(settings.odds_api_key)
    with connect(settings.odds_db_path) as conn:
        result = fetch_and_store_odds(
            conn=conn,
            client=client,
            sport=args.sport or settings.odds_sport_key,
            regions=args.regions or settings.odds_regions,
            markets=args.markets or settings.odds_markets,
            odds_format=args.odds_format or settings.odds_format,
            bookmakers=args.bookmakers,
            commence_time_from=args.commence_time_from,
            commence_time_to=args.commence_time_to,
            dry_run=args.dry_run,
        )
    print(result)


if __name__ == "__main__":
    main()
