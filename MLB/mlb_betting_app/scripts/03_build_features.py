from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlb_betting.config import get_settings
from mlb_betting.db import connect, init_db
from mlb_betting.feature_engineering import (
    build_game_feature_frame,
    load_latest_odds_consensus,
    load_mlb_games,
    load_mlb_pitcher_game_stats,
    load_mlb_team_game_stats,
    save_features,
)
from mlb_betting.logging_utils import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MLB game-level model features.")
    parser.add_argument("--output", default=None, help="Output parquet path")
    parser.add_argument("--completed-only", action="store_true", help="Exclude future/scheduled games")
    parser.add_argument(
        "--include-statcast",
        action="store_true",
        default=True,
        help="Add Statcast-derived features when Statcast tables exist. Default: true.",
    )
    parser.add_argument(
        "--no-statcast",
        dest="include_statcast",
        action="store_false",
        help="Disable Statcast feature attachment.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    settings = get_settings()
    init_db(settings.odds_db_path)
    output = Path(args.output) if args.output else settings.data_dir / "processed" / "mlb_game_features.parquet"

    with connect(settings.odds_db_path) as conn:
        games = load_mlb_games(conn)
        odds = load_latest_odds_consensus(conn)
        pitcher_stats = load_mlb_pitcher_game_stats(conn)
        team_game_stats = load_mlb_team_game_stats(conn)

        features = build_game_feature_frame(
            games,
            odds_consensus=odds,
            include_future=not args.completed_only,
            pitcher_stats=pitcher_stats,
            team_game_stats=team_game_stats,
        )

        statcast_cols_before = 0
        statcast_cols_after = 0
        if args.include_statcast:
            try:
                from mlb_betting.statcast_features import add_statcast_features, get_statcast_feature_columns

                features = add_statcast_features(features, conn)
                statcast_cols_after = len(get_statcast_feature_columns(features))
            except Exception as exc:
                raise RuntimeError(
                    "Statcast feature attachment failed. Re-run with --no-statcast to isolate the base feature build, "
                    "or run scripts/10_validate_statcast_features.py for diagnostics."
                ) from exc

    save_features(features, output)
    print({
        "rows": len(features),
        "columns": len(features.columns),
        "games_rows": len(games),
        "pitcher_game_stat_rows": len(pitcher_stats),
        "team_game_stat_rows": len(team_game_stats),
        "statcast_feature_columns": statcast_cols_after,
        "output": str(output),
    })


if __name__ == "__main__":
    main()
