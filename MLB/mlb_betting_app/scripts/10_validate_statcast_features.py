from __future__ import annotations

"""Validate Statcast feature tables and feature attachment without API calls."""

from pathlib import Path
import sys

import pandas as pd

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
from mlb_betting.statcast_features import add_statcast_features, get_statcast_feature_columns, table_exists


def table_count(conn, table: str) -> int:
    if not table_exists(conn, table):
        return 0
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def main() -> None:
    settings = get_settings()
    init_db(settings.odds_db_path)
    output = settings.data_dir / "processed" / "mlb_game_features_statcast_validation.parquet"

    with connect(settings.odds_db_path) as conn:
        counts = {
            "mlb_games": table_count(conn, "mlb_games"),
            "mlb_pitcher_game_stats": table_count(conn, "mlb_pitcher_game_stats"),
            "mlb_team_game_stats": table_count(conn, "mlb_team_game_stats"),
            "mlb_statcast_team_game": table_count(conn, "mlb_statcast_team_game"),
            "mlb_statcast_team_hand_game": table_count(conn, "mlb_statcast_team_hand_game"),
            "mlb_statcast_pitcher_game": table_count(conn, "mlb_statcast_pitcher_game"),
        }
        print("Input table counts:")
        print(counts)

        games = load_mlb_games(conn)
        odds = load_latest_odds_consensus(conn)
        pitcher_stats = load_mlb_pitcher_game_stats(conn)
        team_game_stats = load_mlb_team_game_stats(conn)
        features = build_game_feature_frame(
            games,
            odds_consensus=odds,
            include_future=True,
            pitcher_stats=pitcher_stats,
            team_game_stats=team_game_stats,
        )
        before_cols = len(features.columns)
        features = add_statcast_features(features, conn)
        statcast_cols = get_statcast_feature_columns(features)
        after_cols = len(features.columns)

    save_features(features, output)

    print({
        "rows": len(features),
        "columns_before_statcast": before_cols,
        "columns_after_statcast": after_cols,
        "statcast_feature_columns": len(statcast_cols),
        "output": str(output),
    })

    groups = {
        "team_offense": [c for c in statcast_cols if "team_off" in c],
        "team_vs_hand": [c for c in statcast_cols if "team_vs_hand" in c],
        "starter": [c for c in statcast_cols if "starter_statcast" in c],
        "bullpen": [c for c in statcast_cols if "bullpen" in c],
    }
    for name, cols in groups.items():
        print(f"{name}: {len(cols)} columns")
        print(cols[:20])

    if statcast_cols:
        missing = features[statcast_cols].isna().mean().sort_values(ascending=False).head(30)
        print("Top Statcast missingness:")
        print(missing.to_string())

    # Leakage sanity: no postgame fields in default statcast cols.
    banned_tokens = ["post_", "delta_", "home_score", "away_score", "target_home_win", "total_runs", "home_margin"]
    bad = [c for c in statcast_cols if any(tok in c for tok in banned_tokens)]
    if bad:
        raise SystemExit(f"Potential leakage columns found in Statcast feature list: {bad}")

    print("Validation completed.")


if __name__ == "__main__":
    main()
