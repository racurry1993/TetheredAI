from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlb_betting.config import get_settings
from mlb_betting.db import connect
from mlb_betting.feature_engineering import (
    build_game_feature_frame,
    get_model_feature_columns,
    load_latest_odds_consensus,
    load_mlb_games,
    load_mlb_pitcher_game_stats,
    load_mlb_team_game_stats,
)

REQUIRED_PREFIXES = {
    "elo": ["home_elo_pre", "away_elo_pre", "diff_elo_pre", "elo_home_win_prob"],
    "starter": ["home_starter_", "away_starter_", "diff_starter_"],
    "bullpen": ["home_bullpen_", "away_bullpen_", "diff_bullpen_"],
    "team_boxscore": ["home_team_box_", "away_team_box_", "diff_team_box_"],
    "team_vs_hand": ["home_team_vs_hand_", "away_team_vs_hand_", "diff_team_vs_hand_"],
}

LEAKAGE_COLUMNS = {
    "home_score", "away_score", "target_home_win", "home_margin", "total_runs",
    "status_code", "detailed_state", "abstract_state",
}


def cols_with_prefix(cols: list[str], prefix: str) -> list[str]:
    return [c for c in cols if c.startswith(prefix)]


def main() -> None:
    settings = get_settings()
    db_path = settings.odds_db_path
    print({"db_path": str(db_path), "exists": db_path.exists()})
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    with connect(db_path) as conn:
        games = load_mlb_games(conn)
        pitcher_stats = load_mlb_pitcher_game_stats(conn)
        team_stats = load_mlb_team_game_stats(conn)
        odds = load_latest_odds_consensus(conn)

    print({
        "mlb_games_rows": len(games),
        "pitcher_game_stat_rows": len(pitcher_stats),
        "team_game_stat_rows": len(team_stats),
        "odds_consensus_rows": len(odds),
    })

    if games.empty:
        raise SystemExit("No games found. Run scripts/02_fetch_mlb_games.py first.")

    features = build_game_feature_frame(
        games=games,
        odds_consensus=odds,
        include_future=True,
        pitcher_stats=pitcher_stats,
        team_game_stats=team_stats,
    )

    print({
        "feature_rows": len(features),
        "feature_cols": len(features.columns),
        "completed_rows": int(features["target_home_win"].notna().sum()) if "target_home_win" in features else None,
        "future_rows": int(features["target_home_win"].isna().sum()) if "target_home_win" in features else None,
        "min_game_datetime": str(pd.to_datetime(features["game_datetime_utc"], utc=True, errors="coerce").min()) if "game_datetime_utc" in features else None,
        "max_game_datetime": str(pd.to_datetime(features["game_datetime_utc"], utc=True, errors="coerce").max()) if "game_datetime_utc" in features else None,
    })

    model_cols = get_model_feature_columns(features, include_market=False, min_non_null_rate=0.05)
    print({"model_feature_cols": len(model_cols)})

    leaks = sorted(set(model_cols) & LEAKAGE_COLUMNS)
    if leaks:
        raise SystemExit(f"Leakage columns found in model feature list: {leaks}")

    if "venue_id" in model_cols:
        raise SystemExit("venue_id is present in model feature list. Use explicit park factors instead of raw venue ID.")

    for group_name, prefixes_or_cols in REQUIRED_PREFIXES.items():
        found = []
        for item in prefixes_or_cols:
            if item.endswith("_"):
                found.extend(cols_with_prefix(list(features.columns), item))
            elif item in features.columns:
                found.append(item)
        non_null_rates = features[found].notna().mean().sort_values(ascending=False).head(10) if found else pd.Series(dtype=float)
        print({
            "group": group_name,
            "columns_found": len(found),
            "top_non_null_rates": non_null_rates.to_dict(),
        })

    output_path = settings.data_dir / "processed" / "mlb_game_features_validation.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    print({"validation_feature_file": str(output_path)})
    print("Feature engineering validation completed successfully.")


if __name__ == "__main__":
    main()
