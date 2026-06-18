from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlb_betting.config import get_settings
from mlb_betting.notebook_feature_builder import (
    align_frame_to_expected_features,
    build_champion_feature_frame,
    load_expected_feature_cols,
    normalize_games_frame,
    schema_diagnostics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build notebook-compatible MLB features for the current moneyline champion model."
    )
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output", default=None)
    parser.add_argument("--db", default=None, help="Optional odds.db path. Used to merge fresh mlb_games rows from the daily fetch.")
    parser.add_argument("--merge-db-games", action="store_true", default=True)
    parser.add_argument("--no-merge-db-games", dest="merge_db_games", action="store_false")
    parser.add_argument("--expected-features", default="models/mlb_moneyline_champion_features.json")
    parser.add_argument("--fallback-model", default="models/mlb_moneyline_champion.joblib")
    parser.add_argument("--allow-missing-model-cols", action="store_true")
    parser.add_argument("--fill-missing-model-cols", action="store_true", default=True)
    parser.add_argument("--no-eda", dest="add_eda", action="store_false", help="Disable notebook EDA features. Do not use for current champion.")
    parser.set_defaults(add_eda=True)
    return parser.parse_args()


def read_parquet_if_exists(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        print({"missing_raw_table": label, "path": str(path)})
        return pd.DataFrame()
    df = pd.read_parquet(path)
    print({"raw_table": label, "path": str(path), "rows": int(len(df)), "columns": int(len(df.columns))})
    return df


def read_db_games(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    with sqlite3.connect(db_path) as conn:
        try:
            df = pd.read_sql_query("SELECT * FROM mlb_games", conn)
        except Exception as exc:
            print({"db_games_warning": repr(exc)})
            return pd.DataFrame()
    print({"db_games_rows": int(len(df)), "db_games_columns": int(len(df.columns)), "db": str(db_path)})
    return df


def combine_games(raw_games: pd.DataFrame, db_games: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    if raw_games is not None and not raw_games.empty:
        pieces.append(normalize_games_frame(raw_games))
    if db_games is not None and not db_games.empty:
        pieces.append(normalize_games_frame(db_games))
    if not pieces:
        return pd.DataFrame()
    out = pd.concat(pieces, ignore_index=True, sort=False)
    if "game_pk" in out.columns:
        out = out.sort_values([c for c in ["game_datetime_utc", "game_pk"] if c in out.columns])
        out = out.drop_duplicates("game_pk", keep="last")
    return out.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    settings = get_settings()
    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_absolute():
        raw_dir = settings.project_root / raw_dir
    output_path = Path(args.output) if args.output else settings.data_dir / "processed" / "mlb_game_features.parquet"
    if not output_path.is_absolute():
        output_path = settings.project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    expected_path = Path(args.expected_features)
    if not expected_path.is_absolute():
        expected_path = settings.project_root / expected_path
    expected_cols = load_expected_feature_cols(expected_path)
    if not expected_cols:
        fallback = Path(args.fallback_model)
        if not fallback.is_absolute():
            fallback = settings.project_root / fallback
        expected_cols = load_expected_feature_cols(fallback)
    print({"expected_feature_count": len(expected_cols), "expected_features_source": str(expected_path)})

    raw_games = read_parquet_if_exists(raw_dir / "mlb_games.parquet", "mlb_games")
    box_team = read_parquet_if_exists(raw_dir / "mlb_box_team_game.parquet", "mlb_box_team_game")
    box_pitcher = read_parquet_if_exists(raw_dir / "mlb_box_pitcher_game.parquet", "mlb_box_pitcher_game")
    odds = read_parquet_if_exists(raw_dir / "mlb_odds_snapshots.parquet", "mlb_odds_snapshots")

    statcast_path = raw_dir / "mlb_statcast_raw.parquet"
    statcast_raw = read_parquet_if_exists(statcast_path, "mlb_statcast_raw")
    if statcast_raw.empty and (raw_dir / "mlb_statcast_raw_sample_1000.parquet").exists():
        print({
            "statcast_raw_note": "mlb_statcast_raw_sample_1000.parquet exists but is ignored for production feature building. Upload full mlb_statcast_raw.parquet for true notebook-schema Statcast features."
        })

    db_games = pd.DataFrame()
    db_path = Path(args.db) if args.db else settings.odds_db_path
    if not db_path.is_absolute():
        db_path = settings.project_root / db_path
    if args.merge_db_games:
        db_games = read_db_games(db_path)

    games = combine_games(raw_games, db_games)
    if games.empty:
        raise SystemExit("No games available. Provide data/raw/mlb_games.parquet or data/odds.db with mlb_games.")

    print({
        "combined_games_rows": int(len(games)),
        "combined_games_min_date": str(games["official_date"].min()) if "official_date" in games else None,
        "combined_games_max_date": str(games["official_date"].max()) if "official_date" in games else None,
        "combined_completed_rows": int(games.get("target_home_win", pd.Series(index=games.index)).notna().sum()),
    })

    if box_team.empty:
        raise SystemExit("Missing data/raw/mlb_box_team_game.parquet. This is required for the box_box_box_* champion features.")
    if box_pitcher.empty:
        raise SystemExit("Missing data/raw/mlb_box_pitcher_game.parquet. This is required for starter_box_* champion features.")

    frame = build_champion_feature_frame(
        games_df=games,
        box_team_df=box_team,
        statcast_raw_df=statcast_raw,
        odds_df=odds,
        box_pitcher_df=box_pitcher,
        add_eda=args.add_eda,
    )
    print({"built_rows": int(len(frame)), "built_columns_before_alignment": int(len(frame.columns))})

    if expected_cols:
        frame, missing = align_frame_to_expected_features(frame, expected_cols, fill_missing=args.fill_missing_model_cols)
        diag = schema_diagnostics(frame, expected_cols)
        print({"schema_diagnostics": diag})
        report_path = output_path.parent / "mlb_game_features_schema_report.json"
        report_path.write_text(json.dumps(diag, indent=2, default=str), encoding="utf-8")
        if missing and not args.allow_missing_model_cols:
            raise SystemExit(
                "Notebook feature build is still missing model columns. "
                f"Missing={len(missing)}. First 50={missing[:50]}. "
                "Do not upload this feature parquet until the missing raw sources are available."
            )

    frame.to_parquet(output_path, index=False)
    print({"output": str(output_path), "rows": int(len(frame)), "columns": int(len(frame.columns))})


if __name__ == "__main__":
    main()
