from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from mlb_betting.betting_math import expected_value_per_unit
from mlb_betting.config import get_settings
from mlb_betting.db import connect, init_db, insert_prediction_rows
from mlb_betting.feature_engineering import get_model_feature_columns
from mlb_betting.logging_utils import configure_logging
from mlb_betting.modeling import latest_model_path, load_model_bundle, utc_now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score upcoming MLB games using the latest model bundle.")
    parser.add_argument("--features", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD. Defaults to today UTC.")
    parser.add_argument("--days-forward", type=int, default=3)
    parser.add_argument("--min-edge", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    settings = get_settings()
    init_db(settings.odds_db_path)
    feature_path = Path(args.features) if args.features else settings.data_dir / "processed" / "mlb_game_features.parquet"
    model_path = Path(args.model) if args.model else latest_model_path(settings.model_dir)
    output_path = Path(args.output) if args.output else settings.data_dir / "predictions" / "mlb_moneyline_predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = load_model_bundle(model_path)
    estimator = bundle["estimator"]
    feature_cols = bundle["feature_cols"]
    run_id = bundle["run_id"]
    frame = pd.read_parquet(feature_path)
    frame["game_datetime_utc"] = pd.to_datetime(frame["game_datetime_utc"], utc=True, errors="coerce")
    today = pd.to_datetime(args.start_date or datetime.now(timezone.utc).date().isoformat(), utc=True)
    end = today + pd.Timedelta(days=args.days_forward + 1)
    upcoming = frame[(frame["game_datetime_utc"] >= today) & (frame["game_datetime_utc"] < end)].copy()
    if upcoming.empty:
        print("No upcoming games found in feature file. Refresh MLB schedule and rebuild features.")
        return
    missing_cols = [c for c in feature_cols if c not in upcoming.columns]
    if missing_cols:
        raise SystemExit(f"Feature file missing model columns: {missing_cols}")
    probs = estimator.predict_proba(upcoming[feature_cols])[:, 1]
    upcoming["model_home_win_prob"] = probs

    # Betting recommendation uses median market no-vig probability where available.
    upcoming["edge_home"] = upcoming["model_home_win_prob"] - upcoming.get("market_home_no_vig_prob", np.nan)
    upcoming["edge_away"] = (1.0 - upcoming["model_home_win_prob"]) - upcoming.get("market_away_no_vig_prob", np.nan)
    rec_side = []
    rec_price = []
    rec_edge = []
    rec_ev = []
    for _, row in upcoming.iterrows():
        home_edge = row.get("edge_home", np.nan)
        away_edge = row.get("edge_away", np.nan)
        if pd.notna(home_edge) and home_edge >= args.min_edge and home_edge >= away_edge:
            side = row.get("home_team_name")
            price = row.get("home_moneyline_median")
            edge = home_edge
            model_prob = row.get("model_home_win_prob")
        elif pd.notna(away_edge) and away_edge >= args.min_edge:
            side = row.get("away_team_name")
            price = row.get("away_moneyline_median")
            edge = away_edge
            model_prob = 1.0 - row.get("model_home_win_prob")
        else:
            side = None
            price = np.nan
            edge = np.nan
            model_prob = np.nan
        rec_side.append(side)
        rec_price.append(price)
        rec_edge.append(edge)
        rec_ev.append(expected_value_per_unit(model_prob, price) if pd.notna(model_prob) and pd.notna(price) else np.nan)
    upcoming["recommended_side"] = rec_side
    upcoming["recommended_price"] = rec_price
    upcoming["edge"] = rec_edge
    upcoming["expected_value_per_unit"] = rec_ev
    upcoming["run_id"] = run_id
    upcoming["scored_at_utc"] = utc_now_iso()

    keep = [
        "run_id", "scored_at_utc", "game_pk", "official_date", "game_datetime_utc",
        "home_team_name", "away_team_name", "model_home_win_prob", "market_home_no_vig_prob",
        "home_moneyline_median", "away_moneyline_median", "recommended_side", "recommended_price",
        "edge", "expected_value_per_unit",
    ]
    pred = upcoming[[c for c in keep if c in upcoming.columns]].copy()
    pred.to_csv(output_path, index=False)

    rows = []
    for idx, row in upcoming.iterrows():
        feature_snapshot = {c: (None if pd.isna(row.get(c)) else row.get(c)) for c in feature_cols}
        rows.append({
            "run_id": run_id,
            "scored_at_utc": row["scored_at_utc"],
            "game_pk": int(row["game_pk"]),
            "official_date": row.get("official_date"),
            "game_datetime_utc": str(row.get("game_datetime_utc")),
            "home_team_name": row.get("home_team_name"),
            "away_team_name": row.get("away_team_name"),
            "model_home_win_prob": float(row.get("model_home_win_prob")),
            "market_home_no_vig_prob": None if pd.isna(row.get("market_home_no_vig_prob")) else float(row.get("market_home_no_vig_prob")),
            "home_moneyline_median": None if pd.isna(row.get("home_moneyline_median")) else float(row.get("home_moneyline_median")),
            "away_moneyline_median": None if pd.isna(row.get("away_moneyline_median")) else float(row.get("away_moneyline_median")),
            "recommended_side": row.get("recommended_side"),
            "recommended_price": None if pd.isna(row.get("recommended_price")) else float(row.get("recommended_price")),
            "edge": None if pd.isna(row.get("edge")) else float(row.get("edge")),
            "expected_value_per_unit": None if pd.isna(row.get("expected_value_per_unit")) else float(row.get("expected_value_per_unit")),
            "feature_snapshot_json": json.dumps(feature_snapshot, default=str),
        })
    with connect(settings.odds_db_path) as conn:
        count = insert_prediction_rows(conn, rows)
        conn.commit()
    print({"predictions": len(pred), "inserted_db_rows": count, "output": str(output_path), "model": str(model_path)})


if __name__ == "__main__":
    main()
