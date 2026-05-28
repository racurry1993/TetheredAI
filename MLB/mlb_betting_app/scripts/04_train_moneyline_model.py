from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from mlb_betting.config import get_settings
from mlb_betting.db import connect, init_db
from mlb_betting.logging_utils import configure_logging
from mlb_betting.modeling import save_model_bundle, tune_moneyline_model
from mlb_betting.data_validation import validate_no_obvious_leakage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and tune MLB moneyline model.")
    parser.add_argument("--features", default=None, help="Feature parquet path")
    parser.add_argument("--holdout-days", type=int, default=45)
    parser.add_argument("--no-tune", dest="tune", action="store_false")
    parser.add_argument("--tune", dest="tune", action="store_true")
    parser.set_defaults(tune=True)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--min-rows", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    settings = get_settings()
    init_db(settings.odds_db_path)
    feature_path = Path(args.features) if args.features else settings.data_dir / "processed" / "mlb_game_features.parquet"
    if not feature_path.exists():
        raise SystemExit(f"Feature file not found: {feature_path}. Run scripts/03_build_features.py first.")
    frame = pd.read_parquet(feature_path)
    completed = frame[frame["target_home_win"].notna()].copy()
    if len(completed) < args.min_rows:
        raise SystemExit(f"Not enough completed games to train. Found {len(completed)}, need {args.min_rows}.")
    result = tune_moneyline_model(
        completed,
        holdout_days=args.holdout_days,
        tune=args.tune,
        calibrate=args.calibrate,
    )
    validate_no_obvious_leakage(result["feature_cols"])
    paths = save_model_bundle(result, settings.model_dir)
    metadata = json.loads(paths["metadata_path"].read_text(encoding="utf-8"))
    with connect(settings.odds_db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO model_runs (
                run_id, created_at_utc, model_name, target, train_start_date, train_end_date,
                test_start_date, test_end_date, n_train, n_test, metrics_json, params_json,
                feature_columns_json, artifact_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result["run_id"], result["created_at_utc"], result["model_name"], result["target"],
                result["train_start_date"], result["train_end_date"], result["test_start_date"], result["test_end_date"],
                result["n_train"], result["n_test"], json.dumps(result["final_metrics"]),
                json.dumps(result["search_results"], default=str), json.dumps(result["feature_cols"]), str(paths["model_path"]),
            ),
        )
        conn.commit()
    print({"run_id": result["run_id"], "model_name": result["model_name"], "metrics": result["final_metrics"], "paths": {k: str(v) for k, v in paths.items()}})


if __name__ == "__main__":
    main()
