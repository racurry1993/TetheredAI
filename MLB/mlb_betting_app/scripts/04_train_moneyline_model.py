from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlb_betting.config import get_settings
from mlb_betting.modeling import build_feature_sets, compare_models, export_champion_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optional CLI trainer. Notebook-based manual promotion is preferred.")
    parser.add_argument("--features", default=None)
    parser.add_argument("--holdout-days", type=int, default=60)
    parser.add_argument("--max-search-iter", type=int, default=25)
    parser.add_argument("--model-name", default=None, help="Optional single model candidate, e.g. lightgbm or xgboost")
    parser.add_argument("--promote-champion", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    feature_path = Path(args.features) if args.features else settings.data_dir / "processed" / "mlb_game_features.parquet"
    if not feature_path.is_absolute():
        feature_path = settings.project_root / feature_path
    frame = pd.read_parquet(feature_path)
    feature_sets = build_feature_sets(frame, min_non_null_rate=0.05)
    model_names = [args.model_name] if args.model_name else None
    comparison = compare_models(
        frame,
        feature_sets=feature_sets,
        model_names=model_names,
        holdout_days=args.holdout_days,
        tune=True,
        calibrate=True,
        max_search_iter=args.max_search_iter,
    )
    results = comparison["results"].copy()
    out = settings.data_dir / "processed" / "moneyline_model_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    print("Best key:", comparison["best_key"])
    print(results.sort_values(["holdout_log_loss", "holdout_brier"]).head(15).to_string(index=False))
    if args.promote_champion:
        best_key = comparison["best_key"]
        best = comparison["fitted"][best_key]
        row = results[results["candidate_key"] == best_key].iloc[0].to_dict()
        paths = export_champion_model(
            estimator=best["estimator"],
            feature_cols=best["feature_cols"],
            metrics={k: v for k, v in row.items() if k.startswith("holdout_") or k in ["cv_log_loss", "calibration"]},
            model_family=best["model_name"],
            feature_set_name=best["feature_set"],
            model_dir=settings.model_dir,
            notes="Champion promoted by CLI trainer. Notebook review is preferred.",
        )
        print(paths)


if __name__ == "__main__":
    main()
