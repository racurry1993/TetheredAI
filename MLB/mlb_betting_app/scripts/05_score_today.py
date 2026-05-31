from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlb_betting.betting_math import expected_value_per_unit, kelly_fraction
from mlb_betting.config import get_settings
from mlb_betting.db import connect, init_db
from mlb_betting.modeling import load_model_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score upcoming MLB moneyline games with the approved champion model.")
    parser.add_argument("--features", default=None, help="Feature parquet path")
    parser.add_argument("--model-path", default="models/mlb_moneyline_champion.joblib", help="Champion model artifact path")
    parser.add_argument("--output", default=None, help="Predictions CSV path")
    parser.add_argument("--days-forward", type=int, default=3)
    parser.add_argument("--min-edge", type=float, default=0.02)
    parser.add_argument("--min-ev", type=float, default=0.0)
    parser.add_argument("--min-minutes-before-start", type=int, default=30)
    parser.add_argument("--bankroll-units", type=float, default=100.0)
    parser.add_argument("--kelly-fraction", type=float, default=0.25)
    return parser.parse_args()


def _resolve(root: Path, value: str | None, default: Path) -> Path:
    path = Path(value) if value else default
    return path if path.is_absolute() else root / path


def _choose_recommendation(row: pd.Series, min_edge: float, min_ev: float, kelly_frac: float) -> dict:
    candidates = []
    if pd.notna(row.get("home_moneyline_median")) and pd.notna(row.get("model_home_win_prob")):
        edge = row.get("edge_home")
        ev = row.get("home_ev_per_unit")
        if pd.notna(edge) and pd.notna(ev) and edge >= min_edge and ev >= min_ev:
            candidates.append({
                "recommended_side": row.get("home_team_name"),
                "recommended_team_type": "home",
                "recommended_price": row.get("home_moneyline_median"),
                "recommended_model_prob": row.get("model_home_win_prob"),
                "recommended_market_prob": row.get("market_home_no_vig_prob"),
                "edge": edge,
                "expected_value_per_unit": ev,
                "kelly_fraction": kelly_fraction(row.get("model_home_win_prob"), row.get("home_moneyline_median"), fraction=kelly_frac),
                "no_bet_reason": "",
            })
    if pd.notna(row.get("away_moneyline_median")) and pd.notna(row.get("model_away_win_prob")):
        edge = row.get("edge_away")
        ev = row.get("away_ev_per_unit")
        if pd.notna(edge) and pd.notna(ev) and edge >= min_edge and ev >= min_ev:
            candidates.append({
                "recommended_side": row.get("away_team_name"),
                "recommended_team_type": "away",
                "recommended_price": row.get("away_moneyline_median"),
                "recommended_model_prob": row.get("model_away_win_prob"),
                "recommended_market_prob": row.get("market_away_no_vig_prob"),
                "edge": edge,
                "expected_value_per_unit": ev,
                "kelly_fraction": kelly_fraction(row.get("model_away_win_prob"), row.get("away_moneyline_median"), fraction=kelly_frac),
                "no_bet_reason": "",
            })
    if candidates:
        return max(candidates, key=lambda x: (x["expected_value_per_unit"], x["edge"]))
    if pd.isna(row.get("home_moneyline_median")) or pd.isna(row.get("away_moneyline_median")):
        reason = "No market odds available yet"
    else:
        reason = "No side met edge/EV thresholds"
    return {
        "recommended_side": "",
        "recommended_team_type": "",
        "recommended_price": np.nan,
        "recommended_model_prob": np.nan,
        "recommended_market_prob": np.nan,
        "edge": np.nan,
        "expected_value_per_unit": np.nan,
        "kelly_fraction": 0.0,
        "no_bet_reason": reason,
    }


def main() -> None:
    args = parse_args()
    settings = get_settings()
    root = settings.project_root
    init_db(settings.odds_db_path)

    feature_path = _resolve(root, args.features, settings.data_dir / "processed" / "mlb_game_features.parquet")
    model_path = _resolve(root, args.model_path, settings.model_dir / "mlb_moneyline_champion.joblib")
    output_path = _resolve(root, args.output, settings.data_dir / "predictions" / "mlb_moneyline_predictions.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Champion model not found: {model_path}. Export it from the EDA notebook first.")

    frame = pd.read_parquet(feature_path)
    frame["game_datetime_utc"] = pd.to_datetime(frame["game_datetime_utc"], utc=True, errors="coerce")

    bundle = load_model_bundle(model_path)
    estimator = bundle["estimator"]
    feature_cols = list(bundle["feature_cols"])
    missing = [c for c in feature_cols if c not in frame.columns]
    if missing:
        raise ValueError(f"Champion model requires missing features: {missing[:30]}{'...' if len(missing) > 30 else ''}")

    scored_at = datetime.now(timezone.utc).replace(microsecond=0)
    start = scored_at + pd.Timedelta(minutes=args.min_minutes_before_start)
    end = scored_at + pd.Timedelta(days=args.days_forward)
    upcoming = frame[(frame["game_datetime_utc"] >= start) & (frame["game_datetime_utc"] < end)].copy()
    if upcoming.empty:
        # Do not silently fail. Write headers for the app but raise a clear summary.
        cols = [
            "run_id", "scored_at_utc", "game_pk", "official_date", "game_datetime_utc",
            "home_team_name", "away_team_name", "model_home_win_prob", "model_away_win_prob",
            "has_market_odds", "recommended_side", "edge", "expected_value_per_unit", "no_bet_reason",
        ]
        pd.DataFrame(columns=cols).to_csv(output_path, index=False)
        print({"predictions": 0, "output": str(output_path), "message": "No upcoming games in scoring window."})
        return

    X = upcoming[feature_cols]
    home_prob = estimator.predict_proba(X)[:, 1]
    upcoming["model_home_win_prob"] = np.clip(home_prob, 1e-6, 1 - 1e-6)
    upcoming["model_away_win_prob"] = 1.0 - upcoming["model_home_win_prob"]
    upcoming["has_market_odds"] = upcoming[["home_moneyline_median", "away_moneyline_median"]].notna().all(axis=1) if {"home_moneyline_median", "away_moneyline_median"}.issubset(upcoming.columns) else False

    if {"market_home_no_vig_prob", "market_away_no_vig_prob"}.issubset(upcoming.columns):
        upcoming["edge_home"] = upcoming["model_home_win_prob"] - upcoming["market_home_no_vig_prob"]
        upcoming["edge_away"] = upcoming["model_away_win_prob"] - upcoming["market_away_no_vig_prob"]
    else:
        upcoming["edge_home"] = np.nan
        upcoming["edge_away"] = np.nan
    if "home_moneyline_median" in upcoming.columns:
        upcoming["home_ev_per_unit"] = [expected_value_per_unit(p, price) if pd.notna(price) else np.nan for p, price in zip(upcoming["model_home_win_prob"], upcoming["home_moneyline_median"])]
    else:
        upcoming["home_ev_per_unit"] = np.nan
    if "away_moneyline_median" in upcoming.columns:
        upcoming["away_ev_per_unit"] = [expected_value_per_unit(p, price) if pd.notna(price) else np.nan for p, price in zip(upcoming["model_away_win_prob"], upcoming["away_moneyline_median"])]
    else:
        upcoming["away_ev_per_unit"] = np.nan

    recs = upcoming.apply(lambda r: _choose_recommendation(r, args.min_edge, args.min_ev, args.kelly_fraction), axis=1, result_type="expand")
    upcoming = pd.concat([upcoming.reset_index(drop=True), recs.reset_index(drop=True)], axis=1)
    upcoming["suggested_units"] = (upcoming["kelly_fraction"].fillna(0.0) * args.bankroll_units).clip(lower=0, upper=3.0)
    upcoming["run_id"] = f"score_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{bundle.get('run_id', 'champion')}"
    upcoming["scored_at_utc"] = scored_at.isoformat().replace("+00:00", "Z")
    upcoming["champion_model_run_id"] = bundle.get("run_id", "")
    upcoming["champion_model_family"] = bundle.get("model_family", "")
    upcoming["champion_feature_set"] = bundle.get("feature_set_name", "")

    preferred_cols = [
        "run_id", "scored_at_utc", "champion_model_run_id", "champion_model_family", "champion_feature_set",
        "game_pk", "official_date", "game_datetime_utc", "home_team_name", "away_team_name",
        "model_home_win_prob", "model_away_win_prob", "market_home_no_vig_prob", "market_away_no_vig_prob",
        "home_moneyline_median", "away_moneyline_median", "edge_home", "edge_away", "home_ev_per_unit", "away_ev_per_unit",
        "has_market_odds", "recommended_side", "recommended_team_type", "recommended_price", "recommended_model_prob", "recommended_market_prob",
        "edge", "expected_value_per_unit", "kelly_fraction", "suggested_units", "no_bet_reason",
    ]
    cols = [c for c in preferred_cols if c in upcoming.columns]
    upcoming[cols].sort_values(["game_datetime_utc", "game_pk"]).to_csv(output_path, index=False)

    with connect(settings.odds_db_path) as conn:
        try:
            upcoming[cols].to_sql("predictions", conn, if_exists="append", index=False)
        except Exception as exc:
            print(f"Warning: could not append predictions to database: {exc}")

    print({
        "predictions": int(len(upcoming)),
        "recommended_bets": int((upcoming["recommended_side"].astype(str) != "").sum()),
        "output": str(output_path),
        "model": str(model_path),
    })


if __name__ == "__main__":
    main()
