from __future__ import annotations

import argparse
import json
import joblib
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from mlb_betting.betting_math import expected_value_per_unit
from mlb_betting.config import get_settings
from mlb_betting.db import connect, init_db, insert_prediction_rows
from mlb_betting.logging_utils import configure_logging


BAD_DETAIL_STATE_PATTERN = "final|postponed|completed|cancelled|canceled|suspended|game over"
DEFAULT_ALLOWED_ABSTRACT_STATES = {"preview"}
LIVE_ALLOWED_ABSTRACT_STATES = {"preview", "live"}


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 Z format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def latest_model_path(model_dir: Path) -> Path:
    """Find the preferred/latest model bundle without depending on mlb_betting.modeling.

    Priority:
      1. models/mlb_moneyline_champion.joblib
      2. newest *.joblib in model_dir by modified time
    """
    model_dir = Path(model_dir)
    champion = model_dir / "mlb_moneyline_champion.joblib"
    if champion.exists():
        return champion

    candidates = sorted(
        model_dir.glob("*.joblib"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No .joblib model files found in {model_dir}")
    return candidates[0]


def load_model_bundle(model_path: Path) -> dict[str, Any]:
    """Load a joblib model bundle safely.

    The notebook exports a dict with keys like model, feature_cols, target_col,
    probability_shrink, and shrink_center. If a raw estimator is ever provided,
    wrap it into a minimal bundle so the rest of the scorer still works.
    """
    model_path = Path(model_path)
    obj = joblib.load(model_path)
    if isinstance(obj, dict):
        return obj
    return {
        "model_name": model_path.stem,
        "model": obj,
        "feature_cols": getattr(obj, "feature_names_in_", None),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score truly upcoming MLB games using the latest model bundle. "
            "This script intentionally does not treat every null target row as upcoming; "
            "historical postponed/unresolved rows and far-future schedule rows are filtered out."
        )
    )
    parser.add_argument(
        "--features",
        default=None,
        help="Path to feature parquet. Defaults to data/processed/mlb_game_features.parquet.",
    )
    parser.add_argument(
        "--model",
        "--model-path",
        dest="model",
        default=None,
        help="Path to model bundle. Defaults to latest model in models/.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output CSV. Defaults to data/predictions/mlb_moneyline_predictions.csv.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help=(
            "YYYY-MM-DD. Optional debug/backtest date. If omitted, scoring starts at now UTC "
            "+ --min-minutes-before-start."
        ),
    )
    parser.add_argument(
        "--days-forward",
        type=int,
        default=3,
        help="Number of days forward to score from start timestamp.",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.02,
        help="Minimum no-vig probability edge required for a bet recommendation.",
    )
    parser.add_argument(
        "--min-ev",
        type=float,
        default=0.00,
        help="Minimum expected value per 1 unit staked required for a recommendation.",
    )
    parser.add_argument(
        "--min-minutes-before-start",
        type=int,
        default=30,
        help="When --start-date is omitted, exclude games starting within this many minutes.",
    )
    parser.add_argument(
        "--include-live",
        action="store_true",
        help="Include rows with abstract_state=Live. Default is scheduled/preview games only.",
    )
    parser.add_argument(
        "--only-bettable",
        action="store_true",
        help="If set, output only rows with matched market odds. By default, outputs model-only rows too.",
    )
    parser.add_argument(
        "--allow-scored-targets",
        action="store_true",
        help=(
            "Debug/backtest only: allow rows with a non-null target in the scoring window. "
            "Production scoring should leave this off."
        ),
    )
    parser.add_argument(
        "--write-debug-candidates",
        action="store_true",
        help="Write unresolved/historical/far-future candidate diagnostics next to the prediction CSV.",
    )
    parser.add_argument(
        "--probability-shrink",
        type=float,
        default=None,
        help=(
            "Optional override for probability shrinkage. If omitted, reads "
            "probability_shrink from the model bundle or adjacent metadata JSON. "
            "Use 1.0 for no shrinkage."
        ),
    )
    parser.add_argument(
        "--shrink-center",
        type=float,
        default=None,
        help=(
            "Optional override for shrink center. If omitted, reads shrink_center "
            "from the model bundle or adjacent metadata JSON. Default is 0.5."
        ),
    )
    return parser.parse_args()


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _safe_ev(model_prob: float | None, price: float | None) -> float:
    model_prob_f = _safe_float(model_prob)
    price_f = _safe_float(price)
    if model_prob_f is None or price_f is None:
        return np.nan
    try:
        return float(expected_value_per_unit(model_prob_f, price_f))
    except Exception:
        return np.nan


def _load_adjacent_metadata(model_path: Path) -> dict[str, Any]:
    """Load optional JSON metadata next to a model bundle.

    Expected default name for the champion artifact is:
      mlb_moneyline_champion.joblib
      mlb_moneyline_champion_metadata.json

    The scorer still works if this file is absent; shrinkage can also be stored
    directly in the joblib bundle.
    """
    candidates = [
        model_path.with_name(f"{model_path.stem}_metadata.json"),
        model_path.with_name("mlb_moneyline_champion_metadata.json"),
    ]
    for meta_path in candidates:
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except Exception as exc:
                print({"metadata_warning": f"Failed to load {meta_path}: {exc}"})
    return {}


def _resolve_shrink_settings(
    args: argparse.Namespace,
    bundle: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[float, float]:
    """Resolve probability shrinkage settings.

    Priority:
      1. CLI overrides
      2. joblib bundle keys
      3. adjacent metadata JSON keys
      4. no shrinkage, centered at 0.5
    """
    shrink = args.probability_shrink
    center = args.shrink_center

    if shrink is None:
        shrink = bundle.get("probability_shrink", metadata.get("probability_shrink", 1.0))
    if center is None:
        center = bundle.get("shrink_center", metadata.get("shrink_center", 0.5))

    try:
        shrink = float(shrink)
    except Exception:
        shrink = 1.0

    try:
        center = float(center)
    except Exception:
        center = 0.5

    if not np.isfinite(shrink):
        shrink = 1.0
    if not np.isfinite(center):
        center = 0.5

    # Keep values sane; shrink > 1 intentionally makes probabilities more extreme,
    # so cap it unless intentionally changed in this code later.
    shrink = max(0.0, min(shrink, 1.0))
    center = max(0.01, min(center, 0.99))
    return shrink, center


def apply_probability_shrink(
    probabilities: np.ndarray | pd.Series | list[float],
    *,
    shrink: float = 1.0,
    center: float = 0.5,
    clip_low: float = 0.01,
    clip_high: float = 0.99,
) -> np.ndarray:
    """Pull probabilities toward a center value to reduce overconfidence.

    Example: center=0.5, shrink=0.80
      raw 0.60 -> 0.5 + 0.80 * (0.60 - 0.5) = 0.58
      raw 0.40 -> 0.5 + 0.80 * (0.40 - 0.5) = 0.42
    """
    p = np.asarray(probabilities, dtype=float)
    p_adj = center + shrink * (p - center)
    return np.clip(p_adj, clip_low, clip_high)


def _normalize_status_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


def _build_scoring_window(args: argparse.Namespace) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    scored_at_ts = pd.Timestamp.now(tz="UTC")
    if args.start_date:
        start_ts = pd.to_datetime(args.start_date, utc=True)
    else:
        start_ts = scored_at_ts + pd.Timedelta(minutes=args.min_minutes_before_start)
    end_ts = start_ts + pd.Timedelta(days=args.days_forward)
    return scored_at_ts, start_ts, end_ts


def _ensure_datetime_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "game_datetime_utc" in out.columns:
        out["game_datetime_utc"] = pd.to_datetime(out["game_datetime_utc"], utc=True, errors="coerce")
    else:
        out["game_datetime_utc"] = pd.NaT

    if "official_date" in out.columns:
        out["official_date_dt"] = pd.to_datetime(out["official_date"], errors="coerce").dt.date
    else:
        out["official_date_dt"] = out["game_datetime_utc"].dt.date

    return out


def build_scoring_candidates(
    frame: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    *,
    include_live: bool = False,
    allow_scored_targets: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return rows that are truly candidates for pregame scoring.

    Do not equate target_home_win.isna() with upcoming. Null targets can include
    postponed/cancelled historical rows or far-future stale schedule rows. This
    function filters by game time, target status, and MLB game status fields.
    """
    df = _ensure_datetime_columns(frame)

    if "target_home_win" in df.columns and not allow_scored_targets:
        unresolved_mask = df["target_home_win"].isna()
    else:
        unresolved_mask = pd.Series(True, index=df.index)

    time_mask = (df["game_datetime_utc"] >= start_ts) & (df["game_datetime_utc"] < end_ts)

    status_mask = pd.Series(True, index=df.index)
    allowed_abstract = LIVE_ALLOWED_ABSTRACT_STATES if include_live else DEFAULT_ALLOWED_ABSTRACT_STATES

    if "abstract_state" in df.columns:
        abstract = _normalize_status_text(df["abstract_state"])
        # If status is present, require it to be a pregame/live state. Missing statuses are allowed
        # because some feature files may not carry status fields all the way through.
        status_mask &= abstract.isna() | abstract.isin(allowed_abstract)

    if "detailed_state" in df.columns:
        detailed = _normalize_status_text(df["detailed_state"])
        status_mask &= ~detailed.str.contains(BAD_DETAIL_STATE_PATTERN, na=False)

    candidates = df[unresolved_mask & time_mask & status_mask].copy()

    unresolved_all = df[unresolved_mask].copy()
    historical_unresolved = unresolved_all[unresolved_all["game_datetime_utc"] < start_ts].copy()
    far_future_unresolved = unresolved_all[unresolved_all["game_datetime_utc"] >= end_ts].copy()
    in_window_unresolved = unresolved_all[time_mask].copy()
    status_filtered_unresolved = in_window_unresolved[~status_mask.loc[in_window_unresolved.index]].copy()

    diagnostics = {
        "total_rows": int(len(df)),
        "unresolved_or_allowed_rows": int(unresolved_mask.sum()),
        "in_time_window_rows": int(time_mask.sum()),
        "status_allowed_rows": int(status_mask.sum()),
        "scoring_candidates": int(len(candidates)),
        "historical_unresolved_rows": int(len(historical_unresolved)),
        "far_future_unresolved_rows": int(len(far_future_unresolved)),
        "in_window_status_filtered_rows": int(len(status_filtered_unresolved)),
        "feature_min_game_datetime_utc": str(df["game_datetime_utc"].min()),
        "feature_max_game_datetime_utc": str(df["game_datetime_utc"].max()),
        "score_start_utc": str(start_ts),
        "score_end_utc": str(end_ts),
    }

    return candidates, diagnostics


def _choose_recommendation(row: pd.Series, min_edge: float, min_ev: float) -> dict[str, object]:
    """Choose the best moneyline side, or return a no-bet reason."""
    has_home_market = pd.notna(row.get("market_home_no_vig_prob")) and pd.notna(row.get("home_moneyline_median"))
    has_away_market = pd.notna(row.get("market_away_no_vig_prob")) and pd.notna(row.get("away_moneyline_median"))

    if not has_home_market and not has_away_market:
        return {
            "recommended_side": None,
            "recommended_price": np.nan,
            "edge": np.nan,
            "expected_value_per_unit": np.nan,
            "no_bet_reason": "no_matched_market_odds",
        }

    candidates = []

    if has_home_market:
        home_edge = row.get("edge_home", np.nan)
        home_ev = row.get("home_expected_value_per_unit", np.nan)
        if pd.notna(home_edge) and pd.notna(home_ev):
            candidates.append({
                "side": row.get("home_team_name"),
                "price": row.get("home_moneyline_median"),
                "edge": float(home_edge),
                "ev": float(home_ev),
                "side_type": "home",
            })

    if has_away_market:
        away_edge = row.get("edge_away", np.nan)
        away_ev = row.get("away_expected_value_per_unit", np.nan)
        if pd.notna(away_edge) and pd.notna(away_ev):
            candidates.append({
                "side": row.get("away_team_name"),
                "price": row.get("away_moneyline_median"),
                "edge": float(away_edge),
                "ev": float(away_ev),
                "side_type": "away",
            })

    qualifying = [c for c in candidates if c["edge"] >= min_edge and c["ev"] >= min_ev]

    if not qualifying:
        max_edge = max([c["edge"] for c in candidates], default=np.nan)
        max_ev = max([c["ev"] for c in candidates], default=np.nan)

        if pd.notna(max_edge) and max_edge < min_edge:
            reason = "below_min_edge"
        elif pd.notna(max_ev) and max_ev < min_ev:
            reason = "below_min_ev"
        else:
            reason = "no_positive_qualifying_side"

        return {
            "recommended_side": None,
            "recommended_price": np.nan,
            "edge": np.nan,
            "expected_value_per_unit": np.nan,
            "no_bet_reason": reason,
        }

    best = max(qualifying, key=lambda c: (c["ev"], c["edge"]))
    return {
        "recommended_side": best["side"],
        "recommended_price": best["price"],
        "edge": best["edge"],
        "expected_value_per_unit": best["ev"],
        "no_bet_reason": "recommended",
    }


def _write_debug_candidate_files(
    frame: pd.DataFrame,
    output_path: Path,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> None:
    debug_dir = output_path.parent / "debug_scoring_candidates"
    debug_dir.mkdir(parents=True, exist_ok=True)

    df = _ensure_datetime_columns(frame)
    unresolved = df[df.get("target_home_win", pd.Series(np.nan, index=df.index)).isna()].copy()

    keep = [
        "game_pk",
        "official_date",
        "game_datetime_utc",
        "away_team_name",
        "home_team_name",
        "abstract_state",
        "detailed_state",
        "target_home_win",
    ]
    keep = [c for c in keep if c in unresolved.columns]

    unresolved[unresolved["game_datetime_utc"] < start_ts][keep].to_csv(
        debug_dir / "historical_unresolved.csv",
        index=False,
    )
    unresolved[unresolved["game_datetime_utc"] >= end_ts][keep].to_csv(
        debug_dir / "far_future_unresolved.csv",
        index=False,
    )


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
    metadata = _load_adjacent_metadata(model_path)
    estimator = bundle.get("estimator") or bundle.get("model") or bundle.get("pipeline")
    feature_cols = bundle.get("feature_cols") or bundle.get("features") or bundle.get("feature_columns")
    run_id = bundle.get("run_id") or bundle.get("model_name") or metadata.get("model_name") or model_path.stem
    probability_shrink, shrink_center = _resolve_shrink_settings(args, bundle, metadata)

    if estimator is None or feature_cols is None:
        raise SystemExit(
            "Model bundle must contain an estimator/model/pipeline and feature_cols/features/feature_columns."
        )

    frame = pd.read_parquet(feature_path)
    scored_at_ts, start_ts, end_ts = _build_scoring_window(args)

    upcoming, diagnostics = build_scoring_candidates(
        frame,
        start_ts,
        end_ts,
        include_live=args.include_live,
        allow_scored_targets=args.allow_scored_targets,
    )
    print({"scoring_window_diagnostics": diagnostics})

    if args.write_debug_candidates:
        _write_debug_candidate_files(frame, output_path, start_ts, end_ts)

    if upcoming.empty:
        raise SystemExit(
            "No true upcoming scoring candidates found. "
            f"Diagnostics: {json.dumps(diagnostics, default=str)}"
        )

    if args.only_bettable:
        required_market_cols = [
            "market_home_no_vig_prob",
            "market_away_no_vig_prob",
            "home_moneyline_median",
            "away_moneyline_median",
        ]
        for col in required_market_cols:
            if col not in upcoming.columns:
                upcoming[col] = np.nan
        upcoming = upcoming[
            upcoming["market_home_no_vig_prob"].notna()
            & upcoming["market_away_no_vig_prob"].notna()
            & upcoming["home_moneyline_median"].notna()
            & upcoming["away_moneyline_median"].notna()
        ].copy()
        if upcoming.empty:
            raise SystemExit("No bettable games with matched market odds found in the scoring window.")

    missing_cols = [c for c in feature_cols if c not in upcoming.columns]
    if missing_cols:
        raise SystemExit(f"Feature file missing model columns: {missing_cols}")

    market_cols = [
        "market_home_no_vig_prob",
        "market_away_no_vig_prob",
        "home_moneyline_median",
        "away_moneyline_median",
    ]
    for col in market_cols:
        if col not in upcoming.columns:
            upcoming[col] = np.nan

    raw_probs = estimator.predict_proba(upcoming[feature_cols])[:, 1]
    shrunk_probs = apply_probability_shrink(
        raw_probs,
        shrink=probability_shrink,
        center=shrink_center,
    )
    upcoming["model_home_win_prob_raw"] = raw_probs
    upcoming["model_home_win_prob"] = shrunk_probs
    upcoming["model_away_win_prob"] = 1.0 - upcoming["model_home_win_prob"]
    upcoming["probability_shrink"] = probability_shrink
    upcoming["shrink_center"] = shrink_center

    upcoming["edge_home"] = upcoming["model_home_win_prob"] - upcoming["market_home_no_vig_prob"]
    upcoming["edge_away"] = upcoming["model_away_win_prob"] - upcoming["market_away_no_vig_prob"]
    upcoming["home_expected_value_per_unit"] = [
        _safe_ev(p, price) for p, price in zip(upcoming["model_home_win_prob"], upcoming["home_moneyline_median"])
    ]
    upcoming["away_expected_value_per_unit"] = [
        _safe_ev(p, price) for p, price in zip(upcoming["model_away_win_prob"], upcoming["away_moneyline_median"])
    ]
    upcoming["has_market_odds"] = (
        upcoming["market_home_no_vig_prob"].notna()
        & upcoming["market_away_no_vig_prob"].notna()
        & upcoming["home_moneyline_median"].notna()
        & upcoming["away_moneyline_median"].notna()
    )

    recs = upcoming.apply(
        lambda row: _choose_recommendation(row, args.min_edge, args.min_ev),
        axis=1,
        result_type="expand",
    )
    for col in recs.columns:
        upcoming[col] = recs[col]

    upcoming["run_id"] = run_id
    upcoming["scored_at_utc"] = scored_at_ts.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    keep = [
        "run_id",
        "scored_at_utc",
        "game_pk",
        "official_date",
        "game_datetime_utc",
        "abstract_state",
        "detailed_state",
        "home_team_name",
        "away_team_name",
        "model_home_win_prob_raw",
        "model_home_win_prob",
        "model_away_win_prob",
        "probability_shrink",
        "shrink_center",
        "market_home_no_vig_prob",
        "market_away_no_vig_prob",
        "home_moneyline_median",
        "away_moneyline_median",
        "edge_home",
        "edge_away",
        "home_expected_value_per_unit",
        "away_expected_value_per_unit",
        "has_market_odds",
        "recommended_side",
        "recommended_price",
        "edge",
        "expected_value_per_unit",
        "no_bet_reason",
    ]
    pred = upcoming[[c for c in keep if c in upcoming.columns]].copy()
    pred.to_csv(output_path, index=False)

    rows = []
    for _, row in upcoming.iterrows():
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
            "market_home_no_vig_prob": _safe_float(row.get("market_home_no_vig_prob")),
            "home_moneyline_median": _safe_float(row.get("home_moneyline_median")),
            "away_moneyline_median": _safe_float(row.get("away_moneyline_median")),
            "recommended_side": row.get("recommended_side"),
            "recommended_price": _safe_float(row.get("recommended_price")),
            "edge": _safe_float(row.get("edge")),
            "expected_value_per_unit": _safe_float(row.get("expected_value_per_unit")),
            "feature_snapshot_json": json.dumps(feature_snapshot, default=str),
        })

    with connect(settings.odds_db_path) as conn:
        count = insert_prediction_rows(conn, rows)
        conn.commit()

    recommended = int(pred["recommended_side"].notna().sum()) if "recommended_side" in pred.columns else 0
    bettable = int(pred["has_market_odds"].sum()) if "has_market_odds" in pred.columns else 0
    print({
        "predictions": len(pred),
        "bettable_with_market_odds": bettable,
        "recommended_bets": recommended,
        "inserted_db_rows": count,
        "output": str(output_path),
        "model": str(model_path),
        "score_start": str(start_ts),
        "score_end": str(end_ts),
        "probability_shrink": probability_shrink,
        "shrink_center": shrink_center,
    })


if __name__ == "__main__":
    main()
