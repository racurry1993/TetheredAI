from __future__ import annotations

import argparse
import json
import joblib
import os
import re
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

try:
    from mlb_betting.betting_math import expected_value_per_unit
except Exception:  # pragma: no cover - fallback for older repos
    def _american_profit_per_unit(price: Any) -> float:
        if price is None or pd.isna(price):
            return np.nan
        price_f = float(price)
        if price_f > 0:
            return price_f / 100.0
        if price_f < 0:
            return 100.0 / abs(price_f)
        return np.nan

    def expected_value_per_unit(model_prob: float, american_price: float) -> float:
        if model_prob is None or american_price is None or pd.isna(model_prob) or pd.isna(american_price):
            return np.nan
        profit = _american_profit_per_unit(american_price)
        if not np.isfinite(profit):
            return np.nan
        return float(model_prob) * profit - (1.0 - float(model_prob))

from mlb_betting.config import get_settings
from mlb_betting.db import connect, init_db, insert_prediction_rows

try:
    from mlb_betting.logging_utils import configure_logging
except Exception:  # pragma: no cover - fallback for older repos
    def configure_logging() -> None:
        return None


BAD_DETAIL_STATE_PATTERN = "final|postponed|completed|cancelled|canceled|suspended|game over"
DEFAULT_ALLOWED_ABSTRACT_STATES = {"preview"}
LIVE_ALLOWED_ABSTRACT_STATES = {"preview", "live"}

PREFERRED_OUTPUT_COLUMNS = [
    "run_id",
    "scored_at_utc",
    "champion_model_run_id",
    "champion_model_name",
    "champion_model_family",
    "champion_pick_tier_method",
    "min_pick_tier",
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
    "model_pick",
    "model_pick_team_type",
    "model_confidence",
    "pick_tier",
    "pick_tier_rank",
    "probability_shrink",
    "shrink_center",
    "market_home_no_vig_prob",
    "market_away_no_vig_prob",
    "home_moneyline_median",
    "away_moneyline_median",
    "odds_event_id",
    "odds_start_diff_minutes",
    "edge_home",
    "edge_away",
    "home_expected_value_per_unit",
    "away_expected_value_per_unit",
    "market_pick_no_vig_prob",
    "model_pick_moneyline",
    "model_pick_edge",
    "model_pick_expected_value_per_unit",
    "has_market_odds",
    "recommended_side",
    "recommended_team_type",
    "recommended_price",
    "recommended_model_prob",
    "recommended_market_prob",
    "edge",
    "expected_value_per_unit",
    "no_bet_reason",
]


def env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 Z format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def latest_model_path(model_dir: Path) -> Path:
    """Find the preferred/latest model bundle.

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

    The production notebook exports a dict with keys like estimator, model,
    feature_cols, pick_tiers, metadata, and run_id. If a raw estimator is ever
    provided, wrap it into a minimal bundle so the rest of the scorer still works.
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
            "Score truly upcoming MLB moneyline games using the production model bundle. "
            "Pick tiers are read from the bundle; confidence thresholds are not hard-coded in this script."
        )
    )
    parser.add_argument(
        "--features",
        default=os.environ.get("FEATURE_PATH"),
        help="Path to feature parquet. Defaults to data/processed/mlb_game_features.parquet.",
    )
    parser.add_argument(
        "--model",
        "--model-path",
        dest="model",
        default=os.environ.get("MODEL_PATH"),
        help="Path to model bundle. Defaults to latest model in models/.",
    )
    parser.add_argument(
        "--output",
        default=os.environ.get("PREDICTIONS_OUTPUT"),
        help="Path to output CSV. Defaults to data/predictions/mlb_moneyline_predictions.csv.",
    )
    parser.add_argument(
        "--start-date",
        default=os.environ.get("SCORE_START_DATE"),
        help=(
            "YYYY-MM-DD. Optional debug/backtest date. If omitted, scoring starts at now UTC "
            "+ --min-minutes-before-start."
        ),
    )
    parser.add_argument(
        "--days-forward",
        type=int,
        default=int(os.environ.get("DAYS_FORWARD", "3")),
        help="Number of days forward to score from start timestamp.",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=float(os.environ.get("MIN_EDGE", "0.02")),
        help="Minimum no-vig probability edge required for a bet recommendation on the model-selected side.",
    )
    parser.add_argument(
        "--min-ev",
        type=float,
        default=float(os.environ.get("MIN_EV", "0.00")),
        help="Minimum expected value per 1 unit staked required for a recommendation on the model-selected side.",
    )
    parser.add_argument(
        "--min-pick-tier",
        default=os.environ.get("MIN_PICK_TIER", "Strong"),
        help=(
            "Minimum learned pick tier required for a recommendation, for example Strong, Premium, Elite. "
            "The confidence threshold for that tier is read from the model bundle. Use Pass/Any/None to disable tier gating."
        ),
    )
    parser.add_argument(
        "--min-minutes-before-start",
        type=int,
        default=int(os.environ.get("MIN_MINUTES_BEFORE_START", "30")),
        help="When --start-date is omitted, exclude games starting within this many minutes.",
    )
    parser.add_argument(
        "--include-live",
        action="store_true",
        default=env_bool("INCLUDE_LIVE", False),
        help="Include rows with abstract_state=Live. Default is scheduled/preview games only.",
    )
    parser.add_argument(
        "--only-bettable",
        action="store_true",
        default=env_bool("ONLY_BETTABLE", False),
        help="If set, output only rows with matched market odds. By default, outputs model-only rows too.",
    )
    parser.add_argument(
        "--allow-scored-targets",
        action="store_true",
        default=env_bool("ALLOW_SCORED_TARGETS", False),
        help=(
            "Debug/backtest only: allow rows with a non-null target in the scoring window. "
            "Production scoring should leave this off."
        ),
    )
    parser.add_argument(
        "--write-debug-candidates",
        action="store_true",
        default=env_bool("WRITE_DEBUG_CANDIDATES", False),
        help="Write unresolved/historical/far-future candidate diagnostics next to the prediction CSV.",
    )
    parser.add_argument(
        "--probability-shrink",
        type=float,
        default=None,
        help=(
            "Optional override for probability shrinkage. If omitted, reads probability_shrink from "
            "the model bundle or adjacent metadata JSON. New learned-tier bundles normally use 1.0."
        ),
    )
    parser.add_argument(
        "--shrink-center",
        type=float,
        default=None,
        help=(
            "Optional override for shrink center. If omitted, reads shrink_center from the model bundle "
            "or adjacent metadata JSON. Default is 0.5."
        ),
    )
    parser.add_argument(
        "--allow-missing-pick-tiers",
        action="store_true",
        default=env_bool("ALLOW_MISSING_PICK_TIERS", False),
        help="Allow scoring with no learned pick_tiers in the bundle. Not recommended for production.",
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


def _normalize_team_name(value: Any) -> str:
    """Normalize team names for matching MLB Stats API rows to Odds API rows."""
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    aliases = {
        "oakland athletics": "athletics",
        "athletics": "athletics",
        "la dodgers": "los angeles dodgers",
        "los angeles dodgers": "los angeles dodgers",
        "la angels": "los angeles angels",
        "los angeles angels": "los angeles angels",
        "ny yankees": "new york yankees",
        "new york yankees": "new york yankees",
        "ny mets": "new york mets",
        "new york mets": "new york mets",
        "sf giants": "san francisco giants",
        "san francisco giants": "san francisco giants",
        "sd padres": "san diego padres",
        "san diego padres": "san diego padres",
        "tb rays": "tampa bay rays",
        "tampa bay rays": "tampa bay rays",
        "kc royals": "kansas city royals",
        "kansas city royals": "kansas city royals",
        "st louis cardinals": "st louis cardinals",
        "saint louis cardinals": "st louis cardinals",
    }
    return aliases.get(text, text)


def _american_to_implied_prob(price: Any) -> float:
    price_f = _safe_float(price)
    if price_f is None or price_f == 0:
        return np.nan
    if price_f > 0:
        return 100.0 / (price_f + 100.0)
    return (-price_f) / ((-price_f) + 100.0)


def _no_vig_probs_from_prices(home_price: Any, away_price: Any) -> tuple[float, float]:
    home_imp = _american_to_implied_prob(home_price)
    away_imp = _american_to_implied_prob(away_price)
    denom = home_imp + away_imp
    if not np.isfinite(denom) or denom <= 0:
        return np.nan, np.nan
    return float(home_imp / denom), float(away_imp / denom)


def _latest_h2h_odds_events(conn) -> pd.DataFrame:
    """Return latest median H2H price by event/side from odds_events + odds_snapshots."""
    try:
        raw = pd.read_sql_query(
            """
            SELECT
                e.event_id,
                e.commence_time_utc,
                e.home_team,
                e.away_team,
                e.home_team_norm,
                e.away_team_norm,
                s.fetched_at_utc,
                s.bookmaker_key,
                s.market_key,
                s.outcome_name,
                s.outcome_name_norm,
                s.outcome_price
            FROM odds_snapshots s
            JOIN odds_events e
              ON e.event_id = s.event_id
            WHERE lower(s.market_key) IN ('h2h', 'moneyline')
              AND s.outcome_price IS NOT NULL
            """,
            conn,
        )
    except Exception as exc:
        print({"odds_attachment_warning": f"Could not read odds tables: {exc}"})
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    raw["fetched_at_utc"] = pd.to_datetime(raw["fetched_at_utc"], utc=True, errors="coerce")
    raw["commence_time_utc"] = pd.to_datetime(raw["commence_time_utc"], utc=True, errors="coerce")
    raw["home_norm"] = raw["home_team_norm"].map(_normalize_team_name)
    raw["away_norm"] = raw["away_team_norm"].map(_normalize_team_name)
    raw["outcome_norm"] = raw["outcome_name_norm"].where(
        raw["outcome_name_norm"].notna(),
        raw["outcome_name"],
    ).map(_normalize_team_name)

    raw = raw.sort_values(["fetched_at_utc", "event_id"])
    raw = raw.drop_duplicates(
        ["event_id", "bookmaker_key", "market_key", "outcome_norm"],
        keep="last",
    )

    raw["side"] = np.where(
        raw["outcome_norm"].eq(raw["home_norm"]),
        "home",
        np.where(raw["outcome_norm"].eq(raw["away_norm"]), "away", None),
    )
    raw = raw[raw["side"].notna()].copy()
    if raw.empty:
        return pd.DataFrame()

    grouped = (
        raw.groupby(
            [
                "event_id",
                "commence_time_utc",
                "home_team",
                "away_team",
                "home_norm",
                "away_norm",
                "side",
            ],
            dropna=False,
        )["outcome_price"]
        .median()
        .reset_index()
    )

    pivot = grouped.pivot_table(
        index=["event_id", "commence_time_utc", "home_team", "away_team", "home_norm", "away_norm"],
        columns="side",
        values="outcome_price",
        aggfunc="median",
    ).reset_index()
    pivot.columns.name = None

    if "home" not in pivot.columns:
        pivot["home"] = np.nan
    if "away" not in pivot.columns:
        pivot["away"] = np.nan

    pivot = pivot.rename(columns={"home": "home_moneyline_median", "away": "away_moneyline_median"})
    probs = pivot.apply(
        lambda row: _no_vig_probs_from_prices(row["home_moneyline_median"], row["away_moneyline_median"]),
        axis=1,
        result_type="expand",
    )
    pivot["market_home_no_vig_prob"] = probs[0]
    pivot["market_away_no_vig_prob"] = probs[1]
    return pivot


def attach_latest_moneyline_odds_from_db(
    upcoming: pd.DataFrame,
    db_path: Path,
    *,
    max_start_diff_minutes: int = 180,
) -> pd.DataFrame:
    """Attach latest H2H odds by normalized teams + start-time tolerance."""
    out = upcoming.copy()
    for col in [
        "market_home_no_vig_prob",
        "market_away_no_vig_prob",
        "home_moneyline_median",
        "away_moneyline_median",
    ]:
        if col not in out.columns:
            out[col] = np.nan

    before = (
        int((
            out["market_home_no_vig_prob"].notna()
            & out["market_away_no_vig_prob"].notna()
            & out["home_moneyline_median"].notna()
            & out["away_moneyline_median"].notna()
        ).sum())
        if len(out)
        else 0
    )

    try:
        with connect(db_path) as conn:
            odds = _latest_h2h_odds_events(conn)
    except Exception as exc:
        print({"odds_attachment_warning": f"Could not attach odds from DB: {exc}"})
        return out

    if odds.empty:
        print({"odds_attachment": {"events_available": 0, "matched_rows": 0, "rows_with_market_before": before, "rows_with_market_after": before}})
        return out

    odds = odds.dropna(subset=["commence_time_utc", "home_norm", "away_norm"]).copy()
    matched = 0
    matched_event_ids: list[str] = []
    start_diffs: list[float] = []

    for idx, row in out.iterrows():
        game_time = pd.to_datetime(row.get("game_datetime_utc"), utc=True, errors="coerce")
        if pd.isna(game_time):
            continue

        home_norm = _normalize_team_name(row.get("home_team_name"))
        away_norm = _normalize_team_name(row.get("away_team_name"))
        candidates = odds[
            odds["home_norm"].eq(home_norm)
            & odds["away_norm"].eq(away_norm)
        ].copy()
        if candidates.empty:
            continue

        candidates["start_diff_minutes"] = (
            candidates["commence_time_utc"] - game_time
        ).dt.total_seconds().abs() / 60.0
        candidates = candidates[candidates["start_diff_minutes"] <= max_start_diff_minutes]
        if candidates.empty:
            continue

        best = candidates.sort_values("start_diff_minutes").iloc[0]
        out.loc[idx, "home_moneyline_median"] = best["home_moneyline_median"]
        out.loc[idx, "away_moneyline_median"] = best["away_moneyline_median"]
        out.loc[idx, "market_home_no_vig_prob"] = best["market_home_no_vig_prob"]
        out.loc[idx, "market_away_no_vig_prob"] = best["market_away_no_vig_prob"]
        out.loc[idx, "odds_event_id"] = best["event_id"]
        out.loc[idx, "odds_start_diff_minutes"] = best["start_diff_minutes"]
        matched += 1
        matched_event_ids.append(str(best["event_id"]))
        start_diffs.append(float(best["start_diff_minutes"]))

    after = (
        int((
            out["market_home_no_vig_prob"].notna()
            & out["market_away_no_vig_prob"].notna()
            & out["home_moneyline_median"].notna()
            & out["away_moneyline_median"].notna()
        ).sum())
        if len(out)
        else 0
    )

    print({
        "odds_attachment": {
            "events_available": int(len(odds)),
            "matched_rows_from_db": int(matched),
            "unique_events_matched": int(len(set(matched_event_ids))),
            "rows_with_market_before": before,
            "rows_with_market_after": after,
            "max_start_diff_minutes": max_start_diff_minutes,
            "matched_start_diff_minutes_max": max(start_diffs) if start_diffs else None,
        }
    })
    return out


def _load_adjacent_metadata(model_path: Path) -> dict[str, Any]:
    """Load optional JSON metadata next to a model bundle."""
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
    """Resolve probability shrinkage settings."""
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
    """Pull probabilities toward a center value to reduce overconfidence."""
    p = np.asarray(probabilities, dtype=float)
    p_adj = center + shrink * (p - center)
    return np.clip(p_adj, clip_low, clip_high)


def predict_home_win_prob(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return P(target_home_win == 1), checking class order when available."""
    proba = model.predict_proba(X)
    classes = getattr(model, "classes_", None)

    if classes is None and hasattr(model, "named_steps"):
        for step_name in reversed(list(model.named_steps.keys())):
            step = model.named_steps[step_name]
            if hasattr(step, "classes_"):
                classes = step.classes_
                break

    if classes is None:
        return np.asarray(proba, dtype=float)[:, 1]

    classes = list(classes)
    if 1 not in classes:
        raise ValueError(f"Expected positive class 1 in estimator classes, got {classes}")
    return np.asarray(proba, dtype=float)[:, classes.index(1)]


def _resolve_pick_tiers(bundle: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any] | None:
    """Return learned pick tiers from bundle or metadata."""
    candidates = [
        bundle.get("pick_tiers"),
        bundle.get("metadata", {}).get("pick_tiers") if isinstance(bundle.get("metadata"), dict) else None,
        metadata.get("pick_tiers"),
    ]
    for obj in candidates:
        if isinstance(obj, dict) and isinstance(obj.get("tiers"), list):
            return obj
    return None


def _tier_names_in_order(pick_tiers: dict[str, Any]) -> list[str]:
    names = []
    for item in pick_tiers.get("tiers", []):
        name = str(item.get("tier", "")).strip()
        if name and name not in names:
            names.append(name)
    return names


def _tier_rank_map(pick_tiers: dict[str, Any]) -> dict[str, int]:
    default_tier = str(pick_tiers.get("default_tier", "Pass")).strip() or "Pass"
    rank = {default_tier.lower(): 0}
    for idx, name in enumerate(_tier_names_in_order(pick_tiers), start=1):
        rank[name.lower()] = idx
    return rank


def _assign_pick_tier_from_learned_tiers(confidence: Any, pick_tiers: dict[str, Any]) -> str:
    """Assign a pick tier using thresholds saved in the model bundle."""
    default_tier = pick_tiers.get("default_tier", "Pass")
    conf = _safe_float(confidence)
    if conf is None:
        return default_tier

    tiers = sorted(
        pick_tiers.get("tiers", []),
        key=lambda x: float(x.get("threshold", np.inf)),
        reverse=True,
    )
    for tier_info in tiers:
        threshold = _safe_float(tier_info.get("threshold"))
        tier_name = tier_info.get("tier")
        if threshold is not None and tier_name and conf >= threshold:
            return str(tier_name)
    return default_tier


def _normalize_min_pick_tier(min_pick_tier: str | None) -> str | None:
    if min_pick_tier is None:
        return None
    value = str(min_pick_tier).strip()
    if value.lower() in {"", "any", "none", "no", "false", "off"}:
        return None
    return value


def _validate_min_pick_tier(min_pick_tier: str | None, pick_tiers: dict[str, Any]) -> str | None:
    value = _normalize_min_pick_tier(min_pick_tier)
    if value is None:
        return None
    rank_map = _tier_rank_map(pick_tiers)
    if value.lower() not in rank_map:
        available = [pick_tiers.get("default_tier", "Pass")] + _tier_names_in_order(pick_tiers)
        raise ValueError(f"min_pick_tier={value!r} is not in learned tiers. Available tiers: {available}")
    return value


def _tier_meets_minimum(tier: Any, min_pick_tier: str | None, pick_tiers: dict[str, Any]) -> bool:
    min_tier = _normalize_min_pick_tier(min_pick_tier)
    if min_tier is None:
        return True
    rank_map = _tier_rank_map(pick_tiers)
    return rank_map.get(str(tier).lower(), -1) >= rank_map.get(str(min_tier).lower(), 0)


def _tier_rank(tier: Any, pick_tiers: dict[str, Any]) -> int:
    rank_map = _tier_rank_map(pick_tiers)
    return int(rank_map.get(str(tier).lower(), -1))


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
    """Return rows that are truly candidates for pregame scoring."""
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


def _add_model_pick_columns(upcoming: pd.DataFrame, pick_tiers: dict[str, Any]) -> pd.DataFrame:
    out = upcoming.copy()
    out["model_pick_team_type"] = np.where(out["model_home_win_prob"] >= 0.5, "home", "away")
    out["model_pick"] = np.where(
        out["model_pick_team_type"].eq("home"),
        out.get("home_team_name", "home"),
        out.get("away_team_name", "away"),
    )
    out["model_confidence"] = np.maximum(out["model_home_win_prob"], out["model_away_win_prob"])
    out["pick_tier"] = out["model_confidence"].apply(lambda x: _assign_pick_tier_from_learned_tiers(x, pick_tiers))
    out["pick_tier_rank"] = out["pick_tier"].apply(lambda x: _tier_rank(x, pick_tiers))
    return out


def _add_model_pick_market_columns(upcoming: pd.DataFrame) -> pd.DataFrame:
    out = upcoming.copy()
    is_home_pick = out["model_pick_team_type"].eq("home")

    out["market_pick_no_vig_prob"] = np.where(
        is_home_pick,
        out["market_home_no_vig_prob"],
        out["market_away_no_vig_prob"],
    )
    out["model_pick_moneyline"] = np.where(
        is_home_pick,
        out["home_moneyline_median"],
        out["away_moneyline_median"],
    )
    out["model_pick_edge"] = np.where(
        is_home_pick,
        out["edge_home"],
        out["edge_away"],
    )
    out["model_pick_expected_value_per_unit"] = np.where(
        is_home_pick,
        out["home_expected_value_per_unit"],
        out["away_expected_value_per_unit"],
    )
    return out


def _choose_recommendation(
    row: pd.Series,
    min_edge: float,
    min_ev: float,
    min_pick_tier: str | None,
    pick_tiers: dict[str, Any],
) -> dict[str, object]:
    """Recommend only the model-selected side if tier, edge, and EV pass.

    The notebook validation tiers are based on selected-side accuracy. Therefore,
    production should not recommend the opposite side just because it has a value edge.
    """
    pick_tier = row.get("pick_tier")
    if not _tier_meets_minimum(pick_tier, min_pick_tier, pick_tiers):
        return {
            "recommended_side": None,
            "recommended_team_type": None,
            "recommended_price": np.nan,
            "recommended_model_prob": np.nan,
            "recommended_market_prob": np.nan,
            "edge": np.nan,
            "expected_value_per_unit": np.nan,
            "no_bet_reason": f"below_min_pick_tier:{pick_tier}",
        }

    side_type = row.get("model_pick_team_type")
    side_name = row.get("model_pick")
    model_prob = _safe_float(row.get("model_confidence"))
    market_prob = _safe_float(row.get("market_pick_no_vig_prob"))
    price = _safe_float(row.get("model_pick_moneyline"))
    edge = _safe_float(row.get("model_pick_edge"))
    ev = _safe_float(row.get("model_pick_expected_value_per_unit"))

    if market_prob is None or price is None:
        return {
            "recommended_side": None,
            "recommended_team_type": side_type,
            "recommended_price": np.nan,
            "recommended_model_prob": model_prob,
            "recommended_market_prob": np.nan,
            "edge": np.nan,
            "expected_value_per_unit": np.nan,
            "no_bet_reason": "no_market_odds_for_model_pick",
        }

    if edge is None or not np.isfinite(edge):
        return {
            "recommended_side": None,
            "recommended_team_type": side_type,
            "recommended_price": price,
            "recommended_model_prob": model_prob,
            "recommended_market_prob": market_prob,
            "edge": np.nan,
            "expected_value_per_unit": ev if ev is not None else np.nan,
            "no_bet_reason": "missing_edge_for_model_pick",
        }

    if ev is None or not np.isfinite(ev):
        return {
            "recommended_side": None,
            "recommended_team_type": side_type,
            "recommended_price": price,
            "recommended_model_prob": model_prob,
            "recommended_market_prob": market_prob,
            "edge": edge,
            "expected_value_per_unit": np.nan,
            "no_bet_reason": "missing_ev_for_model_pick",
        }

    if edge < min_edge:
        return {
            "recommended_side": None,
            "recommended_team_type": side_type,
            "recommended_price": price,
            "recommended_model_prob": model_prob,
            "recommended_market_prob": market_prob,
            "edge": edge,
            "expected_value_per_unit": ev,
            "no_bet_reason": "below_min_edge_model_pick",
        }

    if ev < min_ev:
        return {
            "recommended_side": None,
            "recommended_team_type": side_type,
            "recommended_price": price,
            "recommended_model_prob": model_prob,
            "recommended_market_prob": market_prob,
            "edge": edge,
            "expected_value_per_unit": ev,
            "no_bet_reason": "below_min_ev_model_pick",
        }

    return {
        "recommended_side": side_name,
        "recommended_team_type": side_type,
        "recommended_price": price,
        "recommended_model_prob": model_prob,
        "recommended_market_prob": market_prob,
        "edge": edge,
        "expected_value_per_unit": ev,
        "no_bet_reason": "recommended_model_pick",
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
    target_series = df.get("target_home_win", pd.Series(np.nan, index=df.index))
    unresolved = df[target_series.isna()].copy()

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


def _write_empty_predictions(output_path: Path, message: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=PREFERRED_OUTPUT_COLUMNS).to_csv(output_path, index=False)
    print({"predictions": 0, "recommended_bets": 0, "output": str(output_path), "message": message})


def _json_feature_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return str(value)
    return value


def main() -> None:
    configure_logging()
    args = parse_args()
    settings = get_settings()
    init_db(settings.odds_db_path)

    feature_path = Path(args.features) if args.features else settings.data_dir / "processed" / "mlb_game_features.parquet"
    model_path = Path(args.model) if args.model else latest_model_path(settings.model_dir)
    output_path = Path(args.output) if args.output else settings.data_dir / "predictions" / "mlb_moneyline_predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model bundle not found: {model_path}")

    bundle = load_model_bundle(model_path)
    metadata = _load_adjacent_metadata(model_path)
    estimator = bundle.get("estimator") or bundle.get("model") or bundle.get("pipeline")
    feature_cols = bundle.get("feature_cols") or bundle.get("features") or bundle.get("feature_columns")
    champion_run_id = bundle.get("run_id") or metadata.get("run_id") or bundle.get("model_name") or metadata.get("model_name") or model_path.stem
    champion_model_name = bundle.get("model_name") or metadata.get("model_name") or model_path.stem
    champion_model_family = bundle.get("model_family") or metadata.get("model_family") or ""
    probability_shrink, shrink_center = _resolve_shrink_settings(args, bundle, metadata)

    pick_tiers = _resolve_pick_tiers(bundle, metadata)
    missing_pick_tiers_allowed = False
    if pick_tiers is None:
        if args.allow_missing_pick_tiers:
            missing_pick_tiers_allowed = True
            pick_tiers = {"method": "missing_pick_tiers_fallback", "default_tier": "Pass", "tiers": []}
            print({"pick_tiers_warning": "No learned pick_tiers found. All rows will be Pass unless --allow-missing-pick-tiers is removed."})
        else:
            raise SystemExit(
                "Model bundle does not contain learned pick_tiers. Re-export the model from the production notebook "
                "or rerun with --allow-missing-pick-tiers for debugging only."
            )

    min_pick_tier = None if missing_pick_tiers_allowed else _validate_min_pick_tier(args.min_pick_tier, pick_tiers)

    if estimator is None or feature_cols is None:
        raise SystemExit(
            "Model bundle must contain an estimator/model/pipeline and feature_cols/features/feature_columns."
        )
    feature_cols = list(feature_cols)

    print({
        "loaded_model": str(model_path),
        "champion_run_id": champion_run_id,
        "champion_model_name": champion_model_name,
        "champion_model_family": champion_model_family,
        "feature_count": len(feature_cols),
        "pick_tier_method": pick_tiers.get("method"),
        "learned_tiers": pick_tiers.get("tiers", []),
        "min_pick_tier_for_recommendations": min_pick_tier,
    })

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
        _write_empty_predictions(
            output_path,
            "No true upcoming scoring candidates found. " + json.dumps(diagnostics, default=str),
        )
        return

    upcoming = attach_latest_moneyline_odds_from_db(upcoming, settings.odds_db_path)

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
            _write_empty_predictions(output_path, "No bettable games with matched market odds found in the scoring window.")
            return

    missing_cols = [c for c in feature_cols if c not in upcoming.columns]
    if missing_cols:
        raise SystemExit(f"Feature file missing model columns: {missing_cols[:50]}{'...' if len(missing_cols) > 50 else ''}")

    market_cols = [
        "market_home_no_vig_prob",
        "market_away_no_vig_prob",
        "home_moneyline_median",
        "away_moneyline_median",
    ]
    for col in market_cols:
        if col not in upcoming.columns:
            upcoming[col] = np.nan

    raw_probs = predict_home_win_prob(estimator, upcoming[feature_cols])
    final_probs = apply_probability_shrink(
        raw_probs,
        shrink=probability_shrink,
        center=shrink_center,
    )

    upcoming["model_home_win_prob_raw"] = np.clip(raw_probs, 1e-6, 1 - 1e-6)
    upcoming["model_home_win_prob"] = np.clip(final_probs, 1e-6, 1 - 1e-6)
    upcoming["model_away_win_prob"] = 1.0 - upcoming["model_home_win_prob"]
    upcoming["probability_shrink"] = probability_shrink
    upcoming["shrink_center"] = shrink_center

    upcoming = _add_model_pick_columns(upcoming, pick_tiers)

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

    upcoming = _add_model_pick_market_columns(upcoming)

    recs = upcoming.apply(
        lambda row: _choose_recommendation(row, args.min_edge, args.min_ev, min_pick_tier, pick_tiers),
        axis=1,
        result_type="expand",
    )
    for col in recs.columns:
        upcoming[col] = recs[col]

    score_run_id = f"score_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{champion_run_id}"
    upcoming["run_id"] = score_run_id
    upcoming["champion_model_run_id"] = champion_run_id
    upcoming["champion_model_name"] = champion_model_name
    upcoming["champion_model_family"] = champion_model_family
    upcoming["champion_pick_tier_method"] = pick_tiers.get("method", "")
    upcoming["min_pick_tier"] = min_pick_tier if min_pick_tier is not None else "Any"
    upcoming["scored_at_utc"] = scored_at_ts.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    pred = upcoming[[c for c in PREFERRED_OUTPUT_COLUMNS if c in upcoming.columns]].copy()
    pred = pred.sort_values(["game_datetime_utc", "game_pk"] if "game_pk" in pred.columns else ["game_datetime_utc"])
    pred.to_csv(output_path, index=False)

    rows = []
    for _, row in upcoming.iterrows():
        feature_snapshot = {c: _json_feature_value(row.get(c)) for c in feature_cols}
        rows.append({
            "run_id": score_run_id,
            "scored_at_utc": row["scored_at_utc"],
            "game_pk": int(row["game_pk"]) if pd.notna(row.get("game_pk")) else None,
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
    tier_counts = pred["pick_tier"].value_counts(dropna=False).to_dict() if "pick_tier" in pred.columns else {}
    print({
        "predictions": len(pred),
        "bettable_with_market_odds": bettable,
        "recommended_bets": recommended,
        "tier_counts": tier_counts,
        "inserted_db_rows": count,
        "output": str(output_path),
        "model": str(model_path),
        "score_start": str(start_ts),
        "score_end": str(end_ts),
        "probability_shrink": probability_shrink,
        "shrink_center": shrink_center,
        "min_pick_tier": min_pick_tier if min_pick_tier is not None else "Any",
        "min_edge": args.min_edge,
        "min_ev": args.min_ev,
    })


if __name__ == "__main__":
    main()
