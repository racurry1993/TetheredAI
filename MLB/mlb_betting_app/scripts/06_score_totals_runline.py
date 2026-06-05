from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlb_betting.betting_math import expected_value_per_unit
from mlb_betting.config import get_settings
from mlb_betting.db import connect, init_db
from mlb_betting.logging_utils import configure_logging


BAD_DETAIL_STATE_PATTERN = "final|postponed|completed|cancelled|canceled|suspended|game over"
DEFAULT_ALLOWED_ABSTRACT_STATES = {"preview"}
LIVE_ALLOWED_ABSTRACT_STATES = {"preview", "live"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score MLB totals and run-line markets from exported totals/margin model bundles."
    )
    parser.add_argument("--features", default=None, help="Feature parquet path. Default: data/processed/mlb_game_features.parquet")
    parser.add_argument("--totals-model", default=None, help="Totals regression model bundle. Default: models/mlb_total_runs_champion.joblib")
    parser.add_argument("--margin-model", default=None, help="Home-margin regression model bundle. Default: models/mlb_home_margin_champion.joblib")
    parser.add_argument("--output", default=None, help="Output CSV. Default: data/predictions/mlb_totals_runline_predictions.csv")
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD debug/backtest scoring start. Default: now + min minutes.")
    parser.add_argument("--days-forward", type=int, default=3)
    parser.add_argument("--min-minutes-before-start", type=int, default=30)
    parser.add_argument("--include-live", action="store_true")
    parser.add_argument("--allow-scored-targets", action="store_true")
    parser.add_argument("--only-bettable", action="store_true")
    parser.add_argument("--min-total-runs-edge", type=float, default=0.35, help="Minimum model-vs-market run edge for over/under recs.")
    parser.add_argument("--min-margin-edge", type=float, default=0.25, help="Minimum expected margin-vs-spread edge for run-line recs.")
    parser.add_argument("--min-prob-edge", type=float, default=0.02, help="Minimum no-vig probability edge.")
    parser.add_argument("--min-ev", type=float, default=0.00, help="Minimum expected value per 1 unit staked.")
    parser.add_argument("--max-start-diff-minutes", type=int, default=180)
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


def _safe_ev(model_prob: Any, price: Any) -> float:
    p = _safe_float(model_prob)
    price_f = _safe_float(price)
    if p is None or price_f is None:
        return np.nan
    try:
        return float(expected_value_per_unit(p, price_f))
    except Exception:
        return np.nan


def _american_to_implied_prob(price: Any) -> float:
    price_f = _safe_float(price)
    if price_f is None or price_f == 0:
        return np.nan
    if price_f > 0:
        return 100.0 / (price_f + 100.0)
    return (-price_f) / ((-price_f) + 100.0)


def _no_vig_probs_from_prices(a_price: Any, b_price: Any) -> tuple[float, float]:
    a_imp = _american_to_implied_prob(a_price)
    b_imp = _american_to_implied_prob(b_price)
    denom = a_imp + b_imp
    if not np.isfinite(denom) or denom <= 0:
        return np.nan, np.nan
    return float(a_imp / denom), float(b_imp / denom)


def _norm_cdf(x: Any) -> float:
    x_f = _safe_float(x)
    if x_f is None:
        return np.nan
    return float(0.5 * (1.0 + math.erf(x_f / math.sqrt(2.0))))


def _normalize_team_name(value: Any) -> str:
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


def load_model_bundle(path: Path) -> dict[str, Any]:
    obj = joblib.load(path)
    if isinstance(obj, dict):
        return obj
    return {"model_name": path.stem, "model": obj, "feature_cols": getattr(obj, "feature_names_in_", None)}


def _resolve_estimator(bundle: dict[str, Any]) -> Any:
    estimator = bundle.get("estimator") or bundle.get("model") or bundle.get("pipeline")
    if estimator is None:
        raise ValueError("Model bundle must include estimator/model/pipeline")
    return estimator


def _resolve_feature_cols(bundle: dict[str, Any]) -> list[str]:
    cols = bundle.get("feature_cols") or bundle.get("features") or bundle.get("feature_columns")
    if cols is None:
        raise ValueError("Model bundle must include feature_cols/features/feature_columns")
    return list(cols)


def _resolve_residual_std(bundle: dict[str, Any], *, default: float) -> float:
    sigma = bundle.get("residual_std") or bundle.get("validation_residual_std") or bundle.get("residual_sigma") or default
    try:
        sigma = float(sigma)
    except Exception:
        sigma = default
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = default
    return sigma


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


def _build_scoring_window(args: argparse.Namespace) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    scored_at = pd.Timestamp.now(tz="UTC")
    if args.start_date:
        start_ts = pd.to_datetime(args.start_date, utc=True)
    else:
        start_ts = scored_at + pd.Timedelta(minutes=args.min_minutes_before_start)
    end_ts = start_ts + pd.Timedelta(days=args.days_forward)
    return scored_at, start_ts, end_ts


def build_scoring_candidates(
    frame: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    *,
    include_live: bool = False,
    allow_scored_targets: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = _ensure_datetime_columns(frame)
    if "target_home_win" in df.columns and not allow_scored_targets:
        unresolved_mask = df["target_home_win"].isna()
    else:
        unresolved_mask = pd.Series(True, index=df.index)

    time_mask = (df["game_datetime_utc"] >= start_ts) & (df["game_datetime_utc"] < end_ts)
    status_mask = pd.Series(True, index=df.index)
    allowed_abstract = LIVE_ALLOWED_ABSTRACT_STATES if include_live else DEFAULT_ALLOWED_ABSTRACT_STATES

    if "abstract_state" in df.columns:
        abstract = df["abstract_state"].astype("string").str.strip().str.lower()
        status_mask &= abstract.isna() | abstract.isin(allowed_abstract)
    if "detailed_state" in df.columns:
        detailed = df["detailed_state"].astype("string").str.strip().str.lower()
        status_mask &= ~detailed.str.contains(BAD_DETAIL_STATE_PATTERN, na=False)

    candidates = df[unresolved_mask & time_mask & status_mask].copy()
    diagnostics = {
        "total_rows": int(len(df)),
        "unresolved_or_allowed_rows": int(unresolved_mask.sum()),
        "in_time_window_rows": int(time_mask.sum()),
        "status_allowed_rows": int(status_mask.sum()),
        "scoring_candidates": int(len(candidates)),
        "score_start_utc": str(start_ts),
        "score_end_utc": str(end_ts),
        "feature_min_game_datetime_utc": str(df["game_datetime_utc"].min()),
        "feature_max_game_datetime_utc": str(df["game_datetime_utc"].max()),
    }
    return candidates, diagnostics


def _read_latest_odds(conn, market_key: str) -> pd.DataFrame:
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
                s.outcome_price,
                s.outcome_point
            FROM odds_snapshots s
            JOIN odds_events e
              ON e.event_id = s.event_id
            WHERE lower(s.market_key) = ?
              AND s.outcome_price IS NOT NULL
            """,
            conn,
            params=[market_key.lower()],
        )
    except Exception as exc:
        print({"odds_warning": f"Could not read market {market_key}: {exc}"})
        return pd.DataFrame()

    if raw.empty:
        return raw

    raw["fetched_at_utc"] = pd.to_datetime(raw["fetched_at_utc"], utc=True, errors="coerce")
    raw["commence_time_utc"] = pd.to_datetime(raw["commence_time_utc"], utc=True, errors="coerce")
    raw["home_norm"] = raw["home_team_norm"].map(_normalize_team_name)
    raw["away_norm"] = raw["away_team_norm"].map(_normalize_team_name)
    raw["outcome_norm"] = raw["outcome_name_norm"].where(raw["outcome_name_norm"].notna(), raw["outcome_name"]).map(_normalize_team_name)
    raw = raw.sort_values(["fetched_at_utc", "event_id"])
    raw = raw.drop_duplicates(
        ["event_id", "bookmaker_key", "market_key", "outcome_norm", "outcome_point"],
        keep="last",
    )
    return raw


def _latest_totals_odds_events(conn) -> pd.DataFrame:
    raw = _read_latest_odds(conn, "totals")
    if raw.empty:
        return pd.DataFrame()

    side_text = raw["outcome_name"].astype(str).str.lower()
    raw["total_side"] = np.where(side_text.str.contains("over"), "over", np.where(side_text.str.contains("under"), "under", None))
    raw = raw[raw["total_side"].notna() & raw["outcome_point"].notna()].copy()
    if raw.empty:
        return pd.DataFrame()

    # Pick the consensus total point per event: the point with the most book outcomes.
    point_counts = (
        raw.groupby(["event_id", "outcome_point"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["event_id", "n", "outcome_point"], ascending=[True, False, True])
    )
    best_points = point_counts.drop_duplicates("event_id", keep="first")[["event_id", "outcome_point"]]
    raw = raw.merge(best_points, on=["event_id", "outcome_point"], how="inner")

    grouped = (
        raw.groupby(
            ["event_id", "commence_time_utc", "home_team", "away_team", "home_norm", "away_norm", "outcome_point", "total_side"],
            dropna=False,
        )["outcome_price"]
        .median()
        .reset_index()
    )
    pivot = grouped.pivot_table(
        index=["event_id", "commence_time_utc", "home_team", "away_team", "home_norm", "away_norm", "outcome_point"],
        columns="total_side",
        values="outcome_price",
        aggfunc="median",
    ).reset_index()
    pivot.columns.name = None
    if "over" not in pivot.columns:
        pivot["over"] = np.nan
    if "under" not in pivot.columns:
        pivot["under"] = np.nan
    pivot = pivot.rename(columns={"outcome_point": "market_total_line", "over": "over_price_median", "under": "under_price_median"})
    probs = pivot.apply(lambda row: _no_vig_probs_from_prices(row["over_price_median"], row["under_price_median"]), axis=1, result_type="expand")
    pivot["market_over_no_vig_prob"] = probs[0]
    pivot["market_under_no_vig_prob"] = probs[1]
    return pivot


def _latest_spread_odds_events(conn) -> pd.DataFrame:
    raw = _read_latest_odds(conn, "spreads")
    if raw.empty:
        return pd.DataFrame()

    raw["side"] = np.where(
        raw["outcome_norm"].eq(raw["home_norm"]),
        "home",
        np.where(raw["outcome_norm"].eq(raw["away_norm"]), "away", None),
    )
    raw = raw[raw["side"].notna() & raw["outcome_point"].notna()].copy()
    if raw.empty:
        return pd.DataFrame()

    grouped = (
        raw.groupby(
            ["event_id", "commence_time_utc", "home_team", "away_team", "home_norm", "away_norm", "side"],
            dropna=False,
        )
        .agg(
            outcome_price=("outcome_price", "median"),
            outcome_point=("outcome_point", "median"),
        )
        .reset_index()
    )
    price_pivot = grouped.pivot_table(
        index=["event_id", "commence_time_utc", "home_team", "away_team", "home_norm", "away_norm"],
        columns="side",
        values="outcome_price",
        aggfunc="median",
    ).reset_index()
    price_pivot.columns.name = None
    point_pivot = grouped.pivot_table(
        index=["event_id", "commence_time_utc", "home_team", "away_team", "home_norm", "away_norm"],
        columns="side",
        values="outcome_point",
        aggfunc="median",
    ).reset_index()
    point_pivot.columns.name = None

    out = price_pivot.merge(
        point_pivot,
        on=["event_id", "commence_time_utc", "home_team", "away_team", "home_norm", "away_norm"],
        suffixes=("_price", "_point"),
    )
    for col in ["home_price", "away_price", "home_point", "away_point"]:
        if col not in out.columns:
            out[col] = np.nan
    out = out.rename(
        columns={
            "home_price": "home_runline_price_median",
            "away_price": "away_runline_price_median",
            "home_point": "home_runline_point",
            "away_point": "away_runline_point",
        }
    )
    probs = out.apply(lambda row: _no_vig_probs_from_prices(row["home_runline_price_median"], row["away_runline_price_median"]), axis=1, result_type="expand")
    out["market_home_runline_no_vig_prob"] = probs[0]
    out["market_away_runline_no_vig_prob"] = probs[1]
    return out


def _attach_market_by_team_time(upcoming: pd.DataFrame, market: pd.DataFrame, prefix: str, max_start_diff_minutes: int) -> pd.DataFrame:
    out = upcoming.copy()
    if market.empty:
        print({f"{prefix}_attachment": {"events_available": 0, "matched_rows": 0}})
        return out

    market = market.dropna(subset=["commence_time_utc", "home_norm", "away_norm"]).copy()
    matched = 0
    event_ids: list[str] = []
    for idx, row in out.iterrows():
        game_time = pd.to_datetime(row.get("game_datetime_utc"), utc=True, errors="coerce")
        if pd.isna(game_time):
            continue
        home_norm = _normalize_team_name(row.get("home_team_name"))
        away_norm = _normalize_team_name(row.get("away_team_name"))
        candidates = market[market["home_norm"].eq(home_norm) & market["away_norm"].eq(away_norm)].copy()
        if candidates.empty:
            continue
        candidates["start_diff_minutes"] = (candidates["commence_time_utc"] - game_time).dt.total_seconds().abs() / 60.0
        candidates = candidates[candidates["start_diff_minutes"] <= max_start_diff_minutes]
        if candidates.empty:
            continue
        best = candidates.sort_values("start_diff_minutes").iloc[0]
        skip_cols = {"home_team", "away_team", "home_norm", "away_norm", "commence_time_utc"}
        for col, value in best.items():
            if col in skip_cols:
                continue
            if col == "event_id":
                out.loc[idx, f"{prefix}_odds_event_id"] = value
            elif col == "start_diff_minutes":
                out.loc[idx, f"{prefix}_odds_start_diff_minutes"] = value
            else:
                out.loc[idx, col] = value
        matched += 1
        event_ids.append(str(best.get("event_id")))

    print({
        f"{prefix}_attachment": {
            "events_available": int(len(market)),
            "matched_rows": int(matched),
            "unique_events_matched": int(len(set(event_ids))),
            "max_start_diff_minutes": int(max_start_diff_minutes),
        }
    })
    return out


def attach_totals_and_spreads(upcoming: pd.DataFrame, db_path: Path, max_start_diff_minutes: int) -> pd.DataFrame:
    try:
        with connect(db_path) as conn:
            totals = _latest_totals_odds_events(conn)
            spreads = _latest_spread_odds_events(conn)
    except Exception as exc:
        print({"odds_warning": f"Could not attach totals/spreads: {exc}"})
        return upcoming

    out = _attach_market_by_team_time(upcoming, totals, "totals", max_start_diff_minutes)
    out = _attach_market_by_team_time(out, spreads, "runline", max_start_diff_minutes)
    return out


def _choose_total_recommendation(row: pd.Series, min_runs_edge: float, min_prob_edge: float, min_ev: float) -> dict[str, Any]:
    if pd.isna(row.get("market_total_line")) or pd.isna(row.get("over_price_median")) or pd.isna(row.get("under_price_median")):
        return {"recommended_total_side": None, "recommended_total_price": np.nan, "total_edge_runs": np.nan, "total_expected_value_per_unit": np.nan, "total_no_bet_reason": "no_matched_total_odds"}

    candidates = []
    over_runs_edge = row.get("model_total_runs", np.nan) - row.get("market_total_line", np.nan)
    under_runs_edge = row.get("market_total_line", np.nan) - row.get("model_total_runs", np.nan)
    over_prob_edge = row.get("model_over_prob", np.nan) - row.get("market_over_no_vig_prob", np.nan)
    under_prob_edge = row.get("model_under_prob", np.nan) - row.get("market_under_no_vig_prob", np.nan)
    over_ev = _safe_ev(row.get("model_over_prob"), row.get("over_price_median"))
    under_ev = _safe_ev(row.get("model_under_prob"), row.get("under_price_median"))
    candidates.append({"side": "Over", "price": row.get("over_price_median"), "runs_edge": over_runs_edge, "prob_edge": over_prob_edge, "ev": over_ev})
    candidates.append({"side": "Under", "price": row.get("under_price_median"), "runs_edge": under_runs_edge, "prob_edge": under_prob_edge, "ev": under_ev})
    qualifying = [c for c in candidates if pd.notna(c["runs_edge"]) and c["runs_edge"] >= min_runs_edge and pd.notna(c["prob_edge"]) and c["prob_edge"] >= min_prob_edge and pd.notna(c["ev"]) and c["ev"] >= min_ev]
    if not qualifying:
        return {"recommended_total_side": None, "recommended_total_price": np.nan, "total_edge_runs": np.nan, "total_expected_value_per_unit": np.nan, "total_no_bet_reason": "below_total_thresholds"}
    best = max(qualifying, key=lambda c: (c["ev"], c["prob_edge"], c["runs_edge"]))
    return {"recommended_total_side": best["side"], "recommended_total_price": best["price"], "total_edge_runs": best["runs_edge"], "total_expected_value_per_unit": best["ev"], "total_no_bet_reason": "recommended"}


def _choose_runline_recommendation(row: pd.Series, min_margin_edge: float, min_prob_edge: float, min_ev: float) -> dict[str, Any]:
    if pd.isna(row.get("home_runline_point")) or pd.isna(row.get("away_runline_point")) or pd.isna(row.get("home_runline_price_median")) or pd.isna(row.get("away_runline_price_median")):
        return {"recommended_runline_side": None, "recommended_runline_price": np.nan, "runline_edge_margin": np.nan, "runline_expected_value_per_unit": np.nan, "runline_no_bet_reason": "no_matched_runline_odds"}

    pred_margin = row.get("model_home_margin", np.nan)
    home_point = row.get("home_runline_point", np.nan)
    away_point = row.get("away_runline_point", np.nan)
    # Positive margin edge means expected side margin exceeds its run-line requirement.
    home_margin_edge = pred_margin + home_point
    away_margin_edge = -pred_margin + away_point
    home_prob_edge = row.get("model_home_runline_cover_prob", np.nan) - row.get("market_home_runline_no_vig_prob", np.nan)
    away_prob_edge = row.get("model_away_runline_cover_prob", np.nan) - row.get("market_away_runline_no_vig_prob", np.nan)
    home_ev = _safe_ev(row.get("model_home_runline_cover_prob"), row.get("home_runline_price_median"))
    away_ev = _safe_ev(row.get("model_away_runline_cover_prob"), row.get("away_runline_price_median"))
    candidates = [
        {"side": f"{row.get('home_team_name')} {home_point:+.1f}", "price": row.get("home_runline_price_median"), "margin_edge": home_margin_edge, "prob_edge": home_prob_edge, "ev": home_ev},
        {"side": f"{row.get('away_team_name')} {away_point:+.1f}", "price": row.get("away_runline_price_median"), "margin_edge": away_margin_edge, "prob_edge": away_prob_edge, "ev": away_ev},
    ]
    qualifying = [c for c in candidates if pd.notna(c["margin_edge"]) and c["margin_edge"] >= min_margin_edge and pd.notna(c["prob_edge"]) and c["prob_edge"] >= min_prob_edge and pd.notna(c["ev"]) and c["ev"] >= min_ev]
    if not qualifying:
        return {"recommended_runline_side": None, "recommended_runline_price": np.nan, "runline_edge_margin": np.nan, "runline_expected_value_per_unit": np.nan, "runline_no_bet_reason": "below_runline_thresholds"}
    best = max(qualifying, key=lambda c: (c["ev"], c["prob_edge"], c["margin_edge"]))
    return {"recommended_runline_side": best["side"], "recommended_runline_price": best["price"], "runline_edge_margin": best["margin_edge"], "runline_expected_value_per_unit": best["ev"], "runline_no_bet_reason": "recommended"}


def main() -> None:
    configure_logging()
    args = parse_args()
    settings = get_settings()
    init_db(settings.odds_db_path)

    feature_path = Path(args.features) if args.features else settings.data_dir / "processed" / "mlb_game_features.parquet"
    totals_model_path = Path(args.totals_model) if args.totals_model else settings.model_dir / "mlb_total_runs_champion.joblib"
    margin_model_path = Path(args.margin_model) if args.margin_model else settings.model_dir / "mlb_home_margin_champion.joblib"
    output_path = Path(args.output) if args.output else settings.data_dir / "predictions" / "mlb_totals_runline_predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    totals_bundle = load_model_bundle(totals_model_path)
    margin_bundle = load_model_bundle(margin_model_path)
    totals_estimator = _resolve_estimator(totals_bundle)
    margin_estimator = _resolve_estimator(margin_bundle)
    totals_cols = _resolve_feature_cols(totals_bundle)
    margin_cols = _resolve_feature_cols(margin_bundle)
    total_sigma = _resolve_residual_std(totals_bundle, default=3.0)
    margin_sigma = _resolve_residual_std(margin_bundle, default=4.0)

    frame = pd.read_parquet(feature_path)
    scored_at_ts, start_ts, end_ts = _build_scoring_window(args)
    upcoming, diagnostics = build_scoring_candidates(
        frame,
        start_ts,
        end_ts,
        include_live=args.include_live,
        allow_scored_targets=args.allow_scored_targets,
    )
    print({"totals_runline_scoring_window_diagnostics": diagnostics})
    if upcoming.empty:
        raise SystemExit(f"No scoring candidates found. Diagnostics: {json.dumps(diagnostics, default=str)}")

    upcoming = attach_totals_and_spreads(upcoming, settings.odds_db_path, args.max_start_diff_minutes)
    missing_total_cols = [c for c in totals_cols if c not in upcoming.columns]
    missing_margin_cols = [c for c in margin_cols if c not in upcoming.columns]
    if missing_total_cols:
        raise SystemExit(f"Feature file missing totals model columns: {missing_total_cols[:50]}{'...' if len(missing_total_cols) > 50 else ''}")
    if missing_margin_cols:
        raise SystemExit(f"Feature file missing margin model columns: {missing_margin_cols[:50]}{'...' if len(missing_margin_cols) > 50 else ''}")

    upcoming["model_total_runs"] = totals_estimator.predict(upcoming[totals_cols])
    upcoming["model_home_margin"] = margin_estimator.predict(upcoming[margin_cols])
    upcoming["total_model_residual_std"] = total_sigma
    upcoming["margin_model_residual_std"] = margin_sigma

    # Totals model probabilities assuming approximately normal residuals.
    z_total = (upcoming["market_total_line"] - upcoming["model_total_runs"]) / total_sigma
    upcoming["model_under_prob"] = z_total.map(_norm_cdf)
    upcoming["model_over_prob"] = 1.0 - upcoming["model_under_prob"]
    upcoming["total_runs_edge"] = upcoming["model_total_runs"] - upcoming["market_total_line"]
    upcoming["over_prob_edge"] = upcoming["model_over_prob"] - upcoming.get("market_over_no_vig_prob", np.nan)
    upcoming["under_prob_edge"] = upcoming["model_under_prob"] - upcoming.get("market_under_no_vig_prob", np.nan)
    upcoming["over_expected_value_per_unit"] = [_safe_ev(p, price) for p, price in zip(upcoming["model_over_prob"], upcoming.get("over_price_median", pd.Series(np.nan, index=upcoming.index)))]
    upcoming["under_expected_value_per_unit"] = [_safe_ev(p, price) for p, price in zip(upcoming["model_under_prob"], upcoming.get("under_price_median", pd.Series(np.nan, index=upcoming.index)))]

    # Run-line probabilities. Home covers if home_margin + home_point > 0.
    z_home_cover = ((-upcoming.get("home_runline_point", np.nan)) - upcoming["model_home_margin"]) / margin_sigma
    upcoming["model_home_runline_cover_prob"] = 1.0 - z_home_cover.map(_norm_cdf)
    z_away_cover = (upcoming.get("away_runline_point", np.nan) - upcoming["model_home_margin"]) / margin_sigma
    upcoming["model_away_runline_cover_prob"] = z_away_cover.map(_norm_cdf)
    upcoming["home_runline_prob_edge"] = upcoming["model_home_runline_cover_prob"] - upcoming.get("market_home_runline_no_vig_prob", np.nan)
    upcoming["away_runline_prob_edge"] = upcoming["model_away_runline_cover_prob"] - upcoming.get("market_away_runline_no_vig_prob", np.nan)
    upcoming["home_runline_margin_edge"] = upcoming["model_home_margin"] + upcoming.get("home_runline_point", np.nan)
    upcoming["away_runline_margin_edge"] = -upcoming["model_home_margin"] + upcoming.get("away_runline_point", np.nan)
    upcoming["home_runline_expected_value_per_unit"] = [_safe_ev(p, price) for p, price in zip(upcoming["model_home_runline_cover_prob"], upcoming.get("home_runline_price_median", pd.Series(np.nan, index=upcoming.index)))]
    upcoming["away_runline_expected_value_per_unit"] = [_safe_ev(p, price) for p, price in zip(upcoming["model_away_runline_cover_prob"], upcoming.get("away_runline_price_median", pd.Series(np.nan, index=upcoming.index)))]

    total_recs = upcoming.apply(
        lambda row: _choose_total_recommendation(row, args.min_total_runs_edge, args.min_prob_edge, args.min_ev),
        axis=1,
        result_type="expand",
    )
    for col in total_recs.columns:
        upcoming[col] = total_recs[col]

    runline_recs = upcoming.apply(
        lambda row: _choose_runline_recommendation(row, args.min_margin_edge, args.min_prob_edge, args.min_ev),
        axis=1,
        result_type="expand",
    )
    for col in runline_recs.columns:
        upcoming[col] = runline_recs[col]

    upcoming["scored_at_utc"] = scored_at_ts.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    upcoming["totals_model_name"] = totals_bundle.get("model_name", totals_model_path.stem)
    upcoming["margin_model_name"] = margin_bundle.get("model_name", margin_model_path.stem)
    upcoming["has_total_market_odds"] = upcoming["market_total_line"].notna() & upcoming["over_price_median"].notna() & upcoming["under_price_median"].notna()
    upcoming["has_runline_market_odds"] = upcoming["home_runline_point"].notna() & upcoming["away_runline_point"].notna() & upcoming["home_runline_price_median"].notna() & upcoming["away_runline_price_median"].notna()

    if args.only_bettable:
        upcoming = upcoming[upcoming["has_total_market_odds"] | upcoming["has_runline_market_odds"]].copy()
        if upcoming.empty:
            raise SystemExit("No totals/run-line rows with matched market odds found.")

    keep = [
        "scored_at_utc", "game_pk", "official_date", "game_datetime_utc", "away_team_name", "home_team_name",
        "model_total_runs", "market_total_line", "total_runs_edge", "model_over_prob", "model_under_prob",
        "market_over_no_vig_prob", "market_under_no_vig_prob", "over_price_median", "under_price_median",
        "over_prob_edge", "under_prob_edge", "over_expected_value_per_unit", "under_expected_value_per_unit",
        "recommended_total_side", "recommended_total_price", "total_edge_runs", "total_expected_value_per_unit", "total_no_bet_reason",
        "model_home_margin", "home_runline_point", "away_runline_point", "home_runline_price_median", "away_runline_price_median",
        "model_home_runline_cover_prob", "model_away_runline_cover_prob", "market_home_runline_no_vig_prob", "market_away_runline_no_vig_prob",
        "home_runline_prob_edge", "away_runline_prob_edge", "home_runline_margin_edge", "away_runline_margin_edge",
        "home_runline_expected_value_per_unit", "away_runline_expected_value_per_unit",
        "recommended_runline_side", "recommended_runline_price", "runline_edge_margin", "runline_expected_value_per_unit", "runline_no_bet_reason",
        "has_total_market_odds", "has_runline_market_odds", "totals_odds_event_id", "runline_odds_event_id",
        "totals_model_name", "margin_model_name", "total_model_residual_std", "margin_model_residual_std",
    ]
    out = upcoming[[c for c in keep if c in upcoming.columns]].copy()
    out.to_csv(output_path, index=False)

    print({
        "totals_runline_predictions": int(len(out)),
        "total_market_rows": int(out.get("has_total_market_odds", pd.Series(False, index=out.index)).sum()),
        "runline_market_rows": int(out.get("has_runline_market_odds", pd.Series(False, index=out.index)).sum()),
        "total_recommendations": int(out.get("recommended_total_side", pd.Series(index=out.index, dtype=object)).notna().sum()),
        "runline_recommendations": int(out.get("recommended_runline_side", pd.Series(index=out.index, dtype=object)).notna().sum()),
        "output": str(output_path),
        "totals_model": str(totals_model_path),
        "margin_model": str(margin_model_path),
    })


if __name__ == "__main__":
    main()
