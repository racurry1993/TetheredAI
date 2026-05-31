from __future__ import annotations

import itertools
import json
import math
import platform
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, classification_report, confusion_matrix, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .betting_math import american_to_implied_prob, no_vig_two_way
from .feature_engineering import get_model_feature_columns

try:  # Optional dependency
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:  # Optional dependency
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None


RANDOM_STATE = 42
TARGET_COL = "target_home_win"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def clean_training_frame(frame: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    df = frame.copy()
    df["game_datetime_utc"] = pd.to_datetime(df["game_datetime_utc"], utc=True, errors="coerce")
    df = df[df[target_col].notna()].copy()
    df[target_col] = df[target_col].astype(int)
    return df.sort_values(["game_datetime_utc", "game_pk"]).reset_index(drop=True)


def time_holdout_split(
    frame: pd.DataFrame,
    holdout_days: int = 60,
    date_col: str = "game_datetime_utc",
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    df = frame.copy()
    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    df = df.sort_values([date_col, "game_pk"]).reset_index(drop=True)
    max_date = df[date_col].max()
    if pd.isna(max_date):
        raise ValueError("No valid dates in feature frame.")
    cutoff = max_date - pd.Timedelta(days=holdout_days)
    train = df[df[date_col] < cutoff].copy()
    test = df[df[date_col] >= cutoff].copy()
    if len(train) < 300 or len(test) < 100:
        idx = max(1, int(len(df) * 0.80))
        train = df.iloc[:idx].copy()
        test = df.iloc[idx:].copy()
        cutoff = test[date_col].min() if len(test) else max_date
    return train, test, str(cutoff)


def build_feature_sets(frame: pd.DataFrame, min_non_null_rate: float = 0.05) -> dict[str, list[str]]:
    """Create feature sets for staged model testing. No lineup features are used."""
    all_pure = get_model_feature_columns(frame, include_market=False, min_non_null_rate=min_non_null_rate)

    def cols_containing(*needles: str) -> list[str]:
        return [c for c in all_pure if any(n in c.lower() for n in needles)]

    baseline_tokens = (
        "win_last", "win_season", "runs_for", "runs_against", "run_diff",
        "rest_days", "games_played", "game_month", "game_dayofweek", "elo",
    )
    baseline = [c for c in all_pure if any(t in c.lower() for t in baseline_tokens)]
    starter = cols_containing("starter_")
    bullpen = cols_containing("bullpen_")
    box = cols_containing("team_box_")
    vs_hand = cols_containing("team_vs_hand")
    park = cols_containing("park_")

    feature_sets = {
        "baseline_team_form_elo": baseline,
        "starter_only": starter + [c for c in baseline if "elo" in c.lower() or c in {"game_month", "game_dayofweek"}],
        "baseline_starter_bullpen": sorted(set(baseline + starter + bullpen)),
        "enhanced_no_market": sorted(set(baseline + starter + bullpen + box + vs_hand + park)),
        "all_pure_numeric": all_pure,
    }
    return {k: v for k, v in feature_sets.items() if len(v) > 0}


def _pipeline(estimator: BaseEstimator, scale: bool = False) -> Pipeline:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def candidate_models(random_state: int = RANDOM_STATE, include_optional: bool = True) -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    candidates: dict[str, tuple[Pipeline, dict[str, list[Any]]]] = {}

    candidates["logistic"] = (
        _pipeline(LogisticRegression(max_iter=5000, solver="lbfgs"), scale=True),
        {
            "model__C": [0.03, 0.1, 0.3, 1.0, 3.0],
            "model__class_weight": [None, "balanced"],
        },
    )

    candidates["random_forest"] = (
        _pipeline(RandomForestClassifier(random_state=random_state, n_jobs=-1)),
        {
            "model__n_estimators": [300, 500],
            "model__max_depth": [3, 4, 6, 8, None],
            "model__min_samples_leaf": [20, 50, 75],
            "model__max_features": ["sqrt", 0.35, 0.5],
        },
    )

    candidates["extra_trees"] = (
        _pipeline(ExtraTreesClassifier(random_state=random_state, n_jobs=-1)),
        {
            "model__n_estimators": [300, 500],
            "model__max_depth": [3, 5, 8, None],
            "model__min_samples_leaf": [15, 30, 60],
            "model__max_features": ["sqrt", 0.35, 0.5],
        },
    )

    candidates["hist_gradient_boosting"] = (
        _pipeline(HistGradientBoostingClassifier(random_state=random_state, early_stopping=True)),
        {
            "model__learning_rate": [0.02, 0.04, 0.07],
            "model__max_iter": [150, 250, 350],
            "model__max_leaf_nodes": [7, 15, 31],
            "model__l2_regularization": [0.0, 0.1, 1.0],
            "model__min_samples_leaf": [20, 40, 80],
        },
    )

    candidates["svm_rbf"] = (
        _pipeline(SVC(kernel="rbf", probability=True, random_state=random_state), scale=True),
        {
            "model__C": [0.25, 0.5, 1.0, 2.0],
            "model__gamma": ["scale", 0.01, 0.03, 0.10],
            "model__class_weight": [None, "balanced"],
        },
    )

    if include_optional and XGBClassifier is not None:
        candidates["xgboost"] = (
            _pipeline(XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=random_state,
                n_jobs=-1,
            )),
            {
                "model__n_estimators": [200, 400, 700],
                "model__max_depth": [2, 3, 4],
                "model__learning_rate": [0.02, 0.04, 0.07],
                "model__subsample": [0.7, 0.9],
                "model__colsample_bytree": [0.7, 0.9],
                "model__min_child_weight": [5, 10, 20],
                "model__reg_alpha": [0.0, 0.1, 0.5],
                "model__reg_lambda": [1.0, 2.0, 5.0],
            },
        )

    if include_optional and LGBMClassifier is not None:
        candidates["lightgbm"] = (
            _pipeline(LGBMClassifier(
                objective="binary",
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            )),
            {
                "model__n_estimators": [200, 400, 700],
                "model__num_leaves": [7, 15, 31],
                "model__max_depth": [2, 3, 5, -1],
                "model__learning_rate": [0.02, 0.04, 0.07],
                "model__subsample": [0.7, 0.9, 1.0],
                "model__colsample_bytree": [0.7, 0.9, 1.0],
                "model__min_child_samples": [20, 40, 80],
                "model__reg_alpha": [0.0, 0.1, 0.5],
                "model__reg_lambda": [1.0, 2.0, 5.0],
            },
        )

    return candidates


def evaluate_probabilities(y_true: Iterable[int], probs: Iterable[float]) -> dict[str, float]:
    y = np.asarray(list(y_true), dtype=int)
    p = np.clip(np.asarray(list(probs), dtype=float), 1e-6, 1 - 1e-6)
    out = {
        "n": int(len(y)),
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "accuracy_50pct": float(accuracy_score(y, p >= 0.5)),
        "avg_pred": float(np.nanmean(p)),
        "actual_rate": float(np.nanmean(y)),
    }
    out["roc_auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    return out


def _grid_size(grid: dict[str, list[Any]]) -> int:
    size = 1
    for vals in grid.values():
        size *= len(vals)
    return int(size)


def fit_search(
    pipeline: Pipeline,
    grid: dict[str, list[Any]],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cv: TimeSeriesSplit,
    max_iter: int = 40,
    random_state: int = RANDOM_STATE,
) -> tuple[BaseEstimator, dict[str, Any], float]:
    size = _grid_size(grid)
    if size <= max_iter:
        search = GridSearchCV(pipeline, grid, scoring="neg_log_loss", cv=cv, n_jobs=-1, refit=True, verbose=0)
    else:
        search = RandomizedSearchCV(
            pipeline,
            grid,
            scoring="neg_log_loss",
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=0,
            n_iter=max_iter,
            random_state=random_state,
        )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, -float(search.best_score_)


def compare_models(
    frame: pd.DataFrame,
    feature_sets: Optional[dict[str, list[str]]] = None,
    model_names: Optional[list[str]] = None,
    target_col: str = TARGET_COL,
    holdout_days: int = 60,
    tune: bool = True,
    calibrate: bool = True,
    max_search_iter: int = 40,
    max_cv_splits: int = 4,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    df = clean_training_frame(frame, target_col=target_col)
    feature_sets = feature_sets or build_feature_sets(df)
    train, test, cutoff = time_holdout_split(df, holdout_days=holdout_days)
    cv_splits = min(max_cv_splits, max(2, len(train) // 250))
    cv = TimeSeriesSplit(n_splits=cv_splits)
    candidates = candidate_models(random_state=random_state)
    if model_names:
        candidates = {k: v for k, v in candidates.items() if k in model_names}

    rows = []
    fitted: dict[str, dict[str, Any]] = {}
    y_train = train[target_col].astype(int).to_numpy()
    y_test = test[target_col].astype(int).to_numpy()

    for fs_name, cols in feature_sets.items():
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue
        X_train = train[cols]
        X_test = test[cols]
        for model_name, (pipe, grid) in candidates.items():
            try:
                if tune:
                    estimator, params, cv_log_loss = fit_search(pipe, grid, X_train, y_train, cv, max_iter=max_search_iter, random_state=random_state)
                else:
                    estimator = clone(pipe).fit(X_train, y_train)
                    params = {}
                    cv_log_loss = float("nan")

                raw_probs = estimator.predict_proba(X_test)[:, 1]
                raw_metrics = evaluate_probabilities(y_test, raw_probs)
                final_estimator = estimator
                cal_method = None
                final_probs = raw_probs
                final_metrics = raw_metrics

                if calibrate and len(train) >= 600:
                    for method in (["sigmoid", "isotonic"] if len(train) >= 1200 else ["sigmoid"]):
                        try:
                            cal = CalibratedClassifierCV(estimator=clone(estimator), method=method, cv=cv)
                            cal.fit(X_train, y_train)
                            cal_probs = cal.predict_proba(X_test)[:, 1]
                            cal_metrics = evaluate_probabilities(y_test, cal_probs)
                            if cal_metrics["log_loss"] <= final_metrics["log_loss"] + 0.002:
                                final_estimator = cal
                                final_probs = cal_probs
                                final_metrics = cal_metrics
                                cal_method = method
                        except Exception:
                            continue

                key = f"{fs_name}__{model_name}"
                row = {
                    "candidate_key": key,
                    "feature_set": fs_name,
                    "model_name": model_name,
                    "feature_count": len(cols),
                    "cv_log_loss": cv_log_loss,
                    "calibration": cal_method or "none",
                    "best_params": params,
                    **{f"holdout_{k}": v for k, v in final_metrics.items()},
                    "raw_log_loss": raw_metrics["log_loss"],
                    "raw_brier": raw_metrics["brier"],
                    "raw_auc": raw_metrics["roc_auc"],
                }
                rows.append(row)
                fitted[key] = {
                    "estimator": final_estimator,
                    "raw_estimator": estimator,
                    "feature_cols": cols,
                    "feature_set": fs_name,
                    "model_name": model_name,
                    "best_params": params,
                    "metrics": final_metrics,
                    "calibration": cal_method,
                    "holdout_predictions": pd.DataFrame({
                        "game_pk": test["game_pk"].values,
                        "game_datetime_utc": test["game_datetime_utc"].values,
                        "home_team_name": test.get("home_team_name", pd.Series(index=test.index, dtype=object)).values,
                        "away_team_name": test.get("away_team_name", pd.Series(index=test.index, dtype=object)).values,
                        "target_home_win": y_test,
                        "model_home_win_prob": final_probs,
                    }),
                }
            except Exception as exc:
                rows.append({
                    "candidate_key": f"{fs_name}__{model_name}",
                    "feature_set": fs_name,
                    "model_name": model_name,
                    "feature_count": len(cols),
                    "error": repr(exc),
                })

    results = pd.DataFrame(rows)
    metric_cols = [c for c in ["holdout_log_loss", "holdout_brier", "holdout_roc_auc"] if c in results.columns]
    if metric_cols and "holdout_log_loss" in results:
        valid = results.dropna(subset=["holdout_log_loss"]).copy()
        best_key = None if valid.empty else valid.sort_values(["holdout_log_loss", "holdout_brier"], ascending=[True, True]).iloc[0]["candidate_key"]
    else:
        best_key = None

    return {
        "created_at_utc": utc_now_iso(),
        "cutoff": cutoff,
        "train": train,
        "test": test,
        "results": results,
        "fitted": fitted,
        "best_key": best_key,
    }


def probability_buckets(df: pd.DataFrame, prob_col: str, target_col: str = TARGET_COL, bins: Optional[list[float]] = None) -> pd.DataFrame:
    bins = bins or [0, .35, .40, .45, .50, .55, .60, .65, .70, 1.0]
    tmp = df.copy()
    tmp["prob_bucket"] = pd.cut(tmp[prob_col], bins=bins, include_lowest=True)
    return tmp.groupby("prob_bucket", observed=False).agg(
        games=(target_col, "size"),
        avg_pred_prob=(prob_col, "mean"),
        actual_home_win_rate=(target_col, "mean"),
    ).assign(calibration_error=lambda x: x["actual_home_win_rate"] - x["avg_pred_prob"])


def export_champion_model(
    estimator: BaseEstimator,
    feature_cols: list[str],
    metrics: dict[str, Any],
    model_family: str,
    feature_set_name: str,
    model_dir: Path | str,
    notes: str = "",
    target_col: str = TARGET_COL,
    champion_filename: str = "mlb_moneyline_champion.joblib",
) -> dict[str, Path]:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = model_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    created_at = utc_now_iso()
    run_id = f"mlb_moneyline_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    metadata = {
        "run_id": run_id,
        "model_name": "mlb_moneyline_champion",
        "model_family": model_family,
        "feature_set_name": feature_set_name,
        "target_col": target_col,
        "feature_count": len(feature_cols),
        "feature_cols": list(feature_cols),
        "metrics": metrics,
        "notes": notes,
        "created_at_utc": created_at,
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
    }
    bundle = {
        "estimator": estimator,
        "feature_cols": list(feature_cols),
        "target_col": target_col,
        "model_family": model_family,
        "feature_set_name": feature_set_name,
        "metrics": metrics,
        "metadata": metadata,
        "run_id": run_id,
        "created_at_utc": created_at,
    }
    champion_path = model_dir / champion_filename
    metadata_path = model_dir / "mlb_moneyline_champion_metadata.json"
    archive_path = archive_dir / f"{run_id}.joblib"
    archive_meta = archive_dir / f"{run_id}_metadata.json"

    joblib.dump(bundle, champion_path)
    joblib.dump(bundle, archive_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    archive_meta.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return {"champion_path": champion_path, "metadata_path": metadata_path, "archive_path": archive_path, "archive_metadata_path": archive_meta}


def load_model_bundle(path: Path | str) -> dict[str, Any]:
    bundle = joblib.load(path)
    if "estimator" not in bundle or "feature_cols" not in bundle:
        raise ValueError(f"Model bundle at {path} is missing estimator or feature_cols")
    return bundle


def _american_profit(price: float, stake: float = 1.0) -> float:
    price = float(price)
    if price > 0:
        return stake * price / 100.0
    return stake * 100.0 / abs(price)


def expected_value(prob: float, price: float, stake: float = 1.0) -> float:
    if pd.isna(prob) or pd.isna(price):
        return np.nan
    profit = _american_profit(price, stake)
    return float(prob * profit - (1.0 - prob) * stake)


def add_moneyline_edges(preds: pd.DataFrame) -> pd.DataFrame:
    df = preds.copy()
    if "model_away_win_prob" not in df and "model_home_win_prob" in df:
        df["model_away_win_prob"] = 1.0 - df["model_home_win_prob"]
    if "market_home_no_vig_prob" not in df and {"home_moneyline_median", "away_moneyline_median"}.issubset(df.columns):
        h_imp = df["home_moneyline_median"].map(lambda x: american_to_implied_prob(x) if pd.notna(x) else np.nan)
        a_imp = df["away_moneyline_median"].map(lambda x: american_to_implied_prob(x) if pd.notna(x) else np.nan)
        no_vig = [no_vig_two_way(h, a) if pd.notna(h) and pd.notna(a) else (np.nan, np.nan) for h, a in zip(h_imp, a_imp)]
        df["market_home_no_vig_prob"] = [x[0] for x in no_vig]
        df["market_away_no_vig_prob"] = [x[1] for x in no_vig]
    df["edge_home"] = df["model_home_win_prob"] - df.get("market_home_no_vig_prob", np.nan)
    df["edge_away"] = df["model_away_win_prob"] - df.get("market_away_no_vig_prob", np.nan)
    df["home_ev_per_unit"] = [expected_value(p, price) for p, price in zip(df["model_home_win_prob"], df.get("home_moneyline_median", pd.Series(np.nan, index=df.index)))]
    df["away_ev_per_unit"] = [expected_value(p, price) for p, price in zip(df["model_away_win_prob"], df.get("away_moneyline_median", pd.Series(np.nan, index=df.index)))]
    return df


def tune_moneyline_edge_thresholds(
    pred_frame: pd.DataFrame,
    min_edges: Iterable[float] = (0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05),
    min_evs: Iterable[float] = (0.0, 0.005, 0.01, 0.02),
    stake: float = 1.0,
) -> pd.DataFrame:
    """Tune bet thresholds using completed historical predictions with odds.

    Requires columns: target_home_win, model_home_win_prob, home_moneyline_median, away_moneyline_median.
    Returns one row per threshold pair. This is for EDA/offline tuning only.
    """
    df = add_moneyline_edges(pred_frame)
    needed = ["target_home_win", "home_moneyline_median", "away_moneyline_median"]
    df = df.dropna(subset=[c for c in needed if c in df.columns]).copy()
    if df.empty:
        return pd.DataFrame()

    rows = []
    for edge, min_ev in itertools.product(min_edges, min_evs):
        bets = []
        for _, r in df.iterrows():
            sides = []
            if pd.notna(r.get("edge_home")) and pd.notna(r.get("home_ev_per_unit")) and r["edge_home"] >= edge and r["home_ev_per_unit"] >= min_ev:
                sides.append(("home", r["edge_home"], r["home_ev_per_unit"], r["home_moneyline_median"]))
            if pd.notna(r.get("edge_away")) and pd.notna(r.get("away_ev_per_unit")) and r["edge_away"] >= edge and r["away_ev_per_unit"] >= min_ev:
                sides.append(("away", r["edge_away"], r["away_ev_per_unit"], r["away_moneyline_median"]))
            if not sides:
                continue
            side, ed, ev, price = max(sides, key=lambda x: (x[2], x[1]))
            won = (side == "home" and int(r["target_home_win"]) == 1) or (side == "away" and int(r["target_home_win"]) == 0)
            profit = _american_profit(price, stake) if won else -stake
            bets.append(profit)
        if bets:
            profits = np.asarray(bets, dtype=float)
            rows.append({
                "min_edge": edge,
                "min_ev": min_ev,
                "bets": int(len(profits)),
                "profit_units": float(profits.sum()),
                "roi_per_unit": float(profits.mean()),
                "win_rate": float((profits > 0).mean()),
                "max_drawdown_units": float(_max_drawdown(profits.cumsum())),
            })
        else:
            rows.append({"min_edge": edge, "min_ev": min_ev, "bets": 0, "profit_units": 0.0, "roi_per_unit": np.nan, "win_rate": np.nan, "max_drawdown_units": np.nan})
    return pd.DataFrame(rows).sort_values(["roi_per_unit", "bets"], ascending=[False, False]).reset_index(drop=True)


def _max_drawdown(cumulative: np.ndarray) -> float:
    if len(cumulative) == 0:
        return 0.0
    running_max = np.maximum.accumulate(np.r_[0.0, cumulative])
    series = np.r_[0.0, cumulative]
    drawdown = running_max - series
    return float(drawdown.max())
