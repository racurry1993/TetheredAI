from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .feature_engineering import get_model_feature_columns

LOGGER = logging.getLogger(__name__)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def clean_training_frame(frame: pd.DataFrame, target_col: str = "target_home_win") -> pd.DataFrame:
    df = frame.copy()
    df = df[df[target_col].notna()].copy()
    df[target_col] = df[target_col].astype(int)
    df = df.sort_values(["game_datetime_utc", "game_pk"]).reset_index(drop=True)
    return df


def time_holdout_split(
    frame: pd.DataFrame,
    holdout_days: int = 45,
    date_col: str = "game_datetime_utc",
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    df = frame.copy()
    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    max_date = df[date_col].max()
    if pd.isna(max_date):
        raise ValueError("No valid game dates found")
    cutoff = max_date - pd.Timedelta(days=holdout_days)
    train = df[df[date_col] < cutoff].copy()
    test = df[df[date_col] >= cutoff].copy()
    if len(train) < 100 or len(test) < 20:
        split_idx = int(len(df) * 0.8)
        train = df.iloc[:split_idx].copy()
        test = df.iloc[split_idx:].copy()
        cutoff_str = str(df.iloc[split_idx][date_col]) if len(test) else str(max_date)
    else:
        cutoff_str = cutoff.isoformat()
    return train, test, cutoff_str


def build_candidate_models(random_state: int = 42) -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    logistic = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, solver="lbfgs")),
    ])
    logistic_grid = {
        "model__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        "model__class_weight": [None, "balanced"],
    }

    hgb = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(random_state=random_state, early_stopping=True)),
    ])
    hgb_grid = {
        "model__learning_rate": [0.02, 0.05, 0.08],
        "model__max_iter": [100, 200, 350],
        "model__max_leaf_nodes": [7, 15, 31],
        "model__l2_regularization": [0.0, 0.01, 0.1, 1.0],
        "model__min_samples_leaf": [10, 20, 40],
    }

    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(random_state=random_state, n_jobs=-1)),
    ])
    rf_grid = {
        "model__n_estimators": [300, 600],
        "model__max_depth": [3, 5, 8, None],
        "model__min_samples_leaf": [5, 15, 30],
        "model__max_features": ["sqrt", 0.5],
    }
    return {"logistic": (logistic, logistic_grid), "hist_gradient_boosting": (hgb, hgb_grid), "random_forest": (rf, rf_grid)}


def evaluate_probabilities(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    y_true = np.asarray(y_true, dtype=int)
    metrics = {
        "n": int(len(y_true)),
        "log_loss": float(log_loss(y_true, probs)),
        "brier": float(brier_score_loss(y_true, probs)),
        "accuracy_50pct": float(accuracy_score(y_true, probs >= 0.5)),
    }
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probs))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def tune_moneyline_model(
    frame: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
    target_col: str = "target_home_win",
    holdout_days: int = 45,
    random_state: int = 42,
    tune: bool = True,
    calibrate: bool = True,
    max_cv_splits: int = 5,
) -> dict[str, Any]:
    df = clean_training_frame(frame, target_col=target_col)
    if len(df) < 100:
        raise ValueError(f"Need at least 100 completed games for tuning; found {len(df)}")
    feature_cols = feature_cols or get_model_feature_columns(df)
    if not feature_cols:
        raise ValueError("No numeric feature columns found")

    train, test, cutoff = time_holdout_split(df, holdout_days=holdout_days)
    X_train = train[feature_cols]
    y_train = train[target_col].astype(int).to_numpy()
    X_test = test[feature_cols]
    y_test = test[target_col].astype(int).to_numpy()

    candidates = build_candidate_models(random_state=random_state)
    cv_splits = min(max_cv_splits, max(2, len(train) // 150))
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    search_results = []
    best_name = None
    best_estimator: BaseEstimator | None = None
    best_score = np.inf

    for name, (pipeline, grid) in candidates.items():
        if tune:
            LOGGER.info("Tuning %s with %s CV splits", name, cv_splits)
            search = GridSearchCV(
                pipeline,
                grid,
                scoring="neg_log_loss",
                cv=tscv,
                n_jobs=-1,
                refit=True,
                verbose=0,
            )
            search.fit(X_train, y_train)
            estimator = search.best_estimator_
            cv_log_loss = -float(search.best_score_)
            params = search.best_params_
        else:
            estimator = pipeline.fit(X_train, y_train)
            cv_log_loss = float("nan")
            params = {}

        probs = estimator.predict_proba(X_test)[:, 1]
        metrics = evaluate_probabilities(y_test, probs)
        result = {"model_name": name, "cv_log_loss": cv_log_loss, "holdout_metrics": metrics, "best_params": params}
        search_results.append(result)
        LOGGER.info("%s holdout log_loss=%.4f brier=%.4f", name, metrics["log_loss"], metrics["brier"])
        if metrics["log_loss"] < best_score:
            best_score = metrics["log_loss"]
            best_name = name
            best_estimator = estimator

    assert best_estimator is not None and best_name is not None

    calibrated = None
    calibration_metrics = None
    final_estimator: BaseEstimator = best_estimator
    if calibrate and len(train) >= 300:
        # Calibrate the selected model using time-series cross-validation on the training period.
        LOGGER.info("Calibrating selected model: %s", best_name)
        method = "isotonic" if len(train) >= 1000 else "sigmoid"
        calibrated = CalibratedClassifierCV(best_estimator, method=method, cv=tscv)
        calibrated.fit(X_train, y_train)
        cal_probs = calibrated.predict_proba(X_test)[:, 1]
        calibration_metrics = evaluate_probabilities(y_test, cal_probs)
        if calibration_metrics["log_loss"] <= best_score + 0.005:
            final_estimator = calibrated
            best_score = calibration_metrics["log_loss"]

    test_probs = final_estimator.predict_proba(X_test)[:, 1]
    final_metrics = evaluate_probabilities(y_test, test_probs)

    run_id = f"mlb_moneyline_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    return {
        "run_id": run_id,
        "created_at_utc": utc_now_iso(),
        "model_name": best_name,
        "target": target_col,
        "estimator": final_estimator,
        "feature_cols": feature_cols,
        "cutoff": cutoff,
        "train_index": train.index.tolist(),
        "test_index": test.index.tolist(),
        "train_start_date": str(pd.to_datetime(train["game_datetime_utc"]).min()),
        "train_end_date": str(pd.to_datetime(train["game_datetime_utc"]).max()),
        "test_start_date": str(pd.to_datetime(test["game_datetime_utc"]).min()),
        "test_end_date": str(pd.to_datetime(test["game_datetime_utc"]).max()),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "search_results": search_results,
        "calibration_metrics": calibration_metrics,
        "final_metrics": final_metrics,
        "holdout_predictions": pd.DataFrame({
            "game_pk": test["game_pk"].values,
            "game_datetime_utc": test["game_datetime_utc"].values,
            "home_team_name": test["home_team_name"].values,
            "away_team_name": test["away_team_name"].values,
            "target_home_win": y_test,
            "model_home_win_prob": test_probs,
        }),
    }


def save_model_bundle(result: dict[str, Any], model_dir: Path | str) -> dict[str, Path]:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    run_id = result["run_id"]
    model_path = model_dir / f"{run_id}.joblib"
    meta_path = model_dir / f"{run_id}_metadata.json"
    preds_path = model_dir / f"{run_id}_holdout_predictions.csv"

    bundle = {
        "estimator": result["estimator"],
        "feature_cols": result["feature_cols"],
        "target": result["target"],
        "model_name": result["model_name"],
        "created_at_utc": result["created_at_utc"],
        "run_id": run_id,
    }
    joblib.dump(bundle, model_path)

    serializable = {k: v for k, v in result.items() if k not in {"estimator", "holdout_predictions"}}
    meta_path.write_text(json.dumps(serializable, indent=2, default=str), encoding="utf-8")
    result["holdout_predictions"].to_csv(preds_path, index=False)
    return {"model_path": model_path, "metadata_path": meta_path, "holdout_predictions_path": preds_path}


def load_model_bundle(path: Path | str) -> dict[str, Any]:
    return joblib.load(path)


def latest_model_path(model_dir: Path | str) -> Path:
    paths = sorted(Path(model_dir).glob("mlb_moneyline_*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not paths:
        raise FileNotFoundError(f"No model artifacts found in {model_dir}")
    return paths[0]
