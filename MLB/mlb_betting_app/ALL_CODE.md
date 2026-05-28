

# ==== .env.example ====

# Never commit your real API key.
ODDS_API_KEY=
ODDS_DB_PATH=data/odds.db
DATA_DIR=data
MODEL_DIR=models
ODDS_SPORT_KEY=baseball_mlb
ODDS_REGIONS=us
ODDS_MARKETS=h2h,spreads,totals
ODDS_FORMAT=american


# ==== .github/workflows/nightly_mlb.yml ====

name: Nightly MLB Refresh

on:
  workflow_dispatch:
  schedule:
    # GitHub cron is UTC. This is 5:20 AM Central during daylight saving time.
    - cron: "20 10 * * *"

jobs:
  refresh:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    env:
      ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}
      ODDS_DB_PATH: data/odds.db
      DATA_DIR: data
      MODEL_DIR: models
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run daily refresh
        run: |
          python scripts/00_init_db.py
          python scripts/01_fetch_odds.py --sport baseball_mlb --regions us --markets h2h,spreads,totals
          python scripts/02_fetch_mlb_games.py --days-back 730 --days-forward 14
          python scripts/03_build_features.py
          python scripts/04_train_moneyline_model.py --tune --calibrate --min-rows 100
          python scripts/05_score_today.py --days-forward 3

      - name: Commit updated artifacts
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add data/odds.db data/processed data/predictions models || true
          git commit -m "Nightly MLB refresh" || echo "No changes to commit"
          git push || true


# ==== .gitignore ====

.env
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
.DS_Store
# Keep or remove these depending on your repo strategy.
# data/odds.db
# data/raw/
# data/processed/
# models/


# ==== Makefile ====

PYTHON ?= python

init:
	$(PYTHON) scripts/00_init_db.py

fetch-odds:
	$(PYTHON) scripts/01_fetch_odds.py --sport baseball_mlb --regions us --markets h2h,spreads,totals

fetch-games:
	$(PYTHON) scripts/02_fetch_mlb_games.py --days-back 365 --days-forward 14

features:
	$(PYTHON) scripts/03_build_features.py

train:
	$(PYTHON) scripts/04_train_moneyline_model.py --tune --calibrate

score:
	$(PYTHON) scripts/05_score_today.py --days-forward 3

qa:
	$(PYTHON) scripts/qa_smoke_test.py


# ==== README.md ====

# MLB Betting Pipeline Starter

This project is intentionally split into two layers:

1. **Production-style scripts** under `scripts/` and reusable modules under `src/mlb_betting/`.
2. **Development / QA notebook** under `notebooks/01_mlb_moneyline_dev_qa.ipynb` for EDA, feature engineering checks, model tuning, calibration, and betting backtests.

The first production target is MLB moneyline. The data model is designed to later add run line, totals, NFL, and golf.

## Safety note about API keys

Do not hard-code your odds API key. Put it in `.env` locally and in GitHub Actions secrets for automation.

```bash
cp .env.example .env
# Edit .env and set ODDS_API_KEY=...
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

## Initialize the database

```bash
python scripts/00_init_db.py
```

This creates `data/odds.db` by default. The same SQLite file stores odds snapshots, MLB schedules/results, predictions, and model run metadata.

## Fetch odds

```bash
python scripts/01_fetch_odds.py --sport baseball_mlb --regions us --markets h2h,spreads,totals
```

This stores every bookmaker/market/outcome snapshot. Re-running it throughout the day creates a line-movement history.

## Fetch MLB schedule and results

```bash
python scripts/02_fetch_mlb_games.py --days-back 365 --days-forward 14
```

This uses MLB Stats API schedule data. It stores completed games and future scheduled games in `mlb_games`.

## Build features

```bash
python scripts/03_build_features.py
```

This creates `data/processed/mlb_game_features.parquet`. Features are generated from pregame information only. Rolling team features are shifted so the current game result is not included.

## Train the moneyline model

```bash
python scripts/04_train_moneyline_model.py --tune --calibrate
```

This script performs time-series hyperparameter tuning and saves a model bundle under `models/`.

## Score upcoming games

```bash
python scripts/05_score_today.py --days-forward 3
```

This writes predictions to `data/predictions/mlb_moneyline_predictions.csv` and also stores them in the SQLite database.

## Grade predictions

After games finish and `mlb_games` has been refreshed:

```bash
python scripts/06_grade_bets.py --predictions data/predictions/mlb_moneyline_predictions.csv
```

## Development / QA notebook

Open:

```bash
jupyter lab notebooks/01_mlb_moneyline_dev_qa.ipynb
```

The notebook includes:

- Raw data checks
- Missingness analysis
- Target distribution
- Rolling feature validation
- Odds feature QA
- Time-based train/test split
- Baseline market comparison
- Hyperparameter tuning
- Calibration curves
- Feature importance
- Simple betting simulation

## Important modeling notes

- Do not use random train/test splits for betting models.
- Do not use closing odds as a pregame model feature unless your prediction time is after the closing line is known.
- Accuracy alone is not enough. Prioritize log loss, Brier score, calibration, closing-line value, and ROI by edge bucket.
- SQLite is fine for development. For production, GitHub Actions can commit the DB to a private repo, but a small hosted Postgres database is cleaner once this grows.


# ==== requirements.txt ====

requests>=2.31.0
pandas>=2.2.0
numpy>=1.26.0
python-dotenv>=1.0.1
pyarrow>=15.0.0
scikit-learn>=1.4.0
joblib>=1.3.2
matplotlib>=3.8.0
statsmodels>=0.14.0
jupyterlab>=4.0.0
ipykernel>=6.29.0
tqdm>=4.66.0


# ==== scripts/00_init_db.py ====

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlb_betting.config import get_settings
from mlb_betting.db import init_db
from mlb_betting.logging_utils import configure_logging


def main() -> None:
    configure_logging()
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    init_db(settings.odds_db_path)
    print(f"Initialized database: {settings.odds_db_path}")


if __name__ == "__main__":
    main()


# ==== scripts/01_fetch_odds.py ====

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlb_betting.config import get_settings
from mlb_betting.db import connect, init_db
from mlb_betting.logging_utils import configure_logging
from mlb_betting.odds_api import OddsApiClient, fetch_and_store_odds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch odds from The Odds API and store normalized snapshots.")
    parser.add_argument("--sport", default=None, help="Sport key, e.g. baseball_mlb")
    parser.add_argument("--regions", default=None, help="Comma-separated regions, e.g. us")
    parser.add_argument("--markets", default=None, help="Comma-separated markets, e.g. h2h,spreads,totals")
    parser.add_argument("--bookmakers", default=None, help="Optional comma-separated bookmaker keys. Overrides regions.")
    parser.add_argument("--odds-format", default=None, choices=["american", "decimal"])
    parser.add_argument("--commence-time-from", default=None)
    parser.add_argument("--commence-time-to", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    settings = get_settings()
    if not settings.odds_api_key:
        raise SystemExit("ODDS_API_KEY is missing. Put it in .env or your environment.")
    init_db(settings.odds_db_path)
    client = OddsApiClient(settings.odds_api_key)
    with connect(settings.odds_db_path) as conn:
        result = fetch_and_store_odds(
            conn=conn,
            client=client,
            sport=args.sport or settings.odds_sport_key,
            regions=args.regions or settings.odds_regions,
            markets=args.markets or settings.odds_markets,
            odds_format=args.odds_format or settings.odds_format,
            bookmakers=args.bookmakers,
            commence_time_from=args.commence_time_from,
            commence_time_to=args.commence_time_to,
            dry_run=args.dry_run,
        )
    print(result)


if __name__ == "__main__":
    main()


# ==== scripts/02_fetch_mlb_games.py ====

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlb_betting.config import get_settings
from mlb_betting.db import connect, init_db
from mlb_betting.logging_utils import configure_logging
from mlb_betting.mlb_stats_api import MlbStatsClient, fetch_schedule_to_db


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch MLB schedule/results from MLB Stats API.")
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--days-back", type=int, default=365)
    parser.add_argument("--days-forward", type=int, default=14)
    parser.add_argument("--game-type", default=None, help="Optional MLB game type filter, e.g. R")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    settings = get_settings()
    init_db(settings.odds_db_path)
    today = datetime.now(timezone.utc).date()
    start_date = args.start_date or (today - timedelta(days=args.days_back)).isoformat()
    end_date = args.end_date or (today + timedelta(days=args.days_forward)).isoformat()
    client = MlbStatsClient()
    with connect(settings.odds_db_path) as conn:
        result = fetch_schedule_to_db(conn, client, start_date, end_date, game_type=args.game_type)
    print(result)


if __name__ == "__main__":
    main()


# ==== scripts/03_build_features.py ====

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlb_betting.config import get_settings
from mlb_betting.db import connect, init_db
from mlb_betting.feature_engineering import (
    build_game_feature_frame,
    load_latest_odds_consensus,
    load_mlb_games,
    save_features,
)
from mlb_betting.logging_utils import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MLB game-level model features.")
    parser.add_argument("--output", default=None, help="Output parquet path")
    parser.add_argument("--completed-only", action="store_true", help="Exclude future/scheduled games")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    settings = get_settings()
    init_db(settings.odds_db_path)
    output = Path(args.output) if args.output else settings.data_dir / "processed" / "mlb_game_features.parquet"
    with connect(settings.odds_db_path) as conn:
        games = load_mlb_games(conn)
        odds = load_latest_odds_consensus(conn)
    features = build_game_feature_frame(games, odds_consensus=odds, include_future=not args.completed_only)
    save_features(features, output)
    print({"rows": len(features), "columns": len(features.columns), "output": str(output)})


if __name__ == "__main__":
    main()


# ==== scripts/04_train_moneyline_model.py ====

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


# ==== scripts/05_score_today.py ====

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


# ==== scripts/06_grade_bets.py ====

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from mlb_betting.betting_math import profit_if_win
from mlb_betting.config import get_settings
from mlb_betting.db import connect, read_sql
from mlb_betting.logging_utils import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grade model recommendations after games finish.")
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--stake", type=float, default=1.0)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    settings = get_settings()
    pred_path = Path(args.predictions) if args.predictions else settings.data_dir / "predictions" / "mlb_moneyline_predictions.csv"
    if not pred_path.exists():
        raise SystemExit(f"Predictions file not found: {pred_path}")
    preds = pd.read_csv(pred_path)
    with connect(settings.odds_db_path) as conn:
        games = read_sql(conn, "SELECT game_pk, home_team_name, away_team_name, home_score, away_score, target_home_win FROM mlb_games WHERE target_home_win IS NOT NULL")
    graded = preds.merge(games, on="game_pk", how="left", suffixes=("", "_actual"))
    graded = graded[graded["recommended_side"].notna()].copy()
    if graded.empty:
        print("No recommendations to grade.")
        return
    graded["bet_won"] = np.where(
        graded["recommended_side"] == graded["home_team_name"],
        graded["target_home_win"] == 1,
        graded["target_home_win"] == 0,
    )
    graded["stake"] = args.stake
    graded["profit"] = np.where(
        graded["bet_won"],
        graded["recommended_price"].apply(lambda x: profit_if_win(x, args.stake)),
        -args.stake,
    )
    summary = {
        "graded_bets": int(len(graded)),
        "wins": int(graded["bet_won"].sum()),
        "losses": int((~graded["bet_won"]).sum()),
        "win_rate": float(graded["bet_won"].mean()),
        "profit_units": float(graded["profit"].sum()),
        "roi": float(graded["profit"].sum() / graded["stake"].sum()),
    }
    output = Path(args.output) if args.output else settings.data_dir / "predictions" / "mlb_moneyline_graded.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    graded.to_csv(output, index=False)
    print({"summary": summary, "output": str(output)})


if __name__ == "__main__":
    main()


# ==== scripts/qa_smoke_test.py ====

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlb_betting.config import get_settings
from mlb_betting.db import connect, init_db, read_sql
from mlb_betting.logging_utils import configure_logging


def main() -> None:
    configure_logging()
    settings = get_settings()
    init_db(settings.odds_db_path)
    with connect(settings.odds_db_path) as conn:
        checks = {
            "odds_events": read_sql(conn, "SELECT COUNT(*) AS n FROM odds_events")["n"].iloc[0],
            "odds_snapshots": read_sql(conn, "SELECT COUNT(*) AS n FROM odds_snapshots")["n"].iloc[0],
            "mlb_games": read_sql(conn, "SELECT COUNT(*) AS n FROM mlb_games")["n"].iloc[0],
            "completed_mlb_games": read_sql(conn, "SELECT COUNT(*) AS n FROM mlb_games WHERE target_home_win IS NOT NULL")["n"].iloc[0],
        }
    print(checks)


if __name__ == "__main__":
    main()


# ==== scripts/run_daily_refresh.py ====

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COMMANDS = [
    [sys.executable, "scripts/00_init_db.py"],
    [sys.executable, "scripts/01_fetch_odds.py", "--sport", "baseball_mlb", "--regions", "us", "--markets", "h2h,spreads,totals"],
    [sys.executable, "scripts/02_fetch_mlb_games.py", "--days-back", "730", "--days-forward", "14"],
    [sys.executable, "scripts/03_build_features.py"],
    [sys.executable, "scripts/04_train_moneyline_model.py", "--tune", "--calibrate", "--min-rows", "100"],
    [sys.executable, "scripts/05_score_today.py", "--days-forward", "3"],
]


def main() -> None:
    for cmd in COMMANDS:
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()


# ==== src/mlb_betting/__init__.py ====

__all__ = [
    "config",
    "db",
    "odds_api",
    "mlb_stats_api",
    "feature_engineering",
    "modeling",
    "betting_math",
]


# ==== src/mlb_betting/betting_math.py ====

from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np


def american_to_decimal(price: float) -> float:
    price = float(price)
    if price > 0:
        return 1.0 + price / 100.0
    if price < 0:
        return 1.0 + 100.0 / abs(price)
    raise ValueError("American odds price cannot be zero")


def american_to_implied_prob(price: float) -> float:
    price = float(price)
    if price > 0:
        return 100.0 / (price + 100.0)
    if price < 0:
        return abs(price) / (abs(price) + 100.0)
    raise ValueError("American odds price cannot be zero")


def implied_prob_to_american(prob: float) -> float:
    prob = float(prob)
    if not 0 < prob < 1:
        raise ValueError("Probability must be between 0 and 1")
    if prob >= 0.5:
        return -100.0 * prob / (1.0 - prob)
    return 100.0 * (1.0 - prob) / prob


def no_vig_two_way(prob_a: float, prob_b: float) -> tuple[float, float]:
    total = float(prob_a) + float(prob_b)
    if total <= 0:
        return math.nan, math.nan
    return float(prob_a) / total, float(prob_b) / total


def profit_if_win(price: float, stake: float = 1.0) -> float:
    decimal = american_to_decimal(price)
    return stake * (decimal - 1.0)


def expected_value_per_unit(model_prob: float, american_price: float) -> float:
    """Expected net profit per 1 unit staked."""
    p = float(model_prob)
    win_profit = profit_if_win(american_price, stake=1.0)
    return p * win_profit - (1.0 - p) * 1.0


def kelly_fraction(model_prob: float, american_price: float, fraction: float = 0.25) -> float:
    """Fractional Kelly stake. Returns 0 for negative edge."""
    p = float(model_prob)
    b = american_to_decimal(american_price) - 1.0
    q = 1.0 - p
    full = (b * p - q) / b
    return max(0.0, full * fraction)


def safe_mean(values: Iterable[float]) -> Optional[float]:
    arr = np.array(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return None
    return float(np.mean(arr))


# ==== src/mlb_betting/config.py ====

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    model_dir: Path
    odds_db_path: Path
    odds_api_key: Optional[str]
    odds_sport_key: str
    odds_regions: str
    odds_markets: str
    odds_format: str


def find_project_root(start: Optional[Path] = None) -> Path:
    start = start or Path.cwd()
    for path in [start, *start.parents]:
        if (path / "requirements.txt").exists() and (path / "src").exists():
            return path
    return start


def get_settings() -> Settings:
    root = find_project_root()
    env_path = root / ".env"
    if load_dotenv is not None and env_path.exists():
        load_dotenv(env_path)

    data_dir = Path(os.getenv("DATA_DIR", str(root / "data")))
    if not data_dir.is_absolute():
        data_dir = root / data_dir

    model_dir = Path(os.getenv("MODEL_DIR", str(root / "models")))
    if not model_dir.is_absolute():
        model_dir = root / model_dir

    odds_db_path = Path(os.getenv("ODDS_DB_PATH", str(data_dir / "odds.db")))
    if not odds_db_path.is_absolute():
        odds_db_path = root / odds_db_path

    return Settings(
        project_root=root,
        data_dir=data_dir,
        model_dir=model_dir,
        odds_db_path=odds_db_path,
        odds_api_key=os.getenv("ODDS_API_KEY"),
        odds_sport_key=os.getenv("ODDS_SPORT_KEY", "baseball_mlb"),
        odds_regions=os.getenv("ODDS_REGIONS", "us"),
        odds_markets=os.getenv("ODDS_MARKETS", "h2h,spreads,totals"),
        odds_format=os.getenv("ODDS_FORMAT", "american"),
    )


# ==== src/mlb_betting/data_validation.py ====

from __future__ import annotations

import pandas as pd


def summarize_missingness(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["column", "missing_count", "missing_pct"])
    out = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_pct": df.isna().mean().values,
    })
    return out.sort_values("missing_pct", ascending=False).reset_index(drop=True)


def validate_no_obvious_leakage(feature_cols: list[str]) -> None:
    blocked_tokens = ["score", "target", "margin", "total_runs", "detailed_state", "abstract_state", "status_code"]
    bad = [c for c in feature_cols if any(token in c.lower() for token in blocked_tokens)]
    if bad:
        raise ValueError(f"Potential leakage columns in feature list: {bad}")


def basic_game_checks(games: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(games)),
        "completed_games": int(games["target_home_win"].notna().sum()) if "target_home_win" in games else 0,
        "min_date": str(games["official_date"].min()) if "official_date" in games and not games.empty else None,
        "max_date": str(games["official_date"].max()) if "official_date" in games and not games.empty else None,
        "duplicate_game_pk": int(games["game_pk"].duplicated().sum()) if "game_pk" in games else None,
    }


# ==== src/mlb_betting/db.py ====

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, Mapping, Optional

import pandas as pd

from .team_mapping import normalize_team_name

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS api_usage_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    fetched_at_utc TEXT NOT NULL,
    request_url TEXT,
    status_code INTEGER,
    requests_remaining INTEGER,
    requests_used INTEGER,
    requests_last INTEGER,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS raw_api_payloads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    fetched_at_utc TEXT NOT NULL,
    params_json TEXT,
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS odds_events (
    event_id TEXT PRIMARY KEY,
    sport_key TEXT NOT NULL,
    sport_title TEXT,
    commence_time_utc TEXT,
    home_team TEXT,
    away_team TEXT,
    home_team_norm TEXT,
    away_team_norm TEXT,
    last_seen_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS odds_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    fetched_at_utc TEXT NOT NULL,
    event_id TEXT NOT NULL,
    sport_key TEXT NOT NULL,
    commence_time_utc TEXT,
    home_team TEXT,
    away_team TEXT,
    bookmaker_key TEXT NOT NULL,
    bookmaker_title TEXT,
    bookmaker_last_update_utc TEXT,
    market_key TEXT NOT NULL,
    outcome_name TEXT NOT NULL,
    outcome_name_norm TEXT,
    outcome_price REAL,
    outcome_point REAL,
    outcome_point_key TEXT NOT NULL,
    outcome_description TEXT,
    outcome_link TEXT,
    outcome_sid TEXT,
    FOREIGN KEY(event_id) REFERENCES odds_events(event_id)
);

CREATE INDEX IF NOT EXISTS idx_odds_snapshots_event_market
    ON odds_snapshots(event_id, market_key, fetched_at_utc);
CREATE INDEX IF NOT EXISTS idx_odds_snapshots_bookmaker
    ON odds_snapshots(bookmaker_key, market_key, fetched_at_utc);
CREATE INDEX IF NOT EXISTS idx_odds_events_teams_date
    ON odds_events(home_team_norm, away_team_norm, commence_time_utc);

CREATE TABLE IF NOT EXISTS mlb_games (
    game_pk INTEGER PRIMARY KEY,
    game_guid TEXT,
    season INTEGER,
    game_type TEXT,
    game_date TEXT,
    official_date TEXT,
    game_datetime_utc TEXT,
    status_code TEXT,
    detailed_state TEXT,
    abstract_state TEXT,
    venue_id INTEGER,
    venue_name TEXT,
    home_team_id INTEGER,
    home_team_name TEXT,
    home_team_norm TEXT,
    away_team_id INTEGER,
    away_team_name TEXT,
    away_team_norm TEXT,
    home_score INTEGER,
    away_score INTEGER,
    target_home_win INTEGER,
    home_margin INTEGER,
    total_runs INTEGER,
    probable_home_pitcher_id INTEGER,
    probable_home_pitcher_name TEXT,
    probable_away_pitcher_id INTEGER,
    probable_away_pitcher_name TEXT,
    last_updated_utc TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_mlb_games_date ON mlb_games(official_date);
CREATE INDEX IF NOT EXISTS idx_mlb_games_teams ON mlb_games(home_team_norm, away_team_norm, official_date);

CREATE TABLE IF NOT EXISTS model_runs (
    run_id TEXT PRIMARY KEY,
    created_at_utc TEXT NOT NULL,
    model_name TEXT NOT NULL,
    target TEXT NOT NULL,
    train_start_date TEXT,
    train_end_date TEXT,
    test_start_date TEXT,
    test_end_date TEXT,
    n_train INTEGER,
    n_test INTEGER,
    metrics_json TEXT,
    params_json TEXT,
    feature_columns_json TEXT,
    artifact_path TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    scored_at_utc TEXT NOT NULL,
    game_pk INTEGER,
    official_date TEXT,
    game_datetime_utc TEXT,
    home_team_name TEXT,
    away_team_name TEXT,
    model_home_win_prob REAL,
    market_home_no_vig_prob REAL,
    home_moneyline_median REAL,
    away_moneyline_median REAL,
    recommended_side TEXT,
    recommended_price REAL,
    edge REAL,
    expected_value_per_unit REAL,
    feature_snapshot_json TEXT,
    FOREIGN KEY(game_pk) REFERENCES mlb_games(game_pk)
);
"""


def connect(db_path: Path | str) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | str) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.commit()


def read_sql(conn: sqlite3.Connection, query: str, params: Optional[Mapping] = None) -> pd.DataFrame:
    return pd.read_sql_query(query, conn, params=params or {})


def upsert_odds_event(conn: sqlite3.Connection, event: Mapping, fetched_at_utc: str) -> None:
    conn.execute(
        """
        INSERT INTO odds_events (
            event_id, sport_key, sport_title, commence_time_utc,
            home_team, away_team, home_team_norm, away_team_norm, last_seen_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(event_id) DO UPDATE SET
            sport_key=excluded.sport_key,
            sport_title=excluded.sport_title,
            commence_time_utc=excluded.commence_time_utc,
            home_team=excluded.home_team,
            away_team=excluded.away_team,
            home_team_norm=excluded.home_team_norm,
            away_team_norm=excluded.away_team_norm,
            last_seen_utc=excluded.last_seen_utc
        """,
        (
            event.get("id"),
            event.get("sport_key"),
            event.get("sport_title"),
            event.get("commence_time"),
            event.get("home_team"),
            event.get("away_team"),
            normalize_team_name(event.get("home_team")),
            normalize_team_name(event.get("away_team")),
            fetched_at_utc,
        ),
    )


def insert_api_usage(
    conn: sqlite3.Connection,
    source: str,
    endpoint: str,
    fetched_at_utc: str,
    request_url: Optional[str],
    status_code: Optional[int],
    headers: Optional[Mapping] = None,
    error_message: Optional[str] = None,
) -> None:
    headers = headers or {}
    def get_int(name: str) -> Optional[int]:
        value = headers.get(name) or headers.get(name.lower())
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    conn.execute(
        """
        INSERT INTO api_usage_log (
            source, endpoint, fetched_at_utc, request_url, status_code,
            requests_remaining, requests_used, requests_last, error_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            source,
            endpoint,
            fetched_at_utc,
            request_url,
            status_code,
            get_int("x-requests-remaining"),
            get_int("x-requests-used"),
            get_int("x-requests-last"),
            error_message,
        ),
    )


def insert_raw_payload(
    conn: sqlite3.Connection,
    source: str,
    endpoint: str,
    fetched_at_utc: str,
    params: Mapping,
    payload: object,
) -> None:
    conn.execute(
        """
        INSERT INTO raw_api_payloads (source, endpoint, fetched_at_utc, params_json, payload_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (source, endpoint, fetched_at_utc, json.dumps(params, sort_keys=True), json.dumps(payload)),
    )


def insert_prediction_rows(conn: sqlite3.Connection, rows: Iterable[Mapping]) -> int:
    count = 0
    for row in rows:
        conn.execute(
            """
            INSERT INTO predictions (
                run_id, scored_at_utc, game_pk, official_date, game_datetime_utc,
                home_team_name, away_team_name, model_home_win_prob,
                market_home_no_vig_prob, home_moneyline_median, away_moneyline_median,
                recommended_side, recommended_price, edge, expected_value_per_unit,
                feature_snapshot_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.get("run_id"), row.get("scored_at_utc"), row.get("game_pk"),
                row.get("official_date"), row.get("game_datetime_utc"),
                row.get("home_team_name"), row.get("away_team_name"),
                row.get("model_home_win_prob"), row.get("market_home_no_vig_prob"),
                row.get("home_moneyline_median"), row.get("away_moneyline_median"),
                row.get("recommended_side"), row.get("recommended_price"),
                row.get("edge"), row.get("expected_value_per_unit"),
                row.get("feature_snapshot_json"),
            ),
        )
        count += 1
    return count


# ==== src/mlb_betting/feature_engineering.py ====

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .betting_math import american_to_implied_prob, no_vig_two_way
from .db import read_sql
from .team_mapping import normalize_team_name

LOGGER = logging.getLogger(__name__)

ROLLING_WINDOWS = (3, 5, 10, 20)

POSTGAME_COLUMNS = {
    "home_score", "away_score", "target_home_win", "home_margin", "total_runs",
    "detailed_state", "abstract_state", "status_code",
}


def load_mlb_games(conn) -> pd.DataFrame:
    df = read_sql(conn, "SELECT * FROM mlb_games")
    if df.empty:
        return df
    df["game_datetime_utc"] = pd.to_datetime(df["game_datetime_utc"], utc=True, errors="coerce")
    df["official_date"] = pd.to_datetime(df["official_date"], errors="coerce").dt.date.astype(str)
    return df.sort_values(["game_datetime_utc", "game_pk"]).reset_index(drop=True)


def _latest_snapshot_by_outcome(odds: pd.DataFrame) -> pd.DataFrame:
    if odds.empty:
        return odds
    odds = odds.copy()
    odds["fetched_at_utc"] = pd.to_datetime(odds["fetched_at_utc"], utc=True, errors="coerce")
    sort_cols = ["event_id", "bookmaker_key", "market_key", "outcome_name_norm", "outcome_point_key", "fetched_at_utc"]
    return odds.sort_values(sort_cols).groupby(sort_cols[:-1], dropna=False).tail(1)


def load_latest_odds_consensus(conn) -> pd.DataFrame:
    events = read_sql(conn, "SELECT * FROM odds_events")
    odds = read_sql(conn, "SELECT * FROM odds_snapshots")
    if events.empty or odds.empty:
        return pd.DataFrame()

    events["commence_time_utc"] = pd.to_datetime(events["commence_time_utc"], utc=True, errors="coerce")
    events["event_date"] = events["commence_time_utc"].dt.date.astype(str)
    latest = _latest_snapshot_by_outcome(odds)

    rows = []
    for event_id, ev in events.set_index("event_id").iterrows():
        ev_odds = latest[latest["event_id"] == event_id]
        if ev_odds.empty:
            continue
        home_norm = ev.get("home_team_norm")
        away_norm = ev.get("away_team_norm")
        row = {
            "event_id": event_id,
            "event_date": ev.get("event_date"),
            "commence_time_utc": ev.get("commence_time_utc"),
            "odds_home_team": ev.get("home_team"),
            "odds_away_team": ev.get("away_team"),
            "home_team_norm": home_norm,
            "away_team_norm": away_norm,
        }

        h2h = ev_odds[ev_odds["market_key"] == "h2h"].copy()
        home_prices = pd.to_numeric(h2h.loc[h2h["outcome_name_norm"] == home_norm, "outcome_price"], errors="coerce")
        away_prices = pd.to_numeric(h2h.loc[h2h["outcome_name_norm"] == away_norm, "outcome_price"], errors="coerce")
        row["home_moneyline_median"] = float(home_prices.median()) if home_prices.notna().any() else np.nan
        row["away_moneyline_median"] = float(away_prices.median()) if away_prices.notna().any() else np.nan
        row["book_count_h2h_home"] = int(home_prices.notna().sum())
        row["book_count_h2h_away"] = int(away_prices.notna().sum())
        if pd.notna(row["home_moneyline_median"]) and pd.notna(row["away_moneyline_median"]):
            home_imp = american_to_implied_prob(row["home_moneyline_median"])
            away_imp = american_to_implied_prob(row["away_moneyline_median"])
            row["market_home_implied_prob"] = home_imp
            row["market_away_implied_prob"] = away_imp
            row["market_home_no_vig_prob"], row["market_away_no_vig_prob"] = no_vig_two_way(home_imp, away_imp)
            row["market_vig"] = home_imp + away_imp - 1.0
        else:
            row["market_home_implied_prob"] = np.nan
            row["market_away_implied_prob"] = np.nan
            row["market_home_no_vig_prob"] = np.nan
            row["market_away_no_vig_prob"] = np.nan
            row["market_vig"] = np.nan

        spreads = ev_odds[ev_odds["market_key"] == "spreads"].copy()
        home_spreads = spreads[spreads["outcome_name_norm"] == home_norm]
        away_spreads = spreads[spreads["outcome_name_norm"] == away_norm]
        row["home_spread_median"] = pd.to_numeric(home_spreads["outcome_point"], errors="coerce").median()
        row["away_spread_median"] = pd.to_numeric(away_spreads["outcome_point"], errors="coerce").median()
        row["home_spread_price_median"] = pd.to_numeric(home_spreads["outcome_price"], errors="coerce").median()
        row["away_spread_price_median"] = pd.to_numeric(away_spreads["outcome_price"], errors="coerce").median()

        totals = ev_odds[ev_odds["market_key"] == "totals"].copy()
        totals["outcome_lower"] = totals["outcome_name"].astype(str).str.lower()
        over = totals[totals["outcome_lower"] == "over"]
        under = totals[totals["outcome_lower"] == "under"]
        row["total_points_median"] = pd.to_numeric(totals["outcome_point"], errors="coerce").median()
        row["over_price_median"] = pd.to_numeric(over["outcome_price"], errors="coerce").median()
        row["under_price_median"] = pd.to_numeric(under["outcome_price"], errors="coerce").median()
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out


def build_team_event_frame(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame()
    rows = []
    for _, g in games.iterrows():
        completed = pd.notna(g.get("home_score")) and pd.notna(g.get("away_score")) and pd.notna(g.get("target_home_win"))
        home_score = g.get("home_score") if completed else np.nan
        away_score = g.get("away_score") if completed else np.nan
        common = {
            "game_pk": g.get("game_pk"),
            "season": g.get("season"),
            "official_date": g.get("official_date"),
            "game_datetime_utc": g.get("game_datetime_utc"),
        }
        rows.append({
            **common,
            "team_id": g.get("home_team_id"),
            "team_name": g.get("home_team_name"),
            "team_norm": g.get("home_team_norm"),
            "opponent_id": g.get("away_team_id"),
            "opponent_name": g.get("away_team_name"),
            "is_home": 1,
            "runs_for": home_score,
            "runs_against": away_score,
            "win": 1 if completed and home_score > away_score else (0 if completed else np.nan),
            "run_diff": home_score - away_score if completed else np.nan,
        })
        rows.append({
            **common,
            "team_id": g.get("away_team_id"),
            "team_name": g.get("away_team_name"),
            "team_norm": g.get("away_team_norm"),
            "opponent_id": g.get("home_team_id"),
            "opponent_name": g.get("home_team_name"),
            "is_home": 0,
            "runs_for": away_score,
            "runs_against": home_score,
            "win": 1 if completed and away_score > home_score else (0 if completed else np.nan),
            "run_diff": away_score - home_score if completed else np.nan,
        })
    team_events = pd.DataFrame(rows)
    team_events["game_datetime_utc"] = pd.to_datetime(team_events["game_datetime_utc"], utc=True, errors="coerce")
    return team_events.sort_values(["team_id", "game_datetime_utc", "game_pk"]).reset_index(drop=True)


def add_rolling_team_features(team_events: pd.DataFrame, windows: Iterable[int] = ROLLING_WINDOWS) -> pd.DataFrame:
    if team_events.empty:
        return team_events
    team_events = team_events.copy().sort_values(["team_id", "game_datetime_utc", "game_pk"])
    stat_cols = ["win", "runs_for", "runs_against", "run_diff"]

    def transform_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(["game_datetime_utc", "game_pk"]).copy()
        g["games_played_to_date"] = g["win"].shift(1).notna().cumsum()
        g["prev_game_datetime_utc"] = g["game_datetime_utc"].shift(1)
        g["rest_days"] = (g["game_datetime_utc"] - g["prev_game_datetime_utc"]).dt.total_seconds() / 86400.0
        for col in stat_cols:
            shifted = g[col].shift(1)
            expanding = shifted.expanding(min_periods=1).mean()
            g[f"{col}_season_to_date"] = expanding
            for window in windows:
                g[f"{col}_last{window}"] = shifted.rolling(window=window, min_periods=1).mean()
        return g

    return team_events.groupby("team_id", group_keys=False, dropna=False).apply(transform_group).reset_index(drop=True)


def _prefix_columns(df: pd.DataFrame, prefix: str, exclude: set[str]) -> pd.DataFrame:
    rename = {c: f"{prefix}{c}" for c in df.columns if c not in exclude}
    return df.rename(columns=rename)


def build_game_feature_frame(
    games: pd.DataFrame,
    odds_consensus: Optional[pd.DataFrame] = None,
    include_future: bool = True,
) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame()
    games = games.copy().sort_values(["game_datetime_utc", "game_pk"])
    if not include_future:
        games = games[games["target_home_win"].notna()].copy()

    team_events = build_team_event_frame(games)
    team_features = add_rolling_team_features(team_events)

    id_cols = {"game_pk", "season", "official_date", "game_datetime_utc"}
    rolling_cols = [
        c for c in team_features.columns
        if c in id_cols
        or c == "rest_days"
        or c == "games_played_to_date"
        or c.endswith("_season_to_date")
        or any(c.endswith(f"_last{w}") for w in ROLLING_WINDOWS)
    ]
    home = team_features[team_features["is_home"] == 1][rolling_cols].copy()
    away = team_features[team_features["is_home"] == 0][rolling_cols].copy()

    home = _prefix_columns(home, "home_", exclude=id_cols)
    away = _prefix_columns(away, "away_", exclude=id_cols)

    base_cols = [
        "game_pk", "season", "game_type", "official_date", "game_datetime_utc",
        "venue_id", "venue_name",
        "home_team_id", "home_team_name", "home_team_norm",
        "away_team_id", "away_team_name", "away_team_norm",
        "home_score", "away_score", "target_home_win", "home_margin", "total_runs",
        "probable_home_pitcher_id", "probable_home_pitcher_name",
        "probable_away_pitcher_id", "probable_away_pitcher_name",
        "status_code", "detailed_state", "abstract_state",
    ]
    base_cols = [c for c in base_cols if c in games.columns]
    frame = games[base_cols].merge(home, on=["game_pk", "season", "official_date", "game_datetime_utc"], how="left")
    frame = frame.merge(away, on=["game_pk", "season", "official_date", "game_datetime_utc"], how="left")

    for stat in ["win", "runs_for", "runs_against", "run_diff"]:
        for suffix in ["season_to_date", "last3", "last5", "last10", "last20"]:
            h = f"home_{stat}_{suffix}"
            a = f"away_{stat}_{suffix}"
            if h in frame.columns and a in frame.columns:
                frame[f"diff_{stat}_{suffix}"] = frame[h] - frame[a]
    if "home_rest_days" in frame.columns and "away_rest_days" in frame.columns:
        frame["diff_rest_days"] = frame["home_rest_days"] - frame["away_rest_days"]
    if "home_games_played_to_date" in frame.columns and "away_games_played_to_date" in frame.columns:
        frame["diff_games_played_to_date"] = frame["home_games_played_to_date"] - frame["away_games_played_to_date"]

    frame["game_month"] = pd.to_datetime(frame["game_datetime_utc"], utc=True, errors="coerce").dt.month
    frame["game_dayofweek"] = pd.to_datetime(frame["game_datetime_utc"], utc=True, errors="coerce").dt.dayofweek

    if odds_consensus is not None and not odds_consensus.empty:
        odds = odds_consensus.copy()
        odds["event_date"] = odds["event_date"].astype(str)
        odds["home_team_norm"] = odds["home_team_norm"].map(normalize_team_name)
        odds["away_team_norm"] = odds["away_team_norm"].map(normalize_team_name)
        frame = frame.merge(
            odds.drop(columns=["commence_time_utc"], errors="ignore"),
            left_on=["official_date", "home_team_norm", "away_team_norm"],
            right_on=["event_date", "home_team_norm", "away_team_norm"],
            how="left",
            suffixes=("", "_odds"),
        )
    return frame.sort_values(["game_datetime_utc", "game_pk"]).reset_index(drop=True)


def get_model_feature_columns(frame: pd.DataFrame) -> list[str]:
    blocked = set(POSTGAME_COLUMNS) | {
        "game_pk", "season", "game_type", "official_date", "game_datetime_utc",
        "venue_name", "home_team_name", "home_team_norm", "away_team_name", "away_team_norm",
        "home_team_id", "away_team_id", "home_team", "away_team",
        "probable_home_pitcher_name", "probable_away_pitcher_name",
        "probable_home_pitcher_id", "probable_away_pitcher_id",
        "event_id", "event_date", "odds_home_team", "odds_away_team",
    }
    numeric_cols = frame.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    return [c for c in numeric_cols if c not in blocked and not c.endswith("_score")]


def save_features(frame: pd.DataFrame, output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


# ==== src/mlb_betting/logging_utils.py ====

from __future__ import annotations

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ==== src/mlb_betting/mlb_stats_api.py ====

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterator, Mapping, Optional

import requests

from .db import insert_api_usage, insert_raw_payload
from .team_mapping import normalize_team_name

LOGGER = logging.getLogger(__name__)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def to_mlb_date(value: date | datetime | str) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


class MlbStatsClient:
    def __init__(self, base_url: str = "https://statsapi.mlb.com/api/v1") -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def get_schedule(
        self,
        start_date: date | datetime | str,
        end_date: date | datetime | str,
        sport_id: int = 1,
        hydrate: str = "team,probablePitcher,venue",
        game_type: Optional[str] = None,
    ) -> tuple[dict[str, Any], Mapping[str, Any], str, Mapping[str, Any]]:
        url = f"{self.base_url}/schedule"
        params: dict[str, Any] = {
            "sportId": sport_id,
            "startDate": to_mlb_date(start_date),
            "endDate": to_mlb_date(end_date),
            "hydrate": hydrate,
        }
        if game_type:
            params["gameType"] = game_type
        response = self.session.get(url, params=params, timeout=45)
        response.raise_for_status()
        return response.json(), response.headers, response.url, params


def iter_schedule_games(payload: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    for date_obj in payload.get("dates", []) or []:
        for game in date_obj.get("games", []) or []:
            yield game


def _get_team_side(game: Mapping[str, Any], side: str) -> Mapping[str, Any]:
    return ((game.get("teams") or {}).get(side) or {})


def _get_team_name(game: Mapping[str, Any], side: str) -> Optional[str]:
    team = (_get_team_side(game, side).get("team") or {})
    return team.get("name")


def _get_team_id(game: Mapping[str, Any], side: str) -> Optional[int]:
    team = (_get_team_side(game, side).get("team") or {})
    return team.get("id")


def _get_score(game: Mapping[str, Any], side: str) -> Optional[int]:
    value = _get_team_side(game, side).get("score")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _get_probable_pitcher(game: Mapping[str, Any], side: str) -> tuple[Optional[int], Optional[str]]:
    pitcher = _get_team_side(game, side).get("probablePitcher") or {}
    return pitcher.get("id"), pitcher.get("fullName")


def parse_schedule_game(game: Mapping[str, Any], fetched_at_utc: str) -> dict[str, Any]:
    status = game.get("status") or {}
    venue = game.get("venue") or {}
    home_name = _get_team_name(game, "home")
    away_name = _get_team_name(game, "away")
    home_score = _get_score(game, "home")
    away_score = _get_score(game, "away")
    completed = home_score is not None and away_score is not None and (status.get("abstractGameState") == "Final" or status.get("codedGameState") == "F")
    home_win = None
    home_margin = None
    total_runs = None
    if completed:
        home_margin = int(home_score) - int(away_score)
        home_win = 1 if home_margin > 0 else 0
        total_runs = int(home_score) + int(away_score)

    home_pitcher_id, home_pitcher_name = _get_probable_pitcher(game, "home")
    away_pitcher_id, away_pitcher_name = _get_probable_pitcher(game, "away")

    return {
        "game_pk": game.get("gamePk"),
        "game_guid": game.get("gameGuid"),
        "season": int(game.get("season")) if game.get("season") else None,
        "game_type": game.get("gameType"),
        "game_date": game.get("gameDate"),
        "official_date": game.get("officialDate"),
        "game_datetime_utc": game.get("gameDate"),
        "status_code": status.get("codedGameState"),
        "detailed_state": status.get("detailedState"),
        "abstract_state": status.get("abstractGameState"),
        "venue_id": venue.get("id"),
        "venue_name": venue.get("name"),
        "home_team_id": _get_team_id(game, "home"),
        "home_team_name": home_name,
        "home_team_norm": normalize_team_name(home_name),
        "away_team_id": _get_team_id(game, "away"),
        "away_team_name": away_name,
        "away_team_norm": normalize_team_name(away_name),
        "home_score": home_score,
        "away_score": away_score,
        "target_home_win": home_win,
        "home_margin": home_margin,
        "total_runs": total_runs,
        "probable_home_pitcher_id": home_pitcher_id,
        "probable_home_pitcher_name": home_pitcher_name,
        "probable_away_pitcher_id": away_pitcher_id,
        "probable_away_pitcher_name": away_pitcher_name,
        "last_updated_utc": fetched_at_utc,
    }


def upsert_mlb_game(conn, row: Mapping[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO mlb_games (
            game_pk, game_guid, season, game_type, game_date, official_date,
            game_datetime_utc, status_code, detailed_state, abstract_state,
            venue_id, venue_name, home_team_id, home_team_name, home_team_norm,
            away_team_id, away_team_name, away_team_norm, home_score, away_score,
            target_home_win, home_margin, total_runs,
            probable_home_pitcher_id, probable_home_pitcher_name,
            probable_away_pitcher_id, probable_away_pitcher_name, last_updated_utc
        ) VALUES (
            :game_pk, :game_guid, :season, :game_type, :game_date, :official_date,
            :game_datetime_utc, :status_code, :detailed_state, :abstract_state,
            :venue_id, :venue_name, :home_team_id, :home_team_name, :home_team_norm,
            :away_team_id, :away_team_name, :away_team_norm, :home_score, :away_score,
            :target_home_win, :home_margin, :total_runs,
            :probable_home_pitcher_id, :probable_home_pitcher_name,
            :probable_away_pitcher_id, :probable_away_pitcher_name, :last_updated_utc
        )
        ON CONFLICT(game_pk) DO UPDATE SET
            game_guid=excluded.game_guid,
            season=excluded.season,
            game_type=excluded.game_type,
            game_date=excluded.game_date,
            official_date=excluded.official_date,
            game_datetime_utc=excluded.game_datetime_utc,
            status_code=excluded.status_code,
            detailed_state=excluded.detailed_state,
            abstract_state=excluded.abstract_state,
            venue_id=excluded.venue_id,
            venue_name=excluded.venue_name,
            home_team_id=excluded.home_team_id,
            home_team_name=excluded.home_team_name,
            home_team_norm=excluded.home_team_norm,
            away_team_id=excluded.away_team_id,
            away_team_name=excluded.away_team_name,
            away_team_norm=excluded.away_team_norm,
            home_score=excluded.home_score,
            away_score=excluded.away_score,
            target_home_win=excluded.target_home_win,
            home_margin=excluded.home_margin,
            total_runs=excluded.total_runs,
            probable_home_pitcher_id=excluded.probable_home_pitcher_id,
            probable_home_pitcher_name=excluded.probable_home_pitcher_name,
            probable_away_pitcher_id=excluded.probable_away_pitcher_id,
            probable_away_pitcher_name=excluded.probable_away_pitcher_name,
            last_updated_utc=excluded.last_updated_utc
        """,
        dict(row),
    )


def fetch_schedule_to_db(
    conn,
    client: MlbStatsClient,
    start_date: date | datetime | str,
    end_date: date | datetime | str,
    game_type: Optional[str] = None,
) -> dict[str, Any]:
    fetched_at = utc_now_iso()
    payload, headers, url, params = client.get_schedule(start_date, end_date, game_type=game_type)
    insert_api_usage(
        conn,
        source="mlb_stats_api",
        endpoint="/api/v1/schedule",
        fetched_at_utc=fetched_at,
        request_url=url,
        status_code=200,
        headers=headers,
    )
    insert_raw_payload(conn, "mlb_stats_api", "/api/v1/schedule", fetched_at, params, payload)
    rows = 0
    for game in iter_schedule_games(payload):
        row = parse_schedule_game(game, fetched_at)
        if row.get("game_pk") is None:
            continue
        upsert_mlb_game(conn, row)
        rows += 1
    conn.commit()
    return {"games": rows, "fetched_at_utc": fetched_at, "start_date": str(start_date), "end_date": str(end_date)}


def date_range_from_days(days_back: int, days_forward: int) -> tuple[str, str]:
    today = datetime.now(timezone.utc).date()
    return (today - timedelta(days=days_back)).isoformat(), (today + timedelta(days=days_forward)).isoformat()


# ==== src/mlb_betting/modeling.py ====

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


# ==== src/mlb_betting/odds_api.py ====

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Mapping, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests

from .db import insert_api_usage, insert_raw_payload, upsert_odds_event
from .team_mapping import normalize_team_name

LOGGER = logging.getLogger(__name__)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sanitize_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return url
    parts = urlsplit(url)
    query = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if key.lower() == "apikey":
            value = "REDACTED"
        query.append((key, value))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


class OddsApiClient:
    def __init__(self, api_key: str, base_url: str = "https://api.the-odds-api.com") -> None:
        if not api_key:
            raise ValueError("ODDS_API_KEY is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def get_sports(self, all_sports: bool = False) -> tuple[list[dict[str, Any]], Mapping[str, Any], str]:
        url = f"{self.base_url}/v4/sports/"
        params = {"apiKey": self.api_key}
        if all_sports:
            params["all"] = "true"
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json(), response.headers, response.url

    def get_odds(
        self,
        sport: str = "baseball_mlb",
        regions: str = "us",
        markets: str = "h2h,spreads,totals",
        odds_format: str = "american",
        date_format: str = "iso",
        bookmakers: Optional[str] = None,
        commence_time_from: Optional[str] = None,
        commence_time_to: Optional[str] = None,
        event_ids: Optional[str] = None,
        include_links: bool = False,
        include_sids: bool = True,
        include_bet_limits: bool = False,
    ) -> tuple[list[dict[str, Any]], Mapping[str, Any], str, Mapping[str, Any]]:
        url = f"{self.base_url}/v4/sports/{sport}/odds/"
        params: dict[str, Any] = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
            "dateFormat": date_format,
            "includeSids": str(include_sids).lower(),
            "includeLinks": str(include_links).lower(),
            "includeBetLimits": str(include_bet_limits).lower(),
        }
        if bookmakers:
            params["bookmakers"] = bookmakers
            params.pop("regions", None)
        if commence_time_from:
            params["commenceTimeFrom"] = commence_time_from
        if commence_time_to:
            params["commenceTimeTo"] = commence_time_to
        if event_ids:
            params["eventIds"] = event_ids
        response = self.session.get(url, params=params, timeout=45)
        response.raise_for_status()
        safe_params = dict(params)
        safe_params["apiKey"] = "REDACTED"
        return response.json(), response.headers, response.url, safe_params


def save_odds_payload(conn, payload: list[dict[str, Any]], fetched_at_utc: str) -> int:
    rows = 0
    for event in payload:
        upsert_odds_event(conn, event, fetched_at_utc)
        for bookmaker in event.get("bookmakers", []) or []:
            for market in bookmaker.get("markets", []) or []:
                for outcome in market.get("outcomes", []) or []:
                    point = outcome.get("point")
                    point_key = "NA" if point is None else str(point)
                    conn.execute(
                        """
                        INSERT INTO odds_snapshots (
                            fetched_at_utc, event_id, sport_key, commence_time_utc,
                            home_team, away_team, bookmaker_key, bookmaker_title,
                            bookmaker_last_update_utc, market_key, outcome_name,
                            outcome_name_norm, outcome_price, outcome_point,
                            outcome_point_key, outcome_description, outcome_link, outcome_sid
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            fetched_at_utc,
                            event.get("id"),
                            event.get("sport_key"),
                            event.get("commence_time"),
                            event.get("home_team"),
                            event.get("away_team"),
                            bookmaker.get("key"),
                            bookmaker.get("title"),
                            bookmaker.get("last_update"),
                            market.get("key"),
                            outcome.get("name"),
                            normalize_team_name(outcome.get("name")),
                            outcome.get("price"),
                            point,
                            point_key,
                            outcome.get("description"),
                            outcome.get("link"),
                            outcome.get("sid"),
                        ),
                    )
                    rows += 1
    return rows


def fetch_and_store_odds(
    conn,
    client: OddsApiClient,
    sport: str,
    regions: str,
    markets: str,
    odds_format: str = "american",
    bookmakers: Optional[str] = None,
    commence_time_from: Optional[str] = None,
    commence_time_to: Optional[str] = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    fetched_at = utc_now_iso()
    try:
        payload, headers, url, params = client.get_odds(
            sport=sport,
            regions=regions,
            markets=markets,
            odds_format=odds_format,
            bookmakers=bookmakers,
            commence_time_from=commence_time_from,
            commence_time_to=commence_time_to,
        )
        safe_url = sanitize_url(url)
        insert_api_usage(
            conn,
            source="the_odds_api",
            endpoint="/v4/sports/{sport}/odds",
            fetched_at_utc=fetched_at,
            request_url=safe_url,
            status_code=200,
            headers=headers,
        )
        if dry_run:
            conn.commit()
            return {"events": len(payload), "odds_rows": 0, "fetched_at_utc": fetched_at, "dry_run": True}
        insert_raw_payload(conn, "the_odds_api", "/v4/sports/{sport}/odds", fetched_at, params, payload)
        rows = save_odds_payload(conn, payload, fetched_at)
        conn.commit()
        return {
            "events": len(payload),
            "odds_rows": rows,
            "fetched_at_utc": fetched_at,
            "requests_remaining": headers.get("x-requests-remaining"),
            "requests_used": headers.get("x-requests-used"),
            "requests_last": headers.get("x-requests-last"),
        }
    except requests.HTTPError as exc:
        response = exc.response
        insert_api_usage(
            conn,
            source="the_odds_api",
            endpoint="/v4/sports/{sport}/odds",
            fetched_at_utc=fetched_at,
            request_url=sanitize_url(getattr(response, "url", None)),
            status_code=getattr(response, "status_code", None),
            headers=getattr(response, "headers", None),
            error_message=str(exc),
        )
        conn.commit()
        raise


# ==== src/mlb_betting/team_mapping.py ====

from __future__ import annotations

import re
from typing import Optional

TEAM_ALIASES = {
    "arizona dbacks": "arizona diamondbacks",
    "az diamondbacks": "arizona diamondbacks",
    "chi cubs": "chicago cubs",
    "chi white sox": "chicago white sox",
    "cws": "chicago white sox",
    "la angels": "los angeles angels",
    "los angeles angels of anaheim": "los angeles angels",
    "la dodgers": "los angeles dodgers",
    "ny mets": "new york mets",
    "ny yankees": "new york yankees",
    "oakland athletics": "athletics",
    "oakland as": "athletics",
    "athletics": "athletics",
    "sf giants": "san francisco giants",
    "sd padres": "san diego padres",
    "tb rays": "tampa bay rays",
    "tampa bay devil rays": "tampa bay rays",
    "washington nationals": "washington nationals",
    "wsh nationals": "washington nationals",
}


def normalize_team_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    text = str(name).strip().lower()
    text = text.replace("&", " and ")
    text = text.replace("'", "")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return TEAM_ALIASES.get(text, text)
