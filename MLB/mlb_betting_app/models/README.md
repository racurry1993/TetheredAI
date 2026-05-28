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
