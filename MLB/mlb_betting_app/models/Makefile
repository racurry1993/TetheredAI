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
