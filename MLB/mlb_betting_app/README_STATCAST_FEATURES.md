# TetheredAI MLB Statcast Feature Upgrade

This package adds a leakage-safe Statcast-lite layer to the existing MLB pipeline. It excludes lineup features and focuses on features that can be built from pitch-level Statcast history before each game.

## New files

Copy these into your existing `MLB/mlb_betting_app` repo:

```text
src/mlb_betting/statcast_features.py
scripts/09_fetch_statcast.py
scripts/10_validate_statcast_features.py
scripts/03_build_features.py
notebooks/06_mlb_statcast_model_lab.ipynb
.github/workflows/mlb_feature_refresh_statcast.yml
.github/workflows/mlb_daily_score_statcast.yml
```

Also add this to `requirements.txt`:

```text
pybaseball>=2.2.7
```

## New database tables

`09_fetch_statcast.py` creates and upserts:

```text
mlb_statcast_team_game
mlb_statcast_team_hand_game
mlb_statcast_pitcher_game
```

These are aggregated tables, not a raw pitch-level archive. That keeps `odds.db` smaller while still preserving the features you need.

## Added feature groups

The patched `03_build_features.py` calls `mlb_betting.statcast_features.add_statcast_features()` after the existing feature frame is built.

Feature blocks include:

- team contact-quality and discipline form
- team offense vs opposing starter handedness
- starter Statcast pitch-quality and contact-allowed form
- bullpen Statcast quality and recent workload form

Examples:

```text
home_team_off_sc_xwoba_contact_last20
away_team_off_sc_barrel_rate_last20
diff_team_off_sc_hard_hit_rate_last20
home_team_vs_hand_sc_woba_last20
away_starter_statcast_sc_release_speed_mean_last5
diff_starter_statcast_sc_xwoba_allowed_contact_last10
home_sc_bullpen_pitches_last3
diff_bullpen_sc_whiff_rate_last5
```

## First low-cost GitHub test

Run `MLB Feature Refresh with Statcast` with:

```text
fetch_odds = false
days_back = 90
days_forward = 14
statcast_days_back = 14
statcast_chunk_days = 3
statcast_limit_chunks = 1
```

This validates the wiring without pulling a huge Statcast backfill.

## Full backfill

After the low-cost test passes, run:

```text
fetch_odds = false
days_back = 730
days_forward = 14
statcast_days_back = 730
statcast_chunk_days = 3
statcast_limit_chunks = blank
```

This may take a while. Do it sparingly.

## Notebook workflow

After pulling the refreshed repo locally, open:

```text
notebooks/06_mlb_statcast_model_lab.ipynb
```

Use it to compare baseline, Elo, starter, Statcast starter, team Statcast, bullpen Statcast, XGBoost, LightGBM, logit, and other models. Promote a champion only if it improves log loss, Brier score, calibration, and benchmark comparisons.

## Important caveat

A published model reaching around 60% accuracy does not mean every leakage-safe production model will do that. Treat 60% as an aspirational benchmark, not a guaranteed result. Your first priority should be better log loss, Brier score, calibration, and edge-vs-market behavior.
