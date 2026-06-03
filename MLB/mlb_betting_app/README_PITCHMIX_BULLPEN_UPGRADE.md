# TetheredAI MLB pitch-mix + bullpen availability upgrade

## Files

Copy these into `MLB/mlb_betting_app/`:

```text
scripts/09_fetch_statcast.py
scripts/10_validate_statcast_features.py
src/mlb_betting/statcast_features.py
notebooks/10_mlb_pitchmix_bullpen_model_lab.ipynb
```

## What this adds

### New SQLite aggregate tables

`09_fetch_statcast.py` now creates/fills:

```text
mlb_statcast_team_pitch_type_game
mlb_statcast_pitcher_pitch_type_game
```

These store offense and pitcher performance by coarse pitch group:

```text
fastball
breaking
offspeed
other
```

### New model features

`statcast_features.py` now adds:

```text
home/away/diff bullpen availability and workload features
home/away/diff pitch-mix matchup weighted offense features
home/away/diff starter pitch-type allowed quality features
raw group-specific team and starter pitch-type rolling features
```

Examples:

```text
diff_bullpen_avail_sc_bullpen_pitches_sum_last3
diff_bullpen_avail_sc_bullpen_availability_pressure_last3
diff_pitchmix_matchup_off_sc_pitch_type_woba_last20
diff_pitchmix_matchup_off_sc_pitch_type_avg_ev_last20
diff_starter_pitchmix_allowed_sc_pitch_type_woba_allowed_last20
```

## Important first run

Because these are new SQLite tables, rerun Statcast once with:

```text
REFRESH_STATCAST=true
SKIP_EXISTING_STATCAST=false
```

After the new tables are populated, change back to:

```text
SKIP_EXISTING_STATCAST=true
```

## Recommended Cloud Run update for 2023+ refresh

```bash
export PROJECT_ID="tetheredai-preds"
export REGION="us-central1"
export AR_REPO="tetheredai"
export BUCKET="tetheredai-mlb-state-${PROJECT_ID}"
export IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/mlb-pipeline:latest"

gcloud config set project "$PROJECT_ID"
gcloud config set run/region "$REGION"

cd ~/TetheredAI/MLB/mlb_betting_app
git pull

gcloud builds submit --tag "$IMAGE_URI" .

gcloud run jobs update tetheredai-mlb-feature-refresh \
  --region="$REGION" \
  --image="$IMAGE_URI" \
  --cpu=8 \
  --memory=32Gi \
  --task-timeout=24h \
  --update-env-vars="DEPLOY_TS=$(date +%s),DOWNLOAD_STATE=true,UPLOAD_STATE=true,FETCH_GAMES=true,FETCH_BOXSCORES=true,REFRESH_STATCAST=true,VALIDATE_STATCAST=true,BUILD_FEATURES=true,SCORE_GAMES=false,START_DATE=2023-01-01,END_DATE=,DAYS_BACK=1400,DAYS_FORWARD=14,STATCAST_START_DATE=2023-01-01,STATCAST_END_DATE=,STATCAST_DAYS_BACK=1400,STATCAST_CHUNK_DAYS=7,STATCAST_LIMIT_CHUNKS=,SKIP_EXISTING_STATCAST=false"

gcloud run jobs execute tetheredai-mlb-feature-refresh \
  --region="$REGION" \
  --wait
```

## After success

```bash
cd ~/TetheredAI/MLB/mlb_betting_app
rm -f data/processed/mlb_game_features.parquet
mkdir -p data/processed

gcloud storage cp \
  "gs://${BUCKET}/mlb/processed/mlb_game_features.parquet" \
  data/processed/mlb_game_features.parquet
```

Then run:

```text
notebooks/10_mlb_pitchmix_bullpen_model_lab.ipynb
```
