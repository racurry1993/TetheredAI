#!/usr/bin/env bash
set -euo pipefail

# ---- EDIT IF NEEDED ----
export PROJECT_ID="${PROJECT_ID:-tetheredai-preds}"
export REGION="${REGION:-us-central1}"
export AR_REPO="${AR_REPO:-tetheredai}"
export BUCKET="${BUCKET:-tetheredai-mlb-state-${PROJECT_ID}}"
export RUN_SA_NAME="${RUN_SA_NAME:-tetheredai-mlb-runner}"
export RUN_SA="${RUN_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
export IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/mlb-pipeline:latest"

# This script assumes it is being run from the project app root:
#   TetheredAI/MLB/mlb_betting_app

printf '\n[setup] PROJECT_ID=%s\n[setup] REGION=%s\n[setup] BUCKET=%s\n[setup] IMAGE_URI=%s\n[setup] RUN_SA=%s\n' \
  "$PROJECT_ID" "$REGION" "$BUCKET" "$IMAGE_URI" "$RUN_SA"

gcloud config set project "$PROJECT_ID"
gcloud config set run/region "$REGION"

printf '\n[setup] Enabling APIs...\n'
gcloud services enable \
  run.googleapis.com \
  cloudscheduler.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com \
  secretmanager.googleapis.com

printf '\n[setup] Creating bucket if needed...\n'
if ! gcloud storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://${BUCKET}" --location=US --uniform-bucket-level-access
else
  echo "[setup] Bucket exists: gs://${BUCKET}"
fi

printf '\n[setup] Creating Artifact Registry repo if needed...\n'
if ! gcloud artifacts repositories describe "$AR_REPO" --location="$REGION" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$AR_REPO" \
    --repository-format=docker \
    --location="$REGION" \
    --description="TetheredAI containers"
else
  echo "[setup] Artifact Registry repo exists: $AR_REPO"
fi

printf '\n[setup] Creating service account if needed...\n'
if ! gcloud iam service-accounts describe "$RUN_SA" >/dev/null 2>&1; then
  gcloud iam service-accounts create "$RUN_SA_NAME" --display-name="TetheredAI MLB Runner"
else
  echo "[setup] Service account exists: $RUN_SA"
fi

printf '\n[setup] Granting service account permissions...\n'
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${RUN_SA}" \
  --role="roles/storage.objectAdmin" >/dev/null

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${RUN_SA}" \
  --role="roles/logging.logWriter" >/dev/null

printf '\n[setup] Ensuring odds-api-key secret exists...\n'
if ! gcloud secrets describe odds-api-key >/dev/null 2>&1; then
  echo "The secret odds-api-key does not exist yet. Paste your Odds API key at the hidden prompt."
  read -s -p "Paste Odds API Key: " ODDS_API_KEY_VALUE
  echo
  printf "%s" "$ODDS_API_KEY_VALUE" | gcloud secrets create odds-api-key --data-file=-
  unset ODDS_API_KEY_VALUE
else
  echo "[setup] Secret exists: odds-api-key"
fi

gcloud secrets add-iam-policy-binding odds-api-key \
  --member="serviceAccount:${RUN_SA}" \
  --role="roles/secretmanager.secretAccessor" >/dev/null

printf '\n[setup] Writing .gcloudignore to avoid uploading local data...\n'
cat > .gcloudignore <<'EOF'
.git/
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/

.env
data/
notebooks/
models/archive/
*.zip
*.db
*.sqlite
*.parquet
EOF

printf '\n[setup] Building and pushing image...\n'
gcloud builds submit --tag "$IMAGE_URI" .

printf '\n[setup] Deleting old jobs to remove stale args/images...\n'
gcloud run jobs delete tetheredai-mlb-smoke --region="$REGION" --quiet >/dev/null 2>&1 || true
gcloud run jobs delete tetheredai-mlb-feature-refresh --region="$REGION" --quiet >/dev/null 2>&1 || true
gcloud run jobs delete tetheredai-mlb-daily-score --region="$REGION" --quiet >/dev/null 2>&1 || true

COMMON_ENV="GCS_BUCKET=${BUCKET},GCS_USE_LOCK=true,DOWNLOAD_STATE=true,UPLOAD_STATE=true,ODDS_DB_PATH=data/odds.db,DATA_DIR=data,MODEL_DIR=models,ODDS_SPORT_KEY=baseball_mlb,ODDS_REGIONS=us,ODDS_FORMAT=american"

printf '\n[setup] Creating smoke job...\n'
gcloud run jobs create tetheredai-mlb-smoke \
  --image="$IMAGE_URI" \
  --region="$REGION" \
  --service-account="$RUN_SA" \
  --cpu=1 \
  --memory=1Gi \
  --task-timeout=30m \
  --max-retries=0 \
  --set-env-vars="${COMMON_ENV},PIPELINE_MODE=smoke" \
  --set-secrets="ODDS_API_KEY=odds-api-key:latest"

printf '\n[setup] Creating tiny feature-refresh test job...\n'
gcloud run jobs create tetheredai-mlb-feature-refresh \
  --image="$IMAGE_URI" \
  --region="$REGION" \
  --service-account="$RUN_SA" \
  --cpu=4 \
  --memory=8Gi \
  --task-timeout=8h \
  --max-retries=0 \
  --set-env-vars="${COMMON_ENV},PIPELINE_MODE=feature-refresh,FETCH_ODDS=false,FETCH_GAMES=true,FETCH_BOXSCORES=true,REFRESH_STATCAST=true,VALIDATE_STATCAST=true,BUILD_FEATURES=true,SCORE_GAMES=false,DAYS_BACK=90,DAYS_FORWARD=14,GAME_CHUNK_DAYS=30,STATCAST_DAYS_BACK=14,STATCAST_CHUNK_DAYS=3,STATCAST_LIMIT_CHUNKS=1,SKIP_EXISTING_STATCAST=true" \
  --set-secrets="ODDS_API_KEY=odds-api-key:latest"

printf '\n[setup] Creating daily scoring job...\n'
gcloud run jobs create tetheredai-mlb-daily-score \
  --image="$IMAGE_URI" \
  --region="$REGION" \
  --service-account="$RUN_SA" \
  --cpu=2 \
  --memory=4Gi \
  --task-timeout=2h \
  --max-retries=0 \
  --set-env-vars="${COMMON_ENV},PIPELINE_MODE=daily,FETCH_ODDS=true,ODDS_MARKETS=h2h,FETCH_GAMES=true,FETCH_BOXSCORES=false,REFRESH_STATCAST=false,VALIDATE_STATCAST=false,BUILD_FEATURES=true,SCORE_GAMES=true,DAYS_BACK=14,DAYS_FORWARD=3,MIN_EDGE=0.02,MIN_EV=0.00,MIN_MINUTES_BEFORE_START=30" \
  --set-secrets="ODDS_API_KEY=odds-api-key:latest"

printf '\n[setup] Done. Next commands:\n'
printf 'gcloud run jobs execute tetheredai-mlb-smoke --region="%s" --wait\n' "$REGION"
printf 'gcloud run jobs execute tetheredai-mlb-feature-refresh --region="%s" --wait\n' "$REGION"
printf 'gcloud storage ls "gs://%s/mlb/**"\n' "$BUCKET"
