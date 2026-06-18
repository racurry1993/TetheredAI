from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from google.api_core.exceptions import NotFound, PreconditionFailed
from google.cloud import storage


@dataclass(frozen=True)
class GCSStateConfig:
    bucket_name: str
    db_blob: str = "mlb/state/odds.db"
    features_blob: str = "mlb/processed/mlb_game_features.parquet"
    predictions_blob: str = "mlb/predictions/mlb_moneyline_predictions.csv"
    models_prefix: str = "mlb/models"
    artifacts_prefix: str = "mlb/artifacts"
    lock_blob: str = "mlb/locks/pipeline.lock"


class GCSStateStore:
    """Small helper around Google Cloud Storage for pipeline state.

    This intentionally treats GCS as object storage, not as a transactional DB.
    Use the lock helpers to avoid two writers updating the same SQLite file at once.
    """

    def __init__(self, config: Optional[GCSStateConfig] = None):
        if config is None:
            config = load_gcs_config()
        self.config = config
        self.client = storage.Client()
        self.bucket = self.client.bucket(config.bucket_name)

    def download_if_exists(self, blob_name: str, local_path: str | Path) -> bool:
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob = self.bucket.blob(blob_name)
        try:
            blob.reload()
        except NotFound:
            print(f"GCS object not found; skipping download: gs://{self.config.bucket_name}/{blob_name}")
            return False
        print(f"Downloading gs://{self.config.bucket_name}/{blob_name} -> {local_path}")
        blob.download_to_filename(str(local_path))
        return True

    def upload_file(
        self,
        local_path: str | Path,
        blob_name: str,
        content_type: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> bool:
        local_path = Path(local_path)
        if not local_path.exists():
            print(f"Local file not found; skipping upload: {local_path}")
            return False
        blob = self.bucket.blob(blob_name)
        if metadata:
            blob.metadata = {str(k): str(v) for k, v in metadata.items()}
        print(f"Uploading {local_path} -> gs://{self.config.bucket_name}/{blob_name}")
        blob.upload_from_filename(str(local_path), content_type=content_type)
        return True

    def upload_with_history(
        self,
        local_path: str | Path,
        latest_blob: str,
        history_prefix: str,
        run_id: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> None:
        local_path = Path(local_path)
        if not local_path.exists():
            print(f"Local file not found; skipping upload: {local_path}")
            return
        if run_id is None:
            run_id = utc_run_id()
        self.upload_file(local_path, latest_blob, content_type=content_type, metadata={"run_id": run_id})
        history_blob = f"{history_prefix.rstrip('/')}/{run_id}/{local_path.name}"
        self.upload_file(local_path, history_blob, content_type=content_type, metadata={"run_id": run_id})

    def acquire_lock(self, run_id: Optional[str] = None, ttl_minutes: int = 240) -> bool:
        """Create a lock object using generation precondition.

        If an old lock exists beyond ttl_minutes, delete it and try once more.
        """
        if run_id is None:
            run_id = utc_run_id()
        lock = self.bucket.blob(self.config.lock_blob)
        payload = {
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "ttl_minutes": ttl_minutes,
        }
        try:
            lock.upload_from_string(json.dumps(payload, indent=2), if_generation_match=0)
            print(f"Acquired GCS lock: gs://{self.config.bucket_name}/{self.config.lock_blob}")
            return True
        except PreconditionFailed:
            pass

        try:
            lock.reload()
            age_seconds = time.time() - (lock.updated.timestamp() if lock.updated else time.time())
            if age_seconds > ttl_minutes * 60:
                print("Existing GCS lock is stale; deleting and retrying lock acquisition.")
                lock.delete(if_generation_match=lock.generation)
                lock.upload_from_string(json.dumps(payload, indent=2), if_generation_match=0)
                print(f"Acquired GCS lock after stale cleanup: gs://{self.config.bucket_name}/{self.config.lock_blob}")
                return True
        except NotFound:
            return self.acquire_lock(run_id=run_id, ttl_minutes=ttl_minutes)

        print(f"Could not acquire GCS lock: gs://{self.config.bucket_name}/{self.config.lock_blob}")
        return False

    def release_lock(self) -> None:
        lock = self.bucket.blob(self.config.lock_blob)
        try:
            lock.delete()
            print(f"Released GCS lock: gs://{self.config.bucket_name}/{self.config.lock_blob}")
        except NotFound:
            print("No GCS lock to release.")

    def download_runtime_state(self, project_root: str | Path = ".") -> None:
        root = Path(project_root)
        self.download_if_exists(self.config.db_blob, root / "data" / "odds.db")
        self.download_if_exists(self.config.features_blob, root / "data" / "processed" / "mlb_game_features.parquet")
        self.download_if_exists(self.config.predictions_blob, root / "data" / "predictions" / "mlb_moneyline_predictions.csv")
        self.download_if_exists(f"{self.config.models_prefix}/mlb_moneyline_champion.joblib", root / "models" / "mlb_moneyline_champion.joblib")
        self.download_if_exists(f"{self.config.models_prefix}/mlb_moneyline_champion_metadata.json", root / "models" / "mlb_moneyline_champion_metadata.json")

    def upload_runtime_state(self, project_root: str | Path = ".", run_id: Optional[str] = None, include_db: bool = True) -> None:
        root = Path(project_root)
        if run_id is None:
            run_id = utc_run_id()
        if include_db:
            self.upload_file(root / "data" / "odds.db", self.config.db_blob, content_type="application/x-sqlite3", metadata={"run_id": run_id})
        self.upload_with_history(
            root / "data" / "processed" / "mlb_game_features.parquet",
            self.config.features_blob,
            f"{self.config.artifacts_prefix}/features",
            run_id=run_id,
            content_type="application/octet-stream",
        )
        self.upload_with_history(
            root / "data" / "predictions" / "mlb_moneyline_predictions.csv",
            self.config.predictions_blob,
            f"{self.config.artifacts_prefix}/predictions",
            run_id=run_id,
            content_type="text/csv",
        )
        self.upload_file(root / "models" / "mlb_moneyline_champion.joblib", f"{self.config.models_prefix}/mlb_moneyline_champion.joblib", content_type="application/octet-stream", metadata={"run_id": run_id})
        self.upload_file(root / "models" / "mlb_moneyline_champion_metadata.json", f"{self.config.models_prefix}/mlb_moneyline_champion_metadata.json", content_type="application/json", metadata={"run_id": run_id})


def load_gcs_config() -> GCSStateConfig:
    bucket = os.getenv("GCS_BUCKET")
    if not bucket:
        raise RuntimeError("GCS_BUCKET environment variable is required.")
    return GCSStateConfig(
        bucket_name=bucket,
        db_blob=os.getenv("GCS_DB_BLOB", "mlb/state/odds.db"),
        features_blob=os.getenv("GCS_FEATURES_BLOB", "mlb/processed/mlb_game_features.parquet"),
        predictions_blob=os.getenv("GCS_PREDICTIONS_BLOB", "mlb/predictions/mlb_moneyline_predictions.csv"),
        models_prefix=os.getenv("GCS_MODELS_PREFIX", "mlb/models"),
        artifacts_prefix=os.getenv("GCS_ARTIFACTS_PREFIX", "mlb/artifacts"),
        lock_blob=os.getenv("GCS_LOCK_BLOB", "mlb/locks/pipeline.lock"),
    )


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


# -----------------------------------------------------------------------------
# Backward-compatible helper API
# -----------------------------------------------------------------------------
# Older pipeline scripts import the helpers below directly. Keep these wrappers so
# both the newer GCSStateStore API and the existing scripts/run_gcp_pipeline.py,
# scripts/gcs_download_state.py, and scripts/gcs_upload_state.py continue to work.


def get_bucket(bucket_name: str | None = None):
    """Return a google.cloud.storage Bucket using GCS_BUCKET by default."""
    name = bucket_name or os.getenv("GCS_BUCKET")
    if not name:
        raise RuntimeError("GCS_BUCKET environment variable is required.")
    client = storage.Client()
    return client.bucket(name)


def _bucket_uri(bucket, blob_name: str) -> str:
    return f"gs://{bucket.name}/{blob_name}"


def download_blob_if_exists(bucket, blob_name: str, local_path: str | Path) -> bool:
    """Download one object when it exists. Return True if downloaded."""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    blob = bucket.blob(blob_name)
    try:
        blob.reload()
    except NotFound:
        print(f"[gcs] Missing {_bucket_uri(bucket, blob_name)}; skipping")
        return False
    print(f"[gcs] Downloading {_bucket_uri(bucket, blob_name)} -> {local_path}")
    blob.download_to_filename(str(local_path))
    return True


def download_prefix(bucket, prefix: str, local_dir: str | Path) -> int:
    """Download all objects under a prefix into local_dir. Return object count."""
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    normalized_prefix = prefix.rstrip("/") + "/"
    count = 0
    for blob in bucket.list_blobs(prefix=normalized_prefix):
        if blob.name.endswith("/"):
            continue
        rel = blob.name[len(normalized_prefix):]
        if not rel:
            continue
        target = local_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"[gcs] Downloading {_bucket_uri(bucket, blob.name)} -> {target}")
        blob.download_to_filename(str(target))
        count += 1
    if count == 0:
        print(f"[gcs] No objects found under gs://{bucket.name}/{normalized_prefix}")
    return count


def upload_blob_if_exists(bucket, local_path: str | Path, blob_name: str, content_type: str | None = None) -> bool:
    """Upload one local file when it exists. Return True if uploaded."""
    local_path = Path(local_path)
    if not local_path.exists():
        print(f"[gcs] Local file missing; skipping upload: {local_path}")
        return False
    blob = bucket.blob(blob_name)
    print(f"[gcs] Uploading {local_path} -> {_bucket_uri(bucket, blob_name)}")
    blob.upload_from_filename(str(local_path), content_type=content_type)
    return True


def upload_prefix(bucket, local_dir: str | Path, prefix: str) -> int:
    """Upload all files in local_dir recursively under prefix. Return file count."""
    local_dir = Path(local_dir)
    if not local_dir.exists():
        print(f"[gcs] Local directory missing; skipping upload: {local_dir}")
        return 0
    normalized_prefix = prefix.rstrip("/")
    count = 0
    for path in local_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir).as_posix()
        blob_name = f"{normalized_prefix}/{rel}"
        print(f"[gcs] Uploading {path} -> {_bucket_uri(bucket, blob_name)}")
        bucket.blob(blob_name).upload_from_filename(str(path))
        count += 1
    if count == 0:
        print(f"[gcs] No local files found under {local_dir}")
    return count


class GCSLock:
    """Simple GCS object lock compatible with the older pipeline runner."""

    def __init__(self, bucket, lock_blob: str | None = None, ttl_minutes: int = 240):
        self.bucket = bucket
        self.lock_blob = lock_blob or os.getenv("GCS_LOCK_BLOB", "mlb/locks/pipeline.lock")
        self.ttl_minutes = int(ttl_minutes)

    def acquire(self, run_id: str | None = None) -> bool:
        if run_id is None:
            run_id = utc_run_id()
        payload = {
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "ttl_minutes": self.ttl_minutes,
        }
        lock = self.bucket.blob(self.lock_blob)
        try:
            lock.upload_from_string(json.dumps(payload, indent=2), if_generation_match=0)
            print(f"[gcs-lock] Acquired {_bucket_uri(self.bucket, self.lock_blob)}")
            return True
        except PreconditionFailed:
            pass

        try:
            lock.reload()
            age_seconds = time.time() - (lock.updated.timestamp() if lock.updated else time.time())
            if age_seconds > self.ttl_minutes * 60:
                print("[gcs-lock] Existing lock is stale; deleting and retrying")
                lock.delete(if_generation_match=lock.generation)
                lock.upload_from_string(json.dumps(payload, indent=2), if_generation_match=0)
                print(f"[gcs-lock] Acquired {_bucket_uri(self.bucket, self.lock_blob)}")
                return True
        except NotFound:
            return self.acquire(run_id=run_id)

        print(f"[gcs-lock] Could not acquire {_bucket_uri(self.bucket, self.lock_blob)}")
        return False

    def release(self) -> None:
        lock = self.bucket.blob(self.lock_blob)
        try:
            lock.delete()
            print(f"[gcs-lock] Released {_bucket_uri(self.bucket, self.lock_blob)}")
        except NotFound:
            print("[gcs-lock] No lock to release")
