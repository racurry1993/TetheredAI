from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterable

from google.cloud import storage
from google.api_core.exceptions import PreconditionFailed, NotFound


DEFAULT_STATE_PREFIX = "mlb"


def get_bucket_name(bucket_name: str | None = None) -> str:
    bucket = bucket_name or os.environ.get("GCS_BUCKET")
    if not bucket:
        raise RuntimeError("GCS_BUCKET is required.")
    return bucket


def get_bucket(bucket_name: str | None = None) -> storage.Bucket:
    client = storage.Client()
    return client.bucket(get_bucket_name(bucket_name))


def blob_exists(bucket: storage.Bucket, blob_name: str) -> bool:
    return bucket.blob(blob_name).exists()


def download_blob_if_exists(bucket: storage.Bucket, blob_name: str, local_path: str | Path) -> bool:
    local_path = Path(local_path)
    blob = bucket.blob(blob_name)
    if not blob.exists():
        print(f"[gcs] Missing gs://{bucket.name}/{blob_name}; skipping download", flush=True)
        return False
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[gcs] Downloading gs://{bucket.name}/{blob_name} -> {local_path}", flush=True)
    blob.download_to_filename(str(local_path))
    return True


def upload_blob_if_exists(bucket: storage.Bucket, local_path: str | Path, blob_name: str) -> bool:
    local_path = Path(local_path)
    if not local_path.exists():
        print(f"[gcs] Missing local file {local_path}; skipping upload", flush=True)
        return False
    blob = bucket.blob(blob_name)
    print(f"[gcs] Uploading {local_path} -> gs://{bucket.name}/{blob_name}", flush=True)
    blob.upload_from_filename(str(local_path))
    return True


def download_prefix(bucket: storage.Bucket, gcs_prefix: str, local_dir: str | Path) -> int:
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    prefix = gcs_prefix.rstrip("/") + "/"
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith("/"):
            continue
        rel = blob.name[len(prefix):]
        local_path = local_dir / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[gcs] Downloading gs://{bucket.name}/{blob.name} -> {local_path}", flush=True)
        blob.download_to_filename(str(local_path))
        count += 1
    if count == 0:
        print(f"[gcs] No objects under gs://{bucket.name}/{prefix}", flush=True)
    return count


def upload_prefix(bucket: storage.Bucket, local_dir: str | Path, gcs_prefix: str, skip_parts: Iterable[str] = ("archive", "__pycache__")) -> int:
    local_dir = Path(local_dir)
    if not local_dir.exists():
        print(f"[gcs] Local directory {local_dir} missing; skipping prefix upload", flush=True)
        return 0
    prefix = gcs_prefix.rstrip("/")
    count = 0
    skip_parts = set(skip_parts)
    for path in local_dir.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip_parts for part in path.parts):
            continue
        rel = path.relative_to(local_dir).as_posix()
        blob_name = f"{prefix}/{rel}"
        print(f"[gcs] Uploading {path} -> gs://{bucket.name}/{blob_name}", flush=True)
        bucket.blob(blob_name).upload_from_filename(str(path))
        count += 1
    return count


class GCSLock:
    """A small best-effort lock using GCS generation preconditions.

    It prevents two Cloud Run jobs from writing the same SQLite state at the same time.
    If a job dies, the next run can take the lock after stale_after_seconds.
    """

    def __init__(self, bucket: storage.Bucket, name: str = "mlb/locks/pipeline.lock", stale_after_seconds: int = 4 * 60 * 60):
        self.bucket = bucket
        self.name = name
        self.stale_after_seconds = stale_after_seconds
        self.blob = bucket.blob(name)
        self.acquired = False

    def acquire(self) -> bool:
        payload = f"pid={os.getpid()} time={time.time()}\n"
        try:
            self.blob.upload_from_string(payload, if_generation_match=0)
            self.acquired = True
            print(f"[gcs-lock] Acquired gs://{self.bucket.name}/{self.name}", flush=True)
            return True
        except PreconditionFailed:
            self.blob.reload()
            updated = self.blob.updated.timestamp() if self.blob.updated else 0
            age = time.time() - updated
            if age > self.stale_after_seconds:
                print(f"[gcs-lock] Existing lock is stale ({age:.0f}s); replacing", flush=True)
                self.blob.upload_from_string(payload)
                self.acquired = True
                return True
            print(f"[gcs-lock] Lock exists and is not stale: gs://{self.bucket.name}/{self.name}", flush=True)
            return False

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            self.blob.delete()
            print(f"[gcs-lock] Released gs://{self.bucket.name}/{self.name}", flush=True)
        except NotFound:
            pass
        finally:
            self.acquired = False
