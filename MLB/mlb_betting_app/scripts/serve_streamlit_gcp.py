from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from google.cloud import storage


APP_ENTRY = os.getenv("STREAMLIT_APP", "streamlit_app/app.py")
PORT = os.getenv("PORT", "8080")
GCS_BUCKET = os.getenv("GCS_BUCKET")
SYNC_INTERVAL_SECONDS = int(os.getenv("GCS_SYNC_INTERVAL_SECONDS", "300"))


DOWNLOADS = [
    ("mlb/state/odds.db", "data/odds.db"),
]

PREFIX_DOWNLOADS = [
    ("mlb/processed/", "data/processed"),
    ("mlb/predictions/", "data/predictions"),
    ("mlb/models/", "models"),
]


def download_blob(bucket, blob_name: str, local_path: str) -> None:
    blob = bucket.blob(blob_name)
    if not blob.exists():
        print(f"[web-sync] Missing gs://{bucket.name}/{blob_name}; skipping", flush=True)
        return

    path = Path(local_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(path))
    print(f"[web-sync] Downloaded gs://{bucket.name}/{blob_name} -> {path}", flush=True)


def sync_prefix(bucket, prefix: str, local_dir: str) -> None:
    dest = Path(local_dir)
    dest.mkdir(parents=True, exist_ok=True)

    count = 0
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith("/"):
            continue

        rel = blob.name[len(prefix):]
        if not rel:
            continue

        local_path = dest / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
        count += 1

    print(
        f"[web-sync] Synced {count} objects from gs://{bucket.name}/{prefix} -> {dest}",
        flush=True,
    )


def sync_gcs_once() -> None:
    if not GCS_BUCKET:
        print("[web-sync] GCS_BUCKET not set; skipping GCS sync", flush=True)
        return

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)

    for blob_name, local_path in DOWNLOADS:
        download_blob(bucket, blob_name, local_path)

    for prefix, local_dir in PREFIX_DOWNLOADS:
        sync_prefix(bucket, prefix, local_dir)


def sync_loop() -> None:
    while True:
        try:
            sync_gcs_once()
        except Exception as exc:
            print(f"[web-sync] Background sync failed: {exc!r}", flush=True)

        time.sleep(SYNC_INTERVAL_SECONDS)


def main() -> int:
    print(f"[web] Starting Streamlit app: {APP_ENTRY}", flush=True)
    print(f"[web] Port: {PORT}", flush=True)
    print(f"[web] GCS_BUCKET: {GCS_BUCKET}", flush=True)

    sync_gcs_once()

    thread = threading.Thread(target=sync_loop, daemon=True)
    thread.start()

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        APP_ENTRY,
        "--server.address=0.0.0.0",
        f"--server.port={PORT}",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]

    print("[web] Running:", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())