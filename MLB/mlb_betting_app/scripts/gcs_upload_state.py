from __future__ import annotations

from pathlib import Path

from mlb_betting.gcs_state import upload_blob_if_exists, upload_prefix, get_bucket

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    bucket = get_bucket()
    upload_blob_if_exists(bucket, ROOT / "data" / "odds.db", "mlb/state/odds.db")
    upload_prefix(bucket, ROOT / "data" / "processed", "mlb/processed")
    upload_prefix(bucket, ROOT / "data" / "predictions", "mlb/predictions")
    upload_prefix(bucket, ROOT / "models", "mlb/models")


if __name__ == "__main__":
    main()
