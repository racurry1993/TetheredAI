from __future__ import annotations

from pathlib import Path

from mlb_betting.gcs_state import download_blob_if_exists, download_prefix, get_bucket

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    bucket = get_bucket()
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)
    download_blob_if_exists(bucket, "mlb/state/odds.db", ROOT / "data" / "odds.db")
    download_prefix(bucket, "mlb/models", ROOT / "models")
    download_prefix(bucket, "mlb/processed", ROOT / "data" / "processed")
    download_prefix(bucket, "mlb/predictions", ROOT / "data" / "predictions")


if __name__ == "__main__":
    main()
