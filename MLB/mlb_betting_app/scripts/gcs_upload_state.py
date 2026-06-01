from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlb_betting.gcs_state import GCSStateStore, utc_run_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-db", action="store_true", help="Do not upload data/odds.db")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    store = GCSStateStore()
    store.upload_runtime_state(ROOT, run_id=args.run_id or utc_run_id(), include_db=not args.no_db)


if __name__ == "__main__":
    main()
