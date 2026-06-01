from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlb_betting.gcs_state import GCSStateStore


def main():
    store = GCSStateStore()
    store.download_runtime_state(ROOT)


if __name__ == "__main__":
    main()
