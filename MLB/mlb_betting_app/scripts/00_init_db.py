from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlb_betting.config import get_settings
from mlb_betting.db import init_db
from mlb_betting.logging_utils import configure_logging


def main() -> None:
    configure_logging()
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    init_db(settings.odds_db_path)
    print(f"Initialized database: {settings.odds_db_path}")


if __name__ == "__main__":
    main()
