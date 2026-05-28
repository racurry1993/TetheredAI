from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlb_betting.config import get_settings
from mlb_betting.db import connect, init_db, read_sql
from mlb_betting.logging_utils import configure_logging


def main() -> None:
    configure_logging()
    settings = get_settings()
    init_db(settings.odds_db_path)
    with connect(settings.odds_db_path) as conn:
        checks = {
            "odds_events": read_sql(conn, "SELECT COUNT(*) AS n FROM odds_events")["n"].iloc[0],
            "odds_snapshots": read_sql(conn, "SELECT COUNT(*) AS n FROM odds_snapshots")["n"].iloc[0],
            "mlb_games": read_sql(conn, "SELECT COUNT(*) AS n FROM mlb_games")["n"].iloc[0],
            "completed_mlb_games": read_sql(conn, "SELECT COUNT(*) AS n FROM mlb_games WHERE target_home_win IS NOT NULL")["n"].iloc[0],
        }
    print(checks)


if __name__ == "__main__":
    main()
