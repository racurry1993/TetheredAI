from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path("data/odds.db")

if not DB_PATH.exists():
    raise SystemExit(f"Missing {DB_PATH}. Download it from GCS first.")

conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
try:
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
        conn,
    )["name"].tolist()
    print("Tables:")
    for t in tables:
        print(" -", t)

    if "odds_snapshots" not in tables:
        raise SystemExit("No odds_snapshots table found.")

    print("\nMarket counts:")
    print(pd.read_sql_query(
        "SELECT market_key, COUNT(*) AS rows, COUNT(DISTINCT event_id) AS events FROM odds_snapshots GROUP BY market_key ORDER BY rows DESC",
        conn,
    ).to_string(index=False))

    print("\nRecent events by market:")
    print(pd.read_sql_query(
        """
        SELECT s.market_key, e.commence_time_utc, e.away_team, e.home_team,
               COUNT(*) AS rows
        FROM odds_snapshots s
        JOIN odds_events e ON e.event_id = s.event_id
        GROUP BY s.market_key, e.event_id
        ORDER BY e.commence_time_utc, s.market_key
        LIMIT 100
        """,
        conn,
    ).to_string(index=False))
finally:
    conn.close()
