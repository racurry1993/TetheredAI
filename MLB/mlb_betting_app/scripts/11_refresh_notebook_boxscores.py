from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlb_betting.config import get_settings
from mlb_betting.notebook_feature_builder import (
    normalize_games_frame,
    parse_boxscore_pitcher_rows,
    parse_boxscore_team_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh notebook-compatible rich boxscore raw parquet tables.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--db", default=None)
    parser.add_argument("--days-back", type=int, default=14)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.10)
    parser.add_argument("--refresh", action="store_true", help="Re-fetch games already present in the raw boxscore parquets.")
    return parser.parse_args()


def read_parquet_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def read_db_games(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM mlb_games", conn)


def fetch_mlb_boxscore(game_pk: int) -> dict:
    url = f"https://statsapi.mlb.com/api/v1/game/{int(game_pk)}/boxscore"
    response = requests.get(url, timeout=45)
    response.raise_for_status()
    return response.json()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_absolute():
        raw_dir = settings.project_root / raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    db_path = Path(args.db) if args.db else settings.odds_db_path
    if not db_path.is_absolute():
        db_path = settings.project_root / db_path

    raw_games = read_parquet_if_exists(raw_dir / "mlb_games.parquet")
    db_games = read_db_games(db_path)
    games = pd.concat([normalize_games_frame(x) for x in [raw_games, db_games] if x is not None and not x.empty], ignore_index=True, sort=False)
    if games.empty:
        raise SystemExit("No games available from raw parquet or odds.db")
    games = games.drop_duplicates("game_pk", keep="last")

    # Persist the merged game table too, so the notebook builder has current future rows.
    games.to_parquet(raw_dir / "mlb_games.parquet", index=False)

    team_path = raw_dir / "mlb_box_team_game.parquet"
    pitcher_path = raw_dir / "mlb_box_pitcher_game.parquet"
    existing_team = read_parquet_if_exists(team_path)
    existing_pitcher = read_parquet_if_exists(pitcher_path)
    existing_game_pks = set()
    if not existing_team.empty and "game_pk" in existing_team.columns:
        existing_game_pks |= set(pd.to_numeric(existing_team["game_pk"], errors="coerce").dropna().astype(int).tolist())
    if not existing_pitcher.empty and "game_pk" in existing_pitcher.columns:
        existing_game_pks |= set(pd.to_numeric(existing_pitcher["game_pk"], errors="coerce").dropna().astype(int).tolist())

    cutoff_start = pd.to_datetime(args.start_date).date() if args.start_date else (pd.Timestamp.utcnow().date() - pd.Timedelta(days=args.days_back))
    cutoff_end = pd.to_datetime(args.end_date).date() if args.end_date else pd.Timestamp.utcnow().date()
    games["official_date_dt"] = pd.to_datetime(games["official_date"], errors="coerce").dt.date
    candidates = games[
        games.get("target_home_win", pd.Series(index=games.index)).notna()
        & games["official_date_dt"].ge(cutoff_start)
        & games["official_date_dt"].le(cutoff_end)
    ].copy()
    if not args.refresh:
        candidates = candidates[~pd.to_numeric(candidates["game_pk"], errors="coerce").astype("Int64").isin(existing_game_pks)]
    candidates = candidates.sort_values(["official_date", "game_datetime_utc", "game_pk"])
    if args.limit and args.limit > 0:
        candidates = candidates.head(args.limit)

    print({
        "candidate_completed_games": int(len(candidates)),
        "existing_boxscore_games": int(len(existing_game_pks)),
        "start_date": str(cutoff_start),
        "end_date": str(cutoff_end),
        "refresh": bool(args.refresh),
    })

    team_rows = []
    pitcher_rows = []
    failures = 0
    for i, (_, row) in enumerate(candidates.iterrows(), start=1):
        game_pk = int(row["game_pk"])
        try:
            box = fetch_mlb_boxscore(game_pk)
            team_rows.extend(parse_boxscore_team_rows(row, box))
            pitcher_rows.extend(parse_boxscore_pitcher_rows(row, box))
        except Exception as exc:
            failures += 1
            print({"game_pk": game_pk, "error": repr(exc)})
        if i % 50 == 0 or i == 1:
            print({"processed": i, "total_candidates": int(len(candidates)), "game_pk": game_pk})
        if args.sleep > 0:
            time.sleep(args.sleep)

    new_team = pd.DataFrame(team_rows)
    new_pitcher = pd.DataFrame(pitcher_rows)
    if not new_team.empty:
        combined_team = pd.concat([existing_team, new_team], ignore_index=True, sort=False)
        combined_team = combined_team.drop_duplicates(["game_pk", "team_id"], keep="last")
        combined_team.to_parquet(team_path, index=False)
    if not new_pitcher.empty:
        combined_pitcher = pd.concat([existing_pitcher, new_pitcher], ignore_index=True, sort=False)
        key_cols = [c for c in ["game_pk", "pitcher_id", "team_id"] if c in combined_pitcher.columns]
        combined_pitcher = combined_pitcher.drop_duplicates(key_cols, keep="last") if key_cols else combined_pitcher
        combined_pitcher.to_parquet(pitcher_path, index=False)

    print({
        "new_team_rows": int(len(new_team)),
        "new_pitcher_rows": int(len(new_pitcher)),
        "failures": int(failures),
        "team_path": str(team_path),
        "pitcher_path": str(pitcher_path),
    })


if __name__ == "__main__":
    main()
