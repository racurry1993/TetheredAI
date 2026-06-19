from __future__ import annotations

import argparse
import json
from datetime import date

import pandas as pd

from mlb_common import (
    fetch_and_stage_statcast,
    fetch_game_bundles,
    fetch_players,
    fetch_schedule,
    fetch_teams,
    season_years,
    clamp_year_range,
    upload_json,
    write_game_index,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True, help="GCS bucket name without gs://")
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--include-game-details", action="store_true")
    parser.add_argument("--include-statcast", action="store_true")
    parser.add_argument("--statcast-chunk-days", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--max-workers", type=int, default=6)
    args = parser.parse_args()

    bucket = args.bucket
    all_game_indexes: list[pd.DataFrame] = []

    for season in season_years(args.start_date, args.end_date):
        season_start, season_end = clamp_year_range(season, args.start_date, args.end_date)
        print(f"Processing season {season}: {season_start} to {season_end}")

        raw_prefix = "raw/mlb/statsapi/backfill"
        fetch_teams(bucket, season, raw_prefix)
        fetch_players(bucket, season, raw_prefix)
        _, game_index = fetch_schedule(bucket, season_start, season_end, raw_prefix, season=season)
        all_game_indexes.append(game_index)

        game_index_object = f"bronze/mlb/game_index/backfill/season={season}/games.parquet"
        write_game_index(bucket, game_index, game_index_object)
        print(f"Uploaded season {season} game index with {len(game_index):,} games")

        if args.include_game_details:
            game_pks = [int(x) for x in game_index["game_pk"].dropna().tolist()]
            failures = fetch_game_bundles(
                bucket=bucket,
                game_pks=game_pks,
                object_prefix=f"raw/mlb/statsapi/backfill/games/season={season}",
                max_workers=args.max_workers,
            )
            if failures:
                upload_json(
                    bucket,
                    f"raw/mlb/statsapi/backfill/errors/season={season}/game_detail_failures.json",
                    {"failures": failures},
                )

    if all_game_indexes:
        combined = pd.concat(all_game_indexes, ignore_index=True)
        write_game_index(bucket, combined, "bronze/mlb/game_index/backfill/all/games.parquet")
        print(f"Uploaded combined game index with {len(combined):,} games")

    if args.include_statcast:
        uploaded = fetch_and_stage_statcast(
            bucket=bucket,
            start_date=args.start_date,
            end_date=args.end_date,
            object_prefix="bronze/mlb/statcast/backfill",
            chunk_days=args.statcast_chunk_days,
            sleep_seconds=args.sleep_seconds,
        )
        print(f"Uploaded {len(uploaded)} Statcast parquet chunks")

    print("Historical backfill complete.")


if __name__ == "__main__":
    main()
