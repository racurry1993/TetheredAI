from __future__ import annotations

import argparse
from datetime import date, timedelta

from mlb_common import (
    fetch_and_stage_statcast,
    fetch_game_bundles,
    fetch_schedule,
    write_game_index,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True, help="GCS bucket name without gs://")
    parser.add_argument("--run-date", default=date.today().isoformat())
    parser.add_argument("--lookback-days", type=int, default=3)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--include-game-details", action="store_true")
    parser.add_argument("--include-statcast", action="store_true")
    parser.add_argument("--statcast-chunk-days", type=int, default=1)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    run_date_obj = date.fromisoformat(args.run_date)
    start_date = args.start_date or (run_date_obj - timedelta(days=args.lookback_days)).isoformat()
    end_date = args.end_date or run_date_obj.isoformat()
    bucket = args.bucket

    print(f"Starting MLB daily incremental for {start_date} to {end_date}; run_date={args.run_date}")

    raw_prefix = f"raw/mlb/statsapi/daily/run_date={args.run_date}"
    _, game_index = fetch_schedule(bucket, start_date, end_date, raw_prefix, season=run_date_obj.year)

    game_index_object = f"bronze/mlb/game_index/daily/run_date={args.run_date}/games.parquet"
    write_game_index(bucket, game_index, game_index_object)
    print(f"Uploaded daily game index with {len(game_index):,} games")

    if args.include_game_details:
        game_pks = [int(x) for x in game_index["game_pk"].dropna().tolist()]
        fetch_game_bundles(
            bucket=bucket,
            game_pks=game_pks,
            object_prefix=f"raw/mlb/statsapi/daily/run_date={args.run_date}/games",
            max_workers=args.max_workers,
        )

    if args.include_statcast:
        uploaded = fetch_and_stage_statcast(
            bucket=bucket,
            start_date=start_date,
            end_date=end_date,
            object_prefix=f"bronze/mlb/statcast/daily/run_date={args.run_date}",
            chunk_days=args.statcast_chunk_days,
            sleep_seconds=args.sleep_seconds,
        )
        print(f"Uploaded {len(uploaded)} Statcast parquet chunks")

    print("Daily incremental ingestion complete.")


if __name__ == "__main__":
    main()
