from __future__ import annotations

import argparse
import json

from mlb_common import (
    fetch_game_bundles,
    fetch_players,
    fetch_schedule,
    fetch_statcast_chunk,
    fetch_teams,
    normalize_columns,
    upload_file,
    write_game_index,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True, help="GCS bucket name without gs://")
    parser.add_argument("--start-date", default="2023-03-30")
    parser.add_argument("--end-date", default="2023-04-02")
    parser.add_argument("--season", type=int, default=2023)
    parser.add_argument("--max-game-details", type=int, default=5)
    args = parser.parse_args()

    bucket = args.bucket
    start_date = args.start_date
    end_date = args.end_date
    season = args.season

    print(f"Starting MLB smoke test: {start_date} to {end_date}")

    raw_prefix = "raw/mlb/statsapi/smoke"
    fetch_teams(bucket, season, raw_prefix)
    fetch_players(bucket, season, raw_prefix)

    _, game_index = fetch_schedule(bucket, start_date, end_date, raw_prefix, season=season)
    game_index_object = f"bronze/mlb/game_index/smoke/start_date={start_date}/end_date={end_date}/games.parquet"
    write_game_index(bucket, game_index, game_index_object)
    print(f"Uploaded game index with {len(game_index):,} games")

    game_pks = [int(x) for x in game_index["game_pk"].dropna().tolist()]
    fetch_game_bundles(
        bucket=bucket,
        game_pks=game_pks[: args.max_game_details],
        object_prefix="raw/mlb/statsapi/smoke/games",
        max_workers=3,
    )
    print(f"Uploaded game details for {min(args.max_game_details, len(game_pks))} games")

    print("Fetching Statcast smoke-test data. This can take a few minutes.")
    statcast_df = fetch_statcast_chunk(start_date, end_date)
    if statcast_df is None or statcast_df.empty:
        raise RuntimeError("Statcast returned no rows for the smoke-test window.")

    statcast_df = normalize_columns(statcast_df)
    statcast_local = f"/tmp/mlb_statcast_smoke_{start_date}_{end_date}.parquet"
    statcast_df.to_parquet(statcast_local, index=False)
    statcast_object = f"bronze/mlb/statcast/smoke/start_date={start_date}/end_date={end_date}/statcast.parquet"
    upload_file(bucket, statcast_object, statcast_local)
    print(f"Uploaded Statcast parquet with {len(statcast_df):,} pitch rows")

    print("Smoke test complete.")
    print(json.dumps({"game_index": game_index_object, "statcast": statcast_object}, indent=2))


if __name__ == "__main__":
    main()
