from __future__ import annotations

import argparse
from datetime import date

from google.cloud import bigquery


def table_ref(project_id: str, dataset: str, table: str) -> str:
    return f"{project_id}.{dataset}.{table}"


def load_parquet(client: bigquery.Client, uri: str, table_id: str, write_disposition: str) -> None:
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.PARQUET,
        write_disposition=write_disposition,
    )
    print(f"Loading {uri} into {table_id} with {write_disposition}")
    job = client.load_table_from_uri(uri, table_id, job_config=job_config)
    job.result()
    table = client.get_table(table_id)
    print(f"Loaded {table.num_rows:,} rows into {table_id}")


def run_query(client: bigquery.Client, sql: str) -> None:
    print(sql)
    client.query(sql).result()


def create_or_replace_statcast_with_pitch_uid(client: bigquery.Client, project_id: str, source_table: str, target_table: str) -> None:
    sql = f"""
    CREATE OR REPLACE TABLE `{target_table}` AS
    SELECT
      TO_HEX(SHA256(CONCAT(
        COALESCE(CAST(game_pk AS STRING), ''), '|',
        COALESCE(CAST(at_bat_number AS STRING), ''), '|',
        COALESCE(CAST(pitch_number AS STRING), ''), '|',
        COALESCE(CAST(pitcher AS STRING), ''), '|',
        COALESCE(CAST(batter AS STRING), '')
      ))) AS pitch_uid,
      *
    FROM `{source_table}`
    """
    run_query(client, sql)


def merge_game_index(client: bigquery.Client, stage_table: str, final_table: str) -> None:
    sql = f"""
    MERGE `{final_table}` T
    USING `{stage_table}` S
    ON T.game_pk = S.game_pk
    WHEN MATCHED THEN UPDATE SET
      season = S.season,
      game_date = S.game_date,
      official_date = S.official_date,
      game_datetime = S.game_datetime,
      game_type = S.game_type,
      status_code = S.status_code,
      status_description = S.status_description,
      home_team_id = S.home_team_id,
      home_team_name = S.home_team_name,
      away_team_id = S.away_team_id,
      away_team_name = S.away_team_name,
      venue_id = S.venue_id,
      venue_name = S.venue_name,
      home_probable_pitcher_id = S.home_probable_pitcher_id,
      home_probable_pitcher_name = S.home_probable_pitcher_name,
      away_probable_pitcher_id = S.away_probable_pitcher_id,
      away_probable_pitcher_name = S.away_probable_pitcher_name
    WHEN NOT MATCHED THEN INSERT ROW
    """
    run_query(client, sql)


def merge_statcast(client: bigquery.Client, stage_table_with_uid: str, final_table: str) -> None:
    columns = [field.name for field in client.get_table(stage_table_with_uid).schema]
    update_columns = [c for c in columns if c != "pitch_uid"]
    update_set = ",\n      ".join([f"{c} = S.{c}" for c in update_columns])
    insert_cols = ", ".join(columns)
    insert_vals = ", ".join([f"S.{c}" for c in columns])

    sql = f"""
    MERGE `{final_table}` T
    USING `{stage_table_with_uid}` S
    ON T.pitch_uid = S.pitch_uid
    WHEN MATCHED THEN UPDATE SET
      {update_set}
    WHEN NOT MATCHED THEN INSERT ({insert_cols}) VALUES ({insert_vals})
    """
    run_query(client, sql)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--bucket", required=True, help="GCS bucket name without gs://")
    parser.add_argument("--mode", choices=["smoke", "historical", "daily"], required=True)
    parser.add_argument("--run-date", default=date.today().isoformat())
    parser.add_argument("--start-date", default="2023-03-30")
    parser.add_argument("--end-date", default="2023-04-02")
    args = parser.parse_args()

    client = bigquery.Client(project=args.project_id)
    bucket = args.bucket

    if args.mode == "smoke":
        game_uri = f"gs://{bucket}/bronze/mlb/game_index/smoke/start_date={args.start_date}/end_date={args.end_date}/games.parquet"
        statcast_uri = f"gs://{bucket}/bronze/mlb/statcast/smoke/start_date={args.start_date}/end_date={args.end_date}/statcast.parquet"
        game_table = table_ref(args.project_id, "sports_raw", "mlb_game_index_smoke")
        statcast_stage = table_ref(args.project_id, "sports_raw", "mlb_statcast_smoke_stage")
        statcast_final = table_ref(args.project_id, "sports_raw", "mlb_statcast_smoke")
        load_parquet(client, game_uri, game_table, bigquery.WriteDisposition.WRITE_TRUNCATE)
        load_parquet(client, statcast_uri, statcast_stage, bigquery.WriteDisposition.WRITE_TRUNCATE)
        create_or_replace_statcast_with_pitch_uid(client, args.project_id, statcast_stage, statcast_final)

    elif args.mode == "historical":
        game_uri = f"gs://{bucket}/bronze/mlb/game_index/backfill/all/games.parquet"
        statcast_uri = f"gs://{bucket}/bronze/mlb/statcast/backfill/*/*/statcast.parquet"
        game_table = table_ref(args.project_id, "sports_raw", "mlb_game_index")
        statcast_stage = table_ref(args.project_id, "sports_raw", "mlb_statcast_raw_stage")
        statcast_final = table_ref(args.project_id, "sports_raw", "mlb_statcast_raw")
        load_parquet(client, game_uri, game_table, bigquery.WriteDisposition.WRITE_TRUNCATE)
        load_parquet(client, statcast_uri, statcast_stage, bigquery.WriteDisposition.WRITE_TRUNCATE)
        create_or_replace_statcast_with_pitch_uid(client, args.project_id, statcast_stage, statcast_final)

    elif args.mode == "daily":
        suffix = args.run_date.replace("-", "")
        game_uri = f"gs://{bucket}/bronze/mlb/game_index/daily/run_date={args.run_date}/games.parquet"
        statcast_uri = f"gs://{bucket}/bronze/mlb/statcast/daily/run_date={args.run_date}/*/*/statcast.parquet"
        game_stage = table_ref(args.project_id, "sports_raw", f"mlb_game_index_stage_{suffix}")
        statcast_stage = table_ref(args.project_id, "sports_raw", f"mlb_statcast_stage_{suffix}")
        statcast_stage_uid = table_ref(args.project_id, "sports_raw", f"mlb_statcast_stage_uid_{suffix}")
        game_final = table_ref(args.project_id, "sports_raw", "mlb_game_index")
        statcast_final = table_ref(args.project_id, "sports_raw", "mlb_statcast_raw")
        load_parquet(client, game_uri, game_stage, bigquery.WriteDisposition.WRITE_TRUNCATE)
        load_parquet(client, statcast_uri, statcast_stage, bigquery.WriteDisposition.WRITE_TRUNCATE)
        create_or_replace_statcast_with_pitch_uid(client, args.project_id, statcast_stage, statcast_stage_uid)
        merge_game_index(client, game_stage, game_final)
        merge_statcast(client, statcast_stage_uid, statcast_final)

    print("BigQuery load complete.")


if __name__ == "__main__":
    main()
