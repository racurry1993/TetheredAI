from __future__ import annotations

import argparse
import os
from typing import Iterable

from google.cloud import bigquery, storage


RAW_DATASET = "sports_raw"


def table_ref(project_id: str, dataset: str, table: str) -> str:
    return f"{project_id}.{dataset}.{table}"


def list_gcs_uris(bucket_name: str, prefix: str, suffix: str) -> list[str]:
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
    uris = [
        f"gs://{bucket_name}/{blob.name}"
        for blob in blobs
        if blob.name.endswith(suffix)
    ]
    return sorted(uris)


def keep_uris_with_years(uris: Iterable[str], years: set[str]) -> list[str]:
    return sorted(
        uri for uri in uris
        if any(f"start_date={year}" in uri for year in years)
    )


def print_uris(label: str, uris: list[str], max_preview: int = 10) -> None:
    print(f"{label}: {len(uris):,} file(s)")
    for uri in uris[:max_preview]:
        print(f"  {uri}")
    if len(uris) > max_preview:
        print(f"  ... {len(uris) - max_preview:,} more")


def load_parquet(
    client: bigquery.Client,
    source_uris: list[str],
    table_id: str,
    write_disposition: str = bigquery.WriteDisposition.WRITE_TRUNCATE,
) -> None:
    if not source_uris:
        raise ValueError(f"No source URIs were provided for {table_id}")

    print(f"Loading {len(source_uris):,} parquet file(s) into {table_id} with {write_disposition}")
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.PARQUET,
        write_disposition=write_disposition,
    )
    job = client.load_table_from_uri(source_uris, table_id, job_config=job_config)
    job.result()
    table = client.get_table(table_id)
    print(f"Loaded {table.num_rows:,} rows into {table_id}")


def run_query(client: bigquery.Client, sql: str) -> None:
    print(sql)
    client.query(sql).result()


def create_or_replace_statcast_with_pitch_uid(
    client: bigquery.Client,
    source_table: str,
    target_table: str,
) -> None:
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


def print_statcast_summary(client: bigquery.Client, table_id: str) -> None:
    sql = f"""
    SELECT
      EXTRACT(YEAR FROM DATE(game_date)) AS season,
      COUNT(*) AS pitch_rows,
      COUNT(DISTINCT game_pk) AS games,
      MIN(DATE(game_date)) AS min_game_date,
      MAX(DATE(game_date)) AS max_game_date
    FROM `{table_id}`
    GROUP BY season
    ORDER BY season
    """
    print("Statcast summary by season:")
    rows = client.query(sql).result()
    print("season | pitch_rows | games | min_game_date | max_game_date")
    print("-------|------------|-------|---------------|--------------")
    for row in rows:
        print(f"{row.season} | {row.pitch_rows} | {row.games} | {row.min_game_date} | {row.max_game_date}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load MLB 2023-2026 staged Parquet into BigQuery, using repaired 2023 Statcast files."
    )
    parser.add_argument(
        "--project-id",
        default=os.environ.get("PROJECT_ID", "tetheredai-preds"),
        help="GCP project ID. Defaults to PROJECT_ID env var or tetheredai-preds.",
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("BUCKET", "tetheredai-preds-sports-edge-data"),
        help="GCS bucket name without gs://. Defaults to BUCKET env var.",
    )
    parser.add_argument(
        "--dataset",
        default=RAW_DATASET,
        help="BigQuery raw dataset name.",
    )
    args = parser.parse_args()

    project_id = args.project_id
    bucket = args.bucket
    dataset = args.dataset

    client = bigquery.Client(project=project_id)

    # Game index: prefer season-specific files because backfill/all/games.parquet
    # can be overwritten by whichever year ran most recently.
    game_index_candidates = list_gcs_uris(
        bucket,
        "bronze/mlb/game_index/backfill/",
        "games.parquet",
    )
    game_index_uris = [uri for uri in game_index_candidates if "/season=" in uri]

    if not game_index_uris:
        # Fallback only if season-specific files are not present.
        game_index_uris = [uri for uri in game_index_candidates if "/all/" in uri]

    # Statcast: use repaired 2023 files from backfill_normalized and exclude
    # original 2023 backfill files because they had mixed Parquet numeric types.
    statcast_2023_repaired = list_gcs_uris(
        bucket,
        "bronze/mlb/statcast/backfill_normalized/",
        "statcast.parquet",
    )

    statcast_backfill_all = list_gcs_uris(
        bucket,
        "bronze/mlb/statcast/backfill/",
        "statcast.parquet",
    )
    statcast_2024_2026 = keep_uris_with_years(
        statcast_backfill_all,
        {"2024", "2025", "2026"},
    )

    statcast_uris = sorted(statcast_2023_repaired + statcast_2024_2026)

    print_uris("Game index files", game_index_uris)
    print_uris("2023 repaired Statcast files", statcast_2023_repaired)
    print_uris("2024-2026 Statcast files", statcast_2024_2026)
    print(f"Total Statcast files to load: {len(statcast_uris):,}")

    game_index_table = table_ref(project_id, dataset, "mlb_game_index")
    statcast_stage_table = table_ref(project_id, dataset, "mlb_statcast_raw_stage")
    statcast_raw_table = table_ref(project_id, dataset, "mlb_statcast_raw")

    load_parquet(client, game_index_uris, game_index_table)
    load_parquet(client, statcast_uris, statcast_stage_table)
    create_or_replace_statcast_with_pitch_uid(client, statcast_stage_table, statcast_raw_table)
    print_statcast_summary(client, statcast_raw_table)

    print("All-year MLB BigQuery load complete.")


if __name__ == "__main__":
    main()
