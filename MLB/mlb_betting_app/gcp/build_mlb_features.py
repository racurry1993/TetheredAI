from __future__ import annotations

import argparse

from google.cloud import bigquery


def run_query(client: bigquery.Client, sql: str) -> None:
    print(sql)
    client.query(sql).result()


def source_table(project_id: str, mode: str) -> str:
    if mode == "smoke":
        return f"{project_id}.sports_raw.mlb_statcast_smoke"
    return f"{project_id}.sports_raw.mlb_statcast_raw"


def suffix(mode: str) -> str:
    return "_smoke" if mode == "smoke" else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--mode", choices=["smoke", "prod"], default="smoke")
    args = parser.parse_args()

    client = bigquery.Client(project=args.project_id)
    statcast = source_table(args.project_id, args.mode)
    sfx = suffix(args.mode)

    batter_daily = f"{args.project_id}.sports_features.mlb_batter_daily{sfx}"
    pitcher_daily = f"{args.project_id}.sports_features.mlb_pitcher_daily_allowed{sfx}"
    hr_events = f"{args.project_id}.sports_mlb.vw_mlb_home_run_events{sfx}"

    run_query(
        client,
        f"""
        CREATE OR REPLACE VIEW `{hr_events}` AS
        SELECT
          DATE(game_date) AS game_date,
          game_year,
          game_pk,
          CAST(batter AS STRING) AS batter_id,
          CAST(pitcher AS STRING) AS pitcher_id,
          player_name,
          home_team,
          away_team,
          stand,
          p_throws,
          events,
          SAFE_CAST(launch_speed AS FLOAT64) AS launch_speed,
          SAFE_CAST(launch_angle AS FLOAT64) AS launch_angle,
          SAFE_CAST(hit_distance_sc AS FLOAT64) AS hit_distance_sc,
          bb_type,
          inning,
          inning_topbot
        FROM `{statcast}`
        WHERE events = 'home_run'
        """,
    )

    run_query(
        client,
        f"""
        CREATE OR REPLACE TABLE `{batter_daily}` AS
        SELECT
          DATE(game_date) AS game_date,
          CAST(batter AS STRING) AS batter_id,
          ANY_VALUE(player_name) AS player_name,
          COUNT(*) AS pitches_seen,
          COUNTIF(type = 'X') AS batted_balls,
          COUNTIF(events = 'home_run') AS home_runs,
          AVG(SAFE_CAST(launch_speed AS FLOAT64)) AS avg_exit_velocity,
          AVG(SAFE_CAST(launch_angle AS FLOAT64)) AS avg_launch_angle,
          COUNTIF(SAFE_CAST(launch_speed AS FLOAT64) >= 95) AS hard_hit_balls,
          COUNTIF(SAFE_CAST(launch_speed_angle AS INT64) = 6) AS barrels
        FROM `{statcast}`
        GROUP BY game_date, batter_id
        """,
    )

    run_query(
        client,
        f"""
        CREATE OR REPLACE TABLE `{pitcher_daily}` AS
        SELECT
          DATE(game_date) AS game_date,
          CAST(pitcher AS STRING) AS pitcher_id,
          COUNT(*) AS pitches,
          COUNTIF(type = 'X') AS batted_balls_allowed,
          COUNTIF(events = 'home_run') AS home_runs_allowed,
          AVG(SAFE_CAST(launch_speed AS FLOAT64)) AS avg_exit_velocity_allowed,
          COUNTIF(SAFE_CAST(launch_speed AS FLOAT64) >= 95) AS hard_hit_allowed,
          COUNTIF(SAFE_CAST(launch_speed_angle AS INT64) = 6) AS barrels_allowed
        FROM `{statcast}`
        GROUP BY game_date, pitcher_id
        """,
    )

    print("Feature build complete.")


if __name__ == "__main__":
    main()
