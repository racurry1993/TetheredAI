from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
from google.cloud import storage
from pybaseball import statcast
from tenacity import retry, stop_after_attempt, wait_exponential

BASE_V1 = "https://statsapi.mlb.com/api/v1"
BASE_V11 = "https://statsapi.mlb.com/api/v1.1"


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def date_range_chunks(start_date: str, end_date: str, chunk_days: int) -> Iterable[tuple[str, str]]:
    start = parse_date(start_date)
    end = parse_date(end_date)
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        yield current.isoformat(), chunk_end.isoformat()
        current = chunk_end + timedelta(days=1)


def season_years(start_date: str, end_date: str) -> list[int]:
    start = parse_date(start_date)
    end = parse_date(end_date)
    return list(range(start.year, end.year + 1))


def clamp_year_range(year: int, start_date: str, end_date: str) -> tuple[str, str]:
    start = max(parse_date(start_date), date(year, 1, 1))
    end = min(parse_date(end_date), date(year, 12, 31))
    return start.isoformat(), end.isoformat()


def upload_text(bucket_name: str, object_name: str, text: str, content_type: str = "application/json") -> None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.upload_from_string(text, content_type=content_type)


def upload_json(bucket_name: str, object_name: str, data: dict[str, Any]) -> None:
    upload_text(bucket_name, object_name, json.dumps(data, separators=(",", ":"), ensure_ascii=False))


def upload_file(bucket_name: str, object_name: str, local_path: str, content_type: str = "application/octet-stream") -> None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.upload_from_filename(local_path, content_type=content_type)


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=30))
def get_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.get(url, params=params, timeout=90)
    response.raise_for_status()
    return response.json()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(".", "_").replace(" ", "_") for c in df.columns]

    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("string")

    return df


def schedule_to_game_index(schedule: dict[str, Any], season: int | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for day in schedule.get("dates", []):
        for game in day.get("games", []):
            teams = game.get("teams", {})
            home_team = teams.get("home", {}).get("team", {})
            away_team = teams.get("away", {}).get("team", {})
            venue = game.get("venue", {})
            probable_home = teams.get("home", {}).get("probablePitcher", {})
            probable_away = teams.get("away", {}).get("probablePitcher", {})

            rows.append(
                {
                    "season": season,
                    "game_pk": game.get("gamePk"),
                    "game_date": day.get("date"),
                    "official_date": game.get("officialDate"),
                    "game_datetime": game.get("gameDate"),
                    "game_type": game.get("gameType"),
                    "status_code": game.get("status", {}).get("codedGameState"),
                    "status_description": game.get("status", {}).get("detailedState"),
                    "home_team_id": home_team.get("id"),
                    "home_team_name": home_team.get("name"),
                    "away_team_id": away_team.get("id"),
                    "away_team_name": away_team.get("name"),
                    "venue_id": venue.get("id"),
                    "venue_name": venue.get("name"),
                    "home_probable_pitcher_id": probable_home.get("id"),
                    "home_probable_pitcher_name": probable_home.get("fullName"),
                    "away_probable_pitcher_id": probable_away.get("id"),
                    "away_probable_pitcher_name": probable_away.get("fullName"),
                }
            )

    return normalize_columns(pd.DataFrame(rows))


def fetch_teams(bucket: str, season: int, prefix: str) -> None:
    data = get_json(f"{BASE_V1}/teams", {"sportId": 1, "season": season})
    upload_json(bucket, f"{prefix}/teams/season={season}/teams.json", data)


def fetch_players(bucket: str, season: int, prefix: str) -> None:
    data = get_json(f"{BASE_V1}/sports/1/players", {"season": season})
    upload_json(bucket, f"{prefix}/players/season={season}/players.json", data)


def fetch_schedule(bucket: str, start_date: str, end_date: str, prefix: str, season: int | None = None) -> tuple[dict[str, Any], pd.DataFrame]:
    schedule = get_json(
        f"{BASE_V1}/schedule",
        {
            "sportId": 1,
            "startDate": start_date,
            "endDate": end_date,
            "hydrate": "team,venue,probablePitcher,linescore",
        },
    )
    upload_json(bucket, f"{prefix}/schedule/start_date={start_date}/end_date={end_date}/schedule.json", schedule)
    game_index = schedule_to_game_index(schedule, season=season)
    return schedule, game_index


def write_game_index(bucket: str, game_index: pd.DataFrame, object_name: str) -> None:
    local_path = "/tmp/mlb_game_index.parquet"
    game_index.to_parquet(local_path, index=False)
    upload_file(bucket, object_name, local_path)


def fetch_one_game_bundle(bucket: str, game_pk: int, object_prefix: str, sleep_seconds: float = 0.10) -> dict[str, Any]:
    endpoints = {
        "feed_live": f"{BASE_V11}/game/{game_pk}/feed/live",
        "boxscore": f"{BASE_V1}/game/{game_pk}/boxscore",
        "play_by_play": f"{BASE_V1}/game/{game_pk}/playByPlay",
        "linescore": f"{BASE_V1}/game/{game_pk}/linescore",
    }
    result: dict[str, Any] = {"game_pk": game_pk, "ok": True, "errors": []}

    for name, url in endpoints.items():
        try:
            data = get_json(url)
            upload_json(bucket, f"{object_prefix}/game_pk={game_pk}/{name}.json", data)
            time.sleep(sleep_seconds)
        except Exception as exc:
            result["ok"] = False
            result["errors"].append({"endpoint": name, "error": str(exc)})

    return result


def fetch_game_bundles(bucket: str, game_pks: list[int], object_prefix: str, max_workers: int = 6) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if not game_pks:
        return failures

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_one_game_bundle, bucket, int(game_pk), object_prefix) for game_pk in game_pks]
        for idx, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if not result.get("ok"):
                failures.append(result)
            if idx % 100 == 0 or idx == len(futures):
                print(f"Fetched {idx}/{len(futures)} game bundles")

    return failures


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=5, max=60))
def fetch_statcast_chunk(start_date: str, end_date: str) -> pd.DataFrame:
    return statcast(start_dt=start_date, end_dt=end_date, verbose=True, parallel=False)


def fetch_and_stage_statcast(
    bucket: str,
    start_date: str,
    end_date: str,
    object_prefix: str,
    chunk_days: int = 3,
    sleep_seconds: float = 2.0,
) -> list[str]:
    uploaded: list[str] = []

    for chunk_start, chunk_end in date_range_chunks(start_date, end_date, chunk_days):
        print(f"Fetching Statcast {chunk_start} to {chunk_end}")
        df = fetch_statcast_chunk(chunk_start, chunk_end)
        if df is None or df.empty:
            print(f"No Statcast rows for {chunk_start} to {chunk_end}")
            time.sleep(sleep_seconds)
            continue

        df = normalize_columns(df)
        local_path = f"/tmp/statcast_{chunk_start}_{chunk_end}.parquet"
        object_name = f"{object_prefix}/start_date={chunk_start}/end_date={chunk_end}/statcast.parquet"
        df.to_parquet(local_path, index=False)
        upload_file(bucket, object_name, local_path)
        uploaded.append(object_name)
        print(f"Uploaded {len(df):,} Statcast rows to gs://{bucket}/{object_name}")
        time.sleep(sleep_seconds)

    return uploaded
