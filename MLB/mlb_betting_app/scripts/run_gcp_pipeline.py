from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from mlb_betting.gcs_state import (
    GCSLock,
    download_blob_if_exists,
    download_prefix,
    get_bucket,
    upload_blob_if_exists,
    upload_prefix,
)

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)


def log(message: str) -> None:
    print(f"[tetheredai-gcp] {message}", flush=True)


def env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int | None = None) -> int | None:
    val = os.environ.get(name)
    if val is None or str(val).strip() == "":
        return default
    return int(val)


def env_str(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name)
    if val is None or str(val).strip() == "":
        return default
    return str(val).strip()


def ensure_dirs() -> None:
    for path in [
        ROOT / "data",
        ROOT / "data" / "raw",
        ROOT / "data" / "processed",
        ROOT / "data" / "predictions",
        ROOT / "models",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def run_python(args: list[str], required: bool = True) -> int:
    script = ROOT / args[0]
    if not script.exists():
        message = f"Required script missing: {script}"
        if required:
            raise FileNotFoundError(message)
        log(message + "; skipping")
        return 0

    cmd = [sys.executable] + args
    log("Running: " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT)

    if required and result.returncode != 0:
        raise SystemExit(result.returncode)
    return result.returncode


def script_supports(script_path: str, flag: str) -> bool:
    path = ROOT / script_path
    if not path.exists():
        return False
    result = subprocess.run(
        [sys.executable, script_path, "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    return flag in combined


def download_state(sync_raw: bool = False) -> None:
    bucket = get_bucket()
    ensure_dirs()
    download_blob_if_exists(bucket, "mlb/state/odds.db", ROOT / "data" / "odds.db")
    download_prefix(bucket, "mlb/models", ROOT / "models")
    download_prefix(bucket, "mlb/processed", ROOT / "data" / "processed")
    download_prefix(bucket, "mlb/predictions", ROOT / "data" / "predictions")
    if sync_raw:
        download_prefix(bucket, "mlb/raw", ROOT / "data" / "raw")


def upload_state(sync_raw: bool = False) -> None:
    bucket = get_bucket()
    ensure_dirs()
    upload_blob_if_exists(bucket, ROOT / "data" / "odds.db", "mlb/state/odds.db")
    upload_prefix(bucket, ROOT / "data" / "processed", "mlb/processed")
    upload_prefix(bucket, ROOT / "data" / "predictions", "mlb/predictions")
    upload_prefix(bucket, ROOT / "models", "mlb/models")
    if sync_raw:
        upload_prefix(bucket, ROOT / "data" / "raw", "mlb/raw")


def smoke_test() -> None:
    log(f"ROOT={ROOT}")
    log(f"Python={sys.version}")
    log("Checking required files...")
    required = [
        "scripts/run_gcp_pipeline.py",
        "scripts/00_init_db.py",
        "scripts/02_fetch_mlb_games.py",
        "scripts/03_build_features.py",
        "scripts/03_build_notebook_features.py",
        "src/mlb_betting/notebook_feature_builder.py",
        "src/mlb_betting/gcs_state.py",
    ]
    for rel in required:
        path = ROOT / rel
        log(f"{rel}: {'OK' if path.exists() else 'MISSING'}")
    import pandas as pd  # noqa: F401
    import numpy as np  # noqa: F401
    import pyarrow  # noqa: F401
    from google.cloud import storage  # noqa: F401
    log("Imports OK")


def run_pipeline(args: argparse.Namespace) -> None:
    ensure_dirs()

    lock: GCSLock | None = None
    if args.use_lock:
        bucket = get_bucket()
        lock = GCSLock(bucket)
        if not lock.acquire():
            raise SystemExit("Another pipeline run appears to be active. Exiting due to GCS lock.")

    try:
        if args.download_state:
            download_state(sync_raw=args.sync_raw_state)

        run_python(["scripts/00_init_db.py"])

        if args.fetch_odds:
            run_python([
                "scripts/01_fetch_odds.py",
                "--sport", os.environ.get("ODDS_SPORT_KEY", "baseball_mlb"),
                "--regions", os.environ.get("ODDS_REGIONS", "us"),
                "--markets", args.markets,
            ], required=True)

        if args.fetch_games:
            game_args = ["scripts/02_fetch_mlb_games.py"]
            if args.start_date:
                game_args += ["--start-date", args.start_date]
            else:
                game_args += ["--days-back", str(args.days_back)]
            if args.end_date:
                game_args += ["--end-date", args.end_date]
            else:
                game_args += ["--days-forward", str(args.days_forward)]
            game_args += ["--game-type", "R", "--chunk-days", str(args.game_chunk_days)]
            run_python(game_args, required=True)

        if args.fetch_boxscores:
            box_args = ["scripts/07_fetch_mlb_boxscores.py"]
            if args.start_date and script_supports("scripts/07_fetch_mlb_boxscores.py", "--start-date"):
                box_args += ["--start-date", args.start_date]
            else:
                box_args += ["--days-back", str(args.days_back)]
            if args.end_date and script_supports("scripts/07_fetch_mlb_boxscores.py", "--end-date"):
                box_args += ["--end-date", args.end_date]
            box_args += ["--sleep", str(args.boxscore_sleep)]
            run_python(box_args, required=True)

        if args.refresh_statcast:
            statcast_args = ["scripts/09_fetch_statcast.py"]
            if args.statcast_start_date:
                statcast_args += ["--start-date", args.statcast_start_date]
            elif args.start_date:
                statcast_args += ["--start-date", args.start_date]
            else:
                statcast_args += ["--days-back", str(args.statcast_days_back)]
            if args.statcast_end_date:
                statcast_args += ["--end-date", args.statcast_end_date]
            elif args.end_date:
                statcast_args += ["--end-date", args.end_date]
            statcast_args += ["--chunk-days", str(args.statcast_chunk_days)]
            if args.statcast_limit_chunks is not None and script_supports("scripts/09_fetch_statcast.py", "--limit-chunks"):
                statcast_args += ["--limit-chunks", str(args.statcast_limit_chunks)]
            if args.skip_existing_statcast and script_supports("scripts/09_fetch_statcast.py", "--skip-existing"):
                statcast_args += ["--skip-existing"]
            run_python(statcast_args, required=True)

            # Checkpoint immediately after the expensive Statcast step.
            if args.upload_state:
                log("Checkpoint upload after Statcast step")
                upload_state(sync_raw=args.sync_raw_state)

        if args.refresh_notebook_boxscores:
            nb_box_args = [
                "scripts/11_refresh_notebook_boxscores.py",
                "--days-back", str(args.notebook_boxscore_days_back),
                "--sleep", str(args.boxscore_sleep),
            ]
            run_python(nb_box_args, required=True)
            if args.upload_state:
                log("Checkpoint upload after notebook-compatible raw boxscore refresh")
                upload_state(sync_raw=args.sync_raw_state)

        if args.validate_statcast and (ROOT / "scripts" / "10_validate_statcast_features.py").exists():
            run_python(["scripts/10_validate_statcast_features.py"], required=True)

        if args.build_features:
            if args.feature_builder == "notebook":
                feature_args = [
                    "scripts/03_build_notebook_features.py",
                    "--expected-features", "models/mlb_moneyline_champion_features.json",
                ]
                if args.allow_missing_model_cols:
                    feature_args.append("--allow-missing-model-cols")
            else:
                feature_args = ["scripts/03_build_features.py"]
                if not args.build_features_statcast:
                    feature_args.append("--no-statcast")
            run_python(feature_args, required=True)

        if args.score_games:
            champion = ROOT / "models" / "mlb_moneyline_champion.joblib"
            if champion.exists():
                score_args = [
                    "scripts/05_score_today.py",
                    "--model-path", "models/mlb_moneyline_champion.joblib",
                    "--days-forward", str(args.days_forward),
                ]
                if script_supports("scripts/05_score_today.py", "--min-edge"):
                    score_args += ["--min-edge", str(args.min_edge)]
                if script_supports("scripts/05_score_today.py", "--min-ev"):
                    score_args += ["--min-ev", str(args.min_ev)]
                if script_supports("scripts/05_score_today.py", "--min-minutes-before-start"):
                    score_args += ["--min-minutes-before-start", str(args.min_minutes_before_start)]
                run_python(score_args, required=True)
            else:
                log("Champion model missing at models/mlb_moneyline_champion.joblib; skipping scoring")

        if args.upload_state:
            upload_state(sync_raw=args.sync_raw_state)
    finally:
        if lock is not None:
            lock.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TetheredAI MLB GCP pipeline runner")

    parser.add_argument("--mode", default=os.environ.get("PIPELINE_MODE", "daily"), choices=["smoke", "daily", "feature-refresh"])

    parser.add_argument("--download-state", action="store_true", default=env_bool("DOWNLOAD_STATE", True))
    parser.add_argument("--upload-state", action="store_true", default=env_bool("UPLOAD_STATE", True))
    parser.add_argument("--use-lock", action="store_true", default=env_bool("GCS_USE_LOCK", True))
    parser.add_argument("--sync-raw-state", action="store_true", default=env_bool("SYNC_RAW_STATE", False), help="Download/upload gs://bucket/mlb/raw into data/raw.")

    parser.add_argument("--fetch-odds", action="store_true", default=env_bool("FETCH_ODDS", False))
    parser.add_argument("--fetch-games", action="store_true", default=env_bool("FETCH_GAMES", True))
    parser.add_argument("--fetch-boxscores", action="store_true", default=env_bool("FETCH_BOXSCORES", False))
    parser.add_argument("--refresh-statcast", action="store_true", default=env_bool("REFRESH_STATCAST", False))
    parser.add_argument("--validate-statcast", action="store_true", default=env_bool("VALIDATE_STATCAST", False))
    parser.add_argument("--build-features", action="store_true", default=env_bool("BUILD_FEATURES", True))
    parser.add_argument("--feature-builder", default=env_str("FEATURE_BUILDER", "production"), choices=["production", "notebook"], help="Use the current production builder or the notebook-compatible champion-schema builder.")
    parser.add_argument(
        "--build-features-statcast",
        action="store_true",
        default=env_bool("BUILD_FEATURES_STATCAST", False),
        help=(
            "Include Statcast attachment when running scripts/03_build_features.py. "
            "Default is false for daily Cloud Run stability; use true for feature-refresh/backfill jobs."
        ),
    )
    parser.add_argument("--score-games", action="store_true", default=env_bool("SCORE_GAMES", False))
    parser.add_argument("--refresh-notebook-boxscores", action="store_true", default=env_bool("REFRESH_NOTEBOOK_BOXSCORES", False), help="Update rich notebook-compatible raw boxscore parquet files before building notebook features.")
    parser.add_argument("--allow-missing-model-cols", action="store_true", default=env_bool("ALLOW_MISSING_MODEL_COLS", False), help="Let notebook builder fill missing champion columns with NaN instead of failing.")

    parser.add_argument("--markets", default=os.environ.get("ODDS_MARKETS", "h2h"))
    parser.add_argument("--days-back", type=int, default=env_int("DAYS_BACK", 14))
    parser.add_argument("--days-forward", type=int, default=env_int("DAYS_FORWARD", 3))
    parser.add_argument("--start-date", default=env_str("START_DATE", None), help="YYYY-MM-DD lower bound for games/boxscores; prefer for long backfills such as 2020-01-01")
    parser.add_argument("--end-date", default=env_str("END_DATE", None), help="YYYY-MM-DD upper bound; omit to use today + DAYS_FORWARD")
    parser.add_argument("--game-chunk-days", type=int, default=env_int("GAME_CHUNK_DAYS", 30))
    parser.add_argument("--boxscore-sleep", type=float, default=float(os.environ.get("BOXSCORE_SLEEP", "0.10")))
    parser.add_argument("--notebook-boxscore-days-back", type=int, default=env_int("NOTEBOOK_BOXSCORE_DAYS_BACK", 14))

    parser.add_argument("--statcast-days-back", type=int, default=env_int("STATCAST_DAYS_BACK", 14))
    parser.add_argument("--statcast-start-date", default=env_str("STATCAST_START_DATE", None), help="YYYY-MM-DD lower bound for Statcast; defaults to START_DATE when set")
    parser.add_argument("--statcast-end-date", default=env_str("STATCAST_END_DATE", None), help="YYYY-MM-DD upper bound for Statcast; defaults to END_DATE when set")
    parser.add_argument("--statcast-chunk-days", type=int, default=env_int("STATCAST_CHUNK_DAYS", 3))
    parser.add_argument("--statcast-limit-chunks", type=int, default=env_int("STATCAST_LIMIT_CHUNKS", None))
    parser.add_argument("--skip-existing-statcast", action="store_true", default=env_bool("SKIP_EXISTING_STATCAST", True))

    parser.add_argument("--min-edge", type=float, default=float(os.environ.get("MIN_EDGE", "0.02")))
    parser.add_argument("--min-ev", type=float, default=float(os.environ.get("MIN_EV", "0.00")))
    parser.add_argument("--min-minutes-before-start", type=int, default=env_int("MIN_MINUTES_BEFORE_START", 30))

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log(f"Starting mode={args.mode}")
    log(f"Project root={ROOT}")
    log(f"GCS_BUCKET={os.environ.get('GCS_BUCKET', '')}")
    log(f"START_DATE={getattr(args, 'start_date', None)} END_DATE={getattr(args, 'end_date', None)}")
    log(f"STATCAST_START_DATE={getattr(args, 'statcast_start_date', None)} STATCAST_END_DATE={getattr(args, 'statcast_end_date', None)}")
    log(f"FEATURE_BUILDER={getattr(args, 'feature_builder', None)} SYNC_RAW_STATE={getattr(args, 'sync_raw_state', None)} REFRESH_NOTEBOOK_BOXSCORES={getattr(args, 'refresh_notebook_boxscores', None)}")
    log(f"BUILD_FEATURES={getattr(args, 'build_features', None)} BUILD_FEATURES_STATCAST={getattr(args, 'build_features_statcast', None)} SCORE_GAMES={getattr(args, 'score_games', None)}")

    if args.mode == "smoke":
        smoke_test()
        return

    if args.mode == "daily":
        # Defaults for daily if not explicitly set.
        if not env_bool("FETCH_BOXSCORES", False):
            args.fetch_boxscores = False
        if not env_bool("SCORE_GAMES", False):
            args.score_games = True

    if args.mode == "feature-refresh":
        args.fetch_boxscores = True if os.environ.get("FETCH_BOXSCORES") is None else args.fetch_boxscores
        args.validate_statcast = True if os.environ.get("VALIDATE_STATCAST") is None else args.validate_statcast
        args.build_features = True

    run_pipeline(args)
    log("Done")


if __name__ == "__main__":
    main()
