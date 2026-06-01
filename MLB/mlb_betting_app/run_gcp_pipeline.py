from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlb_betting.gcs_state import GCSStateStore, utc_run_id

PYTHON = sys.executable


def bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def run(args: list[str], allow_failure: bool = False) -> None:
    cmd = [PYTHON] + args
    print("\n" + "=" * 100)
    print("Running:", " ".join(cmd))
    print("=" * 100)
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0 and not allow_failure:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TetheredAI MLB pipeline with GCS-backed state.")
    parser.add_argument("--mode", choices=["daily", "feature-refresh", "statcast-refresh", "score-only"], default=os.getenv("PIPELINE_MODE", "daily"))
    parser.add_argument("--download-state", action="store_true", default=bool_env("DOWNLOAD_STATE", True))
    parser.add_argument("--upload-state", action="store_true", default=bool_env("UPLOAD_STATE", True))
    parser.add_argument("--use-lock", action="store_true", default=bool_env("GCS_USE_LOCK", True))
    parser.add_argument("--fetch-odds", action="store_true", default=bool_env("FETCH_ODDS", False))
    parser.add_argument("--markets", default=os.getenv("ODDS_MARKETS", "h2h"))
    parser.add_argument("--days-back", default=os.getenv("DAYS_BACK", "14"))
    parser.add_argument("--days-forward", default=os.getenv("DAYS_FORWARD", "3"))
    parser.add_argument("--boxscore-days-back", default=os.getenv("BOXSCORE_DAYS_BACK"))
    parser.add_argument("--refresh-statcast", action="store_true", default=bool_env("REFRESH_STATCAST", False))
    parser.add_argument("--statcast-days-back", default=os.getenv("STATCAST_DAYS_BACK", "14"))
    parser.add_argument("--statcast-chunk-days", default=os.getenv("STATCAST_CHUNK_DAYS", "3"))
    parser.add_argument("--statcast-limit-chunks", default=os.getenv("STATCAST_LIMIT_CHUNKS", ""))
    parser.add_argument("--skip-existing-statcast", action="store_true", default=bool_env("SKIP_EXISTING_STATCAST", True))
    parser.add_argument("--score", action="store_true", default=bool_env("SCORE_GAMES", True))
    parser.add_argument("--skip-boxscores", action="store_true", default=bool_env("SKIP_BOXSCORES", False))
    args = parser.parse_args()

    run_id = utc_run_id()
    print(f"Pipeline mode: {args.mode}")
    print(f"Run id: {run_id}")

    store = GCSStateStore()
    locked = False
    try:
        if args.use_lock:
            locked = store.acquire_lock(run_id=run_id, ttl_minutes=int(os.getenv("GCS_LOCK_TTL_MINUTES", "360")))
            if not locked:
                raise RuntimeError("Another pipeline appears to be running; could not acquire GCS lock.")

        if args.download_state:
            store.download_runtime_state(ROOT)

        for path in ["data/raw", "data/processed", "data/predictions", "models"]:
            (ROOT / path).mkdir(parents=True, exist_ok=True)

        run(["scripts/00_init_db.py"])

        if args.fetch_odds:
            run([
                "scripts/01_fetch_odds.py",
                "--sport", "baseball_mlb",
                "--regions", os.getenv("ODDS_REGIONS", "us"),
                "--markets", args.markets,
            ])

        if args.mode != "score-only":
            run([
                "scripts/02_fetch_mlb_games.py",
                "--days-back", str(args.days_back),
                "--days-forward", str(args.days_forward),
                "--game-type", "R",
                "--chunk-days", "30",
            ])

            if not args.skip_boxscores:
                box_days = args.boxscore_days_back or args.days_back
                run([
                    "scripts/07_fetch_mlb_boxscores.py",
                    "--days-back", str(box_days),
                    "--sleep", "0.10",
                ])

            if args.refresh_statcast or args.mode in {"feature-refresh", "statcast-refresh"}:
                statcast_cmd = [
                    "scripts/09_fetch_statcast.py",
                    "--days-back", str(args.statcast_days_back),
                    "--chunk-days", str(args.statcast_chunk_days),
                ]
                if args.skip_existing_statcast:
                    statcast_cmd.append("--skip-existing")
                if args.statcast_limit_chunks:
                    statcast_cmd.extend(["--limit-chunks", str(args.statcast_limit_chunks)])
                run(statcast_cmd)

                # Checkpoint immediately after the expensive fetch.
                if args.upload_state:
                    store.upload_runtime_state(ROOT, run_id=f"{run_id}_post_statcast", include_db=True)

            # Lightweight table validation; should not rebuild all features.
            validate_script = ROOT / "scripts" / "10_validate_statcast_features.py"
            if validate_script.exists():
                run(["scripts/10_validate_statcast_features.py"], allow_failure=False)

            run(["scripts/03_build_features.py"])

        if args.score:
            model_path = ROOT / "models" / "mlb_moneyline_champion.joblib"
            if model_path.exists():
                run([
                    "scripts/05_score_today.py",
                    "--model-path", "models/mlb_moneyline_champion.joblib",
                    "--days-forward", str(args.days_forward),
                    "--min-edge", os.getenv("MIN_EDGE", "0.02"),
                    "--min-ev", os.getenv("MIN_EV", "0.00"),
                    "--min-minutes-before-start", os.getenv("MIN_MINUTES_BEFORE_START", "30"),
                ])
            else:
                print("Champion model not found; skipping scoring. Export/commit/upload a champion model first.")

        if args.upload_state:
            store.upload_runtime_state(ROOT, run_id=run_id, include_db=True)

    finally:
        if locked:
            store.release_lock()


if __name__ == "__main__":
    main()
