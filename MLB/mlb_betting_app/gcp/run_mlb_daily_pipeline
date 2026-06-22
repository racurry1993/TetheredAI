from __future__ import annotations

import os
import subprocess
from datetime import date


def run(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    project_id = os.environ.get("PROJECT_ID", "tetheredai-preds")
    bucket = os.environ.get("BUCKET", f"{project_id}-sports-edge-data")
    run_date = os.environ.get("RUN_DATE", date.today().isoformat())

    run(
        [
            "python",
            "ingest_mlb_daily.py",
            "--bucket",
            bucket,
            "--lookback-days",
            "3",
            "--include-game-details",
            "--include-statcast",
            "--statcast-chunk-days",
            "1",
            "--sleep-seconds",
            "2",
        ]
    )

    run(
        [
            "python",
            "load_mlb_to_bigquery.py",
            "--project-id",
            project_id,
            "--bucket",
            bucket,
            "--mode",
            "daily",
            "--run-date",
            run_date,
        ]
    )

    run(
        [
            "python",
            "build_mlb_features.py",
            "--project-id",
            project_id,
            "--mode",
            "prod",
        ]
    )

    print("Daily MLB pipeline complete.", flush=True)


if __name__ == "__main__":
    main()
