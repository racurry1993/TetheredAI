from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COMMANDS = [
    [sys.executable, "scripts/00_init_db.py"],
    [sys.executable, "scripts/01_fetch_odds.py", "--sport", "baseball_mlb", "--regions", "us", "--markets", "h2h,spreads,totals"],
    [sys.executable, "scripts/02_fetch_mlb_games.py", "--days-back", "730", "--days-forward", "14"],
    [sys.executable, "scripts/03_build_features.py"],
    [sys.executable, "scripts/04_train_moneyline_model.py", "--tune", "--calibrate", "--min-rows", "100"],
    [sys.executable, "scripts/05_score_today.py", "--days-forward", "3"],
]


def main() -> None:
    for cmd in COMMANDS:
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
