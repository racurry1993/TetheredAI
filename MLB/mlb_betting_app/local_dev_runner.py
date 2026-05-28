from pathlib import Path
import subprocess
import sys
import os


ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

print(f"Project root: {ROOT}")
print(f"Python executable: {sys.executable}")


def run_python(args):
    """
    Runs a project script using the same Python executable
    that launched this file.
    """
    cmd = [sys.executable] + args
    print("\n" + "=" * 80)
    print("Running:", " ".join(cmd))
    print("=" * 80)

    result = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=False,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {cmd}")


# Optional: install dependencies from inside Python
# Uncomment this the first time you run the project.
run_python(["-m", "pip", "install", "-r", "requirements.txt"])


# Step 1: initialize database
run_python(["scripts/00_init_db.py"])


# Step 2: fetch odds
# Start with h2h only to save API credits.
run_python([
    "scripts/01_fetch_odds.py",
    "--sport", "baseball_mlb",
    "--regions", "us",
    "--markets", "h2h",
])


# Step 3: fetch MLB games
run_python([
    "scripts/02_fetch_mlb_games.py",
    "--days-back", "730",
    "--days-forward", "14",
])


# Step 4: QA smoke test
run_python(["scripts/qa_smoke_test.py"])


# Step 5: build features
run_python(["scripts/03_build_features.py"])


print("\nPipeline completed through feature build.")