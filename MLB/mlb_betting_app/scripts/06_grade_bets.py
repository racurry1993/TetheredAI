from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from mlb_betting.betting_math import profit_if_win
from mlb_betting.config import get_settings
from mlb_betting.db import connect, read_sql
from mlb_betting.logging_utils import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grade model recommendations after games finish.")
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--stake", type=float, default=1.0)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    settings = get_settings()
    pred_path = Path(args.predictions) if args.predictions else settings.data_dir / "predictions" / "mlb_moneyline_predictions.csv"
    if not pred_path.exists():
        raise SystemExit(f"Predictions file not found: {pred_path}")
    preds = pd.read_csv(pred_path)
    with connect(settings.odds_db_path) as conn:
        games = read_sql(conn, "SELECT game_pk, home_team_name, away_team_name, home_score, away_score, target_home_win FROM mlb_games WHERE target_home_win IS NOT NULL")
    graded = preds.merge(games, on="game_pk", how="left", suffixes=("", "_actual"))
    graded = graded[graded["recommended_side"].notna()].copy()
    if graded.empty:
        print("No recommendations to grade.")
        return
    graded["bet_won"] = np.where(
        graded["recommended_side"] == graded["home_team_name"],
        graded["target_home_win"] == 1,
        graded["target_home_win"] == 0,
    )
    graded["stake"] = args.stake
    graded["profit"] = np.where(
        graded["bet_won"],
        graded["recommended_price"].apply(lambda x: profit_if_win(x, args.stake)),
        -args.stake,
    )
    summary = {
        "graded_bets": int(len(graded)),
        "wins": int(graded["bet_won"].sum()),
        "losses": int((~graded["bet_won"]).sum()),
        "win_rate": float(graded["bet_won"].mean()),
        "profit_units": float(graded["profit"].sum()),
        "roi": float(graded["profit"].sum() / graded["stake"].sum()),
    }
    output = Path(args.output) if args.output else settings.data_dir / "predictions" / "mlb_moneyline_graded.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    graded.to_csv(output, index=False)
    print({"summary": summary, "output": str(output)})


if __name__ == "__main__":
    main()
