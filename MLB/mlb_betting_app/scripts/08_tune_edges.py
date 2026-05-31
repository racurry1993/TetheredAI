from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlb_betting.config import get_settings
from mlb_betting.modeling import tune_moneyline_edge_thresholds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune moneyline edge thresholds from a holdout predictions CSV with odds.")
    parser.add_argument("--input", required=True, help="CSV with target_home_win, model_home_win_prob, and odds columns")
    parser.add_argument("--output", default=None, help="Output CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = settings.project_root / in_path
    output = Path(args.output) if args.output else settings.data_dir / "processed" / "moneyline_edge_tuning.csv"
    if not output.is_absolute():
        output = settings.project_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(in_path)
    results = tune_moneyline_edge_thresholds(df)
    results.to_csv(output, index=False)
    print({"input": str(in_path), "rows": len(results), "output": str(output)})
    if len(results):
        print(results.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
