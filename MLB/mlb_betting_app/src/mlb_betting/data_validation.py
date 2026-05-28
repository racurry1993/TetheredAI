from __future__ import annotations

import pandas as pd


def summarize_missingness(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["column", "missing_count", "missing_pct"])
    out = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_pct": df.isna().mean().values,
    })
    return out.sort_values("missing_pct", ascending=False).reset_index(drop=True)


def validate_no_obvious_leakage(feature_cols: list[str]) -> None:
    blocked_tokens = ["score", "target", "margin", "total_runs", "detailed_state", "abstract_state", "status_code"]
    bad = [c for c in feature_cols if any(token in c.lower() for token in blocked_tokens)]
    if bad:
        raise ValueError(f"Potential leakage columns in feature list: {bad}")


def basic_game_checks(games: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(games)),
        "completed_games": int(games["target_home_win"].notna().sum()) if "target_home_win" in games else 0,
        "min_date": str(games["official_date"].min()) if "official_date" in games and not games.empty else None,
        "max_date": str(games["official_date"].max()) if "official_date" in games and not games.empty else None,
        "duplicate_game_pk": int(games["game_pk"].duplicated().sum()) if "game_pk" in games else None,
    }
