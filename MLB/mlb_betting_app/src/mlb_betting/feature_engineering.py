from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .betting_math import american_to_implied_prob, no_vig_two_way
from .db import read_sql
from .team_mapping import normalize_team_name

LOGGER = logging.getLogger(__name__)

ROLLING_WINDOWS = (3, 5, 10, 20)

POSTGAME_COLUMNS = {
    "home_score", "away_score", "target_home_win", "home_margin", "total_runs",
    "detailed_state", "abstract_state", "status_code",
}


def load_mlb_games(conn) -> pd.DataFrame:
    df = read_sql(conn, "SELECT * FROM mlb_games")
    if df.empty:
        return df
    df["game_datetime_utc"] = pd.to_datetime(df["game_datetime_utc"], utc=True, errors="coerce")
    df["official_date"] = pd.to_datetime(df["official_date"], errors="coerce").dt.date.astype(str)
    return df.sort_values(["game_datetime_utc", "game_pk"]).reset_index(drop=True)


def _latest_snapshot_by_outcome(odds: pd.DataFrame) -> pd.DataFrame:
    if odds.empty:
        return odds
    odds = odds.copy()
    odds["fetched_at_utc"] = pd.to_datetime(odds["fetched_at_utc"], utc=True, errors="coerce")
    sort_cols = ["event_id", "bookmaker_key", "market_key", "outcome_name_norm", "outcome_point_key", "fetched_at_utc"]
    return odds.sort_values(sort_cols).groupby(sort_cols[:-1], dropna=False).tail(1)


def load_latest_odds_consensus(conn) -> pd.DataFrame:
    events = read_sql(conn, "SELECT * FROM odds_events")
    odds = read_sql(conn, "SELECT * FROM odds_snapshots")
    if events.empty or odds.empty:
        return pd.DataFrame()

    events["commence_time_utc"] = pd.to_datetime(events["commence_time_utc"], utc=True, errors="coerce")
    events["event_date"] = events["commence_time_utc"].dt.date.astype(str)
    latest = _latest_snapshot_by_outcome(odds)

    rows = []
    for event_id, ev in events.set_index("event_id").iterrows():
        ev_odds = latest[latest["event_id"] == event_id]
        if ev_odds.empty:
            continue
        home_norm = ev.get("home_team_norm")
        away_norm = ev.get("away_team_norm")
        row = {
            "event_id": event_id,
            "event_date": ev.get("event_date"),
            "commence_time_utc": ev.get("commence_time_utc"),
            "odds_home_team": ev.get("home_team"),
            "odds_away_team": ev.get("away_team"),
            "home_team_norm": home_norm,
            "away_team_norm": away_norm,
        }

        h2h = ev_odds[ev_odds["market_key"] == "h2h"].copy()
        home_prices = pd.to_numeric(h2h.loc[h2h["outcome_name_norm"] == home_norm, "outcome_price"], errors="coerce")
        away_prices = pd.to_numeric(h2h.loc[h2h["outcome_name_norm"] == away_norm, "outcome_price"], errors="coerce")
        row["home_moneyline_median"] = float(home_prices.median()) if home_prices.notna().any() else np.nan
        row["away_moneyline_median"] = float(away_prices.median()) if away_prices.notna().any() else np.nan
        row["book_count_h2h_home"] = int(home_prices.notna().sum())
        row["book_count_h2h_away"] = int(away_prices.notna().sum())
        if pd.notna(row["home_moneyline_median"]) and pd.notna(row["away_moneyline_median"]):
            home_imp = american_to_implied_prob(row["home_moneyline_median"])
            away_imp = american_to_implied_prob(row["away_moneyline_median"])
            row["market_home_implied_prob"] = home_imp
            row["market_away_implied_prob"] = away_imp
            row["market_home_no_vig_prob"], row["market_away_no_vig_prob"] = no_vig_two_way(home_imp, away_imp)
            row["market_vig"] = home_imp + away_imp - 1.0
        else:
            row["market_home_implied_prob"] = np.nan
            row["market_away_implied_prob"] = np.nan
            row["market_home_no_vig_prob"] = np.nan
            row["market_away_no_vig_prob"] = np.nan
            row["market_vig"] = np.nan

        spreads = ev_odds[ev_odds["market_key"] == "spreads"].copy()
        home_spreads = spreads[spreads["outcome_name_norm"] == home_norm]
        away_spreads = spreads[spreads["outcome_name_norm"] == away_norm]
        row["home_spread_median"] = pd.to_numeric(home_spreads["outcome_point"], errors="coerce").median()
        row["away_spread_median"] = pd.to_numeric(away_spreads["outcome_point"], errors="coerce").median()
        row["home_spread_price_median"] = pd.to_numeric(home_spreads["outcome_price"], errors="coerce").median()
        row["away_spread_price_median"] = pd.to_numeric(away_spreads["outcome_price"], errors="coerce").median()

        totals = ev_odds[ev_odds["market_key"] == "totals"].copy()
        totals["outcome_lower"] = totals["outcome_name"].astype(str).str.lower()
        over = totals[totals["outcome_lower"] == "over"]
        under = totals[totals["outcome_lower"] == "under"]
        row["total_points_median"] = pd.to_numeric(totals["outcome_point"], errors="coerce").median()
        row["over_price_median"] = pd.to_numeric(over["outcome_price"], errors="coerce").median()
        row["under_price_median"] = pd.to_numeric(under["outcome_price"], errors="coerce").median()
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out


def build_team_event_frame(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame()
    rows = []
    for _, g in games.iterrows():
        completed = pd.notna(g.get("home_score")) and pd.notna(g.get("away_score")) and pd.notna(g.get("target_home_win"))
        home_score = g.get("home_score") if completed else np.nan
        away_score = g.get("away_score") if completed else np.nan
        common = {
            "game_pk": g.get("game_pk"),
            "season": g.get("season"),
            "official_date": g.get("official_date"),
            "game_datetime_utc": g.get("game_datetime_utc"),
        }
        rows.append({
            **common,
            "team_id": g.get("home_team_id"),
            "team_name": g.get("home_team_name"),
            "team_norm": g.get("home_team_norm"),
            "opponent_id": g.get("away_team_id"),
            "opponent_name": g.get("away_team_name"),
            "is_home": 1,
            "runs_for": home_score,
            "runs_against": away_score,
            "win": 1 if completed and home_score > away_score else (0 if completed else np.nan),
            "run_diff": home_score - away_score if completed else np.nan,
        })
        rows.append({
            **common,
            "team_id": g.get("away_team_id"),
            "team_name": g.get("away_team_name"),
            "team_norm": g.get("away_team_norm"),
            "opponent_id": g.get("home_team_id"),
            "opponent_name": g.get("home_team_name"),
            "is_home": 0,
            "runs_for": away_score,
            "runs_against": home_score,
            "win": 1 if completed and away_score > home_score else (0 if completed else np.nan),
            "run_diff": away_score - home_score if completed else np.nan,
        })
    team_events = pd.DataFrame(rows)
    team_events["game_datetime_utc"] = pd.to_datetime(team_events["game_datetime_utc"], utc=True, errors="coerce")
    return team_events.sort_values(["team_id", "game_datetime_utc", "game_pk"]).reset_index(drop=True)


def add_rolling_team_features(team_events: pd.DataFrame, windows: Iterable[int] = ROLLING_WINDOWS) -> pd.DataFrame:
    if team_events.empty:
        return team_events
    team_events = team_events.copy().sort_values(["team_id", "game_datetime_utc", "game_pk"])
    stat_cols = ["win", "runs_for", "runs_against", "run_diff"]

    def transform_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(["game_datetime_utc", "game_pk"]).copy()
        g["games_played_to_date"] = g["win"].shift(1).notna().cumsum()
        g["prev_game_datetime_utc"] = g["game_datetime_utc"].shift(1)
        g["rest_days"] = (g["game_datetime_utc"] - g["prev_game_datetime_utc"]).dt.total_seconds() / 86400.0
        for col in stat_cols:
            shifted = g[col].shift(1)
            expanding = shifted.expanding(min_periods=1).mean()
            g[f"{col}_season_to_date"] = expanding
            for window in windows:
                g[f"{col}_last{window}"] = shifted.rolling(window=window, min_periods=1).mean()
        return g

    return team_events.groupby("team_id", group_keys=False, dropna=False).apply(transform_group).reset_index(drop=True)


def _prefix_columns(df: pd.DataFrame, prefix: str, exclude: set[str]) -> pd.DataFrame:
    rename = {c: f"{prefix}{c}" for c in df.columns if c not in exclude}
    return df.rename(columns=rename)


def build_game_feature_frame(
    games: pd.DataFrame,
    odds_consensus: Optional[pd.DataFrame] = None,
    include_future: bool = True,
) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame()
    games = games.copy().sort_values(["game_datetime_utc", "game_pk"])
    if not include_future:
        games = games[games["target_home_win"].notna()].copy()

    team_events = build_team_event_frame(games)
    team_features = add_rolling_team_features(team_events)

    id_cols = {"game_pk", "season", "official_date", "game_datetime_utc"}
    rolling_cols = [
        c for c in team_features.columns
        if c in id_cols
        or c == "rest_days"
        or c == "games_played_to_date"
        or c.endswith("_season_to_date")
        or any(c.endswith(f"_last{w}") for w in ROLLING_WINDOWS)
    ]
    home = team_features[team_features["is_home"] == 1][rolling_cols].copy()
    away = team_features[team_features["is_home"] == 0][rolling_cols].copy()

    home = _prefix_columns(home, "home_", exclude=id_cols)
    away = _prefix_columns(away, "away_", exclude=id_cols)

    base_cols = [
        "game_pk", "season", "game_type", "official_date", "game_datetime_utc",
        "venue_id", "venue_name",
        "home_team_id", "home_team_name", "home_team_norm",
        "away_team_id", "away_team_name", "away_team_norm",
        "home_score", "away_score", "target_home_win", "home_margin", "total_runs",
        "probable_home_pitcher_id", "probable_home_pitcher_name",
        "probable_away_pitcher_id", "probable_away_pitcher_name",
        "status_code", "detailed_state", "abstract_state",
    ]
    base_cols = [c for c in base_cols if c in games.columns]
    frame = games[base_cols].merge(home, on=["game_pk", "season", "official_date", "game_datetime_utc"], how="left")
    frame = frame.merge(away, on=["game_pk", "season", "official_date", "game_datetime_utc"], how="left")

    for stat in ["win", "runs_for", "runs_against", "run_diff"]:
        for suffix in ["season_to_date", "last3", "last5", "last10", "last20"]:
            h = f"home_{stat}_{suffix}"
            a = f"away_{stat}_{suffix}"
            if h in frame.columns and a in frame.columns:
                frame[f"diff_{stat}_{suffix}"] = frame[h] - frame[a]
    if "home_rest_days" in frame.columns and "away_rest_days" in frame.columns:
        frame["diff_rest_days"] = frame["home_rest_days"] - frame["away_rest_days"]
    if "home_games_played_to_date" in frame.columns and "away_games_played_to_date" in frame.columns:
        frame["diff_games_played_to_date"] = frame["home_games_played_to_date"] - frame["away_games_played_to_date"]

    frame["game_month"] = pd.to_datetime(frame["game_datetime_utc"], utc=True, errors="coerce").dt.month
    frame["game_dayofweek"] = pd.to_datetime(frame["game_datetime_utc"], utc=True, errors="coerce").dt.dayofweek

    if odds_consensus is not None and not odds_consensus.empty:
        odds = odds_consensus.copy()
        odds["event_date"] = odds["event_date"].astype(str)
        odds["home_team_norm"] = odds["home_team_norm"].map(normalize_team_name)
        odds["away_team_norm"] = odds["away_team_norm"].map(normalize_team_name)
        frame = frame.merge(
            odds.drop(columns=["commence_time_utc"], errors="ignore"),
            left_on=["official_date", "home_team_norm", "away_team_norm"],
            right_on=["event_date", "home_team_norm", "away_team_norm"],
            how="left",
            suffixes=("", "_odds"),
        )
    return frame.sort_values(["game_datetime_utc", "game_pk"]).reset_index(drop=True)


def get_model_feature_columns(frame: pd.DataFrame) -> list[str]:
    blocked = set(POSTGAME_COLUMNS) | {
        "game_pk", "season", "game_type", "official_date", "game_datetime_utc",
        "venue_name", "home_team_name", "home_team_norm", "away_team_name", "away_team_norm",
        "home_team_id", "away_team_id", "home_team", "away_team",
        "probable_home_pitcher_name", "probable_away_pitcher_name",
        "probable_home_pitcher_id", "probable_away_pitcher_id",
        "event_id", "event_date", "odds_home_team", "odds_away_team",
    }
    numeric_cols = frame.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    return [c for c in numeric_cols if c not in blocked and not c.endswith("_score")]


def save_features(frame: pd.DataFrame, output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path
