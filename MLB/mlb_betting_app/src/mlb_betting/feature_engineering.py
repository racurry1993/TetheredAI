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
PITCHER_WINDOWS = (3, 5, 10)
TEAM_VS_HAND_WINDOWS = (10, 20)
TEAM_BOX_WINDOWS = (3, 5, 10, 20)
BULLPEN_WINDOWS = (1, 3, 5, 10)

POSTGAME_COLUMNS = {
    "home_score", "away_score", "target_home_win", "home_margin", "total_runs",
    "detailed_state", "abstract_state", "status_code",
}

MARKET_KEYWORDS = (
    "market_",
    "moneyline",
    "spread",
    "total_points",
    "over_price",
    "under_price",
    "book_count",
    "vig",
    "odds_",
    "event_id",
    "event_date",
)


def load_mlb_games(conn) -> pd.DataFrame:
    df = read_sql(conn, "SELECT * FROM mlb_games")
    if df.empty:
        return df
    df["game_datetime_utc"] = pd.to_datetime(df["game_datetime_utc"], utc=True, errors="coerce")
    df["official_date"] = pd.to_datetime(df["official_date"], errors="coerce").dt.date.astype(str)
    return df.sort_values(["game_datetime_utc", "game_pk"]).reset_index(drop=True)


def load_mlb_pitcher_game_stats(conn) -> pd.DataFrame:
    try:
        df = read_sql(conn, "SELECT * FROM mlb_pitcher_game_stats")
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    df["game_datetime_utc"] = pd.to_datetime(df["game_datetime_utc"], utc=True, errors="coerce")
    df["official_date"] = pd.to_datetime(df["official_date"], errors="coerce").dt.date.astype(str)
    return df.sort_values(["pitcher_id", "game_datetime_utc", "game_pk"]).reset_index(drop=True)


def load_mlb_team_game_stats(conn) -> pd.DataFrame:
    try:
        df = read_sql(conn, "SELECT * FROM mlb_team_game_stats")
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    df["game_datetime_utc"] = pd.to_datetime(df["game_datetime_utc"], utc=True, errors="coerce")
    df["official_date"] = pd.to_datetime(df["official_date"], errors="coerce").dt.date.astype(str)
    return df.sort_values(["team_id", "game_datetime_utc", "game_pk"]).reset_index(drop=True)


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
    return pd.DataFrame(rows)


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
    """Create leakage-safe rolling team form features.

    Uses an explicit per-team loop instead of groupby.apply so grouping columns
    are preserved consistently across pandas versions. All rolling stats are
    shifted by one row, so the current game result is never used to predict
    itself. Future scheduled rows have NaN outcomes and therefore do not update
    result-based history, while schedule-derived rest days remain available.
    """
    if team_events.empty:
        return team_events

    team_events = team_events.copy().sort_values(["team_id", "game_datetime_utc", "game_pk"])
    stat_cols = ["win", "runs_for", "runs_against", "run_diff"]
    outputs = []

    for team_id, g in team_events.groupby("team_id", group_keys=False, dropna=False, sort=False):
        g = g.sort_values(["game_datetime_utc", "game_pk"]).copy()
        g["team_id"] = team_id
        g["games_played_to_date"] = g["win"].shift(1).notna().cumsum()
        g["prev_game_datetime_utc"] = g["game_datetime_utc"].shift(1)
        g["rest_days"] = (
            g["game_datetime_utc"] - g["prev_game_datetime_utc"]
        ).dt.total_seconds() / 86400.0

        new_cols = {}
        for col in stat_cols:
            shifted = g[col].shift(1)
            new_cols[f"{col}_season_to_date"] = shifted.expanding(min_periods=1).mean()
            for window in windows:
                new_cols[f"{col}_last{window}"] = shifted.rolling(window=window, min_periods=1).mean()

        if new_cols:
            g = pd.concat([g, pd.DataFrame(new_cols, index=g.index)], axis=1).copy()
        outputs.append(g)

    return pd.concat(outputs, ignore_index=True) if outputs else team_events


def _prefix_columns(df: pd.DataFrame, prefix: str, exclude: set[str]) -> pd.DataFrame:
    rename = {c: f"{prefix}{c}" for c in df.columns if c not in exclude}
    return df.rename(columns=rename)


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace({0: np.nan})
    return num / den


def _compute_pitcher_starter_rollups(pitcher_stats: pd.DataFrame) -> pd.DataFrame:
    """Build leakage-safe starter history features.

    This intentionally avoids ``groupby.apply`` because newer pandas versions can
    drop grouping columns from the frame passed to ``apply``. That caused
    ``pitcher_id`` to disappear in GitHub Actions. A plain loop is more verbose
    but much more stable across pandas versions.
    """
    if pitcher_stats is None or pitcher_stats.empty:
        return pd.DataFrame()

    required_cols = {"pitcher_id", "game_datetime_utc", "game_pk", "is_starter"}
    missing_required = required_cols - set(pitcher_stats.columns)
    if missing_required:
        LOGGER.warning("Pitcher stats missing required columns: %s", sorted(missing_required))
        return pd.DataFrame()

    df = pitcher_stats.copy()
    df = df[(pd.to_numeric(df.get("is_starter", 0), errors="coerce") == 1)].copy()
    if df.empty:
        LOGGER.warning("No starter pitcher rows found in pitcher stats table.")
        return pd.DataFrame()

    df["game_datetime_utc"] = pd.to_datetime(df["game_datetime_utc"], utc=True, errors="coerce")

    stat_cols = [
        "outs_pitched", "earned_runs", "hits", "walks",
        "strikeouts", "home_runs", "batters_faced", "pitches_thrown",
    ]

    # Be tolerant of partial boxscore payloads. Missing stat columns become NaN,
    # then rolling ratios naturally remain NaN until enough history exists.
    for col in stat_cols:
        if col not in df.columns:
            df[col] = np.nan

    for col in ["pitcher_id", *stat_cols]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "pitcher_name" not in df.columns:
        df["pitcher_name"] = np.nan
    if "pitcher_hand" not in df.columns:
        df["pitcher_hand"] = np.nan

    df = df.dropna(subset=["pitcher_id", "game_datetime_utc"]).sort_values(["pitcher_id", "game_datetime_utc", "game_pk"])
    if df.empty:
        LOGGER.warning("No starter pitcher rows remained after dropping missing pitcher_id/game_datetime_utc.")
        return pd.DataFrame()

    outputs = []

    for pitcher_id, g in df.groupby("pitcher_id", dropna=False, sort=False):
        g = g.sort_values(["game_datetime_utc", "game_pk"]).copy()
        g["pitcher_id"] = pitcher_id
        g["pitcher_hand"] = g["pitcher_hand"].ffill().bfill()

        shifted = g[stat_cols].shift(1)
        shifted_start = pd.Series(1.0, index=g.index).shift(1)

        g["starter_games_to_date"] = shifted_start.expanding(min_periods=1).sum()
        g["starter_days_since_last_start"] = (
            g["game_datetime_utc"] - g["game_datetime_utc"].shift(1)
        ).dt.total_seconds() / 86400.0

        windows: list[tuple[str, Optional[int]]] = [("season_to_date", None)] + [(f"last{w}", w) for w in PITCHER_WINDOWS]
        for suffix, window in windows:
            if window is None:
                sums = shifted.expanding(min_periods=1).sum()
                starts = shifted_start.expanding(min_periods=1).sum()
            else:
                sums = shifted.rolling(window=window, min_periods=1).sum()
                starts = shifted_start.rolling(window=window, min_periods=1).sum()

            outs = sums["outs_pitched"]
            innings = outs / 3.0
            g[f"starter_era_{suffix}"] = _safe_div(sums["earned_runs"] * 27.0, outs)
            g[f"starter_whip_{suffix}"] = _safe_div(sums["hits"] + sums["walks"], innings)
            g[f"starter_k_per_9_{suffix}"] = _safe_div(sums["strikeouts"] * 27.0, outs)
            g[f"starter_bb_per_9_{suffix}"] = _safe_div(sums["walks"] * 27.0, outs)
            g[f"starter_hr_per_9_{suffix}"] = _safe_div(sums["home_runs"] * 27.0, outs)
            g[f"starter_k_minus_bb_per_bf_{suffix}"] = _safe_div(sums["strikeouts"] - sums["walks"], sums["batters_faced"])
            g[f"starter_ip_per_start_{suffix}"] = _safe_div(innings, starts)
            g[f"starter_pitches_per_start_{suffix}"] = _safe_div(sums["pitches_thrown"], starts)

        outputs.append(g)

    if not outputs:
        return pd.DataFrame()

    out = pd.concat(outputs, ignore_index=True)

    keep_cols = [
        "game_pk", "game_datetime_utc", "pitcher_id", "pitcher_name", "pitcher_hand",
        "starter_games_to_date", "starter_days_since_last_start",
    ]
    keep_cols += [c for c in out.columns if c.startswith("starter_") and c not in keep_cols]
    keep_cols = [c for c in keep_cols if c in out.columns]

    return out[keep_cols].sort_values(["pitcher_id", "game_datetime_utc", "game_pk"]).reset_index(drop=True)


def _asof_merge_by_key(target: pd.DataFrame, history: pd.DataFrame, key_col: str, date_col: str, feature_cols: list[str], allow_exact_matches: bool = True) -> pd.DataFrame:
    """As-of merge history features onto target rows within each entity key.

    We loop by ``key_col`` before calling ``merge_asof``, so using the
    ``by=`` argument is unnecessary. Avoiding ``by=`` also prevents pandas
    merge errors when one side materializes identifiers as float64 because of
    missing values while the other side keeps them as int64.
    """
    if target is None or target.empty:
        return target
    if history is None or history.empty:
        out = target.copy()
        for col in feature_cols + ["history_game_datetime_utc"]:
            out[col] = np.nan
        return out

    target = target.copy()
    history = history.copy()

    # Normalize join keys and timestamps. Probable pitcher IDs can contain nulls,
    # which makes pandas represent the column as float; boxscore pitcher IDs are
    # often int. Numeric normalization makes the per-key filter stable.
    target[key_col] = pd.to_numeric(target[key_col], errors="coerce")
    history[key_col] = pd.to_numeric(history[key_col], errors="coerce")
    target[date_col] = pd.to_datetime(target[date_col], utc=True, errors="coerce")
    history[date_col] = pd.to_datetime(history[date_col], utc=True, errors="coerce")

    rows = []
    empty_cols = feature_cols + ["history_game_datetime_utc"]

    for key, tg in target.groupby(key_col, dropna=False, sort=False):
        tg = tg.sort_values(date_col).copy()

        if pd.isna(key):
            for col in empty_cols:
                tg[col] = np.nan
            rows.append(tg)
            continue

        hist = history[history[key_col] == key].sort_values(date_col).copy()
        if hist.empty:
            for col in empty_cols:
                tg[col] = np.nan
            rows.append(tg)
            continue

        available_feature_cols = [c for c in feature_cols if c in hist.columns]
        hist = hist[[date_col] + available_feature_cols].rename(columns={date_col: "history_game_datetime_utc"})

        # Because we already filtered history to the same key, this merge does
        # not need ``by=key_col``. That avoids int/float identifier dtype issues.
        merged = pd.merge_asof(
            tg.sort_values(date_col),
            hist.sort_values("history_game_datetime_utc"),
            left_on=date_col,
            right_on="history_game_datetime_utc",
            direction="backward",
            allow_exact_matches=allow_exact_matches,
        )

        for col in feature_cols:
            if col not in merged.columns:
                merged[col] = np.nan
        if "history_game_datetime_utc" not in merged.columns:
            merged["history_game_datetime_utc"] = np.nan

        rows.append(merged)

    return pd.concat(rows, ignore_index=True) if rows else target




def build_elo_feature_frame(
    games: pd.DataFrame,
    base_rating: float = 1500.0,
    k_factor: float = 20.0,
    home_advantage: float = 35.0,
) -> pd.DataFrame:
    """Build leakage-safe pregame Elo features.

    Ratings are recorded before each game and only updated after completed games.
    Future games therefore receive ratings based on all known completed games.
    """
    if games is None or games.empty:
        return pd.DataFrame()

    required = {"game_pk", "game_datetime_utc", "home_team_id", "away_team_id"}
    if not required.issubset(games.columns):
        LOGGER.warning("Skipping Elo features because games are missing columns: %s", sorted(required - set(games.columns)))
        return pd.DataFrame()

    df = games.copy().sort_values(["game_datetime_utc", "game_pk"])
    df["game_datetime_utc"] = pd.to_datetime(df["game_datetime_utc"], utc=True, errors="coerce")
    ratings: dict[float, float] = {}
    rows = []

    for _, g in df.iterrows():
        game_pk = g.get("game_pk")
        home_id = pd.to_numeric(pd.Series([g.get("home_team_id")]), errors="coerce").iloc[0]
        away_id = pd.to_numeric(pd.Series([g.get("away_team_id")]), errors="coerce").iloc[0]
        if pd.isna(home_id) or pd.isna(away_id):
            continue
        home_id = float(home_id)
        away_id = float(away_id)
        home_rating = ratings.get(home_id, base_rating)
        away_rating = ratings.get(away_id, base_rating)
        elo_home_prob = 1.0 / (1.0 + 10.0 ** ((away_rating - (home_rating + home_advantage)) / 400.0))
        rows.append({
            "game_pk": game_pk,
            "home_elo_pre": home_rating,
            "away_elo_pre": away_rating,
            "diff_elo_pre": home_rating - away_rating,
            "elo_home_win_prob": elo_home_prob,
        })

        # Update only after completed games. This prevents future rows from changing ratings.
        if pd.notna(g.get("target_home_win")):
            actual_home = float(g.get("target_home_win"))
            ratings[home_id] = home_rating + k_factor * (actual_home - elo_home_prob)
            ratings[away_id] = away_rating + k_factor * ((1.0 - actual_home) - (1.0 - elo_home_prob))

    return pd.DataFrame(rows)


def _compute_team_boxscore_rollups(team_stats: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Build team offense/boxscore form features from completed team-game boxscores.

    These rollups are *postgame as-of* rows, merged back with allow_exact_matches=False
    so a training row never sees its own game. Future games see the latest completed
    game, which avoids the one-game-stale issue of shifted exact-game features.
    """
    if team_stats is None or team_stats.empty:
        return pd.DataFrame()

    required = {"game_pk", "game_datetime_utc", "team_id"}
    if not required.issubset(team_stats.columns):
        LOGGER.warning("Skipping team boxscore features because columns are missing: %s", sorted(required - set(team_stats.columns)))
        return pd.DataFrame()

    df = team_stats.copy()
    df["game_datetime_utc"] = pd.to_datetime(df["game_datetime_utc"], utc=True, errors="coerce")
    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce")

    raw_cols = [
        "at_bats", "runs", "hits", "doubles", "triples", "home_runs", "walks",
        "strikeouts", "left_on_base", "stolen_bases", "caught_stealing", "avg", "obp", "slg", "ops",
    ]
    for col in raw_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["team_box_xbh"] = df["doubles"] + df["triples"] + df["home_runs"]
    df["team_box_bb_per_ab"] = _safe_div(df["walks"], df["at_bats"])
    df["team_box_k_per_ab"] = _safe_div(df["strikeouts"], df["at_bats"])
    df["team_box_hr_per_ab"] = _safe_div(df["home_runs"], df["at_bats"])
    df["team_box_xbh_per_hit"] = _safe_div(df["team_box_xbh"], df["hits"])
    df["team_box_sb_attempts"] = df["stolen_bases"] + df["caught_stealing"]
    df["team_box_sb_success_rate"] = _safe_div(df["stolen_bases"], df["team_box_sb_attempts"])
    df["team_box_lob_per_run"] = _safe_div(df["left_on_base"], df["runs"].replace({0: np.nan}))

    stat_cols = [
        "runs", "hits", "home_runs", "walks", "strikeouts", "left_on_base", "avg", "obp", "slg", "ops",
        "team_box_xbh", "team_box_bb_per_ab", "team_box_k_per_ab", "team_box_hr_per_ab",
        "team_box_xbh_per_hit", "team_box_sb_attempts", "team_box_sb_success_rate", "team_box_lob_per_run",
    ]

    outputs = []
    for team_id, g in df.dropna(subset=["team_id", "game_datetime_utc"]).groupby("team_id", sort=False, dropna=False):
        g = g.sort_values(["game_datetime_utc", "game_pk"]).copy()
        g["team_id"] = team_id
        g["team_box_games_to_date"] = np.arange(1, len(g) + 1, dtype=float)
        new_cols = {}
        for col in stat_cols:
            if col not in g.columns:
                continue
            source = g[col]
            clean_name = col.replace("team_box_", "")
            new_cols[f"team_box_{clean_name}_season_to_date"] = source.expanding(min_periods=1).mean()
            for w in TEAM_BOX_WINDOWS:
                new_cols[f"team_box_{clean_name}_last{w}"] = source.rolling(window=w, min_periods=1).mean()
        g = pd.concat([g, pd.DataFrame(new_cols, index=g.index)], axis=1).copy()
        outputs.append(g)

    if not outputs:
        return pd.DataFrame()
    out = pd.concat(outputs, ignore_index=True)
    keep = ["game_pk", "game_datetime_utc", "team_id", "team_box_games_to_date"]
    keep += [c for c in out.columns if c.startswith("team_box_") and c not in keep]
    keep = [c for c in keep if c in out.columns]
    return out[keep].sort_values(["team_id", "game_datetime_utc", "game_pk"]).reset_index(drop=True)


def build_team_boxscore_feature_frame(games: pd.DataFrame, team_stats: Optional[pd.DataFrame]) -> pd.DataFrame:
    if games is None or games.empty or team_stats is None or team_stats.empty:
        return pd.DataFrame()
    roll = _compute_team_boxscore_rollups(team_stats)
    if roll.empty:
        return pd.DataFrame()

    games = games.copy()
    games["game_datetime_utc"] = pd.to_datetime(games["game_datetime_utc"], utc=True, errors="coerce")
    feature_cols = [c for c in roll.columns if c not in {"game_pk", "game_datetime_utc", "team_id"}]

    outputs = []
    for side in ("home", "away"):
        team_col = f"{side}_team_id"
        if team_col not in games.columns:
            continue
        target = games[["game_pk", "game_datetime_utc", team_col]].rename(columns={team_col: "team_id"})
        target["team_id"] = pd.to_numeric(target["team_id"], errors="coerce")
        merged = _asof_merge_by_key(
            target,
            roll,
            "team_id",
            "game_datetime_utc",
            feature_cols,
            allow_exact_matches=False,
        )
        merged = merged.drop(columns=["team_id", "game_datetime_utc", "history_game_datetime_utc"], errors="ignore")
        merged = _prefix_columns(merged, f"{side}_", exclude={"game_pk"})
        outputs.append(merged)

    if not outputs:
        return pd.DataFrame()
    out = outputs[0]
    for extra in outputs[1:]:
        out = out.merge(extra, on="game_pk", how="outer")

    diff_data = {}
    for col in list(out.columns):
        if not col.startswith("home_team_box_"):
            continue
        away_col = col.replace("home_", "away_", 1)
        if away_col in out.columns and pd.api.types.is_numeric_dtype(out[col]) and pd.api.types.is_numeric_dtype(out[away_col]):
            diff_data[f"diff_{col.replace('home_', '', 1)}"] = out[col] - out[away_col]
    if diff_data:
        out = pd.concat([out, pd.DataFrame(diff_data, index=out.index)], axis=1).copy()
    return out


def _compute_bullpen_rollups(pitcher_stats: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Build leakage-safe bullpen workload and performance features.

    Reliever totals are aggregated to team-game level. Rollups include current
    completed game in the history row, then target-game merge uses
    allow_exact_matches=False so training rows only see prior games.
    """
    if pitcher_stats is None or pitcher_stats.empty:
        return pd.DataFrame()
    required = {"game_pk", "team_id", "game_datetime_utc", "is_starter"}
    if not required.issubset(pitcher_stats.columns):
        LOGGER.warning("Skipping bullpen features because pitcher stats are missing columns: %s", sorted(required - set(pitcher_stats.columns)))
        return pd.DataFrame()

    df = pitcher_stats.copy()
    df["game_datetime_utc"] = pd.to_datetime(df["game_datetime_utc"], utc=True, errors="coerce")
    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce")
    df["is_starter"] = pd.to_numeric(df["is_starter"], errors="coerce")
    df = df[(df["is_starter"] != 1) & df["team_id"].notna() & df["game_datetime_utc"].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    stat_cols = ["outs_pitched", "runs", "earned_runs", "hits", "walks", "strikeouts", "home_runs", "pitches_thrown", "batters_faced"]
    for col in stat_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    agg = (
        df.groupby(["game_pk", "team_id", "game_datetime_utc"], as_index=False)[stat_cols]
        .sum(min_count=1)
        .sort_values(["team_id", "game_datetime_utc", "game_pk"])
    )
    agg["bullpen_relief_ip"] = agg["outs_pitched"] / 3.0
    agg["bullpen_era_game"] = _safe_div(agg["earned_runs"] * 27.0, agg["outs_pitched"])
    agg["bullpen_whip_game"] = _safe_div(agg["hits"] + agg["walks"], agg["bullpen_relief_ip"])
    agg["bullpen_k_per_9_game"] = _safe_div(agg["strikeouts"] * 27.0, agg["outs_pitched"])
    agg["bullpen_bb_per_9_game"] = _safe_div(agg["walks"] * 27.0, agg["outs_pitched"])
    agg["bullpen_hr_per_9_game"] = _safe_div(agg["home_runs"] * 27.0, agg["outs_pitched"])
    agg["bullpen_k_minus_bb_per_bf_game"] = _safe_div(agg["strikeouts"] - agg["walks"], agg["batters_faced"])

    perf_base_cols = ["outs_pitched", "earned_runs", "hits", "walks", "strikeouts", "home_runs", "pitches_thrown", "batters_faced", "bullpen_relief_ip"]
    outputs = []
    for team_id, g in agg.groupby("team_id", sort=False, dropna=False):
        g = g.sort_values(["game_datetime_utc", "game_pk"]).copy()
        g["team_id"] = team_id
        g["bullpen_games_to_date"] = np.arange(1, len(g) + 1, dtype=float)
        new_cols = {}
        for w in BULLPEN_WINDOWS:
            sums = g[perf_base_cols].rolling(window=w, min_periods=1).sum()
            outs = sums["outs_pitched"]
            innings = outs / 3.0
            new_cols[f"bullpen_ip_last{w}"] = sums["bullpen_relief_ip"]
            new_cols[f"bullpen_pitches_last{w}"] = sums["pitches_thrown"]
            new_cols[f"bullpen_batters_faced_last{w}"] = sums["batters_faced"]
            new_cols[f"bullpen_era_last{w}"] = _safe_div(sums["earned_runs"] * 27.0, outs)
            new_cols[f"bullpen_whip_last{w}"] = _safe_div(sums["hits"] + sums["walks"], innings)
            new_cols[f"bullpen_k_per_9_last{w}"] = _safe_div(sums["strikeouts"] * 27.0, outs)
            new_cols[f"bullpen_bb_per_9_last{w}"] = _safe_div(sums["walks"] * 27.0, outs)
            new_cols[f"bullpen_hr_per_9_last{w}"] = _safe_div(sums["home_runs"] * 27.0, outs)
            new_cols[f"bullpen_k_minus_bb_per_bf_last{w}"] = _safe_div(sums["strikeouts"] - sums["walks"], sums["batters_faced"])
        g = pd.concat([g, pd.DataFrame(new_cols, index=g.index)], axis=1).copy()
        outputs.append(g)

    if not outputs:
        return pd.DataFrame()
    out = pd.concat(outputs, ignore_index=True)
    keep = ["game_pk", "game_datetime_utc", "team_id", "bullpen_games_to_date"]
    keep += [c for c in out.columns if c.startswith("bullpen_") and c not in keep]
    keep = [c for c in keep if c in out.columns]
    return out[keep].sort_values(["team_id", "game_datetime_utc", "game_pk"]).reset_index(drop=True)


def build_bullpen_feature_frame(games: pd.DataFrame, pitcher_stats: Optional[pd.DataFrame]) -> pd.DataFrame:
    if games is None or games.empty or pitcher_stats is None or pitcher_stats.empty:
        return pd.DataFrame()
    roll = _compute_bullpen_rollups(pitcher_stats)
    if roll.empty:
        return pd.DataFrame()

    games = games.copy()
    games["game_datetime_utc"] = pd.to_datetime(games["game_datetime_utc"], utc=True, errors="coerce")
    feature_cols = [c for c in roll.columns if c not in {"game_pk", "game_datetime_utc", "team_id"}]
    outputs = []
    for side in ("home", "away"):
        team_col = f"{side}_team_id"
        if team_col not in games.columns:
            continue
        target = games[["game_pk", "game_datetime_utc", team_col]].rename(columns={team_col: "team_id"})
        target["team_id"] = pd.to_numeric(target["team_id"], errors="coerce")
        merged = _asof_merge_by_key(
            target,
            roll,
            "team_id",
            "game_datetime_utc",
            feature_cols,
            allow_exact_matches=False,
        )
        merged = merged.drop(columns=["team_id", "game_datetime_utc", "history_game_datetime_utc"], errors="ignore")
        merged = _prefix_columns(merged, f"{side}_", exclude={"game_pk"})
        outputs.append(merged)

    if not outputs:
        return pd.DataFrame()
    out = outputs[0]
    for extra in outputs[1:]:
        out = out.merge(extra, on="game_pk", how="outer")

    diff_data = {}
    for col in list(out.columns):
        if not col.startswith("home_bullpen_"):
            continue
        away_col = col.replace("home_", "away_", 1)
        if away_col in out.columns and pd.api.types.is_numeric_dtype(out[col]) and pd.api.types.is_numeric_dtype(out[away_col]):
            diff_data[f"diff_{col.replace('home_', '', 1)}"] = out[col] - out[away_col]
    if diff_data:
        out = pd.concat([out, pd.DataFrame(diff_data, index=out.index)], axis=1).copy()
    return out


def build_starter_feature_frame(games: pd.DataFrame, pitcher_stats: Optional[pd.DataFrame]) -> pd.DataFrame:
    if pitcher_stats is None or pitcher_stats.empty:
        return pd.DataFrame()
    roll = _compute_pitcher_starter_rollups(pitcher_stats)
    if roll.empty:
        return pd.DataFrame()
    games = games.copy()
    games["game_datetime_utc"] = pd.to_datetime(games["game_datetime_utc"], utc=True, errors="coerce")
    feature_cols = [c for c in roll.columns if c not in {"game_pk", "game_datetime_utc", "pitcher_id"}]
    outputs = []
    for side in ("home", "away"):
        pitcher_col = f"probable_{side}_pitcher_id"
        if pitcher_col not in games.columns:
            continue
        target = games[["game_pk", "game_datetime_utc", pitcher_col]].rename(columns={pitcher_col: "pitcher_id"})
        target["pitcher_id"] = pd.to_numeric(target["pitcher_id"], errors="coerce")
        merged = _asof_merge_by_key(target, roll, "pitcher_id", "game_datetime_utc", feature_cols)
        merged = merged.drop(columns=["pitcher_id", "game_datetime_utc", "history_game_datetime_utc"], errors="ignore")
        merged = _prefix_columns(merged, f"{side}_", exclude={"game_pk"})
        merged = merged.rename(columns={
            f"{side}_pitcher_name": f"{side}_starter_pitcher_name",
            f"{side}_pitcher_hand": f"{side}_starter_pitcher_hand",
        })
        outputs.append(merged)
    if not outputs:
        return pd.DataFrame()
    out = outputs[0]
    for extra in outputs[1:]:
        out = out.merge(extra, on="game_pk", how="outer")

    # Starter matchup differentials. Build all diff columns at once to avoid pandas fragmentation warnings.
    diff_data = {}
    for col in list(out.columns):
        if not col.startswith("home_starter_"):
            continue
        away_col = col.replace("home_", "away_", 1)
        if away_col in out.columns and pd.api.types.is_numeric_dtype(out[col]) and pd.api.types.is_numeric_dtype(out[away_col]):
            diff_data[f"diff_{col.replace('home_', '', 1)}"] = out[col] - out[away_col]
    if diff_data:
        out = pd.concat([out, pd.DataFrame(diff_data, index=out.index)], axis=1).copy()
    return out



def _get_actual_starter_hands(pitcher_stats: pd.DataFrame) -> pd.DataFrame:
    if pitcher_stats is None or pitcher_stats.empty:
        return pd.DataFrame()
    st = pitcher_stats[pd.to_numeric(pitcher_stats.get("is_starter", 0), errors="coerce") == 1].copy()
    if st.empty:
        return pd.DataFrame()
    st = st[["game_pk", "is_home", "pitcher_id", "pitcher_hand"]].copy()
    home = st[st["is_home"] == 1][["game_pk", "pitcher_hand"]].rename(columns={"pitcher_hand": "home_actual_starter_hand"})
    away = st[st["is_home"] == 0][["game_pk", "pitcher_hand"]].rename(columns={"pitcher_hand": "away_actual_starter_hand"})
    return home.merge(away, on="game_pk", how="outer")


def _compute_team_vs_hand_rollups(team_stats: pd.DataFrame, games: pd.DataFrame, pitcher_stats: pd.DataFrame) -> pd.DataFrame:
    """Build leakage-safe rolling team hitting stats against opponent starter handedness.

    This avoids ``groupby().apply()`` because some pandas versions drop grouping
    columns after apply, which caused KeyError for team_id/opp_starter_hand.
    """
    if team_stats is None or team_stats.empty or pitcher_stats is None or pitcher_stats.empty:
        return pd.DataFrame()

    required_cols = {"game_pk", "game_datetime_utc", "team_id", "is_home"}
    if not required_cols.issubset(set(team_stats.columns)):
        missing = sorted(required_cols - set(team_stats.columns))
        LOGGER.warning("Skipping team-vs-hand features because team stats are missing columns: %s", missing)
        return pd.DataFrame()

    hands = _get_actual_starter_hands(pitcher_stats)
    if hands.empty:
        return pd.DataFrame()

    ts = team_stats.copy()
    ts["game_datetime_utc"] = pd.to_datetime(ts["game_datetime_utc"], utc=True, errors="coerce")
    ts["team_id"] = pd.to_numeric(ts["team_id"], errors="coerce")
    ts["is_home"] = pd.to_numeric(ts["is_home"], errors="coerce")
    ts = ts.merge(hands, on="game_pk", how="left")
    ts["opp_starter_hand"] = np.where(ts["is_home"] == 1, ts["away_actual_starter_hand"], ts["home_actual_starter_hand"])
    ts["opp_starter_hand"] = ts["opp_starter_hand"].astype("string")
    ts = ts[ts["opp_starter_hand"].isin(["L", "R"])].copy()
    ts = ts[ts["team_id"].notna()].copy()
    if ts.empty:
        return pd.DataFrame()

    numeric_source_cols = ["runs", "hits", "home_runs", "walks", "strikeouts", "at_bats", "ops", "obp", "slg"]
    for col in numeric_source_cols:
        if col not in ts.columns:
            ts[col] = np.nan
        ts[col] = pd.to_numeric(ts[col], errors="coerce")

    ts["bb_per_ab"] = _safe_div(ts["walks"], ts["at_bats"])
    ts["k_per_ab"] = _safe_div(ts["strikeouts"], ts["at_bats"])
    stat_cols = ["runs", "home_runs", "bb_per_ab", "k_per_ab", "ops", "obp", "slg"]

    outputs = []
    for (team_id, hand), g in ts.groupby(["team_id", "opp_starter_hand"], dropna=False, sort=False):
        if pd.isna(team_id) or hand not in {"L", "R"}:
            continue
        g = g.sort_values(["game_datetime_utc", "game_pk"]).copy()
        g["team_id"] = team_id
        g["opp_starter_hand"] = hand
        g["team_vs_hand_games_to_date"] = pd.Series(1.0, index=g.index).shift(1).expanding(min_periods=1).sum()

        new_cols = {}
        for col in stat_cols:
            shifted = g[col].shift(1)
            new_cols[f"team_vs_hand_{col}_season_to_date"] = shifted.expanding(min_periods=1).mean()
            for window in TEAM_VS_HAND_WINDOWS:
                new_cols[f"team_vs_hand_{col}_last{window}"] = shifted.rolling(window=window, min_periods=1).mean()
        if new_cols:
            g = pd.concat([g, pd.DataFrame(new_cols, index=g.index)], axis=1).copy()
        outputs.append(g)

    if not outputs:
        return pd.DataFrame()

    out = pd.concat(outputs, ignore_index=True)
    keep = ["game_pk", "game_datetime_utc", "team_id", "opp_starter_hand", "team_vs_hand_games_to_date"]
    keep += [c for c in out.columns if c.startswith("team_vs_hand_") and c not in keep]
    keep = [c for c in keep if c in out.columns]

    return out[keep].sort_values(["team_id", "opp_starter_hand", "game_datetime_utc", "game_pk"]).reset_index(drop=True)


def build_team_vs_hand_feature_frame(
    games: pd.DataFrame,
    team_stats: Optional[pd.DataFrame],
    pitcher_stats: Optional[pd.DataFrame],
    starter_features: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if team_stats is None or team_stats.empty or pitcher_stats is None or pitcher_stats.empty:
        return pd.DataFrame()
    roll = _compute_team_vs_hand_rollups(team_stats, games, pitcher_stats)
    if roll.empty:
        return pd.DataFrame()

    games = games.copy()
    games["game_datetime_utc"] = pd.to_datetime(games["game_datetime_utc"], utc=True, errors="coerce")
    required_cols = ["game_pk", "game_datetime_utc", "home_team_id", "away_team_id"]
    if not set(required_cols).issubset(games.columns):
        missing = sorted(set(required_cols) - set(games.columns))
        LOGGER.warning("Skipping team-vs-hand features because games are missing columns: %s", missing)
        return pd.DataFrame()
    target = games[required_cols].copy()

    if starter_features is not None and not starter_features.empty:
        hand_cols = [c for c in ["game_pk", "home_starter_pitcher_hand", "away_starter_pitcher_hand"] if c in starter_features.columns]
        if len(hand_cols) > 1:
            target = target.merge(starter_features[hand_cols], on="game_pk", how="left")

    # Fall back to actual starter hands for completed games.
    actual_hands = _get_actual_starter_hands(pitcher_stats)
    if not actual_hands.empty:
        target = target.merge(actual_hands, on="game_pk", how="left")

    target["home_opp_starter_hand"] = target["away_starter_pitcher_hand"] if "away_starter_pitcher_hand" in target.columns else np.nan
    target["away_opp_starter_hand"] = target["home_starter_pitcher_hand"] if "home_starter_pitcher_hand" in target.columns else np.nan
    if "away_actual_starter_hand" in target.columns:
        target["home_opp_starter_hand"] = target["home_opp_starter_hand"].fillna(target["away_actual_starter_hand"])
    if "home_actual_starter_hand" in target.columns:
        target["away_opp_starter_hand"] = target["away_opp_starter_hand"].fillna(target["home_actual_starter_hand"])

    feature_cols = [c for c in roll.columns if c not in {"game_pk", "game_datetime_utc", "team_id", "opp_starter_hand"}]
    outputs = []
    for side in ("home", "away"):
        team_col = f"{side}_team_id"
        hand_col = f"{side}_opp_starter_hand"
        tg = target[["game_pk", "game_datetime_utc", team_col, hand_col]].rename(columns={team_col: "team_id", hand_col: "opp_starter_hand"})
        tg["team_id"] = pd.to_numeric(tg["team_id"], errors="coerce")
        tg["opp_starter_hand"] = tg["opp_starter_hand"].astype("string")

        rows = []
        for (team_id, hand), g in tg.groupby(["team_id", "opp_starter_hand"], dropna=False, sort=False):
            g = g.sort_values("game_datetime_utc").copy()
            if pd.isna(team_id) or hand not in {"L", "R"}:
                for col in feature_cols + ["history_game_datetime_utc"]:
                    g[col] = np.nan
                rows.append(g)
                continue

            hist = roll[(pd.to_numeric(roll["team_id"], errors="coerce") == team_id) & (roll["opp_starter_hand"].astype("string") == hand)].sort_values("game_datetime_utc").copy()
            if hist.empty:
                for col in feature_cols + ["history_game_datetime_utc"]:
                    g[col] = np.nan
                rows.append(g)
                continue

            available_feature_cols = [c for c in feature_cols if c in hist.columns]
            hist = hist[["game_datetime_utc"] + available_feature_cols].rename(columns={"game_datetime_utc": "history_game_datetime_utc"})

            # We already filtered history to the same team and handedness, so do
            # not use merge_asof(..., by=...). This avoids dtype mismatches and
            # keeps the behavior consistent across pandas versions.
            merged = pd.merge_asof(
                g.sort_values("game_datetime_utc"),
                hist.sort_values("history_game_datetime_utc"),
                left_on="game_datetime_utc",
                right_on="history_game_datetime_utc",
                direction="backward",
                allow_exact_matches=True,
            )
            for col in feature_cols:
                if col not in merged.columns:
                    merged[col] = np.nan
            if "history_game_datetime_utc" not in merged.columns:
                merged["history_game_datetime_utc"] = np.nan
            rows.append(merged)

        side_out = pd.concat(rows, ignore_index=True) if rows else tg
        side_out[f"{side}_opp_starter_is_lhp"] = (side_out["opp_starter_hand"] == "L").astype(float)
        side_out[f"{side}_opp_starter_is_rhp"] = (side_out["opp_starter_hand"] == "R").astype(float)
        side_out = side_out.drop(columns=["game_datetime_utc", "team_id", "opp_starter_hand", "history_game_datetime_utc"], errors="ignore")
        side_out = _prefix_columns(side_out, f"{side}_", exclude={"game_pk", f"{side}_opp_starter_is_lhp", f"{side}_opp_starter_is_rhp"})
        outputs.append(side_out)

    if not outputs:
        return pd.DataFrame()
    out = outputs[0]
    for extra in outputs[1:]:
        out = out.merge(extra, on="game_pk", how="outer")

    # Team-vs-hand matchup differentials. Build all diff columns at once to avoid pandas fragmentation warnings.
    diff_data = {}
    for col in list(out.columns):
        if not col.startswith("home_team_vs_hand_"):
            continue
        away_col = col.replace("home_", "away_", 1)
        if away_col in out.columns and pd.api.types.is_numeric_dtype(out[col]) and pd.api.types.is_numeric_dtype(out[away_col]):
            diff_data[f"diff_{col.replace('home_', '', 1)}"] = out[col] - out[away_col]
    if diff_data:
        out = pd.concat([out, pd.DataFrame(diff_data, index=out.index)], axis=1).copy()
    return out



def build_game_feature_frame(
    games: pd.DataFrame,
    odds_consensus: Optional[pd.DataFrame] = None,
    include_future: bool = True,
    pitcher_stats: Optional[pd.DataFrame] = None,
    team_game_stats: Optional[pd.DataFrame] = None,
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

    elo_features = build_elo_feature_frame(games)
    if not elo_features.empty:
        frame = frame.merge(elo_features, on="game_pk", how="left")

    team_box_features = build_team_boxscore_feature_frame(games, team_game_stats)
    if not team_box_features.empty:
        frame = frame.merge(team_box_features, on="game_pk", how="left")

    bullpen_features = build_bullpen_feature_frame(games, pitcher_stats)
    if not bullpen_features.empty:
        frame = frame.merge(bullpen_features, on="game_pk", how="left")

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

    starter_features = build_starter_feature_frame(games, pitcher_stats)
    if not starter_features.empty:
        frame = frame.merge(starter_features, on="game_pk", how="left")

    team_vs_hand = build_team_vs_hand_feature_frame(games, team_game_stats, pitcher_stats, starter_features=starter_features)
    if not team_vs_hand.empty:
        frame = frame.merge(team_vs_hand, on="game_pk", how="left")

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


def _is_market_feature(col: str) -> bool:
    lower = col.lower()
    return any(keyword in lower for keyword in MARKET_KEYWORDS)


def get_model_feature_columns(
    frame: pd.DataFrame,
    include_market: bool = False,
    min_non_null_rate: float = 0.05,
) -> list[str]:
    blocked = set(POSTGAME_COLUMNS) | {
        "game_pk", "season", "game_type", "official_date", "game_datetime_utc",
        "venue_id", "venue_name", "home_team_name", "home_team_norm", "away_team_name", "away_team_norm",
        "home_team_id", "away_team_id", "home_team", "away_team",
        "probable_home_pitcher_name", "probable_away_pitcher_name",
        "probable_home_pitcher_id", "probable_away_pitcher_id",
        "event_id", "event_date", "odds_home_team", "odds_away_team",
    }
    numeric_cols = frame.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    feature_cols = []
    for c in numeric_cols:
        if c in blocked or c.endswith("_score"):
            continue
        if not include_market and _is_market_feature(c):
            continue
        if min_non_null_rate is not None and frame[c].notna().mean() < min_non_null_rate:
            continue
        feature_cols.append(c)
    return feature_cols


def save_features(frame: pd.DataFrame, output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path
