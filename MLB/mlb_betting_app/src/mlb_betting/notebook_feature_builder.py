from __future__ import annotations

"""Notebook-compatible MLB feature builder.

This module productionizes the feature engineering used by the cloud data export
and local modeling notebooks that produced the current moneyline champion model.
It intentionally preserves the notebook-era column names, including families like
``home_box_box_box_bat_*``, ``home_team_sc_team_off_sc_*``,
``home_starter_starter_statcast_*``, ``robust_*``, ``*_ewm_*``, and
``*_team_cluster_recent``.

The current production feature_engineering.py builds a different schema. Use this
module when scoring a model exported from the notebook feature list.
"""

import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


ROLLING_WINDOWS = [3, 5, 10, 20]

TEXT_COL_HINTS = [
    "team_name", "team_norm", "pitcher_name", "venue", "status", "state",
]
LEAKY_CONTAINS = [
    "target_", "actual_", "final_", "postgame", "post_game", "winner",
    "winning_team", "losing_team",
]
LEAKY_EXACT = {
    "home_score", "away_score", "home_win", "away_win", "home_run_diff",
    "away_run_diff", "target_home_win", "target_total_runs", "target_home_margin",
}
DATE_COLS = {"official_date", "game_datetime_utc", "game_date"}
RANDOM_STATE = 42


def normalize_team_name(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = s.replace(".", "")
    s = s.replace("'", "")
    s = s.replace("&", "and")
    s = " ".join(s.split())
    aliases = {
        "athletics": "oakland athletics",
        "the athletics": "oakland athletics",
        "oakland athletics": "oakland athletics",
        "as": "oakland athletics",
        "a's": "oakland athletics",
        "ath": "oakland athletics",
        "oak": "oakland athletics",
        "ari": "arizona diamondbacks",
        "az": "arizona diamondbacks",
        "atl": "atlanta braves",
        "bal": "baltimore orioles",
        "bos": "boston red sox",
        "chc": "chicago cubs",
        "chw": "chicago white sox",
        "cws": "chicago white sox",
        "cin": "cincinnati reds",
        "cle": "cleveland guardians",
        "col": "colorado rockies",
        "det": "detroit tigers",
        "hou": "houston astros",
        "kc": "kansas city royals",
        "kcr": "kansas city royals",
        "laa": "los angeles angels",
        "lad": "los angeles dodgers",
        "mia": "miami marlins",
        "mil": "milwaukee brewers",
        "min": "minnesota twins",
        "nym": "new york mets",
        "nyy": "new york yankees",
        "phi": "philadelphia phillies",
        "pit": "pittsburgh pirates",
        "sd": "san diego padres",
        "sdp": "san diego padres",
        "sf": "san francisco giants",
        "sfg": "san francisco giants",
        "sea": "seattle mariners",
        "stl": "st louis cardinals",
        "tb": "tampa bay rays",
        "tbr": "tampa bay rays",
        "tex": "texas rangers",
        "tor": "toronto blue jays",
        "wsh": "washington nationals",
        "was": "washington nationals",
        "wsn": "washington nationals",
    }
    return aliases.get(s, s)


def american_to_implied_prob(price: Any) -> float:
    if price is None or pd.isna(price):
        return np.nan
    p = float(price)
    if p > 0:
        return 100.0 / (p + 100.0)
    return abs(p) / (abs(p) + 100.0)


def no_vig_two_way_prob(price_a: Any, price_b: Any) -> tuple[float, float]:
    pa = american_to_implied_prob(price_a)
    pb = american_to_implied_prob(price_b)
    if pd.isna(pa) or pd.isna(pb) or (pa + pb) <= 0:
        return np.nan, np.nan
    return pa / (pa + pb), pb / (pa + pb)


def _first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def is_numeric_id_col(c: str) -> bool:
    c_low = c.lower()
    return c_low == "game_pk" or c_low.endswith("_id") or c_low.endswith("id")


def is_probably_leaky_col(c: str) -> bool:
    c_low = c.lower()
    if c_low in LEAKY_EXACT:
        return True
    return any(p in c_low for p in LEAKY_CONTAINS)


def coerce_numeric_cols(df: pd.DataFrame, skip: set[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    skip = skip or set()
    for c in out.columns:
        if c in skip:
            continue
        if out[c].dtype == object:
            out[c] = pd.to_numeric(out[c].astype(str).str.replace("%", "", regex=False), errors="ignore")
    return out


def normalize_games_frame(games_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize either notebook raw schedule rows or production mlb_games rows."""
    g = games_df.copy()
    if g.empty:
        return g
    g = g.loc[:, ~g.columns.duplicated(keep="first")].copy()

    rename = {}
    if "probable_home_pitcher_id" in g.columns and "home_probable_pitcher_id" not in g.columns:
        rename["probable_home_pitcher_id"] = "home_probable_pitcher_id"
    if "probable_home_pitcher_name" in g.columns and "home_probable_pitcher_name" not in g.columns:
        rename["probable_home_pitcher_name"] = "home_probable_pitcher_name"
    if "probable_away_pitcher_id" in g.columns and "away_probable_pitcher_id" not in g.columns:
        rename["probable_away_pitcher_id"] = "away_probable_pitcher_id"
    if "probable_away_pitcher_name" in g.columns and "away_probable_pitcher_name" not in g.columns:
        rename["probable_away_pitcher_name"] = "away_probable_pitcher_name"
    if rename:
        g = g.rename(columns=rename)

    if "game_pk" in g.columns:
        g["game_pk"] = pd.to_numeric(g["game_pk"], errors="coerce")
    if "official_date" in g.columns:
        g["official_date"] = pd.to_datetime(g["official_date"], errors="coerce")
    if "game_datetime_utc" in g.columns:
        g["game_datetime_utc"] = pd.to_datetime(g["game_datetime_utc"], errors="coerce", utc=True)
    elif "official_date" in g.columns:
        g["game_datetime_utc"] = pd.to_datetime(g["official_date"], errors="coerce", utc=True)

    for c in [
        "home_score", "away_score", "home_team_id", "away_team_id",
        "home_probable_pitcher_id", "away_probable_pitcher_id",
    ]:
        if c in g.columns:
            g[c] = pd.to_numeric(g[c], errors="coerce")

    if "home_team_norm" not in g.columns and "home_team_name" in g.columns:
        g["home_team_norm"] = g["home_team_name"].apply(normalize_team_name)
    if "away_team_norm" not in g.columns and "away_team_name" in g.columns:
        g["away_team_norm"] = g["away_team_name"].apply(normalize_team_name)

    if "is_final" not in g.columns:
        abstract = g.get("abstract_state", pd.Series(index=g.index, dtype=object)).astype(str).str.lower()
        detailed = g.get("detailed_state", pd.Series(index=g.index, dtype=object)).astype(str).str.lower()
        g["is_final"] = abstract.eq("final") | detailed.isin(["final", "completed early", "game over"])
        if "target_home_win" in g.columns:
            g["is_final"] = g["is_final"] | g["target_home_win"].notna()

    if "target_home_win" not in g.columns and {"home_score", "away_score"}.issubset(g.columns):
        g["target_home_win"] = np.where(
            g["is_final"] & g["home_score"].notna() & g["away_score"].notna(),
            (g["home_score"] > g["away_score"]).astype(float),
            np.nan,
        )
    if "target_total_runs" not in g.columns and {"home_score", "away_score"}.issubset(g.columns):
        g["target_total_runs"] = np.where(g["is_final"], g["home_score"] + g["away_score"], np.nan)
    if "target_home_margin" not in g.columns and {"home_score", "away_score"}.issubset(g.columns):
        g["target_home_margin"] = np.where(g["is_final"], g["home_score"] - g["away_score"], np.nan)

    return g.sort_values([c for c in ["official_date", "game_datetime_utc", "game_pk"] if c in g.columns]).reset_index(drop=True)


def _ensure_game_time_columns(d: pd.DataFrame, name: str = "df") -> pd.DataFrame:
    if d is None or d.empty:
        return pd.DataFrame() if d is None else d.copy()
    out = d.copy()
    if not out.columns.is_unique:
        dedup = pd.DataFrame(index=out.index)
        for col in pd.Index(out.columns).unique():
            same = out.loc[:, out.columns == col]
            if same.shape[1] == 1:
                dedup[col] = same.iloc[:, 0]
            else:
                s = same.iloc[:, 0]
                for j in range(1, same.shape[1]):
                    s = s.combine_first(same.iloc[:, j])
                dedup[col] = s
        out = dedup
    if "official_date" in out.columns:
        out["official_date"] = pd.to_datetime(out["official_date"], errors="coerce")
    elif "game_date" in out.columns:
        out["official_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    if "game_datetime_utc" in out.columns:
        out["game_datetime_utc"] = pd.to_datetime(out["game_datetime_utc"], errors="coerce", utc=True)
    elif "official_date" in out.columns:
        out["game_datetime_utc"] = pd.to_datetime(out["official_date"], errors="coerce", utc=True)
    else:
        raise KeyError(f"{name} is missing both game_datetime_utc and official_date. Columns: {list(out.columns)}")
    if "game_pk" in out.columns:
        out["game_pk"] = pd.to_numeric(out["game_pk"], errors="coerce")
    return out


def add_basic_game_outcome_features(games_df: pd.DataFrame) -> pd.DataFrame:
    g = normalize_games_frame(games_df)
    if "is_final" not in g.columns:
        g["is_final"] = g.get("target_home_win", pd.Series(index=g.index)).notna()
    g["home_win"] = np.where(g["is_final"], (g["home_score"] > g["away_score"]).astype(float), np.nan)
    g["away_win"] = np.where(g["is_final"], (g["away_score"] > g["home_score"]).astype(float), np.nan)
    g["home_run_diff"] = np.where(g["is_final"], g["home_score"] - g["away_score"], np.nan)
    g["away_run_diff"] = np.where(g["is_final"], g["away_score"] - g["home_score"], np.nan)
    return g


def team_game_long_from_games(games_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in games_df.iterrows():
        if not bool(r.get("is_final")):
            continue
        for side in ["home", "away"]:
            opp = "away" if side == "home" else "home"
            runs_for = r.get(f"{side}_score")
            runs_against = r.get(f"{opp}_score")
            rows.append({
                "game_pk": r.get("game_pk"),
                "official_date": r.get("official_date"),
                "game_datetime_utc": r.get("game_datetime_utc"),
                "team_side": side,
                "team_id": r.get(f"{side}_team_id"),
                "team_name": r.get(f"{side}_team_name"),
                "team_norm": r.get(f"{side}_team_norm"),
                "opponent_team_id": r.get(f"{opp}_team_id"),
                "opponent_team_name": r.get(f"{opp}_team_name"),
                "runs_for": runs_for,
                "runs_against": runs_against,
                "win": float(runs_for > runs_against) if pd.notna(runs_for) and pd.notna(runs_against) else np.nan,
                "run_diff": runs_for - runs_against if pd.notna(runs_for) and pd.notna(runs_against) else np.nan,
            })
    return pd.DataFrame(rows)


def add_rolling_entity_features(
    long_df: pd.DataFrame,
    entity_col: str,
    value_cols: list[str],
    prefix: str,
    windows: list[int] = ROLLING_WINDOWS,
    season: bool = True,
) -> pd.DataFrame:
    if long_df is None or long_df.empty:
        return pd.DataFrame()
    d = _ensure_game_time_columns(long_df, f"rolling_input_{prefix}")
    if entity_col not in d.columns:
        raise KeyError(f"{prefix}: missing entity column {entity_col}. Columns: {list(d.columns)}")
    d = d.sort_values([entity_col, "official_date", "game_datetime_utc", "game_pk"]).reset_index(drop=True)
    out = d[["game_pk", entity_col]].copy()
    for col in value_cols:
        if col not in d.columns:
            continue
        d[col] = pd.to_numeric(d[col], errors="coerce")
        shifted = d.groupby(entity_col)[col].shift(1)
        if season:
            out[f"{prefix}_{col}_season_to_date"] = shifted.groupby(d[entity_col]).expanding(min_periods=1).mean().reset_index(level=0, drop=True)
        for w in windows:
            out[f"{prefix}_{col}_last{w}"] = shifted.groupby(d[entity_col]).rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
    return pd.concat(
        [
            d[["game_pk", entity_col, "official_date", "game_datetime_utc"]].reset_index(drop=True),
            out.drop(columns=["game_pk", entity_col], errors="ignore").reset_index(drop=True),
        ],
        axis=1,
    )


def compute_elo_features(games_df: pd.DataFrame, k: float = 20.0, home_adv: float = 35.0, base_elo: float = 1500.0) -> pd.DataFrame:
    g = games_df.copy().sort_values(["official_date", "game_datetime_utc", "game_pk"])
    ratings: dict[int, float] = {}
    rows = []
    for _, r in g.iterrows():
        h = int(r["home_team_id"]) if pd.notna(r.get("home_team_id")) else None
        a = int(r["away_team_id"]) if pd.notna(r.get("away_team_id")) else None
        if h is None or a is None:
            continue
        rh = ratings.get(h, base_elo)
        ra = ratings.get(a, base_elo)
        ph = 1.0 / (1.0 + 10 ** (-((rh + home_adv) - ra) / 400.0))
        rows.append({"game_pk": r.get("game_pk"), "home_elo_pre": rh, "away_elo_pre": ra, "diff_elo_pre": rh - ra, "elo_home_win_prob": ph})
        if bool(r.get("is_final")) and pd.notna(r.get("home_score")) and pd.notna(r.get("away_score")):
            outcome = 1.0 if r["home_score"] > r["away_score"] else 0.0
            change = k * (outcome - ph)
            ratings[h] = rh + change
            ratings[a] = ra - change
    return pd.DataFrame(rows)


def rolling_team_features_from_games(games_df: pd.DataFrame) -> pd.DataFrame:
    long = team_game_long_from_games(games_df)
    if long.empty:
        return pd.DataFrame()
    value_cols = ["runs_for", "runs_against", "win", "run_diff"]
    return add_rolling_entity_features(long, "team_id", value_cols, "team")


def rolling_boxscore_features(box_df: pd.DataFrame) -> pd.DataFrame:
    if box_df is None or box_df.empty:
        return pd.DataFrame()
    d = coerce_numeric_cols(box_df, skip={"team_name", "team_norm", "opponent_team_name", "team_side"})
    value_cols: list[str] = []
    for c in d.columns:
        if c.startswith("box_bat_") or c.startswith("box_pitch_") or c in ["runs_for", "runs_against"]:
            if pd.api.types.is_numeric_dtype(d[c]):
                value_cols.append(c)
    # Keep the notebook cap because the champion feature names were exported from that notebook.
    value_cols = value_cols[:80]
    return add_rolling_entity_features(d, "team_id", value_cols, "box")


def pitch_family(pt: Any) -> str:
    if pt is None or pd.isna(pt):
        return "unknown"
    pt = str(pt).upper()
    fast = {"FF", "SI", "FC", "FA", "FS"}
    breaking = {"SL", "CU", "KC", "SV", "ST"}
    offspeed = {"CH", "FS", "FO", "SC"}
    if pt in fast:
        return "fastball"
    if pt in breaking:
        return "breaking"
    if pt in offspeed:
        return "offspeed"
    return "other"


def entropy_from_counts(counts: pd.Series) -> float:
    values = counts[counts > 0].astype(float)
    if values.sum() <= 0:
        return np.nan
    p = values / values.sum()
    return float(-(p * np.log(p)).sum())


def prepare_statcast_pitch_level(sc: pd.DataFrame) -> pd.DataFrame:
    if sc is None or sc.empty:
        return pd.DataFrame()
    d = sc.copy()
    if not d.columns.is_unique:
        out = pd.DataFrame(index=d.index)
        for col in pd.Index(d.columns).unique():
            same = d.loc[:, d.columns == col]
            if same.shape[1] == 1:
                out[col] = same.iloc[:, 0]
            else:
                s = same.iloc[:, 0]
                for j in range(1, same.shape[1]):
                    s = s.combine_first(same.iloc[:, j])
                out[col] = s
        d = out
    if "official_date" in d.columns:
        d["official_date"] = pd.to_datetime(d["official_date"], errors="coerce")
    elif "game_date" in d.columns:
        d["official_date"] = pd.to_datetime(d["game_date"], errors="coerce")
    if "game_datetime_utc" in d.columns:
        d["game_datetime_utc"] = pd.to_datetime(d["game_datetime_utc"], errors="coerce", utc=True)
    elif "official_date" in d.columns:
        d["game_datetime_utc"] = pd.to_datetime(d["official_date"], errors="coerce", utc=True)
    if "game_pk" in d.columns:
        d["game_pk"] = pd.to_numeric(d["game_pk"], errors="coerce")
    for c in [
        "release_speed", "release_spin_rate", "release_extension", "launch_speed", "launch_angle",
        "estimated_woba_using_speedangle", "woba_value", "estimated_ba_using_speedangle",
    ]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    if {"inning_topbot", "home_team", "away_team"}.issubset(d.columns):
        is_top = d["inning_topbot"].astype(str).str.lower().eq("top")
        d["bat_team"] = np.where(is_top, d["away_team"], d["home_team"])
        d["pitch_team"] = np.where(is_top, d["home_team"], d["away_team"])
    d["bat_team_norm"] = d.get("bat_team", pd.Series(index=d.index, dtype=object)).apply(normalize_team_name)
    d["pitch_team_norm"] = d.get("pitch_team", pd.Series(index=d.index, dtype=object)).apply(normalize_team_name)
    desc = d.get("description", pd.Series("", index=d.index)).fillna("").astype(str)
    events = d.get("events", pd.Series("", index=d.index)).fillna("").astype(str)
    d["is_pa_event"] = events.ne("")
    d["is_strikeout"] = events.str.contains("strikeout", case=False, na=False)
    d["is_walk"] = events.str.contains("walk", case=False, na=False) & ~events.str.contains("intent", case=False, na=False)
    d["is_home_run"] = events.str.contains("home_run", case=False, na=False)
    d["is_batted_ball"] = d.get("launch_speed", pd.Series(np.nan, index=d.index)).notna()
    d["is_hard_hit"] = d.get("launch_speed", pd.Series(np.nan, index=d.index)).ge(95)
    d["is_sweetspot"] = d.get("launch_angle", pd.Series(np.nan, index=d.index)).between(8, 32)
    d["is_whiff"] = desc.isin(["swinging_strike", "swinging_strike_blocked", "foul_tip"])
    d["is_called_strike"] = desc.eq("called_strike")
    d["is_swing"] = desc.str.contains("swing|foul|hit_into_play", case=False, regex=True, na=False)
    d["pitch_family"] = d.get("pitch_type", pd.Series("UNK", index=d.index)).fillna("UNK").map(pitch_family)
    return d


def aggregate_statcast_team_game(sc: pd.DataFrame) -> pd.DataFrame:
    if sc is None or sc.empty:
        return pd.DataFrame()
    d = prepare_statcast_pitch_level(sc)
    d = _ensure_game_time_columns(d, "prepared_statcast_team")
    group_cols = ["game_pk", "official_date", "game_datetime_utc", "bat_team_norm"]
    agg = d.groupby(group_cols).agg(
        sc_pitches_seen=("game_pk", "size"),
        sc_pa=("is_pa_event", "sum"),
        sc_avg_ev=("launch_speed", "mean"),
        sc_max_ev=("launch_speed", "max"),
        sc_avg_la=("launch_angle", "mean"),
        sc_hard_hit_rate=("is_hard_hit", "mean"),
        sc_sweetspot_rate=("is_sweetspot", "mean"),
        sc_xwoba_contact=("estimated_woba_using_speedangle", "mean"),
        sc_woba=("woba_value", "mean"),
        sc_k_rate=("is_strikeout", "mean"),
        sc_bb_rate=("is_walk", "mean"),
        sc_hr_rate=("is_home_run", "mean"),
        sc_whiff_rate=("is_whiff", "mean"),
        sc_swing_rate=("is_swing", "mean"),
    ).reset_index().rename(columns={"bat_team_norm": "team_norm"})
    csw = (
        d.assign(csw=d["is_called_strike"] | d["is_whiff"])
        .groupby(group_cols)["csw"]
        .mean()
        .reset_index(name="sc_csw_rate")
        .rename(columns={"bat_team_norm": "team_norm"})
    )
    agg = agg.merge(csw, on=["game_pk", "official_date", "game_datetime_utc", "team_norm"], how="left")
    return agg


def aggregate_statcast_pitcher_game(sc: pd.DataFrame) -> pd.DataFrame:
    if sc is None or sc.empty:
        return pd.DataFrame()
    d = prepare_statcast_pitch_level(sc)
    d = _ensure_game_time_columns(d, "prepared_statcast_pitcher")
    if "pitcher" not in d.columns:
        return pd.DataFrame()
    group_cols = ["game_pk", "official_date", "game_datetime_utc", "pitcher"]
    agg = d.groupby(group_cols).agg(
        sc_pitches=("game_pk", "size"),
        sc_avg_velo=("release_speed", "mean"),
        sc_max_velo=("release_speed", "max"),
        sc_avg_spin=("release_spin_rate", "mean"),
        sc_avg_extension=("release_extension", "mean"),
        sc_whiff_rate=("is_whiff", "mean"),
        sc_called_strike_rate=("is_called_strike", "mean"),
        sc_swing_rate=("is_swing", "mean"),
        sc_hard_hit_allowed=("is_hard_hit", "mean"),
        sc_xwoba_allowed=("estimated_woba_using_speedangle", "mean"),
        sc_woba_allowed=("woba_value", "mean"),
        sc_k_rate=("is_strikeout", "mean"),
        sc_bb_rate=("is_walk", "mean"),
        sc_hr_rate=("is_home_run", "mean"),
    ).reset_index().rename(columns={"pitcher": "pitcher_id"})
    ent = (
        d.groupby(group_cols + ["pitch_family"]).size()
        .reset_index(name="n")
        .groupby(group_cols)["n"]
        .apply(entropy_from_counts)
        .reset_index(name="sc_pitchmix_entropy")
        .rename(columns={"pitcher": "pitcher_id"})
    )
    agg = agg.merge(ent, on=["game_pk", "official_date", "game_datetime_utc", "pitcher_id"], how="left")
    return agg


def aggregate_statcast_pitch_type(sc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if sc is None or sc.empty:
        return pd.DataFrame(), pd.DataFrame()
    d = prepare_statcast_pitch_level(sc)
    d = _ensure_game_time_columns(d, "prepared_statcast_pitch_type")
    team_pt = d.groupby(["game_pk", "official_date", "game_datetime_utc", "bat_team_norm", "pitch_family"]).agg(
        pitches=("game_pk", "size"),
        avg_ev=("launch_speed", "mean"),
        woba=("woba_value", "mean"),
        whiff_rate=("is_whiff", "mean"),
        hard_hit_rate=("is_hard_hit", "mean"),
    ).reset_index().rename(columns={"bat_team_norm": "team_norm"})
    if "pitcher" in d.columns:
        pit_pt = d.groupby(["game_pk", "official_date", "game_datetime_utc", "pitcher", "pitch_family"]).agg(
            pitches=("game_pk", "size"), avg_velo=("release_speed", "mean"), whiff_rate=("is_whiff", "mean"), usage=("game_pk", "size")
        ).reset_index().rename(columns={"pitcher": "pitcher_id"})
    else:
        pit_pt = pd.DataFrame()
    return team_pt, pit_pt


def rolling_statcast_team_features(sc_team_game: pd.DataFrame) -> pd.DataFrame:
    if sc_team_game is None or sc_team_game.empty:
        return pd.DataFrame()
    value_cols = [c for c in sc_team_game.columns if c.startswith("sc_")]
    return add_rolling_entity_features(sc_team_game, "team_norm", value_cols, "team_off")


def rolling_statcast_pitcher_features(sc_pitcher_game: pd.DataFrame) -> pd.DataFrame:
    if sc_pitcher_game is None or sc_pitcher_game.empty:
        return pd.DataFrame()
    value_cols = [c for c in sc_pitcher_game.columns if c.startswith("sc_")]
    return add_rolling_entity_features(sc_pitcher_game, "pitcher_id", value_cols, "starter_statcast")


def rolling_pitchmix_team_features(team_pt: pd.DataFrame) -> pd.DataFrame:
    if team_pt is None or team_pt.empty:
        return pd.DataFrame()
    d = _ensure_game_time_columns(team_pt, "team_pitchmix")
    piv = d.pivot_table(
        index=["game_pk", "official_date", "game_datetime_utc", "team_norm"],
        columns="pitch_family",
        values=["pitches", "avg_ev", "woba", "whiff_rate", "hard_hit_rate"],
        aggfunc="mean",
    )
    piv.columns = [f"pt_{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index()
    value_cols = [c for c in piv.columns if c.startswith("pt_")]
    return add_rolling_entity_features(piv, "team_norm", value_cols, "team_pitchmix")


def merge_home_away_team_features(base: pd.DataFrame, feat: pd.DataFrame, entity_col: str, prefix: str, id_home_col: str, id_away_col: str) -> pd.DataFrame:
    if feat is None or feat.empty:
        return base
    d = base.copy()
    feature_cols = [c for c in feat.columns if c not in {"game_pk", entity_col, "official_date", "game_datetime_utc"}]
    latest = feat[["game_pk", entity_col] + feature_cols].drop_duplicates(["game_pk", entity_col], keep="last")
    home = latest.rename(columns={entity_col: id_home_col, **{c: f"home_{prefix}_{c}" for c in feature_cols}})
    away = latest.rename(columns={entity_col: id_away_col, **{c: f"away_{prefix}_{c}" for c in feature_cols}})
    d = d.merge(home, on=["game_pk", id_home_col], how="left")
    d = d.merge(away, on=["game_pk", id_away_col], how="left")
    diff_data = {}
    for c in feature_cols:
        hc = f"home_{prefix}_{c}"
        ac = f"away_{prefix}_{c}"
        if hc in d.columns and ac in d.columns:
            diff_data[f"diff_{prefix}_{c}"] = d[hc] - d[ac]
    if diff_data:
        d = pd.concat([d, pd.DataFrame(diff_data, index=d.index)], axis=1)
    return d


def merge_home_away_starter_features(base: pd.DataFrame, sp_feat: pd.DataFrame) -> pd.DataFrame:
    if sp_feat is None or sp_feat.empty:
        return base
    d = base.copy()
    feature_cols = [c for c in sp_feat.columns if c not in {"game_pk", "pitcher_id", "official_date", "game_datetime_utc"}]
    latest = sp_feat[["game_pk", "pitcher_id"] + feature_cols].drop_duplicates(["game_pk", "pitcher_id"], keep="last")
    home = latest.rename(columns={"pitcher_id": "home_probable_pitcher_id", **{c: f"home_starter_{c}" for c in feature_cols}})
    away = latest.rename(columns={"pitcher_id": "away_probable_pitcher_id", **{c: f"away_starter_{c}" for c in feature_cols}})
    d = d.merge(home, on=["game_pk", "home_probable_pitcher_id"], how="left")
    d = d.merge(away, on=["game_pk", "away_probable_pitcher_id"], how="left")
    diff_data = {}
    for c in feature_cols:
        hc = f"home_starter_{c}"
        ac = f"away_starter_{c}"
        if hc in d.columns and ac in d.columns:
            diff_data[f"diff_starter_{c}"] = d[hc] - d[ac]
    if diff_data:
        d = pd.concat([d, pd.DataFrame(diff_data, index=d.index)], axis=1)
    return d


def rolling_starter_boxscore_features(box_pitcher_df: pd.DataFrame, windows: list[int] = [1, 3, 5, 10]) -> pd.DataFrame:
    if box_pitcher_df is None or box_pitcher_df.empty:
        return pd.DataFrame()
    d = box_pitcher_df.copy()
    d = _ensure_game_time_columns(d, "box_pitcher_game")
    if "is_starting_pitcher" in d.columns:
        d = d[pd.to_numeric(d["is_starting_pitcher"], errors="coerce").fillna(0).astype(int).eq(1)].copy()
    if d.empty:
        return pd.DataFrame()
    for c in ["pitcher_id", "p_ip_float", "p_hits", "p_earned_runs", "p_base_on_balls", "p_strikeouts", "p_home_runs", "p_pitches_thrown"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        else:
            d[c] = np.nan
    d = d.sort_values(["pitcher_id", "official_date", "game_datetime_utc", "game_pk"]).reset_index(drop=True)
    out = d[["game_pk", "pitcher_id", "official_date", "game_datetime_utc"]].copy()
    grouped = d.groupby("pitcher_id", group_keys=False)
    shifted = pd.DataFrame(index=d.index)
    for c in ["p_ip_float", "p_hits", "p_earned_runs", "p_base_on_balls", "p_strikeouts", "p_home_runs", "p_pitches_thrown"]:
        shifted[c] = grouped[c].shift(1)
    for w in windows:
        label = "last_outing" if w == 1 else f"rolling{w}"
        sums = {}
        for c in shifted.columns:
            sums[c] = shifted.groupby(d["pitcher_id"])[c].rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
        ip = sums["p_ip_float"].replace(0, np.nan)
        out[f"starter_box_{label}_ip"] = ip
        out[f"starter_box_{label}_era"] = 9.0 * sums["p_earned_runs"] / ip
        out[f"starter_box_{label}_whip"] = (sums["p_hits"] + sums["p_base_on_balls"]) / ip
        out[f"starter_box_{label}_k_per_9"] = 9.0 * sums["p_strikeouts"] / ip
        out[f"starter_box_{label}_bb_per_9"] = 9.0 * sums["p_base_on_balls"] / ip
        out[f"starter_box_{label}_hr_per_9"] = 9.0 * sums["p_home_runs"] / ip
        out[f"starter_box_{label}_avg_pitches"] = shifted.groupby(d["pitcher_id"])["p_pitches_thrown"].rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
    return out


def merge_home_away_starter_box_features(base: pd.DataFrame, starter_feat: pd.DataFrame) -> pd.DataFrame:
    if starter_feat is None or starter_feat.empty:
        return base
    d = base.copy()
    feature_cols = [c for c in starter_feat.columns if c not in {"game_pk", "pitcher_id", "official_date", "game_datetime_utc"}]
    latest = starter_feat[["game_pk", "pitcher_id"] + feature_cols].drop_duplicates(["game_pk", "pitcher_id"], keep="last")
    home = latest.rename(columns={"pitcher_id": "home_probable_pitcher_id", **{c: f"home_{c}" for c in feature_cols}})
    away = latest.rename(columns={"pitcher_id": "away_probable_pitcher_id", **{c: f"away_{c}" for c in feature_cols}})
    d = d.merge(home, on=["game_pk", "home_probable_pitcher_id"], how="left")
    d = d.merge(away, on=["game_pk", "away_probable_pitcher_id"], how="left")
    diff_data = {}
    for c in feature_cols:
        hc = f"home_{c}"
        ac = f"away_{c}"
        if hc in d.columns and ac in d.columns:
            diff_data[f"diff_{c}"] = d[hc] - d[ac]
    if diff_data:
        d = pd.concat([d, pd.DataFrame(diff_data, index=d.index)], axis=1)
    return d


def attach_latest_h2h_odds_features(games_df: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
    d = games_df.copy()
    if odds_df is None or odds_df.empty or "market_key" not in odds_df.columns:
        return d
    h2h = odds_df[odds_df["market_key"].astype(str).str.lower().isin(["h2h", "moneyline"])]
    if h2h.empty:
        return d
    if "commence_time_utc" in h2h.columns:
        h2h = h2h.copy()
        h2h["commence_time_utc"] = pd.to_datetime(h2h["commence_time_utc"], errors="coerce", utc=True)
    event_level = []
    for event_id, ev in h2h.groupby("event_id"):
        ev0 = ev.iloc[0]
        home_norm = ev0.get("home_team_norm") or normalize_team_name(ev0.get("home_team"))
        away_norm = ev0.get("away_team_norm") or normalize_team_name(ev0.get("away_team"))
        ev = ev.copy()
        if "outcome_name_norm" not in ev.columns:
            ev["outcome_name_norm"] = ev.get("outcome_name", pd.Series(index=ev.index)).map(normalize_team_name)
        home_prices = pd.to_numeric(ev.loc[ev["outcome_name_norm"].eq(home_norm), "outcome_price"], errors="coerce").dropna()
        away_prices = pd.to_numeric(ev.loc[ev["outcome_name_norm"].eq(away_norm), "outcome_price"], errors="coerce").dropna()
        if home_prices.empty or away_prices.empty:
            continue
        hp = float(home_prices.median())
        ap = float(away_prices.median())
        home_nv, away_nv = no_vig_two_way_prob(hp, ap)
        event_level.append({
            "event_id": event_id,
            "odds_commence_time_utc": ev0.get("commence_time_utc"),
            "home_team_norm": home_norm,
            "away_team_norm": away_norm,
            "home_moneyline_median": hp,
            "away_moneyline_median": ap,
            "market_home_no_vig_prob": home_nv,
            "market_away_no_vig_prob": away_nv,
        })
    evdf = pd.DataFrame(event_level)
    if evdf.empty:
        return d
    rows = []
    for _, g in d.iterrows():
        cand = evdf[(evdf["home_team_norm"].eq(g.get("home_team_norm"))) & (evdf["away_team_norm"].eq(g.get("away_team_norm")))].copy()
        if cand.empty:
            rows.append({})
            continue
        cand["dt_min"] = (cand["odds_commence_time_utc"] - g["game_datetime_utc"]).abs().dt.total_seconds() / 60.0
        cand = cand.sort_values("dt_min")
        best = cand.iloc[0]
        if best["dt_min"] <= 180:
            rows.append(best.drop(labels=["home_team_norm", "away_team_norm"]).to_dict())
        else:
            rows.append({})
    attach = pd.DataFrame(rows)
    d = pd.concat([d.reset_index(drop=True), attach.reset_index(drop=True)], axis=1)
    d["has_market_odds"] = d.get("home_moneyline_median", pd.Series(np.nan, index=d.index)).notna().astype(int)
    return d


def _elo_prob(diff: Any) -> Any:
    return 1.0 / (1.0 + np.power(10.0, -diff / 400.0))


def add_robust_elo_features(
    df: pd.DataFrame,
    k: float = 12.0,
    home_advantage: float = 24.0,
    season_carryover: float = 2.0 / 3.0,
    use_mov: bool = True,
    mean_rating: float = 1500.0,
) -> pd.DataFrame:
    required = ["game_pk", "official_date", "home_team_name", "away_team_name", "target_home_win"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print("Robust Elo skipped; missing columns:", missing)
        return df.copy()
    d = df.copy().loc[:, ~df.columns.duplicated(keep="first")].copy()
    d["official_date"] = pd.to_datetime(d["official_date"], errors="coerce")
    d["_elo_sort_time"] = pd.to_datetime(d.get("game_datetime_utc", d["official_date"]), errors="coerce", utc=True)
    d["_elo_season"] = d["official_date"].dt.year
    d["_orig_order"] = np.arange(len(d))
    work = d.sort_values(["_elo_season", "_elo_sort_time", "game_pk", "_orig_order"], kind="mergesort").copy()
    ratings: dict[Any, float] = {}
    last_season = None
    rows = []
    for idx, row in work.iterrows():
        season = row["_elo_season"]
        if pd.notna(season) and season != last_season:
            if last_season is not None:
                ratings = {team: mean_rating + season_carryover * (rating - mean_rating) for team, rating in ratings.items()}
            last_season = season
        home = row["home_team_name"]
        away = row["away_team_name"]
        if pd.isna(home) or pd.isna(away):
            rows.append((idx, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        h_pre = float(ratings.get(home, mean_rating))
        a_pre = float(ratings.get(away, mean_rating))
        diff_pre = h_pre + home_advantage - a_pre
        p_home = float(_elo_prob(diff_pre))
        rows.append((idx, h_pre, a_pre, h_pre - a_pre, diff_pre, p_home))
        y = row.get("target_home_win")
        if pd.notna(y):
            y = float(y)
            mov_mult = 1.0
            if use_mov and {"home_score", "away_score"}.issubset(work.columns):
                hs = row.get("home_score")
                as_ = row.get("away_score")
                if pd.notna(hs) and pd.notna(as_):
                    margin = abs(float(hs) - float(as_))
                    mov_mult = np.log1p(margin) * (2.2 / (0.001 * abs(diff_pre) + 2.2))
                    if not np.isfinite(mov_mult) or mov_mult <= 0:
                        mov_mult = 1.0
            change = k * mov_mult * (y - p_home)
            ratings[home] = h_pre + change
            ratings[away] = a_pre - change
    elo_df = pd.DataFrame(
        rows,
        columns=["_idx", "robust_home_elo_pre", "robust_away_elo_pre", "robust_diff_elo_pre", "robust_diff_elo_home_adv", "robust_elo_home_win_prob"],
    ).set_index("_idx")
    for c in ["robust_home_elo_pre", "robust_away_elo_pre", "robust_diff_elo_pre", "robust_diff_elo_home_adv", "robust_elo_home_win_prob"]:
        d[c] = elo_df[c]
    d["robust_elo_favorite_is_home"] = (d["robust_elo_home_win_prob"] >= 0.5).astype(float)
    return d.drop(columns=["_elo_sort_time", "_elo_season", "_orig_order"], errors="ignore")


def add_market_implied_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    home_col = _first_existing_col(d, ["home_moneyline_median", "home_moneyline", "market_home_moneyline", "home_ml", "home_price"])
    away_col = _first_existing_col(d, ["away_moneyline_median", "away_moneyline", "market_away_moneyline", "away_ml", "away_price"])
    if home_col is None or away_col is None:
        return d
    home_raw = pd.to_numeric(d[home_col], errors="coerce").map(lambda x: american_to_implied_prob(x) if pd.notna(x) else np.nan)
    away_raw = pd.to_numeric(d[away_col], errors="coerce").map(lambda x: american_to_implied_prob(x) if pd.notna(x) else np.nan)
    denom = home_raw + away_raw
    d["market_home_implied_raw"] = home_raw
    d["market_away_implied_raw"] = away_raw
    d["market_home_win_prob"] = np.where(denom > 0, home_raw / denom, np.nan)
    d["market_away_win_prob"] = np.where(denom > 0, away_raw / denom, np.nan)
    d["market_home_prob_edge_vs_50"] = d["market_home_win_prob"] - 0.5
    d["market_favorite_is_home"] = (d["market_home_win_prob"] >= 0.5).astype(float)
    return d


def add_team_ewm_features_to_features(
    features: pd.DataFrame,
    half_lives: Iterable[int] = (15, 30, 60, 120),
    time_col: str = "game_datetime_utc",
    date_col: str = "official_date",
) -> pd.DataFrame:
    d = features.copy()
    required = ["game_pk", date_col, "home_team_name", "away_team_name", "target_home_win", "home_score", "away_score"]
    missing = [c for c in required if c not in d.columns]
    if missing:
        print("EWMA features skipped; missing columns:", missing)
        return d
    d[time_col] = pd.to_datetime(d.get(time_col, d[date_col]), errors="coerce", utc=True)
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    existing_ewm_cols = [c for c in d.columns if "_ewm_" in c]
    if existing_ewm_cols:
        d = d.drop(columns=existing_ewm_cols)
    completed = d[d["target_home_win"].notna() & d["home_score"].notna() & d["away_score"].notna() & d[time_col].notna()].copy()
    if completed.empty:
        return d
    completed["target_home_win"] = completed["target_home_win"].astype(float)
    completed["home_score"] = completed["home_score"].astype(float)
    completed["away_score"] = completed["away_score"].astype(float)
    base_cols = ["game_pk", date_col, time_col, "home_team_name", "away_team_name", "target_home_win", "home_score", "away_score"]
    home = completed[base_cols].copy()
    home["team_name"] = home["home_team_name"]
    home["is_home"] = 1
    home["team_win"] = home["target_home_win"]
    home["runs_for"] = home["home_score"]
    home["runs_against"] = home["away_score"]
    away = completed[base_cols].copy()
    away["team_name"] = away["away_team_name"]
    away["is_home"] = 0
    away["team_win"] = 1.0 - away["target_home_win"]
    away["runs_for"] = away["away_score"]
    away["runs_against"] = away["home_score"]
    team_games = pd.concat([home, away], ignore_index=True)
    team_games["run_diff"] = team_games["runs_for"] - team_games["runs_against"]
    team_games["total_runs"] = team_games["runs_for"] + team_games["runs_against"]
    team_games["scored_5plus"] = (team_games["runs_for"] >= 5).astype(float)
    team_games["allowed_5plus"] = (team_games["runs_against"] >= 5).astype(float)
    team_games = team_games.sort_values(["team_name", time_col, "game_pk"]).reset_index(drop=True)
    metric_cols = ["team_win", "runs_for", "runs_against", "run_diff", "total_runs", "scored_5plus", "allowed_5plus"]
    ewm_cols: list[str] = []
    for hl in half_lives:
        for metric in metric_cols:
            out_col = f"ewm_{metric}_hl{hl}"
            team_games[out_col] = team_games.groupby("team_name", group_keys=False)[metric].apply(lambda s: s.ewm(halflife=hl, adjust=False).mean())
            ewm_cols.append(out_col)
    history = team_games[["team_name", time_col] + ewm_cols].sort_values(["team_name", time_col]).reset_index(drop=True)
    sched_home = d[["game_pk", time_col, "home_team_name", "away_team_name"]].copy()
    sched_home["team_name"] = sched_home["home_team_name"]
    sched_home["side"] = "home"
    sched_away = d[["game_pk", time_col, "home_team_name", "away_team_name"]].copy()
    sched_away["team_name"] = sched_away["away_team_name"]
    sched_away["side"] = "away"
    sched_long = pd.concat([sched_home, sched_away], ignore_index=True)
    sched_long = sched_long[sched_long[time_col].notna()].copy()
    merged_parts = []
    for team, s in sched_long.groupby("team_name", sort=False):
        s = s.sort_values(time_col).copy()
        h = history[history["team_name"].eq(team)].sort_values(time_col).copy()
        if h.empty:
            for c in ewm_cols:
                s[c] = np.nan
            merged_parts.append(s)
            continue
        merged_parts.append(pd.merge_asof(s, h[[time_col] + ewm_cols], on=time_col, direction="backward", allow_exact_matches=False))
    sched_with_ewm = pd.concat(merged_parts, ignore_index=True)
    home_ewm = sched_with_ewm[sched_with_ewm["side"].eq("home")][["game_pk"] + ewm_cols].drop_duplicates("game_pk").rename(columns={c: f"home_team_{c}" for c in ewm_cols})
    away_ewm = sched_with_ewm[sched_with_ewm["side"].eq("away")][["game_pk"] + ewm_cols].drop_duplicates("game_pk").rename(columns={c: f"away_team_{c}" for c in ewm_cols})
    out = d.merge(home_ewm, on="game_pk", how="left")
    out = out.merge(away_ewm, on="game_pk", how="left")
    diff_data = {}
    for c in ewm_cols:
        h = f"home_team_{c}"
        a = f"away_team_{c}"
        if h in out.columns and a in out.columns:
            diff_data[f"diff_team_{c}"] = out[h] - out[a]
    if diff_data:
        out = pd.concat([out, pd.DataFrame(diff_data, index=out.index)], axis=1)
    return out


def _recent_weights(dates: pd.Series, half_life_days: int = 365) -> np.ndarray:
    dates = pd.to_datetime(dates, errors="coerce")
    max_date = dates.max()
    age_days = (max_date - dates).dt.days.clip(lower=0)
    return np.power(0.5, age_days / half_life_days)


def _weighted_nanmean(values: pd.Series, weights: Sequence[float]) -> float:
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    w = np.asarray(weights, dtype=float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if m.sum() == 0:
        return np.nan
    return float(np.average(v[m], weights=w[m]))


def add_integer_team_clusters(df: pd.DataFrame, k: int = 2, half_life_days: int = 365, min_non_missing_rate: float = 0.50):
    try:
        from sklearn.cluster import KMeans
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        print(f"Integer team clusters skipped; sklearn import failed: {exc!r}")
        return df.copy(), pd.DataFrame(), None
    d = df.copy()
    if "official_date" not in d.columns or "home_team_name" not in d.columns or "away_team_name" not in d.columns:
        return d, pd.DataFrame(), None
    d["official_date"] = pd.to_datetime(d["official_date"], errors="coerce")
    profile_patterns = [
        "elo", "ewm", "starter", "pitchmix", "sc_team", "statcast", "xwoba", "hard_hit",
        "team_team_runs", "team_team_run_diff", "market_home_win_prob", "market_away_win_prob",
    ]
    numeric = d.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    profile_cols: list[str] = []
    for c in numeric:
        c_low = c.lower()
        if is_probably_leaky_col(c) or c_low in {"home_score", "away_score", "game_pk"}:
            continue
        if any(p in c_low for p in profile_patterns):
            if d[c].notna().mean() >= min_non_missing_rate and d[c].nunique(dropna=True) > 1:
                profile_cols.append(c)
    if len(profile_cols) < 2:
        return d, pd.DataFrame(), None
    rows = []
    for side in ["home", "away"]:
        team_col = f"{side}_team_name"
        side_cols = [c for c in profile_cols if c.startswith(f"{side}_") or c.startswith("diff_") or c.startswith("robust_") or c.startswith("market_")]
        if not side_cols:
            continue
        tmp = d[["game_pk", "official_date", team_col, "target_home_win"] + side_cols].copy()
        tmp = tmp.rename(columns={team_col: "team_name"})
        tmp["target_team_win"] = tmp["target_home_win"] if side == "home" else 1.0 - tmp["target_home_win"]
        rename = {}
        for c in side_cols:
            if c.startswith(f"{side}_"):
                rename[c] = "team_" + c[len(f"{side}_"):]
            else:
                rename[c] = c
        tmp = tmp.rename(columns=rename)
        rows.append(tmp)
    if not rows:
        return d, pd.DataFrame(), None
    team_long = pd.concat(rows, ignore_index=True)
    team_long["recency_weight"] = _recent_weights(team_long["official_date"], half_life_days=half_life_days)
    neutral_cols = [c for c in team_long.columns if c not in {"game_pk", "official_date", "team_name", "target_home_win", "target_team_win", "recency_weight"}]
    neutral_cols = [c for c in neutral_cols if pd.api.types.is_numeric_dtype(team_long[c]) and team_long[c].notna().mean() >= min_non_missing_rate]
    neutral_cols = list(dict.fromkeys(neutral_cols))
    profile_rows = []
    for team, g in team_long.dropna(subset=["team_name"]).groupby("team_name"):
        w = g["recency_weight"].to_numpy(dtype=float)
        row = {"team_name": team, "n_team_games": int(len(g)), "cluster_profile_win_rate": _weighted_nanmean(g["target_team_win"], w)}
        for c in neutral_cols:
            row[c] = _weighted_nanmean(g[c], w)
        profile_rows.append(row)
    team_profile = pd.DataFrame(profile_rows)
    if team_profile.shape[0] < k or len(neutral_cols) < 2:
        return d, team_profile, None
    cluster_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=50)),
    ])
    team_profile["team_cluster_int"] = cluster_pipe.fit_predict(team_profile[neutral_cols]).astype(int)
    cluster_map = team_profile[["team_name", "team_cluster_int"]].drop_duplicates()
    out = d.drop(
        columns=[c for c in d.columns if "team_cluster_recent" in c or c in {"same_team_cluster_recent", "cluster_strength_diff", "home_cluster_strength_rate", "away_cluster_strength_rate", "home_cluster_is_strong", "away_cluster_is_strong"}],
        errors="ignore",
    )
    out = out.merge(cluster_map.rename(columns={"team_name": "home_team_name", "team_cluster_int": "home_team_cluster_recent"}), on="home_team_name", how="left")
    out = out.merge(cluster_map.rename(columns={"team_name": "away_team_name", "team_cluster_int": "away_team_cluster_recent"}), on="away_team_name", how="left")
    out["home_team_cluster_recent"] = pd.to_numeric(out["home_team_cluster_recent"], errors="coerce")
    out["away_team_cluster_recent"] = pd.to_numeric(out["away_team_cluster_recent"], errors="coerce")
    out["same_team_cluster_recent"] = out["home_team_cluster_recent"].eq(out["away_team_cluster_recent"]).astype(float)
    return out, team_profile, cluster_pipe


def build_notebook_base_feature_frame(
    games_df: pd.DataFrame,
    box_team_df: pd.DataFrame | None = None,
    statcast_raw_df: pd.DataFrame | None = None,
    odds_df: pd.DataFrame | None = None,
    box_pitcher_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    base = add_basic_game_outcome_features(games_df).copy()
    elo = compute_elo_features(base)
    if not elo.empty:
        base = base.merge(elo, on="game_pk", how="left")
    team_roll = rolling_team_features_from_games(base)
    base = merge_home_away_team_features(base, team_roll, "team_id", "team", "home_team_id", "away_team_id")
    box_roll = rolling_boxscore_features(box_team_df if box_team_df is not None else pd.DataFrame())
    base = merge_home_away_team_features(base, box_roll, "team_id", "box", "home_team_id", "away_team_id")
    if box_pitcher_df is not None and not box_pitcher_df.empty:
        starter_box_roll = rolling_starter_boxscore_features(box_pitcher_df)
        base = merge_home_away_starter_box_features(base, starter_box_roll)
    if statcast_raw_df is not None and not statcast_raw_df.empty:
        sc_team = aggregate_statcast_team_game(statcast_raw_df)
        sc_pitcher = aggregate_statcast_pitcher_game(statcast_raw_df)
        team_pt, _pit_pt = aggregate_statcast_pitch_type(statcast_raw_df)
        sc_team_roll = rolling_statcast_team_features(sc_team)
        base = merge_home_away_team_features(base, sc_team_roll, "team_norm", "team_sc", "home_team_norm", "away_team_norm")
        pt_roll = rolling_pitchmix_team_features(team_pt)
        base = merge_home_away_team_features(base, pt_roll, "team_norm", "team_pitchmix", "home_team_norm", "away_team_norm")
        sp_roll = rolling_statcast_pitcher_features(sc_pitcher)
        base = merge_home_away_starter_features(base, sp_roll)
    base = attach_latest_h2h_odds_features(base, odds_df if odds_df is not None else pd.DataFrame())
    return base


def add_notebook_eda_features(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy().loc[:, ~features.columns.duplicated(keep="first")].copy()
    if {"home_score", "away_score"}.issubset(out.columns):
        if "target_home_win" not in out.columns:
            out["target_home_win"] = np.where(out["home_score"].notna() & out["away_score"].notna(), (out["home_score"] > out["away_score"]).astype(float), np.nan)
        if "target_total_runs" not in out.columns:
            out["target_total_runs"] = out["home_score"] + out["away_score"]
        if "target_home_margin" not in out.columns:
            out["target_home_margin"] = out["home_score"] - out["away_score"]
    out = add_robust_elo_features(out)
    out = add_market_implied_features(out)
    out = add_team_ewm_features_to_features(out)
    out, _team_cluster_profile, _team_cluster_pipe = add_integer_team_clusters(out)
    return out.loc[:, ~out.columns.duplicated(keep="first")].copy()


def build_champion_feature_frame(
    games_df: pd.DataFrame,
    box_team_df: pd.DataFrame | None = None,
    statcast_raw_df: pd.DataFrame | None = None,
    odds_df: pd.DataFrame | None = None,
    box_pitcher_df: pd.DataFrame | None = None,
    add_eda: bool = True,
) -> pd.DataFrame:
    frame = build_notebook_base_feature_frame(
        games_df=games_df,
        box_team_df=box_team_df,
        statcast_raw_df=statcast_raw_df,
        odds_df=odds_df,
        box_pitcher_df=box_pitcher_df,
    )
    if add_eda:
        frame = add_notebook_eda_features(frame)
    return frame.sort_values([c for c in ["game_datetime_utc", "game_pk"] if c in frame.columns]).reset_index(drop=True)


def load_expected_feature_cols(path: str | Path | None) -> list[str]:
    if path is None:
        return []
    p = Path(path)
    if not p.exists():
        return []
    if p.suffix.lower() == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return [str(x) for x in obj]
        if isinstance(obj, dict):
            cols = obj.get("feature_cols") or obj.get("features") or obj.get("feature_columns")
            if cols:
                return [str(x) for x in cols]
    if p.suffix.lower() == ".joblib":
        import joblib
        obj = joblib.load(p)
        if isinstance(obj, dict):
            cols = obj.get("feature_cols") or obj.get("features") or obj.get("feature_columns")
            if cols:
                return [str(x) for x in cols]
    return []


def align_frame_to_expected_features(frame: pd.DataFrame, expected_cols: Sequence[str], fill_missing: bool = True) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    expected = [str(c) for c in expected_cols]
    missing = [c for c in expected if c not in out.columns]
    if missing and fill_missing:
        add = pd.DataFrame({c: np.nan for c in missing}, index=out.index)
        out = pd.concat([out, add], axis=1)
    return out, missing


def schema_diagnostics(frame: pd.DataFrame, expected_cols: Sequence[str]) -> dict[str, Any]:
    expected = [str(c) for c in expected_cols]
    missing = [c for c in expected if c not in frame.columns]
    present = [c for c in expected if c in frame.columns]
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "expected_feature_count": int(len(expected)),
        "present_expected_features": int(len(present)),
        "missing_expected_features": int(len(missing)),
        "missing_expected_feature_examples": missing[:50],
        "all_null_expected_features": int(sum(frame[c].isna().all() for c in present)),
    }


def innings_to_float(value: Any) -> float:
    if value is None or pd.isna(value):
        return np.nan
    s = str(value).strip()
    if not s:
        return np.nan
    try:
        if "." in s:
            whole, outs = s.split(".", 1)
            return float(whole) + float(outs) / 3.0
        return float(s)
    except Exception:
        return np.nan


def parse_boxscore_team_rows(game_row: pd.Series, box: dict) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    teams = box.get("teams", {}) or {}
    for side in ["home", "away"]:
        t = teams.get(side, {}) or {}
        team_meta = t.get("team", {}) or {}
        batting = ((t.get("teamStats", {}) or {}).get("batting", {}) or {})
        pitching = ((t.get("teamStats", {}) or {}).get("pitching", {}) or {})
        opp = "away" if side == "home" else "home"
        row: dict[str, Any] = {
            "game_pk": game_row.get("game_pk"),
            "official_date": game_row.get("official_date"),
            "game_datetime_utc": game_row.get("game_datetime_utc"),
            "team_side": side,
            "team_id": team_meta.get("id") or game_row.get(f"{side}_team_id"),
            "team_name": team_meta.get("name") or game_row.get(f"{side}_team_name"),
            "team_norm": normalize_team_name(team_meta.get("name") or game_row.get(f"{side}_team_name")),
            "opponent_team_id": game_row.get(f"{opp}_team_id"),
            "opponent_team_name": game_row.get(f"{opp}_team_name"),
            "runs_for": game_row.get("home_score" if side == "home" else "away_score"),
            "runs_against": game_row.get("away_score" if side == "home" else "home_score"),
        }
        for k, v in batting.items():
            row[f"box_bat_{k}"] = v
        for k, v in pitching.items():
            row[f"box_pitch_{k}"] = v
        out.append(row)
    return out


def parse_boxscore_pitcher_rows(game_row: pd.Series, box: dict) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    teams = box.get("teams", {}) or {}
    for side in ["home", "away"]:
        t = teams.get(side, {}) or {}
        team_meta = t.get("team", {}) or {}
        players = t.get("players", {}) or {}
        pitcher_order = t.get("pitchers", []) or []
        starter_id = int(pitcher_order[0]) if pitcher_order else None
        if starter_id is None and pd.notna(game_row.get(f"{side}_probable_pitcher_id")):
            starter_id = int(game_row.get(f"{side}_probable_pitcher_id"))
        for _player_key, pinfo in players.items():
            person = pinfo.get("person", {}) or {}
            pid = person.get("id")
            stats = ((pinfo.get("stats", {}) or {}).get("pitching", {}) or {})
            if not stats:
                continue
            ip = innings_to_float(stats.get("inningsPitched"))
            hits = pd.to_numeric(stats.get("hits"), errors="coerce")
            er = pd.to_numeric(stats.get("earnedRuns"), errors="coerce")
            bb = pd.to_numeric(stats.get("baseOnBalls"), errors="coerce")
            so = pd.to_numeric(stats.get("strikeOuts"), errors="coerce")
            hr = pd.to_numeric(stats.get("homeRuns"), errors="coerce")
            pitches = pd.to_numeric(stats.get("pitchesThrown"), errors="coerce")
            row: dict[str, Any] = {
                "game_pk": game_row.get("game_pk"),
                "official_date": game_row.get("official_date"),
                "game_datetime_utc": game_row.get("game_datetime_utc"),
                "team_side": side,
                "team_id": team_meta.get("id") or game_row.get(f"{side}_team_id"),
                "team_name": team_meta.get("name") or game_row.get(f"{side}_team_name"),
                "team_norm": normalize_team_name(team_meta.get("name") or game_row.get(f"{side}_team_name")),
                "pitcher_id": pid,
                "pitcher_name": person.get("fullName"),
                "is_starting_pitcher": int(pid == starter_id) if pid is not None and starter_id is not None else 0,
                "p_ip_float": ip,
                "p_hits": hits,
                "p_runs": pd.to_numeric(stats.get("runs"), errors="coerce"),
                "p_earned_runs": er,
                "p_base_on_balls": bb,
                "p_strikeouts": so,
                "p_home_runs": hr,
                "p_pitches_thrown": pitches,
                "p_game_era": (er * 9.0 / ip) if pd.notna(ip) and ip > 0 and pd.notna(er) else np.nan,
                "p_game_whip": ((hits + bb) / ip) if pd.notna(ip) and ip > 0 and pd.notna(hits) and pd.notna(bb) else np.nan,
                "p_k_per_9": (so * 9.0 / ip) if pd.notna(ip) and ip > 0 and pd.notna(so) else np.nan,
                "p_bb_per_9": (bb * 9.0 / ip) if pd.notna(ip) and ip > 0 and pd.notna(bb) else np.nan,
                "p_hr_per_9": (hr * 9.0 / ip) if pd.notna(ip) and ip > 0 and pd.notna(hr) else np.nan,
            }
            for k, v in stats.items():
                row[f"pitch_box_{k}"] = v
            out.append(row)
    return out
