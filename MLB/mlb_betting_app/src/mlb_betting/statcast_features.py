from __future__ import annotations

"""Statcast-derived feature builders for TetheredAI MLB.

This module is intentionally additive: it can be called after the existing
`build_game_feature_frame` output is created. If Statcast tables are empty or
missing, the module returns the input frame unchanged. This lets the current
pipeline keep working while you backfill Statcast gradually.

Feature philosophy:
- Use only pregame-safe history via shifted/as-of rolling windows.
- Exclude lineup-specific features.
- Produce compact game-level features for moneyline, run-line, and totals.
"""

import logging
import math
import sqlite3
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

STATCAST_WINDOWS = (5, 10, 20, 40)
STARTER_STATCAST_WINDOWS = (3, 5, 10, 20)
BULLPEN_STATCAST_WINDOWS = (1, 3, 5, 10)
TEAM_HAND_STATCAST_WINDOWS = (10, 20, 40)

STATCAST_TEAM_METRICS = [
    "sc_pa",
    "sc_pitches_seen",
    "sc_bbe",
    "sc_woba",
    "sc_xwoba_contact",
    "sc_avg_ev",
    "sc_max_ev",
    "sc_avg_la",
    "sc_avg_batted_ball_ev",
    "sc_median_batted_ball_ev",
    "sc_p90_batted_ball_ev",
    "sc_max_batted_ball_ev",
    "sc_avg_batted_ball_distance",
    "sc_median_batted_ball_distance",
    "sc_p90_batted_ball_distance",
    "sc_max_batted_ball_distance",
    "sc_batted_ball_distance_count",
    "sc_hard_hit_rate",
    "sc_barrel_rate",
    "sc_sweetspot_rate",
    "sc_whiff_rate",
    "sc_csw_rate",
    "sc_k_rate",
    "sc_bb_rate",
    "sc_hr_rate",
]

STATCAST_PITCHER_METRICS = [
    "sc_pitches",
    "sc_pa",
    "sc_bbe_allowed",
    "sc_release_speed_mean",
    "sc_release_speed_max",
    "sc_release_spin_mean",
    "sc_release_extension_mean",
    "sc_pitch_mix_entropy",
    "sc_fastball_pct",
    "sc_breaking_pct",
    "sc_offspeed_pct",
    "sc_zone_rate",
    "sc_whiff_rate",
    "sc_csw_rate",
    "sc_called_strike_rate",
    "sc_xwoba_allowed_contact",
    "sc_woba_allowed",
    "sc_avg_ev_allowed",
    "sc_max_ev_allowed",
    "sc_avg_la_allowed",
    "sc_avg_batted_ball_ev_allowed",
    "sc_median_batted_ball_ev_allowed",
    "sc_p90_batted_ball_ev_allowed",
    "sc_max_batted_ball_ev_allowed",
    "sc_avg_batted_ball_distance_allowed",
    "sc_median_batted_ball_distance_allowed",
    "sc_p90_batted_ball_distance_allowed",
    "sc_max_batted_ball_distance_allowed",
    "sc_batted_ball_distance_allowed_count",
    "sc_hard_hit_rate_allowed",
    "sc_barrel_rate_allowed",
    "sc_sweetspot_rate_allowed",
    "sc_k_rate",
    "sc_bb_rate",
    "sc_hr_rate",
]


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
    ).fetchone()
    return row is not None


def read_table(conn: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    if not table_exists(conn, table_name):
        return pd.DataFrame()
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    if "game_datetime_utc" in df.columns:
        df["game_datetime_utc"] = pd.to_datetime(df["game_datetime_utc"], utc=True, errors="coerce")
    if "official_date" in df.columns:
        df["official_date"] = pd.to_datetime(df["official_date"], errors="coerce").dt.date.astype(str)
    return df


def _safe_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _rolling_history(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    metric_cols: Sequence[str],
    windows: Iterable[int],
    min_periods: int = 1,
) -> pd.DataFrame:
    """Create shifted expanding and rolling mean features.

    For each group, the feature at a game timestamp only uses prior games.
    This is critical: current-game Statcast results must never be included
    when predicting that same game.
    """
    if df.empty:
        return df

    group_cols = list(group_cols)
    metric_cols = [c for c in metric_cols if c in df.columns]
    if not metric_cols:
        return df[[*group_cols, "game_pk", "game_datetime_utc"]].copy()

    base_cols = [c for c in [*group_cols, "game_pk", "game_datetime_utc"] if c in df.columns]
    outputs = []

    for keys, g in df.sort_values([*group_cols, "game_datetime_utc", "game_pk"]).groupby(group_cols, dropna=False, sort=False):
        g = g.sort_values(["game_datetime_utc", "game_pk"]).copy()
        if not isinstance(keys, tuple):
            keys = (keys,)
        for col, val in zip(group_cols, keys):
            g[col] = val

        hist = g[metric_cols].shift(1)
        new_cols = {}
        for metric in metric_cols:
            new_cols[f"{metric}_season_to_date"] = hist[metric].expanding(min_periods=min_periods).mean()
            for w in windows:
                new_cols[f"{metric}_last{w}"] = hist[metric].rolling(w, min_periods=min_periods).mean()
        out = pd.concat([g[base_cols].reset_index(drop=True), pd.DataFrame(new_cols, index=g.index).reset_index(drop=True)], axis=1)
        outputs.append(out)

    if not outputs:
        return pd.DataFrame(columns=base_cols)
    return pd.concat(outputs, ignore_index=True)


def _asof_attach_by_key(
    target: pd.DataFrame,
    history: pd.DataFrame,
    target_key_cols: Sequence[str],
    history_key_cols: Sequence[str],
    feature_cols: Sequence[str],
    prefix: str,
    target_id_col: str = "game_pk",
    time_col: str = "game_datetime_utc",
) -> pd.DataFrame:
    """Attach latest prior history rows to target games.

    Uses explicit per-key loops instead of pandas `merge_asof(..., by=...)`
    because IDs often arrive as int/float/object depending on missingness.
    """
    if target.empty or history.empty:
        return pd.DataFrame({target_id_col: target[target_id_col]}) if target_id_col in target else pd.DataFrame()

    target = target.copy()
    history = history.copy()
    target[time_col] = pd.to_datetime(target[time_col], utc=True, errors="coerce")
    history[time_col] = pd.to_datetime(history[time_col], utc=True, errors="coerce")

    feature_cols = [c for c in feature_cols if c in history.columns]
    if not feature_cols:
        return pd.DataFrame({target_id_col: target[target_id_col]})

    target_key_cols = list(target_key_cols)
    history_key_cols = list(history_key_cols)
    if len(target_key_cols) != len(history_key_cols):
        raise ValueError("target_key_cols and history_key_cols must have same length")

    # String-normalize join keys to avoid int64/float64/object mismatches.
    for c in target_key_cols:
        target[c] = target[c].astype("Int64").astype(str) if pd.api.types.is_numeric_dtype(target[c]) else target[c].astype(str)
    for c in history_key_cols:
        history[c] = history[c].astype("Int64").astype(str) if pd.api.types.is_numeric_dtype(history[c]) else history[c].astype(str)

    pieces = []
    target_key_name = "__key__"
    hist_key_name = "__key__"
    target[target_key_name] = target[target_key_cols].astype(str).agg("|".join, axis=1)
    history[hist_key_name] = history[history_key_cols].astype(str).agg("|".join, axis=1)

    for key, left in target.groupby(target_key_name, sort=False, dropna=False):
        if key is None or "<NA>" in str(key) or str(key).lower() in {"nan", "none"}:
            continue
        right = history[history[hist_key_name] == key]
        if right.empty:
            continue
        left_small = left[[target_id_col, time_col]].sort_values(time_col)
        right_small = right[[time_col, *feature_cols]].sort_values(time_col)
        merged = pd.merge_asof(
            left_small,
            right_small,
            on=time_col,
            direction="backward",
            allow_exact_matches=False,
        )
        pieces.append(merged.drop(columns=[time_col]))

    if pieces:
        attached = pd.concat(pieces, ignore_index=True)
    else:
        attached = pd.DataFrame({target_id_col: target[target_id_col]})

    rename_map = {c: f"{prefix}{c}" for c in feature_cols}
    attached = attached.rename(columns=rename_map)
    return attached



def _dedupe_by_game_id(
    df: pd.DataFrame,
    target_id_col: str = "game_pk",
    label: str = "frame",
) -> pd.DataFrame:
    """Return at most one row per game id.

    The Statcast attachment builders are expected to produce one row per game.
    If a prior attachment accidentally creates duplicate game rows, a later
    merge can become many-to-many and explode memory usage. This guard keeps
    the frame one-row-per-game and makes future joins safe.
    """
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()

    out = df.copy()
    if target_id_col not in out.columns:
        return out

    dup_rows = int(out.duplicated(subset=[target_id_col], keep=False).sum())
    if dup_rows:
        unique_dups = int(out.loc[out.duplicated(subset=[target_id_col], keep=False), target_id_col].nunique())
        LOGGER.warning(
            "%s has duplicated %s values: %s duplicated rows across %s games. "
            "Keeping the last row per game to prevent a many-to-many merge.",
            label,
            target_id_col,
            dup_rows,
            unique_dups,
        )

    return out.drop_duplicates(subset=[target_id_col], keep="last").reset_index(drop=True)


def _merge_home_away(
    base: pd.DataFrame,
    home_df: pd.DataFrame,
    away_df: pd.DataFrame,
    target_id_col: str = "game_pk",
) -> pd.DataFrame:
    """Safely merge home and away attachment frames onto a game frame.

    This is deliberately defensive. A duplicate `game_pk` in either attachment
    frame can trigger a huge many-to-many merge. In Cloud Run this manifested as
    pandas trying to allocate hundreds of GiB. We therefore:
      1. reduce base/home/away to one row per game,
      2. drop incoming columns that already exist in the base frame, and
      3. validate one-to-one merge cardinality.
    """
    out = _dedupe_by_game_id(base, target_id_col=target_id_col, label="base_statcast_frame")

    for label, incoming in (("home", home_df), ("away", away_df)):
        if incoming is None or incoming.empty:
            continue
        if target_id_col not in incoming.columns:
            LOGGER.warning("Skipping %s Statcast attachment because %s is missing.", label, target_id_col)
            continue

        incoming = _dedupe_by_game_id(
            incoming,
            target_id_col=target_id_col,
            label=f"{label}_statcast_attachment",
        )

        keep_cols = [target_id_col]
        keep_cols.extend([c for c in incoming.columns if c != target_id_col and c not in out.columns])
        incoming = incoming[keep_cols]

        # If all feature columns already exist, there is nothing new to attach.
        if len(keep_cols) == 1:
            continue

        out = out.merge(
            incoming,
            on=target_id_col,
            how="left",
            validate="one_to_one",
        )

    return out

def _add_diff_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add home-minus-away differences for matching Statcast columns."""
    diff_cols = {}
    for col in df.columns:
        if not col.startswith("home_"):
            continue
        away_col = "away_" + col[len("home_"):]
        if away_col in df.columns and pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(df[away_col]):
            diff_name = "diff_" + col[len("home_"):]
            if diff_name not in df.columns:
                diff_cols[diff_name] = df[col] - df[away_col]
    if diff_cols:
        df = pd.concat([df, pd.DataFrame(diff_cols, index=df.index)], axis=1)
    return df.copy()


def _prepare_base_games(features: pd.DataFrame, conn: sqlite3.Connection) -> pd.DataFrame:
    games = read_table(conn, "mlb_games")
    if games.empty:
        return features.copy()

    wanted = [
        "game_pk", "game_datetime_utc", "official_date",
        "home_team_id", "away_team_id",
        "probable_home_pitcher_id", "probable_away_pitcher_id",
    ]
    games = games[[c for c in wanted if c in games.columns]].copy()
    base = features.copy()
    if "game_datetime_utc" in base.columns:
        base["game_datetime_utc"] = pd.to_datetime(base["game_datetime_utc"], utc=True, errors="coerce")

    missing_cols = [c for c in games.columns if c not in base.columns or c == "game_pk"]
    # Merge only columns absent in the feature file to avoid suffix confusion.
    merge_cols = ["game_pk", *[c for c in games.columns if c != "game_pk" and c not in base.columns]]
    if len(merge_cols) > 1:
        base = base.merge(games[merge_cols], on="game_pk", how="left")
    return base


def add_statcast_team_offense_features(features: pd.DataFrame, conn: sqlite3.Connection) -> pd.DataFrame:
    stat_team = read_table(conn, "mlb_statcast_team_game")
    if stat_team.empty:
        LOGGER.warning("No mlb_statcast_team_game rows found; skipping Statcast team offense features.")
        return features

    base = _prepare_base_games(features, conn)
    stat_team = _safe_numeric(stat_team, STATCAST_TEAM_METRICS)
    roll = _rolling_history(stat_team, ["team_id"], STATCAST_TEAM_METRICS, STATCAST_WINDOWS)
    feature_cols = [c for c in roll.columns if c.startswith("sc_")]

    home = base[["game_pk", "game_datetime_utc", "home_team_id"]].copy()
    away = base[["game_pk", "game_datetime_utc", "away_team_id"]].copy()
    home_att = _asof_attach_by_key(home, roll, ["home_team_id"], ["team_id"], feature_cols, "home_team_off_")
    away_att = _asof_attach_by_key(away, roll, ["away_team_id"], ["team_id"], feature_cols, "away_team_off_")
    out = _merge_home_away(base, home_att, away_att)
    return _add_diff_columns(out)


def add_statcast_team_vs_hand_features(features: pd.DataFrame, conn: sqlite3.Connection) -> pd.DataFrame:
    stat_hand = read_table(conn, "mlb_statcast_team_hand_game")
    stat_pitch = read_table(conn, "mlb_statcast_pitcher_game")
    if stat_hand.empty:
        LOGGER.warning("No mlb_statcast_team_hand_game rows found; skipping Statcast vs-hand features.")
        return features

    base = _prepare_base_games(features, conn)
    stat_hand = _safe_numeric(stat_hand, STATCAST_TEAM_METRICS)
    if "pitcher_hand" in stat_hand.columns:
        stat_hand["pitcher_hand"] = stat_hand["pitcher_hand"].fillna("UNK").astype(str)
    else:
        stat_hand["pitcher_hand"] = "UNK"

    roll = _rolling_history(stat_hand, ["team_id", "pitcher_hand"], STATCAST_TEAM_METRICS, TEAM_HAND_STATCAST_WINDOWS)
    feature_cols = [c for c in roll.columns if c.startswith("sc_")]

    # Build pitcher hand lookup. Prefer Statcast, fallback to boxscore pitcher table.
    lookup_frames = []
    if not stat_pitch.empty and {"pitcher_id", "pitcher_hand"}.issubset(stat_pitch.columns):
        lookup_frames.append(stat_pitch[["pitcher_id", "pitcher_hand", "game_datetime_utc"]])
    box_pitch = read_table(conn, "mlb_pitcher_game_stats")
    if not box_pitch.empty and {"pitcher_id", "pitcher_hand"}.issubset(box_pitch.columns):
        lookup_frames.append(box_pitch[["pitcher_id", "pitcher_hand", "game_datetime_utc"]])
    if lookup_frames:
        hand_lookup = pd.concat(lookup_frames, ignore_index=True).dropna(subset=["pitcher_id", "pitcher_hand"])
        hand_lookup = hand_lookup.sort_values("game_datetime_utc").drop_duplicates("pitcher_id", keep="last")
        hand_map = dict(zip(hand_lookup["pitcher_id"].astype("Int64").astype(str), hand_lookup["pitcher_hand"].astype(str)))
    else:
        hand_map = {}

    base["home_starter_hand_sc"] = base.get("probable_home_pitcher_id", pd.Series(index=base.index)).astype("Int64").astype(str).map(hand_map).fillna("UNK")
    base["away_starter_hand_sc"] = base.get("probable_away_pitcher_id", pd.Series(index=base.index)).astype("Int64").astype(str).map(hand_map).fillna("UNK")

    home = base[["game_pk", "game_datetime_utc", "home_team_id", "away_starter_hand_sc"]].rename(columns={"away_starter_hand_sc": "pitcher_hand"})
    away = base[["game_pk", "game_datetime_utc", "away_team_id", "home_starter_hand_sc"]].rename(columns={"home_starter_hand_sc": "pitcher_hand"})

    home_att = _asof_attach_by_key(home, roll, ["home_team_id", "pitcher_hand"], ["team_id", "pitcher_hand"], feature_cols, "home_team_vs_hand_sc_")
    away_att = _asof_attach_by_key(away, roll, ["away_team_id", "pitcher_hand"], ["team_id", "pitcher_hand"], feature_cols, "away_team_vs_hand_sc_")
    out = _merge_home_away(base, home_att, away_att)
    return _add_diff_columns(out)


def add_statcast_starter_features(features: pd.DataFrame, conn: sqlite3.Connection) -> pd.DataFrame:
    stat_pitch = read_table(conn, "mlb_statcast_pitcher_game")
    if stat_pitch.empty:
        LOGGER.warning("No mlb_statcast_pitcher_game rows found; skipping Statcast starter features.")
        return features

    base = _prepare_base_games(features, conn)
    stat_pitch = _safe_numeric(stat_pitch, STATCAST_PITCHER_METRICS)
    # Keep all pitcher games. Probable starter merge will select history for the projected starter.
    roll = _rolling_history(stat_pitch, ["pitcher_id"], STATCAST_PITCHER_METRICS, STARTER_STATCAST_WINDOWS)
    feature_cols = [c for c in roll.columns if c.startswith("sc_")]

    home = base[["game_pk", "game_datetime_utc", "probable_home_pitcher_id"]].copy()
    away = base[["game_pk", "game_datetime_utc", "probable_away_pitcher_id"]].copy()
    home_att = _asof_attach_by_key(home, roll, ["probable_home_pitcher_id"], ["pitcher_id"], feature_cols, "home_starter_statcast_")
    away_att = _asof_attach_by_key(away, roll, ["probable_away_pitcher_id"], ["pitcher_id"], feature_cols, "away_starter_statcast_")
    out = _merge_home_away(base, home_att, away_att)
    return _add_diff_columns(out)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return float(values.mean()) if values.notna().any() else np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def _build_statcast_bullpen_team_game(stat_pitch: pd.DataFrame) -> pd.DataFrame:
    if stat_pitch.empty:
        return pd.DataFrame()
    if "is_starter" not in stat_pitch.columns:
        return pd.DataFrame()
    bp = stat_pitch[pd.to_numeric(stat_pitch["is_starter"], errors="coerce").fillna(0).astype(int) == 0].copy()
    if bp.empty:
        return pd.DataFrame()
    bp = _safe_numeric(bp, STATCAST_PITCHER_METRICS)
    rows = []
    for (game_pk, team_id), g in bp.groupby(["game_pk", "team_id"], dropna=False, sort=False):
        row = {
            "game_pk": game_pk,
            "team_id": team_id,
            "official_date": g["official_date"].iloc[0] if "official_date" in g else None,
            "game_datetime_utc": g["game_datetime_utc"].iloc[0] if "game_datetime_utc" in g else None,
            "sc_bullpen_pitchers_used": int(g["pitcher_id"].nunique()) if "pitcher_id" in g else len(g),
            "sc_bullpen_pitches": pd.to_numeric(g.get("sc_pitches"), errors="coerce").sum(),
            "sc_bullpen_pa": pd.to_numeric(g.get("sc_pa"), errors="coerce").sum(),
            "sc_bullpen_bbe_allowed": pd.to_numeric(g.get("sc_bbe_allowed"), errors="coerce").sum(),
        }
        for metric in [
            "sc_xwoba_allowed_contact", "sc_woba_allowed", "sc_avg_ev_allowed",
            "sc_avg_batted_ball_ev_allowed", "sc_p90_batted_ball_ev_allowed",
            "sc_avg_batted_ball_distance_allowed", "sc_p90_batted_ball_distance_allowed",
            "sc_hard_hit_rate_allowed", "sc_barrel_rate_allowed", "sc_sweetspot_rate_allowed",
            "sc_release_speed_mean", "sc_release_spin_mean", "sc_release_extension_mean",
            "sc_whiff_rate", "sc_csw_rate", "sc_k_rate", "sc_bb_rate", "sc_hr_rate",
        ]:
            if metric in g.columns:
                weight = g["sc_pa"] if metric in {"sc_woba_allowed", "sc_k_rate", "sc_bb_rate", "sc_hr_rate"} else g.get("sc_pitches", pd.Series(1, index=g.index))
                row[f"bullpen_{metric}"] = _weighted_mean(g[metric], weight)
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["game_datetime_utc"] = pd.to_datetime(out["game_datetime_utc"], utc=True, errors="coerce")
    return out


def add_statcast_bullpen_features(features: pd.DataFrame, conn: sqlite3.Connection) -> pd.DataFrame:
    stat_pitch = read_table(conn, "mlb_statcast_pitcher_game")
    if stat_pitch.empty:
        LOGGER.warning("No mlb_statcast_pitcher_game rows found; skipping Statcast bullpen features.")
        return features

    bp_team = _build_statcast_bullpen_team_game(stat_pitch)
    if bp_team.empty:
        LOGGER.warning("No non-starter Statcast pitcher rows found; skipping Statcast bullpen features.")
        return features

    base = _prepare_base_games(features, conn)
    metric_cols = [c for c in bp_team.columns if c.startswith("sc_bullpen_") or c.startswith("bullpen_sc_")]
    bp_team = _safe_numeric(bp_team, metric_cols)
    roll = _rolling_history(bp_team, ["team_id"], metric_cols, BULLPEN_STATCAST_WINDOWS)
    feature_cols = [c for c in roll.columns if c.startswith("sc_bullpen_") or c.startswith("bullpen_sc_")]

    home = base[["game_pk", "game_datetime_utc", "home_team_id"]].copy()
    away = base[["game_pk", "game_datetime_utc", "away_team_id"]].copy()
    home_att = _asof_attach_by_key(home, roll, ["home_team_id"], ["team_id"], feature_cols, "home_")
    away_att = _asof_attach_by_key(away, roll, ["away_team_id"], ["team_id"], feature_cols, "away_")
    out = _merge_home_away(base, home_att, away_att)
    return _add_diff_columns(out)


def add_statcast_features(features: pd.DataFrame, conn: sqlite3.Connection) -> pd.DataFrame:
    """Add all available Statcast feature blocks to a game feature frame.

    Safe to call even when no Statcast tables exist.
    """
    if features.empty:
        return features
    out = features.copy()
    out = add_statcast_team_offense_features(out, conn)
    out = add_statcast_team_vs_hand_features(out, conn)
    out = add_statcast_starter_features(out, conn)
    out = add_statcast_bullpen_features(out, conn)
    return out.copy()


def get_statcast_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return Statcast-derived numeric columns suitable for model candidates."""
    prefixes = (
        "home_team_off_sc_", "away_team_off_sc_", "diff_team_off_sc_",
        "home_team_vs_hand_sc_", "away_team_vs_hand_sc_", "diff_team_vs_hand_sc_",
        "home_starter_statcast_sc_", "away_starter_statcast_sc_", "diff_starter_statcast_sc_",
        "home_sc_bullpen_", "away_sc_bullpen_", "diff_sc_bullpen_",
        "home_bullpen_sc_", "away_bullpen_sc_", "diff_bullpen_sc_",
    )
    cols = [c for c in frame.columns if c.startswith(prefixes)]
    return [c for c in cols if pd.api.types.is_numeric_dtype(frame[c])]
