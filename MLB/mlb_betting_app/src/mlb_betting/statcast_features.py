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



PITCH_TYPE_GROUPS = ("fastball", "breaking", "offspeed", "other")
PITCH_TYPE_WINDOWS = (5, 10, 20, 40)
PITCHER_PITCH_TYPE_WINDOWS = (3, 5, 10, 20)
PITCHMIX_MATCHUP_WINDOWS = ("season_to_date", "last5", "last10", "last20")

TEAM_PITCH_TYPE_METRICS = [
    "sc_pitch_type_pa",
    "sc_pitch_type_pitches_seen",
    "sc_pitch_type_bbe",
    "sc_pitch_type_woba",
    "sc_pitch_type_xwoba_contact",
    "sc_pitch_type_avg_ev",
    "sc_pitch_type_p90_ev",
    "sc_pitch_type_avg_batted_ball_distance",
    "sc_pitch_type_p90_batted_ball_distance",
    "sc_pitch_type_hard_hit_rate",
    "sc_pitch_type_barrel_rate",
    "sc_pitch_type_whiff_rate",
    "sc_pitch_type_csw_rate",
    "sc_pitch_type_k_rate",
    "sc_pitch_type_bb_rate",
    "sc_pitch_type_hr_rate",
]

PITCHER_PITCH_TYPE_METRICS = [
    "sc_pitch_type_pitches",
    "sc_pitch_type_pct",
    "sc_pitch_type_pa",
    "sc_pitch_type_bbe_allowed",
    "sc_pitch_type_release_speed_mean",
    "sc_pitch_type_release_speed_max",
    "sc_pitch_type_release_spin_mean",
    "sc_pitch_type_zone_rate",
    "sc_pitch_type_whiff_rate",
    "sc_pitch_type_csw_rate",
    "sc_pitch_type_called_strike_rate",
    "sc_pitch_type_xwoba_allowed_contact",
    "sc_pitch_type_woba_allowed",
    "sc_pitch_type_avg_ev_allowed",
    "sc_pitch_type_p90_ev_allowed",
    "sc_pitch_type_avg_batted_ball_distance_allowed",
    "sc_pitch_type_p90_batted_ball_distance_allowed",
    "sc_pitch_type_hard_hit_rate_allowed",
    "sc_pitch_type_barrel_rate_allowed",
    "sc_pitch_type_k_rate",
    "sc_pitch_type_bb_rate",
    "sc_pitch_type_hr_rate",
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



def _merge_attachment(out: pd.DataFrame, att: pd.DataFrame, target_id_col: str = "game_pk") -> pd.DataFrame:
    """Safely merge a one-row-per-game attachment into an existing feature frame."""
    if att is None or att.empty or target_id_col not in att.columns:
        return out
    out = _dedupe_by_game_id(out, target_id_col=target_id_col, label="feature_frame")
    att = _dedupe_by_game_id(att, target_id_col=target_id_col, label="feature_attachment")
    keep_cols = [target_id_col] + [c for c in att.columns if c != target_id_col and c not in out.columns]
    if len(keep_cols) == 1:
        return out
    return out.merge(att[keep_cols], on=target_id_col, how="left", validate="one_to_one")


def _attach_pitch_type_rolls(
    base: pd.DataFrame,
    roll: pd.DataFrame,
    target_entity_col: str,
    history_entity_col: str,
    feature_cols: Sequence[str],
    prefix: str,
) -> pd.DataFrame:
    """Attach pitch-type rolling rows by game, entity, and pitch_type_group.

    Output is one row per game with group names embedded in column names, e.g.
    home_team_pitchtype_fastball_sc_pitch_type_woba_last20.
    """
    if base.empty or roll.empty or target_entity_col not in base.columns:
        return pd.DataFrame({"game_pk": base["game_pk"]}) if "game_pk" in base else pd.DataFrame()

    attachments = []
    for group in PITCH_TYPE_GROUPS:
        target = base[["game_pk", "game_datetime_utc", target_entity_col]].copy()
        target["pitch_type_group"] = group
        hist = roll[roll.get("pitch_type_group", pd.Series(index=roll.index, dtype="object")).astype(str).eq(group)].copy()
        if hist.empty:
            continue
        att = _asof_attach_by_key(
            target,
            hist,
            [target_entity_col, "pitch_type_group"],
            [history_entity_col, "pitch_type_group"],
            feature_cols,
            f"{prefix}{group}_",
        )
        attachments.append(att)

    out = pd.DataFrame({"game_pk": base["game_pk"]})
    for att in attachments:
        out = _merge_attachment(out, att)
    return out


def _weighted_sum_from_group_cols(frame: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.Series:
    """Row-wise weighted average that tolerates missing groups/weights."""
    numerator = pd.Series(0.0, index=frame.index)
    denominator = pd.Series(0.0, index=frame.index)
    any_pair = pd.Series(False, index=frame.index)

    for value_col, weight_col in pairs:
        if value_col not in frame.columns or weight_col not in frame.columns:
            continue
        v = pd.to_numeric(frame[value_col], errors="coerce")
        w = pd.to_numeric(frame[weight_col], errors="coerce")
        mask = v.notna() & w.notna() & (w > 0)
        numerator = numerator.add((v * w).where(mask, 0.0), fill_value=0.0)
        denominator = denominator.add(w.where(mask, 0.0), fill_value=0.0)
        any_pair = any_pair | mask

    out = numerator / denominator.replace(0, np.nan)
    out = out.where(any_pair, np.nan)
    return out


def add_statcast_pitchmix_matchup_features(features: pd.DataFrame, conn: sqlite3.Connection) -> pd.DataFrame:
    """Add starter pitch-mix vs opponent offense matchup features.

    Example intuition:
    if the away probable starter throws fastballs 58% of the time, compare that
    to the home offense's rolling production versus fastballs. We calculate this
    across coarse pitch groups and add weighted matchup features for each side.
    """
    team_pt = read_table(conn, "mlb_statcast_team_pitch_type_game")
    pitcher_pt = read_table(conn, "mlb_statcast_pitcher_pitch_type_game")
    if team_pt.empty or pitcher_pt.empty:
        LOGGER.warning("Pitch-type Statcast tables missing/empty; skipping pitch-mix matchup features.")
        return features

    base = _prepare_base_games(features, conn)

    team_pt = _safe_numeric(team_pt, TEAM_PITCH_TYPE_METRICS)
    pitcher_pt = _safe_numeric(pitcher_pt, PITCHER_PITCH_TYPE_METRICS)
    team_pt["pitch_type_group"] = team_pt.get("pitch_type_group", "other").fillna("other").astype(str)
    pitcher_pt["pitch_type_group"] = pitcher_pt.get("pitch_type_group", "other").fillna("other").astype(str)

    team_roll = _rolling_history(team_pt, ["team_id", "pitch_type_group"], TEAM_PITCH_TYPE_METRICS, PITCH_TYPE_WINDOWS)
    pitcher_roll = _rolling_history(pitcher_pt, ["pitcher_id", "pitch_type_group"], PITCHER_PITCH_TYPE_METRICS, PITCHER_PITCH_TYPE_WINDOWS)

    team_feature_cols = [c for c in team_roll.columns if c.startswith("sc_pitch_type_")]
    pitcher_feature_cols = [c for c in pitcher_roll.columns if c.startswith("sc_pitch_type_")]

    out = base.copy()
    home_team_att = _attach_pitch_type_rolls(base, team_roll, "home_team_id", "team_id", team_feature_cols, "home_team_pitchtype_")
    away_team_att = _attach_pitch_type_rolls(base, team_roll, "away_team_id", "team_id", team_feature_cols, "away_team_pitchtype_")
    home_starter_att = _attach_pitch_type_rolls(base, pitcher_roll, "probable_home_pitcher_id", "pitcher_id", pitcher_feature_cols, "home_starter_pitchtype_")
    away_starter_att = _attach_pitch_type_rolls(base, pitcher_roll, "probable_away_pitcher_id", "pitcher_id", pitcher_feature_cols, "away_starter_pitchtype_")

    for att in [home_team_att, away_team_att, home_starter_att, away_starter_att]:
        out = _merge_attachment(out, att)

    matchup_metrics = [
        "sc_pitch_type_woba",
        "sc_pitch_type_xwoba_contact",
        "sc_pitch_type_avg_ev",
        "sc_pitch_type_p90_ev",
        "sc_pitch_type_avg_batted_ball_distance",
        "sc_pitch_type_p90_batted_ball_distance",
        "sc_pitch_type_hard_hit_rate",
        "sc_pitch_type_barrel_rate",
        "sc_pitch_type_whiff_rate",
        "sc_pitch_type_k_rate",
        "sc_pitch_type_bb_rate",
        "sc_pitch_type_hr_rate",
    ]

    pitcher_quality_metrics = [
        "sc_pitch_type_woba_allowed",
        "sc_pitch_type_xwoba_allowed_contact",
        "sc_pitch_type_avg_ev_allowed",
        "sc_pitch_type_p90_ev_allowed",
        "sc_pitch_type_avg_batted_ball_distance_allowed",
        "sc_pitch_type_p90_batted_ball_distance_allowed",
        "sc_pitch_type_hard_hit_rate_allowed",
        "sc_pitch_type_barrel_rate_allowed",
        "sc_pitch_type_whiff_rate",
        "sc_pitch_type_k_rate",
        "sc_pitch_type_bb_rate",
        "sc_pitch_type_hr_rate",
    ]

    new_cols: dict[str, pd.Series] = {}
    for window in PITCHMIX_MATCHUP_WINDOWS:
        for metric in matchup_metrics:
            # Home offense weighted by away starter's pitch mix.
            home_pairs = []
            away_pairs = []
            for group in PITCH_TYPE_GROUPS:
                home_value = f"home_team_pitchtype_{group}_{metric}_{window}"
                away_value = f"away_team_pitchtype_{group}_{metric}_{window}"
                away_starter_weight = f"away_starter_pitchtype_{group}_sc_pitch_type_pct_{window}"
                home_starter_weight = f"home_starter_pitchtype_{group}_sc_pitch_type_pct_{window}"
                home_pairs.append((home_value, away_starter_weight))
                away_pairs.append((away_value, home_starter_weight))

            home_name = f"home_pitchmix_matchup_off_{metric}_{window}"
            away_name = f"away_pitchmix_matchup_off_{metric}_{window}"
            home_s = _weighted_sum_from_group_cols(out, home_pairs)
            away_s = _weighted_sum_from_group_cols(out, away_pairs)
            new_cols[home_name] = home_s
            new_cols[away_name] = away_s
            new_cols[f"diff_pitchmix_matchup_off_{metric}_{window}"] = home_s - away_s

        for metric in pitcher_quality_metrics:
            # Starter's own pitch-type quality weighted by their pitch mix.
            home_pairs = []
            away_pairs = []
            for group in PITCH_TYPE_GROUPS:
                home_value = f"home_starter_pitchtype_{group}_{metric}_{window}"
                away_value = f"away_starter_pitchtype_{group}_{metric}_{window}"
                home_weight = f"home_starter_pitchtype_{group}_sc_pitch_type_pct_{window}"
                away_weight = f"away_starter_pitchtype_{group}_sc_pitch_type_pct_{window}"
                home_pairs.append((home_value, home_weight))
                away_pairs.append((away_value, away_weight))

            home_name = f"home_starter_pitchmix_allowed_{metric}_{window}"
            away_name = f"away_starter_pitchmix_allowed_{metric}_{window}"
            home_s = _weighted_sum_from_group_cols(out, home_pairs)
            away_s = _weighted_sum_from_group_cols(out, away_pairs)
            new_cols[home_name] = home_s
            new_cols[away_name] = away_s
            new_cols[f"diff_starter_pitchmix_allowed_{metric}_{window}"] = home_s - away_s

    if new_cols:
        out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)
    return _add_diff_columns(out)


def _build_statcast_bullpen_availability_history(bp_team: pd.DataFrame) -> pd.DataFrame:
    """Pregame bullpen workload/availability features using only prior games."""
    if bp_team.empty:
        return pd.DataFrame()

    needed = ["team_id", "game_pk", "game_datetime_utc", "sc_bullpen_pitches", "sc_bullpen_pitchers_used", "sc_bullpen_pa"]
    for c in needed:
        if c not in bp_team.columns:
            bp_team[c] = np.nan

    rows = []
    for team_id, g in bp_team.sort_values(["team_id", "game_datetime_utc", "game_pk"]).groupby("team_id", dropna=False, sort=False):
        g = g.sort_values(["game_datetime_utc", "game_pk"]).copy().reset_index(drop=True)
        out = g[["team_id", "game_pk", "game_datetime_utc"]].copy()

        prev_dt = g["game_datetime_utc"].shift(1)
        out["sc_bullpen_days_since_last_game"] = (
            (g["game_datetime_utc"] - prev_dt).dt.total_seconds() / 86400.0
        )

        pitches = pd.to_numeric(g["sc_bullpen_pitches"], errors="coerce").shift(1)
        pitchers_used = pd.to_numeric(g["sc_bullpen_pitchers_used"], errors="coerce").shift(1)
        pa = pd.to_numeric(g["sc_bullpen_pa"], errors="coerce").shift(1)
        high_usage = pitches.ge(45).astype(float)
        very_high_usage = pitches.ge(65).astype(float)

        for w in BULLPEN_STATCAST_WINDOWS:
            out[f"sc_bullpen_pitches_sum_last{w}"] = pitches.rolling(w, min_periods=1).sum()
            out[f"sc_bullpen_pitches_mean_last{w}"] = pitches.rolling(w, min_periods=1).mean()
            out[f"sc_bullpen_pitchers_used_sum_last{w}"] = pitchers_used.rolling(w, min_periods=1).sum()
            out[f"sc_bullpen_pa_sum_last{w}"] = pa.rolling(w, min_periods=1).sum()
            out[f"sc_bullpen_high_usage_games_last{w}"] = high_usage.rolling(w, min_periods=1).sum()
            out[f"sc_bullpen_very_high_usage_games_last{w}"] = very_high_usage.rolling(w, min_periods=1).sum()

        # Simple pressure score: recent pitches normalized by rest days. Higher = worse availability.
        rest = out["sc_bullpen_days_since_last_game"].clip(lower=0.5)
        out["sc_bullpen_availability_pressure_last3"] = out["sc_bullpen_pitches_sum_last3"] / rest
        out["sc_bullpen_availability_pressure_last5"] = out["sc_bullpen_pitches_sum_last5"] / rest
        rows.append(out)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def add_statcast_bullpen_availability_features(features: pd.DataFrame, conn: sqlite3.Connection) -> pd.DataFrame:
    """Add workload/rest availability features for each team's bullpen."""
    stat_pitch = read_table(conn, "mlb_statcast_pitcher_game")
    if stat_pitch.empty:
        return features
    bp_team = _build_statcast_bullpen_team_game(stat_pitch)
    if bp_team.empty:
        return features
    base = _prepare_base_games(features, conn)
    avail = _build_statcast_bullpen_availability_history(bp_team)
    if avail.empty:
        return features

    feature_cols = [c for c in avail.columns if c.startswith("sc_bullpen_")]
    home = base[["game_pk", "game_datetime_utc", "home_team_id"]].copy()
    away = base[["game_pk", "game_datetime_utc", "away_team_id"]].copy()
    home_att = _asof_attach_by_key(home, avail, ["home_team_id"], ["team_id"], feature_cols, "home_bullpen_avail_")
    away_att = _asof_attach_by_key(away, avail, ["away_team_id"], ["team_id"], feature_cols, "away_bullpen_avail_")
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
    out = add_statcast_bullpen_availability_features(out, conn)
    out = add_statcast_pitchmix_matchup_features(out, conn)
    return out.copy()


def get_statcast_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return Statcast-derived numeric columns suitable for model candidates."""
    prefixes = (
        "home_team_off_sc_", "away_team_off_sc_", "diff_team_off_sc_",
        "home_team_vs_hand_sc_", "away_team_vs_hand_sc_", "diff_team_vs_hand_sc_",
        "home_starter_statcast_sc_", "away_starter_statcast_sc_", "diff_starter_statcast_sc_",
        "home_sc_bullpen_", "away_sc_bullpen_", "diff_sc_bullpen_",
        "home_bullpen_sc_", "away_bullpen_sc_", "diff_bullpen_sc_",
        "home_bullpen_avail_sc_", "away_bullpen_avail_sc_", "diff_bullpen_avail_sc_",
        "home_team_pitchtype_", "away_team_pitchtype_", "diff_team_pitchtype_",
        "home_starter_pitchtype_", "away_starter_pitchtype_", "diff_starter_pitchtype_",
        "home_pitchmix_matchup_", "away_pitchmix_matchup_", "diff_pitchmix_matchup_",
        "home_starter_pitchmix_", "away_starter_pitchmix_", "diff_starter_pitchmix_",
    )
    cols = [c for c in frame.columns if c.startswith(prefixes)]
    return [c for c in cols if pd.api.types.is_numeric_dtype(frame[c])]
