from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="TetheredAI | Sports Intelligence", page_icon="🔗", layout="wide")

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
ASSET_DIR = APP_DIR / "assets"
PRED_DIR = PROJECT_ROOT / "data" / "predictions"
MLB_H2H_PATH = PRED_DIR / "mlb_moneyline_predictions.csv"


@st.cache_data(show_spinner=False)
def load_predictions(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "game_datetime_utc" in df.columns:
        df["game_datetime_utc"] = pd.to_datetime(df["game_datetime_utc"], utc=True, errors="coerce")
    return df


def fmt_pct(x, digits=1):
    if pd.isna(x):
        return "—"
    return f"{float(x) * 100:.{digits}f}%"


def fmt_units(x, digits=2):
    if pd.isna(x):
        return "—"
    return f"{float(x):.{digits}f}u"


def fmt_price(x):
    if pd.isna(x):
        return "—"
    x = int(round(float(x)))
    return f"+{x}" if x > 0 else str(x)


def normalize_for_score(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(0.0, index=s.index)
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def add_display_score(df: pd.DataFrame, prob_weight: float) -> pd.DataFrame:
    out = df.copy()
    prob = normalize_for_score(out.get("recommended_model_prob", pd.Series(index=out.index, dtype=float)))
    ev = normalize_for_score(out.get("expected_value_per_unit", pd.Series(index=out.index, dtype=float)))
    edge = normalize_for_score(out.get("edge", pd.Series(index=out.index, dtype=float)))
    return_score = 0.70 * ev + 0.30 * edge
    out["tethered_score"] = prob_weight * prob + (1.0 - prob_weight) * return_score
    return out


def image_if_exists(name: str, use_container_width: bool = True):
    path = ASSET_DIR / name
    if path.exists():
        st.image(str(path), use_container_width=use_container_width)


def inject_css():
    st.markdown(
        """
        <style>
        .stApp { background: #08131F; color: #EAF6FF; }
        div[data-testid="stSidebar"] { background: #0B1B2C; }
        .metric-card {
            background: linear-gradient(135deg, rgba(25, 226, 199, 0.12), rgba(84, 123, 255, 0.10));
            border: 1px solid rgba(76, 235, 220, 0.25);
            border-radius: 18px;
            padding: 18px;
            min-height: 118px;
            box-shadow: 0 12px 35px rgba(0,0,0,0.25);
        }
        .bet-card {
            background: #0E2236;
            border: 1px solid rgba(132, 221, 255, 0.18);
            border-radius: 20px;
            padding: 18px;
            margin-bottom: 12px;
        }
        .small-muted { color: #9FB8CC; font-size: 0.9rem; }
        .tag {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            background: rgba(25, 226, 199, 0.15);
            border: 1px solid rgba(25, 226, 199, 0.30);
            color: #66FFF0;
            font-size: 0.85rem;
            margin-right: 6px;
        }
        .warning-box {
            background: rgba(255, 190, 90, 0.09);
            border: 1px solid rgba(255, 190, 90, 0.25);
            border-radius: 14px;
            padding: 12px 16px;
            color: #F8D89B;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_overview():
    with st.expander("How TetheredAI works", expanded=False):
        st.markdown(
            """
            **TetheredAI** is a sports intelligence dashboard that combines historical performance, current schedule context,
            starting-pitcher signals, bullpen workload, team form, matchup splits, market odds, and model-calibrated probabilities.

            For each MLB game, the system estimates win probability, compares it with the market's no-vig probability when odds are
            available, and flags possible edges only when both the probability gap and expected value clear configurable thresholds.

            This public-facing view intentionally avoids revealing proprietary weighting details. It is designed to explain the decision
            framework without exposing the full model internals.
            """
        )
        st.markdown(
            """
            <div class="warning-box">
            TetheredAI is for research and analytics. It is not a guarantee of profit. Bet responsibly and follow applicable laws.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_top_bets(df: pd.DataFrame, prob_weight: float, n: int = 3):
    if df.empty:
        st.info("No prediction rows available yet.")
        return
    bettable = df[df.get("recommended_side", "").fillna("").astype(str).str.len() > 0].copy()
    if bettable.empty:
        st.info("No games currently meet the edge and expected-value thresholds.")
        return
    bettable = add_display_score(bettable, prob_weight).sort_values("tethered_score", ascending=False).head(n)
    cols = st.columns(n)
    for col, (_, row) in zip(cols, bettable.iterrows()):
        with col:
            st.markdown(
                f"""
                <div class="bet-card">
                    <span class="tag">Top Bet</span>
                    <h3>{row.get('recommended_side','')}</h3>
                    <div class="small-muted">{row.get('away_team_name','')} @ {row.get('home_team_name','')}</div>
                    <br/>
                    <b>Price:</b> {fmt_price(row.get('recommended_price'))}<br/>
                    <b>Model probability:</b> {fmt_pct(row.get('recommended_model_prob'))}<br/>
                    <b>Edge:</b> {fmt_pct(row.get('edge'))}<br/>
                    <b>EV / unit:</b> {fmt_units(row.get('expected_value_per_unit'))}<br/>
                    <b>Tethered Score:</b> {float(row.get('tethered_score', 0)):.3f}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_mlb_h2h(prob_weight: float):
    df = load_predictions(str(MLB_H2H_PATH))
    st.subheader("MLB Head-2-Head")
    if df.empty:
        st.warning(f"No MLB moneyline prediction file found at `{MLB_H2H_PATH}`. Run the GitHub scoring workflow first.")
        return
    now = pd.Timestamp.utcnow()
    if "game_datetime_utc" in df.columns:
        df = df[df["game_datetime_utc"].isna() | (df["game_datetime_utc"] >= now - pd.Timedelta(hours=8))].copy()
    render_top_bets(df, prob_weight=prob_weight, n=3)

    st.markdown("### Upcoming games")
    view = df.copy()
    if "recommended_side" in view.columns:
        view["bet_status"] = np.where(view["recommended_side"].fillna("").astype(str).str.len() > 0, "BET", "PASS")
    show_cols = [
        "game_datetime_utc", "away_team_name", "home_team_name",
        "model_home_win_prob", "model_away_win_prob",
        "home_moneyline_median", "away_moneyline_median",
        "recommended_side", "recommended_price", "edge", "expected_value_per_unit",
        "suggested_units", "no_bet_reason",
    ]
    show_cols = [c for c in show_cols if c in view.columns]
    display = view[show_cols].copy()
    for c in ["model_home_win_prob", "model_away_win_prob", "edge"]:
        if c in display.columns:
            display[c] = display[c].map(lambda x: fmt_pct(x) if pd.notna(x) else "—")
    for c in ["home_moneyline_median", "away_moneyline_median", "recommended_price"]:
        if c in display.columns:
            display[c] = display[c].map(fmt_price)
    for c in ["expected_value_per_unit", "suggested_units"]:
        if c in display.columns:
            display[c] = display[c].map(fmt_units)
    st.dataframe(display, use_container_width=True, hide_index=True)


def render_placeholder(market: str):
    st.subheader(f"MLB {market}")
    st.info("This market is wired into the UI but the model/prediction file has not been productionalized yet. The next step is to add market-specific targets and scoring scripts for this tab.")
    st.markdown(
        """
        Planned model inputs:
        - market-specific target and odds history
        - starting pitcher quality
        - bullpen fatigue
        - team offense/pitching form
        - park and run-environment proxies
        - no-vig market probability and line movement, once enough history exists
        """
    )


def main():
    inject_css()
    image_if_exists("tetheredai_hero_banner_wide.png")
    st.title("TetheredAI Sports Intelligence")
    st.caption("Model-assisted sports market analysis. MLB first; NFL and golf are designed as future layers.")
    render_overview()

    with st.sidebar:
        logo = ASSET_DIR / "tetheredai_logo_mark_square.png"
        if logo.exists():
            st.image(str(logo), width=120)
        st.header("Controls")
        sport = st.selectbox("Sport", ["MLB", "NFL", "Golf"], index=0)
        if sport == "MLB":
            market = st.selectbox("Market", ["Head-2-Head", "Spread", "Total Runs"], index=0)
        else:
            market = st.selectbox("Market", ["Coming Soon"], index=0)
        prob_weight = st.slider("Top-bet score: probability weight", 0.0, 1.0, 0.55, 0.05)
        st.caption("Lower values emphasize return/EV; higher values emphasize model probability.")

    if sport != "MLB":
        st.info(f"{sport} support is planned. MLB is the active model layer right now.")
        image_if_exists("tetheredai_app_mockup.png")
        return

    if market == "Head-2-Head":
        render_mlb_h2h(prob_weight)
    elif market == "Spread":
        render_placeholder("Spread")
    elif market == "Total Runs":
        render_placeholder("Total Runs")


if __name__ == "__main__":
    main()
