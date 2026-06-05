from __future__ import annotations

import base64
import os
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# App config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TetheredAI Sports Predictions",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1] if "streamlit_app" in str(Path(__file__).resolve()) else Path.cwd()
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
PREDICTIONS_DIR = DATA_DIR / "predictions"
DEFAULT_PREDICTIONS_FILE = PREDICTIONS_DIR / "mlb_moneyline_predictions.csv"
APP_TIMEZONE = os.getenv("APP_TIMEZONE", "America/Chicago")


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
:root {
    --bg-main: #0b1020;
    --card-bg: rgba(255, 255, 255, 0.075);
    --card-border: rgba(255, 255, 255, 0.16);
    --text-main: #f8fafc;
    --text-soft: #cbd5e1;
    --text-muted: #94a3b8;
    --good: #2dd4bf;
    --warn: #fbbf24;
    --bad: #fb7185;
    --accent: #60a5fa;
    --accent-2: #a78bfa;
}

.stApp {
    background: radial-gradient(circle at top left, #13213f 0, #0b1020 38%, #050814 100%);
}

.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2rem;
}

.hero {
    position: relative;
    padding: 2.1rem 2rem 1.6rem 2rem;
    border: 1px solid rgba(255,255,255,.14);
    border-radius: 24px;
    overflow: hidden;
    background: linear-gradient(135deg, rgba(37,99,235,.62), rgba(88,28,135,.48)), var(--hero-image, none);
    background-position: center;
    background-size: cover;
    box-shadow: 0 22px 55px rgba(0,0,0,.32);
    margin-bottom: 1.2rem;
}

.hero:before {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, rgba(5,8,20,.88), rgba(5,8,20,.42), rgba(5,8,20,.15));
    z-index: 0;
}

.hero-content {
    position: relative;
    z-index: 1;
    max-width: 980px;
}

.hero-kicker {
    display: inline-flex;
    align-items: center;
    gap: .4rem;
    padding: .32rem .68rem;
    border-radius: 999px;
    background: rgba(96,165,250,.20);
    color: #bfdbfe;
    font-size: .84rem;
    font-weight: 700;
    letter-spacing: .02em;
    border: 1px solid rgba(147,197,253,.35);
}

.hero-title {
    margin: .7rem 0 .35rem 0;
    color: var(--text-main);
    font-size: clamp(2rem, 4vw, 4.2rem);
    line-height: .96;
    font-weight: 850;
    letter-spacing: -0.055em;
}

.hero-subtitle {
    color: var(--text-soft);
    max-width: 760px;
    font-size: 1.04rem;
    line-height: 1.55;
    margin: 0 0 1.15rem 0;
}

.league-nav {
    display: flex;
    flex-wrap: wrap;
    gap: .7rem;
    margin-top: 1rem;
}

.league-pill {
    text-decoration: none !important;
    color: #f8fafc !important;
    border: 1px solid rgba(255,255,255,.2);
    background: rgba(255,255,255,.10);
    backdrop-filter: blur(8px);
    border-radius: 999px;
    padding: .68rem 1.05rem;
    min-width: 92px;
    text-align: center;
    font-weight: 800;
    letter-spacing: .03em;
    box-shadow: 0 8px 20px rgba(0,0,0,.18);
    transition: transform .12s ease, border-color .12s ease, background .12s ease;
}

.league-pill:hover {
    transform: translateY(-1px);
    border-color: rgba(125,211,252,.78);
    background: rgba(96,165,250,.24);
}

.league-pill.active {
    background: linear-gradient(135deg, rgba(34,211,238,.38), rgba(96,165,250,.30));
    border-color: rgba(125,211,252,.8);
}

.metric-card, .bet-card, .game-card {
    border: 1px solid var(--card-border);
    background: linear-gradient(180deg, rgba(255,255,255,.09), rgba(255,255,255,.052));
    border-radius: 18px;
    padding: 1rem 1.05rem;
    box-shadow: 0 16px 34px rgba(0,0,0,.18);
}

.metric-label {
    color: var(--text-muted);
    font-size: .78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .06em;
}

.metric-value {
    color: var(--text-main);
    font-size: 1.7rem;
    font-weight: 850;
    margin-top: .12rem;
}

.metric-help {
    color: var(--text-soft);
    font-size: .86rem;
    margin-top: .25rem;
}

.section-title {
    color: var(--text-main);
    font-size: 1.35rem;
    font-weight: 850;
    margin: 1.35rem 0 .7rem 0;
}

.bet-card {
    min-height: 216px;
    position: relative;
}

.bet-rank {
    position: absolute;
    top: 1rem;
    right: 1rem;
    color: #020617;
    background: linear-gradient(135deg, #fbbf24, #fde68a);
    border-radius: 999px;
    width: 34px;
    height: 34px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 900;
}

.bet-matchup, .game-matchup {
    color: var(--text-main);
    font-size: 1.03rem;
    font-weight: 800;
    padding-right: 2rem;
}

.bet-date, .game-date {
    color: var(--text-muted);
    font-size: .82rem;
    margin-top: .15rem;
}

.recommend-line {
    color: var(--text-main);
    font-size: 1.25rem;
    font-weight: 900;
    margin-top: .7rem;
}

.prob-line {
    color: var(--text-soft);
    font-size: .92rem;
    margin-top: .25rem;
}

.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: .42rem;
    margin-top: .78rem;
}

.chip {
    border: 1px solid rgba(255,255,255,.16);
    border-radius: 999px;
    padding: .35rem .58rem;
    font-size: .82rem;
    font-weight: 750;
    color: #dbeafe;
    background: rgba(59,130,246,.16);
}

.chip.good {
    color: #ccfbf1;
    background: rgba(45,212,191,.16);
    border-color: rgba(45,212,191,.32);
}

.chip.pass {
    color: #e2e8f0;
    background: rgba(148,163,184,.16);
    border-color: rgba(148,163,184,.25);
}

.chip.warn {
    color: #fef3c7;
    background: rgba(251,191,36,.16);
    border-color: rgba(251,191,36,.30);
}

.game-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(310px, 1fr));
    gap: .85rem;
}

.game-card.recommend {
    border-color: rgba(45,212,191,.38);
}

.game-card.pass {
    border-color: rgba(148,163,184,.18);
}

.status-badge {
    display: inline-flex;
    border-radius: 999px;
    padding: .24rem .55rem;
    font-size: .75rem;
    font-weight: 900;
    letter-spacing: .04em;
    text-transform: uppercase;
}

.status-badge.recommend {
    color: #022c22;
    background: #5eead4;
}

.status-badge.pass {
    color: #0f172a;
    background: #cbd5e1;
}

.table-wrap {
    border: 1px solid rgba(255,255,255,.12);
    border-radius: 18px;
    padding: .8rem;
    background: rgba(255,255,255,.055);
}

/* Streamlit dataframe tweaks */
[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
}


/* Make Streamlit expanders stand out */
div[data-testid="stExpander"] {
    border: 1px solid rgba(96,165,250,.62) !important;
    border-radius: 18px !important;
    background: rgba(15, 23, 42, .76) !important;
    box-shadow: 0 14px 34px rgba(0,0,0,.24) !important;
    margin-top: 1rem !important;
    margin-bottom: 1.15rem !important;
    overflow: hidden !important;
}

div[data-testid="stExpander"] details summary {
    color: #ffffff !important;
    font-weight: 900 !important;
    font-size: 1rem !important;
    background: linear-gradient(90deg, rgba(37,99,235,.96), rgba(14,165,233,.86)) !important;
    border-radius: 14px !important;
    padding: .92rem 1rem !important;
}

div[data-testid="stExpander"] details summary:hover {
    filter: brightness(1.08);
    cursor: pointer;
}

div[data-testid="stExpander"] div[data-testid="stMarkdownContainer"] {
    color: #e5e7eb !important;
}

.small-note {
    color: var(--text-muted);
    font-size: .86rem;
    margin-top: .25rem;
}

hr {
    border-color: rgba(255,255,255,.10) !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _fmt_pct(value: Any, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 100:.{decimals}f}%"


def _fmt_decimal(value: Any, decimals: int = 3) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{decimals}f}"


def _fmt_moneyline(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    value = int(round(float(value)))
    return f"+{value}" if value > 0 else str(value)


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    return str(value).replace("_", " ").title()


def _is_recommended(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "none", "pass"}


def _best_edge(row: pd.Series) -> float | None:
    rec_edge = row.get("edge")
    if pd.notna(rec_edge):
        return float(rec_edge)

    edge_home = row.get("edge_home")
    edge_away = row.get("edge_away")
    edges = [float(x) for x in [edge_home, edge_away] if pd.notna(x)]
    return max(edges, key=abs) if edges else None


def _expected_side(row: pd.Series) -> tuple[str, float | None]:
    home_prob = row.get("model_home_win_prob")
    away_prob = row.get("model_away_win_prob")

    if pd.isna(away_prob) and pd.notna(home_prob):
        away_prob = 1.0 - float(home_prob)

    home_team = row.get("home_team_name", "Home")
    away_team = row.get("away_team_name", "Away")

    if pd.notna(home_prob) and pd.notna(away_prob):
        if float(home_prob) >= float(away_prob):
            return str(home_team), float(home_prob)
        return str(away_team), float(away_prob)

    return "—", None


def _read_hero_image_css() -> str:
    candidates = [
        Path("streamlit_app/assets/hero.png"),
        Path("streamlit_app/assets/hero.jpg"),
        Path("streamlit_app/assets/header.png"),
        Path("streamlit_app/assets/header.jpg"),
        Path("assets/hero.png"),
        Path("assets/hero.jpg"),
    ]
    for path in candidates:
        if path.exists():
            mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
            return f"url(data:{mime};base64,{encoded})"
    return "none"


@st.cache_data(ttl=300)
def load_predictions(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        candidates = sorted(PREDICTIONS_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            path = candidates[0]
        else:
            return pd.DataFrame()

    df = pd.read_csv(path)

    if "game_datetime_utc" in df.columns:
        df["game_datetime_utc"] = pd.to_datetime(df["game_datetime_utc"], errors="coerce", utc=True)

    if "official_date" in df.columns:
        df["official_date"] = pd.to_datetime(df["official_date"], errors="coerce").dt.date
    elif "game_datetime_utc" in df.columns:
        df["official_date"] = df["game_datetime_utc"].dt.date

    if "model_away_win_prob" not in df.columns and "model_home_win_prob" in df.columns:
        df["model_away_win_prob"] = 1.0 - df["model_home_win_prob"]

    if "has_market_odds" in df.columns:
        df["has_market_odds"] = df["has_market_odds"].fillna(0).astype(int)

    if "recommended_side" not in df.columns:
        df["recommended_side"] = np.nan

    df["is_recommended"] = df["recommended_side"].apply(_is_recommended)

    if "edge" not in df.columns:
        df["edge"] = np.nan
    df["display_edge"] = df.apply(_best_edge, axis=1)

    expected = df.apply(_expected_side, axis=1)
    df["expected_winner"] = [x[0] for x in expected]
    df["expected_win_prob"] = [x[1] for x in expected]

    sort_cols = [c for c in ["official_date", "game_datetime_utc", "home_team_name"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


def render_hero() -> None:
    hero_image = _read_hero_image_css()
    st.markdown(
        f"""
        <style>
        .hero {{ --hero-image: {hero_image}; }}
        </style>
        <div class="hero">
          <div class="hero-content">
            <div class="hero-kicker">TetheredAI · Sports Intelligence</div>
            <div class="hero-title">Game predictions, odds edges, and recommendations.</div>
            <div class="hero-subtitle">
              Daily model outputs powered by matchup features, market odds, and probability calibration.
            </div>
            <div class="league-nav">
              <a class="league-pill active" href="#mlb">MLB</a>
              <a class="league-pill" href="#nba-coming-soon">NBA</a>
              <a class="league-pill" href="#nfl-coming-soon">NFL</a>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric(label: str, value: str, help_text: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_bet_card(row: pd.Series, rank: int) -> None:
    matchup = f"{row.get('away_team_name', 'Away')} at {row.get('home_team_name', 'Home')}"
    game_date = row.get("official_date", "—")
    side = row.get("recommended_side") if _is_recommended(row.get("recommended_side")) else "—"
    price = row.get("recommended_price")
    edge = row.get("edge") if pd.notna(row.get("edge")) else row.get("display_edge")
    ev = row.get("expected_value_per_unit")
    model_prob = row.get("recommended_model_prob")
    market_prob = row.get("recommended_market_prob")

    st.markdown(
        f"""
        <div class="bet-card">
          <div class="bet-rank">{rank}</div>
          <div class="bet-matchup">{matchup}</div>
          <div class="bet-date">{game_date}</div>
          <div class="recommend-line">Recommend: {side}</div>
          <div class="prob-line">Price: <b>{_fmt_moneyline(price)}</b></div>
          <div class="chip-row">
            <span class="chip good">Edge {_fmt_pct(edge)}</span>
            <span class="chip good">EV {_fmt_decimal(ev)}</span>
            <span class="chip">Model {_fmt_pct(model_prob)}</span>
            <span class="chip">Market {_fmt_pct(market_prob)}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_game_card(row: pd.Series) -> None:
    recommended = bool(row.get("is_recommended"))
    status = "Recommend" if recommended else "Pass"
    status_class = "recommend" if recommended else "pass"
    matchup = f"{row.get('away_team_name', 'Away')} at {row.get('home_team_name', 'Home')}"
    game_date = row.get("official_date", "—")

    expected_winner = row.get("expected_winner", "—")
    expected_prob = row.get("expected_win_prob")
    display_edge = row.get("display_edge")

    if recommended:
        main_side = row.get("recommended_side", "—")
        main_label = f"Recommended: {main_side}"
        price = _fmt_moneyline(row.get("recommended_price"))
        reason = f"Price {price}"
    else:
        main_label = "Pass"
        reason = _clean_text(row.get("no_bet_reason")) if row.get("no_bet_reason") is not None else "No qualifying edge"

    st.markdown(
        f"""
        <div class="game-card {status_class}">
          <span class="status-badge {status_class}">{status}</span>
          <div class="game-matchup" style="margin-top:.65rem;">{matchup}</div>
          <div class="game-date">{game_date}</div>
          <div class="recommend-line">{main_label}</div>
          <div class="prob-line">Expected outcome: <b>{expected_winner}</b> · {_fmt_pct(expected_prob)}</div>
          <div class="chip-row">
            <span class="chip {'good' if recommended else 'pass'}">Edge {_fmt_pct(display_edge)}</span>
            <span class="chip">{reason}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_display_table(df: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame()
    table["Date"] = df.get("official_date", pd.Series(dtype="object")).astype(str)
    table["Matchup"] = df.apply(
        lambda r: f"{r.get('away_team_name', 'Away')} at {r.get('home_team_name', 'Home')}",
        axis=1,
    )
    table["Expected Winner"] = df.get("expected_winner", "—")
    table["Expected Win Prob"] = df.get("expected_win_prob", np.nan).apply(_fmt_pct)
    table["Home Win Prob"] = df.get("model_home_win_prob", np.nan).apply(_fmt_pct)
    table["Away Win Prob"] = df.get("model_away_win_prob", np.nan).apply(_fmt_pct)
    table["Home ML"] = df.get("home_moneyline_median", np.nan).apply(_fmt_moneyline)
    table["Away ML"] = df.get("away_moneyline_median", np.nan).apply(_fmt_moneyline)
    table["Best Edge"] = df.get("display_edge", np.nan).apply(_fmt_pct)
    table["Recommendation"] = df.apply(
        lambda r: r.get("recommended_side") if _is_recommended(r.get("recommended_side")) else "Pass",
        axis=1,
    )
    table["Reason"] = df.get("no_bet_reason", pd.Series([None] * len(df))).apply(_clean_text)
    return table


def render_data_dictionary() -> None:
    dictionary = pd.DataFrame(
        [
            ("Date", "MLB official game date. Displayed as date only."),
            ("Matchup", "Away team at home team."),
            ("Expected Winner", "Team with the higher model win probability."),
            ("Expected Win Prob", "Model probability for the expected winner after shrinkage/calibration."),
            ("Home Win Prob", "Calibrated model probability that the home team wins."),
            ("Away Win Prob", "Calibrated model probability that the away team wins."),
            ("Home ML", "Median available home-team moneyline across matched sportsbooks."),
            ("Away ML", "Median available away-team moneyline across matched sportsbooks."),
            ("Best Edge", "Best available probability edge for either team. For recommended bets, this is the selected edge."),
            ("Recommendation", "Recommended betting side if the edge/EV thresholds pass; otherwise Pass."),
            ("Reason", "Why the game is a pass or additional context from the scorer."),
            ("EV", "Expected value per unit wagered using the model probability and market price."),
        ],
        columns=["Column", "Meaning"],
    )

    with st.expander("📘 Data Dictionary — What Each Column Means", expanded=False):
        st.dataframe(dictionary, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------------
render_hero()

st.markdown('<a id="mlb"></a>', unsafe_allow_html=True)
st.markdown('<div class="section-title">MLB Moneyline Predictions</div>', unsafe_allow_html=True)
st.caption(f"Showing MLB official-date games for {pd.Timestamp.now(tz=ZoneInfo(APP_TIMEZONE)).date()} only.")

with st.sidebar:
    st.header("Controls")
    predictions_path = st.text_input("Predictions CSV", str(DEFAULT_PREDICTIONS_FILE))
    only_market = st.checkbox("Show only games with market odds", value=False)
    only_recommended = st.checkbox("Show only recommendations", value=False)
    st.caption(f"Showing today’s MLB official-date games only ({APP_TIMEZONE}).")
    st.caption("The web app syncs GCS prediction artifacts every few minutes in the background.")

predictions = load_predictions(predictions_path)

today_local = pd.Timestamp.now(tz=ZoneInfo(APP_TIMEZONE)).date()

if not predictions.empty:
    if "official_date" in predictions.columns:
        predictions = predictions[predictions["official_date"].eq(today_local)].copy()
    elif "game_datetime_utc" in predictions.columns:
        predictions = predictions[
            predictions["game_datetime_utc"].dt.tz_convert(APP_TIMEZONE).dt.date.eq(today_local)
        ].copy()

if predictions.empty:
    st.warning(
        f"No predictions found for today ({today_local}). Confirm the daily-score job has run "
        "and uploaded mlb/predictions/mlb_moneyline_predictions.csv to GCS."
    )
    st.stop()

filtered = predictions.copy()
if only_market and "has_market_odds" in filtered.columns:
    filtered = filtered[filtered["has_market_odds"].fillna(0).astype(int).eq(1)]
if only_recommended:
    filtered = filtered[filtered["is_recommended"]]

recommended = predictions[predictions["is_recommended"]].copy()
if "expected_value_per_unit" in recommended.columns:
    recommended = recommended.sort_values("expected_value_per_unit", ascending=False, na_position="last")
elif "edge" in recommended.columns:
    recommended = recommended.sort_values("edge", ascending=False, na_position="last")

market_count = int(predictions.get("has_market_odds", pd.Series([0] * len(predictions))).fillna(0).astype(int).sum())
rec_count = int(predictions["is_recommended"].sum())

metric_cols = st.columns(4)
with metric_cols[0]:
    render_metric("Today’s Games", f"{len(predictions):,}", "MLB official-date games shown")
with metric_cols[1]:
    render_metric("Market Matches", f"{market_count:,}", "Games with moneyline odds")
with metric_cols[2]:
    render_metric("Recommendations", f"{rec_count:,}", "Games passing edge/EV rules")
with metric_cols[3]:
    avg_edge = recommended["edge"].mean() if len(recommended) and "edge" in recommended.columns else np.nan
    render_metric("Avg Rec Edge", _fmt_pct(avg_edge), "Average edge for recommendations")

st.markdown('<div class="section-title">Top 3 Recommended Bets</div>', unsafe_allow_html=True)

if recommended.empty:
    st.info("No recommendations currently pass the betting thresholds.")
else:
    top3 = recommended.head(3)
    cols = st.columns(len(top3))
    for idx, (_, row) in enumerate(top3.iterrows(), start=1):
        with cols[idx - 1]:
            render_bet_card(row, idx)

st.markdown('<div class="section-title">Today’s Game Predictions</div>', unsafe_allow_html=True)
st.caption("Each card shows whether the model recommends a bet or passes, while still showing the expected outcome and edge.")

card_rows = filtered.to_dict(orient="records")
if not card_rows:
    st.info("No games match the selected filters.")
else:
    st.markdown('<div class="game-grid">', unsafe_allow_html=True)
    for row_dict in card_rows:
        render_game_card(pd.Series(row_dict))
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Prediction Table</div>', unsafe_allow_html=True)
render_data_dictionary()

display_table = build_display_table(filtered)

with st.container():
    st.markdown('<div class="table-wrap">', unsafe_allow_html=True)
    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True,
        height=min(780, 38 + 36 * max(1, len(display_table))),
    )
    st.markdown('</div>', unsafe_allow_html=True)

with st.expander("Raw prediction rows", expanded=False):
    st.dataframe(filtered, use_container_width=True, hide_index=True)
