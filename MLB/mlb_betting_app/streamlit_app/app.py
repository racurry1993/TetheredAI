from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st


APP_TITLE = "TetheredAI"
APP_TAGLINE = "Data. Models. Edge."


def find_project_root() -> Path:
    """Find a repo/project root that contains data/predictions or assets."""
    candidates = [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
    seen: set[Path] = set()
    for path in candidates:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        if (path / "data" / "predictions").exists() or (path / "MLB" / "mlb_betting_app").exists():
            return path
    return Path.cwd().resolve()


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


PROJECT_ROOT = find_project_root()
APP_DIR = Path(__file__).resolve().parent
ASSET_DIR = APP_DIR / "assets"


st.set_page_config(
    page_title="TetheredAI | Sports Edge",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
:root {
    --bg-main: #050b13;
    --bg-card: rgba(15, 29, 43, 0.82);
    --bg-card-2: rgba(7, 18, 31, 0.96);
    --text-main: #f7fbff;
    --text-muted: #9fb1c5;
    --accent: #23e6df;
    --accent-2: #1d75ff;
    --good: #28f0a0;
    --warn: #ffd166;
    --bad: #ff5c7a;
    --border: rgba(92, 246, 255, 0.16);
}
.stApp {
    background:
        radial-gradient(circle at top left, rgba(35, 230, 223, 0.10), transparent 34%),
        radial-gradient(circle at 85% 10%, rgba(29, 117, 255, 0.13), transparent 30%),
        linear-gradient(180deg, #050b13 0%, #08111d 42%, #04080d 100%);
    color: var(--text-main);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #06111d 0%, #03070c 100%);
    border-right: 1px solid rgba(35, 230, 223, 0.12);
}
.block-container {
    padding-top: 1.25rem;
    padding-bottom: 3rem;
}
h1, h2, h3 {
    color: #ffffff;
    letter-spacing: -0.02em;
}
.tai-hero {
    padding: 1.6rem 1.75rem;
    border: 1px solid var(--border);
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(7, 18, 31, 0.92), rgba(9, 34, 54, 0.72));
    box-shadow: 0 18px 52px rgba(0,0,0,0.38);
    margin-bottom: 1.2rem;
}
.tai-title {
    font-size: 2.85rem;
    font-weight: 800;
    margin-bottom: 0.1rem;
}
.tai-title span {
    color: var(--accent);
}
.tai-subtitle {
    color: var(--text-muted);
    font-size: 1.05rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}
.tai-card {
    border: 1px solid var(--border);
    border-radius: 18px;
    background: var(--bg-card);
    padding: 1.05rem 1.1rem;
    box-shadow: 0 12px 34px rgba(0,0,0,0.22);
    min-height: 145px;
}
.tai-card-rank {
    width: 2rem;
    height: 2rem;
    border-radius: 999px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    color: #02111d;
    font-weight: 800;
    margin-right: .55rem;
}
.tai-pill {
    display: inline-block;
    padding: .25rem .55rem;
    border-radius: 999px;
    background: rgba(35, 230, 223, 0.12);
    border: 1px solid rgba(35, 230, 223, 0.18);
    color: #bafaf6;
    font-size: .78rem;
}
.tai-muted { color: var(--text-muted); }
.tai-good { color: var(--good); font-weight: 700; }
.tai-warn { color: var(--warn); font-weight: 700; }
.tai-bad { color: var(--bad); font-weight: 700; }
.tai-section {
    margin-top: 1.4rem;
    margin-bottom: .55rem;
}
.metric-container, [data-testid="stMetric"] {
    background: rgba(5, 16, 28, 0.62);
    border: 1px solid rgba(35, 230, 223, 0.10);
    border-radius: 16px;
    padding: 0.75rem;
}
[data-testid="stDataFrame"] {
    border: 1px solid rgba(35, 230, 223, 0.13);
    border-radius: 16px;
}
hr { border-color: rgba(35,230,223,.10); }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def pct(value: float | int | None, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{100 * float(value):.{decimals}f}%"


def price_fmt(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    ivalue = int(round(float(value)))
    return f"+{ivalue}" if ivalue > 0 else str(ivalue)


def normalize_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize old and new prediction schemas into UI-friendly columns."""
    df = df.copy()

    for col in ["official_date", "game_datetime_utc", "scored_at_utc"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    if "model_home_win_prob" not in df.columns:
        df["model_home_win_prob"] = np.nan
    if "model_away_win_prob" not in df.columns:
        df["model_away_win_prob"] = 1 - df["model_home_win_prob"]

    # Backward-compatible market fields.
    if "market_home_no_vig_prob" not in df.columns:
        df["market_home_no_vig_prob"] = np.nan
    if "market_away_no_vig_prob" not in df.columns:
        df["market_away_no_vig_prob"] = 1 - df["market_home_no_vig_prob"]

    if "edge_home" not in df.columns:
        df["edge_home"] = df["model_home_win_prob"] - df["market_home_no_vig_prob"]
    if "edge_away" not in df.columns:
        df["edge_away"] = df["model_away_win_prob"] - df["market_away_no_vig_prob"]

    if "home_expected_value_per_unit" not in df.columns:
        df["home_expected_value_per_unit"] = np.nan
    if "away_expected_value_per_unit" not in df.columns:
        df["away_expected_value_per_unit"] = np.nan

    # Older scoring output used generic edge/EV fields for the recommended side.
    if "edge" in df.columns:
        df["recommended_edge"] = df["edge"]
    else:
        df["recommended_edge"] = np.nan

    if "expected_value_per_unit" in df.columns:
        df["recommended_ev"] = df["expected_value_per_unit"]
    else:
        df["recommended_ev"] = np.nan

    if "recommended_side" not in df.columns:
        df["recommended_side"] = ""
    if "recommended_price" not in df.columns:
        df["recommended_price"] = np.nan

    df["has_market_odds"] = df[["home_moneyline_median", "away_moneyline_median"]].notna().all(axis=1) if {
        "home_moneyline_median", "away_moneyline_median"
    }.issubset(df.columns) else False

    # If no explicit recommended side, create a model lean for display only.
    empty_rec = df["recommended_side"].isna() | (df["recommended_side"].astype(str).str.strip() == "")
    model_lean = np.where(df["model_home_win_prob"] >= 0.5, df.get("home_team_name", "Home"), df.get("away_team_name", "Away"))
    df["display_side"] = df["recommended_side"].where(~empty_rec, model_lean)
    df["is_recommended_bet"] = ~empty_rec

    df["display_prob"] = np.where(
        df["display_side"].astype(str).eq(df.get("home_team_name", "" ).astype(str)),
        df["model_home_win_prob"],
        df["model_away_win_prob"],
    )

    # Fill recommendation edge/EV from side-specific fields when available.
    home_side = df["display_side"].astype(str).eq(df.get("home_team_name", "").astype(str))
    df["display_edge"] = df["recommended_edge"]
    df.loc[df["display_edge"].isna() & home_side, "display_edge"] = df.loc[df["display_edge"].isna() & home_side, "edge_home"]
    df.loc[df["display_edge"].isna() & ~home_side, "display_edge"] = df.loc[df["display_edge"].isna() & ~home_side, "edge_away"]

    df["display_ev"] = df["recommended_ev"]
    df.loc[df["display_ev"].isna() & home_side, "display_ev"] = df.loc[df["display_ev"].isna() & home_side, "home_expected_value_per_unit"]
    df.loc[df["display_ev"].isna() & ~home_side, "display_ev"] = df.loc[df["display_ev"].isna() & ~home_side, "away_expected_value_per_unit"]

    if "no_bet_reason" not in df.columns:
        df["no_bet_reason"] = np.where(
            df["has_market_odds"],
            "No qualifying edge",
            "No market odds available",
        )

    df["matchup"] = df.get("away_team_name", "Away").astype(str) + " @ " + df.get("home_team_name", "Home").astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_predictions(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    return normalize_predictions(pd.read_csv(path))


def locate_predictions_file() -> Path | None:
    candidates = [
        PROJECT_ROOT / "data" / "predictions" / "mlb_moneyline_predictions.csv",
        PROJECT_ROOT / "MLB" / "mlb_betting_app" / "data" / "predictions" / "mlb_moneyline_predictions.csv",
        APP_DIR.parent / "data" / "predictions" / "mlb_moneyline_predictions.csv",
        APP_DIR / "data" / "predictions" / "mlb_moneyline_predictions.csv",
    ]
    return first_existing(candidates)


def render_overview() -> None:
    st.markdown("### Overview")
    st.markdown(
        """
        TetheredAI is a sports analytics dashboard that combines historical game outcomes,
        current betting markets, and pre-game model features to estimate probabilities for upcoming games.
        The site is designed to show where the model's probability differs from the market, surface potential value,
        and track results over time.

        At a high level, the workflow is: collect schedule/results and odds snapshots, build leakage-safe pre-game
        features, train calibrated prediction models, compare model probabilities to no-vig market probabilities,
        and rank opportunities by a blend of confidence and expected return. The details behind the model can evolve
        without exposing proprietary feature logic in the user interface.
        """
    )


def render_top_bets(df: pd.DataFrame, prob_weight: float, return_weight: float) -> None:
    st.markdown("<div class='tai-section'><h3>Top 3 Best Bets</h3></div>", unsafe_allow_html=True)

    candidates = df.copy()
    candidates = candidates[candidates["is_recommended_bet"]]
    candidates = candidates[candidates["display_prob"].notna()]
    candidates = candidates[candidates["display_ev"].notna()]
    candidates = candidates[candidates["display_ev"] > 0]

    if candidates.empty:
        st.info("No qualifying best bets are available yet. Pull fresh odds or lower the scoring thresholds once the model is validated.")
        return

    prob_component = ((candidates["display_prob"] - 0.50) / 0.25).clip(lower=0, upper=1)
    return_component = (candidates["display_ev"] / 0.25).clip(lower=0, upper=1)
    candidates["tethered_score"] = prob_weight * prob_component + return_weight * return_component
    top = candidates.sort_values(["tethered_score", "display_ev", "display_prob"], ascending=False).head(3)

    cols = st.columns(3)
    for idx, (_, row) in enumerate(top.iterrows(), start=1):
        with cols[idx - 1]:
            st.markdown(
                f"""
                <div class='tai-card'>
                    <div><span class='tai-card-rank'>{idx}</span><span class='tai-pill'>MLB · Head-2-Head</span></div>
                    <h4 style='margin-bottom:.15rem'>{row.get('matchup', 'Matchup')}</h4>
                    <div class='tai-muted'>Bet: <b style='color:white'>{row.get('display_side', '—')}</b> · Price {price_fmt(row.get('recommended_price'))}</div>
                    <hr/>
                    <div style='display:flex; gap:1.2rem; flex-wrap:wrap'>
                        <div><div class='tai-muted'>Probability</div><div class='tai-good'>{pct(row.get('display_prob'))}</div></div>
                        <div><div class='tai-muted'>Edge</div><div class='tai-good'>{pct(row.get('display_edge'))}</div></div>
                        <div><div class='tai-muted'>Expected Return</div><div class='tai-good'>{pct(row.get('display_ev'))}</div></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_h2h(df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("No predictions file found yet. Run the GitHub Actions pipeline and commit/download `data/predictions/mlb_moneyline_predictions.csv`.")
        return

    prob_weight_pct = st.sidebar.slider("Top Bets: Probability Weight", 0, 100, 55, 5)
    prob_weight = prob_weight_pct / 100
    return_weight = 1 - prob_weight
    st.sidebar.caption(f"Return weight: {int(return_weight * 100)}%")

    render_top_bets(df, prob_weight=prob_weight, return_weight=return_weight)

    st.markdown("<div class='tai-section'><h3>Upcoming Games</h3></div>", unsafe_allow_html=True)

    if "game_datetime_utc" in df.columns:
        min_date = df["game_datetime_utc"].min()
        max_date = df["game_datetime_utc"].max()
        st.caption(f"Prediction window: {min_date} → {max_date}")

    table = df.copy()
    table["Game Time UTC"] = table["game_datetime_utc"].dt.strftime("%Y-%m-%d %H:%M") if "game_datetime_utc" in table.columns else ""
    table["Model Home Win %"] = table["model_home_win_prob"].map(lambda x: pct(x))
    table["Model Away Win %"] = table["model_away_win_prob"].map(lambda x: pct(x))
    table["Best Bet / Lean"] = table["display_side"]
    table["Bettable?"] = np.where(table["is_recommended_bet"], "✅", "—")
    table["Edge"] = table["display_edge"].map(lambda x: pct(x))
    table["Expected Return"] = table["display_ev"].map(lambda x: pct(x))
    table["Price"] = table["recommended_price"].map(price_fmt)
    table["Reason"] = np.where(table["is_recommended_bet"], "Qualified", table["no_bet_reason"])

    display_cols = [
        "Game Time UTC", "matchup", "Best Bet / Lean", "Bettable?", "Price",
        "Model Home Win %", "Model Away Win %", "Edge", "Expected Return", "Reason",
    ]
    display_cols = [c for c in display_cols if c in table.columns]
    st.dataframe(table[display_cols], use_container_width=True, hide_index=True)

    with st.expander("Raw prediction diagnostics"):
        diag_cols = [
            "run_id", "scored_at_utc", "game_pk", "official_date", "home_team_name", "away_team_name",
            "model_home_win_prob", "market_home_no_vig_prob", "home_moneyline_median", "away_moneyline_median",
            "recommended_side", "recommended_price", "display_edge", "display_ev",
        ]
        st.dataframe(df[[c for c in diag_cols if c in df.columns]], use_container_width=True, hide_index=True)


def render_placeholder_market(name: str) -> None:
    st.markdown(f"### {name}")
    st.info(
        f"{name} is planned for the next model layer. The UI route is already reserved so the same TetheredAI layout can support MLB, NFL, golf, and additional bet types over time."
    )
    st.markdown(
        """
        Planned fields:
        - model probability
        - market no-vig probability
        - edge
        - expected value
        - confidence tier
        - performance by threshold
        """
    )


# Sidebar branding
logo_path = first_existing([
    ASSET_DIR / "tetheredai_logo_lockup_dark.png",
    ASSET_DIR / "tetheredai_logo_lockup_wide.png",
])
if logo_path:
    st.sidebar.image(str(logo_path), use_container_width=True)
else:
    st.sidebar.markdown(f"# {APP_TITLE}")
st.sidebar.caption(APP_TAGLINE)

sport = st.sidebar.selectbox("Sport", ["MLB", "NFL", "Golf"], index=0)
market = st.sidebar.selectbox("Market", ["Head-2-Head", "Spread", "Total Runs"], index=0)

pred_path = locate_predictions_file()
if pred_path:
    st.sidebar.success(f"Predictions loaded: {pred_path.name}")
    predictions = load_predictions(str(pred_path))
else:
    st.sidebar.error("Predictions file not found")
    predictions = pd.DataFrame()

# Hero
hero_path = first_existing([
    ASSET_DIR / "tetheredai_hero_banner_wide.png",
    ASSET_DIR / "tetheredai_hero_banner.png",
])
if hero_path:
    st.image(str(hero_path), use_container_width=True)
else:
    st.markdown(
        f"""
        <div class='tai-hero'>
            <div class='tai-title'>Tethered<span>AI</span></div>
            <div class='tai-subtitle'>{APP_TAGLINE}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class='tai-hero'>
        <div class='tai-title'>Tethered<span>AI</span> Sports Edge</div>
        <div class='tai-subtitle'>AI-powered probabilities · market edges · betting discipline</div>
    </div>
    """,
    unsafe_allow_html=True,
)

render_overview()

st.markdown("---")

if sport != "MLB":
    st.info(f"{sport} support is coming later. The app shell is ready for multi-sport routing.")
else:
    st.markdown("## MLB")
    if market == "Head-2-Head":
        render_h2h(predictions)
    elif market == "Spread":
        render_placeholder_market("Spread")
    elif market == "Total Runs":
        render_placeholder_market("Total Runs")

# Footer visual
stadium_path = first_existing([ASSET_DIR / "tetheredai_stadium_banner_wide.png", ASSET_DIR / "tetheredai_stadium_banner.png"])
if stadium_path:
    st.markdown("---")
    st.image(str(stadium_path), use_container_width=True)
