"""
NBA Scouting Dashboard
------------------------
Streamlit app for scouting a single player: a hex bin shot chart that can
be sliced by shot type, next to their shooting splits with league
percentiles.

Run with: streamlit run dashboard.py

Requires: pip install -r requirements.txt
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib import colormaps
from matplotlib.colors import to_hex

from plot_functions import plot_shot_hexbin
from processing_functions import (
    SHOOTING_METRICS,
    SHOT_TYPE_LABELS,
    add_shot_type_column,
    build_shooting_splits,
)

SEASON = "2025-26"
DATA_DIR = Path(__file__).parent / "data"
SHOT_CHART_PATH = DATA_DIR / f"nba_shot_chart_{SEASON}.csv"
BOX_STATS_PATH = DATA_DIR / f"nba_player_box_{SEASON}.csv"

# add_shot_type_column's default new_col would overwrite shotchartdetail's own
# SHOT_TYPE column (the "2PT Field Goal" / "3PT Field Goal" label), which the
# shooting splits rely on, so the classification lands in its own column.
SHOT_TYPE_GROUP_COL = "SHOT_TYPE_GROUP"


@st.cache_data
def load_shots() -> pd.DataFrame:
    """Load per-shot data and tag each shot with its shot type category."""
    shots = pd.read_csv(SHOT_CHART_PATH)
    return add_shot_type_column(shots, new_col=SHOT_TYPE_GROUP_COL)


@st.cache_data
def load_box_stats() -> pd.DataFrame | None:
    """Load season box score totals, or None if they haven't been pulled yet."""
    if not BOX_STATS_PATH.exists():
        return None
    return pd.read_csv(BOX_STATS_PATH)


@st.cache_data
def load_splits() -> pd.DataFrame:
    """Build the league-wide shooting profile dataset with percentiles."""
    return build_shooting_splits(load_shots(), box=load_box_stats())


def shooting_profile(splits: pd.DataFrame, player_id: int) -> pd.DataFrame:
    """One row per shooting metric for a player: value, attempts, percentile."""
    row = splits.loc[splits["PLAYER_ID"] == player_id].iloc[0]

    return pd.DataFrame(
        [
            {
                "Metric": label,
                "Value": pd.to_numeric(row[value_col], errors="coerce"),
                "Attempts": pd.to_numeric(row[attempts_col], errors="coerce"),
                "League %ile": row[f"{value_col}_PCTILE"],
            }
            for value_col, label, attempts_col, _min_attempts in SHOOTING_METRICS
        ]
    )


def percentile_css(percentile: float) -> str:
    """Red-to-green background for a 0-100 percentile, blank if unranked."""
    if pd.isna(percentile):
        return ""
    background = to_hex(colormaps["RdYlGn"](float(percentile) / 100))
    # Fixed dark text: the colormap is light in the middle, so inheriting the
    # theme's font color would wash out in dark mode.
    return f"background-color: {background}; color: #262730"


def style_profile(profile: pd.DataFrame):
    """Format the profile table and color the percentile column red to green.

    Values are formatted to strings up front rather than left to the Styler,
    because st.dataframe renders the underlying data through Arrow and shows
    missing entries as "None" instead of honouring the Styler's na_rep.
    """
    percentiles = pd.to_numeric(profile["League %ile"], errors="coerce")

    display = pd.DataFrame(
        {
            "Metric": profile["Metric"],
            "Value": [f"{v:.1%}" if pd.notna(v) else "—" for v in profile["Value"]],
            "Attempts": [f"{a:,.0f}" if pd.notna(a) else "—" for a in profile["Attempts"]],
            "League %ile": [f"{p:.0f}" if pd.notna(p) else "—" for p in percentiles],
        }
    )
    styles = [percentile_css(p) for p in percentiles]

    return display.style.apply(lambda _column: styles, subset=["League %ile"])


st.set_page_config(page_title="NBA Scouting", layout="wide")

shots = load_shots()
splits = load_splits()
has_box_stats = load_box_stats() is not None

(scouting_tab,) = st.tabs(["Scouting"])

with scouting_tab:
    teams = sorted(splits["TEAM_NAME"].dropna().unique())

    team_col, player_col, type_col = st.columns([1, 1, 2])

    with team_col:
        team = st.selectbox("Team", teams)

    roster = splits.loc[splits["TEAM_NAME"] == team].sort_values("PLAYER_NAME")
    with player_col:
        player_name = st.selectbox("Player", roster["PLAYER_NAME"].tolist())

    player_id = int(roster.loc[roster["PLAYER_NAME"] == player_name, "PLAYER_ID"].iloc[0])
    player_shots = shots.loc[shots["PLAYER_ID"] == player_id]

    # Shot type options come straight from the dummy variable categories, in
    # the same priority order, showing only the types this player actually has.
    counts = player_shots[SHOT_TYPE_GROUP_COL].value_counts()
    options = [
        f"{SHOT_TYPE_LABELS[key]} ({counts[key]})"
        for key in SHOT_TYPE_LABELS
        if key in counts.index
    ]
    option_to_key = {
        f"{SHOT_TYPE_LABELS[key]} ({counts[key]})": key
        for key in SHOT_TYPE_LABELS
        if key in counts.index
    }

    with type_col:
        selected = st.multiselect("Shot types", options, default=options)

    selected_keys = [option_to_key[option] for option in selected]
    chart_shots = player_shots.loc[player_shots[SHOT_TYPE_GROUP_COL].isin(selected_keys)]

    st.divider()

    chart_col, profile_col = st.columns([3, 2])

    with chart_col:
        color_by = st.radio(
            "Color hexagons by", ["FG%", "Volume"], horizontal=True, label_visibility="collapsed"
        )

        if chart_shots.empty:
            st.info("Select at least one shot type to draw the shot chart.")
        else:
            fig, ax = plt.subplots(figsize=(7, 6.6))
            plot_shot_hexbin(
                chart_shots,
                ax=ax,
                gridsize=25,
                mincnt=2,
                color_by="fg_pct" if color_by == "FG%" else "count",
                title=f"{player_name} — {SEASON} ({len(chart_shots):,} shots)",
            )
            st.pyplot(fig)
            plt.close(fig)

    with profile_col:
        st.subheader("Shooting profile")
        st.dataframe(style_profile(shooting_profile(splits, player_id)), hide_index=True)
        st.caption(
            "Season totals across all teams. Percentiles rank the player against "
            "the league on each metric, among players with enough attempts to qualify."
        )
        if not has_box_stats:
            st.warning(
                "True shooting % and free throw % need season box score totals. "
                f"Run `pull_player_box_stats()` from scraper_functions.py and save the "
                f"result to `data/{BOX_STATS_PATH.name}`."
            )
