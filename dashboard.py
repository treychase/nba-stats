"""
NBA Scouting Dashboard
------------------------
Streamlit app for scouting a single player.

Scouting tab: a hex bin shot chart that can be sliced by shot type, next
to their shooting splits with league percentiles.

Touches & Pick and Roll tab: where on the court the player gets the ball,
and what they score out of the pick and roll as a ball handler and as a
roll man, all against league percentiles.

Run with: streamlit run dashboard.py

Requires: pip install -r requirements.txt
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib import colormaps
from matplotlib.colors import to_hex

from plot_functions import plot_shot_hexbin, plot_touch_areas
from processing_functions import (
    PNR_ROLES,
    SHOOTING_METRICS,
    SHOT_TYPE_LABELS,
    TOUCH_AREAS,
    add_shot_type_column,
    build_pick_and_roll_profile,
    build_shooting_splits,
    build_touch_profile,
)

SEASON = "2025-26"
DATA_DIR = Path(__file__).parent / "data"
SHOT_CHART_PATH = DATA_DIR / f"nba_shot_chart_{SEASON}.csv"
BOX_STATS_PATH = DATA_DIR / f"nba_player_box_{SEASON}.csv"
POSSESSIONS_PATH = DATA_DIR / f"tracking_possessions_{SEASON}.csv"
PNR_PATH = DATA_DIR / f"nba_pick_and_roll_combined_{SEASON}.csv"

# add_shot_type_column's default new_col would overwrite shotchartdetail's own
# SHOT_TYPE column (the "2PT Field Goal" / "3PT Field Goal" label), which the
# shooting splits rely on, so the classification lands in its own column.
SHOT_TYPE_GROUP_COL = "SHOT_TYPE_GROUP"

PERCENTILE_COL = "League %ile"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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


@st.cache_data
def load_touches() -> pd.DataFrame:
    """Build the league-wide touch location dataset with percentiles."""
    return build_touch_profile(pd.read_csv(POSSESSIONS_PATH))


@st.cache_data
def load_pick_and_roll() -> pd.DataFrame | None:
    """Build the league-wide pick and roll dataset, or None if not pulled yet."""
    if not PNR_PATH.exists():
        return None
    return build_pick_and_roll_profile(pd.read_csv(PNR_PATH))


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def percentile_color(percentile: float) -> str | None:
    """Red-to-green fill for a 0-100 percentile, or None if unranked."""
    if pd.isna(percentile):
        return None
    return to_hex(colormaps["RdYlGn"](float(percentile) / 100))


def percentile_css(percentile: float) -> str:
    """Cell styling for a 0-100 percentile, blank if unranked."""
    background = percentile_color(percentile)
    if background is None:
        return ""
    # Fixed dark text: the colormap is light in the middle, so inheriting the
    # theme's font color would wash out in dark mode.
    return f"background-color: {background}; color: #262730"


def style_table(table: pd.DataFrame, formats: dict):
    """Format a table's numbers and shade its percentile column red to green.

    Values are formatted to strings up front rather than left to the Styler,
    because st.dataframe renders the underlying data through Arrow and shows
    missing entries as "None" instead of honouring the Styler's na_rep.
    """
    percentiles = pd.to_numeric(table[PERCENTILE_COL], errors="coerce")
    formats = {PERCENTILE_COL: "{:.0f}", **formats}

    display = pd.DataFrame(index=table.index)
    for column in table.columns:
        if column in formats:
            spec = formats[column]
            display[column] = [
                spec.format(value) if pd.notna(value) else "—"
                for value in pd.to_numeric(table[column], errors="coerce")
            ]
        else:
            display[column] = table[column].fillna("—")

    styles = [percentile_css(percentile) for percentile in percentiles]
    return display.style.apply(lambda _column: styles, subset=[PERCENTILE_COL])


# ---------------------------------------------------------------------------
# Per-player tables
# ---------------------------------------------------------------------------

def shooting_profile(splits: pd.DataFrame, player_id: int) -> pd.DataFrame:
    """One row per shooting metric: value, attempts, percentile."""
    row = splits.loc[splits["PLAYER_ID"] == player_id].iloc[0]

    return pd.DataFrame(
        [
            {
                "Metric": label,
                "Value": pd.to_numeric(row[value_col], errors="coerce"),
                "Attempts": pd.to_numeric(row[attempts_col], errors="coerce"),
                PERCENTILE_COL: row[f"{value_col}_PCTILE"],
            }
            for value_col, label, attempts_col, _min_attempts in SHOOTING_METRICS
        ]
    )


def touch_profile(touches: pd.DataFrame, player_id: int) -> pd.DataFrame:
    """One row per court area: share of touches, touch count, percentile."""
    row = touches.loc[touches["PLAYER_ID"] == player_id].iloc[0]

    return pd.DataFrame(
        [
            {
                "Area": label,
                "Share of touches": pd.to_numeric(row[share_col], errors="coerce"),
                "Touches": pd.to_numeric(row[count_col], errors="coerce"),
                PERCENTILE_COL: row[f"{share_col}_PCTILE"],
            }
            for share_col, label, count_col, _area_key in TOUCH_AREAS
        ]
    )


def pnr_profile(pnr: pd.DataFrame, player_id: int) -> pd.DataFrame:
    """One row per pick and roll role: points, possessions, percentile."""
    match = pnr.loc[pnr["PLAYER_ID"] == player_id]
    row = match.iloc[0] if not match.empty else None

    return pd.DataFrame(
        [
            {
                "Role": label,
                "Points": pd.to_numeric(row[f"{prefix}_PTS"], errors="coerce") if row is not None else None,
                "Possessions": pd.to_numeric(row[f"{prefix}_POSS"], errors="coerce") if row is not None else None,
                PERCENTILE_COL: row[f"{prefix}_PTS_PCTILE"] if row is not None else None,
            }
            for prefix, label, _play_type in PNR_ROLES
        ]
    )


def pnr_leaderboard(pnr: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Every player who logs possessions in one pick and roll role."""
    board = pnr.loc[pnr[f"{prefix}_POSS"].notna()].sort_values(f"{prefix}_PTS", ascending=False)

    return pd.DataFrame(
        {
            "Player": board["PLAYER_NAME"].values,
            "Team": (
                board["TEAM_ABBREVIATION"].values
                if "TEAM_ABBREVIATION" in board.columns
                else "—"
            ),
            "Points": board[f"{prefix}_PTS"].values,
            "Possessions": board[f"{prefix}_POSS"].values,
            PERCENTILE_COL: board[f"{prefix}_PTS_PCTILE"].values,
        }
    )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="NBA Scouting", layout="wide")

shots = load_shots()
splits = load_splits()
touches = load_touches()
pnr = load_pick_and_roll()

team_col, player_col, _spacer = st.columns([1, 1, 2])

with team_col:
    team = st.selectbox("Team", sorted(splits["TEAM_NAME"].dropna().unique()))

roster = splits.loc[splits["TEAM_NAME"] == team].sort_values("PLAYER_NAME")
with player_col:
    player_name = st.selectbox("Player", roster["PLAYER_NAME"].tolist())

player_id = int(roster.loc[roster["PLAYER_NAME"] == player_name, "PLAYER_ID"].iloc[0])

scouting_tab, touches_tab = st.tabs(["Scouting", "Touches & Pick and Roll"])

with scouting_tab:
    player_shots = shots.loc[shots["PLAYER_ID"] == player_id]

    # Shot type options come straight from the dummy variable categories, in
    # the same priority order, showing only the types this player actually has.
    counts = player_shots[SHOT_TYPE_GROUP_COL].value_counts()
    option_to_key = {
        f"{SHOT_TYPE_LABELS[key]} ({counts[key]})": key
        for key in SHOT_TYPE_LABELS
        if key in counts.index
    }
    options = list(option_to_key)

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
        st.dataframe(
            style_table(shooting_profile(splits, player_id), {"Value": "{:.1%}", "Attempts": "{:,.0f}"}),
            hide_index=True,
        )
        st.caption(
            "Season totals across all teams. Percentiles rank the player against "
            "the league on each metric, among players with enough attempts to qualify."
        )
        if load_box_stats() is None:
            st.warning(
                "True shooting % and free throw % need season box score totals. "
                f"Run `pull_player_box_stats()` from scraper_functions.py and save the "
                f"result to `data/{BOX_STATS_PATH.name}`."
            )

with touches_tab:
    st.subheader("Touch locations")

    if (touches["PLAYER_ID"] == player_id).any():
        touch_row = touches.loc[touches["PLAYER_ID"] == player_id].iloc[0]
        area_col, touch_table_col = st.columns([3, 2])

        with area_col:
            shares = {key: touch_row[share_col] for share_col, _label, _count, key in TOUCH_AREAS}
            colors = {
                key: percentile_color(touch_row[f"{share_col}_PCTILE"])
                for share_col, _label, _count, key in TOUCH_AREAS
            }
            labels = {key: label for _share, label, _count, key in TOUCH_AREAS}

            total_touches = int(touch_row["TOUCHES"])
            # Wider than tall: the touch areas plot crops the empty top of the court.
            fig, ax = plt.subplots(figsize=(7, 5.2))
            plot_touch_areas(
                shares,
                colors=colors,
                labels=labels,
                ax=ax,
                title=f"{player_name} — {SEASON} ({total_touches:,} touches)",
            )
            st.pyplot(fig)
            plt.close(fig)

        with touch_table_col:
            st.dataframe(
                style_table(
                    touch_profile(touches, player_id),
                    {"Share of touches": "{:.1%}", "Touches": "{:,.0f}"},
                ),
                hide_index=True,
            )
            st.caption(
                "Share of the player's total touches taken in each area, shaded by "
                "league percentile. Areas are the NBA's tracking definitions, so the "
                "elbow overlaps the top of the paint."
            )
    else:
        st.info(f"No touch tracking data for {player_name}.")

    st.divider()
    st.subheader("Pick and roll")

    if pnr is None:
        st.warning(
            "Pick and roll scoring needs the Synergy play type pull. Run "
            "`pull_pick_and_roll()` from scraper_functions.py and save the combined "
            f"result to `data/{PNR_PATH.name}`."
        )
    else:
        st.dataframe(
            style_table(pnr_profile(pnr, player_id), {"Points": "{:,.0f}", "Possessions": "{:,.0f}"}),
            hide_index=True,
        )
        st.caption(
            "Points scored out of the pick and roll in each role. Percentiles rank the "
            "player only against others who log possessions in that role, so roll men "
            "(bigs) and ball handlers (guards) are each ranked against their own group."
        )

        board_cols = st.columns(2)
        for column, (prefix, label, _play_type) in zip(board_cols, PNR_ROLES):
            with column:
                st.markdown(f"**{label} — all players**")
                st.dataframe(
                    style_table(
                        pnr_leaderboard(pnr, prefix), {"Points": "{:,.0f}", "Possessions": "{:,.0f}"}
                    ),
                    hide_index=True,
                    height=360,
                )
