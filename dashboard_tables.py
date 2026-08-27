"""The scouting dashboard's tables, kept out of the Streamlit script.

Everything here is a pure pandas function: give it a league-wide frame and a
player id, get back the small table the app renders. Keeping them here rather
than inside ``dashboard.py`` means they can be imported and tested without a
Streamlit runtime, which is the only way the awkward cases - a player with no
touch tracking, a metric with too few attempts to qualify, a play type the
player has never run - get covered at all.
"""

from __future__ import annotations

import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import to_hex

from processing_functions import PNR_ROLES, SHOOTING_METRICS, TOUCH_AREAS

__all__ = [
    "PERCENTILE_COL",
    "PERCENTILE_CMAP",
    "percentile_color",
    "percentile_css",
    "style_table",
    "player_summary",
    "shooting_profile",
    "touch_profile",
    "pnr_profile",
    "pnr_leaderboard",
]

PERCENTILE_COL = "League %ile"


# ---------------------------------------------------------------------------
# Percentile shading
# ---------------------------------------------------------------------------

# Orange for below average, purple for above, near-white through the middle.
# Deliberately not red-to-green: that pairing is the one diverging scale
# red-green colorblind readers cannot split, and a percentile column is
# useless if half its range reads the same as the other half.
PERCENTILE_CMAP = "PuOr"


def percentile_color(percentile) -> str | None:
    """Diverging fill for a 0-100 percentile, or None if unranked."""
    if percentile is None or pd.isna(percentile):
        return None
    value = float(percentile)
    value = min(100.0, max(0.0, value))
    return to_hex(colormaps[PERCENTILE_CMAP](value / 100))


def _text_color(background: str) -> str:
    """Dark or light text, whichever stays legible on ``background``."""
    r, g, b = (int(background[i:i + 2], 16) / 255 for i in (1, 3, 5))
    # Rec. 709 luminance; the tails of the colormap are dark enough that fixed
    # dark text would disappear into them.
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#262730" if luminance > 0.5 else "#ffffff"


def percentile_css(percentile) -> str:
    """Cell styling for a 0-100 percentile, blank if unranked."""
    background = percentile_color(percentile)
    if background is None:
        return ""
    return f"background-color: {background}; color: {_text_color(background)}"


def style_table(table: pd.DataFrame, formats: dict):
    """Format a table's numbers and shade its percentile column.

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

def player_summary(touches: pd.DataFrame, player_id: int) -> dict:
    """Games, minutes, points and touches for the header card.

    Returns an empty dict when the player has no tracking row, which is the
    signal for the caller to show the card without the counting stats rather
    than to fail.
    """
    match = touches.loc[touches["PLAYER_ID"] == player_id]
    if match.empty:
        return {}
    row = match.iloc[0]
    fields = [("GP", "Games"), ("MIN", "Minutes"), ("POINTS", "Points"),
              ("TOUCHES", "Touches")]
    out = {}
    for column, label in fields:
        if column in row.index:
            value = pd.to_numeric(row[column], errors="coerce")
            if pd.notna(value):
                out[label] = float(value)
    return out


def shooting_profile(splits: pd.DataFrame, player_id: int) -> pd.DataFrame:
    """One row per shooting metric: value, attempts, percentile."""
    match = splits.loc[splits["PLAYER_ID"] == player_id]
    row = match.iloc[0] if not match.empty else None

    def value_of(column):
        if row is None or column not in row.index:
            return None
        return pd.to_numeric(row[column], errors="coerce")

    return pd.DataFrame(
        [
            {
                "Metric": label,
                "Value": value_of(value_col),
                "Attempts": value_of(attempts_col),
                PERCENTILE_COL: value_of(f"{value_col}_PCTILE"),
            }
            for value_col, label, attempts_col, _min_attempts in SHOOTING_METRICS
        ]
    )


def touch_profile(touches: pd.DataFrame, player_id: int) -> pd.DataFrame:
    """One row per court area: share of touches, touch count, percentile."""
    match = touches.loc[touches["PLAYER_ID"] == player_id]
    row = match.iloc[0] if not match.empty else None

    def value_of(column):
        if row is None or column not in row.index:
            return None
        return pd.to_numeric(row[column], errors="coerce")

    return pd.DataFrame(
        [
            {
                "Area": label,
                "Share of touches": value_of(share_col),
                "Touches": value_of(count_col),
                PERCENTILE_COL: value_of(f"{share_col}_PCTILE"),
            }
            for share_col, label, count_col, _area_key in TOUCH_AREAS
        ]
    )


def pnr_profile(pnr: pd.DataFrame, player_id: int) -> pd.DataFrame:
    """One row per pick and roll role: points, possessions, percentile."""
    match = pnr.loc[pnr["PLAYER_ID"] == player_id]
    row = match.iloc[0] if not match.empty else None

    def value_of(column):
        if row is None or column not in row.index:
            return None
        return pd.to_numeric(row[column], errors="coerce")

    return pd.DataFrame(
        [
            {
                "Role": label,
                "Points": value_of(f"{prefix}_PTS"),
                "Possessions": value_of(f"{prefix}_POSS"),
                PERCENTILE_COL: value_of(f"{prefix}_PTS_PCTILE"),
            }
            for prefix, label, _play_type in PNR_ROLES
        ]
    )


def pnr_leaderboard(pnr: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Every player who logs possessions in one pick and roll role."""
    board = pnr.loc[pnr[f"{prefix}_POSS"].notna()].sort_values(
        f"{prefix}_PTS", ascending=False
    )

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
