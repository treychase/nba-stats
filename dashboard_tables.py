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
    "IMPACT_METRICS",
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
    "impact_profile",
    "star_tier_table",
    "star_leaderboard",
]

PERCENTILE_COL = "League %ile"


# ---------------------------------------------------------------------------
# Percentile shading
# ---------------------------------------------------------------------------

# Blue for below average, red for above, near-white through the middle - the
# same scale the shot chart already colours field goal percentage on, so a
# percentile and a hot spot read the same way across the whole app.
#
# Still deliberately not red-to-green: that pairing is the one diverging scale
# red-green colourblind readers cannot split, and a percentile column is
# useless if half its range reads the same as the other half. Red against blue
# separates on lightness as well as hue, so it survives both common forms of
# colour blindness and a black and white printout.
PERCENTILE_CMAP = "coolwarm"


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


# ---------------------------------------------------------------------------
# Impact: box plus/minus and win shares
# ---------------------------------------------------------------------------

IMPACT_METRICS = [
    # (column, label, format, the units the number is in)
    ("BPM", "Box +/-", "{:+.1f}", "points per 100 possessions, over average"),
    ("OBPM", "Offensive box +/-", "{:+.1f}", "the offensive half of it"),
    ("DBPM", "Defensive box +/-", "{:+.1f}", "the defensive half"),
    ("WS48", "Win shares / 48", "{:.3f}", "an average player earns 0.100"),
    ("WS", "Win shares", "{:.1f}", "wins, on the season so far"),
]


def impact_profile(impact, player_id: int) -> pd.DataFrame:
    """One row per impact metric: value and league percentile.

    ``impact`` is an :class:`advanced_metrics.ImpactModel`. A player with no
    reconstructed box score - anyone the shot chart does not cover - comes back
    as a full table of dashes rather than an empty one, so the panel keeps its
    shape instead of disappearing.
    """
    row = impact.of(player_id) if impact is not None else None

    def value_of(column):
        if row is None or column not in row.index:
            return None
        return pd.to_numeric(row[column], errors="coerce")

    return pd.DataFrame(
        [
            {
                "Metric": label,
                "Value": value_of(column),
                "What it means": meaning,
                PERCENTILE_COL: value_of(f"{column}_PCTILE"),
            }
            for column, label, _spec, meaning in IMPACT_METRICS
        ]
    )


def impact_formats() -> dict:
    """Number formats for :func:`impact_profile`, which mixes two scales.

    Box plus/minus wants a sign and one decimal, win shares per 48 wants three
    and no sign, so the column is formatted row by row before it is styled -
    the same reason :func:`style_table` formats to strings up front.
    """
    return {column: spec for column, _label, spec, _meaning in IMPACT_METRICS}


def style_impact(table: pd.DataFrame):
    """Format and shade the impact table, one number format per row."""
    specs = [spec for _column, _label, spec, _meaning in IMPACT_METRICS]
    display = table.copy()
    display["Value"] = [
        spec.format(value) if pd.notna(value) else "—"
        for spec, value in zip(specs, pd.to_numeric(table["Value"], errors="coerce"))
    ]
    percentiles = pd.to_numeric(table[PERCENTILE_COL], errors="coerce")
    display[PERCENTILE_COL] = [
        f"{value:.0f}" if pd.notna(value) else "—" for value in percentiles
    ]
    styles = [percentile_css(percentile) for percentile in percentiles]
    return display.style.apply(lambda _column: styles, subset=[PERCENTILE_COL])


def star_tier_table(tiers) -> pd.DataFrame:
    """Every tier, strongest first, with what defines it and who leads it."""
    if tiers is None:
        return pd.DataFrame()
    summary = tiers.summary()
    return summary.rename(columns={
        "TIER": "Tier", "PLAYERS": "Players", "BPM": "Box +/-",
        "WS48": "WS/48", "MPG": "Minutes / game", "PTS_PG": "Points / game",
        "DEFINES": "What defines it", "LEADERS": "Most win shares",
    })


def star_leaderboard(tiers, impact, n: int | None = None) -> pd.DataFrame:
    """The star tier, best box plus/minus first."""
    if tiers is None:
        return pd.DataFrame()
    board = tiers.stars
    if n is not None:
        board = board.head(n)
    out = pd.DataFrame({
        "Player": board["PLAYER_NAME"].values,
        "Team": (board["TEAM_ABBREVIATION"].values
                 if "TEAM_ABBREVIATION" in board.columns else "—"),
        "Box +/-": board["BPM"].values,
        "WS/48": board["WS48"].values,
        "Win shares": board["WS"].values,
        "Points / game": board["PTS_PG"].values,
        "Minutes / game": board["MPG"].values,
    })
    return out
