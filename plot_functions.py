"""
NBA Court Plotting
--------------------
Draws a regulation NBA half-court using matplotlib, with dimensions
matching the coordinate system used by nba_api's shotchartdetail
endpoint (LOC_X / LOC_Y are in units of 1/10 foot, origin at the
basket, court is 50 ft wide x 47 ft to half court). Also includes a
general-purpose column histogram plotter carried over from another
project.

Requires: pip install matplotlib pandas
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle, Rectangle, Arc


def plot_histogram(data, bins=30):
    """Histogram per column, with the mean marked and the tails called out.

    Accepts a Series or a DataFrame and lays multiple columns out in a grid.
    Bars whose centre falls more than two standard deviations from the mean are
    filled orange, so outliers are visible without reading the axis.
    """
    if isinstance(data, pd.Series):
        data = data.to_frame()

    cols = list(data.columns)
    ncols = min(3, len(cols))
    nrows = -(-len(cols) // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = [axes] if len(cols) == 1 else axes.flatten()

    for ax, col in zip(axes, cols):
        values = data[col].dropna()
        mean = values.mean()
        sd = values.std()
        label = col.replace("_", " ").title()

        counts, bin_edges, patches = ax.hist(values, bins=bins, color="lightblue", edgecolor="black")

        lower, upper = mean - 2 * sd, mean + 2 * sd
        for patch, left_edge, right_edge in zip(patches, bin_edges[:-1], bin_edges[1:]):
            bin_center = (left_edge + right_edge) / 2
            if bin_center < lower or bin_center > upper:
                patch.set_facecolor("orange")

        ax.axvline(mean, color="red", linestyle="--")
        ax.text(
            0.97, 0.97, f"mean = {mean:.2f}\nsd = {sd:.2f}",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"),
        )
        ax.set_xlabel(label)
        ax.set_ylabel("Frequency")
        ax.set_title(f"Distribution of {label}")
        ax.grid(False)

    for ax in axes[len(cols):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def draw_court(ax=None, color="black", lw=2, outer_lines=True):
    """
    Draw an NBA half-court onto a matplotlib Axes.

    Coordinate system matches shotchartdetail: origin (0, 0) is at the
    basket, x ranges roughly -250 to 250 (units of 1/10 ft, so -25ft to
    25ft), y ranges from -47.5 (baseline) up to 422.5 (half court).

    Parameters
    ----------
    ax : matplotlib Axes, optional
        Axes to draw on. Creates a new figure/axes if not provided.
    color : str
        Line color for all court markings.
    lw : float
        Line width for court markings.
    outer_lines : bool
        Whether to draw the outer court boundary and half-court line.

    Returns
    -------
    ax : matplotlib Axes
        The axes with the court drawn on it.
    """
    if ax is None:
        ax = plt.gca()

    # Hoop: 18 in diameter -> radius 7.5 in = 0.75 ft -> 7.5 units
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

    # Backboard: 6 ft wide, mounted 4 ft from baseline (in these units: -40 to 40 on x, at y=-7.5)
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)

    # The paint
    # Outer box: 16 ft wide, 19 ft from baseline to free throw line
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color, fill=False)
    # Inner box: 12 ft wide
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color, fill=False)

    # Free throw top arc (dashed for the part behind the paint is handled by drawing full arcs)
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180, linewidth=lw, color=color, fill=False)
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0, linewidth=lw, color=color, linestyle="dashed")

    # Restricted area arc: 4 ft radius from basket center
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw, color=color)

    # Three-point line
    # Corner three: straight lines 22 ft from basket, up to the point where the arc takes over
    corner_three_left = Rectangle((-220, -47.5), 0, 140, linewidth=lw, color=color)
    corner_three_right = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    # Arc: 23.75 ft radius
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw, color=color)

    # Center court
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0, linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0, linewidth=lw, color=color)

    court_elements = [
        hoop,
        backboard,
        outer_box,
        inner_box,
        top_free_throw,
        bottom_free_throw,
        restricted,
        corner_three_left,
        corner_three_right,
        three_arc,
        center_outer_arc,
        center_inner_arc,
    ]

    if outer_lines:
        outer_lines_box = Rectangle((-250, -47.5), 500, 470, linewidth=lw, color=color, fill=False)
        court_elements.append(outer_lines_box)

    for element in court_elements:
        ax.add_patch(element)

    ax.set_xlim(-260, 260)
    ax.set_ylim(-60, 430)
    ax.set_aspect("equal")
    ax.axis("off")

    return ax


def plot_shot_hexbin(
    df,
    ax=None,
    x_col="LOC_X",
    y_col="LOC_Y",
    gridsize=30,
    cmap="YlOrRd",
    mincnt=1,
    colorbar=True,
    title=None,
    player_id=None,
    headshot_corner="upper right",
):
    """
    Draw a court and overlay a hexbin density plot of shot locations.

    Parameters
    ----------
    df : pandas.DataFrame
        Shot data containing x/y location columns (e.g. output of
        shotchartdetail, which uses LOC_X / LOC_Y in 1/10 ft units).
    ax : matplotlib Axes, optional
        Axes to draw on. Creates a new figure/axes if not provided.
    x_col, y_col : str
        Column names for shot x/y coordinates.
    gridsize : int
        Number of hexagons across the x-axis; controls bin resolution.
    cmap : str
        Matplotlib colormap for hexbin density/frequency.
    mincnt : int
        Minimum count for a hexagon to be shown (filters out empty bins).
    colorbar : bool
        Whether to attach a colorbar for shot count.
    title : str, optional
        Title to place above the plot.
    player_id : int, optional
        If provided, fetches and insets that player's headshot from the
        NBA CDN in a corner of the chart.
    headshot_corner : str
        Corner to place the headshot in, if player_id is given. One of
        "upper left", "upper right", "lower left", "lower right".

    Returns
    -------
    ax : matplotlib Axes
        The axes with the court and hexbin overlay drawn on it.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7.5))

    hb = ax.hexbin(
        df[x_col],
        df[y_col],
        gridsize=gridsize,
        cmap=cmap,
        mincnt=mincnt,
        extent=(-250, 250, -47.5, 422.5),
    )

    draw_court(ax, color="black", lw=1.5)

    if colorbar:
        fig = ax.get_figure()
        fig.colorbar(hb, ax=ax, label="Shot count", fraction=0.035, pad=0.02)

    if title:
        ax.set_title(title, fontsize=13)

    if player_id is not None:
        add_player_headshot(ax, player_id, corner=headshot_corner)

    return ax


# ---------------------------------------------------------------------------
# Player headshot inset
# ---------------------------------------------------------------------------

HEADSHOT_URL_TEMPLATE = "https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"

# Approximate inset positions in axes-fraction coordinates (0-1 range)
CORNER_POSITIONS = {
    "upper left": (0.02, 0.98),
    "upper right": (0.98, 0.98),
    "lower left": (0.02, 0.02),
    "lower right": (0.98, 0.02),
}


def add_player_headshot(ax, player_id, corner="upper right", zoom=0.35):
    """
    Fetch a player's headshot from the NBA CDN and inset it into a corner
    of the given axes (e.g. on top of a shot chart).

    Parameters
    ----------
    ax : matplotlib Axes
        Axes to add the headshot to (typically one already holding a
        court plot / hexbin).
    player_id : int
        NBA PLAYER_ID, matches the ID used in shotchartdetail output.
    corner : str
        One of "upper left", "upper right", "lower left", "lower right".
    zoom : float
        Scale factor for the headshot image size.

    Returns
    -------
    ax : matplotlib Axes
        The axes with the headshot inset added. Returns unchanged ax
        (with a printed warning) if the headshot could not be fetched.
    """
    import urllib.request
    import io
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    url = HEADSHOT_URL_TEMPLATE.format(player_id=player_id)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            img_bytes = response.read()
        img = plt.imread(io.BytesIO(img_bytes), format="png")
    except Exception as e:
        print(f"Could not fetch headshot for player_id={player_id}: {e}")
        return ax

    if corner not in CORNER_POSITIONS:
        raise ValueError(f"corner must be one of {list(CORNER_POSITIONS)}")
    x_frac, y_frac = CORNER_POSITIONS[corner]
    box_alignment = (
        0 if "left" in corner else 1,
        0 if "lower" in corner else 1,
    )

    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(
        imagebox,
        (x_frac, y_frac),
        xycoords="axes fraction",
        box_alignment=box_alignment,
        frameon=False,
        pad=0,
    )
    ax.add_artist(ab)

    return ax


def plot_player_season_shot_chart(
    df,
    player,
    season,
    player_col="PLAYER_NAME",
    season_col="SEASON",
    ax=None,
    gridsize=30,
    cmap="YlOrRd",
    mincnt=1,
    colorbar=True,
    show_headshot=True,
    headshot_corner="upper right",
    zoom=0.35,
):
    """
    Filter a shot chart DataFrame down to one player/season and plot the
    hexbin shot chart on a court, with an optional headshot inset.

    Parameters
    ----------
    df : pandas.DataFrame
        Full shot chart data, e.g. output of pull_shot_chart() from
        nba_data_pull.py. Must contain PLAYER_ID, LOC_X, LOC_Y, and the
        columns named by player_col / season_col.
    player : str or int
        Player name (matches player_col) or PLAYER_ID to filter on.
        If player_col values are strings, pass a string; if you'd rather
        filter by ID directly, pass an int and set player_col="PLAYER_ID".
    season : str
        Season to filter on, e.g. "2025-26". Must match values in
        season_col.
    player_col : str
        Column name to filter player on. Defaults to "PLAYER_NAME".
    season_col : str
        Column name to filter season on. Defaults to "SEASON".
    ax : matplotlib Axes, optional
        Axes to draw on. Creates a new figure/axes if not provided.
    gridsize, cmap, mincnt, colorbar : see plot_shot_hexbin.
    show_headshot : bool
        Whether to inset the player's headshot (requires a PLAYER_ID
        column in df to look up the ID).
    headshot_corner : str
        Corner for the headshot inset, if show_headshot is True.
    zoom : float
        Scale factor for the headshot image size.

    Returns
    -------
    ax : matplotlib Axes
        The axes with the filtered shot chart plotted.

    Raises
    ------
    ValueError
        If no rows match the given player/season filter.
    """
    filtered = df[(df[player_col] == player) & (df[season_col] == season)]

    if filtered.empty:
        raise ValueError(
            f"No shots found for {player_col}={player!r}, {season_col}={season!r}. "
            "Check that the values match what's in the DataFrame."
        )

    player_id = None
    if show_headshot and "PLAYER_ID" in filtered.columns:
        player_id = filtered["PLAYER_ID"].iloc[0]

    display_name = player if player_col == "PLAYER_NAME" else filtered["PLAYER_NAME"].iloc[0]
    title = f"{display_name} — {season} Shot Chart ({len(filtered)} shots)"

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7.5))

    plot_shot_hexbin(
        filtered,
        ax=ax,
        gridsize=gridsize,
        cmap=cmap,
        mincnt=mincnt,
        colorbar=colorbar,
        title=title,
        player_id=player_id if show_headshot else None,
        headshot_corner=headshot_corner,
    )

    return ax


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(8, 7.5))
    draw_court(ax)
    plt.savefig("court_test.png", dpi=150, bbox_inches="tight")
    print("Saved court_test.png")