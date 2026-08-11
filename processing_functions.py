"""
Data Processing Utilities
---------------------------
General-purpose helper functions for inspecting and cleaning
DataFrames across projects.

Requires: pip install pandas
"""

import pandas as pd


def show_unique_values(df, column=None, include_counts=True, dropna=False, sort_by="count"):
    """
    Show all unique values in a categorical column, optionally with
    counts, sorted by frequency or alphabetically.

    Useful for spotting inconsistent categories before modeling or
    merging (e.g. "LAL" vs "Los Angeles Lakers", stray whitespace,
    unexpected NaNs).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the column(s) to inspect.
    column : str, optional
        Name of the column to get unique values for. If not provided,
        runs across every categorical column in df (dtype "object" or
        "category") and returns a dict keyed by column name.
    include_counts : bool
        If True, returns a DataFrame with value counts. If False,
        returns just a list of unique values.
    dropna : bool
        Whether to exclude NaN/None from the results. Defaults to
        False so missing values are visible rather than silently
        dropped.
    sort_by : str
        "count" sorts by frequency (descending). "value" sorts
        alphabetically/numerically. Only applies when include_counts
        is True.

    Returns
    -------
    pandas.DataFrame or list or dict
        If column is given: a DataFrame with columns [column, "count"]
        (or a plain list if include_counts is False).
        If column is None: a dict mapping each categorical column name
        to its own result (DataFrame or list, per include_counts).

    Raises
    ------
    KeyError
        If column is given but not present in df.
    """
    if column is None:
        categorical_cols = df.select_dtypes(include=["object", "category", "str"]).columns.tolist()
        return {
            col: show_unique_values(df, col, include_counts=include_counts, dropna=dropna, sort_by=sort_by)
            for col in categorical_cols
        }

    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame. Available columns: {list(df.columns)}")

    if not include_counts:
        values = df[column].dropna().unique() if dropna else df[column].unique()
        return list(values)

    counts = df[column].value_counts(dropna=dropna)
    result = counts.reset_index()
    result.columns = [column, "count"]

    if sort_by == "value":
        result = result.sort_values(by=column).reset_index(drop=True)
    else:
        result = result.sort_values(by="count", ascending=False).reset_index(drop=True)

    return result


# ---------------------------------------------------------------------------
# Shot type classification (e.g. from shotchartdetail's ACTION_TYPE)
# ---------------------------------------------------------------------------
# Priority order matters: many ACTION_TYPE strings contain several keywords
# at once (e.g. "Turnaround Fadeaway Bank Jump Shot" contains "turnaround",
# "fadeaway", "bank", AND "jump shot"). Each value is checked against the
# categories below in order and assigned to the first match, so the most
# specific/descriptive categories are listed first and "catch_and_shoot"
# (a bare match on "jump shot") is listed last — it's meant as a fallback
# for plain "Jump Shot" entries, not a catch-all for every shot that
# happens to end in "Jump Shot".

SHOT_TYPE_KEYWORDS = [
    ("floater", ["floating"]),
    ("step_back", ["step back"]),
    ("pull_up", ["pullup", "pull-up", "pull up"]),
    ("driving", ["driving"]),
    ("turnaround", ["turnaround"]),
    ("fadeaway", ["fadeaway"]),
    ("hook", ["hook"]),
    ("bank", ["bank"]),
    ("alley_oop", ["alley oop"]),
    ("putback", ["putback"]),
    ("tip", ["tip"]),
    ("reverse", ["reverse"]),
    ("cutting", ["cutting"]),
    ("running", ["running"]),
    ("dunk", ["dunk"]),
    ("layup", ["layup"]),
    ("catch_and_shoot", ["jump shot"]),  # fallback: plain "Jump Shot" with no other qualifier above
]


def add_shot_type_column(df, action_col="ACTION_TYPE", new_col="SHOT_TYPE"):
    """
    Add a new column classifying each shot's ACTION_TYPE into a simplified
    shot type: "floater", "step_back", "pull_up", "driving", "turnaround",
    "fadeaway", "hook", "bank", "alley_oop", "putback", "tip", "reverse",
    "cutting", "running", "dunk", "layup", "catch_and_shoot" (plain jump
    shots), or "other" for anything unmatched.

    Matching is keyword-based and case-insensitive, checked in priority
    order so a value like "Driving Floating Jump Shot" is classified as
    "floater" rather than "driving" (floater checked first). Adjust
    SHOT_TYPE_KEYWORDS at the top of this file to change priority or
    add categories.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the action type column.
    action_col : str
        Name of the column with shot description strings (e.g.
        shotchartdetail's ACTION_TYPE).
    new_col : str
        Name for the new classified column.

    Returns
    -------
    pandas.DataFrame
        A copy of df with new_col added.

    Raises
    ------
    KeyError
        If action_col is not present in df.
    """
    if action_col not in df.columns:
        raise KeyError(f"Column '{action_col}' not found in DataFrame. Available columns: {list(df.columns)}")

    def _classify(action):
        if pd.isna(action):
            return "other"
        text = str(action).lower()
        for label, keywords in SHOT_TYPE_KEYWORDS:
            if any(kw in text for kw in keywords):
                return label
        return "other"

    df = df.copy()
    df[new_col] = df[action_col].apply(_classify)
    return df


# Display names for the categories produced by add_shot_type_column /
# add_shot_type_dummies, in the same priority order as SHOT_TYPE_KEYWORDS.
# Used by the dashboard so shot type filters read as English rather than
# as raw dummy column suffixes.
SHOT_TYPE_LABELS = {
    "floater": "Floater",
    "step_back": "Step Back",
    "pull_up": "Pull Up",
    "driving": "Driving",
    "turnaround": "Turnaround",
    "fadeaway": "Fadeaway",
    "hook": "Hook",
    "bank": "Bank",
    "alley_oop": "Alley Oop",
    "putback": "Putback",
    "tip": "Tip",
    "reverse": "Reverse",
    "cutting": "Cutting",
    "running": "Running",
    "dunk": "Dunk",
    "layup": "Layup",
    "catch_and_shoot": "Catch & Shoot",
    "other": "Other",
}


def add_shot_type_dummies(df, action_col="ACTION_TYPE", shot_type_col="SHOT_TYPE", prefix="SHOT_TYPE"):
    """
    Add the SHOT_TYPE classification column (via add_shot_type_column)
    and then one-hot encode it into dummy columns, including a separate
    dummy for "floater" as its own category.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the action type column.
    action_col : str
        Name of the source column with shot description strings.
    shot_type_col : str
        Name for the intermediate classified column (added before
        dummies are created).
    prefix : str
        Prefix for the resulting dummy columns, e.g. "SHOT_TYPE_driving".

    Returns
    -------
    pandas.DataFrame
        A copy of df with shot_type_col and the dummy columns added.
    """
    df = add_shot_type_column(df, action_col=action_col, new_col=shot_type_col)
    dummies = pd.get_dummies(df[shot_type_col], prefix=prefix)
    return pd.concat([df, dummies], axis=1)


# ---------------------------------------------------------------------------
# Shooting splits (true shooting %, 3P%, mid-range %, FT%) with percentiles
# ---------------------------------------------------------------------------
# Each metric carries its own attempt minimum. A player is only ranked on a
# metric once they clear that minimum, so a 2-for-3 mid-range season doesn't
# land in the 99th percentile and push everyone else down.

SHOOTING_METRICS = [
    # (value column, label, attempts column, minimum attempts to be ranked)
    ("TS_PCT", "True Shooting %", "FGA", 100),
    ("FG3_PCT", "3-Point %", "FG3A", 20),
    ("MID_PCT", "Mid-Range %", "MID_FGA", 20),
    ("FT_PCT", "Free Throw %", "FTA", 20),
]


def build_shooting_splits(shots, box=None, shot_value_col="SHOT_TYPE", zone_col="SHOT_ZONE_BASIC"):
    """
    Build a per-player shooting profile: true shooting %, 3-point %,
    mid-range % and free throw %, each with its league percentile.

    Field goal splits come from per-shot data (shotchartdetail). Free
    throws are not in that feed, so true shooting % and free throw %
    require the season box score totals from pull_player_box_stats() —
    without them those two columns come back as NaN and the rest of the
    profile still works.

    Parameters
    ----------
    shots : pandas.DataFrame
        Per-shot data, one row per field goal attempt. Needs PLAYER_ID,
        PLAYER_NAME, TEAM_NAME, GAME_DATE, SHOT_MADE_FLAG, plus the
        columns named by shot_value_col and zone_col.
    box : pandas.DataFrame, optional
        Season totals per player, e.g. data/nba_player_box_2025-26.csv
        from pull_player_box_stats(). Needs PLAYER_ID, PTS, FGA, FTM,
        FTA.
    shot_value_col : str
        Column holding shotchartdetail's "2PT Field Goal" / "3PT Field
        Goal" label. Note this is the raw SHOT_TYPE column, which
        add_shot_type_column overwrites by default — pass a different
        new_col there (or a different name here) if you classify shot
        types on the same frame.
    zone_col : str
        Column holding shotchartdetail's SHOT_ZONE_BASIC, used to pick
        out mid-range attempts.

    Returns
    -------
    pandas.DataFrame
        One row per player with PLAYER_ID, PLAYER_NAME, TEAM_NAME, the
        made/attempted counts behind each split, the four percentage
        columns from SHOOTING_METRICS, and a matching <METRIC>_PCTILE
        column (0-100) for each. Percentiles are NaN for players below
        that metric's attempt minimum.
    """
    is_three = shots[shot_value_col].astype(str).str.startswith("3")
    is_mid = shots[zone_col] == "Mid-Range"
    made = shots["SHOT_MADE_FLAG"]

    frame = shots.assign(
        _FG3A=is_three.astype(int),
        _FG3M=(is_three & made.astype(bool)).astype(int),
        _MID_FGA=is_mid.astype(int),
        _MID_FGM=(is_mid & made.astype(bool)).astype(int),
    )

    grouped = frame.groupby("PLAYER_ID")
    splits = grouped.agg(
        PLAYER_NAME=("PLAYER_NAME", "first"),
        FGA=("SHOT_MADE_FLAG", "size"),
        FGM=("SHOT_MADE_FLAG", "sum"),
        FG3A=("_FG3A", "sum"),
        FG3M=("_FG3M", "sum"),
        MID_FGA=("_MID_FGA", "sum"),
        MID_FGM=("_MID_FGM", "sum"),
    ).reset_index()

    # Team shown is the one the player took their most recent shot for, so
    # traded players scout under their current team rather than their first.
    latest = frame.sort_values("GAME_DATE").groupby("PLAYER_ID")["TEAM_NAME"].last()
    splits["TEAM_NAME"] = splits["PLAYER_ID"].map(latest)

    splits["FG3_PCT"] = splits["FG3M"] / splits["FG3A"].replace(0, pd.NA)
    splits["MID_PCT"] = splits["MID_FGM"] / splits["MID_FGA"].replace(0, pd.NA)

    if box is not None:
        box_cols = ["PLAYER_ID", "PTS", "FGA", "FTM", "FTA"]
        missing = [c for c in box_cols if c not in box.columns]
        if missing:
            raise KeyError(f"Box score totals are missing columns: {missing}")
        splits = splits.merge(
            box[box_cols].rename(columns={"FGA": "BOX_FGA"}), on="PLAYER_ID", how="left"
        )
        splits["FT_PCT"] = splits["FTM"] / splits["FTA"].replace(0, pd.NA)
        # Box score FGA is the authority for true shooting: it covers the
        # whole season, while the shot chart pull can lag by a game.
        true_shot_attempts = 2 * (splits["BOX_FGA"].fillna(splits["FGA"]) + 0.44 * splits["FTA"])
        splits["TS_PCT"] = splits["PTS"] / true_shot_attempts.replace(0, pd.NA)
    else:
        for col in ("FTM", "FTA", "PTS", "FT_PCT", "TS_PCT"):
            splits[col] = pd.NA

    for value_col, _label, attempts_col, min_attempts in SHOOTING_METRICS:
        values = pd.to_numeric(splits[value_col], errors="coerce")
        qualified = pd.to_numeric(splits[attempts_col], errors="coerce") >= min_attempts
        splits[f"{value_col}_PCTILE"] = values.where(qualified).rank(pct=True) * 100

    ordered = ["PLAYER_ID", "PLAYER_NAME", "TEAM_NAME", "FGA", "FGM", "FG3A", "FG3M",
               "MID_FGA", "MID_FGM", "FTM", "FTA", "PTS"]
    ordered += [c for c in splits.columns if c not in ordered]
    return splits[ordered].sort_values("PLAYER_NAME").reset_index(drop=True)


if __name__ == "__main__":
    # Quick smoke test
    sample = pd.DataFrame({
        "team": ["LAL", "BOS", "LAL", "GSW", None, "BOS", "BOS"],
        "position": ["PG", "SG", "PG", "C", "PF", "SG", "PG"],
        "points": [24, 18, 30, 12, 9, 22, 27],  # numeric, should be skipped when column=None
    })
    print("Single column:")
    print(show_unique_values(sample, "team"))
    print("\nAll categorical columns:")
    for col, result in show_unique_values(sample).items():
        print(f"\n-- {col} --")
        print(result)

    print("\n--- Shot type classification ---")
    shots = pd.DataFrame({
        "ACTION_TYPE": [
            "Jump Shot",
            "Pullup Jump shot",
            "Driving Layup Shot",
            "Driving Floating Jump Shot",
            "Step Back Jump shot",
            "Turnaround Hook Shot",
            "Dunk Shot",
        ]
    })
    result = add_shot_type_dummies(shots)
    print(result)