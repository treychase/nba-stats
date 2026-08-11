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


# ---------------------------------------------------------------------------
# Touch locations (share of a player's touches by area of the court)
# ---------------------------------------------------------------------------

TOUCH_AREAS = [
    # (share column, label, touch count column, court area key used for plotting)
    ("PAINT_TOUCH_PCT", "Paint", "PAINT_TOUCHES", "paint"),
    ("POST_TOUCH_PCT", "Post", "POST_TOUCHES", "post"),
    ("ELBOW_TOUCH_PCT", "Elbow", "ELBOW_TOUCHES", "elbow"),
]

MIN_TOUCHES = 200


def build_touch_profile(possessions, min_touches=MIN_TOUCHES):
    """
    Build a per-player touch profile: what share of a player's touches
    happen in the paint, in the post and at the elbow, each with its
    league percentile.

    Parameters
    ----------
    possessions : pandas.DataFrame
        Possessions tracking data, e.g.
        data/tracking_possessions_2025-26.csv from pull_tracking_stats().
        Needs PLAYER_ID, PLAYER_NAME, TOUCHES, and the per-area touch
        count columns listed in TOUCH_AREAS.
    min_touches : int
        Total touches a player needs before they're ranked. Below this a
        handful of post-ups swings the share wildly, so those players
        still get their shares but no percentile.

    Returns
    -------
    pandas.DataFrame
        One row per player with PLAYER_ID, PLAYER_NAME, TOUCHES, the raw
        per-area counts, a <AREA>_TOUCH_PCT share (0-1) for each area,
        and a matching <AREA>_TOUCH_PCT_PCTILE column (0-100).
    """
    count_cols = [count_col for _share, _label, count_col, _key in TOUCH_AREAS]
    required = ["PLAYER_ID", "PLAYER_NAME", "TOUCHES"] + count_cols
    missing = [c for c in required if c not in possessions.columns]
    if missing:
        raise KeyError(f"Possessions tracking data is missing columns: {missing}")

    profile = possessions[required].copy()
    touches = profile["TOUCHES"].replace(0, pd.NA)
    qualified = profile["TOUCHES"] >= min_touches

    for share_col, _label, count_col, _key in TOUCH_AREAS:
        profile[share_col] = profile[count_col] / touches
        profile[f"{share_col}_PCTILE"] = (
            pd.to_numeric(profile[share_col], errors="coerce").where(qualified).rank(pct=True) * 100
        )

    return profile.sort_values("PLAYER_NAME").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pick and roll scoring (ball handlers and roll men)
# ---------------------------------------------------------------------------
# Synergy splits the pick and roll into two roles, which in practice sorts
# players by position: guards run it as the ball handler, bigs finish it as
# the roll man. Rather than joining a position list, each role is ranked
# against the players who actually log possessions in that role, so a big
# is percentiled against other roll men and a guard against other handlers.

PNR_ROLES = [
    # (column prefix, label, PLAY_TYPE value in the Synergy pull)
    ("HANDLER", "Ball handler", "PRBallHandler"),
    ("ROLL", "Roll man", "PRRollman"),
]

MIN_PNR_POSS = 25


def build_pick_and_roll_profile(pnr, min_poss=MIN_PNR_POSS):
    """
    Build a per-player pick and roll scoring profile: points scored as the
    ball handler and as the roll man, each with its league percentile
    among players who log real volume in that role.

    Parameters
    ----------
    pnr : pandas.DataFrame
        Combined Synergy play type data, e.g.
        data/nba_pick_and_roll_combined_2025-26.csv from
        pull_pick_and_roll(). Needs PLAYER_ID, PLAYER_NAME, PLAY_TYPE,
        TYPE_GROUPING, POSS and PTS. Only offensive rows are used.
    min_poss : int
        Possessions in a role before a player is ranked in it.

    Returns
    -------
    pandas.DataFrame
        One row per player with PLAYER_ID, PLAYER_NAME, TEAM_ABBREVIATION,
        <ROLE>_PTS / <ROLE>_POSS / <ROLE>_PTS_PCTILE per role, and
        PRIMARY_ROLE naming whichever role they run more often.
    """
    required = ["PLAYER_ID", "PLAYER_NAME", "PLAY_TYPE", "TYPE_GROUPING", "POSS", "PTS"]
    missing = [c for c in required if c not in pnr.columns]
    if missing:
        raise KeyError(f"Pick and roll data is missing columns: {missing}")

    offense = pnr[pnr["TYPE_GROUPING"].str.lower() == "offensive"]

    players = (
        offense[["PLAYER_ID", "PLAYER_NAME"]]
        .drop_duplicates(subset="PLAYER_ID")
        .reset_index(drop=True)
    )
    if "TEAM_ABBREVIATION" in offense.columns:
        teams = offense.drop_duplicates(subset="PLAYER_ID").set_index("PLAYER_ID")["TEAM_ABBREVIATION"]
        players["TEAM_ABBREVIATION"] = players["PLAYER_ID"].map(teams)

    profile = players
    for prefix, _label, play_type in PNR_ROLES:
        role = (
            offense[offense["PLAY_TYPE"] == play_type]
            .groupby("PLAYER_ID")[["POSS", "PTS"]]
            .sum()
            .rename(columns={"POSS": f"{prefix}_POSS", "PTS": f"{prefix}_PTS"})
        )
        profile = profile.merge(role, on="PLAYER_ID", how="left")

        qualified = profile[f"{prefix}_POSS"] >= min_poss
        profile[f"{prefix}_PTS_PCTILE"] = (
            pd.to_numeric(profile[f"{prefix}_PTS"], errors="coerce").where(qualified).rank(pct=True) * 100
        )

    handler_poss = profile["HANDLER_POSS"].fillna(0)
    roll_poss = profile["ROLL_POSS"].fillna(0)
    profile["PRIMARY_ROLE"] = pd.Series(
        ["Ball handler" if h >= r else "Roll man" for h, r in zip(handler_poss, roll_poss)],
        index=profile.index,
    ).where(handler_poss + roll_poss > 0)

    return profile.sort_values("PLAYER_NAME").reset_index(drop=True)


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