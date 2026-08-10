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
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
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


if __name__ == "__main__":
    # Quick smoke test
    sample = pd.DataFrame({
        "team": ["LAL", "BOS", "LAL", "GSW", None, "BOS", "BOS"],
    })
    print(show_unique_values(sample, "team"))