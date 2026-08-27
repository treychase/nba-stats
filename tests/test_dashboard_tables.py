import pandas as pd
import pytest

from dashboard_tables import (
    PERCENTILE_COL,
    percentile_color,
    percentile_css,
    player_summary,
    pnr_leaderboard,
    pnr_profile,
    shooting_profile,
    style_table,
    touch_profile,
)
from processing_functions import PNR_ROLES, SHOOTING_METRICS, TOUCH_AREAS


@pytest.fixture
def splits():
    row = {"PLAYER_ID": 1, "PLAYER_NAME": "A Player", "TEAM_NAME": "Team"}
    for value_col, _label, attempts_col, _min_attempts in SHOOTING_METRICS:
        row[value_col] = 0.5
        row[attempts_col] = 100
        row[f"{value_col}_PCTILE"] = 75.0
    return pd.DataFrame([row])


@pytest.fixture
def touches():
    row = {"PLAYER_ID": 1, "PLAYER_NAME": "A Player", "TOUCHES": 500,
           "GP": 40, "MIN": 1200.0, "POINTS": 640}
    for share_col, _label, count_col, _key in TOUCH_AREAS:
        row[share_col] = 0.2
        row[count_col] = 100
        row[f"{share_col}_PCTILE"] = 60.0
    return pd.DataFrame([row])


@pytest.fixture
def pnr():
    row = {"PLAYER_ID": 1, "PLAYER_NAME": "A Player", "TEAM_ABBREVIATION": "OKC"}
    for prefix, _label, _play_type in PNR_ROLES:
        row[f"{prefix}_PTS"] = 200.0
        row[f"{prefix}_POSS"] = 180.0
        row[f"{prefix}_PTS_PCTILE"] = 90.0
    return pd.DataFrame([row])


def test_percentile_colour_diverges_and_clamps():
    low, high = percentile_color(0), percentile_color(100)
    # Two hues, not red-and-green: the low end must carry more red than blue
    # and the high end the reverse, so the scale survives colorblindness.
    assert int(low[1:3], 16) > int(low[5:7], 16)
    assert int(high[5:7], 16) > int(high[1:3], 16)
    assert low != high
    assert percentile_color(-20) == low and percentile_color(150) == high
    assert percentile_color(None) is None
    assert percentile_color(float("nan")) is None
    assert percentile_css(float("nan")) == ""
    assert "background-color" in percentile_css(50)


def test_profiles_have_one_row_per_metric(splits, touches, pnr):
    assert len(shooting_profile(splits, 1)) == len(SHOOTING_METRICS)
    assert len(touch_profile(touches, 1)) == len(TOUCH_AREAS)
    assert len(pnr_profile(pnr, 1)) == len(PNR_ROLES)


def test_a_missing_player_yields_blanks_rather_than_an_error(splits, touches, pnr):
    """The awkward case: a player in one dataset and not another."""
    for table in (shooting_profile(splits, 999), touch_profile(touches, 999),
                  pnr_profile(pnr, 999)):
        assert len(table) > 0
        assert table[PERCENTILE_COL].isna().all()


def test_player_summary_reports_counting_stats_and_survives_absence(touches):
    summary = player_summary(touches, 1)
    assert summary["Games"] == 40 and summary["Points"] == 640
    assert player_summary(touches, 999) == {}


def test_style_table_renders_missing_values_as_a_dash(splits):
    table = shooting_profile(splits, 999)
    styled = style_table(table, {"Value": "{:.1%}", "Attempts": "{:,.0f}"})
    rendered = styled.data
    assert (rendered["Value"] == "—").all()
    assert (rendered[PERCENTILE_COL] == "—").all()


def test_leaderboard_sorts_by_points(pnr):
    other = pnr.iloc[0].copy()
    other["PLAYER_ID"] = 2
    other["PLAYER_NAME"] = "B Player"
    prefix = PNR_ROLES[0][0]
    other[f"{prefix}_PTS"] = 500.0
    board = pnr_leaderboard(pd.concat([pnr, other.to_frame().T], ignore_index=True), prefix)
    assert board["Player"].tolist() == ["B Player", "A Player"]
