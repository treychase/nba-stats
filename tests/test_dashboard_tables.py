import pandas as pd
import pytest

from dashboard_tables import (
    IMPACT_METRICS,
    PERCENTILE_COL,
    impact_profile,
    star_leaderboard,
    star_tier_table,
    style_impact,
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
    # Blue at the bottom, red at the top: the low end must carry more blue than
    # red and the high end the reverse. Not red-and-green, so the scale still
    # survives the common forms of colour blindness.
    assert int(low[5:7], 16) > int(low[1:3], 16)
    assert int(high[1:3], 16) > int(high[5:7], 16)
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


# ---------------------------------------------------------------------------
# Impact and star tiers
# ---------------------------------------------------------------------------

class _Impact:
    """The little of ImpactModel these tables touch."""

    def __init__(self, row):
        self._row = row

    def of(self, player_id):
        return self._row if player_id == 1 else None


class _Tiers:
    def __init__(self, summary, stars):
        self._summary, self.stars = summary, stars

    def summary(self):
        return self._summary


@pytest.fixture
def impact():
    row = {"PLAYER_ID": 1}
    for column, _label, _spec, _meaning in IMPACT_METRICS:
        row[column] = 2.5
        row[f"{column}_PCTILE"] = 88.0
    return _Impact(pd.Series(row))


def test_impact_profile_has_one_row_per_metric(impact):
    table = impact_profile(impact, 1)
    assert list(table["Metric"]) == [label for _c, label, _s, _m in IMPACT_METRICS]
    assert table[PERCENTILE_COL].notna().all()


def test_a_player_with_no_box_score_still_gets_the_table(impact):
    table = impact_profile(impact, 999)
    assert len(table) == len(IMPACT_METRICS)
    assert table["Value"].isna().all()
    assert table[PERCENTILE_COL].isna().all()


def test_no_impact_model_at_all_still_gets_the_table():
    table = impact_profile(None, 1)
    assert len(table) == len(IMPACT_METRICS)
    assert table["Value"].isna().all()


def test_style_impact_formats_each_row_on_its_own_scale(impact):
    rendered = style_impact(impact_profile(impact, 1)).data
    assert rendered["Value"].iloc[0] == "+2.5"      # box plus/minus, signed
    assert rendered["Value"].iloc[3] == "2.500"     # win shares per 48
    assert style_impact(impact_profile(impact, 999)).data["Value"].eq("—").all()


def test_star_tables_are_empty_without_a_clustering():
    assert star_tier_table(None).empty
    assert star_leaderboard(None, None).empty


def test_star_leaderboard_renames_into_display_columns():
    stars = pd.DataFrame({
        "PLAYER_NAME": ["A Player", "B Player"],
        "TEAM_ABBREVIATION": ["LAL", "BOS"],
        "BPM": [8.0, 6.0], "WS48": [0.3, 0.25], "WS": [12.0, 9.0],
        "PTS_PG": [28.0, 24.0], "MPG": [35.0, 33.0],
    })
    board = star_leaderboard(_Tiers(pd.DataFrame(), stars), None)
    assert list(board["Player"]) == ["A Player", "B Player"]
    assert board["Box +/-"].iloc[0] == 8.0
    assert len(star_leaderboard(_Tiers(pd.DataFrame(), stars), None, n=1)) == 1
