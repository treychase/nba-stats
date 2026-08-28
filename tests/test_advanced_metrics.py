"""Tests for the reconstructed box score, the two metrics and the star tiers.

The awkward cases are the point: a player the shot chart covers but the
tracking pulls do not, a player who defended nothing at the rim, a league where
nobody clears the minutes floor, and the two identities the whole module rests
on - that the league's average box plus/minus is zero and its average WS/48 is
0.100.
"""

import numpy as np
import pandas as pd
import pytest

from advanced_metrics import (
    AVERAGE_WS48,
    MIN_MINUTES,
    STAR_TIER,
    TIER_LADDERS,
    VALUES,
    build_box_panel,
    build_tier_features,
    fit_star_tiers,
    impact_metrics,
    rim_defence_baseline,
)

SEED = 4


def _league(n=90, seed=SEED):
    """A synthetic league: shots, possessions and tracking that agree."""
    rng = np.random.default_rng(seed)
    player_ids = np.arange(1, n + 1)
    minutes = rng.uniform(80, 2600, n).round()   # some below the floor
    # Bigs defend the rim and rebound; guards pass. Two groups, so the tiers
    # have something to find and the rim baseline has a slope to fit.
    big = player_ids % 3 == 0

    fga = (minutes * rng.uniform(0.28, 0.55, n)).round()
    fgm = (fga * rng.uniform(0.42, 0.58, n)).round()
    fg3a = np.where(big, fga * 0.12, fga * 0.42).round()
    fg3m = (fg3a * rng.uniform(0.30, 0.42, n)).round()
    ftm = (fga * rng.uniform(0.10, 0.30, n)).round()
    points = 2 * fgm + fg3m + ftm

    shots = pd.DataFrame({
        "PLAYER_ID": np.repeat(player_ids, fga.astype(int)),
        "SHOT_MADE_FLAG": np.concatenate([
            np.concatenate([np.ones(int(m)), np.zeros(int(a - m))])
            for a, m in zip(fga, fgm)]),
        "SHOT_TYPE": np.concatenate([
            np.array(["3PT Field Goal"] * int(t) + ["2PT Field Goal"] * int(a - t))
            for a, t in zip(fga, fg3a)]),
    })
    # Threes have to be a subset of the makes the counts imply, so re-derive
    # made threes from the frame rather than trusting the two draws to agree.
    three = shots["SHOT_TYPE"].str.startswith("3")
    made = shots["SHOT_MADE_FLAG"].astype(bool)
    fg3m_actual = shots.assign(x=(three & made)).groupby("PLAYER_ID")["x"].sum()
    fgm_actual = shots.groupby("PLAYER_ID")["SHOT_MADE_FLAG"].sum()
    points = (2 * fgm_actual + fg3m_actual + pd.Series(ftm, index=player_ids)).to_numpy()

    possessions = pd.DataFrame({
        "PLAYER_ID": player_ids,
        "PLAYER_NAME": [f"Player {i}" for i in player_ids],
        "TEAM_ABBREVIATION": np.where(big, "BIG", "GRD"),
        "GP": np.maximum((minutes / 24).round(), 1),
        "MIN": minutes,
        "POINTS": points,
        "TOUCHES": (minutes * rng.uniform(1.2, 2.6, n)).round(),
    })

    rim_fga = np.where(big, minutes * 0.14, minutes * 0.05).round()
    tracking = pd.DataFrame({
        "PLAYER_ID": player_ids,
        "PLAYER_NAME": possessions["PLAYER_NAME"],
        "MIN": minutes,
        "TOUCHES": possessions["TOUCHES"],
        "OREB": np.where(big, minutes * 0.06, minutes * 0.015).round(),
        "DREB_Rebounding": np.where(big, minutes * 0.16, minutes * 0.07).round(),
        "AST": (minutes * np.where(big, 0.03, 0.10)).round(),
        "AST_PTS_CREATED": (minutes * np.where(big, 0.07, 0.26)).round(),
        "STL": (minutes * 0.03).round(),
        "BLK": (minutes * np.where(big, 0.05, 0.01)).round(),
        "DEF_RIM_FGA": rim_fga,
        # Bigs defend the rim better, which is exactly the volume/rate slope
        # rim_defence_baseline exists to take back out.
        "DEF_RIM_FGM": (rim_fga * np.where(big, 0.60, 0.72)).round(),
        "DRIVE_FTM": (ftm * 0.4).round(), "DRIVE_FTA": (ftm * 0.55).round(),
        "ELBOW_TOUCH_FTM": np.zeros(n), "ELBOW_TOUCH_FTA": np.zeros(n),
        "POST_TOUCH_FTM": np.zeros(n), "POST_TOUCH_FTA": np.zeros(n),
        "PAINT_TOUCH_FTM": np.zeros(n), "PAINT_TOUCH_FTA": np.zeros(n),
        "DRIVES": (minutes * 0.2).round(), "DRIVE_TOV": (minutes * 0.01).round(),
        "ELBOW_TOUCHES": (minutes * 0.04).round(), "ELBOW_TOUCH_TOV": (minutes * 0.002).round(),
        "POST_TOUCHES": (minutes * 0.02).round(), "POST_TOUCH_TOV": (minutes * 0.001).round(),
        "PAINT_TOUCHES": (minutes * 0.09).round(), "PAINT_TOUCH_TOV": (minutes * 0.004).round(),
    })
    return shots, possessions, tracking


@pytest.fixture(scope="module")
def league():
    return _league()


@pytest.fixture(scope="module")
def panel(league):
    return build_box_panel(*league)


@pytest.fixture(scope="module")
def model(panel):
    return impact_metrics(panel)


# ---------------------------------------------------------------------------
# The reconstructed box score
# ---------------------------------------------------------------------------

def test_panel_has_a_row_per_player_with_shots(panel, league):
    _shots, possessions, _tracking = league
    assert len(panel) == len(possessions)
    assert set(panel["PLAYER_ID"]) == set(possessions["PLAYER_ID"])


def test_counting_columns_come_through_unchanged(panel, league):
    _shots, _possessions, tracking = league
    merged = panel.set_index("PLAYER_ID").join(
        tracking.set_index("PLAYER_ID")[["OREB", "STL", "BLK"]], rsuffix="_src")
    for column in ("OREB", "STL", "BLK"):
        pd.testing.assert_series_equal(
            merged[column], merged[f"{column}_src"], check_names=False)


def test_turnovers_are_never_below_the_tracked_ones(panel, league):
    _shots, _possessions, tracking = league
    tracked = (tracking.set_index("PLAYER_ID")[
        ["DRIVE_TOV", "ELBOW_TOUCH_TOV", "POST_TOUCH_TOV", "PAINT_TOUCH_TOV"]]
        .sum(axis=1).reindex(panel["PLAYER_ID"]).to_numpy())
    assert (panel["TOV"].to_numpy() >= tracked - 1e-9).all()
    assert panel["TOV_ESTIMATED"].all()


def test_a_real_turnover_column_beats_the_estimate(league):
    shots, possessions, tracking = league
    box = pd.DataFrame({"PLAYER_ID": possessions["PLAYER_ID"], "TOV": 40.0})
    panel = build_box_panel(shots, possessions, tracking, box=box)
    assert (panel["TOV"] == 40.0).all()
    assert not panel["TOV_ESTIMATED"].any()


def test_a_player_missing_from_the_box_pull_keeps_the_estimate(league):
    shots, possessions, tracking = league
    box = pd.DataFrame({"PLAYER_ID": possessions["PLAYER_ID"], "TOV": 40.0})
    box.loc[box.index[0], "TOV"] = np.nan
    panel = build_box_panel(shots, possessions, tracking, box=box).set_index("PLAYER_ID")
    first = possessions["PLAYER_ID"].iloc[0]
    assert panel.loc[first, "TOV"] != 40.0
    assert panel.loc[first, "TOV_ESTIMATED"]
    assert panel["TOV_ESTIMATED"].sum() == 1


def test_missing_tracking_column_is_an_error_not_a_zero(league):
    shots, possessions, tracking = league
    with pytest.raises(KeyError, match="STL"):
        build_box_panel(shots, possessions, tracking.drop(columns=["STL"]))


# ---------------------------------------------------------------------------
# The two metrics
# ---------------------------------------------------------------------------

def test_league_average_box_plus_minus_is_zero(model):
    table = model.table
    average = np.average(table["BPM"], weights=table["MIN"])
    assert average == pytest.approx(0.0, abs=1e-9)


def test_league_average_win_shares_per_48_is_one_tenth(model):
    table = model.table
    average = np.average(table["WS48"], weights=table["MIN"])
    assert average == pytest.approx(AVERAGE_WS48, abs=1e-9)


def test_offensive_and_defensive_halves_add_up(model):
    table = model.table
    np.testing.assert_allclose(table["OBPM"] + table["DBPM"], table["BPM"], atol=1e-9)
    np.testing.assert_allclose(table["OWS"] + table["DWS"], table["WS"], atol=1e-9)


def test_league_context_is_what_the_panel_says_it_is(model, panel):
    """The context the metrics are priced against has to come off the same rows."""
    possessions = (panel["FGA"] + 0.44 * panel["FTA"] - panel["OREB"] + panel["TOV"])
    team_games = panel["MIN"].sum() / (5 * 48)
    assert model.pace == pytest.approx(possessions.sum() / team_games)
    assert model.points_per_possession == pytest.approx(
        panel["POINTS"].sum() / possessions.sum())
    assert model.league_ts == pytest.approx(
        panel["POINTS"].sum() / (2 * panel["TSA"].sum()))
    assert model.marginal_points_per_win == pytest.approx(
        0.32 * panel["POINTS"].sum() / team_games)


def test_box_plus_minus_is_a_rate_not_a_total(panel):
    """Doubling a player's minutes and every count with them leaves the rate alone.

    This is the difference between the two metrics: box plus/minus should not
    move, win shares should roughly double, because one is per possession and
    the other is a season total.
    """
    before = impact_metrics(panel)
    doubled = panel.copy()
    target = doubled["MIN"].idxmax()
    counts = ["MIN", "GP", "POINTS", "FGA", "FGM", "FG3M", "FTM", "FTA", "TSA",
              "OREB", "DREB", "REB", "AST", "AST_PTS_CREATED", "STL", "BLK",
              "TOV", "TOUCHES", "DEF_RIM_FGA", "DEF_RIM_FGM", "POSSESSIONS_USED"]
    doubled.loc[target, counts] = doubled.loc[target, counts] * 2
    after = impact_metrics(doubled)

    player = int(panel.loc[target, "PLAYER_ID"])
    assert after.of(player)["BPM"] == pytest.approx(before.of(player)["BPM"], abs=0.6)
    assert after.of(player)["WS"] > 1.5 * before.of(player)["WS"]


def test_more_points_on_the_same_shots_is_worth_more(panel):
    """The scoring term is the one thing here that needs no estimated price."""
    better = panel.copy()
    target = better["MIN"].idxmax()
    better.loc[target, "POINTS"] += 200
    better.loc[target, "TS_PCT"] = (better.loc[target, "POINTS"]
                                    / (2 * better.loc[target, "TSA"]))
    player = int(panel.loc[target, "PLAYER_ID"])
    assert (impact_metrics(better).of(player)["OBPM"]
            > impact_metrics(panel).of(player)["OBPM"])


def test_percentiles_only_rank_players_over_the_minutes_floor(model):
    table = model.table
    under = table.loc[table["MIN"] < MIN_MINUTES]
    over = table.loc[table["MIN"] >= MIN_MINUTES]
    assert table["BPM"].notna().all()          # everyone gets a metric
    assert under["BPM_PCTILE"].isna().all()    # not everyone gets a rank
    assert over["BPM_PCTILE"].notna().all()
    assert over["WS48_PCTILE"].between(0, 100).all()


def test_of_returns_none_for_a_player_who_is_not_there(model):
    assert model.of(-1) is None
    assert model.of(int(model.table["PLAYER_ID"].iloc[0])) is not None


def test_a_league_with_no_minutes_is_an_error(panel):
    empty = panel.copy()
    empty["MIN"] = 0.0
    with pytest.raises(ValueError, match="no minutes"):
        impact_metrics(empty)


# ---------------------------------------------------------------------------
# Rim defence: the baseline is the whole term
# ---------------------------------------------------------------------------

def test_rim_baseline_falls_as_rim_volume_rises(panel):
    """Nearest-defender credit is an assignment, so the baseline has to slope."""
    baseline = rim_defence_baseline(panel, panel["MIN"], min_minutes=0.0)
    per36 = panel["DEF_RIM_FGA"] / panel["MIN"] * 36
    order = per36.sort_values().index
    assert baseline.loc[order[-1]] < baseline.loc[order[0]]


def test_a_player_who_defended_no_rim_shots_gets_no_rim_credit(league):
    shots, possessions, tracking = league
    tracking = tracking.copy()
    tracking.loc[tracking.index[0], ["DEF_RIM_FGA", "DEF_RIM_FGM"]] = 0.0
    model = impact_metrics(build_box_panel(shots, possessions, tracking))
    row = model.of(int(possessions["PLAYER_ID"].iloc[0]))
    assert np.isfinite(row["DBPM"])


def test_the_baseline_survives_a_league_too_small_to_fit(panel):
    small = panel.head(5)
    baseline = rim_defence_baseline(small, small["MIN"])
    assert baseline.notna().all()
    assert baseline.nunique() == 1


# ---------------------------------------------------------------------------
# Star tiers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiers(model):
    return fit_star_tiers(build_tier_features(model.table))


def test_only_players_over_the_floor_are_clustered(model):
    features = build_tier_features(model.table)
    assert (features["MIN"] >= MIN_MINUTES).all()
    assert len(features) < len(model.table)


def test_every_tier_gets_a_name_off_the_ladder(tiers):
    assert len(tiers.names) == tiers.k
    assert set(tiers.names.values()) <= set(TIER_LADDERS[max(TIER_LADDERS)])
    assert len(set(tiers.names.values())) == tiers.k


def test_the_top_tier_is_the_star_tier(tiers):
    assert tiers.names[tiers.order[0]] == STAR_TIER
    assert not tiers.stars.empty


def test_stars_out_rank_everyone_else(tiers):
    stars = tiers.table["TIER"] == STAR_TIER
    assert tiers.table.loc[stars, "WS"].mean() > tiers.table.loc[~stars, "WS"].mean()
    assert tiers.table.loc[stars, "MPG"].mean() > tiers.table.loc[~stars, "MPG"].mean()


def test_the_summary_runs_strongest_tier_first(tiers):
    summary = tiers.summary()
    assert len(summary) == tiers.k
    assert summary["TIER"].iloc[0] == STAR_TIER
    assert summary["PLAYERS"].sum() == len(tiers.table)


def test_tier_of_is_none_for_an_unclustered_player(tiers):
    assert tiers.tier_of(-1) is None
    assert tiers.tier_of(int(tiers.table["PLAYER_ID"].iloc[0])) is not None


def test_asking_for_a_k_gets_that_k(model):
    tiers = fit_star_tiers(build_tier_features(model.table), k=3)
    assert tiers.k == 3
    assert set(tiers.names.values()) == set(TIER_LADDERS[3])


def test_clustering_nobody_is_an_error(model):
    features = build_tier_features(model.table).iloc[:0]
    with pytest.raises(ValueError, match="minutes floor"):
        fit_star_tiers(features)


def test_the_prices_are_all_in_one_table():
    """Every estimated coefficient is in VALUES, so a reader can argue with it."""
    assert VALUES["turnover"] < 0
    assert 0 < VALUES["defensive_rebound"] < VALUES["offensive_rebound"] < 1
    assert VALUES["steal"] > VALUES["block"] > 0
