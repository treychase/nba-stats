import numpy as np
import pandas as pd
import pytest

from true_shooting import (
    FT_TRIP_WEIGHT,
    build_shooting_panel,
    fit_shrinkage,
    project_true_shooting,
    split_half_frames,
    TRACKED_FT_COLUMNS,
)


# ---------------------------------------------------------------------------
# Fixtures: a small league whose true answers we know
# ---------------------------------------------------------------------------

def _shots(rng, player_id, n, made_rate, three_rate, start="2025-11-01", days=120):
    dates = pd.to_datetime(start) + pd.to_timedelta(rng.integers(0, days, n), unit="D")
    three = rng.random(n) < three_rate
    return pd.DataFrame({
        "PLAYER_ID": player_id,
        "GAME_DATE": dates.strftime("%Y%m%d").astype(int),
        "SHOT_MADE_FLAG": (rng.random(n) < made_rate).astype(int),
        "SHOT_TYPE": np.where(three, "3PT Field Goal", "2PT Field Goal"),
    })


def _league(n_players=60, seed=3):
    """Shots, possessions and tracking that agree with each other by construction."""
    rng = np.random.default_rng(seed)
    shot_frames, poss_rows, track_rows = [], [], []
    for i in range(n_players):
        pid = 1000 + i
        n = int(rng.integers(120, 900))
        frame = _shots(rng, pid, n, rng.uniform(0.42, 0.56), rng.uniform(0.1, 0.6))
        shot_frames.append(frame)

        three = frame["SHOT_TYPE"].str.startswith("3")
        fg_points = int((frame["SHOT_MADE_FLAG"] * np.where(three, 3, 2)).sum())
        ft_pct = float(rng.uniform(0.6, 0.9))
        ft_att = int(rng.integers(20, 300))
        ft_made = int(round(ft_att * ft_pct))
        tracked = max(4, int(ft_att * rng.uniform(0.35, 0.7)))
        poss_rows.append({
            "PLAYER_ID": pid, "PLAYER_NAME": f"P{i}", "TEAM_ABBREVIATION": "OKC",
            "GP": 70, "MIN": float(rng.uniform(600, 2400)),
            "POINTS": fg_points + ft_made,
        })
        row = {"PLAYER_ID": pid, "_ft_pct": ft_pct, "_fta": ft_att}
        for j, (made_col, att_col) in enumerate(TRACKED_FT_COLUMNS):
            share = tracked // len(TRACKED_FT_COLUMNS) if j else tracked - 3 * (tracked // len(TRACKED_FT_COLUMNS))
            row[att_col] = float(share)
            row[made_col] = float(round(share * ft_pct))
        track_rows.append(row)
    return (pd.concat(shot_frames, ignore_index=True),
            pd.DataFrame(poss_rows), pd.DataFrame(track_rows))


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def test_free_throws_made_are_recovered_exactly():
    shots, poss, track = _league()
    panel = build_shooting_panel(shots, poss, track)
    # FTM is arithmetic, not a fit: points minus what the field goals were worth
    expected = poss.set_index("PLAYER_ID")["POINTS"] - panel.set_index("PLAYER_ID")["FG_PTS"]
    got = panel.set_index("PLAYER_ID")["FTM"]
    assert np.allclose(got.sort_index(), expected.sort_index())
    assert (got >= 0).all()


def test_true_shooting_follows_its_definition():
    shots, poss, track = _league()
    panel = build_shooting_panel(shots, poss, track)
    assert np.allclose(panel["TSA"], panel["FGA"] + FT_TRIP_WEIGHT * panel["FTA"])
    assert np.allclose(panel["TS_PCT"], panel["POINTS"] / (2 * panel["TSA"]))
    # a league of real shooters lands where the real league does
    league = panel["POINTS"].sum() / (2 * panel["TSA"].sum())
    assert 0.50 < league < 0.65


def test_points_that_field_goals_alone_exceed_are_rejected():
    shots, poss, track = _league(n_players=8)
    poss.loc[0, "POINTS"] = 1.0
    with pytest.raises(ValueError, match="same season"):
        build_shooting_panel(shots, poss, track)


def test_a_thin_free_throw_sample_is_pooled_toward_the_league():
    shots, poss, track = _league()
    # one player with three tracked free throws, all made
    for made_col, att_col in TRACKED_FT_COLUMNS:
        track.loc[0, att_col] = 0.0
        track.loc[0, made_col] = 0.0
    track.loc[0, TRACKED_FT_COLUMNS[0][1]] = 3.0
    track.loc[0, TRACKED_FT_COLUMNS[0][0]] = 3.0
    panel = build_shooting_panel(shots, poss, track).set_index("PLAYER_ID")
    thin = panel.loc[track.loc[0, "PLAYER_ID"], "FT_PCT"]
    # three makes must not buy a perfect stroke
    assert 0.7 < thin < 0.9


def test_the_free_throw_prior_is_fitted_not_assumed():
    shots, poss, track = _league()
    panel = build_shooting_panel(shots, poss, track)
    assert 0.6 < panel.attrs["ft_league_rate"] < 0.9
    assert panel.attrs["ft_prior_attempts"] >= 5.0


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def test_the_split_keeps_every_shot_on_exactly_one_side():
    shots, _poss, _track = _league()
    early, late, cut = split_half_frames(shots)
    assert len(early) + len(late) == len(shots)
    assert len(early) and len(late)
    assert pd.to_datetime(early["GAME_DATE"], format="%Y%m%d").max() < cut
    assert pd.to_datetime(late["GAME_DATE"], format="%Y%m%d").min() >= cut


def test_shrinkage_beats_both_extremes_out_of_sample():
    """The point of partial pooling: better than trusting a player, or ignoring him."""
    shots, _poss, _track = _league(n_players=120, seed=11)
    fit = fit_shrinkage(shots)
    assert fit["k"] > 0
    assert fit["rmse"] <= fit["rmse_unpooled"]
    assert fit["rmse"] <= fit["rmse_fully_pooled"]
    assert fit["n_players"] > 30
    assert len(fit["curve"]) > 10


def test_shrinkage_refuses_a_sample_too_small_to_calibrate_on():
    shots, _poss, _track = _league(n_players=5)
    with pytest.raises(ValueError, match="split-half"):
        fit_shrinkage(shots)


# ---------------------------------------------------------------------------
# The projection
# ---------------------------------------------------------------------------

def _projection(k=150.0, **kwargs):
    shots, poss, track = _league()
    panel = build_shooting_panel(shots, poss, track)
    return panel, project_true_shooting(panel, k=k, **kwargs)


def test_the_pooling_weight_is_attempts_over_attempts_plus_k():
    panel, proj = _projection(k=150.0)
    table = proj.table
    assert np.allclose(table["WEIGHT"], table["TSA"] / (table["TSA"] + 150.0))
    assert np.allclose(
        table["PROJECTED_TS"],
        table["WEIGHT"] * table["TS_PCT"] + (1 - table["WEIGHT"]) * table["GROUP_TS"])


def test_small_samples_move_and_large_ones_barely_do():
    _panel, proj = _projection()
    table = proj.table.sort_values("TSA")
    small = table.head(10)["SHIFT"].abs().mean()
    large = table.tail(10)["SHIFT"].abs().mean()
    assert small > large
    # and the estimate is always between what he shot and what his group shot
    lower = np.minimum(table["TS_PCT"], table["GROUP_TS"]) - 1e-9
    upper = np.maximum(table["TS_PCT"], table["GROUP_TS"]) + 1e-9
    assert ((table["PROJECTED_TS"] >= lower) & (table["PROJECTED_TS"] <= upper)).all()


def test_uncertainty_shrinks_as_attempts_grow():
    _panel, proj = _projection()
    table = proj.table.sort_values("TSA")
    assert table["SE"].is_monotonic_decreasing
    assert (table["SE"] > 0).all()
    assert 0 < proj.tau < table["TS_PCT"].std()


def test_a_player_is_pooled_toward_his_archetype_not_the_league():
    shots, poss, track = _league()
    panel = build_shooting_panel(shots, poss, track)
    # split the league in two by observed true shooting, so the groups differ
    order = panel.sort_values("TS_PCT")["PLAYER_ID"]
    groups = pd.Series(["low"] * len(order), index=order.to_numpy())
    groups.iloc[len(order) // 2:] = "high"

    pooled = project_true_shooting(panel, k=150.0, groups=groups, min_attempts=0.0)
    means = dict(zip(pooled.groups["ARCHETYPE"], pooled.groups["GROUP_TS"]))
    assert means["high"] > means["low"]
    for _, row in pooled.table.iterrows():
        assert row["GROUP_TS"] == pytest.approx(means[groups[row["PLAYER_ID"]]])


def test_a_player_without_an_archetype_is_pooled_toward_the_league():
    shots, poss, track = _league()
    panel = build_shooting_panel(shots, poss, track)
    groups = pd.Series({panel["PLAYER_ID"].iloc[0]: "big"})
    proj = project_true_shooting(panel, k=150.0, groups=groups)
    orphans = proj.table.loc[proj.table["ARCHETYPE"].isna()]
    assert len(orphans) == len(panel) - 1
    assert np.allclose(orphans["GROUP_TS"], proj.league)


def test_group_means_ignore_the_small_samples_they_are_about_to_absorb():
    """A group's centre must not be set by the players being pulled toward it."""
    shots, poss, track = _league()
    panel = build_shooting_panel(shots, poss, track)
    groups = pd.Series("everyone", index=panel["PLAYER_ID"].to_numpy())

    loose = project_true_shooting(panel, k=150.0, groups=groups, min_attempts=0.0)
    strict = project_true_shooting(panel, k=150.0, groups=groups, min_attempts=600.0)
    assert strict.groups["PLAYERS"].iloc[0] < loose.groups["PLAYERS"].iloc[0]
    assert strict.league != loose.league


def test_a_floor_nobody_clears_is_an_error_not_an_empty_answer():
    panel, _proj = _projection()
    with pytest.raises(ValueError, match="true shooting attempts"):
        project_true_shooting(panel, k=150.0, min_attempts=1e9)


def test_lookup_returns_none_for_a_player_who_is_not_there():
    _panel, proj = _projection()
    assert proj.of(-1) is None
    assert proj.of(int(proj.table["PLAYER_ID"].iloc[0]))["PLAYER_NAME"]
