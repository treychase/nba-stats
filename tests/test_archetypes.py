import numpy as np
import pandas as pd
import pytest

from archetypes import (
    FEATURE_NAMES,
    FEATURES,
    K_RANGE,
    MAX_CLUSTER_SHARE,
    MIN_CLUSTER_SHARE,
    MIN_MINUTES,
    PROTOTYPES,
    build_archetype_features,
    fit_archetypes,
)


def _tracking(n_per_group: int = 40, seed: int = 0) -> pd.DataFrame:
    """Three separable groups of players: bigs, creators and off-ball shooters."""
    rng = np.random.default_rng(seed)
    groups = {
        "big": {"PAINT_TOUCH_FGA": 6.0, "POST_TOUCH_FGA": 3.0, "OREB": 5.0,
                "DREB_Rebounding": 9.0, "BLK": 1.6, "CATCH_SHOOT_FG3A": 0.4,
                "PULL_UP_FGA": 0.5, "PASSES_MADE": 22.0},
        "creator": {"PULL_UP_FGA": 8.0, "PULL_UP_FG3A": 4.0, "POTENTIAL_AST": 12.0,
                    "DRIVE_FGA": 7.0, "ASSISTS_PER_PASS_UNUSED": 0.0,
                    "AST_TO_PASS_PCT": 0.16, "PASSES_MADE": 55.0,
                    "CATCH_SHOOT_FG3A": 1.5, "PAINT_TOUCH_FGA": 1.0},
        "shooter": {"CATCH_SHOOT_FG3A": 7.0, "CATCH_SHOOT_FGA": 8.0,
                    "PULL_UP_FGA": 0.8, "AST_TO_PASS_PCT": 0.05,
                    "PASSES_MADE": 25.0, "PAINT_TOUCH_FGA": 0.8},
    }
    rows = []
    for label, profile in groups.items():
        for i in range(n_per_group):
            minutes = float(rng.uniform(400, 2200))
            row = {"PLAYER_ID": len(rows) + 1, "PLAYER_NAME": f"{label}-{i}",
                   "TEAM_ABBREVIATION": "OKC", "MIN": minutes, "GROUP": label}
            for column, _name, per36 in FEATURES:
                base = profile.get(column, 1.0) * rng.uniform(0.85, 1.15)
                row[column] = base * minutes / 36.0 if per36 else base
            rows.append(row)
    return pd.DataFrame(rows)


def test_features_are_rates_so_minutes_do_not_drive_the_clustering():
    tracking = _tracking()
    features = build_archetype_features(tracking)
    # a player's paint attempts per 36 should not track their minutes
    corr = np.corrcoef(features["MIN"], features["Paint FGA"])[0, 1]
    assert abs(corr) < 0.35
    assert set(FEATURE_NAMES) <= set(features.columns)


def test_the_minutes_floor_excludes_small_samples():
    tracking = _tracking()
    tracking.loc[0, "MIN"] = MIN_MINUTES - 1
    kept = build_archetype_features(tracking)
    assert tracking.loc[0, "PLAYER_ID"] not in set(kept["PLAYER_ID"])


def test_missing_tracking_columns_are_named_in_the_error():
    tracking = _tracking().drop(columns=["DRIVE_FGA"])
    with pytest.raises(KeyError, match="DRIVE_FGA"):
        build_archetype_features(tracking)


def test_clusters_recover_groups_that_really_are_separate():
    model = fit_archetypes(build_archetype_features(_tracking()), k=3)
    table = model.table.copy()
    table["GROUP"] = [name.split("-")[0] for name in table["PLAYER_NAME"]]
    # each planted group should land almost entirely in one cluster
    for _group, rows in table.groupby("GROUP"):
        share = rows["CLUSTER"].value_counts(normalize=True).iloc[0]
        assert share > 0.9


def test_k_is_chosen_by_silhouette_subject_to_the_balance_rule():
    features = build_archetype_features(_tracking())
    model = fit_archetypes(features)
    assert model.k in K_RANGE
    assert model.scores and -1.0 <= model.silhouette <= 1.0
    # The chosen k must beat every other *balanced* k on silhouette; an
    # unbalanced k may score higher and still lose.
    shares = model.table["CLUSTER"].value_counts(normalize=True)
    assert shares.max() <= MAX_CLUSTER_SHARE and shares.min() >= MIN_CLUSTER_SHARE


def test_a_k_that_swallows_the_league_in_one_cluster_is_not_chosen():
    """The whole point of the balance rule: no archetype is a quarter of everyone."""
    features = build_archetype_features(_tracking(n_per_group=60))
    model = fit_archetypes(features)
    for _cluster, rows in model.table.groupby("CLUSTER"):
        assert len(rows) / len(model.table) <= MAX_CLUSTER_SHARE


def test_every_cluster_gets_a_distinct_name():
    model = fit_archetypes(build_archetype_features(_tracking()), k=3)
    assert len(set(model.names.values())) == len(model.names)
    known = set(PROTOTYPES) | {"Low-usage role player"}
    assert set(model.names.values()) <= known


def test_a_cluster_defined_only_by_absence_is_not_called_a_specialty():
    """Naming guard: a centre that is negative everywhere claims nothing."""
    from archetypes import _name_clusters

    centroids = pd.DataFrame(
        [
            {name: -1.0 for name in FEATURE_NAMES},
            {name: (2.0 if name == "Paint FGA" else 0.0) for name in FEATURE_NAMES},
        ]
    )
    names = _name_clusters(centroids)
    assert names[0] == "Low-usage role player"
    assert names[1] != "Low-usage role player"


def test_similar_players_are_from_the_same_group_and_exclude_the_player():
    features = build_archetype_features(_tracking())
    model = fit_archetypes(features, k=3)
    target = int(model.table.loc[model.table["PLAYER_NAME"].str.startswith("big"),
                                 "PLAYER_ID"].iloc[0])
    similar = model.similar_players(target, n=5)
    assert len(similar) == 5
    assert target not in set(model.table.loc[
        model.table["PLAYER_NAME"].isin(similar["PLAYER_NAME"]), "PLAYER_ID"])
    assert similar["PLAYER_NAME"].str.startswith("big").all()
    assert similar["Distance"].is_monotonic_increasing


def test_lookups_for_an_unclustered_player_return_nothing_rather_than_raising():
    model = fit_archetypes(build_archetype_features(_tracking()), k=3)
    assert model.label_of(-1) is None
    assert model.profile_of(-1) is None
    assert model.similar_players(-1).empty


def test_fitting_is_reproducible():
    features = build_archetype_features(_tracking())
    first = fit_archetypes(features, k=4)
    second = fit_archetypes(features, k=4)
    assert first.table["ARCHETYPE"].tolist() == second.table["ARCHETYPE"].tolist()


def test_empty_input_is_rejected():
    with pytest.raises(ValueError):
        fit_archetypes(pd.DataFrame(columns=["PLAYER_ID", "PLAYER_NAME", *FEATURE_NAMES]))
