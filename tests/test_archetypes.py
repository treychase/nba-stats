import numpy as np
import pandas as pd
import pytest

from archetypes import (
    FEATURE_NAMES,
    FEATURES,
    MIN_MINUTES,
    PROTOTYPES,
    build_archetype_features,
    fit_archetypes,
)


def _tracking(n_per_group: int = 40, seed: int = 0) -> pd.DataFrame:
    """Three separable groups of players: bigs, creators and off-ball shooters."""
    rng = np.random.default_rng(seed)
    groups = {
        "big": {"PAINT_TOUCHES": 6.0, "POST_TOUCHES": 3.0, "OREB_CHANCES": 5.0,
                "DEF_RIM_FGA": 8.0, "BLK": 1.6, "CATCH_SHOOT_FG3A": 0.4,
                "TIME_OF_POSS": 1.2, "PULL_UP_FGA": 0.5},
        "creator": {"TIME_OF_POSS": 6.5, "POTENTIAL_AST": 12.0, "PULL_UP_FGA": 8.0,
                    "DRIVES": 14.0, "AVG_DRIB_PER_TOUCH": 5.5, "AVG_SEC_PER_TOUCH": 5.0,
                    "CATCH_SHOOT_FG3A": 1.5, "PAINT_TOUCHES": 1.0},
        "shooter": {"CATCH_SHOOT_FG3A": 7.0, "AVG_SEC_PER_TOUCH": 1.6,
                    "AVG_DRIB_PER_TOUCH": 0.6, "TIME_OF_POSS": 1.0,
                    "PASSES_MADE": 25.0, "PAINT_TOUCHES": 0.8},
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
    # a player's paint touches per 36 should not track their minutes
    corr = np.corrcoef(features["MIN"], features["Paint touches"])[0, 1]
    assert abs(corr) < 0.35
    assert set(FEATURE_NAMES) <= set(features.columns)


def test_the_minutes_floor_excludes_small_samples():
    tracking = _tracking()
    tracking.loc[0, "MIN"] = MIN_MINUTES - 1
    kept = build_archetype_features(tracking)
    assert tracking.loc[0, "PLAYER_ID"] not in set(kept["PLAYER_ID"])


def test_missing_tracking_columns_are_named_in_the_error():
    tracking = _tracking().drop(columns=["DRIVES"])
    with pytest.raises(KeyError, match="DRIVES"):
        build_archetype_features(tracking)


def test_clusters_recover_groups_that_really_are_separate():
    model = fit_archetypes(build_archetype_features(_tracking()), k=3)
    table = model.table.copy()
    table["GROUP"] = [name.split("-")[0] for name in table["PLAYER_NAME"]]
    # each planted group should land almost entirely in one cluster
    for _group, rows in table.groupby("GROUP"):
        share = rows["CLUSTER"].value_counts(normalize=True).iloc[0]
        assert share > 0.9


def test_k_is_chosen_by_silhouette_when_not_given():
    model = fit_archetypes(build_archetype_features(_tracking()))
    assert model.k in range(4, 9)
    assert model.scores and model.k == max(model.scores, key=model.scores.get)
    assert -1.0 <= model.silhouette <= 1.0


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
            {name: (2.0 if name == "Paint touches" else 0.0) for name in FEATURE_NAMES},
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
