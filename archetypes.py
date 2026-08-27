"""Player archetypes: k-means over the tracking data, with names attached.

Position labels stopped describing how NBA players are used a while ago, so
this clusters players on *what they actually do* - where they get the ball, how
they shoot, how much they create, how they rebound and defend - and gives each
cluster a readable name.

Three decisions shape the result:

**Rates, not totals.** Every counting feature is divided by minutes and scaled
to 36, so a starter and a reserve who play the same way land in the same
cluster instead of the clustering rediscovering the minutes rotation.

**A minutes floor.** Players under :data:`MIN_MINUTES` are not clustered at
all. A player with eighty minutes has rate stats that are mostly noise, and
including them drags cluster centres toward whatever their small samples say.
They come back as ``None`` rather than being forced into a bucket.

**Named by prototype, not by hand.** ``k`` is chosen by silhouette score over
:data:`K_RANGE`, so the number of archetypes is read off the data. Each cluster
centre is then matched to the nearest of the prototypes in
:data:`PROTOTYPES` - a weight vector over the same standardized features - and
takes that prototype's name. The features that actually distinguish the cluster
travel with it, so a name is never the only thing on offer: a coach can see
that "Rim-running big" means paint touches and offensive rebound chances two
standard deviations up, and 3-point volume down.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

__all__ = [
    "FEATURES",
    "K_RANGE",
    "MIN_MINUTES",
    "PROTOTYPES",
    "ArchetypeModel",
    "build_archetype_features",
    "fit_archetypes",
]

MIN_MINUTES = 250.0
K_RANGE = range(4, 9)
PER = 36.0

# (source column, feature name, per-36 or as-is)
FEATURES: list[tuple[str, str, bool]] = [
    ("PAINT_TOUCHES", "Paint touches", True),
    ("POST_TOUCHES", "Post touches", True),
    ("ELBOW_TOUCHES", "Elbow touches", True),
    ("DRIVES", "Drives", True),
    ("CATCH_SHOOT_FG3A", "Catch & shoot 3PA", True),
    ("PULL_UP_FGA", "Pull-up FGA", True),
    ("POTENTIAL_AST", "Potential assists", True),
    ("PASSES_MADE", "Passes made", True),
    ("TIME_OF_POSS", "Time of possession", True),
    ("AVG_SEC_PER_TOUCH", "Seconds per touch", False),
    ("AVG_DRIB_PER_TOUCH", "Dribbles per touch", False),
    ("OREB_CHANCES", "Off. board chances", True),
    ("DREB_CHANCES", "Def. board chances", True),
    ("DEF_RIM_FGA", "Rim shots defended", True),
    ("BLK", "Blocks", True),
    ("STL", "Steals", True),
    ("AVG_SPEED_OFF", "Speed on offense", False),
    ("PTS_PER_TOUCH", "Points per touch", False),
]

FEATURE_NAMES = [name for _column, name, _rate in FEATURES]

# Prototype archetypes as weights over the standardized features. A cluster
# takes the name of whichever prototype its centre points most towards; the
# weights are directions, not thresholds, so a cluster that is emphatically one
# thing wins over a cluster that is mildly several.
PROTOTYPES: dict[str, dict[str, float]] = {
    "Rim-running big": {
        "Paint touches": 1.0, "Off. board chances": 1.0, "Rim shots defended": 0.8,
        "Blocks": 0.6, "Catch & shoot 3PA": -0.8, "Pull-up FGA": -0.6,
        "Time of possession": -0.5,
    },
    "Stretch big": {
        "Catch & shoot 3PA": 1.0, "Def. board chances": 0.8, "Rim shots defended": 0.6,
        "Paint touches": 0.4, "Drives": -0.4, "Dribbles per touch": -0.5,
    },
    "Post hub": {
        "Post touches": 1.2, "Elbow touches": 0.8, "Passes made": 0.5,
        "Seconds per touch": 0.5, "Pull-up FGA": -0.3,
    },
    "Primary creator": {
        "Time of possession": 1.2, "Potential assists": 1.0, "Pull-up FGA": 0.8,
        "Dribbles per touch": 0.8, "Drives": 0.6, "Off. board chances": -0.6,
    },
    "Secondary playmaker": {
        "Passes made": 0.9, "Potential assists": 0.7, "Drives": 0.6,
        "Catch & shoot 3PA": 0.4, "Post touches": -0.4,
    },
    "Off-ball shooter": {
        "Catch & shoot 3PA": 1.2, "Seconds per touch": -0.7, "Dribbles per touch": -0.8,
        "Time of possession": -0.6, "Paint touches": -0.5, "Speed on offense": 0.4,
    },
    "Slashing wing": {
        "Drives": 1.0, "Paint touches": 0.6, "Speed on offense": 0.5,
        "Points per touch": 0.4, "Catch & shoot 3PA": 0.3, "Post touches": -0.3,
    },
    "Defensive specialist": {
        "Steals": 0.9, "Rim shots defended": 0.7, "Blocks": 0.6,
        "Def. board chances": 0.5, "Time of possession": -0.7,
        "Pull-up FGA": -0.6, "Potential assists": -0.5,
    },
    "Low-usage connector": {
        "Passes made": 0.6, "Speed on offense": 0.4, "Time of possession": -0.8,
        "Pull-up FGA": -0.7, "Post touches": -0.5, "Paint touches": -0.3,
    },
}


@dataclass
class ArchetypeModel:
    """A fitted clustering, plus everything the dashboard needs to explain it."""

    table: pd.DataFrame            # one row per clustered player, with the label
    centroids: pd.DataFrame        # cluster x feature, in standard deviations
    names: dict[int, str]          # cluster id -> archetype name
    k: int
    silhouette: float
    scores: dict[int, float] = field(default_factory=dict)   # k -> silhouette
    _z: pd.DataFrame = field(default=None, repr=False)

    def label_of(self, player_id: int) -> str | None:
        """The archetype name for a player, or None if they were not clustered."""
        match = self.table.loc[self.table["PLAYER_ID"] == player_id]
        return None if match.empty else str(match["ARCHETYPE"].iloc[0])

    def profile_of(self, player_id: int) -> pd.Series | None:
        """One player's standardized features, or None if they were not clustered."""
        if self._z is None or player_id not in self._z.index:
            return None
        return self._z.loc[player_id]

    def distinguishing(self, cluster: int, n: int = 5) -> pd.Series:
        """The features that set a cluster apart, largest absolute z first."""
        centre = self.centroids.loc[cluster]
        return centre.reindex(centre.abs().sort_values(ascending=False).index)[:n]

    def similar_players(self, player_id: int, n: int = 5) -> pd.DataFrame:
        """The n closest players in the standardized feature space.

        Distance is over the same features the clustering uses, so a comparison
        is "used the same way", not "scores the same amount".
        """
        if self._z is None or player_id not in self._z.index:
            return pd.DataFrame(columns=["PLAYER_NAME", "ARCHETYPE", "Distance"])
        target = self._z.loc[player_id].to_numpy()
        distances = np.linalg.norm(self._z.to_numpy() - target, axis=1)
        order = np.argsort(distances)
        rows = []
        for index in order:
            other = self._z.index[index]
            if other == player_id:
                continue
            row = self.table.loc[self.table["PLAYER_ID"] == other].iloc[0]
            rows.append({
                "PLAYER_NAME": row["PLAYER_NAME"],
                "ARCHETYPE": row["ARCHETYPE"],
                "Distance": float(distances[index]),
            })
            if len(rows) == n:
                break
        return pd.DataFrame(rows)


def build_archetype_features(tracking: pd.DataFrame,
                             min_minutes: float = MIN_MINUTES) -> pd.DataFrame:
    """Per-36 rate features for every player who clears the minutes floor."""
    missing = [column for column, _name, _rate in FEATURES if column not in tracking.columns]
    if missing:
        raise KeyError(f"tracking data is missing {missing}")

    frame = tracking.copy()
    frame["MIN"] = pd.to_numeric(frame["MIN"], errors="coerce")
    frame = frame.loc[frame["MIN"].fillna(0) >= min_minutes]

    out = pd.DataFrame({
        "PLAYER_ID": frame["PLAYER_ID"].astype(int).values,
        "PLAYER_NAME": frame["PLAYER_NAME"].values,
    })
    if "TEAM_ABBREVIATION" in frame.columns:
        out["TEAM_ABBREVIATION"] = frame["TEAM_ABBREVIATION"].values
    out["MIN"] = frame["MIN"].values

    for column, name, per36 in FEATURES:
        values = pd.to_numeric(frame[column], errors="coerce")
        out[name] = (values / frame["MIN"] * PER).values if per36 else values.values

    return out.dropna(subset=FEATURE_NAMES).reset_index(drop=True)


LOW_USAGE_NAME = "Low-usage role player"


def _defined_by_absence(centre: pd.Series, top: int = 3) -> bool:
    """True when a cluster's signature is everything it does *not* do.

    A centre whose largest departures from average are all negative describes a
    player who is on the floor without the ball, and calling that a specialty -
    "defensive specialist", say - claims something the features do not show.
    """
    strongest = centre.reindex(centre.abs().sort_values(ascending=False).index)[:top]
    return bool((strongest < 0).all())


def _name_clusters(centroids: pd.DataFrame) -> dict[int, str]:
    """Match each cluster centre to its nearest prototype, no name used twice."""
    scores = {}
    for cluster, centre in centroids.iterrows():
        for name, weights in PROTOTYPES.items():
            weight = pd.Series(weights).reindex(centroids.columns).fillna(0.0)
            norm = np.linalg.norm(weight.to_numpy()) * np.linalg.norm(centre.to_numpy())
            scores[(cluster, name)] = float(centre @ weight) / norm if norm else 0.0

    names: dict[int, str] = {}
    taken: set[str] = set()

    # A cluster defined only by absence gets the honest name before any
    # prototype can claim it.
    for cluster, centre in centroids.iterrows():
        if _defined_by_absence(centre):
            names[cluster] = LOW_USAGE_NAME

    # Strongest match first, so the most emphatic cluster gets first claim on a
    # name and the rest fall to their best remaining fit.
    for (cluster, name), _score in sorted(scores.items(), key=lambda kv: -kv[1]):
        if cluster in names or name in taken:
            continue
        names[cluster] = name
        taken.add(name)
    return names


def fit_archetypes(features: pd.DataFrame, k: int | None = None,
                   seed: int = 610) -> ArchetypeModel:
    """Cluster players into archetypes, choosing k by silhouette when not given."""
    if features.empty:
        raise ValueError("no players cleared the minutes floor")

    matrix = features[FEATURE_NAMES].to_numpy(dtype=float)
    z = StandardScaler().fit_transform(matrix)

    scores: dict[int, float] = {}
    if k is None:
        for candidate in K_RANGE:
            if candidate >= len(features):
                continue
            labels = KMeans(n_clusters=candidate, n_init=10,
                            random_state=seed).fit_predict(z)
            scores[candidate] = float(silhouette_score(z, labels))
        if not scores:
            raise ValueError("not enough players to cluster")
        k = max(scores, key=scores.get)

    model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(z)
    labels = model.labels_
    silhouette = float(silhouette_score(z, labels)) if k > 1 else float("nan")

    centroids = pd.DataFrame(model.cluster_centers_, columns=FEATURE_NAMES)
    centroids.index.name = "CLUSTER"
    names = _name_clusters(centroids)

    table = features.copy()
    table["CLUSTER"] = labels
    table["ARCHETYPE"] = [names[label] for label in labels]

    z_frame = pd.DataFrame(z, columns=FEATURE_NAMES, index=features["PLAYER_ID"].values)
    z_frame.index.name = "PLAYER_ID"

    return ArchetypeModel(table=table, centroids=centroids, names=names, k=int(k),
                          silhouette=silhouette, scores=scores, _z=z_frame)
