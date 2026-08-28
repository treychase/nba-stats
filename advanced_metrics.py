"""Box plus/minus and win shares per 48, reconstructed from the tracking pulls.

Two questions a scouting page has to answer that a shot chart cannot. *How
much is this player worth per possession?* and *how many wins is that?* Both
are box score metrics, and the committed pulls do not carry a box score - so
this module builds one, and then builds the two metrics on top of it in a
single possession-value framework.

Reconstructing the box score
----------------------------
Everything except turnovers comes out of the pulls exactly:

===================  ====================================================
Points, minutes, GP  the possessions tracking pull
FGA, FGM, FG3M       the shot chart, one row per attempt
FTM, FTA             :mod:`true_shooting` - made free throws are points
                     minus what the field goals were worth, attempts are
                     that over a pooled free throw percentage
OREB, DREB, AST      the rebounding and passing tracking pulls
STL, BLK, rim FGA    the defense tracking pull
===================  ====================================================

Summed over the league those land on 115.3 points, 88.8 field goal attempts,
32.2 defensive rebounds and 26.6 assists per team game - the season's real
rates - which is the check that the reconstruction is not quietly dropping or
double counting anybody.

**Turnovers are the one estimate.** The tracking pulls count turnovers on the
plays they track - drives, and paint, post and elbow touches - which is about
a third of them. The rest are spread over the touches that nothing tracks, so
each player's untracked touches are charged at a single league rate, and that
rate is set by the one thing about turnovers everybody knows:
:data:`LEAGUE_TOV_RATE` of possessions end in one. That is the same kind of
constant as the 0.44 in true shooting attempts - a league-wide fact used to
close an identity - and it is a module constant so a real box score pull can
replace it. What it cannot do is tell two players apart on the touches nobody
tracks, so a careless passer who does not drive is charged too little. When
``data/nba_player_box_*.csv`` carries a ``TOV`` column,
:func:`build_box_panel` uses it and this estimate is not reached at all.

The possession-value framework
------------------------------
Every box event is priced in points, relative to what an average possession is
worth (``ppp``, about 1.13 this season). The prices are estimates, not fitted
weights - there is no play-by-play here to regress against - so they are all
in one table, :data:`VALUES`, rather than buried in an expression:

* **Scoring** is exact and needs no price: ``2 * TSA * (TS - league TS)`` is
  the points a player scored above what the league would have scored on his
  own shot volume.
* **A turnover** returns nothing where a possession was worth ``ppp``.
* **An offensive rebound** buys a possession the defence was about to take,
  but a teammate would sometimes have got it anyway.
* **A defensive rebound** ends the opponent's possession - which teams do about
  seven times in ten regardless, so only the margin is credit.
* **A steal** ends a possession early *and* starts one in transition, so it is
  worth more than the possession itself.
* **A block** forces a miss, and the defence recovers most of them.
* **Rim defence** is priced exactly rather than estimated, because the pull
  carries it: field goals defended at the rim, against the league's own rate
  from there.
* **Creation** credits the passer with a share of the points he assists.

:func:`box_plus_minus` sums those per 100 possessions and centres the league's
minute-weighted average at zero, which is what makes it a *plus/minus*: it is
points added over the average player, not points produced.

Win shares are the same events read through Dean Oliver's framework instead,
which is why the two are worth showing together rather than one being a
rescaling of the other. Offensive win shares are driven by scoring efficiency
and volume against a replacement baseline of ``0.92 * ppp``; defensive win
shares by stops per defensive possession. The league's minute-weighted average
WS/48 is pinned at :data:`AVERAGE_WS48` = 0.100, which is not a choice - it is
arithmetic. Win shares sum to team wins, teams average half their games, and a
player who plays a fifth of his team's minutes earns a fifth of a .500 team's
wins.

What is missing is worth saying plainly. There is no opponent data in the
committed pulls, so every team is treated as league-average defensively: a
player is credited for the stops he makes and not for the defence around him.
There are no personal fouls in them either. And the shot chart only covers
players over 50 attempts, so the fifth of the league below that has no
reconstructed box score at all and comes back unranked rather than guessed at.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from true_shooting import FT_TRIP_WEIGHT, build_shooting_panel

__all__ = [
    "AVERAGE_WS48",
    "LEAGUE_TOV_RATE",
    "MIN_MINUTES",
    "STAR_TIER",
    "TIER_LADDERS",
    "TRACKED_TOV_COLUMNS",
    "VALUES",
    "ImpactModel",
    "StarTiers",
    "build_box_panel",
    "build_tier_features",
    "fit_star_tiers",
    "rim_defence_baseline",
    "impact_metrics",
]

# Minutes before a player is ranked or clustered. Below it a per-possession
# rate is mostly the schedule, and the same floor the archetypes use keeps the
# two clusterings talking about the same population.
MIN_MINUTES = 250.0

# Turnovers per possession, league-wide. The one number here that does not come
# out of the pulls; see the module docstring.
LEAGUE_TOV_RATE = 0.130

# Turnovers the tracking pulls do count, by the play they happened on.
TRACKED_TOV_COLUMNS = [
    ("DRIVE_TOV", "DRIVES"),
    ("ELBOW_TOUCH_TOV", "ELBOW_TOUCHES"),
    ("POST_TOUCH_TOV", "POST_TOUCHES"),
    ("PAINT_TOUCH_TOV", "PAINT_TOUCHES"),
]

# What each box event is worth, in multiples of the value of a possession.
# Every one of these is an estimate; keeping them in one table is the point,
# so a reader can disagree with a number rather than with an expression.
VALUES = {
    # A possession that ends in a turnover produces nothing.
    "turnover": -1.00,
    # An extra possession, less the share a teammate would have got anyway.
    # The counterfactual for an individual rebound is not "the other team gets
    # it", it is "somebody in the same jersey gets it", which is why both of
    # these are well under a whole possession - and why the defensive one,
    # where a teammate almost always would have, is so much smaller.
    "offensive_rebound": 0.55,
    "defensive_rebound": 0.15,
    # A possession ended before a shot, plus the transition possession it
    # starts, which scores better than a possession off a made basket.
    "steal": 1.15,
    # A forced miss, most of which the defence recovers.
    "block": 0.60,
    # The passer's share of the points he sets up. The shooter keeps the rest
    # through his own scoring, which is why this is well under one.
    "assist_points": 0.30,
    # The share of a defended rim attempt that belongs to the nearest defender
    # rather than to the scheme that sent him there.
    "rim_defence": 0.50,
}

# Oliver's two baselines: offence is measured against a replacement level of
# 0.92 of league efficiency, defence against 1.08 of it.
OFFENSIVE_BASELINE = 0.92
DEFENSIVE_BASELINE = 1.08
# Marginal points per win, as a share of a team's points per game.
MARGINAL_POINTS_PER_WIN = 0.32

# A league-average player earns this many win shares per 48 minutes. Not a
# calibration choice: win shares sum to wins, teams average .500, so a player
# who plays a share of his team's minutes earns that share of 41 wins.
AVERAGE_WS48 = 0.100

MINUTES_PER_TEAM_GAME = 5 * 48.0


# ---------------------------------------------------------------------------
# The reconstructed box score
# ---------------------------------------------------------------------------

def _tracking_columns(tracking: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    """The counting columns the tracking pulls carry, aligned to the panel.

    A blank in a count column means the player never did that thing, which is a
    zero and not a missing value - the pulls simply leave the cell empty.
    """
    wanted = {
        "OREB": "OREB", "DREB": "DREB_Rebounding", "AST": "AST",
        "AST_PTS_CREATED": "AST_PTS_CREATED", "STL": "STL", "BLK": "BLK",
        "TOUCHES": "TOUCHES", "DEF_RIM_FGA": "DEF_RIM_FGA",
        "DEF_RIM_FGM": "DEF_RIM_FGM",
    }
    wanted.update({column: column for pair in TRACKED_TOV_COLUMNS for column in pair})

    missing = [source for source in wanted.values() if source not in tracking.columns]
    if missing:
        raise KeyError(f"tracking data is missing {sorted(set(missing))}")

    frame = tracking.set_index("PLAYER_ID")
    out = pd.DataFrame(index=index)
    for name, source in wanted.items():
        out[name] = pd.to_numeric(frame[source], errors="coerce").reindex(index).fillna(0.0)
    return out


def _estimate_turnovers(panel: pd.DataFrame) -> pd.Series:
    """Charge every untracked touch at the one rate that closes the league total.

    The tracked plays give each player's real turnovers on drives and on paint,
    post and elbow touches. What is left over league-wide is whatever
    :data:`LEAGUE_TOV_RATE` implies minus that, and it is spread across
    untracked touches at a flat rate - so a player who handles the ball more is
    charged more, and nothing else distinguishes them.
    """
    tracked = sum(panel[column] for column, _events in TRACKED_TOV_COLUMNS)
    tracked_events = sum(panel[events] for _column, events in TRACKED_TOV_COLUMNS)
    untracked_touches = (panel["TOUCHES"] - tracked_events).clip(lower=0.0)

    # possessions = FGA + 0.44 * FTA - OREB + TOV, and TOV is a fixed share of
    # possessions, so the league's turnover total follows from the rest.
    without_turnovers = float(
        panel["FGA"].sum() + FT_TRIP_WEIGHT * panel["FTA"].sum() - panel["OREB"].sum())
    league_total = without_turnovers * LEAGUE_TOV_RATE / (1.0 - LEAGUE_TOV_RATE)

    remaining = league_total - float(tracked.sum())
    denominator = float(untracked_touches.sum())
    rate = max(remaining, 0.0) / denominator if denominator > 0 else 0.0
    return tracked + rate * untracked_touches


def build_box_panel(shots: pd.DataFrame,
                    possessions: pd.DataFrame,
                    tracking: pd.DataFrame,
                    box: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per player: the reconstructed season box score.

    ``box`` is the optional season totals pull. Only its ``TOV`` column is used,
    and only if it is there - real turnovers beat the estimate, and nothing else
    in a box score pull improves on what the tracking data already carries
    exactly.
    """
    panel = build_shooting_panel(shots, possessions, tracking)
    panel = panel.set_index("PLAYER_ID")
    panel = panel.join(_tracking_columns(tracking, panel.index))
    panel["REB"] = panel["OREB"] + panel["DREB"]

    observed = None
    if box is not None and "TOV" in box.columns:
        observed = pd.to_numeric(
            box.set_index("PLAYER_ID")["TOV"], errors="coerce").reindex(panel.index)
    estimated = _estimate_turnovers(panel)
    panel["TOV"] = estimated if observed is None else observed.fillna(estimated)
    panel["TOV_ESTIMATED"] = True if observed is None else observed.isna()

    panel["POSSESSIONS_USED"] = (panel["FGA"] + FT_TRIP_WEIGHT * panel["FTA"]
                                 + panel["TOV"])
    return panel.reset_index()


# ---------------------------------------------------------------------------
# The metrics
# ---------------------------------------------------------------------------

@dataclass
class ImpactModel:
    """Box plus/minus and win shares for every player, with what set them."""

    table: pd.DataFrame          # a row per player, metrics and percentiles
    pace: float                  # possessions per 48 team minutes
    points_per_possession: float
    league_ts: float
    league_rim_pct: float
    marginal_points_per_win: float
    turnovers_estimated: bool = True

    def of(self, player_id: int) -> pd.Series | None:
        match = self.table.loc[self.table["PLAYER_ID"] == player_id]
        return None if match.empty else match.iloc[0]


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    total = float(weights.sum())
    return float((values * weights).sum() / total) if total > 0 else 0.0


def impact_metrics(panel: pd.DataFrame,
                   min_minutes: float = MIN_MINUTES) -> ImpactModel:
    """Price every player's box score in points, then in wins.

    Percentiles rank only players over ``min_minutes``; everyone else keeps
    their metrics and comes back unranked, because a rate over eighty minutes
    should not be allowed to push a rotation player down a percentile.
    """
    table = panel.copy()
    minutes = table["MIN"].astype(float)
    if float(minutes.sum()) <= 0:
        raise ValueError("no minutes in the panel")

    # League context, computed over the same players the metrics are for, so
    # the pace and the possessions it prices are the same population.
    possessions = (table["FGA"] + FT_TRIP_WEIGHT * table["FTA"]
                   - table["OREB"] + table["TOV"])
    team_games = float(minutes.sum()) / MINUTES_PER_TEAM_GAME
    pace = float(possessions.sum()) / team_games
    ppp = float(table["POINTS"].sum() / possessions.sum())
    league_ts = float(table["POINTS"].sum() / (2 * table["TSA"].sum()))
    rim_attempts = float(table["DEF_RIM_FGA"].sum())
    league_rim = float(table["DEF_RIM_FGM"].sum() / rim_attempts) if rim_attempts else 0.0

    # Possessions the player was on the floor for, which is what a per-100
    # rate is per - not the possessions he used.
    floor_possessions = minutes / MINUTES_PER_TEAM_GAME * pace * 5

    scoring = 2 * table["TSA"] * (table["TS_PCT"] - league_ts)
    creation = VALUES["assist_points"] * table["AST_PTS_CREATED"]
    turnovers = VALUES["turnover"] * ppp * table["TOV"]
    offensive_rebounds = VALUES["offensive_rebound"] * ppp * table["OREB"]

    defensive_rebounds = VALUES["defensive_rebound"] * ppp * table["DREB"]
    steals = VALUES["steal"] * ppp * table["STL"]
    blocks = VALUES["block"] * ppp * table["BLK"]
    # These are shots actually defended, against what the league allows at the
    # same rim volume rather than against one pooled rate - see
    # rim_defence_baseline for why that distinction is the whole term.
    expected_rim = rim_defence_baseline(table, minutes, min_minutes)
    rim_defence = (VALUES["rim_defence"] * 2 * table["DEF_RIM_FGA"]
                   * (expected_rim - _rim_pct(table)))

    table["OFF_POINTS_ADDED"] = scoring + creation + turnovers + offensive_rebounds
    table["DEF_POINTS_ADDED"] = defensive_rebounds + steals + blocks + rim_defence
    points_added = table["OFF_POINTS_ADDED"] + table["DEF_POINTS_ADDED"]

    raw = points_added / floor_possessions * 100
    # Centred on the league's minute-weighted average, which is what turns
    # points added into points added *over an average player*.
    table["BPM"] = raw - _weighted_mean(raw, minutes)
    table["OBPM"] = (table["OFF_POINTS_ADDED"] / floor_possessions * 100)
    table["OBPM"] -= _weighted_mean(table["OBPM"], minutes)
    table["DBPM"] = table["BPM"] - table["OBPM"]

    # ---- win shares -------------------------------------------------------
    points_per_team_game = float(table["POINTS"].sum()) / team_games
    marginal_points_per_win = MARGINAL_POINTS_PER_WIN * points_per_team_game

    produced = table["POINTS"] + VALUES["assist_points"] * table["AST_PTS_CREATED"]
    used = table["POSSESSIONS_USED"]
    marginal_offense = produced - OFFENSIVE_BASELINE * ppp * used
    table["OWS"] = marginal_offense / marginal_points_per_win

    # With no opponent data every team defends at the league rate, so a
    # player's defensive rating moves only with the stops he makes himself.
    defensive_possessions = floor_possessions
    stops_per_100 = table["DEF_POINTS_ADDED"] / defensive_possessions * 100
    defensive_rating = 100 * ppp - stops_per_100
    marginal_defense = (defensive_possessions
                        * (DEFENSIVE_BASELINE * 100 * ppp - defensive_rating) / 100)
    table["DWS"] = marginal_defense / marginal_points_per_win

    per48 = (table["OWS"] + table["DWS"]) / minutes * 48
    # Pinned, not fitted: see AVERAGE_WS48. The shift lands on defence, which
    # is the half the missing opponent data makes a level rather than a spread.
    shift = AVERAGE_WS48 - _weighted_mean(per48, minutes)
    table["DWS"] = table["DWS"] + shift * minutes / 48
    table["WS"] = table["OWS"] + table["DWS"]
    table["WS48"] = table["WS"] / minutes * 48

    ranked = minutes >= min_minutes
    for column in ("BPM", "OBPM", "DBPM", "WS48", "WS"):
        table[f"{column}_PCTILE"] = (
            table[column].where(ranked).rank(pct=True) * 100)
    table["RANKED"] = ranked

    table = table.sort_values("BPM", ascending=False).reset_index(drop=True)
    return ImpactModel(
        table=table, pace=pace, points_per_possession=ppp, league_ts=league_ts,
        league_rim_pct=league_rim, marginal_points_per_win=marginal_points_per_win,
        turnovers_estimated=bool(table["TOV_ESTIMATED"].any()),
    )


def rim_defence_baseline(table: pd.DataFrame,
                         minutes: pd.Series,
                         min_minutes: float = MIN_MINUTES) -> pd.Series:
    """What the league allows at the rim, at each player's rim volume.

    The tracking pull credits a rim attempt to whoever was nearest, and being
    nearest is an assignment rather than a choice: a guard is the nearest
    defender on a shot at the rim mostly when he has already been beaten,
    while a centre is nearest on every possession he is doing his job. The
    league bears that out - defenders in the bottom fifth by rim attempts
    allow 72%, the top fifth 60% - so scoring every player against one pooled
    league rate would hand every centre free credit and charge every guard for
    playing on the perimeter.

    So the baseline moves with volume. Allowed rate is regressed on the log of
    rim attempts per 36 minutes, weighted by attempts and fitted only on
    players over the minutes floor, and a player is measured against what the
    league allows *at his own volume*. What is left is rim protection rather
    than position.
    """
    attempts = table["DEF_RIM_FGA"].astype(float)
    per36 = np.where(minutes > 0, attempts / minutes.replace(0, np.nan) * 36, np.nan)
    allowed = np.where(attempts > 0, table["DEF_RIM_FGM"] / attempts.replace(0, np.nan), np.nan)

    usable = (minutes >= min_minutes).to_numpy() & (attempts.to_numpy() > 0) & np.isfinite(per36)
    pooled = float(table.loc[attempts > 0, "DEF_RIM_FGM"].sum()
                   / max(attempts[attempts > 0].sum(), 1.0))
    if usable.sum() < 20:
        return pd.Series(pooled, index=table.index)

    x = np.log(per36[usable])
    y = allowed[usable]
    w = attempts.to_numpy()[usable]
    slope, intercept = np.polyfit(x, y, 1, w=np.sqrt(w))

    fitted = intercept + slope * np.log(np.where(per36 > 0, per36, np.nan))
    # A player who defended nothing at the rim has no term either way, so the
    # baseline he is scored against never matters; the pooled rate keeps the
    # arithmetic finite rather than saying anything about him.
    return pd.Series(np.where(np.isfinite(fitted), fitted, pooled), index=table.index)


def _rim_pct(table: pd.DataFrame) -> pd.Series:
    """Opponent field goal percentage at the rim, zero when nothing was defended."""
    attempts = table["DEF_RIM_FGA"].replace(0, np.nan)
    return (table["DEF_RIM_FGM"] / attempts).fillna(0.0)


# ---------------------------------------------------------------------------
# Stars, as a clustering rather than a cutoff
# ---------------------------------------------------------------------------
# The archetypes in `archetypes.py` cluster on per-36 rates and deliberately
# throw volume away, because their question is what a player does, not how much
# of it he does. That is exactly the wrong feature set for finding stars: a
# bench guard and a franchise guard shoot the same shots at different volumes
# and land in the same archetype, which is the archetype working correctly.
#
# So stars get their own clustering, on the other axis. Same method, different
# features: what a player is worth per possession, how many wins that is, and
# how much of the offence goes through him. A tier is then named by where its
# centre ranks, and the top one is the star tier - so "star" is where the
# league separates rather than a number somebody picked.

# Per game, not per 36. That is the whole difference between this clustering
# and the archetypes: they divide volume out on purpose, because their question
# is what a player does. A star is not a player who would be good if he played,
# so here volume goes back in - and a bench centre who rebounds well in fifteen
# minutes stops out-ranking a starter.
TIER_FEATURES: list[tuple[str, str]] = [
    ("BPM", "Box +/-"),
    ("WS", "Win shares"),
    ("PTS_PG", "Points / game"),
    ("USAGE_PG", "Shot volume / game"),
    ("CREATION_PG", "Points created / game"),
    ("MPG", "Minutes / game"),
]
TIER_FEATURE_NAMES = [name for _column, name in TIER_FEATURES]

TIER_K_RANGE = range(3, 7)

# Names run from the strongest centre down, so a tier is never named by hand.
TIER_LADDERS: dict[int, list[str]] = {
    3: ["Star", "Rotation regular", "Bench"],
    4: ["Star", "Starter", "Rotation regular", "Bench"],
    5: ["Star", "Secondary star", "Starter", "Rotation regular", "Bench"],
    6: ["Star", "Secondary star", "Starter", "Rotation regular", "Bench", "Fringe"],
}

# A star tier that holds a fifth of the league is not a star tier. The same
# kind of product constraint the archetypes put on cluster size, for the same
# reason: a label has to mean something to be worth showing.
MAX_STAR_SHARE = 0.12

STAR_TIER = "Star"


@dataclass
class StarTiers:
    """A clustering of players by impact, with the top tier called Star."""

    table: pd.DataFrame           # a row per clustered player, with TIER
    centroids: pd.DataFrame       # cluster x feature, in standard deviations
    names: dict[int, str]         # cluster id -> tier name
    k: int
    silhouette: float
    order: list[int] = field(default_factory=list)   # cluster ids, strongest first
    scores: dict[int, float] = field(default_factory=dict)

    def tier_of(self, player_id: int) -> str | None:
        """The tier a player landed in, or None if he was not clustered."""
        match = self.table.loc[self.table["PLAYER_ID"] == player_id]
        return None if match.empty else str(match["TIER"].iloc[0])

    @property
    def stars(self) -> pd.DataFrame:
        """Everyone in the top tier, best box plus/minus first."""
        return self.table.loc[self.table["TIER"] == STAR_TIER].sort_values(
            "BPM", ascending=False)

    def distinguishing(self, cluster: int, n: int = 3) -> pd.Series:
        """The features that set a tier apart, largest absolute z first."""
        centre = self.centroids.loc[cluster]
        return centre.reindex(centre.abs().sort_values(ascending=False).index)[:n]

    def summary(self) -> pd.DataFrame:
        """One row per tier, strongest first, with what the tier averages.

        Ordered by the ladder rather than by any single column: the tiers below
        the top two separate on minutes as much as on impact, so sorting the
        table by box plus/minus would print them out of the order they were
        named in.
        """
        rows = []
        for rank, (cluster, name) in enumerate(sorted(
                self.names.items(), key=lambda kv: self.order.index(kv[0]))):
            members = self.table.loc[self.table["CLUSTER"] == cluster]
            rows.append({
                "TIER": name,
                "PLAYERS": int(len(members)),
                "BPM": float(members["BPM"].mean()),
                "WS48": float(members["WS48"].mean()),
                "MPG": float(members["MPG"].mean()),
                "PTS_PG": float(members["PTS_PG"].mean()),
                "DEFINES": ", ".join(
                    f"{feature} {value:+.1f}"
                    for feature, value in self.distinguishing(cluster).items()),
                "LEADERS": ", ".join(members.nlargest(3, "WS")["PLAYER_NAME"]),
            })
        return pd.DataFrame(rows)


def build_tier_features(table: pd.DataFrame,
                        min_minutes: float = MIN_MINUTES) -> pd.DataFrame:
    """The impact features the tiers cluster on, for players over the floor."""
    frame = table.loc[table["MIN"].astype(float) >= min_minutes].copy()
    minutes = frame["MIN"].astype(float)
    games = frame["GP"].replace(0, np.nan).astype(float)

    frame["PTS_PG"] = frame["POINTS"] / games
    frame["USAGE_PG"] = frame["TSA"] / games
    frame["CREATION_PG"] = frame["AST_PTS_CREATED"] / games
    frame["MPG"] = minutes / games

    columns = ["PLAYER_ID", "PLAYER_NAME", "MIN", "GP", "WS48"]
    if "TEAM_ABBREVIATION" in frame.columns:
        columns.append("TEAM_ABBREVIATION")
    columns += [column for column, _name in TIER_FEATURES]
    return frame[columns].dropna(subset=[c for c, _ in TIER_FEATURES]).reset_index(drop=True)


def fit_star_tiers(features: pd.DataFrame, k: int | None = None,
                   seed: int = 610) -> StarTiers:
    """Cluster players by impact and name the tiers by rank, top one Star.

    ``k`` is chosen by silhouette over :data:`TIER_K_RANGE`, subject to the top
    tier holding no more than :data:`MAX_STAR_SHARE` of the clustered league.
    """
    if features.empty:
        raise ValueError("no players cleared the minutes floor")

    columns = [column for column, _name in TIER_FEATURES]
    z = StandardScaler().fit_transform(features[columns].to_numpy(dtype=float))

    scores: dict[int, float] = {}
    if k is None:
        allowed: dict[int, float] = {}
        for candidate in TIER_K_RANGE:
            if candidate >= len(features):
                continue
            model = KMeans(n_clusters=candidate, n_init=10, random_state=seed).fit(z)
            score = float(silhouette_score(z, model.labels_))
            scores[candidate] = score
            top = _top_cluster(model.cluster_centers_)
            if (model.labels_ == top).mean() <= MAX_STAR_SHARE:
                allowed[candidate] = score
        if not scores:
            raise ValueError("not enough players to cluster")
        k = max(allowed or scores, key=(allowed or scores).get)

    model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(z)
    labels = model.labels_
    silhouette = float(silhouette_score(z, labels)) if k > 1 else float("nan")

    centroids = pd.DataFrame(model.cluster_centers_, columns=TIER_FEATURE_NAMES)
    centroids.index.name = "CLUSTER"

    # Name by where the centre ranks on impact, so the ladder is read off the
    # clustering rather than assigned to it.
    ladder = TIER_LADDERS.get(k) or TIER_LADDERS[max(TIER_LADDERS)][:k]
    order = pd.Series(_tier_scores(model.cluster_centers_),
                      index=centroids.index).sort_values(ascending=False).index
    names = {int(cluster): ladder[rank] for rank, cluster in enumerate(order)}

    table = features.copy()
    table["CLUSTER"] = labels
    table["TIER"] = [names[int(label)] for label in labels]
    return StarTiers(table=table, centroids=centroids, names=names, k=int(k),
                     silhouette=silhouette, order=[int(c) for c in order],
                     scores=scores)


def _tier_scores(centres: np.ndarray) -> np.ndarray:
    """How prominent each cluster's centre is, across every feature at once.

    Ranking the tiers on impact alone puts a bench centre who rebounds well in
    fourteen minutes above a starter, because per-minute rate is exactly what
    he is good at. A star is not only efficient, he is efficient while the
    offence runs through him for thirty-four minutes a night - so the ladder is
    read off the mean of all six standardized features, impact and load
    together, and the tier that leads on both is the one called Star.
    """
    return centres.mean(axis=1)


def _top_cluster(centres: np.ndarray) -> int:
    """The cluster the ladder will call Star."""
    return int(np.argmax(_tier_scores(centres)))
