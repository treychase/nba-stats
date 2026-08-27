"""True shooting, reconstructed and then partially pooled toward the league.

The question a front office actually asks is not what a player *shot*, it is
what he is likely to shoot next. Those differ most for the players a team is
deciding about: a rotation piece with 180 attempts who shot 64% is mostly
telling you about variance, and taking that number at face value is how a team
talks itself into a contract.

This module answers it in two steps.

**Reconstruct.** True shooting needs free throws, and the committed pulls do
not carry a box score. They do carry enough to recover one exactly:
:data:`~processing_functions` aside, season points come from the possessions
tracking pull and every made field goal comes from the shot chart, so

    FTM = POINTS - (2 * FGM + FG3M)

is arithmetic, not estimation - it comes out non-negative for all 450 players
with a shot chart. Attempts need a free throw percentage, and the tracking
pulls carry free throws for the fouls they track (drives, paint, post and elbow
touches) - about half of a player's trips, but shot from the same line by the
same player, so the percentage carries over. That percentage is itself pooled
toward the league before it is used, for exactly the reason the rest of the
module exists: a player with nine tracked free throws should not be credited
with a 100% stroke.

**Pool.** Observed true shooting is a noisy read on a player's talent, and how
noisy depends entirely on volume. The model is the standard hierarchical
normal one:

    theta_i  ~ Normal(mu_g(i), tau^2)          talent, within archetype g
    TS_i     ~ Normal(theta_i, sigma^2 / n_i)  what we observed, on n_i attempts

which has a closed form posterior mean - no sampler needed:

    theta_hat_i = w_i * TS_i + (1 - w_i) * mu_g(i),   w_i = n_i / (n_i + k)

with ``k = sigma^2 / tau^2`` the number of attempts at which a player's own
record and his group's mean count equally. A 900-attempt starter keeps almost
all of his own number; a 120-attempt reserve is written mostly in terms of the
players he resembles.

Two things make that more than a shrinkage formula. The group mean is the
player's *archetype* mean, not the league's, so a low volume rim-running big is
pulled toward other rim-running bigs rather than toward a league average that
includes pull-up guards - see :mod:`archetypes`. And ``k`` is estimated rather
than picked: :func:`fit_shrinkage` splits the season by date and finds the
value that best predicts the second half from the first, which is the only
out-of-sample handle a single season gives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "FT_PRIOR_ATTEMPTS",
    "MIN_ATTEMPTS",
    "TRACKED_FT_COLUMNS",
    "ShootingProjection",
    "build_shooting_panel",
    "fit_shrinkage",
    "project_true_shooting",
    "split_half_frames",
]

# Free throw pairs the tracking pulls carry. Not every trip to the line, but
# every one of them is a real free throw taken by that player.
TRACKED_FT_COLUMNS = [
    ("DRIVE_FTM", "DRIVE_FTA"),
    ("ELBOW_TOUCH_FTM", "ELBOW_TOUCH_FTA"),
    ("POST_TOUCH_FTM", "POST_TOUCH_FTA"),
    ("PAINT_TOUCH_FTM", "PAINT_TOUCH_FTA"),
]

# Fallback prior strength for the free throw percentage, in free throws, used
# only if the method of moments fit degenerates.
FT_PRIOR_ATTEMPTS = 40.0

# Below this many true shooting attempts a player is reported but not used to
# fit the league and archetype means, so the pool is not set by the noise.
MIN_ATTEMPTS = 150.0

# The 0.44 in TSA = FGA + 0.44 * FTA is the league's long run estimate of how
# many trips to the line end a possession, since and-ones and technicals do not.
FT_TRIP_WEIGHT = 0.44


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def _beta_binomial_prior(makes: pd.Series, attempts: pd.Series) -> tuple[float, float]:
    """League free throw rate and prior strength, by method of moments.

    Fitting the prior strength rather than assuming one matters: it is what
    decides how far a player with a handful of tracked free throws is moved,
    and the spread of free throw shooting across the league is the only thing
    that should decide that.
    """
    usable = attempts >= 10
    if usable.sum() < 20:
        return float(makes.sum() / max(attempts.sum(), 1)), FT_PRIOR_ATTEMPTS
    rate = float(makes[usable].sum() / attempts[usable].sum())
    observed = (makes[usable] / attempts[usable]).astype(float)
    weights = attempts[usable].astype(float)
    # Total variance across players, minus the binomial noise each carries,
    # leaves the spread in true free throw ability.
    total = float(np.average((observed - rate) ** 2, weights=weights))
    noise = float(rate * (1 - rate) * np.average(1 / weights, weights=weights))
    spread = total - noise
    if spread <= 1e-6:
        return rate, FT_PRIOR_ATTEMPTS
    return rate, float(max(5.0, rate * (1 - rate) / spread))


def _field_goals(shots: pd.DataFrame) -> pd.DataFrame:
    """FGA, FGM, FG3M and the points they are worth, per player."""
    made = shots["SHOT_MADE_FLAG"].astype(float)
    three = shots["SHOT_TYPE"].astype(str).str.startswith("3")
    frame = pd.DataFrame({
        "PLAYER_ID": shots["PLAYER_ID"].to_numpy(),
        "made": made.to_numpy(),
        "three_made": (made * three).to_numpy(),
    })
    grouped = frame.groupby("PLAYER_ID")
    out = pd.DataFrame({
        "FGA": grouped.size(),
        "FGM": grouped["made"].sum(),
        "FG3M": grouped["three_made"].sum(),
    })
    out["FG_PTS"] = 2 * out["FGM"] + out["FG3M"]
    return out


def build_shooting_panel(shots: pd.DataFrame,
                         possessions: pd.DataFrame,
                         tracking: pd.DataFrame) -> pd.DataFrame:
    """One row per player: attempts, points, and reconstructed true shooting.

    Raises ``ValueError`` if the reconstruction implies negative free throws
    for anyone, which would mean the shot chart and the possessions pull
    disagree about who scored what and nothing downstream should be trusted.
    """
    goals = _field_goals(shots)
    people = possessions.set_index("PLAYER_ID")[["PLAYER_NAME", "TEAM_ABBREVIATION",
                                                 "GP", "MIN", "POINTS"]]

    tracked = pd.DataFrame(index=tracking["PLAYER_ID"].to_numpy())
    tracked["FT_MADE_TRACKED"] = sum(
        tracking[made].fillna(0).to_numpy() for made, _ in TRACKED_FT_COLUMNS)
    tracked["FT_ATT_TRACKED"] = sum(
        tracking[att].fillna(0).to_numpy() for _, att in TRACKED_FT_COLUMNS)
    tracked.index.name = "PLAYER_ID"

    panel = goals.join(people, how="inner").join(tracked, how="left")
    panel[["FT_MADE_TRACKED", "FT_ATT_TRACKED"]] = (
        panel[["FT_MADE_TRACKED", "FT_ATT_TRACKED"]].fillna(0.0))

    panel["FTM"] = panel["POINTS"] - panel["FG_PTS"]
    negative = panel.index[panel["FTM"] < -1e-9]
    if len(negative):
        raise ValueError(
            "field goal points exceed total points for "
            f"{len(negative)} player(s), e.g. {list(negative[:3])}; the shot "
            "chart and possessions pulls are not from the same season")

    rate, strength = _beta_binomial_prior(panel["FT_MADE_TRACKED"], panel["FT_ATT_TRACKED"])
    panel["FT_PCT"] = ((panel["FT_MADE_TRACKED"] + strength * rate)
                       / (panel["FT_ATT_TRACKED"] + strength))
    panel["FTA"] = panel["FTM"] / panel["FT_PCT"]
    panel["TSA"] = panel["FGA"] + FT_TRIP_WEIGHT * panel["FTA"]
    panel["TS_PCT"] = panel["POINTS"] / (2 * panel["TSA"])
    panel.attrs["ft_league_rate"] = rate
    panel.attrs["ft_prior_attempts"] = strength
    return panel.reset_index().rename(columns={"index": "PLAYER_ID"})


# ---------------------------------------------------------------------------
# Calibrating the pool
# ---------------------------------------------------------------------------

def split_half_frames(shots: pd.DataFrame, cut: str | pd.Timestamp | None = None
                      ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split the shot chart by date into an earlier and a later half.

    The split is on the field goal record alone, which is the part of scoring
    the shot chart observes game by game. Free throws are only in the season
    totals, so a date split cannot see them - which is why ``k`` is calibrated
    on points per field goal attempt and then applied to true shooting, two
    quantities with the same shape of noise.
    """
    dates = pd.to_datetime(shots["GAME_DATE"], format="%Y%m%d", errors="coerce")
    if dates.isna().all():
        raise ValueError("no parseable GAME_DATE in the shot chart")
    cut = pd.Timestamp(cut) if cut is not None else dates.quantile(0.5)
    return shots.loc[dates < cut], shots.loc[dates >= cut], cut


def _points_per_attempt(shots: pd.DataFrame) -> pd.DataFrame:
    goals = _field_goals(shots)
    goals["PPA"] = goals["FG_PTS"] / goals["FGA"]
    return goals


def fit_shrinkage(shots: pd.DataFrame,
                  cut: str | pd.Timestamp | None = None,
                  min_early: int = 10,
                  min_late: int = 30,
                  grid: np.ndarray | None = None) -> dict:
    """Choose ``k`` by asking which value best predicts the rest of the season.

    For each candidate ``k``, every player's first half rate is pulled toward
    the league mean by ``n / (n + k)`` and scored against what he actually shot
    afterwards. The winner is the ``k`` with the lowest held out error - so the
    amount of pooling is a measurement, not a preference.
    """
    early, late, cut = split_half_frames(shots, cut)
    before, after = _points_per_attempt(early), _points_per_attempt(late)
    paired = before.join(after, lsuffix="_early", rsuffix="_late", how="inner")
    paired = paired.loc[(paired["FGA_early"] >= min_early)
                        & (paired["FGA_late"] >= min_late)]
    if len(paired) < 30:
        raise ValueError(f"only {len(paired)} players clear the split-half floors")

    league = float(before["FG_PTS"].sum() / before["FGA"].sum())
    grid = np.arange(0.0, 801.0, 5.0) if grid is None else np.asarray(grid, dtype=float)
    attempts = paired["FGA_early"].to_numpy(dtype=float)
    observed = paired["PPA_early"].to_numpy(dtype=float)
    actual = paired["PPA_late"].to_numpy(dtype=float)

    errors = np.array([
        np.sqrt(np.mean((((attempts * observed + k * league) / (attempts + k)) - actual) ** 2))
        for k in grid
    ])
    best = int(np.argmin(errors))
    return {
        "k": float(grid[best]),
        "rmse": float(errors[best]),
        "rmse_unpooled": float(errors[0] if grid[0] == 0 else
                               np.sqrt(np.mean((observed - actual) ** 2))),
        "rmse_fully_pooled": float(np.sqrt(np.mean((league - actual) ** 2))),
        "league_rate": league,
        "cut": cut,
        "n_players": int(len(paired)),
        "curve": [[float(g), float(e)] for g, e in zip(grid, errors)],
        "players": paired.reset_index()[["PLAYER_ID", "FGA_early", "PPA_early",
                                         "FGA_late", "PPA_late"]],
    }


# ---------------------------------------------------------------------------
# The projection
# ---------------------------------------------------------------------------

@dataclass
class ShootingProjection:
    """Projected true shooting for every player, plus how it was arrived at."""

    table: pd.DataFrame            # a row per player: observed, projected, weight
    k: float                       # attempts at which own record and group tie
    league: float                  # league true shooting
    groups: pd.DataFrame           # per archetype: mean, players, attempts
    tau: float = 0.0               # spread of true talent, once noise is removed
    calibration: dict = field(default_factory=dict)

    def of(self, player_id: int) -> pd.Series | None:
        match = self.table.loc[self.table["PLAYER_ID"] == player_id]
        return None if match.empty else match.iloc[0]


def project_true_shooting(panel: pd.DataFrame,
                          k: float,
                          groups: pd.Series | None = None,
                          min_attempts: float = MIN_ATTEMPTS,
                          calibration: dict | None = None) -> ShootingProjection:
    """Pool every player's true shooting toward his archetype and the league.

    ``groups`` maps PLAYER_ID to an archetype name; players without one are
    pooled toward the league instead. Group means are computed only from
    players over ``min_attempts``, so a group's centre is not itself set by the
    small samples that are about to be pulled toward it.
    """
    table = panel.copy()
    if groups is None:
        table["ARCHETYPE"] = None
    else:
        table["ARCHETYPE"] = table["PLAYER_ID"].map(groups)

    anchored = table.loc[table["TSA"] >= min_attempts]
    if anchored.empty:
        raise ValueError(f"no players clear {min_attempts:.0f} true shooting attempts")
    league = float(anchored["POINTS"].sum() / (2 * anchored["TSA"].sum()))

    # An archetype's mean is itself pooled toward the league, so a twelve player
    # archetype does not get to define its own centre from twelve numbers.
    rows = []
    means = {}
    for name, members in anchored.groupby("ARCHETYPE"):
        attempts = float(members["TSA"].sum())
        raw = float(members["POINTS"].sum() / (2 * attempts))
        weight = attempts / (attempts + k)
        mean = weight * raw + (1 - weight) * league
        means[name] = mean
        rows.append({"ARCHETYPE": name, "PLAYERS": int(len(members)),
                     "TSA": attempts, "RAW_TS": raw, "GROUP_TS": mean})
    # Without archetypes there are no groups, and the whole league is the pool.
    # The frame still has to carry its columns so callers can render it empty.
    group_table = pd.DataFrame(
        rows, columns=["ARCHETYPE", "PLAYERS", "TSA", "RAW_TS", "GROUP_TS"])
    if not group_table.empty:
        group_table = group_table.sort_values("GROUP_TS", ascending=False)

    target = table["ARCHETYPE"].map(means).astype(float)
    table["GROUP_TS"] = target.fillna(league)
    table["WEIGHT"] = table["TSA"] / (table["TSA"] + k)
    table["PROJECTED_TS"] = (table["WEIGHT"] * table["TS_PCT"]
                             + (1 - table["WEIGHT"]) * table["GROUP_TS"])
    table["SHIFT"] = table["PROJECTED_TS"] - table["TS_PCT"]

    # Posterior standard deviation, tau * sqrt(1 - w). The spread of observed
    # true shooting across the league is not tau: it is tau plus each player's
    # own sampling noise. Since sigma^2 = k * tau^2 by construction, the two
    # separate cleanly -
    #     Var(observed) = tau^2 * (1 + k * E[1/n])
    # - and taking the raw spread instead would make every interval too wide,
    # worst for exactly the small-sample players the intervals exist for.
    observed_var = float(np.average((anchored["TS_PCT"] - league) ** 2,
                                    weights=anchored["TSA"]))
    inflation = 1.0 + k * float(np.average(1.0 / anchored["TSA"],
                                           weights=anchored["TSA"]))
    tau = float(np.sqrt(max(observed_var / inflation, 1e-12)))
    table["SE"] = tau * np.sqrt(k / (table["TSA"] + k))

    table = table.sort_values("PROJECTED_TS", ascending=False).reset_index(drop=True)
    return ShootingProjection(table=table, k=float(k), league=league,
                              groups=group_table, tau=tau,
                              calibration=calibration or {})
