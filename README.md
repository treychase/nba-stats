# NBA Stats

Scouting tools over NBA tracking and shot chart data for the 2025-26 season.

## Scouting dashboard

```bash
make install
make run          # or: streamlit run dashboard.py
```

Pick a team and a player at the top and every tab scouts that player. The
selection is mirrored into the URL, so a link like
`?team=Denver+Nuggets&player=Nikola+Jokic` opens on that player — which is the
point of a scouting tool someone else has to be able to send you.

The header card carries the player's headshot, pulled straight from the NBA's
own CDN by player id, alongside their team, season counting stats and their
archetype. Portraits are stacked over an inline monogram of the player's
initials in their team colour, so a player the CDN has no portrait for — a
recent trade, a two-way signing, anyone the CDN has not caught up with — gets a
card the same shape as everyone else's rather than a broken image.

### Scouting

A hex bin shot chart, filterable by any of the shot type categories from
`add_shot_type_dummies` and colored by FG% or by volume, plus a shooting
profile: true shooting %, 3-point %, mid-range % and free throw %, each with
its league percentile.

### Touches & Pick and Roll

Shades the paint, post and elbow areas of the court by the share of the
player's touches taken in each, and lists their pick and roll points as a ball
handler and as a roll man. Percentiles for the two pick and roll roles are
computed within the role, so bigs are ranked against other roll men and guards
against other ball handlers.

### Archetypes

Position labels stopped describing how NBA players are used a while ago, so
`archetypes.py` clusters players on what they actually do, over fifteen
tracking features in three families:

| Family | Features |
| --- | --- |
| Field goal frequency and type | catch-and-shoot attempts and threes, pull-up attempts and threes, drive attempts, paint / post-up / elbow attempts |
| Passing | passes made, potential assists, assists per pass |
| Rebounding and defence | offensive and defensive rebounds, steals, blocks |

Deliberately no touch counts. A touch says a player had the ball; it does not
say what he did with it, so clustering on touches groups everyone who gets fed
regardless of whether they shoot, pass or draw a foul. Attempts split by type
and by where they come from say what a possession turns into. Everything is a
per-36 rate, so the clustering finds roles rather than rediscovering the
minutes rotation.

`k` is chosen by silhouette score rather than picked by hand, subject to one
product constraint: no archetype may hold more than a quarter of the league or
fewer than 3% of it. Unconstrained silhouette prefers four clusters here, one
of which is half of everybody — a true statement about the data and a useless
one to scout with. Each cluster is named by matching its centre against
prototype weight vectors, and a cluster defined only by what its players
*don't* do falls through to "Low-usage role player" instead of borrowing a name
it hasn't earned. Players under the 250-minute floor are left unclustered.

On the committed pull this lands on seven archetypes over 450 players:
post scorer, rim-running big, stretch big, primary creator, secondary
playmaker, off-ball shooter, defensive specialist.

The tab shows the player against their archetype's centre on the features they
differ from league average on most, the closest players to them in the same
feature space, and a table of every archetype with what defines it.

### Projected true shooting

What a player *shot* and what he is likely to shoot next are different
questions, and they diverge most for exactly the players a front office is
deciding about. `true_shooting.py` answers the second one in two steps.

**Reconstruct.** True shooting needs free throws and the committed pulls carry
no box score, but they carry enough to recover one exactly. Season points come
from the possessions pull and every made field goal comes from the shot chart,
so `FTM = POINTS − (2·FGM + FG3M)` is arithmetic rather than estimation — it
comes out non-negative for all 450 players with a shot chart. Attempts need a
free throw percentage, and the tracking pulls carry free throws for the fouls
they track (drives, paint, post and elbow touches): about half of a player's
trips, shot from the same line by the same player. That percentage is itself
pooled toward the league before use, so nine tracked makes don't buy a player a
perfect stroke.

**Pool.** Observed true shooting is a noisy read on talent, and how noisy
depends entirely on volume. The model is the standard hierarchical normal one,
which has a closed-form posterior mean — no sampler needed:

```
θ̂ᵢ = wᵢ·TSᵢ + (1 − wᵢ)·μ_g(i),    wᵢ = nᵢ / (nᵢ + k)
```

Two things make that more than a shrinkage formula. The group mean is the
player's **archetype** mean, not the league's, so a low-volume rim-running big
is pulled toward other rim-running bigs rather than toward an average that
includes pull-up guards. And `k` is **fitted, not chosen**: `fit_shrinkage()`
splits the season by date and finds the value that best predicts the second
half from the first, which is the only out-of-sample handle a single season
gives.

On the committed data that lands at k ≈ 140 attempts, and pooling cuts held-out
prediction error by 18% overall — 27% for players under 100 attempts, 12% for
players over 300. The gain is entirely concentrated where the sample is thin,
which is the point.

Percentile shading runs orange (below average) through near-white to purple
(above average) — deliberately not red-to-green, which is the one diverging
pair red-green colorblind readers can't split.

## Data

Data lives in `data/`. Field goal splits come from `nba_shot_chart_2025-26.csv`,
touch locations from `tracking_possessions_2025-26.csv`, and the archetype
features from `nba_tracking_combined_2025-26.csv`. Two sections need pulls that
aren't committed:

| Missing file | Pull it with | Affects |
| --- | --- | --- |
| `data/nba_player_box_2025-26.csv` | `pull_player_box_stats()` | True shooting %, free throw % |
| `data/nba_pick_and_roll_combined_2025-26.csv` | `pull_pick_and_roll()` | Pick and roll section |

Both live in `scraper_functions.py`. Until the files exist, those rows read `—`
and the app says which pull to run rather than raising.

## Development

```bash
make install   # dependencies, plus pytest
make test      # the test suite
make lint      # byte-compile every module
```

The dashboard's tables (`dashboard_tables.py`), media helpers
(`player_media.py`), clustering (`archetypes.py`) and shooting model
(`true_shooting.py`) are kept out of the
Streamlit script so they can be imported and tested without a Streamlit
runtime — which is the only way the awkward cases get covered: a player with no
touch tracking, a metric with too few attempts to qualify, a play type the
player has never run, a missing portrait. CI runs the same three targets on
Python 3.11 and 3.12.
