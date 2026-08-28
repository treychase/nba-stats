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

### Impact and stars

Two questions a shot chart cannot answer: what is this player worth per
possession, and how many wins is that. Both are box score metrics, and the
committed pulls carry no box score — so `advanced_metrics.py` reconstructs one
and builds both on top of it.

Everything except turnovers comes out of the pulls exactly. Points and minutes
from the possessions pull, field goals from the shot chart, free throws from
the difference between the two (the same arithmetic the true shooting section
uses), and rebounds, assists, steals, blocks and rim defence from the tracking
pulls. Summed over the league that lands on 115.3 points, 88.8 field goal
attempts, 32.2 defensive rebounds and 26.6 assists per team game — the season's
real rates, which is the check that nobody is being dropped or double counted.

**Turnovers are the one estimate.** The pulls count them on the plays they
track — drives, and paint, post and elbow touches — which is about a third of
them. The rest are charged to each player's untracked touches at one league
rate, set so the league total comes out at the 13.0% of possessions everybody
knows it to be. That is the same kind of constant as the 0.44 in true shooting
attempts, and it is a module constant: when `data/nba_player_box_2025-26.csv`
carries a `TOV` column the real number is used and the estimate is never
reached.

Both metrics are built from one **possession-value framework**. Every box event
is priced in points against what a possession is worth, and because those
prices are estimates rather than fitted weights — there is no play-by-play here
to regress against — they all sit in one table, `VALUES`, so a reader can
disagree with a number instead of with an expression. Scoring needs no price at
all: `2 · TSA · (TS − league TS)` is the points a player scored above what the
league would have scored on his own volume.

One term is worth pulling out. The tracking pull credits a shot at the rim to
whoever was nearest, and being nearest is an assignment rather than a choice —
a guard is the closest defender on a rim attempt mostly when he has already
been beaten. The league bears that out: defenders in the bottom fifth by rim
attempts allow 72%, the top fifth 60%. Scoring everyone against one pooled rate
would hand every centre free credit and charge every guard for playing on the
perimeter, so the baseline moves with volume — a player is measured against
what the league allows *at his own rim volume*. What is left is rim protection
rather than position.

Box plus/minus is those points per 100 possessions, centred so the league's
minute-weighted average is zero, which is what makes it a plus/minus rather
than a total. Win shares read the same events through Dean Oliver's framework
instead — offence against a replacement baseline, defence through stops — which
is why the two are worth showing together rather than one being a rescaling of
the other: they correlate at 0.89 and disagree about exactly the players you
would want them to. League average WS/48 is pinned at 0.100, which is not a
calibration choice but arithmetic: win shares sum to wins, teams average .500,
so a player who plays a fifth of his team's minutes earns a fifth of 41 wins.

What is missing is worth saying plainly. There is no opponent data in the
committed pulls, so every team is treated as league-average defensively — a
player is credited for the stops he makes and not for the defence around him.
There are no personal fouls either. And the shot chart only covers players over
50 attempts, so the fifth of the league below that comes back unranked rather
than guessed at.

#### Stars, as a clustering

The archetypes divide volume out on purpose, because their question is *what a
player does*. That makes them exactly the wrong tool for finding stars: a bench
guard and a franchise guard take the same shots at different volumes and land
in the same archetype, which is the archetype working correctly.

So stars get their own clustering, on the other axis. Same method, opposite
feature set — box plus/minus and win shares against points, shot volume, points
created and minutes **per game** rather than per 36. `k` comes off the
silhouette score subject to the top tier holding no more than 12% of the
league, and the tiers are named by where their centres rank rather than by
hand, so "star" is where the league separates instead of a number somebody
picked. On the committed pull that lands on five tiers and 27 stars out of 428
clustered players.

The two middle tiers separate on minutes as much as on quality: a bench big who
rebounds well in nineteen minutes rates ahead of a starter per possession and
behind him per night. The table lists what defines each tier for that reason.

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

Percentile shading runs blue (below average) through near-white to red (above
average), the same scale the shot chart already colours field goal percentage
on, so a percentile and a hot spot read the same way across the whole app. Still
deliberately not red-to-green, which is the one diverging pair red-green
colorblind readers can't split; red against blue separates on lightness as well
as hue, so it survives both common forms of colour blindness and a black and
white printout.

## Data

Data lives in `data/`. Field goal splits come from `nba_shot_chart_2025-26.csv`,
touch locations from `tracking_possessions_2025-26.csv`, and the archetype
features from `nba_tracking_combined_2025-26.csv`. Two sections need pulls that
aren't committed:

| Missing file | Pull it with | Affects |
| --- | --- | --- |
| `data/nba_player_box_2025-26.csv` | `pull_player_box_stats()` | True shooting %, free throw %, and real turnovers in place of the estimate |
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
(`player_media.py`), clustering (`archetypes.py`), shooting model
(`true_shooting.py`) and impact metrics (`advanced_metrics.py`) are kept out of
the Streamlit script so they can be imported and tested without a Streamlit
runtime — which is the only way the awkward cases get covered: a player with no
touch tracking, a metric with too few attempts to qualify, a play type the
player has never run, a missing portrait. CI runs the same three targets on
Python 3.11 and 3.12.
