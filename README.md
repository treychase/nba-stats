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
`archetypes.py` clusters players on what they actually do — where they get the
ball, how they shoot, how much they create, how they rebound and defend — over
eighteen tracking features taken as per-36 rates so the clustering finds roles
rather than rediscovering the minutes rotation.

`k` is chosen by silhouette score rather than picked by hand, and each cluster
is named by matching its centre against prototype weight vectors ("Rim-running
big", "Primary creator", "Off-ball shooter" and so on). A cluster defined only
by what its players *don't* do falls through to "Low-usage role player" instead
of borrowing a name it hasn't earned. Players under the 250-minute floor are
left unclustered: their rate stats are mostly noise and would drag the centres
around.

The tab shows the player against their archetype's centre on the features they
differ from league average on most, the closest players to them in the same
feature space, and a table of every archetype with what defines it.

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
(`player_media.py`) and clustering (`archetypes.py`) are kept out of the
Streamlit script so they can be imported and tested without a Streamlit
runtime — which is the only way the awkward cases get covered: a player with no
touch tracking, a metric with too few attempts to qualify, a play type the
player has never run, a missing portrait. CI runs the same three targets on
Python 3.11 and 3.12.
