# NBA Stats

Master repo for nba stats analysis.

## Scouting dashboard

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

Pick a team and a player at the top; both tabs scout that player.

The **Scouting** tab draws their hex bin shot chart
(filterable by any of the shot type categories from `add_shot_type_dummies`,
colored by FG% or by volume), and shows their shooting profile — true
shooting %, 3-point %, mid-range % and free throw % — with each metric's
league percentile color coded red to green.

The **Touches & Pick and Roll** tab shades the paint, post and elbow areas of
the court by the share of the player's touches taken in each, and lists their
pick and roll points as a ball handler and as a roll man. Percentiles for the
two pick and roll roles are computed within the role, so bigs are ranked
against other roll men and guards against other ball handlers.

Data comes from `data/`. Field goal splits are read from
`nba_shot_chart_2025-26.csv` and touch locations from
`tracking_possessions_2025-26.csv`. Two sections need pulls that aren't
committed:

| Missing file | Pull it with | Affects |
| --- | --- | --- |
| `data/nba_player_box_2025-26.csv` | `pull_player_box_stats()` | True shooting %, free throw % |
| `data/nba_pick_and_roll_combined_2025-26.csv` | `pull_pick_and_roll()` | Pick and roll section |

Both live in `scraper_functions.py`. Until the files exist, those rows read
`—` and the tab says which pull to run.
