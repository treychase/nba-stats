# NBA Stats

Master repo for nba stats analysis.

## Scouting dashboard

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

The Scouting tab takes a team and a player, draws their hex bin shot chart
(filterable by any of the shot type categories from `add_shot_type_dummies`,
colored by FG% or by volume), and shows their shooting profile — true
shooting %, 3-point %, mid-range % and free throw % — with each metric's
league percentile color coded red to green.

Data comes from `data/`. Field goal splits are read from
`nba_shot_chart_2025-26.csv`. True shooting % and free throw % additionally
need season box score totals, which the shot chart feed doesn't carry — run
`pull_player_box_stats()` from `scraper_functions.py` and save the result as
`data/nba_player_box_2025-26.csv`. Those two rows read `—` until it exists.
