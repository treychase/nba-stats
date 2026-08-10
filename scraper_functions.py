"""
NBA Player Tracking + Pick-and-Roll Data Pull
------------------------------------------------
Combines two data pulls for the current season into one script:

1. pull_tracking_stats()   -> all player tracking measure types
                               (Drives, Defense, Catch & Shoot, Passing,
                               Speed & Distance, touches, etc.)
2. pull_pick_and_roll()    -> Synergy play-type stats for pick-and-roll
                               ball handlers and roll men (offense + defense)

Run main() to pull everything and write all CSVs, or import and call
either function independently.

Requires: pip install nba_api pandas
"""

import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from nba_api.stats.endpoints import (
    leaguedashptstats,
    synergyplaytypes,
    shotchartdetail,
    commonallplayers,
    leaguedashplayerstats,
)

SEASON = "2025-26"
SEASON_TYPE = "Regular Season"
REQUEST_DELAY = 1.5  # seconds between calls, to avoid rate limiting

# ---------------------------------------------------------------------------
# Tracking stats
# ---------------------------------------------------------------------------

PT_MEASURE_TYPES = [
    "Drives",
    "Defense",
    "CatchShoot",
    "Passing",
    "Possessions",
    "PullUpShot",
    "Rebounding",
    "SpeedDistance",
    "ElbowTouch",
    "PostTouch",
    "PaintTouch",
]


def _pull_tracking_measure(measure_type: str, player_or_team: str = "Player") -> pd.DataFrame:
    """Pull a single tracking measure type and return as a DataFrame."""
    response = leaguedashptstats.LeagueDashPtStats(
        season=SEASON,
        season_type_all_star=SEASON_TYPE,
        player_or_team=player_or_team,
        pt_measure_type=measure_type,
    )
    df = response.get_data_frames()[0]
    df["PT_MEASURE_TYPE"] = measure_type
    return df


def pull_tracking_stats() -> pd.DataFrame:
    """Pull all tracking measure types, save individual + combined CSVs, return combined DataFrame."""
    base_cols = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION"]
    all_frames = {}

    for measure_type in PT_MEASURE_TYPES:
        print(f"Pulling {measure_type}...")
        try:
            df = _pull_tracking_measure(measure_type)
            all_frames[measure_type] = df
            df.to_csv(f"tracking_{measure_type.lower()}_{SEASON}.csv", index=False)
            print(f"  -> {len(df)} rows")
        except Exception as e:
            print(f"  FAILED: {measure_type} ({e})")

        time.sleep(REQUEST_DELAY)

    print("\nMerging all tracking categories on PLAYER_ID...")
    merged = None
    for measure_type, df in all_frames.items():
        if merged is None:
            merged = df[base_cols + [c for c in df.columns if c not in base_cols]]
        else:
            renamed = df.drop(columns=[c for c in base_cols if c != "PLAYER_ID" and c in df.columns])
            merged = merged.merge(renamed, on="PLAYER_ID", how="outer", suffixes=("", f"_{measure_type}"))

    if merged is not None:
        out_path = f"nba_tracking_combined_{SEASON}.csv"
        merged.to_csv(out_path, index=False)
        print(f"Combined tracking file saved: {out_path} ({len(merged)} rows, {len(merged.columns)} columns)")

    return merged


# ---------------------------------------------------------------------------
# Pick-and-roll play types
# ---------------------------------------------------------------------------

PLAY_TYPES = {
    "PRBallHandler": "pnr_ball_handler",
    "PRRollman": "pnr_roll_man",
}

TYPE_GROUPINGS = ["offensive", "defensive"]


def _pull_play_type(play_type: str, type_grouping: str) -> pd.DataFrame:
    """Pull a single play type / grouping combination and return as a DataFrame."""
    response = synergyplaytypes.SynergyPlayTypes(
        season=SEASON,
        season_type_all_star=SEASON_TYPE,
        play_type_nullable=play_type,
        type_grouping_nullable=type_grouping,
        player_or_team_abbreviation="P",
    )
    df = response.get_data_frames()[0]
    df["PLAY_TYPE"] = play_type
    df["TYPE_GROUPING"] = type_grouping
    return df


def pull_pick_and_roll() -> pd.DataFrame:
    """Pull pick-and-roll ball handler + roll man stats, save individual + combined CSVs, return combined DataFrame."""
    all_frames = []

    for play_type, label in PLAY_TYPES.items():
        for grouping in TYPE_GROUPINGS:
            print(f"Pulling {play_type} ({grouping})...")
            try:
                df = _pull_play_type(play_type, grouping)
                out_path = f"{label}_{grouping}_{SEASON}.csv"
                df.to_csv(out_path, index=False)
                all_frames.append(df)
                print(f"  -> {len(df)} rows, saved to {out_path}")
            except Exception as e:
                print(f"  FAILED: {play_type} ({grouping}) ({e})")

            time.sleep(REQUEST_DELAY)

    combined = None
    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        out_path = f"nba_pick_and_roll_combined_{SEASON}.csv"
        combined.to_csv(out_path, index=False)
        print(f"Combined pick-and-roll file saved: {out_path} ({len(combined)} rows)")

    return combined


# ---------------------------------------------------------------------------
# Shot chart (per-shot location, make/miss, zone, distance)
# ---------------------------------------------------------------------------
# Note: shot clock and per-shot defending player are NOT available here.
# That level of detail was part of the original SportVU raw tracking feed,
# which the NBA stopped publishing publicly around 2016-17. This pulls
# everything shotchartdetail exposes: LOC_X/LOC_Y, SHOT_MADE_FLAG,
# SHOT_ZONE_BASIC/AREA, SHOT_DISTANCE, ACTION_TYPE, period/time context, etc.

def _get_active_player_ids() -> list:
    """Return PLAYER_ID for all players active in the current season."""
    response = commonallplayers.CommonAllPlayers(
        is_only_current_season=1,
        season=SEASON,
    )
    df = response.get_data_frames()[0]
    return df["PERSON_ID"].tolist()


def _get_player_ids_with_min_shots(min_shots: int = 50) -> list:
    """
    Return PLAYER_ID for players whose season field goal attempts (FGA)
    exceed min_shots. Used to skip players with too few shots to be worth
    pulling a full shot chart for (bench/two-way players, etc.).
    """
    response = leaguedashplayerstats.LeagueDashPlayerStats(
        season=SEASON,
        season_type_all_star=SEASON_TYPE,
        measure_type_detailed_defense="Base",
        per_mode_detailed="Totals",
    )
    df = response.get_data_frames()[0]
    filtered = df[df["FGA"] > min_shots]
    return filtered["PLAYER_ID"].tolist()


def _pull_player_shot_chart(player_id: int) -> pd.DataFrame:
    """Pull shot chart detail for a single player for the season."""
    response = shotchartdetail.ShotChartDetail(
        team_id=0,
        player_id=player_id,
        season_nullable=SEASON,
        season_type_all_star=SEASON_TYPE,
        context_measure_simple="FGA",
    )
    df = response.get_data_frames()[0]
    df["SEASON"] = SEASON
    return df


def pull_shot_chart(min_shots: int = 50, max_workers: int = 8) -> pd.DataFrame:
    """
    Pull per-shot chart data in parallel for players with more than
    min_shots field goal attempts on the season, save combined CSV,
    return DataFrame.

    Parameters
    ----------
    min_shots : int
        Minimum season FGA required for a player to be included. Filters
        out low-volume/bench players before hitting the API.
    max_workers : int
        Number of concurrent threads to use for shot chart requests.
        Since this is I/O-bound (waiting on stats.nba.com), threading
        gives a real speedup despite the GIL. Push much above 8-10 and
        you risk getting rate-limited or temporarily blocked.
    """
    print(f"Fetching players with FGA > {min_shots}...")
    player_ids = _get_player_ids_with_min_shots(min_shots)
    print(f"  -> {len(player_ids)} players qualify (out of full league)")

    all_frames = []
    completed = 0
    total = len(player_ids)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_player = {
            executor.submit(_pull_player_shot_chart, player_id): player_id
            for player_id in player_ids
        }

        for future in as_completed(future_to_player):
            player_id = future_to_player[future]
            completed += 1
            try:
                df = future.result()
                if not df.empty:
                    all_frames.append(df)
                print(f"  ({completed}/{total}) player {player_id}: {len(df)} shots")
            except Exception as e:
                print(f"  ({completed}/{total}) FAILED: player {player_id} ({e})")

    combined = None
    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        out_path = f"nba_shot_chart_{SEASON}.csv"
        combined.to_csv(out_path, index=False)
        print(f"Combined shot chart file saved: {out_path} ({len(combined)} rows)")

    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Pulling player tracking stats ===")
    pull_tracking_stats()

    print("\n=== Pulling pick-and-roll play-type stats ===")
    pull_pick_and_roll()

    print("\n=== Pulling shot chart data ===")
    pull_shot_chart()


if __name__ == "__main__":
    main()