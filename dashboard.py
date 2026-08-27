"""
NBA Scouting Dashboard
------------------------
Streamlit app for scouting a single player.

Pick a team and a player at the top. The card that appears carries their
headshot, their team, and their season counting stats; both tabs below then
scout that player.

Scouting tab: a hex bin shot chart that can be sliced by shot type, next
to their shooting splits with league percentiles.

Touches & Pick and Roll tab: where on the court the player gets the ball,
and what they score out of the pick and roll as a ball handler and as a
roll man, all against league percentiles.

Run with: streamlit run dashboard.py

Requires: pip install -r requirements.txt

The team and player are mirrored into the URL, so a scouting view can be
pasted to someone else and it opens on the same player.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from advanced_metrics import (
    AVERAGE_WS48,
    STAR_TIER,
    build_box_panel,
    build_tier_features,
    fit_star_tiers,
    impact_metrics,
)
from archetypes import MIN_MINUTES, build_archetype_features, fit_archetypes
from true_shooting import (
    build_shooting_panel,
    fit_shrinkage,
    project_true_shooting,
)
from dashboard_tables import (
    PERCENTILE_COL,
    impact_profile,
    percentile_color,
    player_summary,
    pnr_leaderboard,
    pnr_profile,
    shooting_profile,
    star_leaderboard,
    star_tier_table,
    style_impact,
    style_table,
    touch_profile,
)
from player_media import headshot_html, team_color
from plot_functions import plot_shot_hexbin, plot_touch_areas
from processing_functions import (
    PNR_ROLES,
    SHOT_TYPE_LABELS,
    TOUCH_AREAS,
    add_shot_type_column,
    build_pick_and_roll_profile,
    build_shooting_splits,
    build_touch_profile,
)

SEASON = "2025-26"
DATA_DIR = Path(__file__).parent / "data"
SHOT_CHART_PATH = DATA_DIR / f"nba_shot_chart_{SEASON}.csv"
TRACKING_PATH = DATA_DIR / f"nba_tracking_combined_{SEASON}.csv"
BOX_STATS_PATH = DATA_DIR / f"nba_player_box_{SEASON}.csv"
POSSESSIONS_PATH = DATA_DIR / f"tracking_possessions_{SEASON}.csv"
PNR_PATH = DATA_DIR / f"nba_pick_and_roll_combined_{SEASON}.csv"

# add_shot_type_column's default new_col would overwrite shotchartdetail's own
# SHOT_TYPE column (the "2PT Field Goal" / "3PT Field Goal" label), which the
# shooting splits rely on, so the classification lands in its own column.
SHOT_TYPE_GROUP_COL = "SHOT_TYPE_GROUP"


# ---------------------------------------------------------------------------
# Data loading
#
# The two required files are checked before anything is read, so a fresh clone
# without them gets a sentence saying which pull to run instead of a traceback
# in the middle of the page.
# ---------------------------------------------------------------------------

REQUIRED = {
    SHOT_CHART_PATH: "pull_shot_chart()",
    POSSESSIONS_PATH: "pull_tracking_stats()",
}


def require_data() -> None:
    """Stop with an explanation if a file the whole app depends on is absent."""
    missing = [(path, call) for path, call in REQUIRED.items() if not path.exists()]
    if not missing:
        return
    lines = "\n".join(
        f"- `data/{path.name}` — run `{call}` from `scraper_functions.py`"
        for path, call in missing
    )
    st.error(f"The dashboard needs data that is not in `data/` yet:\n\n{lines}")
    st.stop()


@st.cache_data(show_spinner="Loading shots…")
def load_shots() -> pd.DataFrame:
    """Load per-shot data and tag each shot with its shot type category."""
    shots = pd.read_csv(SHOT_CHART_PATH)
    return add_shot_type_column(shots, new_col=SHOT_TYPE_GROUP_COL)


@st.cache_data
def load_box_stats() -> pd.DataFrame | None:
    """Load season box score totals, or None if they haven't been pulled yet."""
    if not BOX_STATS_PATH.exists():
        return None
    return pd.read_csv(BOX_STATS_PATH)


@st.cache_data(show_spinner="Building shooting splits…")
def load_splits() -> pd.DataFrame:
    """Build the league-wide shooting profile dataset with percentiles."""
    return build_shooting_splits(load_shots(), box=load_box_stats())


@st.cache_data
def load_possessions() -> pd.DataFrame:
    """Raw tracking possessions: the source of the header card's counting stats."""
    return pd.read_csv(POSSESSIONS_PATH)


@st.cache_data
def load_touches() -> pd.DataFrame:
    """Build the league-wide touch location dataset with percentiles."""
    return build_touch_profile(load_possessions())


@st.cache_data(show_spinner="Clustering archetypes…")
def load_archetypes():
    """Cluster the league into archetypes, or None if tracking is not pulled."""
    if not TRACKING_PATH.exists():
        return None
    features = build_archetype_features(pd.read_csv(TRACKING_PATH))
    if features.empty:
        return None
    return fit_archetypes(features)


@st.cache_data(show_spinner="Projecting true shooting…")
def load_projection():
    """Reconstruct true shooting and pool it, or None if a pull is missing.

    Cached as one unit because the three steps share the same inputs and the
    shrinkage constant has to be fitted before the projection can use it.
    """
    if not (TRACKING_PATH.exists() and POSSESSIONS_PATH.exists()):
        return None
    tracking = pd.read_csv(TRACKING_PATH)
    panel = build_shooting_panel(load_shots(), load_possessions(), tracking)
    calibration = fit_shrinkage(load_shots())
    model = load_archetypes()
    groups = (model.table.set_index("PLAYER_ID")["ARCHETYPE"]
              if model is not None else None)
    return project_true_shooting(panel, k=calibration["k"], groups=groups,
                                 calibration=calibration)


@st.cache_data(show_spinner="Reconstructing the box score…")
def load_impact():
    """Box plus/minus and win shares per 48, or None if a pull is missing.

    Cached as one unit with the star tiers below it because the clustering is
    over the metrics: fitting them separately would reconstruct the same box
    score twice.
    """
    if not (TRACKING_PATH.exists() and POSSESSIONS_PATH.exists()):
        return None
    panel = build_box_panel(load_shots(), load_possessions(),
                            pd.read_csv(TRACKING_PATH), box=load_box_stats())
    return impact_metrics(panel)


@st.cache_data(show_spinner="Finding the stars…")
def load_star_tiers():
    """Cluster the league into impact tiers, or None if the metrics are absent."""
    model = load_impact()
    if model is None:
        return None
    features = build_tier_features(model.table)
    if features.empty:
        return None
    return fit_star_tiers(features)


@st.cache_data
def load_pick_and_roll() -> pd.DataFrame | None:
    """Build the league-wide pick and roll dataset, or None if not pulled yet."""
    if not PNR_PATH.exists():
        return None
    return build_pick_and_roll_profile(pd.read_csv(PNR_PATH))


def player_tricode(possessions: pd.DataFrame, player_id: int) -> str | None:
    """The player's team tricode, used for the card's colour and monogram."""
    match = possessions.loc[possessions["PLAYER_ID"] == player_id]
    if match.empty or "TEAM_ABBREVIATION" not in match.columns:
        return None
    value = match["TEAM_ABBREVIATION"].iloc[0]
    return None if pd.isna(value) else str(value)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="NBA Scouting", page_icon="🏀", layout="wide")

require_data()

shots = load_shots()
splits = load_splits()
possessions = load_possessions()
touches = load_touches()
pnr = load_pick_and_roll()
archetypes = load_archetypes()
impact = load_impact()
tiers = load_star_tiers()

teams = sorted(splits["TEAM_NAME"].dropna().unique())

# The URL carries the current selection, so a view can be shared as a link.
params = st.query_params
default_team = params.get("team")
team_index = teams.index(default_team) if default_team in teams else 0

team_col, player_col, _spacer = st.columns([1, 1, 2])

with team_col:
    team = st.selectbox("Team", teams, index=team_index)

roster = splits.loc[splits["TEAM_NAME"] == team].sort_values("PLAYER_NAME")
names = roster["PLAYER_NAME"].tolist()

if not names:
    st.warning(f"No players with shot data for {team}.")
    st.stop()

default_player = params.get("player")
player_index = names.index(default_player) if default_player in names else 0

with player_col:
    player_name = st.selectbox("Player", names, index=player_index)

player_id = int(roster.loc[roster["PLAYER_NAME"] == player_name, "PLAYER_ID"].iloc[0])
tricode = player_tricode(possessions, player_id)

if params.get("team") != team or params.get("player") != player_name:
    st.query_params.update({"team": team, "player": player_name})

# ---------------------------------------------------------------------------
# Player card
# ---------------------------------------------------------------------------

summary = player_summary(possessions, player_id)
accent = team_color(tricode)

card_left, card_right = st.columns([1, 3])

with card_left:
    st.markdown(headshot_html(player_id, player_name, tricode, height=150),
                unsafe_allow_html=True)

with card_right:
    st.markdown(
        f"<div style='border-left:5px solid {accent};padding-left:14px'>"
        f"<div style='font-size:1.6rem;font-weight:700;line-height:1.2'>{player_name}</div>"
        f"<div style='opacity:.75'>{team}"
        + (f" · {tricode}" if tricode else "")
        + f" · {SEASON}</div></div>",
        unsafe_allow_html=True,
    )
    # Two chips, deliberately: the archetype says how a player is used, the
    # tier says how much he is worth. They are different questions and the
    # answer to one does not imply the other.
    archetype = archetypes.label_of(player_id) if archetypes else None
    tier = tiers.tier_of(player_id) if tiers else None
    chips = []
    if archetype:
        chips.append((archetype, accent, "#fff"))
    if tier:
        # The star tier is the only one that gets to shout.
        star = tier == STAR_TIER
        chips.append((tier, "#B2182B" if star else "transparent",
                      "#fff" if star else "inherit"))
    if chips:
        st.markdown(
            "".join(
                f"<span style='display:inline-block;margin-top:8px;margin-right:8px;"
                f"padding:4px 12px;border-radius:999px;background:{background};"
                f"color:{text};font-size:.85rem;font-weight:600;"
                f"border:1px solid {accent if background == 'transparent' else background}'>"
                f"{label}</span>"
                for label, background, text in chips
            ),
            unsafe_allow_html=True,
        )

    if summary:
        stat_cols = st.columns(len(summary))
        for column, (label, value) in zip(stat_cols, summary.items()):
            column.metric(label, f"{value:,.0f}")
    else:
        st.caption("No tracking row for this player, so counting stats are unavailable.")

st.divider()

scouting_tab, touches_tab, impact_tab, archetype_tab, shooting_tab = st.tabs(
    ["Scouting", "Touches & Pick and Roll", "Impact & stars", "Archetypes",
     "Projected true shooting"]
)

with scouting_tab:
    player_shots = shots.loc[shots["PLAYER_ID"] == player_id]

    # Shot type options come straight from the dummy variable categories, in
    # the same priority order, showing only the types this player actually has.
    counts = player_shots[SHOT_TYPE_GROUP_COL].value_counts()
    option_to_key = {
        f"{SHOT_TYPE_LABELS[key]} ({counts[key]})": key
        for key in SHOT_TYPE_LABELS
        if key in counts.index
    }
    options = list(option_to_key)

    selected = st.multiselect("Shot types", options, default=options)
    selected_keys = [option_to_key[option] for option in selected]
    chart_shots = player_shots.loc[player_shots[SHOT_TYPE_GROUP_COL].isin(selected_keys)]

    st.divider()

    chart_col, profile_col = st.columns([3, 2])

    with chart_col:
        color_by = st.radio(
            "Color hexagons by", ["FG%", "Volume"], horizontal=True, label_visibility="collapsed"
        )

        if chart_shots.empty:
            st.info("Select at least one shot type to draw the shot chart.")
        else:
            fig, ax = plt.subplots(figsize=(7, 6.6))
            plot_shot_hexbin(
                chart_shots,
                ax=ax,
                gridsize=25,
                mincnt=2,
                color_by="fg_pct" if color_by == "FG%" else "count",
                title=f"{player_name} — {SEASON} ({len(chart_shots):,} shots)",
            )
            st.pyplot(fig)
            plt.close(fig)

    with profile_col:
        st.subheader("Shooting profile")
        st.dataframe(
            style_table(shooting_profile(splits, player_id), {"Value": "{:.1%}", "Attempts": "{:,.0f}"}),
            hide_index=True,
        )
        st.caption(
            "Season totals across all teams. Percentiles rank the player against "
            "the league on each metric, among players with enough attempts to qualify."
        )
        if load_box_stats() is None:
            st.warning(
                "True shooting % and free throw % need season box score totals. "
                f"Run `pull_player_box_stats()` from scraper_functions.py and save the "
                f"result to `data/{BOX_STATS_PATH.name}`."
            )

with touches_tab:
    st.subheader("Touch locations")

    if (touches["PLAYER_ID"] == player_id).any():
        touch_row = touches.loc[touches["PLAYER_ID"] == player_id].iloc[0]
        area_col, touch_table_col = st.columns([3, 2])

        with area_col:
            shares = {key: touch_row[share_col] for share_col, _label, _count, key in TOUCH_AREAS}
            colors = {
                key: percentile_color(touch_row[f"{share_col}_PCTILE"])
                for share_col, _label, _count, key in TOUCH_AREAS
            }
            labels = {key: label for _share, label, _count, key in TOUCH_AREAS}

            total_touches = int(touch_row["TOUCHES"])
            # Wider than tall: the touch areas plot crops the empty top of the court.
            fig, ax = plt.subplots(figsize=(7, 5.2))
            plot_touch_areas(
                shares,
                colors=colors,
                labels=labels,
                ax=ax,
                title=f"{player_name} — {SEASON} ({total_touches:,} touches)",
            )
            st.pyplot(fig)
            plt.close(fig)

        with touch_table_col:
            st.dataframe(
                style_table(
                    touch_profile(touches, player_id),
                    {"Share of touches": "{:.1%}", "Touches": "{:,.0f}"},
                ),
                hide_index=True,
            )
            st.caption(
                "Share of the player's total touches taken in each area, shaded by "
                "league percentile. Areas are the NBA's tracking definitions, so the "
                "elbow overlaps the top of the paint."
            )
    else:
        st.info(f"No touch tracking data for {player_name}.")

    st.divider()
    st.subheader("Pick and roll")

    if pnr is None:
        st.warning(
            "Pick and roll scoring needs the Synergy play type pull. Run "
            "`pull_pick_and_roll()` from scraper_functions.py and save the combined "
            f"result to `data/{PNR_PATH.name}`."
        )
    else:
        st.dataframe(
            style_table(pnr_profile(pnr, player_id), {"Points": "{:,.0f}", "Possessions": "{:,.0f}"}),
            hide_index=True,
        )
        st.caption(
            "Points scored out of the pick and roll in each role. Percentiles rank the "
            "player only against others who log possessions in that role, so roll men "
            "(bigs) and ball handlers (guards) are each ranked against their own group."
        )

        board_cols = st.columns(2)
        for column, (prefix, label, _play_type) in zip(board_cols, PNR_ROLES):
            with column:
                st.markdown(f"**{label} — all players**")
                st.dataframe(
                    style_table(
                        pnr_leaderboard(pnr, prefix), {"Points": "{:,.0f}", "Possessions": "{:,.0f}"}
                    ),
                    hide_index=True,
                    height=360,
                )


with impact_tab:
    st.subheader("Impact")

    if impact is None:
        st.warning(
            "Box plus/minus and win shares need the tracking and possessions "
            "pulls. Run `pull_tracking_stats()` from `scraper_functions.py`."
        )
    else:
        st.caption(
            f"There is no box score in the committed pulls, so one is "
            f"reconstructed: points and minutes from the possessions pull, field "
            f"goals from the shot chart, free throws from the difference between "
            f"them, and rebounds, assists, steals and blocks from the tracking "
            f"pulls. League pace comes out at {impact.pace:.0f} possessions per 48 "
            f"minutes and {impact.points_per_possession:.3f} points per possession. "
            + ("Turnovers are the one estimate — the pulls only track them on "
               "drives and on paint, post and elbow touches, so the rest are "
               "spread over each player's untracked touches."
               if impact.turnovers_estimated else
               "Turnovers come from the box score pull.")
        )

        impact_left, impact_right = st.columns([3, 2])

        with impact_left:
            st.dataframe(style_impact(impact_profile(impact, player_id)),
                         hide_index=True)
            st.caption(
                f"Box plus/minus is points added per 100 possessions over an "
                f"average player, so the league averages 0 by construction. Win "
                f"shares per 48 averages {AVERAGE_WS48:.3f} for the same reason "
                f"win shares sum to wins: teams average .500, so a player who "
                f"plays a fifth of his team's minutes earns a fifth of 41 wins. "
                f"Percentiles rank against everyone over {MIN_MINUTES:,.0f} minutes."
            )

        with impact_right:
            row = impact.of(player_id)
            if row is None:
                st.info(
                    f"No shot chart for {player_name}, so there is no "
                    "reconstructed box score to price."
                )
            else:
                st.markdown("**Where the points come from**")
                # The two halves in points, not per 100, because this is about
                # what a player did rather than at what rate.
                halves = pd.DataFrame({
                    "Points added": [row["OFF_POINTS_ADDED"], row["DEF_POINTS_ADDED"]],
                }, index=["Offence", "Defence"])
                st.bar_chart(halves, horizontal=True)
                st.caption(
                    "Every box event priced in points against what an average "
                    "possession is worth: scoring above league true shooting on "
                    "the same volume, the points a pass sets up, a turnover "
                    "against the possession it cost, rebounds against the share "
                    "a teammate would have got anyway, and stops."
                )

    st.divider()
    st.subheader("Stars")

    if tiers is None:
        st.info("The star tiers need the impact metrics above.")
    else:
        star_count = len(tiers.stars)
        st.caption(
            f"A second clustering, on the other axis from the archetypes. Those "
            f"divide volume out on purpose — their question is what a player "
            f"does — so they cannot tell a bench guard from a franchise guard "
            f"who takes the same shots. These put volume back in: box plus/minus "
            f"and win shares against points, shot volume, points created and "
            f"minutes per game. k = {tiers.k}, chosen by silhouette score "
            f"({tiers.silhouette:.3f}) subject to the top tier holding no more "
            f"than a fraction of the league, and the tiers are named by where "
            f"their centres rank rather than by hand. It lands on {star_count} "
            f"stars out of {len(tiers.table)} clustered players."
        )

        tier = tiers.tier_of(player_id)
        if tier is None:
            st.info(
                f"{player_name} is under the {MIN_MINUTES:,.0f} minute floor, so "
                "he is not placed in a tier."
            )
        else:
            st.markdown(f"**{player_name} — {tier}**")

        st.dataframe(
            star_tier_table(tiers),
            hide_index=True,
            column_config={
                "Box +/-": st.column_config.NumberColumn(format="%+.1f"),
                "WS/48": st.column_config.NumberColumn(format="%.3f"),
                "Minutes / game": st.column_config.NumberColumn(format="%.1f"),
                "Points / game": st.column_config.NumberColumn(format="%.1f"),
            },
        )
        st.caption(
            "The two middle tiers separate on minutes as much as on quality — a "
            "bench big who rebounds well in nineteen minutes rates ahead of a "
            "starter per possession and behind him per night, which is what the "
            "column that defines each tier is there to show."
        )

        st.markdown("**The star tier**")
        st.dataframe(
            star_leaderboard(tiers, impact),
            hide_index=True,
            height=360,
            column_config={
                "Box +/-": st.column_config.NumberColumn(format="%+.1f"),
                "WS/48": st.column_config.NumberColumn(format="%.3f"),
                "Win shares": st.column_config.NumberColumn(format="%.1f"),
                "Points / game": st.column_config.NumberColumn(format="%.1f"),
                "Minutes / game": st.column_config.NumberColumn(format="%.1f"),
            },
        )


with archetype_tab:
    st.subheader("Archetypes")

    if archetypes is None:
        st.warning(
            "Archetypes need the combined tracking pull. Run `pull_tracking_stats()` "
            f"from scraper_functions.py and save the result to `data/{TRACKING_PATH.name}`."
        )
    else:
        st.caption(
            f"K-means over {len(archetypes.table)} players with at least "
            f"{MIN_MINUTES:,.0f} minutes, on per-36 tracking rates. "
            f"k = {archetypes.k}, chosen by silhouette score ({archetypes.silhouette:.3f}); "
            "each cluster is named after the prototype its centre points towards, and "
            "the features that earned the name are listed with it."
        )

        label = archetypes.label_of(player_id)
        if label is None:
            st.info(
                f"{player_name} is under the {MIN_MINUTES:,.0f} minute floor, so they are "
                "not clustered. Rate stats on a small sample would move the cluster "
                "centres more than they would describe the player."
            )
        else:
            cluster = int(
                archetypes.table.loc[
                    archetypes.table["PLAYER_ID"] == player_id, "CLUSTER"
                ].iloc[0]
            )
            profile_col, similar_col = st.columns([3, 2])

            with profile_col:
                st.markdown(f"**{player_name} — {label}**")
                player_z = archetypes.profile_of(player_id)
                centre = archetypes.centroids.loc[cluster]
                order = player_z.abs().sort_values(ascending=False).index[:8]
                comparison = pd.DataFrame(
                    {"This player": player_z[order], f"{label} average": centre[order]}
                )
                st.bar_chart(comparison, horizontal=True)
                st.caption(
                    "Standard deviations from the league average, on the eight features "
                    "where this player is furthest from it. The second bar is the "
                    "archetype's own centre, so the gap is how typical they are of it."
                )

            with similar_col:
                st.markdown("**Most similar players**")
                similar = archetypes.similar_players(player_id, n=6)
                st.dataframe(
                    similar.rename(columns={"PLAYER_NAME": "Player", "ARCHETYPE": "Archetype"}),
                    hide_index=True,
                    column_config={
                        "Distance": st.column_config.NumberColumn(format="%.2f")
                    },
                )
                st.caption(
                    "Closest in the same feature space, so this reads as used the same "
                    "way rather than scores the same amount."
                )

        st.divider()
        st.markdown("**The archetypes**")
        summary_rows = []
        for cluster_id, name in sorted(archetypes.names.items(), key=lambda kv: kv[1]):
            members = archetypes.table.loc[archetypes.table["CLUSTER"] == cluster_id]
            top = archetypes.distinguishing(cluster_id, 3)
            summary_rows.append({
                "Archetype": name,
                "Players": len(members),
                "What defines it": ", ".join(
                    f"{feature} {value:+.1f}" for feature, value in top.items()
                ),
                "Most minutes": ", ".join(members.nlargest(3, "MIN")["PLAYER_NAME"]),
            })
        st.dataframe(pd.DataFrame(summary_rows), hide_index=True)


with shooting_tab:
    st.subheader("Projected true shooting")
    projection = load_projection()

    if projection is None:
        st.info(
            "Projected true shooting needs the tracking and possessions pulls. "
            "Run `pull_tracking_stats()` from `scraper_functions.py`."
        )
    else:
        calibration = projection.calibration
        st.caption(
            f"True shooting is reconstructed from the shot chart and season points "
            f"— free throws made are points minus what the field goals were worth — "
            f"then pooled toward the player's archetype. The pooling constant "
            f"k = {projection.k:.0f} attempts is fitted, not chosen: it is the value "
            f"that best predicts shooting after "
            f"{calibration['cut']:%B %-d} from shooting before it, across "
            f"{calibration['n_players']} players."
        )

        row = projection.of(player_id)
        if row is None:
            st.info(f"No shot chart for {player_name}, so there is nothing to project.")
        else:
            low = row["PROJECTED_TS"] - 1.96 * row["SE"]
            high = row["PROJECTED_TS"] + 1.96 * row["SE"]
            metric_cols = st.columns(4)
            metric_cols[0].metric("Shot", f"{row['TS_PCT']:.1%}")
            metric_cols[1].metric("Projected", f"{row['PROJECTED_TS']:.1%}",
                                  delta=f"{row['SHIFT']:+.1%}")
            metric_cols[2].metric("95% interval", f"{low:.1%} – {high:.1%}")
            metric_cols[3].metric("True shooting attempts", f"{row['TSA']:,.0f}")

            group = row["ARCHETYPE"] or "the league"
            st.markdown(
                f"On {row['TSA']:,.0f} true shooting attempts, "
                f"**{row['WEIGHT']:.0%}** of the projection is {player_name}'s own "
                f"record and **{1 - row['WEIGHT']:.0%}** is "
                f"{group} at {row['GROUP_TS']:.1%}."
            )

        st.divider()
        left, right = st.columns([3, 2])

        with left:
            st.markdown("**How far each player is pulled**")
            plot = projection.table[["TSA", "TS_PCT", "PROJECTED_TS"]].copy()
            plot = plot.rename(columns={"TS_PCT": "Shot", "PROJECTED_TS": "Projected"})
            st.scatter_chart(plot, x="TSA", y=["Shot", "Projected"],
                             x_label="True shooting attempts", y_label="True shooting %")
            st.caption(
                "The two clouds are the same players. Out at high volume they sit on "
                "top of each other; the gap opens up on the left, where a season is "
                "too short to tell a hot stretch from a shooter."
            )

        with right:
            st.markdown("**Where each archetype shoots from**")
            groups_table = projection.groups.rename(columns={
                "ARCHETYPE": "Archetype", "PLAYERS": "Players",
                "GROUP_TS": "Pooled TS%"})
            st.dataframe(
                groups_table[["Archetype", "Players", "Pooled TS%"]],
                hide_index=True,
                column_config={"Pooled TS%": st.column_config.NumberColumn(format="%.3f")},
            )
            st.caption(
                f"League {projection.league:.1%}. These are what a low-volume player "
                "is pulled toward, so a reserve big is measured against other bigs "
                "rather than against pull-up guards."
            )

        st.divider()
        st.markdown("**Biggest corrections**")
        movers = projection.table.assign(ABS=projection.table["SHIFT"].abs())
        movers = movers.nlargest(12, "ABS")[
            ["PLAYER_NAME", "TEAM_ABBREVIATION", "ARCHETYPE", "TSA",
             "TS_PCT", "PROJECTED_TS", "SHIFT"]]
        st.dataframe(
            movers.rename(columns={
                "PLAYER_NAME": "Player", "TEAM_ABBREVIATION": "Team",
                "ARCHETYPE": "Archetype", "TSA": "Attempts",
                "TS_PCT": "Shot", "PROJECTED_TS": "Projected", "SHIFT": "Change"}),
            hide_index=True,
            column_config={
                "Attempts": st.column_config.NumberColumn(format="%.0f"),
                "Shot": st.column_config.NumberColumn(format="%.3f"),
                "Projected": st.column_config.NumberColumn(format="%.3f"),
                "Change": st.column_config.NumberColumn(format="%+.3f"),
            },
        )
        st.caption(
            "Every one of these is a small sample. A player who takes 900 attempts "
            "has already told you what he is; a player who takes 90 has not."
        )
