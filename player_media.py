"""Headshots, team logos and team colours for the scouting dashboard.

The NBA serves both headshots and team logos from its own CDN, keyed by the
same player and team ids the stats endpoints return, so nothing here needs a
scrape or a local asset directory:

    headshot:  https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png
    logo:      https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.svg

Two things matter for a dashboard someone else runs. The first is that a
missing or slow image must not leave a broken-image icon in the middle of a
scouting card, so :func:`headshot_html` stacks the portrait over a monogram
drawn inline as a data URI - it needs no network and no file. The second is
that a player who has just been traded, or is on a two-way deal, may not have
a portrait at that id yet; that is the same fallback path rather than a
special case.

The fallback is two CSS background layers rather than an ``<img>`` with an
``onerror`` handler. Streamlit strips event-handler attributes out of the HTML
it renders, so an ``onerror`` fallback silently never fires there and the card
collapses to nothing; a browser that cannot fetch the first background layer
just paints the next one down, with no script involved and no broken-image
glyph on top.
"""

from __future__ import annotations

import base64
import html
import re

__all__ = [
    "HEADSHOT_ASPECT",
    "TEAM_COLORS",
    "headshot_url",
    "initials",
    "monogram_data_uri",
    "headshot_html",
    "team_logo_url",
    "team_color",
]

HEADSHOT_CDN = "https://cdn.nba.com/headshots/nba/latest/{size}/{player_id}.png"
LOGO_CDN = "https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.svg"

# The CDN publishes every headshot size at this ratio, and the monogram below
# uses the same one so the card does not reflow when the portrait is missing.
HEADSHOT_ASPECT = 1040 / 760

# Primary colour per team, keyed by the tricode the stats endpoints return.
TEAM_COLORS = {
    "ATL": "#E03A3E", "BOS": "#007A33", "BKN": "#000000", "CHA": "#1D1160",
    "CHI": "#CE1141", "CLE": "#860038", "DAL": "#00538C", "DEN": "#0E2240",
    "DET": "#C8102E", "GSW": "#1D428A", "HOU": "#CE1141", "IND": "#002D62",
    "LAC": "#C8102E", "LAL": "#552583", "MEM": "#5D76A9", "MIA": "#98002E",
    "MIL": "#00471B", "MIN": "#0C2340", "NOP": "#0C2340", "NYK": "#006BB6",
    "OKC": "#007AC1", "ORL": "#0077C0", "PHI": "#006BB6", "PHX": "#1D1160",
    "POR": "#E03A3E", "SAC": "#5A2D81", "SAS": "#C4CED4", "TOR": "#CE1141",
    "UTA": "#002B5C", "WAS": "#002B5C",
}

DEFAULT_COLOR = "#1D428A"      # the league's own blue, for anything unmapped


def headshot_url(player_id: int | str, size: str = "1040x760") -> str:
    """CDN URL for a player's headshot at one of the sizes the CDN publishes."""
    return HEADSHOT_CDN.format(size=size, player_id=int(player_id))


def team_logo_url(team_id: int | str) -> str:
    """CDN URL for a team's primary logo."""
    return LOGO_CDN.format(team_id=int(team_id))


def team_color(tricode: str | None) -> str:
    """A team's primary colour, or the league blue when it is not recognised."""
    if not tricode:
        return DEFAULT_COLOR
    return TEAM_COLORS.get(str(tricode).strip().upper(), DEFAULT_COLOR)


def initials(name: str, limit: int = 2) -> str:
    """Initials for the monogram fallback: "Shai Gilgeous-Alexander" -> "SG"."""
    words = [word for word in re.split(r"[\s\-']+", str(name)) if word]
    return "".join(word[0].upper() for word in words[:limit]) or "?"


def monogram_data_uri(name: str, color: str = DEFAULT_COLOR) -> str:
    """An inline SVG monogram, used when the CDN has no portrait for a player.

    Returned as a base64 data URI rather than raw SVG so it can sit inside a
    CSS ``url()`` without any quoting of its own to escape.
    """
    letters = initials(name)
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 260 190'>"
        f"<rect width='260' height='190' fill='{color}'/>"
        "<text x='130' y='128' text-anchor='middle' font-size='84' font-weight='700' "
        f"font-family='Helvetica,Arial,sans-serif' fill='#FFFFFF'>{html.escape(letters)}</text>"
        "</svg>"
    )
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def headshot_html(
    player_id: int | str,
    name: str,
    tricode: str | None = None,
    height: int = 150,
    size: str = "1040x760",
) -> str:
    """A player's headshot that degrades to a monogram when the CDN has none.

    The portrait and the monogram are stacked as two background layers on one
    element; if the portrait never loads, the monogram underneath shows through
    unchanged. The element is sized explicitly so a missing portrait leaves the
    card the same shape as a present one.
    """
    color = team_color(tricode)
    fallback = monogram_data_uri(name, color)
    height = int(height)
    width = round(height * HEADSHOT_ASPECT)
    label = html.escape(str(name), quote=True)
    return (
        f'<div role="img" aria-label="{label}" title="{label}" '
        f'style="height:{height}px;width:{width}px;flex:0 0 auto;'
        f"border-radius:6px;background-color:{color};"
        f"background-image:url('{headshot_url(player_id, size=size)}'),url('{fallback}');"
        'background-size:cover,cover;background-position:center top,center;'
        'background-repeat:no-repeat,no-repeat"></div>'
    )
