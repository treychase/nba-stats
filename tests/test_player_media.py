import base64

import pytest

from player_media import (
    DEFAULT_COLOR,
    TEAM_COLORS,
    headshot_html,
    headshot_url,
    initials,
    monogram_data_uri,
    team_color,
    team_logo_url,
)


def test_headshot_url_uses_the_player_id():
    assert headshot_url(1628983).endswith("/1040x760/1628983.png")
    assert headshot_url("1628983", size="260x190").endswith("/260x190/1628983.png")


def test_team_logo_url_uses_the_team_id():
    assert team_logo_url(1610612760).endswith("/1610612760/global/L/logo.svg")


def test_every_team_colour_is_a_hex_triplet():
    assert len(TEAM_COLORS) == 30
    for tricode, color in TEAM_COLORS.items():
        assert len(tricode) == 3
        assert color.startswith("#") and len(color) == 7
        int(color[1:], 16)


def test_team_colour_is_case_insensitive_and_falls_back():
    assert team_color("okc") == team_color("OKC") == TEAM_COLORS["OKC"]
    assert team_color(None) == DEFAULT_COLOR
    assert team_color("XXX") == DEFAULT_COLOR


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Shai Gilgeous-Alexander", "SG"),
        ("De'Aaron Fox", "DA"),
        ("Nikola Jokić", "NJ"),
        ("Nene", "N"),
        ("", "?"),
    ],
)
def test_initials(name, expected):
    assert initials(name) == expected


def test_monogram_is_a_decodable_svg_carrying_the_initials():
    uri = monogram_data_uri("Victor Wembanyama", "#1D1160")
    assert uri.startswith("data:image/svg+xml;base64,")
    svg = base64.b64decode(uri.split(",", 1)[1]).decode("utf-8")
    assert svg.startswith("<svg") and ">VW<" in svg
    assert "#1D1160" in svg


def test_headshot_html_stacks_the_monogram_under_the_portrait():
    markup = headshot_html(1628983, "Shai Gilgeous-Alexander", "OKC", height=120)
    portrait = "https://cdn.nba.com/headshots"
    monogram = "data:image/svg+xml;base64,"
    assert portrait in markup and monogram in markup
    # the portrait has to be the first layer, or it would be painted over
    assert markup.index(portrait) < markup.index(monogram)
    # no event handler: Streamlit strips those, so the fallback cannot rely on
    # one and the element must be sized whether or not the portrait loads
    assert "onerror" not in markup
    assert "height:120px" in markup and "width:164px" in markup
    assert markup.count('"') % 2 == 0


def test_headshot_html_escapes_the_player_name():
    markup = headshot_html(1, 'Bad "Name" <script>', "OKC")
    assert "<script>" not in markup
    assert "&quot;" in markup and "&lt;script&gt;" in markup