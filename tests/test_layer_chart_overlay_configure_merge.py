"""`LayerChart`'s static overlay route (`to_svg()`) resolves chart-level
`configure_*` the same way `+` and `_build_merged()` do: one list -- the
`LayerChart`'s own configure layers, then each member's own list in member
order -- applies identically to every member's marks, every member's color
scale and the one shared legend, in either member order. Before this, each
member read only its own list, so a sibling's `configure_color` never
reached it: the legend (built from one establishing member) could show a
range, domain or scheme a non-establishing member's own marks never drew.

A member's own differing `.theme()` is the other half of that: every
overlay member now also lowers with the primary member's theme (never its
own), the same primary-wins rule `+` already applies, with the same
"secondary layer's theme/facet is ignored" warning -- without it, a member
whose own theme differs from the primary's could still draw its own
(wrong) palette underneath a legend built from the shared, merged config.
"""

from __future__ import annotations

import re
import warnings

import polars as pl
import pytest

import ferrum as fm

_DF = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "g": ["a", "b", "c"]})
_DF2 = pl.DataFrame(
    {
        "x": [1.0, 2.0, 3.0, 4.0],
        "y": [1.0, 2.0, 3.0, 4.0],
        "g": ["a", "b", "a", "c"],
        "q": [1.0, 2.0, 3.0, 4.0],
        "v": [-4.0, 2.0, 4.0, 9.0],
    }
)
_ECONOMIST = fm.themes.economist

# A legend swatch circle has the default radius (`r="4"`) and is immediately
# followed by its category label `<text>`, in document order -- distinct
# from a mark circle, which carries the mark's own (non-"4") radius and no
# adjacent label.
_CIRCLE = re.compile(r'<circle cx="[^"]+" cy="[^"]+" r="([^"]+)" fill="(#[0-9a-fA-F]{6})"/>')
_LEGEND_ENTRY = re.compile(
    r'<circle[^>]*\br="4"[^>]*fill="(#[0-9a-fA-F]{6})"/><text[^>]*>([^<]*)</text>'
)
_LEGEND_TITLE = re.compile(r"<text[^>]*>g</text>")
_COLORBAR_STOP = re.compile(r'<stop offset="[^"]+" stop-color="(#[0-9a-fA-F]{6})"/>')


def _consumer() -> fm.Chart:
    return fm.Chart(_DF).mark_point().encode(x="x", y="y", color="g")


def _plain() -> fm.Chart:
    """A member with no color encoding, so its mark color is unbound."""
    return fm.Chart(_DF).mark_point().encode(x="x", y="y")


def _colored(field: str) -> fm.Chart:
    return fm.Chart(_DF2).mark_point().encode(x="x", y="y", color=field)


# field, configure_color() kwargs: "g" is categorical (scheme), "q" is a
# positive quantitative column (sequential_scheme), "v" is a signed
# quantitative column (diverging_scheme).
_SCHEME_FAMILY: dict[str, tuple[str, dict]] = {
    "scheme": ("g", {"scheme": "tableau10"}),
    "sequential_scheme": ("q", {"sequential_scheme": "magma"}),
    "diverging_scheme": ("v", {"diverging_scheme": "rdbu"}),
}


def _mark_fills(svg: str) -> list[str]:
    """Point-mark fill colors in document order, one per row per member (excludes legend swatches)."""
    return [fill for radius, fill in _CIRCLE.findall(svg) if radius != "4"]


def _legend_category_colors(svg: str) -> list[tuple[str, str]]:
    """`(category label, fill color)` pairs for the color legend's swatches, in document order."""
    return [(label, fill) for fill, label in _LEGEND_ENTRY.findall(svg)]


def _has_legend(svg: str) -> bool:
    return bool(_LEGEND_TITLE.search(svg))


def _colorbar_stops(svg: str) -> list[str]:
    """Gradient stop colors of a continuous color legend's colorbar, in document order."""
    return _COLORBAR_STOP.findall(svg)


def _legend_parity(svg: str, case: str) -> list[str] | list[tuple[str, str]]:
    """The legend oracle for *case*: categorical swatches for `scheme`, colorbar
    stops for `sequential_scheme`/`diverging_scheme`."""
    return _legend_category_colors(svg) if case == "scheme" else _colorbar_stops(svg)


def test_configure_color_range_matches_plus_in_both_member_orders() -> None:
    a = _consumer()
    b = _consumer().configure_color(range=["#dc2626", "#00ff00", "#2563eb"])
    legend = [("a", "#dc2626"), ("b", "#00ff00"), ("c", "#2563eb")]
    marks = ["#dc2626", "#00ff00", "#2563eb"] * 2

    for chart in ((a + b), fm.LayerChart(a, b), fm.LayerChart(b, a)):
        svg = chart.to_svg()
        assert _legend_category_colors(svg) == legend
        assert _mark_fills(svg) == marks


def test_configure_color_domain_matches_plus_in_both_member_orders() -> None:
    a = _consumer()
    b = _consumer().configure_color(domain=["b", "a", "c"])
    legend = [("b", "#2563eb"), ("a", "#dc2626"), ("c", "#d4a017")]
    marks = ["#dc2626", "#2563eb", "#d4a017"] * 2

    for chart in ((a + b), fm.LayerChart(a, b), fm.LayerChart(b, a)):
        svg = chart.to_svg()
        assert _legend_category_colors(svg) == legend
        assert _mark_fills(svg) == marks


def test_configure_legend_orient_none_matches_plus_in_both_member_orders() -> None:
    a = _consumer()
    b = _consumer().configure_legend(orient="none")

    assert not _has_legend((a + b).to_svg())
    assert not _has_legend((b + a).to_svg())
    assert not _has_legend(fm.LayerChart(a, b).to_svg())
    assert not _has_legend(fm.LayerChart(b, a).to_svg())


def test_a_members_own_configure_outranks_the_parent_layer_charts() -> None:
    """A member's own `configure_color` wins over a later `LayerChart.configure_color`
    call, the same precedence `_build_merged()`'s `+` reduction gives it (the
    member's own config is appended after the parent's in the merged list) --
    and every member's own marks, not only the legend, follow the winning range."""
    a = _consumer().configure_color(range=["#dc2626", "#00ff00", "#2563eb"])
    b = _consumer()
    chart = fm.LayerChart(a, b).configure_color(range=["#111111", "#222222", "#333333"])
    legend = [("a", "#dc2626"), ("b", "#00ff00"), ("c", "#2563eb")]
    marks = ["#dc2626", "#00ff00", "#2563eb"] * 2

    for svg in (chart.to_svg(), chart._build_merged().to_svg()):
        assert _legend_category_colors(svg) == legend
        assert _mark_fills(svg) == marks


def test_configure_color_domain_differs_from_the_unconfigured_default() -> None:
    """Vacuity control: a reordered domain must actually change which category
    gets which color, not just reuse the same palette by coincidence."""
    default_order = _legend_category_colors(fm.LayerChart(_consumer(), _consumer()).to_svg())
    reordered = _legend_category_colors(
        fm.LayerChart(_consumer(), _consumer().configure_color(domain=["b", "a", "c"])).to_svg()
    )
    assert default_order != reordered


@pytest.mark.parametrize("case", sorted(_SCHEME_FAMILY))
def test_configure_color_scheme_family_matches_plus_in_both_member_orders(case: str) -> None:
    field, kwargs = _SCHEME_FAMILY[case]
    a = _colored(field)
    b = _colored(field).configure_color(**kwargs)

    for pair in ((a, b), (b, a)):
        x, y = pair
        plus_svg = (x + y).to_svg()
        lc_svg = fm.LayerChart(x, y).to_svg()
        assert _mark_fills(lc_svg) == _mark_fills(plus_svg)
        assert _legend_parity(lc_svg, case) == _legend_parity(plus_svg, case)


@pytest.mark.parametrize("call_order", [0, 1])
@pytest.mark.parametrize("theme_member", [0, 1])
@pytest.mark.parametrize("config_member", [0, 1])
@pytest.mark.parametrize("case", sorted(_SCHEME_FAMILY))
def test_configure_color_scheme_family_matches_plus_across_theme_and_config_placement(
    case: str, config_member: int, theme_member: int, call_order: int
) -> None:
    """A member's own `scheme=`/`sequential_scheme=`/`diverging_scheme=` and a
    member's own differing `.theme()` -- on the same member or on different
    members, in either position -- resolve identically to `+`, since every
    overlay member now lowers with the primary member's theme and the one
    merged configure list, the same as `+` does."""
    field, kwargs = _SCHEME_FAMILY[case]
    members = [_colored(field), _colored(field)]
    members[config_member] = members[config_member].configure_color(**kwargs)
    members[theme_member] = members[theme_member].theme(_ECONOMIST)
    if call_order == 1:
        members = members[::-1]
    a, b = members

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plus_svg = (a + b).to_svg()
        lc_svg = fm.LayerChart(a, b).to_svg()
    assert _mark_fills(lc_svg) == _mark_fills(plus_svg)
    assert _legend_parity(lc_svg, case) == _legend_parity(plus_svg, case)


def test_a_differing_secondary_theme_alone_matches_plus_unbound_mark_color() -> None:
    """With no color encoding at all (so the mark color is unbound, not
    scale-driven), a secondary member's own differing theme must not reach
    it either -- the primary's theme wins on every theme-driven input, as
    it does on `+`."""
    a = _plain()
    b = _plain().theme(_ECONOMIST)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert _mark_fills(fm.LayerChart(a, b).to_svg()) == _mark_fills((a + b).to_svg())
        assert _mark_fills(fm.LayerChart(b, a).to_svg()) == _mark_fills((b + a).to_svg())


def test_a_differing_secondary_theme_warns_the_same_text_as_plus() -> None:
    a = _consumer()
    b = _consumer().theme(_ECONOMIST)

    with pytest.warns(UserWarning, match="secondary layer's theme/facet is ignored"):
        fm.LayerChart(a, b).to_svg()


def test_a_differing_secondary_facet_warns_the_same_text_as_plus() -> None:
    a = _plain()
    b = _plain().facet(col="g")

    with pytest.warns(UserWarning, match="secondary layer's theme/facet is ignored"):
        fm.LayerChart(a, b).to_svg()


def test_no_warning_when_themes_match_or_the_secondary_carries_none() -> None:
    a = _consumer()
    b_matching = _consumer().theme(_ECONOMIST)
    b_none = _consumer()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fm.LayerChart(a.theme(_ECONOMIST), b_matching).to_svg()
        fm.LayerChart(a, b_none).to_svg()
    assert not caught


def test_members_sharing_a_theme_render_the_plus_routes_mark_fills_with_no_warning() -> None:
    """Members that already share one theme render with no warning and the
    same mark fills as the ``+`` route under that theme."""
    a = _consumer()
    b = _consumer()
    shared = fm.LayerChart(a.theme(_ECONOMIST), b.theme(_ECONOMIST))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = shared.to_svg()
    assert not caught
    assert _mark_fills(svg) == _mark_fills((a.theme(_ECONOMIST) + b.theme(_ECONOMIST)).to_svg())
