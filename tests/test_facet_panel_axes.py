"""A faceted chart draws each panel's axes as layout lays them out, on independent and shared axes alike.

- Axis titles: the x title only under the bottom row (a panel with no
  panel below it in a short wrap counts as bottom), the y title only beside
  the left column, whether the axis is shared or independent.
- An explicit Band/Point pixel ``range=`` is re-anchored per panel for the
  marks, and every panel's tick marks and labels sit on that panel's own
  marks: x or y, shared or independent, under ``CoordFlip``, and on a
  secondary y axis.
- ``axis(x=False)`` / ``axis(y=False)`` hides that axis on every panel of an
  independent facet, as on a shared one.

Positions are compared as the SVG strings both sides are written with, so a
match is exact.
"""

from __future__ import annotations

import re

import polars as pl
import pytest

import ferrum as fm

_TICK_DASH = re.compile(
    r'<line x1="([-\d.]+)" y1="([-\d.]+)" x2="([-\d.]+)" y2="([-\d.]+)" '
    r'stroke="#6b7280" stroke-width="1"/>'
)
_TEXT = re.compile(r"<text ([^>]*)>([^<]*)</text>")

# Four panels in a 2x2 grid, numeric x and y.
_GRID = pl.DataFrame(
    {
        "x": list(range(8)),
        "y": [i * 1.5 for i in range(8)],
        "f": ["a", "a", "b", "b", "c", "c", "d", "d"],
    }
)
# Two panels, three categories each.
_CATS = pl.DataFrame(
    {
        "c": ["a", "b", "c"] * 2,
        "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "s": ["u", "w", "z"] * 2,
        "f": ["p"] * 3 + ["q"] * 3,
    }
)


def _tick_dashes(svg: str) -> tuple[list[str], list[str]]:
    """The x and y positions of every axis tick dash *svg* draws (a short
    axis-colored line across its axis), as the SVG writes them."""
    xs, ys = [], []
    for x1, y1, x2, y2 in _TICK_DASH.findall(svg):
        if x1 == x2 and abs(float(y1) - float(y2)) <= 6:
            xs.append(x1)
        elif y1 == y2 and abs(float(x1) - float(x2)) <= 6:
            ys.append(y1)
    return xs, ys


def _point_positions(svg: str, attr: str) -> list[str]:
    return re.findall(rf'<circle[^>]*{attr}="([-\d.]+)"', svg)


def _titled_grid(**share) -> fm.Chart:
    return (
        fm.Chart(_GRID)
        .mark_point()
        .encode(x=fm.X("x", title="XTITLE"), y=fm.Y("y", title="YTITLE"))
        .facet("f", ncols=2)
        .share_scale(**share)
        .properties(width=420, height=360)
    )


class TestAxisTitles:
    @pytest.mark.parametrize(
        "share",
        [
            pytest.param({}, id="shared"),
            pytest.param({"x": "independent"}, id="independent-x"),
            pytest.param({"y": "independent"}, id="independent-y"),
            pytest.param({"x": "independent", "y": "independent"}, id="independent-both"),
        ],
    )
    def test_a_2x2_facet_titles_only_the_bottom_row_and_left_column(self, share):
        svg = _titled_grid(**share).to_svg()
        assert svg.count(">XTITLE<") == 2
        assert svg.count(">YTITLE<") == 2

    def test_a_short_wrap_titles_the_panel_with_nothing_below_it_as_bottom(self):
        """Three panels in two columns: panel b sits in the top row with no
        panel under it. Its x axis is drawn, but the only x title is the one
        under panel c, as on a shared facet."""
        df = _GRID.filter(pl.col("f") != "d")
        for share in ({}, {"x": "independent"}):
            svg = (
                fm.Chart(df)
                .mark_point()
                .encode(x=fm.X("x", title="XTITLE"), y=fm.Y("y", title="YTITLE"))
                .facet("f", ncols=2)
                .share_scale(**share)
                .properties(width=420, height=360)
                .to_svg()
            )
            assert svg.count(">XTITLE<") == 1, share
            assert svg.count(">YTITLE<") == 2, share


class TestExplicitRangeTicksSitOnTheirMarks:
    @pytest.mark.parametrize(
        "scale",
        [
            pytest.param(fm.BandScale(range=[10, 110]), id="band"),
            pytest.param(fm.PointScale(range=[10, 110]), id="point"),
        ],
    )
    @pytest.mark.parametrize("resolve", ["independent", "shared"])
    def test_x_ticks_and_labels_sit_on_every_panels_points(self, scale, resolve):
        svg = (
            fm.Chart(_CATS)
            .mark_point()
            .encode(x=fm.X("c", title="CAT", scale=scale), y=fm.Y("v", title="VAL"))
            .facet("f", ncols=2)
            .share_scale(x=resolve)
            .properties(width=500, height=200)
            .to_svg()
        )
        centers = _point_positions(svg, "cx")
        assert centers == ["26.667", "60", "93.333", "243.402", "276.735", "310.068"]
        x_dashes, _ = _tick_dashes(svg)
        assert x_dashes == centers
        labels = [
            re.search(r'x="([-\d.]+)"', attrs)
            for attrs, text in _TEXT.findall(svg)
            if text in ("a", "b", "c")
        ]
        assert all(labels), "every category label carries an x position"
        assert [match.group(1) for match in labels] == centers

    @pytest.mark.parametrize(
        "scale",
        [
            pytest.param(fm.BandScale(range=[10, 110]), id="band"),
            pytest.param(fm.PointScale(range=[10, 110]), id="point"),
        ],
    )
    @pytest.mark.parametrize("resolve", ["independent", "shared"])
    def test_y_ticks_sit_on_every_panels_points(self, scale, resolve):
        svg = (
            fm.Chart(_CATS)
            .mark_point()
            .encode(y=fm.Y("c", title="CAT", scale=scale), x=fm.X("v", title="VAL"))
            .facet("f", ncols=1)
            .share_scale(y=resolve)
            .properties(width=300, height=300)
            .to_svg()
        )
        centers = _point_positions(svg, "cy")
        assert centers == ["26.667", "60", "93.333", "154.802", "188.135", "221.469"]
        _, y_dashes = _tick_dashes(svg)
        assert y_dashes == centers

    @pytest.mark.parametrize("resolve", ["independent", "shared"])
    def test_a_flipped_explicit_range_sits_on_every_panels_points(self, resolve):
        svg = (
            fm.Chart(_CATS)
            .mark_point()
            .encode(
                x=fm.X("c", title="CAT", scale=fm.BandScale(range=[10, 110])),
                y=fm.Y("v", title="VAL"),
            )
            .coord(fm.CoordFlip())
            .facet("f", ncols=1)
            .share_scale(x=resolve)
            .properties(width=300, height=300)
            .to_svg()
        )
        centers = _point_positions(svg, "cy")
        assert centers == ["26.667", "60", "93.333", "154.802", "188.135", "221.469"]
        _, y_dashes = _tick_dashes(svg)
        assert y_dashes == centers

    def test_an_explicit_range_secondary_y_sits_on_every_panels_points(self):
        base = (
            fm.Chart(_CATS).mark_point().encode(x=fm.X("v", title="VAL"), y=fm.Y("v", title="VAL2"))
        )
        svg = (
            (base + fm.SecondaryY("s", mark="point", scale=fm.BandScale(range=[10, 70])))
            .facet("f", ncols=1)
            .properties(width=300, height=300)
            .to_svg()
        )
        _, y_dashes = _tick_dashes(svg)
        # The secondary layer's points: 20/40/60 in panel 0, re-anchored to
        # 148.135/168.135/188.135 in panel 1.
        secondary = ["20", "40", "60", "148.135", "168.135", "188.135"]
        assert set(secondary) <= set(_point_positions(svg, "cy"))
        assert set(secondary) <= set(y_dashes)


class TestHiddenAxisOnIndependentFacets:
    @pytest.mark.parametrize("hidden", ["x", "y"])
    def test_a_hidden_independent_axis_draws_nothing_on_any_panel(self, hidden):
        svg = (
            fm.Chart(_GRID)
            .mark_point()
            .encode(x=fm.X("x", title="XTITLE"), y=fm.Y("y", title="YTITLE"))
            .facet("f", ncols=2)
            .share_scale(**{hidden: "independent"})
            .axis(**{hidden: False})
            .properties(width=420, height=360)
            .to_svg()
        )
        x_dashes, y_dashes = _tick_dashes(svg)
        hidden_dashes, shown_dashes = (
            (x_dashes, y_dashes) if hidden == "x" else (y_dashes, x_dashes)
        )
        assert hidden_dashes == []
        assert shown_dashes
        assert svg.count(f">{hidden.upper()}TITLE<") == 0
        shown = "Y" if hidden == "x" else "X"
        assert svg.count(f">{shown}TITLE<") == 2
