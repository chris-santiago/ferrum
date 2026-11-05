"""A composite mark call that replaces the chart's mark replaces its style too.

``.mark_ribbon()``, ``.mark_histogram()`` and ``.mark_density()`` desugar to a
single, unlayered mark, built from the chart's own resolved style
(``Chart._mark_kwargs``) merged with the composite's own catch-all keywords.
A primitive mark call already clears ``_mark_kwargs`` and ``_mark_zero`` when
it replaces the chart's mark (``Chart._set_mark``); a composite call that
replaces a mark does the same (``Chart._set_composite_mark``), so a chained
``mark_point(size=30).mark_ribbon()`` draws the ribbon ``mark_ribbon()`` draws
on its own, not a ribbon restyled by the point it replaced. A fitted mark
(``.mark_smooth()``, ``.mark_function()``) is the one call that keeps the
mark it replaces as its own layer, so this reset does not apply there.
"""

from __future__ import annotations

import warnings

import polars as pl
import pytest

import ferrum as fm
from tests._svg_rects import svg_elements

_DF = pl.DataFrame(
    {
        "x": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
        "lo": [0.0, 1.0, 2.0, 1.0, 1.0, 2.0, 3.0, 2.0],
        "hi": [1.0, 2.0, 3.0, 3.0, 2.0, 3.0, 4.0, 4.0],
        "g": ["a"] * 4 + ["b"] * 4,
    }
)
_V = pl.DataFrame({"x": [float(i % 7) + 50.0 for i in range(20)]})
_JOINT_DF = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "b": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0]})


def _svg_and_warnings(chart) -> tuple[str, list[str]]:
    """Render *chart*, returning its SVG and every warning it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = chart.to_svg()
    return svg, [str(w.message) for w in caught]


@pytest.mark.parametrize("style", [{"size": 30}, {"shape": "square"}], ids=["size", "shape"])
def test_a_replaced_points_style_does_not_reach_the_ribbon_that_follows(style: dict) -> None:
    chained_svg, chained_warnings = _svg_and_warnings(
        fm.Chart(_DF).mark_point(**style).encode(x="x", y="lo").mark_ribbon().encode(y2="hi")
    )
    bare_svg, bare_warnings = _svg_and_warnings(
        fm.Chart(_DF).mark_ribbon().encode(x="x", y="lo", y2="hi")
    )
    assert chained_warnings == []
    assert bare_warnings == []
    assert chained_svg == bare_svg


def test_a_layered_ribbons_own_style_does_not_inherit_the_lines_before_it() -> None:
    base = fm.Chart(_DF).mark_line(stroke="red", stroke_width=2).encode(x="x", y="lo")
    chained_svg, chained_warnings = _svg_and_warnings(base + base.mark_ribbon().encode(y2="hi"))
    bare_svg, bare_warnings = _svg_and_warnings(
        base + fm.Chart(_DF).mark_ribbon().encode(x="x", y="lo", y2="hi")
    )
    assert chained_warnings == []
    assert bare_warnings == []
    assert chained_svg == bare_svg


def test_a_replaced_areas_fill_does_not_paint_the_ribbon_that_follows() -> None:
    chained_svg, chained_warnings = _svg_and_warnings(
        fm.Chart(_DF).mark_area(fill="red").mark_ribbon().encode(x="x", y="lo", y2="hi")
    )
    bare_svg, bare_warnings = _svg_and_warnings(
        fm.Chart(_DF).mark_ribbon().encode(x="x", y="lo", y2="hi")
    )
    assert chained_warnings == []
    assert bare_warnings == []
    assert chained_svg == bare_svg


def test_a_replaced_points_shape_does_not_reach_the_histogram_that_follows() -> None:
    chained_svg, chained_warnings = _svg_and_warnings(
        fm.Chart(_V).mark_point(shape="square").mark_histogram().encode(x="x")
    )
    bare_svg, bare_warnings = _svg_and_warnings(fm.Chart(_V).mark_histogram().encode(x="x"))
    assert chained_warnings == []
    assert bare_warnings == []
    assert chained_svg == bare_svg


def test_a_replaced_bars_fill_does_not_paint_the_histogram_that_follows() -> None:
    chained_svg, chained_warnings = _svg_and_warnings(
        fm.Chart(_V).mark_bar(fill="green").mark_histogram(bin_count=5).encode(x="x")
    )
    bare_svg, bare_warnings = _svg_and_warnings(
        fm.Chart(_V).mark_histogram(bin_count=5).encode(x="x")
    )
    assert chained_warnings == []
    assert bare_warnings == []
    assert chained_svg == bare_svg


def test_a_replaced_bars_zero_opt_out_does_not_reach_the_histogram_that_follows() -> None:
    chained_svg, chained_warnings = _svg_and_warnings(
        fm.Chart(_V).mark_bar(zero=False).mark_histogram().encode(x="x")
    )
    bare_svg, bare_warnings = _svg_and_warnings(fm.Chart(_V).mark_histogram().encode(x="x"))
    assert chained_warnings == []
    assert bare_warnings == []
    assert chained_svg == bare_svg


def test_a_replaced_points_size_does_not_reach_the_density_that_follows() -> None:
    chained_svg, chained_warnings = _svg_and_warnings(
        fm.Chart(_V).mark_point(size=30).mark_density().encode(x="x")
    )
    bare_svg, bare_warnings = _svg_and_warnings(fm.Chart(_V).mark_density().encode(x="x"))
    assert chained_warnings == []
    assert bare_warnings == []
    assert chained_svg == bare_svg


def test_a_replaced_points_style_still_reaches_its_kept_layer_beside_a_fitted_mark() -> None:
    """The one exception: ``mark_smooth()`` keeps the primitive mark it
    replaces as its own layer (``prior_mark`` set on
    ``Chart._set_composite_mark``), so the reset above does not run and the
    primitive's own style survives on that kept layer."""
    svg, messages = _svg_and_warnings(
        fm.Chart(_DF).mark_point(fill="#ff0000").encode(x="x", y="lo").mark_smooth(method="linear")
    )
    assert messages == []
    circles = svg_elements(svg, "circle", lambda a: True)
    red_circles = svg_elements(svg, "circle", lambda a: a.get("fill") == "#ff0000")
    assert len(circles) == len(red_circles) == 8


def test_a_jointplots_regression_scatter_keeps_its_joint_kws_style() -> None:
    """``jointplot(kind="reg")`` chains ``mark_point(**joint_kws)
    .mark_smooth(...)`` on one chart (``plots/matrix.py``) -- the same
    fitted-mark exception above, reached through a figure function rather
    than a direct chained call."""
    svg, messages = _svg_and_warnings(
        fm.jointplot(_JOINT_DF, x="a", y="b", kind="reg", joint_kws={"fill": "#ff0000"})
    )
    assert messages == []
    circles = svg_elements(svg, "circle", lambda a: True)
    red_circles = svg_elements(svg, "circle", lambda a: a.get("fill") == "#ff0000")
    assert len(circles) == len(red_circles) == 6


def test_a_replaced_primitives_style_never_reached_the_mark_after_it_either() -> None:
    """The primitive-replacement control: a later primitive mark call has
    always replaced the earlier one's style (``Chart._set_mark``), so this
    reset changes nothing here. Pinned beside the composite cases above so
    a future change to either call keeps them in parity."""
    chained_svg, chained_warnings = _svg_and_warnings(
        fm.Chart(_DF).mark_point(size=30).encode(x="x", y="lo").mark_area()
    )
    bare_svg, bare_warnings = _svg_and_warnings(fm.Chart(_DF).mark_area().encode(x="x", y="lo"))
    assert chained_warnings == []
    assert bare_warnings == []
    assert chained_svg == bare_svg
