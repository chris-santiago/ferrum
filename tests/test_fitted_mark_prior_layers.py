"""A fitted-layer mark (``mark_smooth``/``mark_function``) keeps everything already drawn.

``mark_smooth()`` and ``mark_function()`` are the two marks that add a fitted
line to a chart rather than replacing what the chart already draws. Every
other mark call -- primitive or composite -- replaces the chart's mark and
carries its own style, whatever was drawn before (``ferrum-spec.md``'s
`.mark_point()` entry); a primitive mark call clears a pending composite
mark the same way it replaces the chart's own mark, so
``mark_histogram().mark_point()`` draws points, not the histogram. A
fitted-layer mark is the one exception: it keeps every layer the chart
already has -- an existing layered chart's own layers, or any primitive mark
already set over its own data -- through the same ``+`` composition an
already-layered chart already took, so the prior content survives as its own
layer, with its own data and style, beside the fit. A primitive mark with no
data of its own has nothing to keep as a layer, so its style merges onto the
fitted line's instead, since there is no separate layer to keep it on. A
chart still holding a pending composite mark (``mark_histogram()``, another
``mark_smooth()``, a figure's own unresolved composite, ...) has nowhere to
keep a second one, so a fitted-layer mark called there refuses, naming the
``+`` spelling that composes the two charts instead -- always a real
``Chart`` method (``mark_histogram()``), never a figure function's own name,
even when a figure built the pending mark.

Every test here exercises the public ``Chart`` API end to end, so a
rendering-pipeline regression would show up too.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import ferrum as fm

#: Every `mark_<primitive>` method on `Chart`, mapped to the mark name it
#: resolves to (`mark_circle`/`mark_square` are `mark_point(shape=...)`
#: shorthands, so both resolve to `"point"`).
_PRIMITIVE_MARKS = {
    "point": "point",
    "line": "line",
    "circle": "point",
    "square": "point",
    "bar": "bar",
    "area": "area",
    "rule": "rule",
    "text": "text",
    "tick": "tick",
    "rect": "rect",
    "segment": "segment",
    "arc": "arc",
    "image": "image",
    "geoshape": "geoshape",
    "label": "label",
}


@pytest.fixture()
def fit_df() -> pl.DataFrame:
    return pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "y": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]})


@pytest.fixture()
def seg_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            "y": [0.0, 1.1, 1.9, 3.1, 4.0, 5.2, 5.9, 7.1, 8.0, 9.2],
            "x2": [0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1],
            "y2": [0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2],
        }
    )


@pytest.fixture()
def hist_df() -> pl.DataFrame:
    return pl.DataFrame({"x": [1.0, 1.4, 1.8, 2.1, 2.3, 2.6, 2.9, 3.2, 3.6, 4.0] * 2})


# ---------------------------------------------------------------------------
# Every primitive mark, kept as its own layer, structurally
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("primitive,resolved_mark", sorted(_PRIMITIVE_MARKS.items()))
def test_every_primitive_is_kept_as_its_own_layer_before_mark_function(
    fit_df, primitive, resolved_mark
):
    chart = (
        getattr(fm.Chart(fit_df), f"mark_{primitive}")()
        .encode(x="x", y="y")
        .mark_function(np.sin, domain=[0, 5])
    )
    assert chart._layers is not None
    assert [ly.mark for ly in chart._layers] == [resolved_mark, "line"]


@pytest.mark.parametrize("primitive,resolved_mark", sorted(_PRIMITIVE_MARKS.items()))
def test_every_primitive_is_kept_as_its_own_layer_before_mark_smooth(
    fit_df, primitive, resolved_mark
):
    chart = (
        getattr(fm.Chart(fit_df), f"mark_{primitive}")()
        .encode(x="x", y="y")
        .mark_smooth(method="lm")
    )
    resolved = chart._resolve_pending()
    assert resolved._layers is not None
    assert [ly.mark for ly in resolved._layers] == [resolved_mark, "line"]


# ---------------------------------------------------------------------------
# mark_function: rendered counts, style isolation, both encode orders
# ---------------------------------------------------------------------------


def test_point_before_mark_function_renders_every_point_plus_the_function_line(fit_df):
    chart = fm.Chart(fit_df).mark_point().encode(x="x", y="y").mark_function(np.sin, domain=[0, 5])
    svg = chart.to_svg()
    assert svg.count("<circle") == len(fit_df)
    assert "<polyline" in svg


def test_line_before_mark_function_renders_the_raw_line_plus_the_function_line(fit_df):
    chart = fm.Chart(fit_df).mark_line().encode(x="x", y="y").mark_function(np.sin, domain=[0, 5])
    svg = chart.to_svg()
    assert svg.count("<polyline") == 2


def test_segment_before_mark_function_renders_every_segment_plus_the_function_line(seg_df):
    alone = fm.Chart(seg_df).mark_segment().encode(x="x", y="y", x2="x2", y2="y2").to_svg()
    chart = (
        fm.Chart(seg_df)
        .mark_segment()
        .encode(x="x", y="y", x2="x2", y2="y2")
        .mark_function(np.sin, domain=[0, 9])
    )
    svg = chart.to_svg()
    # The function line draws as a <polyline>, never a <line>, so the
    # segment's own <line> count (its segments plus axis ticks and grid)
    # is unaffected by the function joining it as a layer.
    assert svg.count("<line") == alone.count("<line")
    assert "<polyline" in svg


def test_priors_style_and_the_function_layers_style_are_isolated(fit_df):
    chart = (
        fm.Chart(fit_df)
        .mark_point(opacity=0.3)
        .encode(x="x", y="y")
        .mark_function(np.sin, domain=[0, 5], stroke_width=4)
    )
    assert chart._layers[0].mark_kwargs == {"opacity": 0.3}
    assert chart._layers[1].mark_kwargs == {"stroke_width": 4}


def test_a_prior_lines_own_width_survives_beside_the_functions_own_width(fit_df):
    chart = (
        fm.Chart(fit_df)
        .mark_line(stroke_width=7)
        .encode(x="x", y="y")
        .mark_function(np.sin, domain=[0, 5], stroke_width=2)
    )
    assert chart._layers[0].mark_kwargs == {"stroke_width": 7}
    assert chart._layers[1].mark_kwargs == {"stroke_width": 2}


@pytest.mark.parametrize("encode_order", ["before", "after"])
def test_mark_function_keeps_the_prior_point_regardless_of_encode_order(fit_df, encode_order):
    if encode_order == "before":
        chart = (
            fm.Chart(fit_df).encode(x="x", y="y").mark_point().mark_function(np.sin, domain=[0, 5])
        )
    else:
        chart = (
            fm.Chart(fit_df).mark_point().mark_function(np.sin, domain=[0, 5]).encode(x="x", y="y")
        )
    svg = chart.to_svg()
    assert svg.count("<circle") == len(fit_df)
    assert "<polyline" in svg


def test_the_priors_title_survives_through_the_function_layers_remap(fit_df):
    chart = (
        fm.Chart(fit_df)
        .encode(x=fm.X("x", title="Xt"))
        .mark_point()
        .mark_function(np.sin, domain=[0, 5])
    )
    assert chart._layers[1].encoding["x"].option("title") == "Xt"


# ---------------------------------------------------------------------------
# mark_function: unaffected rows (no prior content drawn)
# ---------------------------------------------------------------------------


def test_mark_function_on_a_bare_encoded_chart_is_unaffected(fit_df):
    chart = fm.Chart(fit_df).encode(x="x", y="y").mark_function(np.sin, domain=[0, 5])
    assert chart._layers is None
    assert chart._mark == "line"


def test_mark_function_on_chart_none_is_unaffected():
    chart = fm.Chart(None).mark_function(np.sin, domain=[0, 6.28])
    assert chart._layers is None
    assert chart._mark == "line"


def test_mark_function_on_an_already_layered_chart_is_unaffected(fit_df):
    a = fm.Chart(fit_df).mark_point().encode(x="x", y="y")
    b = fm.Chart(fit_df).mark_line().encode(x="x", y="y")
    chart = (a + b).mark_function(np.sin, domain=[0, 5])
    assert len(chart._layers) == 3


def test_a_primitive_mark_with_no_data_of_its_own_has_nothing_to_keep():
    """A primitive mark set on a chart with no data (``Chart(None)``) carries
    no content to preserve as its own layer, so it takes the single-chart
    shortcut and merges its own style onto the function's."""
    chart = fm.Chart(None).mark_line(stroke_width=7).mark_function(np.sin, domain=[0, 5])
    assert chart._layers is None
    assert chart._mark == "line"
    assert chart._mark_kwargs.get("stroke_width") == 7


# ---------------------------------------------------------------------------
# mark_smooth: segment's own span channels, and unaffected rows
# ---------------------------------------------------------------------------


def test_segment_before_mark_smooth_renders_both_the_segment_and_the_fit(seg_df):
    alone = fm.Chart(seg_df).mark_segment().encode(x="x", y="y", x2="x2", y2="y2").to_svg()
    chart = (
        fm.Chart(seg_df)
        .mark_segment()
        .encode(x="x", y="y", x2="x2", y2="y2")
        .mark_smooth(method="lm")
    )
    svg = chart.to_svg()
    assert svg.count("<line") == alone.count("<line")
    assert "<polyline" in svg


def test_mark_smooth_on_a_prior_point_or_line_is_unaffected(fit_df):
    for prior in ("point", "line"):
        chart = (
            getattr(fm.Chart(fit_df), f"mark_{prior}")()
            .mark_smooth(method="lm")
            .encode(x="x", y="y")
        )
        resolved = chart._resolve_pending()
        assert [ly.mark for ly in resolved._layers] == [prior, "line"]


def test_mark_smooth_on_an_already_layered_chart_is_unaffected(fit_df):
    a = fm.Chart(fit_df).mark_point().encode(x="x", y="y")
    b = fm.Chart(fit_df).mark_line().encode(x="x", y="y")
    chart = (a + b).mark_smooth(method="lm")
    resolved = chart._resolve_pending()
    assert len(resolved._layers) == 3


# ---------------------------------------------------------------------------
# A primitive mark still replaces a primitive mark (fitted marks are the
# one exception, not the rule)
# ---------------------------------------------------------------------------


def test_a_primitive_mark_still_replaces_a_prior_primitive_mark(fit_df):
    chart = fm.Chart(fit_df).mark_point().mark_line()
    assert chart._mark == "line"
    assert chart._layers is None


# ---------------------------------------------------------------------------
# A primitive mark clears a pending composite mark, the same way it
# replaces the chart's own mark
# ---------------------------------------------------------------------------


def test_a_primitive_mark_after_a_pending_histogram_draws_the_primitive_alone(hist_df):
    chart = fm.Chart(hist_df).mark_histogram().mark_point().encode(x="x", y="x")
    plain = fm.Chart(hist_df).mark_point().encode(x="x", y="x")
    assert chart.to_svg() == plain.to_svg()


def test_a_primitive_mark_after_a_pending_histogram_is_auto_raster_eligible(hist_df):
    """``_composite_kind`` is the other half of the composite state a
    primitive mark call replaces (``_render.py``'s auto-raster gate skips a
    chart whose ``_composite_kind`` is still set, reading it as aggregate
    composite output rather than raw per-element marks). A chart above the
    raster threshold behaves like a fresh ``mark_point()`` chart: both
    substitute ``mark_raster``, not just the one that never called
    ``mark_histogram()`` first."""
    from ferrum.render_config import RenderConfig

    cfg = RenderConfig(raster_threshold=0, raster_behavior="silent")
    chart = (
        fm.Chart(hist_df)
        .mark_histogram()
        .mark_point()
        .encode(x="x:Q", y="x:Q")
        .properties(render_config=cfg)
    )
    plain = fm.Chart(hist_df).mark_point().encode(x="x:Q", y="x:Q").properties(render_config=cfg)
    chart_svg = chart.to_svg()
    assert "<image" in chart_svg
    assert chart_svg == plain.to_svg()


def test_mark_smooth_after_the_composite_to_primitive_chain_draws_both(hist_df):
    chart = (
        fm.Chart(hist_df)
        .mark_histogram()
        .mark_point()
        .encode(x="x", y="x")
        .mark_smooth(method="lm")
    )
    resolved = chart._resolve_pending()
    assert [ly.mark for ly in resolved._layers] == ["point", "line"]
    svg = chart.to_svg()
    assert svg.count("<circle") == len(hist_df)
    assert "<polyline" in svg


def test_mark_function_after_the_composite_to_primitive_chain_draws_both(hist_df):
    chart = (
        fm.Chart(hist_df)
        .mark_histogram()
        .mark_point()
        .encode(x="x", y="x")
        .mark_function(np.sin, domain=[0, 5])
    )
    svg = chart.to_svg()
    assert svg.count("<circle") == len(hist_df)
    assert "<polyline" in svg


# ---------------------------------------------------------------------------
# A fitted-layer mark refuses a pending composite mark on the same chart
# ---------------------------------------------------------------------------


def test_mark_function_refuses_after_a_pending_histogram(hist_df):
    chart = fm.Chart(hist_df).encode(x="x").mark_histogram()
    with pytest.raises(ValueError, match=r"mark_function\(\) cannot follow mark_histogram\(\)"):
        chart.mark_function(np.sin, domain=[0, 5])


def test_mark_function_refuses_after_a_pending_smooth(fit_df):
    chart = fm.Chart(fit_df).encode(x="x", y="y").mark_smooth()
    with pytest.raises(ValueError, match=r"mark_function\(\) cannot follow mark_smooth\(\)"):
        chart.mark_function(np.sin, domain=[0, 5])


def test_mark_function_refuses_after_a_pending_smooth_with_ci(fit_df):
    chart = fm.Chart(fit_df).encode(x="x", y="y").mark_smooth(ci=0.95)
    with pytest.raises(ValueError, match=r"mark_function\(\) cannot follow mark_smooth\(\)"):
        chart.mark_function(np.sin, domain=[0, 5])


def test_mark_smooth_refuses_after_a_pending_histogram(hist_df):
    chart = fm.Chart(hist_df).encode(x="x").mark_histogram()
    with pytest.raises(ValueError, match=r"mark_smooth\(\) cannot follow mark_histogram\(\)"):
        chart.mark_smooth()


def test_mark_smooth_refuses_after_a_pending_smooth(fit_df):
    chart = fm.Chart(fit_df).encode(x="x", y="y").mark_smooth()
    with pytest.raises(ValueError, match=r"mark_smooth\(\) cannot follow mark_smooth\(\)"):
        chart.mark_smooth()


def test_the_refusal_names_the_plus_spelling_that_composes_the_two_calls(hist_df):
    chart = fm.Chart(hist_df).encode(x="x").mark_histogram()
    with pytest.raises(ValueError) as excinfo:
        chart.mark_function(np.sin, domain=[0, 5])
    message = str(excinfo.value)
    assert "chart.mark_histogram() + chart.mark_function(...)" in message


# ---------------------------------------------------------------------------
# A figure's own pending composite (displot) refuses the same way, and the
# "+" remedy names a real Chart method, never the figure's own name
# ---------------------------------------------------------------------------


def test_mark_function_refuses_after_displots_pending_histogram(hist_df):
    chart = fm.displot(hist_df, x="x")
    with pytest.raises(ValueError) as excinfo:
        chart.mark_function(np.sin, domain=[0, 5])
    message = str(excinfo.value)
    assert "mark_function() cannot follow mark_histogram()" in message
    assert "built by displot()" in message
    assert "chart.mark_histogram() + chart.mark_function(...)" in message
    assert "chart.displot()" not in message


def test_mark_smooth_refuses_after_displots_pending_histogram(hist_df):
    chart = fm.displot(hist_df, x="x")
    with pytest.raises(ValueError) as excinfo:
        chart.mark_smooth()
    message = str(excinfo.value)
    assert "mark_smooth() cannot follow mark_histogram()" in message
    assert "built by displot()" in message
    assert "chart.mark_histogram() + chart.mark_smooth(...)" in message
    assert "chart.displot()" not in message
