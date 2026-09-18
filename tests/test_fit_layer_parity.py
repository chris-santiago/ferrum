"""The fit-layer triple is one shape, whoever builds it.

`mark_smooth` (via `desugar_smooth`) and `lmplot(method="logistic"/"glm"/
"robust")` (via `_fit_chart_from_named_transform`) both emit a ribbon + line
(+ metrics) triple over the shared `x, y, ci_lower, ci_upper` fit schema. The
two were independent code paths carrying the same magic constants -- the CI
band's `opacity: 0.3`, the metrics text's `align`/`dx`/`dy` -- so their
agreement held by coincidence of copied literals, with nothing failing if one
drifted.

Written BEFORE `marks/_fit_layers.build_fit_layers` hoisted the shape (both
design reviews' "missing coverage that would de-risk the hoist"), so it pins
the pre-hoist agreement as much as the post-hoist one: the layers are compared
against each other, not against a restatement of the constants.

Since #199 the shape includes the caller's own column names, bound as the
layers' axis titles. That makes each comparison sensitive to which columns its
two charts were built over, so both sides read `_y_column` for the response
rather than hard-coding one each.
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm

_FIT_METHODS = ["logistic", "glm", "robust"]


def _frame(n: int = 60) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": [float(i) for i in range(n)],
            "y": [float(i) * 1.5 + (i % 7) for i in range(n)],
            "p": [float(i % 2) for i in range(n)],
        }
    )


def _by_name(layers) -> dict[str, object]:
    return {layer.name: layer for layer in layers}


def _y_column(method: str) -> str:
    """The response column a given fit method can be run against.

    ``logistic`` needs a 0/1 response, the rest take the continuous one. Both
    helpers below read it, so each comparison builds its two charts over the
    SAME caller columns -- load-bearing since #199: the layers now carry those
    column names as axis titles, so comparing a `y`-response smooth against a
    `p`-response lmplot would fail on the titles alone, which is a difference
    between the two fixtures rather than between the two code paths.
    """
    return "p" if method == "logistic" else "y"


def _smooth_layers(method: str = "lm", **kwargs):
    y = _y_column(method)
    chart = fm.Chart(_frame()).mark_smooth(method="lm", **kwargs).encode(x="x", y=y)
    return _by_name(chart._resolve_pending()._layers)


def _lmplot_layers(method: str, **kwargs):
    return _by_name(
        fm.lmplot(_frame(), x="x", y=_y_column(method), method=method, **kwargs)._layers
    )


@pytest.mark.parametrize("method", _FIT_METHODS)
def test_the_ci_ribbon_is_the_same_layer_on_both_paths(method):
    """Same mark, same band opacity, same channel roles over the registry's
    fit schema -- compared layer against layer, so a change to one path that
    is not made to the other fails here."""
    smooth = _smooth_layers(method, ci=0.95)["ribbon"]
    figure = _lmplot_layers(method)["ribbon"]

    assert figure.mark == smooth.mark == "ribbon"
    assert figure.mark_kwargs == smooth.mark_kwargs
    assert set(figure.encoding) == set(smooth.encoding) == {"x", "y", "y2"}
    assert figure.encoding == smooth.encoding


@pytest.mark.parametrize("method", _FIT_METHODS)
def test_the_fit_line_is_the_same_layer_on_both_paths(method):
    smooth = _smooth_layers(method, ci=0.95)["line"]
    figure = _lmplot_layers(method)["line"]

    assert figure.mark == smooth.mark == "line"
    assert figure.mark_kwargs == smooth.mark_kwargs
    assert figure.encoding == smooth.encoding


def test_the_metrics_corner_is_the_same_layer_on_both_paths():
    """`robust` is the only dedicated fit method that emits metrics
    (`logistic`/`glm` expose no `inject_metrics`), so the offsets are pinned
    against `mark_smooth`'s through it."""
    smooth = _smooth_layers("robust", ci=0.95, show_metrics=True)["metrics"]
    figure = _lmplot_layers("robust", show_metrics=True)["metrics"]

    assert figure.mark == smooth.mark == "text"
    assert figure.mark_kwargs == smooth.mark_kwargs
    assert figure.encoding == smooth.encoding


@pytest.mark.parametrize("method", _FIT_METHODS)
def test_ci_none_drops_the_ribbon_on_both_paths(method):
    assert "ribbon" not in _smooth_layers(method, ci=None, show_metrics=True)
    assert "ribbon" not in _lmplot_layers(method, ci=None)


def test_line_kws_reach_the_fit_line_on_both_paths():
    """`line_kws` reaches the fit line, on the smooth-routed methods and the
    dedicated ones alike.

    Both now forward it into the fit-layer builder at construction. The
    smooth-routed methods used to take a post-pass over the finished chart's
    `_layers` instead, which is why `scatter=False` -- whose layers are not
    materialized until render -- dropped the styling entirely (#199).
    """
    smooth_routed = _lmplot_layers("lm", line_kws={"stroke_width": 3})["line"]
    dedicated = _lmplot_layers("robust", line_kws={"stroke_width": 3})["line"]
    assert smooth_routed.mark_kwargs["stroke_width"] == 3
    assert dedicated.mark_kwargs["stroke_width"] == 3
