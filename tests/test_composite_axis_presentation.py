"""A layered statistic carries the caller's x/y axis presentation onto its layers.

Composite-mark desugars rebuild each layer's x/y encoding from scratch,
carrying only the field name and title (``_prep_axis_titles``), so without
help a layer would draw the DEFAULT tick format and axis config regardless
of what the caller wrote -- ``mark_smooth(ci=...)``, the bivariate
statistics, and the value axis of the categorical composites among them.
``_propagate_positional_axis``/``_propagate_layer_axis`` carry
``scale=``, ``format=``, ``format_type=`` and ``axis=`` onto a layer's own
positional channel together, unless the layer already states one of its
own (see the desugar test below).
"""

from __future__ import annotations

from typing import Any, Callable

import polars as pl
import pytest

import ferrum as fm
from ferrum.encoding import X, Y


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "w": [2.0, 3.0, 1.0, 5.0, 4.0, 2.0, 6.0, 3.0, 7.0, 4.0],
            "g": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"],
        }
    )


#: Every ``_PENDING_PREP`` kind confirmed layered on its default (or, for
#: ``smooth``, its ``ci=0.95``) build, each a ``(mark_call, channel,
#: companion)`` triple: *mark_call* builds the chart up to (not including)
#: ``.encode()``, *channel* is the positional channel this test writes
#: format=/axis= on, and *companion* is every OTHER positional binding the
#: kind needs, held fixed.
_LAYERED_KINDS: dict[str, tuple[Callable[[], Any], str, dict]] = {
    "smooth": (lambda: fm.Chart(_frame()).mark_smooth(ci=0.95), "x", {"y": "w"}),
    "smooth_show_metrics": (
        lambda: fm.Chart(_frame()).mark_smooth(ci=None, show_metrics=True),
        "x",
        {"y": "w"},
    ),
    "density_bivariate": (lambda: fm.Chart(_frame()).mark_density(), "x", {"y": "w"}),
    "contour": (lambda: fm.Chart(_frame()).mark_contour(), "x", {"y": "w"}),
    "hex": (lambda: fm.Chart(_frame()).mark_hex(), "x", {"y": "w"}),
    "qq": (lambda: fm.Chart(_frame()).mark_qq(), "x", {}),
    "boxplot_value_axis": (lambda: fm.Chart(_frame()).mark_boxplot(), "y", {"x": "g"}),
    "violin_value_axis": (lambda: fm.Chart(_frame()).mark_violin(), "y", {"x": "g"}),
    "errorbar_value_axis": (lambda: fm.Chart(_frame()).mark_errorbar(), "y", {"x": "g"}),
    "errorband_value_axis": (lambda: fm.Chart(_frame()).mark_errorband(), "y", {"x": "g"}),
    "boxen_value_axis": (lambda: fm.Chart(_frame()).mark_boxen(), "y", {"x": "g"}),
    "raster": (lambda: fm.Chart(_frame()).mark_raster(), "x", {"y": "w"}),
    "swarm_value_axis": (lambda: fm.Chart(_frame()).mark_swarm(), "y", {"x": "g"}),
}


@pytest.mark.parametrize("kind", sorted(_LAYERED_KINDS))
def test_a_layered_statistic_carries_the_callers_axis_presentation(kind):
    build, channel, companion = _LAYERED_KINDS[kind]
    cls = X if channel == "x" else Y
    field = "v" if channel == "x" else "w"

    plain = build().encode(**companion, **{channel: field}).to_svg()
    formatted = build().encode(**companion, **{channel: cls(field, format=".3f")}).to_svg()
    axis_styled = (
        build().encode(**companion, **{channel: cls(field, axis=fm.Axis(label_angle=45))}).to_svg()
    )

    assert formatted != plain, f"{kind}: format= did not reach the layer"
    assert axis_styled != plain, f"{kind}: axis= did not reach the layer"


# ---------------------------------------------------------------------------
# The model-diagnostic `_PENDING_PREP` kinds (calibration, cv_scores,
# discrimination_threshold) carry the same presentation, through their
# figure functions' own `encode=` kwarg -- these build and resolve a
# pending composite mark internally rather than returning one a test can
# call `.encode()` on before resolution, so they take `encode=` at the
# call instead of `_LAYERED_KINDS`' build-then-`.encode()` shape.
# ---------------------------------------------------------------------------


def _calibration_chart():
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    Xd, y = make_classification(n_samples=200, n_classes=2, random_state=0)
    model = LogisticRegression().fit(Xd, y)
    return lambda **encode: fm.calibration_chart(model, X=Xd, y=y, encode=encode or None)


def _cv_scores_chart():
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier

    Xd, y = make_classification(n_samples=200, n_classes=2, random_state=0)
    model = RandomForestClassifier(random_state=0, n_estimators=5)
    return lambda **encode: fm.cv_scores_chart(
        model, X=Xd, y=y, split="train", cv=3, encode=encode or None
    )


def _discrimination_threshold_chart():
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    Xd, y = make_classification(n_samples=200, n_classes=2, random_state=0)
    model = LogisticRegression().fit(Xd, y)
    return lambda **encode: fm.discrimination_threshold_chart(
        model, X=Xd, y=y, encode=encode or None
    )


#: Each model-diagnostic kind's own (factory, channel, field) triple:
#: *factory* returns a function taking `encode=` kwargs the way the
#: figure's own `encode=` parameter does.
_MODEL_DIAGNOSTIC_KINDS: dict[str, tuple[Callable[[], Callable], str, str]] = {
    "calibration": (_calibration_chart, "x", "mean_predicted"),
    "cv_scores": (_cv_scores_chart, "y", "score"),
    "discrimination_threshold": (_discrimination_threshold_chart, "x", "threshold"),
}


@pytest.mark.parametrize("kind", sorted(_MODEL_DIAGNOSTIC_KINDS))
def test_a_model_diagnostic_kind_carries_the_callers_axis_presentation(kind):
    factory, channel, field = _MODEL_DIAGNOSTIC_KINDS[kind]
    build = factory()
    cls = X if channel == "x" else Y

    plain = build().to_svg()
    formatted = build(**{channel: cls(field, format=".3f")}).to_svg()
    axis_styled = build(**{channel: cls(field, axis=fm.Axis(label_angle=45))}).to_svg()

    assert formatted != plain, f"{kind}: format= did not reach the layer"
    assert axis_styled != plain, f"{kind}: axis= did not reach the layer"


# ---------------------------------------------------------------------------
# A desugar layer that states its own presentation keeps it.
# ---------------------------------------------------------------------------


def test_a_desugar_layers_own_format_wins_over_the_callers():
    """A layer whose own channel already sets a presentation option keeps
    it rather than taking the chart-level one, verified directly against
    ``_merge_positional_channel_axis`` since no shipped desugar states a
    caller-overridable format of its own to exercise through a public
    door."""
    from ferrum._desugar import _merge_positional_channel_axis

    layer_channel = X("x", format=".1f")  # the layer's own, already-set format
    chart_channel = X("v", format=".5f")
    merged = _merge_positional_channel_axis(layer_channel, chart_channel, "x")
    assert merged is None  # nothing to change: the layer's own format wins


# ---------------------------------------------------------------------------
# Single-mark statistics already carry every presentation option, unaffected.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda ch: fm.Chart(_frame()).mark_histogram().encode(**ch),
        lambda ch: fm.Chart(_frame()).mark_density().encode(**ch),
    ],
)
def test_single_mark_statistics_already_carry_format_unaffected(build):
    plain = build({"x": "v"}).to_svg()
    formatted = build({"x": X("v", format=".3f")}).to_svg()
    assert formatted != plain


def test_ecdf_already_carries_format_unaffected():
    plain = fm.displot(_frame(), x="v", kind="ecdf").to_svg()
    formatted = fm.displot(_frame(), x=X("v", format=".3f"), kind="ecdf").to_svg()
    assert formatted != plain


# ---------------------------------------------------------------------------
# A statistic layered beside a primitive keeps every option through the
# primitive, which establishes the axis.
# ---------------------------------------------------------------------------


def test_mark_point_mark_smooth_keeps_format_through_the_prior_primitive():
    plain = fm.Chart(_frame()).mark_point().mark_smooth(ci=None).encode(x="v", y="w").to_svg()
    formatted = (
        fm.Chart(_frame())
        .mark_point()
        .mark_smooth(ci=None)
        .encode(x=X("v", format=".3f"), y="w")
        .to_svg()
    )
    assert formatted != plain


def test_residplot_lowess_keeps_format_through_its_own_point_layer():
    plain = fm.residplot(_frame(), x="v", y="w", lowess=True).to_svg()
    formatted = fm.residplot(_frame(), x=X("v", format=".3f"), y="w", lowess=True).to_svg()
    assert formatted != plain


# ---------------------------------------------------------------------------
# No format=/format_type=/axis= on x/y leaves the layer byte-identical,
# pinned directly against the propagation helper (the full suite + goldens
# are the broader negative control).
# ---------------------------------------------------------------------------


def test_no_presentation_options_leaves_the_layer_encoding_unchanged():
    from ferrum._desugar import _propagate_positional_axis

    layer_encoding = {"x": X("x", title="v"), "y": Y("y", title="w")}
    chart_encoding = {"x": X("v"), "y": Y("w")}
    assert _propagate_positional_axis(chart_encoding, layer_encoding) is layer_encoding


# ---------------------------------------------------------------------------
# format_type= propagates alongside format=/axis=.
# ---------------------------------------------------------------------------


def test_format_type_propagates_onto_a_layered_statistic():
    plain = fm.Chart(_frame()).mark_smooth(ci=0.95).encode(x="v", y="w").to_svg()
    styled = (
        fm.Chart(_frame())
        .mark_smooth(ci=0.95)
        .encode(x=X("v", format=".0%", format_type="percent"), y="w")
        .to_svg()
    )
    assert styled != plain


def test_format_types_camelcase_alias_propagates_the_same_way():
    """``formatType`` (the Vega-compat alias) reaches the layer the same
    way ``format_type`` does -- a channel's own ``_kwargs`` keeps whichever
    spelling the caller wrote, so propagating only ``format_type`` would
    silently drop a ``formatType=``."""
    snake_case = (
        fm.Chart(_frame())
        .mark_smooth(ci=0.95)
        .encode(x=X("v", format=".0%", format_type="percent"), y="w")
        .to_svg()
    )
    camel_case = (
        fm.Chart(_frame())
        .mark_smooth(ci=0.95)
        .encode(x=X("v", format=".0%", formatType="percent"), y="w")
        .to_svg()
    )
    assert camel_case == snake_case
