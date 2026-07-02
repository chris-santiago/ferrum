"""Regression coverage for the scale-through-desugar propagation rule.

Design spec §6 (GH #45 prerequisite): a user-set explicit ``scale=`` on a
composite-mark chart's positional channel (e.g.
``Chart(df).mark_cv_scores(...).encode(y=fm.Y("score", scale=fm.LinearScale(...)))``)
is silently dropped during composite-mark desugar, because desugar rebuilds
each layer's encoding from scratch carrying only field names/titles. The fix
propagates the chart-level explicit x/y scale onto the desugared layers'
matching positional channel at the pending-mark resolution seam
(``ferrum._desugar._resolve_pending_impl``), generically across all
composite marks.

Propagation rule (exact, spec §6):

- layer channel has no scale -> attach the chart-level scale;
- layer channel has a scale without a ``domain`` (e.g. validation_curve's
  ``{"type": "log"}``) -> merge in the chart-level domain only, never
  overwriting ``type``/``range``/other keys;
- layer channel already has a ``domain`` (e.g. SHAP ``x_scale_domain``) ->
  leave untouched, mark-computed domain wins.

Only positional ``x``/``y`` channels are in scope; ``x2``/``y2`` companions
never carry their own scale (``SECONDARY_EXTENT`` honors no ``scale`` kwarg)
so they are not propagation targets — the primary channel's scale is what the
Rust axis-scale resolver consults (``render/scale_resolve/positional.rs``).
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm

from tests._svg_extents import y_axis_extents


def _cv_scores_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "fold": [0, 1, 2, 3, 4] * 2,
            "split": ["train"] * 5 + ["test"] * 5,
            "score": [0.10, 0.12, 0.11, 0.13, 0.09, 0.50, 0.55, 0.52, 0.51, 0.53],
        }
    )


class TestExplicitScaleSurvivesDesugar:
    """Rendered-axis assertions: a chart-level explicit domain must take effect."""

    def test_cv_scores_explicit_y_domain_renders(self) -> None:
        df = _cv_scores_df()
        svg = (
            fm.Chart(df)
            .mark_cv_scores(kind="box")
            .encode(y=fm.Y("score", scale=fm.LinearScale(domain=(0.0, 2.0))))
            .to_svg()
        )
        extents = y_axis_extents(svg)
        assert extents, "expected at least one y-axis tick column"
        lo, hi = extents[0]
        # Data range is ~[0.09, 0.55]; the auto-inferred domain would top out
        # well under 1.0. Only the explicit (0, 2) domain surviving desugar
        # would push the rendered extent this high.
        assert hi >= 1.5, (
            f"y-axis extent {(lo, hi)} does not reflect the explicit domain (0, 2); "
            "the chart-level scale was dropped during composite-mark desugar."
        )

    def test_boxplot_explicit_y_domain_renders(self) -> None:
        df = pl.DataFrame(
            {
                "cat": ["a", "a", "a", "b", "b", "b"],
                "val": [1.0, 2.0, 1.5, 4.0, 5.0, 4.5],
            }
        )
        svg = (
            fm.Chart(df)
            .mark_boxplot()
            .encode(x="cat:N", y=fm.Y("val", scale=fm.LinearScale(domain=(0.0, 20.0))))
            .to_svg()
        )
        extents = y_axis_extents(svg)
        assert extents, "expected at least one y-axis tick column"
        lo, hi = extents[0]
        # Data range is ~[1, 5]; only the explicit (0, 20) domain surviving
        # desugar would push the rendered extent this high.
        assert hi >= 15.0, (
            f"y-axis extent {(lo, hi)} does not reflect the explicit domain (0, 20); "
            "the chart-level scale was dropped during composite-mark desugar."
        )


class TestLogScaleDomainMerge:
    """A layer-authored scale without a domain merges in the chart-level domain
    only, never overwriting its own ``type``."""

    def test_validation_curve_log_scale_keeps_type_when_domain_merges(self) -> None:
        df = pl.DataFrame(
            {
                "param_value": [0.001, 0.01, 0.1, 1.0],
                "split": ["train", "train", "test", "test"],
                "mean_score": [0.5, 0.6, 0.55, 0.65],
                "lower": [0.4, 0.5, 0.45, 0.55],
                "upper": [0.6, 0.7, 0.65, 0.75],
            }
        )
        chart = (
            fm.Chart(df)
            .mark_validation_curve(log_scale=True)
            .encode(x=fm.X("param_value", scale=fm.LinearScale(domain=(1e-4, 10.0))))
        )
        resolved = chart._resolve_pending()
        assert resolved._layers is not None
        band_layer = next(ly for ly in resolved._layers if ly.name == "band")
        x_channel = band_layer.encoding["x"]
        scale = x_channel.option("scale")
        assert scale is not None, "explicit chart-level scale was dropped entirely"
        assert scale.get("type") == "log", (
            f"log_scale=True's 'type': 'log' was overwritten during domain merge: {scale!r}"
        )
        assert list(scale.get("domain")) == pytest.approx([1e-4, 10.0]), (
            f"chart-level domain was not merged into the layer's log scale: {scale!r}"
        )


class TestMarkComputedDomainWins:
    """A desugar-authored domain (e.g. SHAP's ``x_scale_domain``) must not be
    overwritten by a chart-level explicit scale."""

    def test_shap_bar_x_scale_domain_untouched_by_chart_level_scale(self) -> None:
        df = pl.DataFrame(
            {
                "feature": ["a", "b", "c"],
                "abs_mean_shap": [0.1, 0.3, 0.2],
            }
        )
        chart = (
            fm.Chart(df)
            .mark_shap_bar(x_scale_domain=(0.0, 0.5))
            .encode(x=fm.X("abs_mean_shap", scale=fm.LinearScale(domain=(-100.0, 100.0))))
        )
        resolved = chart._resolve_pending()
        assert resolved._layers is not None
        bar_layer = next(ly for ly in resolved._layers if ly.name == "bar")
        x_channel = bar_layer.encoding["x"]
        scale = x_channel.option("scale")
        assert scale is not None
        assert list(scale.get("domain")) == pytest.approx([0.0, 0.5]), (
            f"mark-computed x_scale_domain was overwritten by the chart-level scale: {scale!r}"
        )


class TestNoChartLevelScaleUnaffected:
    """A composite mark with no chart-level explicit positional scale must
    render exactly as before (auto-inferred domain, deterministic output)."""

    def test_cv_scores_without_scale_renders_deterministically(self) -> None:
        df = _cv_scores_df()
        svg1 = fm.Chart(df).mark_cv_scores(kind="box").encode(y="score").to_svg()
        svg2 = fm.Chart(df).mark_cv_scores(kind="box").encode(y="score").to_svg()
        assert svg1 == svg2

        extents = y_axis_extents(svg1)
        assert extents
        lo, hi = extents[0]
        # Auto-inferred domain must track the actual data (~[0.09, 0.55]), not
        # some spuriously-propagated large domain from an unrelated chart.
        assert hi < 0.9, (
            f"unscaled y-axis extent {(lo, hi)} suggests a scale leaked in from "
            "somewhere other than this chart's own encoding."
        )
