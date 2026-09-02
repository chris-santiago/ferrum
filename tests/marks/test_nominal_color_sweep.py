"""Regression pins: the second wave of raw ``color_field`` bindings the
Batch-A design-review Cycle 2 found un-swept onto
:func:`ferrum.marks._desugar_helpers.nominal_color_channel`.

Root cause and history: Cycle 1 of the ``fix/audit-batch-a-appearance``
design review found ``desugar_errorband`` (``marks/composite.py``) left
behind by the sweep that fixed the four ``marks/diagnostic/*`` modules (see
``tests/test_diagnostic_class_column_typing.py``). The Cycle-1 remediation
fixed *only* the named site (``tests/marks/test_errorband_nominal_color.py``)
instead of generalizing the rule -- Cycle 2 re-reviewed and found four more
named siblings (``desugar_boxplot``/``desugar_errorbar`` in
``marks/composite.py``; ``desugar_violin`` in ``marks/heavy_stat.py``) plus,
per this task's grep-complete audit obligation, three more the reviewer had
not named: ``desugar_silhouette`` (``marks/diagnostic/_clustering.py``),
``desugar_class_prediction_error`` (``marks/diagnostic/_classification.py``),
``desugar_rank1d`` (``marks/diagnostic/_ranking.py``), and
``desugar_importance``/``desugar_shap_bar``
(``marks/diagnostic/_explanation.py``).

Every one of these binds a caller-named group/class discriminator field as a
bare string onto a mark that groups, stacks, or fills rows by that field
(``rect``, ``rule``, ``tick``, ``polygon``, ``bar``). Left untyped, an Int64
discriminator column infers Quantitative -> Continuous. On ``rule``/``tick``
(errorbar) and ``polygon`` (violin body) this silently renders the wrong
picture: a continuous-blues body fill instead of the categorical palette, or
a colorbar legend with fabricated 0..1 ticks instead of a discrete swatch
legend for a two-value group column -- **with no warning**, since
``UnsupportedColorScaleOnMark`` (``prepare/legend.rs``) only fires for
line/ribbon consumers. ``bar`` and ``rect`` marks (silhouette,
class_prediction_error, rank1d, importance, shap_bar) were previously
believed safe on the reasoning "the warning doesn't fire for these marks"
(``.sdd/task-5c-report.md``) -- that reasoning conflates "does not warn"
with "renders correctly"; direct probing (Int64 vs. Utf8 same-shape
comparison) proves it renders a continuous colorbar for these marks too,
identically to the violin/errorbar defect.

Point-mark sites (``desugar_violin``'s ``inner="point"`` overlay,
``desugar_residuals``, ``desugar_prediction_error``,
``desugar_intercluster_distance``) are deliberately NOT included in this
sweep: a point layer scatters one mark per original row rather than per
aggregated group, so a genuinely continuous color field is a legitimate
reading there (verified empirically to render a sensible gradient, not a
misleading legend) -- see ``nominal_color_channel``'s docstring for the
stated rule and its point-mark carve-out.
"""

from __future__ import annotations

import re

import polars as pl

import ferrum
from tests._hue_probe import has_colorbar, legend_labels, render

# The swatch regex, the render-and-capture-warnings dance and the colorbar
# probe live in tests/_hue_probe.py -- shared with
# tests/test_diagnostic_class_column_typing.py and
# tests/test_figure_hue_typing.py, which assert the same invariant at the
# diagnostic-desugar and figure-function layers respectively.
_legend_labels = legend_labels
_has_colorbar = has_colorbar


# --- desugar_violin (marks/heavy_stat.py) — the design reviewer's exact repro ---


def _violin_df(color_dtype) -> pl.DataFrame:
    """Build with Int64 group labels (0, 1), then cast to *color_dtype* --
    so a Utf8 render's legend reads "0"/"1" too, and the two renders are
    directly comparable label-for-label, not just count-for-count.
    """
    rows = []
    for cat in ("x", "y"):
        for label in (0, 1):
            base = 0.0 if label == 0 else 5.0
            for i in range(20):
                rows.append({"cat": cat, "grp": label, "val": base + (i % 5) * 0.3})
    return pl.DataFrame(rows).with_columns(pl.col("grp").cast(color_dtype))


def test_mark_violin_integer_color_field_renders_categorical_palette_not_blues_ramp():
    """The reviewer's exact repro: an Int64 ``color_field`` must render the
    same discrete-palette body fills as the dtype-equivalent Utf8 column --
    not a continuous blues ramp with near-invisible pale-alpha fills.
    """
    int_svg, int_caught = render(
        ferrum.Chart(_violin_df(pl.Int64))
        .encode(x="cat", y="val")
        .mark_violin(color_field="grp", inner=None)
    )
    str_svg, str_caught = render(
        ferrum.Chart(_violin_df(pl.Utf8))
        .encode(x="cat", y="val")
        .mark_violin(color_field="grp", inner=None)
    )
    assert not int_caught
    assert not str_caught
    assert not _has_colorbar(int_svg), "violin body must not fall back to a continuous colorbar"
    assert _legend_labels(int_svg) == _legend_labels(str_svg) == ["0", "1"]

    def _fills(svg: str) -> set[str]:
        return set(re.findall(r'fill="(rgba?\([^)]*\)|#[0-9a-fA-F]{3,8})"', svg))

    # Both dtypes must land on the same two categorical-palette colors
    # (module-scoped 0.5 fill_opacity, so rgba(...) form).
    assert _fills(int_svg) & {"rgba(37,99,235,0.349)", "rgba(220,38,38,0.349)"}
    assert _fills(int_svg) == _fills(str_svg)


def test_mark_violin_quartile_inner_integer_color_field_matches_utf8():
    """``inner="quartile"``'s rule layers (heavy_stat.py:415) bind the same
    ``color_field`` independently of the body polygon -- must also route
    through ``nominal_color_channel``.
    """
    int_svg, _ = render(
        ferrum.Chart(_violin_df(pl.Int64))
        .encode(x="cat", y="val")
        .mark_violin(color_field="grp", inner="quartile")
    )
    str_svg, _ = render(
        ferrum.Chart(_violin_df(pl.Utf8))
        .encode(x="cat", y="val")
        .mark_violin(color_field="grp", inner="quartile")
    )
    assert not _has_colorbar(int_svg)
    assert _legend_labels(int_svg) == _legend_labels(str_svg) == ["0", "1"]


# --- desugar_errorbar / desugar_boxplot (marks/composite.py) ---


def _grouped_df(color_dtype) -> pl.DataFrame:
    """Build with Int64 group labels (0, 1), then cast to *color_dtype* --
    see :func:`_violin_df` for why (label-for-label comparability).
    """
    rows = []
    for cat in ("x", "y"):
        for label in (0, 1):
            for i in range(8):
                rows.append(
                    {"cat": cat, "grp": label, "val": float(i) + (5.0 if label == 1 else 0.0)}
                )
    return pl.DataFrame(rows).with_columns(pl.col("grp").cast(color_dtype))


def test_mark_errorbar_integer_color_field_renders_categorical_legend_not_colorbar():
    """The reviewer's exact repro: an Int64 ``color_field`` must render a
    discrete two-entry swatch legend, not a continuous colorbar with
    fabricated 0/0.25/0.5/0.75/1 ticks for a two-value group column.
    """
    int_svg, int_caught = render(
        ferrum.Chart(_grouped_df(pl.Int64))
        .encode(x="cat", y="val")
        .mark_errorbar(color_field="grp")
    )
    str_svg, str_caught = render(
        ferrum.Chart(_grouped_df(pl.Utf8)).encode(x="cat", y="val").mark_errorbar(color_field="grp")
    )
    assert not int_caught
    assert not str_caught
    assert not _has_colorbar(int_svg), "errorbar must not fall back to a continuous colorbar legend"
    assert _legend_labels(int_svg) == _legend_labels(str_svg) == ["0", "1"]


def test_mark_boxplot_integer_color_field_matches_utf8():
    """``desugar_boxplot``'s rect/rule/tick layers (composite.py:257) all
    share the same ``enc()`` closure -- one fix routes all of them.
    """
    int_svg, int_caught = render(
        ferrum.Chart(_grouped_df(pl.Int64))
        .encode(x="cat", y="val")
        .mark_boxplot(color_field="grp", outliers=False)
    )
    str_svg, str_caught = render(
        ferrum.Chart(_grouped_df(pl.Utf8))
        .encode(x="cat", y="val")
        .mark_boxplot(color_field="grp", outliers=False)
    )
    assert not int_caught
    assert not str_caught
    assert not _has_colorbar(int_svg)
    assert _legend_labels(int_svg) == _legend_labels(str_svg) == ["0", "1"]


# --- desugar_silhouette (marks/diagnostic/_clustering.py) ---


def test_mark_silhouette_integer_color_field_no_colorbar():
    df_int = pl.DataFrame(
        {
            "y_position": [0, 1, 2, 3],
            "silhouette_value": [0.2, 0.5, -0.1, 0.7],
            "cluster": [0, 0, 1, 1],
        }
    )
    df_str = df_int.with_columns(pl.col("cluster").cast(pl.Utf8))
    int_svg, int_caught = render(ferrum.Chart(df_int).mark_silhouette(zero_line=False))
    str_svg, str_caught = render(ferrum.Chart(df_str).mark_silhouette(zero_line=False))
    assert not int_caught
    assert not str_caught
    assert not _has_colorbar(int_svg)
    assert _legend_labels(int_svg) == _legend_labels(str_svg) == ["0", "1"]


# --- desugar_class_prediction_error (marks/diagnostic/_classification.py) ---


def test_mark_class_prediction_error_integer_predicted_column_no_colorbar():
    """``color_field`` defaults to ``"predicted"`` -- a class-id column that
    is entirely legal as Int64 (the raw mark's documented data contract).
    """
    df_int = pl.DataFrame(
        {
            "actual": ["a", "a", "b", "b", "a", "b"],
            "predicted": [0, 1, 0, 1, 0, 1],
            "value": [3, 1, 2, 4, 1, 2],
        }
    )
    df_str = df_int.with_columns(pl.col("predicted").cast(pl.Utf8))
    int_svg, int_caught = render(ferrum.Chart(df_int).mark_class_prediction_error())
    str_svg, str_caught = render(ferrum.Chart(df_str).mark_class_prediction_error())
    assert not int_caught
    assert not str_caught
    assert not _has_colorbar(int_svg)
    assert _legend_labels(int_svg) == _legend_labels(str_svg) == ["0", "1"]


# --- desugar_rank1d (marks/diagnostic/_ranking.py) ---


def test_mark_rank1d_integer_color_field_override_no_colorbar():
    df_int = pl.DataFrame(
        {"feature": ["f1", "f2", "f3", "f4"], "score": [0.5, 0.3, 0.8, 0.2], "grp": [0, 1, 0, 1]}
    )
    df_str = df_int.with_columns(pl.col("grp").cast(pl.Utf8))
    int_svg, int_caught = render(ferrum.Chart(df_int).mark_rank1d(color_field="grp"))
    str_svg, str_caught = render(ferrum.Chart(df_str).mark_rank1d(color_field="grp"))
    assert not int_caught
    assert not str_caught
    assert not _has_colorbar(int_svg)
    assert _legend_labels(int_svg) == _legend_labels(str_svg) == ["0", "1"]


# --- desugar_importance / desugar_shap_bar (marks/diagnostic/_explanation.py) ---


def test_mark_importance_integer_color_field_override_no_colorbar():
    df_int = pl.DataFrame(
        {
            "feature": ["f1", "f2", "f3", "f4"],
            "importance": [0.5, 0.3, 0.8, 0.2],
            "grp": [0, 1, 0, 1],
        }
    )
    df_str = df_int.with_columns(pl.col("grp").cast(pl.Utf8))
    int_svg, int_caught = render(
        ferrum.Chart(df_int).mark_importance(color_field="grp", error_bars=False)
    )
    str_svg, str_caught = render(
        ferrum.Chart(df_str).mark_importance(color_field="grp", error_bars=False)
    )
    assert not int_caught
    assert not str_caught
    assert not _has_colorbar(int_svg)
    assert _legend_labels(int_svg) == _legend_labels(str_svg) == ["0", "1"]


def test_mark_shap_bar_integer_color_field_override_no_colorbar():
    df_int = pl.DataFrame(
        {
            "feature": ["f1", "f2", "f3", "f4"],
            "abs_mean_shap": [0.5, 0.3, 0.8, 0.2],
            "grp": [0, 1, 0, 1],
        }
    )
    df_str = df_int.with_columns(pl.col("grp").cast(pl.Utf8))
    int_svg, int_caught = render(ferrum.Chart(df_int).mark_shap_bar(color_field="grp"))
    str_svg, str_caught = render(ferrum.Chart(df_str).mark_shap_bar(color_field="grp"))
    assert not int_caught
    assert not str_caught
    assert not _has_colorbar(int_svg)
    assert _legend_labels(int_svg) == _legend_labels(str_svg) == ["0", "1"]
