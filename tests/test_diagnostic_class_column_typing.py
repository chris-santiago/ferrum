"""Net-new coverage: diagnostic desugars must bind their class/group
discriminator column to the color channel as Nominal, not leave it
untyped.

Root cause (fixed via ``ferrum.marks._desugar_helpers.nominal_color_channel``,
consumed by ``ferrum.marks.diagnostic._classification``, ``._explanation``,
``._selection``, and ``._ranking``): these desugars bound their
``color_field`` as a bare string (``line_enc["color"] = color_field``), so
the color channel's data type was left for Rust to infer from the column's
runtime dtype. A Utf8 discriminator column infers Nominal (correct), but an
Int64 column -- entirely legal input, since these are raw marks a caller
can invoke directly on any DataFrame carrying the documented column names
or pass an arbitrary ``color_field=`` override to -- infers Quantitative,
which resolves to a Continuous color scale. A Continuous scale is inert on
a line/ribbon mark, so the render pipeline's ``UnsupportedColorScaleOnMark``
warning fires and the colorbar (which nothing on the chart could honor
anyway) is suppressed, silently dropping the per-group legend (and on some
marks collapsing the per-group polylines into one).

Each desugar now binds its color/group field through
``nominal_color_channel`` (``Color(field, type_="nominal")``) so the
discriminator column resolves a Categorical scale regardless of its
runtime dtype -- correctly coloring each group's polyline/band and keeping
a symbol legend.

See ``tests/test_finding_p9.py::test_mark_roc_average_integer_class_column_does_not_raise``
for the sibling ``average=`` filter-path regression this shares a root
column (``class``) with. Desugar sweep this file exercises:

- ``_classification.py``: ``desugar_roc``, ``desugar_pr``, ``desugar_gain``,
  ``desugar_lift``, ``desugar_calibration`` (default ``color_field``
  binding -- the initial scope of this fix).
- ``_explanation.py``: ``desugar_pdp`` -- widened into scope (spec-review
  round 1) because a caller-supplied numeric ``color_field`` override hits
  the identical defect on the documented Utf8 ``"feature"`` default. All
  three ``kind`` values' ``color_field``-bound layers are exercised:
  ``kind="average"``'s ``line``, ``kind="individual"``'s ``ice``
  (+ ``mark_style.detail`` splitting), and ``kind="both"``'s ``ice`` and
  ``average`` layers (quality-review round 2 -- round 1 covered
  ``kind="average"`` only, while the module docstring already claimed all
  three; this file's tests and this docstring are now in sync).
- ``_selection.py``: ``desugar_learning_curve``, ``desugar_validation_curve``
  -- widened for the same reason (default ``color_field="split"`` is
  Utf8-safe, but a caller override is not). Both public marks
  (``mark_learning_curve`` and ``mark_validation_curve``, which share the
  same desugar shape but are distinct entry points) and both
  ``ci_style`` branches (``"band"``'s ``ribbon`` and ``"errorbar"``'s
  ``rule``) are exercised (quality-review round 2 -- round 1 covered only
  ``mark_learning_curve``'s default ``ci_style="band"``; see
  ``test_mark_learning_curve_errorbar_integer_color_field_renders_expected_shape``'s
  docstring for why the errorbar branch needed shape assertions rather than
  warning-absence).
- ``_ranking.py``: ``desugar_parallel_coordinates`` -- widened for the same
  reason (``color_field`` defaults to ``None``, opt-in only, but any
  caller-supplied override was unguarded).

``desugar_discrimination_threshold`` (color="metric", always Utf8 by melt
construction) is genuinely safe and not touched. ``desugar_confusion`` and
``desugar_rank2d`` (rect marks colored by a genuinely continuous quantity --
cell count, correlation coefficient -- already explicitly wrapped in
``Color(...)`` at their call sites) and ``desugar_decision_boundary`` /
``desugar_intercluster_distance`` (rect/point, intentionally continuous or a
point-mark scatter reading unsplit per-row data) are also unaffected by this
class of bug and not touched. See ``.sdd/task-5c-report.md`` for that
original per-site reasoning -- **but note its claim that
``desugar_class_prediction_error`` and ``desugar_silhouette`` (rect/bar
marks) were "safe because the inert-color warning only fires for
line/ribbon" was wrong**: "does not warn" is not "renders correctly", and
both were found rendering a continuous colorbar for an Int64 discriminator
column, with no warning at all, in the Batch-A design-review Cycle 2 sweep.
Both are now fixed the same way as this file's sites, pinned in
``tests/marks/test_nominal_color_sweep.py`` alongside the rest of that
cycle's fixes (``desugar_boxplot``/``desugar_errorbar`` in
``marks/composite.py``, ``desugar_violin`` in ``marks/heavy_stat.py``,
``desugar_rank1d``/``desugar_importance``/``desugar_shap_bar``, which share
this file's ``marks/diagnostic/*`` scope but are pinned in that sibling file
per the repo's findings-scoped test-file convention, since they belong to a
distinct finding/task rather than this file's original scope).

Byte-identity scope (spec-review round 1, adjudication a): the bare
``Chart(df).mark_*()`` path -- these desugars build a layered chart with
no chart-level ``.encode()`` call -- renders byte-identical for a
Utf8-typed discriminator column, pinned by
``test_mark_roc_string_class_column_bare_path_byte_identical`` below via a
monkeypatch that reverts ``nominal_color_channel`` to a bare-string
passthrough. The **figure** path (``roc_chart``/``pr_chart``, which layers
an additional chart-level ``.encode(x=, y=)`` after the mark call) is
explicitly NOT byte-identical: it now renders a legend TITLE
(``<text ...>class</text>``) that the pre-fix bare-string binding did not
produce. This is a sanctioned, deliberate behavior change -- a legend
having its title is correct; the title's prior absence was an artifact of
the untyped bare-string binding, not an intentional design choice -- not a
regression to suppress. See the 5 ``pr_chart_*``/5 ``roc_chart_*`` golden
movers in ``.sdd/task-5c-report.md``.
"""

from __future__ import annotations

import re
import warnings

import polars as pl

import ferrum
from tests._hue_probe import legend_labels
from tests.fixtures import load_dataset, load_fixture

# The categorical-legend swatch regex and its accessor live in
# tests/_hue_probe.py: this module, tests/marks/test_nominal_color_sweep.py and
# tests/test_figure_hue_typing.py all assert the same "a discriminator renders a
# swatch legend, not a colorbar" invariant, and had each grown their own copy.
_legend_labels = legend_labels


def _stroke_hex(svg_element: str) -> str:
    """Return the ``stroke="#..."`` hex value of one SVG element.

    Raises via ``assert`` (not a silent ``None``) when the element carries
    no ``stroke`` attribute, so a malformed/unexpected element fails loudly
    at the call site instead of propagating a ``None`` into ``.group()``.
    """
    m = re.search(r'stroke="(#[0-9a-fA-F]+)"', svg_element)
    assert m is not None, f"element has no stroke attribute: {svg_element!r}"
    return m.group(1)


def _assert_no_user_warning(fn) -> str:
    """Run *fn* (a zero-arg callable returning an SVG string) under
    ``error::UserWarning`` and return the SVG. Fails loudly if any
    ``UserWarning`` (including ``UnsupportedColorScaleOnMark``) fires.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        return fn()


def test_mark_roc_integer_class_column_no_warning_and_symbol_legend():
    df = pl.DataFrame(
        {
            "fpr": [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
            "tpr": [0.0, 0.6, 1.0, 0.0, 0.7, 1.0],
            "class": [0, 0, 0, 1, 1, 1],
        }
    )
    chart = ferrum.Chart(df).mark_roc(reference_line=False)
    svg = _assert_no_user_warning(chart.to_svg)

    assert "<svg" in svg
    # One <polyline> per class, colored via a Nominal (categorical) scale.
    assert svg.count("<polyline") == 2
    # Symbol legend: one swatch label per class value -- matched via the
    # swatch-anchored regex so this cannot be satisfied by axis tick text.
    assert _legend_labels(svg) == ["0", "1"], f"got legend entries {_legend_labels(svg)}"


def test_mark_pr_integer_class_column_no_warning_and_symbol_legend():
    df = pl.DataFrame(
        {
            "recall": [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
            "precision": [1.0, 0.8, 0.5, 1.0, 0.9, 0.6],
            "class": [0, 0, 0, 1, 1, 1],
        }
    )
    chart = ferrum.Chart(df).mark_pr()
    svg = _assert_no_user_warning(chart.to_svg)

    assert "<svg" in svg
    assert svg.count("<polyline") == 2
    assert _legend_labels(svg) == ["0", "1"], f"got legend entries {_legend_labels(svg)}"


def test_mark_gain_integer_class_column_no_warning_and_symbol_legend():
    df = pl.DataFrame(
        {
            "percent_population": [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
            "gain": [0.0, 0.6, 1.0, 0.0, 0.7, 1.0],
            "class": [0, 0, 0, 1, 1, 1],
        }
    )
    chart = ferrum.Chart(df).mark_gain(reference_line=False)
    svg = _assert_no_user_warning(chart.to_svg)

    assert "<svg" in svg
    assert svg.count("<polyline") == 2
    assert _legend_labels(svg) == ["0", "1"], f"got legend entries {_legend_labels(svg)}"


def test_mark_lift_integer_class_column_no_warning_and_symbol_legend():
    df = pl.DataFrame(
        {
            "percent_population": [0.2, 0.5, 1.0, 0.2, 0.5, 1.0],
            "lift": [2.0, 1.5, 1.0, 1.8, 1.4, 1.0],
            "class": [0, 0, 0, 1, 1, 1],
        }
    )
    chart = ferrum.Chart(df).mark_lift(reference_line=False)
    svg = _assert_no_user_warning(chart.to_svg)

    assert "<svg" in svg
    assert svg.count("<polyline") == 2
    assert _legend_labels(svg) == ["0", "1"], f"got legend entries {_legend_labels(svg)}"


def test_mark_calibration_integer_color_field_no_warning_and_symbol_legend():
    """``color_field`` is opt-in (``None`` by default) on ``mark_calibration``,
    used by multi-model compare paths. Passing an integer model-id column
    (rather than a Utf8 model name) must not trip the inert-color warning.
    """
    df = pl.DataFrame(
        {
            "mean_predicted": [0.1, 0.5, 0.9, 0.1, 0.5, 0.9],
            "fraction_positive": [0.15, 0.45, 0.85, 0.05, 0.55, 0.95],
            "model": [0, 0, 0, 1, 1, 1],
        }
    )
    chart = ferrum.Chart(df).mark_calibration(reference_line=False, color_field="model")
    svg = _assert_no_user_warning(chart.to_svg)

    assert "<svg" in svg
    assert svg.count("<polyline") == 2
    assert _legend_labels(svg) == ["0", "1"], f"got legend entries {_legend_labels(svg)}"


def test_mark_pdp_integer_color_field_no_warning_and_symbol_legend():
    """``mark_pdp``'s default ``color_field="feature"`` is Utf8-safe by data
    contract, but the caller can override it with any column -- including
    one that happens to be Int64-typed (spec-review round 1 scoping call).
    """
    df = pl.DataFrame(
        {
            "feature": ["a", "a", "a", "b", "b", "b"],
            "feature_value": [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
            "pd_value": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "group": [0, 0, 0, 1, 1, 1],
        }
    )
    chart = ferrum.Chart(df).mark_pdp(color_field="group")
    svg = _assert_no_user_warning(chart.to_svg)

    assert "<svg" in svg
    assert svg.count("<polyline") == 2
    assert _legend_labels(svg) == ["0", "1"], f"got legend entries {_legend_labels(svg)}"


def test_mark_pdp_kind_individual_integer_color_field_no_warning_and_symbol_legend():
    """``kind="individual"`` (per-sample ICE polylines) exercises the same
    ``color_field``-bound ``line`` layer as ``kind="average"`` above, plus
    ``mark_style.detail`` splitting on ``_sample_id_str`` (quality-review
    round 2 -- the round-1 widening only covered ``kind="average"``). The
    raw mark's documented data contract requires ``_sample_id_str``
    pre-injected (only the ``pdp_chart`` figure function does this
    normally; the ``Chart(df).mark_pdp()`` API contract requires the
    caller to supply it directly, same as ``desugar_roc``'s
    ``_auc_label_*`` columns).
    """
    df = pl.DataFrame(
        {
            "feature_value": [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
            "pd_value": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "_sample_id_str": ["0", "0", "0", "1", "1", "1"],
            "group": [0, 0, 0, 1, 1, 1],
        }
    )
    chart = ferrum.Chart(df).mark_pdp(kind="individual", color_field="group")
    svg = _assert_no_user_warning(chart.to_svg)

    assert "<svg" in svg
    # One polyline per sample regardless of color-scale type here (detail
    # grouping by _sample_id_str is independent of color) -- the legend
    # below is the assertion that actually discriminates a reverted fix.
    assert svg.count("<polyline") == 2
    assert _legend_labels(svg) == ["0", "1"], f"got legend entries {_legend_labels(svg)}"


def test_mark_pdp_kind_both_integer_color_field_no_warning_and_symbol_legend():
    """``kind="both"`` (ICE polylines + average overlay) reads its own pair
    of layer-specific y-columns (``_pd_ice_value``/``_pd_avg_value``,
    documented in ``desugar_pdp``) on two separate layers, both binding
    ``color_field`` (quality-review round 2). Like ``kind="individual"``
    above, the raw mark's contract requires these builder-injected columns
    supplied directly -- only literal values are needed here (not
    ``pdp_chart``'s full ``_pdp_split_kind_both`` derivation), since the
    desugar only reads the column names, not their provenance.
    """
    df = pl.DataFrame(
        {
            "feature_value": [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
            "_pd_ice_value": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "_pd_avg_value": [0.15, 0.25, 0.35, 0.15, 0.25, 0.35],
            "_sample_id_str": ["0", "0", "0", "1", "1", "1"],
            "group": [0, 0, 0, 1, 1, 1],
        }
    )
    chart = ferrum.Chart(df).mark_pdp(kind="both", color_field="group")
    svg = _assert_no_user_warning(chart.to_svg)

    assert "<svg" in svg
    # 2 ICE polylines (one per group/sample) + 2 average-overlay polylines
    # (one per group) -- both layers bind color_field.
    assert svg.count("<polyline") == 4
    assert _legend_labels(svg) == ["0", "1"], f"got legend entries {_legend_labels(svg)}"


def test_mark_learning_curve_integer_color_field_no_warning_and_symbol_legend():
    """``mark_learning_curve``'s default ``color_field="split"`` is Utf8-safe
    by data contract, but the caller can override it with any column --
    including an Int64 model-id (spec-review round 1 scoping call). Exercises
    both the ribbon CI-band layer and the mean-score line layer, the two
    layers ``desugar_learning_curve`` binds ``color_field`` on.
    """
    df = pl.DataFrame(
        {
            "train_size": [10, 20, 30, 10, 20, 30],
            "mean_score": [0.5, 0.6, 0.7, 0.55, 0.65, 0.75],
            "lower": [0.4, 0.5, 0.6, 0.45, 0.55, 0.65],
            "upper": [0.6, 0.7, 0.8, 0.65, 0.75, 0.85],
            "model_id": [0, 0, 0, 1, 1, 1],
        }
    )
    chart = ferrum.Chart(df).mark_learning_curve(color_field="model_id")
    svg = _assert_no_user_warning(chart.to_svg)

    assert "<svg" in svg
    assert svg.count("<polyline") == 2
    assert _legend_labels(svg) == ["0", "1"], f"got legend entries {_legend_labels(svg)}"


def test_mark_learning_curve_errorbar_integer_color_field_renders_expected_shape():
    """``ci_style="errorbar"`` regression coverage (quality-review round 2):
    the round-1 widening tested only the default ``ci_style="band"``
    branch, leaving this sibling branch's ``rule`` CI layer untested.

    ``rule`` is not in the line/ribbon set ``UnsupportedColorScaleOnMark``
    fires for, so reverting ``nominal_color_channel`` on this branch
    collapses SILENTLY: measured zero warnings, the mean-score "line"
    layer's polyline count drops 2 -> 1, and its legend drops
    ``["0", "1"]`` -> ``[]``. ``_assert_no_user_warning`` is therefore
    structurally unable to catch a regression on this branch -- this test
    asserts the rendered shape directly instead of warning absence. (The
    CI band's own ``rule`` layer renders no independently-countable SVG
    element in either the fixed or reverted case for this data shape --
    it is genuinely invisible either way, a separate, pre-existing
    rendering gap unrelated to color typing -- so the mean-score line's
    polyline count + legend are the only discriminating assertions here.)
    """
    df = pl.DataFrame(
        {
            "train_size": [10, 20, 30, 10, 20, 30],
            "mean_score": [0.5, 0.6, 0.7, 0.55, 0.65, 0.75],
            "lower": [0.4, 0.5, 0.6, 0.45, 0.55, 0.65],
            "upper": [0.6, 0.7, 0.8, 0.65, 0.75, 0.85],
            "model_id": [0, 0, 0, 1, 1, 1],
        }
    )
    chart = ferrum.Chart(df).mark_learning_curve(ci_style="errorbar", color_field="model_id")
    svg = chart.to_svg()

    assert "<svg" in svg
    assert svg.count("<polyline") == 2, "expected one mean-score polyline per model_id group"
    assert _legend_labels(svg) == ["0", "1"], f"got legend entries {_legend_labels(svg)}"


def test_mark_validation_curve_integer_color_field_no_warning_and_symbol_legend():
    """``mark_validation_curve``'s default ``color_field="split"`` is
    Utf8-safe by data contract, but the caller can override it with any
    column -- including an Int64 model-id (quality-review round 2: a
    distinct public mark from ``mark_learning_curve``, sharing the same
    ``desugar_validation_curve`` shape, left untested by the round-1
    widening).
    """
    df = pl.DataFrame(
        {
            "param_value": [0.1, 1.0, 10.0, 0.1, 1.0, 10.0],
            "mean_score": [0.5, 0.6, 0.7, 0.55, 0.65, 0.75],
            "lower": [0.4, 0.5, 0.6, 0.45, 0.55, 0.65],
            "upper": [0.6, 0.7, 0.8, 0.65, 0.75, 0.85],
            "model_id": [0, 0, 0, 1, 1, 1],
        }
    )
    chart = ferrum.Chart(df).mark_validation_curve(color_field="model_id")
    svg = _assert_no_user_warning(chart.to_svg)

    assert "<svg" in svg
    assert svg.count("<polyline") == 2
    assert _legend_labels(svg) == ["0", "1"], f"got legend entries {_legend_labels(svg)}"


def test_mark_parallel_coordinates_integer_color_field_no_warning_and_symbol_legend():
    """``mark_parallel_coordinates``'s ``color_field`` defaults to ``None``
    (opt-in only), but any caller-supplied override -- including an Int64
    cluster-id column -- must not trip the inert-color warning
    (spec-review round 1 scoping call).
    """
    df = pl.DataFrame(
        {
            "feature": ["a", "a", "b", "b"],
            "value": [0.5, 0.3, 0.8, 0.2],
            "sample_id": ["s0", "s1", "s0", "s1"],
            "cluster": [0, 1, 0, 1],
        }
    )
    chart = ferrum.Chart(df).mark_parallel_coordinates(color_field="cluster")
    svg = _assert_no_user_warning(chart.to_svg)

    assert "<svg" in svg
    assert svg.count("<polyline") == 2
    assert _legend_labels(svg) == ["0", "1"], f"got legend entries {_legend_labels(svg)}"


def test_mark_roc_string_class_column_bare_path_byte_identical(monkeypatch):
    """Pin the byte-identity invariant for a Utf8 class column, scoped to
    where it actually holds (spec-review round 1, adjudication a): the bare
    ``Chart(df).mark_roc()`` path -- no chart-level ``.encode()``, so no
    legend-title side effect -- must render byte-identical whether
    ``color_field`` is bound via ``nominal_color_channel`` (this fix) or the
    pre-fix bare-string passthrough. The figure path (``roc_chart``) is NOT
    byte-identical -- it gains a legend title -- and that is a sanctioned,
    deliberate change out of scope for this test; see the module docstring.

    The original version of this test asserted only polyline count and
    legend-swatch labels, which the spec reviewer found could not actually
    detect the legend-title delta the fix introduces elsewhere (finding 2,
    task 5c round 1) -- it never compared bytes. This version reproduces
    the reviewers' own verification technique: monkeypatch
    ``nominal_color_channel`` back to an identity passthrough (the pre-fix
    shape) and diff the two renders' bytes directly.
    """
    import ferrum.marks.diagnostic._classification as _classification_mod

    df = pl.DataFrame(
        {
            "fpr": [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
            "tpr": [0.0, 0.6, 1.0, 0.0, 0.7, 1.0],
            "class": ["setosa", "setosa", "setosa", "versicolor", "versicolor", "versicolor"],
        }
    )
    fixed_svg = ferrum.Chart(df).mark_roc(reference_line=False).to_svg()
    assert fixed_svg.count("<polyline") == 2
    assert _legend_labels(fixed_svg) == ["setosa", "versicolor"], f"got {_legend_labels(fixed_svg)}"

    monkeypatch.setattr(_classification_mod, "nominal_color_channel", lambda field: field)
    bare_string_svg = ferrum.Chart(df).mark_roc(reference_line=False).to_svg()

    assert fixed_svg == bare_string_svg, (
        "bare Chart(df).mark_roc() path must render byte-identical for a "
        "Utf8 class column whether color_field is bound via "
        "nominal_color_channel or the pre-fix bare-string passthrough"
    )


def test_mark_roc_multiclass_reference_diagonal_stays_single_and_grey():
    """Regression pin for the T5d Rust defect (now fixed -- was ``xfail(strict=True)``
    in earlier rounds of this task; T5d's fix landed in this checkout's
    compiled extension and this test flipped to an unexpected pass, exactly
    the "flip loudly" signal the strict xfail was designed to produce, so
    the marker was removed here rather than silently staying green).

    Once ``desugar_roc``'s ``color_field`` resolves a Categorical/Nominal
    scale (this task's fix), the chance-diagonal 'reference' layer --
    which declares no color channel of its own and sets a literal
    ``mark_kwargs.stroke="#AAAAAA"`` -- must NOT get swept into the
    sibling 'line' layer's per-class color scale: it must stay one grey
    dashed diagonal, not duplicate once per class and get repainted with
    each class's color (the bug this test caught before T5d's fix).
    """
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    y = df["y"]
    chart = ferrum.roc_chart(model, X, y)
    svg = chart.to_svg()

    # The reference layer is the only one with a dashed stroke
    # (mark_kwargs={"stroke_dash": [4, 4]}); the per-class "line" layer
    # polylines are solid. Isolate reference-layer polylines by that marker
    # so a class-recolored *and* duplicated reference line is pinned on
    # both axes: count (exactly one) and color (grey, not class-colored).
    reference_polylines = re.findall(r'<polyline[^>]*stroke-dasharray="4,4"[^>]*/>', svg)
    strokes = [_stroke_hex(p) for p in reference_polylines]
    assert strokes == ["#aaaaaa"], (
        f"expected exactly one grey (#aaaaaa) reference diagonal, got {strokes}"
    )
