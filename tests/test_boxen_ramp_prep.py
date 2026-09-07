"""``_prep_boxen_ramp_depth``'s grouping-column resolution, its
``letter_value.rs``-mirrored row counting, and a suite-guard sweep over
``_k_for_depth``'s four strategies.

Quality-review REVISE, cycle 3 (all findings from the cycle-2 ramp fix):

1. **Grouping mechanism** (S3, blocking): the prep used to resolve its
   grouping column as ``color_field or ctx._color_field() or cat`` -- the
   ``ctx._color_field()`` term consulted the chart-level ``.encode(color=)``
   channel, which ``desugar_boxen`` itself never groups by (`group =
   color_field if color_field else cat`, see ``composite.py``). This module
   pins the FOUR shapes where that could have diverged: a plain chart, a
   chart-level color encoding (must be ignored), the ``color_field=`` mark
   kwarg (must be honored), and ``palette=`` (must skip the prep entirely).
2. **`_k_for_depth` suite guard** (S3): a parametrized sweep -- the same
   4-strategy x 10-``n`` grid the spec reviewer ran by hand in cycle 2 --
   now lives in the suite instead of a one-off manual check, comparing the
   Python mirror against the REAL rendered band count for every point.
3. **NaN/null counting** (S1): ``letter_value::extract_values`` drops a row
   when its VALUE is null or NaN, but keeps a null GROUP as its own group.
   The estimator now mirrors both directions exactly (was: dropped
   null-group rows via a plain ``drop_nulls()`` over both columns, and kept
   NaN values since ``drop_nulls()`` doesn't touch NaN).
4. **`materialized_depth is None` fallback** (S2, test gap): pinned so the
   documented degrade-to-full-cap path is proven, not assumed.

The coordinator's north-star (expose the depth computation from Rust via
PyO3 so Python calls the single source instead of mirroring it) is filed as
a follow-up at batch close, not built here -- this module's sweep is the
interim drift guard for the mirror that ships today.

Per the repo's Python test-file convention this is feature-scoped coverage
for the boxen ramp-prep seam, not a findings-ID file.
"""

from __future__ import annotations

import re

import polars as pl
import pytest

import ferrum as fm
from ferrum.marks.composite import _boxen_depth_count, _boxen_materialized_depth, _k_for_depth


def _rendered_opacities(svg: str) -> list[float]:
    """Alpha channel of every ``rgba(...)`` rect fill, in document order --
    the default (``palette=None``) depth-band opacity ramp. A ``palette=``
    chart emits plain hex fills instead, so this list is empty there."""
    return [float(a) for a in re.findall(r'fill="rgba\([^)]*,\s*([\d.]+)\)"', svg)]


# ---------------------------------------------------------------------------
# 1. Grouping mechanism: the prep's group_field must equal desugar_boxen's
#    own `color_field if color_field else cat`, across every shape where a
#    chart-level color channel could tempt a different resolution.
# ---------------------------------------------------------------------------


def _grouping_fixture() -> pl.DataFrame:
    """Three grouping columns with deliberately DIFFERENT real letter-value
    depths, so a wrong resolution is measurable, not just structurally
    plausible:

    * ``g`` (the categorical axis): 2 categories, 300/100 rows -- tukey
      depth 5 (n=300) / 3 (n=100), materialized max 5.
    * ``c`` (a chart-level ``.encode(color=...)`` column, unrelated to
      grouping): 8 categories of 50 rows each -- tukey depth 2, chosen to
      differ sharply from ``g``'s 5 so a wrong resolution is unmistakable
      (an earlier fixture used a 2-category ``c`` split whose depth
      happened to coincide with ``g``'s, which passed even with the buggy
      ``ctx._color_field()`` fallback still present -- a false negative
      this shape closes).
    * ``hue`` (the ``color_field=`` mark kwarg's target): a different
      8-small-groups split, independently verifying the same depth-2
      shape via the *correct* mechanism (the kwarg) rather than the
      chart-level channel this fixture's ``c`` column exercises.
    """
    g = ["p"] * 300 + ["q"] * 100
    c = []
    for i in range(8):
        c += [f"c{i}"] * 50
    hue = []
    for i in range(8):
        hue += [f"h{i}"] * 50
    v = [float(i) for i in range(400)]
    return pl.DataFrame({"g": g, "v": v, "c": c, "hue": hue})


def test_boxen_ramp_prep_ignores_chart_level_color_encoding():
    """Shape 2: a bare ``.encode(color=...)`` (no ``palette=``) is legal and
    common, and must leave the ramp byte-for-byte unchanged -- boxen's own
    grouping never reads the chart-level color channel (`desugar_boxen`'s
    docstring: "`palette` does not interact with `color_field`"), and
    boxen's `_PENDING_PREP` tuple deliberately omits `_prep_color_field`.
    RED pre-fix: the deleted `ctx._color_field()` fallback made this
    diverge whenever the color column's category sizes differed from the
    categorical axis's."""
    df = _grouping_fixture()
    svg_plain = fm.Chart(df).mark_boxen().encode(x="g", y="v").to_svg()
    svg_colored = fm.Chart(df).mark_boxen().encode(x="g", y="v", color="c").to_svg()
    assert _rendered_opacities(svg_plain) == _rendered_opacities(svg_colored)


def test_boxen_ramp_prep_uses_color_field_kwarg_like_the_real_transform():
    """Shape 3: `mark_boxen(color_field=...)` DOES change boxen's real
    `LetterValue` grouping (unlike the chart-level color encoding above),
    so the ramp must reflect grouping by `hue` (materialized depth 2, from
    8x50-row groups) rather than by the categorical axis `g` (materialized
    depth 5, from 300/100-row groups) -- proving the prep's `color_field`
    priority actually matches the transform's, not merely that it ignores
    chart-level color."""
    df = _grouping_fixture()
    svg_by_cat = fm.Chart(df).mark_boxen().encode(x="g", y="v").to_svg()
    svg_by_hue = fm.Chart(df).mark_boxen(color_field="hue").encode(x="g", y="v").to_svg()

    opacities_by_cat = _rendered_opacities(svg_by_cat)
    opacities_by_hue = _rendered_opacities(svg_by_hue)
    assert set(opacities_by_cat) != set(opacities_by_hue), (
        "grouping by `hue` (depth 2) must produce a visibly different ramp "
        "than grouping by `g` (depth 5) -- if these match, color_field is "
        "not actually driving the estimate"
    )
    # depth 2 (guarded minimum) -> exactly two opacity values, 0.85 and 0.30.
    assert set(opacities_by_hue) == {0.85, 0.3} or set(round(o, 2) for o in opacities_by_hue) == {
        0.85,
        0.3,
    }


def test_boxen_ramp_prep_skipped_entirely_under_palette():
    """Shape 4: `palette=` never uses the opacity ramp at all (bands render
    a flat hex `fill=`, not `rgba(...)`), so the prep must not even run its
    data-reading branch -- proven here by asserting zero `rgba(...)` fills
    appear, regardless of the same grouping ambiguity shapes 1-3 exercise."""
    df = _grouping_fixture()
    svg = fm.Chart(df).mark_boxen(palette=["#111111", "#222222"]).encode(x="g", y="v").to_svg()
    assert _rendered_opacities(svg) == []


def test_boxen_ramp_prep_axis_swap_matches_across_orientations():
    """Pins the prep's `horizontal` axis-swap (`_desugar.py`'s
    `value_field = ctx.x_field if horizontal else ctx.y_field` /
    `cat_field = ctx.y_field if horizontal else ctx.x_field`) against
    `desugar_boxen`'s own swap (`val = x_field if horizontal else y_field`
    / `cat = y_field if horizontal else x_field`, `composite.py`): the real
    letter-value grouping is the same categorical/value pair regardless of
    orientation, so the ramp must be identical across `horizontal=False`
    (default), `horizontal=True`, and the `orient="horizontal"` spelling on
    the same asymmetric (300/100) fixture the grouping-mechanism tests
    above use. An inverted swap in the prep (using the wrong field as the
    group or the value) would silently estimate a different depth for the
    horizontal cases without failing any test above, which only exercises
    the default vertical orientation (quality-review APPROVED cycle-3, one
    non-blocking S1 taken now)."""
    df = _grouping_fixture()
    svg_vertical = fm.Chart(df).mark_boxen().encode(x="g", y="v").to_svg()
    svg_horizontal_flag = fm.Chart(df).mark_boxen(horizontal=True).encode(x="v", y="g").to_svg()
    svg_horizontal_orient = (
        fm.Chart(df).mark_boxen(orient="horizontal").encode(x="v", y="g").to_svg()
    )
    opacities_vertical = _rendered_opacities(svg_vertical)
    opacities_horizontal_flag = _rendered_opacities(svg_horizontal_flag)
    opacities_horizontal_orient = _rendered_opacities(svg_horizontal_orient)
    assert opacities_vertical != [], "fixture must materialize real bands to discriminate"
    assert opacities_vertical == opacities_horizontal_flag == opacities_horizontal_orient


# ---------------------------------------------------------------------------
# 2. `_k_for_depth` suite guard: the spec reviewer's 40-point manual sweep,
#    now a parametrized test comparing the Python mirror against the real
#    rendered band count.
# ---------------------------------------------------------------------------

_SWEEP_NS = [10, 50, 63, 64, 200, 1000, 1024, 5000, 8192, 100_000]
_SWEEP_STRATEGIES = ["tukey", "proportion", "trustworthy", "full"]


@pytest.mark.parametrize("k_depth", _SWEEP_STRATEGIES)
@pytest.mark.parametrize("n", _SWEEP_NS)
def test_boxen_k_for_depth_matches_real_rendered_band_count(k_depth, n):
    """`_k_for_depth(k_depth, k_proportion, n)` must equal the number of
    real depth bands `mark_boxen(k_depth=...)` actually renders for a
    single group of size *n* -- across power-of-two boundaries (63/64,
    1023-adjacent 1024) and the cap-saturation regime (100_000), for all
    four strategies. This is the drift guard the coordinator adjudicated in
    place of exposing the computation from Rust (filed as a follow-up, not
    built here): a Rust-side change to any strategy's formula, or to the
    registry-derived cap, fails here instead of silently degrading the
    ramp's contrast."""
    values = [float(i) for i in range(n)]
    df = pl.DataFrame({"g": ["a"] * n, "v": values})
    svg = fm.Chart(df).mark_boxen(k_depth=k_depth).encode(x="g", y="v").to_svg()
    real_bands = len(_rendered_opacities(svg))
    predicted = _k_for_depth(k_depth, 0.007, n)
    assert real_bands == predicted, (
        f"k_depth={k_depth!r} n={n}: mirror predicted {predicted} bands, "
        f"real render materialized {real_bands}"
    )


# ---------------------------------------------------------------------------
# 3. NaN/null row counting must mirror letter_value::extract_values exactly.
# ---------------------------------------------------------------------------


def test_boxen_materialized_depth_drops_nan_values_like_extract_values():
    """400 rows, 200 of which are NaN in the value column: the real
    transform drops NaN values (`v.is_nan()` in `extract_values`), so the
    real group size is 200, not 400 -- estimated AND rendered depth must
    both be 4 (tukey depth for n=200), not 5 (the depth a naive n=400 count
    would predict). RED pre-fix: a plain `drop_nulls()` over both columns
    doesn't touch NaN, so the old estimator counted all 400 rows."""
    values = [float(i) for i in range(200)] + [float("nan")] * 200
    df = pl.DataFrame({"g": ["a"] * 400, "v": values})

    estimated = _boxen_materialized_depth(
        df, value_field="v", group_field="g", k_depth="tukey", k_proportion=0.007
    )
    assert estimated == 4

    svg = fm.Chart(df).mark_boxen().encode(x="g", y="v").to_svg()
    assert len(_rendered_opacities(svg)) == 4


def test_boxen_materialized_depth_keeps_a_null_group_as_its_own_group():
    """A null value in the GROUP column is its own group in
    `letter_value.rs` (`BTreeMap<Option<String>, ...>` keyed on
    `Option<String>`), not a dropped row -- the estimator must count it the
    same way, not exclude it via a blanket `drop_nulls()` over both
    columns."""
    values = [float(i) for i in range(50)]
    df = pl.DataFrame({"g": [None] * 50, "v": values})
    estimated = _boxen_materialized_depth(
        df, value_field="v", group_field="g", k_depth="tukey", k_proportion=0.007
    )
    # tukey depth for n=50 (the null group's own size) is 2, guarded floor.
    assert estimated == 2


# ---------------------------------------------------------------------------
# 4. `materialized_depth is None` fallback: degrade to the full configured
#    cap, never a crash, when the estimate can't be determined.
# ---------------------------------------------------------------------------


def test_boxen_materialized_depth_returns_none_for_missing_data():
    assert (
        _boxen_materialized_depth(
            None, value_field="v", group_field="g", k_depth="tukey", k_proportion=0.007
        )
        is None
    )


def test_boxen_materialized_depth_returns_none_for_missing_column():
    df = pl.DataFrame({"g": ["a"] * 10, "v": [float(i) for i in range(10)]})
    assert (
        _boxen_materialized_depth(
            df, value_field="not_a_column", group_field="g", k_depth="tukey", k_proportion=0.007
        )
        is None
    )


def test_boxen_ramp_falls_back_to_full_cap_when_materialized_depth_is_none():
    """`desugar_boxen(materialized_depth=None)` (the injected default, and
    what the prep leaves it at when the estimate can't be determined) must
    render the exact cap-normalized ramp -- the documented degrade path,
    pinned rather than assumed."""
    from ferrum.marks.composite import desugar_boxen

    result = desugar_boxen("g", "v", materialized_depth=None)
    depth_layers = [layer for layer in result.layers if layer.name.startswith("depth_")]
    depth_count = _boxen_depth_count()
    assert len(depth_layers) == depth_count
    opacities = {round(layer.mark_kwargs["opacity"], 4) for layer in depth_layers}
    expected = {
        round(min(max(0.85 - 0.55 * (k - 1) / max(depth_count - 1, 1), 0.15), 0.85), 4)
        for k in range(1, depth_count + 1)
    }
    assert opacities == expected


# ---------------------------------------------------------------------------
# 5. Faceted boxen: the ramp must normalize against what a SINGLE panel
#    materializes, not against a group's rows pooled across every panel.
# ---------------------------------------------------------------------------


def test_boxen_materialized_depth_pools_across_panels_without_the_facet_field():
    """Baseline confirming the hypothesis (batch-E close remediation design
    S2/HYPOTHESIS): with no facet field passed, `_boxen_materialized_depth`
    groups by `group_field` ALONE, so three panels' worth of the SAME
    group's rows (60 each) pool into one group of 180 -- tukey depth 4 --
    even though no single panel's own 60 rows would ever produce more than
    depth 2. This is the exact inflation the `extra_groupby_fields` fix
    below corrects."""
    df = pl.concat(
        [
            pl.DataFrame({"panel": [p] * 60, "g": ["only"] * 60, "v": list(range(60))})
            for p in ("p0", "p1", "p2")
        ]
    )
    pooled = _boxen_materialized_depth(
        df, value_field="v", group_field="g", k_depth="tukey", k_proportion=0.007
    )
    assert pooled == 4
    per_panel = _boxen_materialized_depth(
        df,
        value_field="v",
        group_field="g",
        k_depth="tukey",
        k_proportion=0.007,
        extra_groupby_fields=("panel",),
    )
    assert per_panel == 2


def test_faceted_boxen_ramp_matches_each_panels_own_depth_not_the_pooled_one():
    """MATERIAL confirmation, through the real render: a faceted boxen chart
    whose per-panel group is too shallow (own n=60, own depth 2) to reach
    the depth the SAME group's rows would reach if pooled across all three
    panels (180 rows, depth 4) must render each panel's ramp normalized
    against its OWN depth (2 bands, opacities 0.85/0.30), not the pooled one
    (4 bands, opacities 0.85/0.667/0.483/0.30) -- the pooled ramp is what
    every panel rendered before this fix, flattening the contrast of a
    2-band chart toward a 4-band chart's step size. This is the disclosed
    byte-stability rider: a faceted `mark_boxen()` chart's rendered SVG may
    change from before this fix."""
    n = 60
    df = pl.concat(
        [
            pl.DataFrame({"panel": [p] * n, "g": ["only"] * n, "v": list(range(n))})
            for p in ("p0", "p1", "p2")
        ]
    )
    svg = fm.Chart(df).mark_boxen().encode(x="g", y="v").facet(col="panel").to_svg()
    opacities = _rendered_opacities(svg)

    # Own-depth-2 ramp: 2 bands per panel, 3 panels. SVG fills quantize
    # opacity to an 8-bit alpha channel (rgba round-trip), so compare with a
    # tolerance rather than exact float equality -- ~1/255 =~ 0.004 per step.
    assert len(opacities) == 6
    own_denominator = max(2 - 1, 1)
    expected_per_panel = [
        min(max(0.85 - 0.55 * (k - 1) / own_denominator, 0.15), 0.85) for k in (1, 2)
    ]
    for i in range(0, 6, 2):
        panel_opacities = opacities[i : i + 2]
        assert panel_opacities == pytest.approx(expected_per_panel, abs=0.005), (
            f"panel at opacities[{i}:{i + 2}] = {panel_opacities} does not match "
            f"its own depth-2 ramp {expected_per_panel} -- looks pooled instead"
        )

    # And explicitly NOT the pooled depth-4 ramp any panel would show under
    # the pre-fix global-pooled estimate.
    pooled_denominator = max(4 - 1, 1)
    pooled_k2_opacity = 0.85 - 0.55 * (2 - 1) / pooled_denominator
    assert opacities[1] != pytest.approx(pooled_k2_opacity, abs=0.005)


def test_layered_boxen_facet_still_pools_across_panels_pending_lifecycle_fix():
    """Scope pin for the facet fix (close-remediation cycle-2 design review
    S3): `Chart.__add__` and `Chart.layer()` both call `self._resolve_pending()`
    (`chart.py`) BEFORE doing their own work, so a chart built as
    `mark_boxen() + mark_point()` resolves the boxen desugar -- and runs
    `_prep_boxen_ramp_depth` -- while `ctx.chart._facet` is still `None`,
    strictly BEFORE the later `.facet(col=...)` call ever populates it. The
    facet-aware fix in `_prep_boxen_ramp_depth`/`_boxen_materialized_depth`
    therefore does NOT reach this lifecycle: the chart still pools every
    panel's rows together and renders the SAME (wrong, pooled depth-4) ramp
    on all three panels, identical to `test_faceted_boxen_ramp_matches_...`'s
    documented pre-fix pooled values (0.851/0.667 for k=1/k=2).

    This is NOT a bug this test asks to be fixed -- it pins the CURRENT,
    known-incomplete scope so the layered-path lifecycle re-plumb (filed as
    a follow-up, not attempted in this batch) has a visible, regression-
    detectable blast radius rather than an invisible one, the same
    treatment the honored-gate-asymmetry tests give that follow-up."""
    n = 60
    df = pl.concat(
        [
            pl.DataFrame({"panel": [p] * n, "g": ["only"] * n, "v": list(range(n))})
            for p in ("p0", "p1", "p2")
        ]
    )
    chart = (
        fm.Chart(df).mark_boxen().encode(x="g", y="v")
        + fm.Chart(df).mark_point().encode(x="g", y="v")
    ).facet(col="panel")
    opacities = _rendered_opacities(chart.to_svg())

    # 2 boxen bands per panel, 3 panels (the point layer renders a plain
    # hex fill, not `rgba(...)`, so it contributes nothing to this list).
    # Pooled depth-4 ramp, identical across all three panels -- NOT each
    # panel's own depth-2 ramp, unlike the `.facet()`-first construction
    # order pinned above.
    assert len(opacities) == 6
    pooled_denominator = max(4 - 1, 1)
    pooled_ramp = [min(max(0.85 - 0.55 * (k - 1) / pooled_denominator, 0.15), 0.85) for k in (1, 2)]
    for i in range(0, 6, 2):
        panel_opacities = opacities[i : i + 2]
        assert panel_opacities == pytest.approx(pooled_ramp, abs=0.005), (
            f"panel at opacities[{i}:{i + 2}] = {panel_opacities} does not match "
            f"the pooled depth-4 ramp {pooled_ramp} -- the layered lifecycle fix "
            "landed, this test needs updating rather than being deleted"
        )
    # Every panel renders identically (the whole point of this test): a
    # per-panel-correct fix would make panels DIFFER, since each panel's own
    # n=60 subset only reaches depth 2.
    assert opacities == opacities[:2] * 3
