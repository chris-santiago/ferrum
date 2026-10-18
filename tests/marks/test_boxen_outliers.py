"""``mark_boxen()``'s outlier-point layer: fenced-count correctness,
``outlier_threshold`` sensitivity, the depth-cap-derivation sibling
(``k_depth="tukey"`` vs ``"full"`` diverging once the mark actually renders
every depth the registry declares), and the ``outlier_threshold=0.0``
default -- the fence sits at the outermost letter values themselves, so
every observation strictly beyond them is an outlier and a tied value is
not, on both ``mark_boxen()`` and ``LetterValue`` used directly outside
its composite desugaring. The default's render effect is also checked
against a seeded frame shaped like ``design-docs/demos/demo_grammar_marks.py``'s
boxen fixture (four groups, n=80, normal).

F-L03-04b/11 (frozen texts: ``findings/L03.md``): the outlier layer used to
draw circles for the whole ``"outliers"`` batch and boxen's depth-band loop
was hardcoded to 6, silently dropping real depth data (Rust's ``letter_value``
transform already computes deeper bands for large ``n``). Task 5 filtered the
Rust-side outlier batch to true outliers only and raised the registry depth
cap to 12; this module renders the fixed Python side and checks the two
against an independent Python re-derivation of ``letter_value``'s own
documented algorithm (Tukey depth selection, the ``2**-k`` / ``1-2**-k``
quantile pair, and the outermost-band-span outlier fence) -- not a
hand-built shape, but the same public formulas the Rust transform implements
(``crates/ferrum-core/src/transform/letter_value.rs``), used here only to
independently predict what the real render should show.

Per the repo's Python test-file convention this is feature-scoped coverage
for the boxen outlier/depth-cap seam, not a findings-ID file.
"""

from __future__ import annotations

import math
import re

import numpy as np
import polars as pl

import ferrum as fm
from ferrum import LetterValue
from ferrum._core import secondary_key
from ferrum._layer import _Layer
from ferrum._registry import registry_secondary_columns
from ferrum.marks.composite import _LV_OUTLIERS_KEY, _LV_TRANSFORM_NAME, _lv_depth_key

# ---------------------------------------------------------------------------
# Independent oracle: mirrors letter_value.rs's documented formulas
# (numeric_util::quantile_sorted's linear interpolation, k_for_depth's Tukey
# branch, and secondary_outputs' outermost-band-span fence) so this module can
# predict the real render's outlier count without reading Rust internals.
# ---------------------------------------------------------------------------


def _quantile_sorted(sorted_vals: list[float], p: float) -> float:
    """Mirrors ``numeric_util::quantile_sorted`` (linear interpolation)."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    h = p * (n - 1)
    lo = math.floor(h)
    hi = min(math.ceil(h), n - 1)
    frac = h - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _k_for_depth_tukey(n: int) -> int:
    """Mirrors ``k_for_depth``'s ``KDepth::Tukey`` branch, clamped to the
    registry-derived depth cap."""
    from ferrum.marks.composite import _boxen_depth_count

    raw = math.floor(math.log2(n)) - 3
    return max(min(raw, _boxen_depth_count()), 1)


def _expected_outlier_count(values: list[float], threshold: float) -> int:
    """Independently predicts ``letter_value``'s outlier count for one group
    under ``k_depth="tukey"`` (the mark's default): fence = the outermost
    computed band's own span, scaled by ``threshold`` (mirrors
    ``secondary_outputs``'s ``low_fence``/``high_fence`` formula)."""
    s = sorted(values)
    k = _k_for_depth_tukey(len(s))
    p = 0.5**k
    lo = _quantile_sorted(s, p)
    hi = _quantile_sorted(s, 1.0 - p)
    span = hi - lo
    low_fence = lo - threshold * span
    high_fence = hi + threshold * span
    return sum(1 for v in values if v < low_fence or v > high_fence)


def _circle_count(svg: str) -> int:
    """Every ``<circle>`` in the SVG. Boxen's only point-mark layer is the
    outlier layer (bands are ``rect``, the median is a ``tick``/``line``), so
    with no color legend (no ``.encode(color=...)``) this counts exactly the
    rendered outlier points."""
    return len(re.findall(r"<circle", svg))


def _multi_group_outlier_fixture() -> tuple[pl.DataFrame, int]:
    """Two groups, each with a run of in-range values plus a few values far
    enough beyond the outermost letter-value band to be real outliers under
    an explicit ``outlier_threshold=1.5``. Returns the frame and the
    independently-predicted total outlier count across both groups."""
    group_a = [float(i) for i in range(150)] + [800.0, -800.0, 900.0]
    group_b = [float(i) for i in range(120)] + [700.0]
    expected = _expected_outlier_count(group_a, 1.5) + _expected_outlier_count(group_b, 1.5)
    df = pl.DataFrame(
        {
            "g": ["a"] * len(group_a) + ["b"] * len(group_b),
            "v": group_a + group_b,
        }
    )
    return df, expected


def _gallery_boxen_frame() -> tuple[pl.DataFrame, int]:
    """Four groups x n=80, drawn from ``rng.normal(mu, 1.0)`` with distinct
    means, the same shape as ``design-docs/demos/demo_grammar_marks.py``'s
    boxen fixture (``df_box``): four groups, n=80 each, the same per-group
    means. Returns the frame and the independently-predicted total outlier
    count at the mark's default ``outlier_threshold=0.0`` -- every
    observation strictly beyond its group's own outermost letter value.
    At ``n=80`` the Tukey depth is ``k=3`` (``floor(log2(80)) - 3 = 3``),
    putting the outermost quantile's interpolation position at
    ``h = 0.125 * 79 = 9.875`` -- strictly between the 10th and 11th order
    statistics. For continuous data (ties have probability zero), the
    fence therefore separates exactly 10 values per tail, so 20 outliers
    per group and 80 across the four groups holds for any continuous data
    of this shape, not only this seed."""
    rng = np.random.default_rng(42)
    groups = ["Control", "Treatment A", "Treatment B", "Treatment C"]
    mus = [0.0, 0.8, -0.4, 1.2]
    g_col: list[str] = []
    v_col: list[float] = []
    expected = 0
    for g, mu in zip(groups, mus):
        vals = [float(rng.normal(mu, 1.0)) for _ in range(80)]
        g_col += [g] * len(vals)
        v_col += vals
        expected += _expected_outlier_count(vals, 0.0)
    return pl.DataFrame({"g": g_col, "v": v_col}), expected


def test_boxen_outlier_circle_count_matches_the_fenced_count():
    """The rendered outlier-point count equals the independently-predicted
    fenced count, across two groups with distinct per-group fences (proves
    the outlier layer reads the real per-group ``"outliers"`` batch rather
    than, e.g., every input row or a global fence)."""
    df, expected = _multi_group_outlier_fixture()
    assert expected > 0, "fixture must contain real outliers to be a meaningful pin"
    svg = fm.Chart(df).mark_boxen(outlier_threshold=1.5).encode(x="g", y="v").to_svg()
    assert _circle_count(svg) == expected


def test_boxen_zero_outliers_renders_zero_circles():
    """A dataset with nothing beyond the outermost band's fence, under an
    explicit ``outlier_threshold=1.5``, renders no outlier circles at
    all."""
    values = [float(i) for i in range(200)]
    df = pl.DataFrame({"g": ["a"] * len(values), "v": values})
    assert _expected_outlier_count(values, 1.5) == 0, "fixture must have no real outliers"
    svg = fm.Chart(df).mark_boxen(outlier_threshold=1.5).encode(x="g", y="v").to_svg()
    assert _circle_count(svg) == 0


def test_boxen_default_gallery_frame_draws_80_outlier_circles():
    """The default ``outlier_threshold=0.0`` flags every observation
    strictly beyond its group's own outermost letter value: on a frame
    shaped like the project's gallery chart (4 groups, n=80 each, normal),
    the rendered circle count equals an independent numpy count of such
    observations (``test_boxen_explicit_threshold_1_5_on_gallery_frame_draws_zero_circles``
    pins the same frame under a widened fence, where none of them are
    outliers)."""
    df, expected = _gallery_boxen_frame()
    assert expected == 80, "fixture must reproduce the documented gallery-frame count"
    svg = fm.Chart(df).mark_boxen().encode(x="g", y="v").to_svg()
    assert _circle_count(svg) == expected


def test_boxen_explicit_threshold_1_5_on_gallery_frame_draws_zero_circles():
    """``outlier_threshold=1.5`` passed explicitly still draws 0 circles on
    the gallery-shaped frame: every one of its observations sits inside
    the 1.5x-widened fence, where none sit inside the unwidened one."""
    df, _ = _gallery_boxen_frame()
    svg = fm.Chart(df).mark_boxen(outlier_threshold=1.5).encode(x="g", y="v").to_svg()
    assert _circle_count(svg) == 0


def test_boxen_ties_at_the_outermost_letter_value_are_not_flagged():
    """At the default ``outlier_threshold=0.0`` the fence sits exactly at
    the outermost letter values: a value tied with the fence is not
    flagged, only values strictly beyond it are. ``n=9`` with
    ``k_depth="full"`` gives an exact (non-interpolated) depth-3 fence of
    ``[2.0, 8.0]`` -- ``2.0``/``8.0`` are ties and must not render;
    ``1.0``/``9.0`` are strictly beyond and must."""
    values = [float(i) for i in range(1, 10)]
    df = pl.DataFrame({"g": ["a"] * len(values), "v": values})
    svg = fm.Chart(df).mark_boxen(k_depth="full").encode(x="g", y="v").to_svg()
    assert _circle_count(svg) == 2


def test_boxen_group_with_nothing_beyond_the_fence_renders_an_empty_outlier_layer():
    """A group whose values are all identical collapses every letter-value
    quantile to that one value, so nothing is strictly beyond the fence at
    the default ``outlier_threshold=0.0`` and the outlier layer draws no
    circles."""
    values = [5.0] * 10
    df = pl.DataFrame({"g": ["a"] * len(values), "v": values})
    svg = fm.Chart(df).mark_boxen().encode(x="g", y="v").to_svg()
    assert _circle_count(svg) == 0


def test_letter_value_used_directly_fences_at_the_outermost_values():
    """``LetterValue`` used directly -- not through ``mark_boxen()``'s
    composite desugaring -- fences at the outermost letter values under
    its own default (``outlier_threshold=0.0``): a layer reading the
    transform's own ``outliers`` secondary output renders exactly the
    independently predicted count, ties at the fence excluded (mirrors
    ``test_boxen_ties_at_the_outermost_letter_value_are_not_flagged``'s
    n=9/k_depth="full" fixture). The output's declared schema -- E5's
    ``group``/``value``/``is_outlier``, only flagged rows emitted -- is
    unchanged: a missing or renamed column raises here before the render
    assertion runs."""
    schema = registry_secondary_columns(
        "letter_value", _LV_OUTLIERS_KEY, ["group", "value", "is_outlier"]
    )
    assert set(schema) == {"group", "value", "is_outlier"}

    values = [float(i) for i in range(1, 10)]
    df = pl.DataFrame({"g": ["a"] * len(values), "v": values})
    # Each layer reads one of the transform's own outputs, as mark_boxen()'s
    # bands and outlier points do: the innermost band reads its depth output
    # and the points the outlier rows. The chart's own rows are the source
    # frame, which carries no ``group``.
    chart = (
        fm.Chart(df)
        .transform(LetterValue(value="v", group="g", k_depth="full", name=_LV_TRANSFORM_NAME))
        .properties()
    )
    chart._layers = [
        _Layer(
            name="band",
            mark="rect",
            encoding={"x": "group", "y": "lower", "y2": "upper"},
            data_source=secondary_key(_LV_TRANSFORM_NAME, _lv_depth_key(1)),
        ),
        _Layer(
            name="outlier",
            mark="point",
            encoding={"x": "group", "y": "value"},
            data_source=secondary_key(_LV_TRANSFORM_NAME, _LV_OUTLIERS_KEY),
        ),
    ]
    svg = chart.to_svg()
    assert _circle_count(svg) == 2


def test_boxen_outlier_threshold_sensitivity():
    """A stricter (smaller) ``outlier_threshold`` flags MORE points as
    outliers than a looser (larger) one, on data with points that sit just
    beyond the strict fence but well inside the loose one (direction-shaped:
    the two counts must differ, not pinned to specific magic numbers)."""
    values = [float(i) for i in range(200)] + [250.0, 300.0]
    strict, loose_a, loose_b = 0.1, 1.5, 5.0
    expected_strict = _expected_outlier_count(values, strict)
    expected_loose_a = _expected_outlier_count(values, loose_a)
    expected_loose_b = _expected_outlier_count(values, loose_b)
    assert expected_strict > expected_loose_a, "fixture must discriminate strict vs. loose"

    df = pl.DataFrame({"g": ["a"] * len(values), "v": values})
    counts = {}
    for threshold in (strict, loose_a, loose_b):
        svg = fm.Chart(df).mark_boxen(outlier_threshold=threshold).encode(x="g", y="v").to_svg()
        counts[threshold] = _circle_count(svg)

    assert counts[strict] == expected_strict
    assert counts[loose_a] == expected_loose_a
    assert counts[loose_b] == expected_loose_b
    assert counts[strict] > counts[loose_a]
    assert counts[strict] > counts[loose_b]


def test_boxen_k_depth_tukey_vs_full_differ_at_n_2000():
    """``k_depth="tukey"`` and ``k_depth="full"`` reach different real
    depths at n=2000 (``floor(log2(2000))-3=7`` vs ``floor(log2(2000))=10``),
    and -- now that the depth-band loop actually builds a layer for every
    depth the registry declares, instead of stopping at a hardcoded 6 --
    that difference is visible in the render: ``"full"`` must draw strictly
    more depth-band rects than the default ``"tukey"`` (F-L03-04b). Before
    the fix, both were silently truncated to the same 6 bands and this
    assertion would fail."""
    values = [float(i) for i in range(2000)]
    df = pl.DataFrame({"g": ["a"] * len(values), "v": values})
    svg_tukey = fm.Chart(df).mark_boxen().encode(x="g", y="v").to_svg()
    svg_full = fm.Chart(df).mark_boxen(k_depth="full").encode(x="g", y="v").to_svg()
    rects_tukey = len(re.findall(r"<rect", svg_tukey))
    rects_full = len(re.findall(r"<rect", svg_full))
    assert rects_full > rects_tukey
