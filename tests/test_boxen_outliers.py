"""``mark_boxen()``'s outlier-point layer: fenced-count correctness,
``outlier_threshold`` sensitivity, and the depth-cap-derivation sibling
(``k_depth="tukey"`` vs ``"full"`` diverging once the mark actually renders
every depth the registry declares).

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

import polars as pl

import ferrum as fm

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
    the mark's default ``outlier_threshold=1.5``. Returns the frame and the
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


def test_boxen_outlier_circle_count_matches_the_fenced_count():
    """The rendered outlier-point count equals the independently-predicted
    fenced count, across two groups with distinct per-group fences (proves
    the outlier layer reads the real per-group ``"outliers"`` batch rather
    than, e.g., every input row or a global fence)."""
    df, expected = _multi_group_outlier_fixture()
    assert expected > 0, "fixture must contain real outliers to be a meaningful pin"
    svg = fm.Chart(df).mark_boxen().encode(x="g", y="v").to_svg()
    assert _circle_count(svg) == expected


def test_boxen_zero_outliers_renders_zero_circles():
    """A dataset with nothing beyond the outermost band's fence renders no
    outlier circles at all."""
    values = [float(i) for i in range(200)]
    df = pl.DataFrame({"g": ["a"] * len(values), "v": values})
    assert _expected_outlier_count(values, 1.5) == 0, "fixture must have no real outliers"
    svg = fm.Chart(df).mark_boxen().encode(x="g", y="v").to_svg()
    assert _circle_count(svg) == 0


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
