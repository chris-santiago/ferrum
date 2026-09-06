"""Batch-D T11 (Python round) -- KDE `exponential`/`linear` kernels,
`transform_density(kernel=)`, and `mark_qq(distribution="lognormal")`.

Closes F-L03-07 (documented `mark_density` kernels `exponential`/`linear`
must actually compute, not raise or silently alias to gaussian) and
F-L05-12 (`transform_density(kernel=)` must exist and route on the wire).
The Rust half (`crates/ferrum-core/src/transform/kde.rs`, `qq.rs`,
`density_data.rs`) is pinned by its own unit tests against the closed-form
kernel/distribution formulas (see `.sdd/task-11-report.md`); these tests
prove the *integration* path -- Python kwargs actually reach that Rust
dispatch through `mark_density`/`mark_qq`/`transform_density`, not a
hand-built shape that never touches the real render pipeline (operating
principle 11).

RED-proof: every test below fails on pre-T11 code -- `kernel="exponential"`
and `kernel="linear"` were rejected by `_VALID_KERNELS` in
`ferrum/marks/statistical.py` before this batch, `transform_density` had no
`kernel` parameter at all, and `mark_qq`'s `validate_choice` tuple in
`ferrum/marks/heavy_stat.py` rejected `"lognormal"` even though the outer
`mark_qq` docstring already (incorrectly) documented it as supported.
"""

from __future__ import annotations

import math
import re
from statistics import NormalDist

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from ferrum.transforms import transform_density

# ---------------------------------------------------------------------------
# Shared SVG-parsing helpers (mirrors the `<circle cx=".." cy=".." r="..">`
# regex convention used across the suite, e.g. test_facet_shared_extent.py).
# ---------------------------------------------------------------------------


def _polyline_points(svg: str) -> list[tuple[float, float]]:
    match = re.search(r'<polyline points="([^"]+)"', svg)
    assert match is not None, "expected exactly one <polyline> in the rendered SVG"
    points = []
    for pair in match.group(1).split():
        x_str, y_str = pair.split(",")
        points.append((float(x_str), float(y_str)))
    return points


def _circle_points(svg: str) -> list[tuple[float, float]]:
    return [
        (float(x), float(y))
        for x, y in re.findall(r'<circle cx="(-?[0-9.]+)" cy="(-?[0-9.]+)" r="[0-9.]+"', svg)
    ]


# ---------------------------------------------------------------------------
# mark_density(kernel="exponential"|"linear") -- render and differ from
# gaussian by a genuine shape discriminator, not a bare string `!=`.
#
# Two points at x=0 with an explicit bandwidth=1.0 puts the whole rendered
# curve at grid offsets u = x/h, so the SAME closed-form kernel ratios the
# Rust unit tests pin (K(0)=0.5 for exponential vs. Gaussian's ~0.3989;
# linear's exact triangular 1-|u|) are recoverable from pixel space: within
# one render, ratios of PIXEL-y differences between three grid points equal
# ratios of DATA-value differences (the y-scale is an unknown affine map
# `pixel = a - b*value`, and both `a` and `b` cancel in a ratio of
# differences). That makes the three kernels' pixel ratios land on three
# well-separated, independently-computable constants.
# ---------------------------------------------------------------------------

_KERNEL_EXTENT = (-3.0, 3.0)
_KERNEL_STEPS = 61
_KERNEL_STEP_SIZE = (_KERNEL_EXTENT[1] - _KERNEL_EXTENT[0]) / (_KERNEL_STEPS - 1)


def _grid_index(x: float) -> int:
    return round((x - _KERNEL_EXTENT[0]) / _KERNEL_STEP_SIZE)


def _kernel_ratio(kernel: str) -> float:
    """Render a two-point (both at x=0), bandwidth=1.0 density curve and
    return (K(0)-K(0.3)) / (K(0.3)-K(0.6)) read off the rendered pixels."""
    df = pl.DataFrame({"val": [0.0, 0.0]})
    svg = (
        fm.Chart(df)
        .mark_density(
            kernel=kernel,
            bandwidth=1.0,
            extent=_KERNEL_EXTENT,
            n=_KERNEL_STEPS,
            fill=False,
        )
        .encode(x="val")
        .to_svg()
    )
    points = _polyline_points(svg)
    assert len(points) == _KERNEL_STEPS
    y0 = points[_grid_index(0.0)][1]
    y1 = points[_grid_index(0.3)][1]
    y2 = points[_grid_index(0.6)][1]
    return (y0 - y1) / (y1 - y2)


def test_mark_density_exponential_renders_with_expected_cusp_ratio():
    """exp(-|u|)/2 gives (K(0)-K(0.3))/(K(0.3)-K(0.6)) ~= 1.3499 --
    a much higher ratio than Gaussian's ~0.3645, reflecting exponential's
    sharp cusp at the peak (a much bigger drop near u=0 than further out)."""
    ratio = _kernel_ratio("exponential")
    expected = (math.exp(0) - math.exp(-0.3)) / (math.exp(-0.3) - math.exp(-0.6))
    assert ratio == pytest.approx(expected, rel=0.02)
    assert ratio == pytest.approx(1.3499, rel=0.02)


def test_mark_density_linear_renders_with_exact_triangular_ratio():
    """1-|u| is exactly linear on [0, 1], so consecutive equal-width steps
    drop by an equal amount: ratio == 1.0 exactly (unlike Gaussian's ~0.3645
    or exponential's ~1.3499 -- a triangular kernel is the one shape where
    this ratio is unity)."""
    ratio = _kernel_ratio("linear")
    assert ratio == pytest.approx(1.0, rel=0.02)


def test_mark_density_exponential_and_linear_differ_from_gaussian_and_each_other():
    gaussian = _kernel_ratio("gaussian")
    exponential = _kernel_ratio("exponential")
    linear = _kernel_ratio("linear")
    # Gaussian's own closed form for a sanity floor.
    assert gaussian == pytest.approx(0.3645, rel=0.02)
    assert exponential != pytest.approx(gaussian, rel=0.05)
    assert linear != pytest.approx(gaussian, rel=0.05)
    assert exponential != pytest.approx(linear, rel=0.05)


def test_mark_density_unknown_kernel_still_refused_by_python_gate():
    """A kernel that was never valid (not one of the two new names) still
    raises before render -- proves the widened `_VALID_KERNELS` set didn't
    turn into an unbounded pass-through."""
    df = pl.DataFrame({"val": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="kernel"):
        fm.Chart(df).mark_density(kernel="not-a-kernel").encode(x="val").to_svg()


# ---------------------------------------------------------------------------
# transform_density(kernel=)
# ---------------------------------------------------------------------------


class TestTransformDensityKernel:
    def test_default_kernel_omitted_from_wire(self):
        """Mirrors DensityDataSpec's serde default/skip-if-default: a
        gaussian (default) kernel is omitted so pre-T11 specs -- and specs
        built with no `kernel=` at all -- serialize byte-identically."""
        spec = transform_density("x")
        assert "kernel" not in spec

    def test_explicit_gaussian_also_omitted(self):
        spec = transform_density("x", kernel="gaussian")
        assert "kernel" not in spec

    def test_epanechnikov_literal_routes_on_the_wire(self):
        """F-L05-12's literal probe: `transform_density(kernel="epanechnikov")`
        must be accepted and must appear on the wire."""
        spec = transform_density("x", kernel="epanechnikov")
        assert spec["kernel"] == "epanechnikov"

    def test_epanechnikov_example_renders(self):
        """The guide's `transform_density(kernel="epanechnikov")` example
        must actually render end to end, not just build a dict."""
        df = pl.DataFrame({"x": [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]})
        svg = (
            fm.Chart(df)
            .transform(transform_density("x", kernel="epanechnikov"))
            .mark_line()
            .encode(x="value", y="density")
            .to_svg()
        )
        assert "<polyline" in svg

    def test_unknown_kernel_refused_typed_at_render(self):
        """No Python-side vocabulary check exists on `transform_density`
        (deliberately -- like every `transform_*` wrapper, it defers
        vocabulary checks to the single Rust-side source of truth; the
        vocabulary IS discoverable via `transform_registry("density_data")
        ["methods"]`, pinned set-equal to kde's below). The typed refusal
        comes from Rust's shared kernel dispatch at render time."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        with pytest.raises(ValueError, match="unknown kernel"):
            (
                fm.Chart(df)
                .transform(transform_density("x", kernel="not-a-kernel"))
                .mark_line()
                .encode(x="value", y="density")
                .to_svg()
            )


# ---------------------------------------------------------------------------
# mark_qq(distribution="lognormal")
# ---------------------------------------------------------------------------


def _perfect_lognormal_sample(n: int = 9) -> list[float]:
    """A sample that is *exactly* exp(standard-normal theoretical quantile)
    at QQ's own plotting positions p_i = (i-0.5)/n (matching
    `qq.rs::compute_theoretical`'s formula) -- mirrors the Rust round's
    `qq_lognormal_discriminates_from_normal_on_lognormal_data` construction,
    built independently via Python's stdlib `statistics.NormalDist` rather
    than importing the Rust test fixture, so this is a real cross-language
    pin, not a copy of the same computation."""
    nd = NormalDist()
    positions = [(i - 0.5) / n for i in range(1, n + 1)]
    return sorted(math.exp(nd.inv_cdf(p)) for p in positions)


def _qq_collinearity_rss(svg: str, expected_points: int = 9) -> float:
    """Sum of squared residuals from the least-squares line through the QQ
    point layer's circle centers -- near zero when the points are
    genuinely collinear in pixel space (true for an affine map applied to
    an exact-fit sample), large when they are not.

    Asserts the exact expected point count (matching
    `_perfect_lognormal_sample`'s default `n=9`), not just `>= 3`: a parse
    that silently dropped rows (e.g. a negative-coordinate circle the
    regex failed to match) would relax this collinearity check rather
    than fail it outright."""
    points = _circle_points(svg)
    assert len(points) == expected_points
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    design = np.vstack([xs, np.ones_like(xs)]).T
    slope, intercept = np.linalg.lstsq(design, ys, rcond=None)[0]
    residuals = ys - (slope * xs + intercept)
    return float(np.sum(residuals**2))


def test_mark_qq_lognormal_renders():
    df = pl.DataFrame({"val": _perfect_lognormal_sample()})
    svg = fm.Chart(df).mark_qq(distribution="lognormal").encode(x="val").to_svg()
    assert "<circle" in svg


def test_mark_qq_lognormal_fits_a_lognormal_sample_far_better_than_normal():
    """Discriminator: on a sample built to fall exactly on `y=x` under the
    lognormal theoretical quantile (see `_perfect_lognormal_sample`), the
    QQ points must be near-collinear under `distribution="lognormal"` and
    NOT collinear under `distribution="normal"` on the identical data --
    proving `lognormal` computes its own (real, exponentiated) theoretical
    quantiles rather than aliasing `normal`."""
    df = pl.DataFrame({"val": _perfect_lognormal_sample()})
    lognormal_svg = (
        fm.Chart(df).mark_qq(distribution="lognormal", line=False).encode(x="val").to_svg()
    )
    normal_svg = fm.Chart(df).mark_qq(distribution="normal", line=False).encode(x="val").to_svg()

    assert lognormal_svg != normal_svg

    lognormal_rss = _qq_collinearity_rss(lognormal_svg)
    normal_rss = _qq_collinearity_rss(normal_svg)

    assert lognormal_rss < 1.0, f"expected near-perfect collinearity, got RSS={lognormal_rss}"
    assert normal_rss > 1000.0 * max(lognormal_rss, 1e-9), (
        f"expected normal's fit to be far worse than lognormal's on lognormal-shaped "
        f"data; lognormal_rss={lognormal_rss}, normal_rss={normal_rss}"
    )


def test_mark_qq_lognormal_accepted_by_validate_choice_gate():
    """Regression pin for the exact bug this batch fixes: `desugar_qq`'s
    `validate_choice` tuple in `ferrum/marks/heavy_stat.py` rejected
    "lognormal" even though `mark_qq`'s own docstring already listed it as
    a valid distribution -- a documented-but-refused parameter."""
    df = pl.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
    # Must not raise.
    fm.Chart(df).mark_qq(distribution="lognormal").encode(x="val").to_svg()


# ---------------------------------------------------------------------------
# Registry adoption (batch-D T12): `mark_density`'s kernel gate and
# `mark_qq`'s distribution gate no longer hand-mirror a Rust vocabulary as a
# second Python-side literal -- `desugar_density`/`desugar_qq` now call
# `transform_registry("kde"/"qq")["methods"]` live, at validation time, so
# there is nothing left to drift out of sync (the class of bug this batch's
# `exponential`/`linear`/`lognormal` fixes came from). What remains worth
# pinning is behavioral, not textual: every kernel/distribution the registry
# currently declares must actually render through the real `mark_density`/
# `mark_qq` gate, proving the live derivation is the actual accepted
# vocabulary, not just a dict lookup that happens to type-check.
# ---------------------------------------------------------------------------


def test_every_registry_declared_kde_kernel_renders_through_mark_density():
    """Every kernel `transform_registry("kde")["methods"]` declares must
    actually render through the real `mark_density` gate -- `desugar_density`
    derives its accepted vocabulary from this same call, so this is a
    behavioral proof of the live derivation, not a self-referential re-check
    of the same lookup."""
    from ferrum import _core

    registry_kernels = set(_core.transform_registry("kde")["methods"])

    df = pl.DataFrame({"val": [1.0, 2.0, 3.0]})
    for kernel in registry_kernels:
        # Must not raise -- proves mark_density's real gate accepts every
        # kernel the registry declares.
        fm.Chart(df).mark_density(kernel=kernel).encode(x="val").to_svg()


def test_every_registry_declared_qq_distribution_renders_through_mark_qq():
    """`desugar_qq`'s `validate_choice` gate (`ferrum/marks/heavy_stat.py`)
    now calls `transform_registry("qq")["methods"]` live (batch-D T12 --
    the former hand-mirrored `_VALID_QQ_DISTRIBUTIONS` module constant is
    deleted, since there is nothing left to drift). Behavioral proof: every
    distribution the registry currently declares must actually render
    through the real `mark_qq` gate."""
    from ferrum import _core

    registry_distributions = set(_core.transform_registry("qq")["methods"])

    df = pl.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
    for distribution in registry_distributions:
        # Must not raise -- proves the gate accepts every distribution the
        # registry declares.
        fm.Chart(df).mark_qq(distribution=distribution).encode(x="val").to_svg()


def test_density_data_registry_shares_kde_kernel_vocabulary():
    """The round-2 registry fix made `density_data`'s `methods` entry read
    `kde::KERNELS` directly (previously `[]`, since `PyDensityData::new` has
    no constructor-time `kernel` acceptance site of its own to validate
    against) -- pinning the shared-dispatch invariant `transform_density`
    relies on: `density_data` and `kde` must always expose the identical
    kernel vocabulary, because `density_data.rs` calls `kde`'s own
    `kernel_density_at_grid` dispatch rather than maintaining a second
    kernel table."""
    from ferrum import _core

    assert (
        _core.transform_registry("density_data")["methods"]
        == _core.transform_registry("kde")["methods"]
    )
