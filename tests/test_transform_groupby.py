"""Tests for `transform_regression(groupby=)` / `transform_loess(groupby=)`.

Covers M3a (finding F-L05-03): both transforms used to declare `groupby` on
the wire and silently ignore it — one global fit, the groupby column absent
from output, so `color="g:N"` render-errored with "unknown column 'g'". After
the fix, `regression_data.rs`/`loess_data.rs` partition via
`group_key::group_partition_multi` (the `smooth.rs` multi-column template),
fit each partition independently, and re-emit the groupby column(s) at their
original dtype.

Mirrors `test_smooth_groupby.py`'s structure (render-through, polyline-count
assertions) plus exact numeric verification for the linear-regression case,
since a hand-computed per-group fit is decidable and the finding's own repro
is reproduced verbatim (`TestSinCosRepro`).
"""

from __future__ import annotations

import hashlib
import re

import polars as pl
import pytest

import ferrum as fm

# ---------------------------------------------------------------------------
# Byte-identity pins (ungrouped path must be untouched by the groupby refactor)
# ---------------------------------------------------------------------------
#
# These hashes were captured from the SAME fixture and chart shape against
# the pre-refactor `apply()` (single global fit, no `apply_one_group`/
# `apply_grouped` split) before `regression_data.rs`/`loess_data.rs` were
# touched for M3a. A change to the ungrouped fit math, grid generation, or
# schema would move these hashes.
_UNGROUPED_FIXTURE = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 2.0, 4.0, 6.0]})
_REGRESSION_UNGROUPED_MD5 = "4bfa2a2c429abe3479cb747781624080"
_LOESS_UNGROUPED_MD5 = "8320296207e02c10e4e34e5a8fc4cfff"


def _polyline_points(svg: str) -> list[list[tuple[float, float]]]:
    """Extract every `<polyline points="...">` element's points, in document order."""
    out = []
    for pts_str in re.findall(r'<polyline points="([^"]+)"', svg):
        out.append([tuple(map(float, p.split(","))) for p in pts_str.split()])
    return out


def _polylines_with_stroke(svg: str) -> list[tuple[str, list[tuple[float, float]]]]:
    """Extract every `<polyline>` element's (stroke color, points), in document order."""
    out = []
    for pts_str, stroke in re.findall(r'<polyline points="([^"]+)"[^/]*stroke="([^"]+)"', svg):
        out.append((stroke, [tuple(map(float, p.split(","))) for p in pts_str.split()]))
    return out


def _legend_color_labels(svg: str) -> dict[str, str]:
    """Map legend swatch fill color -> label text, from adjacent `<circle fill=.../><text>label</text>`."""
    return dict(re.findall(r'<circle[^>]*fill="([^"]+)"[^>]*/><text[^>]*>([^<]*)</text>', svg))


class TestUngroupedByteIdentical:
    """The refactor must not move a single byte of ungrouped output.

    Byte-identity is pinned for healthy-row ungrouped inputs (4+ rows).
    See TestUngroupedDegenerateBehavior for the degenerate-row exception.
    """

    def test_regression_ungrouped_byte_identical(self):
        chart = (
            fm.Chart(_UNGROUPED_FIXTURE, width=400, height=300)
            .transform(fm.transform_regression("x", "y", method="linear", as_=("rx", "ry")))
            .mark_line()
            .encode(x="rx", y="ry")
        )
        svg = chart.to_svg()
        assert hashlib.md5(svg.encode()).hexdigest() == _REGRESSION_UNGROUPED_MD5, (
            "ungrouped transform_regression output moved — the groupby refactor "
            "must be byte-identical when groupby is unset"
        )

    def test_loess_ungrouped_byte_identical(self):
        chart = (
            fm.Chart(_UNGROUPED_FIXTURE, width=400, height=300)
            .transform(fm.transform_loess("x", "y", bandwidth=0.5, as_=("lx", "ly")))
            .mark_line()
            .encode(x="lx", y="ly")
        )
        svg = chart.to_svg()
        assert hashlib.md5(svg.encode()).hexdigest() == _LOESS_UNGROUPED_MD5, (
            "ungrouped transform_loess output moved — the groupby refactor "
            "must be byte-identical when groupby is unset"
        )


class TestSinCosRepro:
    """The finding's own repro (F-L05-03): sin/cos two-group regression."""

    def test_regression_groupby_sin_cos_renders_two_curves(self):
        import numpy as np

        xs = np.linspace(0, 4 * np.pi, 20)
        df = pl.DataFrame(
            {
                "g": ["a"] * 20 + ["b"] * 20,
                "x": list(xs) * 2,
                "y": np.sin(xs).tolist() + np.cos(xs).tolist(),
            }
        )
        t = fm.transform_regression("x", "y", groupby=["g"], as_=("rx", "ry"))
        svg = fm.Chart(df).transform(t).mark_line().encode(x="rx", y="ry", color="g:N").to_svg()
        assert svg.startswith("<svg")
        polylines = _polyline_points(svg)
        assert len(polylines) == 2, (
            f"expected one colored curve per group ('a', 'b'), got {len(polylines)}"
        )
        # Geometric distinctness, not just count: a regression that re-emits
        # the group column but fits ONE global curve for both groups would
        # also produce 2 identical polylines here.
        sin_pts, cos_pts = polylines
        max_dy = max(abs(sp[1] - cp[1]) for sp, cp in zip(sin_pts, cos_pts))
        assert max_dy > 1.0, (
            f"sin and cos per-group fits must diverge (max |Δy|={max_dy} px) — "
            "identical curves indicate a shared/blended fit, not per-group"
        )


class TestRegressionGroupbyHandComputed:
    """Exact per-group linear fits vs. a global blended fit (discriminating).

    Group A: y = 2x (exact, no noise). Group B: y = 10 - 3x (exact).
    A regression that ignores groupby computes ONE global OLS fit over all
    8 points: mean_x=1.5, mean_y=4.25, slope=-0.5, intercept=5.0 — i.e.
    y = 5.0 - 0.5x, which agrees with NEITHER per-group line. A correct
    per-group fit must reproduce each group's exact slope/intercept.

    x and y scales are given EXPLICIT domain/range (padded past the exact
    data extent to dodge float-epsilon domain-boundary clipping), so the
    rendered pixel coordinates decode back to data space via the identical
    `LinearScale.invert()` used to build the encoding — a production
    entry point, not a hand-rolled inversion formula.
    """

    def _grouped_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "g": ["A", "A", "A", "A", "B", "B", "B", "B"],
                "x": [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0],
                "y": [0.0, 2.0, 4.0, 6.0, 10.0, 7.0, 4.0, 1.0],
            }
        )

    def test_per_group_fit_matches_hand_computation_not_global_blend(self):
        df = self._grouped_df()
        xscale = fm.LinearScale(domain=(-0.5, 3.5), range=(0.0, 300.0))
        yscale = fm.LinearScale(domain=(-1.0, 11.0), range=(0.0, 100.0))

        t = fm.transform_regression("x", "y", method="linear", groupby=["g"], as_=("rx", "ry"))
        chart = (
            fm.Chart(df, width=400, height=300)
            .transform(t)
            .mark_line()
            .encode(x=fm.X("rx", scale=xscale), y=fm.Y("ry", scale=yscale), color="g:N")
        )
        svg = chart.to_svg()
        polylines_with_stroke = _polylines_with_stroke(svg)
        assert len(polylines_with_stroke) == 2, (
            f"expected 2 per-group fit lines, got {len(polylines_with_stroke)}"
        )

        # Bind each polyline to its group by legend swatch color rather than
        # by render order, so a future render-order change fails with a
        # legible "no line for group A" message instead of a silent
        # positional mismatch.
        color_to_label = _legend_color_labels(svg)
        pts_by_label = {color_to_label[stroke]: pts for stroke, pts in polylines_with_stroke}
        assert set(pts_by_label) == {"A", "B"}, (
            f"expected lines for A and B, got {set(pts_by_label)}"
        )
        a_pts, b_pts = pts_by_label["A"], pts_by_label["B"]
        assert len(a_pts) == 100 and len(b_pts) == 100, "regression grid defaults to 100 points"

        approx = pytest.approx
        ax0, ay0 = xscale.invert(a_pts[0][0]), yscale.invert(a_pts[0][1])
        ax1, ay1 = xscale.invert(a_pts[-1][0]), yscale.invert(a_pts[-1][1])
        bx0, by0 = xscale.invert(b_pts[0][0]), yscale.invert(b_pts[0][1])
        bx1, by1 = xscale.invert(b_pts[-1][0]), yscale.invert(b_pts[-1][1])

        # Group A: y = 2x exactly, over its own x range [0, 3].
        assert ax0 == approx(0.0, abs=1e-3) and ay0 == approx(0.0, abs=1e-3)
        assert ax1 == approx(3.0, abs=1e-3) and ay1 == approx(6.0, abs=1e-3)

        # Group B: y = 10 - 3x exactly, over its own x range [0, 3].
        assert bx0 == approx(0.0, abs=1e-3) and by0 == approx(10.0, abs=1e-3)
        assert bx1 == approx(3.0, abs=1e-3) and by1 == approx(1.0, abs=1e-3)

        # Neither per-group endpoint matches the blended global fit
        # (y = 5.0 - 0.5x -> (0, 5.0), (3, 3.5)) — proves independence.
        assert abs(ay0 - 5.0) > 1.0
        assert abs(by0 - 5.0) > 1.0


class TestLoessGroupbySmoke:
    """LOESS groupby smoke test: per-group monotone curves render separately."""

    def test_loess_groupby_renders_per_group_monotone_curves(self):
        n = 20
        xs = [float(i) for i in range(n)]
        df = pl.DataFrame(
            {
                "g": ["up"] * n + ["down"] * n,
                "x": xs + xs,
                "y": [2.0 * x for x in xs] + [20.0 - 1.5 * x for x in xs],
            }
        )
        t = fm.transform_loess("x", "y", bandwidth=0.5, groupby=["g"], as_=("lx", "ly"))
        chart = fm.Chart(df).transform(t).mark_line().encode(x="lx", y="ly", color="g:N")
        svg = chart.to_svg()
        assert svg.startswith("<svg")
        polylines = _polyline_points(svg)
        assert len(polylines) == 2

        # Per-group monotonicity: "up" group's y-pixels move one direction,
        # "down" group's the opposite. y is SVG-down by ferrum's default
        # orientation, so we only check sign consistency (monotone), not
        # a specific direction.
        for pts in polylines:
            ys = [p[1] for p in pts]
            diffs = [b - a for a, b in zip(ys, ys[1:])]
            all_nonneg = all(d >= -1e-6 for d in diffs)
            all_nonpos = all(d <= 1e-6 for d in diffs)
            assert all_nonneg or all_nonpos, "each group's LOESS curve must be monotone"

        # Discriminating check: the two curves must run in OPPOSITE overall
        # directions, not merely each be independently monotone. A regression
        # that re-emits the group column but fits ONE global LOESS curve for
        # both groups would produce two identical monotone curves here (the
        # blended mean over this fixture, 10 + 0.25x, is itself monotone) —
        # same count (2), same per-curve monotonicity, wrong distinctness.
        up_pts, down_pts = polylines
        up_delta = up_pts[-1][1] - up_pts[0][1]
        down_delta = down_pts[-1][1] - down_pts[0][1]
        assert up_delta * down_delta < 0, (
            "'up' and 'down' groups must move in opposite pixel directions; "
            f"got deltas {up_delta} and {down_delta} — a blended global fit "
            "would produce two curves moving the same way"
        )


class TestMultiColumnGroupby:
    """Two groupby columns partition into the cross-product of distinct combos."""

    def test_regression_groupby_two_columns(self):
        n = 5
        xs = [float(i) for i in range(n)]
        combos = [("A", 1), ("A", 2), ("B", 1), ("B", 2)]
        rows = []
        for g1, g2 in combos:
            slope = 1.0 if g1 == "A" else 2.0
            for x in xs:
                rows.append({"g1": g1, "g2": g2, "x": x, "y": slope * x + g2})
        df = pl.DataFrame(rows)

        t = fm.transform_regression("x", "y", groupby=["g1", "g2"], as_=("rx", "ry"))
        chart = (
            fm.Chart(df)
            .transform(t)
            .mark_line()
            .encode(x="rx", y="ry", color="g1:N", detail="g2:N")
        )
        svg = chart.to_svg()
        assert svg.startswith("<svg")
        polylines = _polyline_points(svg)
        assert len(polylines) == len(combos), (
            f"expected {len(combos)} distinct (g1, g2) fit lines, got {len(polylines)}"
        )
        # Geometric distinctness: each combo's fit line must differ from
        # every other's, not merely exist as a separate <polyline> element
        # (a regression could emit N identical polylines and still pass a
        # count-only check).
        for i, pi in enumerate(polylines):
            for pj in polylines[i + 1 :]:
                assert pi != pj, "two distinct (g1, g2) combos rendered identical fit lines"

    def test_loess_groupby_two_columns(self):
        n = 10
        xs = [float(i) for i in range(n)]
        combos = [("A", 1), ("A", 2), ("B", 1), ("B", 2)]
        rows = []
        for g1, g2 in combos:
            slope = 1.0 if g1 == "A" else 2.0
            for x in xs:
                rows.append({"g1": g1, "g2": g2, "x": x, "y": slope * x + g2})
        df = pl.DataFrame(rows)

        t = fm.transform_loess("x", "y", bandwidth=0.5, groupby=["g1", "g2"], as_=("lx", "ly"))
        chart = (
            fm.Chart(df)
            .transform(t)
            .mark_line()
            .encode(x="lx", y="ly", color="g1:N", detail="g2:N")
        )
        svg = chart.to_svg()
        assert svg.startswith("<svg")
        polylines = _polyline_points(svg)
        assert len(polylines) == len(combos)
        # Geometric distinctness alongside the count (see the regression
        # sibling test above for why count alone doesn't discriminate).
        for i, pi in enumerate(polylines):
            for pj in polylines[i + 1 :]:
                assert pi != pj, "two distinct (g1, g2) combos rendered identical fit lines"


class TestDegenerateGroupPreserved:
    """A below-fit-threshold group is preserved (row count + group key),
    not silently dropped from the output/legend/domain.

    Batch-D T6 round 2 (rust-quality Required Fix 1): a group with fewer
    than 2 usable rows (regression) or 3 (loess) cannot support a fit.
    `apply_one_group` emits N_OUT NaN-valued rows for it instead of an
    empty batch, mirroring `smooth.rs::all_nan_output` — so the group keeps
    its row count and group key in the concatenated output. The NaN rows
    draw no visible curve (mark_line filters non-finite points before
    building the polyline), but the group's identity survives into the
    color-scale domain and the legend.
    """

    def test_regression_below_threshold_group_keeps_legend_entry(self):
        # Group A: 4 healthy rows (exact fit). Group B: 1 row — below the
        # <2-point regression threshold.
        df = pl.DataFrame(
            {
                "g": ["A", "A", "A", "A", "B"],
                "x": [0.0, 1.0, 2.0, 3.0, 5.0],
                "y": [0.0, 2.0, 4.0, 6.0, 9.0],
            }
        )
        t = fm.transform_regression("x", "y", groupby=["g"], as_=("rx", "ry"))
        chart = fm.Chart(df).transform(t).mark_line().encode(x="rx", y="ry", color="g:N")
        svg = chart.to_svg()

        # Only A produces a visible curve; B's NaN grid contributes no
        # renderable line segment.
        assert len(_polyline_points(svg)) == 1, "only the healthy group should draw a curve"
        # But B is NOT dropped from the output: its legend entry survives.
        labels = set(_legend_color_labels(svg).values())
        assert labels == {"A", "B"}, (
            f"below-threshold group B must keep its legend entry, got {labels}"
        )

    def test_loess_below_threshold_group_keeps_legend_entry(self):
        # Group A: 10 healthy rows. Group B: 2 rows — below the <3-point
        # LOESS threshold.
        n = 10
        xs = [float(i) for i in range(n)]
        df = pl.DataFrame(
            {
                "g": ["A"] * n + ["B"] * 2,
                "x": xs + [0.0, 1.0],
                "y": [2.0 * x for x in xs] + [5.0, 7.0],
            }
        )
        t = fm.transform_loess("x", "y", bandwidth=0.5, groupby=["g"], as_=("lx", "ly"))
        chart = fm.Chart(df).transform(t).mark_line().encode(x="lx", y="ly", color="g:N")
        svg = chart.to_svg()

        assert len(_polyline_points(svg)) == 1, "only the healthy group should draw a curve"
        labels = set(_legend_color_labels(svg).values())
        assert labels == {"A", "B"}, (
            f"below-threshold group B must keep its legend entry, got {labels}"
        )


class TestUngroupedDegenerateBehavior:
    """Batch-D T6 round 2 (python-quality Required Fix 2): ungrouped
    below-threshold inputs yield the 100-row NaN grid, not 0 rows.

    When `groupby=None` (ungrouped) and the input frame has fewer than
    the fit threshold (2 rows for regression, 3 for loess), `apply_one_group`
    emits `all_nan_output` — mirroring the grouped-degenerate and `smooth.rs`
    templates. This is a deliberate family-aligned choice, not a side effect.
    The NaN rows draw no visible curve (mark_line filters non-finite points
    before building polylines), but the output shape is preserved.
    """

    def test_regression_ungrouped_below_threshold_renders_no_curve(self):
        # Only 1 row: below the <2-point regression threshold.
        # The transform emits a 100-row NaN-valued output (not 0 rows or an error).
        df = pl.DataFrame(
            {
                "x": [0.5],
                "y": [1.0],
            }
        )
        t = fm.transform_regression("x", "y", method="linear", as_=("rx", "ry"))
        chart = fm.Chart(df).transform(t).mark_line().encode(x="rx", y="ry")
        svg = chart.to_svg()

        # Observable: no visible polyline from NaN rows.
        assert len(_polyline_points(svg)) == 0, (
            "degenerate ungrouped regression should emit NaN output that draws no curve"
        )
        # Sanity: the SVG is well-formed and renders (axes exist).
        assert svg.startswith("<svg")

    def test_loess_ungrouped_below_threshold_renders_no_curve(self):
        # Only 2 rows: below the <3-point LOESS threshold.
        # The transform emits a 100-row NaN-valued output (not 0 rows or an error).
        df = pl.DataFrame(
            {
                "x": [0.0, 1.0],
                "y": [1.0, 2.0],
            }
        )
        t = fm.transform_loess("x", "y", bandwidth=0.5, as_=("lx", "ly"))
        chart = fm.Chart(df).transform(t).mark_line().encode(x="lx", y="ly")
        svg = chart.to_svg()

        assert len(_polyline_points(svg)) == 0, (
            "degenerate ungrouped loess should emit NaN output that draws no curve"
        )
        assert svg.startswith("<svg")


class TestGroupDtypePreservation:
    """An int-typed groupby column stays int end to end (not stringified)."""

    def test_regression_int_groupby_column_stays_quantitative_by_default(self):
        # An unannotated int-typed color channel infers a CONTINUOUS
        # (quantitative) color scale, which ferrum warns is a no-op on
        # `mark_line` (no per-mark effect). That warning firing is the
        # production-entry-point signal that the output group column is
        # still Int64 — a nominal/categorical inference (the warning would
        # NOT fire) would only happen if the group column had been coerced
        # to a string somewhere in the pipeline.
        n = 4
        rows = []
        for g in (10, 20):
            for i in range(n):
                x = float(i)
                y = (2.0 if g == 10 else -1.0) * x + g
                rows.append({"g": g, "x": x, "y": y})
        df = pl.DataFrame(rows)

        t = fm.transform_regression("x", "y", groupby=["g"], as_=("rx", "ry"))
        chart = fm.Chart(df).transform(t).mark_line().encode(x="rx", y="ry", color="g")
        # Coupling note: this warning text is `RenderWarning::UnsupportedColorScaleOnMark`
        # (crates/ferrum-core/src/render/mod.rs). If that warning's wording or
        # its "continuous scale on a mark that ignores it" trigger condition
        # changes, this dtype pin needs to move with it — it is a proxy for
        # "still Int64", not a direct dtype assertion.
        with pytest.warns(UserWarning, match="continuous color scale is not supported on line"):
            svg = chart.to_svg()
        assert svg.startswith("<svg")

    def test_regression_int_groupby_column_partitions_as_nominal(self):
        # With an explicit ":N" hint, the SAME int group column drives
        # per-group line splitting — proving the two groups' rows are
        # distinguishable by the int group column's actual values (10, 20),
        # not merely by row order.
        n = 4
        rows = []
        for g in (10, 20):
            for i in range(n):
                x = float(i)
                y = (2.0 if g == 10 else -1.0) * x + g
                rows.append({"g": g, "x": x, "y": y})
        df = pl.DataFrame(rows)

        t = fm.transform_regression("x", "y", groupby=["g"], as_=("rx", "ry"))
        chart = fm.Chart(df).transform(t).mark_line().encode(x="rx", y="ry", color="g:N")
        svg = chart.to_svg()
        assert svg.startswith("<svg")
        polylines = _polyline_points(svg)
        assert len(polylines) == 2, (
            f"expected 2 per-group fit lines (g=10, g=20), got {len(polylines)}"
        )
        # Geometric distinctness: the two fit lines must differ, not be identical
        # (a regression could emit N identical polylines and still pass a count check).
        assert polylines[0] != polylines[1], (
            "the two int-group partitions (10, 20) rendered identical fit lines"
        )
