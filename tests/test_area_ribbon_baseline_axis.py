"""`#172`(2)+(3): the area fill and the ribbon band follow their own axis.

Two mark drawers hard-coded a y-axis idiom that `CoordFlip` invalidated.

`mark_area` closed every fill on the panel BOTTOM, so a flipped area shaded
very nearly the complement of its region (measured pre-fix on the flipped
fixture below: ``L343.265 420.961 L206.898 420.961 Z`` — two points sharing the
panel-bottom y instead of one vertical segment at the x baseline). The same
hard-code sits under `mark_density(orientation="horizontal")`, which reaches it
with NO flip at all: that desugar remaps the density onto `x` directly and, at
the default ``multiple="layer"``, stamps no stack either, so nothing in the
spec declared the series horizontal. The fix gives producers a way to say so —
the ``value_axis`` mark kwarg, stamped by `desugar_density`
(`src/ferrum/marks/statistical.py`) for `orientation="horizontal"` — and this
module's job on that half is the GUARD: an area that declares nothing is
vertical however its rows are ordered. The horizontal-density end-to-end
closure is `test_horizontal_density_closes_on_its_own_left_baseline`; the
`fill=False` (line) twin's wire-level pin is
`test_horizontal_density_line_declares_value_axis_on_the_wire`.

`mark_ribbon` read only `encoding.y2`, and `flip_positional` moves an authored
`y2` to `x2` — so a flipped ribbon emitted zero paths, silently, taking
`mark_errorband`, `mark_smooth(ci=)`, `mark_learning_curve`,
`mark_validation_curve` and the QQ CI region with it.

Every assertion below is on rendered geometry: which coordinate the closure
segment holds constant, and which coordinate a band's vertices pair off on.
Asserting that a path merely exists would pass for a fill closed on the wrong
edge, which is exactly the shape that shipped.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

import ferrum as fm

from _svg_corpus import path_points
from _svg_rects import plot_rect, svg_elements

# Tolerance for "these two rendered coordinates are the same edge". Pixel
# coordinates are emitted rounded to 3 decimals, so anything below ~1e-3 would
# be asserting on the formatter rather than on the geometry.
EPS = 1e-3


def filled_path_vertices(svg: str) -> list[tuple[float, float]]:
    """Vertices of the chart's single filled ``<path>``, in emission order."""
    paths = svg_elements(
        svg,
        "path",
        lambda a: "d" in a and a.get("fill") not in (None, "none"),
    )
    assert len(paths) == 1, f"expected exactly one filled path, got {len(paths)}"
    return path_points(paths[0]["d"])


def stroked_path_ds(svg: str) -> list[str]:
    """The ``d`` of every unfilled (border) ``<path>``, in emission order."""
    return [
        a["d"] for a in svg_elements(svg, "path", lambda a: "d" in a and a.get("fill") == "none")
    ]


@pytest.fixture()
def wiggle() -> pl.DataFrame:
    """Five rows whose y turns twice, so no closure can coincide with the curve."""
    return pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.0, 4.0, 1.0, 5.0, 3.0]})


# ---------------------------------------------------------------------------
# #172(2) — the area fill's closure
# ---------------------------------------------------------------------------


def test_unflipped_area_closes_on_the_panel_bottom(wiggle):
    """The negative control, and the byte-stability half of the fix.

    An ordinary vertical area's two closing vertices share the panel-bottom y
    and sit at the curve's own last and first x. Nothing about `#172`(2) may
    move this, so it is asserted as geometry rather than assumed.
    """
    svg = fm.Chart(wiggle).mark_area().encode(x="x:Q", y="y:Q").to_svg()
    px, py, pw, ph = plot_rect(svg)
    verts = filled_path_vertices(svg)

    last_close, first_close = verts[-2], verts[-1]
    assert abs(last_close[1] - (py + ph)) < EPS, (
        f"the closure must land on the panel bottom y={py + ph}; got {last_close}"
    )
    assert abs(first_close[1] - (py + ph)) < EPS, f"…on both ends; got {first_close}"
    assert abs(last_close[0] - verts[-3][0]) < EPS, (
        "the first closing segment drops straight down from the curve's last point"
    )
    assert abs(first_close[0] - verts[0][0]) < EPS, "and the second returns to the curve's first x"


def test_coord_flipped_area_closes_on_the_x_baseline(wiggle):
    """`#172`(2) headline: a flipped fill closes on the LEFT edge, vertically.

    Pre-fix the two closing vertices shared the panel-bottom y
    (``L343.265 420.961 L206.898 420.961 Z``), which shades the region to the
    right of the curve — nearly the complement of the correct one. After the
    fix they share the plot rect's x and keep the curve's own last and first y.
    """
    svg = fm.Chart(wiggle).mark_area().encode(x="x:Q", y="y:Q").coord(fm.CoordFlip()).to_svg()
    px, py, pw, ph = plot_rect(svg)
    verts = filled_path_vertices(svg)

    last_close, first_close = verts[-2], verts[-1]
    assert abs(last_close[0] - px) < EPS, (
        f"the closure must land on the x baseline (plot left, x={px}); got {last_close}"
    )
    assert abs(first_close[0] - px) < EPS, f"…on both ends; got {first_close}"
    assert abs(last_close[1] - verts[-3][1]) < EPS, (
        "the first closing segment runs straight across from the curve's last point"
    )
    assert abs(first_close[1] - verts[0][1]) < EPS, "and the second returns to the curve's first y"
    assert abs(last_close[1] - first_close[1]) > EPS, (
        "the closing segment is VERTICAL and non-degenerate — a bottom-closed fill "
        f"would instead have these two share a y; got {last_close} and {first_close}"
    )


@pytest.mark.parametrize(
    ("name", "xs", "ys"),
    [
        # The three shapes measured on the settled tree. The first is an
        # ordinary area; the second and third are the ones an inference got
        # wrong, and they are here as the guard against re-introducing one.
        ("sorted x", [1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 1.0, 5.0, 3.0]),
        ("unsorted x", [3.0, 1.0, 5.0, 2.0, 4.0], [2.0, 4.0, 1.0, 5.0, 3.0]),
        ("monotone y, unsorted x", [3.0, 1.0, 5.0, 2.0, 4.0], [1.0, 2.0, 3.0, 4.0, 5.0]),
    ],
)
def test_an_undeclared_area_closes_on_the_panel_bottom_whatever_its_row_order(name, xs, ys):
    """The guard that decided `#172`(2)'s mechanism.

    An area whose producer declares nothing — no `CoordFlip`, no `Stack`, no
    ``value_axis`` — is vertical, and stays vertical however its rows are
    ordered. Ferrum traces in row order and never sorts, so all three shapes
    below are ordinary user data.

    `#172`(2)'s first attempt inferred the value axis from the traced edge's
    monotonicity instead of taking a declaration. That inference cannot
    distinguish rows 2 and 3 here from a genuinely horizontal series, and
    measured, it closed both of them on the panel's LEFT edge — a silent wrong
    render. The inference is gone; this is what keeps it gone.
    """
    df = pl.DataFrame({"x": xs, "y": ys})
    svg = fm.Chart(df).mark_area().encode(x="x:Q", y="y:Q").to_svg()
    px, py, pw, ph = plot_rect(svg)
    verts = filled_path_vertices(svg)

    last_close, first_close = verts[-2], verts[-1]
    assert abs(last_close[1] - (py + ph)) < EPS and abs(first_close[1] - (py + ph)) < EPS, (
        f"{name}: an undeclared area closes on the panel bottom (y={py + ph}); got "
        f"{last_close} and {first_close}"
    )


def test_vertical_density_closes_on_the_panel_bottom():
    """The orientation control for the density family.

    ``mark_density()`` with no orientation is vertical and must keep closing on
    the panel bottom. Its horizontal twin is
    `test_horizontal_density_closes_on_its_own_left_baseline`, below.
    """
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"v": rng.normal(size=400)})
    svg = fm.Chart(df).mark_density().encode(x="v:Q").to_svg()
    px, py, pw, ph = plot_rect(svg)
    verts = filled_path_vertices(svg)

    last_close, first_close = verts[-2], verts[-1]
    assert abs(last_close[1] - (py + ph)) < EPS and abs(first_close[1] - (py + ph)) < EPS, (
        f"a vertical density closes on the panel bottom (y={py + ph}); got "
        f"{last_close} and {first_close}"
    )


def test_horizontal_density_closes_on_its_own_left_baseline():
    """`#172`(2)'s headline case on the real production path.

    Pre-stamp, `mark_density(orientation="horizontal")` reached the same
    hard-coded panel-bottom closure as an undeclared area (this is exactly
    the shape measured in `tests/test_phase_9_e2e/goldens/jointplot_kde_marginals.svg`
    before this task: ``... L75.842 24 L75.842 420.961 L112.33 420.961 Z``,
    two points sharing the panel-bottom y). `desugar_density`
    (`src/ferrum/marks/statistical.py`) now stamps
    ``mark_kwargs={"value_axis": "x"}`` for this orientation, which reaches
    `DrawCtx::series_curve_axis` through `MarkKwargsSpec.value_axis` (the
    wire contract published in `.sdd/task-5-report.md`) and moves the
    closure to a vertical segment on the plot's own left edge — the fill's
    own baseline, not the panel's.
    """
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"v": rng.normal(size=400)})
    svg = fm.Chart(df).mark_density(orientation="horizontal").encode(y="v:Q").to_svg()
    px, py, pw, ph = plot_rect(svg)
    verts = filled_path_vertices(svg)

    last_close, first_close = verts[-2], verts[-1]
    assert abs(last_close[0] - px) < EPS and abs(first_close[0] - px) < EPS, (
        f"a horizontal density closes on its own x baseline (plot left, x={px}); got "
        f"{last_close} and {first_close}"
    )
    assert abs(last_close[1] - first_close[1]) > EPS, (
        "the closing segment is VERTICAL and non-degenerate — a bottom-closed fill "
        f"would instead have these two share a y; got {last_close} and {first_close}"
    )


def test_a_users_flip_cancels_the_producers_horizontal_orientation():
    """The two flips cancel: ``orientation="horizontal"`` + ``CoordFlip`` is vertical.

    ``value_axis`` is stamped by `desugar_density` against its PRE-flip
    encoding (density on `x`), and read by `DrawCtx::series_curve_axis` against
    the POST-flip one. `LayerPrepared::flip_coords` moves it into post-flip
    space along with the channels themselves, so the two flips cancel and this
    chart is an ordinary vertical area.

    Left unflipped, the declaration and `coord_flipped()` both reported
    "horizontal" for a post-flip encoding (``x``=observation, ``y``=density)
    that is nothing of the sort, and the fill closed on the panel's LEFT edge.
    Measured on the pre-fix build: ``(69.303, 410.089) -> (69.303, 412.961)``,
    a near-degenerate 2.9px sliver at the left edge — nearly invisible on a
    density whose tails vanish, which is why reading the path tail alone was
    inconclusive and why this test asserts the SIDE-BY-SIDE instead.

    The plain vertical density is rendered on identical data and its closure
    compared vertex for vertex. Both charts have the same post-flip encoding,
    so any difference between them is the bug; asserting "closes on the bottom"
    alone would have passed on a fill that closed at the wrong x too.
    """
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"v": rng.normal(size=400)})

    flipped_svg = (
        fm.Chart(df)
        .mark_density(orientation="horizontal")
        .encode(y="v:Q")
        .coord(fm.CoordFlip())
        .to_svg()
    )
    plain_svg = fm.Chart(df).mark_density().encode(x="v:Q").to_svg()

    assert plot_rect(flipped_svg) == plot_rect(plain_svg), (
        "test premise: the two charts must lay out identically, or a closure "
        "difference below could be a layout difference instead"
    )
    px, py, pw, ph = plot_rect(flipped_svg)
    flipped = filled_path_vertices(flipped_svg)[-2:]
    plain = filled_path_vertices(plain_svg)[-2:]

    assert all(abs(v[1] - (py + ph)) < EPS for v in flipped), (
        f"a producer-horizontal density under CoordFlip is VERTICAL — the two flips "
        f"cancel — so it closes on the panel bottom (y={py + ph}); got {flipped}"
    )
    assert [(round(x, 3), round(y, 3)) for x, y in flipped] == [
        (round(x, 3), round(y, 3)) for x, y in plain
    ], (
        "…and vertex for vertex it closes exactly where the plain vertical density on "
        f"the same data closes: flipped {flipped} vs vertical {plain}"
    )


def test_horizontal_density_line_declares_value_axis_on_the_wire():
    """`mark_density(fill=False, orientation="horizontal")` desugars to
    `mark="line"`, not `"area"` — a different consumer entirely
    (`render/marks/line.rs`, not `render/marks/area.rs`). Task 5's wire
    contract claims both read the SAME `DrawCtx::series_curve_axis` ("the
    consumer is... the whole line family"); this is the assertion in this
    module that verifies the line's half of that claim rather than assuming
    it, by pinning the stamp onto the compiled wire spec for the `"line"`
    mark specifically.

    A rendered-geometry check (the style used above for the area case) is
    not available here: `mark_density`'s default interpolation is
    ``"linear"``, which emits a plain `<polyline>`, and reflecting a
    straight point-to-point path through `CurveAxis::X` vs `CurveAxis::Y` is
    the identity — the two axes draw byte-identical polylines. Widening
    `mark_density`'s honored kwargs to accept `interpolate=` (needed to force
    a `monotone`/`step` curve, whose control points DO encode the axis; see
    `curve_axis.rs`'s `observe_monotone_trace`) would be a public-API change
    outside this task's scope, so the wire-level pin is the correct,
    discriminating substitute.
    """
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"v": rng.normal(size=200)})
    spec = fm.Chart(df).mark_density(fill=False, orientation="horizontal").encode(y="v:Q").to_spec()
    payload = json.loads(spec.to_json())

    assert payload["mark"] == "line", (
        f"fill=False must desugar to a line mark, the other series_curve_axis consumer; "
        f"got {payload['mark']!r}"
    )
    assert payload.get("mark_style", {}).get("value_axis") == "x", (
        f"the horizontal line must declare value_axis='x' on the wire; got "
        f"{payload.get('mark_style')}"
    )


def test_vertical_density_line_omits_value_axis_on_the_wire():
    """The byte-stability control for the wire check above.

    A `fill=False` density with no (or vertical) orientation must NOT carry
    `mark_style.value_axis` at all — absence, not `"y"` — because
    `MarkKwargsSpec.value_axis` is
    `#[serde(default, skip_serializing_if = "Option::is_none")]` and every
    payload written before this field existed must stay byte-identical
    (task 5's wire contract).
    """
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"v": rng.normal(size=200)})
    spec = fm.Chart(df).mark_density(fill=False).encode(x="v:Q").to_spec()
    payload = json.loads(spec.to_json())

    assert payload["mark"] == "line"
    assert "value_axis" not in payload.get("mark_style", {}), payload.get("mark_style")


def test_area_edge_borders_outline_the_baseline_the_fill_closes_on(wiggle):
    """S10 (`borders=True`) carried the identical hard-code and moves with it.

    The border is an OPEN two-point segment; it must trace the same baseline
    the fill closes on, in both orientations. A border left on the panel bottom
    under a flip would draw a line the fill does not touch.
    """
    base = fm.Chart(wiggle).mark_area(borders=True).encode(x="x:Q", y="y:Q")

    svg = base.to_svg()
    px, py, pw, ph = plot_rect(svg)
    baseline_border = path_points(stroked_path_ds(svg)[-1])
    assert len(baseline_border) == 2, f"the baseline border is one segment; got {baseline_border}"
    assert all(abs(p[1] - (py + ph)) < EPS for p in baseline_border), (
        f"unflipped, it runs along the panel bottom y={py + ph}; got {baseline_border}"
    )

    svg = base.coord(fm.CoordFlip()).to_svg()
    px, py, pw, ph = plot_rect(svg)
    baseline_border = path_points(stroked_path_ds(svg)[-1])
    assert len(baseline_border) == 2, f"the baseline border is one segment; got {baseline_border}"
    assert all(abs(p[0] - px) < EPS for p in baseline_border), (
        f"flipped, it runs along the x baseline x={px}; got {baseline_border}"
    )
    fill_verts = filled_path_vertices(svg)
    assert {round(p[1], 3) for p in baseline_border} == {
        round(fill_verts[-2][1], 3),
        round(fill_verts[-1][1], 3),
    }, "and it spans exactly the two points the fill closes on"


# ---------------------------------------------------------------------------
# #172(3) — the ribbon band
# ---------------------------------------------------------------------------


@pytest.fixture()
def band() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "lo": [2.0, 4.0, 1.0, 5.0, 3.0],
            "hi": [3.0, 5.0, 2.0, 6.0, 4.0],
        }
    )


def band_pairs(
    verts: list[tuple[float, float]],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Pair each lower-edge vertex with the upper-edge vertex that closes it.

    A ribbon's ring walks the lower edge forward then the upper edge back, so
    vertex ``k`` pairs with vertex ``n - 1 - k``.
    """
    n = len(verts)
    assert n % 2 == 0, f"a band's ring has an even vertex count; got {n}"
    return [(verts[k], verts[n - 1 - k]) for k in range(n // 2)]


def test_unflipped_ribbon_spans_y_at_a_shared_x(band):
    """The control: an authored ``y``/``y2`` band pairs off on x."""
    svg = fm.Chart(band).mark_ribbon().encode(x="x:Q", y="lo:Q", y2="hi:Q").to_svg()
    pairs = band_pairs(filled_path_vertices(svg))
    assert len(pairs) == 5, f"one pair per row; got {len(pairs)}"
    for lower, upper in pairs:
        assert abs(lower[0] - upper[0]) < EPS, (
            f"a y..y2 band spans y at a SHARED x; got {lower} and {upper}"
        )
        assert abs(lower[1] - upper[1]) > EPS, (
            f"and the y difference IS the band; got {lower} and {upper}"
        )


def test_coord_flipped_ribbon_spans_x_at_a_shared_y(band):
    """`#172`(3) headline: a flipped ribbon draws, y-parameterized.

    Pre-fix this chart emitted ZERO paths with no warning — `flip_positional`
    had moved the authored ``y2`` to ``x2``, which the drawer did not read.
    Asserting the pairing rather than the path count is what separates "a band
    is drawn" from "the band is drawn across the right axis": a band built
    against x would pair its vertices on x here too.
    """
    svg = (
        fm.Chart(band)
        .mark_ribbon()
        .encode(x="x:Q", y="lo:Q", y2="hi:Q")
        .coord(fm.CoordFlip())
        .to_svg()
    )
    pairs = band_pairs(filled_path_vertices(svg))
    assert len(pairs) == 5, f"one pair per row; got {len(pairs)}"
    for lower, upper in pairs:
        assert abs(lower[1] - upper[1]) < EPS, (
            f"a flipped band spans x at a SHARED y; got {lower} and {upper}"
        )
        assert abs(lower[0] - upper[0]) > EPS, (
            f"and the x difference IS the band; got {lower} and {upper}"
        )


@pytest.mark.parametrize("build", ["errorband", "smooth_ci"])
def test_ribbon_consumers_recover_under_coord_flip(build):
    """The marks that desugar to a ribbon come back with it.

    ``mark_errorband`` and ``mark_smooth(ci=)`` both lay a ribbon layer under a
    line; under a flip that layer drew nothing, so the chart rendered as a bare
    line with its uncertainty silently missing. The assertion is on the band's
    geometry (a filled ring whose vertices pair on y), not on the path count,
    for the same reason as above.
    """
    rng = np.random.default_rng(7)
    df = pl.DataFrame(
        {
            "x": [float(i) for i in range(12)] * 2,
            "y": list(rng.normal(size=24)),
        }
    )
    chart = fm.Chart(df)
    mark = chart.mark_errorband() if build == "errorband" else chart.mark_smooth(ci=0.95)
    svg = mark.encode(x="x:Q", y="y:Q").coord(fm.CoordFlip()).to_svg()

    pairs = band_pairs(filled_path_vertices(svg))
    assert pairs, f"{build} must emit a confidence band under CoordFlip"
    shared_y = [p for p in pairs if abs(p[0][1] - p[1][1]) < EPS]
    assert len(shared_y) == len(pairs), (
        f"{build}'s flipped band must pair its vertices on y (x..x2 at a shared y); "
        f"{len(pairs) - len(shared_y)} of {len(pairs)} pairs did not"
    )
    assert any(abs(p[0][0] - p[1][0]) > EPS for p in pairs), (
        f"{build}'s band must have non-zero width somewhere, or it is a line in disguise"
    )
