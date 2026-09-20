"""`#182` + `#94`: a gutter exists if and only if something is drawn in it.

Layout reserved an axis title gutter and a tick-label band off the axis
*inputs* -- `AxisInput.title.is_some()`, `AxisInput.tick_labels` -- while the
decision to DRAW that chrome was made later, and separately. So the two could
disagree, and did:

* `.axis(show=False)` (JointChart marginals, ClusterMap dendrograms) kept a
  full title gutter on both sides plus a label band on the y side, for an axis
  line, ticks, labels and title that were never emitted.
* A `CoordPolar` chart, which has no Cartesian axes at all to hide, reserved a
  Cartesian left gutter and bottom band anyway -- `#94`, the same defect
  reached from the other direction, which is why the fix gates on "is this
  axis drawn" rather than on the `.axis(show=)` flags alone.

The observable in both cases is the same: the plot area is smaller than the
drawn chrome justifies, and the chart sits off-centre in its own frame. So
every assertion here is a MEASURED plot rect (`_svg_rects.plot_rect` -- the one
`<rect>` a chart emits with no `fill`), compared against the padded viewport or
against a sibling render that differs in exactly one knob. A chart whose
padding is 16 on every side and which draws no axis chrome has exactly one
correct plot rect, and it is stated here as a literal.

The one thing NOT collapsed is the axis title on a `fm.Axis(labels=False)`
axis: that axis still draws its title, so it keeps its title gutter and loses
only the label band. `title_and_label_bands_are_separately_gated` pins that as
an additive identity over four renders rather than as two inequalities, because
"smaller than the default" would also hold for a fix that collapsed both.
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm
from _svg_rects import plot_rect

#: Viewport and padding chosen so the fully-collapsed answer is exact and
#: readable: 640x480 minus 16px on each side is (16, 16, 608, 448).
WIDTH, HEIGHT = 640, 480
PAD = 16
PADDED_RECT = (float(PAD), float(PAD), float(WIDTH - 2 * PAD), float(HEIGHT - 2 * PAD))

THEME = fm.Theme(padding=PAD)

FRAME = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4000.0, 5000.0, 6000.0]})

#: A polar pie: `theta=` + `color=`, no y encoding, no Cartesian chrome. The
#: color legend keeps a RIGHT-side gutter, which is why the polar assertions
#: read the left/top/bottom insets and never the width.
POLAR_FRAME = pl.DataFrame({"cat": ["a", "b", "c"], "v": [1.0, 2.0, 3.0]})

#: A unit-square GeoJSON polygon (mirrors `test_break_axis_geometry.py`'s
#: `_GEO_UNIT_SQUARE`) -- enough for `mark_geoshape` to project real geometry.
GEO_UNIT_SQUARE = {
    "type": "Polygon",
    "coordinates": [[[-10, -10], [10, -10], [10, 10], [-10, 10], [-10, -10]]],
}


def base_chart() -> fm.Chart:
    """A titled, labeled, two-axis scatter -- every gutter the fix gates."""
    return (
        fm.Chart(FRAME, width=WIDTH, height=HEIGHT).mark_point().encode(x="a", y="b").theme(THEME)
    )


#: Tolerance for an inset. `left`/`top` are read straight off the rect, but
#: `right`/`bottom` are derived by subtraction and so carry float noise in the
#: last bits (a 16px padding measures 16.000000000000114). Well below the
#: smallest real band this module distinguishes, which is a ~14px label band.
EPS = 1e-6


def insets(chart) -> tuple[float, float, float, float]:
    """`(left, top, right, bottom)` pixels between the viewport and the plot rect."""
    x, y, w, h = plot_rect(chart.to_svg())
    return (x, y, WIDTH - x - w, HEIGHT - y - h)


def test_default_chart_reserves_a_real_band_on_both_gated_sides():
    """The control every collapse below is measured against.

    Without this, a fix that accidentally collapsed EVERY chart's gutters would
    satisfy the rest of this module: `(16, 16, 608, 448)` is only the right
    answer for a chart that draws nothing there.
    """
    left, top, right, bottom = insets(base_chart())
    assert left > PAD, f"a drawn y axis reserves a left band beyond the padding, got {left}"
    assert bottom > PAD, f"a drawn x axis reserves a bottom band beyond the padding, got {bottom}"
    assert (top, right) == pytest.approx((PAD, PAD), abs=EPS), (
        "an unlegended, untitled chart reserves nothing on the top/right"
    )


def test_axis_show_false_collapses_to_the_padded_viewport():
    """`#182`'s headline: `.axis(show=False)` leaves exactly the padded rect.

    Pre-fix this measured a 23.729px phantom title gutter on each of the two
    axis sides plus a 17.381px phantom y label band -- the plot rect was
    `(57.11, 16.0, 566.89, 424.271)` on this fixture, i.e. shifted right and
    short at the bottom, for chrome that is not in the SVG.
    """
    chart = base_chart().axis(show=False)
    assert plot_rect(chart.to_svg()) == PADDED_RECT


def test_axis_show_false_draws_no_title_text_either():
    """The other half of "a gutter exists iff something is drawn in it".

    Shrinking the gutter would be a REGRESSION if the title still rendered --
    it would then be drawn over the marks. This pins that the hidden axis's
    title is genuinely absent from the SVG, so the collapsed rect above is
    consistent with the render rather than merely smaller than it.
    """
    assert "a</text>" in base_chart().to_svg(), "the drawn x axis titles itself 'a'"
    hidden = base_chart().axis(show=False).to_svg()
    for title in ("a", "b"):
        assert f">{title}</text>" not in hidden, f"hidden axis must not draw its {title!r} title"


@pytest.mark.parametrize(
    ("kwargs", "collapsed_sides", "kept_sides"),
    [
        ({"y": False}, ("left",), ("bottom",)),
        ({"x": False}, ("bottom",), ("left",)),
    ],
)
def test_hiding_one_axis_collapses_only_that_axis_side(kwargs, collapsed_sides, kept_sides):
    """Per-side, not per-chart: hiding y frees the LEFT band and leaves the
    bottom one exactly as the default chart had it (and vice versa).

    A fix that zeroed both bands whenever either axis was hidden would pass
    `test_axis_show_false_collapses_to_the_padded_viewport` and fail here.
    """
    by_side = dict(zip(("left", "top", "right", "bottom"), insets(base_chart().axis(**kwargs))))
    default_by_side = dict(zip(("left", "top", "right", "bottom"), insets(base_chart())))
    for side in collapsed_sides:
        assert by_side[side] == pytest.approx(PAD, abs=EPS), (
            f"{side} band must collapse to the padding"
        )
    for side in kept_sides:
        assert by_side[side] == pytest.approx(default_by_side[side], abs=EPS), (
            f"{side} band must be untouched by axis({kwargs})"
        )


def test_title_and_label_bands_are_separately_gated():
    """`fm.Axis(labels=False)` collapses the label band ALONE.

    Stated as an additive identity over four renders, on the y side (whose band
    carries both pieces): with `L` the label band and `T` the title gutter,

        default        left inset = PAD + L + T
        labels=False   left inset = PAD + T
        title=None     left inset = PAD + L
        neither        left inset = PAD

    so `default + PAD == labels_off + title_off` must hold, and each of the two
    middle renders must sit strictly between. Two inequalities alone ("smaller
    than default") would also pass for a fix that collapsed both pieces at the
    first hint of either being hidden; the identity pins that each knob removes
    exactly its own piece and nothing else.
    """

    def left_inset(y_channel) -> float:
        chart = (
            fm.Chart(FRAME, width=WIDTH, height=HEIGHT)
            .mark_point()
            .encode(x="a", y=y_channel)
            .theme(THEME)
        )
        return insets(chart)[0]

    default = left_inset("b")
    labels_off = left_inset(fm.Y("b", axis=fm.Axis(labels=False)))
    title_off = left_inset(fm.Y("b", title=None))
    neither = left_inset(fm.Y("b", title=None, axis=fm.Axis(labels=False)))

    assert neither == float(PAD), "no title and no labels means no y band at all"
    assert PAD < labels_off < default, f"the title gutter alone: {labels_off}"
    assert PAD < title_off < default, f"the label band alone: {title_off}"
    assert default + PAD == pytest.approx(labels_off + title_off, abs=1e-6), (
        f"default({default}) + PAD must equal labels_off({labels_off}) + title_off({title_off}): "
        "each knob must remove exactly its own piece of the band"
    )


def test_labels_false_still_draws_the_axis_title():
    """The companion fact to the identity above: the title gutter that
    `labels=False` keeps is reserved for a title that really is drawn."""
    svg = (
        fm.Chart(FRAME, width=WIDTH, height=HEIGHT)
        .mark_point()
        .encode(x="a", y=fm.Y("b", axis=fm.Axis(labels=False)))
        .theme(THEME)
        .to_svg()
    )
    assert ">b</text>" in svg, "labels=False must not suppress the axis title too"
    assert ">4000</text>" not in svg, "labels=False must suppress the tick labels"


def polar_pie() -> fm.Chart:
    return (
        fm.Chart(POLAR_FRAME, width=WIDTH, height=HEIGHT)
        .mark_arc()
        .encode(theta="v", color="cat")
        .coord(fm.CoordPolar())
        .theme(THEME)
    )


def test_polar_reserves_no_cartesian_margins():
    """`#94`: a chart with no Cartesian axes reserves no Cartesian gutters.

    This is why the fix cannot gate on `.axis(show=)` alone -- nothing here
    hides an axis; there is no Cartesian axis to hide. Pre-fix this reserved
    23.214px on the left and 43.039px on the bottom, both for chrome
    `render::scene_build` suppresses outright under a projected coord.

    The right inset is deliberately not asserted: the `color=` legend lives
    there and is real drawn chrome.
    """
    left, top, right, bottom = insets(polar_pie())
    assert (left, top, bottom) == pytest.approx((PAD, PAD, PAD), abs=EPS)
    assert right > PAD, "the color legend's gutter is real chrome and must survive"


def test_polar_margins_still_follow_the_theme_padding():
    """Discriminating control for the polar collapse: the left/bottom insets
    above are the PADDING, not a hardcoded 16 and not zero. Doubling the
    padding doubles them, which a fix that simply zeroed the polar insets
    would fail."""
    doubled = fm.Theme(padding=2 * PAD)
    left, top, _right, bottom = insets(polar_pie().theme(doubled))
    assert (left, top, bottom) == pytest.approx((2 * PAD, 2 * PAD, 2 * PAD), abs=EPS)


def test_polar_still_draws_its_own_radial_chrome():
    """The polar collapse must remove the CARTESIAN chrome only. The polar
    axis's own boundary + radial ticks (`build_polar_axes`) are unaffected, so
    the bigger plot rect is a bigger pie, not a bare one."""
    svg = polar_pie().to_svg()
    assert svg.count("<path") >= 3, "the three arc slices must still be drawn"
    assert ">a</text>" in svg, "the color legend's entries must still be drawn"


def test_geo_reserves_no_cartesian_margins():
    """`CoordGeo` shares `#94`'s shape and the same predicate closes it:
    `mark_geoshape` bbox-fits projected lon/lat into the plot rect and
    `scene_build` emits no Cartesian axis node, so no Cartesian band may be
    reserved either."""
    chart = (
        fm.Chart(GEO_UNIT_SQUARE, width=WIDTH, height=HEIGHT)
        .mark_geoshape()
        .coord(fm.CoordGeo(projection="equirectangular"))
        .theme(THEME)
    )
    assert plot_rect(chart.to_svg()) == PADDED_RECT


def test_a_hidden_axis_ignores_its_title_so_the_clustermap_workaround_is_inert():
    """One mechanism for one rule.

    `plots/matrix.py`'s dendrogram layers used to pass `title=""` purely to
    reach `compute_*_title_width`'s `title.is_some()` route, because their
    `axis(show=False)` panels reserved a phantom title gutter that ate into
    `dendrogram_ratio`. With the gutter gated on the axis being drawn, blanking
    the title is a no-op on a hidden axis -- byte-for-byte, not merely in the
    plot rect -- which is what makes removing that workaround safe rather than
    a behavior change.
    """
    titled = base_chart().axis(show=False).to_svg()
    blanked = (
        fm.Chart(FRAME, width=WIDTH, height=HEIGHT)
        .mark_point()
        .encode(x=fm.X("a", title=""), y=fm.Y("b", title=""))
        .theme(THEME)
        .axis(show=False)
        .to_svg()
    )
    assert titled == blanked
    assert plot_rect(titled) == PADDED_RECT
