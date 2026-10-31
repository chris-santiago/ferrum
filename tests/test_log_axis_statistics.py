"""A statistic drawn on a log positional axis computes in that axis's own
space, so a strictly positive histogram, kde, or ecdf keeps every row on
every door that resolves a `LogScale`: `displot(log_scale=True)`, a channel
`scale=fm.LogScale()`, `.override(<ch>_scale_type="log")`, a sibling layer's
scale (`+`, `LayerChart`, a composite leaf), and a composite's shared scale
all reach the lowered statistic by the one resolved-axis rule. A row the
axis cannot place is left out and warned once, naming the field the axis
shows (when the establishing layer draws a statistic: its title when one is
set, no field clause at all when the title is suppressed; a primitive
establishing layer names its raw field) and counting the caller's own
off-side rows -- never an internal bin-edge or kde-grid-point count, and
never doubled across every layer that reads the same rows. The statistic
reads the caller's column in whatever numeric type it has, each of its
inputs once in that input's own axis space, and every other channel that
reads the same column (a per-row point's y or color, a mean of the binned
field) reads it in data units. A `LogScale` on a count, density or
aggregate axis governs how that axis draws; it never moves which values the
bin or fit axis reads. A step the caller wrote directly in `.transform()`
is never reached by this: its space stays whatever the caller's own data is
in.
"""

from __future__ import annotations

import re
import warnings
from typing import Any, Callable

import numpy as np
import polars as pl
import pyarrow as pa
import pytest

import ferrum as fm
from tests._svg_extents import (
    log_axis_pixels_per_unit,
    log_value_for_pixel,
    value_axis_ladder,
    value_axis_ladders,
    value_for_pixel,
)
from tests._svg_rects import data_bar_rects, mark_groups, mark_layer, polylines, svg_elements

#: A frame a chart reads: the polars and pyarrow tables the dtype doors build.
_Frame = pl.DataFrame | pa.Table


def _warnings_of(render: Callable[[], Any]) -> tuple[list[str], Any]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = render()
    return [str(w.message) for w in caught], result


def _axis_units_per_pixel(ladder: dict[float, float]) -> float:
    """Data units per pixel for a (linear) axis tick ladder: exact for a
    linear axis. A log axis's own units-per-pixel is
    `1 / log_axis_pixels_per_unit(svg, axis=axis)` (in `log(base)` units)."""
    lo, hi = min(ladder), max(ladder)
    return abs((hi - lo) / (ladder[hi] - ladder[lo]))


def _bar_edge_widths(svg: str) -> list[float]:
    return [float(b["width"]) for b in data_bar_rects(svg)]


def _assert_panels_share_one_bin_width(svg: str, *, panel_totals: list[float]) -> None:
    """Each facet panel's bars sum to that panel's own row count (panels in
    drawing order, read off each panel's own count axis), and every bar in
    every panel renders at one pixel width: the panels bin over one shared
    extent at one bin width."""
    groups = mark_groups(svg)
    ladders = value_axis_ladders(svg, axis="y")
    assert len(groups) == len(panel_totals), len(groups)
    assert len(ladders) == len(panel_totals), len(ladders)

    all_widths: list[float] = []
    for group, (_pos, ladder), expected in zip(groups, ladders, panel_totals):
        units = _axis_units_per_pixel(ladder)
        bars = data_bar_rects(group)
        assert bars, "a panel draws no bars"
        all_widths.extend(float(b["width"]) for b in bars)
        total = sum(float(b["height"]) * units for b in bars)
        assert total == pytest.approx(expected, abs=0.1), (total, expected)

    assert max(all_widths) == pytest.approx(min(all_widths), rel=0.02), all_widths


def _total_bar_count(svg: str, *, count_kind: str) -> float:
    """Sum every rendered bar's own row count, converting its pixel extent
    along the count axis back to data units. `count_kind` is `"y_height"`
    for a vertical histogram (counts on the bar's `height`, read off the y
    axis) or `"x_width"` for one drawn sideways, e.g. by `CoordFlip` (counts
    on `width`, read off the x axis)."""
    bars = data_bar_rects(svg)
    axis = "y" if count_kind == "y_height" else "x"
    dim = "height" if count_kind == "y_height" else "width"
    units = _axis_units_per_pixel(value_axis_ladder(svg, axis=axis))
    return sum(float(b[dim]) * units for b in bars)


def _polyline_peak_by_stroke(svg: str) -> dict[str, float]:
    """Map each `<polyline>`'s stroke color to its peak (min-y, since SVG y
    grows downward) point."""
    return {
        stroke: min(y for _x, y in points)
        for stroke, points in polylines(svg)
        if stroke is not None
    }


def _leftmost_ladder(svg: str, *, axis: str) -> dict[float, float]:
    """A bivariate kde panel's own axis ladder: the leftmost of the
    candidates `value_axis_ladders` returns. A density colorbar draws a
    second, further-right numeric ladder on the same side that is not the
    plot's own positional axis."""
    return min(value_axis_ladders(svg, axis=axis), key=lambda entry: entry[0])[1]


def _path_vertices(svg: str) -> list[tuple[float, float]]:
    """Every `<path d="...">`'s `(x, y)` pixel vertices, in document order."""
    coords: list[tuple[float, float]] = []
    for d in re.findall(r'<path[^>]*\sd="([^"]+)"', svg):
        for m in re.finditer(r"([-\d.]+)\s+([-\d.]+)", d):
            coords.append((float(m.group(1)), float(m.group(2))))
    return coords


def _invert_log(pixel: float, ladder: dict[float, float], *, base: float = 10.0) -> float:
    """A drawn pixel's data value on a log ladder: the shared
    `log_axis_pixels_per_unit` fit, inverted."""
    return log_value_for_pixel(ladder, pixel, base=base)


def _contour_vertices_in_data_units(
    svg: str, *, x_log: bool, y_log: bool
) -> tuple[list[float], list[float]]:
    """Every vertex of every `<path>` mark (a contour mesh, or a univariate
    density's area), converted back to data units through each axis's own
    tick ladder: the log fit when that axis is a `LogScale` of any base, the
    linear fit otherwise."""
    coords = _path_vertices(mark_layer(svg))
    x_ladder = _leftmost_ladder(svg, axis="x")
    y_ladder = _leftmost_ladder(svg, axis="y")
    x_values = [
        _invert_log(px, x_ladder) if x_log else value_for_pixel(x_ladder, px) for px, _py in coords
    ]
    y_values = [
        _invert_log(py, y_ladder) if y_log else value_for_pixel(y_ladder, py) for _px, py in coords
    ]
    return x_values, y_values


def _from_log10(value: float) -> float:
    return 10.0**value


def _from_log2(value: float) -> float:
    return 2.0**value


def _unchanged(value: float) -> float:
    return value


def _assert_same_vertices_in_data_units(
    log_svg: str,
    oracle_svg: str,
    *,
    x_log: bool,
    y_log: bool,
    x_from_oracle: Callable[[float], float],
    y_from_oracle: Callable[[float], float],
) -> None:
    """The log-axis chart's path vertices, read back to data units, equal
    pair for pair the oracle's: the same statistic drawn on linear axes over
    columns the test transformed itself (`log10(v)`, `log2(v)` or `v`), its
    vertices mapped back by `x_from_oracle` / `y_from_oracle`."""
    log_x, log_y = _contour_vertices_in_data_units(log_svg, x_log=x_log, y_log=y_log)
    oracle_x, oracle_y = _contour_vertices_in_data_units(oracle_svg, x_log=False, y_log=False)
    assert log_x, "the chart draws no path"
    assert len(log_x) == len(oracle_x), (len(log_x), len(oracle_x))
    assert log_x == pytest.approx([x_from_oracle(x) for x in oracle_x], rel=1e-3)
    assert log_y == pytest.approx([y_from_oracle(y) for y in oracle_y], rel=1e-3)


# ---------------------------------------------------------------------------
# Fixtures shared across this module.
# ---------------------------------------------------------------------------

_V20 = np.arange(1, 21, dtype=float)
#: Strictly positive, 20 rows -- the one dataset every "keeps every bar"
#: door below renders, so a dropped bar is unambiguously the defect.
_DF20 = pl.DataFrame({"v": _V20})
#: The mirror image, for the negative-side (explicit negative domain) door.
_DF20_NEG = pl.DataFrame({"v": -_V20})
#: `_DF20` plus one row that can never sit on ANY no-domain LogScale side,
#: for the negative control: the "every bar kept" result above is a genuine
#: positive-data outcome, not a warning check that can never fire.
_DF20_PLUS_ZERO = pl.DataFrame({"v": np.concatenate([[0.0], _V20])})
#: `_DF20` plus two off-side rows (0.0 and -2.0), for the off-side warning's
#: naming and count.
_DF_MIXED = pl.DataFrame({"v": np.concatenate([_V20, [0.0, -2.0]])})
#: Every row off-side, no domain to mirror against -- the refusal path.
_DF_ALL_NEG5 = pl.DataFrame({"v": [-1.0, -2.0, -3.0, -4.0, -5.0]})

#: The warning a primitive layer on a no-domain `LogScale` gives for
#: `_DF_MIXED`'s two off-side rows, naming the raw field.
_PRIMITIVE_OFF_SIDE_SENTENCE = (
    "x scale: 2 row(s) of 'v' are zero or negative, which a LogScale cannot place, "
    "so they are not drawn; filter them, or use fm.SymlogScale() for data that "
    "reaches zero"
)


# ---------------------------------------------------------------------------
# A strictly positive histogram on a log axis keeps every bar, on every door
# that resolves a LogScale.
# ---------------------------------------------------------------------------


def _hist_plus_point(df: _Frame) -> fm.Chart:
    return fm.Chart(df).mark_histogram().encode(x="v") + fm.Chart(df).mark_point().encode(
        x=fm.X("v", scale=fm.LogScale()), y="v"
    )


def _point_plus_hist(df: _Frame) -> fm.Chart:
    return fm.Chart(df).mark_point().encode(x=fm.X("v", scale=fm.LogScale()), y="v") + fm.Chart(
        df
    ).mark_histogram().encode(x="v")


def _layer_chart_hist_then_point(df: _Frame) -> fm.Chart:
    return fm.LayerChart(
        fm.Chart(df).mark_histogram().encode(x="v"),
        fm.Chart(df).mark_point().encode(x=fm.X("v", scale=fm.LogScale()), y="v"),
    )


def _layer_chart_point_then_hist(df: _Frame) -> fm.Chart:
    return fm.LayerChart(
        fm.Chart(df).mark_point().encode(x=fm.X("v", scale=fm.LogScale()), y="v"),
        fm.Chart(df).mark_histogram().encode(x="v"),
    )


def _door_displot_log_scale() -> fm.Chart:
    return fm.displot(_DF20, x="v", kind="hist", log_scale=True)


def _door_mark_histogram_channel_log_scale() -> fm.Chart:
    return fm.Chart(_DF20).mark_histogram().encode(x=fm.X("v", scale=fm.LogScale()))


def _door_mark_bar_bin_true_log_scale() -> fm.Chart:
    return (
        fm.Chart(_DF20)
        .mark_bar()
        .encode(x=fm.X("v", bin=True, scale=fm.LogScale()), y=fm.Y(aggregate="count"))
    )


def _door_override_log() -> fm.Chart:
    return fm.Chart(_DF20).mark_histogram().encode(x="v").override(x_scale_type="log")


def _door_log_base_2() -> fm.Chart:
    return fm.Chart(_DF20).mark_histogram().encode(x=fm.X("v", scale=fm.LogScale(base=2)))


def _door_negative_side() -> fm.Chart:
    """The explicit domain is stated past the data's own bounds (`v` sits in
    `[-20, -1]`), not flush with them. `Bin`'s own `nice=True` extent
    widening disagrees with a domain that matches the data exactly: this
    interaction also reproduces through a plain `LinearScale(domain=[1.0,
    20.0])` on positive `1..20`, so widening the domain past the bounds
    here isolates this door's assertions to the log-space bin edges."""
    return (
        fm.Chart(_DF20_NEG)
        .mark_histogram()
        .encode(x=fm.X("v", scale=fm.LogScale(domain=[-30.0, -0.5])))
    )


def _door_log_y_via_coord_flip() -> fm.Chart:
    """`CoordFlip` mirrors the binned axis onto y: the bin axis the demand
    reaches is still `x` (what the caller bound), but it is drawn
    vertically, so the row count reads off the bars' pixel `width`."""
    return (
        fm.Chart(_DF20)
        .mark_histogram()
        .encode(x=fm.X("v", scale=fm.LogScale()))
        .coord(fm.CoordFlip())
    )


#: door name -> (builder, count_kind). Every builder renders 20 rows.
_EVERY_BAR_KEPT_DOORS: dict[str, tuple[Callable[[], fm.Chart], str]] = {
    "displot(log_scale=True)": (_door_displot_log_scale, "y_height"),
    "mark_histogram channel LogScale": (_door_mark_histogram_channel_log_scale, "y_height"),
    "mark_bar bin=True on LogScale": (_door_mark_bar_bin_true_log_scale, "y_height"),
    "override(x_scale_type='log')": (_door_override_log, "y_height"),
    "hist + point(LogScale)": (lambda: _hist_plus_point(_DF20), "y_height"),
    "point(LogScale) + hist": (lambda: _point_plus_hist(_DF20), "y_height"),
    "LayerChart(hist, point(LogScale))": (lambda: _layer_chart_hist_then_point(_DF20), "y_height"),
    "LayerChart(point(LogScale), hist)": (lambda: _layer_chart_point_then_hist(_DF20), "y_height"),
    "log base 2": (_door_log_base_2, "y_height"),
    "negative side (explicit negative LogScale domain)": (_door_negative_side, "y_height"),
    "log y (CoordFlip draws the bin axis vertically)": (_door_log_y_via_coord_flip, "x_width"),
}


@pytest.mark.parametrize("door", sorted(_EVERY_BAR_KEPT_DOORS))
def test_a_positive_histogram_on_a_log_axis_keeps_every_bar(door):
    """Every door that resolves a `LogScale` over a histogram's bin axis
    keeps every one of 20 strictly positive rows, with no off-side
    warning."""
    build, count_kind = _EVERY_BAR_KEPT_DOORS[door]
    msgs, svg = _warnings_of(lambda: build().to_svg())
    assert msgs == [], (door, msgs)
    assert _total_bar_count(svg, count_kind=count_kind) == pytest.approx(20.0, abs=0.1), door


def test_a_positive_histogram_on_a_log_axis_warns_only_for_a_genuinely_off_side_row():
    """Negative control for the invariant above: a single `0.0` row can
    never sit on a no-domain LogScale, so it is left out and warned about by
    name -- the "every bar kept" result is a genuine positive-data outcome,
    not a warning check that can never fire."""
    msgs, _ = _warnings_of(
        lambda: fm.displot(_DF20_PLUS_ZERO, x="v", kind="hist", log_scale=True).to_svg()
    )
    assert len(msgs) == 1, msgs
    assert "1 row(s) of 'v'" in msgs[0], msgs


# ---------------------------------------------------------------------------
# A LogScale on the count, density or mean-aggregate axis governs how that
# axis draws; it never moves which x values a bin or fit reads.
# ---------------------------------------------------------------------------


def _door_hist_log_count_axis() -> fm.Chart:
    return fm.Chart(_DF_MIXED).mark_histogram().encode(x="v", y=fm.Y(scale=fm.LogScale()))


def _door_bar_bin_true_log_count() -> fm.Chart:
    return (
        fm.Chart(_DF_MIXED)
        .mark_bar()
        .encode(x=fm.X("v", bin=True), y=fm.Y(aggregate="count", scale=fm.LogScale()))
    )


def _door_hist_override_log_count_axis() -> fm.Chart:
    return fm.Chart(_DF_MIXED).mark_histogram().encode(x="v").override(y_scale_type="log")


def _door_density_log_y() -> fm.Chart:
    return fm.Chart(_DF_MIXED).mark_density().encode(x="v", y=fm.Y(scale=fm.LogScale()))


def _assert_every_row_binned_in_equal_data_width_bins(svg: str) -> None:
    """Every one of `_DF_MIXED`'s rows lands in a bar (each bar's count read
    off the log count axis at its top edge), and the bars are equal pixel
    width on the linear x axis: the bins are equal width in data units."""
    widths = _bar_edge_widths(svg)
    assert len(widths) >= 3, widths
    assert max(widths) == pytest.approx(min(widths), rel=0.02), widths
    count_ladder = value_axis_ladder(svg, axis="y")
    total = sum(_invert_log(float(b["y"]), count_ladder) for b in data_bar_rects(svg))
    assert total == pytest.approx(float(_DF_MIXED.height), abs=0.1), total


def _assert_the_fit_reads_every_row_in_data_units(svg: str) -> None:
    """The density curve reaches `_DF_MIXED`'s lowest row (`-2.0`) on the
    linear x axis: a fit that left the off-side rows out would start above
    zero."""
    x_ladder = value_axis_ladder(svg, axis="x")
    xs = [value_for_pixel(x_ladder, px) for px, _py in _path_vertices(mark_layer(svg))]
    assert xs, "the chart draws no curve"
    assert min(xs) <= -2.0 + 0.01, min(xs)


#: door name -> (builder, the check that the bin or fit axis read every row
#: in data units).
_LOG_COUNT_AXIS_DOORS: dict[str, tuple[Callable[[], fm.Chart], Callable[[str], None]]] = {
    "mark_histogram y=Y(scale=LogScale())": (
        _door_hist_log_count_axis,
        _assert_every_row_binned_in_equal_data_width_bins,
    ),
    "mark_bar x bin=True, y count on LogScale": (
        _door_bar_bin_true_log_count,
        _assert_every_row_binned_in_equal_data_width_bins,
    ),
    "mark_histogram override(y_scale_type='log')": (
        _door_hist_override_log_count_axis,
        _assert_every_row_binned_in_equal_data_width_bins,
    ),
    "mark_density y=Y(scale=LogScale())": (
        _door_density_log_y,
        _assert_the_fit_reads_every_row_in_data_units,
    ),
}


@pytest.mark.parametrize("door", sorted(_LOG_COUNT_AXIS_DOORS))
def test_a_log_count_axis_does_not_move_the_binned_axis(door):
    """A `LogScale` on the count or density axis governs how counts and
    densities are drawn; it never moves which x values a histogram bins or
    a density fits. `_DF_MIXED` carries two rows (`0.0`, `-2.0`) that a log
    x axis would leave out: no warning, every row counted in an
    equal-data-width bin, and a density curve reaching `-2.0` prove the bin
    or fit axis stayed in data space."""
    build, reads_every_row = _LOG_COUNT_AXIS_DOORS[door]
    msgs, svg = _warnings_of(lambda: build().to_svg())
    assert msgs == [], (door, msgs)
    reads_every_row(svg)


def test_a_log_mean_aggregate_axis_does_not_move_the_binned_axis():
    """mark_bar(x bin=True) with a `mean` aggregate on a log y: the mean
    itself can be non-positive for a bin whose rows average at or below
    zero, which is the y channel's own off-side value, never the x bin
    axis's -- the x bins stay equal width in data units regardless."""
    msgs, svg = _warnings_of(
        lambda: (
            fm.Chart(_DF_MIXED)
            .mark_bar()
            .encode(x=fm.X("v", bin=True), y=fm.Y("v", aggregate="mean", scale=fm.LogScale()))
            .to_svg()
        )
    )
    widths = _bar_edge_widths(svg)
    assert len(widths) >= 3, widths
    assert max(widths) == pytest.approx(min(widths), rel=0.02), widths
    assert not any(m.startswith("x scale:") for m in msgs), msgs


def test_a_both_axes_log_histogram_keeps_every_bar_in_log_bins():
    """A histogram with both axes log bins only its own x field in log
    space; the count axis's own `LogScale` plays no part in choosing the
    bin edges, so every bar still renders at equal pixel width."""
    svg = (
        fm.Chart(_DF20)
        .mark_histogram()
        .encode(x=fm.X("v", scale=fm.LogScale()), y=fm.Y(scale=fm.LogScale()))
        .to_svg()
    )
    widths = _bar_edge_widths(svg)
    assert len(widths) >= 3, widths
    assert max(widths) == pytest.approx(min(widths), rel=0.02), widths


def test_log_axis_y_binned_count_x_keeps_every_bar():
    """A y-binned `LogScale` axis with x the count axis exercises the y arm
    of the log demand directly (the CoordFlip door above still binds its
    `LogScale` on x): bin edges render at equal pixel height in log space,
    with no off-side warning on this strictly positive data."""
    msgs, svg = _warnings_of(
        lambda: (
            fm.Chart(_DF20)
            .mark_bar()
            .encode(y=fm.Y("v", bin=True, scale=fm.LogScale()), x=fm.X(aggregate="count"))
            .to_svg()
        )
    )
    assert msgs == [], msgs
    heights = [float(b["height"]) for b in data_bar_rects(svg)]
    assert len(heights) >= 3, heights
    assert max(heights) == pytest.approx(min(heights), rel=0.02), heights


# ---------------------------------------------------------------------------
# Every other channel that reads a log-binned field reads the caller's own
# values: a per-row point's y and color, and a bar's mean of that field.
# ---------------------------------------------------------------------------

_V89 = [float(value) for value in range(1, 90)]


def _rightmost_y_ladder(svg: str) -> dict[float, float]:
    """The furthest-right numeric y ladder: a colorbar's, when the chart
    draws one beside its plot's own y axis."""
    ladders = value_axis_ladders(svg, axis="y")
    assert len(ladders) == 2, [sorted(ladder) for _pos, ladder in ladders]
    return ladders[-1][1]


def test_a_per_row_log_bin_draws_each_rows_own_value_on_y():
    """A `mark_point` binned on a log x keeps every row, and each point's
    y, read back through the y axis's own tick ladder, is its row's own
    `v` in data units."""
    df = pl.DataFrame({"v": _V89})
    msgs, svg = _warnings_of(
        lambda: (
            fm.Chart(df)
            .mark_point()
            .encode(x=fm.X("v", bin=True, scale=fm.LogScale()), y="v")
            .to_svg()
        )
    )
    assert msgs == [], msgs
    y_ladder = value_axis_ladder(svg, axis="y")
    circles = svg_elements(mark_layer(svg), "circle", lambda a: "cy" in a)
    drawn = sorted(value_for_pixel(y_ladder, float(c["cy"])) for c in circles)
    assert drawn == pytest.approx(_V89, abs=0.01)


def test_a_per_row_log_bin_colors_by_the_callers_own_values():
    """`color="v"` on a `mark_point` binned on a log x draws the same
    colorbar ticks as the same chart binned on a linear x: the color scale
    reads `v` in data units either way, with no row left out."""
    df = pl.DataFrame({"v": _V89})

    def render(x: fm.X) -> tuple[list[str], str]:
        return _warnings_of(
            lambda: fm.Chart(df).mark_point().encode(x=x, y="v", color="v").to_svg()
        )

    log_msgs, log_svg = render(fm.X("v", bin=True, scale=fm.LogScale()))
    linear_msgs, linear_svg = render(fm.X("v", bin=True))
    assert log_msgs == [] and linear_msgs == [], (log_msgs, linear_msgs)
    assert sorted(_rightmost_y_ladder(log_svg)) == sorted(_rightmost_y_ladder(linear_svg))


def test_a_log_bin_aggregating_its_own_field_draws_the_data_unit_mean():
    """`mark_bar` binned on a log x with `y=fm.Y("v", aggregate="mean")`
    draws each bar at the data-unit mean of the rows inside its edges (the
    edges read back through the log x ladder), and every row lands in a
    bar. The data sits clear of every bin edge, so each row's bin is
    unambiguous."""
    values = [value + 0.5 for value in range(1, 21)]
    msgs, svg = _warnings_of(
        lambda: (
            fm.Chart(pl.DataFrame({"v": values}))
            .mark_bar()
            .encode(x=fm.X("v", bin=True, scale=fm.LogScale()), y=fm.Y("v", aggregate="mean"))
            .to_svg()
        )
    )
    assert msgs == [], msgs
    x_ladder = value_axis_ladder(svg, axis="x")
    y_ladder = value_axis_ladder(svg, axis="y")
    bars = data_bar_rects(mark_layer(svg))
    assert len(bars) >= 3, bars
    counted = 0
    for bar in bars:
        left = float(bar["x"])
        lo = _invert_log(left, x_ladder)
        hi = _invert_log(left + float(bar["width"]), x_ladder)
        inside = [value for value in values if lo <= value < hi]
        assert inside, (lo, hi)
        counted += len(inside)
        drawn = value_for_pixel(y_ladder, float(bar["y"]))
        assert drawn == pytest.approx(sum(inside) / len(inside), rel=0.01), (lo, hi, inside)
    assert counted == len(values)


# ---------------------------------------------------------------------------
# Bin edges sit in log space: equal width in log10 units, equal pixel width
# on the log axis, including under multiple="dodge" and across facet panels.
# ---------------------------------------------------------------------------


def test_log_axis_bins_are_equal_width_in_log_space():
    """Bin edges equally spaced in log10 units render as equal-PIXEL-width
    bars on a log x axis: a bin edge equally spaced in DATA units instead
    would render narrower bars the further right the bin sits."""
    svg = fm.Chart(_DF20).mark_histogram().encode(x=fm.X("v", scale=fm.LogScale())).to_svg()
    widths = _bar_edge_widths(svg)
    assert len(widths) >= 3, widths
    assert max(widths) == pytest.approx(min(widths), rel=0.02), widths


def test_log_axis_dodge_slots_are_equal_width():
    """`multiple="dodge"` splits each (equal-log-width) bin into equal
    pixel-width slots, uniform across every bin and every hue group."""
    grouped = _DF20.with_columns(g=pl.Series(["a", "b"] * 10))
    svg = fm.displot(grouped, x="v", hue="g", multiple="dodge", log_scale=True).to_svg()
    widths = _bar_edge_widths(svg)
    assert len(widths) >= 6, widths
    assert max(widths) == pytest.approx(min(widths), rel=0.02), widths


def test_log_axis_facets_share_one_log_space_extent_across_panels():
    """A column-faceted log histogram pins ONE log-space extent across
    every panel (the facet pin a per-panel `Bin` run shares), so a strictly
    positive row count never drops in any panel and every panel's bars
    render at the same pixel width as every other panel's."""
    df = pl.DataFrame({"v": [float(i) for i in range(1, 41)], "g": ["a"] * 20 + ["b"] * 20})
    msgs, svg = _warnings_of(lambda: fm.displot(df, x="v", col="g", log_scale=True).to_svg())
    assert msgs == [], msgs
    _assert_panels_share_one_bin_width(svg, panel_totals=[20.0, 20.0])


def test_log_axis_float_bin_width_is_interpreted_in_log_units():
    """A float `bins=` width means log10 units on a log axis, not data
    units: `log10(20) - log10(1)` is about 1.3, so a width of 0.5 gives a
    handful of bins, not the dozens a data-unit width of 0.5 produces on
    `v` ranging 1..20."""
    svg = fm.displot(_DF20, x="v", bins=0.5, log_scale=True).to_svg()
    bars = data_bar_rects(svg)
    assert 1 <= len(bars) <= 6, len(bars)
    units = _axis_units_per_pixel(value_axis_ladder(svg, axis="y"))
    total = sum(float(b["height"]) * units for b in bars)
    assert total == pytest.approx(20.0, abs=0.5)


def test_log_axis_histogram_density_integrates_to_one_in_log_space():
    """`stat="density"` on a log axis is density PER LOG UNIT: each bar's
    area (its density times its own log10 width) sums to 1 across the whole
    histogram, the same "area under the curve" contract a linear-axis
    density histogram already keeps in data units."""
    svg = fm.displot(_DF20, x="v", stat="density", log_scale=True).to_svg()
    bars = data_bar_rects(svg)
    y_units = _axis_units_per_pixel(value_axis_ladder(svg, axis="y"))
    rate = log_axis_pixels_per_unit(svg, axis="x")
    assert rate is not None
    log_units = 1.0 / abs(rate)
    total_area = sum(float(b["height"]) * y_units * float(b["width"]) * log_units for b in bars)
    assert total_area == pytest.approx(1.0, abs=0.15), total_area


def test_log_axis_refuses_an_explicit_extent_off_the_side():
    """An explicit `extent=` endpoint off a LogScale's side is malformed
    input for that scale, refused naming the extent and the scale -- not
    silently rendered against an extent the axis can never place."""
    with pytest.raises(ValueError) as excinfo:
        fm.Chart(_DF20).mark_histogram(extent=(-5.0, 20.0)).encode(
            x=fm.X("v", scale=fm.LogScale())
        ).to_svg()
    message = str(excinfo.value)
    assert "-5" in message, message
    assert "LogScale" in message, message


# ---------------------------------------------------------------------------
# A kde=True overlay (unfaceted and faceted) and a bivariate kde track
# log-space bars/fit.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("multiple", ["layer", "stack", "dodge"])
def test_a_kde_overlay_tracks_log_space_bars(multiple):
    """Each hue's `kde=True` overlay curve peak tracks that hue's own
    tallest bar once the bars (and the overlay's count-scale bin width) are
    computed in the same log space the x axis resolves to -- the
    `test_overlay_curve_tracks_its_own_bars_in_every_count_mode` ratio, on
    strictly positive, log-separated groups under `log_scale=True`."""
    rng = np.random.default_rng(0)
    n = 300
    df = pl.DataFrame(
        {
            "x": np.concatenate([np.exp(rng.normal(0.0, 0.5, n)), np.exp(rng.normal(1.5, 0.5, n))]),
            "g": ["a"] * n + ["b"] * n,
        }
    )
    svg = fm.displot(
        df,
        x="x",
        hue="g",
        kind="hist",
        kde=True,
        stat="count",
        multiple=multiple,
        fill=False,
        log_scale=True,
    ).to_svg()

    bars = data_bar_rects(svg)
    baseline = max(float(b["y"]) + float(b["height"]) for b in bars)
    tallest_by_fill: dict[str, float] = {}
    for bar in bars:
        fill = bar["fill"]
        tallest_by_fill[fill] = max(tallest_by_fill.get(fill, 0.0), float(bar["height"]))

    peaks = _polyline_peak_by_stroke(svg)
    assert len(peaks) == 2, f"expected 2 hue curves, got {len(peaks)}"
    for stroke, peak_y in peaks.items():
        ratio = (baseline - peak_y) / tallest_by_fill[stroke]
        assert ratio == pytest.approx(1.0, abs=0.15), (
            f"multiple={multiple!r}, stroke {stroke}: overlay peak is {ratio:.3f}x this "
            "group's tallest bar -- the log-space count-scaled bin width does not match "
            "the bars'"
        )


def test_a_faceted_kde_overlay_tracks_log_space_bars_in_every_panel():
    """The `kde=True` overlay's count-scale extent is pinned in the same
    log space the facet pin gives the bars, in every `multiple=` mode, so
    the overlay tracks each panel's own bars once faceted too."""
    rng = np.random.default_rng(1)
    n = 300
    df = pl.DataFrame(
        {
            "v": np.concatenate([np.exp(rng.normal(0.5, 0.3, n)), np.exp(rng.normal(2.0, 0.3, n))]),
            "g": ["a"] * n + ["b"] * n,
        }
    )
    for multiple in ["layer", "stack", "dodge"]:
        svg = fm.displot(
            df,
            x="v",
            col="g",
            kind="hist",
            kde=True,
            multiple=multiple,
            fill=False,
            log_scale=True,
        ).to_svg()
        panels = mark_groups(svg)
        assert len(panels) == 2, (multiple, len(panels))
        for panel in panels:
            bars = data_bar_rects(panel)
            assert bars, (multiple, "a panel draws no bars")
            peaks = list(_polyline_peak_by_stroke(panel).values())
            assert len(peaks) == 1, (multiple, peaks)
            tallest = max(float(b["height"]) for b in bars)
            panel_baseline = max(float(b["y"]) + float(b["height"]) for b in bars)
            ratio = (panel_baseline - peaks[0]) / tallest
            assert ratio == pytest.approx(1.0, abs=0.15), (multiple, ratio)


def _bivariate_log_df() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    return pl.DataFrame(
        {"v": np.exp(rng.normal(1.0, 1.0, 200)), "w": np.exp(rng.normal(0.5, 0.7, 200))}
    )


@pytest.mark.parametrize(
    "door",
    ["displot both axes bound", "mark_density x LogScale only", "mark_density x and y LogScale"],
)
def test_bivariate_kde_fits_and_grids_in_log_space(door):
    """A bivariate kde's fit, grid and isolines compute in log units on
    every axis a `LogScale` resolves: the chart's contour vertices, read
    back to data units through each axis's own tick ladder, land where
    fitting directly on `log10(v)`/`log10(w)` with linear axes (then
    mapping the result by `10**`) does."""
    df = _bivariate_log_df()
    v, w = df["v"].to_numpy(), df["w"].to_numpy()

    if door == "displot both axes bound":
        # displot's `log_scale=True` reaches only the x channel; y stays linear.
        log_svg = fm.displot(df, x="v", y="w", kind="kde", log_scale=True).to_svg()
        oracle_df = pl.DataFrame({"logv": np.log10(v), "w": w})
        oracle_svg = fm.displot(oracle_df, x="logv", y="w", kind="kde").to_svg()
        y_log = False
    elif door == "mark_density x LogScale only":
        log_svg = (
            fm.Chart(df).mark_density().encode(x=fm.X("v", scale=fm.LogScale()), y="w").to_svg()
        )
        oracle_df = pl.DataFrame({"logv": np.log10(v), "w": w})
        oracle_svg = fm.Chart(oracle_df).mark_density().encode(x="logv", y="w").to_svg()
        y_log = False
    else:
        log_svg = (
            fm.Chart(df)
            .mark_density()
            .encode(x=fm.X("v", scale=fm.LogScale()), y=fm.Y("w", scale=fm.LogScale()))
            .to_svg()
        )
        oracle_df = pl.DataFrame({"logv": np.log10(v), "logw": np.log10(w)})
        oracle_svg = fm.Chart(oracle_df).mark_density().encode(x="logv", y="logw").to_svg()
        y_log = True

    _assert_same_vertices_in_data_units(
        log_svg,
        oracle_svg,
        x_log=True,
        y_log=y_log,
        x_from_oracle=_from_log10,
        y_from_oracle=_from_log10 if y_log else _unchanged,
    )


#: door name -> (the chart over `v` on both axes, the oracle chart over the
#: test's own transformed columns, whether y is a log axis, the map from the
#: oracle's y values back to `v`). x is a base-10 log axis on every door.
_ONE_COLUMN_KDE_2D_DOORS: dict[
    str,
    tuple[
        Callable[[pl.DataFrame], fm.Chart],
        Callable[[np.ndarray], fm.Chart],
        bool,
        Callable[[float], float],
    ],
] = {
    "displot(x='v', y='v', log_scale=True)": (
        lambda df: fm.displot(df, x="v", y="v", kind="kde", log_scale=True),
        lambda v: fm.displot(
            pl.DataFrame({"logv": np.log10(v), "v2": v}), x="logv", y="v2", kind="kde"
        ),
        False,
        _unchanged,
    ),
    "mark_density x LogScale, y='v'": (
        lambda df: fm.Chart(df).mark_density().encode(x=fm.X("v", scale=fm.LogScale()), y="v"),
        lambda v: (
            fm.Chart(pl.DataFrame({"logv": np.log10(v), "v2": v}))
            .mark_density()
            .encode(x="logv", y="v2")
        ),
        False,
        _unchanged,
    ),
    "mark_density x and y LogScale": (
        lambda df: (
            fm.Chart(df)
            .mark_density()
            .encode(x=fm.X("v", scale=fm.LogScale()), y=fm.Y("v", scale=fm.LogScale()))
        ),
        lambda v: (
            fm.Chart(pl.DataFrame({"logv": np.log10(v), "logv2": np.log10(v)}))
            .mark_density()
            .encode(x="logv", y="logv2")
        ),
        True,
        _from_log10,
    ),
    "mark_density x LogScale, y LogScale(base=2)": (
        lambda df: (
            fm.Chart(df)
            .mark_density()
            .encode(x=fm.X("v", scale=fm.LogScale()), y=fm.Y("v", scale=fm.LogScale(base=2)))
        ),
        lambda v: (
            fm.Chart(pl.DataFrame({"logv": np.log10(v), "log2v": np.log2(v)}))
            .mark_density()
            .encode(x="logv", y="log2v")
        ),
        True,
        _from_log2,
    ),
}


@pytest.mark.parametrize("door", sorted(_ONE_COLUMN_KDE_2D_DOORS))
def test_a_kde_2d_reading_one_column_on_both_axes_fits_each_axis_in_its_own_space(door):
    """A bivariate kde whose x and y read the same column computes each axis
    from that column in that axis's own space, once: x in log10 units, y in
    data units on a linear y, in log10 units on a log y and in log2 units on
    a base-2 log y. On strictly positive data nothing warns, and the
    contours equal those fitted on linear axes over `log10(v)` and the
    y-axis column the test transformed itself, mapped back to `v`."""
    build, build_oracle, y_log, y_from_oracle = _ONE_COLUMN_KDE_2D_DOORS[door]
    df = _bivariate_log_df()
    msgs, log_svg = _warnings_of(lambda: build(df).to_svg())
    assert msgs == [], (door, msgs)
    oracle_svg = build_oracle(df["v"].to_numpy()).to_svg()
    _assert_same_vertices_in_data_units(
        log_svg,
        oracle_svg,
        x_log=True,
        y_log=y_log,
        x_from_oracle=_from_log10,
        y_from_oracle=y_from_oracle,
    )


def test_log_axis_univariate_mark_density_fits_in_log_space():
    """A univariate `mark_density` with a channel `LogScale` fits in log10
    units: on strictly positive data it warns nothing, and its curve, read
    back to data units, equals the density fitted on a linear axis over
    `log10(v)` with x mapped back by `10**` (a density per log unit, so y is
    the same on both)."""
    df = _bivariate_log_df().select("v")
    msgs, log_svg = _warnings_of(
        lambda: fm.Chart(df).mark_density().encode(x=fm.X("v", scale=fm.LogScale())).to_svg()
    )
    assert msgs == [], msgs
    oracle_df = pl.DataFrame({"logv": np.log10(df["v"].to_numpy())})
    oracle_svg = fm.Chart(oracle_df).mark_density().encode(x="logv").to_svg()
    _assert_same_vertices_in_data_units(
        log_svg,
        oracle_svg,
        x_log=True,
        y_log=False,
        x_from_oracle=_from_log10,
        y_from_oracle=_unchanged,
    )


def test_bivariate_kde_off_side_rows_warn_once_with_the_callers_count():
    """A bivariate kde's off-side rows warn once, naming the caller's own
    field and row count -- not a count of contour mesh vertices."""
    rng = np.random.default_rng(0)
    v = np.concatenate([np.exp(rng.normal(1.0, 1.0, 198)), [0.0, -2.0]])
    w = np.exp(rng.normal(0.5, 0.7, 200))
    df = pl.DataFrame({"v": v, "w": w})
    msgs, _ = _warnings_of(
        lambda: fm.Chart(df).mark_density().encode(x=fm.X("v", scale=fm.LogScale()), y="w").to_svg()
    )
    assert len(msgs) == 1, msgs
    assert "2 row(s) of 'v'" in msgs[0], msgs


def test_log_axis_univariate_mark_density_keeps_every_off_side_row_counted():
    """A univariate `mark_density` with a channel `LogScale` warns once
    about its off-side rows, naming the caller's field and row count, as
    `displot(kind="kde")` does."""
    msgs, _ = _warnings_of(
        lambda: fm.Chart(_DF_MIXED).mark_density().encode(x=fm.X("v", scale=fm.LogScale())).to_svg()
    )
    assert len(msgs) == 1, msgs
    assert "2 row(s) of 'v'" in msgs[0], msgs


def test_log_axis_reaches_an_hconcat_leaf():
    """A composite leaf's own `LogScale` (not only a flat chart's) reaches
    the histogram it binds, through that leaf's own run."""
    hist_leaf = fm.Chart(_DF20).mark_histogram().encode(x=fm.X("v", scale=fm.LogScale()))
    other_leaf = fm.Chart(_DF20).mark_point().encode(x="v", y="v")
    msgs, svg = _warnings_of(lambda: fm.hconcat(hist_leaf, other_leaf).to_svg())
    assert msgs == [], msgs
    assert len(data_bar_rects(svg)) == 6


# ---------------------------------------------------------------------------
# Off-side rows warn by the name the axis shows, counting the caller's own
# rows once -- never an internal bin-edge or kde-grid-point count, and never
# once per layer that reads the same rows.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["hist", "kde", "ecdf"])
def test_log_axis_off_side_rows_warn_by_the_name_the_axis_shows(kind):
    """`hist`, `kde` and `ecdf` all name the caller's own field and count
    the caller's own off-side rows (2, for `_DF_MIXED`'s `0.0` and `-2.0`),
    never an internal bin-edge or grid-point count."""
    msgs, _ = _warnings_of(lambda: fm.displot(_DF_MIXED, x="v", kind=kind, log_scale=True).to_svg())
    assert len(msgs) == 1, (kind, msgs)
    assert "2 row(s) of 'v'" in msgs[0], (kind, msgs[0])


def test_log_axis_off_side_rows_warn_by_the_callers_title():
    """The same off-side count names the axis by the caller's own `title=`
    when one is set, exactly as the domain warning already does."""
    msgs, _ = _warnings_of(
        lambda: fm.displot(
            _DF_MIXED, x=fm.X("v", title="Price"), kind="hist", log_scale=True
        ).to_svg()
    )
    assert len(msgs) == 1, msgs
    assert "2 row(s) of 'Price'" in msgs[0], msgs


def test_log_axis_off_side_rows_drop_the_field_clause_under_a_suppressed_title():
    """A suppressed title (`title=""`) drops the "of '<field>'" clause
    entirely, the same rule the domain warning already follows -- rather
    than naming an internal column the caller never wrote."""
    msgs, _ = _warnings_of(
        lambda: fm.displot(_DF_MIXED, x=fm.X("v", title=""), kind="hist", log_scale=True).to_svg()
    )
    assert len(msgs) == 1, msgs
    message = msgs[0]
    assert "2 row(s)" in message, message
    assert "are zero or negative" in message, message
    assert " of '" not in message, message


def test_a_primitive_layers_off_side_warning_stays_byte_identical():
    """A primitive `mark_point` on a `LogScale` already names the caller's
    own field directly (no statistic re-point stands in the way), so this
    exact sentence renders identically on every door."""
    msgs, _ = _warnings_of(
        lambda: (
            fm.Chart(_DF_MIXED)
            .mark_point()
            .encode(x=fm.X("v", scale=fm.LogScale()), y="v")
            .to_svg()
        )
    )
    assert msgs == [_PRIMITIVE_OFF_SIDE_SENTENCE]


def test_a_primitive_layers_sentence_ignores_a_titled_or_suppressed_title():
    """A primitive layer's sentence never reads its channel's `title=`: a
    titled or suppressed-title `mark_point` on a `LogScale` names the raw
    field exactly as an untitled one does."""
    for title in ("Price", ""):
        msgs, _ = _warnings_of(
            lambda title=title: (
                fm.Chart(_DF_MIXED)
                .mark_point()
                .encode(x=fm.X("v", title=title, scale=fm.LogScale()), y="v")
                .to_svg()
            )
        )
        assert msgs == [_PRIMITIVE_OFF_SIDE_SENTENCE], title


def _titled_point_then_hist(title: str, combine: str) -> fm.Chart:
    point = (
        fm.Chart(_DF_MIXED)
        .mark_point()
        .encode(x=fm.X("v", title=title, scale=fm.LogScale()), y="v")
    )
    hist = fm.Chart(_DF_MIXED).mark_histogram().encode(x="v")
    return point + hist if combine == "+" else fm.LayerChart(point, hist)


@pytest.mark.parametrize("title", ["Price", ""], ids=["titled", "suppressed title"])
@pytest.mark.parametrize("combine", ["+", "LayerChart"])
def test_a_primitive_establishing_layer_names_its_field_beside_a_histogram(combine, title):
    """A titled or suppressed-title `mark_point` that establishes a log x
    axis, layered first over a histogram of the same rows, warns with the
    primitive sentence naming the raw field: the axis shows the point's
    column, not a statistic's, so the histogram beside it never turns the
    sentence into a title-named one. `+` and `LayerChart` name the same."""
    msgs, _ = _warnings_of(lambda: _titled_point_then_hist(title, combine).to_svg())
    assert msgs == [_PRIMITIVE_OFF_SIDE_SENTENCE], (combine, title, msgs)


def test_log_axis_no_value_on_side_refusal_matches_the_sibling_primitive_route():
    """The sibling refusal (every row off-side, no domain to mirror
    against) names the caller's field and row count exactly as the
    primitive route does on the same data -- never the histogram's
    internal bin count."""

    def refuse(build: Callable[[], fm.Chart]) -> str:
        with pytest.raises(ValueError) as excinfo:
            build().to_svg()
        return str(excinfo.value)

    hist_message = refuse(
        lambda: fm.Chart(_DF_ALL_NEG5).mark_histogram().encode(x=fm.X("v", scale=fm.LogScale()))
    )
    point_message = refuse(
        lambda: fm.Chart(_DF_ALL_NEG5).mark_point().encode(x=fm.X("v", scale=fm.LogScale()), y="v")
    )
    assert hist_message == point_message
    assert "bin_start" not in hist_message, hist_message


_NO_VALUE_ON_SIDE_REMEDY = (
    "is negative, and a LogScale with no domain maps only positive values; pass an "
    "explicit negative domain, e.g. fm.LogScale(domain=[-5, -1]), to draw a mirrored "
    "log scale"
)


@pytest.mark.parametrize(
    "title,named",
    [("Price", "every value of 'Price' (5 rows)"), ("", "every value (5 rows)")],
    ids=["titled", "suppressed title"],
)
def test_a_histogram_refusal_names_the_axis_by_its_title(title, named):
    """A histogram whose every row is off-side refuses naming the axis the
    way its warning does: by the caller's title when one is set, with the
    field clause dropped under a suppressed title, and counting the
    caller's own rows rather than a bin count."""
    with pytest.raises(ValueError) as excinfo:
        fm.Chart(_DF_ALL_NEG5).mark_histogram().encode(
            x=fm.X("v", title=title, scale=fm.LogScale())
        ).to_svg()
    assert str(excinfo.value) == f"x scale: {named} {_NO_VALUE_ON_SIDE_REMEDY}"


def test_log_axis_kde_no_value_on_side_refuses_like_the_sibling_primitive_route():
    """A univariate kde over all-off-side data refuses with the exact
    sentence the primitive route gives on the same data, rather than
    rendering a degraded chart."""

    def refuse(build: Callable[[], fm.Chart]) -> str:
        with pytest.raises(ValueError) as excinfo:
            build().to_svg()
        return str(excinfo.value)

    kde_message = refuse(lambda: fm.displot(_DF_ALL_NEG5, x="v", kind="kde", log_scale=True))
    point_message = refuse(
        lambda: fm.Chart(_DF_ALL_NEG5).mark_point().encode(x=fm.X("v", scale=fm.LogScale()), y="v")
    )
    assert kde_message == point_message


@pytest.mark.parametrize(
    "door",
    [
        "displot hist kde=True overlay",
        "displot hist rug=True overlay",
        "hist + point(LogScale)",
        "point(LogScale) + hist",
        "LayerChart(hist, point(LogScale))",
        "LayerChart(point(LogScale), hist)",
    ],
)
def test_log_axis_off_side_rows_warn_once_however_many_layers_read_them(door):
    """Every layer reading the same off-side rows (a statistic and its
    overlay, or a statistic and a sibling primitive) warns ONCE with the
    caller's own row count, never once per reading layer."""
    builders: dict[str, Callable[[], fm.Chart]] = {
        "displot hist kde=True overlay": lambda: fm.displot(
            _DF_MIXED, x="v", kind="hist", kde=True, log_scale=True
        ),
        "displot hist rug=True overlay": lambda: fm.displot(
            _DF_MIXED, x="v", kind="hist", rug=True, log_scale=True
        ),
        "hist + point(LogScale)": lambda: _hist_plus_point(_DF_MIXED),
        "point(LogScale) + hist": lambda: _point_plus_hist(_DF_MIXED),
        "LayerChart(hist, point(LogScale))": lambda: _layer_chart_hist_then_point(_DF_MIXED),
        "LayerChart(point(LogScale), hist)": lambda: _layer_chart_point_then_hist(_DF_MIXED),
    }
    msgs, _ = _warnings_of(lambda: builders[door]().to_svg())
    assert len(msgs) == 1, (door, msgs)
    assert "2 row(s) of 'v'" in msgs[0], (door, msgs[0])


@pytest.mark.parametrize("kind", ["hist", "kde", "ecdf"])
def test_log_axis_faceted_off_side_rows_warn_once_with_the_whole_frame_count(kind):
    """A faceted log axis's off-side count sums every panel's own off-side
    rows into the one warning, not only the first panel's: 3 off-side rows
    split 1 and 2 across `col=` still warn `3 row(s) of 'v'`."""
    group_a = np.concatenate([_V20, [0.0]])
    group_b = np.concatenate([_V20, [-1.0, -2.0]])
    df = pl.DataFrame(
        {"v": np.concatenate([group_a, group_b]), "g": ["a"] * len(group_a) + ["b"] * len(group_b)}
    )
    msgs, _ = _warnings_of(
        lambda: fm.displot(df, x="v", col="g", kind=kind, log_scale=True).to_svg()
    )
    assert len(msgs) == 1, (kind, msgs)
    assert "3 row(s) of 'v'" in msgs[0], (kind, msgs[0])


# ---------------------------------------------------------------------------
# A step the caller wrote directly in .transform() keeps data space; a log
# count axis renders identically through every sibling route; empty data
# keeps its existing path.
# ---------------------------------------------------------------------------


def test_a_transform_step_on_a_log_axis_keeps_data_space():
    """A `Bin` the caller wrote directly in `.transform()` is never the
    lowered statistic the log-space demand reaches, so its edges stay in
    data space (decreasing pixel width as the bin's values climb), warned
    by its own `bin_start` column like any other primitive read of that
    column."""
    msgs, svg = _warnings_of(
        lambda: (
            fm.Chart(_DF20)
            .transform(fm.Bin(field="v"))
            .mark_bar()
            .encode(x=fm.X("bin_start", scale=fm.LogScale()), x2="bin_end", y="count")
            .to_svg()
        )
    )
    widths = _bar_edge_widths(svg)
    assert len(widths) >= 3, widths
    assert max(widths) > min(widths) * 1.5, widths
    assert msgs == [
        "x scale: 1 row(s) of 'bin_start' are zero or negative, which a LogScale cannot "
        "place, so they are not drawn; filter them, or use fm.SymlogScale() for data that "
        "reaches zero"
    ]


def test_a_log_count_axis_renders_identically_through_every_sibling_route():
    """A histogram with a `LogScale` on its count axis -- a route the
    run-plan's demand never reaches, since nothing draws a lowered
    statistic's output on that channel -- renders byte-identical whether
    the scale sits on a channel object or comes from `.override()`, and
    bins its x field equal-width in data units either way."""
    via_channel = (
        fm.Chart(_DF20).mark_histogram().encode(x="v", y=fm.Y(scale=fm.LogScale())).to_svg()
    )
    via_override = (
        fm.Chart(_DF20).mark_histogram().encode(x="v").override(y_scale_type="log").to_svg()
    )
    assert via_channel == via_override

    widths = _bar_edge_widths(via_channel)
    assert len(widths) >= 3, widths
    assert max(widths) == pytest.approx(min(widths), rel=0.02), widths


def test_empty_data_on_a_log_axis_still_draws_an_empty_panel():
    """Empty data takes the existing no-rows-to-draw path, not the
    log-scale machinery: the chart still draws its frame and warns once, as
    it does for any other empty chart."""
    empty_df = pl.DataFrame({"v": pl.Series([], dtype=pl.Float64)})
    msgs, svg = _warnings_of(lambda: fm.displot(empty_df, x="v", log_scale=True).to_svg())
    assert any("no rows to draw" in m for m in msgs), msgs
    assert "<svg" in svg


# ---------------------------------------------------------------------------
# The negative side (an explicit negative LogScale domain) orders every
# pairwise extent, cumulative sum and dodge slot in data order, the same
# direction the positive side uses.
# ---------------------------------------------------------------------------


def test_a_faceted_negative_side_histogram_renders_every_bar():
    """A faceted histogram on a negative-side `LogScale` keeps every row in
    every panel, and every panel's bars render at one pixel width: the
    negative side's facet pin mirrors the positive side's."""
    v = -_V20
    df = pl.DataFrame({"v": np.concatenate([v, v]), "g": ["a"] * 20 + ["b"] * 20})
    msgs, svg = _warnings_of(
        lambda: fm.displot(
            df, x=fm.X("v", scale=fm.LogScale(domain=[-30.0, -0.5])), col="g", kind="hist"
        ).to_svg()
    )
    assert msgs == [], msgs
    _assert_panels_share_one_bin_width(svg, panel_totals=[20.0, 20.0])


def test_a_negative_side_cumulative_histogram_grows_in_data_order():
    """A negative-side cumulative histogram's bar heights climb as x
    climbs (toward zero), in the same left-to-right data order the
    positive side uses."""
    df = pl.DataFrame({"v": -_V20})
    svg = (
        fm.Chart(df)
        .mark_histogram(cumulative=True)
        .encode(x=fm.X("v", scale=fm.LogScale(domain=[-30.0, -0.5])))
        .to_svg()
    )
    bars = sorted(data_bar_rects(svg), key=lambda b: float(b["x"]))
    heights = [float(b["height"]) for b in bars]
    assert heights == sorted(heights), heights


def test_a_negative_side_dodge_orders_slots_as_the_positive_side_does():
    """Negative-side dodge slots order their groups left to right the same
    way the positive side does, group for group."""
    positive_df = pl.DataFrame({"v": np.concatenate([_V20, _V20]), "g": ["a"] * 20 + ["b"] * 20})
    positive_svg = fm.displot(
        positive_df, x="v", hue="g", multiple="dodge", log_scale=True
    ).to_svg()
    positive_order = [
        bar["fill"] for bar in sorted(data_bar_rects(positive_svg), key=lambda b: float(b["x"]))
    ]

    negative_v = -_V20
    negative_df = pl.DataFrame(
        {"v": np.concatenate([negative_v, negative_v]), "g": ["a"] * 20 + ["b"] * 20}
    )
    negative_svg = fm.displot(
        negative_df,
        x=fm.X("v", scale=fm.LogScale(domain=[-30.0, -0.5])),
        hue="g",
        multiple="dodge",
    ).to_svg()
    negative_order = [
        bar["fill"] for bar in sorted(data_bar_rects(negative_svg), key=lambda b: float(b["x"]))
    ]

    assert negative_order == positive_order


# ---------------------------------------------------------------------------
# Every numeric column type a log door reads renders like the same values in
# a Float64 column: the statistic reads the caller's column in its own type,
# and the caller's column reaches every other channel unchanged.
# ---------------------------------------------------------------------------

_DTYPE_V = list(range(1, 90))
#: A permutation of `_DTYPE_V`, so a bivariate kde over `(v, w)` has spread
#: on both axes.
_DTYPE_W = [(37 * value) % 89 + 1 for value in _DTYPE_V]
_DTYPE_G = ["a", "b"] * 44 + ["a"]


def _dtype_columns(off_side: bool) -> tuple[list[int], list[int], list[str]]:
    """The `v`, `w` and `g` columns every dtype door renders: `v` in 1..89,
    plus one `v = 0` row (a value every numeric type holds, and no log axis
    places) when `off_side`."""
    if not off_side:
        return _DTYPE_V, _DTYPE_W, _DTYPE_G
    return [0, *_DTYPE_V], [45, *_DTYPE_W], ["b", *_DTYPE_G]


def _polars_frame(dtype: pl.DataType) -> Callable[[list[int], list[int], list[str]], _Frame]:
    def build(v: list[int], w: list[int], g: list[str]) -> _Frame:
        return pl.DataFrame(
            {"v": pl.Series(v, dtype=dtype), "w": pl.Series(w, dtype=dtype), "g": g}
        )

    return build


def _pyarrow_int32_frame(v: list[int], w: list[int], g: list[str]) -> _Frame:
    return pa.table({"v": pa.array(v, type=pa.int32()), "w": pa.array(w, type=pa.int32()), "g": g})


def _pyarrow_non_nullable_float64_frame(v: list[int], w: list[int], g: list[str]) -> _Frame:
    """`v` and `w` declared `nullable=False`. The frame the renderer receives
    declares them nullable, so this pins that such a frame renders like any
    Float64 one; the renderer's own handling of a non-nullable field is
    unit-tested beside the log-space mapper."""
    schema = pa.schema(
        [
            pa.field("v", pa.float64(), nullable=False),
            pa.field("w", pa.float64(), nullable=False),
            pa.field("g", pa.string()),
        ]
    )
    return pa.Table.from_arrays(
        [
            pa.array([float(x) for x in v]),
            pa.array([float(x) for x in w]),
            pa.array(g),
        ],
        schema=schema,
    )


_FLOAT64_FRAME = _polars_frame(pl.Float64)

#: dtype name -> builder of the `(v, w, g)` frame in that type.
_DTYPE_FRAMES: dict[str, Callable[[list[int], list[int], list[str]], _Frame]] = {
    "polars Int64": _polars_frame(pl.Int64),
    "pyarrow Int32": _pyarrow_int32_frame,
    "polars UInt8": _polars_frame(pl.UInt8),
    "polars Float32": _polars_frame(pl.Float32),
    "pyarrow Float64 non-nullable": _pyarrow_non_nullable_float64_frame,
}

#: door name -> the chart that door builds over a `(v, w, g)` frame.
_DTYPE_DOORS: dict[str, Callable[[_Frame], fm.Chart]] = {
    "displot hist": lambda df: fm.displot(df, x="v", log_scale=True),
    "displot kde": lambda df: fm.displot(df, x="v", kind="kde", log_scale=True),
    "displot ecdf": lambda df: fm.displot(df, x="v", kind="ecdf", log_scale=True),
    "displot bivariate kde": lambda df: fm.displot(df, x="v", y="w", kind="kde", log_scale=True),
    "displot bivariate kde x == y": lambda df: fm.displot(
        df, x="v", y="v", kind="kde", log_scale=True
    ),
    "mark_histogram x LogScale": lambda df: (
        fm.Chart(df).mark_histogram().encode(x=fm.X("v", scale=fm.LogScale()))
    ),
    "mark_bar x bin=True LogScale": lambda df: (
        fm.Chart(df)
        .mark_bar()
        .encode(x=fm.X("v", bin=True, scale=fm.LogScale()), y=fm.Y(aggregate="count"))
    ),
    "override(x_scale_type='log')": lambda df: (
        fm.Chart(df).mark_histogram().encode(x="v").override(x_scale_type="log")
    ),
    "faceted displot hist col='g'": lambda df: fm.displot(df, x="v", col="g", log_scale=True),
    "hist + point(LogScale)": _hist_plus_point,
    "point(LogScale) + hist": _point_plus_hist,
    "LayerChart(hist, point(LogScale))": _layer_chart_hist_then_point,
    "LayerChart(point(LogScale), hist)": _layer_chart_point_then_hist,
    "hconcat leaf": lambda df: fm.hconcat(
        fm.Chart(df).mark_histogram().encode(x=fm.X("v", scale=fm.LogScale())),
        fm.Chart(df).mark_point().encode(x="w", y="w"),
    ),
    "mark_density x LogScale": lambda df: (
        fm.Chart(df).mark_density().encode(x=fm.X("v", scale=fm.LogScale()))
    ),
    "mark_point x bin=True LogScale, y='v'": lambda df: (
        fm.Chart(df).mark_point().encode(x=fm.X("v", bin=True, scale=fm.LogScale()), y="v")
    ),
}


@pytest.mark.parametrize("off_side", [False, True], ids=["positive", "with a 0 row"])
@pytest.mark.parametrize("dtype", sorted(_DTYPE_FRAMES))
@pytest.mark.parametrize("door", sorted(_DTYPE_DOORS))
def test_a_log_door_renders_every_numeric_dtype_like_float64(door, dtype, off_side):
    """Every log door renders an integer, Float32 or non-nullable column
    with the same marks and the same warnings as the same values in a
    polars Float64 column: one `1 row(s) of 'v'` warning when a `0` row is
    present, none otherwise."""
    build = _DTYPE_DOORS[door]
    columns = _dtype_columns(off_side)
    reference_msgs, reference_svg = _warnings_of(lambda: build(_FLOAT64_FRAME(*columns)).to_svg())
    msgs, svg = _warnings_of(lambda: build(_DTYPE_FRAMES[dtype](*columns)).to_svg())

    assert mark_layer(reference_svg), door
    if off_side:
        assert len(reference_msgs) == 1, reference_msgs
        assert "1 row(s) of 'v'" in reference_msgs[0], reference_msgs
    else:
        assert reference_msgs == [], reference_msgs
    assert msgs == reference_msgs, (door, dtype)
    assert mark_layer(svg) == mark_layer(reference_svg), (door, dtype)


def test_a_faceted_int64_histogram_shares_one_log_space_extent_across_panels():
    """A column-faceted log histogram over an Int64 column bins every panel
    over one log-space extent: each panel keeps all its rows, and every bar
    in every panel renders at one pixel width."""
    v, _w, g = _dtype_columns(off_side=False)
    df = pl.DataFrame({"v": pl.Series(v, dtype=pl.Int64), "g": g})
    msgs, svg = _warnings_of(lambda: fm.displot(df, x="v", col="g", log_scale=True).to_svg())
    assert msgs == [], msgs
    _assert_panels_share_one_bin_width(svg, panel_totals=[float(g.count("a")), float(g.count("b"))])
