"""The numeric-category dtype matrix, and the band geometry it is judged by.

Shared contract for the `#171`/`#185`/`#184` family (wave 5). Every member of
that family is one instance of the same defect: a column the user means
*categorically* behaves differently depending on which Arrow dtype it happens
to be stored in. A single `Int64` case proves one cell of that, so the family
is validated as a matrix -- this module owns the dtype set every consumer
sweeps and the geometry extractor every consumer judges a cell by, so the
three tasks that consume it cannot end up sweeping three different dtype sets
or measuring three different things.

Rect extraction, and mark-layer scoping, are both `tests/_svg_rects.py`'s, not
a fourth copy: `band_geometry` calls `data_bar_rects`, `svg_elements`, and
`mark_layer` rather than re-spelling any of the three. The line half
genuinely needs mark-layer scoping: the renderer paints its category-axis
ticks as 4px `<line>` elements in `stroke="#6b7280"` -- the exact stroke and
element the boxplot's own whisker caps and median use, since those marks are
styled `stroke="theme:label"`. No attribute separates them, so the split has
to be structural: the marks (and only the marks) are emitted inside the plot
area's `<g clip-path="url(#ferrum-clip-N)">` group, which `mark_layer` scopes
to. The rect half is scoped through the same group for one reason rather than
two -- a legend swatch is a filled `<rect>` that `data_bar_rects` would
otherwise count as a band.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import ferrum as fm
import polars as pl

from _svg_rects import data_bar_rects, mark_layer, svg_elements

#: The dtype set every consumer of this module sweeps, in a fixed order so a
#: parametrized test's ids are stable. `Utf8` and `Categorical` are in it as
#: the *references*, not as cases under suspicion: the acceptance criterion for
#: a numeric cell is "renders the geometry the equivalent string column
#: renders", so the string columns have to be swept by the same code path that
#: produces the number under test.
CATEGORY_DTYPES: dict[str, Any] = {
    "Int8": pl.Int8,
    "Int32": pl.Int32,
    "Int64": pl.Int64,
    "UInt8": pl.UInt8,
    "Float32": pl.Float32,
    "Float64": pl.Float64,
    "Boolean": pl.Boolean,
    "Utf8": pl.Utf8,
    "Categorical": pl.Categorical,
}

#: The three category levels used for every dtype that can hold them. Small
#: non-negative integers so `Int8`/`UInt8` hold them without overflow and
#: `Float32` holds them exactly -- a cell that differed from its `Utf8` twin
#: because of float rounding in the *data* would be a bogus failure, not a
#: finding about scale dispatch.
_LEVELS = (4, 6, 8)

#: `Boolean` can carry only two levels, so its band width is a two-band width
#: and cannot be compared against a three-band render. Consumers get the right
#: reference automatically by building the twin through
#: :func:`utf8_twin_frame`, which mirrors whatever levels the dtype actually
#: has.
_BOOLEAN_LEVELS = (False, True)

#: Per-level observation values. Fixed (never sampled) so every dtype's render
#: is byte-comparable, and spread widely enough per level that `BoxStats` and
#: `LetterValue` produce a non-degenerate box rather than a zero-height rect
#: that would render identically whether the scale resolved ordinal or not.
_VALUES_PER_LEVEL: tuple[tuple[float, ...], ...] = (
    (29.0, 30.0, 31.0, 33.0, 28.0),
    (21.0, 22.0, 23.0, 25.0, 20.0),
    (14.0, 15.0, 16.0, 18.0, 13.0),
)

#: The pixel length `tick.rs`'s quantitative fallthrough gives a mark that had
#: no band to size itself against. It is a fixed constant there, not a
#: data-derived length, which is exactly what makes it a usable tripwire: a
#: chart whose category channel resolved ordinal never emits one.
STUB_LENGTH_PX = 8.0


def category_levels(dtype_name: str) -> tuple:
    """Return the category levels :func:`category_frame` builds for *dtype_name*.

    Two for `Boolean`, three for everything else. Exposed rather than left
    implicit because the expected *number of bands* in a render is a function
    of it, and a consumer asserting "three band-width rects" has to know which
    cells are the two-band exception.
    """
    _require_known(dtype_name)
    return _BOOLEAN_LEVELS if dtype_name == "Boolean" else _LEVELS


def category_frame(
    dtype_name: str, *, cat_col: str = "cat", value_col: str = "value"
) -> pl.DataFrame:
    """A two-column frame whose category column is stored as *dtype_name*.

    The value column is always `Float64` and always carries the same
    per-level observations, so the ONLY thing that varies across the matrix is
    the category column's Arrow dtype -- which is precisely the variable the
    family's defect is a function of.
    """
    _require_known(dtype_name)
    levels = category_levels(dtype_name)
    cats: list = []
    values: list[float] = []
    for level, level_values in zip(levels, _VALUES_PER_LEVEL):
        cats.extend([level] * len(level_values))
        values.extend(level_values)
    frame = pl.DataFrame({cat_col: cats, value_col: values})
    # `Categorical` has no direct cast from an integer column (polars refuses
    # `i64 -> cat` outright), so it goes through `Utf8` -- the same two-step a
    # user would write, and the reason the levels round-trip as `"4"`/`"6"`/
    # `"8"` rather than as numbers.
    expr = pl.col(cat_col)
    if CATEGORY_DTYPES[dtype_name] == pl.Categorical:
        expr = expr.cast(pl.Utf8)
    return frame.with_columns(expr.cast(CATEGORY_DTYPES[dtype_name]))


def utf8_twin_frame(
    dtype_name: str, *, cat_col: str = "cat", value_col: str = "value"
) -> pl.DataFrame:
    """The `Utf8` reference render's frame for *dtype_name*'s cell.

    Same row count, same values, same number of distinct levels -- only the
    category column's storage differs. This is the frame a numeric cell's
    geometry is compared AGAINST, and it mirrors the cell's own level count
    (two for `Boolean`) rather than hardcoding three, because band width is a
    function of how many bands share the axis.
    """
    frame = category_frame(dtype_name, cat_col=cat_col, value_col=value_col)
    return frame.with_columns(pl.col(cat_col).cast(pl.Utf8))


def _require_known(dtype_name: str) -> None:
    if dtype_name not in CATEGORY_DTYPES:
        raise KeyError(
            f"{dtype_name!r} is not in the category dtype matrix; "
            f"CATEGORY_DTYPES holds {sorted(CATEGORY_DTYPES)}"
        )


class BandGeometry(NamedTuple):
    """What a composite statistical mark drew, reduced to the band question.

    `rect_band_extents`
        Every data `<rect>`'s extent ALONG the band axis, in emission order --
        its width for a vertical chart, its height for a horizontal one. A
        boxplot's IQR box, a boxen's depth bands. Empty is the pre-`#171`
        failure signature: the `rect` mark refuses to draw at all without an
        ordinal axis to size its band-relative extent against.
    `span_lengths`
        The length of every mark segment drawn ALONG the band axis -- a
        boxplot's whisker caps and median, a `tick` mark's crossbar. On a
        working ordinal axis these are all band-relative and equal; the
        pre-fix render emits none at all.
    `stub_count`
        How many mark segments of exactly :data:`STUB_LENGTH_PX` were drawn
        ACROSS the band axis. That is `tick.rs`'s fixed-size quantitative
        fallthrough, and a correctly-banded chart emits zero. Read it together
        with `span_lengths`: a whisker whose data length happened to be 8.0px
        would also land here, so "no stubs" is the discriminating claim only
        alongside "the spans are band-width".
    """

    rect_band_extents: tuple[float, ...]
    span_lengths: tuple[float, ...]
    stub_count: int


def band_geometry(svg: str, *, band_axis: str = "x") -> BandGeometry:
    """Extract *svg*'s band geometry.

    `band_axis` names the axis the category bands are laid out along: `"x"`
    for an ordinary vertical boxplot/violin/errorbar, `"y"` for its
    `horizontal=True` / `orient="horizontal"` twin. It decides which segments
    count as band-relative spans and which as across-band stubs; getting it
    wrong swaps the two, which shows up immediately as zero spans.
    """
    if band_axis not in ("x", "y"):
        raise ValueError(f"band_axis must be 'x' or 'y', got {band_axis!r}")
    marks = mark_layer(svg)
    spans: list[float] = []
    stubs = 0
    for attrs in svg_elements(marks, "line", lambda a: "x1" in a and "y1" in a):
        dx = abs(float(attrs["x2"]) - float(attrs["x1"]))
        dy = abs(float(attrs["y2"]) - float(attrs["y1"]))
        along, across = (dx, dy) if band_axis == "x" else (dy, dx)
        if across == 0.0 and along > 0.0:
            spans.append(round(along, 3))
        elif along == 0.0 and round(across, 3) == STUB_LENGTH_PX:
            stubs += 1
    extent_attr = "width" if band_axis == "x" else "height"
    return BandGeometry(
        rect_band_extents=tuple(round(float(a[extent_attr]), 3) for a in data_bar_rects(marks)),
        span_lengths=tuple(spans),
        stub_count=stubs,
    )


#: The composite statistical marks whose category channel lays out as bands.
#: Built here rather than in either consumer because both
#: `test_declared_type_through_desugar.py` (#171's own half) and
#: `test_numeric_category_dispatch.py` (the #171+#185 family contract) sweep
#: the same marks with the same constructor kwargs, and a second copy is how
#: the two modules would silently drift apart on, say, `mark_boxplot`'s
#: `size=`. `boxen` is separated out because it additionally needs #185's
#: transform fix, so a consumer pinning only #171's half excludes it.
BAND_MARKS = {
    "boxplot": lambda chart: chart.mark_boxplot(size=0.6),
    "violin": lambda chart: chart.mark_violin(),
    "errorbar": lambda chart: chart.mark_errorbar(),
}

#: `BAND_MARKS` plus `boxen` -- the full family, which only renders once both
#: `#171` (ordinal dispatch) and `#185` (transform group dtypes) have landed.
FAMILY_MARKS = {**BAND_MARKS, "boxen": lambda chart: chart.mark_boxen()}


def render_mark(build, frame, **encode) -> str:
    """Render *build*'s mark over *frame*, encoded by **encode**.

    The one-liner both consumers had privately. Kept here so the family's
    render path is stated once: a consumer that needed to pass, say, a theme
    or a coord would otherwise fork it.
    """
    return build(fm.Chart(frame)).encode(**encode).to_svg()
