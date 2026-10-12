"""The `#171` + `#185` family contract -- neither half alone renders this matrix.

`#171` (landed, wave 5 T1) made the declared/defaulted channel type survive a
composite desugar, so a category column stored as a numeric dtype resolves
ordinal instead of quantitative. `#185` (landed, wave 5 T2) made
`letter_value` (the Rust transform behind `mark_boxen`) accept a non-Utf8
groupby column and return it in its original dtype. Boxen on an `Int64`
column needed BOTH: `#171` alone left it raising (`letter_value` refused the
dtype before the resolved channel type ever mattered), and `#185` alone left
it rendering 8px `tick.rs` stubs with no boxes (the transform ran, but the
category channel still resolved quantitative). `tests/test_declared_type_
through_desugar.py` pins `#171`'s own half -- it sweeps `{boxplot, violin,
errorbar}` over the full dtype matrix and, by its own docstring, deliberately
stops short of boxen's *rendered* numeric cells because those need `#185`
too. This module is where that joint claim gets proven: the same dtype
matrix, now including `boxen`, plus the series-partitioning face of the same
numeric-category dispatch (`stroke_dash=`/`color=` on a declared-categorical
numeric column).

The dtype set, frame builders, and band-geometry extractor are all
`tests/_category_dtype_matrix.py`'s -- consumed here, not rebuilt.
"""

from __future__ import annotations

import warnings

import polars as pl
import pytest

import ferrum as fm
from _category_dtype_matrix import (
    CATEGORY_DTYPES,
    FAMILY_MARKS,
    band_geometry,
    category_frame,
    category_levels,
    render_mark,
    utf8_twin_frame,
)
from _svg_rects import mark_layer, svg_elements

#: `FAMILY_MARKS` is why this module exists rather than folding into
#: `test_declared_type_through_desugar.py`: that module sweeps `BAND_MARKS`
#: (`boxplot`, `violin`, `errorbar`) and stops there, pinning `boxen`'s
#: numeric cells only at the *spec* level -- its rendered geometry needs
#: `#185` too, which is this module's other half. Both tables and the shared
#: `render_mark` helper live in `_category_dtype_matrix`, hoisted there once
#: both consumers existed; a second copy is how the two modules would have
#: drifted apart on, say, `mark_boxplot`'s `size=`.

#: The subset of the matrix that is genuinely numeric (excludes `Boolean`,
#: `Utf8`, `Categorical`) -- the dtypes `catplot(native_scale=True)` actually
#: accepts (it raises on the other three, pinned in
#: `test_catplot_native_scale.py`), and the witnesses swept below for the
#: series-partitioning face.
_NUMERIC_DTYPES = ("Int8", "Int32", "Int64", "UInt8", "Float32", "Float64")


@pytest.mark.parametrize("dtype", sorted(CATEGORY_DTYPES))
@pytest.mark.parametrize("mark", sorted(FAMILY_MARKS))
def test_family_band_geometry_is_dtype_invariant(mark, dtype):
    """Every `{boxplot, violin, boxen, errorbar} x dtype` cell renders what
    its `Utf8` twin renders -- spec section 9's headline acceptance line for
    this family, stated as one matrix rather than three marks' worth of
    already-covered cells plus boxen's newly-covered ones, because the point
    of this module is that the matrix is now complete.

    Not merely `cell == twin`: an all-empty geometry satisfies equality while
    proving the mark drew nothing at all (`#171`'s own failure signature), so
    the non-degenerate guard is checked first.
    """
    build = FAMILY_MARKS[mark]
    cell = band_geometry(render_mark(build, category_frame(dtype), x="cat", y="value"))
    twin = band_geometry(render_mark(build, utf8_twin_frame(dtype), x="cat", y="value"))

    assert cell.stub_count == 0, f"{mark} on {dtype} fell through to tick.rs's quantitative stub"
    assert cell.span_lengths, f"{mark} on {dtype} drew no band-relative spans"
    assert cell == twin


def test_boxen_int64_is_the_cell_that_needs_both_halves():
    """The headline repro, restated for `boxen` specifically.

    Three depth-band rects and no stub, matching the `Utf8` twin exactly --
    where `#171` alone would still raise (the transform never got a groupby
    column it would accept) and `#185` alone would render zero rects and nine
    8px stubs (the category channel still resolved quantitative).
    """
    numeric = band_geometry(
        render_mark(FAMILY_MARKS["boxen"], category_frame("Int64"), x="cat", y="value")
    )
    reference = band_geometry(
        render_mark(FAMILY_MARKS["boxen"], utf8_twin_frame("Int64"), x="cat", y="value")
    )

    assert len(numeric.rect_band_extents) == 3
    assert numeric.stub_count == 0
    assert numeric == reference


@pytest.mark.parametrize("dtype", _NUMERIC_DTYPES)
def test_catplot_native_scale_true_stays_quantitative_across_numeric_dtypes(dtype):
    """The negative control, extended across the family's numeric dtype set.

    `tests/test_declared_type_through_desugar.py::
    test_catplot_native_scale_stays_quantitative` already pins this for
    `Int64`; it is not duplicated here. A blanket force-nominal
    implementation of `#171`'s default would have to special-case every one
    of these dtypes to keep passing the matrix above while still honoring a
    stated `type_="Q"` -- measuring all six numeric dtypes, not just `Int64`,
    is what rules out a per-dtype escape hatch rather than a real "honor the
    declaration" dispatch.
    """
    frame = category_frame(dtype)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = fm.catplot(frame, x="cat", y="value", kind="box", native_scale=True).to_svg()
    geometry = band_geometry(svg)

    assert geometry.rect_band_extents == ()
    assert any("drew nothing" in str(w.message) for w in caught), (
        f"native_scale=True on {dtype} must still warn that its band-relative "
        "layer drew nothing, exactly as it does on Utf8/Int64"
    )


# ---------------------------------------------------------------------------
# mark_tick(band_size=): the deliberate carve-out (spec section 9, amended
# 2026-09-18). Not a composite desugar, so #171's "default the category
# channel to Nominal" does not reach it -- it bands when its channel is
# DECLARED nominal (any dtype) and stubs as a true rug when the channel is an
# undeclared numeric. Neither half is exercised by the matrix above, which is
# exactly why both are pinned here instead.
# ---------------------------------------------------------------------------


def test_tick_bands_when_its_channel_is_declared_nominal():
    """`mark_tick(band_size=)` on a declared-nominal `Int64` x renders the
    same band-relative spans its `Utf8` twin renders -- the primitive's own
    half of the split contract, not routed through composite desugar at all.
    """
    cell = band_geometry(
        fm.Chart(category_frame("Int64"))
        .mark_tick(band_size=0.6)
        .encode(x="cat:N", y="value")
        .to_svg()
    )
    twin = band_geometry(
        fm.Chart(utf8_twin_frame("Int64"))
        .mark_tick(band_size=0.6)
        .encode(x="cat", y="value")
        .to_svg()
    )

    assert cell.rect_band_extents == ()
    assert cell.span_lengths, "a declared-nominal tick channel drew no band-relative spans"
    assert cell.stub_count == 0
    assert cell == twin


def test_tick_stubs_as_a_true_rug_when_its_channel_is_an_undeclared_numeric():
    """The other half: with NO declared type, `mark_tick` keeps inferring
    from dtype like the primitive it is, and an `Int64` x resolves
    quantitative -- one stub per row (a true rug), not band-relative spans.
    This is the one shape in the whole family that must NOT match its `Utf8`
    twin, because `mark_tick` is deliberately exempt from the composite
    default that makes every other mark in this module dtype-invariant.
    """
    frame = category_frame("Int64")
    cell = band_geometry(
        fm.Chart(frame).mark_tick(band_size=0.6).encode(x="cat", y="value").to_svg()
    )

    assert cell.rect_band_extents == ()
    assert cell.span_lengths == ()
    assert cell.stub_count == frame.height


# ---------------------------------------------------------------------------
# Series partitioning: stroke_dash=/color= on a numeric column, the other
# live shape of the same numeric-category dispatch (#187's territory --
# closes as already-fixed by batch A c1dd3ebf, so this pins the contract
# rather than fixing a defect). A declared type_="N"/"O" numeric column
# partitions mark_line into one polyline per category, exactly as a Utf8
# column already does; a bare numeric column keeps one polyline and the
# documented DASH_PALETTE-index contract (ferrum-spec.md §3.2, "StrokeDash
# and the dash palette").
# ---------------------------------------------------------------------------

#: Witness dtypes for the partitioning face: one signed int, one float, one
#: two-level dtype (`Boolean`), rather than the full nine-dtype sweep above --
#: the Rust-side dispatch (`infer_spec_type` honoring `enc.type_` before
#: dtype, then reading the column through the same dtype-generic
#: `col_as_ordinal_category_str` every categorical channel already uses) is
#: the identical mechanism `#171`'s own matrix already exercises across all
#: nine dtypes; three witnesses are enough to show it is not Int64-specific
#: without re-running the whole matrix a second time for a mark that isn't
#: one of the four composites.
_PARTITION_WITNESS_DTYPES = ("Int64", "Float64", "Boolean")


def _partition_frame(dtype: str) -> pl.DataFrame:
    """An `x`/`y`/`series` frame for the `stroke_dash=`/`color=` face,
    `series` stored as *dtype*. Built on `category_frame` (same per-level
    values, same dtype casting) rather than a fourth frame-builder -- only
    the column names and the monotonic row-index `x` (so each `series` group
    is a connected, increasing polyline) differ from the band-geometry
    frame's shape.
    """
    frame = category_frame(dtype, cat_col="series", value_col="y")
    return frame.with_row_index("x").with_columns(pl.col("x").cast(pl.Float64))


def _polylines(svg: str) -> list[dict[str, str]]:
    return svg_elements(mark_layer(svg), "polyline", lambda a: "points" in a)


@pytest.mark.parametrize("type_", ["N", "O"])
@pytest.mark.parametrize("dtype", _PARTITION_WITNESS_DTYPES)
def test_stroke_dash_declared_categorical_numeric_partitions_the_series(dtype, type_):
    """A numeric `stroke_dash` column the caller types `"N"`/`"O"` draws one
    polyline per category -- `dash_cols.categorical` resolves (`stroke_dash_
    column_loader`, `channels.rs`) exactly as it would for a Utf8 column,
    because the gate is the RESOLVED type, not the Arrow dtype.
    """
    frame = _partition_frame(dtype)
    svg = fm.Chart(frame).mark_line().encode(x="x", y="y", stroke_dash=f"series:{type_}").to_svg()
    assert len(_polylines(svg)) == len(category_levels(dtype))


@pytest.mark.parametrize("type_", ["N", "O"])
@pytest.mark.parametrize("dtype", _PARTITION_WITNESS_DTYPES)
def test_color_declared_categorical_numeric_partitions_the_series(dtype, type_):
    """The same claim for `color=`: a declared-categorical numeric column
    resolves `ColorInput::Category`, so `mark_line` groups rows by color
    value into distinct polylines instead of resolving a continuous scale.
    """
    frame = _partition_frame(dtype)
    svg = fm.Chart(frame).mark_line().encode(x="x", y="y", color=f"series:{type_}").to_svg()
    assert len(_polylines(svg)) == len(category_levels(dtype))


def test_stroke_dash_bare_numeric_column_keeps_one_polyline_and_dash_palette_index():
    """The counterpoint: with NO declared type, a numeric `stroke_dash`
    column must NOT partition -- one polyline over every row, its dash
    pattern read as a `DASH_PALETTE` index from the group's first row
    (`resolve_row_stroke_dash`'s pre-`#171`/`#185` contract, untouched by
    either fix). `category_frame`'s first level for `Int64` is `4`, which
    `resolve_stroke_dash` rounds and clamps to index `3` -- `DASH_PALETTE[2]`,
    the long-short pattern `[6, 3, 2, 3]`.
    """
    frame = _partition_frame("Int64")
    svg = fm.Chart(frame).mark_line().encode(x="x", y="y", stroke_dash="series").to_svg()
    polys = _polylines(svg)

    assert len(polys) == 1, "a bare numeric stroke_dash column must not partition the series"
    assert polys[0].get("stroke-dasharray") == "6,3,2,3"


def test_color_bare_numeric_column_keeps_one_polyline():
    """The `color=` counterpoint: a bare numeric column resolves a
    continuous color scale, which `mark_line` cannot paint per-segment, so it
    draws one solid-color polyline and warns instead of silently partitioning
    or silently blanking (`ferrum-spec.md`'s "Continuous/discretizing color
    on mark_line/mark_ribbon fails loudly" note).
    """
    frame = _partition_frame("Int64")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = fm.Chart(frame).mark_line().encode(x="x", y="y", color="series").to_svg()
    polys = _polylines(svg)

    assert len(polys) == 1, "a bare numeric color column must not partition the series"
    assert any("continuous color scale is not supported on line" in str(w.message) for w in caught)
