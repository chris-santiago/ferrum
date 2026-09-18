"""`#171` -- the declared scale type survives the composite-mark desugar.

`Chart._resolve_pending` reduces the chart-level positional channels to bare
field-name strings before a composite mark's desugar runs, and every one of
those desugars rebuilds its categorical channel from that bare string. A bare
string infers its scale type from the column's runtime Arrow dtype, so a
category column stored as `Int64` inferred Quantitative and every band-geometry
consumer downstream lost the band it needed to size against: measured,
`mark_boxplot(size=0.6)` drew **zero** IQR rects and nine 8px `tick.rs`
fallthrough stubs where the identical `Utf8` column drew three band-width
rects and nine band-width spans. The caller's own `"cat:N"` could not rescue it
either -- the declared type was discarded at the same place the field name was
taken.

The fix is two halves and this module pins both, because each alone would be
wrong:

- **Declared-type survival.** The type the caller stated reaches the desugar
  and the channels it builds. `test_errorband_*` is the discriminating pair
  for this half: `mark_errorband`'s x is a documented *continuous* axis, so it
  takes no categorical default, and the only thing that can make it band is the
  caller having said so.
- **Nominal default on a category channel.** Where the caller declared
  nothing, a composite mark whose channel IS the category axis defaults it to
  Nominal. `test_catplot_native_scale_stays_quantitative` is the negative
  control for this half: `catplot(native_scale=True)` deliberately states
  `type_="Q"` on a numeric category column, and a blanket force-nominal would
  silently overrule it.

The dtype sweep is `tests/_category_dtype_matrix.py`'s -- this defect class is
"one dtype behaves unlike its siblings", so a single `Int64` case would prove
one cell of nine. `mark_boxen` is swept only on the string dtypes here: its
numeric cells additionally need `letter_value`'s Rust-side group-dtype gate
(`#185`), which is a separate task, so this module pins boxen's *Python* half
at the spec it builds and leaves the rendered cells to the family's combined
matrix.
"""

from __future__ import annotations

import json
import warnings

import polars as pl
import pytest

import ferrum as fm
from _category_dtype_matrix import (
    CATEGORY_DTYPES,
    band_geometry,
    category_frame,
    category_levels,
    mark_layer,
    utf8_twin_frame,
)
from _svg_rects import svg_elements

#: The composite marks whose categorical channel now defaults to Nominal AND
#: whose geometry is readable as rects/spans. `mark_swarm` is exercised
#: separately (it draws circles, not bands) and `mark_boxen` at the spec level
#: (see the module docstring).
_BAND_MARKS = {
    "boxplot": lambda chart: chart.mark_boxplot(size=0.6),
    "violin": lambda chart: chart.mark_violin(),
    "errorbar": lambda chart: chart.mark_errorbar(),
}


def _svg(build, frame, **encode):
    return build(fm.Chart(frame)).encode(**encode).to_svg()


def test_boxplot_int64_draws_the_bands_its_utf8_twin_draws():
    """The headline repro, stated as the issue states it.

    Three band-width IQR rects and no 8px stub -- where before the fix there
    were zero rects and nine stubs. Asserted against the `Utf8` twin's measured
    geometry rather than against a literal, so the claim is "the dtype stopped
    mattering", not "these happen to be today's pixel values".
    """
    numeric = band_geometry(
        _svg(_BAND_MARKS["boxplot"], category_frame("Int64"), x="cat", y="value")
    )
    reference = band_geometry(
        _svg(_BAND_MARKS["boxplot"], utf8_twin_frame("Int64"), x="cat", y="value")
    )

    assert len(numeric.rect_band_extents) == 3
    assert numeric.stub_count == 0
    assert numeric == reference


@pytest.mark.parametrize("mark", sorted(_BAND_MARKS))
@pytest.mark.parametrize("dtype", sorted(CATEGORY_DTYPES))
def test_band_geometry_is_dtype_invariant(mark, dtype):
    """Every cell of the matrix renders what its `Utf8` twin renders.

    The twin is built with the *same level count* as the cell, which is what
    makes `Boolean` (two levels, so wider bands) a real comparison rather than
    an exclusion.
    """
    build = _BAND_MARKS[mark]
    cell = band_geometry(_svg(build, category_frame(dtype), x="cat", y="value"))
    twin = band_geometry(_svg(build, utf8_twin_frame(dtype), x="cat", y="value"))

    assert cell.stub_count == 0
    # Not merely `cell == twin`: an all-empty geometry would satisfy equality
    # while proving the mark drew nothing at all, which is the very failure
    # this file exists for.
    assert cell.span_lengths, f"{mark} on {dtype} drew no band-relative spans"
    assert cell == twin


@pytest.mark.parametrize("dtype", ["Int64", "Float64", "Boolean", "Utf8"])
def test_swarm_points_land_on_band_centres(dtype):
    """`mark_swarm`'s points sit at the band centres, one per category.

    Swarm draws circles, so it has no rect or span to measure; the property
    that carries the same meaning is *where* the points sit. On an ordinal
    axis every point in a category shares one band centre, and there are
    exactly as many distinct centres as there are levels. On the continuous
    axis a numeric column used to infer, the points spread over the data
    values instead.
    """
    levels = category_levels(dtype)
    cell = _swarm_centres(category_frame(dtype))
    twin = _swarm_centres(utf8_twin_frame(dtype))

    assert len(cell) == len(levels)
    assert cell == twin


def _swarm_centres(frame: pl.DataFrame) -> list[float]:
    svg = fm.Chart(frame).mark_swarm().encode(x="cat", y="value").to_svg()
    marks = mark_layer(svg)
    return sorted(
        {round(float(a["cx"]), 2) for a in svg_elements(marks, "circle", lambda a: "cx" in a)}
    )


def test_horizontal_boxplot_reads_the_y_declaration():
    """The `horizontal=True` twin resolves its category axis off `y`, not `x`.

    Same claim as the headline, on the mirrored axis, because the cat-vs-value
    pick is made independently in every desugar and a fix threaded only
    through the x branch would leave the horizontal half broken.
    """
    frame = category_frame("Int64")
    cell = band_geometry(
        fm.Chart(frame).mark_boxplot(size=0.6, horizontal=True).encode(x="value", y="cat").to_svg(),
        band_axis="y",
    )
    twin = band_geometry(
        fm.Chart(utf8_twin_frame("Int64"))
        .mark_boxplot(size=0.6, horizontal=True)
        .encode(x="value", y="cat")
        .to_svg(),
        band_axis="y",
    )

    assert len(cell.rect_band_extents) == 3
    assert cell.stub_count == 0
    assert cell == twin


def test_boxen_desugar_types_its_group_channel_nominal():
    """`mark_boxen`'s Python half, pinned at the spec it builds.

    Boxen's rendered numeric cells are gated on `letter_value`'s Rust-side
    group-dtype acceptance (`#185`, a separate task), so this asserts the one
    thing `#171` owns for this mark: the channel the desugar binds to
    `letter_value`'s renamed `group` column declares Nominal. Without it, the
    band rects would resolve continuous the moment the transform starts
    returning that column in its original numeric dtype.
    """
    spec = json.loads(
        fm.Chart(category_frame("Int64"))
        .mark_boxen()
        .encode(x="cat", y="value")
        .to_spec()
        .to_json()
    )
    x_channels = [layer["encoding"]["x"] for layer in spec["layers"]]

    assert x_channels, "mark_boxen emitted no layers to check"
    assert all(channel["type"] == "nominal" for channel in x_channels)
    assert all(channel["field"] == "group" for channel in x_channels)


def test_errorband_numeric_x_stays_continuous_by_default():
    """The nominal default is scoped to *category* channels, not applied blanket.

    `desugar_errorband`'s `x_field` is a documented continuous axis the
    transform merely aggregates by. Left undeclared it must keep inferring
    from dtype -- so a numeric x still places its ribbon vertices at the data
    positions, NOT at evenly-spaced band centres. Band centres for three
    levels would be equally spaced; genuine data positions for `4, 6, 8` on a
    padded linear domain are not the same set.
    """
    spec = json.loads(
        fm.Chart(category_frame("Int64"))
        .mark_errorband()
        .encode(x="cat", y="value")
        .to_spec()
        .to_json()
    )
    x_channels = [layer["encoding"]["x"] for layer in spec["layers"]]

    assert x_channels
    # A bare string, exactly as this mark bound it before #171: no `type` key
    # at all, so Rust infers Continuous from the Float-ish dtype.
    assert all(channel.get("type") is None for channel in x_channels), (
        f"errorband applied a categorical default it must not: {x_channels}"
    )


def test_errorband_honors_a_declared_nominal_x():
    """...and the caller's own `:N` now reaches it, which it never used to.

    This is the discriminating half of the pair above: the only thing that can
    make errorband's x ordinal is the declaration, so a green assertion here
    cannot be explained by a blanket default.
    """
    spec = json.loads(
        fm.Chart(category_frame("Int64"))
        .mark_errorband()
        .encode(x="cat:N", y="value")
        .to_spec()
        .to_json()
    )
    x_channels = [layer["encoding"]["x"] for layer in spec["layers"]]

    assert x_channels
    assert all(channel["type"] == "nominal" for channel in x_channels)
    assert all(channel["field"] == "cat" for channel in x_channels)


def test_catplot_native_scale_stays_quantitative():
    """The negative control: a stated `type_="Q"` is not overruled.

    `catplot(native_scale=True)` types its category channel Quantitative on
    purpose, and its two band-relative layers consequently draw nothing and say
    so -- behavior `tests/test_catplot_native_scale.py` pins and `#328` owns.
    If `#171` had been implemented as "force the category channel Nominal"
    rather than "honor the declaration, default only where there is none",
    this chart would silently start banding and that warning would vanish.
    """
    frame = category_frame("Int64")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = fm.catplot(frame, x="cat", y="value", kind="box", native_scale=True).to_svg()
    geometry = band_geometry(svg)

    assert geometry.rect_band_extents == ()
    assert geometry.stub_count == 9
    assert any("drew nothing" in str(w.message) for w in caught)


def test_catplot_default_bands_a_numeric_category_column():
    """The same call without `native_scale=` is the fixed case.

    `catplot`'s default already stated `type_="N"`; before #171 that statement
    was discarded on the way into the desugar, so a numeric category column
    rendered as if it had never been made.
    """
    frame = category_frame("Int64")
    cell = band_geometry(fm.catplot(frame, x="cat", y="value", kind="box").to_svg())
    twin = band_geometry(
        fm.catplot(utf8_twin_frame("Int64"), x="cat", y="value", kind="box").to_svg()
    )

    assert len(cell.rect_band_extents) == 3
    assert cell.stub_count == 0
    assert cell == twin
