"""Text-aware per-end range inset.

On a continuous positional scale at default padding (no explicit
``padding=``, no explicit ``domain=``, no explicit ``range=``), a
``mark_text`` layer drawing on that scale insets each end of the pixel
range so every glyph's rendered ink clears the plot frame by at least
``TEXT_INSET_FLOOR_PX``, measured with the mark's own
``align``/``baseline``/``dx``/``dy``/``angle``/``size``/``format``. A tick
and a mark at the same domain value always land on the same pixel, on
every route the inset is built for: a flat chart, shared and independent
facets, a secondary y axis, a ``LayerChart``, ``CoordFlip``, and an
explicit scale object that itself sets no padding/domain/range. A label
too wide for the panel to clear warns exactly once per channel, counting
every overrunning row once regardless of how many panels or composite
leaves resolve that channel; a secondary y slot's own overrun is counted
once chart-wide too, never multiplied by the panel count.

The oracle never re-implements the Rust glyph measurement or the Rust
tick/mark pixel solve. A tick's or a mark's own pixel is read straight
off the rendered SVG (a tick's own dash-and-label pairing, a mark's own
drawn ``<text>``). A label's own clearance is measured by isolating that
one ``<text>`` element -- its own attributes, transform kept, fill forced
solid and unclipped -- into a minimal standalone SVG and rasterizing it
through ``ferrum._core.rasterize_svg`` (the shipped PNG path, which loads
the bundled Inter font), then reading the actual ink pixels back against
a strict 50%-coverage threshold. Measuring one label at a time, with no
clip, means a glyph pushed fully past a frame edge still measures where
it landed instead of disappearing from the scan. Most test labels are
four characters wide (e.g. ``"AAAA"``), wide enough that the floor binds
by a clear margin rather than coinciding with the plain default pad.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
import warnings

import polars as pl
import pytest

import ferrum as fm
from ferrum._layer import _Layer
from tests._snapshots import decode_png_ink_mask
from tests._svg_extents import (
    axis_tick_dash_label_pairs,
    ordered_text_entries,
    text_elements,
    tick_along,
)
from tests._svg_rects import Frame, mark_groups_by_frame

# The spec's named floor (crates/ferrum-core/src/layout/geometry.rs
# `TEXT_INSET_FLOOR_PX`). Not exposed through the PyO3 boundary, so the
# test pins the contract value directly, as the spec itself does.
TEXT_INSET_FLOOR_PX = 4.0

# Grid lines are drawn exactly through every tick/mark position, which is
# exactly where a glyph's ink sits too; disabling them removes a confound
# for the pixel-ink measurements below without moving the frame or any
# tick/mark pixel (see test_grid_lines_do_not_affect_the_frame_or_tick_pixels).
_NO_GRID = fm.Theme(grid=False)

# The scale an isolated single-label SVG is rasterized at, and the
# ink-coverage threshold against it: a pixel is ink once it is at least
# 50% covered by solid black text against the isolated render's solid
# white background (765 = pure white, 0 = pure black; 382 is their
# midpoint).
_RASTER_SCALE = 4.0
_INK_THRESHOLD = 382

_ROOT_OPEN_RE = re.compile(r"<svg[^>]*>")
_OPACITY_ATTR_RE = re.compile(r'\s(?:opacity|fill-opacity)="[^"]*"')
_FILL_ATTR_RE = re.compile(r'\sfill="[^"]*"')


def _corners_df() -> pl.DataFrame:
    """Five rows whose extreme glyph sits at a distinct frame end each.

    ``x`` increases A..E (A leftmost, E rightmost); ``y`` is scrambled
    (3, 5, 1, 4, 2) so the topmost row (B, y=5) and the bottommost row
    (C, y=1) are different rows than the leftmost/rightmost ones. Every
    one of the four frame ends is therefore probed by its own distinct
    glyph. Each label is four characters wide (not a single letter) so the
    floor binds by a clear margin -- see the module docstring.
    """
    return pl.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [3, 5, 1, 4, 2],
            "label": ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE"],
        }
    )


def _corners_df_wide() -> pl.DataFrame:
    """``_corners_df()`` with each label widened to ten characters.

    A wider label is what a route's clearance needs to actually
    discriminate a real shortfall: at four characters a shortfall can
    still coincidentally clear the floor, the same way a single-character
    glyph always would (see the module docstring).
    """
    return pl.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [3, 5, 1, 4, 2],
            "label": ["AAAAAAAAAA", "BBBBBBBBBB", "CCCCCCCCCC", "DDDDDDDDDD", "EEEEEEEEEE"],
        }
    )


def _corners_chart(data: pl.DataFrame | None = None, **mark_kwargs) -> fm.Chart:
    data = _corners_df() if data is None else data
    return (
        fm.Chart(data).mark_text(**mark_kwargs).encode(x="x", y="y", text="label").theme(_NO_GRID)
    )


def _frame_rects(svg: str) -> list[Frame]:
    """Every plot panel's own frame, in document order, paired to its clip id."""
    return [frame for frame, _ in mark_groups_by_frame(svg)]


def _frame_covers(frame: Frame, x: float, y: float, tol: float = 1.0) -> bool:
    fx0, fy0, fx1, fy1 = frame
    return fx0 - tol <= x <= fx1 + tol and fy0 - tol <= y <= fy1 + tol


def _tick_label_positions(svg: str) -> set[tuple[float, float, str]]:
    """Every ``(x, y, text)`` rendered as a tick's own label on either axis.

    ``axis_tick_dash_label_pairs`` groups every ``<text>`` node from one
    tick's dash up to the NEXT dash, which means the group following the
    last tick on an axis also sweeps up that axis's title and every mark
    drawn after it (there is no further ``<line>`` to close the group).
    Only the first label in each group is the tick's own; the rest, for
    the last group, is exactly the content this function must not treat
    as a tick.
    """
    positions: set[tuple[float, float, str]] = set()
    for axis in ("x", "y"):
        for group in axis_tick_dash_label_pairs(svg, axis=axis):
            if group.labels:
                positions.add(group.labels[0])
    return positions


def _mark_text_entries(svg: str) -> list[tuple[float, float, str]]:
    """Every rendered ``<text>`` node that is not a tick's own label.

    A mark's own drawn content is whatever ``ordered_text_entries`` finds
    that is not one of those tick labels -- this holds regardless of
    whether a mark's content happens to look like a tick value (e.g. a
    numeric fallback label), since tick labels are identified structurally
    (by their dash), not by content.
    """
    tick_positions = _tick_label_positions(svg)
    return [e for e in ordered_text_entries(svg) if e not in tick_positions]


def _mark_anchor(svg: str, content: str, *, frame: Frame | None = None) -> tuple[float, float]:
    """The ``(x, y)`` anchor of the one mark-drawn ``<text>`` node reading ``content``."""
    matches = [
        (x, y)
        for x, y, text in _mark_text_entries(svg)
        if text == content and (frame is None or _frame_covers(frame, x, y))
    ]
    where = f" in frame {frame}" if frame is not None else ""
    assert matches, f"no mark labelled {content!r} found{where}"
    assert len(set(matches)) == 1, f"ambiguous mark anchor for {content!r}{where}: {matches}"
    return matches[0]


def _tick_pixel(svg: str, *, axis: str, label: str, frame: Frame | None = None) -> float:
    """The along-axis pixel of the one ``axis`` tick labelled ``label``.

    Only each group's first label is a tick's own (see
    ``_tick_label_positions``'s docstring on why the rest of the last
    group must not be read as tick content). When ``frame`` is given, only
    ticks whose own along-axis position falls within that panel's frame
    (plus a small tolerance for the tick's own dash length) are considered
    -- the tick's across-axis position sits outside the frame by
    construction (ticks are drawn beyond the plot edge), so only the
    along-axis coordinate is checked.
    """
    candidates = []
    for group in axis_tick_dash_label_pairs(svg, axis=axis):
        if not group.labels or group.labels[0][2] != label:
            continue
        pixel = tick_along(group.dash, axis=axis)
        if frame is not None:
            fx0, fy0, fx1, fy1 = frame
            lo, hi = (fx0, fx1) if axis == "x" else (fy0, fy1)
            if not (lo - 2.0 <= pixel <= hi + 2.0):
                continue
        candidates.append(pixel)
    where = f" in frame {frame}" if frame is not None else ""
    assert candidates, f"no {axis} tick labelled {label!r} found{where}"
    assert len(set(candidates)) == 1, (
        f"ambiguous {axis} tick pixel for {label!r}{where}: {candidates}"
    )
    return candidates[0]


def _same_offset(a: float, b: float, *, tol: float = 0.01) -> bool:
    """Whether two independently-rounded pixel differences agree.

    Each side here is itself a subtraction of two already-3-decimal-
    quantized SVG coordinates (e.g. a tick's pixel minus its own panel's
    frame origin), so comparing two such differences for exact equality
    can fail on the last bit of float rounding even when both sides
    represent the identical geometric offset -- this compares them within
    a tolerance far tighter than any real misalignment this module tests
    for (which are always several pixels).
    """
    return abs(a - b) < tol


def _label_ink_box(svg: str, attrs: str, text: str) -> Frame:
    """The rendered ink bounding box of one label, isolated and unclipped.

    Builds a minimal SVG carrying the source root's own ``width``/
    ``height``/``viewBox``, a white background, and only this one
    ``<text>`` element with its own attributes (its ``transform`` kept),
    fill forced to solid black and any opacity removed, so measuring it
    never competes with chrome, a neighbouring glyph, or the source
    render's own fill/opacity. It rasterizes through
    ``ferrum._core.rasterize_svg``, the shipped PNG path that loads the
    bundled Inter font -- not ``resvg_py``'s system fallback. The box is
    in SVG units, each ink pixel's own outer edge (``left/top = p /
    scale``, ``right/bottom = (p + 1) / scale``), and there is no clip, so
    a glyph pushed fully past a frame edge still measures where it
    actually landed.
    """
    root_match = _ROOT_OPEN_RE.search(svg)
    assert root_match, "no <svg> root element found"
    root_attrs = dict(re.findall(r'([\w-]+)="([^"]*)"', root_match.group(0)))
    width, height, view_box = root_attrs["width"], root_attrs["height"], root_attrs["viewBox"]
    isolated_attrs = _OPACITY_ATTR_RE.sub("", attrs)
    isolated_attrs = (
        _FILL_ATTR_RE.sub(' fill="#000000"', isolated_attrs)
        if _FILL_ATTR_RE.search(isolated_attrs)
        else isolated_attrs + ' fill="#000000"'
    )
    isolated_svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="{view_box}"><rect width="100%" height="100%" fill="#ffffff"/>'
        f"<text {isolated_attrs}>{text}</text></svg>"
    )
    png = fm._core.rasterize_svg(isolated_svg, scale=_RASTER_SCALE)
    bbox = decode_png_ink_mask(png, threshold=_INK_THRESHOLD)
    assert bbox is not None, f"label {text!r} drew no ink"
    x_min, x_max, y_min, y_max = bbox
    return (
        x_min / _RASTER_SCALE,
        y_min / _RASTER_SCALE,
        (x_max + 1) / _RASTER_SCALE,
        (y_max + 1) / _RASTER_SCALE,
    )


def _attr_xy(attrs: str) -> tuple[float, float]:
    x_match = re.search(r'x="([^"]+)"', attrs)
    y_match = re.search(r'y="([^"]+)"', attrs)
    assert x_match and y_match, f"no x/y position in {attrs!r}"
    return float(x_match.group(1)), float(y_match.group(1))


def _label_clearances(svg: str, expected: dict[str, int]) -> dict[str, dict[str, float]]:
    """Each expected label's own clearance from the four ends of its own panel frame.

    ``expected`` maps each label's own drawn text to the index of the
    panel frame (``_frame_rects(svg)``'s order) it must be found in.
    ``mark_text``'s glyphs render as siblings right after their panel's
    (otherwise empty) clip group, never inside it, so a label's frame is
    the one its own anchor position falls within, never found through
    clip-group content. Every expected label must be present, or this
    fails outright -- a label the render dropped is exactly the kind of
    bug the floor exists to catch, not a clearance of infinity.
    """
    frames = _frame_rects(svg)
    out: dict[str, dict[str, float]] = {}
    for label, panel_idx in expected.items():
        frame = frames[panel_idx]
        candidates = [
            (attrs, x, y)
            for attrs, text in text_elements(svg)
            if text == label
            for x, y in (_attr_xy(attrs),)
            if _frame_covers(frame, x, y)
        ]
        assert candidates, f"label {label!r} not found in frame {frame}"
        assert len(candidates) == 1, f"ambiguous label {label!r} in frame {frame}: {candidates}"
        attrs, _, _ = candidates[0]
        x_min, y_min, x_max, y_max = _label_ink_box(svg, attrs, label)
        fx0, fy0, fx1, fy1 = frame
        out[label] = {
            "left": x_min - fx0,
            "right": fx1 - x_max,
            "top": y_min - fy0,
            "bottom": fy1 - y_max,
        }
    return out


def _assert_labels_clear(svg: str, expected: dict[str, int]) -> None:
    """Every expected label's ink clears the floor on every one of its frame's ends."""
    for label, ends in _label_clearances(svg, expected).items():
        for end, value in ends.items():
            assert value >= TEXT_INSET_FLOOR_PX, (
                f"{label!r} {end} clearance {value} < {TEXT_INSET_FLOOR_PX}"
            )


_CORNER_LABELS = ("AAAA", "BBBB", "CCCC", "DDDD", "EEEE")
_CORNER_LABELS_WIDE = ("AAAAAAAAAA", "BBBBBBBBBB", "CCCCCCCCCC", "DDDDDDDDDD", "EEEEEEEEEE")


def _corners_at(panel_idx: int = 0, labels: tuple[str, ...] = _CORNER_LABELS) -> dict[str, int]:
    """``{label: panel_idx}`` for every one of ``labels`` in panel ``panel_idx``."""
    return {label: panel_idx for label in labels}


def _digest(svg: str) -> str:
    return hashlib.sha256(svg.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Case: every end clears the floor at default padding.
# ---------------------------------------------------------------------------


def test_mark_text_clears_all_four_ends_at_default_padding():
    svg = _corners_chart().to_svg()
    _assert_labels_clear(svg, _corners_at(0))


def test_grid_lines_do_not_affect_the_frame_or_tick_pixels():
    data = _corners_df()
    with_grid = fm.Chart(data).mark_text().encode(x="x", y="y", text="label").to_svg()
    without_grid = _corners_chart(data).to_svg()
    assert _frame_rects(with_grid) == _frame_rects(without_grid)
    assert set(_mark_text_entries(with_grid)) >= {
        e
        for e in _mark_text_entries(without_grid)
        if e[2] in {"AAAA", "BBBB", "CCCC", "DDDD", "EEEE"}
    }


# ---------------------------------------------------------------------------
# Case: a tick and a mark at the same domain value share one pixel, on
# every route the inset is built for -- plus ink clearance on the axis
# where the floor binds, with labels wide enough that the floor actually
# has to grow past the plain default pad for the check to mean anything.
# ---------------------------------------------------------------------------


def test_tick_pixel_equals_mark_pixel_flat_x_and_y():
    svg = _corners_chart().to_svg()
    frame = _frame_rects(svg)[0]
    ax, _ = _mark_anchor(svg, "AAAA", frame=frame)  # x=1
    assert ax == _tick_pixel(svg, axis="x", label="1", frame=frame)
    ex, _ = _mark_anchor(svg, "EEEE", frame=frame)  # x=5
    assert ex == _tick_pixel(svg, axis="x", label="5", frame=frame)
    _, by = _mark_anchor(svg, "BBBB", frame=frame)  # y=5
    assert by == _tick_pixel(svg, axis="y", label="5", frame=frame)
    _, cy = _mark_anchor(svg, "CCCC", frame=frame)  # y=1
    assert cy == _tick_pixel(svg, axis="y", label="1", frame=frame)
    _assert_labels_clear(svg, _corners_at(0))


def _shared_facet_df_with_a_sparse_panel() -> pl.DataFrame:
    """Two panels sharing one x domain; panel ``p2`` has no row at either end.

    Panel ``p1`` draws the full set (x=1..5); panel ``p2`` draws only
    x=2..4 (``BBBB``, ``CCCC``, ``DDDD``) -- it has no row at the domain's
    boundary values, so its own rows alone could never establish the
    shared inset.
    """
    full = _corners_df()
    sparse = full.filter(pl.col("x").is_in([2, 3, 4])).with_columns(g=pl.lit("p2"))
    return pl.concat([full.with_columns(g=pl.lit("p1")), sparse])


def test_tick_pixel_equals_mark_pixel_shared_facet_with_a_panel_missing_the_boundary_rows():
    df = _shared_facet_df_with_a_sparse_panel()
    svg = (
        fm.Chart(df)
        .mark_text()
        .encode(x="x", y="y", text="label", facet_col="g")
        .theme(_NO_GRID)
        .to_svg()
    )
    frames = _frame_rects(svg)
    assert len(frames) == 2
    frame_p1, frame_p2 = frames

    ax, _ = _mark_anchor(svg, "AAAA", frame=frame_p1)
    tick_x_p1 = _tick_pixel(svg, axis="x", label="1", frame=frame_p1)
    assert ax == tick_x_p1

    # Panel p2 has no "AAAA" row at x=1, but the shared x tick for x=1
    # still has to sit at the same offset from p2's own frame as it does
    # from p1's -- proving the shared demand (built from the union of
    # every panel's rows) reaches a panel that contributes none of the
    # boundary rows itself.
    tick_x_p2 = _tick_pixel(svg, axis="x", label="1", frame=frame_p2)
    assert _same_offset(tick_x_p1 - frame_p1[0], tick_x_p2 - frame_p2[0])

    # The y axis is shared too, and is where the floor actually binds for
    # this label width: panel p2's own "DDDD" (y=4) is not at the shared y
    # domain's boundary, but p2's own tick/mark agreement and clearance on
    # its own frame are exactly what measuring each panel from its own
    # rows alone, instead of the union every panel shares, would break.
    _, dy = _mark_anchor(svg, "DDDD", frame=frame_p2)
    assert dy == _tick_pixel(svg, axis="y", label="4", frame=frame_p2)
    _assert_labels_clear(svg, _corners_at(0))
    _assert_labels_clear(svg, {"BBBB": 1, "CCCC": 1, "DDDD": 1})


def test_tick_pixel_equals_mark_pixel_independent_facet_x():
    df = _shared_facet_df_with_a_sparse_panel()
    svg = (
        fm.Chart(df)
        .mark_text()
        .encode(x="x", y="y", text="label", facet_col="g")
        .share_scale(x="independent")
        .theme(_NO_GRID)
        .to_svg()
    )
    frames = _frame_rects(svg)
    assert len(frames) == 2
    frame_p1, frame_p2 = frames

    ax, _ = _mark_anchor(svg, "AAAA", frame=frame_p1)
    assert ax == _tick_pixel(svg, axis="x", label="1", frame=frame_p1)
    _assert_labels_clear(svg, _corners_at(0))

    # Panel p2's own independent x domain is [2, 4]; its own boundary row
    # is "DDDD" (x=4). The x axis is the one each panel resolves on its
    # own, so this is where an independent-facet split would show up.
    dx, _ = _mark_anchor(svg, "DDDD", frame=frame_p2)
    assert dx == _tick_pixel(svg, axis="x", label="4", frame=frame_p2)
    _assert_labels_clear(svg, {"BBBB": 1, "CCCC": 1, "DDDD": 1})


def test_tick_pixel_equals_mark_pixel_independent_facet_y():
    df = _shared_facet_df_with_a_sparse_panel()
    svg = (
        fm.Chart(df)
        .mark_text()
        .encode(x="x", y="y", text="label", facet_col="g")
        .share_scale(y="independent")
        .theme(_NO_GRID)
        .to_svg()
    )
    frames = _frame_rects(svg)
    assert len(frames) == 2
    frame_p1, frame_p2 = frames

    _, by = _mark_anchor(svg, "BBBB", frame=frame_p1)  # y=5, p1's own max
    assert by == _tick_pixel(svg, axis="y", label="5", frame=frame_p1)
    _, cy = _mark_anchor(svg, "CCCC", frame=frame_p1)  # y=1, p1's own min
    assert cy == _tick_pixel(svg, axis="y", label="1", frame=frame_p1)
    _assert_labels_clear(svg, _corners_at(0))

    # Panel p2's own independent y domain is [1, 5] too (its rows are
    # BBBB, CCCC, DDDD), so its own boundary rows are BBBB (y=5) and CCCC
    # (y=1) again.
    _, by2 = _mark_anchor(svg, "BBBB", frame=frame_p2)
    assert by2 == _tick_pixel(svg, axis="y", label="5", frame=frame_p2)
    _, cy2 = _mark_anchor(svg, "CCCC", frame=frame_p2)
    assert cy2 == _tick_pixel(svg, axis="y", label="1", frame=frame_p2)
    _assert_labels_clear(svg, {"BBBB": 1, "CCCC": 1, "DDDD": 1})


def test_tick_pixel_equals_mark_pixel_secondary_y_text():
    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "z": [30, 50, 10, 40, 20],
            "label": ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE"],
        }
    )
    base = fm.Chart(df).mark_point().encode(x="x", y="x")
    chart = (base + fm.SecondaryY("z", mark="text")).theme(_NO_GRID)
    svg = chart.to_svg()
    frame = _frame_rects(svg)[0]

    # The secondary layer draws z's own value as its label (no explicit
    # text= channel on SecondaryY), so its glyphs read "30".."50".
    _, by = _mark_anchor(svg, "50", frame=frame)  # z=50, the secondary max
    assert by == _tick_pixel(svg, axis="y", label="50", frame=frame)
    _, cy = _mark_anchor(svg, "10", frame=frame)  # z=10, the secondary min
    assert cy == _tick_pixel(svg, axis="y", label="10", frame=frame)
    clearances = _label_clearances(svg, {"50": 0, "10": 0})
    assert clearances["50"]["top"] >= TEXT_INSET_FLOOR_PX
    assert clearances["10"]["bottom"] >= TEXT_INSET_FLOOR_PX


def _secondary_text_layer(df: pl.DataFrame, **mark_kwargs) -> fm.Chart:
    """A secondary-y text layer with its own mark style, via the same
    ``resolve={"y": "independent"}`` mechanism ``chart + SecondaryY(...)``
    desugars into (``SecondaryY`` itself accepts no style kwargs, so this
    is the one way to give the secondary text layer its own font_size/dy).
    """
    base = fm.Chart(df).mark_point().encode(x="x", y="x")
    text_layer = fm.Chart(df).mark_text(**mark_kwargs).encode(x="x", y="z", text="label")
    return fm.LayerChart(base, text_layer, resolve={"y": "independent"}).theme(_NO_GRID)


def test_tick_pixel_equals_mark_pixel_layerchart_resolve_y_independent():
    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "z": [30, 50, 10, 40, 20],
            "label": ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE"],
        }
    )
    svg = _secondary_text_layer(df).to_svg()
    frame = _frame_rects(svg)[0]

    _, by = _mark_anchor(svg, "BBBB", frame=frame)  # z=50
    assert by == _tick_pixel(svg, axis="y", label="50", frame=frame)
    _, ay = _mark_anchor(svg, "AAAA", frame=frame)  # z=30
    assert ay == _tick_pixel(svg, axis="y", label="30", frame=frame)
    clearances = _label_clearances(svg, {"BBBB": 0, "CCCC": 0})
    assert clearances["BBBB"]["top"] >= TEXT_INSET_FLOOR_PX  # z=50, the secondary max
    assert clearances["CCCC"]["bottom"] >= TEXT_INSET_FLOOR_PX  # z=10, the secondary min


def test_secondary_y_text_layer_with_its_own_font_size_clears_the_frame():
    # A 30px font on the secondary layer only: the host's own style (11px)
    # must not be what gets measured, or this clearance fails.
    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "z": [30, 50, 10, 40, 20],
            "label": ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE"],
        }
    )
    svg = _secondary_text_layer(df, font_size=30).to_svg()
    clearances = _label_clearances(svg, {"BBBB": 0, "CCCC": 0})
    assert clearances["BBBB"]["top"] >= TEXT_INSET_FLOOR_PX
    assert clearances["CCCC"]["bottom"] >= TEXT_INSET_FLOOR_PX


def test_secondary_y_text_layer_with_its_own_dy_offset_clears_the_frame():
    # A -20px dy on the secondary layer pushes every label up; the inset
    # must fold the layer's own offset in, not the host's (which has none).
    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "z": [30, 50, 10, 40, 20],
            "label": ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE"],
        }
    )
    svg = _secondary_text_layer(df, dy=-20).to_svg()
    clearances = _label_clearances(svg, {"BBBB": 0, "CCCC": 0})
    assert clearances["BBBB"]["top"] >= TEXT_INSET_FLOOR_PX
    assert clearances["CCCC"]["bottom"] >= TEXT_INSET_FLOOR_PX


@pytest.mark.parametrize("layer_order", ["text_first", "points_first"])
def test_tick_pixel_equals_mark_pixel_layerchart_text_and_points_overlay(layer_order):
    df = _corners_df()
    text_layer = fm.Chart(df).mark_text().encode(x="x", y="y", text="label")
    point_layer = fm.Chart(df).mark_point().encode(x="x", y="y")
    layers = (text_layer, point_layer) if layer_order == "text_first" else (point_layer, text_layer)
    svg = fm.LayerChart(*layers).theme(_NO_GRID).to_svg()
    frame = _frame_rects(svg)[0]

    _, by = _mark_anchor(svg, "BBBB", frame=frame)  # y=5
    assert by == _tick_pixel(svg, axis="y", label="5", frame=frame)
    _, cy = _mark_anchor(svg, "CCCC", frame=frame)  # y=1
    assert cy == _tick_pixel(svg, axis="y", label="1", frame=frame)
    _assert_labels_clear(svg, _corners_at(0))


def test_tick_pixel_equals_mark_pixel_coord_flip_unlayered():
    # Wide (ten-character) labels on distinct x/y fields: this is the
    # fixture that discriminates a flipped unlayered chart reading the
    # wrong channel's rows, where the module's usual four-character
    # labels can coincidentally still clear the floor.
    svg = _corners_chart(data=_corners_df_wide()).coord(fm.CoordFlip()).to_svg()
    frame = _frame_rects(svg)[0]

    # Under CoordFlip the x encoding (1..5) draws on the vertical axis and
    # the y encoding (scrambled 3/5/1/4/2) draws on the horizontal one.
    # AAAAAAAAAA (x=1) and EEEEEEEEEE (x=5) are the vertical extremes;
    # BBBBBBBBBB (y=5) and CCCCCCCCCC (y=1) are the horizontal extremes.
    _, ay = _mark_anchor(svg, "AAAAAAAAAA", frame=frame)
    assert ay == _tick_pixel(svg, axis="y", label="1", frame=frame)
    _, ey = _mark_anchor(svg, "EEEEEEEEEE", frame=frame)
    assert ey == _tick_pixel(svg, axis="y", label="5", frame=frame)
    bx, _ = _mark_anchor(svg, "BBBBBBBBBB", frame=frame)
    assert bx == _tick_pixel(svg, axis="x", label="5", frame=frame)
    cx, _ = _mark_anchor(svg, "CCCCCCCCCC", frame=frame)
    assert cx == _tick_pixel(svg, axis="x", label="1", frame=frame)
    _assert_labels_clear(svg, _corners_at(0, labels=_CORNER_LABELS_WIDE))


def test_tick_pixel_equals_mark_pixel_coord_flip_layered():
    df = _corners_df()
    svg = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x", y="y")
        .layer(fm.Layer(mark="text", encoding={"text": "label"}))
        .coord(fm.CoordFlip())
        .theme(_NO_GRID)
        .to_svg()
    )
    frame = _frame_rects(svg)[0]
    _, ay = _mark_anchor(svg, "AAAA", frame=frame)
    assert ay == _tick_pixel(svg, axis="y", label="1", frame=frame)
    bx, _ = _mark_anchor(svg, "BBBB", frame=frame)
    assert bx == _tick_pixel(svg, axis="x", label="5", frame=frame)
    _assert_labels_clear(svg, _corners_at(0))


@pytest.mark.parametrize(
    "scale",
    [
        fm.LinearScale(),
        fm.LinearScale(nice=True),
        fm.LinearScale(reverse=True),
        fm.PowScale(),
        fm.SqrtScale(),
    ],
    ids=["LinearScale()", "nice=True", "reverse=True", "PowScale()", "SqrtScale()"],
)
def test_tick_pixel_equals_mark_pixel_with_a_default_scale_object(scale):
    df = _corners_df()
    svg = (
        fm.Chart(df)
        .mark_text()
        .encode(x="x", y=fm.Y("y", scale=scale), text="label")
        .theme(_NO_GRID)
        .to_svg()
    )
    frame = _frame_rects(svg)[0]
    _, by = _mark_anchor(svg, "BBBB", frame=frame)
    assert by == _tick_pixel(svg, axis="y", label="5", frame=frame)
    _, cy = _mark_anchor(svg, "CCCC", frame=frame)
    assert cy == _tick_pixel(svg, axis="y", label="1", frame=frame)
    clearances = _label_clearances(svg, {"BBBB": 0, "CCCC": 0})
    assert clearances["BBBB"]["top"] >= TEXT_INSET_FLOOR_PX
    assert clearances["CCCC"]["bottom"] >= TEXT_INSET_FLOOR_PX


def test_tick_pixel_equals_mark_pixel_with_a_default_log_scale():
    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [10, 1000, 1, 100, 10000],
            "label": ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE"],
        }
    )
    svg = (
        fm.Chart(df)
        .mark_text()
        .encode(x="x", y=fm.Y("y", scale=fm.LogScale()), text="label")
        .theme(_NO_GRID)
        .to_svg()
    )
    frame = _frame_rects(svg)[0]
    _, ey = _mark_anchor(svg, "EEEE", frame=frame)  # y=10000, the max
    assert ey == _tick_pixel(svg, axis="y", label="10000", frame=frame)
    _, cy = _mark_anchor(svg, "CCCC", frame=frame)  # y=1, the min
    assert cy == _tick_pixel(svg, axis="y", label="1", frame=frame)
    clearances = _label_clearances(svg, {"EEEE": 0, "CCCC": 0})
    assert clearances["EEEE"]["top"] >= TEXT_INSET_FLOOR_PX
    assert clearances["CCCC"]["bottom"] >= TEXT_INSET_FLOOR_PX


def test_tick_pixel_equals_mark_pixel_with_a_default_time_scale():
    df = pl.DataFrame(
        {
            "t": [
                dt.date(2024, 1, 1),
                dt.date(2024, 1, 2),
                dt.date(2024, 1, 3),
                dt.date(2024, 1, 4),
                dt.date(2024, 1, 5),
            ],
            "y": [3, 5, 1, 4, 2],
            "label": ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE"],
        }
    )
    svg = (
        fm.Chart(df)
        .mark_text()
        .encode(x=fm.X("t", scale=fm.TimeScale()), y="y", text="label")
        .theme(_NO_GRID)
        .to_svg()
    )
    frame = _frame_rects(svg)[0]
    # The temporal x axis's tick labels are date strings, not plain
    # numbers, so the x floor is checked by clearance rather than a tick
    # label match; the y axis (plain numeric) still gets a tick match.
    _, by = _mark_anchor(svg, "BBBB", frame=frame)
    assert by == _tick_pixel(svg, axis="y", label="5", frame=frame)
    _assert_labels_clear(svg, _corners_at(0))


# ---------------------------------------------------------------------------
# Case: an hconcat with a shared x reads the union of the text rows, even
# on a leaf that draws no text of its own.
# ---------------------------------------------------------------------------


def test_hconcat_shared_x_reads_the_union_of_the_text_rows():
    text_df = _corners_df()
    point_df = pl.DataFrame({"x": [2, 3, 4], "y": [1, 2, 3]})
    text_chart = fm.Chart(text_df).mark_text().encode(x="x", y="y", text="label")
    point_chart = fm.Chart(point_df).mark_point().encode(x="x", y="y")
    hc = fm.HConcatChart([text_chart, point_chart], resolve={"x": "shared"}).theme(_NO_GRID)
    svg = hc.to_svg()
    frames = _frame_rects(svg)
    assert len(frames) == 2
    frame_text, frame_points = frames

    tick_text = _tick_pixel(svg, axis="x", label="1", frame=frame_text)
    ax, _ = _mark_anchor(svg, "AAAA", frame=frame_text)
    assert ax == tick_text

    # The point-only leaf draws no text at all, but its x tick for value 1
    # has to sit at the SAME offset from its own frame as the text leaf's
    # does -- the shared demand (built from the text leaf's rows) reaching
    # a leaf that contributes none of its own.
    tick_points = _tick_pixel(svg, axis="x", label="1", frame=frame_points)
    assert _same_offset(tick_text - frame_text[0], tick_points - frame_points[0])


# ---------------------------------------------------------------------------
# Case: every end clears the floor for every align/baseline/angle/dx/dy
# variant, not only the end that variant happens to push toward.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mark_kwargs",
    [
        {},
        {"baseline": "middle"},
        {"baseline": "top"},
        {"baseline": "bottom"},
        {"align": "left"},
        {"align": "right"},
        {"angle": 90},
        {"dx": 6},
        {"dx": -6},
        {"dy": 6},
        {"dy": -6},
    ],
    ids=[
        "default",
        "baseline=middle",
        "baseline=top",
        "baseline=bottom",
        "align=left",
        "align=right",
        "angle=90",
        "dx=+6",
        "dx=-6",
        "dy=+6",
        "dy=-6",
    ],
)
def test_every_variant_clears_all_four_ends(mark_kwargs):
    svg = _corners_chart(**mark_kwargs).to_svg()
    _assert_labels_clear(svg, _corners_at(0))


# ---------------------------------------------------------------------------
# Case: dx/dy are folded into the shared inset, not ignored.
# ---------------------------------------------------------------------------


def test_dx_toward_the_left_edge_grows_the_shared_x_inset():
    # A negative dx on every row pushes the label left, toward the frame's
    # left edge; the x-channel's shared tick/mark position must move right
    # to keep the leftmost glyph's ink clear of the frame -- a fixed,
    # dx-blind floor would leave the tick exactly where it was at dx=0.
    svg_flat = _corners_chart().to_svg()
    svg_pushed = _corners_chart(dx=-6).to_svg()
    tick_flat = _tick_pixel(svg_flat, axis="x", label="1")
    tick_pushed = _tick_pixel(svg_pushed, axis="x", label="1")
    assert tick_pushed > tick_flat
    _assert_labels_clear(svg_pushed, _corners_at(0))


def test_dy_toward_the_top_edge_grows_the_shared_y_inset():
    # BBBB (y=5) sits topmost; a negative dy pushes it up, toward the top
    # edge, so the shared y tick for value 5 must move down to compensate.
    svg_flat = _corners_chart().to_svg()
    svg_pushed = _corners_chart(dy=-6).to_svg()
    tick_flat = _tick_pixel(svg_flat, axis="y", label="5")
    tick_pushed = _tick_pixel(svg_pushed, axis="y", label="5")
    assert tick_pushed > tick_flat
    _assert_labels_clear(svg_pushed, _corners_at(0))


# ---------------------------------------------------------------------------
# Case: a row's requirement is not limited to rows exactly at the domain
# end -- a wide label just inside the end still has to clear the frame,
# and a domain widened by a sibling layer still measures each row at its
# own (now interior) position.
# ---------------------------------------------------------------------------


def test_a_long_label_near_but_not_at_the_end_still_clears_the_frame():
    df = pl.DataFrame({"x": [1.0, 1.2, 5.0], "label": ["a", "Revenue growth year over year", "b"]})
    chart = fm.Chart(df).mark_text().encode(x="x", y="x", text="label").theme(_NO_GRID)
    svg = chart.to_svg()
    clearances = _label_clearances(svg, {"Revenue growth year over year": 0})
    assert clearances["Revenue growth year over year"]["left"] >= TEXT_INSET_FLOOR_PX
    assert clearances["Revenue growth year over year"]["bottom"] >= TEXT_INSET_FLOOR_PX


def test_clearance_holds_when_a_sibling_layer_widens_the_domain():
    # The text rows span x=[1, 5]; a sibling point layer at x=[0.7, 5.3]
    # widens the domain so no text row sits exactly on the boundary.
    text_df = _corners_df()
    point_df = pl.DataFrame({"x": [0.7, 5.3], "y": [0.7, 5.3]})
    text_layer = fm.Chart(text_df).mark_text().encode(x="x", y="y", text="label")
    point_layer = fm.Chart(point_df).mark_point().encode(x="x", y="y")
    svg = fm.LayerChart(text_layer, point_layer).theme(_NO_GRID).to_svg()
    _assert_labels_clear(svg, _corners_at(0))


# ---------------------------------------------------------------------------
# Case: the measured glyph is the drawn glyph -- format, size and angle
# encodings all reach the measurement.
# ---------------------------------------------------------------------------


def test_a_formatted_text_channel_measures_the_formatted_string():
    df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
    chart = (
        fm.Chart(df)
        .mark_text()
        .encode(x="x", y="x", text=fm.Text("x", format=".6f"))
        .theme(_NO_GRID)
    )
    svg = chart.to_svg()
    clearances = _label_clearances(svg, {"5.000000": 0})
    assert clearances["5.000000"]["right"] >= TEXT_INSET_FLOOR_PX
    assert clearances["5.000000"]["top"] >= TEXT_INSET_FLOOR_PX


def test_a_size_encoded_label_measures_its_own_row_size():
    # A kerned ten-character label at a much larger size than its
    # siblings: the measured glyph must be the one actually drawn at its
    # own row's size, not the chart's base font size.
    df = pl.DataFrame({"x": [1, 2, 3], "label": ["a", "b", "WWWWWWWWWW"], "sz": [11, 11, 40]})
    chart = fm.Chart(df).mark_text().encode(x="x", y="x", text="label", size="sz").theme(_NO_GRID)
    svg = chart.to_svg()
    clearances = _label_clearances(svg, {"WWWWWWWWWW": 0})
    assert clearances["WWWWWWWWWW"]["right"] >= TEXT_INSET_FLOOR_PX
    assert clearances["WWWWWWWWWW"]["top"] >= TEXT_INSET_FLOOR_PX


def test_an_angle_encoded_label_measures_its_own_row_angle():
    df = pl.DataFrame({"x": [1, 2, 3], "label": ["a", "b", "cccc"], "ang": [0, 0, 90]})
    chart = fm.Chart(df).mark_text().encode(x="x", y="x", text="label", angle="ang").theme(_NO_GRID)
    svg = chart.to_svg()
    clearances = _label_clearances(svg, {"cccc": 0})
    assert clearances["cccc"]["right"] >= TEXT_INSET_FLOOR_PX
    assert clearances["cccc"]["top"] >= TEXT_INSET_FLOOR_PX


# ---------------------------------------------------------------------------
# Case: polar text is unaffected -- a Cartesian plot-frame inset has
# nothing to apply to under CoordPolar, so the render is pinned bytewise.
# ---------------------------------------------------------------------------


def test_polar_text_renders_its_pinned_bytes():
    df = pl.DataFrame({"g": ["a", "b", "c"], "z": [3, 5, 2]})
    chart = (
        fm.Chart(df)
        .mark_bar()
        .encode(theta="g", radius="z")
        .coord(fm.CoordPolar())
        .layer(fm.Layer(mark="text", encoding={"text": "g"}))
    )
    svg = chart.to_svg()
    assert _digest(svg) == "13cf1863d31b26416a33ee517afaa4aa4a317f23ab612d902d194ce68fbac041"


# ---------------------------------------------------------------------------
# Case: a label too wide for the panel to hold warns exactly once per
# channel, counting every overrunning row once regardless of how many
# panels or composite leaves resolve that channel.
# ---------------------------------------------------------------------------

_TEXT_INK_WARNING = re.compile(
    r"(\d+) mark_text rows? on the (\w+) axis still cross the plot frame"
)


def _text_ink_warnings(caught: list[warnings.WarningMessage]) -> list[tuple[int, str]]:
    """Each ``TextInkPastFrame`` warning caught, as ``(row_count, channel)``."""
    out: list[tuple[int, str]] = []
    for w in caught:
        match = _TEXT_INK_WARNING.search(str(w.message))
        if match:
            out.append((int(match.group(1)), match.group(2)))
    return out


def _overrun_counts(chart: fm.Chart) -> list[tuple[int, str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chart.to_svg()
    return _text_ink_warnings(caught)


def test_an_overwide_label_warns_once_per_channel_on_a_flat_chart():
    df = pl.DataFrame(
        {
            "x": [1, 2, 3],
            "y": [10, 20, 30],
            "label": ["a", "b", "THIS LABEL IS VERY VERY VERY LONG INDEED YES"],
        }
    )
    chart = (
        fm.Chart(df)
        .mark_text()
        .encode(x="x", y="y", text="label")
        .properties(width=120, height=400)
    )
    assert _overrun_counts(chart) == [(2, "x")]


def test_an_overwide_label_warns_once_per_channel_on_a_two_panel_facet():
    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 1, 2, 3],
            "y": [10, 20, 30, 10, 20, 30],
            "label": [
                "a",
                "b",
                "THIS LABEL IS VERY VERY VERY LONG INDEED YES",
                "a",
                "b",
                "ANOTHER QUITE LONG LABEL HERE TOO",
            ],
            "g": ["p1", "p1", "p1", "p2", "p2", "p2"],
        }
    )
    chart = (
        fm.Chart(df)
        .mark_text()
        .encode(x="x", y="y", text="label", facet_col="g")
        .properties(width=250, height=400)
    )
    assert _overrun_counts(chart) == [(4, "x")]


def test_an_overwide_label_warns_once_per_channel_on_independent_facets():
    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 1, 2, 3],
            "y": [10, 20, 30, 10, 20, 30],
            "label": [
                "a",
                "b",
                "THIS LABEL IS VERY VERY VERY LONG INDEED YES",
                "a",
                "b",
                "ANOTHER QUITE LONG LABEL HERE TOO",
            ],
            "g": ["p1", "p1", "p1", "p2", "p2", "p2"],
        }
    )
    chart = (
        fm.Chart(df)
        .mark_text()
        .encode(x="x", y="y", text="label", facet_col="g")
        .share_scale(x="independent")
        .properties(width=250, height=400)
    )
    assert _overrun_counts(chart) == [(4, "x")]


def test_an_overwide_label_warns_once_per_channel_on_a_layerchart_text_and_points_overlay():
    df = pl.DataFrame(
        {
            "x": [1, 2, 3],
            "y": [10, 20, 30],
            "label": ["a", "b", "THIS LABEL IS VERY VERY VERY LONG INDEED YES"],
        }
    )
    text_layer = fm.Chart(df).mark_text().encode(x="x", y="y", text="label")
    point_layer = fm.Chart(df).mark_point().encode(x="x", y="y")
    chart = fm.LayerChart(text_layer, point_layer).properties(width=120, height=400)
    assert _overrun_counts(chart) == [(2, "x")]


def test_an_overwide_label_warns_once_per_channel_on_a_layerchart_text_and_text_overlay():
    df = pl.DataFrame(
        {
            "x": [1, 2, 3],
            "y": [10, 20, 30],
            "label": ["a", "b", "THIS LABEL IS VERY VERY VERY LONG INDEED YES"],
        }
    )
    text_layer = fm.Chart(df).mark_text().encode(x="x", y="y", text="label")
    other_text_layer = fm.Chart(df).mark_text().encode(x="x", y="y", text="label")
    chart = fm.LayerChart(text_layer, other_text_layer).properties(width=120, height=400)
    assert _overrun_counts(chart) == [(4, "x")]


def test_an_overwide_label_warns_once_per_channel_on_an_hconcat_shared_x():
    df = pl.DataFrame(
        {
            "x": [1, 2, 3],
            "y": [10, 20, 30],
            "label": ["a", "b", "THIS LABEL IS VERY VERY VERY LONG INDEED YES"],
        }
    )
    point_df = pl.DataFrame({"x": [2, 3, 4], "y": [1, 2, 3]})
    text_chart = fm.Chart(df).mark_text().encode(x="x", y="y", text="label")
    point_chart = fm.Chart(point_df).mark_point().encode(x="x", y="y")
    hc = fm.HConcatChart([text_chart, point_chart], resolve={"x": "shared"}).properties(
        width=120, height=400
    )
    assert _overrun_counts(hc) == [(2, "x")]


def test_an_overwide_label_warns_once_per_channel_on_coord_flip():
    df = pl.DataFrame(
        {
            "x": [1, 2, 3],
            "y": [10, 20, 30],
            "label": ["a", "b", "THIS LABEL IS VERY VERY VERY LONG INDEED YES"],
        }
    )
    chart = (
        fm.Chart(df)
        .mark_text()
        .encode(x="x", y="y", text="label")
        .coord(fm.CoordFlip())
        .properties(width=120, height=400)
    )
    # Under CoordFlip the warning names the channel the caller actually
    # wrote (y), not the physical axis the overrun shows up on.
    assert _overrun_counts(chart) == [(2, "y")]


def test_an_overwide_label_warns_once_per_channel_on_a_flipped_facet_composite():
    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 1, 2, 3],
            "y": [10, 20, 30, 10, 20, 30],
            "label": [
                "a",
                "b",
                "THIS LABEL IS VERY VERY VERY LONG INDEED YES",
                "a",
                "b",
                "ANOTHER QUITE LONG LABEL HERE TOO",
            ],
            "g": ["p1", "p1", "p1", "p2", "p2", "p2"],
        }
    )
    chart = (
        fm.Chart(df)
        .mark_text()
        .encode(x="x", y="y", text="label", facet_col="g")
        .coord(fm.CoordFlip())
        .properties(width=250, height=400)
    )
    assert _overrun_counts(chart) == [(4, "y")]


def test_independent_y_facet_primary_axis_is_unaffected_by_a_secondary_texts_rows():
    """An independent-y facet's primary y axis reads only primary-slot rows.

    A ``SecondaryY`` text layer's own wide label must not widen the
    primary y inset it never draws on: the primary point's own pixel is
    identical whether the facet's y resolves ``shared`` (the default) or
    ``independent``.
    """
    df = pl.DataFrame(
        {
            "g": ["a", "a"],
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
            "y2": [0.0, 1.0],
            "label": ["x", "W" * 80],
        }
    )
    base = fm.Chart(df).mark_point().encode(x="x", y="y")
    # Mirrors the fields chart._desugar_secondary_y sets on the layer it
    # appends for `chart + SecondaryY(...)` (src/ferrum/chart.py's `_desugar_secondary_y`):
    # SecondaryY itself takes no text= kwarg, so this builds the same
    # layer shape by hand to give it its own text binding.
    secondary = _Layer(
        mark="text",
        encoding={"y": "y2", "text": "label"},
        independent_y=True,
        positional_source="written",
        chart_data=True,
    )
    chart = base.layer(secondary).facet("g").theme(_NO_GRID)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        svg_shared = chart.to_svg()
        svg_independent = chart.share_scale(y="independent").to_svg()
    shared_points = sorted(re.findall(r'<circle cx="([\d.]+)" cy="([\d.]+)"', svg_shared))
    independent_points = sorted(re.findall(r'<circle cx="([\d.]+)" cy="([\d.]+)"', svg_independent))
    assert shared_points, "no primary points rendered"
    assert shared_points == independent_points


def test_faceted_secondary_y_text_overrun_warns_with_the_unfaceted_row_count():
    """A secondary y slot's overrun count does not multiply by panel count.

    The slot's text demand is built once, chart-wide; a facet
    partitioning the SAME rows across panels reports the identical row
    count the unfaceted chart does, not that count doubled by the panel
    count.
    """
    df = pl.DataFrame(
        {
            "g": ["a", "b"],
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
            "y2": [0.0, 1.0],
            "label": ["W" * 10, "W" * 10],
        }
    )
    base = fm.Chart(df).mark_point().encode(x="x", y="y")
    # Mirrors the fields chart._desugar_secondary_y sets on the layer it
    # appends for `chart + SecondaryY(...)` (src/ferrum/chart.py's `_desugar_secondary_y`):
    # SecondaryY itself takes no text= kwarg, so this builds the same
    # layer shape by hand to give it its own text binding.
    secondary = _Layer(
        mark="text",
        encoding={"y": "y2", "text": "label"},
        mark_kwargs={"font_size": 1000},
        independent_y=True,
        positional_source="written",
        chart_data=True,
    )
    faceted = base.layer(secondary).facet("g").theme(_NO_GRID)
    unfaceted = base.layer(secondary).theme(_NO_GRID)

    def y_overrun_counts(chart: fm.Chart) -> list[int]:
        return [count for count, channel in _overrun_counts(chart) if channel == "y"]

    faceted_counts = y_overrun_counts(faceted)
    unfaceted_counts = y_overrun_counts(unfaceted)
    assert len(faceted_counts) == 1, faceted_counts
    assert len(unfaceted_counts) == 1, unfaceted_counts
    assert faceted_counts == unfaceted_counts == [2]


# ---------------------------------------------------------------------------
# Case: an explicit padding=0 wins outright -- no floor is added.
# ---------------------------------------------------------------------------


def test_explicit_padding_zero_places_the_anchor_exactly_on_the_edge():
    df = _corners_df()
    chart = (
        fm.Chart(df)
        .mark_text()
        .encode(
            x=fm.X("x", scale=fm.LinearScale(padding=0.0)),
            y=fm.Y("y", scale=fm.LinearScale(padding=0.0)),
            text="label",
        )
        .theme(_NO_GRID)
    )
    svg = chart.to_svg()
    frame = _frame_rects(svg)[0]
    glyph_a_x, _ = _mark_anchor(svg, "AAAA", frame=frame)
    assert glyph_a_x == frame[0]


# ---------------------------------------------------------------------------
# Case: NaN/inf viewports terminate.
# ---------------------------------------------------------------------------


def test_non_finite_explicit_domain_is_refused_not_hung():
    df = _corners_df()
    with pytest.raises(ValueError, match="finite"):
        fm.Chart(df).mark_text().encode(
            x=fm.X("x", scale=fm.LinearScale(domain=[float("nan"), 5])), y="y", text="label"
        )


def test_all_nan_inferred_domain_renders_without_hanging():
    df = pl.DataFrame(
        {
            "x": [float("nan")] * 5,
            "y": [3, 5, 1, 4, 2],
            "label": ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE"],
        }
    )
    svg = fm.Chart(df).mark_text().encode(x="x", y="y", text="label").to_svg()
    assert len(svg) > 0


# ---------------------------------------------------------------------------
# Byte-identical controls: cases the inset never reaches. Every digest
# below was confirmed, by rendering, to be identical whether or not the
# text-inset floor applies to the chart that produces it.
# ---------------------------------------------------------------------------

_y_neutral = fm.Y("v", scale=fm.LinearScale(padding=0.0))


def _band_bar_labels() -> fm.Chart:
    # The y channel's padding is pinned explicitly so a real (and correct)
    # change to the continuous y axis's own text clearance can never move
    # this control; the claim under test is the categorical x (band) axis
    # alone, which a `BandScale` always yields an empty text demand for.
    df = pl.DataFrame({"g": ["a", "b", "c"], "v": [3, 7, 5]})
    return (
        fm.Chart(df)
        .mark_bar()
        .encode(x="g", y=_y_neutral)
        .layer(fm.Layer(mark="text", encoding={"text": fm.Text("v")}))
    )


def _point_scale_labels() -> fm.Chart:
    df = pl.DataFrame({"g": ["a", "b", "c"], "v": [3, 7, 5]})
    return (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("g", scale=fm.PointScale()), y=_y_neutral)
        .layer(fm.Layer(mark="text", encoding={"text": fm.Text("v")}))
    )


def _mark_label_chart() -> fm.Chart:
    # Default padding on both channels: mark_label is not Mark::Text and
    # contributes no rows to the text demand, so this control exercises
    # the real default-padding path end to end, not an explicit padding=0
    # that would mask the claim regardless of whether mark_label is
    # actually exempt.
    df = _corners_df()
    return (
        fm.Chart(df)
        .mark_point()
        .encode(x="x", y="y")
        .layer(fm.Layer(mark="label", encoding={"text": fm.Text("label")}))
    )


def _heatmap_annotations() -> fm.Chart:
    df = pl.DataFrame({"row": ["r1", "r2"], "c1": [1.0, 3.0], "c2": [2.0, 4.0]})
    return fm.heatmap(df, annot=True)


def _no_text_point_chart() -> fm.Chart:
    df = pl.DataFrame({"x": [1, 2, 3, 4, 5], "y": [3, 5, 1, 4, 2]})
    return fm.Chart(df).mark_point().encode(x="x", y="y")


_CONTROLS = {
    "band_bar_labels": _band_bar_labels,
    "point_scale_labels": _point_scale_labels,
    "mark_label": _mark_label_chart,
    "heatmap_annotations": _heatmap_annotations,
    "no_text_point": _no_text_point_chart,
}

_CONTROL_DIGESTS = {
    "band_bar_labels": "f2410ee5462b0a808ab830a47c4b9103ab51a749dc7464a528784c1c0ff41cbb",
    "point_scale_labels": "d6e2fc55245513c9f3251fb97755d2f077dc85a0a592d8b5578fc6612ef893b5",
    "mark_label": "112140d398c107b26680b203c7a97035d1194c4ed1d06f51cbb994928d9b7e82",
    "heatmap_annotations": "6e132212abe40f5186ab6d2c410d1533c484df392d50855614836d546ab09ca8",
    "no_text_point": "b852c5ac3f1ba12265121a3d88295e54a9dcc60f12b13ee02ee31ae052937c05",
}


@pytest.mark.parametrize("case", sorted(_CONTROLS))
def test_a_control_renders_its_pinned_bytes(case):
    svg = _CONTROLS[case]().to_svg()
    assert _digest(svg) == _CONTROL_DIGESTS[case]


def test_mark_text_moves_under_the_same_scale_change_mark_label_does_not():
    # What tells "exempt" and "measured" apart: the same chart with
    # mark="text" instead of mark="label" moves when the text demand
    # reaches it, so the mark_label control's byte-identity above is a
    # real claim about the mark, not a chart that never moves at all.
    df = _corners_df()
    label_chart = _mark_label_chart()
    text_chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x", y="y")
        .layer(fm.Layer(mark="text", encoding={"text": fm.Text("label")}))
    )
    assert _digest(label_chart.to_svg()) == _CONTROL_DIGESTS["mark_label"]
    assert label_chart.to_svg() != text_chart.to_svg()


def test_explicit_padding_is_byte_identical_to_the_pinned_control():
    df = _corners_df()
    chart = (
        fm.Chart(df)
        .mark_text()
        .encode(
            x=fm.X("x", scale=fm.LinearScale(padding=0.0)),
            y=fm.Y("y", scale=fm.LinearScale(padding=0.0)),
            text="label",
        )
        .theme(_NO_GRID)
    )
    assert (
        _digest(chart.to_svg())
        == "8ddf986657d5e65f4c1fd295ca2aad4cfd69833040ea20677f82018262d9f67e"
    )


def test_explicit_domain_is_byte_identical_to_the_pinned_control():
    df = _corners_df()
    chart = (
        fm.Chart(df)
        .mark_text()
        .encode(
            x=fm.X("x", scale=fm.LinearScale(domain=[0, 6])),
            y=fm.Y("y", scale=fm.LinearScale(domain=[0, 6])),
            text="label",
        )
        .theme(_NO_GRID)
    )
    assert (
        _digest(chart.to_svg())
        == "65eb8e3c15be5c3b511f9f62a2271822ab7d559d2d3df42105e85535c89f4bb2"
    )


def test_explicit_padding_point_05_is_byte_identical_to_the_pinned_control():
    df = _corners_df()
    chart = (
        fm.Chart(df)
        .mark_text()
        .encode(
            x=fm.X("x", scale=fm.LinearScale(padding=0.05)),
            y=fm.Y("y", scale=fm.LinearScale(padding=0.05)),
            text="label",
        )
        .theme(_NO_GRID)
    )
    assert (
        _digest(chart.to_svg())
        == "58e8ea96dc36336330668f0904a6ac1f46d2ddba25ab6c6bd404f41d0e5504a2"
    )


def test_explicit_range_is_byte_identical_to_the_pinned_control():
    df = _corners_df()
    chart = (
        fm.Chart(df)
        .mark_text()
        .encode(
            x=fm.X("x", scale=fm.LinearScale(range=[0, 500])),
            y=fm.Y("y", scale=fm.LinearScale(range=[0, 400])),
            text="label",
        )
        .theme(_NO_GRID)
    )
    assert (
        _digest(chart.to_svg())
        == "4f97624a9fa8f44efeb6e65ca6f57efa5f178a5c2461f12883dfd5d1d364453f"
    )


def test_coord_cartesian_xlim_ylim_is_byte_identical_to_the_pinned_control():
    df = _corners_df()
    chart = (
        fm.Chart(df)
        .mark_text()
        .encode(x="x", y="y", text="label")
        .coord(fm.CoordCartesian(xlim=(0, 6), ylim=(0, 6)))
        .theme(_NO_GRID)
    )
    assert (
        _digest(chart.to_svg())
        == "b5e4dcf87509536e39038fd41ebe68e0a92bebb581cc476fd4e91b39538e8616"
    )
