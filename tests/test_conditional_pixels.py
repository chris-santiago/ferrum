"""Pixel-level proof that WASM conditional encodings repaint the channel they name.

Rows are derived from :data:`ferrum.encoding._channel_policy._CONDITIONAL_CHANNELS`
(never hand-listed) crossed with mark kind, packing, value form, and every
interaction driver — the full product, no rotation. Each row drives a real
DOM event through an exported ``.interactive().save()`` chart, samples pixels
in the resulting screenshot, and asserts both that the conditioned property
changed as predicted and that the properties the channel does not name
stayed put (property isolation, checked against an unselected neighbor mark).

Skipped whole-module when the system Chrome channel Playwright needs is
unavailable — checked lazily inside the ``browser`` fixture, never at
collection time. Every failure names the committed bundle under test
(``src/ferrum/_wasm/ferrum_wasm_bg.wasm``) since a conditional, shader, or
scene-load change needs ``wasm-pack build`` before this harness can see it.
"""

from __future__ import annotations

import atexit
import functools
import itertools
import math
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import polars as pl
import pytest

import ferrum as fm
import ferrum._core as _core
from ferrum import value
from ferrum.encoding._channel_policy import _CONDITIONAL_CHANNELS
from ferrum.marks.base import _LINETYPE_MAP

from tests import _pixel_capture as pc
from tests._conditional_oracle import (
    assert_radius_close,
    assert_rgb_close,
    classify_ring,
    predicted_fill_edge,
    ring_is_alternating,
    ring_is_solid,
    ring_sample_count,
    rgb_distance3,
)

pytestmark = pytest.mark.wasm_pixels

_BUNDLE_PATH = "src/ferrum/_wasm/ferrum_wasm_bg.wasm"

BASE_FILL = "#2563eb"
BASE_STROKE = "#111827"
BASE_STROKE_W = 3.0
DASH_STROKE_W = 6.0
BASE_MARK_SIZE = 452.0

MarkKind = Literal["point", "rect"]
Packing = Literal["unpacked", "packed"]
Driver = Literal["click", "legend", "brush"]

_MARKS: tuple[MarkKind, ...] = ("point", "rect")
_PACKINGS: tuple[Packing, ...] = ("unpacked", "packed")
_DRIVERS: tuple[Driver, ...] = ("click", "legend", "brush")

# Value forms, derived per channel from the PoC's discriminating pair
# (a named linetype vs a palette pattern vs a non-palette pattern) and the
# decision doc's stated population ("color hex and name; stroke_dash named,
# palette, non-palette"). Every other channel gets one representative
# numeric literal.
_FORMS: dict[str, tuple[tuple[str, Any], ...]] = {
    "color": (("hex", "#dc2626"), ("name", "red")),
    "size": (("num", None),),  # per-mark value; see _SIZE_VALUES
    "opacity": (("num", 0.35),),
    "stroke_width": (("num", 8.0),),
    "stroke_opacity": (("num", 0.3),),
    "fill_opacity": (("num", 0.3),),
    "angle": (("num", 40.0),),
    "stroke_dash": (("named", "dashed"), ("palette", [6, 3]), ("nonpalette", [7, 2, 1, 2])),
}

# The "if_not" (unselected) literal for each channel's conditional. Once
# any selection exists, every unselected mark (including the isolation
# probe's neighbor mark, which is never the click/legend/brush target)
# re-evaluates its conditional against `if_not` — a resting-vs-selected
# state change, not a bug (confirmed directly: a `color` conditional whose
# `if_not` differs from the style default visibly repaints every
# unselected mark, including the isolation neighbor, the instant the first
# selection is made). Every baseline here equals the channel's own
# style/internal default so an unselected mark's `if_not` appearance is
# indistinguishable from its resting appearance, and the isolation check
# below is testing what it claims to test rather than this expected
# resting-to-if_not transition.
_BASELINE: dict[str, Any] = {
    "color": BASE_FILL,
    "size": None,  # per-mark value; see _SIZE_VALUES
    "opacity": 1.0,
    "stroke_width": 0.0,
    "stroke_opacity": 1.0,
    "fill_opacity": 1.0,
    "angle": 0.0,
    "stroke_dash": "solid",
}

# `size` is mark-dependent, not a single shared literal: `EncodingValue::Size`
# sets a circle's radius to `sqrt(value / pi)` (an area, matching the style
# `size=` convention) but sets a rect's raw `width`/`height` directly to
# `value` pixels (crates/ferrum-core/src/render/pack_instances.rs's
# `RectInstance` layout; `position` is the rect's fixed top-left corner, so a
# raw-pixel point mark's own literal (900) would place the rect's right edge
# 900px from its own left edge, off any reasonably sized canvas). Rect's
# values are small raw pixel counts instead.
_SIZE_VALUES: dict[MarkKind, dict[str, float]] = {
    "point": {"baseline": 400.0, "selected": 900.0},
    # Small enough that the selected extent (minus the stroke inset) stays
    # within one packed-grid cell's own spacing (~28-30 SVG px): a bigger
    # selected width would visually cross into a neighboring packed rect's
    # own stroke ring, which the edge-scan below cannot distinguish from
    # the target's own far edge.
    "rect": {"baseline": 15.0, "selected": 25.0},
}


def _size_value(mark: MarkKind, kind: str) -> float:
    return _SIZE_VALUES[mark][kind]


@dataclass(frozen=True)
class Row:
    """One generated pixel-harness test case."""

    channel: str
    mark: MarkKind
    packing: Packing
    form_id: str
    form_value: Any
    driver: Driver

    @property
    def id(self) -> str:
        return f"{self.channel}-{self.form_id}-{self.mark}-{self.packing}-{self.driver}"


def _generate_rows() -> list[Row]:
    """Derive every row from ``_CONDITIONAL_CHANNELS`` x mark x packing x driver x form.

    The full product, no rotation: every channel gets every (mark, packing,
    driver) cell for every one of its value forms. Not hand-listed: a
    channel absent from ``_CONDITIONAL_CHANNELS`` produces no rows, which
    ``test_coverage_covers_every_conditional_channel`` catches.
    """
    rows: list[Row] = []
    for channel in sorted(_CONDITIONAL_CHANNELS):
        for form_id, form_value in _FORMS[channel]:
            for mark in _MARKS:
                form_value_for_mark = (
                    _size_value(mark, "selected") if channel == "size" else form_value
                )
                for packing in _PACKINGS:
                    for driver in _DRIVERS:
                        rows.append(
                            Row(channel, mark, packing, form_id, form_value_for_mark, driver)
                        )
    return rows


ROWS: list[Row] = _generate_rows()

# Packed-grid layout shared by every packing="packed" fixture: a 50x24 grid
# (1200 marks) spaced widely enough that a mark's own fill never touches its
# neighbor's, verified empirically against the committed bundle's actual
# instance geometry.
_PACKED_N = 1200
_PACKED_COLS = 50
_PACKED_TARGET_IDX = 625
_PACKED_SIZE = (1500.0, 900.0)
_UNPACKED_SIZE = (640.0, 480.0)
_NEIGHBOR_IDX = 0  # always a different category from either target index


@dataclass
class Fixture:
    """A built chart plus everything the oracle needs to probe it."""

    svg: str
    svg_w: float
    svg_h: float
    html_path: Path
    target: tuple[float, ...]  # circle (cx, cy, r) or rect (x, y, w, h)
    target_idx: int
    target_category: str
    neighbor_center: tuple[float, float]
    viewport: dict[str, float]


def _packed_categories(n: int) -> list[str]:
    letters = ["a", "b", "c", "d"]
    return [letters[i % len(letters)] for i in range(n)]


def _point_data(packing: Packing) -> tuple[pl.DataFrame, int]:
    if packing == "unpacked":
        df = pl.DataFrame(
            {"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.0, 4.0, 1.0, 5.0, 3.0], "k": list("abcde")}
        )
        return df, 1
    xs = [float(i % _PACKED_COLS) for i in range(_PACKED_N)]
    ys = [float(i // _PACKED_COLS) for i in range(_PACKED_N)]
    df = pl.DataFrame({"x": xs, "y": ys, "k": _packed_categories(_PACKED_N)})
    return df, _PACKED_TARGET_IDX


def _rect_data(packing: Packing) -> tuple[pl.DataFrame, int]:
    if packing == "unpacked":
        xs = [0.0, 1.0, 2.0, 3.0, 4.0]
        df = pl.DataFrame(
            {
                "x": xs,
                "x2": [x + 0.8 for x in xs],
                "y": [0.0] * 5,
                "y2": [1.0] * 5,
                "k": list("abcde"),
            }
        )
        return df, 1
    xs = [float(i % _PACKED_COLS) for i in range(_PACKED_N)]
    ys = [float(i // _PACKED_COLS) for i in range(_PACKED_N)]
    df = pl.DataFrame(
        {
            "x": xs,
            "x2": [x + 0.8 for x in xs],
            "y": ys,
            "y2": [y + 0.8 for y in ys],
            "k": _packed_categories(_PACKED_N),
        }
    )
    return df, _PACKED_TARGET_IDX


def _mark_style(mark: MarkKind, channel: str, tested_stroke_w: float) -> dict[str, Any]:
    """Fixed (never-conditioned) style kwargs for the mark under test.

    ``fill`` is always set, including for ``color`` rows: the PoC's resting
    appearance is the mark's own style (``#2563eb``), never the conditional's
    ``if_not`` literal — the static SVG path never evaluates a conditional at
    all, so `encode(color=<ConditionalSpec>)` falls back to the style fill
    for its (identical) resting render. Every channel other than the one
    under test keeps a plain, fully-opaque style so its own probe reads
    cleanly: a visible stroke for ring-based checks, no rotation.
    """
    kw: dict[str, Any] = {"fill": BASE_FILL, "stroke": BASE_STROKE}
    if channel != "stroke_width":
        kw["stroke_width"] = tested_stroke_w
    if mark == "point" and channel != "size":
        kw["size"] = BASE_MARK_SIZE
    return kw


def _build_fixture(row: Row, tmp_path: Path, *, suffix: str = "") -> Fixture:
    if row.channel == "stroke_dash":
        tested_stroke_w = DASH_STROKE_W
    elif row.channel == "angle":
        # A thin stroke (rather than BASE_STROKE_W's 3px inset) leaves the
        # rect's corner-probe oracle (`_check_angle_rect`) enough margin to
        # clear both the stroke ring and the anti-aliased edge on the
        # packed grid's much smaller per-cell rects.
        tested_stroke_w = 0.5
    else:
        tested_stroke_w = BASE_STROKE_W
    style = _mark_style(row.mark, row.channel, tested_stroke_w)

    group_field = "k"
    if row.driver == "legend":
        # clear="false": a point selection's default clear="mouseout" would
        # otherwise drop the selection the instant the pointer leaves the
        # legend label (`_pixel_capture.click_at`/`legend_click` always move
        # the pointer off-chart afterward to dodge a stray hover tooltip).
        sel = fm.selection_point(name="s", bind="legend", fields=[group_field], clear="false")
    elif row.driver == "brush":
        sel = fm.selection_interval(name="s")
    else:
        sel = fm.selection_point(name="s", clear="false")

    baseline = (
        _size_value(row.mark, "baseline") if row.channel == "size" else _BASELINE[row.channel]
    )
    # color's own legend driver: the legend needs a field-backed category
    # scale to build its swatches, but `color` is already the tested
    # channel, so there is no room for a second, separate `color=` field
    # encoding (see the general legend branch below). The canonical
    # legend-highlight pattern instead puts the field reference in the
    # conditional's own `otherwise` branch — `fm.Color(k)` alone builds the
    # legend, and the literal `if_selected` branch still applies normally
    # when a category is toggled on. Verified directly: this renders 5
    # legend swatches and the target's own pixel goes from the style fill
    # (#2563eb, at rest) to the exact selected literal (once toggled).
    if row.channel == "color" and row.driver == "legend":
        cond = sel.when(value(row.form_value)).otherwise(fm.Color(group_field))
    else:
        cond = sel.when(value(row.form_value)).otherwise(value(baseline))

    if row.mark == "point":
        df, idx = _point_data(row.packing)
        chart = fm.Chart(df).mark_point(**style)
        encode_kwargs: dict[str, Any] = {"x": "x", "y": "y", "tooltip": "k", row.channel: cond}
        if row.driver == "legend" and row.channel != "color":
            encode_kwargs["color"] = group_field
        chart = chart.encode(**encode_kwargs).add_selection(sel)
    else:
        df, idx = _rect_data(row.packing)
        chart = fm.Chart(df).mark_rect(**style)
        encode_kwargs = {
            "x": "x",
            "x2": "x2",
            "y": "y",
            "y2": "y2",
            "tooltip": "k",
            row.channel: cond,
        }
        if row.driver == "legend" and row.channel != "color":
            encode_kwargs["color"] = group_field
        chart = chart.encode(**encode_kwargs).add_selection(sel)

    width, height = _UNPACKED_SIZE if row.packing == "unpacked" else _PACKED_SIZE
    if row.packing == "packed":
        chart = chart.properties(width=width, height=height)

    svg = chart.to_svg()
    svg_w, svg_h = pc.parse_svg_size(svg)
    if row.mark == "point":
        shapes = pc.parse_circles(svg)
        target = shapes[idx]
        neighbor_center = shapes[_NEIGHBOR_IDX][:2]
    else:
        shapes = pc.parse_rects(svg)
        target = shapes[idx]
        nx, ny, nw, nh = shapes[_NEIGHBOR_IDX]
        if row.channel == "size":
            # A `size` conditional resizes the neighbor too, once any
            # selection exists (an unselected mark still evaluates
            # `if_not`) — anchored at its own fixed top-left corner
            # (`position`), not its static center, so the static center
            # can fall outside the resized rect entirely. A small inset
            # from the corner stays inside the rect at both the baseline
            # and selected sizes.
            neighbor_center = (nx + BASE_STROKE_W + 1.0, ny + BASE_STROKE_W + 1.0)
        else:
            neighbor_center = (nx + nw / 2, ny + nh / 2)

    html_path = tmp_path / f"chart{suffix}.html"
    chart.interactive().save(str(html_path))

    target_category = (
        _packed_categories(_PACKED_N)[idx] if row.packing == "packed" else "abcde"[idx]
    )

    return Fixture(
        svg=svg,
        svg_w=svg_w,
        svg_h=svg_h,
        html_path=html_path,
        target=target,
        target_idx=idx,
        target_category=target_category,
        neighbor_center=neighbor_center,
        viewport={"width": width, "height": height},
    )


def _trigger(page: Any, row: Row, fx: Fixture, rect: pc.CanvasRect) -> None:
    if row.mark == "point":
        cx, cy, _ = fx.target
    else:
        rx, ry, rw, rh = fx.target
        cx, cy = rx + rw / 2, ry + rh / 2

    if row.driver == "click":
        client_pt = pc.svg_to_client(rect, fx.svg_w, fx.svg_h, (cx, cy))
        pc.click_at(page, client_pt)
    elif row.driver == "legend":
        pc.legend_click(page, fx.target_category)
    else:
        # A too-small drag is unreliable (the brush behavior can drop a drag
        # that never moves far from its start), so the pad is generous
        # relative to any mark's own footprint rather than tight around it.
        pad = 40.0
        start = pc.svg_to_client(rect, fx.svg_w, fx.svg_h, (cx - pad, cy - pad))
        end = pc.svg_to_client(rect, fx.svg_w, fx.svg_h, (cx + pad, cy + pad))
        pc.brush_drag(page, start, end)


def _target_fill_rgb(fx: Fixture, idx: int) -> tuple[int, int, int]:
    """Parse the target mark's own fill from the static SVG, in document order.

    The static SVG never evaluates conditionals (only the interactive
    runtime does), so this is the chart's true resting appearance — the
    "before" oracle this harness's before-frame check is anchored to.
    """
    if len(fx.target) == 3:
        fills = re.findall(r"<circle[^>]*\bfill=\"([^\"]+)\"", fx.svg)
    else:
        fills = re.findall(r"<rect(?=[^>]*\bstroke=)[^>]*\bfill=\"([^\"]+)\"", fx.svg)
    r, g, b, _a = _core.parse_color_to_rgba(fills[idx])
    return (r, g, b)


def _center_point(fx: Fixture) -> tuple[float, float]:
    if len(fx.target) == 3:
        return fx.target[0], fx.target[1]
    x, y, w, h = fx.target
    return x + w / 2, y + h / 2


def _ring_point(fx: Fixture, *, inward: float = 2.0) -> tuple[float, float]:
    """A point just inside the mark's outer edge, inside the (inset) stroke ring."""
    if len(fx.target) == 3:
        cx, cy, r = fx.target
        return cx + r - inward, cy
    x, y, w, h = fx.target
    return x + inward, y + h / 2


# ── Per-channel own-effect oracles ──────────────────────────────────────────


def _check_color(page: Any, fx: Fixture, rect: pc.CanvasRect, after: bytes, expected: str) -> None:
    r, g, b, _a = _core.parse_color_to_rgba(expected)
    dev = pc.svg_to_device(rect, fx.svg_w, fx.svg_h, _center_point(fx))
    got = pc.sample_pixels(page, after, [dev])[0]
    assert_rgb_close(got, (r, g, b), tol=0, label="color conditional")


# One reference capture per (channel, mark, packing, form_value, driver,
# probe). This is the "computed expected" for the opacity-family channels.
#
# A hand-derived `a*fg + (1-a)*bg` blend formula was tried in both
# sRGB-numeric and linearized colorspace and measured directly against a
# plain-style-kwarg render: for `opacity=0.35` on fill `#2563eb` over a
# white background, the sRGB-numeric blend predicts (179, 200, 248) and the
# linearized blend predicts (212, 217, 248), while the plain-style-kwarg
# render actually measures (220, 219, 227) — 41 and 21 RGB units off
# respectively, both well past a meaningful ±2 tolerance, so asserting a
# formula-predicted value would be a guaranteed false failure. Neither
# formula matches what this renderer actually produces for a partial-alpha
# instance; this is a candidate renderer-wide compositing discrepancy this
# harness does not paper over by loosening the tolerance.
#
# Rendering the reference through the conditional-apply path itself
# (`sel.when(value(x)).otherwise(value(x))`) was tried and rejected: it is
# self-referential — a wrong alpha, or a value misrouted to the wrong
# instance field identically in both the "expected" and "actual" renders,
# cancels out and the row cannot fail. The reference here is a genuinely
# independent path instead: a plain style kwarg (`opacity=` — the one
# opacity-family channel `mark_point`/`mark_rect` both honor as a literal
# style override) or a plain (non-conditional) constant-column encode
# (`fill_opacity=`/`stroke_opacity=`, both point/rect except rect's
# `stroke_opacity`, which the xfail above already accounts for), with no
# selection at all. `_check_opacity_family` additionally asserts the
# reference differs from the resting (pre-interaction) pixel at the same
# probe — the non-vacuity control: if `form_value` rendered identically to
# resting, or the conditional were a silent no-op leaving `actual ==
# resting`, that would show up as `actual == resting != reference`, a real
# failure, not a pass.
#
# Memoized with `functools.lru_cache` rather than a module-level dict: the
# cache key is the function's own arguments (every field of `row` that
# distinguishes one reference render from another, plus `probe_svg`), so
# there is no separate cache-invalidation surface to keep in sync. Each
# cache miss writes its reference html under one directory shared by every
# call in this process (`_REFERENCE_HTML_DIR`, created once, removed at
# interpreter exit) rather than borrowing whichever test's own `tmp_path`
# happened to miss the cache first — a pytest tmp-path fixture is not part
# of the cache key (pytest's `TempPathFactory` is not hashable), and this
# reference render has no assertions of its own to benefit from pytest's
# per-test cleanup. A per-call `tempfile.mkdtemp` directory would leak one
# unremoved directory per cache miss for the life of the process; one
# directory with a uniquely-named file per miss does not. Created lazily
# (on the first actual cache miss, not at import time): a default pytest
# run deselects every `wasm_pixels` row, and this module is still imported
# for collection, so an eager directory would be created on every worker
# even when no reference render ever happens.
@functools.lru_cache(maxsize=1)
def _reference_html_dir() -> Path:
    html_dir = Path(tempfile.mkdtemp(prefix="ferrum_conditional_ref_"))
    atexit.register(shutil.rmtree, html_dir, ignore_errors=True)
    return html_dir


_REFERENCE_HTML_COUNTER = itertools.count()


@functools.lru_cache(maxsize=None)
def _reference_pixel(
    browser: Any, row: Row, probe_svg: tuple[float, float]
) -> tuple[int, int, int]:
    style = _mark_style(row.mark, row.channel, BASE_STROKE_W)
    ref_col = "_ref"
    if row.channel == "opacity":
        style["opacity"] = row.form_value
        channel_encode: dict[str, Any] = {}
    else:
        channel_encode = {row.channel: ref_col}

    group_field = "k"
    if row.mark == "point":
        df, idx = _point_data(row.packing)
        if channel_encode:
            df = df.with_columns(pl.lit(row.form_value).alias(ref_col))
        chart = fm.Chart(df).mark_point(**style)
        encode_kwargs: dict[str, Any] = {"x": "x", "y": "y", "tooltip": "k", **channel_encode}
    else:
        df, idx = _rect_data(row.packing)
        if channel_encode:
            df = df.with_columns(pl.lit(row.form_value).alias(ref_col))
        chart = fm.Chart(df).mark_rect(**style)
        encode_kwargs = {
            "x": "x",
            "x2": "x2",
            "y": "y",
            "y2": "y2",
            "tooltip": "k",
            **channel_encode,
        }
    if row.driver == "legend":
        encode_kwargs["color"] = group_field
    chart = chart.encode(**encode_kwargs)

    width, height = _UNPACKED_SIZE if row.packing == "unpacked" else _PACKED_SIZE
    if row.packing == "packed":
        chart = chart.properties(width=width, height=height)

    svg_w, svg_h = pc.parse_svg_size(chart.to_svg())
    html_path = _reference_html_dir() / f"chart_{next(_REFERENCE_HTML_COUNTER)}.html"
    chart.interactive().save(str(html_path))

    page = browser.new_page(
        viewport={"width": width, "height": height}, device_scale_factor=pc.DEVICE_SCALE_FACTOR
    )
    try:
        # No selection, no conditional, no interaction: this is a plain
        # render of `form_value`, loaded and screenshotted directly.
        pc.load_interactive(page, html_path)
        rect = pc.canvas_rect(page)
        dev = pc.svg_to_device(rect, svg_w, svg_h, probe_svg)
        return pc.sample_pixels(page, pc.screenshot(page), [dev])[0][:3]
    finally:
        page.close()


def _check_opacity_family(
    browser: Any,
    row: Row,
    fx: Fixture,
    rect: pc.CanvasRect,
    before: bytes,
    after: bytes,
    page: Any,
) -> None:
    probe = _ring_point(fx) if row.channel == "stroke_opacity" else _center_point(fx)
    dev = pc.svg_to_device(rect, fx.svg_w, fx.svg_h, probe)
    resting = pc.sample_pixels(page, before, [dev])[0]
    expected = _reference_pixel(browser, row, probe)
    assert rgb_distance3(expected, resting) > 5, (
        f"{row.channel} conditional: the plain-render reference for form_value="
        f"{row.form_value!r} ({expected}) is indistinguishable from the resting pixel "
        f"({resting}) — this row cannot discriminate a no-op conditional "
        f"(committed bundle: {_BUNDLE_PATH})"
    )
    actual = pc.sample_pixels(page, after, [dev])[0]
    assert_rgb_close(actual, expected, tol=2, label=f"{row.channel} conditional")


def _check_size(page: Any, fx: Fixture, rect: pc.CanvasRect, after: bytes, row: Row) -> None:
    """Assert the after-frame fill edge matches ``predicted_fill_edge`` for ``row.form_value``.

    No before-frame assertion here: the resting (pre-interaction) state
    never evaluates *any* conditional, for every channel (confirmed
    directly: a conditional whose `when` and `otherwise` branches are set
    to the same distinguishable value still renders the mark's own
    style/default at rest, not that value), so a conditional-only `size`
    fixture's resting extent reflects whatever geometry existed before the
    conditional could apply at all — the built-in default point area (36)
    for a circle with no companion field/style literal, or the plain
    `x2`/`y2`-derived width for a rect — not `_size_value(..., "baseline")`.
    That is a real, orthogonal fact about the resting state, not a claim
    this channel's own oracle needs to re-assert; the after-frame check
    below is what proves the conditional applies `row.form_value`
    correctly.

    A circle's anchor (its center) never moves, so the scan starts there. A
    rect's anchor is its own top-left corner (`position`, fixed regardless
    of size), which sits at the *stroke* ring, not inside the fill — the
    scan starts a little past the stroke instead (guaranteed inside fill)
    and the skipped distance is added back onto the measured edge.
    """
    fill_rgb = _target_fill_rgb(fx, fx.target_idx)
    predicted_after = predicted_fill_edge(row.mark, row.form_value, BASE_STROKE_W)
    if row.mark == "point":
        anchor_after = _center_point(fx)
        base_offset = 0.0
    else:
        x, y = fx.target[0], fx.target[1]
        base_offset = BASE_STROKE_W + 1.0
        anchor_after = (x + base_offset, y + row.form_value / 2)
    r_after = base_offset + pc.scan_radius(
        page, after, rect, fx.svg_w, fx.svg_h, anchor_after, fill_rgb, max_r=200
    )
    assert_radius_close(r_after, predicted_after, tol=3.0, label="size conditional")


def _rotated_contains(
    px: float, py: float, cx: float, cy: float, half_w: float, half_h: float, angle_deg: float
) -> bool:
    a = math.radians(-angle_deg)
    dx, dy = px - cx, py - cy
    lx = dx * math.cos(a) - dy * math.sin(a)
    ly = dx * math.sin(a) + dy * math.cos(a)
    return abs(lx) <= half_w and abs(ly) <= half_h


def _check_angle_rect(
    page: Any, fx: Fixture, rect: pc.CanvasRect, before: bytes, after: bytes
) -> None:
    x, y, w, h = fx.target
    cx, cy = x + w / 2, y + h / 2
    half_w, half_h = w / 2, h / 2
    # A fixed absolute inset (rather than a fraction of w/h) keeps the probe
    # clear of the anti-aliased edge blur on a packed grid's much smaller
    # per-cell rects, where an 8%-of-width inset would land inside the blur.
    inset = min(2.0, w * 0.15, h * 0.15)
    probe = (x + w - inset, y + inset)
    baseline_in = _rotated_contains(probe[0], probe[1], cx, cy, half_w, half_h, 0.0)
    rotated_in = _rotated_contains(probe[0], probe[1], cx, cy, half_w, half_h, 40.0)
    assert baseline_in != rotated_in, "angle probe point must differ inside/outside pre-rotation"
    fill_rgb = _target_fill_rgb(fx, fx.target_idx)
    dev = pc.svg_to_device(rect, fx.svg_w, fx.svg_h, probe)
    b_px = pc.sample_pixels(page, before, [dev])[0]
    a_px = pc.sample_pixels(page, after, [dev])[0]
    b_is_fill = rgb_distance3(b_px, fill_rgb) < 30
    a_is_fill = rgb_distance3(a_px, fill_rgb) < 30
    assert b_is_fill == baseline_in, (
        f"angle baseline: corner probe fill-membership {b_is_fill} disagrees with the "
        f"unrotated geometry (committed bundle: {_BUNDLE_PATH})"
    )
    assert a_is_fill == rotated_in, (
        f"angle conditional: corner probe fill-membership {a_is_fill} disagrees with "
        f"the rotated geometry (committed bundle: {_BUNDLE_PATH})"
    )


def _check_angle_point_identical(
    page: Any, fx: Fixture, rect: pc.CanvasRect, before: bytes, after: bytes
) -> None:
    """A plain circle is rotationally symmetric around its own center: rotating it is invisible.

    The only honest oracle is that nothing changed — checked with a small
    per-pixel tolerance (not raw byte equality) across a ring of samples
    straddling the circle's own edge: rotating the vertex quad shifts which
    device pixels the rasterizer covers by a sub-pixel amount even though
    the fragment shader's fill test is itself rotation-invariant, producing
    a few units of genuine anti-aliasing jitter at the boundary that is not
    a content change.
    """
    cx, cy, r = fx.target
    pts = [(cx + r + d, cy) for d in (-4, -2, -1, 0, 1, 2, 4)]
    pts.append((cx, cy))
    dev_pts = [pc.svg_to_device(rect, fx.svg_w, fx.svg_h, p) for p in pts]
    b_samples = pc.sample_pixels(page, before, dev_pts)
    a_samples = pc.sample_pixels(page, after, dev_pts)
    for pt, b_px, a_px in zip(pts, b_samples, a_samples):
        assert rgb_distance3(b_px, a_px) < 12, (
            f"angle conditional on a circle: expected an unchanged frame (a circle is "
            f"rotationally symmetric), pixel at {pt} moved from {b_px} to {a_px} "
            f"(committed bundle: {_BUNDLE_PATH})"
        )


def _ring_perimeter(fx: Fixture, stroke_w: float) -> float:
    """The target's own stroke-ring perimeter, in SVG-space units.

    Circle: the circumference of the (inward-inset) stroke ring. Rect: the
    outer perimeter (the stroke inset only moves each sample a fixed
    distance inward, not the ring's total length). Fed to
    `ring_sample_count` so the sampling density scales with the mark's own
    size rather than a single fixed count.
    """
    if len(fx.target) == 3:
        _cx, _cy, r = fx.target
        return 2 * math.pi * (r - stroke_w / 2)
    _x, _y, w, h = fx.target
    return 2 * (w + h)


def _dash_min_segment(row: Row) -> float | None:
    """The row's own dash pattern's shortest on/off segment, or `None` if it renders solid.

    Only `named`/`palette` forms actually alternate on the renderer;
    `nonpalette` falls back to solid regardless of its literal pattern
    (`stroke_dash_index`'s non-palette fallback), so it needs no density
    beyond the floor. `named` rows carry a linetype name (resolved through
    `_LINETYPE_MAP`); `palette` rows already carry the lengths directly.
    """
    if row.form_id not in _DASH_EXPECTED_FORMS:
        return None
    if row.form_id == "named":
        return min(_LINETYPE_MAP[row.form_value])
    return min(row.form_value)


def _ring_samples(
    page: Any, fx: Fixture, rect: pc.CanvasRect, png: bytes, stroke_w: float, n: int
) -> list[tuple[int, ...]]:
    """Sample `n` points around the target's stroke ring, in perimeter order.

    `n` must be dense enough that the shortest segment of the row's own
    dash pattern spans several samples instead of at most one —
    `denoise_ring` only ever collapses an isolated single sample, so a
    segment that dense survives it (see `ring_sample_count`, which the
    caller uses to compute `n` from the target's own perimeter and the
    pattern's shortest segment). Rect samples walk the perimeter
    continuously (top left-to-right, right top-to-bottom, bottom
    right-to-left, left bottom-to-top) rather than interleaving the four
    sides per iteration: `ring_transitions` and `denoise_ring` treat the
    list as a circular perimeter sequence, so an interleaved order
    compares points on different sides of the rect as if they were
    adjacent.
    """
    pts: list[tuple[float, float]] = []
    if len(fx.target) == 3:
        cx, cy, r = fx.target
        ring_r = r - stroke_w / 2
        for i in range(n):
            a = 2 * math.pi * i / n
            pts.append((cx + ring_r * math.cos(a), cy + ring_r * math.sin(a)))
    else:
        x, y, w, h = fx.target
        inset = stroke_w / 2
        per_side = max(1, n // 4)
        for i in range(per_side):  # top, left-to-right
            t = (i + 0.5) / per_side
            pts.append((x + w * t, y + inset))
        for i in range(per_side):  # right, top-to-bottom
            t = (i + 0.5) / per_side
            pts.append((x + w - inset, y + h * t))
        for i in range(per_side):  # bottom, right-to-left
            t = (i + 0.5) / per_side
            pts.append((x + w * (1 - t), y + h - inset))
        for i in range(per_side):  # left, bottom-to-top
            t = (i + 0.5) / per_side
            pts.append((x + inset, y + h * (1 - t)))
    dev_pts = [pc.svg_to_device(rect, fx.svg_w, fx.svg_h, p) for p in pts]
    return pc.sample_pixels(page, png, dev_pts)


def _check_stroke_width(
    page: Any, fx: Fixture, rect: pc.CanvasRect, before: bytes, after: bytes
) -> None:
    probe = _ring_point(fx, inward=3.0)
    dev = pc.svg_to_device(rect, fx.svg_w, fx.svg_h, probe)
    fill_rgb = _target_fill_rgb(fx, fx.target_idx)
    stroke_rgb = _core.parse_color_to_rgba(BASE_STROKE)[:3]
    b_px = pc.sample_pixels(page, before, [dev])[0]
    a_px = pc.sample_pixels(page, after, [dev])[0]
    assert rgb_distance3(b_px, fill_rgb) < 30, (
        f"stroke_width baseline: expected fill at 3px inside the edge, got {b_px} "
        f"(committed bundle: {_BUNDLE_PATH})"
    )
    assert rgb_distance3(a_px, stroke_rgb) < 30, (
        f"stroke_width conditional: expected the widened stroke to reach 3px inside "
        f"the edge, got {a_px} (committed bundle: {_BUNDLE_PATH})"
    )


def _check_stroke_dash(
    page: Any, fx: Fixture, rect: pc.CanvasRect, after: bytes, row: Row, *, expect_dashed: bool
) -> None:
    stroke_rgb = _core.parse_color_to_rgba(BASE_STROKE)[:3]
    perimeter = _ring_perimeter(fx, DASH_STROKE_W)
    n = ring_sample_count(perimeter, _dash_min_segment(row))
    samples = _ring_samples(page, fx, rect, after, DASH_STROKE_W, n)
    classes = classify_ring(samples, stroke_rgb)
    if expect_dashed:
        assert ring_is_alternating(classes), (
            f"stroke_dash conditional: expected the ring to alternate stroke/background "
            f"around the perimeter (>=4 transitions), got {''.join(classes)} "
            f"(committed bundle: {_BUNDLE_PATH})"
        )
    else:
        assert ring_is_solid(classes), (
            f"stroke_dash conditional: expected a solidly-painted (>=90% stroke) ring for "
            f"a non-palette pattern, got {''.join(classes)} (committed bundle: {_BUNDLE_PATH})"
        )


# The same-mark probe that isolates a property `channel` does not own, on
# the TARGET mark itself, alongside the neighbor-mark probe below.
# `opacity` has no untouched region on its own mark (it is the
# whole-instance alpha, confirmed in circle.wgsl: `color.a *= in.opacity`
# applies after the fill/stroke mix); `size` and `angle`'s own-effect
# check already IS a geometry probe on the target, so there is no
# separate "other property" left to probe there.
_SAME_MARK_PROBE: dict[str, Literal["ring", "center"] | None] = {
    "color": "ring",
    "opacity": None,
    "size": None,
    "stroke_width": "center",
    "stroke_opacity": "center",
    "fill_opacity": "ring",
    "angle": None,
    "stroke_dash": "center",
}


def _check_isolation(
    page: Any, fx: Fixture, rect: pc.CanvasRect, before: bytes, after: bytes, row: Row
) -> None:
    """Assert the pixels ``row.channel``'s conditional does not name are unchanged.

    Two probes: an unselected neighbor mark elsewhere (proving the
    conditional does not leak beyond the marks it should affect) and a
    same-mark probe on the target itself (a property the channel does not
    own — see ``_SAME_MARK_PROBE``).
    """
    # `color`'s legend driver puts a field reference (`fm.Color(k)`) in the
    # conditional's own `if_not` branch (see `_build_fixture`) so the
    # legend has something to build swatches from; once any selection
    # exists, an unselected neighbor legitimately repaints to its own
    # category color instead of the resting style fill — a real
    # selected-vs-resting transition (see `_BASELINE`'s docstring), not a
    # leak, so this specific cell has no untouched neighbor pixel to probe.
    #
    # `size` x point: the *before*-frame sample is the same degenerate
    # tiny-circle-smaller-than-its-own-stroke case the before-frame check
    # above skips for the same channel — comparing it to any afterward
    # value is comparing against unreliable noise, not a leak (rect's
    # resting geometry comes from its own `x2`/`y2` encoding instead, with
    # a normal fill core, so rect x size still runs this probe).
    skip_neighbor = (row.channel == "color" and row.driver == "legend") or (
        row.channel == "size" and row.mark == "point"
    )
    if not skip_neighbor:
        dev = pc.svg_to_device(rect, fx.svg_w, fx.svg_h, fx.neighbor_center)
        b_px = pc.sample_pixels(page, before, [dev])[0]
        a_px = pc.sample_pixels(page, after, [dev])[0]
        assert rgb_distance3(b_px, a_px) < 20, (
            f"{row.channel} conditional: an unselected neighbor mark's pixel moved from "
            f"{b_px} to {a_px} (committed bundle: {_BUNDLE_PATH})"
        )

    kind = _SAME_MARK_PROBE[row.channel]
    if kind is None:
        return
    # Mid-band (half the stroke width in from the outer edge), not the
    # shared default: a ring probe close to either the stroke's outer or
    # inner AA transition picks up a few RGB units of the *other* side's
    # color as normal antialiasing, which a big same-row fill swap (this
    # probe's own channel, `color`, changing to a very different literal)
    # can push past a fixed isolation tolerance — not a real leak (measured
    # directly: the residual tracks the fill's own change, not the stroke).
    # Mid-band maximizes the margin from both AA transition zones.
    probe = _center_point(fx) if kind == "center" else _ring_point(fx, inward=BASE_STROKE_W / 2)
    dev2 = pc.svg_to_device(rect, fx.svg_w, fx.svg_h, probe)
    b_px2 = pc.sample_pixels(page, before, [dev2])[0]
    a_px2 = pc.sample_pixels(page, after, [dev2])[0]
    assert rgb_distance3(b_px2, a_px2) < 20, (
        f"{row.channel} conditional: the target's own {kind} pixel moved from {b_px2} to "
        f"{a_px2} (committed bundle: {_BUNDLE_PATH})"
    )


# ── stroke_dash form scoping ─────────────────────────────────────────────────

# stroke_dash's "named" (a linetype name, e.g. "dashed", resolved to
# lengths through `_LINETYPE_MAP`) and "palette" (a literal pattern already
# in the palette, e.g. [6, 3]) forms both resolve to a real palette entry
# and are expected to render as alternating stroke/background dashes on
# both marks. "nonpalette" ([7, 2, 1, 2]) is a pattern outside the palette,
# which renders solid on instance marks by design (disclosed in
# `stroke_dash_index`'s doc comment: an unrecognized pattern falls back to
# solid) — stays solid on both marks.
_DASH_EXPECTED_FORMS = frozenset({"named", "palette"})

# `stroke_opacity` on `rect`: there is no working plain-render path to
# derive the opacity-family oracle's "expected" value from, for two
# independent, confirmed reasons. (1) `mark_rect(stroke_opacity=...)` as a
# style kwarg raises `TypeError: unknown keyword argument 'stroke_opacity'`
# (rejected outright — probed directly, `mark_rect`'s honored kwargs list
# has no `stroke_opacity`). (2) The plain (non-conditional) constant-column
# *encode* (`mark_rect(...).encode(stroke_opacity="_ref")`) is instead
# silently accepted but inert — the static SVG carries no `stroke-opacity`
# attribute at all, while the identical setup on a circle correctly emits
# one. The conditional itself is not this bug: probed directly, a rect
# `stroke_opacity` conditional's own click-driven pixel moves from the
# opaque stroke color to a visibly lighter blend (see
# `test_opacity_family_xfail_rows_still_react`, which is not xfailed and
# asserts exactly that movement for these 6 rows). Filed, not fixed (no
# production change ships with this task); these 6 cells (rect x
# stroke_opacity x {unpacked, packed} x {click, legend, brush}) are xfail
# citing the plain-encode defect rather than silently skipped.
_RECT_STROKE_OPACITY_XFAIL_REASON = (
    "mark_rect rejects stroke_opacity= as a style kwarg with a TypeError, and its plain "
    "(non-conditional) constant-column encode is silently inert (no stroke-opacity attribute "
    "in the static SVG) — no working plain-render reference exists to derive the expected "
    "blend from; the conditional itself is confirmed to react (see the unxfailed "
    "test_opacity_family_xfail_rows_still_react)"
)

# `opacity` (the whole-mark alpha channel, not `stroke_opacity`/`fill_opacity`):
# a click-applied conditional value and the identical value baked in via the
# plain `opacity=` style kwarg render to measurably different pixels on this
# bundle, confirmed across 4 independent (mark, packing, driver) combinations,
# e.g. point/unpacked/click: plain render (220, 219, 227) vs
# conditional-applied (194, 197, 230), diffs up to 26; rect/packed/brush:
# plain (236, 235, 241) vs conditional (208, 211, 240), diffs up to 28. Both
# paths write the same GPU instance `opacity` field
# (`crates/ferrum-core/src/render/pack_instances.rs`'s initial pack and
# `crates/ferrum-wasm/src/conditional.rs`'s `*view.opacity = *o as f32`
# update). But the plain-style-kwarg reference is not a trusted ground
# truth either: neither an sRGB nor a linearized `a*fg + (1-a)*bg` blend
# formula matches it (measured: sRGB predicts (179, 200, 248), linearized
# predicts (212, 217, 248), the actual render is (220, 219, 227)) — its own
# B channel (227) sits below *both* the fill's B (235) and the white
# background's B (255), which no plain alpha blend of one over the other can
# produce. Which of the two paths (plain-style-kwarg bake, or the
# conditional's own incremental-update write) is the wrong one — or whether
# both share a root cause — is unestablished; this is a candidate
# renderer-wide compositing defect with open attribution (crates/ off-limits
# for this task). The conditional itself is confirmed to react (not inert):
# see `test_opacity_family_xfail_rows_still_react`, not xfailed, which
# asserts the pixel moves off resting toward the reference for these 12 rows.
_OPACITY_XFAIL_REASON = (
    "whole-mark opacity: the conditional-applied pixel and the plain-style-kwarg reference "
    "diverge by up to 28 RGB units, but the reference itself also matches neither an sRGB nor "
    "a linearized alpha-blend formula (its B channel is below both the fill's and the "
    "background's) — both the conditional-update path and the plain-bake path are suspects, "
    "attribution open — candidate renderer-wide compositing defect"
)


def _row_param(row: Row) -> Any:
    if row.channel == "stroke_opacity" and row.mark == "rect":
        marks = pytest.mark.xfail(
            strict=True, raises=AssertionError, reason=_RECT_STROKE_OPACITY_XFAIL_REASON
        )
        return pytest.param(row, id=row.id, marks=marks)
    if row.channel == "opacity":
        marks = pytest.mark.xfail(strict=True, raises=AssertionError, reason=_OPACITY_XFAIL_REASON)
        return pytest.param(row, id=row.id, marks=marks)
    return pytest.param(row, id=row.id)


# ── The oracle test ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("row", [_row_param(r) for r in ROWS])
def test_conditional_channel_pixels(row: Row, browser: Any, tmp_path: Path) -> None:
    fx = _build_fixture(row, tmp_path)
    page = browser.new_page(viewport=fx.viewport, device_scale_factor=pc.DEVICE_SCALE_FACTOR)
    try:
        pc.load_interactive(page, fx.html_path)
        rect = pc.canvas_rect(page)

        before = pc.screenshot(page)
        # Before-frame equals the static SVG's base appearance at the
        # sampled mark (static SVG never evaluates conditionals) — except
        # for `size`: its conditional-only resting circle defaults to the
        # built-in point area (36, r=3.385 SVG px), a radius *smaller than
        # its own 3px inset stroke*. `_target_fill_rgb` reads the raw
        # SVG `fill=` attribute, but even a perfect rasterizer's center
        # pixel for that geometry sits inside the stroke's own blend, not
        # the fill — the attribute is the wrong ground truth for this one
        # degenerate case, not a renderer defect (measured directly: the
        # drift ranges from ~18 RGB units on an unpacked point up to
        # ~125+ on a packed+legend one, where the mark's own category
        # color is also involved). No before-frame assertion for `size`;
        # every other channel's resting mark is large enough for this
        # comparison to mean something.
        if row.channel != "size":
            expected_fill = _target_fill_rgb(fx, fx.target_idx)
            dev_center = pc.svg_to_device(rect, fx.svg_w, fx.svg_h, _center_point(fx))
            before_center = pc.sample_pixels(page, before, [dev_center])[0]
            assert_rgb_close(before_center, expected_fill, tol=0, label="before-frame")

        _trigger(page, row, fx, rect)
        after = pc.screenshot(page)

        if row.channel == "color":
            _check_color(page, fx, rect, after, row.form_value)
        elif row.channel in ("opacity", "stroke_opacity", "fill_opacity"):
            _check_opacity_family(browser, row, fx, rect, before, after, page)
        elif row.channel == "size":
            _check_size(page, fx, rect, after, row)
        elif row.channel == "stroke_width":
            _check_stroke_width(page, fx, rect, before, after)
        elif row.channel == "angle":
            if row.mark == "point":
                _check_angle_point_identical(page, fx, rect, before, after)
            else:
                _check_angle_rect(page, fx, rect, before, after)
        elif row.channel == "stroke_dash":
            expect_dashed = row.form_id in _DASH_EXPECTED_FORMS
            _check_stroke_dash(page, fx, rect, after, row, expect_dashed=expect_dashed)
        else:  # pragma: no cover - exhaustive over _CONDITIONAL_CHANNELS
            raise AssertionError(f"no oracle wired for channel {row.channel!r}")

        _check_isolation(page, fx, rect, before, after, row)
    finally:
        page.close()


# ── Opacity-family xfail rows: the conditional itself must still react ───────

# The 12 whole-mark `opacity` rows and the 6 rect x `stroke_opacity` rows are
# strict xfail above (their exact-match leg fails for reasons independent of
# whether the conditional itself works — see `_OPACITY_XFAIL_REASON` and
# `_RECT_STROKE_OPACITY_XFAIL_REASON`). Folding a "did it move at all" check
# into that same xfailed test function would give it no real discriminating
# power: a future regression that makes the conditional fully inert
# (`actual == resting`) would raise the *same* `AssertionError` the strict
# xfail already expects, and would still read as xfailed. This is a
# separate, never-xfailed test so that regression has somewhere to fail for
# real, while the exact-match leg stays a documented, evidenced xfail above.
_OPACITY_FAMILY_XFAIL_ROWS = [
    r
    for r in ROWS
    if r.channel == "opacity" or (r.channel == "stroke_opacity" and r.mark == "rect")
]


@pytest.mark.parametrize("row", [pytest.param(r, id=r.id) for r in _OPACITY_FAMILY_XFAIL_ROWS])
def test_opacity_family_xfail_rows_still_react(row: Row, browser: Any, tmp_path: Path) -> None:
    fx = _build_fixture(row, tmp_path)
    page = browser.new_page(viewport=fx.viewport, device_scale_factor=pc.DEVICE_SCALE_FACTOR)
    try:
        pc.load_interactive(page, fx.html_path)
        rect = pc.canvas_rect(page)
        before = pc.screenshot(page)
        _trigger(page, row, fx, rect)
        after = pc.screenshot(page)

        probe = _ring_point(fx) if row.channel == "stroke_opacity" else _center_point(fx)
        dev = pc.svg_to_device(rect, fx.svg_w, fx.svg_h, probe)
        resting = pc.sample_pixels(page, before, [dev])[0]
        actual = pc.sample_pixels(page, after, [dev])[0]
        assert rgb_distance3(actual, resting) > 5, (
            f"{row.channel} conditional on {row.mark}: pixel did not move off resting at all "
            f"({resting} -> {actual}) — the conditional itself appears inert, not just its "
            f"exact-value comparison (committed bundle: {_BUNDLE_PATH})"
        )
        if row.channel == "opacity":
            # A working plain-render reference exists for this channel (see
            # `_RECT_STROKE_OPACITY_XFAIL_REASON` for why rect stroke_opacity
            # has none) — assert the movement is in the right direction, not
            # just present.
            expected = _reference_pixel(browser, row, probe)
            assert rgb_distance3(actual, expected) < rgb_distance3(resting, expected), (
                f"opacity conditional: pixel moved away from the plain-render reference "
                f"{expected}, not toward it (resting {resting} -> actual {actual}) "
                f"(committed bundle: {_BUNDLE_PATH})"
            )
    finally:
        page.close()


# ── Determinism ──────────────────────────────────────────────────────────────


def test_determinism_two_captures_are_byte_identical(browser: Any, tmp_path: Path) -> None:
    row = Row("color", "point", "unpacked", "hex", "#dc2626", "click")
    fx = _build_fixture(row, tmp_path)
    page = browser.new_page(viewport=fx.viewport, device_scale_factor=pc.DEVICE_SCALE_FACTOR)
    try:
        pc.load_interactive(page, fx.html_path)
        rect = pc.canvas_rect(page)
        _trigger(page, row, fx, rect)
        first = pc.screenshot(page)
        second = pc.screenshot(page)
        assert first == second, (
            f"two consecutive captures of one state must be byte-identical "
            f"(committed bundle: {_BUNDLE_PATH})"
        )
    finally:
        page.close()


# ── No-visual-effect rows: line/area/path/text and EncodingValue::Field ───────
#
# The WASM runtime writes only Circle and Rect instances (conditional.rs);
# a conditional on any other mark kind has no rendered effect. To make
# "identical" non-vacuous, a companion point mark shares the same
# click-driven selection and carries its own (always-working, per the
# opacity-family rows above) opacity conditional — its own pixel changing
# proves the click registered, while the target mark's own sampled pixel
# stays put.

_COMPANION_DF = pl.DataFrame({"x": [1.0], "y": [1.0], "k": ["zz"]})


def _companion_chart(sel: Any) -> Any:
    cond = sel.when(value(0.2)).otherwise(value(1.0))
    return (
        fm.Chart(_COMPANION_DF)
        .mark_point(
            size=BASE_MARK_SIZE, fill="#16a34a", stroke=BASE_STROKE, stroke_width=BASE_STROKE_W
        )
        .encode(x="x", y="y", tooltip="k", opacity=cond)
    )


def _companion_center(svg: str) -> tuple[float, float]:
    """Locate the companion mark's own circle by its distinctive fill color.

    Not the last ``<circle>`` in document order: a target chart whose
    conditional branch is a field reference (`fm.Color("k")`) gets its own
    legend, whose small swatch circles render after both layers — the
    companion's fill (`#16a34a`, unique in these fixtures) is the only
    reliable way to find it.
    """
    m = re.search(r'<circle[^>]*\bcx="([-\d.]+)"[^>]*\bcy="([-\d.]+)"[^>]*\bfill="#16a34a"', svg)
    return float(m.group(1)), float(m.group(2))


_INERT_DF = pl.DataFrame(
    {"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.0, 4.0, 1.0, 5.0, 3.0], "k": list("abcde")}
)
_GEO_UNIT_SQUARE = {
    "type": "Polygon",
    "coordinates": [[-10, -10], [10, -10], [10, 10], [-10, 10], [-10, -10]],
}


def _line_chart(sel: Any) -> Any:
    cond = sel.when(value("#dc2626")).otherwise(value(BASE_FILL))
    return fm.Chart(_INERT_DF).mark_line(stroke_width=6).encode(x="x", y="y", color=cond)


def _area_chart(sel: Any) -> Any:
    cond = sel.when(value("#dc2626")).otherwise(value(BASE_FILL))
    return fm.Chart(_INERT_DF).mark_area().encode(x="x", y="y", color=cond)


def _path_chart(sel: Any) -> Any:
    cond = sel.when(value("#dc2626")).otherwise(value(BASE_FILL))
    geo = {"type": "Polygon", "coordinates": [_GEO_UNIT_SQUARE["coordinates"]]}
    return (
        fm.Chart(geo)
        .mark_geoshape(fill=BASE_FILL)
        .coord(fm.CoordGeo(projection="equirectangular"))
        .encode(color=cond)
    )


def _text_chart(sel: Any) -> Any:
    cond = sel.when(value("#dc2626")).otherwise(value(BASE_FILL))
    return fm.Chart(_INERT_DF).mark_text(fill=BASE_FILL).encode(x="x", y="y", text="k", color=cond)


def _inert_probe(mark: str, svg: str) -> tuple[float, float]:
    """Locate a stable probe point on *mark*'s own geometry from the combined SVG."""
    if mark == "line":
        m = re.search(r'<polyline[^>]*\bpoints="([^"]+)"[^>]*\bstroke="#2563eb"', svg)
        pts = [tuple(map(float, p.split(","))) for p in m.group(1).split()]
        return (pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2
    if mark in ("area", "path"):
        m = re.search(r'<path[^>]*\bd="([^"]+)"', svg)
        coords = re.findall(r"([-\d.]+)[ ,]([-\d.]+)", m.group(1))
        pts = [(float(a), float(b)) for a, b in coords]
        if mark == "area":
            x0, y0 = pts[0]
            x1, y1 = pts[1]
            return (x0 + x1) / 2, (y0 + y1) / 2 + 20
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return sum(xs) / len(xs), sum(ys) / len(ys)
    if mark == "text":
        m = re.search(r'<text[^>]*\bx="([-\d.]+)"[^>]*\by="([-\d.]+)"[^>]*\bfill="#2563eb"', svg)
        return float(m.group(1)), float(m.group(2))
    raise ValueError(f"no probe locator for mark {mark!r}")


_INERT_BUILDERS: dict[str, Callable[[Any], Any]] = {
    "line": _line_chart,
    "area": _area_chart,
    "path": _path_chart,
    "text": _text_chart,
}


@pytest.mark.parametrize("mark", ["line", "area", "path", "text"])
def test_non_instance_marks_conditional_is_inert(mark: str, browser: Any, tmp_path: Path) -> None:
    sel = fm.selection_point(name="s", clear="false")
    target = _INERT_BUILDERS[mark](sel)
    companion = _companion_chart(sel)
    combo = (target + companion).add_selection(sel)

    svg = combo.to_svg()
    svg_w, svg_h = pc.parse_svg_size(svg)
    target_probe = _inert_probe(mark, svg)
    companion_cx, companion_cy = _companion_center(svg)

    html_path = tmp_path / "chart.html"
    combo.interactive().save(str(html_path))
    page = browser.new_page(viewport=pc.VIEWPORT, device_scale_factor=pc.DEVICE_SCALE_FACTOR)
    try:
        pc.load_interactive(page, html_path)
        rect = pc.canvas_rect(page)
        dev_target = pc.svg_to_device(rect, svg_w, svg_h, target_probe)
        dev_companion = pc.svg_to_device(rect, svg_w, svg_h, (companion_cx, companion_cy))

        before = pc.screenshot(page)
        b_target = pc.sample_pixels(page, before, [dev_target])[0]
        b_companion = pc.sample_pixels(page, before, [dev_companion])[0]

        client_pt = pc.svg_to_client(rect, svg_w, svg_h, (companion_cx, companion_cy))
        pc.click_at(page, client_pt)
        after = pc.screenshot(page)
        a_target = pc.sample_pixels(page, after, [dev_target])[0]
        a_companion = pc.sample_pixels(page, after, [dev_companion])[0]

        assert rgb_distance3(b_companion, a_companion) > 10, (
            f"{mark} inert-row registration proof: the companion point's opacity "
            f"conditional did not visibly change ({b_companion} -> {a_companion}), so the "
            f"click never registered (committed bundle: {_BUNDLE_PATH})"
        )
        assert rgb_distance3(b_target, a_target) < 6, (
            f"{mark} conditional: expected the mark's own pixel unchanged, moved from "
            f"{b_target} to {a_target} (committed bundle: {_BUNDLE_PATH})"
        )
    finally:
        page.close()


def test_field_branch_conditional_is_inert(browser: Any, tmp_path: Path) -> None:
    """A conditional branch built from a field (not ``value(...)``) has no rendered effect."""
    df = pl.DataFrame(
        {"x": [10.0, 11.0, 12.0, 13.0, 14.0], "y": [2.0, 4.0, 1.0, 5.0, 3.0], "k": list("abcde")}
    )
    sel = fm.selection_point(name="s", clear="false")
    cond = sel.when(fm.Color("k")).otherwise(value(BASE_FILL))
    target = (
        fm.Chart(df)
        .mark_point(size=BASE_MARK_SIZE, stroke=BASE_STROKE, stroke_width=BASE_STROKE_W)
        .encode(x="x", y="y", color=cond)
    )
    companion = _companion_chart(sel)
    combo = (target + companion).add_selection(sel)

    svg = combo.to_svg()
    svg_w, svg_h = pc.parse_svg_size(svg)
    target_cx, target_cy, _r = pc.parse_circles(svg)[1]
    companion_cx, companion_cy = _companion_center(svg)

    html_path = tmp_path / "chart.html"
    combo.interactive().save(str(html_path))
    page = browser.new_page(viewport=pc.VIEWPORT, device_scale_factor=pc.DEVICE_SCALE_FACTOR)
    try:
        pc.load_interactive(page, html_path)
        rect = pc.canvas_rect(page)
        dev_target = pc.svg_to_device(rect, svg_w, svg_h, (target_cx, target_cy))
        dev_companion = pc.svg_to_device(rect, svg_w, svg_h, (companion_cx, companion_cy))

        before = pc.screenshot(page)
        b_target = pc.sample_pixels(page, before, [dev_target])[0]
        b_companion = pc.sample_pixels(page, before, [dev_companion])[0]

        client_pt = pc.svg_to_client(rect, svg_w, svg_h, (companion_cx, companion_cy))
        pc.click_at(page, client_pt)
        after = pc.screenshot(page)
        a_target = pc.sample_pixels(page, after, [dev_target])[0]
        a_companion = pc.sample_pixels(page, after, [dev_companion])[0]

        assert rgb_distance3(b_companion, a_companion) > 10, (
            f"Field-branch registration proof: the companion's opacity conditional did not "
            f"visibly change (committed bundle: {_BUNDLE_PATH})"
        )
        assert rgb_distance3(b_target, a_target) < 6, (
            f"Field-branch conditional: expected the mark's own pixel unchanged, moved from "
            f"{b_target} to {a_target} (committed bundle: {_BUNDLE_PATH})"
        )
    finally:
        page.close()


# ── Coverage leg ─────────────────────────────────────────────────────────────


def test_coverage_covers_every_conditional_channel() -> None:
    """Every channel in ``_CONDITIONAL_CHANNELS`` must have at least one generated row."""
    covered = {row.channel for row in ROWS}
    missing = _CONDITIONAL_CHANNELS - covered
    assert not missing, f"channels with no pixel-harness row: {sorted(missing)}"
