"""Pixel-level proof that the interactive view draws a sized line's taper.

A sized ``mark_line`` carries one width per vertex. On screen its thickness at
every point must equal its scene width times the layout magnification, in CSS
pixels, measured perpendicular to the on-screen centerline: at rest, after a
uniform wheel zoom, and after a brush-driven domain rescale that stretches x
alone (``sx != sy``). The width stays constant in screen pixels while the
centerline follows the zoom.

Rows are the full product of the ``interpolate=`` vocabulary (read from Rust,
never mirrored), {solid, dashed} and {rest, zoom, rescale}. Each row exports an
overview + detail composite whose overview brush rescales the detail's x
domain, captures it in headless Chrome, and compares the detail line against
the static SVG of the same chart (the oracle) rasterized at the capture's
device scale. The oracle supplies centerline points, their tangent and their
thickness; each point is mapped through the detail panel's affine and the
capture is read along the on-screen perpendicular there. Dashed rows also
check that the stroke is on and off along the centerline exactly where the
dashed oracle is.

An untapered line (no ``size``) runs through the same product at rest and
under a uniform zoom, against its own static oracle: it keeps its constant
width. The at-rest capture of every tapered-line golden is pinned to that
golden's own rasterization, except the goldens named with their reason in
``_GOLDENS_NOT_CAPTURED``. Two exports of one chart, each loaded in its own
page, capture byte-identical frames.

Skipped whole-module when the system Chrome channel Playwright needs is
unavailable. Every capture tests the committed bundle
(``src/ferrum/_wasm/ferrum_wasm_bg.wasm``), so a tessellation or scene-load
change needs ``wasm-pack build`` before this harness can see it.
"""

from __future__ import annotations

import functools
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import polars as pl
import pytest

import ferrum as fm
from ferrum._core import interpolate_accepted_values
from tests import _pixel_capture as pc
from tests._snapshots import decode_png_rgba
from tests._tapered_line_charts import GOLDEN_CHARTS, GOLDENS_DIR, WIDE, dashed_solid

pytestmark = pytest.mark.wasm_pixels

_BUNDLE_PATH = "src/ferrum/_wasm/ferrum_wasm_bg.wasm"

State = Literal["rest", "zoom", "rescale"]
_STATES: tuple[State, ...] = ("rest", "zoom", "rescale")
# Dashes and gaps long against the widest stroke (22.6 px), so points well
# inside each lie a stroke width from its ends.
_DASHES: dict[str, list[float] | None] = {"solid": None, "dashed": [40.0, 48.0]}

_DPR = pc.DEVICE_SCALE_FACTOR
_VIEWPORT = {"width": 900, "height": 900}
_INK = (0, 0, 0)
# One device pixel at DPR 2: the contract's tolerance on every thickness.
_THICKNESS_TOL = 1.0
# How far (device px) the capture's stroke centre may sit from the
# affine-mapped oracle centerline: guards the mapping itself, so a wrong
# affine fails here instead of comparing thicknesses at unrelated points.
_LOCATION_TOL = 1.5
_PROBE = 2  # device px between neighbouring columns of a sample site


# ── Image access ─────────────────────────────────────────────────────────────


def _srgb_to_linear(c: int) -> float:
    v = c / 255.0
    return 255.0 * (v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4)


@dataclass(frozen=True)
class Image:
    """A decoded frame whose colors read in the space its renderer blended in.

    The interactive canvas is an sRGB surface the GPU blends in linear light,
    so a half-covered edge pixel is half the linear ink, not half the encoded
    byte; the static rasterizer blends the encoded bytes. Reading each frame
    in its own blend space makes coverage mean the same on both.
    """

    rows: list[list[tuple[int, int, int, int]]]
    w: int
    h: int
    linear: bool

    @classmethod
    def from_png(cls, png: bytes, *, linear: bool) -> Image:
        rows, w, h = decode_png_rgba(png)
        return cls(rows, w, h, linear)

    def rgb(self, x: int, y: int) -> tuple[float, ...]:
        """The pixel at ``(x, y)``, the nearest edge pixel beyond the frame."""
        x, y = min(max(x, 0), self.w - 1), min(max(y, 0), self.h - 1)
        return self.color(self.rows[y][x][:3])

    def color(self, rgb: tuple[int, ...]) -> tuple[float, ...]:
        """An encoded sRGB color in this frame's blend space."""
        return tuple(_srgb_to_linear(c) if self.linear else float(c) for c in rgb)

    def bilinear(self, x: float, y: float) -> tuple[float, ...]:
        """The color at a continuous point (pixel ``i`` spans ``[i, i + 1)``)."""
        x0, y0 = math.floor(x - 0.5), math.floor(y - 0.5)
        fx, fy = x - 0.5 - x0, y - 0.5 - y0
        c = [self.rgb(x0 + i, y0 + j) for j in (0, 1) for i in (0, 1)]
        w = [(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy]
        return tuple(sum(wk * ck[ch] for wk, ck in zip(w, c)) for ch in range(3))


def _coverage(px: tuple[float, ...], bg: tuple[float, ...], ink: tuple[float, ...]) -> float:
    """Fraction of *px* covered by *ink* over *bg*: the projection of the pixel's
    color onto the bg-to-ink segment, clamped to [0, 1]. A background that is
    itself ink (a read running off the stroke's end) counts as covered."""
    d = [i - b for i, b in zip(ink, bg)]
    denom = sum(c * c for c in d)
    if denom == 0:
        return 1.0
    t = sum((p - b) * c for p, b, c in zip(px, bg, d)) / denom
    return min(1.0, max(0.0, t))


def _is_ink(px: tuple[float, ...], bg: tuple[float, ...], ink: tuple[float, ...]) -> bool:
    """Whether *px* is mostly *ink* over *bg*: more than half covered, and a
    blend of the two rather than some other color (another series' stroke
    projects onto the bg-to-ink segment too, but lies far off it)."""
    d = [i - b for i, b in zip(ink, bg)]
    t = _coverage(px, bg, ink)
    off = math.dist(px, [b + t * c for b, c in zip(bg, d)])
    return t > 0.5 and off < 0.25 * math.hypot(*d)


@dataclass(frozen=True)
class Walk:
    """Ink crossed by a walk through a stroke (device px) and the walk offset
    of its coverage-weighted centre (0 when the walk is centred on the band)."""

    ink: float
    offset: float


def _walk(
    img: Image,
    center: tuple[float, float],
    normal: tuple[float, float],
    reach: float,
    ink: tuple[int, ...],
) -> Walk:
    """Walk *normal* (a unit device vector) through *center* from ``-reach`` to
    ``+reach`` device px, summing bilinear coverage in quarter-pixel steps
    against the background at the walk's two ends.

    Across a band, the ink is its thickness along the walk: the perpendicular
    thickness when the walk is perpendicular to it.
    """
    ink_c = img.color(ink)

    def at(t: float) -> tuple[float, ...]:
        return img.bilinear(center[0] + t * normal[0], center[1] + t * normal[1])

    ends = (at(-reach), at(reach))
    bg = tuple((a + b) / 2 for a, b in zip(*ends))
    step = 0.25
    ts = [-reach + (k + 0.5) * step for k in range(int(2 * reach / step))]
    cov = [_coverage(at(t), bg, ink_c) for t in ts]
    total = sum(cov)
    offset = sum(t * c for t, c in zip(ts, cov)) / total if total > 0 else 0.0
    return Walk(total * step, offset)


# ── Oracle sample sites: columns of the static render ────────────────────────


@dataclass(frozen=True)
class Run:
    """One column's crossing of the stroke (device px): its two edges and ink."""

    top: float
    bottom: float
    integral: float

    @property
    def center(self) -> float:
        return (self.top + self.bottom) / 2


def _column_runs(img: Image, x: int, y_lo: int, y_hi: int, ink: tuple[int, ...]) -> list[Run]:
    """Every crossing of the stroke along column *x* within ``[y_lo, y_hi)``.

    A crossing is a maximal stretch of pixels more than half covered by
    *ink* (see :func:`_is_ink`, so another color's stroke never counts), widened
    by two pixels on each side to collect its antialiased edges. The core
    threshold is against the column's lightest pixel; the integral's background
    is the mean of the pixels just beyond the widened stretch. Each edge sits
    where the coverage accumulated from the crossing's middle pixel outward
    runs out, placing a partially covered edge pixel to sub-pixel precision.
    """
    ink_c = img.color(ink)
    col = [img.rgb(x, y) for y in range(y_lo, y_hi)]
    page_bg = max(col, key=sum)
    core = [_is_ink(c, page_bg, ink_c) for c in col]
    runs: list[Run] = []
    y = 0
    while y < len(col):
        if not core[y]:
            y += 1
            continue
        top = y
        while y < len(col) and core[y]:
            y += 1
        a, b = max(0, top - 2), min(len(col), y + 2)
        above, below = col[max(0, a - 4)], col[min(len(col) - 1, b + 3)]
        bg = tuple((p + q) / 2 for p, q in zip(above, below))
        cov = [_coverage(col[i], bg, ink_c) for i in range(a, b)]
        mid = (top + y) // 2 - a
        upper, lower = sum(cov[:mid]), sum(cov[mid:])
        origin = y_lo + a + mid
        runs.append(Run(origin - upper, origin + lower, upper + lower))
    return runs


_PROBES = range(-3, 4)  # columns of a sample site, in units of _PROBE


def _quadratic(values: list[float]) -> tuple[float, float, float, float]:
    """Least-squares ``a + b*k + c*k^2`` through *values* at ``k`` in
    :data:`_PROBES`; returns ``(a, b, c, worst residual)``."""
    ks = list(_PROBES)
    n, s2, s4 = len(ks), sum(k * k for k in ks), sum(k**4 for k in ks)
    sy = sum(values)
    b = sum(k * v for k, v in zip(ks, values)) / s2
    c = (sum(k * k * v for k, v in zip(ks, values)) - s2 * sy / n) / (s4 - s2 * s2 / n)
    a = (sy - c * s2) / n
    worst = max(abs(v - (a + b * k + c * k * k)) for k, v in zip(ks, values))
    return a, b, c, worst


@dataclass(frozen=True)
class Site:
    """Where a column crosses a smooth band: centre row, slope (dy/dx) and the
    crossing's thickness across the band (device px)."""

    center: float
    slope: float
    thickness: float


def _site(img: Image, x: int, y_lo: int, y_hi: int, ink: tuple[int, ...]) -> Site | None:
    """A sample site at column *x*, or ``None`` where the stroke is not one
    smooth band there.

    Reads seven columns ``_PROBE`` apart centred on *x*; each must cross the
    stroke exactly once, both edges must follow a quadratic (no kink), and the
    crossing's thickness across the band must change no faster than a taper.
    That rejects joins, caps and dash ends; a quadratic through the centres
    gives the centerline's slope.
    """
    picked: list[Run] = []
    for k in _PROBES:
        runs = _column_runs(img, x + k * _PROBE, y_lo, y_hi, ink)
        if len(runs) != 1:
            return None
        picked.append(runs[0])
    center, b, c, _ = _quadratic([r.center for r in picked])
    slope = b / _PROBE
    if abs(slope) > 3.0:
        return None
    if (
        _quadratic([r.top for r in picked])[3] > 0.5
        or _quadratic([r.bottom for r in picked])[3] > 0.5
    ):
        return None
    across = [
        r.integral / math.hypot(1.0, (b + 2 * c * k) / _PROBE) for k, r in zip(_PROBES, picked)
    ]
    if max(across) - min(across) > 0.1 * len(across) * _PROBE:
        return None
    return Site(center, slope, across[len(across) // 2])


# ── Geometry: scene px -> device px ──────────────────────────────────────────


@dataclass(frozen=True)
class Affine:
    """A panel affine on canvas CSS px: a mark at canvas CSS point ``p`` draws
    at ``(p.x * sx + tx, p.y * sy + ty)``, its translation in CSS px too (the
    units the d3-zoom transform the runtime forwards carries)."""

    sx: float = 1.0
    sy: float = 1.0
    tx: float = 0.0
    ty: float = 0.0


@dataclass(frozen=True)
class PlotArea:
    x: float
    y: float
    w: float
    h: float


def _plot_areas(svg: str) -> list[PlotArea]:
    """Each panel's plot area, from the clip rects the SVG walker emits per panel."""
    return [
        PlotArea(*(float(v) for v in m))
        for m in re.findall(
            r'<clipPath id="ferrum-clip-\d+"><rect x="([-\d.]+)" y="([-\d.]+)" '
            r'width="([\d.]+)" height="([\d.]+)"/>',
            svg,
        )
    ]


@dataclass(frozen=True)
class Screen:
    """Where the canvas sits in the capture and how its focus panel is transformed."""

    rect: pc.CanvasRect
    svg_w: float
    affine: Affine

    @property
    def css_per_scene(self) -> float:
        return self.rect.w / self.svg_w

    def point(self, x: float, y: float) -> tuple[float, float]:
        """A scene point of the focus panel's marks, in device px."""
        a, k = self.affine, self.css_per_scene
        return (
            (self.rect.x + x * k * a.sx + a.tx) * _DPR,
            (self.rect.y + y * k * a.sy + a.ty) * _DPR,
        )

    def normal(self, slope: float) -> tuple[float, float]:
        """The on-screen unit normal of a scene tangent ``(1, slope)``."""
        tx, ty = self.affine.sx, self.affine.sy * slope
        n = math.hypot(tx, ty)
        return (-ty / n, tx / n)

    def window(self, area: PlotArea) -> tuple[float, float, float, float]:
        """A chrome rect (the scissor never moves) in device px: x0, x1, y0, y1."""
        k = self.css_per_scene * _DPR
        return (
            self.rect.x * _DPR + area.x * k,
            self.rect.x * _DPR + (area.x + area.w) * k,
            self.rect.y * _DPR + area.y * k,
            self.rect.y * _DPR + (area.y + area.h) * k,
        )


# ── Oracle: the static SVG of the same chart ─────────────────────────────────


@dataclass(frozen=True)
class OracleSample:
    """A centerline point of the static render (scene px), the centerline's
    slope there and its perpendicular thickness (scene px)."""

    x: float
    y: float
    slope: float
    thickness: float


@dataclass(frozen=True)
class Oracle:
    svg: str
    detail: PlotArea
    samples: tuple[OracleSample, ...]


def _rasterize(svg: str) -> Image:
    import resvg_py

    return Image.from_png(bytes(resvg_py.svg_to_bytes(svg_string=svg, zoom=_DPR)), linear=False)


def _normal(slope: float) -> tuple[float, float]:
    n = math.hypot(1.0, slope)
    return (-slope / n, 1.0 / n)


def _oracle(svg: str, ink: tuple[int, ...]) -> Oracle:
    """The first panel's line in the static SVG, rasterized at device scale.

    Sites step two scene px across the plot area, kept 8 px from its edges
    (see :func:`_site`). A walk along the perpendicular at each site centres
    the sample on the band and reads its thickness. A sample is dropped when
    its band runs within 3 px of the plot's top or bottom, when its column
    runs down a step's riser, when the walk and the column disagree on the
    thickness (a tight bend), or when a quarter of the sites within half its
    width are not samples (a join or a cap nearby): a join's shape is drawn in
    screen pixels, so under a non-uniform rescale the region around it is not
    the affine image of the static one.
    """
    img = _rasterize(svg)
    detail = _plot_areas(svg)[0]
    y_lo, y_hi = int(detail.y * _DPR) + 2, int((detail.y + detail.h) * _DPR) - 2
    found: dict[int, OracleSample] = {}
    x = detail.x + 8.0
    while x < detail.x + detail.w - 8.0:
        col = round(x * _DPR)
        site = _site(img, col, y_lo, y_hi, ink)
        if site is not None:
            normal = _normal(site.slope)
            first = _walk(img, (col + 0.5, site.center), normal, site.thickness / 2 + 4.0, ink)
            cx, cy = col + 0.5 + first.offset * normal[0], site.center + first.offset * normal[1]
            w = _walk(img, (cx, cy), normal, first.ink / 2 + 4.0, ink)
            # A band of thickness w is at least w wide along a row; a column
            # running down a step's riser reads the riser's length instead.
            wide = _walk(img, (cx, cy), (1.0, 0.0), w.ink + 8.0, ink).ink
            if (
                wide >= w.ink - 1.0
                and abs(w.ink - site.thickness) <= 2.0
                and y_lo + 6 < cy - w.ink / 2
                and cy + w.ink / 2 < y_hi - 6
            ):
                found[col] = OracleSample(cx / _DPR, cy / _DPR, site.slope, w.ink / _DPR)
        x += 2.0
    step = round(2.0 * _DPR)

    def clear_of_joins(col: int, s: OracleSample) -> bool:
        # A join or cap rejects every site across its own extent; a curve
        # only drops the odd site whose edges stray from a quadratic.
        near = range(-math.ceil(s.thickness / 4), math.ceil(s.thickness / 4) + 1)
        return sum(col + k * step in found for k in near) >= 0.75 * len(near)

    samples = [s for col, s in found.items() if clear_of_joins(col, s)]
    return Oracle(svg, detail, tuple(samples))


# ── Fixtures ─────────────────────────────────────────────────────────────────

_DF = pl.DataFrame(
    {
        "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [0.0, 0.7, 0.3, 1.1, 0.6, 1.4],
        "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    }
)


Width = Literal["tapered", "untapered"]


def _composite(interpolate: str, dash: list[float] | None, width: Width = "tapered") -> Any:
    """A detail chart over an overview whose x brush rescales the detail's x domain.

    Black stroke, no grid: coverage reads ink over a flat background. The
    detail is the focus panel the wheel zooms. A tapered line encodes
    ``size``; an untapered one is the same line at the default stroke width.
    """
    brush = fm.selection_interval(name="brush", encodings=["x"])
    mark_kwargs: dict[str, Any] = {"color": "black", "interpolate": interpolate}
    if dash is not None:
        mark_kwargs["stroke_dash"] = dash
    size = {"size": fm.Size("v", scale=WIDE)} if width == "tapered" else {}

    def line(**props: Any) -> Any:
        return (
            fm.Chart(_DF)
            .mark_line(**mark_kwargs)
            .encode(x="x", y="y", **size)
            .configure_grid(x=False, y=False)
            .properties(**props)
        )

    detail = line(width=640, height=440).encode(x=fm.X("x", scale={"domain": brush}))
    overview = line(width=640, height=200).add_params(brush)
    return fm.vconcat(detail, overview)


@functools.lru_cache(maxsize=None)
def _oracle_for(interpolate: str, width: Width = "tapered") -> Oracle:
    """The solid line's oracle: where the centerline runs and how thick it is."""
    return _oracle(_composite(interpolate, None, width).to_svg(), _INK)


@functools.lru_cache(maxsize=None)
def _dashed_raster(interpolate: str, width: Width = "tapered") -> Image:
    """The dashed line's static render: the same centerline, cut into dashes."""
    return _rasterize(_composite(interpolate, _DASHES["dashed"], width).to_svg())


def _drive(page: Any, state: State, svg: str, rect: pc.CanvasRect, svg_w: float) -> Affine:
    """Put the first panel in *state* and return its affine."""
    if state == "rest":
        return Affine()
    detail, overview = _plot_areas(svg)[:2]
    scale = rect.w / svg_w
    if state == "zoom":
        anchor = (
            rect.x + (detail.x + detail.w * 0.45) * scale,
            rect.y + (detail.y + detail.h * 0.5) * scale,
        )
        pc.wheel_zoom(page, anchor, -300.0)
        k, x, y = pc.zoom_transform(page)
        assert k > 1.2, f"the wheel did not zoom the detail panel (k={k})"
        return Affine(k, k, x, y)
    # Rescale: brush the overview over the middle of its x range. The
    # overview shares the detail's x scale and plot x-extent, so the brush's
    # scene span [b0, b1] becomes the detail's whole plot width. The drag
    # spans most of the overview's height too: a brush with no height is
    # empty and selects nothing.
    assert (overview.x, overview.w) == (detail.x, detail.w), "panels must share their x extent"
    b0, b1 = detail.x + detail.w * 0.3, detail.x + detail.w * 0.75
    y0, y1 = overview.y + overview.h * 0.2, overview.y + overview.h * 0.8
    pc.brush_drag(
        page,
        (rect.x + b0 * scale, rect.y + y0 * scale),
        (rect.x + b1 * scale, rect.y + y1 * scale),
    )
    sx = detail.w / (b1 - b0)
    return Affine(sx, 1.0, (detail.x - b0 * sx) * scale, 0.0)


def _load_on_whole_pixels(browser: Any, html: Path) -> tuple[Any, pc.CanvasRect]:
    """Open *html* with the canvas at a whole CSS pixel.

    The page centres the chart, so its canvas can land on a half CSS pixel,
    which the compositor snaps to the next whole one: the frame then sits a
    device pixel off the client rect the DOM reports. Widening the viewport
    by one pixel recentres the chart onto a whole pixel, where the rect is
    exact.
    """
    viewport = dict(_VIEWPORT)
    for _ in range(2):
        page = browser.new_page(viewport=viewport, device_scale_factor=_DPR)
        pc.load_interactive(page, html)
        rect = pc.canvas_rect(page)
        whole_x, whole_y = float(rect.x).is_integer(), float(rect.y).is_integer()
        if whole_x and whole_y:
            return page, rect
        page.close()
        viewport = {
            "width": viewport["width"] + (0 if whole_x else 1),
            "height": viewport["height"] + (0 if whole_y else 1),
        }
    raise AssertionError(f"canvas never landed on a whole CSS pixel: {rect}")


def _capture(browser: Any, html: Path, state: State, svg: str) -> tuple[Image, Screen]:
    page, rect = _load_on_whole_pixels(browser, html)
    try:
        svg_w, _ = pc.parse_svg_size(svg)
        affine = _drive(page, state, svg, rect, svg_w)
        return Image.from_png(pc.screenshot(page), linear=True), Screen(rect, svg_w, affine)
    finally:
        page.close()


def _visible(screen: Screen, oracle: Oracle, s: OracleSample) -> bool:
    """Whether the sample's whole perpendicular walk lands inside the first
    panel's plot area on screen (the scissor clips the rest)."""
    x0, x1, y0, y1 = screen.window(oracle.detail)
    cx, cy = screen.point(s.x, s.y)
    reach = s.thickness * screen.css_per_scene * _DPR / 2 + 6.0
    return x0 + reach < cx < x1 - reach and y0 + reach < cy < y1 - reach


def _read(img: Image, screen: Screen, s: OracleSample, ink: tuple[int, ...]) -> tuple[float, Walk]:
    """The expected device thickness at a sample and the capture's walk there."""
    want = s.thickness * screen.css_per_scene * _DPR
    return want, _walk(img, screen.point(s.x, s.y), screen.normal(s.slope), want / 2 + 4.0, ink)


def _check_thickness(
    img: Image, screen: Screen, oracle: Oracle, label: str, ink: tuple[int, ...] = _INK
) -> int:
    """At every visible oracle sample the capture's stroke is centred on the
    mapped centerline and as thick as the oracle's (scaled to device px)
    within one device px, measured perpendicular to the on-screen centerline;
    returns how many samples were compared."""
    compared = 0
    failures: list[str] = []
    for s in oracle.samples:
        if not _visible(screen, oracle, s):
            continue
        want, got = _read(img, screen, s, ink)
        compared += 1
        if abs(got.offset) > _LOCATION_TOL:
            failures.append(
                f"x={s.x:.1f}: stroke centre {got.offset:+.2f} px off the mapped centerline"
            )
        if abs(got.ink - want) > _THICKNESS_TOL:
            failures.append(f"x={s.x:.1f}: {got.ink:.2f} device px thick, want {want:.2f}")
    assert not failures, (
        f"{label}: {len(failures)} misses over {compared} samples "
        f"(committed bundle: {_BUNDLE_PATH}):\n  " + "\n  ".join(failures[:12])
    )
    return compared


def _dash_states(
    solid: Oracle, dashed: Image, ink: tuple[int, ...], margin_px: float, margin_widths: float
) -> list[tuple[OracleSample, float]]:
    """Each solid-line sample with the dashed render's ink across it there,
    kept only where the dashed render is plainly on (the solid thickness) or
    plainly off (no stroke ink) both at the sample and ``margin_px +
    margin_widths * width`` scene px either way along its tangent, so a dash
    end never decides: under a non-uniform rescale the on-screen
    perpendicular through a point leans in scene space, and near a dash end
    it would cross the cut. Probing along the tangent rather than at
    neighbouring samples keeps the margin whole where samples lie farther
    apart than it (a thin line's steep segment)."""

    def state(x: float, y: float, s: OracleSample) -> float | None:
        w = s.thickness * _DPR
        got = _walk(dashed, (x * _DPR, y * _DPR), _normal(s.slope), w / 2 + 4.0, ink).ink
        return got if abs(got - w) < 0.75 or got < 1.0 else None

    out: list[tuple[OracleSample, float]] = []
    for s in solid.samples:
        got = state(s.x, s.y, s)
        if got is None:
            continue
        m = (margin_px + margin_widths * s.thickness) / math.hypot(1.0, s.slope)
        ends = [state(s.x + k * m, s.y + k * m * s.slope, s) for k in (-1, 1)]
        if all(e is not None and (e >= 1.0) == (got >= 1.0) for e in ends):
            out.append((s, got))
    return out


def _check_dashes(
    img: Image,
    screen: Screen,
    solid: Oracle,
    dashed: Image,
    label: str,
    *,
    ink: tuple[int, ...] = _INK,
    margin_px: float = 1.0,
    margin_widths: float = 0.75,
) -> int:
    """Along the centerline the capture is on exactly where the dashed oracle
    is: inside every dash its perpendicular thickness is the oracle's (the
    taper's width there) within one device px, and inside every gap it carries
    no more ink than the oracle's gap; returns how many points were compared."""
    states: list[bool] = []
    failures: list[str] = []
    for s, oracle_ink in _dash_states(solid, dashed, ink, margin_px, margin_widths):
        if not _visible(screen, solid, s):
            continue
        want, got = _read(img, screen, s, ink)
        on = oracle_ink >= 1.0
        states.append(on)
        if on and abs(got.ink - want) > _THICKNESS_TOL:
            failures.append(f"x={s.x:.1f}: dash {got.ink:.2f} device px thick, want {want:.2f}")
        if not on and got.ink > oracle_ink * screen.css_per_scene + 1.0:
            failures.append(f"x={s.x:.1f}: {got.ink:.2f} device px of ink inside a dash gap")
    # The compared points must cover both a dash and a gap, several points
    # each, for the check to say anything about the pattern (a rescaled view
    # can show as little as one dash end).
    changes = sum(a != b for a, b in zip(states, states[1:]))
    assert changes >= 1 and min(states.count(True), states.count(False)) >= 3, (
        f"{label}: compared points do not cover a dash and a gap: {states}"
    )
    assert not failures, (
        f"{label}: {len(failures)} misses over {len(states)} dash points "
        f"(committed bundle: {_BUNDLE_PATH}):\n  " + "\n  ".join(failures[:12])
    )
    return len(states)


# ── The matrix ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Row:
    interpolate: str
    dash_id: str
    state: State

    @property
    def id(self) -> str:
        return f"{self.interpolate}-{self.dash_id}-{self.state}"


ROWS = [Row(i, d, s) for i in interpolate_accepted_values() for d in _DASHES for s in _STATES]

# An untapered line is pinned at rest and under a uniform zoom only. Under a
# non-uniform rescale a constant-width stroke thickens beside each inner
# join, whose single vertex carries a miter normal the affine maps wrongly;
# that defect belongs to constant-width strokes and is tracked on its own.
_UNTAPERED_STATES: tuple[State, ...] = ("rest", "zoom")
UNTAPERED_ROWS = [
    Row(i, d, s) for i in interpolate_accepted_values() for d in _DASHES for s in _UNTAPERED_STATES
]


def test_rows_cover_the_whole_matrix() -> None:
    families = interpolate_accepted_values()
    assert len(ROWS) == len(families) * len(_DASHES) * len(_STATES)
    assert {r.interpolate for r in ROWS} == set(families)
    assert len(UNTAPERED_ROWS) == len(families) * len(_DASHES) * len(_UNTAPERED_STATES)


def _check_row(row: Row, width: Width, oracle: Oracle, browser: Any, tmp_path: Path) -> None:
    """Capture *row*'s chart in its state and compare it against *oracle*."""
    html = tmp_path / "chart.html"
    _composite(row.interpolate, _DASHES[row.dash_id], width).interactive().save(str(html))
    img, screen = _capture(browser, html, row.state, oracle.svg)
    if row.dash_id == "dashed":
        dashed = _dashed_raster(row.interpolate, width)
        compared = _check_dashes(img, screen, oracle, dashed, row.id)
    else:
        compared = _check_thickness(img, screen, oracle, row.id)
    assert compared >= 5, f"{row.id}: only {compared} points visible to compare"


@pytest.mark.parametrize("row", [pytest.param(r, id=r.id) for r in ROWS])
def test_tapered_line_thickness_on_screen(row: Row, browser: Any, tmp_path: Path) -> None:
    oracle = _oracle_for(row.interpolate)
    widths = [s.thickness for s in oracle.samples]
    # Vacuity control: the oracle itself must taper, so a constant-width
    # capture cannot match it.
    assert len(widths) >= 8, f"oracle has too few readable samples ({len(widths)})"
    assert max(widths) - min(widths) > 6.0, f"oracle does not taper: {min(widths)}..{max(widths)}"
    _check_row(row, "tapered", oracle, browser, tmp_path)


@pytest.mark.parametrize("row", [pytest.param(r, id=r.id) for r in UNTAPERED_ROWS])
def test_untapered_line_thickness_on_screen(row: Row, browser: Any, tmp_path: Path) -> None:
    """A line with no ``size`` draws at its constant width, as its static render does."""
    oracle = _oracle_for(row.interpolate, "untapered")
    widths = [s.thickness for s in oracle.samples]
    # Vacuity control: the oracle is constant and thinner than any tapered
    # width, so a capture that drew a taper (or the tapered widths) fails.
    assert len(widths) >= 8, f"oracle has too few readable samples ({len(widths)})"
    assert max(widths) - min(widths) < 0.5, f"oracle is not constant: {min(widths)}..{max(widths)}"
    assert max(widths) < min(s.thickness for s in _oracle_for(row.interpolate).samples) - 1.0
    _check_row(row, "untapered", oracle, browser, tmp_path)


# ── At rest against the committed goldens ────────────────────────────────────

# Goldens the at-rest capture does not pin, each with its reason. Every other
# golden in GOLDEN_CHARTS is captured.
_GOLDENS_NOT_CAPTURED: dict[str, str] = {
    "hairpin_opacity.svg": (
        "the interactive view alpha-blends each overlapping piece of a stroke "
        "on its own, so a self-overlapping line below opacity 1 paints its "
        "overlap twice where the static outline paints it once (a known "
        "limit of the WASM mesh pipeline, tracked on its own); and every "
        "column of the hairpin crosses it more than once, so the column "
        "oracle has no single-band site to read its thickness at"
    ),
}
_CAPTURED_GOLDENS = sorted(set(GOLDEN_CHARTS) - set(_GOLDENS_NOT_CAPTURED))

# The solid chart sharing a dashed golden's centerline: the oracle reads
# thickness on the solid line and on/off from the dashed one.
_SOLID_TWINS = {"dashed.svg": dashed_solid}


def test_every_golden_is_captured_or_excluded() -> None:
    assert set(_GOLDENS_NOT_CAPTURED) <= set(GOLDEN_CHARTS)
    assert set(_SOLID_TWINS) <= set(_CAPTURED_GOLDENS)


def _outline_inks(svg: str) -> list[tuple[int, int, int]]:
    """Each tapered outline's fill color (the stroke color it paints), in
    document order without repeats."""
    fills = re.findall(r'<path[^>]*\bfill="#([0-9a-fA-F]{6})"[^>]*\bstroke="none"', svg)
    assert fills, "no tapered outline in the golden"
    return [(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)) for h in dict.fromkeys(fills)]


@pytest.mark.parametrize("name", _CAPTURED_GOLDENS)
def test_at_rest_thickness_matches_the_golden(name: str, browser: Any, tmp_path: Path) -> None:
    chart = GOLDEN_CHARTS[name]()
    golden = (GOLDENS_DIR / name).read_text()
    assert chart.to_svg() == golden, f"{name}: the chart no longer renders its golden"
    solid_svg = _SOLID_TWINS[name]().to_svg() if name in _SOLID_TWINS else golden

    html = tmp_path / "chart.html"
    chart.interactive().save(str(html))
    img, screen = _capture(browser, html, "rest", golden)
    for ink in _outline_inks(golden):
        label = f"{name} ink {ink}"
        solid = _oracle(solid_svg, ink)
        assert len(solid.samples) >= 5, f"{label}: too few readable samples ({len(solid.samples)})"
        if name in _SOLID_TWINS:
            # At rest the walk never leans and the tapered dash is cut where
            # the static one is, so a sample's own plain state decides: no
            # margin.
            raster = _rasterize(golden)
            margin = {"margin_px": 0.0, "margin_widths": 0.0}
            _check_dashes(img, screen, solid, raster, label, ink=ink, **margin)
        else:
            _check_thickness(img, screen, solid, label, ink)


# ── Determinism ──────────────────────────────────────────────────────────────


def test_two_loads_of_a_tapered_line_are_byte_identical(browser: Any, tmp_path: Path) -> None:
    """Two exports of one chart, each loaded (scene load and tessellation)
    in its own page, capture byte-identical frames."""
    frames = []
    for k in range(2):
        html = tmp_path / f"chart{k}.html"
        _composite("linear", _DASHES["dashed"]).interactive().save(str(html))
        page, _ = _load_on_whole_pixels(browser, html)
        try:
            frames.append(pc.screenshot(page))
        finally:
            page.close()
    assert frames[0] == frames[1], (
        f"two loads of one chart capture different frames (committed bundle: {_BUNDLE_PATH})"
    )
