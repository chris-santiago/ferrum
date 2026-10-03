"""Headless-Chrome pixel capture for the WASM conditional-encoding harness.

Mirrors ``tests/_snapshots.py``'s role for SVG rasterization: this module
holds the browser mechanics (system Chrome via Playwright, software-GL
launch flags, DOM-event driving, screenshot decoding, coordinate mapping)
that ``tests/test_conditional_pixels.py``'s oracle consumes. No assertions
live here.

Every capture tests the committed WASM bundle at
``src/ferrum/_wasm/ferrum_wasm_bg.wasm`` — an exported ``.interactive()``
chart embeds that binary inline. A conditional, shader, or scene-load change
needs ``wasm-pack build`` before this harness can see it.
"""

from __future__ import annotations

import base64
import functools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tests._conditional_oracle import rgb_distance3

# Software-GL flags: headless Chrome has no real GPU, so WebGL is routed
# through SwiftShader rather than silently falling back to no-WebGL.
LAUNCH_ARGS = [
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "--ignore-gpu-blocklist",
]
DEVICE_SCALE_FACTOR = 2
VIEWPORT = {"width": 900, "height": 560}
# Bounded settle windows. The runtime applies a conditional/click/brush
# synchronously in Rust (confirmed by reading conditional.rs: handleClick,
# toggleLegend and handleDrag all call apply_conditionals_and_render and
# render_frame before returning) — there is no async transition to wait for
# on the WASM side (transition.rs's tick_transition/startTransition machinery
# is for scene-to-scene data updates, never entered by a conditional). The
# real race is the browser's own compositor: a screenshot taken immediately
# after the synchronous call can still miss the paint the GPU submission
# triggers. Two chained `requestAnimationFrame` calls are the browser's own
# "a frame has been scheduled and composited" signal, so every settle below
# pumps that instead of guessing a sleep duration; the short fixed window on
# top absorbs remaining SwiftShader compositing latency.
_LOAD_SETTLE_MS = 1200
_INTERACTION_SETTLE_MS = 300
# Canvas-attach time (`page.goto` through `wait_for_selector`) measured
# directly across every page load in one full serial run of this module
# (192 loads): mean 1.95s, max 3.53s (the first, cold-start load), with no
# growth from the first load to the last. `page.close()` also closes the
# implicit context `browser.new_page()` created (Playwright's own
# documented behavior), so there is no per-row context leak to account for.
# The one observed failure at the previous 90_000ms ceiling was a single
# stall far outside that steady-state range, not a proportional slowdown:
# the same occasional CDP-pipe stall `screenshot`'s own docstring already
# documents ruling out a scaled multiple of measured time for. This ceiling
# is a generous absolute bound for that kind of rare stall, not a multiple
# of the ~2s steady state: 2x the previous ceiling, roughly 51x the
# measured max. Shared by every long-running page action in this module
# (screenshot, navigation, canvas-attach wait) rather than each picking its
# own ceiling, since all three hit the same class of rare stall.
_ACTION_TIMEOUT_MS = 180_000

_DOUBLE_RAF_JS = (
    "() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))"
)


def _settle(page: Any) -> None:
    """Pump two animation frames, then a short fixed window for GPU compositing."""
    page.evaluate(_DOUBLE_RAF_JS)
    page.wait_for_timeout(_INTERACTION_SETTLE_MS)


_DECODE_JS = """async ([b64, points]) => {
  const img = new Image();
  img.src = 'data:image/png;base64,' + b64;
  await img.decode();
  const c = document.createElement('canvas');
  c.width = img.width; c.height = img.height;
  const g = c.getContext('2d');
  g.drawImage(img, 0, 0);
  return points.map(([x, y]) =>
    Array.from(g.getImageData(Math.round(x), Math.round(y), 1, 1).data));
}"""


@functools.lru_cache(maxsize=1)
def chrome_available() -> bool:
    """Return whether Playwright can launch the system Chrome channel.

    Cached for the process lifetime — used as a module-level skip condition,
    so repeated collection does not re-launch a browser per check. Only
    catches the specific errors a missing/unlaunchable Chrome raises
    (`playwright.sync_api.Error`, e.g. "Unsupported chromium channel" or a
    missing executable; `OSError` for an OS-level launch failure such as a
    permission error) — anything else (an unrelated bug in this function)
    still raises, rather than reading as a benign "Chrome unavailable" skip.
    """
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome", headless=True, args=LAUNCH_ARGS)
            browser.close()
        return True
    except (PlaywrightError, OSError):
        return False


@dataclass(frozen=True)
class CanvasRect:
    """The WASM canvas's client-space bounding rect, as reported by the DOM."""

    x: float
    y: float
    w: float
    h: float


def load_interactive(page: Any, html_path: Path) -> None:
    """Navigate *page* to an exported ``.interactive().save()`` chart and settle.

    Waits for the canvas element to attach, then for two chained animation
    frames (the render loop's own signal that a frame has been scheduled and
    painted) before a short fixed settle for GPU compositing under
    SwiftShader.
    """
    page.goto(html_path.resolve().as_uri(), timeout=_ACTION_TIMEOUT_MS)
    page.wait_for_selector("canvas", state="attached", timeout=_ACTION_TIMEOUT_MS)
    page.evaluate(_DOUBLE_RAF_JS)
    page.wait_for_timeout(_LOAD_SETTLE_MS)


def canvas_rect(page: Any) -> CanvasRect:
    """Return the WASM canvas's client-space bounding rect."""
    r = page.evaluate(
        "() => { const c = document.querySelector('canvas').getBoundingClientRect();"
        " return [c.x, c.y, c.width, c.height]; }"
    )
    return CanvasRect(*r)


def svg_to_client(
    rect: CanvasRect, svg_w: float, svg_h: float, point: tuple[float, float]
) -> tuple[float, float]:
    """Map an SVG-space point to a client-space (CSS pixel) point on the canvas."""
    x, y = point
    return (rect.x + x * rect.w / svg_w, rect.y + y * rect.h / svg_h)


def svg_to_device(
    rect: CanvasRect, svg_w: float, svg_h: float, point: tuple[float, float]
) -> tuple[float, float]:
    """Map an SVG-space point to a device-pixel point in the screenshot's own coordinates."""
    cx, cy = svg_to_client(rect, svg_w, svg_h, point)
    return (cx * DEVICE_SCALE_FACTOR, cy * DEVICE_SCALE_FACTOR)


_OFF_CHART = (1.0, 1.0)


def click_at(page: Any, client_point: tuple[float, float]) -> None:
    """Click at a client-space point via a real DOM mouse event.

    Moves the pointer off-chart afterward so a hover tooltip never survives
    into the capture. Selections driven by this helper must set
    ``clear="false"`` — the default ``clear="mouseout"`` would otherwise
    clear the very selection this click just made when the pointer leaves.
    """
    page.mouse.click(*client_point)
    page.mouse.move(*_OFF_CHART)
    _settle(page)


def legend_click(page: Any, category: str) -> None:
    """Click a legend entry's label by its ``data-legend-category`` value."""
    page.locator(f'[data-legend-category="{category}"]').click()
    page.mouse.move(*_OFF_CHART)
    _settle(page)


def brush_drag(page: Any, start: tuple[float, float], end: tuple[float, float]) -> None:
    """Drag a brush from client-space *start* to *end* via real DOM mouse events."""
    page.mouse.move(*start)
    page.mouse.down()
    page.mouse.move(*end, steps=5)
    page.mouse.up()
    page.mouse.move(*_OFF_CHART)
    _settle(page)


_ZOOM_TRANSFORM_JS = """() => {
  for (const e of document.querySelectorAll('div')) {
    if (e.__zoom) return [e.__zoom.k, e.__zoom.x, e.__zoom.y];
  }
  return null;
}"""


def zoom_transform(page: Any) -> tuple[float, float, float]:
    """Return the chart's current d3-zoom transform ``(k, x, y)``.

    The runtime forwards exactly this transform to the focus panel's affine
    (``renderer.setTransform(focusPanel, k, x, y)``): a canvas point ``p``
    draws at ``p * k + (x, y)`` CSS pixels. d3-zoom keeps the transform on
    the element it is attached to (the chart wrapper ``div``) as ``__zoom``.
    """
    t = page.evaluate(_ZOOM_TRANSFORM_JS)
    if t is None:
        raise RuntimeError("zoom_transform: no element carries a d3-zoom transform")
    k, x, y = t
    return float(k), float(x), float(y)


def wheel_zoom(page: Any, client_point: tuple[float, float], delta_y: float) -> None:
    """Zoom the focus panel by a real DOM wheel event at a client-space point.

    A negative *delta_y* zooms in, anchored at *client_point*. Read the
    resulting transform with :func:`zoom_transform`.
    """
    page.mouse.move(*client_point)
    page.mouse.wheel(0, delta_y)
    page.mouse.move(*_OFF_CHART)
    _settle(page)


def screenshot(page: Any) -> bytes:
    """Capture the page as PNG bytes.

    Playwright's default action timeout (30s) was observed failing under
    ``-n 2`` (`Page.screenshot: Timeout 30000ms exceeded`) on a packed+brush
    row, even though a direct measurement of that exact fixture — both a
    single browser and two concurrent browsers actively driving the same
    brush-drag-then-screenshot loop — took 0.17-0.65s per capture every
    time. Since the actual encode/transfer work is not proportionally
    slower under contention, a "safe multiple" of the *measured* time (even
    50x would still land under 30s) would not have covered the failure — it
    is an occasional CDP-pipe stall, not a proportional slowdown, so the fix
    is a much larger absolute ceiling rather than a scaled one.
    """
    return page.screenshot(timeout=_ACTION_TIMEOUT_MS)


def sample_pixels(
    page: Any, png_bytes: bytes, points: list[tuple[float, float]]
) -> list[tuple[int, int, int, int]]:
    """Decode *png_bytes* in a fresh 2D canvas and sample RGBA at each device-space point.

    Decoding through an ``<img>``/2D-canvas round trip (rather than reading
    the WebGL canvas directly) avoids the false-blank readback trap: the
    WebGL drawing buffer is not guaranteed to still hold pixels by the time
    a script reads it back.
    """
    b64 = base64.b64encode(png_bytes).decode("ascii")
    pts = [[x, y] for x, y in points]
    raw = page.evaluate(_DECODE_JS, [b64, pts])
    return [tuple(int(c) for c in p) for p in raw]


def parse_svg_size(svg: str) -> tuple[float, float]:
    """Return the ``(width, height)`` declared on the root ``<svg>`` element."""
    w = re.search(r'<svg[^>]*\bwidth="([\d.]+)', svg)
    h = re.search(r'<svg[^>]*\bheight="([\d.]+)', svg)
    if w is None or h is None:
        raise ValueError("parse_svg_size: no <svg width=.. height=..> found")
    return float(w.group(1)), float(h.group(1))


def parse_circles(svg: str) -> list[tuple[float, float, float]]:
    """Return every ``<circle cx= cy= r=>`` triple, in document order."""
    return [
        (float(cx), float(cy), float(r))
        for cx, cy, r in re.findall(
            r'<circle[^>]*\bcx="([-\d.]+)"[^>]*\bcy="([-\d.]+)"[^>]*\br="([-\d.]+)"', svg
        )
    ]


def scan_radius(
    page: Any,
    png_bytes: bytes,
    rect: CanvasRect,
    svg_w: float,
    svg_h: float,
    center: tuple[float, float],
    fill_rgb: tuple[int, ...],
    *,
    max_r: int = 40,
    tolerance: float = 30.0,
) -> float:
    """Scan outward from *center* along +x and return the SVG-space radius at the fill edge.

    The edge is the first offset (in whole SVG pixels) whose sampled color
    differs from *fill_rgb* by more than *tolerance* in RGB distance. Used
    to measure a mark's achieved on-screen radius directly from a capture,
    rather than assuming a formula the renderer may not honor for every
    conditional value.
    """
    cx, cy = center
    points = [svg_to_device(rect, svg_w, svg_h, (cx + r, cy)) for r in range(max_r)]
    samples = sample_pixels(page, png_bytes, points)
    for r, px in enumerate(samples):
        if rgb_distance3(px, fill_rgb) > tolerance:
            return float(r)
    return float(max_r)


def parse_rects(svg: str) -> list[tuple[float, float, float, float]]:
    """Return every mark ``<rect x= y= width= height=>`` quadruple, in document order.

    Requires a ``stroke=`` attribute on the element, which only a mark rect
    carries (the background and plot-area rects never do; a zero
    ``stroke-width`` is omitted from the tag entirely, so that attribute
    cannot be the discriminator) — this distinguishes mark rects from chrome
    rects without depending on group nesting or class names.
    """
    return [
        (float(x), float(y), float(w), float(h))
        for x, y, w, h in re.findall(
            r'<rect(?=[^>]*\bstroke=)[^>]*\bx="([-\d.]+)"[^>]*\by="([-\d.]+)"'
            r'[^>]*\bwidth="([\d.]+)"[^>]*\bheight="([\d.]+)"',
            svg,
        )
    ]
