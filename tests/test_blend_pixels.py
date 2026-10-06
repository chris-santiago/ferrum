"""Pixel-level proof that ``blend="additive"`` composites a mark over what lies beneath.

Four families of cells, every capture in headless Chrome:

**Parity across outputs.** Every mark whose Rust style table
(``mark_style_fields``) draws ``blend`` composites over what is beneath it on
the static SVG (``mix-blend-mode:screen``, the nearest mode SVG renderers
draw) and on the interactive view (an additive GPU pipeline; text and label,
which the page draws in its DOM text overlay, in a screen-blended overlay
layer of their own), and ``blend="normal"`` draws exactly as the mark does
with no ``blend`` at all. Each chart draws a green mark over a dark-red
background: compositing with that background only ever raises a pixel's red
channel and leaves green and blue as the mark painted them, under both screen
and additive, whatever color the mark is. The oracle counts the pixels whose
red rose that way against the same chart without ``blend``, so it needs no
mark color and the raster mark (colored by its scheme) shares it. The
population is every primitive mark a ``Chart.mark_<name>`` door reaches,
derived rather than hand-listed, and every member must have a fixture; the
``image`` member is drawn through ``mark_raster`` (``mark_image``'s url tiles
are not drawn by the WASM renderer at all). The ``blend="normal"`` cells are
the vacuity control. Opacity cells check that ``opacity < 1`` attenuates the
added color on both outputs, and export cells that the interactive view's
"Save PNG" composites blended text the way the page draws it.

**Text paint order.** Interactive text is drawn in the page's DOM overlay in
runs split at each blended batch, so blended text keeps layer order: an
unblended text layer drawn before a blended one stays beneath it, on the
static SVG, the interactive view and its saved PNG alike. One JS-level check
(no browser) pins a narrower invariant on that split: a zoom's
tick-coincidence heuristic can re-emit a mid-batch label as an unblended
tick label, splitting one batch's run into two non-adjacent pieces that
must not share a run key, or ``_placeTextRuns`` resolves both to the same
element and the second overwrites the first's glyphs.

**Exact interactive values.** The interactive view composites in linear light
into an sRGB target, so an additively blended opaque pixel is
``srgb(linear(src) + linear(dst))`` per channel. Rows for point, line, area
and raster read the pixel inside the mark that the static SVG of the same
chart says it covers.

**Layer scoping.** In a layered ``a + b`` with one blended layer, only that
layer adds: blended underneath an opaque layer it is hidden, blended on top
it adds to the layer below.

Every SVG capture renders the static SVG in the same headless Chrome that
captures the interactive view; every interactive capture tests the committed
WASM bundle at ``src/ferrum/_wasm/ferrum_wasm_bg.wasm``, so a blend pipeline
change needs ``wasm-pack build`` before this harness can see it.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import polars as pl
import pytest

import ferrum as fm
from ferrum._core import mark_style_fields
from tests import _pixel_capture as pc
from tests._snapshots import decode_png_rgba, rasterize_svg

pytestmark = pytest.mark.wasm_pixels

_BUNDLE_PATH = "src/ferrum/_wasm/ferrum_wasm_bg.wasm"

# ── Parity across outputs ────────────────────────────────────────────────────

_DARK_RED = "#800000"
_GREEN = "#008000"
# A blended pixel's red rises at least this far over the reference while its
# green and blue stay within the tolerance.
_RISE = 40
_CHANNEL_TOL = 16
_MIN_BLENDED = 150
# The opacity cells: below 1, the added green falls at least this far under
# the opaque blended render while the red the background gives holds.
_PARTIAL_OPACITY = 0.5
_GREEN_DROP = 20

Output = Literal["svg", "wasm"]
# An output, or the PNG the interactive view's "Save PNG" exports.
Capture = Literal["svg", "wasm", "saved"]
Blend = Literal["normal", "additive"]

_DF = pl.DataFrame(
    {
        "x": [1.0, 2.0, 3.0, 4.0],
        "y": [1.0, 3.0, 2.0, 4.0],
        "x2": [1.5, 2.5, 3.5, 4.5],
        "y2": [2.0, 4.0, 3.0, 5.0],
        "g": ["a", "b", "c", "d"],
    }
)
_GEO = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"name": "square"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [20, 0], [20, 20], [0, 20], [0, 0]]],
            },
        }
    ],
}


def _chart() -> fm.Chart:
    return fm.Chart(_DF)


def _opaque(kw: dict[str, Any]) -> dict[str, Any]:
    """*kw* with an opaque ``opacity`` unless the cell sets its own."""
    return {"opacity": 1, **kw}


# Each fixture takes the mark kwargs carrying `blend` (and `opacity` for the
# opacity cells; empty for the unblended reference render) and draws a large
# mark, opaque unless the kwargs say otherwise.
_FIXTURES: dict[str, Callable[[dict[str, Any]], fm.Chart]] = {
    "point": lambda kw: (
        _chart().mark_point(color=_GREEN, size=900, **_opaque(kw)).encode(x="x", y="y")
    ),
    "line": lambda kw: _chart().mark_line(color=_GREEN, stroke_width=10, **kw).encode(x="x", y="y"),
    "bar": lambda kw: _chart().mark_bar(color=_GREEN, **kw).encode(x="g", y="y"),
    "area": lambda kw: _chart().mark_area(color=_GREEN, **_opaque(kw)).encode(x="x", y="y"),
    "rule": lambda kw: _chart().mark_rule(color=_GREEN, stroke_width=10, **kw).encode(x="x"),
    "text": lambda kw: (
        _chart()
        .mark_text(color=_GREEN, font_size=48, font_weight="bold", **kw)
        .encode(x="x", y="y", text="g")
    ),
    "tick": lambda kw: _chart().mark_tick(color=_GREEN, stroke_width=10, **kw).encode(x="x", y="g"),
    "rect": lambda kw: (
        _chart().mark_rect(color=_GREEN, **kw).encode(x="x", x2="x2", y="y", y2="y2")
    ),
    "image": lambda kw: _chart().mark_raster(resolution=20, **kw).encode(x="x", y="y"),
    "ribbon": lambda kw: (
        _chart().mark_ribbon(color=_GREEN, **_opaque(kw)).encode(x="x", y="y", y2="y2")
    ),
    "segment": lambda kw: (
        _chart()
        .mark_segment(color=_GREEN, stroke_width=10, **kw)
        .encode(x="x", y="y", x2="x2", y2="y2")
    ),
    "arc": lambda kw: (
        _chart().mark_arc(color=_GREEN, **kw).encode(theta="x").coord(fm.CoordPolar())
    ),
    "label": lambda kw: (
        _chart()
        .mark_label(color=_GREEN, font_size=48, font_weight="bold", **kw)
        .encode(x="x", y="y", text="g")
    ),
    "geoshape": lambda kw: (
        fm.Chart(_GEO).mark_geoshape(color=_GREEN, **kw).coord(fm.CoordGeo(projection="mercator"))
    ),
}

# Marks the interactive view draws in the page's DOM text overlay, not on the GPU.
_OVERLAY_MARKS = ("label", "text")


def _primitive_mark(door: str) -> str | None:
    """The primitive mark a ``Chart.mark_<name>`` door names, if Rust knows it."""
    name = door.removeprefix("mark_")
    try:
        mark_style_fields(name)
    except ValueError:
        return None
    return name


def _population(*fields: str) -> tuple[str, ...]:
    """The door-reachable primitive marks whose style table draws every one of *fields*."""
    doors = (n for n in dir(fm.Chart) if n.startswith("mark_"))
    marks = {m for m in map(_primitive_mark, doors) if m is not None}
    return tuple(sorted(m for m in marks if set(fields) <= mark_style_fields(m).keys()))


_POPULATION = _population("blend")
_OPACITY_POPULATION = _population("blend", "opacity")


def test_every_blend_mark_has_a_fixture() -> None:
    assert set(_FIXTURES) == set(_POPULATION)


def test_the_overlay_marks_draw_blend() -> None:
    assert set(_OVERLAY_MARKS) <= set(_POPULATION)


def _parity_chart(mark: str, kw: dict[str, Any]) -> fm.Chart:
    return _FIXTURES[mark](kw).theme(fm.Theme(background=_DARK_RED))


def _cell_name(name: str, kw: dict[str, Any]) -> str:
    return "-".join([name, *(f"{k}{v}" for k, v in sorted(kw.items()))])


def _render(browser: Any, path: Path, chart: Any, output: Capture) -> bytes:
    """*chart* drawn by *output*: a page screenshot of the static SVG or the
    interactive view, or the PNG the interactive view's "Save PNG" button
    downloads."""
    page = browser.new_page(viewport=pc.VIEWPORT, device_scale_factor=pc.DEVICE_SCALE_FACTOR)
    try:
        if output == "svg":
            path = path.with_suffix(".svg")
            path.write_text(chart.to_svg())
            pc.load_svg(page, path)
            return pc.screenshot(page)
        path = path.with_suffix(".html")
        chart.interactive().save(str(path))
        pc.load_interactive(page, path)
        if output == "wasm":
            return pc.screenshot(page)
        with page.expect_download() as download:
            page.evaluate("document.querySelector('[title=\"Save PNG\"]').click()")
        return Path(download.value.path()).read_bytes()
    finally:
        page.close()


def _parity_capture(
    browser: Any, tmp_path: Path, mark: str, output: Capture, kw: dict[str, Any]
) -> bytes:
    path = tmp_path / _cell_name(mark, kw)
    return _render(browser, path, _parity_chart(mark, kw), output)


def _pixel_pairs(a: bytes, b: bytes) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    rows_a, w, h = decode_png_rgba(a)
    rows_b, w_b, h_b = decode_png_rgba(b)
    assert (w, h) == (w_b, h_b), "captures differ in size"
    return [
        (pa, pb)
        for row_a, row_b in zip(rows_a, rows_b, strict=True)
        for pa, pb in zip(row_a, row_b, strict=True)
    ]


def _blended_pixels(blended: bytes, reference: bytes) -> int:
    """Pixels whose red rose by ``_RISE`` over the reference while green and
    blue held: the trace of compositing over the dark-red background."""
    return sum(
        pb[0] >= pr[0] + _RISE
        and abs(pb[1] - pr[1]) <= _CHANNEL_TOL
        and abs(pb[2] - pr[2]) <= _CHANNEL_TOL
        for pb, pr in _pixel_pairs(blended, reference)
    )


def _attenuated_pixels(partial: bytes, opaque: bytes) -> int:
    """Pixels where the partly opaque blended render keeps the opaque blended
    render's red (the background, still composited under) while its green
    (the mark's added color) fell by ``_GREEN_DROP``."""
    return sum(
        abs(pp[0] - po[0]) <= _CHANNEL_TOL and po[1] >= pp[1] + _GREEN_DROP
        for pp, po in _pixel_pairs(partial, opaque)
    )


def _parity_cells() -> list[Any]:
    return [
        pytest.param(mark, output, blend, id=f"{mark}-{output}-{blend}")
        for mark in _POPULATION
        for output in ("svg", "wasm")
        for blend in ("additive", "normal")
    ]


@pytest.mark.parametrize(("mark", "output", "blend"), _parity_cells())
def test_blend_composites_over_what_is_beneath(
    mark: str, output: Output, blend: Blend, browser: Any, tmp_path: Path
) -> None:
    reference = _parity_capture(browser, tmp_path, mark, output, {})
    blended = _parity_capture(browser, tmp_path, mark, output, {"blend": blend})
    count = _blended_pixels(blended, reference)
    if blend == "additive":
        assert count >= _MIN_BLENDED, f"{mark} {output}: only {count} blended pixels"
    else:
        assert count == 0, f"{mark} {output}: blend='normal' blended {count} pixels"


@pytest.mark.parametrize("output", ["svg", "wasm"])
@pytest.mark.parametrize("mark", _OPACITY_POPULATION)
def test_opacity_below_one_attenuates_the_added_color(
    mark: str, output: Output, browser: Any, tmp_path: Path
) -> None:
    opaque = _parity_capture(browser, tmp_path, mark, output, {"blend": "additive", "opacity": 1})
    partial = _parity_capture(
        browser, tmp_path, mark, output, {"blend": "additive", "opacity": _PARTIAL_OPACITY}
    )
    count = _attenuated_pixels(partial, opaque)
    assert count >= _MIN_BLENDED, f"{mark} {output}: only {count} attenuated blended pixels"


@pytest.mark.parametrize("blend", ["additive", "normal"])
@pytest.mark.parametrize("mark", _OVERLAY_MARKS)
def test_saved_png_composites_blended_text_as_the_page_draws_it(
    mark: str, blend: Blend, browser: Any, tmp_path: Path
) -> None:
    reference = _parity_capture(browser, tmp_path, mark, "saved", {})
    blended = _parity_capture(browser, tmp_path, mark, "saved", {"blend": blend})
    count = _blended_pixels(blended, reference)
    if blend == "additive":
        assert count >= _MIN_BLENDED, f"{mark} saved PNG: only {count} blended pixels"
    else:
        assert count == 0, f"{mark} saved PNG: blend='normal' blended {count} pixels"


# ── Text paint order ─────────────────────────────────────────────────────────

_BLUE = "#0000ff"


def _text_layer(color: str, **kw: Any) -> fm.Chart:
    return (
        _chart()
        .mark_text(color=color, font_size=48, font_weight="bold", **kw)
        .encode(x="x", y="y", text="g")
    )


def _screened_over_ink(layered: bytes, ink_only: bytes) -> int:
    """Pixels where *ink_only* drew the blue ink and *layered* raised its green
    while keeping its blue: the green text screened over the blue text."""
    return sum(
        ink[2] >= 200
        and ink[0] <= _CHANNEL_TOL
        and ink[1] <= _CHANNEL_TOL
        and lay[1] >= ink[1] + _RISE
        and abs(lay[2] - ink[2]) <= _CHANNEL_TOL
        for lay, ink in _pixel_pairs(layered, ink_only)
    )


@pytest.mark.parametrize("blend", ["additive", "normal"])
@pytest.mark.parametrize("output", ["svg", "wasm", "saved"])
def test_blended_text_paints_over_unblended_text_drawn_before_it(
    output: Capture, blend: Blend, browser: Any, tmp_path: Path
) -> None:
    """An unblended text layer drawn before a blended one stays beneath it.

    The blue layer is drawn first and the green blended layer second, on the
    same glyphs. Screened in layer order, the green adds to the blue ink
    beneath it; drawn in the wrong order, the blue covers the green. With
    ``blend="normal"`` the green covers the blue (the vacuity control).
    """
    theme = fm.Theme(background=_DARK_RED)
    ink_only = _render(browser, tmp_path / "ink", _text_layer(_BLUE).theme(theme), output)
    layered_chart = (_text_layer(_BLUE) + _text_layer(_GREEN, blend=blend)).theme(theme)
    layered = _render(browser, tmp_path / f"layered-{blend}", layered_chart, output)
    count = _screened_over_ink(layered, ink_only)
    if blend == "additive":
        assert count >= _MIN_BLENDED, f"{output}: only {count} pixels screened over the ink"
    else:
        assert count == 0, f"{output}: blend='normal' screened {count} pixels"


# ── Zoom text-run key collision (no browser) ─────────────────────────────────
#
# `build_zoomed_text_json`'s tick-coincidence heuristic (text_json.rs) re-emits
# any text element whose content matches an axis tick's label string, at a row
# or column close enough to that axis, as a `tick_label_json` tick rather than
# its ordinary `text_element_to_json` form. The latter carries the `blend` key
# an additive batch's text needs; `tick_label_json` never does. A reclassified
# element sitting in the MIDDLE of an otherwise-contiguous blended batch thus
# splits that batch's run into two pieces that share one `panel:batch` key
# unless each run key is unique per occurrence (ferrum-anywidget.js's
# `_textRuns`). `_placeTextRuns` resolves a repeated key to one DOM element, so
# the second piece's `_joinLabels` call overwrites the first piece's glyphs.
#
# Reproducing the real coincidence pixel-for-pixel needs a mark positioned
# exactly on the axis's own tick row/column, which the renderer clips away
# before it ever reaches a page — so this drives the REAL `_textRuns` and
# `_placeTextRuns`, extracted verbatim from the committed `ferrum-anywidget.js`
# (not reimplemented), against a two-tick input shaped exactly like that split
# (one `text_element_to_json` element, one bare `tick_label_json` element, one
# more `text_element_to_json` element, all three sharing a batch), run under
# a minimal fake DOM standing in for the page. No browser needed for this one.


def _extract_js_function(source: str, name: str) -> str:
    """The exact source text of ``function name(...) { ... }``, by brace balance."""
    marker = f"function {name}("
    start = source.index(marker)
    brace_start = source.index("{", start)
    depth = 0
    i = brace_start
    while True:
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    return source[start : i + 1]


# A fake DOM with just enough surface for `_placeTextRuns` (createElementNS,
# setAttribute/getAttribute, dataset, style.cssText, querySelectorAll,
# insertBefore) to run under plain `node`, no browser. `eval` below runs the
# REAL `_textRuns`/`_placeTextRuns` source sliced out of this repo's own
# committed `ferrum-anywidget.js` (interpolated in as a JSON string literal by
# the caller) — not external or user-supplied input, so this is evaluating
# our own production code under test, the JS-level equivalent of the file
# read everywhere else in this module.
_FAKE_DOM_HARNESS = """
function _makeEl() {
  const attrs = {};
  return {
    dataset: {},
    style: { cssText: '' },
    setAttribute(k, v) { attrs[k] = v; },
    getAttribute(k) { return attrs[k]; },
    remove() {
      const i = _parent.children.indexOf(this);
      if (i >= 0) _parent.children.splice(i, 1);
    },
  };
}

const _parent = {
  children: [],
  querySelectorAll(_sel) {
    return _parent.children.filter((c) => c.getAttribute('class') === 'ferrum-text-run');
  },
  insertBefore(el, ref) {
    const i = _parent.children.indexOf(el);
    if (i >= 0) _parent.children.splice(i, 1);
    const j = _parent.children.indexOf(ref);
    _parent.children.splice(j < 0 ? _parent.children.length : j, 0, el);
  },
};

global.document = { createElementNS(_ns, _tag) { return _makeEl(); } };

// Records every `_joinLabels` call per element identity, so a collision (two
// runs resolving to the same DOM element in one `_placeTextRuns` pass) is
// directly observable rather than inferred from which content survives.
let _joinCalls = [];
function _joinLabels(el, texts) {
  _joinCalls.push({ el, content: texts.map((t) => t.content) });
  el.content = texts.map((t) => t.content);
}

const _svgEl = {
  parentNode: _parent,
  getAttribute(k) { return { width: '640', height: '480' }[k]; },
};

// eslint-disable-next-line no-eval -- see module docstring above this block.
eval(__TEXT_RUNS_SRC__);
// eslint-disable-next-line no-eval -- see module docstring above this block.
eval(__PLACE_RUNS_SRC__);

// Mirrors exactly what text_json.rs can emit for one additive batch split by
// a mid-batch reclassification: `text_element_to_json` carries `blend`;
// `tick_label_json` never does.
function _texts() {
  return [
    { x: 10, y: 20, content: 'alpha', blend: { panel: 0, batch: 0 } },
    { x: 15, y: 437.961, content: '2' },
    { x: 10, y: 20, content: 'omega', blend: { panel: 0, batch: 0 } },
  ];
}

// Tick 1 seeds the DOM with (possibly colliding) `ferrum-text-run` elements.
// Tick 2 is where a stale-cleanup pass resolves a colliding key to one
// surviving element, so two distinct runs both `_joinLabels` it: the second
// overwrites the first run's glyphs within that tick.
_placeTextRuns(_svgEl, _textRuns(_texts()).runs);
_joinCalls = [];
const _runs2 = _textRuns(_texts()).runs;
_placeTextRuns(_svgEl, _runs2);

const _byEl = new Map();
for (const call of _joinCalls) {
  _byEl.set(call.el, (_byEl.get(call.el) || []).concat([call.content]));
}
const _collided = [..._byEl.values()].filter((calls) => calls.length > 1);
console.log(JSON.stringify({
  runKeysUnique: new Set(_runs2.map((r) => r.key)).size === _runs2.length,
  collided: _collided,
}));
"""

_ANYWIDGET_JS = Path(__file__).resolve().parent.parent / "src/ferrum/_wasm/ferrum-anywidget.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_zoom_text_run_keys_stay_unique_when_a_batch_splits() -> None:
    """A tick-coincident mid-batch label must not make two runs share a key.

    See the section docstring above for the full mechanism. Without a unique
    key per occurrence, `_placeTextRuns` resolves the split batch's two
    blended pieces to one DOM element and the second `_joinLabels` call
    overwrites the first piece's glyphs.
    """
    source = _ANYWIDGET_JS.read_text()
    text_runs_src = _extract_js_function(source, "_textRuns")
    place_runs_src = _extract_js_function(source, "_placeTextRuns")
    script = _FAKE_DOM_HARNESS.replace("__TEXT_RUNS_SRC__", json.dumps(text_runs_src)).replace(
        "__PLACE_RUNS_SRC__", json.dumps(place_runs_src)
    )
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"node harness crashed: {result.stderr}"
    payload = json.loads(result.stdout)
    assert payload["runKeysUnique"], "two runs of the split batch share a key"
    assert payload["collided"] == [], (
        "a text-run element received two _joinLabels calls in one zoom tick "
        f"(the second overwrote the first): {payload['collided']}"
    )


# ── Exact interactive values ─────────────────────────────────────────────────

_GRAY = (0x40, 0x40, 0x40)
_INK = (0x20, 0x30, 0x70)
_THEME = fm.Theme(background="#404040", grid=False)
# Channel tolerance: one sRGB rounding step each way through the linear
# round trip, plus SwiftShader's own quantization.
_TOLERANCE = 4

_LINE = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 1.0, 2.0, 3.0]})
_ONE = pl.DataFrame({"x": [1.0], "y": [1.0]})
# A 4x4 grid whose every cell holds a different count, so every raster cell
# is drawn and none shares its neighbor's color.
_GRID = pl.DataFrame(
    [(float(i), float(j)) for i in range(4) for j in range(4) for _ in range(i + 4 * j + 1)],
    schema=["x", "y"],
    orient="row",
)


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#" + "".join(f"{c:02x}" for c in rgb)


def _with_blend(style: dict[str, Any], blend: str | None) -> dict[str, Any]:
    return {**style, "blend": blend} if blend else style


def _point(blend: str | None) -> Any:
    style = {"fill": _hex(_INK), "opacity": 1.0, "size": 3000.0, "stroke": "none"}
    return fm.Chart(_ONE).mark_point(**_with_blend(style, blend)).encode(x="x", y="y")


def _line(blend: str | None) -> Any:
    style = {"stroke": _hex(_INK), "opacity": 1.0, "stroke_width": 24.0}
    return fm.Chart(_LINE).mark_line(**_with_blend(style, blend)).encode(x="x", y="y")


def _area(blend: str | None) -> Any:
    style = {"fill": _hex(_INK), "opacity": 1.0}
    return fm.Chart(_LINE).mark_area(**_with_blend(style, blend)).encode(x="x", y="y")


def _raster(blend: str | None) -> Any:
    extra = {"blend": blend} if blend else {}
    return fm.Chart(_GRID).mark_raster(resolution=4, **extra).encode(x="x", y="y")


_MARKS: dict[str, Callable[[str | None], Any]] = {
    "point": _point,
    "line": _line,
    "area": _area,
    "raster": _raster,
}


def _to_linear(c: int) -> float:
    s = c / 255.0
    return s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4


def _to_srgb(v: float) -> int:
    v = min(v, 1.0)
    s = v * 12.92 if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055
    return round(s * 255.0)


def _added(src: tuple[int, ...], dst: tuple[int, ...]) -> tuple[int, ...]:
    """The pixel additive blending draws *src* onto *dst*."""
    return tuple(_to_srgb(_to_linear(s) + _to_linear(d)) for s, d in zip(src[:3], dst[:3]))


def _assert_close(actual: tuple[int, ...], expected: tuple[int, ...], what: str) -> None:
    assert all(abs(a - e) <= _TOLERANCE for a, e in zip(actual[:3], expected[:3])), (
        f"{what}: captured {actual[:3]}, expected {expected[:3]} (committed bundle: {_BUNDLE_PATH})"
    )


def _probe(svg: str, mark: str) -> tuple[float, float]:
    """The SVG point the mark covers, read from the static render of the chart.

    The centroid of the ink pixels in the rasterized SVG for the marks drawn
    in :data:`_INK`; for a raster, the center of the image's top-left cell.
    """
    if mark == "raster":
        image = re.search(
            r'<image[^>]*x="([\d.]+)"[^>]*y="([\d.]+)"[^>]*width="([\d.]+)"[^>]*height="([\d.]+)"',
            svg,
        )
        assert image is not None
        x, y, w, h = map(float, image.groups())
        return (x + w / 8, y + h / 8)
    rows, width, height = decode_png_rgba(rasterize_svg(svg))
    ink = [(x, y) for y in range(height) for x in range(width) if rows[y][x][:3] == _INK]
    assert ink, "the static render draws no ink pixel"
    svg_w, svg_h = pc.parse_svg_size(svg)
    return (
        sum(x for x, _ in ink) / len(ink) * svg_w / width,
        sum(y for _, y in ink) / len(ink) * svg_h / height,
    )


def _capture(browser: Any, chart: Any, svg: str, point: tuple[float, float], path: Path) -> tuple:
    chart.interactive().save(str(path))
    svg_w, svg_h = pc.parse_svg_size(svg)
    page = browser.new_page(
        viewport={"width": int(svg_w) + 40, "height": int(svg_h) + 40},
        device_scale_factor=pc.DEVICE_SCALE_FACTOR,
    )
    try:
        pc.load_interactive(page, path)
        rect = pc.canvas_rect(page)
        shot = pc.screenshot(page)
        (pixel,) = pc.sample_pixels(page, shot, [pc.svg_to_device(rect, svg_w, svg_h, point)])
    finally:
        page.close()
    return pixel


@pytest.mark.parametrize("mark", sorted(_MARKS))
def test_additive_adds_the_marks_color_to_what_lies_beneath(
    mark: str, browser: Any, tmp_path: Path
) -> None:
    build = _MARKS[mark]
    normal = build(None).theme(_THEME)
    svg = normal.to_svg()
    point = _probe(svg, mark)
    drawn = _capture(browser, normal, svg, point, tmp_path / "normal.html")
    # Vacuity control: the mark is drawn where the probe reads, not the
    # background it would otherwise add onto.
    assert drawn[:3] != _GRAY, f"no {mark} drawn at {point}"
    if mark != "raster":
        _assert_close(drawn, _INK, f"{mark} without blend")
    added = _capture(
        browser, build("additive").theme(_THEME), svg, point, tmp_path / "additive.html"
    )
    _assert_close(added, _added(drawn, _GRAY), f"{mark} with blend='additive'")


# ── Layer scoping ────────────────────────────────────────────────────────────

# (layered chart, expected pixel): only the layer that set blend adds.
_LAYERED: dict[str, tuple[Callable[[], Any], tuple[int, ...]]] = {
    # The blended layer is drawn first and the opaque one covers it.
    "blended below": (lambda: _point("additive") + _point(None), _INK),
    # The blended layer is drawn second and adds to the opaque one.
    "blended above": (lambda: _point(None) + _point("additive"), _added(_INK, _INK)),
}


@pytest.mark.parametrize("case", sorted(_LAYERED))
def test_only_the_layer_that_set_blend_blends(case: str, browser: Any, tmp_path: Path) -> None:
    build, expected = _LAYERED[case]
    chart = build().theme(_THEME)
    svg = chart.to_svg()
    assert svg.count("mix-blend-mode:screen") == 1
    pixel = _capture(
        browser,
        chart,
        svg,
        _probe(_point(None).theme(_THEME).to_svg(), "point"),
        tmp_path / "layered.html",
    )
    _assert_close(pixel, expected, case)
