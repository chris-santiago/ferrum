# Phase 12: Spec Completeness Design Spec

## 1. Scope

Close the five largest gaps between `ferrum-spec.md` and the implementation: data transforms, scale types, utility modules (`ferrum.color`, `ferrum.config`), missing composition classes (`LayerChart`, `ConcatChart`), and per-channel `Axis` / `Legend` value classes. After this phase, every public API surface promised in the spec either works or is explicitly removed from the spec with a dated note.

## 2. Goals

- All 17 data transforms from §3.5 are callable, execute in Rust, and participate in the existing `TransformSpec` dispatch macro.
- All 16 scale classes from §3.6 are importable from `ferrum` and serialize correctly to the renderer.
- `ferrum.color` exposes programmatic palette access (categorical, sequential, diverging).
- `ferrum.config` provides process-level defaults (width, height, renderer, max_rows) via a `contextvars`-backed store matching the `set_default_theme()` pattern.
- `LayerChart` and `ConcatChart` are importable and composable with `|`, `&`, `+` operators.
- `Axis(...)` and `Legend(...)` are importable value classes, accepted by encoding channels via `axis=` / `legend=` kwargs, and plumbed through to the Rust renderer.

## 3. Non-goals

- New marks (arc, label, geoshape renderers remain deferred per code archaeology).
- Polar coordinate rendering (Theta/Radius channels remain warn-once).
- Full interactive cross-filtering (selections within a single view work; multi-view linked brushing is Phase 11 scope).
- Cyclical palette implementation (`rainbow`, `sinebow` remain reserved).
- Breaking changes to existing APIs — all additions are additive.

## 4. System behavior

### Data transforms

A user writes `chart.transform(transform_filter("datum.x > 10"))` or passes a list of transforms. Transforms execute in Rust before stat transforms, in declaration order. `transform_calculate` accepts a Vega-style expression string parsed by a minimal expression evaluator in Rust (field access, arithmetic, comparisons, ternary, string concat). `transform_window` supports `row_number`, `rank`, `dense_rank`, `lag`, `lead`, `first_value`, `last_value`, and rolling aggregates (`sum`, `mean`, `count`, `min`, `max`) with configurable `frame` and `groupby`.

### Scales

`Scale(type="pow", exponent=2)` and `ScalePow(exponent=2)` are equivalent. All scale classes serialize to a dict consumed by the renderer's scale-resolution pass. Band/Point scales inform the categorical position allocator (bar width, dot spacing). Color scales (`ScaleSequential`, `ScaleDiverging`, `ScaleThreshold`, `ScaleQuantile`, `ScaleQuantize`, `ScaleBinOrdinal`) drive the color-mapping pipeline independent of the palette registry.

### ferrum.color

`ferrum.color.palette("tableau10", n=5)` returns a list of hex strings. `ferrum.color.to_hex((0.2, 0.4, 0.6))` converts RGB tuples. `ferrum.color.sequential("viridis", n=256)` returns interpolated hex list. `ferrum.color.diverging("redblue", n=11)` returns centered diverging palette.

### ferrum.config

`ferrum.config.set(width=600, height=400)` sets process-level defaults. `ferrum.config.get("width")` retrieves. Context-manager usage: `with ferrum.config.defaults(width=800): ...`. Per-chart `.properties(width=...)` always wins (same precedence as theme: explicit > config > built-in default).

### LayerChart / ConcatChart

`LayerChart(chart_a, chart_b)` or `chart_a + chart_b` produces a shared-axes overlay. Layers share x/y scales by default (union domain). `ConcatChart(c1, c2, c3, columns=2)` produces a wrapping grid. Both accept `resolve=`, `title=`, `.theme()`, `.properties()`, `.save()`, `.show()`.

### Axis / Legend value classes

`X("field", axis=Axis(title="Speed", grid=False, label_angle=-45))` passes an `Axis` object that serializes to a dict consumed by the renderer's axis-layout pass. `Legend(orient="bottom", columns=3)` on any appearance channel controls that channel's legend independently. `legend=None` suppresses (existing behavior preserved). `axis=False` suppresses (shorthand for `Axis(domain=False, ticks=False, labels=False, title=None, grid=False)`).

## 5. Architecture

### Data transforms — Rust-first, expression-evaluated

Each transform is a new variant in the `for_each_transform!` macro table. The expression evaluator for `transform_calculate` and `transform_filter` lives in a new `crates/ferrum-core/src/transform/expr.rs` module — a recursive-descent parser over a minimal grammar (field refs `datum.field`, literals, arithmetic `+-*/`, comparison `> < >= <= == !=`, logical `&& || !`, ternary `? :`). No user-defined functions; no side effects.

Python constructors (`transform_filter(...)`, etc.) are thin wrappers that produce a `TransformSpec` JSON dict. They live in `src/ferrum/transforms.py`.

### Scales — PyO3 classes with dict serialization

New scale classes (`ScalePow`, `ScaleSqrt`, `ScalePoint`, `ScaleBand`, `ScaleSequential`, `ScaleDiverging`, `ScaleThreshold`, `ScaleQuantile`, `ScaleQuantize`, `ScaleBinOrdinal`, `ScaleUtc`) are PyO3 structs in `crates/ferrum-core/src/scale/`. Each implements `to_dict() -> dict` for renderer consumption. The existing `_scale_to_dict()` bridge in `src/ferrum/encoding/_scale.py` dispatches on type.

### Utility modules — pure Python over existing Rust palettes

`ferrum/color.py` wraps the Rust `ContinuousScheme` and categorical registry. `ferrum/config.py` is a `contextvars.ContextVar[dict]` store — no Rust involvement.

### Composition — Python-only, existing compositor

`LayerChart` overlays SVGs by combining layers into a single `<svg>` with shared viewBox. `ConcatChart` delegates to `compose_svg_grid` (already available) with auto-computed `columns`. Both inherit from `_ChartLike`.

### Axis / Legend — Python value classes, dict-serialized

Frozen dataclasses `Axis` and `Legend` in `src/ferrum/axis.py` and `src/ferrum/legend.py`. Encoding channels accept instances via `axis=` / `legend=` kwargs. Serialization produces the dict the Rust renderer already consumes from the existing dict-passthrough path.

## 6. Canonical interfaces / data contracts

```python
# Data transforms (src/ferrum/transforms.py)
def transform_filter(predicate: str | dict | Selection) -> TransformSpec: ...
def transform_calculate(as_: str, expr: str) -> TransformSpec: ...
def transform_window(*ops, sort=None, groupby=None, frame=None) -> TransformSpec: ...

# Scales (importable from ferrum)
class ScalePow(exponent: float = 2, *, domain=None, range=None, clamp=False): ...
class ScaleBand(*, domain=None, padding=0.1, padding_inner=None, padding_outer=None, align=0.5): ...
class ScalePoint(*, domain=None, padding=0.5, align=0.5): ...
class ScaleSequential(scheme=None, *, domain=None, reverse=False): ...
class ScaleDiverging(scheme=None, *, domain=None, domain_mid=None): ...
class ScaleThreshold(domain: list, range: list): ...
class ScaleQuantile(domain: list, range: list): ...
class ScaleQuantize(domain: tuple[float, float], range: list): ...
class ScaleBinOrdinal(bins: list, scheme=None): ...

# Axis / Legend (src/ferrum/axis.py, src/ferrum/legend.py)
@dataclass(frozen=True, slots=True)
class Axis:
    title: str | None = None
    orient: str | None = None
    ticks: bool = True
    grid: bool = True
    labels: bool = True
    label_angle: float | None = None
    label_format: str | None = None
    domain: bool = True
    # ... (all §3.7 parameters)

@dataclass(frozen=True, slots=True)
class Legend:
    title: str | None = None
    orient: str = "right"
    direction: str = "vertical"
    columns: int | None = None
    # ... (all §3.7 parameters)

# Composition
class LayerChart(_ChartLike):
    def __init__(self, *charts, resolve=None, title=None): ...

class ConcatChart(_ChartLike):
    def __init__(self, *charts, columns=None, spacing=10.0, resolve=None): ...

# ferrum.color
def palette(name: str, n: int | None = None) -> list[str]: ...
def to_hex(color: tuple | str) -> str: ...
def sequential(name: str, n: int = 256) -> list[str]: ...
def diverging(name: str, n: int = 11) -> list[str]: ...

# ferrum.config
def set(**kwargs) -> None: ...
def get(key: str) -> Any: ...
def defaults(**kwargs) -> ContextManager: ...
```

## 7. Invariants and constraints

- **No matplotlib.** Color utilities must not import matplotlib's colormaps.
- **No global mutable state** beyond the existing `contextvars` pattern. `ferrum.config` uses `ContextVar` (same as `set_default_theme`).
- **Arrow CDI boundary unchanged.** Data transforms execute in Rust on Arrow arrays; results return over CDI. No row-level Python iteration.
- **Expression evaluator is sandboxed.** No file I/O, no imports, no function calls beyond built-in math. Malformed expressions raise `ValueError` at parse time, not at apply time.
- **Backward compatible.** Existing `scale=LinearScale(...)` and `axis={"title": "foo"}` dict usage continues to work. `Axis(...)` and dicts are both accepted.
- **Serialization stability.** Scale/Axis/Legend `.to_dict()` output must be JSON-round-trippable and consumed by the current Rust renderer without renderer changes (for Axis/Legend) or with additive renderer changes (for new scale types).

## 8. Key decisions and tradeoffs

| Decision | Rationale | Alternative rejected |
|----------|-----------|---------------------|
| Expression evaluator in Rust (not Python `eval`) | Sandboxed, fast, no GIL re-entry; expressions run per-row on potentially millions of rows | Python lambda with `apply()` — violates "data stays in Rust" principle |
| `transform_calculate` takes a string, not a callable | Serializable to JSON spec; reproducible; enables future WASM execution | Callable API — breaks serialization, requires GIL |
| Color module wraps existing Rust registry | Single source of truth for palette data; no duplication | Pure-Python palette definitions — divergence risk with renderer |
| `LayerChart` reuses shared-viewBox SVG overlay (not `compose_svg_grid`) | Layers share axes by construction; grid compositor would add unnecessary gutters | Always use grid — adds whitespace between layers |
| `Axis`/`Legend` are frozen dataclasses (not PyO3 structs) | Pure Python for ergonomics; no Rust computation needed; dict-serialize for the existing renderer dict path | PyO3 classes — adds binding complexity for no computational benefit |
| `ferrum.config` is `contextvars`-based | Matches `set_default_theme()` pattern; thread-safe; auto-reverts in context managers | Module-level dict — not thread-safe; no auto-revert |
| Band/Point scales inform categorical position allocator | Spec requires control over bar width and dot spacing; allocator already reads scale dicts | Ignore band/point — lose bar-width control |

## 9. Acceptance criteria

1. `from ferrum import transform_filter, transform_calculate, transform_window` — all 17 transforms importable.
2. `Chart(...).transform(transform_filter("datum.x > 5")).show()` renders with filtered data.
3. `Chart(...).encode(x=X("field", scale=ScaleBand(padding=0.2)))` renders bars with correct padding.
4. `ferrum.color.palette("tableau10")` returns 10 hex strings.
5. `with ferrum.config.defaults(width=800): chart.show()` renders at 800px width.
6. `(chart_a + chart_b).show()` renders a `LayerChart` with shared axes.
7. `ConcatChart(c1, c2, c3, columns=2).show()` renders a 2-column wrapping grid.
8. `X("speed", axis=Axis(title="Speed (km/h)", grid=False))` renders with custom axis title and no grid.
9. `Color("species", legend=Legend(orient="bottom", columns=3))` renders bottom-oriented legend in 3 columns.
10. `cargo test` passes. `uv run pytest -n auto` passes. No regressions in existing test suite.
11. Expression evaluator rejects `import os` and `open(...)` at parse time with `ValueError`.

## 10. Validation strategy

- **Unit tests per transform:** each data transform gets a dedicated test verifying row-level output against hand-computed expected values.
- **Scale round-trip tests:** construct → `to_dict()` → feed to renderer → verify output domain/range.
- **Golden SVG tests:** `LayerChart` and `ConcatChart` produce goldens; visually inspected via `snapshot-goldens.py`.
- **Expression fuzzing:** property-based tests (Hypothesis) generating random valid/invalid expression strings; parser must never panic.
- **Config isolation tests:** verify `contextvars` scoping — nested context managers, threading, async.
- **Axis/Legend serialization:** verify dict output matches what the renderer already accepts via existing rendering tests.

## 11. Open questions

1. **Expression grammar scope:** Should `transform_calculate` support `datum["field with spaces"]` bracket notation, or require valid Python identifiers only? (Recommendation: support bracket notation for parity with Vega-Lite.)
2. **`ScaleUtc` vs `ScaleTime`:** Are these distinct classes or is UTC a boolean flag on `ScaleTime`? (Recommendation: `ScaleTime(utc=True)` — one fewer class to maintain.)
