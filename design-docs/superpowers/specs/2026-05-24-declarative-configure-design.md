# Declarative Configuration Surface — Design Spec

**Date:** 2026-05-24
**Status:** draft
**Slug:** `declarative-configure`

---

## 1. Scope

A rich declarative configuration surface that gives users fine-grained control over chart appearance — axis formatting, legend layout, annotations, secondary axes, broken axes, and inset panels — without breaking out of ferrum's declarative model. Extends the existing Theme system with chart-level structural configuration, a full annotation layer, and structural composition primitives. Eliminates the most common reasons users reach for matplotlib.

---

## 2. Goals

- Cover the 20 most common matplotlib customization patterns declaratively (axis formatting, tick control, legend placement, annotation, dual axes, insets)
- Maintain the existing `+` composition model — configuration, annotations, and structural features are immutable objects composed with Chart via `__add__`
- Provide typed, IDE-discoverable `.configure_*()` methods as the primary ergonomic surface
- Named format presets that eliminate d3-format memorization for common cases
- Full annotation layer (text, arrows, shapes, callouts) positioned in data, pixel, or normalized coordinates
- Structural features (secondary Y, broken axes, inset panels) as first-class composable spec layers
- Thin override escape hatch for the 5% case where the typed surface hasn't caught up
- Clear cascade precedence between override, per-channel config, chart-level configure, and theme
- Comprehensive documentation: conceptual guides, concept pages, recipes, migration table

---

## 3. Non-goals

- Imperative post-render mutation model (no mutable "axes" objects returned after render)
- Callable formatters (would break Rust-side computation invariant)
- matplotlib as a dependency or backend
- Replacing the existing Theme system (Configure complements it)
- Arbitrary Python callbacks in the render pipeline

---

## 4. System behavior

### Configuration

Users set chart-level defaults via `.configure_*()` or `+ Configure(...)`. These apply to all matching elements (all axes, all legends) unless targeted to a specific axis (x/y/y2). Per-channel `axis=Axis(...)` on an encoding still wins over chart-level configure.

```python
chart.configure_axis(label_angle=-45, format="currency")
chart.configure(axis_x=AxisConfig(format="date_short"), axis_y=AxisConfig(format="si"))
```

### Format presets

`label_format="currency"` resolves to a d3-format string in Python before reaching Rust. Preset and raw format are mutually exclusive — setting one clears the other.

### Annotations

Annotations are scene-graph nodes rendered after marks. They accept three coordinate systems: bare numbers (data), `fm.px(n)` (pixels from plot-area origin), `fm.norm(f)` (0–1 fraction of plot area). Auto-placement on callouts uses the existing label-collision system as a best-effort heuristic.

### Structural features

- **SecondaryY** creates an independent Y2 scale and axis (right side by default). Marks bound to it read from the same DataFrame with a different y field.
- **BreakAxis** splits a continuous scale into segments with a visual discontinuity. Marks spanning the gap are clipped into segments. Annotations in the removed range are suppressed with a warning.
- **Inset** embeds a self-contained sub-chart at a bounding box. The inset has independent scales, axes, marks, and configuration. It overlays without reflowing the parent.

### Cascade

Resolution order (highest precedence wins):

1. `chart.override(...)` — spec-path escape hatch
2. Per-channel `axis=`/`legend=` on encoding — "this specific axis"
3. `chart.configure_*()` / `+ Configure(...)` — "all axes on this chart"
4. `chart.theme(...)` — per-chart visual identity
5. `set_default_theme(...)` — session visual identity
6. Rust renderer defaults — built-in fallback

---

## 5. Architecture

### Composition model

`Chart.__add__` dispatches on the right operand's type:

| Operand type | Effect |
|---|---|
| `Chart` | Adds a mark layer (existing) |
| `Configure` | Merges config into chart's config slot |
| `Annotate` / annotation primitive | Appends annotation nodes |
| `SecondaryY` | Attaches y2 scale + mark layer |
| `BreakAxis` | Attaches scale-break spec |
| `Inset` | Attaches sub-chart with bounds |

All operands are immutable frozen dataclasses. `+` returns a new Chart.

### Data flow

1. Python: `.configure_*()` sugar constructs typed config objects, stored on Chart
2. Python: Format presets resolved to d3-format strings
3. Python→Rust (CDI): Config dict, annotation specs, structural specs passed alongside ChartSpec
4. Rust: Layout engine consumes config (axis formatting, grid, padding auto-expansion)
5. Rust: Annotation nodes positioned (auto-placement for callouts)
6. Rust: Structural features applied (scale splitting, inset embedding)
7. Rust: Scene graph emitted with annotations at specified z-order

### Coordinate tagging

`fm.px(n)` and `fm.norm(f)` are lightweight wrapper types that serialize distinctly in the spec dict, allowing Rust to resolve them against the computed plot-area rect at layout time.

---

## 6. Canonical interfaces / data contracts

### Config objects

```python
@dataclass(frozen=True)
class AxisConfig:
    x: bool = True
    y: bool = True
    label_angle: float | None = None
    label_font_size: float | None = None
    label_color: str | None = None
    label_format: str | None = None       # named preset
    label_format_raw: str | None = None   # d3-format escape
    label_overlap: str | None = None      # parity | greedy | rotate | hide
    tick_count: int | None = None
    tick_size: float | None = None
    tick_values: list | None = None
    title_font_size: float | None = None
    title_color: str | None = None
    title_padding: float | None = None
    domain: bool | None = None
    domain_color: str | None = None
    domain_width: float | None = None
    grid: bool | None = None
    grid_color: str | None = None
    grid_dash: list[float] | None = None
    grid_width: float | None = None
    domain_min: float | None = None
    domain_max: float | None = None
    nice: bool | None = None
    zero: bool | None = None

@dataclass(frozen=True)
class LegendConfig:
    orient: str | None = None        # right | left | top | bottom | none
    direction: str | None = None     # vertical | horizontal
    columns: int | None = None
    title_font_size: float | None = None
    label_font_size: float | None = None
    symbol_size: float | None = None
    symbol_type: str | None = None
    gradient_length: float | None = None
    offset: float | None = None
    padding: float | None = None

@dataclass(frozen=True)
class TitleConfig:
    font_size: float | None = None
    font_weight: str | None = None
    anchor: str | None = None        # start | middle | end
    color: str | None = None
    offset: float | None = None
    subtitle_font_size: float | None = None
    subtitle_color: str | None = None

@dataclass(frozen=True)
class GridConfig:
    x: bool | None = None
    y: bool | None = None
    color: str | None = None
    width: float | None = None
    dash: list[float] | None = None
    opacity: float | None = None
    band_colors: list[str] | None = None  # alternating fills; None = disabled

@dataclass(frozen=True)
class PaddingConfig:
    top: float | None = None
    right: float | None = None
    bottom: float | None = None
    left: float | None = None
    auto: bool = True  # auto-expand to fit labels

@dataclass(frozen=True)
class ColorConfig:
    scheme: str | None = None              # categorical
    sequential_scheme: str | None = None
    diverging_scheme: str | None = None
    domain: list | None = None             # explicit color domain
    range: list[str] | None = None         # explicit color range
```

### Chart methods

```python
class Chart:
    def configure_axis(self, **kwargs) -> "Chart": ...
    def configure_legend(self, **kwargs) -> "Chart": ...
    def configure_title(self, **kwargs) -> "Chart": ...
    def configure_grid(self, **kwargs) -> "Chart": ...
    def configure_padding(self, **kwargs) -> "Chart": ...
    def configure_color(self, **kwargs) -> "Chart": ...
    def configure(self, *, axis=None, axis_x=None, axis_y=None, axis_y2=None,
                  legend=None, title=None, grid=None, padding=None, color=None) -> "Chart": ...
    def override(self, **kwargs) -> "Chart": ...
```

### Annotation primitives

```python
# All return frozen dataclass instances composable via + or Annotate([...])
fm.annotation.text(x, y, text, *, font_size=12, color="#333", anchor="start",
                   baseline="middle", angle=0, dx=0, dy=0, z="above_marks")
fm.annotation.arrow(x, y, x2, y2, *, stroke="#333", stroke_width=1.5,
                    head_size=8, curve="straight")
fm.annotation.rect(x1, y1, x2, y2, *, fill, opacity=0.1, stroke=None, corner_radius=0)
fm.annotation.line(x1, y1, x2, y2, *, stroke="#333", stroke_width=1, dash=None)
fm.annotation.span(axis, start, end, *, fill, opacity=0.3, label=None, label_position="top")
fm.annotation.bracket(x1, y1, x2, y2, *, label, direction="above", stroke="#333", tip_length=6)
fm.annotation.callout(x, y, text, *, text_x=None, text_y=None, arrow="curved",
                      padding=4, background="#fff", border_color="#ccc", border_radius=3)
fm.annotation.image(x, y, src, *, width=50, height=50, anchor="center")
```

### Coordinate wrappers

```python
fm.px(value: float) -> PixelCoord
fm.norm(value: float) -> NormCoord
# Bare float/int = data coordinate (default)
```

### Structural features

```python
@dataclass(frozen=True)
class SecondaryY:
    field: str
    mark: str = "line"
    axis: Axis | None = None
    color: str | None = None
    opacity: float | None = None
    scale: Scale | None = None

@dataclass(frozen=True)
class BreakAxis:
    axis: str                          # "x" | "y"
    gap: tuple | list[tuple]           # single (start, end) or list of ranges
    break_size: float = 12             # pixel height of break indicator
    break_style: str = "slash"         # slash | zigzag | wave | gap

@dataclass(frozen=True)
class Inset:
    chart: "Chart"
    bounds: tuple                       # (left, top, right, bottom) in norm or px coords
    border: bool = True
    border_color: str = "#999"
    border_dash: list[float] | None = None
    background: str | None = "#fff"
    shadow: bool = False
    connect_to: tuple | None = None    # data coords of source region
    connect_style: str = "lines"       # bracket | lines | none
```

### Format presets (subset — full table in implementation)

| Preset | d3-format | Example |
|---|---|---|
| `"integer"` | `,.0f` | 1,234 |
| `"percent"` | `.1%` | 45.2% |
| `"currency"` | `$,.0f` | $1,234 |
| `"si"` | `.2s` | 1.2k |
| `"date_short"` | `%b %-d` | Jan 5 |
| `"month_year"` | `%b %Y` | Jan 2026 |

---

## 7. Invariants and constraints

- **Immutability.** All config, annotation, and structural objects are frozen. `+` and `.configure_*()` return new Charts.
- **No callables cross the FFI boundary.** Format presets resolve to strings in Python before Rust sees them. No lambdas, no Python-side post-processing of rendered tick labels.
- **No matplotlib.** Hard constraint (per CLAUDE.md). Not as dependency, backend, or optional extra.
- **Cascade is strict.** Override > per-channel > configure > theme > default theme > Rust defaults. No ambiguity. Later `+` of the same type merges (last wins on key conflict).
- **Unknown override paths error.** `FerrumOverrideError` at render time with closest-match suggestion. No silent no-ops.
- **Backward compatible.** Existing `Theme(...)`, `Axis(...)`, `Legend(...)`, `.properties()`, `.labs()` all continue to work unchanged. New surface is additive.
- **Annotations participate in margin expansion.** An annotation near the edge triggers auto-padding growth (when `PaddingConfig.auto=True`).
- **Insets are independent.** An inset's scales, axes, and config are self-contained. Parent's configure does not cascade into inset.

---

## 8. Key decisions and tradeoffs

| Decision | Rationale | Rejected alternative |
|---|---|---|
| Composable layer model (`+` operator) | Matches existing mark-layering; keeps Chart lean; features compose naturally | Flat method extension (bloats Chart, features can't compose) |
| Named format presets over callables | Stays in Rust; no FFI complexity; serializable | `FuncFormatter`-style callables (break Rust computation invariant) |
| `.configure_*()` sugar + `Configure` object | Discoverability for one-offs + reusability for repeated configs | Object-only (too verbose for simple cases) |
| Override as escape hatch, not primary API | Prevents untyped sprawl; docs steer to typed surface | No escape hatch (users hit walls), or override-first (no type safety) |
| Auto-placement as best-effort heuristic | Good enough for 80%; explicit coords override | Mandatory explicit placement (poor ergonomics), Python-side layout (breaks Rust invariant) |
| `BreakAxis` clips spanning marks | Semantically correct — a bar from 0→1000 in a broken scale must show as two segments | Hide marks in gap (loses data), error on spanning marks (too strict) |
| Inset overlays without reflow | Simple, predictable positioning; matches user mental model of "pasted on top" | Reflow-aware inset (complex layout engine changes, unclear benefit) |
| Per-axis targeting via `axis_x`/`axis_y` in `.configure()` | Clean separation when X and Y need different settings | Single AxisConfig with `x`/`y` booleans only (less readable for asymmetric cases) |

---

## 9. Acceptance criteria

- All 6 config objects accepted by `.configure_*()` methods and `+ Configure(...)`
- Format presets resolve correctly for all ~20 named presets (verified by unit tests comparing output to expected d3-format strings)
- All 8 annotation primitives render in SVG at correct data/pixel/normalized positions
- Callout auto-placement avoids mark overlap in a standard scatter plot (visual golden test)
- SecondaryY renders independent right-side axis with correct scale mapping
- BreakAxis visually splits scale and clips a spanning mark into segments
- Inset renders with independent axes and does not affect parent layout
- Override with valid path applies at render time; unknown path raises `FerrumOverrideError`
- Cascade precedence verified: per-channel axis beats configure, configure beats theme
- All existing tests continue to pass (backward compat)
- `cargo test` passes
- Documentation: conceptual guide, 7 concept pages, 12 recipes, migration table all present

---

## 10. Validation strategy

- **Unit tests:** Format preset → d3-format resolution; config object construction and validation; cascade precedence (mock render with conflicting config at multiple levels)
- **Golden SVG tests:** One golden per annotation primitive; one per structural feature; one combined chart exercising configure + annotations + secondary Y
- **Visual inspection:** All goldens rasterized to PNG via `snapshot-goldens.py` and visually confirmed before commit (per CLAUDE.md golden inspection rule)
- **Integration tests:** End-to-end chart with `.configure_axis(format="currency")` renders correct tick labels; `+ fm.annotation.callout(...)` appears at data coordinates
- **Override validation:** Test that unknown paths error; test that valid paths apply; test deprecation warning for paths with typed equivalents
- **Backward compat:** Existing test suite passes without modification

---

## 11. Open questions

- **Ordinal format preset:** Requires custom Rust implementation (not a d3-format string). Confirm this is worth the Rust complexity vs. deferring to a future pass.
- **Auto-placement algorithm:** Which heuristic for callout positioning — greedy force-directed, or simpler quadrant-based? Affects quality vs. implementation cost.
- **`BreakAxis` + `SecondaryY` interaction:** Should a broken primary Y affect the secondary Y scale? Initial position: no — they are independent scales. Confirm.
