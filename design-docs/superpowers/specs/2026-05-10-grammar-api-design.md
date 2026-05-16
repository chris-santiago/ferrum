# Phase 8a — Grammar API Surface (Python): Design Spec

**Date:** 2026-05-10
**Phase:** 8a (Grammar API surface, part 1)
**Phase slug:** `grammar-api`
**Depends on:** Phase 4 (Scale engine), Phase 5 (Stat engine — `Bin`/`Kde`/`Smooth`/`Aggregate`/`Summary`), Phase 6 (Layout — `FacetSpec`), Phase 7 (Static renderer — `render_svg`/`render_png`, `ThemeInputs`, `Encoding.color`)
**Unblocks:** Phase 8b (composite + heavy statistical marks), Phase 9 (convenience API), Phase 10 (model diagnostics), Phase 11 (interactive renderer)

---

## §1 Goal

Ship the user-facing Python grammar API on top of Phase 7's renderer. After Phase 8a, the following idioms work:

```python
import ferrum as fr
import polars as pl

df = pl.read_csv("iris.csv")

# single-layer
fr.Chart(df).mark_point().encode(
    x="sepal_length", y="sepal_width", color="species"
).show()

# multi-layer (+) — single coordinate system, layered marks
points = fr.Chart(df).mark_point().encode(x="sepal_length", y="sepal_width", color="species")
fit    = fr.Chart(df).mark_smooth(method="loess").encode(x="sepal_length", y="sepal_width")
(points + fit).show()

# concat (|, &)
hist = fr.Chart(df).mark_histogram().encode(x="sepal_length")
kde  = fr.Chart(df).mark_density().encode(x="sepal_length")
(hist | kde).save("compare.svg")

# faceting
fr.Chart(df).mark_point().encode(x="sepal_length", y="sepal_width").facet(col="species").show()

# theme as value
my_theme = fr.themes.dark.update(font_family="Inter", title_font_size=18)
chart.theme(my_theme).save("dark.png")

# theme as process default (notebook ergonomic)
fr.set_default_theme(fr.themes.dark)  # also returns a contextmanager
```

The Python layer is the declaration API; the Rust layer (Phase 7's `render_svg`/`render_png`) does the rendering. Phase 8a is mostly Python with additive Rust changes for multi-layer support, three new encoding channels (size/shape/opacity), a CoordFlip swap in `prepare.rs`, six more categorical palettes, and an SVG compositor for `|`/`&`.

---

## §2 Scope

### In scope (Phase 8a)
- `ferrum.Chart` and `ferrum.Layer` Python classes.
- All 31 encoding channel classes from `ferrum-spec.md §3.2`: positional (X, Y, X2, Y2, XError, YError, XError2, YError2, Theta, Radius — 10), appearance (Color, Fill, Stroke, Opacity, FillOpacity, StrokeOpacity, StrokeWidth, StrokeDash, Size, Shape, Angle — 11), text/detail/tooltip (Text, Detail, Tooltip, TooltipField, Href, Description, Key — 7; `TooltipField` is a Tooltip value-helper but ships as a constructible class), facet (Facet, FacetRow, FacetCol — 3).
- 8 primitive `mark_*()` methods (point, line, bar, area, rule, text, tick, rect) + 3 simple statistical marks (`mark_density`, `mark_histogram`, `mark_smooth` without CI band).
- Channels actually rendered: `x`, `y`, `color`, **and new in 8a: `size`, `shape`, `opacity`**.
- Channel kwargs honored: `type`, `bin`, `aggregate`, `scale`, `title`. Other kwargs (`axis`, `legend`, `sort`, `stack`, `impute`, `scheme`, `format`) accepted, stored on `EncodingSpec`, with one-time `UserWarning` per (channel-class, kwarg) per process.
- Shorthand strings: `"field"`, `"mean(field)"`, `"field:Q"` parsed at `.encode()` time.
- Composition: `+` (multi-layer in one coord system) via `ChartSpec.layers: Option<Vec<Layer>>` (additive); `|` and `&` via Python-orchestrate-Rust SVG compositor.
- `Theme` value class + 8 built-ins (`default`, `minimal`, `dark`, `publication`, `economist`, `fivethirtyeight`, `solarized_light`, `solarized_dark`) + `Chart.theme(t)` per-chart override + `set_default_theme(t)` returning a context manager (single primitive, dual usage).
- Faceting: `Facet`/`FacetRow`/`FacetCol` channel classes + `Chart.facet()` method.
- Annotations: `annotate_hline`, `annotate_vline`, `annotate_rect`, `annotate_text` (sugar over primitive marks with inline 1-row tables).
- `CoordFlip()` (swap X/Y axis roles) via `Chart.coord()` method.
- Data inputs: polars (direct CDI), pyarrow `Table`/`RecordBatch` (passthrough), pandas + modin + cuDF + dask + ibis (via narwhals), dict-of-arrays, list-of-records, numpy 2D.
- `.show()` (Jupyter inline `_repr_svg_`/`_repr_html_` + browser fallback via `webbrowser.open`), `.show_svg()`, `.show_png()`, `.save(path)` (svg/png from extension).
- Updates to `ferrum-spec.md` (dated notes for deferrals) and to `CLAUDE.md` (theme-contextvars exception).

### Deferred to Phase 8b
- 4 composite marks: `mark_boxplot`, `mark_errorbar`, `mark_errorband`, `mark_ribbon`.
- 7 heavy statistical marks: `mark_contour`, `mark_violin`, `mark_qq`, `mark_raster`, `mark_swarm`, `mark_hex`, `mark_function`.
- ~7 new Phase 5 transforms: `Outliers`, `ErrorExtent`, `Contour`, `QQ`, `Raster`, `Hex`, `Swarm`, `BoxStats`, `Violin`.
- New SVG primitives in `SvgBuffer`: `image()` (raster), `polygon()` / `path` for hex/contour, beeswarm collision-resolving point placement.
- `mark_smooth` CI band (depends on Phase 8b's ribbon mark).

### Deferred further (Phase 9+)
- Figure-level functions (`displot`, `lmplot`, `roc_chart`, `pairplot`, etc.) — Phase 9.
- `ModelSource`, `ComparedModelSource`, model-diagnostic marks — Phase 10.
- Selections (`selection_point`, `selection_interval`) and interactivity — Phase 11.
- File-path inputs (`Chart("file.csv")`) — Phase 9.
- `JointChart`, `RepeatChart`, `RepeatSpec` — Phase 9.
- AUCLabel, OutlierLabel annotations — Phase 9/10.
- Coord systems other than CoordFlip (`CoordCartesian.xlim/ylim`, `CoordPolar`, `CoordGeo`, `CoordFixed`) — Phase 9+.
- Sixel terminal output, HTML wrapper output — Phase 9+.
- The deferred channel kwargs (`axis`, `legend`, `sort`, `stack`, `impute`, `scheme`, `format`, `formatType`) honored at the renderer — Phase 9.
- `mark_arc`, `mark_image`, `mark_geoshape`, `mark_segment`, `mark_label` — Phase 9+.
- `Chart.add_selection()` / `.interactive()` raise `NotImplementedError` in 8a; Phase 11 implements.

---

## §3 Architecture

### §3.1 Module layout

**Python (`src/ferrum/`):**

```
src/ferrum/
  __init__.py             # public re-exports: Chart, Layer, mark_*, Theme, themes, channels, ...
  _core.pyi               # type stubs (existing, extended for 8a additions)
  _coerce.py              # data → pyarrow.Table normalization (narwhals + ferrum branches)
  _shorthand.py           # parse "mean(field)", "field:Q" → (field, type, aggregate)
  _warn.py                # warn-once registry keyed by (channel, kwarg) per-process
  chart.py                # Chart class (data, layers, theme, coord, facet, etc.)
  layer.py                # Layer class
  marks/
    __init__.py           # mark_point, mark_line, ..., mark_text, mark_tick, mark_rect (8 primitives)
    statistical.py        # mark_density, mark_histogram, mark_smooth (3 in 8a)
    base.py               # MarkBase (kwargs: stroke, fill, opacity, corner_radius, ...)
    deferred.py           # mark_boxplot/mark_violin/etc. raise NotImplementedError("Phase 8b")
  encoding/
    __init__.py           # re-exports
    positional.py         # X, Y, X2, Y2, XError, YError, XError2, YError2, Theta, Radius
    appearance.py         # Color, Fill, Stroke, Opacity, FillOpacity, StrokeOpacity,
                          # StrokeWidth, StrokeDash, Size, Shape, Angle
    text.py               # Text, Detail, Tooltip, TooltipField, Href, Description, Key
    facet.py              # Facet, FacetRow, FacetCol
    base.py               # ChannelBase: __init__ accepts kwargs, validates, builds EncodingSpec
  composition.py          # LayerChart (.__add__), HConcatChart (.__or__), VConcatChart (.__and__)
  themes/
    __init__.py           # Theme value class + .update() + 8 builtins re-exported
    builtins.py           # default, minimal, dark, publication, economist, fivethirtyeight,
                          # solarized_light, solarized_dark
    _defaults.py          # contextvars-backed default theme stack;
                          # set_default_theme(), theme_context()
  annotations.py          # annotate_hline, annotate_vline, annotate_rect, annotate_text
  coord.py                # CoordFlip (8a); CoordCartesian/Polar/Geo/Fixed raise NotImplementedError
  display.py              # Chart.show / show_svg / show_png / save / _repr_svg_ / _repr_html_
```

**Rust (`crates/ferrum-core/src/`) — additive changes only:**

```
crates/ferrum-core/src/
  spec/
    chart.rs            # ChartSpec gains `layers: Option<Vec<Layer>>` + `coord: Option<CoordKind>`
    encoding.rs         # EncodingSpec gains typed Option<> fields for deferred kwargs (axis,
                        # legend, sort, stack, impute, scheme, format, formatType)
                        # + `scale: Option<ScaleSpec>` and `title: Option<String>` (honored)
    layer.rs            # NEW. struct Layer { mark, encoding, transforms } (per-layer override)
    mark.rs             # Mark enum unchanged; size/shape/opacity stay encoding-driven
    coord.rs            # NEW. enum CoordKind { Cartesian, Flip } (only Flip honored in 8a)
  render/
    prepare.rs          # Updated: handle multi-layer when layers.is_some(); apply CoordFlip swap
    scale_resolve.rs    # Updated: honor explicit Scale; build size/shape/opacity scales
    palette.rs          # 6 more categorical palettes (tableau10, set1, set2, paired, pastel, dark2)
    marks/
      point.rs          # Updated: respect size + shape + opacity per-row
      ...               # other marks unchanged unless they newly honor opacity
    compositor.rs       # NEW. SVG concat helpers: hconcat(svgs, spacing) -> str,
                        # vconcat(svgs, spacing) -> str.
    binding.rs          # Add compose_svg_horizontal / compose_svg_vertical Python entry points
```

### §3.2 Data flow (single chart)

```
Chart.show_svg() →

  1. data_table = _coerce.to_arrow(self._data)             # narwhals + ferrum branches
  2. theme = self._theme or themes._defaults.get_default_theme()
     theme_inputs_dict = theme.to_theme_inputs_dict()
  3. spec = ChartSpec(
         mark=self._mark,
         x=..., y=..., color=...,
         data="default",
         transforms=self._transforms,
     )
     # If self._layers, also pass layers=[...]
     # If self._facet, also pass facet=...
     # If self._coord, also pass coord=...
  4. viewport = (self._width or theme.width or DEFAULT_WIDTH,
                 self._height or theme.height or DEFAULT_HEIGHT)
  5. svg_string = ferrum._core.render_svg(
         spec, data_table, viewport=viewport, theme=theme_inputs_dict, config=...
     )
  6. return svg_string
```

`Chart.show_png()` is identical except step 5 calls `render_png` and returns bytes.

### §3.3 Data flow (multi-layer same-data, `+`)

```
(chart_a + chart_b).show_svg() →

  1. assert chart_a._data is chart_b._data    # or pyarrow-equal; else fall through to §3.4
  2. merged_layers = [
         Layer(mark=chart_a._mark, encoding=chart_a._encoding, transforms=chart_a._transforms),
         Layer(mark=chart_b._mark, encoding=chart_b._encoding, transforms=chart_b._transforms),
     ]
  3. spec = ChartSpec(
         mark=chart_a._mark,                  # primary; renderer uses for fallback
         encoding=chart_a._encoding,
         layers=merged_layers,
         data="default",
         facet=chart_a._facet,                # facet applies to whole compound
         coord=chart_a._coord,
     )
  4. render_svg(spec, chart_a._data_table, ...)
     # Rust: scales built from union of all layers' encodings
     # Rust: per panel, iterate layers in order; each draws into the panel's plot_area
```

Rule: the **primary chart** (left operand of `+`) supplies data, theme, facet, coord, viewport. The right operand contributes only its mark, encoding, transforms. If the right operand has a different theme/facet/coord, a `UserWarning` fires explaining that the primary's wins.

### §3.4 Data flow (concat `|`/`&` and mixed-data layers)

```
(chart_a | chart_b).show_svg() →

  1. svg_a = chart_a.show_svg()    # full pipeline; each child is independent
  2. svg_b = chart_b.show_svg()
  3. svg_out = ferrum._core.compose_svg_horizontal(
         [svg_a, svg_b], spacing=10.0, align="top",
     )
     # Rust compositor:
     #   parses each <svg> root for width/height (regex on root attrs — guaranteed deterministic
     #   from our SvgBuffer)
     #   strips inner <defs>/font-face block from all but the first (no duplication)
     #   wraps each child's body in <g transform="translate(x_offset, y_offset)">
     #   emits an outer <svg> with combined viewport
  4. return svg_out
```

Mixed-data `+` (i.e. `chart_a._data is not chart_b._data`) falls through to `chart_a | chart_b` after a warning: *"Chart `+` with differing data sources renders as horizontal concatenation; for true layered overlay, ensure both layers share the same DataFrame."*

### §3.5 Multi-layer ChartSpec extension (additive, back-compat)

```rust
#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChartSpec {
    #[serde(default)]
    pub data: DataRef,
    pub mark: Mark,
    #[serde(default)]
    pub encoding: Encoding,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub transforms: Vec<TransformSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub facet: Option<FacetSpec>,
    // NEW in Phase 8a:
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layers: Option<Vec<Layer>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coord: Option<CoordKind>,
}

pub struct Layer {
    pub mark: Mark,
    #[serde(default)]
    pub encoding: Encoding,            // overrides chart-level encoding when fields are Some
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub transforms: Vec<TransformSpec>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum CoordKind {
    Cartesian,    // explicit no-op for Phase 8a
    Flip,         // swap x/y
}
```

When `layers.is_none()`: status quo — renderer uses `mark` + `encoding` as a single layer. **All existing Phase 3–7 goldens stay byte-identical.** When `layers.is_some()`: renderer ignores top-level `mark`+`encoding` for drawing (they remain valid as the "primary" defaults inherited by layers that omit fields) and iterates the layer list inside each panel, sharing x/y/color scales by default.

### §3.6 Per-mark style overrides

Phase 7's `MarkStyle` is computed inside `prepare.rs` from theme + spec defaults. Phase 8a adds per-mark constant overrides (e.g. `mark_point(size=100, stroke="red", opacity=0.5)`). Two design choices:

- **Adopted:** `ChartSpec` and `Layer` each gain `mark_style: Option<MarkKwargsSpec>` where `MarkKwargsSpec` carries optional constant overrides for every per-mark visual property (size, stroke, fill, opacity, corner_radius, stroke_width, stroke_dash, font_size, font_weight, align, baseline, dx, dy, angle, ...). All fields default-omit.

```rust
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct MarkKwargsSpec {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stroke: Option<String>,        // hex or named color, parsed via render::color::from_hex_str
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fill: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opacity: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub corner_radius: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stroke_width: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stroke_dash: Option<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub font_size: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub font_weight: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub align: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dx: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dy: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub angle: Option<f64>,
}
```

`prepare.rs` is updated: when computing `MarkStyle` for a mark, after applying theme defaults, override each field with the corresponding `mark_style.X.unwrap_or(default)` value if `mark_style.is_some()`. Per-mark draw fns are unchanged — they continue to consume `MarkStyle`.

Resolution priority: `mark_kwargs (Layer) > mark_kwargs (Chart) > theme defaults`.

When an encoding for the same channel is also present (e.g. user passes both `mark_point(size=100)` AND `encode(size="weight")`), the encoding wins for any row whose channel value is non-null; the mark_kwargs constant is the fallback. This is the same precedence Vega-Lite uses and matches user intuition (data binding > styling constant).

---

## §4 Per-component contracts

### §4.1 `Chart`

```python
class Chart:
    def __init__(
        self,
        data: object | None = None,
        *,
        width: int | str | None = None,
        height: int | str | None = None,
        title: str | None = None,
        description: str | None = None,
    ): ...

    # Mark methods (each returns a NEW Chart; instances are immutable values)
    def mark_point(self, **kwargs) -> "Chart": ...
    def mark_line(self, **kwargs) -> "Chart": ...
    def mark_bar(self, **kwargs) -> "Chart": ...
    def mark_area(self, **kwargs) -> "Chart": ...
    def mark_rule(self, **kwargs) -> "Chart": ...
    def mark_text(self, **kwargs) -> "Chart": ...
    def mark_tick(self, **kwargs) -> "Chart": ...
    def mark_rect(self, **kwargs) -> "Chart": ...
    # statistical (8a)
    def mark_density(self, **kwargs) -> "Chart": ...     # → mark_area + Kde transform
    def mark_histogram(self, **kwargs) -> "Chart": ...   # → mark_bar + Bin transform
    def mark_smooth(self, **kwargs) -> "Chart": ...      # → mark_line + Smooth transform

    def encode(self, **channels) -> "Chart": ...
    def transform(self, *transforms) -> "Chart": ...
    def facet(self, field=None, *, row=None, col=None, ncols=None, nrows=None) -> "Chart": ...
    def theme(self, theme: "Theme") -> "Chart": ...
    def coord(self, coord) -> "Chart": ...               # CoordFlip in 8a
    def properties(self, **kwargs) -> "Chart": ...
    def layer(self, *layers: "Layer") -> "Chart": ...

    # Output
    def show(self) -> None: ...
    def show_svg(self) -> str: ...
    def show_png(self) -> bytes: ...
    def save(self, path, *, format=None, scale=2.0, **render_kwargs) -> None: ...
    def to_spec(self) -> "ChartSpec": ...
    def to_json(self, *, indent: int | None = None) -> str: ...

    # Operators
    def __add__(self, other: "Chart") -> "Chart": ...        # multi-layer / LayerChart wrapper
    def __or__(self, other: "Chart") -> "HConcatChart": ...
    def __and__(self, other: "Chart") -> "VConcatChart": ...

    # Jupyter rich display
    def _repr_svg_(self) -> str: ...
    def _repr_html_(self) -> str: ...

    # Phase 11 stubs
    def add_selection(self, *selections) -> "Chart":
        raise NotImplementedError("selections require .interactive() — Phase 11")
    def interactive(self) -> "Chart":
        raise NotImplementedError("interactive renderer — Phase 11")
```

**Immutability rule:** every method returns a new `Chart`. The internal spec dict is deep-copied per fluent call. This makes chains free of aliasing surprises.

### §4.2 `Layer`

```python
class Layer:
    def __init__(
        self,
        data: object | None = None,
        mark: object | None = None,             # Mark instance OR string "point"
        *,
        encoding: dict | None = None,
        transforms: list | None = None,
    ): ...
```

Used internally by `Chart.__add__` to build the multi-layer spec; users rarely construct it directly. When `data` is None, layer inherits the chart's data. When `data` differs from the chart's, the chart routes `+` through the SVG compositor (per §3.4).

### §4.3 Encoding channels

All channels share `ChannelBase`:

```python
class ChannelBase:
    _channel_name: ClassVar[str]               # "x", "y", "color", ...
    _renders_in_phase_8a: ClassVar[bool]       # x/y/color/size/shape/opacity = True; rest = False
    _honored_kwargs: ClassVar[frozenset[str]]  # {"type", "bin", "aggregate", "scale", "title"} typically

    def __init__(self, field: str, **kwargs):
        self.field = field
        self._kwargs = kwargs
        self._validate()                        # type-checks each kwarg
        for k in kwargs:
            if k not in self._honored_kwargs:
                _warn.warn_once(self._channel_name, k)

    def to_encoding_spec_dict(self) -> dict:
        """Returns kwargs for the Rust EncodingSpec constructor."""
        return {
            "field": self.field,
            "type_": self._kwargs.get("type"),
            "scale": self._kwargs.get("scale"),
            "title": self._kwargs.get("title"),
            # ...the 8 deferred fields (axis, legend, sort, stack, impute, scheme,
            # format, formatType), set to whatever was passed (None if absent)
        }

    def to_implicit_transforms(self) -> list:
        """Returns Bin/Aggregate transform objects to append to spec.transforms."""
        out = []
        if (b := self._kwargs.get("bin")):
            out.append(Bin(self.field, **(b if isinstance(b, dict) else {})))
        if (agg := self._kwargs.get("aggregate")):
            out.append(Aggregate([AggregateOp(self.field, agg, f"{agg}_{self.field}")]))
        return out
```

Per-channel subclasses (`X`, `Y`, `Color`, `Size`, `Shape`, ...) just set `_channel_name`, `_renders_in_phase_8a`, `_honored_kwargs`. ~20 LOC each.

When `_renders_in_phase_8a` is False (e.g. `Stroke`, `StrokeDash`, `Text`, `Tooltip`), the channel object is still constructed and stored on the spec, but the renderer ignores it — and a one-time warning fires per `(channel, render call)`.

**Shorthand string parsing** (`encode(x="mean(price):Q")`) lives in `_shorthand.py`:
```
parse_shorthand("mean(price):Q") → ("price", "Q", "mean")
parse_shorthand("price")          → ("price", None, None)
parse_shorthand("price:Q")        → ("price", "Q", None)
parse_shorthand("count()")        → (None, None, "count")
```
Result is fed into `X(field, type=type, aggregate=agg)` etc.

### §4.4 Marks

Each `Chart.mark_*()` method desugars consistently:

```python
def mark_point(self, **mark_kwargs) -> "Chart":
    return self._with(mark="point", mark_kwargs=mark_kwargs)

# Statistical sugar:
def mark_density(self, *, bandwidth="scott", kernel="gaussian", n=512, ...) -> "Chart":
    # 1. capture the channel that needs density (typically x)
    # 2. append Kde transform; replace the encoding field with the kde output column
    # 3. set mark to "area"
    return self._with(
        mark="area",
        transforms=self.transforms + [Kde(field, bandwidth=bandwidth, n=n, ...)],
        encoding_remap={"y": "density"},
    )
```

`mark_histogram` → `Bin` transform + `mark="bar"`.
`mark_smooth` → `Smooth` transform + `mark="line"` (CI band deferred to 8b — needs ribbon mark).

`MarkBase`-style kwargs (`size`, `stroke`, `fill`, `opacity`, `corner_radius`, `stroke_width`, `stroke_dash`, etc.) are stored in `mark_kwargs` and applied as MarkStyle overrides at the renderer boundary.

### §4.5 Theme

```python
class Theme:
    """Immutable value class; ~50 properties from §3.13."""
    def __init__(self, **kwargs):
        self._props = {k: v for k, v in kwargs.items() if v is not None}
    def update(self, **kwargs) -> "Theme":
        return Theme(**{**self._props, **kwargs})
    def to_theme_inputs_dict(self) -> dict:
        """Maps to ThemeInputs Rust struct fields. Unknown keys passed through."""
        ...
    def __eq__(self, other) -> bool: ...
    def __hash__(self) -> int: ...

# 8 builtins (themes/builtins.py)
default            = Theme()  # all None → Rust defaults
minimal            = Theme(grid=False, axis_line=False, padding=20)
dark               = Theme(background="#1a1a2e", font_color="#e6e6e6", axis_line_color="#666",
                           grid_color="#333", color_scheme="okabe_ito_dark", title_color="#fff")
publication        = Theme(background=None, grid=False, color_scheme="tableau10", font_family="Inter",
                           title_font_weight="bold", axis_line_color="#000", font_color="#000")
economist          = Theme(background="#d3e0e6", font_family="Inter", title_color="#c00",
                           grid_color="#b0c4cc", axis_line=False)
fivethirtyeight    = Theme(background="#f0f0f0", color_scheme="redblue", grid_color="#cccccc",
                           axis_line=False, font_family="Inter")
solarized_light    = Theme(background="#fdf6e3", font_color="#586e75", grid_color="#eee8d5", ...)
solarized_dark     = Theme(background="#002b36", font_color="#93a1a1", grid_color="#073642", ...)

# Defaults stack (themes/_defaults.py)
_default_theme: contextvars.ContextVar[Theme] = ContextVar("_ferrum_default_theme", default=default)

def set_default_theme(theme: Theme) -> "_DefaultThemeCM":
    """Set the process-default theme. Returns a context manager that restores
    the previous default on __exit__. Idempotent if used as a fire-and-forget call."""
    token = _default_theme.set(theme)
    return _DefaultThemeCM(token)

def get_default_theme() -> Theme:
    return _default_theme.get()
```

Theme resolution order at render: explicit `Chart.theme(t)` > `set_default_theme()` value > `Theme()` (Rust defaults).

**Theme builtin sourcing:** RGB constants for `dark`, `fivethirtyeight`, `economist`, `solarized_*` reference vega-lite theme JSONs where the spec is ambiguous; the mapping is documented in a comment block in `builtins.py`.

### §4.6 Composition

```python
class HConcatChart:
    def __init__(self, charts: list[Chart], *, spacing: float = 10.0): ...
    def show_svg(self) -> str:
        svgs = [c.show_svg() for c in self.charts]
        return ferrum._core.compose_svg_horizontal(svgs, spacing=self.spacing)
    def show_png(self) -> bytes: ...
    # save, show, _repr_*_, __or__, __and__ all delegate

class VConcatChart: ... # analogous, vertical
class LayerChart(Chart):
    """Used when chart_a + chart_b has differing data; falls through to hconcat with warning."""
```

**Operator precedence note:** Python binds `&` tighter than `|`, so `a | b & c` parses as `a | (b & c)`. Documented in `Chart.__or__` docstring and tested both ways.

### §4.7 Faceting

```python
# encoding/facet.py
class Facet(ChannelBase): _channel_name = "facet"
class FacetRow(ChannelBase): _channel_name = "facet_row"
class FacetCol(ChannelBase): _channel_name = "facet_col"

# Chart.facet() — sugar
def facet(self, field=None, *, row=None, col=None, ncols=None, nrows=None) -> "Chart":
    """Single-dim wrap (field+ncols/nrows) or grid (row, col)."""
    # builds FacetSpec (Phase 6 Rust struct) with FacetMode::Wrap{ncols} or Grid{row, col}
```

Phase 6+7 already render facets; this is just the Python sugar wrapping `FacetSpec`.

### §4.8 Annotations

```python
def annotate_hline(y: float, *, label=None, stroke=None, stroke_dash=None) -> "Chart":
    """Returns a single-mark Chart wrapping a mark_rule at constant y."""
def annotate_vline(x: float, *, label=None, stroke=None, stroke_dash=None) -> "Chart":
def annotate_rect(x1, x2, y1, y2, *, fill=None, opacity=0.1, label=None) -> "Chart":
    """mark_rect with explicit X/Y/X2/Y2."""
def annotate_text(x, y, text: str, *, dx=0, dy=0, align="center", baseline="middle",
                  font_size=None, color=None, angle=None) -> "Chart":
```

Each returns a `Chart` you can add via `+`: `scatter + annotate_hline(0)`. They're sugar over primitive marks with an inline 1-row data table.

### §4.9 CoordFlip

```python
class CoordFlip:
    """No state; marker class."""
class CoordCartesian:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("CoordCartesian is planned for Phase 9")
class CoordPolar:  # same
class CoordGeo:    # same
class CoordFixed:  # same

# Chart.coord(CoordFlip()) sets ChartSpec.coord = CoordKind::Flip in Rust
# Renderer's prepare.rs swaps the X and Y scale bindings before drawing
```

Implementation: ~30 LOC in `prepare.rs`. The `marks/*.rs` draw functions see swapped scales transparently.

### §4.10 Data coercion

```python
# _coerce.py
def to_arrow_table(data: object) -> "pyarrow.Table":
    """Normalize any supported input to a pyarrow.Table."""
    import pyarrow as pa
    import polars as pl

    if data is None:
        raise ValueError("Chart(data=None) requires per-layer data — not yet supported in Phase 8a")

    # Fast paths (no narwhals)
    if isinstance(data, pl.DataFrame):
        return data.to_arrow()                  # zero-copy CDI
    if isinstance(data, pa.Table):
        return data
    if isinstance(data, pa.RecordBatch):
        return pa.Table.from_batches([data])

    # Dict / list / numpy
    if isinstance(data, dict):
        return pa.Table.from_pydict(data)
    if isinstance(data, list):
        return pa.Table.from_pylist(data)
    if _is_numpy_2d(data):
        return _from_numpy_2d(data)             # auto-named col_0, col_1, ...
    if _is_numpy_1d(data):
        raise TypeError("1D numpy arrays need column names — pass `Chart({'value': arr})`")

    # Everything else: try narwhals
    try:
        import narwhals as nw
        nw_df = nw.from_native(data, eager_only=True)
        return nw_df.to_arrow()
    except ImportError:
        raise ImportError(
            f"Input type {type(data).__name__} requires narwhals. "
            f"Install with `pip install narwhals`."
        )
    except (TypeError, NotImplementedError) as e:
        raise TypeError(
            f"Unsupported data type: {type(data).__name__}. "
            f"Supported: polars, pyarrow, pandas, modin, cuDF, dask, ibis, dict, list, numpy 2D. "
            f"Got: {e}"
        ) from e
```

**Narwhals + `pyarrow.RecordBatch`:** verified at plan stage; if `nw.from_native` rejects RecordBatch, ferrum's `_coerce` already converts at the boundary (the `isinstance(data, pa.RecordBatch)` branch above). Cheap fallback either way.

`narwhals` becomes a hard runtime dep (pure Python, lightweight) — added to `pyproject.toml`.

---

## §5 Algorithm — render pipeline

### §5.1 Single chart

```
Chart.show_svg() →

1. data_table = _coerce.to_arrow(self._data)
   → pyarrow.Table (validated, normalized)

2. theme = self._theme or themes._defaults.get_default_theme()
   theme_inputs_dict = theme.to_theme_inputs_dict()
   # ~50 keys, sent to Rust as a Python dict (existing Phase 7 binding shape)

3. spec = ferrum._core.ChartSpec(
       mark=self._mark,
       x=self._encoding.get("x"),         # EncodingSpec or None
       y=self._encoding.get("y"),
       color=self._encoding.get("color"),
       data="default",
       transforms=self._transforms,        # may include channel-derived Bin/Aggregate
   )
   # If self._layers, also pass layers=[...]
   # If self._facet, also pass facet=...
   # If self._coord, also pass coord=...

4. viewport = (self._width or theme.width or DEFAULT_WIDTH,
               self._height or theme.height or DEFAULT_HEIGHT)

5. svg_string = ferrum._core.render_svg(
       spec, data_table, viewport=viewport, theme=theme_inputs_dict, config=...
   )

6. return svg_string
```

`Chart.show_png()` is identical except step 5 calls `render_png` and returns `bytes`.

### §5.2 Multi-layer same-data (`+`)

Per §3.3. The primary chart (left operand of `+`) supplies data, theme, facet, coord, viewport. Right operand's theme/facet/coord/viewport are ignored with a warning.

### §5.3 Concat (`|`/`&`) and mixed-data layers

Per §3.4. Each child chart renders independently via the full Phase 7 pipeline; `compose_svg_horizontal` / `compose_svg_vertical` stitch the resulting SVG strings deterministically.

### §5.4 Theme resolution

```
At render time:
  if self._theme is not None:
      theme = self._theme                              # explicit per-chart wins
  else:
      theme = themes._defaults.get_default_theme()     # contextvar default
  theme_inputs_dict = theme.to_theme_inputs_dict()
```

### §5.5 Encoding kwarg desugaring

```
chart.encode(x=X("price", bin=True, aggregate="mean"), y="weight") →

Inside Chart.encode:
  for channel_name, channel in channels.items():
      if isinstance(channel, str):
          channel = ChannelClass(*parse_shorthand(channel))   # via _shorthand
      elif not isinstance(channel, ChannelBase):
          raise TypeError(...)

      # Honored: type, scale, title → flow to EncodingSpec
      # Honored: bin, aggregate → flow to spec.transforms
      encoding_spec_kwargs = channel.to_encoding_spec_dict()
      implicit_transforms = channel.to_implicit_transforms()

      new_encoding[channel_name] = EncodingSpec(**encoding_spec_kwargs)
      new_transforms.extend(implicit_transforms)
```

### §5.6 Facet pipeline

Already implemented Phase 6+7. Phase 8a constructs `FacetSpec` from Python `Facet`/`FacetRow`/`FacetCol` channel objects:

```
Chart.facet(col="species") → FacetSpec(field="species", mode=FacetMode::Wrap{ncols: 3})
Chart.facet(row="year", col="species") → FacetSpec mode=Grid{row, col}
```

### §5.7 CoordFlip pipeline

```
Chart.coord(CoordFlip()) sets spec.coord = CoordKind::Flip
In prepare.rs, when coord is Flip:
    swap x_scale ↔ y_scale before passing to scale_resolve / draw
    swap axis sides accordingly
    layout.compute_layout sees the swap because axes_input is built post-swap
```

Roughly 30 LOC in `prepare.rs`; no per-mark code changes (marks consume scales, not axes).

### §5.8 Constants

| Constant | Value | Purpose |
|---|---|---|
| `DEFAULT_WIDTH` | `600` | Default chart width when neither `Chart(width=...)`, `Theme(width=...)`, nor `RenderConfig(width=...)` is set |
| `DEFAULT_HEIGHT` | `400` | Same for height |
| `DEFAULT_HCONCAT_SPACING` | `10.0` | Pixels between hconcat panels |
| `DEFAULT_VCONCAT_SPACING` | `10.0` | Pixels between vconcat panels |
| `WARN_ONCE_KEY_FORMAT` | `"{channel}.{kwarg}"` | Key for `_warn` deduplication registry |

---

## §6 Error policy

Mirrors Phase 5/6/7 hybrid: structural errors raise `PyValueError`/`PyTypeError`/`NotImplementedError`; geometric/semantic edge cases warn and proceed.

| Class | Trigger | Response |
|---|---|---|
| **Structural** | Unknown mark name; channel field references absent column; theme value out of range; data type unsupported; numpy 1D without column names; `Chart(data=None)` without per-layer data; selection used in 8a; `.interactive()` called; CoordPolar/CoordGeo/CoordFixed/CoordCartesian used; `mark_violin`/`mark_boxplot`/`mark_qq`/etc. called (deferred to 8b/9) | `PyValueError` / `PyTypeError` / `NotImplementedError` |
| **Channel kwarg deferred** | `X(field, axis=Axis(...))`, `Color(field, legend=Legend(...))`, `Y(field, sort=...)`, `..., stack=...`, `..., impute=...`, `Color(scheme=...)`, `..., format=...` | One-time `UserWarning` per `(channel_class, kwarg)` per process: *"X(axis=...) is accepted but not honored in Phase 8a; planned for Phase 9."* Spec fields stored. |
| **Channel deferred** | `Stroke`, `StrokeDash`, `StrokeWidth`, `Fill`, `FillOpacity`, `StrokeOpacity`, `Angle`, `Text`, `Detail`, `Tooltip`, `TooltipField`, `Href`, `Description`, `Key`, `X2`, `Y2`, `XError`, `YError`, `XError2`, `YError2`, `Theta`, `Radius` used as encoding | One-time `UserWarning` per `(channel_class, "_channel_")` per process (registry key sentinel): *"Channel 'Stroke' is accepted but not rendered in Phase 8a; planned for Phase 9."* Spec field stored. |
| **Theme prop deferred** | Theme prop unknown to `ThemeInputs` Rust struct (e.g. `font_family`) | Stored in extra dict; passed through to `SvgBuffer` at the binding boundary; if `SvgBuffer` doesn't honor it, silent — Phase 7 already accepts unknown theme keys. |
| **Layered chart with mixed data** | `chart_a + chart_b` where `chart_a._data is not chart_b._data` (and not pyarrow-equal) | One-time `UserWarning` per `+` call: *"Layered charts with differing data render as horizontal concatenation. Use a shared DataFrame for true overlay."* Falls through to `chart_a | chart_b`. |
| **Layered chart with conflicting theme/facet/coord** | `chart_a + chart_b` where `chart_b._theme/facet/coord != chart_a`'s | One-time `UserWarning`: *"Layered chart `+`: secondary layer's theme/facet/coord is ignored; primary layer wins."* |
| **Mark not yet implemented** | `Chart.mark_boxplot()`, `mark_errorbar`, `mark_errorband`, `mark_ribbon`, `mark_violin`, `mark_contour`, `mark_qq`, `mark_raster`, `mark_swarm`, `mark_hex`, `mark_function` | `NotImplementedError("mark_X is planned for Phase 8b")` |
| **Mark not yet implemented (Phase 9+)** | `mark_arc`, `mark_image`, `mark_geoshape`, `mark_segment`, `mark_label` | `NotImplementedError("mark_X is planned for Phase 9+")` |
| **`mark_smooth` CI band** | `mark_smooth(ci=0.95)` called in 8a | One-time `UserWarning`: *"`ci=` band requires the ribbon mark, deferred to Phase 8b. Smooth curve rendered without CI band."* |
| **Geometric edge** (inherited from Phase 7) | Out-of-domain rows, color-palette overflow, layout warnings, etc. | `RenderWarning` collected on the result; surfaced via `warnings.warn` at the binding boundary |
| **Theme contextvar lifecycle** | `set_default_theme()` token not exited cleanly (e.g. exception during a `with` block) | Python's `ContextVar.reset(token)` semantics handle this; no special action — token is stored on the returned CM, restored on `__exit__` even under exceptions |

### §6.1 Warn-once registry

```python
# _warn.py
_seen: set[tuple[str, str]] = set()

def warn_once(channel: str, kwarg: str, message: str | None = None) -> None:
    key = (channel, kwarg)
    if key in _seen:
        return
    _seen.add(key)
    msg = message or f"{channel}({kwarg}=...) is accepted but not honored in Phase 8a; planned for Phase 9."
    import warnings
    warnings.warn(msg, UserWarning, stacklevel=3)

def reset_warnings() -> None:
    """For tests."""
    _seen.clear()
```

Tests use `reset_warnings()` in setup so each test sees a fresh slate.

---

## §7 New external dependencies

| Package | Pin discipline | Purpose | Notes |
|---|---|---|---|
| `narwhals` | Range pin (`narwhals = "~1.x"`) — exact x set at plan time after verifying current line | DataFrame compatibility layer; covers pandas/modin/cuDF/dask/ibis without per-type code | Runtime dep added to `pyproject.toml`. Lightweight (pure Python, no compiled extensions). Pin to a range to allow patch updates. Verify `narwhals.from_native` accepts `pyarrow.RecordBatch` at plan time; if not, ferrum's `_coerce` already converts at the boundary. |

No new Rust crates. The SVG compositor is hand-rolled (~150 LOC of Rust string manipulation, mirroring Phase 7's `SvgBuffer` discipline).

`pandas`, `modin`, `cudf`, `dask`, `ibis` are NOT runtime deps — narwhals duck-types them.

---

## §8 Test plan

### §8.1 Cargo tests (target ≥ 30 new; cargo total ≥ 291)

#### `spec/layer.rs` (~5 tests)
- `Layer { mark, encoding, transforms }` round-trips through serde JSON.
- `ChartSpec.layers = None` produces JSON without a `layers` field (`skip_serializing_if`).
- `ChartSpec.layers = Some(vec![...])` round-trips with multiple layers.
- Phase 3–7 single-layer canonical-JSON tests still pass byte-identical.
- Layer with empty encoding inherits chart-level encoding at draw time (verified by render test).

#### `spec/encoding.rs` (~6 tests)
- `EncodingSpec` gains 8 deferred Option fields (axis, legend, sort, stack, impute, scheme, format, formatType); all default-omit when None.
- `EncodingSpec` with `scale: Some(ScaleSpec::Log{...})` round-trips.
- `EncodingSpec` with `title: Some(...)` round-trips and replaces auto axis title.
- Backwards compat: pre-Phase-8 JSON (no scale/title fields) deserializes correctly.
- Color-channel `EncodingSpec` with `scheme: Some("tableau10")` round-trips (warn-once happens in Python).
- `EncodingSpec` with `axis: Some(AxisSpec)` round-trips and is ignored by renderer.

#### `spec/coord.rs` (~3 tests)
- `CoordKind::Flip` round-trips.
- `ChartSpec.coord = None` produces JSON without `coord` field.
- `Flip` swaps x/y in `prepare_render_inputs` (verified by render test asserting axis sides).

#### `render/marks/point.rs` (~3 tests)
- `Size` encoding scales point radius across rows.
- `Shape` encoding selects from a 6-shape palette (circle, square, cross, diamond, triangle-up, triangle-down).
- `Opacity` encoding sets per-row fill-opacity.

#### `render/scale_resolve.rs` (~4 tests)
- Explicit `Scale = ScaleLog` on x channel overrides auto-detected `LinearScale`.
- Size scale: defaults to `LinearScale` over `[min, max] → [3, 30]` (Theme.point_size_min/max).
- Shape scale: ordinal scale over distinct values → shape names in fixed order.
- Opacity scale: defaults to `LinearScale` over `[min, max] → [0.1, 1.0]`.

#### `render/compositor.rs` (~6 tests)
- `compose_svg_horizontal([svg_a, svg_b], spacing=10.0)` produces an outer `<svg>` with width = a.w + 10 + b.w, height = max(a.h, b.h).
- Inner SVG roots are extracted and wrapped in `<g transform="translate(...)">` with correct offsets.
- Font-defs block from second child stripped (single `<defs>` survives).
- Vertical compositor analogous: width = max, height = sum + spacing.
- Composing 3 charts hconcat works.
- Hconcat then vconcat (`(a | b) & c`) produces correctly-sized outer SVG.

#### `render/prepare.rs` (~3 tests)
- Multi-layer spec: scales built from union of all layers' encoded fields.
- CoordFlip: x_scale and y_scale swap before being passed to draw context.
- Layer with secondary `transforms` runs them only on that layer's data view (deferred — Phase 8a runs all transforms once on the chart-level batch; per-layer transforms warned-once).

### §8.2 Pytest tests (target ≥ 90 new; pytest total ≥ 179)

`tests/test_chart.py` (~25 tests):
- `Chart(df).mark_point().encode(x="a", y="b").show_svg()` returns SVG.
- Each fluent method returns a NEW Chart (immutability).
- `Chart(df).encode(x="a")` then later `Chart(df).encode(x="b")` produces independent charts.
- `Chart(pl.DataFrame({...}))`, `Chart(pa.table({...}))`, `Chart(pa.RecordBatch.from_pylist([...]))` all work.
- `Chart({"x": [1,2,3], "y": [4,5,6]})` (dict) works.
- `Chart([{"x": 1, "y": 4}, {"x": 2, "y": 5}])` (list of records) works.
- `Chart(np.array([[1,2],[3,4]]))` produces auto-named col_0, col_1.
- `Chart(np.array([1,2,3]))` raises clear TypeError.
- pandas DataFrame works via narwhals (skip if pandas not installed).
- Properties: `.properties(width=800, height=600, title="...")`.
- `.to_spec()` returns a `ChartSpec`.
- `.to_json()` round-trips.

`tests/test_marks.py` (~15 tests):
- Each of 8 primitive marks renders a minimal chart.
- `mark_density` produces a kde-shaped area; transform appended to spec.
- `mark_histogram` produces bars; `Bin` transform appended.
- `mark_smooth` produces a line; `Smooth` transform appended; `ci=0.95` warns.
- Mark kwargs (size, stroke, fill, opacity) flow to MarkStyle.
- Calling `mark_boxplot`/`mark_violin`/etc. raises `NotImplementedError` with helpful message.

`tests/test_encoding.py` (~20 tests):
- Each of 31 channel classes constructs without error from `("field_name")`.
- Shorthand string `"mean(price):Q"` parses correctly via `encode(x="mean(price):Q")`.
- `X(field, type="Q")` sets `EncodingSpec.type_ = Quantitative`.
- `X(field, bin=True)` appends Bin transform.
- `X(field, aggregate="mean")` appends Aggregate transform.
- `X(field, scale=ScaleLog(...))` stored on EncodingSpec.
- `X(field, title="My title")` stored on EncodingSpec.
- `X(field, axis=Axis(...))` warns once and stores.
- `Stroke("color")` warns once on render and is ignored.
- Repeated `Stroke` use across renders only warns once per process.
- Color channel with > 8 distinct values surfaces palette-overflow warning (from Phase 7).
- `_warn.reset_warnings()` clears state for next test.

`tests/test_composition.py` (~10 tests):
- `chart_a + chart_b` (same data) produces multi-layer ChartSpec; SVG contains both marks.
- `chart_a + chart_b` (different data) warns and falls through to hconcat.
- `chart_a | chart_b` produces hconcat; SVG width = sum + spacing.
- `chart_a & chart_b` produces vconcat.
- `(a | b) & c` operator-precedence: parses as `(a | b) & c`.
- `a | b & c` operator-precedence: parses as `a | (b & c)`; documented.
- Conflict warning: `chart_a + chart_b.theme(other)` warns about theme override.
- LayerChart returns a Chart (supports further `+` / `|` / `&`).

`tests/test_theme.py` (~8 tests):
- Each of 8 builtins exists and is a `Theme` instance.
- `theme.update(font_family="Inter")` returns new Theme with merged props.
- Original theme unmodified after `.update()` (immutability).
- `Chart.theme(my_theme)` overrides default.
- `set_default_theme(t)` makes subsequent charts use t.
- `with set_default_theme(t):` reverts on exit.
- Nested `with` blocks restore correctly.
- Explicit `.theme()` overrides `set_default_theme()`.

`tests/test_facet.py` (~5 tests):
- `Chart.facet(col="species")` produces FacetSpec mode=Wrap.
- `Chart.facet(row="year", col="species")` produces FacetSpec mode=Grid.
- Faceted SVG contains 3 strip-title text elements (Phase 7 path).
- `Facet(field, sort="...")` sort kwarg warns-once.

`tests/test_annotations.py` (~5 tests):
- `chart + annotate_hline(0)` produces a layer with mark_rule at y=0.
- `chart + annotate_vline(5)` produces vertical rule.
- `chart + annotate_text(1, 2, "hi")` produces a text mark.
- `chart + annotate_rect(0, 1, 0, 1, opacity=0.1)` produces a rect.

`tests/test_coord.py` (~2 tests):
- `Chart.coord(CoordFlip()).mark_bar().encode(x="cat", y="val")` produces horizontal bars.
- CoordPolar/CoordGeo/CoordFixed/CoordCartesian raise `NotImplementedError`.

`tests/test_show_save.py` (~6 tests):
- `chart.save("out.svg")` writes valid SVG.
- `chart.save("out.png")` writes PNG with magic bytes.
- `chart.save("out.html")` raises NotImplementedError ("html output deferred").
- `chart.save("out.unknown")` raises ValueError.
- `chart._repr_svg_()` returns SVG string (Jupyter rich display).
- `chart.show()` in non-Jupyter env opens browser (mocked test for `webbrowser.open`).

`tests/test_coerce.py` (~4 tests):
- `to_arrow_table(polars_df)` zero-copy.
- `to_arrow_table(pandas_df)` via narwhals.
- `to_arrow_table({"a": [1]})` via `pa.Table.from_pydict`.
- `to_arrow_table(unsupported_obj)` raises clear TypeError.

### §8.3 Test count baseline at end of Phase 8a

- `cargo test -p ferrum-core`: ≥ 291 (currently 261; +30 minimum)
- `uv run pytest`: ≥ 179 (currently 89; +90 minimum)

---

## §9 Done-criteria gate

From `ferrum-phases.md` Phase 8 done criteria, evaluated against Phase 8a scope:

- [ ] **`Chart(data).mark_point().encode(x="col_a", y="col_b").show()` works** → covered by `tests/test_chart.py` smoke tests + golden SVG.
- [ ] **Layer composition (`+`), hstack (`|`), vstack (`&`) work** → covered by `tests/test_composition.py` (10 tests) + 3 new SVG goldens.
- [ ] **`Theme` objects are values passed to `Chart`, not global state** → covered by `tests/test_theme.py`. CLAUDE.md updated with the contextvar exception (themes-as-values invariant honored: `Chart.theme(t)` always wins; `set_default_theme()` is a contextvars-backed default, not module-level mutable state). Phase 8a explicitly avoids any module-level `theme = Theme(...)` rebinding.
- [ ] **No `matplotlib` in the dependency tree** → CI check: `pip show matplotlib` returns nothing in fresh install. Already true; will not regress.
- [ ] **All encoding channels from `ferrum-spec.md §3.2` are implemented** → all 31 channels exist as Python classes (constructible, validated, store on spec). Renderer honors x/y/color/size/shape/opacity; rest warn-once. Spec interpretation: "implemented" = "exists as a typed Python value-class that produces a valid spec." Spec gets a dated note clarifying this interpretation.

A Phase-8a-done PR shows all five boxes ticked, `cargo test -p ferrum-core` ≥ 291 passing, `uv run pytest` ≥ 179 passing, and the 6 SVG goldens + 1 PNG hash from Phase 7 still match.

Phase 8b (separate spec, separate session) re-evaluates the done criteria again with composite + heavy stat marks added; no done-criterion regression is possible because all 8a work is purely additive.

---

## §10 Locked decisions

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | Phase 8 split | Phase 8a (this spec) ships API surface + primitives + 3 simple stat marks; Phase 8b ships composites + 7 heavy stat marks + new Phase 5 transforms | Total scope was 8000–12000 LOC. Splitting yields two reviewable PRs that each land before Phase 9. 8a is the "nothing visible to users without it" set; 8b is purely additive marks. |
| 2 | Multi-layer ChartSpec shape | Additive `layers: Option<Vec<Layer>>` field on `ChartSpec`, not an enum variant | Phase 3–7 goldens stay byte-identical when `layers.is_none()`. Existing single-layer JSON shape preserved; renderer's existing draw path is the `None` branch. Less invasive than switching to a tagged enum. |
| 3 | Layered chart with mixed data | Falls through to `HConcatChart` with one-time warning; renderer never grows multi-batch logic | One `ChartSpec` = one `RecordBatch` is a load-bearing simplification. Multi-batch in the renderer would balloon `scale_resolve.rs` and `prepare.rs` significantly. The `\|` path already handles "two charts side by side" correctly. |
| 4 | Composition for `\|`/`&` | Python orchestrates child renders, Rust SVG compositor stitches; no top-level multi-canvas layout | Reuses Phase 7's renderer per child unchanged. Compositor is ~150 LOC of deterministic SVG string manipulation, mirroring `SvgBuffer`'s discipline. Avoids a second pass at `compute_layout`. |
| 5 | Encoding channel surface | All 31 classes from §3.2 exist; `x/y/color/size/shape/opacity` rendered; rest warn-once and store on spec | "Implemented" interpreted as "constructible Python value-class that flows to a valid spec." Avoids the silent foot-gun of `aggregate=` being silently ignored, and avoids the 2–3× scope of wiring deferred channels into the renderer this phase. |
| 6 | Channel kwargs honored | `type`, `bin`, `aggregate`, `scale`, `title` actively translated; `axis`, `legend`, `sort`, `stack`, `impute`, `scheme`, `format` accepted-and-warned | Honored kwargs map 1:1 to existing Phase 4/5/7 capabilities — no new Rust required for desugaring. Deferred kwargs need new renderer or transform code; Phase 9 picks them up. |
| 7 | EncodingSpec storage for deferred kwargs | Typed `Option<>` fields on `EncodingSpec` (axis, legend, sort, stack, impute, scheme, format, formatType), not an opaque `HashMap<String, JsonValue>` | Typed fields are discoverable in `_core.pyi`, defensible against Phase 9 honoring them, and serde-validated. Opaque bag would require re-typing everything in Phase 9. |
| 8 | Shorthand string parsing | `"field"`, `"mean(field)"`, `"field:Q"` parsed inline at `.encode()` time via `_shorthand.parse_shorthand` | Matches Vega-Lite/Altair convention. Enables `chart.encode(x="mean(price):Q")` without channel-class boilerplate. |
| 9 | Mark scope (8a) | 8 primitives + `mark_density` + `mark_histogram` + `mark_smooth` (no CI band) | The three statistical marks map 1:1 to existing Phase 5 `Kde`/`Bin`/`Smooth` transforms. Composites + heavy stats → Phase 8b. |
| 10 | `mark_function` | Phase 8b; Python-side eval (`fn(x_array) → y_array`) → pass result through CDI as a regular line mark; no PyO3 Rust→Python callback | Avoids reaching back into the Python interpreter from Rust draw code. ~10 LOC at Phase 8b time. |
| 11 | Theme | `Theme` immutable value class; `Chart.theme(t)` per-chart override; `ferrum.set_default_theme(t)` returns a contextvars-backed CM (single primitive, dual usage); 8 builtins from §3.13 | Friendliest UX choice — matches seaborn ergonomic for notebooks. Per-chart override always wins over default. CLAUDE.md gets an addendum explicitly allowing the contextvars-backed default (per-thread, scope-bounded). |
| 12 | Theme builtin sourcing | Reference vega-lite theme JSONs for RGB constants where the spec is ambiguous (dark, fivethirtyeight, economist, solarized_*); document the mapping in a comment block | Spec §3.13 lists 8 themes with 1–2 sentences each. Concrete colors must come from somewhere; vega-lite is well-tested and visually consistent across our reference points. |
| 13 | Composition operator precedence | Python evaluates `&` tighter than `\|`, so `a \| b & c` = `a \| (b & c)`; documented in `Chart.__or__` docstring + spec note | Python operator precedence is fixed; can't override. Explicit parenthesization is the recommended idiom; tested both ways. |
| 14 | Data input compatibility | narwhals (~1.x) for DataFrame inputs; direct CDI for polars; ferrum branches for pyarrow Table/RecordBatch, dict, list, numpy 2D | Narwhals already absorbed pandas dtype gotchas (tz-aware datetime, categorical, object-dtype). Inheriting their fixes is cheaper and safer than authoring our own ~250 LOC normalization pass. modin/cuDF/dask/ibis support is "free" for narwhals' supported backends. |
| 15 | Narwhals ↔ pyarrow.RecordBatch | Verified at plan stage; if `nw.from_native` rejects RecordBatch, ferrum's `_coerce` already converts at the boundary | Spec §3.18 lists RecordBatch; cheap fallback if needed. |
| 16 | File path inputs (`Chart("file.csv")`) | Deferred to Phase 9 with clear `TypeError` | Format-detection + lazy-loading is real polars-reader plumbing; not blocking the API surface. |
| 17 | `.show()` env detection | Jupyter inline (`_repr_svg_`/`_repr_html_`) → browser fallback (write temp HTML, `webbrowser.open`); sixel terminal deferred to Phase 9 | Sixel users are a niche; non-blocking. Browser fallback covers all non-Jupyter cases. |
| 18 | `.save(path)` | Format from extension: `.svg` → `render_svg`; `.png` → `render_png`; `.html`/`.json` deferred to Phase 9 | json deferred even though `to_json` exists — it's the wrapping html that takes work. |
| 19 | Chart immutability | Every fluent method returns a new `Chart`; spec dict deep-copied per call | Eliminates aliasing surprises. Chains compose freely. Cost: small per-call allocation, negligible. |
| 20 | CoordFlip | In scope (Phase 8a); `prepare.rs` swaps x/y scales; ~30 LOC | Cheap, useful for horizontal bar charts. Other coord systems (Cartesian xlim/ylim, Polar, Geo, Fixed) deferred to Phase 9+. |
| 21 | Annotations | `annotate_hline`/`vline`/`rect`/`text` ship as Python sugar over primitive marks with inline 1-row tables | Quality-of-life win; ~80 LOC; no new Rust. AUCLabel/OutlierLabel deferred (need stat hookups). |
| 22 | Faceting | `Facet`/`FacetRow`/`FacetCol` Python channel classes + `Chart.facet()` method; sugar over existing Phase 6 `FacetSpec` | Phase 6+7 already render facets; this is just the Python API. |
| 23 | Selections in 8a | `.add_selection()` and `.interactive()` raise `NotImplementedError` with Phase 11 reference | Per spec §3.10, selections are silently ignored in SVG mode in normal use. Phase 8a hard-errors so users don't write code that quietly drops state. Phase 11 will accept selections. |
| 24 | ferrum-spec.md updates | Dated 2026-05-10 notes added to §3.2 (channel-render scope), §3.13 (CLAUDE.md ref for default-theme contextvar exception), §3.16 (.show env detection scope), §3.18 (file-path/ModelSource defer) | Spec stays the contract; phase-by-phase honest accounting via dated notes is the established pattern (Phase 7 §3.16 note). |
| 25 | CLAUDE.md updates | Three additions/refinements committed alongside this spec: (a) **Hard constraints** line refined: "No global mutable state. Themes are values passed to `Chart`; `Chart.theme()` always wins. The single exception is `ferrum.set_default_theme()`, which mutates a per-thread `contextvars.ContextVar` (not a module-level binding) — scope-bounded, documented." (b) **Key architectural decisions** table gains a row for **DataFrame compatibility:** "narwhals (~1.x) added (phase 8a+) for non-polars DataFrame inputs (pandas, modin, cuDF, dask, ibis). Direct CDI path preserved for polars + pyarrow Table/RecordBatch. Decision: own ~250 LOC of pandas dtype normalization vs. inherit narwhals' battle-tested handling — chose narwhals to dramatically shrink dtype-bug surface and to inherit modin/cuDF/dask/ibis support for free. Same library altair adopted in 2024 for the same problem." (c) **Key architectural decisions** table gains a row for **Multi-layer ChartSpec extension:** "`layers: Option<Vec<Layer>>` additive field on `ChartSpec` (phase 8a+). Single-layer JSON shape preserved when `layers.is_none()` so Phase 3–7 goldens stay byte-identical. One `ChartSpec` = one `RecordBatch`; mixed-data layers route through the SVG compositor instead of growing renderer multi-batch logic." | Resolves the tension between CLAUDE.md ("no global mutable state, no `set_theme()`") and spec §3.13 (`set_default_theme`). Pins narwhals and the multi-layer extension at the architectural-decision level — both are session-survival facts that future sessions need without re-reading this spec. All three are deliverables of this spec; commit alongside. |

---

## §11 Cross-phase notes

### Phase 7 (Static renderer) — what Phase 8a calls
- `render_svg(spec, batch, theme, viewport, config)` — single-chart rendering. Phase 8a constructs `spec` from Python `Chart` and calls this directly.
- `render_png(...)` — same shape, returns bytes.
- `prepare_render_inputs` — Phase 8a's `ChartSpec` gains `layers` and `coord` fields; `prepare` extends additively to handle them.
- `scale_resolve.rs` — extended for size/shape/opacity scales (linear → [3, 30] for size; ordinal over distinct values for shape; linear → [0.1, 1.0] for opacity).
- `palette.rs` — extended with 6 more categorical palettes (`tableau10`, `set1`, `set2`, `paired`, `pastel`, `dark2`); accessed via `theme.color_scheme` Python prop.
- `marks/point.rs` — extended to honor per-row size, shape, opacity from resolved scales.

### Phase 6 (Layout) — what Phase 8a calls
- `compute_layout` — called from inside `render_svg`. No changes for Phase 8a (multi-layer doesn't change panel sizes). `FacetSpec` already supported.

### Phase 5 (Stat) — what Phase 8a calls
- `Bin`, `Kde`, `Smooth`, `Aggregate`, `Summary` transforms — desugared from channel kwargs (`bin=True`, `aggregate="mean"`) and from `mark_density`/`mark_histogram`/`mark_smooth`.

### Phase 4 (Scale) — what Phase 8a calls
- All scale classes — `Scale=ScaleLog(...)` on a channel flows to `EncodingSpec.scale` and `scale_resolve.rs` honors it.

### Phase 8b (next spec, separate session) — what it inherits from 8a
- Full Python API surface: `Chart`, `Layer`, all 31 encoding channels, theme system, composition operators, faceting, annotations.
- Multi-layer `ChartSpec.layers` extension — composite marks (`mark_boxplot`) desugar to `chart.layer(box_layer, whisker_layer, outlier_layer)`.
- 7 new Phase 5 transforms (`Outliers`, `ErrorExtent`, `Contour`, `QQ`, `Raster`, `Hex`, `Swarm`, `BoxStats`, `Violin`) added per Phase 5's existing pattern.
- New SVG primitives in `SvgBuffer`: `image()` (for raster), `polygon()` (for hex/contour), with deterministic emission.

### Phase 9 (Convenience API) — what it inherits from 8a
- `displot`, `lmplot`, `roc_chart`, `pairplot` etc. desugar into `Chart` calls. The Python surface is the API they wrap.
- File-path inputs (`Chart("file.csv")`) added at this phase.
- `JointChart`, `RepeatChart` added.
- `CoordCartesian.xlim/ylim`, `CoordPolar`, `CoordGeo`, `CoordFixed` added.
- HTML/JSON output and sixel terminal added to `.show()`.
- AUCLabel/OutlierLabel annotations added.
- The 8 deferred channel kwargs (axis, legend, sort, stack, impute, scheme, format, formatType) get wired into the renderer.

### Phase 10 (Model diagnostics) — what it inherits from 8a
- `ModelSource` and `ComparedModelSource` accepted as `Chart(data=...)` inputs; `_coerce.to_arrow_table` learns to call `.predictions()`/`.roc_curve()`/etc.
- 25+ model-diagnostic marks added.

### Phase 11 (Interactive) — what it inherits from 8a
- `Chart.interactive()` no longer raises; switches render target.
- Selections accepted and resolved via WASM.
- All 8a channel encodings carried through to the WASM renderer's draw path.

### Phase 12 (Extension points) — what it inherits from 8a
- The Python channel-class pattern (`ChannelBase` subclassing) becomes the user-facing custom-encoding extension surface.
- Theme value class is the user-facing custom-theme extension surface.

---

## §12 Spec refinements (post-approval, plan-stage)

This section is reserved for refinements that surface during plan drafting. Items added here resolve under-specified inputs without changing scope or any locked decision in §10.

*(empty at spec-write time — populate during implementation planning.)*
