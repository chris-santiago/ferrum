# Phase 8b — Composite + Heavy Statistical Marks (Design Spec)

**Phase:** 8b
**Slug:** `composite-stat-marks`
**Depends on:** Phase 8a (`docs/superpowers/specs/2026-05-10-grammar-api-design.md`)
**Status at write time:** spec drafted; plan TBD; implementation TBD
**Date:** 2026-05-10

---

## §1 Goal

Phase 8b completes the user-facing mark surface that Phase 8a left as `NotImplementedError("planned for Phase 8b")` stubs. After 8b, every mark in `ferrum-spec.md §3.3` *primitive* and *composite* and *statistical* tables either renders end-to-end or has an explicit, dated deferral note in the spec. Specifically:

- Composite marks `boxplot`/`errorbar`/`errorband`/`ribbon` work on real data.
- Heavy statistical marks `contour`/`violin`/`qq`/`raster`/`swarm`/`hex`/`function` work on real data.
- `mark_smooth(ci=...)` finally renders its CI band (8a deferred this).
- The X2/Y2 channels accepted-but-not-rendered in 8a are wired through for the new error/ribbon marks.
- Phase 5's transform engine grows from 5 transforms (Bin/Kde/Smooth/Aggregate/Summary) to 15 (adds Outliers, ErrorExtent, BoxStats, Violin, Contour, Kde2D, QQ, Raster, Hex, Swarm).
- Phase 7's `SvgBuffer` grows three primitives (`image`, `polygon`, `beeswarm`).
- The Phase 5 color subsystem grows continuous colormaps (viridis family) — necessary for raster/hex/bivariate-density and previously absent.

**Non-goals:**
- No new encoding channel classes (8a shipped all 31; only the renderer for X2/Y2 is new in 8b).
- No new top-level `Chart` methods beyond the 11 mark methods.
- No interactivity (`mark_function` does not get a render-time recompute hook — that's Phase 11).
- No SHAP/PDP/learning-curve/etc. — those are Phase 10 model-diagnostic marks.
- No `mark_arc`/`mark_image`/`mark_geoshape`/`mark_segment`/`mark_label` — those stay as PHASE_9_PLUS_MARKS stubs.

---

## §2 Scope

### In scope (Phase 8b)
- 11 new working `Chart.mark_*` methods (replacing the 11 stubs in `marks/deferred.py::PHASE_8B_MARKS`).
- 10 new Phase 5 transforms with PyO3 wrappers + JSON round-trip + correctness tests.
- 3 new SVG primitives in `SvgBuffer` (`image`, `polygon`, `beeswarm`) with deterministic emission.
- Continuous colormap subsystem: `colorous` workspace dep, `ContinuousScheme` enum, `continuous_palette(name)` Python lookup mirroring `categorical_palette`, integration with the existing color-encoding code path.
- X2/Y2 channel wiring through the renderer for ribbon/errorband.
- `mark_smooth(ci=...)` CI band rendering (lifts the warn-once from 8a).
- Spec updates to `ferrum-spec.md` (dated 2026-05-10 notes for any 8b-specific clarifications).
- Phase 8b done-criteria checklist update in `ferrum-phases.md` (9→10 transforms).

### Deferred to Phase 9+ (explicitly out of scope)
- `mark_arc`, `mark_image`, `mark_geoshape`, `mark_segment`, `mark_label` (the 5 PHASE_9_PLUS_MARKS stubs stay).
- Auto-raster policy layer (`raster_threshold`, `raster_behavior`) — `mark_raster` exists explicitly; auto-substitution is Phase 9+.
- HConcatChart/VConcatChart `show_png()` (still raises NotImplementedError).
- `mark_function` re-evaluation on zoom/pan (interactive renderer concern).
- Per-axis CoordCartesian.xlim/ylim (Phase 9+ per 8a deferral list).

### Cross-phase dependencies inherited from 8a
- `ChartSpec.layers: Option<Vec<Layer>>` (additive field, byte-identical when None).
- `MarkKwargsSpec` per-mark style overrides.
- `_pending_stat_mark` deferral mechanism in `Chart`.
- Themes-as-values invariant + `set_default_theme()` contextvar exception (no new exceptions).
- Polars dtype handling in Rust (LargeUtf8/UInt*/etc. — already fixed in 596c440).

---

## §3 Architecture

### §3.1 Module layout (new files)

```
crates/ferrum-core/src/
  transform/
    outliers.rs         (NEW) — per-group outlier rows by IQR rule
    error_extent.rs     (NEW) — ci/stderr/stdev/iqr aggregation per group
    box_stats.rs        (NEW) — q1/median/q3/whiskers per group
    violin.rs           (NEW) — KDE per group → mirrored polygon coords
    contour.rs          (NEW) — Marching Squares isolines + isobands
    kde_2d.rs           (NEW) — 2D Gaussian KDE on a grid
    qq.rs               (NEW) — sample vs theoretical quantiles + reference line
    raster.rs           (NEW) — 2D bin aggregation → cell grid
    hex.rs              (NEW) — hexagonal bin aggregation (axial coords)
    swarm.rs            (NEW) — greedy-sweep collision-resolved positions
  render/
    marks/
      polygon.rs        (NEW) — generic polygon mark (used by contour/hex/violin)
      image.rs          (NEW) — image mark (used by raster)
      ribbon.rs         (NEW) — area-between-Y-and-Y2 mark
    color/
      mod.rs            (NEW dir; absorbs current color.rs)
      categorical.rs    (moved from color.rs; existing 6 palettes)
      continuous.rs     (NEW) — colorous-backed viridis/plasma/magma/inferno/cividis
      scheme.rs         (NEW) — Scheme enum + dispatch
    rasterize.rs        (NEW) — RGBA grid → PNG bytes via `png` crate (pinned settings)

src/ferrum/
  marks/
    composite.py        (NEW; pure-Python desugar helpers — boxplot/errorbar/errorband/ribbon)
    heavy_stat.py       (NEW; pure-Python desugar helpers — contour/violin/qq/raster/swarm/hex/function)
  schemes.py            (extend with continuous_palette() lookup)
```

No existing files are deleted. `marks/deferred.py::PHASE_8B_MARKS` shrinks to empty by end of phase.

### §3.2 Data flow — composite mark (e.g., `mark_boxplot`)

```
Chart(df).mark_boxplot().encode(x="group", y="value")
    │
    ▼  (Python, in Chart._resolve_pending_then_build_spec)
desugar_boxplot(x_field="group", y_field="value", **mark_kwargs)
    │
    ▼  returns (transforms, layers)
transforms = [BoxStats(field="value", groupby=["group"], name="box"),
              Outliers(field="value", groupby=["group"], extent=1.5, name="outliers")]
layers = [
    Layer(mark=Mark("rule"),  encoding={x:"group", y:"lower_whisker", y2:"upper_whisker"},
          data_source="box"),
    Layer(mark=Mark("rect"),  encoding={x:"group", y:"q1", y2:"q3"}, mark_kwargs={width:0.6},
          data_source="box"),
    Layer(mark=Mark("tick"),  encoding={x:"group", y:"median"}, data_source="box"),
    Layer(mark=Mark("point"), encoding={x:"group", y:"value"},   data_source="outliers"),
]
    │
    ▼
ChartSpec { layers: Some(layers), transforms, ... }   (Rust side)
    │
    ▼  (Rust, render pipeline)
For each layer L:
    apply transforms producing L's input batch (data_source-routed)
    invoke render::marks::<L.mark.name>::draw(L, batch, scales, viewport, svg)
```

The boxplot is **one ChartSpec, one input RecordBatch, one set of transforms producing multiple named columns**, with each layer selecting its columns via encoding. The `data_source` annotation routes layers to specific transform outputs (see §3.7).

### §3.3 Data flow — heavy stat mark (e.g., `mark_contour`)

```
Chart(df).mark_contour().encode(x="a", y="b")
    │
    ▼  (Python desugar in chart.py, hooks _pending_stat_mark)
mark_name="polygon", transforms=[Kde2D(x="a", y="b"), Contour(thresholds=6, fill=False)]
encoding_remap = {x: "contour_x", y: "contour_y", detail: "level_id"}
    │
    ▼  ChartSpec { mark: polygon, transforms, encoding (remapped) }
    │
    ▼  Rust render
Kde2D produces (grid_x[N], grid_y[M], density[N*M])
Contour consumes that grid → produces rows {contour_x, contour_y, level_id, level_value}
                              one row per polyline vertex; level_id groups vertices into paths
render::marks::polygon iterates by level_id, emits SVG <path>
```

For `mark_contour(fill=True)` (and bivariate `mark_density`), Contour produces *isoband* polygons instead of isolines (Marching Squares with two thresholds bounding each band). The polygon mark renders them with `fill-rule="evenodd"` to handle ring-shaped contours.

### §3.4 Data flow — `mark_raster`

```
Chart(df).mark_raster(aggregate="count", resolution=256).encode(x="a", y="b")
    │
    ▼  desugar
mark_name="image", transforms=[Raster(x, y, aggregate, field, resolution, ...)]
    │
    ▼  Rust
Raster transform produces a single-row batch with columns:
    x_min, x_max, y_min, y_max  (extent of grid in data space)
    width, height               (pixel dimensions of grid)
    pixel_data                  (binary column: row-major Vec<f64> of cell values)
    │
    ▼  render::marks::image
1. Read extent + grid from the batch's single row
2. Map cell values through ContinuousScheme (cmap kwarg, default "viridis")
3. Pack RGBA bytes; encode as PNG via render::rasterize::encode_png (Filter::Sub, level 9)
4. Base64-encode; emit <image href="data:image/png;base64,..." x=... y=... width=... height=.../>
   with x/y/width/height computed by mapping (x_min, y_min)/(x_max, y_max) through the X/Y scales
```

For `resolution="screen"`, the Raster transform is told the panel pixel size by the renderer at draw time (one pre-pass: the layout solver hands the panel rect to the transform engine before stats apply). The transform spec stores `resolution: ResolutionSpec::Screen | Fixed(u32) | XY(u32, u32)`, and the render pipeline injects panel dims when it sees `Screen`.

### §3.5 Data flow — `mark_function`

```
Chart(df).mark_function(np.sin, domain=(0, 2*np.pi), n=200).encode(x="t", y="value")
    │
    ▼  Python (chart.py, before any spec serialization)
1. Resolve domain:
   - explicit domain kwarg → use it
   - else: scan sibling layers' x-data for min/max → use that
   - else: raise ValueError with the three-tier rule
2. xs = np.linspace(*domain, n)
3. ys = fn(xs)   # returns numpy array
4. synthetic_table = pyarrow.Table.from_pydict({"x": xs, "y": ys})
5. Build a ChartSpec with mark="line", encoding={x:"x", y:"y"}, data=synthetic_table
6. If the parent Chart had other layers: this becomes one layer in a multi-layer ChartSpec
   with data_source pointing at the synthetic batch
```

No PyO3 Rust→Python callback. The function evaluates exactly once at Chart-build-time. Domain inference reads from the parent Chart's coerced data, so it works even when this layer is added via `+` to a chart with concrete data.

### §3.6 Continuous colormap subsystem

```
ContinuousScheme enum (Rust):
    Named(NamedContinuous)     — "viridis", "plasma", "magma", "inferno", "cividis"
    Gradient(Vec<(f64, Color)>) — user-defined stops [(t0, c0), (t1, c1), ...]
    Reverse(Box<ContinuousScheme>)

NamedContinuous variants are backed by colorous v0.6 (~12 KB, MIT, pinned LUTs).

dispatch:
    fn sample(&self, t: f64) -> Color   // t ∈ [0, 1]; clamp out-of-range

Python-side mirror in src/ferrum/schemes.py:
    continuous_palette(name) -> ContinuousScheme    (PyO3 wrapper)
    continuous_palette.list() -> [str]              (5 names)
    Linear interpolation for Gradient case happens in Rust

Color-encoding code path (existing render/scale_resolve.rs):
    encoding.color is String → Categorical (8a behavior, unchanged)
    encoding.color is Quantitative → ContinuousScheme.sample(scale.normalize(value))
                                     (NEW; previously errored or fell back)
```

The continuous-color path lights up automatically for any quantitative color encoding, not just for raster/hex. This unblocks heatmaps and bivariate scatter with continuous color in 8a-style charts as a side benefit.

### §3.7 Data-source routing for multi-output transforms

Some 8b transforms produce *multiple* logical outputs that different layers consume:
- `BoxStats` produces aggregated rows (one per group): used by box/whisker/median layers.
- `Outliers` produces row-level outliers (variable count per group): used by the outlier point layer.

Phase 8a's `ChartSpec` has `transforms: Vec<TransformSpec>` applied in pipeline order, with each transform consuming the previous output. Phase 8b adds:

```rust
struct Layer {
    // ... existing 8a fields ...
    data_source: Option<String>,  // (NEW) name of a transform producing this layer's input
}

struct TransformSpec {
    // ... existing variants ...
    name: Option<String>,  // (NEW) optional name for downstream layer references
}
```

When `Layer.data_source` is `None`, the layer consumes the final pipeline output (8a behavior — byte-identical). When `Some(name)`, the layer consumes the named transform's output. Composite-mark desugar emits both: BoxStats gets `name="box"`, Outliers gets `name="outliers"`, the four layers point at the appropriate name.

This is the *minimal* extension needed for boxplot. Errorbar/errorband/violin/qq reuse it.

---

## §4 Per-component contracts

### §4.1 New `Chart` methods (11)

Each method below is a thin Python wrapper that:
1. Validates kwargs against the spec signature.
2. Calls a desugar helper (in `src/ferrum/marks/composite.py` or `heavy_stat.py`).
3. Stores the result in `self._pending_stat_mark` (existing 8a slot) so encode-then-mark and mark-then-encode both work.
4. Returns `self`.

**Composite marks (4):**

```python
Chart.mark_boxplot(*, extent=1.5, size=None, outliers=True, **mark_kwargs) -> Chart
Chart.mark_errorbar(*, extent="ci", ticks=True, **mark_kwargs) -> Chart
Chart.mark_errorband(*, extent="ci", borders=False, **mark_kwargs) -> Chart
Chart.mark_ribbon(*, opacity=0.3, interpolate="linear", **mark_kwargs) -> Chart
```

`extent` accepts:
- `mark_boxplot`: `"min-max"` | float (IQR multiplier; default 1.5)
- `mark_errorbar`/`mark_errorband`: `"ci"` (95% bootstrap CI) | `"stderr"` | `"stdev"` | `"iqr"`
- `mark_ribbon`: not used (ribbon takes Y/Y2 directly, no aggregation)

**Heavy stat marks (7):**

```python
Chart.mark_contour(*, bandwidth="scott", thresholds=6, smooth=True, fill=False, **mark_kwargs) -> Chart
Chart.mark_violin(*, bandwidth="scott", inner="box", **mark_kwargs) -> Chart
Chart.mark_qq(*, distribution="normal", dequantize=False, line=True, **mark_kwargs) -> Chart
Chart.mark_raster(*, aggregate="count", field=None, cmap="viridis",
                  resolution="screen", blend="alpha", min_count=None, log_scale=False,
                  **mark_kwargs) -> Chart
Chart.mark_swarm(*, size=4, orient="vertical", spacing=1.0, side="both",
                 dodge=None, **mark_kwargs) -> Chart
Chart.mark_hex(*, bin_size=None, aggregate="count", field=None, cmap="viridis",
               stroke=None, stroke_width=0, **mark_kwargs) -> Chart
Chart.mark_function(fn, *, domain=None, n=200, clip=True, **mark_kwargs) -> Chart
```

`mark_violin(inner=...)` accepts `"box"` | `"quartile"` | `"point"` | `None`. Each value triggers a different inner-layer composition (see §4.2.5).

`mark_function` is the only one that takes a positional argument (the callable).

**Spec-listed kwargs *not* honored in 8b (warn-once, dated note in spec):**
- `mark_violin`: nothing — full coverage.
- `mark_qq`: nothing.
- `mark_raster`: `blend="additive"` warns (alpha blending only); `min_count` honored.
- `mark_hex`: `aggregate=` accepts count/mean/sum (other Vega-Lite aggregates warn).
- `mark_swarm`: `dodge=` warns (single-group only); rest honored.

### §4.2 Composite mark desugar contracts

Each composite returns `(transforms: list[Transform], layers: list[Layer])`. `_resolve_pending_then_build_spec` consumes the tuple to build a multi-layer ChartSpec.

#### §4.2.1 `desugar_boxplot(x_field, y_field, *, extent, outliers, size, **kw)`

```
transforms = [
  BoxStats(field=y_field, groupby=[x_field], whisker_extent=extent, name="box"),
  *( [Outliers(field=y_field, groupby=[x_field], extent=extent, name="outliers")]
     if outliers else [] ),
]
layers = [
  Layer(mark=Mark("rule"),  encoding={x:x_field, y:"lower_whisker", y2:"upper_whisker"},
        data_source="box"),
  Layer(mark=Mark("rect"),  encoding={x:x_field, y:"q1", y2:"q3"},
        mark_kwargs={"width": size or 0.6}, data_source="box"),
  Layer(mark=Mark("tick"),  encoding={x:x_field, y:"median"},
        mark_kwargs={"band_size": size or 0.6}, data_source="box"),
  *( [Layer(mark=Mark("point"), encoding={x:x_field, y:y_field},
            data_source="outliers")]
     if outliers else [] ),
]
```

Horizontal boxplot (CoordFlip in effect, or X numeric + Y categorical) swaps x/y in encoding maps. Detection: if encoding's x is numeric and y is categorical, flip. Color encoding: if present, becomes a per-group color (BoxStats and Outliers groupby gets the color field appended).

#### §4.2.2 `desugar_errorbar(x_field, y_field, *, extent, ticks, **kw)`

```
transforms = [ErrorExtent(field=y_field, groupby=[x_field], method=extent, name="err")]
layers = [
  Layer(mark=Mark("rule"), encoding={x:x_field, y:"lower", y2:"upper"}, data_source="err"),
  *( [Layer(mark=Mark("tick"), encoding={x:x_field, y:"lower"},
            mark_kwargs={"band_size": 6}, data_source="err"),
      Layer(mark=Mark("tick"), encoding={x:x_field, y:"upper"},
            mark_kwargs={"band_size": 6}, data_source="err")]
     if ticks else [] ),
]
```

#### §4.2.3 `desugar_errorband(x_field, y_field, *, extent, borders, **kw)`

```
transforms = [ErrorExtent(field=y_field, groupby=[x_field], method=extent, name="err")]
layers = [
  Layer(mark=Mark("ribbon"), encoding={x:x_field, y:"lower", y2:"upper"},
        mark_kwargs={"opacity": 0.3}, data_source="err"),
  *( [Layer(mark=Mark("line"), encoding={x:x_field, y:"lower"}, data_source="err"),
      Layer(mark=Mark("line"), encoding={x:x_field, y:"upper"}, data_source="err")]
     if borders else [] ),
]
```

#### §4.2.4 `desugar_ribbon(x_field, y_field, y2_field, **kw)`

```
transforms = []  # no aggregation; ribbon takes Y and Y2 directly
layers = [
  Layer(mark=Mark("ribbon"), encoding={x:x_field, y:y_field, y2:y2_field}, mark_kwargs=kw),
]
```

If the user did not provide a Y2 encoding (via `encode(y2=...)` or `Y2(...)` channel), raise a clear error: `"mark_ribbon requires both y and y2 encodings"`.

#### §4.2.5 `desugar_violin(x_field, y_field, *, bandwidth, inner, **kw)`

```
transforms = [Violin(field=y_field, groupby=[x_field], bandwidth=bandwidth, name="violin")]
violin_layer = Layer(mark=Mark("polygon"),
                     encoding={x:x_field, y:"violin_y", detail:"group_id"},
                     mark_kwargs={"fill_opacity": 0.5}, data_source="violin")

if inner == "box":
    box_transforms, box_layers = desugar_boxplot(x_field, y_field, extent=1.5, outliers=False, size=0.1)
    return ([*transforms, *box_transforms], [violin_layer, *box_layers])
elif inner == "quartile":
    transforms.append(BoxStats(field=y_field, groupby=[x_field], name="quart"))
    quart_layers = [
      Layer(mark=Mark("rule"), encoding={x:x_field, y:"q1"},
            mark_kwargs={"stroke_dash":[2,2]}, data_source="quart"),
      Layer(mark=Mark("rule"), encoding={x:x_field, y:"median"}, data_source="quart"),
      Layer(mark=Mark("rule"), encoding={x:x_field, y:"q3"},
            mark_kwargs={"stroke_dash":[2,2]}, data_source="quart"),
    ]
    return (transforms, [violin_layer, *quart_layers])
elif inner == "point":
    return (transforms, [violin_layer, Layer(mark=Mark("point"), encoding={x:x_field, y:y_field})])
elif inner is None:
    return (transforms, [violin_layer])
```

Violin transform output: per group, polygon vertices that traverse the right side top-to-bottom (KDE values offset right of category center) then mirror back the left side. `group_id` column ties vertices into one polygon per group.

### §4.3 Heavy stat mark desugar contracts (the simpler ones)

#### §4.3.1 `desugar_contour(x_field, y_field, *, bandwidth, thresholds, smooth, fill, **kw)`

```
transforms = [Kde2D(x=x_field, y=y_field, bandwidth=bandwidth, n=128),
              Contour(thresholds=thresholds, fill=fill, smooth=smooth)]
encoding = {x: "contour_x", y: "contour_y", detail: "level_id"}
mark = Mark("polygon", mark_kwargs={"fill_opacity": 0.3 if fill else 0,
                                     "stroke_width": 0 if fill else 1.5})
```

Single layer (the layered case is bivariate-density routing, see §4.3.6).

#### §4.3.2 `desugar_qq(field, *, distribution, dequantize, line)`

```
transforms = [QQ(field=field, distribution=distribution, dequantize=dequantize,
                 emit_line=line, name="qq_main")]
layers = [Layer(mark=Mark("point"), encoding={x:"theoretical", y:"sample"}, data_source="qq_main")]
if line:
    layers.append(Layer(mark=Mark("rule"),
                        encoding={x:"qq_line_x_start", y:"qq_line_y_start",
                                  x2:"qq_line_x_end", y2:"qq_line_y_end"},
                        data_source="qq_line"))
    # QQ transform emits two named outputs: "qq_main" (the points) and "qq_line" (the reference line)
```

#### §4.3.3 `desugar_raster(x_field, y_field, *, aggregate, field, cmap, resolution, blend, min_count, log_scale)`

```
transforms = [Raster(x=x_field, y=y_field, aggregate=aggregate, field=field,
                     resolution=resolution, min_count=min_count, log_scale=log_scale)]
mark = Mark("image", mark_kwargs={"cmap": cmap, "blend": blend})
encoding = {}  # raster mark reads grid extent + pixel data from the batch directly
```

The image mark needs no x/y encoding — the Raster transform output carries extent and dimensions in fixed columns.

#### §4.3.4 `desugar_swarm(x_field, y_field, *, size, orient, spacing, side, **kw)`

```
transforms = [Swarm(category=(x_field if orient=="vertical" else y_field),
                    value=(y_field if orient=="vertical" else x_field),
                    point_size=size, spacing=spacing, side=side)]
mark = Mark("point", mark_kwargs={"size": size, **kw})
encoding = {x: "swarm_x", y: "swarm_y"}
```

Swarm transform output: row-per-input with collision-resolved (swarm_x, swarm_y) plus a copy of any other columns (color, opacity, etc. carry through).

#### §4.3.5 `desugar_hex(x_field, y_field, *, bin_size, aggregate, field, cmap, stroke, stroke_width)`

```
transforms = [Hex(x=x_field, y=y_field, bin_size=bin_size, aggregate=aggregate, field=field)]
mark = Mark("polygon", mark_kwargs={"stroke": stroke, "stroke_width": stroke_width, "cmap": cmap})
encoding = {x: "hex_x", y: "hex_y", color: "value", detail: "hex_id"}
```

Hex transform output: 6 vertices per non-empty hex cell, grouped by `hex_id`. The polygon mark draws one filled hexagon per group.

#### §4.3.6 `desugar_density` (bivariate routing)

In `desugar_density(field, **kw)`, *if* the encoding has both X and Y bound to quantitative fields, route through `desugar_contour(x_field=encoding.x, y_field=encoding.y, fill=True, **kw)` instead of the 1D KDE path. Single Contour codepath; no parallel 2D pipeline.

#### §4.3.7 `desugar_function(fn, *, domain, n, clip, parent_chart)`

Implementation per §3.5. Materializes a synthetic `pyarrow.Table`, returns it as a tuple `("line", [], encoding_remap, synthetic_data)` — the synthetic_data slot is consumed by Chart's spec builder. To keep the desugar contract uniform, the Phase 8a 3-tuple convention is extended to a 4-tuple (4th slot = optional synthetic data, default None) for *all* desugars.

### §4.4 New transforms — Rust contracts

Each transform follows the existing `transform/<name>.rs` pattern: `<Name>Spec` struct (Serialize+Deserialize), `apply(&Spec, &RecordBatch) -> PyResult<RecordBatch>`, `Py<Name>` PyO3 wrapper. JSON tag = snake_case name. All registered in `transform/core.rs::TransformSpec` enum + dispatch.

| Transform | Input columns | Output columns | Key behavior |
|---|---|---|---|
| `Outliers` | `field` (f64), `groupby` (any) | original schema, filtered to outlier rows only | Flags rows where value falls outside [q1 - k·IQR, q3 + k·IQR] for k=`extent` |
| `ErrorExtent` | `field` (f64), `groupby` (any) | groupby cols + `mean`, `lower`, `upper` | Method ∈ {ci, stderr, stdev, iqr}; bootstrap n=1000, seeded |
| `BoxStats` | `field` (f64), `groupby` (any) | groupby cols + `q1`, `median`, `q3`, `lower_whisker`, `upper_whisker` | whisker_extent="min-max" or k·IQR clipped to data range |
| `Violin` | `field` (f64), `groupby` (any) | `group_id`, `<groupby>`, `violin_x` (offset from group center), `violin_y` | Per-group KDE → mirrored polygon vertices; max kde value normalized to `width` (default 0.4 of band) |
| `Kde2D` | `x` (f64), `y` (f64) | single-row batch: `grid_x` (List\<f64\>), `grid_y` (List\<f64\>), `density` (List\<f64\>), `nx`, `ny`, `extent` (List\<f64\>) | Gaussian KDE on uniform grid, default 128×128 |
| `Contour` | output of Kde2D (single-row grid batch) | `contour_x`, `contour_y`, `level_id`, `level_value` | Marching Squares; `fill=False` → isolines (vertices per polyline); `fill=True` → isobands (closed polygons, level_id includes hole-ring index) |
| `QQ` | `field` (f64) | `theoretical`, `sample` per row + (if `emit_line`) named output "qq_line" with single row of `qq_line_{x_start,x_end,y_start,y_end}` | Theoretical from `distribution` ∈ {normal, uniform, exponential}; line via robust resistant fit (1st & 3rd quartiles) |
| `Raster` | `x` (f64), `y` (f64), optional `field` | single-row: `x_min`, `x_max`, `y_min`, `y_max`, `width`, `height`, `pixel_data` (Binary col, row-major Vec\<f64\>) | aggregate ∈ {count, density, mean, sum, any}; resolution = Screen \| Fixed(u32) \| XY(u32, u32); for Screen, render-time injection of panel size |
| `Hex` | `x` (f64), `y` (f64), optional `field` | `hex_x`, `hex_y` (vertex coords), `hex_id`, `value` | Pointy-top hex; bin_size auto-computed from data range / 30 if None; one of count/mean/sum aggregate |
| `Swarm` | `category`, `value` (f64) + carry-through cols | input cols + `swarm_x`, `swarm_y` | Greedy sweep: sort by value with stable tiebreak on row index; place each point at the closest non-overlapping offset on the categorical axis |

All 10 transforms have:
- `Debug + Clone + Serialize + Deserialize + PartialEq` on the spec struct.
- A round-trip test (`serde_json::to_string` + `from_str` recovers PartialEq).
- A correctness test against a hand-computed reference (numpy/scipy values computed offline).
- A schema test (output columns and dtypes match the table above).

### §4.5 New SVG primitives — `SvgBuffer` API additions

```rust
impl SvgBuffer {
    /// Embed a PNG as <image href="data:image/png;base64,..." x=... y=... width=... height=.../>.
    /// `png_bytes` must be a valid PNG-encoded RGBA buffer (use render::rasterize::encode_png).
    /// Emits no whitespace or newlines outside the href value.
    pub fn image(&mut self, x: f64, y: f64, w: f64, h: f64, png_bytes: &[u8]);

    /// Emit a closed filled/stroked polygon as <path d="M ... Z" fill-rule="evenodd"/>.
    /// `paths`: each inner Vec<(f64, f64)> is one ring; multiple rings → first is outer,
    /// rest are holes. fill-rule="evenodd" handles winding automatically.
    pub fn polygon(&mut self, paths: &[Vec<(f64, f64)>], style: &FillStroke);

    /// Emit a batch of <circle> elements at pre-resolved positions.
    /// Equivalent to N circle() calls with deterministic ordering, but emits as a single
    /// <g> group for compactness. Used by mark_swarm to keep beeswarm DOM size manageable.
    pub fn beeswarm(&mut self, points: &[(f64, f64)], radius: f64, style: &FillStroke);
}
```

All three primitives:
- Honor the existing `fmt_f` precision/locale rules (no `,` in floats).
- Match `circle()`/`rect()`/`path()` deterministic attribute ordering.
- Pass a Phase 7-style "byte-identical across runs" snapshot test.

### §4.6 Continuous colormap API

**Rust:**

```rust
pub enum ContinuousScheme {
    Named(NamedContinuous),                // viridis|plasma|magma|inferno|cividis
    Gradient(Vec<(f64, Color)>),           // user-defined stops
    Reverse(Box<ContinuousScheme>),
}
impl ContinuousScheme {
    pub fn sample(&self, t: f64) -> Color; // t ∈ [0,1]; clamped
}
```

**Python:**

```python
ferrum.continuous_palette(name: str) -> ContinuousScheme   # 5 named maps + .reversed()
ferrum.continuous_palette.list() -> list[str]              # ["viridis", "plasma", "magma", "inferno", "cividis"]
ferrum.Gradient(stops: list[tuple[float, str]]) -> ContinuousScheme   # ("#ff0000" or "red" → Color via existing parser)
```

**Wire-up to encoding:** `render/scale_resolve.rs`'s color resolver gains a quantitative branch: if `encoding.color` field is numeric, use the layer's `mark_kwargs.cmap` (or the chart's color scheme, defaulting to `"viridis"`) to map normalized values to colors. Categorical color path (8a) is unchanged.

### §4.7 Data-source routing — `Layer.data_source`

```rust
struct Layer {
    /* existing 8a fields */
    pub data_source: Option<String>,  // (NEW) name of a transform whose output this layer consumes
}
```

**Resolution rule** (in `render/prepare.rs`):
1. Run all transforms in pipeline order. If a transform spec has `name: Some(s)`, register its output as `outputs[s] = batch`. The final-pipeline output is also registered as `outputs["__final__"]`.
2. For each layer L, compute its input batch as `outputs[L.data_source.as_deref().unwrap_or("__final__")]`.
3. Error if a layer references an unknown name.

Phase 8a charts (no named transforms, all layers data_source=None) stay byte-identical.

### §4.8 `mark_smooth(ci=)` integration (lifts the 8a deferral)

In `desugar_smooth` (existing 8a function):

```python
if ci is not None:
    # Smooth transform already produces ci_lower/ci_upper when ci is set.
    transforms = [Smooth(x_field, y_field, method=method, ci=ci, bandwidth=bandwidth,
                         degree=degree, n=n, name="smooth")]
    layers = [
        Layer(mark=Mark("ribbon"),
              encoding={x:"x", y:"ci_lower", y2:"ci_upper"},
              mark_kwargs={"opacity": 0.3}, data_source="smooth"),
        Layer(mark=Mark("line"),
              encoding={x:"x", y:"y"}, data_source="smooth"),
    ]
    return ("__layered__", transforms, None, layers)
```

The Phase 8a warn-once for `mark_smooth(ci=)` is removed; its test is updated to assert no warning + assert two layers in the spec.

---

## §5 Algorithms

### §5.1 Marching Squares (Contour transform)

**Inputs:** Kde2D output — uniform grid of width `nx` × height `ny` density values plus extent `(x_min, x_max, y_min, y_max)`.

**Isoline mode (`fill=False`):**
1. For each grid cell (square of 4 corners), compute case index ∈ 0..16 by comparing each corner to the threshold value.
2. Use the standard 16-entry case table to emit 0/1/2 line segments per cell, with linear interpolation along the edges crossed.
3. Stitch segments into polylines: a hash map keyed by (rounded endpoint coords with eps=1e-12) joins shared endpoints. Each polyline gets a unique `level_id`.
4. For ambiguous cases 5 and 10 (saddle points), use the *cell center value* (averaged from corners) to pick the connection — a deterministic, side-effect-free rule.

**Isoband mode (`fill=True`):**
1. Run Marching Squares twice per band (low threshold + high threshold), producing two isolines.
2. Walk both isolines together to construct closed polygon rings: outer ring = high contour, hole = low contour if and only if low contour lies entirely *inside* the outer ring (point-in-polygon test).
3. Multi-mode densities (e.g., bimodal) produce multiple disjoint polygon groups per band — each gets its own `level_id`.

Output schema: `level_id: u32`, `level_value: f64`, `contour_x: f64`, `contour_y: f64`, with one row per polyline vertex. For isobands, `level_id` encodes (band_index << 16) | ring_index so the polygon mark groups vertices into ring sets.

**Rationale:** Marching Squares is O(nx × ny) and deterministic. Saddle-point handling via cell-center disambiguation is the standard approach (matches d3-contour and matplotlib internally).

### §5.2 Beeswarm greedy sweep (Swarm transform)

**Inputs:** `category` column (any), `value` column (f64), `point_size` (pixels), `spacing` (multiplier), `side` ∈ {both, left, right}.

**Algorithm (per category group):**
1. Sort rows by value ascending. Stable sort with tiebreak on original row index — this is the determinism guarantee.
2. Convert `point_size * spacing` to data-space radius using the value-axis scale's inverse. Scale info passes via `apply_with_context` (see §5.9), not via embedding scale info in TransformSpec.
3. For each point in sorted order:
   - Candidate offsets: `[0, +d, -d, +2d, -2d, +3d, -3d, ...]` where `d = 2 * radius` (for `side="both"`); `[0, +d, +2d, ...]` for `side="right"`; `[0, -d, -2d, ...]` for `side="left"`.
   - For each candidate, check if a circle of `radius` at that offset overlaps with any previously placed point in the same group within ±2·radius value-distance. (Range query via a sorted index of placed points.)
   - Place at the first candidate with no overlap.
4. Emit `swarm_x` and `swarm_y` as the (offset, value) coordinates in data space, plus all carry-through columns from the input.

**Complexity:** O(n²) worst case per group; with the value-sorted index, expected O(n·k) where k = local density. For n > 50,000 in one group, the spec docs `mark_raster` as the recommended fallback.

### §5.3 Hexagonal binning (Hex transform)

**Inputs:** `x`, `y` (f64), optional `field` (f64), `bin_size` (data-space hex side length).

**Algorithm:**
1. If `bin_size` is None, set `bin_size = (x_extent / 30)` (matches d3-hexbin default).
2. Convert each (x, y) to fractional axial hex coordinates: `q_frac = (sqrt(3)/3 · x - y/3) / bin_size`, `r_frac = (2/3 · y) / bin_size`.
3. Round to integer axial coords using cube-rounding (avoids the well-known fractional-axial rounding bias).
4. Aggregate per (q, r): count, mean(field), or sum(field).
5. For each non-empty (q, r), emit 6 vertex rows:
   - Hex center: `(cx, cy) = bin_size · (sqrt(3)·(q + r/2), 1.5·r)`
   - 6 vertices for pointy-top hex: `cx + bin_size·sin(θ_i)`, `cy + bin_size·cos(θ_i)` for `θ_i = i·60°`, i = 0..5
   - All 6 rows share `hex_id = q*65536 + r` (unique per hex), `value = aggregate`.

Output schema: `hex_x: f64`, `hex_y: f64`, `hex_id: i64`, `value: f64`.

### §5.4 BoxStats and Outliers

**BoxStats:**
1. Group rows by `groupby` columns.
2. Per group: compute Q1, median, Q3 via Type-7 (linear interpolation) quantile method (matches numpy default, scipy default, R `quantile(type=7)`).
3. IQR = Q3 - Q1. Whisker bounds:
   - `whisker_extent = "min-max"`: lower_whisker = group min, upper_whisker = group max.
   - `whisker_extent = k` (float): lower_whisker = max(group_min, Q1 - k·IQR), upper_whisker = min(group_max, Q3 + k·IQR). Whiskers always clip to actual data range — never extend beyond observed values.
4. Output: one row per group with groupby cols + (q1, median, q3, lower_whisker, upper_whisker).

**Outliers** (using same group + same k as BoxStats by convention; but Outliers takes `extent` as an independent param):
1. Compute Q1, Q3, IQR per group (same as BoxStats step 2-3).
2. Filter: rows where `value < Q1 - k·IQR` or `value > Q3 + k·IQR`.
3. Preserve full input schema (Outliers is a row filter, not an aggregator).

### §5.5 ErrorExtent

| `method` | `mean` | `lower` / `upper` |
|---|---|---|
| `"ci"` | sample mean | bootstrap percentile 95% (n=1000, seed via PyO3 from spec.seed=0) |
| `"stderr"` | sample mean | mean ± SEM (= stdev/√n) |
| `"stdev"` | sample mean | mean ± stdev |
| `"iqr"` | median | Q1 / Q3 |

Bootstrap uses `rand_chacha` (already a workspace dep) seeded from `spec.seed` for byte-determinism.

### §5.6 QQ transform

1. Sort sample values ascending; n = sample size.
2. Plotting positions: `p_i = (i - 0.5) / n` for i = 1..n (Hazen formula, the matplotlib/scipy default).
3. Theoretical quantiles via inverse CDF of the chosen distribution:
   - `"normal"`: `μ + σ · Φ⁻¹(p)` where μ, σ estimated from sample (mean, sample stdev).
   - `"uniform"`: `min + (max - min) · p`.
   - `"exponential"`: `-mean · ln(1 - p)`.
4. If `dequantize=True`, jitter ties by adding U(0, 1e-9 · range).
5. If `emit_line=True`, emit a second named-output batch `"qq_line"` with single row computed via robust resistant fit (slope = (sample_q3 - sample_q1) / (theo_q3 - theo_q1), intercept = sample_q2 - slope · theo_q2). Produces `qq_line_x_start`, `qq_line_x_end`, `qq_line_y_start`, `qq_line_y_end` spanning the theoretical extent.

### §5.7 Kde2D + Raster grid

**Kde2D:**
1. Compute marginal bandwidths: `h_x = scott(x)`, `h_y = scott(y)` (existing Phase 5 Scott's rule).
2. Build uniform grid: `nx = ny = 128` (default; `n` kwarg overrides).
3. For each grid cell (gx, gy): density = `(1/n) · Σ_i K_x((gx - x_i)/h_x) · K_y((gy - y_i)/h_y) / (h_x · h_y)` where K is the Gaussian kernel.
4. Optimization: separable kernel allows O(N · (nx + ny)) instead of O(N · nx · ny) by precomputing per-axis convolution intermediates.
5. Output: single-row batch with `grid_x: List<f64>`, `grid_y: List<f64>`, `density: List<f64>`, `nx: u32`, `ny: u32`, `extent: List<f64>` (4 values).

**Raster:**
1. Choose grid dimensions from `resolution`:
   - `Fixed(n)`: nx = ny = n.
   - `XY(nx, ny)`: as given.
   - `Screen`: panel pixel size injected at apply-with-context call (see §5.9 mechanism).
2. Compute 2D bin counts/values via histogram2d.
3. For `aggregate="density"`, divide counts by (cell_area × n).
4. If `min_count` is set, mask cells below threshold (set to NaN; renderer maps NaN to transparent).
5. If `log_scale=True`, apply `log1p` to non-NaN values.
6. Pack RGBA bytes via `cmap.sample(normalize(value))`. Normalize uses min/max over non-masked cells.
7. Encode PNG: `png` crate, `Filter::Sub`, compression level `Best` (level 9). `Best` instead of `Default` because raster goldens benefit from determinism over encode speed (the SVG goldens already pin every emitted byte).
8. Output: single-row batch with `x_min`, `x_max`, `y_min`, `y_max`, `width`, `height`, `pixel_data: Binary`.

### §5.8 Continuous colormap normalization

For a quantitative color encoding in a layer:
1. The color scale (existing 8a infra: `ColorScale::Linear { domain }`) computes `t = (value - domain_min) / (domain_max - domain_min)`, clamped to [0, 1].
2. `ContinuousScheme.sample(t)` returns a Color via:
   - `Named`: lookup in colorous's pinned LUT, linearly interpolated between adjacent stops.
   - `Gradient`: binary search for the bracketing stops; linear interpolation in linear sRGB space (not gamma-space — matches d3-interpolate default).
   - `Reverse`: sample inner with `1 - t`.

### §5.9 `resolution="screen"` render-time injection

Render pipeline gains a new pre-pass right after layout, before transform application:

```
layout solver computes panel rect (in pixels) for each panel
    ↓
TransformContext { panel_pixel_size: (u32, u32) } injected into apply call
    ↓
For Raster transform: if spec.resolution == Screen, set width/height = panel_pixel_size
For Swarm transform: pass panel_pixel_size for radius unit conversion
For other transforms: ignored (most don't need viewport info)
    ↓
apply_with_context(spec, batch, context) — new function in transform/core.rs
                                             defaults to apply(spec, batch) for transforms that ignore context
```

This is the only structural change to the transform engine API in 8b.

When `apply_with_context` is called *without* context (e.g., from a JSON-replay test), Raster falls back to default 256×256 and Swarm uses a fixed 4-pixel radius assumption (logged as a warn-once).

---

## §6 Error policy

### §6.1 New errors (raised, not warned)

| Trigger | Message |
|---|---|
| `mark_ribbon` without Y2 encoding | `"mark_ribbon requires both y and y2 encodings"` |
| `mark_function` with no domain and no inferable x-data | `"mark_function requires explicit domain when chart has no other data layers"` |
| `mark_function` callable returns wrong shape | `"mark_function callable must return numpy array of shape (n,); got shape {shape}"` |
| `mark_violin(inner=...)` invalid value | `"mark_violin inner must be one of 'box', 'quartile', 'point', or None; got {value}"` |
| `mark_qq(distribution=...)` unknown | `"mark_qq distribution must be 'normal', 'uniform', or 'exponential'; got {value}"` |
| `mark_raster(field=...)` required but missing | `"mark_raster aggregate='{agg}' requires field=..."` |
| `mark_hex(field=...)` required but missing | same as above for hex |
| `Layer.data_source` references unknown name | `"Layer references data_source '{name}'; available names: [...]"` |
| Transform output schema mismatch | `"Transform '{name}' produced unexpected schema: expected {expected}, got {actual}"` |

### §6.2 New warn-once categories (extends 8a's registry)

| Category | When | Message |
|---|---|---|
| `("mark_raster", "blend_additive")` | `blend="additive"` passed | `"mark_raster blend='additive' deferred to Phase 11; using alpha blending"` |
| `("mark_swarm", "dodge")` | `dodge=` non-None | `"mark_swarm dodge= is not yet supported; rendering single-group swarm"` |
| `("mark_hex", "aggregate_unsupported")` | aggregate not in {count, mean, sum} | `"mark_hex aggregate='{agg}' deferred; falling back to 'count'"` |
| `("mark_function", "domain_inferred")` | domain inferred from sibling layer | informational: `"mark_function domain inferred from sibling layer's x-range: ({min}, {max})"` (debug-level, opt-in) |

### §6.3 Removed warn-once (lifted in 8b)

| Category | Reason |
|---|---|
| `("mark_smooth", "ci")` | CI band now renders via ribbon mark |

### §6.4 Determinism guarantees

- All transforms with stochastic components (ErrorExtent bootstrap, Smooth's loess CI bootstrap) seed from `spec.seed: u64` (default 0). Same input + same spec → byte-identical output across runs and platforms.
- All SVG primitives emit attributes in fixed order. Float formatting uses `fmt_f` (Phase 7's helper).
- PNG bytes from raster are reproducible: pinned `png` crate version (0.18), `Filter::Sub`, compression `Best` (level 9). The Phase 7 PNG hash golden continues to pass (uses resvg for SVG→PNG, a different code path; verified independent in current cargo lockfile).
- Beeswarm: stable sort + row-index tiebreak ensures deterministic placement.
- Marching Squares: saddle disambiguation via cell-center value (deterministic).

### §6.5 Spec-implementation drift notes (to add to ferrum-spec.md)

End-of-phase, append dated 2026-05-10 notes to `ferrum-spec.md` for:
- §3.3 `mark_raster`: clarify that `blend="additive"` is deferred to Phase 11 (interactive renderer).
- §3.3 `mark_raster`: clarify auto-raster policy is Phase 9+; explicit `mark_raster` is Phase 8b.
- §3.3 `mark_swarm`: `dodge=` parameter deferred (single-group only in 8b).
- §3.3 `mark_hex`: only count/mean/sum aggregates supported in 8b.
- §3.4: Add `Kde2D` transform alongside `stat_kde_2d`. (Spec already lists `stat_kde_2d`; the 10th transform aligns with that.)
- Phase 8b done-criteria in `ferrum-phases.md`: update from 9 transforms to 10 (add `Kde2D`).

---

## §7 New external dependencies

### §7.1 Rust workspace deps

| Crate | Version | Used by | Rationale |
|---|---|---|---|
| `colorous` | `0.6` | `render::color::continuous` | Continuous colormaps (viridis/plasma/magma/inferno/cividis). Tiny (~12 KB compiled), MIT, pinned 256-stop LUTs. Alternative considered: hand-rolled tables (rejected — would be ~50 LOC and 3840 floats, no real saving once interpolation logic is added). Maintained by sibling project of d3-scale-chromatic, so visual semantics match standard tools. |
| `png` | `0.18` | `render::rasterize` | PNG encoding for raster mark. **Promoted from transitive (via resvg) to direct workspace dep** so the version is explicit and the encoder settings are pinned at the use site. No new code added to the build graph. |

### §7.2 Python deps — none

No new Python deps. `numpy` and `pyarrow` (already required runtime deps) cover everything `mark_function` needs for synthetic-data construction.

### §7.3 Workspace-Cargo.toml diff

```toml
[workspace.dependencies]
colorous = "0.6"   # NEW — continuous colormaps
png      = "0.18"  # NEW — pinned for raster determinism (was transitive)
```

`crates/ferrum-core/Cargo.toml` adds both to its `[dependencies]` block.

### §7.4 No-matplotlib audit

No new dep transitively depends on matplotlib. Cleared via `cargo tree -p ferrum-core | grep -i matplotlib` (returns empty).

---

## §8 Test plan

Targets at end of 8b: **≥379 cargo tests** (currently 309 → +70 minimum), **≥397 pytest** (currently 217 → +180 minimum). Below adds ~75 cargo and ~210 pytest with comfortable margin.

### §8.1 Cargo tests (target ≥+70 new; cargo total ≥384)

Per new transform module (`crates/ferrum-core/src/transform/<name>.rs`), each gets:

| Module | Tests | Coverage |
|---|---|---|
| `outliers.rs` | 4 | basic 1.5·IQR; min-max no outliers; per-group; row-filter preserves schema |
| `error_extent.rs` | 5 | each of 4 methods (ci/stderr/stdev/iqr) + bootstrap reproducibility (same seed → same bytes) |
| `box_stats.rs` | 5 | Q1/median/Q3 against scipy reference; whisker clamping; per-group; min-max mode; PartialEq round-trip |
| `violin.rs` | 3 | polygon vertex count and symmetry; per-group; bandwidth ∈ {scott, silverman, float} |
| `contour.rs` | 6 | isoline schema; isoband schema; saddle disambiguation; ring-with-hole; bivariate-density routing; round-trip |
| `kde_2d.rs` | 3 | density sums to ~1.0; grid extent; round-trip |
| `qq.rs` | 4 | normal/uniform/exponential; reference line emission; dequantize jitter |
| `raster.rs` | 5 | each aggregate (count/density/mean/sum); resolution = Fixed/XY/Screen (Screen via mock context); min_count masking; log_scale |
| `hex.rs` | 4 | count/mean/sum aggregates; bin_size auto vs explicit; cube-rounding correctness; vertex count = 6N |
| `swarm.rs` | 4 | side=both/left/right; sort tiebreak determinism (re-running yields byte-identical placements); spacing |

Subtotal: **43 transform tests**.

Per new SVG-primitive / render module:

| Module | Tests | Coverage |
|---|---|---|
| `render/svg.rs` (additions) | 6 | image attribute order; polygon evenodd; beeswarm group; image href encoding; polygon multi-ring; beeswarm deterministic ordering |
| `render/marks/polygon.rs` | 3 | rendering one ring; rendering hole; multi-polygon batch |
| `render/marks/image.rs` | 3 | RGBA→PNG→base64 round-trip; cmap dispatch; image positioning via X/Y scale |
| `render/marks/ribbon.rs` | 3 | Y/Y2 path emission; opacity; interpolate=linear |
| `render/color/continuous.rs` | 3 | named lookup vs colorous reference; gradient interpolation in linear sRGB; Reverse |
| `render/rasterize.rs` | 3 | PNG byte determinism (same RGBA → same hash 3x); pixel ordering; large grid (1024×1024) |
| `render/prepare.rs` (additions) | 3 | named-output routing; missing-name error; default-fallback when data_source=None |

Subtotal: **24 render tests**.

Spec/binding tests:

| Module | Tests | Coverage |
|---|---|---|
| `spec/layer.rs` (additions) | 2 | data_source field round-trip; backwards-compat (None serializes as missing) |
| `spec/chart.rs` (additions) | 2 | Multi-output transform chain serialization; layer name resolution |
| `binding.rs` | 4 | Each new transform has a PyO3 wrapper with `__repr__` test (pattern from 8a) — 10 transforms but parametrized into one fixture, counted as 4 conceptual tests |

Subtotal: **8 spec/binding tests**.

**Cargo total: ~75 new tests → 384 total** (target ≥379 met).

### §8.2 Pytest tests (target ≥+180 new; pytest total ≥397)

Per new mark, dedicated test file under `tests/marks/`:

| Test file | Tests | Coverage |
|---|---|---|
| `test_boxplot.py` | 15 | basic; horizontal (CoordFlip); per-group color; outliers True/False; extent ∈ {min-max, 0.5, 1.5, 3.0}; size kwarg; error: y missing; per-mark style overrides; layer count assertion (3 or 4); CDI round-trip; 3 polars dtype variants (Int64/Float64/Utf8 group) |
| `test_errorbar.py` | 10 | each extent ∈ {ci, stderr, stdev, iqr}; ticks True/False; per-group; CDI; spec round-trip; layer assertions |
| `test_errorband.py` | 10 | as above + borders True/False |
| `test_ribbon.py` | 8 | basic; missing y2 error; opacity; interpolate; X/Y2 channel wiring; CDI; spec round-trip; 2 dtype variants |
| `test_smooth_ci.py` | 6 | lifts 8a deferral (no warning); two layers (ribbon + line) in spec; ribbon-then-line z-order; CI band data covers x range; loess vs lm methods; CDI |
| `test_contour.py` | 14 | lines mode; fill mode; thresholds=int variants; bandwidth ∈ {scott, silverman, float}; bivariate-density routing; smooth=True/False; saddle case (4-point cell); ring-with-hole (bimodal); per-mark style; layer count = 1 |
| `test_violin.py` | 14 | inner ∈ {box, quartile, point, None} (4); bandwidth variants (3); per-group color; horizontal; layer count assertions per inner mode (4); polygon symmetry; CDI |
| `test_qq.py` | 10 | distribution ∈ {normal, uniform, exponential} (3); line=True/False; dequantize; reference-line slope correctness; per-mark style; CDI; spec round-trip; 2 dtype variants |
| `test_raster.py` | 16 | aggregate ∈ {count, density, mean, sum, any} (5); resolution ∈ {"screen", 128, 256, (200, 100)} (4); cmap variants (viridis/plasma/magma); log_scale; min_count masking; blend="additive" warns; field= required error for mean/sum; CDI |
| `test_swarm.py` | 12 | orient=vertical/horizontal; side ∈ {both, left, right}; spacing variants (3); size kwarg; dodge warns; determinism (3 runs same output); large n (5000); CDI; per-mark style |
| `test_hex.py` | 12 | aggregate ∈ {count, mean, sum} (3); bin_size auto vs explicit; cmap variants; stroke / stroke_width; field error for mean/sum; vertex count assertion; CDI; spec round-trip; 2 dtype variants |
| `test_function.py` | 10 | explicit domain; inferred from sibling layer; np.sin/np.cos/lambda; n parameter; clip; missing-domain error; wrong-shape return error; numpy and python lists in/out; CDI; spec round-trip |

Subtotal: **137 mark tests**.

Cross-cutting:

| Test file | Tests | Coverage |
|---|---|---|
| `test_continuous_palette.py` | 8 | each named map (5); list(); .reversed(); Gradient with custom stops |
| `test_data_source_routing.py` | 7 | named-output routing; missing-name error; default-fallback; multi-output transform; round-trip; 8a charts unaffected |
| `test_image_primitive.py` | 4 | smoke; href starts with "data:image/png;base64,"; deterministic across runs; image positioned via scale |
| `test_polygon_primitive.py` | 4 | one ring; multi-ring (hole); multi-polygon batch; evenodd attribute |
| `test_beeswarm_primitive.py` | 3 | smoke; circle count; group wrapping |
| `desugar/test_composite_desugar.py` | 10 | each composite returns (transforms, layers) tuple shape; idempotent; horizontal-orientation flip |
| `desugar/test_heavy_stat_desugar.py` | 16 | each heavy stat returns expected (mark, transforms, encoding_remap, [synthetic_data]) shape; bivariate-density routing |
| `test_phase_8b_e2e.py` | 14 | one full-render assertion per new mark (11) + 3 multi-mark layered charts |
| `test_spec_drift.py` | 4 | PHASE_8B_MARKS set is empty; ferrum-spec dated notes for 8b present; transform count = 15 in TransformSpec enum exposure; ferrum-phases checklist updated |
| `test_warn_once_lift.py` | 3 | mark_smooth(ci=) no longer warns; new warn-once categories fire correctly; warn-once registry size = 8a + 4 |

Subtotal: **73 cross-cutting tests**.

**Pytest total: ~210 new → 427 total** (target ≥397 met with margin).

### §8.3 Goldens

- All 6 SVG goldens + 1 PNG hash from Phase 7 must remain byte-identical (regression check).
- Phase 8a goldens (multi-layer +/|/&) must remain byte-identical.
- New deterministic-output snapshot tests for the 3 new SVG primitives (image attribute ordering, polygon evenodd, beeswarm group).
- 2 new PNG hash goldens for raster output (one viridis 128×128, one plasma 256×256) — verifies png-crate version + Filter::Sub + level 9 stay pinned.

### §8.4 Performance smoke (advisory, not a gate)

For the larger transforms, document expected runtimes in test comments (not asserted, but flagged if significantly off):
- Kde2D 128×128 grid, 10K points: <50 ms
- Marching Squares contour, 128×128 grid, 8 thresholds: <20 ms
- Beeswarm, 5K points in one group: <100 ms
- Raster 256×256, 100K points: <30 ms

If any transform exceeds 5x its smoke target on the dev machine, surface it in the writing-plans step as a perf issue requiring algorithm refinement before merge.

### §8.5 Test count baseline at end of Phase 8b

| Suite | Phase 8a baseline | Phase 8b target | Phase 8b expected |
|---|---|---|---|
| `cargo test -p ferrum-core` | 309 | ≥379 | ~384 |
| `uv run pytest` | 217 (+ 3 skipped) | ≥397 | ~427 |

---

## §9 Done-criteria gate

Phase 8b is `done` when **all** of the following are true:

### Done criteria from `ferrum-phases.md` (verbatim, with one revision)

- [ ] All 4 composite marks (`mark_boxplot`, `mark_errorbar`, `mark_errorband`, `mark_ribbon`) work.
- [ ] All 7 heavy statistical marks (`mark_contour`, `mark_violin`, `mark_qq`, `mark_raster`, `mark_swarm`, `mark_hex`, `mark_function`) work.
- [ ] **10** new Phase 5 transforms (Outliers, ErrorExtent, BoxStats, Violin, **Kde2D**, Contour, QQ, Raster, Hex, Swarm) all have round-trip + correctness tests.
  - *Revision from 9→10:* `Kde2D` added as a separate transform per §10 decision. `ferrum-phases.md §"Phase 8b"` checklist updated in the same PR that lands this spec.
- [ ] New SVG primitives in `SvgBuffer` (image, polygon, beeswarm) emit deterministic SVG (snapshot tests pass on 3 platforms).
- [ ] `mark_smooth(ci=...)` CI band renders via the new ribbon mark; the 8a warn-once is removed.

### Additional gates (from this spec)

- [ ] `cargo test -p ferrum-core` passes with **≥379 tests** (current 309).
- [ ] `uv run pytest` passes with **≥397 tests** (current 217).
- [ ] All 6 SVG goldens + 1 PNG hash from Phase 7 byte-identical.
- [ ] All Phase 8a goldens (multi-layer +/|/&) byte-identical.
- [ ] 2 new PNG hash goldens for raster output (viridis 128×128, plasma 256×256) green and stable across 3 consecutive runs.
- [ ] `marks/deferred.py::PHASE_8B_MARKS` is the empty `frozenset()`; PHASE_9_PLUS_MARKS unchanged.
- [ ] `cargo tree -p ferrum-core | grep -i matplotlib` returns empty (no-matplotlib audit).
- [ ] `ferrum-spec.md` has dated 2026-05-10 notes for the 6 8b clarifications listed in §6.5.
- [ ] `ferrum-phases.md` Phase 8b row updated: status → `done`, done-criteria checklist all checked.
- [ ] No new `set_default_*` mutators introduced (themes-as-values invariant intact).
- [ ] `Chart(df).mark_<each>().encode(...).show_svg()` works smoke-tested for all 11 new marks (one assertion per mark in `test_phase_8b_e2e.py`).
- [ ] Pre-existing `_pending_stat_mark` deferral works for all 11 new marks regardless of method order: both `Chart(df).encode(x=..., y=...).mark_X()` and `Chart(df).mark_X().encode(x=..., y=...)` produce equivalent specs.

### Non-gates (deliberately not blocking)

- HConcatChart `show_png()` may still raise NotImplementedError (carry-over Phase 8a limitation).
- Performance smoke targets in §8.4 are advisory.
- Auto-raster policy (`raster_threshold`, `raster_behavior`) remains deferred to Phase 9+.

---

## §10 Locked decisions

These are settled. Re-litigation requires a written reason in the journal and an updated dated note in `ferrum-spec.md`.

| # | Decision | Where it appears |
|---|---|---|
| 1 | Single spec, single plan, one merge to `main` (mirror Phase 8a) | §2 |
| 2 | Composite marks desugar Python-side via `ChartSpec.layers` (no new "composite" Mark variant in Rust) | §3.2, §4.2 |
| 3 | `mark_smooth(ci=)` desugars to `[ribbon(ci_lower, ci_upper), line(y)]` over the existing Smooth transform | §4.8 |
| 4 | Raster image embedding: RGBA → `png` v0.18 (Filter::Sub, level 9) → base64 → SVG `<image href="data:image/png;base64,...">` | §3.4, §5.7, §6.4 |
| 5 | `resolution="screen"` for mark_raster resolved at render time = panel pixel size (Phase 8b, not deferred) | §3.4, §5.9 |
| 6 | Beeswarm: greedy sweep, deterministic via stable sort + tiebreak on original row index | §5.2 |
| 7 | Contour: lines + filled bands via Marching Squares (isoline + isoband). Filled mode uses `<path>` + `fill-rule="evenodd"` for holes | §3.3, §5.1 |
| 8 | `Kde2D` is a separate, 10th transform (not absorbed into Contour) | §3.1, §4.4, §6.5 |
| 9 | Continuous colormap subsystem: `colorous` workspace dep; 5 named maps + Gradient + Reverse | §3.6, §4.6, §7.1 |
| 10 | `mark_density` bivariate routes through `mark_contour(fill=True)` (single Contour codepath) | §4.3.6 |
| 11 | `mark_function`: Python-side eval; domain rule = explicit → infer-from-siblings → error | §3.5, §6.1 |
| 12 | New SVG primitives: `image()`, `polygon()` (path + evenodd, multi-ring), `beeswarm()` (batched circles) | §4.5 |
| 13 | `Layer.data_source: Option<String>` + `TransformSpec.name: Option<String>` for multi-output transform routing. None preserves 8a byte-identical behavior | §3.7, §4.7 |
| 14 | Composite-mark groupby inferred from categorical encoding (X for vertical box, Y for horizontal). Outliers also per-group | §3.2, §5.4 |
| 15 | `apply_with_context` is the only Phase 5 engine API addition; transforms ignoring context use the existing `apply` path | §5.9 |
| 16 | Test count targets: cargo ≥379, pytest ≥397 (current 309 / 217) | §8.5 |
| 17 | Auto-raster policy explicitly deferred to Phase 9+ (dated note in spec §3.3) | §2, §6.5 |
| 18 | `desugar_function` returns a 4-tuple; the 8a 3-tuple desugar contract is uniformly extended to 4-tuple (4th slot = optional synthetic data) | §4.1, §4.3.7 |
| 19 | Bootstrap CI uses `rand_chacha` seeded from `spec.seed: u64` (default 0) for byte-determinism across platforms | §5.5, §6.4 |
| 20 | `Hex` transform output is 6 vertices/cell (not center+render-time vertices) so the Polygon mark is geometry-agnostic | §5.3 |

---

## §11 Cross-phase notes

### Phase 5 (Stat engine) — what 8b extends
- 10 new variants in `TransformSpec` enum; 10 new files under `transform/`.
- Single API addition: `apply_with_context(spec, batch, context)` alongside existing `apply(spec, batch)`. Default impl forwards to `apply` ignoring context.
- Existing transforms (Bin/Kde/Smooth/Aggregate/Summary) unchanged — their JSON, output schemas, and PyO3 wrappers stay byte-identical.

### Phase 6 (Layout) — what 8b calls
- `apply_with_context` needs panel pixel size from layout. Layout solver passes panel rect through to the prepare/transform pre-pass via a new `TransformContext { panel_pixel_size: (u32, u32) }`. No layout-engine algorithm change; just a passthrough.

### Phase 7 (Static renderer) — what 8b extends
- 3 new `SvgBuffer` methods (image, polygon, beeswarm). Existing methods byte-identical.
- 3 new `render::marks::*` files (polygon, image, ribbon). Existing mark drawers untouched.
- New `render::color/` directory subsumes the existing `render/color.rs` with backwards-compatible re-exports.
- New `render::rasterize.rs` for PNG encoding (independent of existing resvg path used for chart-level SVG→PNG).

### Phase 8a (Grammar API) — what 8b lights up
- The 11 NotImplementedError stubs in `Chart` become working desugars.
- `_pending_stat_mark` deferral mechanism reused for all 11 new marks (no new deferral plumbing).
- `MarkKwargsSpec` per-mark style overrides reused for raster cmap, hex stroke, etc.
- 8a's warn-once registry extended (4 new categories, 1 removed).
- `ChartSpec.layers` field finally exercised by composite marks (was infrastructure-ready in 8a).
- X2/Y2 channel encoding (accepted but not rendered in 8a) wired through the renderer for ribbon and errorband.

### Phase 9 (Convenience API) — what 8b unblocks
- `displot(data, x, kind="hist")` → `Chart(data).mark_histogram().encode(x=x)` — already worked in 8a.
- `displot(data, x, kind="kde")` — already worked.
- `displot(data, x, y, kind="kde")` — bivariate KDE — *now* works via §4.3.6 routing.
- `lmplot(data, x, y, ci=0.95)` — `Chart(data).mark_smooth(ci=0.95).encode(...)` — *now* works (uses 8b's ribbon).
- `boxplot(data, x, y)` → `Chart(data).mark_boxplot().encode(...)` — *now* works.
- `pairplot(data)` — needs `RepeatChart` (Phase 9 itself) but each cell uses 8a/8b marks.
- Auto-raster policy is the single Phase 9 unlock that lives entirely in Phase 9 (uses 8b's `mark_raster` as its target).

### Phase 10 (Model diagnostics) — what 8b lights up
- `mark_residuals` will use `mark_smooth(ci=)` for the loess reference line (8b unblocks).
- `mark_calibration` reuses `mark_errorband` for the per-bin CI.
- `mark_learning_curve(ci_style="band")` reuses `mark_errorband`.
- `mark_pdp(kind="both")` reuses `mark_function` for the marginal curve overlay.
- `mark_decision_boundary` reuses `mark_raster` for the background.

### Phase 11 (Interactive) — what 8b enables
- `Chart.interactive()` will need to handle dynamic recompute for `mark_function` (zoom/pan over a callable). 8b's design intentionally evaluates fn once at chart-build-time; the interactive path will need a separate "callable-bound" mark variant. Note this in the Phase 11 spec when written.
- `mark_raster` resolution="screen" already plumbed to render-time; the interactive renderer just substitutes its own panel size.
- `mark_swarm` collision algorithm runs in Rust, so it stays in WASM (no recompute issues).

### Phase 12 (Extension points) — what 8b feeds in
- The `TransformSpec` registration pattern (one file + one enum variant + one PyO3 wrapper) becomes the documentation template for "custom stat transforms" extension.
- The `render::marks::*` file pattern becomes the template for "custom marks" extension.
- The `ContinuousScheme` enum pattern becomes the template for "custom colormaps" extension.

---

## §12 Spec refinements (post-approval, plan-stage)

This section is reserved for refinements that surface during plan drafting. Items added here resolve under-specified inputs without changing scope or any locked decision in §10.

### §12.1 Layered-desugar resolver contract

The composite-mark desugar functions in `marks/composite.py` and `marks/heavy_stat.py` return Python dicts for each layer (e.g., `{"mark": "rule", "encoding": {"x": ..., "y": ...}, "data_source": "box"}`) under the `"__layered__"` sentinel. The `Chart._resolve_pending_then_build_spec` resolver must convert these dicts into proper `Layer` PyO3 instances before constructing the multi-layer `ChartSpec`.

Conversion rules:
- `mark`: str → `Mark(name)` PyO3 instance.
- `encoding`: dict[str, str] → `Encoding(...)` instance with each field bound to the appropriate channel class (X/Y/X2/Y2/Color/Detail) by key name.
- `mark_kwargs`: dict → forwarded as-is to the `Mark` constructor's per-mark style overrides (existing 8a `MarkKwargsSpec` mechanism).
- `data_source`: Optional[str] → forwarded to `Layer.data_source` field (8b §3.7).

This contract must be implemented and tested as a standalone task **before** any composite mark depends on it (plan Task 22b).

### §12.2 `mark_function` scope restriction in Phase 8b

`mark_function` is restricted to **single-layer charts** in Phase 8b. The synthetic Arrow table generated by Python-side fn evaluation replaces the chart's data for that chart instance.

When `mark_function` is added as a layer via `+` composition (e.g., `Chart(df).mark_point() + Chart(other).mark_function(np.sin)`), the second chart can be single-layer (with synthetic data) and the compositor handles each chart's data independently. This works in Phase 8b because the SVG compositor (8a infrastructure) treats each composed chart as its own ChartSpec with its own data.

What is **deferred to Phase 9+**: a Layer.data field that lets a single ChartSpec mix per-layer data sources beyond the synthetic-replaces-chart-data path. This would be needed if the user wrote `Chart(df).mark_point().encode(x="t", y="t").mark_function(np.sin).encode(x="t", y="t")` as a single chart with two layers each from different data — that's a Phase 9 capability.

Add a dated 2026-05-10 note to `ferrum-spec.md §3.3 mark_function`:

> *(2026-05-10) Phase 8b: `mark_function` is restricted to single-layer charts. Use it as a separate `Chart(...)` composed via `+` to overlay on existing data. Per-layer data routing within one `ChartSpec` is deferred to Phase 9+.*

### §12.3 Quantitative color wiring in polygon mark drawer

The polygon mark drawer (Task 20) must honor a quantitative color encoding (`encoding.color` field bound to a numeric column). Required for `mark_hex` (color = "value" → continuous colormap) and `mark_contour(fill=True)` (color = "level_value" → continuous colormap or fixed-per-band).

The drawer should:
1. Read `layer.encoding.color` field (if present) from the input batch as f64.
2. Look up `mark_kwargs.cmap` (default "viridis") via the new `ContinuousScheme` API (§4.6).
3. Per polygon group, compute the color from the median value of that group's color column (since one polygon = one logical entity, e.g., one hex bin or one contour level).
4. Pass the resolved color into the `FillStroke` style instead of taking it from `mark_kwargs.fill`.

This is wired in plan Task 20 (polygon drawer) — *not* in Task 36 (X2/Y2). Task 36's title remains "X2/Y2 wiring" but its scope is unchanged.

### §12.4 QQ secondary_outputs — sanctioned multi-output transform pattern

The QQ transform (Task 16) emits two named outputs: `qq_main` (the (theoretical, sample) points) and `qq_line` (the reference line endpoints). The simpler architectural fit is a `secondary_outputs(&self, primary: &RecordBatch) -> PyResult<Vec<(String, RecordBatch)>>` method on `TransformSpec`, defaulting to empty `Vec` for all existing transforms; QQ overrides to compute the reference-line batch.

`render/prepare.rs::apply_transforms_named` is extended to also iterate `spec.secondary_outputs(&primary)` after applying each transform and register each `(name, batch)` pair into the named-outputs map.

This is the only multi-output transform in 8b. Future multi-output transforms slot in the same way. The mechanism is not exposed to Python users (only via the QQ wrapper's `emit_line=True` parameter).

This is a refinement of §3.7 (which described single-output named transforms as the only routing path). Update §3.7 in the next spec revision pass to reference §12.4 as the multi-output extension.
