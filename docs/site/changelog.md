# Changelog

All notable changes to Ferrum are documented here.

## Unreleased

*No unreleased changes.*

## 0.16.1

*2026-06-15*

This release makes several documented-but-broken features actually render. No new
public API was added; methods and parameters that previously stored their values
without effect now take effect, and silently-dropped inputs now fail loud.

### Added

- [`Chart.override(**kwargs)`][ferrum.Chart.override] now applies at render. It is a fail-loud escape hatch that injects presentation-spec values (per-axis/legend config, per-channel scales, mark style, coord, `width`/`height`) and **wins the cascade** over `configure_*`/theme. Previously the overrides were stored but never read. Unknown or misspelled paths now raise `FerrumOverrideError` with a closest-match suggestion.
- Per-channel axis and legend styling now renders. Every documented field on [`fm.Axis`][ferrum.Axis] and [`fm.Legend`][ferrum.Legend] — label/grid/domain/title colors and sizes, `orient`, `translate`, `min_extent`/`max_extent`, `tick_extra`/`tick_min_step`, `grid_opacity`, `title_orient`, `zindex`, `label_flush`, `label_overlap`; legend `symbol_size`/`symbol_stroke_width`, `label_color`, `label_limit`, `clip_height`, `row_padding`/`column_padding`, `padding`/`title_padding`, `offset` — now takes effect when set on an encoding, matching `configure_axis`/`configure_legend`. Previously most of these silently dropped. A misspelled per-channel key now raises.
- `configure_axis(...)` and `configure_legend(...)` gained the same orphan styling fields, so they can be set chart-wide too.

### Fixed

- Rotated x-axis tick labels are now end-anchored (were center-anchored and overlapped the plot) with an angle-aware gap, and the x-axis title clears them.
- `configure(axis_x=…)` per-axis styling is no longer clobbered by a general `configure(axis=…)`; a per-axis/legend `label_font_size` now sizes the reserved layout (not just the drawn text); `fm.Legend(label_color=…)` applies to continuous colorbars; `configure_title(subtitle_font_size=…, subtitle_color=…)` styling is honored.
- [`clustermap`][ferrum.clustermap] accepts the documented integer `z_score`/`standard_scale` values (`0`/`1`); they previously raised `TypeError`.

### Changed

- Per-channel axis/legend options are now typed structs with `deny_unknown_fields` (replacing an opaque pass-through map), so typos fail at render instead of silently dropping. Internal: axis style overrides were unified into one struct and the `orient` precedence moved to an `Option` sentinel.

### Deprecated

- `AxisConfig.x` / `configure_axis(x=…, y=…)` are deprecated and emit a `DeprecationWarning` — they never had any effect. Use [`Chart.axis(x=False)`][ferrum.Chart.axis] / `Chart.axis(y=False)` to show or hide an axis.

## 0.16.0

*2026-06-04*

### Added

- [`Chart.to_html()`][ferrum.Chart.to_html] returns the self-contained interactive HTML page as a string, byte-identical to what `save("out.html")` writes. It accepts the same `embed_wasm=`, `toolbar=`, and `raster=` keywords as the HTML save path. This completes the `to_*` converter family (`to_svg`, `to_png`, `to_html`).

### Changed

- Renamed the output converters `Chart.show_svg()` → [`Chart.to_svg()`][ferrum.Chart.to_svg] and `Chart.show_png()` → [`Chart.to_png()`][ferrum.Chart.to_png], establishing a clear convention across the export surface: `to_*` returns an in-memory value, [`save(path)`][ferrum.Chart.save] writes to disk, and [`show()`][ferrum.Chart.show] displays. The old `show_svg`/`show_png` names remain as deprecated aliases that emit a `DeprecationWarning` and will be removed after 0.16.0.

### Fixed

- Statistical transforms now accept **integer columns**. `mark_density` (1-D and 2-D KDE), `mark_qq`, `mark_hex`, and 2-D binning previously raised `column '<x>' must be Float64` on integer data (counts, ages, years, ranks); they now coerce numeric columns to floating point. Genuinely non-numeric columns still raise a clear error.
- [`jointplot(kind="hist")`][ferrum.jointplot] now renders a 2-D histogram instead of raising `unknown column 'bin_x_start'`. The center panel is encoded against the correct `Bin2D` output columns (`x_lo`/`x_hi`/`y_lo`/`y_hi`/`count`) so the bins tile the plane instead of collapsing.
- Removed two panic-prone internal downcasts in the 2-D binning transform; bad input now returns a normal error rather than risking a panic.

## 0.15.3

*2026-06-03*

### Fixed

- Interactive (WASM) reactive **domain-rescale** (focus+context brushing) no longer shears sibling panels or makes the overview line disappear during the brush. Each panel's marks now render with their own transform, so rescaling one panel leaves the others untouched.
- Interactive **box-zoom** (magnifying-glass tool) on a focus+context overview now rescales the detail panel instead of making it vanish. Wheel, pan, and box-zoom operate on the focus panel; a box-zoom drawn on the overview drives the detail's domain like a brush.
- Interactive **axis lines and stroked marks are antialiased** via 4× MSAA on the render pass, removing the occasional 1px axis-line seam/gap where ticks cross the baseline or adjacent facet panels meet. Silently falls back to the previous (non-MSAA) path on GPUs/browsers without 4× support — no regression there.

## 0.15.2

*2026-06-03*

### Fixed

- Interactive (WASM) line and area **stroke width is now constant under reactive domain-rescale** — the "ribbon"/variable-width artifact when a focus+context brush rescaled an axis non-uniformly is gone. Stroke width is applied in screen space and is invariant to the zoom/rescale transform. As part of this, interactive line joins are now beveled (the static SVG renderer is unchanged and keeps miter joins).

### Known limitations

- A few interactive-renderer issues remain tracked for a follow-up pass: the reactive-rescale brush-mode default, sibling panels shearing during a cross-panel rescale, and occasional axis-line segment breaks. These do not affect the static SVG renderer.

## 0.15.1

*2026-06-03*

A post-v0.15.0 audit (bug-hunters, seam auditors, and heavyweight cohesion reviews) found a cluster of incomplete-unification, sibling-drift, and silent-failure bugs in the flexibility-campaign surface. All are fixed here, each with a regression test.

### Fixed

- Finished the group-by-key unification: `transform_window`, `data_stack`, `stat_aggregate`'s join/pivot/aggregate paths now accept integer/uint/bool groupby columns (they previously collapsed every row into one group). A null group key now stays distinct from a real `0`/`false`/empty (FA-9), which also fixes a crash in `summary`/`boxplot`/`violin`/`errorband` when a groupby column contained a null
- Cross-panel reactive rescale (focus+context brushing) now zooms the detail panel **in-bounds** — the rescale is reprojected through the shared data domain instead of the source panel's pixels, which previously sent the detail marks off-screen
- Polar bars, ranged rects, and arc marks now honor the per-row `opacity`/`stroke_width`/`stroke_opacity`/`stroke_dash`/`fill_opacity`/`angle`/`tooltip`/`href`/`description` channels their sibling marks already supported
- `Axis(title=None)` and `Legend(title=None)` truly suppress the title (no phantom text node), matching the channel-level `fm.X(title=None)`
- `add_params`/`add_selection` raise a clear `TypeError` on wrong-typed arguments, and a selection colliding with a same-named parameter raises instead of silently shadowing it
- A non-finite (Inf/NaN) value in an `fm.param` domain raises a legible error naming the parameter; a `bind="legend"` selection registered via `add_params` is now wired correctly; the `ChartSpec` type stub covers all constructor arguments
- `mark_area` with an `x2` channel, and `mark_bar` with both `x2` and `y2`, now raise a clear error pointing to `mark_rect` instead of silently dropping the second extent

### Known limitations

- A single-panel reactive-rescale chart may still default to pan mode; click the box-select tool to brush (FA-17). The line "ribbon" appearance under a non-uniform rescale (FA-16) is tracked separately.

## 0.15.0

*2026-06-02*

### Added

- Reactive parameters: `fm.param`, `fm.when`, and unified `selection_interval`/`selection_point`, with `bind="legend"` toggles. Parameters drive `scale.domain`, `transform_filter`, and conditional encodings; they statically resolve to their initial value for SVG export and power a live WASM/JS runtime in interactive HTML (domain rescale, crossfilter, legend toggling)
- Polar `theta2`/`radius2` second extents with radial stacking, enabling annular wedges, coxcombs, and sunburst-style charts
- Figure-level `title`/`subtitle`/`caption` on composite charts via `.properties()`, rendered once around the composite rather than per panel
- Typed continuous scales (`LinearScale`, `LogScale`, etc.) auto-infer their domain from the data, matching the dict form
- `resolve=` on `vconcat`/`hconcat`; `pairplot` shares a single color domain across panels
- 2-D density (`kde_2d` / `mark_contour`) splits by categorical hue, drawing one contour set per group

### Changed

- **Breaking:** `transform_window(..., frame=(preceding, following))` now follows the documented Vega/Altair convention, where a negative `preceding` counts rows *before* the current row. A trailing window is `frame=(-k, 0)` (the `k` preceding rows through the current row). The sign was previously inverted, so trailing rolling aggregates silently produced all-null or forward-looking results. Pipelines that compensated for the inverted sign must drop the workaround.

### Fixed

- Continuous color scales bound through `SequentialScale`/`DivergingScale` now honor the named scheme, `domain`, and `clamp` (every scheme rendered blue before); added `reds`/`greens`/`oranges`/`purples`; diverging midpoint placed via piecewise normalization
- `mark_bar`/`mark_area` honor a second positional extent (`y2`/`x2`) for candlestick, floating, and diverging bars; `mark_bar(zero=False)` opts out of the zero-anchored y-domain
- Integer and nominal columns render on categorical channels (integer-keyed heatmaps, nominal bar `y`) and stack correctly; integer storage still defaults to quantitative
- Annotation span `label_position` (`top`/`middle`/`bottom`) is honored; unsupported `stack=` on non-stackable marks warns instead of silently dropping marks; empty facet partitions name the dropped key in the warning
- Two-way faceting (`.facet(row=, col=)`) renders a true grid with per-partition transforms, layered marks, and independent/shared scales — the row dimension was previously dropped
- Silent failures across the render pipeline now surface as warnings or errors instead of producing blank or wrong output
- `mark_line` `detail=` groups by non-Utf8 columns; axis titles use the source field name rather than the internal transform column
- `mark_violin` and `mark_area` honor color/hue instead of collapsing groups; per-layer `aggregate=` and `bin=` are no longer dropped on layered charts; `transform_top_k` aggregates integer columns instead of counting rows
- `fm.when` numeric conditionals apply instead of silently no-op'ing; radial bars stack outward and `stack=` is normalized across `x`/`y`/`theta`/`radius`
- `mark_arc` with nominal theta renders a Nightingale coxcomb; polar bars render equal full-circle angular bands; `stat_aggregate` and all stat transforms accept integer/uint/bool groupby keys; ordinal-color area legend swatches match fills; box-inner layers color-encode by hue
- `title=None` and `Axis(title="")` truly suppress axis titles; annotations anchor to categorical/ordinal axes
- errorbar/errorband compute per-hue extents instead of pooling across groups; unsupported mark shapes and stack offsets raise instead of silently defaulting
- Zoomed lines and areas stay clipped within their panel in interactive exports; two-way facet row-strip labels render on the right edge

### Other

- Expanded the docs-site Showcase with composed charts unlocked by the flexibility work; flattened the Gallery/Showcase navigation

## 0.14.0

*2026-05-31*

### Added

- `catplot(height=, aspect=)` for per-panel facet sizing, matching `displot`
- Automatic temporal scale inference from `Datetime`/`Date` dtypes (no explicit `:T` needed); microsecond/nanosecond timestamps normalized to milliseconds
- Size and shape legends, with multi-legend stacking and same-field legend merging

### Fixed

- Float ordinal domains, displot facet width, and boxen sort on pandas/pyarrow inputs
- Explicit constant stroke now beats an inherited color encoding on rule/segment marks
- 3- and 4-digit hex colors, `~e`/`~s` scientific-notation trimming, ranged-mark per-row color, and raster Y-axis orientation
- Deterministic batch selection in `locate_field` for named transform outputs
- Integer/float ordinal columns render correctly; per-panel facet sizing on `displot`
- Channel-level `axis=None` hides the axis
- Date/datetime/ISO coordinates accepted in annotations
- Color routes to stroke on line-family marks
- Value-sort and the full sort spec honored on axes and composite marks
- Order-independent color-scale merge on layered charts
- Per-channel axis `label_format` with the full d3-format grammar and chrono time formatting
- Categorical color-string ranges; rect/heatmap scheme routing

### Changed

- Boxen sort moved into the Rust `LetterValue` transform (new `sort=` kwarg); Python shim removed
- Typed ordinal `scale.range` wire format
- Unified categorical-axis sort resolution; removed the dead errorbar `y_sort`
- New `ChannelBase.option()` accessor; dropped private `_kwargs` reach-ins
- Deduplicated inferred-type application; reconstruct the promoted color channel on layer merge
- Shared fill/stroke color-resolution helpers across marks
- Colorbar tick formatting routes through the full d3-format grammar
- Extracted `build_default_categorical_scale`

### Other

- Fixed docstring parameter drift across the public API; added a docstring-drift auditor (`scripts/audit_docstring_drift.py`)
- Added a capabilities showcase docs page for power-user chart designs
- Regenerated configure goldens for corrected 3-digit hex colors

## 0.13.0

*2026-05-30*

### Added

- `ferrum.Grid` value class for theme-level gridline control: major/minor levels with per-level styling (`major_color`/`minor_color`, `major_width`/`minor_width`, dash, opacity) plus bare `color`/`width`/`dash`/`opacity` shorthand that sets both levels
- Minor gridlines on continuous axes (linear, log, time, pow, sqrt, symlog); log axes place minors at the 2–9 intra-decade multiples; categorical/discretizing axes have no minors
- Constant mark-style kwargs on composite/statistical marks — `mark_density(opacity=0.4)`, `mark_smooth(stroke_width=2)`, `mark_boxplot(fill=...)`, etc. now work like simple marks, applied to every emitted layer
- `mark_hex(stroke=, stroke_width=)` renders hex-cell borders
- Native Rust diagnostic curve kernels (ROC, precision-recall, calibration, confusion matrix, threshold sweep); raw-array (`y_true=`/`y_pred=`) diagnostics are now fully scikit-learn-free
- `SceneNode::Raw` rendering in the WASM interactive renderer (chrome-vs-data anchoring) and a dedicated `MarkBatchKind::Label` scene-graph kind

### Fixed

- Continuous-axis gridlines and ticks now coincide with the data marks they label (previously used uniform-slot placement, causing visible misalignment on continuous scales)
- WASM colorbar gradient id collision when a colorbar chart was placed in an inset (outer and inset colorbars collapsed to one id)
- Diagnostic kernels reject null-containing Arrow inputs instead of silently mishandling them
- Degenerate single-class ROC matches scikit-learn's NaN convention

### Changed

- Diagnostic chart paths (precomputed and model-backed) route through the shared Rust kernels; scikit-learn is now required only when a fitted model is passed
- Internal cohesion refactors (no API change): post-review cleanups in the scale-projection/grid code, consolidated diagnostic frame assembly into a shared `_curve_frames` module, and unified precision/recall cores in the PR kernel

### Other

- Documented `ferrum.Grid`, minor gridlines, and composite mark-style kwargs across the guides and API reference; clarified that raw-array diagnostics need no scikit-learn; regenerated guide PNGs for the gridline-coincidence rendering

## 0.12.0

*2026-05-24*

### Added

- Declarative configuration surface: `configure_axis`, `configure_legend`, `configure_title`, `configure_grid`, `configure_padding`, `configure_color`, and unified `configure()` methods on `Chart` and all composition types
- Annotation rendering module with coordinate resolution (text, arrow, rect, line, span, bracket, callout, image)
- Structural feature rendering: `SecondaryY`, `BreakAxis`, `Inset` with full composition support
- ChartConfig domain fields wired into render pipeline
- `label_padding` theme key, per-side padding support

### Fixed

- Break-axis x-label disappearance and y2 right-margin clipping
- Inset viewBox scaling and span fill_opacity
- Format-presets revenue chart uses nominal encoding for monthly bars
- Break-axis remaps same-axis ticks, SecondaryY bar marks with nominal x
- Inset recipe PNG referenced wrong image
- ColorConfigSpec.domain accepts string and float values
- Annotation text font-family rendering
- 10 bug-hunt R3 bugs (26 failing tests) and 21 bug-hunt bugs across coercion, composition, figures, interactive, scale, projection, render, and layout
- 31 bug-hunt bugs from configure field wiring

### Changed

- **Major internal refactoring (no API changes):**
  - `chart.py` split from 5,779 to 2,794 lines via `StatisticalMarksMixin` (15 methods) and `DiagnosticMarksMixin` (26 methods)
  - `configure_*` methods unified into `ConfigureMixin`, eliminating ~635 lines of duplication between `Chart` and `_ChartLike`
  - `Chart._clone` auto-generated from `__slots__` (prevents silent data loss)
  - `to_spec()` decomposed into 3 focused helpers
  - Rust `ThemeInputs` decomposed from 42 flat fields into 9 sub-structs
  - Rust `PreparedInputs` legend overrides grouped into `LegendPreparedOverrides` sub-struct
  - Rust `render_svg`/`render_scene_json` deduplicated via shared `prepare_and_layout` pipeline (fixes missing secondary-Y padding in interactive renders)
  - Rust `scale_resolve.rs` (2,205 lines) split into 6 sub-modules
  - Rust `ScaleSpec` common fields factored into `ContinuousScaleCommon`
  - Rust `Encoding::inherit_*` duplication eliminated
- Added `cargo_test` nox session with correct macOS DYLD paths

### Other

- Cohesion/complexity audit report and refactor plan
- Docs: customization guide, 7 concept pages, 12 recipes, 16 golden SVGs
- Regression tests for break-axis labels, y2 right-margin, font-family, and auditor-found bugs

## 0.11.2

*2026-05-23*

### Added

- Graduated axis label collision cascade — replaces flat→-45°→elide with wrap → font shrink → graduated rotate → tick cull → elide, keeping categorical labels legible without user intervention
- Multi-line label wrapping: snake_case, space, and camelCase labels split at natural boundaries instead of truncating
- Dynamic bottom margin: rotation-aware gutter estimation prevents rotated labels from clipping at the viewport boundary
- Theme-configurable `cull_threshold` for tick label density reduction (`Theme(cull_threshold=N)`)

### Fixed

- Y-axis title duplicated across all facet columns — now suppressed on non-leftmost columns (mirrors existing x-axis bottom-row suppression)
- Panel annotations silently dropped from SVG output — now emitted after marks, matching WASM z-order

## 0.11.1

*2026-05-22*

### Fixed

- PNG export: DPR-aware canvas with pHYs DPI metadata replaces the old DPR-upscale approach, fixing gridline artifacts in composed charts (HConcat) caused by stale pixel-snap positions after canvas resize
- PNG export: full Retina resolution via 2D offscreen canvas — wide charts clamped by GPU max texture size now export at correct @2x dimensions with proper DPI metadata
- Canvas initialization: safe three-phase init (CSS size → GPU create → DPR-clamped resize) prevents `Surface::configure` crash on wide HConcat charts exceeding GPU max texture size
- Mouse coordinate mapping uses scene dimensions instead of canvas backing store, fixing hit-test and tooltip positioning on DPR-aware canvases

## 0.11.0

*2026-05-22*

### Added

- R-tree spatial indexing (`rstar` crate) for O(log n) hit-testing on large point clouds
- Interactive toolbar: Pan, Box Zoom, Box Select, Reset, Save PNG buttons (Bokeh-style)
- `Chart.interactive(toolbar=True/False)` and `Chart.save(toolbar=True/False)` kwargs
- Auto-tooltips: interactive renders automatically inject tooltip fields from encoded channels
- `WasmRenderer.maxTextureSize()` WASM export for adaptive DPR capture
- Keyboard shortcuts for toolbar modes (P/Z/S/R/Escape)

### Fixed

- Opacity semantics: `fill_opacity` used correctly for packed batches (>1000 marks), `stroke_opacity` applied in tessellated paths/polygons, no double-apply on strokes
- Grid pixel-snap in WASM renderer eliminates sub-pixel aliasing
- Mark clip region: GPU scissor rect clips zoomed marks to panel plot area (DPR-aware)
- Tick label clipping: zoomed tick labels filtered by panel plot area
- Save PNG: adaptive DPR clamping to GPU max texture size (prevents crash on wide charts)
- Save PNG: ResizeObserver disconnected during capture (prevents race condition)
- Save PNG: SVG viewBox uses actual canvas dimensions (not stale scene dimensions)
- Annotation z-order: reference lines drawn above data marks (matching SVG renderer)
- Reset button forwards selection state to Jupyter kernel
- Cursor CSS targets SVG overlay in select/boxzoom modes
- HTML title XSS: `html.escape()` applied to `<title>` tag
- Hex color shorthand: 3-char (`#ccc`) and 4-char (`#abcd`) correctly expanded
- Malformed hex colors now emit a warning instead of silently returning black
- `_render_scene_json` collapsed into `_render_scene` (eliminated duplication)

### Changed

- `SceneCollector` accumulator replaces 8-parameter `collect_nodes`/`emit_draw_commands`
- Uniform clip vec4 removed from shaders — GPU scissor rect handles clipping (32-byte uniforms)
- `text_json.rs` extracted from `lib.rs` with deduplicated text serialization helpers
- `lib.rs` split: tooltip formatting, conditional render, transform upload moved to focused modules
- Partial GPU buffer update for conditional re-render (only re-uploads instance data)
- Dead packed-batch linear fallback removed from hit-test (spatial index covers it)
- Hand-rolled JSON replaced with `serde_json` in hit-test and tooltip paths
- `_auto_tooltips` removed from public `Chart.to_spec()` signature
- `_ChartLike.save()` uses explicit `toolbar` parameter instead of `kwargs.pop`
- `ResizeObserver` reads CSS layout size (not backing-store size)
- Transition RAF loop cancels stale animations on rapid scene changes

## 0.10.0

*2026-05-20*

### Added

- `.labs()` fluent method for post-hoc axis labels and titles
- `.xlim()` / `.ylim()` convenience methods for axis limits
- `.to_dict()` on Chart (returns parsed JSON spec as a Python dict)
- `mark_circle()` / `mark_square()` convenience methods (aliases for `mark_point(shape=...)`)
- `mark_line(point=True)` overlays points on lines (altair parity)
- `color=` and `alpha=` mark kwarg aliases (maps to `fill` and `opacity`)
- `linetype=` mark kwarg alias with named values (`"dashed"`, `"dotted"`, `"dashdot"`, `"longdash"`, `"solid"`)
- `Axis(label_map={"a": "Group A"})` categorical label remapping
- `annotate_abline(slope, intercept)` for slope+intercept reference lines
- PNG `scale=` parameter on `show_png()` and `save()` for DPI control
- PDF export via `save("chart.pdf")` (zero-dependency minimal PDF writer)
- `reverse=` on continuous positional scales
- 5 new aggregate functions: `variance`, `stdev`, `q1`, `q3`, `distinct`
- 4 KDE kernels: `epanechnikov`, `tophat`, `cosine` (plus existing `gaussian`)
- Smooth method aliases: `"linear"`, `"quadratic"`, `"cubic"`, `"log"`, `"sqrt"`
- `"|"` / `"-"` (vline/hline) point shapes
- 148 CSS named colors in theme overrides (e.g. `Theme(mark_color="steelblue")`)
- Altair migration guide (`docs/site/comparison/altair.md`)
- Pandas/polars `Series` accepted as data input
- Polars `Categorical` / `Enum` columns auto-cast to string
- Polars `Duration` columns auto-cast to int64 nanoseconds
- PyArrow `Date32` / `Date64` columns auto-cast to timestamp
- Composite mark kwargs forwarding (errorbar, boxplot, errorband, ribbon, boxen)
- Mark tick rendering for ordinal-y-only and ordinal-x-only encodings

### Fixed

- `type_="Q"` kwarg on channel constructors was silently dropped
- `count():Q` aggregate shorthand crashed (empty field rejected by Rust)
- Aggregate shorthand `mean(val):Q` missing auto-groupby from sibling channels
- `nice=True` on Scale constructors silently dropped during serialization
- `mark_smooth(method="linear")` documented but crashed (Rust only accepted `"lm"`)
- Non-Int64 integer columns failed when encoded as nominal (pyarrow/pandas users)
- Config defaults (640x480) didn't match render defaults (was hardcoded 600x400)
- Shorthand parser rejected hyphens, dots, and spaces in column names
- `ChannelName` enum missing `StrokeOpacity` / `FillOpacity` / `Angle` variants (conditional encodings on extended channels were broken)
- WASM conditional resolver missing match arms for 3 extended channels
- `mark_density(kernel=...)` never forwarded kernel to Kde transform
- `_apply_label_maps` broad `except Exception` narrowed to specific types with warning

### Other

- Plotnine/Altair migration gap audit (`design-docs/superpowers/audits/`)
- 226 new regression tests across 6 test files
- Overhaul README with hero strip, benchmarks, comparison table, and theme grid
- Embed interactive demos in docs
- Add plotnine to benchmarks, comparison table, and migration guides
- 10M-row stress tests for scale ceiling validation

## 0.9.1

*2026-05-18*

### Added

- Auto-inject selection fields into tooltip for cross-panel linking
- D3 interaction layer, SVG text rendering, and cross-panel linked selection
- Interactive HTML export with interval selection, composition support, and adapter pattern
- Nox `docs` session for docs site build verification

### Fixed

- Resolve 13 scene-pipeline audit findings across all 4 stages
- Include tooltip field names in SVG output and escape Group attrs
- Wire `strip_text_size` and `strip_padding` theme keys end-to-end
- Resolve 4 PyO3 binding audit findings (B1, B2, W1, W4)
- Bug-hunt across 5 interactive subsystems — 189 tests, 7 bugs fixed
- Resolve 9 wiring-audit findings in interactive HTML export (B1-B5, W1, W3, W6, W8)
- Deduplicate and harden interactive HTML export pipeline (Rust + Python)
- Eagerly populate `LAYER_NAME_CATALOG` from all mark modules
- Embed Inter font in shared CSS for Jupyter and HTML export
- sRGB-to-linear color conversion for correct GPU rendering
- Rebuild WASM artifacts with B4 `startTransition` fix

### Changed

- Rename audit skills/agents to prefix pattern (`audit-*`, `auditor-*`)
- Docs site: add classification meta tags, update author metadata, plain display names for API nav

## 0.9.0

*2026-05-17*

### Added

- **Phase 12: Spec completeness** — full implementation of all remaining `ferrum-spec.md` features
- **Data transforms API** — 17 transform constructors (`FilterTransform`, `AggregateTransform`, `WindowTransform`, `BinTransform`, `CalculateTransform`, `FoldTransform`, `PivotTransform`, `StackTransform`, `FlattenTransform`, `SampleTransform`, `ImputeTransform`, `DensityTransform`, `RegressionTransform`, `LoessTransform`, `QuantileTransform`, `LookupTransform`, `TimeUnitTransform`) with Rust-backed execution and expression evaluator
- **New scale types** — `PowScale`, `BandScale`, `PointScale`, `SequentialScale`, `DivergingScale`, `QuantizeScale`, `BinOrdinalScale`
- **Power/sqrt transform** — `PowScale(exponent=0.5)` for square-root position resolution
- **`Axis` and `Legend` value classes** — full configuration objects for axis and legend customization
- **`ferrum.color` and `ferrum.config` modules** — color utilities and configuration API
- **`LayerChart` and `ConcatChart`** — first-class composition types for programmatic multi-view construction
- **9 infinity domain regression tests** — continuous-x bar width, infinity filtering across mark types
- **Docs: 4 recipe pages** — transforms, ConcatChart, Axis/Legend, PowScale with rendered PNGs
- **Docs: API reference pages** — transforms, scales, color, config, axis, legend
- **Docs: data transforms guide**

### Fixed

- **Infinity domain filtering** — `±inf` values no longer corrupt scale domain auto-computation
- **Continuous-x bar width** — bars with continuous x-axis and no `x2` now auto-compute width
- **Safe Rust downcasts** — temporal type handling and Time bar rendering hardened
- **Python type annotations and dead code** — review-lite findings addressed across Python surface

### Changed

- **Design-docs reorganization** — `design-docs/` split into `architecture/` and `narrative/` subdirectories
- **Code archaeology updated** — Phase 12 items marked resolved, deferred marks inventory refreshed
- **Documentation expanded** — Phase 12 pages, cross-reference links, missing docstrings added

## 0.8.3

### Added

- pytest-xdist for parallel test execution (`-n auto`)
- 132 new tests from 5-round test-sweep campaign (scale, transform, composition, facet, theme, coord, encoding dimensions)
- Scale tests for SHAP, ICE, and PDP figure functions

### Fixed

- Multi-layer scale domain union — layers now merge domains correctly instead of using only the first layer's range
- TickLevel infinity serde — tick-level values no longer panic on serialize/deserialize with infinity bounds
- Coord system × position adjustment interaction (dodge offsets under CoordFlip)
- Figure function × data shapes gap (sweep-1)
- Missing `packed_instances` field in scale-stat test fixture
- Stale `docs/superpowers` paths updated to `design-docs/superpowers` in CLAUDE.md

### Changed

- Test-sweep skill: parallel Rust test track, coding agent dispatch enforcement

## 0.8.2

### Added

- `/test-sweep` skill — iterative combinatorial test-and-fix campaign
- `python-coder` and `rust-coder` coding agents for language-specific dispatch
- Plotly added to scatter benchmark suite
- Social card generation script with iris lmplot

### Fixed

- **Per-instance channel rendering** — `opacity`, `stroke_width`, `fill_opacity`, `size`, and `angle` encoding channels now affect SVG output for all 9 primitive marks (previously only `mark_point` had full wiring)
- **All-null data handling** — charts with entirely null/NaN columns now render an empty chart gracefully instead of crashing with `ValueError`
- **Histogram auto-groupby** — `displot(hue=)` and `jointplot(hue=)` no longer crash; the Bin transform now preserves the color-groupby column
- **RepeatPlaceholder guard** — `pairplot` mark overrides no longer crash when encodings contain unresolved repeat placeholders
- **Chained layer column-overlap** — `mark_rule` layering with overlapping column names resolved
- **Scale constructors** — `range` parameter now optional on all scale constructors
- **License packaging** — NOTICE file declared in `license-files` for sdist inclusion

### Changed

- Benchmark tables updated with Plotly results (4-library comparison)
- Documentation expanded: 25 new-user gaps addressed, API reflinks, stale gallery PNGs regenerated

## 0.8.1

### Fixed

- **Scale `range` now optional** — `LinearScale`, `LogScale`, `SymlogScale`, `TimeScale`, and `OrdinalScale` no longer require `range=` in their constructors. The renderer auto-fills from the plot-area dimensions, so users can write `fm.LogScale(domain=[100, 100000], base=10)` without pixel math.
- **`mark_rule` layering with overlapping columns** — layering a rule on a scatter (`scatter + hline`) no longer draws a line per scatter row when both DataFrames share a column name (e.g. `"y"`).
- **Chained `+` layering** — `scatter + hline + vline + label` now renders correctly. Added `Identity` transform and `inherit_non_positional` to prevent chart-level positional encoding from polluting routed layers.
- **Docs code examples** — fixed `CoordFlip` encoding order, `stroke_dash` string→list, `Stack` Int64→Float64, `to_theme_inputs_dict()`→`to_spec_dict()`, `chart.render_config()`→`chart.properties(render_config=)`, `feature_importances_chart`→`importance_chart`, `ferrum[...]`→`ferrum-viz[...]`.

### Added

- **25 new documentation sections** — type suffix explanation, position adjustments (Dodge/Stack/Jitter), axis customization (log scale, limits, reversed), smooth method table, legend control, palette cycling, shared scales, regplot, multi-model comparison, precomputed scores, output methods summary, DPI note, PDF note, interactive compatibility, CoordFlip recipe, annotation recipe, chart sizing recipe, category sorting recipe, time-series recipe.
- **13 guide PNGs** — visuals for every new section (position adjustments, axis customization, legend suppression, shared scales, regplot, multi-model ROC, CoordFlip, annotations, sizing, category order, time-series).
- **12-theme visual grid** — side-by-side comparison of all built-in themes on the same chart.
- **Complete 54-mark table** — all mark methods listed with reflinks in marks-encodings guide.
- **Consistent API reflinks** — every public symbol mentioned in prose now links to its API reference across all doc pages.
- **`Identity` transform** — Rust pass-through transform for named-output routing in layered compositions.
- **11 regression tests** — covering scale range, rule layering, and chained layer fixes.
- **PyPI metadata** — keywords, classifiers, and project URLs added to `pyproject.toml`.

### Changed

- **README** — updated install section with `ferrum-viz[all]`, added docs link, interactive rendering feature, corrected dev commands.

## 0.8.0

Grammar-of-graphics core with Rust rendering engine.

### Added

- **12 built-in themes** — Paper Ink (default), Slate Citrus, Arctic Signal, Observable, Minimal, Dark, Publication, Economist, FiveThirtyEight, Solarized Light, Solarized Dark, plus explicit `paper_ink` for derivation.
- **3 original categorical palettes** — `paper_ink`, `slate_citrus`, `arctic_signal` (8 colors each).
- **9 continuous color schemes** — `cool_blue`, `warm_ochre`, `night_blue`, `electric_lime`, `signal_blue`, `ember_orange`, `blue_to_red`, `cyan_to_amber`, `blue_to_violet`.
- **[`fm.hconcat()`][ferrum.hconcat] / [`fm.vconcat()`][ferrum.vconcat]** — top-level convenience functions for multi-chart concatenation.
- **[`Chart.axis()`][ferrum.Chart.axis]** — method for suppressing or restoring axis decorations.
- **Facet-before-transform** — faceting now partitions data before statistical transforms run, so each panel gets its own transform subset.
- **Grouped smooth** — `mark_smooth` supports group-by; no explicit per-group layering needed.
- **Continuous colorbar** — rendered alongside heatmaps and continuous-color charts.
- **Full Theme key wiring** — all `ferrum-spec.md` §3.13 keys plumbed end-to-end from Python through Rust renderer. Unknown keys raise `ValueError` at construction.
- **Model diagnostics (Phase 10)** — [`ModelSource`][ferrum.ModelSource], [`ComparedModelSource`][ferrum.ComparedModelSource], 32 visualizer classes, 31 figure-level helpers covering classification, regression, feature explanation, model selection, and clustering.
- **9 figure-level helpers** — [`displot`][ferrum.displot], [`catplot`][ferrum.catplot], [`relplot`][ferrum.relplot], [`lmplot`][ferrum.lmplot], [`residplot`][ferrum.residplot], [`pairplot`][ferrum.pairplot], [`heatmap`][ferrum.heatmap], [`clustermap`][ferrum.clustermap], [`jointplot`][ferrum.jointplot].
- **54 mark methods** — primitives, statistical, distribution, uncertainty, scale-aware, and diagnostic marks.
- **DataFrame pluralism** — polars, pandas, modin, cuDF, dask, ibis, and pyarrow all accepted via `Chart(data)`.
- **Title rendering** — `Chart.properties(title="...")` rendered in the SVG with theme-controlled typography.
- **Grid lines** — theme-controlled grid rendering with configurable color, width, dash, and opacity.
- **Legend** — categorical and continuous legends with theme-controlled positioning and styling.
- **Interactive rendering (Phase 11)** — [`Chart.interactive()`][ferrum.Chart.interactive] switches to a GPU-backed WASM renderer with selections, zoom/pan, linked views, and tooltips. Backed by `anywidget` for Jupyter integration.
- **Selection API** — [`selection_point`][ferrum.selection_point], [`selection_interval`][ferrum.selection_interval], [`selection_single`][ferrum.selection_single], [`selection_multi`][ferrum.selection_multi] for declaring interactive state. Conditional encodings via `sel.when(...).otherwise(...)`.
- **[`SelectionMark`][ferrum.SelectionMark]** — configurable brush styling for interval selections.
- **`InteractiveChart`** — anywidget-based Jupyter widget with `on_selection_change` callback and self-contained HTML export via `.save()`.
- **Scene graph renderer** — Rust-side `render_interactive` produces a SceneGraph JSON consumed by the WASM GPU renderer.
- **`compose_svg_horizontal` / `compose_svg_vertical` / `compose_svg_grid`** — low-level Rust SVG composition helpers.
- **t-SNE and UMAP in pure Rust** — [`ManifoldVisualizer`][ferrum.ManifoldVisualizer] runs both via `manifolds-rs`, no Python `umap-learn` dependency.
- **7 new diagnostic helpers** — [`classification_report_chart`][ferrum.classification_report_chart], [`class_balance_chart`][ferrum.class_balance_chart], [`cooks_distance_chart`][ferrum.cooks_distance_chart], [`prediction_error_chart`][ferrum.prediction_error_chart], [`silhouette_chart`][ferrum.silhouette_chart], [`elbow_chart`][ferrum.elbow_chart], [`manifold_chart`][ferrum.manifold_chart].
- **[`mark_label`][ferrum.Chart.mark_label]** — text labels with collision avoidance (`avoid_overlap=True`).
- **[`mark_image`][ferrum.Chart.mark_image]** — image tiles from URL fields.
- **[`RenderConfig`][ferrum.RenderConfig]** — per-chart auto-raster policy configuration (threshold, behavior, aggregate, colormap).
- **`raster=` keyword** — one-off auto-raster override on `.show()`, `.save()`, `.show_svg()`, `.show_png()`.
- **`score()` on visualizers** — all Group A visualizers implement `.score()` for sklearn-protocol compatibility.
- **`width=` on [`mark_boxplot`][ferrum.Chart.mark_boxplot]** — API symmetry with other marks.
- **`stroke`, `angle`, `fill_opacity` channels** — wired end-to-end to SVG attribute emission and WASM renderer.
- **Legend `format=` and `columns=`** — kwargs now wired through to legend rendering.
- **Packed tooltips** — field-level tooltip content sent via binary buffer for interactive performance.
- **Binary instance bridge** — GPU data bypasses JSON serialization in interactive rendering.

### Fixed

- **Facet encoding** — `facet=`, `facet_col=`, `facet_row=` now correctly partition data into panels.
- **`mark_tick` y-rug** — single-axis tick marks render correctly in both x and y orientations.
- **CoordFlip rendering** — no longer drops violin paths or boxplot rects under coordinate flip.
- **Histogram `multiple='stack'`/`'fill'`** — bin edges now align across groups.
- **[`calibration_chart`][ferrum.calibration_chart] rendering** — layering wiring gap resolved; renders correctly via `.show_svg()`.
- **Chart decomposition** — `chart.py` split into rendering, encoding helpers, and composition helpers for maintainability.

### Changed

- **`+` operator always layers** — no longer ambiguous between layering and concatenation. Use `|`, `&`, [`fm.hconcat()`][ferrum.hconcat], or [`fm.vconcat()`][ferrum.vconcat] for concatenation.
- **Default theme identity is Paper Ink** — warm cream background, blue marks, warm grid. Previously was Observable-style white background.
- **Default categorical palette is `paper_ink`** — previously `okabe_ito`.
