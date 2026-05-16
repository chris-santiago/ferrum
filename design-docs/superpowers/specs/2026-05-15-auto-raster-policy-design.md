# Auto-Raster Policy Design Spec

## 1. Scope

Implement the auto-raster policy described in ferrum-spec.md §3.3 and §3.17: when a chart layer's mark count exceeds a configurable threshold, the rendering pipeline transparently substitutes the per-element mark (e.g., 1M `<circle>` elements) with a rasterized representation (`mark_raster`-style pixel aggregation), producing compact output regardless of input size. This closes the gap between the spec's "three rendering backends" promise and the current reality where `show_svg()` unconditionally emits per-element SVG.

## 2. Goals

- Charts with >500k marks render to compact output (<2MB) without user intervention.
- `show()`, `show_svg()`, `show_png()`, and `save()` all participate in the policy.
- Users are warned by default when auto-raster fires (configurable).
- Explicit `mark_raster()` remains the opt-in path; auto-raster is the safety net.
- No behavioral change for charts below the threshold.
- The WASM/interactive path (`interactive()`) is unaffected — it already handles scale via GPU instancing.

## 3. Non-goals

- Direct-to-pixel rendering that bypasses SVG entirely (future optimization; current tiny-skia path rasterizes from SVG and is adequate for auto-raster output since the substituted SVG is small).
- Auto-raster for faceted charts where each facet independently exceeds the threshold (treat the total mark count across all panels).
- Automatic decimation or LOD for line/area marks (polyline vertex count, not mark count, drives SVG size for these; separate concern).

## 4. System behavior

### Mark counting

After transforms resolve and before rendering begins, count the total scene elements that would be emitted. For per-element marks (point, bar, rect, tick, rule, segment, text), the count equals the row count of the resolved data. For aggregate marks (hex, raster, histogram, area, line, ribbon), the count equals the output group/bin count, which is always small. Faceted charts sum across all panels.

### Substitution decision

Auto-raster fires when ALL of the following hold:

1. `mark_count >= raster_threshold` (default 500,000).
2. The mark type is a per-element type: `point`, `bar`, `rect`, `tick`, `rule`, `segment`.
3. The layer has **no active color encoding** (spec §3.3: "Auto-raster will not fire if the chart has an active color encoding — doing so would silently discard user intent").
4. The layer has both `x` and `y` quantitative encodings (raster requires a 2D numeric field).

When condition 3 or 4 fails but mark count exceeds the threshold, the policy emits a warning with guidance: "Chart has N marks which may produce large output. Use `.mark_raster()` for efficient rendering, or set `raster_threshold=None` to suppress this warning."

### What happens when auto-raster fires

1. The original mark and its encodings are replaced with a `mark_raster(aggregate="count", cmap="viridis")` equivalent.
2. Tooltip encodings on the original layer are dropped.
3. A density colorbar legend replaces the original legend.
4. A `UserWarning` is emitted (when `raster_behavior="warn"`): "Auto-raster: substituted mark_raster for mark_point (1,000,000 marks > threshold 500,000). Set raster_threshold=None to disable."

### Behavior matrix by call site

| Call | Below threshold | Above threshold (auto-raster eligible) | Above threshold (ineligible) |
|------|------|------|------|
| `show_svg()` | SVG as-is | Auto-raster substitution → compact SVG | SVG as-is + warning |
| `show_png()` | SVG→PNG | Auto-raster substitution → compact SVG→PNG | SVG→PNG + warning |
| `show()` | SVG inline | Auto-raster substitution → SVG inline | SVG inline + warning |
| `save("x.svg")` | SVG | Auto-raster substitution → compact SVG | SVG + warning |
| `save("x.png")` | SVG→PNG | Auto-raster substitution → compact SVG→PNG | SVG→PNG + warning |
| `interactive()` | GPU scene | GPU scene (no change) | GPU scene (no change) |

### Configuration

```python
fm.Chart(df).mark_point().encode(x="x", y="y").properties(
    render_config=fm.RenderConfig(
        raster_threshold=500_000,    # int | None; None disables auto-raster
        raster_behavior="warn",      # "warn" | "silent" | "error"
        raster_aggregate="count",    # default aggregate for substitution
        raster_cmap="viridis",       # default colormap for substitution
    )
)
```

`raster_threshold=None` disables auto-raster entirely — the chart renders all marks, even if the SVG is 57MB. `raster_behavior="silent"` keeps auto-raster active but suppresses the warning. `raster_behavior="error"` raises `ValueError` instead of substituting.

## 5. Architecture

The policy is a **Python-side pre-render intercept**, not a Rust-side concern. It operates between `_resolve_pending()` and `render_svg()`/`render_png()`:

```
Chart._render_inputs()
  → _resolve_pending()       # transforms, desugar
  → _apply_auto_raster()     # NEW: count marks, decide, substitute
  → to_spec() + to_arrow()
  → Rust render_svg/render_png
```

The substitution reuses the existing `desugar_raster()` machinery from `marks/heavy_stat.py`. No new Rust code is needed for the core policy — the Raster transform and image renderer already exist.

### Component responsibilities

- **`Chart._apply_auto_raster()`** (new): Counts marks, checks eligibility, performs substitution by cloning the chart with `mark_raster()` applied, returns the (possibly modified) chart.
- **`RenderConfig`** (Python dataclass, new): Holds threshold, behavior, aggregate, cmap. Stored on `Chart._render_config`. Passed through `properties(render_config=...)`.
- **`RenderConfig`** (Rust struct, extended): Add `raster_threshold` and `raster_behavior` fields for forward compatibility, though the policy logic lives in Python.

## 6. Canonical interfaces

```python
@dataclass
class RenderConfig:
    raster_threshold: int | None = 500_000
    raster_behavior: str = "warn"          # "warn" | "silent" | "error"
    raster_aggregate: str = "count"
    raster_cmap: str = "viridis"
    scale: float = 2.0
    embed_fonts: bool = True
    background: str | None = None
    width: float | None = None
    height: float | None = None
```

The `_apply_auto_raster` method signature:

```python
def _apply_auto_raster(self) -> "Chart":
    """Return self (unchanged) or a substituted chart with mark_raster."""
```

Warning message format:

```
"Auto-raster: substituted mark_raster for {original_mark} "
"({mark_count:,} marks > threshold {threshold:,}). "
"Set raster_threshold=None to disable."
```

## 7. Invariants and constraints

- **No silent data loss.** When color encoding is present, auto-raster must NOT fire — it would collapse categorical information into a density map without the user's consent. Emit a guidance warning instead.
- **Idempotent.** Calling `_apply_auto_raster()` on an already-rasterized chart (mark_raster, mark_hex, mark_image) is a no-op.
- **No matplotlib.** The raster path uses the existing Rust `Raster` transform + `tiny-skia`. No new dependencies.
- **Backward compatible.** Default threshold of 500k means all existing charts with <500k marks see zero behavioral change. Charts with >500k marks that previously produced huge SVGs now get auto-raster — this is an intentional behavior improvement, not a regression.
- **Per-chart, not global.** `RenderConfig` on a chart instance overrides any future global config. No module-level mutable state beyond what `ferrum.set_default_theme()` already establishes.

## 8. Key decisions and tradeoffs

**Policy lives in Python, not Rust.** The mark count is known after Python-side transform resolution. Doing the substitution in Python means reusing `desugar_raster()` directly and keeping the Rust renderer stateless. The alternative (Rust-side threshold check) would require duplicating the raster substitution logic in Rust and complicating the PyO3 boundary.

**Default threshold: 500,000.** The spec says 500k. Empirically, 50k marks produces a ~3.5MB SVG (usable but slow in browsers); 500k produces ~35MB (unusable). 500k is the right "something is clearly wrong" threshold. Users who want per-element SVG at any cost set `raster_threshold=None`.

**Color encoding blocks auto-raster.** This is spec-mandated (§3.3). A scatter plot colored by category cannot be rasterized to a density map without losing the category information. The user must explicitly choose `mark_raster(aggregate="count")` or remove the color encoding.

**Line and area marks excluded.** These marks produce one polyline/path per group, not one element per row. A 1M-point line is one `<polyline>` element with 1M vertices — large but not 1M elements. Auto-raster's "substitute mark_raster" doesn't apply to these topologically. Vertex decimation (Ramer-Douglas-Peucker) is a separate future concern.

## 9. Acceptance criteria

1. `fm.Chart(df_1M).mark_point().encode(x="x:Q", y="y:Q").show_svg()` produces a compact SVG (<2MB) with auto-raster substitution and emits a `UserWarning`.
2. Same chart with `.properties(render_config=fm.RenderConfig(raster_threshold=None))` produces the full 57MB SVG with no warning.
3. Same chart with `raster_behavior="error"` raises `ValueError`.
4. Same chart with `raster_behavior="silent"` produces compact SVG with no warning.
5. Chart with color encoding + 1M marks: auto-raster does NOT fire; guidance warning is emitted; SVG contains per-element marks.
6. Chart with <500k marks: no change in behavior.
7. `save("chart.png")` on a 1M-mark chart: auto-raster fires, compact SVG is generated internally, PNG is produced.
8. All existing tests pass unchanged (threshold is above any existing test's mark count).
9. Scale test `test_scatter_1m_svg_size` updated: 1M scatter SVG is now <2MB (was 57MB).

## 10. Validation strategy

- The existing `tests/test_scale_rendering.py::TestMillionRows` tests become the primary validation. Update `test_scatter_1m_svg_size` to assert <2MB (currently asserts <100MB).
- Add tests for each `raster_behavior` mode.
- Add test that color-encoded chart does NOT get auto-raster.
- Add test that `raster_threshold=None` disables auto-raster.
- Add test that mark types excluded from auto-raster (line, area, hex, raster) are unaffected.

- Docstrings on `show_svg()`, `show()`, `save()`, and `RenderConfig` must document the auto-raster behavior and that `raster_threshold=None` disables it for users who want per-element SVG at any cost.

## 11. Open questions

None — the spec already defines the behavior completely. This is implementation of a deferred spec feature, not a new design.
