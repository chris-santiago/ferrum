# Phase 11e — Stat/Mark/Encoding Gap Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every remaining `NotImplementedError`, warn-fallback, and feature gap in the stat, mark, and encoding layers. After 11e, there are zero deferred stat/mark/encoding features.

**Spec:** `docs/superpowers/specs/2026-05-13-interactive-renderer-design.md` §9 (9.1–9.10), §12.5 (testing).

**Dependencies:** 11e can run in parallel with 11d after 11b is done. Some tasks have soft dependencies on 11c (noted per-task):
- 11e1–11e5, 11e7: No dependency on other 11-series sub-phases beyond 11a (SceneGraph IR).
- 11e6 (blend): Needs `svg_walk.rs` from 11a; WASM path deferred to 11b.
- 11e8 (condition): Python wiring + ChartSpec serialization are standalone; runtime resolution in WASM depends on 11c.
- 11e10 (Key): ChartSpec→MarkBatch wiring is standalone; animated transitions depend on 11c.

**Tech Stack:** Rust (transforms, position adjustments, scale, scene graph), Python (encoding channels, mark desugaring, chart API). No new external crate dependencies except `chrono` (already available transitively via `arrow`).

---

## File map

### Modified files (by task)

| Task | File | Change |
|---|---|---|
| 11e1 | `crates/ferrum-core/src/transform/kde.rs` | Add `bw_adjust` field to `KdeSpec` (shared with 11e2) |
| 11e1 | `src/ferrum/marks/statistical.py` | Remove `multiple` NotImplementedError; emit Stack/NormalizeStack/Dodge position adjustments per `multiple` value |
| 11e1 | `crates/ferrum-core/src/render/position.rs` | Add `NormalizeStack` variant to `PositionAdjust` if existing `Stack { offset: Normalize }` does not normalize per-x-slice for continuous KDE output; add Dodge-for-density logic |
| 11e1 | `crates/ferrum-core/src/spec/position.rs` | Add `NormalizeStack` variant if needed (see investigation step) |
| 11e2 | `crates/ferrum-core/src/transform/kde.rs` | Apply `bw_adjust` multiplier after bandwidth rule resolution |
| 11e2 | `src/ferrum/marks/statistical.py` | Remove `bw_adjust` NotImplementedError; pass `bw_adjust` through to `Kde` transform |
| 11e3 | `crates/ferrum-core/src/transform/hex.rs` | Extend `Aggregator` struct and `apply()` to support `min`, `max`, `median`, `std`, `var` |
| 11e4 | `crates/ferrum-core/src/transform/swarm.rs` | Add `dodge` field to `SwarmSpec`; partition-then-offset logic |
| 11e5 | `src/ferrum/chart.py` | Replace mark_function NotImplementedError with multi-layer function evaluation |
| 11e6 | `crates/ferrum-core/src/render/svg_walk.rs` | Emit `<filter>` + `<feComposite>` for `BlendMode::Additive` |
| 11e6 | `crates/ferrum-core/src/render/scene_build.rs` | Read blend from ChartSpec/Layer and propagate to `MarkBatch.blend` |
| 11e6 | `src/ferrum/marks/heavy_stat.py` | Remove `blend="additive"` warn-fallback in `mark_raster` |
| 11e7 | `src/ferrum/encoding/appearance.py` | Add `"legend"` to `_honored_kwargs` for Size, Shape, Opacity |
| 11e7 | `src/ferrum/encoding/base.py` | Ensure `legend` kwarg flows through `to_encoding_spec_dict()` (already does for Color — verify for others) |
| 11e7 | `crates/ferrum-core/src/render/marks/legend.rs` | Respect `legend.disabled` for size/shape/opacity channels (currently only wired for color) |
| 11e8 | `src/ferrum/encoding/appearance.py` | Add `"condition"` to `_honored_kwargs` for all appearance channels |
| 11e8 | `src/ferrum/encoding/base.py` | Serialize `condition` kwarg into `to_encoding_spec_dict()` output |
| 11e8 | `crates/ferrum-core/src/spec/encoding.rs` | Add `condition` field to `EncodingSpec` |
| 11e8 | `crates/ferrum-core/src/render/scene_build.rs` | Propagate conditional encodings into `InteractionConfig.conditionals` |
| 11e9 | `crates/ferrum-core/src/scale/ticks.rs` | Replace approximate `MONTH`/`YEAR` constants with calendar-aware generation |
| 11e9 | `crates/ferrum-core/src/scale/time.rs` | Rewrite `time_ticks()` and `time_nice()` to snap to calendar boundaries |
| 11e10 | `src/ferrum/encoding/text.py` | Ensure `Key` channel values flow into ChartSpec (verify current wiring) |
| 11e10 | `crates/ferrum-core/src/spec/encoding.rs` | Add `key` field to `Encoding` struct if not present |
| 11e10 | `crates/ferrum-core/src/render/scene_build.rs` | Read key field from encoding, populate `MarkBatch.keys` |

### New files

| Task | File | Purpose |
|---|---|---|
| 11e1 | `tests/test_phase_11e_density_multiple.py` | Golden + numeric tests for density stack/fill/dodge |
| 11e2 | `tests/test_phase_11e_bw_adjust.py` | Tests for bw_adjust with string bandwidth rules |
| 11e3 | `tests/test_phase_11e_hex_aggregates.py` | Tests for hex min/max/median/std/var |
| 11e4 | `tests/test_phase_11e_swarm_dodge.py` | Golden test for grouped swarm layout |
| 11e5 | `tests/test_phase_11e_mark_function_layer.py` | Test for mark_function in multi-layer chart |
| 11e6 | `tests/test_phase_11e_blend_additive.py` | Golden test for additive blend SVG filter |
| 11e9 | `tests/test_phase_11e_time_calendar.py` | Test for calendar-aware tick snapping |
| 11e10 | `tests/test_phase_11e_key_channel.py` | Test for Key encoding → MarkBatch.keys wiring |

---

## Task 11e1: mark_density(multiple="stack"|"fill"|"dodge")

**Spec reference:** §9.1

**Context:** Currently `desugar_density()` in `src/ferrum/marks/statistical.py` raises `NotImplementedError` when `multiple != "layer"` (line 141–146). The Rust KDE transform already emits per-group density curves when `groupby` is set. The gap is applying position adjustments to the KDE output.

### Investigation step (must do first)

- [ ] **Step 0: Check existing Stack+Normalize behavior on continuous data**

  The `PositionAdjust::Stack { offset: StackOffset::Normalize }` variant already exists (defined in `crates/ferrum-core/src/spec/position.rs` line 15). Determine whether `apply_stack()` in `position.rs` normalizes per-x-slice (divides each y by the column sum at that x value) when applied to continuous area data, or only works with discrete bar data.

  Read `crates/ferrum-core/src/render/position.rs` — find `apply_stack` and check if it groups by x-value for normalization. If it already handles continuous x (matching KDE's `value` column as the x axis), then `"fill"` just needs `Stack { offset: Normalize }`. If it only groups by discrete ordinal x, a `NormalizeStack` variant or a continuous-x code path is needed.

  **Decision tree:**
  - If existing `Stack { offset: Normalize }` works on continuous x → `"fill"` = Stack with Normalize offset, no new enum variant.
  - If it does not → add a continuous normalization path to `apply_stack` that bins or aligns by x-value (the KDE grid points are identical across groups, so exact equality on x works).

### Steps

- [ ] **Step 1: Modify KdeSpec to share extent across groups**

  In `crates/ferrum-core/src/transform/kde.rs`, the `apply_grouped()` function already calls `apply_one_group()` per group. For stacking to work correctly, all groups must share the same x-grid. Currently each group independently computes its extent from its own data.

  Add a pre-pass in `apply_grouped()` that computes a global extent across all groups, then passes that extent to each `apply_one_group()` call. This ensures every group's output has identical `value` column values, which is required for per-x-slice stacking.

  ```rust
  // In apply_grouped(), before the per-group loop:
  // Compute global extent if spec.extent is None
  let global_extent = match spec.extent {
      Some(ext) => ext,
      None => {
          let mut lo = f64::INFINITY;
          let mut hi = f64::NEG_INFINITY;
          for ixs in group_idx_map.values() {
              for &i in ixs {
                  if arr.is_null(i) { continue; }
                  let v = arr.value(i);
                  if v.is_nan() { continue; }
                  lo = lo.min(v);
                  hi = hi.max(v);
              }
          }
          (lo, hi)
      }
  };
  // Create a modified spec with the global extent for each group call
  let group_spec = KdeSpec { extent: Some(global_extent), ..spec.clone() };
  ```

- [ ] **Step 1b: Trace the tuple consumer chain**

  Before modifying the return shape of `desugar_density`, read the full call chain to understand what tuple shapes are consumed:

  1. Read `_set_composite_mark()` in `src/ferrum/chart.py` — this stores the desugar function.
  2. Read `_resolve_pending()` in `src/ferrum/chart.py` — this calls the stored desugar function and unpacks its return value.
  3. Identify exactly where the return tuple is destructured and what shapes are already supported (3-tuple, 5-tuple `__layered__`, etc.).
  4. Determine the minimal change to thread a position adjustment through.

  **Do this investigation before writing any code.** The exact approach (extending the tuple, emitting a `__layered__` 5-tuple with a positioned layer, or another mechanism) depends on what the consumer chain already supports.

- [ ] **Step 2: Update desugar_density for "stack" mode**

  In `src/ferrum/marks/statistical.py`, replace the `NotImplementedError` block (lines 141–146) with mode-specific logic. The exact return shape depends on Step 1b's findings. The conceptual change:

  ```python
  if multiple == "stack":
      # Ensure groupby is set (stacking requires multiple groups).
      # If color encoding is present, groupby should already be wired
      # by the _resolve_density adapter.
      transforms = [Kde(field, **kde_kwargs)]
      mark = "area" if fill else "line"
      # Attach a Stack position adjustment to cumulate densities.
      from ferrum.position import Stack
      position_adj = Stack(offset="zero")
      # Return shape must match what _resolve_pending/_set_composite_mark expects.
      # See Step 1b for the exact tuple layout.
      ...
  ```

  The position adjustment needs to reach the rendered layer. Thread it through whatever mechanism Step 1b identifies — whether that's extending the 3-tuple to a 4-tuple, using the `__layered__` 5-tuple pattern with a single `_Layer`, or storing it as a side-channel on the desugar result.

- [ ] **Step 3: Update desugar_density for "fill" mode**

  Same as "stack" but with `Stack(offset="normalize")`:

  ```python
  elif multiple == "fill":
      transforms = [Kde(field, **kde_kwargs)]
      mark = "area" if fill else "line"
      from ferrum.position import Stack
      position_adj = Stack(offset="normalize")
      return (mark, transforms, encoding_remap, None, None, position_adj)
  ```

  If Step 0 determined that `Stack { offset: Normalize }` does not handle continuous x correctly, implement the continuous normalization path in `position.rs` first (see Step 3a below).

- [ ] **Step 3a (conditional): Add continuous-x normalization to apply_stack**

  Only if Step 0 reveals the existing `Normalize` path doesn't work for continuous x:

  In `crates/ferrum-core/src/render/position.rs`, within the `Normalize` offset handling of `apply_stack()`, add a code path for when the x column is `Float64` (continuous). Group rows by exact x value (safe because KDE grid points are identical across groups), compute the sum of y values at each x, and divide each y by that sum.

  ```rust
  // Pseudocode within apply_stack's Normalize branch:
  // if x_dtype == Float64:
  //     for each unique x_value:
  //         sum_y = sum of y values across all groups at this x
  //         for each row at this x: y_out[row] = y_cumulative[row] / sum_y
  ```

- [ ] **Step 4: Update desugar_density for "dodge" mode**

  "Dodge" for density is conceptually different from position-adjustment dodge. It subdivides the y-axis range by group count: each group's density is scaled to fit a fraction of the total y extent, and offset vertically. This is NOT a general position adjustment — it is y-axis rescaling specific to KDE output.

  **Design decision (resolve before coding):** Two approaches:

  **Approach A (preferred — Python-side desugar, no new Rust enum variant):** Handle dodge in `desugar_density` by emitting per-group layers with pre-scaled y values. This is consistent with the "composite marks desugar Python-side" architectural decision. The desugar function:
  1. Runs the KDE transform with `groupby` to get per-group densities.
  2. Determines the number of distinct groups from the data or the color encoding.
  3. Emits a `__layered__` result with one `_Layer` per group, where each layer's density column is scaled by `1.0 / n_groups` and offset by `group_index / n_groups * max_density`.

  The challenge with Approach A is that the data is not available at desugar time (desugar happens before rendering). So instead: emit a single KDE transform with a `normalize_mode="dodge"` parameter, and let the Rust KDE transform handle the per-group scaling.

  **Approach B (Rust-side normalize mode on KDE):** Add a `normalize_mode: Option<String>` field to `KdeSpec`. When `"dodge"`, the Rust `apply_grouped()` function post-processes the combined output: divides the density axis by `n_groups` and offsets each group's density to its own band. This keeps the PositionAdjust enum clean and confines the density-specific logic to the KDE transform.

  Choose Approach B — it is cleaner (no new PositionAdjust variant, no leaking density-specific logic into the general position system). In `desugar_density()`:

  ```python
  elif multiple == "dodge":
      kde_kwargs["normalize_mode"] = "dodge"
      transforms = [Kde(field, **kde_kwargs)]
      mark = "area" if fill else "line"
      return (mark, transforms, encoding_remap)
  ```

  **Rust-side:** Add to `KdeSpec`:

  ```rust
  #[serde(skip_serializing_if = "Option::is_none", default)]
  pub normalize_mode: Option<String>,  // "dodge" | None
  ```

  In `apply_grouped()`, after stacking all group results, if `normalize_mode == Some("dodge")`:
  - Count distinct groups `n_groups`.
  - For each group at index `i`: `density[row] = density[row] / n_groups + (i as f64 / n_groups as f64) * global_max_density`.
  - This produces side-by-side (vertically staggered) density ridges.

  **Python-side:** Ensure the `Kde` transform constructor accepts and serializes `normalize_mode`.

- [ ] **Step 5: Write tests**

  Create `tests/test_phase_11e_density_multiple.py`:

  ```python
  import polars as pl
  import ferrum as fm

  df = pl.DataFrame({
      "value": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
      "group": ["A", "A", "A", "B", "B", "B"],
  })

  def test_density_stack_renders():
      c = fm.Chart(df).mark_density(multiple="stack").encode(x="value", color="group")
      svg = c.to_svg()
      assert "<svg" in svg
      # Stacked density should have area elements
      assert "<path" in svg or "d=" in svg

  def test_density_fill_renders():
      c = fm.Chart(df).mark_density(multiple="fill").encode(x="value", color="group")
      svg = c.to_svg()
      assert "<svg" in svg

  def test_density_dodge_renders():
      c = fm.Chart(df).mark_density(multiple="dodge").encode(x="value", color="group")
      svg = c.to_svg()
      assert "<svg" in svg

  def test_density_layer_still_works():
      """Regression: default multiple='layer' must not break."""
      c = fm.Chart(df).mark_density(multiple="layer").encode(x="value", color="group")
      svg = c.to_svg()
      assert "<svg" in svg
  ```

  Generate golden SVGs for stack/fill/dodge, rasterize with `snapshot-goldens.py`, visually inspect PNGs.

- [ ] **Step 6: Verify**

  ```bash
  uv run pytest tests/test_phase_11e_density_multiple.py -v
  DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test
  ```

- [ ] **Step 7: Commit**

  ```
  feat(density): add multiple="stack"|"fill"|"dodge" to mark_density
  ```

---

## Task 11e2: mark_density(bw_adjust=) with string bandwidth rules

**Spec reference:** §9.2

**Context:** Currently `desugar_density()` raises `NotImplementedError` when `bw_adjust != 1.0` and bandwidth is a string rule like `"scott"` (lines 148–157 in `statistical.py`). The fix is two-sided: (1) add `bw_adjust` to the Rust `KdeSpec` so the rule is resolved first then multiplied, and (2) remove the Python-side error.

### Steps

- [ ] **Step 1: Add bw_adjust to KdeSpec**

  In `crates/ferrum-core/src/transform/kde.rs`, add to `KdeSpec`:

  ```rust
  #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
  pub(crate) struct KdeSpec {
      pub field: String,
      pub bandwidth: BandwidthSpec,
      pub n: usize,
      #[serde(skip_serializing_if = "Option::is_none", default)]
      pub extent: Option<(f64, f64)>,
      #[serde(default)]
      pub cumulative: bool,
      #[serde(skip_serializing_if = "Option::is_none", default)]
      pub groupby: Option<String>,
      #[serde(skip_serializing_if = "Option::is_none", default)]
      pub name: Option<String>,
      /// Bandwidth multiplier applied AFTER rule resolution. Default 1.0.
      #[serde(default = "default_bw_adjust")]
      pub bw_adjust: f64,
  }

  fn default_bw_adjust() -> f64 { 1.0 }
  ```

- [ ] **Step 2: Apply bw_adjust in bandwidth computation**

  In `apply_one_group()`, after the `bandwidth()` call (around line 97–98), multiply by `bw_adjust`:

  ```rust
  let h = bandwidth(&clean, &spec.bandwidth)?;
  let h = h * spec.bw_adjust;  // <-- NEW
  ```

  This is the "one-line Rust change" the spec describes. The `bw_adjust` multiplier applies regardless of whether the bandwidth was a fixed value, `scott`, or `silverman`.

- [ ] **Step 3: Update Python Kde transform constructor**

  In the Python `Kde` class (find it via `grep -rn 'class Kde' src/ferrum/`), add `bw_adjust` as a parameter that flows through to the JSON spec:

  ```python
  class Kde:
      def __init__(self, field, *, bandwidth="scott", n=512, extent=None,
                   cumulative=False, groupby=None, bw_adjust=1.0):
          ...
          self.bw_adjust = bw_adjust

      def to_spec_dict(self):
          d = { ... existing fields ... }
          if self.bw_adjust != 1.0:
              d["bw_adjust"] = self.bw_adjust
          return d
  ```

- [ ] **Step 4: Remove Python NotImplementedError**

  In `src/ferrum/marks/statistical.py`, replace lines 147–157:

  ```python
  # OLD:
  # if bw_adjust != 1.0:
  #     if isinstance(bandwidth, (int, float)):
  #         bandwidth = float(bandwidth) * float(bw_adjust)
  #     else:
  #         raise NotImplementedError(...)

  # NEW: always pass bw_adjust through to the Kde transform.
  # Rust resolves the bandwidth rule first, then multiplies by bw_adjust.
  ```

  **Always pass `bw_adjust` to Rust.** Remove the Python-side multiplication entirely. The Rust `bandwidth()` function already handles `BandwidthSpec::Fixed { value }` — it returns the fixed value, then `h *= bw_adjust` applies the multiplier. This is simpler than maintaining two code paths.

  Remove lines 147–157 entirely (both the numeric-path multiplication and the string-rule `NotImplementedError`). Update the `kde_kwargs` dict to include `bw_adjust`:

  ```python
  kde_kwargs: dict = dict(
      bandwidth=bandwidth,
      n=n,
      extent=extent,
      cumulative=cumulative,
      bw_adjust=bw_adjust,  # Rust handles rule × adjust for ALL paths
  )
  ```

  The numeric bandwidth is no longer pre-multiplied Python-side — Rust sees the original bandwidth value and the separate `bw_adjust` multiplier, and applies the multiplication after rule resolution. This produces identical results for all bandwidth types.

- [ ] **Step 5: Write tests**

  Create `tests/test_phase_11e_bw_adjust.py`:

  ```python
  import polars as pl
  import ferrum as fm

  df = pl.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})

  def test_bw_adjust_with_scott():
      """bw_adjust with 'scott' rule should not raise."""
      c = fm.Chart(df).mark_density(bandwidth="scott", bw_adjust=0.5).encode(x="val")
      svg = c.to_svg()
      assert "<svg" in svg

  def test_bw_adjust_with_silverman():
      c = fm.Chart(df).mark_density(bandwidth="silverman", bw_adjust=2.0).encode(x="val")
      svg = c.to_svg()
      assert "<svg" in svg

  def test_bw_adjust_with_numeric():
      c = fm.Chart(df).mark_density(bandwidth=1.0, bw_adjust=0.3).encode(x="val")
      svg = c.to_svg()
      assert "<svg" in svg

  def test_bw_adjust_default_no_change():
      """bw_adjust=1.0 (default) must produce identical output."""
      c1 = fm.Chart(df).mark_density(bandwidth="scott").encode(x="val")
      c2 = fm.Chart(df).mark_density(bandwidth="scott", bw_adjust=1.0).encode(x="val")
      assert c1.to_svg() == c2.to_svg()
  ```

- [ ] **Step 6: Verify**

  ```bash
  uv run pytest tests/test_phase_11e_bw_adjust.py -v
  DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test
  ```

- [ ] **Step 7: Commit**

  ```
  feat(kde): wire bw_adjust through to Rust for string bandwidth rules
  ```

---

## Task 11e3: mark_hex full aggregates (min, max, median, std, var)

**Spec reference:** §9.3

**Context:** The `Aggregator` struct in `crates/ferrum-core/src/transform/hex.rs` currently tracks only `count: u64` and `sum: f64`, supporting `"count"`, `"mean"`, and `"sum"` aggregates. Adding `min`, `max`, `median`, `std`, and `var` requires structural changes — `median` in particular needs all per-bin values collected, not just running accumulators.

### Steps

- [ ] **Step 1: Extend the Aggregator struct**

  In `crates/ferrum-core/src/transform/hex.rs`, replace the `Aggregator` struct:

  ```rust
  #[derive(Default)]
  struct Aggregator {
      count: u64,
      sum: f64,
      min: f64,
      max: f64,
      sum_sq: f64,       // for std/var: sum of (x - mean)^2 via Welford or two-pass
      values: Vec<f64>,  // for median: collect all values per bin
  }

  impl Aggregator {
      fn new() -> Self {
          Aggregator {
              count: 0,
              sum: 0.0,
              min: f64::INFINITY,
              max: f64::NEG_INFINITY,
              sum_sq: 0.0,
              values: Vec::new(),
          }
      }

      fn push(&mut self, v: f64) {
          self.count += 1;
          self.sum += v;
          self.min = self.min.min(v);
          self.max = self.max.max(v);
          self.values.push(v);
      }

      fn finalize(&mut self, agg: &str) -> f64 {
          match agg {
              "count" => self.count as f64,
              "sum" => self.sum,
              "mean" => if self.count == 0 { f64::NAN } else { self.sum / self.count as f64 },
              "min" => if self.count == 0 { f64::NAN } else { self.min },
              "max" => if self.count == 0 { f64::NAN } else { self.max },
              "median" => {
                  if self.values.is_empty() { return f64::NAN; }
                  self.values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                  let n = self.values.len();
                  if n % 2 == 0 {
                      (self.values[n / 2 - 1] + self.values[n / 2]) / 2.0
                  } else {
                      self.values[n / 2]
                  }
              }
              "std" => {
                  if self.count < 2 { return f64::NAN; }
                  let mean = self.sum / self.count as f64;
                  let var = self.values.iter()
                      .map(|v| (v - mean).powi(2))
                      .sum::<f64>() / (self.count as f64 - 1.0);
                  var.sqrt()
              }
              "var" => {
                  if self.count < 2 { return f64::NAN; }
                  let mean = self.sum / self.count as f64;
                  self.values.iter()
                      .map(|v| (v - mean).powi(2))
                      .sum::<f64>() / (self.count as f64 - 1.0)
              }
              _ => f64::NAN,
          }
      }
  }
  ```

  **Memory note:** `values: Vec<f64>` is only needed for `median`, `std`, and `var`. For `count`, `mean`, `sum`, `min`, `max` the running accumulators suffice. An optimization (not required for correctness) is to only push to `values` when the aggregate is one of `median`, `std`, `var`. This can be done by passing the aggregate name to `push()` or by checking at the call site. Implement the optimization to avoid unnecessary memory allocation for the common `count` aggregate:

  ```rust
  fn push(&mut self, v: f64, needs_values: bool) {
      self.count += 1;
      self.sum += v;
      self.min = self.min.min(v);
      self.max = self.max.max(v);
      if needs_values {
          self.values.push(v);
      }
  }
  ```

  Compute `needs_values` once from the aggregate string:

  ```rust
  let needs_values = matches!(agg, "median" | "std" | "var");
  ```

- [ ] **Step 2: Update aggregate validation**

  In `apply()`, update the validation check (around line 104–109):

  ```rust
  // OLD:
  if !matches!(agg, "count" | "mean" | "sum") {
      return Err(PyValueError::new_err(format!(
          "stat_hex: unknown aggregate '{}'; expected 'count' | 'mean' | 'sum'",
          agg
      )));
  }

  // NEW:
  if !matches!(agg, "count" | "mean" | "sum" | "min" | "max" | "median" | "std" | "var") {
      return Err(PyValueError::new_err(format!(
          "stat_hex: unknown aggregate '{}'; expected one of: count, mean, sum, min, max, median, std, var",
          agg
      )));
  }
  ```

  Also update `needs_field`: aggregates `min`, `max`, `median`, `std`, `var` all require a `field`:

  ```rust
  let needs_field = matches!(agg, "mean" | "sum" | "min" | "max" | "median" | "std" | "var");
  ```

- [ ] **Step 3: Update the accumulation loop**

  Find the loop in `apply()` that accumulates values into `Aggregator` entries in the `BTreeMap`. Currently it does:

  ```rust
  agg_entry.count += 1;
  agg_entry.sum += fv;
  ```

  Replace with:

  ```rust
  agg_entry.push(fv, needs_values);
  ```

- [ ] **Step 4: Update the finalization loop**

  Where the aggregated value is computed per hex, replace the existing match on aggregate with a call to `finalize()`:

  ```rust
  // OLD (approximate):
  let value = match agg {
      "count" => entry.count as f64,
      "mean" => entry.sum / entry.count as f64,
      "sum" => entry.sum,
      _ => unreachable!(),
  };

  // NEW:
  let value = entry.finalize(agg);
  ```

- [ ] **Step 5: Update the default_aggregate comment**

  The `default_aggregate()` function (line 30–32) returns `"count"`. The comment on `HexSpec.aggregate` (line 41) says `"count" | "mean" | "sum"` — update it:

  ```rust
  pub aggregate: String, // "count" | "mean" | "sum" | "min" | "max" | "median" | "std" | "var"
  ```

- [ ] **Step 6: Write tests**

  Create `tests/test_phase_11e_hex_aggregates.py`:

  ```python
  import polars as pl
  import ferrum as fm
  import math

  df = pl.DataFrame({
      "x": [1.0, 1.1, 1.2, 5.0, 5.1, 5.2],
      "y": [1.0, 1.1, 1.2, 5.0, 5.1, 5.2],
      "val": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
  })

  def test_hex_min():
      c = fm.Chart(df).mark_hex(aggregate="min", field="val").encode(x="x", y="y")
      svg = c.to_svg()
      assert "<svg" in svg

  def test_hex_max():
      c = fm.Chart(df).mark_hex(aggregate="max", field="val").encode(x="x", y="y")
      svg = c.to_svg()
      assert "<svg" in svg

  def test_hex_median():
      c = fm.Chart(df).mark_hex(aggregate="median", field="val").encode(x="x", y="y")
      svg = c.to_svg()
      assert "<svg" in svg

  def test_hex_std():
      c = fm.Chart(df).mark_hex(aggregate="std", field="val").encode(x="x", y="y")
      svg = c.to_svg()
      assert "<svg" in svg

  def test_hex_var():
      c = fm.Chart(df).mark_hex(aggregate="var", field="val").encode(x="x", y="y")
      svg = c.to_svg()
      assert "<svg" in svg

  def test_hex_count_unchanged():
      """Regression: default count aggregate must still work."""
      c = fm.Chart(df).mark_hex().encode(x="x", y="y")
      svg = c.to_svg()
      assert "<svg" in svg
  ```

  Also add Rust unit tests in `hex.rs`:

  ```rust
  #[test]
  fn test_aggregator_median_odd() {
      let mut a = Aggregator::new();
      for v in [3.0, 1.0, 2.0] { a.push(v, true); }
      assert_eq!(a.finalize("median"), 2.0);
  }

  #[test]
  fn test_aggregator_median_even() {
      let mut a = Aggregator::new();
      for v in [1.0, 2.0, 3.0, 4.0] { a.push(v, true); }
      assert_eq!(a.finalize("median"), 2.5);
  }

  #[test]
  fn test_aggregator_std() {
      let mut a = Aggregator::new();
      for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] { a.push(v, true); }
      let std = a.finalize("std");
      assert!((std - 2.138089935299395).abs() < 1e-10);
  }
  ```

- [ ] **Step 7: Verify**

  ```bash
  uv run pytest tests/test_phase_11e_hex_aggregates.py -v
  DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core
  ```

- [ ] **Step 8: Commit**

  ```
  feat(hex): add min, max, median, std, var aggregates to hex transform
  ```

---

## Task 11e4: mark_swarm(dodge=...) grouped beeswarm

**Spec reference:** §9.4

**Context:** The Rust swarm transform (`crates/ferrum-core/src/transform/swarm.rs`) already partitions data by `category` and computes beeswarm positions within each category. The `dodge` parameter adds a second level of grouping: within each category, data is further split by a dodge field, swarm positions are computed per sub-group, and sub-groups are offset from each other along the cross axis.

### Steps

- [ ] **Step 1: Add dodge field to SwarmSpec**

  In `crates/ferrum-core/src/transform/swarm.rs`, add to `SwarmSpec`:

  ```rust
  #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
  pub(crate) struct SwarmSpec {
      pub category: String,
      pub value: String,
      #[serde(default = "default_point_size")]
      pub point_size: f64,
      #[serde(default = "default_spacing")]
      pub spacing: f64,
      #[serde(default)]
      pub side: SwarmSide,
      #[serde(default)]
      pub orient: SwarmOrient,
      #[serde(skip_serializing_if = "Option::is_none", default)]
      pub name: Option<String>,
      /// When set, partition data within each category by this field and
      /// offset sub-groups along the cross axis (dodge).
      #[serde(skip_serializing_if = "Option::is_none", default)]
      pub dodge: Option<String>,
  }
  ```

- [ ] **Step 2: Implement dodged swarm logic**

  In `apply_with_context()`, after the existing category-partitioning logic, add a dodge branch. The algorithm:

  1. If `dodge` is `None`, run the existing single-group swarm as before.
  2. If `dodge` is `Some(field)`, for each category:
     a. Further partition points by the dodge field.
     b. Compute swarm positions for each sub-group independently.
     c. Apply a cross-axis offset to each sub-group: offset = `(sub_group_index - (n_sub_groups - 1) / 2.0) * dodge_width`.
     d. `dodge_width` = `bandwidth / n_sub_groups` where bandwidth is the ordinal band width for the category (from `TransformContext`), falling back to `2 * (point_size + spacing)` per sub-group.

  ```rust
  pub(crate) fn apply_with_context(
      spec: &SwarmSpec,
      batch: &RecordBatch,
      ctx: &TransformContext,
  ) -> PyResult<RecordBatch> {
      if let Some(dodge_field) = &spec.dodge {
          return apply_dodged(spec, batch, ctx, dodge_field);
      }
      // ... existing non-dodge logic ...
  }

  fn apply_dodged(
      spec: &SwarmSpec,
      batch: &RecordBatch,
      ctx: &TransformContext,
      dodge_field: &str,
  ) -> PyResult<RecordBatch> {
      // 1. Validate dodge column exists (Utf8 or Float64).
      // 2. Group rows by (category, dodge_value).
      // 3. For each category:
      //    a. Determine distinct dodge groups within this category.
      //    b. Swarm each dodge group independently.
      //    c. Offset each dodge group's cross-axis positions.
      // 4. Assemble output batch with dodge column preserved.
      ...
  }
  ```

- [ ] **Step 3: Wire dodge through Python mark_swarm**

  Find the `mark_swarm` method in `src/ferrum/chart.py` (or the desugar function). Add `dodge` as a keyword argument that flows to the Rust `SwarmSpec`:

  ```python
  def mark_swarm(self, *, dodge=None, **kwargs) -> "Chart":
      ...
      if dodge is not None:
          kwargs["dodge"] = dodge
      ...
  ```

  Ensure the Python `Swarm` transform class (find via `grep -rn 'class Swarm' src/ferrum/`) accepts and serializes `dodge`.

- [ ] **Step 4: Write tests**

  Create `tests/test_phase_11e_swarm_dodge.py`:

  ```python
  import polars as pl
  import ferrum as fm

  df = pl.DataFrame({
      "species": ["setosa"] * 6 + ["versicolor"] * 6,
      "sex": ["M", "F"] * 6,
      "petal_length": [1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 4.7, 4.5, 4.9, 4.0, 4.6, 4.5],
  })

  def test_swarm_dodge_renders():
      c = (fm.Chart(df)
           .mark_swarm(dodge="sex")
           .encode(x="species", y="petal_length", color="sex"))
      svg = c.to_svg()
      assert "<svg" in svg
      # Should have distinct mark positions for M and F within each species

  def test_swarm_no_dodge_still_works():
      """Regression: mark_swarm without dodge."""
      c = fm.Chart(df).mark_swarm().encode(x="species", y="petal_length")
      svg = c.to_svg()
      assert "<svg" in svg
  ```

  Generate golden SVG, rasterize, visually inspect.

- [ ] **Step 5: Verify**

  ```bash
  uv run pytest tests/test_phase_11e_swarm_dodge.py -v
  DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core
  ```

- [ ] **Step 6: Commit**

  ```
  feat(swarm): add dodge parameter for grouped beeswarm layout
  ```

---

## Task 11e5: mark_function multi-layer support

**Spec reference:** §9.5

**Context:** `Chart.mark_function()` in `src/ferrum/chart.py` (line 1815–1818) raises `NotImplementedError` when the chart already has layers. The fix evaluates the function callable Python-side, injects the result as a named data source, and adds a line layer using the existing `Layer.data_source` mechanism.

### Design note

The spec says: "during `_render_inputs()`, detect `mark_function` layers, evaluate their Python callables, inject the generated data as named data sources using the existing `Layer.data_source` / `TransformSpec.name` mechanism." This means `mark_function()` should NOT evaluate eagerly at call time — it should store the callable and parameters, and evaluation happens later in `_render_inputs()` when domain information from co-layers is available. Eager evaluation would break `domain=None` inference when function layers are composed with data layers that haven't been added yet (e.g. `scatter + function_chart`).

### Steps

- [ ] **Step 1: Read the existing layer and _render_inputs architecture**

  Before coding, read:
  1. `src/ferrum/_layer.py` — understand the `_Layer` class fields and how layers carry data/transforms.
  2. `src/ferrum/chart.py` — find `_render_inputs()` and understand how it processes layers, resolves data, and builds the ChartSpec.
  3. Identify how `Layer.data_source` and `TransformSpec.name` are used to attach named data to specific layers (the Phase 8a mechanism for composite marks).

- [ ] **Step 2: Remove the NotImplementedError, store callable on the layer**

  In `src/ferrum/chart.py`, replace lines 1815–1819:

  ```python
  # OLD:
  if self._layers is not None and self._layers:
      raise NotImplementedError(
          "mark_function as a layer in a multi-layer Chart is deferred to Phase 9+; "
          "use a separate Chart composed via + instead"
      )

  # NEW: Allow mark_function in multi-layer charts.
  # Store the callable and parameters; evaluation deferred to _render_inputs().
  ```

  The `mark_function()` method should store `fn`, `domain`, `n`, `clip` on the resulting layer as metadata (e.g., `_function_spec` dict). The layer's mark is `"line"`, its encoding remap is `{"x": "x", "y": "y"}`, and it carries no data yet — the data will be injected during `_render_inputs()`.

- [ ] **Step 3: Evaluate function callables in _render_inputs()**

  In `_render_inputs()` (or whatever method assembles the final ChartSpec + data), add a step that detects function layers and evaluates them:

  ```python
  # In _render_inputs(), before data is shipped to Rust:
  for layer in layers:
      if hasattr(layer, '_function_spec') and layer._function_spec is not None:
          fspec = layer._function_spec
          fn = fspec["fn"]
          domain = fspec.get("domain")
          n = fspec.get("n", 200)

          # Infer domain from co-layers' x data if not explicitly provided.
          if domain is None:
              domain = self._infer_x_extent_from_layers(layers)

          # Evaluate the function.
          import numpy as np
          xs = np.linspace(domain[0], domain[1], n)
          ys = fn(xs)

          # Inject as a named data source on the layer.
          import pyarrow as pa
          synthetic = pa.table({"x": xs, "y": ys})
          layer.data = synthetic
          layer.data_source = f"__function_{id(layer)}"
  ```

  The `_infer_x_extent_from_layers()` helper walks all data-bearing layers, finds their x columns, and returns `(min, max)`. This enables `domain=None` to work when the function layer is composed with a scatter layer.

- [ ] **Step 4: Ensure standalone mark_function still works**

  The standalone path (no existing layers) still uses the existing `desugar_function()` in `heavy_stat.py` and evaluates eagerly — that code path is unchanged. Only the multi-layer path defers evaluation.

- [ ] **Step 4: Write tests**

  Create `tests/test_phase_11e_mark_function_layer.py`:

  ```python
  import numpy as np
  import polars as pl
  import ferrum as fm

  df = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [0.1, 0.9, 2.1, 2.9]})

  def test_function_overlay_on_scatter():
      """mark_function as layer on top of scatter should not raise."""
      c = (fm.Chart(df)
           .mark_point()
           .encode(x="x", y="y")
           + fm.Chart(df).mark_function(np.sin, domain=[0, 3]))
      svg = c.to_svg()
      assert "<svg" in svg
      # Should contain both point marks and a line path

  def test_function_with_explicit_domain():
      c = (fm.Chart(df).mark_point().encode(x="x", y="y")
           + fm.Chart(None).mark_function(lambda x: x**2, domain=[0, 3]))
      svg = c.to_svg()
      assert "<svg" in svg

  def test_function_standalone_still_works():
      """Regression: standalone mark_function."""
      c = fm.Chart(None).mark_function(np.sin, domain=[0, 6.28])
      svg = c.to_svg()
      assert "<svg" in svg
  ```

- [ ] **Step 5: Verify**

  ```bash
  uv run pytest tests/test_phase_11e_mark_function_layer.py -v
  ```

- [ ] **Step 6: Commit**

  ```
  feat(function): support mark_function as a layer in multi-layer charts
  ```

---

## Task 11e6: blend="additive" (SVG filter + GPU blend state note)

**Spec reference:** §9.6

**Context:** `BlendMode::Additive` already exists in the `ferrum-scene` types (`crates/ferrum-scene/src/types.rs` line 73). The scene builder (`scene_build.rs` line 170) always emits `BlendMode::Normal`. Two things need to happen: (1) propagate `blend` from ChartSpec/Layer to `MarkBatch.blend`, and (2) make the SVG walker emit the correct SVG filter for additive blending.

### Steps

- [ ] **Step 1: Propagate blend from spec to MarkBatch**

  In `crates/ferrum-core/src/render/scene_build.rs`, where `MarkBatch` is constructed (around line 170), read the blend mode from the layer/spec:

  ```rust
  // Determine blend mode from the layer or spec.
  let blend = match layer_or_spec_blend_str {
      Some("additive") => BlendMode::Additive,
      _ => BlendMode::Normal,
  };
  ```

  The exact source of the blend string depends on how the Python side serializes it. Currently `chart.py` has `"blend": blend` in the mark layer kwargs (line 1613). Find where this ends up in the ChartSpec JSON and read it in scene_build.

  Check if `blend` is already on `ChartSpec`, `Layer`, or the mark spec. If not, add it:

  In `crates/ferrum-core/src/spec/chart.rs` (or wherever `Layer` is defined), add:

  ```rust
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub blend: Option<String>,
  ```

- [ ] **Step 2: Emit SVG filter for additive blending**

  In `crates/ferrum-core/src/render/svg_walk.rs`, when walking a `MarkBatch` with `BlendMode::Additive`:

  1. Emit a `<filter>` definition in the `<defs>` section:

  ```xml
  <filter id="blend-additive-{batch_index}">
    <feComposite in="SourceGraphic" in2="BackgroundImage"
                 operator="arithmetic" k1="0" k2="1" k3="1" k4="0"/>
  </filter>
  ```

  2. Wrap the batch's `<g>` element with `filter="url(#blend-additive-{batch_index})"`.

  **Do NOT use `mix-blend-mode: screen` as an alternative.** Screen blend is `1-(1-a)(1-b)` which saturates; additive is `a+b` which can exceed 1.0. For density overlap visualization these produce visibly different results. The `feComposite arithmetic` approach is the correct implementation of additive blending.

  **resvg compatibility:** `resvg` supports SVG filters including `feComposite`. Verify this works by rendering a test SVG with the additive filter through the existing `rasterize_svg()` helper and inspecting the PNG. If `enable-background="new"` is needed for the `in2="BackgroundImage"` input, add it to the `<svg>` root element. As a fallback, use `feBlend mode="screen"` (built-in SVG filter, no `enable-background` required) with a comment noting it is an approximation.

- [ ] **Step 3: Remove Python warn-fallback**

  In `src/ferrum/marks/heavy_stat.py`, remove lines 456–462 (the `warn_once` for `blend="additive"`):

  ```python
  # OLD:
  if blend == "additive":
      from ferrum._warn import warn_once
      warn_once(
          "blend_additive",
          "mark_raster blend='additive' deferred to Phase 11; using alpha blending",
      )

  # NEW: blend="additive" is now supported; pass through to the renderer.
  ```

- [ ] **Step 4: Write tests**

  Create `tests/test_phase_11e_blend_additive.py`:

  ```python
  import polars as pl
  import ferrum as fm

  df = pl.DataFrame({
      "x": [1.0, 2.0, 3.0, 1.5, 2.5],
      "y": [1.0, 2.0, 3.0, 1.5, 2.5],
  })

  def test_blend_additive_svg_contains_blend():
      c = fm.Chart(df).mark_raster(blend="additive").encode(x="x", y="y")
      svg = c.to_svg()
      assert "<svg" in svg
      # Should contain SVG filter for additive blending
      assert "feComposite" in svg or "feBlend" in svg

  def test_blend_alpha_default():
      """Regression: default alpha blend."""
      c = fm.Chart(df).mark_raster().encode(x="x", y="y")
      svg = c.to_svg()
      assert "<svg" in svg
      # Should NOT contain additive blend artifacts
  ```

  **Note for WASM path:** The GPU blend state (`wgpu::BlendState`) is wired in 11b/11c. Add a `// TODO(11b): wire BlendMode::Additive to wgpu::BlendState` comment in the WASM renderer code when it exists.

- [ ] **Step 5: Verify**

  ```bash
  uv run pytest tests/test_phase_11e_blend_additive.py -v
  ```

- [ ] **Step 6: Commit**

  ```
  feat(blend): implement additive blend mode in SVG backend
  ```

---

## Task 11e7: legend kwarg on Size, Shape, Opacity

**Spec reference:** §9.7

**Context:** The `Color` channel already honors `legend` (it's in `_honored_kwargs` and `to_encoding_spec_dict()` serializes `legend=None`/`False` as `{"disabled": true}`). Size, Shape, and Opacity accept `legend` without error (it goes through `ChannelBase.__init__`) but list it as "reserved for future use" and trigger a `warn_once`. The fix is to add `"legend"` to their `_honored_kwargs` and ensure the Rust renderer respects the `legend.disabled` flag for these channels.

### Steps

- [ ] **Step 1: Add "legend" to _honored_kwargs**

  In `src/ferrum/encoding/appearance.py`:

  ```python
  # Size (line 88):
  _honored_kwargs = frozenset(["type", "scale", "title", "legend"])

  # Shape (line 122):
  _honored_kwargs = frozenset(["type", "scale", "title", "legend"])

  # Opacity (line 155):
  _honored_kwargs = frozenset(["type", "scale", "title", "legend"])
  ```

  This stops the `warn_once` for these channels when `legend` is passed.

- [ ] **Step 2: Update docstrings**

  Update the `Notes` section of `Size`, `Shape`, `Opacity` to document `legend` as honored (remove the "reserved for future use" language):

  ```python
  class Size(ChannelBase):
      """...
      Notes
      -----
      ``legend`` is honored: passing ``legend=None`` or ``legend=False``
      suppresses the size legend in the rendered SVG.  ``condition`` is
      accepted but reserved for future use.
      ...
      """
  ```

  Same for `Shape` and `Opacity`.

- [ ] **Step 3: Verify legend suppression in Rust renderer**

  The `legend.disabled` flag flows through `EncodingSpec.legend` (which is `Option<LegendSpec>` in `crates/ferrum-core/src/spec/encoding.rs`). The legend builder in `prepare.rs` / layout code must check this flag for all channels, not just color.

  Read the legend construction path:
  1. `crates/ferrum-core/src/render/prepare.rs` or `crates/ferrum-core/src/layout/` — find where size/shape/opacity legends are constructed.
  2. Check if the `legend.disabled` check already applies to all channels or only color.
  3. If only color, extend the check:

  ```rust
  // Pseudocode in the legend construction path:
  fn should_build_legend(encoding: &EncodingSpec) -> bool {
      if let Some(ref legend) = encoding.legend {
          if let Some(disabled) = legend.extra.get("disabled") {
              if disabled.as_bool() == Some(true) {
                  return false;
              }
          }
      }
      true
  }
  ```

  Apply this check for size, shape, and opacity channels, not just color.

- [ ] **Step 4: Also add "legend" to Fill and Stroke**

  `Fill` already has `"legend"` in its `_honored_kwargs` (line 187). Check `Stroke` — if missing, add it. Also add to `FillOpacity`, `StrokeOpacity`, `StrokeWidth`, `StrokeDash` for completeness (spec says "all appearance channels").

- [ ] **Step 5: Write tests**

  ```python
  import polars as pl
  import ferrum as fm

  df = pl.DataFrame({
      "x": [1.0, 2.0, 3.0],
      "y": [1.0, 2.0, 3.0],
      "s": [10.0, 20.0, 30.0],
      "cat": ["A", "B", "C"],
  })

  def test_size_legend_suppressed():
      c = fm.Chart(df).mark_point().encode(
          x="x", y="y", size=fm.Size("s", legend=None)
      )
      svg = c.to_svg()
      assert "<svg" in svg
      # No size legend should appear

  def test_shape_legend_suppressed():
      c = fm.Chart(df).mark_point().encode(
          x="x", y="y", shape=fm.Shape("cat", legend=False)
      )
      svg = c.to_svg()
      assert "<svg" in svg

  def test_opacity_legend_suppressed():
      c = fm.Chart(df).mark_point().encode(
          x="x", y="y", opacity=fm.Opacity("s", legend=None)
      )
      svg = c.to_svg()
      assert "<svg" in svg
  ```

- [ ] **Step 6: Verify**

  ```bash
  uv run pytest tests/test_phase_11e_density_multiple.py tests/test_phase_11e_bw_adjust.py -v  # regression
  uv run pytest -k "legend" -v
  ```

- [ ] **Step 7: Commit**

  ```
  feat(encoding): honor legend kwarg on Size, Shape, Opacity channels
  ```

---

## Task 11e8: condition kwarg on all appearance channels

**Spec reference:** §9.8

**Context:** The `condition` kwarg is accepted by all appearance channels (they don't raise on unknown kwargs — `ChannelBase.__init__` calls `warn_once` for unrecognized kwargs). Phase 11e wires it through: Python validates the condition, serializes it into the ChartSpec, and the SceneGraph carries `ConditionalEncoding` entries. SVG mode silently ignores them; WASM resolves them at runtime (11c dependency).

### Steps

- [ ] **Step 1: Add "condition" to _honored_kwargs for all appearance channels**

  In `src/ferrum/encoding/appearance.py`, add `"condition"` to `_honored_kwargs` for every channel class:

  ```python
  # Color (line 55):
  _honored_kwargs = frozenset(["type", "scheme", "scale", "title", "legend", "sort", "condition"])

  # Size (update from Step 11e7):
  _honored_kwargs = frozenset(["type", "scale", "title", "legend", "condition"])

  # Shape:
  _honored_kwargs = frozenset(["type", "scale", "title", "legend", "condition"])

  # Opacity:
  _honored_kwargs = frozenset(["type", "scale", "title", "legend", "condition"])

  # Fill:
  _honored_kwargs = frozenset(["type", "scheme", "scale", "title", "legend", "condition"])

  # Stroke:
  _honored_kwargs = frozenset(["type", "scheme", "scale", "title", "condition"])

  # FillOpacity:
  _honored_kwargs = frozenset(["type", "scale", "title", "condition"])

  # StrokeOpacity:
  _honored_kwargs = frozenset(["type", "condition"])

  # StrokeWidth:
  _honored_kwargs = frozenset(["type", "condition"])

  # StrokeDash:
  _honored_kwargs = frozenset(["type", "condition"])
  ```

- [ ] **Step 1b: Read ConditionalEncoding shape in ferrum-scene**

  Before writing the serialization logic, read `crates/ferrum-scene/src/selection.rs` and find the `ConditionalEncoding` struct. Match the Python-side dict keys exactly to the Rust struct's serde field names. The `_serialize_condition` method below assumes specific field names (`selection`, `channel`, `if_value`, `else_value`) — verify these against the actual struct before coding.

- [ ] **Step 2: Validate and serialize condition in ChannelBase**

  In `src/ferrum/encoding/base.py`, add condition handling to `to_encoding_spec_dict()`. The dict shape MUST match the `ConditionalEncoding` struct fields discovered in Step 1b:

  ```python
  def to_encoding_spec_dict(self) -> dict:
      out: dict = {"field": self.field}
      ...existing logic...

      # Condition: extract selection name and if/else encoding values.
      if (cond := self._kwargs.get("condition")) is not None:
          out["condition"] = self._serialize_condition(cond)

      return out

  def _serialize_condition(self, cond) -> dict:
      """Serialize a condition kwarg to the dict shape the Rust spec expects.

      Expected input shapes:
      1. dict: {"selection": "sel_name", "value": "red"}
         → if the selection is active, use "red"; else use the channel's field mapping.
      2. dict: {"selection": "sel_name", "value": "red", "else": "blue"}
         → if active, "red"; else "blue".
      3. Selection object with .name attribute + value/else keys.

      Output shape matches ConditionalEncoding in ferrum-scene:
      {"selection": "sel_name", "channel": "color",
       "if_value": {"constant": "red"}, "else_value": {"field": "origin"}}
      """
      if isinstance(cond, dict):
          sel = cond.get("selection", cond.get("sel", ""))
          if_val = cond.get("value", cond.get("if"))
          else_val = cond.get("else")
          return {
              "selection": sel,
              "channel": self._channel_name,
              "if_value": self._encode_value(if_val),
              "else_value": self._encode_value(else_val) if else_val is not None else None,
          }
      # If it's a Selection object, extract .name
      sel_name = getattr(cond, "name", str(cond))
      return {
          "selection": sel_name,
          "channel": self._channel_name,
          "if_value": None,
          "else_value": None,
      }

  @staticmethod
  def _encode_value(val) -> dict:
      if isinstance(val, str):
          return {"constant": val}
      if isinstance(val, (int, float)):
          return {"constant": val}
      if isinstance(val, dict) and "field" in val:
          return {"field": val["field"]}
      return {"constant": val}
  ```

- [ ] **Step 3: Add condition field to Rust EncodingSpec**

  In `crates/ferrum-core/src/spec/encoding.rs`, add:

  ```rust
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub condition: Option<serde_json::Value>,
  ```

  This stores the condition as opaque JSON for now. The scene builder reads it and converts to `ConditionalEncoding` (already defined in `ferrum-scene`).

- [ ] **Step 4: Propagate conditions to SceneGraph**

  In `crates/ferrum-core/src/render/scene_build.rs`, when building the `InteractionConfig` for each panel, collect all condition values from the panel's encodings and convert them to `ConditionalEncoding` entries:

  ```rust
  let mut conditionals: Vec<ConditionalEncoding> = Vec::new();
  // For each encoding channel that has a `condition` field:
  for (channel_name, enc) in encoding_channels(spec) {
      if let Some(cond_json) = &enc.condition {
          if let Ok(ce) = serde_json::from_value::<ConditionalEncoding>(cond_json.clone()) {
              conditionals.push(ce);
          }
      }
  }
  // Set on the Panel's InteractionConfig:
  interaction_config.conditionals = conditionals;
  ```

  The `ConditionalEncoding` type is already in `crates/ferrum-scene/src/selection.rs`. Verify its fields match the dict shape from Step 2.

- [ ] **Step 5: SVG walker ignores conditions (no-op)**

  In `crates/ferrum-core/src/render/svg_walk.rs`, verify that the walker does not attempt to resolve conditions. It should simply skip `InteractionConfig.conditionals` when emitting SVG. This is the expected behavior — conditions are runtime-only in the WASM renderer.

  Add a comment:

  ```rust
  // NOTE: ConditionalEncoding entries are ignored in SVG mode.
  // The WASM renderer (11c) resolves them at runtime based on
  // active selections.
  ```

- [ ] **Step 6: Write tests**

  ```python
  import polars as pl
  import ferrum as fm

  df = pl.DataFrame({
      "x": [1.0, 2.0, 3.0],
      "y": [1.0, 2.0, 3.0],
      "cat": ["A", "B", "C"],
  })

  def test_condition_kwarg_accepted():
      """condition kwarg should not trigger warn_once."""
      c = fm.Chart(df).mark_point().encode(
          x="x", y="y",
          color=fm.Color("cat", condition={"selection": "hover", "value": "red"}),
      )
      # Should not raise; SVG output ignores the condition.
      svg = c.to_svg()
      assert "<svg" in svg

  def test_condition_in_chartspec_json():
      """condition should appear in the serialized ChartSpec."""
      c = fm.Chart(df).mark_point().encode(
          x="x", y="y",
          color=fm.Color("cat", condition={"selection": "hover", "value": "red"}),
      )
      json_str = c.to_json()
      assert "condition" in json_str
      assert "hover" in json_str

  def test_condition_on_multiple_channels():
      c = fm.Chart(df).mark_point().encode(
          x="x", y="y",
          color=fm.Color("cat", condition={"selection": "sel1", "value": "blue"}),
          opacity=fm.Opacity("x", condition={"selection": "sel1", "value": 1.0, "else": 0.3}),
      )
      svg = c.to_svg()
      assert "<svg" in svg
  ```

- [ ] **Step 7: Verify**

  ```bash
  uv run pytest -k "condition" -v
  ```

- [ ] **Step 8: Commit**

  ```
  feat(encoding): wire condition kwarg through all appearance channels to ChartSpec
  ```

---

## Task 11e9: TimeScale calendar-aware month/year ticks

**Spec reference:** §9.9

**Context:** The current `nice_time_interval_ms()` in `crates/ferrum-core/src/scale/ticks.rs` uses approximate constants: `MONTH = 30 * DAY` and `YEAR = 365 * DAY` (lines 73–74). The `time_ticks()` method in `crates/ferrum-core/src/scale/time.rs` generates ticks by stepping at fixed millisecond intervals, which means month-level ticks land on "every 30 days" instead of on actual calendar boundaries (Jan 1, Feb 1, Mar 1, etc.). This is visually wrong for any chart spanning months or years.

### Steps

- [ ] **Step 1: Add chrono dependency to ferrum-core**

  `chrono` is already a transitive dependency via `arrow`. Add it as a direct dependency to make the import explicit:

  In `crates/ferrum-core/Cargo.toml`:

  ```toml
  [dependencies]
  chrono = { version = "0.4", default-features = false, features = ["std"] }
  ```

  Verify the version matches what `arrow` pulls in to avoid duplicate builds:

  ```bash
  cargo tree -p ferrum-core | grep chrono
  ```

- [ ] **Step 2: Add calendar-aware tick generation**

  In `crates/ferrum-core/src/scale/ticks.rs`, add a new function:

  ```rust
  use chrono::{Datelike, NaiveDateTime, Months, Duration as ChronoDuration};

  /// Calendar interval level for time tick generation.
  #[derive(Debug, Clone, Copy, PartialEq, Eq)]
  pub(crate) enum CalendarInterval {
      Milliseconds(u64),
      Months(u32),
      Years(i32),
  }

  /// Choose the best calendar-aware interval for the given span.
  pub(crate) fn nice_calendar_interval(span_ms: f64, count: usize) -> CalendarInterval {
      if count == 0 || !span_ms.is_finite() || span_ms <= 0.0 {
          return CalendarInterval::Milliseconds(1000);
      }
      let target = span_ms / count as f64;

      const SECOND: f64 = 1_000.0;
      const MINUTE: f64 = 60.0 * SECOND;
      const HOUR: f64 = 60.0 * MINUTE;
      const DAY: f64 = 24.0 * HOUR;
      const WEEK: f64 = 7.0 * DAY;
      const MONTH_APPROX: f64 = 30.44 * DAY;
      const YEAR_APPROX: f64 = 365.25 * DAY;

      // Sub-month intervals use fixed millisecond steps.
      let fixed_candidates: &[(f64, u64)] = &[
          (SECOND, 1_000),
          (5.0 * SECOND, 5_000),
          (15.0 * SECOND, 15_000),
          (30.0 * SECOND, 30_000),
          (MINUTE, 60_000),
          (5.0 * MINUTE, 300_000),
          (15.0 * MINUTE, 900_000),
          (30.0 * MINUTE, 1_800_000),
          (HOUR, 3_600_000),
          (3.0 * HOUR, 10_800_000),
          (6.0 * HOUR, 21_600_000),
          (12.0 * HOUR, 43_200_000),
          (DAY, 86_400_000),
          (2.0 * DAY, 172_800_000),
          (WEEK, 604_800_000),
      ];

      // If target is below one month, use fixed intervals.
      if target < MONTH_APPROX {
          let mut chosen = fixed_candidates[0].1;
          for &(approx, ms) in fixed_candidates {
              if approx <= target {
                  chosen = ms;
              }
          }
          return CalendarInterval::Milliseconds(chosen);
      }

      // Month-level intervals.
      let month_candidates: &[(f64, u32)] = &[
          (MONTH_APPROX, 1),
          (3.0 * MONTH_APPROX, 3),
          (6.0 * MONTH_APPROX, 6),
      ];
      for &(approx, months) in month_candidates {
          if target < approx * 1.5 {
              return CalendarInterval::Months(months);
          }
      }

      // Year-level intervals.
      let years = (target / YEAR_APPROX).round().max(1.0) as i32;
      CalendarInterval::Years(years)
  }

  /// Generate calendar-aware ticks between `lo_ms` and `hi_ms` (epoch milliseconds).
  pub(crate) fn calendar_ticks(lo_ms: f64, hi_ms: f64, interval: CalendarInterval) -> Vec<f64> {
      match interval {
          CalendarInterval::Milliseconds(step) => {
              // Fixed-step ticks (unchanged from current logic).
              let step = step as f64;
              let start = (lo_ms / step).ceil() * step;
              let end = (hi_ms / step).floor() * step;
              let n = ((end - start) / step).round() as i64;
              if n < 0 { return Vec::new(); }
              (0..=(n as usize)).map(|i| start + (i as f64) * step).collect()
          }
          CalendarInterval::Months(step) => {
              // Snap to first-of-month boundaries.
              let start_dt = epoch_ms_to_naive(lo_ms);
              let mut dt = snap_to_month_start(start_dt);
              if naive_to_epoch_ms(&dt) < lo_ms {
                  dt = advance_months(dt, step);
              }
              let mut ticks = Vec::new();
              loop {
                  let ms = naive_to_epoch_ms(&dt);
                  if ms > hi_ms { break; }
                  ticks.push(ms);
                  dt = advance_months(dt, step);
              }
              ticks
          }
          CalendarInterval::Years(step) => {
              // Snap to Jan 1 boundaries.
              let start_dt = epoch_ms_to_naive(lo_ms);
              let mut year = start_dt.year();
              if start_dt > NaiveDateTime::new(
                  chrono::NaiveDate::from_ymd_opt(year, 1, 1).unwrap(),
                  chrono::NaiveTime::MIN,
              ) {
                  year += step;
              }
              // Round year to nearest multiple of step.
              year = ((year as f64 / step as f64).ceil() * step as f64) as i32;
              let mut ticks = Vec::new();
              loop {
                  let dt = NaiveDateTime::new(
                      chrono::NaiveDate::from_ymd_opt(year, 1, 1).unwrap(),
                      chrono::NaiveTime::MIN,
                  );
                  let ms = naive_to_epoch_ms(&dt);
                  if ms > hi_ms { break; }
                  if ms >= lo_ms {
                      ticks.push(ms);
                  }
                  year += step;
              }
              ticks
          }
      }
  }

  fn epoch_ms_to_naive(ms: f64) -> NaiveDateTime {
      let secs = (ms / 1000.0).floor() as i64;
      let nsecs = ((ms % 1000.0) * 1_000_000.0) as u32;
      NaiveDateTime::from_timestamp_opt(secs, nsecs)
          .unwrap_or_else(|| NaiveDateTime::from_timestamp_opt(0, 0).unwrap())
  }

  fn naive_to_epoch_ms(dt: &NaiveDateTime) -> f64 {
      dt.and_utc().timestamp_millis() as f64
  }

  fn snap_to_month_start(dt: NaiveDateTime) -> NaiveDateTime {
      NaiveDateTime::new(
          chrono::NaiveDate::from_ymd_opt(dt.year(), dt.month(), 1).unwrap(),
          chrono::NaiveTime::MIN,
      )
  }

  fn advance_months(dt: NaiveDateTime, months: u32) -> NaiveDateTime {
      dt.checked_add_months(Months::new(months))
          .unwrap_or(dt)
  }
  ```

- [ ] **Step 3: Rewrite TimeScale::time_ticks() to use calendar ticks**

  In `crates/ferrum-core/src/scale/time.rs`, replace the `time_ticks()` method:

  ```rust
  fn time_ticks(&self, count: usize) -> Vec<f64> {
      let [d0, d1] = self.0.domain;
      let lo = d0.min(d1);
      let hi = d0.max(d1);
      let span = hi - lo;

      let interval = super::ticks::nice_calendar_interval(span, count);
      let mut out = super::ticks::calendar_ticks(lo, hi, interval);

      if d0 > d1 {
          out.reverse();
      }
      out
  }
  ```

- [ ] **Step 4: Rewrite TimeScale::time_nice() to snap to calendar boundaries**

  ```rust
  fn time_nice(&self) -> Self {
      let [d0, d1] = self.0.domain;
      let lo = d0.min(d1);
      let hi = d0.max(d1);

      let interval = super::ticks::nice_calendar_interval(hi - lo, 10);
      let (new_lo, new_hi) = match interval {
          super::ticks::CalendarInterval::Milliseconds(step) => {
              let step = step as f64;
              ((lo / step).floor() * step, (hi / step).ceil() * step)
          }
          super::ticks::CalendarInterval::Months(_) => {
              let lo_dt = super::ticks::epoch_ms_to_naive(lo);
              let hi_dt = super::ticks::epoch_ms_to_naive(hi);
              let lo_snapped = super::ticks::snap_to_month_start(lo_dt);
              let hi_snapped = super::ticks::advance_months(
                  super::ticks::snap_to_month_start(hi_dt), 1
              );
              (super::ticks::naive_to_epoch_ms(&lo_snapped),
               super::ticks::naive_to_epoch_ms(&hi_snapped))
          }
          super::ticks::CalendarInterval::Years(step) => {
              let lo_dt = super::ticks::epoch_ms_to_naive(lo);
              let hi_dt = super::ticks::epoch_ms_to_naive(hi);
              let lo_year = lo_dt.year();
              let hi_year = hi_dt.year() + step;
              let lo_snapped = chrono::NaiveDateTime::new(
                  chrono::NaiveDate::from_ymd_opt(lo_year, 1, 1).unwrap(),
                  chrono::NaiveTime::MIN,
              );
              let hi_snapped = chrono::NaiveDateTime::new(
                  chrono::NaiveDate::from_ymd_opt(hi_year, 1, 1).unwrap(),
                  chrono::NaiveTime::MIN,
              );
              (super::ticks::naive_to_epoch_ms(&lo_snapped),
               super::ticks::naive_to_epoch_ms(&hi_snapped))
          }
      };

      let new_domain = if d0 <= d1 { [new_lo, new_hi] } else { [new_hi, new_lo] };
      TimeScale(
          LinearScaleData { domain: new_domain, range: self.0.range, clamp: self.0.clamp },
          self.1,
      )
  }
  ```

  **Visibility note:** The helper functions `epoch_ms_to_naive`, `naive_to_epoch_ms`, `snap_to_month_start`, `advance_months` need to be `pub(crate)` in `ticks.rs` so `time.rs` can call them.

- [ ] **Step 5: Keep nice_time_interval_ms for backward compatibility**

  The existing `nice_time_interval_ms()` function may be called from other places. Keep it but mark it with a deprecation comment. The new code path uses `nice_calendar_interval()` and `calendar_ticks()`.

- [ ] **Step 6: Write tests**

  Create `tests/test_phase_11e_time_calendar.py`:

  ```python
  import polars as pl
  import ferrum as fm
  from datetime import datetime

  def test_month_ticks_snap_to_calendar():
      """Ticks spanning months should land on month boundaries."""
      dates = [datetime(2024, 1, 15), datetime(2024, 6, 15)]
      df = pl.DataFrame({
          "date": dates,
          "y": [1.0, 2.0],
      })
      c = fm.Chart(df).mark_line().encode(x=fm.X("date", type_="T"), y="y")
      svg = c.to_svg()
      # Tick labels should be month names, not "Jan 30", "Mar 1" etc.
      # Look for month-boundary evidence in the SVG text.
      assert "<svg" in svg

  def test_year_ticks_snap_to_jan1():
      dates = [datetime(2020, 6, 1), datetime(2025, 6, 1)]
      df = pl.DataFrame({
          "date": dates,
          "y": [1.0, 2.0],
      })
      c = fm.Chart(df).mark_line().encode(x=fm.X("date", type_="T"), y="y")
      svg = c.to_svg()
      assert "<svg" in svg
  ```

  Also add Rust unit tests in `ticks.rs`:

  ```rust
  #[test]
  fn test_calendar_ticks_month_boundaries() {
      // Domain: 2024-01-15 to 2024-06-15 (epoch ms)
      let lo = 1705276800000.0; // 2024-01-15T00:00:00Z
      let hi = 1718409600000.0; // 2024-06-15T00:00:00Z
      let interval = nice_calendar_interval(hi - lo, 6);
      let ticks = calendar_ticks(lo, hi, interval);
      // Ticks should be on month boundaries: Feb 1, Mar 1, Apr 1, May 1, Jun 1
      for t in &ticks {
          let dt = epoch_ms_to_naive(*t);
          assert_eq!(dt.day(), 1, "tick {t} does not land on day 1: {:?}", dt);
      }
  }

  #[test]
  fn test_calendar_ticks_year_boundaries() {
      let lo = 1577836800000.0; // 2020-01-01T00:00:00Z
      let hi = 1735689600000.0; // 2025-01-01T00:00:00Z
      let interval = nice_calendar_interval(hi - lo, 5);
      let ticks = calendar_ticks(lo, hi, interval);
      for t in &ticks {
          let dt = epoch_ms_to_naive(*t);
          assert_eq!(dt.month(), 1, "tick not on January: {:?}", dt);
          assert_eq!(dt.day(), 1, "tick not on day 1: {:?}", dt);
      }
  }
  ```

- [ ] **Step 7: Verify**

  ```bash
  DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core
  uv run pytest tests/test_phase_11e_time_calendar.py -v
  ```

  Also run the full test suite to verify no regressions in existing temporal axis tests:

  ```bash
  uv run pytest -v
  ```

  **Golden breakage warning:** Changing `time_ticks()` behavior will shift tick positions on every existing temporal-axis chart. Any golden SVGs with temporal axes WILL fail byte-equality. After this task, regenerate and re-inspect all temporal-axis goldens:

  ```bash
  # Find temporal goldens:
  grep -rl 'TimeScale\|type_="T"\|:T' tests/ | head -20
  # Regenerate affected goldens, then rasterize and visually inspect each one.
  python scripts/snapshot-goldens.py
  ```

  This is an intentional, correct change — the new tick positions are better (calendar-snapped). Update the golden reference files and visually confirm the new tick labels.

- [ ] **Step 8: Commit**

  ```
  feat(scale): calendar-aware month/year tick generation for TimeScale
  ```

---

## Task 11e10: Key channel wiring (encoding → ChartSpec → MarkBatch.keys)

**Spec reference:** §9.10

**Context:** The `Key(field)` encoding class exists in `src/ferrum/encoding/text.py` (line 211). The `MarkBatch.keys` field exists in `ferrum-scene` (line 45 of `types.rs`). The gap is wiring the Key encoding through the ChartSpec to populate `MarkBatch.keys` in the scene graph. Animated transitions (which consume keys) are wired in 11c.

### Steps

- [ ] **Step 1: Check if Key already flows into ChartSpec**

  Read the `Encoding` struct in `crates/ferrum-core/src/spec/encoding.rs` — check if there's a `key` field. If not, add one:

  ```rust
  #[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
  pub struct Encoding {
      pub x: Option<EncodingSpec>,
      pub y: Option<EncodingSpec>,
      pub color: Option<EncodingSpec>,
      pub size: Option<EncodingSpec>,
      pub shape: Option<EncodingSpec>,
      pub opacity: Option<EncodingSpec>,
      ... existing fields ...
      #[serde(default, skip_serializing_if = "Option::is_none")]
      pub key: Option<EncodingSpec>,   // <-- NEW
  }
  ```

- [ ] **Step 2: Check Python-side encoding serialization**

  Read the `Chart._build_spec()` or `Chart._render_inputs()` path to verify that `key=fm.Key("id")` in `.encode(...)` gets serialized into the ChartSpec JSON. The `Key` class extends `ChannelBase`, so its `to_encoding_spec_dict()` should produce `{"field": "id"}` with no issues. The question is whether the chart encoding collector maps the `key` channel name to the `Encoding.key` field.

  Find the encoding collection code in `chart.py` (look for where `_encoding` dict is built into the spec). Ensure `"key"` maps to the spec's `key` field.

- [ ] **Step 3: Populate MarkBatch.keys in scene_build.rs**

  In `crates/ferrum-core/src/render/scene_build.rs`, where `MarkBatch` is constructed, read the `key` encoding field and populate `keys`:

  ```rust
  // Read the key field from the layer or spec encoding.
  let keys: Option<Vec<String>> = encoding.key.as_ref().and_then(|key_enc| {
      let key_field = &key_enc.field;
      // Look up the key column in the data batch.
      let schema = batch.schema();
      schema.index_of(key_field).ok().map(|idx| {
          let col = batch.column(idx);
          // Convert to strings regardless of type.
          (0..col.len()).map(|i| {
              if col.is_null(i) {
                  String::new()
              } else {
                  // Use arrow's string representation.
                  arrow::util::display::array_value_to_string(col, i)
                      .unwrap_or_default()
              }
          }).collect()
      })
  });

  MarkBatch {
      kind: ...,
      nodes: ...,
      keys,  // <-- populate from above
      ...
  }
  ```

- [ ] **Step 4: Write tests**

  Create `tests/test_phase_11e_key_channel.py`:

  ```python
  import polars as pl
  import ferrum as fm
  import json

  df = pl.DataFrame({
      "x": [1.0, 2.0, 3.0],
      "y": [1.0, 2.0, 3.0],
      "id": ["a", "b", "c"],
  })

  def test_key_in_chartspec_json():
      """Key encoding should appear in the serialized ChartSpec."""
      c = fm.Chart(df).mark_point().encode(x="x", y="y", key=fm.Key("id"))
      spec = json.loads(c.to_json())
      assert "key" in spec.get("encoding", {})
      assert spec["encoding"]["key"]["field"] == "id"

  def test_key_renders_without_error():
      c = fm.Chart(df).mark_point().encode(x="x", y="y", key=fm.Key("id"))
      svg = c.to_svg()
      assert "<svg" in svg

  def test_key_absent_by_default():
      """No key encoding → MarkBatch.keys should be None."""
      c = fm.Chart(df).mark_point().encode(x="x", y="y")
      spec = json.loads(c.to_json())
      assert spec.get("encoding", {}).get("key") is None
  ```

  Add a Rust test that verifies the scene graph's `MarkBatch.keys` is populated when a key encoding is present.

- [ ] **Step 5: Verify**

  ```bash
  uv run pytest tests/test_phase_11e_key_channel.py -v
  DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core
  ```

- [ ] **Step 6: Commit**

  ```
  feat(encoding): wire Key channel through ChartSpec to MarkBatch.keys
  ```

---

## Validation checklist

Run after all 11e tasks are complete:

- [ ] **Full Python test suite:** `uv run pytest -v`
- [ ] **Full Rust test suite:** `DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test`
- [ ] **No remaining NotImplementedError in density/function/blend:**
  ```bash
  grep -rn 'NotImplementedError\|warn_once.*deferred\|warn_once.*Phase 11' src/ferrum/ | grep -v __pycache__ | grep -v deferred.py
  ```
  The only remaining `NotImplementedError`s should be in `marks/deferred.py` for `arc`, `image`, `geoshape`, `label` (those are 11d, not 11e).
- [ ] **Golden SVGs for new marks:** Generate and visually inspect goldens for density stack/fill/dodge, swarm dodge, mark_function multi-layer, blend additive, and calendar time ticks.
  ```bash
  python scripts/snapshot-goldens.py
  ```
  Read each PNG and confirm correctness.
- [ ] **Byte-identical existing goldens (non-temporal):** Ensure all pre-existing golden SVGs WITHOUT temporal axes still pass:
  ```bash
  uv run pytest tests/ -k golden -v
  ```
  **Exception:** Temporal-axis goldens will change due to 11e9 (calendar-aware ticks). Those must be regenerated, rasterized, and visually re-inspected — not byte-compared against old references.
- [ ] **No regressions in mark_density/mark_hex/mark_swarm defaults:** Run any existing tests for these marks to verify the default code paths are unchanged.
- [ ] **Lite review gate:** Before committing, dispatch `rust-review-lite` on staged `*.rs` changes and `python-review-lite` on staged `*.py` changes. Act on the verdicts (clean → commit, block → fix, escalate → halt).

---

## Parallelization notes

Tasks 11e1–11e10 are largely independent and can be executed in parallel with the following constraints:

- **11e1 and 11e2 share KdeSpec changes.** If executed in parallel, coordinate on the `KdeSpec` struct (both add fields). Easier to do 11e2 first (smaller) then 11e1.
- **11e7 and 11e8 both modify `appearance.py` `_honored_kwargs`.** If parallel, merge the `_honored_kwargs` changes at the end.
- **11e6 depends on 11a** (scene_build.rs + svg_walk.rs must exist).
- **11e8 and 11e10 both modify scene_build.rs.** Coordinate if parallel.

Recommended execution order for a single engineer: 11e2 → 11e3 → 11e4 → 11e7 → 11e1 → 11e5 → 11e6 → 11e8 → 11e9 → 11e10 (easiest first, building confidence before the more involved calendar and condition tasks).
