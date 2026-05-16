# Schwabish audit — remaining issues remediation

**Date:** 2026-05-12
**Branch:** `chore/schwabish-remaining`
**Source:** `docs/superpowers/followups/2026-05-12-schwabish-audit-remaining.md`

---

## Scope

Remediate all open items from the Schwabish audit heavyweight reviews, excluding
P9 (examples extension for Schwabish-evolved functions). Ten work items organized
bottom-up by risk on a single branch with one commit per item.

---

## WI-1: Ruff D-rule fixes in `figures.py`

Four pre-existing docstring violations in `src/ferrum/figures.py`:

| Line  | Rule | Fix |
|-------|------|-----|
| ~95   | D205 | Add blank line between summary and description in `_resolve_source` |
| ~706  | D202 | Remove blank line after `importance_chart` docstring |
| ~1032 | D202 | Remove blank line after `pdp_chart` docstring |
| ~1272 | D401 | Rewrite `validation_curve_chart` first line to imperative mood |

Verify with `uv run ruff check src/ferrum/figures.py --select D`.

---

## WI-2: Pyright baseline

No `pyrightconfig.json` exists today. Establish a clean baseline:

1. Add `pyrightconfig.json` with `typeCheckingMode: "basic"`,
   `pythonVersion: "3.10"`, `include: ["src/ferrum"]`.
2. Run pyright, triage errors:
   - **Real import issues** — fix them.
   - **PyO3 `_core` stubs** — `_core.pyi` already exists; if pyright can't
     resolve it, add the stub path to `extraPaths`.
   - **Persistent false positives** — add targeted `# type: ignore[...]` with
     the specific error code, not bare `# type: ignore`.
3. Goal: zero pyright errors on `src/ferrum/` so the check can gate CI.

---

## WI-3: R8b regression test (`axis_batch_for_y` silent swallow)

**File:** `crates/ferrum-core/src/render/position.rs`

The `Err(_) => Cow::Borrowed(primary_batch)` fallback in `axis_batch_for_y` is
correct but fragile — a future refactor could remove the downstream
`apply_stack` call that re-derives the error. Add a Rust test that:

1. Constructs a `RecordBatch` where `apply_stack` will fail (e.g., a Boolean x
   column that isn't Float64 or Utf8).
2. Calls `axis_batch_for_y` and asserts it returns `Cow::Borrowed` (the
   fallback path).
3. Calls `apply_stack` directly on the same batch and asserts it returns `Err`.

This pins the contract: "if stack fails, we fall back to raw data AND the error
is still surfaceable downstream."

---

## WI-4: Rust dead-code cleanup

Target: reduce ~106 warnings to <30 by deleting structural dead code.

### Delete outright

| Item | Location | Reason |
|------|----------|--------|
| `apply_transforms` | `transform/core.rs` | Legacy entry point; `apply_transforms_named` is the real one |
| `apply_transforms_with_context` | `transform/core.rs` | Same — legacy twin |
| `CategoricalPalette` + `Scheme` enum | `render/color/scheme.rs` | Unfinished scaffolding; `render::palette::categorical_palette()` is the real path |
| `OutlierRow` | `transform/letter_value.rs` | Never constructed; outlier data built inline |
| `range_pair()` × 5 | `scale/{linear,log,symlog,time,ordinal}.rs` | Planned uniform API that was never integrated |

### Migrate tests

Tests in `transform/core.rs` that call `apply_transforms` /
`apply_transforms_with_context` must be migrated to call
`apply_transforms_named` instead, since that is the live entry point.

### Evaluate and likely delete

Unused constants (`DEFAULT_HEURISTIC_K`, `DEFAULT_PADDING`,
`DEFAULT_AXIS_TITLE_PADDING`, `DEFAULT_GRID_ENABLED`, `INTER_FONT_FAMILY`) and
unused functions (`categorical_color`, `resolve_scales`, `scale_value`). Confirm
no references in comments/docs as intended future values; if none, delete.

### Leave alone

- The 2 existing `#[allow(dead_code)]` sites (`ticks.rs`, `arrow_cast.rs`).
- Any item where removal would break the PyO3 API surface.

---

## WI-5: LOESS O(n²) optimization

**File:** `crates/ferrum-core/src/transform/smooth.rs`

`loess_at_point` re-sorts `xs`/`ys` per query point — O(n log n) per query,
O(n² log n) total. For n > ~1000 this is noticeable.

### Fix

1. In the LOESS entry point (the closure at ~line 232), sort `(xs, ys)` pairs
   by `xs` once before the query loop.
2. Replace the per-query sort in `loess_at_point` with a
   `partition_point`-bounded slice on the pre-sorted array.
3. Tricube weights and local regression stay unchanged — only window-finding
   changes.

### Testing

Existing LOESS golden tests cover correctness (the output must be
bit-identical since the same points end up in each window). Add a test with
n=5000 to confirm results match the naive approach within f64 tolerance.

---

## WI-6: Prep closure dedup

**File:** `src/ferrum/chart.py`

Four closures share a pattern: capture flags → guard on column presence →
augment DataFrame with `_`-prefixed columns → return. The bodies diverge
(F1-optimum computation vs. text casting vs. constant injection vs. geometric
bounds), so a full generic helper would be more abstract than the callsites.

### Fix

Extract the repeated guard-and-augment skeleton into a module-level helper in
`chart.py`, placed near the existing `_inject_constant` helper:

```python
def _prep_if_column(df, column: str, augment_fn):
    """Pass-through if column is missing; otherwise apply augment_fn."""
    if column not in df.columns:
        return df
    return augment_fn(df)
```

Each closure calls `_prep_if_column(df, "threshold", lambda df: ...)` with its
specific column-building logic inline. This trims the boilerplate without forcing
dissimilar logic into a shared mold.

---

## WI-7: Wire `lmplot` reserved kwargs

**File:** `src/ferrum/figure/regression.py`

### `scatter_kws`

Forward `**scatter_kws` to the `mark_point(...)` call that builds the scatter
layer (~line 202). Type: `dict | None`.

### `line_kws`

Forward `**line_kws` to the `mark_smooth(...)` call that builds the fit line
(~line 211). Type: `dict | None`.

### `truncate`

**Resolve during implementation:** The spec default is `truncate=False` (extend
fit line beyond data range). Ferrum's Smooth transform currently generates a grid
from `x_min` to `x_max` of the data — effectively `truncate=True`.

Implementation plan:
- Add `x_range: Option<(f64, f64)>` to `SmoothSpec` in Rust.
- When `truncate=False`, Python computes padded bounds (5% beyond data range for
  linear methods) and passes them as `x_range`.
- For LOESS/robust, extrapolation beyond data range is statistically meaningless.
  Emit a `UserWarning` and clip to data range regardless. This matches seaborn's
  effective behavior (LOESS extrapolation produces wild artifacts; seaborn shows
  them, ferrum warns and clips — better UX).

### `x_estimator`

Already works for `method="lm"`. For non-LM methods it is silently ignored.
Document the limitation in the docstring (the spec doesn't specify cross-method
behavior).

### Testing

- `scatter_kws={"opacity": 0.3}` → verify the point mark layer has `opacity=0.3`.
- `line_kws={"stroke_width": 3}` → verify the smooth mark layer has
  `stroke_width=3`.
- `truncate=True` vs `truncate=False` → verify the smooth grid x-extent differs.

---

## WI-8: Wire `residplot` reserved kwargs

**File:** `src/ferrum/figure/regression.py`

### `dropna`

When `True` (the default per spec), drop rows where `x` or `y` is null before
passing to the chart:

```python
if dropna:
    data = data.drop_nulls(subset=[x, y])
```

Remove the `del dropna, label` line at ~line 338.

### `label`

Inject a constant `_label` column and encode `color="_label"` to generate a
legend entry. Same pattern as the `model` column in `_resolve_source` compare
routing:

```python
if label is not None:
    data = data.with_columns(pl.lit(label).alias("_label"))
    chart = chart.encode(color="_label")
```

### Testing

- `dropna=True` with NaN rows → verify chart data has fewer rows.
- `dropna=False` with NaN rows → verify chart data retains all rows.
- `label="Residuals"` → verify color encoding is `"_label"` and legend appears.

---

## WI-9: `compare=` routing expansion

**Figures to wire:** `gain_chart`, `lift_chart`, `discrimination_threshold_chart`

These are curve-based diagnostic figures in the same family as `roc_chart` and
`pr_chart`. A user who learns `roc_chart(model, X, y, compare={"alt": other})`
will naturally expect the same pattern on gain/lift/disc_threshold.

### Implementation

1. Add `compare=None` parameter to each function's signature.
2. Forward `compare=compare` to `_resolve_source(...)` call.
3. The existing `_resolve_source` machinery handles the rest — concatenates
   per-model DataFrames with a `model` column, which the chart builder routes to
   `color="model"`.

### Spec update

Add a dated note to `ferrum-spec.md` §3.14 Model Diagnostics:

> **2026-05-12:** `gain_chart`, `lift_chart`, and
> `discrimination_threshold_chart` gain `compare=None` for multi-model overlay,
> consistent with the `roc_chart` / `pr_chart` pattern.

Update the function signatures in the spec to include `compare=None`.

### Testing

For each figure: pass a 2-model dict, verify the returned chart has color
encoded on `"model"` and the data contains both model names.

---

## WI-10: P11 — `Chart.layer()` method and overlay migration

### Add `Chart.layer(*layers)`

The spec (§3.1) defines `.layer(*layers)` — it accepts `Layer` objects. The
`Layer` class exists (`src/ferrum/layer.py`) but `Chart.layer()` is not
implemented.

Implementation:
1. Add `def layer(self, *layers: Layer) -> "Chart"` to `Chart`.
2. For each `Layer`, convert to the internal `_Layer` dataclass:
   - `Layer.mark` → resolve to mark string or `MarkSpec`
   - `Layer.encoding` → encoding dict
   - `Layer.transforms` → transform list
   - `Layer.data` → when `None`, inherit from parent chart (the common case).
     When non-`None`, set `_Layer.data_source` following the existing
     multi-data-source pattern.
3. Call `_expand_layers()` first if the chart is in single-mark state.
4. Append converted layers to `self._layers`.
5. Return `self`.

### Migrate 6 sites

All 6 sites share the same data as the parent chart (`Layer(data=None)` →
inherit):

| Site | File | Current pattern | New pattern |
|------|------|-----------------|-------------|
| `_overlay_metrics_corner` | `_diagnostics/charts.py:122` | `chart + Chart(data).mark_text(...).encode(...)` | `chart.layer(Layer(mark="text", encoding={...}))` |
| `_pr_chart_from_source` | `_diagnostics/charts.py:563` | `chart + Chart(data).mark_rule(...).encode(...)` | `chart.layer(Layer(mark="rule", encoding={...}))` |
| `_importance_chart_from_source` | `_diagnostics/charts.py:984` | `chart + Chart(data).mark_text(...).encode(...)` | `chart.layer(Layer(mark="text", encoding={...}))` |
| `_decision_boundary_chart` | `_diagnostics/charts.py:1907` | `grid_chart + Chart(data).mark_point(...)` | `grid_chart.layer(Layer(mark="point", encoding={...}))` |
| `_classification_report_chart` | `_diagnostics/charts.py:896` | `heatmap + Chart(data).mark_text(...)` | `heatmap.layer(Layer(mark="text", encoding={...}))` |
| `_direct_label_endpoint` | `_direct_label.py:130` | `chart_aug + Chart(data).mark_text(...)` | `chart_aug.layer(Layer(mark="text", encoding={...}))` |

### Mark kwargs

The current `Layer` class doesn't accept mark-level kwargs (e.g., `dx=5`,
`opacity=0.3`). Two options:

- **Option A:** Pass a `MarkSpec` object as `mark=` instead of a string:
  `Layer(mark=mark_text(dx=5), encoding={...})`. This is what the spec implies
  with "Equivalent to `.layer(Layer(mark=mark_*(...)))`".
- **Option B:** Add `mark_kwargs` to `Layer.__init__`.

Option A is spec-aligned. The mark constructor functions (`mark_text`,
`mark_rule`, etc.) already exist as methods on `Chart`; they need to be usable
standalone. Check if they are — if they currently require a `Chart` instance,
extract them as module-level functions or class methods.

**Resolve during implementation:** Check whether `mark_text(dx=5)` can be called
standalone or only as `chart.mark_text(dx=5)`. If the latter, the migration
needs either standalone mark constructors or the `mark_kwargs` approach.

### `+` operator unchanged

The `+` operator stays as the public user-facing composition API. `.layer()` is
the builder-internal alternative.

---

## Commit ordering

| # | Work item | Risk | Touches |
|---|-----------|------|---------|
| C1 | WI-1: Ruff D-rule fixes | zero | Python docstrings only |
| C2 | WI-2: Pyright baseline | zero | config + type annotations |
| C3 | WI-3: R8b regression test | zero | Rust test only |
| C4 | WI-4: Rust dead-code cleanup | low | Rust structural deletes |
| C5 | WI-5: LOESS optimization | low | Rust transform internals |
| C6 | WI-6: Prep closure dedup | low | Python refactor, no API change |
| C7 | WI-7: Wire lmplot kwargs | medium | Python API wiring + Rust `x_range` |
| C8 | WI-8: Wire residplot kwargs | medium | Python API wiring |
| C9 | WI-9: compare= routing | medium | Python API + spec update |
| C10 | WI-10: Chart.layer() + migration | medium | Python API + spec alignment |

---

## Out of scope

- **P9 (examples extension)** — excluded by user decision.
- **LOESS O(n²) for the metrics path specifically** — WI-5 fixes the general
  LOESS path; the metrics-specific combo is only relevant if real usage surfaces
  the slowdown.
