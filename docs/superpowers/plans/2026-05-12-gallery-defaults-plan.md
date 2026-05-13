# Gallery Defaults Remediation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship cohesive, high-quality default chart output for all 30 affected gallery rows, closing the gap with sklearn/seaborn/yellowbrick defaults.

**Architecture:** Three tiers executed in dependency order: Tier A (global visual defaults — CI band styling, reference line color, bar zero-anchor) → Tier B (axis labels + chart titles across ~25 builders) → Tier C (per-chart-type fixes — boxplot, colormaps, markers, legend labels, etc.). All changes are Python-side mark desugars and chart builders; Rust changes limited to plumbing two new theme keys.

**Tech Stack:** Python (marks/composite.py, marks/diagnostic.py, marks/statistical.py, _diagnostics/charts.py, themes/), Rust (crates/ferrum-core/src/layout/mod.rs, render/binding.rs)

---

## Task 1: A1 — CI/Error Band Styling

**Files:**
- Modify: `src/ferrum/marks/composite.py:283-285` (errorband ribbon layer)
- Modify: `src/ferrum/marks/composite.py:304` (ribbon default opacity param)
- Modify: `src/ferrum/marks/composite.py:364-369` (ribbon layer)
- Test: existing `tests/` — `uv run pytest`

- [ ] **Step 1: Update `desugar_errorband` ribbon layer**

In `src/ferrum/marks/composite.py`, change the ribbon layer in `desugar_errorband` (line ~283-285):

```python
# Before
_Layer(
    mark="ribbon",
    encoding={"x": x_field, "y": "lower", "y2": "upper"},
    mark_kwargs={"opacity": 0.3},
    data_source="err",
),

# After
_Layer(
    mark="ribbon",
    encoding={"x": x_field, "y": "lower", "y2": "upper"},
    mark_kwargs={"opacity": 0.2, "stroke": "none"},
    data_source="err",
),
```

- [ ] **Step 2: Update `desugar_ribbon` default opacity**

In `src/ferrum/marks/composite.py`, change the `desugar_ribbon` function signature and its layer (line ~304 and ~364-369):

```python
# Before (signature)
def desugar_ribbon(
    x_field: str | None,
    y_field: str | None,
    *,
    y2_field: str | None = None,
    opacity: float = 0.3,
    interpolate: str = "linear",
) -> tuple:

# After (signature)
def desugar_ribbon(
    x_field: str | None,
    y_field: str | None,
    *,
    y2_field: str | None = None,
    opacity: float = 0.2,
    interpolate: str = "linear",
) -> tuple:
```

And the layer inside `desugar_ribbon`:

```python
# Before
_Layer(
    mark="ribbon",
    encoding={"x": x_field, "y": y_field, "y2": y2_field},
    mark_kwargs={"opacity": opacity},
),

# After
_Layer(
    mark="ribbon",
    encoding={"x": x_field, "y": y_field, "y2": y2_field},
    mark_kwargs={"opacity": opacity, "stroke": "none"},
),
```

- [ ] **Step 3: Update docstrings**

Update the docstrings for `desugar_errorband` and `desugar_ribbon` that mention `opacity=0.3` to say `opacity=0.2`. In `desugar_errorband`:

```python
# Line ~243 — change:
1. ``ribbon`` — ``y=lower``, ``y2=upper``, ``opacity=0.3`` (shaded band).
# To:
1. ``ribbon`` — ``y=lower``, ``y2=upper``, ``opacity=0.2``, ``stroke="none"`` (shaded band).
```

In `desugar_ribbon`:

```python
# Line ~336 — change:
opacity : float, default 0.3
# To:
opacity : float, default 0.2
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest -x -q`
Expected: all pass — this is an additive default change

- [ ] **Step 5: Commit**

```
git add src/ferrum/marks/composite.py
git commit -m "fix(defaults): lower CI band opacity to 0.2, remove stroke outline"
```

---

## Task 2: A2 — Reference Line Color Normalization

**Files:**
- Modify: `src/ferrum/themes/__init__.py` (add keys to `_KNOWN_KEYS`)
- Modify: `src/ferrum/themes/builtins.py` (set per-theme values)
- Modify: `crates/ferrum-core/src/layout/mod.rs` (add fields to `ThemeInputs`)
- Modify: `crates/ferrum-core/src/render/binding.rs` (add fields to `ThemeOverridesSpec` + `apply_theme_overrides`)
- Modify: `src/ferrum/marks/diagnostic.py` (hardcode `#AAAAAA` + `[4,4]` consistently)
- Test: `uv run pytest` + `cargo test`

**Architecture note:** Diagnostic mark desugars run before the theme is resolved (they're called from `Chart._resolve_pending`, not from the render path). They cannot read theme keys at desugar time. The plan: add `reference_line_color`/`reference_line_dash` as theme keys for future use and built-in theme customization, but hardcode the default `#AAAAAA`/`[4,4]` in every desugar. User override is per-mark `stroke=`/`stroke_dash=`.

- [ ] **Step 1: Add keys to Python Theme `_KNOWN_KEYS`**

In `src/ferrum/themes/__init__.py`, add to the `_KNOWN_KEYS` frozenset (after the `"strip_background_color"` line):

```python
            # Reference lines
            "reference_line_color",
            "reference_line_dash",
```

- [ ] **Step 2: Add fields to Rust `ThemeInputs` struct**

In `crates/ferrum-core/src/layout/mod.rs`, add after the `legend_title_font_size` field (~line 175):

```rust
    // Reference lines
    pub reference_line_color: palette::Srgba<u8>,
    pub reference_line_dash: Option<Vec<f64>>,
```

In `Default for ThemeInputs` (after `legend_direction: None,` ~line 257), add:

```rust
            // Reference lines
            reference_line_color: palette::Srgba::new(0xAA, 0xAA, 0xAA, 0xFF),
            reference_line_dash: Some(vec![4.0, 4.0]),
```

- [ ] **Step 3: Add fields to `ThemeOverridesSpec` + `apply_theme_overrides`**

In `crates/ferrum-core/src/render/binding.rs`, add to `ThemeOverridesSpec` (after `row_padding`):

```rust
    // Reference lines
    reference_line_color: Option<String>,
    reference_line_dash: Option<Vec<f64>>,
```

In `apply_theme_overrides` (after the spacing block):

```rust
    // Reference lines
    if let Some(s) = spec.reference_line_color { t.reference_line_color = parse_hex(&s)?; }
    if let Some(v) = spec.reference_line_dash { t.reference_line_dash = Some(v); }
```

- [ ] **Step 4: Set per-theme values in builtins.py**

In `src/ferrum/themes/builtins.py`, add `reference_line_color` to the dark theme:

```python
dark = Theme(
    ...
    reference_line_color="#666666",
)
```

And to publication:

```python
publication = Theme(
    ...
    reference_line_color="#999999",
)
```

- [ ] **Step 5: Normalize all diagnostic mark reference lines**

In `src/ferrum/marks/diagnostic.py`, find every `_Layer` with `stroke_dash` in its `mark_kwargs` and normalize to `"stroke": "#AAAAAA", "stroke_dash": [4, 4]`. There are ~12 instances. Apply this search-and-replace pattern:

For layers that have `mark_kwargs={"stroke_dash": [4, 4]}` (no explicit stroke):
```python
# Before
mark_kwargs={"stroke_dash": [4, 4]},
# After
mark_kwargs={"stroke": "#AAAAAA", "stroke_dash": [4, 4]},
```

For layers that have `mark_kwargs={"stroke_dash": [3, 3], "stroke": "#8a8a8a"}`:
```python
# Before
mark_kwargs={"stroke_dash": [3, 3], "stroke": "#8a8a8a"},
# After
mark_kwargs={"stroke": "#AAAAAA", "stroke_dash": [4, 4]},
```

For layers that have `mark_kwargs={"stroke_dash": [2, 4], "opacity": 0.6}`:
```python
# Before
mark_kwargs={"stroke_dash": [2, 4], "opacity": 0.6},
# After
mark_kwargs={"stroke": "#AAAAAA", "stroke_dash": [4, 4], "opacity": 0.6},
```

Also normalize `_diagnostics/charts.py` line ~573:
```python
# Before
mark_kwargs={"stroke_dash": [3, 3], "stroke": "#8a8a8a"},
# After
mark_kwargs={"stroke": "#AAAAAA", "stroke_dash": [4, 4]},
```

- [ ] **Step 6: Build and test**

Run: `unset CONDA_PREFIX && uv run --no-sync maturin develop`
Run: `uv run pytest -x -q`
Run: `DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test`

Expected: all pass

- [ ] **Step 7: Commit**

```
git add src/ferrum/themes/__init__.py src/ferrum/themes/builtins.py \
    src/ferrum/marks/diagnostic.py src/ferrum/_diagnostics/charts.py \
    crates/ferrum-core/src/layout/mod.rs crates/ferrum-core/src/render/binding.rs
git commit -m "feat(defaults): add reference_line_color/dash theme keys, normalize to #AAAAAA/[4,4]"
```

---

## Task 3: A3 — Bar Y-Axis Zero-Anchoring

**Files:**
- Modify: `src/ferrum/chart.py:349-413` (`_resolve_pending`)
- Test: `tests/test_bar_zero.py` (new)

**Architecture note:** `mark_bar` is a primitive mark — no desugar function. The injection point is `_resolve_pending`: when the resolved mark is `"bar"` and the y-encoding has no explicit scale domain, inject `scale.zero=True`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_bar_zero.py`:

```python
"""Bar chart y-axis zero-anchoring default."""
import polars as pl
import ferrum as fm


def test_bar_default_zero_anchor():
    """mark_bar injects scale.zero=True on y-encoding by default."""
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10, 20, 15]})
    chart = fm.Chart(df).mark_bar().encode(x="cat", y="val")
    spec = chart.to_spec()
    y_enc = spec.get("encoding", {}).get("y", {})
    scale = y_enc.get("scale", {})
    assert scale.get("zero") is True


def test_bar_explicit_domain_no_zero():
    """User-supplied domain suppresses the zero injection."""
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10, 20, 15]})
    chart = (
        fm.Chart(df)
        .mark_bar()
        .encode(x="cat", y=fm.Y("val", scale={"domain": [5, 25]}))
    )
    spec = chart.to_spec()
    y_enc = spec.get("encoding", {}).get("y", {})
    scale = y_enc.get("scale", {})
    assert "zero" not in scale or scale.get("zero") is not True
    assert scale.get("domain") == [5, 25]


def test_bar_explicit_zero_false():
    """User can opt out of zero-anchoring."""
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10, 20, 15]})
    chart = (
        fm.Chart(df)
        .mark_bar()
        .encode(x="cat", y=fm.Y("val", scale={"zero": False}))
    )
    spec = chart.to_spec()
    y_enc = spec.get("encoding", {}).get("y", {})
    scale = y_enc.get("scale", {})
    assert scale.get("zero") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_bar_zero.py -v`
Expected: FAIL on `test_bar_default_zero_anchor` — `scale` has no `zero` key

- [ ] **Step 3: Implement in `_resolve_pending`**

In `src/ferrum/chart.py`, at the end of `_resolve_pending` (just before `return new` on line ~413), add the bar zero-anchor injection. Insert this block after the `remap` handling (line ~412) and before the final `return new`:

```python
        # Bar y-axis zero-anchor: inject scale.zero=True when mark is "bar"
        # and the y-encoding has no explicit scale domain or zero override.
        if new._mark == "bar":
            y_enc = new._encoding.get("y")
            if y_enc is not None:
                from ferrum.encoding import Y
                if isinstance(y_enc, ChannelBase):
                    scale = y_enc._kwargs.get("scale")
                    if scale is None:
                        new._encoding["y"] = Y(
                            y_enc.field,
                            **{**y_enc._kwargs, "scale": {"zero": True}},
                        )
                    elif isinstance(scale, dict) and "domain" not in scale and "zero" not in scale:
                        new._encoding["y"] = Y(
                            y_enc.field,
                            **{**y_enc._kwargs, "scale": {**scale, "zero": True}},
                        )
                elif isinstance(y_enc, str):
                    from ferrum.encoding import Y
                    new._encoding["y"] = Y(y_enc, scale={"zero": True})
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_bar_zero.py -v`
Expected: all 3 pass

Run: `uv run pytest -x -q`
Expected: all pass (no regressions)

- [ ] **Step 5: Commit**

```
git add src/ferrum/chart.py tests/test_bar_zero.py
git commit -m "feat(defaults): bar charts anchor y-axis at zero by default"
```

---

## Task 4: B — Human-Readable Axis Labels + Chart Titles

**Files:**
- Modify: `src/ferrum/_diagnostics/charts.py` (~25 builder functions)
- Modify: `src/ferrum/marks/statistical.py` (histogram y-axis label)
- Test: `uv run pytest`

This task applies a single mechanical pattern across all builder functions. For each builder: wrap bare string field references in `X(field, title=...)` / `Y(field, title=...)` channel objects, and add `.properties(title=ferrum.Title("..."))` where missing.

**Required import at top of `_diagnostics/charts.py`** (if not already present):
```python
from ferrum.encoding import X, Y
```

The import is already used in ~3 builders; make sure it's at module level, not repeated per-function.

### Label table

Apply this table row by row. Each row names the builder function and the exact `title=` strings to use. "—" means no change. "(already)" means the builder already has a title.

| Builder function | x-title | y-title | Chart title |
|---|---|---|---|
| `_roc_chart_from_source` | `"False Positive Rate"` | `"True Positive Rate"` | `"ROC Curve"` (replace `"ROC"`) |
| `_pr_chart_from_source` | `"Recall"` | `"Precision"` | `"Precision–Recall Curve"` (replace `"Precision–Recall"`) |
| `_confusion_chart_from_source` | `"Predicted label"` | `"True label"` | `"Confusion Matrix"` |
| `_calibration_chart_from_source` | `"Mean predicted probability"` | `"Fraction of positives"` | `"Calibration Curve"` (replace `"Calibration"`) |
| `_learning_curve_chart_from_source` | `"Training instances"` | `"Score"` | `"Learning Curve"` (replace `"Learning curve"`) |
| `_residuals_chart_from_source` | — (panels handle titles) | — | (already) |
| `_residuals_panel` | See panel sub-table below | See panel sub-table below | Add per-panel title |
| `_prediction_error_chart_from_source` | `"Predicted value"` | `"True value"` | `"Prediction Error"` |
| `_pdp_chart_from_source` | — (feature name is correct) | `"Partial dependence"` | `"Partial Dependence"` |
| `_validation_curve_chart_from_source` | — (param name is correct) | `"Score"` | (already, keep title-case: `"Validation Curve — {param}"`) |
| `_cv_scores_chart_from_source` | `"Fold"` or `"Split"` (depends on kind) | `"Score"` | `"Cross-Validation Scores"` |
| `_alpha_selection_chart_from_source` | — (alpha or param name) | `"Score"` | `"Alpha Selection"` |
| `_class_prediction_error_chart_from_source` | `"Actual class"` | `"Number of predictions"` | `"Class Prediction Error"` |
| `_discrimination_threshold_chart_from_source` | `"Discrimination threshold"` | `"Score"` | `"Discrimination Threshold"` (replace `"Discrimination threshold"`) |
| `_gain_chart_from_source` | `"Percentage of sample"` | `"Gain"` | `"Cumulative Gains Curve"` (replace `"Cumulative gain"`) |
| `_lift_chart_from_source` | `"Percentage of sample"` | `"Lift"` | `"Lift Curve"` (replace `"Lift"`) |
| `_importance_chart_from_source` | — | — | (already) |
| `_pca_scree_chart_from_source` | `"Component"` | `"Explained variance"` | `"PCA Explained Variance"` |
| `_intercluster_distance_chart_from_source` | title based on method (e.g. `"MDS1"`, `"PC1"`) | same (e.g. `"MDS2"`, `"PC2"`) | `"Intercluster Distance Map"` |
| `_decision_boundary_chart_from_source` | — (feature names) | — (feature names) | `"Decision Boundary"` |
| `_parallel_coords_chart_from_dataframe` | — | — | `"Parallel Coordinates"` |
| `_cluster_diagnostics_chart` | `"k"` | `"Distortion score"` / `"Silhouette score"` | `"Cluster Diagnostics"` |
| `_shap_beeswarm_chart_from_source` | `"SHAP value"` | `"Feature"` | `"SHAP Summary"` |
| `_rank1d_chart_from_dataframe` | — | — | — |
| `_rank2d_chart_from_dataframe` | — | — | `"Feature Correlation"` |

### Residuals panel sub-table

In `_residuals_panel`, add per-panel titles using `.properties(title=...)`:

| Panel name | x-title | y-title | Panel title |
|---|---|---|---|
| `residuals_vs_fitted` | `"Fitted values"` | `"Studentized residual"` | `"Residuals vs Fitted"` |
| `qq` | `"Theoretical quantiles"` | `"Sample quantiles"` | `"Normal Q–Q"` |
| `scale_location` | `"Fitted values"` | `"√|Studentized residual|"` | `"Scale–Location"` |
| `residuals_vs_leverage` | `"Leverage"` | `"Studentized residual"` | `"Residuals vs Leverage"` |

### Histogram y-axis label

In `src/ferrum/marks/statistical.py`, `desugar_histogram` (line ~177): the encoding remap currently returns `{"y": "count"}` or `{"y": "density"}`. Wrap the y remap value in a `Y` channel with `title=`:

```python
# In the encoding_remap returned by desugar_histogram:
# Before
y_field_name = "density" if density else "count"
# ... remap = {"x": "bin_start", "x2": "bin_end", "y": y_field_name}

# After — add title to the remap y value
y_title = "Density" if density else "Count"
```

Note: since `desugar_histogram` returns a 3-tuple `(mark, transforms, remap)` where remap values are strings, and `_resolve_pending` wraps them in `Y(remap["y"], type="Q")`, the title needs to be threaded through differently. The cleanest approach: set `remap["y_title"] = y_title` and have `_resolve_pending` check for `remap.get("y_title")` when constructing the `Y` object. Alternatively, just set the title in the builder functions that call `mark_histogram` (the grammar-level `Chart.mark_histogram` path).

- [ ] **Step 1: Add module-level import**

Ensure `from ferrum.encoding import X, Y` is at the module level of `_diagnostics/charts.py` (not inside individual functions). It's already imported in ~3 functions; move to module top if needed.

- [ ] **Step 2: Apply the label table to each builder**

For each builder in the table above, apply the pattern. Example for `_roc_chart_from_source`:

```python
# Before (line ~393):
chart = ferrum.Chart(df).mark_roc(...)

# After — wrap the .encode() call with titled channels:
# The mark_roc desugar handles encoding internally, so add titles via
# chart-level .encode() override after the mark call:
chart = chart.encode(
    x=X("fpr", title="False Positive Rate"),
    y=Y("tpr", title="True Positive Rate"),
)

# Before (line ~408):
chart = chart.properties(title=ferrum.Title("ROC", subtitle=subtitle))
# After:
chart = chart.properties(title=ferrum.Title("ROC Curve", subtitle=subtitle))

# Before (line ~405):
title=ferrum.Title(f"ROC — AUC {auc_value:.3f}", subtitle=subtitle),
# After:
title=ferrum.Title(f"ROC Curve — AUC {auc_value:.3f}", subtitle=subtitle),
```

For diagnostic marks that handle encoding internally (e.g., `mark_roc`, `mark_residuals`), add a `.encode()` call AFTER the mark call to overlay axis titles. This works because ferrum's `.encode()` merges with existing encodings — the titled `X`/`Y` objects override the field's display title without changing the field binding.

For builders that already use explicit `X()`/`Y()` objects (like `_intercluster_distance_chart_from_source`), just add `title=` to the existing channel constructors.

- [ ] **Step 3: Apply residuals panel titles**

In `_residuals_panel`, after each panel's chart construction, add `.properties(title=ferrum.Title("Panel Title"))`. Example for the `residuals_vs_fitted` panel:

```python
if name == "residuals_vs_fitted":
    ...
    chart = chart.encode(
        x=X("y_pred", title="Fitted values"),
        y=Y(y_col, title="Studentized residual"),
    )
    chart = chart.properties(title=ferrum.Title("Residuals vs Fitted"))
    return _overlay_metrics_corner(chart)
```

- [ ] **Step 4: Handle histogram y-axis label**

In `src/ferrum/marks/statistical.py`, the histogram desugar's remap doesn't easily support titles. Instead, apply the title at the grammar level. In `_diagnostics/charts.py`, any builder that calls `.mark_histogram()` should follow with:

```python
chart = chart.encode(y=Y("count", title="Count"))  # or "Density" if density=True
```

If no diagnostic builder calls `mark_histogram` directly (it's a grammar-level mark), then the title for row 08 needs to be set in the gallery audit panel script, not here. Skip this sub-step if histogram is only used at the grammar level.

- [ ] **Step 5: Run tests**

Run: `uv run pytest -x -q`
Expected: all pass

- [ ] **Step 6: Commit**

```
git add src/ferrum/_diagnostics/charts.py src/ferrum/marks/statistical.py
git commit -m "feat(defaults): add human-readable axis labels and titles to all chart builders"
```

---

## Task 5: C1 — Boxplot Improvements

**Files:**
- Modify: `src/ferrum/marks/composite.py:16-129` (`desugar_boxplot`)
- Test: `uv run pytest`

- [ ] **Step 1: Fix y-axis title bug**

The boxplot desugar's layers use computed column names (`lower_whisker`, `q1`, `median`, etc.) as y-field, which become the axis label. Fix by passing `Y` channel objects with `title=val` (the original variable name) instead of bare strings. `_Layer.encoding` already supports `ChannelBase` objects (verified in `chart.py:4116`).

In `desugar_boxplot`, import `Y` (or `X` for horizontal) at the top:

```python
from ferrum.encoding import X, Y
```

Then update the `enc()` helper to thread the title:

```python
    def enc(y_col, y2_col=None, *, title=None):
        if horizontal:
            d: dict = {"x": X(y_col, title=title) if title else y_col, "y": cat}
            if y2_col:
                d["x2"] = y2_col
        else:
            d = {"x": cat, "y": Y(y_col, title=title) if title else y_col}
            if y2_col:
                d["y2"] = y2_col
        return d
```

Pass `title=val` on the layers that define the visible y-axis:

```python
    layers = [
        _Layer(mark="rule", encoding=enc("lower_whisker", "upper_whisker"), data_source="box"),
        _Layer(
            mark="rect", encoding=enc("q1", "q3", title=val), mark_kwargs={"width": band}, data_source="box"
        ),
        ...
    ]
```

- [ ] **Step 2: Add whisker caps (T-bars)**

Add two `mark_tick` layers at `lower_whisker` and `upper_whisker` positions with a small `band_size` for the cap width:

```python
    layers = [
        # Whisker rule
        _Layer(mark="rule", encoding=enc("lower_whisker", "upper_whisker"), data_source="box"),
        # Whisker caps
        _Layer(
            mark="tick",
            encoding=enc("lower_whisker"),
            mark_kwargs={"band_size": 0.3},
            data_source="box",
        ),
        _Layer(
            mark="tick",
            encoding=enc("upper_whisker"),
            mark_kwargs={"band_size": 0.3},
            data_source="box",
        ),
        # IQR box
        _Layer(
            mark="rect", encoding=enc("q1", "q3", title=val), mark_kwargs={"width": band}, data_source="box"
        ),
        # Median line — dark, visually distinct
        _Layer(
            mark="tick",
            encoding=enc("median"),
            mark_kwargs={"band_size": band, "stroke": "#222222", "stroke_width": 2},
            data_source="box",
        ),
    ]
```

- [ ] **Step 3: Make outlier markers unfilled**

Change the outlier layer's mark_kwargs:

```python
    if outliers:
        layers.append(
            _Layer(
                mark="point",
                encoding=enc(val),
                mark_kwargs={"filled": False},
                data_source="outliers",
            )
        )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest -x -q`
Expected: all pass

- [ ] **Step 5: Commit**

```
git add src/ferrum/marks/composite.py
git commit -m "fix(boxplot): fix y-axis label, add whisker caps, darken median, unfill outliers"
```

---

## Task 6: C2 — Heatmap / Confusion / Correlation Colormaps

**Files:**
- Modify: `src/ferrum/_diagnostics/charts.py` (confusion, rank2d builders)
- Test: `uv run pytest`

- [ ] **Step 1: Find the confusion chart builder and update colormap**

In `_confusion_chart_from_source` (~line 831), find where the heatmap/raster mark is called and change the default colormap to `"blues"`:

```python
# Add cmap="blues" to the mark call or encoding
chart = ferrum.Chart(df).mark_confusion(cmap="blues")
```

If the `mark_confusion` desugar accepts a `cmap` kwarg, pass it there. If not, set it via the color encoding's `scheme=` parameter. Also add a title:

```python
chart = chart.properties(title=ferrum.Title("Confusion Matrix"))
chart = chart.encode(
    x=X("predicted", title="Predicted label"),
    y=Y("actual", title="True label"),
)
```

- [ ] **Step 2: Update rank2d / correlation chart colormap**

In `_rank2d_chart_from_dataframe` (~line 1756), change the default colormap to `"rdbu"` diverging, centered at 0:

```python
chart = ferrum.Chart(df).mark_rank2d(cmap="rdbu")
# Add title
chart = chart.properties(title=ferrum.Title("Feature Correlation"))
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest -x -q`
Expected: all pass

- [ ] **Step 4: Commit**

```
git add src/ferrum/_diagnostics/charts.py
git commit -m "feat(defaults): use blues for confusion, rdbu for correlation colormaps"
```

---

## Task 7: C3 — Markers on Discrete-Data Line Charts

**Files:**
- Modify: `src/ferrum/_diagnostics/charts.py` (calibration, learning, validation, alpha, scree, cluster diagnostics builders)
- Test: `uv run pytest`

Add a `mark_point(size=40, filled=True)` overlay layer at discrete data positions. The overlay shares the same data and encoding as the line layer.

The pattern for each builder: after constructing the line chart, add a point overlay via `.layer()`:

```python
from ferrum.layer import Layer

# After the main chart construction, add point overlay:
chart = chart.layer(
    Layer(mark="point", encoding={"x": x_field, "y": y_field}, mark_kwargs={"size": 40, "filled": True})
)
```

- [ ] **Step 1: Add markers to calibration chart**

In `_calibration_chart_from_source` (~line 678), after the main chart is built, add point overlay. The calibration mark's x/y are `"mean_predicted_prob"` / `"fraction_of_positives"` — add a point layer matching those fields.

- [ ] **Step 2: Add markers to learning curve chart**

In `_learning_curve_chart_from_source` (~line 1454), add point overlay at `"train_size"` / `"mean_score"`.

- [ ] **Step 3: Add markers to validation curve chart**

In `_validation_curve_chart_from_source` (~line 1495), add point overlay at `"param_value"` / `"mean_score"`.

- [ ] **Step 4: Add markers to alpha selection chart**

In `_alpha_selection_chart_from_source` (~line 1574), add point overlay at `"alpha"` / `"mean_score"`.

- [ ] **Step 5: Add markers to PCA scree chart**

In `_pca_scree_chart_from_source` (~line 1625), add point overlay on the cumulative line.

- [ ] **Step 6: Add markers to cluster diagnostics chart**

In `_cluster_diagnostics_chart` (~line 2127), add point overlay at `"k"` / `"inertia"` and `"k"` / `"silhouette"`.

- [ ] **Step 7: Run tests**

Run: `uv run pytest -x -q`
Expected: all pass

- [ ] **Step 8: Commit**

```
git add src/ferrum/_diagnostics/charts.py
git commit -m "feat(defaults): add data-point markers on discrete-data line charts"
```

---

## Task 8: C4 — Legend Label Formatting

**Files:**
- Modify: `src/ferrum/_diagnostics/charts.py` (ROC, PR, learning, validation, gain, lift builders)
- Test: `uv run pytest`

The pattern: rename the color-field values in the DataFrame BEFORE passing to the chart, so the legend entries show human-readable labels.

- [ ] **Step 1: ROC — AUC in legend labels**

In `_roc_chart_from_source`, when `annotate_auc=True`, compute per-class AUC values and rename the color-field values:

```python
import numpy as np

# After computing df from source.roc_curve():
if annotate_auc and color_field is not None:
    groups = df[color_field].unique().to_list()
    rename_map = {}
    for g in groups:
        mask = df.filter(pl.col(color_field) == g)
        fpr = np.asarray(mask["fpr"].to_list(), dtype=float)
        tpr = np.asarray(mask["tpr"].to_list(), dtype=float)
        auc = _trapezoid_auc(fpr, tpr)
        rename_map[str(g)] = f"{g} (AUC = {auc:.3f})"
    df = df.with_columns(
        pl.col(color_field).cast(pl.Utf8).replace(rename_map).alias(color_field)
    )
```

- [ ] **Step 2: PR — AP in legend labels**

Same pattern in `_pr_chart_from_source` using average precision.

- [ ] **Step 3: Learning/validation — rename split labels**

In `_learning_curve_chart_from_source` and `_validation_curve_chart_from_source`, rename the `"split"` column values:

```python
df = df.with_columns(
    pl.col("split").replace({"train": "Training Score", "test": "Cross-Validation Score"})
)
```

- [ ] **Step 4: Gain/lift — rename class labels**

In `_gain_chart_from_source` and `_lift_chart_from_source`, rename bare `0`/`1` to `Class 0`/`Class 1` and `baseline` to `Baseline`:

```python
df = df.with_columns(
    pl.col(color_field).cast(pl.Utf8).replace({
        "0": "Class 0", "1": "Class 1", "baseline": "Baseline",
    })
)
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest -x -q`
Expected: all pass

- [ ] **Step 6: Commit**

```
git add src/ferrum/_diagnostics/charts.py
git commit -m "feat(defaults): format legend labels with metrics and human-readable names"
```

---

## Task 9: C5 — CV Scores Redesign (Per-Fold Bars)

**Files:**
- Modify: `src/ferrum/_diagnostics/charts.py:1549-1571` (`_cv_scores_chart_from_source`)
- Modify: `src/ferrum/_diagnostics/source.py` (if `cv_scores()` needs output shape change)
- Test: `tests/test_cv_scores_defaults.py` (new)

- [ ] **Step 1: Check `ModelSource.cv_scores()` output shape**

Read `src/ferrum/_diagnostics/source.py` and find the `cv_scores` method. Verify whether it already returns per-fold rows (one row per fold × split) or aggregated rows. If it already returns per-fold, skip the source change.

- [ ] **Step 2: Write failing test**

Create `tests/test_cv_scores_defaults.py`:

```python
"""CV scores chart defaults — per-fold bars + mean reference line."""
import polars as pl
import ferrum as fm


def test_cv_scores_has_title():
    """CV scores chart should have a title."""
    # Use a mock source or build chart from known data
    df = pl.DataFrame({
        "fold": [1, 2, 3, 4, 5],
        "score": [0.85, 0.87, 0.83, 0.86, 0.84],
        "split": ["test"] * 5,
    })
    chart = fm.Chart(df).mark_bar().encode(x="fold", y="score")
    chart = chart.properties(title=fm.Title("Cross-Validation Scores"))
    spec = chart.to_spec()
    assert spec.get("title", {}).get("text") == "Cross-Validation Scores"
```

- [ ] **Step 3: Update `_cv_scores_chart_from_source`**

Redesign to show per-fold bars instead of aggregated train/test bars. Add a dashed mean-score horizontal reference line:

```python
def _cv_scores_chart_from_source(
    source, *, cv=5, scoring=None, kind="bar", split="both",
    subtitle=None, theme=None,
):
    import ferrum
    from ferrum.encoding import X, Y

    df = source.cv_scores(cv=cv, scoring=scoring)
    if split != "both":
        df = df.filter(pl.col("split") == split)

    # Per-fold bars
    chart = (
        ferrum.Chart(df)
        .mark_bar()
        .encode(
            x=X("fold", title="Fold"),
            y=Y("score", title="Score"),
        )
    )

    # Mean score reference line
    mean_score = float(df["score"].mean())
    chart = chart + ferrum.annotate_hline(
        mean_score,
        stroke="#AAAAAA",
        stroke_dash=[4, 4],
    )

    chart = chart.properties(
        title=ferrum.Title("Cross-Validation Scores", subtitle=subtitle),
    )
    if theme is not None:
        chart = chart.theme(theme)
    return chart
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest -x -q`
Expected: all pass

- [ ] **Step 5: Commit**

```
git add src/ferrum/_diagnostics/charts.py tests/test_cv_scores_defaults.py
git commit -m "feat(defaults): redesign CV scores to per-fold bars with mean reference line"
```

---

## Task 10: C6 + C7 — Class Prediction Error Axis Fix + Gain/Lift Baseline Labels

**Files:**
- Modify: `src/ferrum/_diagnostics/charts.py` (class prediction error, gain, lift builders)
- Test: `uv run pytest`

- [ ] **Step 1: Verify class prediction error axis assignment**

Read `_class_prediction_error_chart_from_source` (~line 851). Check whether actual class is on x-axis and predicted class is the stacked bar color. If inverted, swap the encoding:

```python
# Standard convention:
# x-axis = actual class
# color/stack = predicted class
chart = chart.encode(
    x=X("actual", title="Actual class"),
    y=Y("count", title="Number of predictions"),
    color="predicted",
)
```

Add title:
```python
chart = chart.properties(title=ferrum.Title("Class Prediction Error"))
```

- [ ] **Step 2: Update gain/lift baseline labels**

In `_gain_chart_from_source` and `_lift_chart_from_source`, rename the baseline row's color-field value to `"Baseline"`:

```python
# After loading df from source:
df = df.with_columns(
    pl.col(color_field).cast(pl.Utf8).replace({"baseline": "Baseline"})
)
```

This overlaps with Task 8 (C4) — if Task 8 already renames `"baseline"` → `"Baseline"`, verify consistency.

- [ ] **Step 3: Run tests**

Run: `uv run pytest -x -q`
Expected: all pass

- [ ] **Step 4: Commit**

```
git add src/ferrum/_diagnostics/charts.py
git commit -m "fix(defaults): correct class prediction error axes, format gain/lift baseline labels"
```

---

## Task 11: C8 — SHAP Summary Improvements

**Files:**
- Modify: `src/ferrum/_diagnostics/charts.py:1090` (`_shap_beeswarm_chart_from_source`)
- Modify: `src/ferrum/marks/diagnostic.py` (SHAP desugar, if applicable)
- Test: `uv run pytest`

- [ ] **Step 1: Sort features by mean |SHAP value| descending**

In `_shap_beeswarm_chart_from_source`, pre-sort the DataFrame:

```python
# After loading df from source:
feature_order = (
    df.group_by("feature")
    .agg(pl.col("shap_value").abs().mean().alias("mean_abs_shap"))
    .sort("mean_abs_shap", descending=True)
    ["feature"].to_list()
)
df = df.with_columns(
    pl.col("feature").cast(pl.Categorical).cat.set_ordering("physical")
)
# Or use sort= on the encoding:
chart = chart.encode(
    y=Y("feature", title="Feature", sort=feature_order),
    x=X("shap_value", title="SHAP value"),
)
```

- [ ] **Step 2: Set diverging colormap**

Pass `cmap="rdbu"` to the SHAP beeswarm mark (per the diverging colormap role).

- [ ] **Step 3: Add zero-reference vertical line**

```python
chart = chart + ferrum.annotate_vline(0, stroke="#AAAAAA", stroke_dash=[4, 4])
```

- [ ] **Step 4: Add title**

```python
chart = chart.properties(title=ferrum.Title("SHAP Summary"))
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest -x -q`
Expected: all pass

- [ ] **Step 6: Commit**

```
git add src/ferrum/_diagnostics/charts.py src/ferrum/marks/diagnostic.py
git commit -m "feat(defaults): sort SHAP features, diverging colormap, zero-ref line, title"
```

---

## Task 12: C9 + C10 — Pairplot and Clustermap Defaults

**Files:**
- Modify: `src/ferrum/figure/statistical.py` (pairplot builder)
- Modify: `src/ferrum/_diagnostics/charts.py` or `src/ferrum/figure/` (clustermap builder)
- Test: `uv run pytest`

- [ ] **Step 1: Find pairplot builder**

Locate the pairplot implementation. Run: `grep -rn 'def pairplot\|def _pairplot' src/ferrum/`

- [ ] **Step 2: Update pairplot defaults**

- Change diagonal panels from histogram to KDE: pass `diagonal="kde"` or equivalent
- Set tighter layout: reduce `row_padding` / `column_padding`
- Ensure single shared legend for entire grid

- [ ] **Step 3: Find clustermap builder**

Locate the clustermap implementation. Run: `grep -rn 'def clustermap\|def _clustermap' src/ferrum/`

- [ ] **Step 4: Update clustermap defaults**

- Change default colormap to `"magma"` (dense heatmap role)
- Remove internal axis labels (`_row_id`, `column`) if present
- Limit row labels when count > 50 (truncate or hide)

- [ ] **Step 5: Run tests**

Run: `uv run pytest -x -q`
Expected: all pass

- [ ] **Step 6: Commit**

```
git add src/ferrum/figure/statistical.py src/ferrum/_diagnostics/charts.py
git commit -m "feat(defaults): pairplot KDE diagonal + clustermap magma colormap"
```

---

## Task 13: C11 — Miscellaneous Single-Item Fixes

**Files:**
- Modify: `src/ferrum/_diagnostics/charts.py` (multiple builders)
- Test: `uv run pytest`

Each sub-step is a small, independent fix applied to a specific builder.

- [ ] **Step 1: Regression scatter — R² annotation (row 10)**

In the regression/lmplot builder, add R² annotation using the existing `_inject_metrics_corner` + `_overlay_metrics_corner` pattern (same as residuals).

- [ ] **Step 2: Validation curve — auto-detect log-scale (row 14)**

Verify the validation curve builder already has `log_scale="auto"` logic (it does — line 1518-1523). If the auto-detect threshold is correct (ratio > 100), no change needed. Mark as done.

- [ ] **Step 3: Alpha selection — best-alpha annotation + auto log-scale (row 16)**

Verify `_alpha_selection_chart_from_source` already has `log_scale=True` default. Add best-alpha annotation: find the alpha with highest mean_score and add a dashed vertical reference line + text label:

```python
best_idx = df["mean_score"].arg_max()
best_alpha = float(df["alpha"][best_idx])
best_score = float(df["mean_score"][best_idx])
chart = chart + ferrum.annotate_vline(
    best_alpha, stroke="#AAAAAA", stroke_dash=[4, 4],
)
chart = chart + ferrum.annotate_text(
    best_alpha, best_score,
    f"α = {best_alpha:.3f}", dx=8, dy=-8,
)
chart = chart.properties(title=ferrum.Title("Alpha Selection"))
```

- [ ] **Step 4: Intercluster — embedding axis labels (row 21)**

In `_intercluster_distance_chart_from_source`, update X/Y titles to include the method:

```python
method_label = method.upper()  # "MDS" or "PCA"
chart = chart.encode(
    x=X("x", title=f"{method_label}1", scale=...),
    y=Y("y", title=f"{method_label}2", scale=...),
)
chart = chart.properties(title=ferrum.Title("Intercluster Distance Map"))
```

- [ ] **Step 5: Parallel coordinates — lower alpha + title (row 23)**

In `_parallel_coords_chart_from_dataframe`, set line opacity to 0.3 and add title:

```python
# Add opacity to the mark or encoding
chart = chart.mark_line(opacity=0.3)  # or mark_kwargs
n_features = len(features)
chart = chart.properties(title=ferrum.Title(f"Parallel Coordinates for {n_features} Features"))
```

- [ ] **Step 6: PCA scree — threshold reference line (row 24)**

Verify `_pca_scree_chart_from_source` already has `threshold_line` parameter (it does — line 1639). Confirm it renders. Add title:

```python
chart = chart.properties(title=ferrum.Title("PCA Explained Variance"))
```

- [ ] **Step 7: Decision boundary defaults (row 18)**

In `_decision_boundary_chart_from_source`, add title and dark scatter outlines:

```python
chart = chart.properties(title=ferrum.Title("Decision Boundary"))
# Add dark outlines to scatter points via mark_kwargs:
# mark_point(stroke="#333333", stroke_width=0.5)
```

Probability gradient shading and per-class palettes depend on the existing decision boundary implementation — read the builder, assess feasibility, and apply what's possible without new Rust renderer logic.

- [ ] **Step 8: Discrimination threshold defaults (row 19)**

In `_discrimination_threshold_chart_from_source`, add vertical dashed line at optimal threshold with value label:

```python
# Find optimal threshold (max F1 or user-specified metric)
# Add annotation
chart = chart + ferrum.annotate_vline(
    optimal_threshold, stroke="#AAAAAA", stroke_dash=[4, 4],
)
chart = chart + ferrum.annotate_text(
    optimal_threshold, max_score,
    f"threshold = {optimal_threshold:.3f}", dx=8, dy=-8,
)
```

- [ ] **Step 9: Residplot — larger point size (row 27)**

Find the residplot builder (may be `_residuals_chart_from_source` with `panels="single"` or a separate function). Increase default point size via mark_kwargs:

```python
chart = chart.mark_residuals(kind=kind, cook_threshold=cook_threshold, size=50)
```

- [ ] **Step 10: Jointplot — tighter margins (row 30)**

Find the jointplot builder. Reduce margins between scatter and marginal histograms by adjusting `row_padding` / `column_padding` in the layout or theme override for that chart.

- [ ] **Step 11: Cluster diagnostics — elbow detection + title (row 31)**

In `_cluster_diagnostics_chart`, add simple elbow detection (maximum second derivative of inertia) and a dashed vertical annotation:

```python
import numpy as np

inertias = np.array([r["inertia"] for r in rows])
if len(inertias) >= 3:
    second_diff = np.diff(inertias, n=2)
    elbow_idx = int(np.argmax(second_diff)) + 1  # +1 for diff offset
    elbow_k = int(ks[elbow_idx])
    elbow_score = float(inertias[elbow_idx])
    elbow = elbow + ferrum.annotate_vline(
        elbow_k, stroke="#AAAAAA", stroke_dash=[4, 4],
    )
    elbow = elbow + ferrum.annotate_text(
        elbow_k, elbow_score,
        f"elbow at k={elbow_k}", dx=8, dy=-8,
    )

# Add titles to each panel
elbow = elbow.properties(title=ferrum.Title("Distortion Score Elbow"))
sil = sil.properties(title=ferrum.Title("Silhouette Score"))
```

- [ ] **Step 12: Run tests**

Run: `uv run pytest -x -q`
Expected: all pass

- [ ] **Step 13: Commit**

```
git add src/ferrum/_diagnostics/charts.py
git commit -m "feat(defaults): misc per-chart fixes — R² annotation, elbow detection, alpha annotation, titles"
```

---

## Task 14: Golden Regeneration + Visual Inspection

**Files:**
- Modify: `tests/goldens/` (regenerated SVGs)
- Test: visual PNG inspection via `scripts/snapshot-goldens.py`

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest -x -q`
Expected: some golden tests may fail due to changed defaults. This is expected.

- [ ] **Step 2: Identify failing golden tests**

Run: `uv run pytest -v 2>&1 | grep FAIL`
List all failing golden tests.

- [ ] **Step 3: Regenerate affected goldens**

For each failing golden, regenerate:
```bash
uv run python scripts/snapshot-goldens.py <golden_name>
```

Or regenerate all:
```bash
uv run python scripts/snapshot-goldens.py
```

- [ ] **Step 4: Visual inspection**

For each regenerated golden, read the PNG and confirm:
- Axis labels are human-readable (not raw column names)
- Chart title appears and is title-case
- Reference lines are light gray, dashed
- CI bands have no outline, are translucent
- Bar charts start at y=0
- Boxplots have whisker caps, dark median, unfilled outliers

- [ ] **Step 5: Commit regenerated goldens**

```
git add tests/goldens/
git commit -m "chore: regenerate goldens after defaults remediation"
```

---

## Task 15: Final Validation

- [ ] **Step 1: Full test suite**

Run: `uv run pytest -x -q`
Run: `DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test`
Expected: all pass

- [ ] **Step 2: Ruff lint**

Run: `uv run ruff check src/ferrum/ --select E,W,F`
Expected: clean (or pre-existing only)

- [ ] **Step 3: Summary**

Verify all 30 rows from `gallery_feedback.md` are addressed. Row 07 (Feature Importance) is the only intentionally unchanged row.
