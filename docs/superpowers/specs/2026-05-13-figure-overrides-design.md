# Figure-Level Override Passthrough

**Date:** 2026-05-13
**Status:** Design approved, implementation pending

---

## Problem

Ferrum's 24 figure-level functions (`calibration_chart`, `roc_chart`, `residuals_chart`, etc.) expose domain-specific parameters but no general mechanism to customize the underlying grammar components. Users who want to suppress markers, change stroke widths, override axis titles, or add annotation layers must drop to the low-level `Chart` grammar API and rebuild from scratch.

This doesn't scale. Every new customization request becomes a bespoke parameter on a single function (`markers=False`, `stroke_width=`, etc.) rather than a general capability.

## Solution

Add four keyword-only parameters to every figure function that forward user overrides to the grammar layer:

```python
def calibration_chart(
    model_or_source, X=None, y=None, *,
    # domain-specific params (unchanged)
    n_bins=10, strategy="uniform", annotate_brier=True,
    subtitle=None, compare=None, random_state=None,
    # --- override surface ---
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    # existing
    theme=None,
)
```

All 24 builders call one shared `_apply_overrides` utility after assembling the default chart, just before `.theme()`.

---

## Override Parameters

### `mark=` — Sub-layer mark overrides

For composite marks (the majority of figure functions), keys are **named sub-layers**:

```python
# Suppress the point markers on calibration
calibration_chart(model, X, y, mark={'point': False})

# Thicken the line, suppress markers
calibration_chart(model, X, y, mark={'line': {'stroke_width': 3}, 'point': False})
```

`False` suppresses the layer entirely. A dict of mark kwargs merges into that layer's existing kwargs.

For single-mark charts (rare at figure level), the dict is flat mark kwargs:

```python
mark={'opacity': 0.5}
```

The utility distinguishes composite from single-mark by inspecting `chart.layer_names`.

Unknown sub-layer keys raise `ValueError` listing valid names.

### `encode=` — Channel overrides

Channel-by-channel merge. User-supplied channels replace matching defaults; unmentioned channels are untouched.

```python
# Override color scheme
roc_chart(model, X, y, encode={'color': fm.Color('class', scheme='dark2')})

# Custom axis title
calibration_chart(model, X, y, encode={'x': fm.X('mean_predicted', title='P(positive)')})
```

Values are channel objects (`fm.X(...)`, `fm.Color(...)`) or shorthand strings (`"field:Q"`), same as grammar-layer `.encode()`.

### `properties=` — Chart property overrides

Key-by-key merge. Only supplied keys override; the figure's defaults for other keys are preserved.

```python
calibration_chart(model, X, y, properties={'width': 800, 'title': 'My Title'})
```

Supported keys: `width`, `height`, `title`, `description`.

### `layers=` — Extra layers

Appended after the figure's built-in layers. User layers render on top.

```python
from ferrum import Layer
calibration_chart(model, X, y, layers=[
    Layer(mark='rule', encoding={'y': fm.datum(0.5)}, mark_kwargs={'stroke_dash': [4, 4]})
])
```

Post-hoc composition via `+` / `|` / `&` remains fully supported for more complex cases.

---

## Named Sub-Layers

### `_Layer.name` field

Add an optional `name: str | None = None` field to the `_Layer` dataclass. Every composite mark desugar assigns semantic names to its layers.

### Naming convention

Names describe the layer's **role**, not its mark type, to avoid ambiguity when the same mark type appears twice. Where role and mark type coincide, use the mark type.

### Sub-layer catalog

| Composite Mark | Sub-layer names |
|---|---|
| `mark_calibration` | `line`, `reference`, `point` |
| `mark_residuals` | `point`, `reference`, `outlier` |
| `mark_prediction_error` | `point`, `band`, `identity` |
| `mark_roc` | `line`, `reference`, `auc_label` |
| `mark_pr` | `line`, `iso_line`, `iso_label`, `ap_label` |
| `mark_confusion` | `rect`, `label` |
| `mark_importance` | `bar`, `errorbar` |
| `mark_class_prediction_error` | `bar`, `label` |
| `mark_discrimination_threshold` | `line`, `threshold`, `optimum_label` |
| `mark_learning_curve` | `band`, `line` |
| `mark_validation_curve` | `band`, `line` |
| `mark_cv_scores` (box) | inherits boxplot names |
| `mark_cv_scores` (bar) | `bar`, `mean` |
| `mark_cv_scores` (strip) | `point` |
| `mark_pdp` (average) | `line` |
| `mark_pdp` (individual) | `ice` |
| `mark_pdp` (both) | `ice`, `average` |
| `mark_gain` | `line` |
| `mark_lift` | `line` |
| `mark_boxplot` | `whisker`, `lower_cap`, `upper_cap`, `box`, `median`, `outlier` |
| `mark_errorbar` | `rule`, `lower_cap`, `upper_cap` |
| `mark_errorband` | `ribbon`, `lower_border`, `upper_border` |
| `mark_smooth` (with CI) | `ribbon`, `line`, `metrics` |
| `mark_violin` | `body`, then inner-dependent |
| `mark_qq` | `point`, `reference` |
| `mark_hex` | `polygon` |
| `mark_raster` | `image` |
| `mark_swarm` | `point` |
| `mark_contour` (filled) | `polygon` |
| `mark_contour` (outline) | `segment` |
| `mark_boxen` | `depth_1`..`depth_6`, `median`, `outlier` |
| `mark_silhouette` | `rect`, `reference` |
| `mark_pca_scree` | `bar`, `cumulative` |
| `mark_rank1d` | `bar` |
| `mark_rank2d` | `rect`, `label` |
| `mark_parallel_coordinates` | `line` |
| `mark_decision_boundary` | `rect` |
| `mark_intercluster_distance` | `point`, `label` |
| `mark_alpha_selection` | `line`, `best` |
| `mark_shap_beeswarm` | `point`, `reference` |
| `mark_shap_bar` | `bar` |
| `mark_shap_waterfall` | `bar` |

### `chart.layer_names` property

Read-only property on `Chart` that returns `list[str]` of named sub-layers after desugar resolution. Returns `[]` for single-mark charts with no layers. Works on any chart, not just figure-function output.

**Desugar forcing:** Accessing `layer_names` forces resolution of any pending `_PendingMark`. This ensures `_apply_overrides` works regardless of whether the builder happened to trigger desugar earlier (e.g. via `.layer()` or `.encode()`). `_apply_overrides` reads `layer_names` as its first step, so desugar is always resolved before any mutation.

```python
chart = fm.calibration_chart(model, X, y)
chart.layer_names  # → ['line', 'reference', 'point']
```

---

## `_apply_overrides` Utility

Location: `src/ferrum/_overrides.py` (shared, not diagnostics-private — grammar-layer charts may use it too).

```python
def _apply_overrides(
    chart: Chart,
    *,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
) -> Chart:
```

### Application order

`mark` → `encode` → `properties` → `layers`

Mark overrides apply to already-desugared layers. Encode runs after so user channel overrides see the final layer structure. Properties and layers are independent. Theme stays outside this utility (applied by the builder afterward, as today).

### Mark override logic

1. Inspect `chart.layer_names` to determine composite vs. single-mark.
2. **Composite:** Match each key against layer names. `False` → remove layer. Dict → merge into layer's `mark_kwargs`. Unknown key → `ValueError` listing valid names.
3. **Single-mark:** Forward dict as flat mark kwargs.

**Conditional sub-layers:** Many layers are conditionally present depending on domain params (boxplot's `outlier` only when `outliers=True`, errorbar's caps only when `ticks=True`, smooth's `ribbon` only when `ci` is set). Validation uses the **full catalog of possible names for that mark type**, not just the currently-active layers. This means `mark={'outlier': False}` on a boxplot where outliers were already disabled is silently a no-op, not a `ValueError`. This avoids coupling override validation to domain-param state and is the less surprising behavior. Each desugar registers its full name catalog as a class-level or mark-level constant.

### Encode override logic

Forward to `chart.encode(**encode)`. The existing `.encode()` method already does additive merge (new channels override matching ones, unmentioned ones stay).

### Properties override logic

Forward to `chart.properties(**properties)`. Existing method does key-by-key merge.

### Layers override logic

Forward to `chart.layer(*layers)`. Appends after built-in layers.

### Error handling

- Unknown sub-layer name in `mark=` → `ValueError` with valid names listed.
- Invalid mark kwarg for a layer's mark type → existing `TypeError` from `MarkBase` validation.
- Invalid channel name in `encode=` → existing `TypeError` from `.encode()`.

---

## Docstring Convention

Every figure function with a composite mark gets a `Sub-layers` section in its NumPy docstring, between the existing prose and `Parameters`:

```
Sub-layers
----------
line : calibration curve
reference : perfect-calibration diagonal
point : binned probability markers

Use ``mark=`` to override or suppress sub-layers::

    calibration_chart(model, X, y, mark={'point': False})
    calibration_chart(model, X, y, mark={'line': {'stroke_width': 3}})
```

The four new parameters are documented in `Parameters` with cross-references to the sub-layers list.

---

## Scope

### In scope

- Add `name` field to `_Layer` dataclass
- Assign names in all desugar functions (see catalog above)
- Add `layer_names` property to `Chart`
- Implement `_apply_overrides` utility
- Add `mark=`, `encode=`, `properties=`, `layers=` to all 24 figure functions and their `_*_from_source` builders
- Update docstrings with `Sub-layers` sections
- Builder-added layers (e.g. calibration's point layer, residuals metrics corner) get names too

### Out of scope

- Changing desugar architecture or mark resolution order
- Interactive/selection-based conditional overrides
- Override persistence or default-override registry
