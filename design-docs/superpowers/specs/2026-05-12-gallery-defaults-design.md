# Gallery Defaults Remediation — Design Spec

**Date:** 2026-05-12
**Source:** `gallery_feedback.md` (collected via interactive gallery audit walkthrough)
**Branch:** TBD (feature branch from main)

---

## Problem

Ferrum's default chart output lacks information and visual quality that competitors (sklearn, seaborn, yellowbrick, scikit-plot, SHAP) ship out of the box. A side-by-side gallery audit identified gaps in 30 of 31 reviewed rows — primarily raw column names instead of human-readable axis labels, missing chart titles, and per-chart-type visual defaults that don't match domain conventions.

## Principle

**Strong defaults, always overridable.** Every change in this spec establishes a new default that fires automatically. Every default can be overridden by the user via the existing API surface (explicit `title=` on encoding channels, explicit `scale=` domains, explicit mark kwargs). No new API is introduced — this is purely a defaults-quality pass over existing code.

## Cohesion Conventions

These cross-cutting decisions ensure the 30 per-chart fixes feel like one intentional design system, not a patchwork:

### Reference lines — theme-aware

Add `reference_line_color` and `reference_line_dash` as theme keys. Default values: `#AAAAAA` / `[4, 4]`. All reference/baseline lines (diagonals, zero-lines, threshold markers) read from the theme so dark/publication themes auto-adjust. Mark desugars read these keys; user can override per-mark via `stroke=` / `stroke_dash=`.

### Metric format — `.3f` everywhere

AUC, AP, Brier, R², RMSE, MAE — all formatted to 3 decimal places. Consistent across legend labels, corner annotations, and threshold callouts.

### Colormap roles — three semantic slots

Rather than picking colormaps per-chart, define three roles:

| Role | Default | Use case |
|---|---|---|
| **Sequential** | `"blues"` | Counts, magnitudes (confusion matrix, single-variable heatmaps) |
| **Diverging** | `"rdbu"` | Centered-at-zero (correlation matrices, SHAP values) |
| **Dense heatmap** | `"magma"` | Many-row heatmaps needing perceptual uniformity (clustermap 100+ rows) |

Each builder picks its role, not its colormap. User overrides via `cmap=` kwarg.

### Discrete-data markers — one style

All overlay markers on discrete-data line charts (calibration, learning, validation, alpha, scree, cluster diagnostics) use: `mark_point(size=40, filled=True)`, inheriting line color from the color encoding. Filled circles, no special shapes.

### Title capitalization — title case

Every chart title uses title case: "ROC Curve", "Precision–Recall Curve", "Cross-Validation Scores". No sentence case, no ALL CAPS.

### Annotation positioning — three patterns

| Pattern | Examples | Convention |
|---|---|---|
| **Corner metrics** | R², RMSE, MAE | Top-right of plot area. Font size = axis label size. Color = theme `font_color` |
| **Legend-integrated** | AUC, AP | Appended to legend label: `ClassName (AUC = 0.XXX)` |
| **Threshold/elbow** | Optimal threshold, elbow k | Vertical dashed reference line (theme `reference_line_color`) + text label above |

---

## Architecture

Changes live in three tiers, ordered by dependency (Tier A first, since downstream tiers inherit its effects):

| Tier | Scope | Location | Leverage |
|---|---|---|---|
| **A** | Global visual defaults | `marks/composite.py`, `marks/diagnostic.py`, mark desugars | High — every chart inherits |
| **B** | Axis labels + chart titles | `_diagnostics/charts.py` builders, grammar-level marks | Medium — mechanical, ~25 builders |
| **C** | Per-chart-type fixes | Per-mark desugar + per-builder | Varied — each fix is chart-specific |

---

## Tier A — Global Visual Defaults

### A1. CI/Error Band Styling

**Files:** `src/ferrum/marks/composite.py` (ribbon/errorband desugar)

- Lower default ribbon opacity from 0.3 → 0.2
- Set `stroke: "none"` on ribbon layers (currently inherits theme mark_color, which draws a visible outline)
- Affects: learning curve CI, validation curve CI, regression scatter CI, errorband, any ribbon overlay
- **Override:** `mark_errorband(opacity=0.5, stroke="#000")`

### A2. Reference Line Theme Keys

**Files:** `src/ferrum/themes/__init__.py` (add keys), `src/ferrum/themes/builtins.py` (set per-theme defaults), `crates/ferrum-core/src/layout/mod.rs` (add to `ThemeInputs`), `src/ferrum/marks/diagnostic.py` (read from theme)

- Add `reference_line_color` (default `"#AAAAAA"`) and `reference_line_dash` (default `[4, 4]`) to the `Theme` class and `ThemeInputs` Rust struct
- All diagnostic mark desugars read these from the resolved theme instead of hardcoding colors
- Dark themes set `reference_line_color` to a lighter value (e.g. `"#666666"`)
- Currently: most reference lines inherit theme mark_color (blue); some use `"#8a8a8a"` with `[3,3]`. All normalize to theme keys
- **Override:** user can set `stroke=` / `stroke_dash=` on any mark, or set theme-level `reference_line_color`

### A3. Bar Y-Axis Zero-Anchoring

**Files:** Bar mark desugar (Python-side, in `marks/base.py` or `marks/composite.py`)

- When mark type is `bar` and the user has not explicitly set a y-scale domain, inject `scale.zero=True` on the y-encoding
- Visible in `.spec` JSON — user can inspect and override
- **Override:** `.encode(y=Y('col', scale={'zero': False}))` or explicit domain `scale={'domain': [min, max]}`

---

## Tier B — Human-Readable Axis Labels + Chart Titles

### Pattern

Each builder function in `_diagnostics/charts.py` that currently does:
```python
.encode(x="fpr", y="tpr")
```
becomes:
```python
.encode(x=X("fpr", title="False Positive Rate"), y=Y("tpr", title="True Positive Rate"))
```

Builders that already use `X()`/`Y()` objects add `title=`. Builders that already have `.properties(title=...)` keep theirs; those without get one.

Labels are inline per-builder (no shared registry — each chart owns its own labels, avoiding global mutable state).

### Title Convention

- `ferrum.Title("Chart Type")` for charts without model context
- `ferrum.Title("Chart Type for <Model>")` when a model name is available from `ModelSource`
- Subtitles remain optional, passed through from the figure function's `subtitle=` kwarg

### Full Label Mapping

| Row | Chart | x-axis title | y-axis title | Chart title |
|---|---|---|---|---|
| 01 | ROC | False Positive Rate | True Positive Rate | ROC Curve |
| 02 | PR | Recall | Precision | Precision–Recall Curve |
| 03 | Confusion | Predicted label | True label | Confusion Matrix |
| 04 | Calibration | Mean predicted probability | Fraction of positives | Calibration Curve |
| 05 | Learning | Training instances | Score | Learning Curve |
| 06 | Residuals | *(per-panel)* | *(per-panel)* | Residuals *(already set)* |
| 07 | Feature Imp. | *(no change)* | *(no change)* | *(no change)* |
| 08 | Histogram | *(variable name)* | Count | — |
| 09 | Boxplot | *(group or none)* | *(variable — fix bug)* | — |
| 10 | Regression | *(x variable)* | *(y variable)* | — |
| 11 | Correlation | — | — | Correlation Matrix |
| 12 | Bar w/ Error | *(group field)* | *(agg field)* | — |
| 13 | PDP | *(feature name)* | Partial dependence | Partial Dependence |
| 14 | Validation | *(param name)* | Score | Validation Curve — {param} *(already)* |
| 15 | CV Scores | Fold | Score | Cross-Validation Scores |
| 16 | Alpha | *(param name)* | Score | Alpha Selection |
| 17 | Class Pred Err | Actual class | Number of predictions | Class Prediction Error |
| 18 | Decision Bound | *(feature 1)* | *(feature 2)* | Decision Boundary |
| 19 | Disc. Threshold | Discrimination threshold | Score | Discrimination Threshold |
| 20 | Gain | Percentage of sample | Gain | Cumulative Gains Curve |
| 21 | Intercluster | *(embed dim 1)* | *(embed dim 2)* | Intercluster Distance Map |
| 22 | Lift | Percentage of sample | Lift | Lift Curve |
| 23 | Parallel Coords | — | — | Parallel Coordinates |
| 24 | PCA Scree | Component | Explained variance | PCA Explained Variance |
| 25 | Rank | *(same as 11)* | *(same)* | Feature Correlation |
| 26 | SHAP Summary | SHAP value | Feature | SHAP Summary |
| 27 | Residplot | *(variable name)* | Residual | — |
| 28 | Pairplot | *(variable names)* | *(variable names)* | — |
| 29 | Clustermap | — | — | — |
| 30 | Jointplot | *(variable names)* | *(variable names)* | — |
| 31 | Cluster Diag | k | *(per-panel)* | Cluster Diagnostics |

Entries marked *(variable name)* use the user's actual column name — these are grammar-level charts where the field name IS the meaningful label. Entries with — mean no title is needed or is already correct.

### Residuals Panel Titles (Row 06)

| Panel | x-axis | y-axis | Panel title |
|---|---|---|---|
| residuals_vs_fitted | Fitted values | Studentized residual | Residuals vs Fitted |
| qq | Theoretical quantiles | Sample quantiles | Normal Q–Q |
| scale_location | Fitted values | √|Studentized residual| | Scale–Location |
| residuals_vs_leverage | Leverage | Studentized residual | Residuals vs Leverage |

---

## Tier C — Per-Chart-Type Fixes

### C1. Boxplot (row 09)

**Files:** `src/ferrum/marks/composite.py` (boxplot desugar)

- **Bug fix:** y-axis currently shows `lower_whisker` — the desugar must set the y-encoding title to the original variable name
- Add visible median line: dark horizontal `mark_rule` layer inside each box (distinct from the box fill)
- Add whisker caps: short horizontal `mark_tick` at each whisker endpoint (T-bar style)
- Open outlier markers: `filled=False` on the outlier `mark_point` layer

### C2. Heatmap / Confusion / Correlation Colormaps (rows 03, 11, 25)

**Files:** `_diagnostics/charts.py` builders for confusion, correlation, rank2d

- Confusion matrix: default colormap → `"blues"` sequential palette; add colorbar
- Correlation heatmap / rank2d: default colormap → `"rdbu"` diverging, centered at 0; add colorbar
- Fix x-axis label truncation on correlation matrices: rotate labels when count exceeds threshold (existing layout engine collision-avoidance should handle this — verify it fires)

### C3. Markers on Discrete-Data Line Charts (rows 04, 05, 14, 16, 24, 31)

**Files:** `_diagnostics/charts.py` builders for calibration, learning, validation, alpha, scree, cluster diagnostics

- Overlay a `mark_point` layer at each discrete data position
- These are charts where x-axis values are discrete measured points (bin midpoints, fold counts, hyperparameter values) — NOT smooth continuous curves
- Implement as a `point` overlay layer in the builder, using the same data and encoding as the line layer
- Do NOT add markers globally to all line charts — smooth curves (KDE, LOESS, etc.) should remain marker-free

### C4. Legend Label Formatting (rows 01, 02, 04, 05, 20, 22, 24)

**Files:** `_diagnostics/charts.py` builders

- ROC: when `annotate_auc=True`, format legend entries as `ClassName (AUC = 0.XXX)` by renaming the color-field values in the DataFrame before encoding
- PR: same pattern with `AP = 0.XXX`
- Calibration: label diagonal reference line as "Perfectly calibrated" in legend
- Learning/validation curves: rename `train`/`test` → `Training Score` / `Cross-Validation Score` in the split column
- Gain/lift: rename bare `0`/`1` → `Class 0`/`Class 1` in the class column
- PCA scree: legend labels `Cumulative` / `Explained variance` / `95% threshold`

### C5. CV Scores Redesign (row 15)

**Files:** `_diagnostics/charts.py` `_cv_scores_chart_from_source`

- Per-fold bars (fold 1–N) instead of aggregated train/test bars
- Dashed mean-score horizontal reference line with value annotation (using `annotate_hline` pattern)
- Requires changes to `ModelSource.cv_scores()` output shape (one row per fold instead of aggregated)

### C6. Class Prediction Error Axis Fix (row 17)

**Files:** `_diagnostics/charts.py` `_class_prediction_error_chart_from_source`

- Actual class on x-axis, predicted class as stacked bar color (standard convention)
- Verify current axis assignment and swap if inverted

### C7. Gain/Lift Baseline Labels (rows 20, 22)

**Files:** `_diagnostics/charts.py` gain/lift builders

- Baseline row's class/color-field value → `"Baseline"` (currently bare `0` or `baseline`)
- Inherits Tier A reference line color (`#AAAAAA`, dashed)

### C8. SHAP Summary (row 26)

**Files:** `_diagnostics/charts.py` `_shap_beeswarm_chart_from_source`, `marks/diagnostic.py` SHAP desugar

- Sort features by mean |SHAP value| descending (pre-sort the DataFrame)
- Blue→pink diverging colormap (`"rdbu"` or custom SHAP-convention palette)
- Vertical zero-reference line at SHAP value = 0 (via `annotate_vline`)
- Labeled colorbar: title "Feature value", endpoints "Low" / "High"
- Reduce beeswarm vertical jitter (reduce swarm `band_size` or jitter amplitude)

### C9. Pairplot (row 28)

**Files:** `src/ferrum/figure/statistical.py` or equivalent pairplot builder

- KDE on diagonal panels instead of histograms
- Single shared legend for entire grid (currently may duplicate per-panel)
- Shared axes across rows/columns + tighter layout (reduce `row_padding` / `column_padding`)

### C10. Clustermap (row 29)

**Files:** `_diagnostics/charts.py` or `src/ferrum/figure/` clustermap builder

- Row labels on right side (not overlapping dendrogram)
- Fix row label overflow: truncate, sample, or hide when row count exceeds threshold (~50)
- Default colormap → `magma` or `inferno` for dense heatmaps (viridis has low contrast in dense fills)
- Remove internal axis labels (`_row_id` / `column`) — these are implementation artifacts

### C11. Miscellaneous Single-Item Fixes

| Row | Chart | Fix | Location |
|---|---|---|---|
| 08 | Histogram | y-axis label → "Count" | `marks/statistical.py` histogram desugar |
| 10 | Regression | Add R² annotation via `_inject_metrics_corner` pattern | `_diagnostics/charts.py` or regression builder |
| 14 | Validation | Auto-detect log-scale x when values span >2 orders of magnitude | `_diagnostics/charts.py` validation builder |
| 16 | Alpha | Best-alpha annotation + auto-detect log-scale x | `_diagnostics/charts.py` alpha builder |
| 18 | Decision Bound | Probability gradient shading, per-class palettes, dark scatter outlines | `_diagnostics/charts.py` decision boundary builder |
| 19 | Disc. Threshold | CI bands per metric, vertical dashed line at optimum with value | `_diagnostics/charts.py` disc. threshold builder |
| 21 | Intercluster | Larger proportional circles, embedding axis labels (MDS1/MDS2 or PC1/PC2) | `_diagnostics/charts.py` intercluster builder |
| 23 | Parallel Coords | Lower line alpha (~0.3), prominent vertical axis lines, title with feature count | parallel coords builder |
| 24 | PCA Scree | 95% cumulative variance threshold reference line | `_diagnostics/charts.py` scree builder |
| 27 | Residplot | Slightly larger default point size | residplot builder or mark default |
| 30 | Jointplot | Tighter margins between scatter and marginal histograms | jointplot builder layout params |
| 31 | Cluster Diag | Elbow auto-detection + dashed vertical annotation at elbow k | `_diagnostics/charts.py` cluster diagnostics builder |

---

## Testing Strategy

- **Golden SVG updates:** Every chart whose default output changes needs its golden regenerated, rasterized to PNG via `scripts/snapshot-goldens.py`, and visually inspected before committing (per CLAUDE.md hard constraint)
- **Existing tests:** `pytest` must pass after each tier. No test should break — these are additive default changes, not API changes
- **New tests:** Each Tier C fix that adds new behavior (elbow detection, log-scale auto-detect, CV fold bars) gets a targeted test
- **Gallery re-audit:** After all tiers land, re-run `/audit-gallery` on affected rows to verify the gaps closed

## Scope Exclusions

- No new public API is introduced
- Rust changes are limited to plumbing new theme keys (`reference_line_color`, `reference_line_dash`) through `ThemeInputs` — no renderer logic changes
- Row 07 (Feature Importance) is unchanged
- Row 08 optional KDE overlay is not forced by default (noted in feedback as "optional, not forced")
- Row 11 lower-triangle mask is optional — not included as a default change
