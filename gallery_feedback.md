# Gallery Feedback — Remediation Plan

Collected 2026-05-12 via interactive gallery audit walkthrough.

Each row records what the user wants changed in ferrum's default output, informed by side-by-side comparison with sklearn, scikit-plot, yellowbrick, seaborn, and SHAP.

---

## Cross-cutting themes

These recur across many rows and should be treated as global defaults rather than per-chart fixes:

| Theme | Rows affected | Description |
|---|---|---|
| **Human-readable axis labels** | 01–06, 08–10, 13–17, 19–22, 27, 31 | Replace raw column names (`fpr`, `tpr`, `bin_start`, `y_pred`, `pd_value`, etc.) with descriptive labels (`False Positive Rate`, `Predicted Value`, `Partial dependence`, etc.) |
| **Descriptive default titles** | 01–06, 14–17, 19–24, 31 | Auto-generate chart titles from chart type + model name where available |
| **CI/error bands: no outline, lower alpha** | 05, 10, 14 | Shaded confidence/error bands should have no stroke outline and lower opacity |
| **Markers on data points** | 04, 05, 14, 24, 31 | Add markers (circle/square) at discrete data points on line charts |
| **Y-axis starts at 0 for bar charts** | 12, 15, 17 | Bar charts must anchor y-axis at 0 |
| **Baseline/reference lines: neutral color, dashed** | 20, 22 | Baseline diagonal/horizontal should be dashed and light gray, not red |

---

## Per-row feedback

### Row 01 — ROC Curve
- [ ] AUC score in legend labels (`ClassName (AUC = 0.XX)`)
- [ ] Per-class ROC curves + micro/macro averages for multiclass
- [ ] Human-readable axis labels (`False Positive Rate` / `True Positive Rate`)
- [ ] Descriptive default title (`ROC Curve` or `ROC Curve for <Model>`)

### Row 02 — Precision-Recall Curve
- [ ] AP score in legend labels (`ClassName (AP = 0.XX)`)
- [ ] Per-class PR curves + micro-average for multiclass
- [ ] Human-readable axis labels (`Recall` / `Precision`)
- [ ] Descriptive default title (`Precision-Recall Curve`)

### Row 03 — Confusion Matrix
- [ ] Colorbar showing count-to-color mapping
- [ ] Human-readable axis labels (`True label` / `Predicted label`) + title
- [ ] Color scheme similar to scikit-plot (blue sequential palette instead of viridis)

### Row 04 — Calibration Curve
- [ ] Markers on calibration line at each bin point
- [ ] Label diagonal reference as "Perfectly calibrated" in legend
- [ ] Human-readable axis labels (`Mean predicted probability` / `Fraction of positives`)
- [ ] Keep Brier score annotation (ferrum is ahead here)

### Row 05 — Learning Curve
- [ ] Markers on data points at each training-size step
- [ ] Human-readable axis labels (`Training Instances` / `Score`)
- [ ] Full legend labels (`Training Score` / `Cross Validation Score`)
- [ ] CI bands: no outline, lower alpha (more transparent)

### Row 06 — Residuals (4-panel diagnostic)
- [ ] Train/test split coloring when both splits available
- [ ] Human-readable axis labels for each subplot
- [ ] Each subplot needs its own title
- [ ] Keep 4-panel layout (ferrum is ahead here)

### Row 07 — Feature Importance
- Good as-is. No changes.

### Row 08 — Histogram
- [ ] Better axis labels (variable name for x, `Count` for y)
- [ ] Optional KDE overlay (not forced by default)

### Row 09 — Boxplot
- [ ] **Bug fix:** y-axis says `lower_whisker` instead of the variable name
- [ ] Visible median line (dark horizontal line inside each box)
- [ ] Whisker caps (T-bars at whisker endpoints)
- [ ] Open (unfilled) outlier markers

### Row 10 — Regression Scatter
- [ ] Remove CI band stroke outline
- [ ] Lower CI band alpha
- [ ] Add R² annotation

### Row 11 — Correlation Heatmap
- [ ] Diverging colormap centered at 0 (red-blue or similar, not viridis)
- [ ] Lower-triangle mask (optional, since matrix is symmetric)
- [ ] Fix x-axis label truncation (rotate or resize for full names)
- [ ] Keep in-cell correlation values (ferrum is ahead here)

### Row 12 — Bar with Error
- [ ] Error bars (CI whiskers) by default when aggregating
- [ ] Y-axis starts at 0
- [ ] Default aggregation = mean (not max/sum)

### Row 13 — PDP (Partial Dependence Plot)
- [ ] Human-readable axis labels (`Partial dependence` for y, feature name for x)
- [ ] Keep vertical stack layout
- [ ] Fix x-axis text overlap in top subplots (text not visible)

### Row 14 — Validation Curve
- [ ] Auto-detect log-scale x-axis when hyperparameter values span orders of magnitude
- [ ] Markers on data points at each hyperparameter value
- [ ] CI bands: no outline, lower alpha

### Row 15 — CV Scores
- [ ] Per-fold bars (fold 1–N) instead of aggregated train/test bars
- [ ] Dashed mean-score horizontal reference line with value
- [ ] Y-axis starts at 0
- [ ] Human-readable axis labels (`Score` / `Fold`) + title (`Cross Validation Scores for <Model>`)

### Row 16 — Alpha Selection
- [ ] Best-alpha value annotated in legend or on chart
- [ ] Auto-detect log-scale x-axis
- [ ] Add data point markers

### Row 17 — Class Prediction Error
- [ ] Human-readable axis labels (`actual class` / `number of predicted class`) + title
- [ ] Fix axis semantics: actual class on x-axis, predicted class as color (standard convention)
- [ ] Y-axis starts at 0

### Row 18 — Decision Boundary
- [ ] Probability gradient shading (confidence contours, not flat fills)
- [ ] Per-class distinct color palettes (blue/orange/green families)
- [ ] Dark edge outlines on scatter points
- [ ] Human-readable axis labels (feature names)

### Row 19 — Discrimination Threshold
- [ ] CI bands per metric line (showing variance across folds)
- [ ] Vertical dashed line at optimal threshold with value label
- [ ] Human-readable axis labels (`discrimination threshold` / `score`) + title

### Row 20 — Gain Chart
- [ ] Human-readable axis labels (`Percentage of sample` / `Gain`) + title (`Cumulative Gains Curve`)
- [ ] Class labels in legend (`Class 0` / `Class 1` instead of bare `0` / `1`)
- [ ] Dashed baseline line in light gray (not solid red)

### Row 21 — Intercluster Distance
- [ ] Larger proportional circles (dramatically sized to show relative cluster populations)
- [ ] Embedding axis labels (`PC1`/`PC2` or `MDS1`/`MDS2`)
- [ ] Title with method info (`KMeans Intercluster Distance Map (via MDS)`)

### Row 22 — Lift Chart
- [ ] Same fixes as gain chart (row 20): human-readable labels, class names in legend, dashed baseline in light gray
- [ ] Legend

### Row 23 — Parallel Coordinates
- [ ] Lower line alpha for overlap visibility
- [ ] Prominent vertical axis lines at each feature
- [ ] Title with feature count (`Parallel Coordinates for N Features`)

### Row 24 — PCA Scree
- [ ] 95% cumulative variance threshold reference line
- [ ] Markers on cumulative line
- [ ] Legend labels (`cumulative`, `explained variance`, `95% threshold`)

### Row 25 — Rank (Feature Correlation)
- Same fixes as row 11 (correlation heatmap): diverging colormap, lower-triangle mask, fix label truncation, keep in-cell values

### Row 26 — SHAP Summary
- [ ] Sort features by mean |SHAP value| descending
- [ ] Blue→pink diverging colormap (matches SHAP convention)
- [ ] Vertical zero-reference line at SHAP value = 0
- [ ] Labeled colorbar (`Feature value` title, `High`/`Low` endpoints)
- [ ] Reduce beeswarm vertical jitter (less messy)

### Row 27 — Residplot
- [ ] Human-readable axis labels (variable/feature names)
- [ ] Slightly larger default point size

### Row 28 — Pairplot
- [ ] KDE on diagonal instead of histograms
- [ ] Single shared legend for entire grid
- [ ] Shared axes across rows/columns + tighter layout

### Row 29 — Clustermap
- [ ] Row labels on right side (not overlapping dendrogram)
- [ ] Fix row label overflow (truncate/sample/hide when too many rows)
- [ ] Better colormap (magma or inferno instead of viridis for dense heatmaps)
- [ ] Remove internal axis labels (`_row_id` / `column`)

### Row 30 — Jointplot
- [ ] Tighter margins between scatter and marginal histograms

### Row 31 — Cluster Diagnostics
- [ ] Elbow point auto-detection + dashed vertical annotation (`elbow at k=N, score=X`)
- [ ] Markers on data points at each k
- [ ] Title per subplot (`Distortion Score Elbow`, `Silhouette Score`)
- [ ] Human-readable axis labels (`Distortion score` / `Silhouette score`)

---

## Summary statistics

- **Rows reviewed:** 31
- **Rows with no changes needed:** 1 (row 07 — Feature Importance)
- **Rows where ferrum is ahead of comparators:** 3 partial (rows 04, 06, 11 — Brier annotation, 4-panel layout, in-cell values)
- **Most common fix category:** Human-readable axis labels + titles (~22 rows)
- **Bug fixes:** 1 (row 09 — boxplot y-axis label says `lower_whisker`)
