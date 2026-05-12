# Gallery Audit — Findings & Notes

**Run date:** 2026-05-11
**Rows audited:** 16 (all wired)
**Source verdicts:** `.claude/skills/gallery-audit/output/<row>/verdict.md`
**Full report:** `.claude/skills/gallery-audit/output/REPORT.md` (mirrored at `gallery/REPORT.md`)

Severity counts: **HIGH 8 · MEDIUM 5 · LOW 3**

---

## HIGH severity

### 01_roc — ROC curve
- y-axis labeled **"fpr"** (same as x-axis) — likely a label-binding bug
- No AUC annotation in legend or on plot (sklearn / yellowbrick / scikit-plot all show `AUC = X.XX`)
- No reference diagonal (chance line at y = x)

**My notes:**
I want the plot to be similar to yellowbrick in terms of axis labels and legend.

---

### 02_pr — Precision-recall curve
- No Average Precision (AP) annotation
- No plot title
- Legend shows meaningless single character `"1"` where references show `"LogisticRegression (AP = 1.00)"` or per-class AUPRC values

**My notes:**
I want the plot to be similar to skp in terms of axis labels and legend, but instead of `area = x.xxx` i want `AP x.xxx`.

---

### 03_confusion_matrix
- Shows **row-normalized fractions** instead of raw counts (sklearn / yellowbrick / scikit-plot all show counts by default)
- 2 of 4 cell annotations render **dark-on-dark** (illegible) — needs auto-contrast text color based on cell luminance
- No colorbar
- No title

**My notes:**
I want this to look closer to skp, both in axes labels and color gradient.

---

### 04_calibration
- `ferrum.calibration_chart` raises **NotImplementedError** in single-model mode (reserved for Phase 10h)
- All reference panels (sklearn, scikit-plot) ship: y=x diagonal, axis labels, legend by default
- Brier score / ECE annotation missing across all libraries — a shared improvement opportunity

**My notes:**
This absolutely needs to be implemented, now. I don't know how it missed phase 10H.

---

### 05_learning_curve

**My notes:**
Axes and legend should be closer to sklearn. i also like the points from yellowbrick.

---

### 06_residuls

**My notes:**
I want this much closer to yellowbrick, in all but colors.

---

### 07_feature_importance
- `error_bars=True` is the documented default but **no whiskers are drawn** (scikit-plot ships ±std whiskers by default)
- No auto-title

**My notes:**

---

### 09_boxplot
- y-axis labeled **"lower_whisker"** — internal transform field name leaked into label
- **No median line** drawn in box body
- Whiskers reach **raw min/max** instead of Tukey 1.5×IQR

**My notes:**
<!-- write here -->

---

### 13_pdp — Partial dependence
- All 4 PDP facets forced onto a **single shared x-axis** (~50–1300), collapsing f0/f1/f2 curves to a thin stripe at the far left — per-facet independent x-scales required
- No decile rug (sklearn ships rug ticks by default along the x-axis)
- No per-feature x-axis labels

**My notes:**
<!-- write here -->

---

### 15_cv_scores — Per-fold CV scores
- No title
- No mean-score annotation
- No mean reference line (yellowbrick ships horizontal mean line by default)
- No legend disambiguating train vs. test series

**My notes:**
<!-- write here -->

---

## MEDIUM severity

### 06_residuals
- Reference line is drawn at **~y = 0.15** (mean of studentized residuals) rather than **y = 0** — diverges from all three reference panels

**My notes:**
<!-- write here -->

---

### 08_histogram
- Requested **KDE overlay is dropped** (panel is bars only; seaborn `histplot(kde=True)` ships the smoothed curve)
- x-axis labeled **"bin_start"** (internal transform field) instead of source column name `"total_bill"`

**My notes:**
<!-- write here -->

---

### 11_correlation_heatmap
- Uses **sequential viridis** on signed correlation data — should default to a **diverging** palette centered at 0 (seaborn uses coolwarm/RdBu by default)
- **No colorbar**
- Bottom x-tick labels clipped

**My notes:**
<!-- write here -->

---

### 12_bar_with_error
- No 95% CI error bars on aggregate bars (seaborn ships error bars by default for `barplot`)
- Bar heights ~2.4× seaborn's means — suggests aggregate default is **sum, not mean** (likely a Rust transform default bug)

**My notes:**
<!-- write here -->

---

### 16_alpha_selection
- No title
- No legend
- No on-plot best-α annotation (yellowbrick prints `α = 0.010` in its legend)
- **Wins:** log-scaled x-axis correctly applied by default

**My notes:**
Why are the values so different? Isn't this run on same data?

---

## LOW severity

### 05_learning_curve
- Missing default title (sklearn also omits one — minor)
- All B/C/D rubric items pass: train/val bands, legend, colorblind-safe colors all match references

**My notes:**
<!-- write here -->

---

### 10_regression_scatter
- Matches seaborn on all information items: scatter + OLS line + 95% CI band + axis labels
- Style deltas only: heavier CI band, denser y-ticks
- Both panels share: no title, no R²/fit-equation annotation (seaborn `regplot` doesn't either)

**My notes:**
<!-- write here -->

---

### 14_validation_curve
- Log-axis ticks rendered as 4-digit decimals (`"0.0032"`, `"31.6228"`) instead of clean `10ⁿ` powers
- No title (sklearn also omits one)

**My notes:**
<!-- write here -->

---

## Cross-cutting patterns

A few themes appear across multiple rows — worth considering as systemic fixes rather than per-row patches:

1. **Internal transform field names leak as axis labels** (08 `bin_start`, 09 `lower_whisker`). Likely a shared bug in how Rust transforms emit output column names back into ChartSpec encodings — fix once.
2. **Default annotations missing** (AUC on 01, AP on 02, mean on 15, best-α on 16). Diagnostic chart functions should compute and annotate their headline statistic by default.
3. **Default reference lines missing** (chance diagonal on 01, y=0 on 06, mean line on 15). Each diagnostic chart has a canonical reference geometry — should be auto-added.
4. **Diverging vs. sequential palette selection** (11). Heatmaps with signed data should auto-pick a diverging palette centered at zero.
5. **Auto-titles** (02, 07, 11, 14, 15, 16 all flag missing titles). May warrant a project-wide decision on whether figure-level functions emit default titles.

**My notes on cross-cutting patterns:**
<!-- write here -->

---

# Addendum — Rows 17–31 (added 2026-05-11, second pass)

After the first audit, 15 additional ferrum APIs were identified as uncovered and scaffolded into new rows. This addendum lists their findings without touching the row-1–16 notes above.

**New severity counts:** HIGH 4 · MEDIUM 6 · LOW 5  (rows 17–31)
**Combined run total (rows 1–31):** HIGH 12 · MEDIUM 11 · LOW 8

---

## HIGH severity (new)

### 21_intercluster_distance — Intercluster distance map
- No membership-size scale legend (yellowbrick shows a "Membership" reference circle with size→count)
- No title
- Generic `"x"` / `"y"` axis labels (should be embedding dims or hidden)
- Fully opaque categorical fills — overlapping clusters occlude rather than blend

**My notes:**
<!-- write here -->

---

### 26_shap — SHAP summary (beeswarm)
- Features rendered in **index order** instead of by **mean absolute SHAP value** (the canonical SHAP ordering — `order='abs_mean'` is documented as default but not effective)
- No `x = 0` reference rule
- x-axis labeled with the raw column instead of idiomatic `"SHAP value (impact on model output)"`
- Colorbar missing `"Feature value"` label

**My notes:**
<!-- write here -->

---

### 29_clustermap — Cluster heatmap
- `ferrum.clustermap(df)` **raises at render time**: `ValueError: unknown column '_row_id' referenced by an encoding`
- This is a real bug, not a default-quality gap — confirmed in the smoke test before scaffolding
- Seaborn reference shows the canonical target shape: row + column dendrograms, sequential cmap with colorbar, column labels visible, sparse row-index ticks

**My notes:**
<!-- write here -->

---

### 31_cluster_diagnostics — Cluster diagnostics (elbow + silhouette)
- No annotation of the chosen elbow `k` (yellowbrick KElbowVisualizer prints the picked k and the time-to-fit by default)
- x-axis renders non-integer ticks (`1.6`, `2.7`, `3.8`, …) when k is an ordinal/integer sweep — should snap to integer ticks

**My notes:**
<!-- write here -->

---

## MEDIUM severity (new)

### 17_class_prediction_error — Class prediction error
- No title
- Misleading axis labels: actual-class axis is labeled `"predicted"`, count axis is labeled `"value"`
- Possible stacking-baseline bug visible on the class-2 bar (segments not starting at zero)

**My notes:**
<!-- write here -->

---

### 19_discrimination_threshold — Discrimination threshold
- No per-series ±std uncertainty bands (yellowbrick draws shaded uncertainty bands by default from CV folds)
- No optimal-threshold vertical marker (yellowbrick draws a dashed vline at the f1-optimal threshold by default)
- No title; weak axis labels

**My notes:**
<!-- write here -->

---

### 24_pca_scree — PCA scree
- No legend distinguishing the three series (per-component bars / cumulative line / 95% threshold line)
- All three series rendered in the same color (no visual disambiguation)
- The 95% threshold reference line appears to render at **~0.885** instead of **0.95** — likely a `pca_scree_chart` defaults bug

**My notes:**
<!-- write here -->

---

### 27_residplot — Residual scatter
- No `y = 0` reference line (seaborn `residplot` ships the horizontal zero line by default)
- x-axis labeled `"x"` (placeholder) instead of the source column name (related to the cross-cutting transform-field-label-leak pattern)

**My notes:**
<!-- write here -->

---

### 28_pairplot — Pairplot
- Species legend duplicated in **all 16 subpanels** (should appear once at figure level)
- Tick and axis labels rendered on every interior cell (should be **shared outer-only** axes — only left column + bottom row labelled)
- Diagonal cells use **stacked histograms** where seaborn defaults to **overlaid per-class KDE**

**My notes:**
<!-- write here -->

---

## LOW severity (new)

### 18_decision_boundary — Decision boundary
- Uses sequential **viridis** cmap + continuous colorbar to encode **nominal 3-class** iris labels (cmap-type mismatch; should be discrete categorical palette)
- Axis labels are placeholder `"x"` / `"y"` instead of feature names
- No title (sklearn `DecisionBoundaryDisplay` also omits one — parity)

**My notes:**
<!-- write here -->

---

### 20_gain — Cumulative gain curve
- No title
- x-axis labeled `"percent_population"` (raw column name leak)
- Legend shows bare integer class values (`0`, `1`) instead of `"Class 0"` / `"Class 1"`
- Baseline (random model) rendered as a colored series instead of a dashed/grey guide line

**My notes:**
<!-- write here -->

---

### 22_lift — Lift curve
- No title
- x-axis labeled `"percent_population"` (raw column name leak)
- Bare integer class labels (`0`, `1`) in legend
- (No `lift=1` reference line, but skp also lacks one — parity)

**My notes:**
<!-- write here -->

---

### 23_parallel_coordinates — Parallel coordinates
- No title
- Net parity slightly favors ferrum: yellowbrick lacks axis labels and uses an unfortunate red/green palette

**My notes:**
<!-- write here -->

---

### 25_rank — Feature rank (Rank2D heatmap)
- Sequential **viridis** cmap on **signed correlation** matrix — should be diverging + symmetric around 0 (same issue as row 11 `correlation_heatmap`)
- No title
- Tick labels rendered as `f0..f12` placeholders instead of real wine-dataset feature names (related to row 13 `pdp` per-feature-label issue)

**My notes:**
<!-- write here -->

---

### 30_jointplot — Joint plot
- Information-equivalent to seaborn at defaults — only delta is layout polish (non-square central panel with visible gutter to marginals vs. seaborn's square + flush layout)
- Minor top-marginal outline rendering artifact

**My notes:**
<!-- write here -->

---

## Cross-cutting patterns (updated with rows 17–31)

The original 5 cross-cutting patterns from rows 1–16 all repeat in rows 17–31; new patterns emerge too:

1. **Internal transform field names leak as axis labels** — already on 08 (`bin_start`) and 09 (`lower_whisker`); now also on 20 (`percent_population`), 22 (`percent_population`), 27 (`x`), 18 (`x`/`y`), 21 (`x`/`y`). This is the **single most frequent finding across the entire 31-row audit** — fix at the encoding layer once.
2. **Default annotations missing** — extends to 24 (95% threshold value), 26 (SHAP `x=0` rule + "SHAP value" xlabel), 31 (elbow-k callout), 19 (optimal-threshold vline).
3. **Default reference lines missing** — extends to 27 (`y=0`), 26 (`x=0`).
4. **Diverging vs. sequential palette selection** — extends to 25 (Rank2D heatmap, same bug as row 11); 18 uses sequential viridis on nominal classes (separate but related cmap-type-mismatch).
5. **Auto-titles** — now flagged on 17, 19, 20, 21, 22, 23, 24, 25, 31 in addition to the row-1–16 list. Effectively every ferrum chart without a title is being dinged.
6. **(New) Categorical legend rendering for class labels** — bare integer values shown across 20, 22, and contributing to 02's `"1"` legend (row-1–16). Should map to `"Class N"` or use real class names.
7. **(New) Real feature names not propagated through model-aware chart functions** — 13 (`pdp`), 25 (`rank`), 18 (`decision_boundary`). When sklearn estimators were fit on a DataFrame, `model.feature_names_in_` should populate axis labels.
8. **(New) Legend duplication in faceted/repeat charts** — 28 (`pairplot` shows species legend in all 16 cells). Likely a RepeatChart default.
9. **(New) Two real ferrum bugs surfaced by the audit**, separate from defaults:
   - `clustermap` raises `unknown column '_row_id' referenced by an encoding`
   - `pca_scree_chart` threshold line appears to render at ~0.885 instead of the documented 0.95 default

**My notes on cross-cutting patterns (rows 17–31 addendum):**
<!-- write here -->
