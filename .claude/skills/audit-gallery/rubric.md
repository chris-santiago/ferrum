# Gallery Audit Rubric

Score each item as `yes` / `no` / `n/a` for both the **ferrum** panel and the **reference** panel(s). The *delta* — where the reference passes and ferrum fails — is the actionable finding.

## A. Identity & labels

- **A1. Title** — Plot has a title that names what is being plotted (e.g. "ROC curve", "Confusion matrix").
- **A2. X-axis label** — X axis has a meaningful label with units where applicable (e.g. "False positive rate", not "x").
- **A3. Y-axis label** — Same as A2 for the y axis.
- **A4. Tick legibility** — Tick labels don't overlap; precision is appropriate (no `0.34782619` when `0.35` suffices).

## B. Domain-expected annotations

Per-plot expectations. Mark `n/a` for items that don't apply.

- **B1. ROC: AUC value displayed** (in legend, in title, or as on-plot text).
- **B2. PR: Average Precision / AUPRC displayed.**
- **B3. Confusion matrix: per-cell counts overlaid** (and/or per-cell percentages).
- **B4. Calibration: Brier score or ECE shown.**
- **B5. Residuals / regression: R², MAE, or RMSE shown.**
- **B6. Regression scatter: fit equation or R².**
- **B7. Correlation heatmap: cell values overlaid.**
- **B8. Feature importance: importance values shown on bars or as ticks.**

## C. Reference lines / guides

- **C1. ROC: diagonal chance line** (y=x from (0,0) to (1,1)).
- **C2. PR: baseline rate horizontal line** (positive-class prevalence).
- **C3. Calibration: y=x perfect-calibration line.**
- **C4. Residuals: y=0 horizontal line.**
- **C5. Q-Q / calibration: identity line.**
- **C6. Learning curve: train vs validation series visually distinguishable** (color, dash, or labeled markers).

## D. Uncertainty / variance

- **D1. Learning curve: shaded ±std band around each curve.**
- **D2. Regression scatter: CI band around fit line.**
- **D3. Bar chart: error bars** (CI or SE) when summarizing aggregates.
- **D4. Time series: shaded confidence band.**

## E. Legend

- **E1. Present when ≥2 series.**
- **E2. Series names are meaningful** (e.g. "class_0", not "series 0" or column index).
- **E3. Positioned without occluding data.**

## F. Color

- **F1. Colorblind-safe** (no red/green pairings as the primary distinguishing channel).
- **F2. Cmap type matches data type** — sequential for ordered/unsigned, diverging for signed (e.g. residuals), categorical for nominal classes.
- **F3. Saturation reasonable** — no 100%-saturated fills that fight the data.

## G. Layout

- **G1. Aspect ratio appropriate** — square for confusion matrix, wider-than-tall for time/learning curves, etc.
- **G2. Margins don't crop labels.**
- **G3. Gridlines help rather than distract** — present and subtle, not absent and not dominant.
- **G4. Data-ink ratio sensible** — no chartjunk, but not so sparse it's hard to read.

## Severity assignment

After scoring, compute severity for the row's "ferrum lacks X" set:

- **HIGH** — Missing a B-category annotation (domain-expected metric: AUC on ROC, cell counts on confusion matrix, etc.). These are the things a domain expert would immediately notice are absent.
- **MEDIUM** — Missing a C-category reference line, or a D-category uncertainty band, or an E1 legend.
- **LOW** — A/F/G items: missing axis label, suboptimal color choice, awkward aspect ratio. Real issues, but cosmetic relative to information loss.

Pick the highest severity present. A row with both a missing AUC (HIGH) and a missing axis label (LOW) is HIGH.
