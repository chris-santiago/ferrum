# Resume guide — completing the gallery audit

This file is the recovery point for continuing the gallery audit when ferrum's
Python API gains new visualizers, or when previously-stubbed rows are ready to
be wired up.

## How to read this file

For each row, the status is one of:

- **WIRED** — `config.toml` + all panel scripts exist; row runs end-to-end.
- **READY** — ferrum API exists; comparator panels not yet written. Pick up here.
- **PARTIAL** — ferrum API exists but a default may be missing; the audit itself
  will surface this — write the panels and run.
- **BLOCKED** — ferrum API not implemented; needs work in `src/ferrum/` first.

The per-row `plots/<row>/TODO.md` always names the exact ferrum function and
the exact comparator function calls. Treat this RESUME.md as the index; treat
the TODO.md as the recipe.

## Current status

| Row | Plot | Ferrum status | Notes |
|---|---|---|---|
| 01_roc | ROC curve | **WIRED** | 4 panels rendering. |
| 02_pr | Precision-recall | **WIRED** | 4 panels rendering. |
| 03_confusion_matrix | Confusion matrix | **WIRED** | 4 panels rendering (500×500). |
| 04_calibration | Calibration | **WIRED (PARTIAL)** | 2/3 panels render; ferrum.calibration_chart raises `NotImplementedError` in single-model mode (reserved for Phase 10h). Audit flags this. |
| 05_learning_curve | Learning curve | **WIRED** | 3 panels rendering. Unblocked by Task 28 (`ferrum.learning_curve_chart`, commit `cf6858a`). **First audit run surfaced a ferrum-side polygon-vertex-ordering bug on the test-curve ribbon — flagged for follow-up.** |
| 06_residuals | Residuals | **WIRED** | 4 panels rendering on load_diabetes. |
| 07_feature_importance | Feature importance | **WIRED** | 3 panels rendering. Unblocked by Task 21 (`ferrum.importance_chart`, commit `bcaeb65`). |
| 08_histogram | Histogram + KDE | **WIRED** | 2 panels rendering (ferrum, seaborn). |
| 09_boxplot | Boxplot | **WIRED** | 2 panels rendering. |
| 10_regression_scatter | Scatter + reg | **WIRED** | 2 panels rendering. |
| 11_correlation_heatmap | Heatmap | **WIRED** | 3 panels rendering (yellowbrick uses Bunch.data, not as_frame). |
| 12_bar_with_error | Bar + error | **WIRED** | 2 panels rendering; expect default-CI gap finding. |
| 13_pdp | Partial dependence | **WIRED** | 2 panels rendering (ferrum + sklearn — no yellowbrick/scikit-plot equivalent). Added after Task 23 (`ferrum.pdp_chart`, commit `4679e86`). |
| 14_validation_curve | Validation curve | **WIRED** | 3 panels rendering. Added after Task 28 (`ferrum.validation_curve_chart`, commit `cf6858a`). |
| 15_cv_scores | Per-fold CV scores | **WIRED** | 2 panels (ferrum + yellowbrick — no sklearn equivalent). Added after Task 28 (`ferrum.cv_scores_chart`). |
| 16_alpha_selection | Alpha selection | **WIRED** | 2 panels (ferrum + yellowbrick using RidgeCV; no sklearn equivalent). Added after Task 28 (`ferrum.alpha_selection_chart`). |
| 17_class_prediction_error | Class prediction error | **WIRED** | 2 panels rendering (ferrum + yellowbrick). |
| 18_decision_boundary | Decision boundary | **WIRED** | 2 panels rendering (ferrum + yellowbrick). |
| 19_discrimination_threshold | Discrimination threshold | **WIRED** | 2 panels rendering (ferrum + yellowbrick). |
| 20_gain | Gain curve | **WIRED** | 2 panels rendering (ferrum + yellowbrick). |
| 21_intercluster_distance | Intercluster distance | **WIRED** | 2 panels rendering (ferrum + yellowbrick). |
| 22_lift | Lift curve | **WIRED** | 2 panels rendering (ferrum + yellowbrick). |
| 23_parallel_coordinates | Parallel coordinates | **WIRED** | 2 panels rendering (ferrum + yellowbrick). |
| 24_pca_scree | PCA scree | **WIRED** | 2 panels rendering (ferrum + yellowbrick). |
| 25_rank | Rank 1D/2D | **WIRED** | 2 panels rendering (ferrum + yellowbrick). |
| 26_shap | SHAP beeswarm | **WIRED** | 2 panels rendering (ferrum + shap). |
| 27_residplot | Residplot (EDA) | **WIRED** | 2 panels rendering (ferrum + seaborn). |
| 28_pairplot | Pair plot | **WIRED** | 2 panels rendering (ferrum + seaborn). |
| 29_clustermap | Cluster heatmap | **WIRED** | 2 panels rendering (ferrum + seaborn). |
| 30_jointplot | Joint plot | **WIRED** | 2 panels rendering (ferrum + seaborn). |
| 31_cluster_diagnostics | Cluster diagnostics | **WIRED** | 2 panels rendering (ferrum + yellowbrick). Elbow + silhouette scoring. |
| 32_silhouette | Per-sample silhouette | **WIRED** | 2 panels (ferrum + yellowbrick). `SilhouetteVisualizer` — per-sample horizontal bars by cluster. |
| 33_prediction_error | Prediction error | **WIRED** | 3 panels (ferrum + sklearn + yellowbrick). `PredictionErrorVisualizer` — actual-vs-predicted scatter. |
| 34_cooks_distance | Cook's distance | **WIRED** | 2 panels (ferrum + yellowbrick). `CooksDistanceVisualizer` — influence diagnostic. |
| 35_manifold | Manifold embedding | **WIRED** | 2 panels (ferrum + yellowbrick). `ManifoldVisualizer` with `method="tsne"`. |
| 36_classification_report | Classification report | **WIRED** | 2 panels (ferrum + yellowbrick). Precision/recall/F1 per-class heatmap. |
| 37_class_balance | Class balance | **WIRED** | 2 panels (ferrum + yellowbrick). Per-class label count bars. |
| 38_catplot | Categorical strip | **WIRED** | 2 panels (ferrum + seaborn). Default `kind="strip"` (box/bar covered by rows 09/12). |
| 39_shap_bar | SHAP bar | **WIRED** | 2 panels (ferrum + shap). Mean |SHAP| importance bar chart. |
| 40_shap_waterfall | SHAP waterfall | **WIRED** | 2 panels (ferrum + shap). Single-observation waterfall explanation. |

## Resume protocol

When a new ferrum visualizer lands (or when the user wants to wire up a READY row):

1. **Read** the relevant `plots/<row>/TODO.md`.
2. **Copy** `plots/01_roc/<library>_panel.py` as the template; swap the dataset
   load, the model fit, and the call into the comparator library.
3. **Update** `plots/<row>/config.toml`:
   - Set `panels = ["ferrum", ...]` (whichever you wrote).
   - Set `ferrum_status = "READY"` if it was BLOCKED.
4. **Run** the row in isolation: `audit.py all --rows <N>`.
5. **Read** the resulting `output/<row>/verdict.md` — that's the gap list.
6. **Hand off** to `gallery-fixer` if any HIGH-severity items appear.

## Resume protocol when ferrum BLOCKED rows unblock

1. Implement the ferrum API per the row's TODO.md "Needed in ferrum" section.
2. Add the function to `ferrum/__init__.py` `__all__` and the relevant figure
   module.
3. Add `ferrum_panel.py` to the row directory.
4. Flip `ferrum_status = "READY"` in `config.toml`.
5. Run the audit. The verdict will tell you whether your new default matches
   what the canonical libraries ship.

## Adding new rows

If a new plot type becomes worth comparing:

1. Pick the next free `NN_<slug>/` directory under `plots/`.
2. Add `config.toml` + `TODO.md` following any existing row as the template.
3. Add a row to the table in this RESUME.md so future-Claude can find it.
4. Write the panel scripts and run.

The orchestrator auto-discovers any row directory with a `config.toml`, so no
registration step is required beyond writing the files.

## Things that should never change without explicit user approval

- **No matplotlib in the project venv.** Comparator panels MUST stay in
  isolated PEP 723 envs (`uv run --no-project --script`). If you find yourself
  adding `matplotlib` to `pyproject.toml` to "make Pyright happy" or "simplify
  things", stop. This is a hard ferrum constraint.
- **Defaults only.** Never modify a panel script to tweak ferrum's output to
  make it look better against a reference. The audit measures default
  behavior; tweaking defeats the entire point.
- **Determinism env vars.** All panel scripts must consume `FERRUM_AUDIT_*`
  env vars from `audit.py` rather than hardcoding seeds, fonts, or sizes.
  Mixing hardcoded values into one panel and env-driven values into another
  will create spurious "ferrum is different" findings.
