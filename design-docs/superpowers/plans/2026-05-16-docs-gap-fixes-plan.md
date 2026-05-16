# Docs Gap Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

## 1. Objective

Address 25 documentation gaps identified by a new-user audit — adding missing explanations, examples, sections, and visual content to `docs/site/` with no source code changes.

## 2. Spec references

- New-user audit results (this conversation, 2026-05-16)
- Source verification: 54 marks confirmed, smooth methods = `loess` | `lm` + `logistic` (Rust `SmoothMethod` enum), CoordFlip/CoordPolar/CoordGeo/CoordFixed all exist in `src/ferrum/coord.py`, `share_scale()` exists on `_ChartLike` and `RepeatChart`, DPI not user-controllable (hardcoded `"resolution": "screen"`), `regplot` at `src/ferrum/plots/regression.py:869`

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `docs/site/getting-started/first-plot.md` | Q3: type suffix explanation |
| Modify | `docs/site/guide/model-diagnostics.md` | Q11: compare example, Q12: non-sklearn note |
| Modify | `docs/site/guide/saving-and-export.md` | Q2: output methods guide, Q14: DPI note, Q15: PDF note |
| Modify | `docs/site/guide/interactive.md` | Q16: compatibility note |
| Modify | `docs/site/guide/marks-encodings.md` | Q5: positions, Q6: axis, Q7: smooth table, Q8: full mark table, Q17: legend, Q25: palette cycling |
| Modify | `docs/site/guide/composition.md` | Q9: shared scales |
| Modify | `docs/site/guide/figure-helpers.md` | Q10: regplot section |
| Modify | `docs/site/guide/recipes.md` | Q4: sizing, Q5/Q6/Q9/Q17: recipes, Q18: cat sort, Q19: time-series, Q20: coord, Q23: annotations |
| Modify | `docs/site/guide/themes.md` | Q13: theme grid image |
| Create | `docs/site/guide/img/theme-grid.png` | Q13: generated 12-theme visual comparison |

## 4. Constraints

- Documentation only — no source code changes
- Code examples must be runnable against current API; verify before writing
- DPI is not user-controllable — document this honestly (`"resolution": "screen"` is hardcoded)
- Smooth methods are `"loess"` and `"lm"` only (plus `"logistic"` in a separate transform); do not overclaim
- CoordPolar and CoordGeo exist as classes but may have limited renderer support — verify rendering before documenting
- Use reflink style: `[`Symbol`][ferrum.Symbol]`

## 5. Tasks

### Task 1: Quick inline fixes
- [ ] Q3: In `first-plot.md`, where `"species:N"` first appears, add: *The `:N` suffix declares the field as Nominal. Ferrum supports four type codes: `:Q` (quantitative), `:N` (nominal), `:O` (ordinal), `:T` (temporal). See [Marks & encodings](../guide/marks-encodings.md#encoding-channels) for details.*
- [ ] Q12: In `model-diagnostics.md` Caveats section, add bullet: any object exposing `predict()` / `predict_proba()` works (XGBoost, LightGBM, CatBoost); for frameworks without sklearn-compatible APIs, use the `y_true=` / `y_pred=` precomputed path
- [ ] Q15: In `saving-and-export.md`, add note: PDF not natively supported; convert SVG via CairoSVG (`cairosvg file.svg -o file.pdf`), Inkscape, or browser print
- [ ] Q16: In `interactive.md`, add compatibility note: anywidget-based — works in JupyterLab, VS Code notebooks, Google Colab; classic Notebook also supported
- [ ] Verify: read each modified file to confirm edits are in correct location

### Task 2: marks-encodings.md new sections
- [ ] Q8: Expand mark tables to list all 54 marks, grouped: Primitives (point, line, area, bar, rect, rule, text, label, image, tick, segment), Statistical (smooth, smooth_ci=smooth with CI, errorbar, errorband, histogram, density, contour, hex, raster, qq, function), Distribution (boxplot, violin, boxen, swarm), Composition (ribbon), Diagnostic (roc, pr, calibration, confusion, residuals, prediction_error, importance, shap_beeswarm, shap_bar, shap_waterfall, pdp, learning_curve, validation_curve, cv_scores, alpha_selection, discrimination_threshold, class_prediction_error, gain, lift, silhouette, pca_scree, intercluster_distance, decision_boundary, rank1d, rank2d, parallel_coordinates, geoshape, arc)
- [ ] Q5: Add "Position adjustments" section after encoding channels — explain `fm.Dodge()`, `fm.Stack()`, `fm.Jitter()` with bar chart examples
- [ ] Q6: Add "Axis customization" section — `fm.X("col", scale="log")`, axis limits, reversed axis; verify actual API kwargs against source before writing
- [ ] Q7: Add smooth method table: `"loess"` (locally-weighted polynomial, default), `"lm"` (ordinary least squares linear fit), `"logistic"` (logistic regression via separate mark_smooth path). List key kwargs: `ci`, `bandwidth`, `degree`, `n`
- [ ] Q17: Add "Legend control" section — legend position, title, hide, reorder; verify actual API first
- [ ] Q25: Add "Palette cycling" note — what happens when categories exceed palette size
- [ ] Verify: review full file for coherence after all additions

### Task 3: Other guide page sections
- [ ] Q9: In `composition.md`, add "Shared scales" section showing `(chart_a | chart_b).share_scale(x="shared")`; add corresponding recipe
- [ ] Q10: In `figure-helpers.md`, add `regplot` section — read `src/ferrum/plots/regression.py:869` for actual signature; explain difference from `lmplot` (single-axes vs. FacetGrid-style)
- [ ] Q11: In `model-diagnostics.md`, add `compare=` example after the ModelSource section — show `fm.roc_chart(model_a, X_test, y_test, compare={"GBM": model_b})`; verify against `_resolve_source` logic
- [ ] Q14: In `saving-and-export.md`, note that PNG resolution is not user-configurable (screen resolution is the default); if a user needs high-DPI, suggest increasing `width`/`height` in `.properties()`
- [ ] Q2: In `saving-and-export.md`, add "Output methods" section near top: `.show_svg()` → str, `.show_png()` → bytes, `.show()` → Jupyter inline display, `.save(path)` → file
- [ ] Q20: In `recipes.md`, add coord recipes — `CoordFlip` (horizontal bars), note that `CoordPolar`/`CoordGeo` classes exist but have limited rendering support (verify by attempting to render before documenting)
- [ ] Q23: In `recipes.md`, add annotations recipe — `mark_rule` for reference lines, `mark_text` for callouts, layering annotation marks onto existing charts
- [ ] Verify: review each modified file

### Task 4: Standalone content
- [ ] Q4: In `recipes.md`, add "Chart sizing" recipe — `.properties(width=600, height=400)`
- [ ] Q13: Generate 12-theme grid PNG — write a script that renders a simple scatter+smooth chart in all 12 themes, composites into a grid PNG, saves to `docs/site/guide/img/theme-grid.png`; add image to `themes.md`
- [ ] Q18: In `recipes.md`, add "Custom category order" recipe — show how to control category axis order
- [ ] Q19: In `recipes.md`, add "Time-series" recipe — line chart with temporal encoding `:T`; verify `:T` is actually supported by rendering a chart before writing
- [ ] Verify: all new images render correctly; all code examples run

## 6. Acceptance checks

- All modified `.md` files render without broken reflinks (spot-check with `uv run zensical build --clean` if available)
- Code examples in new sections are runnable: `uv run --no-sync python -c "<example>"` for each
- `theme-grid.png` exists and shows all 12 themes visually
- No source code files modified

## 7. Open questions

- Does `:T` temporal encoding actually work end-to-end with datetime columns? Must verify before writing the time-series recipe.
- Legend control API — need to read source to determine exact kwargs (`legend=False`, `legend_position=`, etc.) before writing Q17.
- CoordPolar/CoordGeo — `.coord()` docstring says "currently only CoordFlip is supported" but classes exist; must verify renderer behavior before documenting.
