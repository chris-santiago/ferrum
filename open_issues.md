# open_issues.md

## Closed in 2026-05-11 sweep (4 parallel sonnet agents)

Every item surfaced by the docstring-write pass has been addressed. 875 pytest + 515 cargo all green.

### `_diagnostics/` (Agent A)

- ✅ `_shap_order_features` now raises `ValueError` on unknown `order` strings (was silently falling through to `.max()`). `_shap_bar_chart_from_source` got the same guard. Tests added in `test_explanation.py`.
- ✅ `ComparedModelSource.__getattr__` now proxies `_capabilities`. Tests added in `test_compare.py`.
- ✅ `FerrumVisualizer.has_score: bool = False` class sentinel added; flipped to `True` on `ROCVisualizer`, `ResidualsVisualizer`, `PredictionErrorVisualizer` (the three with real `score()` bodies). Tests in `test_regression.py`.
- ✅ `ModelSource.compare(**kwargs)` now validates kwargs against `{random_state, feature_names, class_names, sample_weight}` with an informative `TypeError`. Tests in `test_source.py`.

### `chart.py` (Agent B)

- ✅ `Chart.to_json(indent=...)` now forwards via `json.loads`/`json.dumps(indent=)` round-trip. Tests added.
- ✅ `Chart.mark_importance(top_k=...)` confirmed consumed by `_importance_chart_from_source` (line 678: `df.head(top_k)`); `desugar_importance` itself ignores it (split responsibility). Docstring corrected to describe the split rather than implying stub.
- ✅ `Chart.__add__` HConcat-fallthrough `UserWarning` message now suggests concrete remediation: combine the two DataFrames into one with null padding (decision_boundary_chart as the example). Test asserts the new phrasing.
- ✅ `Chart.mark_shap_waterfall(sample_idx=-1)` sentinel now raises `ValueError` immediately at the `Chart.mark_shap_waterfall` call site (was previously a runtime trap deep in the chart builder). Tests in `test_chart.py`.
- ✅ `Chart.mark_segment` no longer double-sets `_position`; replaced with the same `_set_mark(..., position=position, **kwargs)` pattern used by other marks. Test asserts position is set exactly once.

### themes / annotations / figure / matrix / Phase 9 cleanup (Agent C)

- ✅ `Theme.background` is now consumed by the Rust renderer. `to_theme_inputs_dict()` renames `background` → `background_color` (the key Rust's `render/binding.rs:170` reads) before passing across the boundary; public API stays `background=`. Test asserts the rendered SVG contains the configured background color.
- ✅ `annotate_rect` now encodes `x2="_x2"`, `y2="_y2"` so rects actually render as rects (not degenerate points). Test asserts the encoding dict.
- ✅ `annotate_text` now encodes `text="_text"`. Test asserts the encoding dict.
- ✅ `clustermap(cmap=...)` now forwards via `Color("value", scheme=cmap)` on the center heatmap. Tests cover both default `viridis` and explicit cmap.
- ✅ `clustermap` dendrogram panels now disable gridlines via an internal `Theme(grid=False)` base; user-supplied `theme=` merges on top, preserving non-grid customization. Tests cover both paths.
- ✅ Stale Phase-9 language scan: `marks/deferred.py` ("Phase 9+" → "Phase 11+"), `_warn.py`, `composition.py` NotImplementedError messages. `test_marks.py`/`test_warn.py` updated to match.

### marks/ (Agent D — mostly confirmation, no false alarms)

- ✅ `MarkBase.to_mark_kwargs_dict` ghost reference: previously fixed in the earlier sweep. No code change needed.
- ✅ `desugar_errorbar(extent=...)` non-CI values: confirmed NOT a bug. `extent` is forwarded to `ErrorExtent(method=extent)` and the Rust transform already supports all four values (`ci`, `stderr`, `stdev`, `iqr`) with `PyValueError` on unknowns.
- ✅ 11 stub-param desugar entries (`desugar_histogram.right/multiple`, `desugar_density.kernel`, `desugar_swarm.dodge`, `desugar_roc.average`, `desugar_calibration.n_bins/strategy`, `desugar_gain.reference_lines`, `desugar_lift.reference_line`, `desugar_discrimination_threshold.metrics/n_thresholds`, `desugar_confusion.normalize`, `desugar_alpha_selection.ci_style`, `desugar_decision_boundary.proba`): confirmed all are intentional informational pass-throughs — the chart builder owns the shaping, the mark layer records the kwarg for downstream introspection. Docstrings already correct.
- ✅ Cross-agent reconciliation: Agent B changed `Chart.mark_shap_waterfall` to raise `ValueError` at the call site; Agent D updated `desugar_shap_waterfall` (now never reached for `sample_idx=-1`) from `TypeError` → `ValueError` for consistency, and the existing test in `test_explanation.py` was updated to match.

---

## Test result

- 875 pytest passed (up from 851 before the sweep — 24 new regression tests)
- 515 cargo tests passed
- 0 regressions

No open items remain from the audit. Future findings should be appended here as `## NEW <date>` sections.
