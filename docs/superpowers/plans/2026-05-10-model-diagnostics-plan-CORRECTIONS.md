# Phase 10 Plan — Corrections Found During 10a Execution

**Status:** Plan needs revision before resuming. Captured here so the next session can fix the plan in one focused pass before continuing implementation.

**Session date:** 2026-05-11
**Current commit on `feat/phase-10`:** `fa26114` (Task 7 complete)
**Tasks completed:** 1, 2, 3, 6, 7
**Tasks skipped:** 5 (redundant — see correction §1)
**Tasks remaining in 10a:** 8, 9, 10, 11

---

## Correction §1: `RenderConfig.numeric_precision` is redundant — drop Task 5

**Plan claim:** The plan assumes the SVG renderer emits floats at full IEEE 754 precision by default and proposes adding a `numeric_precision: Option<u8>` field on `RenderConfig` to quantize for cross-platform-stable goldens.

**Actual codebase state:** `crates/ferrum-core/src/render/svg.rs` already routes every float through `fmt_f(x)` which calls `format!("{:.*}", FLOAT_PRECISION)` with `FLOAT_PRECISION = 3` (defined at `crates/ferrum-core/src/render/mod.rs:26`). So Phase 9 SVG goldens are **already 3-decimal-place quantized**.

**Correction:**
- Drop Task 5 from the plan entirely.
- The "tiered goldens" design (byte_identical/ vs quantized_4dp/) collapses to a single tier — everything renders at 3 dp by the existing renderer. All Phase 10 goldens go under `tests/goldens/phase_10/byte_identical/` (or just `tests/goldens/phase_10/`).
- If Phase 10 solver-sensitive figures (SHAP-Kernel, UMAP, t-SNE, MDS, learning_curve, …) prove not to be byte-identical across platforms even at 3 dp, address it as an empirical issue when it shows up. Don't preemptively add the field.
- Acceptance criteria §14 should drop the "~12 quantized" target. Adjust to "~35–40 byte-identical goldens at 3 dp."
- Drift note for §3.16 about `numeric_precision` should be dropped from Task 41.

---

## Correction §2: Composite mark pattern doesn't match codebase

**Plan claim:** Every Phase 10 mark uses this pattern:

```python
@dataclass(frozen=True)
class mark_residuals:
    kind: str = "studentized"
    reference_line: bool = True

    def _expand(self, chart_ctx: "ChartContext") -> list["LayerSpec"]:
        return [LayerSpec(mark=..., encoding={...})]
```

with a `chart_ctx.color_field_or_default()` helper and `LayerSpec` value class.

**Actual codebase state (see `src/ferrum/marks/composite.py:14-200`):**

- Composite marks are wired via Chart methods that set `chart._pending_stat_mark = (layered_kind, kwargs, desugar_fn)`.
- The desugar function has signature `desugar_<name>(x_field, y_field, **kwargs) -> tuple`, returning the 5-tuple `("__layered__", transforms_list, None, None, layers_list)`.
- `transforms_list` is a list of existing Rust transform instances (e.g. `BoxStats(...)`, `ErrorExtent(...)`).
- `layers_list` is a list of **plain dicts**, each shaped `{"mark": str, "encoding": dict, "mark_kwargs": dict (opt), "data_source": str | None (opt)}`.
- There is no `LayerSpec` class. There is no `chart_ctx` parameter. The desugar receives raw `x_field` / `y_field` strings (or None) extracted from `chart._encoding` at compile time.

**Correction for every Phase 10 mark task:**

Rewrite all 26 mark code blocks (in plan §6.1, §6.2, and per-sub-batch tasks) using the real pattern. Reference implementations to mimic:

- `desugar_boxplot` (composite.py:15-61) — simple stat-driven layered mark.
- `desugar_errorband` (composite.py:88-108) — single-layer ribbon mark.
- `desugar_ribbon` (composite.py:111-134) — uses chart-level y2 encoding from kwargs.
- `desugar_boxen` (composite.py:137+) — multi-layer with multiple named transforms.

**For diagnostic marks specifically:** most of the time the underlying data has hard-coded column names from a `ModelSource` method (e.g. `fpr`, `tpr`, `class` for ROC). These should be referenced literally in the encoding dicts, ignoring the `x_field` / `y_field` arguments. Example shape for `mark_roc`:

```python
def desugar_roc(
    x_field: str | None,
    y_field: str | None,
    *,
    average: str | None = None,
    reference_line: bool = True,
    annotate_auc: bool = True,
    color_field: str | None = "class",  # auto-detected
    **mark_kwargs: Any,
) -> tuple:
    layers = [
        {"mark": "line", "encoding": {"x": "fpr", "y": "tpr", "color": color_field}},
    ]
    if reference_line:
        layers.append({"mark": "rule", "encoding": {"x": "fpr", "y": "fpr"},
                       "mark_kwargs": {"strokeDash": [4, 4]}})
    return ("__layered__", [], None, None, layers)
```

The `color_field` default of `"class"` should resolve to `"model"` instead when the input DataFrame contains a `"model"` column (compare-source path). Handle this either:
- In the figure-function builder, by inspecting `df.columns` before passing the desugar kwarg, OR
- In the desugar by hard-coding a sentinel and resolving at compile time (preferred — keeps the figure builder slim).

A small `_choose_color_field(df, *candidates)` helper in `src/ferrum/_diagnostics/charts.py` is the cleanest place.

---

## Correction §3: Test pythonpath needs project root

**Plan claim:** Test code uses `from tests.fixtures import load_fixture, load_dataset`.

**Actual codebase state:** Pre-Phase 10, `pyproject.toml` had `pythonpath = ["src"]` only — `tests/` was not importable as a package.

**Correction:** Added `pythonpath = ["src", "."]` to `[tool.pytest.ini_options]` in Task 2's commit. This is a one-time fix and is already on the branch. **No further action needed**; just noted here for completeness.

---

## Correction §4 (preventive): "LayerSpec" symbol referenced repeatedly in plan

Search the plan for the string `LayerSpec` — it appears in many `_expand` examples across §6.1, §7.3, §8.2, §9.x. None of those will work; they all need the dict-shape rewrite. Also remove the `from ferrum import LayerSpec` (or `# type: ignore[attr-defined]`) markers.

---

## Recommended plan-revision pass before resuming

1. Edit `docs/superpowers/plans/2026-05-10-model-diagnostics-plan.md`:
   - Strike Task 5 (replace with a short note explaining the existing 3-dp quantization).
   - Rewrite §2.2 desugaring table footnote: not "_expand" — describe `desugar_<name>` pattern explicitly.
   - Rewrite every `### Task N` mark code block (8, 15, 18, 19, 21, 22, 23, 28, 30, 31, 32, 37) to use `desugar_<name>` + dict-layers.
   - Drop "quantized_4dp/" tier from §11 and §12 and §14.
   - Drop §13 drift note about `RenderConfig.numeric_precision`.
2. Re-commit the revised plan with message: `docs(phase-10): correct plan to match codebase mark-desugar pattern`.
3. Then resume implementation at Task 8.

The corrections do **not** change Phase 10's user-facing surface (marks, methods, figures, visualizers, signatures, schemas, semantics). They're internal pattern-fitting only. The design doc at `docs/superpowers/specs/2026-05-10-model-diagnostics-design.md` is largely unaffected — only the spec drift notes section needs the small `numeric_precision` drop.

---

## What works as-of `fa26114`

- `feat/phase-10` branch created from `main` at `2d6f994`.
- 5 commits land cleanly:
  - `09734c0` build deps + extras
  - `e883c44` test fixtures (skops + parquet)
  - `dc14a02` conftest session check
  - `3897ff6` `_diagnostics` package + lazy-import helpers
  - `fa26114` `ModelSource` class with `.predictions()` + `.probabilities()`
- 10 diagnostics tests passing.
- `import ferrum` does not load sklearn, shap, or umap.
- `ModelSource(non_sklearn_model, X)` does not load sklearn.
- Pre-fit `.skops` fixtures for binary_logistic, multiclass_logistic, regression_ridge, regression_rf, kmeans_3cluster, pca_4comp.
- Parquet datasets for binary/multiclass classification, regression, clustering.

Resume from Task 8 after the plan revision pass.
