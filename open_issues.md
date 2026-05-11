# open_issues.md — genuine bugs surfaced during the 2026-05-11 docstring write pass

4 sonnet agents in parallel wrote ~135 docstrings across 26 files (~3700 lines net), bringing every public Python function/class up to the NumPy convention from the ferrum-docstrings skill. Tests still green: 851 pytest + 515 cargo.

This file collects only the genuine bugs / suspect behavior the agents flagged while documenting — drift between docstring and code, stub-params (accepted but unused), references to non-existent methods, etc. Each item is a separate follow-up.

---

## Agent 1 — Phase 10 diagnostics + figures

# Open Issues — Phase 10 Docstring Audit (Agent 1 v2)

Recorded 2026-05-11. Items below are **genuine code bugs or wiring gaps** found
during the docstring-fix sweep, not docstring-only issues (those were fixed in
place).

---

## 1. `shap_chart` `order` parameter accepts `"abs_max"` in the old docstring — the actual code uses `"max"`

**Status:** Docstring already corrected to `"max"` in this sweep. However the
underlying `_shap_order_features` helper in
`src/ferrum/_diagnostics/charts.py` (line 711) checks `order == "abs_mean"`
and falls through to `.max()` for any other string — meaning a caller who
passes `order="abs_max"` (the previously documented spelling) silently gets
`.max()` behaviour without an error. The function should explicitly reject
unknown `order` values with a `ValueError`.

**File:** `src/ferrum/_diagnostics/charts.py` — `_shap_order_features`,
`_shap_bar_chart_from_source`, `_shap_beeswarm_chart_from_source`.

---

## 2. `ComparedModelSource` proxies `_X`/`_y`/`_feature_names`/`_class_names` only via `__getattr__`

`ComparedModelSource` uses `__slots__ = ("_sources",)`, so any access to
`_X`, `_y`, `_feature_names`, `_class_names` resolves through `__getattr__`.
This is correct today because `CalibrationVisualizer.fit` and several chart
builders access `source._X` and `source._y` directly. However, if a future
chart builder calls `source._capabilities` (not currently in the proxy list in
`__getattr__`) it will raise `AttributeError` instead of the intended
`"ComparedModelSource has no single _model"` error. Consider adding
`"_capabilities"` to the `__getattr__` proxy or raising a clearer message.

**File:** `src/ferrum/_diagnostics/source.py` — `ComparedModelSource.__getattr__`.

---

## 3. `FerrumVisualizer.score` has no base implementation for unsupervised visualizers

`score(X, y)` always raises `NotImplementedError` on the base class. Several
visualizers (e.g. `SilhouetteVisualizer`, `PCADecompositionVisualizer`) have
no natural supervised score. The base class docstring now documents this as
intentional, but the class hierarchy gives no way for a caller to test whether
a given visualizer implements `score` before calling it (no `has_score`
property or similar). Consider adding a `has_score: bool = False` class
attribute that concrete overrides flip to `True`.

**File:** `src/ferrum/_diagnostics/visualizers/base.py`.

---

## 4. `ModelSource.compare` signature accepts `**kwargs` but the docstring only lists `random_state` as an example

The `compare` classmethod passes `**kwargs` verbatim to each `ModelSource`
constructor, so `feature_names`, `class_names`, and `sample_weight` all flow
through. However the parameter is typed as `**kwargs: Any` with no documented
constraint. A user who mistakenly passes a positional kwarg that `ModelSource`
does not accept gets a `TypeError` with no hint about which kwargs are valid.
This is not urgent but worth a follow-up to enumerate accepted kwargs in the
signature or add a guard.

**File:** `src/ferrum/_diagnostics/source.py` — `ModelSource.compare`.

---

## Agent 2 — chart.py

# Open Issues — chart.py docstring audit (agent 2 v2)

Issues found while writing NumPy-style docstrings for `src/ferrum/chart.py`.

---

## BUG-1: `Chart.to_json(indent=)` is a stub parameter — accepted but never read

**File:** `src/ferrum/chart.py`  
**Method:** `Chart.to_json(self, *, indent=None) -> str`  
**Body:**
```python
def to_json(self, *, indent=None) -> str:
    spec = self.to_spec()
    return spec.to_json()   # indent is never passed
```
`indent` is silently ignored. `spec.to_json()` (Rust `serde_json`) does not accept an indent argument, so there is no direct fix, but the parameter should either be removed from the signature or a Python-side `json.loads` / `json.dumps(indent=indent)` round-trip should be added to honour it.  
**Impact:** callers passing `indent=2` get compact JSON without a warning.

---

## BUG-2: `Chart.mark_importance(top_k=)` is a stub parameter — accepted but never read

**File:** `src/ferrum/chart.py`  
**Method:** `Chart.mark_importance(self, *, ..., top_k: int | None = None, ...)`  
**Body:** `top_k` is passed through into the `_pending_stat_mark` kwargs dict and forwarded to `desugar_importance`, but the docstring for `ModelSource.importances()` states that truncation is the chart *builder's* responsibility.  Verify whether `desugar_importance` actually uses `top_k` or ignores it.  If ignored, `top_k` should be removed from the mark method's signature (truncation belongs in the figure builder).  
**Impact:** callers who pass `top_k=10` to `mark_importance` directly may get no truncation.

---

## BUG-3: `Chart.__add__` falls back to `HConcatChart` for differing data — `UserWarning` may surprise users

**File:** `src/ferrum/chart.py`  
**Method:** `Chart.__add__`  
**Note:** The `+` operator silently falls back to `__or__` (horizontal concat) when data differs, emitting only a `UserWarning`. This is a design-level issue: `a + b` producing an `HConcatChart` violates user expectations when `+` is documented as "overlay". Consider raising `TypeError` instead and requiring explicit `|` for horizontal concat.  
**Impact:** silent wrong-type return; test suites that construct `+` chains with unintentionally different data frames receive layout charts instead of layered ones.

---

## BUG-4: `Chart.mark_shap_waterfall(sample_idx=-1)` default is a runtime trap, not a clear API error

**File:** `src/ferrum/chart.py`  
**Method:** `Chart.mark_shap_waterfall(self, *, sample_idx: int = -1, ...)`  
**Note:** The default value `-1` is a sentinel that causes `TypeError` at *desugar time* (not at call time). This means the error is deferred until `show_svg()` / `to_spec()` is called, not when `.mark_shap_waterfall()` is called. Users can construct a broken chart without immediate feedback.  
**Recommendation:** Raise `ValueError` immediately in `mark_shap_waterfall` when `sample_idx == -1`, or use a different sentinel (e.g. `None`) with an explicit guard.

---

## BUG-5: `Chart.mark_segment` sets `new._position = position` twice

**File:** `src/ferrum/chart.py`  
**Method:** `Chart.mark_segment`  
**Body:**
```python
def mark_segment(self, *, position=None, **kwargs) -> "Chart":
    ...
    new = self._set_mark("segment", **kwargs)
    new._position = position   # redundant: _set_mark already handles position via kwargs pop
    return new
```
`_set_mark` already pops `position` from kwargs and validates it. `mark_segment` pops `position` separately from `**kwargs` before calling `_set_mark`, then sets `new._position = position` on the result. This is doubly-setting the same field but not structurally broken; however if `_set_mark` is ever changed to default `_position` to something other than `None`, the override here could mask that. Minor inconsistency — all other marks delegate position handling entirely to `_set_mark`.

---

## Agent 3 — encoding / themes / figure / composition

# Open Issues — Agent 3 v2 (docstring sweep)

Logged 2026-05-11 during the NumPy-style docstring sweep of the 23 in-scope
source files. These are genuine code-level bugs (not docstring typos) found
while reading function bodies to verify stub-param honesty and output-column
accuracy.

---

## BUG-1: `Theme.background` key is silently dropped by the Rust renderer

**Files:** `src/ferrum/themes/builtins.py`, `src/ferrum/themes/__init__.py`,
`crates/ferrum-core/src/render/mod.rs`

**Severity:** High — every built-in theme that sets a background color
(`dark`, `publication`, `economist`, `fivethirtyeight`, `solarized_light`,
`solarized_dark`) silently renders with the default background because the
key is never forwarded to Rust.

**Root cause:**

- All built-in themes pass `background=<hex>` to `Theme(...)`.
- `Theme.to_theme_inputs_dict()` returns `dict(self._props)` verbatim, so
  the dict key is `"background"`.
- The Rust binding reads `theme_inputs.get("background_color")` (confirmed at
  `crates/ferrum-core/src/render/mod.rs:175`).
- Result: `"background"` is never read by Rust; the background stays at the
  renderer default for every named built-in.

**Fix options (pick one):**

1. Rename the Rust dict key lookup from `"background_color"` to `"background"`
   (keeps builtins correct, minimal Rust change).
2. Rename all `background=` kwargs in `builtins.py` to `background_color=`
   (keeps Rust correct, minimal Python change).
3. Add a key-normalisation step in `to_theme_inputs_dict()` that maps
   `"background"` → `"background_color"` before the dict is handed to Rust.

Option 1 is cleanest (the public Python spelling `background` is already in
all builtins and docs).

---

## BUG-2: `annotate_rect` stores `x2`/`y2` columns but never encodes them

**File:** `src/ferrum/annotations.py` (`annotate_rect`)

**Severity:** Medium — the annotation rect is pinned at `(x1, y1)` regardless
of the `x2`/`y2` arguments supplied by the caller.

**Root cause:**

```python
# line 135-139 (simplified)
df = pl.DataFrame({"_x1": [x1], "_x2": [x2], "_y1": [y1], "_y2": [y2]})
...
return Chart(df).mark_rect(**kwargs).encode(x="_x1", y="_y1")
#                                                          ^ _x2, _y2 never encoded
```

The `_x2` and `_y2` columns exist in the DataFrame but `encode(x2=...,
y2=...)` is never called, so the rect has no width or height.

**Fix:** Add `x2="_x2", y2="_y2"` to the `encode()` call (once X2/Y2 channels
are confirmed to route through the renderer correctly in phase 11+, or sooner
if the channels already work for rect marks).

---

## BUG-3: `annotate_text` stores `_text` column but never encodes it

**File:** `src/ferrum/annotations.py` (`annotate_text`)

**Severity:** Medium — text annotations render as invisible positioned marks
because the text content is never bound to the `Text` encoding channel.

**Root cause:**

```python
# line 194-202 (simplified)
df = pl.DataFrame({"_x": [x], "_y": [y], "_text": [text]})
...
return Chart(df).mark_text(**kwargs).encode(x="_x", y="_y")
#                                           ^ text="_text" is never added
```

**Fix:** Add `text="_text"` to the `encode()` call — `Text` channel support
exists in the codebase; this is simply a missing kwarg.

---

## BUG-4: `clustermap` accepts `cmap=` but never forwards it

**File:** `src/ferrum/figure/matrix.py` (`clustermap`)

**Severity:** Low — stub-param drift; `cmap` is documented as "reserved" in
the docstring but the drift was not caught during initial authoring.

**Root cause:** The `cmap` parameter is accepted in the function signature but
the function body never reads or uses it. The `Color` encoding inside
`clustermap` always uses the default continuous palette regardless of the
`cmap` argument.

**Fix:** Either (a) wire `cmap` through to `continuous_palette(cmap)` for the
color encoding, or (b) add a `warnings.warn` so callers know the argument has
no effect today (consistent with the "reserved for future use" pattern used
elsewhere).

---

## META: Stale "Phase 9+" language

The following files contained references to functionality "planned for Phase
9+" after Phase 9 was marked done. These were updated to "Phase 11+" during
the docstring sweep:

- `src/ferrum/coord.py` — module docstring + all four deferred coord classes
- `src/ferrum/encoding/positional.py` — Theta and Radius channel Notes
- `src/ferrum/annotations.py` — `annotate_rect` and `annotate_text` notes
- `src/ferrum/display.py` — `save_chart` error messages

---

## Agent 4 — marks / utilities

# Open Issues — Agent 4 Docstring Sweep (v2)

Genuine bugs found during the marks + utility module docstring sweep.
Format: `FILE::SYMBOL — description`.

---

## base.py

### `base.py::MarkBase.to_mark_kwargs_dict` — ghost method reference
The docstring (as it existed before this sweep) states: "Other kwargs (e.g.
statistical mark kwargs like `bandwidth`) are returned in `to_transform_kwargs()`
if applicable, not here." No method named `to_transform_kwargs` exists on
`MarkBase` or anywhere in the codebase. Statistical kwargs are actually threaded
through the desugar functions (e.g. `desugar_density`, `desugar_smooth`) which
build the transform objects directly — they never go through a `MarkBase` method.
**Fix applied:** docstring updated in this sweep to remove the ghost reference.
**Residual code issue:** if `to_transform_kwargs()` was intended to exist, it
should be added or the design reconsidered.

---

## marks/composite.py

### `composite.py::desugar_errorbar` — `extent` parameter is no-op for non-CI values
`del` is not called on `extent`, and it is forwarded to `ErrorExtent(method=extent)`.
This is correct behavior. No bug.

---

## marks/statistical.py

### `statistical.py::desugar_histogram` — `right` and `multiple` stub params
`del right, multiple` — both are documented user-facing kwargs in ferrum-spec.md
§3.3 that are silently dropped. These are partially deferred (spec §3.3 mentions
`multiple` for overlapping/stacked histograms; `right` controls open/closed bins).
**Status:** stub-param, should be documented as reserved for future use.

### `statistical.py::desugar_density` — `kernel` stub param
`del kernel` — informational only; the underlying `Kde` transform uses gaussian
exclusively. Any value other than "gaussian" is silently accepted and ignored.
**Status:** stub-param; documented in this sweep.

---

## marks/heavy_stat.py

### `heavy_stat.py::desugar_swarm` — `dodge` stub param
`mark_swarm(dodge=...)` emits a `warn_once` but still renders as if `dodge` is
absent. The parameter is accepted and silently degraded, not raised. This is a
warn-then-no-op pattern, not a hard error.
**Status:** documented in this sweep.

---

## marks/diagnostic.py

### `diagnostic.py::desugar_prediction_error` — docstring contradicted code (FIXED)
Original docstring stated: "``ci`` and ``reference_band`` are reserved for Phase
10h; passing non-default values raises ``NotImplementedError``." The code actually
emits a `ribbon` layer using `_pe_band_lo` / `_pe_band_hi` when `ci is not None`
or `reference_band` is truthy. The behavior is the exact opposite of what the
docstring claimed. **Fixed in this sweep.**

### `diagnostic.py::desugar_pr` — iso-lines column inventory incomplete (FIXED)
The docstring listed `_iso_recall`, `_iso_precision`, `_iso_label` but omitted the
`_iso_f` (the color grouping column), `_iso_label_x`, and `_iso_label_y` columns
required by the text layer. The layer encoding at lines 223-238 uses all six.
**Fixed in this sweep.**

### `diagnostic.py::desugar_intercluster_distance` — wrong default values in docstring (FIXED)
Docstring cited `[100, 1500]` as the size range but the actual defaults are
`min_size=60.0` / `max_size=600.0`. **Fixed in this sweep.**

### `diagnostic.py::desugar_roc` — `average` stub param
`del average` — informational; the figure builder shapes the data. Not a code
defect but a stub-param that should be documented. **Documented in this sweep.**

### `diagnostic.py::desugar_calibration` — `n_bins` / `strategy` stub params
`del n_bins, strategy` — the data is already binned upstream; these are no-ops
at the mark layer. **Documented in this sweep.**

### `diagnostic.py::desugar_gain` — `reference_lines` stub param
`del reference_lines` — baseline rows already in data. **Documented in this sweep.**

### `diagnostic.py::desugar_lift` — `reference_line` stub param
`del reference_line` — same pattern as desugar_gain. **Documented in this sweep.**

### `diagnostic.py::desugar_discrimination_threshold` — `metrics` / `n_thresholds` stub params
`del metrics, n_thresholds` — data is pre-melted. **Documented in this sweep.**

### `diagnostic.py::desugar_confusion` — `normalize` stub param
`del normalize` — data is pre-shaped. **Documented in this sweep.**

### `diagnostic.py::desugar_alpha_selection` — `ci_style` stub param
`del ci_style` — alpha_selection renders a single curve without CI bands.
**Documented in this sweep.**

### `diagnostic.py::desugar_decision_boundary` — `proba` stub param
`del proba` — informational; renderer handles both class-index and probability z
identically. **Documented in this sweep.**

---

