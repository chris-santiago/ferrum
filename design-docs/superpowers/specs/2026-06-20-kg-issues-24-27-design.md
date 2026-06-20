# KG Issues #24–#27 Fix Design Spec

> Source: the four known-gap GitHub issues filed from the archaeology #6/#7/#8 review. KG-6 (#24), KG-7 (#25), KG-8 (#26), KG-9 (#27). User decisions (2026-06-20): KG-6 → change the default layout; KG-7 → guard test + close as not-a-bug.

## 1. Scope

Close the four known-gap issues: fix the `facet(col=)` default layout (KG-6), lock in correct square-glyph clipping with a regression test and close #25 as not-reproducible (KG-7), honor `sort` on the shape encoding (KG-8), and expose a readable `EncodingSpec.condition` getter (KG-9).

## 2. Goals

- **KG-6:** `facet(col=X)` with no explicit `ncols` lays panels **side-by-side** (a single row of `n_distinct(X)` columns), matching Altair/seaborn. `facet(row=X)` stays a vertical stack. Affected goldens are regenerated and visually inspected.
- **KG-7:** A regression test proves square (and other Rect/Path) glyphs in faceted charts stay within the panel clip and never exceed it more than a circle of the same `size`. #25 is closed as not-reproducible with the evidence.
- **KG-8:** The `shape` encoding honors `EncodingSpec.sort` (alphabetical, data-aware `"x"/"-x"/"y"/"-y"`, sort-field object, explicit array), mirroring the color/positional paths.
- **KG-9:** `EncodingSpec.condition` is readable via a getter; the stub declares it; the parity test no longer needs a write-only allowlist entry for it.
- No regressions; non-faceted output byte-identical except the intended KG-6 col-facet layout change.

## 3. Non-goals

- **KG-6:** No change to `facet(row=X)` (vertical stack is correct) or to the generic `facet(field=X)` wrap default (out of #24's scope; remains `ncols=1` unless the user passes `ncols`). No change when `ncols` is explicit.
- **KG-7:** No production code change — investigation established squares render inside the clip (half-extent `r*0.8` < circle `r`); only a guard test is added.
- **KG-8:** No new public API; `sort` already exists on `EncodingSpec`. No change to shape-palette overflow behavior.
- **KG-9:** No change to the 5 concrete stub defaults flagged in #27 (verified to match the true runtime defaults; the stub is more informative, not wrong).

## 4. System behavior

- **KG-6:** `fm.Chart(df).mark_point().encode(...).facet(col="g")` renders `n_distinct(g)` panels in one horizontal row (was: one vertical column). `.facet(col="g", ncols=2)` is unchanged. `.facet(row="g")` is unchanged (vertical). Every col-only faceted chart's SVG changes; goldens regenerate.
- **KG-7:** No behavior change. New test asserts faceted square-glyph bounding boxes ⊆ their panel clip rect.
- **KG-8:** `shape=fm.Encoding("g", sort="ascending")` (or `"-y"`, or `["b","a","c"]`, or `{"field":"v","op":"mean","order":"descending"}`) reorders which glyph each category receives and the legend order, identically to how `color=` already responds to `sort`. Absent `sort` → first-appearance order (unchanged).
- **KG-9:** `EncodingSpec(field, condition=...).condition` returns the dict/list passed at construction (was: `AttributeError`).

## 5. Architecture

- **KG-6** lives entirely in the Python declaration layer (`src/ferrum/chart.py`). The Rust `FacetSpec` already supports the needed geometry; only the Python→Rust `_to_facet_dict` serialization and the `_Facet` value object change. The orientation (col vs row) — currently discarded when `facet()` collapses both into `mode_kind="wrap"` — must be preserved so serialization can infer `ncols` for col-orientation only.
- **KG-7** is a test-only addition (Python, in the rendering test suite).
- **KG-8** lives in the Rust scale layer (`scale_resolve/auxiliary.rs::build_shape_scale`), reusing the existing `SortContext` + `apply_sort_to_domain` from `domain.rs` exactly as `color.rs` does. Aligning shape's warning return to color's `Vec<RenderWarning>` is a cohesion side-benefit; the only caller is `build_auxiliary_scales` (mod.rs).
- **KG-9** is a Rust PyO3 getter (`spec/encoding.rs`) mirroring the existing `sort` getter, plus the Python stub (`_core.pyi`) and the parity test.

## 6. Canonical interfaces / data contracts

**KG-6 — `_Facet` gains orientation; `_to_facet_dict` wrap branch infers `ncols`:**
```python
# _Facet: add a field recording wrap orientation (None for generic field= wrap)
wrap_orient: Optional[str] = None   # "col" | "row" | None

# facet(): set wrap_orient when routing col=/row= to the wrap branch
#   col=X only  -> _Facet(mode_kind="wrap", field=col, wrap_orient="col", ...)
#   row=X only  -> _Facet(mode_kind="wrap", field=row, wrap_orient="row", ...)
#   field=X     -> _Facet(mode_kind="wrap", field=field, wrap_orient=None, ...)

# _to_facet_dict() wrap branch:
if f.ncols is not None:
    ncols = f.ncols
elif f.wrap_orient == "col":
    ncols = self._infer_facet_cardinality(f.field)   # side-by-side row
else:                                                # "row" or generic wrap
    ncols = 1
```
`wrap_orient` must survive `_clone`/`__copy__`/any `_Facet` reconstruction (it is a frozen dataclass field).

**KG-8 — `build_shape_scale` honors sort; return aligns to color:**
```rust
pub fn build_shape_scale(
    encoding: &Encoding,
    batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    facet_shared: bool,
) -> Result<(Option<ShapeScale>, Vec<RenderWarning>), RenderError>   // was (Option<ShapeScale>, Option<RenderWarning>)
```
After `let mut distinct = distinct_values_in_order(domain_batch, &shape_enc.field)?;`, build a `SortContext { category_field: &shape_enc.field, batch: domain_batch, x_field: encoding.x.as_ref().map(|e| e.field.as_str()), y_field: encoding.y.as_ref().map(|e| e.field.as_str()) }` and call `apply_sort_to_domain(&mut distinct, shape_enc.sort.as_ref(), &sort_ctx, &mut warnings)`. Collect the existing palette-overflow warning into the same `warnings: Vec`. `build_auxiliary_scales` (mod.rs) updates its shape arm to `let (shape, shape_warns) = build_shape_scale(...)?; warnings.extend(shape_warns);`.

**KG-9 — `EncodingSpec.condition` getter (mirror the `sort` getter at encoding.rs:528):**
```rust
/// Conditional encoding rules (selection-driven); returns what was passed at construction.
#[getter]
fn condition(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
    match &self.condition {
        None => Ok(None),
        Some(s) => {
            let json = serde_json::to_string(s).map_err(|e| PyValueError::new_err(e.to_string()))?;
            let json_module = py.import("json")?;
            Ok(Some(json_module.call_method1("loads", (json,))?.unbind()))
        }
    }
}
```
Stub: add `condition` to `EncodingSpec`'s readable attribute block in `_core.pyi`. Parity test: drop any write-only allowlist exception for `EncodingSpec.condition`.

## 7. Invariants and constraints

- **KG-6:** `row=`-only, generic `field=` wrap, and any explicit-`ncols` facet are byte-identical to today. Only col-only-without-ncols changes. Goldens that change MUST be regenerated AND visually inspected (rasterize to PNG, confirm side-by-side panels render correctly) before commit — the orchestrator blesses PNGs; coders must not self-bless.
- **KG-7:** No production code change. The guard test must be a real discriminator (fail if a square's bbox ever exceeds its panel clip).
- **KG-8:** Non-sort behavior byte-identical (absent `sort` → first-appearance order). Data-aware sort referencing a missing field/op emits `SortSpecIgnored` (never panics), inherited from `apply_sort_to_domain`. Faceted shape sort resolves against the shared global batch (the `domain_batch` already chosen by `shared_categorical_batch`), matching the global legend.
- **KG-9:** Round-trip — `EncodingSpec(field, condition=c).condition == c` for dict/list/None. No change to construction or any other getter.
- No matplotlib; no global mutable state; `cargo test` must pass; goldens visually inspected before commit.

## 8. Key decisions and tradeoffs

- **KG-6 preserves orientation rather than re-routing through the grid path.** Keeping the wrap render path (`_merge_child_scenes_grid` with `columns=ncols`) and only changing the inferred `ncols` minimizes the blast radius (row-only/field= unchanged) versus routing col-only through the grid/sparse path (different renderer → wider golden churn). Recording orientation on `_Facet` is the smallest change that recovers the lost col/row distinction.
- **KG-7 is closed as not-a-bug with evidence, not "fixed."** Squares render inside the clip and exceed less than circles; a speculative code change would risk a real regression for a non-existent bug. A guard test locks in the correct behavior.
- **KG-8 aligns `build_shape_scale`'s return to `build_color_scale`'s `Vec<RenderWarning>`** — sibling-API cohesion; the single caller (`build_auxiliary_scales`, from the rust-review R1 refactor) absorbs the change in one place.
- **KG-9 adds the getter** (the complete fix per the no-defer rule) rather than documenting `condition` as write-only.

## 9. Acceptance criteria

- **KG-6:** `.facet(col="g")` (no ncols) emits `mode:{kind:"wrap", ncols:n_distinct(g)}`; a new/updated test asserts col-only panels are horizontally arranged (distinct cx ranges, shared cy) and row-only stays vertical. Full pytest green after golden regeneration + visual inspection.
- **KG-7:** New test fails if any faceted square glyph bbox exceeds its panel clip; passes on current code. #25 ready to close (evidence in the report).
- **KG-8:** Rust unit in `scale_resolve/tests.rs` asserts `shape` domain order follows `sort` (alphabetical + data-aware + explicit array); a Python test asserts the rendered glyph↔category assignment + legend order respond to `shape=Encoding(..., sort=...)`. `cargo test -p ferrum-core` green.
- **KG-9:** `EncodingSpec(field, condition={...}).condition` returns the value; `test_core_stub_parity.py` passes with `condition` as a readable attr (allowlist entry removed). `cargo test -p ferrum-core` + the stub-parity test green.
- Full suite: `cargo test -p ferrum-core` + `-p ferrum-wasm` + full pytest all green.

## 10. Validation strategy

KG-6: discriminating layout test (cx/cy panel arrangement) + golden byte-diff regeneration with mandatory PNG inspection. KG-7: bbox-vs-clip parser asserting containment across all glyph node types in a faceted shape chart. KG-8: fail-before/pass-after Rust unit (domain order) + Python render test (glyph assignment + legend). KG-9: construction+read round-trip + the programmatic signature/attr parity test. The full cargo + pytest suites are the binding gate; any unexpected golden byte-diff outside col-only facets is a regression.

## 11. Open questions

- KG-6 golden blast radius is broad (~107 col-facet call sites; the subset with byte-diff goldens regenerates). If a position-asserting test (e.g. `test_facet_shared_extent.py` helpers that assume a layout) breaks, update it to the correct orientation rather than pinning the old default. None of this blocks the design.
