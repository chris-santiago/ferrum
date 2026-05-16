# Identity Transform + data_source Routing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

## 1. Objective

Replace the column-rename approach to layer data isolation with a principled `Identity` transform + `data_source` routing pattern. This fixes the chained `+` layering bug and hardens two other diagonal-concat sites against the same class of column-overlap issue.

## 2. Spec references

- Commit `5041463` on `docs/gap-fixes` — current column-rename fix (to be replaced)
- `crates/ferrum-core/src/render/prepare.rs:62-77` — `from_chart_and_layer` + `inherit_from`
- `crates/ferrum-core/src/render/scene_build.rs:99-106` — `data_source` batch routing
- `crates/ferrum-core/src/render/scale_resolve.rs:695-738` — named-output domain unioning
- `crates/ferrum-core/src/transform/reference_line.rs` — existing named-output transform (model)
- `src/ferrum/chart.py:3930-3980` — `__add__` method
- `src/ferrum/chart.py:3810-3832` — `PublicLayer` diagonal concat in `_resolve_pending`

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Create | `crates/ferrum-core/src/transform/identity.rs` | New Identity transform variant |
| Modify | `crates/ferrum-core/src/transform/core.rs` | Register Identity in TransformSpec enum |
| Modify | `crates/ferrum-core/src/transform/mod.rs` | Add module |
| Modify | `crates/ferrum-core/src/lib.rs` | Export PyIdentity class |
| Modify | `src/ferrum/chart.py` | Replace column-rename with data_source routing in `__add__` and PublicLayer path |
| Modify | `src/ferrum/encoding/base.py` | Remove `_replace_field` (no longer needed) |
| Modify | `tests/test_regression_2026_05_16.py` | Update + add chained-layer tests |

## 4. Constraints

- Same-data layering (`Chart(df) + Chart(df)`) must remain a no-op — no transforms injected
- Transform-based layering (scatter + smooth via `_NamedTransform`) must remain unchanged
- `calibration_chart`'s existing `ReferenceLine` routing must not regress
- Scale domain unioning must include Identity-routed data (already handled by `numeric_domain_union` for named outputs)
- The `Identity` transform must be minimal — pass input batch through unchanged, publish under a name

## 5. Tasks

### Task 1: Rust Identity transform
- [ ] Create `crates/ferrum-core/src/transform/identity.rs`
- [ ] Struct: `IdentitySpec { name: Option<String> }` (serde-compatible)
- [ ] `apply()`: return input batch unchanged
- [ ] Add `Identity(IdentitySpec)` variant to `TransformSpec` enum in `core.rs`
- [ ] Wire through `apply_transforms_named` dispatch
- [ ] Add PyO3 class `PyIdentity` with `#[new] fn new(name: String)` — the `name` field is required (it's the named-output key)
- [ ] Register in `lib.rs`: `m.add_class::<PyIdentity>()?;`
- [ ] Verify: `cargo test`

### Task 2: Update `__add__` to use Identity routing
- [ ] Remove the column-rename logic (`rhs_col_renames`, `_rename_encoding_fields`)
- [ ] When `data differs AND columns overlap AND no RHS transforms`: wrap the RHS data as a named `Identity` transform. Keep the RHS DataFrame as a separate Arrow table, inject it via `_NamedTransform(Identity(name), name)`, and set `data_source=name` on all RHS layers
- [ ] When `data differs AND columns do NOT overlap`: keep the existing diagonal-concat path (no rename needed, no Identity needed)
- [ ] When `data differs AND RHS has transforms`: keep existing `_NamedTransform` routing (lines 3971-3978)
- [ ] Handle chained `+`: when LHS is already a multi-layer chart, the LHS data may already contain named outputs. The new Identity for the RHS just adds another named output — no conflict
- [ ] Remove `_rename_encoding_fields` from `chart.py`
- [ ] Remove `_replace_field` from `encoding/base.py`

### Task 3: Update PublicLayer diagonal-concat site
- [ ] In `_resolve_pending` (line 3810-3832): when a `PublicLayer` has `ly.data` that overlaps with the chart's columns, use the same Identity routing instead of diagonal concat
- [ ] Set `data_source` on the converted `_Layer`

### Task 4: Tests
- [ ] Update existing regression tests that assert column-rename behavior (`test_rule_layer_column_rename`) — the renamed columns will no longer exist; assert `data_source` routing instead
- [ ] Add `test_chained_3_layer_rule`: `scatter + hline + vline` — exactly 2 rule lines
- [ ] Add `test_chained_4_layer_mixed`: `scatter + hline + vline + label` — renders correctly
- [ ] Add `test_public_layer_data_overlap`: PublicLayer with overlapping columns routes correctly
- [ ] Verify all existing tests: `uv run pytest -x -q`
- [ ] Verify: `cargo test`

## 6. Acceptance checks

- `scatter + hline` renders 1 horizontal rule (2-layer, must not regress)
- `scatter + hline + vline` renders 1 horizontal + 1 vertical rule (3-layer chained)
- `scatter + hline + vline + label` renders correctly (4-layer chained)
- Same-data layering unchanged
- Transform-based layering (scatter + smooth) unchanged
- `calibration_chart` renders correctly
- All existing tests pass: `uv run pytest -x -q` and `cargo test`
- No column-rename artifacts (`__rhs_` suffixes) in any DataFrame or encoding
