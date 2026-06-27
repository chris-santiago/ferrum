# ScaleSpec ↔ PyO3 *Scale Reconciliation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use chris-code:subagent-driven-development (recommended) or chris-code:executing-plans to implement this plan task-by-task.

## 1. Objective

Make `ScaleSpec` the single canonical scale representation by linking every `*Scale`
pyclass to it through an inherent Rust `to_scale_spec`, collapsing the `_scale_to_dict`
hand-bridge, and adding `Quantile`/`Threshold` wire variants — strictly non-breaking.

## 2. Spec references

- `design-docs/superpowers/specs/2026-06-27-scale-spec-reconciliation-design.md` — whole spec
- §6 Canonical interfaces — variant shapes, `to_scale_spec` / `_to_scale_spec_dict` contract, byte-identity correctness invariant
- §7 Invariants — non-breaking, render/`to_json` byte-identical, positional fallback parity
- §9 Acceptance criteria, §10 Validation strategy

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `crates/ferrum-core/src/spec/encoding.rs` | add `ScaleSpec::Quantile` + `Threshold` variants; Rust round-trip tests |
| Modify | `crates/ferrum-core/src/render/scale_resolve/positional.rs` | add `build_from_scale_spec` arms (Quantile/Threshold → Linear fallback) |
| Modify | `crates/ferrum-core/src/scale/linear.rs` | `to_scale_spec` + `_to_scale_spec_dict` for LinearScale |
| Modify | `crates/ferrum-core/src/scale/log.rs` | same for LogScale |
| Modify | `crates/ferrum-core/src/scale/time.rs` | same for TimeScale (utc flag → `Utc`/`Time` variant) |
| Modify | `crates/ferrum-core/src/scale/symlog.rs` | same for SymlogScale |
| Modify | `crates/ferrum-core/src/scale/pow.rs` | same for PowScale **and** SqrtScale |
| Modify | `crates/ferrum-core/src/scale/ordinal.rs` | same for OrdinalScale (polymorphic range) |
| Modify | `crates/ferrum-core/src/scale/band.rs` | same for BandScale (padding_inner/outer → variant) |
| Modify | `crates/ferrum-core/src/scale/point.rs` | same for PointScale |
| Modify | `crates/ferrum-core/src/scale/sequential.rs` | same for SequentialScale |
| Modify | `crates/ferrum-core/src/scale/diverging.rs` | same for DivergingScale |
| Modify | `crates/ferrum-core/src/scale/quantize.rs` | same for QuantizeScale |
| Modify | `crates/ferrum-core/src/scale/bin_ordinal.rs` | same for BinOrdinalScale |
| Modify | `crates/ferrum-core/src/scale/quantile.rs` | `to_scale_spec` + py wrapper for QuantileScale (new `Quantile` variant) |
| Modify | `crates/ferrum-core/src/scale/threshold.rs` | same for ThresholdScale (new `Threshold` variant) |
| Modify | `crates/ferrum-core/src/scale/mod.rs` | update the dual-representation doc note to reflect the now-enforced link |
| Modify | `src/ferrum/encoding/_scale.py` | collapse 13 isinstance branches to one delegation; keep dict/Parameter/None branches |
| Modify | `tests/test_phase_12_scales.py` | update wire-shape assertions to canonical serialization; add Quantile/Threshold |
| Test | `tests/test_scale_spec_parity.py` | drift guard: enumerate every `*Scale` pyclass → deserializable round-tripping `ScaleSpec`; byte-identity of `to_json()` per scale type |

## 4. Constraints

- **Strictly non-breaking.** No public symbol removed/renamed; no constructor signature changed. `ferrum-spec.md` is the API contract.
- **Render + `to_json()` byte-identical** for every pre-existing scale type. The correctness contract (spec §6): `to_scale_spec(s) == deserialize_as_ScaleSpec(old _scale_to_dict(s))` for each already-bridged class. Verify empirically, do not assume.
- **No new serialization helper** — reuse `encode_serde_value_for_py` (the existing `EncodingSpec.scale` getter serializer) for `_to_scale_spec_dict`.
- **Quantile/Threshold variant shape:** numeric `range: Option<Vec<f64>>` (NOT Quantize's `Option<Vec<String>>`); `domain: Option<Vec<f64>>`; serde tag `"quantile"`/`"threshold"` (enum uses `tag = "type", rename_all = "lowercase"`); computed `quantiles` NOT transmitted.
- **Positional fallback:** Quantile/Threshold resolve to `ScaleKind::Linear`, identical to the existing Quantize/Sequential/Diverging arm.
- **Out of scope (do not implement):** discrete-color binned rendering; `*Scale` compute-method math; storing `ScaleSpec` inside the `*Scale` structs. (spec §3)
- **`_scale_to_dict` dict / `Parameter` / `None` branches stay byte-for-byte unchanged.** Only the pyclass `isinstance` branches collapse.
- **Build/test commands** (CONDA_PREFIX conflict + macOS DYLD): rebuild ext with `unset CONDA_PREFIX && uv run --no-sync maturin develop`; Rust tests with `DYLD_LIBRARY_PATH=$(uv run python -c "import sys; print(sys.base_prefix + '/lib')") cargo test`; Python tests `uv run pytest -n auto`.
- **`cargo test` + `pytest -n auto` green** before the issue is closed (hard constraint).

## 5. Tasks

### Task 1: Add Quantile/Threshold wire variants + resolver arms (rust-coder)
- [ ] Add `ScaleSpec::Quantile` and `ScaleSpec::Threshold` per spec §6 (shape, serde attrs).
- [ ] Add `build_from_scale_spec` arms in `positional.rs`: both → `ScaleKind::Linear` (mirror the existing Quantize/Sequential/Diverging arm).
- [ ] Add Rust tests `scale_spec_quantile_round_trip` / `scale_spec_threshold_round_trip` mirroring `scale_spec_quantize_round_trip` (deserialize compact JSON, match variant, assert fields, re-serialize tag check).
- [ ] Verify: `DYLD_LIBRARY_PATH=$(uv run python -c "import sys; print(sys.base_prefix + '/lib')") cargo test` — compiles (exhaustive match satisfied) and new tests pass.

### Task 2: `to_scale_spec` + `_to_scale_spec_dict` on every *Scale pyclass (rust-coder)
- Consumes: `Quantile`/`Threshold` variants from Task 1; `encode_serde_value_for_py` from `spec/encoding.rs`.
- [ ] For each pyclass in §3 (Linear, Log, Time, Symlog, Pow, Sqrt, Ordinal, Band, Point, Sequential, Diverging, Quantize, BinOrdinal, Quantile, Threshold): add inherent `pub(crate) fn to_scale_spec(&self) -> ScaleSpec` reading its struct fields into the matching variant, satisfying the spec §6 correctness contract (reproduce the same `ScaleSpec` today's `_scale_to_dict` → deserialize yields, incl. Band's `padding`/inner/outer/align and Time's utc→`Utc`/`Time`).
- [ ] Add a `#[pymethods]` `_to_scale_spec_dict(&self, py) -> PyResult<Py<PyAny>>` per pyclass delegating to `encode_serde_value_for_py` over `to_scale_spec()`. (A shared macro/helper is fine if it keeps the per-class contract.)
- [ ] Update the `scale/mod.rs` doc note: the link is now compiler/test-enforced via `to_scale_spec`, not absent.
- [ ] Verify: rebuild `unset CONDA_PREFIX && uv run --no-sync maturin develop`; `cargo test` green.

### Task 3: Collapse the Python bridge + drift-guard + wire tests (python-coder)
- Consumes: `_to_scale_spec_dict` on each pyclass from Task 2 (extension rebuilt).
- [ ] Collapse `_scale_to_dict`: replace the 13 `isinstance(*Scale)` branches with a single `if hasattr(scale, "_to_scale_spec_dict"): return scale._to_scale_spec_dict()`. Leave dict/`Parameter`/`None` branches unchanged. Update the module docstring's scale-type list.
- [ ] Update `tests/test_phase_12_scales.py` wire-shape assertions to the canonical `ScaleSpec` serialization (e.g. Band now includes `padding`/`align`); add Quantile/Threshold `_scale_to_dict` cases.
- [ ] Create `tests/test_scale_spec_parity.py`: (a) enumerate every `*Scale` class exported from `ferrum._core`, construct a representative instance, assert `_scale_to_dict` yields a dict whose `type` builds a valid chart (no `TypeError`); (b) for one chart per pre-existing scale type, assert `to_json()` is byte-identical to a captured pre-change baseline (the §10 byte-identity guard).
- [ ] Add the encode-path smoke: `encode(color=fr.X(..., scale=fr.QuantileScale(...)))` and `ThresholdScale(...)` build + render without error.
- [ ] Verify: `uv run pytest -n auto tests/test_phase_12_scales.py tests/test_scale_spec_parity.py -v`.

## 6. Acceptance checks

- `DYLD_LIBRARY_PATH=$(uv run python -c "import sys; print(sys.base_prefix + '/lib')") cargo test` — all pass.
- `uv run pytest -n auto` — all pass.
- The reproduced `TypeError` on `QuantileScale`/`ThresholdScale` in `encode(scale=...)` is gone (builds + renders).
- `_scale_to_dict` contains no `isinstance(*Scale)` branch.
- `to_json()` byte-identical for Linear/Log/Band/Point/Ordinal/Sequential/Diverging/Quantize/BinOrdinal/Pow/Sqrt/Time/Symlog charts.
- Lite-review gates (`rust-review-lite`, `python-review-lite`) clean per commit.

## 7. Open questions

- (None blocking. Discrete-color rendering is a deliberate non-goal / logged follow-up — spec §3.)
