# Phase 12: Spec Completeness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

## 1. Objective

Close all five spec-vs-implementation gaps identified in the Phase 12 design spec, delivering the full `ferrum-spec.md` §3.5–§3.12 API surface.

## 2. Spec references

- `design-docs/superpowers/specs/2026-05-17-phase-12-spec-completeness-design.md` — all sections
- `ferrum-spec.md §3.5` — Data transforms
- `ferrum-spec.md §3.6` — Scale classes + color scheme constants
- `ferrum-spec.md §3.7` — Axis and Legend value classes
- `ferrum-spec.md §3.12` — Compound views (LayerChart, ConcatChart)

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Create | `crates/ferrum-core/src/transform/expr.rs` | Expression evaluator |
| Create | `crates/ferrum-core/src/transform/filter.rs` | transform_filter |
| Create | `crates/ferrum-core/src/transform/calculate.rs` | transform_calculate |
| Create | `crates/ferrum-core/src/transform/window.rs` | transform_window |
| Create | `crates/ferrum-core/src/transform/fold.rs` | transform_fold |
| Create | `crates/ferrum-core/src/transform/pivot.rs` | transform_pivot |
| Create | `crates/ferrum-core/src/transform/join_aggregate.rs` | transform_join_aggregate |
| Create | `crates/ferrum-core/src/transform/impute.rs` | transform_impute |
| Create | `crates/ferrum-core/src/transform/flatten.rs` | transform_flatten |
| Create | `crates/ferrum-core/src/transform/sample.rs` | transform_sample |
| Create | `crates/ferrum-core/src/transform/top_k.rs` | transform_top_k |
| Create | `crates/ferrum-core/src/transform/stack.rs` | transform_stack |
| Create | `crates/ferrum-core/src/transform/timeunit.rs` | transform_timeunit |
| Create | `crates/ferrum-core/src/transform/regression.rs` | transform_regression |
| Create | `crates/ferrum-core/src/transform/loess.rs` | transform_loess |
| Create | `crates/ferrum-core/src/transform/density_transform.rs` | transform_density (distinct from stat_kde) |
| Modify | `crates/ferrum-core/src/transform/core.rs` | Add 17 variants to `for_each_transform!` macro |
| Modify | `crates/ferrum-core/src/transform/mod.rs` | Register new modules |
| Create | `src/ferrum/transforms.py` | Python constructors for all 17 data transforms |
| Create | `crates/ferrum-core/src/scale/pow.rs` | ScalePow/ScaleSqrt |
| Create | `crates/ferrum-core/src/scale/band.rs` | ScaleBand/ScalePoint |
| Create | `crates/ferrum-core/src/scale/sequential.rs` | ScaleSequential |
| Create | `crates/ferrum-core/src/scale/diverging.rs` | ScaleDiverging |
| Create | `crates/ferrum-core/src/scale/quantize.rs` | ScaleQuantize/ScaleBinOrdinal |
| Modify | `crates/ferrum-core/src/scale/mod.rs` | Export new scale types |
| Modify | `src/ferrum/encoding/_scale.py` | Extend `_scale_to_dict()` for new types |
| Create | `src/ferrum/color.py` | `palette()`, `to_hex()`, `sequential()`, `diverging()` |
| Create | `src/ferrum/config.py` | `set()`, `get()`, `defaults()` context manager |
| Create | `src/ferrum/axis.py` | `Axis` frozen dataclass |
| Create | `src/ferrum/legend.py` | `Legend` frozen dataclass |
| Modify | `src/ferrum/composition.py` | Add `LayerChart`, `ConcatChart` |
| Modify | `src/ferrum/__init__.py` | Export all new public API |
| Modify | `src/ferrum/encoding/base.py` | Accept `Axis`/`Legend` instances (not just dicts) |
| Create | `tests/test_phase_12_transforms.py` | Data transform tests |
| Create | `tests/test_phase_12_scales.py` | Scale class tests |
| Create | `tests/test_phase_12_color_config.py` | Color + config module tests |
| Create | `tests/test_phase_12_composition.py` | LayerChart/ConcatChart tests |
| Create | `tests/test_phase_12_axis_legend.py` | Axis/Legend value class tests |

## 4. Constraints

- **Dispatch rule:** All `.rs` code via `rust-coder` agent; all `.py` code via `python-coder` agent. No general-purpose agents write code.
- **Review gates:** After each task's code is staged, dispatch `rust-review-lite` (for `.rs`) or `python-review-lite` (for `.py`) before committing.
- **Bug fixes → regression tests:** Any bug discovered during implementation triggers the `regression-test` skill before moving on.
- **Docstrings:** All new public API follows `ferrum-docstrings` skill (NumPy convention, PyO3 placement).
- **Branch:** All work on `feat/phase-12-spec-completeness` off `main`.
- **No matplotlib.** Color utilities use the existing Rust palette registry only.
- **Expression evaluator is sandboxed.** No file I/O, no imports, no arbitrary function calls. `ValueError` at parse time for invalid input.
- **Macro pattern.** New transforms extend `for_each_transform!` — do not add ad-hoc match arms.
- **Backward compat.** Existing `scale=LinearScale(...)`, `axis={"title": ...}` dict usage must continue working unchanged.
- **Arrow CDI boundary.** Data transforms run on Arrow arrays in Rust. No row-level Python iteration.

## 5. Tasks

### Task 1: Branch + expression evaluator (Rust)

- [ ] Create branch `feat/phase-12-spec-completeness` from `main`
- [ ] Implement `expr.rs`: recursive-descent parser for Vega-style expressions (spec §5 architecture)
- [ ] Support: `datum.field`, `datum["field"]`, literals, arithmetic, comparison, logical, ternary
- [ ] Reject: imports, function calls, file I/O — `ValueError` at parse time
- [ ] Unit tests in `crates/ferrum-core/src/transform/expr.rs` (inline `#[cfg(test)]`)
- [ ] Verify: `cargo test -p ferrum-core -- expr`
- [ ] **Gate:** `rust-review-lite`

### Task 2: Data transforms — Rust core (Rust)

- [ ] Add 17 variants to `for_each_transform!` macro table in `core.rs`
- [ ] Implement each transform module (spec §3.5 for parameters, spec §5 for architecture)
- [ ] `transform_filter` and `transform_calculate` consume `expr.rs` evaluator
- [ ] `transform_window`: rolling aggregates with `frame`, `groupby`, rank functions
- [ ] `transform_stack`: `offset` modes (zero, normalize, center)
- [ ] Verify: `cargo test -p ferrum-core`
- [ ] **Gate:** `rust-review-lite`

### Task 3: Data transforms — Python API (Python)

- [ ] Create `src/ferrum/transforms.py` with all 17 constructor functions (spec §6 interfaces)
- [ ] Each returns a dict matching the Rust `TransformSpec` serde contract
- [ ] Wire `Chart.transform(*transforms)` to prepend data transforms before stat transforms
- [ ] Docstrings per `ferrum-docstrings` skill
- [ ] Export from `src/ferrum/__init__.py`
- [ ] Write `tests/test_phase_12_transforms.py` — one test per transform with hand-computed expected output
- [ ] Verify: `uv run pytest tests/test_phase_12_transforms.py -v`
- [ ] **Gate:** `python-review-lite`

### Task 4: Scale types — Rust (Rust)

- [ ] Implement `pow.rs` (ScalePow, ScaleSqrt), `band.rs` (ScaleBand, ScalePoint), `sequential.rs`, `diverging.rs`, `quantize.rs` (ScaleQuantize, ScaleBinOrdinal)
- [ ] Existing `QuantileScale` and `ThresholdScale` already imported — verify PyO3 bindings are complete
- [ ] All new scales expose `to_dict()` for renderer consumption
- [ ] Verify: `cargo test -p ferrum-core -- scale`
- [ ] **Gate:** `rust-review-lite`

### Task 5: Scale types — Python wiring (Python)

- [ ] Extend `_scale_to_dict()` in `encoding/_scale.py` for new types
- [ ] Export all new scale classes from `__init__.py`
- [ ] `ScaleUtc` implemented as `ScaleTime(utc=True)` per spec open question recommendation
- [ ] Docstrings per `ferrum-docstrings` skill
- [ ] Write `tests/test_phase_12_scales.py` — construct, serialize, verify renderer accepts
- [ ] Verify: `uv run pytest tests/test_phase_12_scales.py -v`
- [ ] **Gate:** `python-review-lite`

### Task 6: `ferrum.color` + `ferrum.config` modules (Python)

- [ ] Create `src/ferrum/color.py` wrapping Rust `ContinuousScheme` + categorical registry (spec §4, §6)
- [ ] Create `src/ferrum/config.py` with `contextvars.ContextVar` store (spec §4, §5)
- [ ] Config precedence: explicit per-chart > config defaults > built-in defaults
- [ ] Export from `__init__.py`
- [ ] Docstrings per `ferrum-docstrings` skill
- [ ] Write `tests/test_phase_12_color_config.py` — palette access, config context-manager isolation, threading
- [ ] Verify: `uv run pytest tests/test_phase_12_color_config.py -v`
- [ ] **Gate:** `python-review-lite`

### Task 7: `LayerChart` + `ConcatChart` (Python)

- [ ] Add `LayerChart` to `composition.py` — shared-viewBox SVG overlay, `+` operator (spec §4, §6)
- [ ] Add `ConcatChart` to `composition.py` — wrapping grid via `compose_svg_grid`, auto `columns` (spec §4, §6)
- [ ] Both support `resolve=`, `title=`, `.theme()`, `.properties()`, `.save()`, `.show()`
- [ ] Export from `__init__.py`
- [ ] Docstrings per `ferrum-docstrings` skill
- [ ] Write `tests/test_phase_12_composition.py` — layer overlay, concat wrapping, resolve behavior
- [ ] Golden SVGs for both; visually inspect via `snapshot-goldens.py`
- [ ] Verify: `uv run pytest tests/test_phase_12_composition.py -v`
- [ ] **Gate:** `python-review-lite`

### Task 8: `Axis` + `Legend` value classes (Python)

- [ ] Create `src/ferrum/axis.py` — frozen dataclass with all §3.7 parameters
- [ ] Create `src/ferrum/legend.py` — frozen dataclass with all §3.7 parameters
- [ ] Modify `encoding/base.py` to accept `Axis`/`Legend` instances alongside dicts
- [ ] `axis=False` shorthand → `Axis(domain=False, ticks=False, labels=False, title=None, grid=False)`
- [ ] Export from `__init__.py`
- [ ] Docstrings per `ferrum-docstrings` skill
- [ ] Write `tests/test_phase_12_axis_legend.py` — serialization, renderer integration, backward compat with dicts
- [ ] Verify: `uv run pytest tests/test_phase_12_axis_legend.py -v`
- [ ] **Gate:** `python-review-lite`

### Task 9: Integration + full suite

- [ ] `unset CONDA_PREFIX && uv run --no-sync maturin develop`
- [ ] `uv run pytest -n auto` — full suite, no regressions
- [ ] `cargo test` — all Rust tests pass
- [ ] Verify all 11 acceptance criteria from spec §9

## 6. Acceptance checks

- `cargo test -p ferrum-core` — all pass (including expr evaluator, 17 transforms, new scales)
- `uv run pytest -n auto` — full suite passes, zero regressions
- `from ferrum import transform_filter, ScaleBand, Axis, Legend, LayerChart, ConcatChart` — importable
- `ferrum.color.palette("tableau10")` returns 10 hex strings
- `with ferrum.config.defaults(width=800): ...` scopes correctly
- Golden SVGs for LayerChart/ConcatChart visually inspected

## 7. Open questions

- Expression grammar: support `datum["field with spaces"]` bracket notation? (Spec recommends yes.)
- `ScaleUtc`: separate class or `ScaleTime(utc=True)` flag? (Spec recommends flag.)
