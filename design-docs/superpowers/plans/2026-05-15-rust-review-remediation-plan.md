# Rust Review Remediation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

## 1. Objective

Address the 10 findings from the 2026-05-15 full Rust review: eliminate library panics, deduplicate repeated patterns across mark and transform modules, remove dead code, and narrow stale lint suppressions.

## 2. Spec references

- Rust review findings (this conversation — architecture map, drift report, refactor roadmap)
- `.claude/skills/rust-review/references/heuristics.md` — heuristics #1 (bool smell), #4 (parallel API drift), #6 (error fragmentation)

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `crates/ferrum-core/src/render/draw.rs` | Add `resolve_stroke_dash()` helper |
| Modify | `crates/ferrum-core/src/render/marks/point.rs` | Use shared dash helper |
| Modify | `crates/ferrum-core/src/render/marks/bar.rs` | Use shared dash helper |
| Modify | `crates/ferrum-core/src/render/marks/line.rs` | Use shared dash helper |
| Modify | `crates/ferrum-core/src/render/marks/rule.rs` | Use shared dash helper |
| Modify | `crates/ferrum-core/src/transform/linalg.rs` | Replace `assert_eq!` with `PyResult` |
| Modify | `crates/ferrum-core/src/scale/ticks.rs` | Remove blanket `#![allow(dead_code)]` |
| Modify | `crates/ferrum-wasm/src/selection_state.rs` | Remove dead `toggle_point` fn |
| Modify | `crates/ferrum-core/src/transform/*.rs` | Convert naked `.unwrap()` → `?` / `.ok_or_else()` in library code (~502 sites; test code excluded) |

## 4. Constraints

- **No behavior changes.** Every edit is a pure refactor — same inputs produce same outputs. The only observable difference: panics become `PyErr` propagation.
- **`cargo test` must pass** after every task (ferrum-core + ferrum-wasm).
- **No public API changes.** All new helpers are `pub(crate)`.
- **Don't touch test code.** `.unwrap()` in `#[cfg(test)]` blocks and test modules is fine — tests *should* panic on unexpected state.

## 5. Tasks

### Task 1: Extract `resolve_stroke_dash()` helper
- [ ] Add `pub(crate) fn resolve_stroke_dash(idx: f64) -> Option<Vec<f64>>` to `render/draw.rs`
- [ ] Replace inline match blocks in point.rs:54, bar.rs:72, line.rs:213, rule.rs:33 with calls to the shared helper
- [ ] Verify: `cargo test -p ferrum-core` — all mark tests pass

### Task 2: Fix `linalg::mat_from_flat` panic
- [ ] Replace `assert_eq!(data.len(), nrows * ncols, ...)` at linalg.rs:186 with a guard that returns `Err(PyValueError::new_err(...))`
- [ ] Change return type from `Mat<f64>` to `PyResult<Mat<f64>>`
- [ ] Update all callers (grep `mat_from_flat` in transform/) to propagate `?`
- [ ] Verify: `cargo test -p ferrum-core` — linalg and stats tests pass

### Task 3: Remove dead code and stale lint suppression
- [ ] Remove `#![allow(dead_code)]` from `scale/ticks.rs:3` — all functions are used; let the compiler verify
- [ ] Remove `#[allow(dead_code)] fn toggle_point(...)` at `ferrum-wasm/src/selection_state.rs:281` and any associated code
- [ ] Verify: `cargo test -p ferrum-core && cargo test -p ferrum-wasm` — no dead-code warnings, all tests pass

### Task 4: Convert transform `.unwrap()` → error propagation
- [ ] Systematic pass through all `crates/ferrum-core/src/transform/*.rs` files (excluding test blocks)
- [ ] For each naked `.unwrap()` on column access / downcast / map lookup: replace with `.ok_or_else(|| PyValueError::new_err("descriptive message"))?` or `.map_err(|e| ...)?`
- [ ] Preserve `.expect("invariant: ...")` where a prior guard truly guarantees the value (document the invariant in the expect message)
- [ ] Priority files (highest unwrap count): aggregate.rs, bin.rs, box_stats.rs, smooth.rs, raster.rs, violin.rs
- [ ] Verify: `cargo test -p ferrum-core` after each file; `cargo clippy -p ferrum-core -- -D warnings` clean at end

### Task 5: Convert render marks `.unwrap()` → error propagation
- [ ] Same pass for `crates/ferrum-core/src/render/marks/*.rs` (~95 sites across point, bar, rect, image, line, polygon, rule, ribbon, tick, text)
- [ ] Verify: `cargo test -p ferrum-core`; `cargo clippy -p ferrum-core -- -D warnings` clean

## 6. Acceptance checks

- `cargo test -p ferrum-core` — all pass
- `cargo test -p ferrum-wasm` — all pass (wasm target not required; native tests suffice)
- `cargo clippy -p ferrum-core -- -D warnings` — clean
- `cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings` — clean (WASM clippy per CLAUDE.md build commands)
- `grep -rn '\.unwrap()' crates/ferrum-core/src/transform/*.rs | grep -v test | wc -l` — significantly reduced from 502
- `grep -rn '#!\[allow(dead_code)\]' crates/ferrum-core/src/scale/ticks.rs` — no results
- `uv run pytest` — Python tests still pass (no behavior change)
