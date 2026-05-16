# Phase 4 — Scale Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land seven scale primitives (`LinearScale`, `LogScale`, `TimeScale`, `SymlogScale`, `OrdinalScale`, `QuantileScale`, `ThresholdScale`) as `#[pyclass]` types in `ferrum._core`, each providing domain/range mapping, inversion, tick generation, and `nice()` where meaningful.

**Architecture:** A sealed internal `enum Scale { Linear{..}, Log{..}, ... }` lives in `crates/ferrum-core/src/scale/core.rs` and centralizes all math via match-dispatch. Seven thin pyclass facade structs (one per file) wrap an enum variant and expose a typed Python constructor + the methods that apply to that variant's group (continuous / ordinal / bin). Shared tick infrastructure lives in `crates/ferrum-core/src/scale/ticks.rs` so Phase 5 (binning) can later reuse the Sturges helper.

**Tech Stack:** Rust 2021, PyO3 0.28 (abi3-py310), `serde` is already in workspace deps but not used in this phase, std-only math (no `chrono`, no `nalgebra`), pytest 8.

**Spec:** [`docs/superpowers/specs/2026-05-09-scale-engine-design.md`](../specs/2026-05-09-scale-engine-design.md)

**TDD posture:** Task B1 is strict test-first (write `unimplemented!()` stubs, watch tests panic, then implement). Tasks C1–D3 ship the variant's tests *alongside* its implementation in a single `core.rs` edit — the project precedent set by Phase 3. If using `superpowers:subagent-driven-development`, treat the alongside-tests pattern as the agreed deviation; if using `superpowers:executing-plans`, no special handling needed.

---

## Build commands (memorize these — every step uses them)

| Action | Command |
|---|---|
| Rebuild Python extension | `unset CONDA_PREFIX && uv run --no-sync maturin develop` |
| Rust-side tests | `DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core` |
| Python tests | `uv run pytest` |
| Smoke verify (extends Phase 3) | `unset CONDA_PREFIX && uv run --no-sync python -c "from ferrum._core import LinearScale; s = LinearScale(domain=[0.0, 10.0], range=[0.0, 1.0]); assert abs(s.scale(5.0) - 0.5) < 1e-12; print('OK')"` |

If `cargo` isn't on PATH, run `source ~/.cargo/env` first.

---

## File structure (lock this in before starting)

| File | Purpose |
|---|---|
| `crates/ferrum-core/src/lib.rs` | Add `mod scale;` and seven `m.add_class::<...>()` registrations |
| `crates/ferrum-core/src/scale/mod.rs` | Submodule declarations; no re-exports outside the crate |
| `crates/ferrum-core/src/scale/core.rs` | `Scale` enum + impl Scale (math dispatch, validators) |
| `crates/ferrum-core/src/scale/ticks.rs` | `sturges_floor`, `nice_step`, `nice_ticks`, `nice_time_interval_ms` |
| `crates/ferrum-core/src/scale/linear.rs` | `LinearScale` pyclass facade |
| `crates/ferrum-core/src/scale/log.rs` | `LogScale` pyclass facade |
| `crates/ferrum-core/src/scale/time.rs` | `TimeScale` pyclass facade |
| `crates/ferrum-core/src/scale/symlog.rs` | `SymlogScale` pyclass facade |
| `crates/ferrum-core/src/scale/ordinal.rs` | `OrdinalScale` pyclass facade |
| `crates/ferrum-core/src/scale/quantile.rs` | `QuantileScale` pyclass facade |
| `crates/ferrum-core/src/scale/threshold.rs` | `ThresholdScale` pyclass facade |
| `src/ferrum/_core.pyi` | Add seven scale class stubs with typed signatures |
| `src/ferrum/__init__.py` | Re-export the seven classes at `ferrum.` level |
| `tests/test_scales.py` | Python-level smoke and boundary tests |
| `docs/superpowers/ferrum-phases.md` | Mark Phase 4 status `done`, link this plan |

---

## Task index

- [Section A — Setup](#section-a--setup)
  - [Task A1](#task-a1-verify-baseline-tests-pass-create-feature-branch)
  - [Task A2](#task-a2-create-scale-module-skeleton)
- [Section B — Shared infrastructure](#section-b--shared-infrastructure)
  - [Task B1](#task-b1-implement-scaleticksrs-with-tdd)
- [Section C — Continuous scales](#section-c--continuous-scales)
  - [Task C1](#task-c1-linearscale)
  - [Task C2](#task-c2-logscale)
  - [Task C3](#task-c3-symlogscale)
  - [Task C4](#task-c4-timescale)
- [Section D — Discrete scales](#section-d--discrete-scales)
  - [Task D1](#task-d1-ordinalscale)
  - [Task D2](#task-d2-thresholdscale)
  - [Task D3](#task-d3-quantilescale)
- [Section E — Python boundary](#section-e--python-boundary)
  - [Task E1](#task-e1-update-_corepyi)
  - [Task E2](#task-e2-re-export-from-ferruminit)
  - [Task E3](#task-e3-write-testtest_scalespy)
- [Section F — Closure](#section-f--closure)
  - [Task F1](#task-f1-update-ferrum-phasesmd)
  - [Task F2](#task-f2-final-verification)

---

## Section A — Setup

### Task A1: Verify baseline tests pass, create feature branch

**Files:** none modified.

- [ ] **Step 1: Confirm working tree is clean and on `main`**

```bash
git status
```

Expected: `On branch main`, `nothing to commit, working tree clean`.

- [ ] **Step 2: Run the existing Rust test suite**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core
```

Expected: 24 tests pass. If any fail, stop and investigate before proceeding.

- [ ] **Step 3: Run the existing Python test suite**

```bash
uv run pytest
```

Expected: 18 tests pass.

- [ ] **Step 4: Create the feature branch**

```bash
git checkout -b feat/scale-engine
```

Expected: `Switched to a new branch 'feat/scale-engine'`.

- [ ] **Step 5: No commit yet — Task A2 lands the first commit on this branch.**

---

### Task A2: Create scale module skeleton

**Files:**
- Create: `crates/ferrum-core/src/scale/mod.rs`
- Create: `crates/ferrum-core/src/scale/core.rs` (empty for now)
- Create: `crates/ferrum-core/src/scale/ticks.rs` (empty for now)
- Modify: `crates/ferrum-core/src/lib.rs`

- [ ] **Step 1: Create `crates/ferrum-core/src/scale/mod.rs`**

```rust
pub(crate) mod core;
pub(crate) mod ticks;
```

- [ ] **Step 2: Create `crates/ferrum-core/src/scale/core.rs` (placeholder)**

```rust
//! Sealed `Scale` enum that centralises math for every scale variant.
//! Each task in section C/D extends this with a new variant and the
//! corresponding match arms in the dispatch methods.

#![allow(dead_code)] // populated incrementally; suppress until first variant lands
```

- [ ] **Step 3: Create `crates/ferrum-core/src/scale/ticks.rs` (placeholder)**

```rust
//! Shared tick-generation and binning helpers.
//! Populated by Task B1.

#![allow(dead_code)]
```

- [ ] **Step 4: Update `crates/ferrum-core/src/lib.rs`**

Replace the file contents with:

```rust
use pyo3::prelude::*;

mod transport;
mod spec;
mod scale;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(transport::process_batch, m)?)?;
    m.add_class::<spec::chart::ChartSpec>()?;
    m.add_class::<spec::encoding::EncodingSpec>()?;
    Ok(())
}
```

The new line is `mod scale;`. The seven `m.add_class::<scale::...>()?` registrations land one-by-one in Tasks C1 through D3 as each pyclass is implemented.

- [ ] **Step 5: Build to confirm the skeleton compiles**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
```

Expected: build succeeds. Warnings about unused module are acceptable; the `#![allow(dead_code)]` suppresses them.

- [ ] **Step 6: Run the full test suite to confirm no regressions**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core
uv run pytest
```

Expected: 24 Rust + 18 Python pass.

- [ ] **Step 7: Commit**

```bash
git add crates/ferrum-core/src/scale crates/ferrum-core/src/lib.rs
git commit -m "feat(scale): add empty scale module skeleton"
```

---

## Section B — Shared infrastructure

### Task B1: Implement `scale/ticks.rs` with TDD

**Files:**
- Modify: `crates/ferrum-core/src/scale/ticks.rs`

The four helpers in this file are pure math with no PyO3 surface — perfect TDD targets. We write tests first, then implement.

- [ ] **Step 1: Write failing tests in `scale/ticks.rs`**

Replace the placeholder content with the test module first (no implementation yet):

```rust
//! Shared tick-generation and binning helpers.

#![allow(dead_code)]

pub(crate) fn sturges_floor(_n: usize) -> usize {
    unimplemented!()
}

pub(crate) fn nice_step(_d_lo: f64, _d_hi: f64, _count: usize) -> f64 {
    unimplemented!()
}

pub(crate) fn nice_ticks(_d_lo: f64, _d_hi: f64, _count: usize) -> Vec<f64> {
    unimplemented!()
}

pub(crate) fn nice_time_interval_ms(_span_ms: f64, _count: usize) -> f64 {
    unimplemented!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sturges_floor_known_values() {
        // ceil(log2(n) + 1)
        assert_eq!(sturges_floor(0), 1);
        assert_eq!(sturges_floor(1), 1);
        assert_eq!(sturges_floor(2), 2);
        assert_eq!(sturges_floor(8), 4);
        assert_eq!(sturges_floor(10), 5);    // ceil(log2(10)+1) = ceil(4.32) = 5
        assert_eq!(sturges_floor(100), 8);   // ceil(log2(100)+1) = ceil(7.64) = 8
        assert_eq!(sturges_floor(1024), 11);
    }

    #[test]
    fn test_sturges_floor_returns_at_least_one() {
        assert!(sturges_floor(0) >= 1);
        assert!(sturges_floor(1) >= 1);
    }

    #[test]
    fn test_nice_step_simple_decades() {
        // Span 10, target 10 ticks → step ≈ 1.0
        let s = nice_step(0.0, 10.0, 10);
        assert!((s - 1.0).abs() < 1e-12, "got {s}");

        // Span 100, target 10 ticks → step ≈ 10.0
        let s = nice_step(0.0, 100.0, 10);
        assert!((s - 10.0).abs() < 1e-12, "got {s}");

        // Span 1, target 5 ticks → step ≈ 0.2 → nice round-up to 0.2
        let s = nice_step(0.0, 1.0, 5);
        assert!((s - 0.2).abs() < 1e-12, "got {s}");
    }

    #[test]
    fn test_nice_step_handles_zero_span() {
        assert_eq!(nice_step(5.0, 5.0, 10), 0.0);
    }

    #[test]
    fn test_nice_step_handles_invalid_inputs() {
        assert!(nice_step(0.0, 10.0, 0).is_nan());
        assert!(nice_step(f64::NAN, 10.0, 5).is_nan());
        assert!(nice_step(0.0, f64::INFINITY, 5).is_nan());
    }

    #[test]
    fn test_nice_ticks_inclusive_endpoints() {
        let ticks = nice_ticks(0.0, 10.0, 10);
        assert_eq!(ticks.first().copied(), Some(0.0));
        assert_eq!(ticks.last().copied(), Some(10.0));
        assert_eq!(ticks.len(), 11);
    }

    #[test]
    fn test_nice_ticks_count_approx() {
        let ticks = nice_ticks(0.0, 100.0, 10);
        assert!(ticks.len() >= 5 && ticks.len() <= 15, "got {} ticks: {ticks:?}", ticks.len());
    }

    #[test]
    fn test_nice_ticks_descending_input_descending_output() {
        let ticks = nice_ticks(10.0, 0.0, 10);
        assert!(ticks.first().copied().unwrap() > ticks.last().copied().unwrap());
    }

    #[test]
    fn test_nice_ticks_zero_span_returns_singleton() {
        let ticks = nice_ticks(5.0, 5.0, 10);
        assert_eq!(ticks, vec![5.0]);
    }

    #[test]
    fn test_nice_time_interval_returns_second_for_small_spans() {
        // 10s span, 10 ticks → 1s interval
        let iv = nice_time_interval_ms(10_000.0, 10);
        assert_eq!(iv, 1_000.0);
    }

    #[test]
    fn test_nice_time_interval_returns_day_for_week_span() {
        // 7d span, 7 ticks → 1d interval
        let iv = nice_time_interval_ms(7.0 * 24.0 * 3600_000.0, 7);
        assert_eq!(iv, 24.0 * 3600_000.0);
    }

    #[test]
    fn test_nice_time_interval_invalid_inputs() {
        assert!(nice_time_interval_ms(0.0, 5).is_nan());
        assert!(nice_time_interval_ms(-1.0, 5).is_nan());
        assert!(nice_time_interval_ms(1000.0, 0).is_nan());
        assert!(nice_time_interval_ms(f64::NAN, 5).is_nan());
    }
}
```

- [ ] **Step 2: Run tests; verify they fail with `unimplemented!()` panics**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core ticks::
```

Expected: each test panics on `unimplemented!`.

- [ ] **Step 3: Implement `sturges_floor`**

Replace the `sturges_floor` body in `scale/ticks.rs`:

```rust
pub(crate) fn sturges_floor(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let v = ((n as f64).log2() + 1.0).ceil();
    if v < 1.0 { 1 } else { v as usize }
}
```

- [ ] **Step 4: Implement `nice_step`**

Replace the `nice_step` body:

```rust
pub(crate) fn nice_step(d_lo: f64, d_hi: f64, count: usize) -> f64 {
    if count == 0 || !d_lo.is_finite() || !d_hi.is_finite() {
        return f64::NAN;
    }
    let span = (d_hi - d_lo).abs();
    if span == 0.0 {
        return 0.0;
    }
    let step0 = span / (count as f64);
    let exp = step0.log10().floor();
    let pow10 = 10f64.powf(exp);
    let frac = step0 / pow10;
    let nice_frac = if frac >= 7.5 {
        10.0
    } else if frac >= 3.5 {
        5.0
    } else if frac >= 1.5 {
        2.0
    } else {
        1.0
    };
    nice_frac * pow10
}
```

- [ ] **Step 5: Implement `nice_ticks`**

Replace the `nice_ticks` body:

```rust
pub(crate) fn nice_ticks(d_lo: f64, d_hi: f64, count: usize) -> Vec<f64> {
    if count == 0 || !d_lo.is_finite() || !d_hi.is_finite() {
        return Vec::new();
    }
    let (lo, hi, reverse) = if d_lo <= d_hi {
        (d_lo, d_hi, false)
    } else {
        (d_hi, d_lo, true)
    };
    if lo == hi {
        return vec![lo];
    }
    let step = nice_step(lo, hi, count);
    if !step.is_finite() || step == 0.0 {
        return vec![lo];
    }
    let start = (lo / step).ceil() * step;
    let end = (hi / step).floor() * step;
    let n_steps = ((end - start) / step).round() as i64;
    if n_steps < 0 {
        return Vec::new();
    }
    let n = (n_steps + 1) as usize;
    let mut out: Vec<f64> = (0..n).map(|i| start + (i as f64) * step).collect();
    if reverse {
        out.reverse();
    }
    out
}
```

- [ ] **Step 6: Implement `nice_time_interval_ms`**

Replace the `nice_time_interval_ms` body:

```rust
pub(crate) fn nice_time_interval_ms(span_ms: f64, count: usize) -> f64 {
    const SECOND: f64 = 1_000.0;
    const MINUTE: f64 = 60.0 * SECOND;
    const HOUR:   f64 = 60.0 * MINUTE;
    const DAY:    f64 = 24.0 * HOUR;
    const WEEK:   f64 = 7.0 * DAY;
    const MONTH:  f64 = 30.0 * DAY;   // approximate; calendar-aware deferred
    const YEAR:   f64 = 365.0 * DAY;  // approximate

    if count == 0 || !span_ms.is_finite() || span_ms <= 0.0 {
        return f64::NAN;
    }
    let target = span_ms / count as f64;
    let candidates: [f64; 19] = [
        SECOND, 5.0 * SECOND, 15.0 * SECOND, 30.0 * SECOND,
        MINUTE, 5.0 * MINUTE, 15.0 * MINUTE, 30.0 * MINUTE,
        HOUR, 3.0 * HOUR, 6.0 * HOUR, 12.0 * HOUR,
        DAY, 2.0 * DAY,
        WEEK,
        MONTH, 3.0 * MONTH, 6.0 * MONTH,
        YEAR,
    ];
    // Pick the largest candidate ≤ target; if none, return the smallest.
    let mut chosen = candidates[0];
    for &c in candidates.iter() {
        if c <= target {
            chosen = c;
        } else {
            break;
        }
    }
    chosen
}
```

- [ ] **Step 7: Run the tick tests; verify they pass**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core ticks::
```

Expected: 11 tests pass.

- [ ] **Step 8: Run full Rust suite to confirm no regressions**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core
```

Expected: 35 tests pass (24 baseline + 11 new).

- [ ] **Step 9: Commit**

```bash
git add crates/ferrum-core/src/scale/ticks.rs
git commit -m "feat(scale): tick helpers (sturges, nice_step, nice_ticks, nice_time_interval_ms)"
```

---

## Section C — Continuous scales

### Task C1: LinearScale

This task is the template the other six scale tasks follow. Reading subsequent tasks before completing C1 is fine, but C1 is the longest because it lands the `Scale` enum scaffold.

**Files:**
- Modify: `crates/ferrum-core/src/scale/core.rs`
- Create: `crates/ferrum-core/src/scale/linear.rs`
- Modify: `crates/ferrum-core/src/scale/mod.rs`
- Modify: `crates/ferrum-core/src/lib.rs`

- [ ] **Step 1: Replace `crates/ferrum-core/src/scale/core.rs` with the enum scaffold + Linear arm + tests**

```rust
//! Sealed `Scale` enum that centralises math for every scale variant.
//! Each scale-task in section C/D extends this with a new variant.

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use super::ticks::{nice_step, nice_ticks};

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Scale {
    Linear { domain: [f64; 2], range: [f64; 2], clamp: bool },
}

impl Scale {
    pub(crate) fn scale_f64(&self, x: f64) -> f64 {
        match self {
            Scale::Linear { domain, range, clamp } => {
                if x.is_nan() { return f64::NAN; }
                let [d0, d1] = *domain;
                let [r0, r1] = *range;
                let t = (x - d0) / (d1 - d0);
                let mapped = r0 + t * (r1 - r0);
                if *clamp {
                    let (lo, hi) = if r0 <= r1 { (r0, r1) } else { (r1, r0) };
                    mapped.clamp(lo, hi)
                } else if x < d0.min(d1) || x > d0.max(d1) {
                    f64::NAN
                } else {
                    mapped
                }
            }
        }
    }

    pub(crate) fn invert_f64(&self, y: f64) -> f64 {
        match self {
            Scale::Linear { domain, range, clamp } => {
                if y.is_nan() { return f64::NAN; }
                let [d0, d1] = *domain;
                let [r0, r1] = *range;
                let t = (y - r0) / (r1 - r0);
                let mapped = d0 + t * (d1 - d0);
                if *clamp {
                    let (lo, hi) = if d0 <= d1 { (d0, d1) } else { (d1, d0) };
                    mapped.clamp(lo, hi)
                } else if y < r0.min(r1) || y > r0.max(r1) {
                    f64::NAN
                } else {
                    mapped
                }
            }
        }
    }

    pub(crate) fn ticks(&self, count: Option<usize>) -> Vec<f64> {
        match self {
            Scale::Linear { domain, .. } => {
                nice_ticks(domain[0], domain[1], count.unwrap_or(10))
            }
        }
    }

    pub(crate) fn nice(self) -> Self {
        match self {
            Scale::Linear { domain, range, clamp } => {
                let step = nice_step(domain[0], domain[1], 10);
                if !step.is_finite() || step == 0.0 {
                    return Scale::Linear { domain, range, clamp };
                }
                let lo_min = domain[0].min(domain[1]);
                let hi_max = domain[0].max(domain[1]);
                let nice_lo = (lo_min / step).floor() * step;
                let nice_hi = (hi_max / step).ceil() * step;
                let new_domain = if domain[0] <= domain[1] {
                    [nice_lo, nice_hi]
                } else {
                    [nice_hi, nice_lo]
                };
                Scale::Linear { domain: new_domain, range, clamp }
            }
        }
    }
}

// ---------- validators (used by pyclass facades) ----------

pub(crate) fn validate_finite(name: &str, values: &[f64]) -> PyResult<()> {
    for v in values {
        if !v.is_finite() {
            return Err(PyValueError::new_err(format!(
                "{name} must contain only finite values; found {v}"
            )));
        }
    }
    Ok(())
}

pub(crate) fn validate_continuous_pair(domain: &[f64], range: &[f64]) -> PyResult<()> {
    if domain.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "domain must have length 2; got {}",
            domain.len()
        )));
    }
    if range.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "range must have length 2; got {}",
            range.len()
        )));
    }
    validate_finite("domain", domain)?;
    validate_finite("range", range)?;
    if domain[0] == domain[1] {
        return Err(PyValueError::new_err(
            "domain endpoints must differ (lo != hi)",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_scale_basic() {
        let s = Scale::Linear { domain: [0.0, 10.0], range: [0.0, 1.0], clamp: false };
        assert!((s.scale_f64(5.0) - 0.5).abs() < 1e-12);
        assert!((s.scale_f64(0.0) - 0.0).abs() < 1e-12);
        assert!((s.scale_f64(10.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_linear_inversion_round_trip() {
        let s = Scale::Linear { domain: [-50.0, 50.0], range: [0.0, 100.0], clamp: false };
        for x in [-50.0, -25.0, 0.0, 17.5, 50.0] {
            let y = s.scale_f64(x);
            let back = s.invert_f64(y);
            assert!((back - x).abs() < 1e-9, "round-trip failed at x={x}: got {back}");
        }
    }

    #[test]
    fn test_linear_out_of_domain_returns_nan_when_unclamped() {
        let s = Scale::Linear { domain: [0.0, 10.0], range: [0.0, 1.0], clamp: false };
        assert!(s.scale_f64(-1.0).is_nan());
        assert!(s.scale_f64(11.0).is_nan());
    }

    #[test]
    fn test_linear_clamp_clamps_output() {
        let s = Scale::Linear { domain: [0.0, 10.0], range: [0.0, 1.0], clamp: true };
        assert_eq!(s.scale_f64(-1.0), 0.0);
        assert_eq!(s.scale_f64(11.0), 1.0);
    }

    #[test]
    fn test_linear_nan_propagates() {
        let s = Scale::Linear { domain: [0.0, 10.0], range: [0.0, 1.0], clamp: false };
        assert!(s.scale_f64(f64::NAN).is_nan());
        assert!(s.invert_f64(f64::NAN).is_nan());
    }

    #[test]
    fn test_linear_ticks_default_count() {
        let s = Scale::Linear { domain: [0.0, 10.0], range: [0.0, 1.0], clamp: false };
        let t = s.ticks(None);
        assert!(t.len() >= 5, "got {} ticks: {t:?}", t.len());
    }

    #[test]
    fn test_linear_nice_idempotent() {
        let s = Scale::Linear { domain: [0.13, 9.7], range: [0.0, 1.0], clamp: false };
        let n1 = s.clone().nice();
        let n2 = n1.clone().nice();
        assert_eq!(n1, n2);
    }

    #[test]
    fn test_validate_continuous_pair_rejects_wrong_length() {
        assert!(validate_continuous_pair(&[0.0], &[0.0, 1.0]).is_err());
        assert!(validate_continuous_pair(&[0.0, 1.0], &[]).is_err());
    }

    #[test]
    fn test_validate_continuous_pair_rejects_degenerate_domain() {
        assert!(validate_continuous_pair(&[5.0, 5.0], &[0.0, 1.0]).is_err());
    }

    #[test]
    fn test_validate_continuous_pair_rejects_non_finite() {
        assert!(validate_continuous_pair(&[0.0, f64::NAN], &[0.0, 1.0]).is_err());
        assert!(validate_continuous_pair(&[0.0, 10.0], &[f64::INFINITY, 1.0]).is_err());
    }
}
```

- [ ] **Step 2: Run the new tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core scale::core::
```

Expected: 10 tests pass.

- [ ] **Step 3: Create `crates/ferrum-core/src/scale/linear.rs`**

```rust
use pyo3::prelude::*;

use super::core::{validate_continuous_pair, Scale};

#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, PartialEq)]
pub struct LinearScale(Scale);

impl LinearScale {
    pub(crate) fn repr_string(&self) -> String {
        match &self.0 {
            Scale::Linear { domain, range, clamp } => format!(
                "LinearScale(domain=[{}, {}], range=[{}, {}], clamp={})",
                domain[0], domain[1], range[0], range[1], if *clamp { "True" } else { "False" }
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

#[pymethods]
impl LinearScale {
    #[new]
    #[pyo3(signature = (*, domain, range, clamp = false, nice = false))]
    fn new(domain: Vec<f64>, range: Vec<f64>, clamp: bool, nice: bool) -> PyResult<Self> {
        validate_continuous_pair(&domain, &range)?;
        let mut s = Scale::Linear {
            domain: [domain[0], domain[1]],
            range:  [range[0],  range[1]],
            clamp,
        };
        if nice {
            s = s.nice();
        }
        Ok(LinearScale(s))
    }

    fn scale(&self, x: f64) -> f64 {
        self.0.scale_f64(x)
    }

    fn invert(&self, y: f64) -> f64 {
        self.0.invert_f64(y)
    }

    #[pyo3(signature = (count = 10))]
    fn ticks(&self, count: usize) -> Vec<f64> {
        self.0.ticks(Some(count))
    }

    fn nice(&self) -> Self {
        LinearScale(self.0.clone().nice())
    }

    #[getter]
    fn domain(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Linear { domain, .. } => domain.to_vec(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    #[getter]
    fn range(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Linear { range, .. } => range.to_vec(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    #[getter]
    fn clamp(&self) -> bool {
        match &self.0 {
            Scale::Linear { clamp, .. } => *clamp,
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    fn __repr__(&self) -> String {
        self.repr_string()
    }
}
```

- [ ] **Step 4: Update `crates/ferrum-core/src/scale/mod.rs`**

```rust
pub(crate) mod core;
pub(crate) mod ticks;
pub(crate) mod linear;
```

- [ ] **Step 5: Update `crates/ferrum-core/src/lib.rs`**

```rust
use pyo3::prelude::*;

mod transport;
mod spec;
mod scale;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(transport::process_batch, m)?)?;
    m.add_class::<spec::chart::ChartSpec>()?;
    m.add_class::<spec::encoding::EncodingSpec>()?;
    m.add_class::<scale::linear::LinearScale>()?;
    Ok(())
}
```

- [ ] **Step 6: Build the extension**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
```

Expected: build succeeds.

- [ ] **Step 7: Smoke-test from Python**

```bash
unset CONDA_PREFIX && uv run --no-sync python -c "
from ferrum._core import LinearScale
s = LinearScale(domain=[0.0, 10.0], range=[0.0, 1.0])
assert abs(s.scale(5.0) - 0.5) < 1e-12, s.scale(5.0)
assert abs(s.invert(0.5) - 5.0) < 1e-12, s.invert(0.5)
assert s.domain == [0.0, 10.0]
assert s.range == [0.0, 1.0]
assert s.clamp is False
assert repr(s) == 'LinearScale(domain=[0, 10], range=[0, 1], clamp=False)'
print('OK')
"
```

Expected: prints `OK`.

- [ ] **Step 8: Run full Rust + Python suites**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core
uv run pytest
```

Expected: 45 Rust + 18 Python pass (35 from B1 + 10 new in C1).

- [ ] **Step 9: Commit**

```bash
git add crates/ferrum-core/src/scale crates/ferrum-core/src/lib.rs
git commit -m "feat(scale): LinearScale with sealed Scale enum scaffold"
```

---

### Task C2: LogScale

**Files:**
- Modify: `crates/ferrum-core/src/scale/core.rs` (add `Log` variant + arms)
- Create: `crates/ferrum-core/src/scale/log.rs`
- Modify: `crates/ferrum-core/src/scale/mod.rs`
- Modify: `crates/ferrum-core/src/lib.rs`

- [ ] **Step 1: Add `Log` variant to the `Scale` enum in `core.rs`**

In `crates/ferrum-core/src/scale/core.rs`, replace:

```rust
pub(crate) enum Scale {
    Linear { domain: [f64; 2], range: [f64; 2], clamp: bool },
}
```

with:

```rust
pub(crate) enum Scale {
    Linear { domain: [f64; 2], range: [f64; 2], clamp: bool },
    Log    { domain: [f64; 2], range: [f64; 2], base: f64, clamp: bool },
}
```

- [ ] **Step 2: Add `Log` arm in each method in `core.rs`**

In `scale_f64`, replace the existing match with:

```rust
    pub(crate) fn scale_f64(&self, x: f64) -> f64 {
        match self {
            Scale::Linear { domain, range, clamp } => {
                if x.is_nan() { return f64::NAN; }
                let [d0, d1] = *domain;
                let [r0, r1] = *range;
                let t = (x - d0) / (d1 - d0);
                let mapped = r0 + t * (r1 - r0);
                if *clamp {
                    let (lo, hi) = if r0 <= r1 { (r0, r1) } else { (r1, r0) };
                    mapped.clamp(lo, hi)
                } else if x < d0.min(d1) || x > d0.max(d1) {
                    f64::NAN
                } else {
                    mapped
                }
            }
            Scale::Log { domain, range, base, clamp } => {
                if x.is_nan() { return f64::NAN; }
                let [d0, d1] = *domain;
                let [r0, r1] = *range;
                let neg = d0 < 0.0;
                let sign = if neg { -1.0 } else { 1.0 };
                if (x * sign) <= 0.0 && !*clamp { return f64::NAN; }
                let log_base = base.ln();
                let lx = (x * sign).max(f64::MIN_POSITIVE).ln() / log_base;
                let ld0 = (d0 * sign).ln() / log_base;
                let ld1 = (d1 * sign).ln() / log_base;
                let t = (lx - ld0) / (ld1 - ld0);
                let mapped = r0 + t * (r1 - r0);
                if *clamp {
                    let (lo, hi) = if r0 <= r1 { (r0, r1) } else { (r1, r0) };
                    mapped.clamp(lo, hi)
                } else if (x * sign) < (d0 * sign).min(d1 * sign) || (x * sign) > (d0 * sign).max(d1 * sign) {
                    f64::NAN
                } else {
                    mapped
                }
            }
        }
    }
```

In `invert_f64`, replace with:

```rust
    pub(crate) fn invert_f64(&self, y: f64) -> f64 {
        match self {
            Scale::Linear { domain, range, clamp } => {
                if y.is_nan() { return f64::NAN; }
                let [d0, d1] = *domain;
                let [r0, r1] = *range;
                let t = (y - r0) / (r1 - r0);
                let mapped = d0 + t * (d1 - d0);
                if *clamp {
                    let (lo, hi) = if d0 <= d1 { (d0, d1) } else { (d1, d0) };
                    mapped.clamp(lo, hi)
                } else if y < r0.min(r1) || y > r0.max(r1) {
                    f64::NAN
                } else {
                    mapped
                }
            }
            Scale::Log { domain, range, base, clamp } => {
                if y.is_nan() { return f64::NAN; }
                let [d0, d1] = *domain;
                let [r0, r1] = *range;
                let neg = d0 < 0.0;
                let sign = if neg { -1.0 } else { 1.0 };
                let log_base = base.ln();
                let ld0 = (d0 * sign).ln() / log_base;
                let ld1 = (d1 * sign).ln() / log_base;
                let t = (y - r0) / (r1 - r0);
                let lmapped = ld0 + t * (ld1 - ld0);
                let mapped = sign * base.powf(lmapped);
                if *clamp {
                    let (lo, hi) = if d0 <= d1 { (d0, d1) } else { (d1, d0) };
                    mapped.clamp(lo, hi)
                } else if y < r0.min(r1) || y > r0.max(r1) {
                    f64::NAN
                } else {
                    mapped
                }
            }
        }
    }
```

In `ticks`, replace with:

```rust
    pub(crate) fn ticks(&self, count: Option<usize>) -> Vec<f64> {
        match self {
            Scale::Linear { domain, .. } => {
                nice_ticks(domain[0], domain[1], count.unwrap_or(10))
            }
            Scale::Log { domain, base, .. } => {
                let n = count.unwrap_or(10);
                let neg = domain[0] < 0.0;
                let sign: f64 = if neg { -1.0 } else { 1.0 };
                let lo = (domain[0] * sign).min(domain[1] * sign);
                let hi = (domain[0] * sign).max(domain[1] * sign);
                let log_base = base.ln();
                let lo_exp = (lo.ln() / log_base).floor() as i64;
                let hi_exp = (hi.ln() / log_base).ceil() as i64;
                let span_decades = (hi_exp - lo_exp).max(1) as usize;
                if span_decades >= n {
                    // produce one tick per decade
                    let mut out: Vec<f64> = (lo_exp..=hi_exp)
                        .map(|e| sign * base.powi(e as i32))
                        .filter(|t| (t.abs() >= lo) && (t.abs() <= hi))
                        .collect();
                    if domain[0] > domain[1] { out.reverse(); }
                    out
                } else {
                    // fall back to nice ticks of the linearised values
                    let lvals = nice_ticks(lo.ln() / log_base, hi.ln() / log_base, n);
                    let mut out: Vec<f64> = lvals.into_iter().map(|lv| sign * base.powf(lv)).collect();
                    if domain[0] > domain[1] { out.reverse(); }
                    out
                }
            }
        }
    }
```

In `nice`, replace with:

```rust
    pub(crate) fn nice(self) -> Self {
        match self {
            Scale::Linear { domain, range, clamp } => {
                let step = nice_step(domain[0], domain[1], 10);
                if !step.is_finite() || step == 0.0 {
                    return Scale::Linear { domain, range, clamp };
                }
                let lo_min = domain[0].min(domain[1]);
                let hi_max = domain[0].max(domain[1]);
                let nice_lo = (lo_min / step).floor() * step;
                let nice_hi = (hi_max / step).ceil() * step;
                let new_domain = if domain[0] <= domain[1] {
                    [nice_lo, nice_hi]
                } else {
                    [nice_hi, nice_lo]
                };
                Scale::Linear { domain: new_domain, range, clamp }
            }
            Scale::Log { domain, range, base, clamp } => {
                let neg = domain[0] < 0.0;
                let sign: f64 = if neg { -1.0 } else { 1.0 };
                let log_base = base.ln();
                let lo = (domain[0] * sign).min(domain[1] * sign);
                let hi = (domain[0] * sign).max(domain[1] * sign);
                let lo_exp = (lo.ln() / log_base).floor();
                let hi_exp = (hi.ln() / log_base).ceil();
                let new_lo = sign * base.powf(lo_exp);
                let new_hi = sign * base.powf(hi_exp);
                let new_domain = if domain[0] <= domain[1] {
                    [new_lo, new_hi]
                } else {
                    [new_hi, new_lo]
                };
                Scale::Log { domain: new_domain, range, base, clamp }
            }
        }
    }
```

- [ ] **Step 3: Add Rust unit tests for Log in `core.rs` `mod tests`**

Append to the existing `mod tests {}` block:

```rust
    #[test]
    fn test_log_scale_basic_decades() {
        let s = Scale::Log { domain: [1.0, 1000.0], range: [0.0, 3.0], base: 10.0, clamp: false };
        assert!((s.scale_f64(1.0) - 0.0).abs() < 1e-12);
        assert!((s.scale_f64(10.0) - 1.0).abs() < 1e-12);
        assert!((s.scale_f64(1000.0) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_log_inversion_round_trip() {
        let s = Scale::Log { domain: [1.0, 1_000_000.0], range: [0.0, 6.0], base: 10.0, clamp: false };
        for x in [1.0, 10.0, 100.0, 12345.0, 999999.0] {
            let y = s.scale_f64(x);
            let back = s.invert_f64(y);
            assert!((back / x - 1.0).abs() < 1e-9, "round-trip failed at x={x}: got {back}");
        }
    }

    #[test]
    fn test_log_negative_domain_supported() {
        let s = Scale::Log { domain: [-1000.0, -1.0], range: [0.0, 3.0], base: 10.0, clamp: false };
        let y = s.scale_f64(-10.0);
        let back = s.invert_f64(y);
        assert!((back / -10.0 - 1.0).abs() < 1e-9, "negative round-trip failed: got {back}");
    }

    #[test]
    fn test_log_ticks_one_per_decade() {
        let s = Scale::Log { domain: [1.0, 1000.0], range: [0.0, 3.0], base: 10.0, clamp: false };
        let t = s.ticks(Some(4));
        // span 3 decades, count 4 → fall through to per-decade path (3 decades >= 4? no, 3<4 so fallback to nice)
        // OR per-decade path returns 4 ticks (1, 10, 100, 1000). Either is acceptable; we just want at least 3 ticks.
        assert!(t.len() >= 3, "got {} ticks: {t:?}", t.len());
    }

    #[test]
    fn test_log_nice_rounds_to_decades() {
        let s = Scale::Log { domain: [3.0, 700.0], range: [0.0, 1.0], base: 10.0, clamp: false };
        let n = s.nice();
        match n {
            Scale::Log { domain, .. } => {
                assert!((domain[0] - 1.0).abs() < 1e-9);
                assert!((domain[1] - 1000.0).abs() < 1e-9);
            }
            _ => panic!("unexpected variant"),
        }
    }
```

- [ ] **Step 4: Run Rust tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core scale::core::
```

Expected: 15 tests pass (10 from C1 + 5 new).

- [ ] **Step 5: Create `crates/ferrum-core/src/scale/log.rs`**

```rust
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::core::{validate_continuous_pair, Scale};

#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, PartialEq)]
pub struct LogScale(Scale);

impl LogScale {
    pub(crate) fn repr_string(&self) -> String {
        match &self.0 {
            Scale::Log { domain, range, base, clamp } => format!(
                "LogScale(domain=[{}, {}], range=[{}, {}], base={}, clamp={})",
                domain[0], domain[1], range[0], range[1], base, if *clamp { "True" } else { "False" }
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

#[pymethods]
impl LogScale {
    #[new]
    #[pyo3(signature = (*, domain, range, base = 10.0, clamp = false, nice = false))]
    fn new(domain: Vec<f64>, range: Vec<f64>, base: f64, clamp: bool, nice: bool) -> PyResult<Self> {
        validate_continuous_pair(&domain, &range)?;
        if !base.is_finite() || base <= 0.0 || base == 1.0 {
            return Err(PyValueError::new_err(format!(
                "base must be finite, > 0, and != 1; got {base}"
            )));
        }
        if domain[0] == 0.0 || domain[1] == 0.0 {
            return Err(PyValueError::new_err(
                "log scale domain must not contain 0",
            ));
        }
        if domain[0].signum() != domain[1].signum() {
            return Err(PyValueError::new_err(
                "log scale domain endpoints must have the same sign",
            ));
        }
        let mut s = Scale::Log {
            domain: [domain[0], domain[1]],
            range:  [range[0],  range[1]],
            base,
            clamp,
        };
        if nice {
            s = s.nice();
        }
        Ok(LogScale(s))
    }

    fn scale(&self, x: f64) -> f64 { self.0.scale_f64(x) }
    fn invert(&self, y: f64) -> f64 { self.0.invert_f64(y) }

    #[pyo3(signature = (count = 10))]
    fn ticks(&self, count: usize) -> Vec<f64> { self.0.ticks(Some(count)) }

    fn nice(&self) -> Self { LogScale(self.0.clone().nice()) }

    #[getter]
    fn domain(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Log { domain, .. } => domain.to_vec(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    #[getter]
    fn range(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Log { range, .. } => range.to_vec(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    #[getter]
    fn base(&self) -> f64 {
        match &self.0 {
            Scale::Log { base, .. } => *base,
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    #[getter]
    fn clamp(&self) -> bool {
        match &self.0 {
            Scale::Log { clamp, .. } => *clamp,
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    fn __repr__(&self) -> String { self.repr_string() }
}
```

- [ ] **Step 6: Update `mod.rs`**

```rust
pub(crate) mod core;
pub(crate) mod ticks;
pub(crate) mod linear;
pub(crate) mod log;
```

- [ ] **Step 7: Update `lib.rs` to register `LogScale`**

```rust
use pyo3::prelude::*;

mod transport;
mod spec;
mod scale;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(transport::process_batch, m)?)?;
    m.add_class::<spec::chart::ChartSpec>()?;
    m.add_class::<spec::encoding::EncodingSpec>()?;
    m.add_class::<scale::linear::LinearScale>()?;
    m.add_class::<scale::log::LogScale>()?;
    Ok(())
}
```

- [ ] **Step 8: Build + smoke**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
unset CONDA_PREFIX && uv run --no-sync python -c "
from ferrum._core import LogScale
s = LogScale(domain=[1.0, 1000.0], range=[0.0, 3.0])
assert abs(s.scale(10.0) - 1.0) < 1e-12, s.scale(10.0)
assert abs(s.invert(2.0) - 100.0) < 1e-9, s.invert(2.0)
assert s.base == 10.0
print('OK')
"
```

Expected: prints `OK`.

- [ ] **Step 9: Reject invalid constructors**

```bash
unset CONDA_PREFIX && uv run --no-sync python -c "
from ferrum._core import LogScale
try:
    LogScale(domain=[0.0, 1000.0], range=[0.0, 3.0])
    raise SystemExit('expected ValueError for log domain containing 0')
except ValueError as e:
    print('OK:', e)
"
```

Expected: prints `OK: log scale domain must not contain 0`.

- [ ] **Step 10: Run full Rust suite**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core
```

Expected: 50 tests pass.

- [ ] **Step 11: Commit**

```bash
git add crates/ferrum-core/src/scale crates/ferrum-core/src/lib.rs
git commit -m "feat(scale): LogScale with negative-domain support and base validation"
```

---

### Task C3: SymlogScale

**Files:**
- Modify: `crates/ferrum-core/src/scale/core.rs` (add `Symlog` variant + arms)
- Create: `crates/ferrum-core/src/scale/symlog.rs`
- Modify: `crates/ferrum-core/src/scale/mod.rs`
- Modify: `crates/ferrum-core/src/lib.rs`

- [ ] **Step 1: Add `Symlog` variant in `core.rs`**

In the `Scale` enum, add the variant:

```rust
pub(crate) enum Scale {
    Linear { domain: [f64; 2], range: [f64; 2], clamp: bool },
    Log    { domain: [f64; 2], range: [f64; 2], base: f64, clamp: bool },
    Symlog { domain: [f64; 2], range: [f64; 2], constant: f64, clamp: bool },
}
```

- [ ] **Step 2: Add a private `symlog_transform` helper at the top of the `impl Scale` block**

Above `pub(crate) fn scale_f64(...)`, add:

```rust
    fn symlog_fwd(x: f64, c: f64) -> f64 {
        x.signum() * (x.abs() / c).ln_1p()
    }

    fn symlog_inv(y: f64, c: f64) -> f64 {
        y.signum() * c * (y.abs().exp() - 1.0)
    }
```

- [ ] **Step 3: Add `Symlog` arm in each method**

In `scale_f64`, after the `Scale::Log` arm:

```rust
            Scale::Symlog { domain, range, constant, clamp } => {
                if x.is_nan() { return f64::NAN; }
                let [d0, d1] = *domain;
                let [r0, r1] = *range;
                let f = |v: f64| Self::symlog_fwd(v, *constant);
                let t = (f(x) - f(d0)) / (f(d1) - f(d0));
                let mapped = r0 + t * (r1 - r0);
                if *clamp {
                    let (lo, hi) = if r0 <= r1 { (r0, r1) } else { (r1, r0) };
                    mapped.clamp(lo, hi)
                } else if x < d0.min(d1) || x > d0.max(d1) {
                    f64::NAN
                } else {
                    mapped
                }
            }
```

In `invert_f64`, after the `Scale::Log` arm:

```rust
            Scale::Symlog { domain, range, constant, clamp } => {
                if y.is_nan() { return f64::NAN; }
                let [d0, d1] = *domain;
                let [r0, r1] = *range;
                let f = |v: f64| Self::symlog_fwd(v, *constant);
                let t = (y - r0) / (r1 - r0);
                let lmapped = f(d0) + t * (f(d1) - f(d0));
                let mapped = Self::symlog_inv(lmapped, *constant);
                if *clamp {
                    let (lo, hi) = if d0 <= d1 { (d0, d1) } else { (d1, d0) };
                    mapped.clamp(lo, hi)
                } else if y < r0.min(r1) || y > r0.max(r1) {
                    f64::NAN
                } else {
                    mapped
                }
            }
```

In `ticks`, after the `Scale::Log` arm:

```rust
            Scale::Symlog { domain, .. } => {
                nice_ticks(domain[0], domain[1], count.unwrap_or(10))
            }
```

In `nice`, after the `Scale::Log` arm:

```rust
            Scale::Symlog { domain, range, constant, clamp } => {
                let step = nice_step(domain[0], domain[1], 10);
                if !step.is_finite() || step == 0.0 {
                    return Scale::Symlog { domain, range, constant, clamp };
                }
                let lo_min = domain[0].min(domain[1]);
                let hi_max = domain[0].max(domain[1]);
                let nice_lo = (lo_min / step).floor() * step;
                let nice_hi = (hi_max / step).ceil() * step;
                let new_domain = if domain[0] <= domain[1] {
                    [nice_lo, nice_hi]
                } else {
                    [nice_hi, nice_lo]
                };
                Scale::Symlog { domain: new_domain, range, constant, clamp }
            }
```

- [ ] **Step 4: Add Rust unit tests for Symlog in `core.rs` `mod tests`**

Append to `mod tests {}`:

```rust
    #[test]
    fn test_symlog_scale_handles_zero() {
        let s = Scale::Symlog { domain: [-100.0, 100.0], range: [0.0, 1.0], constant: 1.0, clamp: false };
        let y = s.scale_f64(0.0);
        assert!(y.is_finite(), "scale(0) returned {y}");
        assert!((y - 0.5).abs() < 1e-12, "expected 0.5, got {y}");
    }

    #[test]
    fn test_symlog_inversion_round_trip_across_zero() {
        let s = Scale::Symlog { domain: [-1000.0, 1000.0], range: [0.0, 1.0], constant: 1.0, clamp: false };
        for x in [-1000.0, -100.0, -1.0, 0.0, 1.0, 100.0, 1000.0] {
            let y = s.scale_f64(x);
            let back = s.invert_f64(y);
            assert!((back - x).abs() < 1e-6, "round-trip failed at x={x}: got {back}");
        }
    }

    #[test]
    fn test_symlog_constant_changes_curvature() {
        let s1 = Scale::Symlog { domain: [-100.0, 100.0], range: [0.0, 1.0], constant: 1.0,   clamp: false };
        let s2 = Scale::Symlog { domain: [-100.0, 100.0], range: [0.0, 1.0], constant: 100.0, clamp: false };
        // larger constant → behaves more linearly near zero
        let y1 = s1.scale_f64(50.0);
        let y2 = s2.scale_f64(50.0);
        assert!(y2 > y1, "expected y2={y2} > y1={y1} for larger constant");
    }
```

- [ ] **Step 5: Run Rust tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core scale::core::
```

Expected: 18 tests pass (15 from C2 + 3 new).

- [ ] **Step 6: Create `crates/ferrum-core/src/scale/symlog.rs`**

```rust
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::core::{validate_continuous_pair, Scale};

#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, PartialEq)]
pub struct SymlogScale(Scale);

impl SymlogScale {
    pub(crate) fn repr_string(&self) -> String {
        match &self.0 {
            Scale::Symlog { domain, range, constant, clamp } => format!(
                "SymlogScale(domain=[{}, {}], range=[{}, {}], constant={}, clamp={})",
                domain[0], domain[1], range[0], range[1], constant, if *clamp { "True" } else { "False" }
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

#[pymethods]
impl SymlogScale {
    #[new]
    #[pyo3(signature = (*, domain, range, constant = 1.0, clamp = false, nice = false))]
    fn new(domain: Vec<f64>, range: Vec<f64>, constant: f64, clamp: bool, nice: bool) -> PyResult<Self> {
        validate_continuous_pair(&domain, &range)?;
        if !constant.is_finite() || constant <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "constant must be finite and > 0; got {constant}"
            )));
        }
        let mut s = Scale::Symlog {
            domain: [domain[0], domain[1]],
            range:  [range[0],  range[1]],
            constant,
            clamp,
        };
        if nice {
            s = s.nice();
        }
        Ok(SymlogScale(s))
    }

    fn scale(&self, x: f64) -> f64 { self.0.scale_f64(x) }
    fn invert(&self, y: f64) -> f64 { self.0.invert_f64(y) }

    #[pyo3(signature = (count = 10))]
    fn ticks(&self, count: usize) -> Vec<f64> { self.0.ticks(Some(count)) }

    fn nice(&self) -> Self { SymlogScale(self.0.clone().nice()) }

    #[getter]
    fn domain(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Symlog { domain, .. } => domain.to_vec(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    #[getter]
    fn range(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Symlog { range, .. } => range.to_vec(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    #[getter]
    fn constant(&self) -> f64 {
        match &self.0 {
            Scale::Symlog { constant, .. } => *constant,
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    #[getter]
    fn clamp(&self) -> bool {
        match &self.0 {
            Scale::Symlog { clamp, .. } => *clamp,
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    fn __repr__(&self) -> String { self.repr_string() }
}
```

- [ ] **Step 7: Update `mod.rs`**

```rust
pub(crate) mod core;
pub(crate) mod ticks;
pub(crate) mod linear;
pub(crate) mod log;
pub(crate) mod symlog;
```

- [ ] **Step 8: Update `lib.rs` to register `SymlogScale`**

Add the line `m.add_class::<scale::symlog::SymlogScale>()?;` after the `LogScale` registration.

- [ ] **Step 9: Build + smoke + cargo test**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
unset CONDA_PREFIX && uv run --no-sync python -c "
from ferrum._core import SymlogScale
s = SymlogScale(domain=[-100.0, 100.0], range=[0.0, 1.0])
assert abs(s.scale(0.0) - 0.5) < 1e-12, s.scale(0.0)
print('OK')
"
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core
```

Expected: smoke prints `OK`; 53 Rust tests pass.

- [ ] **Step 10: Commit**

```bash
git add crates/ferrum-core/src/scale crates/ferrum-core/src/lib.rs
git commit -m "feat(scale): SymlogScale (handles zero crossing via signed log1p)"
```

---

### Task C4: TimeScale

`TimeScale` is arithmetically identical to `LinearScale` but with f64 ms-epoch domain and `nice_time_interval_ms`-based ticks. We piggy-back on `Scale::Linear`'s math and only differentiate at the pyclass layer for the tick/nice interval choice.

> **Implementation note:** because `TimeScale` *behavior* is `Linear` math + time-aware ticks, we avoid creating a new enum variant. The pyclass simply wraps `Scale::Linear` and overrides `ticks` and `nice` at the pyclass level using `nice_time_interval_ms`.

**Files:**
- Create: `crates/ferrum-core/src/scale/time.rs`
- Modify: `crates/ferrum-core/src/scale/mod.rs`
- Modify: `crates/ferrum-core/src/lib.rs`

- [ ] **Step 1: Create `crates/ferrum-core/src/scale/time.rs`**

```rust
use pyo3::prelude::*;

use super::core::{validate_continuous_pair, Scale};
use super::ticks::nice_time_interval_ms;

#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, PartialEq)]
pub struct TimeScale(Scale);

impl TimeScale {
    pub(crate) fn repr_string(&self) -> String {
        match &self.0 {
            Scale::Linear { domain, range, clamp } => format!(
                "TimeScale(domain=[{}, {}], range=[{}, {}], clamp={})",
                domain[0], domain[1], range[0], range[1], if *clamp { "True" } else { "False" }
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    fn time_ticks(&self, count: usize) -> Vec<f64> {
        let (d0, d1) = match &self.0 {
            Scale::Linear { domain, .. } => (domain[0], domain[1]),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        };
        let lo = d0.min(d1);
        let hi = d0.max(d1);
        let span = hi - lo;
        let interval = nice_time_interval_ms(span, count);
        if !interval.is_finite() || interval <= 0.0 {
            return Vec::new();
        }
        let start = (lo / interval).ceil() * interval;
        let end = (hi / interval).floor() * interval;
        let n_steps = ((end - start) / interval).round() as i64;
        if n_steps < 0 {
            return Vec::new();
        }
        let n = (n_steps + 1) as usize;
        let mut out: Vec<f64> = (0..n).map(|i| start + (i as f64) * interval).collect();
        if d0 > d1 {
            out.reverse();
        }
        out
    }

    fn time_nice(&self) -> Self {
        let (d0, d1, range, clamp) = match &self.0 {
            Scale::Linear { domain, range, clamp } => (domain[0], domain[1], *range, *clamp),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        };
        let lo = d0.min(d1);
        let hi = d0.max(d1);
        let interval = nice_time_interval_ms(hi - lo, 10);
        if !interval.is_finite() || interval <= 0.0 {
            return self.clone();
        }
        let new_lo = (lo / interval).floor() * interval;
        let new_hi = (hi / interval).ceil() * interval;
        let new_domain = if d0 <= d1 { [new_lo, new_hi] } else { [new_hi, new_lo] };
        TimeScale(Scale::Linear { domain: new_domain, range, clamp })
    }
}

#[pymethods]
impl TimeScale {
    #[new]
    #[pyo3(signature = (*, domain, range, clamp = false, nice = false))]
    fn new(domain: Vec<f64>, range: Vec<f64>, clamp: bool, nice: bool) -> PyResult<Self> {
        validate_continuous_pair(&domain, &range)?;
        let inner = Scale::Linear {
            domain: [domain[0], domain[1]],
            range:  [range[0],  range[1]],
            clamp,
        };
        let s = TimeScale(inner);
        if nice {
            Ok(s.time_nice())
        } else {
            Ok(s)
        }
    }

    fn scale(&self, x: f64) -> f64 { self.0.scale_f64(x) }
    fn invert(&self, y: f64) -> f64 { self.0.invert_f64(y) }

    #[pyo3(signature = (count = 10))]
    fn ticks(&self, count: usize) -> Vec<f64> { self.time_ticks(count) }

    fn nice(&self) -> Self { self.time_nice() }

    #[getter]
    fn domain(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Linear { domain, .. } => domain.to_vec(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    #[getter]
    fn range(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Linear { range, .. } => range.to_vec(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    #[getter]
    fn clamp(&self) -> bool {
        match &self.0 {
            Scale::Linear { clamp, .. } => *clamp,
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    fn __repr__(&self) -> String { self.repr_string() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_scale_round_trip_ms() {
        // 2026-01-01 00:00:00 UTC = 1767225600000.0 ms
        // 2026-12-31 23:59:59 UTC ≈ 1798761599000.0 ms
        let t = TimeScale::new(
            vec![1_767_225_600_000.0, 1_798_761_599_000.0],
            vec![0.0, 1000.0],
            false,
            false,
        ).unwrap();
        let mid = (1_767_225_600_000.0 + 1_798_761_599_000.0) / 2.0;
        let y = t.scale(mid);
        let back = t.invert(y);
        assert!((back - mid).abs() < 1e-3, "round-trip failed: got {back}");
    }

    #[test]
    fn test_time_ticks_returns_some_ticks_for_year_span() {
        let t = TimeScale::new(
            vec![1_767_225_600_000.0, 1_798_761_599_000.0],
            vec![0.0, 1000.0],
            false,
            false,
        ).unwrap();
        let ticks = t.ticks(10);
        assert!(!ticks.is_empty(), "expected non-empty ticks");
    }
}
```

- [ ] **Step 2: Update `mod.rs`**

```rust
pub(crate) mod core;
pub(crate) mod ticks;
pub(crate) mod linear;
pub(crate) mod log;
pub(crate) mod symlog;
pub(crate) mod time;
```

- [ ] **Step 3: Update `lib.rs` to register `TimeScale`**

Add `m.add_class::<scale::time::TimeScale>()?;` after the `SymlogScale` registration.

- [ ] **Step 4: Build + smoke + cargo test**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
unset CONDA_PREFIX && uv run --no-sync python -c "
from ferrum._core import TimeScale
t = TimeScale(domain=[0.0, 86400000.0], range=[0.0, 1.0])  # 1 day in ms
assert abs(t.scale(43200000.0) - 0.5) < 1e-12, t.scale(43200000.0)
print('OK')
"
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core
```

Expected: smoke prints `OK`; 55 Rust tests pass (53 + 2 in time module).

- [ ] **Step 5: Commit**

```bash
git add crates/ferrum-core/src/scale crates/ferrum-core/src/lib.rs
git commit -m "feat(scale): TimeScale (Linear math + nice_time_interval_ms ticks)"
```

---

## Section D — Discrete scales

### Task D1: OrdinalScale

**Files:**
- Modify: `crates/ferrum-core/src/scale/core.rs` (add `Ordinal` variant + arms)
- Create: `crates/ferrum-core/src/scale/ordinal.rs`
- Modify: `crates/ferrum-core/src/scale/mod.rs`
- Modify: `crates/ferrum-core/src/lib.rs`

- [ ] **Step 1: Add `Ordinal` variant to the `Scale` enum**

```rust
pub(crate) enum Scale {
    Linear  { domain: [f64; 2], range: [f64; 2], clamp: bool },
    Log     { domain: [f64; 2], range: [f64; 2], base: f64, clamp: bool },
    Symlog  { domain: [f64; 2], range: [f64; 2], constant: f64, clamp: bool },
    Ordinal { domain: Vec<String>, range: Vec<f64>, padding: f64 },
}
```

- [ ] **Step 2: Add Ordinal helper to `impl Scale` block**

Above `pub(crate) fn scale_f64`, add this private helper. The return is `(first_band_center, step, half_band)`: the first band center is `r_lo + step/2`; `step` is the per-band stride; `half_band` is the band's half-width used by `invert_band` for membership checks.

```rust
    fn ordinal_layout(domain: &[String], range: &[f64], padding: f64) -> (f64, f64, f64) {
        let r_lo = *range.first().unwrap();
        let r_hi = *range.last().unwrap();
        let n = domain.len() as f64;
        let step = (r_hi - r_lo) / n;
        let half_band = step.abs() * (1.0 - padding) / 2.0;
        let first_center = r_lo + step / 2.0;
        (first_center, step, half_band)
    }
```

- [ ] **Step 3: Add `Ordinal` arms in dispatch methods**

In `scale_f64`, after the `Scale::Symlog` arm:

```rust
            Scale::Ordinal { .. } => f64::NAN,
```

In `invert_f64`, after the `Scale::Symlog` arm:

```rust
            Scale::Ordinal { .. } => f64::NAN,
```

In `ticks`, after the `Scale::Symlog` arm:

```rust
            Scale::Ordinal { range, .. } => range.clone(),
```

In `nice`, after the `Scale::Symlog` arm:

```rust
            Scale::Ordinal { domain, range, padding } => {
                Scale::Ordinal { domain, range, padding }
            }
```

- [ ] **Step 4: Add `scale_str` and `invert_band` methods on `impl Scale`**

After `nice`, add:

```rust
    pub(crate) fn scale_str(&self, s: &str) -> f64 {
        match self {
            Scale::Ordinal { domain, range, padding } => {
                let idx = match domain.iter().position(|c| c == s) {
                    Some(i) => i,
                    None => return f64::NAN,
                };
                let (first_center, step, _half_band) = Self::ordinal_layout(domain, range, *padding);
                first_center + (idx as f64) * step
            }
            _ => f64::NAN,
        }
    }

    pub(crate) fn invert_band(&self, y: f64) -> Option<String> {
        match self {
            Scale::Ordinal { domain, range, padding } => {
                if y.is_nan() { return None; }
                let (first_center, step, half_band) = Self::ordinal_layout(domain, range, *padding);
                if step == 0.0 { return None; }
                // Find candidate index by inverse step
                let raw = (y - first_center) / step;
                let idx = raw.round() as i64;
                if idx < 0 || idx as usize >= domain.len() { return None; }
                let center = first_center + (idx as f64) * step;
                if (y - center).abs() <= half_band {
                    Some(domain[idx as usize].clone())
                } else {
                    None
                }
            }
            _ => None,
        }
    }
```

- [ ] **Step 5: Add Ordinal-specific validator to `core.rs`**

Below `validate_continuous_pair`, add:

```rust
pub(crate) fn validate_ordinal(domain: &[String], range: &[f64], padding: f64) -> PyResult<()> {
    if domain.is_empty() {
        return Err(PyValueError::new_err("domain must be non-empty"));
    }
    if range.len() < 2 {
        return Err(PyValueError::new_err(format!(
            "range must have length >= 2 (extent endpoints); got {}",
            range.len()
        )));
    }
    validate_finite("range", range)?;
    if !padding.is_finite() || !(0.0..=1.0).contains(&padding) {
        return Err(PyValueError::new_err(format!(
            "padding must be in [0, 1]; got {padding}"
        )));
    }
    let mut seen = std::collections::HashSet::new();
    for c in domain {
        if !seen.insert(c.as_str()) {
            return Err(PyValueError::new_err(format!(
                "duplicate category in domain: '{c}'"
            )));
        }
    }
    Ok(())
}
```

- [ ] **Step 6: Add Ordinal tests to `core.rs` `mod tests`**

Append:

```rust
    #[test]
    fn test_ordinal_band_centers_no_padding() {
        let s = Scale::Ordinal {
            domain: vec!["a".into(), "b".into(), "c".into()],
            range: vec![0.0, 30.0],
            padding: 0.0,
        };
        // 3 bands of width 10, centers at 5, 15, 25
        assert!((s.scale_str("a") - 5.0).abs() < 1e-12);
        assert!((s.scale_str("b") - 15.0).abs() < 1e-12);
        assert!((s.scale_str("c") - 25.0).abs() < 1e-12);
    }

    #[test]
    fn test_ordinal_invert_round_trip() {
        let s = Scale::Ordinal {
            domain: vec!["a".into(), "b".into(), "c".into()],
            range: vec![0.0, 30.0],
            padding: 0.0,
        };
        for cat in ["a", "b", "c"] {
            let y = s.scale_str(cat);
            let back = s.invert_band(y);
            assert_eq!(back.as_deref(), Some(cat), "round-trip failed for {cat}");
        }
    }

    #[test]
    fn test_ordinal_invert_outside_band_returns_none() {
        // padding=0.5 → half_band = step * 0.5 / 2 = 2.5; band of width 5 around each center
        let s = Scale::Ordinal {
            domain: vec!["a".into(), "b".into(), "c".into()],
            range: vec![0.0, 30.0],
            padding: 0.5,
        };
        // y=10 is exactly between centers 5 and 15; outside both bands of width 5
        assert!(s.invert_band(10.0).is_none());
    }

    #[test]
    fn test_ordinal_unknown_category_returns_nan() {
        let s = Scale::Ordinal {
            domain: vec!["a".into()],
            range: vec![0.0, 10.0],
            padding: 0.0,
        };
        assert!(s.scale_str("z").is_nan());
    }

    #[test]
    fn test_validate_ordinal_rejects_empty_domain() {
        let r = validate_ordinal(&[], &[0.0, 10.0], 0.0);
        assert!(r.is_err());
    }

    #[test]
    fn test_validate_ordinal_rejects_duplicates() {
        let r = validate_ordinal(
            &["a".to_string(), "a".to_string()],
            &[0.0, 10.0],
            0.0,
        );
        assert!(r.is_err());
    }

    #[test]
    fn test_validate_ordinal_rejects_bad_padding() {
        let r = validate_ordinal(&["a".to_string()], &[0.0, 10.0], 1.5);
        assert!(r.is_err());
    }
```

- [ ] **Step 7: Run Rust tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core scale::core::
```

Expected: 25 tests pass (18 from C3 + 7 new).

- [ ] **Step 8: Create `crates/ferrum-core/src/scale/ordinal.rs`**

```rust
use pyo3::prelude::*;

use super::core::{validate_ordinal, Scale};

#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, PartialEq)]
pub struct OrdinalScale(Scale);

impl OrdinalScale {
    pub(crate) fn repr_string(&self) -> String {
        match &self.0 {
            Scale::Ordinal { domain, range, padding } => format!(
                "OrdinalScale(domain={:?}, range=[{}, {}], padding={})",
                domain, range.first().copied().unwrap_or(0.0), range.last().copied().unwrap_or(0.0), padding
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

#[pymethods]
impl OrdinalScale {
    #[new]
    #[pyo3(signature = (*, domain, range, padding = 0.0))]
    fn new(domain: Vec<String>, range: Vec<f64>, padding: f64) -> PyResult<Self> {
        validate_ordinal(&domain, &range, padding)?;
        Ok(OrdinalScale(Scale::Ordinal { domain, range, padding }))
    }

    fn scale(&self, value: &str) -> f64 {
        self.0.scale_str(value)
    }

    fn invert(&self, y: f64) -> Option<String> {
        self.0.invert_band(y)
    }

    fn ticks(&self) -> Vec<String> {
        match &self.0 {
            Scale::Ordinal { domain, .. } => domain.clone(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    fn nice(&self) -> Self {
        self.clone()
    }

    #[getter]
    fn domain(&self) -> Vec<String> {
        match &self.0 {
            Scale::Ordinal { domain, .. } => domain.clone(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    #[getter]
    fn range(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Ordinal { range, .. } => range.clone(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    #[getter]
    fn padding(&self) -> f64 {
        match &self.0 {
            Scale::Ordinal { padding, .. } => *padding,
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    fn __repr__(&self) -> String { self.repr_string() }
}
```

- [ ] **Step 9: Update `mod.rs`** to add `pub(crate) mod ordinal;`.

- [ ] **Step 10: Update `lib.rs`** to add `m.add_class::<scale::ordinal::OrdinalScale>()?;`.

- [ ] **Step 11: Build + smoke + cargo test**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
unset CONDA_PREFIX && uv run --no-sync python -c "
from ferrum._core import OrdinalScale
s = OrdinalScale(domain=['a', 'b', 'c'], range=[0.0, 30.0])
assert abs(s.scale('a') - 5.0) < 1e-12, s.scale('a')
assert s.invert(5.0) == 'a'
assert s.invert(100.0) is None
assert s.ticks() == ['a', 'b', 'c']
print('OK')
"
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core
```

Expected: smoke prints `OK`; 62 Rust tests pass.

- [ ] **Step 12: Commit**

```bash
git add crates/ferrum-core/src/scale crates/ferrum-core/src/lib.rs
git commit -m "feat(scale): OrdinalScale with band-padding semantics"
```

---

### Task D2: ThresholdScale

**Files:**
- Modify: `crates/ferrum-core/src/scale/core.rs` (add `Threshold` variant + arms)
- Create: `crates/ferrum-core/src/scale/threshold.rs`
- Modify: `crates/ferrum-core/src/scale/mod.rs`
- Modify: `crates/ferrum-core/src/lib.rs`

- [ ] **Step 1: Add `Threshold` variant to `Scale` enum**

```rust
pub(crate) enum Scale {
    Linear   { domain: [f64; 2], range: [f64; 2], clamp: bool },
    Log      { domain: [f64; 2], range: [f64; 2], base: f64, clamp: bool },
    Symlog   { domain: [f64; 2], range: [f64; 2], constant: f64, clamp: bool },
    Ordinal  { domain: Vec<String>, range: Vec<f64>, padding: f64 },
    Threshold{ domain: Vec<f64>, range: Vec<f64> },
}
```

- [ ] **Step 2: Add `Threshold` arms in dispatch methods**

In `scale_f64`, after `Ordinal`:

```rust
            Scale::Threshold { domain, range } => {
                if x.is_nan() { return f64::NAN; }
                let idx = domain.partition_point(|t| *t <= x);
                range[idx]
            }
```

In `invert_f64`, after `Ordinal`:

```rust
            Scale::Threshold { .. } => f64::NAN,
```

In `ticks`, after `Ordinal`:

```rust
            Scale::Threshold { domain, .. } => domain.clone(),
```

In `nice`, after `Ordinal`:

```rust
            Scale::Threshold { domain, range } => Scale::Threshold { domain, range },
```

- [ ] **Step 3: Add `invert_extent` method on `impl Scale`**

After `invert_band`, add:

```rust
    pub(crate) fn invert_extent(&self, y: f64) -> (f64, f64) {
        match self {
            Scale::Threshold { domain, range } => {
                if y.is_nan() { return (f64::NAN, f64::NAN); }
                let idx = match range.iter().position(|r| *r == y) {
                    Some(i) => i,
                    None => return (f64::NAN, f64::NAN),
                };
                let lo = if idx == 0 { f64::NEG_INFINITY } else { domain[idx - 1] };
                let hi = if idx >= domain.len() { f64::INFINITY } else { domain[idx] };
                (lo, hi)
            }
            _ => (f64::NAN, f64::NAN),
        }
    }
```

- [ ] **Step 4: Add `validate_threshold` validator to `core.rs`**

Below `validate_ordinal`, add:

```rust
pub(crate) fn validate_threshold(domain: &[f64], range: &[f64]) -> PyResult<()> {
    if range.is_empty() {
        return Err(PyValueError::new_err("range must be non-empty"));
    }
    if domain.len() + 1 != range.len() {
        return Err(PyValueError::new_err(format!(
            "range length must equal domain length + 1; got domain={}, range={}",
            domain.len(),
            range.len()
        )));
    }
    validate_finite("domain", domain)?;
    validate_finite("range", range)?;
    for w in domain.windows(2) {
        if w[0] >= w[1] {
            return Err(PyValueError::new_err(
                "domain must be strictly sorted ascending",
            ));
        }
    }
    Ok(())
}
```

- [ ] **Step 5: Add Threshold tests to `core.rs` `mod tests`**

```rust
    #[test]
    fn test_threshold_scale_basic() {
        let s = Scale::Threshold {
            domain: vec![0.0, 10.0],
            range: vec![1.0, 2.0, 3.0],
        };
        assert_eq!(s.scale_f64(-1.0), 1.0);
        assert_eq!(s.scale_f64(0.0), 2.0);   // partition_point with <= places 0.0 into bin 1
        assert_eq!(s.scale_f64(5.0), 2.0);
        assert_eq!(s.scale_f64(10.0), 3.0);
        assert_eq!(s.scale_f64(20.0), 3.0);
    }

    #[test]
    fn test_threshold_invert_extent_round_trip() {
        let s = Scale::Threshold {
            domain: vec![0.0, 10.0],
            range: vec![1.0, 2.0, 3.0],
        };
        let (lo, hi) = s.invert_extent(2.0);
        assert_eq!((lo, hi), (0.0, 10.0));
        let (lo, hi) = s.invert_extent(1.0);
        assert!(lo.is_infinite() && lo.is_sign_negative());
        assert_eq!(hi, 0.0);
        let (lo, hi) = s.invert_extent(3.0);
        assert_eq!(lo, 10.0);
        assert!(hi.is_infinite() && hi.is_sign_positive());
    }

    #[test]
    fn test_threshold_invert_extent_unknown_returns_nan() {
        let s = Scale::Threshold {
            domain: vec![0.0],
            range: vec![1.0, 2.0],
        };
        let (lo, hi) = s.invert_extent(99.0);
        assert!(lo.is_nan() && hi.is_nan());
    }

    #[test]
    fn test_validate_threshold_rejects_arity_mismatch() {
        let r = validate_threshold(&[0.0, 10.0], &[1.0, 2.0]);
        assert!(r.is_err());
    }

    #[test]
    fn test_validate_threshold_rejects_unsorted_domain() {
        let r = validate_threshold(&[10.0, 0.0], &[1.0, 2.0, 3.0]);
        assert!(r.is_err());
    }
```

- [ ] **Step 6: Run Rust tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core scale::core::
```

Expected: 30 tests pass (25 from D1 + 5 new).

- [ ] **Step 7: Create `crates/ferrum-core/src/scale/threshold.rs`**

```rust
use pyo3::prelude::*;

use super::core::{validate_threshold, Scale};

#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, PartialEq)]
pub struct ThresholdScale(Scale);

impl ThresholdScale {
    pub(crate) fn repr_string(&self) -> String {
        match &self.0 {
            Scale::Threshold { domain, range } => format!(
                "ThresholdScale(domain={:?}, range={:?})",
                domain, range
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

#[pymethods]
impl ThresholdScale {
    #[new]
    #[pyo3(signature = (*, domain, range))]
    fn new(domain: Vec<f64>, range: Vec<f64>) -> PyResult<Self> {
        validate_threshold(&domain, &range)?;
        Ok(ThresholdScale(Scale::Threshold { domain, range }))
    }

    fn scale(&self, x: f64) -> f64 { self.0.scale_f64(x) }

    fn invert_extent(&self, y: f64) -> (f64, f64) { self.0.invert_extent(y) }

    fn ticks(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Threshold { domain, .. } => domain.clone(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    fn nice(&self) -> Self { self.clone() }

    #[getter]
    fn domain(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Threshold { domain, .. } => domain.clone(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    #[getter]
    fn range(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Threshold { range, .. } => range.clone(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    fn __repr__(&self) -> String { self.repr_string() }
}
```

- [ ] **Step 8: Update `mod.rs`** to add `pub(crate) mod threshold;`.

- [ ] **Step 9: Update `lib.rs`** to add `m.add_class::<scale::threshold::ThresholdScale>()?;`.

- [ ] **Step 10: Build + smoke + cargo test**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
unset CONDA_PREFIX && uv run --no-sync python -c "
from ferrum._core import ThresholdScale
s = ThresholdScale(domain=[0.0, 10.0], range=[1.0, 2.0, 3.0])
assert s.scale(-1.0) == 1.0
assert s.scale(5.0) == 2.0
assert s.scale(11.0) == 3.0
lo, hi = s.invert_extent(2.0)
assert (lo, hi) == (0.0, 10.0)
print('OK')
"
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core
```

Expected: smoke prints `OK`; 67 Rust tests pass.

- [ ] **Step 11: Reject arity mismatch from Python**

```bash
unset CONDA_PREFIX && uv run --no-sync python -c "
from ferrum._core import ThresholdScale
try:
    ThresholdScale(domain=[0.0, 10.0], range=[1.0, 2.0])
    raise SystemExit('expected ValueError')
except ValueError as e:
    print('OK:', e)
"
```

Expected: prints `OK: range length must equal domain length + 1; got domain=2, range=2`.

- [ ] **Step 12: Commit**

```bash
git add crates/ferrum-core/src/scale crates/ferrum-core/src/lib.rs
git commit -m "feat(scale): ThresholdScale with invert_extent and arity validation"
```

---

### Task D3: QuantileScale

**Files:**
- Modify: `crates/ferrum-core/src/scale/core.rs` (add `Quantile` variant + arms + helper)
- Create: `crates/ferrum-core/src/scale/quantile.rs`
- Modify: `crates/ferrum-core/src/scale/mod.rs`
- Modify: `crates/ferrum-core/src/lib.rs`

- [ ] **Step 1: Add `Quantile` variant**

```rust
pub(crate) enum Scale {
    Linear   { domain: [f64; 2], range: [f64; 2], clamp: bool },
    Log      { domain: [f64; 2], range: [f64; 2], base: f64, clamp: bool },
    Symlog   { domain: [f64; 2], range: [f64; 2], constant: f64, clamp: bool },
    Ordinal  { domain: Vec<String>, range: Vec<f64>, padding: f64 },
    Threshold{ domain: Vec<f64>, range: Vec<f64> },
    Quantile { domain: Vec<f64>, range: Vec<f64>, quantiles: Vec<f64> },
}
```

- [ ] **Step 2: Add `compute_quantile_cuts` helper to `impl Scale`**

```rust
    pub(crate) fn compute_quantile_cuts(sorted_sample: &[f64], k: usize) -> Vec<f64> {
        // R-7 / numpy default: linear interpolation between order statistics.
        // Returns k-1 cut points dividing the sample into k bins.
        if k <= 1 || sorted_sample.is_empty() { return Vec::new(); }
        let n = sorted_sample.len();
        let mut cuts = Vec::with_capacity(k - 1);
        for i in 1..k {
            let p = (i as f64) / (k as f64);
            let h = p * (n as f64 - 1.0);
            let lo = h.floor() as usize;
            let hi = (h.ceil() as usize).min(n - 1);
            let frac = h - h.floor();
            let v = sorted_sample[lo] * (1.0 - frac) + sorted_sample[hi] * frac;
            cuts.push(v);
        }
        cuts
    }
```

- [ ] **Step 3: Add `Quantile` arms in dispatch methods**

In `scale_f64`, after `Threshold`:

```rust
            Scale::Quantile { range, quantiles, .. } => {
                if x.is_nan() { return f64::NAN; }
                let idx = quantiles.partition_point(|q| *q <= x);
                range[idx]
            }
```

In `invert_f64`, after `Threshold`:

```rust
            Scale::Quantile { .. } => f64::NAN,
```

In `ticks`, after `Threshold`:

```rust
            Scale::Quantile { domain, quantiles, .. } => {
                let target = count.unwrap_or_else(|| crate::scale::ticks::sturges_floor(domain.len()));
                if target >= quantiles.len() {
                    quantiles.clone()
                } else {
                    // Sample evenly from the cuts
                    let step = quantiles.len() as f64 / target as f64;
                    (0..target)
                        .map(|i| quantiles[((i as f64 + 0.5) * step).floor() as usize])
                        .collect()
                }
            }
```

In `nice`, after `Threshold`:

```rust
            Scale::Quantile { domain, range, quantiles } => {
                Scale::Quantile { domain, range, quantiles }
            }
```

- [ ] **Step 4: Extend `invert_extent` to handle `Quantile`**

Replace the body of `invert_extent`:

```rust
    pub(crate) fn invert_extent(&self, y: f64) -> (f64, f64) {
        match self {
            Scale::Threshold { domain, range } => {
                if y.is_nan() { return (f64::NAN, f64::NAN); }
                let idx = match range.iter().position(|r| *r == y) {
                    Some(i) => i,
                    None => return (f64::NAN, f64::NAN),
                };
                let lo = if idx == 0 { f64::NEG_INFINITY } else { domain[idx - 1] };
                let hi = if idx >= domain.len() { f64::INFINITY } else { domain[idx] };
                (lo, hi)
            }
            Scale::Quantile { range, quantiles, .. } => {
                if y.is_nan() { return (f64::NAN, f64::NAN); }
                let idx = match range.iter().position(|r| *r == y) {
                    Some(i) => i,
                    None => return (f64::NAN, f64::NAN),
                };
                let lo = if idx == 0 { f64::NEG_INFINITY } else { quantiles[idx - 1] };
                let hi = if idx >= quantiles.len() { f64::INFINITY } else { quantiles[idx] };
                (lo, hi)
            }
            _ => (f64::NAN, f64::NAN),
        }
    }
```

- [ ] **Step 5: Add `validate_quantile` to `core.rs`**

Below `validate_threshold`:

```rust
pub(crate) fn validate_quantile(domain: &[f64], range: &[f64]) -> PyResult<()> {
    if range.is_empty() {
        return Err(PyValueError::new_err("range must be non-empty"));
    }
    if domain.len() < 2 {
        return Err(PyValueError::new_err(format!(
            "domain (sample) must have length >= 2; got {}",
            domain.len()
        )));
    }
    validate_finite("domain", domain)?;
    validate_finite("range", range)?;
    Ok(())
}
```

- [ ] **Step 6: Add Quantile tests to `core.rs` `mod tests`**

```rust
    #[test]
    fn test_quantile_cuts_known_values() {
        // sample [1, 2, 3, 4, 5], 3 bins → cuts at p=1/3 and p=2/3
        // R-7: h_1/3 = (1/3)*4 = 1.333; sample[1]*0.667 + sample[2]*0.333 = 2*0.667 + 3*0.333 = 2.333
        // h_2/3 = (2/3)*4 = 2.667; sample[2]*0.333 + sample[3]*0.667 = 3*0.333 + 4*0.667 = 3.667
        let sample = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cuts = Scale::compute_quantile_cuts(&sample, 3);
        assert_eq!(cuts.len(), 2);
        assert!((cuts[0] - 7.0/3.0).abs() < 1e-9, "got {}", cuts[0]);
        assert!((cuts[1] - 11.0/3.0).abs() < 1e-9, "got {}", cuts[1]);
    }

    #[test]
    fn test_quantile_scale_basic() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cuts = Scale::compute_quantile_cuts(&sorted, 3);
        let s = Scale::Quantile {
            domain: sorted.clone(),
            range: vec![10.0, 20.0, 30.0],
            quantiles: cuts,
        };
        assert_eq!(s.scale_f64(0.0), 10.0);  // below first cut
        assert_eq!(s.scale_f64(2.5), 20.0);  // middle bin
        assert_eq!(s.scale_f64(10.0), 30.0); // above last cut
    }

    #[test]
    fn test_quantile_invert_extent_round_trip() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cuts = Scale::compute_quantile_cuts(&sorted, 3);
        let s = Scale::Quantile {
            domain: sorted,
            range: vec![10.0, 20.0, 30.0],
            quantiles: cuts.clone(),
        };
        let (lo, hi) = s.invert_extent(20.0);
        assert!((lo - cuts[0]).abs() < 1e-9);
        assert!((hi - cuts[1]).abs() < 1e-9);
    }

    #[test]
    fn test_quantile_ticks_default_uses_sturges_floor() {
        // domain length = 5, sturges_floor(5) = ceil(log2(5)+1) = ceil(3.32) = 4
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cuts = Scale::compute_quantile_cuts(&sorted, 10); // 9 cuts
        let s = Scale::Quantile {
            domain: sorted,
            range: vec![0.0; 10],
            quantiles: cuts,
        };
        let t = s.ticks(None);
        // sturges_floor(5) = 4; len(quantiles) = 9; target < quantiles.len(), so we sample.
        assert_eq!(t.len(), 4, "expected 4 ticks, got {}: {t:?}", t.len());
    }

    #[test]
    fn test_validate_quantile_rejects_short_domain() {
        assert!(validate_quantile(&[1.0], &[0.0, 1.0]).is_err());
    }
```

- [ ] **Step 7: Run Rust tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core scale::core::
```

Expected: 35 tests pass (30 from D2 + 5 new).

- [ ] **Step 8: Create `crates/ferrum-core/src/scale/quantile.rs`**

```rust
use pyo3::prelude::*;

use super::core::{validate_quantile, Scale};

#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, PartialEq)]
pub struct QuantileScale(Scale);

impl QuantileScale {
    pub(crate) fn repr_string(&self) -> String {
        match &self.0 {
            Scale::Quantile { domain, range, quantiles } => format!(
                "QuantileScale(domain=<{} samples>, range={:?}, quantiles={:?})",
                domain.len(), range, quantiles
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

#[pymethods]
impl QuantileScale {
    #[new]
    #[pyo3(signature = (*, domain, range))]
    fn new(domain: Vec<f64>, range: Vec<f64>) -> PyResult<Self> {
        validate_quantile(&domain, &range)?;
        let mut sorted = domain.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let quantiles = Scale::compute_quantile_cuts(&sorted, range.len());
        Ok(QuantileScale(Scale::Quantile {
            domain: sorted,
            range,
            quantiles,
        }))
    }

    fn scale(&self, x: f64) -> f64 { self.0.scale_f64(x) }

    fn invert_extent(&self, y: f64) -> (f64, f64) { self.0.invert_extent(y) }

    #[pyo3(signature = (count = None))]
    fn ticks(&self, count: Option<usize>) -> Vec<f64> { self.0.ticks(count) }

    fn nice(&self) -> Self { self.clone() }

    #[getter]
    fn domain(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Quantile { domain, .. } => domain.clone(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    #[getter]
    fn range(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Quantile { range, .. } => range.clone(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    #[getter]
    fn quantiles(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Quantile { quantiles, .. } => quantiles.clone(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    fn __repr__(&self) -> String { self.repr_string() }
}
```

- [ ] **Step 9: Update `mod.rs`** to add `pub(crate) mod quantile;`.

- [ ] **Step 10: Update `lib.rs`** to add `m.add_class::<scale::quantile::QuantileScale>()?;`.

- [ ] **Step 11: Build + smoke + cargo test**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
unset CONDA_PREFIX && uv run --no-sync python -c "
from ferrum._core import QuantileScale
s = QuantileScale(domain=[1.0, 2.0, 3.0, 4.0, 5.0], range=[10.0, 20.0, 30.0])
assert s.scale(2.5) == 20.0, s.scale(2.5)
assert s.scale(0.0) == 10.0
assert s.scale(100.0) == 30.0
lo, hi = s.invert_extent(20.0)
assert lo < hi
print('OK')
"
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core
```

Expected: smoke prints `OK`; 72 Rust tests pass.

- [ ] **Step 12: Commit**

```bash
git add crates/ferrum-core/src/scale crates/ferrum-core/src/lib.rs
git commit -m "feat(scale): QuantileScale with R-7 cut computation and Sturges default ticks"
```

---

## Section E — Python boundary

### Task E1: Update `_core.pyi`

**Files:**
- Modify: `src/ferrum/_core.pyi`

- [ ] **Step 1: Replace `src/ferrum/_core.pyi` with the extended stubs**

```python
from typing import Any, Literal, Optional, Sequence, Union

DataTypeStr = Literal[
    "Q", "N", "O", "T",
    "quantitative", "nominal", "ordinal", "temporal",
]
MarkStr = Literal[
    "point", "line", "bar", "area", "rule", "text", "tick", "rect",
]


def process_batch(data: Any) -> Any: ...


class EncodingSpec:
    field: str
    type_: Optional[str]
    def __init__(self, field: str, type_: Optional[DataTypeStr] = None) -> None: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...


class ChartSpec:
    mark: str
    x: Optional[EncodingSpec]
    y: Optional[EncodingSpec]
    data: str

    def __init__(
        self,
        *,
        mark: MarkStr,
        x: Union[str, EncodingSpec, None] = None,
        y: Union[str, EncodingSpec, None] = None,
        data: Optional[str] = None,
    ) -> None: ...
    def to_json(self) -> str: ...
    @classmethod
    def from_json(cls, s: str) -> "ChartSpec": ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...


# ---------- Scales (Phase 4) ----------

class LinearScale:
    domain: list[float]
    range: list[float]
    clamp: bool
    def __init__(
        self,
        *,
        domain: Sequence[float],
        range: Sequence[float],
        clamp: bool = False,
        nice: bool = False,
    ) -> None: ...
    def scale(self, x: float) -> float: ...
    def invert(self, y: float) -> float: ...
    def ticks(self, count: int = 10) -> list[float]: ...
    def nice(self) -> "LinearScale": ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...


class LogScale:
    domain: list[float]
    range: list[float]
    base: float
    clamp: bool
    def __init__(
        self,
        *,
        domain: Sequence[float],
        range: Sequence[float],
        base: float = 10.0,
        clamp: bool = False,
        nice: bool = False,
    ) -> None: ...
    def scale(self, x: float) -> float: ...
    def invert(self, y: float) -> float: ...
    def ticks(self, count: int = 10) -> list[float]: ...
    def nice(self) -> "LogScale": ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...


class TimeScale:
    domain: list[float]
    range: list[float]
    clamp: bool
    def __init__(
        self,
        *,
        domain: Sequence[float],
        range: Sequence[float],
        clamp: bool = False,
        nice: bool = False,
    ) -> None: ...
    def scale(self, x: float) -> float: ...
    def invert(self, y: float) -> float: ...
    def ticks(self, count: int = 10) -> list[float]: ...
    def nice(self) -> "TimeScale": ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...


class SymlogScale:
    domain: list[float]
    range: list[float]
    constant: float
    clamp: bool
    def __init__(
        self,
        *,
        domain: Sequence[float],
        range: Sequence[float],
        constant: float = 1.0,
        clamp: bool = False,
        nice: bool = False,
    ) -> None: ...
    def scale(self, x: float) -> float: ...
    def invert(self, y: float) -> float: ...
    def ticks(self, count: int = 10) -> list[float]: ...
    def nice(self) -> "SymlogScale": ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...


class OrdinalScale:
    domain: list[str]
    range: list[float]
    padding: float
    def __init__(
        self,
        *,
        domain: Sequence[str],
        range: Sequence[float],
        padding: float = 0.0,
    ) -> None: ...
    def scale(self, value: str) -> float: ...
    def invert(self, y: float) -> Optional[str]: ...
    def ticks(self) -> list[str]: ...
    def nice(self) -> "OrdinalScale": ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...


class ThresholdScale:
    domain: list[float]
    range: list[float]
    def __init__(
        self,
        *,
        domain: Sequence[float],
        range: Sequence[float],
    ) -> None: ...
    def scale(self, x: float) -> float: ...
    def invert_extent(self, y: float) -> tuple[float, float]: ...
    def ticks(self) -> list[float]: ...
    def nice(self) -> "ThresholdScale": ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...


class QuantileScale:
    domain: list[float]
    range: list[float]
    quantiles: list[float]
    def __init__(
        self,
        *,
        domain: Sequence[float],
        range: Sequence[float],
    ) -> None: ...
    def scale(self, x: float) -> float: ...
    def invert_extent(self, y: float) -> tuple[float, float]: ...
    def ticks(self, count: Optional[int] = None) -> list[float]: ...
    def nice(self) -> "QuantileScale": ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
```

- [ ] **Step 2: Commit**

```bash
git add src/ferrum/_core.pyi
git commit -m "feat(scale): _core.pyi stubs for the seven scale classes"
```

---

### Task E2: Re-export from `ferrum.__init__`

**Files:**
- Modify: `src/ferrum/__init__.py`

- [ ] **Step 1: Read the current `__init__.py`**

```bash
cat src/ferrum/__init__.py
```

Expected output (post-Phase 3): an `__init__.py` that re-exports `ChartSpec` and `EncodingSpec`.

- [ ] **Step 2: Add the seven scale classes to the re-export list**

Replace the import block in `src/ferrum/__init__.py` so it reads:

```python
from ferrum._core import (
    ChartSpec,
    EncodingSpec,
    LinearScale,
    LogScale,
    TimeScale,
    SymlogScale,
    OrdinalScale,
    QuantileScale,
    ThresholdScale,
    process_batch,
)

__all__ = [
    "ChartSpec",
    "EncodingSpec",
    "LinearScale",
    "LogScale",
    "TimeScale",
    "SymlogScale",
    "OrdinalScale",
    "QuantileScale",
    "ThresholdScale",
    "process_batch",
]
```

> If the existing file uses a different style (e.g., `from ferrum import _core; ChartSpec = _core.ChartSpec`), keep the existing style and just add the seven new names alongside.

- [ ] **Step 3: Verify the re-exports work**

```bash
unset CONDA_PREFIX && uv run --no-sync python -c "
import ferrum
from ferrum import (LinearScale, LogScale, TimeScale, SymlogScale,
                    OrdinalScale, QuantileScale, ThresholdScale)
print('OK')
"
```

Expected: prints `OK`.

- [ ] **Step 4: Commit**

```bash
git add src/ferrum/__init__.py
git commit -m "feat(scale): re-export seven scale classes from ferrum package"
```

---

### Task E3: Write `tests/test_scales.py`

**Files:**
- Create: `tests/test_scales.py`

- [ ] **Step 1: Create `tests/test_scales.py`**

```python
"""Smoke + boundary tests for Phase 4 scales.

Math correctness is covered by `cargo test`; these tests verify the
Python boundary conveys f64s, errors propagate as ValueError, and the
public API surface matches the spec.
"""

import math
import pytest

from ferrum import (
    LinearScale, LogScale, TimeScale, SymlogScale,
    OrdinalScale, QuantileScale, ThresholdScale,
)


# ---------- construction smoke ----------

def test_linear_construct_and_basic_scale():
    s = LinearScale(domain=[0.0, 10.0], range=[0.0, 1.0])
    assert math.isclose(s.scale(5.0), 0.5)
    assert math.isclose(s.invert(0.5), 5.0)
    assert s.domain == [0.0, 10.0]
    assert s.range == [0.0, 1.0]
    assert s.clamp is False


def test_log_construct_and_basic_scale():
    s = LogScale(domain=[1.0, 1000.0], range=[0.0, 3.0])
    assert math.isclose(s.scale(10.0), 1.0)
    assert math.isclose(s.invert(2.0), 100.0)
    assert s.base == 10.0


def test_time_construct_and_round_trip():
    s = TimeScale(domain=[0.0, 86400000.0], range=[0.0, 1.0])
    mid = 43200000.0
    assert math.isclose(s.scale(mid), 0.5)
    assert math.isclose(s.invert(0.5), mid)


def test_symlog_handles_zero():
    s = SymlogScale(domain=[-100.0, 100.0], range=[0.0, 1.0])
    assert math.isclose(s.scale(0.0), 0.5)
    assert s.constant == 1.0


def test_ordinal_basic():
    s = OrdinalScale(domain=["a", "b", "c"], range=[0.0, 30.0])
    assert math.isclose(s.scale("a"), 5.0)
    assert math.isclose(s.scale("b"), 15.0)
    assert math.isclose(s.scale("c"), 25.0)
    assert s.invert(5.0) == "a"
    assert s.invert(100.0) is None
    assert s.ticks() == ["a", "b", "c"]


def test_threshold_basic():
    s = ThresholdScale(domain=[0.0, 10.0], range=[1.0, 2.0, 3.0])
    assert s.scale(-1.0) == 1.0
    assert s.scale(5.0) == 2.0
    assert s.scale(11.0) == 3.0
    assert s.invert_extent(2.0) == (0.0, 10.0)


def test_quantile_basic():
    s = QuantileScale(domain=[1.0, 2.0, 3.0, 4.0, 5.0], range=[10.0, 20.0, 30.0])
    assert s.scale(0.0) == 10.0
    assert s.scale(2.5) == 20.0
    assert s.scale(100.0) == 30.0
    lo, hi = s.invert_extent(20.0)
    assert lo < hi
    assert len(s.quantiles) == 2


# ---------- inversion round trips ----------

def test_continuous_inversion_round_trip():
    for cls, kwargs in [
        (LinearScale, dict(domain=[0.0, 100.0], range=[-1.0, 1.0])),
        (LogScale, dict(domain=[1.0, 1e6], range=[0.0, 6.0])),
        (TimeScale, dict(domain=[0.0, 1e10], range=[0.0, 1.0])),
        (SymlogScale, dict(domain=[-100.0, 100.0], range=[0.0, 1.0])),
    ]:
        s = cls(**kwargs)
        for x in [kwargs["domain"][0], (kwargs["domain"][0] + kwargs["domain"][1]) / 2, kwargs["domain"][1]]:
            y = s.scale(x)
            back = s.invert(y)
            assert math.isclose(back, x, rel_tol=1e-6, abs_tol=1e-6), f"{cls.__name__}: x={x} → y={y} → back={back}"


# ---------- nan propagation ----------

def test_scale_nan_propagates():
    s = LinearScale(domain=[0.0, 10.0], range=[0.0, 1.0])
    assert math.isnan(s.scale(float("nan")))
    assert math.isnan(s.invert(float("nan")))


def test_scale_out_of_domain_returns_nan_when_unclamped():
    s = LinearScale(domain=[0.0, 10.0], range=[0.0, 1.0])
    assert math.isnan(s.scale(-1.0))
    assert math.isnan(s.scale(11.0))


def test_scale_clamp_clamps_output():
    s = LinearScale(domain=[0.0, 10.0], range=[0.0, 1.0], clamp=True)
    assert s.scale(-1.0) == 0.0
    assert s.scale(11.0) == 1.0


# ---------- constructor errors ----------

def test_linear_rejects_wrong_domain_length():
    with pytest.raises(ValueError, match="domain must have length 2"):
        LinearScale(domain=[0.0, 1.0, 2.0], range=[0.0, 1.0])


def test_linear_rejects_degenerate_domain():
    with pytest.raises(ValueError, match="domain endpoints must differ"):
        LinearScale(domain=[5.0, 5.0], range=[0.0, 1.0])


def test_log_rejects_zero_in_domain():
    with pytest.raises(ValueError, match="must not contain 0"):
        LogScale(domain=[0.0, 100.0], range=[0.0, 2.0])


def test_log_rejects_mixed_signs():
    with pytest.raises(ValueError, match="same sign"):
        LogScale(domain=[-1.0, 100.0], range=[0.0, 2.0])


def test_log_rejects_invalid_base():
    with pytest.raises(ValueError, match="base must be"):
        LogScale(domain=[1.0, 100.0], range=[0.0, 2.0], base=1.0)


def test_symlog_rejects_invalid_constant():
    with pytest.raises(ValueError, match="constant must be"):
        SymlogScale(domain=[-1.0, 1.0], range=[0.0, 1.0], constant=0.0)


def test_ordinal_rejects_empty_domain():
    with pytest.raises(ValueError, match="domain must be non-empty"):
        OrdinalScale(domain=[], range=[0.0, 10.0])


def test_ordinal_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        OrdinalScale(domain=["a", "a"], range=[0.0, 10.0])


def test_ordinal_rejects_bad_padding():
    with pytest.raises(ValueError, match="padding"):
        OrdinalScale(domain=["a"], range=[0.0, 10.0], padding=1.5)


def test_threshold_rejects_arity_mismatch():
    with pytest.raises(ValueError, match="range length must equal domain length"):
        ThresholdScale(domain=[0.0, 10.0], range=[1.0, 2.0])


def test_threshold_rejects_unsorted_domain():
    with pytest.raises(ValueError, match="strictly sorted"):
        ThresholdScale(domain=[10.0, 0.0], range=[1.0, 2.0, 3.0])


def test_quantile_rejects_short_domain():
    with pytest.raises(ValueError, match="length >= 2"):
        QuantileScale(domain=[1.0], range=[0.0, 1.0])


# ---------- ticks ----------

def test_linear_ticks_default_count():
    s = LinearScale(domain=[0.0, 10.0], range=[0.0, 1.0])
    t = s.ticks()
    assert len(t) >= 5


def test_quantile_ticks_default_uses_sturges():
    # domain length 100, sturges = ceil(log2(100)+1) = 8
    sample = [float(i) for i in range(100)]
    s = QuantileScale(domain=sample, range=[float(i) for i in range(20)])
    t = s.ticks()
    # default count should be sturges_floor(100) = 8
    assert len(t) == 8, f"expected 8 (sturges of 100), got {len(t)}: {t}"


def test_threshold_ticks_returns_thresholds():
    s = ThresholdScale(domain=[0.0, 10.0, 20.0], range=[1.0, 2.0, 3.0, 4.0])
    assert s.ticks() == [0.0, 10.0, 20.0]


# ---------- nice ----------

def test_linear_nice_extends_domain():
    s = LinearScale(domain=[0.13, 9.7], range=[0.0, 1.0]).nice()
    assert s.domain[0] <= 0.13
    assert s.domain[1] >= 9.7


def test_ordinal_nice_is_identity():
    s = OrdinalScale(domain=["a", "b"], range=[0.0, 10.0])
    n = s.nice()
    assert n.domain == s.domain
    assert n.range == s.range
```

- [ ] **Step 2: Run the new Python tests**

```bash
uv run pytest tests/test_scales.py -v
```

Expected: 28 tests pass (7 smoke + 1 round-trip + 3 NaN/clamp + 12 constructor errors + 3 ticks + 2 nice).

- [ ] **Step 3: Run the full Python suite**

```bash
uv run pytest
```

Expected: 46 tests pass (18 baseline + 28 new).

- [ ] **Step 4: Commit**

```bash
git add tests/test_scales.py
git commit -m "test: Phase 4 Python smoke and boundary tests for seven scales"
```

---

## Section F — Closure

### Task F1: Update `ferrum-phases.md`

**Files:**
- Modify: `docs/superpowers/ferrum-phases.md`

- [ ] **Step 1: Update the Phase 4 row in the phases table**

In `docs/superpowers/ferrum-phases.md`, find the Phase 4 row in the phase table (around line 63):

Replace:

```
| **4** | Scale engine | `LinearScale`, `LogScale`, `TimeScale`, `OrdinalScale`, `QuantileScale`, `ThresholdScale`, `SymlogScale`; domain/range mapping, tick generation | 3 | *(not yet written)* | pending |
```

with:

```
| **4** | Scale engine | `LinearScale`, `LogScale`, `TimeScale`, `OrdinalScale`, `QuantileScale`, `ThresholdScale`, `SymlogScale`; domain/range mapping, tick generation | 3 | [`2026-05-09-scale-engine-design.md`](specs/2026-05-09-scale-engine-design.md) | **done** |
```

- [ ] **Step 2: Update the "Last updated" line at the top**

Change `**Last updated:** 2026-05-09` to `**Last updated:** 2026-05-09` (no change if same day; otherwise reflect current date).

- [ ] **Step 3: Tick the done-criteria checkboxes for Phase 4**

In the "Phase 4 — Scale engine" done-criteria section (around line 97), replace each `- [ ]` with `- [x]`:

```
### Phase 4 — Scale engine
- [x] All seven scale types are implemented in Rust and exposed via `ferrum._core`
- [x] Domain/range mapping is correct for boundary values (including log(0), symlog threshold, ordinal padding)
- [x] Tick generation passes the spec's "Sturges floor" requirement for binning
- [x] Python-facing type stubs in `_core.pyi` cover all scale constructors
- [x] `cargo test` covers at least one inversion test per scale type
```

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/ferrum-phases.md
git commit -m "chore: mark Phase 4 done; link scale-engine spec"
```

---

### Task F2: Final verification

**Files:** none modified.

- [ ] **Step 1: Run the full Rust test suite**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core
```

Expected: ≥ 50 tests pass (target was 50; actual is around 72 with all sub-tests).

- [ ] **Step 2: Run the full Python test suite**

```bash
uv run pytest
```

Expected: 46 tests pass (18 baseline + 28 new).

- [ ] **Step 3: Verify import paths from a fresh shell**

```bash
unset CONDA_PREFIX && uv run --no-sync python -c "
from ferrum._core import (
    LinearScale, LogScale, TimeScale, SymlogScale,
    OrdinalScale, QuantileScale, ThresholdScale,
)
from ferrum import (
    LinearScale as L, LogScale as Lo, TimeScale as T, SymlogScale as Sy,
    OrdinalScale as O, QuantileScale as Q, ThresholdScale as Th,
)
assert L is LinearScale
print('all seven scales importable from ferrum and ferrum._core')
"
```

Expected: prints `all seven scales importable from ferrum and ferrum._core`.

- [ ] **Step 4: Confirm no new external dependencies were added**

```bash
git diff main -- crates/ferrum-core/Cargo.toml Cargo.toml
```

Expected: empty diff. If anything changed in the dependency tables, investigate; the spec requires no new deps.

- [ ] **Step 5: Confirm matplotlib has not been pulled in**

```bash
uv run python -c "
try:
    import matplotlib
    raise SystemExit(f'FAIL: matplotlib found at {matplotlib.__file__}')
except ImportError:
    print('OK: no matplotlib')
"
```

Expected: prints `OK: no matplotlib`.

- [ ] **Step 6: Print a diff summary for review**

```bash
git log main..HEAD --oneline
git diff --stat main..HEAD
```

Expected: roughly 13 commits (1 skeleton, 1 ticks, 7 scales, 1 .pyi, 1 __init__, 1 tests, 1 phases-doc) and ~2000 lines added across `crates/ferrum-core/src/scale/`, `src/ferrum/_core.pyi`, `src/ferrum/__init__.py`, `tests/test_scales.py`, `docs/superpowers/ferrum-phases.md`.

- [ ] **Step 7: Confirm with the user before merging**

The branch is ready for review. Do not push or merge without explicit user approval. Per `CLAUDE.md`: "Do not `git push` unless the user explicitly asks."

---

## Done

When all tasks above are complete and Task F2 verification passes, Phase 4 is `done`. The branch `feat/scale-engine` is ready to merge into `main`.
