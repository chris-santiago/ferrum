# Phase 4 — Scale Engine — Design

**Status:** approved 2026-05-09
**Phase:** 4 (depends on Phase 3)
**Slug:** `scale-engine`
**Implementation plan:** *(to be written by `superpowers:writing-plans`)*

---

## 1. Goals

Phase 4 produces the seven scale primitives Phase 7's static renderer will need. It is a pure-Rust math library with PyO3 bindings — no chart spec changes, no rendering, no Arrow data flow.

The phase is `done` when every item in §11 is verifiable.

## 2. Scope

### In scope
The seven scale types named in `docs/superpowers/ferrum-phases.md` Phase 4 done criteria:

`LinearScale`, `LogScale`, `TimeScale`, `SymlogScale`, `OrdinalScale`, `QuantileScale`, `ThresholdScale`.

Each is constructible from Python, each provides domain → range mapping, inversion (signature varies by group), tick generation, and `nice()` where meaningful.

### Out of scope (deferred)

- The other nine scales in `ferrum-spec.md §3.6` — `ScalePow`, `ScaleSqrt`, `ScaleUtc`, `ScalePoint`, `ScaleBand`, `ScaleSequential`, `ScaleDiverging`, `ScaleQuantize`, `ScaleBinOrdinal`. Picked up in a Phase 4.5 follow-up or absorbed into Phase 8 (grammar API) when channels demand them.
- Color schemes (`ferrum.schemes`) — no scale outputs colors in Phase 4; ranges are `Vec<f64>` universally.
- JSON serialization for scales — added in Phase 7 when the renderer needs scale persistence.
- `EncodingSpec.scale` field — not added; Phase 7 or 8 wires the attachment.
- Custom user-defined scales — Phase 12 territory.
- Calendar-aware time ticks (month/year boundaries that respect varying day counts) — deferred with `ScaleUtc`.

## 3. Locked decisions

| # | Decision | Choice |
|---|---|---|
| 1 | Scale count | 7 |
| 2 | Behavior surface | `scale`, `invert` / `invert_extent`, `ticks`, `nice` |
| 3 | Wiring | standalone primitives; no IR/encoding changes; no JSON |
| 4 | Range type | `Vec<f64>` for all scales |
| 5 | Time representation | `f64` milliseconds since Unix epoch |
| 6 | Sturges floor | default tick count for `QuantileScale` / `ThresholdScale` is `max(⌈log2(n)+1⌉, 1)`; helper reused by Phase 5 binning |
| 7 | Python class shape | seven distinct `#[pyclass]` types |
| 8 | Inversion API | continuous: `invert(y) -> f64`; bin (`Quantile`/`Threshold`): `invert_extent(y) -> (f64, f64)`; ordinal: `invert(y) -> Optional[str]` |
| 9 | Errors | `PyValueError` at construct (static violations); `f64::NAN` at runtime (out-of-domain) — matches d3.js |
| 10 | Internal Rust shape | sealed `enum Scale { Linear{..}, Log{..}, ... }` plus seven thin pyclass facades |
| 11 | External dependencies | none added (no `chrono`, no `nalgebra`); std-only math |

## 4. Public Python API

All seven types live in `ferrum._core` and are re-exported from `ferrum.` (matching the Phase 3 precedent set for `ChartSpec` / `EncodingSpec`).

### 4.1 Continuous group: `LinearScale`, `LogScale`, `TimeScale`, `SymlogScale`

```python
LinearScale(*, domain: Sequence[float], range: Sequence[float],
            clamp: bool = False, nice: bool = False)

LogScale(*, domain: Sequence[float], range: Sequence[float],
         base: float = 10.0, clamp: bool = False, nice: bool = False)

TimeScale(*, domain: Sequence[float], range: Sequence[float],
          clamp: bool = False, nice: bool = False)

SymlogScale(*, domain: Sequence[float], range: Sequence[float],
            constant: float = 1.0, clamp: bool = False, nice: bool = False)
```

Methods:

| Method | Signature | Behavior |
|---|---|---|
| `scale` | `(value: float) -> float` | domain → range; out-of-domain returns `NaN` unless `clamp=True` |
| `invert` | `(y: float) -> float` | range → domain; out-of-range returns `NaN` unless `clamp=True` |
| `ticks` | `(count: int = 10) -> list[float]` | d3-style nice tick array |
| `nice` | `() -> Self` | returns a *new* scale with niced domain (no in-place mutation; "themes are values" principle) |

Read-only `@getter` properties: `domain`, `range`, plus scale-specific config (`base` on `LogScale`, `constant` on `SymlogScale`, `clamp` on all four). `__repr__` and `__eq__` follow Phase 3 conventions.

`domain` and `range` must be length 2 (`[lo, hi]`); the constructor enforces this.

### 4.2 Ordinal group: `OrdinalScale`

```python
OrdinalScale(*, domain: Sequence[str], range: Sequence[float],
             padding: float = 0.0)
```

| Method | Signature | Behavior |
|---|---|---|
| `scale` | `(value: str) -> float` | category → band-center position within the range extent `[range[0], range[-1]]`; unknown category returns `NaN` |
| `invert` | `(y: float) -> Optional[str]` | the category whose band contains `y`; `None` if outside the band extent |
| `ticks` | `() -> list[str]` | returns the categories (categories *are* the ticks) |
| `nice` | `() -> Self` | identity (returns a clone) |

Getters: `domain`, `range`, `padding`. `padding` is the inter-band gap as a fraction of the per-band step.

### 4.3 Bin group: `QuantileScale`, `ThresholdScale`

```python
QuantileScale(*, domain: Sequence[float], range: Sequence[float])
# domain = sample data (any length >= 2); k-1 quantile cut points computed at construct,
# where k = len(range)

ThresholdScale(*, domain: Sequence[float], range: Sequence[float])
# domain = k-1 thresholds, sorted ascending; range length must be k
```

| Method | Signature | Behavior |
|---|---|---|
| `scale` | `(value: float) -> float` | input value → range value of containing bin; `NaN` propagates |
| `invert_extent` | `(y: float) -> tuple[float, float]` | `(lo, hi)` of the input range that maps to `y`; `(NaN, NaN)` if `y` is not in range |
| `ticks` | `(count: Optional[int] = None) -> list[float]` | Quantile: cached cut points (defaults to Sturges floor of domain size). Threshold: returns the thresholds themselves |
| `nice` | `() -> Self` | identity |

Getters: `domain`, `range`. `QuantileScale` additionally exposes `quantiles` (the cached cut points).

### 4.4 Type stubs (`src/ferrum/_core.pyi`)

`_core.pyi` adds entries for all seven classes mirroring the signatures above, using `Literal` / `Sequence` / `Optional` from `typing`. No new `Literal[...]` aliases are needed — scales don't use enum strings.

## 5. Internal Rust shape

### 5.1 File layout

Extends Phase 3's `spec/` pattern.

```
crates/ferrum-core/src/
├── lib.rs                  # add 7 pyclass registrations
├── transport.rs            # unchanged
├── spec/                   # unchanged (Phase 3)
└── scale/
    ├── mod.rs              # pub(crate) re-exports
    ├── core.rs             # sealed `enum Scale { ... }` + impl Scale (math dispatch)
    ├── ticks.rs            # nice-ticks helper, Sturges helper, time-interval helper
    ├── linear.rs           # pyclass LinearScale, wraps Scale::Linear
    ├── log.rs              # pyclass LogScale, wraps Scale::Log
    ├── time.rs             # pyclass TimeScale, wraps Scale::Time
    ├── symlog.rs           # pyclass SymlogScale, wraps Scale::Symlog
    ├── ordinal.rs          # pyclass OrdinalScale, wraps Scale::Ordinal
    ├── quantile.rs         # pyclass QuantileScale, wraps Scale::Quantile
    └── threshold.rs        # pyclass ThresholdScale, wraps Scale::Threshold
```

`ticks.rs` is shared infrastructure — pulled out so Phase 5 (binning) can call `sturges_floor(n)` without importing the scale module.

### 5.2 Sealed enum

```rust
// scale/core.rs
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Scale {
    Linear   { domain: [f64; 2], range: [f64; 2], clamp: bool },
    Log      { domain: [f64; 2], range: [f64; 2], base: f64, clamp: bool },
    Time     { domain: [f64; 2], range: [f64; 2], clamp: bool },
    Symlog   { domain: [f64; 2], range: [f64; 2], constant: f64, clamp: bool },
    Ordinal  { domain: Vec<String>, range: Vec<f64>, padding: f64 },
    Quantile { domain: Vec<f64>, range: Vec<f64>, quantiles: Vec<f64> },
    Threshold{ domain: Vec<f64>, range: Vec<f64> },
}

impl Scale {
    pub(crate) fn scale_f64(&self, x: f64) -> f64 { /* match */ }
    pub(crate) fn invert_f64(&self, y: f64) -> f64 { /* continuous variants only */ }
    pub(crate) fn invert_extent(&self, y: f64) -> (f64, f64) { /* Quantile/Threshold only */ }
    pub(crate) fn scale_str(&self, s: &str) -> f64 { /* Ordinal only */ }
    pub(crate) fn invert_band(&self, y: f64) -> Option<&str> { /* Ordinal only */ }
    pub(crate) fn ticks(&self, count: Option<usize>) -> Vec<f64> { /* match */ }
    pub(crate) fn nice(self) -> Self { /* match — Linear/Log/Time/Symlog return niced; others identity */ }
}
```

The variant-specific methods are guarded by `unreachable!()` paths for variants that shouldn't reach them. Those paths are never reached in practice because the pyclass surface enforces which methods exist on which type.

### 5.3 Pyclass facade pattern

Each pyclass is a thin newtype wrapper:

```rust
#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, PartialEq)]
pub struct LinearScale(Scale);

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
        if nice { s = s.nice(); }
        Ok(LinearScale(s))
    }
    fn scale(&self, x: f64) -> f64 { self.0.scale_f64(x) }
    fn invert(&self, y: f64) -> f64 { self.0.invert_f64(y) }
    fn ticks(&self, count: Option<usize>) -> Vec<f64> { self.0.ticks(count) }
    fn nice(&self) -> Self { LinearScale(self.0.clone().nice()) }
    // getters, __repr__
}
```

Methods that don't apply to a variant are simply *not exposed* — `LinearScale` has no `invert_extent`, `QuantileScale` has no `invert`. Phase 3's `repr_string()` helper pattern transfers directly: each pyclass has a `pub(crate) fn repr_string(&self) -> String` that `__repr__` delegates to.

## 6. Math semantics

### 6.1 Linear

`scale(x) = r0 + (x - d0) * (r1 - r0) / (d1 - d0)`

`clamp=true` clamps the *output*, not the input (d3 convention). Inversion is the symmetric formula. No special edge cases beyond `lo == hi` (rejected at construct).

### 6.2 Log

`scale(x) = linear_scale(log_base(x))` over `[log_base(d0), log_base(d1)]`. The constructor rejects:
- `domain` containing `0`
- `domain` values with mixed signs (e.g., `[-1, 1]`)
- `base ≤ 0` or `base == 1`

Default `base = 10.0`. Negative-only domains (e.g., `[-100, -1]`) are supported via `log_base(|x|)` with sign tracking — the implementation negates internally, mirroring d3.

### 6.3 Time

Identical to Linear arithmetically (domain values are `f64` ms-epoch). Tick generation differs — see §7. No timezone support in Phase 4; output is UTC-equivalent ms.

### 6.4 Symlog

d3's transform: `f(x) = sign(x) * log_base(|x|/c + 1)` where `c = constant`. Inversion is the symmetric formula. `constant` must be `> 0`. Handles zero and negative values without rejection. The implementation uses `f64::ln_1p` for numeric stability near zero.

### 6.5 Ordinal

`domain` is `Vec<String>`. The range parameter is treated as a 1D extent: only `range[0]` and `range[-1]` are used as endpoints; any intermediate values are ignored (they are accepted to keep the constructor signature uniform with the other scale groups, but carry no semantics here). The extent is divided into `len(domain)` equal bands with `padding` fraction subtracted from each side. `scale(s)` returns the band *center*.

`padding` is the fraction of the per-band step removed equally from inner gaps.

Constructor rejects:
- empty domain
- duplicate categories
- `padding < 0` or `padding > 1`
- `range` length below 2 (need extent endpoints)

### 6.6 Quantile

At construct time:
1. Sort `domain` (sample) ascending.
2. Compute `len(range) - 1` quantile cut points using R-7 / numpy default (linear interpolation between order statistics).
3. Cache cut points in the enum variant.

`scale(x)` performs `slice::partition_point` on the cached cut points to find the bin index, returning `range[bin_index]`. Out-of-range inputs (`x < min` or `x > max` of the original sample) clip to first/last bin (Quantile is defined over all of ℝ once cut points are fixed).

`invert_extent(y)`:
- Find the first bin index `i` where `range[i] == y` (linear scan; `range` is small).
- Return `(quantile_cut[i-1], quantile_cut[i])`, with `(-∞, quantile_cut[0])` for the first bin and `(quantile_cut[-1], +∞)` for the last.
- If `y` is not in `range`, return `(NaN, NaN)`.

Constructor rejects:
- `domain.len() < 2`
- `range.len() < 1`
- non-finite values in `domain` or `range`

### 6.7 Threshold

`domain` is `k-1` sorted thresholds; `range` is `k` values. `scale(x)` is `range[bisect_left(domain, x)]` (via `slice::partition_point`).

`invert_extent(y)`:
- Find first bin index `i` where `range[i] == y` (linear scan).
- Return `(domain[i-1], domain[i])`, with `(-∞, domain[0])` for the first bin and `(domain[-1], +∞)` for the last.
- `(NaN, NaN)` if `y` not in `range`.

Constructor rejects:
- `domain` not strictly sorted ascending
- `range.len() != domain.len() + 1`
- non-finite values in `domain` or `range`

## 7. Tick generation

### 7.1 Shared helpers (`scale/ticks.rs`)

```rust
pub(crate) fn nice_ticks(d_lo: f64, d_hi: f64, count: usize) -> Vec<f64>;
pub(crate) fn nice_step(d_lo: f64, d_hi: f64, count: usize) -> f64;
pub(crate) fn sturges_floor(n: usize) -> usize;  // max(ceil(log2(n) + 1), 1)
pub(crate) fn nice_time_interval_ms(span_ms: f64, count: usize) -> f64;
```

`nice_ticks` follows d3's algorithm: pick a `1/2/5 × 10^k` step that produces approximately `count` evenly-spaced ticks within `[d_lo, d_hi]`. `nice_step` exposes the chosen step for `nice()` to round to.

`sturges_floor` is `pub(crate)` so Phase 5's binning module can reuse it.

### 7.2 Per-scale tick behavior

| Scale | `ticks(count)` |
|---|---|
| Linear | `nice_ticks(d_lo, d_hi, count.unwrap_or(10))` |
| Log | If `domain` spans ≥ `count` decades: integer multiples of `base` within domain. Else: fall back to `nice_ticks` of the linearized values. |
| Time | Choose a nice time interval (`1s`, `1m`, `1h`, `1d`, `1w`, ~`1mo`, ~`1y`) that yields ~`count` ticks; emit tick instants. Month/year intervals use 30d / 365d approximations in Phase 4 (calendar-accurate intervals deferred). |
| Symlog | Falls back to `nice_ticks` over the domain (matches d3 — symlog has no meaningful tick algorithm of its own). |
| Ordinal | Ignores `count`; returns the categories. |
| Quantile | `count = count.unwrap_or(sturges_floor(domain.len()))`; returns the cached quantile cut points (truncated/extended to `count`). |
| Threshold | Ignores `count`; returns the thresholds verbatim. |

### 7.3 `nice()`

| Scale | `nice()` behavior |
|---|---|
| Linear | Extends domain to align with the chosen tick step (per d3). |
| Log | Rounds domain endpoints to powers of `base`. |
| Time | Rounds to the chosen tick interval. |
| Symlog | Falls back to linear nicing of the domain. |
| Ordinal / Quantile / Threshold | Identity (returns `self`). |

## 8. Error policy

### 8.1 Constructor errors (`PyValueError`)

Static violations raise `PyValueError` from the constructor:

- Empty domain (`Linear` / `Log` / `Time` / `Symlog` / `Ordinal`)
- `domain.len() != 2` for continuous scales
- `lo == hi` (degenerate domain) — except `Ordinal` where `len(domain) == 1` is allowed
- `Log` domain containing `0` or values with mixed signs
- `Log.base ≤ 0` or `base == 1`
- `Symlog.constant ≤ 0`
- `Ordinal` duplicate categories
- `Ordinal.padding < 0` or `padding > 1`
- `Threshold` domain not strictly sorted ascending, or `len(range) != len(domain) + 1`
- `Quantile.range` empty or `domain.len() < 2`
- Any non-finite (`NaN`, `±∞`) value in `domain` or `range`

### 8.2 Runtime errors (NaN propagation)

Runtime out-of-domain or out-of-range never panics:

- `scale(NaN)` → `NaN`
- `scale(x)` outside domain when `clamp=False` → `NaN` for continuous scales; clipped-to-edge for `Quantile` / `Threshold` (defined over all of ℝ)
- `invert(NaN)` → `NaN`
- `invert(y)` outside range when `clamp=False` → `NaN`
- `OrdinalScale.scale(unknown_category)` → `NaN`
- `OrdinalScale.invert(y)` outside the band-extent → `None`
- `QuantileScale.invert_extent(y)` / `ThresholdScale.invert_extent(y)` for `y` not in range → `(NaN, NaN)`

### 8.3 Validation helpers

Two private helpers cover most constructor checks:

```rust
fn validate_finite(name: &str, values: &[f64]) -> PyResult<()>;
fn validate_continuous_pair(domain: &[f64], range: &[f64]) -> PyResult<()>;
```

Per-scale validators add variant-specific rules (sortedness, arity, sign coherence).

## 9. Testing strategy

### 9.1 Rust tests (`cargo test -p ferrum-core`)

Per scale, at minimum:

- **Round-trip / inversion** — satisfies the "one inversion test per scale type" done criterion.
  - Continuous: `scale.invert(scale.scale(x)) ≈ x` (within `1e-9` relative tolerance).
  - Bin (`Quantile`, `Threshold`): `invert_extent(scale.scale(x))` brackets `x`.
  - Ordinal: `invert(scale.scale(s)) == Some(s)`.
- **Boundary cases** named in the done criteria:
  - `LogScale::new` with `domain` containing `0` returns `Err`.
  - `SymlogScale.scale(0.0)` returns finite output.
  - `OrdinalScale` padding affects band-extent math (parameterized at `padding=0.0` vs `padding=0.5`).
- **Tick generation**: each scale produces a non-empty tick array for a representative domain. `Quantile` / `Threshold` honor Sturges floor when called with default `count=None`.
- **`nice()` idempotence** for continuous scales: `s.nice().nice() == s.nice()`.

Tests live in `crates/ferrum-core/src/scale/<variant>.rs` `#[cfg(test)] mod tests {}` blocks (matches Phase 3 layout). Cross-cutting tests for `sturges_floor`, `nice_ticks`, and `nice_time_interval_ms` live in `scale/ticks.rs`.

### 9.2 Python tests (`uv run pytest`)

Light coverage; the math is Rust-tested. `tests/test_scales.py` adds:

- Construction smoke test for each of the seven classes.
- One round-trip per group (continuous, ordinal, bin) confirming the Python boundary conveys `f64`s correctly.
- Constructor error cases for the most likely user mistakes (empty domain, mismatched threshold arity, log of zero) — verifies `PyValueError` propagates cleanly.

### 9.3 Test count targets

Baseline at HEAD: 24 Rust + 18 Python.
Phase 4 target: ≈ 50 Rust + ≈ 30 Python.

## 10. Build & verification

Standard project commands (from `CLAUDE.md`):

| Action | Command |
|---|---|
| Build | `unset CONDA_PREFIX && uv run --no-sync maturin develop` |
| Python tests | `uv run pytest` |
| Rust tests | `DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core` |

Verification one-liner (extends the Phase 3 verifier):

```bash
unset CONDA_PREFIX && uv run --no-sync python -c "
from ferrum._core import LinearScale, LogScale, TimeScale, OrdinalScale, QuantileScale, ThresholdScale, SymlogScale
s = LinearScale(domain=[0.0, 10.0], range=[0.0, 1.0])
assert abs(s.scale(5.0) - 0.5) < 1e-12
assert abs(s.invert(0.5) - 5.0) < 1e-12
print('OK')
"
```

## 11. Done criteria (verifiable checklist)

- [ ] All seven `#[pyclass]` types exposed in `ferrum._core` and re-exported from `ferrum.`
- [ ] `cargo test -p ferrum-core` passes (with the `DYLD_LIBRARY_PATH` invocation)
- [ ] `uv run pytest` passes
- [ ] `_core.pyi` covers all seven constructors and methods
- [ ] At least one inversion test per scale type
- [ ] Sturges floor honored by `QuantileScale` / `ThresholdScale` default tick counts
- [ ] Boundary-value tests pass: `Log` (zero), `Symlog` (zero crossing), `Ordinal` (padding)
- [ ] No new external dependencies added to `Cargo.toml` (no `chrono`, no `nalgebra`)
- [ ] `docs/superpowers/ferrum-phases.md` Phase 4 status updated to `done`
- [ ] This spec doc (`docs/superpowers/specs/2026-05-09-scale-engine-design.md`) committed alongside the implementation

## 12. Branching & commit posture

Per `CLAUDE.md`: Phase 4 lands on a feature branch (suggested `feat/scale-engine`), not directly on `main`. The implementation plan from `superpowers:writing-plans` will sequence commits; expect roughly 5–7 atomic commits — one per scale module, one for shared infrastructure (`ticks.rs`, validators), one for Python stubs and tests, one for status updates.

## 13. Open questions for the implementation plan

The following are deliberately deferred to `superpowers:writing-plans` because they are sequencing decisions, not design decisions:

- Order of scale implementation (recommended: `Linear` first as the math template, then `Log` / `Time` / `Symlog` reusing it, then the discrete trio).
- Whether the shared `ticks.rs` lands as its own commit before or after the first scale.
- Test parameterization style (proptest vs hand-written cases — current Phase 3 tests are hand-written; staying consistent is the default).
