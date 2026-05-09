# Ferrum — Implementation Phases (Meta-Roadmap)

**Last updated:** 2026-05-09
**Concept spec:** `ferrum-spec.md` (at repo root — philosophy, API surface, design constraints)
**This document:** implementation order, phase dependencies, done criteria, and session-orientation guide

---

## How to orient yourself in a new session

1. Read this file first.
2. Find the first phase whose **Status** is not `done`.
3. If that phase has a spec doc linked in the **Spec** column, read it.
4. If no spec doc exists yet, run the brainstorming skill (`superpowers:brainstorming`) for that phase before writing any code.
5. If a spec exists but no implementation plan exists, run `superpowers:writing-plans`.
6. If a plan exists, run `superpowers:executing-plans`.

Never start coding without a spec and plan for that phase. Never re-litigate a decision recorded in a spec without a stated reason.

---

## Locked architectural decisions (do not re-litigate)

These were settled in the brainstorming session on 2026-05-09. They affect every phase.

| Decision | Choice | Rationale |
|---|---|---|
| Rust toolchain | `rustup`, stable, minimal profile | Supports multi-toolchain, cross-compile targets, component installs |
| Build backend | `maturin >= 1.7` | De facto standard for PyO3; works with `uv` via PEP 517 |
| Crate layout | Cargo workspace from day one | `ferrum-core` (PyO3 extension) + room for `ferrum-wasm`, `ferrum-shared` |
| Python source | `src/ferrum/` (src layout) | `pyproject.toml` `python-source = "src"` |
| Compiled module | `ferrum._core` (underscore = impl detail) | `[tool.maturin] module-name = "ferrum._core"` |
| ABI target | `abi3-py310` | One wheel per platform-arch for all Python ≥ 3.10 |
| `extension-module` | Feature-gated (`[features] extension-module = ["pyo3/extension-module"]`) | Allows `cargo test` to link libpython; maturin enables the gate at build time |
| Data transport | Arrow C Data Interface via `pyo3-arrow` (phase 2) | Zero row-level Python access after initial handoff; CDI chosen over IPC bytes for zero-copy polars support (polars implements `__arrow_c_stream__` natively); spec §"Zero unnecessary copies" |
| ChartSpec serialization | JSON via `serde` + `serde_json` (phase 3) | Tree-structured config; matches public `chart.to_json()` API; schema-evolves cleanly as phases 4–10 add fields; human-readable for debugging and Vega-Lite interop. Arrow schema rejected as a category mismatch (describes columns, not config trees); binary codec rejected (size class makes perf gains irrelevant; loses readability and interop) |
| Release profile | `lto = "thin"`, `codegen-units = 1` | Set once in workspace root; all future crates inherit |
| Python version | `requires-python = ">=3.10"` | `.python-version` = 3.10 |

---

## Phase table

### Dependency key
An arrow `→` means "must be done before." Phases with no arrow have no predecessors.

```
1 → 2 → 3 → 4
              → 5
         3 → 6 → 7 → 8 → 9
                           → 10
                      8 → 11
              3 → 12 (cross-cutting, unlocks after 8)
```

### Phases

| # | Name | What it produces | Depends on | Spec doc | Status |
|---|---|---|---|---|---|
| **1** | Build & packaging skeleton | Cargo workspace + maturin backend + `ferrum._core.add()` compiles and imports | — | [`2026-05-09-rust-skeleton-design.md`](specs/2026-05-09-rust-skeleton-design.md) | **done** |
| **2** | Python↔Rust data-handoff layer | Arrow CDI bridge (pyo3-arrow): DataFrame → RecordBatch in → transformed RecordBatch out via C Data Interface; no row-level Python after handoff | 1 | [`2026-05-09-arrow-ipc-design.md`](specs/2026-05-09-arrow-ipc-design.md) | **done** |
| **3** | Chart spec IR + serialization | Internal Rust representation of a `Chart`; Python builds it, Rust consumes it; round-trip tests | 2 | *(not yet written)* | pending |
| **4** | Scale engine | `LinearScale`, `LogScale`, `TimeScale`, `OrdinalScale`, `QuantileScale`, `ThresholdScale`, `SymlogScale`; domain/range mapping, tick generation | 3 | *(not yet written)* | pending |
| **5** | Stat engine | KDE, bootstrap CI, linear/LOESS regression, binning (Sturges floor), aggregation — all as Rust stat transforms declared in the chart spec | 3 | *(not yet written)* | pending |
| **6** | Layout engine | Constraint solver for facet sizes, legend placement, axis label collision avoidance | 3 | *(not yet written)* | pending |
| **7** | Static renderer (SVG/PNG) | First end-to-end chart output: a scatter plot from spec → SVG file. Primitive marks only (point, line, bar, area, rect, rule, text, tick) | 4, 5, 6 | *(not yet written)* | pending |
| **8** | Grammar API surface (Python) | `Chart`, `Layer`, encoding channels (`X`, `Y`, `Color`, `Size`, etc.), `+`/`\|`/`&` composition operators, `Facet`, `Repeat`, themes-as-values | 7 | *(not yet written)* | pending |
| **9** | Convenience / figure-level API | `displot`, `lmplot`, `roc_chart`, `pairplot`, etc. as sugar over the grammar — they must desugar to valid `Chart` specs, not bypass the engine | 8 | *(not yet written)* | pending |
| **10** | Model diagnostics layer | `ModelSource` (sklearn-protocol adapter), model-diagnostic marks (`ConfusionMark`, `ROCMark`, `CalibrationMark`, etc.), `Visualizer` convenience wrappers | 8 | *(not yet written)* | pending |
| **11** | Interactive renderer (WASM) | `ferrum-wasm` crate + `ferrum._wasm` module; `.interactive()` switches render target; selections, zoom, pan, linked views declared in chart spec | 8 | *(not yet written)* | pending |
| **12** | Extension points | Public APIs for custom marks, custom stat transforms, custom themes, renderer plugins; stable enough to document and not break | 8 | *(not yet written)* | pending |

---

## Done criteria (per phase)

A phase is `done` when all of the following are true:

### Phase 1 — Build & packaging skeleton
- [x] `rustup` installed, `cargo --version` works in the project shell
- [x] `uv run python -c "import ferrum; assert ferrum.add(2, 3) == 5; print('OK')"` outputs `OK`
- [x] `uv run pytest` passes the smoke test in `tests/test_smoke.py`
- [x] `Cargo.toml` (workspace), `crates/ferrum-core/Cargo.toml`, `crates/ferrum-core/src/lib.rs`, `src/ferrum/__init__.py`, `src/ferrum/_core.pyi` all committed to `main`

### Phase 2 — Data-handoff layer
- [x] A polars DataFrame and a pyarrow RecordBatch each cross the PyO3 boundary via the Arrow C Data Interface (pyo3-arrow crate)
- [x] Rust receives a `RecordBatch`, applies a trivial transform (column rename), returns a `RecordBatch` via CDI
- [x] Python receives the result with zero row-level access in between
- [x] `cargo test` passes in `crates/ferrum-core` (tests the Arrow round-trip on the Rust side)

### Phase 3 — Chart spec IR
- [ ] A `ChartSpec` Rust struct exists with enough fields to represent a single-layer scatter plot (data ref, x/y encoding, mark type)
- [ ] Python can construct it via a `ferrum._core.ChartSpec` binding and pass it to Rust
- [ ] Rust can round-trip serialize/deserialize `ChartSpec` to/from JSON via `serde_json` (decision 2026-05-09 — see locked-decisions table)
- [ ] `cargo test` covers at least one round-trip case

### Phase 4 — Scale engine
- [ ] All seven scale types are implemented in Rust and exposed via `ferrum._core`
- [ ] Domain/range mapping is correct for boundary values (including log(0), symlog threshold, ordinal padding)
- [ ] Tick generation passes the spec's "Sturges floor" requirement for binning
- [ ] Python-facing type stubs in `_core.pyi` cover all scale constructors
- [ ] `cargo test` covers at least one inversion test per scale type

### Phase 5 — Stat engine
- [ ] KDE, bootstrap CI, linear regression, LOESS, binning, and basic aggregation implemented in Rust
- [ ] Each transform declared in a `ChartSpec` and executed by the engine before layout
- [ ] `cargo test` covers numeric correctness against a reference (scipy/numpy values computed offline)

### Phase 6 — Layout engine
- [ ] Facet grid sizes computed correctly for `wrap` and `grid` facet modes
- [ ] Legend placement does not overlap chart area for a 1-layer scatter plot
- [ ] Axis label collision avoidance (rotation or elision) fires at a configurable threshold
- [ ] `cargo test` covers basic facet layout arithmetic

### Phase 7 — Static renderer
- [ ] A scatter plot from a spec file renders to a valid SVG file
- [ ] All eight primitive marks render without panics on a minimal spec
- [ ] PNG output works (resvg or equivalent)
- [ ] Output includes correct scale ticks, axis labels, and a legend

### Phase 8 — Grammar API surface
- [ ] `import ferrum; ferrum.Chart(data).mark_point().encode(x="col_a", y="col_b").show()` works
- [ ] Layer composition (`+`), hstack (`|`), vstack (`&`) work
- [ ] `Theme` objects are values passed to `Chart`, not global state
- [ ] No `matplotlib` in the dependency tree (`pip show matplotlib` returns nothing)
- [ ] All encoding channels from `ferrum-spec.md §3.2` are implemented

### Phase 9 — Convenience API
- [ ] Each figure-level function in `ferrum-spec.md §3.14` is implemented
- [ ] Each one can be deconstructed: calling the function and inspecting `.spec` yields a valid `ChartSpec`

### Phase 10 — Model diagnostics
- [ ] `ModelSource` wraps any object with `predict`/`predict_proba`/`transform`
- [ ] All model-diagnostic marks from `ferrum-spec.md §3.3` render correctly
- [ ] Sklearn is not imported unless the user's model is from sklearn

### Phase 11 — Interactive renderer
- [ ] `chart.interactive()` produces an HTML bundle with a WASM renderer
- [ ] Selections, zoom, and pan declared in the chart spec work in a browser
- [ ] `ferrum-wasm` crate added to workspace; wheel build still works for the Python package

### Phase 12 — Extension points
- [ ] Custom mark, stat transform, theme, and renderer plugin protocols are documented
- [ ] At least one of each is implemented as an example in `examples/`
- [ ] Adding a custom mark does not require modifying `ferrum-core`

---

## Session workflow

Each sub-project follows this exact cycle. Do not skip steps.

```
New session starts
    ↓
Read ferrum-phases.md                ← orientation, find current phase
    ↓
Read phase's spec doc (if exists)    ← design decisions, file contents
    ↓
(If no spec) → brainstorming skill   → write spec doc → commit
    ↓
Read implementation plan (if exists) ← execution steps
    ↓
(If no plan) → writing-plans skill   → write plan doc → commit
    ↓
executing-plans skill                ← implement, verify, commit
    ↓
Mark phase done in this file         ← update Status column, check done criteria
    ↓
/clear or next session
```

---

## Spec and plan doc naming convention

```
docs/superpowers/specs/YYYY-MM-DD-<phase-slug>-design.md    ← brainstorming output
docs/superpowers/plans/YYYY-MM-DD-<phase-slug>-plan.md      ← writing-plans output
```

Phase slugs:
- `rust-skeleton` (phase 1)
- `arrow-ipc` (phase 2)
- `chart-spec-ir` (phase 3)
- `scale-engine` (phase 4)
- `stat-engine` (phase 5)
- `layout-engine` (phase 6)
- `static-renderer` (phase 7)
- `grammar-api` (phase 8)
- `convenience-api` (phase 9)
- `model-diagnostics` (phase 10)
- `interactive-renderer` (phase 11)
- `extension-points` (phase 12)

---

## Notes for future sessions

- **Do not install matplotlib.** Ever. Not as a dev dependency, not for testing. This is a hard constraint from `ferrum-spec.md §2`.
- **`uv run maturin develop`** is the dev build command. Plain `pip install -e .` will not compile the Rust extension.
- **`cargo test` must pass** before any phase is marked done (phases 2+). Phase 1 has no Rust-side tests — that is the only exception.
- **The PyO3 version pin** in root `Cargo.toml` must be checked at the start of each Rust-touching session. PyO3 publishes minor releases; the `Bound<'_, PyModule>` API is ≥ 0.21.
- **`ferrum-spec.md`** is the API contract. If implementation diverges from it, update the spec with a dated note — never silently drift.
- **Themes are values, not global state.** No `set_theme()`, no module-level `rcParams` equivalent. This constraint applies to every phase that touches rendering.
