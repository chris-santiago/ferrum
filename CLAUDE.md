# Ferrum — Project Instructions

Ferrum is a Rust-backed Python statistical visualization library. The Python layer is the declaration API; the Rust layer (`crates/ferrum-core`, compiled to `ferrum._core`) is the computation engine. Data moves between them once, over Arrow IPC.

---

## Start every session here

1. **Read `docs/superpowers/ferrum-phases.md`** — it lists all 12 implementation phases, their dependency order, done criteria, and the current status of each. Find the first phase that is not `done` and start there.

2. **Read the phase's spec doc** (linked in the phases table) before writing any code. If no spec exists yet, run `superpowers:brainstorming` before touching anything.

3. **Read `ferrum-spec.md`** (repo root) only if you need the user-facing API contract for the phase you are working on. It is the concept spec, not the implementation guide.

---

## Build commands

| Action | Command |
|---|---|
| Install / rebuild Rust extension | `unset CONDA_PREFIX && uv run --no-sync maturin develop` |
| Release build | `unset CONDA_PREFIX && uv run --no-sync maturin develop --release` |
| Run tests | `uv run pytest` |
| Rust-side tests | `DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test` |
| Verify skeleton | `unset CONDA_PREFIX && uv run --no-sync python -c "from ferrum._core import ChartSpec; s=ChartSpec(mark='point', x='a', y='b'); assert s == ChartSpec.from_json(s.to_json()); print('OK')"` |

> **Note:** `--no-sync` is required for `maturin` commands to avoid a conflict between
> conda's `CONDA_PREFIX` and uv's `VIRTUAL_ENV`. Miniforge base sets `CONDA_PREFIX` even
> outside conda envs, which maturin rejects when uv also sets `VIRTUAL_ENV`. The
> `unset CONDA_PREFIX` prefix clears it for that shell invocation only. Source
> `~/.cargo/env` first if `cargo` is not on your PATH (`source ~/.cargo/env`).

> **macOS `cargo test` note:** On macOS with uv-managed Python, the test binary cannot
> resolve `@rpath/libpython3.10.dylib` at runtime without `DYLD_LIBRARY_PATH` pointing to
> the Python lib directory. The command above uses `sysconfig` to find the path dynamically.
> This is a macOS SIP + uv RPATH constraint; it does not affect `maturin develop` or pytest.

`pip install -e .` will **not** compile the Rust extension. Always use `maturin develop`.

---

## Hard constraints (never violate)

- **No matplotlib.** Not as a dependency, not as a dev dependency, not as an optional extra. Ever.
- **No global mutable state.** No module-level config objects, no module-level theme rebinding. Themes are values passed to `Chart`; per-chart `.theme()` always wins. The single documented exception is `ferrum.set_default_theme()` (phase 8a+), which mutates a per-thread `contextvars.ContextVar` — scope-bounded, automatic-revert when used as a context manager, and overridden by per-chart `.theme()` at render time. Do not introduce other process-scoped mutators.
- **`ferrum-spec.md` is the API contract.** If implementation diverges, update the spec with a dated note. Never silently drift.
- **`cargo test` must pass** before any phase (2+) is marked done. Phase 1 is the only exception.
- **Do not `git push`** unless the user explicitly asks.
- **Confirm before committing to `main`** on non-trivial work. Phase 1 commits directly to main by user decision (greenfield); subsequent phases use feature branches unless the user says otherwise.

---

## Implementation philosophy (Phase 9 and beyond)

**Do the work now. Do it the right way. Enable a better end-user experience now.**

- Do NOT propose "defer X to a later phase / follow-up ticket" as a scope-reduction strategy.
- If a `ferrum-spec.md` parameter is hard to ship completely, ship it completely (with whatever Rust transform, mark, encoding, or position-adjustment subsystem it needs) — not a warn-fallback or `NotImplementedError`.
- Earlier phases accumulated deferred work that landed on Phase 9; further deferral compounds the problem and ships a worse end-user experience than the spec promises.
- Use sub-phase decomposition (e.g. 9a / 9b / 9c / 9d) to manage build order, not to drop scope.
- "Implement everything fully" is the default. Warn-fallbacks are not the path.
- Review `PHASE_9_PLUS_MARKS` in `src/ferrum/marks/deferred.py` at each new phase — pull marks into scope if they're needed by `ferrum-spec.md` §3.14 figure-level functions or any other in-phase contract.

This rule governs Phase 9 forward; it does not retroactively reopen closed phases.

---

## Where things live

| Artifact | Path |
|---|---|
| Implementation phases & roadmap | `docs/superpowers/ferrum-phases.md` |
| Concept + API specification | `ferrum-spec.md` |
| Per-phase design specs | `docs/superpowers/specs/YYYY-MM-DD-<slug>-design.md` |
| Per-phase implementation plans | `docs/superpowers/plans/YYYY-MM-DD-<slug>-plan.md` |
| Python package source | `src/ferrum/` |
| Rust extension crate | `crates/ferrum-core/` |
| Python tests | `tests/` |

---

## Key architectural decisions

- **Build backend:** `maturin >= 1.7`
- **Workspace:** Cargo workspace at repo root; `ferrum-core` is the only member today; `ferrum-wasm` and `ferrum-shared` are reserved for phases 11 and beyond
- **ABI:** `abi3-py310` — one wheel per platform-arch, works for Python ≥ 3.10
- **`extension-module` feature:** feature-gated (not unconditional) so `cargo test` can link libpython
- **PyO3 version:** pinned in `[workspace.dependencies]`; re-verify against crates.io at the start of any session that adds a new PyO3 API
- **Data transport:** Arrow C Data Interface via `pyo3-arrow` crate (phase 2+); NOT Arrow IPC bytes. Polars DataFrames implement `__arrow_c_stream__` natively — CDI hands off the buffer pointer directly with zero copies. Before phase 2, no DataFrames cross the boundary. (Decision made 2026-05-09: spec originally said "Arrow IPC" but CDI was chosen for zero-copy polars support.)
- **Chart spec serialization:** JSON via `serde` + `serde_json` (phase 3+); NOT Arrow schema metadata, NOT a binary codec. The `ChartSpec` IR is a tree-structured config (mark, encodings, scales, transforms, layers) — JSON matches the public `chart.to_json()` API in `ferrum-spec.md §3.1` and `§3.16`, evolves cleanly across phases as new optional fields are added, and stays human-readable for debugging and test fixtures. Spec size is small (KB), so binary-codec performance gains are irrelevant. Vega-Lite interop (phase 7+ `engine="vega-lite"` output) stays open without translation. (Decision made 2026-05-09: phases doc said "JSON or Arrow schema" — Arrow schema rejected because it describes columns, not config trees, and would require flattening to dotted keys or embedding JSON inside metadata.)
- **DataFrame compatibility layer:** `narwhals` (~1.x) added (phase 8a+) for non-polars DataFrame inputs (pandas, modin, cuDF, dask, ibis). Direct CDI path preserved for `polars.DataFrame` and pyarrow `Table`/`RecordBatch`; everything else flows through `narwhals.from_native(data, eager_only=True).to_arrow()`. Dict-of-arrays, list-of-records, and 2D numpy handled by direct `pyarrow.Table.from_*` branches in `src/ferrum/_coerce.py`. (Decision made 2026-05-10: alternative was ~250 LOC of in-house pandas dtype normalization; narwhals owns those bugs, ships modin/cuDF/dask/ibis support for free, and is the same compatibility layer altair adopted in 2024 for an identical problem.)
- **Multi-layer `ChartSpec`:** `layers: Option<Vec<Layer>>` additive field on `ChartSpec` (phase 8a+). When `layers.is_none()`, the renderer uses single-layer `mark` + `encoding` and the JSON shape is byte-identical to phases 3–7 — existing goldens stay valid. When `layers.is_some()`, the renderer iterates layers within each panel, sharing x/y/color scales by default. **One `ChartSpec` = one `RecordBatch`** is load-bearing: mixed-data layered charts (`Chart(df1) + Chart(df2)`) route through the SVG compositor instead of growing multi-batch logic in the renderer. (Decision made 2026-05-10.)
- **Themes are values; one documented contextvar exception.** `Theme` is an immutable Python value class (phase 8a+). `Chart.theme(t)` per-chart override always wins. `ferrum.set_default_theme(t)` returns a context manager backed by a per-thread `contextvars.ContextVar` for notebook ergonomics — the only sanctioned process-scoped theme state. See the **Hard constraints** section above for the full statement. (Decision made 2026-05-10.)
- **Composite marks desugar Python-side; no Rust `Composite` Mark variant.** Composite marks (`mark_boxplot`, `mark_errorbar`, `mark_errorband`, `mark_ribbon`, `mark_violin`, `mark_boxen`, …) build multi-layer `ChartSpec`s in Python via `chart.layer(...)` over primitive marks (rect, rule, point, line, ribbon). Rust has no awareness that a layered spec came from a composite — it just renders layers. Multi-output transforms (`BoxStats`, `Outliers`, `Violin`, `Hex`, …) feed individual layers via `Layer.data_source: Option<String>` matched against `TransformSpec.name: Option<String>`; when both are `None`, behavior is byte-identical to single-layer 8a. (Decision made 2026-05-10: alternative was a Rust `MarkSpec::Composite { layers: Vec<Mark> }` variant — rejected because it duplicates the multi-layer machinery already in `ChartSpec.layers`, and would force every composite-mark expansion to cross the PyO3 boundary as opaque payloads. Same pattern applies to Phase 9's `mark_boxen` and any future composite.)
- **Byte-deterministic randomness via seeded `rand_chacha`.** Every transform that uses randomness — bootstrap CI (`Smooth`, `Aggregate`), beeswarm tiebreak, Phase 9 `Jitter` — seeds `ChaCha8Rng` from a `u64` (transform's `seed` field or `spec.seed`, default `0`). Never `rand::thread_rng()`, never `SystemRandom`, never platform RNG. This makes SVG goldens byte-identical across macOS / Linux / CI and across Rust toolchain versions. The same rule applies to any future transform or mark that introduces randomness — pick a seed field, document the default, plumb it through `ChaCha8Rng`. (Decision made 2026-05-10: existing 8b transforms ship this way; codified here so Phase 9+ doesn't accidentally reintroduce non-determinism.)
