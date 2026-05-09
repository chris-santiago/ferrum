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
- **No global mutable state.** No `set_theme()`, no module-level config objects. Themes are values passed to `Chart`.
- **`ferrum-spec.md` is the API contract.** If implementation diverges, update the spec with a dated note. Never silently drift.
- **`cargo test` must pass** before any phase (2+) is marked done. Phase 1 is the only exception.
- **Do not `git push`** unless the user explicitly asks.
- **Confirm before committing to `main`** on non-trivial work. Phase 1 commits directly to main by user decision (greenfield); subsequent phases use feature branches unless the user says otherwise.

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

## Key architectural decisions (settled 2026-05-09)

- **Build backend:** `maturin >= 1.7`
- **Workspace:** Cargo workspace at repo root; `ferrum-core` is the only member today; `ferrum-wasm` and `ferrum-shared` are reserved for phases 11 and beyond
- **ABI:** `abi3-py310` — one wheel per platform-arch, works for Python ≥ 3.10
- **`extension-module` feature:** feature-gated (not unconditional) so `cargo test` can link libpython
- **PyO3 version:** pinned in `[workspace.dependencies]`; re-verify against crates.io at the start of any session that adds a new PyO3 API
- **Data transport:** Arrow C Data Interface via `pyo3-arrow` crate (phase 2+); NOT Arrow IPC bytes. Polars DataFrames implement `__arrow_c_stream__` natively — CDI hands off the buffer pointer directly with zero copies. Before phase 2, no DataFrames cross the boundary. (Decision made 2026-05-09: spec originally said "Arrow IPC" but CDI was chosen for zero-copy polars support.)
- **Chart spec serialization:** JSON via `serde` + `serde_json` (phase 3+); NOT Arrow schema metadata, NOT a binary codec. The `ChartSpec` IR is a tree-structured config (mark, encodings, scales, transforms, layers) — JSON matches the public `chart.to_json()` API in `ferrum-spec.md §3.1` and `§3.16`, evolves cleanly across phases as new optional fields are added, and stays human-readable for debugging and test fixtures. Spec size is small (KB), so binary-codec performance gains are irrelevant. Vega-Lite interop (phase 7+ `engine="vega-lite"` output) stays open without translation. (Decision made 2026-05-09: phases doc said "JSON or Arrow schema" — Arrow schema rejected because it describes columns, not config trees, and would require flattening to dotted keys or embedding JSON inside metadata.)
