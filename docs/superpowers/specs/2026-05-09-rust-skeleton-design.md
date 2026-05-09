# Ferrum Sub-project #1 — Build & Packaging Skeleton

**Date:** 2026-05-09
**Author:** Chris Santiago
**Status:** Design — pending user review
**Parent project:** Ferrum (see `ferrum-spec.md` at repo root)
**Successor:** Sub-project #2 — Python↔Rust data-handoff layer (Arrow IPC)

---

## 1. Purpose & Scope

Ferrum's parent spec describes a statistical visualization library with a Python declaration layer and a Rust computation layer. None of that can be built until the package itself compiles, installs, and exposes a Rust-defined symbol to Python. This sub-project produces exactly that and nothing more.

### In scope

- Install the Rust toolchain (`rustup`, stable channel, minimal profile).
- Switch the Python build backend from `hatchling` to `maturin`.
- Establish a Cargo workspace at the repo root.
- Add a single PyO3 extension crate, `crates/ferrum-core`, that compiles to the importable Python module `ferrum._core`.
- Expose one trivial sanity function (`add(a: int, b: int) -> int`) end-to-end.
- Wire the Python `__init__.py` to re-export from `ferrum._core` and ship a hand-written `.pyi` stub so the symbol type-checks.
- Document the dev loop and verification command.

### Out of scope (deferred to later sub-projects)

- Arrow IPC bridge, any `DataFrame` semantics, any chart-spec types — sub-project #2.
- `clippy` / `rustfmt` / pre-commit hooks — deferred until the codebase is large enough to benefit.
- Cargo unit-test scaffolding — deferred (the skeleton is too thin to test on the Rust side).
- CI wheel build with `maturin-action` or `cibuildwheel` — first follow-up before any real user.
- The future `ferrum-wasm` crate (interactive renderer) and any shared crate (`ferrum-shared`) — only their *placeholder* is reserved via the workspace structure; no code yet.
- Stripping `add()` from the public API. See **§7 Open Questions**.

### Success criterion

After the implementation plan executes, this command succeeds with output `OK`:

```bash
uv run python -c "import ferrum; assert ferrum.add(2, 3) == 5; print('OK')"
```

If it doesn't, the design failed.

---

## 2. Architecture

The skeleton has three concerns, isolated:

1. **Toolchain.** Rust compiler + Cargo, installed via `rustup` so future cross-compile targets and toolchain components (clippy, rustfmt, wasm32 target) are addable without reinstalling. Lives under `~/.cargo`, not the repo.

2. **Build backend.** `maturin` reads `pyproject.toml` (PEP 621 `[project]` + `[tool.maturin]`), drives `cargo build` for the workspace member crate, and produces a wheel that bundles the resulting cdylib at the configured submodule path. `uv` invokes maturin transparently via PEP 517.

3. **Source layout.** A Cargo workspace at the repo root with one member crate today (`crates/ferrum-core`). The Python package source remains at `src/ferrum/`. The compiled `cdylib` is dropped at `src/ferrum/_core.<abi>.so` during `maturin develop`, or bundled into the wheel under `ferrum/_core.<abi>.so` during `maturin build`.

### Why a workspace from day one

The parent spec commits to a second crate, `ferrum-wasm` (interactive renderer), and shared math (scales, ticks, color schemes) will need to render identically static and interactive — i.e., a third crate (`ferrum-shared`) is essentially guaranteed. Promoting to a workspace later is a known mechanical refactor (~30 min) but touches every Rust file. The cost of starting in a workspace is one extra root `Cargo.toml`, one `crates/` directory, and one `manifest-path` line in `[tool.maturin]`. We pay the small fixed cost now.

### Final tree after this sub-project lands

```
ferrum/
├── Cargo.toml                    # workspace manifest (NEW)
├── Cargo.lock                    # committed (NEW)
├── pyproject.toml                # build-backend = maturin (CHANGED)
├── README.md                     # (unchanged)
├── .python-version               # (unchanged)
├── .gitignore                    # adds target/, *.so, dist/, .venv/ (CHANGED or NEW)
├── ferrum-spec.md                # (unchanged)
├── docs/
│   └── superpowers/specs/
│       └── 2026-05-09-rust-skeleton-design.md   # this file
├── crates/
│   └── ferrum-core/              # NEW
│       ├── Cargo.toml
│       └── src/
│           └── lib.rs
└── src/
    └── ferrum/
        ├── __init__.py           # re-exports from ._core (CHANGED)
        ├── _core.pyi             # type stub (NEW)
        └── py.typed              # (unchanged)
```

---

## 3. The Four-Place Naming Invariant

Three Rust/maturin settings must spell `_core` and one must spell `ferrum._core`. Mismatching them is the single most common first-time-maturin failure mode. They are not independent; the design treats them as one decision.

| Place | Value | Reason |
|---|---|---|
| `crates/ferrum-core/Cargo.toml` `[package].name` | `ferrum-core` | Crate name. Hyphen is conventional for crate names; this name does not propagate to Python. |
| `crates/ferrum-core/Cargo.toml` `[lib].name` | `_core` | Determines the cdylib filename: `_core.<abi>.so`. |
| `crates/ferrum-core/src/lib.rs` `#[pymodule] fn <name>` | `_core` | PyO3 generates `PyInit__core`; must match `[lib].name`. |
| `pyproject.toml` `[tool.maturin].module-name` | `ferrum._core` | Tells maturin the dotted path inside the wheel where the cdylib lands. |

If `add()` ever moves out of `ferrum/__init__.py`'s re-export list, the user-facing import path becomes `from ferrum._core import add` — the underscore makes the submodule conventionally private, signalling "implementation detail; import via the package."

---

## 4. File Contents (verbatim)

These are the files the implementation plan will write. Verbatim contents are included so the plan does not re-derive them.

### 4.1 `pyproject.toml` (full replacement)

```toml
[project]
name = "ferrum"
version = "0.1.0"
description = "A grammar-of-graphics statistical visualization library with a Rust core"
readme = "README.md"
authors = [
    { name = "chris-santiago", email = "cjsantiago@gatech.edu" }
]
requires-python = ">=3.10"
dependencies = []

[build-system]
requires = ["maturin>=1.7,<2.0"]
build-backend = "maturin"

[tool.maturin]
module-name = "ferrum._core"
manifest-path = "crates/ferrum-core/Cargo.toml"
python-source = "src"
features = ["extension-module"]
strip = true

[dependency-groups]
dev = ["maturin>=1.7,<2.0", "pytest>=8"]
```

### 4.2 Root `Cargo.toml` (workspace manifest, NEW)

```toml
[workspace]
resolver = "2"
members = ["crates/ferrum-core"]

[workspace.package]
version = "0.1.0"
edition = "2021"
license = "MIT"
authors = ["Chris Santiago <cjsantiago@gatech.edu>"]
repository = "https://github.com/chris-santiago/ferrum"

[workspace.dependencies]
pyo3 = { version = "0.22", features = ["abi3-py310"] }

[profile.release]
lto = "thin"
codegen-units = 1
```

**Notes for the implementation plan:**

- `resolver = "2"` is the modern feature resolver. Workspaces require it explicitly; without it Cargo emits a warning.
- `pyo3 = "0.22"` is the version target as of 2026-05; the implementation plan must verify this is still current at execution time and adjust if PyO3 has released a new minor with API changes (the `Bound<'_, PyModule>` API in `lib.rs` is from PyO3 0.21+).
- `lto = "thin"` and `codegen-units = 1` make release builds slower but produce ~5–15% faster numeric code. Set once, applies to every future crate in the workspace.

### 4.3 `crates/ferrum-core/Cargo.toml` (NEW)

```toml
[package]
name = "ferrum-core"
version.workspace = true
edition.workspace = true
license.workspace = true
authors.workspace = true
repository.workspace = true

[lib]
name = "_core"
crate-type = ["cdylib"]

[features]
extension-module = ["pyo3/extension-module"]

[dependencies]
pyo3 = { workspace = true }
```

**Why `extension-module` is feature-gated, not unconditional:**

`pyo3/extension-module` tells PyO3 to skip linking against `libpython`, which is what makes the resulting `.so` loadable by any Python interpreter. Correct for `maturin build` and `maturin develop` — wrong for `cargo test`, which builds a standalone test binary that *needs* libpython linked (without it the test binary fails to link with `undefined reference to _PyExc_*`).

The fix is the maturin-recommended feature-gate pattern: declare an optional crate feature `extension-module` that re-exports `pyo3/extension-module`, leave `pyo3` in `[dependencies]` without that feature, and have `[tool.maturin].features` enable the gate at wheel-build time only. Result: `maturin develop` works (feature on), `cargo test` works (feature off). The skeleton has no Rust tests yet, but sub-project #2 (Arrow IPC) will absolutely want `cargo test` for the data-handoff logic — paying this one-line cost now avoids retrofitting later.

**Why `abi3-py310` lives at the workspace level:**

`abi3-py310` controls the *ABI target* and applies to every consumer that links PyO3. Declared once in `[workspace.dependencies]`, every member crate inherits it consistently. A future pure-math crate (`ferrum-shared`) that depends on `pyo3` for type conversions would also inherit `abi3-py310` automatically without needing the `extension-module` gate.

### 4.4 `crates/ferrum-core/src/lib.rs` (NEW)

```rust
use pyo3::prelude::*;

/// Sanity check that the Rust↔Python bridge works. Remove once real bindings exist.
#[pyfunction]
fn add(a: i64, b: i64) -> i64 {
    a + b
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    Ok(())
}
```

### 4.5 `src/ferrum/__init__.py` (replacement, was 52 bytes of placeholder)

```python
"""Ferrum — a statistical visualization library with a Rust core."""

from ferrum._core import add

__all__ = ["add"]
__version__ = "0.1.0"
```

### 4.6 `src/ferrum/_core.pyi` (NEW)

```python
def add(a: int, b: int) -> int: ...
```

Hand-written for now. When the Rust API surface exceeds ~10 functions we switch to `pyo3-stub-gen` autogeneration.

### 4.7 `.gitignore` (additions; create file if missing)

```
target/
*.so
*.pyd
*.dylib
dist/
.venv/
__pycache__/
*.egg-info/
```

The `*.so` line is intentional: maturin drops the compiled cdylib into `src/ferrum/` during `develop`, but it is build output, not source. Wheel builds re-emit it.

---

## 5. Toolchain Install & Dev Loop

### 5.1 One-time install (executed by Claude with confirmation)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
. "$HOME/.cargo/env"
```

`--profile minimal` skips local docs and rust-src (~600 MB). `clippy` and `rustfmt` are added later when the repo benefits from lint/format gating.

### 5.2 Project-level setup

The dev dependencies are already declared in `pyproject.toml` (§4.1). Resolve and install:

```bash
uv sync
```

Writes `uv.lock` and creates `.venv/` with `maturin` and `pytest` available. (`uv sync` includes the `dev` group by default in `uv >= 0.4`.)

### 5.3 Build + install editable

```bash
uv run maturin develop
```

Compiles the cdylib (debug profile by default), drops `_core.<abi>.so` into `src/ferrum/`, links the project as editable. ~1–3 s incrementally after the first build.

### 5.4 Iteration rules

- Edits to `.rs` files: re-run `uv run maturin develop`.
- Edits to `.py` files: live, no rebuild needed (Python source is symlinked).
- For perf measurement: `uv run maturin develop --release`.

---

## 6. Verification, Tests, Failure Modes

### 6.1 Acceptance gate (the success criterion from §1)

```bash
uv run python -c "import ferrum; assert ferrum.add(2, 3) == 5; print('OK')"
```

Output must be exactly `OK`. This single assertion verifies six independent properties: (a) the wheel built, (b) the cdylib was placed where Python can find it, (c) the four-place naming invariant holds, (d) `__init__.py` re-exported the symbol, (e) the `i64 ↔ Python int` conversion works, (f) the Python interpreter can load an abi3 module compiled for ≥3.10.

### 6.2 Pytest smoke test

`tests/test_smoke.py`:

```python
def test_core_add():
    from ferrum import add
    assert add(2, 3) == 5
```

Run with `uv run pytest`. One test is sufficient — the skeleton's job is to validate the toolchain, not features.

### 6.3 Failure-mode lookup table

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: dynamic module does not define module export function (PyInit__core)` | `[lib].name` and `#[pymodule] fn <name>` disagree. | Make both `_core`. |
| `ModuleNotFoundError: No module named 'ferrum._core'` | `[tool.maturin].module-name` wrong, or `python-source` missing/incorrect. | `module-name = "ferrum._core"`, `python-source = "src"`. |
| `error: linking with 'cc' failed`, references to `_PyExc_*` | `pyo3` lacks `extension-module` feature in the crate. | Add `features = ["extension-module"]` to the crate's `pyo3` line. |
| `error: failed to select a version for ...` (resolver) | `resolver = "2"` missing from workspace manifest. | Add it under `[workspace]`. |
| `ImportError: cannot import name 'add' from 'ferrum'` | `__init__.py` still the 52-byte placeholder. | Apply §4.5 contents. |

---

## 7. Open Questions (for the implementation plan, not blocking)

1. **Strip `add()` before merge, or leave it as a smoke target until sub-project #2 lands?**
   *For leaving:* gives sub-project #2 an existing regression to keep passing while it wires Arrow IPC. *For stripping:* avoids a public function with no real purpose. Recommend leave; remove in sub-project #2's PR.

2. **PyO3 version pin.** This design targets `pyo3 = "0.22"`. The implementation plan must reverify against current crates.io at execution time and bump if a newer minor is out, adjusting the `Bound<'_, PyModule>` API in §4.4 if needed.

3. **CI wheel matrix.** Out of scope here. First follow-up sub-project before any real users — `maturin-action` is the standard fit for a project this size.

---

## 8. Decisions Locked In (audit trail)

| Decision | Choice | Alternatives considered |
|---|---|---|
| Toolchain installer | `rustup`, run by Claude with confirmation | manual install, Homebrew |
| Build backend | `maturin` | `setuptools-rust`, `hatch-rust` |
| Skeleton scope | trivial sanity function only | + dev tooling, + CI wheels |
| Crate layout | Cargo workspace, single member today | single crate at root, `rust/` subdir |
| ABI target | `abi3-py310` (one wheel per platform, all Pythons ≥3.10) | per-version wheels |
| `extension-module` activation | feature-gated via `[features]` table; enabled by maturin only | unconditional on crate (breaks `cargo test`); push tests to a separate crate |
| Branch / worktree | direct to `main` (greenfield, user-confirmed override of global rule) | worktree, in-place feature branch |

---

## 9. Hand-off

Once the user approves this design, control passes to `superpowers:writing-plans` to produce the implementation plan. The plan will sequence the install/setup/file-write/verify steps from §5–§6, resolve the open questions in §7 that block execution, and check the PyO3 version per §7.2 before pinning.
