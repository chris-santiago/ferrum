# Rust/Maturin Build Skeleton — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Install the Rust toolchain, switch the Python build backend to maturin, wire a Cargo workspace with one PyO3 extension crate (`ferrum._core`), and verify that `import ferrum; ferrum.add(2, 3)` returns `5`.

**Architecture:** Cargo workspace at repo root (`Cargo.toml`) contains one member crate, `crates/ferrum-core`, compiled to a cdylib (`_core.so`). maturin builds that cdylib and places it at `ferrum._core` inside the Python package. Python's `src/ferrum/__init__.py` re-exports `add` from `ferrum._core`.

**Tech Stack:** Rust (stable, via rustup), PyO3 0.22+, maturin ≥ 1.7, uv, pytest ≥ 8

---

## File map

| File | Action | Responsibility |
|---|---|---|
| `tests/test_smoke.py` | Create | Pytest smoke test — one assertion that `ferrum.add(2, 3) == 5` |
| `.gitignore` | Create | Exclude `target/`, `*.so`, `*.pyd`, `*.dylib`, `dist/`, `.venv/`, `__pycache__/`, `*.egg-info/` |
| `Cargo.toml` | Create | Workspace manifest — declares `ferrum-core` as the only member; pins PyO3 with `abi3-py310`; sets release profile |
| `crates/ferrum-core/Cargo.toml` | Create | PyO3 cdylib crate — `[lib].name = "_core"`, feature-gated `extension-module` |
| `crates/ferrum-core/src/lib.rs` | Create | `#[pymodule] fn _core` that registers `#[pyfunction] fn add` |
| `pyproject.toml` | Modify | Swap `hatchling` → `maturin` in `[build-system]`; add `[tool.maturin]` block; add `[dependency-groups].dev` |
| `src/ferrum/__init__.py` | Modify | Re-export `add` from `ferrum._core`; set `__version__` and `__all__` |
| `src/ferrum/_core.pyi` | Create | Hand-written type stub so mypy/pyright know the `add` signature |

---

## Task 1 — Install Rust toolchain

**Files:** none (installs to `~/.cargo/`, not the repo)

- [ ] **Step 1.1: Install rustup non-interactively**

Run:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
```

`--profile minimal` skips docs and rust-src (~600 MB saved). You will see output like:
```
info: installing component 'rustc'
info: installing component 'cargo'
...
Rust is installed now. Great!
```

- [ ] **Step 1.2: Source the Rust environment**

Run:
```bash
. "$HOME/.cargo/env"
```

No output is expected. This adds `~/.cargo/bin` to `PATH` for the current shell session. New terminals pick it up automatically via the rc-file modification rustup made.

- [ ] **Step 1.3: Verify cargo is available**

Run:
```bash
cargo --version
```

Expected output (minor version may differ):
```
cargo 1.87.0 (99624be96 2025-05-06)
```

If command not found, check that `~/.cargo/bin` is on `PATH`: `echo $PATH | tr ':' '\n' | grep cargo`.

---

## Task 2 — Write the failing smoke test and .gitignore

**Files:**
- Create: `tests/test_smoke.py`
- Create: `.gitignore`

- [ ] **Step 2.1: Create the smoke test**

Create `tests/test_smoke.py` with this exact content:

```python
def test_core_add():
    from ferrum import add
    assert add(2, 3) == 5
```

- [ ] **Step 2.2: Note — confirm-red step is deferred to Task 6**

We write the test now to maintain TDD discipline (test before implementation), but we cannot confirm the red state until `pytest` is installed. The red confirmation happens at Task 6 Step 6.2, immediately before the Rust build. Do not skip that step.

- [ ] **Step 2.3: Create .gitignore**

Create `.gitignore` with this exact content:

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

Note: `*.so` intentionally excludes the compiled cdylib that `maturin develop` drops into `src/ferrum/`. Build artifacts are not committed; `maturin develop` regenerates them on each rebuild.

---

## Task 3 — Verify PyO3 version and create the workspace Cargo.toml

**Files:**
- Create: `Cargo.toml` (repo root)

- [ ] **Step 3.1: Check the current PyO3 stable version**

Run:
```bash
cargo search pyo3 --limit 1
```

Expected output (version number may be higher than 0.22):
```
pyo3 = "0.22.6"    # Bindings for the Python interpreter
```

If the output shows `0.23.x` or higher, use that version instead of `0.22` in Step 3.2. The `Bound<'_, PyModule>` API used in Task 4 was introduced in PyO3 0.21 and is stable — no changes needed to `lib.rs` for any 0.22+ or 0.23+ version.

- [ ] **Step 3.2: Create the workspace manifest**

Create `Cargo.toml` at the repo root (not inside `crates/`) with this content, substituting the PyO3 version from Step 3.1 if it differs from `0.22`:

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

Key decisions baked in here:
- `resolver = "2"` — the modern Cargo feature resolver; workspaces require it explicitly or Cargo warns.
- `abi3-py310` on the workspace-level PyO3 dep — all member crates inherit this; produces one wheel per platform that works on Python 3.10+.
- `lto = "thin"` and `codegen-units = 1` — applied to release builds across the entire workspace; produces ~5–15% faster numeric code at the cost of slower release compilation.

---

## Task 4 — Create the ferrum-core crate and verify it type-checks

**Files:**
- Create: `crates/ferrum-core/Cargo.toml`
- Create: `crates/ferrum-core/src/lib.rs`

- [ ] **Step 4.1: Create the crate directory structure**

Run:
```bash
mkdir -p /Users/chrissantiago/Dropbox/GitHub/ferrum/crates/ferrum-core/src
```

- [ ] **Step 4.2: Create crates/ferrum-core/Cargo.toml**

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

Three things to understand here:
- `[lib].name = "_core"` — the cdylib filename will be `_core.<abi>.so`. This must match the `#[pymodule] fn _core` in `lib.rs`.
- `[features] extension-module = ["pyo3/extension-module"]` — feature-gated, not unconditional. `maturin develop` enables it via `[tool.maturin].features`; `cargo test` doesn't, so test binaries can link libpython.
- `pyo3 = { workspace = true }` — inherits version and `abi3-py310` from the workspace; the crate adds no extra pyo3 features here.

- [ ] **Step 4.3: Create crates/ferrum-core/src/lib.rs**

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

The function name `_core` in `#[pymodule] fn _core` must match `[lib].name = "_core"` in Cargo.toml. PyO3's macro generates `PyInit__core` from this name; Python's import machinery calls that symbol when loading the extension.

- [ ] **Step 4.4: Verify the Rust code type-checks**

Run (from repo root):
```bash
cargo check
```

On first run, Cargo downloads and compiles PyO3 and its proc-macro dependencies. This takes 1–3 minutes. Expected final lines:
```
   Compiling ferrum-core v0.1.0 (.../crates/ferrum-core)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in Xs
```

If you see errors, consult the failure-mode table in the design spec (`docs/superpowers/specs/2026-05-09-rust-skeleton-design.md §6.3`).

---

## Task 5 — Switch pyproject.toml to maturin and update the Python files

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/ferrum/__init__.py`
- Create: `src/ferrum/_core.pyi`

- [ ] **Step 5.1: Replace pyproject.toml**

Replace the entire contents of `pyproject.toml` with:

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

Critical: `[tool.maturin].features = ["extension-module"]` is what tells maturin to pass `--features extension-module` to cargo at build time, enabling `pyo3/extension-module` for wheel builds only.

Also add `[tool.pytest.ini_options]` so pytest can discover `src/ferrum/` without the project being installed (required for the TDD red-confirmation step in Task 6):

```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
```

Full final `pyproject.toml`:

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

[tool.pytest.ini_options]
pythonpath = ["src"]

[dependency-groups]
dev = ["maturin>=1.7,<2.0", "pytest>=8"]
```

- [ ] **Step 5.2: Replace src/ferrum/__init__.py**

Replace the entire contents of `src/ferrum/__init__.py` with:

```python
"""Ferrum — a statistical visualization library with a Rust core."""

from ferrum._core import add

__all__ = ["add"]
__version__ = "0.1.0"
```

- [ ] **Step 5.3: Create src/ferrum/_core.pyi**

Create `src/ferrum/_core.pyi` with:

```python
def add(a: int, b: int) -> int: ...
```

This stub tells mypy and pyright the type signature of the compiled symbol. Without it, any file that imports `from ferrum._core import add` will get "Module has no attribute 'add'" from type checkers. When the API surface grows past ~10 functions, switch to `pyo3-stub-gen` autogeneration.

---

## Task 6 — Install dev deps, confirm red, then build the extension

**Files:** none created (writes `uv.lock`, creates `.venv/`, writes `_core.<abi>.so` into `src/ferrum/`)

- [ ] **Step 6.1: Install dev dependencies without building the Rust extension**

Run (from repo root):
```bash
uv sync --no-install-project
```

`--no-install-project` installs all deps from `[dependency-groups].dev` (maturin, pytest) into `.venv/` but skips building and installing the project itself. This means no Rust compilation happens yet. Expected output:
```
Resolved N packages in Xs
Installed N packages in Xs
 + maturin==1.x.x
 + pytest==8.x.x
 + ...
```

- [ ] **Step 6.2: Confirm the test fails (TDD: red)**

Run:
```bash
uv run pytest tests/test_smoke.py -v
```

Because `[tool.pytest.ini_options] pythonpath = ["src"]` is set, pytest finds `src/ferrum/__init__.py`. That file tries `from ferrum._core import add`, but the compiled `_core.so` does not yet exist.

Expected failure:
```
FAILED tests/test_smoke.py::test_core_add - ModuleNotFoundError: No module named 'ferrum._core'
```

This is the TDD red state. If the test unexpectedly passes, stop — the `.so` is already present somehow. Check `ls src/ferrum/` and remove any stale `.so` before continuing.

- [ ] **Step 6.3: Build the Rust extension in editable mode (TDD: green)**

Run:
```bash
uv run maturin develop
```

What this does:
1. Reads `pyproject.toml` and finds `build-backend = "maturin"`.
2. Reads `[tool.maturin]` to locate the crate (`manifest-path`) and know where to place the result (`module-name`, `python-source`).
3. Runs `cargo build --features extension-module` on `crates/ferrum-core`.
4. Copies the compiled `_core.<abi>.so` into `src/ferrum/`.
5. Registers the project as editable so `import ferrum` resolves to `src/ferrum/`.

Expected output:
```
🔒 Found pyproject.toml
🍹 Building a mixed python/rust project
...
   Compiling ferrum-core v0.1.0 (.../crates/ferrum-core)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in Xs
📦 Built wheel for CPython 3.10+ (abi3) to /tmp/...
✏️  Setting installed package as editable
🛠 Installed ferrum-0.1.0
```

After this step, `src/ferrum/` will contain a file named something like `_core.cpython-310-abi3-darwin.so` (macOS) or `_core.abi3.so` (Linux). This file is gitignored (`.gitignore` `*.so` line).

If the build fails, check the failure-mode table in `docs/superpowers/specs/2026-05-09-rust-skeleton-design.md §6.3`.

---

## Task 7 — Verify end-to-end and run tests

**Files:** none

- [ ] **Step 7.1: Run the acceptance gate**

Run:
```bash
uv run python -c "import ferrum; assert ferrum.add(2, 3) == 5; print('OK')"
```

Expected output:
```
OK
```

This single assertion verifies six things simultaneously: (a) the wheel built, (b) the cdylib landed where Python can find it, (c) the four-place naming invariant holds, (d) `__init__.py` re-exported the symbol, (e) the `i64 ↔ Python int` conversion works, (f) the interpreter loaded an abi3 module compiled for ≥ 3.10.

If output is anything other than `OK`, see the failure-mode table in the design spec §6.3 before proceeding.

- [ ] **Step 7.2: Run the pytest smoke test**

Run:
```bash
uv run pytest tests/test_smoke.py -v
```

Expected output:
```
tests/test_smoke.py::test_core_add PASSED                                [ 100%]
1 passed in 0.XXs
```

The test was written in Task 2 before any implementation existed (it failed with `ImportError` then). It now passes because the full build chain is in place.

---

## Task 8 — Commit

**Files:** all new and modified files

- [ ] **Step 8.1: Verify final working tree**

Run:
```bash
git status
```

Expected untracked / modified files (order may vary):
```
Changes not staged for commit:
  modified:   pyproject.toml
  modified:   src/ferrum/__init__.py

Untracked files:
  .gitignore
  Cargo.lock
  Cargo.toml
  crates/
  src/ferrum/_core.pyi
  tests/
```

`src/ferrum/_core*.so` should NOT appear — it is covered by the `*.so` gitignore rule. If it appears, confirm `.gitignore` was saved correctly.

- [ ] **Step 8.2: Stage all relevant files**

```bash
git add .gitignore Cargo.toml Cargo.lock crates/ pyproject.toml src/ferrum/__init__.py src/ferrum/_core.pyi tests/
```

Do not use `git add -A` or `git add .` — stage files explicitly to avoid accidentally including `.venv/` or `target/` if the gitignore didn't take effect.

- [ ] **Step 8.3: Confirm staged diff**

```bash
git diff --staged --stat
```

Expected files in the staged diff (sizes will vary):
```
 .gitignore                           |   8 +
 Cargo.lock                           | 300+ +
 Cargo.toml                           |  17 +
 crates/ferrum-core/Cargo.toml        |  16 +
 crates/ferrum-core/src/lib.rs        |  12 +
 pyproject.toml                       |  17 +-
 src/ferrum/__init__.py               |   5 +-
 src/ferrum/_core.pyi                 |   1 +
 tests/test_smoke.py                  |   4 +
```

- [ ] **Step 8.4: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat: wire Rust/maturin build skeleton (phase 1)

Adds Cargo workspace with ferrum-core PyO3 extension crate. Switches
pyproject.toml build backend from hatchling to maturin. Exposes a
trivial add() sanity function through ferrum._core. Smoke test passes:
`import ferrum; ferrum.add(2, 3) == 5`.
EOF
)"
```

- [ ] **Step 8.5: Update ferrum-phases.md — mark phase 1 done**

In `docs/superpowers/ferrum-phases.md`, find the Phase 1 row in the phase table and update `Status` from `in progress` to `done`:

```
| **1** | Build & packaging skeleton | ... | — | [...](specs/...) | **done** |
```

Then commit:
```bash
git add docs/superpowers/ferrum-phases.md
git commit -m "docs: mark phase 1 complete in ferrum-phases.md"
```

---

## Open questions resolved by this plan

| Question (from spec §7) | Resolution |
|---|---|
| Strip `add()` before merge, or leave it? | **Leave it.** It serves as a regression target for sub-project #2 (Arrow IPC). Remove in that PR. |
| PyO3 version pin | Resolved at Task 3 Step 3.1 — checked at execution time via `cargo search pyo3`. |
| CI wheel matrix | Out of scope. First follow-up before any real users. |
