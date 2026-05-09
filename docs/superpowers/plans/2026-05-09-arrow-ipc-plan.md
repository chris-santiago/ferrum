# Phase 2 — Arrow CDI Data-Handoff Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the Python↔Rust data-handoff layer using the Arrow C Data Interface via `pyo3-arrow` so polars DataFrames and pyarrow Tables cross the PyO3 boundary with zero row-level Python access after handoff.

**Architecture:** A pure Rust `rename_column` function (no PyO3 dependency) handles the Phase 2 proof-of-concept transform. A thin `process_batch` PyO3 shim accepts a `PyRecordBatchReader` (any `__arrow_c_stream__` implementor), collects the stream into batches, applies the transform per batch, and returns a new `PyRecordBatchReader`. A Python `_transport.py` module wraps the shim with an `__arrow_c_stream__` type guard before the Rust boundary. This pattern — pure fn + thin shim + Python guard — is the template for all subsequent data-touching phases.

**Tech Stack:** Rust (`arrow` crate from apache/arrow-rs, `pyo3-arrow` crate, PyO3 0.28), Python (polars ≥ 1.0, pyarrow ≥ 15.0, maturin ≥ 1.7, pytest)

---

## File map

| File | Action | Responsibility |
|---|---|---|
| `Cargo.toml` (workspace root) | Modify | Add `pyo3-arrow`, `arrow` to `[workspace.dependencies]` |
| `crates/ferrum-core/Cargo.toml` | Modify | Pull workspace deps into crate `[dependencies]` |
| `crates/ferrum-core/src/lib.rs` | Modify | `rename_column` (pure fn), `process_batch` (PyO3 shim), Rust unit tests |
| `pyproject.toml` | Modify | Add `polars>=1.0`, `pyarrow>=15.0` to `[project.dependencies]` |
| `src/ferrum/_transport.py` | Create | Python wrapper with `__arrow_c_stream__` type guard |
| `src/ferrum/_core.pyi` | Modify | Add `process_batch` stub |
| `tests/test_transport.py` | Create | Four Python integration tests |

---

## Tasks

### Task 1: Verify and pin Cargo dependency versions

**Files:**
- Modify: `Cargo.toml` (workspace root)

Dependency versions in the spec (`pyo3-arrow = "0.4"`, `arrow = "55"`) are estimates. Verify against crates.io before pinning.

- [ ] **Step 1: Check current pyo3-arrow version**

```bash
cargo search pyo3-arrow
```

Look for the crate published by `kylebarron`. Note the latest version number. If the latest is `0.4.x` or higher, use it. If pyo3-arrow does not yet declare compatibility with PyO3 0.28, check the crate's GitHub for a release branch or an in-progress PR — you may need to pin a git revision temporarily and leave a comment in `Cargo.toml`.

- [ ] **Step 2: Check current arrow version**

```bash
cargo search arrow --limit 5
```

Find the `arrow` crate from the Apache Software Foundation. Note the latest stable version (the meta-crate that re-exports arrow-array, arrow-schema, arrow-ipc, etc.).

- [ ] **Step 3: Add workspace dependencies to `Cargo.toml`**

Open `Cargo.toml` (repo root). Current `[workspace.dependencies]`:
```toml
[workspace.dependencies]
pyo3 = { version = "0.28", features = ["abi3-py310"] }
```

Replace with (substituting the versions found in Steps 1–2):
```toml
[workspace.dependencies]
pyo3       = { version = "0.28", features = ["abi3-py310"] }
pyo3-arrow = { version = "0.4" }
arrow      = { version = "55", default-features = false, features = ["ipc"] }
```

`default-features = false` on `arrow` keeps the binary lean — we pull only the core types plus IPC support. Core array/schema types (`RecordBatch`, `Schema`, `RecordBatchIterator`) are available regardless of features.

> **Note on `pyo3-arrow` API:** If `PyRecordBatchReader::into_reader()` requires a `Python<'_>` GIL token in the version you resolve (check the crate docs), update the shim signature in Task 6 to `fn process_batch(py: Python<'_>, reader: PyRecordBatchReader)` and call `reader.into_reader(py)?`.

- [ ] **Step 4: Commit**

```bash
git add Cargo.toml
git commit -m "chore: pin pyo3-arrow and arrow workspace dependencies (phase 2)"
```

---

### Task 2: Wire Cargo dependencies into ferrum-core

**Files:**
- Modify: `crates/ferrum-core/Cargo.toml`

- [ ] **Step 1: Add dependencies to the crate manifest**

Current `[dependencies]` in `crates/ferrum-core/Cargo.toml`:
```toml
[dependencies]
pyo3 = { workspace = true }
```

Replace with:
```toml
[dependencies]
pyo3       = { workspace = true }
pyo3-arrow = { workspace = true }
arrow      = { workspace = true }
```

- [ ] **Step 2: Verify the crate compiles**

```bash
source ~/.cargo/env && cargo check -p ferrum-core
```

Expected: exits 0, no errors. If `pyo3-arrow` fails to resolve at the pinned version, try the previous minor version (e.g. `0.3` instead of `0.4`).

- [ ] **Step 3: Commit**

```bash
git add crates/ferrum-core/Cargo.toml
git commit -m "chore: add pyo3-arrow and arrow to ferrum-core dependencies (phase 2)"
```

---

### Task 3: Add Python runtime dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add polars and pyarrow to `[project.dependencies]`**

Current `pyproject.toml` `[project]` section:
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
```

Change the `dependencies` line to:
```toml
dependencies = [
    "polars>=1.0",
    "pyarrow>=15.0",
]
```

- [ ] **Step 2: Sync the environment and update the lock file**

```bash
uv sync
```

Expected: uv resolves polars and pyarrow, updates `uv.lock`. No errors.

- [ ] **Step 3: Verify both imports work**

```bash
uv run python -c "import polars as pl; import pyarrow as pa; print(pl.__version__, pa.__version__)"
```

Expected: prints two version strings, no ImportError.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add polars and pyarrow as hard runtime dependencies (phase 2)"
```

---

### Task 4: Write failing Rust unit tests

**Files:**
- Modify: `crates/ferrum-core/src/lib.rs`

Write the tests before implementing `rename_column`. A compile error counts as "failing" in TDD.

- [ ] **Step 1: Append test module to lib.rs**

Append this block to the end of `crates/ferrum-core/src/lib.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int32Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn make_two_col_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int32, false),
            Field::new("y", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(Float64Array::from(vec![4.0, 5.0, 6.0])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_rename_round_trip() {
        let batch = make_two_col_batch();
        let result = rename_column(batch, "x", "x_renamed").unwrap();
        assert_eq!(result.schema().field(0).name(), "x_renamed");
        assert_eq!(result.num_rows(), 3);
    }

    #[test]
    fn test_rename_unknown_column_errors() {
        let batch = make_two_col_batch();
        let err = rename_column(batch, "nonexistent", "new_name");
        assert!(err.is_err());
        let msg = err.unwrap_err().to_string();
        assert!(msg.contains("nonexistent"), "error message was: {msg}");
    }

    #[test]
    fn test_rename_preserves_other_columns() {
        let batch = make_two_col_batch();
        let result = rename_column(batch, "x", "x_renamed").unwrap();
        assert_eq!(result.num_columns(), 2);
        assert_eq!(result.schema().field(1).name(), "y");
    }
}
```

- [ ] **Step 2: Confirm compile failure (expected)**

```bash
source ~/.cargo/env && cargo test -p ferrum-core 2>&1 | head -20
```

Expected: compile error — `cannot find function \`rename_column\` in this scope`. This confirms the tests are wired.

---

### Task 5: Implement `rename_column` — make Rust tests pass

**Files:**
- Modify: `crates/ferrum-core/src/lib.rs`

- [ ] **Step 1: Add imports**

At the top of `crates/ferrum-core/src/lib.rs`, after the existing `use pyo3::prelude::*;` line, add:

```rust
use arrow::array::RecordBatch;
use arrow::datatypes::{ArrowError, Field, Schema};
use std::sync::Arc;
```

- [ ] **Step 2: Add `rename_column` before the `add` function**

```rust
fn rename_column(
    batch: RecordBatch,
    old_name: &str,
    new_name: &str,
) -> Result<RecordBatch, ArrowError> {
    let schema = batch.schema();
    let idx = schema.index_of(old_name).map_err(|_| {
        ArrowError::InvalidArgumentError(format!(
            "column '{}' not found; available: {:?}",
            old_name,
            schema.fields().iter().map(|f| f.name()).collect::<Vec<_>>()
        ))
    })?;
    let new_fields: Vec<Field> = schema
        .fields()
        .iter()
        .enumerate()
        .map(|(i, f)| {
            if i == idx {
                Field::new(new_name, f.data_type().clone(), f.is_nullable())
            } else {
                (**f).clone()
            }
        })
        .collect();
    RecordBatch::try_new(Arc::new(Schema::new(new_fields)), batch.columns().to_vec())
}
```

- [ ] **Step 3: Run Rust tests — expect 3 passing**

```bash
source ~/.cargo/env && cargo test -p ferrum-core
```

Expected:
```
running 3 tests
test tests::test_rename_preserves_other_columns ... ok
test tests::test_rename_round_trip ... ok
test tests::test_rename_unknown_column_errors ... ok

test result: ok. 3 passed; 0 failed
```

- [ ] **Step 4: Commit**

```bash
git add crates/ferrum-core/src/lib.rs
git commit -m "feat: implement rename_column pure Rust fn with unit tests (phase 2)"
```

---

### Task 6: Implement `process_batch` PyO3 shim

**Files:**
- Modify: `crates/ferrum-core/src/lib.rs`

- [ ] **Step 1: Update imports**

Replace the arrow import added in Task 5 with the expanded version (adds `RecordBatchIterator`) and add the pyo3-arrow import. The top of `lib.rs` should now read:

```rust
use arrow::array::{RecordBatch, RecordBatchIterator};
use arrow::datatypes::{ArrowError, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyRecordBatchReader;
use std::sync::Arc;
```

- [ ] **Step 2: Add `process_batch` after `rename_column`**

```rust
#[pyfunction]
fn process_batch(reader: PyRecordBatchReader) -> PyResult<PyRecordBatchReader> {
    let reader = reader.into_reader()?;
    let schema = reader.schema();

    let first_col_name = schema
        .fields()
        .first()
        .ok_or_else(|| PyValueError::new_err("input has zero columns"))?
        .name()
        .clone();
    let new_name = format!("{}_renamed", first_col_name);

    // Build output schema up front so it is available even for empty streams.
    let out_schema = Arc::new(Schema::new(
        schema
            .fields()
            .iter()
            .enumerate()
            .map(|(i, f)| {
                if i == 0 {
                    Field::new(&new_name, f.data_type().clone(), f.is_nullable())
                } else {
                    (**f).clone()
                }
            })
            .collect::<Vec<_>>(),
    ));

    let batches: Vec<RecordBatch> = reader
        .collect::<Result<_, _>>()
        .map_err(|e: ArrowError| PyValueError::new_err(e.to_string()))?;

    let transformed: Vec<RecordBatch> = batches
        .into_iter()
        .map(|b| rename_column(b, &first_col_name, &new_name))
        .collect::<Result<_, _>>()
        .map_err(|e: ArrowError| PyValueError::new_err(e.to_string()))?;

    let out_reader = RecordBatchIterator::new(
        transformed.into_iter().map(Ok::<_, ArrowError>),
        out_schema,
    );
    Ok(PyRecordBatchReader::new(Box::new(out_reader)))
}
```

- [ ] **Step 3: Register `process_batch` in the `_core` module**

Update the `_core` pymodule function:

```rust
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(process_batch, m)?)?;
    Ok(())
}
```

- [ ] **Step 4: Confirm Rust tests still pass**

```bash
source ~/.cargo/env && cargo test -p ferrum-core
```

Expected: 3 tests pass, 0 failures.

- [ ] **Step 5: Build the Python extension**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
```

Expected: `✅ ferrum` (maturin success line). No linker errors. If you see a PyO3 version mismatch, check that the `pyo3` version in `Cargo.toml` matches what maturin expects.

- [ ] **Step 6: Smoke-test the new binding**

```bash
uv run python -c "from ferrum._core import process_batch; print(process_batch)"
```

Expected: `<built-in function process_batch>` (or a similar built-in descriptor). No ImportError.

- [ ] **Step 7: Commit**

```bash
git add crates/ferrum-core/src/lib.rs
git commit -m "feat: implement process_batch PyO3 shim and register in _core (phase 2)"
```

---

### Task 7: Update `_core.pyi` type stub

**Files:**
- Modify: `src/ferrum/_core.pyi`

- [ ] **Step 1: Replace the stub file**

Current content of `src/ferrum/_core.pyi`:
```python
def add(a: int, b: int) -> int: ...
```

Replace with:
```python
from typing import Any

def add(a: int, b: int) -> int: ...
def process_batch(data: Any) -> Any:
    """Accept any Arrow stream (__arrow_c_stream__), apply column rename, return Arrow stream.

    Returns a PyRecordBatchReader. Consume with pl.from_arrow(result) or
    pa.Table.from_batches(list(result)).
    """
    ...
```

`Any` is intentional for Phase 2. Phase 3 will introduce typed `ChartSpec` bindings that narrow the signature.

- [ ] **Step 2: Commit**

```bash
git add src/ferrum/_core.pyi
git commit -m "chore: add process_batch stub to _core.pyi (phase 2)"
```

---

### Task 8: Write failing Python integration tests

**Files:**
- Create: `tests/test_transport.py`

- [ ] **Step 1: Create the test file**

Create `tests/test_transport.py`:

```python
import pyarrow as pa
import polars as pl
import pytest

from ferrum._transport import process_batch


def test_polars_round_trip():
    df = pl.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
    result = process_batch(df)
    out = pl.from_arrow(result)
    assert "x_renamed" in out.columns
    assert "y" in out.columns
    assert len(out) == 3


def test_pyarrow_round_trip():
    table = pa.table({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
    result = process_batch(table)
    out = pa.Table.from_batches(list(result))
    assert out.schema.field(0).name == "x_renamed"
    assert out.schema.field(1).name == "y"
    assert len(out) == 3


def test_pyarrow_multichunk_round_trip():
    batch1 = pa.record_batch({"x": [1, 2], "y": [3.0, 4.0]})
    batch2 = pa.record_batch({"x": [5, 6], "y": [7.0, 8.0]})
    table = pa.Table.from_batches([batch1, batch2])
    result = process_batch(table)
    out = pa.Table.from_batches(list(result))
    assert out.schema.field(0).name == "x_renamed"
    assert len(out) == 4


def test_invalid_input_raises():
    with pytest.raises(TypeError, match="Arrow-compatible"):
        process_batch({"not": "arrow"})
```

- [ ] **Step 2: Run tests — expect failure**

```bash
uv run pytest tests/test_transport.py -v
```

Expected: `ModuleNotFoundError: No module named 'ferrum._transport'` for all 4 tests. This confirms they are wired correctly.

---

### Task 9: Implement `_transport.py` — make Python tests pass

**Files:**
- Create: `src/ferrum/_transport.py`

- [ ] **Step 1: Create `_transport.py`**

Create `src/ferrum/_transport.py`:

```python
from __future__ import annotations

from typing import Any

from ferrum._core import process_batch as _process_batch


def process_batch(data: Any) -> Any:
    """Pass an Arrow-compatible object through the Rust pipeline.

    Accepts any object implementing __arrow_c_stream__:
    polars DataFrame, pyarrow Table, pyarrow RecordBatch, etc.
    Returns a PyRecordBatchReader. Consume with:
        polars  — pl.from_arrow(result)
        pyarrow — pa.Table.from_batches(list(result))
    """
    if not hasattr(data, "__arrow_c_stream__"):
        raise TypeError(
            f"Expected an Arrow-compatible object (polars DataFrame, "
            f"pyarrow Table/RecordBatch), got {type(data).__name__!r}"
        )
    return _process_batch(data)
```

- [ ] **Step 2: Run tests — expect 4 passing**

```bash
uv run pytest tests/test_transport.py -v
```

Expected:
```
PASSED tests/test_transport.py::test_polars_round_trip
PASSED tests/test_transport.py::test_pyarrow_round_trip
PASSED tests/test_transport.py::test_pyarrow_multichunk_round_trip
PASSED tests/test_transport.py::test_invalid_input_raises

4 passed in X.XXs
```

If `test_polars_round_trip` fails with a column-name mismatch, confirm that polars' `__arrow_c_stream__` export uses the column name `"x"` (not an integer index) and that the Rust `rename_column` receives `"x"` as `old_name`. If the column name in the stream differs, adjust the test DataFrame column name to match what Rust sees.

- [ ] **Step 3: Commit**

```bash
git add src/ferrum/_transport.py tests/test_transport.py
git commit -m "feat: add _transport.py Python wrapper and integration tests (phase 2)"
```

---

### Task 10: Full verification and phase completion

**Files:**
- Modify: `docs/superpowers/ferrum-phases.md`

- [ ] **Step 1: Run full Rust test suite**

```bash
source ~/.cargo/env && cargo test -p ferrum-core
```

Expected: 3 passed, 0 failed.

- [ ] **Step 2: Run full Python test suite**

```bash
uv run pytest -v
```

Expected: `test_smoke.py::test_core_add` + 4 transport tests = **5 passed, 0 failed**.

- [ ] **Step 3: Verify the Phase 1 smoke test still passes**

```bash
uv run --no-sync python -c "import ferrum; assert ferrum.add(2,3)==5; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Check all Phase 2 done criteria**

Open `docs/superpowers/ferrum-phases.md`, section `### Phase 2`. Verify each criterion is satisfied:

- [ ] A polars DataFrame crosses the PyO3 boundary via Arrow CDI — `test_polars_round_trip` ✓
- [ ] A pyarrow RecordBatch crosses the PyO3 boundary via Arrow CDI — `test_pyarrow_round_trip` ✓
- [ ] Rust receives a RecordBatch, applies column rename, returns via CDI — `process_batch` ✓
- [ ] Python receives result with zero row-level access — `_transport.py` never iterates rows ✓
- [ ] `cargo test` passes in `crates/ferrum-core` — Step 1 above ✓

- [ ] **Step 5: Mark phase 2 done and check off criteria**

In `docs/superpowers/ferrum-phases.md`:

Change the phase table row status from `pending` to `**done**`:
```markdown
| **2** | Python↔Rust data-handoff layer | ... | 1 | [`2026-05-09-arrow-ipc-design.md`](specs/2026-05-09-arrow-ipc-design.md) | **done** |
```

Change each `- [ ]` in the `### Phase 2` done-criteria block to `- [x]`.

- [ ] **Step 6: Final commit**

```bash
git add docs/superpowers/ferrum-phases.md
git commit -m "chore: mark phase 2 done — Arrow CDI data-handoff layer complete"
```
