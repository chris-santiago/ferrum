# Phase 2 Design — Python↔Rust Data-Handoff Layer

**Date:** 2026-05-09
**Phase slug:** `arrow-ipc` (name retained for filesystem consistency; transport mechanism is the Arrow C Data Interface — see amendment note below)
**Status:** approved, pending implementation
**Depends on:** Phase 1 (build & packaging skeleton — done)

---

## Amendment note

The original roadmap and `ferrum-spec.md` said "Arrow IPC" for the data transport. After design review, we chose the **Arrow C Data Interface (CDI)** instead. Polars DataFrames implement `__arrow_c_stream__` natively — CDI passes buffer pointers directly with zero copies. Arrow IPC would require serialising to bytes then deserialising on the Rust side (two copies, extra memory). The `pyo3-arrow` crate mediates the CDI boundary in PyO3.

Both `ferrum-spec.md` and `CLAUDE.md` have been updated with dated amendment notes. The done criteria in `ferrum-phases.md` have been updated to reflect CDI.

---

## Goals

- Prove that a Python DataFrame crosses the PyO3 boundary with zero row-level Python access after handoff
- Establish the `RecordBatchReader` stream protocol as the standard data transport contract for all subsequent phases
- Add `polars` and `pyarrow` as hard runtime dependencies
- Pass `cargo test` in `crates/ferrum-core`

## Non-goals (Phase 2)

- No `ChartSpec` — data transport only, no chart semantics
- No pandas support — polars + pyarrow only
- No stat transforms, layout, or rendering
- The trivial Rust transform (column rename) is a proof-of-concept only; it will be replaced in Phase 3

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Python caller                                           │
│  pl.DataFrame / pa.Table / pa.RecordBatch                │
│  (any object implementing __arrow_c_stream__)            │
└───────────────────────────┬──────────────────────────────┘
                            │  src/ferrum/_transport.py
                            │  thin normalisation wrapper
                            ▼
┌──────────────────────────────────────────────────────────┐
│  ferrum._core.process_batch(data)   [PyO3 boundary]      │
│  accepts PyRecordBatchReader → any __arrow_c_stream__    │
└───────────────────────────┬──────────────────────────────┘
                            │  C Data Interface (zero-copy pointer)
                            ▼
┌──────────────────────────────────────────────────────────┐
│  Rust: ferrum-core                                       │
│  ┌─────────────────────────────────────────────────┐    │
│  │  process_batch() — PyO3 shim (thin)             │    │
│  │    reads PyRecordBatchReader stream              │    │
│  │    calls rename_column() per batch              │    │
│  │    wraps result in new PyRecordBatchReader       │    │
│  └──────────────────────┬──────────────────────────┘    │
│                         │                                │
│  ┌──────────────────────▼──────────────────────────┐    │
│  │  rename_column(batch, old, new)                 │    │
│  │  pure Rust fn — no PyO3, fully unit-testable    │    │
│  └─────────────────────────────────────────────────┘    │
└───────────────────────────┬──────────────────────────────┘
                            │  C Data Interface (zero-copy pointer)
                            ▼
┌──────────────────────────────────────────────────────────┐
│  Python: receives PyRecordBatchReader                    │
│  polars: pl.from_arrow(result)                          │
│  pyarrow: pa.Table.from_batches(list(result))           │
└──────────────────────────────────────────────────────────┘
```

**Structural invariants:**

- `_transport.py` never iterates rows. It validates input type via `__arrow_c_stream__` duck-typing only.
- The `process_batch` PyO3 shim is thin — all computation delegates to pure Rust functions.
- Pure Rust transform functions (`rename_column` and all future transforms) have no PyO3 dependency so `cargo test` runs them natively without a Python runtime.
- This architecture is the template for every subsequent data-touching phase.

---

## Rust API surface

### `crates/ferrum-core/Cargo.toml`

New dependencies (versions also pinned in workspace root):

```toml
[dependencies]
pyo3       = { workspace = true }
pyo3-arrow = { workspace = true }
arrow      = { workspace = true, default-features = false, features = ["ipc"] }
```

### Root `Cargo.toml` — workspace dependencies additions

```toml
[workspace.dependencies]
pyo3       = { version = "0.28", features = ["abi3-py310"] }
pyo3-arrow = { version = "0.4" }   # verify against crates.io at build time
arrow      = { version = "55", default-features = false, features = ["ipc"] }
```

> **Note:** Verify `pyo3-arrow` and `arrow` versions against crates.io at the start of the implementation session. PyO3 0.28 compatibility must be confirmed for `pyo3-arrow`.

### `crates/ferrum-core/src/lib.rs` — structure

```rust
use pyo3::prelude::*;
use pyo3_arrow::PyRecordBatchReader;
use arrow_array::RecordBatch;
use arrow_schema::ArrowError;

/// Pure transform — no PyO3 dependency, fully unit-testable via `cargo test`.
fn rename_column(
    batch: RecordBatch,
    old_name: &str,
    new_name: &str,
) -> Result<RecordBatch, ArrowError> {
    // rebuild schema with renamed field, wrap existing column arrays
}

/// PyO3 shim — thin boundary layer only.
/// Trivial transform for Phase 2: renames the first column to "{name}_renamed".
/// This fixed transform exists solely to prove the round-trip; it is replaced in Phase 3.
#[pyfunction]
fn process_batch(reader: PyRecordBatchReader) -> PyResult<PyRecordBatchReader> {
    // consume stream, apply rename_column(batch, first_col_name, "{first_col_name}_renamed")
    // per batch, collect into Vec<RecordBatch>, return new PyRecordBatchReader
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(process_batch, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    // test rename_column directly — no Python runtime required
}
```

### `src/ferrum/_core.pyi` — additions

```python
from typing import Any

def add(a: int, b: int) -> int: ...
def process_batch(data: Any) -> Any: ...
```

`Any` is intentional for Phase 2. Phase 3 will introduce typed `ChartSpec` bindings that narrow the signature. Using `Any` now avoids stub churn.

---

## Python API surface

### `pyproject.toml` — runtime dependencies

```toml
[project]
dependencies = [
    "polars>=1.0",
    "pyarrow>=15.0",
]
```

Version floors rationale:
- `polars>=1.0` — stable public API baseline; `__arrow_c_stream__` support confirmed
- `pyarrow>=15.0` — PyCapsule Interface (`__arrow_c_stream__`) present and stable

### `src/ferrum/_transport.py` — new module

```python
from __future__ import annotations
from typing import Any
from ferrum._core import process_batch as _process_batch


def process_batch(data: Any) -> Any:
    """Pass an Arrow-compatible object through the Rust pipeline.

    Accepts any object implementing __arrow_c_stream__:
    polars DataFrame, pyarrow Table, pyarrow RecordBatch, etc.
    Returns a PyRecordBatchReader consumable by polars or pyarrow.
    """
    if not hasattr(data, "__arrow_c_stream__"):
        raise TypeError(
            f"Expected an Arrow-compatible object (polars DataFrame, "
            f"pyarrow Table/RecordBatch), got {type(data).__name__!r}"
        )
    return _process_batch(data)
```

### `src/ferrum/__init__.py`

No changes. `_transport` remains private (`_`-prefixed). The public `Chart` object (Phase 8) will consume it internally.

---

## Error handling

| Where | Condition | Surfaces as |
|---|---|---|
| Python `_transport.py` | Input lacks `__arrow_c_stream__` | `TypeError` — clear message, before Rust boundary |
| Rust `rename_column` | `old_name` column not found in schema | `ArrowError` → PyO3 converts to `ValueError` |
| Rust `rename_column` | Empty batch (zero columns) | `ArrowError` → `ValueError` |
| Rust `process_batch` | CDI stream yields a batch error | `PyErr` propagated via `?` operator |

Rule: **Python validates type; Rust validates content.** No Python code inspects data values.

---

## Testing plan

### Rust unit tests (`cargo test`)

| Test | Validates |
|---|---|
| `test_rename_round_trip` | `rename_column` produces correct schema and preserves row count |
| `test_rename_unknown_column` | `rename_column` returns `ArrowError` for missing column |
| `test_rename_preserves_other_columns` | other columns untouched after rename |

### Python integration tests (`uv run pytest`)

| Test | Input | Validates |
|---|---|---|
| `test_polars_round_trip` | `pl.DataFrame` | CDI path, polars → Rust → polars |
| `test_pyarrow_round_trip` | `pa.Table` (single batch) | pyarrow → Rust → `pa.Table.from_batches()` |
| `test_pyarrow_multichunk_round_trip` | `pa.Table` with chunked arrays | `RecordBatchReader` multi-batch path; result collected via `pa.Table.from_batches()` |
| `test_invalid_input_raises` | `dict` | `TypeError` with correct message |

The multi-chunk test directly validates the reason `RecordBatchReader` was chosen over `PyRecordBatch`.

---

## Files changed summary

| File | Change |
|---|---|
| `Cargo.toml` (workspace root) | Add `pyo3-arrow`, `arrow` to `[workspace.dependencies]` |
| `crates/ferrum-core/Cargo.toml` | Add `pyo3-arrow`, `arrow` to `[dependencies]` |
| `crates/ferrum-core/src/lib.rs` | Add `rename_column` (pure fn), `process_batch` (PyO3 shim), Rust unit tests |
| `pyproject.toml` | Add `polars>=1.0`, `pyarrow>=15.0` to `[project.dependencies]` |
| `src/ferrum/_transport.py` | New — Python wrapper with `__arrow_c_stream__` guard |
| `src/ferrum/_core.pyi` | Add `process_batch(data: Any) -> Any` stub |
| `tests/test_transport.py` | New — four Python integration tests |
