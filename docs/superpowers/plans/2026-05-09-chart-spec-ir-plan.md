# Phase 3 — Chart Spec IR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land a typed, JSON-serializable Rust `ChartSpec` IR that Python can construct and round-trip via the `ferrum._core` extension, with `cargo test` and `pytest` covering minimal-scatter round-trip plus all 8 mark variants.

**Architecture:** Pure-Rust IR types (`ChartSpec`, `Mark`, `Encoding`, `EncodingSpec`, `DataType`, `DataRef`) under `crates/ferrum-core/src/spec/`, all `#[derive(Serialize, Deserialize)]`. PyO3 wrappers on `ChartSpec` and `EncodingSpec` only — other types are accepted via string/kwarg coercion at the boundary. Phase 2's `process_batch` / `rename_column` move out of `lib.rs` into a new `transport.rs` module as a separate refactor commit.

**Tech Stack:** Rust 2021, PyO3 0.28 (abi3-py310), `serde = "1"`, `serde_json = "1"`, `pyo3-arrow 0.17`, `arrow 58`, polars ≥ 1.0, pyarrow ≥ 15.0, pytest 8.

**Spec:** [`docs/superpowers/specs/2026-05-09-chart-spec-ir-design.md`](../specs/2026-05-09-chart-spec-ir-design.md)

---

## Build commands (memorize these — every step uses them)

| Action | Command |
|---|---|
| Rebuild Python extension | `unset CONDA_PREFIX && uv run --no-sync maturin develop` |
| Rust-side tests | `DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core` |
| Python tests | `uv run pytest` |
| Verify smoke | `unset CONDA_PREFIX && uv run --no-sync python -c "from ferrum._core import ChartSpec; print(ChartSpec(mark='point', x='a', y='b').to_json())"` |

If `cargo` isn't on PATH, run `source ~/.cargo/env` first.

---

## File structure (lock this in before starting)

| File | Purpose |
|---|---|
| `crates/ferrum-core/src/lib.rs` | Module declarations and `#[pymodule]` registration only |
| `crates/ferrum-core/src/transport.rs` | Phase 2's `process_batch` + `rename_column` + their unit tests |
| `crates/ferrum-core/src/spec/mod.rs` | Submodule declarations + `pub use` re-exports |
| `crates/ferrum-core/src/spec/mark.rs` | `Mark` enum + `FromStr` impl + helpers |
| `crates/ferrum-core/src/spec/encoding.rs` | `DataType` enum, `EncodingSpec` struct (`#[pyclass]`), `Encoding` struct |
| `crates/ferrum-core/src/spec/data_ref.rs` | `DataRef` enum + `Default` impl |
| `crates/ferrum-core/src/spec/chart.rs` | `ChartSpec` struct (`#[pyclass]`) + `#[pymethods]` |
| `src/ferrum/_core.pyi` | Add `ChartSpec`, `EncodingSpec`, `MarkStr`, `DataTypeStr`; remove `add` |
| `src/ferrum/__init__.py` | Drop `add` re-export |
| `tests/test_smoke.py` | Replace `add(2,3)==5` smoke with `ChartSpec` round-trip smoke |
| `tests/test_chart_spec.py` | New — 12 Python integration tests |
| `Cargo.toml` (workspace) | Add `serde` and `serde_json` to `[workspace.dependencies]` |
| `crates/ferrum-core/Cargo.toml` | Add `serde` and `serde_json` to `[dependencies]` |
| `CLAUDE.md` | Update verify-skeleton command to use ChartSpec |

---

## Plan layout

- **Phase A:** Refactor — move Phase 2 to `transport.rs`, drop `add`. **One commit.** No new behavior.
- **Phase B:** Build the Rust IR types via TDD. Pure-Rust, `cargo test` only — no Python rebuilds.
- **Phase C:** Add `#[pyclass]` PyO3 bindings to `ChartSpec` and `EncodingSpec`. One Python rebuild.
- **Phase D:** Python integration tests via pytest.
- **Phase E:** Final verification + mark Phase 3 done in `ferrum-phases.md`. **One commit closes Phase 3.**

Phases B, C, D land as a single feature commit (`feat: Phase 3 — Chart Spec IR`) at the end. Phase E is the wrap-up commit (`chore: mark phase 3 done`).

---

## Phase A — Refactor (Commit 1)

### Task A1: Verify baseline tests pass before any changes

**Files:** none modified

- [ ] **Step 1: Run Rust tests on the current codebase**

```
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core
```

Expected: 3 tests pass (`test_rename_round_trip`, `test_rename_unknown_column_errors`, `test_rename_preserves_other_columns`).

- [ ] **Step 2: Run Python tests on the current codebase**

```
uv run pytest
```

Expected: smoke + transport tests pass (current Phase 2 baseline).

If anything fails at this step, **stop and investigate** — Phase A's contract is that tests pass identically before and after.

---

### Task A2: Move `process_batch` and `rename_column` to `transport.rs`

**Files:**
- Create: `crates/ferrum-core/src/transport.rs`
- Modify: `crates/ferrum-core/src/lib.rs`

- [ ] **Step 1: Create `crates/ferrum-core/src/transport.rs` with the existing Phase 2 contents**

```rust
use arrow::array::{RecordBatch, RecordBatchIterator};
use arrow::datatypes::{Field, Schema};
use arrow::error::ArrowError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyRecordBatchReader;
use std::sync::Arc;

pub(crate) fn rename_column(
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

#[pyfunction]
pub(crate) fn process_batch(reader: PyRecordBatchReader) -> PyResult<PyRecordBatchReader> {
    let reader = reader.into_reader()?;
    let schema = reader.schema();

    let old_name = schema
        .fields()
        .first()
        .ok_or_else(|| PyValueError::new_err("input has zero columns"))?
        .name()
        .clone();
    let renamed_col = format!("{}_renamed", old_name);

    let out_schema = Arc::new(Schema::new(
        schema
            .fields()
            .iter()
            .enumerate()
            .map(|(i, f)| {
                if i == 0 {
                    Field::new(&renamed_col, f.data_type().clone(), f.is_nullable())
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
        .map(|b| rename_column(b, &old_name, &renamed_col))
        .collect::<Result<_, _>>()
        .map_err(|e: ArrowError| PyValueError::new_err(e.to_string()))?;

    let out_reader = RecordBatchIterator::new(
        transformed.into_iter().map(Ok::<_, ArrowError>),
        out_schema,
    );
    Ok(PyRecordBatchReader::new(Box::new(out_reader)))
}

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

Note: `process_batch` and `rename_column` are now `pub(crate)`, since `lib.rs` accesses them via the `transport` module.

- [ ] **Step 2: Replace `crates/ferrum-core/src/lib.rs` entirely with this minimal content**

```rust
use pyo3::prelude::*;

mod transport;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(transport::process_batch, m)?)?;
    Ok(())
}
```

The `add` function is dropped — its job (proving the bridge works) is now done by real bindings.

- [ ] **Step 3: Rebuild the extension**

```
unset CONDA_PREFIX && uv run --no-sync maturin develop
```

Expected: clean build, no warnings about unused imports.

- [ ] **Step 4: Verify tests still pass identically**

```
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core
```

Expected: same 3 tests pass (now under `transport::tests::*`).

```
uv run pytest
```

Expected: transport tests pass; smoke test for `add` will fail at this step — that's expected and is fixed in Task A3.

---

### Task A3: Drop `add` from Python side and update CLAUDE.md verify command

**Files:**
- Modify: `src/ferrum/__init__.py`
- Modify: `src/ferrum/_core.pyi`
- Modify: `tests/test_smoke.py`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Read the current `src/ferrum/__init__.py`**

The file likely re-exports `add` from `_core`. Drop that line. After editing, the file should not reference `add` at all. If `__init__.py` only contained the `add` re-export and a docstring, it can be left as just the docstring (or empty).

- [ ] **Step 2: Read and edit `src/ferrum/_core.pyi`**

Remove the `def add(a: int, b: int) -> int: ...` line. (`process_batch` stub stays.) Phase B will add `ChartSpec` / `EncodingSpec` stubs in Task C5.

- [ ] **Step 3: Edit `tests/test_smoke.py`**

Read the existing file. Replace any `from ferrum import add` / `assert add(2, 3) == 5` test with a placeholder smoke test that simply imports the package:

```python
def test_import_ferrum():
    import ferrum  # noqa: F401
```

This keeps the test file alive but doesn't depend on any specific binding. Phase C will replace this with a `ChartSpec` round-trip smoke.

- [ ] **Step 4: Update `CLAUDE.md` verify-skeleton command**

In the build commands table, find the row:

```
| Verify skeleton | `uv run --no-sync python -c "import ferrum; assert ferrum.add(2,3)==5; print('OK')"` |
```

Replace with:

```
| Verify skeleton | `unset CONDA_PREFIX && uv run --no-sync python -c "import ferrum; print('OK')"` |
```

The `unset CONDA_PREFIX` is included for consistency with the other build commands. (This row will be updated again in Task E2 once `ChartSpec` is in place.)

- [ ] **Step 5: Verify all tests pass**

```
unset CONDA_PREFIX && uv run --no-sync maturin develop
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core
uv run pytest
```

Expected: all green. Rust 3 tests; Python smoke + transport tests.

- [ ] **Step 6: Commit Phase A**

```
git add crates/ferrum-core/src/lib.rs crates/ferrum-core/src/transport.rs \
        src/ferrum/__init__.py src/ferrum/_core.pyi \
        tests/test_smoke.py CLAUDE.md
git commit -m "refactor: split lib.rs into transport module; drop Phase 1 add()

Move Phase 2's process_batch and rename_column from lib.rs into
a dedicated transport module. Drop the Phase 1 add() sanity check
now that real bindings exist. No behavior change — cargo test and
pytest pass identically before and after."
```

---

## Phase B — Build the Rust IR types via TDD (no Python rebuilds)

### Task B1: Add `serde` and `serde_json` workspace dependencies

**Files:**
- Modify: `Cargo.toml` (workspace root)
- Modify: `crates/ferrum-core/Cargo.toml`

- [ ] **Step 1: Add to workspace `Cargo.toml`**

In the `[workspace.dependencies]` table, after the `arrow` line, add:

```toml
serde      = { version = "1", features = ["derive"] }
serde_json = { version = "1" }
```

Verified versions on crates.io (2026-05-09): `serde 1.0.228`, `serde_json 1.0.149`.

- [ ] **Step 2: Add to `crates/ferrum-core/Cargo.toml`**

In the `[dependencies]` table, after the `arrow` line:

```toml
serde      = { workspace = true }
serde_json = { workspace = true }
```

- [ ] **Step 3: Verify it compiles**

```
source ~/.cargo/env
cargo build -p ferrum-core
```

Expected: clean build, no warnings.

---

### Task B2: Add `DataRef` with TDD round-trip test

**Files:**
- Create: `crates/ferrum-core/src/spec/mod.rs`
- Create: `crates/ferrum-core/src/spec/data_ref.rs`
- Modify: `crates/ferrum-core/src/lib.rs`

- [ ] **Step 1: Update `lib.rs` to declare the new `spec` module**

```rust
use pyo3::prelude::*;

mod transport;
mod spec;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(transport::process_batch, m)?)?;
    Ok(())
}
```

- [ ] **Step 2: Create `crates/ferrum-core/src/spec/mod.rs` (placeholder for now)**

```rust
pub(crate) mod data_ref;
```

- [ ] **Step 3: Create `crates/ferrum-core/src/spec/data_ref.rs` with the failing test scaffold**

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum DataRef {
    Named { name: String },
}

impl Default for DataRef {
    fn default() -> Self {
        DataRef::Named { name: "default".into() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_ref_named_round_trip() {
        let original = DataRef::Named { name: "my_table".into() };
        let json = serde_json::to_string(&original).unwrap();
        assert_eq!(json, r#"{"kind":"named","name":"my_table"}"#);
        let parsed: DataRef = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn test_data_ref_default() {
        let d = DataRef::default();
        assert_eq!(d, DataRef::Named { name: "default".into() });
    }
}
```

- [ ] **Step 4: Run tests; verify pass**

```
source ~/.cargo/env
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core
```

Expected: 5 tests pass (3 transport + 2 new data_ref).

---

### Task B3: Add `Mark` enum with `FromStr` and round-trip + parse-error tests

**Files:**
- Create: `crates/ferrum-core/src/spec/mark.rs`
- Modify: `crates/ferrum-core/src/spec/mod.rs`

- [ ] **Step 1: Update `spec/mod.rs`**

```rust
pub(crate) mod data_ref;
pub(crate) mod mark;
```

- [ ] **Step 2: Create `crates/ferrum-core/src/spec/mark.rs`**

```rust
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Mark {
    Point,
    Line,
    Bar,
    Area,
    Rule,
    Text,
    Tick,
    Rect,
}

impl Mark {
    pub fn as_str(&self) -> &'static str {
        match self {
            Mark::Point => "point",
            Mark::Line => "line",
            Mark::Bar => "bar",
            Mark::Area => "area",
            Mark::Rule => "rule",
            Mark::Text => "text",
            Mark::Tick => "tick",
            Mark::Rect => "rect",
        }
    }
}

impl fmt::Display for Mark {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug)]
pub struct ParseMarkError(pub String);

impl fmt::Display for ParseMarkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unknown mark '{}'; expected one of [point, line, bar, area, rule, text, tick, rect]",
            self.0
        )
    }
}

impl std::error::Error for ParseMarkError {}

impl FromStr for Mark {
    type Err = ParseMarkError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "point" => Ok(Mark::Point),
            "line" => Ok(Mark::Line),
            "bar" => Ok(Mark::Bar),
            "area" => Ok(Mark::Area),
            "rule" => Ok(Mark::Rule),
            "text" => Ok(Mark::Text),
            "tick" => Ok(Mark::Tick),
            "rect" => Ok(Mark::Rect),
            other => Err(ParseMarkError(other.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mark_round_trip_each_variant() {
        for m in [
            Mark::Point, Mark::Line, Mark::Bar, Mark::Area,
            Mark::Rule, Mark::Text, Mark::Tick, Mark::Rect,
        ] {
            let json = serde_json::to_string(&m).unwrap();
            let parsed: Mark = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, m, "round-trip failed for {m:?}");
        }
    }

    #[test]
    fn test_mark_serde_form_is_lowercase() {
        assert_eq!(serde_json::to_string(&Mark::Point).unwrap(), "\"point\"");
        assert_eq!(serde_json::to_string(&Mark::Bar).unwrap(),   "\"bar\"");
    }

    #[test]
    fn test_mark_from_str_known() {
        assert_eq!(Mark::from_str("point").unwrap(), Mark::Point);
        assert_eq!(Mark::from_str("rect").unwrap(),  Mark::Rect);
    }

    #[test]
    fn test_mark_from_str_unknown_lists_variants() {
        let err = Mark::from_str("pont").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("'pont'"), "msg was: {msg}");
        assert!(msg.contains("point"), "msg was: {msg}");
        assert!(msg.contains("rect"),  "msg was: {msg}");
    }
}
```

- [ ] **Step 3: Run tests; verify pass**

```
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core
```

Expected: 9 tests pass (3 transport + 2 data_ref + 4 mark).

---

### Task B4: Add `DataType`, `EncodingSpec`, `Encoding`

**Files:**
- Create: `crates/ferrum-core/src/spec/encoding.rs`
- Modify: `crates/ferrum-core/src/spec/mod.rs`

- [ ] **Step 1: Update `spec/mod.rs`**

```rust
pub(crate) mod data_ref;
pub(crate) mod mark;
pub(crate) mod encoding;
```

- [ ] **Step 2: Create `crates/ferrum-core/src/spec/encoding.rs`**

```rust
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DataType {
    Quantitative,
    Nominal,
    Ordinal,
    Temporal,
}

impl DataType {
    pub fn as_str(&self) -> &'static str {
        match self {
            DataType::Quantitative => "quantitative",
            DataType::Nominal => "nominal",
            DataType::Ordinal => "ordinal",
            DataType::Temporal => "temporal",
        }
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug)]
pub struct ParseDataTypeError(pub String);

impl fmt::Display for ParseDataTypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unknown data type '{}'; expected one of [Q, N, O, T, quantitative, nominal, ordinal, temporal]",
            self.0
        )
    }
}

impl std::error::Error for ParseDataTypeError {}

impl FromStr for DataType {
    type Err = ParseDataTypeError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Q" | "quantitative" => Ok(DataType::Quantitative),
            "N" | "nominal" => Ok(DataType::Nominal),
            "O" | "ordinal" => Ok(DataType::Ordinal),
            "T" | "temporal" => Ok(DataType::Temporal),
            other => Err(ParseDataTypeError(other.to_string())),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EncodingSpec {
    pub field: String,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none", default)]
    pub type_: Option<DataType>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Encoding {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub x: Option<EncodingSpec>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub y: Option<EncodingSpec>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_type_short_and_long_forms() {
        assert_eq!(DataType::from_str("Q").unwrap(), DataType::Quantitative);
        assert_eq!(DataType::from_str("quantitative").unwrap(), DataType::Quantitative);
        assert_eq!(DataType::from_str("N").unwrap(), DataType::Nominal);
        assert_eq!(DataType::from_str("nominal").unwrap(), DataType::Nominal);
    }

    #[test]
    fn test_data_type_unknown() {
        let err = DataType::from_str("Z").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("'Z'"), "msg: {msg}");
        assert!(msg.contains("quantitative"), "msg: {msg}");
    }

    #[test]
    fn test_data_type_serde_long_form() {
        assert_eq!(serde_json::to_string(&DataType::Quantitative).unwrap(), "\"quantitative\"");
    }

    #[test]
    fn test_encoding_spec_round_trip_no_type() {
        let original = EncodingSpec { field: "price".into(), type_: None };
        let json = serde_json::to_string(&original).unwrap();
        assert_eq!(json, r#"{"field":"price"}"#);
        let parsed: EncodingSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn test_encoding_spec_round_trip_with_type() {
        let original = EncodingSpec {
            field: "weight".into(),
            type_: Some(DataType::Quantitative),
        };
        let json = serde_json::to_string(&original).unwrap();
        assert_eq!(json, r#"{"field":"weight","type":"quantitative"}"#);
        let parsed: EncodingSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn test_encoding_round_trip_both_axes() {
        let e = Encoding {
            x: Some(EncodingSpec { field: "price".into(), type_: None }),
            y: Some(EncodingSpec { field: "weight".into(), type_: Some(DataType::Quantitative) }),
        };
        let json = serde_json::to_string(&e).unwrap();
        assert_eq!(
            json,
            r#"{"x":{"field":"price"},"y":{"field":"weight","type":"quantitative"}}"#,
        );
        let parsed: Encoding = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, e);
    }

    #[test]
    fn test_encoding_omits_none_fields() {
        let e = Encoding::default();
        let json = serde_json::to_string(&e).unwrap();
        assert_eq!(json, "{}");
    }
}
```

- [ ] **Step 3: Run tests; verify pass**

```
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core
```

Expected: 16 tests pass (3 transport + 2 data_ref + 4 mark + 7 encoding).

---

### Task B5: Add `ChartSpec` (pure-Rust, no PyO3 yet) with full round-trip suite

**Files:**
- Create: `crates/ferrum-core/src/spec/chart.rs`
- Modify: `crates/ferrum-core/src/spec/mod.rs`

- [ ] **Step 1: Update `spec/mod.rs` to add `chart` and re-exports**

```rust
pub(crate) mod data_ref;
pub(crate) mod mark;
pub(crate) mod encoding;
pub(crate) mod chart;
```

- [ ] **Step 2: Create `crates/ferrum-core/src/spec/chart.rs`**

```rust
use serde::{Deserialize, Serialize};

use crate::spec::data_ref::DataRef;
use crate::spec::encoding::Encoding;
use crate::spec::mark::Mark;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChartSpec {
    #[serde(default)]
    pub data: DataRef,
    pub mark: Mark,
    #[serde(default)]
    pub encoding: Encoding,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::encoding::{DataType, EncodingSpec};

    fn minimal_scatter() -> ChartSpec {
        ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "price".into(), type_: None }),
                y: Some(EncodingSpec {
                    field: "weight".into(),
                    type_: Some(DataType::Quantitative),
                }),
            },
        }
    }

    #[test]
    fn test_chart_spec_round_trip_minimal() {
        let original = minimal_scatter();
        let json = serde_json::to_string(&original).unwrap();
        let parsed: ChartSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn test_chart_spec_round_trip_idempotent_json() {
        let original = minimal_scatter();
        let json1 = serde_json::to_string(&original).unwrap();
        let parsed: ChartSpec = serde_json::from_str(&json1).unwrap();
        let json2 = serde_json::to_string(&parsed).unwrap();
        assert_eq!(json1, json2, "two-pass JSON differed");
    }

    #[test]
    fn test_chart_spec_round_trip_each_mark_variant() {
        for m in [
            Mark::Point, Mark::Line, Mark::Bar, Mark::Area,
            Mark::Rule, Mark::Text, Mark::Tick, Mark::Rect,
        ] {
            let mut spec = minimal_scatter();
            spec.mark = m;
            let json = serde_json::to_string(&spec).unwrap();
            let parsed: ChartSpec = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, spec, "round-trip failed for {m:?}");
        }
    }

    #[test]
    fn test_data_ref_defaults_when_omitted() {
        let json = r#"{"mark":"point","encoding":{}}"#;
        let parsed: ChartSpec = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.data, DataRef::Named { name: "default".into() });
    }

    #[test]
    fn test_unknown_mark_in_json_errors() {
        let json = r#"{"data":{"kind":"named","name":"d"},"mark":"spaghetti","encoding":{}}"#;
        let err = serde_json::from_str::<ChartSpec>(json).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("spaghetti") || msg.contains("variant"), "msg: {msg}");
    }

    #[test]
    fn test_missing_required_field_errors() {
        let json = r#"{"encoding":{}}"#;
        let err = serde_json::from_str::<ChartSpec>(json).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("mark"), "expected 'mark' in error, got: {msg}");
    }

    #[test]
    fn test_unknown_field_silently_dropped() {
        let json = r#"{"mark":"point","encoding":{},"future_field":42}"#;
        let parsed: ChartSpec = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.mark, Mark::Point);
    }

    #[test]
    fn test_canonical_json_shape() {
        let spec = ChartSpec {
            data: DataRef::Named { name: "default".into() },
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "price".into(), type_: None }),
                y: Some(EncodingSpec {
                    field: "weight".into(),
                    type_: Some(DataType::Quantitative),
                }),
            },
        };
        let json = serde_json::to_string(&spec).unwrap();
        assert_eq!(
            json,
            r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"price"},"y":{"field":"weight","type":"quantitative"}}}"#,
        );
    }
}
```

- [ ] **Step 3: Run tests; verify pass**

```
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core
```

Expected: 24 tests pass (3 transport + 2 data_ref + 4 mark + 7 encoding + 8 chart).

If `test_unknown_mark_in_json_errors` fails on the message-contents check, inspect the actual serde error format — different serde versions phrase "unknown variant" differently. Loosen the assertion to whatever phrasing the real error uses, but keep at least one substring check (`spaghetti` is the safe bet).

---

## Phase C — PyO3 bindings (one Python rebuild)

### Task C1: Add `#[pyclass]` to `EncodingSpec`

**Files:**
- Modify: `crates/ferrum-core/src/spec/encoding.rs`

- [ ] **Step 1: Add PyO3 imports at the top of `encoding.rs`**

After the existing `use` lines:

```rust
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
```

- [ ] **Step 2: Decorate `EncodingSpec` with `#[pyclass]`**

Replace the existing `EncodingSpec` definition with:

```rust
#[pyclass(eq, get_all, module = "ferrum._core")]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EncodingSpec {
    pub field: String,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none", default)]
    pub type_: Option<DataType>,
}
```

`get_all` makes `field` and `type_` readable from Python via attribute access. `eq` wires `__eq__` to `PartialEq`.

Note: `type_` will be visible from Python as `type_` (PyO3 strips the trailing underscore convention is not automatic — verify in Task C5 stub).

- [ ] **Step 3: Add `#[pymethods] impl EncodingSpec` block at the bottom of the file (before `#[cfg(test)]`)**

```rust
#[pymethods]
impl EncodingSpec {
    #[new]
    #[pyo3(signature = (field, type_ = None))]
    fn new(field: &str, type_: Option<&str>) -> PyResult<Self> {
        if field.is_empty() {
            return Err(PyValueError::new_err("field must be non-empty"));
        }
        let type_ = match type_ {
            Some(s) => Some(
                s.parse::<DataType>()
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            ),
            None => None,
        };
        Ok(EncodingSpec { field: field.to_string(), type_ })
    }

    fn __repr__(&self) -> String {
        match &self.type_ {
            None => format!("EncodingSpec(field={:?})", self.field),
            Some(t) => format!("EncodingSpec(field={:?}, type_={:?})", self.field, t.as_str()),
        }
    }
}
```

The `_` prefix on the unused `PyTypeError` import: keep the import — it's used in `chart.rs` Task C2. Or remove it now and re-add later. Easier: remove it now, add it back when it's needed.

Actually: keep the `PyTypeError` import out of `encoding.rs` if it's not used here. Add it only where it's referenced. Cargo will warn on unused imports.

- [ ] **Step 4: Try to build, expect compilation success**

```
source ~/.cargo/env
cargo build -p ferrum-core
```

Expected: clean build. If you get `unused import: PyTypeError`, remove that import (it's only needed in `chart.rs` Task C2).

- [ ] **Step 5: Run cargo tests; verify all still pass**

```
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core
```

Expected: same 24 tests pass — adding `#[pyclass]` to `EncodingSpec` didn't change Rust semantics.

---

### Task C2: Add `#[pyclass]` and Python sugar constructor to `ChartSpec`

**Files:**
- Modify: `crates/ferrum-core/src/spec/chart.rs`

- [ ] **Step 1: Add PyO3 imports at the top of `chart.rs`**

After the existing `use` lines:

```rust
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::str::FromStr;

use crate::spec::encoding::EncodingSpec;
```

- [ ] **Step 2: Decorate `ChartSpec` with `#[pyclass]`**

Replace the struct definition with:

```rust
#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChartSpec {
    #[serde(default)]
    pub data: DataRef,
    pub mark: Mark,
    #[serde(default)]
    pub encoding: Encoding,
}
```

We do NOT use `get_all` here because Python's `mark`, `x`, `y`, `data` views are computed (not direct field accessors). We hand-write them in the next step.

- [ ] **Step 3: Add `#[pymethods] impl ChartSpec` block at the bottom of the file (before `#[cfg(test)]`)**

```rust
#[pymethods]
impl ChartSpec {
    #[new]
    #[pyo3(signature = (*, mark, x = None, y = None, data = None))]
    fn new(
        mark: &str,
        x: Option<&Bound<'_, PyAny>>,
        y: Option<&Bound<'_, PyAny>>,
        data: Option<&str>,
    ) -> PyResult<Self> {
        let mark = Mark::from_str(mark)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let x = x.map(coerce_encoding).transpose()?;
        let y = y.map(coerce_encoding).transpose()?;

        let data = match data {
            None => DataRef::default(),
            Some(name) if name.is_empty() => {
                return Err(PyValueError::new_err("data name must be non-empty"))
            }
            Some(name) => DataRef::Named { name: name.to_string() },
        };

        Ok(ChartSpec {
            data,
            mark,
            encoding: Encoding { x, y },
        })
    }

    #[getter]
    fn mark(&self) -> &'static str {
        self.mark.as_str()
    }

    #[getter]
    fn x(&self) -> Option<EncodingSpec> {
        self.encoding.x.clone()
    }

    #[getter]
    fn y(&self) -> Option<EncodingSpec> {
        self.encoding.y.clone()
    }

    #[getter]
    fn data(&self) -> &str {
        match &self.data {
            DataRef::Named { name } => name,
        }
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(self).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[classmethod]
    fn from_json<'py>(_cls: &Bound<'py, PyType>, s: &str) -> PyResult<Self> {
        serde_json::from_str(s).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        let mark = self.mark.as_str();
        let data = match &self.data {
            DataRef::Named { name } => name.as_str(),
        };
        let x = match &self.encoding.x {
            None => "None".to_string(),
            Some(e) => format!("EncodingSpec(field={:?})", e.field), // brief — full repr in EncodingSpec
        };
        let y = match &self.encoding.y {
            None => "None".to_string(),
            Some(e) => format!("EncodingSpec(field={:?})", e.field),
        };
        format!("ChartSpec(mark={mark!r}, x={x}, y={y}, data={data!r})")
    }
}

fn coerce_encoding(obj: &Bound<'_, PyAny>) -> PyResult<EncodingSpec> {
    if let Ok(s) = obj.extract::<String>() {
        if s.is_empty() {
            return Err(PyValueError::new_err("encoding field name must be non-empty"));
        }
        return Ok(EncodingSpec { field: s, type_: None });
    }
    if let Ok(spec) = obj.extract::<EncodingSpec>() {
        return Ok(spec);
    }
    Err(PyTypeError::new_err(
        "expected str or EncodingSpec for encoding channel",
    ))
}
```

- [ ] **Step 4: Register `ChartSpec` and `EncodingSpec` in the pymodule**

Edit `crates/ferrum-core/src/lib.rs`:

```rust
use pyo3::prelude::*;

mod transport;
mod spec;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(transport::process_batch, m)?)?;
    m.add_class::<spec::chart::ChartSpec>()?;
    m.add_class::<spec::encoding::EncodingSpec>()?;
    Ok(())
}
```

- [ ] **Step 5: Run cargo tests; verify all still pass**

```
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core
```

Expected: 24 tests pass.

---

### Task C3: Build the Python extension and smoke-test the bindings

**Files:** none modified

- [ ] **Step 1: Build**

```
unset CONDA_PREFIX && uv run --no-sync maturin develop
```

Expected: clean build. If you see errors about `eq` on `#[pyclass]`, your PyO3 version may need `eq, eq_int` together — but for non-int enums on a struct, plain `eq` is sufficient on 0.28.

- [ ] **Step 2: Smoke-test the Python surface**

```
unset CONDA_PREFIX && uv run --no-sync python -c "
from ferrum._core import ChartSpec, EncodingSpec
spec = ChartSpec(mark='point', x='price', y=EncodingSpec(field='weight', type_='Q'))
print(spec)
print(spec.to_json())
spec2 = ChartSpec.from_json(spec.to_json())
assert spec == spec2, 'round-trip failed'
print('OK')
"
```

Expected output (last line `OK`; the JSON line should match the canonical shape):

```
ChartSpec(mark='point', x=EncodingSpec(field='price'), y=EncodingSpec(field='weight'), data='default')
{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"price"},"y":{"field":"weight","type":"quantitative"}}}
OK
```

If the smoke fails, debug here before proceeding — a broken extension cascades to every Phase D test.

---

### Task C4: Update the smoke test to round-trip ChartSpec

**Files:**
- Modify: `tests/test_smoke.py`

- [ ] **Step 1: Read the current `tests/test_smoke.py`**

It should be the placeholder `test_import_ferrum` from Task A3.

- [ ] **Step 2: Replace with a ChartSpec smoke**

```python
def test_chart_spec_smoke():
    from ferrum._core import ChartSpec

    spec = ChartSpec(mark="point", x="a", y="b")
    json = spec.to_json()
    spec2 = ChartSpec.from_json(json)
    assert spec == spec2
    assert spec.to_json() == spec2.to_json()
```

- [ ] **Step 3: Run pytest; verify pass**

```
uv run pytest tests/test_smoke.py -v
```

Expected: `test_chart_spec_smoke PASSED`.

---

### Task C5: Update `_core.pyi` with the new types

**Files:**
- Modify: `src/ferrum/_core.pyi`

- [ ] **Step 1: Replace the file contents**

```python
from typing import Any, Literal, Optional, Union

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
```

- [ ] **Step 2: Sanity-check the stub matches reality**

```
unset CONDA_PREFIX && uv run --no-sync python -c "
from ferrum._core import ChartSpec, EncodingSpec
spec = ChartSpec(mark='point', x='a', y='b')
assert isinstance(spec.mark, str), type(spec.mark)
assert spec.x is not None and spec.x.field == 'a'
assert spec.y is not None and spec.y.field == 'b'
assert spec.data == 'default'
print('OK')
"
```

If the assertion on `spec.x.field` fails because `field` is not accessible, check that `EncodingSpec` was decorated with `get_all`. If `spec.mark` returns a non-string, check the `#[getter]` for `mark` returns `&'static str`.

---

## Phase D — Python integration tests

### Task D1: Write `tests/test_chart_spec.py` with 12 tests

**Files:**
- Create: `tests/test_chart_spec.py`

- [ ] **Step 1: Create the test file**

```python
"""Phase 3 — ChartSpec Python integration tests.

These tests exercise the Python boundary: the #[pyclass] constructors,
the string-shorthand sugar for x/y, the round-trip through JSON, and
the canonical wire format. The Rust side already verifies serde
mechanics; these tests verify Python semantics.
"""

import pytest

from ferrum._core import ChartSpec, EncodingSpec


# -- Construction ---------------------------------------------------------

def test_construct_minimal():
    spec = ChartSpec(mark="point", x="price", y="weight")
    assert spec.mark == "point"
    assert spec.x is not None and spec.x.field == "price"
    assert spec.y is not None and spec.y.field == "weight"
    assert spec.data == "default"


def test_x_y_string_shorthand():
    spec = ChartSpec(mark="point", x="price", y="weight")
    assert isinstance(spec.x, EncodingSpec)
    assert spec.x.field == "price"
    assert spec.x.type_ is None


def test_x_y_encoding_spec_explicit():
    e = EncodingSpec(field="price", type_="Q")
    spec = ChartSpec(mark="point", x=e, y="weight")
    assert spec.x is not None
    assert spec.x.field == "price"
    assert spec.x.type_ == "quantitative"


def test_data_default_when_omitted():
    spec = ChartSpec(mark="point", x="a", y="b")
    assert spec.data == "default"


def test_data_named():
    spec = ChartSpec(mark="point", x="a", y="b", data="my_table")
    assert spec.data == "my_table"


def test_data_type_short_and_long_forms_equivalent():
    s_short = ChartSpec(mark="point", x=EncodingSpec(field="p", type_="Q"), y="w")
    s_long = ChartSpec(
        mark="point",
        x=EncodingSpec(field="p", type_="quantitative"),
        y="w",
    )
    assert s_short == s_long
    assert s_short.to_json() == s_long.to_json()


# -- Errors ---------------------------------------------------------------

def test_unknown_mark_raises():
    with pytest.raises(ValueError) as exc_info:
        ChartSpec(mark="spaghetti", x="a", y="b")
    msg = str(exc_info.value)
    assert "spaghetti" in msg
    assert "point" in msg  # variant list present


def test_unknown_data_type_raises():
    with pytest.raises(ValueError) as exc_info:
        EncodingSpec(field="x", type_="Z")
    msg = str(exc_info.value)
    assert "'Z'" in msg
    assert "quantitative" in msg


# -- Round-trip -----------------------------------------------------------

def test_python_to_json_round_trip():
    spec = ChartSpec(mark="point", x="price", y=EncodingSpec(field="weight", type_="Q"))
    json = spec.to_json()
    spec2 = ChartSpec.from_json(json)
    assert spec == spec2


def test_python_to_json_idempotent():
    spec = ChartSpec(mark="point", x="price", y=EncodingSpec(field="weight", type_="Q"))
    json1 = spec.to_json()
    spec2 = ChartSpec.from_json(json1)
    json2 = spec2.to_json()
    assert json1 == json2


def test_canonical_json_shape():
    spec = ChartSpec(
        mark="point",
        x="price",
        y=EncodingSpec(field="weight", type_="Q"),
    )
    expected = (
        '{"data":{"kind":"named","name":"default"},'
        '"mark":"point",'
        '"encoding":{"x":{"field":"price"},'
        '"y":{"field":"weight","type":"quantitative"}}}'
    )
    assert spec.to_json() == expected


# -- Repr -----------------------------------------------------------------

def test_repr_contains_fields():
    spec = ChartSpec(mark="point", x="price", y="weight")
    r = repr(spec)
    assert "mark='point'" in r
    assert "price" in r
    assert "weight" in r
```

- [ ] **Step 2: Run the new tests; verify all pass**

```
uv run pytest tests/test_chart_spec.py -v
```

Expected: 12 PASSED.

If `test_unknown_mark_raises` fails because the variant list is partial (e.g., the `from_str` error message doesn't mention `point`), check that `ParseMarkError`'s `Display` impl uses the list from `Mark::as_str` for ALL variants. The plan's `mark.rs` Step 2 includes the full list explicitly.

If `test_x_y_encoding_spec_explicit` fails because `spec.x.type_` returns `None` (Python doesn't see the type), check that `EncodingSpec` has `get_all` and that `type_` field name on the Rust struct matches what Python sees. PyO3 0.28 strips the trailing underscore from public field names by default — verify by inspecting `dir(spec.x)`.

If the field is `type` (not `type_`) in Python, update the test assertions and the `_core.pyi` stub accordingly. The internal serde rename to `"type"` for JSON does not affect Python attribute access — only the Rust field name does.

---

### Task D2: Run the full suite end-to-end

**Files:** none modified

- [ ] **Step 1: Run all Rust tests**

```
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core
```

Expected: 24 tests pass.

- [ ] **Step 2: Run all Python tests**

```
uv run pytest -v
```

Expected: smoke + transport + chart_spec tests all pass. Total Python tests: at least 1 smoke + 4 transport + 12 chart_spec = 17.

- [ ] **Step 3: Manually verify the verify-skeleton command works**

```
unset CONDA_PREFIX && uv run --no-sync python -c "from ferrum._core import ChartSpec; print(ChartSpec(mark='point', x='a', y='b').to_json())"
```

Expected: prints the canonical-form JSON ending in `OK`-equivalent (just the JSON string).

- [ ] **Step 4: Commit Phase B+C+D as the Phase 3 feature commit**

```
git add Cargo.toml \
        crates/ferrum-core/Cargo.toml \
        crates/ferrum-core/src/lib.rs \
        crates/ferrum-core/src/spec/ \
        src/ferrum/_core.pyi \
        tests/test_smoke.py \
        tests/test_chart_spec.py
git commit -m "feat: Phase 3 — Chart Spec IR with JSON round-trip

Add ChartSpec, EncodingSpec, Mark, DataType, Encoding, DataRef under
crates/ferrum-core/src/spec/. Pure Rust types with serde derives
plus PyO3 #[pyclass] bindings on ChartSpec and EncodingSpec. Python
construction accepts string shorthand for x/y; mark and data type
validation parses lowercase strings to typed enums with
helpful error messages.

Round-trip: ChartSpec.to_json() -> ChartSpec.from_json() preserves
equality; canonical JSON shape pinned by both Rust and Python tests.

Tests: 24 cargo tests (3 transport + 21 spec); 12 pytest tests
covering construction sugar, error paths, round-trip, and wire
format."
```

---

## Phase E — Mark Phase 3 done

### Task E1: Update `ferrum-phases.md` to mark Phase 3 status

**Files:**
- Modify: `docs/superpowers/ferrum-phases.md`

- [ ] **Step 1: Mark Phase 3 done in the Phases table**

Find the Phase 3 row:

```
| **3** | Chart spec IR + serialization | ... | 2 | *(not yet written)* | pending |
```

Update to:

```
| **3** | Chart spec IR + serialization | Internal Rust representation of a `Chart`; Python builds it, Rust consumes it; round-trip tests | 2 | [`2026-05-09-chart-spec-ir-design.md`](specs/2026-05-09-chart-spec-ir-design.md) | **done** |
```

- [ ] **Step 2: Check the four boxes in the Phase 3 done-criteria section**

```
### Phase 3 — Chart spec IR
- [x] A `ChartSpec` Rust struct exists with enough fields to represent a single-layer scatter plot (data ref, x/y encoding, mark type)
- [x] Python can construct it via a `ferrum._core.ChartSpec` binding and pass it to Rust
- [x] Rust can round-trip serialize/deserialize `ChartSpec` to/from JSON via `serde_json` (decision 2026-05-09 — see locked-decisions table)
- [x] `cargo test` covers at least one round-trip case
```

- [ ] **Step 3: Update the "Last updated" date if you want — not strictly required**

---

### Task E2: Update CLAUDE.md verify command to use ChartSpec

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Find the verify-skeleton row**

After Task A3 it reads:

```
| Verify skeleton | `unset CONDA_PREFIX && uv run --no-sync python -c "import ferrum; print('OK')"` |
```

- [ ] **Step 2: Replace it with a ChartSpec round-trip**

```
| Verify skeleton | `unset CONDA_PREFIX && uv run --no-sync python -c "from ferrum._core import ChartSpec; s=ChartSpec(mark='point', x='a', y='b'); assert s == ChartSpec.from_json(s.to_json()); print('OK')"` |
```

This is a real round-trip smoke now, which is more valuable than the import-only check.

---

### Task E3: Final commit

**Files:** none modified beyond the previous tasks

- [ ] **Step 1: Run the entire suite one more time before closing**

```
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core
uv run pytest
```

Expected: all green.

- [ ] **Step 2: Commit the doc updates**

```
git add docs/superpowers/ferrum-phases.md CLAUDE.md
git commit -m "chore: mark Phase 3 done; update verify command to ChartSpec"
```

- [ ] **Step 3: Verify the branch state is clean**

```
git status
```

Expected: working tree clean.

---

## Done criteria check (from `ferrum-phases.md`)

By the end of Task E3:

- [x] A `ChartSpec` Rust struct exists with enough fields to represent a single-layer scatter plot (data ref, x/y encoding, mark type)
- [x] Python can construct it via a `ferrum._core.ChartSpec` binding and pass it to Rust
- [x] Rust can round-trip serialize/deserialize `ChartSpec` to/from JSON via `serde_json`
- [x] `cargo test` covers at least one round-trip case (in fact, 21 spec tests)

Phase 3 is then ready for merge into `main` (after PR review per the user's normal flow).

---

## Notes on potential snags

**`#[pyclass(eq)]` on PyO3 0.28.** This is the supported syntax for wiring `__eq__` to `PartialEq`. If the build complains, fall back to writing `fn __eq__(&self, other: &Self) -> bool { self == other }` inside `#[pymethods]`.

**`get_all` and field access.** PyO3 0.28's `get_all` exposes all `pub` fields as Python attributes. Field name `type_` is exposed as `type_` in Python (no automatic stripping). The `_core.pyi` stub matches this.

**Serde error message text varies.** The `test_unknown_mark_in_json_errors` test asserts that a substring is present. If the assertion fails on a serde version bump, inspect the actual error text and update the assertion to match — the goal is to verify *some* informative error, not a specific phrasing.

**`data` getter returns `&str`.** PyO3 borrows from `&self`, so the lifetime is tied to the Python call. Python sees a Python `str` (an owned copy). No lifetime drama at the boundary — but if the Rust compiler complains about a missing lifetime annotation on the `#[getter]` for `data`, inline a clone: `match &self.data { DataRef::Named { name } => name.clone() }` returning `String`.

**Tests that depend on canonical JSON ordering.** Serde preserves struct field declaration order. If anyone reorders the fields in `ChartSpec`, `Encoding`, or `EncodingSpec`, the canonical-JSON-shape test breaks. That's the test working as intended — it's the wire-format pin.
