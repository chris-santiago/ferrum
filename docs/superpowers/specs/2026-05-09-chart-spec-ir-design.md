# Phase 3 Design — Chart Spec IR + Serialization

**Date:** 2026-05-09
**Phase slug:** `chart-spec-ir`
**Status:** approved, pending implementation
**Depends on:** Phase 2 (Arrow CDI data-handoff layer — done)

---

## Goals

- Establish the internal Rust representation (`ChartSpec`) that all subsequent phases (scales, stats, layout, render, grammar) consume and produce.
- Lock in the serialization format (JSON via `serde` + `serde_json`) and the canonical wire shape — phases 4–10 add fields to this IR; getting the patterns right now prevents downstream rework.
- Lock in the Python binding pattern (typed `#[pyclass]` with string-shaped enums at the boundary) so phases 4–8 add types consistently.
- Pass `cargo test -p ferrum-core` with full IR round-trip coverage across all eight primitive mark variants and both encoding-type states.

## Non-goals (Phase 3)

- No data binding. The spec is pure metadata; data still arrives separately via the Phase 2 CDI transport at render time.
- No render. No layout. No scales. No stats. The IR is just a typed config.
- No appearance encoding channels (Color, Size, Shape, Tooltip, etc.). The roadmap assigns "all encoding channels from `§3.2`" to Phase 8.
- No Python-side `Chart` wrapper. `ChartSpec` lives in `ferrum._core` only; Phase 8 introduces `ferrum.Chart`.
- No semantic validation (mark-vs-encoding compatibility, field existence). Phase 7+ owns those.

---

## Locked decisions (re-stated for cross-reference)

These were settled during the brainstorming session on 2026-05-09 and are also recorded in `CLAUDE.md` and `docs/superpowers/ferrum-phases.md`.

| Decision | Choice | Rejected alternatives |
|---|---|---|
| Serialization format | JSON via `serde` + `serde_json` | Arrow schema metadata (category mismatch — describes columns, not config trees), binary codec (loses readability and Vega-Lite interop; size class makes perf gains irrelevant) |
| Python binding shape | Typed `#[pyclass]` (Rust-owned, opaque handle from Python) | JSON-string boundary (defers all validation to deserialize time), `pythonize` (extra dep, gives up typed-class benefits) |
| IR breadth | "Minimum-with-shape" — all 8 primitive mark variants, `EncodingSpec { field, type_ }` struct, X/Y only | Bare minimum (under-tests serde patterns), full primitive baseline with Color/Size (out of Phase 3's lane) |
| `DataRef` shape | Sum type with one variant (`Named { name: String }`) | Single-string struct (would force a wire-format break when adding Url/Inline variants), bare placeholder |
| Enum exposure to Python | Strings at the boundary, parsed to typed Rust enums internally | Per-enum `#[pyclass]` (verbose, doesn't compose with eventual `mark_*` shorthand) |

---

## Architecture & module layout

Phase 3 stays inside `crates/ferrum-core/`. The Phase 1/2 single-file `lib.rs` splits into a module tree.

```
crates/ferrum-core/src/
├── lib.rs              # pymodule registration only — thin
├── transport.rs        # Phase 2 process_batch + rename_column (refactored out of lib.rs)
└── spec/
    ├── mod.rs          # pub use re-exports + #[pyclass] wrappers
    ├── chart.rs        # ChartSpec (top-level)
    ├── mark.rs         # Mark enum + FromStr
    ├── encoding.rs     # Encoding, EncodingSpec, DataType
    └── data_ref.rs     # DataRef enum + Default impl
```

**Why split now:** phases 4–7 each add 2–6 types (scales, stats, layout primitives). Setting the module pattern in Phase 3 (when the tree is small) means later phases just add files; without it, `lib.rs` becomes a pile by Phase 7. The Phase 2 → Phase 3 refactor (moving `process_batch` out of `lib.rs`) is intentionally bundled so the layout decision is made once.

**`lib.rs` after Phase 3:**

```rust
mod transport;
mod spec;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(transport::process_batch, m)?)?;
    m.add_class::<spec::ChartSpec>()?;
    m.add_class::<spec::EncodingSpec>()?;
    Ok(())
}
```

The Phase 1 `add` sanity-check function is removed — it has been superseded by real bindings.

---

## Rust data model

Five types across four files. JSON shape and serde decisions baked in.

### `spec/chart.rs`

```rust
use serde::{Deserialize, Serialize};
use crate::spec::{data_ref::DataRef, encoding::Encoding, mark::Mark};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChartSpec {
    #[serde(default)]
    pub data: DataRef,
    pub mark: Mark,
    #[serde(default)]
    pub encoding: Encoding,
}
```

### `spec/encoding.rs`

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Encoding {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub x: Option<EncodingSpec>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub y: Option<EncodingSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EncodingSpec {
    pub field: String,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none", default)]
    pub type_: Option<DataType>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DataType {
    Quantitative,
    Nominal,
    Ordinal,
    Temporal,
}
```

### `spec/mark.rs`

```rust
use serde::{Deserialize, Serialize};

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
```

`FromStr` impl accepts the same lowercase strings serde produces. Errors include the variant list:

```
unknown mark 'pont'; expected one of [point, line, bar, area, rule, text, tick, rect]
```

### `spec/data_ref.rs`

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum DataRef {
    Named { name: String },
    // Future variants (phases 7+): Url { url: String }, Inline { ... }
}

impl Default for DataRef {
    fn default() -> Self {
        DataRef::Named { name: "default".into() }
    }
}
```

### Canonical JSON shape

```json
{
  "data": {"kind": "named", "name": "default"},
  "mark": "point",
  "encoding": {
    "x": {"field": "price"},
    "y": {"field": "weight", "type": "quantitative"}
  }
}
```

### Three deliberate choices

- **`Encoding` is a nested struct, not a `HashMap<String, EncodingSpec>`.** A map keyed by channel name loses field-level type safety and produces less self-documenting JSON. Vega-Lite uses the same nested-struct shape, which keeps Phase 7+ Vega-Lite emission natural.
- **`Encoding.x` / `.y` are `Option<EncodingSpec>` even though scatter requires both.** Required-ness varies by mark (a rule mark may bind only Y); enforcing it at the type level locks Phase 4+ marks out of reusing the struct. Required-ness is enforced at the consumer layer (Phase 7's renderer), not at the IR.
- **`DataType` serializes long-form (`"quantitative"`).** Python users may pass either `"Q"` or `"quantitative"`; both map to the same enum variant. JSON canonical form is the long word — clearer in saved spec files, less collision risk than single letters as the IR grows.

---

## Serialization & round-trip

Two methods on `ChartSpec`, both backed by `serde_json`:

```rust
#[pymethods]
impl ChartSpec {
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(self).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, s: &str) -> PyResult<Self> {
        serde_json::from_str(s).map_err(|e| PyValueError::new_err(e.to_string()))
    }
}
```

### Default-on-missing for forward compatibility

| Field | `#[serde(default)]`? | Rationale |
|---|---|---|
| `ChartSpec.data` | yes | Defaults to `DataRef::Named { name: "default" }` via `Default` impl |
| `ChartSpec.mark` | **no** | No sensible default — every chart has a mark; missing it is a real error |
| `ChartSpec.encoding` | yes | Defaults to `Encoding { x: None, y: None }` — meaningful for marks that don't need encoding |
| `EncodingSpec.field` | no | Required — encoding without a field is meaningless |
| `EncodingSpec.type_` | yes | Optional throughout the spec |

### Round-trip contract (validated by tests)

```python
spec1 = ChartSpec(mark="point", x="price", y="weight")
json1 = spec1.to_json()
spec2 = ChartSpec.from_json(json1)

assert spec1 == spec2          # Rust PartialEq via __eq__
assert json1 == spec2.to_json() # idempotent — JSON output is stable
```

The second equality (idempotent JSON) is what catches subtle drift: if `from_json` silently fills a default that `to_json` then emits, the two-pass JSON differs. This is the test that actually validates the round-trip.

### Strict vs. lax JSON parsing

**Lax** for Phase 3 — no `deny_unknown_fields`. Unknown JSON fields are silently dropped on deserialize. This favors forward-compatibility: a Phase-7-shaped spec deserializes cleanly through a Phase-3 binary even if it carries new fields. The cost (a saved Phase-7 spec round-tripping through a Phase-3 deserializer drops fields) is theoretical for now and revisited when the IR stabilizes (Phase 12).

### Pretty-printing deferred

`to_json()` returns compact form. Phase 8 can add `to_json(*, indent=None)` when public API needs it. Compact form is correct for canonical comparison and avoids whitespace-noise in tests.

---

## Python binding surface

Two `#[pyclass]` types are visible from Python: `ChartSpec` and `EncodingSpec`. `DataRef`, `Mark`, `DataType`, and `Encoding` are accepted via string/kwarg coercion at the boundary — they don't need their own Python classes for Phase 3.

### Imports

```python
from ferrum._core import ChartSpec, EncodingSpec
```

### `EncodingSpec` constructor

```python
EncodingSpec(
    field: str,
    type_: Optional[Literal["Q","N","O","T","quantitative","nominal","ordinal","temporal"]] = None,
)
```

Both short (`"Q"`) and long (`"quantitative"`) forms are accepted at the boundary. Internal storage is the long form. The `_core.pyi` stub advertises a `Literal[...]` of valid values so IDEs autocomplete.

> **Implementation note (2026-05-09):** Field access on `EncodingSpec` is provided by hand-written `#[getter]` methods rather than PyO3's `get_all` derive. Reason: `Option<DataType>` is not `IntoPyObject` (because the internal `DataType` enum is intentionally not a `#[pyclass]`), so `get_all` won't compile. The hand-written getters return `field: &str` and `type_: Option<&'static str>` — `type_` returns the lowercase long-form string (`"quantitative"`, etc.) at the Python boundary, which matches the `Optional[str]` Python attribute type advertised in `_core.pyi`. Behavior from a Python user's perspective is identical to what `get_all` would have produced.

### `ChartSpec` constructor — Phase 3 sugar form

```python
ChartSpec(
    *,
    mark: Literal["point","line","bar","area","rule","text","tick","rect"],
    x: Union[str, EncodingSpec, None] = None,
    y: Union[str, EncodingSpec, None] = None,
    data: Optional[str] = None,   # name for DataRef::Named; defaults to "default"
)
```

Boundary coercion rules:

- `x="price"` (bare string) → `EncodingSpec { field: "price", type_: None }`
- `x=EncodingSpec(field="price", type_="Q")` → passed through
- `x=None` (or omitted) → encoding's `x` stays `None`
- `data="my_table"` → `DataRef::Named { name: "my_table" }`
- `data=None` (or omitted) → `DataRef::Named { name: "default" }`

`mark` is required (no default). Construction with an unknown variant raises:

```
ValueError: unknown mark 'pont'; expected one of [point, line, bar, area, rule, text, tick, rect]
```

### Dunder methods

- `__repr__` — `ChartSpec(mark='point', x=EncodingSpec(field='price'), y=..., data='default')` — stable, useful in tests and debugging.
- `__eq__` — wraps Rust `PartialEq`. Required for the round-trip equality test.
- `__hash__` — explicitly omitted (Python convention: `__eq__` without `__hash__` makes the type unhashable).

### Read-side accessors (getters only — Phase 3 specs are immutable from Python)

```python
spec = ChartSpec.from_json(s)
spec.mark           # "point"
spec.x              # EncodingSpec(field="price", type_=None) or None
spec.y              # EncodingSpec(field="weight", type_="quantitative") or None
spec.data           # "default" (the name string from DataRef::Named)
```

`spec.encoding` is **not** exposed as a separate attribute — it's an internal grouping in JSON, but in Python `spec.x` and `spec.y` are flattened conveniences. Keeps the Phase 3 Python surface small.

### Stub file `_core.pyi` additions

```python
from typing import Literal, Optional, Union

DataTypeStr = Literal["Q","N","O","T","quantitative","nominal","ordinal","temporal"]
MarkStr = Literal["point","line","bar","area","rule","text","tick","rect"]

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

---

## Validation & error handling

Three validation gates, each at its proper boundary:

| Gate | What it checks | Error type |
|---|---|---|
| `ChartSpec.__init__` | `mark` is a known variant; `data` (if provided) is a non-empty string; `x`/`y` (if provided) are `str | EncodingSpec | None` | `ValueError` for unknown enum strings; `TypeError` for wrong arg types |
| `EncodingSpec.__init__` | `type_` (if provided) is a known DataType variant; `field` is non-empty | `ValueError` / `TypeError` |
| `ChartSpec.from_json` | JSON parses; required fields present; enum tags valid | `ValueError` with serde's location-aware message |

**Not validated in Phase 3** (deferred to render time / Phase 7+):

- Whether `field` names reference real columns in any data source.
- Mark-vs-encoding semantic compatibility (e.g., `bar` needing a Y axis).
- Cross-channel consistency.

Phase 3's discipline: **validate structural integrity, defer semantics.** A `ChartSpec` with `mark="point"` and no `x` is structurally valid IR; the renderer rejects it later.

### Error message style

```
ValueError: unknown mark 'pont'; expected one of [point, line, bar, area, rule, text, tick, rect]
ValueError: unknown data type 'X'; expected one of [Q, N, O, T, quantitative, nominal, ordinal, temporal]
ValueError: missing field `mark` at line 3 column 1   (serde-generated)
```

---

## Testing plan

### Rust unit tests (`cargo test -p ferrum-core`)

| Test | Validates |
|---|---|
| `test_chart_spec_round_trip_minimal` | Construct → `to_json` → `from_json` → equal; covers `Mark::Point`, both encodings, default `DataRef` |
| `test_chart_spec_round_trip_idempotent_json` | Two-pass `to_json` after `from_json` produces identical bytes — catches default-fill drift |
| `test_chart_spec_round_trip_with_type` | `EncodingSpec.type_ = Some(DataType::Quantitative)` survives the round-trip |
| `test_chart_spec_round_trip_each_mark_variant` | Iterates all 8 mark variants — justifies enumerating them all in Phase 3 |
| `test_data_ref_defaults_when_omitted` | JSON without `"data"` field deserializes to `DataRef::Named { name: "default" }` |
| `test_unknown_mark_in_json_errors` | `"mark": "spaghetti"` produces a serde error mentioning the variant list |
| `test_missing_required_field_errors` | JSON without `"mark"` produces a clear missing-field error |
| `test_unknown_field_silently_dropped` | `"future_field": ...` deserializes successfully (pins lax-mode behavior) |
| `test_canonical_json_shape` | A known fixture produces the exact JSON string from the data-model section — wire-format pin |

### Python integration tests (`uv run pytest tests/test_chart_spec.py`)

| Test | Validates |
|---|---|
| `test_construct_minimal` | `ChartSpec(mark="point", x="price", y="weight")` works; getters return expected types |
| `test_x_y_string_shorthand` | `x="price"` becomes `EncodingSpec(field="price", type_=None)` |
| `test_x_y_encoding_spec_explicit` | `x=EncodingSpec(field="price", type_="Q")` accepted and preserved |
| `test_data_default_when_omitted` | `spec.data == "default"` |
| `test_data_named` | `data="my_table"` → `spec.data == "my_table"` |
| `test_data_type_short_and_long_forms_equivalent` | `type_="Q"` and `type_="quantitative"` produce the same internal state and JSON |
| `test_unknown_mark_raises` | Unknown variant in `mark` raises `ValueError`; message lists valid variants |
| `test_unknown_data_type_raises` | Unknown variant in `type_` raises `ValueError`; message lists valid variants |
| `test_python_to_json_round_trip` | Construct → `to_json` → `from_json` → `==` |
| `test_python_to_json_idempotent` | Two-pass JSON equal |
| `test_canonical_json_shape` | Known input produces exact JSON string — Python-side wire-format pin |
| `test_repr_contains_fields` | `repr(spec)` includes `mark='point'` and the encoded fields |

The canonical-JSON-shape test (both Rust and Python sides) is the wire-format pin — a serde refactor that silently changes the JSON shape will fail this test before saved specs in user pipelines break invisibly.

---

## Files changed summary

| File | Change |
|---|---|
| `Cargo.toml` (workspace root) | Add `serde = { version = "1", features = ["derive"] }` and `serde_json = "1"` to `[workspace.dependencies]` |
| `crates/ferrum-core/Cargo.toml` | Add `serde` and `serde_json` to `[dependencies]` |
| `crates/ferrum-core/src/lib.rs` | **Refactor** — strip inline functions, become module registration only; drop `add` |
| `crates/ferrum-core/src/transport.rs` | **New** — Phase 2's `process_batch`, `rename_column`, and Rust unit tests moved here verbatim |
| `crates/ferrum-core/src/spec/mod.rs` | **New** — submodule declarations + re-exports |
| `crates/ferrum-core/src/spec/chart.rs` | **New** — `ChartSpec` struct + `#[pymethods]` |
| `crates/ferrum-core/src/spec/mark.rs` | **New** — `Mark` enum + `FromStr` |
| `crates/ferrum-core/src/spec/encoding.rs` | **New** — `Encoding`, `EncodingSpec`, `DataType` + `#[pymethods]` for `EncodingSpec` |
| `crates/ferrum-core/src/spec/data_ref.rs` | **New** — `DataRef` enum + `Default` impl |
| `src/ferrum/_core.pyi` | Add `ChartSpec`, `EncodingSpec`, `MarkStr`, `DataTypeStr` Literals; remove `add` stub |
| `src/ferrum/__init__.py` | Drop `add` re-export. No new public re-exports — `ChartSpec` stays in `ferrum._core` until Phase 8 |
| `tests/test_smoke.py` | Replace `add(2,3)==5` smoke check with a `ChartSpec` import + round-trip smoke |
| `tests/test_chart_spec.py` | **New** — 12 Python integration tests |
| `CLAUDE.md` | Update the verify-skeleton command — replace `assert ferrum.add(2,3)==5` with a `ChartSpec` round-trip one-liner |

---

## Implementation order (two commits in the Phase 3 feature branch)

**Commit 1 — Refactor.** Move Phase 2's `process_batch` / `rename_column` from `lib.rs` to `transport.rs`. Drop `add`. Update `_core.pyi`, `__init__.py`, `tests/test_smoke.py`, `CLAUDE.md`. No new behavior — `cargo test` and `uv run pytest` pass identically before and after.

**Commit 2 — Phase 3.** Add `serde` and `serde_json` deps. Create the `spec/` module tree with all five types. Wire `ChartSpec` and `EncodingSpec` into the pymodule. Add the new tests. `cargo test -p ferrum-core` and `uv run pytest` both pass.

Splitting these makes the refactor reviewable on its own — if the module restructuring is ever questioned, it can be reverted without touching the new spec types.

---

## Pre-implementation verification

Confirm at the start of the implementation session (per `CLAUDE.md` guidance):

- `serde` and `serde_json` versions on crates.io still on the `1.x` line (effectively guaranteed but cheap to check).
- PyO3 0.28 `#[classmethod]` signature still accepts `_cls: &Bound<'_, PyType>` (used for `ChartSpec.from_json`).
- `pyo3-arrow 0.17` and `arrow 58` versions unchanged from Phase 2 — no incidental upgrade.
