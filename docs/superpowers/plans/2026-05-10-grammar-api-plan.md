# Phase 8a — Grammar API Surface (Python) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the user-facing Python grammar API on top of Phase 7's renderer — `Chart`, `Layer`, all 31 encoding channels, themes-as-values, `+`/`|`/`&` composition, faceting, annotations, CoordFlip, and three simple statistical marks.

**Architecture:** Python is the declaration API; Rust adds an additive `layers: Option<Vec<Layer>>` field on `ChartSpec` (single-layer JSON shape preserved byte-identically), three new encoding channels (size/shape/opacity), per-mark style overrides via `MarkKwargsSpec`, a `CoordFlip` swap in `prepare.rs`, six more categorical palettes, and a deterministic SVG compositor for `|`/`&`. Composition operators that span multiple data sources route through the SVG compositor instead of growing renderer multi-batch logic.

**Tech Stack:** Rust (PyO3, serde, palette, fontdue, usvg, resvg, tiny-skia — all already in workspace), Python 3.10+, polars + pyarrow + narwhals (~1.x, new) as runtime deps.

**Spec:** `docs/superpowers/specs/2026-05-10-grammar-api-design.md`

---

## File map

### Rust changes (`crates/ferrum-core/src/`)
- **Modify** `spec/chart.rs` — add `layers: Option<Vec<Layer>>`, `coord: Option<CoordKind>`, `mark_style: Option<MarkKwargsSpec>` fields
- **Modify** `spec/encoding.rs` — add 8 deferred typed Option fields + `scale: Option<ScaleSpec>` + `title: Option<String>`
- **Modify** `spec/mod.rs` — re-export new modules
- **Create** `spec/layer.rs` — `Layer { mark, encoding, transforms, mark_style }` struct
- **Create** `spec/coord.rs` — `CoordKind { Cartesian, Flip }` enum
- **Create** `spec/mark_style.rs` — `MarkKwargsSpec` struct (~14 optional fields)
- **Modify** `render/prepare.rs` — handle multi-layer iteration, CoordFlip swap, MarkKwargsSpec overrides
- **Modify** `render/scale_resolve.rs` — honor explicit `Scale` from EncodingSpec; build size/shape/opacity scales
- **Modify** `render/marks/point.rs` — honor per-row size/shape/opacity from resolved scales
- **Modify** `render/palette.rs` — add 6 categorical palettes + `categorical_palette(name: &str) -> Option<&'static [Color]>`
- **Create** `render/compositor.rs` — `compose_svg_horizontal/vertical(svgs, spacing, align) -> String`
- **Modify** `render/mod.rs` — re-export compositor
- **Modify** `render/binding.rs` — add PyO3 binding for `compose_svg_horizontal/vertical`

### Python changes (`src/ferrum/`)
- **Create** `_coerce.py` — `to_arrow_table(data) -> pyarrow.Table` (narwhals + ferrum branches)
- **Create** `_shorthand.py` — `parse_shorthand(s: str) -> tuple[str|None, str|None, str|None]`
- **Create** `_warn.py` — warn-once registry with `warn_once()` and `reset_warnings()`
- **Create** `chart.py` — `Chart` immutable value class
- **Create** `layer.py` — `Layer` value class
- **Create** `composition.py` — `LayerChart`, `HConcatChart`, `VConcatChart`
- **Create** `annotations.py` — `annotate_hline/vline/rect/text`
- **Create** `coord.py` — `CoordFlip` (and NotImplementedError stubs for others)
- **Create** `display.py` — `show/show_svg/show_png/save/_repr_svg_/_repr_html_` mixin
- **Create** `marks/__init__.py` — 8 primitive `mark_*()` builder functions
- **Create** `marks/base.py` — `MarkBase` (mark-kwargs validation)
- **Create** `marks/statistical.py` — `mark_density/mark_histogram/mark_smooth`
- **Create** `marks/deferred.py` — NotImplementedError stubs for 8b/9 marks
- **Create** `encoding/__init__.py` — re-exports
- **Create** `encoding/base.py` — `ChannelBase`
- **Create** `encoding/positional.py` — 10 positional channel classes
- **Create** `encoding/appearance.py` — 11 appearance channel classes
- **Create** `encoding/text.py` — 7 text/detail/tooltip classes
- **Create** `encoding/facet.py` — 3 facet channel classes
- **Create** `themes/__init__.py` — `Theme` value class + 8 builtins re-exported
- **Create** `themes/builtins.py` — 8 builtin themes
- **Create** `themes/_defaults.py` — `set_default_theme()` + contextvar stack
- **Modify** `__init__.py` — re-export all public surface
- **Modify** `_core.pyi` — type stubs for new Rust bindings

### Test files (`tests/`)
- **Create** `tests/test_chart.py` — Chart construction + immutability + data-input variety (~25 tests)
- **Create** `tests/test_marks.py` — primitive + statistical marks (~15 tests)
- **Create** `tests/test_encoding.py` — channels + shorthand + warn-once (~20 tests)
- **Create** `tests/test_composition.py` — `+`/`|`/`&` (~10 tests)
- **Create** `tests/test_theme.py` — Theme + builtins + set_default_theme (~8 tests)
- **Create** `tests/test_facet.py` — Facet/FacetRow/FacetCol (~5 tests)
- **Create** `tests/test_annotations.py` — annotate_* (~5 tests)
- **Create** `tests/test_coord.py` — CoordFlip (~2 tests)
- **Create** `tests/test_show_save.py` — output methods (~6 tests)
- **Create** `tests/test_coerce.py` — data ingestion (~4 tests)

### Docs
- **Modify** `ferrum-spec.md` — dated notes for §3.2, §3.13, §3.16, §3.18
- **Modify** `docs/superpowers/ferrum-phases.md` — mark Phase 8a done; add Phase 8b row

### Build config
- **Modify** `pyproject.toml` — add `narwhals` runtime dep
- **Modify** `Cargo.toml` (workspace) — no new crate deps required (compositor is hand-rolled)

---

## Test count baseline targets

- `cargo test -p ferrum-core`: 261 → ≥ 291 (+30)
- `uv run pytest`: 89 → ≥ 179 (+90)

---

## Task list (38 tasks across 6 groups)

**Group A — Rust spec extensions (Tasks 1–5)**
**Group B — Rust render pipeline (Tasks 6–10)**
**Group C — Rust SVG compositor (Task 11)**
**Group D — Python utilities (Tasks 12–14)**
**Group E — Python encoding channels (Tasks 15–19)**
**Group F — Python theme system (Tasks 20–22)**
**Group G — Python marks (Tasks 23–26)**
**Group H — Python Chart + composition (Tasks 27–29)**
**Group I — Python annotations + coord + display (Tasks 30–33)**
**Group J — Wiring + tests + docs + verification (Tasks 34–38)**

---

## Group A — Rust spec extensions

### Task 1: Add narwhals dep + verify pyarrow.RecordBatch handling

**Files:**
- Modify: `pyproject.toml`
- Test: `tests/test_coerce_smoke.py` (temporary; deleted at end of task)

- [ ] **Step 1: Add narwhals to runtime deps**

In `pyproject.toml` `[project] dependencies`, add `"narwhals>=1.0,<2"` (range pin per spec §10 row 14).

- [ ] **Step 2: Sync deps**

Run: `uv sync`
Expected: `narwhals` shows up in lockfile.

- [ ] **Step 3: Write a one-off probe to verify narwhals + RecordBatch**

Create `tests/test_coerce_smoke.py`:

```python
import pyarrow as pa
import narwhals as nw

def test_narwhals_accepts_pyarrow_recordbatch_or_table():
    """Verify whether narwhals.from_native accepts pa.RecordBatch directly.
    If it doesn't, our _coerce.py must convert RecordBatch → Table at the boundary."""
    rb = pa.RecordBatch.from_pylist([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    tbl = pa.Table.from_batches([rb])

    # Table should always work
    nw_tbl = nw.from_native(tbl, eager_only=True)
    assert nw_tbl.to_arrow().num_rows == 2

    # RecordBatch may or may not work — record the result
    try:
        nw_rb = nw.from_native(rb, eager_only=True)
        result = nw_rb.to_arrow()
        print(f"RecordBatch directly accepted: {result.num_rows} rows")
        assert result.num_rows == 2
    except (TypeError, NotImplementedError) as e:
        print(f"RecordBatch NOT directly accepted: {e}. _coerce.py must convert at boundary.")
        # Either way, ferrum's _coerce.py handles RecordBatch via pa.Table.from_batches
```

- [ ] **Step 4: Run the probe**

Run: `uv run pytest tests/test_coerce_smoke.py -v -s`
Expected: PASS. The print output tells us which branch `_coerce.py` must take. (Spec §10 row 15: ferrum's `_coerce` already handles both branches, so either result is fine.)

- [ ] **Step 5: Delete the probe + commit deps**

```bash
rm tests/test_coerce_smoke.py
git add pyproject.toml uv.lock
git commit -m "deps(phase-8a): add narwhals ~1.x runtime dep for DataFrame compatibility"
```

---

### Task 2: `spec/layer.rs` — `Layer` struct (Rust)

**Files:**
- Create: `crates/ferrum-core/src/spec/layer.rs`
- Modify: `crates/ferrum-core/src/spec/mod.rs`
- Modify: `crates/ferrum-core/src/spec/chart.rs` (add `layers` field)

- [ ] **Step 1: Write failing test for Layer struct + ChartSpec.layers round-trip**

Create `crates/ferrum-core/src/spec/layer.rs`:

```rust
use serde::{Deserialize, Serialize};

use crate::spec::encoding::Encoding;
use crate::spec::mark::Mark;
use crate::transform::core::TransformSpec;

/// A single layer within a multi-layer ChartSpec. Inherits chart-level
/// encoding for any field set to None at the layer level.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Layer {
    pub mark: Mark,
    #[serde(default)]
    pub encoding: Encoding,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub transforms: Vec<TransformSpec>,
    // mark_style: Option<MarkKwargsSpec> added in Task 5
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::encoding::EncodingSpec;

    #[test]
    fn layer_round_trips_minimal() {
        let layer = Layer {
            mark: Mark::Point,
            encoding: Encoding::default(),
            transforms: Vec::new(),
        };
        let json = serde_json::to_string(&layer).unwrap();
        let parsed: Layer = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, layer);
    }

    #[test]
    fn layer_round_trips_with_encoding() {
        let layer = Layer {
            mark: Mark::Line,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: None,
            },
            transforms: Vec::new(),
        };
        let json = serde_json::to_string(&layer).unwrap();
        let parsed: Layer = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, layer);
    }
}
```

> **Note:** `..Default::default()` requires `EncodingSpec` to derive `Default`. If it doesn't yet, this will fail to compile — Task 3 adds that derive. For now, in Task 2's tests, write out all fields explicitly: `EncodingSpec { field: "x".into(), type_: None }` (matching the current Phase 7 `EncodingSpec` shape).

- [ ] **Step 2: Add module to spec/mod.rs**

In `crates/ferrum-core/src/spec/mod.rs`, add:

```rust
pub mod layer;
pub use layer::Layer;
```

- [ ] **Step 3: Add `layers: Option<Vec<Layer>>` to ChartSpec**

In `crates/ferrum-core/src/spec/chart.rs`, add the field to the struct (after `facet`):

```rust
#[serde(default, skip_serializing_if = "Option::is_none")]
pub layers: Option<Vec<Layer>>,
```

Add `use crate::spec::layer::Layer;` at the top.

In the `#[new]` constructor, add `layers: Option<&Bound<'_, PyAny>> = None` to the signature, and after the existing `transforms` parsing add a parser:

```rust
let layers = match layers {
    None => None,
    Some(obj) => Some(coerce_layers(obj)?),
};
```

Add the `coerce_layers` helper function (mirror `coerce_transforms`):

```rust
fn coerce_layers(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Layer>> {
    use pyo3::types::PyDict;
    let list = obj.downcast::<pyo3::types::PyList>()
        .map_err(|_| PyValueError::new_err("layers must be a list"))?;
    let mut out = Vec::with_capacity(list.len());
    for (i, item) in list.iter().enumerate() {
        // Layer items pass through as JSON dicts from Python; deserialize via serde_json
        let json_str: String = item.call_method0("__repr__")?.extract()?;
        // Python sends dicts; convert via py_any → JSON string → Layer
        let py_dict: &Bound<PyDict> = item.downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err(format!("layers[{i}] must be a dict")))?;
        // Use pythonize to convert PyDict → serde_json::Value if needed; else manual
        // For Phase 8a, layers are constructed Python-side as dicts mirroring Layer fields
        let json = pyo3::Python::with_gil(|py| {
            let json_module = py.import("json")?;
            let s: String = json_module.call_method1("dumps", (py_dict,))?.extract()?;
            Ok::<String, pyo3::PyErr>(s)
        })?;
        let layer: Layer = serde_json::from_str(&json)
            .map_err(|e| PyValueError::new_err(format!("layers[{i}]: {e}")))?;
        out.push(layer);
    }
    Ok(out)
}
```

> **Simpler alternative:** if `pythonize` is already a dep, use `pythonize::depythonize(item)` to skip the JSON round-trip. Check `crates/ferrum-core/Cargo.toml` first.

In the `ChartSpec { ... }` constructor body, add `layers,`. Default to `None` if not passed.

Add a getter `#[getter] fn layers(&self, py: Python) -> PyResult<Option<Vec<Py<PyAny>>>>` that mirrors `transforms` getter (returns Python dict per layer).

- [ ] **Step 4: Round-trip tests for ChartSpec.layers**

Add to `crates/ferrum-core/src/spec/chart.rs` `mod tests`:

```rust
#[test]
fn test_chart_spec_layers_default_when_omitted() {
    let json = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{}}"#;
    let parsed: ChartSpec = serde_json::from_str(json).unwrap();
    assert!(parsed.layers.is_none());
}

#[test]
fn test_chart_spec_layers_omitted_in_canonical_json_when_none() {
    let spec = minimal_scatter();  // existing test helper
    let json = serde_json::to_string(&spec).unwrap();
    assert!(!json.contains("layers"), "layers=None should be skipped: {json}");
}

#[test]
fn test_chart_spec_layers_round_trip() {
    use crate::spec::layer::Layer;
    let mut spec = minimal_scatter();
    spec.layers = Some(vec![
        Layer { mark: Mark::Point, encoding: Encoding::default(), transforms: Vec::new() },
        Layer { mark: Mark::Line, encoding: Encoding::default(), transforms: Vec::new() },
    ]);
    let json = serde_json::to_string(&spec).unwrap();
    assert!(json.contains(r#""layers":["#));
    let parsed: ChartSpec = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, spec);
}

#[test]
fn test_existing_phase_7_canonical_json_unchanged() {
    // Asserts Phase 3-7 byte-identical JSON shape when layers.is_none()
    let spec = minimal_scatter();
    let json = serde_json::to_string(&spec).unwrap();
    assert_eq!(
        json,
        r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"price"},"y":{"field":"weight","type":"quantitative"}}}"#,
    );
}
```

- [ ] **Step 5: Build + run tests**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core layer
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core chart
```

Expected: all 5 new tests pass; existing 261 tests still pass.

- [ ] **Step 6: Commit**

```bash
git add crates/ferrum-core/src/spec/layer.rs \
        crates/ferrum-core/src/spec/mod.rs \
        crates/ferrum-core/src/spec/chart.rs
git commit -m "feat(spec): add Layer struct + additive ChartSpec.layers field"
```

---

### Task 3: Extend `EncodingSpec` with deferred + honored kwargs (Rust)

**Files:**
- Modify: `crates/ferrum-core/src/spec/encoding.rs`

- [ ] **Step 1: Define ScaleSpec, AxisSpec, LegendSpec stub structs**

EncodingSpec needs typed Option fields. The deferred kwargs (axis, legend, sort, stack, impute, scheme, format, formatType) need typed structs. For Phase 8a these are "opaque-but-typed" — they accept any JSON shape and round-trip correctly without the renderer interpreting them.

Add to `crates/ferrum-core/src/spec/encoding.rs` (after the existing types):

```rust
/// Scale override on an encoding channel. Honored by scale_resolve.rs in Phase 8a.
/// Mirrors the Python ScaleLog/ScalePow/etc. classes via tagged enum.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ScaleSpec {
    Linear { #[serde(default, skip_serializing_if = "Option::is_none")] domain: Option<Vec<f64>>,
             #[serde(default, skip_serializing_if = "Option::is_none")] range: Option<Vec<f64>>,
             #[serde(default)] nice: bool,
             #[serde(default)] zero: bool,
             #[serde(default)] clamp: bool },
    Log    { #[serde(default = "default_log_base")] base: f64,
             #[serde(default, skip_serializing_if = "Option::is_none")] domain: Option<Vec<f64>>,
             #[serde(default, skip_serializing_if = "Option::is_none")] range: Option<Vec<f64>>,
             #[serde(default)] nice: bool,
             #[serde(default)] clamp: bool },
    Time   { #[serde(default, skip_serializing_if = "Option::is_none")] domain: Option<Vec<f64>>,
             #[serde(default, skip_serializing_if = "Option::is_none")] range: Option<Vec<f64>>,
             #[serde(default)] nice: bool,
             #[serde(default)] clamp: bool },
    Symlog { #[serde(default = "default_symlog_constant")] constant: f64,
             #[serde(default, skip_serializing_if = "Option::is_none")] domain: Option<Vec<f64>>,
             #[serde(default, skip_serializing_if = "Option::is_none")] range: Option<Vec<f64>>,
             #[serde(default)] nice: bool,
             #[serde(default)] clamp: bool },
    Ordinal { #[serde(default, skip_serializing_if = "Option::is_none")] domain: Option<Vec<String>>,
              #[serde(default, skip_serializing_if = "Option::is_none")] range: Option<Vec<f64>>,
              #[serde(default)] padding: f64 },
}

fn default_log_base() -> f64 { 10.0 }
fn default_symlog_constant() -> f64 { 1.0 }

/// Opaque-but-typed axis spec. Round-trips JSON; renderer ignores in 8a.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct AxisSpec {
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct LegendSpec {
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}
```

- [ ] **Step 2: Extend EncodingSpec with new fields**

Replace the existing `EncodingSpec` struct in the same file with:

```rust
#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct EncodingSpec {
    pub field: String,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none", default)]
    pub type_: Option<DataType>,

    // NEW honored fields (Phase 8a):
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<ScaleSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,

    // NEW deferred fields (Phase 8a — round-trip + warn-once at Python layer):
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub axis: Option<AxisSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub legend: Option<LegendSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sort: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stack: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub impute: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scheme: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    #[serde(rename = "formatType", default, skip_serializing_if = "Option::is_none")]
    pub format_type: Option<String>,
}
```

> **Back-compat:** `Default` derive is added so `..Default::default()` works in tests. All new fields default to `None` and `skip_serializing_if = "Option::is_none"` ensures Phase 3-7 JSON outputs stay byte-identical.

- [ ] **Step 3: Update PyO3 constructor**

The existing `#[new]` for `EncodingSpec` only takes `(field, type_=None)`. Phase 8a keeps this signature unchanged so Python channel-class code works without modification — the new fields are populated via the `ChartSpec` constructor (which accepts pre-built EncodingSpec dicts) or via direct serde deserialization. Leave the `#[new]` alone.

Add getters for the new fields (mirror existing `field` and `type_` getters):

```rust
#[getter] fn scale(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
    match &self.scale {
        None => Ok(None),
        Some(s) => {
            let json = serde_json::to_string(s).map_err(|e| PyValueError::new_err(e.to_string()))?;
            let json_module = py.import("json")?;
            Ok(Some(json_module.call_method1("loads", (json,))?.unbind()))
        }
    }
}
#[getter] fn title(&self) -> Option<&str> { self.title.as_deref() }
#[getter] fn scheme(&self) -> Option<&str> { self.scheme.as_deref() }
// (axis, legend, sort, stack, impute, format, format_type getters analogous;
//  return JSON-roundtripped Python objects for opaque types)
```

- [ ] **Step 4: Tests**

Add to `mod tests` in `encoding.rs`:

```rust
#[test]
fn encoding_spec_round_trips_with_scale() {
    let e = EncodingSpec {
        field: "price".into(),
        type_: Some(DataType::Quantitative),
        scale: Some(ScaleSpec::Log { base: 10.0, domain: None, range: None, nice: true, clamp: false }),
        ..Default::default()
    };
    let json = serde_json::to_string(&e).unwrap();
    assert!(json.contains(r#""scale":{"type":"log""#));
    let parsed: EncodingSpec = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, e);
}

#[test]
fn encoding_spec_round_trips_with_title() {
    let e = EncodingSpec {
        field: "x".into(), type_: None,
        title: Some("My X Axis".into()),
        ..Default::default()
    };
    let json = serde_json::to_string(&e).unwrap();
    assert!(json.contains(r#""title":"My X Axis""#));
    let parsed: EncodingSpec = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, e);
}

#[test]
fn encoding_spec_round_trips_with_axis_opaque() {
    use serde_json::json;
    let mut axis_extra = serde_json::Map::new();
    axis_extra.insert("grid".into(), json!(false));
    axis_extra.insert("orient".into(), json!("bottom"));
    let e = EncodingSpec {
        field: "x".into(), type_: None,
        axis: Some(AxisSpec { extra: axis_extra }),
        ..Default::default()
    };
    let json = serde_json::to_string(&e).unwrap();
    let parsed: EncodingSpec = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, e);
}

#[test]
fn encoding_spec_phase_7_canonical_json_byte_identical_when_no_new_fields() {
    let e = EncodingSpec { field: "x".into(), type_: None, ..Default::default() };
    assert_eq!(serde_json::to_string(&e).unwrap(), r#"{"field":"x"}"#);

    let e2 = EncodingSpec {
        field: "y".into(), type_: Some(DataType::Quantitative),
        ..Default::default()
    };
    assert_eq!(serde_json::to_string(&e2).unwrap(), r#"{"field":"y","type":"quantitative"}"#);
}

#[test]
fn encoding_spec_round_trips_pre_phase_8_json() {
    // Existing JSON without any new fields must deserialize.
    let json = r#"{"field":"price","type":"quantitative"}"#;
    let parsed: EncodingSpec = serde_json::from_str(json).unwrap();
    assert_eq!(parsed.field, "price");
    assert_eq!(parsed.type_, Some(DataType::Quantitative));
    assert_eq!(parsed.scale, None);
    assert_eq!(parsed.title, None);
}

#[test]
fn scale_spec_log_default_base_is_10() {
    let json = r#"{"type":"log"}"#;
    let parsed: ScaleSpec = serde_json::from_str(json).unwrap();
    match parsed {
        ScaleSpec::Log { base, .. } => assert_eq!(base, 10.0),
        _ => panic!("expected Log variant"),
    }
}
```

- [ ] **Step 5: Build + run**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core encoding
```

Expected: 6 new tests pass; all existing encoding tests still pass.

- [ ] **Step 6: Commit**

```bash
git add crates/ferrum-core/src/spec/encoding.rs
git commit -m "feat(spec): EncodingSpec gains scale/title (honored) + 6 deferred kwargs"
```

---

### Task 4: `spec/coord.rs` — `CoordKind` enum (Rust)

**Files:**
- Create: `crates/ferrum-core/src/spec/coord.rs`
- Modify: `crates/ferrum-core/src/spec/mod.rs`
- Modify: `crates/ferrum-core/src/spec/chart.rs` (add `coord` field)

- [ ] **Step 1: Create coord.rs**

```rust
use serde::{Deserialize, Serialize};

/// Coordinate system. Phase 8a honors Cartesian (default no-op) and Flip (swap x/y).
/// Other variants (Polar, Geo, Fixed) are added in Phase 9+.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum CoordKind {
    Cartesian,
    Flip,
}

impl Default for CoordKind {
    fn default() -> Self { CoordKind::Cartesian }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coord_kind_round_trip_cartesian() {
        let c = CoordKind::Cartesian;
        let json = serde_json::to_string(&c).unwrap();
        assert_eq!(json, r#"{"kind":"cartesian"}"#);
        let parsed: CoordKind = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, c);
    }

    #[test]
    fn coord_kind_round_trip_flip() {
        let c = CoordKind::Flip;
        let json = serde_json::to_string(&c).unwrap();
        assert_eq!(json, r#"{"kind":"flip"}"#);
        let parsed: CoordKind = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, c);
    }
}
```

- [ ] **Step 2: Register module**

In `crates/ferrum-core/src/spec/mod.rs` add:

```rust
pub mod coord;
pub use coord::CoordKind;
```

- [ ] **Step 3: Add coord field to ChartSpec**

In `crates/ferrum-core/src/spec/chart.rs`, add after `layers`:

```rust
#[serde(default, skip_serializing_if = "Option::is_none")]
pub coord: Option<CoordKind>,
```

Add `use crate::spec::coord::CoordKind;` at the top.

In the `#[new]` constructor signature, add `coord: Option<&str> = None`. Body:

```rust
let coord = match coord {
    None => None,
    Some("cartesian") => Some(CoordKind::Cartesian),
    Some("flip") => Some(CoordKind::Flip),
    Some(other) => return Err(PyValueError::new_err(format!(
        "unknown coord kind: '{other}'; expected 'cartesian' or 'flip'"
    ))),
};
```

Add `coord,` to the constructor body.

Add a getter:

```rust
#[getter]
fn coord(&self) -> Option<&'static str> {
    match self.coord {
        None => None,
        Some(CoordKind::Cartesian) => Some("cartesian"),
        Some(CoordKind::Flip) => Some("flip"),
    }
}
```

- [ ] **Step 4: Round-trip tests for ChartSpec.coord**

Add to `chart.rs` `mod tests`:

```rust
#[test]
fn test_chart_spec_coord_default_when_omitted() {
    let json = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{}}"#;
    let parsed: ChartSpec = serde_json::from_str(json).unwrap();
    assert!(parsed.coord.is_none());
}

#[test]
fn test_chart_spec_coord_omitted_in_canonical_json_when_none() {
    let spec = minimal_scatter();
    let json = serde_json::to_string(&spec).unwrap();
    assert!(!json.contains("coord"));
}

#[test]
fn test_chart_spec_coord_flip_round_trip() {
    use crate::spec::coord::CoordKind;
    let mut spec = minimal_scatter();
    spec.coord = Some(CoordKind::Flip);
    let json = serde_json::to_string(&spec).unwrap();
    assert!(json.contains(r#""coord":{"kind":"flip"}"#));
    let parsed: ChartSpec = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, spec);
}
```

- [ ] **Step 5: Build + run**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core coord
```

Expected: 5 new tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/ferrum-core/src/spec/coord.rs \
        crates/ferrum-core/src/spec/mod.rs \
        crates/ferrum-core/src/spec/chart.rs
git commit -m "feat(spec): add CoordKind enum + ChartSpec.coord field"
```

---

### Task 5: `spec/mark_style.rs` — `MarkKwargsSpec` struct (Rust)

**Files:**
- Create: `crates/ferrum-core/src/spec/mark_style.rs`
- Modify: `crates/ferrum-core/src/spec/mod.rs`
- Modify: `crates/ferrum-core/src/spec/chart.rs` (add `mark_style` field)
- Modify: `crates/ferrum-core/src/spec/layer.rs` (add `mark_style` field)

- [ ] **Step 1: Create mark_style.rs**

```rust
use serde::{Deserialize, Serialize};

/// Per-mark constant style overrides. Phase 8a fields cover all kwargs accepted
/// by the 8 primitive mark_*() Python methods. All None defaults; renderer falls
/// back to theme defaults when None.
///
/// Resolution priority in prepare.rs: layer.mark_style > chart.mark_style > theme.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct MarkKwargsSpec {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stroke: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fill: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opacity: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub corner_radius: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stroke_width: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stroke_dash: Option<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub font_size: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub font_weight: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub align: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dx: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dy: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub angle: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mark_kwargs_default_omits_all_fields() {
        let m = MarkKwargsSpec::default();
        let json = serde_json::to_string(&m).unwrap();
        assert_eq!(json, "{}");
    }

    #[test]
    fn mark_kwargs_round_trip_with_size_and_stroke() {
        let m = MarkKwargsSpec {
            size: Some(100.0),
            stroke: Some("#ff0000".into()),
            opacity: Some(0.5),
            ..Default::default()
        };
        let json = serde_json::to_string(&m).unwrap();
        let parsed: MarkKwargsSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, m);
    }

    #[test]
    fn mark_kwargs_round_trip_with_stroke_dash() {
        let m = MarkKwargsSpec {
            stroke_dash: Some(vec![5.0, 3.0]),
            ..Default::default()
        };
        let json = serde_json::to_string(&m).unwrap();
        assert!(json.contains(r#""stroke_dash":[5.0,3.0]"#));
        let parsed: MarkKwargsSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, m);
    }
}
```

- [ ] **Step 2: Register module**

In `spec/mod.rs`:

```rust
pub mod mark_style;
pub use mark_style::MarkKwargsSpec;
```

- [ ] **Step 3: Add mark_style to ChartSpec and Layer**

In `spec/chart.rs`, add after `coord`:

```rust
#[serde(default, skip_serializing_if = "Option::is_none")]
pub mark_style: Option<MarkKwargsSpec>,
```

Add `use crate::spec::mark_style::MarkKwargsSpec;`.

In the `#[new]` constructor, add `mark_style: Option<&Bound<'_, PyAny>> = None`. Body:

```rust
let mark_style = match mark_style {
    None => None,
    Some(obj) => {
        let py = obj.py();
        let json_module = py.import("json")?;
        let s: String = json_module.call_method1("dumps", (obj,))?.extract()?;
        Some(serde_json::from_str(&s).map_err(|e| PyValueError::new_err(format!("mark_style: {e}")))?)
    }
};
```

Add `mark_style,` to constructor body.

In `spec/layer.rs`, replace the struct with:

```rust
pub struct Layer {
    pub mark: Mark,
    #[serde(default)]
    pub encoding: Encoding,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub transforms: Vec<TransformSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mark_style: Option<crate::spec::mark_style::MarkKwargsSpec>,
}
```

Update Layer's existing tests to include `mark_style: None` in their constructions.

- [ ] **Step 4: Add round-trip tests for ChartSpec.mark_style and Layer.mark_style**

In `chart.rs` `mod tests`:

```rust
#[test]
fn test_chart_spec_mark_style_default_when_omitted() {
    let json = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{}}"#;
    let parsed: ChartSpec = serde_json::from_str(json).unwrap();
    assert!(parsed.mark_style.is_none());
}

#[test]
fn test_chart_spec_mark_style_round_trip() {
    use crate::spec::mark_style::MarkKwargsSpec;
    let mut spec = minimal_scatter();
    spec.mark_style = Some(MarkKwargsSpec {
        size: Some(100.0),
        stroke: Some("#ff0000".into()),
        ..Default::default()
    });
    let json = serde_json::to_string(&spec).unwrap();
    assert!(json.contains(r#""mark_style":{"size":100.0,"stroke":"#ff0000""#));
    let parsed: ChartSpec = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, spec);
}
```

In `layer.rs` `mod tests`:

```rust
#[test]
fn layer_round_trips_with_mark_style() {
    use crate::spec::mark_style::MarkKwargsSpec;
    let layer = Layer {
        mark: Mark::Point,
        encoding: Encoding::default(),
        transforms: Vec::new(),
        mark_style: Some(MarkKwargsSpec {
            size: Some(50.0),
            ..Default::default()
        }),
    };
    let json = serde_json::to_string(&layer).unwrap();
    let parsed: Layer = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, layer);
}
```

- [ ] **Step 5: Build + run all spec tests**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core spec
```

Expected: all spec tests pass (Tasks 1+2+3+4+5 cumulative ≥ 25 new tests).

- [ ] **Step 6: Commit**

```bash
git add crates/ferrum-core/src/spec/mark_style.rs \
        crates/ferrum-core/src/spec/mod.rs \
        crates/ferrum-core/src/spec/chart.rs \
        crates/ferrum-core/src/spec/layer.rs
git commit -m "feat(spec): MarkKwargsSpec for per-mark style overrides on Chart and Layer"
```

---

## Group B — Rust render pipeline

### Task 6: `prepare.rs` — multi-layer iteration + CoordFlip swap

**Files:**
- Modify: `crates/ferrum-core/src/render/prepare.rs`

Phase 7 currently treats `ChartSpec` as single-layer. Phase 8a extends `prepare_render_inputs` to: (a) when `spec.layers.is_some()`, return per-layer prepared inputs so the draw loop can iterate; (b) when `spec.coord == Some(Flip)`, swap x and y scales / axes / encoding bindings before downstream computation.

- [ ] **Step 1: Locate the existing `prepare_render_inputs` function**

```bash
grep -n "fn prepare_render_inputs" crates/ferrum-core/src/render/prepare.rs
```

Read the function and its returned `PreparedInputs` struct to confirm shape.

- [ ] **Step 2: Add per-layer container**

Add to the file:

```rust
/// Per-layer prepared rendering data. When ChartSpec.layers.is_none(), exactly one
/// LayerPrepared is constructed from the chart-level mark + encoding.
#[derive(Debug, Clone)]
pub struct LayerPrepared {
    pub mark: crate::spec::mark::Mark,
    pub encoding: crate::spec::encoding::Encoding,
    pub transforms: Vec<crate::transform::core::TransformSpec>,
    pub mark_style: Option<crate::spec::mark_style::MarkKwargsSpec>,
}

impl LayerPrepared {
    /// Build a single layer from chart-level fields (single-layer mode).
    pub(crate) fn from_chart_only(spec: &crate::spec::chart::ChartSpec) -> Self {
        Self {
            mark: spec.mark,
            encoding: spec.encoding.clone(),
            transforms: spec.transforms.clone(),
            mark_style: spec.mark_style.clone(),
        }
    }

    /// Build a layer by inheriting from chart-level when layer's encoding fields are None.
    pub(crate) fn from_chart_and_layer(
        spec: &crate::spec::chart::ChartSpec,
        layer: &crate::spec::layer::Layer,
    ) -> Self {
        let mut encoding = layer.encoding.clone();
        // Inherit chart-level encoding when layer encoding fields are unset.
        if encoding.x.is_none() { encoding.x = spec.encoding.x.clone(); }
        if encoding.y.is_none() { encoding.y = spec.encoding.y.clone(); }
        if encoding.color.is_none() { encoding.color = spec.encoding.color.clone(); }
        Self {
            mark: layer.mark,
            encoding,
            transforms: layer.transforms.clone(),
            mark_style: layer.mark_style.clone().or_else(|| spec.mark_style.clone()),
        }
    }
}
```

- [ ] **Step 3: Extend PreparedInputs to carry per-layer data**

Modify the existing `PreparedInputs` struct (assumed shape — adapt to actual):

```rust
pub struct PreparedInputs {
    /// One entry per layer. Single-layer charts have len() == 1.
    pub layers: Vec<LayerPrepared>,
    pub batch: arrow::record_batch::RecordBatch,           // post-transform, post-CoordFlip
    pub axes_input: AxesInput,
    pub facet_groups: Vec<FacetGroup>,
    pub legend_entries: Vec<LegendEntry>,
    pub scales: ResolvedScales,                             // x/y already swapped if CoordFlip
    pub coord_flipped: bool,
}
```

- [ ] **Step 4: Modify `prepare_render_inputs` body**

Add after the existing transforms-application step (skeleton; adapt to current code):

```rust
// Build per-layer prepared inputs
let layers: Vec<LayerPrepared> = match &spec.layers {
    None => vec![LayerPrepared::from_chart_only(spec)],
    Some(layer_vec) => layer_vec.iter()
        .map(|l| LayerPrepared::from_chart_and_layer(spec, l))
        .collect(),
};

// CoordFlip: swap x ↔ y in each layer's encoding before scale resolution
let coord_flipped = matches!(spec.coord, Some(crate::spec::coord::CoordKind::Flip));
let layers = if coord_flipped {
    layers.into_iter().map(|mut lp| {
        let tmp = lp.encoding.x.take();
        lp.encoding.x = lp.encoding.y.take();
        lp.encoding.y = tmp;
        lp
    }).collect()
} else { layers };
```

The downstream `scale_resolve` call should be updated to compute scales over the **union** of all layers' encoded fields (so layered marks share scale domains). For single-layer mode (the current Phase 7 behavior), the union of one layer is the layer itself — fully back-compat.

- [ ] **Step 5: Tests**

Add to `prepare.rs` `mod tests`:

```rust
#[test]
fn prepare_single_layer_produces_one_layer_prepared() {
    let spec = single_layer_spec_fixture();   // existing or new helper
    let prepared = prepare_render_inputs(&spec, &test_batch(), &theme_inputs()).unwrap();
    assert_eq!(prepared.layers.len(), 1);
    assert_eq!(prepared.layers[0].mark, crate::spec::mark::Mark::Point);
}

#[test]
fn prepare_multi_layer_produces_multiple_layer_prepared() {
    use crate::spec::layer::Layer;
    let mut spec = single_layer_spec_fixture();
    spec.layers = Some(vec![
        Layer { mark: crate::spec::mark::Mark::Point,
                encoding: spec.encoding.clone(), transforms: vec![], mark_style: None },
        Layer { mark: crate::spec::mark::Mark::Line,
                encoding: spec.encoding.clone(), transforms: vec![], mark_style: None },
    ]);
    let prepared = prepare_render_inputs(&spec, &test_batch(), &theme_inputs()).unwrap();
    assert_eq!(prepared.layers.len(), 2);
}

#[test]
fn prepare_coord_flip_swaps_x_y_in_each_layer() {
    use crate::spec::coord::CoordKind;
    let mut spec = single_layer_spec_fixture();   // x="price", y="weight"
    spec.coord = Some(CoordKind::Flip);
    let prepared = prepare_render_inputs(&spec, &test_batch(), &theme_inputs()).unwrap();
    assert!(prepared.coord_flipped);
    // After flip, the layer that originally had x=price should now have y=price (and vice versa)
    assert_eq!(prepared.layers[0].encoding.y.as_ref().unwrap().field, "price");
    assert_eq!(prepared.layers[0].encoding.x.as_ref().unwrap().field, "weight");
}
```

- [ ] **Step 6: Update the draw loop to iterate layers**

Find the `dispatch_mark` call in `render/mod.rs` (per Phase 7 plan §6 step 6.b) and wrap it:

```rust
// Within each panel's draw block:
for layer in &prepared.layers {
    let ctx = DrawCtx {
        panel: &panel,
        theme: &theme_inputs,
        scales: &prepared.scales,
        batch: &panel_batch,
        mark_style: &resolve_mark_style(layer.mark_style.as_ref(), theme_inputs),
        // ...
    };
    dispatch_mark(&layer.mark, &ctx, &mut out);
}
```

(`resolve_mark_style` is added in Task 7.)

- [ ] **Step 7: Build + run**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core prepare
```

Expected: 3 new prepare tests pass; existing render tests still pass; existing 6 SVG goldens still match (single-layer code path preserved).

- [ ] **Step 8: Commit**

```bash
git add crates/ferrum-core/src/render/prepare.rs crates/ferrum-core/src/render/mod.rs
git commit -m "feat(render): prepare.rs handles multi-layer + CoordFlip; draw loop iterates layers"
```

---

### Task 7: `prepare.rs` — apply `MarkKwargsSpec` overrides to `MarkStyle`

**Files:**
- Modify: `crates/ferrum-core/src/render/prepare.rs` (add `resolve_mark_style` helper)
- Modify: `crates/ferrum-core/src/render/draw.rs` if `MarkStyle` lives there

- [ ] **Step 1: Find current MarkStyle resolution**

```bash
grep -n "struct MarkStyle\|fn resolve_mark_style\|mark_style:" crates/ferrum-core/src/render/draw.rs crates/ferrum-core/src/render/prepare.rs
```

Phase 7 builds `MarkStyle` from `ThemeInputs` at draw-context-construction time. Phase 8a adds an optional `&MarkKwargsSpec` parameter that overrides theme defaults field-by-field.

- [ ] **Step 2: Write `resolve_mark_style` helper**

Add to `prepare.rs` (or `draw.rs` adjacent to `MarkStyle`):

```rust
use crate::spec::mark_style::MarkKwargsSpec;
use crate::layout::ThemeInputs;
use crate::render::draw::MarkStyle;
use crate::render::color::{from_hex_str, with_opacity};

pub(crate) fn resolve_mark_style(
    overrides: Option<&MarkKwargsSpec>,
    theme: &ThemeInputs,
) -> MarkStyle {
    // Start from theme defaults (existing Phase 7 logic).
    let mut style = MarkStyle::from_theme(theme);
    let Some(o) = overrides else { return style };

    if let Some(size) = o.size { style.point_size = size; }
    if let Some(opacity) = o.opacity { style.opacity = opacity; }
    if let Some(cr) = o.corner_radius { style.corner_radius = cr; }
    if let Some(sw) = o.stroke_width { style.stroke_width = sw; }
    if let Some(ref dash) = o.stroke_dash { style.stroke_dash = Some(dash.clone()); }

    if let Some(ref hex) = o.stroke {
        match from_hex_str(hex) {
            Ok(c) => style.stroke = Some(c),
            Err(_) => {} // silently skip invalid color; warn at Python layer
        }
    }
    if let Some(ref hex) = o.fill {
        match from_hex_str(hex) {
            Ok(c) => style.fill = c,
            Err(_) => {}
        }
    }
    // font_size / font_weight / align / baseline / dx / dy / angle apply only to text marks;
    // store on style if MarkStyle has those fields, else leave for Task 13/15 follow-up.
    style
}
```

If `MarkStyle` lacks the text-mark fields (font_size/etc.), add them as `Option<>` to `MarkStyle` in this task too — they default to None and per-mark draw fns fall back to theme.

- [ ] **Step 3: Tests**

```rust
#[test]
fn resolve_mark_style_with_no_overrides_returns_theme_defaults() {
    let theme = ThemeInputs::default();
    let style = resolve_mark_style(None, &theme);
    assert_eq!(style.point_size, theme.point_size);
}

#[test]
fn resolve_mark_style_overrides_point_size() {
    let theme = ThemeInputs::default();
    let overrides = MarkKwargsSpec { size: Some(100.0), ..Default::default() };
    let style = resolve_mark_style(Some(&overrides), &theme);
    assert_eq!(style.point_size, 100.0);
}

#[test]
fn resolve_mark_style_overrides_stroke_color() {
    let theme = ThemeInputs::default();
    let overrides = MarkKwargsSpec { stroke: Some("#ff0000".into()), ..Default::default() };
    let style = resolve_mark_style(Some(&overrides), &theme);
    let stroke = style.stroke.expect("stroke should be set");
    assert_eq!(stroke.red, 0xff);
    assert_eq!(stroke.green, 0x00);
    assert_eq!(stroke.blue, 0x00);
}

#[test]
fn resolve_mark_style_invalid_color_silently_skipped() {
    let theme = ThemeInputs::default();
    let overrides = MarkKwargsSpec { stroke: Some("not-a-color".into()), ..Default::default() };
    let style = resolve_mark_style(Some(&overrides), &theme);
    // theme default stroke is None; invalid color does NOT set it
    assert_eq!(style.stroke, MarkStyle::from_theme(&theme).stroke);
}
```

- [ ] **Step 4: Wire `resolve_mark_style` into the draw loop**

In the layer-iteration loop added in Task 6, replace `&MarkStyle::from_theme(...)` with `&resolve_mark_style(layer.mark_style.as_ref(), &theme_inputs)`.

- [ ] **Step 5: Build + run**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core resolve_mark_style
```

Expected: 4 new tests pass; goldens unchanged (no MarkKwargsSpec means theme defaults — same as Phase 7).

- [ ] **Step 6: Commit**

```bash
git add crates/ferrum-core/src/render/prepare.rs crates/ferrum-core/src/render/draw.rs
git commit -m "feat(render): MarkStyle resolution honors per-mark MarkKwargsSpec overrides"
```

---

### Task 8: `scale_resolve.rs` — honor explicit `Scale`; build size/shape/opacity scales

**Files:**
- Modify: `crates/ferrum-core/src/render/scale_resolve.rs`

Phase 8a extends `ResolvedScales` with size, shape, opacity scales (each Optional, populated when the corresponding encoding is present), and modifies x/y/color scale construction to honor `EncodingSpec.scale: Option<ScaleSpec>` when present.

- [ ] **Step 1: Extend ResolvedScales**

In `scale_resolve.rs`, replace the existing `ResolvedScales` struct:

```rust
pub struct ResolvedScales {
    pub x: ScaleKind,
    pub y: ScaleKind,
    pub color: Option<ColorScale>,
    // NEW Phase 8a:
    pub size: Option<SizeScale>,
    pub shape: Option<ShapeScale>,
    pub opacity: Option<OpacityScale>,
}

#[derive(Debug, Clone)]
pub struct SizeScale {
    pub inner: ScaleKind,            // typically Linear
    pub min_px: f64,                  // default 3.0
    pub max_px: f64,                  // default 30.0
}

#[derive(Debug, Clone)]
pub struct ShapeScale {
    pub domain: Vec<String>,          // distinct values in encounter order
    pub shapes: Vec<ShapeKind>,       // mapped from a fixed 6-shape palette
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ShapeKind {
    Circle, Square, Cross, Diamond, TriangleUp, TriangleDown,
}

const SHAPE_PALETTE: [ShapeKind; 6] = [
    ShapeKind::Circle, ShapeKind::Square, ShapeKind::Cross,
    ShapeKind::Diamond, ShapeKind::TriangleUp, ShapeKind::TriangleDown,
];

#[derive(Debug, Clone)]
pub struct OpacityScale {
    pub inner: ScaleKind,
    pub min_opacity: f64,             // default 0.1
    pub max_opacity: f64,             // default 1.0
}
```

- [ ] **Step 2: Honor explicit `Scale` in x/y construction**

Modify the existing x/y scale construction (skeleton — adapt to current code):

```rust
fn build_x_scale(
    encoding: &Encoding,
    batch: &RecordBatch,
) -> Result<ScaleKind, RenderError> {
    let Some(x_enc) = &encoding.x else { return Err(RenderError::MissingEncoding("x".into())) };

    if let Some(scale_spec) = &x_enc.scale {
        return build_from_scale_spec(scale_spec, x_enc, batch);
    }
    // Existing auto-detection path follows...
    auto_detect_scale(x_enc, batch)
}

fn build_from_scale_spec(
    spec: &ScaleSpec,
    enc: &EncodingSpec,
    batch: &RecordBatch,
) -> Result<ScaleKind, RenderError> {
    let domain = compute_column_domain(&enc.field, batch)?;     // existing helper
    use crate::spec::encoding::ScaleSpec;
    Ok(match spec {
        ScaleSpec::Linear { domain: d, range, nice, zero, clamp } => {
            ScaleKind::Linear(crate::scale::LinearScale::new(
                d.clone().unwrap_or(domain.numeric()),
                range.clone().unwrap_or_else(default_pixel_range),
                *clamp, *nice,
            ))
        }
        ScaleSpec::Log { base, domain: d, range, nice, clamp } => {
            ScaleKind::Log(crate::scale::LogScale::new(
                d.clone().unwrap_or(domain.numeric()),
                range.clone().unwrap_or_else(default_pixel_range),
                *base, *clamp, *nice,
            ))
        }
        ScaleSpec::Time { domain: d, range, nice, clamp } => {
            ScaleKind::Time(crate::scale::TimeScale::new(
                d.clone().unwrap_or(domain.numeric()),
                range.clone().unwrap_or_else(default_pixel_range),
                *clamp, *nice,
            ))
        }
        ScaleSpec::Symlog { constant, domain: d, range, nice, clamp } => {
            ScaleKind::Symlog(crate::scale::SymlogScale::new(
                d.clone().unwrap_or(domain.numeric()),
                range.clone().unwrap_or_else(default_pixel_range),
                *constant, *clamp, *nice,
            ))
        }
        ScaleSpec::Ordinal { domain: d, range, padding } => {
            ScaleKind::Ordinal(crate::scale::OrdinalScale::new(
                d.clone().unwrap_or(domain.ordinal()),
                range.clone().unwrap_or_else(default_pixel_range),
                *padding,
            ))
        }
    })
}
```

Use `*_internal` constructors per Phase 7 §11 Phase-4-binding pattern (Phase 4 scale classes only expose `#[pymethods]`; Phase 7 added `pub(crate) *_internal` shims for Rust-side use).

- [ ] **Step 3: Build size/shape/opacity scales**

Add to the same module:

```rust
pub fn build_size_scale(encoding: &Encoding, batch: &RecordBatch, theme: &ThemeInputs)
    -> Result<Option<SizeScale>, RenderError>
{
    let Some(size_enc) = &encoding.size else { return Ok(None) };
    let domain = compute_column_domain(&size_enc.field, batch)?.numeric();
    let inner = ScaleKind::Linear(crate::scale::LinearScale::new_internal(
        domain, vec![theme.point_size_min, theme.point_size_max], false, true,
    ));
    Ok(Some(SizeScale {
        inner,
        min_px: theme.point_size_min,
        max_px: theme.point_size_max,
    }))
}

pub fn build_shape_scale(encoding: &Encoding, batch: &RecordBatch)
    -> Result<Option<ShapeScale>, RenderError>
{
    let Some(shape_enc) = &encoding.shape else { return Ok(None) };
    let distinct = compute_distinct_strings(&shape_enc.field, batch)?;   // existing or new helper
    let n = distinct.len().min(SHAPE_PALETTE.len());
    if distinct.len() > SHAPE_PALETTE.len() {
        // emit RenderWarning::ShapePaletteOverflowed { categories: distinct.len() as u32 }
    }
    let shapes: Vec<ShapeKind> = (0..distinct.len())
        .map(|i| SHAPE_PALETTE[i % SHAPE_PALETTE.len()])
        .collect();
    Ok(Some(ShapeScale { domain: distinct, shapes }))
}

pub fn build_opacity_scale(encoding: &Encoding, batch: &RecordBatch, theme: &ThemeInputs)
    -> Result<Option<OpacityScale>, RenderError>
{
    let Some(op_enc) = &encoding.opacity else { return Ok(None) };
    let domain = compute_column_domain(&op_enc.field, batch)?.numeric();
    let inner = ScaleKind::Linear(crate::scale::LinearScale::new_internal(
        domain, vec![theme.opacity_min, theme.opacity_max], true, false,
    ));
    Ok(Some(OpacityScale { inner, min_opacity: theme.opacity_min, max_opacity: theme.opacity_max }))
}
```

> **Encoding additions:** add `size: Option<EncodingSpec>`, `shape: Option<EncodingSpec>`, `opacity: Option<EncodingSpec>` fields to `Encoding` in `spec/encoding.rs` (mirror existing `color` field). All default-omit. This is a follow-up edit to Task 3 — add it now and rerun encoding tests.

> **ThemeInputs additions:** add `point_size_min: f64` (default 3.0), `point_size_max: f64` (default 30.0), `opacity_min: f64` (default 0.1), `opacity_max: f64` (default 1.0) to `ThemeInputs` in `crates/ferrum-core/src/layout/`. Mirror Phase 7's additive theme-field pattern. All default-resolved.

- [ ] **Step 4: Tests**

```rust
#[test]
fn explicit_log_scale_overrides_auto_detection() {
    let mut enc = encoding_with_x_quantitative();
    enc.x.as_mut().unwrap().scale = Some(ScaleSpec::Log {
        base: 10.0, domain: None, range: None, nice: false, clamp: false,
    });
    let scale = build_x_scale(&enc, &batch_with_quantitative_x()).unwrap();
    assert!(matches!(scale, ScaleKind::Log(_)));
}

#[test]
fn size_scale_defaults_to_3_to_30_px() {
    let enc = encoding_with_size_quantitative();
    let theme = ThemeInputs::default();
    let scale = build_size_scale(&enc, &batch(), &theme).unwrap().unwrap();
    assert_eq!(scale.min_px, 3.0);
    assert_eq!(scale.max_px, 30.0);
}

#[test]
fn shape_scale_picks_from_6_shape_palette_in_order() {
    let enc = encoding_with_shape_field("species");   // 3 distinct values
    let scale = build_shape_scale(&enc, &batch_with_3_species()).unwrap().unwrap();
    assert_eq!(scale.shapes.len(), 3);
    assert_eq!(scale.shapes[0], ShapeKind::Circle);
    assert_eq!(scale.shapes[1], ShapeKind::Square);
    assert_eq!(scale.shapes[2], ShapeKind::Cross);
}

#[test]
fn opacity_scale_defaults_to_0_1_to_1_0() {
    let enc = encoding_with_opacity_quantitative();
    let theme = ThemeInputs::default();
    let scale = build_opacity_scale(&enc, &batch(), &theme).unwrap().unwrap();
    assert_eq!(scale.min_opacity, 0.1);
    assert_eq!(scale.max_opacity, 1.0);
}
```

- [ ] **Step 5: Wire size/shape/opacity into ResolvedScales construction**

In the `resolve_scales` function (top-level), call the three new builders and assign into the new fields.

- [ ] **Step 6: Build + run**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core scale_resolve
```

Expected: 4+ new tests pass; existing scale_resolve tests still pass.

- [ ] **Step 7: Commit**

```bash
git add crates/ferrum-core/src/render/scale_resolve.rs \
        crates/ferrum-core/src/spec/encoding.rs \
        crates/ferrum-core/src/layout/
git commit -m "feat(render): scale_resolve honors explicit Scale + builds size/shape/opacity scales"
```

---

### Task 9: `marks/point.rs` — honor per-row size/shape/opacity

**Files:**
- Modify: `crates/ferrum-core/src/render/marks/point.rs`

Phase 7's `marks/point.rs::draw` emits `<circle>` per row using `ctx.mark_style.point_size`. Phase 8a extends to honor per-row size/shape/opacity from `ctx.scales.size/shape/opacity` when populated.

- [ ] **Step 1: Read current draw fn + extend signature consumption**

```bash
grep -n "fn draw\|<circle\|point_size" crates/ferrum-core/src/render/marks/point.rs
```

Note current circle-emission shape; Phase 8a needs to switch between `<circle>` (default + Circle shape) and other shape glyphs based on `ctx.scales.shape`.

- [ ] **Step 2: Add shape glyph helpers**

```rust
use crate::render::scale_resolve::ShapeKind;
use crate::render::svg::SvgBuffer;

/// Emit one shape glyph centered at (cx, cy) with given radius and style.
fn emit_shape(out: &mut SvgBuffer, kind: ShapeKind, cx: f64, cy: f64, r: f64,
              style: &crate::render::svg::FillStroke) {
    match kind {
        ShapeKind::Circle => out.circle(cx, cy, r, style),
        ShapeKind::Square => {
            let s = r * 1.6;  // visual area parity with circle
            out.rect(crate::layout::Rect { x: cx - s/2.0, y: cy - s/2.0, w: s, h: s },
                     style, None);
        }
        ShapeKind::Cross => {
            let arm = r * 0.5;
            // two stroked perpendicular lines
            let stroke = crate::render::svg::Stroke {
                stroke: style.fill.unwrap_or(crate::render::color::from_rgb(0, 0, 0)),
                stroke_width: r * 0.4,
                stroke_dash: None,
            };
            out.line(cx - arm, cy, cx + arm, cy, &stroke);
            out.line(cx, cy - arm, cx, cy + arm, &stroke);
        }
        ShapeKind::Diamond => {
            let d = r * 1.4;
            let path = format!("M {} {} L {} {} L {} {} L {} {} Z",
                cx, cy - d, cx + d, cy, cx, cy + d, cx - d, cy);
            out.path(&path, style);
        }
        ShapeKind::TriangleUp => {
            let h = r * 1.4;
            let path = format!("M {} {} L {} {} L {} {} Z",
                cx, cy - h, cx + h*0.866, cy + h*0.5, cx - h*0.866, cy + h*0.5);
            out.path(&path, style);
        }
        ShapeKind::TriangleDown => {
            let h = r * 1.4;
            let path = format!("M {} {} L {} {} L {} {} Z",
                cx, cy + h, cx + h*0.866, cy - h*0.5, cx - h*0.866, cy - h*0.5);
            out.path(&path, style);
        }
    }
}
```

- [ ] **Step 3: Extend `draw` to use per-row size/shape/opacity**

```rust
pub fn draw(ctx: &crate::render::draw::DrawCtx, out: &mut SvgBuffer) {
    let panel = ctx.panel;
    let batch = ctx.batch;
    let scales = ctx.scales;

    let x_col = batch.column_by_name(/* x field */).expect("x column present");
    let y_col = batch.column_by_name(/* y field */).expect("y column present");
    let size_col = scales.size.as_ref().and_then(|_| batch.column_by_name(/* size field */));
    let shape_col = scales.shape.as_ref().and_then(|_| batch.column_by_name(/* shape field */));
    let opacity_col = scales.opacity.as_ref().and_then(|_| batch.column_by_name(/* opacity field */));

    for i in 0..batch.num_rows() {
        let cx = panel.plot_area.x + scales.x.scale_value(read_f64(x_col, i));
        let cy = panel.plot_area.y + panel.plot_area.h - scales.y.scale_value(read_f64(y_col, i));

        let r = match (size_col, &scales.size) {
            (Some(c), Some(s)) => f64::sqrt(s.inner.scale_value(read_f64(c, i)) / std::f64::consts::PI),
            _ => f64::sqrt(ctx.mark_style.point_size / std::f64::consts::PI),
        };

        let shape_kind = match (shape_col, &scales.shape) {
            (Some(c), Some(s)) => {
                let val = read_string(c, i);
                let idx = s.domain.iter().position(|d| d == &val).unwrap_or(0);
                s.shapes[idx]
            }
            _ => ShapeKind::Circle,
        };

        let opacity = match (opacity_col, &scales.opacity) {
            (Some(c), Some(s)) => s.inner.scale_value(read_f64(c, i)),
            _ => ctx.mark_style.opacity,
        };

        let fill = crate::render::color::with_opacity(ctx.mark_style.fill, opacity);
        let style = crate::render::svg::FillStroke {
            fill: Some(fill),
            stroke: ctx.mark_style.stroke,
            stroke_width: ctx.mark_style.stroke_width,
        };
        emit_shape(out, shape_kind, cx, cy, r, &style);
    }
}
```

> **Helpers:** `read_f64(col, i)` and `read_string(col, i)` likely already exist in the render module from Phase 7. If not, add them as small `pub(crate)` helpers in `prepare.rs` or `draw.rs`.

- [ ] **Step 4: Tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_with_size_encoding_scales_radius_per_row() {
        let ctx = test_draw_ctx_with_size_encoding();   // helper: batch with size column [10, 20, 30]
        let mut buf = SvgBuffer::new(test_viewport(), None, false);
        draw(&ctx, &mut buf);
        let svg = buf.finish();
        // Three circles with r values that scale linearly across the size encoding
        assert_eq!(svg.matches("<circle").count(), 3);
        // Specific radii depend on theme.point_size_min/max; assert ordering only
        let radii = extract_circle_radii(&svg);
        assert!(radii[0] < radii[1] && radii[1] < radii[2]);
    }

    #[test]
    fn point_with_shape_encoding_emits_3_shape_kinds() {
        let ctx = test_draw_ctx_with_shape_encoding();   // 3 distinct shape values
        let mut buf = SvgBuffer::new(test_viewport(), None, false);
        draw(&ctx, &mut buf);
        let svg = buf.finish();
        assert_eq!(svg.matches("<circle").count(), 1);   // first species
        assert_eq!(svg.matches("<rect").count(), 1);     // second species (square)
        // Cross uses two <line> elements; diamond/triangle use <path>
    }

    #[test]
    fn point_with_opacity_encoding_sets_fill_opacity_per_row() {
        let ctx = test_draw_ctx_with_opacity_encoding();
        let mut buf = SvgBuffer::new(test_viewport(), None, false);
        draw(&ctx, &mut buf);
        let svg = buf.finish();
        // Each circle gets a different rgba(...) fill (opacity baked in)
        assert!(svg.contains("rgba("));
    }
}
```

- [ ] **Step 5: Build + run**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core marks::point
```

Expected: 3 new point tests pass; existing point tests pass; existing 6 SVG goldens still match (no encoding for size/shape/opacity = Circle defaults).

- [ ] **Step 6: Commit**

```bash
git add crates/ferrum-core/src/render/marks/point.rs
git commit -m "feat(render): mark_point honors per-row size/shape/opacity from resolved scales"
```

---

### Task 10: `palette.rs` — 6 more categorical palettes + scheme-name lookup

**Files:**
- Modify: `crates/ferrum-core/src/render/palette.rs`

Phase 7 has only OKABE_ITO. Phase 8a adds `tableau10`, `set1`, `set2`, `paired`, `pastel`, `dark2` (named in spec §3.6) plus a lookup function so `theme.color_scheme` (a string from Python) selects the palette.

- [ ] **Step 1: Define the 6 new palettes**

In `palette.rs`, add (after OKABE_ITO):

```rust
pub const TABLEAU10: &[Color; 10] = &[
    from_rgb(0x4C, 0x78, 0xA8), from_rgb(0xF5, 0x8E, 0x18),
    from_rgb(0xE4, 0x57, 0x56), from_rgb(0x72, 0xB7, 0xB2),
    from_rgb(0x54, 0xA2, 0x4B), from_rgb(0xEE, 0xCA, 0x3B),
    from_rgb(0xB2, 0x79, 0xA2), from_rgb(0xFF, 0x9D, 0xA6),
    from_rgb(0x9D, 0x75, 0x5D), from_rgb(0xBA, 0xB0, 0xAC),
];

pub const SET1: &[Color; 9] = &[
    from_rgb(0xE4, 0x1A, 0x1C), from_rgb(0x37, 0x7E, 0xB8),
    from_rgb(0x4D, 0xAF, 0x4A), from_rgb(0x98, 0x4E, 0xA3),
    from_rgb(0xFF, 0x7F, 0x00), from_rgb(0xFF, 0xFF, 0x33),
    from_rgb(0xA6, 0x56, 0x28), from_rgb(0xF7, 0x81, 0xBF),
    from_rgb(0x99, 0x99, 0x99),
];

pub const SET2: &[Color; 8] = &[
    from_rgb(0x66, 0xC2, 0xA5), from_rgb(0xFC, 0x8D, 0x62),
    from_rgb(0x8D, 0xA0, 0xCB), from_rgb(0xE7, 0x8A, 0xC3),
    from_rgb(0xA6, 0xD8, 0x54), from_rgb(0xFF, 0xD9, 0x2F),
    from_rgb(0xE5, 0xC4, 0x94), from_rgb(0xB3, 0xB3, 0xB3),
];

pub const PAIRED: &[Color; 12] = &[
    from_rgb(0xA6, 0xCE, 0xE3), from_rgb(0x1F, 0x78, 0xB4),
    from_rgb(0xB2, 0xDF, 0x8A), from_rgb(0x33, 0xA0, 0x2C),
    from_rgb(0xFB, 0x9A, 0x99), from_rgb(0xE3, 0x1A, 0x1C),
    from_rgb(0xFD, 0xBF, 0x6F), from_rgb(0xFF, 0x7F, 0x00),
    from_rgb(0xCA, 0xB2, 0xD6), from_rgb(0x6A, 0x3D, 0x9A),
    from_rgb(0xFF, 0xFF, 0x99), from_rgb(0xB1, 0x59, 0x28),
];

pub const PASTEL: &[Color; 9] = &[
    from_rgb(0xFB, 0xB4, 0xAE), from_rgb(0xB3, 0xCD, 0xE3),
    from_rgb(0xCC, 0xEB, 0xC5), from_rgb(0xDE, 0xCB, 0xE4),
    from_rgb(0xFE, 0xD9, 0xA6), from_rgb(0xFF, 0xFF, 0xCC),
    from_rgb(0xE5, 0xD8, 0xBD), from_rgb(0xFD, 0xDA, 0xEC),
    from_rgb(0xF2, 0xF2, 0xF2),
];

pub const DARK2: &[Color; 8] = &[
    from_rgb(0x1B, 0x9E, 0x77), from_rgb(0xD9, 0x5F, 0x02),
    from_rgb(0x75, 0x70, 0xB3), from_rgb(0xE7, 0x29, 0x8A),
    from_rgb(0x66, 0xA6, 0x1E), from_rgb(0xE6, 0xAB, 0x02),
    from_rgb(0xA6, 0x76, 0x1D), from_rgb(0x66, 0x66, 0x66),
];

/// Look up a categorical palette by scheme name. Returns OKABE_ITO when the
/// name is unknown (caller may emit a warning).
pub fn categorical_palette(name: &str) -> &'static [Color] {
    match name {
        "okabe_ito"  => OKABE_ITO,
        "tableau10"  => TABLEAU10,
        "set1"       => SET1,
        "set2"       => SET2,
        "paired"     => PAIRED,
        "pastel"     => PASTEL,
        "dark2"      => DARK2,
        _            => OKABE_ITO,   // fallback
    }
}
```

- [ ] **Step 2: Tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn categorical_palette_returns_named_palette() {
        assert!(std::ptr::eq(categorical_palette("tableau10").as_ptr(), TABLEAU10.as_ptr()));
        assert!(std::ptr::eq(categorical_palette("set1").as_ptr(), SET1.as_ptr()));
        assert!(std::ptr::eq(categorical_palette("dark2").as_ptr(), DARK2.as_ptr()));
    }

    #[test]
    fn categorical_palette_unknown_falls_back_to_okabe_ito() {
        assert!(std::ptr::eq(categorical_palette("nonexistent").as_ptr(), OKABE_ITO.as_ptr()));
    }

    #[test]
    fn each_named_palette_has_at_least_8_colors() {
        for name in &["okabe_ito", "tableau10", "set1", "set2", "paired", "pastel", "dark2"] {
            assert!(categorical_palette(name).len() >= 8, "{name} has < 8 colors");
        }
    }
}
```

- [ ] **Step 3: Wire scheme-name into ColorScale construction in scale_resolve**

In `scale_resolve.rs::build_color_scale` (or equivalent), check `EncodingSpec.scheme: Option<String>` and call `categorical_palette(name)` instead of the hardcoded OKABE_ITO. Phase 7 hardcoded OKABE_ITO as the only choice — Phase 8a respects scheme overrides.

```rust
let palette: &'static [Color] = match &color_enc.scheme {
    Some(name) => crate::render::palette::categorical_palette(name),
    None => crate::render::palette::OKABE_ITO,
};
```

- [ ] **Step 4: Build + run**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core palette
```

Expected: 3 new tests pass; existing palette tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/ferrum-core/src/render/palette.rs crates/ferrum-core/src/render/scale_resolve.rs
git commit -m "feat(render): add 6 categorical palettes + scheme-name lookup"
```

---

## Group C — Rust SVG compositor

### Task 11: `render/compositor.rs` + PyO3 binding

**Files:**
- Create: `crates/ferrum-core/src/render/compositor.rs`
- Modify: `crates/ferrum-core/src/render/mod.rs` (re-export)
- Modify: `crates/ferrum-core/src/render/binding.rs` (add PyO3 entry points)

The compositor parses our deterministic-output SVGs (per Phase 7 §4.4 the `<svg>` root has a fixed attribute order) and stitches them inside a wrapping `<svg>` with `<g transform="translate(...)">` per child. Because our `SvgBuffer` always emits `<svg xmlns="..." width="W" height="H" viewBox="0 0 W H">` in that exact order, we can extract W/H by regex without a real XML parser.

- [ ] **Step 1: Create compositor.rs skeleton**

```rust
use crate::render::svg::escape_attr_value;

/// Parse the width and height attributes from an SVG root element.
/// Returns (width, height, body_start, body_end) where body is the content
/// between the opening <svg ...> tag and the closing </svg>.
fn parse_svg_root(svg: &str) -> Result<ParsedSvg<'_>, CompositorError> {
    // Find <svg ... > ... </svg>
    let svg_open_start = svg.find("<svg").ok_or(CompositorError::NoSvgRoot)?;
    let svg_open_end = svg[svg_open_start..].find('>').ok_or(CompositorError::MalformedRoot)?
        + svg_open_start + 1;
    let svg_close = svg.rfind("</svg>").ok_or(CompositorError::NoClosingTag)?;

    let attrs = &svg[svg_open_start..svg_open_end];
    let width = extract_attr_f64(attrs, "width")?;
    let height = extract_attr_f64(attrs, "height")?;

    Ok(ParsedSvg {
        width, height,
        body: &svg[svg_open_end..svg_close],
    })
}

struct ParsedSvg<'a> {
    width: f64,
    height: f64,
    body: &'a str,
}

fn extract_attr_f64(attrs: &str, name: &str) -> Result<f64, CompositorError> {
    let needle = format!(r#"{}=""#, name);
    let start = attrs.find(&needle).ok_or(CompositorError::MissingAttr(name.into()))?;
    let val_start = start + needle.len();
    let val_end = attrs[val_start..].find('"').ok_or(CompositorError::MalformedAttr(name.into()))?
        + val_start;
    attrs[val_start..val_end].parse::<f64>()
        .map_err(|_| CompositorError::AttrNotNumeric(name.into()))
}

#[derive(Debug, Clone, PartialEq)]
pub enum CompositorError {
    NoSvgRoot,
    MalformedRoot,
    NoClosingTag,
    MissingAttr(String),
    MalformedAttr(String),
    AttrNotNumeric(String),
    EmptyInput,
}

impl std::fmt::Display for CompositorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompositorError::NoSvgRoot => write!(f, "input does not contain <svg ...>"),
            CompositorError::MalformedRoot => write!(f, "<svg> open tag is malformed"),
            CompositorError::NoClosingTag => write!(f, "input does not contain </svg>"),
            CompositorError::MissingAttr(n) => write!(f, "<svg> missing required attr '{n}'"),
            CompositorError::MalformedAttr(n) => write!(f, "<svg> attr '{n}' is malformed"),
            CompositorError::AttrNotNumeric(n) => write!(f, "<svg> attr '{n}' is not numeric"),
            CompositorError::EmptyInput => write!(f, "no svgs to compose"),
        }
    }
}
impl std::error::Error for CompositorError {}

const FONT_DEFS_MARKER: &str = "<defs><style>@font-face";
```

- [ ] **Step 2: Write `compose_svg_horizontal`**

```rust
pub fn compose_svg_horizontal(
    svgs: &[String],
    spacing: f64,
    align: VerticalAlign,
) -> Result<String, CompositorError> {
    if svgs.is_empty() {
        return Err(CompositorError::EmptyInput);
    }
    let parsed: Vec<ParsedSvg> = svgs.iter().map(|s| parse_svg_root(s)).collect::<Result<_, _>>()?;

    let total_width: f64 = parsed.iter().map(|p| p.width).sum::<f64>()
        + spacing * (svgs.len() - 1) as f64;
    let max_height: f64 = parsed.iter().map(|p| p.height).fold(0.0_f64, f64::max);

    let mut out = String::with_capacity(svgs.iter().map(|s| s.len()).sum::<usize>() + 256);
    out.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
        format_f64(total_width), format_f64(max_height),
        format_f64(total_width), format_f64(max_height),
    ));

    let mut x_offset = 0.0_f64;
    for (i, p) in parsed.iter().enumerate() {
        let y_offset = match align {
            VerticalAlign::Top => 0.0,
            VerticalAlign::Center => (max_height - p.height) / 2.0,
            VerticalAlign::Bottom => max_height - p.height,
        };
        out.push_str(&format!(
            r#"<g transform="translate({},{})">"#,
            format_f64(x_offset), format_f64(y_offset),
        ));
        // Strip the <defs><style>@font-face ...</style></defs> from all but the first
        let body = if i == 0 { p.body } else { strip_font_defs(p.body) };
        out.push_str(body);
        out.push_str("</g>");
        x_offset += p.width + spacing;
    }
    out.push_str("</svg>");
    Ok(out)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VerticalAlign { Top, Center, Bottom }

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HorizontalAlign { Left, Center, Right }

fn strip_font_defs(body: &str) -> &str {
    let Some(start) = body.find(FONT_DEFS_MARKER) else { return body };
    // Walk back to find <defs and forward to find </defs>
    let defs_start = body[..start].rfind("<defs").unwrap_or(start);
    let defs_end_marker = "</defs>";
    let Some(end_rel) = body[defs_start..].find(defs_end_marker) else { return body };
    let defs_end = defs_start + end_rel + defs_end_marker.len();
    // Return body with the defs slice removed (use a Cow if you need to splice in the middle)
    // Simpler: return a new String — but this fn returns &str. Switch to returning String.
    panic!("strip_font_defs needs to return String (see impl below)")
}
```

> Replace `strip_font_defs(&str) -> &str` with `String`-returning version (since splicing requires allocation):
> ```rust
> fn strip_font_defs(body: &str) -> String {
>     let Some(start) = body.find(FONT_DEFS_MARKER) else { return body.to_string() };
>     let defs_start = body[..start].rfind("<defs").unwrap_or(start);
>     let defs_end_marker = "</defs>";
>     let Some(end_rel) = body[defs_start..].find(defs_end_marker) else { return body.to_string() };
>     let defs_end = defs_start + end_rel + defs_end_marker.len();
>     let mut out = String::with_capacity(body.len() - (defs_end - defs_start));
>     out.push_str(&body[..defs_start]);
>     out.push_str(&body[defs_end..]);
>     out
> }
> ```
> And in `compose_svg_horizontal`'s loop:
> ```rust
> let body_owned;
> let body: &str = if i == 0 { p.body } else { body_owned = strip_font_defs(p.body); &body_owned };
> ```

- [ ] **Step 3: Write `compose_svg_vertical` (analogous)**

```rust
pub fn compose_svg_vertical(
    svgs: &[String],
    spacing: f64,
    align: HorizontalAlign,
) -> Result<String, CompositorError> {
    if svgs.is_empty() { return Err(CompositorError::EmptyInput); }
    let parsed: Vec<ParsedSvg> = svgs.iter().map(|s| parse_svg_root(s)).collect::<Result<_, _>>()?;

    let max_width: f64 = parsed.iter().map(|p| p.width).fold(0.0_f64, f64::max);
    let total_height: f64 = parsed.iter().map(|p| p.height).sum::<f64>()
        + spacing * (svgs.len() - 1) as f64;

    let mut out = String::with_capacity(svgs.iter().map(|s| s.len()).sum::<usize>() + 256);
    out.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
        format_f64(max_width), format_f64(total_height),
        format_f64(max_width), format_f64(total_height),
    ));

    let mut y_offset = 0.0_f64;
    for (i, p) in parsed.iter().enumerate() {
        let x_offset = match align {
            HorizontalAlign::Left => 0.0,
            HorizontalAlign::Center => (max_width - p.width) / 2.0,
            HorizontalAlign::Right => max_width - p.width,
        };
        out.push_str(&format!(
            r#"<g transform="translate({},{})">"#,
            format_f64(x_offset), format_f64(y_offset),
        ));
        let body_owned;
        let body: &str = if i == 0 { p.body } else { body_owned = strip_font_defs(p.body); &body_owned };
        out.push_str(body);
        out.push_str("</g>");
        y_offset += p.height + spacing;
    }
    out.push_str("</svg>");
    Ok(out)
}
```

> `format_f64` is the deterministic float formatter from Phase 7 `svg.rs`. Re-export it as `pub(crate)` if necessary, or duplicate the 6-line implementation here.

- [ ] **Step 4: Tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn make_svg(w: f64, h: f64, body: &str) -> String {
        format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">{}</svg>"#,
            w, h, w, h, body,
        )
    }

    #[test]
    fn parse_svg_root_extracts_dimensions() {
        let s = make_svg(100.0, 50.0, "<rect/>");
        let parsed = parse_svg_root(&s).unwrap();
        assert_eq!(parsed.width, 100.0);
        assert_eq!(parsed.height, 50.0);
        assert_eq!(parsed.body, "<rect/>");
    }

    #[test]
    fn hconcat_two_svgs_sums_widths_plus_spacing() {
        let a = make_svg(100.0, 50.0, "<circle/>");
        let b = make_svg(80.0, 60.0, "<rect/>");
        let out = compose_svg_horizontal(&[a, b], 10.0, VerticalAlign::Top).unwrap();
        // Total width = 100 + 10 + 80 = 190; max height = 60
        assert!(out.contains(r#"width="190""#));
        assert!(out.contains(r#"height="60""#));
    }

    #[test]
    fn hconcat_wraps_each_child_in_translate_g() {
        let a = make_svg(100.0, 50.0, "<circle/>");
        let b = make_svg(80.0, 60.0, "<rect/>");
        let out = compose_svg_horizontal(&[a, b], 10.0, VerticalAlign::Top).unwrap();
        assert!(out.contains(r#"<g transform="translate(0,0)">"#));
        assert!(out.contains(r#"<g transform="translate(110,0)">"#));
    }

    #[test]
    fn vconcat_two_svgs_sums_heights() {
        let a = make_svg(100.0, 50.0, "<circle/>");
        let b = make_svg(80.0, 60.0, "<rect/>");
        let out = compose_svg_vertical(&[a, b], 10.0, HorizontalAlign::Left).unwrap();
        // Total height = 50 + 10 + 60 = 120; max width = 100
        assert!(out.contains(r#"width="100""#));
        assert!(out.contains(r#"height="120""#));
    }

    #[test]
    fn font_defs_stripped_from_second_child_only() {
        let a = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50" viewBox="0 0 100 50"><defs><style>@font-face{{src:url("data:fake")}}</style></defs><circle/></svg>"#,
        );
        let b = a.clone();
        let out = compose_svg_horizontal(&[a, b], 0.0, VerticalAlign::Top).unwrap();
        // Exactly one occurrence of @font-face in the composed output
        assert_eq!(out.matches("@font-face").count(), 1);
    }

    #[test]
    fn compose_three_svgs_hconcat() {
        let a = make_svg(50.0, 50.0, "<circle/>");
        let b = make_svg(50.0, 50.0, "<rect/>");
        let c = make_svg(50.0, 50.0, "<line/>");
        let out = compose_svg_horizontal(&[a, b, c], 5.0, VerticalAlign::Top).unwrap();
        // Total = 50 + 5 + 50 + 5 + 50 = 160
        assert!(out.contains(r#"width="160""#));
    }

    #[test]
    fn compose_empty_returns_error() {
        assert_eq!(
            compose_svg_horizontal(&[], 0.0, VerticalAlign::Top),
            Err(CompositorError::EmptyInput),
        );
    }
}
```

- [ ] **Step 5: Re-export from render/mod.rs**

In `crates/ferrum-core/src/render/mod.rs`:

```rust
pub mod compositor;
pub use compositor::{compose_svg_horizontal, compose_svg_vertical, VerticalAlign, HorizontalAlign, CompositorError};
```

- [ ] **Step 6: Add PyO3 binding**

In `crates/ferrum-core/src/render/binding.rs` (or wherever `render_svg`/`render_png` are bound):

```rust
#[pyfunction]
#[pyo3(signature = (svgs, *, spacing = 10.0, align = "top"))]
pub fn compose_svg_horizontal_py(
    svgs: Vec<String>,
    spacing: f64,
    align: &str,
) -> PyResult<String> {
    let align = match align {
        "top" => crate::render::compositor::VerticalAlign::Top,
        "center" => crate::render::compositor::VerticalAlign::Center,
        "bottom" => crate::render::compositor::VerticalAlign::Bottom,
        other => return Err(pyo3::exceptions::PyValueError::new_err(
            format!("align must be one of 'top'|'center'|'bottom', got '{other}'")
        )),
    };
    crate::render::compositor::compose_svg_horizontal(&svgs, spacing, align)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (svgs, *, spacing = 10.0, align = "left"))]
pub fn compose_svg_vertical_py(
    svgs: Vec<String>,
    spacing: f64,
    align: &str,
) -> PyResult<String> {
    let align = match align {
        "left" => crate::render::compositor::HorizontalAlign::Left,
        "center" => crate::render::compositor::HorizontalAlign::Center,
        "right" => crate::render::compositor::HorizontalAlign::Right,
        other => return Err(pyo3::exceptions::PyValueError::new_err(
            format!("align must be one of 'left'|'center'|'right', got '{other}'")
        )),
    };
    crate::render::compositor::compose_svg_vertical(&svgs, spacing, align)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}
```

In the module-init function (`#[pymodule] fn _core(...)`), add:

```rust
m.add_function(wrap_pyfunction!(compose_svg_horizontal_py, m)?)?;
m.add_function(wrap_pyfunction!(compose_svg_vertical_py, m)?)?;
```

> Also rename Python-facing names in the binding macro: use `#[pyo3(name = "compose_svg_horizontal")]` so Python sees `ferrum._core.compose_svg_horizontal(...)` (without the `_py` suffix).

- [ ] **Step 7: Add stubs to `_core.pyi`**

```python
def compose_svg_horizontal(svgs: list[str], *, spacing: float = 10.0,
                           align: Literal["top", "center", "bottom"] = "top") -> str: ...
def compose_svg_vertical(svgs: list[str], *, spacing: float = 10.0,
                         align: Literal["left", "center", "right"] = "left") -> str: ...
```

- [ ] **Step 8: Build + run**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core compositor
uv run python -c "from ferrum._core import compose_svg_horizontal; print(compose_svg_horizontal(['<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"50\" height=\"50\" viewBox=\"0 0 50 50\"></svg>'], spacing=0.0))"
```

Expected: 7+ compositor tests pass; Python smoke command prints a composed `<svg>` string.

- [ ] **Step 9: Commit**

```bash
git add crates/ferrum-core/src/render/compositor.rs \
        crates/ferrum-core/src/render/mod.rs \
        crates/ferrum-core/src/render/binding.rs \
        src/ferrum/_core.pyi
git commit -m "feat(render): SVG compositor for hconcat/vconcat + PyO3 binding"
```

---

## Group D — Python utilities

### Task 12: `_coerce.py` — narwhals + ferrum branches

**Files:**
- Create: `src/ferrum/_coerce.py`
- Test: `tests/test_coerce.py`

- [ ] **Step 1: Write failing tests first**

Create `tests/test_coerce.py`:

```python
import numpy as np
import polars as pl
import pyarrow as pa
import pytest

from ferrum._coerce import to_arrow_table


def test_polars_dataframe_zero_copy():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    tbl = to_arrow_table(df)
    assert isinstance(tbl, pa.Table)
    assert tbl.num_rows == 3
    assert tbl.column_names == ["a", "b"]


def test_pyarrow_table_passthrough():
    tbl_in = pa.table({"x": [1, 2], "y": ["a", "b"]})
    tbl_out = to_arrow_table(tbl_in)
    assert tbl_out is tbl_in


def test_pyarrow_recordbatch_converted_to_table():
    rb = pa.RecordBatch.from_pylist([{"a": 1}, {"a": 2}])
    tbl = to_arrow_table(rb)
    assert isinstance(tbl, pa.Table)
    assert tbl.num_rows == 2


def test_dict_of_arrays():
    tbl = to_arrow_table({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert isinstance(tbl, pa.Table)
    assert tbl.num_rows == 3
    assert tbl.column_names == ["a", "b"]


def test_list_of_records():
    tbl = to_arrow_table([{"a": 1, "b": 4}, {"a": 2, "b": 5}])
    assert isinstance(tbl, pa.Table)
    assert tbl.num_rows == 2


def test_numpy_2d_with_auto_column_names():
    arr = np.array([[1, 2], [3, 4], [5, 6]])
    tbl = to_arrow_table(arr)
    assert isinstance(tbl, pa.Table)
    assert tbl.column_names == ["col_0", "col_1"]
    assert tbl.num_rows == 3


def test_numpy_1d_raises_clear_typeerror():
    arr = np.array([1, 2, 3])
    with pytest.raises(TypeError, match="1D numpy arrays need column names"):
        to_arrow_table(arr)


def test_none_raises_value_error():
    with pytest.raises(ValueError, match="per-layer data"):
        to_arrow_table(None)


def test_pandas_via_narwhals():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    tbl = to_arrow_table(df)
    assert isinstance(tbl, pa.Table)
    assert tbl.num_rows == 3


def test_unsupported_type_raises_clear_typeerror():
    class WeirdData:
        pass
    with pytest.raises(TypeError, match="Unsupported data type"):
        to_arrow_table(WeirdData())
```

Run: `uv run pytest tests/test_coerce.py -v`
Expected: ALL FAIL (`_coerce` module doesn't exist).

- [ ] **Step 2: Implement `_coerce.py`**

Create `src/ferrum/_coerce.py`:

```python
"""Data ingestion: normalize any supported input to a pyarrow.Table.

Supports (per spec §3.18):
- polars.DataFrame, polars.LazyFrame
- pyarrow.Table, pyarrow.RecordBatch
- pandas + modin + cuDF + dask + ibis (via narwhals)
- dict[str, list], list[dict]
- numpy.ndarray (2D, auto-named "col_0", "col_1", ...)

Raises TypeError for unsupported types or numpy 1D without column names.
"""
from __future__ import annotations

from typing import Any


def to_arrow_table(data: Any) -> "pyarrow.Table":
    """Normalize any supported input to a pyarrow.Table.

    Raises:
        ValueError: if data is None.
        TypeError: if input is numpy 1D, or an unsupported type.
        ImportError: if narwhals is required for the input type but not installed.
    """
    import pyarrow as pa

    if data is None:
        raise ValueError(
            "Chart(data=None) requires per-layer data — not yet supported in Phase 8a"
        )

    # Fast path: polars (zero-copy via Arrow C Data Interface)
    try:
        import polars as pl
        if isinstance(data, pl.DataFrame):
            return data.to_arrow()
        if isinstance(data, pl.LazyFrame):
            return data.collect().to_arrow()
    except ImportError:
        pass

    # Fast path: pyarrow native
    if isinstance(data, pa.Table):
        return data
    if isinstance(data, pa.RecordBatch):
        return pa.Table.from_batches([data])

    # Direct conversions: dict, list, numpy
    if isinstance(data, dict):
        return pa.Table.from_pydict(data)
    if isinstance(data, list):
        if not data:
            raise ValueError("Cannot construct Chart from empty list")
        if not isinstance(data[0], dict):
            raise TypeError(
                f"Chart(list) expects a list of dicts (one per row), got list of "
                f"{type(data[0]).__name__}"
            )
        return pa.Table.from_pylist(data)

    # numpy
    try:
        import numpy as np
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                raise TypeError(
                    "1D numpy arrays need column names — pass `Chart({'value': arr})` "
                    "or `Chart(arr.reshape(-1, 1), columns=['value'])`."
                )
            if data.ndim == 2:
                cols = [f"col_{i}" for i in range(data.shape[1])]
                return pa.table({cols[i]: data[:, i] for i in range(data.shape[1])})
            raise TypeError(f"numpy arrays with ndim={data.ndim} not supported (use 2D)")
    except ImportError:
        pass

    # Everything else: try narwhals
    try:
        import narwhals as nw
    except ImportError as e:
        raise ImportError(
            f"Input type {type(data).__name__} requires narwhals. "
            f"Install with `pip install narwhals` (or use polars/pyarrow directly)."
        ) from e

    try:
        nw_df = nw.from_native(data, eager_only=True)
        return nw_df.to_arrow()
    except (TypeError, NotImplementedError) as e:
        raise TypeError(
            f"Unsupported data type: {type(data).__name__}. "
            f"Supported: polars, pyarrow, pandas, modin, cuDF, dask, ibis, dict, list, numpy 2D. "
            f"Underlying error: {e}"
        ) from e
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_coerce.py -v`
Expected: 9 PASS (skip pandas test if pandas not installed).

- [ ] **Step 4: Commit**

```bash
git add src/ferrum/_coerce.py tests/test_coerce.py
git commit -m "feat(coerce): to_arrow_table with narwhals + ferrum branches"
```

---

### Task 13: `_shorthand.py` — parse encoding-string shorthands

**Files:**
- Create: `src/ferrum/_shorthand.py`
- Test: `tests/test_shorthand.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_shorthand.py`:

```python
from ferrum._shorthand import parse_shorthand


def test_bare_field():
    assert parse_shorthand("price") == ("price", None, None)


def test_field_with_type():
    assert parse_shorthand("price:Q") == ("price", "Q", None)
    assert parse_shorthand("year:T") == ("year", "T", None)
    assert parse_shorthand("species:N") == ("species", "N", None)
    assert parse_shorthand("rank:O") == ("rank", "O", None)


def test_aggregate_with_field():
    assert parse_shorthand("mean(price)") == ("price", None, "mean")
    assert parse_shorthand("median(latency)") == ("latency", None, "median")
    assert parse_shorthand("q50(latency)") == ("latency", None, "q50")


def test_aggregate_without_field():
    assert parse_shorthand("count()") == (None, None, "count")


def test_aggregate_with_field_and_type():
    assert parse_shorthand("mean(price):Q") == ("price", "Q", "mean")
    assert parse_shorthand("count():Q") == (None, "Q", "count")


def test_field_name_with_underscores_and_digits():
    assert parse_shorthand("col_42") == ("col_42", None, None)
    assert parse_shorthand("mean(col_42):Q") == ("col_42", "Q", "mean")


def test_invalid_type_raises():
    import pytest
    with pytest.raises(ValueError, match="unknown type"):
        parse_shorthand("price:Z")


def test_unbalanced_parens_raises():
    import pytest
    with pytest.raises(ValueError, match="unbalanced"):
        parse_shorthand("mean(price")
```

Run: `uv run pytest tests/test_shorthand.py -v`
Expected: ALL FAIL.

- [ ] **Step 2: Implement `_shorthand.py`**

```python
"""Encoding-string shorthand parser.

Supports (per spec §3.2 Channel shorthand strings):
- "fieldname"            → (fieldname, None, None)
- "fieldname:Q"          → (fieldname, "Q", None)
- "agg(fieldname)"       → (fieldname, None, "agg")
- "agg()"                → (None, None, "agg")  (e.g. count())
- "agg(fieldname):Q"     → (fieldname, "Q", "agg")
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

_VALID_TYPES = frozenset(["Q", "N", "O", "T"])
_PATTERN = re.compile(
    r"""
    ^                                       # start
    (?:                                     # optional aggregate prefix:
        (?P<agg>[a-z][a-z0-9_]*)            #   agg name (lowercase identifier)
        \(                                  #   open paren
        (?P<aggfield>[a-zA-Z_][a-zA-Z0-9_]*)?  # optional field inside parens
        \)                                  #   close paren
    )?
    (?(agg)|(?P<field>[a-zA-Z_][a-zA-Z0-9_]*))  # if no agg, require bare field
    (?::(?P<type>[A-Z]))?                   # optional type suffix
    $                                       # end
    """,
    re.VERBOSE,
)


def parse_shorthand(s: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse a shorthand encoding string into (field, type, aggregate).

    Returns (None, type, agg) for "count()".
    Raises ValueError for malformed input or unknown type letters.
    """
    if "(" in s and ")" not in s:
        raise ValueError(f"unbalanced parens in shorthand: {s!r}")
    if ")" in s and "(" not in s:
        raise ValueError(f"unbalanced parens in shorthand: {s!r}")

    m = _PATTERN.match(s)
    if not m:
        raise ValueError(f"could not parse shorthand: {s!r}")

    type_ = m.group("type")
    if type_ is not None and type_ not in _VALID_TYPES:
        raise ValueError(
            f"unknown type {type_!r} in {s!r}; expected one of Q, N, O, T"
        )

    agg = m.group("agg")
    if agg is not None:
        return (m.group("aggfield"), type_, agg)
    return (m.group("field"), type_, None)
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_shorthand.py -v`
Expected: 9 PASS.

- [ ] **Step 4: Commit**

```bash
git add src/ferrum/_shorthand.py tests/test_shorthand.py
git commit -m "feat(shorthand): parse encoding-string shorthands (field, type, aggregate)"
```

---

### Task 14: `_warn.py` — warn-once registry

**Files:**
- Create: `src/ferrum/_warn.py`
- Test: `tests/test_warn.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_warn.py`:

```python
import warnings

import pytest

from ferrum._warn import warn_once, reset_warnings


def test_first_call_emits_warning():
    reset_warnings()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_once("X", "axis", "X(axis=...) is deferred")
    assert len(w) == 1
    assert issubclass(w[0].category, UserWarning)
    assert "axis" in str(w[0].message)


def test_repeated_calls_only_warn_once():
    reset_warnings()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_once("X", "axis", "first")
        warn_once("X", "axis", "second")
        warn_once("X", "axis", "third")
    assert len(w) == 1
    assert "first" in str(w[0].message)


def test_distinct_keys_each_warn():
    reset_warnings()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_once("X", "axis")
        warn_once("X", "legend")
        warn_once("Y", "axis")
    assert len(w) == 3


def test_default_message_when_none_provided():
    reset_warnings()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_once("X", "axis")
    assert "X(axis=...)" in str(w[0].message)
    assert "Phase" in str(w[0].message)


def test_reset_warnings_allows_re_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_once("X", "axis")
        reset_warnings()
        warn_once("X", "axis")
    assert len(w) == 2
```

Run: `uv run pytest tests/test_warn.py -v`
Expected: ALL FAIL.

- [ ] **Step 2: Implement `_warn.py`**

```python
"""Warn-once registry: each (channel, kwarg) tuple emits at most one
UserWarning per process. Tests use reset_warnings() to clear state.
"""
from __future__ import annotations

import warnings
from typing import Optional

_seen: set[tuple[str, str]] = set()


def warn_once(channel: str, kwarg: str, message: Optional[str] = None) -> None:
    """Emit a UserWarning the first time this (channel, kwarg) pair is seen.
    Subsequent calls with the same key are silent.
    """
    key = (channel, kwarg)
    if key in _seen:
        return
    _seen.add(key)
    msg = message or (
        f"{channel}({kwarg}=...) is accepted but not honored in Phase 8a; "
        f"planned for Phase 9."
    )
    warnings.warn(msg, UserWarning, stacklevel=3)


def reset_warnings() -> None:
    """Clear the warn-once registry. For tests."""
    _seen.clear()
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_warn.py -v`
Expected: 5 PASS.

- [ ] **Step 4: Commit**

```bash
git add src/ferrum/_warn.py tests/test_warn.py
git commit -m "feat(warn): warn-once registry keyed by (channel, kwarg)"
```

---

## Group E — Python encoding channels

### Task 15: `encoding/base.py` — `ChannelBase`

**Files:**
- Create: `src/ferrum/encoding/__init__.py`
- Create: `src/ferrum/encoding/base.py`
- Test: `tests/test_encoding.py` (initial setup)

- [ ] **Step 1: Write failing tests for ChannelBase contract**

Create `tests/test_encoding.py`:

```python
import warnings
import pytest

from ferrum._warn import reset_warnings
from ferrum.encoding.base import ChannelBase


class _TestChannel(ChannelBase):
    _channel_name = "x"
    _renders_in_phase_8a = True
    _honored_kwargs = frozenset(["type", "scale", "title"])


def test_channelbase_stores_field_and_kwargs():
    reset_warnings()
    c = _TestChannel("price", type="Q", title="Price")
    assert c.field == "price"
    assert c._kwargs == {"type": "Q", "title": "Price"}


def test_channelbase_warns_once_on_deferred_kwarg():
    reset_warnings()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _TestChannel("price", axis={"grid": False})
    assert len(w) == 1
    assert "axis" in str(w[0].message)


def test_to_encoding_spec_dict_has_field_and_type():
    reset_warnings()
    c = _TestChannel("price", type="Q")
    d = c.to_encoding_spec_dict()
    assert d["field"] == "price"
    assert d["type_"] == "Q"


def test_to_implicit_transforms_with_bin_kwarg():
    class _BinTestChannel(ChannelBase):
        _channel_name = "x"
        _renders_in_phase_8a = True
        _honored_kwargs = frozenset(["type", "bin", "aggregate"])

    reset_warnings()
    c = _BinTestChannel("price", bin=True)
    transforms = c.to_implicit_transforms()
    assert len(transforms) == 1
    # First (and only) transform should be a Bin instance
    from ferrum import Bin
    assert isinstance(transforms[0], Bin)


def test_to_implicit_transforms_with_aggregate_kwarg():
    class _AggTestChannel(ChannelBase):
        _channel_name = "y"
        _renders_in_phase_8a = True
        _honored_kwargs = frozenset(["type", "aggregate"])

    reset_warnings()
    c = _AggTestChannel("latency", aggregate="mean")
    transforms = c.to_implicit_transforms()
    assert len(transforms) == 1
    from ferrum import Aggregate
    assert isinstance(transforms[0], Aggregate)
```

Run: `uv run pytest tests/test_encoding.py -v`
Expected: ALL FAIL.

- [ ] **Step 2: Implement `encoding/__init__.py` (initially empty)**

```python
"""Encoding channels for Phase 8a."""
```

- [ ] **Step 3: Implement `encoding/base.py`**

```python
"""ChannelBase — the parent class of all encoding-channel value objects."""
from __future__ import annotations

from typing import Any, ClassVar, Optional

from ferrum._warn import warn_once


class ChannelBase:
    """Base class for all encoding-channel value objects.

    Subclasses set _channel_name, _renders_in_phase_8a, and _honored_kwargs.
    Constructor accepts a `field` positional arg + arbitrary keyword arguments;
    unknown kwargs trigger warn_once.
    """

    _channel_name: ClassVar[str] = "_unknown_"
    _renders_in_phase_8a: ClassVar[bool] = False
    _honored_kwargs: ClassVar[frozenset[str]] = frozenset(["type"])

    def __init__(self, field: Optional[str] = None, **kwargs: Any) -> None:
        if field is not None and not isinstance(field, str):
            raise TypeError(
                f"{self.__class__.__name__}: field must be str or None, "
                f"got {type(field).__name__}"
            )
        self.field = field
        self._kwargs = dict(kwargs)
        self._validate()

        for k in self._kwargs:
            if k not in self._honored_kwargs:
                warn_once(self._channel_name, k)

    def _validate(self) -> None:
        """Subclasses may override to enforce kwarg-value constraints."""
        type_ = self._kwargs.get("type")
        if type_ is not None and type_ not in ("Q", "N", "O", "T",
                                                 "quantitative", "nominal", "ordinal", "temporal"):
            raise ValueError(
                f"{self.__class__.__name__}(type={type_!r}): "
                f"expected one of Q, N, O, T, quantitative, nominal, ordinal, temporal"
            )

    def to_encoding_spec_dict(self) -> dict:
        """Return kwargs for the Rust EncodingSpec constructor / serde JSON."""
        out: dict = {"field": self.field}
        if (t := self._kwargs.get("type")) is not None:
            out["type_"] = t
        for k in ("scale", "title", "axis", "legend", "sort", "stack",
                  "impute", "scheme", "format", "formatType"):
            if (v := self._kwargs.get(k)) is not None:
                out[k] = v
        return out

    def to_implicit_transforms(self) -> list:
        """Return a list of transform objects derived from kwargs (bin, aggregate)."""
        out: list = []
        bin_arg = self._kwargs.get("bin")
        if bin_arg:
            from ferrum import Bin
            if isinstance(bin_arg, dict):
                out.append(Bin(self.field, **bin_arg))
            elif isinstance(bin_arg, bool):
                out.append(Bin(self.field))
            else:
                # Bin instance passed directly
                out.append(bin_arg)
        agg = self._kwargs.get("aggregate")
        if agg:
            from ferrum import Aggregate, AggregateOp
            out.append(Aggregate([AggregateOp(self.field or "", agg, f"{agg}_{self.field or 'all'}")]))
        return out

    def __repr__(self) -> str:
        kw_parts = [f"{k}={v!r}" for k, v in self._kwargs.items()]
        body = ", ".join([repr(self.field)] + kw_parts)
        return f"{self.__class__.__name__}({body})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ChannelBase):
            return NotImplemented
        return (self.__class__ == other.__class__
                and self.field == other.field
                and self._kwargs == other._kwargs)

    def __hash__(self) -> int:
        return hash((self.__class__, self.field,
                     tuple(sorted((k, repr(v)) for k, v in self._kwargs.items()))))
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_encoding.py -v -k "channelbase"`
Expected: 5 PASS (the bin/aggregate tests need `from ferrum import Bin, Aggregate` which already exist from Phase 5 — these should work as-is).

- [ ] **Step 5: Commit**

```bash
git add src/ferrum/encoding/__init__.py src/ferrum/encoding/base.py tests/test_encoding.py
git commit -m "feat(encoding): ChannelBase with warn-once + transform desugaring"
```

---

### Task 16: `encoding/positional.py` — 10 positional channels

**Files:**
- Create: `src/ferrum/encoding/positional.py`
- Modify: `src/ferrum/encoding/__init__.py`
- Test: `tests/test_encoding.py` (extend)

- [ ] **Step 1: Add tests**

Append to `tests/test_encoding.py`:

```python
from ferrum.encoding import (
    X, Y, X2, Y2, XError, YError, XError2, YError2, Theta, Radius,
)


def test_x_renders_in_phase_8a():
    assert X._renders_in_phase_8a is True


def test_y_renders_in_phase_8a():
    assert Y._renders_in_phase_8a is True


def test_secondary_positional_channels_are_deferred():
    for cls in (X2, Y2, XError, YError, XError2, YError2, Theta, Radius):
        assert cls._renders_in_phase_8a is False, f"{cls.__name__} should be deferred"


def test_x_construction_with_full_honored_kwargs():
    reset_warnings()
    from ferrum import LinearScale
    c = X("price", type="Q", bin=True, aggregate="mean",
          scale=LinearScale(domain=[0, 100], range=[0, 600]),
          title="Price")
    assert c.field == "price"
    assert c._kwargs["type"] == "Q"


def test_x_warns_on_deferred_kwargs():
    reset_warnings()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        X("price", axis={"grid": False}, sort="ascending")
    assert len(w) == 2
```

- [ ] **Step 2: Implement `encoding/positional.py`**

```python
"""Positional encoding channels (X, Y, X2, Y2, errors, polar)."""
from __future__ import annotations

from ferrum.encoding.base import ChannelBase


class X(ChannelBase):
    _channel_name = "x"
    _renders_in_phase_8a = True
    _honored_kwargs = frozenset([
        "type", "bin", "aggregate", "scale", "title",
        # deferred but accepted (warn-once via base):
        "axis", "sort", "stack", "impute", "format", "formatType",
    ])
    # (axis/sort/stack/impute are listed in _honored_kwargs to AVOID the warn-once
    #  for them at construction time, since we WILL store them on the EncodingSpec.
    #  Whether they're actually used at render time is the renderer's call.
    #  ...Wait — that contradicts the spec. Let me re-read.)
```

> **Design correction:** Per spec §6 row "Channel kwarg deferred" — deferred kwargs trigger warn-once. So `_honored_kwargs` should ONLY include the ones the **renderer** acts on (type, bin, aggregate, scale, title). The deferred kwargs (axis, sort, stack, impute, scheme, format, formatType, legend) trigger the warn-once. Rewrite:

```python
"""Positional encoding channels (X, Y, X2, Y2, errors, polar)."""
from __future__ import annotations

from ferrum.encoding.base import ChannelBase


_RENDERED_HONORED = frozenset(["type", "bin", "aggregate", "scale", "title"])


class X(ChannelBase):
    _channel_name = "x"
    _renders_in_phase_8a = True
    _honored_kwargs = _RENDERED_HONORED


class Y(ChannelBase):
    _channel_name = "y"
    _renders_in_phase_8a = True
    _honored_kwargs = _RENDERED_HONORED


class X2(ChannelBase):
    _channel_name = "x2"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])


class Y2(ChannelBase):
    _channel_name = "y2"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])


class XError(ChannelBase):
    _channel_name = "x_error"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])


class YError(ChannelBase):
    _channel_name = "y_error"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])


class XError2(ChannelBase):
    _channel_name = "x_error2"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])


class YError2(ChannelBase):
    _channel_name = "y_error2"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])


class Theta(ChannelBase):
    _channel_name = "theta"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type", "stack"])


class Radius(ChannelBase):
    _channel_name = "radius"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])
```

- [ ] **Step 3: Re-export from encoding/__init__.py**

```python
"""Encoding channels for Phase 8a."""
from ferrum.encoding.positional import (
    X, Y, X2, Y2, XError, YError, XError2, YError2, Theta, Radius,
)

__all__ = [
    "X", "Y", "X2", "Y2", "XError", "YError", "XError2", "YError2",
    "Theta", "Radius",
]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_encoding.py -v -k "positional or x_ or y_ or _x or _y or theta or radius"`
Expected: 5 new tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/ferrum/encoding/positional.py src/ferrum/encoding/__init__.py tests/test_encoding.py
git commit -m "feat(encoding): 10 positional channels (X, Y, X2, Y2, errors, polar)"
```

---

### Task 17: `encoding/appearance.py` — 11 appearance channels

**Files:**
- Create: `src/ferrum/encoding/appearance.py`
- Modify: `src/ferrum/encoding/__init__.py`

- [ ] **Step 1: Implement appearance.py**

```python
"""Appearance encoding channels (Color, Fill, Stroke, Opacity, Size, Shape, Angle)."""
from __future__ import annotations

from ferrum.encoding.base import ChannelBase


_RENDERED_HONORED = frozenset(["type", "scale", "title"])


# Phase 8a renders these (added to scale_resolve in Task 8):
class Color(ChannelBase):
    _channel_name = "color"
    _renders_in_phase_8a = True
    _honored_kwargs = frozenset(["type", "scheme", "scale", "title"])


class Size(ChannelBase):
    _channel_name = "size"
    _renders_in_phase_8a = True
    _honored_kwargs = _RENDERED_HONORED


class Shape(ChannelBase):
    _channel_name = "shape"
    _renders_in_phase_8a = True
    _honored_kwargs = _RENDERED_HONORED


class Opacity(ChannelBase):
    _channel_name = "opacity"
    _renders_in_phase_8a = True
    _honored_kwargs = _RENDERED_HONORED


# Deferred to Phase 9:
class Fill(ChannelBase):
    _channel_name = "fill"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])


class Stroke(ChannelBase):
    _channel_name = "stroke"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])


class FillOpacity(ChannelBase):
    _channel_name = "fill_opacity"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])


class StrokeOpacity(ChannelBase):
    _channel_name = "stroke_opacity"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])


class StrokeWidth(ChannelBase):
    _channel_name = "stroke_width"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])


class StrokeDash(ChannelBase):
    _channel_name = "stroke_dash"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])


class Angle(ChannelBase):
    _channel_name = "angle"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])
```

- [ ] **Step 2: Re-export**

Update `encoding/__init__.py`:

```python
from ferrum.encoding.appearance import (
    Color, Fill, Stroke, Opacity, FillOpacity, StrokeOpacity,
    StrokeWidth, StrokeDash, Size, Shape, Angle,
)

__all__ += [
    "Color", "Fill", "Stroke", "Opacity", "FillOpacity", "StrokeOpacity",
    "StrokeWidth", "StrokeDash", "Size", "Shape", "Angle",
]
```

- [ ] **Step 3: Tests**

Append to `tests/test_encoding.py`:

```python
from ferrum.encoding import (
    Color, Fill, Stroke, Opacity, FillOpacity, StrokeOpacity,
    StrokeWidth, StrokeDash, Size, Shape, Angle,
)


def test_color_renders_in_phase_8a():
    assert Color._renders_in_phase_8a is True


def test_size_shape_opacity_render_in_phase_8a():
    for cls in (Size, Shape, Opacity):
        assert cls._renders_in_phase_8a is True, f"{cls.__name__} must render in 8a"


def test_other_appearance_channels_deferred():
    for cls in (Fill, Stroke, FillOpacity, StrokeOpacity, StrokeWidth, StrokeDash, Angle):
        assert cls._renders_in_phase_8a is False, f"{cls.__name__} should be deferred"


def test_color_with_scheme_kwarg_no_warning():
    reset_warnings()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Color("species", scheme="tableau10")
    assert len(w) == 0  # scheme is honored for Color in 8a


def test_stroke_with_field_warns_once_on_render_attempt():
    # Channel-deferred warning fires at construction OR at render; both acceptable.
    # In our impl, since Stroke is deferred and Stroke isn't in honored_kwargs,
    # any kwarg triggers warn_once. Bare construction is fine.
    reset_warnings()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Stroke("color")
    # Bare construction with just `field` doesn't pass kwargs → no warning yet.
    assert len(w) == 0
```

- [ ] **Step 4: Run tests + commit**

```bash
uv run pytest tests/test_encoding.py -v -k "color or size or shape or opacity or fill or stroke or angle"
git add src/ferrum/encoding/appearance.py src/ferrum/encoding/__init__.py tests/test_encoding.py
git commit -m "feat(encoding): 11 appearance channels (Color, Size, Shape, Opacity rendered; rest deferred)"
```

---

### Task 18: `encoding/text.py` — 7 text/detail/tooltip classes

**Files:**
- Create: `src/ferrum/encoding/text.py`

- [ ] **Step 1: Implement**

```python
"""Text/Detail/Tooltip/Href/Description/Key channels (all deferred to Phase 9)."""
from __future__ import annotations

from ferrum.encoding.base import ChannelBase


class Text(ChannelBase):
    _channel_name = "text"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type", "format", "formatType"])


class Detail(ChannelBase):
    _channel_name = "detail"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])


class Tooltip(ChannelBase):
    _channel_name = "tooltip"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])

    def __init__(self, *fields, **kwargs):
        # Tooltip(*fields) is a special case: takes a list of fields, not just one
        if len(fields) == 1:
            super().__init__(fields[0], **kwargs)
            self._field_list = [fields[0]]
        else:
            super().__init__(None, **kwargs)
            self._field_list = list(fields)


class TooltipField(ChannelBase):
    """Helper class used inside Tooltip(*fields). Not used as a channel directly."""
    _channel_name = "tooltip_field"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type", "title", "format", "formatType"])


class Href(ChannelBase):
    _channel_name = "href"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])


class Description(ChannelBase):
    _channel_name = "description"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])


class Key(ChannelBase):
    _channel_name = "key"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])
```

- [ ] **Step 2: Re-export from `encoding/__init__.py`**

```python
from ferrum.encoding.text import (
    Text, Detail, Tooltip, TooltipField, Href, Description, Key,
)
__all__ += ["Text", "Detail", "Tooltip", "TooltipField", "Href", "Description", "Key"]
```

- [ ] **Step 3: Tests**

```python
def test_text_channels_all_deferred():
    from ferrum.encoding import Text, Detail, Tooltip, TooltipField, Href, Description, Key
    for cls in (Text, Detail, Tooltip, TooltipField, Href, Description, Key):
        assert cls._renders_in_phase_8a is False


def test_tooltip_accepts_multiple_fields():
    from ferrum.encoding import Tooltip
    t = Tooltip("a", "b", "c")
    assert t._field_list == ["a", "b", "c"]
```

Run: `uv run pytest tests/test_encoding.py -v -k "text or tooltip or href"`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/ferrum/encoding/text.py src/ferrum/encoding/__init__.py tests/test_encoding.py
git commit -m "feat(encoding): 7 text/detail/tooltip classes (all deferred to Phase 9)"
```

---

### Task 19: `encoding/facet.py` — 3 facet channels

**Files:**
- Create: `src/ferrum/encoding/facet.py`

- [ ] **Step 1: Implement**

```python
"""Facet encoding channels."""
from __future__ import annotations

from ferrum.encoding.base import ChannelBase


class Facet(ChannelBase):
    _channel_name = "facet"
    _renders_in_phase_8a = True   # rendered via Phase 6 facet pipeline
    _honored_kwargs = frozenset(["type", "title"])


class FacetRow(ChannelBase):
    _channel_name = "facet_row"
    _renders_in_phase_8a = True
    _honored_kwargs = frozenset(["type", "title"])


class FacetCol(ChannelBase):
    _channel_name = "facet_col"
    _renders_in_phase_8a = True
    _honored_kwargs = frozenset(["type", "title"])
```

- [ ] **Step 2: Re-export + tests + commit**

```python
# encoding/__init__.py
from ferrum.encoding.facet import Facet, FacetRow, FacetCol
__all__ += ["Facet", "FacetRow", "FacetCol"]
```

```python
# tests/test_encoding.py
def test_facet_channels_render():
    from ferrum.encoding import Facet, FacetRow, FacetCol
    for cls in (Facet, FacetRow, FacetCol):
        assert cls._renders_in_phase_8a is True
```

```bash
uv run pytest tests/test_encoding.py -v
git add src/ferrum/encoding/facet.py src/ferrum/encoding/__init__.py tests/test_encoding.py
git commit -m "feat(encoding): 3 facet channels (Facet, FacetRow, FacetCol)"
```

---

## Group F — Python theme system

### Task 20: `themes/__init__.py` + `Theme` value class

**Files:**
- Create: `src/ferrum/themes/__init__.py`
- Test: `tests/test_theme.py` (initial)

- [ ] **Step 1: Write failing tests**

Create `tests/test_theme.py`:

```python
import pytest

from ferrum.themes import Theme


def test_theme_default_has_no_props():
    t = Theme()
    assert t._props == {}


def test_theme_with_kwargs_stores_them():
    t = Theme(background="#000", font_family="Inter")
    assert t._props == {"background": "#000", "font_family": "Inter"}


def test_theme_omits_none_values():
    t = Theme(background="#000", font_family=None)
    assert t._props == {"background": "#000"}


def test_theme_update_returns_new_theme_with_merged_props():
    t1 = Theme(background="#000")
    t2 = t1.update(font_family="Inter")
    assert t1._props == {"background": "#000"}
    assert t2._props == {"background": "#000", "font_family": "Inter"}
    assert t1 is not t2


def test_theme_update_overrides_existing_prop():
    t1 = Theme(background="#000")
    t2 = t1.update(background="#fff")
    assert t1._props == {"background": "#000"}
    assert t2._props == {"background": "#fff"}


def test_theme_eq_when_props_match():
    t1 = Theme(background="#000", font_family="Inter")
    t2 = Theme(font_family="Inter", background="#000")
    assert t1 == t2


def test_theme_to_theme_inputs_dict_passes_through_props():
    t = Theme(background="#1a1a2e", font_color="#e6e6e6")
    d = t.to_theme_inputs_dict()
    assert d["background"] == "#1a1a2e"
    assert d["font_color"] == "#e6e6e6"


def test_theme_hashable():
    t = Theme(background="#000")
    s = {t}
    assert t in s
```

Run: `uv run pytest tests/test_theme.py -v`
Expected: ALL FAIL.

- [ ] **Step 2: Implement**

Create `src/ferrum/themes/__init__.py`:

```python
"""Theme value class + 8 builtins + set_default_theme."""
from __future__ import annotations

from typing import Any


class Theme:
    """Immutable theme value class. Pass via Chart.theme(t) or set_default_theme(t).

    All props default to None and are dropped from the dict passed to the
    Rust ThemeInputs binding (so Rust falls back to its defaults).

    Use .update(**kwargs) to derive a new Theme with overrides; the source
    theme is unchanged.
    """

    __slots__ = ("_props",)

    def __init__(self, **kwargs: Any) -> None:
        self._props: dict = {k: v for k, v in kwargs.items() if v is not None}

    def update(self, **kwargs: Any) -> "Theme":
        merged = {**self._props}
        for k, v in kwargs.items():
            if v is None:
                merged.pop(k, None)
            else:
                merged[k] = v
        return Theme(**merged)

    def to_theme_inputs_dict(self) -> dict:
        """Return a dict suitable for ferrum._core.render_svg(theme=...)."""
        return dict(self._props)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Theme):
            return NotImplemented
        return self._props == other._props

    def __hash__(self) -> int:
        return hash(tuple(sorted(self._props.items(), key=lambda kv: kv[0])))

    def __repr__(self) -> str:
        if not self._props:
            return "Theme()"
        kv = ", ".join(f"{k}={v!r}" for k, v in sorted(self._props.items()))
        return f"Theme({kv})"
```

- [ ] **Step 3: Run tests + commit**

```bash
uv run pytest tests/test_theme.py -v
git add src/ferrum/themes/__init__.py tests/test_theme.py
git commit -m "feat(themes): Theme immutable value class with .update()"
```

---

### Task 21: `themes/builtins.py` — 8 builtin themes

**Files:**
- Create: `src/ferrum/themes/builtins.py`
- Modify: `src/ferrum/themes/__init__.py` (re-export builtins)

- [ ] **Step 1: Implement builtins**

Create `src/ferrum/themes/builtins.py`:

```python
"""8 built-in themes. Color and font choices for `dark`, `fivethirtyeight`,
`economist`, `solarized_*` reference vega-lite theme JSONs where the spec
is ambiguous. See spec §3.13 + §10 row 12.
"""
from __future__ import annotations

from ferrum.themes import Theme


# Ferrum defaults (all None → Rust ThemeInputs::default())
default = Theme()

# Minimal: no grid, no axis lines, generous padding
minimal = Theme(
    grid=False,
    axis_line=False,
    padding=20,
)

# Dark: low-contrast dark background, light text, dark-friendly palette
dark = Theme(
    background="#1a1a2e",
    font_color="#e6e6e6",
    title_color="#ffffff",
    axis_line_color="#666666",
    grid_color="#333333",
    color_scheme="dark2",
)

# Publication: print-ready, no background, high contrast, Tableau10
publication = Theme(
    background=None,
    grid=False,
    color_scheme="tableau10",
    font_family="Inter",
    title_font_weight="bold",
    axis_line_color="#000000",
    font_color="#000000",
)

# Economist: red accents, light blue background, no axis lines
economist = Theme(
    background="#d3e0e6",
    font_family="Inter",
    title_color="#c00000",
    grid_color="#b0c4cc",
    axis_line=False,
    color_scheme="set1",
)

# FiveThirtyEight-style: grey bg, divergent palette, no axis lines
fivethirtyeight = Theme(
    background="#f0f0f0",
    color_scheme="set1",
    grid_color="#cccccc",
    axis_line=False,
    font_family="Inter",
)

# Solarized light: warm cream bg
solarized_light = Theme(
    background="#fdf6e3",
    font_color="#586e75",
    title_color="#073642",
    grid_color="#eee8d5",
    axis_line_color="#93a1a1",
    color_scheme="set2",
)

# Solarized dark
solarized_dark = Theme(
    background="#002b36",
    font_color="#93a1a1",
    title_color="#fdf6e3",
    grid_color="#073642",
    axis_line_color="#586e75",
    color_scheme="set2",
)
```

- [ ] **Step 2: Re-export from themes/__init__.py**

Append to `src/ferrum/themes/__init__.py`:

```python
from ferrum.themes import builtins as _builtins  # noqa: E402

# Re-export the 8 builtins as module attributes
default = _builtins.default
minimal = _builtins.minimal
dark = _builtins.dark
publication = _builtins.publication
economist = _builtins.economist
fivethirtyeight = _builtins.fivethirtyeight
solarized_light = _builtins.solarized_light
solarized_dark = _builtins.solarized_dark

__all__ = [
    "Theme", "default", "minimal", "dark", "publication", "economist",
    "fivethirtyeight", "solarized_light", "solarized_dark",
]
```

- [ ] **Step 3: Tests**

Append to `tests/test_theme.py`:

```python
def test_eight_builtins_exist():
    from ferrum.themes import (default, minimal, dark, publication,
                                 economist, fivethirtyeight,
                                 solarized_light, solarized_dark)
    for t in (default, minimal, dark, publication, economist,
              fivethirtyeight, solarized_light, solarized_dark):
        assert isinstance(t, Theme)


def test_default_theme_has_no_props():
    from ferrum.themes import default
    assert default._props == {}


def test_dark_theme_has_dark_background():
    from ferrum.themes import dark
    assert dark._props["background"] == "#1a1a2e"
```

- [ ] **Step 4: Run tests + commit**

```bash
uv run pytest tests/test_theme.py -v
git add src/ferrum/themes/builtins.py src/ferrum/themes/__init__.py tests/test_theme.py
git commit -m "feat(themes): 8 builtin themes (default, minimal, dark, publication, economist, fivethirtyeight, solarized_*)"
```

---

### Task 22: `themes/_defaults.py` — `set_default_theme` + contextvar stack

**Files:**
- Create: `src/ferrum/themes/_defaults.py`
- Modify: `src/ferrum/themes/__init__.py` (re-export)
- Modify: `src/ferrum/__init__.py` (re-export `set_default_theme`)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_theme.py`:

```python
def test_set_default_theme_sets_process_default():
    import ferrum
    from ferrum.themes import dark, default, get_default_theme
    from ferrum.themes._defaults import _default_theme

    # Save and restore to avoid bleed-through
    original = get_default_theme()
    try:
        ferrum.set_default_theme(dark)
        assert get_default_theme() is dark
    finally:
        _default_theme.set(original)


def test_set_default_theme_returns_context_manager():
    import ferrum
    from ferrum.themes import dark, default, get_default_theme

    original = get_default_theme()
    with ferrum.set_default_theme(dark):
        assert get_default_theme() is dark
    assert get_default_theme() is original


def test_nested_set_default_theme_restores_correctly():
    import ferrum
    from ferrum.themes import dark, minimal, get_default_theme

    original = get_default_theme()
    with ferrum.set_default_theme(dark):
        with ferrum.set_default_theme(minimal):
            assert get_default_theme() is minimal
        assert get_default_theme() is dark
    assert get_default_theme() is original
```

Run: `uv run pytest tests/test_theme.py -v -k "default_theme"`
Expected: ALL FAIL (`set_default_theme` doesn't exist).

- [ ] **Step 2: Implement**

Create `src/ferrum/themes/_defaults.py`:

```python
"""Process-default theme stack backed by contextvars.

Per spec §10 row 11: the only sanctioned global theme state.
Per-chart Chart.theme(t) always wins over this default.
"""
from __future__ import annotations

import contextvars

from ferrum.themes import Theme, default as _ferrum_default


_default_theme: contextvars.ContextVar[Theme] = contextvars.ContextVar(
    "_ferrum_default_theme", default=_ferrum_default,
)


class _DefaultThemeCM:
    """Context manager returned by set_default_theme(). Restores prior default on __exit__.
    Also acts as a plain object for fire-and-forget set_default_theme(t) usage."""

    __slots__ = ("_token",)

    def __init__(self, token: contextvars.Token) -> None:
        self._token = token

    def __enter__(self) -> "_DefaultThemeCM":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        _default_theme.reset(self._token)


def set_default_theme(theme: Theme) -> _DefaultThemeCM:
    """Set the process-default theme. Per-chart Chart.theme(t) overrides this.

    Returns a context manager that restores the previous default on __exit__.
    Fire-and-forget usage (without `with`) is also supported — the previous
    theme stays restorable via the returned token.
    """
    if not isinstance(theme, Theme):
        raise TypeError(f"theme must be a Theme instance, got {type(theme).__name__}")
    token = _default_theme.set(theme)
    return _DefaultThemeCM(token)


def get_default_theme() -> Theme:
    """Return the current process-default theme."""
    return _default_theme.get()


def theme_context(theme: Theme) -> _DefaultThemeCM:
    """Alias for set_default_theme() — explicit context-manager spelling."""
    return set_default_theme(theme)
```

- [ ] **Step 3: Re-export**

In `src/ferrum/themes/__init__.py`, append:

```python
from ferrum.themes._defaults import set_default_theme, get_default_theme, theme_context
__all__ += ["set_default_theme", "get_default_theme", "theme_context"]
```

In `src/ferrum/__init__.py`, add (after existing imports):

```python
from ferrum.themes import (
    Theme, set_default_theme, get_default_theme, theme_context,
)
import ferrum.themes as themes  # so users can write ferrum.themes.dark

__all__ += ["Theme", "themes", "set_default_theme", "get_default_theme", "theme_context"]
```

- [ ] **Step 4: Run tests + commit**

```bash
uv run pytest tests/test_theme.py -v
git add src/ferrum/themes/_defaults.py src/ferrum/themes/__init__.py src/ferrum/__init__.py tests/test_theme.py
git commit -m "feat(themes): set_default_theme returns contextvars-backed CM"
```

---

## Group G — Python marks

### Task 23: `marks/base.py` — `MarkBase`

**Files:**
- Create: `src/ferrum/marks/__init__.py`
- Create: `src/ferrum/marks/base.py`

- [ ] **Step 1: Implement `MarkBase`**

Create `src/ferrum/marks/__init__.py`:

```python
"""Mark builder functions and base class."""
```

Create `src/ferrum/marks/base.py`:

```python
"""MarkBase — kwarg validation + storage for mark style overrides.

Phase 8a: only constant overrides are supported (e.g. mark_point(size=100)).
Encoding-driven overrides come through .encode(size=Size("col")).
"""
from __future__ import annotations

from typing import Any, ClassVar


_VALID_MARK_KWARGS = frozenset([
    "size", "stroke", "fill", "opacity", "corner_radius",
    "stroke_width", "stroke_dash", "font_size", "font_weight",
    "align", "baseline", "dx", "dy", "angle",
    # Mark-specific (validated per-mark):
    "interpolate", "stroke_cap", "stroke_join",            # line/area
    "orient",                                              # bar/tick
    "filled", "shape",                                      # point
    "limit",                                               # text
    "band_size",                                           # tick
    "line", "borders",                                     # area / errorband
    # Statistical mark kwargs (forwarded to transform):
    "method", "ci", "bandwidth", "degree", "n",            # smooth
    "kernel", "extent", "cumulative",                      # density
    "bin_count", "bin_width", "density", "right",          # histogram
    "multiple",                                            # density/histogram
])


class MarkBase:
    """Validate + store mark-level keyword arguments.

    Used by mark_*() builder functions in marks/__init__.py to validate kwargs
    before serializing them into ChartSpec.mark_style.
    """

    def __init__(self, mark_name: str, **kwargs: Any) -> None:
        self.mark_name = mark_name
        for k in kwargs:
            if k not in _VALID_MARK_KWARGS:
                raise TypeError(
                    f"mark_{mark_name}: unknown keyword argument {k!r}. "
                    f"Valid: {sorted(_VALID_MARK_KWARGS)}"
                )
        self._kwargs = dict(kwargs)

    def to_mark_kwargs_dict(self) -> dict:
        """Subset of kwargs that map to MarkKwargsSpec fields. Other kwargs
        (e.g. statistical mark kwargs like `bandwidth`) are returned in
        `to_transform_kwargs()` if applicable, not here."""
        out = {}
        for k in ("size", "stroke", "fill", "opacity", "corner_radius",
                  "stroke_width", "stroke_dash", "font_size", "font_weight",
                  "align", "baseline", "dx", "dy", "angle"):
            if k in self._kwargs:
                out[k] = self._kwargs[k]
        return out
```

- [ ] **Step 2: Tests**

Create `tests/test_marks.py`:

```python
import pytest

from ferrum.marks.base import MarkBase


def test_markbase_accepts_valid_kwargs():
    m = MarkBase("point", size=100, stroke="#ff0000", opacity=0.5)
    assert m._kwargs == {"size": 100, "stroke": "#ff0000", "opacity": 0.5}


def test_markbase_rejects_unknown_kwargs():
    with pytest.raises(TypeError, match="unknown keyword"):
        MarkBase("point", squiggly=True)


def test_to_mark_kwargs_dict_filters_to_style_only():
    m = MarkBase("smooth", size=100, method="loess", bandwidth=0.5)
    d = m.to_mark_kwargs_dict()
    assert d == {"size": 100}   # method and bandwidth go to transforms, not style
```

Run: `uv run pytest tests/test_marks.py -v -k "markbase"`
Expected: 3 PASS.

- [ ] **Step 3: Commit**

```bash
git add src/ferrum/marks/__init__.py src/ferrum/marks/base.py tests/test_marks.py
git commit -m "feat(marks): MarkBase with kwarg validation"
```

---

### Task 24: `marks/__init__.py` — 8 primitive `mark_*()` builder functions

**Note:** For Phase 8a these are accessed via `Chart.mark_point(...)` not as standalone module-level functions (per spec §3.1 / §4.1). The builder logic lives in `Chart` (Task 27). This task just defines internal helpers + the deferred-mark stubs.

**Files:**
- Create: `src/ferrum/marks/deferred.py`
- Modify: `src/ferrum/marks/__init__.py`

- [ ] **Step 1: Implement deferred-mark stubs**

Create `src/ferrum/marks/deferred.py`:

```python
"""Marks deferred to Phase 8b or Phase 9+. These exist as Chart methods that
raise NotImplementedError with a clear forward-pointer."""
from __future__ import annotations

# Phase 8b marks
PHASE_8B_MARKS = frozenset([
    "boxplot", "errorbar", "errorband", "ribbon",        # composite
    "contour", "violin", "qq", "raster", "swarm", "hex", "function",   # heavy stat
])

# Phase 9+ marks
PHASE_9_PLUS_MARKS = frozenset([
    "arc", "image", "geoshape", "segment", "label",
])


def deferred_mark_error(mark_name: str) -> NotImplementedError:
    """Build an informative NotImplementedError for a deferred mark."""
    if mark_name in PHASE_8B_MARKS:
        return NotImplementedError(
            f"mark_{mark_name} is planned for Phase 8b. "
            f"See docs/superpowers/ferrum-phases.md for the roadmap."
        )
    if mark_name in PHASE_9_PLUS_MARKS:
        return NotImplementedError(
            f"mark_{mark_name} is planned for Phase 9+. "
            f"See docs/superpowers/ferrum-phases.md for the roadmap."
        )
    return NotImplementedError(f"mark_{mark_name} is not implemented.")
```

- [ ] **Step 2: Re-export**

Update `src/ferrum/marks/__init__.py`:

```python
"""Marks — primitive + statistical (Phase 8a). Composite + heavy stat = Phase 8b.

Marks are normally accessed as Chart methods: chart.mark_point(...). The
mark functions below exist for direct construction in figure-level code paths.
"""
from ferrum.marks.base import MarkBase
from ferrum.marks.deferred import deferred_mark_error, PHASE_8B_MARKS, PHASE_9_PLUS_MARKS

__all__ = ["MarkBase", "deferred_mark_error", "PHASE_8B_MARKS", "PHASE_9_PLUS_MARKS"]
```

- [ ] **Step 3: Tests**

Append to `tests/test_marks.py`:

```python
def test_deferred_mark_error_for_8b_mark():
    from ferrum.marks import deferred_mark_error, PHASE_8B_MARKS
    e = deferred_mark_error("boxplot")
    assert isinstance(e, NotImplementedError)
    assert "Phase 8b" in str(e)


def test_deferred_mark_error_for_9_plus_mark():
    from ferrum.marks import deferred_mark_error
    e = deferred_mark_error("arc")
    assert "Phase 9+" in str(e)


def test_phase_8b_marks_set_includes_composites_and_heavy_stats():
    from ferrum.marks import PHASE_8B_MARKS
    assert {"boxplot", "errorbar", "violin", "raster"}.issubset(PHASE_8B_MARKS)
```

Run: `uv run pytest tests/test_marks.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/ferrum/marks/__init__.py src/ferrum/marks/deferred.py tests/test_marks.py
git commit -m "feat(marks): deferred-mark stubs for Phase 8b and 9+"
```

---

### Task 25: `marks/statistical.py` — mark_density / mark_histogram / mark_smooth

**Note:** Like the primitives, these become Chart methods in Task 27. This task defines the desugaring logic as helper functions consumed by `Chart`.

**Files:**
- Create: `src/ferrum/marks/statistical.py`

- [ ] **Step 1: Implement helpers**

```python
"""Statistical mark desugaring — convert mark_density/histogram/smooth kwargs
into (mark, transforms, encoding_remap) tuples consumed by Chart."""
from __future__ import annotations

from typing import Any

from ferrum import Bin, Kde, Smooth


def desugar_density(field: str, **kwargs: Any) -> tuple[str, list, dict]:
    """mark_density → mark_area + Kde(field, ...) + remap y → density column."""
    bandwidth = kwargs.pop("bandwidth", "scott")
    kernel = kwargs.pop("kernel", "gaussian")
    n = kwargs.pop("n", 512)
    extent = kwargs.pop("extent", None)
    cumulative = kwargs.pop("cumulative", False)
    # `multiple` parameter from spec §3.3 deferred (no stack support yet)
    if kwargs.pop("multiple", "layer") != "layer":
        # warn-once at Chart layer; here we just drop it
        pass

    transforms = [Kde(field, bandwidth=bandwidth, n=n, extent=extent, cumulative=cumulative)]
    # Phase 5 Kde produces columns (field, "density"); encoding_remap tells Chart
    # to treat the density column as y when wiring the area mark
    encoding_remap = {"y": "density"}
    return ("area", transforms, encoding_remap)


def desugar_histogram(field: str, **kwargs: Any) -> tuple[str, list, dict]:
    """mark_histogram → mark_bar + Bin(field, ...) + count or density on y."""
    bin_count = kwargs.pop("bin_count", None)
    bin_width = kwargs.pop("bin_width", None)
    extent = kwargs.pop("extent", None)
    nice = kwargs.pop("nice", True)
    density = kwargs.pop("density", False)
    cumulative = kwargs.pop("cumulative", False)
    right = kwargs.pop("right", False)
    multiple = kwargs.pop("multiple", "layer")

    transforms = [Bin(field, bin_count=bin_count, bin_width=bin_width, extent=extent, nice=nice)]
    # Phase 5 Bin produces columns (bin_start, bin_end, count, density)
    y_column = "density" if density else "count"
    encoding_remap = {"x": "bin_start", "x2": "bin_end", "y": y_column}
    return ("bar", transforms, encoding_remap)


def desugar_smooth(x_field: str, y_field: str, **kwargs: Any) -> tuple[str, list, dict]:
    """mark_smooth → mark_line + Smooth(x, y, ...). Phase 8a does NOT render the CI band."""
    method = kwargs.pop("method", "loess")
    ci = kwargs.pop("ci", None)
    bandwidth = kwargs.pop("bandwidth", 0.75)
    degree = kwargs.pop("degree", 2)
    n = kwargs.pop("n", 200)

    if ci is not None:
        # warn-once: CI band requires Phase 8b ribbon mark
        from ferrum._warn import warn_once
        warn_once("mark_smooth", "ci",
                  "mark_smooth(ci=...) requires the ribbon mark; deferred to Phase 8b. "
                  "Smooth curve rendered without CI band.")

    transforms = [Smooth(x_field, y_field, method=method, ci=None, bandwidth=bandwidth,
                         degree=degree, n=n)]
    # Phase 5 Smooth produces (x, y) columns named after as_ tuple; default ("x", "y")
    encoding_remap = {"x": "x", "y": "y"}
    return ("line", transforms, encoding_remap)
```

- [ ] **Step 2: Tests**

Append to `tests/test_marks.py`:

```python
def test_desugar_density_returns_area_with_kde_transform():
    from ferrum.marks.statistical import desugar_density
    from ferrum import Kde
    mark, transforms, remap = desugar_density("price")
    assert mark == "area"
    assert len(transforms) == 1 and isinstance(transforms[0], Kde)
    assert remap == {"y": "density"}


def test_desugar_histogram_returns_bar_with_bin_transform():
    from ferrum.marks.statistical import desugar_histogram
    from ferrum import Bin
    mark, transforms, remap = desugar_histogram("price", bin_count=20)
    assert mark == "bar"
    assert isinstance(transforms[0], Bin)
    assert remap == {"x": "bin_start", "x2": "bin_end", "y": "count"}


def test_desugar_smooth_warns_on_ci_kwarg():
    import warnings
    from ferrum._warn import reset_warnings
    from ferrum.marks.statistical import desugar_smooth

    reset_warnings()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        desugar_smooth("x_col", "y_col", ci=0.95)
    assert any("ci=" in str(wi.message) and "Phase 8b" in str(wi.message) for wi in w)
```

Run: `uv run pytest tests/test_marks.py -v -k "density or histogram or smooth"`
Expected: 3 PASS.

- [ ] **Step 3: Commit**

```bash
git add src/ferrum/marks/statistical.py tests/test_marks.py
git commit -m "feat(marks): mark_density/mark_histogram/mark_smooth desugaring helpers"
```

---

### Task 26: Wire mark builder helpers (no-op placeholder for Group H)

**Note:** This task is a placeholder — the actual `Chart.mark_*()` methods land in Task 27. Skip if you prefer; or use it to verify the imports flow cleanly.

```bash
# Sanity check that all marks/* modules import cleanly
uv run python -c "from ferrum.marks import MarkBase, deferred_mark_error, PHASE_8B_MARKS; from ferrum.marks.statistical import desugar_density, desugar_histogram, desugar_smooth; print('marks imports OK')"
```

If this prints `marks imports OK`, proceed to Group H.

(No commit needed — this is a verification step.)

---

## Group H — Python Chart + composition

### Task 27: `chart.py` — `Chart` class core (data, immutability, encode, mark methods)

**Files:**
- Create: `src/ferrum/chart.py`
- Test: `tests/test_chart.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_chart.py`:

```python
import polars as pl
import pyarrow as pa
import pytest

from ferrum import Chart


def test_chart_construction_with_polars():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    c = Chart(df)
    assert c._data is df


def test_chart_immutability_mark_returns_new_chart():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    c1 = Chart(df)
    c2 = c1.mark_point()
    assert c1 is not c2
    assert c1._mark is None
    assert c2._mark == "point"


def test_chart_encode_returns_new_chart():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    c1 = Chart(df).mark_point()
    c2 = c1.encode(x="a", y="b")
    assert c1 is not c2
    assert c1._encoding == {}
    assert "x" in c2._encoding


def test_chart_mark_point_with_kwargs():
    df = pl.DataFrame({"a": [1], "b": [2]})
    c = Chart(df).mark_point(size=100, stroke="#ff0000")
    assert c._mark == "point"
    assert c._mark_kwargs == {"size": 100, "stroke": "#ff0000"}


def test_chart_encode_with_string_field():
    df = pl.DataFrame({"price": [1.0], "weight": [2.0]})
    c = Chart(df).mark_point().encode(x="price", y="weight")
    assert c._encoding["x"].field == "price"
    assert c._encoding["y"].field == "weight"


def test_chart_encode_with_shorthand_aggregate():
    df = pl.DataFrame({"price": [1.0]})
    c = Chart(df).mark_bar().encode(y="mean(price)")
    # The shorthand should desugar into an Aggregate transform
    assert any(t.__class__.__name__ == "Aggregate" for t in c._transforms)


def test_chart_encode_with_explicit_channel_class():
    from ferrum.encoding import X, Y
    df = pl.DataFrame({"a": [1], "b": [2]})
    c = Chart(df).mark_point().encode(x=X("a", type="Q"), y=Y("b"))
    assert c._encoding["x"].field == "a"
    assert c._encoding["x"]._kwargs["type"] == "Q"


def test_chart_to_spec_returns_chartspec():
    from ferrum import ChartSpec
    df = pl.DataFrame({"a": [1], "b": [2]})
    c = Chart(df).mark_point().encode(x="a", y="b")
    spec = c.to_spec()
    assert isinstance(spec, ChartSpec)
    assert spec.mark == "point"


def test_chart_to_json_round_trip():
    df = pl.DataFrame({"a": [1], "b": [2]})
    c = Chart(df).mark_point().encode(x="a", y="b")
    j = c.to_json()
    assert "point" in j
    assert "\"x\":" in j


def test_chart_data_input_pyarrow_table():
    tbl = pa.table({"a": [1, 2], "b": [3, 4]})
    c = Chart(tbl).mark_point().encode(x="a", y="b")
    # show_svg actually exercises the coerce path; smoke-test only here
    spec = c.to_spec()
    assert spec.mark == "point"


def test_chart_data_input_dict():
    c = Chart({"a": [1, 2], "b": [3, 4]}).mark_point().encode(x="a", y="b")
    assert c._mark == "point"


def test_chart_data_input_list_of_records():
    c = Chart([{"a": 1, "b": 2}, {"a": 3, "b": 4}]).mark_point().encode(x="a", y="b")
    assert c._mark == "point"


def test_chart_data_input_numpy_2d():
    np = pytest.importorskip("numpy")
    arr = np.array([[1, 2], [3, 4]])
    c = Chart(arr).mark_point().encode(x="col_0", y="col_1")
    assert c._mark == "point"


def test_chart_data_input_numpy_1d_raises():
    np = pytest.importorskip("numpy")
    arr = np.array([1, 2, 3])
    with pytest.raises(TypeError, match="1D numpy"):
        Chart(arr).mark_point().show_svg()  # show_svg triggers coerce


def test_chart_properties_sets_metadata():
    df = pl.DataFrame({"a": [1], "b": [2]})
    c = Chart(df).mark_point().properties(width=800, height=600, title="Hello")
    assert c._width == 800
    assert c._height == 600
    assert c._title == "Hello"
```

Run: `uv run pytest tests/test_chart.py -v`
Expected: ALL FAIL (Chart doesn't exist).

- [ ] **Step 2: Implement Chart class**

Create `src/ferrum/chart.py`:

```python
"""Chart — the user-facing top-level value class.

Immutability rule: every fluent method returns a new Chart. The internal
spec is deep-copied on each call so chains compose without aliasing surprises.
"""
from __future__ import annotations

import copy
from typing import Any, Optional, Union

from ferrum._coerce import to_arrow_table
from ferrum._shorthand import parse_shorthand
from ferrum.encoding.base import ChannelBase
from ferrum.marks.base import MarkBase
from ferrum.marks.deferred import deferred_mark_error, PHASE_8B_MARKS, PHASE_9_PLUS_MARKS
from ferrum.marks.statistical import desugar_density, desugar_histogram, desugar_smooth


_PRIMITIVE_MARKS = frozenset(["point", "line", "bar", "area", "rule", "text", "tick", "rect"])

_CHANNEL_CLASSES_BY_NAME: dict = {}


def _channel_class_for(name: str):
    """Return the channel-class for a given parameter name (lazy import to avoid cycles)."""
    if not _CHANNEL_CLASSES_BY_NAME:
        from ferrum.encoding import (
            X, Y, X2, Y2, XError, YError, XError2, YError2, Theta, Radius,
            Color, Fill, Stroke, Opacity, FillOpacity, StrokeOpacity,
            StrokeWidth, StrokeDash, Size, Shape, Angle,
            Text, Detail, Tooltip, TooltipField, Href, Description, Key,
            Facet, FacetRow, FacetCol,
        )
        _CHANNEL_CLASSES_BY_NAME.update({
            "x": X, "y": Y, "x2": X2, "y2": Y2,
            "x_error": XError, "y_error": YError, "x_error2": XError2, "y_error2": YError2,
            "theta": Theta, "radius": Radius,
            "color": Color, "fill": Fill, "stroke": Stroke,
            "opacity": Opacity, "fill_opacity": FillOpacity, "stroke_opacity": StrokeOpacity,
            "stroke_width": StrokeWidth, "stroke_dash": StrokeDash,
            "size": Size, "shape": Shape, "angle": Angle,
            "text": Text, "detail": Detail, "tooltip": Tooltip, "tooltip_field": TooltipField,
            "href": Href, "description": Description, "key": Key,
            "facet": Facet, "facet_row": FacetRow, "facet_col": FacetCol,
        })
    return _CHANNEL_CLASSES_BY_NAME.get(name)


class Chart:
    """Top-level chart value class. Immutable — every method returns a new Chart."""

    __slots__ = (
        "_data", "_mark", "_mark_kwargs", "_encoding", "_transforms",
        "_facet", "_coord", "_theme", "_layers",
        "_width", "_height", "_title", "_description",
    )

    def __init__(
        self,
        data: Any = None,
        *,
        width: Optional[Union[int, str]] = None,
        height: Optional[Union[int, str]] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        self._data = data
        self._mark = None
        self._mark_kwargs = {}
        self._encoding: dict = {}
        self._transforms: list = []
        self._facet = None
        self._coord = None
        self._theme = None
        self._layers: Optional[list] = None
        self._width = width
        self._height = height
        self._title = title
        self._description = description

    def _clone(self) -> "Chart":
        new = object.__new__(Chart)
        new._data = self._data
        new._mark = self._mark
        new._mark_kwargs = dict(self._mark_kwargs)
        new._encoding = dict(self._encoding)
        new._transforms = list(self._transforms)
        new._facet = self._facet
        new._coord = self._coord
        new._theme = self._theme
        new._layers = None if self._layers is None else list(self._layers)
        new._width = self._width
        new._height = self._height
        new._title = self._title
        new._description = self._description
        return new

    # ---- Marks (primitives) ----

    def _set_mark(self, name: str, **kwargs: Any) -> "Chart":
        m = MarkBase(name, **kwargs)
        new = self._clone()
        new._mark = name
        new._mark_kwargs = m.to_mark_kwargs_dict()
        return new

    def mark_point(self, **kwargs):  return self._set_mark("point", **kwargs)
    def mark_line(self, **kwargs):   return self._set_mark("line", **kwargs)
    def mark_bar(self, **kwargs):    return self._set_mark("bar", **kwargs)
    def mark_area(self, **kwargs):   return self._set_mark("area", **kwargs)
    def mark_rule(self, **kwargs):   return self._set_mark("rule", **kwargs)
    def mark_text(self, **kwargs):   return self._set_mark("text", **kwargs)
    def mark_tick(self, **kwargs):   return self._set_mark("tick", **kwargs)
    def mark_rect(self, **kwargs):   return self._set_mark("rect", **kwargs)

    # ---- Marks (statistical) ----

    def mark_density(self, **kwargs) -> "Chart":
        # Field comes from .encode(x=...) chain; call after .encode() typically
        x_field = self._encoding.get("x")
        if x_field is None:
            raise ValueError("mark_density() requires .encode(x=...) to specify the density field")
        field = x_field.field if isinstance(x_field, ChannelBase) else x_field
        mark, transforms, remap = desugar_density(field, **kwargs)
        new = self._clone()
        new._mark = mark
        new._transforms = list(self._transforms) + transforms
        # Remap encoding
        from ferrum.encoding import Y
        new._encoding["y"] = Y(remap["y"], type="Q")
        return new

    def mark_histogram(self, **kwargs) -> "Chart":
        x_field = self._encoding.get("x")
        if x_field is None:
            raise ValueError("mark_histogram() requires .encode(x=...)")
        field = x_field.field if isinstance(x_field, ChannelBase) else x_field
        mark, transforms, remap = desugar_histogram(field, **kwargs)
        new = self._clone()
        new._mark = mark
        new._transforms = list(self._transforms) + transforms
        from ferrum.encoding import X, X2, Y
        new._encoding["x"] = X(remap["x"], type="Q")
        new._encoding["x2"] = X2(remap["x2"], type="Q")
        new._encoding["y"] = Y(remap["y"], type="Q")
        return new

    def mark_smooth(self, **kwargs) -> "Chart":
        x_enc = self._encoding.get("x")
        y_enc = self._encoding.get("y")
        if x_enc is None or y_enc is None:
            raise ValueError("mark_smooth() requires .encode(x=..., y=...)")
        x_field = x_enc.field if isinstance(x_enc, ChannelBase) else x_enc
        y_field = y_enc.field if isinstance(y_enc, ChannelBase) else y_enc
        mark, transforms, remap = desugar_smooth(x_field, y_field, **kwargs)
        new = self._clone()
        new._mark = mark
        new._transforms = list(self._transforms) + transforms
        return new

    # ---- Marks (deferred) ----

    def mark_boxplot(self, **kwargs):       raise deferred_mark_error("boxplot")
    def mark_errorbar(self, **kwargs):      raise deferred_mark_error("errorbar")
    def mark_errorband(self, **kwargs):     raise deferred_mark_error("errorband")
    def mark_ribbon(self, **kwargs):        raise deferred_mark_error("ribbon")
    def mark_contour(self, **kwargs):       raise deferred_mark_error("contour")
    def mark_violin(self, **kwargs):        raise deferred_mark_error("violin")
    def mark_qq(self, **kwargs):            raise deferred_mark_error("qq")
    def mark_raster(self, **kwargs):        raise deferred_mark_error("raster")
    def mark_swarm(self, **kwargs):         raise deferred_mark_error("swarm")
    def mark_hex(self, **kwargs):           raise deferred_mark_error("hex")
    def mark_function(self, fn, **kwargs):  raise deferred_mark_error("function")
    def mark_arc(self, **kwargs):           raise deferred_mark_error("arc")
    def mark_image(self, **kwargs):         raise deferred_mark_error("image")
    def mark_geoshape(self, **kwargs):      raise deferred_mark_error("geoshape")
    def mark_segment(self, **kwargs):       raise deferred_mark_error("segment")
    def mark_label(self, **kwargs):         raise deferred_mark_error("label")

    # ---- Encoding ----

    def encode(self, **channels: Any) -> "Chart":
        new = self._clone()
        for name, value in channels.items():
            cls = _channel_class_for(name)
            if cls is None:
                raise ValueError(f"unknown encoding channel: {name!r}")

            if isinstance(value, ChannelBase):
                channel = value
            elif isinstance(value, str):
                field, type_, agg = parse_shorthand(value)
                kw = {}
                if type_: kw["type"] = type_
                if agg: kw["aggregate"] = agg
                channel = cls(field, **kw)
            else:
                raise TypeError(
                    f"encode({name}=...) expects str or {cls.__name__} instance, "
                    f"got {type(value).__name__}"
                )

            new._encoding[name] = channel
            new._transforms.extend(channel.to_implicit_transforms())
        return new

    def transform(self, *transforms) -> "Chart":
        new = self._clone()
        new._transforms = list(self._transforms) + list(transforms)
        return new

    # ---- Properties ----

    def properties(self, *, width=None, height=None, title=None, description=None) -> "Chart":
        new = self._clone()
        if width is not None: new._width = width
        if height is not None: new._height = height
        if title is not None: new._title = title
        if description is not None: new._description = description
        return new

    # ---- Output (stubs; implemented in Task 32+) ----

    def to_spec(self):
        from ferrum import ChartSpec
        # Build the EncodingSpec arguments for each registered channel
        kw = {"mark": self._mark or "point", "data": "default"}
        for axis in ("x", "y", "color"):
            if axis in self._encoding:
                ch = self._encoding[axis]
                # For Phase 8a, pass field name (Phase 7 ChartSpec accepts str OR EncodingSpec)
                kw[axis] = ch.field if ch.field is not None else ""
        if self._transforms:
            kw["transforms"] = list(self._transforms)
        return ChartSpec(**kw)

    def to_json(self, *, indent=None) -> str:
        spec = self.to_spec()
        return spec.to_json()

    def show_svg(self) -> str:
        # Stub — full impl in Task 32
        from ferrum._core import render_svg
        spec = self.to_spec()
        data = to_arrow_table(self._data)
        viewport = (self._width or 600.0, self._height or 400.0)
        theme_dict = (self._theme.to_theme_inputs_dict() if self._theme else {})
        return render_svg(spec, data, viewport=viewport, theme=theme_dict)

    def show_png(self) -> bytes:
        from ferrum._core import render_png
        spec = self.to_spec()
        data = to_arrow_table(self._data)
        viewport = (self._width or 600.0, self._height or 400.0)
        theme_dict = (self._theme.to_theme_inputs_dict() if self._theme else {})
        return render_png(spec, data, viewport=viewport, theme=theme_dict)

    # Stubs for Phase 11
    def add_selection(self, *selections):
        raise NotImplementedError("selections require .interactive() — Phase 11")

    def interactive(self):
        raise NotImplementedError("interactive renderer — Phase 11")

    def __repr__(self) -> str:
        return f"Chart(mark={self._mark!r}, encoding={list(self._encoding.keys())})"
```

- [ ] **Step 3: Re-export from `__init__.py`**

In `src/ferrum/__init__.py`:

```python
from ferrum.chart import Chart
__all__ += ["Chart"]
```

- [ ] **Step 4: Run tests**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
uv run pytest tests/test_chart.py -v
```

Expected: most chart tests PASS. The `to_spec` round-trip test may need adjustment based on which encoding kwargs the current Phase 7 `ChartSpec` constructor accepts; iterate as needed.

- [ ] **Step 5: Commit**

```bash
git add src/ferrum/chart.py src/ferrum/__init__.py tests/test_chart.py
git commit -m "feat(chart): Chart class core with immutability + mark/encode/properties"
```

---

### Task 28: `chart.py` — facet, coord, theme methods

**Files:**
- Modify: `src/ferrum/chart.py`
- Test: `tests/test_facet.py`, `tests/test_coord.py`, extend `tests/test_theme.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_facet.py`:

```python
import polars as pl

from ferrum import Chart


def test_facet_with_col_only():
    df = pl.DataFrame({"a": [1, 2, 3], "species": ["s1", "s2", "s1"]})
    c = Chart(df).mark_point().encode(x="a", y="a").facet(col="species")
    assert c._facet is not None
    assert c._facet["field"] == "species"


def test_facet_with_row_and_col_grid():
    df = pl.DataFrame({"a": [1], "year": ["2024"], "species": ["s1"]})
    c = Chart(df).mark_point().encode(x="a", y="a").facet(row="year", col="species")
    assert c._facet is not None
    # grid mode produces a different shape than wrap; assert mode-distinguishing field
    assert c._facet.get("mode_kind") == "grid"
```

Create `tests/test_coord.py`:

```python
import pytest
import polars as pl

from ferrum import Chart, CoordFlip


def test_coord_flip_sets_chartspec_coord():
    df = pl.DataFrame({"a": [1], "b": [2]})
    c = Chart(df).mark_bar().encode(x="a", y="b").coord(CoordFlip())
    assert c._coord == "flip"


def test_coord_other_kinds_raise_notimplemented():
    from ferrum.coord import CoordPolar, CoordGeo, CoordFixed, CoordCartesian
    df = pl.DataFrame({"a": [1]})
    chart = Chart(df).mark_point().encode(x="a", y="a")
    for cls in (CoordPolar, CoordGeo, CoordFixed, CoordCartesian):
        with pytest.raises(NotImplementedError, match="Phase 9"):
            cls()  # constructors raise immediately
```

Append to `tests/test_theme.py`:

```python
def test_chart_theme_attaches_theme_to_chart():
    import polars as pl
    from ferrum import Chart
    from ferrum.themes import dark
    df = pl.DataFrame({"a": [1], "b": [2]})
    c = Chart(df).mark_point().encode(x="a", y="b").theme(dark)
    assert c._theme is dark


def test_chart_theme_per_chart_overrides_default(monkeypatch):
    import polars as pl
    from ferrum import Chart, set_default_theme
    from ferrum.themes import dark, minimal, get_default_theme
    df = pl.DataFrame({"a": [1], "b": [2]})
    with set_default_theme(dark):
        c = Chart(df).mark_point().encode(x="a", y="b").theme(minimal)
        # When show_svg is called, c's theme (minimal) is used, not dark.
        assert c._theme is minimal
        assert get_default_theme() is dark
```

Run: `uv run pytest tests/test_facet.py tests/test_coord.py tests/test_theme.py -v`
Expected: tests fail (`facet/coord/theme` methods don't exist on Chart).

- [ ] **Step 2: Add methods to Chart**

In `src/ferrum/chart.py`, add inside the `Chart` class:

```python
    def facet(self, field=None, *, row=None, col=None, ncols=None, nrows=None) -> "Chart":
        new = self._clone()
        if field is not None:
            mode_kind = "wrap"
            new._facet = {"field": field, "mode_kind": mode_kind, "ncols": ncols, "nrows": nrows}
        elif row is not None and col is not None:
            new._facet = {"row": row, "col": col, "mode_kind": "grid"}
        elif col is not None:
            new._facet = {"field": col, "mode_kind": "wrap", "ncols": ncols}
        elif row is not None:
            new._facet = {"field": row, "mode_kind": "wrap", "nrows": nrows}
        else:
            raise ValueError("facet() requires either `field=`, or `row=`/`col=`")
        return new

    def theme(self, theme) -> "Chart":
        new = self._clone()
        new._theme = theme
        return new

    def coord(self, coord) -> "Chart":
        from ferrum.coord import CoordFlip
        new = self._clone()
        if isinstance(coord, CoordFlip):
            new._coord = "flip"
        else:
            raise TypeError(f"unsupported coord: {type(coord).__name__}; only CoordFlip in Phase 8a")
        return new
```

Update `to_spec()` in the same file to include facet/coord:

```python
    def to_spec(self):
        from ferrum import ChartSpec
        kw = {"mark": self._mark or "point", "data": "default"}
        for axis in ("x", "y", "color"):
            if axis in self._encoding:
                ch = self._encoding[axis]
                kw[axis] = ch.field if ch.field is not None else ""
        if self._transforms:
            kw["transforms"] = list(self._transforms)
        # Phase 7 ChartSpec doesn't currently accept facet/coord/layers/mark_style as kwargs;
        # this requires extending the PyO3 binding. For Phase 8a, those flow via
        # JSON round-trip until the binding is widened — see Task 34.
        return ChartSpec(**kw)
```

- [ ] **Step 3: Run tests + commit**

```bash
uv run pytest tests/test_facet.py tests/test_coord.py tests/test_theme.py -v
git add src/ferrum/chart.py tests/test_facet.py tests/test_coord.py tests/test_theme.py
git commit -m "feat(chart): facet/theme/coord fluent methods"
```

---

### Task 29: `composition.py` + Chart `__add__`/`__or__`/`__and__`

**Files:**
- Create: `src/ferrum/composition.py`
- Create: `src/ferrum/layer.py`
- Modify: `src/ferrum/chart.py` (add operators)
- Test: `tests/test_composition.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_composition.py`:

```python
import warnings

import polars as pl
import pytest

from ferrum import Chart


@pytest.fixture
def df():
    return pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})


def test_layer_same_data_produces_layered_chart(df):
    c1 = Chart(df).mark_point().encode(x="a", y="b")
    c2 = Chart(df).mark_line().encode(x="a", y="b")
    layered = c1 + c2
    # Same data → wrapped layer ChartSpec, not HConcat
    assert layered._layers is not None
    assert len(layered._layers) == 2


def test_layer_different_data_falls_through_to_hconcat(df):
    df2 = pl.DataFrame({"a": [10], "b": [20]})
    c1 = Chart(df).mark_point().encode(x="a", y="b")
    c2 = Chart(df2).mark_line().encode(x="a", y="b")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = c1 + c2
    # Falls through to HConcat
    from ferrum.composition import HConcatChart
    assert isinstance(result, HConcatChart)
    assert any("differing data" in str(wi.message) for wi in w)


def test_hconcat_two_charts(df):
    c1 = Chart(df).mark_point().encode(x="a", y="b")
    c2 = Chart(df).mark_line().encode(x="a", y="b")
    result = c1 | c2
    from ferrum.composition import HConcatChart
    assert isinstance(result, HConcatChart)
    assert len(result.charts) == 2


def test_vconcat_two_charts(df):
    c1 = Chart(df).mark_point().encode(x="a", y="b")
    c2 = Chart(df).mark_line().encode(x="a", y="b")
    result = c1 & c2
    from ferrum.composition import VConcatChart
    assert isinstance(result, VConcatChart)


def test_operator_precedence_and_tighter_than_or(df):
    a = Chart(df).mark_point().encode(x="a", y="b")
    b = Chart(df).mark_line().encode(x="a", y="b")
    c = Chart(df).mark_bar().encode(x="a", y="b")
    # a | b & c should parse as a | (b & c), not (a | b) & c
    result = a | b & c
    from ferrum.composition import HConcatChart, VConcatChart
    assert isinstance(result, HConcatChart)
    # Inner (b & c) is the second item
    assert isinstance(result.charts[1], VConcatChart)


def test_explicit_parens_overrides_precedence(df):
    a = Chart(df).mark_point().encode(x="a", y="b")
    b = Chart(df).mark_line().encode(x="a", y="b")
    c = Chart(df).mark_bar().encode(x="a", y="b")
    result = (a | b) & c
    from ferrum.composition import VConcatChart
    assert isinstance(result, VConcatChart)
```

Run: expected ALL FAIL.

- [ ] **Step 2: Implement `composition.py`**

```python
"""Composition wrappers: HConcatChart, VConcatChart, LayerChart."""
from __future__ import annotations

from typing import List


class _CompositeBase:
    """Base for HConcat/VConcat. Holds a list of children + spacing."""

    def __init__(self, charts: List, *, spacing: float = 10.0) -> None:
        self.charts = list(charts)
        self.spacing = spacing

    def __or__(self, other):
        return HConcatChart([self, other])

    def __and__(self, other):
        return VConcatChart([self, other])


class HConcatChart(_CompositeBase):
    def show_svg(self) -> str:
        from ferrum._core import compose_svg_horizontal
        svgs = [c.show_svg() for c in self.charts]
        return compose_svg_horizontal(svgs, spacing=self.spacing, align="top")

    def show_png(self) -> bytes:
        # Render the composed SVG and then rasterize via render_png on the SVG bytes
        # — but render_png expects a ChartSpec. For Phase 8a, do SVG → PNG via a
        # round-trip through resvg, exposed as a separate Rust helper if needed.
        # Simpler stop-gap: use resvg directly via a Python helper, OR raise.
        raise NotImplementedError(
            "HConcatChart.show_png not yet wired in Phase 8a; use .save('out.png') "
            "after expanding the binding to accept SVG strings (Phase 8a follow-up)."
        )

    def save(self, path: str, *, format=None, **kwargs):
        from pathlib import Path
        path = Path(path)
        fmt = format or path.suffix.lstrip(".")
        if fmt == "svg":
            path.write_text(self.show_svg())
        else:
            raise NotImplementedError(f"HConcatChart.save({fmt!r}) not yet supported in Phase 8a")

    def show(self):
        # Delegate to display.py logic in Task 33; for now just print
        print(self.show_svg())

    def _repr_svg_(self) -> str:
        return self.show_svg()


class VConcatChart(_CompositeBase):
    def show_svg(self) -> str:
        from ferrum._core import compose_svg_vertical
        svgs = [c.show_svg() for c in self.charts]
        return compose_svg_vertical(svgs, spacing=self.spacing, align="left")

    def show_png(self) -> bytes:
        raise NotImplementedError(
            "VConcatChart.show_png not yet wired in Phase 8a; use .save('out.svg') instead."
        )

    def save(self, path: str, *, format=None, **kwargs):
        from pathlib import Path
        path = Path(path)
        fmt = format or path.suffix.lstrip(".")
        if fmt == "svg":
            path.write_text(self.show_svg())
        else:
            raise NotImplementedError(f"VConcatChart.save({fmt!r}) not yet supported in Phase 8a")

    def show(self):
        print(self.show_svg())

    def _repr_svg_(self) -> str:
        return self.show_svg()
```

- [ ] **Step 3: Add operators to Chart**

In `src/ferrum/chart.py`, add inside the `Chart` class:

```python
    def __add__(self, other: "Chart") -> "Chart":
        if not isinstance(other, Chart):
            return NotImplemented

        # Same data → multi-layer; different data → fall through to hconcat
        same_data = self._data is other._data
        if not same_data:
            try:
                # Try pyarrow equality if both can be coerced
                a = to_arrow_table(self._data)
                b = to_arrow_table(other._data)
                same_data = a.equals(b)
            except Exception:
                same_data = False

        if not same_data:
            import warnings
            warnings.warn(
                "Layered charts with differing data render as horizontal concatenation. "
                "Use a shared DataFrame for true overlay.",
                UserWarning, stacklevel=2,
            )
            return self.__or__(other)

        # Same data — build a multi-layer chart
        new = self._clone()
        new._layers = [
            {"mark": self._mark, "encoding": self._encoding,
             "transforms": self._transforms, "mark_style": self._mark_kwargs},
            {"mark": other._mark, "encoding": other._encoding,
             "transforms": other._transforms, "mark_style": other._mark_kwargs},
        ]
        # Warn if right-side has conflicting theme/facet/coord
        if (other._theme is not None and other._theme != self._theme) \
           or other._facet != self._facet or other._coord != self._coord:
            import warnings
            warnings.warn(
                "Layered chart `+`: secondary layer's theme/facet/coord is ignored; "
                "primary layer wins.",
                UserWarning, stacklevel=2,
            )
        return new

    def __or__(self, other: "Chart") -> "HConcatChart":
        from ferrum.composition import HConcatChart
        return HConcatChart([self, other])

    def __and__(self, other: "Chart") -> "VConcatChart":
        from ferrum.composition import VConcatChart
        return VConcatChart([self, other])
```

Add `from ferrum._coerce import to_arrow_table` at top of chart.py.

- [ ] **Step 4: Re-export composition + Layer**

```python
# src/ferrum/layer.py
"""Layer value class — used internally by Chart.__add__."""
from __future__ import annotations


class Layer:
    """Internal layer wrapper; users construct Layer rarely."""
    def __init__(self, data=None, mark=None, *, encoding=None, transforms=None):
        self.data = data
        self.mark = mark
        self.encoding = encoding or {}
        self.transforms = transforms or []
```

In `src/ferrum/__init__.py`:

```python
from ferrum.layer import Layer
from ferrum.composition import HConcatChart, VConcatChart
__all__ += ["Layer", "HConcatChart", "VConcatChart"]
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_composition.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/ferrum/composition.py src/ferrum/layer.py \
        src/ferrum/chart.py src/ferrum/__init__.py tests/test_composition.py
git commit -m "feat(composition): +/|/& operators with mixed-data fallthrough"
```

---

## Group I — Python annotations, coord, display

### Task 30: `annotations.py` — annotate_hline / vline / rect / text

**Files:**
- Create: `src/ferrum/annotations.py`
- Test: `tests/test_annotations.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_annotations.py`:

```python
import polars as pl

from ferrum import Chart
from ferrum.annotations import annotate_hline, annotate_vline, annotate_rect, annotate_text


def test_annotate_hline_returns_chart_with_rule_mark():
    h = annotate_hline(0)
    assert h._mark == "rule"


def test_annotate_vline_returns_chart_with_rule_mark():
    v = annotate_vline(5)
    assert v._mark == "rule"


def test_annotate_rect_returns_chart_with_rect_mark():
    r = annotate_rect(0, 1, 0, 1, opacity=0.1)
    assert r._mark == "rect"


def test_annotate_text_returns_chart_with_text_mark():
    t = annotate_text(1.0, 2.0, "hi")
    assert t._mark == "text"


def test_annotate_hline_can_be_added_to_scatter():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    scatter = Chart(df).mark_point().encode(x="a", y="b")
    composed = scatter + annotate_hline(5)
    # annotate_hline uses different data → falls through to hconcat with warning,
    # OR uses an inline 1-row table that matches the chart's column shape.
    # Phase 8a impl: annotate_* return charts with empty data; the + path
    # detects "same data" check fails → hconcat fallback. That's acceptable for 8a;
    # Phase 9 will improve via a shared-data resolver.
    # For this test, just assert no exception raised.
    # (Stricter assertions can be added once the resolver exists.)
```

Run: expected fail.

- [ ] **Step 2: Implement annotations**

```python
"""Lightweight annotation helpers — sugar over primitive marks."""
from __future__ import annotations

from typing import Optional

import polars as pl

from ferrum.chart import Chart


def annotate_hline(y: float, *, label: Optional[str] = None,
                   stroke: Optional[str] = None, stroke_dash=None) -> Chart:
    """Horizontal reference line at y. Returns a single-mark Chart."""
    df = pl.DataFrame({"_y": [y]})
    kwargs = {}
    if stroke is not None: kwargs["stroke"] = stroke
    if stroke_dash is not None: kwargs["stroke_dash"] = stroke_dash
    return Chart(df).mark_rule(**kwargs).encode(y="_y")


def annotate_vline(x: float, *, label: Optional[str] = None,
                   stroke: Optional[str] = None, stroke_dash=None) -> Chart:
    """Vertical reference line at x."""
    df = pl.DataFrame({"_x": [x]})
    kwargs = {}
    if stroke is not None: kwargs["stroke"] = stroke
    if stroke_dash is not None: kwargs["stroke_dash"] = stroke_dash
    return Chart(df).mark_rule(**kwargs).encode(x="_x")


def annotate_rect(x1: float, x2: float, y1: float, y2: float, *,
                  fill: Optional[str] = None, opacity: float = 0.1,
                  label: Optional[str] = None) -> Chart:
    """Shaded rectangle region between (x1, y1) and (x2, y2)."""
    df = pl.DataFrame({"_x1": [x1], "_x2": [x2], "_y1": [y1], "_y2": [y2]})
    kwargs = {"opacity": opacity}
    if fill is not None: kwargs["fill"] = fill
    return Chart(df).mark_rect(**kwargs).encode(x="_x1", y="_y1")
    # Phase 8a: x2/y2 are accepted-and-deferred channels; this annotation produces
    # a degenerate rect at (x1, y1) until the renderer honors X2/Y2 (Phase 9).


def annotate_text(x: float, y: float, text: str, *, dx: float = 0, dy: float = 0,
                  align: str = "center", baseline: str = "middle",
                  font_size: Optional[float] = None, color: Optional[str] = None,
                  angle: Optional[float] = None) -> Chart:
    """Free text annotation at (x, y)."""
    df = pl.DataFrame({"_x": [x], "_y": [y], "_text": [text]})
    kwargs = {"dx": dx, "dy": dy, "align": align, "baseline": baseline}
    if font_size is not None: kwargs["font_size"] = font_size
    if color is not None: kwargs["fill"] = color
    if angle is not None: kwargs["angle"] = angle
    return Chart(df).mark_text(**kwargs).encode(x="_x", y="_y")
    # Text channel is accepted-and-deferred; the actual text content goes via
    # mark_kwargs once rendered (Phase 9 wires Text channel properly).
```

- [ ] **Step 3: Re-export**

In `src/ferrum/__init__.py`:

```python
from ferrum.annotations import annotate_hline, annotate_vline, annotate_rect, annotate_text
__all__ += ["annotate_hline", "annotate_vline", "annotate_rect", "annotate_text"]
```

- [ ] **Step 4: Run tests + commit**

```bash
uv run pytest tests/test_annotations.py -v
git add src/ferrum/annotations.py src/ferrum/__init__.py tests/test_annotations.py
git commit -m "feat(annotations): annotate_hline/vline/rect/text as sugar over primitive marks"
```

---

### Task 31: `coord.py` — `CoordFlip` + NotImplementedError stubs

**Files:**
- Create: `src/ferrum/coord.py`

- [ ] **Step 1: Implement**

```python
"""Coordinate systems. Phase 8a ships CoordFlip only.
CoordCartesian/Polar/Geo/Fixed raise NotImplementedError pointing to Phase 9+."""
from __future__ import annotations


class CoordFlip:
    """Swap X and Y axis roles. Useful for horizontal bar charts."""
    def __repr__(self) -> str:
        return "CoordFlip()"


class _DeferredCoord:
    """Base for deferred coord systems."""
    _phase = "Phase 9+"
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} is planned for {self._phase}. "
            f"Phase 8a ships CoordFlip only."
        )


class CoordCartesian(_DeferredCoord):
    pass


class CoordPolar(_DeferredCoord):
    pass


class CoordGeo(_DeferredCoord):
    pass


class CoordFixed(_DeferredCoord):
    pass
```

- [ ] **Step 2: Re-export**

```python
# src/ferrum/__init__.py
from ferrum.coord import CoordFlip, CoordCartesian, CoordPolar, CoordGeo, CoordFixed
__all__ += ["CoordFlip", "CoordCartesian", "CoordPolar", "CoordGeo", "CoordFixed"]
```

- [ ] **Step 3: Tests already exist (in Task 28's test_coord.py); run + commit**

```bash
uv run pytest tests/test_coord.py -v
git add src/ferrum/coord.py src/ferrum/__init__.py
git commit -m "feat(coord): CoordFlip in 8a; CoordCartesian/Polar/Geo/Fixed deferred"
```

---

### Task 32: `display.py` — `save()` + format dispatch

**Files:**
- Create: `src/ferrum/display.py`
- Modify: `src/ferrum/chart.py` (add save method, delegate to display)
- Test: `tests/test_show_save.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_show_save.py`:

```python
import tempfile
from pathlib import Path

import polars as pl
import pytest

from ferrum import Chart


@pytest.fixture
def chart():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    return Chart(df).mark_point().encode(x="a", y="b")


def test_save_svg(chart, tmp_path):
    out = tmp_path / "out.svg"
    chart.save(out)
    text = out.read_text()
    assert text.startswith("<svg") or text.startswith("<?xml")


def test_save_png(chart, tmp_path):
    out = tmp_path / "out.png"
    chart.save(out)
    bytes_ = out.read_bytes()
    assert bytes_.startswith(b"\x89PNG\r\n\x1a\n")


def test_save_html_raises_not_implemented(chart, tmp_path):
    with pytest.raises(NotImplementedError, match="html"):
        chart.save(tmp_path / "out.html")


def test_save_json_raises_not_implemented(chart, tmp_path):
    with pytest.raises(NotImplementedError, match="json"):
        chart.save(tmp_path / "out.json")


def test_save_unknown_extension_raises(chart, tmp_path):
    with pytest.raises(ValueError, match="extension"):
        chart.save(tmp_path / "out.weird")


def test_save_explicit_format_overrides_extension(chart, tmp_path):
    out = tmp_path / "out.txt"
    chart.save(out, format="svg")
    text = out.read_text()
    assert "<svg" in text or "<?xml" in text
```

Run: expected fail.

- [ ] **Step 2: Implement display.py**

```python
"""Output orchestration: save, show, _repr_*_."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from ferrum.chart import Chart


def save_chart(chart: "Chart", path: Union[str, Path], *,
               format: str | None = None, **render_kwargs) -> None:
    """Save chart to disk. Format inferred from extension when format=None."""
    path = Path(path)
    fmt = format or path.suffix.lstrip(".").lower()
    if fmt == "svg":
        path.write_text(chart.show_svg())
    elif fmt == "png":
        path.write_bytes(chart.show_png())
    elif fmt in ("html", "json"):
        raise NotImplementedError(f"save({fmt!r}) is planned for Phase 9. "
                                   f"Use 'svg' or 'png' in Phase 8a.")
    elif fmt == "":
        raise ValueError(f"save({str(path)!r}) requires a format= or a path with extension.")
    else:
        raise ValueError(
            f"unknown extension {fmt!r}; supported: svg, png. "
            f"(html, json planned for Phase 9.)"
        )


def show_chart(chart: "Chart") -> None:
    """Display chart. Order: Jupyter inline → browser fallback."""
    if _is_jupyter():
        try:
            from IPython.display import display, SVG
            display(SVG(chart.show_svg()))
            return
        except Exception:
            pass
    # Browser fallback: write temp HTML, open in browser
    import tempfile, webbrowser
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        f.write(_wrap_svg_in_html(chart.show_svg(), title=chart._title or "Ferrum chart"))
        url = f"file://{f.name}"
    webbrowser.open(url)


def _is_jupyter() -> bool:
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ in ("ZMQInteractiveShell", "TerminalInteractiveShell")
    except ImportError:
        return False


def _wrap_svg_in_html(svg: str, *, title: str = "Ferrum chart") -> str:
    return (
        f"<!doctype html><html><head><title>{title}</title></head>"
        f"<body style='margin:0;padding:20px;font-family:sans-serif'>"
        f"<h2>{title}</h2>{svg}</body></html>"
    )
```

- [ ] **Step 3: Wire into Chart**

In `src/ferrum/chart.py`, add:

```python
    def save(self, path, *, format=None, **render_kwargs) -> None:
        from ferrum.display import save_chart
        save_chart(self, path, format=format, **render_kwargs)

    def show(self) -> None:
        from ferrum.display import show_chart
        show_chart(self)

    def _repr_svg_(self) -> str:
        try:
            return self.show_svg()
        except Exception:
            return None  # let Jupyter fall back to __repr__

    def _repr_html_(self) -> str | None:
        # Returning the SVG is acceptable for Jupyter HTML representation
        try:
            return f"<div>{self.show_svg()}</div>"
        except Exception:
            return None
```

- [ ] **Step 4: Run tests + commit**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
uv run pytest tests/test_show_save.py -v
git add src/ferrum/display.py src/ferrum/chart.py tests/test_show_save.py
git commit -m "feat(display): save (svg/png) + show (Jupyter+browser fallback) + Jupyter rich display"
```

---

### Task 33: `display.py` — `_repr_html_` polish + show error paths

This task is mostly verification of Task 32's display logic in non-Jupyter and Jupyter environments.

- [ ] **Step 1: Add tests for show fallback**

Append to `tests/test_show_save.py`:

```python
def test_show_in_non_jupyter_opens_browser(chart, monkeypatch):
    """When not in Jupyter, .show() writes a temp HTML and calls webbrowser.open."""
    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))
    monkeypatch.setattr("ferrum.display._is_jupyter", lambda: False)
    chart.show()
    assert len(opened) == 1
    assert opened[0].startswith("file://")
    assert opened[0].endswith(".html")


def test_repr_svg_returns_string_for_jupyter(chart):
    s = chart._repr_svg_()
    assert s is not None
    assert "<svg" in s or "<?xml" in s


def test_repr_html_returns_div_wrapped_svg(chart):
    s = chart._repr_html_()
    assert s is not None
    assert s.startswith("<div>")
```

- [ ] **Step 2: Run + commit**

```bash
uv run pytest tests/test_show_save.py -v
git add tests/test_show_save.py
git commit -m "test(display): show browser fallback + Jupyter rich-display path"
```

---

## Group J — Wiring, broader tests, docs, verification

### Task 34: Widen Rust ChartSpec PyO3 binding to accept facet/coord/layers/mark_style kwargs

**Files:**
- Modify: `crates/ferrum-core/src/spec/chart.rs` (extend `#[new]` keyword args)
- Modify: `src/ferrum/chart.py::to_spec` (pass new kwargs through)
- Modify: `src/ferrum/_core.pyi` (extend ChartSpec.__init__ signature)

This is the bridge that exposes the Group A Rust extensions (Tasks 1, 4, 5) to Python's `ChartSpec(...)` constructor — so Chart's `to_spec()` can pass `facet=`, `coord=`, `layers=`, `mark_style=` directly.

- [ ] **Step 1: Extend `ChartSpec.__new__` signature**

The Rust `#[new]` was extended in Tasks 1, 4, 5 to take `layers`, `coord`, `mark_style` as `Option<&Bound<'_, PyAny>>` parameters. Add `facet: Option<&Bound<'_, PyAny>> = None` if not already present, and parse it analogously:

```rust
let facet = match facet {
    None => None,
    Some(obj) => {
        let py = obj.py();
        let json_module = py.import("json")?;
        let s: String = json_module.call_method1("dumps", (obj,))?.extract()?;
        Some(serde_json::from_str(&s).map_err(|e| PyValueError::new_err(format!("facet: {e}")))?)
    }
};
```

- [ ] **Step 2: Update Python Chart.to_spec**

```python
    def to_spec(self):
        from ferrum import ChartSpec
        kw = {"mark": self._mark or "point", "data": "default"}
        for axis in ("x", "y", "color"):
            if axis in self._encoding:
                ch = self._encoding[axis]
                kw[axis] = ch.field if ch.field is not None else ""
        if self._transforms:
            kw["transforms"] = list(self._transforms)
        if self._facet:
            # Convert Python facet dict to the Rust FacetSpec JSON shape
            kw["facet"] = self._build_facet_dict()
        if self._coord:
            kw["coord"] = self._coord  # str: "flip" or "cartesian"
        if self._layers:
            kw["layers"] = self._build_layers_list()
        if self._mark_kwargs:
            kw["mark_style"] = dict(self._mark_kwargs)
        return ChartSpec(**kw)

    def _build_facet_dict(self) -> dict:
        f = self._facet
        if f.get("mode_kind") == "wrap":
            return {"field": f["field"], "mode": {"wrap": {"ncols": f.get("ncols") or 3}}}
        elif f.get("mode_kind") == "grid":
            return {"field": f.get("col") or f.get("row"),
                    "mode": {"grid": {"row": f.get("row"), "col": f.get("col")}}}
        raise ValueError(f"unknown facet mode_kind: {f.get('mode_kind')}")

    def _build_layers_list(self) -> list:
        out = []
        for lyr in self._layers:
            d = {"mark": lyr["mark"], "encoding": {}, "transforms": []}
            for axis in ("x", "y", "color", "size", "shape", "opacity"):
                if axis in lyr["encoding"]:
                    ch = lyr["encoding"][axis]
                    d["encoding"][axis] = ch.to_encoding_spec_dict() if hasattr(ch, "to_encoding_spec_dict") else ch
            d["transforms"] = lyr.get("transforms", [])
            if lyr.get("mark_style"):
                d["mark_style"] = lyr["mark_style"]
            out.append(d)
        return out
```

> The exact `FacetSpec` JSON shape comes from `crates/ferrum-core/src/layout/facet.rs`. Verify by:
> ```bash
> grep -n "FacetSpec\|FacetMode" crates/ferrum-core/src/layout/facet.rs
> ```
> Adjust `_build_facet_dict` to match exactly what serde produces for `FacetSpec { field, mode: FacetMode::Wrap { ncols }, ... }`.

- [ ] **Step 3: Update _core.pyi**

```python
class ChartSpec:
    def __init__(
        self,
        *,
        mark: MarkStr,
        x: Union[str, EncodingSpec, None] = None,
        y: Union[str, EncodingSpec, None] = None,
        color: Union[str, EncodingSpec, None] = None,
        size: Union[str, EncodingSpec, None] = None,
        shape: Union[str, EncodingSpec, None] = None,
        opacity: Union[str, EncodingSpec, None] = None,
        data: Optional[str] = None,
        transforms: Optional[List[object]] = None,
        facet: Optional[dict] = None,
        layers: Optional[List[dict]] = None,
        coord: Optional[Literal["cartesian", "flip"]] = None,
        mark_style: Optional[dict] = None,
    ) -> None: ...
```

- [ ] **Step 4: Build + smoke test**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
uv run python -c "
import polars as pl
from ferrum import Chart, CoordFlip
df = pl.DataFrame({'a':[1,2,3],'b':[4,5,6]})
c = Chart(df).mark_bar().encode(x='a', y='b').coord(CoordFlip())
spec = c.to_spec()
print(spec.to_json())
"
```

Expected: JSON output includes `\"coord\":{\"kind\":\"flip\"}`.

- [ ] **Step 5: Commit**

```bash
git add crates/ferrum-core/src/spec/chart.rs src/ferrum/chart.py src/ferrum/_core.pyi
git commit -m "feat(binding): ChartSpec accepts facet/coord/layers/mark_style from Python"
```

---

### Task 35: Wire `__init__.py` public surface + update _core.pyi

**Files:**
- Modify: `src/ferrum/__init__.py`
- Modify: `src/ferrum/_core.pyi`

- [ ] **Step 1: Audit current `__init__.py`**

```bash
cat src/ferrum/__init__.py
```

Confirm all the new names from Tasks 12–33 are re-exported. Add any missing.

- [ ] **Step 2: Final `__init__.py` shape**

```python
"""Ferrum — a statistical visualization library with a Rust core."""

from ferrum._core import (
    Aggregate, AggregateOp, Bin, ChartSpec, EncodingSpec, Kde,
    LinearScale, LogScale, TimeScale, SymlogScale, OrdinalScale,
    QuantileScale, ThresholdScale, Smooth, Summary,
    compute_layout, process_batch, render_png, render_svg,
    compose_svg_horizontal, compose_svg_vertical,
)

# Phase 8a additions
from ferrum.chart import Chart
from ferrum.layer import Layer
from ferrum.composition import HConcatChart, VConcatChart
from ferrum.coord import CoordFlip, CoordCartesian, CoordPolar, CoordGeo, CoordFixed
from ferrum.themes import (
    Theme, set_default_theme, get_default_theme, theme_context,
)
import ferrum.themes as themes
import ferrum.encoding as encoding
from ferrum.encoding import (
    X, Y, X2, Y2, XError, YError, XError2, YError2, Theta, Radius,
    Color, Fill, Stroke, Opacity, FillOpacity, StrokeOpacity,
    StrokeWidth, StrokeDash, Size, Shape, Angle,
    Text, Detail, Tooltip, TooltipField, Href, Description, Key,
    Facet, FacetRow, FacetCol,
)
from ferrum.annotations import (
    annotate_hline, annotate_vline, annotate_rect, annotate_text,
)

__version__ = "0.1.0"

__all__ = [
    # Phase 1-7 core
    "Aggregate", "AggregateOp", "Bin", "ChartSpec", "EncodingSpec", "Kde",
    "LinearScale", "LogScale", "TimeScale", "SymlogScale", "OrdinalScale",
    "QuantileScale", "ThresholdScale", "Smooth", "Summary",
    "compute_layout", "process_batch", "render_png", "render_svg",
    "compose_svg_horizontal", "compose_svg_vertical",
    # Phase 8a
    "Chart", "Layer", "HConcatChart", "VConcatChart",
    "CoordFlip", "CoordCartesian", "CoordPolar", "CoordGeo", "CoordFixed",
    "Theme", "themes", "set_default_theme", "get_default_theme", "theme_context",
    "encoding",
    "X", "Y", "X2", "Y2", "XError", "YError", "XError2", "YError2",
    "Theta", "Radius",
    "Color", "Fill", "Stroke", "Opacity", "FillOpacity", "StrokeOpacity",
    "StrokeWidth", "StrokeDash", "Size", "Shape", "Angle",
    "Text", "Detail", "Tooltip", "TooltipField", "Href", "Description", "Key",
    "Facet", "FacetRow", "FacetCol",
    "annotate_hline", "annotate_vline", "annotate_rect", "annotate_text",
]
```

- [ ] **Step 2: Smoke test**

```bash
uv run python -c "
import ferrum
from ferrum import (Chart, Layer, Theme, set_default_theme, themes,
                    X, Y, Color, Size, Shape, Opacity, Facet,
                    annotate_hline, CoordFlip)
print('All imports OK')
print(f'ferrum.themes.dark = {themes.dark}')
"
```

Expected: `All imports OK` + a Theme repr.

- [ ] **Step 3: Commit**

```bash
git add src/ferrum/__init__.py src/ferrum/_core.pyi
git commit -m "feat(public-api): re-export full Phase 8a surface from ferrum.__init__"
```

---

### Task 36: End-to-end pytest suite — round out remaining test files

**Files:**
- Modify: all `tests/test_*.py` (extend to hit the full §8.2 spec test count)

- [ ] **Step 1: Audit current test count**

```bash
uv run pytest --collect-only 2>&1 | tail -5
```

Note current count.

- [ ] **Step 2: Add missing tests per spec §8.2**

Walk the spec §8.2 test list and add any missing tests to the appropriate test file. Specific gaps from the per-task tests:

In `tests/test_chart.py`, add:
```python
def test_chart_with_pandas_dataframe():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    c = Chart(df).mark_point().encode(x="a", y="b")
    spec = c.to_spec()
    assert spec.mark == "point"


def test_chart_immutability_chain_independence():
    """base.encode(x='a') and base.encode(x='b') are independent."""
    df = pl.DataFrame({"a": [1], "b": [2]})
    base = Chart(df).mark_point()
    ca = base.encode(x="a")
    cb = base.encode(x="b")
    assert ca._encoding["x"].field == "a"
    assert cb._encoding["x"].field == "b"
    # base unaffected
    assert base._encoding == {}
```

In `tests/test_encoding.py`, add the warn-once-across-renders test:
```python
def test_stroke_warns_once_across_multiple_renders():
    import polars as pl
    from ferrum import Chart, Stroke
    from ferrum._warn import reset_warnings

    reset_warnings()
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": ["x", "y"]})

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for _ in range(3):
            Chart(df).mark_point().encode(x="a", y="b", stroke=Stroke("c"))
    # Only 1 warning despite 3 constructions
    stroke_warnings = [wi for wi in w if "stroke" in str(wi.message).lower()]
    assert len(stroke_warnings) <= 1, f"got {len(stroke_warnings)}: {[str(wi.message) for wi in stroke_warnings]}"
```

In `tests/test_marks.py`, add the NotImplementedError tests:
```python
@pytest.mark.parametrize("method,phase", [
    ("mark_boxplot", "8b"),
    ("mark_violin", "8b"),
    ("mark_qq", "8b"),
    ("mark_arc", "9"),
])
def test_deferred_mark_methods_raise_with_phase_pointer(method, phase):
    import polars as pl
    df = pl.DataFrame({"a": [1]})
    c = Chart(df).encode(x="a", y="a")
    with pytest.raises(NotImplementedError, match=f"Phase {phase}"):
        getattr(c, method)()
```

In `tests/test_facet.py`, add the strip-title golden:
```python
def test_faceted_svg_contains_strip_titles():
    df = pl.DataFrame({
        "a": [1, 2, 3, 4, 5, 6],
        "b": [1, 2, 3, 4, 5, 6],
        "species": ["A", "A", "B", "B", "C", "C"],
    })
    svg = Chart(df).mark_point().encode(x="a", y="b").facet(col="species").show_svg()
    # Phase 7 strip-title implementation emits 3 text elements for 3 facets
    text_count = svg.count("<text")
    # Strip titles + axis labels: at least 3 strip titles
    assert text_count >= 3
```

- [ ] **Step 3: Run full pytest suite**

```bash
uv run pytest -v 2>&1 | tail -20
```

Expected: ≥ 179 tests pass (89 baseline + 90 from Phase 8a).

- [ ] **Step 4: Run cargo tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core 2>&1 | tail -5
```

Expected: ≥ 291 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/
git commit -m "test(phase-8a): round out pytest suite to 179+ tests covering full surface"
```

---

### Task 37: Update `ferrum-spec.md` with dated notes for deferrals

**Files:**
- Modify: `ferrum-spec.md` (§3.2, §3.13, §3.16, §3.18)

- [ ] **Step 1: Add dated note to §3.2**

After the channel tables, add:

```markdown
> **2026-05-10 (Phase 8a):** All 31 channel classes are constructible Python
> value objects. Renderer honors `x`, `y`, `color`, `size`, `shape`, `opacity`
> in Phase 8a. Other channels (Stroke, Fill, FillOpacity, StrokeOpacity,
> StrokeWidth, StrokeDash, Angle, Text, Detail, Tooltip, TooltipField, Href,
> Description, Key, X2, Y2, XError, YError, XError2, YError2, Theta, Radius)
> are accepted at the API and stored on `EncodingSpec`, but the renderer
> ignores them with a one-time `UserWarning` per (channel, render call).
> Phase 9 wires the remaining channels.
>
> Channel kwargs honored in 8a: `type`, `bin`, `aggregate`, `scale`, `title`.
> Other kwargs (`axis`, `legend`, `sort`, `stack`, `impute`, `scheme`, `format`,
> `formatType`) are accepted, stored typed on `EncodingSpec`, and warn-once.
```

- [ ] **Step 2: Add dated note to §3.13**

After the existing description of `set_default_theme`:

```markdown
> **2026-05-10 (Phase 8a):** `set_default_theme(theme)` is implemented as a
> contextvars-backed setter that returns a context manager. Per-chart
> `Chart.theme(t)` always overrides this default. CLAUDE.md §"Hard
> constraints" documents this as the single sanctioned exception to
> "no global mutable state."
```

- [ ] **Step 3: Add dated note to §3.16**

```markdown
> **2026-05-10 (Phase 8a):** `.show()` env detection in 8a covers Jupyter
> inline (`_repr_svg_` / `_repr_html_` rich display) and a browser fallback
> (writes temp HTML, calls `webbrowser.open`). Sixel terminal output and the
> standalone HTML wrapper output are deferred to Phase 9. `.save()` honors
> `.svg` and `.png` extensions; `.html` and `.json` raise
> `NotImplementedError` pointing to Phase 9.
```

- [ ] **Step 4: Add dated note to §3.18**

```markdown
> **2026-05-10 (Phase 8a):** Data input compatibility provided via narwhals
> (~1.x) for pandas, modin, cuDF, dask, ibis. Polars goes via direct CDI
> (zero-copy). pyarrow `Table` and `RecordBatch` accepted as native. Dict,
> list-of-records, and 2D numpy with auto-named columns supported. File path
> inputs (`Chart("file.csv")`) and `ModelSource`/`ComparedModelSource`
> deferred to Phases 9 and 10 respectively.
```

- [ ] **Step 5: Commit**

```bash
git add ferrum-spec.md
git commit -m "docs(spec): dated 2026-05-10 notes for Phase 8a deferrals (§3.2, §3.13, §3.16, §3.18)"
```

---

### Task 38: Update `ferrum-phases.md` + final verification

**Files:**
- Modify: `docs/superpowers/ferrum-phases.md`

- [ ] **Step 1: Mark Phase 8a as done; insert Phase 8b row**

In the Phase table, replace the row 8 with two rows:

```markdown
| **8a** | Grammar API surface (Python) — primitives + simple stats | `Chart`, `Layer`, all 31 encoding channels, themes-as-values, `+`/`\|`/`&` composition, `Facet`, `CoordFlip`, annotations, `mark_density/histogram/smooth` | 7 | [`2026-05-10-grammar-api-design.md`](specs/2026-05-10-grammar-api-design.md) | **done** |
| **8b** | Composite + heavy statistical marks | `mark_boxplot/errorbar/errorband/ribbon`, `mark_contour/violin/qq/raster/swarm/hex/function` + ~7 new Phase 5 transforms (Outliers, ErrorExtent, Contour, QQ, Raster, Hex, Swarm, BoxStats, Violin) + new SVG primitives (image, polygon, beeswarm) | 8a | *(not yet written)* | pending |
```

In the dependency arrow diagram, replace `8 →` with `8a → 8b →`.

In the Done criteria section, replace "Phase 8" with "Phase 8a" and add a "Phase 8b" subsection:

```markdown
### Phase 8a — Grammar API surface (primitives + simple stats)
- [x] `import ferrum; ferrum.Chart(data).mark_point().encode(x="col_a", y="col_b").show()` works
- [x] Layer composition (`+`), hstack (`|`), vstack (`&`) work
- [x] `Theme` objects are values passed to `Chart`. `set_default_theme()` is the sanctioned contextvars-backed exception (per CLAUDE.md)
- [x] No `matplotlib` in the dependency tree
- [x] All 31 encoding channel classes from `ferrum-spec.md §3.2` exist as Python value classes; renderer honors x/y/color/size/shape/opacity (others warn-once)
- [x] `mark_density`, `mark_histogram`, `mark_smooth` (without CI band) work over Phase 5 transforms

### Phase 8b — Composite + heavy statistical marks
- [ ] All 4 composite marks (`mark_boxplot`, `mark_errorbar`, `mark_errorband`, `mark_ribbon`) work
- [ ] All 7 heavy statistical marks (`mark_contour`, `mark_violin`, `mark_qq`, `mark_raster`, `mark_swarm`, `mark_hex`, `mark_function`) work
- [ ] New Phase 5 transforms (Outliers, ErrorExtent, Contour, QQ, Raster, Hex, Swarm, BoxStats, Violin) all have round-trip + correctness tests
- [ ] New SVG primitives in `SvgBuffer` (image, polygon, beeswarm) emit deterministic SVG
- [ ] `mark_smooth(ci=...)` CI band renders via the new ribbon mark
```

- [ ] **Step 2: Final verification**

```bash
# Full test suite
unset CONDA_PREFIX && uv run --no-sync maturin develop --release
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core
uv run pytest

# Verify the example from §1 of the spec actually runs
uv run python -c "
import polars as pl
import ferrum as fr

df = pl.DataFrame({
    'sepal_length': [5.1, 4.9, 7.0, 6.4, 6.3, 5.8],
    'sepal_width': [3.5, 3.0, 3.2, 3.2, 3.3, 2.7],
    'species': ['setosa', 'setosa', 'versicolor', 'versicolor', 'virginica', 'virginica'],
})

# Single-layer
svg = fr.Chart(df).mark_point().encode(
    x='sepal_length', y='sepal_width', color='species'
).show_svg()
assert svg.startswith('<svg') or svg.startswith('<?xml')
print('Single-layer OK')

# Multi-layer
points = fr.Chart(df).mark_point().encode(x='sepal_length', y='sepal_width', color='species')
fit = fr.Chart(df).mark_smooth().encode(x='sepal_length', y='sepal_width')
layered_svg = (points + fit).show_svg()
assert '<svg' in layered_svg
print('Multi-layer (+) OK')

# Concat
hist = fr.Chart(df).mark_histogram().encode(x='sepal_length')
kde = fr.Chart(df).mark_density().encode(x='sepal_length')
hconcat_svg = (hist | kde).show_svg()
assert '<svg' in hconcat_svg
print('HConcat (|) OK')

# Faceting
faceted_svg = fr.Chart(df).mark_point().encode(
    x='sepal_length', y='sepal_width'
).facet(col='species').show_svg()
assert '<svg' in faceted_svg
print('Facet OK')

# Theme as value
themed_svg = fr.Chart(df).mark_point().encode(
    x='sepal_length', y='sepal_width'
).theme(fr.themes.dark).show_svg()
assert '<svg' in themed_svg
print('Theme OK')

# No matplotlib
import importlib.util
assert importlib.util.find_spec('matplotlib') is None, 'matplotlib should not be installed'
print('No matplotlib OK')

print('\\nALL DONE-CRITERIA EXAMPLES PASS')
"
```

Expected: all 6 example sections print OK. If any fail, debug before marking Phase 8a done.

- [ ] **Step 3: Commit + final**

```bash
git add docs/superpowers/ferrum-phases.md
git commit -m "docs(phases): mark Phase 8a done; add Phase 8b row + done criteria"

# Optional: push branch (only if user explicitly asks)
# git push -u origin feat/phase-8a-grammar-api
```

- [ ] **Step 4: Open a PR (only if user asks)**

Phase 8a is mergeable. Per CLAUDE.md, `git push` requires explicit user request. When asked:

```bash
git push -u origin feat/phase-8a-grammar-api
gh pr create --title "Phase 8a — Grammar API surface (Python)" --body "$(cat <<'EOF'
## Summary
- Phase 8a ships the user-facing Python grammar API on top of Phase 7's renderer
- All 31 encoding channel classes; renderer honors x/y/color/size/shape/opacity; rest warn-once
- Composition operators (`+`/`|`/`&`) via additive multi-layer ChartSpec extension + SVG compositor
- Theme value class + 8 builtins + contextvars-backed `set_default_theme()` (the one sanctioned global mutation)
- Faceting, annotations, CoordFlip, `mark_density`/`mark_histogram`/`mark_smooth`
- Data ingestion via narwhals (pandas/modin/cuDF/dask/ibis support for free)

## Done criteria (from ferrum-phases.md)
- [x] `Chart(data).mark_point().encode(x="...", y="...").show()` works
- [x] `+`/`|`/`&` composition works
- [x] Themes are values; `set_default_theme()` is the documented contextvar exception
- [x] No matplotlib in deps
- [x] All 31 encoding channel classes exist; rendered subset = x/y/color/size/shape/opacity
- [x] Three statistical marks (density/histogram/smooth) work over existing Phase 5 transforms

## Test results
- `cargo test -p ferrum-core`: 291+ pass
- `uv run pytest`: 179+ pass
- 6 SVG goldens + 1 PNG hash from Phase 7 still match (additive ChartSpec extension preserved JSON shape)

## Spec + plan
- Spec: `docs/superpowers/specs/2026-05-10-grammar-api-design.md`
- Plan: `docs/superpowers/plans/2026-05-10-grammar-api-plan.md`
EOF
)"
```

---

## Summary

Phase 8a ships the full user-facing Python API surface on top of Phase 7's renderer. The implementation is deliberately additive — every Rust spec extension uses `Option<>` with `skip_serializing_if = "Option::is_none"` to preserve Phase 3-7 JSON byte-identicality. The 6 SVG goldens and 1 PNG hash from Phase 7 must still match at the end of Phase 8a.

After Phase 8a lands:
- Phase 8b ships composite + heavy statistical marks on the same API surface (purely additive)
- Phase 9 ships figure-level convenience functions and wires the deferred channel kwargs into the renderer
- Phase 10 adds ModelSource and model-diagnostic marks
- Phase 11 enables `.interactive()` via the WASM renderer

---

## Self-Review

After writing the plan, look at the spec with fresh eyes:

**1. Spec coverage:** Each spec section (§1–§12) should map to one or more tasks above. Quick check:
- §1 Goal — Task 38 verification example demonstrates the exact §1 idioms
- §2 Scope — Tasks 1–35 cover everything in scope; deferred-mark stubs (Tasks 24, 27) cover everything out-of-scope
- §3 Architecture — Tasks 1–11 (Rust changes) + Tasks 12–35 (Python modules) match the spec's module layout
- §4 Per-component contracts — each subsection has a corresponding task (Chart=27, Layer=29, Channels=15-19, Marks=23-25, Theme=20-22, Composition=29, Facet=28, Annotations=30, CoordFlip=31, Coerce=12)
- §5 Algorithm — Tasks 6, 7 (multi-layer + CoordFlip), 8 (scale resolution), 9 (per-row size/shape/opacity), 11 (compositor)
- §6 Error policy — Tasks 14 (warn-once registry), 24 (deferred-mark errors), 27 (NotImplementedError stubs), 31 (coord stubs), 32 (save format errors)
- §7 New deps — Task 1 (narwhals)
- §8 Test plan — distributed across all tasks; Task 36 backfills any gaps
- §9 Done-criteria gate — Task 38 verification script
- §10 Locked decisions — implicit in all tasks; Task 37 surfaces in dated notes
- §11 Cross-phase notes — Task 38 phases-doc update

**2. Placeholder scan:** Re-grep for TODO/TBD/etc. — none should remain in step bodies. The §12 spec refinements section is intentionally empty per Phase 6 precedent.

**3. Type consistency:** Names used across tasks match. `MarkKwargsSpec` (Task 5) is referenced in Tasks 6, 7, 34. `EncodingSpec.scale: Option<ScaleSpec>` (Task 3) is consumed in Task 8. `ChannelBase` (Task 15) is the parent of every channel class in Tasks 16–19. Chart's fluent method names (Task 27) are referenced consistently in Tasks 28–34.

If issues are found, fix them inline.
