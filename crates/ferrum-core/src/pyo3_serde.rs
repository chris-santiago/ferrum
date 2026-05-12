//! Bridge between PyO3 and serde for ad-hoc Python value ↔ Rust struct
//! conversions where we don't want to write per-type `FromPyObject` /
//! `IntoPyObject` impls.
//!
//! Both directions go via JSON: a Rust value serializes to a `String`,
//! which `json.loads` turns into a Python value; conversely, a Python
//! value is fed through `json.dumps` and `serde_json::from_str`. This
//! costs an extra encode/decode per call but keeps the impl trivial,
//! works for every `#[derive(Serialize)]`/`#[derive(Deserialize)]` type
//! in the crate, and matches the prior in-place idiom across
//! `spec::chart::new` (facet/mark_style/position dicts) and the
//! matching getters.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{de::DeserializeOwned, Serialize};

/// Deserialize a Rust value from a Python object (typically a dict). The
/// `context` string is prefixed onto any serde error so the caller's
/// intent ("facet:", "position:", `layers[3]:`, …) is preserved.
pub(crate) fn from_py<T: DeserializeOwned>(
    obj: &Bound<'_, PyAny>,
    context: &str,
) -> PyResult<T> {
    let py = obj.py();
    let json_module = py.import("json")?;
    let s: String = json_module.call_method1("dumps", (obj,))?.extract()?;
    serde_json::from_str(&s).map_err(|e| PyValueError::new_err(format!("{context}: {e}")))
}

/// Serialize a Rust value to a fresh Python object — equivalent to
/// `json.loads(json.dumps(value))` for `value`'s serde representation.
pub(crate) fn to_py<T: Serialize>(py: Python<'_>, value: &T) -> PyResult<Py<PyAny>> {
    let s = serde_json::to_string(value).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let json_module = py.import("json")?;
    let val = json_module.call_method1("loads", (s,))?;
    Ok(val.unbind())
}
