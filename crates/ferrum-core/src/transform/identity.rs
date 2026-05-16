//! Identity transform — passes the input batch through unchanged.
//!
//! Used by `Chart.__add__` to route a layer's DataFrame as a named output
//! when column names overlap between LHS and RHS. The layer sets
//! `data_source` to the named key so it reads its own data slice rather
//! than the merged primary batch.

use arrow::array::RecordBatch;
use pyo3::prelude::*;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct IdentitySpec {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

pub(crate) fn apply(_spec: &IdentitySpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    Ok(batch.clone())
}

/// Identity (pass-through) transform.
///
/// Publishes the input batch unchanged under a named key so that layers
/// with ``data_source`` set to that key read their own data rather than
/// the merged primary batch.
///
/// Parameters
/// ----------
/// name : str
///     Named-output key. Layers that should read this data set
///     ``data_source`` to the same string.
#[pyclass(module = "ferrum._core")]
#[derive(Debug, Clone)]
pub(crate) struct PyIdentity(pub(crate) super::core::TransformSpec);

#[pymethods]
impl PyIdentity {
    #[new]
    #[pyo3(signature = (name))]
    fn new(name: String) -> Self {
        PyIdentity(super::core::TransformSpec::Identity(IdentitySpec {
            name: Some(name),
        }))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            super::core::TransformSpec::Identity(s) => format!("Identity(name={:?})", s.name),
            _ => "Identity(?)".to_string(),
        }
    }
}
