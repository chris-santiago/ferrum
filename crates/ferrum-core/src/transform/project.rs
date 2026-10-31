//! Data transform: Project — keep exactly the listed columns, in order.
//!
//! The name and semantics of Vega's `project` transform. A composition over
//! different frames concatenates them into one batch, where each frame's
//! rows carry the other frame's columns as null padding; projecting an
//! operand's rows to its own frame's columns makes a binding to a column only
//! the other frame has fail as an unknown column instead of reading the
//! padding.
//!
//! Inside a facet partition run the partition's facet fields are kept too
//! ([`TransformContext::facet_fields`]): a panel's facet value belongs to
//! every row in it, as the facet column re-injected into every output already
//! treats it, so a statistic grouping by the facet field still groups per
//! panel.

use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::{ArrayRef, RecordBatch};
use arrow::datatypes::{Field, Schema};
use arrow::error::ArrowError;
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};

use crate::transform::context::TransformContext;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProjectSpec {
    /// The columns to keep, in output order. Every one must be in the input.
    pub fields: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

pub(crate) fn apply(spec: &ProjectSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    project(spec, batch, &[])
}

pub(crate) fn apply_with_context(
    spec: &ProjectSpec,
    batch: &RecordBatch,
    ctx: &TransformContext,
) -> PyResult<RecordBatch> {
    project(spec, batch, &ctx.facet_fields)
}

/// `rows` with `column` under `name`: in place of a column of that name, or
/// appended. The field is nullable, as a calculated column's is.
pub(crate) fn with_column(rows: &RecordBatch, name: &str, column: ArrayRef) -> Result<RecordBatch, ArrowError> {
    let schema = rows.schema();
    let mut fields: Vec<Field> = schema.fields().iter().map(|f| f.as_ref().clone()).collect();
    let mut columns = rows.columns().to_vec();
    let field = Field::new(name, column.data_type().clone(), true);
    match schema.index_of(name) {
        Ok(index) => {
            fields[index] = field;
            columns[index] = column;
        }
        Err(_) => {
            fields.push(field);
            columns.push(column);
        }
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
}

/// `batch` cut to `spec.fields`, in that order, followed by each of `keep`
/// the list does not already name and the batch holds. A listed field the
/// batch lacks, or one listed twice, is a malformed spec and is refused.
pub(crate) fn project(spec: &ProjectSpec, batch: &RecordBatch, keep: &[String]) -> PyResult<RecordBatch> {
    let schema = batch.schema();
    let mut listed: HashSet<&str> = HashSet::with_capacity(spec.fields.len());
    let mut indices: Vec<usize> = Vec::with_capacity(spec.fields.len() + keep.len());
    for field in &spec.fields {
        if !listed.insert(field.as_str()) {
            return Err(PyValueError::new_err(format!("project: column '{field}' is listed twice")));
        }
        let index = schema.index_of(field).map_err(|_| {
            let available: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
            PyValueError::new_err(format!(
                "project: column '{field}' is not in the data; its columns are {available:?}"
            ))
        })?;
        indices.push(index);
    }
    for field in keep {
        if listed.contains(field.as_str()) {
            continue;
        }
        if let Ok(index) = schema.index_of(field) {
            indices.push(index);
        }
    }
    batch.project(&indices).map_err(|e| PyValueError::new_err(format!("project: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{ArrayRef, Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Float64, true),
            Field::new("b", DataType::Utf8, true),
            Field::new("c", DataType::Utf8, true),
            Field::new("pad", DataType::Float64, true),
        ]));
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(StringArray::from(vec!["x", "y"])),
            Arc::new(StringArray::from(vec!["p", "q"])),
            Arc::new(Float64Array::from(vec![None, None])),
        ];
        RecordBatch::try_new(schema, columns).expect("schema and columns agree")
    }

    fn names(batch: &RecordBatch) -> Vec<String> {
        batch.schema().fields().iter().map(|f| f.name().clone()).collect()
    }

    fn spec(fields: &[&str]) -> ProjectSpec {
        ProjectSpec { fields: fields.iter().map(|f| (*f).to_string()).collect(), name: None }
    }

    /// The output holds exactly the listed columns in the listed order; an
    /// unlisted column (the padding) is gone. The rejected reading, a row
    /// step that leaves the schema alone, keeps all four in input order.
    #[test]
    fn keeps_exactly_the_listed_columns_in_order() {
        let out = apply(&spec(&["b", "a"]), &batch()).expect("both columns exist");
        assert_eq!(names(&out), ["b", "a"]);
        assert_eq!(out.num_rows(), 2);
        let b = out.column(0).as_any().downcast_ref::<StringArray>().expect("b is Utf8");
        assert_eq!(b.value(1), "y");
    }

    /// A listed column the input lacks is a malformed spec, refused with a
    /// message naming it, never an empty or null column.
    #[test]
    fn refuses_a_listed_column_the_input_lacks() {
        let err = apply(&spec(&["a", "missing"]), &batch()).expect_err("missing is not a column");
        let text = err.to_string();
        assert!(text.contains("project: column 'missing' is not in the data"), "{text}");
    }

    /// A column listed twice is refused rather than emitting two columns of
    /// one name.
    #[test]
    fn refuses_a_column_listed_twice() {
        let err = apply(&spec(&["a", "a"]), &batch()).expect_err("a is listed twice");
        assert!(err.to_string().contains("project: column 'a' is listed twice"), "{err}");
    }

    /// Inside a facet partition run the facet field survives a list that
    /// omits it, after the listed columns; one the list names keeps its
    /// listed place, and one the input lacks is not invented. Outside a
    /// partition run (no facet fields) the same list drops it.
    #[test]
    fn a_partition_run_keeps_its_facet_fields() {
        let ctx = TransformContext {
            facet_fields: vec!["c".to_string(), "b".to_string(), "absent".to_string()],
            ..TransformContext::default()
        };
        let out = apply_with_context(&spec(&["b", "a"]), &batch(), &ctx).expect("columns exist");
        assert_eq!(names(&out), ["b", "a", "c"]);
        let plain = apply_with_context(&spec(&["b", "a"]), &batch(), &TransformContext::default())
            .expect("columns exist");
        assert_eq!(names(&plain), ["b", "a"]);
    }

    /// The wire form: `{"type": "project", "fields": [...]}` round-trips, and
    /// an unknown key is refused.
    #[test]
    fn the_wire_form_round_trips() {
        let spec: crate::transform::core::TransformSpec =
            serde_json::from_str(r#"{"type":"project","fields":["a","b"]}"#).expect("valid wire");
        assert_eq!(
            spec,
            crate::transform::core::TransformSpec::Project(ProjectSpec {
                fields: vec!["a".into(), "b".into()],
                name: None,
            })
        );
        assert_eq!(serde_json::to_string(&spec).expect("serializes"), r#"{"type":"project","fields":["a","b"]}"#);
        assert!(serde_json::from_str::<crate::transform::core::TransformSpec>(
            r#"{"type":"project","fields":["a"],"columns":["b"]}"#
        )
        .is_err());
    }
}
