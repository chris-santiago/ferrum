use pyo3::prelude::*;
use arrow::array::RecordBatch;
use arrow::datatypes::{Field, Schema};
use arrow::error::ArrowError;
use std::sync::Arc;

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
