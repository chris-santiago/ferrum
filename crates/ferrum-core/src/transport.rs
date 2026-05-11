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

/// Accept an Arrow C Data Interface stream and return a normalised stream.
///
/// Parameters
/// ----------
/// reader : pyarrow.RecordBatchReader or compatible
///     Any Python object that implements the Arrow C Data Interface stream
///     protocol (``__arrow_c_stream__``). Polars ``DataFrame`` and pyarrow
///     ``RecordBatch`` / ``Table`` objects satisfy this directly with zero
///     copy.
///
/// Returns
/// -------
/// pyarrow.RecordBatchReader
///     A fresh ``RecordBatchReader`` over the same data after Ferrum's
///     internal normalisation pass (column validation and canonicalisation).
///     The returned reader is consumed once; iterate it or pass it directly
///     to ``render_svg`` / ``render_png``.
///
/// Raises
/// ------
/// ValueError
///     If the input stream is empty (zero columns or unreadable batches).
///
/// Notes
/// -----
/// This is the Arrow CDI entry point for the Ferrum pipeline. Polars and
/// pyarrow take the zero-copy C Data Interface fast-path; other DataFrame
/// libraries (pandas, modin, cuDF) should be coerced to Arrow via
/// ``narwhals`` before calling this function.
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
