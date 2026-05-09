use pyo3::prelude::*;

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
