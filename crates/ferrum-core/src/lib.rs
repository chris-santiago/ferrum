use pyo3::prelude::*;

mod transport;
mod spec;
mod scale;
mod transform;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(transport::process_batch, m)?)?;
    m.add_class::<spec::chart::ChartSpec>()?;
    m.add_class::<spec::encoding::EncodingSpec>()?;
    m.add_class::<scale::linear::LinearScale>()?;
    m.add_class::<scale::log::LogScale>()?;
    m.add_class::<scale::symlog::SymlogScale>()?;
    m.add_class::<scale::time::TimeScale>()?;
    m.add_class::<scale::ordinal::OrdinalScale>()?;
    m.add_class::<scale::threshold::ThresholdScale>()?;
    m.add_class::<scale::quantile::QuantileScale>()?;
    Ok(())
}
