use pyo3::prelude::*;

mod transport;
mod spec;
mod scale;
pub(crate) mod transform;
pub(crate) mod layout;
pub(crate) mod render;

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
    m.add_class::<transform::bin::PyBin>()?;
    m.add_class::<transform::kde::PyKde>()?;
    m.add_class::<transform::smooth::PySmooth>()?;
    m.add_class::<transform::aggregate::PyAggregateOp>()?;
    m.add_class::<transform::aggregate::PyAggregate>()?;
    m.add_class::<transform::summary::PySummary>()?;
    m.add_class::<transform::outliers::PyOutliers>()?;
    m.add_class::<transform::error_extent::PyErrorExtent>()?;
    m.add_class::<transform::box_stats::PyBoxStats>()?;
    m.add_class::<transform::violin::PyViolin>()?;
    m.add_class::<transform::kde_2d::PyKde2D>()?;
    m.add_class::<transform::contour::PyContour>()?;
    m.add_class::<transform::qq::PyQQ>()?;
    m.add_function(wrap_pyfunction!(layout::binding::compute_layout, m)?)?;
    m.add_function(wrap_pyfunction!(render::binding::render_svg, m)?)?;
    m.add_function(wrap_pyfunction!(render::binding::render_png, m)?)?;
    m.add_function(wrap_pyfunction!(render::binding::compose_svg_horizontal_py, m)?)?;
    m.add_function(wrap_pyfunction!(render::binding::compose_svg_vertical_py, m)?)?;
    Ok(())
}
