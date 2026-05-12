use pyo3::prelude::*;

mod transport;
mod pyo3_serde;
mod spec;
mod scale;
pub(crate) mod transform;
pub(crate) mod layout;
pub(crate) mod render;
pub(crate) mod diagnostics;

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
    m.add_class::<transform::bin_2d::PyBin2D>()?;
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
    m.add_class::<transform::qq::PyQq>()?;
    m.add_class::<transform::raster::PyRaster>()?;
    m.add_class::<transform::hex::PyHex>()?;
    m.add_class::<transform::swarm::PySwarm>()?;
    m.add_class::<transform::unpivot::PyUnpivot>()?;
    m.add_class::<transform::reorder::PyReorder>()?;
    m.add_class::<transform::reference_line::PyReferenceLine>()?;
    m.add_class::<transform::linkage::PyLinkage>()?;
    m.add_class::<transform::letter_value::PyLetterValue>()?;
    m.add_class::<transform::logistic::PyLogistic>()?;
    m.add_class::<transform::glm::PyGlm>()?;
    m.add_class::<transform::robust::PyRobust>()?;
    m.add_function(wrap_pyfunction!(layout::binding::compute_layout, m)?)?;
    m.add_function(wrap_pyfunction!(render::binding::render_svg, m)?)?;
    m.add_function(wrap_pyfunction!(render::binding::render_png, m)?)?;
    m.add_function(wrap_pyfunction!(render::binding::compose_svg_horizontal_py, m)?)?;
    m.add_function(wrap_pyfunction!(render::binding::compose_svg_vertical_py, m)?)?;
    m.add_function(wrap_pyfunction!(render::binding::compose_svg_grid_py, m)?)?;
    // Phase 8b Task 37: continuous color schemes.
    m.add_class::<render::color::continuous::PyContinuousScheme>()?;
    m.add_function(wrap_pyfunction!(render::color::continuous::Gradient, m)?)?;
    // Phase 10g Task 35: Kendall's tau-b (Knight's O(n log n)).
    m.add_function(wrap_pyfunction!(diagnostics::py_kendall_tau_b, m)?)?;
    Ok(())
}
