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
    // Transform PyO3 wrappers, driven by the single-source-of-truth macro
    // in transform/core.rs. Adding a new transform is one line there;
    // registration happens automatically here.
    macro_rules! register_transforms {
        ($($V:ident => $mod:ident : $py:ident,)*) => {{
            $( m.add_class::<crate::transform::$mod::$py>()?; )*
        }};
    }
    crate::transform::core::for_each_transform!(register_transforms);
    // PyAggregateOp is the op-spec helper class, not a TransformSpec
    // variant — registered manually.
    m.add_class::<transform::aggregate::PyAggregateOp>()?;
    m.add_function(wrap_pyfunction!(layout::binding::compute_layout, m)?)?;
    m.add_function(wrap_pyfunction!(render::binding::render_svg, m)?)?;
    m.add_function(wrap_pyfunction!(render::binding::render_png, m)?)?;
    m.add_function(wrap_pyfunction!(render::binding::rasterize_svg, m)?)?;
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
