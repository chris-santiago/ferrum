use pyo3::prelude::*;

mod transport;
mod pyo3_serde;
mod spec;
mod scale;
pub(crate) mod transform;
pub(crate) mod layout;
pub(crate) mod render;
pub(crate) mod diagnostics;
pub(crate) mod projection;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(transport::process_batch, m)?)?;
    m.add_class::<spec::chart::ChartSpec>()?;
    m.add_class::<spec::encoding::EncodingSpec>()?;
    // Batch-C task 4 (round 4): the scale wire-key gate's own accepted-key
    // table, published to Python so `_spec_build.py`'s override-scale merge
    // filters against the SAME source of truth the gate enforces instead of
    // a hand-mirrored copy.
    m.add_function(wrap_pyfunction!(spec::encoding::scale_accepted_keys, m)?)?;
    m.add_class::<scale::linear::LinearScale>()?;
    m.add_class::<scale::log::LogScale>()?;
    m.add_class::<scale::symlog::SymlogScale>()?;
    m.add_class::<scale::time::TimeScale>()?;
    m.add_class::<scale::ordinal::OrdinalScale>()?;
    m.add_class::<scale::threshold::ThresholdScale>()?;
    m.add_class::<scale::quantile::QuantileScale>()?;
    m.add_class::<scale::pow::PowScale>()?;
    m.add_class::<scale::pow::SqrtScale>()?;
    m.add_class::<scale::band::BandScale>()?;
    m.add_class::<scale::point::PointScale>()?;
    m.add_class::<scale::sequential::SequentialScale>()?;
    m.add_class::<scale::diverging::DivergingScale>()?;
    m.add_class::<scale::quantize::QuantizeScale>()?;
    m.add_class::<scale::bin_ordinal::BinOrdinalScale>()?;
    // Transform PyO3 wrappers, driven by the single-source-of-truth macro
    // in transform/core.rs. Adding a new transform with a Python class is one
    // line in `for_each_py_transform!`; registration happens automatically here.
    // The dict-only Phase-12 `Data*` transforms are intentionally absent from
    // that table (SEAM-02) — they expose no Python class.
    macro_rules! register_transforms {
        ($($V:ident => $mod:ident : $py:ident,)*) => {{
            $( m.add_class::<crate::transform::$mod::$py>()?; )*
        }};
    }
    crate::transform::core::for_each_py_transform!(register_transforms);
    // PyAggregateOp is the op-spec helper class, not a TransformSpec
    // variant — registered manually.
    m.add_class::<transform::aggregate::PyAggregateOp>()?;
    m.add_function(wrap_pyfunction!(layout::binding::compute_layout, m)?)?;
    m.add_function(wrap_pyfunction!(render::binding::render_svg, m)?)?;
    m.add_function(wrap_pyfunction!(render::binding::render_png, m)?)?;
    m.add_function(wrap_pyfunction!(render::binding::render_interactive, m)?)?;
    m.add_function(wrap_pyfunction!(render::binding::render_composite_svg, m)?)?;
    m.add_function(wrap_pyfunction!(render::binding::render_composite_interactive, m)?)?;
    m.add_function(wrap_pyfunction!(render::binding::rasterize_svg, m)?)?;
    m.add_function(wrap_pyfunction!(render::binding::wrap_svg_with_chrome_py, m)?)?;
    m.add_function(wrap_pyfunction!(render::binding::figure_title_nodes_py, m)?)?;
    // Theme key contract (D-THEME-1): `ThemeOverridesSpec` is the single
    // source of truth; Python derives its key lists from these accessors.
    m.add_function(wrap_pyfunction!(render::binding::theme_known_keys, m)?)?;
    m.add_function(wrap_pyfunction!(render::binding::theme_color_keys, m)?)?;
    // Phase 8b Task 37: continuous color schemes.
    m.add_class::<render::color::continuous::PyContinuousScheme>()?;
    m.add_function(wrap_pyfunction!(render::color::continuous::Gradient, m)?)?;
    // T2.2 (D-COLOR-1): palette registry accessors — the single source of truth
    // that Python `color.py` consumes instead of hand-mirroring hex tables.
    m.add_function(wrap_pyfunction!(render::color::palette::list_palettes, m)?)?;
    m.add_function(wrap_pyfunction!(render::color::palette::palette_kind, m)?)?;
    m.add_function(wrap_pyfunction!(render::color::palette::palette_colors, m)?)?;
    m.add_function(wrap_pyfunction!(render::color::palette::palette_sample, m)?)?;
    // Batch A Task 1 (F-L02-01/NF-A1): the one full-CSS color parser exposed
    // to Python — `ferrum.color.to_hex`'s string path routes through this
    // instead of hand-rolling a second color vocabulary in Python.
    m.add_function(wrap_pyfunction!(render::color::primitive::parse_color_to_hex, m)?)?;
    // Phase 10g Task 35: Kendall's tau-b (Knight's O(n log n)).
    m.add_function(wrap_pyfunction!(diagnostics::py_kendall_tau_b, m)?)?;
    // Classification diagnostic curve kernels (sklearn-parity).
    m.add_function(wrap_pyfunction!(diagnostics::roc_curve_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(diagnostics::roc_auc, m)?)?;
    m.add_function(wrap_pyfunction!(diagnostics::pr_curve_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(diagnostics::average_precision, m)?)?;
    m.add_function(wrap_pyfunction!(diagnostics::calibration_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(diagnostics::confusion_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(diagnostics::prf_at_thresholds, m)?)?;
    // Tier 3: Rust-native statistics (replaces stats.py).
    m.add_function(wrap_pyfunction!(transform::stats::hat_matrix_stats, m)?)?;
    m.add_function(wrap_pyfunction!(transform::stats::studentized_residual_no_x, m)?)?;
    m.add_function(wrap_pyfunction!(transform::stats::py_shapiro_w, m)?)?;
    m.add_function(wrap_pyfunction!(transform::stats::py_rankdata_average, m)?)?;
    m.add_function(wrap_pyfunction!(transform::stats::py_rank1d, m)?)?;
    m.add_function(wrap_pyfunction!(transform::stats::py_rank1d_with_y, m)?)?;
    m.add_function(wrap_pyfunction!(transform::stats::py_rank2d, m)?)?;
    m.add_function(wrap_pyfunction!(transform::stats::pca_scores, m)?)?;
    m.add_function(wrap_pyfunction!(transform::stats::pca_variance, m)?)?;
    m.add_function(wrap_pyfunction!(transform::stats::mds_classical, m)?)?;
    m.add_function(wrap_pyfunction!(transform::stats::silhouette_samples, m)?)?;
    m.add_function(wrap_pyfunction!(transform::stats::silhouette_score, m)?)?;
    m.add_function(wrap_pyfunction!(transform::stats::calinski_harabasz_score, m)?)?;
    // Phase 10: Rust-native t-SNE and UMAP via manifolds-rs.
    m.add_function(wrap_pyfunction!(transform::stats::tsne_embedding, m)?)?;
    m.add_function(wrap_pyfunction!(transform::stats::umap_embedding, m)?)?;
    Ok(())
}
