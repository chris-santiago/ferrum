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
    // Batch-C close remediation: validates a raw scale dict through the SAME
    // gate `EncodingSpec(field, scale={...})` enforces, so `_spec_build.py`'s
    // override-scale merge can stop constructing a throwaway `EncodingSpec`
    // for the side effect of its exception (rust-design recommendation 5 /
    // python-design S2, converged).
    m.add_function(wrap_pyfunction!(spec::encoding::validate_scale_dict, m)?)?;
    // `#157`: the override door's own data-free scale-value check (today,
    // band padding's d3 bounds), so `_override_consume.py`'s
    // `refuse_malformed_scale_overrides` can ask Rust "is this one override
    // value valid?" and name the call itself.
    m.add_function(wrap_pyfunction!(spec::encoding::validate_scale_values, m)?)?;
    // The one owner of what a scale-type switch keeps: padding-family keys
    // translated by meaning, every other key kept or dropped by the accepted
    // key table, for `_override_consume.py`'s override-scale merge.
    m.add_function(wrap_pyfunction!(spec::encoding::translate_scale_switch, m)?)?;
    // The padding-family keys a scale type accepts, so a caller selecting
    // them reads this set instead of restating it.
    m.add_function(wrap_pyfunction!(spec::encoding::scale_padding_keys, m)?)?;
    // The sort-field-object vocabulary (`parse_sort_field_object`/
    // `EncodingSpec::new` enforce), published so
    // every sort-accepting Python channel (`X`/`Y`/`Facet`/...) validates
    // against the SAME `op`/`order` spellings instead of holding a copy.
    m.add_function(wrap_pyfunction!(spec::encoding::sort_accepted_vocabulary, m)?)?;
    // `#159`: the published scale-tag vocabulary (the 16 `ScaleSpec` wire
    // `"type"` tags, and the `time`/`utc` temporal subset), so
    // `_override_apply.py`/`_scale.py` read this instead of holding their
    // own literal copies.
    m.add_function(wrap_pyfunction!(spec::encoding::scale_type_tags, m)?)?;
    m.add_function(wrap_pyfunction!(spec::encoding::scale_temporal_type_tags, m)?)?;
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
    // Batch-D task 1 (spec §4.6): the transform metadata registry — method
    // vocabularies, primary output schemas, and wire params per transform
    // kind, one Rust-owned table Python queries instead of hand-mirroring.
    m.add_function(wrap_pyfunction!(transform::registry::transform_registry, m)?)?;
    // The numeric columns `Linkage`/`Reorder` select from a table, so
    // heatmap/clustermap split id from numeric columns by Rust's rule.
    m.add_function(wrap_pyfunction!(transform::numeric_util::numeric_columns, m)?)?;
    // How the render reads a column's dtype as categorical, so the Python bar
    // zero-anchor picks the same value channel the bar builders draw along.
    m.add_function(wrap_pyfunction!(render::binding::column_reads_categorical, m)?)?;
    // The positional gate's unknown-column refusal text, so a Python check
    // that knows a layer's schema raises the gate's words, not a copy.
    m.add_function(wrap_pyfunction!(render::binding::layer_unknown_column_message, m)?)?;
    // Wave-4 task 9 (#193): the `bins=` vocabulary gate, published so a figure
    // entry (`displot`) can refuse a malformed `bins=` naming ITSELF rather
    // than the private `Bin` transform it builds — without a second copy of the
    // accepted spellings living in Python. Same shape as `validate_scale_dict`.
    m.add_function(wrap_pyfunction!(transform::bin_mode::validate_bin_rule, m)?)?;
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
    // #123: the aux-legend channel names
    // (`render/prepare/legend.rs::AUX_LEGEND_CHANNEL_NAMES`), published so
    // `ferrum.encoding.base` reads the list instead of hand-mirroring it.
    m.add_function(wrap_pyfunction!(render::prepare::aux_legend_channels, m)?)?;
    // The channels Rust builds a chart-wide auxiliary scale for
    // (`spec/encoding/mod.rs::AUXILIARY_SCALE_CHANNELS`), published so
    // a layer merge promotes exactly the bindings Rust scales.
    m.add_function(wrap_pyfunction!(render::scale_resolve::auxiliary_scale_channels, m)?)?;
    // The numeric auxiliary channels' `scale=` gate
    // (`render/scale_resolve/auxiliary.rs::numeric_aux_scale`), published so
    // a channel class refuses at construction with the render-time text.
    m.add_function(wrap_pyfunction!(render::scale_resolve::validate_numeric_aux_scale, m)?)?;
    // #176: the `interpolate=` vocabulary
    // (`render/marks/curve.rs::Interpolate::ALL`), published so `marks/base.py`
    // gates it at declaration instead of the renderer refusing it.
    m.add_function(wrap_pyfunction!(render::marks::curve::interpolate_accepted_values, m)?)?;
    // `Smooth(x_estimator=)`'s vocabulary
    // (`transform/aggregate.rs::XEstimator`), published the same way.
    m.add_function(wrap_pyfunction!(transform::aggregate::x_estimator_accepted_values, m)?)?;
    // #165(1): the "{name}_{key}" secondary-output naming rule
    // (`transform/registry/mod.rs::secondary_key`), its one home.
    m.add_function(wrap_pyfunction!(transform::registry::secondary_key, m)?)?;
    // The mark-kwargs wire keys (`spec/mark_style.rs::mark_kwargs_field_names`),
    // published so Python reads them instead of parsing the Rust source.
    m.add_function(wrap_pyfunction!(spec::mark_style::mark_kwargs_fields, m)?)?;
    // Its numeric (`f64`/`usize`) subset, the fields the Python funnel coerces.
    m.add_function(wrap_pyfunction!(spec::mark_style::mark_numeric_kwargs, m)?)?;
    // The `usize` subset of the numeric fields, routed through a
    // non-negative-integer contract instead of the real-number one.
    m.add_function(wrap_pyfunction!(spec::mark_style::mark_integer_kwargs, m)?)?;
    // The bool and string-shaped subsets, by the same serde probe, routed
    // through a type gate at declaration.
    m.add_function(wrap_pyfunction!(spec::mark_style::mark_bool_kwargs, m)?)?;
    m.add_function(wrap_pyfunction!(spec::mark_style::mark_string_kwargs, m)?)?;
    // The size rule (`spec/mark.rs::Mark::size_drawing`, `render/draw.rs::check_constant_size`),
    // published so every Python door that builds mark kwargs refuses with the
    // renderer's own rule and text.
    m.add_function(wrap_pyfunction!(spec::mark::marks_drawing_constant_size, m)?)?;
    m.add_function(wrap_pyfunction!(spec::mark::mark_style_fields, m)?)?;
    m.add_function(wrap_pyfunction!(spec::mark::style_table_marks, m)?)?;
    // Which inherited positional bindings a layer's mark draws (`Mark::inherited_positional`),
    // the rule Python's effective encoding and Rust's layer completion share.
    m.add_function(wrap_pyfunction!(spec::mark::mark_inherited_positional, m)?)?;
    m.add_function(wrap_pyfunction!(spec::mark::mark_draws_inherited, m)?)?;
    m.add_function(wrap_pyfunction!(spec::mark::mark_drawn_channels, m)?)?;
    // The columns one Bin computes (`BinSpec::computed_columns`), read by
    // Python's bin declaration instead of a naming rule of its own.
    m.add_function(wrap_pyfunction!(transform::bin::bin_output_columns, m)?)?;
    m.add_function(wrap_pyfunction!(render::marks::image::drawn_positional_columns_py, m)?)?;
    m.add_function(wrap_pyfunction!(render::draw::blend_accepted_values, m)?)?;
    m.add_function(wrap_pyfunction!(render::draw::stroke_cap_accepted_values, m)?)?;
    m.add_function(wrap_pyfunction!(render::draw::stroke_join_accepted_values, m)?)?;
    m.add_function(wrap_pyfunction!(render::draw::check_mark_size, m)?)?;
    m.add_function(wrap_pyfunction!(render::draw::mark_size_reason, m)?)?;
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
    // Returns the raw (r, g, b, a) byte tuple from the same parse
    // `parse_color_to_hex` uses, so a caller that needs channel bytes (the
    // selection wire helper) reads them directly instead of decoding hex.
    m.add_function(wrap_pyfunction!(render::color::primitive::parse_color_to_rgba, m)?)?;
    // The canonical stroke-dash palette (`ferrum_scene::DASH_PALETTE`),
    // published so `_LINETYPE_MAP` asserts its patterns are palette members
    // instead of holding a copy of the table.
    m.add_function(wrap_pyfunction!(render::draw::stroke_dash_palette, m)?)?;
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
