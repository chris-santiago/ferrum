//! Facet transform-extent pinning.
//!
//! When faceting is active, extent-carrying / extent-deriving transforms (Kde,
//! Bin, Violin, Kde2D, Bin2D, DensityData, Hex, Raster, DataBin) would each
//! compute their value-axis range from their own per-panel partition, making
//! positions non-comparable across panels (archaeology bug #7). This module
//! pins every such transform's extent to the global pre-facet range before
//! partitioning, but only when the user left the extent unset.

use arrow::record_batch::RecordBatch;

/// Pin a shared value-axis extent across facet panels for every extent-carrying
/// transform — the 1-D trio (`Kde`, `Bin`, `Violin`) and the 2-D pair (`Kde2D`,
/// `Bin2D`) — that has not been given an explicit extent.
///
/// Returns a new transform list where each such spec's extent field is set to the
/// global range of its value field(s) over the **full pre-facet `batch`**. Because
/// the extent is computed before partitioning, every per-panel partition (and
/// every hue group within a panel) inherits the same value axis, so panels and
/// groups are visually comparable. This is the correct default for faceted
/// density / histogram / violin charts and faceted 2-D density / heatmap / contour
/// charts (archaeology bug #7, extended to 2-D by R5).
///
/// The extent is derived over the entire field column(s), ignoring `groupby` and
/// facet partitioning — so the multi-group (hue) case is covered without
/// special-casing (spec §8). Each transform's owning module computes the extent
/// (`kde::global_extent` / `bin::global_extent` / `violin::global_extent` /
/// `kde_2d::global_extent` / `bin_2d::global_extent` / etc.); this seam only
/// orchestrates (it does not re-derive extents in the render layer).
///
/// NICENESS CONTRACT (XFORM-08) — the one place the niced-vs-raw divergence is
/// documented. Every `global_extent` shares the same `numeric_util::column_extent`
/// fold, but **`Bin` alone nices** the result inside its own `global_extent`, so
/// the pinned range reproduces the same bin edges every per-group partition would
/// compute. Every other transform here returns the RAW range:
/// `Kde`/`Violin`/`DensityData`/`DataBin` because they control a continuous grid
/// or nice from the pinned range inside `apply`, and `Kde2D`/`Bin2D`/`Hex`/`Raster`
/// because their `apply` divides the raw `(min, max)` directly. The divergence is
/// owned by each transform's `global_extent` (each carries a `NICENESS CONTRACT`
/// note), not by this orchestrator.
///
/// A spec that already carries an explicit extent (user-provided) is left
/// unchanged; `Bin2D`'s per-axis `extent_x`/`extent_y` are pinned independently so
/// a partially-specified extent keeps the user axis and pins only the unset one.
/// All non-extent-carrying transforms are passed through unchanged.
pub(crate) fn fix_transform_extents_for_facet(
    transforms: &[crate::transform::core::TransformSpec],
    batch: &RecordBatch,
) -> Vec<crate::transform::core::TransformSpec> {
    use crate::transform::core::TransformSpec;

    transforms
        .iter()
        .map(|t| match t {
            TransformSpec::Kde(spec) if spec.extent.is_none() => {
                match crate::transform::kde::global_extent(spec, batch) {
                    Some(extent) => TransformSpec::Kde(crate::transform::kde::KdeSpec {
                        extent: Some(extent),
                        ..spec.clone()
                    }),
                    None => t.clone(),
                }
            }
            TransformSpec::Bin(spec) if spec.extent.is_none() => {
                match crate::transform::bin::global_extent(spec, batch) {
                    Some(extent) => TransformSpec::Bin(crate::transform::bin::BinSpec {
                        extent: Some(extent),
                        ..spec.clone()
                    }),
                    None => t.clone(),
                }
            }
            TransformSpec::Violin(spec) if spec.extent.is_none() => {
                match crate::transform::violin::global_extent(spec, batch) {
                    Some(extent) => TransformSpec::Violin(crate::transform::violin::ViolinSpec {
                        extent: Some(extent),
                        ..spec.clone()
                    }),
                    None => t.clone(),
                }
            }
            // 2-D extent pin (archaeology R5): close the #7 class for 2-D
            // transforms. A faceted 2-D density (Kde2D) or 2-D-binned heatmap
            // (Bin2D) without an explicit extent would otherwise drift per panel
            // exactly as the 1-D trio did. Pin the spec's extent field(s) to the
            // global 2-D range over the full pre-facet batch, only when unset.
            TransformSpec::Kde2D(spec) if spec.extent.is_none() => {
                match crate::transform::kde_2d::global_extent(spec, batch) {
                    Some(extent) => TransformSpec::Kde2D(crate::transform::kde_2d::Kde2DSpec {
                        extent: Some(extent),
                        ..spec.clone()
                    }),
                    None => t.clone(),
                }
            }
            // Bin2D carries a separate `extent_x`/`extent_y` pair rather than a
            // 4-tuple. Pin each axis only when that axis is unset, so a partially
            // user-specified extent (e.g. only `extent_x`) keeps the user value
            // on the set axis and gains the global pin on the unset one.
            TransformSpec::Bin2D(spec)
                if spec.extent_x.is_none() || spec.extent_y.is_none() =>
            {
                match crate::transform::bin_2d::global_extent(spec, batch) {
                    Some((x_lo, x_hi, y_lo, y_hi)) => {
                        TransformSpec::Bin2D(crate::transform::bin_2d::Bin2DSpec {
                            extent_x: spec.extent_x.or(Some((x_lo, x_hi))),
                            extent_y: spec.extent_y.or(Some((y_lo, y_hi))),
                            ..spec.clone()
                        })
                    }
                    None => t.clone(),
                }
            }
            // DensityData carries a value-axis `extent: Option<(f64,f64)>`.
            // Without a pin, each facet panel would compute its own KDE range —
            // the same #7 defect class. DensityData does not nice (mirrors KDE):
            // the pinned extent is the raw global min/max over the full batch
            // (archaeology bug #7 extended to DensityData by round-3 T1).
            TransformSpec::DensityData(spec) if spec.extent.is_none() => {
                match crate::transform::density_data::global_extent(spec, batch) {
                    Some(extent) => TransformSpec::DensityData(
                        crate::transform::density_data::DensityDataSpec {
                            extent: Some(extent),
                            ..spec.clone()
                        },
                    ),
                    None => t.clone(),
                }
            }
            // Extent-DERIVING transforms (archaeology defect class #7 completion):
            // Hex, Raster, and DataBin compute their grid or bin boundaries from
            // the per-partition data range, so each facet panel drifts to its own
            // geometry. Pin each to the global pre-facet range, using the same
            // per-axis `.or(Some(...))` pattern as Bin2D so a user-set axis is
            // never clobbered.
            TransformSpec::Hex(spec)
                if spec.extent_x.is_none() || spec.extent_y.is_none() =>
            {
                match crate::transform::hex::global_extent(spec, batch) {
                    Some((x_lo, x_hi, y_lo, y_hi)) => {
                        TransformSpec::Hex(crate::transform::hex::HexSpec {
                            extent_x: spec.extent_x.or(Some((x_lo, x_hi))),
                            extent_y: spec.extent_y.or(Some((y_lo, y_hi))),
                            ..spec.clone()
                        })
                    }
                    None => t.clone(),
                }
            }
            TransformSpec::Raster(spec)
                if spec.extent_x.is_none() || spec.extent_y.is_none() =>
            {
                match crate::transform::raster::global_extent(spec, batch) {
                    Some((x_lo, x_hi, y_lo, y_hi)) => {
                        TransformSpec::Raster(crate::transform::raster::RasterSpec {
                            extent_x: spec.extent_x.or(Some((x_lo, x_hi))),
                            extent_y: spec.extent_y.or(Some((y_lo, y_hi))),
                            ..spec.clone()
                        })
                    }
                    None => t.clone(),
                }
            }
            TransformSpec::DataBin(spec) if spec.extent.is_none() => {
                match crate::transform::data_bin::global_extent(spec, batch) {
                    Some(extent) => {
                        TransformSpec::DataBin(crate::transform::data_bin::DataBinSpec {
                            extent: Some(extent),
                            ..spec.clone()
                        })
                    }
                    None => t.clone(),
                }
            }
            _ => t.clone(),
        })
        .collect()
}
