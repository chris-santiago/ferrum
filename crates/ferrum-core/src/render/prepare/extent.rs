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
            // ── 1-D single-extent transforms ──────────────────────────────────
            // Each calls its own `global_extent(spec, batch)` which owns the
            // niced-vs-raw decision (XFORM-08: Bin nices; all others are raw).
            TransformSpec::Kde(spec) if spec.extent.is_none() => {
                pin_single_extent(crate::transform::kde::global_extent(spec, batch), |e| {
                    TransformSpec::Kde(crate::transform::kde::KdeSpec {
                        extent: Some(e),
                        ..spec.clone()
                    })
                })
                .unwrap_or_else(|| t.clone())
            }
            TransformSpec::Bin(spec) if spec.extent.is_none() => {
                pin_single_extent(crate::transform::bin::global_extent(spec, batch), |e| {
                    TransformSpec::Bin(crate::transform::bin::BinSpec {
                        extent: Some(e),
                        ..spec.clone()
                    })
                })
                .unwrap_or_else(|| t.clone())
            }
            TransformSpec::Violin(spec) if spec.extent.is_none() => {
                pin_single_extent(crate::transform::violin::global_extent(spec, batch), |e| {
                    TransformSpec::Violin(crate::transform::violin::ViolinSpec {
                        extent: Some(e),
                        ..spec.clone()
                    })
                })
                .unwrap_or_else(|| t.clone())
            }
            // DensityData does not nice (mirrors KDE): pinned extent is the raw
            // global min/max (archaeology bug #7 extended to DensityData by T1).
            TransformSpec::DensityData(spec) if spec.extent.is_none() => {
                pin_single_extent(
                    crate::transform::density_data::global_extent(spec, batch),
                    |e| {
                        TransformSpec::DensityData(
                            crate::transform::density_data::DensityDataSpec {
                                extent: Some(e),
                                ..spec.clone()
                            },
                        )
                    },
                )
                .unwrap_or_else(|| t.clone())
            }
            TransformSpec::DataBin(spec) if spec.extent.is_none() => {
                pin_single_extent(
                    crate::transform::data_bin::global_extent(spec, batch),
                    |e| {
                        TransformSpec::DataBin(crate::transform::data_bin::DataBinSpec {
                            extent: Some(e),
                            ..spec.clone()
                        })
                    },
                )
                .unwrap_or_else(|| t.clone())
            }
            // ── 2-D dual-axis extent transforms ───────────────────────────────
            // Each carries per-axis `extent_x`/`extent_y` (or a single 4-tuple
            // `extent`). Pin each axis only when that axis is unset, so a
            // partially user-specified extent keeps the user value on the set
            // axis and gains the global pin on the unset one (archaeology R5).
            TransformSpec::Kde2D(spec) if spec.extent.is_none() => {
                // Kde2D carries a single `(f64, f64, f64, f64)` extent field,
                // not two separate per-axis fields — treat it like a 1-D pin.
                match crate::transform::kde_2d::global_extent(spec, batch) {
                    Some(extent) => TransformSpec::Kde2D(crate::transform::kde_2d::Kde2DSpec {
                        extent: Some(extent),
                        ..spec.clone()
                    }),
                    None => t.clone(),
                }
            }
            TransformSpec::Bin2D(spec)
                if spec.extent_x.is_none() || spec.extent_y.is_none() =>
            {
                pin_dual_extent(
                    crate::transform::bin_2d::global_extent(spec, batch),
                    spec.extent_x,
                    spec.extent_y,
                    |ex, ey| {
                        TransformSpec::Bin2D(crate::transform::bin_2d::Bin2DSpec {
                            extent_x: ex,
                            extent_y: ey,
                            ..spec.clone()
                        })
                    },
                )
                .unwrap_or_else(|| t.clone())
            }
            TransformSpec::Hex(spec)
                if spec.extent_x.is_none() || spec.extent_y.is_none() =>
            {
                pin_dual_extent(
                    crate::transform::hex::global_extent(spec, batch),
                    spec.extent_x,
                    spec.extent_y,
                    |ex, ey| {
                        TransformSpec::Hex(crate::transform::hex::HexSpec {
                            extent_x: ex,
                            extent_y: ey,
                            ..spec.clone()
                        })
                    },
                )
                .unwrap_or_else(|| t.clone())
            }
            TransformSpec::Raster(spec)
                if spec.extent_x.is_none() || spec.extent_y.is_none() =>
            {
                pin_dual_extent(
                    crate::transform::raster::global_extent(spec, batch),
                    spec.extent_x,
                    spec.extent_y,
                    |ex, ey| {
                        TransformSpec::Raster(crate::transform::raster::RasterSpec {
                            extent_x: ex,
                            extent_y: ey,
                            ..spec.clone()
                        })
                    },
                )
                .unwrap_or_else(|| t.clone())
            }
            _ => t.clone(),
        })
        .collect()
}

/// Helper for 1-D (single-extent) transforms: if `global` is `Some`, call
/// `build` to produce the pinned transform variant; otherwise return `None`
/// (caller falls back to the original transform unchanged).
#[inline]
fn pin_single_extent<T, F>(
    global: Option<(f64, f64)>,
    build: F,
) -> Option<T>
where
    F: FnOnce((f64, f64)) -> T,
{
    global.map(build)
}

/// Helper for 2-D (dual-axis) transforms: if `global` is `Some`, selectively
/// apply the pinned extents to the axes that were `None` (`current_x` / `current_y`
/// are the already-set per-axis values from the spec — `None` means unset). Calls
/// `build(pinned_x, pinned_y)` to produce the updated variant; returns `None` when
/// no global extent is available (caller falls back unchanged).
#[inline]
fn pin_dual_extent<T, F>(
    global: Option<(f64, f64, f64, f64)>,
    current_x: Option<(f64, f64)>,
    current_y: Option<(f64, f64)>,
    build: F,
) -> Option<T>
where
    F: FnOnce(Option<(f64, f64)>, Option<(f64, f64)>) -> T,
{
    global.map(|(x_lo, x_hi, y_lo, y_hi)| {
        build(
            current_x.or(Some((x_lo, x_hi))),
            current_y.or(Some((y_lo, y_hi))),
        )
    })
}
