//! Facet transform-extent pinning.
//!
//! When faceting is active, extent-carrying / extent-deriving transforms (Kde,
//! Bin, Violin, Kde2D, Bin2D, DensityData, Hex, Raster, DataBin) would each
//! compute their value-axis range from their own per-panel partition, making
//! positions non-comparable across panels (archaeology bug #7). This module
//! pins every such transform's extent to the global pre-facet range before
//! partitioning, but only when the user left the extent unset.

use arrow::record_batch::RecordBatch;

use crate::transform::color_bin_group::SpaceDemand;
use crate::transform::core::SpaceSlot;

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
/// note), not by this orchestrator. `Kde::count_scale_extent` (batch F,
/// quality-review cycle-4 finding B) is the one exception carried on a
/// non-`Bin` transform: it pins NICED, via `kde::count_scale_global_extent`
/// delegating to `bin::global_extent` on a synthetic `BinSpec`, because it is
/// standing in for a `Bin`'s own bin range, not `Kde`'s grid domain.
///
/// A spec that already carries an explicit extent (user-provided) is left
/// unchanged; `Bin2D`'s per-axis `extent_x`/`extent_y` are pinned independently so
/// a partially-specified extent keeps the user axis and pins only the unset one.
/// `Kde`'s `extent` and `count_scale_extent` are likewise pinned independently —
/// see the `Kde` arm's own comment. All non-extent-carrying transforms are
/// passed through unchanged; a `Branch` is rebuilt with its sub-chain pinned
/// by the same rule.
pub(crate) fn fix_transform_extents_for_facet(
    transforms: &[crate::transform::core::TransformSpec],
    batch: &RecordBatch,
) -> Vec<crate::transform::core::TransformSpec> {
    fix_transform_extents_for_facet_impl(transforms, batch, &[], None)
}

/// [`fix_transform_extents_for_facet`], additionally pinning a log-demanded
/// `Bin`/`Kde`/`Kde2D` in that axis's log space ([`SpaceDemand`]) rather
/// than data space — a linear-space pin can straddle zero and read as an
/// off-side `extent=` to the demanded step's own `apply_in_log_space`,
/// refusing a chart whose data is entirely on-side. `space_demand` is
/// `None` from every call site but `render::prepare`'s own (every
/// recursive call, every other transform's own facet-pin test), and with
/// `None` every step pins in data space exactly as
/// [`fix_transform_extents_for_facet`] pins it.
pub(crate) fn fix_transform_extents_for_facet_in_space(
    transforms: &[crate::transform::core::TransformSpec],
    batch: &RecordBatch,
    space_demand: Option<&SpaceDemand>,
) -> Vec<crate::transform::core::TransformSpec> {
    fix_transform_extents_for_facet_impl(transforms, batch, &[], space_demand)
}

fn fix_transform_extents_for_facet_impl(
    transforms: &[crate::transform::core::TransformSpec],
    batch: &RecordBatch,
    path: &[usize],
    space_demand: Option<&SpaceDemand>,
) -> Vec<crate::transform::core::TransformSpec> {
    use crate::transform::core::TransformSpec;

    transforms
        .iter()
        .enumerate()
        .map(|(index, t)| {
        let id: Vec<usize> = path.iter().copied().chain([index]).collect();
        // Whichever axis (if any) this step's own key is demanded into —
        // the same lookup `Run::step` makes per step, read here BEFORE any
        // transform runs so the pin lands in that space from the start.
        let space = space_demand.map(|demand| demand.space(&id)).unwrap_or_default();
        match t {
            // ── 1-D single-extent transforms ──────────────────────────────────
            // Each calls its own `global_extent(spec, batch)` which owns the
            // niced-vs-raw decision (XFORM-08: Bin nices; all others are raw).
            //
            // `Kde` carries TWO independent extent-like slots — `extent`
            // (its own grid domain, pinned RAW like every non-`Bin`
            // transform here) and `count_scale_extent` (the histogram bin
            // range `KdeSpec::count_scale` derives its width from, pinned
            // NICED — mirroring the `Bin` arm below exactly, via
            // `kde::count_scale_global_extent`'s synthetic `BinSpec`). These
            // are pinned independently (unlike every other arm here, which
            // is all-or-nothing per variant) because a spec can have one
            // set and not the other — e.g. an explicit `extent=` with
            // `count_scale` still set, or vice versa (quality-review
            // cycle-4 finding B: the un-pinned `count_scale_extent` let a
            // faceted `count_scale` bin over each panel's own local span
            // while the real `Bin` binned over the whole-dataset span).
            TransformSpec::Kde(spec) => {
                let mut pinned = spec.clone();
                let mut changed = false;
                let axis_space = space.of(SpaceSlot::Field);
                if spec.extent.is_none() {
                    let extent = match axis_space {
                        Some(s) => crate::transform::kde::global_extent_in_space(spec, batch, s),
                        None => crate::transform::kde::global_extent(spec, batch),
                    };
                    if let Some(e) = extent {
                        pinned.extent = Some(e);
                        changed = true;
                    }
                }
                if spec.count_scale.is_some() && spec.count_scale_extent.is_none() {
                    let extent = match axis_space {
                        Some(s) => crate::transform::kde::count_scale_global_extent_in_space(spec, batch, s),
                        None => crate::transform::kde::count_scale_global_extent(spec, batch),
                    };
                    if let Some(e) = extent {
                        pinned.count_scale_extent = Some(e);
                        changed = true;
                    }
                }
                if changed { TransformSpec::Kde(pinned) } else { t.clone() }
            }
            TransformSpec::Bin(spec) if spec.extent.is_none() => {
                let extent = match space.of(SpaceSlot::Field) {
                    Some(slot) => crate::transform::bin::global_extent_in_space(spec, batch, slot),
                    None => crate::transform::bin::global_extent(spec, batch),
                };
                pin_single_extent(extent, |e| {
                    TransformSpec::Bin(crate::transform::bin::BinSpec {
                        extent: Some(e),
                        ..spec.clone()
                    })
                })
                .unwrap_or_else(|| t.clone())
            }
            TransformSpec::Kde2D(spec) if spec.extent.is_none() && !space.is_empty() => {
                pin_single_extent(crate::transform::kde_2d::global_extent_in_space(spec, batch, &space), |e| {
                    TransformSpec::Kde2D(crate::transform::kde_2d::Kde2DSpec {
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
            // `Smooth`/`Robust`: `x_range_pad` without a caller-written
            // `x_range` means "extend past the data's own range", so the
            // facet pin gives every panel the SAME `x_range` to extend
            // from — the whole
            // pre-facet batch's own `x` extent, in data units, under the
            // step's demanded `X` slot the on-side extent mapped back
            // (`log_space::column_extent_in_space`, mirroring
            // `bin::global_extent_in_space`). `x_range_pad` itself is left
            // on the spec, so each panel still pads this shared range in
            // its own space inside `apply_one_group`. A caller-written
            // `x_range` is never touched here — the explicit guard mirrors
            // every other arm's "only pin what the user left unset" rule.
            TransformSpec::Smooth(spec) if spec.x_range.is_none() && spec.x_range_pad.is_some() => {
                let extent = match space.of(SpaceSlot::X) {
                    Some(slot) => crate::transform::log_space::column_extent_in_space(batch, &spec.x, SpaceSlot::X, slot, "stat_smooth"),
                    None => crate::transform::numeric_util::column_extent(batch, &spec.x),
                };
                pin_single_extent(extent, |(lo, hi)| {
                    TransformSpec::Smooth(crate::transform::smooth::SmoothSpec {
                        x_range: Some([lo, hi]),
                        ..spec.clone()
                    })
                })
                .unwrap_or_else(|| t.clone())
            }
            TransformSpec::Robust(spec) if spec.x_range.is_none() && spec.x_range_pad.is_some() => {
                let extent = match space.of(SpaceSlot::X) {
                    Some(slot) => crate::transform::log_space::column_extent_in_space(batch, &spec.x, SpaceSlot::X, slot, "Robust"),
                    None => crate::transform::numeric_util::column_extent(batch, &spec.x),
                };
                pin_single_extent(extent, |(lo, hi)| {
                    TransformSpec::Robust(crate::transform::robust::RobustSpec {
                        x_range: Some([lo, hi]),
                        ..spec.clone()
                    })
                })
                .unwrap_or_else(|| t.clone())
            }
            // ── 2-D dual-axis extent transforms ───────────────────────────────
            // Each carries per-axis `extent_x`/`extent_y` (or a single 4-tuple
            // `extent`). Pin each axis only when that axis is unset, so a
            // partially user-specified extent keeps the user value on the set
            // axis and gains the global pin on the unset one (archaeology R5).
            // Kde2D carries a single `(f64, f64, f64, f64)` extent field, not
            // two separate per-axis fields — `pin_single_extent` is generic
            // over the extent's tuple shape, so it pins this the same way as
            // the 1-D arms above rather than needing its own hand-rolled match.
            TransformSpec::Kde2D(spec) if spec.extent.is_none() => {
                pin_single_extent(crate::transform::kde_2d::global_extent(spec, batch), |e| {
                    TransformSpec::Kde2D(crate::transform::kde_2d::Kde2DSpec {
                        extent: Some(e),
                        ..spec.clone()
                    })
                })
                .unwrap_or_else(|| t.clone())
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
            // A branch's sub-chain runs per panel too, so its extent-carrying
            // steps pin against the same whole-frame batch: for an operand's
            // framed branch, that operand's own rows of it. A frame that
            // cannot pick them is left unpinned, since the run refuses it.
            TransformSpec::Branch(branch) => {
                let transforms = match &branch.frame {
                    None => fix_transform_extents_for_facet_impl(&branch.transforms, batch, &id, space_demand),
                    Some(frame) => match frame.rows(&branch.name, batch, &[]) {
                        Ok(rows) => fix_transform_extents_for_facet_impl(&branch.transforms, &rows, &id, space_demand),
                        Err(_) => branch.transforms.clone(),
                    },
                };
                TransformSpec::Branch(crate::transform::branch::BranchSpec { transforms, ..branch.clone() })
            }
            // A composed copy's step runs per panel as written, so it pins
            // like the same step written directly — in data space always:
            // a composition's own step is never lowered (`Run::chain`
            // sends a composed copy to `place`/`run_copy`, which never
            // sets `log_space`), so no space demand ever reaches it here
            // either, and passing `None` states that rather than leaving
            // it implicit.
            TransformSpec::Composed(composed) => {
                let step = fix_transform_extents_for_facet_impl(std::slice::from_ref(&*composed.step), batch, &id, None);
                match step.into_iter().next() {
                    Some(step) => TransformSpec::Composed(crate::transform::composed::ComposedSpec {
                        group: composed.group.clone(),
                        step: Box::new(step),
                    }),
                    None => t.clone(),
                }
            }
            _ => t.clone(),
        }
        })
        .collect()
}

/// Helper for single-extent transforms: if `global` is `Some`, call `build`
/// to produce the pinned transform variant; otherwise return `None` (caller
/// falls back to the original transform unchanged). Generic over the extent
/// value's own shape (`E`) rather than fixed to `(f64, f64)`, so both the
/// 1-D arms' `(f64, f64)` extent and `Kde2D`'s single `(f64, f64, f64, f64)`
/// extent field share this one spelling of the pin idiom.
#[inline]
fn pin_single_extent<E, T, F>(global: Option<E>, build: F) -> Option<T>
where
    F: FnOnce(E) -> T,
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::Arc;

    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;

    use crate::transform::core::TransformSpec;
    use crate::transform::registry::{all_transform_kinds, entry_for_kind};

    use super::{fix_transform_extents_for_facet, fix_transform_extents_for_facet_in_space};

    /// A one-column batch: Float64 field `v` = [0.3, 0.5, 1.2, 4.8, 9.7].
    /// Shared by every 1-D single-extent pin probe below (`Bin`, `Violin`,
    /// `Kde`, `DataBin`, `DensityData` all read a field literally named
    /// `v`) — these probes only need a finite, non-empty range to pin
    /// against ("did the arm pin it"), not the disjoint-panel values
    /// `render/prepare/mod.rs`'s richer per-variant tests use to pin exact
    /// niced-vs-raw numeric outputs ("what did it pin it to").
    fn v_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, false)]));
        RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![0.3, 0.5, 1.2, 4.8, 9.7]))],
        )
        .unwrap()
    }

    /// A two-column batch: Float64 `x`/`y`. Shared by every 2-D dual-extent
    /// pin probe below (`Kde2D`, `Bin2D`, `Hex`, `Raster`).
    fn xy_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![0.0, 1.0, 10.0])),
                Arc::new(Float64Array::from(vec![0.0, 1.0, 100.0])),
            ],
        )
        .unwrap()
    }

    // ── Per-kind behavioral pin probes ─────────────────────────────────────
    //
    // Each builds a real `TransformSpec` with its extent unset, runs the
    // REAL `fix_transform_extents_for_facet`, and asserts the RETURNED
    // spec's extent field(s) came back `Some(..)`. This observes the arm
    // doing its job rather than asserting a name appears in a literal —
    // spec-review cycle 1 proved by mutation (forcing the real `Bin2D`
    // arm's guard to `if false`, so `Bin2D` silently stopped being pinned)
    // that the previous hand-typed `handled` literal did not catch an
    // existing arm disappearing, only a new field name appearing with no
    // matching literal entry. Each probe returns its own kind name so the
    // guard test below can fold "is this kind behaviorally covered" and
    // "is the covered set exactly the registry's extent-carrying set" into
    // one assertion, mirroring `assert_params_match_serde_fields`'s
    // return-the-kind-you-checked pattern (`transform/registry/tests.rs`).

    fn assert_bin_pins() -> &'static str {
        use crate::transform::bin::BinSpec;
        use crate::transform::bin_mode::BinMode;
        let spec = BinSpec {
            written_groupby: Vec::new(),
            field: "v".into(),
            mode: BinMode::Fixed { n: 10 },
            extent: None,
            nice: true,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
            agg_field: None,
            agg: None,
            name: None,
            per_row: false,
            aggregates: None,
        };
        let out = fix_transform_extents_for_facet(&[TransformSpec::Bin(spec)], &v_batch());
        let TransformSpec::Bin(pinned) = &out[0] else { panic!("expected a Bin transform back") };
        assert!(pinned.extent.is_some(), "bin: extent was not pinned — is the Bin arm still there?");
        "bin"
    }

    fn assert_violin_pins() -> &'static str {
        use crate::transform::kde::BandwidthSpec;
        use crate::transform::violin::ViolinSpec;
        let spec = ViolinSpec {
            field: "v".into(),
            groupby: Vec::new(),
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 64,
            width: 0.4,
            extent: None,
            shared_extent: false,
            name: None,
            horizontal: false,
        };
        let out = fix_transform_extents_for_facet(&[TransformSpec::Violin(spec)], &v_batch());
        let TransformSpec::Violin(pinned) = &out[0] else {
            panic!("expected a Violin transform back")
        };
        assert!(pinned.extent.is_some(), "violin: extent was not pinned — is the Violin arm still there?");
        "violin"
    }

    fn assert_kde_pins() -> &'static str {
        use crate::transform::kde::{BandwidthSpec, KdeSpec};
        let spec = KdeSpec {
            field: "v".into(),
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 64,
            extent: None,
            cumulative: false,
            shared_extent: false,
            kernel: "gaussian".into(),
            groupby: vec![],
            count_scale: None,
            count_scale_extent: None,
            name: None,
        };
        let out = fix_transform_extents_for_facet(&[TransformSpec::Kde(spec)], &v_batch());
        let TransformSpec::Kde(pinned) = &out[0] else { panic!("expected a Kde transform back") };
        assert!(pinned.extent.is_some(), "kde: extent was not pinned — is the Kde arm still there?");
        "kde"
    }

    fn assert_kde2d_pins() -> &'static str {
        use crate::transform::kde::BandwidthSpec;
        use crate::transform::kde_2d::Kde2DSpec;
        let spec = Kde2DSpec {
            x: "x".into(),
            y: "y".into(),
            bandwidth: BandwidthSpec::Scott,
            n: 16,
            extent: None,
            groupby: vec![],
            name: None,
        };
        let out = fix_transform_extents_for_facet(&[TransformSpec::Kde2D(spec)], &xy_batch());
        let TransformSpec::Kde2D(pinned) = &out[0] else {
            panic!("expected a Kde2D transform back")
        };
        assert!(pinned.extent.is_some(), "kde_2d: extent was not pinned — is the Kde2D arm still there?");
        "kde_2d"
    }

    fn assert_bin2d_pins() -> &'static str {
        use crate::transform::bin_2d::Bin2DSpec;
        use crate::transform::bin_mode::BinMode;
        let spec = Bin2DSpec {
            x: "x".into(),
            y: "y".into(),
            bins_x: BinMode::Fixed { n: 10 },
            bins_y: BinMode::Fixed { n: 10 },
            extent_x: None,
            extent_y: None,
            cumulative: false,
            name: None,
        };
        let out = fix_transform_extents_for_facet(&[TransformSpec::Bin2D(spec)], &xy_batch());
        let TransformSpec::Bin2D(pinned) = &out[0] else {
            panic!("expected a Bin2D transform back")
        };
        assert!(
            pinned.extent_x.is_some() && pinned.extent_y.is_some(),
            "bin_2d: extent_x/extent_y were not pinned — is the Bin2D arm still there?"
        );
        "bin_2d"
    }

    fn assert_hex_pins() -> &'static str {
        use crate::transform::hex::HexSpec;
        let spec = HexSpec {
            x: "x".into(),
            y: "y".into(),
            bin_size: None,
            aggregate: Default::default(),
            field: None,
            name: None,
            extent_x: None,
            extent_y: None,
        };
        let out = fix_transform_extents_for_facet(&[TransformSpec::Hex(spec)], &xy_batch());
        let TransformSpec::Hex(pinned) = &out[0] else { panic!("expected a Hex transform back") };
        assert!(
            pinned.extent_x.is_some() && pinned.extent_y.is_some(),
            "hex: extent_x/extent_y were not pinned — is the Hex arm still there?"
        );
        "hex"
    }

    fn assert_raster_pins() -> &'static str {
        use crate::transform::raster::{RasterSpec, ResolutionSpec};
        let spec = RasterSpec {
            x: "x".into(),
            y: "y".into(),
            aggregate: Default::default(),
            field: None,
            resolution: ResolutionSpec::Fixed(4),
            min_count: None,
            log_scale: false,
            name: None,
            extent_x: None,
            extent_y: None,
        };
        let out = fix_transform_extents_for_facet(&[TransformSpec::Raster(spec)], &xy_batch());
        let TransformSpec::Raster(pinned) = &out[0] else {
            panic!("expected a Raster transform back")
        };
        assert!(
            pinned.extent_x.is_some() && pinned.extent_y.is_some(),
            "raster: extent_x/extent_y were not pinned — is the Raster arm still there?"
        );
        "raster"
    }

    fn assert_data_bin_pins() -> &'static str {
        use crate::transform::data_bin::DataBinSpec;
        let spec = DataBinSpec {
            field: "v".into(),
            as_: None,
            maxbins: Some(10),
            step: None,
            nice: false,
            name: None,
            extent: None,
        };
        let out = fix_transform_extents_for_facet(&[TransformSpec::DataBin(spec)], &v_batch());
        let TransformSpec::DataBin(pinned) = &out[0] else {
            panic!("expected a DataBin transform back")
        };
        assert!(pinned.extent.is_some(), "data_bin: extent was not pinned — is the DataBin arm still there?");
        "data_bin"
    }

    fn assert_density_data_pins() -> &'static str {
        use crate::transform::density_data::DensityDataSpec;
        let spec = DensityDataSpec {
            field: "v".into(),
            bandwidth: None,
            groupby: None,
            extent: None,
            steps: None,
            cumulative: false,
            kernel: crate::transform::kde::default_kernel(),
            as_: ("value".into(), "density".into()),
            name: None,
        };
        let out = fix_transform_extents_for_facet(&[TransformSpec::DensityData(spec)], &v_batch());
        let TransformSpec::DensityData(pinned) = &out[0] else {
            panic!("expected a DensityData transform back")
        };
        assert!(
            pinned.extent.is_some(),
            "density_data: extent was not pinned — is the DensityData arm still there?"
        );
        "density_data"
    }

    /// `fix_transform_extents_for_facet`'s match ends in a wildcard
    /// (`_ => t.clone()`), which is deliberate — `TransformSpec` has 43
    /// variants and only 9 carry an extent to pin, so an exhaustive
    /// wildcard-free match (the `ScaleSpec::positional_extent` precedent)
    /// would add 34 no-op arms. But a wildcard means a new extent-carrying
    /// variant can land with no pin and nothing fails: archaeology bug #7's
    /// exact failure mode, which already re-armed once for `DensityData`
    /// (`render/prepare/mod.rs`'s density-data facet-extent test comment
    /// records the incident).
    ///
    /// This guard closes that gap at test time instead of compile time, and
    /// derives BOTH sides of its equality from real code — not a
    /// hand-maintained literal on either side (spec-review cycle 1 found the
    /// first version of this test derived only the right-hand side):
    ///
    /// - `extent_carrying` (right-hand side): every kind in
    ///   `all_transform_kinds()` (the canonical enumeration — the same
    ///   `for_each_transform!` table `TransformSpec`'s 43 variants are
    ///   generated from) whose `entry_for_kind(kind).params` carries one of
    ///   the four extent field names. The registry's `params` list is
    ///   itself drift-proven equal to each spec struct's real serde fields
    ///   by `params_match_spec_struct_serde_fields`
    ///   (`transform/registry/tests.rs`), so this derives "does this kind
    ///   carry an extent field" from that data instead of a second
    ///   hand-maintained field list — the
    ///   `accepted_keys_for_scale_type_matches_every_variants_serialized_keys`
    ///   (`spec/encoding.rs`) precedent.
    /// - `covered` (left-hand side): the kinds whose `assert_*_pins` probe,
    ///   just above, actually ran the REAL `fix_transform_extents_for_facet`
    ///   and observed its output extent field(s) come back `Some(..)`. A
    ///   probe panics (with a message naming the kind) before this line is
    ///   even reached if its arm is disabled, deleted, or falls through to
    ///   the wildcard — so `covered` reflects live behavior, not a list a
    ///   human typed and could forget to update.
    ///
    /// Field NAME is not field TYPE: `outliers`'s `extent: f64` is an IQR
    /// multiplier (`transform/outliers.rs`'s `OutliersSpec::extent`), not a
    /// value-axis range, and the registry's string-keyed `params` cannot
    /// express that distinction. It is excluded by name, with this comment
    /// as the reason, rather than silently matched.
    #[test]
    fn handled_extent_kinds_match_extent_carrying_spec_fields() {
        const EXTENT_FIELD_NAMES: &[&str] =
            &["extent", "extent_x", "extent_y", "count_scale_extent"];
        const NAME_ONLY_EXCLUSIONS: &[&str] = &["outliers"];

        let extent_carrying: BTreeSet<&str> = all_transform_kinds()
            .into_iter()
            .filter(|kind| !NAME_ONLY_EXCLUSIONS.contains(kind))
            .filter(|kind| {
                let params = entry_for_kind(kind).unwrap().params;
                EXTENT_FIELD_NAMES.iter().any(|f| params.contains(f))
            })
            .collect();

        // Every element here only enters `covered` if the REAL function
        // actually pinned that kind's extent field(s) for an unset-extent
        // spec, just above — a disabled or deleted arm panics before this
        // line runs, rather than silently leaving the set unchanged.
        let covered: BTreeSet<&str> = [
            assert_bin_pins(),
            assert_violin_pins(),
            assert_kde_pins(),
            assert_kde2d_pins(),
            assert_bin2d_pins(),
            assert_hex_pins(),
            assert_raster_pins(),
            assert_data_bin_pins(),
            assert_density_data_pins(),
        ]
        .into_iter()
        .collect();

        assert_eq!(
            covered, extent_carrying,
            "fix_transform_extents_for_facet's behaviorally-covered kinds have drifted from \
             the extent-carrying TransformSpec variants (a field named extent / extent_x / \
             extent_y / count_scale_extent).\n\
             covered here but no longer/never extent-carrying: {:?}\n\
             extent-carrying but NOT covered by a pin probe above — a new variant is missing \
             both its facet pin and its coverage: {:?}",
            covered.difference(&extent_carrying).collect::<Vec<_>>(),
            extent_carrying.difference(&covered).collect::<Vec<_>>(),
        );
    }

    /// A `Branch` runs per panel like the top-level chain, so the
    /// extent-carrying steps inside it, nested branches included, pin
    /// against the whole frame; a branch left unpinned would bin each panel
    /// over its own span.
    #[test]
    fn a_branch_sub_chain_is_pinned_recursively() {
        use crate::transform::branch::BranchSpec;
        use crate::transform::bin::BinSpec;
        use crate::transform::bin_mode::BinMode;
        let bin = TransformSpec::Bin(BinSpec {
            written_groupby: Vec::new(),
            field: "v".into(),
            mode: BinMode::Fixed { n: 10 },
            extent: None,
            nice: true,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
            agg_field: None,
            agg: None,
            name: None,
            per_row: false,
            aggregates: None,
        });
        let branch = |name: &str, transforms| {
            TransformSpec::Branch(BranchSpec { name: name.into(), input: None, frame: None, transforms })
        };
        let spec = branch("outer", vec![bin.clone(), branch("inner", vec![bin])]);
        let out = fix_transform_extents_for_facet(&[spec], &v_batch());
        let TransformSpec::Branch(outer) = &out[0] else { panic!("expected the Branch back") };
        let TransformSpec::Bin(pinned) = &outer.transforms[0] else { panic!("expected a Bin") };
        assert!(pinned.extent.is_some(), "the branch's own Bin was not pinned");
        let TransformSpec::Branch(inner) = &outer.transforms[1] else { panic!("expected the nested Branch") };
        let TransformSpec::Bin(pinned) = &inner.transforms[0] else { panic!("expected a Bin") };
        assert!(pinned.extent.is_some(), "the nested branch's Bin was not pinned");
        assert_eq!((outer.name.as_str(), inner.name.as_str()), ("outer", "inner"));
    }

    /// An operand's framed branch pins its statistics against that
    /// operand's own rows of the whole frame: the right operand's `v`
    /// (100..105, renamed `v__r` in the combined batch), never the left's
    /// `v` (0..1) the combined batch holds under that name. Rejects pinning
    /// from the combined batch, which bins the right's histogram over the
    /// left's range so every bar is empty.
    #[test]
    fn a_framed_branch_pins_against_its_operands_own_rows() {
        use crate::transform::bin::BinSpec;
        use crate::transform::bin_mode::BinMode;
        use arrow::array::{BooleanArray, Float64Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;
        let combined = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("v", DataType::Float64, true),
                Field::new("v__r", DataType::Float64, true),
                Field::new("right", DataType::Boolean, false),
            ])),
            vec![
                Arc::new(Float64Array::from(vec![Some(0.0), Some(1.0), None, None])),
                Arc::new(Float64Array::from(vec![None, None, Some(100.0), Some(105.0)])),
                Arc::new(BooleanArray::from(vec![false, false, true, true])),
            ],
        )
        .unwrap();
        let spec: TransformSpec = serde_json::from_value(serde_json::json!({
            "type": "branch", "name": "op",
            "frame": {"marker": "right", "renames": {"v": "v__r"}, "columns": ["v"]},
            "transforms": [{"type": "bin", "field": "v", "bin_count": 5, "nice": false}],
        }))
        .unwrap();
        let out = fix_transform_extents_for_facet(&[spec], &combined);
        let TransformSpec::Branch(op) = &out[0] else { panic!("expected the Branch back") };
        let TransformSpec::Bin(pinned) = &op.transforms[0] else { panic!("expected a Bin") };
        assert_eq!(pinned.extent, Some((100.0, 105.0)));
        assert!(op.frame.is_some(), "the frame is kept");
    }

    // ── The facet pin under a log-demanded step ────────────────────────────
    //
    // A demanded step's facet pin must land ON-SIDE: a plain linear pin can
    // straddle zero (the data minimum niced down past it), which the
    // demanded step's own `apply_in_log_space` would then refuse as if the
    // caller had written that off-side `extent=` directly, so a faceted
    // log-scaled chart over all-positive data would refuse.

    use crate::transform::color_bin_group::{LogAxisSpace, SlotSpace, SpaceDemand, StepSpace};
    use crate::transform::core::SpaceSlot;

    /// A step demanded on its one `field` slot along `x` in `space`.
    fn field_slot(space: LogAxisSpace) -> StepSpace {
        StepSpace::for_slots([(SpaceSlot::Field, SlotSpace { space, axes: vec![crate::spec::layer::PositionalAxis::X], records_caller_input: true })])
    }

    fn positive_space() -> LogAxisSpace {
        LogAxisSpace { base: 10.0, side: crate::scale::log::LogSide::Positive }
    }

    fn negative_space() -> LogAxisSpace {
        LogAxisSpace { base: 10.0, side: crate::scale::log::LogSide::Negative }
    }

    fn batch_of(field: &str, values: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(field, DataType::Float64, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(values))]).unwrap()
    }

    /// A log-demanded `Bin`'s facet pin comes back strictly on-side (both
    /// endpoints positive), never straddling zero the way the equivalent
    /// linear pin would for this exact data (the data minimum of `1.0`
    /// niced down to `0.0` in data units). Rejects a fix that pins in
    /// linear space and leaves the demanded step to refuse it.
    #[test]
    fn log_demanded_bin_facet_pin_stays_on_side() {
        use crate::transform::bin::BinSpec;
        use crate::transform::bin_mode::BinMode;
        let batch = batch_of("v", (1..=40).map(|v| v as f64).collect());
        let spec = BinSpec {
            written_groupby: Vec::new(),
            field: "v".into(),
            mode: BinMode::Sturges,
            extent: None,
            nice: true,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
            agg_field: None,
            agg: None,
            name: None,
            per_row: false,
            aggregates: None,
        };
        // The linear pin DOES straddle zero for this data (confirms the
        // test exercises the real defect rather than a fixture that never
        // could have crossed it).
        let linear = fix_transform_extents_for_facet(&[TransformSpec::Bin(spec.clone())], &batch);
        let TransformSpec::Bin(linear_pinned) = &linear[0] else { panic!("expected a Bin") };
        let (linear_lo, _) = linear_pinned.extent.expect("linear pin");
        assert!(linear_lo <= 0.0, "fixture must straddle zero under a linear pin to be a real repro: {linear_lo}");

        let demand = SpaceDemand::single(vec![0], field_slot(positive_space()));
        let out = fix_transform_extents_for_facet_in_space(&[TransformSpec::Bin(spec)], &batch, Some(&demand));
        let TransformSpec::Bin(pinned) = &out[0] else { panic!("expected a Bin") };
        let (lo, hi) = pinned.extent.expect("log-space pin");
        assert!(lo > 0.0 && hi > 0.0, "a positive-side pin must stay on-side: ({lo}, {hi})");

        // The pinned extent is not itself refused by the demanded step it
        // feeds: an on-side extent maps into the step's log space.
        use crate::transform::context::TransformContext;
        let demanded_ctx = TransformContext { log_space: Some(std::rc::Rc::new(field_slot(positive_space()))), ..TransformContext::default() };
        crate::transform::bin::apply_with_context(&pinned.clone(), &batch, &demanded_ctx)
            .expect("the pinned extent must not be refused by the step it was pinned for");
    }

    /// A log-demanded `Bin`'s facet pin on all-negative data (the mirrored
    /// axis) also stays on-side: both endpoints negative.
    #[test]
    fn log_demanded_bin_facet_pin_stays_on_the_negative_side() {
        use crate::transform::bin::BinSpec;
        use crate::transform::bin_mode::BinMode;
        let batch = batch_of("v", (1..=40).map(|v| -(v as f64)).collect());
        let spec = BinSpec {
            written_groupby: Vec::new(),
            field: "v".into(),
            mode: BinMode::Sturges,
            extent: None,
            nice: true,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
            agg_field: None,
            agg: None,
            name: None,
            per_row: false,
            aggregates: None,
        };
        let demand = SpaceDemand::single(vec![0], field_slot(negative_space()));
        let out = fix_transform_extents_for_facet_in_space(&[TransformSpec::Bin(spec)], &batch, Some(&demand));
        let TransformSpec::Bin(pinned) = &out[0] else { panic!("expected a Bin") };
        let (lo, hi) = pinned.extent.expect("log-space pin");
        assert!(lo < 0.0 && hi < 0.0, "a negative-side pin must stay on-side: ({lo}, {hi})");
        assert!(lo <= hi, "the pinned extent keeps lo <= hi: ({lo}, {hi})");

        // The step the pin feeds accepts it and bins every row in data
        // order inside it.
        let demanded_ctx = crate::transform::context::TransformContext {
            log_space: Some(std::rc::Rc::new(field_slot(negative_space()))),
            ..Default::default()
        };
        let binned = crate::transform::bin::apply_with_context(pinned, &batch, &demanded_ctx).expect("the step accepts its own pin");
        let starts = crate::render::arrow_cast::col_as_f64(&binned, "bin_start").unwrap();
        let ends = crate::render::arrow_cast::col_as_f64(&binned, "bin_end").unwrap();
        let within = |v: Option<f64>| v.is_some_and(|v| lo * (1.0 + 1e-9) <= v && v <= hi * (1.0 - 1e-9));
        assert!(starts.iter().chain(&ends).all(|&v| within(v)), "every edge inside ({lo}, {hi}): {starts:?} {ends:?}");
        assert!(starts.windows(2).all(|w| w[0] < w[1]), "the bins ascend in data order: {starts:?}");
        let counts = crate::render::arrow_cast::col_as_f64(&binned, "count").unwrap();
        assert_eq!(counts.iter().flatten().sum::<f64>(), 40.0, "every row is binned");
    }

    /// A log-demanded `Kde`'s count-scale facet pin (the mirrored `Bin` it
    /// stands in for) also stays on-side, mirroring the real histogram's
    /// own pin exactly.
    #[test]
    fn log_demanded_kde_count_scale_facet_pin_stays_on_side() {
        use crate::transform::bin_mode::BinMode;
        use crate::transform::kde::{BandwidthSpec, KdeSpec};
        let batch = batch_of("v", (1..=40).map(|v| v as f64).collect());
        let spec = KdeSpec {
            field: "v".into(),
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 64,
            extent: None,
            cumulative: false,
            shared_extent: false,
            kernel: "gaussian".into(),
            groupby: vec![],
            count_scale: Some(BinMode::Sturges),
            count_scale_extent: None,
            name: None,
        };
        let demand = SpaceDemand::single(vec![0], field_slot(positive_space()));
        let out = fix_transform_extents_for_facet_in_space(&[TransformSpec::Kde(spec)], &batch, Some(&demand));
        let TransformSpec::Kde(pinned) = &out[0] else { panic!("expected a Kde") };
        let (lo, hi) = pinned.extent.expect("log-space grid pin");
        assert!(lo > 0.0 && hi > 0.0, "the grid pin must stay on-side: ({lo}, {hi})");
        let (clo, chi) = pinned.count_scale_extent.expect("log-space count-scale pin");
        assert!(clo > 0.0 && chi > 0.0, "the count-scale pin must stay on-side: ({clo}, {chi})");
    }

    // ── `Smooth`/`Robust`: the `x_range` facet pin ──────────────────────────
    //
    // Unlike the unconditional arms above, these only fire when the caller
    // left `x_range` unset AND wrote `x_range_pad` (the arm's own guard) —
    // see this file's `TransformSpec::Smooth`/`TransformSpec::Robust` arms'
    // doc comment for why. Both routes (demanded and plain) must equal the
    // SAME function the arm itself calls, not a value a test recomputes by
    // hand — `column_extent_in_space` on a demanded slot, `column_extent`
    // in data space otherwise.

    /// A step demanded on its `x` slot along `x` in `space` — the `Smooth`/
    /// `Robust` twin of `field_slot` above (their slot is `X`, not `Field`).
    fn x_slot(space: LogAxisSpace) -> StepSpace {
        StepSpace::for_slots([(SpaceSlot::X, SlotSpace { space, axes: vec![crate::spec::layer::PositionalAxis::X], records_caller_input: true })])
    }

    fn smooth_pad_spec(x_range_pad: f64) -> crate::transform::smooth::SmoothSpec {
        crate::transform::smooth::SmoothSpec {
            residual_columns: Default::default(),
            x: "x".into(), y: "y".into(),
            method: crate::transform::smooth::SmoothMethod::Lm,
            ci: None,
            bandwidth: 0.0, degree: 1, k: crate::transform::gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
            x_bins: None, x_estimator: None, output: crate::transform::smooth::SmoothOutput::Fitted,
            inject_zero_ref: false, inject_metrics: false, x_range: None, x_range_pad: Some(x_range_pad),
            groupby: vec![],
            name: None,
        }
    }

    /// A demanded `Smooth`'s facet pin equals `column_extent_in_space`'s
    /// own on-side result exactly — not a linear pin that happens to stay
    /// on-side for this data, but the SAME value the arm itself calls.
    #[test]
    fn log_demanded_smooth_facet_pin_equals_column_extent_in_space() {
        let batch = xy_batch();
        let spec = smooth_pad_spec(0.1);
        let step_space = x_slot(positive_space());
        let demand = SpaceDemand::single(vec![0], x_slot(positive_space()));
        let slot_space = step_space.of(SpaceSlot::X).expect("x slot demanded");
        let expected = crate::transform::log_space::column_extent_in_space(&batch, "x", SpaceSlot::X, slot_space, "stat_smooth");
        let out = fix_transform_extents_for_facet_in_space(&[TransformSpec::Smooth(spec)], &batch, Some(&demand));
        let TransformSpec::Smooth(pinned) = &out[0] else { panic!("expected a Smooth") };
        let range = pinned.x_range.expect("demanded pin");
        assert!(range[0] > 0.0 && range[1] > 0.0, "a positive-side pin must stay on-side: {range:?}");
        let expected = expected.expect("the fixture's x column has a real extent");
        assert_eq!(range, [expected.0, expected.1], "the pin must equal column_extent_in_space's own on-side result, not a value this test recomputes");
    }

    /// The same `Smooth` with no demand at all pins from `column_extent` in
    /// plain data space — the arm's other branch.
    #[test]
    fn undemanded_smooth_facet_pin_equals_column_extent_in_data_space() {
        let batch = xy_batch();
        let spec = smooth_pad_spec(0.1);
        let expected = crate::transform::numeric_util::column_extent(&batch, "x").expect("the fixture's x column has a real extent");
        let out = fix_transform_extents_for_facet(&[TransformSpec::Smooth(spec)], &batch);
        let TransformSpec::Smooth(pinned) = &out[0] else { panic!("expected a Smooth") };
        let range = pinned.x_range.expect("plain pin");
        assert_eq!(range, [expected.0, expected.1], "the undemanded pin must equal column_extent's own data-space result");
    }

    /// `Robust` mirrors `Smooth`'s arm exactly — same guard, same two
    /// routes, same underlying extent functions.
    fn robust_pad_spec(x_range_pad: f64) -> crate::transform::robust::RobustSpec {
        crate::transform::robust::RobustSpec {
            residual_columns: Default::default(),
            x: "x".into(), y: "y".into(),
            n_grid: 5, ci: None, huber_c: 1.345, max_iter: 50, tol: 1e-9,
            output: crate::transform::smooth::SmoothOutput::Fitted,
            inject_zero_ref: false, inject_metrics: false, x_range: None, x_range_pad: Some(x_range_pad),
            groupby: vec![],
            name: None,
        }
    }

    #[test]
    fn log_demanded_robust_facet_pin_equals_column_extent_in_space() {
        let batch = xy_batch();
        let spec = robust_pad_spec(0.1);
        let step_space = x_slot(positive_space());
        let demand = SpaceDemand::single(vec![0], x_slot(positive_space()));
        let slot_space = step_space.of(SpaceSlot::X).expect("x slot demanded");
        let expected = crate::transform::log_space::column_extent_in_space(&batch, "x", SpaceSlot::X, slot_space, "Robust");
        let out = fix_transform_extents_for_facet_in_space(&[TransformSpec::Robust(spec)], &batch, Some(&demand));
        let TransformSpec::Robust(pinned) = &out[0] else { panic!("expected a Robust") };
        let range = pinned.x_range.expect("demanded pin");
        assert!(range[0] > 0.0 && range[1] > 0.0, "a positive-side pin must stay on-side: {range:?}");
        let expected = expected.expect("the fixture's x column has a real extent");
        assert_eq!(range, [expected.0, expected.1], "the pin must equal column_extent_in_space's own on-side result, not a value this test recomputes");
    }

    #[test]
    fn undemanded_robust_facet_pin_equals_column_extent_in_data_space() {
        let batch = xy_batch();
        let spec = robust_pad_spec(0.1);
        let expected = crate::transform::numeric_util::column_extent(&batch, "x").expect("the fixture's x column has a real extent");
        let out = fix_transform_extents_for_facet(&[TransformSpec::Robust(spec)], &batch);
        let TransformSpec::Robust(pinned) = &out[0] else { panic!("expected a Robust") };
        let range = pinned.x_range.expect("plain pin");
        assert_eq!(range, [expected.0, expected.1], "the undemanded pin must equal column_extent's own data-space result");
    }

    /// A `Kde2D` step demanded on its `x` slot, `y` slot, or both, each
    /// independently in its own `space` — the 2-D twin of `field_slot`,
    /// since a `Kde2D`'s two positional slots can each resolve a different
    /// `LogScale` (or none) even though `field_slot`'s single `Field` slot
    /// never needs to.
    fn kde2d_slots(x: Option<LogAxisSpace>, y: Option<LogAxisSpace>) -> StepSpace {
        let mut entries = Vec::new();
        if let Some(space) = x {
            entries.push((SpaceSlot::X, SlotSpace { space, axes: vec![crate::spec::layer::PositionalAxis::X], records_caller_input: true }));
        }
        if let Some(space) = y {
            entries.push((SpaceSlot::Y, SlotSpace { space, axes: vec![crate::spec::layer::PositionalAxis::Y], records_caller_input: true }));
        }
        StepSpace::for_slots(entries)
    }

    /// A positive-side `LogAxisSpace` of the given `base`, reusing
    /// [`positive_space`]'s side rather than restating it.
    fn positive_space_base(base: f64) -> LogAxisSpace {
        LogAxisSpace { base, ..positive_space() }
    }

    /// A two-column batch with distinct, strictly positive x/y ranges this
    /// module's `Kde2D` facet-pin tests assert against: x spans 2..32, y
    /// spans 3..243, both compatible with base-10 and base-2 log mapping.
    fn kde2d_positive_xy_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![2.0, 4.0, 8.0, 16.0, 32.0])),
                Arc::new(Float64Array::from(vec![3.0, 9.0, 27.0, 81.0, 243.0])),
            ],
        )
        .unwrap()
    }

    /// A log-demanded `Kde2D`'s facet pin is the DATA-UNIT extent of its
    /// demanded axes. `global_extent_in_space` maps each demanded axis's
    /// column into that slot's log space to fold the raw min/max there
    /// (the forward map), then must map that pair BACK onto the axis before
    /// pinning it (the backward map). This test's own data-unit assertion
    /// below exercises both halves of that round trip at once — a forward-map
    /// bug would land the pin somewhere other than `(min_x, max_x, min_y,
    /// max_y)` too, the same as a backward-map bug would. Dropping only the
    /// backward half would pin `(log_base(min), log_base(max))` instead —
    /// for this fixture still on-side, so not refused, just a silently far
    /// narrower grid than the real data occupies. Covers x-only, y-only and
    /// both axes demanded, at base 10 and base 2; an undemanded axis is
    /// never mapped at all, so it comes back exactly as `global_extent`
    /// (no space) computes it — the same assertion covers both, since this
    /// fixture's data-unit extent does not depend on which axis was
    /// demanded. The pinned spec's own `apply_with_context` run also
    /// reproduces the unpinned run's grid, extent and density, since the
    /// pin is exactly what that run would have computed for its own
    /// extent.
    #[test]
    fn log_demanded_kde_2d_facet_pin_is_the_data_unit_extent() {
        use crate::transform::kde::BandwidthSpec;
        use crate::transform::kde_2d::tests::{assert_close, list_values};
        use crate::transform::kde_2d::Kde2DSpec;

        let batch = kde2d_positive_xy_batch();
        let (min_x, max_x, min_y, max_y) = (2.0, 32.0, 3.0, 243.0);
        let spec = Kde2DSpec {
            x: "x".into(),
            y: "y".into(),
            bandwidth: BandwidthSpec::Scott,
            n: 8,
            extent: None,
            groupby: vec![],
            name: None,
        };
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * b.abs().max(1.0);

        for base in [10.0, 2.0] {
            for (x_demanded, y_demanded) in [(true, false), (false, true), (true, true)] {
                let x_space = x_demanded.then(|| positive_space_base(base));
                let y_space = y_demanded.then(|| positive_space_base(base));
                let space = kde2d_slots(x_space, y_space);
                let demand = SpaceDemand::single(vec![0], space.clone());
                let out = fix_transform_extents_for_facet_in_space(
                    &[TransformSpec::Kde2D(spec.clone())],
                    &batch,
                    Some(&demand),
                );
                let TransformSpec::Kde2D(pinned) = &out[0] else { panic!("expected a Kde2D") };
                let (px_lo, px_hi, py_lo, py_hi) = pinned.extent.expect("the pin is set");

                assert!(
                    close(px_lo, min_x) && close(px_hi, max_x),
                    "base {base}, x demanded {x_demanded}: x pin ({px_lo}, {px_hi}) != data-unit ({min_x}, {max_x})"
                );
                assert!(
                    close(py_lo, min_y) && close(py_hi, max_y),
                    "base {base}, y demanded {y_demanded}: y pin ({py_lo}, {py_hi}) != data-unit ({min_y}, {max_y})"
                );

                use crate::transform::context::TransformContext;
                let demanded_ctx = TransformContext { log_space: Some(std::rc::Rc::new(space)), ..Default::default() };
                let unpinned = crate::transform::kde_2d::apply_with_context(&spec, &batch, &demanded_ctx)
                    .expect("the unpinned run");
                let with_pin = crate::transform::kde_2d::apply_with_context(pinned, &batch, &demanded_ctx)
                    .expect("the pinned extent must not be refused by the step it was pinned for");
                for name in ["grid_x", "grid_y", "density", "extent"] {
                    assert_close(name, &list_values(&with_pin, name), &list_values(&unpinned, name));
                }
            }
        }
    }

    /// A log-demanded `Kde2D`'s facet pin over a demanded column that is
    /// degenerate (every value tied) is left unpinned: `global_extent` has
    /// no principled range for a column with none, so the spec comes back
    /// exactly as it went in. A demanded column holding a null, by
    /// contrast, IS pinned — the null is simply excluded from the min/max
    /// fold, the same way `global_extent` already excludes it — and the
    /// pin equals the unpinned run's OWN emitted `extent` output (the
    /// oracle the real `apply_with_context` path computes, not a
    /// hand-picked value), within relative `1e-9`. That pinned spec's own
    /// `apply_with_context` run then reproduces the unpinned run's grid,
    /// extent and density too. An off-side demanded value is refused by
    /// the demanded step itself, not by this pin, and is out of this
    /// test's domain.
    #[test]
    fn log_demanded_kde_2d_facet_pin_on_boundary_data_matches_the_unpinned_run() {
        use crate::transform::context::TransformContext;
        use crate::transform::kde::BandwidthSpec;
        use crate::transform::kde_2d::tests::{assert_close, list_values};
        use crate::transform::kde_2d::Kde2DSpec;

        let spec = Kde2DSpec {
            x: "x".into(),
            y: "y".into(),
            bandwidth: BandwidthSpec::Scott,
            n: 8,
            extent: None,
            groupby: vec![],
            name: None,
        };
        let space = kde2d_slots(Some(positive_space_base(10.0)), None);
        let pin = |batch: &RecordBatch| {
            let out = fix_transform_extents_for_facet_in_space(
                &[TransformSpec::Kde2D(spec.clone())],
                batch,
                Some(&SpaceDemand::single(vec![0], space.clone())),
            );
            let TransformSpec::Kde2D(pinned) = out.into_iter().next().unwrap() else { panic!("expected a Kde2D") };
            pinned
        };

        let tied = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("x", DataType::Float64, false),
                Field::new("y", DataType::Float64, false),
            ])),
            vec![
                Arc::new(Float64Array::from(vec![5.0, 5.0, 5.0])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            ],
        )
        .unwrap();
        let tied_pinned = pin(&tied);
        assert!(
            tied_pinned.extent.is_none(),
            "an all-tied demanded column has no range to pin, so the spec stays unpinned: {:?}",
            tied_pinned.extent
        );

        let with_null = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("x", DataType::Float64, true),
                Field::new("y", DataType::Float64, false),
            ])),
            vec![
                Arc::new(Float64Array::from(vec![Some(2.0), None, Some(8.0), Some(16.0)])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
            ],
        )
        .unwrap();
        let null_pinned = pin(&with_null);
        let (px_lo, px_hi, py_lo, py_hi) = null_pinned
            .extent
            .expect("a null is excluded from the fold, not made degenerate, so this demanded column still pins");

        let demanded_ctx = TransformContext { log_space: Some(std::rc::Rc::new(space)), ..Default::default() };
        let unpinned = crate::transform::kde_2d::apply_with_context(&spec, &with_null, &demanded_ctx)
            .expect("the unpinned run");
        let oracle_extent = list_values(&unpinned, "extent");
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * b.abs().max(1.0);
        assert!(
            close(px_lo, oracle_extent[0])
                && close(px_hi, oracle_extent[1])
                && close(py_lo, oracle_extent[2])
                && close(py_hi, oracle_extent[3]),
            "the pin ({px_lo}, {px_hi}, {py_lo}, {py_hi}) must equal the unpinned run's own extent output {oracle_extent:?}"
        );

        let with_pin = crate::transform::kde_2d::apply_with_context(&null_pinned, &with_null, &demanded_ctx)
            .expect("the pinned extent must not be refused by the step it was pinned for");
        for name in ["grid_x", "grid_y", "density", "extent"] {
            assert_close(name, &list_values(&with_pin, name), &list_values(&unpinned, name));
        }
    }

    /// With no space demand, `fix_transform_extents_for_facet_in_space` pins
    /// in data space, equal to `fix_transform_extents_for_facet`.
    #[test]
    fn fix_transform_extents_for_facet_in_space_with_no_demand_matches_the_plain_entry_point() {
        use crate::transform::bin::BinSpec;
        use crate::transform::bin_mode::BinMode;
        let spec = BinSpec {
            written_groupby: Vec::new(),
            field: "v".into(),
            mode: BinMode::Fixed { n: 10 },
            extent: None,
            nice: true,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
            agg_field: None,
            agg: None,
            name: None,
            per_row: false,
            aggregates: None,
        };
        let plain = fix_transform_extents_for_facet(&[TransformSpec::Bin(spec.clone())], &v_batch());
        let in_space = fix_transform_extents_for_facet_in_space(&[TransformSpec::Bin(spec)], &v_batch(), None);
        assert_eq!(plain, in_space);
    }
}
