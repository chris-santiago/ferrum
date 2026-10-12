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
    use crate::transform::core::TransformSpec;

    transforms
        .iter()
        .map(|t| match t {
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
                if spec.extent.is_none() {
                    if let Some(e) = crate::transform::kde::global_extent(spec, batch) {
                        pinned.extent = Some(e);
                        changed = true;
                    }
                }
                if spec.count_scale.is_some() && spec.count_scale_extent.is_none() {
                    if let Some(e) = crate::transform::kde::count_scale_global_extent(spec, batch) {
                        pinned.count_scale_extent = Some(e);
                        changed = true;
                    }
                }
                if changed { TransformSpec::Kde(pinned) } else { t.clone() }
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
            // steps pin against the same whole-frame batch.
            TransformSpec::Branch(branch) => {
                TransformSpec::Branch(crate::transform::branch::BranchSpec {
                    name: branch.name.clone(),
                    input: branch.input.clone(),
                    transforms: fix_transform_extents_for_facet(&branch.transforms, batch),
                })
            }
            _ => t.clone(),
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

    use super::fix_transform_extents_for_facet;

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
    /// (`_ => t.clone()`), which is deliberate — `TransformSpec` has 42
    /// variants and only 9 carry an extent to pin, so an exhaustive
    /// wildcard-free match (the `ScaleSpec::positional_extent` precedent)
    /// would add 33 no-op arms. But a wildcard means a new extent-carrying
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
    ///   `for_each_transform!` table `TransformSpec`'s 42 variants are
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
        });
        let branch = |name: &str, transforms| {
            TransformSpec::Branch(BranchSpec { name: name.into(), input: None, transforms })
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
}
