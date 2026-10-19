//! `prepare` test coverage: the `prepare_render_inputs` pipeline end to end
//! (layer geometry and coordinate flips, facet partitioning and sort, legend
//! and auxiliary-scale inputs, the render-time refusals for hand-built
//! specs, warning de-duplication across the panel pass) plus the unit tests
//! of the private helpers those stages call.
//!
//! It lives in its own file rather than inline in `mod.rs` because the
//! suite is roughly twice the size of the production module it covers;
//! inline, the tests buried the pipeline they exercise. As a child module it
//! keeps `super::*` access to `prepare`'s private helpers, which is why it
//! cannot move out to `tests/`.

use super::*;
use crate::layout::{AuxLegendInput, SymbolKind};
use crate::scale::discrete::{DiscreteLayout, DEFAULT_ALIGN};
use crate::spec::encoding::{Encoding, EncodingSpec};
use crate::spec::mark::Mark;
use arrow::array::{Float64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

/// `#172`(2) cycle-3: `value_axis` travels with the channel swap.
///
/// It is authored in PRE-flip space by a desugar and read in POST-flip
/// space by `DrawCtx::series_curve_axis`, so a flip must move it exactly
/// as it moves `x`/`y` themselves. Asserted through `build_layers` — the
/// production entry, which is also what proves a SINGLE-layer chart
/// (`spec.layers == None`) gets the treatment, since that is the shape
/// `mark_density(orientation="horizontal").coord(CoordFlip())` has.
///
/// The unflipped row is the control: without a flip the declaration must
/// arrive exactly as the producer wrote it.
#[test]
fn a_flip_moves_value_axis_into_post_flip_space_with_the_channels() {
    use crate::spec::chart::ChartSpec;
    use crate::spec::mark_style::MarkKwargsSpec;
    use crate::spec::position::StackValueAxis;

    // The spec `desugar_density(orientation="horizontal")` emits: the
    // density on `x`, the observation on `y`, and the declaration saying so.
    let spec = ChartSpec {
        mark: Mark::Area,
        encoding: Encoding {
            x: Some(EncodingSpec {
                field: "density".into(),
                ..Default::default()
            }),
            y: Some(EncodingSpec {
                field: "value".into(),
                ..Default::default()
            }),
            ..Default::default()
        },
        mark_style: Some(MarkKwargsSpec {
            value_axis: Some(StackValueAxis::X),
            ..Default::default()
        }),
        ..Default::default()
    };

    // `build_layers` reads the flip off each layer's own effective coord
    // (`#172`(1)) rather than taking it as an argument, so the flipped case
    // is now spelled as the spec it always described: the same chart with
    // `coord: Flip`. The property under test is unchanged.
    let flipped_spec = ChartSpec {
        coord: Some(crate::spec::coord::CoordKind::Flip),
        ..spec.clone()
    };

    let unflipped = build_layers(&spec).unwrap();
    assert_eq!(
        unflipped.len(),
        1,
        "a single-layer chart prepares one layer"
    );
    assert_eq!(
        unflipped[0].mark_style.as_ref().unwrap().value_axis,
        Some(StackValueAxis::X),
        "control: with no flip the declaration arrives as the producer wrote it"
    );
    assert_eq!(
        unflipped[0].encoding.x.as_ref().unwrap().field,
        "density",
        "control premise: the channels did not move either"
    );

    let flipped = build_layers(&flipped_spec).unwrap();
    assert_eq!(
        flipped[0].encoding.x.as_ref().unwrap().field,
        "value",
        "premise: the flip moved the observation onto x"
    );
    assert_eq!(
        flipped[0].mark_style.as_ref().unwrap().value_axis,
        Some(StackValueAxis::Y),
        "…so the declaration must move with it: post-flip the value is on Y, an \
         ordinary vertical area. Left unflipped it reads X, and \
         `mark_density(orientation=\"horizontal\").coord(CoordFlip())` closes its fill \
         on the panel's left edge while its unflipped twin closes on the bottom."
    );

    // A layer carrying no mark kwargs at all must survive the flip
    // untouched — the `None` arm, which is every chart in the corpus.
    let mut bare = flipped_spec.clone();
    bare.mark_style = None;
    assert!(
        build_layers(&bare).unwrap()[0].mark_style.is_none(),
        "a layer with no mark kwargs gains none from the flip"
    );
}

#[test]
fn parse_label_overlap_maps_vega_tokens() {
    use crate::layout::LabelOverlap;
    assert_eq!(parse_label_overlap("true"), Some(LabelOverlap::ShowAll));
    assert_eq!(parse_label_overlap("parity"), Some(LabelOverlap::Parity));
    assert_eq!(parse_label_overlap("rotate"), Some(LabelOverlap::Rotate));
    assert_eq!(parse_label_overlap("greedy"), Some(LabelOverlap::Greedy));
    assert_eq!(parse_label_overlap("false"), Some(LabelOverlap::Greedy));
    // Case-insensitive + whitespace-tolerant.
    assert_eq!(parse_label_overlap("  PARITY "), Some(LabelOverlap::Parity));
    // Empty → no override (leave the cascade unmodified).
    assert_eq!(parse_label_overlap(""), None);
    // Unrecognized → bounded to the cascade default (greedy), not an error.
    assert_eq!(parse_label_overlap("nonsense"), Some(LabelOverlap::Greedy));
}

// --- R3 (#89 part C): `UnflippedRenderError` compile-enforcement --------------
//
// `parse_axis_orient`/`parse_title_orient` return `UnflippedRenderError`, not
// `RenderError` — there is no `From<UnflippedRenderError> for RenderError`, so
// a bare `?` on their result cannot compile into a `RenderError`-returning fn
// (manually verified during implementation: a probe fn doing exactly that was
// inserted, confirmed to fail with E0277 "the trait `From<UnflippedRenderError>`
// is not implemented for `RenderError`", then removed — trybuild-style
// automated coverage is not available in this crate: `ferrum-core`'s `[lib]`
// is `crate-type = ["cdylib"]` only, and `UnflippedRenderError` is
// `pub(crate)`, so no external test crate can name it). These two tests pin
// the RUNTIME half of the contract: `.resolve(coord_flipped)` is the sole way
// to reach a `RenderError`, and it applies the correction requested.
#[test]
fn unflipped_render_error_resolve_applies_requested_flag() {
    let err = parse_axis_orient("nonsense", "x").unwrap_err();
    match err.resolve(true) {
        RenderError::InvalidAxisOrient {
            channel,
            coord_flipped,
            ..
        } => {
            assert_eq!(channel, "x");
            assert!(coord_flipped, "resolve(true) must patch coord_flipped in");
        }
        other => panic!("expected InvalidAxisOrient, got {other:?}"),
    }
}

#[test]
fn unflipped_render_error_resolve_false_is_the_exempt_chains_explicit_choice() {
    let err = parse_title_orient("nonsense", "y").unwrap_err();
    match err.resolve(false) {
        RenderError::InvalidAxisOrient {
            channel,
            coord_flipped,
            ..
        } => {
            assert_eq!(channel, "y");
            assert!(
                !coord_flipped,
                "resolve(false) is the exempt chart-level chain's explicit choice"
            );
        }
        other => panic!("expected InvalidAxisOrient, got {other:?}"),
    }
}

// --- SPINE-08: build_axis_input + resolve_axis_title -------------------------

/// The 3-way title idiom: absent → fallback; present-empty → suppress;
/// present-nonempty → use it (whitespace counts as empty).
#[test]
fn resolve_axis_title_three_way() {
    assert_eq!(
        resolve_axis_title(None, Some("field".into())),
        Some("field".into())
    );
    assert_eq!(resolve_axis_title(None, None), None);
    assert_eq!(resolve_axis_title(Some(""), Some("field".into())), None);
    assert_eq!(resolve_axis_title(Some("   "), Some("field".into())), None);
    assert_eq!(
        resolve_axis_title(Some("Custom"), Some("field".into())),
        Some("Custom".into())
    );
    // Explicit non-empty wins even with no fallback.
    assert_eq!(
        resolve_axis_title(Some("Custom"), None),
        Some("Custom".into())
    );
}

/// `Channel` carries the orient default and the reverse policy.
#[test]
fn channel_carries_orient_default_and_reverse_policy() {
    assert_eq!(Channel::X.default_orient(), AxisOrient::Bottom);
    assert_eq!(Channel::Y.default_orient(), AxisOrient::Left);
    assert!(!Channel::X.reverses());
    assert!(Channel::Y.reverses());
    assert_eq!(Channel::X.token(), "x");
    assert_eq!(Channel::Y.token(), "y");
}

/// R3: `user_facing_channel` is identity for every token when the chart
/// isn't flipped — the byte-identity default for the overwhelming majority
/// of charts.
#[test]
fn user_facing_channel_identity_when_not_flipped() {
    for token in [
        "x",
        "y",
        "x2",
        "y2",
        "color",
        "size",
        "x2 and y2",
        "field_name",
    ] {
        assert_eq!(user_facing_channel(token, false), token);
    }
}

/// R3: under `coord_flipped`, the four positional tokens swap in the two
/// pairs `build_layers` actually swaps (x<->y, x2<->y2); the swap is its
/// own inverse.
#[test]
fn user_facing_channel_swaps_positional_pairs_when_flipped() {
    assert_eq!(user_facing_channel("x", true), "y");
    assert_eq!(user_facing_channel("y", true), "x");
    assert_eq!(user_facing_channel("x2", true), "y2");
    assert_eq!(user_facing_channel("y2", true), "x2");
    // Involution: applying twice returns the original token.
    for token in ["x", "y", "x2", "y2"] {
        assert_eq!(
            user_facing_channel(user_facing_channel(token, true), true),
            token
        );
    }
}

/// R3: non-positional tokens (channel names `build_layers` never touches,
/// plus compound/free-text strings) are always identity, flipped or not.
#[test]
fn user_facing_channel_non_positional_tokens_always_identity() {
    for token in [
        "color",
        "size",
        "shape",
        "opacity",
        "x2 and y2",
        "some_field",
    ] {
        assert_eq!(user_facing_channel(token, true), token);
        assert_eq!(user_facing_channel(token, false), token);
    }
}

/// A continuous (Linear) scale over [0, 10]. Tick labels run "0".."10" in
/// domain order; the non-ordinal-y reversal flips them.
fn linear_scale_0_10() -> crate::render::scale_resolve::ScaleKind {
    use crate::render::scale_resolve::ScaleKind;
    use crate::scale::linear::LinearScale;
    ScaleKind::Linear(LinearScale::new_internal(
        vec![0.0, 10.0],
        vec![0.0, 1.0],
        false,
        false,
    ))
}

/// An ordinal scale over three categories (no continuum to reverse).
fn ordinal_scale_abc() -> crate::render::scale_resolve::ScaleKind {
    use crate::render::scale_resolve::ScaleKind;
    use crate::scale::ordinal::OrdinalScale;
    ScaleKind::Ordinal(OrdinalScale::new_internal(
        vec!["a".into(), "b".into(), "c".into()],
        vec![0.0, 1.0],
        DiscreteLayout::UNPADDED,
    ))
}

/// x and y axis inputs built from the SAME continuous scale agree on every
/// field EXCEPT that the non-ordinal y reverses its tick labels and
/// projected major fractions, and picks the Left orient default. This is the
/// load-bearing parity check for the SPINE-08 extraction: the y derivation is
/// NOT mechanically identical to x — it interleaves the reversal.
#[test]
fn build_axis_input_x_vs_y_continuous_reverse_parity() {
    let theme = crate::layout::ThemeInputs::default();
    let x_enc = EncodingSpec {
        field: "v".into(),
        ..Default::default()
    };
    let y_enc = EncodingSpec {
        field: "v".into(),
        ..Default::default()
    };
    let scale = linear_scale_0_10();

    let mut x = build_axis_input(Channel::X, Some(&x_enc), Some(&x_enc), &scale, 10, &theme)
        .expect("x axis input");
    let mut y = build_axis_input(Channel::Y, Some(&y_enc), Some(&y_enc), &scale, 10, &theme)
        .expect("y axis input");
    // `build_axis_input` alone never folds in the theme default
    // (`AxesInput::apply_show_defaults` does that at chart-config-resolve
    // time); do it here too, exactly as production does, so the
    // `show_domain()`/`show_grid()` checks below don't trip their
    // debug-only unresolved-precondition assert.
    x.overrides.domain.show.get_or_insert(theme.axis.axis_line);
    x.overrides.grid.show.get_or_insert(theme.grid.grid);
    y.overrides.domain.show.get_or_insert(theme.axis.axis_line);
    y.overrides.grid.show.get_or_insert(theme.grid.grid);

    // Orient defaults differ by channel.
    assert_eq!(x.orient, AxisOrient::Bottom);
    assert_eq!(y.orient, AxisOrient::Left);
    // Title falls through to the field name on both.
    assert_eq!(x.title.get().as_deref(), Some("v"));
    assert_eq!(y.title.get().as_deref(), Some("v"));
    // Show toggles default true on both.
    assert!(x.show_labels() && x.show_ticks() && x.show_domain() && x.show_grid());
    assert!(y.show_labels() && y.show_ticks() && y.show_domain() && y.show_grid());

    // The y tick labels are the x tick labels reversed (non-ordinal-y rule).
    assert!(
        !x.tick_labels.is_empty(),
        "continuous axis must have labels"
    );
    let mut x_reversed = x.tick_labels.clone();
    x_reversed.reverse();
    assert_eq!(y.tick_labels, x_reversed);

    // The projected major fractions are likewise reversed in lockstep.
    let x_major = x
        .tick_projection
        .as_ref()
        .expect("x projection")
        .major
        .clone();
    let y_major = y
        .tick_projection
        .as_ref()
        .expect("y projection")
        .major
        .clone();
    let mut x_major_reversed = x_major.clone();
    x_major_reversed.reverse();
    assert_eq!(y_major, x_major_reversed);
}

/// On an ORDINAL scale the non-ordinal-y reversal does NOT apply: y keeps the
/// same order as x (top-down for heatmaps / confusion matrices). The only
/// remaining difference is the orient default.
#[test]
fn build_axis_input_y_ordinal_does_not_reverse() {
    let theme = crate::layout::ThemeInputs::default();
    let enc = EncodingSpec {
        field: "cat".into(),
        ..Default::default()
    };
    let scale = ordinal_scale_abc();

    let x = build_axis_input(Channel::X, Some(&enc), Some(&enc), &scale, 10, &theme)
        .expect("x axis input");
    let y = build_axis_input(Channel::Y, Some(&enc), Some(&enc), &scale, 10, &theme)
        .expect("y axis input");

    assert_eq!(x.tick_labels, y.tick_labels, "ordinal y must not reverse");
    assert_eq!(x.orient, AxisOrient::Bottom);
    assert_eq!(y.orient, AxisOrient::Left);
}

/// Spec-level title wins over the layer-0 title, which wins over the field
/// name; an explicit empty spec-level title suppresses entirely.
#[test]
fn build_axis_input_title_precedence() {
    let theme = crate::layout::ThemeInputs::default();
    let scale = linear_scale_0_10();

    // spec-level title wins over layer-0 title.
    let rendering = EncodingSpec {
        field: "v".into(),
        title: Some("layer".into()),
        ..Default::default()
    };
    let spec_enc = EncodingSpec {
        field: "v".into(),
        title: Some("spec".into()),
        ..Default::default()
    };
    let a = build_axis_input(
        Channel::X,
        Some(&rendering),
        Some(&spec_enc),
        &scale,
        10,
        &theme,
    )
    .expect("axis input");
    assert_eq!(a.title.get().as_deref(), Some("spec"));

    // explicit empty spec-level title suppresses (no fallback to field/layer).
    let spec_suppress = EncodingSpec {
        field: "v".into(),
        title: Some("".into()),
        ..Default::default()
    };
    let b = build_axis_input(
        Channel::X,
        Some(&rendering),
        Some(&spec_suppress),
        &scale,
        10,
        &theme,
    )
    .expect("axis input");
    assert_eq!(*b.title.get(), None);

    // absent spec-level title falls through to the layer-0 title.
    let spec_absent = EncodingSpec {
        field: "v".into(),
        ..Default::default()
    };
    let c = build_axis_input(
        Channel::X,
        Some(&rendering),
        Some(&spec_absent),
        &scale,
        10,
        &theme,
    )
    .expect("axis input");
    assert_eq!(c.title.get().as_deref(), Some("layer"));
}

// --- archaeology #7: generalized facet extent pin ---------------------------

/// A two-column batch: a Float64 value field `v` plus a Utf8 hue field `g`.
/// Used to drive `fix_transform_extents_for_facet` over the full pre-facet
/// dataset (the global extent must ignore `g`).
///
/// Values are [0.3, 0.5, 1.2, 4.8, 9.7] (raw range 0.3..9.7), chosen so
/// that nicing with bin_count=10 actually changes the extent: nice_step
/// produces step=1.0, which rounds 0.3 down to 0.0 and 9.7 up to 10.0.
fn extent_pin_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("v", DataType::Float64, false),
        Field::new("g", DataType::Utf8, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![0.3, 0.5, 1.2, 4.8, 9.7])),
            Arc::new(StringArray::from(vec!["a", "a", "b", "b", "b"])),
        ],
    )
    .unwrap()
}

/// A faceted `Bin` transform with `extent=None` gets its extent pinned to the
/// niced global range over the full pre-facet batch.
#[test]
fn fix_extents_pins_bin_to_niced_global_extent() {
    use crate::transform::bin::BinSpec;
    use crate::transform::bin_mode::BinMode;
    use crate::transform::core::TransformSpec;
    let batch = extent_pin_batch();
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
        per_row: false,
        aggregates: None,
    };
    let out = fix_transform_extents_for_facet(&[TransformSpec::Bin(spec.clone())], &batch);
    let TransformSpec::Bin(pinned) = &out[0] else {
        panic!("expected a Bin transform back");
    };
    // Concrete expected value: raw range is (0.3, 9.7); nice_step(0.3, 9.7, 10)
    // → step0=0.94, log10(0.94)≈-0.027 → floor=-1 → pow10=0.1, frac=9.4 ≥ 7.5
    // → nice_frac=10.0 → step=1.0; niced lo=floor(0.3/1.0)*1.0=0.0,
    // niced hi=ceil(9.7/1.0)*1.0=10.0.  The niced range differs from the raw
    // range, so this assertion actively exercises the nicing path.
    assert_eq!(
        pinned.extent,
        Some((0.0, 10.0)),
        "Bin extent must be the niced (0.0, 10.0), not the raw (0.3, 9.7)"
    );
    // Also confirm orchestration: the pinned value matches global_extent's output.
    assert_eq!(
        pinned.extent,
        crate::transform::bin::global_extent(&spec, &batch),
        "Bin extent must be pinned to the niced global extent"
    );
}

/// A faceted `Violin` transform with `extent=None` gets its extent pinned to
/// the global range over the full pre-facet batch.
#[test]
fn fix_extents_pins_violin_to_global_extent() {
    use crate::transform::core::TransformSpec;
    use crate::transform::kde::BandwidthSpec;
    use crate::transform::violin::ViolinSpec;
    let batch = extent_pin_batch();
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
    let out = fix_transform_extents_for_facet(&[TransformSpec::Violin(spec.clone())], &batch);
    let TransformSpec::Violin(pinned) = &out[0] else {
        panic!("expected a Violin transform back");
    };
    // Violin pins to the RAW global (min, max) over v = (0.3, 9.7). Unlike
    // Bin, KDE/Violin do not nice their extent (no bin edges to align), so
    // the pinned value is the unrounded global range, not (0.0, 10.0).
    assert_eq!(
        pinned.extent,
        Some((0.3, 9.7)),
        "Violin extent must be pinned to the raw global (min, max) over v"
    );
}

/// The previously-excluded multi-group case: a faceted `Kde` transform WITH a
/// `groupby` (hue) and `extent=None` now gets pinned to the global extent over
/// the full dataset, so panels and hue groups share one value axis (spec §8).
#[test]
fn fix_extents_pins_kde_with_groupby_multi_group() {
    use crate::transform::core::TransformSpec;
    use crate::transform::kde::KdeSpec;
    let batch = extent_pin_batch();
    let spec = KdeSpec {
        field: "v".into(),
        bandwidth: crate::transform::kde::BandwidthSpec::Scott,
        bw_adjust: 1.0,
        n: 64,
        extent: None,
        cumulative: false,
        shared_extent: false,
        kernel: "gaussian".into(),
        groupby: vec!["g".into()],
        count_scale: None,
        count_scale_extent: None,
        name: None,
    };
    let out = fix_transform_extents_for_facet(&[TransformSpec::Kde(spec.clone())], &batch);
    let TransformSpec::Kde(pinned) = &out[0] else {
        panic!("expected a Kde transform back");
    };
    // Raw global extent over v = (0.3, 9.7) regardless of the `g` groupby —
    // the multi-group fix the old groupby.is_some() early-return blocked. KDE
    // does not nice (only Bin does), so the pin is the raw global min/max.
    assert_eq!(
        pinned.extent,
        Some((0.3, 9.7)),
        "KDE with a groupby must still be pinned to the full-dataset extent"
    );
}

/// A user-provided explicit `extent` must never be overridden by the pin.
#[test]
fn fix_extents_respects_user_extent() {
    use crate::transform::bin::BinSpec;
    use crate::transform::bin_mode::BinMode;
    use crate::transform::core::TransformSpec;
    use crate::transform::kde::KdeSpec;
    let batch = extent_pin_batch();
    let user = Some((-5.0, 100.0));
    let bin = BinSpec {
        field: "v".into(),
        mode: BinMode::Fixed { n: 10 },
        extent: user,
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
    let kde = KdeSpec {
        field: "v".into(),
        bandwidth: crate::transform::kde::BandwidthSpec::Scott,
        bw_adjust: 1.0,
        n: 64,
        extent: user,
        cumulative: false,
        shared_extent: false,
        kernel: "gaussian".into(),
        groupby: vec![],
        count_scale: None,
        count_scale_extent: None,
        name: None,
    };
    let out = fix_transform_extents_for_facet(
        &[TransformSpec::Bin(bin), TransformSpec::Kde(kde)],
        &batch,
    );
    let TransformSpec::Bin(pinned_bin) = &out[0] else {
        panic!("expected a Bin transform back");
    };
    let TransformSpec::Kde(pinned_kde) = &out[1] else {
        panic!("expected a Kde transform back");
    };
    assert_eq!(pinned_bin.extent, user, "user Bin extent must be preserved");
    assert_eq!(pinned_kde.extent, user, "user KDE extent must be preserved");
}

/// Pins the niced-vs-raw contract between `bin::global_extent`, `kde::global_extent`,
/// and `violin::global_extent`.
///
/// For the shared fixture data [0.3, 0.5, 1.2, 4.8, 9.7] (raw range 0.3..9.7):
///
/// - **Bin** nices its extent to align bin edges: step=1.0 → (0.0, 10.0).
/// - **KDE** and **Violin** pin to the raw (min, max) because they have no bin
///   edges to align: (0.3, 9.7).
///
/// This test guards the exact defect class that regressed in archaeology bug #7:
/// the `fix_extents_pins_violin_to_global_extent` and
/// `fix_extents_pins_kde_with_groupby_multi_group` tests had their assertions
/// set to the niced value (0.0, 10.0) instead of the raw value (0.3, 9.7).
/// Any future change that makes KDE/Violin nice, or makes Bin not nice, will
/// trip this test and force an explicit re-evaluation of the contract.
#[test]
fn global_extent_nices_for_bin_but_raw_for_kde_and_violin() {
    use crate::transform::bin::BinSpec;
    use crate::transform::bin_mode::BinMode;
    use crate::transform::kde::{BandwidthSpec, KdeSpec};
    use crate::transform::violin::ViolinSpec;

    let batch = extent_pin_batch();

    let bin_spec = BinSpec {
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
    let kde_spec = KdeSpec {
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
    let violin_spec = ViolinSpec {
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

    // Bin nices: raw (0.3, 9.7) with bin_count=10 → step=1.0 → (0.0, 10.0).
    assert_eq!(
        crate::transform::bin::global_extent(&bin_spec, &batch),
        Some((0.0, 10.0)),
        "Bin must return the NICED global extent, not the raw (0.3, 9.7)"
    );
    // KDE does not nice: returns the raw (min, max).
    assert_eq!(
        crate::transform::kde::global_extent(&kde_spec, &batch),
        Some((0.3, 9.7)),
        "KDE must return the RAW global extent (0.3, 9.7), not the niced (0.0, 10.0)"
    );
    // Violin does not nice: returns the raw (min, max).
    assert_eq!(
        crate::transform::violin::global_extent(&violin_spec, &batch),
        Some((0.3, 9.7)),
        "Violin must return the RAW global extent (0.3, 9.7), not the niced (0.0, 10.0)"
    );
}

// --- archaeology R5: 2-D facet extent pin (Kde2D / Bin2D) -------------------

/// A three-column batch: Float64 `x`, Float64 `y`, and a Utf8 facet field `p`.
/// Panel "A" lives at x in 0..1, y in 0..1; panel "B" lives at x in 10..11,
/// y in 100..101. The two panels' x AND y ranges are fully DISJOINT, so a
/// per-panel extent (the pre-R5 bug) would give panel A x∈[0,1]/y∈[0,1] and
/// panel B x∈[10,11]/y∈[100,101). The pin computes the global extent over the
/// whole batch → x∈[0,11], y∈[0,101] for BOTH panels.
fn extent_pin_batch_2d() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
        Field::new("p", DataType::Utf8, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![
                0.0, 0.5, 1.0, // panel A
                10.0, 10.5, 11.0, // panel B
            ])),
            Arc::new(Float64Array::from(vec![
                0.0, 0.5, 1.0, // panel A
                100.0, 100.5, 101.0, // panel B
            ])),
            Arc::new(StringArray::from(vec!["A", "A", "A", "B", "B", "B"])),
        ],
    )
    .unwrap()
}

/// A faceted `Kde2D` with no extent gets pinned to the RAW global 2-D range
/// over the full pre-facet batch, so every panel shares the same x AND y
/// extents. Discriminating: an unpinned (per-panel) extent would not span the
/// global x∈[0,11]/y∈[0,101] range.
#[test]
fn fix_extents_pins_kde2d_to_global_2d_extent() {
    use crate::transform::core::TransformSpec;
    use crate::transform::kde::BandwidthSpec;
    use crate::transform::kde_2d::Kde2DSpec;
    let batch = extent_pin_batch_2d();
    let spec = Kde2DSpec {
        x: "x".into(),
        y: "y".into(),
        bandwidth: BandwidthSpec::Scott,
        n: 16,
        extent: None,
        groupby: vec![],
        name: None,
    };
    let out = fix_transform_extents_for_facet(&[TransformSpec::Kde2D(spec.clone())], &batch);
    let TransformSpec::Kde2D(pinned) = &out[0] else {
        panic!("expected a Kde2D transform back");
    };
    // Raw global extent over the whole batch: x∈[0,11], y∈[0,101]. Kde2D does
    // not nice. The disjoint per-panel ranges make this fail if unpinned.
    assert_eq!(
        pinned.extent,
        Some((0.0, 11.0, 0.0, 101.0)),
        "Kde2D extent must be pinned to the raw global 2-D range across panels"
    );
    // Orchestration sanity: the pinned value matches global_extent's output.
    assert_eq!(
        pinned.extent,
        crate::transform::kde_2d::global_extent(&spec, &batch),
    );
}

/// A faceted `Bin2D` with no extent gets pinned per-axis to the RAW global
/// range over the full pre-facet batch, so every panel shares the same x AND
/// y bin edges. Bin2D never nices, so the pin is the raw range.
#[test]
fn fix_extents_pins_bin2d_to_global_2d_extent() {
    use crate::transform::bin_2d::Bin2DSpec;
    use crate::transform::bin_mode::BinMode;
    use crate::transform::core::TransformSpec;
    let batch = extent_pin_batch_2d();
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
    let out = fix_transform_extents_for_facet(&[TransformSpec::Bin2D(spec.clone())], &batch);
    let TransformSpec::Bin2D(pinned) = &out[0] else {
        panic!("expected a Bin2D transform back");
    };
    assert_eq!(
        pinned.extent_x,
        Some((0.0, 11.0)),
        "Bin2D extent_x must be the raw global x-range across panels"
    );
    assert_eq!(
        pinned.extent_y,
        Some((0.0, 101.0)),
        "Bin2D extent_y must be the raw global y-range across panels"
    );
}

/// A user-provided explicit 2-D extent must never be overridden by the pin.
/// Covers both Kde2D (4-tuple) and Bin2D (per-axis extent_x/extent_y).
#[test]
fn fix_extents_respects_user_2d_extent() {
    use crate::transform::bin_2d::Bin2DSpec;
    use crate::transform::bin_mode::BinMode;
    use crate::transform::core::TransformSpec;
    use crate::transform::kde::BandwidthSpec;
    use crate::transform::kde_2d::Kde2DSpec;
    let batch = extent_pin_batch_2d();

    let user_kde = (-5.0, 50.0, -5.0, 500.0);
    let kde = Kde2DSpec {
        x: "x".into(),
        y: "y".into(),
        bandwidth: BandwidthSpec::Scott,
        n: 16,
        extent: Some(user_kde),
        groupby: vec![],
        name: None,
    };
    // Bin2D with BOTH axes user-set must be left fully untouched.
    let user_bin_x = (-1.0, 20.0);
    let user_bin_y = (-2.0, 200.0);
    let bin = Bin2DSpec {
        x: "x".into(),
        y: "y".into(),
        bins_x: BinMode::Fixed { n: 10 },
        bins_y: BinMode::Fixed { n: 10 },
        extent_x: Some(user_bin_x),
        extent_y: Some(user_bin_y),
        cumulative: false,
        name: None,
    };
    let out = fix_transform_extents_for_facet(
        &[TransformSpec::Kde2D(kde), TransformSpec::Bin2D(bin)],
        &batch,
    );
    let TransformSpec::Kde2D(pinned_kde) = &out[0] else {
        panic!("expected a Kde2D transform back");
    };
    let TransformSpec::Bin2D(pinned_bin) = &out[1] else {
        panic!("expected a Bin2D transform back");
    };
    assert_eq!(
        pinned_kde.extent,
        Some(user_kde),
        "user Kde2D extent must be preserved"
    );
    assert_eq!(
        pinned_bin.extent_x,
        Some(user_bin_x),
        "user Bin2D extent_x must be preserved"
    );
    assert_eq!(
        pinned_bin.extent_y,
        Some(user_bin_y),
        "user Bin2D extent_y must be preserved"
    );
}

/// A faceted `Bin2D` with `extent_x` user-set but `extent_y = None` goes through
/// the dispatch arm's `.or` composition: the user-provided `extent_x` is kept
/// unchanged while `extent_y` is pinned to the global y-range over the full
/// pre-facet batch. This guards against a future x/y transposition in the arm
/// (e.g. accidentally writing `extent_x: spec.extent_y.or(...)`) by asserting
/// the final pinned values at both axes simultaneously.
#[test]
fn fix_extents_bin2d_partial_user_extent_keeps_x_pins_y() {
    use crate::transform::bin_2d::Bin2DSpec;
    use crate::transform::bin_mode::BinMode;
    use crate::transform::core::TransformSpec;
    let batch = extent_pin_batch_2d();

    // User has explicitly set extent_x but left extent_y as None.
    let user_x = (2.0, 15.0);
    let spec = Bin2DSpec {
        x: "x".into(),
        y: "y".into(),
        bins_x: BinMode::Fixed { n: 10 },
        bins_y: BinMode::Fixed { n: 10 },
        extent_x: Some(user_x),
        extent_y: None,
        cumulative: false,
        name: None,
    };
    let out = fix_transform_extents_for_facet(&[TransformSpec::Bin2D(spec.clone())], &batch);
    let TransformSpec::Bin2D(pinned) = &out[0] else {
        panic!("expected a Bin2D transform back");
    };
    // The user-provided extent_x must survive unchanged.
    assert_eq!(
        pinned.extent_x,
        Some(user_x),
        "user extent_x must be preserved when extent_y is None"
    );
    // extent_y was None; the pin must fill it with the global y-range [0, 101].
    // Raw y values in the fixture: 0.0, 0.5, 1.0, 100.0, 100.5, 101.0.
    assert_eq!(
        pinned.extent_y,
        Some((0.0, 101.0)),
        "extent_y must be pinned to the global y-range when user left it None"
    );
}

// --- archaeology #7 round-7 T1: Hex / Raster / DataBin dispatch tests -------

/// A faceted `Hex` with no extent gets both axes pinned to the RAW global
/// range over the full pre-facet batch. The two-panel fixture has disjoint
/// x ∈ [0,1] vs [10,11] and y ∈ [0,1] vs [100,101]; the global raw range is
/// x∈[0,11], y∈[0,101]. An un-pinned hex lattice would anchor differently per
/// panel, making the bin edges incomparable across panels.
#[test]
fn fix_extents_pins_hex_to_global_2d_extent() {
    use crate::transform::core::TransformSpec;
    use crate::transform::hex::HexSpec;
    let batch = extent_pin_batch_2d();
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
    let out = fix_transform_extents_for_facet(&[TransformSpec::Hex(spec.clone())], &batch);
    let TransformSpec::Hex(pinned) = &out[0] else {
        panic!("expected a Hex transform back");
    };
    assert_eq!(
        pinned.extent_x,
        Some((0.0, 11.0)),
        "Hex extent_x must be pinned to the raw global x-range across panels"
    );
    assert_eq!(
        pinned.extent_y,
        Some((0.0, 101.0)),
        "Hex extent_y must be pinned to the raw global y-range across panels"
    );
}

/// A faceted `Hex` with `extent_x` user-set but `extent_y = None` keeps the
/// user value on x and pins y to the global raw range. Guards against an
/// accidental x/y transposition in the dispatch arm.
#[test]
fn fix_extents_hex_partial_user_extent_keeps_x_pins_y() {
    use crate::transform::core::TransformSpec;
    use crate::transform::hex::HexSpec;
    let batch = extent_pin_batch_2d();

    let user_x = (2.0, 15.0);
    let spec = HexSpec {
        x: "x".into(),
        y: "y".into(),
        bin_size: None,
        aggregate: Default::default(),
        field: None,
        name: None,
        extent_x: Some(user_x),
        extent_y: None,
    };
    let out = fix_transform_extents_for_facet(&[TransformSpec::Hex(spec.clone())], &batch);
    let TransformSpec::Hex(pinned) = &out[0] else {
        panic!("expected a Hex transform back");
    };
    assert_eq!(
        pinned.extent_x,
        Some(user_x),
        "user extent_x must be preserved when extent_y is None"
    );
    assert_eq!(
        pinned.extent_y,
        Some((0.0, 101.0)),
        "extent_y must be pinned to the global y-range when user left it None"
    );
}

/// A faceted `Raster` with no extent gets both axes pinned to the RAW global
/// range over the full pre-facet batch: x∈[0,11], y∈[0,101].
#[test]
fn fix_extents_pins_raster_to_global_2d_extent() {
    use crate::transform::core::TransformSpec;
    use crate::transform::raster::{RasterSpec, ResolutionSpec};
    let batch = extent_pin_batch_2d();
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
    let out = fix_transform_extents_for_facet(&[TransformSpec::Raster(spec.clone())], &batch);
    let TransformSpec::Raster(pinned) = &out[0] else {
        panic!("expected a Raster transform back");
    };
    assert_eq!(
        pinned.extent_x,
        Some((0.0, 11.0)),
        "Raster extent_x must be pinned to the raw global x-range across panels"
    );
    assert_eq!(
        pinned.extent_y,
        Some((0.0, 101.0)),
        "Raster extent_y must be pinned to the raw global y-range across panels"
    );
}

/// A faceted `Raster` with `extent_x` user-set but `extent_y = None` keeps
/// the user value on x and pins y to the global raw range. Guards against an
/// accidental x/y transposition in the dispatch arm.
#[test]
fn fix_extents_raster_partial_user_extent_keeps_x_pins_y() {
    use crate::transform::core::TransformSpec;
    use crate::transform::raster::{RasterSpec, ResolutionSpec};
    let batch = extent_pin_batch_2d();

    let user_x = (2.0, 15.0);
    let spec = RasterSpec {
        x: "x".into(),
        y: "y".into(),
        aggregate: Default::default(),
        field: None,
        resolution: ResolutionSpec::Fixed(4),
        min_count: None,
        log_scale: false,
        name: None,
        extent_x: Some(user_x),
        extent_y: None,
    };
    let out = fix_transform_extents_for_facet(&[TransformSpec::Raster(spec.clone())], &batch);
    let TransformSpec::Raster(pinned) = &out[0] else {
        panic!("expected a Raster transform back");
    };
    assert_eq!(
        pinned.extent_x,
        Some(user_x),
        "user extent_x must be preserved when extent_y is None"
    );
    assert_eq!(
        pinned.extent_y,
        Some((0.0, 101.0)),
        "extent_y must be pinned to the global y-range when user left it None"
    );
}

/// A faceted `DataBin` with `extent = None` gets pinned to the RAW global
/// `(min, max)` over the full pre-facet batch. The fixture has `v` values
/// [0.3, 0.5, 1.2, 4.8, 9.7] (raw range 0.3..9.7). Unlike `Bin`, DataBin
/// does not nice the pinned extent — the raw range drives the bin boundary
/// computation directly.
#[test]
fn fix_extents_pins_data_bin_to_global_raw_extent() {
    use crate::transform::core::TransformSpec;
    use crate::transform::data_bin::DataBinSpec;
    let batch = extent_pin_batch();
    let spec = DataBinSpec {
        field: "v".into(),
        as_: None,
        maxbins: Some(10),
        step: None,
        nice: false,
        name: None,
        extent: None,
    };
    let out = fix_transform_extents_for_facet(&[TransformSpec::DataBin(spec.clone())], &batch);
    let TransformSpec::DataBin(pinned) = &out[0] else {
        panic!("expected a DataBin transform back");
    };
    // Raw range over [0.3, 0.5, 1.2, 4.8, 9.7] = (0.3, 9.7). DataBin never
    // nices the pin, so the pinned value is the exact raw (min, max).
    assert_eq!(
        pinned.extent,
        Some((0.3, 9.7)),
        "DataBin extent must be pinned to the raw global (min, max) across panels"
    );
}

/// A faceted `DataBin` with a user-provided `extent` must never have it
/// overridden. The dispatch arm guards on `spec.extent.is_none()`, so this
/// confirms the guard fires correctly and the user value is left untouched.
#[test]
fn fix_extents_data_bin_user_extent_is_preserved() {
    use crate::transform::core::TransformSpec;
    use crate::transform::data_bin::DataBinSpec;
    let batch = extent_pin_batch();

    let user_extent = (0.0, 20.0);
    let spec = DataBinSpec {
        field: "v".into(),
        as_: None,
        maxbins: Some(10),
        step: None,
        nice: false,
        name: None,
        extent: Some(user_extent),
    };
    let out = fix_transform_extents_for_facet(&[TransformSpec::DataBin(spec.clone())], &batch);
    let TransformSpec::DataBin(pinned) = &out[0] else {
        panic!("expected a DataBin transform back");
    };
    assert_eq!(
        pinned.extent,
        Some(user_extent),
        "user DataBin extent must never be clobbered by the pin"
    );
}

// --- archaeology #7 round-3 T1: DensityData facet extent pin ----------------

/// A batch with two panels whose value ranges are fully DISJOINT.
/// Panel "A" has x in [1, 3], panel "B" has x in [10, 30].
/// Used to drive `fix_transform_extents_for_facet` for DensityData: an
/// unpinned per-panel KDE would give each panel a different x extent.
fn density_disjoint_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, false),
        Field::new("panel", DataType::Utf8, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![
                1.0, 2.0, 3.0, // panel A
                10.0, 20.0, 30.0, // panel B
            ])),
            Arc::new(StringArray::from(vec!["A", "A", "A", "B", "B", "B"])),
        ],
    )
    .unwrap()
}

fn base_density_spec() -> crate::transform::density_data::DensityDataSpec {
    crate::transform::density_data::DensityDataSpec {
        field: "x".into(),
        bandwidth: None,
        groupby: None,
        extent: None,
        steps: None,
        cumulative: false,
        kernel: crate::transform::kde::default_kernel(),
        as_: ("value".into(), "density".into()),
        name: None,
    }
}

/// A faceted `DensityData` with `extent=None` must be pinned to the RAW
/// global (min, max) over the full pre-facet batch so every panel shares
/// the same value axis. This is discriminating: without the T1 fix the
/// DensityData arm did not exist in `fix_transform_extents_for_facet` and
/// the spec fell through the `_ => t.clone()` arm with extent still `None`.
#[test]
fn fix_extents_pins_density_data_to_global_extent() {
    use crate::transform::core::TransformSpec;
    let batch = density_disjoint_batch();
    let spec = base_density_spec(); // extent: None
    let out =
        fix_transform_extents_for_facet(&[TransformSpec::DensityData(spec.clone())], &batch);
    let TransformSpec::DensityData(pinned) = &out[0] else {
        panic!("expected a DensityData transform back");
    };
    // Raw global extent over full batch: x ∈ [1.0, 30.0].
    // DensityData does not nice (mirrors KDE), so the pin is the raw (min, max).
    assert_eq!(
        pinned.extent,
        Some((1.0, 30.0)),
        "DensityData extent must be pinned to the raw global (1.0, 30.0)"
    );
    // Orchestration sanity: matches global_extent's output directly.
    assert_eq!(
        pinned.extent,
        crate::transform::density_data::global_extent(&spec, &batch),
        "pinned extent must equal global_extent's own output"
    );
}

/// A user-provided explicit `DensityData` `extent` must never be overridden.
#[test]
fn fix_extents_density_data_respects_user_extent() {
    use crate::transform::core::TransformSpec;
    let batch = density_disjoint_batch();
    let user = Some((-5.0, 50.0));
    let spec = crate::transform::density_data::DensityDataSpec {
        extent: user,
        ..base_density_spec()
    };
    let out = fix_transform_extents_for_facet(&[TransformSpec::DensityData(spec)], &batch);
    let TransformSpec::DensityData(pinned) = &out[0] else {
        panic!("expected a DensityData transform back");
    };
    assert_eq!(
        pinned.extent, user,
        "user DensityData extent must be preserved"
    );
}

/// `DensityData::global_extent` returns raw (min, max) — no nicing, mirroring KDE.
/// Guards the contract that DensityData uses raw extent (not niced like Bin).
#[test]
fn density_data_global_extent_is_raw_not_niced() {
    use crate::transform::density_data;
    let batch = density_disjoint_batch();
    let spec = base_density_spec();
    assert_eq!(
        density_data::global_extent(&spec, &batch),
        Some((1.0, 30.0)),
        "DensityData global_extent must return the raw (min, max), not a niced value"
    );
}

fn batch3() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
        Field::new("species", DataType::Utf8, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
            Arc::new(StringArray::from(vec!["a", "b", "a"])),
        ],
    )
    .unwrap()
}

fn spec_color_facet() -> ChartSpec {
    ChartSpec {
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec {
                field: "x".into(),
                type_: None,
                ..Default::default()
            }),
            y: Some(EncodingSpec {
                field: "y".into(),
                type_: None,
                ..Default::default()
            }),
            color: Some(EncodingSpec {
                field: "species".into(),
                type_: None,
                ..Default::default()
            }),
            ..Default::default()
        },
        facet: Some(crate::layout::FacetSpec {
            field: "species".into(),
            row: None,
            title: None,
            row_title: None,
            sort: None,
            row_sort: None,
            repeat: Vec::new(),
            mode: crate::layout::FacetMode::Wrap { ncols: 2 },
            spacing: None,
            resolve: crate::layout::facet::FacetResolve::default(),
        }),
        ..Default::default()
    }
}

#[test]
fn prepare_returns_axes_and_groups_and_legend() {
    let spec = spec_color_facet();
    let batch = batch3();
    let prep = prepare_render_inputs(
        &spec,
        &batch,
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport {
            width: 600.0,
            height: 400.0,
        },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .unwrap();
    assert_eq!(prep.axes.x.title.get().as_deref(), Some("x"));
    assert_eq!(prep.axes.y.title.get().as_deref(), Some("y"));
    assert!(!prep.axes.x.tick_labels.is_empty());
    assert_eq!(prep.facet_groups.len(), 2);
    assert_eq!(prep.facet_groups[0].n_rows, 2);
    assert_eq!(prep.facet_groups[1].n_rows, 1);
    assert_eq!(prep.legend_entries.len(), 2);
    assert_eq!(prep.legend_entries[0].label, "a");
}

// ── `#176`(11) + N3: `Facet(sort=)` orders the facet partition ───────────

fn facet_group_values(prep: &PreparedInputs) -> Vec<String> {
    prep.facet_groups.iter().map(|g| g.key.value.clone()).collect()
}

/// Control cell of the door × sort-form matrix: no `sort` set, panel
/// order is data first-appearance order, byte-identical to the unsorted case.
#[test]
fn facet_sort_absent_is_first_appearance_order() {
    let spec = spec_color_facet();
    let batch = batch3();
    let prep = prepare_render_inputs(
        &spec,
        &batch,
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport { width: 600.0, height: 400.0 },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .unwrap();
    assert_eq!(facet_group_values(&prep), vec!["a", "b"]);
    assert!(
        prep.warnings.iter().all(|w| !matches!(w, RenderWarning::SortSpecIgnored { .. })),
        "no sort spec means no sort warning: {:?}",
        prep.warnings
    );
}

/// `Facet(sort="descending")`: alphabetical descending reverses the
/// batch's `["a", "b"]` first-appearance order to `["b", "a"]`.
#[test]
fn facet_sort_descending_orders_wrap_panels() {
    let mut spec = spec_color_facet();
    spec.facet.as_mut().unwrap().sort = Some(serde_json::json!("descending"));
    let batch = batch3();
    let prep = prepare_render_inputs(
        &spec,
        &batch,
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport { width: 600.0, height: 400.0 },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .unwrap();
    assert_eq!(facet_group_values(&prep), vec!["b", "a"]);
}

/// `Facet(sort=[...])`: an explicit list wins for the values it names;
/// a domain value the list omits (`"a"`) is appended after, so no facet
/// panel is dropped for want of a listed value.
#[test]
fn facet_sort_explicit_list_orders_wrap_panels() {
    let mut spec = spec_color_facet();
    spec.facet.as_mut().unwrap().sort = Some(serde_json::json!(["b"]));
    let batch = batch3();
    let prep = prepare_render_inputs(
        &spec,
        &batch,
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport { width: 600.0, height: 400.0 },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .unwrap();
    assert_eq!(facet_group_values(&prep), vec!["b", "a"]);
}

/// `Facet(sort={"field": "x", "op": "sum"})`: orders by the aggregate,
/// the same ordinal-domain sort resolution `X(sort=)` uses. `batch3`'s
/// species "a" sums x=1.0+3.0=4.0, "b" sums x=2.0; ascending (the
/// object form's default order) puts "b" first.
#[test]
fn facet_sort_field_object_orders_wrap_panels_by_aggregate() {
    let mut spec = spec_color_facet();
    spec.facet.as_mut().unwrap().sort = Some(serde_json::json!({"field": "x", "op": "sum"}));
    let batch = batch3();
    let prep = prepare_render_inputs(
        &spec,
        &batch,
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport { width: 600.0, height: 400.0 },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .unwrap();
    assert_eq!(facet_group_values(&prep), vec!["b", "a"]);
}

/// A two-way grid: `FacetCol`'s `sort` (the aggregate object form) and
/// `FacetRow`'s `sort` (`"descending"`) each order their own dimension
/// independently, and the row-major panel order reflects both.
#[test]
fn facet_sort_orders_row_and_col_dimensions_independently_in_grid_mode() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
        Field::new("species", DataType::Utf8, false),
        Field::new("col_cat", DataType::Utf8, false),
        Field::new("row_cat", DataType::Utf8, false),
        Field::new("v", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0])),
            Arc::new(StringArray::from(vec!["a", "a", "a", "a"])),
            Arc::new(StringArray::from(vec!["c1", "c2", "c1", "c2"])),
            Arc::new(StringArray::from(vec!["r1", "r1", "r2", "r2"])),
            // col_cat sum: c1 = 100+200 = 300, c2 = 1+2 = 3 -- ascending
            // by sum reverses the c1-first data order to [c2, c1].
            Arc::new(Float64Array::from(vec![100.0, 1.0, 200.0, 2.0])),
        ],
    )
    .unwrap();

    let mut spec = spec_color_facet();
    spec.facet = Some(crate::layout::FacetSpec {
        field: "col_cat".into(),
        row: Some("row_cat".into()),
        title: None,
        row_title: None,
        sort: Some(serde_json::json!({"field": "v", "op": "sum"})),
        row_sort: Some(serde_json::json!("descending")),
        repeat: Vec::new(),
        mode: crate::layout::FacetMode::Grid { nrows: 2, ncols: 2 },
        spacing: None,
        resolve: crate::layout::facet::FacetResolve::default(),
    });

    let prep = prepare_render_inputs(
        &spec,
        &batch,
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport { width: 600.0, height: 400.0 },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .unwrap();
    let pairs: Vec<(String, String)> = prep
        .facet_groups
        .iter()
        .map(|g| (g.row_key.as_ref().unwrap().value.clone(), g.key.value.clone()))
        .collect();
    // row_sort="descending" reverses ["r1", "r2"] -> ["r2", "r1"];
    // sort={"field":"v","op":"sum"} orders columns ["c2", "c1"];
    // row-major: (r2,c2), (r2,c1), (r1,c2), (r1,c1).
    assert_eq!(
        pairs,
        vec![
            ("r2".to_string(), "c2".to_string()),
            ("r2".to_string(), "c1".to_string()),
            ("r1".to_string(), "c2".to_string()),
            ("r1".to_string(), "c1".to_string()),
        ]
    );
}

// ── Multivariate B1: size / shape / merged legends ───────────────────

fn batch_pop() -> RecordBatch {
    // x, y, a numeric "pop" field (size/color domain ≈ [10, 100]), and a
    // categorical "region" field for color/shape.
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
        Field::new("pop", DataType::Float64, false),
        Field::new("region", DataType::Utf8, false),
        Field::new("grp", DataType::Utf8, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0])),
            Arc::new(Float64Array::from(vec![10.0, 40.0, 70.0, 100.0])),
            Arc::new(StringArray::from(vec!["AS", "EU", "AS", "AF"])),
            Arc::new(StringArray::from(vec!["lo", "hi", "lo", "hi"])),
        ],
    )
    .unwrap()
}

fn base_spec() -> ChartSpec {
    ChartSpec {
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec {
                field: "x".into(),
                type_: None,
                ..Default::default()
            }),
            y: Some(EncodingSpec {
                field: "y".into(),
                type_: None,
                ..Default::default()
            }),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn prep(spec: &ChartSpec, batch: &RecordBatch) -> PreparedInputs {
    prepare_render_inputs(
        spec,
        batch,
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport {
            width: 600.0,
            height: 400.0,
        },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .unwrap()
}

// ── M2 (batch D task 5): TransformContext.panel_pixel_size pre-layout
//    estimate, threaded from `run_transforms_with_facet`'s viewport/theme
//    params. These exercise the one production construction site through
//    `prepare_render_inputs` (not a hand-built `TransformContext`), so a
//    regression to `TransformContext::default()` there fails them. ────

fn raster_screen_spec() -> ChartSpec {
    use crate::transform::core::TransformSpec;
    use crate::transform::raster::{RasterSpec, ResolutionSpec};
    let mut spec = base_spec();
    spec.transforms = vec![TransformSpec::Raster(RasterSpec {
        x: "x".into(),
        y: "y".into(),
        aggregate: Default::default(),
        field: None,
        resolution: ResolutionSpec::Named("screen".into()),
        min_count: None,
        log_scale: false,
        name: None,
        extent_x: None,
        extent_y: None,
    })];
    spec
}

fn raster_fixed_spec(n: u32) -> ChartSpec {
    use crate::transform::core::TransformSpec;
    use crate::transform::raster::{RasterSpec, ResolutionSpec};
    let mut spec = base_spec();
    spec.transforms = vec![TransformSpec::Raster(RasterSpec {
        x: "x".into(),
        y: "y".into(),
        aggregate: Default::default(),
        field: None,
        resolution: ResolutionSpec::Fixed(n),
        min_count: None,
        log_scale: false,
        name: None,
        extent_x: None,
        extent_y: None,
    })];
    spec
}

/// Row 0's `(width, height)` columns of the raster transform's output
/// batch — the grid dimensions `resolution="screen"` actually resolved to.
fn raster_grid_dims(outputs: &HashMap<String, RecordBatch>) -> (u32, u32) {
    use arrow::array::UInt32Array;
    let batch = outputs.get(FINAL_OUTPUT_KEY).unwrap();
    let w = batch
        .column_by_name("width")
        .unwrap()
        .as_any()
        .downcast_ref::<UInt32Array>()
        .unwrap()
        .value(0);
    let h = batch
        .column_by_name("height")
        .unwrap()
        .as_any()
        .downcast_ref::<UInt32Array>()
        .unwrap()
        .value(0);
    (w, h)
}

/// F-L03-06's literal repro, at `run_transforms_with_facet` (the one
/// production `TransformContext` construction site): a larger viewport
/// must yield a larger `resolution="screen"` grid, and neither dimension
/// is pinned to the old hardcoded 256x256 fallback.
#[test]
fn raster_screen_resolution_scales_with_viewport() {
    let spec = raster_screen_spec();
    let batch = extent_pin_batch_2d();
    let small = run_transforms_with_facet(
        &spec,
        &batch,
        crate::layout::Viewport {
            width: 300.0,
            height: 200.0,
        },
        &crate::layout::ThemeInputs::default(), None,
    )
    .unwrap()
    .0;
    let large = run_transforms_with_facet(
        &spec,
        &batch,
        crate::layout::Viewport {
            width: 1200.0,
            height: 900.0,
        },
        &crate::layout::ThemeInputs::default(), None,
    )
    .unwrap()
    .0;
    let (sw, sh) = raster_grid_dims(&small);
    let (lw, lh) = raster_grid_dims(&large);
    assert_ne!(
        (sw, sh),
        (256, 256),
        "must not be pinned to the old hardcoded fallback"
    );
    assert_ne!(
        (lw, lh),
        (256, 256),
        "must not be pinned to the old hardcoded fallback"
    );
    assert!(
        lw > sw && lh > sh,
        "larger viewport must yield a larger screen grid: small=({sw},{sh}) large=({lw},{lh})"
    );
}

/// An explicit `resolution=64` is unaffected by the viewport — same
/// grid regardless of chart size.
#[test]
fn raster_explicit_resolution_unaffected_by_viewport() {
    let spec = raster_fixed_spec(64);
    let batch = extent_pin_batch_2d();
    let small = run_transforms_with_facet(
        &spec,
        &batch,
        crate::layout::Viewport {
            width: 300.0,
            height: 200.0,
        },
        &crate::layout::ThemeInputs::default(), None,
    )
    .unwrap()
    .0;
    let large = run_transforms_with_facet(
        &spec,
        &batch,
        crate::layout::Viewport {
            width: 1200.0,
            height: 900.0,
        },
        &crate::layout::ThemeInputs::default(), None,
    )
    .unwrap()
    .0;
    assert_eq!(raster_grid_dims(&small), (64, 64));
    assert_eq!(raster_grid_dims(&large), (64, 64));
}

/// A faceted chart's per-cell estimate divides the viewport by the
/// facet grid's own dimensions: `extent_pin_batch_2d()` carries two `p`
/// groups ("A", "B"); wrap mode with `ncols=1` makes `nrows=2`, so the
/// height estimate is roughly HALF the unfaceted estimate at the same
/// viewport, while the width estimate (1 column) is unchanged.
#[test]
fn raster_screen_resolution_facet_grid_divides_estimate() {
    let batch = extent_pin_batch_2d();
    let viewport = crate::layout::Viewport {
        width: 600.0,
        height: 400.0,
    };
    let theme = crate::layout::ThemeInputs::default();

    let unfaceted = run_transforms_with_facet(&raster_screen_spec(), &batch, viewport, &theme, None)
        .unwrap()
        .0;
    let (w0, h0) = raster_grid_dims(&unfaceted);

    let mut faceted_spec = raster_screen_spec();
    faceted_spec.facet = Some(crate::layout::FacetSpec {
        field: "p".into(),
        row: None,
        title: None,
        row_title: None,
        sort: None,
        row_sort: None,
        repeat: Vec::new(),
        mode: crate::layout::FacetMode::Wrap { ncols: 1 },
        spacing: None,
        resolve: crate::layout::FacetResolve::default(),
    });
    let faceted = run_transforms_with_facet(&faceted_spec, &batch, viewport, &theme, None)
        .unwrap()
        .0;
    let (w1, h1) = raster_grid_dims(&faceted);

    assert_eq!(w1, w0, "one column: width estimate must be unchanged");
    assert!(
        h1 < h0 && (h1 as f64 - h0 as f64 / 2.0).abs() <= 1.0,
        "two rows: height estimate must be roughly half the unfaceted estimate, got h0={h0} h1={h1}"
    );
}

/// A transform that never reads `panel_pixel_size` (Bin) is byte-identical
/// regardless of the viewport the estimate is computed from — the M2
/// estimate is plumbed to every transform's `TransformContext`, but only
/// raster/swarm/violin actually consume it.
#[test]
fn non_pixel_dependent_transform_byte_identical_across_viewports() {
    let spec = spec_with_one_bin(None);
    let batch = price_weight_batch();
    let theme = crate::layout::ThemeInputs::default();
    let small = run_transforms_with_facet(
        &spec,
        &batch,
        crate::layout::Viewport {
            width: 300.0,
            height: 200.0,
        },
        &theme, None,
    )
    .unwrap()
    .0;
    let large = run_transforms_with_facet(
        &spec,
        &batch,
        crate::layout::Viewport {
            width: 1200.0,
            height: 900.0,
        },
        &theme, None,
    )
    .unwrap()
    .0;
    let small_batch = small.get(FINAL_OUTPUT_KEY).unwrap();
    let large_batch = large.get(FINAL_OUTPUT_KEY).unwrap();
    assert_eq!(
        small_batch, large_batch,
        "Bin output must be byte-identical across viewports; it never reads panel_pixel_size"
    );
}

/// Round-2 required fix (rust-quality Fix 3): a degenerate viewport
/// whose (default) theme padding meets or exceeds it must not panic
/// `run_transforms_with_facet` — the production `TransformContext`
/// construction site. `layout::estimate_panel_pixel_size` floors each
/// axis at 1 exactly so this case can never hand the raster `"screen"`
/// arm a zero-sized `(nx, ny)` (the S4 bug the zero-guard fix closes).
/// The actual user-facing refusal for this geometry is
/// `compute_layout`'s typed `LayoutError::PaddingExceedsViewport`,
/// raised downstream of this call — see
/// `tests/test_panel_size_transforms.py`'s
/// `test_raster_on_degenerate_viewport_raises_typed_layout_error` for
/// that half of the contract through the public Python API.
#[test]
fn raster_screen_resolution_survives_degenerate_viewport_without_panicking() {
    let spec = raster_screen_spec();
    let batch = extent_pin_batch_2d();
    // 30x30 viewport with the default 16px-per-side theme padding: the
    // true inner extent is negative on both axes.
    let viewport = crate::layout::Viewport {
        width: 30.0,
        height: 30.0,
    };
    let theme = crate::layout::ThemeInputs::default();
    let outputs = run_transforms_with_facet(&spec, &batch, viewport, &theme, None)
        .expect("must not panic or error on a degenerate viewport")
        .0;
    assert_eq!(
        raster_grid_dims(&outputs),
        (1, 1),
        "the floored (1, 1) estimate, not a panic-inducing zero"
    );
}

// ── Batch-D T9 round 3: Smooth's rank-deficiency refusal must not fire
//    per facet panel. `render::prepare::run_transforms_with_facet`
//    partitions by the facet field BEFORE running transforms, so an
//    ungrouped `Smooth` on a facet panel is indistinguishable from a
//    true whole-frame call unless `TransformContext.is_partition`
//    disambiguates them (`smooth.rs`'s degenerate-input policy doc).
//    These tests are the RED proof this round's dispatch required: the
//    `col='sp'` faceted repro, a `groupby` twin proving the same data
//    degrades identically under `hue=`'s mechanism, and a regression
//    guard that a genuine whole-frame rank-deficient call still refuses. ──

/// Two facet panels sharing one `x`/`y` numeric pair: `"few"` has 3 rows
/// (rank-deficient for a degree-3 fit, which needs >= 4), `"many"` has 8
/// rows (sufficient). Mirrors `fm.lmplot(order=3, col='sp')`'s repro
/// shape at the Rust transform level.
fn facet_smooth_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
        Field::new("sp", DataType::Utf8, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![
                0.0, 1.0, 2.0, // "few": 3 rows
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, // "many": 8 rows
            ])),
            Arc::new(Float64Array::from(vec![
                0.0, 1.0, 4.0, // "few"
                0.0, 1.0, 8.0, 27.0, 64.0, 125.0, 216.0, 343.0, // "many": y = x^3
            ])),
            Arc::new(StringArray::from(vec![
                "few", "few", "few", "many", "many", "many", "many", "many", "many", "many",
                "many",
            ])),
        ],
    )
    .unwrap()
}

/// A `ChartSpec` fitting a degree-3 polynomial over `x`/`y`, optionally
/// faceted or grouped by `sp` (never both — the two are the "col=" vs
/// "hue=" shapes this round's fix must treat symmetrically).
fn smooth_cubic_spec(
    facet: Option<crate::layout::FacetSpec>,
    groupby: Vec<String>,
) -> ChartSpec {
    use crate::transform::core::TransformSpec;
    use crate::transform::smooth::{SmoothMethod, SmoothOutput, SmoothSpec};
    let mut spec = base_spec();
    spec.facet = facet;
    spec.transforms = vec![TransformSpec::Smooth(SmoothSpec {
        x: "x".into(),
        y: "y".into(),
        method: SmoothMethod::Poly { degree: 3 },
        ci: None,
        bandwidth: 0.75,
        degree: 3,
        k: crate::transform::gam::DEFAULT_BASIS_DIM,
        n: 5,
        seed: 0,
        x_bins: None,
        x_estimator: None,
        output: SmoothOutput::Fitted,
        inject_zero_ref: false,
        inject_metrics: false,
        groupby,
        name: None,
        x_range: None,
    })];
    spec
}

fn float_col(batch: &RecordBatch, name: &str) -> Vec<f64> {
    batch
        .column_by_name(name)
        .unwrap_or_else(|| panic!("missing column '{name}'"))
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap_or_else(|| panic!("column '{name}' is not Float64"))
        .values()
        .to_vec()
}

fn str_col(batch: &RecordBatch, name: &str) -> Vec<String> {
    batch
        .column_by_name(name)
        .unwrap_or_else(|| panic!("missing column '{name}'"))
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap_or_else(|| panic!("column '{name}' is not Utf8"))
        .iter()
        .map(|v| v.unwrap_or_default().to_string())
        .collect()
}

/// The blocking defect this round fixes: `fm.lmplot(order=3, col='sp')`
/// on a sparse level used to raise `ValueError: transform failed:
/// Smooth: degree=3 needs at least 4 data points...` even though the
/// identical data under `hue='sp'` rendered. A facet panel is an
/// ungrouped `apply()` call on a data subset — `only_indices.is_none()`
/// alone cannot tell it apart from a true whole-frame call. Post-fix:
/// the render succeeds, the sparse `"few"` panel degrades to NaN (T6's
/// contract), and the well-supported `"many"` panel still fits.
#[test]
fn smooth_facet_panel_rank_deficient_degrades_instead_of_refusing() {
    let facet = crate::layout::FacetSpec {
        field: "sp".into(),
        row: None,
        title: None,
        row_title: None,
        sort: None,
        row_sort: None,
        repeat: Vec::new(),
        mode: crate::layout::FacetMode::Wrap { ncols: 2 },
        spacing: None,
        resolve: crate::layout::FacetResolve::default(),
    };
    let spec = smooth_cubic_spec(Some(facet), vec![]);
    let batch = facet_smooth_batch();
    let viewport = crate::layout::Viewport {
        width: 600.0,
        height: 400.0,
    };
    let theme = crate::layout::ThemeInputs::default();

    let (outputs, warnings, _, _) = run_transforms_with_facet(&spec, &batch, viewport, &theme, None)
        .expect("a facet panel too sparse for degree=3 must degrade, not fail the chart");
    let out = outputs.get(FINAL_OUTPUT_KEY).unwrap();
    // #162: the sparse 'few' panel's degenerate fit must warn.
    assert!(
        warnings.iter().any(|w| matches!(
            w,
            crate::transform::context::TransformWarning::DegenerateFit { transform, .. }
                if transform == "Smooth"
        )),
        "a facet panel too sparse for degree=3 must push a DegenerateFit TransformWarning; \
         got {warnings:?}"
    );

    let sp = str_col(out, "sp");
    let y = float_col(out, "y");
    assert_eq!(sp.len(), 10, "5 grid points per panel x 2 panels");

    let few_y: Vec<f64> = sp
        .iter()
        .zip(&y)
        .filter(|(s, _)| *s == "few")
        .map(|(_, v)| *v)
        .collect();
    let many_y: Vec<f64> = sp
        .iter()
        .zip(&y)
        .filter(|(s, _)| *s == "many")
        .map(|(_, v)| *v)
        .collect();
    assert_eq!(few_y.len(), 5);
    assert_eq!(many_y.len(), 5);
    assert!(
        few_y.iter().all(|v| v.is_nan()),
        "the rank-deficient 'few' panel (3 points, degree=3) must degrade to NaN: {few_y:?}"
    );
    assert!(
        many_y.iter().all(|v| v.is_finite()),
        "the well-supported 'many' panel (8 points, degree=3) must still fit: {many_y:?}"
    );
}

/// Symmetry twin (this round's dispatch: "the hue= twin asserting
/// symmetry"): the identical batch and degree, routed through `Smooth`'s
/// own `groupby` (the `hue=` mechanism) instead of chart-level faceting.
/// `groupby` already degraded correctly before this round (T6's
/// contract) — this pins that the facet fix did not change that path,
/// and that both partitioning mechanisms now produce the same per-slice
/// disposition for identical data.
#[test]
fn smooth_groupby_rank_deficient_degrades_symmetrically_with_facet() {
    let spec = smooth_cubic_spec(None, vec!["sp".into()]);
    let batch = facet_smooth_batch();
    let viewport = crate::layout::Viewport {
        width: 600.0,
        height: 400.0,
    };
    let theme = crate::layout::ThemeInputs::default();

    let (outputs, warnings, _, _) = run_transforms_with_facet(&spec, &batch, viewport, &theme, None)
        .expect("a groupby group too sparse for degree=3 must degrade, not fail the chart");
    let out = outputs.get(FINAL_OUTPUT_KEY).unwrap();
    assert!(
        warnings.iter().any(|w| matches!(
            w,
            crate::transform::context::TransformWarning::DegenerateFit { transform, group, .. }
                if transform == "Smooth" && group.as_deref() == Some("sp=few")
        )),
        "the sparse 'few' groupby group's degenerate fit must push a named DegenerateFit \
         warning; got {warnings:?}"
    );

    let sp = str_col(out, "sp");
    let y = float_col(out, "y");
    let few_y: Vec<f64> = sp
        .iter()
        .zip(&y)
        .filter(|(s, _)| *s == "few")
        .map(|(_, v)| *v)
        .collect();
    let many_y: Vec<f64> = sp
        .iter()
        .zip(&y)
        .filter(|(s, _)| *s == "many")
        .map(|(_, v)| *v)
        .collect();
    assert_eq!(few_y.len(), 5);
    assert_eq!(many_y.len(), 5);
    assert!(
        few_y.iter().all(|v| v.is_nan()),
        "groupby's sparse 'few' group must degrade to NaN, matching the facet panel: {few_y:?}"
    );
    assert!(
        many_y.iter().all(|v| v.is_finite()),
        "groupby's well-supported 'many' group must still fit: {many_y:?}"
    );
}

/// Regression guard: a genuine whole-frame call (no facet, no groupby,
/// the batch itself too sparse for the requested degree) must still
/// refuse with the typed `PyValueError` — the D9 closure this round
/// must not undo while fixing the facet-panel misfire above.
#[test]
fn smooth_whole_frame_rank_deficient_still_refuses() {
    let spec = smooth_cubic_spec(None, vec![]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0, 4.0])),
        ],
    )
    .unwrap();
    let viewport = crate::layout::Viewport {
        width: 600.0,
        height: 400.0,
    };
    let theme = crate::layout::ThemeInputs::default();

    let err = run_transforms_with_facet(&spec, &batch, viewport, &theme, None)
        .expect_err("a true whole-frame rank-deficient degree=3 fit must still refuse");
    let msg = err.to_string();
    assert!(
        msg.contains("degree=3") && msg.contains("rank-deficient"),
        "expected the D9 rank-deficiency refusal, got: {msg}"
    );
}

// ── Batch-D T9 round 4: `concat_batches_by_name` must be dtype-generic ──
//
// Round 3's `is_partition` fix eliminated the metrics-column schema
// divergence at its source (`smooth::build_degenerate_smooth_batch` now
// always appends `_metrics_text`/`_metrics_y`, null, when requested), so
// that divergence no longer reaches `concat_batches_by_name`'s widening
// path. A DIFFERENT divergence still does: a facet panel whose
// `groupby` field is entirely null degrades via `smooth::apply_grouped`'s
// empty-partition fallback, which returns a plain (ungrouped-shaped)
// batch carrying NO group column at all, while a sibling panel with real
// groups appends that column at its ORIGINAL dtype (XFORM-03). Before
// this round's fix, a non-Float64/Utf8 group column in that situation
// raised "schema divergence across panels is only reconcilable for
// Float64/Utf8 columns" — the live repro the rust-quality cycle-3
// verdict reproduced through the public Python API
// (`groupby="gi"` (Int64) + `show_metrics` + `.facet(col=...)`).

/// Two facet panels sharing one `x`/`y`/`gi` triple, `gi` an `Int64`
/// `groupby` field: panel `"p1"`'s `gi` is entirely null (so
/// `apply_grouped`'s empty-partition fallback fires and its own output
/// carries no `gi` column), panel `"p2"`'s `gi` has two valid groups (so
/// its output carries a real `Int64` `gi` column).
fn facet_groupby_int64_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
        Field::new("sp", DataType::Utf8, false),
        Field::new("gi", DataType::Int64, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![
                0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 1.0,
            ])),
            Arc::new(Float64Array::from(vec![
                0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 2.0,
            ])),
            Arc::new(StringArray::from(vec![
                "p1", "p1", "p1", "p1", "p2", "p2", "p2", "p2",
            ])),
            Arc::new(arrow::array::Int64Array::from(vec![
                None,
                None,
                None,
                None,
                Some(1),
                Some(1),
                Some(2),
                Some(2),
            ])),
        ],
    )
    .unwrap()
}

/// Boolean twin of [`facet_groupby_int64_batch`] — same shape, `gb`
/// instead of `gi`, pinning the second dtype the reviewer's finding
/// named.
fn facet_groupby_boolean_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
        Field::new("sp", DataType::Utf8, false),
        Field::new("gb", DataType::Boolean, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![
                0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 1.0,
            ])),
            Arc::new(Float64Array::from(vec![
                0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 2.0,
            ])),
            Arc::new(StringArray::from(vec![
                "p1", "p1", "p1", "p1", "p2", "p2", "p2", "p2",
            ])),
            Arc::new(arrow::array::BooleanArray::from(vec![
                None,
                None,
                None,
                None,
                Some(true),
                Some(true),
                Some(false),
                Some(false),
            ])),
        ],
    )
    .unwrap()
}

/// A degree-1 (`Lm`) `Smooth`, optionally faceted and/or grouped, with
/// `inject_metrics: true` — combining the metrics-column presence (round
/// 3's already-closed divergence) with whatever group-column divergence
/// `groupby` introduces, matching the reviewer's exact repro shape.
fn smooth_lm_groupby_spec(
    facet: Option<crate::layout::FacetSpec>,
    groupby: Vec<String>,
) -> ChartSpec {
    use crate::transform::core::TransformSpec;
    use crate::transform::smooth::{SmoothMethod, SmoothOutput, SmoothSpec};
    let mut spec = base_spec();
    spec.facet = facet;
    spec.transforms = vec![TransformSpec::Smooth(SmoothSpec {
        x: "x".into(),
        y: "y".into(),
        method: SmoothMethod::Lm,
        ci: None,
        bandwidth: 0.75,
        degree: 1,
        k: crate::transform::gam::DEFAULT_BASIS_DIM,
        n: 3,
        seed: 0,
        x_bins: None,
        x_estimator: None,
        output: SmoothOutput::Fitted,
        inject_zero_ref: false,
        inject_metrics: true,
        groupby,
        name: None,
        x_range: None,
    })];
    spec
}

fn int64_col(batch: &RecordBatch, name: &str) -> Vec<Option<i64>> {
    batch
        .column_by_name(name)
        .unwrap_or_else(|| panic!("missing column '{name}'"))
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .unwrap_or_else(|| panic!("column '{name}' is not Int64"))
        .iter()
        .collect()
}

fn bool_col(batch: &RecordBatch, name: &str) -> Vec<Option<bool>> {
    batch
        .column_by_name(name)
        .unwrap_or_else(|| panic!("missing column '{name}'"))
        .as_any()
        .downcast_ref::<arrow::array::BooleanArray>()
        .unwrap_or_else(|| panic!("column '{name}' is not Boolean"))
        .iter()
        .collect()
}

/// The live repro's dtype: an `Int64` `groupby` column present in one
/// facet panel's output and absent from another's must widen (null-pad
/// the panel missing it), not raise "only reconcilable for
/// Float64/Utf8".
#[test]
fn smooth_facet_groupby_int64_column_widens_instead_of_erroring() {
    let facet = crate::layout::FacetSpec {
        field: "sp".into(),
        row: None,
        title: None,
        row_title: None,
        sort: None,
        row_sort: None,
        repeat: Vec::new(),
        mode: crate::layout::FacetMode::Wrap { ncols: 2 },
        spacing: None,
        resolve: crate::layout::FacetResolve::default(),
    };
    let spec = smooth_lm_groupby_spec(Some(facet), vec!["gi".into()]);
    let batch = facet_groupby_int64_batch();
    let viewport = crate::layout::Viewport {
        width: 600.0,
        height: 400.0,
    };
    let theme = crate::layout::ThemeInputs::default();

    let outputs = run_transforms_with_facet(&spec, &batch, viewport, &theme, None)
        .expect("Int64 groupby column divergence across facet panels must widen, not error")
        .0;
    let out = outputs.get(FINAL_OUTPUT_KEY).unwrap();

    // Round 3's metrics-column fix already makes these uniform; confirm
    // both fixes coexist rather than one masking a regression in the other.
    assert!(out.schema().index_of("_metrics_text").is_ok());
    assert!(out.schema().index_of("_metrics_y").is_ok());

    let sp = str_col(out, "sp");
    let gi = int64_col(out, "gi");
    assert_eq!(sp.len(), gi.len());
    for (s, g) in sp.iter().zip(&gi) {
        if s == "p1" {
            assert!(
                g.is_none(),
                "panel 'p1' produced no 'gi' groups; its rows must pad null: {gi:?}"
            );
        } else {
            assert!(
                g.is_some(),
                "panel 'p2' has real 'gi' groups; its rows must carry the real value: {gi:?}"
            );
        }
    }
    assert!(
        gi.iter().any(|g| *g == Some(1)) && gi.iter().any(|g| *g == Some(2)),
        "panel 'p2's two groups (gi=1, gi=2) must both survive the merge: {gi:?}"
    );
}

/// Boolean twin: the reviewer's second named dtype.
#[test]
fn smooth_facet_groupby_boolean_column_widens_instead_of_erroring() {
    let facet = crate::layout::FacetSpec {
        field: "sp".into(),
        row: None,
        title: None,
        row_title: None,
        sort: None,
        row_sort: None,
        repeat: Vec::new(),
        mode: crate::layout::FacetMode::Wrap { ncols: 2 },
        spacing: None,
        resolve: crate::layout::FacetResolve::default(),
    };
    let spec = smooth_lm_groupby_spec(Some(facet), vec!["gb".into()]);
    let batch = facet_groupby_boolean_batch();
    let viewport = crate::layout::Viewport {
        width: 600.0,
        height: 400.0,
    };
    let theme = crate::layout::ThemeInputs::default();

    let outputs = run_transforms_with_facet(&spec, &batch, viewport, &theme, None)
        .expect("Boolean groupby column divergence across facet panels must widen, not error")
        .0;
    let out = outputs.get(FINAL_OUTPUT_KEY).unwrap();

    let sp = str_col(out, "sp");
    let gb = bool_col(out, "gb");
    assert_eq!(sp.len(), gb.len());
    for (s, g) in sp.iter().zip(&gb) {
        if s == "p1" {
            assert!(
                g.is_none(),
                "panel 'p1' produced no 'gb' groups; its rows must pad null: {gb:?}"
            );
        } else {
            assert!(
                g.is_some(),
                "panel 'p2' has real 'gb' groups; its rows must carry the real value: {gb:?}"
            );
        }
    }
    assert!(
        gb.iter().any(|g| *g == Some(true)) && gb.iter().any(|g| *g == Some(false)),
        "panel 'p2's two groups (gb=true, gb=false) must both survive the merge: {gb:?}"
    );
}

#[test]
fn size_encoding_produces_graduated_size_legend() {
    let mut spec = base_spec();
    spec.encoding.size = Some(EncodingSpec {
        field: "pop".into(),
        type_: None,
        ..Default::default()
    });
    let p = prep(&spec, &batch_pop());
    assert_eq!(
        p.aux_legends.len(),
        1,
        "expected exactly one (size) aux legend"
    );
    let AuxLegendInput::Size { entries, title, .. } = &p.aux_legends[0] else {
        panic!("expected a Size aux legend, got {:?}", p.aux_legends[0]);
    };
    assert_eq!(title.as_deref(), Some("pop"));
    // ~5 nice round values across the [10, 100] domain. nice_ticks(10,100,5)
    // yields 20,40,60,80,100 — labels are the round values, not raw quantiles.
    assert!(
        (4..=6).contains(&entries.len()),
        "expected ~5 graduated entries, got {}: {:?}",
        entries.len(),
        entries
    );
    // Labels are human-friendly round numbers.
    for e in entries {
        assert!(e.radius > 0.0, "size symbol radius must be positive");
        let v: f64 = e.label.parse().expect("size legend label must be numeric");
        assert_eq!(
            v.fract(),
            0.0,
            "expected round value label, got {}",
            e.label
        );
    }
    // Radii increase with value (larger pop → bigger symbol).
    assert!(
        entries.first().unwrap().radius < entries.last().unwrap().radius,
        "radii should grow across the size domain"
    );
}

#[test]
fn shape_encoding_produces_shape_legend_one_per_category() {
    let mut spec = base_spec();
    spec.encoding.shape = Some(EncodingSpec {
        field: "region".into(),
        type_: None,
        ..Default::default()
    });
    let p = prep(&spec, &batch_pop());
    assert_eq!(p.aux_legends.len(), 1);
    let AuxLegendInput::Shape { entries, title } = &p.aux_legends[0] else {
        panic!("expected a Shape aux legend");
    };
    assert_eq!(title.as_deref(), Some("region"));
    // Distinct regions in encounter order: AS, EU, AF.
    let labels: Vec<&str> = entries.iter().map(|e| e.label.as_str()).collect();
    assert_eq!(labels, vec!["AS", "EU", "AF"]);
    // First three palette shapes: circle, square, cross.
    assert_eq!(entries[0].shape_name, "circle");
    assert_eq!(entries[1].shape_name, "square");
    assert_eq!(entries[2].shape_name, "cross");
}

#[test]
fn color_size_shape_together_produce_two_aux_blocks_plus_color() {
    // color on region (categorical → color legend), size on pop, shape on a
    // 4th distinct field. Color legend is separate (legend_entries); size +
    // shape are two stacked aux blocks.
    let mut spec = base_spec();
    spec.encoding.color = Some(EncodingSpec {
        field: "region".into(),
        type_: None,
        ..Default::default()
    });
    spec.encoding.size = Some(EncodingSpec {
        field: "pop".into(),
        type_: None,
        ..Default::default()
    });
    spec.encoding.shape = Some(EncodingSpec {
        field: "grp".into(),
        type_: None,
        ..Default::default()
    });
    let p = prep(&spec, &batch_pop());
    // Color drives a categorical legend (legend_entries), not an aux block.
    assert!(
        !p.legend_entries.is_empty(),
        "color legend entries expected"
    );
    // Two aux blocks: size then shape, stable order.
    assert_eq!(p.aux_legends.len(), 2, "expected size + shape aux blocks");
    assert!(
        matches!(p.aux_legends[0], AuxLegendInput::Size { .. }),
        "size first"
    );
    assert!(
        matches!(p.aux_legends[1], AuxLegendInput::Shape { .. }),
        "shape second"
    );
}

/// A binned color draws one legend: the stepped bar of its bins, labelled at
/// each bin edge so each band reads as its bin's range, and titled by the
/// field, or by the binding's own `title=`. Rejects a categorical legend of
/// the raw values and a gradient colorbar of the continuous ramp.
#[test]
fn a_binned_color_draws_one_legend_of_its_bin_ranges_titled_by_its_field() {
    for (title, expected_title) in [(None, "pop"), (Some("Population"), "Population")] {
        let mut spec = base_spec();
        spec.encoding.color = Some(EncodingSpec {
            field: "pop".into(),
            title: title.map(String::from),
            scale: Some(
                serde_json::from_value(serde_json::json!({"type": "bin-ordinal", "bin": {"bin_count": 3, "nice": false}}))
                    .unwrap(),
            ),
            ..Default::default()
        });
        let p = prep(&spec, &batch_pop());
        assert!(p.legend_entries.is_empty(), "no categorical legend: {:?}", p.legend_entries);
        assert!(p.aux_legends.is_empty(), "no second legend");
        let colorbar = p.colorbar.as_ref().expect("the binned legend");
        assert_eq!(colorbar.tick_labels, vec!["10", "40", "70", "100"], "one label per bin edge");
        assert_eq!(colorbar.stops.len(), 6, "three flat bands, two stops each");
        for pair in colorbar.stops.chunks_exact(2) {
            assert_eq!(pair[0].1, pair[1].1, "each band is one flat swatch");
        }
        assert_eq!(p.legend_title.as_deref(), Some(expected_title));
    }
}

#[test]
fn color_and_size_on_same_field_merge_into_one_block() {
    // color (continuous) AND size both on "pop" → a single combined block:
    // a size legend whose symbols also carry color (color_hex), and NO
    // colorbar (it would be the second, redundant legend).
    let mut spec = base_spec();
    spec.encoding.color = Some(EncodingSpec {
        field: "pop".into(),
        type_: Some(crate::spec::encoding::DataType::Quantitative),
        ..Default::default()
    });
    spec.encoding.size = Some(EncodingSpec {
        field: "pop".into(),
        type_: None,
        ..Default::default()
    });
    let p = prep(&spec, &batch_pop());
    // Exactly one merged block, and the colorbar is suppressed.
    assert_eq!(p.aux_legends.len(), 1, "merge → single block, not two");
    assert!(
        p.colorbar.is_none(),
        "colorbar must be suppressed in the merged case"
    );
    let AuxLegendInput::Size { entries, .. } = &p.aux_legends[0] else {
        panic!("merged block should be a Size legend carrying color");
    };
    assert!(
        entries.iter().all(|e| e.color_hex.is_some()),
        "merged size entries must carry per-entry color"
    );
}

#[test]
fn size_legend_disabled_suppresses_block() {
    let mut spec = base_spec();
    spec.encoding.size = Some(EncodingSpec {
        field: "pop".into(),
        legend: Some(Box::new(crate::render::chart_config::LegendStyleSpec {
            disabled: Some(true),
            ..Default::default()
        })),
        ..Default::default()
    });
    let p = prep(&spec, &batch_pop());
    assert!(
        p.aux_legends.is_empty(),
        "legend=None on size must suppress the block"
    );
}

#[test]
fn color_only_chart_has_no_aux_legends() {
    // Regression: a color-only chart produces zero aux legends (the color
    // legend / colorbar path is untouched).
    let mut spec = base_spec();
    spec.encoding.color = Some(EncodingSpec {
        field: "region".into(),
        type_: None,
        ..Default::default()
    });
    let p = prep(&spec, &batch_pop());
    assert!(p.aux_legends.is_empty());
    assert!(!p.legend_entries.is_empty(), "color legend still present");
}

#[test]
fn empty_batch_errors() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(Vec::<f64>::new())),
            Arc::new(Float64Array::from(Vec::<f64>::new())),
        ],
    )
    .unwrap();
    let mut spec = spec_color_facet();
    spec.encoding.color = None;
    spec.facet = None;
    let err = prepare_render_inputs(
        &spec,
        &batch,
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport {
            width: 600.0,
            height: 400.0,
        },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .unwrap_err();
    assert!(matches!(err, RenderError::EmptyBatch));
}

// --- Phase 8a Task 6 tests ---

/// Helper: simple 2-column float batch with named fields.
fn price_weight_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("price", DataType::Float64, false),
        Field::new("weight", DataType::Float64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
        ],
    )
    .unwrap()
}

/// Helper: single-layer spec with x="price", y="weight".
fn single_layer_spec() -> ChartSpec {
    ChartSpec {
        data: crate::spec::data_ref::DataRef::default(),
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec {
                field: "price".into(),
                type_: None,
                ..Default::default()
            }),
            y: Some(EncodingSpec {
                field: "weight".into(),
                type_: None,
                ..Default::default()
            }),
            color: None,
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn prepare_single_layer_produces_one_layer_prepared() {
    let spec = single_layer_spec();
    let batch = price_weight_batch();
    let prepared = prepare_render_inputs(
        &spec,
        &batch,
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport {
            width: 600.0,
            height: 400.0,
        },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .unwrap();
    assert_eq!(prepared.layers.len(), 1);
    assert_eq!(prepared.layers[0].mark, Mark::Point);
    assert!(!prepared.coord_flipped);
    // Encoding fields should match spec
    assert_eq!(
        prepared.layers[0].encoding.x.as_ref().unwrap().field,
        "price"
    );
    assert_eq!(
        prepared.layers[0].encoding.y.as_ref().unwrap().field,
        "weight"
    );
}

#[test]
fn prepare_multi_layer_produces_multiple_layer_prepared() {
    use crate::spec::layer::Layer;
    let mut spec = single_layer_spec();
    // Two layers: point on price/weight, line inheriting chart encoding
    spec.layers = Some(vec![
        Layer {
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec {
                    field: "price".into(),
                    type_: None,
                    ..Default::default()
                }),
                y: Some(EncodingSpec {
                    field: "weight".into(),
                    type_: None,
                    ..Default::default()
                }),
                ..Default::default()
            },
            ..Default::default()
        },
        Layer {
            mark: Mark::Line,
            encoding: Encoding::default(), // inherits from chart-level
            transforms: vec![],
            ..Default::default()
        },
    ]);
    let batch = price_weight_batch();
    let prepared = prepare_render_inputs(
        &spec,
        &batch,
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport {
            width: 600.0,
            height: 400.0,
        },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .unwrap();
    assert_eq!(prepared.layers.len(), 2);
    assert_eq!(prepared.layers[0].mark, Mark::Point);
    assert_eq!(prepared.layers[1].mark, Mark::Line);
    // Layer 2 inherits chart-level encoding
    assert_eq!(
        prepared.layers[1].encoding.x.as_ref().unwrap().field,
        "price"
    );
    assert_eq!(
        prepared.layers[1].encoding.y.as_ref().unwrap().field,
        "weight"
    );
}

// --- Phase 8b Task 9: named-output transform routing ---

/// Build a ChartSpec with one Bin transform whose `name` is configurable,
/// and `bin_count` such that the transform succeeds on price_weight_batch().
///
/// When `name` is None, the bin transform is unnamed → it chains, so
/// `__final__` has bin output schema and the encoding is pointed at bin
/// columns. When `name` is Some, fan-out semantics apply: `__final__`
/// retains the original schema, so the encoding stays on the original
/// columns to keep `resolve_scales` happy.
fn spec_with_one_bin(name: Option<String>) -> ChartSpec {
    use crate::transform::bin::BinSpec;
    use crate::transform::bin_mode::BinMode;
    use crate::transform::core::TransformSpec;
    let named = name.is_some();
    let mut spec = single_layer_spec();
    spec.transforms = vec![TransformSpec::Bin(BinSpec {
        field: "price".into(),
        mode: BinMode::Fixed { n: 2 },
        extent: Some((10.0, 30.0)),
        nice: false,
        cumulative: false,
        shared_extent: false,
        groupby: vec![],
        agg_field: None,
        agg: None,
        name,
        per_row: false,
        aggregates: None,
    })];
    if !named {
        // Unnamed/chained: after Bin, the encoding fields ("price", "weight")
        // no longer exist in __final__ — point at bin output columns so
        // resolve_scales doesn't fail.
        spec.encoding.x = Some(crate::spec::encoding::EncodingSpec {
            field: "bin_start".into(),
            type_: None,
            ..Default::default()
        });
        spec.encoding.y = Some(crate::spec::encoding::EncodingSpec {
            field: "count".into(),
            type_: None,
            ..Default::default()
        });
    }
    // Named/fan-out: __final__ keeps the original price/weight schema, so
    // the chart-level encoding (price, weight) from `single_layer_spec()`
    // still resolves against __final__ correctly.
    spec
}

#[test]
fn data_source_none_uses_final_pipeline_output() {
    pyo3::Python::initialize();
    let spec = spec_with_one_bin(None);
    let batch = price_weight_batch();
    let prep = prepare_render_inputs(
        &spec,
        &batch,
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport {
            width: 600.0,
            height: 400.0,
        },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .unwrap();
    // __final__ is always present.
    assert!(
        prep.transform_outputs.contains_key("__final__"),
        "transform_outputs must always publish __final__"
    );
    // Bin had no name → its output is NOT separately keyed.
    assert_eq!(
        prep.transform_outputs.len(),
        1,
        "expected only __final__, got keys: {:?}",
        prep.transform_outputs.keys().collect::<Vec<_>>()
    );
    // final_batch() returns the FINAL_OUTPUT_KEY entry.
    let final_batch = prep.transform_outputs.get("__final__").unwrap();
    assert_eq!(prep.final_batch().num_rows(), final_batch.num_rows());
    assert_eq!(prep.final_batch().num_columns(), final_batch.num_columns());
    assert_eq!(
        prep.final_batch().schema(),
        final_batch.schema(),
        "final_batch() and __final__ schemas must match"
    );
}

#[test]
fn data_source_some_publishes_named_transform_output() {
    pyo3::Python::initialize();
    let spec = spec_with_one_bin(Some("box".into()));
    let batch = price_weight_batch();
    let prep = prepare_render_inputs(
        &spec,
        &batch,
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport {
            width: 600.0,
            height: 400.0,
        },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .unwrap();
    assert!(prep.transform_outputs.contains_key("box"));
    assert!(prep.transform_outputs.contains_key("__final__"));
    // Under fan-out semantics, named transforms run on the ORIGINAL input
    // and do NOT advance the chained pipeline. The named "box" output is
    // the bin output; __final__ is the original input (since no unnamed
    // transforms advanced the chain).
    let named = prep.transform_outputs.get("box").unwrap();
    let fin = prep.transform_outputs.get("__final__").unwrap();
    // The named bin output has the bin schema (bin_start/bin_end/count/density).
    let named_schema = named.schema();
    let named_fields: Vec<&str> = named_schema
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .collect();
    assert!(
        named_fields.contains(&"bin_start") && named_fields.contains(&"count"),
        "named output should have bin schema, got: {:?}",
        named_fields
    );
    // __final__ retains the original schema (price + weight).
    let final_schema = fin.schema();
    let final_fields: Vec<&str> = final_schema
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .collect();
    assert!(
        final_fields.contains(&"price") && final_fields.contains(&"weight"),
        "__final__ should have original schema, got: {:?}",
        final_fields
    );
    // And — proving the change — the named output and __final__ schemas differ.
    assert_ne!(named.schema(), fin.schema());
}

/// A hand-built layered spec whose layer carries its own non-empty
/// `transforms` is refused naming that layer; an empty list prepares.
#[test]
fn layer_own_transforms_refused_naming_the_layer() {
    pyo3::Python::initialize();
    use crate::spec::layer::Layer;
    let layer = |transforms| Layer {
        mark: Mark::Point,
        encoding: Encoding::default(),
        transforms,
        ..Default::default()
    };
    let prepare = |spec: &ChartSpec| {
        prepare_render_inputs(
            spec,
            &price_weight_batch(),
            &crate::layout::ThemeInputs::default(),
            crate::layout::Viewport {
                width: 600.0,
                height: 400.0,
            },
            &crate::render::chart_config::ChartConfig::default(),
            None,
        )
    };

    let mut spec = spec_with_one_bin(None);
    let bin = spec.transforms.clone();
    spec.layers = Some(vec![layer(vec![]), layer(bin)]);
    let err = prepare(&spec)
        .err()
        .expect("layer 1's own transforms must refuse");
    assert!(
        matches!(
            err,
            RenderError::LayerTransformsUnsupported { layer: 1, count: 1 }
        ),
        "wrong variant: {err:?}"
    );
    assert_eq!(
        err.to_string(),
        "layer 1: layers[1].transforms carries 1 transform(s), but a layer does not run its \
         own transforms. Give the layer its own chart with .transform(...) and compose it \
         with +, which routes the transform to that layer alone."
    );

    // Vacuity control: the same spec with every layer's list empty prepares.
    spec.layers = Some(vec![layer(vec![]), layer(vec![])]);
    assert!(
        prepare(&spec).is_ok(),
        "empty layer transforms must stay accepted"
    );
}

/// A hand-built spec whose `layers` list is empty refuses as a typed error
/// rather than reading a layer 0 that does not exist; one layer prepares.
#[test]
fn an_empty_layers_list_refuses() {
    pyo3::Python::initialize();
    use crate::spec::layer::Layer;
    let prepare = |spec: &ChartSpec| {
        prepare_render_inputs(
            spec,
            &price_weight_batch(),
            &crate::layout::ThemeInputs::default(),
            crate::layout::Viewport {
                width: 600.0,
                height: 400.0,
            },
            &crate::render::chart_config::ChartConfig::default(),
            None,
        )
    };
    let mut spec = spec_with_one_bin(None);
    spec.layers = Some(vec![]);
    let err = prepare(&spec).err().expect("an empty layers list must refuse");
    assert!(matches!(err, RenderError::LayeredChartNoLayers), "wrong variant: {err:?}");
    assert_eq!(
        err.to_string(),
        "ChartSpec.layers is an empty list: a layered chart draws its layers, and this one \
         has none. Add at least one layer, or leave layers unset to draw the chart's own mark."
    );

    spec.layers = Some(vec![Layer { mark: Mark::Point, encoding: Encoding::default(), ..Default::default() }]);
    assert!(prepare(&spec).is_ok(), "one layer must prepare");
}

// ── `#176`(11) + N3: malformed sort refuses at render ─────────────────────
//
// Python's construction-time gate (`EncodingSpec.__new__`,
// `parse_sort_field_object`) refuses every malformed sort-field object
// built through `fm.X`/`fm.Facet`/etc. before a spec can exist. These four
// routes bypass that gate entirely -- `refuse_malformed_sort_specs`'s
// reason for existing -- so each gets its own test against the SAME
// deserialization/construction machinery the real bypass uses, not a
// hand-built `RenderError` or a shape the production path never emits.

fn prepare_with(spec: &ChartSpec, batch: &RecordBatch) -> Result<PreparedInputs, RenderError> {
    prepare_render_inputs(
        spec,
        batch,
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport { width: 600.0, height: 400.0 },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
}

/// Route 1: `ChartSpec.from_json`. `EncodingSpec`'s derived `Deserialize`
/// (unlike its PyO3 `#[new]` constructor) does not validate `sort` at
/// all, so a malformed chart-channel sort object deserializes cleanly and
/// only meets a check at render. `ChartSpec::from_json` itself is a thin
/// `serde_json::from_str::<ChartSpec>` wrapper around a PyO3 classmethod;
/// parsing the identical JSON string exercises the identical
/// deserialization path without needing a GIL `PyType` object.
#[test]
fn refuse_malformed_sort_specs_chart_channel_via_from_json() {
    pyo3::Python::initialize();
    let json = r#"{"mark":"point","encoding":{"x":{"field":"price"},"y":{"field":"weight","sort":{"op":"mean"}}}}"#;
    let spec: ChartSpec = serde_json::from_str(json)
        .expect("a malformed sort object still deserializes: EncodingSpec's derived Deserialize does not validate it");
    let err = prepare_with(&spec, &price_weight_batch())
        .expect_err("a malformed chart-channel sort object must refuse");
    assert!(
        matches!(&err, RenderError::SortSpecMalformed { location, .. } if location == "y"),
        "wrong variant/location: {err:?}"
    );
    assert_eq!(
        err.to_string(),
        "y: sort={\"op\":\"mean\"} is malformed (sort object missing string \"field\")"
    );
}

/// Route 2: a layer's own `encoding` (`coerce_layers`, raw serde
/// `Deserialize`, never `EncodingSpec::new`). Mirrors
/// `layer_own_transforms_refused_naming_the_layer`'s fixture shape.
#[test]
fn refuse_malformed_sort_specs_layer_channel_names_the_layer() {
    pyo3::Python::initialize();
    use crate::spec::layer::Layer;
    let layer = |y_sort: Option<serde_json::Value>| Layer {
        mark: Mark::Point,
        encoding: Encoding {
            y: Some(EncodingSpec { field: "weight".into(), sort: y_sort, ..Default::default() }),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut spec = single_layer_spec();
    spec.layers = Some(vec![layer(None), layer(Some(serde_json::json!({"field": 1})))]);
    let err = prepare_with(&spec, &price_weight_batch())
        .expect_err("layer 1's own malformed sort must refuse");
    assert_eq!(
        err.to_string(),
        "layers[1].y: sort={\"field\":1} is malformed (sort \"field\" must be a string, got 1)"
    );

    // Vacuity control: the same spec with a well-formed sort on layer 1 prepares.
    spec.layers = Some(vec![layer(None), layer(Some(serde_json::json!({"field": "weight"})))]);
    assert!(
        prepare_with(&spec, &price_weight_batch()).is_ok(),
        "a well-formed layer sort must not refuse"
    );
}

/// A 2-row batch with two categorical facet dimensions (`col_cat`,
/// `row_cat`) alongside `x`/`y`, for the facet-sort test routes below.
fn facet_grid_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
        Field::new("col_cat", DataType::Utf8, false),
        Field::new("row_cat", DataType::Utf8, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0])),
            Arc::new(StringArray::from(vec!["a", "b"])),
            Arc::new(StringArray::from(vec!["r1", "r2"])),
        ],
    )
    .unwrap()
}

/// Route 3: `FacetSpec.sort`, reached via `ChartSpec(facet=dict)`'s
/// `pyo3_serde::from_py` path -- which, like `from_json`, deserializes
/// `FacetSpec` through its plain derived `Deserialize` with no validation
/// step of its own. Exercised here via the same underlying
/// `serde_json::from_str::<ChartSpec>` deserialization `from_json` uses
/// (both routes deserialize the identical `FacetSpec` type through the
/// identical derive; there is no second `FacetSpec` parser to diverge).
#[test]
fn refuse_malformed_sort_specs_facet_sort_via_wire_dict() {
    pyo3::Python::initialize();
    let json = r#"{"mark":"point","encoding":{"x":{"field":"x"},"y":{"field":"y"}},
            "facet":{"field":"col_cat","mode":{"kind":"wrap","ncols":2},"sort":{"op":"mean"}}}"#;
    let spec: ChartSpec = serde_json::from_str(json).expect("malformed facet sort still deserializes");
    let err = prepare_with(&spec, &facet_grid_batch())
        .expect_err("a malformed facet.sort must refuse");
    assert_eq!(
        err.to_string(),
        "facet.sort: sort={\"op\":\"mean\"} is malformed (sort object missing string \"field\")"
    );
}

/// A `null` `"op"` (JSON `{"field": "v", "op": null}`) is its own
/// malformed shape, distinct from a missing key: `serde_json::Map::get`
/// returns `Some(Value::Null)` for an explicit `null`, which is neither
/// the `None` (absent) nor `Some(Value::String(_))` arms
/// `parse_sort_field_object` treats as valid, so it refuses too. Exercised
/// via the facet door specifically because Python's own construction-time
/// gate has a known, separate gap for exactly this shape
/// (`validate_optional_choice` treats an explicit `op=None` as unset) --
/// this render-time check is the backstop that still refuses it rather
/// than silently rendering the wrong panel order.
#[test]
fn refuse_malformed_sort_specs_facet_sort_null_op_refuses() {
    pyo3::Python::initialize();
    let json = r#"{"mark":"point","encoding":{"x":{"field":"x"},"y":{"field":"y"}},
            "facet":{"field":"col_cat","mode":{"kind":"wrap","ncols":2},"sort":{"field":"y","op":null}}}"#;
    let spec: ChartSpec = serde_json::from_str(json).expect("a null op still deserializes");
    let err = prepare_with(&spec, &facet_grid_batch())
        .expect_err("a null op must refuse, not silently render in insertion order");
    assert_eq!(
        err.to_string(),
        "facet.sort: sort={\"field\":\"y\",\"op\":null} is malformed (sort \"op\" must be \
         a string, got null)"
    );
}

/// Route 4: `FacetSpec.row_sort` set while `FacetSpec.row` is absent
/// (wrap mode) -- a structurally distinct malformation from an invalid
/// sort-field object (the value here, `"descending"`, is itself
/// well-formed): there is no row dimension for it to order. Mirrors
/// `row_title`'s documented wrap-mode no-op scoping, EXCEPT this refuses
/// rather than silently doing nothing, since an ordering the caller
/// declared and that can never apply is malformed input.
#[test]
fn refuse_malformed_sort_specs_facet_row_sort_without_row() {
    pyo3::Python::initialize();
    let json = r#"{"mark":"point","encoding":{"x":{"field":"x"},"y":{"field":"y"}},
            "facet":{"field":"col_cat","mode":{"kind":"wrap","ncols":2},"row_sort":"descending"}}"#;
    let spec: ChartSpec = serde_json::from_str(json).expect("row_sort without row still deserializes");
    let err = prepare_with(&spec, &facet_grid_batch())
        .expect_err("row_sort with no row dimension must refuse");
    assert_eq!(
        err.to_string(),
        "facet.row_sort: sort=\"descending\" is malformed (the facet has no row dimension \
         (row=) for row_sort to order)"
    );
}

/// Route 5, the valid control: a well-formed sort at every location this
/// check walks -- a chart channel, a layer channel, `facet.sort`, and
/// `facet.row_sort` (with `row` set, in grid mode) -- prepares without
/// error. Proves the check is not simply refusing every sort it sees.
#[test]
fn refuse_malformed_sort_specs_valid_forms_at_every_location_prepare() {
    pyo3::Python::initialize();
    use crate::spec::layer::Layer;
    let mut spec = ChartSpec {
        data: crate::spec::data_ref::DataRef::default(),
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec {
                field: "y".into(),
                type_: None,
                sort: Some(serde_json::json!("ascending")),
                ..Default::default()
            }),
            ..Default::default()
        },
        facet: Some(crate::layout::FacetSpec {
            field: "col_cat".into(),
            row: Some("row_cat".into()),
            title: None,
            row_title: None,
            sort: Some(serde_json::json!({"field": "y", "op": "mean"})),
            row_sort: Some(serde_json::json!("descending")),
            repeat: Vec::new(),
            mode: crate::layout::FacetMode::Grid { nrows: 2, ncols: 2 },
            spacing: None,
            resolve: crate::layout::facet::FacetResolve::default(),
        }),
        layers: Some(vec![Layer {
            mark: Mark::Point,
            encoding: Encoding {
                y: Some(EncodingSpec {
                    field: "y".into(),
                    sort: Some(serde_json::json!(["b", "a"])),
                    ..Default::default()
                }),
                ..Default::default()
            },
            inherited_positional: vec![],
            inherited_channels: vec![],
            transforms: vec![],
            mark_style: None,
            data_source: None,
            position: None,
            name: None,
            independent_y: false,
            reads_chart_data: false,
            coord: None,
            entry: None,
        }]),
        ..Default::default()
    };
    assert!(
        prepare_with(&spec, &facet_grid_batch()).is_ok(),
        "well-formed sort at every location must not refuse"
    );
    // A sanity check that the fixture was not vacuously well-formed by
    // never actually reaching the check: flipping one location back to
    // malformed still refuses.
    spec.facet.as_mut().unwrap().sort = Some(serde_json::json!({"op": "mean"}));
    assert!(prepare_with(&spec, &facet_grid_batch()).is_err());
}

/// The predicate itself, directly: `SortSpecIgnored` (positional/color
/// axis sort, re-derived per panel by `resolve_panel_scales`) IS treated
/// as a provisional-pass duplicate once a panel resolves;
/// `FacetSortSpecIgnored` (the facet partition's own sort, resolved
/// exactly once, before any panel loop) is NOT -- collapsing the two
/// would silently drop the facet warning on every render with a
/// non-empty panel, which is what
/// `facet_sort_unresolvable_well_formed_object_warns_and_survives_the_panel_dedup`
/// (below) proves end-to-end; this pins the same fact at the predicate.
#[test]
fn provisional_scale_warning_superseded_by_panel_pass_excludes_the_facet_variant() {
    assert!(provisional_scale_warning_superseded_by_panel_pass(
        &RenderWarning::SortSpecIgnored { reason: "x".into() }
    ));
    assert!(!provisional_scale_warning_superseded_by_panel_pass(
        &RenderWarning::FacetSortSpecIgnored { reason: "x".into() }
    ));
}

/// `#120`(7) + N2: every `RenderWarning` `scale_resolve::auxiliary`
/// pushes is a provisional-pass duplicate too, alongside the color-scheme
/// and sort members above — see the predicate's doc for why these two,
/// plus `ColorPaletteOverflowed`, are whole-chart facts at the
/// provisional/panel-real boundary despite looking data-dependent.
#[test]
fn provisional_scale_warning_superseded_by_panel_pass_includes_every_auxiliary_warning() {
    assert!(provisional_scale_warning_superseded_by_panel_pass(
        &RenderWarning::ShapePaletteOverflowed { categories: 9 }
    ));
    assert!(provisional_scale_warning_superseded_by_panel_pass(
        &RenderWarning::StrokeDashPaletteOverflowed { categories: 9 }
    ));
    assert!(provisional_scale_warning_superseded_by_panel_pass(
        &RenderWarning::ColorPaletteOverflowed { categories: 12 }
    ));
}

/// `#120`(7) + N2, faceted extension: the panel-loop predicate includes
/// exactly the two auxiliary variants, and — the asymmetry the whole
/// mechanism exists to preserve — excludes `ColorPaletteOverflowed`,
/// whose own per-panel copies must survive.
#[test]
fn panel_pass_warning_restated_by_every_panel_excludes_color() {
    assert!(panel_pass_warning_restated_by_every_panel(
        &RenderWarning::ShapePaletteOverflowed { categories: 9 }
    ));
    assert!(panel_pass_warning_restated_by_every_panel(
        &RenderWarning::StrokeDashPaletteOverflowed { categories: 9 }
    ));
    assert!(!panel_pass_warning_restated_by_every_panel(
        &RenderWarning::ColorPaletteOverflowed { categories: 12 }
    ));
}

/// `collapse_panel_restated_duplicates` keeps the first copy of a
/// restated whole-chart fact and drops the rest, keeps every copy of an
/// excluded variant (`ColorPaletteOverflowed`), and leaves every other
/// warning's relative order untouched.
#[test]
fn collapse_panel_restated_duplicates_keeps_first_copy_and_every_color_copy() {
    let opacity = RenderWarning::ShapePaletteOverflowed { categories: 9 };
    let color = RenderWarning::ColorPaletteOverflowed { categories: 12 };
    let sort = RenderWarning::SortSpecIgnored { reason: "x".into() };
    let input = vec![
        opacity.clone(),
        color.clone(),
        sort.clone(),
        opacity.clone(),
        color.clone(),
        opacity.clone(),
        color.clone(),
    ];
    let collapsed = collapse_panel_restated_duplicates(input);
    assert_eq!(
        collapsed,
        vec![opacity, color.clone(), sort, color.clone(), color],
        "one opacity copy survives (the first), every color copy survives, order is preserved"
    );
}

/// A well-formed sort the facet partition cannot resolve (here: a
/// sort-field object naming a field absent from the batch) warns exactly
/// as `X(sort=)` does on an ordinary axis, with `FacetSortSpecIgnored`
/// rather than `SortSpecIgnored` -- and, unlike the axis case, it must
/// SURVIVE `scene_build::build_scene`'s provisional-warning dedup step
/// even though this chart's panels resolve (the bug this variant split
/// exists to fix: `SortSpecIgnored` would have been stripped as an
/// assumed panel-real duplicate that a facet's one-shot partition never
/// actually produces).
#[test]
fn facet_sort_unresolvable_well_formed_object_warns_and_survives_the_panel_dedup() {
    let mut spec = spec_color_facet();
    spec.facet.as_mut().unwrap().sort = Some(serde_json::json!({"field": "does_not_exist"}));
    let out = crate::render::render_svg(
        &spec,
        &batch3(),
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport { width: 600.0, height: 400.0 },
        &crate::render::config::RenderConfig::default(),
        &crate::render::chart_config::ChartConfig::default(),
    )
    .expect("the fixture renders with at least one non-empty panel");
    let facet_sort_warnings: Vec<&RenderWarning> = out
        .warnings
        .iter()
        .filter(|w| matches!(w, RenderWarning::FacetSortSpecIgnored { .. }))
        .collect();
    assert_eq!(
        facet_sort_warnings.len(),
        1,
        "one unresolvable facet sort is one fact, and it must survive: {:?}",
        out.warnings
    );
    assert_eq!(
        facet_sort_warnings[0].to_string(),
        "facet sort spec could not be applied (cannot read sort value field \
         \"does_not_exist\": unknown column 'does_not_exist' referenced by an encoding); \
         panels fall back to insertion order"
    );
    // The plain axis variant never fires for this fact -- a regression
    // that silently widened `provisional_scale_warning_superseded_by_panel_pass`
    // to also match `FacetSortSpecIgnored` would still pass a `.len() == 1`
    // assertion on the wrong variant, so check the type explicitly too.
    assert!(
        out.warnings.iter().all(|w| !matches!(w, RenderWarning::SortSpecIgnored { .. })),
        "a facet-partition failure must never produce the axis variant: {:?}",
        out.warnings
    );
}

#[test]
fn unknown_data_source_raises_clear_error() {
    pyo3::Python::initialize();
    use crate::spec::layer::Layer;
    // Pipeline publishes "step1"; layer asks for "missing".
    let mut spec = spec_with_one_bin(Some("step1".into()));
    spec.layers = Some(vec![Layer {
        mark: Mark::Point,
        encoding: Encoding::default(),
        data_source: Some("missing".into()),
        ..Default::default()
    }]);
    let batch = price_weight_batch();
    let err = prepare_render_inputs(
        &spec,
        &batch,
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport {
            width: 600.0,
            height: 400.0,
        },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("missing"),
        "error must name the bogus key: {msg}"
    );
    // Available keys list must mention either the named transform or the sentinel.
    assert!(
        msg.contains("step1") || msg.contains("__final__"),
        "error must list available keys: {msg}"
    );
}

#[test]
fn prepare_coord_flip_swaps_x_y_in_each_layer() {
    use crate::spec::coord::CoordKind;
    let mut spec = single_layer_spec(); // x="price", y="weight"
    spec.coord = Some(CoordKind::Flip);
    let batch = price_weight_batch();
    let prepared = prepare_render_inputs(
        &spec,
        &batch,
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport {
            width: 600.0,
            height: 400.0,
        },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .unwrap();
    assert!(prepared.coord_flipped);
    // After flip: x should have "weight", y should have "price"
    assert_eq!(
        prepared.layers[0].encoding.x.as_ref().unwrap().field,
        "weight",
        "CoordFlip should swap x←weight (was y)"
    );
    assert_eq!(
        prepared.layers[0].encoding.y.as_ref().unwrap().field,
        "price",
        "CoordFlip should swap y←price (was x)"
    );
    // Axes titles should also reflect the flip
    assert_eq!(prepared.axes.x.title.get().as_deref(), Some("weight"));
    assert_eq!(prepared.axes.y.title.get().as_deref(), Some("price"));
}

// --- D3 (flexibility campaign): per-channel Axis(label_format, tick_count) ---

/// Build an `EncodingSpec` carrying a per-channel `axis` from a JSON object,
/// deserialized through the typed `AxisStyleSpec` (so the test exercises the
/// real wire path, including `deny_unknown_fields`).
fn enc_with_axis(field: &str, axis: serde_json::Value) -> EncodingSpec {
    let axis: crate::render::chart_config::AxisStyleSpec =
        serde_json::from_value(axis).expect("valid axis style json");
    EncodingSpec {
        field: field.into(),
        axis: Some(Box::new(axis)),
        ..Default::default()
    }
}

#[test]
fn per_channel_label_format_reaches_override() {
    // A per-channel `Axis(label_format=",.0f")` on x must reach the axis
    // `label_format_override` (root-cause fix for the hardcoded `None`).
    let mut spec = single_layer_spec();
    spec.encoding.x = Some(enc_with_axis(
        "price",
        serde_json::json!({ "label_format": ",.0f" }),
    ));
    let batch = price_weight_batch();
    let prep = prepare_render_inputs(
        &spec,
        &batch,
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport {
            width: 600.0,
            height: 400.0,
        },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .unwrap();
    assert_eq!(
        prep.axes.x.overrides.labels.format.get().spec.as_deref(),
        Some(",.0f")
    );
}

#[test]
fn per_channel_axis_styling_reaches_axis_input() {
    // B5: a per-channel `Axis(grid_color=, label_color=, domain_width=)` must
    // land on the AxisInput's per-axis style override fields (which
    // build_axis/build_grid consult with a theme fallback).
    let mut spec = single_layer_spec();
    spec.encoding.x = Some(enc_with_axis(
        "price",
        serde_json::json!({ "grid_color": "#cccccc", "label_color": "#ff00ff", "domain_width": 3.0 }),
    ));
    let batch = price_weight_batch();
    let prep = prepare_render_inputs(
        &spec,
        &batch,
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport {
            width: 600.0,
            height: 400.0,
        },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .unwrap();
    let gx = prep
        .axes
        .x
        .overrides
        .grid
        .color
        .expect("per-channel grid_color must reach x AxisInput");
    assert_eq!([gx.red, gx.green, gx.blue], [0xcc, 0xcc, 0xcc]);
    let lx = prep
        .axes
        .x
        .overrides
        .labels
        .color
        .expect("per-channel label_color must reach x AxisInput");
    assert_eq!([lx.red, lx.green, lx.blue], [0xff, 0x00, 0xff]);
    assert_eq!(prep.axes.x.overrides.domain.width, Some(3.0));
    // The y-axis must be untouched (per-channel applies only to its own axis).
    assert!(prep.axes.y.overrides.grid.color.is_none());
    assert!(prep.axes.y.overrides.labels.color.is_none());
}

// ── B5 unit 2: orphan positioning / tick fields ─────────────────────────

#[test]
fn per_channel_orient_reaches_axis_input() {
    // `Axis(orient="top")` on x must set the x AxisInput's orient to Top.
    let mut spec = single_layer_spec();
    spec.encoding.x = Some(enc_with_axis(
        "price",
        serde_json::json!({ "orient": "top" }),
    ));
    let batch = price_weight_batch();
    let prep = prepare_render_inputs(
        &spec,
        &batch,
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport {
            width: 600.0,
            height: 400.0,
        },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .unwrap();
    assert_eq!(prep.axes.x.orient, AxisOrient::Top);
    // y untouched (default Left).
    assert_eq!(prep.axes.y.orient, AxisOrient::Left);
}

#[test]
fn cross_dimension_orient_fails_loud() {
    // `Axis(orient="left")` on the x channel is a cross-dimension error and
    // must surface a RenderError rather than silently dropping.
    let mut spec = single_layer_spec();
    spec.encoding.x = Some(enc_with_axis(
        "price",
        serde_json::json!({ "orient": "left" }),
    ));
    let batch = price_weight_batch();
    let err = prepare_render_inputs(
        &spec,
        &batch,
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport {
            width: 600.0,
            height: 400.0,
        },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .expect_err("x orient='left' must fail loud");
    match err {
        crate::render::RenderError::InvalidAxisOrient {
            channel,
            orient,
            coord_flipped,
        } => {
            assert_eq!(channel, "x");
            assert_eq!(orient, "left");
            // Unflipped chart: `with_coord_flipped` boundary correction is a no-op.
            assert!(!coord_flipped);
        }
        other => panic!("expected InvalidAxisOrient, got {other:?}"),
    }
}

/// R3: under `CoordFlip`, a per-channel `Axis(orient=...)` cross-dimension
/// error must name the axis the user actually configured, not the resolved
/// (post-flip) one it landed on. The user writes `Axis(orient="left")` on
/// `y` — valid for y pre-flip — but `build_layers` swaps this whole
/// `EncodingSpec` into the x slot, where `"left"` is invalid, so the error
/// fires for the RESOLVED `"x"` channel. The message must still say `y`.
#[test]
fn cross_dimension_orient_fails_loud_names_users_channel_under_flip() {
    use crate::spec::coord::CoordKind;
    let mut spec = single_layer_spec();
    spec.coord = Some(CoordKind::Flip);
    spec.encoding.y = Some(enc_with_axis(
        "weight",
        serde_json::json!({ "orient": "left" }),
    ));
    let batch = price_weight_batch();
    let err = prepare_render_inputs(
        &spec,
        &batch,
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport {
            width: 600.0,
            height: 400.0,
        },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .expect_err("y orient='left', swapped onto x, must fail loud");
    match err {
        crate::render::RenderError::InvalidAxisOrient {
            channel,
            orient,
            coord_flipped,
        } => {
            // Resolved/internal token: the object landed on x post-swap.
            assert_eq!(channel, "x");
            assert_eq!(orient, "left");
            assert!(coord_flipped);
        }
        other => panic!("expected InvalidAxisOrient, got {other:?}"),
    }
    let text = format!(
        "{}",
        prepare_render_inputs(
            &spec,
            &batch,
            &crate::layout::ThemeInputs::default(),
            crate::layout::Viewport {
                width: 600.0,
                height: 400.0
            },
            &crate::render::chart_config::ChartConfig::default(),
            None
        )
        .unwrap_err()
    );
    assert_eq!(
        text,
        "axis orient 'left' is invalid for the y axis (expected 'top' or 'bottom' — under \
         CoordFlip, y renders as the horizontal axis)",
        "message must name the channel the user wrote ('y'), not the resolved one ('x'), and \
         must explain why 'top'/'bottom' applies to something named 'y'"
    );
}

// The chart-level `configure_axis(orient=...)` chain is a documented R3
// EXEMPTION (deliberately NOT patched via `with_coord_flipped`) — see
// `chart_level_orient_error_names_resolved_axis_under_flip` in
// `render::mod`'s test module — it needs `prepare_and_layout`, which is
// private to that module.

#[test]
fn per_channel_orphan_positioning_fields_reach_axis_input() {
    let mut spec = single_layer_spec();
    spec.encoding.x = Some(enc_with_axis(
        "price",
        serde_json::json!({
            "translate": 12.0,
            "min_band": 70.0,
            "max_band": 120.0,
            "grid_opacity": 0.25,
            "title_orient": "bottom",
            "zindex": 1
        }),
    ));
    let batch = price_weight_batch();
    let prep = prepare_render_inputs(
        &spec,
        &batch,
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport {
            width: 600.0,
            height: 400.0,
        },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .unwrap();
    assert_eq!(prep.axes.x.overrides.placement.translate, Some(12.0));
    assert_eq!(prep.axes.x.overrides.placement.min_band, Some(70.0));
    assert_eq!(prep.axes.x.overrides.placement.max_band, Some(120.0));
    assert_eq!(prep.axes.x.overrides.grid.opacity, Some(0.25));
    assert_eq!(prep.axes.x.overrides.title.orient, Some(AxisOrient::Bottom));
    assert_eq!(prep.axes.x.overrides.placement.zindex, Some(1));
}

#[test]
fn tick_min_step_thins_dense_ticks() {
    // A continuous x-axis with tick_min_step set must drop ticks closer than
    // the step in data space, leaving fewer labels.
    let mut spec = single_layer_spec();
    let batch = price_weight_batch();
    let baseline = prepare_render_inputs(
        &spec,
        &batch,
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport {
            width: 600.0,
            height: 400.0,
        },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .unwrap();
    let base_count = baseline.axes.x.tick_labels.len();

    // Pick a min_step larger than the natural tick spacing to force thinning.
    spec.encoding.x = Some(enc_with_axis(
        "price",
        serde_json::json!({ "tick_min_step": 1e9 }),
    ));
    let mut prep = prepare_render_inputs(
        &spec,
        &batch,
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport {
            width: 600.0,
            height: 400.0,
        },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .unwrap();
    // The adjustment runs in `config_apply::resync_ticks_after_axis_merge`;
    // invoke it directly here.
    let tc = prep.x_tick_count;
    adjust_axis_ticks(&mut prep.axes.x, &prep.provisional_scales.x, tc, false);
    assert!(
        prep.axes.x.tick_labels.len() < base_count,
        "tick_min_step=1e9 must thin ticks: base={base_count}, after={}",
        prep.axes.x.tick_labels.len()
    );
    // At least one tick survives.
    assert!(!prep.axes.x.tick_labels.is_empty());
    // Labels and projected fractions stay index-aligned.
    if let Some(proj) = prep.axes.x.tick_projection.as_ref() {
        assert_eq!(proj.major.len(), prep.axes.x.tick_labels.len());
    }
}

/// Quality-review S3 fix (2026-09-03): a chart-level TIME format must
/// survive `tick_min_step` thinning, not silently revert to the DEFAULT
/// spacing-keyed granularity. Root cause: `render::config_apply::apply_label_format_to_axis`
/// (which runs AFTER this fn) re-derives raw temporal values via
/// `scale.temporal_tick_values(tick_count)` — an UN-thinned, full-length
/// vec — so its `values.len() == axis.tick_labels.len()` guard fails once
/// thinning has shrunk `tick_labels`, and it silently falls through to a
/// no-op. Fixed by finalizing the temporal reformat HERE, using the
/// SAME correctly-thinned `raw` this fn already computes in lockstep
/// with the just-retained `tick_labels`.
#[test]
fn tick_min_step_thinning_reformats_temporal_labels_instead_of_dropping_the_format() {
    use crate::render::scale_resolve::ScaleKind;
    use crate::scale::time::TimeScale;

    let epoch_lo = 1_514_764_800_000.0; // 2018-01-01
    let epoch_hi = epoch_lo + 4.0 * 365.0 * 86_400_000.0; // ~2022-01-01
    let scale = ScaleKind::Time(TimeScale::new_internal(
        vec![epoch_lo, epoch_hi],
        vec![0.0, 1.0],
        false,
        false,
    ));
    let tick_count = 8;
    let default_labels = scale.tick_labels(tick_count);
    let n = default_labels.len();
    assert!(n > 2, "fixture must produce enough ticks to actually thin");

    let mut axis = crate::layout::AxisInput::new(
        crate::layout::AxisOrient::Bottom,
        None,
        default_labels,
        None,
    );
    axis.tick_projection = Some(crate::layout::TickProjection {
        padding_frac: scale.padding_fraction(),
        major: scale.tick_fractions(tick_count),
        minor: Vec::new(),
    });
    // A chart-level `configure_axis(label_format="date_iso", tick_min_step=...)`
    // shape: both fields are chart-level-only-owned (see
    // `LabelFormat::format_type`'s doc), set here directly to
    // isolate `adjust_axis_ticks`'s own behavior.
    // The reviewer's own repro value: 1.5 years, forcing roughly half of
    // the ~yearly-spaced default ticks to drop.
    axis.overrides.ticks.min_step = Some(1.5 * 365.0 * 86_400_000.0);
    axis.overrides
        .labels
        .format
        .set(crate::layout::LabelFormat {
            spec: Some("%Y-%m-%d".to_string()),
            format_type: Some("time".to_string()),
        });

    adjust_axis_ticks(&mut axis, &scale, tick_count, false);

    assert!(
        axis.tick_labels.len() < n,
        "tick_min_step must thin: before={n}, after={}",
        axis.tick_labels.len()
    );
    assert!(!axis.tick_labels.is_empty());
    for label in &axis.tick_labels {
        // The DEFAULT (year-granularity) label shape at this span is a
        // bare "2018"/"2020"-style year — an ISO "YYYY-MM-DD" shape can
        // only come from the real reformat, never a coincidence.
        assert_eq!(
            label.len(),
            10,
            "expected an ISO date after thinning, got {label:?}"
        );
        assert!(
            label.starts_with("20") && label.contains('-'),
            "expected an ISO date, got {label:?}"
        );
    }
}

#[test]
fn tick_extra_appends_domain_boundaries() {
    // tick_extra must add a tick at each domain boundary not already present.
    // Use awkward bounds (3.0 .. 97.0) so the scale's nice ticks (0,20,40,…)
    // do NOT already include the exact boundaries, forcing an actual append.
    let schema = Arc::new(Schema::new(vec![
        Field::new("price", DataType::Float64, false),
        Field::new("weight", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![3.0, 50.0, 97.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
        ],
    )
    .unwrap();
    let mut spec = single_layer_spec();
    spec.encoding.x = Some(enc_with_axis(
        "price",
        serde_json::json!({ "tick_extra": true }),
    ));
    let mut prep = prepare_render_inputs(
        &spec,
        &batch,
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport {
            width: 600.0,
            height: 400.0,
        },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .unwrap();
    let before = prep.axes.x.tick_labels.len();
    let tc = prep.x_tick_count;
    adjust_axis_ticks(&mut prep.axes.x, &prep.provisional_scales.x, tc, false);
    let after = prep.axes.x.tick_labels.len();
    assert!(
        after > before,
        "tick_extra must append boundary ticks: {before} -> {after}"
    );
    // Labels and projected fractions stay index-aligned.
    let proj = prep.axes.x.tick_projection.as_ref().unwrap();
    assert_eq!(
        proj.major.len(),
        prep.axes.x.tick_labels.len(),
        "labels/fractions aligned"
    );
    // The exact domain-boundary labels ("3" and "97") are present.
    assert!(
        prep.axes.x.tick_labels.iter().any(|l| l == "3"),
        "domain min tick present"
    );
    assert!(
        prep.axes.x.tick_labels.iter().any(|l| l == "97"),
        "domain max tick present"
    );
}

#[test]
fn per_channel_axis_style_wins_over_chart_config() {
    // B5 cascade: a per-channel `Axis(grid_width=...)` on x, applied in
    // `prepare_render_inputs`, is set BEFORE the chart-level
    // `configure_axis(grid_width=...)` fill (which only fills `None`), so the
    // per-channel value wins. We assert the prep-stage value here, then verify
    // the chart-level apply does not overwrite it.
    let mut spec = single_layer_spec();
    spec.encoding.x = Some(enc_with_axis(
        "price",
        serde_json::json!({ "grid_width": 4.0 }),
    ));
    let batch = price_weight_batch();
    let mut prep = prepare_render_inputs(
        &spec,
        &batch,
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport {
            width: 600.0,
            height: 400.0,
        },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .unwrap();
    assert_eq!(
        prep.axes.x.overrides.grid.width,
        Some(4.0),
        "per-channel value set in prep"
    );
    // Chart-level configure_axis(grid_width=1.0) must NOT overwrite per-channel.
    let cfg = crate::render::chart_config::AxisConfigSpec {
        style: crate::render::chart_config::AxisStyleSpec {
            grid_width: Some(1.0),
            ..Default::default()
        },
        ..Default::default()
    };
    crate::render::config_apply::apply_axis_style_to_axis_input(&mut prep.axes.x, &cfg.style)
        .unwrap();
    assert_eq!(
        prep.axes.x.overrides.grid.width,
        Some(4.0),
        "per-channel must win over configure_axis"
    );
}

#[test]
fn per_channel_show_toggle_wins_over_chart_config() {
    // B5 Unit-1 fix: a per-channel `Axis(grid=False)`/`Axis(domain=False)` on x
    // must survive a conflicting chart-level `configure_axis(grid=True)` /
    // `configure_axis(domain=True)`. The per-channel prepare path is the sole
    // owner of `AxisInput.show_*`; the chart-level toggle flows through its
    // global theme/gate path, not by clobbering the per-axis `show_*` value.
    let mut spec = single_layer_spec();
    spec.encoding.x = Some(enc_with_axis(
        "price",
        serde_json::json!({ "grid": false, "domain": false }),
    ));
    let batch = price_weight_batch();
    let mut prep = prepare_render_inputs(
        &spec,
        &batch,
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport {
            width: 600.0,
            height: 400.0,
        },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .unwrap();
    // Probing `prep.axes.{x,y}` HERE, before `AxesInput::apply_show_defaults`
    // has run (that fold happens later, right before layout, in real
    // production), is the point of this assertion — so read the raw
    // override slot with the same `unwrap_or(true)` fallback `show_grid()`/
    // `show_domain()` apply, rather than through those accessors, which
    // now assert this precondition in debug builds.
    assert!(
        !prep.axes.x.overrides.grid.show.unwrap_or(true),
        "per-channel grid=False set in prep"
    );
    assert!(
        !prep.axes.x.overrides.domain.show.unwrap_or(true),
        "per-channel domain=False set in prep"
    );
    assert!(
        prep.axes.y.overrides.grid.show.unwrap_or(true),
        "y grid default true"
    );
    assert!(
        prep.axes.y.overrides.domain.show.unwrap_or(true),
        "y domain default true"
    );

    // Chart-level configure_axis(grid=True, domain=True) applies to BOTH axes
    // via the shared `axis` key, the same wire step
    // `config_apply::fill_axis_slots_specific_before_shared` runs.
    let cfg = crate::render::chart_config::AxisConfigSpec {
        style: crate::render::chart_config::AxisStyleSpec {
            grid: Some(true),
            domain: Some(true),
            ..Default::default()
        },
        ..Default::default()
    };
    crate::render::config_apply::apply_axis_config_to_axis_input(&mut prep.axes.x, Some(&cfg))
        .unwrap();
    crate::render::config_apply::apply_axis_config_to_axis_input(&mut prep.axes.y, Some(&cfg))
        .unwrap();

    // Per-channel x suppression survives the conflicting chart-level toggle.
    // Same pre-`apply_show_defaults` raw-slot read as above.
    assert!(
        !prep.axes.x.overrides.grid.show.unwrap_or(true),
        "per-channel grid=False must survive configure_axis(grid=True)"
    );
    assert!(
        !prep.axes.x.overrides.domain.show.unwrap_or(true),
        "per-channel domain=False must survive configure_axis(domain=True)"
    );
    // The other axis (no per-channel override) is unaffected: still shown.
    assert!(
        prep.axes.y.overrides.grid.show.unwrap_or(true),
        "y axis show_grid unaffected"
    );
    assert!(
        prep.axes.y.overrides.domain.show.unwrap_or(true),
        "y axis show_domain unaffected"
    );
}

#[test]
fn per_channel_label_format_formats_after_central_apply() {
    // End-to-end through render::mod's central application: the numeric spec
    // must actually reformat the labels. We replicate the override-apply step
    // that `render_pipeline` performs.
    let mut spec = single_layer_spec();
    spec.encoding.x = Some(enc_with_axis(
        "price",
        serde_json::json!({ "label_format": "$,.2f" }),
    ));
    let batch = price_weight_batch();
    let mut prep = prepare_render_inputs(
        &spec,
        &batch,
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport {
            width: 600.0,
            height: 400.0,
        },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .unwrap();
    // Apply the threaded override exactly as render/mod.rs does.
    prep.axes.x.tick_labels = apply_tick_format(
        std::mem::take(&mut prep.axes.x.tick_labels),
        prep.axes.x.overrides.labels.format.get().spec.as_deref(),
        None,
    );
    // Every numeric label should now carry the "$" prefix and 2 decimals.
    assert!(
        prep.axes
            .x
            .tick_labels
            .iter()
            .all(|l| l.starts_with('$') && l.contains('.')),
        "got: {:?}",
        prep.axes.x.tick_labels
    );
}

#[test]
fn per_channel_tick_count_limits_numeric_ticks() {
    // Default tick generation yields several ticks over [10, 30]; with
    // `tick_count=2` the axis should produce noticeably fewer labels.
    let batch = price_weight_batch();

    let mut default_spec = single_layer_spec();
    default_spec.encoding.x = Some(EncodingSpec {
        field: "price".into(),
        ..Default::default()
    });
    let default_prep = prepare_render_inputs(
        &default_spec,
        &batch,
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport {
            width: 600.0,
            height: 400.0,
        },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .unwrap();

    let mut limited_spec = single_layer_spec();
    limited_spec.encoding.x = Some(enc_with_axis(
        "price",
        serde_json::json!({ "tick_count": 2 }),
    ));
    let limited_prep = prepare_render_inputs(
        &limited_spec,
        &batch,
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport {
            width: 600.0,
            height: 400.0,
        },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .unwrap();

    assert!(
        limited_prep.axes.x.tick_labels.len() < default_prep.axes.x.tick_labels.len(),
        "tick_count=2 should produce fewer labels than default; limited={:?} default={:?}",
        limited_prep.axes.x.tick_labels,
        default_prep.axes.x.tick_labels
    );
}

// --- F8: colorbar tick formatting via full d3 grammar ---

/// Build a spec with a continuous color encoding on "pop" so that the colorbar
/// code path activates. Returns a `ColorbarInput` with `format_spec` applied.
fn colorbar_with_format(format_spec: &str) -> crate::layout::ColorbarInput {
    use crate::spec::encoding::DataType;
    let mut spec = base_spec();
    spec.encoding.color = Some(EncodingSpec {
        field: "pop".into(),
        type_: Some(DataType::Quantitative),
        legend: Some(Box::new(crate::render::chart_config::LegendStyleSpec {
            format: Some(format_spec.to_owned()),
            ..Default::default()
        })),
        ..Default::default()
    });
    let p = prep(&spec, &batch_pop());
    p.colorbar
        .expect("continuous color encoding must produce a colorbar")
}

/// (a) Byte-stability: `.2f` colorbar output must match today's expected strings.
///
/// Domain [10, 100]; 5 ticks at t=0,0.25,0.5,0.75,1.0 → v=10,32.5,55,77.5,100.
/// `.2f` → fixed 2 decimal places.
#[test]
fn colorbar_format_dot2f_byte_stable() {
    let colorbar = colorbar_with_format(".2f");
    assert_eq!(
        colorbar.tick_labels,
        vec!["10.00", "32.50", "55.00", "77.50", "100.00"],
        "`.2f` colorbar tick labels must match fixed-2-decimal output"
    );
}

/// (a) Byte-stability: `.0%` colorbar output must match today's expected strings.
///
/// Domain [0.0, 1.0] (use batch_pop() but override to a [0,1] range by using
/// a spec whose pop values happen to span [10,100]; for a cleaner [0,1] domain
/// we build a dedicated batch).
#[test]
fn colorbar_format_dot0pct_byte_stable() {
    use crate::spec::encoding::DataType;
    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType as ArrowDt, Field, Schema};

    // Build a batch with "pop" in [0.0, 1.0] so the colorbar domain is [0,1].
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ArrowDt::Float64, false),
        Field::new("y", ArrowDt::Float64, false),
        Field::new("pop", ArrowDt::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0])),
        ],
    )
    .unwrap();

    let mut spec = base_spec();
    spec.encoding.color = Some(EncodingSpec {
        field: "pop".into(),
        type_: Some(DataType::Quantitative),
        legend: Some(Box::new(crate::render::chart_config::LegendStyleSpec {
            format: Some(".0%".to_owned()),
            ..Default::default()
        })),
        ..Default::default()
    });
    let p = prep(&spec, &batch);
    let colorbar = p
        .colorbar
        .expect("continuous color encoding must produce a colorbar");

    // 5 ticks at t=0,0.25,0.5,0.75,1.0 → v=0,0.25,0.5,0.75,1.0
    // `.0%` → multiply by 100, 0 decimals, append `%` → "0%","25%","50%","75%","100%"
    assert_eq!(
        colorbar.tick_labels,
        vec!["0%", "25%", "50%", "75%", "100%"],
        "`.0%` colorbar tick labels must match percent-0-decimal output"
    );
}

/// (b) Capability widening: `,` (grouped thousands) now formats via the full
///     d3 grammar rather than falling through to the auto-precision path.
///
/// Domain [10, 100] from batch_pop(); 5 ticks → 10, 32.5, 55, 77.5, 100.
/// With `,` (no type) these are integer-valued or near-integer; the d3 grammar
/// groups integer-valued inputs as plain integers: "10", "33", "55", "78", "100"
/// (or with grouping applied, which only triggers at ≥4 digits). What matters is
/// that the output does NOT look like the old fallback (auto-precision float).
#[test]
fn colorbar_format_comma_grouped_widened() {
    // Domain [1000, 100000] so we can see grouping separators.
    use crate::spec::encoding::DataType;
    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType as ArrowDt, Field, Schema};

    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ArrowDt::Float64, false),
        Field::new("y", ArrowDt::Float64, false),
        Field::new("pop", ArrowDt::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(Float64Array::from(vec![1000.0, 100_000.0])),
        ],
    )
    .unwrap();

    let mut spec = base_spec();
    spec.encoding.color = Some(EncodingSpec {
        field: "pop".into(),
        type_: Some(DataType::Quantitative),
        legend: Some(Box::new(crate::render::chart_config::LegendStyleSpec {
            format: Some(",.0f".to_owned()),
            ..Default::default()
        })),
        ..Default::default()
    });
    let p = prep(&spec, &batch);
    let colorbar = p
        .colorbar
        .expect("continuous color encoding must produce a colorbar");

    // 5 ticks at t=0..1 over [1000, 100000]:
    // v = 1000, 25750, 50500, 75250, 100000
    // `,.0f` → integer with grouping: "1,000" "25,750" "50,500" "75,250" "100,000"
    assert!(
        colorbar.tick_labels.iter().any(|l| l.contains(',')),
        "`,` grouped format must produce commas in large-number colorbar ticks; got {:?}",
        colorbar.tick_labels
    );
    // None of the ticks should contain a decimal point (`.0f` rounds to integer).
    assert!(
        colorbar.tick_labels.iter().all(|l| !l.contains('.')),
        "`.0f` format must not produce decimal points; got {:?}",
        colorbar.tick_labels
    );
}

/// (b) Capability widening: `~s` (SI with trim) formats via the full grammar.
///
/// Domain [1000, 1_000_000]; first tick (1000) → "1k", last (1M) → "1M".
#[test]
fn colorbar_format_si_trim_widened() {
    use crate::spec::encoding::DataType;
    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType as ArrowDt, Field, Schema};

    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ArrowDt::Float64, false),
        Field::new("y", ArrowDt::Float64, false),
        Field::new("pop", ArrowDt::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(Float64Array::from(vec![1_000.0, 1_000_000.0])),
        ],
    )
    .unwrap();

    let mut spec = base_spec();
    spec.encoding.color = Some(EncodingSpec {
        field: "pop".into(),
        type_: Some(DataType::Quantitative),
        legend: Some(Box::new(crate::render::chart_config::LegendStyleSpec {
            format: Some("~s".to_owned()),
            ..Default::default()
        })),
        ..Default::default()
    });
    let p = prep(&spec, &batch);
    let colorbar = p
        .colorbar
        .expect("continuous color encoding must produce a colorbar");

    // First tick (v=1000) → "1k", last tick (v=1_000_000) → "1M".
    assert_eq!(
        colorbar.tick_labels.first().map(String::as_str),
        Some("1k"),
        "`~s` must format 1000 as '1k'; got {:?}",
        colorbar.tick_labels
    );
    assert_eq!(
        colorbar.tick_labels.last().map(String::as_str),
        Some("1M"),
        "`~s` must format 1_000_000 as '1M'; got {:?}",
        colorbar.tick_labels
    );
}

// ── #9 [FA-15]: conditional-color legend regression ──────────────────────

/// Build a batch with an `x`, `y`, and `cat` (categorical) column.
fn cat_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
        Field::new("cat", DataType::Utf8, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
            Arc::new(StringArray::from(vec!["a", "b", "a", "c"])),
        ],
    )
    .unwrap()
}

/// A spec with `color = None` and a `Color` conditional using Field{name="cat"} in
/// `if_selected` must produce non-empty `legend_entries` whose labels match the
/// field's distinct values in first-appearance order ("a", "b", "c").
#[test]
fn conditional_color_field_builds_legend_entries() {
    use ferrum_scene::{ChannelName, ConditionalEncoding, EncodingValue};

    let mut spec = base_spec();
    // No base color encoding — this is the conditional-only case.
    spec.encoding.color = None;
    spec.conditionals = vec![ConditionalEncoding {
        selection_name: "sel".into(),
        channel: ChannelName::Color,
        if_selected: EncodingValue::Field { name: "cat".into() },
        if_not: EncodingValue::Color {
            value: ferrum_scene::Color {
                r: 200,
                g: 200,
                b: 200,
                a: 255,
            },
        },
    }.into()];

    let batch = cat_batch();
    let prep = prepare_render_inputs(
        &spec,
        &batch,
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport {
            width: 600.0,
            height: 400.0,
        },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .unwrap();

    assert!(
        !prep.legend_entries.is_empty(),
        "conditional Color Field must produce legend entries when base color is absent"
    );
    let labels: Vec<&str> = prep
        .legend_entries
        .iter()
        .map(|e| e.label.as_str())
        .collect();
    assert_eq!(
        labels,
        vec!["a", "b", "c"],
        "legend entries must match distinct values in first-appearance order"
    );
    // All symbols should be Circle to match the categorical convention.
    for entry in &prep.legend_entries {
        assert_eq!(
            entry.symbol,
            SymbolKind::Circle,
            "conditional-color legend entries must use SymbolKind::Circle"
        );
    }
    // Legend title must be the conditional field name.
    assert_eq!(
        prep.legend_title.as_deref(),
        Some("cat"),
        "legend title must be the conditional color field name"
    );
}

/// A chart WITH a base color encoding must be byte-identical to the pre-fix
/// behavior — the new path must not activate when `provisional_scales.color` is Some.
#[test]
fn base_color_chart_legend_unchanged_by_conditional_fix() {
    // Use the existing `base_spec` + `batch_pop` helpers from the aux-legend tests.
    // Those live in the same mod so they are accessible here.
    let mut spec = base_spec();
    // A categorical base color encoding drives `provisional_scales.color = Some(Categorical)`.
    spec.encoding.color = Some(EncodingSpec {
        field: "region".into(),
        type_: None,
        ..Default::default()
    });
    // Also add a conditional — must NOT affect the base-color legend.
    use ferrum_scene::{ChannelName, ConditionalEncoding, EncodingValue};
    spec.conditionals = vec![ConditionalEncoding {
        selection_name: "sel".into(),
        channel: ChannelName::Color,
        if_selected: EncodingValue::Field {
            name: "region".into(),
        },
        if_not: EncodingValue::Color {
            value: ferrum_scene::Color {
                r: 200,
                g: 200,
                b: 200,
                a: 255,
            },
        },
    }.into()];

    let p = prep(&spec, &batch_pop());
    // The categorical arm (not the conditional arm) should have populated legend_entries.
    assert!(
        !p.legend_entries.is_empty(),
        "base-color chart must still produce legend entries"
    );
    // Entries must equal the domain of the base color field, not vary with the conditional.
    // `batch_pop` has "Asia", "Europe", "Americas" in first-appearance order.
    let labels: Vec<&str> = p.legend_entries.iter().map(|e| e.label.as_str()).collect();
    assert_eq!(
        labels,
        vec!["AS", "EU", "AF"],
        "base-color chart legend entries must not be affected by a conditional"
    );
}

// ── SPINE-03: build_axis_tick_inputs characterization ────────────────────
//
// These pin that the single `build_axis_tick_inputs` helper reproduces the
// OLD open-coded independent-axis sequence (formerly in scene_build.rs:
// tick_labels → non-ordinal-y reverse → apply_tick_format → projection with
// minor: Vec::new()) byte-for-byte. The oracle below is that exact old
// sequence, inlined here so any drift trips the test.

use crate::render::scale_resolve::ScaleKind;
use crate::scale::linear::LinearScale;
use crate::scale::ordinal::OrdinalScale;

fn linear_scale(lo: f64, hi: f64) -> ScaleKind {
    ScaleKind::Linear(LinearScale::new_internal(
        vec![lo, hi],
        vec![0.0, 1.0],
        false,
        false,
    ))
}

/// The old independent-X open-coded sequence (no reverse), used as an oracle.
fn old_independent_x(
    scale: &ScaleKind,
    count: usize,
    fmt: Option<&str>,
    fmt_type: Option<&str>,
) -> (Vec<String>, Option<TickProjection>) {
    let labels = apply_tick_format(scale.tick_labels(count), fmt, fmt_type);
    let proj = if matches!(scale, ScaleKind::Ordinal(_)) {
        None
    } else {
        let fractions = scale.tick_fractions(count);
        (!fractions.is_empty()).then(|| TickProjection {
            padding_frac: scale.padding_fraction(),
            major: fractions,
            minor: Vec::new(),
        })
    };
    (labels, proj)
}

/// The old independent-Y open-coded sequence (non-ordinal reverse), as oracle.
fn old_independent_y(
    scale: &ScaleKind,
    count: usize,
    fmt: Option<&str>,
    fmt_type: Option<&str>,
) -> (Vec<String>, Option<TickProjection>) {
    let mut raw = scale.tick_labels(count);
    if !matches!(scale, ScaleKind::Ordinal(_)) {
        raw.reverse();
    }
    let labels = apply_tick_format(raw, fmt, fmt_type);
    let proj = if matches!(scale, ScaleKind::Ordinal(_)) {
        None
    } else {
        let mut fractions = scale.tick_fractions(count);
        (!fractions.is_empty()).then(|| {
            fractions.reverse();
            TickProjection {
                padding_frac: scale.padding_fraction(),
                major: fractions,
                minor: Vec::new(),
            }
        })
    };
    (labels, proj)
}

/// X (no reverse), no format: build_axis_tick_inputs(Immediate) == oracle.
#[test]
fn build_axis_tick_inputs_x_matches_old_independent_no_format() {
    let scale = linear_scale(0.0, 100.0);
    let (labels, proj, threaded) = build_axis_tick_inputs(
        &scale,
        10,
        TickFormatMode::Immediate {
            format: None,
            format_type: None,
        },
        false,
        Vec::new(),
    )
    .unwrap();
    let (o_labels, o_proj) = old_independent_x(&scale, 10, None, None);
    assert_eq!(labels, o_labels);
    assert_eq!(proj, o_proj);
    assert_eq!(threaded, None, "Immediate mode threads nothing");
}

/// X with a d3 numeric format applied immediately.
#[test]
fn build_axis_tick_inputs_x_matches_old_independent_with_format() {
    let scale = linear_scale(0.0, 1.0);
    let (labels, proj, _t) = build_axis_tick_inputs(
        &scale,
        5,
        TickFormatMode::Immediate {
            format: Some(".0%"),
            format_type: None,
        },
        false,
        Vec::new(),
    )
    .unwrap();
    let (o_labels, o_proj) = old_independent_x(&scale, 5, Some(".0%"), None);
    assert_eq!(labels, o_labels);
    assert_eq!(proj, o_proj);
}

/// Y (non-ordinal reverse) matches the old reversed-label / reversed-fraction
/// sequence in lockstep.
#[test]
fn build_axis_tick_inputs_y_matches_old_independent_reversed() {
    let scale = linear_scale(-50.0, 50.0);
    let (labels, proj, _t) = build_axis_tick_inputs(
        &scale,
        8,
        TickFormatMode::Immediate {
            format: None,
            format_type: None,
        },
        true,
        Vec::new(),
    )
    .unwrap();
    let (o_labels, o_proj) = old_independent_y(&scale, 8, None, None);
    assert_eq!(labels, o_labels);
    assert_eq!(proj, o_proj);
    // Sanity: the y projection's fractions are descending (reversed).
    let major = &proj.unwrap().major;
    assert!(
        major.windows(2).all(|w| w[0] >= w[1]),
        "reversed y fractions must be non-increasing"
    );
}

/// Ordinal scale: no projection on either path, no y-reverse of labels.
#[test]
fn build_axis_tick_inputs_ordinal_has_no_projection() {
    let scale = ScaleKind::Ordinal(OrdinalScale::new_internal(
        vec!["a".into(), "b".into(), "c".into()],
        vec![0.0, 30.0],
        DiscreteLayout::band(0.1, 0.0, DEFAULT_ALIGN),
    ));
    let (x_labels, x_proj, _t) = build_axis_tick_inputs(
        &scale,
        10,
        TickFormatMode::Immediate {
            format: None,
            format_type: None,
        },
        false,
        Vec::new(),
    )
    .unwrap();
    assert!(x_proj.is_none(), "ordinal scale yields no tick projection");
    let (y_labels, y_proj, _t) = build_axis_tick_inputs(
        &scale,
        10,
        TickFormatMode::Immediate {
            format: None,
            format_type: None,
        },
        true,
        Vec::new(),
    )
    .unwrap();
    assert!(y_proj.is_none());
    // Ordinal labels are NOT reversed for y (top-down convention preserved).
    assert_eq!(x_labels, y_labels);
}

/// MOD-09 seam: the shared (`Thread`) and independent (`Immediate`) paths run
/// the SAME `build_axis_tick_inputs` and agree on labels + major fractions for
/// the same scale, while the independent path keeps `minor` EMPTY even when the
/// shared path is handed a non-empty minor vec. This pins the byte-identity
/// invariant the independent-facet goldens rely on: routing both axis paths
/// through one helper must not start emitting per-panel minor ticks.
#[test]
fn build_axis_tick_inputs_independent_keeps_minor_empty_while_shared_carries_it() {
    let scale = linear_scale(0.0, 100.0);

    // Shared/global path: numeric format threads forward (not applied here),
    // and a non-empty minor vec is carried into the projection.
    let shared_minor = vec![0.1_f64, 0.3, 0.7];
    let (shared_labels, shared_proj, shared_threaded) = build_axis_tick_inputs(
        &scale,
        10,
        TickFormatMode::Thread {
            format: None,
            format_type: None,
        },
        false,
        shared_minor.clone(),
    )
    .unwrap();

    // Independent/per-panel path: format applied immediately, minor empty.
    let (indep_labels, indep_proj, indep_threaded) = build_axis_tick_inputs(
        &scale,
        10,
        TickFormatMode::Immediate {
            format: None,
            format_type: None,
        },
        false,
        Vec::new(),
    )
    .unwrap();

    // Same scale, same count, no format on either → identical labels + majors.
    assert_eq!(shared_labels, indep_labels);
    let shared_proj = shared_proj.expect("continuous scale yields a projection");
    let indep_proj = indep_proj.expect("continuous scale yields a projection");
    assert_eq!(
        shared_proj.major, indep_proj.major,
        "major fractions must agree"
    );
    assert_eq!(shared_proj.padding_frac, indep_proj.padding_frac);

    // The minor vec is the only intentional divergence: shared carries the
    // caller's minor ticks; independent stays empty (no per-panel minors).
    assert_eq!(
        shared_proj.minor, shared_minor,
        "shared path carries minor ticks"
    );
    assert!(
        indep_proj.minor.is_empty(),
        "independent path keeps minor empty"
    );

    // Neither mode threads a format when none is supplied; only Thread can
    // ever thread, and only for a numeric format string.
    assert_eq!(shared_threaded, None);
    assert_eq!(indep_threaded, None);
}

/// `#147`: the two modes resolve a temporal tick format through ONE
/// discipline, so an independent per-panel axis renders the caller's
/// pattern instead of the scale's own default labels.
///
/// This is the probe matrix's one observable predicate disagreement. Only
/// `apply_axis_format_or_thread` can re-derive raw epoch-ms values from the
/// scale; `apply_tick_format`'s only input is the already-formatted label
/// STRINGS, which on a temporal scale cannot be re-parsed as timestamps.
/// `Immediate` used to go straight there, so every temporal format was
/// silently DROPPED on an independent-x/y facet panel while the shared
/// axis honored it — for the same chart, both spellings.
///
/// Discriminating in both directions: the `%Y/%m/%d` pattern differs from
/// the default spacing-keyed labels at every spacing (the earlier `%b %Y`
/// probe was not discriminating at 90-day spacing, where the default
/// formatter happens to emit exactly that), and the assertion pins BOTH
/// that the labels changed and that they equal `Thread`'s.
#[test]
fn build_axis_tick_inputs_immediate_applies_a_temporal_format_like_thread_does() {
    use crate::render::scale_resolve::ScaleKind;
    use crate::scale::time::TimeScale;

    let epoch_lo = 1_514_764_800_000.0; // 2018-01-01
    let epoch_hi = epoch_lo + 4.0 * 365.0 * 86_400_000.0; // ~2022-01-01
    let scale = ScaleKind::Time(TimeScale::new_internal(
        vec![epoch_lo, epoch_hi],
        vec![0.0, 1.0],
        false,
        false,
    ));
    let pattern = "%Y/%m/%d";
    let default_labels = scale.tick_labels(6);

    for format_type in [None, Some("time")] {
        let (immediate, _proj, threaded) = build_axis_tick_inputs(
            &scale,
            6,
            TickFormatMode::Immediate {
                format: Some(pattern),
                format_type,
            },
            false,
            Vec::new(),
        )
        .unwrap();
        assert_eq!(threaded, None, "Immediate mode threads nothing");
        assert_ne!(
            immediate, default_labels,
            "the pattern must actually be applied, not dropped ({format_type:?})"
        );
        assert!(
            immediate.iter().all(|l| l.matches('/').count() == 2),
            "every label must be rendered through {pattern}, got {immediate:?}"
        );

        let (thread, _proj, _t) = build_axis_tick_inputs(
            &scale,
            6,
            TickFormatMode::Thread {
                format: Some(pattern.to_string()),
                format_type,
            },
            false,
            Vec::new(),
        )
        .unwrap();
        assert_eq!(
            immediate, thread,
            "one format discipline: Immediate must agree with Thread ({format_type:?})"
        );
    }
}

/// The other side of `#147`'s `Immediate` delegation: a NUMERIC format is
/// still applied immediately and still threads nothing, so routing through
/// `apply_axis_format_or_thread` did not turn the per-panel surface into
/// the deferred one (whose `label_format_override` `scene_build` never
/// applies).
#[test]
fn build_axis_tick_inputs_immediate_still_applies_a_numeric_format_itself() {
    let scale = linear_scale(0.0, 1.0);
    let (labels, _proj, threaded) = build_axis_tick_inputs(
        &scale,
        5,
        TickFormatMode::Immediate {
            format: Some(".0%"),
            format_type: None,
        },
        false,
        Vec::new(),
    )
    .unwrap();
    assert_eq!(
        threaded, None,
        "Immediate must apply the numeric format, never defer it"
    );
    assert!(
        labels.iter().all(|l| l.ends_with('%')),
        "numeric format applied in place, got {labels:?}"
    );
}

/// #133, per-channel `Thread` surface: a time-typed format
/// (`format_type="time"`) on a non-temporal resolved scale must refuse,
/// naming the channel and the offending format — not silently reparse
/// numeric tick labels as epoch-ms and render a nonsense 1970 date.
#[test]
fn build_axis_tick_inputs_thread_refuses_time_format_on_non_temporal_scale() {
    let scale = linear_scale(0.0, 100.0);
    let err = build_axis_tick_inputs(
        &scale,
        10,
        TickFormatMode::Thread {
            format: Some("%Y-%m-%d".to_string()),
            format_type: Some("time"),
        },
        false,
        Vec::new(),
    )
    .unwrap_err();
    assert_eq!(
        err,
        RenderError::TimeFormatOnNonTemporalScale {
            channel: "x",
            surface: crate::render::FormatSurface::Axis,
            format: Some("%Y-%m-%d".into()),
            coord_flipped: false,
        }
    );
}

/// #133, the third (per-panel independent-facet-axis) surface the issue's
/// "both paths" phrasing did not name: `Immediate` mode must refuse
/// identically to `Thread` mode for the same non-temporal scale.
#[test]
fn build_axis_tick_inputs_immediate_refuses_time_format_on_non_temporal_scale() {
    let scale = linear_scale(0.0, 100.0);
    let err = build_axis_tick_inputs(
        &scale,
        10,
        TickFormatMode::Immediate {
            format: Some("%Y-%m-%d"),
            format_type: Some("time"),
        },
        true,
        Vec::new(),
    )
    .unwrap_err();
    assert_eq!(
        err,
        RenderError::TimeFormatOnNonTemporalScale {
            channel: "y",
            surface: crate::render::FormatSurface::Axis,
            format: Some("%Y-%m-%d".into()),
            coord_flipped: false,
        }
    );
}

/// #133 control: an ambiguous bare `%`-containing spec with NO declared
/// `format_type` (a plain d3 percent format like `".1%"`) must NOT
/// refuse on a non-temporal scale — it must keep resolving as the numeric
/// percent format it actually is. Guards against a fix that over-refuses
/// every bare-`%` spec regardless of declared type.
///
/// Since `#147` T23 the pair resolves by the scale's temporality up
/// front, so on a numeric scale it is a d3 spec from the start and takes
/// the d3 path every d3 spec takes: `Thread` threads it, `Immediate`
/// applies it. (It used to be guessed as strftime, un-guessed, and
/// applied immediately in both modes.)
#[test]
fn build_axis_tick_inputs_ambiguous_percent_spec_does_not_refuse_on_non_temporal_scale() {
    let scale = linear_scale(0.0, 1.0);
    let (_labels, _proj, threaded) = build_axis_tick_inputs(
        &scale,
        5,
        TickFormatMode::Thread {
            format: Some(".1%".to_string()),
            format_type: None,
        },
        false,
        Vec::new(),
    )
    .unwrap();
    assert_eq!(threaded.as_deref(), Some(".1%"), "threaded like every d3 spec");
    let (labels, _proj, threaded) = build_axis_tick_inputs(
        &scale,
        5,
        TickFormatMode::Immediate { format: Some(".1%"), format_type: None },
        false,
        Vec::new(),
    )
    .unwrap();
    assert_eq!(threaded, None);
    assert!(
        labels.iter().all(|l| l.ends_with('%')),
        "must format as the numeric percent spec it is, got {labels:?}"
    );
}

/// `#147` close (wave 6A T17), output-moving: a `%`-free spec declared
/// `"time"` names no pattern, so a temporal axis keeps its own
/// spacing-keyed default labels in BOTH modes, the fallback the colorbar
/// and text mark give the identical pair. Before, `Thread` threaded the
/// spec onward, the caller re-seeded it untyped, and it was applied as d3
/// (`2018.00` on year ticks); `Immediate` re-parsed the year strings as
/// epoch-ms (`1970-01-01`).
#[test]
fn build_axis_tick_inputs_time_typed_spec_without_a_pattern_keeps_the_temporal_default() {
    use crate::scale::time::TimeScale;
    let epoch_lo = 1_514_764_800_000.0; // 2018-01-01
    let epoch_hi = epoch_lo + 6.0 * 365.0 * 86_400_000.0;
    let scale = ScaleKind::Time(TimeScale::new_internal(
        vec![epoch_lo, epoch_hi],
        vec![0.0, 1.0],
        false,
        false,
    ));
    let default_labels = scale.tick_labels(6);
    for spec in [".2f", ".3e"] {
        let (thread, _p, thread_threaded) = build_axis_tick_inputs(
            &scale,
            6,
            TickFormatMode::Thread {
                format: Some(spec.to_string()),
                format_type: Some("time"),
            },
            false,
            Vec::new(),
        )
        .unwrap();
        assert_eq!(thread, default_labels, "Thread, {spec}");
        assert_eq!(thread_threaded, None, "Thread must not re-seed {spec} as d3");
        let (immediate, _p, _t) = build_axis_tick_inputs(
            &scale,
            6,
            TickFormatMode::Immediate {
                format: Some(spec),
                format_type: Some("time"),
            },
            false,
            Vec::new(),
        )
        .unwrap();
        assert_eq!(immediate, default_labels, "Immediate, {spec}");
    }
}

/// The non-temporal side of the same pair: a time-declared spec on a
/// linear scale is refused (#133) by BOTH modes. Before, `Thread` threaded
/// it and rendered d3 numbers while `Immediate` refused, so a shared and an
/// independent axis answered the identical spelling differently.
#[test]
fn build_axis_tick_inputs_time_typed_spec_without_a_pattern_refuses_on_non_temporal() {
    let scale = linear_scale(0.0, 100.0);
    let expected = RenderError::TimeFormatOnNonTemporalScale {
        channel: "x",
        surface: crate::render::FormatSurface::Axis,
        format: Some(".2f".into()),
        coord_flipped: false,
    };
    let thread = build_axis_tick_inputs(
        &scale,
        10,
        TickFormatMode::Thread {
            format: Some(".2f".to_string()),
            format_type: Some("time"),
        },
        false,
        Vec::new(),
    )
    .unwrap_err();
    assert_eq!(thread, expected);
    let immediate = build_axis_tick_inputs(
        &scale,
        10,
        TickFormatMode::Immediate {
            format: Some(".2f"),
            format_type: Some("time"),
        },
        false,
        Vec::new(),
    )
    .unwrap_err();
    assert_eq!(immediate, expected);
}

/// `#147` close (wave 6A T21): the field-checked surfaces (legend, text
/// channel) take "temporal" from the scale resolver's own rule — a
/// declared type wins, else the column dtype — and refuse through the
/// one constructor the axes use, carrying their own surface and a `None`
/// format for a bare `format_type="time"`.
#[test]
fn refuse_format_temporality_mismatch_field_keys_on_the_resolved_field_type() {
    use crate::render::FormatSurface;
    use crate::spec::encoding::{DataType as SpecDataType, EncodingSpec};
    use arrow::array::{Float64Array, TimestampMillisecondArray};
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("n", DataType::Float64, false),
            Field::new(
                "d",
                DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None),
                false,
            ),
        ])),
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(TimestampMillisecondArray::from(vec![0_i64, 86_400_000])),
        ],
    )
    .unwrap();
    let enc = |field: &str, type_: Option<SpecDataType>| EncodingSpec {
        field: field.into(),
        type_,
        ..Default::default()
    };
    let refuse = |e: &EncodingSpec, fmt: Option<&str>| {
        refuse_format_temporality_mismatch_field(
            e,
            &batch,
            fmt,
            Some("time"),
            "color",
            FormatSurface::Legend,
        )
    };
    // Quantitative by dtype: both time-typed kinds refuse, naming the surface.
    assert_eq!(
        refuse(&enc("n", None), Some(".2f")),
        Err(RenderError::TimeFormatOnNonTemporalScale {
            channel: "color",
            surface: FormatSurface::Legend,
            format: Some(".2f".into()),
            coord_flipped: false,
        })
    );
    assert_eq!(
        refuse(&enc("n", None), None),
        Err(RenderError::TimeFormatOnNonTemporalScale {
            channel: "color",
            surface: FormatSurface::Legend,
            format: None,
            coord_flipped: false,
        })
    );
    // Temporal by dtype, or declared `:T` over epoch-ms numbers: accepted.
    assert_eq!(refuse(&enc("d", None), Some("%Y")), Ok(()));
    assert_eq!(refuse(&enc("n", Some(SpecDataType::Temporal)), None), Ok(()));
    // The ordinal preset and a d3 spec are not time formats.
    assert_eq!(refuse(&enc("n", None), Some("__ordinal__")), Ok(()));
    assert_eq!(
        refuse_format_temporality_mismatch_field(
            &enc("n", None),
            &batch,
            Some(".2f"),
            None,
            "text",
            FormatSurface::Channel,
        ),
        Ok(())
    );
    // An undeclared field the batch does not carry has no answer here.
    assert_eq!(refuse(&enc("missing", None), Some(".2f")), Ok(()));
}

/// `#147` close (wave 6A T22): the two cells the second intent re-check
/// found, each given one answer in BOTH axis modes (the shared and the
/// independent-facet axis).
///
/// 1. An untyped `%`-bearing spec on a scale that is not temporal is read
///    as d3 like every other surface reads it: `"%Y"` refuses with the d3
///    grammar error (it used to render `250.000000%`), while `".1%"`, a
///    real d3 percent spec, still applies.
/// 2. The ordinal preset on a temporal scale refuses, with or without a
///    stray `"time"` type (it used to render `2001st` on the shared axis
///    and drop the format on the independent one).
#[test]
fn build_axis_tick_inputs_undo_a_wrong_strftime_guess_and_refuse_ordinal_on_time() {
    use crate::scale::time::TimeScale;
    let run = |scale: &ScaleKind, format: &str, format_type: Option<&'static str>| {
        let thread = build_axis_tick_inputs(
            scale,
            6,
            TickFormatMode::Thread { format: Some(format.to_string()), format_type },
            false,
            Vec::new(),
        );
        let immediate = build_axis_tick_inputs(
            scale,
            6,
            TickFormatMode::Immediate { format: Some(format), format_type },
            false,
            Vec::new(),
        );
        (thread, immediate)
    };

    let linear = linear_scale(0.0, 5.0);
    let (thread, immediate) = run(&linear, "%Y", None);
    for (mode, got) in [("Thread", thread), ("Immediate", immediate)] {
        match got {
            Err(RenderError::InvalidFormatSpec { spec, reason }) => {
                assert_eq!(spec, "%Y", "{mode}");
                assert!(reason.contains("unrecognized token"), "{mode}: {reason}");
            }
            other => panic!("{mode}: expected the d3 grammar refusal, got {other:?}"),
        }
    }
    let (thread, immediate) = run(&linear, ".1%", None);
    assert!(thread.is_ok() && immediate.is_ok(), "a real d3 percent spec still applies");

    let time = ScaleKind::Time(TimeScale::new_internal(
        vec![1_514_764_800_000.0, 1_514_764_800_000.0 + 6.0 * 365.0 * 86_400_000.0],
        vec![0.0, 1.0],
        false,
        false,
    ));
    let expected = RenderError::OrdinalFormatOnTemporalScale {
        channel: "x",
        surface: crate::render::FormatSurface::Axis,
        coord_flipped: false,
    };
    for format_type in [None, Some("time"), Some("number")] {
        let (thread, immediate) = run(&time, "__ordinal__", format_type);
        assert_eq!(thread.unwrap_err(), expected, "Thread, {format_type:?}");
        assert_eq!(immediate.unwrap_err(), expected, "Immediate, {format_type:?}");
    }
}

/// `#147` close (wave 6A T22): the field-checked surfaces refuse the
/// ordinal preset on a temporal field through the same constructor, and
/// accept it on a numeric one.
#[test]
fn refuse_ordinal_format_on_a_temporal_field() {
    use crate::render::FormatSurface;
    use crate::spec::encoding::EncodingSpec;
    use arrow::array::{Float64Array, TimestampMillisecondArray};
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("n", DataType::Float64, false),
            Field::new(
                "d",
                DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None),
                false,
            ),
        ])),
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(TimestampMillisecondArray::from(vec![0_i64, 86_400_000])),
        ],
    )
    .unwrap();
    let enc = |field: &str| EncodingSpec { field: field.into(), ..Default::default() };
    for format_type in [None, Some("time")] {
        assert_eq!(
            refuse_format_temporality_mismatch_field(
                &enc("d"),
                &batch,
                Some("__ordinal__"),
                format_type,
                "text",
                FormatSurface::Channel,
            ),
            Err(RenderError::OrdinalFormatOnTemporalScale {
                channel: "text",
                surface: FormatSurface::Channel,
                coord_flipped: false,
            }),
            "{format_type:?}"
        );
        assert_eq!(
            refuse_format_temporality_mismatch_field(
                &enc("n"),
                &batch,
                Some("__ordinal__"),
                format_type,
                "text",
                FormatSurface::Channel,
            ),
            Ok(()),
            "{format_type:?}"
        );
    }
    let msg = RenderError::OrdinalFormatOnTemporalScale {
        channel: "color",
        surface: FormatSurface::Legend,
        coord_flipped: false,
    }
    .to_string();
    assert_eq!(
        msg,
        "the \"ordinal\" format on the 'color' legend formats numbers and requires a \
         non-temporal field, but the field is temporal"
    );
}

/// `#147` close (wave 6A T23, user ruling), the two untyped cells on a
/// temporal value, decided once by the rule owner for every surface:
///
/// 1. A d3 number format (`.2f`, `,.0f`, `d`, or a `"number"`-typed
///    preset) on a temporal field refuses. The axes used to format the
///    year, and the legend and text the epoch-ms count. The typed
///    `.2f` + `"time"` pair still falls back to the temporal default.
/// 2. An untyped `"%Y"` on a temporal field is a date pattern, as on the
///    axis (the legend and text used to refuse it as d3); on a numeric
///    field it stays d3 and refuses with the d3 grammar error. A `%`
///    spec a temporal field cannot read as a date pattern (`".1%"`)
///    refuses with the strftime grammar error instead of echoing itself.
#[test]
fn untyped_specs_on_a_temporal_field_get_one_answer() {
    use crate::render::FormatSurface;
    use crate::spec::encoding::EncodingSpec;
    use arrow::array::{Float64Array, TimestampMillisecondArray};
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("n", DataType::Float64, false),
            Field::new(
                "d",
                DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None),
                false,
            ),
        ])),
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(TimestampMillisecondArray::from(vec![0_i64, 86_400_000])),
        ],
    )
    .unwrap();
    let enc = |field: &str| EncodingSpec { field: field.into(), ..Default::default() };
    let check = |field: &str, fmt: &str, format_type: Option<&str>| {
        refuse_format_temporality_mismatch_field(
            &enc(field),
            &batch,
            Some(fmt),
            format_type,
            "color",
            FormatSurface::Legend,
        )
    };
    for (fmt, format_type) in
        [(".2f", None), (",.0f", None), ("d", None), (".1%", Some("number"))]
    {
        assert_eq!(
            check("d", fmt, format_type),
            Err(RenderError::NumberFormatOnTemporalScale {
                channel: "color",
                surface: FormatSurface::Legend,
                format: fmt.into(),
                coord_flipped: false,
            }),
            "{fmt} {format_type:?}"
        );
        assert_eq!(check("n", fmt, format_type), Ok(()), "{fmt} on a number");
    }
    assert_eq!(check("d", ".2f", Some("time")), Ok(()), "typed: temporal default");
    assert_eq!(check("d", "", None), Ok(()), "an empty spec names no format");

    assert_eq!(check("d", "%Y", None), Ok(()));
    match check("n", "%Y", None) {
        Err(RenderError::InvalidFormatSpec { spec, reason }) => {
            assert_eq!(spec, "%Y");
            assert!(reason.contains("unrecognized token"), "{reason}");
        }
        other => panic!("expected the d3 grammar refusal, got {other:?}"),
    }
    match check("d", ".1%", None) {
        Err(RenderError::InvalidFormatSpec { spec, reason }) => {
            assert_eq!(spec, ".1%");
            assert!(reason.contains("invalid strftime pattern"), "{reason}");
        }
        other => panic!("expected the strftime grammar refusal, got {other:?}"),
    }
    assert_eq!(check("n", ".1%", None), Ok(()), "a real d3 percent on a number");

    // The applier formats with the same resolution: years, not epoch ms.
    let d = 978_307_200_000.0; // 2001-01-01
    assert_eq!(crate::render::format::format_value_with_spec(
        d,
        Some("%Y"),
        None,
        crate::render::format::Temporality::Temporal,
    ), "2001");
    use crate::render::format::Temporality;
    assert_eq!(field_temporality(&enc("d"), &batch), Some(Temporality::Temporal));
    assert_eq!(field_temporality(&enc("n"), &batch), Some(Temporality::NonTemporal));
    assert_eq!(field_temporality(&enc("missing"), &batch), None);
}

/// `#147` T23 on the axes: a d3 number format on a `Time` scale refuses
/// in both modes (the shared axis rendered `2001.00`, the independent
/// facet axis dropped the format), while the untyped `"%Y"` renders years
/// in both.
#[test]
fn build_axis_tick_inputs_refuse_a_number_format_on_time() {
    use crate::scale::time::TimeScale;
    let time = ScaleKind::Time(TimeScale::new_internal(
        vec![1_514_764_800_000.0, 1_514_764_800_000.0 + 6.0 * 365.0 * 86_400_000.0],
        vec![0.0, 1.0],
        false,
        false,
    ));
    let run = |format: &str, format_type: Option<&'static str>| {
        let thread = build_axis_tick_inputs(
            &time,
            6,
            TickFormatMode::Thread { format: Some(format.to_string()), format_type },
            true,
            Vec::new(),
        );
        let immediate = build_axis_tick_inputs(
            &time,
            6,
            TickFormatMode::Immediate { format: Some(format), format_type },
            true,
            Vec::new(),
        );
        (thread, immediate)
    };
    for fmt in [".2f", ",.0f", "d"] {
        let expected = RenderError::NumberFormatOnTemporalScale {
            channel: "y",
            surface: crate::render::FormatSurface::Axis,
            format: fmt.into(),
            coord_flipped: false,
        };
        let (thread, immediate) = run(fmt, None);
        assert_eq!(thread.unwrap_err(), expected, "Thread {fmt}");
        assert_eq!(immediate.unwrap_err(), expected, "Immediate {fmt}");
    }
    let (thread, immediate) = run("%Y", None);
    for (mode, got) in [("Thread", thread), ("Immediate", immediate)] {
        let (labels, _, threaded) = got.unwrap();
        assert_eq!(threaded, None, "{mode}");
        assert!(labels.iter().all(|l| l.len() == 4 && l.starts_with("20")), "{mode}: {labels:?}");
    }
    let (thread, immediate) = run(".2f", Some("time"));
    assert!(thread.is_ok() && immediate.is_ok(), "typed: temporal default stands");
}

/// `#147` close (wave 6A T17), output-moving: the ordinal preset beside a
/// stray `"time"` type is ordinal in both modes, and is not refused as a
/// time format on a non-temporal scale. Before, `Immediate` refused it up
/// front on the raw `format_type` while `Thread` rendered ordinals.
#[test]
fn build_axis_tick_inputs_ordinal_preset_wins_over_a_stray_time_type() {
    let scale = linear_scale(1.0, 5.0);
    let (thread, _p, threaded) = build_axis_tick_inputs(
        &scale,
        5,
        TickFormatMode::Thread {
            format: Some("__ordinal__".to_string()),
            format_type: Some("time"),
        },
        false,
        Vec::new(),
    )
    .unwrap();
    assert_eq!(threaded.as_deref(), Some("__ordinal__"));
    // The threaded spec is applied later by `apply_tick_format`; the
    // stray type must not un-declare it there either.
    let applied = apply_tick_format(thread, threaded.as_deref(), Some("time"));
    let (immediate, _p, _t) = build_axis_tick_inputs(
        &scale,
        5,
        TickFormatMode::Immediate {
            format: Some("__ordinal__"),
            format_type: Some("time"),
        },
        false,
        Vec::new(),
    )
    .unwrap();
    assert_eq!(immediate, applied);
    assert!(immediate.contains(&"1st".to_string()), "{immediate:?}");
    assert!(immediate.contains(&"2nd".to_string()), "{immediate:?}");
}

// ── #52 Task 3: provisional secondary y-axis inputs ──────────────────────

/// `prepare_render_inputs` derives one `AxesInput.secondary_y` entry per
/// `independent_y` layer, titled from that layer's own y encoding and
/// carrying tick labels resolved against ITS OWN domain — not the
/// primary's (secondary-y-axis, GH #52 Task 3; the layout stage this feeds
/// reserves one right-side band + emits one axis per entry).
#[test]
fn prepare_render_inputs_independent_y_layer_produces_secondary_axis_input() {
    use crate::spec::layer::Layer;

    let primary = Layer {
        mark: Mark::Line,
        encoding: Encoding {
            y: Some(EncodingSpec {
                field: "y0".into(),
                ..Default::default()
            }),
            ..Default::default()
        },
        ..Default::default()
    };
    let secondary = Layer {
        mark: Mark::Line,
        encoding: Encoding {
            y: Some(EncodingSpec {
                field: "y1".into(),
                title: Some("Secondary Title".into()),
                ..Default::default()
            }),
            ..Default::default()
        },
        independent_y: true,
        ..Default::default()
    };

    let spec = ChartSpec {
        mark: Mark::Line,
        encoding: Encoding {
            x: Some(EncodingSpec {
                field: "x".into(),
                ..Default::default()
            }),
            ..Default::default()
        },
        layers: Some(vec![primary, secondary]),
        ..Default::default()
    };

    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, false),
        Field::new("y0", DataType::Float64, false),
        Field::new("y1", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![100.0, 200.0, 300.0])),
        ],
    )
    .unwrap();

    let theme = crate::layout::ThemeInputs::default();
    let prep = prepare_render_inputs(
        &spec,
        &batch,
        &theme,
        crate::layout::Viewport {
            width: 600.0,
            height: 400.0,
        },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .unwrap();

    assert_eq!(
        prep.axes.secondary_y.len(),
        1,
        "one independent_y layer → one secondary axis input"
    );
    let secondary_axis = &prep.axes.secondary_y[0];
    assert_eq!(
        secondary_axis.title.get().as_deref(),
        Some("Secondary Title")
    );
    assert_ne!(
        secondary_axis.tick_labels, prep.axes.y.tick_labels,
        "the secondary axis resolves its OWN domain, not the primary's"
    );
}

/// R3: the third `InvalidAxisOrient` production chain —
/// `build_secondary_y_axis_inputs`'s `build_axis_input` call (distinct from
/// `build_axes`'s primary-axis chain, which also patches, and the
/// chart-level `configure_axis` chain, which is exempt — see the field doc
/// on `RenderError::InvalidAxisOrient`) — must also un-flip under
/// `CoordFlip`. `build_secondary_y_axis_inputs` always validates
/// `layer_encoding.y` (post-flip) as `Channel::Y` (secondary axes are
/// always drawn right-side, regardless of flip), so to reach it, the
/// user's `fm.Axis(orient="top")` must be attached to the layer's own PRE-
/// flip `x` — `build_layers`' whole-`EncodingSpec` swap moves it onto the
/// POST-flip `y` slot this chain reads. The message must still say the
/// axis is invalid for `x` (what the user wrote), not the resolved `y`.
#[test]
fn secondary_y_axis_orient_error_names_users_channel_under_flip() {
    use crate::spec::coord::CoordKind;
    use crate::spec::layer::Layer;

    let primary = Layer {
        mark: Mark::Line,
        encoding: Encoding {
            y: Some(EncodingSpec {
                field: "y0".into(),
                ..Default::default()
            }),
            ..Default::default()
        },
        ..Default::default()
    };
    let secondary = Layer {
        mark: Mark::Line,
        encoding: Encoding {
            // Pre-flip x carries the bad orient; `build_layers` swaps it
            // onto post-flip y, where `build_secondary_y_axis_inputs`
            // actually validates it (see the doc comment above).
            x: Some(enc_with_axis("x", serde_json::json!({ "orient": "top" }))),
            y: Some(EncodingSpec {
                field: "y1".into(),
                ..Default::default()
            }),
            ..Default::default()
        },
        independent_y: true,
        ..Default::default()
    };

    let spec = ChartSpec {
        mark: Mark::Line,
        encoding: Encoding {
            x: Some(EncodingSpec {
                field: "x".into(),
                ..Default::default()
            }),
            ..Default::default()
        },
        layers: Some(vec![primary, secondary]),
        coord: Some(CoordKind::Flip),
        ..Default::default()
    };

    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, false),
        Field::new("y0", DataType::Float64, false),
        Field::new("y1", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![100.0, 200.0, 300.0])),
        ],
    )
    .unwrap();

    let theme = crate::layout::ThemeInputs::default();
    let err = prepare_render_inputs(
        &spec,
        &batch,
        &theme,
        crate::layout::Viewport {
            width: 600.0,
            height: 400.0,
        },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .expect_err("orient='top' on the secondary y axis must fail loud");
    match &err {
        crate::render::RenderError::InvalidAxisOrient {
            channel,
            orient,
            coord_flipped,
        } => {
            assert_eq!(
                *channel, "y",
                "resolved token: secondary axes always validate as Channel::Y"
            );
            assert_eq!(orient, "top");
            assert!(*coord_flipped);
        }
        other => panic!("expected InvalidAxisOrient, got {other:?}"),
    }
    assert_eq!(
        format!("{err}"),
        "axis orient 'top' is invalid for the x axis (expected 'left' or 'right' — under \
         CoordFlip, x renders as the vertical axis)",
        "message must name the channel the user wrote it under ('x'), not the resolved one ('y'); \
         `allowed` reflects the RESOLVED axis (y → left/right), and the parenthetical explains \
         why 'left'/'right' applies to something named 'x'"
    );
}

/// R3 byte-identity: the same secondary-y-axis cross-dimension `orient`
/// error, unflipped, renders exactly the pre-fix message text.
#[test]
fn secondary_y_axis_orient_error_unchanged_when_not_flipped() {
    use crate::spec::layer::Layer;

    let primary = Layer {
        mark: Mark::Line,
        encoding: Encoding {
            y: Some(EncodingSpec {
                field: "y0".into(),
                ..Default::default()
            }),
            ..Default::default()
        },
        ..Default::default()
    };
    let secondary = Layer {
        mark: Mark::Line,
        encoding: Encoding {
            y: Some(enc_with_axis("y1", serde_json::json!({ "orient": "top" }))),
            ..Default::default()
        },
        independent_y: true,
        ..Default::default()
    };

    let spec = ChartSpec {
        mark: Mark::Line,
        encoding: Encoding {
            x: Some(EncodingSpec {
                field: "x".into(),
                ..Default::default()
            }),
            ..Default::default()
        },
        layers: Some(vec![primary, secondary]),
        ..Default::default()
    };

    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, false),
        Field::new("y0", DataType::Float64, false),
        Field::new("y1", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![100.0, 200.0, 300.0])),
        ],
    )
    .unwrap();

    let theme = crate::layout::ThemeInputs::default();
    let err = prepare_render_inputs(
        &spec,
        &batch,
        &theme,
        crate::layout::Viewport {
            width: 600.0,
            height: 400.0,
        },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .expect_err("orient='top' on the secondary y axis must fail loud");
    assert_eq!(
        format!("{err}"),
        "axis orient 'top' is invalid for the y axis (expected 'left' or 'right')"
    );
}

/// Wire back-compat: no layer sets `independent_y` → `secondary_y` stays
/// empty. Byte-stable gate mirroring the scene_build.rs Task 2 gate.
#[test]
fn prepare_render_inputs_no_independent_y_layer_leaves_secondary_y_empty() {
    let spec = ChartSpec {
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec {
                field: "x".into(),
                ..Default::default()
            }),
            y: Some(EncodingSpec {
                field: "y".into(),
                ..Default::default()
            }),
            ..Default::default()
        },
        ..Default::default()
    };
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
        ],
    )
    .unwrap();

    let theme = crate::layout::ThemeInputs::default();
    let prep = prepare_render_inputs(
        &spec,
        &batch,
        &theme,
        crate::layout::Viewport {
            width: 600.0,
            height: 400.0,
        },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .unwrap();
    assert!(prep.axes.secondary_y.is_empty());
}

// ── #95: a sort shorthand binds to the channel the user WROTE ───────────
//
// End-to-end through `prepare_render_inputs`, which is the production
// entry that applies the `CoordFlip` swap (`build_layers`) and then
// resolves the category axis against the post-flip rendering encoding.
// A unit call into `resolve_scales` would hand it an un-swapped encoding
// and prove nothing about the two-spaces defect.

/// `cat` in insertion order a, b, c; `val` per category a=3, b=1, c=2. So
/// the descending-by-`val` order is [a, c, b] — distinct from insertion
/// order (a, b, c) AND from alphabetical descending (c, b, a), so the
/// asserted vector can only come from a data-aware sort that actually
/// aggregated `val`.
fn sort_shorthand_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("cat", DataType::Utf8, false),
        Field::new("val", DataType::Float64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
            Arc::new(Float64Array::from(vec![3.0, 1.0, 2.0])),
        ],
    )
    .unwrap()
}

/// A bar chart with the category AUTHORED on `x` and the value on `y`,
/// carrying `sort` on the category encoding. `flipped` adds
/// `.coord(CoordFlip())`, which is the ONLY difference between the twins —
/// the authored encoding (and therefore the authored `sort` text) is
/// byte-identical either way.
fn sorted_bar_spec(sort: serde_json::Value, flipped: bool) -> ChartSpec {
    ChartSpec {
        mark: Mark::Bar,
        encoding: Encoding {
            x: Some(EncodingSpec {
                field: "cat".into(),
                sort: Some(sort),
                ..Default::default()
            }),
            y: Some(EncodingSpec {
                field: "val".into(),
                ..Default::default()
            }),
            ..Default::default()
        },
        coord: flipped.then_some(crate::spec::coord::CoordKind::Flip),
        ..Default::default()
    }
}

/// The category axis's resolved domain, read off the axis the categories
/// actually render on: `x` unflipped, `y` under `CoordFlip`. Returns the
/// tick labels alongside every `SortSpecIgnored` reason the prepare pass
/// produced, so a test can assert order and silence together.
fn category_axis_order(sort: serde_json::Value, flipped: bool) -> (Vec<String>, Vec<String>) {
    let spec = sorted_bar_spec(sort, flipped);
    let prep = prepare_render_inputs(
        &spec,
        &sort_shorthand_batch(),
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport {
            width: 600.0,
            height: 400.0,
        },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .expect("the fixture renders");
    let axis = if flipped { &prep.axes.y } else { &prep.axes.x };
    let ignored = prep
        .warnings
        .iter()
        .filter_map(|w| match w {
            RenderWarning::SortSpecIgnored { reason } => Some(reason.clone()),
            _ => None,
        })
        .collect();
    (axis.tick_labels.clone(), ignored)
}

/// #95, the defect: `sort="-y"` names the channel the user's own
/// `.encode(y=...)` bound, so a flipped chart must order its categories
/// exactly like its unflipped twin. Before the fix the flipped twin looked
/// up the RENDERED `y` field (the category column `cat`, post-swap),
/// found no numeric aggregate worth the name, and fell back to insertion
/// order with a `SortSpecIgnored` — measured [a, b, c] against the
/// unflipped [a, c, b].
#[test]
fn sort_shorthand_binds_to_the_authored_channel_under_flip() {
    let (unflipped, unflipped_ignored) = category_axis_order(serde_json::json!("-y"), false);
    assert_eq!(
        unflipped,
        ["a", "c", "b"],
        "control: unflipped, sort=\"-y\" orders categories by descending val"
    );
    assert!(
        unflipped_ignored.is_empty(),
        "control must not warn: {unflipped_ignored:?}"
    );

    let (flipped, flipped_ignored) = category_axis_order(serde_json::json!("-y"), true);
    assert_eq!(
        flipped, unflipped,
        "under CoordFlip the authored \"-y\" still names the user's own y \
         channel (the value column), so the category order must match the \
         unflipped twin"
    );
    assert!(
        flipped_ignored.is_empty(),
        "the authored spelling must resolve, not degrade: {flipped_ignored:?}"
    );
}

/// Negative control for the test above. The sort-FIELD object names a
/// FIELD, never a channel, so it was already flip-invariant before the
/// fix — which is what makes the two forms' pre-fix DISAGREEMENT the
/// argument for authored semantics. After the fix they must agree, and
/// both must equal the object form's flip-invariant answer.
#[test]
fn sort_field_object_is_flip_invariant_and_agrees_with_the_shorthand() {
    let object_form = serde_json::json!({"field": "val", "op": "sum", "order": "descending"});
    let (unflipped, unflipped_ignored) = category_axis_order(object_form.clone(), false);
    let (flipped, flipped_ignored) = category_axis_order(object_form, true);
    assert_eq!(
        unflipped,
        ["a", "c", "b"],
        "the object form names a field, not a channel"
    );
    assert_eq!(
        flipped, unflipped,
        "the object form was always flip-invariant"
    );
    assert!(unflipped_ignored.is_empty() && flipped_ignored.is_empty());

    let (shorthand_flipped, _) = category_axis_order(serde_json::json!("-y"), true);
    assert_eq!(
        shorthand_flipped, flipped,
        "the two spellings of one intent must produce one order under flip"
    );
}

/// A genuinely unresolvable sort must be reported ONCE per chart. The
/// provisional placeholder pass here and `scene_build`'s panel-real pass
/// both resolve the same chart-level fact, so without
/// `provisional_scale_warning_superseded_by_panel_pass` naming it the user
/// gets two identical `UserWarning`s for one mistake.
#[test]
fn unresolvable_sort_warns_once_per_chart() {
    let spec = sorted_bar_spec(serde_json::json!({"field": "does_not_exist"}), false);
    let out = crate::render::render_svg(
        &spec,
        &sort_shorthand_batch(),
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport {
            width: 600.0,
            height: 400.0,
        },
        &crate::render::config::RenderConfig::default(),
        &crate::render::chart_config::ChartConfig::default(),
    )
    .expect("the fixture renders");
    let sort_warnings: Vec<&RenderWarning> = out
        .warnings
        .iter()
        .filter(|w| matches!(w, RenderWarning::SortSpecIgnored { .. }))
        .collect();
    assert_eq!(
        sort_warnings.len(),
        1,
        "one ignored sort spec is one fact: {sort_warnings:?}"
    );
}

/// The companion to the test above, and the reason the fix belongs in
/// `provisional_scale_warning_superseded_by_panel_pass` rather than in
/// `drop_placeholder_only_warnings`: when the data is empty after
/// transforms, `build_scene`'s panel loop skips every zero-row panel, so
/// no panel-real pass ever re-reports the fact. The provisional copy is
/// then the only report this render will ever make and must survive.
#[test]
fn unresolvable_sort_still_warns_when_no_panel_resolves() {
    let mut spec = sorted_bar_spec(serde_json::json!({"field": "does_not_exist"}), false);
    // A `top_k` keeping zero groups: the batch is non-empty on entry (so
    // `prepare_render_inputs` runs its provisional pass) but every panel
    // is zero-row by the time `build_scene` loops. Chosen over a `filter`
    // that matches nothing because `filter`'s predicate is evaluated
    // through the Python interpreter, which a `cargo test` binary has not
    // initialized.
    spec.transforms = vec![serde_json::from_value(serde_json::json!({
        "type": "top_k",
        "n": 0,
        "field": "val"
    }))
    .expect("top_k transform spec")];
    let out = crate::render::render_svg(
        &spec,
        &sort_shorthand_batch(),
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport {
            width: 600.0,
            height: 400.0,
        },
        &crate::render::config::RenderConfig::default(),
        &crate::render::chart_config::ChartConfig::default(),
    )
    .expect("an all-filtered chart still renders its axes");
    assert_eq!(
        out.warnings
            .iter()
            .filter(|w| matches!(w, RenderWarning::SortSpecIgnored { .. }))
            .count(),
        1,
        "the provisional copy is the only report this render can make: {:?}",
        out.warnings
    );
}

// ── Utf8View/LargeUtf8 normalization via arrow::compute::cast ──

/// A raw input batch carrying the SAME string column at `Utf8`,
/// `Utf8View`, and `LargeUtf8` normalizes all three to byte-identical
/// `Utf8` output (values and nulls), so a polars frame (which may export
/// either non-`Utf8` string physical type depending on version) renders
/// exactly like the `Utf8` frame it is logically equal to.
#[test]
fn normalize_string_views_casts_utf8view_and_large_utf8_to_byte_identical_utf8() {
    use arrow::array::{LargeStringArray, StringViewArray};

    let values: Vec<Option<&str>> = vec![Some("alpha"), None, Some("beta"), Some("")];

    let utf8_batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)])),
        vec![Arc::new(StringArray::from(values.clone())) as ArrayRef],
    )
    .unwrap();
    let utf8view_batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8View, true)])),
        vec![Arc::new(StringViewArray::from(values.clone())) as ArrayRef],
    )
    .unwrap();
    let large_utf8_batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("s", DataType::LargeUtf8, true)])),
        vec![Arc::new(LargeStringArray::from(values.clone())) as ArrayRef],
    )
    .unwrap();

    let want = normalize_string_views(&utf8_batch).expect("Utf8 passes through");
    let from_view = normalize_string_views(&utf8view_batch).expect("Utf8View normalizes");
    let from_large = normalize_string_views(&large_utf8_batch).expect("LargeUtf8 normalizes");

    for got in [&from_view, &from_large] {
        assert_eq!(got.schema(), want.schema(), "normalized schema must be Utf8");
        let want_col = want.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let got_col = got.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let want_values: Vec<Option<&str>> =
            (0..want_col.len()).map(|i| (!want_col.is_null(i)).then_some(want_col.value(i))).collect();
        let got_values: Vec<Option<&str>> =
            (0..got_col.len()).map(|i| (!got_col.is_null(i)).then_some(got_col.value(i))).collect();
        assert_eq!(
            want_values, got_values,
            "normalized Utf8View/LargeUtf8 must render byte-identical to the Utf8 original"
        );
    }
}

/// A raw batch's `Utf8View`/`LargeUtf8` category column renders
/// byte-identical SVG to the same chart over the `Utf8` column — the
/// category axis's tick labels are the string content itself, so any
/// normalization gap would draw different (or missing) label text
/// rather than merely differ at the batch level.
#[test]
fn utf8view_and_large_utf8_raw_input_render_byte_identical_svg_to_utf8() {
    use arrow::array::{LargeStringArray, StringViewArray};

    fn bar_spec_over(field: &str) -> ChartSpec {
        ChartSpec {
            mark: Mark::Bar,
            encoding: Encoding {
                x: Some(EncodingSpec { field: field.into(), ..Default::default() }),
                y: Some(EncodingSpec { field: "val".into(), ..Default::default() }),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn render(batch: &RecordBatch) -> String {
        crate::render::render_svg(
            &bar_spec_over("cat"),
            batch,
            &crate::layout::ThemeInputs::default(),
            crate::layout::Viewport { width: 600.0, height: 400.0 },
            &crate::render::config::RenderConfig::default(),
            &crate::render::chart_config::ChartConfig::default(),
        )
        .expect("the fixture renders")
        .bytes
    }

    let cats = vec!["alpha", "beta", "gamma"];
    let vals = vec![3.0, 1.0, 2.0];
    let val_field = Field::new("val", DataType::Float64, false);

    let utf8_batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("cat", DataType::Utf8, false), val_field.clone()])),
        vec![
            Arc::new(StringArray::from(cats.clone())) as ArrayRef,
            Arc::new(Float64Array::from(vals.clone())) as ArrayRef,
        ],
    )
    .unwrap();
    let view_batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("cat", DataType::Utf8View, false), val_field.clone()])),
        vec![
            Arc::new(StringViewArray::from(cats.clone())) as ArrayRef,
            Arc::new(Float64Array::from(vals.clone())) as ArrayRef,
        ],
    )
    .unwrap();
    let large_batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("cat", DataType::LargeUtf8, false), val_field])),
        vec![
            Arc::new(LargeStringArray::from(cats)) as ArrayRef,
            Arc::new(Float64Array::from(vals)) as ArrayRef,
        ],
    )
    .unwrap();

    let want = render(&utf8_batch);
    assert_eq!(
        render(&view_batch), want,
        "Utf8View raw input must render byte-identical SVG to Utf8"
    );
    assert_eq!(
        render(&large_batch), want,
        "LargeUtf8 raw input must render byte-identical SVG to Utf8"
    );
}

/// `stat_bin`'s grouped path echoes its `groupby` column back at ITS OWN
/// dtype (`group_key::materialize_groupby_col`): this fixture confirms
/// `large_out`'s `g` column is genuinely `LargeUtf8` before normalizing,
/// then proves `normalize_transform_outputs` casts that real `stat_bin`
/// output to `Utf8` with the same values a `Utf8`-groupby run produces.
#[test]
fn normalize_transform_outputs_casts_a_real_transforms_large_utf8_groupby_echo() {
    use crate::transform::bin::{apply as bin_apply, BinSpec};
    use crate::transform::bin_mode::BinMode;
    use arrow::array::LargeStringArray;

    fn batch_with_group_dtype(group_dtype: DataType) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("g", group_dtype.clone(), false),
        ]));
        let g_col: ArrayRef = match group_dtype {
            DataType::Utf8 => Arc::new(StringArray::from(vec!["p", "q", "p", "q"])),
            DataType::LargeUtf8 => Arc::new(LargeStringArray::from(vec!["p", "q", "p", "q"])),
            other => unreachable!("test only constructs Utf8/LargeUtf8 batches, got {other:?}"),
        };
        RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![1.0, 9.0, 1.0, 9.0])) as ArrayRef, g_col],
        )
        .unwrap()
    }

    fn grouped_bin_spec() -> BinSpec {
        BinSpec {
            field: "x".into(),
            mode: BinMode::Fixed { n: 2 },
            extent: Some((0.0, 10.0)),
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: vec!["g".into()],
            agg_field: None,
            agg: None,
            name: None,
            per_row: false,
            aggregates: None,
        }
    }

    let utf8_out = bin_apply(&grouped_bin_spec(), &batch_with_group_dtype(DataType::Utf8)).unwrap();
    let large_out = bin_apply(&grouped_bin_spec(), &batch_with_group_dtype(DataType::LargeUtf8)).unwrap();
    let g_idx = large_out.schema().index_of("g").unwrap();
    assert_eq!(
        large_out.schema().field(g_idx).data_type(), &DataType::LargeUtf8,
        "stat_bin must echo the groupby column's own dtype before normalization, \
         or this fixture is not exercising a real transform's LargeUtf8 output"
    );

    let mut outputs = HashMap::new();
    outputs.insert("binned".to_string(), large_out);
    let normalized = normalize_transform_outputs(outputs)
        .expect("a real transform's LargeUtf8 groupby echo normalizes");
    let got = &normalized["binned"];
    let got_idx = got.schema().index_of("g").unwrap();
    assert_eq!(got.schema().field(got_idx).data_type(), &DataType::Utf8);

    let want_idx = utf8_out.schema().index_of("g").unwrap();
    let want_g = utf8_out.column(want_idx).as_any().downcast_ref::<StringArray>().unwrap();
    let got_g = got.column(got_idx).as_any().downcast_ref::<StringArray>().unwrap();
    let want_values: Vec<&str> = (0..want_g.len()).map(|i| want_g.value(i)).collect();
    let got_values: Vec<&str> = (0..got_g.len()).map(|i| got_g.value(i)).collect();
    assert_eq!(
        got_values, want_values,
        "the normalized LargeUtf8 groupby echo must match the Utf8 baseline byte-for-byte"
    );
}

// ── secondary y beside a caller's y scale ─────────────────────────────────

/// `prepare_render_inputs` over a wire spec and three-row batch whose `y`
/// column holds `y` and `z` column holds `z`.
fn prepare_secondary_y_spec(spec: serde_json::Value, z: arrow::array::ArrayRef) -> Result<PreparedInputs, RenderError> {
    let spec: crate::spec::chart::ChartSpec = serde_json::from_value(spec).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Utf8, false),
        Field::new("y", DataType::Float64, false),
        Field::new("z", z.data_type().clone(), false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(StringArray::from(vec!["a", "b", "c"])), Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])), z],
    )
    .unwrap();
    prepare_render_inputs(
        &spec,
        &batch,
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport { width: 600.0, height: 400.0 },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
}

/// An `independent_y` layer's own caller scale configures only its own axis:
/// beside a bar, `SecondaryY("z", scale=LogScale())` leaves the primary y the
/// bar's zero-anchored linear extent, `(0, 3)`, and the secondary axis keeps
/// its log ladder. The spec is the one `Chart.mark_bar().encode(x="x",
/// y="y") + SecondaryY("z", scale=fm.LogScale())` sends. If the independent
/// layer's scale established the primary axis, that axis would be a log scale
/// over the bar's own values, `(1, 3)`, with no zero anchor.
#[test]
fn an_independent_y_layers_own_scale_does_not_establish_the_primary_axis() {
    let spec = serde_json::json!({
        "mark": "bar",
        "encoding": {
            "x": {"field": "x"},
            "y": {"field": "y", "scale": {
                "type": "linear", "clamp": false, "nice": false, "zero": true, "systemInjected": true
            }}
        },
        "layers": [
            {"mark": "bar", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}}},
            {
                "mark": "line",
                "encoding": {
                    "x": {"field": "x"},
                    "y": {"field": "z", "scale": {"type": "log", "base": 10.0, "clamp": false, "nice": false}}
                },
                "independent_y": true
            }
        ]
    });
    let prep = prepare_secondary_y_spec(spec, Arc::new(Float64Array::from(vec![1.0, 10.0, 100.0]))).unwrap();
    assert!(
        matches!(prep.provisional_scales.y, crate::render::scale_resolve::ScaleKind::Linear(_)),
        "primary y is the bar's linear scale, got {:?}",
        prep.provisional_scales.y
    );
    assert_eq!(prep.provisional_scales.y.data_domain(), Some((0.0, 3.0)));
    let primary = &prep.axes.y.tick_labels;
    assert_eq!(
        (primary.first().map(String::as_str), primary.last().map(String::as_str)),
        (Some("3"), Some("0")),
        "the primary ladder runs from the bar's top to zero: {primary:?}"
    );
    assert_eq!(prep.axes.secondary_y.len(), 1);
    assert_eq!(
        prep.axes.secondary_y[0].tick_labels,
        ["100", "63.0957", "39.8107", "25.1189", "15.8489", "10", "6.3096", "3.9811", "2.5119", "1.5849", "1"],
        "the secondary axis keeps its log ladder"
    );
}

/// A secondary-y layer draws on its own axis, so a non-numeric column there
/// is no member of a caller-established primary y axis and renders: the spec
/// `Chart.mark_point().encode(x=..., y=fm.Y("y", scale=fm.LinearScale())) +
/// SecondaryY("z")` sends, with `z` a string column, resolves its secondary
/// axis over the categories.
#[test]
fn a_non_numeric_secondary_y_column_beside_a_caller_y_scale_renders() {
    let caller = serde_json::json!({"type": "linear", "clamp": false, "nice": false, "zero": false});
    let spec = serde_json::json!({
        "mark": "point",
        "encoding": {"x": {"field": "x"}, "y": {"field": "y", "scale": caller}},
        "layers": [
            {"mark": "point", "encoding": {"x": {"field": "x"}, "y": {"field": "y", "scale": caller}}},
            {"mark": "line", "encoding": {"x": {"field": "x"}, "y": {"field": "z"}}, "independent_y": true}
        ]
    });
    let prep = prepare_secondary_y_spec(spec, Arc::new(StringArray::from(vec!["lo", "mid", "hi"])))
        .expect("a secondary-y string column is not judged against the primary axis");
    assert_eq!(prep.axes.secondary_y.len(), 1);
    assert_eq!(prep.axes.secondary_y[0].tick_labels, vec!["lo", "mid", "hi"]);
}

// ── Legend title from the binding that built the color scale ─────────────

/// A two-layer chart over `batch3` whose only `color` binding is
/// `layer0_color` on layer 0 (`chart_color` at chart level), the shape a
/// composite mark's own layer-level hue produces.
fn layered_color_spec(chart_color: Option<EncodingSpec>, layer0_color: Option<EncodingSpec>) -> ChartSpec {
    let xy = Encoding {
        x: Some(EncodingSpec { field: "x".into(), ..Default::default() }),
        y: Some(EncodingSpec { field: "y".into(), ..Default::default() }),
        ..Default::default()
    };
    let layer = |mark, color| crate::spec::layer::Layer {
        mark,
        encoding: Encoding { color, ..Default::default() },
        ..Default::default()
    };
    ChartSpec {
        mark: Mark::Line,
        encoding: Encoding { color: chart_color, ..xy },
        layers: Some(vec![layer(Mark::Line, layer0_color), layer(Mark::Point, None)]),
        ..Default::default()
    }
}

/// The legend is titled, and styled, from the color binding its scale was
/// built from: the first layer's resolved encoding, whether that binding is
/// the layer's own or inherited from the chart. A layer-level `color` with a
/// `title=`, and one with a `Legend(title=)`, title the legend exactly as the
/// same bindings at chart level do; a chart with no color binding anywhere
/// is the control and stays untitled.
#[test]
fn a_layer_level_color_binding_titles_its_legend_like_a_chart_level_one() {
    use crate::render::chart_config::LegendStyleSpec;
    let species = EncodingSpec { field: "species".into(), ..Default::default() };
    let titled = EncodingSpec { title: Some("Kind".into()), ..species.clone() };
    let legend_titled = EncodingSpec {
        legend: Some(Box::new(LegendStyleSpec { title: Some("Group".into()), ..Default::default() })),
        ..species.clone()
    };
    let batch = batch3();
    for (binding, title, override_title) in
        [(&species, "species", None), (&titled, "Kind", None), (&legend_titled, "species", Some("Group"))]
    {
        let chart_level = prep(&layered_color_spec(Some(binding.clone()), None), &batch);
        let layer_level = prep(&layered_color_spec(None, Some(binding.clone())), &batch);
        for p in [&chart_level, &layer_level] {
            assert_eq!(p.legend_entries.iter().map(|e| e.label.as_str()).collect::<Vec<_>>(), ["a", "b"]);
            assert_eq!(p.legend_title.as_deref(), Some(title));
            assert_eq!(p.legend_overrides.title.as_deref(), override_title);
        }
    }
    let uncolored = prep(&layered_color_spec(None, None), &batch);
    assert!(uncolored.legend_entries.is_empty());
    assert_eq!(uncolored.legend_title, None);
}

/// The aux legend channels, each bound at chart level or on the first layer
/// only, over a two-layer chart whose first layer draws the channel (a line
/// for `stroke_dash`, a point otherwise).
fn layered_aux_spec(channel: &str, chart: Option<EncodingSpec>, layer0: Option<EncodingSpec>) -> ChartSpec {
    let bind = |enc: Option<EncodingSpec>, base: Encoding| match channel {
        "size" => Encoding { size: enc, ..base },
        "shape" => Encoding { shape: enc, ..base },
        "stroke_dash" => Encoding { stroke_dash: enc, ..base },
        other => panic!("not an aux legend channel: {other}"),
    };
    let xy = Encoding {
        x: Some(EncodingSpec { field: "x".into(), ..Default::default() }),
        y: Some(EncodingSpec { field: "y".into(), ..Default::default() }),
        ..Default::default()
    };
    let first = if channel == "stroke_dash" { Mark::Line } else { Mark::Point };
    let layer = |mark, encoding| crate::spec::layer::Layer { mark, encoding, ..Default::default() };
    ChartSpec {
        mark: first,
        encoding: bind(chart, xy),
        layers: Some(vec![layer(first, bind(layer0, Encoding::default())), layer(Mark::Point, Encoding::default())]),
        ..Default::default()
    }
}

/// Each aux legend (size, shape, stroke_dash) is built, titled and styled
/// from the binding its scale was built from: the first layer's resolved
/// encoding. For every channel, a plain binding, one with `title=` and one
/// with `Legend(title=)` give the same aux block on the first layer as at
/// chart level, titled field, `Kind` and `Group` respectively. A chart with
/// the channel bound nowhere is the control and draws no aux block.
#[test]
fn a_layer_level_aux_binding_draws_its_legend_like_a_chart_level_one() {
    use crate::render::chart_config::LegendStyleSpec;
    let batch = batch3();
    for (channel, field) in [("size", "x"), ("shape", "species"), ("stroke_dash", "species")] {
        let plain = EncodingSpec { field: field.into(), ..Default::default() };
        let titled = EncodingSpec { title: Some("Kind".into()), ..plain.clone() };
        let legend_titled = EncodingSpec {
            legend: Some(Box::new(LegendStyleSpec { title: Some("Group".into()), ..Default::default() })),
            ..plain.clone()
        };
        for (binding, title) in [(&plain, field), (&titled, "Kind"), (&legend_titled, "Group")] {
            let chart_level = prep(&layered_aux_spec(channel, Some(binding.clone()), None), &batch).aux_legends;
            let layer_level = prep(&layered_aux_spec(channel, None, Some(binding.clone())), &batch).aux_legends;
            assert_eq!(chart_level.len(), 1, "{channel}: {chart_level:?}");
            let got = match &chart_level[0] {
                AuxLegendInput::Size { title, .. } if channel == "size" => title,
                AuxLegendInput::Shape { title, .. } if channel == "shape" => title,
                AuxLegendInput::StrokeDash { title, .. } if channel == "stroke_dash" => title,
                other => panic!("{channel}: wrong block {other:?}"),
            };
            assert_eq!(got.as_deref(), Some(title), "{channel}");
            assert_eq!(layer_level, chart_level, "{channel} {title}");
        }
        let unbound = prep(&layered_aux_spec(channel, None, None), &batch);
        assert!(unbound.aux_legends.is_empty(), "{channel}: {:?}", unbound.aux_legends);
    }
}

// ── `inherited_positional` ───────────────────────────────────────────────
//
// `Layer::inherited_positional` names a positional channel Python's `+`
// routing fills onto a routed layer's own encoding from the RHS operand's
// chart-level encoding (so a layer that gets a `data_source` — and so loses
// `inherit_non_positional`'s ordinary chart-level inheritance — still has
// something to plot), while recording that the layer never declared that
// channel itself. `Layer::completed_encoding` reads it so a filled channel
// is an inherited one, kept only when the layer's mark draws it — the same
// distinction `color_is_own` draws between a chart-inherited and a
// layer-declared channel, extended to a channel Python filled.

/// A `Mark::Rule` layer routed to its own `data_source` (so ordinary
/// chart-level positional inheritance does NOT apply, per
/// `from_chart_and_layer`'s `data_source.is_some()` branch): it declares its
/// own `x`, and carries a `y` value Python filled from the RHS operand's
/// chart-level encoding. Listed in `inherited_positional`, `y` is inherited,
/// and a rule declaring only `x` draws no inherited `y`: completion drops it,
/// exactly the standalone-inherited shape.
#[test]
fn a_filled_positional_channel_listed_as_inherited_is_not_own() {
    let x = EncodingSpec { field: "a".into(), ..Default::default() };
    let y = EncodingSpec { field: "b".into(), ..Default::default() };
    let spec = ChartSpec {
        mark: Mark::Rule,
        encoding: Encoding { x: Some(x.clone()), y: Some(y.clone()), ..Default::default() },
        ..Default::default()
    };
    let layer = crate::spec::layer::Layer {
        mark: Mark::Rule,
        encoding: Encoding { x: Some(x), y: Some(y), ..Default::default() },
        data_source: Some("tagged".into()),
        inherited_positional: vec![crate::spec::layer::PositionalChannel::Y],
        ..Default::default()
    };
    let prepared = LayerPrepared::from_chart_and_layer(&spec, &layer);
    assert!(prepared.encoding.x.is_some(), "x was declared by this layer");
    assert!(prepared.encoding.y.is_none(), "y is listed in inherited_positional, and a one-axis rule draws none");
}

/// The mutation guard: the identical layer WITHOUT the `inherited_positional`
/// entry — same `data_source`, same filled `y` value — keeps `y` as its
/// own, proving the field (not the data_source or the value) is what
/// decides it.
#[test]
fn the_same_filled_channel_without_the_list_reads_as_own() {
    let x = EncodingSpec { field: "a".into(), ..Default::default() };
    let y = EncodingSpec { field: "b".into(), ..Default::default() };
    let spec = ChartSpec {
        mark: Mark::Rule,
        encoding: Encoding { x: Some(x.clone()), y: Some(y.clone()), ..Default::default() },
        ..Default::default()
    };
    let layer = crate::spec::layer::Layer {
        mark: Mark::Rule,
        encoding: Encoding { x: Some(x), y: Some(y), ..Default::default() },
        data_source: Some("tagged".into()),
        inherited_positional: vec![],
        ..Default::default()
    };
    let prepared = LayerPrepared::from_chart_and_layer(&spec, &layer);
    assert!(prepared.encoding.x.is_some());
    assert!(prepared.encoding.y.is_some(), "with no inherited_positional entry, a present y is own");
}

// The end-to-end rendered-geometry consequence (a rule drawing the same
// vertical span standalone inheritance draws) is pinned in `scene_build`'s
// own test module, next to the fixture and helpers
// (`line_endpoints`, `build_scene_for`) it reuses — see
// `a_filled_and_listed_y_renders_the_same_vertical_rule_as_standalone_inheritance`.

// ── `inherited_channels` ─────────────────────────────────────────────────
//
// `Layer::inherited_channels` names a non-positional channel Python wrote
// onto a layer's own encoding from its parent. `from_chart_and_layer` reads
// it so `color_is_own`, `size_is_own` and `stroke_width_is_own` mean
// "declared by this layer", as `inherited_positional` does for x/y. The
// rendered consequences (a text label and a literal-paint line not taking
// the color) are pinned in `scene_build`'s test module.

/// A layer whose own encoding carries `color`, `size` and `stroke_width`,
/// listing `inherited` in `inherited_channels`, under a chart that binds none
/// of them, so the layer's own encoding is the only place they sit.
fn layer_carrying_channels(
    mark: Mark,
    inherited: Vec<crate::spec::layer::InheritedChannel>,
) -> LayerPrepared {
    let field = |f: &str| Some(EncodingSpec { field: f.into(), ..Default::default() });
    let spec = ChartSpec { mark, ..Default::default() };
    let layer = crate::spec::layer::Layer {
        mark,
        encoding: Encoding {
            x: field("x"),
            y: field("y"),
            color: field("c"),
            size: field("z"),
            stroke_width: field("w"),
            ..Default::default()
        },
        inherited_channels: inherited,
        ..Default::default()
    };
    LayerPrepared::from_chart_and_layer(&spec, &layer)
}

#[test]
fn a_channel_listed_as_inherited_is_not_own() {
    use crate::spec::layer::InheritedChannel;
    for (listed, expect_color, expect_size, expect_width) in [
        (InheritedChannel::Color, false, true, true),
        (InheritedChannel::Size, true, false, true),
        (InheritedChannel::StrokeWidth, true, true, false),
    ] {
        let prepared = layer_carrying_channels(Mark::Point, vec![listed]);
        assert_eq!(prepared.color_is_own, expect_color, "{listed:?} listed: color_is_own");
        assert_eq!(prepared.size_is_own, expect_size, "{listed:?} listed: size_is_own");
        assert_eq!(prepared.stroke_width_is_own, expect_width, "{listed:?} listed: stroke_width_is_own");
        assert!(
            prepared.encoding.color.is_some()
                && prepared.encoding.size.is_some()
                && prepared.encoding.stroke_width.is_some(),
            "{listed:?} listed: the carried values themselves are untouched"
        );
    }
}

/// The mutation guard: the identical layer with nothing listed reads every
/// carried channel as its own.
#[test]
fn the_same_carried_channels_without_the_list_read_as_own() {
    let prepared = layer_carrying_channels(Mark::Point, vec![]);
    assert!(prepared.color_is_own);
    assert!(prepared.size_is_own);
    assert!(prepared.stroke_width_is_own);
}

/// A line that owns `stroke_width` yields a listed `size` to it, exactly as it
/// yields a chart-inherited one, so the size is neither drawn nor counted by
/// the size legend. Unlisted, the line owns both widths and yields neither.
#[test]
fn a_line_yields_a_listed_size_to_its_own_stroke_width() {
    use crate::render::prepare::YieldedWidths;
    use crate::spec::layer::InheritedChannel;
    let listed = layer_carrying_channels(Mark::Line, vec![InheritedChannel::Size]);
    assert_eq!(listed.yielded_widths(), YieldedWidths { size: true, stroke_width: false });
    assert_eq!(listed.drawn_size_field(), None, "a yielded size is not drawn");

    let unlisted = layer_carrying_channels(Mark::Line, vec![]);
    assert_eq!(unlisted.yielded_widths(), YieldedWidths::default());
}

/// The flat render's prepare pass resolves a layered axis against layer 0's
/// rendering encoding, while a layer on the chart's data that binds nothing
/// there draws the chart-level field: `(P(x="a") + P(y only)).encode(x="c")`.
/// The x axis must reach the second layer's `c` (100..300). The rejected reading
/// takes the inherited field from the rendering encoding (layer 0's `a`) and
/// resolves 1..3.
#[test]
fn a_layer_inheriting_a_re_encoded_chart_field_widens_the_axis() {
    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;
    let spec: ChartSpec = serde_json::from_str(
        r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"c"},"y":{"field":"b"}},"layers":[{"mark":"point","encoding":{"x":{"field":"a"},"y":{"field":"b"}}},{"mark":"point","encoding":{"y":{"field":"b"}}}]}"#,
    )
    .expect("layered spec JSON must deserialize");
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Float64, false),
        Field::new("b", DataType::Float64, false),
        Field::new("c", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![100.0, 200.0, 300.0])),
        ],
    )
    .unwrap();
    let prep = prepare_render_inputs(
        &spec,
        &batch,
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport { width: 600.0, height: 400.0 },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .unwrap();
    let (lo, hi) = prep.provisional_scales.x.data_domain().expect("a continuous x");
    assert!(lo <= 1.0 && hi >= 300.0, "x domain {lo}..{hi} must cover a (1..3) and c (100..300)");
}

// ── an independent-y layer's y scale is its own ─────────────────────────

/// The y scale a bar's zero anchor puts on the chart encoding, as the wire
/// carries it.
fn bar_zero_anchor_scale() -> crate::spec::encoding::ScaleSpec {
    serde_json::from_value(serde_json::json!({"type": "linear", "zero": true, "systemInjected": true}))
        .expect("the anchor scale deserializes")
}

/// A layer binding `y` to `field`, with `scale` as its own.
fn y_layer(
    mark: Mark,
    field: &str,
    scale: Option<crate::spec::encoding::ScaleSpec>,
    independent_y: bool,
) -> crate::spec::layer::Layer {
    crate::spec::layer::Layer {
        mark,
        encoding: Encoding {
            y: Some(EncodingSpec { field: field.into(), scale, ..Default::default() }),
            ..Default::default()
        },
        independent_y,
        ..Default::default()
    }
}

/// A bar chart whose chart-level `y` binds `v` under the bar's zero anchor,
/// with `layers` over it.
fn anchored_bar_chart(layers: Vec<crate::spec::layer::Layer>) -> crate::spec::chart::ChartSpec {
    crate::spec::chart::ChartSpec {
        mark: Mark::Bar,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "x".into(), ..Default::default() }),
            y: Some(EncodingSpec {
                field: "v".into(),
                scale: Some(bar_zero_anchor_scale()),
                ..Default::default()
            }),
            ..Default::default()
        },
        layers: Some(layers),
        ..Default::default()
    }
}

/// An independent-y layer binding the chart's own `y` field does not take
/// the chart's y scale: the chart's scale belongs to the primary axis, and
/// the layer's own axis spans its own data. Its own scale, when it has one,
/// is kept. A shared-axis layer on the same field still inherits the
/// chart's scale, as `inherit_from` documents.
#[test]
fn an_independent_y_layer_keeps_its_own_y_scale_and_never_inherits_the_charts() {
    let own = || serde_json::from_value(serde_json::json!({"type": "linear", "zero": false})).unwrap();
    let spec = anchored_bar_chart(vec![]);

    let independent = LayerPrepared::from_chart_and_layer(&spec, &y_layer(Mark::Line, "v", None, true));
    let y = independent.encoding.y.expect("the layer binds y");
    assert_eq!(y.field, "v");
    assert_eq!(y.scale, None, "the chart's anchor scale must not reach an independent-y layer");

    let with_own = LayerPrepared::from_chart_and_layer(&spec, &y_layer(Mark::Line, "v", Some(own()), true));
    assert_eq!(with_own.encoding.y.unwrap().scale, Some(own()), "its own scale is kept");

    let shared = LayerPrepared::from_chart_and_layer(&spec, &y_layer(Mark::Line, "v", None, false));
    assert_eq!(
        shared.encoding.y.unwrap().scale,
        Some(bar_zero_anchor_scale()),
        "a shared-axis layer inherits the chart's scale on the same field"
    );
}

/// The secondary axis's resolved domain for `layer1` over a bar on `v`,
/// with `v` spanning 2..4.
fn secondary_domain_over_anchored_bar(layer1: crate::spec::layer::Layer) -> (f64, f64) {
    let spec = anchored_bar_chart(vec![y_layer(Mark::Bar, "v", Some(bar_zero_anchor_scale()), false), layer1]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Utf8, false),
        Field::new("v", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
            Arc::new(Float64Array::from(vec![2.0, 3.0, 4.0])),
        ],
    )
    .unwrap();
    let prep = prepare_render_inputs(
        &spec,
        &batch,
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport { width: 600.0, height: 400.0 },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .unwrap();
    assert_eq!(prep.secondary_y_scales.len(), 1, "one independent-y layer, one secondary axis");
    prep.secondary_y_scales[0].data_domain().expect("a continuous secondary axis")
}

/// A bar chart plus an independent-y line on the SAME field: the line's axis
/// spans its own data (2..4), as `fm.LayerChart(..., resolve={"y":
/// "independent"})` draws it, not 0..4 from the bar's inherited zero anchor.
#[test]
fn an_independent_y_line_on_the_bars_field_spans_its_own_data() {
    let (lo, hi) = secondary_domain_over_anchored_bar(y_layer(Mark::Line, "v", None, true));
    assert!(lo > 1.0, "the secondary axis starts near its data's minimum (2), not at 0: {lo}..{hi}");
    assert!(hi >= 4.0, "{lo}..{hi}");
}

/// An independent-y bar on the same field still reaches 0, and only through
/// the anchor scale it carries on its own `y` (the one the Python bar
/// producer injects); the chart's anchor no longer reaches it.
#[test]
fn an_independent_y_bar_reaches_zero_only_through_its_own_anchor() {
    let (lo, _) =
        secondary_domain_over_anchored_bar(y_layer(Mark::Bar, "v", Some(bar_zero_anchor_scale()), true));
    assert_eq!(lo, 0.0, "the bar's own anchor puts 0 in its axis");
    let (lo, _) = secondary_domain_over_anchored_bar(y_layer(Mark::Bar, "v", None, true));
    assert!(lo > 1.0, "without its own anchor, the chart's anchor does not reach it: {lo}");
}

// ── auxiliary scales, legends and titles from the layers that draw them ─────

/// Rows for the drawn-channel cases: `g` in a/b and `k` in p/q (strings),
/// `v` over 1..10 and `w` over 20..50 (numbers).
fn drawn_channel_batch() -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
            Field::new("k", DataType::Utf8, false),
            Field::new("v", DataType::Float64, false),
            Field::new("w", DataType::Float64, false),
        ])),
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
            Arc::new(Float64Array::from(vec![1.0, 3.0, 2.0, 4.0])),
            Arc::new(StringArray::from(vec!["a", "b", "a", "b"])),
            Arc::new(StringArray::from(vec!["p", "q", "p", "q"])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 10.0])),
            Arc::new(Float64Array::from(vec![20.0, 30.0, 40.0, 50.0])),
        ],
    )
    .unwrap()
}

/// `bindings` with `x`/`y` bound to the batch's `x`/`y` added.
fn with_xy(bindings: serde_json::Value) -> serde_json::Value {
    let mut encoding = serde_json::json!({"x": {"field": "x"}, "y": {"field": "y"}});
    for (channel, binding) in bindings.as_object().expect("bindings are an object") {
        encoding[channel] = binding.clone();
    }
    encoding
}

/// A layered chart whose chart-level encoding is `chart` plus `x`/`y`, with
/// one layer per `(mark, own bindings)`.
fn drawn_channel_spec(chart: serde_json::Value, layers: &[(&str, serde_json::Value)]) -> ChartSpec {
    let layers: Vec<serde_json::Value> = layers
        .iter()
        .map(|(mark, own)| serde_json::json!({"mark": mark, "encoding": own}))
        .collect();
    serde_json::from_value(serde_json::json!({
        "mark": layers[0]["mark"],
        "encoding": with_xy(chart),
        "layers": layers,
    }))
    .expect("the layered spec deserializes")
}

/// The unlayered chart of one `mark` drawing `bindings` plus `x`/`y`: what a
/// layer drawing those bindings draws alone.
fn drawn_channel_alone(mark: &str, bindings: serde_json::Value) -> ChartSpec {
    serde_json::from_value(serde_json::json!({"mark": mark, "encoding": with_xy(bindings)}))
        .expect("the flat spec deserializes")
}

fn drawn_channel_prepared(spec: &ChartSpec) -> PreparedInputs {
    pyo3::Python::initialize();
    prepare_render_inputs(
        spec,
        &drawn_channel_batch(),
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport { width: 600.0, height: 400.0 },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .expect("the spec prepares")
}

/// The full `prepare → layout → build_scene` render of `spec` over
/// [`drawn_channel_batch`], with every warning the render reports.
fn drawn_channel_scene(spec: &ChartSpec) -> (ferrum_scene::SceneGraph, Vec<RenderWarning>) {
    scene_over(spec, &drawn_channel_batch(), &crate::layout::ThemeInputs::default())
}

/// [`drawn_channel_scene`] over `batch` under `theme`.
fn scene_over(
    spec: &ChartSpec,
    batch: &RecordBatch,
    theme: &crate::layout::ThemeInputs,
) -> (ferrum_scene::SceneGraph, Vec<RenderWarning>) {
    pyo3::Python::initialize();
    let chart_config = crate::render::chart_config::ChartConfig::default();
    let crate::render::PipelineOutput { prep, layout, applied, mut warnings } = crate::render::prepare_and_layout(
        spec,
        batch,
        theme,
        crate::layout::Viewport { width: 600.0, height: 400.0 },
        &chart_config,
        None,
    )
    .expect("the spec prepares and lays out");
    let scene = crate::render::scene_build::build_scene(
        spec,
        &prep,
        &layout,
        &applied.effective_theme,
        &crate::render::config::RenderConfig::default(),
        &mut warnings,
        &chart_config,
        None,
    )
    .expect("the spec builds a scene");
    (scene, warnings)
}

/// Every `LayerScaleExcluded` among `warnings`.
fn scale_exclusions(warnings: &[RenderWarning]) -> Vec<RenderWarning> {
    warnings.iter().filter(|w| matches!(w, RenderWarning::LayerScaleExcluded { .. })).cloned().collect()
}

/// The one exclusion of `layer` from the `channel` scale `establishing` set.
fn excluded(layer: usize, establishing: usize, channel: &str) -> Vec<RenderWarning> {
    vec![RenderWarning::LayerScaleExcluded {
        layer,
        establishing,
        channel: channel.to_string(),
        placement: crate::render::ExcludedLayerPlacement::SharedAxis,
    }]
}

/// The `Debug` form of the chart's resolved scale for an auxiliary `channel`
/// (`"None"` when it resolves none).
fn aux_scale_debug(scales: &ResolvedScales, channel: &str) -> String {
    match channel {
        "color" => format!("{:?}", scales.color),
        "size" => format!("{:?}", scales.size),
        "opacity" => format!("{:?}", scales.opacity),
        "fill_opacity" => format!("{:?}", scales.fill_opacity),
        "stroke_opacity" => format!("{:?}", scales.stroke_opacity),
        "shape" => format!("{:?}", scales.shape),
        "stroke_dash" => format!("{:?}", scales.stroke_dash),
        "stroke_width" => format!("{:?}", scales.stroke_width),
        other => panic!("{other} resolves no auxiliary scale"),
    }
}

/// The scale and every legend input a prepared chart draws for `channel`.
type DrawnChannelOutputs = (
    String,
    Vec<crate::layout::LegendEntry>,
    Option<String>,
    Option<crate::layout::ColorbarInput>,
    Vec<AuxLegendInput>,
);

fn drawn_channel_outputs(prep: &PreparedInputs, channel: &str) -> DrawnChannelOutputs {
    (
        aux_scale_debug(&prep.provisional_scales, channel),
        prep.legend_entries.clone(),
        prep.legend_title.clone(),
        prep.colorbar.clone(),
        prep.aux_legends.clone(),
    )
}

/// Every auxiliary-scale channel with a binding it scales, as `(mark,
/// channel, binding)`: color both categorical and continuous.
fn drawn_channel_cases() -> Vec<(&'static str, &'static str, serde_json::Value)> {
    let field = |name: &str| serde_json::json!({"field": name});
    vec![
        ("point", "color", field("g")),
        ("point", "color", serde_json::json!({"field": "v", "type": "quantitative"})),
        ("point", "size", field("v")),
        ("point", "opacity", field("v")),
        ("point", "fill_opacity", field("v")),
        ("point", "stroke_opacity", field("v")),
        ("point", "shape", field("g")),
        ("line", "stroke_dash", field("g")),
        ("line", "stroke_width", field("w")),
    ]
}

fn bound(channel: &str, binding: &serde_json::Value) -> serde_json::Value {
    serde_json::json!({ channel: binding })
}

/// A channel only layer 1 draws (neither the chart level nor layer 0 binds
/// it) gets the scale and the legend layer 1 draws alone, on every
/// auxiliary-scale channel. Rejects scale existence read off the chart-level
/// (or layer 0's) encoding, which builds no scale and no legend here.
#[test]
fn a_channel_only_a_later_layer_draws_gets_the_scale_and_legend_that_layer_draws_alone() {
    for (mark, channel, binding) in drawn_channel_cases() {
        let layered = drawn_channel_spec(
            serde_json::json!({}),
            &[(mark, serde_json::json!({})), (mark, bound(channel, &binding))],
        );
        let alone = drawn_channel_alone(mark, bound(channel, &binding));
        let alone_outputs = drawn_channel_outputs(&drawn_channel_prepared(&alone), channel);
        assert_ne!(alone_outputs.0, "None", "{channel} {binding}: the standalone chart scales it");
        assert_eq!(
            drawn_channel_outputs(&drawn_channel_prepared(&layered), channel),
            alone_outputs,
            "{channel} {binding}"
        );
    }
}

/// The scene of a chart whose layer 1 alone binds color draws layer 1's
/// points exactly as that layer draws them alone, and leaves layer 0's points
/// in the theme's default fill: the scale exists without reaching a layer
/// that does not draw the channel.
#[test]
fn a_later_layers_own_color_draws_its_marks_and_leaves_layer_zero_uncolored() {
    use ferrum_scene::SceneNode;
    let color = serde_json::json!({"field": "g"});
    let layered = drawn_channel_spec(
        serde_json::json!({}),
        &[("point", serde_json::json!({})), ("point", bound("color", &color))],
    );
    let (scene, _) = drawn_channel_scene(&layered);
    let (alone, _) = drawn_channel_scene(&drawn_channel_alone("point", bound("color", &color)));
    let marks = &scene.panels[0].marks;
    assert_eq!(marks.len(), 2, "one mark batch per layer");
    assert_eq!(marks[1].nodes, alone.panels[0].marks[0].nodes, "layer 1 draws as it does alone");
    let mark_color = crate::render::draw::to_scene_color(crate::layout::ThemeInputs::default().colors.mark_color);
    let layer0_fills: Vec<_> = marks[0]
        .nodes
        .iter()
        .filter_map(|n| if let SceneNode::Circle { style, .. } = n { Some(style.fill) } else { None })
        .collect();
    assert_eq!(layer0_fills, vec![Some(mark_color); 4], "layer 0 binds no color");
}

/// `stroke_width` draws its column's values as widths through no scale on any
/// chart, so a layer drawing it alone draws the widths it draws standalone.
/// A control: there is no scale or legend for it to gain.
#[test]
fn a_later_layers_own_stroke_width_draws_as_it_does_alone() {
    let width = serde_json::json!({"field": "v"});
    let layered = drawn_channel_spec(
        serde_json::json!({}),
        &[("line", serde_json::json!({})), ("line", bound("stroke_width", &width))],
    );
    let (scene, _) = drawn_channel_scene(&layered);
    let (alone, _) = drawn_channel_scene(&drawn_channel_alone("line", bound("stroke_width", &width)));
    assert_eq!(scene.panels[0].marks[1].nodes, alone.panels[0].marks[0].nodes);
}

/// The scale's kind: layer 0 sets it when it draws the channel, otherwise the
/// first layer that does, and a later layer whose column reads the other way
/// is excluded with one `LayerScaleExcluded`, in both orders. Rejects reading
/// the kind off a later layer, or off the last drawing layer.
#[test]
fn the_first_layer_drawing_color_sets_its_kind_and_a_conflicting_later_layer_is_excluded() {
    let nominal = serde_json::json!({"field": "g"});
    let quantitative = serde_json::json!({"field": "v", "type": "quantitative"});
    let none = serde_json::json!({});

    let spec = drawn_channel_spec(
        none.clone(),
        &[("point", none.clone()), ("point", bound("color", &nominal)), ("point", bound("color", &quantitative))],
    );
    let (scene, warnings) = drawn_channel_scene(&spec);
    assert_eq!(scale_exclusions(&warnings), excluded(2, 1, "color"));
    let prep = drawn_channel_prepared(&spec);
    match &prep.provisional_scales.color {
        Some(crate::render::scale_resolve::ColorScale::Categorical { domain, .. }) => {
            assert_eq!(domain, &vec!["a".to_string(), "b".to_string()]);
        }
        other => panic!("layer 1's nominal g establishes a categorical scale: {other:?}"),
    }
    assert!(!scene.legend.is_empty());

    let spec = drawn_channel_spec(
        none.clone(),
        &[("point", none.clone()), ("point", bound("color", &quantitative)), ("point", bound("color", &nominal))],
    );
    let (_, warnings) = drawn_channel_scene(&spec);
    assert_eq!(scale_exclusions(&warnings), excluded(2, 1, "color"));
    match &drawn_channel_prepared(&spec).provisional_scales.color {
        Some(crate::render::scale_resolve::ColorScale::Continuous { domain, .. }) => assert_eq!(*domain, (1.0, 10.0)),
        other => panic!("layer 1's quantitative v establishes a continuous scale: {other:?}"),
    }

    // Control: layer 0 draws the channel, so layer 0 establishes it.
    let spec = drawn_channel_spec(
        none.clone(),
        &[("point", bound("color", &nominal)), ("point", none.clone()), ("point", bound("color", &quantitative))],
    );
    let (_, warnings) = drawn_channel_scene(&spec);
    assert_eq!(scale_exclusions(&warnings), excluded(2, 0, "color"));
}

/// The same establishing rule on the other kind-bearing channels: size and
/// the opacity family are numeric, so a later string column is excluded;
/// stroke-dash is categorical over a string column and an index palette over
/// a numeric one, in both orders.
#[test]
fn the_first_drawing_layer_sets_the_kind_of_every_kind_bearing_auxiliary_channel() {
    let none = serde_json::json!({});
    let numeric = serde_json::json!({"field": "v"});
    let string = serde_json::json!({"field": "g"});
    for channel in ["size", "opacity", "fill_opacity", "stroke_opacity"] {
        let spec = drawn_channel_spec(
            none.clone(),
            &[("point", none.clone()), ("point", bound(channel, &numeric)), ("point", bound(channel, &string))],
        );
        let (_, warnings) = drawn_channel_scene(&spec);
        assert_eq!(scale_exclusions(&warnings), excluded(2, 1, channel), "{channel}");
        let scale = aux_scale_debug(&drawn_channel_prepared(&spec).provisional_scales, channel);
        assert_ne!(scale, "None", "{channel}");
    }

    let spec = drawn_channel_spec(
        none.clone(),
        &[("line", none.clone()), ("line", bound("stroke_dash", &string)), ("line", bound("stroke_dash", &numeric))],
    );
    let (_, warnings) = drawn_channel_scene(&spec);
    assert_eq!(scale_exclusions(&warnings), excluded(2, 1, "stroke_dash"));
    let dash = drawn_channel_prepared(&spec).provisional_scales.stroke_dash.expect("a categorical dash scale");
    assert_eq!(dash.domain, vec!["a".to_string(), "b".to_string()]);

    let spec = drawn_channel_spec(
        none.clone(),
        &[("line", none.clone()), ("line", bound("stroke_dash", &numeric)), ("line", bound("stroke_dash", &string))],
    );
    let (_, warnings) = drawn_channel_scene(&spec);
    assert_eq!(scale_exclusions(&warnings), excluded(2, 1, "stroke_dash"));
    assert!(drawn_channel_prepared(&spec).provisional_scales.stroke_dash.is_none(), "numeric dash reads indices");
}

/// A layered chart over [`drawn_channel_batch`] whose named output `agg`
/// holds, per `groupby`, the mean `x` and `y` and the sum of `v` named
/// `sum_as`: per `g` the sums are a: 4, b: 12, and per `k` p: 4, q: 12.
fn drawn_channel_spec_with_output(
    groupby: &str,
    sum_as: &str,
    layers: &[(&str, Option<&str>, serde_json::Value)],
) -> ChartSpec {
    use crate::transform::aggregate::{AggFn, AggregateOp, AggregateSpec};
    use crate::transform::core::TransformSpec;
    let op = |field: &str, fn_, as_: &str| AggregateOp { field: field.into(), fn_, as_: as_.into() };
    let layers: Vec<serde_json::Value> = layers
        .iter()
        .map(|(mark, source, own)| match source {
            Some(source) => serde_json::json!({"mark": mark, "data_source": source, "encoding": with_xy(own.clone())}),
            None => serde_json::json!({"mark": mark, "encoding": own}),
        })
        .collect();
    let mut spec: ChartSpec = serde_json::from_value(serde_json::json!({
        "mark": layers[0]["mark"],
        "encoding": with_xy(serde_json::json!({})),
        "layers": layers,
    }))
    .expect("the layered spec deserializes");
    spec.transforms = vec![TransformSpec::Aggregate(AggregateSpec {
        ops: vec![op("x", AggFn::Mean, "x"), op("y", AggFn::Mean, "y"), op("v", AggFn::Sum, sum_as)],
        groupby: vec![groupby.into()],
        name: Some("agg".into()),
    })];
    spec
}

/// The domain is the union of what every admitted layer draws, each read from
/// the batch that layer draws: numeric extents, and categories in layer
/// order. Rejects a domain read from the establishing layer alone, or from
/// the chart's rows for a layer drawing its own output.
#[test]
fn an_auxiliary_domain_is_the_union_over_every_drawing_layer() {
    let none = serde_json::json!({});
    let field = |name: &str| serde_json::json!({"field": name});

    let spec = drawn_channel_spec(
        none.clone(),
        &[("point", none.clone()), ("point", bound("opacity", &field("v"))), ("point", bound("opacity", &field("w")))],
    );
    let opacity = drawn_channel_prepared(&spec).provisional_scales.opacity.expect("an opacity scale");
    assert_eq!(opacity.inner.data_domain(), Some((1.0, 50.0)));

    for (first, second, want) in [("g", "k", ["a", "b", "p", "q"]), ("k", "g", ["p", "q", "a", "b"])] {
        let spec = drawn_channel_spec(
            none.clone(),
            &[("point", none.clone()), ("point", bound("color", &field(first))), ("point", bound("color", &field(second)))],
        );
        match drawn_channel_prepared(&spec).provisional_scales.color {
            Some(crate::render::scale_resolve::ColorScale::Categorical { domain, .. }) => {
                assert_eq!(domain, want.map(String::from).to_vec(), "{first} then {second}");
            }
            other => panic!("a categorical color: {other:?}"),
        }
    }

    // Layer 1 draws `agg`'s sum_v (4..12) and establishes; layer 2 draws the
    // chart's v (1..10).
    let spec = drawn_channel_spec_with_output("g", "sum_v", &[
        ("point", None, none.clone()),
        ("point", Some("agg"), bound("opacity", &field("sum_v"))),
        ("point", None, bound("opacity", &field("v"))),
    ]);
    let opacity = drawn_channel_prepared(&spec).provisional_scales.opacity.expect("an opacity scale");
    assert_eq!(opacity.inner.data_domain(), Some((1.0, 12.0)));
}

/// Shape and stroke-dash read every layer that draws them, not only the
/// chart-level (or layer 0's) binding: layer 0's `g` and layer 1's `k` make
/// one domain. Rejects the domain of layer 0's column alone.
#[test]
fn shape_and_stroke_dash_domains_read_every_drawing_layer() {
    let g = serde_json::json!({"field": "g"});
    let k = serde_json::json!({"field": "k"});
    let want: Vec<String> = ["a", "b", "p", "q"].map(String::from).to_vec();

    let spec = drawn_channel_spec(
        serde_json::json!({}),
        &[("point", bound("shape", &g)), ("point", bound("shape", &k))],
    );
    let shape = drawn_channel_prepared(&spec).provisional_scales.shape.expect("a shape scale");
    assert_eq!(shape.domain, want);

    let spec = drawn_channel_spec(
        serde_json::json!({}),
        &[("line", bound("stroke_dash", &g)), ("line", bound("stroke_dash", &k))],
    );
    let dash = drawn_channel_prepared(&spec).provisional_scales.stroke_dash.expect("a dash scale");
    assert_eq!(dash.domain, want);
}

/// A legend's title is the establishing layer's binding title, else its
/// field, never a later layer's. Rejects a title read off the last drawing
/// layer.
#[test]
fn a_legend_title_comes_from_the_establishing_layers_binding() {
    let none = serde_json::json!({});
    let titled = |field: &str, title: &str| serde_json::json!({"field": field, "title": title});
    let field = |name: &str| serde_json::json!({"field": name});

    let spec = drawn_channel_spec(
        none.clone(),
        &[
            ("point", none.clone()),
            ("point", bound("color", &titled("g", "Group"))),
            ("point", bound("color", &titled("k", "Other"))),
        ],
    );
    assert_eq!(drawn_channel_prepared(&spec).legend_title.as_deref(), Some("Group"));

    let spec = drawn_channel_spec(
        none.clone(),
        &[("point", none.clone()), ("point", bound("color", &field("g"))), ("point", bound("color", &titled("k", "Other")))],
    );
    assert_eq!(drawn_channel_prepared(&spec).legend_title.as_deref(), Some("g"));

    let spec = drawn_channel_spec(
        none.clone(),
        &[("point", none.clone()), ("point", bound("size", &field("v"))), ("point", bound("size", &titled("w", "W")))],
    );
    let titles: Vec<Option<String>> = drawn_channel_prepared(&spec)
        .aux_legends
        .iter()
        .map(|block| match block {
            AuxLegendInput::Size { title, .. }
            | AuxLegendInput::Shape { title, .. }
            | AuxLegendInput::StrokeDash { title, .. }
            | AuxLegendInput::Opacity { title, .. }
            | AuxLegendInput::StrokeWidth { title, .. } => title.clone(),
        })
        .collect();
    assert_eq!(titles, vec![Some("v".to_string())]);
}

/// A control: a binding on the chart-level encoding renders the same scene
/// as the same binding moved onto layer 0's own encoding, on every
/// auxiliary-scale channel (layer 1 binds it itself in both, so the layers
/// that draw it are the same).
#[test]
fn a_chart_level_binding_renders_as_the_same_binding_on_layer_zero() {
    for (mark, channel, binding) in drawn_channel_cases() {
        let own = bound(channel, &binding);
        let chart_level = drawn_channel_spec(own.clone(), &[(mark, serde_json::json!({})), (mark, own.clone())]);
        let on_layer_zero = drawn_channel_spec(serde_json::json!({}), &[(mark, own.clone()), (mark, own.clone())]);
        assert_eq!(drawn_channel_scene(&chart_level).0, drawn_channel_scene(&on_layer_zero).0, "{channel} {binding}");
    }
}

/// A control: an independent-y layer keeps its own y scale whether or not it
/// also draws a color no other layer draws.
#[test]
fn an_independent_y_layer_keeps_its_own_y_scale_beside_its_own_color() {
    let secondary = |own: serde_json::Value| {
        let mut spec = drawn_channel_spec(
            serde_json::json!({}),
            &[("point", serde_json::json!({})), ("point", with_xy(own))],
        );
        let layer = &mut spec.layers.as_mut().unwrap()[1];
        layer.independent_y = true;
        layer.encoding.y = Some(EncodingSpec { field: "w".into(), ..Default::default() });
        let prep = drawn_channel_prepared(&spec);
        assert_eq!(prep.secondary_y_scales.len(), 1);
        prep.secondary_y_scales[0].data_domain().expect("a continuous secondary axis")
    };
    let plain = secondary(serde_json::json!({}));
    assert!(plain.0 > 10.0, "the secondary axis spans w, not y: {plain:?}");
    assert_eq!(secondary(serde_json::json!({"color": {"field": "g"}})), plain);
}

/// The establishing layer's column decides the kind, read from the batch that
/// layer draws: layer 1 draws `agg`'s `g`, the numeric per-`k` sum of `v`
/// (4, 12), while the chart's own `g` is strings. So color is continuous over
/// 4..12 and stroke-dash reads the column as palette indices (no scale).
/// Rejects locating the field primary-first, which finds the chart's string
/// `g` and builds a categorical scale for a layer that never draws it.
#[test]
fn the_establishing_layers_own_batch_decides_the_kind_over_a_same_named_chart_column() {
    let none = serde_json::json!({});
    let g = serde_json::json!({"field": "g"});

    let spec = drawn_channel_spec_with_output("k", "g", &[("point", None, none.clone()), ("point", Some("agg"), bound("color", &g))]);
    match drawn_channel_prepared(&spec).provisional_scales.color {
        Some(crate::render::scale_resolve::ColorScale::Continuous { domain, .. }) => assert_eq!(domain, (4.0, 12.0)),
        other => panic!("agg's numeric g establishes a continuous color: {other:?}"),
    }

    let spec =
        drawn_channel_spec_with_output("k", "g", &[("line", None, none.clone()), ("line", Some("agg"), bound("stroke_dash", &g))]);
    assert!(
        drawn_channel_prepared(&spec).provisional_scales.stroke_dash.is_none(),
        "agg's numeric g reads as palette indices"
    );
}

// ── Opacity-family and stroke-width legends, and the stroke-width scale ──────

/// Every node of `nodes`, groups flattened, in drawing order.
fn flat_nodes(nodes: &[ferrum_scene::SceneNode]) -> Vec<&ferrum_scene::SceneNode> {
    let mut out = Vec::new();
    for node in nodes {
        match node {
            ferrum_scene::SceneNode::Group { children, .. } => out.extend(flat_nodes(children)),
            other => out.push(other),
        }
    }
    out
}

/// The alpha every circle swatch in the scene's legend paints, in order: a
/// filled swatch's fill alpha (checked against the style's opacity, which
/// carries the same value), an unfilled swatch's stroke alpha, each to the
/// nearest 8-bit alpha step the SVG paints.
fn legend_circle_opacities(scene: &ferrum_scene::SceneGraph) -> Vec<f64> {
    flat_nodes(&scene.legend)
        .into_iter()
        .filter_map(|n| match n {
            ferrum_scene::SceneNode::Circle { style, .. } => Some(match (&style.fill, &style.stroke) {
                (Some(fill), _) => {
                    let fill_alpha = f64::from(fill.a) / 255.0;
                    assert!((fill_alpha - style.opacity).abs() <= 0.5 / 255.0, "fill alpha {fill_alpha} vs opacity {}", style.opacity);
                    style.opacity
                }
                (None, Some(stroke)) => f64::from(stroke.a) / 255.0,
                (None, None) => panic!("a swatch paints nothing: {style:?}"),
            }),
            _ => None,
        })
        .collect()
}

/// The stroke width of every line swatch in the scene's legend, in order.
fn legend_line_widths(scene: &ferrum_scene::SceneGraph) -> Vec<f64> {
    flat_nodes(&scene.legend)
        .into_iter()
        .filter_map(|n| match n {
            ferrum_scene::SceneNode::Line { style, .. } => Some(style.width),
            _ => None,
        })
        .collect()
}

/// The text of the scene's legend, in order.
fn legend_text(scene: &ferrum_scene::SceneGraph) -> Vec<String> {
    flat_nodes(&scene.legend)
        .into_iter()
        .filter_map(|n| match n {
            ferrum_scene::SceneNode::Text { content, .. } => Some(content.clone()),
            _ => None,
        })
        .collect()
}

/// The stroke width each point of the first panel's first mark batch draws.
fn point_stroke_widths(scene: &ferrum_scene::SceneGraph) -> Vec<f64> {
    scene.panels[0].marks[0]
        .nodes
        .iter()
        .map(|n| match n {
            ferrum_scene::SceneNode::Circle { style, .. } => style.stroke_width,
            other => panic!("a point draws a circle: {other:?}"),
        })
        .collect()
}

/// The width of every polyline of the first panel, in mark-batch order.
fn polyline_widths(scene: &ferrum_scene::SceneGraph) -> Vec<f64> {
    scene.panels[0]
        .marks
        .iter()
        .flat_map(|batch| batch.nodes.iter())
        .filter_map(|n| match n {
            ferrum_scene::SceneNode::Polyline { style, .. } => Some(style.width),
            _ => None,
        })
        .collect()
}

/// `values` mapped linearly from `domain` onto `range`: the oracle every
/// opacity and stroke-width expectation below is computed from.
fn linear_map(values: &[f64], domain: (f64, f64), range: (f64, f64)) -> Vec<f64> {
    values.iter().map(|v| range.0 + (v - domain.0) / (domain.1 - domain.0) * (range.1 - range.0)).collect()
}

fn assert_close(actual: &[f64], expected: &[f64], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: {actual:?} vs {expected:?}");
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < 1e-9, "{context}: {actual:?} vs {expected:?}");
    }
}

/// [`assert_close`] for alphas a painted color carries, as one of 256 steps.
fn assert_painted_close(actual: &[f64], expected: &[f64], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: {actual:?} vs {expected:?}");
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() <= 0.5 / 255.0 + 1e-12, "{context}: {actual:?} vs {expected:?}");
    }
}

/// The nice values of `w`'s 20..50 extent a stroke-width legend lists.
const NICE_W: [&str; 7] = ["20", "25", "30", "35", "40", "45", "50"];

/// The widths [`NICE_W`] map to on the default 1..4 px band.
fn nice_w_widths() -> Vec<f64> {
    linear_map(&[20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0], (20.0, 50.0), (1.0, 4.0))
}

/// The only aux legend block of a prepared chart.
fn only_aux_block(prep: &PreparedInputs) -> &AuxLegendInput {
    match prep.aux_legends.as_slice() {
        [block] => block,
        other => panic!("expected one aux legend block, got {other:?}"),
    }
}

/// Each opacity-family channel bound to a quantitative field draws one
/// legend from its scale, as Vega-Lite's opacity legend does: a circle at
/// each nice value of the domain (`v` spans 1..10, so 2, 4, 6, 8, 10), each
/// drawn at the alpha that value maps to on the theme band `[0.1, 1.0]`,
/// titled by its field. With all three bound there are three legends, one
/// per channel. Rejects a legend that does not exist (the channels drew none
/// before) and one whose swatches are opaque.
#[test]
fn opacity_family_channels_each_draw_one_legend_of_alphas_at_nice_values() {
    let nice = [2.0, 4.0, 6.0, 8.0, 10.0];
    let alphas = linear_map(&nice, (1.0, 10.0), (0.1, 1.0));
    for channel in ["opacity", "fill_opacity", "stroke_opacity"] {
        let spec = drawn_channel_alone("point", bound(channel, &serde_json::json!({"field": "v"})));
        match only_aux_block(&drawn_channel_prepared(&spec)) {
            AuxLegendInput::Opacity { title, entries, paint } => {
                assert_eq!(title.as_deref(), Some("v"), "{channel}");
                let faded = if channel == "stroke_opacity" { crate::layout::OpacityPaint::Stroke } else { crate::layout::OpacityPaint::Fill };
                assert_eq!(*paint, faded, "{channel}");
                let labels: Vec<&str> = entries.iter().map(|e| e.label.as_str()).collect();
                assert_eq!(labels, ["2", "4", "6", "8", "10"], "{channel}");
                assert_close(&entries.iter().map(|e| e.opacity).collect::<Vec<_>>(), &alphas, channel);
            }
            other => panic!("{channel}: expected an opacity legend, got {other:?}"),
        }
        let (scene, _) = drawn_channel_scene(&spec);
        assert_painted_close(&legend_circle_opacities(&scene), &alphas, channel);
        assert_eq!(legend_text(&scene), ["v", "2", "4", "6", "8", "10"], "{channel}");
    }
    let all = drawn_channel_alone(
        "point",
        serde_json::json!({"opacity": {"field": "v"}, "fill_opacity": {"field": "w"}, "stroke_opacity": {"field": "v", "title": "S"}}),
    );
    let titles: Vec<Option<String>> = drawn_channel_prepared(&all)
        .aux_legends
        .iter()
        .map(|block| match block {
            AuxLegendInput::Opacity { title, .. } => title.clone(),
            other => panic!("only opacity legends: {other:?}"),
        })
        .collect();
    assert_eq!(titles, [Some("v".to_string()), Some("w".to_string()), Some("S".to_string())]);
}

/// An opacity channel's explicit `scale=` domain and range set its legend's
/// entries and alphas: domain `[0, 20]` onto `[0.2, 0.8]` lists 0, 5, 10,
/// 15, 20 at `0.2 + 0.6 * value / 20`, and marks outside that domain clamp
/// onto the range's ends.
#[test]
fn an_opacity_legend_reads_an_explicit_domain_and_range() {
    let binding = serde_json::json!({
        "field": "w",
        "scale": {"type": "linear", "domain": [0.0, 20.0], "range": [0.2, 0.8]},
    });
    for channel in ["opacity", "fill_opacity", "stroke_opacity"] {
        let spec = drawn_channel_alone("point", bound(channel, &binding));
        match only_aux_block(&drawn_channel_prepared(&spec)) {
            AuxLegendInput::Opacity { entries, .. } => {
                let labels: Vec<&str> = entries.iter().map(|e| e.label.as_str()).collect();
                assert_eq!(labels, ["0", "5", "10", "15", "20"], "{channel}");
                let expected = linear_map(&[0.0, 5.0, 10.0, 15.0, 20.0], (0.0, 20.0), (0.2, 0.8));
                assert_close(&entries.iter().map(|e| e.opacity).collect::<Vec<_>>(), &expected, channel);
            }
            other => panic!("{channel}: expected an opacity legend, got {other:?}"),
        }
    }
}

/// `stroke_width` bound to a field maps through a linear scale from the
/// data extent (`w` spans 20..50) onto the theme's 1..4 px band: points draw
/// 1, 2, 3 and 4 px, and a line draws its first row's mapped 1 px. Rejects
/// drawing the raw values as pixel widths (20..50 px).
#[test]
fn a_stroke_width_field_draws_mapped_widths_never_raw_values() {
    let w = bound("stroke_width", &serde_json::json!({"field": "w"}));
    let (scene, _) = drawn_channel_scene(&drawn_channel_alone("point", w.clone()));
    assert_close(&point_stroke_widths(&scene), &linear_map(&[20.0, 30.0, 40.0, 50.0], (20.0, 50.0), (1.0, 4.0)), "point");
    let (scene, _) = drawn_channel_scene(&drawn_channel_alone("line", w));
    assert_close(&polyline_widths(&scene), &[1.0], "line");
}

/// An explicit `scale=` range overrides the theme band, an explicit domain
/// overrides the data extent, and a value outside an explicit domain draws
/// at the nearer end of the range.
#[test]
fn a_stroke_width_scale_honors_an_explicit_domain_and_range_and_clamps() {
    let w = [20.0, 30.0, 40.0, 50.0];
    let widths = |scale: serde_json::Value| {
        let spec = drawn_channel_alone("point", bound("stroke_width", &serde_json::json!({"field": "w", "scale": scale})));
        point_stroke_widths(&drawn_channel_scene(&spec).0)
    };
    assert_close(&widths(serde_json::json!({"type": "linear", "range": [2.0, 8.0]})), &linear_map(&w, (20.0, 50.0), (2.0, 8.0)), "range");
    assert_close(&widths(serde_json::json!({"type": "linear", "domain": [0.0, 100.0]})), &linear_map(&w, (0.0, 100.0), (1.0, 4.0)), "domain");
    assert_close(
        &widths(serde_json::json!({"type": "linear", "domain": [25.0, 45.0], "range": [2.0, 6.0]})),
        &[2.0, 3.0, 5.0, 6.0],
        "values outside the domain clamp",
    );
}

/// The stroke-width scale's default range is the theme's
/// `[line_stroke_width_min, line_stroke_width_max]`, 1 to 4 px by default,
/// read the way the size scale reads its point-size band: a theme with
/// another band draws another range.
#[test]
fn the_stroke_width_range_is_the_themes_line_stroke_width_band() {
    let defaults = crate::layout::ThemeInputs::default().sizes;
    assert_eq!((defaults.line_stroke_width_min, defaults.line_stroke_width_max), (1.0, 4.0));
    let mut theme = crate::layout::ThemeInputs::default();
    theme.sizes.line_stroke_width_min = 0.5;
    theme.sizes.line_stroke_width_max = 6.5;
    let spec = drawn_channel_alone("point", bound("stroke_width", &serde_json::json!({"field": "w"})));
    let (scene, _) = scene_over(&spec, &drawn_channel_batch(), &theme);
    assert_close(&point_stroke_widths(&scene), &[0.5, 2.5, 4.5, 6.5], "theme band");
}

/// A `stroke_width` field draws a legend of line samples at the widths its
/// nice domain values map to, as ggplot2's linewidth legend does: 20, 25,
/// …, 50 at 1, 1.5, …, 4 px, titled by the binding's title, else its field.
#[test]
fn a_stroke_width_field_draws_a_legend_of_line_samples_at_mapped_widths() {
    for (binding, title) in [
        (serde_json::json!({"field": "w"}), "w"),
        (serde_json::json!({"field": "w", "title": "Width"}), "Width"),
    ] {
        let spec = drawn_channel_alone("line", bound("stroke_width", &binding));
        match only_aux_block(&drawn_channel_prepared(&spec)) {
            AuxLegendInput::StrokeWidth { title: t, entries } => {
                assert_eq!(t.as_deref(), Some(title));
                let labels: Vec<&str> = entries.iter().map(|e| e.label.as_str()).collect();
                assert_eq!(labels, NICE_W);
                assert_close(&entries.iter().map(|e| e.width).collect::<Vec<_>>(), &nice_w_widths(), title);
            }
            other => panic!("expected a stroke-width legend, got {other:?}"),
        }
        let (scene, _) = drawn_channel_scene(&spec);
        assert_close(&legend_line_widths(&scene), &nice_w_widths(), title);
        let mut text = vec![title];
        text.extend(NICE_W);
        assert_eq!(legend_text(&scene), text);
    }
}

/// `legend=None` on the wire suppresses each new legend exactly as it
/// suppresses the size legend, and the scale still applies to the marks.
#[test]
fn a_disabled_legend_suppresses_each_new_legend_and_keeps_the_mapping() {
    let disabled = |field: &str| serde_json::json!({"field": field, "legend": {"disabled": true}});
    for channel in ["opacity", "fill_opacity", "stroke_opacity", "stroke_width", "size"] {
        let spec = drawn_channel_alone("point", bound(channel, &disabled("w")));
        let prep = drawn_channel_prepared(&spec);
        assert!(prep.aux_legends.is_empty(), "{channel}: {:?}", prep.aux_legends);
        assert_ne!(aux_scale_debug(&prep.provisional_scales, channel), "None", "{channel} still scales");
    }
    let spec = drawn_channel_alone("point", bound("stroke_width", &disabled("w")));
    assert_close(&point_stroke_widths(&drawn_channel_scene(&spec).0), &[1.0, 2.0, 3.0, 4.0], "mapped widths");
}

/// On a layered chart the stroke-width domain is the union over every layer
/// that draws it, each reading its own column: `v` (1..10) and `w`
/// (20..50) give one scale over 1..50, niced to 0..50, and one legend.
#[test]
fn a_layered_stroke_width_spans_every_drawing_layer() {
    let spec = drawn_channel_spec(
        serde_json::json!({}),
        &[("line", bound("stroke_width", &serde_json::json!({"field": "v"}))), ("line", bound("stroke_width", &serde_json::json!({"field": "w"})))],
    );
    let prep = drawn_channel_prepared(&spec);
    let scale = prep.provisional_scales.stroke_width.as_ref().expect("a stroke-width scale");
    assert_eq!(scale.inner.data_domain(), Some((0.0, 50.0)));
    match only_aux_block(&prep) {
        AuxLegendInput::StrokeWidth { title, entries } => {
            assert_eq!(title.as_deref(), Some("v"));
            let labels: Vec<&str> = entries.iter().map(|e| e.label.as_str()).collect();
            assert_eq!(labels, ["0", "10", "20", "30", "40", "50"]);
        }
        other => panic!("expected one stroke-width legend, got {other:?}"),
    }
    // Each line draws its first row through the one scale: v=1 and w=20.
    let (scene, _) = drawn_channel_scene(&spec);
    assert_close(&polyline_widths(&scene), &linear_map(&[1.0, 20.0], (0.0, 50.0), (1.0, 4.0)), "union");
}

/// A batch of `x`/`y` over `n` rows plus one numeric column `c` of `values`.
fn one_column_batch(values: Vec<Option<f64>>) -> RecordBatch {
    let n = values.len();
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("c", DataType::Float64, true),
        ])),
        vec![
            Arc::new(Float64Array::from((0..n).map(|i| i as f64).collect::<Vec<_>>())),
            Arc::new(Float64Array::from((0..n).map(|i| (i * i) as f64).collect::<Vec<_>>())),
            Arc::new(Float64Array::from(values)),
        ],
    )
    .unwrap()
}

/// A one-row chart and a constant column each draw a one-entry legend on
/// every new channel, and every channel follows the size scale's
/// degenerate-domain rule: `stroke_width` draws the middle of its range
/// (2.5 px) and each opacity-family channel the middle of its band (0.55).
#[test]
fn a_degenerate_domain_draws_a_one_entry_legend_on_every_new_channel() {
    let theme = crate::layout::ThemeInputs::default();
    for batch in [one_column_batch(vec![Some(0.3)]), one_column_batch(vec![Some(0.3), Some(0.3), Some(0.3)])] {
        for (channel, expected) in [("opacity", 0.55), ("fill_opacity", 0.55), ("stroke_opacity", 0.55), ("stroke_width", 2.5)] {
            let spec = drawn_channel_alone("point", bound(channel, &serde_json::json!({"field": "c"})));
            let (scene, _) = scene_over(&spec, &batch, &theme);
            let swatches = if channel == "stroke_width" { legend_line_widths(&scene) } else { legend_circle_opacities(&scene) };
            assert_painted_close(&swatches, &[expected], channel);
            assert_eq!(legend_text(&scene), ["c", "0.3"], "{channel}");
            if channel == "stroke_width" {
                assert!(point_stroke_widths(&scene).iter().all(|w| (w - 2.5).abs() < 1e-9), "{channel}");
            }
            let faded = match channel {
                "opacity" => Some(Faded::Mark),
                "fill_opacity" => Some(Faded::Fill),
                _ => None,
            };
            if let Some(faded) = faded {
                for (node, _) in drawn_rows(&scene) {
                    let painted = painted_alpha(node, faded);
                    assert!((painted - expected).abs() <= 0.5 / 255.0 + 1e-12, "{channel}: the mark draws {painted}");
                }
            }
        }
    }
}

/// A null value draws its mark at the channel's default, stays out of the
/// domain and adds no legend entry: with row 1 null, the other rows map over
/// 20..50 exactly as without it, and row 1 draws what an all-null column
/// draws for every row. An all-null column draws every mark at the default,
/// builds no scale and draws no legend.
#[test]
fn a_null_value_draws_at_the_default_and_adds_no_legend_entry() {
    let theme = crate::layout::ThemeInputs::default();
    let all_null = one_column_batch(vec![None, None, None, None]);
    let partial = one_column_batch(vec![Some(20.0), None, Some(40.0), Some(50.0)]);
    for channel in ["stroke_width", "opacity", "fill_opacity", "stroke_opacity"] {
        let spec = drawn_channel_alone("point", bound(channel, &serde_json::json!({"field": "c"})));
        pyo3::Python::initialize();
        let prep = |batch: &RecordBatch| {
            prepare_render_inputs(&spec, batch, &theme, crate::layout::Viewport { width: 600.0, height: 400.0 }, &crate::render::chart_config::ChartConfig::default(), None)
                .expect("the spec prepares")
        };
        let none = prep(&all_null);
        assert_eq!(aux_scale_debug(&none.provisional_scales, channel), "None", "{channel}: an all-null column builds no scale");
        assert!(none.aux_legends.is_empty(), "{channel}: {:?}", none.aux_legends);
        let some = prep(&partial);
        let labels: Vec<String> = match only_aux_block(&some) {
            AuxLegendInput::StrokeWidth { entries, .. } => entries.iter().map(|e| e.label.clone()).collect(),
            AuxLegendInput::Opacity { entries, .. } => entries.iter().map(|e| e.label.clone()).collect(),
            other => panic!("{channel}: {other:?}"),
        };
        assert_eq!(labels, NICE_W, "{channel}");
    }
    let spec = drawn_channel_alone("point", bound("stroke_width", &serde_json::json!({"field": "c"})));
    let default_width = point_stroke_widths(&scene_over(&spec, &all_null, &theme).0);
    let widths = point_stroke_widths(&scene_over(&spec, &partial, &theme).0);
    assert_close(&[widths[0], widths[2], widths[3]], &linear_map(&[20.0, 40.0, 50.0], (20.0, 50.0), (1.0, 4.0)), "valued rows");
    assert_eq!(widths[1], default_width[1], "the null row draws the default width");
}

/// A stroke-width binding no layer draws gets no legend: an area carries
/// the binding without drawing it, and a line yields an inherited
/// `stroke_width` to a `size` of its own.
#[test]
fn a_stroke_width_no_layer_draws_gets_no_legend() {
    let w = serde_json::json!({"field": "w"});
    let area = drawn_channel_alone("area", bound("stroke_width", &w));
    assert!(drawn_channel_prepared(&area).aux_legends.is_empty());
    let yielded = drawn_channel_spec(bound("stroke_width", &w), &[("line", bound("size", &serde_json::json!({"field": "v"})))]);
    let prep = drawn_channel_prepared(&yielded);
    assert!(
        !prep.aux_legends.iter().any(|b| matches!(b, AuxLegendInput::StrokeWidth { .. })),
        "{:?}",
        prep.aux_legends
    );
}

/// The new legends reach the interactive scene the WASM runtime paints,
/// drawn from the existing circle and line legend nodes.
#[test]
fn the_new_legends_reach_the_interactive_scene() {
    pyo3::Python::initialize();
    let spec = drawn_channel_alone(
        "point",
        serde_json::json!({"opacity": {"field": "v"}, "stroke_width": {"field": "w"}}),
    );
    let (json, _, _) = crate::render::render_scene_json(
        &spec,
        &drawn_channel_batch(),
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport { width: 600.0, height: 400.0 },
        &crate::render::config::RenderConfig::default(),
        &crate::render::chart_config::ChartConfig::default(),
    )
    .expect("the interactive scene renders");
    let scene: ferrum_scene::SceneGraph = serde_json::from_str(&json).expect("the scene parses");
    assert_painted_close(&legend_circle_opacities(&scene), &linear_map(&[2.0, 4.0, 6.0, 8.0, 10.0], (1.0, 10.0), (0.1, 1.0)), "opacity");
    assert_close(&legend_line_widths(&scene), &nice_w_widths(), "stroke_width");
}

/// A constant (`stroke_width=` kwarg) width draws exactly that width with no
/// stroke-width scale or legend, and a chart binding none of the four
/// channels builds none of their scales or legends.
#[test]
fn constant_widths_and_unbound_channels_build_no_new_scale_or_legend() {
    let mut constant = drawn_channel_alone("point", serde_json::json!({}));
    constant.mark_style = Some(crate::spec::mark_style::MarkKwargsSpec { stroke_width: Some(3.0), ..Default::default() });
    let prep = drawn_channel_prepared(&constant);
    assert!(prep.provisional_scales.stroke_width.is_none());
    assert!(prep.aux_legends.is_empty());
    assert!(point_stroke_widths(&drawn_channel_scene(&constant).0).iter().all(|w| *w == 3.0));
    let prep = drawn_channel_prepared(&drawn_channel_alone("point", serde_json::json!({})));
    for channel in ["opacity", "fill_opacity", "stroke_opacity", "stroke_width"] {
        assert_eq!(aux_scale_debug(&prep.provisional_scales, channel), "None", "{channel}");
    }
    assert!(prep.aux_legends.is_empty());
}

/// Which paint of a node an opacity-family channel fades.
#[derive(Clone, Copy, Debug)]
enum Faded {
    /// The fill where the node has one, else its stroke (`opacity`).
    Mark,
    /// The fill (`fill_opacity`).
    Fill,
    /// The stroke (`stroke_opacity`).
    Stroke,
}

/// The alpha a node paints the faded paint with, as the SVG walker paints it:
/// a fill's alpha times `fill_opacity`, a stroke's alpha times
/// `stroke_opacity`, a text's color alpha times its opacity.
fn painted_alpha(node: &ferrum_scene::SceneNode, faded: Faded) -> f64 {
    use ferrum_scene::SceneNode as N;
    let alpha = |c: &ferrum_scene::Color| f64::from(c.a) / 255.0;
    match node {
        N::Circle { style, .. } | N::Rect { style, .. } | N::Path { style, .. } | N::Polygon { style, .. } => {
            let fill = || style.fill.as_ref().map(|c| alpha(c) * style.fill_opacity);
            let stroke = || style.stroke.as_ref().map(|c| alpha(c) * style.stroke_opacity);
            match faded {
                Faded::Mark => fill().or_else(stroke),
                Faded::Fill => fill(),
                Faded::Stroke => stroke(),
            }
            .unwrap_or_else(|| panic!("{faded:?}: the node paints no such paint: {node:?}"))
        }
        N::Line { style, .. } | N::Polyline { style, .. } => {
            alpha(style.color.as_ref().expect("a painted stroke")) * style.stroke_opacity
        }
        N::Text { style, .. } => alpha(style.color.as_ref().expect("a painted text")) * style.opacity,
        other => panic!("no painted alpha for {other:?}"),
    }
}

/// Every mark node of `scene`'s first panel paired with the row it draws.
fn drawn_rows(scene: &ferrum_scene::SceneGraph) -> Vec<(&ferrum_scene::SceneNode, usize)> {
    scene.panels[0]
        .marks
        .iter()
        .flat_map(|batch| {
            let rows = batch.data_indices.as_ref().expect("mark nodes carry their rows");
            batch.nodes.iter().zip(rows.iter().copied())
        })
        .collect()
}


// ── The numeric auxiliary channels' scale= gate ──────────────────────────────

/// A point chart of `c` (1.3, 2, 3, 9.7) on `channel` under `scale`, with a
/// stroke so the stroke channels show, and `params` declared.
fn numeric_aux_chart(channel: &str, scale: Option<serde_json::Value>, params: serde_json::Value) -> ChartSpec {
    let mut binding = serde_json::json!({"field": "c"});
    if let Some(scale) = scale {
        binding["scale"] = scale;
    }
    let mut spec: ChartSpec = serde_json::from_value(serde_json::json!({
        "mark": "point",
        "encoding": with_xy(bound(channel, &binding)),
        "params": params,
    }))
    .expect("the spec deserializes");
    spec.mark_style = Some(crate::spec::mark_style::MarkKwargsSpec { stroke: Some("black".into()), ..Default::default() });
    spec
}

fn numeric_aux_batch() -> RecordBatch {
    one_column_batch(vec![Some(1.3), Some(2.0), Some(3.0), Some(9.7)])
}

/// The marks and the legend a render draws, as JSON, or the refusal text.
fn marks_and_legend(spec: &ChartSpec) -> Result<(serde_json::Value, serde_json::Value), String> {
    pyo3::Python::initialize();
    let theme = crate::layout::ThemeInputs::default();
    let chart_config = crate::render::chart_config::ChartConfig::default();
    let batch = numeric_aux_batch();
    let crate::render::PipelineOutput { prep, layout, applied, mut warnings } = crate::render::prepare_and_layout(
        spec,
        &batch,
        &theme,
        crate::layout::Viewport { width: 600.0, height: 400.0 },
        &chart_config,
        None,
    )
    .map_err(|e| e.to_string())?;
    let scene = crate::render::scene_build::build_scene(
        spec,
        &prep,
        &layout,
        &applied.effective_theme,
        &crate::render::config::RenderConfig::default(),
        &mut warnings,
        &chart_config,
        None,
    )
    .map_err(|e| e.to_string())?;
    let marks: Vec<&ferrum_scene::SceneNode> = scene.panels[0].marks.iter().flat_map(|b| b.nodes.iter()).collect();
    Ok((serde_json::to_value(marks).unwrap(), serde_json::to_value(&scene.legend).unwrap()))
}

/// What a `linear` scale key does on a numeric auxiliary channel.
#[derive(Debug, PartialEq)]
enum KeyOutcome {
    /// Refused with this text.
    Refused(String),
    /// Changes the marks and the legend together.
    Drawn,
    /// Changes nothing: the scale already behaves this way.
    Inert,
}

/// Every key a `linear` scale accepts, on each numeric auxiliary channel, is
/// either refused or changes the marks and the legend together; the two
/// exceptions are documented no-ops (`clamp`: these scales always clamp;
/// `systemInjected`: a positional-only flag) and `nice` on size and stroke
/// width, whose data-derived domain is already niced. A key added to the
/// variant fails this test until it is classified here. A bare `linear`
/// scale draws exactly what no `scale=` draws.
#[test]
fn every_linear_scale_key_is_refused_or_drawn_on_every_numeric_aux_channel() {
    let keys = crate::spec::encoding::scale_accepted_keys("linear").expect("linear has keys");
    let param = serde_json::json!([{"name": "p", "kind": "variable", "value": [0, 20]}]);
    for channel in crate::render::scale_resolve::NUMERIC_AUX_CHANNELS {
        let range = if channel.contains("opacity") { serde_json::json!([0.2, 0.6]) } else { serde_json::json!([2.0, 30.0]) };
        let baseline = marks_and_legend(&numeric_aux_chart(channel, Some(serde_json::json!({"type": "linear"})), param.clone()))
            .expect("a bare linear scale renders");
        assert_eq!(baseline, marks_and_legend(&numeric_aux_chart(channel, None, param.clone())).unwrap(), "{channel}");
        let refused = |reason: &str| {
            KeyOutcome::Refused(format!(
                "{channel} scale: {reason}; pass fm.LinearScale(domain=[...], range=[...]) or drop scale="
            ))
        };
        for key in &keys {
            let (value, expected) = match key.as_str() {
                "domain" => (serde_json::json!([0.0, 20.0]), KeyOutcome::Drawn),
                "range" => (range.clone(), KeyOutcome::Drawn),
                "reverse" | "zero" => (serde_json::json!(true), KeyOutcome::Drawn),
                "nice" if channel.contains("opacity") => (serde_json::json!(true), KeyOutcome::Drawn),
                "nice" => (serde_json::json!(true), KeyOutcome::Inert),
                "domainParam" => (serde_json::json!("p"), KeyOutcome::Drawn),
                "clamp" | "systemInjected" => (serde_json::json!(true), KeyOutcome::Inert),
                "padding" => (serde_json::json!(0.1), refused("padding is not supported (it insets an axis)")),
                "scheme" => (serde_json::json!("blues"), refused("scheme is not supported (it names a color scheme)")),
                other => panic!("{other}: a linear scale key this test does not classify"),
            };
            let mut scale = serde_json::json!({"type": "linear"});
            scale[key.as_str()] = value;
            let outcome = match marks_and_legend(&numeric_aux_chart(channel, Some(scale), param.clone())) {
                Err(text) => KeyOutcome::Refused(text),
                Ok((marks, legend)) if marks != baseline.0 && legend != baseline.1 => KeyOutcome::Drawn,
                Ok((marks, legend)) if marks == baseline.0 && legend == baseline.1 => KeyOutcome::Inert,
                Ok((marks, _)) => panic!("{channel} {key}: the marks {} but the legend does not agree", if marks == baseline.0 { "hold" } else { "change" }),
            };
            assert_eq!(outcome, expected, "{channel} {key}");
        }
    }
}

/// Every scale type other than `linear` refuses on every numeric auxiliary
/// channel, naming the type by its wire tag, as does a domain or range that
/// is not `[lo, hi]` (with the positional sentence).
#[test]
fn a_non_linear_or_malformed_scale_refuses_on_every_numeric_aux_channel() {
    for channel in crate::render::scale_resolve::NUMERIC_AUX_CHANNELS {
        for tag in crate::spec::encoding::ScaleSpec::TAGS.into_iter().filter(|t| *t != "linear") {
            let text = marks_and_legend(&numeric_aux_chart(channel, Some(serde_json::json!({"type": tag})), serde_json::json!([])))
                .expect_err("a non-linear scale refuses");
            assert_eq!(
                text,
                format!("{channel} scale: type '{tag}' is not supported; pass fm.LinearScale(domain=[...], range=[...]) or drop scale="),
            );
        }
        for (key, len) in [("domain", 3), ("range", 1)] {
            let values: Vec<f64> = (0..len).map(f64::from).collect();
            let text = marks_and_legend(&numeric_aux_chart(channel, Some(serde_json::json!({"type": "linear", key: values})), serde_json::json!([])))
                .expect_err("a malformed extent refuses");
            assert_eq!(text, format!("{channel} scale: {key} must have length 2; got {len}"));
        }
    }
}

/// Size reads an explicit domain, and a value outside it draws at the nearer
/// end of the range: domain `[2, 3]` onto `[10, 40]` draws 10, 10, 40, 40.
#[test]
fn a_size_domain_is_honored_and_clamps() {
    let spec = numeric_aux_chart("size", Some(serde_json::json!({"type": "linear", "domain": [2.0, 3.0], "range": [10.0, 40.0]})), serde_json::json!([]));
    let scene = scene_over(&spec, &numeric_aux_batch(), &crate::layout::ThemeInputs::default()).0;
    let areas: Vec<f64> = scene.panels[0].marks[0]
        .nodes
        .iter()
        .map(|n| match n {
            ferrum_scene::SceneNode::Circle { r, .. } => std::f64::consts::PI * r * r,
            other => panic!("{other:?}"),
        })
        .collect();
    assert_close(&areas, &[10.0, 10.0, 40.0, 40.0], "size areas");
}

/// A range crossing the channel's drawable interval is clamped into it before
/// the scale is built, so the mapping stays monotone and the legend lists every
/// entry: stroke width `[-2, 4]` over `w` (20..50) draws 0, 4/3, 8/3, 4, and
/// size `[-10, 100]` draws areas from 0.
#[test]
fn a_range_outside_the_drawable_interval_clamps_into_it() {
    let w = bound("stroke_width", &serde_json::json!({"field": "w", "scale": {"type": "linear", "range": [-2.0, 4.0]}}));
    let spec = drawn_channel_alone("point", w);
    let (scene, _) = drawn_channel_scene(&spec);
    assert_close(&point_stroke_widths(&scene), &linear_map(&[20.0, 30.0, 40.0, 50.0], (20.0, 50.0), (0.0, 4.0)), "widths");
    assert_eq!(legend_line_widths(&scene).len(), NICE_W.len(), "every entry is listed");

    let size = bound("size", &serde_json::json!({"field": "v", "scale": {"type": "linear", "range": [-10.0, 100.0]}}));
    match drawn_channel_prepared(&drawn_channel_alone("point", size)).provisional_scales.size {
        Some(scale) => assert_eq!(scale.inner.pixel_range(), (0.0, 100.0)),
        None => panic!("a size scale"),
    }
}

/// A `domainParam` on a layer's own encoding, and on fill_opacity and
/// stroke_width, is substituted: the marks and the legend map the param's
/// `[0, 20]`.
#[test]
fn a_domain_param_is_substituted_on_every_numeric_aux_channel_and_on_a_layer() {
    let param = serde_json::json!([{"name": "p", "kind": "variable", "value": [0, 20]}]);
    let with_param = |channel: &str| numeric_aux_chart(channel, Some(serde_json::json!({"type": "linear", "domainParam": "p"})), param.clone());
    let with_domain = |channel: &str| numeric_aux_chart(channel, Some(serde_json::json!({"type": "linear", "domain": [0.0, 20.0]})), param.clone());
    for channel in ["fill_opacity", "stroke_opacity", "stroke_width"] {
        assert_eq!(marks_and_legend(&with_param(channel)).unwrap(), marks_and_legend(&with_domain(channel)).unwrap(), "{channel}");
    }
    let layered = |scale: serde_json::Value| {
        let own = bound("opacity", &serde_json::json!({"field": "c", "scale": scale}));
        let mut spec = drawn_channel_spec(serde_json::json!({}), &[("point", serde_json::json!({})), ("point", own)]);
        spec.params = serde_json::from_value(param.clone()).unwrap();
        spec
    };
    let batch = numeric_aux_batch();
    let theme = crate::layout::ThemeInputs::default();
    let opacities = |spec: &ChartSpec| legend_circle_opacities(&scene_over(spec, &batch, &theme).0);
    assert_eq!(
        opacities(&layered(serde_json::json!({"type": "linear", "domainParam": "p"}))),
        opacities(&layered(serde_json::json!({"type": "linear", "domain": [0.0, 20.0]}))),
    );
}

/// A label's leader line paints at half the mark's opacity, carried in its
/// stroke color (the value the SVG walker and the interactive renderer both
/// paint): at `opacity=0.4` the leader paints 0.2.
#[test]
fn a_label_leader_paints_at_half_the_mark_opacity() {
    let mut spec = drawn_channel_alone("label", serde_json::json!({"text": {"field": "g"}}));
    spec.mark_style = Some(crate::spec::mark_style::MarkKwargsSpec {
        leader_line: Some(true),
        opacity: Some(0.4),
        ..Default::default()
    });
    let (scene, _) = drawn_channel_scene(&spec);
    let leaders: Vec<f64> = scene.panels[0]
        .marks
        .iter()
        .flat_map(|b| b.nodes.iter())
        .filter(|n| matches!(n, ferrum_scene::SceneNode::Line { .. }))
        .map(|n| painted_alpha(n, Faded::Stroke))
        .collect();
    assert_eq!(leaders.len(), 4, "one leader per label");
    assert_painted_close(&leaders, &[0.2; 4], "leader");
}

/// A minimal 1x1 PNG as a base64 data URL, for the url-tile image mark.
fn tiny_png_url() -> String {
    let png_bytes: Vec<u8> = vec![
        137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 1, 0, 0, 0, 1, 8, 2, 0, 0, 0, 144,
        119, 83, 222, 0, 0, 0, 12, 73, 68, 65, 84, 8, 215, 99, 248, 207, 192, 0, 0, 0, 2, 0, 1, 227, 33, 188, 51, 0,
        0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
    ];
    use base64::Engine;
    format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(&png_bytes))
}

/// [`drawn_channel_batch`] for the marks that read it; for an image, url
/// tiles at four points; for a geoshape, four point geometries. Each carries
/// `v` (1, 2, 3, 10) for the opacity channels.
fn opacity_table_batch(mark: Mark) -> RecordBatch {
    let v = Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 10.0]));
    match mark {
        Mark::Image => {
            let url = tiny_png_url();
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![
                    Field::new("x", DataType::Float64, false),
                    Field::new("y", DataType::Float64, false),
                    Field::new("url", DataType::Utf8, false),
                    Field::new("v", DataType::Float64, false),
                ])),
                vec![
                    Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
                    Arc::new(Float64Array::from(vec![1.0, 3.0, 2.0, 4.0])),
                    Arc::new(StringArray::from(vec![url.as_str(); 4])),
                    v,
                ],
            )
            .unwrap()
        }
        Mark::Geoshape => {
            let points: Vec<String> =
                (0..4).map(|i| format!(r#"{{"type":"Point","coordinates":[{},{}]}}"#, -90.0 + i as f64, 40.0 + i as f64)).collect();
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![
                    Field::new("__geometry__", DataType::Utf8, true),
                    Field::new("v", DataType::Float64, false),
                ])),
                vec![Arc::new(StringArray::from(points.iter().map(String::as_str).collect::<Vec<_>>())), v],
            )
            .unwrap()
        }
        _ => drawn_channel_batch(),
    }
}

/// A minimal chart of `mark` over [`opacity_table_batch`] binding `extra`,
/// with a black stroke where the mark takes one (so a stroke channel has a
/// paint to fade) and `arrow=True` when asked.
fn opacity_table_chart(mark: Mark, extra: serde_json::Value, arrow: bool) -> ChartSpec {
    let f = |field: &str| serde_json::json!({"field": field});
    let mut encoding = match mark {
        Mark::Point | Mark::Line | Mark::Area | Mark::Polygon => serde_json::json!({"x": f("x"), "y": f("y")}),
        Mark::Bar => serde_json::json!({"x": f("g"), "y": f("y")}),
        Mark::Tick => serde_json::json!({"x": f("x"), "y": f("g")}),
        Mark::Ribbon => serde_json::json!({"x": f("x"), "y": f("y"), "y2": f("w")}),
        Mark::Rule | Mark::Rect | Mark::Segment => serde_json::json!({"x": f("x"), "y": f("y"), "x2": f("w"), "y2": f("w")}),
        Mark::Text | Mark::Label => serde_json::json!({"x": f("x"), "y": f("y"), "text": f("g")}),
        Mark::Arc => serde_json::json!({"x": f("x")}),
        Mark::Image => serde_json::json!({"x": f("x"), "y": f("y"), "url": f("url")}),
        Mark::Geoshape => serde_json::json!({}),
    };
    for (channel, binding) in extra.as_object().expect("an object") {
        encoding[channel] = binding.clone();
    }
    let mut spec = serde_json::json!({"mark": mark, "encoding": encoding});
    match mark {
        Mark::Arc => spec["coord"] = serde_json::json!({"kind": "polar"}),
        Mark::Geoshape => spec["coord"] = serde_json::json!({"kind": "geo", "projection": "equirectangular"}),
        _ => {}
    }
    let mut spec: ChartSpec = serde_json::from_value(spec).expect("the spec deserializes");
    let stroked = mark.style_field("stroke").is_some();
    spec.mark_style = Some(crate::spec::mark_style::MarkKwargsSpec {
        stroke: stroked.then(|| "black".to_string()),
        arrow: arrow.then_some(true),
        ..Default::default()
    });
    spec
}

/// The (mark, channel, arrow) triples whose mark draws the channel
/// ([`crate::spec::mark::Mark::draws_opacity_channel`], a segment's
/// `fill_opacity` with `arrow=True` only), and every other triple.
fn opacity_table_cases() -> Vec<(Mark, crate::render::scale_resolve::OpacityChannel, bool, bool)> {
    use crate::render::scale_resolve::OpacityChannel;
    let mut cases = Vec::new();
    for mark in crate::spec::mark::all_marks() {
        for arrow in if mark == Mark::Segment { vec![false, true] } else { vec![false] } {
            for channel in [OpacityChannel::Opacity, OpacityChannel::Fill, OpacityChannel::Stroke] {
                let drawn = mark.draws_opacity_channel(channel)
                    && !(mark == Mark::Segment && channel == OpacityChannel::Fill && !arrow);
                cases.push((mark, channel, arrow, drawn));
            }
        }
    }
    cases
}

/// Every mark draws an opacity-family channel exactly where its drawing
/// table says: binding the channel to `v` changes the marks it paints and
/// draws the channel's legend, or changes neither. Every mark is covered,
/// image and geoshape included. Rejects a legend over a channel no mark
/// paints (a line's `fill_opacity`) and a table entry for a channel the
/// builder ignores.
#[test]
fn every_mark_draws_an_opacity_channel_exactly_where_its_table_says() {
    let theme = crate::layout::ThemeInputs::default();
    let marks_of = |scene: &ferrum_scene::SceneGraph| {
        serde_json::to_value(scene.panels[0].marks.iter().flat_map(|b| b.nodes.iter()).collect::<Vec<_>>()).unwrap()
    };
    for (mark, channel, arrow, drawn) in opacity_table_cases() {
        let batch = opacity_table_batch(mark);
        let unbound = marks_of(&scene_over(&opacity_table_chart(mark, serde_json::json!({}), arrow), &batch, &theme).0);
        let name = channel.context();
        let spec = opacity_table_chart(mark, bound(name, &serde_json::json!({"field": "v"})), arrow);
        let painted = marks_of(&scene_over(&spec, &batch, &theme).0) != unbound;
        pyo3::Python::initialize();
        let prep = prepare_render_inputs(
            &spec,
            &batch,
            &theme,
            crate::layout::Viewport { width: 600.0, height: 400.0 },
            &crate::render::chart_config::ChartConfig::default(),
            None,
        )
        .expect("the spec prepares");
        let legend = prep.aux_legends.iter().any(|b| matches!(b, AuxLegendInput::Opacity { .. }));
        assert_eq!((painted, legend), (drawn, drawn), "{mark:?} {name} arrow={arrow}: (painted, legend)");
    }
}

/// Every (mark, channel) pair the drawing table says is drawn paints, at
/// each row, the alpha the channel's legend shows for that row's value: `v`
/// (1, 2, 3, 10) maps onto the 0.1..1.0 band, so `opacity` draws 0.1, 0.2,
/// 0.3 and 1.0 in place of the mark's own opacity, and the rows at 2 and 10
/// draw exactly the legend's "2" and "10" swatches. `fill_opacity` and
/// `stroke_opacity` fade the paint the mark draws without the channel (an
/// area's 0.35 fill) by that same alpha; a node without that paint (a
/// segment's shaft under `fill_opacity`) is not one the channel fades.
/// Rejects a mark that draws the raw clamped value (1.0 for every row).
#[test]
fn every_mark_draws_the_alpha_its_opacity_legend_shows() {
    use crate::render::scale_resolve::OpacityChannel;
    let theme = crate::layout::ThemeInputs::default();
    let v = [1.0, 2.0, 3.0, 10.0];
    let has_paint = |node: &ferrum_scene::SceneNode, faded: Faded| match (node, faded) {
        (ferrum_scene::SceneNode::Circle { style, .. }
        | ferrum_scene::SceneNode::Rect { style, .. }
        | ferrum_scene::SceneNode::Path { style, .. }
        | ferrum_scene::SceneNode::Polygon { style, .. }, Faded::Fill) => style.fill.is_some(),
        (ferrum_scene::SceneNode::Circle { style, .. }
        | ferrum_scene::SceneNode::Rect { style, .. }
        | ferrum_scene::SceneNode::Path { style, .. }
        | ferrum_scene::SceneNode::Polygon { style, .. }, Faded::Stroke) => style.stroke.is_some(),
        (ferrum_scene::SceneNode::Text { .. } | ferrum_scene::SceneNode::Line { .. } | ferrum_scene::SceneNode::Polyline { .. }, Faded::Fill) => false,
        (ferrum_scene::SceneNode::Text { .. }, Faded::Stroke) => false,
        _ => true,
    };
    let mut wrong: Vec<String> = Vec::new();
    for (mark, channel, arrow, drawn) in opacity_table_cases() {
        if !drawn {
            continue;
        }
        let faded = match channel {
            OpacityChannel::Opacity => Faded::Mark,
            OpacityChannel::Fill => Faded::Fill,
            OpacityChannel::Stroke => Faded::Stroke,
        };
        let name = channel.context();
        let context = format!("{mark:?} {name} arrow={arrow}");
        let batch = opacity_table_batch(mark);
        let unfaded = scene_over(&opacity_table_chart(mark, serde_json::json!({}), arrow), &batch, &theme).0;
        let base_of = |row: usize| -> f64 {
            match faded {
                Faded::Mark => 1.0,
                _ => drawn_rows(&unfaded)
                    .into_iter()
                    .find(|(n, r)| *r == row && has_paint(n, faded))
                    .map(|(n, _)| painted_alpha(n, faded))
                    .expect("the unfaded mark paints the row"),
            }
        };
        let spec = opacity_table_chart(mark, bound(name, &serde_json::json!({"field": "v"})), arrow);
        let legend: Vec<(String, f64)> = match prepare_render_inputs(
            &spec,
            &batch,
            &theme,
            crate::layout::Viewport { width: 600.0, height: 400.0 },
            &crate::render::chart_config::ChartConfig::default(),
            None,
        )
        .expect("the spec prepares")
        .aux_legends
        .as_slice()
        {
            [AuxLegendInput::Opacity { entries, .. }] => entries.iter().map(|e| (e.label.clone(), e.opacity)).collect(),
            other => panic!("{context}: expected one opacity legend, got {other:?}"),
        };
        let swatch = |label: &str| legend.iter().find(|(l, _)| l == label).map(|(_, a)| *a).expect("a legend swatch");
        let scene = scene_over(&spec, &batch, &theme).0;
        let checked: Vec<(&ferrum_scene::SceneNode, usize)> =
            drawn_rows(&scene).into_iter().filter(|(n, _)| has_paint(n, faded)).collect();
        if checked.is_empty() {
            wrong.push(format!("{context}: no node paints the faded paint"));
        }
        for (node, row) in checked {
            let base = base_of(row);
            let painted = painted_alpha(node, faded) / base;
            let expected = linear_map(&[v[row]], (1.0, 10.0), (0.1, 1.0))[0];
            // A painted color carries its alpha as one of 256 steps, scaled
            // here by the base the alpha was divided by.
            let tolerance = 0.5 / 255.0 / base + 1e-12;
            if (painted - expected).abs() > tolerance {
                wrong.push(format!("{context} row {row}: drew {painted}, the scale maps {expected}"));
            }
            if row == 1 || row == 3 {
                let legend_alpha = swatch(if row == 1 { "2" } else { "10" });
                if (painted - legend_alpha).abs() > tolerance {
                    wrong.push(format!("{context} row {row}: drew {painted}, the legend shows {legend_alpha}"));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "marks drawing another alpha than their legend:\n{}", wrong.join("\n"));
}

/// On a layered chart the opacity legend follows the layers that draw the
/// channel: bound only on a line's `fill_opacity` (which no line paints) it
/// draws none; bound on that line and on a point, one.
#[test]
fn a_layered_opacity_legend_follows_the_layers_that_draw_it() {
    let fo = bound("fill_opacity", &serde_json::json!({"field": "v"}));
    let only_line = drawn_channel_spec(serde_json::json!({}), &[("point", serde_json::json!({})), ("line", fo.clone())]);
    assert!(drawn_channel_prepared(&only_line).aux_legends.is_empty());
    let both = drawn_channel_spec(serde_json::json!({}), &[("point", fo.clone()), ("line", fo)]);
    let blocks = drawn_channel_prepared(&both).aux_legends;
    assert_eq!(blocks.iter().filter(|b| matches!(b, AuxLegendInput::Opacity { .. })).count(), 1, "{blocks:?}");
}

/// A `stroke_opacity` legend swatch is an unfilled ring whose stroke carries
/// the mapped alpha, as Vega-Lite applies `strokeOpacity` to the symbol
/// stroke; an `opacity` or `fill_opacity` swatch fades its fill.
#[test]
fn a_stroke_opacity_swatch_fades_its_stroke_not_its_fill() {
    for (channel, stroked) in [("opacity", false), ("fill_opacity", false), ("stroke_opacity", true)] {
        let (scene, _) = drawn_channel_scene(&drawn_channel_alone("point", bound(channel, &serde_json::json!({"field": "v"}))));
        let swatches: Vec<&ferrum_scene::FillStroke> = flat_nodes(&scene.legend)
            .into_iter()
            .filter_map(|n| match n {
                ferrum_scene::SceneNode::Circle { style, .. } => Some(style),
                _ => None,
            })
            .collect();
        assert_eq!(swatches.len(), 5, "{channel}");
        for style in swatches {
            assert_eq!(style.fill.is_none(), stroked, "{channel}: {style:?}");
            assert_eq!(style.stroke.is_some(), stroked, "{channel}: {style:?}");
        }
    }
}

/// A two-frame batch for the facet repeat rule: the left operand's rows carry
/// `c` (`p`, `q`, `p`), the right operand's rows (marked `m`) come from a
/// frame without `c`, so their `c` is null padding, and one more left row has
/// a real null `c`.
fn facet_repeat_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("c", DataType::Utf8, true),
        Field::new("v", DataType::Float64, true),
        Field::new("m", DataType::Boolean, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec![Some("p"), Some("q"), Some("p"), None, None, None])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 10.0, 20.0, 4.0])),
            Arc::new(arrow::array::BooleanArray::from(vec![None, None, None, Some(true), Some(true), None])),
        ],
    )
    .expect("schema and columns agree")
}

/// The right operand's spec shape: its rows picked out by marker, projected
/// to its own frame's columns, then a step reading the facet field. `repeat`
/// declares (or not) that its frame lacks `c`.
fn facet_repeat_spec(repeat: &str) -> crate::spec::chart::ChartSpec {
    let json = format!(
        r#"{{"data":{{"kind":"named","name":"default"}},"mark":"point","encoding":{{"x":{{"field":"v"}},"y":{{"field":"v"}}}},
        "transforms":[{{"type":"filter","predicate":"datum[\"m\"] == true"}},{{"type":"project","fields":["v"]}},{{"type":"calculate","as_field":"panel","expr":"datum.c"}}],
        "facet":{{"field":"c","mode":{{"kind":"wrap","ncols":2}}{repeat}}}}}"#
    );
    serde_json::from_str(&json).expect("valid spec")
}

/// A frame lacking the facet field repeats in every panel, with the panel's
/// value in the field, so a step after the projection still reads its panel
/// (`panel`); the real null-`c` row of the frame that has `c` is dropped with
/// one warning naming the field and the count. The rejected readings: no
/// repeat (the marked rows land in no panel, the output is empty), and a
/// projection that drops the facet field (`panel` reads null).
#[test]
fn rows_whose_frame_lacks_the_facet_field_repeat_in_every_panel() {
    let spec = facet_repeat_spec(r#","repeat":[{"marker":"m","fields":["c"]}]"#);
    let theme = crate::layout::ThemeInputs::default();
    let viewport = crate::layout::Viewport { width: 600.0, height: 400.0 };
    let (outputs, warnings, _, _) = run_transforms_with_facet(&spec, &facet_repeat_batch(), viewport, &theme, None)
        .expect("a declared repeat renders");
    let out = outputs.get(FINAL_OUTPUT_KEY).expect("the tail is published");
    assert_eq!(float_col(out, "v"), [10.0, 20.0, 10.0, 20.0]);
    assert_eq!(str_col(out, "c"), ["p", "p", "q", "q"]);
    assert_eq!(str_col(out, "panel"), ["p", "p", "q", "q"]);
    assert_eq!(
        warnings,
        [crate::transform::context::TransformWarning::NullKeyRowsDropped { scope: crate::transform::context::NullKeyScope::Facet, field: "c".into(), rows: 1 }]
    );
    assert_eq!(
        warnings[0].to_string(),
        "facet 'c': 1 row has a null 'c' and belongs to no panel, so it was dropped; fill or filter the nulls in 'c'"
    );
}

/// Without a repeat entry every null-`c` row is a row of a frame that has
/// `c`: all three are dropped, and the one warning counts them.
#[test]
fn null_facet_values_without_a_repeat_are_dropped_with_a_count() {
    let spec = facet_repeat_spec("");
    let theme = crate::layout::ThemeInputs::default();
    let viewport = crate::layout::Viewport { width: 600.0, height: 400.0 };
    let (_, warnings, _, _) = run_transforms_with_facet(&spec, &facet_repeat_batch(), viewport, &theme, None)
        .expect("null facet values drop, they do not fail the chart");
    assert_eq!(
        warnings,
        [crate::transform::context::TransformWarning::NullKeyRowsDropped { scope: crate::transform::context::NullKeyScope::Facet, field: "c".into(), rows: 3 }]
    );
    assert!(warnings[0].to_string().starts_with("facet 'c': 3 rows have a null 'c'"), "{}", warnings[0]);
}

/// A repeat entry naming a field the facet does not split by, or a marker
/// the batch lacks, is a malformed wire and is refused.
#[test]
fn a_malformed_facet_repeat_is_refused() {
    let theme = crate::layout::ThemeInputs::default();
    let viewport = crate::layout::Viewport { width: 600.0, height: 400.0 };
    let not_a_facet = facet_repeat_spec(r#","repeat":[{"marker":"m","fields":["v"]}]"#);
    let err = run_transforms_with_facet(&not_a_facet, &facet_repeat_batch(), viewport, &theme, None)
        .expect_err("v is not a facet field");
    assert!(err.to_string().contains("names 'v', which is not a facet field"), "{err}");
    let no_marker = facet_repeat_spec(r#","repeat":[{"marker":"absent","fields":["c"]}]"#);
    let err = run_transforms_with_facet(&no_marker, &facet_repeat_batch(), viewport, &theme, None)
        .expect_err("the marker column is absent");
    assert!(matches!(err, RenderError::UnknownColumn { ref name } if name == "absent"), "{err}");
}

/// Every row null in the facet field leaves no panel at all: a typed error
/// naming the field, not a missing pipeline tail.
#[test]
fn a_facet_with_no_values_is_refused() {
    let spec = facet_repeat_spec("");
    let schema = Arc::new(Schema::new(vec![
        Field::new("c", DataType::Utf8, true),
        Field::new("v", DataType::Float64, true),
        Field::new("m", DataType::Boolean, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec![None::<&str>, None])),
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(arrow::array::BooleanArray::from(vec![Some(true), Some(true)])),
        ],
    )
    .expect("schema and columns agree");
    let theme = crate::layout::ThemeInputs::default();
    let viewport = crate::layout::Viewport { width: 600.0, height: 400.0 };
    let err = run_transforms_with_facet(&spec, &batch, viewport, &theme, None).expect_err("no panel exists");
    assert_eq!(err.to_string(), "facet: no usable values found for field 'c'");
}

/// The rows a layer reading its chart's data starts from, for the remedy a
/// missing column names: a branch reading a published `input` (a
/// composition's left operand's own rows) starts from that input, never from
/// the merged source, which also holds a sibling operand's `w`. So `w`, absent
/// from the left's rows and from every produced batch, gets the plain message
/// (`None`). The rejected reading consults the merged source and names the
/// Source remedy. The control: a branch reading the original batch keeps the
/// Source remedy for a source column its chain dropped.
#[test]
fn the_remedy_source_is_the_rows_the_layers_branch_starts_from() {
    use crate::transform::branch::BranchSpec;
    use crate::transform::core::TransformSpec;
    use crate::transform::filter::FilterSpec;
    let merged_source = Schema::new(vec![
        Field::new("x", DataType::Float64, true),
        Field::new("w", DataType::Float64, true),
    ]);
    let left_rows = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, true)])),
        vec![Arc::new(Float64Array::from(vec![1.0, 2.0]))],
    )
    .expect("schema and columns agree");
    let outputs = HashMap::from([("left_rows".to_string(), left_rows.clone()), ("layer_rows".to_string(), left_rows)]);
    let branch = |input: Option<&str>| {
        TransformSpec::Branch(BranchSpec {
            name: "layer_rows".into(),
            input: input.map(str::to_string),
            transforms: vec![TransformSpec::Filter(FilterSpec { predicate: "true".into(), name: None, param: None })],
        })
    };
    let layer = crate::spec::layer::Layer {
        mark: Mark::Line,
        encoding: Encoding { y: Some(EncodingSpec { field: "w".into(), ..Default::default() }), ..Default::default() },
        data_source: Some("layer_rows".into()),
        reads_chart_data: true,
        ..Default::default()
    };
    let prepared = LayerPrepared::from_chart_and_layer(&ChartSpec::default(), &layer);
    let reading_input = [branch(Some("left_rows"))];
    let pipeline = PipelineBatches { source: &merged_source, outputs: &outputs, transforms: &reading_input };
    assert_eq!(missing_column_remedy("w", &prepared, &pipeline), None);
    let reading_source = [branch(None)];
    let pipeline = PipelineBatches { source: &merged_source, outputs: &outputs, transforms: &reading_source };
    assert_eq!(missing_column_remedy("w", &prepared, &pipeline), Some(crate::render::MissingColumnRemedy::Source));
}

/// A repeat marker that is not Boolean is a malformed wire, refused by name.
#[test]
fn a_non_boolean_facet_repeat_marker_is_refused() {
    let spec = facet_repeat_spec(r#","repeat":[{"marker":"v","fields":["c"]}]"#);
    let theme = crate::layout::ThemeInputs::default();
    let viewport = crate::layout::Viewport { width: 600.0, height: 400.0 };
    let err = run_transforms_with_facet(&spec, &facet_repeat_batch(), viewport, &theme, None)
        .expect_err("v is Float64, not a Boolean marker");
    assert_eq!(err.to_string(), RenderError::ScaleResolutionFailed("facet repeat marker 'v' must be Boolean".into()).to_string());
}

/// A grid facet warns once per facet field, counting each field's own null
/// rows: two rows null in `c`, two null in `r`, one null in both (counted in
/// each). The marked row repeats along `r`, so its null `r` is not counted.
#[test]
fn a_grid_facet_warns_per_field_for_its_null_rows() {
    let spec: crate::spec::chart::ChartSpec = serde_json::from_str(
        r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"v"},"y":{"field":"v"}},
        "facet":{"field":"c","row":"r","mode":{"kind":"grid","nrows":2,"ncols":2},"repeat":[{"marker":"m","fields":["r"]}]}}"#,
    )
    .expect("valid spec");
    let schema = Arc::new(Schema::new(vec![
        Field::new("c", DataType::Utf8, true),
        Field::new("r", DataType::Utf8, true),
        Field::new("v", DataType::Float64, true),
        Field::new("m", DataType::Boolean, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec![Some("p"), Some("q"), None, Some("p"), None, Some("q")])),
            Arc::new(StringArray::from(vec![Some("s"), Some("t"), Some("s"), None, None, None])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
            Arc::new(arrow::array::BooleanArray::from(vec![None, None, None, None, None, Some(true)])),
        ],
    )
    .expect("schema and columns agree");
    let theme = crate::layout::ThemeInputs::default();
    let viewport = crate::layout::Viewport { width: 600.0, height: 400.0 };
    let (outputs, warnings, _, _) =
        run_transforms_with_facet(&spec, &batch, viewport, &theme, None).expect("null facet rows drop, they do not fail");
    use crate::transform::context::{NullKeyScope::Facet, TransformWarning::NullKeyRowsDropped};
    assert_eq!(
        warnings,
        [
            NullKeyRowsDropped { scope: Facet, field: "c".into(), rows: 2 },
            NullKeyRowsDropped { scope: Facet, field: "r".into(), rows: 2 },
        ]
    );
    // The repeated row (v = 6) joins both rows of its `q` column.
    let out = outputs.get(FINAL_OUTPUT_KEY).expect("the tail is published");
    assert_eq!(float_col(out, "v").iter().filter(|v| **v == 6.0).count(), 2);
}

// ── The column gate covers every channel ─────────────────────────────────────
//
// A layer beyond the first that names a column its own rows lack, on any
// channel, is refused with the positional gate's classified message
// (`validate_layer_columns`). Each test renders the whole chart, since which
// batch a layer is judged against is chosen by the scene build.

/// The gate fixture's rows: `x`, `y`, `z` (Float64) and `g`, `k` (Utf8).
fn column_gate_batch() -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("z", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
            Field::new("k", DataType::Utf8, false),
        ])),
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
            Arc::new(Float64Array::from(vec![2.0, 1.0, 4.0, 3.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0])),
            Arc::new(StringArray::from(vec!["a", "b", "a", "b"])),
            Arc::new(StringArray::from(vec!["k0", "k1", "k2", "k3"])),
        ],
    )
    .expect("schema and columns agree")
}

/// A two-layer chart over [`column_gate_batch`]: `layers` as written,
/// `transforms` before them. The chart-level encoding is layer 0's `x` and
/// `y`, so every other binding is the layer's own.
fn column_gate_spec(transforms: serde_json::Value, layers: serde_json::Value) -> ChartSpec {
    let layer0 = &layers[0]["encoding"];
    serde_json::from_value(serde_json::json!({
        "data": {"kind": "named", "name": "default"},
        "mark": layers[0]["mark"],
        "encoding": {"x": layer0["x"], "y": layer0["y"]},
        "transforms": transforms,
        "layers": layers,
    }))
    .expect("the layered spec deserializes")
}

/// A named statistic: the per-`g` mean of `z` as `mean_z`, published as
/// `stats`. A layer reading the chart's rows does not see it.
fn named_mean_z() -> serde_json::Value {
    serde_json::json!([{"type": "aggregate", "ops": [{"field": "z", "fn": "mean", "as": "mean_z"}], "groupby": ["g"], "name": "stats"}])
}

/// An unnamed statistic the chart's rows advance through: per `g`, the sum
/// of `x` as `total`. `z` and `k` are source columns it drops.
fn chain_dropping_z_and_k() -> serde_json::Value {
    serde_json::json!([{"type": "aggregate", "ops": [{"field": "x", "fn": "sum", "as": "total"}], "groupby": ["g"]}])
}

/// The scene the whole pipeline builds for `spec` over `batch`, or its refusal,
/// with the warnings raised either way.
fn gate_render(spec: &ChartSpec, batch: &RecordBatch) -> (Result<ferrum_scene::SceneGraph, RenderError>, Vec<RenderWarning>) {
    pyo3::Python::initialize();
    let chart_config = crate::render::chart_config::ChartConfig::default();
    let theme = crate::layout::ThemeInputs::default();
    let crate::render::PipelineOutput { prep, layout, applied, mut warnings } = match crate::render::prepare_and_layout(
        spec,
        batch,
        &theme,
        crate::layout::Viewport { width: 600.0, height: 400.0 },
        &chart_config,
        None,
    ) {
        Ok(out) => out,
        Err(e) => return (Err(e), Vec::new()),
    };
    let scene = crate::render::scene_build::build_scene(
        spec,
        &prep,
        &layout,
        &applied.effective_theme,
        &crate::render::config::RenderConfig::default(),
        &mut warnings,
        &chart_config,
        None,
    );
    (scene, warnings)
}

/// The refusal `spec` raises, which must be the gate's.
fn gate_refusal(spec: &ChartSpec) -> RenderError {
    match gate_render(spec, &column_gate_batch()).0 {
        Err(e @ RenderError::LayerUnknownColumn { .. }) => e,
        other => panic!("expected the column gate's refusal, got {:?}", other.map(|_| "a rendered scene")),
    }
}

/// The gate's message for a column on layer 1, as its formatter writes it.
fn gate_message(channel: &'static str, name: &str, remedy: Option<crate::render::MissingColumnRemedy>, independent_y: bool) -> String {
    RenderError::LayerUnknownColumn { layer: 1, entry: None, channel, name: name.into(), coord_flipped: false, remedy, independent_y }
        .to_string()
}

/// Every non-positional channel a layer binds itself, naming a column found
/// nowhere, refuses with the plain message naming the channel as `encode`
/// writes it (`tooltip` for a `tooltip_fields` entry), each on a mark that
/// draws the channel. Layer 0 binds the channels that build a chart-wide
/// scale to a real column, so only the gate can refuse layer 1. Rejects the
/// positional-only gate, under which every one of these rendered with the
/// channel silently dropped.
#[test]
fn every_channel_naming_an_absent_column_refuses_with_the_plain_message() {
    // (wire channel, as `encode` names it, layer 0's field for it, layer 0's mark, layer 1's mark)
    let cases: [(&str, &'static str, Option<&str>, &str, &str); 16] = [
        ("color", "color", Some("g"), "point", "point"),
        ("size", "size", Some("z"), "point", "point"),
        ("shape", "shape", Some("g"), "point", "point"),
        ("opacity", "opacity", Some("z"), "point", "point"),
        ("fill_opacity", "fill_opacity", Some("z"), "point", "point"),
        ("stroke_opacity", "stroke_opacity", Some("z"), "point", "point"),
        ("stroke_dash", "stroke_dash", Some("g"), "line", "line"),
        ("stroke_width", "stroke_width", Some("z"), "line", "line"),
        ("angle", "angle", None, "point", "point"),
        ("text", "text", None, "point", "text"),
        ("tooltip", "tooltip", None, "point", "point"),
        ("tooltip_fields", "tooltip", None, "point", "point"),
        ("key", "key", None, "point", "point"),
        ("href", "href", None, "point", "point"),
        ("description", "description", None, "point", "point"),
        ("url", "url", None, "point", "image"),
    ];
    for (wire, voiced, layer0_field, layer0_mark, layer1_mark) in cases {
        let binding = |field: &str| {
            let one = serde_json::json!({"field": field});
            if wire == "tooltip_fields" { serde_json::json!([{"field": "x"}, one]) } else { one }
        };
        let mut layer0 = serde_json::json!({"x": {"field": "x"}, "y": {"field": "y"}});
        if let Some(field) = layer0_field {
            layer0[wire] = binding(field);
        }
        let mut layer1 = serde_json::json!({"x": {"field": "x"}, "y": {"field": "y"}});
        layer1[wire] = binding("nope");
        let spec = column_gate_spec(
            serde_json::json!([]),
            serde_json::json!([{"mark": layer0_mark, "encoding": layer0}, {"mark": layer1_mark, "encoding": layer1}]),
        );
        let err = gate_refusal(&spec);
        assert!(
            matches!(err, RenderError::LayerUnknownColumn { layer: 1, channel, ref name, remedy: None, .. } if channel == voiced && name == "nope"),
            "{wire}: {err:?}"
        );
        assert_eq!(err.to_string(), gate_message(voiced, "nope", None, false), "{wire}");
    }
}

/// A binding on a channel the layer's mark does not draw names no column it
/// reads, so the gate does not judge it, own or inherited: a line's own
/// `text` and `url`, and a chart-level `angle` every layer inherits that a
/// line never draws (layer 0 is a line too, so no layer draws it). The
/// control: the same own `text` on a text mark refuses.
#[test]
fn a_binding_on_a_channel_the_mark_does_not_draw_is_not_judged() {
    let spec = |layer1_mark: &str| {
        let mut spec = column_gate_spec(
            serde_json::json!([]),
            serde_json::json!([
                {"mark": "line", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}}},
                {"mark": layer1_mark, "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "text": {"field": "nope"}, "url": {"field": "nope_url"}}},
            ]),
        );
        spec.encoding.angle = Some(EncodingSpec { field: "nope_angle".into(), ..Default::default() });
        spec
    };
    let (scene, _) = gate_render(&spec("line"), &column_gate_batch());
    let scene = scene.expect("a line draws no text, url or angle");
    assert!(scene.panels[0].marks.iter().all(|batch| !batch.nodes.is_empty()), "both layers draw");
    assert_eq!(gate_refusal(&spec("text")).to_string(), gate_message("text", "nope", None, false));
}

/// The gate answers before any scale reads a column: a later layer that
/// establishes a chart-wide scale (no earlier layer draws the channel) from
/// a column its rows lack refuses in the gate's words, not as the scale
/// builder's unlocated column. Spec §4.5's example, `color`, then `size` and
/// `opacity`.
#[test]
fn a_later_layer_establishing_a_scale_from_an_absent_column_refuses_in_the_gates_words() {
    for (wire, mark) in [("color", "line"), ("size", "point"), ("opacity", "point")] {
        let mut layer1 = serde_json::json!({"x": {"field": "x"}, "y": {"field": "y"}});
        layer1[wire] = serde_json::json!({"field": "nope"});
        let spec = column_gate_spec(
            serde_json::json!([]),
            serde_json::json!([
                {"mark": "point", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}}},
                {"mark": mark, "encoding": layer1},
            ]),
        );
        assert_eq!(gate_refusal(&spec).to_string(), gate_message(wire, "nope", None, false), "{wire}");
    }
}

/// A statistic sub-layer (reading a named output no `Branch` publishes) that
/// inherits a drawn chart-level binding whose column its statistic dropped,
/// though the source holds it, draws without that channel and warns once,
/// in the layer's voice. A chart-wide conditional's field branch it inherits
/// is dropped the same way. Rejects refusing the inherited read, and rejects
/// dropping it silently.
#[test]
fn a_statistic_sub_layer_drops_an_inherited_read_its_statistic_does_not_carry_with_a_warning() {
    let mut spec = column_gate_spec(
        named_mean_z(),
        serde_json::json!([
            {"mark": "point", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "color": {"field": "k"}}},
            {"mark": "point", "data_source": "stats", "encoding": {"x": {"field": "mean_z"}, "y": {"field": "mean_z"}}},
        ]),
    );
    spec.encoding.color = Some(EncodingSpec { field: "k".into(), ..Default::default() });
    spec.conditionals = vec![serde_json::from_value(serde_json::json!({
        "selection_name": "pick", "channel": "size",
        "if_selected": {"kind": "field", "name": "z"},
        "if_not": {"kind": "size", "value": 10.0},
    }))
    .expect("a conditional")];
    let (scene, warnings) = gate_render(&spec, &column_gate_batch());
    let scene = scene.expect("the statistic layer draws without the dropped channels");
    assert!(scene.panels[0].marks.iter().all(|batch| !batch.nodes.is_empty()), "both layers draw");
    let dropped: Vec<String> = warnings
        .iter()
        .filter(|w| matches!(w, RenderWarning::StatisticDroppedChannel { .. }))
        .map(ToString::to_string)
        .collect();
    assert_eq!(
        dropped,
        [
            "layer 1: mark_point: color 'k' was dropped: the statistic this point mark draws does not carry 'k', so it draws without color",
            "layer 1: mark_point: size 'z' was dropped: the statistic this point mark draws does not carry 'z', so it draws without size",
        ]
    );
}

/// What still refuses beside a statistic sub-layer's dropped read: the same
/// column bound by the statistic layer itself (its own binding), the same
/// inherited binding on a layer over the user's rows (a `Branch` output),
/// and an inherited column found nowhere, even in the source.
#[test]
fn own_bindings_raw_row_layers_and_unknown_columns_still_refuse_beside_a_statistic_layer() {
    let own = column_gate_spec(
        named_mean_z(),
        serde_json::json!([
            {"mark": "point", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "color": {"field": "k"}}},
            {"mark": "point", "data_source": "stats", "encoding": {"x": {"field": "mean_z"}, "y": {"field": "mean_z"}, "color": {"field": "k"}}},
        ]),
    );
    assert_eq!(gate_refusal(&own).to_string(), gate_message("color", "k", None, false));

    let mut branch = column_gate_spec(
        serde_json::json!([{"type": "branch", "name": "rows", "transforms": chain_dropping_z_and_k()}]),
        serde_json::json!([
            {"mark": "point", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "color": {"field": "k"}}},
            {"mark": "point", "data_source": "rows", "encoding": {"x": {"field": "total"}, "y": {"field": "total"}}},
        ]),
    );
    branch.encoding.color = Some(EncodingSpec { field: "k".into(), ..Default::default() });
    assert_eq!(gate_refusal(&branch).to_string(), gate_message("color", "k", None, false));

    let mut nowhere = column_gate_spec(
        named_mean_z(),
        serde_json::json!([
            {"mark": "point", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}}},
            {"mark": "point", "data_source": "stats", "encoding": {"x": {"field": "mean_z"}, "y": {"field": "mean_z"}}},
        ]),
    );
    // Layer 0 binds its own tooltip, so only the statistic layer inherits the
    // chart-level one.
    nowhere.layers.as_mut().expect("layers")[0].encoding.tooltip = Some(EncodingSpec { field: "x".into(), ..Default::default() });
    nowhere.encoding.tooltip = Some(EncodingSpec { field: "nope".into(), ..Default::default() });
    assert_eq!(gate_refusal(&nowhere).to_string(), gate_message("tooltip", "nope", None, false));
}

/// A layer's `detail` grouping column is checked as its channels are.
#[test]
fn a_detail_naming_an_absent_column_refuses() {
    let spec = column_gate_spec(
        serde_json::json!([]),
        serde_json::json!([
            {"mark": "line", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}}},
            {"mark": "line", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}}, "mark_style": {"detail": "nope"}},
        ]),
    );
    assert_eq!(gate_refusal(&spec).to_string(), gate_message("detail", "nope", None, false));
}

/// A field branch of a conditional layer 1 declared is checked against
/// layer 1's rows and named by the conditional's channel. The control: the
/// same conditional declared only by layer 0 is layer 0's, and the refusal
/// names layer 0.
#[test]
fn a_conditional_field_branch_naming_an_absent_column_refuses() {
    let conditional = |field: &str, layers: serde_json::Value| {
        serde_json::json!({
            "selection_name": "pick", "channel": "size",
            "if_selected": {"kind": "field", "name": field},
            "if_not": {"kind": "size", "value": 10.0},
            "layers": layers,
        })
    };
    let layers = serde_json::json!([
        {"mark": "point", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}}},
        {"mark": "point", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}}},
    ]);
    let mut spec = column_gate_spec(serde_json::json!([]), layers.clone());
    spec.conditionals = vec![serde_json::from_value(conditional("nope", serde_json::json!([1]))).expect("a conditional")];
    assert_eq!(gate_refusal(&spec).to_string(), gate_message("size", "nope", None, false));

    let mut declared_by_layer_0 = column_gate_spec(serde_json::json!([]), layers);
    declared_by_layer_0.conditionals = vec![serde_json::from_value(conditional("nope", serde_json::json!([0]))).expect("a conditional")];
    assert!(
        matches!(gate_refusal(&declared_by_layer_0), RenderError::LayerUnknownColumn { layer: 0, channel: "size", ref name, .. } if name == "nope"),
        "a conditional layer 1 did not declare is not layer 1's"
    );
}

/// A column the layer's own statistic produces is in the layer's own rows:
/// a layer reading `stats` draws `mean_z` on color and passes.
#[test]
fn a_column_the_layers_own_statistic_produces_passes() {
    let spec = column_gate_spec(
        named_mean_z(),
        serde_json::json!([
            {"mark": "point", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "color": {"field": "z"}}},
            {"mark": "point", "data_source": "stats",
             "encoding": {"x": {"field": "mean_z"}, "y": {"field": "mean_z"}, "color": {"field": "mean_z"}, "tooltip": {"field": "g"}}},
        ]),
    );
    let scene = gate_render(&spec, &column_gate_batch()).0.expect("every column layer 1 names is in its own rows");
    assert!(scene.panels[0].marks.iter().all(|batch| !batch.nodes.is_empty()), "both layers draw");
}

/// Layer 0 of a layered spec is judged by the gate like every later layer:
/// a column it names that its rows lack is refused in the gate's words,
/// naming layer 0, before any scale reads it.
#[test]
fn layer_zero_of_a_layered_spec_is_judged_by_the_column_gate() {
    let spec = column_gate_spec(
        serde_json::json!([]),
        serde_json::json!([
            {"mark": "point", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "tooltip": {"field": "nope"}}},
            {"mark": "point", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}}},
        ]),
    );
    assert_eq!(
        gate_refusal(&spec).to_string(),
        "layer 0: encoding 'tooltip' references unknown column 'nope' (not present in that layer's data)"
    );
}

/// A layer reading a named output of its own (a `data_source` the wire does
/// not mark as the chart's rows) keeps the plain message, even for a column
/// the chart's source holds: no chart-data remedy describes its rows.
#[test]
fn a_data_source_layer_not_reading_the_charts_rows_keeps_the_plain_message() {
    let spec = column_gate_spec(
        named_mean_z(),
        serde_json::json!([
            {"mark": "point", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}}},
            {"mark": "point", "data_source": "stats",
             "encoding": {"x": {"field": "mean_z"}, "y": {"field": "mean_z"}, "tooltip": {"field": "k"}}},
        ]),
    );
    assert_eq!(gate_refusal(&spec).to_string(), gate_message("tooltip", "k", None, false));
}

/// A non-positional column the chart's chain dropped names the source
/// remedy, spelled for the layer's axis: `+` on the shared y axis,
/// `fm.LayerChart(..., resolve={"y": "independent"})` on its own (the y-slot
/// route, which runs the same gate before resolving).
#[test]
fn a_source_column_the_chain_dropped_names_the_source_remedy_on_either_axis() {
    use crate::render::MissingColumnRemedy::Source;
    for independent_y in [false, true] {
        let spec = column_gate_spec(
            chain_dropping_z_and_k(),
            serde_json::json!([
                {"mark": "point", "encoding": {"x": {"field": "total"}, "y": {"field": "total"}, "color": {"field": "g"}}},
                {"mark": "point", "independent_y": independent_y,
                 "encoding": {"x": {"field": "total"}, "y": {"field": "total"}, "color": {"field": "z"}, "key": {"field": "k"}}},
            ]),
        );
        let message = gate_refusal(&spec).to_string();
        assert_eq!(message, gate_message("color", "z", Some(Source), independent_y));
        let remedy_door = if independent_y { "fm.LayerChart(" } else { "with `+`" };
        assert!(message.contains(remedy_door), "{message}");
    }
}

/// A non-positional column only another batch the pipeline produced holds
/// names the statistic remedy, spelled for the layer's axis.
#[test]
fn a_statistic_column_names_the_statistic_remedy_on_either_axis() {
    use crate::render::MissingColumnRemedy::Statistic;
    for independent_y in [false, true] {
        let spec = column_gate_spec(
            named_mean_z(),
            serde_json::json!([
                {"mark": "point", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}}},
                {"mark": "point", "independent_y": independent_y,
                 "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "tooltip_fields": [{"field": "mean_z"}]}},
            ]),
        );
        let message = gate_refusal(&spec).to_string();
        assert_eq!(message, gate_message("tooltip", "mean_z", Some(Statistic), independent_y));
        let remedy_door = if independent_y { "fm.LayerChart(" } else { "composed with `+`" };
        assert!(message.contains(remedy_door), "{message}");
    }
}

/// A later layer whose color reads categorical under the numeric color scale
/// layer 0 establishes is left out of that scale with one warning and draws
/// its constant paint, on a tick (a stroke-only mark reading per-row color
/// through the fallible reader) as on a point. Rejects refusing the tick's
/// `g` as an unsupported numeric dtype.
#[test]
fn a_color_left_out_of_a_numeric_scale_draws_unscaled_on_every_mark() {
    for mark in ["tick", "point", "rule"] {
        let mut layer1 = serde_json::json!({"x": {"field": "x"}, "y": {"field": "y"}, "color": {"field": "g"}});
        if mark == "rule" {
            layer1["x2"] = serde_json::json!({"field": "z"});
        }
        let spec = column_gate_spec(
            serde_json::json!([]),
            serde_json::json!([
                {"mark": "point", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "color": {"field": "z"}}},
                {"mark": mark, "encoding": layer1},
            ]),
        );
        let (scene, warnings) = gate_render(&spec, &column_gate_batch());
        let scene = scene.unwrap_or_else(|e| panic!("{mark}: the left-out layer draws: {e}"));
        assert_eq!(scale_exclusions(&warnings), excluded(1, 0, "color"), "{mark}");
        assert!(scene.panels[0].marks.iter().all(|batch| !batch.nodes.is_empty()), "{mark}: both layers draw");
    }
}

/// A layer reads the user's rows when its `data_source` is a `Branch`
/// output, at any nesting (a nested branch's tail is published scoped under
/// its enclosing branch), and a statistic's output when it names any other
/// published key: a named step's output, scoped or not.
#[test]
fn a_layer_reads_a_statistic_unless_a_branch_publishes_its_data_source() {
    let transforms: Vec<crate::transform::core::TransformSpec> = serde_json::from_value(serde_json::json!([
        {"type": "branch", "name": "outer", "transforms": [
            {"type": "branch", "name": "inner", "transforms": [{"type": "filter", "predicate": "true"}]},
            {"type": "aggregate", "ops": [{"field": "z", "fn": "mean", "as": "mean_z"}], "groupby": ["g"], "name": "stats"},
        ]},
    ]))
    .expect("the transforms deserialize");
    let reading = |data_source: Option<&str>| {
        let layer = crate::spec::layer::Layer { data_source: data_source.map(str::to_string), ..Default::default() };
        reads_statistic_output(&LayerPrepared::from_chart_and_layer(&ChartSpec::default(), &layer), &transforms)
    };
    assert!(!reading(None));
    assert!(!reading(Some("outer")));
    assert!(!reading(Some("outer_inner")));
    assert!(reading(Some("outer_stats")));
}

/// A dropped column warns once per layer, naming the first channel that
/// dropped it: an inherited `tooltip_fields` entry mirroring an inherited
/// `color` on the same column is cleared with it and raises no second
/// warning.
#[test]
fn a_column_a_statistic_layer_drops_on_two_channels_warns_once_naming_the_first() {
    let mut spec = column_gate_spec(
        named_mean_z(),
        serde_json::json!([
            {"mark": "point", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "color": {"field": "k"}}},
            {"mark": "point", "data_source": "stats", "encoding": {"x": {"field": "mean_z"}, "y": {"field": "mean_z"}}},
        ]),
    );
    spec.encoding.color = Some(EncodingSpec { field: "k".into(), ..Default::default() });
    spec.encoding.tooltip_fields = Some(vec![EncodingSpec { field: "k".into(), ..Default::default() }]);
    let (scene, warnings) = gate_render(&spec, &column_gate_batch());
    scene.expect("the statistic layer draws without k");
    let dropped: Vec<String> = warnings
        .iter()
        .filter(|w| matches!(w, RenderWarning::StatisticDroppedChannel { .. }))
        .map(ToString::to_string)
        .collect();
    assert_eq!(
        dropped,
        ["layer 1: mark_point: color 'k' was dropped: the statistic this point mark draws does not carry 'k', so it draws without color"]
    );
}

/// A layer-voiced message names a layer by its index only when the caller
/// placed it: a layer whose entry is its own mark's primitive, or that
/// records none. A desugaring call's sub-layer, and an unlayered chart's one
/// layer, carry no index.
#[test]
fn public_layer_names_only_a_layer_the_caller_placed() {
    let spec = |entry: Option<&str>| {
        let mut layered = column_gate_spec(
            serde_json::json!([]),
            serde_json::json!([
                {"mark": "point", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}}},
                {"mark": "point", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}}},
            ]),
        );
        layered.layers.as_mut().expect("layered")[1].entry = entry.map(str::to_string);
        layered
    };
    assert_eq!(public_layer(&spec(Some("mark_point")), 1, Mark::Point), Some(1));
    assert_eq!(public_layer(&spec(None), 1, Mark::Point), Some(1));
    assert_eq!(public_layer(&spec(Some("mark_boxen")), 1, Mark::Point), None);
    assert_eq!(public_layer(&ChartSpec { mark: Mark::Point, ..Default::default() }, 0, Mark::Point), None);
}

/// Two sibling sub-layers one desugaring call built, of the same mark, that
/// drop the same inherited column raise identical text (neither names an
/// index), which is reported once. Rejects one warning per sub-layer.
#[test]
fn sibling_sub_layers_of_one_call_dropping_one_column_warn_once() {
    let sub_layer = serde_json::json!(
        {"mark": "point", "data_source": "stats", "entry": "mark_boxen",
         "encoding": {"x": {"field": "mean_z"}, "y": {"field": "mean_z"}}}
    );
    let mut spec = column_gate_spec(
        named_mean_z(),
        serde_json::json!([
            {"mark": "point", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "color": {"field": "k"}}},
            sub_layer.clone(),
            sub_layer,
        ]),
    );
    spec.encoding.color = Some(EncodingSpec { field: "k".into(), ..Default::default() });
    let (scene, warnings) = gate_render(&spec, &column_gate_batch());
    scene.expect("both sub-layers draw without k");
    let dropped: Vec<String> = warnings
        .iter()
        .filter(|w| matches!(w, RenderWarning::StatisticDroppedChannel { .. }))
        .map(ToString::to_string)
        .collect();
    assert_eq!(
        dropped,
        ["mark_boxen: color 'k' was dropped: the statistic this point mark draws does not carry 'k', so it draws without color"]
    );
}

/// The prepared inputs `spec` builds over [`column_gate_batch`], or its
/// refusal: the provisional scales and the rendering encoding every axis and
/// legend reads.
fn gate_prepare(spec: &ChartSpec) -> Result<PreparedInputs, RenderError> {
    pyo3::Python::initialize();
    prepare_render_inputs(
        spec,
        &column_gate_batch(),
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport { width: 600.0, height: 400.0 },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
}

/// A statistic sub-layer at index 0 that inherits a drawn binding its
/// statistic's output does not carry warns as a later one does, and when no
/// layer draws the channel it gets no scale, so no legend describes marks
/// that do not draw it. Rejects layer 0's exemption, which established the
/// color scale and legend from the source rows no mark drew.
#[test]
fn a_statistic_layer_zero_drops_an_inherited_read_and_no_layer_draws_the_channel() {
    let mut spec = column_gate_spec(
        named_mean_z(),
        serde_json::json!([
            {"mark": "point", "data_source": "stats", "encoding": {"x": {"field": "mean_z"}, "y": {"field": "mean_z"}}},
            {"mark": "rule", "data_source": "stats", "encoding": {"y": {"field": "mean_z"}}},
        ]),
    );
    spec.encoding.color = Some(EncodingSpec { field: "k".into(), ..Default::default() });
    let (scene, warnings) = gate_render(&spec, &column_gate_batch());
    scene.expect("the statistic layers draw without the dropped channel");
    let dropped: Vec<String> = warnings
        .iter()
        .filter(|w| matches!(w, RenderWarning::StatisticDroppedChannel { .. }))
        .map(ToString::to_string)
        .collect();
    assert_eq!(
        dropped,
        [
            "layer 0: mark_point: color 'k' was dropped: the statistic this point mark draws does not carry 'k', so it draws without color",
            "layer 1: mark_rule: color 'k' was dropped: the statistic this rule mark draws does not carry 'k', so it draws without color",
        ]
    );
    let prep = gate_prepare(&spec).expect("prepares");
    assert!(prep.rendering_encoding().color.is_none(), "no layer draws color");
    assert!(prep.provisional_scales.color.is_none(), "a channel no layer draws builds no scale");
    assert!(prep.legend_entries.is_empty() && prep.colorbar.is_none(), "and no legend");
}

/// An auxiliary-scale channel is established by the first layer whose mark
/// draws it: a line's own `shape` (a line draws no shape) neither decides the
/// shape scale nor joins its domain, which is the point layer's `k` alone.
/// Rejects establishing from the first layer that merely binds the channel.
#[test]
fn an_auxiliary_channel_is_established_by_the_first_layer_whose_mark_draws_it() {
    let spec = column_gate_spec(
        serde_json::json!([]),
        serde_json::json!([
            {"mark": "line", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "shape": {"field": "g"}}},
            {"mark": "point", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "shape": {"field": "k"}}},
        ]),
    );
    let prep = gate_prepare(&spec).expect("prepares");
    assert_eq!(prep.rendering_encoding().shape.as_ref().map(|s| s.field.as_str()), Some("k"));
    let shape = prep.provisional_scales.shape.as_ref().expect("the point draws shape");
    assert_eq!(shape.domain, ["k0", "k1", "k2", "k3"], "the line's g is drawn by no mark");
    assert!(
        !prep.warnings.iter().any(|w| matches!(w, RenderWarning::ChannelNotDrawn { .. })),
        "a layer draws shape, so nothing warns"
    );
}

/// A channel bound where no layer's mark draws it builds no scale and no
/// legend and warns once, naming the channel, the field and the mark: an
/// unlayered line's `shape`, and two line layers binding it.
#[test]
fn a_channel_no_layer_draws_builds_no_scale_and_warns_once() {
    let flat: ChartSpec = serde_json::from_value(serde_json::json!({
        "data": {"kind": "named", "name": "default"},
        "mark": "line",
        "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "shape": {"field": "g"}},
    }))
    .expect("the flat spec deserializes");
    let layered = column_gate_spec(
        serde_json::json!([]),
        serde_json::json!([
            {"mark": "line", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "shape": {"field": "g"}}},
            {"mark": "line", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "shape": {"field": "g"}}},
        ]),
    );
    for (spec, voice) in [(flat, "mark_line"), (layered, "layer 0: mark_line")] {
        let prep = gate_prepare(&spec).expect("prepares");
        assert!(prep.provisional_scales.shape.is_none(), "{voice}: no scale for a channel nothing draws");
        assert!(prep.aux_legends.is_empty(), "{voice}: and no legend");
        let undrawn: Vec<String> = prep
            .warnings
            .iter()
            .filter(|w| matches!(w, RenderWarning::ChannelNotDrawn { .. }))
            .map(ToString::to_string)
            .collect();
        assert_eq!(
            undrawn,
            [format!(
                "{voice}: shape 'g' has no effect: no layer's mark draws shape (this line mark does not), so it \
                 gets no scale and no legend; it was ignored"
            )]
        );
    }
}

/// A `group` key belongs only on a layer that inherits a color it does not
/// draw (a text's). On the chart-level encoding, or on a layer whose mark
/// draws its inherited color, nothing reads it, so it refuses with the
/// remedy; on a text it prepares.
#[test]
fn a_group_key_off_a_text_layer_refuses() {
    let layer = |mark: &str| {
        serde_json::json!({"mark": mark, "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "text": {"field": "g"}, "group": {"field": "g"}}})
    };
    let layered = |mark: &str| {
        let mut spec = serde_json::to_value(column_gate_spec(
            serde_json::json!([]),
            serde_json::json!([{"mark": "point", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}}}, layer(mark)]),
        ))
        .expect("serializes");
        spec["encoding"]["color"] = serde_json::json!({"field": "g"});
        serde_json::from_value::<ChartSpec>(spec).expect("deserializes")
    };
    let err = gate_prepare(&layered("point")).expect_err("a point draws its inherited color").to_string();
    assert_eq!(
        err,
        "layer 1: layers[1].encoding.group is set on a mark_point layer, which draws the color it inherits and \
         groups by it, so group has no effect there. Remove group from this layer, or bind color on it to group \
         it by another field."
    );
    gate_prepare(&layered("text")).expect("a text holds its inherited color as its group key");

    let mut chart_level = serde_json::to_value(layered("text")).expect("serializes");
    chart_level["encoding"]["group"] = serde_json::json!({"field": "g"});
    let chart_level: ChartSpec = serde_json::from_value(chart_level).expect("deserializes");
    let err = gate_prepare(&chart_level).expect_err("the chart-level encoding is no layer's").to_string();
    assert_eq!(
        err,
        "ChartSpec.encoding.group is set on the chart-level encoding, where nothing reads it: group is a layer's \
         grouping key for a color that layer inherits but does not draw (a text's). Remove it, and bind color on \
         the chart to group its layers."
    );
}

/// A text-only chart whose color only a parent binds draws no color: a text
/// draws only a color it binds itself, so nothing builds a color scale or a
/// legend, and the color warns once as not drawn. The same holds when the
/// texts hold the color as their `group` key, the wire Python writes. A text
/// binding its own color draws it.
#[test]
fn a_text_only_chart_with_a_parent_color_warns_it_is_not_drawn() {
    let text = |encoding: serde_json::Value| {
        let mut layer = serde_json::json!({"mark": "text", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "text": {"field": "g"}}});
        for (key, value) in encoding.as_object().expect("an object") {
            layer["encoding"][key] = value.clone();
        }
        layer
    };
    let with_chart_color = |layers: serde_json::Value| {
        let mut spec = serde_json::to_value(column_gate_spec(serde_json::json!([]), layers)).expect("serializes");
        spec["encoding"]["color"] = serde_json::json!({"field": "g"});
        serde_json::from_value::<ChartSpec>(spec).expect("deserializes")
    };
    let group = serde_json::json!({"group": {"field": "g"}});
    for (case, spec) in [
        ("parent color", with_chart_color(serde_json::json!([text(serde_json::json!({})), text(serde_json::json!({}))]))),
        ("group key", with_chart_color(serde_json::json!([text(group.clone()), text(group)]))),
    ] {
        let prep = gate_prepare(&spec).unwrap_or_else(|e| panic!("{case}: {e}"));
        assert!(prep.provisional_scales.color.is_none(), "{case}: no color scale");
        assert!(prep.legend_entries.is_empty(), "{case}: no legend");
        let undrawn: Vec<String> = prep
            .warnings
            .iter()
            .filter(|w| matches!(w, RenderWarning::ChannelNotDrawn { .. }))
            .map(ToString::to_string)
            .collect();
        assert_eq!(
            undrawn,
            ["layer 0: mark_text: color 'g' has no effect: no layer's mark draws color (a text mark draws only a \
              color it binds itself), so it gets no scale and no legend; it was ignored"],
            "{case}"
        );
    }
    let own = with_chart_color(serde_json::json!([text(serde_json::json!({"color": {"field": "g"}})), text(serde_json::json!({}))]));
    let prep = gate_prepare(&own).expect("prepares");
    assert!(prep.provisional_scales.color.is_some(), "a text's own color draws");
    assert!(!prep.warnings.iter().any(|w| matches!(w, RenderWarning::ChannelNotDrawn { .. })));
}

/// A column only another `+` operand's rows hold is no statistic: a layer
/// whose rows are its operand's own snapshot (a branch over a published
/// `Identity` row marker) refuses with the plain message, though the other
/// operand's branch, its row marker and the unadvanced tail all hold the
/// column. Rejects classifying any named output as a statistic.
#[test]
fn a_column_only_another_operands_rows_hold_gets_the_plain_message() {
    let spec = column_gate_spec(
        serde_json::json!([
            {"type": "branch", "name": "left", "transforms": [
                {"type": "project", "fields": ["x", "y"]},
                {"type": "identity", "name": "rows"},
            ]},
            {"type": "branch", "name": "right", "transforms": [
                {"type": "project", "fields": ["x", "y", "z"]},
                {"type": "identity", "name": "rows"},
            ]},
            {"type": "branch", "name": "secondary", "input": "left_rows", "transforms": [{"type": "filter", "predicate": "true"}]},
        ]),
        serde_json::json!([
            {"mark": "point", "data_source": "left", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}}},
            {"mark": "point", "data_source": "right", "encoding": {"x": {"field": "x"}, "y": {"field": "z"}}},
            {"mark": "line", "data_source": "secondary", "reads_chart_data": true, "encoding": {"x": {"field": "x"}, "y": {"field": "z"}}},
        ]),
    );
    let err = gate_refusal(&spec);
    assert_eq!(
        err.to_string(),
        "layer 2: encoding 'y' references unknown column 'z' (not present in that layer's data)"
    );
}

/// The gate's refusal on a layer a desugaring call built names that call,
/// not an internal layer index the caller never wrote; a layer the caller
/// placed keeps its index.
#[test]
fn a_gate_refusal_on_a_desugared_layer_names_its_public_entry() {
    let mut spec = column_gate_spec(
        serde_json::json!([]),
        serde_json::json!([
            {"mark": "point", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "color": {"field": "nope"}}},
            {"mark": "point", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}}},
        ]),
    );
    spec.entry = Some("residplot".into());
    assert_eq!(
        gate_refusal(&spec).to_string(),
        "residplot: encoding 'color' references unknown column 'nope' (not present in that layer's data)"
    );
}

/// The rows each kind of layer starts from before its own statistic: a
/// layer on the chart's rows starts from them (every panel's under a shared
/// facet channel), a layer on a row output from that output, and a layer on
/// a statistic's output from the rows its scope's statistic consumed: the
/// row snapshot of the branch it is published under, or the chart's rows for
/// a top-level statistic.
#[test]
fn a_layers_pre_statistic_rows_are_the_rows_its_statistic_consumed() {
    let transforms: Vec<crate::transform::core::TransformSpec> = serde_json::from_value(serde_json::json!([
        {"type": "branch", "name": "b", "transforms": [
            {"type": "identity", "name": "marker"},
            {"type": "aggregate", "ops": [{"field": "x", "fn": "sum", "as": "total"}], "groupby": ["g"], "name": "box"},
        ]},
        {"type": "aggregate", "ops": [{"field": "x", "fn": "sum", "as": "total"}], "groupby": ["g"], "name": "stats"},
    ]))
    .expect("transforms");
    let rows = |n: usize| {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, false)])),
            vec![Arc::new(Float64Array::from(vec![0.0; n]))],
        )
        .expect("rows")
    };
    let primary = rows(1);
    let mut outputs = HashMap::new();
    for (key, n) in [(FINAL_OUTPUT_KEY, 2), ("b", 3), ("b_marker", 4), ("b_box", 5), ("stats", 6)] {
        outputs.insert(key.to_string(), rows(n));
    }
    let read = |source: Option<&str>, include_final: bool| {
        pre_statistic_rows(source, &transforms, &primary, &outputs, include_final).num_rows()
    };
    assert_eq!(read(None, false), 1, "chart rows");
    assert_eq!(read(None, true), 2, "every panel's chart rows");
    assert_eq!(read(Some("b"), false), 3, "a row output");
    assert_eq!(read(Some("b_marker"), false), 4, "a row snapshot");
    assert_eq!(read(Some("b_box"), false), 4, "a branch statistic's consumed rows: the branch's snapshot");
    assert_eq!(read(Some("stats"), false), 1, "a top-level statistic's consumed rows: the chart's");
}

/// The user's row outputs: every branch tail, every named Identity snapshot
/// at any nesting (the `+` operand's row marker, displot's rug data, a
/// threshold line's data), and the tail when no unnamed step advanced it.
#[test]
fn user_row_outputs_name_the_three_producer_shapes_and_the_unadvanced_tail() {
    let parse = |json: serde_json::Value| -> Vec<crate::transform::core::TransformSpec> {
        serde_json::from_value(json).expect("transforms")
    };
    let operand = parse(serde_json::json!([
        {"type": "branch", "name": "left", "transforms": [{"type": "filter", "predicate": "true"}, {"type": "identity", "name": "row"}]},
    ]));
    assert_eq!(user_row_outputs(&operand), ["left", "left_row", FINAL_OUTPUT_KEY]);
    let rug = parse(serde_json::json!([
        {"type": "identity", "name": "rug_data"},
        {"type": "aggregate", "ops": [{"field": "x", "fn": "sum", "as": "total"}], "groupby": [], "name": "hist"},
    ]));
    assert_eq!(user_row_outputs(&rug), ["rug_data", FINAL_OUTPUT_KEY]);
    let advanced = parse(serde_json::json!([
        {"type": "identity", "name": "threshold"},
        {"type": "aggregate", "ops": [{"field": "x", "fn": "sum", "as": "total"}], "groupby": []},
    ]));
    assert_eq!(user_row_outputs(&advanced), ["threshold"], "an advanced tail is a statistic's rows");
}

/// A data-aware sort orders a layered axis by the establishing layer's own
/// rows and its own opposite binding: a bar operand sorted by its own `y`
/// beside a rule over another frame orders its categories by the bar's
/// values, though the rule's rows (layer 0) hold the same category column
/// in another order. Rejects sorting over the chart rows.
#[test]
fn a_sort_orders_by_the_establishing_layers_own_rows_and_binding() {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("side", DataType::Utf8, false),
            Field::new("g", DataType::Utf8, false),
            Field::new("v", DataType::Float64, false),
            Field::new("w", DataType::Float64, false),
        ])),
        vec![
            Arc::new(StringArray::from(vec!["l", "l", "l", "r", "r", "r"])),
            Arc::new(StringArray::from(vec!["a", "b", "c", "a", "b", "c"])),
            Arc::new(Float64Array::from(vec![9.0, 1.0, 5.0, 0.0, 0.0, 0.0])),
            Arc::new(Float64Array::from(vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0])),
        ],
    )
    .expect("batch");
    let spec: ChartSpec = serde_json::from_value(serde_json::json!({
        "data": {"kind": "named", "name": "default"},
        "mark": "rule",
        "encoding": {"y": {"field": "v"}},
        "transforms": [
            {"type": "branch", "name": "left", "transforms": [{"type": "filter", "predicate": "datum.side == 'l'"}]},
            {"type": "branch", "name": "right", "transforms": [{"type": "filter", "predicate": "datum.side == 'r'"}]},
        ],
        "layers": [
            {"mark": "rule", "data_source": "left", "encoding": {"y": {"field": "v"}}},
            {"mark": "bar", "data_source": "right", "encoding": {"x": {"field": "g", "sort": "-y"}, "y": {"field": "w"}}},
        ],
    }))
    .expect("spec");
    let prep = prepare_render_inputs(
        &spec,
        &batch,
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport { width: 600.0, height: 400.0 },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .expect("prepares");
    match &prep.provisional_scales.x {
        crate::render::scale_resolve::ScaleKind::Ordinal(scale) => {
            assert_eq!(scale.ticks_internal(), ["c", "b", "a"], "ordered by the bar's own w, descending")
        }
        other => panic!("expected a categorical x axis, got {other:?}"),
    }
}

/// When no layer sharing the primary y axis draws y, the first independent-y
/// layer that does owns the primary axis: no secondary slot, and the primary
/// y is that layer's. Rejects an untitled 0..1 left axis beside the layer's
/// own right axis.
#[test]
fn the_only_y_drawer_owns_the_primary_axis_though_it_is_independent() {
    let spec = column_gate_spec(
        serde_json::json!([]),
        serde_json::json!([
            {"mark": "rule", "encoding": {"x": {"field": "x"}}},
            {"mark": "bar", "independent_y": true, "encoding": {"x": {"field": "z"}, "y": {"field": "g"}}},
        ]),
    );
    let prep = gate_prepare(&spec).expect("prepares");
    assert!(!prep.y_slot_plan.has_independent(), "no secondary axis");
    assert_eq!(prep.rendering_encoding().y.as_ref().map(|y| y.field.as_str()), Some("g"));
    assert!(matches!(prep.provisional_scales.y, crate::render::scale_resolve::ScaleKind::Ordinal(_)));
}

/// A raster image draws color through its cells: a color binding naming the
/// field they hold builds the color scale over the drawn cell values (and so
/// a colorbar), with no undrawn warning; one naming another field draws
/// nothing, gets no scale, and warns naming the field the cells hold.
#[test]
fn a_raster_image_draws_the_color_its_cells_hold() {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("z", DataType::Float64, false),
        ])),
        vec![
            Arc::new(Float64Array::from(vec![0.0, 0.0, 1.0, 1.0])),
            Arc::new(Float64Array::from(vec![0.0, 0.0, 1.0, 1.0])),
            Arc::new(Float64Array::from(vec![-4.0, -2.0, 6.0, 8.0])),
        ],
    )
    .expect("batch");
    let spec = |color: &str| -> ChartSpec {
        serde_json::from_value(serde_json::json!({
            "data": {"kind": "named", "name": "default"},
            "mark": "point",
            "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "color": {"field": color}},
            "transforms": [{"type": "raster", "x": "x", "y": "y", "aggregate": "mean", "field": "z", "resolution": 2, "name": "raster"}],
            "layers": [{"mark": "image", "data_source": "raster", "entry": "mark_raster", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}}}],
        }))
        .expect("spec")
    };
    let prepare = |spec: &ChartSpec| {
        prepare_render_inputs(
            spec,
            &batch,
            &crate::layout::ThemeInputs::default(),
            crate::layout::Viewport { width: 600.0, height: 400.0 },
            &crate::render::chart_config::ChartConfig::default(),
            None,
        )
        .expect("prepares")
    };
    let drawn = prepare(&spec("z"));
    match &drawn.provisional_scales.color {
        Some(crate::render::scale_resolve::ColorScale::Continuous { domain, .. }) => {
            assert_eq!(*domain, (-3.0, 7.0), "the drawn cell means, negatives kept")
        }
        other => panic!("expected a continuous color scale over the cells, got {other:?}"),
    }
    assert!(drawn.colorbar.is_some(), "a colorbar describes the cells");
    assert!(!drawn.warnings.iter().any(|w| matches!(w, RenderWarning::ChannelNotDrawn { .. })));

    let other = prepare(&spec("x"));
    assert!(other.provisional_scales.color.is_none());
    let undrawn: Vec<String> = other
        .warnings
        .iter()
        .filter(|w| matches!(w, RenderWarning::ChannelNotDrawn { .. }))
        .map(ToString::to_string)
        .collect();
    assert_eq!(
        undrawn,
        ["mark_raster: color 'x' has no effect: the image colors its cells by their 'z' values (field=) and does \
          not read color='x', so it gets no scale and no legend; it was ignored"]
    );
}

/// The colorbar of a raster drawing color through its cells spans the domain
/// the image draws: a count raster colored by its `count` cells runs from
/// zero to the densest cell, as the image's own scale does, so one scale
/// describes the fill. Rejects a colorbar over `[min, max]` beside a fill
/// anchored at zero.
#[test]
fn a_raster_colorbar_spans_the_domain_the_image_draws() {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ])),
        vec![
            Arc::new(Float64Array::from(vec![0.0, 0.1, 0.0, 1.0, 1.0])),
            Arc::new(Float64Array::from(vec![0.0, 0.1, 0.0, 1.0, 1.0])),
        ],
    )
    .expect("batch");
    let spec: ChartSpec = serde_json::from_value(serde_json::json!({
        "data": {"kind": "named", "name": "default"},
        "mark": "point",
        "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "color": {"field": "count"}},
        "transforms": [{"type": "raster", "x": "x", "y": "y", "aggregate": "count", "resolution": 2, "name": "raster"}],
        "layers": [{"mark": "image", "data_source": "raster", "entry": "mark_raster", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}}}],
    }))
    .expect("spec");
    let prep = prepare_render_inputs(
        &spec,
        &batch,
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport { width: 600.0, height: 400.0 },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .expect("prepares");
    let (lo, hi) = crate::render::marks::image::raster_cell_extent(&prep.transform_outputs["raster"]).expect("cells");
    assert_eq!((lo, hi), (2.0, 3.0), "the drawn counts");
    match &prep.provisional_scales.color {
        Some(crate::render::scale_resolve::ColorScale::Continuous { domain, .. }) => {
            assert_eq!(*domain, crate::render::marks::image::raster_cell_domain((lo, hi)));
            assert_eq!(*domain, (0.0, 3.0), "from zero, as the image draws");
        }
        other => panic!("expected a continuous color scale over the cells, got {other:?}"),
    }
}

use crate::render::scale_resolve::ColorScale;

/// Rows for the raster tests: a dense cluster near (1, 1), a sparse one near
/// (0, 0), a value `z` spanning negatives, a category `g` and a facet `f`
/// whose panels differ in their densest cell.
fn raster_rows() -> RecordBatch {
    let mut x = vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0];
    let mut y = x.clone();
    let mut f: Vec<&str> = vec!["m"; 7];
    f.extend(["n", "n"]);
    x.push(0.5);
    y.push(0.5);
    f.push("n");
    let n = x.len();
    let z: Vec<f64> = (0..n).map(|i| i as f64 - 4.0).collect();
    let g: Vec<&str> = (0..n).map(|i| if i % 2 == 0 { "a" } else { "b" }).collect();
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("z", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
            Field::new("f", DataType::Utf8, false),
        ])),
        vec![
            Arc::new(Float64Array::from(x)),
            Arc::new(Float64Array::from(y)),
            Arc::new(Float64Array::from(z)),
            Arc::new(StringArray::from(g)),
            Arc::new(StringArray::from(f)),
        ],
    )
    .expect("rows")
}

/// A raster wire as Python lowers `mark_raster`: the image layer on the
/// Raster output `raster`, `raster` the transform's JSON, `color` the
/// chart's color binding, `extra` more layers after the image, `facet` a
/// facet field.
fn raster_spec(raster: serde_json::Value, color: Option<serde_json::Value>, extra: serde_json::Value, facet: Option<&str>) -> ChartSpec {
    let mut encoding = serde_json::json!({"x": {"field": "x"}, "y": {"field": "y"}});
    if let Some(color) = color {
        encoding["color"] = color;
    }
    let mut layers = vec![serde_json::json!({
        "mark": "image", "data_source": "raster", "entry": "mark_raster",
        "encoding": {"x": {"field": "x"}, "y": {"field": "y"}},
    })];
    layers.extend(extra.as_array().cloned().unwrap_or_default());
    let mut spec = serde_json::json!({
        "data": {"kind": "named", "name": "default"},
        "mark": "point",
        "encoding": encoding,
        "transforms": [raster],
        "layers": layers,
    });
    if let Some(field) = facet {
        spec["facet"] = serde_json::json!({"field": field, "mode": {"kind": "wrap", "ncols": 2}});
    }
    serde_json::from_value(spec).expect("raster spec")
}

fn count_raster(log_scale: bool) -> serde_json::Value {
    serde_json::json!({"type": "raster", "x": "x", "y": "y", "aggregate": "count", "resolution": 2, "log_scale": log_scale, "name": "raster"})
}

fn prepare_rows(spec: &ChartSpec) -> Result<PreparedInputs, RenderError> {
    pyo3::Python::initialize();
    prepare_render_inputs(
        spec,
        &raster_rows(),
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport { width: 600.0, height: 400.0 },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
}

/// The pipeline's outputs are its own: no synthetic cells output is
/// published beside a raster, so a layer naming one is refused as an
/// unknown data source listing only pipeline outputs, a user output named
/// like one renders, and a column a layer lacks gets the remedy it gets
/// with no raster present. Rejects publishing the cells as an output.
#[test]
fn a_raster_publishes_no_output_of_its_own_cells() {
    let prep = prepare_rows(&raster_spec(count_raster(false), None, serde_json::json!([]), None)).expect("prepares");
    assert!(!prep.transform_outputs.keys().any(|key| key.ends_with("__cells")), "{:?}", prep.transform_outputs.keys());

    let naming = raster_spec(
        count_raster(false),
        None,
        serde_json::json!([{"mark": "point", "data_source": "raster__cells", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}}}]),
        None,
    );
    let err = prepare_rows(&naming).err().expect("refused").to_string();
    assert!(err.contains("data_source 'raster__cells' not found"), "{err}");
    assert!(!err.contains("raster__cells,") && !err.contains("raster__cells]"), "only pipeline outputs listed: {err}");

    let mut own = raster_spec(
        count_raster(false),
        None,
        serde_json::json!([{"mark": "point", "data_source": "raster__cells", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}}}]),
        None,
    );
    own.transforms.push(serde_json::from_value(serde_json::json!({"type": "branch", "name": "raster__cells", "transforms": [{"type": "filter", "predicate": "true"}]})).unwrap());
    prepare_rows(&own).expect("a user output named like the cells renders");

    let remedy = |with_raster: bool| {
        let mut spec = raster_spec(
            count_raster(false),
            None,
            serde_json::json!([{"mark": "point", "encoding": {"x": {"field": "x"}, "y": {"field": "count"}}}]),
            None,
        );
        if !with_raster {
            spec.layers.as_mut().unwrap().remove(0);
            spec.transforms.clear();
        }
        prepare_rows(&spec).err().expect("count is no column of the rows").to_string()
    };
    assert_eq!(remedy(true).replace("layer 1", "layer 0"), remedy(false), "the raster's cells are no statistic another layer computes");
}

/// One cell scale spans every panel of a faceted raster drawing no color,
/// as a bound color scale does: the `raster_cells` scale's domain is every
/// panel's cells, from zero, so a count colors alike in panels whose
/// densest cells differ. Rejects each panel's own extent.
#[test]
fn a_faceted_raster_drawing_no_color_shares_one_cell_scale() {
    let prep = prepare_rows(&raster_spec(count_raster(false), None, serde_json::json!([]), Some("f"))).expect("prepares");
    match &prep.provisional_scales.raster_cells {
        Some(ColorScale::Continuous { domain, .. }) => assert_eq!(*domain, (0.0, 5.0), "panel m's densest cell is 5, n's is 2"),
        other => panic!("expected one continuous cell scale, got {other:?}"),
    }
}

/// A raster's cells are numbers: a nominal or ordinal color naming them, or
/// another layer's categorical color establishing the chart's scale, has no
/// color for a cell, so each is refused naming the
/// image and why. An untyped or quantitative binding draws through a
/// continuous scale. Rejects the transparent image a categorical scale drew.
#[test]
fn a_categorical_color_over_a_rasters_cells_is_refused() {
    for kind in ["nominal", "ordinal"] {
        let spec = raster_spec(count_raster(false), Some(serde_json::json!({"field": "count", "type": kind})), serde_json::json!([]), None);
        assert_eq!(
            prepare_rows(&spec).err().expect("refused").to_string(),
            format!(
                "mark_raster: encoding 'color' names the raster's cell values 'count', which are quantitative, but it is \
                 declared {kind}. Drop type= (or the categorical color binding), or draw categorical cells with mark_rect."
            )
        );
    }
    let mut point_first = raster_spec(
        count_raster(false),
        None,
        serde_json::json!([{"mark": "point", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "color": {"field": "g"}}}]),
        None,
    );
    point_first.layers.as_mut().unwrap().rotate_left(1);
    point_first.layers.as_mut().unwrap()[1].encoding.color = Some(EncodingSpec { field: "count".into(), ..Default::default() });
    assert_eq!(
        prepare_rows(&point_first).err().expect("refused").to_string(),
        "mark_raster: encoding 'color' names the raster's cell values 'count', which are quantitative, but the chart's \
         color scale is categorical, established by layer 0 ('g'). Drop type= (or the categorical color binding), or \
         draw categorical cells with mark_rect."
    );
    for binding in [serde_json::json!({"field": "count"}), serde_json::json!({"field": "count", "type": "quantitative"})] {
        let prep = prepare_rows(&raster_spec(count_raster(false), Some(binding), serde_json::json!([]), None)).expect("draws");
        assert!(matches!(prep.provisional_scales.color, Some(ColorScale::Continuous { .. })));
    }
}

/// A count raster under `log_scale` keeps its raw cells and colors them
/// through a symlog scale, so a cell normalizes to log1p(v)/log1p(max), as
/// the image drew before, and its colorbar reads raw counts with the densest
/// cell on top. A `LinearScale` maps the counts linearly; a mean raster under
/// `log_scale` draws its negative cells, sign-preserving. Rejects a colorbar
/// labelled in log1p values.
#[test]
fn a_log_scale_raster_colors_raw_cells_through_symlog_and_its_colorbar_reads_counts() {
    use crate::render::scale_resolve::ColorTransform;
    let prep = prepare_rows(&raster_spec(count_raster(true), Some(serde_json::json!({"field": "count"})), serde_json::json!([]), None)).expect("draws");
    let Some(ColorScale::Continuous { domain, transform, .. }) = &prep.provisional_scales.color else { panic!("continuous") };
    assert_eq!(*domain, (0.0, 7.0));
    assert_eq!(*transform, ColorTransform::Symlog { constant: 1.0 });
    let t = transform.normalize(*domain, None, 2.0);
    assert!((t - 2.0_f64.ln_1p() / 7.0_f64.ln_1p()).abs() < 1e-12, "t={t}");
    let labels = &prep.colorbar.as_ref().expect("a colorbar").tick_labels;
    assert_eq!(labels.first().map(String::as_str), Some("0"));
    assert_eq!(labels.last().map(String::as_str), Some("7"), "{labels:?}");

    let linear = raster_spec(count_raster(true), Some(serde_json::json!({"field": "count", "scale": {"type": "linear"}})), serde_json::json!([]), None);
    let prep = prepare_rows(&linear).expect("draws");
    assert!(matches!(prep.provisional_scales.color, Some(ColorScale::Continuous { transform: ColorTransform::Linear, .. })));

    let mean = serde_json::json!({"type": "raster", "x": "x", "y": "y", "aggregate": "mean", "field": "z", "resolution": 2, "log_scale": true, "name": "raster"});
    let prep = prepare_rows(&raster_spec(mean, None, serde_json::json!([]), None)).expect("draws");
    let Some(scale @ ColorScale::Continuous { domain, .. }) = &prep.provisional_scales.raster_cells else { panic!("cell scale") };
    assert!(domain.0 < 0.0, "negative cells kept: {domain:?}");
    assert_ne!(scale.lookup_f64(domain.0), scale.lookup_f64(0.0), "a negative cell draws, not clamped to zero");
}

/// `Color(scale=SymlogScale())` maps through symlog on any mark, as the
/// positional symlog scale does.
#[test]
fn a_symlog_color_scale_maps_through_symlog_on_a_point_chart() {
    use crate::render::scale_resolve::ColorTransform;
    let spec = column_gate_spec(
        serde_json::json!([]),
        serde_json::json!([
            {"mark": "point", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "color": {"field": "z", "scale": {"type": "symlog", "constant": 1.0}}}},
        ]),
    );
    let prep = gate_prepare(&spec).expect("prepares");
    let Some(ColorScale::Continuous { transform, domain, .. }) = &prep.provisional_scales.color else { panic!("continuous") };
    assert_eq!(*transform, ColorTransform::Symlog { constant: 1.0 });
    assert_eq!(*domain, (10.0, 40.0));
}


/// Each facet panel's raster image as its decoded RGBA8 cells, in panel
/// order, with the prepared inputs the scene drew from.
fn raster_panel_cells(spec: &ChartSpec) -> (PreparedInputs, Vec<Vec<[u8; 4]>>) {
    let prep = prepare_rows(spec).expect("prepares");
    let scene = gate_render(spec, &raster_rows()).0.expect("renders");
    let cells = scene
        .panels
        .iter()
        .flat_map(|panel| panel.marks.iter().flat_map(|batch| batch.nodes.iter()))
        .filter_map(|node| match node {
            ferrum_scene::SceneNode::Image { data: ferrum_scene::ImageData::Inline { bytes, .. }, .. } => {
                let mut reader = png::Decoder::new(std::io::Cursor::new(bytes.as_slice())).read_info().expect("a png");
                let mut buf = vec![0; reader.output_buffer_size().expect("sized")];
                let info = reader.next_frame(&mut buf).expect("a frame");
                Some(buf[..info.buffer_size()].chunks_exact(4).map(|px| [px[0], px[1], px[2], px[3]]).collect())
            }
            _ => None,
        })
        .collect();
    (prep, cells)
}

fn rgba(scale: &ColorScale, value: f64) -> [u8; 4] {
    let c = scale.lookup_f64(value).expect("a color");
    [c.red, c.green, c.blue, c.alpha]
}

const EMPTY_CELL: [u8; 4] = [0, 0, 0, 0];

/// The drawn cells of `cells`, sorted, with the empty ones dropped.
fn drawn(cells: &[[u8; 4]]) -> Vec<[u8; 4]> {
    let mut drawn: Vec<[u8; 4]> = cells.iter().copied().filter(|px| *px != EMPTY_CELL).collect();
    drawn.sort();
    drawn
}

/// A raster count colored through a discretizing scale (`Quantize`,
/// `Threshold`, `BinOrdinal` with explicit `bins`, a binned color's
/// `BinOrdinal` with `bin`, `Quantile`) steps its cells through that one
/// scale, over the cells' zero-anchored domain (a quantile's cut-points and
/// a binned color's edges are taken from the cells' values): the cells
/// holding 3 and 7 draw distinct bucket colors, empty cells stay
/// transparent, and the legend is the scale's stepped bar (two stops per
/// swatch at its band edges, one label per bound). Rejects the continuous
/// ramp the cells drew under any binding, a quantile cut from no sample, and
/// a binned color that read no cell.
#[test]
fn a_discretizing_color_steps_a_rasters_cells_and_draws_its_stepped_legend() {
    let cases = [
        (serde_json::json!({"type": "quantize", "range": ["#ff0000", "#00ff00", "#0000ff"]}), vec![0.0, 7.0 / 3.0, 14.0 / 3.0, 7.0]),
        (serde_json::json!({"type": "threshold", "domain": [4.0]}), vec![0.0, 4.0, 7.0]),
        (serde_json::json!({"type": "bin-ordinal", "bins": [2.0, 5.0]}), vec![0.0, 2.0, 5.0, 7.0]),
        (serde_json::json!({"type": "bin-ordinal", "bin": {"bin_count": 2, "nice": false}}), vec![3.0, 5.0, 7.0]),
        (serde_json::json!({"type": "quantile", "range": [0.0, 1.0]}), vec![3.0, 5.0, 7.0]),
    ];
    for (scale, bounds) in cases {
        let color = serde_json::json!({"field": "count", "scale": scale});
        let (prep, panels) = raster_panel_cells(&raster_spec(count_raster(false), Some(color), serde_json::json!([]), None));
        let Some(stepped @ ColorScale::Discretizing(buckets)) = &prep.provisional_scales.color else {
            panic!("{scale}: expected a discretizing scale, got {:?}", prep.provisional_scales.color)
        };
        assert_eq!(buckets.bounds(), bounds.as_slice(), "{scale}");
        let [cells] = panels.as_slice() else { panic!("{scale}: one image, got {}", panels.len()) };
        assert_eq!(cells.iter().filter(|px| **px == EMPTY_CELL).count(), 2, "{scale}: the empty cells stay transparent");
        let mut expected = vec![rgba(stepped, 3.0), rgba(stepped, 7.0)];
        expected.sort();
        assert_eq!(drawn(cells), expected, "{scale}");
        assert_ne!(expected[0], expected[1], "{scale}: 3 and 7 fall in distinct buckets");

        let colorbar = prep.colorbar.as_ref().expect("a stepped legend");
        let k = buckets.colors().len();
        assert_eq!(colorbar.stops.len(), 2 * k, "{scale}");
        for (i, pair) in colorbar.stops.chunks_exact(2).enumerate() {
            assert_eq!(pair[0].1, pair[1].1, "{scale}: swatch {i} is flat");
            assert_eq!((pair[0].0, pair[1].0), (i as f64 / k as f64, (i + 1) as f64 / k as f64), "{scale}");
        }
        assert_eq!(colorbar.tick_labels.len(), k + 1, "{scale}: {:?}", colorbar.tick_labels);
    }
}

/// Faceted rasters share one discretizing scale's steps: its domain spans
/// every panel's cells (densest 5 in panel m, 2 in n), so a cell holding 2
/// draws the same bucket color in both panels. Rejects stepping each panel
/// over its own cells, under which n's densest cell took the top swatch.
#[test]
fn faceted_rasters_share_a_discretizing_scales_steps() {
    let color = serde_json::json!({"field": "count", "scale": {"type": "quantize", "range": ["#ff0000", "#00ff00", "#0000ff"]}});
    let (prep, panels) = raster_panel_cells(&raster_spec(count_raster(false), Some(color), serde_json::json!([]), Some("f")));
    let Some(stepped @ ColorScale::Discretizing(buckets)) = &prep.provisional_scales.color else { panic!("discretizing") };
    assert_eq!(buckets.bounds(), [0.0, 5.0 / 3.0, 10.0 / 3.0, 5.0]);
    let [m, n] = panels.as_slice() else { panic!("two panels, got {}", panels.len()) };
    let sorted = |mut v: Vec<[u8; 4]>| {
        v.sort();
        v
    };
    assert_eq!(drawn(m), sorted(vec![rgba(stepped, 2.0), rgba(stepped, 5.0)]));
    assert_eq!(drawn(n), sorted(vec![rgba(stepped, 1.0), rgba(stepped, 2.0)]));
    assert_eq!(rgba(stepped, 2.0), [0, 255, 0, 255], "a 2 is the middle swatch in both panels");
}

/// Faceted rasters share one binned color's edges, cut from every panel's
/// cells (1..5 over panels m and n), so a cell holding 2 draws the same bin
/// color in both panels. Rejects edges cut from each panel's own cells.
#[test]
fn faceted_rasters_share_a_binned_colors_edges() {
    let color = serde_json::json!({"field": "count", "scale": {"type": "bin-ordinal", "bin": {"bin_count": 2, "nice": false}}});
    let (prep, panels) = raster_panel_cells(&raster_spec(count_raster(false), Some(color), serde_json::json!([]), Some("f")));
    let Some(stepped @ ColorScale::Discretizing(buckets)) = &prep.provisional_scales.color else { panic!("discretizing") };
    assert_eq!(buckets.bounds(), [1.0, 3.0, 5.0]);
    let [m, n] = panels.as_slice() else { panic!("two panels, got {}", panels.len()) };
    let sorted = |mut v: Vec<[u8; 4]>| {
        v.sort();
        v
    };
    assert_eq!(drawn(m), sorted(vec![rgba(stepped, 2.0), rgba(stepped, 5.0)]));
    assert_eq!(drawn(n), sorted(vec![rgba(stepped, 1.0), rgba(stepped, 2.0)]));
}

/// Rows for a binned-color area: `x` 0..2 for each of two bins of `z`
/// (`bin_count` 2 without nicing over 1..9 cuts at 5). The high bin's rows
/// come first and every `z` is distinct, so first-appearance grouping, raw
/// category grouping and bin grouping all differ. `y` is 1 in the high bin
/// and 2 in the low one.
fn binned_area_rows() -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("z", DataType::Float64, false),
        ])),
        vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0])),
            Arc::new(Float64Array::from(vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0])),
            Arc::new(Float64Array::from(vec![8.0, 9.0, 8.5, 1.0, 2.0, 1.5])),
        ],
    )
    .unwrap()
}

/// An area chart whose color is the binned `z`, stacked when `stack`.
fn binned_area_spec(stack: bool) -> ChartSpec {
    let mut spec = serde_json::json!({
        "data": {"kind": "named", "name": "default"},
        "mark": "area",
        "encoding": {
            "x": {"field": "x"},
            "y": {"field": "y"},
            "color": {"field": "z", "scale": {"type": "bin-ordinal", "bin": {"bin_count": 2, "nice": false}}},
        },
    });
    if stack {
        spec["position"] = serde_json::json!({"type": "stack"});
    }
    serde_json::from_value(spec).expect("the area spec deserializes")
}

/// Each closed area band the scene draws: its fill and its vertices.
fn area_bands(scene: &ferrum_scene::SceneGraph) -> Vec<(ferrum_scene::Color, Vec<(f64, f64)>)> {
    scene
        .panels
        .iter()
        .flat_map(|panel| panel.marks.iter().flat_map(|batch| batch.nodes.iter()))
        .filter_map(|node| match node {
            ferrum_scene::SceneNode::Path { commands, style, closed: true, .. } => {
                let points = commands
                    .iter()
                    .filter_map(|cmd| match cmd {
                        ferrum_scene::PathCmd::MoveTo { x, y } | ferrum_scene::PathCmd::LineTo { x, y } => Some((*x, *y)),
                        _ => None,
                    })
                    .collect();
                Some((style.fill.expect("a filled band"), points))
            }
            _ => None,
        })
        .collect()
}

/// A binned color on an area groups the area by bin: one band per bin (two,
/// where the six distinct `z` values made six one-point groups that drew
/// nothing), drawn low bin first, each band filled with its bin's step of
/// the binned scale, under one legend of the bin ranges. Rejects the
/// categorical scale an area forced on its color, and first-appearance
/// band order (the high bin's rows come first).
#[test]
fn a_binned_color_on_an_area_draws_one_band_per_bin_in_bin_order() {
    let spec = binned_area_spec(false);
    let p = prep(&spec, &binned_area_rows());
    let Some(ColorScale::Discretizing(buckets)) = &p.provisional_scales.color else {
        panic!("expected the binned scale, got {:?}", p.provisional_scales.color)
    };
    assert_eq!(buckets.bounds(), [1.0, 5.0, 9.0]);
    assert!(p.legend_entries.is_empty(), "no categorical legend: {:?}", p.legend_entries);
    assert_eq!(p.colorbar.as_ref().expect("the bin legend").tick_labels, vec!["1", "5", "9"]);

    let scene = gate_render(&spec, &binned_area_rows()).0.expect("renders");
    let bands = area_bands(&scene);
    assert_eq!(bands.len(), 2, "one band per bin");
    for (bin, (fill, _)) in bands.iter().enumerate() {
        let step = crate::render::draw::to_scene_color(buckets.colors()[bin]);
        assert_eq!((fill.r, fill.g, fill.b), (step.r, step.g, step.b), "band {bin} takes bin {bin}'s step");
    }
    // The low bin's band (y 2) reaches higher on the panel than the high
    // bin's (y 1): its top vertex has the smaller pixel y.
    let top = |points: &[(f64, f64)]| points.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    assert!(top(&bands[0].1) < top(&bands[1].1), "the first band drawn is the low bin's");
}

/// A stacked area under a binned color stacks one layer per bin, low bin at
/// the bottom: the high bin's band sits on the low bin's top, each filled
/// with its bin's step. Rejects the first-appearance stack, which put the
/// high bin (whose rows come first) at the bottom.
#[test]
fn a_stacked_area_under_a_binned_color_stacks_one_layer_per_bin_low_bin_first() {
    let spec = binned_area_spec(true);
    let scene = gate_render(&spec, &binned_area_rows()).0.expect("renders");
    let bands = area_bands(&scene);
    assert_eq!(bands.len(), 2, "one stacked layer per bin");
    let p = prep(&spec, &binned_area_rows());
    // (bottom, top) pixel rows of a band; pixel y grows downward.
    let span = |points: &[(f64, f64)]| {
        let bottom = points.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
        let top = points.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
        (bottom, top)
    };
    let (low, high) = (span(&bands[0].1), span(&bands[1].1));
    assert!((low.1 - high.0).abs() < 1e-6, "the high bin sits on the low bin's top: {low:?} {high:?}");
    assert!(low.0 > high.0, "the low bin is the bottom layer: {low:?} {high:?}");
    let Some(ColorScale::Discretizing(buckets)) = &p.provisional_scales.color else { panic!("binned") };
    for (bin, (fill, _)) in bands.iter().enumerate() {
        let step = crate::render::draw::to_scene_color(buckets.colors()[bin]);
        assert_eq!((fill.r, fill.g, fill.b), (step.r, step.g, step.b), "layer {bin} takes bin {bin}'s step");
    }
}

/// A dodged bar under a binned color takes one slot per bin, low bin on the
/// left: 8 distinct `z` values over 2 bins (`bin_count` 2 without nicing over
/// 1..14 cuts at 7.5) put each band's 4 bars in 2 slots, the left one filled
/// with the low bin's step. Rejects one slot per raw value (4 per band).
#[test]
fn a_dodge_under_a_binned_color_takes_one_slot_per_bin_in_bin_order() {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, false),
            Field::new("y", DataType::Float64, false),
            Field::new("z", DataType::Float64, false),
        ])),
        vec![
            Arc::new(StringArray::from(vec!["a", "a", "a", "a", "b", "b", "b", "b"])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])),
            Arc::new(Float64Array::from(vec![11.0, 1.0, 12.0, 2.0, 13.0, 3.0, 14.0, 4.0])),
        ],
    )
    .unwrap();
    let spec: ChartSpec = serde_json::from_value(serde_json::json!({
        "data": {"kind": "named", "name": "default"},
        "mark": "bar",
        "position": {"type": "dodge"},
        "encoding": {
            "x": {"field": "g"},
            "y": {"field": "y"},
            "color": {"field": "z", "scale": {"type": "bin-ordinal", "bin": {"bin_count": 2, "nice": false}}},
        },
    }))
    .expect("the dodged spec deserializes");
    let p = prep(&spec, &batch);
    let Some(ColorScale::Discretizing(buckets)) = &p.provisional_scales.color else { panic!("binned") };
    assert_eq!(buckets.bounds(), [1.0, 7.5, 14.0]);
    let scene = gate_render(&spec, &batch).0.expect("renders");
    let mut bars: Vec<(f64, ferrum_scene::Color)> = scene
        .panels
        .iter()
        .flat_map(|panel| panel.marks.iter().flat_map(|batch| batch.nodes.iter()))
        .filter_map(|node| match node {
            ferrum_scene::SceneNode::Rect { x, style, .. } => style.fill.map(|fill| (*x, fill)),
            _ => None,
        })
        .collect();
    assert_eq!(bars.len(), 8, "every bar draws");
    bars.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut slots: Vec<f64> = bars.iter().map(|bar| bar.0).collect();
    slots.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    assert_eq!(slots.len(), 4, "two slots in each of two bands: {slots:?}");
    for (slot, bin) in [(0, 0), (1, 1), (2, 0), (3, 1)] {
        let step = crate::render::draw::to_scene_color(buckets.colors()[bin]);
        let fills: Vec<_> = bars.iter().filter(|bar| (bar.0 - slots[slot]).abs() < 1e-6).map(|bar| (bar.1.r, bar.1.g, bar.1.b)).collect();
        assert_eq!(fills, vec![(step.r, step.g, step.b); 2], "slot {slot} holds bin {bin}'s two bars");
    }
}

/// `raster_rows` with a numeric `count` column a sibling can color by.
fn raster_rows_with_count() -> RecordBatch {
    let rows = raster_rows();
    let count: Vec<f64> = (0..rows.num_rows()).map(|i| i as f64).collect();
    let mut fields: Vec<Field> = rows.schema().fields().iter().map(|field| field.as_ref().clone()).collect();
    fields.push(Field::new("count", DataType::Float64, false));
    let mut columns = rows.columns().to_vec();
    columns.push(Arc::new(Float64Array::from(count)));
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
}

/// A raster beside a sibling colored by the field its cells hold (`count`
/// for a count raster, `z` for a mean of `z`) draws its cells, by `+` in
/// either order and by the Layer door (a point layer binding only color): an
/// image's color is its own binding, never a sibling's, so it maps its cells
/// through its own cell scale while the sibling keeps its color scale and
/// legend. Rejects the image that drew nothing, without a warning, when the
/// sibling's binding was read as the image's.
#[test]
fn a_raster_beside_a_sibling_colored_by_its_value_field_draws_its_cells() {
    let mean = serde_json::json!({"type": "raster", "x": "x", "y": "y", "aggregate": "mean", "field": "z", "resolution": 2, "name": "raster"});
    let xy = serde_json::json!({"x": {"field": "x"}, "y": {"field": "y"}});
    for (raster, field) in [(count_raster(false), "count"), (mean, "z")] {
        let colored = serde_json::json!({"mark": "point", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "color": {"field": field}}});
        let layer_door = serde_json::json!({"mark": "point", "encoding": {"color": {"field": field}}});
        let raster_first = raster_spec(raster.clone(), None, serde_json::json!([colored]), None);
        let mut point_first = raster_first.clone();
        point_first.layers.as_mut().unwrap().rotate_left(1);
        let door = raster_spec(raster.clone(), None, serde_json::json!([layer_door]), None);
        for (door_name, spec) in [("r + p", raster_first), ("p + r", point_first), ("Layer", door)] {
            assert_eq!(spec.encoding, serde_json::from_value(xy.clone()).unwrap(), "{door_name}: no chart-level color");
            pyo3::Python::initialize();
            let prep = prepare_render_inputs(
                &spec,
                &raster_rows_with_count(),
                &crate::layout::ThemeInputs::default(),
                crate::layout::Viewport { width: 600.0, height: 400.0 },
                &crate::render::chart_config::ChartConfig::default(),
                None,
            )
            .expect("prepares");
            assert!(
                matches!(prep.provisional_scales.raster_cells, Some(ColorScale::Continuous { .. })),
                "{door_name} {field}: the image's own cell scale, got {:?}",
                prep.provisional_scales.raster_cells
            );
            assert!(matches!(prep.provisional_scales.color, Some(ColorScale::Continuous { .. })), "{door_name} {field}: the point's color");
            assert!(prep.colorbar.is_some(), "{door_name} {field}: the point's color legend");
            let (scene, warnings) = gate_render(&spec, &raster_rows_with_count());
            let images: Vec<_> = scene
                .expect("renders")
                .panels
                .iter()
                .flat_map(|panel| panel.marks.iter().flat_map(|batch| batch.nodes.clone()))
                .filter(|node| matches!(node, ferrum_scene::SceneNode::Image { .. }))
                .collect();
            assert_eq!(images.len(), 1, "{door_name} {field}: {warnings:?}");
        }
    }
}

/// A layered spec over a count raster: its layers in order, the chart's
/// mark the first layer's, as Python lowers `a + b`.
fn layered_over_raster(layers: serde_json::Value) -> ChartSpec {
    let mark = layers[0]["mark"].clone();
    serde_json::from_value(serde_json::json!({
        "data": {"kind": "named", "name": "default"}, "mark": mark,
        "encoding": {"x": {"field": "x"}, "y": {"field": "y"}},
        "transforms": [count_raster(false)],
        "layers": layers,
    }))
    .expect("layered spec")
}

/// Only an area layer that draws color groups color by category, so the
/// operand order never changes the outcome: an uncolored area beside a
/// raster colored by its cells renders in either order (the cells through a
/// continuous scale), an area colored by `g` beside it is refused in either
/// order naming the area's layer and field, and a point's quantitative color
/// beside an uncolored area stays continuous in either order. Rejects keying
/// the grouping on the chart's mark (the first operand's), under which
/// `area + raster` refused and `raster + area` rendered.
#[test]
fn an_area_groups_color_by_category_only_when_it_draws_color() {
    let xy = serde_json::json!({"x": {"field": "x"}, "y": {"field": "y"}});
    let area = |color: Option<&str>| {
        let mut encoding = xy.clone();
        if let Some(field) = color {
            encoding["color"] = serde_json::json!({"field": field});
        }
        serde_json::json!({"mark": "area", "encoding": encoding})
    };
    let image = serde_json::json!({"mark": "image", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "color": {"field": "count"}}, "data_source": "raster", "entry": "mark_raster"});
    let point = serde_json::json!({"mark": "point", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "color": {"field": "z"}}});

    for layers in [serde_json::json!([area(None), image.clone()]), serde_json::json!([image.clone(), area(None)])] {
        let prep = prepare_rows(&layered_over_raster(layers.clone())).expect("renders");
        assert!(matches!(prep.provisional_scales.color, Some(ColorScale::Continuous { .. })), "{layers}: {:?}", prep.provisional_scales.color);
    }
    for (layers, layer) in [(serde_json::json!([area(Some("g")), image.clone()]), 0), (serde_json::json!([image, area(Some("g"))]), 1)] {
        assert_eq!(
            prepare_rows(&layered_over_raster(layers.clone())).err().map(|e| e.to_string()),
            Some(format!(
                "mark_raster: encoding 'color' names the raster's cell values 'count', which are quantitative, but the \
                 chart's color scale is categorical, established by layer {layer} ('g'). Drop type= (or the categorical \
                 color binding), or draw categorical cells with mark_rect."
            )),
            "{layers}"
        );
    }
    for layers in [serde_json::json!([area(None), point.clone()]), serde_json::json!([point.clone(), area(None)])] {
        let prep = prepare_rows(&layered_over_raster(layers.clone())).expect("renders");
        assert!(matches!(prep.provisional_scales.color, Some(ColorScale::Continuous { .. })), "{layers}: {:?}", prep.provisional_scales.color);
    }
    let prep = prepare_rows(&layered_over_raster(serde_json::json!([point, area(Some("z"))]))).expect("renders");
    assert!(matches!(prep.provisional_scales.color, Some(ColorScale::Categorical { .. })), "an area drawing color groups it");
}

/// A frame faceted on `f` whose grouped statistic drops null `h` keys.
fn null_key_facet_run(
    transform: &str,
    f: Vec<&str>,
    h: Vec<Option<&str>>,
    v: Vec<f64>,
) -> Result<TransformRun, RenderError> {
    let json = format!(
        r#"{{"data":{{"kind":"named","name":"default"}},"mark":"point","encoding":{{"x":{{"field":"v"}},"y":{{"field":"v"}}}},
        "transforms":[{transform}],
        "facet":{{"field":"f","mode":{{"kind":"wrap","ncols":2}}}}}}"#
    );
    let spec: crate::spec::chart::ChartSpec = serde_json::from_str(&json).expect("valid spec");
    let schema = Arc::new(Schema::new(vec![
        Field::new("f", DataType::Utf8, true),
        Field::new("h", DataType::Utf8, true),
        Field::new("v", DataType::Float64, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(f)),
            Arc::new(StringArray::from(h)),
            Arc::new(Float64Array::from(v)),
        ],
    )
    .expect("schema and columns agree");
    let theme = crate::layout::ThemeInputs::default();
    let viewport = crate::layout::Viewport { width: 600.0, height: 400.0 };
    run_transforms_with_facet(&spec, &batch, viewport, &theme, None)
}

const NULL_KEY_TRANSFORMS: [(&str, &str); 2] = [
    ("LetterValue", r#"{"type":"letter_value","value":"v","groupby":["h"],"k_depth":{"type":"full"}}"#),
    ("QQ", r#"{"type":"qq","field":"v","groupby":["h"],"emit_line":false}"#),
];

/// A facet panel whose every `h` key is null is emptied by the null-key drop
/// and degrades to an empty panel, warned, while the other panel renders: the
/// refusal belongs to the whole chart left with nothing, not to one panel.
#[test]
fn a_facet_panel_emptied_by_the_null_key_drop_degrades_and_warns() {
    for (name, transform) in NULL_KEY_TRANSFORMS {
        let (outputs, warnings, _, _) = null_key_facet_run(
            transform,
            vec!["p", "p", "p", "q", "q"],
            vec![Some("x"), Some("x"), Some("x"), None, None],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        )
        .unwrap_or_else(|e| panic!("{name}: one emptied panel must not fail the chart: {e}"));
        let out = outputs.get(FINAL_OUTPUT_KEY).expect("the tail is published");
        assert!(out.num_rows() > 0, "{name}: panel p renders");
        assert!(str_col(out, "f").iter().all(|f| f == "p"), "{name}: only panel p has rows");
        assert_eq!(
            warnings.iter().map(|w| w.to_string()).collect::<Vec<_>>(),
            [format!(
                "{name}: 2 rows have a null 'h' and belong to no group, so they were dropped; \
                 fill or filter the nulls in 'h'"
            )],
            "{name}"
        );
    }
}

/// Two panels each dropping one null-key row report the chart's count once:
/// one warning saying 2 rows, not two identical warnings saying 1 that
/// Python's warning filter would collapse into one.
#[test]
fn the_null_key_drop_count_sums_across_facet_panels() {
    for (name, transform) in NULL_KEY_TRANSFORMS {
        let (_, warnings, _, _) = null_key_facet_run(
            transform,
            vec!["p", "p", "p", "q", "q", "q"],
            vec![Some("x"), Some("x"), None, Some("x"), Some("x"), None],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            warnings.iter().map(|w| w.to_string()).collect::<Vec<_>>(),
            [format!(
                "{name}: 2 rows have a null 'h' and belong to no group, so they were dropped; \
                 fill or filter the nulls in 'h'"
            )],
            "{name}"
        );
    }
}

/// Rows whose binned color `v` puts two distinct values in each of four
/// unit bins (1..5) for each `g`, every (g, v) twice: 32 rows, 16 distinct
/// (g, v) pairs, 8 (g, bin) pairs. `y` is the row number from 1, `x` from 0.
fn binned_stat_rows() -> RecordBatch {
    let vs = [1.1, 1.6, 2.2, 2.7, 3.1, 3.8, 4.4, 4.9];
    let mut g = Vec::new();
    let mut v = Vec::new();
    for group in ["a", "b"] {
        for value in vs {
            for _ in 0..2 {
                g.push(group);
                v.push(value);
            }
        }
    }
    let n = g.len();
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, false),
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("v", DataType::Float64, false),
        ])),
        vec![
            Arc::new(StringArray::from(g)),
            Arc::new(Float64Array::from((0..n).map(|i| i as f64).collect::<Vec<_>>())),
            Arc::new(Float64Array::from((1..=n).map(|i| i as f64).collect::<Vec<_>>())),
            Arc::new(Float64Array::from(v)),
        ],
    )
    .unwrap()
}

/// The marker that ends a chain's user steps: the steps after it were
/// lowered from layers' encodings and marks.
fn statistics_marker() -> serde_json::Value {
    serde_json::json!({"type": "identity", "statistics": true})
}

/// The binned color over `v` with four unit bins, 1..5.
fn unit_binned_v() -> serde_json::Value {
    serde_json::json!({"field": "v", "scale": {"type": "bin-ordinal", "bin": {"bin_width": 1.0, "extent": [1.0, 5.0], "nice": false}}})
}

/// The distinct values of `name` in `batch`, ascending.
fn distinct_f64(batch: &RecordBatch, name: &str) -> Vec<f64> {
    let mut values = float_col(batch, name);
    values.sort_by(f64::total_cmp);
    values.dedup();
    values
}

/// `bar(x='g', y='mean(y)', color=binned v)` as the flat wire lowers it,
/// under `position` (`None` for none).
fn binned_mean_bar_spec(position: Option<&str>) -> ChartSpec {
    let mut spec = serde_json::json!({
        "data": {"kind": "named", "name": "default"},
        "mark": "bar",
        "transforms": [statistics_marker(), {"type": "aggregate", "ops": [{"field": "y", "fn": "mean", "as": "mean_y"}], "groupby": ["g", "v"]}],
        "encoding": {"x": {"field": "g"}, "y": {"field": "mean_y"}, "color": unit_binned_v()},
    });
    if let Some(position) = position {
        spec["position"] = serde_json::json!({"type": position});
    }
    serde_json::from_value(spec).expect("the binned mean bar deserializes")
}

/// Every bar a scene draws: `(x, y, w, h, fill)`.
fn scene_bars(scene: &ferrum_scene::SceneGraph) -> Vec<(f64, f64, f64, f64, (u8, u8, u8))> {
    scene
        .panels
        .iter()
        .flat_map(|panel| panel.marks.iter().flat_map(|batch| batch.nodes.iter()))
        .filter_map(|node| match node {
            ferrum_scene::SceneNode::Rect { x, y, w, h, style, .. } => style.fill.map(|c| (*x, *y, *w, *h, (c.r, c.g, c.b))),
            _ => None,
        })
        .collect()
}

/// The fill a binned scale draws its bin `bin` in.
fn bin_fill(buckets: &crate::render::scale_resolve::DiscretizedColors, bin: usize) -> (u8, u8, u8) {
    let step = crate::render::draw::to_scene_color(buckets.colors()[bin]);
    (step.r, step.g, step.b)
}

/// An encoding aggregate grouped by a binned color groups by its bins: one
/// row per (g, bin), carrying the bin start, its mean over the bin's four
/// rows. Rejects grouping by the raw value (16 rows, two per bin).
#[test]
fn an_encoding_aggregate_under_a_binned_color_groups_by_bin() {
    let p = prep(&binned_mean_bar_spec(None), &binned_stat_rows());
    let out = &p.transform_outputs[FINAL_OUTPUT_KEY];
    assert_eq!(out.num_rows(), 8, "one row per (g, bin)");
    assert_eq!(distinct_f64(out, "v"), [1.0, 2.0, 3.0, 4.0]);
    let means: Vec<(f64, f64)> = float_col(out, "v").into_iter().zip(float_col(out, "mean_y")).collect();
    assert_eq!(&means[..4], [(1.0, 2.5), (2.0, 6.5), (3.0, 10.5), (4.0, 14.5)], "group a's bin means");
    let Some(ColorScale::Discretizing(buckets)) = &p.provisional_scales.color else { panic!("binned") };
    assert_eq!(buckets.bounds(), [1.0, 2.0, 3.0, 4.0, 5.0]);
}

/// A stacked mean bar under a binned color draws one segment per (g, bin),
/// low bin at the bottom, each segment its bin's step; a dodged one draws
/// one slot per bin in each band with no two bars sharing a slot. Rejects
/// the 16 bars, two per bin, that grouping by the raw value drew (stacked
/// as separate segments, dodged two to a slot).
#[test]
fn a_stacked_or_dodged_mean_bar_under_a_binned_color_draws_one_bar_per_bin() {
    let rows = binned_stat_rows();
    for position in ["stack", "dodge"] {
        let spec = binned_mean_bar_spec(Some(position));
        let p = prep(&spec, &rows);
        let Some(ColorScale::Discretizing(buckets)) = &p.provisional_scales.color else { panic!("binned") };
        let scene = gate_render(&spec, &rows).0.expect("renders");
        let mut bars = scene_bars(&scene);
        assert_eq!(bars.len(), 8, "{position}: one bar per (g, bin)");
        bars.sort_by(|a, b| a.0.total_cmp(&b.0).then(b.1.total_cmp(&a.1)));
        let fills: Vec<(u8, u8, u8)> = bars.iter().map(|bar| bar.4).collect();
        let in_bin_order: Vec<(u8, u8, u8)> = (0..2).flat_map(|_| (0..4).map(|bin| bin_fill(buckets, bin))).collect();
        assert_eq!(fills, in_bin_order, "{position}: each band draws its bins low to high");
        let mut slots: Vec<f64> = bars.iter().map(|bar| bar.0).collect();
        slots.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
        assert_eq!(slots.len(), if position == "stack" { 2 } else { 8 }, "{position}: {slots:?}");
    }
}

/// Every statistic that groups by a binned color makes one group per bin:
/// an aggregating positional bin, a density, a boxplot's statistics, a
/// violin and a smooth, each read by a layer that draws the binned color.
/// Rejects one statistic per raw value (eight distinct `v`).
#[test]
fn every_statistic_grouped_by_a_binned_color_makes_one_group_per_bin() {
    let cases = [
        (serde_json::json!({"type": "bin", "field": "x", "bin_count": 4, "nice": false, "groupby": ["v"]}), "bin_start", "count"),
        (serde_json::json!({"type": "kde", "field": "y", "bandwidth": {"kind": "scott"}, "n": 5, "groupby": ["v"]}), "value", "density"),
        (serde_json::json!({"type": "box_stats", "field": "y", "groupby": ["g", "v"]}), "median", "q1"),
        (serde_json::json!({"type": "violin", "field": "y", "groupby": ["v"]}), "violin_x", "violin_y"),
        (serde_json::json!({"type": "smooth", "x": "x", "y": "y", "method": "lm", "bandwidth": 0.75, "degree": 1, "n": 5, "groupby": ["v"]}), "x", "y"),
    ];
    let rows = binned_stat_rows();
    for (mut statistic, x, y) in cases {
        let kind = statistic["type"].clone();
        statistic["name"] = serde_json::json!("stat");
        let spec: ChartSpec = serde_json::from_value(serde_json::json!({
            "data": {"kind": "named", "name": "default"},
            "mark": "point",
            "transforms": [statistics_marker(), statistic],
            "encoding": {},
            "layers": [{"mark": "point", "data_source": "stat", "encoding": {"x": {"field": x}, "y": {"field": y}, "color": unit_binned_v()}}],
        }))
        .expect("the statistic spec deserializes");
        let p = prep(&spec, &rows);
        assert_eq!(distinct_f64(&p.transform_outputs["stat"], "v"), [1.0, 2.0, 3.0, 4.0], "{kind}: one group per bin");
    }
}

/// A histogram colored by a binned field draws one histogram per bin, each
/// counting its bin's rows (8 each, 32 in all). Rejects one histogram per
/// raw value.
#[test]
fn a_histogram_under_a_binned_color_draws_one_histogram_per_bin() {
    let spec: ChartSpec = serde_json::from_value(serde_json::json!({
        "data": {"kind": "named", "name": "default"},
        "mark": "bar",
        "transforms": [statistics_marker(), {"type": "bin", "field": "x", "bin_count": 4, "nice": false, "groupby": ["v"]}],
        "encoding": {"x": {"field": "bin_start"}, "x2": {"field": "bin_end"}, "y": {"field": "count"}, "color": unit_binned_v()},
    }))
    .expect("the histogram deserializes");
    let p = prep(&spec, &binned_stat_rows());
    let out = &p.transform_outputs[FINAL_OUTPUT_KEY];
    assert_eq!(distinct_f64(out, "v"), [1.0, 2.0, 3.0, 4.0]);
    for bin in [1.0, 2.0, 3.0, 4.0] {
        let count: f64 = float_col(out, "v").iter().zip(crate::render::arrow_cast::col_as_f64(out, "count").unwrap()).filter(|(v, _)| **v == bin).map(|(_, c)| c.unwrap_or(0.0)).sum();
        assert_eq!(count, 8.0, "bin {bin} counts its 8 rows");
    }
}

/// A binned color's edges are cut from the rows its statistic consumes (the
/// 32 chart rows), not from the statistic's output (16 grouped values):
/// with derived bins the two give different edge counts. Rejects edges cut
/// after the statistic ran.
#[test]
fn a_binned_colors_edges_come_from_the_rows_its_statistic_consumes() {
    let bin = serde_json::json!({"nice": false});
    let mut spec = binned_mean_bar_spec(None);
    spec.encoding.color = Some(serde_json::from_value(serde_json::json!({"field": "v", "scale": {"type": "bin-ordinal", "bin": bin}})).unwrap());
    let rows = binned_stat_rows();
    let raw = float_col(&rows, "v");
    let mut grouped = raw.clone();
    grouped.dedup();
    let edges = |values: &[f64]| {
        crate::transform::bin::scale_bin_edges(&serde_json::from_value(bin.clone()).unwrap(), values).unwrap().unwrap()
    };
    assert_ne!(edges(&raw), edges(&grouped), "the fixture discriminates");
    let p = prep(&spec, &rows);
    let Some(ColorScale::Discretizing(buckets)) = &p.provisional_scales.color else { panic!("binned") };
    assert_eq!(buckets.bounds(), edges(&raw).as_slice());
    let out = &p.transform_outputs[FINAL_OUTPUT_KEY];
    let starts = &edges(&raw)[..edges(&raw).len() - 1];
    assert!(float_col(out, "v").iter().all(|v| starts.contains(v)), "the statistic grouped by those edges");
}

/// Facet panels share one binned color's edges and group by them: panel a
/// holds `v` over 1..3 and panel b over 3..9, so two bins over all rows cut
/// at 5, and panel a's statistic makes one group. Rejects each panel cutting
/// its own edges (panel a's would cut at 2).
#[test]
fn faceted_statistics_group_by_one_binned_edge_set() {
    let rows = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("f", DataType::Utf8, false),
            Field::new("y", DataType::Float64, false),
            Field::new("v", DataType::Float64, false),
        ])),
        vec![
            Arc::new(StringArray::from(vec!["a", "a", "a", "b", "b", "b"])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 3.0, 6.0, 9.0])),
        ],
    )
    .unwrap();
    let spec: ChartSpec = serde_json::from_value(serde_json::json!({
        "data": {"kind": "named", "name": "default"},
        "mark": "bar",
        "facet": {"field": "f", "mode": {"kind": "wrap", "ncols": 2}},
        "transforms": [statistics_marker(), {"type": "aggregate", "ops": [{"field": "y", "fn": "mean", "as": "mean_y"}], "groupby": ["v"]}],
        "encoding": {"x": {"field": "v"}, "y": {"field": "mean_y"}, "color": {"field": "v", "scale": {"type": "bin-ordinal", "bin": {"bin_count": 2, "nice": false}}}},
    }))
    .expect("the faceted spec deserializes");
    let p = prep(&spec, &rows);
    let Some(ColorScale::Discretizing(buckets)) = &p.provisional_scales.color else { panic!("binned") };
    assert_eq!(buckets.bounds(), [1.0, 5.0, 9.0]);
    let out = &p.transform_outputs[FINAL_OUTPUT_KEY];
    let panels = crate::render::arrow_cast::col_as_ordinal_category_str(out, "f").unwrap();
    let groups: Vec<(String, f64, f64)> = panels
        .into_iter()
        .zip(float_col(out, "v"))
        .zip(float_col(out, "mean_y"))
        .map(|((f, v), m)| (f.unwrap_or_default(), v, m))
        .collect();
    assert_eq!(groups, [("a".to_string(), 1.0, 2.0), ("b".to_string(), 1.0, 4.0), ("b".to_string(), 5.0, 5.5)]);
}

/// A text's grouping-only `group` key over a binned color dodges the label
/// by the color's bins, so each label sits over its own bar: one label per
/// bar, each bar holding exactly one. Rejects a group key read by raw value,
/// which slotted the labels in row order (here high bin first) while the
/// bars take bin order.
#[test]
fn a_texts_binned_group_key_dodges_its_label_onto_its_bar() {
    let mean = |name: &str| serde_json::json!({"type": "aggregate", "ops": [{"field": "y", "fn": "mean", "as": "mean_y"}], "groupby": ["g", "v"], "name": name});
    let spec: ChartSpec = serde_json::from_value(serde_json::json!({
        "data": {"kind": "named", "name": "default"},
        "mark": "bar",
        "transforms": [statistics_marker(), mean("bars"), mean("labels")],
        "encoding": {},
        "layers": [
            {"mark": "bar", "data_source": "bars", "position": {"type": "dodge"},
             "encoding": {"x": {"field": "g"}, "y": {"field": "mean_y"}, "color": unit_binned_v()}},
            {"mark": "text", "data_source": "labels", "position": {"type": "dodge"},
             "encoding": {"x": {"field": "g"}, "y": {"field": "mean_y"}, "text": {"field": "mean_y"}, "group": unit_binned_v()}},
        ],
    }))
    .expect("the labelled dodge deserializes");
    let rows = binned_stat_rows();
    let high_bin_first = arrow::array::UInt32Array::from((0..rows.num_rows() as u32).rev().collect::<Vec<_>>());
    let rows = arrow::compute::take_record_batch(&rows, &high_bin_first).unwrap();
    let scene = gate_render(&spec, &rows).0.expect("renders");
    let bars = scene_bars(&scene);
    let labels: Vec<(f64, f64)> = scene
        .panels
        .iter()
        .flat_map(|panel| panel.marks.iter().flat_map(|batch| batch.nodes.iter()))
        .filter_map(|node| match node {
            ferrum_scene::SceneNode::Text { x, y, .. } => Some((*x, *y)),
            _ => None,
        })
        .collect();
    assert_eq!((bars.len(), labels.len()), (8, 8), "one bar and one label per (g, bin)");
    for bar in &bars {
        let held: Vec<&(f64, f64)> = labels.iter().filter(|(x, _)| bar.0 < *x && *x < bar.0 + bar.2).collect();
        assert_eq!(held.len(), 1, "bar at {} holds one label: {labels:?}", bar.0);
        assert!((held[0].1 - bar.1).abs() < 1e-6, "the label over a bar sits at its top: {bar:?} {labels:?}");
    }
}

/// A binned color over a non-numeric column keeps its typed refusal when a
/// statistic groups by it.
#[test]
fn a_statistic_grouped_by_a_non_numeric_binned_color_is_refused() {
    let rows = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, false),
            Field::new("y", DataType::Float64, false),
            Field::new("v", DataType::Utf8, false),
        ])),
        vec![
            Arc::new(StringArray::from(vec!["a", "b"])),
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(StringArray::from(vec!["p", "q"])),
        ],
    )
    .unwrap();
    let err = gate_render(&binned_mean_bar_spec(None), &rows).0.expect_err("refused");
    assert_eq!(err.to_string(), "Color(bin=...) bins numbers, but column 'v' is Utf8. Bind a numeric column, or drop bin=");
}

/// A spec without a binned color, a binned color no statistic groups by, and
/// a binned color over an all-null column each run the pipeline once, as
/// written: no statistic regroups and no pre-statistic rows are published.
#[test]
fn a_chart_no_statistic_groups_by_a_binned_color_runs_its_pipeline_as_written() {
    let theme = crate::layout::ThemeInputs::default();
    let viewport = crate::layout::Viewport { width: 600.0, height: 400.0 };
    let rows = binned_stat_rows();
    let mut unbinned = binned_mean_bar_spec(None);
    unbinned.encoding.color = Some(serde_json::from_value(serde_json::json!({"field": "v"})).unwrap());
    let ungrouped: ChartSpec = serde_json::from_value(serde_json::json!({
        "data": {"kind": "named", "name": "default"},
        "mark": "point",
        "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "color": unit_binned_v()},
    }))
    .unwrap();
    for spec in [unbinned, ungrouped] {
        let (outputs, _, _, origins) = run_pipeline(&spec, &rows, viewport, &theme, None).expect("runs");
        let (as_written, _, _, _) = run_transforms_with_facet(&spec, &rows, viewport, &theme, None).expect("runs");
        assert!(origins.is_empty());
        assert_eq!(outputs, as_written);
    }
    let nulls = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, false),
            Field::new("y", DataType::Float64, false),
            Field::new("v", DataType::Float64, true),
        ])),
        vec![
            Arc::new(StringArray::from(vec!["a", "b"])),
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(Float64Array::from(vec![None::<f64>, None])),
        ],
    )
    .unwrap();
    let spec = binned_mean_bar_spec(None);
    let (outputs, _, _, _) = run_pipeline(&spec, &nulls, viewport, &theme, None).expect("runs");
    let (as_written, _, _, _) = run_transforms_with_facet(&spec, &nulls, viewport, &theme, None).expect("runs");
    assert_eq!(outputs[FINAL_OUTPUT_KEY], as_written[FINAL_OUTPUT_KEY], "an all-null binned color groups as written");
}

/// The mean of `y` by `[g, v]`, named `name` (unnamed when `None`).
fn mean_by_g_v(name: Option<&str>) -> serde_json::Value {
    let mut step = serde_json::json!({"type": "aggregate", "ops": [{"field": "y", "fn": "mean", "as": "mean_y"}], "groupby": ["g", "v"]});
    if let Some(name) = name {
        step["name"] = serde_json::json!(name);
    }
    step
}

/// A flat mean bar under the binned `v` over `transforms`.
fn flat_binned_bar(transforms: serde_json::Value) -> ChartSpec {
    serde_json::from_value(serde_json::json!({
        "data": {"kind": "named", "name": "default"},
        "mark": "bar",
        "transforms": transforms,
        "encoding": {"x": {"field": "g"}, "y": {"field": "mean_y"}, "color": unit_binned_v()},
    }))
    .expect("the flat bar deserializes")
}

/// Only the steps after a chain's statistics marker group a binned color by
/// bin: the user's own aggregate before the marker keeps its 16 raw (g, v)
/// groups, the lowered one after it makes 8 (g, bin) groups, and a chain
/// with no marker is all user steps, so nothing bins. Rejects grouping a
/// user's step by bin, and treating an unmarked chain as lowered.
#[test]
fn only_steps_after_the_statistics_marker_group_by_bin() {
    let rows = binned_stat_rows();
    for (transforms, rows_out, label) in [
        (serde_json::json!([mean_by_g_v(None), statistics_marker()]), 16, "a user step before the marker"),
        (serde_json::json!([mean_by_g_v(None)]), 16, "a chain with no marker"),
        (serde_json::json!([statistics_marker(), mean_by_g_v(None)]), 8, "a lowered step after the marker"),
    ] {
        let p = prep(&flat_binned_bar(transforms), &rows);
        assert_eq!(p.transform_outputs[FINAL_OUTPUT_KEY].num_rows(), rows_out, "{label}");
    }
}

/// A sibling whose statistic groups by `v` for its own `detail` keeps its
/// standalone grain beside a bar binning `v`, whether its statistic is a
/// named fork (`+` over one frame, `Chart.layer`) or a `Branch` (`+` over
/// its own rows): 16 points, as alone. The bar beside it draws one bar per
/// (g, bin). Rejects grouping a sibling's statistic by bin because it
/// groups by a column another layer bins.
#[test]
fn a_siblings_statistic_keeps_its_own_grain_beside_a_binned_color() {
    let rows = binned_stat_rows();
    let point_layer = |source: &str| serde_json::json!({"mark": "point", "data_source": source,
        "encoding": {"x": {"field": "g"}, "y": {"field": "mean_y"}, "detail": {"field": "v"}}});
    let bar_layer = serde_json::json!({"mark": "bar", "data_source": "bars",
        "encoding": {"x": {"field": "g"}, "y": {"field": "mean_y"}, "color": unit_binned_v()}});
    let alone: ChartSpec = serde_json::from_value(serde_json::json!({
        "data": {"kind": "named", "name": "default"},
        "mark": "point",
        "transforms": [statistics_marker(), mean_by_g_v(None)],
        "encoding": {"x": {"field": "g"}, "y": {"field": "mean_y"}, "detail": {"field": "v"}},
    }))
    .unwrap();
    let points = |p: &PreparedInputs, key: &str| p.transform_outputs[key].num_rows();
    assert_eq!(points(&prep(&alone, &rows), FINAL_OUTPUT_KEY), 16, "the point alone");
    let forked: ChartSpec = serde_json::from_value(serde_json::json!({
        "data": {"kind": "named", "name": "default"},
        "mark": "bar",
        "transforms": [statistics_marker(), mean_by_g_v(Some("bars")), mean_by_g_v(Some("points"))],
        "encoding": {},
        "layers": [bar_layer.clone(), point_layer("points")],
    }))
    .unwrap();
    let p = prep(&forked, &rows);
    assert_eq!((points(&p, "bars"), points(&p, "points")), (8, 16), "named forks");
    let branched: ChartSpec = serde_json::from_value(serde_json::json!({
        "data": {"kind": "named", "name": "default"},
        "mark": "bar",
        "transforms": [statistics_marker(), mean_by_g_v(Some("bars")),
            {"type": "branch", "name": "points", "transforms": [statistics_marker(), mean_by_g_v(None)]}],
        "encoding": {},
        "layers": [bar_layer, point_layer("points")],
    }))
    .unwrap();
    let p = prep(&branched, &rows);
    assert_eq!((points(&p, "bars"), points(&p, "points")), (8, 16), "a branch statistic");
}

/// The derived edges `bin={"nice": false}` cuts over `values`.
fn sturges_edges(values: &[f64]) -> Vec<f64> {
    crate::transform::bin::scale_bin_edges(&serde_json::from_value(serde_json::json!({"nice": false})).unwrap(), values)
        .unwrap()
        .expect("values to bin")
}

/// The binned `v` with derived (Sturges) edges.
fn sturges_binned_v() -> serde_json::Value {
    serde_json::json!({"field": "v", "scale": {"type": "bin-ordinal", "bin": {"nice": false}}})
}

/// The bounds of `p`'s binned color scale.
fn binned_bounds(p: &PreparedInputs) -> Vec<f64> {
    match &p.provisional_scales.color {
        Some(ColorScale::Discretizing(buckets)) => buckets.bounds().to_vec(),
        other => panic!("binned color scale, got {other:?}"),
    }
}

/// A binned color's derived edges are the `Bin` transform's over the data,
/// alone and with a second layer reading the same rows: the rows count once
/// however many layers read them. Rejects pooling the rows once per reading
/// layer, which raised the derived bin count.
#[test]
fn a_binned_colors_edges_equal_the_bins_over_the_data_with_a_second_reading_layer() {
    let rows = binned_stat_rows();
    let expected = sturges_edges(&float_col(&rows, "v"));
    let mut alone = flat_binned_bar(serde_json::json!([statistics_marker(), mean_by_g_v(None)]));
    alone.encoding.color = Some(serde_json::from_value(sturges_binned_v()).unwrap());
    assert_eq!(binned_bounds(&prep(&alone, &rows)), expected, "alone");
    let layered: ChartSpec = serde_json::from_value(serde_json::json!({
        "data": {"kind": "named", "name": "default"},
        "mark": "bar",
        "transforms": [statistics_marker(), mean_by_g_v(Some("bars"))],
        "encoding": {},
        "layers": [
            {"mark": "bar", "data_source": "bars", "encoding": {"x": {"field": "g"}, "y": {"field": "mean_y"}, "color": sturges_binned_v()}},
            {"mark": "point", "encoding": {"x": {"field": "g"}, "y": {"field": "y"}, "color": sturges_binned_v()}},
        ],
    }))
    .unwrap();
    let p = prep(&layered, &rows);
    assert_eq!(binned_bounds(&p), expected, "with a point over the same rows");
    assert_eq!(p.transform_outputs["bars"].num_rows(), prep(&alone, &rows).transform_outputs[FINAL_OUTPUT_KEY].num_rows(), "the bar's grain is unchanged");
}

/// A boxplot's six sub-layers over one input cut the same edges as a
/// histogram of the same rows, both the `Bin` transform's over the data.
/// Rejects counting the rows once per sub-layer.
#[test]
fn a_binned_boxplot_cuts_the_same_edges_as_a_histogram() {
    let rows = binned_stat_rows();
    let expected = sturges_edges(&float_col(&rows, "v"));
    let colored = |encoding: serde_json::Value| {
        let mut encoding = encoding;
        encoding["color"] = sturges_binned_v();
        encoding
    };
    let boxplot: ChartSpec = serde_json::from_value(serde_json::json!({
        "data": {"kind": "named", "name": "default"},
        "mark": "point",
        "transforms": [statistics_marker(),
            {"type": "box_stats", "field": "y", "groupby": ["g", "v"], "name": "box"},
            {"type": "outliers", "field": "y", "groupby": ["g", "v"], "extent": 1.5, "name": "outliers"}],
        "encoding": {},
        "layers": [
            {"mark": "rule", "data_source": "box", "encoding": colored(serde_json::json!({"x": {"field": "g"}, "y": {"field": "lower_whisker"}, "y2": {"field": "upper_whisker"}}))},
            {"mark": "tick", "data_source": "box", "encoding": colored(serde_json::json!({"x": {"field": "g"}, "y": {"field": "lower_whisker"}}))},
            {"mark": "tick", "data_source": "box", "encoding": colored(serde_json::json!({"x": {"field": "g"}, "y": {"field": "upper_whisker"}}))},
            {"mark": "rect", "data_source": "box", "encoding": colored(serde_json::json!({"x": {"field": "g"}, "y": {"field": "q1"}, "y2": {"field": "q3"}}))},
            {"mark": "tick", "data_source": "box", "encoding": colored(serde_json::json!({"x": {"field": "g"}, "y": {"field": "median"}}))},
            {"mark": "point", "data_source": "outliers", "encoding": colored(serde_json::json!({"x": {"field": "g"}, "y": {"field": "y"}}))},
        ],
    }))
    .unwrap();
    let histogram: ChartSpec = serde_json::from_value(serde_json::json!({
        "data": {"kind": "named", "name": "default"},
        "mark": "bar",
        "transforms": [statistics_marker(), {"type": "bin", "field": "x", "bin_count": 4, "nice": false, "groupby": ["v"]}],
        "encoding": {"x": {"field": "bin_start"}, "x2": {"field": "bin_end"}, "y": {"field": "count"}, "color": sturges_binned_v()},
    }))
    .unwrap();
    assert_eq!(binned_bounds(&prep(&histogram, &rows)), expected, "histogram");
    assert_eq!(binned_bounds(&prep(&boxplot, &rows)), expected, "boxplot");
}

/// `binned_stat_rows` with three more rows for group `a`: `v` NaN, +inf and
/// -inf, `y` 100.
fn binned_rows_with_non_finite() -> RecordBatch {
    let rows = binned_stat_rows();
    let extra = RecordBatch::try_new(
        rows.schema(),
        vec![
            Arc::new(StringArray::from(vec!["a", "a", "a"])),
            Arc::new(Float64Array::from(vec![90.0, 91.0, 92.0])),
            Arc::new(Float64Array::from(vec![100.0, 100.0, 100.0])),
            Arc::new(Float64Array::from(vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY])),
        ],
    )
    .unwrap();
    arrow::compute::concat_batches(&rows.schema(), &[rows, extra]).unwrap()
}

/// A NaN or infinite binned value keeps its row as its own bar: the mean bar
/// draws the 8 (g, bin) bars and one bar each for NaN, +inf and -inf, as
/// grouping by the raw value drew them. Rejects dropping those rows.
#[test]
fn non_finite_binned_values_keep_their_own_bars() {
    let out = &prep(&flat_binned_bar(serde_json::json!([statistics_marker(), mean_by_g_v(None)])), &binned_rows_with_non_finite())
        .transform_outputs[FINAL_OUTPUT_KEY]
        .clone();
    assert_eq!(out.num_rows(), 11, "8 binned bars and 3 non-finite ones");
    let v = float_col(out, "v");
    assert!(v.iter().any(|v| v.is_nan()) && v.contains(&f64::INFINITY) && v.contains(&f64::NEG_INFINITY), "{v:?}");
    let means: Vec<f64> = v.iter().zip(float_col(out, "mean_y")).filter(|(v, _)| !v.is_finite()).map(|(_, m)| m).collect();
    assert_eq!(means, [100.0, 100.0, 100.0]);
}

/// A statistic inside a `Branch` (a `+` operand's) groups by the edges its
/// legend draws: the edges are cut from the rows the branch's statistic
/// read, so every bar's bin start is one of the legend's edges and the
/// legend's edges are the `Bin` transform's over those rows. Rejects
/// sampling the branch's output, whose bin starts gave a legend other than
/// the edges the bars grouped by.
#[test]
fn a_branch_statistics_legend_matches_its_bars() {
    let rows = binned_stat_rows();
    let spec: ChartSpec = serde_json::from_value(serde_json::json!({
        "data": {"kind": "named", "name": "default"},
        "mark": "bar",
        "transforms": [{"type": "branch", "name": "operand", "transforms": [statistics_marker(), mean_by_g_v(None)]}],
        "encoding": {},
        "layers": [{"mark": "bar", "data_source": "operand",
            "encoding": {"x": {"field": "g"}, "y": {"field": "mean_y"}, "color": sturges_binned_v()}}],
    }))
    .unwrap();
    let p = prep(&spec, &rows);
    let expected = sturges_edges(&float_col(&rows, "v"));
    assert_eq!(binned_bounds(&p), expected, "the legend's edges");
    let starts = &expected[..expected.len() - 1];
    let bars = float_col(&p.transform_outputs["operand"], "v");
    assert!(bars.iter().all(|v| starts.contains(v)), "every bar sits in a legend bin: {bars:?} {starts:?}");
}

/// A layer reading a branch whose input is a statistic's branch (a step
/// written on a composition after a binned operand) groups that statistic
/// by the legend's bins, and the legend's edges are cut from the rows the
/// statistic read, recorded for the reading branch too. Rejects claiming
/// only the branch the layer reads, which left the statistic grouped by raw
/// value.
#[test]
fn a_layer_reading_a_statistic_through_a_branch_groups_it_by_bin() {
    let rows = binned_stat_rows();
    let spec: ChartSpec = serde_json::from_value(serde_json::json!({
        "data": {"kind": "named", "name": "default"},
        "mark": "bar",
        "transforms": [
            {"type": "branch", "name": "operand", "transforms": [statistics_marker(), mean_by_g_v(None)]},
            {"type": "branch", "name": "post", "input": "operand", "transforms": [{"type": "filter", "predicate": "true"}]},
        ],
        "encoding": {},
        "layers": [{"mark": "bar", "data_source": "post",
            "encoding": {"x": {"field": "g"}, "y": {"field": "mean_y"}, "color": sturges_binned_v()}}],
    }))
    .unwrap();
    let p = prep(&spec, &rows);
    let expected = sturges_edges(&float_col(&rows, "v"));
    assert_eq!(binned_bounds(&p), expected, "the legend's edges");
    let starts = &expected[..expected.len() - 1];
    for key in ["operand", "post"] {
        let bars = float_col(&p.transform_outputs[key], "v");
        assert!(bars.iter().all(|v| starts.contains(v)), "{key}: every bar sits in a legend bin: {bars:?} {starts:?}");
    }
}

/// The origins a run publishes are the same on the run that groups by the
/// edges as on the run that cuts them, so the final scale's edges are the
/// edges the statistics grouped by; and with no demanded step nothing is
/// published.
#[test]
fn origins_are_the_same_on_both_runs() {
    let theme = crate::layout::ThemeInputs::default();
    let viewport = crate::layout::Viewport { width: 600.0, height: 400.0 };
    let rows = binned_stat_rows();
    let mut spec = flat_binned_bar(serde_json::json!([statistics_marker(), mean_by_g_v(None)]));
    spec.encoding.color = Some(serde_json::from_value(sturges_binned_v()).unwrap());
    let demand = binned_color_demand(&spec).expect("the lowered aggregate is demanded");
    let mut plan = crate::transform::color_bin_group::ColorBinPlan { demand, edges: None };
    let (first, _, _, first_origins) = run_transforms_with_facet(&spec, &rows, viewport, &theme, Some(&plan)).unwrap();
    let edges = binned_color_edges(&spec, &first, &first_origins, &theme, None).expect("edges");
    plan.edges = Some(std::rc::Rc::from(edges.clone()));
    let (second, _, _, second_origins) = run_transforms_with_facet(&spec, &rows, viewport, &theme, Some(&plan)).unwrap();
    assert!(!first_origins.is_empty(), "the demanded step records its origin");
    assert_eq!(first_origins, second_origins);
    assert_eq!(binned_bounds(&prep(&spec, &rows)), edges, "the final scale cuts the grouping edges");
    let starts = &edges[..edges.len() - 1];
    assert!(float_col(&second[FINAL_OUTPUT_KEY], "v").iter().all(|v| starts.contains(v)));
}

/// A layer reading a statistic that groups by the binned `z` adds the `z`
/// that statistic consumed (the chart rows, 5 and 6), while a layer binning
/// `m`, a column its statistic computes, adds the `m` it draws (0 and 2),
/// never the chart rows' same-named `m` (40 and 50).
#[test]
fn a_binned_statistic_adds_the_rows_it_consumed_and_a_computed_column_adds_what_it_draws() {
    let rows = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, false),
            Field::new("y", DataType::Float64, false),
            Field::new("z", DataType::Float64, false),
            Field::new("m", DataType::Float64, false),
        ])),
        vec![
            Arc::new(StringArray::from(vec!["a", "b"])),
            Arc::new(Float64Array::from(vec![0.0, 2.0])),
            Arc::new(Float64Array::from(vec![5.0, 6.0])),
            Arc::new(Float64Array::from(vec![40.0, 50.0])),
        ],
    )
    .unwrap();
    let binned = |field: &str| serde_json::json!({"field": field, "scale": {"type": "bin-ordinal", "bin": {"bin_count": 2, "nice": false}}});
    let layered = |field: &str, groupby: serde_json::Value| -> ChartSpec {
        serde_json::from_value(serde_json::json!({
            "data": {"kind": "named", "name": "default"},
            "mark": "point",
            "transforms": [statistics_marker(), {"type": "aggregate", "ops": [{"field": "y", "fn": "mean", "as": "m"}], "groupby": groupby, "name": "s"}],
            "encoding": {},
            "layers": [{"mark": "point", "data_source": "s", "encoding": {"x": {"field": "g"}, "y": {"field": "m"}, "color": binned(field)}}],
        }))
        .unwrap()
    };
    assert_eq!(binned_bounds(&prep(&layered("z", serde_json::json!(["g", "z"])), &rows)), [5.0, 5.5, 6.0], "z from the rows the statistic consumed");
    let computed = prep(&layered("m", serde_json::json!(["g"])), &rows);
    assert_eq!(binned_bounds(&computed), [0.0, 1.0, 2.0], "m as the statistic computes it");
}

/// A binned-color layer whose `data_source` names a key no step publishes
/// is refused with the unknown-`data_source` error, not drawn unbinned; so is
/// one naming the spelling under which a binned sibling's consumed rows were
/// once published among the outputs, which are no output a layer reads.
#[test]
fn a_binned_layer_reading_an_unpublished_key_is_refused() {
    let rows = binned_stat_rows();
    for source in ["nope", "__binned_origin__\u{1f}v\u{1f}bars"] {
        let spec: ChartSpec = serde_json::from_value(serde_json::json!({
            "data": {"kind": "named", "name": "default"},
            "mark": "bar",
            "transforms": [statistics_marker(), mean_by_g_v(Some("bars"))],
            "encoding": {},
            "layers": [
                {"mark": "bar", "data_source": "bars", "encoding": {"x": {"field": "g"}, "y": {"field": "mean_y"}, "color": unit_binned_v()}},
                {"mark": "point", "data_source": source, "encoding": {"x": {"field": "g"}, "y": {"field": "mean_y"}, "color": unit_binned_v()}},
            ],
        }))
        .unwrap();
        let err = prepare_render_inputs(
            &spec,
            &rows,
            &crate::layout::ThemeInputs::default(),
            crate::layout::Viewport { width: 600.0, height: 400.0 },
            &crate::render::chart_config::ChartConfig::default(),
            None,
        )
        .expect_err("an unpublished data_source is refused");
        let text = err.to_string();
        assert!(text.contains(&format!("data_source '{source}' not found")), "{text}");
        assert!(text.contains("available keys: [__final__, bars]"), "the binned rows are no output: {text}");
    }
}

/// The rows a binned statistic consumed are recorded beside the outputs,
/// never among them, so no reader of the outputs (a positional domain's
/// fallback over every output holding a field, a `data_source`, a
/// `Reorder(from=)`) can take them: the outputs are exactly what the steps
/// published, and the consumed rows sit under the output they belong to.
#[test]
fn a_binned_statistics_consumed_rows_sit_beside_the_outputs() {
    let rows = binned_stat_rows();
    let spec = flat_binned_bar(serde_json::json!([
        {"type": "aggregate", "ops": [{"field": "y", "fn": "mean", "as": "w"}], "groupby": ["g"], "name": "means"},
        statistics_marker(),
        mean_by_g_v(None),
    ]));
    let p = prep(&spec, &rows);
    let mut outputs: Vec<&str> = p.transform_outputs.keys().map(String::as_str).collect();
    outputs.sort_unstable();
    assert_eq!(outputs, ["__final__", "means"]);
    let origins: Vec<&str> = p.transform_origins.keys().map(String::as_str).collect();
    assert_eq!(origins, ["__final__"], "the tail's lineage grouped v by bin");
    assert_eq!(p.transform_origins["__final__"].rows_of("v").expect("v was grouped").num_rows(), rows.num_rows());
}

/// Rows with a numeric `n`, a string `s` and a timestamp `t` beside a
/// categorical `g` and numeric `x`, `y`: two rows per `g`.
fn mixed_dtype_rows() -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, false),
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("n", DataType::Float64, false),
            Field::new("s", DataType::Utf8, false),
            Field::new("t", DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None), false),
        ])),
        vec![
            Arc::new(StringArray::from(vec!["a", "a", "b", "b"])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
            Arc::new(Float64Array::from(vec![4.0, 3.0, 2.0, 1.0])),
            Arc::new(Float64Array::from(vec![1.5, 2.5, 3.5, 4.5])),
            Arc::new(StringArray::from(vec!["lo", "hi", "lo", "hi"])),
            Arc::new(arrow::array::TimestampMillisecondArray::from(vec![0_i64, 1_000, 2_000, 3_000])),
        ],
    )
    .unwrap()
}

/// A binned color over `field`, written as `class` (no `bin_channel` when
/// `None`).
fn binned_over(field: &str, class: Option<&str>) -> serde_json::Value {
    let mut binding = serde_json::json!({"field": field, "scale": {"type": "bin-ordinal", "bin": {"bin_count": 2, "nice": false}}});
    if let Some(class) = class {
        binding["bin_channel"] = serde_json::json!(class);
    }
    binding
}

/// A two-layer point spec over the chart rows, each layer coloring by its
/// own binding.
fn two_point_layers(color0: serde_json::Value, color1: serde_json::Value) -> ChartSpec {
    let layer = |color: serde_json::Value| serde_json::json!({"mark": "point", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "color": color}});
    serde_json::from_value(serde_json::json!({
        "data": {"kind": "named", "name": "default"},
        "mark": "point",
        "encoding": {"x": {"field": "x"}, "y": {"field": "y"}},
        "layers": [layer(color0), layer(color1)],
    }))
    .unwrap()
}

/// The refusal text of a binned color written as `class` over a column of
/// `dtype`.
fn not_numeric(class: &str, field: &str, dtype: &str) -> String {
    format!("{class}(bin=...) bins numbers, but column '{field}' is {dtype}. Bind a numeric column, or drop bin=")
}

/// A later layer whose own binned color reads a string column is refused
/// naming the class it was written as, whatever establishes the scale:
/// an unbinned categorical layer 0 (`fill='s' + Fill('s', bin=True)`), a
/// binned numeric one (`Fill('n', bin=True) + Fill('s', bin=True)`) and a
/// continuous one. Rejects judging only the establishing binding, which let
/// the first draw silently and turned the others into a scale exclusion.
#[test]
fn a_later_layers_binned_color_over_strings_is_refused_naming_its_class() {
    let rows = mixed_dtype_rows();
    for (establishing, label) in [
        (serde_json::json!({"field": "s"}), "categorical"),
        (binned_over("n", Some("Fill")), "binned"),
        (serde_json::json!({"field": "n", "type": "quantitative"}), "continuous"),
    ] {
        let spec = two_point_layers(establishing, binned_over("s", Some("Fill")));
        let err = gate_render(&spec, &rows).0.expect_err(label);
        assert_eq!(err.to_string(), not_numeric("Fill", "s", "Utf8"), "{label}");
    }
}

/// A flat binned color over a column the `Bin` transform cannot read is
/// refused naming the class its wire carries (`Color` when it carries
/// none) and the Arrow dtype, whatever type the binding declares: a string
/// column and a timestamp one. Rejects the categorical scale a string
/// column drew, silently dropping the bins.
#[test]
fn a_flat_binned_color_over_a_non_numeric_column_is_refused_naming_its_class() {
    let rows = mixed_dtype_rows();
    for (field, class, declared, dtype) in [
        ("s", Some("Fill"), None, "Utf8"),
        ("s", Some("Stroke"), Some("ordinal"), "Utf8"),
        ("s", None, None, "Utf8"),
        ("t", Some("Fill"), None, "Timestamp(Millisecond, None)"),
    ] {
        let mut color = binned_over(field, class);
        if let Some(declared) = declared {
            color["type"] = serde_json::json!(declared);
        }
        let spec: ChartSpec = serde_json::from_value(serde_json::json!({
            "data": {"kind": "named", "name": "default"},
            "mark": "point",
            "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "color": color},
        }))
        .unwrap();
        let err = gate_render(&spec, &rows).0.expect_err(field);
        assert_eq!(err.to_string(), not_numeric(class.unwrap_or("Color"), field, dtype));
    }
}

/// A layer whose rows a filter emptied is still refused on its binned
/// column's dtype, which stands without rows. Rejects a check that runs
/// after the empty-rows skip.
#[test]
fn an_emptied_layers_binned_color_over_strings_is_still_refused() {
    let spec: ChartSpec = serde_json::from_value(serde_json::json!({
        "data": {"kind": "named", "name": "default"},
        "mark": "point",
        "transforms": [{"type": "filter", "predicate": "datum.x > 100", "name": "none"}],
        "encoding": {"x": {"field": "x"}, "y": {"field": "y"}},
        "layers": [
            {"mark": "point", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "color": binned_over("n", Some("Stroke"))}},
            {"mark": "point", "data_source": "none", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "color": binned_over("s", Some("Stroke"))}},
        ],
    }))
    .unwrap();
    let err = gate_render(&spec, &mixed_dtype_rows()).0.expect_err("refused");
    assert_eq!(err.to_string(), not_numeric("Stroke", "s", "Utf8"));
}

/// Negative control: a later layer whose own color is unbinned over a
/// string column, under a binned numeric layer 0, still renders, left out
/// of the binned scale with one `LayerScaleExcluded`; and a binned layer
/// reading a statistic's output (numeric bin starts) renders.
#[test]
fn an_unbinned_later_color_and_a_binned_statistic_still_render() {
    let rows = mixed_dtype_rows();
    let spec = two_point_layers(binned_over("n", Some("Fill")), serde_json::json!({"field": "s"}));
    let (scene, warnings) = gate_render(&spec, &rows);
    scene.expect("an unbinned later color renders");
    assert_eq!(scale_exclusions(&warnings), excluded(1, 0, "color"));
    let rows = binned_stat_rows();
    let spec: ChartSpec = serde_json::from_value(serde_json::json!({
        "data": {"kind": "named", "name": "default"},
        "mark": "bar",
        "transforms": [statistics_marker(), mean_by_g_v(Some("stat"))],
        "encoding": {},
        "layers": [{"mark": "bar", "data_source": "stat",
            "encoding": {"x": {"field": "g"}, "y": {"field": "mean_y"}, "color": unit_binned_v()}}],
    }))
    .unwrap();
    gate_render(&spec, &rows).0.expect("a binned statistic renders");
}

/// A layer's binding on the parent's field with no scale of its own,
/// under a parent `Fill(bin=...)` over a string column, inherits the
/// binned scale with its class and is refused naming `Fill`. Rejects
/// adopting the scale without the class, which named `Color`.
#[test]
fn an_inherited_binned_color_is_refused_in_the_parents_class() {
    let spec: ChartSpec = serde_json::from_value(serde_json::json!({
        "data": {"kind": "named", "name": "default"},
        "mark": "point",
        "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "color": binned_over("s", Some("Fill"))},
        "layers": [{"mark": "point", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "color": {"field": "s"}}}],
    }))
    .unwrap();
    let err = gate_render(&spec, &mixed_dtype_rows()).0.expect_err("refused");
    assert_eq!(err.to_string(), not_numeric("Fill", "s", "Utf8"));
}

/// A hand-built spec whose binding writes a binned color's class with no
/// binned color scale is refused at render, naming where it was written,
/// on the chart encoding and on a layer's own. Rejects ignoring the field.
#[test]
fn a_bin_channel_without_a_binned_color_is_refused_at_render() {
    let rows = mixed_dtype_rows();
    let spec = two_point_layers(serde_json::json!({"field": "n"}), serde_json::json!({"field": "n", "bin_channel": "Stroke"}));
    let err = gate_render(&spec, &rows).0.expect_err("a layer's class without a scale");
    assert!(matches!(err, RenderError::BinChannelWithoutBinnedColor { .. }), "{err:?}");
    assert_eq!(
        err.to_string(),
        "layers[1].color: bin_channel='Stroke' names the class of a binned color, but this binding has no binned color \
         scale (a 'bin-ordinal' scale with 'bin'); drop bin_channel"
    );
    let spec: ChartSpec = serde_json::from_value(serde_json::json!({
        "data": {"kind": "named", "name": "default"},
        "mark": "point",
        "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "color": {"field": "n", "scale": {"type": "linear"}, "bin_channel": "Fill"}},
    }))
    .unwrap();
    let err = gate_render(&spec, &rows).0.expect_err("a chart encoding's class beside an unbinned scale").to_string();
    assert!(err.starts_with("color: bin_channel='Fill'"), "{err}");
}

/// A chart-level binned color over `field`, written as `class`, drawn by
/// layers that each carry their own literal `fill`; the layers inherit it
/// into the color scale without drawing it.
fn literal_fill_layers(field: &str, class: Option<&str>) -> ChartSpec {
    let layer = |fill: &str| serde_json::json!({"mark": "point", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}}, "mark_style": {"fill": fill}});
    serde_json::from_value(serde_json::json!({
        "data": {"kind": "named", "name": "default"},
        "mark": "point",
        "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "color": binned_over(field, class)},
        "layers": [layer("black"), layer("red")],
    }))
    .unwrap()
}

/// Layers whose own literal fill clears the chart's binned color from what
/// they draw still read it into the color scale, so a string column there
/// is refused naming the class written, not as a wrong-channel type
/// mismatch. Rejects judging only the bindings a layer draws. A numeric
/// column renders with no warning.
#[test]
fn literal_fill_layers_under_a_binned_color_over_strings_are_refused_naming_its_class() {
    let rows = mixed_dtype_rows();
    for (class, voice) in [(Some("Fill"), "Fill"), (Some("Stroke"), "Stroke"), (None, "Color")] {
        let err = gate_render(&literal_fill_layers("s", class), &rows).0.expect_err(voice);
        assert_eq!(err.to_string(), not_numeric(voice, "s", "Utf8"));
    }
    let (scene, warnings) = gate_render(&literal_fill_layers("n", Some("Fill")), &rows);
    scene.expect("a numeric binned color renders");
    assert_eq!(warnings, Vec::new());
}

/// A two-layer spec whose later layer, over its own rows where `n` is a
/// string, carries a literal fill and inherits the chart's binned
/// `Fill('n')`, under a layer 0 whose `n` is numeric. `numeric_later`
/// gives the later layer numbers instead.
fn literal_fill_layer_over_other_rows(numeric_later: bool) -> ChartSpec {
    let n_later = if numeric_later { "datum.y" } else { "datum.s" };
    serde_json::from_value(serde_json::json!({
        "data": {"kind": "named", "name": "default"},
        "mark": "point",
        "transforms": [{"type": "branch", "name": "other", "transforms": [{"type": "calculate", "as_field": "n", "expr": n_later}]}],
        "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "color": binned_over("n", Some("Fill"))},
        "layers": [
            {"mark": "point", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "color": binned_over("n", Some("Fill"))}},
            {"mark": "point", "data_source": "other", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}},
                "inherited_positional": ["x", "y"], "mark_style": {"fill": "red"}},
        ],
    }))
    .unwrap()
}

/// A literal-fill layer over other rows that inherits a binned color over
/// a string column is refused naming the class written, not left out of
/// the scale with a `LayerScaleExcluded`. Rejects the silent degrade. The
/// same layer over numbers renders with no warning.
#[test]
fn a_literal_fill_layer_over_strings_under_a_binned_color_is_refused_not_excluded() {
    let rows = mixed_dtype_rows();
    let err = gate_render(&literal_fill_layer_over_other_rows(false), &rows).0.expect_err("refused");
    assert_eq!(err.to_string(), not_numeric("Fill", "n", "Utf8"));
    let (scene, warnings) = gate_render(&literal_fill_layer_over_other_rows(true), &rows);
    scene.expect("numbers render");
    assert_eq!(warnings, Vec::new());
}

/// A text layer inheriting a binned color over a string column, with no
/// statistic grouping by it, draws no color of its own and reads none into
/// the scale, so this check does not refuse it.
#[test]
fn a_text_inheriting_a_binned_color_over_strings_is_not_refused() {
    let spec: ChartSpec = serde_json::from_value(serde_json::json!({
        "data": {"kind": "named", "name": "default"},
        "mark": "point",
        "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "color": binned_over("s", Some("Fill"))},
        "layers": [
            {"mark": "point", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "color": binned_over("n", Some("Fill"))}},
            {"mark": "text", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "text": {"field": "g"}},
                "inherited_channels": ["color"]},
        ],
    }))
    .unwrap();
    gate_render(&spec, &mixed_dtype_rows()).0.expect("the text renders");
}

// ── a line-family mark under a binned color ─────────────────────────────────

/// Rows for a line-family mark under a binned color: `x` 0..7 ascending and
/// `z` alternating between the low bin and the high one of `bin_count` 2
/// over 1..9 (cut at 5), so the two bins interleave along `x` and one
/// polyline over every row zigzags between them. `d` pairs rows p, p, q, q;
/// `y_hi` is `y` + 1; `s` holds strings; `f` puts the first four rows in
/// panel m and the last four in n.
fn binned_line_rows(z: [f64; 8]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("y_hi", DataType::Float64, false),
            Field::new("z", DataType::Float64, false),
            Field::new("d", DataType::Utf8, false),
            Field::new("s", DataType::Utf8, false),
            Field::new("f", DataType::Utf8, false),
        ])),
        vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])),
            Arc::new(Float64Array::from(vec![2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0])),
            Arc::new(Float64Array::from(vec![3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0])),
            Arc::new(Float64Array::from(z.to_vec())),
            Arc::new(StringArray::from(vec!["p", "p", "q", "q", "p", "p", "q", "q"])),
            Arc::new(StringArray::from(vec!["a", "b", "a", "b", "a", "b", "a", "b"])),
            Arc::new(StringArray::from(vec!["m", "m", "m", "m", "n", "n", "n", "n"])),
        ],
    )
    .unwrap()
}

/// `z` interleaving the low bin (rows 0, 2, 4, 6) and the high bin.
const INTERLEAVED_Z: [f64; 8] = [1.0, 9.0, 2.0, 8.0, 1.5, 8.5, 1.2, 8.8];

/// A binned `z` color: two bins, edges cut from the data without nicing.
fn binned_z() -> serde_json::Value {
    serde_json::json!({"field": "z", "scale": {"type": "bin-ordinal", "bin": {"bin_count": 2, "nice": false}}})
}

/// A flat `mark` over `x`/`y` (and `y2` = `y_hi` for a ribbon) whose color is
/// `color`, with `extra` merged into the spec's top level.
fn line_family_spec(mark: &str, color: serde_json::Value, extra: serde_json::Value) -> ChartSpec {
    let mut encoding = serde_json::json!({"x": {"field": "x"}, "y": {"field": "y"}, "color": color});
    if mark == "ribbon" {
        encoding["y2"] = serde_json::json!({"field": "y_hi"});
    }
    let mut spec = serde_json::json!({"data": {"kind": "named", "name": "default"}, "mark": mark, "encoding": encoding});
    for (key, value) in extra.as_object().expect("extra is an object") {
        spec[key] = value.clone();
    }
    serde_json::from_value(spec).expect("the line-family spec deserializes")
}

/// Every open series a panel draws (polyline, or path under a width or a
/// non-linear interpolation): its stroke and vertices, in draw order.
fn open_series(panel: &ferrum_scene::Panel) -> Vec<(ferrum_scene::Color, Vec<(f64, f64)>)> {
    panel
        .marks
        .iter()
        .flat_map(|batch| batch.nodes.iter())
        .filter_map(|node| match node {
            ferrum_scene::SceneNode::Polyline { points, style, .. } => {
                Some((style.color.expect("a stroked series"), points.clone()))
            }
            ferrum_scene::SceneNode::Path { commands, style, closed: false, .. } => {
                let points = commands
                    .iter()
                    .filter_map(|cmd| match cmd {
                        ferrum_scene::PathCmd::MoveTo { x, y } | ferrum_scene::PathCmd::LineTo { x, y } => Some((*x, *y)),
                        _ => None,
                    })
                    .collect();
                Some((style.stroke.expect("a stroked series"), points))
            }
            _ => None,
        })
        .collect()
}

/// The binned scale's step for bin `bin`, as the scene paints it.
fn bin_step(p: &PreparedInputs, bin: usize) -> (u8, u8, u8) {
    let Some(ColorScale::Discretizing(buckets)) = &p.provisional_scales.color else {
        panic!("expected the binned scale, got {:?}", p.provisional_scales.color)
    };
    let step = crate::render::draw::to_scene_color(buckets.colors()[bin]);
    (step.r, step.g, step.b)
}

/// Asserts `series` is one series per bin, low bin first, each stroked with
/// its bin's step and running left to right, and that the series hold
/// `points_per_series` vertices each.
fn assert_one_series_per_bin(p: &PreparedInputs, series: &[(ferrum_scene::Color, Vec<(f64, f64)>)], points_per_series: usize) {
    assert_eq!(series.len(), 2, "one series per bin: {series:?}");
    for (bin, (stroke, points)) in series.iter().enumerate() {
        assert_eq!((stroke.r, stroke.g, stroke.b), bin_step(p, bin), "series {bin} takes bin {bin}'s step");
        assert_eq!(points.len(), points_per_series, "series {bin}: {points:?}");
        assert!(points.windows(2).all(|w| w[0].0 < w[1].0), "series {bin} runs along x: {points:?}");
    }
}

/// A line under a binned color draws one series per bin, low bin first,
/// each in its bin's step and ordered along x, under the bin legend, with
/// no warning. Rejects the one polyline that zigzagged across the bins
/// under a suppressed legend and an `UnsupportedColorScaleOnMark` warning.
#[test]
fn a_line_under_a_binned_color_draws_one_series_per_bin() {
    let spec = line_family_spec("line", binned_z(), serde_json::json!({}));
    let rows = binned_line_rows(INTERLEAVED_Z);
    let p = prep(&spec, &rows);
    assert_eq!(p.colorbar.as_ref().expect("the bin legend").tick_labels, vec!["1", "5", "9"]);
    let (scene, warnings) = gate_render(&spec, &rows);
    let scene = scene.expect("renders");
    assert_eq!(warnings, Vec::new());
    let series = open_series(&scene.panels[0]);
    assert_one_series_per_bin(&p, &series, 4);
    // The low bin (y 2) sits above the high bin (y 1): smaller pixel y.
    assert!(series[0].1[0].1 < series[1].1[0].1, "the first series drawn is the low bin's");
}

/// A sized line (a trail, its width following `y` along each series) and a
/// stepped line (a path) each still split by bin, in their bins' steps.
#[test]
fn a_sized_or_curved_line_under_a_binned_color_draws_one_series_per_bin() {
    let rows = binned_line_rows(INTERLEAVED_Z);
    let sized = || {
        let mut spec = line_family_spec("line", binned_z(), serde_json::json!({}));
        spec.encoding.size = Some(EncodingSpec { field: "y".into(), ..Default::default() });
        spec
    };
    let stepped = || line_family_spec("line", binned_z(), serde_json::json!({"mark_style": {"interpolate": "step"}}));
    for (label, spec) in [("sized", sized()), ("step", stepped())] {
        let p = prep(&spec, &rows);
        let (scene, warnings) = gate_render(&spec, &rows);
        let series = open_series(&scene.expect(label).panels[0]);
        assert_eq!(warnings, Vec::new(), "{label}");
        assert_eq!(series.len(), 2, "{label}: one series per bin");
        for (bin, (stroke, _)) in series.iter().enumerate() {
            assert_eq!((stroke.r, stroke.g, stroke.b), bin_step(&p, bin), "{label}: series {bin}");
        }
    }
}

/// A ribbon under a binned color draws one closed band per bin, low bin
/// first, each filled with its bin's step, under the bin legend and with no
/// warning. Rejects the one band over every row, filled with the constant
/// fill.
#[test]
fn a_ribbon_under_a_binned_color_draws_one_band_per_bin() {
    let spec = line_family_spec("ribbon", binned_z(), serde_json::json!({}));
    let rows = binned_line_rows(INTERLEAVED_Z);
    let p = prep(&spec, &rows);
    assert!(p.colorbar.is_some(), "the bin legend");
    let (scene, warnings) = gate_render(&spec, &rows);
    assert_eq!(warnings, Vec::new());
    let bands = area_bands(&scene.expect("renders"));
    assert_eq!(bands.len(), 2, "one band per bin");
    for (bin, (fill, points)) in bands.iter().enumerate() {
        assert_eq!((fill.r, fill.g, fill.b), bin_step(&p, bin), "band {bin} takes bin {bin}'s step");
        assert_eq!(points.len(), 8, "band {bin} rings its bin's four rows: {points:?}");
    }
}

/// A detail or a categorical stroke dash subdivides each bin's series: four
/// series over two bins and two `d` values, each in its bin's step and
/// along x, low bin first. Rejects splitting by the detail alone (two
/// series zigzagging across the bins) and by the bin alone.
#[test]
fn a_detail_or_stroke_dash_splits_a_binned_lines_series_by_key_and_bin() {
    let rows = binned_line_rows(INTERLEAVED_Z);
    let detail = line_family_spec("line", binned_z(), serde_json::json!({"mark_style": {"detail": "d"}}));
    let mut dashed = line_family_spec("line", binned_z(), serde_json::json!({}));
    dashed.encoding.stroke_dash = Some(EncodingSpec { field: "d".into(), ..Default::default() });
    for (label, spec) in [("detail", detail), ("stroke_dash", dashed)] {
        let p = prep(&spec, &rows);
        let series = open_series(&gate_render(&spec, &rows).0.expect(label).panels[0]);
        assert_eq!(series.len(), 4, "{label}: one series per (bin, d): {series:?}");
        for (i, (stroke, points)) in series.iter().enumerate() {
            assert_eq!((stroke.r, stroke.g, stroke.b), bin_step(&p, i / 2), "{label}: series {i}");
            assert_eq!(points.len(), 2, "{label}: series {i}");
            assert!(points[0].0 < points[1].0, "{label}: series {i} runs along x");
        }
    }
}

/// A quantize color on a line has no series split: the line draws one
/// polyline, its legend is suppressed and `UnsupportedColorScaleOnMark`
/// still warns. Only a binned color splits a line's series.
#[test]
fn a_quantize_color_on_a_line_stays_one_series_and_warns() {
    let quantize = serde_json::json!({"field": "z", "scale": {"type": "quantize", "range": ["#ff0000", "#0000ff"]}});
    let spec = line_family_spec("line", quantize, serde_json::json!({}));
    let rows = binned_line_rows(INTERLEAVED_Z);
    let (scene, warnings) = gate_render(&spec, &rows);
    assert_eq!(open_series(&scene.expect("renders").panels[0]).len(), 1);
    assert!(prep(&spec, &rows).colorbar.is_none(), "the legend is suppressed");
    assert!(
        warnings.iter().any(|w| matches!(w, RenderWarning::UnsupportedColorScaleOnMark { scale_kind, .. } if scale_kind == "discretizing")),
        "{warnings:?}"
    );
}

/// Faceted lines under a binned color cut one edge set from every panel's
/// rows (1..9, cut at 5): panel n's `z` (3..4.5) all falls in the low bin,
/// so it draws one low-bin series while panel m draws one per bin. Rejects
/// edges cut from each panel's own rows, under which n split at 3.75.
#[test]
fn faceted_lines_under_a_binned_color_share_one_edge_set() {
    let rows = binned_line_rows([1.0, 9.0, 2.0, 8.0, 3.0, 4.0, 3.5, 4.5]);
    let spec = line_family_spec(
        "line",
        binned_z(),
        serde_json::json!({"facet": {"field": "f", "mode": {"kind": "wrap", "ncols": 2}}}),
    );
    let p = prep(&spec, &rows);
    let scene = gate_render(&spec, &rows).0.expect("renders");
    let [m, n] = scene.panels.as_slice() else { panic!("two panels, got {}", scene.panels.len()) };
    let (m, n) = (open_series(m), open_series(n));
    assert_eq!(m.len(), 2, "panel m draws both bins: {m:?}");
    assert_eq!(n.len(), 1, "panel n draws only the low bin: {n:?}");
    assert_eq!((n[0].0.r, n[0].0.g, n[0].0.b), bin_step(&p, 0));
    assert_eq!(n[0].1.len(), 4);
}

/// A binned line drawn as a layer, and as an independent-y layer (the
/// SecondaryY route), splits by bin beside a point layer that draws the same
/// binned color. Rejects the single zigzag on the layered routes.
#[test]
fn a_binned_line_layer_splits_by_bin_on_the_layered_and_independent_y_routes() {
    let rows = binned_line_rows(INTERLEAVED_Z);
    for independent_y in [false, true] {
        let spec: ChartSpec = serde_json::from_value(serde_json::json!({
            "data": {"kind": "named", "name": "default"},
            "mark": "point",
            "encoding": {"x": {"field": "x"}, "y": {"field": "y"}},
            "layers": [
                {"mark": "point", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "color": binned_z()}},
                {"mark": "line", "encoding": {"x": {"field": "x"}, "y": {"field": "y_hi"}, "color": binned_z()},
                    "independent_y": independent_y},
            ],
        }))
        .unwrap();
        let p = prep(&spec, &rows);
        let (scene, warnings) = gate_render(&spec, &rows);
        assert_eq!(warnings, Vec::new(), "independent_y={independent_y}");
        let series = open_series(&scene.expect("renders").panels[0]);
        assert_one_series_per_bin(&p, &series, 4);
    }
}

/// A line whose binned color reads a string column keeps the typed refusal.
#[test]
fn a_line_under_a_binned_color_over_strings_is_refused() {
    let color = serde_json::json!({"field": "s", "scale": {"type": "bin-ordinal", "bin": {"bin_count": 2, "nice": false}}});
    let spec = line_family_spec("line", color, serde_json::json!({}));
    let err = gate_render(&spec, &binned_line_rows(INTERLEAVED_Z)).0.expect_err("refused");
    assert_eq!(err.to_string(), not_numeric("Color", "s", "Utf8"));
}

// ── a binned layer under a color scale its bin rule did not cut ─────────────

/// A chart of two layers over `binned_line_rows`, each a `(mark, color)`
/// pair drawing `x`/`y` (a ribbon draws to `y_hi`, an area its own band).
fn two_colored_layers(first: (&str, serde_json::Value), second: (&str, serde_json::Value)) -> ChartSpec {
    let layer = |(mark, color): (&str, serde_json::Value)| {
        serde_json::json!({"mark": mark, "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "color": color}})
    };
    serde_json::from_value(serde_json::json!({
        "data": {"kind": "named", "name": "default"},
        "mark": "point",
        "encoding": {"x": {"field": "x"}, "y": {"field": "y"}},
        "layers": [layer(first), layer(second)],
    }))
    .unwrap()
}

/// `z` under a binned color of `bin_count` bins, without nicing.
fn binned_z_of(bin_count: usize) -> serde_json::Value {
    serde_json::json!({"field": "z", "scale": {"type": "bin-ordinal", "bin": {"bin_count": bin_count, "nice": false}}})
}

/// `z` under a two-bucket quantize scale.
fn quantized_z() -> serde_json::Value {
    serde_json::json!({"field": "z", "scale": {"type": "quantize", "range": ["#ff0000", "#0000ff"]}})
}

/// The one `LayerScaleExcluded` of layer 1 off layer 0's color scale.
fn color_excluded_layer_one() -> Vec<RenderWarning> {
    vec![RenderWarning::LayerScaleExcluded {
        layer: 1,
        establishing: 0,
        channel: "color".to_string(),
        placement: crate::render::ExcludedLayerPlacement::SharedAxis,
    }]
}

fn exclusions(warnings: &[RenderWarning]) -> Vec<RenderWarning> {
    warnings.iter().filter(|w| matches!(w, RenderWarning::LayerScaleExcluded { .. })).cloned().collect()
}

fn inert_color(warnings: &[RenderWarning]) -> Vec<RenderWarning> {
    warnings.iter().filter(|w| matches!(w, RenderWarning::UnsupportedColorScaleOnMark { .. })).cloned().collect()
}

/// A plain `z` line beside a binned `z` line, and a quantize `z` line beside
/// one: the first layer establishes a scale no bin rule cut, so the binned
/// layer is excluded with one `LayerScaleExcluded`, both lines draw one
/// series, `z` is inert on them (`UnsupportedColorScaleOnMark`) and no
/// colorbar is drawn. Rejects the binned line judged as drawing the scale
/// (an orphan colorbar and no warning) and, under quantize, split by the
/// quantize buckets its statistics do not group by.
#[test]
fn a_binned_line_under_a_scale_no_bin_rule_cut_is_excluded_and_inert() {
    let rows = binned_line_rows(INTERLEAVED_Z);
    for (label, establishing, scale_kind) in [
        ("continuous", serde_json::json!({"field": "z"}), "continuous"),
        ("quantize", quantized_z(), "discretizing"),
    ] {
        let spec = two_colored_layers(("line", establishing), ("line", binned_z()));
        let (scene, warnings) = gate_render(&spec, &rows);
        assert_eq!(open_series(&scene.expect(label).panels[0]).len(), 2, "{label}: one series per line");
        assert_eq!(exclusions(&warnings), color_excluded_layer_one(), "{label}");
        let inert = vec![RenderWarning::UnsupportedColorScaleOnMark {
            marks: vec!["line".to_string()],
            scale_kind: scale_kind.to_string(),
            suppressed: true,
        }];
        assert_eq!(inert_color(&warnings), inert, "{label}");
        assert!(prep(&spec, &rows).colorbar.is_none(), "{label}: the colorbar is suppressed");
    }
}

/// A plain `z` point beside a binned `z` point: the binned point's bin rule
/// takes no effect under the continuous scale, so it is excluded with one
/// `LayerScaleExcluded` and draws through that scale. Rejects dropping the
/// bin rule silently.
#[test]
fn a_binned_point_under_a_continuous_color_is_excluded() {
    let rows = binned_line_rows(INTERLEAVED_Z);
    let spec = two_colored_layers(("point", serde_json::json!({"field": "z"})), ("point", binned_z()));
    let (scene, warnings) = gate_render(&spec, &rows);
    scene.expect("renders");
    assert_eq!(exclusions(&warnings), color_excluded_layer_one());
    assert_eq!(inert_color(&warnings), Vec::new());
    let p = prep(&spec, &rows);
    assert!(matches!(p.provisional_scales.color, Some(ColorScale::Continuous { .. })), "{:?}", p.provisional_scales.color);
}

/// A binned `z` line beside a plain `z` line: the binned layer establishes
/// the binned scale, the plain line has no bin rule, so it joins without a
/// warning and draws one series, while the binned line splits by bin.
#[test]
fn a_plain_line_beside_an_establishing_binned_line_joins_without_a_warning() {
    let rows = binned_line_rows(INTERLEAVED_Z);
    let spec = two_colored_layers(("line", binned_z()), ("line", serde_json::json!({"field": "z"})));
    let (scene, warnings) = gate_render(&spec, &rows);
    assert_eq!(warnings, Vec::new());
    assert_eq!(open_series(&scene.expect("renders").panels[0]).len(), 3, "two bins plus one plain series");
}

/// Two binned `z` lines with different bin counts: the first layer's rule
/// (three bins over 1..9; the middle one empty) cuts the scale, the second
/// is excluded with one `LayerScaleExcluded` and still draws through that
/// binned scale: each line splits into the low and high bins' series, in
/// those bins' steps. Rejects the second layer's five bins taking effect.
#[test]
fn a_binned_layer_with_another_bin_rule_is_excluded_and_draws_through_the_first() {
    let rows = binned_line_rows(INTERLEAVED_Z);
    let spec = two_colored_layers(("line", binned_z_of(3)), ("line", binned_z_of(5)));
    let p = prep(&spec, &rows);
    let Some(ColorScale::Discretizing(buckets)) = &p.provisional_scales.color else { panic!("binned") };
    assert_eq!(buckets.colors().len(), 3, "the first layer's three bins");
    let (scene, warnings) = gate_render(&spec, &rows);
    assert_eq!(exclusions(&warnings), color_excluded_layer_one());
    assert_eq!(inert_color(&warnings), Vec::new());
    let series = open_series(&scene.expect("renders").panels[0]);
    let steps: Vec<(u8, u8, u8)> = series.iter().map(|(stroke, _)| (stroke.r, stroke.g, stroke.b)).collect();
    let (low, high) = (bin_step(&p, 0), bin_step(&p, 2));
    assert_eq!(steps, vec![low, high, low, high], "each line splits by the first layer's bins");
}

/// A plain `z` point beside a binned `z` area: the area forces a categorical
/// scale over `z`, which no bin rule cut, so the area is excluded with one
/// `LayerScaleExcluded` and draws one band per raw value (two, 1 and 9), each
/// in its category's palette color, as before.
#[test]
fn a_binned_area_under_a_categorical_color_is_excluded() {
    let rows = binned_line_rows([1.0, 9.0, 1.0, 9.0, 1.0, 9.0, 1.0, 9.0]);
    let spec = two_colored_layers(("point", serde_json::json!({"field": "z"})), ("area", binned_z()));
    let (scene, warnings) = gate_render(&spec, &rows);
    assert_eq!(exclusions(&warnings), color_excluded_layer_one());
    let p = prep(&spec, &rows);
    let Some(categorical @ ColorScale::Categorical { .. }) = &p.provisional_scales.color else {
        panic!("expected the area's categorical scale, got {:?}", p.provisional_scales.color)
    };
    let fills: Vec<(u8, u8, u8)> = area_bands(&scene.expect("renders")).iter().map(|(fill, _)| (fill.r, fill.g, fill.b)).collect();
    let palette = |value: &str| {
        let color = crate::render::draw::to_scene_color(categorical.lookup(value).expect("a category"));
        (color.r, color.g, color.b)
    };
    assert_eq!(fills, vec![palette("1"), palette("9")], "one band per raw value");
}

/// A dodged bar binned on `z` under an establishing quantize or continuous
/// `z` point: no bin rule cut the scale, so the bars dodge by raw value
/// (four slots per band) and the bar layer is excluded with one
/// `LayerScaleExcluded`. Rejects dodging by the quantize buckets (two slots
/// per band).
#[test]
fn a_dodged_binned_bar_under_a_scale_no_bin_rule_cut_dodges_by_value() {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, false),
            Field::new("y", DataType::Float64, false),
            Field::new("z", DataType::Float64, false),
        ])),
        vec![
            Arc::new(StringArray::from(vec!["a", "a", "a", "a", "b", "b", "b", "b"])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])),
            Arc::new(Float64Array::from(vec![11.0, 1.0, 12.0, 2.0, 13.0, 3.0, 14.0, 4.0])),
        ],
    )
    .unwrap();
    for (label, establishing) in [("quantize", quantized_z()), ("continuous", serde_json::json!({"field": "z"}))] {
        let spec: ChartSpec = serde_json::from_value(serde_json::json!({
            "data": {"kind": "named", "name": "default"},
            "mark": "point",
            "encoding": {"x": {"field": "g"}, "y": {"field": "y"}},
            "layers": [
                {"mark": "point", "encoding": {"x": {"field": "g"}, "y": {"field": "y"}, "color": establishing}},
                {"mark": "bar", "position": {"type": "dodge"}, "encoding": {"x": {"field": "g"}, "y": {"field": "y"}, "color": binned_z()}},
            ],
        }))
        .unwrap();
        let (scene, warnings) = gate_render(&spec, &batch);
        let mut slots: Vec<f64> = scene
            .expect(label)
            .panels
            .iter()
            .flat_map(|panel| panel.marks.iter().flat_map(|batch| batch.nodes.iter()))
            .filter_map(|node| match node {
                ferrum_scene::SceneNode::Rect { x, .. } => Some(*x),
                _ => None,
            })
            .collect();
        assert_eq!(slots.len(), 8, "{label}: every bar draws");
        slots.sort_by(f64::total_cmp);
        slots.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
        assert_eq!(slots.len(), 8, "{label}: one slot per raw value: {slots:?}");
        assert_eq!(exclusions(&warnings), color_excluded_layer_one(), "{label}");
    }
}

/// A raster's cells establish a continuous color; a later point binned on
/// `z` is excluded with one `LayerScaleExcluded`. The cells still draw
/// through the continuous scale and the point draws every row.
#[test]
fn a_binned_point_beside_raster_cells_is_excluded() {
    let point = serde_json::json!([
        {"mark": "point", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "color": binned_z()}},
    ]);
    let spec = raster_spec(count_raster(false), Some(serde_json::json!({"field": "count"})), point, None);
    let (scene, warnings) = gate_render(&spec, &raster_rows());
    assert_eq!(exclusions(&warnings), color_excluded_layer_one());
    let scene = scene.expect("renders");
    let points = scene
        .panels
        .iter()
        .flat_map(|panel| panel.marks.iter().flat_map(|batch| batch.nodes.iter()))
        .filter(|node| matches!(node, ferrum_scene::SceneNode::Circle { .. }))
        .count();
    assert_eq!(points, raster_rows().num_rows(), "the point draws every row");
    let (p, panels) = raster_panel_cells(&spec);
    assert!(matches!(p.provisional_scales.color, Some(ColorScale::Continuous { .. })), "{:?}", p.provisional_scales.color);
    let [cells] = panels.as_slice() else { panic!("one image, got {}", panels.len()) };
    assert!(!drawn(cells).is_empty(), "the cells draw");
}
