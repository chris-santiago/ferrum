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
        &crate::layout::ThemeInputs::default(),
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
        &crate::layout::ThemeInputs::default(),
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
        &crate::layout::ThemeInputs::default(),
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
        &crate::layout::ThemeInputs::default(),
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

    let unfaceted = run_transforms_with_facet(&raster_screen_spec(), &batch, viewport, &theme)
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
        mode: crate::layout::FacetMode::Wrap { ncols: 1 },
        spacing: None,
        resolve: crate::layout::FacetResolve::default(),
    });
    let faceted = run_transforms_with_facet(&faceted_spec, &batch, viewport, &theme)
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
        &theme,
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
        &theme,
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
    let outputs = run_transforms_with_facet(&spec, &batch, viewport, &theme)
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

    let (outputs, warnings, _) = run_transforms_with_facet(&spec, &batch, viewport, &theme)
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

    let (outputs, warnings, _) = run_transforms_with_facet(&spec, &batch, viewport, &theme)
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

    let err = run_transforms_with_facet(&spec, &batch, viewport, &theme)
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

    let outputs = run_transforms_with_facet(&spec, &batch, viewport, &theme)
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

    let outputs = run_transforms_with_facet(&spec, &batch, viewport, &theme)
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
            transforms: vec![],
            mark_style: None,
            data_source: None,
            position: None,
            name: None,
            independent_y: false,
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
/// and sort members above — see the predicate's doc for why these three,
/// plus `ColorPaletteOverflowed`, are whole-chart facts at the
/// provisional/panel-real boundary despite looking data-dependent.
#[test]
fn provisional_scale_warning_superseded_by_panel_pass_includes_every_auxiliary_warning() {
    assert!(provisional_scale_warning_superseded_by_panel_pass(
        &RenderWarning::UnsupportedOpacityScale {
            channel: "fill_opacity".into(),
            scale_kind: "log".into(),
            fallback: crate::render::OpacityScaleFallback::Linear,
        }
    ));
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
/// exactly the three auxiliary variants, and — the asymmetry the whole
/// mechanism exists to preserve — excludes `ColorPaletteOverflowed`,
/// whose own per-panel copies must survive.
#[test]
fn panel_pass_warning_restated_by_every_panel_excludes_color() {
    assert!(panel_pass_warning_restated_by_every_panel(
        &RenderWarning::UnsupportedOpacityScale {
            channel: "fill_opacity".into(),
            scale_kind: "log".into(),
            fallback: crate::render::OpacityScaleFallback::Linear,
        }
    ));
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
    let opacity = RenderWarning::UnsupportedOpacityScale {
        channel: "fill_opacity".into(),
        scale_kind: "log".into(),
        fallback: crate::render::OpacityScaleFallback::Linear,
    };
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
