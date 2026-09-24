//! Positional scale resolution: build x/y axis scales from encoding specs.

use std::collections::HashMap;

use arrow::array::Array;
use arrow::record_batch::RecordBatch;

use crate::scale::discrete::{DiscreteLayout, DEFAULT_ALIGN};
use crate::scale::linear::LinearScale;
use crate::scale::log::LogScale;
use crate::scale::ordinal::OrdinalScale;
use crate::scale::pow::PowScale;
use crate::scale::symlog::SymlogScale;
use crate::scale::time::TimeScale;
use crate::spec::chart::ChartSpec;
use crate::spec::encoding::DataType as SpecDataType;

use crate::render::{RenderError, RenderWarning};

use super::domain::{
    apply_sort_to_domain, locate_field, numeric_domain_union, numeric_domain_union_over,
    positional_category_union, LayerUnion,
    SortContext,
};
use super::{
    column_min_max_f64, distinct_positional_categories, infer_spec_type, ScaleKind, SharedDomain,
};
use crate::transform::core::FINAL_OUTPUT_KEY;

/// The x/y field names bound at chart level, used to resolve data-aware sort
/// forms (channel shorthand `"-y"` etc.). Threaded into `build_axis_scale` so an
/// ordinal axis can sort its categories by the aggregate of the opposite
/// channel's field. Either side may be `None` (single-axis Tick/Rule marks).
#[derive(Clone, Copy, Default)]
pub(in crate::render) struct PositionalFields<'a> {
    pub(in crate::render) x: Option<&'a str>,
    pub(in crate::render) y: Option<&'a str>,
}

impl<'a> PositionalFields<'a> {
    /// The field bound on the **opposite** positional channel from `channel`:
    /// `y` for `"x"`, `x` for `"y"`.
    ///
    /// This is the "paired positional channel" of `#247`B's null-band occupancy
    /// rule (`arrow_cast::null_category_band_is_occupied`) — the second
    /// coordinate a mark needs before it can be placed at all — and NOT
    /// `build_axis_scale`'s `paired_enc`, which is the `x2`/`y2` span partner.
    /// Both sides are the RENDERED (post-`CoordFlip`) channels, like the struct's
    /// own fields, so the pairing is the one the marks actually draw with.
    ///
    /// `None` for any other channel token: only x and y have an opposite, and an
    /// auxiliary channel asking would be a caller bug rather than a domain whose
    /// null band should quietly vanish.
    fn paired_positional(self, channel: &str) -> Option<&'a str> {
        match channel {
            "x" => self.y,
            "y" => self.x,
            _ => {
                debug_assert!(false, "paired_positional called with non-positional channel {channel:?}");
                None
            }
        }
    }
}

// The padding-inset constants and `inset_pixel_range` now live in
// `crate::layout::geometry` (the geometry layer) so that the layout engine can
// reproduce the *exact same* inset when projecting axis ticks. Re-export the
// helper here under its existing `pub(in crate::render)` path so render-side
// callers are unchanged.
use crate::layout::geometry::{Rect, DEFAULT_SCALE_PADDING_FRAC};
pub(in crate::render) use crate::layout::geometry::inset_pixel_range;

/// Resolve the effective padding fraction.
///
/// Precedence:
///  - `Some(p)` → use `p` (including 0.0 to disable).
///  - `None && has_explicit_domain` → 0.0 (user-specified domain wins).
///  - `None && !has_explicit_domain` → `DEFAULT_SCALE_PADDING_FRAC`.
pub(in crate::render) fn resolve_padding_fraction(scale_padding: Option<f64>, has_explicit_domain: bool) -> f64 {
    if let Some(p) = scale_padding {
        return p;
    }
    if has_explicit_domain {
        return 0.0;
    }
    DEFAULT_SCALE_PADDING_FRAC
}

/// #183: a categorical axis asked to place more categories into its pixel
/// range than the range has room for degrades honestly (there is no invalid
/// input to refuse — d3's band/point math is well-defined for any domain
/// length) but silently: every category's band/point allotment rounds down
/// under one pixel, so distinct categories draw on top of each other or
/// vanish entirely, and nothing about the rendered chart says why. Checked
/// once, right after an [`OrdinalScale`] is built, from every positional
/// Ordinal/Band/Point construction site in this module — mirroring
/// [`RenderWarning::ColorPaletteOverflowed`]/[`RenderWarning::ShapePaletteOverflowed`]'s
/// "the degradation stays, but it is never silent" contract (spec §4.3).
///
/// `categories` is passed in rather than re-read off `scale` because two call
/// sites move their domain `Vec` into [`OrdinalScale::new_internal`] before
/// this runs; the caller already has the length in hand from before that
/// move.
///
/// **No placeholder-range special-casing here.** `build_axis_scale`/
/// `build_from_scale_spec` are also called with a metadata-only `(0.0, 1.0)`
/// pixel range from two sites that never place a single pixel —
/// `prepare/mod.rs`'s `provisional_scales` (needs only the domain/tick set
/// for `compute_layout`, before the real per-panel plot area exists) and
/// `build_secondary_y_axis_inputs`'s per-layer axis-input pass (same
/// reason). Over a 1px-wide placeholder, ANY domain of 2+ categories divides
/// to a sub-pixel bandwidth by construction, which would fire on every
/// ordinal-axis chart regardless of its real rendered size if this function
/// had to distinguish "placeholder" from "real" itself. It doesn't have to:
/// both placeholder call sites filter this SPECIFIC warning class out of
/// this function's returned `Vec` at the source
/// (`prepare/mod.rs::drop_placeholder_only_warnings`), so this function
/// stays a pure "is this bandwidth sub-pixel" check — the placeholder-vs-real
/// distinction is the CALLER's structural fact to know, not a magnitude this
/// function has to infer from the numbers it receives.
///
/// That filter is scoped to THIS warning class only (quality-review finding
/// #1, batch F/wave-1 task 11): every other scale warning the placeholder
/// pass can produce (color scheme, palette overflow, sort-spec-ignored, ...)
/// does not depend on the pixel range and is real regardless of which range
/// resolved it, so those survive the filter and still reach the user even
/// for a chart whose data is empty after transforms (where the panel-real
/// pass never runs to "re-emit" anything, since `build_scene`'s panel loop
/// skips every zero-row panel).
fn warn_if_ordinal_cardinality_degenerate(
    scale: &OrdinalScale,
    categories: usize,
    warnings: &mut Vec<RenderWarning>,
) {
    if categories > 0 && scale.bandwidth() < 1.0 {
        warnings.push(RenderWarning::GroupCardinalityDegenerate {
            categories: categories as u32,
            bandwidth_px: scale.bandwidth(),
        });
    }
}

#[allow(clippy::too_many_arguments)]
pub(in crate::render) fn build_axis_scale(
    channel: &'static str,
    enc: &crate::spec::encoding::EncodingSpec,
    paired_enc: Option<&crate::spec::encoding::EncodingSpec>,
    positional_fields: PositionalFields<'_>,
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    pixel_range: (f64, f64),
    spec: &ChartSpec,
    // Faceted + this channel resolves `ResolveMode::Shared`: the auto-inferred
    // domain also unions the global all-panels batch (`FINAL_OUTPUT_KEY`) so a
    // faceted raw field's positional domain spans every panel (T4). `false` for
    // non-faceted charts and `Independent`-mode channels → byte-identical
    // per-panel behavior. On the explicit `enc.scale` bypass it reaches only a
    // layered spec's domain-less continuous scale, through the same union.
    include_final: bool,
    // Composite-shared domain for this channel (D4b). `Some` only for a
    // composite leaf whose channel shares a domain across the tree; it seeds this
    // leaf's extent (numeric) or category vector (ordinal) on the SAME auto path
    // facet-shared panels use, so composite-shared axes get the auto path's
    // padding/`nice`, never the explicit-scale bypass. `None` for every
    // standalone (flat/facet) render → byte-identical. Consulted only after the
    // explicit `enc.scale` bypass below, so a genuine user scale still wins —
    // a SYSTEM-INJECTED scale (#247A) is the one thing that bypass now yields
    // to, and only when an auto path has a cross-mark domain to offer (this, or
    // a layered spec's own cross-layer union); see the gate below.
    shared: Option<&SharedDomain>,
    warnings: &mut Vec<RenderWarning>,
) -> Result<ScaleKind, RenderError> {
    let located = locate_field(&enc.field, primary_batch, transform_outputs)
        .ok_or_else(|| RenderError::UnknownColumn { name: enc.field.clone() })?;
    let dtype = infer_spec_type(enc, located.col.data_type());
    let pr = axis_pixel_range(channel, &dtype, pixel_range);
    // #95: `positional_fields` names the RENDERED channels (the caller derives
    // them from the post-`CoordFlip` encoding), while a `sort="-y"` shorthand
    // names the channel the user AUTHORED. `SortContext` carries the flag that
    // reconciles the two — see its `coord_flipped` doc. `spec.coord` is
    // untouched by the swap `prepare::build_layers` applies to the ENCODINGS,
    // which is why it is readable here and at every other site that asks the
    // same question off the same field (`domain::numeric_domain_union`'s
    // layer-union loop, `build_auxiliary_scales`, `resolve_scales`'s
    // `EncodingTypeMismatch`).
    let coord_flipped = matches!(spec.coord, Some(crate::spec::coord::CoordKind::Flip));

    // #247A: the explicit-scale bypass is for a scale the CALLER wrote. A scale
    // ferrum's own spec build injected (the bar zero-anchor — see
    // `ScaleSpec::Linear`'s `system_injected` doc) is not one, and it yields the
    // bypass to whichever auto path has a CROSS-MARK domain to offer. Before
    // this, the bypass returned before either was consulted, so a zero-anchored
    // bar could neither contribute to nor receive that domain and every other
    // mark in the chart drew against a scale the rendered ticks did not describe.
    //
    // The two auto paths are the two ways "layer a bar with a point" lowers, and
    // both reach this one gate:
    //
    //   * `shared.is_some()` — a composite leaf (`fm.layer(bar, point)`), whose
    //     shared domain already carries the anchor, applied to the bar leaf's own
    //     contribution by `composite::leaf_positional_domain`;
    //   * `spec.layers.is_some()` — the flat layered lowering (`bar + point`,
    //     whose chart-level encoding is the LHS bar's, so the injected scale sat
    //     on the axis every layer resolves against). Here the anchor comes from
    //     `domain::numeric_domain_union`'s own `layer_wants_bar_zero_anchor`,
    //     which exists for precisely this shape and was unreachable through it
    //     while the bypass fired first. That asymmetry was visible: `bar + point`
    //     rendered `0..3` while `point + bar` (chart-level encoding = the point's,
    //     so no injected scale, so the auto path) rendered the correct `0..80`.
    //
    // With NEITHER — a standalone single-mark chart — there is nothing to yield
    // to and the bypass stands, so every single bar render is byte-identical by
    // construction. That is why provenance was the route rather than deleting the
    // injection (spec §8): the auto path's padding/`nice` treatment never reaches
    // an unlayered bar.
    //
    // A layered spec whose layers contain no bar loses the anchor here, and that
    // is correct rather than tolerated: `prepare::build_layers` builds layers from
    // `spec.layers` ALONE when it is `Some`, and `Layer::mark` is required, so
    // `spec.mark` draws nothing in a layered spec. No bar is drawn, so no bar
    // baseline needs keeping in view — today's behavior anchors an axis for a mark
    // that does not exist. Pinned by
    // `layered_spec_with_no_bar_layer_takes_the_union_without_an_anchor`.
    // `is_some_and(!is_empty)`, not a bare `is_some()`: the question is "are there
    // layers whose fields the cross-layer union can read", and a `Some(vec![])`
    // spec has none — it would strip the anchor while offering nothing to union
    // against. No Python lowering emits an empty layers list today (the key is
    // written only when `_layers is not None`, and every producer appends at least
    // the LHS's own layer), so this is the predicate saying what it means rather
    // than a fix for a live shape. Pinned by
    // `injected_scale_with_an_empty_layers_list_keeps_the_bypass`.
    let layered = spec.layers.as_ref().is_some_and(|l| !l.is_empty());
    let yields_to_auto_path = shared.is_some() || layered;
    let explicit_scale =
        enc.scale.as_ref().filter(|s| !(yields_to_auto_path && s.is_system_injected()));
    if let Some(scale_spec) = explicit_scale {
        let sort_ctx = SortContext {
            category_field: &enc.field,
            batch: located.batch,
            x_field: positional_fields.x,
            y_field: positional_fields.y,
            coord_flipped,
        };
        // What a continuous scale stating no `domain` spans: the data of every
        // layer carrying this same scale, with their paired `x2`/`y2`
        // endpoints and the facet-shared global batch. A composite mark's
        // layers each carry a copy of the one scale the caller wrote on it, so
        // the axis covers the whole mark; a sibling layer without it stays out
        // (see `LayerUnion::CarryingScale`). Unlayered, it is the bound column
        // and its paired `x2`/`y2` column.
        let data_extent = || {
            if layered {
                numeric_domain_union_over(
                    channel,
                    &enc.field,
                    paired_enc.map(|p| p.field.as_str()),
                    primary_batch,
                    transform_outputs,
                    spec,
                    include_final,
                    LayerUnion::CarryingScale(scale_spec),
                )
                .map_err(|e| match e {
                    RenderError::UnsupportedDtype { field, dtype, context: None } => {
                        RenderError::UnsupportedDtype { field, dtype, context: Some("scale") }
                    }
                    e => e,
                })
            } else {
                let (lo, hi) = column_extent(located.batch, &enc.field)?;
                match paired_enc.and_then(|p| locate_field(&p.field, primary_batch, transform_outputs).map(|l| (l, p))) {
                    Some((l, p)) => {
                        let (plo, phi) = column_extent(l.batch, &p.field)?;
                        Ok((lo.min(plo), hi.max(phi)))
                    }
                    None => Ok((lo, hi)),
                }
            }
        };
        return build_from_scale_spec(channel, scale_spec, enc, located.batch, pr, &sort_ctx, &data_extent, warnings);
    }

    let inset = inset_pixel_range(pr, resolve_padding_fraction(None, false));

    match dtype {
        SpecDataType::Quantitative => {
            let (min, max) = resolve_numeric_extent(
                shared, channel, enc, paired_enc, primary_batch, transform_outputs, spec, include_final,
            )?;
            Ok(ScaleKind::Linear(LinearScale::new_internal(
                vec![min, max], vec![inset.0, inset.1], false, false,
            )))
        }
        SpecDataType::Ordinal | SpecDataType::Nominal => {
            // Composite-shared ordinal (D4b): the seeded union vector arrives
            // ALREADY ordered (the resolve pass's order-preserving union, D2)
            // and is authoritative — local data-aware re-sorting is skipped,
            // because each leaf would compute the sort aggregate from its OWN
            // batch and different leaves could reorder the "shared" domain
            // differently (there is no cross-leaf merged batch to sort on,
            // unlike facet-shared's FINAL_OUTPUT_KEY). Falls back to this
            // leaf's own categories + normal sort when no composite sharing
            // applies.
            let mut domain = match shared {
                Some(SharedDomain::Ordinal(cats)) => {
                    let scale = OrdinalScale::new_internal(
                        cats.clone(), vec![pr.0, pr.1], DiscreteLayout::UNPADDED,
                    );
                    warn_if_ordinal_cardinality_degenerate(&scale, cats.len(), warnings);
                    return Ok(ScaleKind::Ordinal(scale));
                }
                // The chart-level column AND every layer's own column for this
                // channel (task 12c) — the ordinal sibling of the Quantitative
                // arm's `numeric_domain_union` just above, so both axes answer
                // "does the shared domain cover every layer" the same way. A
                // layered chart whose layers name the same positional field
                // resolves byte-identically; a cross-DataFrame `a + b` merge,
                // whose RHS column is renamed, stops losing the RHS categories
                // (and with them every RHS mark, for want of a band).
                _ => positional_category_union(
                    channel,
                    &enc.field,
                    positional_fields.paired_positional(channel),
                    primary_batch,
                    transform_outputs,
                    spec,
                    include_final,
                )?,
            };
            // For data-aware sort forms (`"-y"`, `"y"`, `{field, op, order}`),
            // `apply_sort_to_domain` computes a per-category aggregate from
            // `sort_ctx.batch`.  When the channel resolves Shared (`include_final`),
            // using the per-panel batch causes each panel to re-sort the shared
            // category vector by its OWN aggregate, placing marks under the wrong
            // tick.  Mirror `positional_category_union`: use the global
            // batch (`FINAL_OUTPUT_KEY`) so every panel's data-aware sort agrees
            // with the global/provisional axis order.  Fall back to the per-panel
            // batch when the global batch is absent (defensive, same as the
            // membership helper).  For Independent channels (`include_final == false`)
            // keep the per-panel batch so that escape hatch remains byte-identical.
            let sort_batch = if include_final {
                transform_outputs
                    .get(FINAL_OUTPUT_KEY)
                    .unwrap_or(located.batch)
            } else {
                located.batch
            };
            let sort_ctx = SortContext {
                category_field: &enc.field,
                batch: sort_batch,
                x_field: positional_fields.x,
                y_field: positional_fields.y,
                coord_flipped,
            };
            apply_sort_to_domain(&mut domain, enc.sort.as_ref(), &sort_ctx, warnings);
            // An auto-inferred categorical axis has no user padding to honor:
            // the unpadded band model, which reduces to the pre-F-L04-03
            // symmetric model bit-exactly (see `crate::scale::discrete`).
            let categories = domain.len();
            let scale = OrdinalScale::new_internal(
                domain, vec![pr.0, pr.1], DiscreteLayout::UNPADDED,
            );
            warn_if_ordinal_cardinality_degenerate(&scale, categories, warnings);
            Ok(ScaleKind::Ordinal(scale))
        }
        SpecDataType::Temporal => {
            let (min, max) = resolve_numeric_extent(
                shared, channel, enc, paired_enc, primary_batch, transform_outputs, spec, include_final,
            )?;
            Ok(ScaleKind::Time(TimeScale::new_internal(
                vec![min, max], vec![inset.0, inset.1], false, false,
            )))
        }
    }
}

/// Resolve a numeric/temporal axis extent: the composite shared domain (D4b) when
/// the leaf participates in composite sharing on this channel, otherwise the
/// leaf's own facet-union-aware auto-inferred extent. Consulted on the AUTO path
/// only (the explicit-scale bypass returns before reaching here), so the shared
/// extent receives the same padding/`nice` treatment as a facet-shared panel. A
/// non-numeric `shared` (never produced for a numeric leaf by the resolve pass)
/// falls through to the union, so this is a safe no-op when `shared` is `None`.
#[allow(clippy::too_many_arguments)]
fn resolve_numeric_extent(
    shared: Option<&SharedDomain>,
    channel: &str,
    enc: &crate::spec::encoding::EncodingSpec,
    paired_enc: Option<&crate::spec::encoding::EncodingSpec>,
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    spec: &ChartSpec,
    include_final: bool,
) -> Result<(f64, f64), RenderError> {
    if let Some(SharedDomain::Numeric { lo, hi }) = shared {
        return Ok((*lo, *hi));
    }
    numeric_domain_union(
        channel,
        &enc.field,
        paired_enc.map(|p| p.field.as_str()),
        primary_batch,
        transform_outputs,
        spec,
        include_final,
    )
}

/// Y-axis pixel range is inverted ONLY for quantitative/temporal scales.
pub(in crate::render) fn axis_pixel_range(channel: &str, dtype: &SpecDataType, pixel_range: (f64, f64)) -> (f64, f64) {
    if channel == "y" && !matches!(dtype, SpecDataType::Ordinal | SpecDataType::Nominal) {
        (pixel_range.1, pixel_range.0)
    } else {
        pixel_range
    }
}

/// The fraction of `scale`'s data domain at `d` — `0.0` at `domain.0`, `1.0`
/// at `domain.1`, and past either end for an out-of-domain value.
///
/// `None` for an ordinal scale (no numeric domain) and `Some(0.5)` for a
/// degenerate one (both endpoints equal), which is the midpoint every caller
/// wants there.
fn domain_fraction(scale: &ScaleKind, d: f64) -> Option<f64> {
    let (lo, hi) = scale.data_domain()?;
    Some(if (hi - lo).abs() < f64::EPSILON {
        0.5
    } else {
        (d - lo) / (hi - lo)
    })
}

/// Resolve a data-space **x** value to an absolute pixel, including the
/// out-of-domain/ordinal fallback `to_pixel_f64` refuses.
///
/// Unlike [`data_pixel_y`], this needs no flip: [`axis_pixel_range`] just
/// above only swaps the pixel range for the y channel, so a continuous x
/// axis's `domain.0` already maps to the LEFT pixel and the fraction applies
/// directly. An ordinal scale has no numeric position for a number, so the
/// value centers.
pub(in crate::render) fn data_pixel_x(scale: &ScaleKind, plot_area: &Rect, d: f64) -> f64 {
    scale.to_pixel_f64(d).unwrap_or_else(|| match domain_fraction(scale, d) {
        Some(frac) => plot_area.x + frac * plot_area.w,
        None => plot_area.x + plot_area.w * 0.5,
    })
}

/// Resolve a data-space **y** value to an absolute pixel, including the
/// out-of-domain/ordinal fallback `to_pixel_f64` refuses.
///
/// `to_pixel_f64` returns `None` for an OUT-OF-DOMAIN value on a continuous
/// scale, not only for an ordinal one (see its doc comment), and that is the
/// case this fallback exists for.
///
/// Every non-ordinal y axis's pixel RANGE is flipped relative to its domain,
/// unconditionally — that is [`axis_pixel_range`], immediately above, and the
/// reason this pair lives beside it rather than in the annotation module
/// that used to own it: domain[0] maps to the BOTTOM pixel and domain[1] to
/// the TOP. (`reverse=` only swaps which value sits at domain[0] vs
/// domain[1], never which pixel each endpoint maps to, so `data_domain()`'s
/// `(lo, hi)` already encodes `reverse=` and this flip composes with it for
/// free.) `(1.0 - frac)` is that same flip: frac=0 (at `lo`) lands at the
/// bottom, frac=1 (at `hi`) at the top — matching every in-domain annotation
/// and every mark on this axis.
///
/// Before this flip landed the formula used `frac` directly, so an
/// out-of-domain value landed on the WRONG side — on broken AND unbroken
/// charts alike (`annotate_hline(y=-50)` on a `[0, 100]` axis rendered
/// ABOVE the plot); a `BreakAxis` only made it conspicuous by clamping the
/// value back onto the visible canvas. Not separately audited under
/// `CoordFlip` — that swap happens upstream of this module.
///
/// This pair is THE fallback for both of its callers —
/// `annotation::ScaleContext`'s `resolve_x`/`resolve_y` and `scene_build`'s
/// `Inset(connect_to=…)` resolution — so the flip cannot be fixed in one and
/// left wrong in the other, which is exactly what happened while the formula
/// was duplicated.
pub(in crate::render) fn data_pixel_y(scale: &ScaleKind, plot_area: &Rect, d: f64) -> f64 {
    scale.to_pixel_f64(d).unwrap_or_else(|| match domain_fraction(scale, d) {
        Some(frac) => plot_area.y + (1.0 - frac) * plot_area.h,
        None => plot_area.y + plot_area.h * 0.5,
    })
}

/// Build a ScaleKind from an explicit ScaleSpec, using the given pixel range.
///
/// `data_extent` answers what a continuous scale stating no `domain` spans; it
/// is asked only then, so a categorical scale on a non-numeric column never
/// reads it.
///
/// `channel` (`"x"`/`"y"`) names the axis in the #152 explicit-domain/range shape
/// refusal below — the same field `apply_axis_domain_config`'s sibling refusal on
/// this type already carries — and picks the opposite positional channel for the
/// discrete arms' null-band occupancy rule (`#247`B).
#[allow(clippy::too_many_arguments)]
pub(in crate::render) fn build_from_scale_spec(
    channel: &str,
    scale_spec: &crate::spec::encoding::ScaleSpec,
    enc: &crate::spec::encoding::EncodingSpec,
    batch: &RecordBatch,
    pr: (f64, f64),
    sort_ctx: &SortContext<'_>,
    data_extent: &dyn Fn() -> Result<(f64, f64), RenderError>,
    warnings: &mut Vec<RenderWarning>,
) -> Result<ScaleKind, RenderError> {
    use crate::spec::encoding::ScaleSpec;

    if batch.column_by_name(&enc.field).is_none() {
        return Err(RenderError::UnknownColumn { name: enc.field.clone() });
    }

    // `#247`B: an explicit Ordinal/Band/Point scale that names no `domain` resolves
    // one from the data, so it bands nulls by the same rule the auto path does.
    // `SortContext` already carries the rendered x/y field pair (it needs them for
    // data-aware sort forms), so the pairing is re-read off it rather than passed a
    // second time — one helper answers "which channel is opposite this one" for
    // both entry points, so the two cannot disagree about the null band.
    let paired_positional =
        PositionalFields { x: sort_ctx.x_field, y: sort_ctx.y_field }.paired_positional(channel);

    Ok(match scale_spec {
        // `system_injected` is read by `build_axis_scale`'s bypass gate above,
        // which decides WHETHER this function is called at all; once a scale
        // reaches here it resolves identically whoever wrote it (#247A).
        ScaleSpec::Linear { common, nice, zero, system_injected: _ } => {
            validate_explicit_continuous_shape(channel, &common.domain, &common.range)?;
            let (d, r) = resolve_continuous_domain_and_range(&common.domain, &common.range, common.padding, data_extent, pr, *zero, common.reverse)?;
            ScaleKind::Linear(LinearScale::new_internal(d, r, common.clamp, *nice))
        }
        ScaleSpec::Log { base, common, nice } => {
            validate_explicit_continuous_shape(channel, &common.domain, &common.range)?;
            let (d, r) = resolve_continuous_domain_and_range(&common.domain, &common.range, common.padding, data_extent, pr, false, common.reverse)?;
            ScaleKind::Log(LogScale::new_internal(d, r, *base, common.clamp, *nice))
        }
        ScaleSpec::Time { common, nice } => {
            validate_explicit_continuous_shape(channel, &common.domain, &common.range)?;
            let (d, r) = resolve_continuous_domain_and_range(&common.domain, &common.range, common.padding, data_extent, pr, false, common.reverse)?;
            ScaleKind::Time(TimeScale::new_internal(d, r, common.clamp, *nice))
        }
        ScaleSpec::Symlog { constant, common, nice } => {
            validate_explicit_continuous_shape(channel, &common.domain, &common.range)?;
            let (d, r) = resolve_continuous_domain_and_range(&common.domain, &common.range, common.padding, data_extent, pr, false, common.reverse)?;
            ScaleKind::Symlog(SymlogScale::new_internal(d, r, *constant, common.clamp, *nice))
        }
        ScaleSpec::Ordinal { domain, range, padding } => {
            let mut d = match domain {
                Some(d) => d.clone(),
                None => distinct_positional_categories(batch, &enc.field, paired_positional)?,
            };
            apply_sort_to_domain(&mut d, enc.sort.as_ref(), sort_ctx, warnings);
            // Extract numeric pixel range from the typed polymorphic `range`.
            // String values (color ranges) are handled by `build_color_scale` —
            // fall back to the pixel-range default when the range has no numbers.
            let (pixel_range, explicit) = ordinal_pixel_range(range.as_deref(), pr);
            // `ScaleSpec::Ordinal` carries only an inner padding (no outer, no
            // align) — the band model with d3's default alignment.
            let layout = DiscreteLayout::band(*padding, 0.0, DEFAULT_ALIGN);
            let categories = d.len();
            let scale = OrdinalScale::new_internal(d, pixel_range, layout)
                .with_explicit_range(explicit);
            warn_if_ordinal_cardinality_degenerate(&scale, categories, warnings);
            ScaleKind::Ordinal(scale)
        }
        ScaleSpec::Pow { exponent, common } => {
            validate_explicit_continuous_shape(channel, &common.domain, &common.range)?;
            let (d, r) = resolve_continuous_domain_and_range(&common.domain, &common.range, common.padding, data_extent, pr, false, common.reverse)?;
            ScaleKind::Pow(PowScale::new_internal(d, r, *exponent, common.clamp))
        }
        ScaleSpec::Sqrt { common } => {
            validate_explicit_continuous_shape(channel, &common.domain, &common.range)?;
            let (d, r) = resolve_continuous_domain_and_range(&common.domain, &common.range, common.padding, data_extent, pr, false, common.reverse)?;
            ScaleKind::Pow(PowScale::new_internal(d, r, 0.5, common.clamp))
        }
        ScaleSpec::Utc { common, nice } => {
            validate_explicit_continuous_shape(channel, &common.domain, &common.range)?;
            let (d, r) = resolve_continuous_domain_and_range(&common.domain, &common.range, common.padding, data_extent, pr, false, common.reverse)?;
            // F-L04-06: `utc=true` survives resolution — the wire tag no
            // longer collapses into an indistinguishable `TimeScale` once
            // resolved (`new_internal` used to default it away here).
            ScaleKind::Time(
                TimeScale::new_internal(d, r, common.clamp, *nice).with_utc(true),
            )
        }
        ScaleSpec::Band { domain, padding, padding_inner, padding_outer, align, range } => {
            let mut d = match domain {
                Some(d) => d.clone(),
                None => distinct_positional_categories(batch, &enc.field, paired_positional)?,
            };
            apply_sort_to_domain(&mut d, enc.sort.as_ref(), sort_ctx, warnings);
            // d3's `padding` is a shorthand that sets BOTH inner and outer;
            // an explicit `padding_inner`/`padding_outer` overrides its side.
            let layout = DiscreteLayout::band(
                padding_inner.unwrap_or(*padding),
                padding_outer.unwrap_or(*padding),
                *align,
            );
            let categories = d.len();
            let (band_range, explicit) = band_point_pixel_range(range.as_deref(), pr);
            let scale = OrdinalScale::new_internal(d, band_range, layout)
                .with_explicit_range(explicit);
            warn_if_ordinal_cardinality_degenerate(&scale, categories, warnings);
            ScaleKind::Ordinal(scale)
        }
        ScaleSpec::Point { domain, padding, align, range, reverse } => {
            let mut d = match domain {
                Some(d) => d.clone(),
                None => distinct_positional_categories(batch, &enc.field, paired_positional)?,
            };
            apply_sort_to_domain(&mut d, enc.sort.as_ref(), sort_ctx, warnings);
            // PointScale(reverse=True) reverses the resolved domain order (GH #65).
            // Domain reversal — not a pixel-range flip — so axis tick labels
            // (domain-order via tick_labels/uniform_center) and explicit-range band
            // centers follow the marks automatically. Reversing the domain mirrors
            // every position about the range midpoint, which is the reverse
            // semantic of the pyclass facade (scale/point.rs). The facade's BASE
            // positions used to diverge from this resolver's symmetric band model
            // (the former #67 north star); as of F-L04-03 both compute the true
            // point formula from `crate::scale::discrete`, so only the reverse
            // *mechanism* differs now, not the positions. Applied after sort:
            // "sort, then reverse", matching d3/Vega composition.
            if *reverse {
                d.reverse();
            }
            let categories = d.len();
            let (point_range, explicit) = band_point_pixel_range(range.as_deref(), pr);
            let scale =
                OrdinalScale::new_internal(d, point_range, DiscreteLayout::point(*padding, *align))
                    .with_explicit_range(explicit);
            warn_if_ordinal_cardinality_degenerate(&scale, categories, warnings);
            ScaleKind::Ordinal(scale)
        }
        // Continuous-degrading variants: whether the spec's `domain` is a
        // positional extent or a discrete-binning artifact is classified by
        // ScaleSpec::positional_extent() (exhaustive — see issue #40).
        //
        // These discretizing/diverging positional scales degrade to
        // `ScaleKind::Linear` before minor-tick generation is reached, so
        // they receive linear-subdivided minor ticks (semantic corner,
        // archaeology R7).
        //
        // #152 deliberately does NOT call `validate_explicit_continuous_shape`
        // here: `positional_extent()` (below) already normalizes a Some(..)
        // domain to exactly 2 elements or `None` before it ever reaches
        // `resolve_continuous_domain_and_range`, so the length panic #152
        // fixes cannot reach this arm — a different domain shape (Diverging's
        // 3-element `[lo, mid, hi]`) is what this family actually carries.
        // #152's frozen report is specifically the 7 `ContinuousScaleCommon`
        // variants' `domain`/`range` fields (`scale/linear.rs:134`'s panic);
        // widening the same refusal onto this sibling family's OWN degenerate
        // extents is a separate, undisclosed change this task does not make.
        ScaleSpec::Sequential { .. }
        | ScaleSpec::Diverging { .. }
        | ScaleSpec::Quantize { .. }
        | ScaleSpec::BinOrdinal { .. }
        | ScaleSpec::Quantile { .. }
        | ScaleSpec::Threshold { .. } => {
            // Neither `zero` nor `reverse` apply to a degraded positional
            // extent — none of these variants carries a `ContinuousScaleCommon`
            // (color-channel `reverse`, where one of them has it, is a
            // completely separate resolution path — `scale_resolve::color`),
            // so both are hard-coded `false` rather than threading a value
            // that could never mean anything else here.
            let (d, r) = resolve_continuous_domain_and_range(&scale_spec.positional_extent(), &None, None, data_extent, pr, false, false)?;
            ScaleKind::Linear(LinearScale::new_internal(d, r, false, false))
        }
    })
}

/// Resolve the pixel-coordinate range for a `ScaleSpec::Ordinal`, plus
/// whether that range counts as **explicit** for band-geometry purposes
/// (GH #39 phase 2): `true` only when at least 2 numeric entries were found,
/// mirroring `band_point_pixel_range`'s `>= 2`-entry gate so Band, Point, and
/// positional Ordinal scales are never special-cased against each other.
///
/// The typed `range` carries `Number` entries (pixel coordinates) and/or `Str`
/// entries (color strings, handled by `build_color_scale`). This helper pulls
/// the numbers for the positional resolver. When `range` is absent or carries
/// no numbers (i.e. an all-string color range), the plot-area extent
/// `[pr.0, pr.1]` is returned so the positional scale still functions. A
/// single numeric entry is passed through unchanged (pre-existing arithmetic,
/// untouched here) but does not count as explicit — a user-supplied pixel
/// range needs both endpoints before the axis may place its ticks on it.
fn ordinal_pixel_range(
    range: Option<&[crate::scale::ordinal::OrdinalRangeValue]>,
    pr: (f64, f64),
) -> (Vec<f64>, bool) {
    match range {
        Some(values) => {
            let nums = crate::scale::ordinal::OrdinalRangeValue::numbers(values);
            if nums.is_empty() {
                (vec![pr.0, pr.1], false)
            } else {
                let explicit = nums.len() >= 2;
                (nums, explicit)
            }
        }
        None => (vec![pr.0, pr.1], false),
    }
}

/// Resolve the pixel-coordinate range for a `ScaleSpec::Band`/`ScaleSpec::Point`,
/// plus whether that range was explicitly supplied by the user (GH #39
/// phase 2): `true` only for a usable (>= 2-entry) numeric range; the
/// panel-extent fallback is never explicit.
///
/// Explicit numeric pixel range for band/point scales; falls back to the
/// panel extent when absent or degenerate (fewer than 2 entries).
fn band_point_pixel_range(range: Option<&[f64]>, pr: (f64, f64)) -> (Vec<f64>, bool) {
    match range {
        Some(r) if r.len() >= 2 => (vec![r[0], r[1]], true),
        _ => (vec![pr.0, pr.1], false),
    }
}

/// Refuse an EXPLICIT continuous `domain`/`range` whose LENGTH a raw-dict
/// scale spec can smuggle past every pyclass constructor's own guard (#152):
/// `scale={'type': 'linear', 'domain': [5.0]}` is a wire shape the array gate
/// accepts, and used to reach `LinearScale::new_internal`'s `domain[1]`
/// indexing unchecked, panicking `index out of bounds` across the PyO3
/// boundary instead of raising. Reuses `scale::core`'s own length-check
/// wording (`"domain must have length 2"` / `"range must have length 2"`) so
/// a raw-dict scale refuses with the exact sentence its pyclass twin
/// (`LinearScale(domain=...)`, resolved via `scale::core::resolve_continuous`)
/// already uses for the same shape. `min > max` (a reversed pair) is
/// deliberately NOT refused, matching the class surface's own
/// accepted-reversed-axis contract.
///
/// Deliberately does NOT also refuse a degenerate `[c, c]` domain/range,
/// unlike `validate_continuous_domain`'s own sibling check and
/// `apply_axis_domain_config`'s `InvalidScaleDomainConfig` a few lines above:
/// `resolve_param_domains`'s reactive `domainParam` substitution (D6) clears
/// `domain_param` and writes the resolved value straight into this same
/// `common.domain` field with no surviving marker, so this funnel cannot
/// distinguish a hand-authored literal domain from a domainParam's current
/// (possibly transiently degenerate, e.g. a slider mid-drag) value — and
/// `tests/test_bug_hunt_reactive_params.py::TestDomainParamDegenerate
/// ::test_collapsed_domain_no_nan` already pins that a degenerate
/// `domainParam` value must keep rendering (via the scale's own GH #104
/// zero-span-to-midpoint handling), not refuse. Refusing degenerate domains
/// here would silently break that sanctioned, pre-existing contract for
/// every domainParam whose CURRENT value happens to collapse — an
/// undisclosed regression, not a #152 fix. #152's frozen report is the
/// length-panic only; the length check above carries no such conflict
/// (`ParamStore::numeric_domain` already filters a too-short param value to
/// `None` — auto-infer — before it ever reaches `common.domain`, per
/// `test_single_element_array_falls_back`).
///
/// Called from the seven `ContinuousScaleCommon` arms of
/// `build_from_scale_spec` ONLY — see the comment on the sibling
/// "continuous-degrading" arm (`Sequential`/`Diverging`/`Quantize`/
/// `BinOrdinal`/`Quantile`/`Threshold`) for why that family is out of scope.
fn validate_explicit_continuous_shape(
    channel: &str,
    domain: &Option<Vec<f64>>,
    range: &Option<Vec<f64>>,
) -> Result<(), RenderError> {
    if let Some(d) = domain {
        if d.len() != 2 {
            return Err(RenderError::InvalidScaleDomainConfig {
                channel: channel.to_string(),
                reason: format!("domain must have length 2; got {}", d.len()),
            });
        }
    }
    if let Some(r) = range {
        if r.len() != 2 {
            return Err(RenderError::InvalidScaleDomainConfig {
                channel: channel.to_string(),
                reason: format!("range must have length 2; got {}", r.len()),
            });
        }
    }
    Ok(())
}

/// Widen a resolved `[lo, hi]` extent so it includes zero — the zero-anchor's
/// arithmetic, in one place instead of per-caller.
///
/// Spelled as the two guarded comparisons rather than `lo.min(0.0)` /
/// `hi.max(0.0)` deliberately: `f64::min`/`max` return the non-NaN operand, so
/// they would turn an all-null column's `NaN` extent into `0.0`, while the
/// comparison form leaves `NaN` untouched (`NaN > 0.0` is false). All three
/// callers relied on that, so the extraction is behavior-preserving rather than
/// merely equivalent-looking.
///
/// **Every site that widens an extent to zero binds here** — the ferrum-wide
/// answer to "does this axis reach zero", in one function:
///
/// * [`resolve_continuous_domain_and_range`] below — a
///   `ScaleSpec::Linear { zero: true }` resolving on the explicit-scale path;
/// * `composite::leaf_positional_domain` — a zero-anchored bar leaf's
///   contribution to a cross-leaf domain union (#247A), where an injected
///   scale's anchor lands now that the leaf no longer takes the bypass;
/// * `domain::numeric_domain_union`'s `bar_zero_wanted` block — a layered bar on
///   the auto path, which is the mechanism [`build_axis_scale`]'s own
///   system-injected gate hands the flat-layered arm's anchor to, so it is
///   downstream of that gate rather than a neighbour of it.
///
/// A fourth site would be a fourth chance for the answers to disagree, which is
/// what happened before this existed: the third site carried a comment claiming
/// to mirror the first "exactly (same two guarded assignments)", by which time
/// the first had become a call to this function.
pub(in crate::render) fn extend_extent_to_zero(lo: f64, hi: f64) -> (f64, f64) {
    (if lo > 0.0 { 0.0 } else { lo }, if hi < 0.0 { 0.0 } else { hi })
}

/// Resolve `(domain, range)` for a continuous ScaleSpec variant, then apply
/// `zero`'s zero-extension and `reverse`'s domain-swap sugar, in that order,
/// as the LAST two steps (rust-design review S3-1 / refactor 4).
///
/// This is now the ONE call every continuous variant's arm in
/// `build_from_scale_spec` makes — replacing seven call sites that each
/// separately restated `apply_domain_reverse(d, common.reverse)` after
/// calling this function, a duplication nothing enforced (an eighth
/// continuous variant could have called this function, resolved fine, and
/// silently never reversed). `zero`/`reverse` are required parameters here,
/// not optional ones a future call site could omit: a variant that means
/// neither passes `false, false` explicitly (see the continuous-degrading
/// arm below), so forgetting to decide is a missing-argument compile error,
/// not a runtime gap.
///
/// `zero` is `ScaleSpec::Linear`'s own field — the only continuous variant
/// that carries one — applied BEFORE the swap: `apply_domain_reverse`'s doc
/// explains why reversing first would corrupt `zero`'s ascending-pair
/// assumption rather than merely fail to match a hand-written descending
/// domain. Every other variant passes `false`, a guaranteed no-op (the
/// `d.len() == 2` guard below).
fn resolve_continuous_domain_and_range(
    domain: &Option<Vec<f64>>,
    range: &Option<Vec<f64>>,
    padding: Option<f64>,
    data_extent: &dyn Fn() -> Result<(f64, f64), RenderError>,
    pr: (f64, f64),
    zero: bool,
    reverse: bool,
) -> Result<(Vec<f64>, Vec<f64>), RenderError> {
    let mut d = match domain {
        Some(d) => d.clone(),
        None => {
            let (mn, mx) = data_extent()?;
            vec![mn, mx]
        }
    };
    let r = if let Some(r) = range {
        r.clone()
    } else {
        let frac = resolve_padding_fraction(padding, domain.is_some());
        let inset = inset_pixel_range(pr, frac);
        vec![inset.0, inset.1]
    };
    if zero && d.len() == 2 {
        let (lo, hi) = extend_extent_to_zero(d[0], d[1]);
        d[0] = lo;
        d[1] = hi;
    }
    let d = apply_domain_reverse(d, reverse);
    Ok((d, r))
}

/// The data extent of `field` in `batch`, refused in the scale's voice: what
/// an explicit continuous scale stating no `domain` reads per column on an
/// unlayered spec.
fn column_extent(batch: &RecordBatch, field: &str) -> Result<(f64, f64), RenderError> {
    let col = batch
        .column_by_name(field)
        .ok_or_else(|| RenderError::UnknownColumn { name: field.to_string() })?;
    column_min_max_f64(col).map_err(|_| RenderError::UnsupportedDtype {
        field: field.to_string(),
        dtype: format!("{:?}", col.data_type()),
        context: Some("scale"),
    })
}

/// Domain-swap sugar for `ContinuousScaleCommon::reverse` (F-L04-07, spec
/// §4C): swaps the resolved domain pair. For an explicit `domain=[a, b]`
/// with `zero=false` (the only case where no other step reorders the pair),
/// this is exactly equivalent to writing `domain=[b, a]` by hand — see
/// `ContinuousScaleCommon::reverse`'s doc for the two cases where it is not
/// (auto-inferred domain's padding inset; `zero=true`'s degenerate-collapse
/// hazard covered next).
///
/// Called from ONE place — `resolve_continuous_domain_and_range`'s own tail
/// (rust-design review S3-1 / refactor 4; formerly restated once per
/// continuous variant arm in `build_from_scale_spec`, after that call, with
/// nothing enforcing the restatement) — as the LAST step, after that
/// function's own `zero`-extension block (Linear's `zero` field; every other
/// variant passes `zero: false`). This ordering is load-bearing, and the
/// reason is NOT that it matches a hand-written descending domain — it does
/// not: `domain=[100, 0]` with `zero=true` collapses to `[0.0, 0.0]` today
/// (pre-existing, independent of `reverse`), because the `zero` block treats
/// `d[0]` as the minimum (`if d[0] > 0.0 { d[0] = 0.0; }`, `if d[1] < 0.0 {
/// d[1] = 0.0; }`), an assumption only an ASCENDING pair satisfies. The real
/// reason to reverse LAST is that reversing FIRST would hand that same
/// ascending-pair assumption a descending pair instead, silently corrupting
/// the result — NOT necessarily to `[0.0, 0.0]` (the exact wrong value
/// depends on the domain: an auto-inferred `[20, 80]` reversed-then-zeroed
/// comes out `[0.0, 20.0]`, not `[0.0, 0.0]`, and not the correct `[80.0,
/// 0.0]` either) — see
/// `linear_zero_true_reverse_true_yields_non_degenerate_descending_domain`
/// below, which pins the CORRECT ordering by failing (on the concrete
/// `[0.0, 20.0]` wrong value above) if the swap is hoisted above the `zero`
/// block.
///
/// Range orientation, the structural y-inversion predicate
/// (`axis_pixel_range`), and tick label/fraction pairing are untouched —
/// this only flips which value lands at each end of the domain pair.
/// Downstream (`apply_axis_domain_config`'s lo>hi normalize/restore, tick
/// generation, mark positioning) already tolerates a descending domain
/// (batch-B's domain-config cascade proved this), so no other call site
/// needs to change. A no-op on anything but a 2-element domain, mirroring
/// the `zero` guard immediately above it.
fn apply_domain_reverse(mut d: Vec<f64>, reverse: bool) -> Vec<f64> {
    if reverse && d.len() == 2 {
        d.swap(0, 1);
    }
    d
}

/// Apply coord-level domain and padding overrides to the x/y scales.
pub(in crate::render) fn apply_coord_domain_overrides(
    spec: &ChartSpec,
    x: &mut ScaleKind,
    y: &mut ScaleKind,
    x_pixel_range: (f64, f64),
    y_pixel_range: (f64, f64),
) {
    use crate::spec::coord::CoordKind;

    let (x_domain_override, y_domain_override, expand) = match &spec.coord {
        Some(CoordKind::Cartesian { x_domain, y_domain, expand, .. }) => {
            (*x_domain, *y_domain, *expand)
        }
        Some(CoordKind::Fixed { x_domain, y_domain, expand, .. }) => {
            (*x_domain, *y_domain, *expand)
        }
        _ => return,
    };

    if let Some((lo, hi)) = x_domain_override {
        let pr = if expand {
            inset_pixel_range(x_pixel_range, resolve_padding_fraction(None, false))
        } else {
            x_pixel_range
        };
        *x = ScaleKind::Linear(LinearScale::new_internal(
            vec![lo, hi], vec![pr.0, pr.1], false, false,
        ));
    } else if !expand {
        if let ScaleKind::Linear(ref inner) = x {
            let d = inner.domain_pair().to_vec();
            *x = ScaleKind::Linear(LinearScale::new_internal(
                d, vec![x_pixel_range.0, x_pixel_range.1], false, false,
            ));
        }
    }

    if let Some((lo, hi)) = y_domain_override {
        let pr = if expand {
            inset_pixel_range(y_pixel_range, resolve_padding_fraction(None, false))
        } else {
            y_pixel_range
        };
        *y = ScaleKind::Linear(LinearScale::new_internal(
            vec![lo, hi], vec![pr.1, pr.0], false, false,
        ));
    } else if !expand {
        if let ScaleKind::Linear(ref inner) = y {
            let d = inner.domain_pair().to_vec();
            *y = ScaleKind::Linear(LinearScale::new_internal(
                d, vec![y_pixel_range.1, y_pixel_range.0], false, false,
            ));
        }
    }
}

/// One axis's chart-level scale-domain configuration (`configure_axis(
/// domain_min=, domain_max=, nice=, zero=)` and its `axis_x`/`axis_y`/
/// `axis_y2` spellings), in the neutral value vocabulary this module speaks.
///
/// `ChartConfig` itself stays out of `scale_resolve` per the layering the
/// seam doc records (`seam.rs`: `render::mod` and `scene_build` call DOWN into
/// the scale engine, never the reverse), so the caller extracts this and the
/// engine never learns the config's shape.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(in crate::render) struct AxisDomainConfig {
    pub(in crate::render) min: Option<f64>,
    pub(in crate::render) max: Option<f64>,
    pub(in crate::render) nice: Option<bool>,
    pub(in crate::render) zero: Option<bool>,
}

impl AxisDomainConfig {
    /// Whether the caller asked for anything at all. `false` → applying this
    /// config is a guaranteed no-op, which is what keeps every chart that
    /// doesn't use these fields byte-identical.
    pub(in crate::render) fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    /// The field names the caller set, for the wrong-surface warning.
    fn named_fields(&self) -> Vec<String> {
        [
            ("domain_min", self.min.is_some()),
            ("domain_max", self.max.is_some()),
            ("nice", self.nice.is_some()),
            ("zero", self.zero.is_some()),
        ]
        .into_iter()
        .filter(|(_, set)| *set)
        .map(|(name, _)| name.to_string())
        .collect()
    }
}

/// Apply one axis's chart-level scale-domain config to its resolved scale
/// (D3, spec §4.2).
///
/// Precedence, per the decision doc: an encoding-level `scale=` domain wins
/// ENTIRELY and silently — it is the more specific surface, so the documented
/// cascade already answers the question and a warning would be noise. That is
/// why `encoding_scale_explicit` is a parameter rather than something this fn
/// infers: only the caller can see the encoding.
///
/// On an ordinal/band axis the four fields describe nothing that exists (there
/// are no numeric bounds to clamp, round, or extend to zero), so they are
/// refused loudly instead: [`RenderWarning::ScaleDomainConfigOnOrdinalAxis`],
/// naming the axis and the fields. Wrong surface, not a cascade loss.
///
/// Order of operations on a continuous axis: `zero` first (extend the extent
/// to include 0), then `nice` (round outward to human bounds), then the
/// explicit `min`/`max` clamps — so an explicit bound is never re-rounded away
/// by `nice`, and `zero` cannot re-widen past an explicit bound.
///
/// `nice` delegates to [`ScaleKind::niced_domain`], which dispatches to THIS
/// scale kind's own `nice()` rather than re-deriving a kind-independent
/// rounding here — a log axis nices in log space (power-of-`base` rounding)
/// and a time axis nices calendar-aware, matching exactly what the
/// encoding-level `Scale(nice=True)` surface would produce on the same
/// domain. Rounding every kind with the same linear `nice_step` (the
/// pre-fix behavior) could drive a log axis's bound to 0 — refused by every
/// log-scale constructor, so `configure_axis(nice=True)` on a log axis used
/// to raise `InvalidScaleDomainConfig` instead of rendering.
///
/// `zero` and `nice` are ONE-DIRECTIONAL by design, and the `false` spelling
/// is a deliberate no-op rather than an oversight: both name an opt-in
/// widening of an already-resolved domain, and there is nothing on the other
/// side for `false` to switch off. Nothing in the auto path forces either —
/// `build_axis_scale` constructs every auto positional scale with
/// `nice: false`, and no mark or theme injects a zero baseline into the
/// domain (a bar chart's y domain reaches 0 because the post-Stack VALUES
/// start at 0, which `zero=false` must not and does not erase). So
/// `zero=false`/`nice=false` request the default and get it. Pinned as
/// byte-identical no-ops in `tests/test_axis_config_plumbing.py`; stated in
/// the manifest reasons and on the Python `AxisConfig` fields.
pub(in crate::render) fn apply_axis_domain_config(
    scale: &mut ScaleKind,
    cfg: &AxisDomainConfig,
    channel: &'static str,
    encoding_scale_explicit: bool,
    warnings: &mut Vec<RenderWarning>,
) -> Result<(), RenderError> {
    if cfg.is_empty() || encoding_scale_explicit {
        return Ok(());
    }
    let Some((lo, hi)) = scale.data_domain() else {
        warnings.push(RenderWarning::ScaleDomainConfigOnOrdinalAxis {
            channel: channel.to_string(),
            fields: cfg.named_fields(),
        });
        return Ok(());
    };
    // A y axis carries its domain in display order (`[hi, lo]` for the
    // inverted pixel range is the RANGE, not the domain — the domain stays
    // ascending), but a reversed scale can still present `lo > hi`. Work in
    // ascending order and restore the caller's orientation at the end so a
    // reversed axis stays reversed.
    let reversed = lo > hi;
    let (mut d_lo, mut d_hi) = if reversed { (hi, lo) } else { (lo, hi) };
    if cfg.zero == Some(true) {
        d_lo = d_lo.min(0.0);
        d_hi = d_hi.max(0.0);
    }
    if cfg.nice == Some(true) {
        let (nlo, nhi) = scale.niced_domain(d_lo, d_hi);
        d_lo = nlo;
        d_hi = nhi;
    }
    if let Some(min) = cfg.min {
        d_lo = min;
    }
    if let Some(max) = cfg.max {
        d_hi = max;
    }
    // The computed domain must survive everything its own scale kind would
    // have refused at construction — a config-written domain is a USER-set
    // domain, so it meets the user-set contract, not the auto-inferred
    // fallback (spec §4.2). Two layers, both quoting
    // the constructors' own sentences so the words never drift:
    //
    //   1. KIND-INDEPENDENT: a degenerate pair, which every continuous
    //      constructor rejects via `core::validate_continuous_domain`.
    //      Checked on the COMPUTED result, so a zero-width domain produced by
    //      `zero`/`nice`/`min`/`max` in combination is caught too. Reversed
    //      (`lo > hi`) is NOT degenerate — an accepted reversed axis on the
    //      sibling surface too.
    //   2. KIND-SPECIFIC: whatever this scale kind constrains further, asked
    //      of the SCALE (`ScaleKind::validate_user_domain`) rather than
    //      re-derived here. That is what stops this site from growing a third
    //      hand-written check per kind, and what makes a future scale kind's
    //      own rule impossible to omit.
    if d_lo == d_hi {
        return Err(RenderError::InvalidScaleDomainConfig {
            channel: channel.to_string(),
            reason: crate::scale::core::DEGENERATE_DOMAIN_MESSAGE.to_string(),
        });
    }
    let (new_lo, new_hi) = if reversed { (d_hi, d_lo) } else { (d_lo, d_hi) };
    scale
        .validate_user_domain(new_lo, new_hi)
        .map_err(|reason| RenderError::InvalidScaleDomainConfig {
            channel: channel.to_string(),
            reason: reason.to_string(),
        })?;
    scale.set_data_domain(new_lo, new_hi);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{band_point_pixel_range, ordinal_pixel_range};
    use crate::render::{RenderError, RenderWarning};
    use crate::scale::discrete::{CategoricalPlacement, DiscreteLayout};
    use arrow::record_batch::RecordBatch;
    use crate::render::scale_resolve::{ScaleKind, SharedDomain};
    use crate::scale::linear::LinearScale;
    use crate::scale::log::LogScale;
    use crate::scale::pow::PowScale;
    use crate::scale::symlog::SymlogScale;
    use crate::scale::time::TimeScale;

    // ── Batch B design review S4 (2026-09-03): `ScaleKind::niced_domain` ────
    //
    // Reviewer-prescribed property test: `configure_axis(nice=True)`'s
    // dispatch point (`ScaleKind::niced_domain`) must land on exactly the
    // domain each continuous kind's OWN `nice=True` construction produces —
    // the encoding-level `Scale(nice=True)` surface — for all five
    // continuous kinds. The "direct" side below constructs via
    // `<Kind>Scale::new_internal(..., nice: true)`, the exact crate-internal
    // constructor the Python-facing `#[new]` delegates to for its own `nice`
    // handling (verified by reading each — `if nice { d = d.nice(); }`),
    // rather than the private `#[new]` itself, which isn't reachable from
    // this module. Before the fix this property held only for
    // Linear/Pow/Symlog (which happen to share the same `nice_step`
    // rounding), by coincidence rather than by design; Log and Time
    // genuinely diverged.

    /// Linear: count-10 `nice_step` rounding.
    #[test]
    fn niced_domain_matches_linear_scale_nice_true() {
        let domain = vec![3.0, 847.0];
        let range = vec![0.0, 500.0];
        let auto = ScaleKind::Linear(LinearScale::new_internal(
            domain.clone(), range.clone(), false, false,
        ));
        let direct = LinearScale::new_internal(domain.clone(), range, false, true);
        let [want_lo, want_hi] = direct.domain_pair();
        assert_eq!(auto.niced_domain(domain[0], domain[1]), (want_lo, want_hi));
    }

    /// Log: power-of-`base` rounding — the S4 bug's own kind. `(10, 1000)` is
    /// the reviewer-reproduced repro's domain shape (see
    /// `render::orchestration_tests::render_svg_log_axis_configure_axis_nice_true_renders_instead_of_refusing`),
    /// which the pre-fix `nice_step` rounding drove to a `0` lower bound.
    #[test]
    fn niced_domain_matches_log_scale_nice_true() {
        let domain = vec![10.0, 1000.0];
        let range = vec![0.0, 500.0];
        let auto = ScaleKind::Log(LogScale::new_internal(
            domain.clone(), range.clone(), 10.0, false, false,
        ));
        let direct = LogScale::new_internal(domain.clone(), range, 10.0, false, true);
        let [want_lo, want_hi] = direct.domain_pair();
        assert_eq!(auto.niced_domain(domain[0], domain[1]), (want_lo, want_hi));
        // And, concretely, neither bound is 0 — the shape of the bug this
        // property test exists to catch.
        assert_ne!(want_lo, 0.0);
    }

    /// Symlog: shares Linear's `nice_step` rounding on its (already-linear)
    /// domain representation.
    #[test]
    fn niced_domain_matches_symlog_scale_nice_true() {
        let domain = vec![3.0, 847.0];
        let range = vec![0.0, 500.0];
        let auto = ScaleKind::Symlog(SymlogScale::new_internal(
            domain.clone(), range.clone(), 1.0, false, false,
        ));
        let direct = SymlogScale::new_internal(domain.clone(), range, 1.0, false, true);
        let [want_lo, want_hi] = direct.domain_pair();
        assert_eq!(auto.niced_domain(domain[0], domain[1]), (want_lo, want_hi));
    }

    /// Pow: shares Linear's `nice_step` rounding on its raw (untransformed)
    /// domain (verified by reading `PowScaleData::nice`, which is
    /// structurally identical to `LinearScaleData::nice`). `PowScale::new_internal`
    /// has no `nice` parameter (production never constructs an auto Pow scale
    /// with `nice: true` — `build_axis_scale` always passes `nice: false`),
    /// so the "direct" side here hand-derives the expected bounds via the
    /// same shared `nice_step` primitive `PowScaleData::nice` calls, rather
    /// than a from-scratch scale construction.
    #[test]
    fn niced_domain_matches_pow_scale_nice_true() {
        let domain = vec![3.0, 847.0];
        let range = vec![0.0, 500.0];
        let auto = ScaleKind::Pow(PowScale::new_internal(domain.clone(), range, 0.5, false));
        let step = crate::scale::ticks::nice_step(domain[0], domain[1], 10);
        let want_lo = (domain[0] / step).floor() * step;
        let want_hi = (domain[1] / step).ceil() * step;
        assert_eq!(auto.niced_domain(domain[0], domain[1]), (want_lo, want_hi));
    }

    /// Time: calendar-aware rounding (month/year boundaries via `chrono`),
    /// not a raw epoch-ms `nice_step` round — the other divergent kind the
    /// design review named alongside Log.
    #[test]
    fn niced_domain_matches_time_scale_nice_true() {
        // ~400 days apart, in epoch-ms.
        let domain = vec![1_700_000_000_000.0, 1_734_560_000_000.0];
        let range = vec![0.0, 500.0];
        let auto = ScaleKind::Time(TimeScale::new_internal(
            domain.clone(), range.clone(), false, false,
        ));
        let direct = TimeScale::new_internal(domain.clone(), range, false, true);
        let [want_lo, want_hi] = direct.domain_pair();
        assert_eq!(auto.niced_domain(domain[0], domain[1]), (want_lo, want_hi));
    }

    /// Issue #39: an explicit two-entry range is honored verbatim, and marked
    /// explicit for band-geometry consumers.
    #[test]
    fn band_point_pixel_range_honors_explicit_range() {
        let range = vec![40.0, 260.0];
        assert_eq!(
            band_point_pixel_range(Some(&range), (0.0, 500.0)),
            (vec![40.0, 260.0], true)
        );
    }

    /// Issue #39: an absent range falls back to the panel extent, and is not
    /// marked explicit.
    #[test]
    fn band_point_pixel_range_falls_back_when_absent() {
        assert_eq!(band_point_pixel_range(None, (0.0, 500.0)), (vec![0.0, 500.0], false));
    }

    /// Issue #39: a degenerate (fewer than 2 entries) range falls back to the
    /// panel extent rather than panicking on out-of-bounds indexing, and is
    /// not marked explicit.
    #[test]
    fn band_point_pixel_range_falls_back_when_too_short() {
        let range = vec![40.0];
        assert_eq!(
            band_point_pixel_range(Some(&range), (0.0, 500.0)),
            (vec![0.0, 500.0], false)
        );

        let empty: Vec<f64> = vec![];
        assert_eq!(
            band_point_pixel_range(Some(&empty), (0.0, 500.0)),
            (vec![0.0, 500.0], false)
        );
    }

    // ── explicitness contract (GH #39 phase 2, band-geometry unification) ──

    /// A Band scale built with an explicit two-entry range lays its bands out
    /// over that range: two 110px slots across `[40, 260]`.
    #[test]
    fn band_scale_explicit_range_places_bands_over_that_range() {
        let (range, explicit) = band_point_pixel_range(Some(&[40.0, 260.0]), (0.0, 500.0));
        assert!(explicit);
        let scale = ScaleKind::Ordinal(
            crate::scale::ordinal::OrdinalScale::new_internal(
                vec!["a".into(), "b".into()],
                range,
                DiscreteLayout::UNPADDED,
            )
            .with_explicit_range(explicit),
        );
        assert_eq!(
            scale.categorical_placement(),
            Some(CategoricalPlacement::Absolute(vec![95.0, 205.0]))
        );
        assert_eq!(scale.bandwidth(), Some(110.0));
    }

    /// A reversed explicit range descends: the same two bands in the opposite
    /// pixel order, with a still-positive bandwidth.
    #[test]
    fn band_scale_explicit_reversed_range_descends() {
        let (range, explicit) = band_point_pixel_range(Some(&[260.0, 40.0]), (0.0, 500.0));
        assert!(explicit);
        let scale = ScaleKind::Ordinal(
            crate::scale::ordinal::OrdinalScale::new_internal(
                vec!["a".into(), "b".into()],
                range,
                DiscreteLayout::UNPADDED,
            )
            .with_explicit_range(explicit),
        );
        assert_eq!(
            scale.categorical_placement(),
            Some(CategoricalPlacement::Absolute(vec![205.0, 95.0]))
        );
        assert_eq!(scale.bandwidth(), Some(110.0));
    }

    /// A Band scale falling back to the panel extent carries the MODEL rather
    /// than absolute pixels, even though its range is numerically identical to
    /// what an explicit range spanning the same pixels would be — explicitness
    /// is recorded at construction, not inferred from the numbers, and the
    /// fallback's pixels are a pre-layout placeholder that must not be trusted.
    #[test]
    fn band_scale_fallback_range_carries_the_model_not_pixels() {
        let (range, explicit) = band_point_pixel_range(None, (0.0, 500.0));
        assert!(!explicit);
        let scale = ScaleKind::Ordinal(
            crate::scale::ordinal::OrdinalScale::new_internal(
                vec!["a".into(), "b".into()],
                range,
                DiscreteLayout::UNPADDED,
            )
            .with_explicit_range(explicit),
        );
        assert_eq!(
            scale.categorical_placement(),
            Some(CategoricalPlacement::PanelExtent {
                layout: DiscreteLayout::UNPADDED,
                categories: 2
            })
        );
        // The band geometry itself is unaffected by provenance: only which
        // FORM the placement takes is.
        assert_eq!(scale.bandwidth(), Some(250.0));
    }

    /// A positional Ordinal scale with an explicit >= 2-entry numeric range
    /// behaves identically to Band/Point (no special-casing).
    #[test]
    fn ordinal_positional_explicit_range_places_bands_over_that_range() {
        let range_values = vec![
            crate::scale::ordinal::OrdinalRangeValue::Number(10.0),
            crate::scale::ordinal::OrdinalRangeValue::Number(210.0),
        ];
        let (range, explicit) = ordinal_pixel_range(Some(&range_values), (0.0, 500.0));
        assert!(explicit);
        let scale = ScaleKind::Ordinal(
            crate::scale::ordinal::OrdinalScale::new_internal(
                vec!["a".into(), "b".into()],
                range,
                DiscreteLayout::UNPADDED,
            )
            .with_explicit_range(explicit),
        );
        assert_eq!(
            scale.categorical_placement(),
            Some(CategoricalPlacement::Absolute(vec![60.0, 160.0]))
        );
        assert_eq!(scale.bandwidth(), Some(100.0));
    }

    /// A non-ordinal (Linear) scale reports `None` from both band accessors —
    /// they are gated to ordinal positional scales only, continuous axes
    /// placing their ticks through `tick_fractions` instead.
    #[test]
    fn linear_scale_never_reports_band_geometry() {
        let scale = ScaleKind::Linear(crate::scale::linear::LinearScale::new_internal(
            vec![0.0, 100.0],
            vec![40.0, 260.0],
            false,
            false,
        ));
        assert_eq!(scale.categorical_placement(), None);
        assert_eq!(scale.bandwidth(), None);
    }

    // ── PointScale(reverse=True) domain reversal (GH #65) ──

    /// Resolve `spec` for a one-column string batch of `cats` through the real
    /// `build_from_scale_spec` arm — the same seam `build_axis_scale` calls
    /// when an encoding carries an explicit `scale`. Panel extent `(0, 500)`.
    fn resolve_scale_spec_for_cats(
        spec: &crate::spec::encoding::ScaleSpec,
        cats: &[String],
    ) -> ScaleKind {
        let batch = category_batch("cat", cats);

        let enc = crate::spec::encoding::EncodingSpec {
            field: "cat".into(),
            type_: None,
            ..Default::default()
        };
        let sort_ctx = super::SortContext {
            category_field: "cat",
            batch: &batch,
            x_field: None,
            y_field: None,
            coord_flipped: false,
        };
        let mut warnings = Vec::new();
        super::build_from_scale_spec("x", spec, &enc, &batch, (0.0, 500.0), &sort_ctx, &|| super::column_extent(&batch, "cat"), &mut warnings)
            .unwrap()
    }

    /// A zero-padding `ScaleSpec::Point` over `explicit_domain`, resolved.
    fn resolve_point_scale(
        explicit_domain: Vec<String>,
        range: Option<Vec<f64>>,
        reverse: bool,
    ) -> ScaleKind {
        let spec = crate::spec::encoding::ScaleSpec::Point {
            domain: Some(explicit_domain.clone()),
            padding: 0.0,
            align: 0.5,
            reverse,
            range,
        };
        resolve_scale_spec_for_cats(&spec, &explicit_domain)
    }

    /// GH #65: `reverse=True` reverses the resolved domain order, so the
    /// *first* category of the original domain lands on the *last* point
    /// position (and vice versa) — a domain-vector reversal, not a pixel-range
    /// flip.
    ///
    /// The positions are the d3 point formula's (F-L04-03): these fixtures set
    /// `padding = 0.0`, which puts the outer categories exactly on the range
    /// endpoints. Before F-L04-03 the resolver placed point scales on the band
    /// model's `(i + 0.5)·step` centers (62.5 / 187.5 / 312.5 / 437.5 here) and
    /// `padding` moved nothing.
    #[test]
    fn point_scale_reverse_true_reverses_domain_order() {
        let domain = vec!["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string()];
        let scale = resolve_point_scale(domain, None, true);
        let ScaleKind::Ordinal(ordinal) = scale else { panic!("expected Ordinal scale") };

        // Panel-extent range (0.0, 500.0), 4 categories, padding 0 → step =
        // 500/3, positions 0 / 166.6… / 333.3… / 500. Non-reversed, "a" would
        // sit on the first (0.0); reversed, it sits on the last (500.0).
        assert_eq!(ordinal.scale_internal("a"), Some(500.0));
        assert_eq!(ordinal.scale_internal("d"), Some(0.0));
    }

    /// GH #65: `reverse=True` composes with an explicit pixel range — with
    /// `padding = 0.0` over `[40, 260]` the original domain `[a, b, c, d]`
    /// mirrors onto `[260, 186.6…, 113.3…, 40]`.
    #[test]
    fn point_scale_reverse_true_composes_with_explicit_range() {
        let domain = vec!["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string()];
        let scale = resolve_point_scale(domain, Some(vec![40.0, 260.0]), true);
        let ScaleKind::Ordinal(ordinal) = scale else { panic!("expected Ordinal scale") };

        let step = 220.0 / 3.0;
        for (cat, expected) in
            [("a", 260.0), ("b", 260.0 - step), ("c", 40.0 + step), ("d", 40.0)]
        {
            let got = ordinal.scale_internal(cat).expect("category in domain");
            assert!((got - expected).abs() < 1e-9, "{cat}: {got} vs {expected}");
        }
    }

    /// GH #65 regression: `reverse=False` (the pre-fix default) preserves
    /// domain order — positions ascend in original domain order, and with
    /// `padding = 0.0` the first and last sit on the range endpoints.
    #[test]
    fn point_scale_reverse_false_preserves_domain_order() {
        let domain = vec!["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string()];
        let scale = resolve_point_scale(domain, Some(vec![40.0, 260.0]), false);
        let ScaleKind::Ordinal(ordinal) = scale else { panic!("expected Ordinal scale") };

        let step = 220.0 / 3.0;
        for (cat, expected) in
            [("a", 40.0), ("b", 40.0 + step), ("c", 260.0 - step), ("d", 260.0)]
        {
            let got = ordinal.scale_internal(cat).expect("category in domain");
            assert!((got - expected).abs() < 1e-9, "{cat}: {got} vs {expected}");
        }
    }

    /// The d3 point default (`padding = 0.5`, what `fr.PointScale()` emits)
    /// reproduces the pre-F-L04-03 symmetric band centers exactly — the
    /// coincidence that keeps every default point-scale render byte-identical
    /// while `padding` becomes real. Discriminating against the `padding = 0`
    /// case above, which lands on the endpoints instead.
    #[test]
    fn point_scale_default_padding_matches_former_band_centers() {
        use crate::spec::encoding::ScaleSpec;
        let domain = vec!["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string()];
        let spec = ScaleSpec::Point {
            domain: Some(domain.clone()),
            padding: 0.5,
            align: 0.5,
            reverse: false,
            range: Some(vec![40.0, 260.0]),
        };
        let scale = resolve_scale_spec_for_cats(&spec, &domain);
        let ScaleKind::Ordinal(ordinal) = scale else { panic!("expected Ordinal scale") };

        assert_eq!(ordinal.scale_internal("a"), Some(67.5));
        assert_eq!(ordinal.scale_internal("b"), Some(122.5));
        assert_eq!(ordinal.scale_internal("c"), Some(177.5));
        assert_eq!(ordinal.scale_internal("d"), Some(232.5));
    }

    // ── ScaleSpec::Band → d3 layout threading (F-L04-03) ────────────────────

    /// Resolve a `ScaleSpec::Band` over `[a, b, c, d]` on `[40, 260]` with the
    /// given padding trio, and report its centers and bandwidth.
    fn band_geometry(
        padding: f64,
        padding_inner: Option<f64>,
        padding_outer: Option<f64>,
        align: f64,
    ) -> (Vec<f64>, f64) {
        let cats: Vec<String> = ["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect();
        let spec = crate::spec::encoding::ScaleSpec::Band {
            domain: Some(cats.clone()),
            padding,
            padding_inner,
            padding_outer,
            align,
            range: Some(vec![40.0, 260.0]),
        };
        let ScaleKind::Ordinal(ordinal) = resolve_scale_spec_for_cats(&spec, &cats) else {
            panic!("expected Ordinal scale")
        };
        (
            cats.iter().map(|c| ordinal.scale_internal(c).expect("in domain")).collect(),
            ordinal.bandwidth(),
        )
    }

    /// A zero-padding band spec resolves to the former symmetric geometry
    /// exactly — the reduction every default golden depends on.
    #[test]
    fn band_spec_zero_padding_keeps_symmetric_centers() {
        let (centers, bandwidth) = band_geometry(0.0, None, None, 0.5);
        assert_eq!(centers, vec![67.5, 122.5, 177.5, 232.5]);
        assert_eq!(bandwidth, 55.0);
    }

    /// d3's `padding` shorthand sets BOTH sides: `padding = 0.2` must resolve
    /// identically to `padding_inner = padding_outer = 0.2`, and differently
    /// from setting only one side (F-L04-03's mapping contract).
    #[test]
    fn band_spec_padding_shorthand_sets_inner_and_outer() {
        let shorthand = band_geometry(0.2, None, None, 0.5);
        let both_explicit = band_geometry(0.0, Some(0.2), Some(0.2), 0.5);
        assert_eq!(shorthand, both_explicit, "padding must set both sides");

        let inner_only = band_geometry(0.0, Some(0.2), None, 0.5);
        assert_ne!(
            shorthand, inner_only,
            "padding_inner alone must not reproduce the shorthand's outer padding"
        );
        // Explicit sides override the shorthand independently.
        assert_eq!(band_geometry(0.2, Some(0.0), None, 0.5), band_geometry(0.0, None, Some(0.2), 0.5));
    }

    /// Hand-computed d3 oracle through the resolver: `[a, b, c, d]` on
    /// `[40, 260]` with `padding_inner = 0.2`, `padding_outer = 0.3`.
    /// denom = 4 − 0.2 + 0.6 = 4.4 → step = 50, bandwidth = 40,
    /// start = 40 + 0.3·50 = 55 → band middles 75 / 125 / 175 / 225.
    #[test]
    fn band_spec_padding_moves_geometry_to_d3_values() {
        let (centers, bandwidth) = band_geometry(0.0, Some(0.2), Some(0.3), 0.5);
        assert!((bandwidth - 40.0).abs() < 1e-9, "bandwidth {bandwidth}");
        for (got, want) in centers.iter().zip([75.0, 125.0, 175.0, 225.0]) {
            assert!((got - want).abs() < 1e-9, "center {got} vs {want} in {centers:?}");
        }

        // Discriminating companion: `padding_inner = 0.2, padding_outer = 0.1`
        // leaves the denominator at 4.0, so the step and every band middle are
        // the unpadded ones — inner padding narrows the band without moving it.
        let (unpadded_centers, unpadded_bandwidth) = band_geometry(0.0, None, None, 0.5);
        let (narrowed_centers, narrowed_bandwidth) = band_geometry(0.0, Some(0.2), Some(0.1), 0.5);
        assert_eq!(narrowed_centers, unpadded_centers);
        assert_eq!(unpadded_bandwidth, 55.0);
        assert_eq!(narrowed_bandwidth, 44.0);
    }

    /// `align` reaches the resolved scale: it is inert while the denominator
    /// clamp leaves no leftover, and moves the lone band when it does (n=1,
    /// `padding_inner = 0.5` → denom 0.5, clamped to 1).
    #[test]
    fn band_spec_align_threads_through_to_the_clamped_case() {
        let cats = vec!["only".to_string()];
        let center = |align: f64| {
            let spec = crate::spec::encoding::ScaleSpec::Band {
                domain: Some(cats.clone()),
                padding: 0.0,
                padding_inner: Some(0.5),
                padding_outer: Some(0.0),
                align,
                range: Some(vec![0.0, 100.0]),
            };
            let ScaleKind::Ordinal(ordinal) = resolve_scale_spec_for_cats(&spec, &cats) else {
                panic!("expected Ordinal scale")
            };
            ordinal.scale_internal("only").expect("in domain")
        };
        // step = 100 (clamped), leftover = 50, bandwidth = 50, and the band's
        // middle is `start + bandwidth/2`, so align sweeps it across the
        // leftover without ever leaving [0, 100].
        assert_eq!(center(0.0), 25.0);
        assert_eq!(center(0.5), 50.0);
        assert_eq!(center(1.0), 75.0);
    }

    // ── explicitness gates: degenerate inputs (ported from
    // tests/bug_hunt_band_point_range.rs, R1) ────────────────────────────────

    /// A wire range with MORE than 2 entries truncates to the first two and
    /// still counts as explicit — matching the pyclass constructor's
    /// `[v[0], v[1]]`.
    #[test]
    fn band_point_gate_truncates_extra_entries_still_explicit() {
        let (r, explicit) = band_point_pixel_range(Some(&[40.0, 260.0, 500.0]), (0.0, 600.0));
        assert_eq!(r, vec![40.0, 260.0]);
        assert!(explicit);
    }

    /// Documents the validation gap the gate does NOT close: non-finite range
    /// entries pass straight through, still marked explicit. The
    /// constructor-level gap (no `validate_finite` for Band/Point ranges) is
    /// proven end-to-end by the Python tests; this pins that the resolver
    /// gate is not the layer catching it.
    #[test]
    fn band_point_gate_passes_non_finite_entries_through() {
        let (r, explicit) = band_point_pixel_range(Some(&[f64::NAN, 260.0]), (0.0, 600.0));
        assert!(r[0].is_nan(), "gate does not filter NaN (documented gap)");
        assert_eq!(r[1], 260.0);
        assert!(explicit, "a NaN-carrying 2-entry range is still treated as explicit");
    }

    /// Ordinal gate: a single NUMERIC entry is passed through unchanged but
    /// NOT marked explicit (an explicit pixel range needs 2 entries) — the
    /// documented asymmetry vs `band_point_pixel_range`, which falls back to
    /// the panel extent.
    #[test]
    fn ordinal_gate_single_number_passthrough_not_explicit() {
        use crate::scale::ordinal::OrdinalRangeValue;
        let (r, explicit) = ordinal_pixel_range(Some(&[OrdinalRangeValue::Number(40.0)]), (0.0, 600.0));
        assert_eq!(r, vec![40.0], "single numeric entry passes through unchanged");
        assert!(!explicit, "1-entry range must not count as explicit");
    }

    /// Ordinal gate: an all-string (color) range has no numbers → the panel
    /// extent fallback, not explicit.
    #[test]
    fn ordinal_gate_all_string_range_falls_back() {
        use crate::scale::ordinal::OrdinalRangeValue;
        let (r, explicit) = ordinal_pixel_range(
            Some(&[OrdinalRangeValue::Str("#ccc".into()), OrdinalRangeValue::Str("#e4572e".into())]),
            (0.0, 600.0),
        );
        assert_eq!(r, vec![0.0, 600.0]);
        assert!(!explicit);
    }

    /// Ordinal gate: mixed [num, str, num] extracts both numbers and IS
    /// explicit — the seam the Python mixed-range render test exercises
    /// end-to-end.
    #[test]
    fn ordinal_gate_mixed_range_extracts_numbers_explicit() {
        use crate::scale::ordinal::OrdinalRangeValue;
        let (r, explicit) = ordinal_pixel_range(
            Some(&[OrdinalRangeValue::Number(40.0), OrdinalRangeValue::Str("#ccc".into()), OrdinalRangeValue::Number(260.0)]),
            (0.0, 600.0),
        );
        assert_eq!(r, vec![40.0, 260.0]);
        assert!(explicit);
    }

    /// Ordinal gate: an empty range falls back to the panel extent, never
    /// explicit, never panicking on indexing. (`band_point_pixel_range`'s
    /// empty-range fallback is already pinned by
    /// `band_point_pixel_range_falls_back_when_too_short`.)
    #[test]
    fn ordinal_gate_empty_range_falls_back_no_panic() {
        let empty_vals: Vec<crate::scale::ordinal::OrdinalRangeValue> = Vec::new();
        assert_eq!(ordinal_pixel_range(Some(&empty_vals), (5.0, 95.0)), (vec![5.0, 95.0], false));
    }

    // ── Continuous `reverse` domain-swap (F-L04-07, batch-C task 1) ──
    //
    // `reverse` on `ContinuousScaleCommon` is domain-swap sugar (spec §4C):
    // after domain resolution, `reverse=true` swaps the resolved domain
    // pair, exactly equivalent to writing `domain=[hi, lo]` by hand. Every
    // test below resolves through the real `build_from_scale_spec` arm —
    // the same seam `build_axis_scale` calls for an encoding's explicit
    // `scale` — mirroring the `PointScale(reverse=True)` tests above rather
    // than hand-sequencing the swap.

    fn numeric_batch(field: &str, values: Vec<f64>) -> RecordBatch {
        use arrow::array::Float64Array;
        use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![Field::new(field, ArrowDataType::Float64, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(values))]).unwrap()
    }

    /// A one-column Utf8 batch, mirroring `resolve_scale_spec_for_cats`'s
    /// inline fixture but as a standalone helper for tests (below) that call
    /// `build_axis_scale` directly rather than `build_from_scale_spec`.
    fn category_batch(field: &str, cats: &[String]) -> RecordBatch {
        use arrow::array::StringArray;
        use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![Field::new(field, ArrowDataType::Utf8, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(cats.to_vec()))]).unwrap()
    }

    /// Resolve `spec` through the real `build_from_scale_spec` arm without
    /// unwrapping — the shape `resolve_scale` below wants for its happy-path
    /// callers, and the #152 shape-refusal tests want for their error-path
    /// ones, so the enc/sort_ctx plumbing isn't hand-duplicated between them.
    fn try_resolve_scale(
        spec: &crate::spec::encoding::ScaleSpec,
        field: &str,
        batch: &RecordBatch,
        pr: (f64, f64),
    ) -> Result<ScaleKind, RenderError> {
        let enc = crate::spec::encoding::EncodingSpec {
            field: field.into(),
            type_: None,
            scale: Some(spec.clone()),
            ..Default::default()
        };
        let sort_ctx = super::SortContext {
            category_field: field,
            batch,
            x_field: None,
            y_field: None,
            coord_flipped: false,
        };
        let mut warnings = Vec::new();
        super::build_from_scale_spec("x", spec, &enc, batch, pr, &sort_ctx, &|| super::column_extent(batch, field), &mut warnings)
    }

    fn resolve_scale(
        spec: &crate::spec::encoding::ScaleSpec,
        field: &str,
        batch: &RecordBatch,
        pr: (f64, f64),
    ) -> ScaleKind {
        try_resolve_scale(spec, field, batch, pr).expect("resolve_scale: expected Ok in this test")
    }

    /// A `ContinuousScaleCommon` with everything but `domain`/`range`/`reverse`
    /// at its wire default — the shape every one of these tests varies along
    /// exactly one axis at a time.
    fn reverse_common(
        domain: Option<Vec<f64>>,
        range: Option<Vec<f64>>,
        reverse: bool,
    ) -> crate::spec::encoding::ContinuousScaleCommon {
        crate::spec::encoding::ContinuousScaleCommon {
            domain,
            range,
            clamp: false,
            padding: None,
            scheme: None,
            domain_param: None,
            reverse,
        }
    }

    // ── #152: explicit continuous domain/range LENGTH refusal ──
    //
    // `scale={'type': 'linear', 'domain': [5.0]}` is a wire shape the
    // raw-dict key gate (`test_scale_dict_gate.py`) accepts — a legal key
    // with a malformed VALUE — and used to reach `LinearScale::new_internal`'s
    // `domain[1]` indexing unchecked, panicking `index out of bounds: the
    // len is 1 but the index is 1` at `scale/linear.rs:134` across the PyO3
    // boundary instead of raising. `resolve_scale`/`try_resolve_scale` above
    // already resolve through the real `build_from_scale_spec` arm — the
    // class pyclass spelling (`LinearScale(domain=[5.0])`) and the raw-dict
    // spelling (`{'type': 'linear', 'domain': [5.0]}`) both deserialize to
    // this identical `ScaleSpec::Linear` shape before either ever reaches
    // Rust, so resolving THIS shape covers both Python spellings at once —
    // `tests/test_scales.py`'s new raw-dict matrix is the render-boundary
    // (PyO3-crossing) proof for the class-side equivalent already pinned in
    // that file's constructor tests.
    //
    // Scoped to LENGTH only, not degeneracy — see
    // `validate_explicit_continuous_shape`'s doc comment for why a
    // degenerate `[c, c]` domain must NOT also refuse here: the reactive
    // `domainParam` substitution path (D6) writes its current value into
    // this identical `common.domain` field with no surviving marker, and
    // `tests/test_bug_hunt_reactive_params.py`'s
    // `TestDomainParamDegenerate::test_collapsed_domain_no_nan` already pins
    // that a degenerate domainParam value must keep rendering gracefully.

    /// Isolated regression test for the exact reported shape, ahead of the
    /// full 7-kind matrix below — a single-element `domain` on `linear`
    /// refuses typed, byte-pinned, instead of panicking.
    #[test]
    fn linear_single_element_domain_refuses_typed_not_panics() {
        use crate::spec::encoding::ScaleSpec;
        let batch = numeric_batch("v", vec![0.0, 100.0]);
        let spec = ScaleSpec::Linear {
            common: reverse_common(Some(vec![5.0]), None, false),
            nice: false,
            zero: false,
            system_injected: false,
        };
        let err = try_resolve_scale(&spec, "v", &batch, (0.0, 500.0)).unwrap_err();
        assert!(
            matches!(err, RenderError::InvalidScaleDomainConfig { .. }),
            "expected InvalidScaleDomainConfig, got {err:?}"
        );
        assert_eq!(
            err.to_string(),
            "x axis scale-domain config: domain must have length 2; got 1"
        );
    }

    /// One `ScaleSpec` builder per continuous kind, keyed by its wire `type`
    /// tag — lets the matrix test below drive all 7 kinds through one body
    /// instead of seven hand-duplicated copies (mirroring how the file's own
    /// `reverse`-sugar tests are one-per-kind but share `reverse_common` and
    /// `resolve_scale`).
    fn continuous_kind_builders() -> Vec<(
        &'static str,
        fn(Option<Vec<f64>>, Option<Vec<f64>>) -> crate::spec::encoding::ScaleSpec,
    )> {
        use crate::spec::encoding::ScaleSpec;
        vec![
            ("linear", |d, r| ScaleSpec::Linear { common: reverse_common(d, r, false), nice: false, zero: false, system_injected: false }),
            ("log", |d, r| ScaleSpec::Log { base: 10.0, common: reverse_common(d, r, false), nice: false }),
            ("time", |d, r| ScaleSpec::Time { common: reverse_common(d, r, false), nice: false }),
            ("symlog", |d, r| ScaleSpec::Symlog { constant: 1.0, common: reverse_common(d, r, false), nice: false }),
            ("pow", |d, r| ScaleSpec::Pow { exponent: 2.0, common: reverse_common(d, r, false) }),
            ("sqrt", |d, r| ScaleSpec::Sqrt { common: reverse_common(d, r, false) }),
            ("utc", |d, r| ScaleSpec::Utc { common: reverse_common(d, r, false), nice: false }),
        ]
    }

    /// A domain valid for `kind` — `log` needs a strictly-positive pair
    /// (its own sign/zero constructor rule, unrelated to #152), every other
    /// kind tolerates the plain `[0.0, 100.0]` default.
    fn valid_domain_for(kind: &str) -> Vec<f64> {
        if kind == "log" { vec![1.0, 100.0] } else { vec![0.0, 100.0] }
    }

    /// The full #152 acceptance matrix: all 7 `ContinuousScaleCommon` kinds
    /// x (domain, range) x (length != 2, degenerate), resolved through the
    /// real `build_from_scale_spec` arm. Both spellings converge to this
    /// `ScaleSpec` shape (see the section doc above), so this is the
    /// authoritative coverage for the shape-refusal logic itself; the Python
    /// suite adds the render-boundary (PyO3-crossing) raw-dict proof on top.
    #[test]
    fn explicit_continuous_domain_range_shape_matrix() {
        let batch = numeric_batch("v", vec![0.0, 100.0]);
        let pr = (0.0, 500.0);

        for (kind, build) in continuous_kind_builders() {
            // (1) domain too short refuses, naming the length-2 message.
            let spec = build(Some(vec![5.0]), None);
            let err = try_resolve_scale(&spec, "v", &batch, pr).unwrap_err();
            assert!(
                err.to_string().contains("domain must have length 2; got 1"),
                "{kind}: 1-element domain must refuse with the length message, got {err}"
            );

            // (2) domain too long also refuses (not just too-short).
            let spec = build(Some(vec![1.0, 2.0, 3.0]), None);
            let err = try_resolve_scale(&spec, "v", &batch, pr).unwrap_err();
            assert!(
                err.to_string().contains("domain must have length 2; got 3"),
                "{kind}: 3-element domain must refuse too, got {err}"
            );

            // (3) a degenerate `[c, c]` domain is NOT refused (deliberately
            // narrower than the pyclass constructor's own
            // `validate_continuous_domain`): see
            // `validate_explicit_continuous_shape`'s doc for why — the
            // reactive `domainParam` substitution path shares this exact
            // field and must keep rendering a transiently-degenerate value.
            let spec = build(Some(vec![5.0, 5.0]), None);
            assert!(
                try_resolve_scale(&spec, "v", &batch, pr).is_ok(),
                "{kind}: degenerate domain must still render (domainParam contract), got {:?}",
                try_resolve_scale(&spec, "v", &batch, pr).err()
            );

            // (4) range too short refuses (domain omitted -> auto-inferred).
            let spec = build(None, Some(vec![5.0]));
            let err = try_resolve_scale(&spec, "v", &batch, pr).unwrap_err();
            assert!(
                err.to_string().contains("range must have length 2; got 1"),
                "{kind}: 1-element range must refuse with the length message, got {err}"
            );

            // (5) a degenerate `[c, c]` RANGE is NOT refused either: unlike
            // domain, `validate_continuous_range` never checked degeneracy
            // even at the pyclass constructor, and #152 only closes the
            // length-panic gap.
            let spec = build(Some(valid_domain_for(kind)), Some(vec![5.0, 5.0]));
            assert!(
                try_resolve_scale(&spec, "v", &batch, pr).is_ok(),
                "{kind}: degenerate range must still render, matching the class spelling"
            );
        }
    }

    /// The "continuous-degrading" family (`Sequential`/`Diverging`/
    /// `Quantize`/etc.) is explicitly OUT of #152's scope (see the doc
    /// comment on that match arm in `build_from_scale_spec`): its domain
    /// shape is normalized to exactly 2 elements (or `None`) by
    /// `ScaleSpec::positional_extent()` before it ever reaches the funnel,
    /// so the length refusal above cannot apply to it, and a degenerate
    /// extent must keep rendering exactly as before this task (a Sequential
    /// color scale used positionally is a corner case, not one #152's
    /// frozen report names). Pinned so a future refactor that folds this
    /// family through `validate_explicit_continuous_shape` too must
    /// consciously re-decide that scope question, not drift into it.
    #[test]
    fn continuous_degrading_family_degenerate_extent_still_renders() {
        use crate::spec::encoding::ScaleSpec;
        let batch = numeric_batch("v", vec![0.0, 100.0]);
        let spec = ScaleSpec::Sequential {
            domain: Some(vec![5.0, 5.0]),
            scheme: None,
            reverse: false,
            stops: None,
        };
        assert!(
            try_resolve_scale(&spec, "v", &batch, (0.0, 500.0)).is_ok(),
            "the degrading family's own degenerate extent is out of #152's scope"
        );
    }

    /// F-L04-07: `reverse=true` on an auto-inferred domain swaps the
    /// resolved `[min, max]` pair to `[max, min]` — one test per continuous
    /// kind, so a future kind that forgets to wire `apply_domain_reverse`
    /// into its arm fails here instead of silently rendering ascending.
    #[test]
    fn linear_reverse_true_swaps_auto_inferred_domain() {
        use crate::spec::encoding::ScaleSpec;
        let batch = numeric_batch("v", vec![0.0, 100.0]);
        let spec = ScaleSpec::Linear { common: reverse_common(None, None, true), nice: false, zero: false, system_injected: false };
        let scale = resolve_scale(&spec, "v", &batch, (0.0, 500.0));
        assert_eq!(scale.data_domain(), Some((100.0, 0.0)));
    }

    #[test]
    fn log_reverse_true_swaps_auto_inferred_domain() {
        use crate::spec::encoding::ScaleSpec;
        let batch = numeric_batch("v", vec![1.0, 1000.0]);
        let spec = ScaleSpec::Log { base: 10.0, common: reverse_common(None, None, true), nice: false };
        let scale = resolve_scale(&spec, "v", &batch, (0.0, 500.0));
        assert_eq!(scale.data_domain(), Some((1000.0, 1.0)));
    }

    #[test]
    fn time_reverse_true_swaps_auto_inferred_domain() {
        use crate::spec::encoding::ScaleSpec;
        let batch = numeric_batch("v", vec![0.0, 86_400_000.0]);
        let spec = ScaleSpec::Time { common: reverse_common(None, None, true), nice: false };
        let scale = resolve_scale(&spec, "v", &batch, (0.0, 500.0));
        assert_eq!(scale.data_domain(), Some((86_400_000.0, 0.0)));
    }

    #[test]
    fn symlog_reverse_true_swaps_auto_inferred_domain() {
        use crate::spec::encoding::ScaleSpec;
        let batch = numeric_batch("v", vec![-100.0, 100.0]);
        let spec = ScaleSpec::Symlog { constant: 1.0, common: reverse_common(None, None, true), nice: false };
        let scale = resolve_scale(&spec, "v", &batch, (0.0, 500.0));
        assert_eq!(scale.data_domain(), Some((100.0, -100.0)));
    }

    #[test]
    fn pow_reverse_true_swaps_auto_inferred_domain() {
        use crate::spec::encoding::ScaleSpec;
        let batch = numeric_batch("v", vec![0.0, 100.0]);
        let spec = ScaleSpec::Pow { exponent: 2.0, common: reverse_common(None, None, true) };
        let scale = resolve_scale(&spec, "v", &batch, (0.0, 500.0));
        assert_eq!(scale.data_domain(), Some((100.0, 0.0)));
    }

    #[test]
    fn sqrt_reverse_true_swaps_auto_inferred_domain() {
        use crate::spec::encoding::ScaleSpec;
        let batch = numeric_batch("v", vec![0.0, 100.0]);
        let spec = ScaleSpec::Sqrt { common: reverse_common(None, None, true) };
        let scale = resolve_scale(&spec, "v", &batch, (0.0, 500.0));
        assert_eq!(scale.data_domain(), Some((100.0, 0.0)));
    }

    #[test]
    fn utc_reverse_true_swaps_auto_inferred_domain() {
        use crate::spec::encoding::ScaleSpec;
        let batch = numeric_batch("v", vec![0.0, 86_400_000.0]);
        let spec = ScaleSpec::Utc { common: reverse_common(None, None, true), nice: false };
        let scale = resolve_scale(&spec, "v", &batch, (0.0, 500.0));
        assert_eq!(scale.data_domain(), Some((86_400_000.0, 0.0)));
    }

    /// F-L04-06 (batch-C task 3): the `Utc` wire tag survives resolution.
    /// `new_internal` used to hardcode `utc: false` regardless of which
    /// `ScaleSpec` variant resolved it, so a wire round-trip through
    /// `ScaleSpec::Utc` silently lost the tag on the resolved `TimeScale`.
    /// `ScaleSpec::Time` and `ScaleSpec::Utc` must now resolve to
    /// distinguishable `utc()` values through the SAME resolver arm every
    /// encoding's explicit `scale=` goes through.
    #[test]
    fn utc_and_time_scale_spec_resolve_to_distinct_utc_tag() {
        use crate::spec::encoding::ScaleSpec;
        let batch = numeric_batch("v", vec![0.0, 86_400_000.0]);

        let utc_spec = ScaleSpec::Utc { common: reverse_common(None, None, false), nice: false };
        let ScaleKind::Time(utc_scale) = resolve_scale(&utc_spec, "v", &batch, (0.0, 500.0)) else {
            panic!("expected ScaleKind::Time");
        };
        assert!(utc_scale.utc_flag(), "ScaleSpec::Utc must resolve with utc=true");

        let time_spec = ScaleSpec::Time { common: reverse_common(None, None, false), nice: false };
        let ScaleKind::Time(time_scale) = resolve_scale(&time_spec, "v", &batch, (0.0, 500.0)) else {
            panic!("expected ScaleKind::Time");
        };
        assert!(!time_scale.utc_flag(), "ScaleSpec::Time must resolve with utc=false");
    }

    /// `reverse=false` (the wire default) leaves every kind's domain in
    /// ascending order — the regression guard alongside the seven flips
    /// above.
    #[test]
    fn linear_reverse_false_preserves_auto_inferred_domain() {
        use crate::spec::encoding::ScaleSpec;
        let batch = numeric_batch("v", vec![0.0, 100.0]);
        let spec = ScaleSpec::Linear { common: reverse_common(None, None, false), nice: false, zero: false, system_injected: false };
        let scale = resolve_scale(&spec, "v", &batch, (0.0, 500.0));
        assert_eq!(scale.data_domain(), Some((0.0, 100.0)));
    }

    /// Pins the ordering `apply_domain_reverse`'s doc calls load-bearing:
    /// the swap must run AFTER the `zero` block, not before. The `zero`
    /// block (`positional.rs:233-236`) assumes `d[0]` is the minimum — with
    /// an auto-inferred `[20, 80]` domain, `zero=true` extends it to
    /// `[0.0, 80.0]`, and reversing THAT (the implemented order) yields the
    /// correct `(80.0, 0.0)`. Reversing FIRST would hand `zero`'s
    /// ascending-pair assumption the already-swapped `[80, 20]` pair
    /// instead: `d[0]=80 > 0.0` zeros it, `d[1]=20 < 0.0` is false, so the
    /// wrong-order run corrupts the result to `(0.0, 20.0)` — a RED value
    /// confirmed by temporarily hoisting `apply_domain_reverse` above the
    /// `zero` block and running this test in place (it failed with exactly
    /// `Some((0.0, 20.0))` vs. the expected `Some((80.0, 0.0))`), then
    /// reverting the hoist back to its original position.
    #[test]
    fn linear_zero_true_reverse_true_yields_non_degenerate_descending_domain() {
        use crate::spec::encoding::ScaleSpec;
        let batch = numeric_batch("v", vec![20.0, 80.0]);
        let spec = ScaleSpec::Linear { common: reverse_common(None, None, true), nice: false, zero: true, system_injected: false };
        let scale = resolve_scale(&spec, "v", &batch, (0.0, 500.0));
        assert_eq!(
            scale.data_domain(),
            Some((80.0, 0.0)),
            "zero=true, reverse=true must compose to the non-degenerate descending pair, not collapse to (0, 0)"
        );
    }

    // ── §4C composition rules ──

    /// Rule 1: `reverse=True` + explicit `domain=[a,b]` → the GIVEN pair is
    /// swapped (not the auto-inferred one — the column here carries values
    /// the explicit domain ignores entirely).
    #[test]
    fn linear_reverse_true_with_explicit_domain_swaps_the_given_pair() {
        use crate::spec::encoding::ScaleSpec;
        let batch = numeric_batch("v", vec![-999.0, 999.0]);
        let spec = ScaleSpec::Linear {
            common: reverse_common(Some(vec![10.0, 90.0]), None, true),
            nice: false,
            zero: false,
            system_injected: false,
        };
        let scale = resolve_scale(&spec, "v", &batch, (0.0, 500.0));
        assert_eq!(scale.data_domain(), Some((90.0, 10.0)));
    }

    /// Rule 2: `reverse=True` + explicit `range=[hi,lo]` → both apply as
    /// stated. The domain still swaps; the range passes through completely
    /// untouched by `reverse` (a reversed explicit range means what it
    /// says, independent of the domain-swap sugar).
    #[test]
    fn linear_reverse_true_composes_with_reversed_explicit_range() {
        use crate::spec::encoding::ScaleSpec;
        let batch = numeric_batch("v", vec![0.0, 100.0]);
        let spec = ScaleSpec::Linear {
            common: reverse_common(None, Some(vec![500.0, 0.0]), true),
            nice: false,
            zero: false,
            system_injected: false,
        };
        let scale = resolve_scale(&spec, "v", &batch, (0.0, 500.0));
        assert_eq!(scale.data_domain(), Some((100.0, 0.0)), "reverse still swaps the domain");
        let ScaleKind::Linear(linear) = &scale else { panic!("expected Linear scale") };
        assert_eq!(
            linear.range_pair(),
            [500.0, 0.0],
            "the explicit reversed range passes through unmodified by `reverse`"
        );
    }

    /// Rule 3: `coord_cartesian(x_domain=)` wins over the encoding scale's
    /// `reverse` — it replaces the domain wholesale and its own endpoint
    /// order is respected, regardless of what the encoding's scale asked
    /// for. Asserted through the real production sequence
    /// (`resolve_scales` → `build_axis_scale` → `apply_coord_domain_overrides`)
    /// rather than hand-calling the two functions in the test body, so a
    /// future reorder of that pipeline would be caught here.
    #[test]
    fn coord_cartesian_x_domain_override_wins_over_encoding_reverse() {
        use crate::spec::coord::CoordKind;
        use crate::spec::data_ref::DataRef;
        use crate::spec::encoding::{Encoding, EncodingSpec, ScaleSpec};
        use crate::spec::mark::Mark;

        let batch = {
            use arrow::array::Float64Array;
            use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
            use std::sync::Arc;
            let schema = Arc::new(Schema::new(vec![
                Field::new("x", ArrowDataType::Float64, false),
                Field::new("y", ArrowDataType::Float64, false),
            ]));
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Float64Array::from(vec![0.0, 100.0])),
                    Arc::new(Float64Array::from(vec![0.0, 1.0])),
                ],
            )
            .unwrap()
        };
        let spec = crate::spec::chart::ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec {
                    field: "x".into(),
                    type_: None,
                    scale: Some(ScaleSpec::Linear {
                        common: reverse_common(None, None, true),
                        nice: false,
                        zero: false,
                        system_injected: false,
                    }),
                    ..Default::default()
                }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: Some(CoordKind::Cartesian {
                x_domain: Some((5.0, 20.0)),
                y_domain: None,
                expand: true,
                clip: true,
            }),
            mark_style: None,
            position: None,
            title: None,
            axis_x: None,
            axis_y: None,
            selections: Vec::new(),
            conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        };
        let theme = crate::layout::ThemeInputs::default();
        let (scales, _) =
            crate::render::scale_resolve::resolve_scales(&spec, &batch, (0.0, 500.0), (0.0, 500.0), &theme).unwrap();
        // The override's own (5.0, 20.0) endpoint order wins outright — the
        // encoding scale's `reverse=true` never gets a chance to swap it,
        // because `apply_coord_domain_overrides` replaces the whole scale.
        assert_eq!(scales.x.data_domain(), Some((5.0, 20.0)));
    }

    // ── spec §9's discriminating y-axis assertion ──

    /// A minimal, mark/coord-agnostic `ChartSpec` for tests that call
    /// `build_axis_scale`/`build_from_scale_spec` directly: the auto-inferred
    /// Ordinal/Nominal arm never reads `spec` at all, and the explicit
    /// `enc.scale` paths return before `spec` would matter (`resolve_numeric_extent`
    /// is only reachable from the auto path) — so any well-formed `ChartSpec`
    /// satisfies the parameter for these entry-point tests.
    fn minimal_point_chart_spec() -> crate::spec::chart::ChartSpec {
        use crate::spec::data_ref::DataRef;
        use crate::spec::mark::Mark;
        crate::spec::chart::ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Default::default(),
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
            position: None,
            title: None,
            axis_x: None,
            axis_y: None,
            selections: Vec::new(),
            conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        }
    }

    /// Resolves a `ScaleSpec::Linear` y-axis scale through the REAL
    /// `build_axis_scale("y", ...)` entry point (not `build_from_scale_spec`
    /// directly), so `axis_pixel_range`'s structural pixel-range inversion
    /// for the y channel is actually computed by production code and fed
    /// forward — a future reorder of that composition would fail this test,
    /// not just a hand-mirrored fixture.
    ///
    /// An explicit `domain=[0, 100]` (with no explicit `range`) keeps the
    /// pixel math exact: an explicit domain suppresses the default 5%
    /// padding inset (`resolve_padding_fraction`), so the un-padded plot
    /// range `(0.0, 400.0)` passed in comes back out of `axis_pixel_range`'s
    /// y-inversion as exactly `(400.0, 0.0)`, with no inset arithmetic to
    /// account for in the assertions below.
    fn build_y_axis_scale(reverse: bool) -> ScaleKind {
        use crate::spec::encoding::{EncodingSpec, ScaleSpec};
        use std::collections::HashMap;

        let batch = numeric_batch("v", vec![0.0, 100.0]);
        let enc = EncodingSpec {
            field: "v".into(),
            type_: None,
            scale: Some(ScaleSpec::Linear {
                common: reverse_common(Some(vec![0.0, 100.0]), None, reverse),
                nice: false,
                zero: false,
                system_injected: false,
            }),
            ..Default::default()
        };
        let spec = minimal_point_chart_spec();
        let outputs: HashMap<String, RecordBatch> = HashMap::new();
        let mut warnings = Vec::new();
        super::build_axis_scale(
            "y",
            &enc,
            None,
            super::PositionalFields { x: None, y: Some("v") },
            &batch,
            &outputs,
            (0.0, 400.0),
            &spec,
            false,
            None,
            &mut warnings,
        )
        .unwrap()
    }

    // ── #247A: scale provenance vs. the explicit-scale bypass ──

    /// A two-column `[1, 3]` / `[40, 80]` batch: the bar's own values and a
    /// sibling mark's wider ones, so which domain a scale resolved against is
    /// readable off the resolved domain alone.
    fn bar_and_sibling_batch() -> RecordBatch {
        use arrow::array::Float64Array;
        use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("v", ArrowDataType::Float64, false),
            Field::new("p", ArrowDataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 3.0])),
                Arc::new(Float64Array::from(vec![40.0, 80.0])),
            ],
        )
        .unwrap()
    }

    /// A layer binding `y` to `field` with `mark`, and nothing else set.
    fn y_layer(mark: crate::spec::mark::Mark, field: &str) -> crate::spec::layer::Layer {
        use crate::spec::encoding::{Encoding, EncodingSpec};
        let mut encoding = Encoding::default();
        encoding.y = Some(EncodingSpec { field: field.into(), type_: None, ..Default::default() });
        crate::spec::layer::Layer {
            mark,
            encoding,
            transforms: Vec::new(),
            mark_style: None,
            data_source: None,
            position: None,
            blend: None,
            name: None,
            independent_y: false, coord: None,
        }
    }

    /// Resolve a y axis whose encoding carries the bar zero-anchor's shape
    /// (`{type: linear, zero: true}`, no domain) through the REAL
    /// `build_axis_scale` entry point, with the provenance flag, the composite
    /// shared domain, and the spec's layers as the three variables — the three
    /// inputs the #247A gate reads.
    ///
    /// The bar's column is `[1, 3]`, its sibling's is `[40, 80]`, and any shared
    /// domain passed in is `[0, 80]`, so every route is distinguishable by the
    /// resolved domain alone: the bypass resolves the bar column's own extent
    /// (zero-extended to `[0, 3]`), the composite arm takes the shared extent,
    /// and the layered arm takes the cross-layer union.
    fn build_zero_anchored_y(
        system_injected: bool,
        shared: Option<&SharedDomain>,
        layers: Option<Vec<crate::spec::layer::Layer>>,
    ) -> ScaleKind {
        use crate::spec::encoding::{EncodingSpec, ScaleSpec};
        use std::collections::HashMap;

        let batch = bar_and_sibling_batch();
        let enc = EncodingSpec {
            field: "v".into(),
            type_: None,
            scale: Some(ScaleSpec::Linear {
                common: reverse_common(None, None, false),
                nice: false,
                zero: true,
                system_injected,
            }),
            ..Default::default()
        };
        let mut spec = minimal_point_chart_spec();
        spec.mark = crate::spec::mark::Mark::Bar;
        spec.layers = layers;
        let outputs: HashMap<String, RecordBatch> = HashMap::new();
        let mut warnings = Vec::new();
        super::build_axis_scale(
            "y",
            &enc,
            None,
            super::PositionalFields { x: None, y: Some("v") },
            &batch,
            &outputs,
            (0.0, 400.0),
            &spec,
            false,
            shared,
            &mut warnings,
        )
        .unwrap()
    }

    /// The defect itself: a composite leaf whose scale ferrum injected must take
    /// the shared domain, not its own. Pre-fix this returned `(0, 3)` — the
    /// bypass fired before `shared` was ever read, which is what left every
    /// sibling layer of a zero-anchored bar drawing against a scale the rendered
    /// ticks did not describe.
    #[test]
    fn injected_scale_yields_to_the_composite_shared_domain() {
        let shared = SharedDomain::Numeric { lo: 0.0, hi: 80.0 };
        let scale = build_zero_anchored_y(true, Some(&shared), None);
        assert_eq!(scale.data_domain(), Some((0.0, 80.0)));
    }

    /// The same defect through the OTHER lowering: a flat layered spec
    /// (`bar + point`), where the injected scale sits on the chart-level
    /// encoding every layer resolves against. No shared domain exists here — the
    /// cross-layer union inside `domain::numeric_domain_union` is what the scale
    /// yields to, and that union's own `layer_wants_bar_zero_anchor` supplies the
    /// anchor, which is the rule's whole purpose and was unreachable through this
    /// shape while the bypass fired first.
    ///
    /// `(0, 80)` is discriminating in both directions: the bypass would answer
    /// `(0, 3)` (the bar column alone) and a union that lost the anchor would
    /// answer `(1, 80)`.
    #[test]
    fn injected_scale_yields_to_a_layered_specs_cross_layer_union() {
        use crate::spec::mark::Mark;
        let layers = vec![y_layer(Mark::Bar, "v"), y_layer(Mark::Point, "p")];
        let scale = build_zero_anchored_y(true, None, Some(layers));
        assert_eq!(scale.data_domain(), Some((0.0, 80.0)));
    }

    /// The shape the widened gate's risk lives in, pinned rather than left
    /// unknown: a layered spec carrying chart-level `mark: bar` whose layers
    /// contain NO bar. The union runs and no anchor is applied, so the domain is
    /// the raw `(1, 80)`.
    ///
    /// That is correct, not merely tolerated. `prepare::build_layers` builds its
    /// layers from `spec.layers` ALONE when it is `Some`, and `Layer::mark` is a
    /// required field, so a layered spec's `spec.mark` draws nothing whatsoever —
    /// no bar exists here, so no bar baseline needs keeping in view, and the old
    /// behavior (anchoring at zero off the chart-level bar mark) was anchoring an
    /// axis for a mark that is never drawn.
    ///
    /// No public-API construction reaches this shape: every lowering that keeps a
    /// chart-level bar mark alongside layers derives layer 0 from that same bar
    /// chart (measured across `+`, `fm.layer` merged, `resolve={"y":
    /// "independent"}`, `SecondaryY`, annotations, `catplot(kind="bar")` and
    /// `displot(kind="hist")` — see `tests/test_layer_union_provenance.py::
    /// test_no_public_lowering_produces_a_bar_marked_spec_without_a_bar_layer`,
    /// which fails if one ever starts to). This test exists so that if one does,
    /// the behavior it lands on is the one written down here.
    #[test]
    fn layered_spec_with_no_bar_layer_takes_the_union_without_an_anchor() {
        use crate::spec::mark::Mark;
        let layers = vec![y_layer(Mark::Point, "v"), y_layer(Mark::Line, "p")];
        let scale = build_zero_anchored_y(true, None, Some(layers));
        assert_eq!(scale.data_domain(), Some((1.0, 80.0)));
    }

    /// The provenance half of the gate, as a control: the identical scale with
    /// `zero: true` but written by the CALLER keeps the bypass even when a shared
    /// domain is offered. `zero` is not what decides — authorship is.
    #[test]
    fn user_written_zero_scale_keeps_the_bypass_against_a_shared_domain() {
        let shared = SharedDomain::Numeric { lo: 0.0, hi: 80.0 };
        let scale = build_zero_anchored_y(false, Some(&shared), None);
        assert_eq!(scale.data_domain(), Some((0.0, 3.0)));
    }

    /// The same control on the layered arm: a caller's own scale is not overruled
    /// by the presence of layers either, so their `padding`/`nice`/`range` keep
    /// being honored in a layered chart.
    #[test]
    fn user_written_zero_scale_keeps_the_bypass_in_a_layered_spec() {
        use crate::spec::mark::Mark;
        let layers = vec![y_layer(Mark::Bar, "v"), y_layer(Mark::Point, "p")];
        let scale = build_zero_anchored_y(false, None, Some(layers));
        assert_eq!(scale.data_domain(), Some((0.0, 3.0)));
    }

    /// A composite mark's layers (a box's whisker rule `lo..hi` plus its
    /// outlier points `out`), the whisker carrying the caller-written scale and
    /// the outliers carrying a copy of it only when `outliers_scaled`.
    /// Columns: `lo = [2, 3]`, `hi = [5, 6]`, `out = [0.5, 9]`, so the bound
    /// column alone spans `[2, 3]`, the whisker `[2, 6]`, and the whole mark
    /// `[0.5, 9]`.
    fn build_composite_y(
        common: crate::spec::encoding::ContinuousScaleCommon,
        outliers_scaled: bool,
    ) -> Result<ScaleKind, RenderError> {
        use crate::spec::encoding::{EncodingSpec, ScaleSpec};
        use crate::spec::mark::Mark;
        use arrow::array::Float64Array;
        use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
        use std::collections::HashMap;
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("lo", ArrowDataType::Float64, false),
            Field::new("hi", ArrowDataType::Float64, false),
            Field::new("out", ArrowDataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![2.0, 3.0])),
                Arc::new(Float64Array::from(vec![5.0, 6.0])),
                Arc::new(Float64Array::from(vec![0.5, 9.0])),
            ],
        )
        .unwrap();
        let scale = ScaleSpec::Linear { common, nice: false, zero: false, system_injected: false };
        let scaled = |field: &str| EncodingSpec {
            field: field.into(),
            type_: None,
            scale: Some(scale.clone()),
            ..Default::default()
        };
        let mut whisker = y_layer(Mark::Rule, "lo");
        whisker.encoding.y = Some(scaled("lo"));
        whisker.encoding.y2 = Some(EncodingSpec { field: "hi".into(), type_: None, ..Default::default() });
        let mut outliers = y_layer(Mark::Point, "out");
        if outliers_scaled {
            outliers.encoding.y = Some(scaled("out"));
        }
        let mut spec = minimal_point_chart_spec();
        spec.layers = Some(vec![whisker, outliers]);
        let mut warnings = Vec::new();
        super::build_axis_scale(
            "y",
            &scaled("lo"),
            None,
            super::PositionalFields { x: None, y: Some("lo") },
            &batch,
            &HashMap::new(),
            (0.0, 400.0),
            &spec,
            false,
            None,
            &mut warnings,
        )
    }

    /// A caller's scale that declares no `domain` (here `reverse`) spans every
    /// layer carrying a copy of it, reversed: `(9, 0.5)`. It used to resolve the
    /// bound column alone, `(3, 2)`, so a boxplot drawn with
    /// `scale={"reverse": True}` lost its upper whiskers and outliers off the
    /// axis while the same chart without the scale covered them.
    #[test]
    fn domainless_user_scale_spans_every_layer_carrying_it() {
        let scale = build_composite_y(reverse_common(None, None, true), true).unwrap();
        assert_eq!(scale.data_domain(), Some((9.0, 0.5)));
    }

    /// A sibling layer that does not carry the scale stays out: the axis is the
    /// scaled whisker's own `lo..hi`, `(6, 2)` reversed, not the outliers' `9`.
    #[test]
    fn domainless_user_scale_leaves_out_a_layer_without_it() {
        let scale = build_composite_y(reverse_common(None, None, true), false).unwrap();
        assert_eq!(scale.data_domain(), Some((6.0, 2.0)));
    }

    /// Control: a stated `domain` is the caller's extent and is taken
    /// literally, layers or not.
    #[test]
    fn stated_domain_on_a_layered_spec_is_taken_literally() {
        let scale = build_composite_y(reverse_common(Some(vec![1.0, 4.0]), None, false), true).unwrap();
        assert_eq!(scale.data_domain(), Some((1.0, 4.0)));
    }

    /// A caller's continuous scale on a non-numeric column refuses in the
    /// scale's voice whether or not the spec is layered.
    #[test]
    fn layered_domainless_scale_on_a_text_column_refuses_as_a_scale() {
        use crate::spec::encoding::{EncodingSpec, ScaleSpec};
        use crate::spec::mark::Mark;
        use std::collections::HashMap;

        let batch = category_batch("cat", &["a".to_string(), "b".to_string()]);
        let enc = EncodingSpec {
            field: "cat".into(),
            type_: None,
            scale: Some(ScaleSpec::Linear {
                common: reverse_common(None, None, true),
                nice: false,
                zero: false,
                system_injected: false,
            }),
            ..Default::default()
        };
        let mut layer = y_layer(Mark::Point, "cat");
        layer.encoding.y = Some(enc.clone());
        let mut spec = minimal_point_chart_spec();
        spec.layers = Some(vec![layer]);
        let mut warnings = Vec::new();
        let err = super::build_axis_scale(
            "y",
            &enc,
            None,
            super::PositionalFields { x: None, y: Some("cat") },
            &batch,
            &HashMap::new(),
            (0.0, 400.0),
            &spec,
            false,
            None,
            &mut warnings,
        )
        .unwrap_err();
        assert_eq!(err.to_string(), "scale: column 'cat' has unsupported dtype: Utf8");
    }

    /// Resolve `y` for `enc` (bound field) and `paired` (its `y2`) through the
    /// real `build_axis_scale`, with `layers` as the spec's layers.
    fn resolve_y(
        batch: &RecordBatch,
        enc: &crate::spec::encoding::EncodingSpec,
        paired: Option<&crate::spec::encoding::EncodingSpec>,
        layers: Option<Vec<crate::spec::layer::Layer>>,
    ) -> ScaleKind {
        let mut spec = minimal_point_chart_spec();
        spec.layers = layers;
        let mut warnings = Vec::new();
        super::build_axis_scale(
            "y",
            enc,
            paired,
            super::PositionalFields { x: None, y: Some(enc.field.as_str()) },
            batch,
            &std::collections::HashMap::new(),
            (0.0, 400.0),
            &spec,
            false,
            None,
            &mut warnings,
        )
        .unwrap()
    }

    /// `field` bound to y with a caller-written linear scale.
    fn linear_y(field: &str, zero: bool, reverse: bool) -> crate::spec::encoding::EncodingSpec {
        crate::spec::encoding::EncodingSpec {
            field: field.into(),
            type_: None,
            scale: Some(crate::spec::encoding::ScaleSpec::Linear {
                common: reverse_common(None, None, reverse),
                nice: false,
                zero,
                system_injected: false,
            }),
            ..Default::default()
        }
    }

    /// A `mark` layer binding y to `enc`.
    fn scaled_layer(mark: crate::spec::mark::Mark, enc: crate::spec::encoding::EncodingSpec) -> crate::spec::layer::Layer {
        let mut layer = y_layer(mark, &enc.field);
        layer.encoding.y = Some(enc);
        layer
    }

    /// An unlayered `y`/`y2` mark with a domain-less caller scale spans both
    /// endpoints: `v = [1, 2]`, `w = [30, 40]` resolves `(40, 1)` reversed, not
    /// the bound column's `(2, 1)` with every `y2` endpoint off the axis.
    #[test]
    fn unlayered_domainless_scale_spans_its_paired_endpoint() {
        use crate::spec::encoding::EncodingSpec;
        let schema_batch = {
            use arrow::array::Float64Array;
            use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
            use std::sync::Arc;
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![
                    Field::new("v", ArrowDataType::Float64, false),
                    Field::new("w", ArrowDataType::Float64, false),
                ])),
                vec![
                    Arc::new(Float64Array::from(vec![1.0, 2.0])),
                    Arc::new(Float64Array::from(vec![30.0, 40.0])),
                ],
            )
            .unwrap()
        };
        let paired = EncodingSpec { field: "w".into(), type_: None, ..Default::default() };
        let scale = resolve_y(&schema_batch, &linear_y("v", false, true), Some(&paired), None);
        assert_eq!(scale.data_domain(), Some((40.0, 1.0)));
    }

    /// One row, `v = 5`, under a caller scale with `zero: true`: the layered
    /// spec (a bar and a point both carrying the scale) resolves `(0, 5)`, the
    /// same as the unlayered one. The auto path's single-value `±0.5` widening
    /// is not the caller scale's to take, and would give `(0, 5.5)`.
    #[test]
    fn single_value_caller_scale_resolves_the_same_layered_and_unlayered() {
        use crate::spec::mark::Mark;
        let batch = numeric_batch("v", vec![5.0]);
        let enc = linear_y("v", true, false);
        let flat = resolve_y(&batch, &enc, None, None);
        let layers = vec![scaled_layer(Mark::Bar, enc.clone()), scaled_layer(Mark::Point, enc.clone())];
        let layered = resolve_y(&batch, &enc, None, Some(layers));
        assert_eq!(flat.data_domain(), Some((0.0, 5.0)));
        assert_eq!(layered.data_domain(), flat.data_domain());
    }

    /// A bar layer carrying a caller scale with `zero: false` is not anchored
    /// at zero: its scale's own `zero` decides, not the auto path's bar rule.
    /// `v = [1, 3]` and `p = [40, 80]`, both scaled, resolve `(1, 80)`.
    #[test]
    fn a_scaled_bar_layer_takes_its_scales_zero_not_the_bar_anchor() {
        use crate::spec::mark::Mark;
        let batch = bar_and_sibling_batch();
        let enc = linear_y("v", false, false);
        let layers = vec![scaled_layer(Mark::Bar, enc.clone()), scaled_layer(Mark::Point, linear_y("p", false, false))];
        let scale = resolve_y(&batch, &enc, None, Some(layers));
        assert_eq!(scale.data_domain(), Some((1.0, 80.0)));
    }

    /// The other half: with NEITHER a shared domain nor layers there is nothing
    /// to yield to, so an injected scale resolves through the explicit path
    /// exactly as it always did. This is the byte-stability of every standalone
    /// single-mark bar render, stated as an assertion rather than assumed —
    /// `(0, 3)` is the bar column's extent `[1, 3]` zero-extended by the bypass's
    /// own `zero` handling.
    ///
    /// This is also what keeps a FACETED single-mark bar unchanged: a facet passes
    /// no shared domain and a faceted single-mark spec has no layers, so it stays
    /// on the bypass. (That shape has its own separate defect — its panels resolve
    /// per-panel extents under a shared facet axis — which this task does not
    /// touch and which cannot be closed by another disjunct here: with no layers,
    /// `layer_wants_bar_zero_anchor` has nothing to fire on, so the auto path
    /// would trade the mark/axis disagreement for a lost zero baseline. Measured,
    /// and logged with the north-star unification.)
    #[test]
    fn injected_scale_on_an_unlayered_standalone_chart_keeps_the_bypass() {
        let scale = build_zero_anchored_y(true, None, None);
        assert_eq!(scale.data_domain(), Some((0.0, 3.0)));
    }

    /// An EMPTY layers list is not a layered spec: there are no layer fields to
    /// union and no layer mark to take an anchor from, so yielding would strip the
    /// zero anchor in exchange for nothing. Unreachable from Python today (the
    /// `layers` key is written only when `_layers is not None`, and every producer
    /// appends at least the LHS's own layer), which is exactly why it is pinned —
    /// the gate reads `is_some_and(!is_empty)` rather than `is_some()` on purpose,
    /// and a later refactor back to the shorter spelling fails here.
    #[test]
    fn injected_scale_with_an_empty_layers_list_keeps_the_bypass() {
        let scale = build_zero_anchored_y(true, None, Some(Vec::new()));
        assert_eq!(scale.data_domain(), Some((0.0, 3.0)));
    }

    /// `extend_extent_to_zero` keeps the guarded-comparison semantics its two
    /// call sites had before the extraction: a `NaN` extent (an all-null column)
    /// stays `NaN` rather than being silently turned into `0.0`, which
    /// `f64::min`/`max` would have done.
    #[test]
    fn extend_extent_to_zero_widens_without_laundering_nan() {
        assert_eq!(super::extend_extent_to_zero(5.0, 30.0), (0.0, 30.0));
        assert_eq!(super::extend_extent_to_zero(-30.0, -5.0), (-30.0, 0.0));
        assert_eq!(super::extend_extent_to_zero(-4.0, 9.0), (-4.0, 9.0));
        let (lo, hi) = super::extend_extent_to_zero(f64::NAN, f64::NAN);
        assert!(lo.is_nan() && hi.is_nan(), "got ({lo}, {hi})");
    }

    // ── #183: degenerate/high-cardinality ordinal grouping ──

    /// A one-column Utf8 batch of `cats` distinct categories, resolved
    /// through the REAL auto-inferred `build_axis_scale` entry point a
    /// `catplot(kind="box")`-style chart takes: no explicit `enc.scale`, a
    /// plain categorical field binding, `type_: None` so `infer_spec_type`
    /// classifies the Utf8 column as Nominal exactly like production does.
    fn build_x_axis_scale_for_categories(cats: Vec<String>, pixel_range: (f64, f64)) -> (ScaleKind, Vec<RenderWarning>) {
        use crate::spec::encoding::EncodingSpec;
        use std::collections::HashMap;

        let batch = category_batch("cat", &cats);
        let enc = EncodingSpec { field: "cat".into(), type_: None, ..Default::default() };
        let spec = minimal_point_chart_spec();
        let outputs: HashMap<String, RecordBatch> = HashMap::new();
        let mut warnings = Vec::new();
        let scale = super::build_axis_scale(
            "x",
            &enc,
            None,
            super::PositionalFields { x: Some("cat"), y: None },
            &batch,
            &outputs,
            pixel_range,
            &spec,
            false,
            None,
            &mut warnings,
        )
        .unwrap();
        (scale, warnings)
    }

    /// RED proof for #118/#183 (D §183): a 500-category axis squeezed into a
    /// 300px plot area — the shape a `catplot(kind="box")` over a
    /// high-cardinality grouping column produces — used to resolve silently
    /// into sub-pixel bands. It now names the category count.
    #[test]
    fn build_axis_scale_warns_on_500_category_sub_pixel_bandwidth() {
        let cats: Vec<String> = (0..500).map(|i| format!("cat{i}")).collect();
        let (scale, warnings) = build_x_axis_scale_for_categories(cats, (0.0, 300.0));
        assert!(matches!(scale, ScaleKind::Ordinal(_)));
        assert_eq!(warnings.len(), 1, "got {warnings:?}");
        match &warnings[0] {
            RenderWarning::GroupCardinalityDegenerate { categories, bandwidth_px } => {
                assert_eq!(*categories, 500);
                assert!(*bandwidth_px < 1.0, "bandwidth_px was {bandwidth_px}");
            }
            other => panic!("expected GroupCardinalityDegenerate, got {other:?}"),
        }
    }

    /// A handful of categories over a generous plot area stays silent — the
    /// common case must not gain a spurious warning.
    #[test]
    fn build_axis_scale_silent_on_healthy_bandwidth() {
        let cats: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
        let (scale, warnings) = build_x_axis_scale_for_categories(cats, (0.0, 300.0));
        assert!(matches!(scale, ScaleKind::Ordinal(_)));
        assert!(warnings.is_empty(), "got {warnings:?}");
    }

    /// `reverse=true` composes with the y-channel's structural pixel-range
    /// inversion (`axis_pixel_range`, computed inside `build_axis_scale`
    /// before `build_from_scale_spec` ever sees the range) without one
    /// mechanism clobbering the other: marks (`to_pixel_f64`) and tick
    /// labels (`tick_labels`, index-aligned with `tick_fractions`) flip
    /// together, not independently.
    #[test]
    fn linear_reverse_true_on_y_channel_flips_marks_and_labels_together() {
        let baseline = build_y_axis_scale(false);
        let reversed = build_y_axis_scale(true);

        // Marks: the minimum data value (0) sits at the BOTTOM (pixel 400)
        // on the default axis, and at the TOP (pixel 0) once reversed.
        assert_eq!(baseline.to_pixel_f64(0.0), Some(400.0));
        assert_eq!(reversed.to_pixel_f64(0.0), Some(0.0));
        assert_eq!(baseline.to_pixel_f64(100.0), Some(0.0));
        assert_eq!(reversed.to_pixel_f64(100.0), Some(400.0));

        // Labels: `tick_labels`/`tick_fractions` stay index-aligned on BOTH
        // scales, and the reversed scale's own tick VALUES (the same order
        // `tick_labels` uses) come out descending — the label order flipped
        // along with the marks, not independently of them.
        let base_values = baseline.tick_values_raw(5).unwrap();
        let rev_values = reversed.tick_values_raw(5).unwrap();
        assert_eq!(baseline.tick_labels(5).len(), baseline.tick_fractions(5).len());
        assert_eq!(reversed.tick_labels(5).len(), reversed.tick_fractions(5).len());
        assert!(!rev_values.is_empty());
        assert!(base_values.windows(2).all(|w| w[0] <= w[1]), "baseline ticks ascend: {base_values:?}");
        assert!(rev_values.windows(2).all(|w| w[0] >= w[1]), "reversed ticks descend: {rev_values:?}");

        // Same underlying tick set, reordered — `reverse` must not change
        // WHICH values get labeled, only the order they're labeled in.
        let mut base_sorted = base_values.clone();
        let mut rev_sorted = rev_values.clone();
        base_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        rev_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(base_sorted, rev_sorted, "reverse reorders ticks, it does not pick a different set");
    }
}
