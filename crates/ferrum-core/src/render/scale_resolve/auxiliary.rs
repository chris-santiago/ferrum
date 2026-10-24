//! Auxiliary scale resolution: size, shape, opacity-family, and stroke-dash
//! encodings.

use std::collections::HashMap;

use arrow::record_batch::RecordBatch;

use crate::layout::ThemeInputs;
use crate::scale::linear::LinearScale;
use crate::scale::log::LogScale;
use crate::scale::pow::PowScale;

use ferrum_scene::{categorical_slot_dash, STROKE_DASH_LEGACY_SLOTS};

use crate::render::RenderError;
use crate::spec::encoding::{Encoding, EncodingSpec, ScaleSpec};

use super::domain::{admitted_aux_reads, aux_category_union, sided_aux_numeric_union, apply_sort_to_domain, explicit_ordinal_domain, explicit_ordinal_range, legend_sort_context, AuxRead};
use super::{distinct_values_in_order, infer_spec_type, shared_categorical_batch, sided_min_max_f64, OpacityScale, ScaleKind, ShapeKind, ShapeScale, SharedDomain, SizeScale, StrokeDashScale, StrokeWidthScale, SHAPE_PALETTE};

/// The channels whose scale maps a quantitative field onto a bounded output
/// interval: the one gate [`numeric_aux_scale`] answers for.
pub(crate) const NUMERIC_AUX_CHANNELS: [&str; 5] = ["size", "opacity", "fill_opacity", "stroke_opacity", "stroke_width"];

/// [`log_aux_drawn_rows`]'s result: the Log-filtered batch, and the SOURCE
/// row (an index into the batch `log_aux_drawn_rows` was given) of every row
/// the filtered batch kept, in the filtered batch's own order — the mapping
/// `scene_build::restore_source_rows` uses to un-renumber a mark builder's
/// `data_indices`/`vertex_rows` back to source rows.
pub(in crate::render) struct LogAuxFiltered {
    pub(in crate::render) batch: RecordBatch,
    pub(in crate::render) source_rows: Vec<usize>,
}

/// Drop a row an off-side `Log` numeric auxiliary scale cannot map: a row whose
/// `size`/`opacity`/`fill_opacity`/`stroke_opacity`/`stroke_width` binding
/// reads a FINITE value `v` with `!LogScale::admits(v)` is not drawn — the
/// same treatment an off-side `x`/`y` row already gets
/// (`ScaleKind::to_pixel_f64`'s `None`), now on every numeric auxiliary
/// channel too, since `to_pixel_f64`'s `None` cannot express "not drawn" on
/// an aux reader (every one of them folds `None` into the channel's
/// default). A null or NaN value is kept (it draws at the default,
/// unchanged) — only a value the scale has actually REJECTED for being on
/// the excluded side is dropped. A row rejected on ANY bound Log aux
/// channel is dropped once, not once per channel: the whole row is absent
/// from every mark it would have drawn, mirroring how a single off-side x/y
/// row removes the whole row rather than one channel's worth of it.
///
/// `encoding` is the layer's own DRAWN encoding
/// (`LayerPrepared::drawn_encoding`), so only a channel this layer's mark
/// actually draws is checked — a channel the mark does not draw, or that
/// resolved to no scale at all, contributes nothing. `scales` is this
/// layer's resolved `ResolvedScales` (the y-slot clone for an independent
/// layer, same as every other read in `scene_build::build_panel_mark_batches`'s
/// loop).
///
/// Returns `None` when no numeric auxiliary channel this layer draws
/// resolved to a `Log` scale, or none of its rows are off-side on one: the
/// caller keeps borrowing its batch unchanged, so every chart without a
/// `Log` numeric auxiliary channel renders byte-identically. Otherwise
/// `Some(LogAuxFiltered)`: `arrow::compute::filter_record_batch` with one
/// combined mask (mirroring `render::filter_batch_by_facet` and
/// `transform::swarm::drop_null_dodge_rows`) paired with the kept rows'
/// own source indices, so the caller can un-renumber a mark builder's
/// `data_indices`/`vertex_rows` back to them afterward.
pub(in crate::render) fn log_aux_drawn_rows(
    encoding: &Encoding,
    scales: &super::ResolvedScales,
    batch: &RecordBatch,
) -> Result<Option<LogAuxFiltered>, RenderError> {
    let channels: [(Option<&EncodingSpec>, Option<&ScaleKind>); 5] = [
        (encoding.size.as_ref(), scales.size.as_ref().map(|s| &s.inner)),
        (encoding.opacity.as_ref(), scales.opacity.as_ref().map(|s| &s.inner)),
        (encoding.fill_opacity.as_ref(), scales.fill_opacity.as_ref().map(|s| &s.inner)),
        (encoding.stroke_opacity.as_ref(), scales.stroke_opacity.as_ref().map(|s| &s.inner)),
        (encoding.stroke_width.as_ref(), scales.stroke_width.as_ref().map(|s| &s.inner)),
    ];
    let mut keep: Option<Vec<bool>> = None;
    for (enc, scale) in channels {
        let (Some(enc), Some(ScaleKind::Log(log))) = (enc, scale) else { continue };
        if batch.column_by_name(&enc.field).is_none() {
            continue;
        }
        let Ok(values) = crate::render::arrow_cast::col_as_f64(batch, &enc.field) else { continue };
        // A row is kept when it is null, `NaN` (drawn at the channel's own
        // default, unchanged) or on this scale's side —
        // `on_side` is the sign alone, so an
        // on-side infinity is kept (and clamps downstream, like `Linear`)
        // and an off-side infinity is dropped here, same as a finite one.
        let admits_row: Vec<bool> = values
            .iter()
            .map(|v| match v {
                Some(v) if v.is_nan() => true,
                Some(v) => log.on_side(*v),
                None => true,
            })
            .collect();
        keep = Some(match keep {
            None => admits_row,
            Some(existing) => existing.into_iter().zip(admits_row).map(|(a, b)| a && b).collect(),
        });
    }
    let Some(keep) = keep else { return Ok(None) };
    if keep.iter().all(|&k| k) {
        return Ok(None);
    }
    let source_rows: Vec<usize> = keep.iter().enumerate().filter(|&(_, &k)| k).map(|(i, _)| i).collect();
    let mask: arrow::array::BooleanArray = keep.into_iter().map(Some).collect();
    let filtered = arrow::compute::filter_record_batch(batch, &mask)
        .map_err(|e| RenderError::ScaleResolutionFailed(format!("filter: {e}")))?;
    Ok(Some(LogAuxFiltered { batch: filtered, source_rows }))
}

/// The continuous scale kinds a numeric auxiliary channel's `scale=` may
/// select, each carrying the one extra parameter it needs beyond the
/// domain/range/reverse every kind shares ([`NumericAuxOverrides`]).
///
/// Mirrors [`super::positional::build_from_scale_spec`]'s `Linear`/`Log`/
/// `Pow`/`Sqrt` arms exactly: `Sqrt` resolves here too, as `Pow { exponent:
/// 0.5 }` (the same `ScaleKind::Pow(0.5)` the positional resolver produces
/// for it). `Pow` (and so `Sqrt`) carries no `nice` field at all —
/// `ScaleSpec::Pow`/`Sqrt` have none, and `PowScale::new_internal` never
/// nices — so there is nothing to thread for that variant, matching the
/// positional resolver's `Pow`/`Sqrt` arms, which never call `.nice()`.
#[derive(Debug, Clone, Copy)]
pub(in crate::render) enum NumericAuxKind {
    Linear { nice: bool, zero: bool },
    Log { base: f64, nice: bool },
    Pow { exponent: f64 },
}

impl Default for NumericAuxKind {
    /// No `scale=` at all resolves through the plain linear path, as before
    /// this type existed.
    fn default() -> Self {
        NumericAuxKind::Linear { nice: false, zero: false }
    }
}

/// What a numeric auxiliary channel (size, the opacity family, stroke width)
/// takes from its `scale=`, once [`numeric_aux_scale`] has admitted it.
///
/// These scales always clamp: a value outside an explicit domain draws at the
/// nearer end of the range, so `clamp=` has no other setting to honor.
#[derive(Debug, Clone, Copy, Default)]
pub(in crate::render) struct NumericAuxOverrides {
    /// An explicit `[lo, hi]` domain, which replaces the data extent.
    domain: Option<(f64, f64)>,
    /// An explicit `[lo, hi]` range, which replaces the channel's theme band.
    range: Option<(f64, f64)>,
    /// `reverse=`: the range endpoints swap, after the drawable-interval clamp.
    reverse: bool,
    /// Which continuous scale kind this channel's `scale=` selected, and the
    /// kind-specific parameter(s) it carries.
    kind: NumericAuxKind,
}

impl NumericAuxOverrides {
    /// This channel's `Log` side ([`crate::scale::log::LogSide::of`] over
    /// `domain`), or `None` for every other kind — the one place this is
    /// computed, read by the three builders before their
    /// [`numeric_aux_extent`] call and by [`numeric_aux_scale_kind`]'s own
    /// `Log` arm, so both agree on which side a stated domain names.
    fn log_side(&self) -> Option<crate::scale::log::LogSide> {
        match self.kind {
            NumericAuxKind::Log { .. } => {
                let domain = self.domain.map(|(d0, d1)| [d0, d1]);
                Some(crate::scale::log::LogSide::of(domain.as_ref().map(|a| a.as_slice())))
            }
            _ => None,
        }
    }
}

/// The one gate a numeric auxiliary channel's `scale=` passes. These scales
/// map a quantitative field onto a bounded interval through the same
/// continuous transform a positional axis would use, so a scale of a
/// non-continuous or not-yet-supported type, a `padding` (an axis inset) or a
/// `scheme` (a color scheme) is something this resolution would ignore or
/// misread, and refuses with [`RenderError::AuxiliaryScaleUnsupported`]
/// naming the supported types; a domain or range that is not `[lo, hi]`
/// refuses with the positional sentence
/// ([`super::positional::validate_explicit_continuous_shape`]). Every other
/// field of a supported type is honored ([`NumericAuxOverrides`]). `channel`
/// is the channel's wire name. No `scale=` admits the channel's defaults.
///
/// `linear`, `log`, `pow` and `sqrt` are supported — the same four continuous
/// shapes [`super::positional::build_from_scale_spec`] resolves on x/y, read
/// through the identical `LinearScale`/`LogScale`/`PowScale` constructors in
/// [`numeric_aux_scale_kind`], so an in-domain value lands at the same
/// fraction of its range as it would there. The channels diverge on what an
/// out-of-domain value draws: a numeric auxiliary channel always clamps to
/// the nearer range end (positional axes default to `clamp=False` and do
/// not draw the row at all), while a `Log` value on the far side of zero is
/// not drawn on EITHER channel class — the aux clamp
/// stays the rule for an in-range-sign, out-of-domain value; only a value
/// the scale cannot map at all (the wrong sign) is dropped, at the draw
/// seam ([`log_aux_drawn_rows`]), not clamped.
///
/// Published to Python as `ferrum._core.validate_numeric_aux_scale`, so a
/// channel class refuses at construction with the text the wire refuses with
/// at render.
pub(in crate::render) fn numeric_aux_scale(
    channel: &'static str,
    scale: &Option<ScaleSpec>,
) -> Result<NumericAuxOverrides, RenderError> {
    let Some(spec) = scale else {
        return Ok(NumericAuxOverrides::default());
    };
    let unsupported = |reason: String| RenderError::AuxiliaryScaleUnsupported { channel: channel.to_string(), reason };
    let (common, kind) = match spec {
        ScaleSpec::Linear { common, nice, zero, system_injected: _ } => {
            (common, NumericAuxKind::Linear { nice: *nice, zero: *zero })
        }
        ScaleSpec::Log { base, common, nice } => {
            super::positional::validate_explicit_log_domain(channel, common.domain.as_deref())?;
            (common, NumericAuxKind::Log { base: *base, nice: *nice })
        }
        ScaleSpec::Pow { exponent, common } => (common, NumericAuxKind::Pow { exponent: *exponent }),
        ScaleSpec::Sqrt { common } => (common, NumericAuxKind::Pow { exponent: 0.5 }),
        other => return Err(unsupported(format!("type '{}' is not supported", other.tag()))),
    };
    if common.padding.is_some() {
        return Err(unsupported("padding is not supported (it insets an axis)".to_string()));
    }
    if common.scheme.is_some() {
        return Err(unsupported("scheme is not supported (it names a color scheme)".to_string()));
    }
    super::positional::validate_explicit_continuous_shape(channel, &common.domain, &common.range)?;
    let pair = |values: &Option<Vec<f64>>| values.as_ref().map(|v| (v[0], v[1]));
    Ok(NumericAuxOverrides {
        domain: pair(&common.domain),
        range: pair(&common.range),
        reverse: common.reverse,
        kind,
    })
}

/// The output values a numeric auxiliary channel can draw: an alpha for the
/// opacity family, a non-negative area or width for size and stroke width.
#[derive(Debug, Clone, Copy)]
enum DrawableInterval {
    /// `[0, 1]`.
    Alpha,
    /// `[0, inf)`.
    NonNegative,
}

impl DrawableInterval {
    /// `range` with each endpoint clamped into the interval, so marks and
    /// legend read one bounded value and the mapping stays monotone.
    fn clamp(self, (lo, hi): (f64, f64)) -> (f64, f64) {
        match self {
            DrawableInterval::Alpha => (lo.clamp(0.0, 1.0), hi.clamp(0.0, 1.0)),
            DrawableInterval::NonNegative => (lo.max(0.0), hi.max(0.0)),
        }
    }
}

/// The continuous scale of a numeric auxiliary channel: `extent` (the data's,
/// or a composite's shared one) unless the scale states a domain, widened to
/// zero when asked (`Linear` only — the one variant with a `zero=` field),
/// onto `band` unless the scale states a range, that range clamped into
/// `interval` and then reversed when asked. `overrides.kind` picks which
/// scale this builds, through the SAME `LinearScale`/`LogScale`/`PowScale`
/// constructors [`super::positional::build_from_scale_spec`] uses for x/y, so
/// an in-domain value lands at the same fraction of its range it would
/// there. `clamp` is always `true` here (every numeric auxiliary channel's
/// own contract: see [`numeric_aux_scale`]'s doc) — a positional axis
/// defaults to `false` and does not draw an out-of-domain row at all, where
/// this clamps it to the nearer range end on every kind but `Log`, whose
/// off-side value is left out instead, handled before
/// this function ever builds the scale — see `extent`'s own doc below).
///
/// A data-derived domain is niced when `nice_data_domain` (the channel's
/// default) or the scale's own `nice=` says so, on `Linear` and `Log` alike;
/// an explicit domain only when the scale says so. `Pow` (and `Sqrt`, which
/// resolves as `Pow { exponent: 0.5 }`) carries no `nice` parameter at all —
/// mirroring `build_from_scale_spec`'s `Pow`/`Sqrt` arms, which never call
/// `.nice()` — so `nice_data_domain` has nothing to apply to for that kind.
///
/// `extent` is sided: for `Log`, `channel`/`field` and `warnings` feed
/// [`positional::resolve_log_extent`](super::positional::resolve_log_extent)
/// — the drop-and-warn step shared with x/y — before
/// the domain is built, so a row on the wrong side of zero is counted (one
/// [`RenderWarning::LogScaleRowsLeftOut`] for the WHOLE frame) and
/// left out of this SCALE's domain, and a channel with nothing left on its
/// side refuses with [`RenderError::LogScaleNoValueOnSide`]. The row itself
/// is removed at the draw seam ([`log_aux_drawn_rows`]), not here:
/// this function only resolves the domain/range the IN-range rows map
/// through. Every other kind reads `extent` unsided (`left_out` is always
/// `0` there), byte-identical to before this function could fail or warn at
/// all.
#[allow(clippy::too_many_arguments)]
fn numeric_aux_scale_kind(
    channel: &'static str,
    field: &str,
    extent: super::SidedExtent,
    tally: &super::LeftOutTally,
    overrides: NumericAuxOverrides,
    band: (f64, f64),
    interval: DrawableInterval,
    nice_data_domain: bool,
    warnings: &mut Vec<crate::render::RenderWarning>,
) -> Result<ScaleKind, RenderError> {
    let (lo, hi) = interval.clamp(overrides.range.unwrap_or(band));
    let (lo, hi) = if overrides.reverse { (hi, lo) } else { (lo, hi) };
    Ok(match overrides.kind {
        NumericAuxKind::Linear { nice, zero } => {
            let (mut lo_d, mut hi_d) = overrides.domain.unwrap_or((extent.lo, extent.hi));
            if zero {
                (lo_d, hi_d) = super::positional::extend_extent_to_zero(lo_d, hi_d);
            }
            let nice = nice || (nice_data_domain && overrides.domain.is_none());
            ScaleKind::Linear(LinearScale::new_internal(vec![lo_d, hi_d], vec![lo, hi], true, nice))
        }
        NumericAuxKind::Log { base, nice } => {
            let domain_arr = overrides.domain.map(|(d0, d1)| [d0, d1]);
            let side = crate::scale::log::LogSide::of(domain_arr.as_ref().map(|a| a.as_slice()));
            // Asked even under an explicit domain, so the rows the other
            // side leaves out are counted; its returned `(lo, hi)` is the
            // DATA's own on-side extent, used only when no domain is
            // stated — an explicit domain's bounds, below, are never
            // replaced by it, mirroring `resolve_log_scale`'s identical
            // split (`resolve_continuous_domain_and_range` reads the
            // explicit domain directly and never calls its own data-extent
            // closure when one is stated).
            let resolved = super::positional::resolve_log_extent(
                channel,
                field,
                overrides.domain.is_some(),
                side,
                (extent.lo, extent.hi),
                tally,
                base,
                // An auxiliary channel is never flipped (CoordFlip only
                // swaps x/y), mirroring LogScaleRowsLeftOut's own doc.
                false,
                warnings,
            )?;
            let (lo_d, hi_d) = overrides.domain.unwrap_or(resolved);
            let nice = nice || (nice_data_domain && overrides.domain.is_none());
            ScaleKind::Log(LogScale::new_internal(vec![lo_d, hi_d], vec![lo, hi], base, true, nice))
        }
        NumericAuxKind::Pow { exponent } => {
            let (lo_d, hi_d) = overrides.domain.unwrap_or((extent.lo, extent.hi));
            ScaleKind::Pow(PowScale::new_internal(vec![lo_d, hi_d], vec![lo, hi], exponent, true))
        }
    })
}

/// Shared prelude for a *sorted categorical domain*: resolve the (possibly
/// facet-shared) domain batch, take distinct values in first-appearance
/// order (unless `explicit_domain` overrides it, #123 D1), then reorder per
/// `sort`. [`build_shape_scale`] and [`build_stroke_dash_scale`] both need
/// exactly this sequence — same domain resolution, same sort context shape
/// — so a future change to sort context or facet-shared domain resolution
/// is made once here, not twice at each call site.
///
/// `explicit_domain` is the encoding's own declared `scale.domain`
/// ([`explicit_ordinal_domain`]), when present. It wins outright over the
/// data-derived first-appearance order — mirroring color's D1 domain
/// resolution (`color.rs`'s categorical resolver) — and, unlike color's
/// default (no-explicit-range) path, is honored here unconditionally: a
/// `Shape`/`StrokeDash(scale=OrdinalScale(domain=[...]))` with no range must
/// already reassign glyphs/patterns (#123, user ruling Q1), so there is no
/// range-gated carve-out to mirror. `sort=`, when present, still applies on
/// top of either source.
///
/// `layer_reads` are a layered spec's admitted reads of the channel
/// ([`AuxRead`]): the data-derived domain is then every drawing layer's
/// categories in layer order ([`aux_category_union`]), and a data-aware
/// `sort=` reads the first contributing read's own pre-statistic rows
/// ([`legend_sort_context`]). Empty for an unlayered spec, which reads
/// `batch`.
#[allow(clippy::too_many_arguments)]
fn sorted_categorical_domain(
    encoding: &Encoding,
    field: &str,
    explicit_domain: Option<Vec<String>>,
    sort: Option<&serde_json::Value>,
    batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    transform_lineages: &crate::transform::run::OutputLineages,
    facet_shared: bool,
    coord_flipped: bool,
    layer_reads: &[AuxRead<'_>],
) -> Result<(Vec<String>, Vec<crate::render::RenderWarning>), RenderError> {
    // A layered spec's every read draws its field, so the union is `None`
    // only for an unlayered spec, which reads its one batch.
    let (layer_categories, domain_batch, lead) = match aux_category_union(layer_reads, batch, transform_outputs, facet_shared)? {
        Some((categories, first_batch, read)) => (Some(categories), first_batch, Some(read)),
        None => (None, shared_categorical_batch(batch, field, transform_outputs, facet_shared), None),
    };
    let mut distinct = match (explicit_domain, layer_categories) {
        (Some(declared), _) => declared,
        (None, Some(categories)) => categories,
        (None, None) => distinct_values_in_order(domain_batch, field)?,
    };

    let mut warnings: Vec<crate::render::RenderWarning> = Vec::new();
    // `encoding` is the RENDERED (post-`CoordFlip`) encoding, so these two
    // fields name the rendered channels; `coord_flipped` is what lets an
    // authored `sort="-y"` still find the channel the user wrote (#95). A
    // shape/stroke-dash domain sorts by the same rule the positional axes
    // do, because it runs through the same `apply_sort_to_domain`.
    let sort_ctx = legend_sort_context(
        field,
        lead,
        domain_batch,
        transform_lineages,
        encoding.x.as_ref().map(|e| e.field.as_str()),
        encoding.y.as_ref().map(|e| e.field.as_str()),
        coord_flipped,
    );
    apply_sort_to_domain(&mut distinct, sort, &sort_ctx, &mut warnings)?;
    Ok((distinct, warnings))
}

/// The data extent of a numeric auxiliary channel (size, the opacity family)
/// bound to `field`. On a layered spec (`layer_reads` non-empty) it is the
/// union over every drawing layer's own column ([`sided_aux_numeric_union`])
/// that reads as numbers: the scale is continuous, so a later layer's categorical
/// column stays out, with one `LayerScaleExcluded` in `warnings`
/// ([`admitted_aux_reads`]). On an unlayered spec, `field` read from `batch`,
/// unioned on a faceted chart with the all-panels batch so marks scale
/// through the same domain as the legend. `channel` names the channel in a
/// dtype refusal and an exclusion.
///
/// `side`: a `Log` channel's side of zero (`None` for every other scale
/// type, byte-identical to before this parameter existed). Every path —
/// unlayered, unlayered-faceted, and layered alike — reads its
/// `left_out`/`left_out_zero`/`left_out_opposite` from the same sided fold
/// the positional union uses: the facet-shared case counts from the
/// all-panels batch ALONE when it carries the field (the whole frame's
/// count, not panel-plus-global), falling back to the panel's own count
/// exactly as the unsided path falls back to the panel's own extent; the
/// layered case folds EVERY drawing layer's own column through
/// [`sided_aux_numeric_union`], so a `Log` layered channel's left-out rows
/// are counted once, over the whole layer set, not per layer and not
/// dropped — the same whole-frame accounting x/y's
/// `numeric_domain_union_over` already gives a flat chart's layers.
#[allow(clippy::too_many_arguments)]
fn numeric_aux_extent(
    field: &str,
    channel: &'static str,
    batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    facet_shared: bool,
    layer_reads: &[AuxRead<'_>],
    side: Option<crate::scale::log::LogSide>,
    warnings: &mut Vec<crate::render::RenderWarning>,
) -> Result<(super::SidedExtent, super::LeftOutTally), RenderError> {
    let layer_reads = admitted_aux_reads(layer_reads, channel, false, batch, transform_outputs, warnings)?;
    let sided = |col: &dyn arrow::array::Array, field: &str| {
        sided_min_max_f64(col, side).map_err(|_| RenderError::UnsupportedDtype {
            field: field.to_string(),
            dtype: format!("{:?}", col.data_type()),
            context: Some(channel),
        })
    };
    let tally_of = |col: &dyn arrow::array::Array, extent: super::SidedExtent| {
        let mut tally = super::LeftOutTally::new();
        if col.null_count() != col.len() {
            tally.count(col, extent);
        }
        tally
    };
    if !layer_reads.is_empty() {
        return sided_aux_numeric_union(&layer_reads, batch, transform_outputs, facet_shared, side, channel)?
            .ok_or_else(|| RenderError::UnknownColumn { name: field.to_string() });
    }
    let col = batch
        .column_by_name(field)
        .ok_or_else(|| RenderError::UnknownColumn { name: field.to_string() })?;
    let panel = sided(col.as_ref(), field)?;
    if !facet_shared {
        return Ok((panel, tally_of(col.as_ref(), panel)));
    }
    // The all-panels batch already contains every panel's rows, so it ALONE
    // (not panel-plus-global, and NOT `union_panel_with_global_extent`,
    // which unions the two) is the shared extent — the same reading
    // `numeric_domain_union_over`'s facet-shared contributions give the
    // positional union. This is a deliberate behavior change from
    // `union_panel_with_global_extent`'s own rule, not a restatement of it:
    // an all-null faceted column now builds no scale and draws no legend,
    // exactly as the same column renders unfaceted, on every route
    // (unlayered-faceted here, and a layered-then-faceted chart through
    // `AuxRead::extent`'s identical global-alone read) — where
    // `union_panel_with_global_extent` fabricated a `(0.0, 1.0)` legend for
    // a column with no value anywhere, because its own all-null default
    // lives in `numeric_extent`, which this function never calls. Falls
    // back to the panel's own extent only when the global batch or the
    // field is absent (an unfaceted call, or a field not yet in `FINAL`).
    use crate::transform::core::FINAL_OUTPUT_KEY;
    let Some(global_batch) = transform_outputs.get(FINAL_OUTPUT_KEY) else { return Ok((panel, tally_of(col.as_ref(), panel))) };
    let Some(global_col) = global_batch.column_by_name(field) else { return Ok((panel, tally_of(col.as_ref(), panel))) };
    let extent = sided(global_col.as_ref(), field)?;
    Ok((extent, tally_of(global_col.as_ref(), extent)))
}

/// Whether a [`numeric_aux_extent`] holds any value. A column whose every
/// row is null (or NaN) reads as an inverted, unbounded extent; a channel
/// bound to it has nothing to map, so it builds no scale and every mark
/// draws at the channel's default.
fn extent_has_values((lo, hi): (f64, f64)) -> bool {
    lo.is_finite() && hi.is_finite() && lo <= hi
}

/// Build a SizeScale if `encoding.size` is present.
///
/// A linear (or, under an explicit `LogScale`/`PowScale`/`SqrtScale`, that
/// scale's own) map from the data extent (niced) onto
/// `[theme.sizes.point_size_min, theme.sizes.point_size_max]`, through the
/// [`numeric_aux_scale`] gate: an explicit `scale=` domain replaces the extent,
/// an explicit range replaces the band (clamped to a non-negative area), and
/// `reverse`/`nice`/`zero` are honored (`zero` only on `LinearScale`, its own
/// field). The scale clamps, so a value outside an explicit domain draws at
/// the nearer end of the range.
///
/// `facet_shared`: when `true` (chart is faceted with no independent option for
/// size), unions `batch`'s extent with the global `FINAL_OUTPUT_KEY` batch so
/// that per-panel marks normalize against the same domain as the global legend.
/// Non-faceted callers pass `false`; the per-panel-only path is byte-identical.
///
/// Returns the scale (if built) and a vec of warnings: one
/// `LayerScaleExcluded` per later layer whose own size field reads as
/// categories, left out of the domain ([`numeric_aux_extent`]).
///
/// `composite_domain` is the 10-pre-b composite seam: `Some` only for a composite
/// leaf whose parent shares `size`. Its [`SharedDomain::Numeric`] extent (unioned
/// across the composition's leaves) replaces the per-leaf `[min, max]` so every
/// leaf's marks and legend normalize through the same domain. `None` (every
/// standalone and faceted caller) reproduces the pre-10-pre-b path byte-for-byte.
///
/// `layer_reads` are a layered spec's reads of this channel ([`AuxRead`]): the
/// data extent is then the union over every layer's own column
/// ([`sided_aux_numeric_union`]). Empty for an unlayered spec, which reads `batch`.
pub fn build_size_scale(
    encoding: &crate::spec::encoding::Encoding,
    batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    facet_shared: bool,
    theme: &ThemeInputs,
    composite_domain: Option<&SharedDomain>,
    layer_reads: &[AuxRead<'_>],
) -> Result<(Option<SizeScale>, Vec<crate::render::RenderWarning>), RenderError> {
    let Some(size_enc) = &encoding.size else {
        return Ok((None, Vec::new()));
    };
    let overrides = numeric_aux_scale("size", &size_enc.scale)?;
    let mut warnings = Vec::new();
    let (extent, tally) = numeric_aux_extent(
        &size_enc.field,
        "size",
        batch,
        transform_outputs,
        facet_shared,
        layer_reads,
        overrides.log_side(),
        &mut warnings,
    )?;
    // 10-pre-b: a composite shared size domain (unioned across the composition's
    // leaves) overrides the per-leaf extent. The union already subsumes this
    // leaf's own extent (and, for a shared `Log` domain, already counted its
    // own off-side rows when it was built), so overriding with an empty
    // tally — like an imposed positional member — is correct.
    let (extent, tally) = match composite_domain {
        Some(SharedDomain::Numeric { lo, hi }) => (
            super::SidedExtent { lo: *lo, hi: *hi, left_out: 0, left_out_zero: false, left_out_opposite: None },
            super::LeftOutTally::new(),
        ),
        _ => (extent, tally),
    };
    // The guard `build_opacity_channel_scale`/
    // `build_stroke_width_scale` already have — a column with no value
    // anywhere in the frame (and nothing left out, so this is a genuinely
    // all-null column, not a `Log` channel that needs `numeric_aux_scale_kind`
    // to refuse it) builds no scale, matching `LinearScale`'s own no-value
    // path and every other numeric auxiliary channel.
    if !extent_has_values((extent.lo, extent.hi)) && tally.rows() == 0 {
        return Ok((None, warnings));
    }
    let band = (theme.sizes.point_size_min, theme.sizes.point_size_max);
    let inner = numeric_aux_scale_kind("size", &size_enc.field, extent, &tally, overrides, band, DrawableInterval::NonNegative, true, &mut warnings)?;
    Ok((Some(SizeScale { inner }), warnings))
}

/// Build a ShapeScale if `encoding.shape` is present.
///
/// Returns the scale (if built) and a vec of warnings (palette overflow and/or
/// sort warnings). Mirrors `build_color_scale`'s `Vec<RenderWarning>` return so
/// the only caller (`build_auxiliary_scales`) can use `warnings.extend(...)` for
/// both channels consistently.
///
/// `facet_shared`: when `true` (chart is faceted), resolves the categorical
/// domain from the global `FINAL_OUTPUT_KEY` batch so that every panel assigns
/// the same glyph to the same category string — matching the global shape legend.
/// Falls back to `batch` when the global batch or field is absent.
/// Non-faceted callers pass `false`; the per-panel-only path is byte-identical.
///
/// `layer_reads` are a layered spec's reads of `shape` ([`AuxRead`]): the
/// domain is then every drawing layer's categories in layer order. A shape
/// scale reads every column as categories, so no layer's column conflicts
/// with it and every read is admitted. Empty for an unlayered spec.
pub fn build_shape_scale(
    encoding: &crate::spec::encoding::Encoding,
    batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    transform_lineages: &crate::transform::run::OutputLineages,
    facet_shared: bool,
    // `spec.coord` is `CoordFlip`: `encoding` above is the RENDERED encoding,
    // so an authored `sort="-y"` shorthand needs this to find the channel the
    // user wrote (#95). `false` for every unflipped chart → byte-identical.
    coord_flipped: bool,
    layer_reads: &[AuxRead<'_>],
) -> Result<(Option<ShapeScale>, Vec<crate::render::RenderWarning>), RenderError> {
    let Some(shape_enc) = &encoding.shape else {
        return Ok((None, Vec::new()));
    };
    // T3-shape: when faceted (Shared), resolve the domain from the global
    // FINAL_OUTPUT_KEY batch so every panel's glyph assignment matches the
    // global shape legend. Falls back to `batch` when the global batch or
    // field is absent (non-faceted path is byte-identical: facet_shared=false).
    //
    // KG-8: honors `encoding.shape.sort` by reordering the domain, mirroring
    // the categorical color path in color.rs. When no sort is set the domain
    // stays in first-appearance order (byte-identical to pre-KG-8).
    //
    // #123 D1: an explicit `scale.domain` (`explicit_ordinal_domain`) wins
    // over the data-derived order — see [`sorted_categorical_domain`]'s doc
    // comment for why this is unconditional, unlike color's range-gated D1.
    let (distinct, mut warnings) = sorted_categorical_domain(
        encoding,
        &shape_enc.field,
        explicit_ordinal_domain(shape_enc),
        shape_enc.sort.as_ref(),
        batch,
        transform_outputs,
        transform_lineages,
        facet_shared,
        coord_flipped,
        layer_reads,
    )?;

    // #123 D1: an explicit ordinal range assigns glyphs by the named shapes
    // (cycling positionally against `distinct`) instead of the fixed
    // `SHAPE_PALETTE`. All-or-nothing, mirroring `ColorRangeParseFailure`'s
    // rule (#126): one unrecognized entry refuses the whole range rather
    // than silently drawing every unmatched glyph as a circle. Parsed
    // through the strict point-shape table
    // ([`crate::render::marks::point::shape_from_str_strict`], no circle
    // fallback) — `Shape(scale=...)`'s primary gate is Python's own eager
    // `_VALID_POINT_SHAPES` check at construction; this is the defense for a
    // hand-built spec that bypasses it.
    //
    // Reads [`explicit_ordinal_range`] (every declared entry, whatever its
    // type): a range that mixes in a non-string entry is declared but
    // malformed, not absent, so it refuses like an unrecognized name instead
    // of falling through to the default palette.
    if let Some(range) = explicit_ordinal_range(shape_enc) {
        let range_shapes: Vec<ShapeKind> = range
            .iter()
            .map(|entry| {
                entry
                    .as_str()
                    .and_then(crate::render::marks::point::shape_from_str_strict)
                    .ok_or_else(|| RenderError::ShapeRangeEntryUnrecognized {
                        entry: entry.to_string(),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        // No overflow warning: the user supplied the range explicitly and
        // owns the mapping, mirroring color's explicit-range carve-out.
        let shapes: Vec<ShapeKind> = (0..distinct.len())
            .map(|i| range_shapes[i % range_shapes.len()])
            .collect();
        return Ok((Some(ShapeScale { domain: distinct, shapes }), warnings));
    }

    if distinct.len() > SHAPE_PALETTE.len() {
        warnings.push(crate::render::RenderWarning::ShapePaletteOverflowed {
            categories: distinct.len() as u32,
        });
    }
    let shapes: Vec<ShapeKind> = (0..distinct.len())
        .map(|i| SHAPE_PALETTE[i % SHAPE_PALETTE.len()])
        .collect();
    Ok((Some(ShapeScale { domain: distinct, shapes }), warnings))
}

/// Which opacity-family encoding channel an [`OpacityScale`] is built for.
///
/// The three channels resolve identically — quantitative field, data extent (or
/// the `scale=` domain) onto the theme opacity band (or the `scale=` range) —
/// so they share one builder rather than three drifting copies. The variant
/// selects the encoding slot and names the channel in error messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpacityChannel {
    /// `opacity`: whole-mark alpha.
    Opacity,
    /// `fill_opacity`: SVG `fill-opacity`.
    Fill,
    /// `stroke_opacity`: SVG `stroke-opacity`.
    Stroke,
}

impl OpacityChannel {
    /// This channel's slot on an encoding.
    pub(in crate::render) fn slot(self, encoding: &Encoding) -> &Option<EncodingSpec> {
        match self {
            OpacityChannel::Opacity => &encoding.opacity,
            OpacityChannel::Fill => &encoding.fill_opacity,
            OpacityChannel::Stroke => &encoding.stroke_opacity,
        }
    }

    /// The channel's wire name, carried in `UnsupportedDtype`'s `context`.
    pub(in crate::render) fn context(self) -> &'static str {
        match self {
            OpacityChannel::Opacity => "opacity",
            OpacityChannel::Fill => "fill_opacity",
            OpacityChannel::Stroke => "stroke_opacity",
        }
    }
}

/// Build the [`OpacityScale`] for one opacity-family channel, if it is bound.
///
/// Domain: the column's data extent, unless the encoding's `scale=` carries an
/// explicit `[lo, hi]` domain, which wins outright (mirroring the positional
/// explicit-scale bypass). Range: the theme opacity band
/// `[sizes.opacity_min, sizes.opacity_max]`, unless `scale=` carries an explicit
/// `[lo, hi]` range (spec §4.3), clamped to a legal alpha. The `scale=` passes
/// the [`numeric_aux_scale`] gate, which also accepts `LogScale`/`PowScale`/
/// `SqrtScale` (resolved through that scale's own transform instead of
/// linear), so `reverse`/`nice` are honored (`zero` only on `LinearScale`)
/// and a scale of any other type refuses.
///
/// `layer_reads`: a layered spec's reads of this channel ([`AuxRead`]), whose
/// union over every layer's own column is the data extent
/// ([`sided_aux_numeric_union`]); empty for an unlayered spec, which reads `batch`.
///
/// `facet_shared`: when `true` (chart is faceted — these channels have no
/// independent option), unions `batch`'s extent with the global
/// `FINAL_OUTPUT_KEY` batch so that per-panel marks normalize against the same
/// domain as the global legend. Non-faceted callers pass `false`; the
/// per-panel-only path is byte-identical.
///
/// A column with no value (every row null) builds no scale: each mark draws
/// at the channel's default and no legend reads it. A constant column follows
/// the degenerate-domain rule the size and stroke-width scales follow (and
/// Vega-Lite's linear scales): every row maps to the middle of the range, on
/// all three channels alike.
///
/// Returns the scale (if built) and a vec of warnings.
#[allow(clippy::too_many_arguments)]
pub fn build_opacity_channel_scale(
    channel: OpacityChannel,
    encoding: &Encoding,
    batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    facet_shared: bool,
    theme: &ThemeInputs,
    layer_reads: &[AuxRead<'_>],
) -> Result<(Option<OpacityScale>, Vec<crate::render::RenderWarning>), RenderError> {
    let Some(op_enc) = channel.slot(encoding) else {
        return Ok((None, Vec::new()));
    };
    let overrides = numeric_aux_scale(channel.context(), &op_enc.scale)?;
    let mut warnings: Vec<crate::render::RenderWarning> = Vec::new();
    let (extent, tally) = numeric_aux_extent(
        &op_enc.field,
        channel.context(),
        batch,
        transform_outputs,
        facet_shared,
        layer_reads,
        overrides.log_side(),
        &mut warnings,
    )?;
    // Genuinely no value anywhere (every row null, nothing left out either):
    // no scale, the channel's default draws. A `Log` channel whose on-side
    // extent is empty but the whole-frame tally left rows out falls through
    // instead, so `numeric_aux_scale_kind` can refuse it with
    // `RenderError::LogScaleNoValueOnSide` rather than silently taking this
    // no-scale path.
    if !extent_has_values((extent.lo, extent.hi)) && tally.rows() == 0 {
        return Ok((None, warnings));
    }
    // An explicit `scale=` range is honored as an alpha, so its endpoints are
    // clamped to a legal alpha range before the scale is built: an unclamped
    // `range=[0, 5]` would let the GPU-packed instance buffer carry alpha 5.0
    // while svg.rs omits the attribute at >= 1.0, and the two backends would
    // disagree.
    let band = (theme.sizes.opacity_min, theme.sizes.opacity_max);
    let inner = numeric_aux_scale_kind(channel.context(), &op_enc.field, extent, &tally, overrides, band, DrawableInterval::Alpha, false, &mut warnings)?;
    Ok((Some(OpacityScale { inner }), warnings))
}

/// Build a [`StrokeDashScale`] if `encoding.stroke_dash` is bound to a
/// *categorical* field.
///
/// A quantitative `stroke_dash` field keeps the numeric palette-index contract
/// (`ferrum_scene::numeric_index_dash`: 0 solid, 1–3 patterns) and resolves
/// NO scale — this returns `None` for it, which is
/// what tells mark builders to read the column as indices. A nominal/ordinal
/// field (including a numeric column the user typed `"N"`) resolves here
/// instead. See [`StrokeDashScale`] for the contract.
///
/// `facet_shared`: when `true`, resolves the domain from the global
/// `FINAL_OUTPUT_KEY` batch so every panel assigns the same dash to the same
/// category — matching the global legend. Mirrors [`build_shape_scale`].
///
/// `layer_reads` are a layered spec's reads of `stroke_dash` ([`AuxRead`]).
/// The establishing layer's column, read from the batch it draws, decides
/// between the categorical scale and the index palette; a later layer whose
/// column reads the other way is left out, with one `LayerScaleExcluded`
/// first among the warnings ([`admitted_aux_reads`]), and a categorical
/// domain is every admitted layer's categories in layer order. Empty for an
/// unlayered spec, which reads `batch`.
pub fn build_stroke_dash_scale(
    encoding: &Encoding,
    batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    transform_lineages: &crate::transform::run::OutputLineages,
    facet_shared: bool,
    // See [`build_shape_scale`]'s `coord_flipped` — the same authored-vs-
    // rendered channel question, same answer (#95).
    coord_flipped: bool,
    layer_reads: &[AuxRead<'_>],
) -> Result<(Option<StrokeDashScale>, Vec<crate::render::RenderWarning>), RenderError> {
    let Some(dash_enc) = &encoding.stroke_dash else {
        return Ok((None, Vec::new()));
    };
    // #123: an explicit `scale.range` is refused outright, regardless of the
    // field's dtype: `stroke_dash`'s output is the categorical scale's own
    // fixed legacy dash-index range (`STROKE_DASH_LEGACY_SLOTS`, not the
    // full, longer `DASH_PALETTE` table), not a value a scale can redeclare,
    // so there is no entry-by-entry parse (unlike `shape`'s range, above);
    // the refusal echoes the received range. Python's `StrokeDash._validate`
    // refuses `range=` at construction, so only a hand-built spec
    // (`ChartSpec::from_json` and similar) reaches this.
    if let Some(range) = explicit_ordinal_range(dash_enc) {
        return Err(RenderError::StrokeDashRangeUnsupported {
            entries: range.iter().map(|v| v.to_string()).collect(),
        });
    }
    let col = match layer_reads.first() {
        Some(lead) => lead.locate(batch, transform_outputs).map(|located| located.col),
        None => batch.column_by_name(&dash_enc.field).map(|col| col.as_ref()),
    }
    .ok_or_else(|| RenderError::UnknownColumn { name: dash_enc.field.clone() })?;
    let categorical =
        infer_spec_type(dash_enc, col.data_type()) != crate::spec::encoding::DataType::Quantitative;
    let mut warnings = Vec::new();
    let layer_reads = admitted_aux_reads(layer_reads, "stroke_dash", categorical, batch, transform_outputs, &mut warnings)?;
    if !categorical {
        // Numeric column: palette indices, no categorical resolution.
        return Ok((None, warnings));
    }

    // #123 D1: an explicit `scale.domain` wins over the data-derived order —
    // see [`sorted_categorical_domain`]'s doc comment. `stroke_dash` has no
    // explicit-range case to reach here (refused above), so only the domain
    // override threads through.
    let (distinct, sort_warnings) = sorted_categorical_domain(
        encoding,
        &dash_enc.field,
        explicit_ordinal_domain(dash_enc),
        dash_enc.sort.as_ref(),
        batch,
        transform_outputs,
        transform_lineages,
        facet_shared,
        coord_flipped,
        &layer_reads,
    )?;
    warnings.extend(sort_warnings);

    if distinct.len() > STROKE_DASH_LEGACY_SLOTS {
        warnings.push(crate::render::RenderWarning::StrokeDashPaletteOverflowed {
            categories: distinct.len() as u32,
        });
    }
    let patterns: Vec<Vec<f64>> = (0..distinct.len())
        .map(|i| categorical_slot_dash(i).unwrap_or_default())
        .collect();
    Ok((Some(StrokeDashScale { domain: distinct, patterns }), warnings))
}

/// Build the [`StrokeWidthScale`] if `encoding.stroke_width` is bound.
///
/// A linear (or, under an explicit `LogScale`/`PowScale`/`SqrtScale`, that
/// scale's own) map from the data extent onto the theme width band
/// `[sizes.line_stroke_width_min, sizes.line_stroke_width_max]`, as
/// [`build_size_scale`] maps onto `[sizes.point_size_min,
/// sizes.point_size_max]`: through the [`numeric_aux_scale`] gate, an explicit
/// `scale=` range overrides the band (clamped to a non-negative width), an
/// explicit domain overrides the extent, and `reverse`/`nice`/`zero` are
/// honored (`zero` only on `LinearScale`). The scale clamps, so a value
/// outside an explicit domain draws at the nearer end of the range. A
/// data-derived domain is niced as the size scale's is (on `LinearScale` and
/// `LogScale` alike); an explicit one is kept as written unless `nice` says
/// so. A constant column maps every row to the middle of the range (the
/// degenerate-domain rule the size scale follows), and a column with no
/// value builds no scale, so every mark draws at its default width.
///
/// `facet_shared` and `layer_reads` read the extent exactly as they do for
/// the size scale ([`numeric_aux_extent`]): a later layer whose column reads
/// as categories stays out with one `LayerScaleExcluded` in the warnings, and
/// a string column on the establishing layer refuses as an unsupported dtype.
pub fn build_stroke_width_scale(
    encoding: &Encoding,
    batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    facet_shared: bool,
    theme: &ThemeInputs,
    layer_reads: &[AuxRead<'_>],
) -> Result<(Option<StrokeWidthScale>, Vec<crate::render::RenderWarning>), RenderError> {
    let Some(width_enc) = &encoding.stroke_width else {
        return Ok((None, Vec::new()));
    };
    let overrides = numeric_aux_scale("stroke_width", &width_enc.scale)?;
    let mut warnings = Vec::new();
    let (extent, tally) = numeric_aux_extent(
        &width_enc.field,
        "stroke_width",
        batch,
        transform_outputs,
        facet_shared,
        layer_reads,
        overrides.log_side(),
        &mut warnings,
    )?;
    // See `build_opacity_channel_scale`'s identical guard: a `Log` channel
    // that left rows out falls through to `numeric_aux_scale_kind` instead
    // of taking this no-scale path, so it can refuse instead of silently
    // drawing every mark at its default width.
    if !extent_has_values((extent.lo, extent.hi)) && tally.rows() == 0 {
        return Ok((None, warnings));
    }
    let band = (theme.sizes.line_stroke_width_min, theme.sizes.line_stroke_width_max);
    let inner = numeric_aux_scale_kind("stroke_width", &width_enc.field, extent, &tally, overrides, band, DrawableInterval::NonNegative, true, &mut warnings)?;
    Ok((Some(StrokeWidthScale { inner }), warnings))
}
