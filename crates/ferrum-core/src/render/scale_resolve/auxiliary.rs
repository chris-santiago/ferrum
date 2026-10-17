//! Auxiliary scale resolution: size, shape, opacity-family, and stroke-dash
//! encodings.

use std::collections::HashMap;

use arrow::record_batch::RecordBatch;

use crate::layout::ThemeInputs;
use crate::scale::linear::LinearScale;

use ferrum_scene::{categorical_slot_dash, STROKE_DASH_LEGACY_SLOTS};

use crate::render::RenderError;
use crate::spec::encoding::{Encoding, EncodingSpec, ScaleSpec};

use super::domain::{admitted_aux_reads, aux_category_union, aux_numeric_union, apply_sort_to_domain, explicit_ordinal_domain, explicit_ordinal_range, AuxRead, SortContext};
use super::{column_min_max_f64, distinct_values_in_order, infer_spec_type, shared_categorical_batch, union_panel_with_global_extent, OpacityScale, ScaleKind, ShapeKind, ShapeScale, SharedDomain, SizeScale, StrokeDashScale, StrokeWidthScale, SHAPE_PALETTE};

/// The channels whose scale maps a quantitative field linearly onto a
/// bounded output interval: the one gate [`numeric_aux_scale`] answers for.
pub(crate) const NUMERIC_AUX_CHANNELS: [&str; 5] = ["size", "opacity", "fill_opacity", "stroke_opacity", "stroke_width"];

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
    /// `nice=`: the domain is niced, on top of the channel's own default.
    nice: bool,
    /// `zero=`: the domain widens to include zero.
    zero: bool,
}

/// The one gate a numeric auxiliary channel's `scale=` passes. These scales
/// map linearly onto a bounded interval, so a scale of another type, a
/// `padding` (an axis inset) or a `scheme` (a color scheme) is something this
/// resolution would ignore or misread, and refuses with
/// [`RenderError::AuxiliaryScaleUnsupported`]; a domain or range that is not
/// `[lo, hi]` refuses with the positional sentence
/// ([`super::positional::validate_explicit_continuous_shape`]). Every other
/// `linear` field is honored ([`NumericAuxOverrides`]). `channel` is the
/// channel's wire name. No `scale=` admits the channel's defaults.
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
    let ScaleSpec::Linear { common, nice, zero, system_injected: _ } = spec else {
        return Err(unsupported(format!("type '{}' is not supported", spec.tag())));
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
        nice: *nice,
        zero: *zero,
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

/// The linear scale of a numeric auxiliary channel: `extent` (the data's, or
/// a composite's shared one) unless the scale states a domain, widened to
/// zero when asked, onto `band` unless the scale states a range, that range
/// clamped into `interval` and then reversed when asked. A data-derived
/// domain is niced when `nice_data_domain` (the channel's default) or the
/// scale's `nice` says so; an explicit one only when the scale says so.
fn numeric_aux_linear(
    extent: (f64, f64),
    overrides: NumericAuxOverrides,
    band: (f64, f64),
    interval: DrawableInterval,
    nice_data_domain: bool,
) -> ScaleKind {
    let (mut lo_d, mut hi_d) = overrides.domain.unwrap_or(extent);
    if overrides.zero {
        (lo_d, hi_d) = super::positional::extend_extent_to_zero(lo_d, hi_d);
    }
    let (lo, hi) = interval.clamp(overrides.range.unwrap_or(band));
    let (lo, hi) = if overrides.reverse { (hi, lo) } else { (lo, hi) };
    let nice = overrides.nice || (nice_data_domain && overrides.domain.is_none());
    ScaleKind::Linear(LinearScale::new_internal(vec![lo_d, hi_d], vec![lo, hi], true, nice))
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
/// `sort=` aggregates over the first contributing batch. Empty for an
/// unlayered spec, which reads `batch`.
#[allow(clippy::too_many_arguments)]
fn sorted_categorical_domain(
    encoding: &Encoding,
    field: &str,
    explicit_domain: Option<Vec<String>>,
    sort: Option<&serde_json::Value>,
    batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    facet_shared: bool,
    coord_flipped: bool,
    layer_reads: &[AuxRead<'_>],
) -> Result<(Vec<String>, Vec<crate::render::RenderWarning>), RenderError> {
    let (layer_categories, domain_batch) = match aux_category_union(layer_reads, batch, transform_outputs, facet_shared)? {
        Some((categories, first_batch)) => (Some(categories), first_batch),
        None => (None, shared_categorical_batch(batch, field, transform_outputs, facet_shared)),
    };
    let mut distinct = match (explicit_domain, layer_categories) {
        (Some(declared), _) => declared,
        (None, Some(categories)) => categories,
        (None, None) => distinct_values_in_order(domain_batch, field)?,
    };

    let mut warnings: Vec<crate::render::RenderWarning> = Vec::new();
    let sort_ctx = SortContext {
        category_field: field,
        batch: domain_batch,
        // `encoding` is the RENDERED (post-`CoordFlip`) encoding, so these two
        // fields name the rendered channels; `coord_flipped` is what lets an
        // authored `sort="-y"` still find the channel the user wrote (#95).
        // A shape/stroke-dash domain sorts by the same rule the positional
        // axes do, because it runs through the same `apply_sort_to_domain`.
        x_field: encoding.x.as_ref().map(|e| e.field.as_str()),
        y_field: encoding.y.as_ref().map(|e| e.field.as_str()),
        coord_flipped,
    };
    apply_sort_to_domain(&mut distinct, sort, &sort_ctx, &mut warnings)?;
    Ok((distinct, warnings))
}

/// The data extent of a numeric auxiliary channel (size, the opacity family)
/// bound to `field`. On a layered spec (`layer_reads` non-empty) it is the
/// union over every layer's own column ([`aux_numeric_union`]) that reads as
/// numbers: the scale is continuous, so a later layer's categorical column
/// stays out, with one `LayerScaleExcluded` in `warnings`
/// ([`admitted_aux_reads`]). Otherwise, or when no layer's batch holds its
/// field, `field` read from `batch`, unioned on a faceted chart with the
/// all-panels batch so marks scale through the same domain as the legend.
/// `channel` names the channel in a dtype refusal and an exclusion.
fn numeric_aux_extent(
    field: &str,
    channel: &'static str,
    batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    facet_shared: bool,
    layer_reads: &[AuxRead<'_>],
    warnings: &mut Vec<crate::render::RenderWarning>,
) -> Result<(f64, f64), RenderError> {
    let layer_reads = admitted_aux_reads(layer_reads, channel, false, batch, transform_outputs, warnings);
    let extent = |col: &dyn arrow::array::Array, field: &str| {
        column_min_max_f64(col).map_err(|_| RenderError::UnsupportedDtype {
            field: field.to_string(),
            dtype: format!("{:?}", col.data_type()),
            context: Some(channel),
        })
    };
    if let Some(union) = aux_numeric_union(&layer_reads, batch, transform_outputs, facet_shared, extent)? {
        return Ok(union);
    }
    let col = batch
        .column_by_name(field)
        .ok_or_else(|| RenderError::UnknownColumn { name: field.to_string() })?;
    let panel = extent(col.as_ref(), field)?;
    Ok(if facet_shared { union_panel_with_global_extent(panel, field, transform_outputs) } else { panel })
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
/// A linear map from the data extent (niced) onto
/// `[theme.sizes.point_size_min, theme.sizes.point_size_max]`, through the
/// [`numeric_aux_scale`] gate: an explicit `scale=` domain replaces the extent,
/// an explicit range replaces the band (clamped to a non-negative area), and
/// `reverse`/`nice`/`zero` are honored. The scale clamps, so a value outside
/// an explicit domain draws at the nearer end of the range.
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
/// ([`aux_numeric_union`]). Empty for an unlayered spec, which reads `batch`.
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
    let (min, max) = numeric_aux_extent(
        &size_enc.field,
        "size",
        batch,
        transform_outputs,
        facet_shared,
        layer_reads,
        &mut warnings,
    )?;
    // 10-pre-b: a composite shared size domain (unioned across the composition's
    // leaves) overrides the per-leaf extent. The union already subsumes this
    // leaf's own extent, so overriding is correct.
    let (min, max) = match composite_domain {
        Some(SharedDomain::Numeric { lo, hi }) => (*lo, *hi),
        _ => (min, max),
    };
    let band = (theme.sizes.point_size_min, theme.sizes.point_size_max);
    let inner = numeric_aux_linear((min, max), overrides, band, DrawableInterval::NonNegative, true);
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
/// the [`numeric_aux_scale`] gate, so `reverse`/`nice`/`zero` are honored and a
/// non-linear scale refuses.
///
/// `layer_reads`: a layered spec's reads of this channel ([`AuxRead`]), whose
/// union over every layer's own column is the data extent
/// ([`aux_numeric_union`]); empty for an unlayered spec, which reads `batch`.
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
    let (min, max) = numeric_aux_extent(
        &op_enc.field,
        channel.context(),
        batch,
        transform_outputs,
        facet_shared,
        layer_reads,
        &mut warnings,
    )?;
    if !extent_has_values((min, max)) {
        return Ok((None, warnings));
    }
    // An explicit `scale=` range is honored as an alpha, so its endpoints are
    // clamped to a legal alpha range before the scale is built: an unclamped
    // `range=[0, 5]` would let the GPU-packed instance buffer carry alpha 5.0
    // while svg.rs omits the attribute at >= 1.0, and the two backends would
    // disagree.
    let band = (theme.sizes.opacity_min, theme.sizes.opacity_max);
    let inner = numeric_aux_linear((min, max), overrides, band, DrawableInterval::Alpha, false);
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
    let col = match layer_reads.first().and_then(|lead| lead.locate(batch, transform_outputs)) {
        Some(located) => located.col,
        None => batch
            .column_by_name(&dash_enc.field)
            .ok_or_else(|| RenderError::UnknownColumn { name: dash_enc.field.clone() })?
            .as_ref(),
    };
    let categorical =
        infer_spec_type(dash_enc, col.data_type()) != crate::spec::encoding::DataType::Quantitative;
    let mut warnings = Vec::new();
    let layer_reads = admitted_aux_reads(layer_reads, "stroke_dash", categorical, batch, transform_outputs, &mut warnings);
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
/// A linear map from the data extent onto the theme width band
/// `[sizes.line_stroke_width_min, sizes.line_stroke_width_max]`, as
/// [`build_size_scale`] maps onto `[sizes.point_size_min,
/// sizes.point_size_max]`: through the [`numeric_aux_scale`] gate, an explicit
/// `scale=` range overrides the band (clamped to a non-negative width), an
/// explicit domain overrides the extent, and `reverse`/`nice`/`zero` are
/// honored. The scale clamps, so a value outside an explicit domain draws at
/// the nearer end of the range. A data-derived domain is niced as the size
/// scale's is; an explicit one is kept as written unless `nice` says so. A constant column maps
/// every row to the middle of the range (the degenerate-domain rule the size
/// scale follows), and a column with no value builds no scale, so every mark
/// draws at its default width.
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
    let extent = numeric_aux_extent(
        &width_enc.field,
        "stroke_width",
        batch,
        transform_outputs,
        facet_shared,
        layer_reads,
        &mut warnings,
    )?;
    if !extent_has_values(extent) {
        return Ok((None, warnings));
    }
    let band = (theme.sizes.line_stroke_width_min, theme.sizes.line_stroke_width_max);
    let inner = numeric_aux_linear(extent, overrides, band, DrawableInterval::NonNegative, true);
    Ok((Some(StrokeWidthScale { inner }), warnings))
}
