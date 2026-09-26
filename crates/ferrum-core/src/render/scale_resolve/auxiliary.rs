//! Auxiliary scale resolution: size, shape, opacity-family, and stroke-dash
//! encodings.

use std::collections::HashMap;

use arrow::record_batch::RecordBatch;

use crate::layout::ThemeInputs;
use crate::scale::linear::LinearScale;

use ferrum_scene::{resolve_stroke_dash, STROKE_DASH_LEGACY_SLOTS};

use crate::render::RenderError;
use crate::spec::encoding::{Encoding, EncodingSpec, ScaleSpec};

use super::domain::{apply_sort_to_domain, explicit_ordinal_domain, explicit_ordinal_range, SortContext};
use super::{column_min_max_f64, distinct_values_in_order, infer_spec_type, shared_categorical_batch, union_panel_with_global_extent, OpacityScale, ScaleKind, ShapeKind, ShapeScale, SharedDomain, SizeScale, StrokeDashScale, SHAPE_PALETTE};

/// The explicit `domain`/`range` a user's `scale=` puts on a *numeric auxiliary*
/// channel (size, opacity family).
///
/// Only [`ScaleSpec::Linear`] contributes: these channels map a quantitative
/// field linearly onto a bounded output band, so a `Log`/`Pow`/… spec would
/// describe a curve this resolution does not apply, and honoring only its
/// endpoints would misreport the mapping. Every other spec (and a domain/range
/// that is not a two-element `[lo, hi]`) leaves the corresponding slot `None`,
/// which is the pre-batch-A behavior byte-for-byte.
#[derive(Debug, Clone, Copy, Default)]
struct LinearOverrides {
    domain: Option<(f64, f64)>,
    range: Option<(f64, f64)>,
}

/// Read the `[lo, hi]` overrides off an encoding's `scale=`.
fn linear_overrides(scale: &Option<ScaleSpec>) -> LinearOverrides {
    let Some(ScaleSpec::Linear { common, .. }) = scale else {
        return LinearOverrides::default();
    };
    LinearOverrides { domain: endpoints(&common.domain), range: endpoints(&common.range) }
}

/// A two-element `[lo, hi]` list as a pair; any other length is not an extent.
fn endpoints(values: &Option<Vec<f64>>) -> Option<(f64, f64)> {
    match values {
        Some(v) if v.len() == 2 => Some((v[0], v[1])),
        _ => None,
    }
}

/// A short kind name for a `ScaleSpec` variant, used to name the dropped scale
/// type in [`crate::render::RenderWarning::UnsupportedOpacityScale`] when an
/// opacity-family channel's `scale=` is not `Linear`. Always the `ScaleSpec`
/// wire `"type"` tag spelling ([`ScaleSpec::tag()`]), the same spelling
/// every other tag-producing site in the crate uses — pinned by
/// `non_linear_bin_ordinal_opacity_scale_warns_with_the_wire_tag_spelling`
/// in `render/scale_resolve/tests.rs`.
fn scale_spec_kind_name(spec: &ScaleSpec) -> &'static str {
    spec.tag()
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
) -> Result<(Vec<String>, Vec<crate::render::RenderWarning>), RenderError> {
    let domain_batch = shared_categorical_batch(batch, field, transform_outputs, facet_shared);
    let mut distinct = match explicit_domain {
        Some(declared) => declared,
        None => distinct_values_in_order(domain_batch, field)?,
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
    apply_sort_to_domain(&mut distinct, sort, &sort_ctx, &mut warnings);
    Ok((distinct, warnings))
}

/// Build a SizeScale if `encoding.size` is present.
///
/// Honors a user-supplied `scale.range` (Phase 10f); when absent, falls back
/// to `[theme.sizes.point_size_min, theme.sizes.point_size_max]`.
///
/// `facet_shared`: when `true` (chart is faceted with no independent option for
/// size), unions `batch`'s extent with the global `FINAL_OUTPUT_KEY` batch so
/// that per-panel marks normalize against the same domain as the global legend.
/// Non-faceted callers pass `false`; the per-panel-only path is byte-identical.
///
/// Returns the scale (if built) and a vec of warnings. Currently emits no
/// warnings; the `Vec` is returned to match `build_color_scale`/`build_shape_scale`
/// so `build_auxiliary_scales` can use `warnings.extend(...)` uniformly for all
/// four channels.
///
/// `composite_domain` is the 10-pre-b composite seam: `Some` only for a composite
/// leaf whose parent shares `size`. Its [`SharedDomain::Numeric`] extent (unioned
/// across the composition's leaves) replaces the per-leaf `[min, max]` so every
/// leaf's marks and legend normalize through the same domain. `None` (every
/// standalone and faceted caller) reproduces the pre-10-pre-b path byte-for-byte.
pub fn build_size_scale(
    encoding: &crate::spec::encoding::Encoding,
    batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    facet_shared: bool,
    theme: &ThemeInputs,
    composite_domain: Option<&SharedDomain>,
) -> Result<(Option<SizeScale>, Vec<crate::render::RenderWarning>), RenderError> {
    let Some(size_enc) = &encoding.size else {
        return Ok((None, Vec::new()));
    };
    let col = batch
        .column_by_name(&size_enc.field)
        .ok_or_else(|| RenderError::UnknownColumn { name: size_enc.field.clone() })?;
    let (min, max) = column_min_max_f64(col).map_err(|_| RenderError::UnsupportedDtype {
        field: size_enc.field.clone(),
        dtype: format!("{:?}", col.data_type()),
        context: Some("size"),
    })?;
    // T3: When faceted (Shared), union the per-panel extent with the global
    // FINAL_OUTPUT_KEY batch so marks scale through the same domain as the legend.
    let (min, max) = if facet_shared {
        union_panel_with_global_extent((min, max), &size_enc.field, transform_outputs)
    } else {
        (min, max)
    };
    // 10-pre-b: a composite shared size domain (unioned across the composition's
    // leaves) overrides the per-leaf extent. The union already subsumes this
    // leaf's own extent, so overriding is correct.
    let (min, max) = match composite_domain {
        Some(SharedDomain::Numeric { lo, hi }) => (*lo, *hi),
        _ => (min, max),
    };
    // Phase 10f: an explicit `scale.range` overrides the theme size band. The
    // explicit *domain* is deliberately NOT read here — size-scale behavior is a
    // batch-A non-goal (spec §3), so this stays the pre-batch-A resolution.
    let (lo, hi) = linear_overrides(&size_enc.scale)
        .range
        .unwrap_or((theme.sizes.point_size_min, theme.sizes.point_size_max));
    let inner = ScaleKind::Linear(LinearScale::new_internal(
        vec![min, max],
        vec![lo, hi],
        false,
        true,
    ));
    let _ = (lo, hi); // bounds now read from inner.pixel_range() via accessors
    Ok((Some(SizeScale { inner }), Vec::new()))
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
pub fn build_shape_scale(
    encoding: &crate::spec::encoding::Encoding,
    batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    facet_shared: bool,
    // `spec.coord` is `CoordFlip`: `encoding` above is the RENDERED encoding,
    // so an authored `sort="-y"` shorthand needs this to find the channel the
    // user wrote (#95). `false` for every unflipped chart → byte-identical.
    coord_flipped: bool,
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
    fn slot(self, encoding: &Encoding) -> &Option<EncodingSpec> {
        match self {
            OpacityChannel::Opacity => &encoding.opacity,
            OpacityChannel::Fill => &encoding.fill_opacity,
            OpacityChannel::Stroke => &encoding.stroke_opacity,
        }
    }

    /// Whether a *constant* column on this channel is a literal alpha the user
    /// asked for, rather than data to spread across the theme band.
    ///
    /// `fill_opacity`/`stroke_opacity` say yes: their column values ARE alphas
    /// in the channel's own units, so `fill_opacity="fo"` over a column of
    /// `0.3`s means "every mark at 0.3" — mapping that degenerate extent onto
    /// the band would silently repaint it at the band midpoint
    /// ([`LinearScale`]'s documented degenerate-domain rule, GH #104). Those
    /// charts must render byte-identically to pre-batch-A (spec §7), so no
    /// scale resolves and each row keeps its value.
    ///
    /// `opacity` says no: mapping a constant column to the band midpoint is its
    /// established pre-batch-A behavior, which batch A does not touch.
    fn constant_column_is_a_literal_alpha(self) -> bool {
        match self {
            OpacityChannel::Opacity => false,
            OpacityChannel::Fill | OpacityChannel::Stroke => true,
        }
    }

    /// The channel name carried in `UnsupportedDtype`'s `context`.
    fn context(self) -> &'static str {
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
/// `[lo, hi]` range (spec §4.3).
///
/// `facet_shared`: when `true` (chart is faceted — these channels have no
/// independent option), unions `batch`'s extent with the global
/// `FINAL_OUTPUT_KEY` batch so that per-panel marks normalize against the same
/// domain as the global legend. Non-faceted callers pass `false`; the
/// per-panel-only path is byte-identical.
///
/// Returns the scale (if built) and a vec of warnings.
///
/// *(Amended 2026-09-01, spec §4.3, T6 quality review — ruling 2):* a
/// `scale=` whose spec is present but not `Linear` (`Log`/`Pow`/`Sqrt`/…)
/// pushes a [`crate::render::RenderWarning::UnsupportedOpacityScale`] naming
/// the channel, the dropped scale kind, and which fallback the channel
/// actually took — never silent, and full non-linear opacity-curve support
/// stays a logged campaign follow-up.
///
/// *(Amended 2026-09-24, spec §4.2):* the fallback named in that warning is
/// decided at the SAME point the constant-column early return below decides
/// it, so a constant `fill_opacity`/`stroke_opacity` column is reported as
/// [`OpacityScaleFallback::LiteralAlpha`](crate::render::OpacityScaleFallback::LiteralAlpha)
/// rather than always claiming the default (data-extent-or-domain-override →
/// theme-band-or-range-override) linear resolution.
pub fn build_opacity_channel_scale(
    channel: OpacityChannel,
    encoding: &Encoding,
    batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    facet_shared: bool,
    theme: &ThemeInputs,
) -> Result<(Option<OpacityScale>, Vec<crate::render::RenderWarning>), RenderError> {
    let Some(op_enc) = channel.slot(encoding) else {
        return Ok((None, Vec::new()));
    };
    let col = batch
        .column_by_name(&op_enc.field)
        .ok_or_else(|| RenderError::UnknownColumn { name: op_enc.field.clone() })?;
    let (min, max) = column_min_max_f64(col).map_err(|_| RenderError::UnsupportedDtype {
        field: op_enc.field.clone(),
        dtype: format!("{:?}", col.data_type()),
        context: Some(channel.context()),
    })?;
    // T3: When faceted (Shared), union the per-panel extent with the global
    // FINAL_OUTPUT_KEY batch so marks scale through the same domain as the legend.
    let (min, max) = if facet_shared {
        union_panel_with_global_extent((min, max), &op_enc.field, transform_outputs)
    } else {
        (min, max)
    };
    let mut warnings: Vec<crate::render::RenderWarning> = Vec::new();
    let overrides = linear_overrides(&op_enc.scale);
    // A constant column has no spread to lay across a band; for the two
    // channels whose values are literal alphas that means no scale at all, so
    // each row keeps its own value. An explicit `scale=` domain is a real
    // domain even over a constant column, so it takes precedence over this.
    let is_literal_alpha =
        overrides.domain.is_none() && min == max && channel.constant_column_is_a_literal_alpha();
    if let Some(spec) = &op_enc.scale {
        if !matches!(spec, ScaleSpec::Linear { .. }) {
            let fallback = if is_literal_alpha {
                crate::render::OpacityScaleFallback::LiteralAlpha
            } else {
                crate::render::OpacityScaleFallback::Linear
            };
            warnings.push(crate::render::RenderWarning::UnsupportedOpacityScale {
                channel: channel.context().to_string(),
                scale_kind: scale_spec_kind_name(spec).to_string(),
                fallback,
            });
        }
    }
    if is_literal_alpha {
        return Ok((None, warnings));
    }
    let (min, max) = overrides.domain.unwrap_or((min, max));
    let (lo, hi) = overrides
        .range
        .unwrap_or((theme.sizes.opacity_min, theme.sizes.opacity_max));
    // Ruling 3: an explicit `scale=` range is honored as an alpha, so its
    // endpoints must themselves be a legal alpha range — an unclamped
    // `range=[0, 5]` would let the GPU-packed instance buffer carry alpha 5.0
    // (svg.rs only omits the attribute at >= 1.0, so SVG and WASM would
    // silently disagree). Clamping here, before the scale is built, keeps
    // both backends reading the same already-bounded value.
    let (lo, hi) = (lo.clamp(0.0, 1.0), hi.clamp(0.0, 1.0));
    let inner = ScaleKind::Linear(LinearScale::new_internal(
        vec![min, max],
        vec![lo, hi],
        true,
        false,
    ));
    Ok((Some(OpacityScale { inner }), warnings))
}

/// Build a [`StrokeDashScale`] if `encoding.stroke_dash` is bound to a
/// *categorical* field.
///
/// A quantitative `stroke_dash` field keeps the numeric palette-index contract
/// (`channels::resolve_row_stroke_dash`'s own legacy clamp: 0 solid, 1–3
/// patterns — `resolve_stroke_dash` itself resolves the full 0..=7 palette;
/// the legacy range is that one caller's own clamp, not a limit of the
/// function) and resolves NO scale — this returns `None` for it, which is
/// what tells mark builders to read the column as indices. A nominal/ordinal
/// field (including a numeric column the user typed `"N"`) resolves here
/// instead. See [`StrokeDashScale`] for the contract.
///
/// `facet_shared`: when `true`, resolves the domain from the global
/// `FINAL_OUTPUT_KEY` batch so every panel assigns the same dash to the same
/// category — matching the global legend. Mirrors [`build_shape_scale`].
pub fn build_stroke_dash_scale(
    encoding: &Encoding,
    batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    facet_shared: bool,
    // See [`build_shape_scale`]'s `coord_flipped` — the same authored-vs-
    // rendered channel question, same answer (#95).
    coord_flipped: bool,
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
    let col = batch
        .column_by_name(&dash_enc.field)
        .ok_or_else(|| RenderError::UnknownColumn { name: dash_enc.field.clone() })?;
    if infer_spec_type(dash_enc, col.data_type()) == crate::spec::encoding::DataType::Quantitative {
        // Numeric column: palette indices, no categorical resolution.
        return Ok((None, Vec::new()));
    }

    // #123 D1: an explicit `scale.domain` wins over the data-derived order —
    // see [`sorted_categorical_domain`]'s doc comment. `stroke_dash` has no
    // explicit-range case to reach here (refused above), so only the domain
    // override threads through.
    let (distinct, mut warnings) = sorted_categorical_domain(
        encoding,
        &dash_enc.field,
        explicit_ordinal_domain(dash_enc),
        dash_enc.sort.as_ref(),
        batch,
        transform_outputs,
        facet_shared,
        coord_flipped,
    )?;

    // The categorical scale's cycling slot count is pinned to the legacy
    // 4-slot span (solid plus the three pre-append patterns), independent of
    // `DASH_PALETTE`'s full length: see `STROKE_DASH_LEGACY_SLOTS`'s doc
    // comment for why appending the named linetypes must not move which
    // pattern an existing categorical domain gets.
    let slots = STROKE_DASH_LEGACY_SLOTS;
    if distinct.len() > slots {
        warnings.push(crate::render::RenderWarning::StrokeDashPaletteOverflowed {
            categories: distinct.len() as u32,
        });
    }
    let patterns: Vec<Vec<f64>> = (0..distinct.len())
        .map(|i| resolve_stroke_dash((i % slots) as f64).unwrap_or_default())
        .collect();
    Ok((Some(StrokeDashScale { domain: distinct, patterns }), warnings))
}

