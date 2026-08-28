//! Phase 9c — position-adjustment render pass.
//!
//! Rewrites a layer's RecordBatch *data values* (or injects synthetic offset
//! columns, for ordinal x) per the PositionAdjust on the layer. Runs AFTER
//! scale_resolve (so we know ordinal bandwidth or continuous-x median spacing)
//! but BEFORE mark drawing. The adjusted RecordBatch is then passed to
//! `draw::dispatch_mark` in place of the original.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};

use crate::render::scale_resolve::{ResolvedScales, ScaleKind};
use crate::render::RenderWarning;
use crate::spec::chart::ChartSpec;
use crate::spec::position::{PositionAdjust, StackAnchor, StackOffset, StackValueAxis};

/// Return the batch the y-scale should resolve against, accounting for a
/// Stack position adjustment.
///
/// When a layer (or the single-layer spec) carries a `Stack` adjustment
/// whose encoded y matches `y_field` AND whose resolved value axis (GH #77:
/// [`resolve_stack_value_on_x`], using this spec's real `coord_flipped` —
/// `rendering_spec.coord` survives the CoordFlip encoding swap unchanged, so
/// it always reflects the true flip state here) is Y, the rendered y values
/// are the *cumulative* values from `apply_stack`, not the original column.
/// Resolving the y-scale from the raw batch would clip stacked tops outside
/// the domain — `LinearScale` returns NaN for out-of-domain inputs and
/// `bar.rs` drops every row whose top falls past it. Returning the
/// post-stack batch here keeps stacked bars visible.
///
/// Borrowed `primary_batch` is returned when no Stack matches, or when a
/// matching Stack's value axis resolves to X instead (GH #77 follow-up —
/// widening belongs to `axis_batch_for_x` in that case, not here). The owned
/// stacked batch is returned (boxed by `Cow`) when the value axis resolves
/// to Y. On a stack failure the caller's primary batch is returned — the
/// scale resolves from raw data and the downstream `apply_stack` re-attempt
/// during drawing will surface the same error to the user.
///
/// Pre-F15 this logic lived in `scale_resolve::resolve_scales_with_outputs`
/// via a private `find_stack_for_y` helper. The Stack handling belongs
/// alongside the other Stack code; scale resolution shouldn't have to
/// know which specific position adjustment is in play.
pub(crate) fn axis_batch_for_y<'a>(
    spec: &'a ChartSpec,
    y_field: &str,
    primary_batch: &'a RecordBatch,
) -> Cow<'a, RecordBatch> {
    let Some((by, offset, anchor, value_axis, layer_enc)) = find_stack_for_y(spec, y_field) else {
        return Cow::Borrowed(primary_batch);
    };
    let coord_flipped = matches!(spec.coord, Some(crate::spec::coord::CoordKind::Flip));
    // GH #77 follow-up: only proceed when this Stack's resolved value axis
    // is actually Y — the SAME triad `apply_stack` uses (shared via
    // `resolve_stack_value_on_x`), not a per-axis guess. A matching Stack
    // whose value lives on X (explicit `value_axis: Some(X)`, or a real
    // CoordFlip) is `axis_batch_for_x`'s to widen, not this function's.
    if resolve_stack_value_on_x(value_axis, coord_flipped) {
        return Cow::Borrowed(primary_batch);
    }
    // This is the scale-resolve *preview* stack (to widen the y-domain to the
    // stacked tops); the real draw-pass stack runs later via `apply_position`
    // with the panel's live warnings sink. Any `PositionAdjustSkipped` warning
    // resolve_group_channel would push here is therefore emitted exactly once
    // by that draw pass — so we discard this preview's warnings to avoid a
    // duplicate.
    let mut discard_warnings = Vec::new();
    match apply_stack(primary_batch, by, offset, anchor, value_axis, layer_enc, coord_flipped, &mut discard_warnings) {
        Ok(b) => Cow::Owned(b),
        Err(_) => Cow::Borrowed(primary_batch),
    }
}

/// Return the batch the x-scale should resolve against, accounting for a
/// Stack position adjustment (GH #77 follow-up).
///
/// Symmetric counterpart to [`axis_batch_for_y`] above — same rationale
/// (stacked cumulative tops must widen the axis domain or `LinearScale`
/// clips them and drawers silently skip every row past the first group),
/// same preview-stack / discard-warnings structure, same per-layer walk
/// shape, same real-`coord_flipped` + [`resolve_stack_value_on_x`] gating —
/// only proceeding when the resolved value axis is X. Needed once GH #77's
/// `value_axis: Some(X)` (horizontal composite-mark desugars) or a real
/// `CoordFlip` can put the stacked *value* column on X instead of Y —
/// before this, X was never anything but an ordinal/binned category axis,
/// which needs no value-based widening, so no X-side counterpart existed.
///
/// An earlier version of this function passed a *hardcoded* `true` (the
/// per-axis "assume value is here" guess `axis_batch_for_y` used to pass
/// `false`) instead of the real `coord_flipped`, relying on `apply_stack`'s
/// value-column type check to fail safe when the guess was wrong. That is
/// unsound whenever both axes are Float64 (e.g. a shared-bin-edge
/// histogram, where `x`/`y` are both numeric): a plain vertical stacked
/// histogram would spuriously match here, and if the wrong-axis column's
/// values happened to repeat across groups (entirely plausible — the
/// standard vertical stacked-histogram shape is exactly that, since
/// `shared_extent=True` makes bin edges repeat by design) the "wrong"
/// cumulation could silently corrupt real category data instead of merely
/// failing a type check. Using the real `coord_flipped` (via the same
/// triad `apply_stack` uses) removes the guess entirely: the gate above
/// answers "is the value really on this axis" from the same source of
/// truth the draw pass uses, so a vertical (non-flipped, no explicit
/// `value_axis`) spec's `axis_batch_for_x` call never even attempts
/// `apply_stack` — byte-identical no-op, not a fail-safe one.
pub(crate) fn axis_batch_for_x<'a>(
    spec: &'a ChartSpec,
    x_field: &str,
    primary_batch: &'a RecordBatch,
) -> Cow<'a, RecordBatch> {
    let Some((by, offset, anchor, value_axis, layer_enc)) = find_stack_for_x(spec, x_field) else {
        return Cow::Borrowed(primary_batch);
    };
    let coord_flipped = matches!(spec.coord, Some(crate::spec::coord::CoordKind::Flip));
    if !resolve_stack_value_on_x(value_axis, coord_flipped) {
        return Cow::Borrowed(primary_batch);
    }
    let mut discard_warnings = Vec::new();
    match apply_stack(primary_batch, by, offset, anchor, value_axis, layer_enc, coord_flipped, &mut discard_warnings) {
        Ok(b) => Cow::Owned(b),
        Err(_) => Cow::Borrowed(primary_batch),
    }
}

/// The `Stack` fields [`find_stack_for_y`]/[`find_stack_for_x`] extract,
/// plus the layer/spec `Encoding` the match came from. A named alias
/// (rather than an inline 5-tuple) keeps both functions' signatures under
/// clippy's `type_complexity` threshold — GH #77 added `value_axis` as the
/// 4th element; the GH #77 follow-up reused the same alias for the new
/// X-side finder rather than duplicating it.
type StackForAxis<'a> = (
    Option<&'a str>,
    &'a StackOffset,
    &'a crate::spec::position::StackAnchor,
    Option<crate::spec::position::StackValueAxis>,
    &'a crate::spec::encoding::Encoding,
);

/// Find the first Stack position adjustment in the spec whose layer (or
/// the chart itself, in the single-layer case) encodes the given y-field.
/// Multi-Stack layers are not merged here — the first match wins.
fn find_stack_for_y<'a>(spec: &'a ChartSpec, y_field: &str) -> Option<StackForAxis<'a>> {
    if let Some(layers) = spec.layers.as_ref() {
        for layer in layers {
            let layer_y = layer
                .encoding
                .y
                .as_ref()
                .map(|e| e.field.as_str())
                .or_else(|| spec.encoding.y.as_ref().map(|e| e.field.as_str()));
            if layer_y != Some(y_field) {
                continue;
            }
            if let Some(PositionAdjust::Stack { by, offset, anchor, value_axis }) =
                layer.position.as_ref().or(spec.position.as_ref())
            {
                return Some((by.as_deref(), offset, anchor, *value_axis, &layer.encoding));
            }
        }
    }
    if let Some(PositionAdjust::Stack { by, offset, anchor, value_axis }) = spec.position.as_ref() {
        let spec_y = spec.encoding.y.as_ref().map(|e| e.field.as_str());
        if spec_y == Some(y_field) {
            return Some((by.as_deref(), offset, anchor, *value_axis, &spec.encoding));
        }
    }
    None
}

/// Find the first Stack position adjustment in the spec whose layer (or
/// the chart itself, in the single-layer case) encodes the given x-field.
/// Symmetric counterpart to [`find_stack_for_y`] — same layer-walk shape,
/// checking `encoding.x` instead of `encoding.y`. Multi-Stack layers are
/// not merged here — the first match wins.
fn find_stack_for_x<'a>(spec: &'a ChartSpec, x_field: &str) -> Option<StackForAxis<'a>> {
    if let Some(layers) = spec.layers.as_ref() {
        for layer in layers {
            let layer_x = layer
                .encoding
                .x
                .as_ref()
                .map(|e| e.field.as_str())
                .or_else(|| spec.encoding.x.as_ref().map(|e| e.field.as_str()));
            if layer_x != Some(x_field) {
                continue;
            }
            if let Some(PositionAdjust::Stack { by, offset, anchor, value_axis }) =
                layer.position.as_ref().or(spec.position.as_ref())
            {
                return Some((by.as_deref(), offset, anchor, *value_axis, &layer.encoding));
            }
        }
    }
    if let Some(PositionAdjust::Stack { by, offset, anchor, value_axis }) = spec.position.as_ref() {
        let spec_x = spec.encoding.x.as_ref().map(|e| e.field.as_str());
        if spec_x == Some(x_field) {
            return Some((by.as_deref(), offset, anchor, *value_axis, &spec.encoding));
        }
    }
    None
}

/// Apply a position adjustment to a layer batch, returning a new batch with
/// rewritten coordinate columns (or, for ordinal-x Dodge / Jitter into bands,
/// with two synthetic `__pos_x_offset__` / `__pos_y_offset__` Float64 columns
/// appended). Returns a clone of the input unchanged if `position` is None
/// or Identity, or if the adjustment doesn't apply (e.g., Dodge with no
/// group channel set or ≤ 1 distinct groups).
pub(crate) fn apply_position(
    batch: &RecordBatch,
    position: Option<&PositionAdjust>,
    scales: &ResolvedScales,
    encoding: &crate::spec::encoding::Encoding,
    coord_flipped: bool,
    warnings: &mut Vec<RenderWarning>,
) -> Result<RecordBatch, crate::render::RenderError> {
    // D9: when no explicit position is set, check encoding.y.stack for an
    // encoding-level stacking directive. When coord_flipped, the value channel
    // lives in encoding.x (prepare.rs swapped x/y), so look there instead.
    let enc_stack: Option<PositionAdjust>;
    let effective_position: Option<&PositionAdjust> = if position.is_none() {
        let stack_source = if coord_flipped {
            encoding.x.as_ref().and_then(|e| e.stack.as_deref())
        } else {
            encoding.y.as_ref().and_then(|e| e.stack.as_deref())
        };
        enc_stack = stack_source.and_then(|s| {
            let offset = match s {
                "zero" => StackOffset::Zero,
                "normalize" => StackOffset::Normalize,
                "center" => StackOffset::Center,
                "false" | "null" | "none" => return None,
                _ => return None,
            };
            // No explicit `value_axis`: this synthetic Stack comes from the
            // `encoding.{x,y}.stack` shorthand, whose axis is already
            // selected by `coord_flipped` above via `stack_source` — not a
            // composite-mark desugar bypass, so `apply_stack`'s
            // `coord_flipped` fallback is the correct (and only) signal here.
            Some(PositionAdjust::Stack { by: None, offset, anchor: StackAnchor::Top, value_axis: None })
        });
        enc_stack.as_ref()
    } else {
        enc_stack = None;
        position
    };
    let Some(p) = effective_position else { return Ok(batch.clone()); };
    match p {
        PositionAdjust::Identity => Ok(batch.clone()),
        PositionAdjust::Dodge { by, padding } => {
            apply_dodge(batch, by.as_deref(), *padding, scales, encoding, coord_flipped, warnings)
        }
        PositionAdjust::Jitter { axis, width, seed } => {
            apply_jitter(batch, axis, *width, *seed, scales, encoding)
        }
        PositionAdjust::Stack { by, offset, anchor, value_axis } => {
            apply_stack(batch, by.as_deref(), offset, anchor, *value_axis, encoding, coord_flipped, warnings)
        }
    }
}

// ---------------------------------------------------------------------------
// Dodge
// ---------------------------------------------------------------------------

/// Resolve a position adjustment's grouping channel to one category string per
/// row, applying the single uniform grouping policy (RSUP-05).
///
/// Shared by both `apply_dodge` and `apply_stack` so the two siblings make the
/// exact same resolution decision for any given `by`/color channel. The only
/// per-adjustment content is the warning label, passed as `adjustment_label`
/// (`"dodge"` / `"stack"`); the four-case policy and the null→`""` mapping are
/// identical for both.
///
/// The resolution target is `by_field`, else `encoding.color.field`, else none.
/// The four outcomes:
///
/// 1. **No grouping channel at all** (no `by` *and* no color) → `None`, **no
///    warning**. This is the one documented intentional no-op: an adjustment
///    requested with nothing to group by has nothing to do.
/// 2. **Named target absent** from the batch → push a `PositionAdjustSkipped`
///    warning, return `None`. A typo'd or missing named grouping channel is a
///    user error and is surfaced, not silently dropped.
/// 3. **Target present and categorizable** (Utf8 / Int* / UInt* / Float* /
///    Boolean) → `Some(per-row category strings)` via
///    [`col_as_ordinal_category_str`]. Null rows map to `""` to preserve the
///    pre-RSUP-05 grouping key (arrow `StringArray::value` returned `""` for
///    nulls), keeping existing string-column goldens byte-identical.
/// 4. **Target present but un-categorizable** (timestamp / duration →
///    `col_as_ordinal_category_str` returns `Err`) → push a warning, return
///    `None`.
///
/// Cases 2 and 4 are the same failure class — a named grouping channel that
/// cannot yield categories — and are handled uniformly: warn and no-op the
/// adjustment, never crash. Returning `None` from any path leaves the caller to
/// clone the batch unchanged.
fn resolve_group_channel(
    batch: &RecordBatch,
    by_field: Option<&str>,
    encoding: &crate::spec::encoding::Encoding,
    adjustment_label: &str,
    warnings: &mut Vec<RenderWarning>,
) -> Option<Vec<String>> {
    // Case 1: resolution target = by_field, else color, else nothing requested.
    let by_col_name = match by_field {
        Some(s) => s.to_string(),
        None => encoding.color.as_ref()?.field.clone(),
    };

    // Case 2: named target absent from the batch.
    if batch.schema().index_of(&by_col_name).is_err() {
        warnings.push(RenderWarning::PositionAdjustSkipped {
            adjustment: adjustment_label.into(),
            reason: format!("by-column '{by_col_name}' not found in data"),
        });
        return None;
    }

    // Cases 3 & 4: present — categorize, or warn-and-skip if un-categorizable.
    match crate::render::arrow_cast::col_as_ordinal_category_str(batch, &by_col_name) {
        Ok(cats) => {
            // Map None (null rows) → "" to replicate the pre-RSUP-05 grouping
            // key (arrow StringArray::value(i) on a null returned ""), so
            // existing Utf8 goldens stay byte-identical.
            Some(cats.into_iter().map(|c| c.unwrap_or_default()).collect())
        }
        Err(_) => {
            let dtype = batch
                .schema()
                .field(batch.schema().index_of(&by_col_name).unwrap())
                .data_type()
                .clone();
            warnings.push(RenderWarning::PositionAdjustSkipped {
                adjustment: adjustment_label.into(),
                reason: format!(
                    "by-column '{by_col_name}' has dtype {dtype:?}, which cannot group categories"
                ),
            });
            None
        }
    }
}

/// Order the distinct dodge groups present in `by_cats` into left→right
/// sub-band slot order.
///
/// When `domain_order` is `Some` — the dodge grouping field IS the color field
/// and the color scale is categorical — groups are ordered by their position in
/// that domain (the resolved color-scale domain == legend order), so dodged
/// sub-bands read left→right in the same order the legend lists them. Any group
/// present in the data but absent from the domain falls after all domain groups,
/// in first-appearance order.
///
/// When `domain_order` is `None` — the dodge field is not the color field, or
/// there is no categorical color domain — pure first-appearance (row-encounter)
/// order is used. This is the pre-fix behavior; for raw-data layers encounter
/// order already equals the color domain order, so those goldens stay
/// byte-identical.
fn ordered_dodge_groups(by_cats: &[String], domain_order: Option<&[String]>) -> Vec<String> {
    // Distinct groups in first-appearance order. Group counts are tiny
    // (sub-band count), so linear membership scans are cheaper than a set.
    let mut encounter: Vec<String> = Vec::new();
    for g in by_cats {
        if !encounter.iter().any(|e| e == g) {
            encounter.push(g.clone());
        }
    }

    let Some(domain) = domain_order else {
        return encounter;
    };

    let mut ordered: Vec<String> = Vec::with_capacity(encounter.len());
    // Domain-present groups first, in domain order.
    for d in domain {
        if encounter.iter().any(|g| g == d) {
            ordered.push(d.clone());
        }
    }
    // Groups absent from the domain trail after, in first-appearance order.
    for g in &encounter {
        if !domain.iter().any(|d| d == g) {
            ordered.push(g.clone());
        }
    }
    ordered
}

/// Resolve the color-scale domain order to use for dodge slot assignment.
///
/// Returns `Some(domain)` only when the dodge grouping field is the color field
/// (either `by_field` is `None`, so the group field *is* the color field, or
/// `by_field` names the same column as `encoding.color`) AND the resolved color
/// scale is categorical. In every other case returns `None`, which
/// `ordered_dodge_groups` treats as "keep first-appearance order".
fn dodge_slot_domain<'a>(
    by_field: Option<&str>,
    scales: &'a ResolvedScales,
    encoding: &crate::spec::encoding::Encoding,
) -> Option<&'a [String]> {
    let color_enc = encoding.color.as_ref()?;
    let group_is_color = by_field.is_none_or(|bf| bf == color_enc.field);
    if !group_is_color {
        return None;
    }
    scales.color.as_ref()?.categorical_domain()
}

fn apply_dodge(
    batch: &RecordBatch,
    by_field: Option<&str>,
    padding: f64,
    scales: &ResolvedScales,
    encoding: &crate::spec::encoding::Encoding,
    coord_flipped: bool,
    warnings: &mut Vec<RenderWarning>,
) -> Result<RecordBatch, crate::render::RenderError> {
    // Resolve the `by` grouping channel to per-row category strings under the
    // single uniform policy. `None` = no-op the dodge (warning already pushed
    // when appropriate); return the batch unchanged.
    let Some(by_cats) = resolve_group_channel(batch, by_field, encoding, "dodge", warnings) else {
        return Ok(batch.clone());
    };

    // Sub-band slot order: match the legend when the dodge grouping field is the
    // color field and the color scale is categorical; otherwise first-appearance
    // order (the pre-fix behavior). Resolved once and threaded through both the
    // continuous-band and ordinal-band paths.
    let slot_domain = dodge_slot_domain(by_field, scales, encoding);

    // Dodge offsets marks along the categorical BAND axis. The band axis is
    // whichever positional channel actually resolved to an Ordinal scale —
    // mirroring the per-axis ordinality check `apply_jitter` already uses
    // (`x_is_ordinal` / `y_is_ordinal` there) — not `coord_flipped` alone.
    //
    // `coord_flipped` used to be the sole signal: under CoordFlip, prepare.rs
    // swaps x/y in the encoding, so the band lands in encoding.y/scales.y, and
    // `apply_stack` mirrors that swap the same way. But composite-mark
    // desugars can swap x/y themselves WITHOUT setting CoordFlip — e.g.
    // `mark_boxplot(horizontal=True)` (composite.py's `enc()`) puts the
    // continuous value on encoding.x and the categorical band on encoding.y
    // while `coord_flipped == false`. Picking encoding.x there dodges the
    // continuous value axis instead of the category axis, and — because each
    // box sub-layer (whisker/box/median) carries a different value column —
    // desyncs the sub-layers from each other (GH #75 cohesion-review defect).
    //
    // When exactly one axis is Ordinal, that one is unambiguously the band.
    // When both or neither are (the continuous-band case exercised by
    // `dodge_continuous_x_rewrites_x_column`, where dodge offsets a
    // quantitative-but-discrete x), scale kind can't disambiguate, so we fall
    // back to the `coord_flipped` convention — identical to the pre-fix
    // default and keeping every currently-passing goldens/tests byte-stable.
    let x_is_ordinal = matches!(scales.x, ScaleKind::Ordinal(_));
    let y_is_ordinal = matches!(scales.y, ScaleKind::Ordinal(_));
    let band_on_y = match (x_is_ordinal, y_is_ordinal) {
        (true, false) => false,
        (false, true) => true,
        _ => coord_flipped,
    };
    let (band_enc, band_scale) = if band_on_y { (encoding.y.as_ref(), &scales.y) } else { (encoding.x.as_ref(), &scales.x) };

    // Resolve the band column (the axis being dodged). None of Dodge's
    // messages name a positional channel token, so `coord_flipped` (carried
    // for struct-field uniformity across `PositionAdjustFailed`, R3) is
    // never read by `Display` here — the real value is passed anyway since
    // it's already in scope.
    let band_field = band_enc.ok_or_else(|| {
        crate::render::RenderError::PositionAdjustFailed {
            adjustment: "Dodge",
            reason: crate::render::PositionAdjustReason::Message("band-axis encoding required".into()),
            coord_flipped,
        }
    })?;
    let band_col_idx = batch.schema().index_of(&band_field.field).map_err(|_| {
        crate::render::RenderError::PositionAdjustFailed {
            adjustment: "Dodge",
            reason: crate::render::PositionAdjustReason::Message(format!(
                "band column '{}' not found",
                band_field.field
            )),
            coord_flipped,
        }
    })?;
    let is_ordinal_band = batch.schema().field(band_col_idx).data_type() != &DataType::Float64;
    if is_ordinal_band {
        return apply_dodge_ordinal(batch, &by_cats, padding, band_scale, band_on_y, slot_domain);
    }
    let band_arr = batch
        .column(band_col_idx)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| crate::render::RenderError::PositionAdjustFailed {
            adjustment: "Dodge",
            reason: crate::render::PositionAdjustReason::Message("band axis must be Float64".into()),
            coord_flipped,
        })?;

    // 1. Compute median spacing of unique band values (bandwidth proxy for a
    //    continuous band axis).
    let mut uniques: Vec<f64> = (0..band_arr.len())
        .filter(|i| !band_arr.is_null(*i))
        .map(|i| band_arr.value(i))
        .collect();
    uniques.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    uniques.dedup();
    if uniques.len() < 2 {
        return Ok(batch.clone());
    }
    let mut diffs: Vec<f64> = uniques.windows(2).map(|w| w[1] - w[0]).collect();
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let bandwidth = diffs[diffs.len() / 2];

    // 2. Determine group→slot order: legend order when the dodge field is the
    //    categorical color field, else first-appearance order (see
    //    `ordered_dodge_groups`).
    let groups_in_order = ordered_dodge_groups(&by_cats, slot_domain);
    let mut seen: HashMap<String, usize> = HashMap::new();
    for (idx, g) in groups_in_order.iter().enumerate() {
        seen.insert(g.clone(), idx);
    }
    let n_groups = groups_in_order.len();
    if n_groups <= 1 {
        return Ok(batch.clone());
    }

    let pad_total = bandwidth * padding * 2.0;
    let sub_band = (bandwidth - pad_total) / n_groups as f64;

    let mut new_band = Vec::with_capacity(band_arr.len());
    for (i, g) in by_cats.iter().enumerate() {
        let group_idx = *seen.get(g).unwrap();
        let offset =
            -bandwidth / 2.0 + bandwidth * padding + sub_band * (group_idx as f64 + 0.5);
        new_band.push(band_arr.value(i) + offset);
    }

    let mut cols: Vec<ArrayRef> = batch.columns().to_vec();
    cols[band_col_idx] = Arc::new(Float64Array::from(new_band));
    let schema = batch.schema();
    RecordBatch::try_new(schema, cols).map_err(|e| crate::render::RenderError::PositionAdjustFailed {
        adjustment: "Dodge",
        reason: crate::render::PositionAdjustReason::Message(format!("{e}")),
        coord_flipped,
    })
}

/// Ordinal-band Dodge — operates in pixel space because the categorical band
/// axis cannot be rewritten in data space. Injects two synthetic Float64
/// columns named `__pos_x_offset__` and `__pos_y_offset__`. The sub-band offset
/// is emitted into the axis that carries the categorical band: `__pos_x_offset__`
/// when `band_on_y` is false, `__pos_y_offset__` when `band_on_y` is true. The
/// other column is always 0. All positional mark drawers read these columns
/// post-scale-resolve via [`read_position_offsets`] and add them to the
/// rendered position.
///
/// `by_cats` is the per-row grouping category resolved by
/// [`resolve_group_channel`] (one entry per row; null group rows are `""`).
/// `band_scale` is the ordinal scale of the band axis (scales.x when
/// `band_on_y` is false, scales.y when true — resolved by [`apply_dodge`]
/// from which axis actually carries the Ordinal scale, not from
/// `coord_flipped` alone) and supplies the pixel bandwidth. `slot_domain` is
/// the color-scale domain order threaded from [`apply_dodge`]: `Some` orders
/// sub-band slots by legend order, `None` keeps first-appearance order.
fn apply_dodge_ordinal(
    batch: &RecordBatch,
    by_cats: &[String],
    padding: f64,
    band_scale: &ScaleKind,
    band_on_y: bool,
    slot_domain: Option<&[String]>,
) -> Result<RecordBatch, crate::render::RenderError> {
    let schema = batch.schema();
    let bandwidth_px = match band_scale {
        ScaleKind::Ordinal(s) => s.bandwidth(),
        _ => return Ok(batch.clone()),
    };

    let group_order = ordered_dodge_groups(by_cats, slot_domain);
    let mut group_idx: HashMap<String, usize> = HashMap::new();
    for (idx, g) in group_order.iter().enumerate() {
        group_idx.insert(g.clone(), idx);
    }
    let n_groups = group_order.len();
    if n_groups <= 1 {
        return Ok(batch.clone());
    }

    let pad_total = bandwidth_px * padding * 2.0;
    let sub_band = (bandwidth_px - pad_total) / n_groups as f64;

    let n = by_cats.len();
    let mut x_offsets: Vec<f64> = Vec::with_capacity(n);
    let mut y_offsets: Vec<f64> = Vec::with_capacity(n);
    for g in by_cats {
        let gi = *group_idx.get(g).unwrap();
        let off = -bandwidth_px / 2.0 + bandwidth_px * padding + sub_band * (gi as f64 + 0.5);
        // Emit the sub-band offset into whichever axis carries the categorical
        // band, as resolved by the caller.
        if band_on_y {
            x_offsets.push(0.0);
            y_offsets.push(off);
        } else {
            x_offsets.push(off);
            y_offsets.push(0.0);
        }
    }

    let mut cols: Vec<ArrayRef> = batch.columns().to_vec();
    cols.push(Arc::new(Float64Array::from(x_offsets)));
    cols.push(Arc::new(Float64Array::from(y_offsets)));

    let mut fields: Vec<Field> = schema.fields().iter().map(|f| f.as_ref().clone()).collect();
    fields.push(Field::new("__pos_x_offset__", DataType::Float64, false));
    fields.push(Field::new("__pos_y_offset__", DataType::Float64, false));

    // Stamp the dodge group count as explicit schema metadata (Task 3c
    // remediation). `BatchPositionMeta::dodge_n_groups` reads this key rather
    // than inferring the group count from distinct offset values: Jitter's
    // ordinal branch (`apply_jitter`) writes per-row noise into these SAME
    // `__pos_x_offset__` / `__pos_y_offset__` columns, so a distinct-value
    // heuristic would misread jitter noise as ≈row-count dodge groups. Only
    // this function (the ordinal-band Dodge path) ever sets this key.
    //
    // Also stamp the computed pixel sub-band width (GH #66 remediation):
    // `sub_band` here is the true per-group slot width in the same pixel
    // space the offsets above are computed in. Mark-width formulas
    // (`bar_width = band_extent / n_categories / n_groups * 0.8`, and the
    // analogous box/tick formulas) are blind to `padding` — they narrow by
    // `n_groups` but never account for the padding eaten out of each
    // sub-band, so at `padding > ~0.1` (bar) the 0.8-factor width can exceed
    // this `sub_band` and adjacent dodge groups overlap. Stamping `sub_band`
    // here — where `padding` is already in scope — lets those mark renderers
    // clamp their width to it (`BatchPositionMeta::clamp_width`) without
    // threading `padding` through `DrawCtx` or re-deriving it from a
    // per-layer `ChartSpec` that (for multi-layer charts) may not carry the
    // same `PositionAdjust` that was actually applied to this batch.
    let mut metadata = schema.metadata().clone();
    BatchPositionMeta::stamp_dodge(&mut metadata, n_groups, Some(sub_band));
    let new_schema = Arc::new(Schema::new(fields).with_metadata(metadata));

    RecordBatch::try_new(new_schema, cols).map_err(|e| crate::render::RenderError::PositionAdjustFailed {
        adjustment: "Dodge",
        reason: crate::render::PositionAdjustReason::Message(format!("ordinal: {e}")),
        // No positional channel token in this message and no `coord_flipped`
        // in scope (this fn resolves the band axis purely from `band_scale`)
        // — structurally inert, see the field doc on `PositionAdjustFailed`.
        coord_flipped: false,
    })
}

// ---------------------------------------------------------------------------
// Jitter
// ---------------------------------------------------------------------------

fn apply_jitter(
    batch: &RecordBatch,
    axis: &crate::spec::position::JitterAxis,
    width: f64,
    seed: Option<u64>,
    scales: &ResolvedScales,
    encoding: &crate::spec::encoding::Encoding,
) -> Result<RecordBatch, crate::render::RenderError> {
    use crate::spec::position::JitterAxis;
    use rand::{RngCore, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use twox_hash::xxh3;

    let x_idx = encoding
        .x
        .as_ref()
        .and_then(|e| batch.schema().index_of(&e.field).ok());
    let y_idx = encoding
        .y
        .as_ref()
        .and_then(|e| batch.schema().index_of(&e.field).ok());

    let do_x = matches!(axis, JitterAxis::X | JitterAxis::Both);
    let do_y = matches!(axis, JitterAxis::Y | JitterAxis::Both);

    // Per-axis ordinality check — for ordinal axes we must NOT overwrite the
    // string column with float noise. Instead we emit a pixel-offset column
    // (`__pos_x_offset__` / `__pos_y_offset__`) that the mark renderers add
    // post-scale. The pixel offset is `(u - 0.5) * width * bandwidth_px`, so
    // `width=1.0` spans the full band; `width=0.4` (default) keeps points
    // well within their band.
    let x_is_ordinal = matches!(scales.x, ScaleKind::Ordinal(_));
    let y_is_ordinal = matches!(scales.y, ScaleKind::Ordinal(_));
    let x_bandwidth = if let ScaleKind::Ordinal(s) = &scales.x { s.bandwidth() } else { 1.0 };
    let y_bandwidth = if let ScaleKind::Ordinal(s) = &scales.y { s.bandwidth() } else { 1.0 };

    let x_arr = x_idx.and_then(|j| batch.column(j).as_any().downcast_ref::<Float64Array>());
    let y_arr = y_idx.and_then(|j| batch.column(j).as_any().downcast_ref::<Float64Array>());

    let n = batch.num_rows();
    let mut new_x: Vec<f64> = Vec::with_capacity(n);
    let mut new_y: Vec<f64> = Vec::with_capacity(n);
    let mut x_pixel_offsets: Vec<f64> = Vec::with_capacity(n);
    let mut y_pixel_offsets: Vec<f64> = Vec::with_capacity(n);

    for i in 0..n {
        let xv = x_arr.map(|a| if a.is_null(i) { f64::NAN } else { a.value(i) }).unwrap_or(f64::NAN);
        let yv = y_arr.map(|a| if a.is_null(i) { f64::NAN } else { a.value(i) }).unwrap_or(f64::NAN);

        let row_seed = match seed {
            Some(s) => s.wrapping_add(i as u64),
            None => {
                let key = format!("{xv}|{yv}");
                xxh3::hash64(key.as_bytes())
            }
        };
        let mut rng = ChaCha8Rng::seed_from_u64(row_seed);
        let u = (rng.next_u64() as f64) / (u64::MAX as f64);
        let noise_x_data = (u - 0.5) * width;
        let u2 = (rng.next_u64() as f64) / (u64::MAX as f64);
        let noise_y_data = (u2 - 0.5) * width;

        new_x.push(if do_x && !x_is_ordinal { xv + noise_x_data } else { xv });
        new_y.push(if do_y && !y_is_ordinal { yv + noise_y_data } else { yv });
        x_pixel_offsets.push(if do_x && x_is_ordinal { (u - 0.5) * width * x_bandwidth } else { 0.0 });
        y_pixel_offsets.push(if do_y && y_is_ordinal { (u2 - 0.5) * width * y_bandwidth } else { 0.0 });
    }

    let mut cols: Vec<ArrayRef> = batch.columns().to_vec();
    if let (Some(j), true) = (x_idx, do_x && !x_is_ordinal) {
        cols[j] = Arc::new(Float64Array::from(new_x));
    }
    if let (Some(j), true) = (y_idx, do_y && !y_is_ordinal) {
        cols[j] = Arc::new(Float64Array::from(new_y));
    }

    let need_offsets = (do_x && x_is_ordinal) || (do_y && y_is_ordinal);
    if !need_offsets {
        let schema = batch.schema();
        // No positional channel token in this message and no `coord_flipped`
        // in scope (`apply_jitter` doesn't take it) — structurally inert,
        // see the field doc on `PositionAdjustFailed`.
        return RecordBatch::try_new(schema, cols).map_err(|e| crate::render::RenderError::PositionAdjustFailed {
            adjustment: "Jitter",
            reason: crate::render::PositionAdjustReason::Message(format!("{e}")),
            coord_flipped: false,
        });
    }

    cols.push(Arc::new(Float64Array::from(x_pixel_offsets)));
    cols.push(Arc::new(Float64Array::from(y_pixel_offsets)));
    let mut fields: Vec<Field> = batch.schema().fields().iter().map(|f| f.as_ref().clone()).collect();
    fields.push(Field::new("__pos_x_offset__", DataType::Float64, false));
    fields.push(Field::new("__pos_y_offset__", DataType::Float64, false));
    let new_schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(new_schema, cols).map_err(|e| crate::render::RenderError::PositionAdjustFailed {
        adjustment: "Jitter",
        reason: crate::render::PositionAdjustReason::Message(format!("ordinal: {e}")),
        coord_flipped: false,
    })
}

// ---------------------------------------------------------------------------
// Stack
// ---------------------------------------------------------------------------

/// Resolve whether a Stack's cumulated *value* column lives on X (GH #77).
///
/// The single canonical triad: an explicit `value_axis` always wins
/// (`Some(X)` → true, `Some(Y)` → false); `None` falls back to
/// `coord_flipped` — real `CoordFlip` always structurally puts value on X
/// (prepare.rs's `build_layers` swap convention). Shared by [`apply_stack`]
/// (the draw-time cumulation) and `axis_batch_for_x`/`axis_batch_for_y`
/// (the scale-resolve domain-widening preview, GH #77 follow-up) so the two
/// passes can never disagree about which axis holds the value.
pub(crate) fn resolve_stack_value_on_x(
    value_axis: Option<StackValueAxis>,
    coord_flipped: bool,
) -> bool {
    match value_axis {
        Some(StackValueAxis::X) => true,
        Some(StackValueAxis::Y) => false,
        None => coord_flipped,
    }
}

/// Position-adjust a layer's batch for a stacked layout.
///
/// Computes per-row segment bounds within each x-bin and writes them
/// back to the batch as the y column plus a synthetic
/// ``__stack_y_base__`` column. The y output is selected by
/// ``StackAnchor`` (Schwabish C6 audit-rework, 2026-05-12):
///
/// - ``StackAnchor::Top`` — y = top of segment (default; the renderer
///   draws rect-style marks from ``__stack_y_base__`` → ``y``). This
///   is byte-identical to pre-Schwabish behaviour.
/// - ``StackAnchor::Mid`` — y = midpoint of segment, so an annotation
///   lands at the visual centre of the stacked-bar segment for the
///   same row. The Python composite-mark desugar sets this on a
///   ``mark_text`` overlay so per-segment labels read cleanly
///   (e.g. ``class_prediction_error_chart(show_counts=True)``).
///
/// The renderer stays mark-agnostic; the choice of anchor lives in
/// the position spec and is set by whichever composite-mark desugar
/// is producing the layer.
///
/// GH #77 added `value_axis` as the 8th parameter (the explicit axis
/// override), pushing this past clippy's `too_many_arguments` default
/// threshold of 7 — matching the existing precedent for wide,
/// single-purpose render-pass functions elsewhere in this crate (e.g.
/// `layout::legend`, `render::annotation`, `render::scale_resolve`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_stack(
    batch: &RecordBatch,
    by_field: Option<&str>,
    offset: &crate::spec::position::StackOffset,
    anchor: &crate::spec::position::StackAnchor,
    value_axis: Option<crate::spec::position::StackValueAxis>,
    encoding: &crate::spec::encoding::Encoding,
    coord_flipped: bool,
    warnings: &mut Vec<RenderWarning>,
) -> Result<RecordBatch, crate::render::RenderError> {
    use crate::spec::position::{StackAnchor, StackOffset};
    use std::collections::BTreeMap;

    // Resolve the `by` grouping channel to per-row category strings under the
    // same single uniform policy `apply_dodge` uses (RSUP-05 sibling
    // completion). `None` = no-op the stack (warning already pushed when a
    // named channel was absent or un-categorizable); the genuine "no grouping
    // channel at all" case returns `None` silently. Pre-fix this block was a
    // raw `downcast_ref::<StringArray>()` that silently no-op'd on ANY non-Utf8
    // dtype while the same column would dodge fine — the sibling drift this
    // fix eliminates. Existing Utf8 stack goldens stay byte-identical because
    // the resolver's null→"" mapping reproduces the old `StringArray::value`.
    let Some(by_cats) = resolve_group_channel(batch, by_field, encoding, "stack", warnings) else {
        return Ok(batch.clone());
    };

    // GH #77: the value vs. category axis is resolved from an explicit
    // `value_axis` (set by composite-mark desugars that swap x/y directly,
    // e.g. `desugar_histogram`/`desugar_density` with
    // `orientation="horizontal"`, WITHOUT setting CoordFlip) when present;
    // `None` falls back to the `coord_flipped`-only convention (real
    // `CoordFlip`, where prepare.rs has already swapped x/y in the
    // encoding — the numeric (value) column lands in encoding.x and the
    // categorical (grouping) column in encoding.y). This is
    // byte-identical to the pre-#77 behavior whenever `value_axis` is
    // `None`. Shared with `axis_batch_for_x`/`axis_batch_for_y` (GH #77
    // follow-up) via `resolve_stack_value_on_x` so domain widening can
    // never desync from cumulation.
    let value_on_x = resolve_stack_value_on_x(value_axis, coord_flipped);
    let (value_enc, cat_enc) = if value_on_x {
        (encoding.x.as_ref(), encoding.y.as_ref())
    } else {
        (encoding.y.as_ref(), encoding.x.as_ref())
    };
    // R3 (restructured, #89 part C): the resolved slot each role reads from
    // swaps with `value_on_x` (which itself already factors in
    // `coord_flipped` — see `resolve_stack_value_on_x` above), so the
    // "(x)"/"(y)" annotation in these messages must swap with it too, rather
    // than staying hardcoded to the pre-#77 x=category/y=value assumption.
    // These are the RESOLVED tokens `apply_stack` actually checked — NOT
    // un-flipped here. `Display` (not this constructor) un-flips them back
    // to what the user wrote via `user_facing_channel`, reading the
    // `coord_flipped` carried alongside on the error itself; identity when
    // `!coord_flipped` (byte-identical default), including for the
    // `value_axis` override case (unrelated to `CoordFlip`, so nothing to
    // un-flip there).
    let cat_token: &'static str = if value_on_x { "y" } else { "x" };
    let value_token: &'static str = if value_on_x { "x" } else { "y" };

    let cat_field = cat_enc.ok_or(crate::render::RenderError::PositionAdjustFailed {
        adjustment: "Stack",
        reason: crate::render::PositionAdjustReason::MissingEncoding { role: "category", channel: cat_token },
        coord_flipped,
    })?;
    let value_field = value_enc.ok_or(crate::render::RenderError::PositionAdjustFailed {
        adjustment: "Stack",
        reason: crate::render::PositionAdjustReason::MissingEncoding { role: "value", channel: value_token },
        coord_flipped,
    })?;
    let xi = batch.schema().index_of(&cat_field.field).map_err(|_| {
        crate::render::RenderError::PositionAdjustFailed {
            adjustment: "Stack",
            reason: crate::render::PositionAdjustReason::Message(format!(
                "category col '{}' not found",
                cat_field.field
            )),
            coord_flipped,
        }
    })?;
    let yi = batch.schema().index_of(&value_field.field).map_err(|_| {
        crate::render::RenderError::PositionAdjustFailed {
            adjustment: "Stack",
            reason: crate::render::PositionAdjustReason::Message(format!(
                "value col '{}' not found",
                value_field.field
            )),
            coord_flipped,
        }
    })?;
    // Stack accepts Float64 directly; for UInt64 (e.g. Bin's `count` column)
    // and all signed integer types (Int8/Int16/Int32/Int64 — common for Polars
    // integer columns that have not been explicitly cast to Float64), we
    // transparently widen to f64 so stacked charts work without an explicit
    // cast at the Python boundary.
    let y_col = batch.column(yi);
    let ya_vals: Vec<f64> = if let Some(a) = y_col.as_any().downcast_ref::<Float64Array>() {
        (0..a.len()).map(|i| if a.is_null(i) { 0.0 } else { a.value(i) }).collect()
    } else if let Some(a) = y_col.as_any().downcast_ref::<arrow::array::UInt64Array>() {
        (0..a.len()).map(|i| if a.is_null(i) { 0.0 } else { a.value(i) as f64 }).collect()
    } else if let Some(a) = y_col.as_any().downcast_ref::<arrow::array::Int64Array>() {
        (0..a.len()).map(|i| if a.is_null(i) { 0.0 } else { a.value(i) as f64 }).collect()
    } else if let Some(a) = y_col.as_any().downcast_ref::<arrow::array::Int32Array>() {
        (0..a.len()).map(|i| if a.is_null(i) { 0.0 } else { a.value(i) as f64 }).collect()
    } else if let Some(a) = y_col.as_any().downcast_ref::<arrow::array::Int16Array>() {
        (0..a.len()).map(|i| if a.is_null(i) { 0.0 } else { a.value(i) as f64 }).collect()
    } else if let Some(a) = y_col.as_any().downcast_ref::<arrow::array::Int8Array>() {
        (0..a.len()).map(|i| if a.is_null(i) { 0.0 } else { a.value(i) as f64 }).collect()
    } else {
        // R3: `y_col` is always the VALUE column (bound from `yi` = the resolved
        // `value_field` index), whichever of x/y it actually reads from — see
        // `value_token` above, computed from the same `value_on_x`/`coord_flipped`
        // resolution this dtype check is downstream of.
        return Err(crate::render::RenderError::PositionAdjustFailed {
            adjustment: "Stack",
            reason: crate::render::PositionAdjustReason::ValueDtype {
                channel: value_token,
                dtype: format!("{:?}", y_col.data_type()),
            },
            coord_flipped,
        });
    };
    let ya_len = ya_vals.len();

    // x may be Float64 (continuous) or Utf8 (ordinal). Build a stable u64 key
    // for the BTreeMap from either case.
    let x_col = batch.column(xi);
    let x_keys: Vec<u64> = if x_col.data_type() == &DataType::Float64 {
        let xa = x_col.as_any().downcast_ref::<Float64Array>().unwrap();
        (0..xa.len()).map(|i| xa.value(i).to_bits()).collect()
    } else if let Some(xs) = x_col.as_any().downcast_ref::<StringArray>() {
        // Stable hash of the string for grouping; we never decode back, only bin.
        use twox_hash::xxh3;
        (0..xs.len())
            .map(|i| xxh3::hash64(xs.value(i).as_bytes()))
            .collect()
    } else {
        // R3: `x_col` is always the CATEGORY column (bound from `xi` = the
        // resolved `cat_field` index) — see `cat_token` above.
        return Err(crate::render::RenderError::PositionAdjustFailed {
            adjustment: "Stack",
            reason: crate::render::PositionAdjustReason::CategoryDtype { channel: cat_token },
            coord_flipped,
        });
    };

    // Group order from `by` channel (first-appearance). `by_cats` indexes 1:1
    // with rows, exactly as the old `by_arr` StringArray did.
    let mut group_idx_map: HashMap<String, usize> = HashMap::new();
    let mut group_order: Vec<String> = Vec::new();
    for g in &by_cats {
        if !group_idx_map.contains_key(g) {
            group_idx_map.insert(g.clone(), group_order.len());
            group_order.push(g.clone());
        }
    }

    // bins: x_key → Vec<(group_idx, row_idx, y)>
    let mut bins: BTreeMap<u64, Vec<(usize, usize, f64)>> = BTreeMap::new();
    for i in 0..ya_len {
        let gi = *group_idx_map.get(&by_cats[i]).unwrap();
        bins.entry(x_keys[i]).or_default().push((gi, i, ya_vals[i]));
    }

    let totals: HashMap<u64, f64> = bins
        .iter()
        .map(|(k, rows)| (*k, rows.iter().map(|(_, _, y)| y).sum::<f64>()))
        .collect();

    // new_y holds the cumulative TOP of each segment; new_y_base holds the
    // cumulative BOTTOM (the previous segment's top within the same bin, or
    // 0 / -mid for the first segment). Bar / area renderers draw each
    // segment from new_y_base[i] to new_y[i] so segments don't overlap.
    let mut new_y = vec![0.0_f64; ya_len];
    let mut new_y_base = vec![0.0_f64; ya_len];
    for (xkey, rows) in bins.iter_mut() {
        rows.sort_by_key(|(gi, _, _)| *gi);
        let total = totals.get(xkey).copied().unwrap_or(0.0);
        let mut acc = 0.0_f64;
        for (_, row_idx, y) in rows.iter() {
            let normalized = match offset {
                StackOffset::Zero => *y,
                StackOffset::Normalize => {
                    if total != 0.0 {
                        y / total
                    } else {
                        0.0
                    }
                }
                StackOffset::Center => *y,
            };
            new_y_base[*row_idx] = acc;
            acc += normalized;
            new_y[*row_idx] = acc;
        }
        if matches!(offset, StackOffset::Center) {
            let mid = acc / 2.0;
            for (_, row_idx, _) in rows.iter() {
                new_y[*row_idx] -= mid;
                new_y_base[*row_idx] -= mid;
            }
        }
    }

    // Schwabish C6 audit-rework (2026-05-12): the segment y output is
    // selected by the position spec's ``anchor`` field, not the calling
    // mark variant. ``Top`` = segment top (rect-style marks draw
    // base→top); ``Mid`` = segment midpoint (annotation marks like
    // mark_text land at the visual centre).
    let y_output: Vec<f64> = match anchor {
        StackAnchor::Mid => (0..ya_len)
            .map(|i| 0.5 * (new_y[i] + new_y_base[i]))
            .collect(),
        StackAnchor::Top => new_y.clone(),
    };

    let mut cols: Vec<ArrayRef> = batch.columns().to_vec();
    cols[yi] = Arc::new(Float64Array::from(y_output));
    // Rebuild schema with the y column promoted to Float64 (in case the
    // input was UInt64, e.g. Bin's `count` column).
    let mut new_fields: Vec<Field> = batch
        .schema()
        .fields()
        .iter()
        .map(|f| f.as_ref().clone())
        .collect();
    new_fields[yi] = Field::new(new_fields[yi].name(), DataType::Float64, true);

    // Append a synthetic __stack_y_base__ column so mark drawers can emit
    // per-segment rects (base → top) instead of drawing every segment from
    // y=0. Bar / area renderers look this up via `col_as_f64` when present.
    cols.push(Arc::new(Float64Array::from(new_y_base)));
    new_fields.push(Field::new("__stack_y_base__", DataType::Float64, true));

    // GH #77 follow-up: stamp explicit schema metadata (mirroring the
    // `DODGE_N_GROUPS_KEY` / `DODGE_SUB_BAND_PX_KEY` pattern from
    // `apply_dodge_ordinal`, GH #66) recording that the stacked *value*
    // column landed on the X axis this pass, not Y. `__stack_y_base__` is
    // always the same column regardless of axis, so mark drawers (`bar.rs`,
    // `area.rs`) that consume it need an explicit signal for which scale to
    // map it through — they can't infer it from the batch shape alone.
    // Absence of the key (the `!value_on_x` branch below) is the pre-#77
    // default and keeps every existing vertical-stack mark byte-identical.
    let new_schema = if value_on_x {
        let mut metadata = batch.schema().metadata().clone();
        BatchPositionMeta::stamp_stack_value_on_x(&mut metadata);
        Arc::new(Schema::new(new_fields).with_metadata(metadata))
    } else {
        Arc::new(Schema::new(new_fields))
    };
    RecordBatch::try_new(new_schema, cols).map_err(|e| crate::render::RenderError::PositionAdjustFailed {
        adjustment: "Stack",
        reason: crate::render::PositionAdjustReason::Message(format!("{e}")),
        coord_flipped,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::Rect;
    use crate::render::scale_resolve::ColorScale;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::position::{JitterAxis, PositionAdjust, StackAnchor, StackOffset, StackValueAxis};
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn dummy_scales() -> ResolvedScales {
        // Construct a minimal ResolvedScales — Linear x/y with [0,100] domain & range.
        use crate::scale::linear::LinearScale;
        let lx = LinearScale::new_internal(vec![0.0, 10.0], vec![0.0, 100.0], false, false);
        let ly = LinearScale::new_internal(vec![0.0, 100.0], vec![0.0, 100.0], false, false);
        ResolvedScales {
            x: ScaleKind::Linear(lx),
            y: ScaleKind::Linear(ly),
            color: None,
            size: None,
            shape: None,
            opacity: None,
            x2: None,
            y2: None,
            y_slots: Default::default(),
        }
    }

    fn enc_xy(xf: &str, yf: &str, color: Option<&str>) -> Encoding {
        Encoding {
            x: Some(EncodingSpec { field: xf.into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: yf.into(), type_: None, ..Default::default() }),
            color: color.map(|c| EncodingSpec {
                field: c.into(),
                type_: None,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn batch_xyg() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 1.0, 2.0, 2.0])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0])),
                Arc::new(StringArray::from(vec!["a", "b", "a", "b"])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn identity_returns_clone() {
        let b = batch_xyg();
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let out = apply_position(&b, Some(&PositionAdjust::Identity), &s, &enc, false, &mut Vec::new()).unwrap();
        assert_eq!(out.num_rows(), b.num_rows());
        assert_eq!(out.num_columns(), b.num_columns());
    }

    #[test]
    fn none_position_returns_clone() {
        let b = batch_xyg();
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let out = apply_position(&b, None, &s, &enc, false, &mut Vec::new()).unwrap();
        assert_eq!(out.num_rows(), b.num_rows());
    }

    #[test]
    fn dodge_continuous_x_rewrites_x_column() {
        let b = batch_xyg();
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Dodge { by: Some("g".into()), padding: 0.0 };
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut Vec::new()).unwrap();
        // Two unique x values: 1.0, 2.0 → bandwidth = 1.0.
        // Two groups (a, b) → sub_band = 0.5; offsets a=-0.25, b=+0.25.
        let xa = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert!((xa.value(0) - 0.75).abs() < 1e-9, "row0 x={}", xa.value(0));
        assert!((xa.value(1) - 1.25).abs() < 1e-9, "row1 x={}", xa.value(1));
        assert!((xa.value(2) - 1.75).abs() < 1e-9, "row2 x={}", xa.value(2));
        assert!((xa.value(3) - 2.25).abs() < 1e-9, "row3 x={}", xa.value(3));
    }

    #[test]
    fn dodge_single_group_is_noop() {
        // All rows in group "a" → n_groups == 1 → return clone.
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        let b = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
                Arc::new(Float64Array::from(vec![10.0, 20.0])),
                Arc::new(StringArray::from(vec!["a", "a"])),
            ],
        )
        .unwrap();
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Dodge { by: Some("g".into()), padding: 0.05 };
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut Vec::new()).unwrap();
        let xa = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(xa.value(0), 1.0);
        assert_eq!(xa.value(1), 2.0);
    }

    // ---- Ordinal-band Dodge + CoordFlip (Task 3b) ---------------------------

    /// ResolvedScales with one ordinal axis (2 bands over pixel range [0,100],
    /// so `bandwidth() == 50`) and the other axis Linear. `ordinal_on_x` selects
    /// which visual axis carries the categorical band.
    fn scales_one_ordinal(ordinal_on_x: bool) -> ResolvedScales {
        use crate::scale::linear::LinearScale;
        use crate::scale::ordinal::OrdinalScale;
        let ord = ScaleKind::Ordinal(OrdinalScale::new_internal(
            vec!["a".into(), "b".into()],
            vec![0.0, 100.0],
            0.0,
        ));
        let lin = ScaleKind::Linear(LinearScale::new_internal(
            vec![0.0, 100.0],
            vec![0.0, 100.0],
            false,
            false,
        ));
        let (x, y) = if ordinal_on_x { (ord, lin) } else { (lin, ord) };
        ResolvedScales { x, y, color: None, size: None, shape: None, opacity: None, x2: None, y2: None, y_slots: Default::default() }
    }

    /// Batch with a categorical band column `cat`, a numeric value column `val`,
    /// and a two-level grouping column `grp` (p/q interleaved).
    fn batch_cat_val_grp() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, false),
            Field::new("val", DataType::Float64, false),
            Field::new("grp", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "a", "b", "b"])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0])),
                Arc::new(StringArray::from(vec!["p", "q", "p", "q"])),
            ],
        )
        .unwrap()
    }

    fn offset_col(batch: &RecordBatch, name: &str) -> Vec<f64> {
        let idx = batch.schema().index_of(name).unwrap();
        let a = batch.column(idx).as_any().downcast_ref::<Float64Array>().unwrap();
        (0..a.len()).map(|i| a.value(i)).collect()
    }

    #[test]
    fn dodge_ordinal_band_x_unflipped_offsets_x_only() {
        // Baseline (unflipped): band axis = ordinal x. Sub-band offset lands in
        // __pos_x_offset__; __pos_y_offset__ is all zero. bandwidth=50, 2 groups,
        // padding=0 → sub_band=25 → offsets -12.5 (p) / +12.5 (q).
        let b = batch_cat_val_grp();
        let enc = enc_xy("cat", "val", Some("grp"));
        let s = scales_one_ordinal(true);
        let pos = PositionAdjust::Dodge { by: Some("grp".into()), padding: 0.0 };
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut Vec::new()).unwrap();
        assert_eq!(offset_col(&out, "__pos_x_offset__"), vec![-12.5, 12.5, -12.5, 12.5]);
        assert_eq!(offset_col(&out, "__pos_y_offset__"), vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn dodge_ordinal_band_flipped_offsets_y_matches_unflipped_x() {
        // Under CoordFlip, prepare.rs swapped x/y: encoding.x = value ("val"),
        // encoding.y = categorical band ("cat"), band ordinal scale = scales.y.
        // The sub-band offset must land in __pos_y_offset__ (zero into
        // __pos_x_offset__), with the SAME magnitudes the unflipped path emits
        // into __pos_x_offset__.
        let b = batch_cat_val_grp();
        let enc = enc_xy("val", "cat", Some("grp"));
        let s = scales_one_ordinal(false);
        let pos = PositionAdjust::Dodge { by: Some("grp".into()), padding: 0.0 };
        let out = apply_position(&b, Some(&pos), &s, &enc, true, &mut Vec::new()).unwrap();

        let y_off = offset_col(&out, "__pos_y_offset__");
        assert_eq!(y_off, vec![-12.5, 12.5, -12.5, 12.5]);
        assert_eq!(offset_col(&out, "__pos_x_offset__"), vec![0.0, 0.0, 0.0, 0.0]);

        // Mirror check: flipped y-offset == unflipped x-offset math.
        let unflipped = apply_position(
            &b,
            Some(&pos),
            &scales_one_ordinal(true),
            &enc_xy("cat", "val", Some("grp")),
            false,
            &mut Vec::new(),
        )
        .unwrap();
        assert_eq!(y_off, offset_col(&unflipped, "__pos_x_offset__"));
    }

    #[test]
    fn dodge_ordinal_band_on_y_without_coord_flip_offsets_y_not_x() {
        // GH #75 cohesion-review defect: a natively-horizontal composite mark
        // (mark_boxplot(horizontal=True)) swaps x/y in its Python desugar
        // WITHOUT setting CoordFlip, so encoding.x is the continuous value
        // channel and encoding.y is the categorical band — the mirror image of
        // `dodge_ordinal_band_flipped_offsets_y_matches_unflipped_x` but with
        // `coord_flipped == false`. Selecting the band axis from `coord_flipped`
        // alone (the pre-fix behavior) wrongly treats "val" as the band and
        // silently no-ops the dodge (or corrupts "val") instead of offsetting
        // "cat". The band axis must be chosen by which channel actually
        // resolved to an Ordinal scale.
        let b = batch_cat_val_grp();
        let enc = enc_xy("val", "cat", Some("grp"));
        let s = scales_one_ordinal(false); // ordinal scale lives on y, not x
        let pos = PositionAdjust::Dodge { by: Some("grp".into()), padding: 0.0 };
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut Vec::new()).unwrap();

        let y_off = offset_col(&out, "__pos_y_offset__");
        assert_eq!(y_off, vec![-12.5, 12.5, -12.5, 12.5], "band offset must land in y (the ordinal axis)");
        assert_eq!(offset_col(&out, "__pos_x_offset__"), vec![0.0, 0.0, 0.0, 0.0], "value axis (x) must not be offset");
    }

    #[test]
    fn dodge_ordinal_band_flipped_single_group_is_noop() {
        // n_groups == 1 under flip → no offset columns appended, batch unchanged.
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, false),
            Field::new("val", DataType::Float64, false),
            Field::new("grp", DataType::Utf8, false),
        ]));
        let b = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "b"])),
                Arc::new(Float64Array::from(vec![10.0, 30.0])),
                Arc::new(StringArray::from(vec!["p", "p"])),
            ],
        )
        .unwrap();
        let enc = enc_xy("val", "cat", Some("grp"));
        let s = scales_one_ordinal(false);
        let pos = PositionAdjust::Dodge { by: Some("grp".into()), padding: 0.0 };
        let out = apply_position(&b, Some(&pos), &s, &enc, true, &mut Vec::new()).unwrap();
        assert_eq!(out.num_columns(), b.num_columns());
        assert!(out.schema().index_of("__pos_x_offset__").is_err());
        assert!(out.schema().index_of("__pos_y_offset__").is_err());
    }

    // ---- Dodge sub-band slot order matches legend (Task 5b) ----------------

    /// Ordinal-x ResolvedScales carrying a categorical color scale whose domain
    /// is `color_domain` (== legend order). bandwidth stays 50 (2 bands over
    /// [0,100]).
    fn scales_ordinal_x_with_color(color_domain: &[&str]) -> ResolvedScales {
        use crate::scale::linear::LinearScale;
        use crate::scale::ordinal::OrdinalScale;
        let ord = ScaleKind::Ordinal(OrdinalScale::new_internal(
            vec!["a".into(), "b".into()],
            vec![0.0, 100.0],
            0.0,
        ));
        let lin = ScaleKind::Linear(LinearScale::new_internal(
            vec![0.0, 100.0],
            vec![0.0, 100.0],
            false,
            false,
        ));
        let color = ColorScale::Categorical {
            domain: color_domain.iter().map(|s| s.to_string()).collect(),
            palette: std::borrow::Cow::Owned(vec![
                crate::render::color::from_rgb(0x00, 0x00, 0x00),
                crate::render::color::from_rgb(0xFF, 0xFF, 0xFF),
            ]),
        };
        ResolvedScales {
            x: ord,
            y: lin,
            color: Some(color),
            size: None,
            shape: None,
            opacity: None,
            x2: None,
            y2: None,
            y_slots: Default::default(),
        }
    }

    /// Batch with an ordinal band `cat`, a numeric `val`, and a two-level model
    /// column whose rows arrive in *sorted* key order ["alt", "base"] — the
    /// order a BTreeMap-bucketing transform (e.g. BoxStats) emits.
    fn batch_cat_val_model_sorted() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, false),
            Field::new("val", DataType::Float64, false),
            Field::new("model", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "a"])),
                Arc::new(Float64Array::from(vec![10.0, 20.0])),
                Arc::new(StringArray::from(vec!["alt", "base"])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn dodge_ordinal_slot0_follows_color_domain_not_encounter_order() {
        // (a) Rows arrive sorted-key ["alt","base"] (BoxStats bucketing), but the
        // color domain is ["base","alt"] (registration order). The dodge field IS
        // the color field, so slot 0 must be "base" (the domain-order leftmost),
        // not "alt" (the encounter-order first). bandwidth=50, padding=0, 2 groups
        // → sub_band=25 → slot0=-12.5, slot1=+12.5. So base→-12.5, alt→+12.5.
        let b = batch_cat_val_model_sorted();
        let enc = enc_xy("cat", "val", Some("model"));
        let s = scales_ordinal_x_with_color(&["base", "alt"]);
        let pos = PositionAdjust::Dodge { by: Some("model".into()), padding: 0.0 };
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut Vec::new()).unwrap();
        // row0 = "alt" → slot 1 → +12.5; row1 = "base" → slot 0 → -12.5.
        assert_eq!(offset_col(&out, "__pos_x_offset__"), vec![12.5, -12.5]);
    }

    #[test]
    fn dodge_ordinal_non_color_field_keeps_encounter_order() {
        // (b) Regression: dodge by "grp" while the color channel is a DIFFERENT
        // field ("cat"). A categorical color domain exists but must be ignored
        // because the dodge field is not the color field → first-appearance order.
        // grp encounter = [p, q] → p=slot0 (-12.5), q=slot1 (+12.5).
        let b = batch_cat_val_grp();
        let enc = enc_xy("cat", "val", Some("cat"));
        // Color domain deliberately reversed relative to encounter order; it must
        // have no effect because "grp" != "cat".
        let s = scales_ordinal_x_with_color(&["q", "p"]);
        let pos = PositionAdjust::Dodge { by: Some("grp".into()), padding: 0.0 };
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut Vec::new()).unwrap();
        assert_eq!(offset_col(&out, "__pos_x_offset__"), vec![-12.5, 12.5, -12.5, 12.5]);
    }

    #[test]
    fn dodge_ordinal_no_color_scale_keeps_encounter_order() {
        // (c) Regression: dodge by the color field ("grp") but there is NO
        // resolved color scale → first-appearance order. grp encounter = [p, q]
        // → p=slot0 (-12.5), q=slot1 (+12.5).
        let b = batch_cat_val_grp();
        let enc = enc_xy("cat", "val", Some("grp"));
        let s = scales_one_ordinal(true); // color: None
        let pos = PositionAdjust::Dodge { by: None, padding: 0.0 };
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut Vec::new()).unwrap();
        assert_eq!(offset_col(&out, "__pos_x_offset__"), vec![-12.5, 12.5, -12.5, 12.5]);
    }

    #[test]
    fn jitter_explicit_seed_deterministic() {
        let b = batch_xyg();
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Jitter { axis: JitterAxis::X, width: 0.5, seed: Some(42) };
        let a = apply_position(&b, Some(&pos), &s, &enc, false, &mut Vec::new()).unwrap();
        let bb = apply_position(&b, Some(&pos), &s, &enc, false, &mut Vec::new()).unwrap();
        let ax = a.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        let bx = bb.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        for i in 0..4 {
            assert_eq!(ax.value(i).to_bits(), bx.value(i).to_bits());
        }
    }

    #[test]
    fn jitter_none_seed_is_deterministic_via_hash() {
        let b = batch_xyg();
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Jitter { axis: JitterAxis::X, width: 0.5, seed: None };
        let a = apply_position(&b, Some(&pos), &s, &enc, false, &mut Vec::new()).unwrap();
        let bb = apply_position(&b, Some(&pos), &s, &enc, false, &mut Vec::new()).unwrap();
        let ax = a.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        let bx = bb.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        for i in 0..4 {
            assert_eq!(ax.value(i).to_bits(), bx.value(i).to_bits());
        }
    }

    #[test]
    fn stack_zero_accumulates_y() {
        let b = batch_xyg();
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Stack { by: Some("g".into()), offset: StackOffset::Zero, anchor: StackAnchor::Top, value_axis: None };
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut Vec::new()).unwrap();
        let ya = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        // Group order: a=0, b=1. At x=1: a=10 → 10, b=20 → 30. At x=2: a=30 → 30, b=40 → 70.
        assert_eq!(ya.value(0), 10.0);
        assert_eq!(ya.value(1), 30.0);
        assert_eq!(ya.value(2), 30.0);
        assert_eq!(ya.value(3), 70.0);
    }

    #[test]
    fn stack_normalize_sums_to_one_per_x() {
        let b = batch_xyg();
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Stack {
            by: Some("g".into()),
            offset: StackOffset::Normalize,
            anchor: StackAnchor::Top,
            value_axis: None,
        };
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut Vec::new()).unwrap();
        let ya = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        // For each x bin the top of the highest stack should be 1.0.
        // x=1: top group (b) reaches 1.0 → row 1.
        // x=2: top group (b) reaches 1.0 → row 3.
        assert!((ya.value(1) - 1.0).abs() < 1e-9);
        assert!((ya.value(3) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn stack_center_symmetric_around_zero() {
        let b = batch_xyg();
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Stack { by: Some("g".into()), offset: StackOffset::Center, anchor: StackAnchor::Top, value_axis: None };
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut Vec::new()).unwrap();
        let ya = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        // x=1: total=30, mid=15. a row goes 0..10 → top at 10-15=-5.
        // b row goes 10..30 → top at 30-15=15.
        assert!((ya.value(0) + 5.0).abs() < 1e-9);
        assert!((ya.value(1) - 15.0).abs() < 1e-9);
    }

    #[test]
    fn stack_anchor_mid_zero_offset_emits_midpoint_y() {
        // Schwabish C6 audit-rework (2026-05-12): StackAnchor::Mid on a
        // stacked layer outputs the segment midpoint as y, so an
        // annotation overlay (mark_text, mark_point, …) lands at the
        // visual centre of each segment.
        let b = batch_xyg();
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Stack {
            by: Some("g".into()),
            offset: StackOffset::Zero,
            anchor: StackAnchor::Mid,
            value_axis: None,
        };
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut Vec::new()).unwrap();
        let ya = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        // x=1: a segment 0..10 → mid 5;  b segment 10..30 → mid 20.
        // x=2: a segment 0..30 → mid 15; b segment 30..70 → mid 50.
        assert!((ya.value(0) - 5.0).abs()  < 1e-9, "row 0 mid={}", ya.value(0));
        assert!((ya.value(1) - 20.0).abs() < 1e-9, "row 1 mid={}", ya.value(1));
        assert!((ya.value(2) - 15.0).abs() < 1e-9, "row 2 mid={}", ya.value(2));
        assert!((ya.value(3) - 50.0).abs() < 1e-9, "row 3 mid={}", ya.value(3));
        // __stack_y_base__ still carries segment bottoms unchanged.
        let base = out.schema().index_of("__stack_y_base__").unwrap();
        let ba = out.column(base).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(ba.value(0), 0.0);
        assert_eq!(ba.value(1), 10.0);
        assert_eq!(ba.value(2), 0.0);
        assert_eq!(ba.value(3), 30.0);
    }

    #[test]
    fn stack_anchor_mid_normalize_emits_proportion_midpoint() {
        // C6 coverage: Mid × Normalize. Each x-bin sums to 1.0; segment
        // midpoints land at 0.5 * (top + base) in proportion space.
        let b = batch_xyg();
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Stack {
            by: Some("g".into()),
            offset: StackOffset::Normalize,
            anchor: StackAnchor::Mid,
            value_axis: None,
        };
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut Vec::new()).unwrap();
        let ya = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        // x=1: total=30. a→10/30 ⇒ top=1/3, mid=1/6. b→20/30 ⇒ top=1, mid=2/3.
        assert!((ya.value(0) - (1.0 / 6.0)).abs() < 1e-9);
        assert!((ya.value(1) - (2.0 / 3.0)).abs() < 1e-9);
        // x=2: total=70. a→30/70 ⇒ mid=15/70. b→40/70 ⇒ mid=50/70.
        assert!((ya.value(2) - (15.0 / 70.0)).abs() < 1e-9);
        assert!((ya.value(3) - (50.0 / 70.0)).abs() < 1e-9);
    }

    #[test]
    fn stack_anchor_mid_center_emits_streamgraph_midpoint() {
        // C6 coverage: Mid × Center. Stack is symmetric around y=0;
        // segment midpoints land at 0.5 * (top + base) in centered space.
        let b = batch_xyg();
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Stack {
            by: Some("g".into()),
            offset: StackOffset::Center,
            anchor: StackAnchor::Mid,
            value_axis: None,
        };
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut Vec::new()).unwrap();
        let ya = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        // x=1 total=30, mid_axis=15. a: base=-15, top=-5  → mid=-10.
        //                              b: base=-5,  top=15 → mid=5.
        assert!((ya.value(0) + 10.0).abs() < 1e-9, "row 0 mid={}", ya.value(0));
        assert!((ya.value(1) -  5.0).abs() < 1e-9, "row 1 mid={}", ya.value(1));
    }

    // R8b regression: axis_batch_for_y falls back to primary_batch on
    // stack failure AND the same error is re-derivable by apply_stack.
    #[test]
    fn axis_batch_for_y_falls_back_on_stack_error() {
        use crate::spec::chart::ChartSpec;
        use crate::spec::mark::Mark;
        use arrow::array::BooleanArray;

        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Utf8, false),
            Field::new("y", DataType::Boolean, false), // intentionally wrong type
            Field::new("g", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "a"])),
                Arc::new(BooleanArray::from(vec![true, false])),
                Arc::new(StringArray::from(vec!["g1", "g2"])),
            ],
        )
        .unwrap();

        let spec = ChartSpec {
            data: Default::default(),
            mark: Mark::Bar,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: Some(EncodingSpec { field: "g".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
            position: Some(PositionAdjust::Stack {
                by: Some("g".into()),
                offset: StackOffset::Zero,
                anchor: StackAnchor::Top,
                value_axis: None,
            }),
            title: None,
            axis_x: None,
            axis_y: None,
            selections: Vec::new(),
            conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        };

        let result = axis_batch_for_y(&spec, "y", &batch);
        assert!(
            matches!(result, Cow::Borrowed(_)),
            "expected Borrowed fallback on stack failure"
        );

        let direct = apply_stack(
            &batch,
            Some("g"),
            &StackOffset::Zero,
            &StackAnchor::Top,
            None,
            &spec.encoding,
            false,
            &mut Vec::new(),
        );
        assert!(
            direct.is_err(),
            "apply_stack should fail on Boolean y column"
        );
    }

    // ------------------------------------------------------------------
    // GH #77 follow-up (E2E-discovered gap): `axis_batch_for_x`, the X-side
    // counterpart to `axis_batch_for_y`. Before this, only Y ever got
    // Stack-aware domain widening; a horizontal stack's x-domain resolved
    // from the RAW per-group values, so `LinearScaleData::scale` returned
    // NaN (`clamp=false`) for the true stacked total and drawers silently
    // skipped every row past the first group.
    // ------------------------------------------------------------------

    /// RED against the pre-fix code (no x-widening at all): the x-domain
    /// auto-sizes from the raw per-group column (max 5.0, plus padding),
    /// so `to_pixel_f64(8.0)` — the true stacked total — returns `None`.
    #[test]
    fn axis_batch_for_x_widens_horizontal_stack_domain() {
        use crate::layout::ThemeInputs;
        use crate::render::scale_resolve::resolve_scales;
        use crate::spec::chart::ChartSpec;
        use crate::spec::mark::Mark;
        use crate::spec::position::StackValueAxis;

        // Horizontal desugar shape: count (value) on x, bin_start (category)
        // on y — one bin, two stack groups "a" (3.0) and "b" (5.0); stacked
        // total = 8.0.
        let schema = Arc::new(Schema::new(vec![
            Field::new("count", DataType::Float64, false),
            Field::new("bin_start", DataType::Float64, false),
            Field::new("grp", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![3.0, 5.0])),
                Arc::new(Float64Array::from(vec![0.0, 0.0])),
                Arc::new(StringArray::from(vec!["a", "b"])),
            ],
        )
        .unwrap();

        let spec = ChartSpec {
            data: Default::default(),
            mark: Mark::Bar,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "count".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "bin_start".into(), type_: None, ..Default::default() }),
                color: Some(EncodingSpec { field: "grp".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
            position: Some(PositionAdjust::Stack {
                by: Some("grp".into()),
                offset: StackOffset::Zero,
                anchor: StackAnchor::Top,
                value_axis: Some(StackValueAxis::X),
            }),
            title: None,
            axis_x: None,
            axis_y: None,
            selections: Vec::new(),
            conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        };

        let theme = ThemeInputs::default();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 300.0), (0.0, 100.0), &theme).unwrap();

        let px = scales.x.to_pixel_f64(8.0);
        assert!(
            px.is_some(),
            "x-domain must widen to include the stacked total (8.0); to_pixel_f64 returned None"
        );
    }

    /// Byte-stability: a standard vertical Stack (value on y, the pre-#77
    /// default) leaves `axis_batch_for_x` a no-op — the X-side's
    /// value-on-X fallback assumption is wrong here (x is the category),
    /// so `apply_stack` fails cleanly (Utf8 "cat" can't be cumulated) and
    /// `axis_batch_for_x` falls back to `Cow::Borrowed`. `axis_batch_for_y`
    /// keeps its existing (pre-#77) behavior unchanged — same cumulated
    /// values as `stack_zero_accumulates_y`.
    #[test]
    fn axis_batch_for_x_is_noop_for_vertical_stack() {
        let b = batch_xyg();
        let enc = enc_xy("x", "y", Some("g"));
        let spec = crate::spec::chart::ChartSpec {
            data: Default::default(),
            mark: crate::spec::mark::Mark::Bar,
            encoding: enc,
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
            position: Some(PositionAdjust::Stack {
                by: Some("g".into()),
                offset: StackOffset::Zero,
                anchor: StackAnchor::Top,
                value_axis: None,
            }),
            title: None,
            axis_x: None,
            axis_y: None,
            selections: Vec::new(),
            conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        };

        let x_result = axis_batch_for_x(&spec, "x", &b);
        assert!(matches!(x_result, Cow::Borrowed(_)), "x-side must be a no-op for a vertical stack");

        let y_result = axis_batch_for_y(&spec, "y", &b);
        match y_result {
            Cow::Owned(out) => {
                let ya = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
                // Same expected values as stack_zero_accumulates_y.
                assert_eq!(ya.value(0), 10.0);
                assert_eq!(ya.value(1), 30.0);
                assert_eq!(ya.value(2), 30.0);
                assert_eq!(ya.value(3), 70.0);
            }
            Cow::Borrowed(_) => panic!("expected y-side widening to succeed for a vertical stack"),
        }
    }

    /// Real CoordFlip+Stack (no composite desugar, so `value_axis` stays
    /// `None` — the swap is structural, baked into the encoding by
    /// `prepare::build_layers` before this code ever runs): `x` now holds
    /// the value column, `y` the category. `spec.coord` here is
    /// `Some(CoordKind::Flip)` — exactly what `prepare::build_layers`'s
    /// `rendering_spec` retains (the encoding gets swapped, but `coord`
    /// passes through unchanged via struct-update syntax), which is the
    /// real signal `axis_batch_for_x`/`axis_batch_for_y` now read (GH #77
    /// follow-up) instead of guessing. `axis_batch_for_x` must resolve
    /// this correctly through the `None`-fallback path (same as the
    /// draw-time `apply_stack` call); `axis_batch_for_y` must gate out
    /// (no-op) since y is now the category.
    #[test]
    fn axis_batch_for_x_widens_through_coord_flip_fallback() {
        // Mirrors stack_coord_flipped_uses_x_as_value_column's batch shape:
        // x=val (numeric, the value after the CoordFlip swap), y=cat
        // (categorical), exactly what `prepare::build_layers` produces.
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, false),
            Field::new("val", DataType::Float64, false),
            Field::new("grp", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "a", "b", "b"])),
                Arc::new(Float64Array::from(vec![3.0, 5.0, 2.0, 4.0])),
                Arc::new(StringArray::from(vec!["x", "y", "x", "y"])),
            ],
        )
        .unwrap();

        let enc = enc_xy("val", "cat", Some("grp"));
        let spec = crate::spec::chart::ChartSpec {
            data: Default::default(),
            mark: crate::spec::mark::Mark::Bar,
            encoding: enc,
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: Some(crate::spec::coord::CoordKind::Flip),
            mark_style: None,
            position: Some(PositionAdjust::Stack {
                by: Some("grp".into()),
                offset: StackOffset::Zero,
                anchor: StackAnchor::Top,
                value_axis: None,
            }),
            title: None,
            axis_x: None,
            axis_y: None,
            selections: Vec::new(),
            conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        };

        let x_result = axis_batch_for_x(&spec, "val", &batch);
        match x_result {
            Cow::Owned(out) => {
                let val_idx = out.schema().index_of("val").unwrap();
                let va = out.column(val_idx).as_any().downcast_ref::<Float64Array>().unwrap();
                // Same expected values as stack_coord_flipped_uses_x_as_value_column.
                assert_eq!(va.value(0), 3.0);
                assert_eq!(va.value(1), 8.0);
                assert_eq!(va.value(2), 2.0);
                assert_eq!(va.value(3), 6.0);
            }
            Cow::Borrowed(_) => panic!("expected x-side widening to succeed through the CoordFlip fallback"),
        }

        let y_result = axis_batch_for_y(&spec, "cat", &batch);
        assert!(matches!(y_result, Cow::Borrowed(_)), "y-side must be a no-op once value has flipped to x");
    }

    /// Pin for an intentional behavior change from this GH #77 follow-up
    /// (not silent drift): `axis_batch_for_y` used to hardcode
    /// `coord_flipped = false` unconditionally, so a real-CoordFlip'd Stack
    /// whose category column happened to be Float64 (not Utf8 — e.g. a
    /// numeric bin-edge-style category, which passes `apply_stack`'s
    /// value-column type check same as a genuine value column would) got
    /// spuriously cumulated and Y-widened here, even though the value had
    /// actually flipped onto X. That path never rendered correctly anyway
    /// (the draw-time `apply_stack` call — which always used the *real*
    /// `coord_flipped` — cumulated the correct X column, so the drawer's
    /// x-domain was fine and this stray Y-widening was inert at best, but
    /// silently corrupting at worst if a caller ever inspected the
    /// preview batch's y column). Reading the real `coord_flipped` here
    /// now gates this out deliberately: this is a bug fix, not a
    /// regression.
    #[test]
    fn axis_batch_for_y_gates_out_for_flipped_numeric_category_stack() {
        // Real CoordFlip, numeric category (Float64, NOT Utf8) — the exact
        // shape that used to slip past the old hardcoded-false check.
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Float64, false),
            Field::new("val", DataType::Float64, false),
            Field::new("grp", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![0.0, 0.0, 1.0, 1.0])),
                Arc::new(Float64Array::from(vec![3.0, 5.0, 2.0, 4.0])),
                Arc::new(StringArray::from(vec!["x", "y", "x", "y"])),
            ],
        )
        .unwrap();

        let enc = enc_xy("val", "cat", Some("grp"));
        let spec = crate::spec::chart::ChartSpec {
            data: Default::default(),
            mark: crate::spec::mark::Mark::Bar,
            encoding: enc,
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: Some(crate::spec::coord::CoordKind::Flip),
            mark_style: None,
            position: Some(PositionAdjust::Stack {
                by: Some("grp".into()),
                offset: StackOffset::Zero,
                anchor: StackAnchor::Top,
                value_axis: None,
            }),
            title: None,
            axis_x: None,
            axis_y: None,
            selections: Vec::new(),
            conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        };

        let y_result = axis_batch_for_y(&spec, "cat", &batch);
        assert!(
            matches!(y_result, Cow::Borrowed(_)),
            "axis_batch_for_y must gate out (Cow::Borrowed) for a flipped numeric-category \
             stack — a deliberate bug fix, not silent drift: value resolved to x here, so y \
             (the category) must never be cumulated or widened"
        );
    }

    #[test]
    fn stack_coord_flipped_uses_x_as_value_column() {
        // When coord_flipped=true, the prepare step has swapped x/y in the
        // encoding. The encoding now has:
        //   x = original y (numeric, the value to cumulate)
        //   y = original x (categorical, the grouping axis)
        // Stack should cumulate encoding.x (the value column) grouped by
        // encoding.y (the category column).
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, false),
            Field::new("val", DataType::Float64, false),
            Field::new("grp", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "a", "b", "b"])),
                Arc::new(Float64Array::from(vec![3.0, 5.0, 2.0, 4.0])),
                Arc::new(StringArray::from(vec!["x", "y", "x", "y"])),
            ],
        )
        .unwrap();

        // After coord flip, the encoding has x=val (numeric), y=cat (categorical).
        // This mirrors what prepare.rs does: swap the original x/y.
        let enc = enc_xy("val", "cat", Some("grp"));
        let s = dummy_scales();
        let pos = PositionAdjust::Stack {
            by: Some("grp".into()),
            offset: StackOffset::Zero,
            anchor: StackAnchor::Top,
            value_axis: None,
        };

        // With coord_flipped=true, Stack should treat encoding.x ("val") as
        // the value column and encoding.y ("cat") as the category column.
        let out = apply_position(&batch, Some(&pos), &s, &enc, true, &mut Vec::new()).unwrap();

        // The value column ("val") should be cumulated.
        let val_idx = out.schema().index_of("val").unwrap();
        let va = out.column(val_idx).as_any().downcast_ref::<Float64Array>().unwrap();
        // Group order: x=0, y=1. At cat="a": x=3→3, y=5→8. At cat="b": x=2→2, y=4→6.
        assert_eq!(va.value(0), 3.0);  // cat=a, grp=x → first in stack
        assert_eq!(va.value(1), 8.0);  // cat=a, grp=y → 3+5=8
        assert_eq!(va.value(2), 2.0);  // cat=b, grp=x → first in stack
        assert_eq!(va.value(3), 6.0);  // cat=b, grp=y → 2+4=6

        // __stack_y_base__ should be present with correct bases.
        let base_idx = out.schema().index_of("__stack_y_base__").unwrap();
        let ba = out.column(base_idx).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(ba.value(0), 0.0);
        assert_eq!(ba.value(1), 3.0);
        assert_eq!(ba.value(2), 0.0);
        assert_eq!(ba.value(3), 2.0);
    }

    #[test]
    fn stack_coord_flipped_false_still_uses_y_as_value() {
        // Verify that coord_flipped=false preserves the original behavior:
        // encoding.y is the value column, encoding.x is the category.
        let b = batch_xyg();
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Stack {
            by: Some("g".into()),
            offset: StackOffset::Zero,
            anchor: StackAnchor::Top,
            value_axis: None,
        };
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut Vec::new()).unwrap();
        let ya = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        // Same assertions as stack_zero_accumulates_y.
        assert_eq!(ya.value(0), 10.0);
        assert_eq!(ya.value(1), 30.0);
        assert_eq!(ya.value(2), 30.0);
        assert_eq!(ya.value(3), 70.0);
    }

    // --- R3: user-facing channel names under CoordFlip (Stack's role labels) ---

    /// Unflipped: `Stack`'s missing-encoding messages hardcode "category (x)" /
    /// "value (y)" exactly as before this fix — byte-identical default.
    #[test]
    fn stack_missing_encoding_messages_unchanged_when_not_flipped() {
        let b = batch_xyg();
        let s = dummy_scales();
        let pos = PositionAdjust::Stack {
            by: Some("g".into()),
            offset: StackOffset::Zero,
            anchor: StackAnchor::Top,
            value_axis: None,
        };

        // No category (x) encoding.
        let enc_no_x = Encoding { x: None, ..enc_xy("x", "y", Some("g")) };
        let err = apply_position(&b, Some(&pos), &s, &enc_no_x, false, &mut Vec::new()).unwrap_err();
        assert_eq!(format!("{err}"), "Stack: category (x) encoding required");

        // No value (y) encoding.
        let enc_no_y = Encoding { y: None, ..enc_xy("x", "y", Some("g")) };
        let err = apply_position(&b, Some(&pos), &s, &enc_no_y, false, &mut Vec::new()).unwrap_err();
        assert_eq!(format!("{err}"), "Stack: value (y) encoding required");
    }

    /// R3: under `CoordFlip` (no `value_axis` override), `apply_stack` reads
    /// the value column from the RESOLVED (post-flip) x slot and the category
    /// column from the RESOLVED y slot (`stack_coord_flipped_uses_x_as_value_column`
    /// above pins that resolution). A missing RESOLVED x (value) slot is, from
    /// the user's own perspective, their unset Y encoding — and a missing
    /// RESOLVED y (category) slot is their unset X encoding. The message must
    /// name the channel the user themselves left unset, not the internal
    /// post-flip slot `apply_stack` actually checked.
    #[test]
    fn stack_missing_encoding_messages_name_users_channel_under_flip() {
        let b = batch_xyg();
        let s = dummy_scales();
        let pos = PositionAdjust::Stack {
            by: Some("g".into()),
            offset: StackOffset::Zero,
            anchor: StackAnchor::Top,
            value_axis: None,
        };

        // Post-flip x (value slot) missing → the user's own y is unset.
        let enc_no_x = Encoding { x: None, ..enc_xy("x", "y", Some("g")) };
        let err = apply_position(&b, Some(&pos), &s, &enc_no_x, true, &mut Vec::new()).unwrap_err();
        assert_eq!(
            format!("{err}"),
            "Stack: value (y) encoding required",
            "post-flip x missing must be reported as the user's own y"
        );

        // Post-flip y (category slot) missing → the user's own x is unset.
        let enc_no_y = Encoding { y: None, ..enc_xy("x", "y", Some("g")) };
        let err = apply_position(&b, Some(&pos), &s, &enc_no_y, true, &mut Vec::new()).unwrap_err();
        assert_eq!(
            format!("{err}"),
            "Stack: category (x) encoding required",
            "post-flip y missing must be reported as the user's own x"
        );
    }

    /// R3: the Stack dtype-check messages ("{channel} must be Float64...",
    /// "{channel} column must be Float64 or Utf8") swap consistently with the
    /// value/category role under flip too — not just the missing-encoding pair.
    #[test]
    fn stack_dtype_error_messages_name_users_channel_under_flip() {
        use arrow::array::BooleanArray;

        // Post-flip x (value slot) is Boolean — an unsupported value dtype.
        // Under flip this is the user's own y column.
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Boolean, false),
            Field::new("y", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(BooleanArray::from(vec![true, false])),
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
                Arc::new(StringArray::from(vec!["a", "b"])),
            ],
        )
        .unwrap();
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Stack {
            by: Some("g".into()),
            offset: StackOffset::Zero,
            anchor: StackAnchor::Top,
            value_axis: None,
        };
        let err = apply_position(&batch, Some(&pos), &s, &enc, true, &mut Vec::new()).unwrap_err();
        let text = format!("{err}");
        assert!(
            text.starts_with("Stack: y must be Float64, UInt64, or a signed integer type"),
            "post-flip x's bad dtype must be reported as the user's own y; got: {text}"
        );
    }

    /// R3 (#89 part C restructure): the Stack CATEGORY-dtype check
    /// ("{channel} column must be Float64 or Utf8") also swaps with the
    /// value/category role under flip — sibling coverage to
    /// `stack_dtype_error_messages_name_users_channel_under_flip` above, which
    /// only pins the VALUE-dtype message. `PositionAdjustReason::CategoryDtype`
    /// was not previously exercised by any test.
    ///
    /// Also pins the VALUE-dtype message's FULL rendered string (quality-review
    /// cycle 2 required fix): `ValueDtype`'s literal was reshaped into a
    /// `\`-continuation join in `render/mod.rs`'s `Display` impl, and
    /// `stack_dtype_error_messages_name_users_channel_under_flip`'s
    /// `starts_with("Stack: y must be Float64, UInt64, or a signed integer type")`
    /// stops one character before that join, leaving
    /// `(Int8/Int16/Int32/Int64); got Boolean` — the exact part a lost or
    /// doubled space in the reformatted join would corrupt — entirely
    /// unpinned. The two `assert_eq!` blocks at the end of this fn (unflipped,
    /// then flipped) close that gap, matching HEAD's `format!` output verified
    /// byte-identical under `rustc`.
    #[test]
    fn stack_category_dtype_error_message_names_users_channel_under_flip() {
        use arrow::array::BooleanArray;

        // Unflipped: the resolved category slot IS the user's own x — Boolean
        // "x" is directly the bad column.
        let schema_unflipped = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Boolean, false),
            Field::new("y", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        let batch_unflipped = RecordBatch::try_new(
            schema_unflipped,
            vec![
                Arc::new(BooleanArray::from(vec![true, false])),
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
                Arc::new(StringArray::from(vec!["a", "b"])),
            ],
        )
        .unwrap();
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Stack {
            by: Some("g".into()),
            offset: StackOffset::Zero,
            anchor: StackAnchor::Top,
            value_axis: None,
        };
        let err = apply_position(&batch_unflipped, Some(&pos), &s, &enc, false, &mut Vec::new()).unwrap_err();
        assert_eq!(format!("{err}"), "Stack: x column must be Float64 or Utf8");

        // Flipped: the resolved category slot is post-flip y — Boolean "y" is
        // the bad column, but it must still be reported as the user's own x
        // (the same message text as the unflipped case above).
        let schema_flipped = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Boolean, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        let batch_flipped = RecordBatch::try_new(
            schema_flipped,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
                Arc::new(BooleanArray::from(vec![true, false])),
                Arc::new(StringArray::from(vec!["a", "b"])),
            ],
        )
        .unwrap();
        let err = apply_position(&batch_flipped, Some(&pos), &s, &enc, true, &mut Vec::new()).unwrap_err();
        assert_eq!(
            format!("{err}"),
            "Stack: x column must be Float64 or Utf8",
            "post-flip y (category) dtype failure must be reported as the user's own x"
        );

        // Required fix (quality review cycle 2): pin the VALUE-dtype
        // message's FULL rendered string, exactly, in both arms — the
        // `\`-continuation join in the `ValueDtype` `Display` arm
        // (`render/mod.rs`) had no exact-match test; only `starts_with`,
        // which stops before the `(Int8/Int16/Int32/Int64); got <dtype>` tail.

        // Unflipped: value_enc resolves to encoding.y ("y"); Boolean "y" is
        // the bad value column, reported under its own (identity) name.
        let schema_value_unflipped = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Utf8, false),
            Field::new("y", DataType::Boolean, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        let batch_value_unflipped = RecordBatch::try_new(
            schema_value_unflipped,
            vec![
                Arc::new(StringArray::from(vec!["a", "b"])),
                Arc::new(BooleanArray::from(vec![true, false])),
                Arc::new(StringArray::from(vec!["a", "b"])),
            ],
        )
        .unwrap();
        let err = apply_position(&batch_value_unflipped, Some(&pos), &s, &enc, false, &mut Vec::new()).unwrap_err();
        assert_eq!(
            format!("{err}"),
            "Stack: y must be Float64, UInt64, or a signed integer type (Int8/Int16/Int32/Int64); \
             got Boolean",
            "unflipped: exact VALUE-dtype message, verified byte-identical to HEAD's format! output"
        );

        // Flipped: the same scenario `stack_dtype_error_messages_name_users_channel_under_flip`
        // above exercises via `starts_with` — Boolean "x", coord_flipped=true,
        // reported as the user's own y — pinned here with an exact match.
        let schema_value_flipped = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Boolean, false),
            Field::new("y", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        let batch_value_flipped = RecordBatch::try_new(
            schema_value_flipped,
            vec![
                Arc::new(BooleanArray::from(vec![true, false])),
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
                Arc::new(StringArray::from(vec!["a", "b"])),
            ],
        )
        .unwrap();
        let err = apply_position(&batch_value_flipped, Some(&pos), &s, &enc, true, &mut Vec::new()).unwrap_err();
        assert_eq!(
            format!("{err}"),
            "Stack: y must be Float64, UInt64, or a signed integer type (Int8/Int16/Int32/Int64); \
             got Boolean",
            "post-flip x's bad dtype must be reported as the user's own y — exact match"
        );
    }

    /// R3 sign-off item: a deliberate, narrow unflipped-text correction, NOT
    /// collateral drift. `value_axis: Some(X)` (a composite-mark desugar
    /// override, e.g. `orientation="horizontal"` — unrelated to `CoordFlip`)
    /// makes `value_on_x == true` while `coord_flipped == false`. HEAD's four
    /// Stack role-label messages were hardcoded to the pre-#77 x=category/
    /// y=value assumption and never tracked `value_on_x`, so in exactly this
    /// configuration they were factually wrong even though "unflipped":
    ///   - missing-category: HEAD said `"category (x)"` — but `cat_enc` here
    ///     is `encoding.y` (`value_on_x` swaps the pair), so the true missing
    ///     channel is y. HEAD: `"category (x) encoding required"`.
    ///     Now:  `"category (y) encoding required"`.
    ///   - missing-value:    HEAD said `"value (y)"` — true missing channel is
    ///     x. HEAD: `"value (y) encoding required"`.
    ///     Now:  `"value (x) encoding required"`.
    /// This test pins the corrected (now-accurate) text for both. It is the
    /// ONLY configuration where an "unflipped" (`coord_flipped == false`)
    /// Stack message's text changed from HEAD — every `coord_flipped == false`
    /// `value_axis: None` case (the overwhelming majority of charts) stays
    /// byte-identical, per `stack_missing_encoding_messages_unchanged_when_not_flipped`.
    #[test]
    fn stack_value_axis_override_corrects_stale_role_label_unflipped() {
        let b = batch_xyg();
        let s = dummy_scales();
        let pos = PositionAdjust::Stack {
            by: Some("g".into()),
            offset: StackOffset::Zero,
            anchor: StackAnchor::Top,
            value_axis: Some(StackValueAxis::X),
        };

        // value_axis=X, coord_flipped=false → value_on_x=true → cat_enc=y.
        // Missing y (category) → HEAD wrongly said "(x)"; now correctly "(y)".
        let enc_no_y = Encoding { y: None, ..enc_xy("x", "y", Some("g")) };
        let err = apply_position(&b, Some(&pos), &s, &enc_no_y, false, &mut Vec::new()).unwrap_err();
        assert_eq!(
            format!("{err}"),
            "Stack: category (y) encoding required",
            "HEAD said 'category (x)', which was factually wrong for value_axis=X: \
             cat_enc is encoding.y in this configuration"
        );

        // value_axis=X, coord_flipped=false → value_enc=x. Missing x (value) →
        // HEAD wrongly said "(y)"; now correctly "(x)".
        let enc_no_x = Encoding { x: None, ..enc_xy("x", "y", Some("g")) };
        let err = apply_position(&b, Some(&pos), &s, &enc_no_x, false, &mut Vec::new()).unwrap_err();
        assert_eq!(
            format!("{err}"),
            "Stack: value (x) encoding required",
            "HEAD said 'value (y)', which was factually wrong for value_axis=X: \
             value_enc is encoding.x in this configuration"
        );
    }

    // D4 regression: Stack must accept Int64 (and other signed integer) measure
    // columns — widening them to f64 transparently, identical to UInt64.
    #[test]
    fn stack_int64_measure_accepted_and_widened_to_f64() {
        use arrow::array::Int64Array;

        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Int64, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 1.0, 2.0, 2.0])),
                Arc::new(Int64Array::from(vec![10_i64, 20, 30, 40])),
                Arc::new(StringArray::from(vec!["a", "b", "a", "b"])),
            ],
        )
        .unwrap();

        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Stack {
            by: Some("g".into()),
            offset: StackOffset::Zero,
            anchor: StackAnchor::Top,
            value_axis: None,
        };

        // Must not error — Int64 should be widened to f64.
        let out = apply_position(&batch, Some(&pos), &s, &enc, false, &mut Vec::new())
            .expect("Stack must accept Int64 measure without error");

        // Output y column must be Float64 (promoted from Int64).
        assert_eq!(
            out.schema().field(1).data_type(),
            &DataType::Float64,
            "y column must be promoted to Float64 after stacking"
        );

        let ya = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        // Group order: a=0, b=1. At x=1: a=10→10, b=20→30. At x=2: a=30→30, b=40→70.
        // These are the same cumulative values as the Float64 test (stack_zero_accumulates_y).
        assert_eq!(ya.value(0), 10.0, "x=1 a segment top");
        assert_eq!(ya.value(1), 30.0, "x=1 b segment top");
        assert_eq!(ya.value(2), 30.0, "x=2 a segment top");
        assert_eq!(ya.value(3), 70.0, "x=2 b segment top");
    }

    #[test]
    fn stack_int32_measure_accepted_and_widened_to_f64() {
        use arrow::array::Int32Array;

        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Int32, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 1.0, 2.0, 2.0])),
                Arc::new(Int32Array::from(vec![10_i32, 20, 30, 40])),
                Arc::new(StringArray::from(vec!["a", "b", "a", "b"])),
            ],
        )
        .unwrap();

        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Stack {
            by: Some("g".into()),
            offset: StackOffset::Zero,
            anchor: StackAnchor::Top,
            value_axis: None,
        };

        let out = apply_position(&batch, Some(&pos), &s, &enc, false, &mut Vec::new())
            .expect("Stack must accept Int32 measure without error");

        assert_eq!(out.schema().field(1).data_type(), &DataType::Float64);
        let ya = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(ya.value(0), 10.0);
        assert_eq!(ya.value(1), 30.0);
        assert_eq!(ya.value(2), 30.0);
        assert_eq!(ya.value(3), 70.0);
    }

    // -----------------------------------------------------------------------
    // RSUP-05: one Dodge grouping policy (behavior change).
    //
    // Pre-RSUP-05, a non-Utf8 `by` column hard-errored ("must be Utf8") and a
    // missing/absent `by` column silently no-op'd. Now: int/float/bool dodge
    // resolves through `col_as_ordinal_category_str`; missing or
    // un-categorizable grouping channels warn-and-skip; only "no grouping
    // channel at all" is a silent no-op.
    // -----------------------------------------------------------------------

    /// Build a continuous-x dodge batch whose group column has the given array
    /// type, with two x-bins (1.0, 2.0) and an alternating two-level group.
    fn dodge_batch_with_group(g_field: Field, g_col: ArrayRef) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            g_field,
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 1.0, 2.0, 2.0])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0])),
                g_col,
            ],
        )
        .unwrap()
    }

    /// The four x offsets a two-group, two-bin dodge with padding=0 produces.
    /// Bandwidth = 1.0, two groups → sub_band = 0.5; offsets a=-0.25, b=+0.25.
    const DODGE_ORACLE: [f64; 4] = [0.75, 1.25, 1.75, 2.25];

    fn assert_dodge_oracle(out: &RecordBatch) {
        let xa = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        for (i, want) in DODGE_ORACLE.iter().enumerate() {
            assert!(
                (xa.value(i) - want).abs() < 1e-9,
                "row{i} x={} want={want}",
                xa.value(i)
            );
        }
    }

    #[test]
    fn dodge_int64_by_column_resolves_like_utf8() {
        // Headline drift fix: an Int64 grouping column dodges (was a hard Err).
        // Group values 0,1,0,1 must produce the SAME offsets as the equivalent
        // Utf8 "0"/"1" column.
        use arrow::array::Int64Array;
        let b = dodge_batch_with_group(
            Field::new("g", DataType::Int64, false),
            Arc::new(Int64Array::from(vec![0_i64, 1, 0, 1])),
        );
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Dodge { by: Some("g".into()), padding: 0.0 };
        let mut warnings = Vec::new();
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut warnings).unwrap();
        assert!(warnings.is_empty(), "int64 dodge should not warn: {warnings:?}");
        assert_dodge_oracle(&out);
    }

    #[test]
    fn dodge_float64_by_column_resolves() {
        // Integer-valued Float64 (2000.0) groups dodge identically to ints.
        let b = dodge_batch_with_group(
            Field::new("g", DataType::Float64, false),
            Arc::new(Float64Array::from(vec![1000.0, 2000.0, 1000.0, 2000.0])),
        );
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Dodge { by: Some("g".into()), padding: 0.0 };
        let mut warnings = Vec::new();
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut warnings).unwrap();
        assert!(warnings.is_empty(), "float64 dodge should not warn: {warnings:?}");
        assert_dodge_oracle(&out);
    }

    #[test]
    fn dodge_boolean_by_column_resolves() {
        use arrow::array::BooleanArray;
        let b = dodge_batch_with_group(
            Field::new("g", DataType::Boolean, false),
            Arc::new(BooleanArray::from(vec![false, true, false, true])),
        );
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Dodge { by: Some("g".into()), padding: 0.0 };
        let mut warnings = Vec::new();
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut warnings).unwrap();
        assert!(warnings.is_empty(), "boolean dodge should not warn: {warnings:?}");
        assert_dodge_oracle(&out);
    }

    #[test]
    fn dodge_named_absent_by_column_warns_and_noops() {
        // A named-but-absent `by` column was silent; now it warns and no-ops.
        let b = batch_xyg();
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Dodge { by: Some("nope".into()), padding: 0.0 };
        let mut warnings = Vec::new();
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut warnings).unwrap();
        // Batch unchanged — x column is the original 1.0/1.0/2.0/2.0.
        let xa = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(xa.value(0), 1.0);
        assert_eq!(xa.value(3), 2.0);
        assert_eq!(warnings.len(), 1, "expected exactly one warning");
        match &warnings[0] {
            RenderWarning::PositionAdjustSkipped { adjustment, reason } => {
                assert_eq!(adjustment, "dodge");
                assert!(reason.contains("nope"), "reason was: {reason}");
                assert!(reason.contains("not found"), "reason was: {reason}");
            }
            other => panic!("wrong warning variant: {other:?}"),
        }
    }

    #[test]
    fn dodge_timestamp_by_column_warns_and_noops() {
        // A timestamp `by` column was a hard Err; now it warns and no-ops
        // (consistent with the absent-column case).
        use arrow::array::TimestampMillisecondArray;
        let b = dodge_batch_with_group(
            Field::new(
                "g",
                DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None),
                false,
            ),
            Arc::new(TimestampMillisecondArray::from(vec![0_i64, 1, 0, 1])),
        );
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Dodge { by: Some("g".into()), padding: 0.0 };
        let mut warnings = Vec::new();
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut warnings).unwrap();
        // Batch unchanged.
        let xa = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(xa.value(0), 1.0);
        assert_eq!(xa.value(3), 2.0);
        assert_eq!(warnings.len(), 1, "expected exactly one warning");
        match &warnings[0] {
            RenderWarning::PositionAdjustSkipped { adjustment, reason } => {
                assert_eq!(adjustment, "dodge");
                assert!(
                    reason.contains("cannot group categories"),
                    "reason was: {reason}"
                );
            }
            other => panic!("wrong warning variant: {other:?}"),
        }
    }

    #[test]
    fn dodge_no_by_no_color_is_silent_noop() {
        // The one documented intentional no-op: dodge requested with no
        // grouping channel at all → batch unchanged, ZERO warnings.
        let b = batch_xyg();
        let enc = enc_xy("x", "y", None); // no color encoding
        let s = dummy_scales();
        let pos = PositionAdjust::Dodge { by: None, padding: 0.0 };
        let mut warnings = Vec::new();
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut warnings).unwrap();
        let xa = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(xa.value(0), 1.0);
        assert_eq!(xa.value(3), 2.0);
        assert!(warnings.is_empty(), "no-grouping dodge must not warn: {warnings:?}");
    }

    #[test]
    fn dodge_utf8_by_column_offsets_match_oracle() {
        // Byte-identity pin: the Utf8 path through the new resolver produces the
        // same offsets as the hand-computed oracle (guards the goldens).
        let b = batch_xyg(); // g = ["a","b","a","b"], Utf8
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Dodge { by: Some("g".into()), padding: 0.0 };
        let mut warnings = Vec::new();
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut warnings).unwrap();
        assert!(warnings.is_empty());
        assert_dodge_oracle(&out);
    }

    // -----------------------------------------------------------------------
    // S4FIX / RSUP-05 sibling completion: apply_stack now resolves its `by`
    // grouping channel through the SAME `resolve_group_channel` policy as
    // apply_dodge. Pre-fix, a non-Utf8 stack `by` silently no-op'd (rendered
    // un-stacked, zero diagnostics) while the same column dodged fine. After
    // the fix: int/float/bool stack `by` works; absent/un-categorizable named
    // channels warn-and-skip; only "no grouping channel at all" stays silent.
    // -----------------------------------------------------------------------

    /// Build a continuous-x stack batch whose group column has the given array
    /// type. Two x-bins (1.0, 2.0), an alternating two-level group, and y
    /// values 10/20/30/40 — identical layout to `batch_xyg` so stacked tops
    /// are the well-known [10, 30, 30, 70] oracle regardless of `by` dtype.
    fn stack_batch_with_group(g_field: Field, g_col: ArrayRef) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            g_field,
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 1.0, 2.0, 2.0])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0])),
                g_col,
            ],
        )
        .unwrap()
    }

    /// The cumulative stacked tops a two-group, two-bin stack produces.
    /// Group order a/0 first, b/1 second → x=1: 10,30; x=2: 30,70.
    const STACK_TOP_ORACLE: [f64; 4] = [10.0, 30.0, 30.0, 70.0];

    fn assert_stack_top_oracle(out: &RecordBatch) {
        let ya = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        for (i, want) in STACK_TOP_ORACLE.iter().enumerate() {
            assert!(
                (ya.value(i) - want).abs() < 1e-9,
                "row{i} stack-top={} want={want}",
                ya.value(i)
            );
        }
    }

    #[test]
    fn stack_int64_by_column_resolves_like_utf8() {
        // Headline drift fix: an Int64 grouping column now stacks (was a silent
        // no-op). Group values 0,1,0,1 produce the SAME stacked tops as the
        // equivalent Utf8 "0"/"1" column — the finding's requested "dodge and
        // stack make the same resolution decision" parity, on the stack side.
        use arrow::array::Int64Array;
        let b = stack_batch_with_group(
            Field::new("g", DataType::Int64, false),
            Arc::new(Int64Array::from(vec![0_i64, 1, 0, 1])),
        );
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Stack {
            by: Some("g".into()),
            offset: StackOffset::Zero,
            anchor: StackAnchor::Top,
            value_axis: None,
        };
        let mut warnings = Vec::new();
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut warnings).unwrap();
        assert!(warnings.is_empty(), "int64 stack should not warn: {warnings:?}");
        assert_stack_top_oracle(&out);
    }

    #[test]
    fn stack_named_absent_by_column_warns_and_noops() {
        // A named-but-absent `by` column was a silent no-op; now it warns
        // (PositionAdjustSkipped { adjustment: "stack", … }) and clones.
        let b = batch_xyg();
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Stack {
            by: Some("nope".into()),
            offset: StackOffset::Zero,
            anchor: StackAnchor::Top,
            value_axis: None,
        };
        let mut warnings = Vec::new();
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut warnings).unwrap();
        // Batch unchanged — y column is the original 10/20/30/40 (un-stacked).
        let ya = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(ya.value(0), 10.0);
        assert_eq!(ya.value(3), 40.0);
        assert_eq!(warnings.len(), 1, "expected exactly one warning");
        match &warnings[0] {
            RenderWarning::PositionAdjustSkipped { adjustment, reason } => {
                assert_eq!(adjustment, "stack");
                assert!(reason.contains("nope"), "reason was: {reason}");
                assert!(reason.contains("not found"), "reason was: {reason}");
            }
            other => panic!("wrong warning variant: {other:?}"),
        }
    }

    #[test]
    fn stack_timestamp_by_column_warns_and_noops() {
        // A timestamp `by` column was a silent no-op; now it warns
        // (un-categorizable) and clones — no crash.
        use arrow::array::TimestampMillisecondArray;
        let b = stack_batch_with_group(
            Field::new(
                "g",
                DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None),
                false,
            ),
            Arc::new(TimestampMillisecondArray::from(vec![0_i64, 1, 0, 1])),
        );
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Stack {
            by: Some("g".into()),
            offset: StackOffset::Zero,
            anchor: StackAnchor::Top,
            value_axis: None,
        };
        let mut warnings = Vec::new();
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut warnings).unwrap();
        // Batch unchanged — y column is the original 10/20/30/40.
        let ya = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(ya.value(0), 10.0);
        assert_eq!(ya.value(3), 40.0);
        assert_eq!(warnings.len(), 1, "expected exactly one warning");
        match &warnings[0] {
            RenderWarning::PositionAdjustSkipped { adjustment, reason } => {
                assert_eq!(adjustment, "stack");
                assert!(
                    reason.contains("cannot group categories"),
                    "reason was: {reason}"
                );
            }
            other => panic!("wrong warning variant: {other:?}"),
        }
    }

    #[test]
    fn stack_no_by_no_color_is_silent_noop() {
        // The one documented intentional no-op: stack requested with no
        // grouping channel at all → batch unchanged, ZERO warnings (the
        // preserved Case 1). Without color the resolver returns None silently.
        let b = batch_xyg();
        let enc = enc_xy("x", "y", None); // no color encoding
        let s = dummy_scales();
        let pos = PositionAdjust::Stack {
            by: None,
            offset: StackOffset::Zero,
            anchor: StackAnchor::Top,
            value_axis: None,
        };
        let mut warnings = Vec::new();
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut warnings).unwrap();
        let ya = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(ya.value(0), 10.0);
        assert_eq!(ya.value(3), 40.0);
        assert!(warnings.is_empty(), "no-grouping stack must not warn: {warnings:?}");
    }

    #[test]
    fn stack_utf8_by_column_tops_match_oracle() {
        // Byte-identity pin: the Utf8 path through the shared resolver produces
        // the same stacked tops as the hand-computed oracle (guards the Utf8
        // stack goldens — the common case is byte-identical post-fix).
        let b = batch_xyg(); // g = ["a","b","a","b"], Utf8
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Stack {
            by: Some("g".into()),
            offset: StackOffset::Zero,
            anchor: StackAnchor::Top,
            value_axis: None,
        };
        let mut warnings = Vec::new();
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut warnings).unwrap();
        assert!(warnings.is_empty());
        assert_stack_top_oracle(&out);
    }

    // =======================================================================
    // BUG HUNT — Step 4: position-adjustment `by`-channel widening.
    //
    // The campaign unified apply_stack/apply_dodge `by` resolution into the
    // shared `resolve_group_channel`, which now accepts Int*/UInt*/Float*/Bool
    // `by` columns (previously stack silently no-op'd on anything non-Utf8).
    // This block probes the newly-reachable input surface for correctness and
    // panic-safety: single-unique groups, all-null columns, 0-row batches,
    // NaN/±inf Float64 `by`, Boolean `by`, int/float string collisions,
    // `by` == value column, color-vs-explicit-`by` precedence, and timestamp
    // `by` on a 0-row batch. Every assertion checks a real invariant (group
    // counts, stacked-top sums, exactly-once warning with the right label),
    // never `is_ok()`.
    // =======================================================================

    /// Helper: extract the y (value) column from a stack output as a Vec<f64>.
    fn stack_tops(out: &RecordBatch, yi: usize) -> Vec<f64> {
        let ya = out.column(yi).as_any().downcast_ref::<Float64Array>().unwrap();
        (0..ya.len()).map(|i| ya.value(i)).collect()
    }

    /// Helper: extract the __stack_y_base__ column as a Vec<f64>.
    fn stack_bases(out: &RecordBatch) -> Vec<f64> {
        let idx = out.schema().index_of("__stack_y_base__").unwrap();
        let ba = out.column(idx).as_any().downcast_ref::<Float64Array>().unwrap();
        (0..ba.len()).map(|i| ba.value(i)).collect()
    }

    // ── Single unique `by` value (one group) ────────────────────────────────

    #[test]
    fn bughunt_stack_single_unique_by_value_one_segment_per_bin() {
        // A `by` column with one distinct value → one group → each x-bin has a
        // single segment whose top == the raw value and base == 0. No warning.
        use arrow::array::Int64Array;
        let b = stack_batch_with_group(
            Field::new("g", DataType::Int64, false),
            Arc::new(Int64Array::from(vec![7_i64, 7, 7, 7])),
        );
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Stack {
            by: Some("g".into()),
            offset: StackOffset::Zero,
            anchor: StackAnchor::Top,
            value_axis: None,
        };
        let mut warnings = Vec::new();
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut warnings).unwrap();
        assert!(warnings.is_empty(), "single-group stack must not warn: {warnings:?}");
        // One group, but each x-bin still has TWO rows: stack accumulates every
        // row within a bin (not just across distinct groups). bin x=1: 10 then
        // 20 → tops 10, 30; bin x=2: 30 then 40 → tops 30, 70.
        assert_eq!(stack_tops(&out, 1), vec![10.0, 30.0, 30.0, 70.0]);
        assert_eq!(stack_bases(&out), vec![0.0, 10.0, 0.0, 30.0]);
    }

    #[test]
    fn bughunt_dodge_single_unique_by_value_is_noop_no_warning() {
        // Dodge with one distinct `by` value → n_groups == 1 → clone, no offset,
        // and (per the documented policy) NO warning.
        use arrow::array::Int64Array;
        let b = dodge_batch_with_group(
            Field::new("g", DataType::Int64, false),
            Arc::new(Int64Array::from(vec![7_i64, 7, 7, 7])),
        );
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Dodge { by: Some("g".into()), padding: 0.0 };
        let mut warnings = Vec::new();
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut warnings).unwrap();
        assert!(warnings.is_empty(), "single-group dodge must not warn: {warnings:?}");
        let xa = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!((xa.value(0), xa.value(3)), (1.0, 2.0), "x must be unchanged");
    }

    // ── All-null `by` column (every row null → "") ──────────────────────────

    #[test]
    fn bughunt_stack_all_null_by_collapses_to_single_group() {
        // Every row null → "" via the resolver's null→"" mapping → ONE group.
        // Stack therefore puts one segment per bin; no panic, no warning.
        use arrow::array::Int64Array;
        let b = stack_batch_with_group(
            Field::new("g", DataType::Int64, true),
            Arc::new(Int64Array::from(vec![None, None, None, None])),
        );
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Stack {
            by: Some("g".into()),
            offset: StackOffset::Zero,
            anchor: StackAnchor::Top,
            value_axis: None,
        };
        let mut warnings = Vec::new();
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut warnings).unwrap();
        assert!(warnings.is_empty(), "all-null `by` must not warn: {warnings:?}");
        // All rows in one ("") group, but each x-bin still has two rows that
        // accumulate: bin x=1 → 10,30; bin x=2 → 30,70.
        assert_eq!(stack_tops(&out, 1), vec![10.0, 30.0, 30.0, 70.0]);
        assert_eq!(stack_bases(&out), vec![0.0, 10.0, 0.0, 30.0]);
    }

    #[test]
    fn bughunt_dodge_all_null_by_is_noop_no_warning() {
        // All-null `by` → "" → one group → dodge no-ops with no warning.
        use arrow::array::Int64Array;
        let b = dodge_batch_with_group(
            Field::new("g", DataType::Int64, true),
            Arc::new(Int64Array::from(vec![None, None, None, None])),
        );
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Dodge { by: Some("g".into()), padding: 0.0 };
        let mut warnings = Vec::new();
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut warnings).unwrap();
        assert!(warnings.is_empty(), "all-null `by` dodge must not warn: {warnings:?}");
        let xa = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!((xa.value(0), xa.value(3)), (1.0, 2.0));
    }

    #[test]
    fn bughunt_resolver_maps_every_null_row_to_empty_string() {
        // Direct probe of resolve_group_channel: a `by` column with a mix of
        // null and real values maps every null → "" (documented pre-RSUP-05
        // key), and a genuine empty-string Utf8 value COLLIDES with null into
        // the same "" group. This pins the documented collision so a future
        // change to NULL handling is caught.
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("g", DataType::Utf8, true),
        ]));
        let b = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 1.0, 2.0, 2.0])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0])),
                Arc::new(StringArray::from(vec![Some("a"), None, Some(""), Some("a")])),
            ],
        )
        .unwrap();
        let enc = enc_xy("x", "y", Some("g"));
        let mut warnings = Vec::new();
        let cats = resolve_group_channel(&b, Some("g"), &enc, "dodge", &mut warnings).unwrap();
        assert!(warnings.is_empty());
        // null (row 1) and "" (row 2) both → "" — documented collision.
        assert_eq!(cats, vec!["a", "", "", "a"]);
    }

    // ── Empty (0-row) RecordBatch ───────────────────────────────────────────

    fn empty_xyg(g_field: Field, g_col: ArrayRef) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            g_field,
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(Vec::<f64>::new())),
                Arc::new(Float64Array::from(Vec::<f64>::new())),
                g_col,
            ],
        )
        .unwrap()
    }

    #[test]
    fn bughunt_stack_zero_row_batch_appends_base_column_no_panic() {
        // 0-row batch: resolver returns Some(empty); stack's row loops never
        // execute; the output must still gain a Float64 __stack_y_base__ column
        // and a Float64-widened y column, with 0 rows. No panic, no warning.
        use arrow::array::Int64Array;
        let b = empty_xyg(
            Field::new("g", DataType::Int64, false),
            Arc::new(Int64Array::from(Vec::<i64>::new())),
        );
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Stack {
            by: Some("g".into()),
            offset: StackOffset::Zero,
            anchor: StackAnchor::Top,
            value_axis: None,
        };
        let mut warnings = Vec::new();
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut warnings).unwrap();
        assert!(warnings.is_empty(), "0-row stack must not warn: {warnings:?}");
        assert_eq!(out.num_rows(), 0);
        let base_idx = out.schema().index_of("__stack_y_base__")
            .expect("__stack_y_base__ must be present even for 0 rows");
        assert_eq!(out.column(base_idx).data_type(), &DataType::Float64);
        assert_eq!(out.schema().field(1).data_type(), &DataType::Float64);
    }

    #[test]
    fn bughunt_dodge_zero_row_batch_is_noop_no_panic() {
        // 0-row continuous-x dodge: uniques is empty → < 2 → clone. No panic.
        use arrow::array::Int64Array;
        let b = empty_xyg(
            Field::new("g", DataType::Int64, false),
            Arc::new(Int64Array::from(Vec::<i64>::new())),
        );
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Dodge { by: Some("g".into()), padding: 0.0 };
        let mut warnings = Vec::new();
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut warnings).unwrap();
        assert!(warnings.is_empty());
        assert_eq!(out.num_rows(), 0);
        assert_eq!(out.num_columns(), 3, "0-row dodge must clone unchanged");
    }

    #[test]
    fn bughunt_stack_zero_row_timestamp_by_warns_once_no_panic() {
        // 0-row batch with an un-categorizable (timestamp) `by`: the resolver
        // must still warn exactly once (the dtype check precedes row access),
        // and must not panic on the empty column.
        use arrow::array::TimestampMillisecondArray;
        let b = empty_xyg(
            Field::new(
                "g",
                DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None),
                false,
            ),
            Arc::new(TimestampMillisecondArray::from(Vec::<i64>::new())),
        );
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Stack {
            by: Some("g".into()),
            offset: StackOffset::Zero,
            anchor: StackAnchor::Top,
            value_axis: None,
        };
        let mut warnings = Vec::new();
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut warnings).unwrap();
        assert_eq!(out.num_rows(), 0);
        assert_eq!(warnings.len(), 1, "exactly one warning expected");
        match &warnings[0] {
            RenderWarning::PositionAdjustSkipped { adjustment, reason } => {
                assert_eq!(adjustment, "stack");
                assert!(reason.contains("cannot group categories"), "reason: {reason}");
            }
            other => panic!("wrong warning variant: {other:?}"),
        }
    }

    // ── Float64 `by` with NaN / ±inf ────────────────────────────────────────

    #[test]
    fn bughunt_stack_nan_by_groups_nan_together_distinct_from_finite() {
        // Float64 `by` = [NaN, 1.0, NaN, 1.0] over x = [1,1,2,2]. NaN floats
        // stringify to "NaN" via float_as_ordinal_str, so the two NaN rows
        // group together (even though NaN != NaN), distinct from the "1" group.
        // Each bin has one NaN + one finite row → two segments stacked.
        let b = stack_batch_with_group(
            Field::new("g", DataType::Float64, false),
            Arc::new(Float64Array::from(vec![f64::NAN, 1.0, f64::NAN, 1.0])),
        );
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Stack {
            by: Some("g".into()),
            offset: StackOffset::Zero,
            anchor: StackAnchor::Top,
            value_axis: None,
        };
        let mut warnings = Vec::new();
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut warnings).unwrap();
        assert!(warnings.is_empty(), "NaN-float `by` must categorize, not warn: {warnings:?}");
        // Group order first-appearance: "NaN"=0, "1"=1.
        // bin x=1: rows 0("NaN",10) then 1("1",20) → tops 10, 30.
        // bin x=2: rows 2("NaN",30) then 3("1",40) → tops 30, 70.
        let tops = stack_tops(&out, 1);
        assert_eq!(tops, vec![10.0, 30.0, 30.0, 70.0]);
        // No NaN may leak into the stacked output (the NaN is only a GROUP KEY,
        // never a value): all tops/bases are finite.
        assert!(tops.iter().all(|v| v.is_finite()), "no NaN in stacked tops: {tops:?}");
        assert!(stack_bases(&out).iter().all(|v| v.is_finite()));
    }

    #[test]
    fn bughunt_stack_pos_neg_inf_by_are_distinct_groups() {
        // +inf and -inf as `by` values must form two DISTINCT groups
        // ("inf" vs "-inf"), not collapse together. x = [1,1,2,2].
        let b = stack_batch_with_group(
            Field::new("g", DataType::Float64, false),
            Arc::new(Float64Array::from(vec![
                f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY,
            ])),
        );
        let enc = enc_xy("x", "y", Some("g"));
        let mut warnings = Vec::new();
        let cats = resolve_group_channel(&b, Some("g"), &enc, "stack", &mut warnings).unwrap();
        assert!(warnings.is_empty());
        assert_eq!(cats, vec!["inf", "-inf", "inf", "-inf"]);
        // Two distinct group keys, not one.
        let distinct: std::collections::HashSet<_> = cats.iter().collect();
        assert_eq!(distinct.len(), 2, "inf and -inf must be distinct groups");
    }

    #[test]
    fn bughunt_dodge_nan_by_produces_finite_offsets_no_nan_in_x() {
        // A NaN group KEY must not poison the dodge x offsets: offsets are a
        // function of group INDEX (0,1,…), never the key value. All output x
        // must be finite.
        let b = dodge_batch_with_group(
            Field::new("g", DataType::Float64, false),
            Arc::new(Float64Array::from(vec![f64::NAN, 1.0, f64::NAN, 1.0])),
        );
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Dodge { by: Some("g".into()), padding: 0.0 };
        let mut warnings = Vec::new();
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut warnings).unwrap();
        assert!(warnings.is_empty());
        let xa = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        for i in 0..4 {
            assert!(xa.value(i).is_finite(), "x[{i}]={} must be finite", xa.value(i));
        }
        // Same 2-group oracle as any 2-level dodge: NaN group=0, "1" group=1.
        assert_dodge_oracle(&out);
    }

    // ── Boolean `by` for stack ──────────────────────────────────────────────

    #[test]
    fn bughunt_stack_boolean_by_resolves_like_two_groups() {
        // Boolean `by` stringifies to "false"/"true"; first-appearance order
        // gives false=0, true=1. Over x=[1,1,2,2], by=[false,true,false,true]
        // the stacked tops match the canonical [10,30,30,70] oracle.
        use arrow::array::BooleanArray;
        let b = stack_batch_with_group(
            Field::new("g", DataType::Boolean, false),
            Arc::new(BooleanArray::from(vec![false, true, false, true])),
        );
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Stack {
            by: Some("g".into()),
            offset: StackOffset::Zero,
            anchor: StackAnchor::Top,
            value_axis: None,
        };
        let mut warnings = Vec::new();
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut warnings).unwrap();
        assert!(warnings.is_empty(), "boolean stack must not warn: {warnings:?}");
        assert_stack_top_oracle(&out);
    }

    // ── Mixed int / float string collision parity ───────────────────────────

    #[test]
    fn bughunt_int_one_and_float_one_stringify_identically() {
        // An Int64 `1` and a Float64 `1.0` both categorize to "1". A single
        // column cannot mix dtypes, but this pins that the two siblings produce
        // the SAME group key for the "same" numeric value, so an Int64-keyed
        // golden and a Float64-keyed golden of the same logical data stack the
        // same way.
        use arrow::array::Int64Array;
        let bi = stack_batch_with_group(
            Field::new("g", DataType::Int64, false),
            Arc::new(Int64Array::from(vec![1_i64, 2, 1, 2])),
        );
        let bf = stack_batch_with_group(
            Field::new("g", DataType::Float64, false),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 1.0, 2.0])),
        );
        let enc = enc_xy("x", "y", Some("g"));
        let ci = resolve_group_channel(&bi, Some("g"), &enc, "stack", &mut Vec::new()).unwrap();
        let cf = resolve_group_channel(&bf, Some("g"), &enc, "stack", &mut Vec::new()).unwrap();
        assert_eq!(ci, cf, "int and integer-valued float `by` must share group keys");
        assert_eq!(ci, vec!["1", "2", "1", "2"]);
    }

    // ── Stack where `by` is the SAME column as the value column ──────────────

    #[test]
    fn bughunt_stack_by_equals_value_column_each_row_own_group() {
        // `by` == the y (value) column. Each distinct y becomes its own group;
        // within an x-bin the segments stack in encounter (group-index) order.
        // The value column is rewritten to the cumulative tops in place. This
        // must not panic and must produce coherent monotonic tops per bin.
        let b = batch_xyg(); // x=[1,1,2,2], y=[10,20,30,40]
        // by = "y": group keys "10","20","30","40", all distinct.
        let enc = enc_xy("x", "y", Some("y"));
        let s = dummy_scales();
        let pos = PositionAdjust::Stack {
            by: Some("y".into()),
            offset: StackOffset::Zero,
            anchor: StackAnchor::Top,
            value_axis: None,
        };
        let mut warnings = Vec::new();
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut warnings).unwrap();
        assert!(warnings.is_empty(), "by==value column must not warn: {warnings:?}");
        // bin x=1: 10 then 20 → tops 10, 30; bin x=2: 30 then 40 → 30, 70.
        assert_eq!(stack_tops(&out, 1), vec![10.0, 30.0, 30.0, 70.0]);
        assert_eq!(stack_bases(&out), vec![0.0, 10.0, 0.0, 30.0]);
    }

    // ── Color vs explicit `by` precedence ───────────────────────────────────

    #[test]
    fn bughunt_stack_int_by_orders_segments_by_group_across_bins() {
        // Prove the widened Int64 `by` genuinely controls cross-bin stacking
        // ORDER (not just no-ops): group 0 always sits at the bottom of each
        // bin even when its row appears LAST in that bin's source order.
        //   bin x=1: row0 group=1 (y=10), row1 group=0 (y=20)
        //   bin x=2: row2 group=0 (y=30), row3 group=1 (y=40)
        // After sort-by-group: bin x=1 stacks group0 (20) then group1 (10):
        //   row1 base=0 top=20 ; row0 base=20 top=30.
        // bin x=2 stacks group0 (30) then group1 (40):
        //   row2 base=0 top=30 ; row3 base=30 top=70.
        use arrow::array::Int64Array;
        let b = stack_batch_with_group(
            Field::new("g", DataType::Int64, false),
            Arc::new(Int64Array::from(vec![1_i64, 0, 0, 1])),
        );
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Stack {
            by: Some("g".into()),
            offset: StackOffset::Zero,
            anchor: StackAnchor::Top,
            value_axis: None,
        };
        let mut warnings = Vec::new();
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut warnings).unwrap();
        assert!(warnings.is_empty());
        // Group first-appearance: "1"=0, "0"=1 (row0's value 1 appears first).
        // So segment ordering is by that group index, NOT numeric value.
        // bin x=1: row0(g="1",idx0,y10) bottom, row1(g="0",idx1,y20) on top.
        //   row0 base=0 top=10 ; row1 base=10 top=30.
        // bin x=2: row2(g="0",idx1,y30), row3(g="1",idx0,y40) → sort by idx:
        //   row3(idx0,y40) bottom base=0 top=40 ; row2(idx1,y30) base=40 top=70.
        assert_eq!(stack_tops(&out, 1), vec![10.0, 30.0, 70.0, 40.0]);
        assert_eq!(stack_bases(&out), vec![0.0, 10.0, 40.0, 0.0]);
    }

    #[test]
    fn bughunt_explicit_by_overrides_color_for_grouping() {
        // When both an explicit `by` AND a color encoding are present, the
        // explicit `by` wins (resolve_group_channel: by_field, else color).
        // Build a batch where `by` and color would give DIFFERENT groupings and
        // assert the stack follows `by`, not color.
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("by_col", DataType::Utf8, false),
            Field::new("color_col", DataType::Utf8, false),
        ]));
        let b = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 1.0, 2.0, 2.0])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0])),
                // `by` splits each bin into two groups a/b.
                Arc::new(StringArray::from(vec!["a", "b", "a", "b"])),
                // color would put EVERY row in one group "z" → un-stacked.
                Arc::new(StringArray::from(vec!["z", "z", "z", "z"])),
            ],
        )
        .unwrap();
        // Encoding carries the color column; the position carries explicit `by`.
        let enc = Encoding {
            x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            color: Some(EncodingSpec { field: "color_col".into(), type_: None, ..Default::default() }),
            ..Default::default()
        };
        let s = dummy_scales();
        let pos = PositionAdjust::Stack {
            by: Some("by_col".into()),
            offset: StackOffset::Zero,
            anchor: StackAnchor::Top,
            value_axis: None,
        };
        let mut warnings = Vec::new();
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut warnings).unwrap();
        assert!(warnings.is_empty());
        // Grouped by `by_col` (a/b) → canonical [10,30,30,70]. If color had won,
        // every bin would be a single group "z" → tops would equal raw values
        // (10,20,30,40), which we explicitly reject here.
        let tops = stack_tops(&out, 1);
        assert_eq!(tops, vec![10.0, 30.0, 30.0, 70.0], "explicit `by` must override color");
        assert_ne!(tops[1], 20.0, "color must NOT have won the grouping");
    }

    #[test]
    fn bughunt_color_used_when_no_explicit_by() {
        // The dual: with `by: None` but a color encoding present, the resolver
        // falls through to color. So a color-only stack groups by color.
        let b = batch_xyg(); // color "g" = ["a","b","a","b"]
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Stack {
            by: None,
            offset: StackOffset::Zero,
            anchor: StackAnchor::Top,
            value_axis: None,
        };
        let mut warnings = Vec::new();
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut warnings).unwrap();
        assert!(warnings.is_empty());
        assert_stack_top_oracle(&out);
    }

    // ── Normalize numeric edge: a bin summing to zero ───────────────────────

    #[test]
    fn bughunt_stack_normalize_bin_total_zero_no_div_by_zero() {
        // A bin whose values cancel to total 0.0 (10 and -10). The `total != 0`
        // guard must make every segment 0.0, never NaN/inf from a 0/0 division.
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        let b = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 1.0])),
                Arc::new(Float64Array::from(vec![10.0, -10.0])),
                Arc::new(StringArray::from(vec!["a", "b"])),
            ],
        )
        .unwrap();
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Stack {
            by: Some("g".into()),
            offset: StackOffset::Normalize,
            anchor: StackAnchor::Top,
            value_axis: None,
        };
        let mut warnings = Vec::new();
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut warnings).unwrap();
        assert!(warnings.is_empty());
        let tops = stack_tops(&out, 1);
        assert!(
            tops.iter().all(|v| v.is_finite()),
            "zero-total normalize must not yield NaN/inf, got {tops:?}"
        );
        assert_eq!(tops, vec![0.0, 0.0], "all segments collapse to 0 when total is 0");
    }

    #[test]
    fn resolve_group_channel_same_decision_for_dodge_and_stack() {
        // Unification proof: the SHARED resolver returns the same Option-shape
        // for the dodge-label and stack-label inputs across every case — there
        // is one policy, not two copies. The only per-adjustment difference is
        // the warning label, asserted separately.
        use arrow::array::{BooleanArray, Int64Array, TimestampMillisecondArray};

        // Case 1: no by + no color → None, no warning, for both labels.
        let no_color = enc_xy("x", "y", None);
        let b = batch_xyg();
        let mut wd = Vec::new();
        let mut ws = Vec::new();
        let rd = resolve_group_channel(&b, None, &no_color, "dodge", &mut wd);
        let rs = resolve_group_channel(&b, None, &no_color, "stack", &mut ws);
        assert!(rd.is_none() && rs.is_none(), "case1 both None");
        assert!(wd.is_empty() && ws.is_empty(), "case1 both silent");

        // Case 2: named absent → None + one warning each, label differs.
        let with_color = enc_xy("x", "y", Some("g"));
        let mut wd = Vec::new();
        let mut ws = Vec::new();
        let rd = resolve_group_channel(&b, Some("nope"), &with_color, "dodge", &mut wd);
        let rs = resolve_group_channel(&b, Some("nope"), &with_color, "stack", &mut ws);
        assert!(rd.is_none() && rs.is_none(), "case2 both None");
        assert_eq!(wd.len(), 1);
        assert_eq!(ws.len(), 1);
        match (&wd[0], &ws[0]) {
            (
                RenderWarning::PositionAdjustSkipped { adjustment: ad, reason: rdn },
                RenderWarning::PositionAdjustSkipped { adjustment: as_, reason: rsn },
            ) => {
                assert_eq!(ad, "dodge");
                assert_eq!(as_, "stack");
                assert_eq!(rdn, rsn, "reason text identical apart from label");
            }
            other => panic!("wrong warning variants: {other:?}"),
        }

        // Case 3: categorizable Int64 → Some(same Vec) for both labels, silent.
        let bi = stack_batch_with_group(
            Field::new("g", DataType::Int64, false),
            Arc::new(Int64Array::from(vec![0_i64, 1, 0, 1])),
        );
        let mut wd = Vec::new();
        let mut ws = Vec::new();
        let rd = resolve_group_channel(&bi, Some("g"), &with_color, "dodge", &mut wd);
        let rs = resolve_group_channel(&bi, Some("g"), &with_color, "stack", &mut ws);
        assert_eq!(rd, rs, "case3 categorized vectors identical");
        assert_eq!(rd, Some(vec!["0".into(), "1".into(), "0".into(), "1".into()]));
        assert!(wd.is_empty() && ws.is_empty(), "case3 both silent");

        // Boolean categorizes the same way for both.
        let bb = stack_batch_with_group(
            Field::new("g", DataType::Boolean, false),
            Arc::new(BooleanArray::from(vec![false, true, false, true])),
        );
        let rd = resolve_group_channel(&bb, Some("g"), &with_color, "dodge", &mut Vec::new());
        let rs = resolve_group_channel(&bb, Some("g"), &with_color, "stack", &mut Vec::new());
        assert_eq!(rd, rs);

        // Case 4: un-categorizable timestamp → None + one warning each.
        let bt = stack_batch_with_group(
            Field::new(
                "g",
                DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None),
                false,
            ),
            Arc::new(TimestampMillisecondArray::from(vec![0_i64, 1, 0, 1])),
        );
        let mut wd = Vec::new();
        let mut ws = Vec::new();
        let rd = resolve_group_channel(&bt, Some("g"), &with_color, "dodge", &mut wd);
        let rs = resolve_group_channel(&bt, Some("g"), &with_color, "stack", &mut ws);
        assert!(rd.is_none() && rs.is_none(), "case4 both None");
        assert_eq!(wd.len(), 1);
        assert_eq!(ws.len(), 1);
    }

    // ── n_dodge_groups (Task 3c, remediated to the metadata contract) ────────

    /// Build a 1-column batch, optionally with `__pos_x_offset__` /
    /// `__pos_y_offset__` offset columns (jitter-shaped, no metadata) and/or the
    /// `__dodge_n_groups__` schema-metadata key (the real Dodge producer marker).
    fn batch_with_offsets_and_metadata(
        x: Option<Vec<f64>>,
        y: Option<Vec<f64>>,
        n_groups_metadata: Option<usize>,
    ) -> RecordBatch {
        let n = x.as_ref().map(|v| v.len()).or_else(|| y.as_ref().map(|v| v.len())).unwrap_or(1);
        let mut fields = vec![Field::new("v", DataType::Float64, false)];
        let mut cols: Vec<ArrayRef> = vec![Arc::new(Float64Array::from(vec![0.0; n]))];
        if let Some(xo) = x {
            fields.push(Field::new("__pos_x_offset__", DataType::Float64, false));
            cols.push(Arc::new(Float64Array::from(xo)));
        }
        if let Some(yo) = y {
            fields.push(Field::new("__pos_y_offset__", DataType::Float64, false));
            cols.push(Arc::new(Float64Array::from(yo)));
        }
        let mut schema = Schema::new(fields);
        if let Some(n_groups) = n_groups_metadata {
            let mut metadata = HashMap::new();
            BatchPositionMeta::stamp_dodge(&mut metadata, n_groups, None);
            schema = schema.with_metadata(metadata);
        }
        RecordBatch::try_new(Arc::new(schema), cols).unwrap()
    }

    #[test]
    fn n_dodge_groups_no_columns_no_metadata_is_one() {
        // No offset columns, no metadata → no Dodge → 1 (callers stay byte-identical).
        let b = batch_with_offsets_and_metadata(None, None, None);
        assert_eq!(BatchPositionMeta::from_batch(&b).dodge_n_groups(), 1);
    }

    #[test]
    fn n_dodge_groups_reads_explicit_metadata() {
        // The real Dodge producer contract: `apply_dodge_ordinal` stamps
        // `__dodge_n_groups__` alongside the offset columns; `n_dodge_groups`
        // reads that key directly, not the offset values.
        let b = batch_with_offsets_and_metadata(
            Some(vec![-12.5, 12.5, -12.5, 12.5]),
            Some(vec![0.0; 4]),
            Some(2),
        );
        assert_eq!(BatchPositionMeta::from_batch(&b).dodge_n_groups(), 2);
    }

    #[test]
    fn n_dodge_groups_jitter_shaped_offsets_without_metadata_is_one() {
        // Regression for the quality-review bug: `apply_jitter`'s ordinal branch
        // writes per-row noise into the SAME __pos_x_offset__/__pos_y_offset__
        // columns Dodge uses, but never stamps __dodge_n_groups__. Four DISTINCT
        // jitter-noise values (one per row — the old to_bits-distinct heuristic
        // would have read this as 4 dodge groups) must resolve to 1 (no
        // narrowing) because the metadata key is absent.
        let b = batch_with_offsets_and_metadata(
            Some(vec![3.1, -7.4, 0.9, -2.2]),
            Some(vec![0.0; 4]),
            None,
        );
        assert_eq!(BatchPositionMeta::from_batch(&b).dodge_n_groups(), 1);
    }

    #[test]
    fn n_dodge_groups_end_to_end_via_apply_dodge_ordinal() {
        // Integration: drive the real apply_dodge_ordinal producer and confirm
        // n_dodge_groups reads back the group count it stamped.
        let b = batch_cat_val_grp();
        let enc = enc_xy("cat", "val", Some("grp"));
        let s = scales_one_ordinal(true);
        let pos = PositionAdjust::Dodge { by: Some("grp".into()), padding: 0.0 };
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut Vec::new()).unwrap();
        assert_eq!(BatchPositionMeta::from_batch(&out).dodge_n_groups(), 2, "batch_cat_val_grp has 2 groups (p, q)");
    }

    /// Shared GH #66 dodge-pipeline driver: builds a `ChartSpec`/`PanelLayout`/
    /// theme/`DrawCtx` around `batch`, applies a `Dodge { by, padding }`
    /// position adjustment through the real `apply_position ->
    /// dispatch_mark_build` seam (the same two calls `scene_build.rs`'s
    /// `build_panel_mark_batches` makes), and returns the adjusted batch
    /// alongside every emitted bar's `(x, width)` in row order. Extracted
    /// after the third near-identical inline copy of this ~50-line pipeline
    /// accumulated across the GH #66 dodge tests below — they now differ only
    /// in their input batch/scales/panel size and the Dodge `padding`.
    fn dodge_bar_rects_via_seam(
        batch: &RecordBatch,
        enc: Encoding,
        scales: &ResolvedScales,
        by: &str,
        padding: f64,
        panel_area: Rect,
    ) -> (RecordBatch, Vec<(f64, f64)>) {
        use crate::layout::{PanelLayout, ThemeInputs};
        use crate::render::draw::{dispatch_mark_build, resolve_mark_style, DrawCtx};
        use crate::spec::chart::ChartSpec;
        use crate::spec::mark::Mark;
        use ferrum_scene::SceneNode;

        let pos = PositionAdjust::Dodge { by: Some(by.into()), padding };
        // Real producer: same call scene_build.rs makes before mark dispatch.
        let adjusted = apply_position(batch, Some(&pos), scales, &enc, false, &mut Vec::new()).unwrap();

        let spec = ChartSpec {
            data: Default::default(),
            mark: Mark::Bar,
            encoding: enc,
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
            position: Some(pos),
            title: None,
            axis_x: None,
            axis_y: None,
            selections: Vec::new(),
            conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        };
        let panel = PanelLayout {
            plot_area: panel_area,
            facet_key: None,
            row: 0,
            col: 0,
            strip_title: None,
            row_strip_title: None,
            row_facet_key: None,
        };
        let theme = ThemeInputs::default();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Bar);

        // Real seam: DrawCtx built from the ADJUSTED batch (mirrors
        // scene_build.rs:795-824), dispatched through the real mark-build entry
        // point — not bar::build called directly on a hand-built batch.
        let ctx = DrawCtx {
            spec: &spec,
            panel: &panel,
            theme: &theme,
            scales,
            batch: &adjusted,
            mark_style: &mark_style,
        };
        let result = dispatch_mark_build(&spec.mark, &ctx);
        let rects: Vec<(f64, f64)> = result
            .nodes
            .iter()
            .filter_map(|n| if let SceneNode::Rect { x, w, .. } = n { Some((*x, *w)) } else { None })
            .collect();
        (adjusted, rects)
    }

    /// Batch with a 3-category band column `cat` (a/b/c), numeric `val`, and a
    /// two-level grouping column `grp` (p/q interleaved) — the GH #66 dodge
    /// sub-band fixture shared by the tests below.
    fn batch_cat3_val_grp() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, false),
            Field::new("val", DataType::Float64, false),
            Field::new("grp", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "a", "b", "b", "c", "c"])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0])),
                Arc::new(StringArray::from(vec!["p", "q", "p", "q", "p", "q"])),
            ],
        )
        .unwrap()
    }

    /// Ordinal x (3 categories over `[0, 600]`) + linear y `[0, 100]` — the
    /// GH #66 dodge sub-band fixture's scales, paired with `batch_cat3_val_grp`.
    fn scales_cat3_ordinal() -> ResolvedScales {
        use crate::scale::linear::LinearScale;
        use crate::scale::ordinal::OrdinalScale;
        let ord = ScaleKind::Ordinal(OrdinalScale::new_internal(
            vec!["a".into(), "b".into(), "c".into()],
            vec![0.0, 600.0],
            0.0,
        ));
        let lin = ScaleKind::Linear(LinearScale::new_internal(
            vec![0.0, 100.0], vec![0.0, 100.0], false, false,
        ));
        ResolvedScales { x: ord, y: lin, color: None, size: None, shape: None, opacity: None, x2: None, y2: None, y_slots: Default::default() }
    }

    /// Design-gate finding: the two tests above only exercise
    /// `apply_dodge_ordinal` → `n_dodge_groups` in isolation. Neither drives the
    /// REAL production seam — `apply_position` → `dispatch_mark_build` — through
    /// which the `__dodge_n_groups__` schema metadata must survive unmodified
    /// for dodged bars to render narrowed. This test walks that full seam,
    /// mirroring `scene_build.rs`'s `build_panel_mark_batches` wiring exactly
    /// (`apply_position` output batch → `DrawCtx { batch: adjusted, .. }` →
    /// `dispatch_mark_build`, see scene_build.rs:795-824), so a future pipeline
    /// change that rebuilds the batch/schema between those two calls and drops
    /// the metadata fails this test loudly (bars would revert to full
    /// sub-band-overlapping width) instead of silently regressing.
    #[test]
    fn dodge_narrows_bar_width_through_real_apply_position_to_mark_build_seam() {
        let b = batch_cat_val_grp();
        let enc = enc_xy("cat", "val", Some("grp"));
        let s = scales_one_ordinal(true);

        // panel range matches scales_one_ordinal's [0, 100] pixel range.
        let (adjusted, rects) = dodge_bar_rects_via_seam(
            &b, enc, &s, "grp", 0.0, Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
        );
        assert_eq!(
            BatchPositionMeta::from_batch(&adjusted).dodge_n_groups(), 2,
            "sanity: real apply_position output must carry the 2-group metadata"
        );

        assert_eq!(rects.len(), 4, "expected 4 dodged bars (2 categories x 2 groups)");
        // panel.w=100, 2 categories, 2 dodge groups -> bar_width = (100/2/2)*0.8 = 20.0.
        // Full (non-narrowed) width would be (100/2/1)*0.8 = 40.0 — the value this
        // test would see if the metadata were silently dropped in transit.
        for (_, w) in &rects {
            assert!(
                (w - 20.0).abs() < 1e-9,
                "dodged bar width through the real apply_position -> dispatch_mark_build \
                 seam must be 20.0 (narrowed by 2 groups); got {w}. A value of 40.0 means \
                 __dodge_n_groups__ was dropped between the two calls."
            );
        }
    }

    /// GH #66: E=600, n=3 categories, g=2 dodge groups, Dodge padding=0.2
    /// through the same real `apply_position` -> `dispatch_mark_build` seam
    /// as the test above. Pre-fix (0.8-factor width, no sub-band clamp),
    /// adjacent sub-bars within category "a" overlapped by 20px — this test
    /// pins the real-pipeline fix (`BatchPositionMeta::clamp_width`), covering
    /// the edge cases ported from the deleted bug_hunt_dodge_subband_overlap.rs.
    #[test]
    fn dodge_sub_band_clamp_prevents_overlap_at_high_padding() {
        let b = batch_cat3_val_grp();
        let enc = enc_xy("cat", "val", Some("grp"));
        let s = scales_cat3_ordinal();
        let (_, rects) = dodge_bar_rects_via_seam(
            &b, enc, &s, "grp", 0.2, Rect { x: 0.0, y: 0.0, w: 600.0, h: 100.0 },
        );
        assert_eq!(rects.len(), 6, "expected 6 dodged bars (3 categories x 2 groups)");
        // Category "a" is the first two rects (row order preserved by MarkNodes).
        // bandwidth_px = 600/3 = 200; sub_band = 200*(1 - 2*0.2)/2 = 60; the
        // 0.8-factor raw width (200/2*0.8 = 80) is clamped down to 60, so the
        // two sub-bars sit exactly edge-to-edge: [40,100) and [100,160).
        let (x0, w0) = rects[0];
        let (x1, w1) = rects[1];
        assert!((w0 - 60.0).abs() < 1e-9, "clamped sub-bar width must be 60.0; got {w0}");
        assert!((w1 - 60.0).abs() < 1e-9, "clamped sub-bar width must be 60.0; got {w1}");
        assert!((x0 - 40.0).abs() < 1e-9, "group0 x must be 40.0; got {x0}");
        assert!((x1 - 100.0).abs() < 1e-9, "group1 x must be 100.0; got {x1}");
        assert!(
            x0 + w0 <= x1 + 1e-9,
            "adjacent dodge sub-bars in category 'a' must not overlap: group0 spans \
             [{x0}, {}], group1 starts at {x1} — overlap of {}",
            x0 + w0, (x0 + w0) - x1
        );
        // GH #66 (ported from tests/bug_hunt_dodge_subband_overlap.rs, R1): the
        // clamp is per-category, not a special-case for "a" — every category's
        // adjacent sub-bars must clamp to the same 60px sub-band and sit
        // edge-to-edge, since bandwidth/sub_band depend only on band index.
        for (cat_idx, cat) in ["a", "b", "c"].iter().enumerate() {
            let (xg0, wg0) = rects[cat_idx * 2];
            let (xg1, wg1) = rects[cat_idx * 2 + 1];
            assert!((wg0 - 60.0).abs() < 1e-9, "category {cat}: clamped width must be 60.0; got {wg0}");
            assert!((wg1 - 60.0).abs() < 1e-9, "category {cat}: clamped width must be 60.0; got {wg1}");
            assert!(
                xg0 + wg0 <= xg1 + 1e-9,
                "category {cat}: adjacent dodge sub-bars must not overlap: group0 spans \
                 [{xg0}, {}], group1 starts at {xg1}",
                xg0 + wg0
            );
        }
    }

    /// GH #66's refuted premise: the issue as filed blamed `BandScale(padding_inner=)`
    /// for dodge overlap. `OrdinalScale::bandwidth()` and the render-side band
    /// centers never read `padding` — it only feeds `invert_band` hit-testing,
    /// never the dodge geometry seam. Pins that inertness directly against the
    /// real `OrdinalScale`: bandwidth and every category's `scale_internal`
    /// center are byte-identical whether `padding` is 0.0 or 0.9 (ported from
    /// tests/bug_hunt_dodge_subband_overlap.rs `dodge_padding_inner_is_geometrically_inert`, R1).
    #[test]
    fn dodge_padding_inner_is_geometrically_inert() {
        use crate::scale::ordinal::OrdinalScale;
        let domain = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let low_padding = OrdinalScale::new_internal(domain.clone(), vec![0.0, 600.0], 0.0);
        let high_padding = OrdinalScale::new_internal(domain.clone(), vec![0.0, 600.0], 0.9);

        assert_eq!(low_padding.bandwidth(), 200.0);
        assert_eq!(low_padding.bandwidth(), high_padding.bandwidth(), "padding must not perturb bandwidth");

        for cat in &domain {
            assert_eq!(
                low_padding.scale_internal(cat),
                high_padding.scale_internal(cat),
                "padding must not perturb the band center for {cat}"
            );
        }
        assert_eq!(
            domain.iter().map(|c| low_padding.scale_internal(c)).collect::<Vec<_>>(),
            vec![Some(100.0), Some(300.0), Some(500.0)],
        );
    }

    /// Default Dodge `padding=0.05` (E=600, n=3 categories, g=2 groups): the
    /// 0.8-factor raw bar width (80.0) already fits inside the sub-band
    /// (90.0), so the GH #66 clamp (`BatchPositionMeta::clamp_width`) is a
    /// documented no-op and every emitted rect matches the pre-fix geometry
    /// byte-for-byte — no golden churn at default padding (ported from
    /// tests/bug_hunt_dodge_subband_overlap.rs `dodge_default_padding_output_unchanged`, R1).
    #[test]
    fn dodge_default_padding_bar_width_clamp_is_noop() {
        let b = batch_cat3_val_grp();
        let enc = enc_xy("cat", "val", Some("grp"));
        let s = scales_cat3_ordinal();
        let (_, rects) = dodge_bar_rects_via_seam(
            &b, enc, &s, "grp", 0.05, Rect { x: 0.0, y: 0.0, w: 600.0, h: 100.0 },
        );
        // Rows in (category, group) order a/p, a/q, b/p, b/q, c/p, c/q —
        // mirroring `MarkNodes`' row-order-preserving accumulator.
        let expected = [
            (15.0, 80.0),
            (105.0, 80.0),
            (215.0, 80.0),
            (305.0, 80.0),
            (415.0, 80.0),
            (505.0, 80.0),
        ];
        assert_eq!(rects.len(), expected.len());
        for (i, (rect, exp)) in rects.iter().zip(expected.iter()).enumerate() {
            assert!(
                (rect.0 - exp.0).abs() < 1e-9 && (rect.1 - exp.1).abs() < 1e-9,
                "row {i}: expected {exp:?}, got {rect:?}"
            );
        }
    }

    // ------------------------------------------------------------------
    // GH #77: Stack.value_axis (composite-mark horizontal desugars swap
    // x/y without setting CoordFlip, so coord_flipped stays false while
    // the value channel lives on encoding.x).
    // ------------------------------------------------------------------

    #[test]
    fn stack_value_axis_x_cumulates_along_x_without_coord_flip() {
        // Mirrors desugar_histogram/desugar_density's orientation="horizontal"
        // shape: x = count/density (value), y = bin_start/value (category),
        // with coord_flipped=false (no real CoordFlip involved). Pre-#77 code
        // picked the value/category axis purely from coord_flipped, so it
        // would have treated "count" (Float64, on x) as the CATEGORY column
        // and "bin_start" (also Float64, on y) as the VALUE column to
        // cumulate — silently wrong. With `value_axis: Some(X)` set (as the
        // Python-side desugar fix will do), Stack must cumulate encoding.x
        // ("count") grouped by encoding.y ("bin_start") instead.
        let schema = Arc::new(Schema::new(vec![
            Field::new("count", DataType::Float64, false),
            Field::new("bin_start", DataType::Float64, false),
            Field::new("grp", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![3.0, 5.0, 2.0, 4.0])),
                Arc::new(Float64Array::from(vec![0.0, 0.0, 1.0, 1.0])),
                Arc::new(StringArray::from(vec!["x", "y", "x", "y"])),
            ],
        )
        .unwrap();

        let enc = enc_xy("count", "bin_start", Some("grp"));
        let s = dummy_scales();
        let pos = PositionAdjust::Stack {
            by: Some("grp".into()),
            offset: StackOffset::Zero,
            anchor: StackAnchor::Top,
            value_axis: Some(StackValueAxis::X),
        };

        let out = apply_position(&batch, Some(&pos), &s, &enc, false, &mut Vec::new()).unwrap();

        // "count" (encoding.x, the value column) must be cumulated.
        let count_idx = out.schema().index_of("count").unwrap();
        let ca = out.column(count_idx).as_any().downcast_ref::<Float64Array>().unwrap();
        // Group order: x=0, y=1. At bin_start=0: x=3→3, y=5→8. At bin_start=1: x=2→2, y=4→6.
        assert_eq!(ca.value(0), 3.0);
        assert_eq!(ca.value(1), 8.0);
        assert_eq!(ca.value(2), 2.0);
        assert_eq!(ca.value(3), 6.0);

        // "bin_start" (encoding.y, the category column) must be untouched.
        let bin_idx = out.schema().index_of("bin_start").unwrap();
        let bin_a = out.column(bin_idx).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(bin_a.value(0), 0.0);
        assert_eq!(bin_a.value(1), 0.0);
        assert_eq!(bin_a.value(2), 1.0);
        assert_eq!(bin_a.value(3), 1.0);

        // __stack_y_base__ carries the cumulative bases for the value column.
        let base_idx = out.schema().index_of("__stack_y_base__").unwrap();
        let ba = out.column(base_idx).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(ba.value(0), 0.0);
        assert_eq!(ba.value(1), 3.0);
        assert_eq!(ba.value(2), 0.0);
        assert_eq!(ba.value(3), 2.0);
    }

    #[test]
    fn stack_value_axis_none_coord_flipped_false_byte_identical() {
        // value_axis: None + coord_flipped=false must reproduce today's
        // vertical-stack behavior exactly (byte-identical to
        // stack_zero_accumulates_y, which predates GH #77).
        let b = batch_xyg();
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Stack {
            by: Some("g".into()),
            offset: StackOffset::Zero,
            anchor: StackAnchor::Top,
            value_axis: None,
        };
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut Vec::new()).unwrap();
        let ya = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(ya.value(0), 10.0);
        assert_eq!(ya.value(1), 30.0);
        assert_eq!(ya.value(2), 30.0);
        assert_eq!(ya.value(3), 70.0);

        let base_idx = out.schema().index_of("__stack_y_base__").unwrap();
        let ba = out.column(base_idx).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(ba.value(0), 0.0);
        assert_eq!(ba.value(1), 10.0);
        assert_eq!(ba.value(2), 0.0);
        assert_eq!(ba.value(3), 30.0);
    }

    #[test]
    fn stack_value_axis_none_coord_flipped_true_preserved() {
        // value_axis: None + coord_flipped=true must reproduce today's real
        // CoordFlip stack behavior exactly (byte-identical to
        // stack_coord_flipped_uses_x_as_value_column, which predates GH #77).
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, false),
            Field::new("val", DataType::Float64, false),
            Field::new("grp", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "a", "b", "b"])),
                Arc::new(Float64Array::from(vec![3.0, 5.0, 2.0, 4.0])),
                Arc::new(StringArray::from(vec!["x", "y", "x", "y"])),
            ],
        )
        .unwrap();

        let enc = enc_xy("val", "cat", Some("grp"));
        let s = dummy_scales();
        let pos = PositionAdjust::Stack {
            by: Some("grp".into()),
            offset: StackOffset::Zero,
            anchor: StackAnchor::Top,
            value_axis: None,
        };

        let out = apply_position(&batch, Some(&pos), &s, &enc, true, &mut Vec::new()).unwrap();

        let val_idx = out.schema().index_of("val").unwrap();
        let va = out.column(val_idx).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(va.value(0), 3.0);
        assert_eq!(va.value(1), 8.0);
        assert_eq!(va.value(2), 2.0);
        assert_eq!(va.value(3), 6.0);

        let base_idx = out.schema().index_of("__stack_y_base__").unwrap();
        let ba = out.column(base_idx).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(ba.value(0), 0.0);
        assert_eq!(ba.value(1), 3.0);
        assert_eq!(ba.value(2), 0.0);
        assert_eq!(ba.value(3), 2.0);
    }

    #[test]
    fn stack_value_axis_serde_round_trip_present_and_absent() {
        // Wire-contract pin: value_axis Some(X) serializes with the key
        // present ("value_axis":"x"); None must omit the key entirely so
        // pre-#77 spec JSON (and the scale_wire_baseline.json fixture) stays
        // byte-identical. Mirrors spec::position's own serde tests.
        let with_axis = PositionAdjust::Stack {
            by: Some("g".into()),
            offset: StackOffset::Zero,
            anchor: StackAnchor::Top,
            value_axis: Some(StackValueAxis::X),
        };
        let json = serde_json::to_string(&with_axis).unwrap();
        assert!(json.contains(r#""value_axis":"x""#), "got {json}");
        let parsed: PositionAdjust = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, with_axis);

        let without_axis = PositionAdjust::Stack {
            by: Some("g".into()),
            offset: StackOffset::Zero,
            anchor: StackAnchor::Top,
            value_axis: None,
        };
        let json = serde_json::to_string(&without_axis).unwrap();
        assert!(!json.contains("value_axis"), "got {json}");
        let parsed: PositionAdjust = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, without_axis);
    }
}

/// Read per-row pixel offsets from synthetic `__pos_x_offset__` /
/// `__pos_y_offset__` columns. Returns `(Vec<f64>, Vec<f64>)` of zeros-by-
/// default when the columns are absent. Mark drawers call this near the top
/// of `draw()` and add the per-row offset to their resolved pixel position
/// before emitting SVG.
pub(crate) fn read_position_offsets(batch: &RecordBatch) -> (Vec<f64>, Vec<f64>) {
    let n = batch.num_rows();
    let xo = batch
        .schema()
        .index_of("__pos_x_offset__")
        .ok()
        .and_then(|i| {
            batch.column(i).as_any().downcast_ref::<Float64Array>().map(|a| {
                (0..a.len()).map(|j| a.value(j)).collect::<Vec<f64>>()
            })
        })
        .unwrap_or_else(|| vec![0.0; n]);
    let yo = batch
        .schema()
        .index_of("__pos_y_offset__")
        .ok()
        .and_then(|i| {
            batch.column(i).as_any().downcast_ref::<Float64Array>().map(|a| {
                (0..a.len()).map(|j| a.value(j)).collect::<Vec<f64>>()
            })
        })
        .unwrap_or_else(|| vec![0.0; n]);
    (xo, yo)
}

/// Schema metadata key [`apply_dodge_ordinal`] stamps with the dodge group
/// count. Read back via [`BatchPositionMeta::dodge_n_groups`].
const DODGE_N_GROUPS_KEY: &str = "__dodge_n_groups__";

/// Schema metadata key [`apply_dodge_ordinal`] stamps with the computed
/// pixel sub-band width (GH #66). Read back via
/// [`BatchPositionMeta::clamp_width`].
const DODGE_SUB_BAND_PX_KEY: &str = "__dodge_sub_band_px__";

/// Schema metadata key [`apply_stack`] stamps when the resolved stack
/// *value* axis (GH #77's `value_on_x`) is X rather than Y — i.e. whenever
/// `value_axis: Some(X)` (a horizontal composite-mark desugar) or a real
/// `CoordFlip` (`coord_flipped` fallback) put the cumulated value column on
/// `encoding.x`. `__stack_y_base__` is the same column name regardless of
/// which axis holds the value, so mark drawers that consume it
/// (`marks::bar`, `marks::area`) need this explicit signal to know whether
/// to map the base through `scales.x` or `scales.y` — they can't infer it
/// from batch shape alone. Read back via
/// [`BatchPositionMeta::stack_value_on_x`].
const STACK_VALUE_ON_X_KEY: &str = "__stack_value_on_x__";

/// Typed view over the Dodge/Stack schema-metadata side-channel that
/// [`apply_dodge_ordinal`] and [`apply_stack`] stamp onto a batch and that
/// mark drawers (`marks::bar`, `marks::rect`, `marks::tick`, `marks::area`)
/// read back. Each mark builder parses this **once** per call
/// ([`from_batch`](Self::from_batch)) instead of re-reading raw schema
/// metadata on every per-call width/base computation.
///
/// The three fields mirror the three metadata keys this module used to
/// expose as separate consts + free-function readers:
///
/// - `dodge_n_groups` — the number of ordinal-band Dodge groups active on
///   the batch. Band-dimension marks (bar bodies, box bodies, band-fraction
///   boxplot ticks) shrink their per-category extent to
///   `extent / dodge_n_groups`, otherwise dodged sub-bands overlap their
///   neighbours. The count comes from explicit metadata rather than
///   inferring it off `__pos_x_offset__` / `__pos_y_offset__` offset values,
///   because Jitter's ordinal branch writes per-row noise into those SAME
///   two columns; a distinct-value heuristic would misread jitter noise as
///   ≈row-count dodge groups. Defaults to `1` (no narrowing) when the key is
///   absent or unparseable, so every non-dodged batch divides by `1` and
///   stays byte-identical (exact in IEEE-754).
/// - `dodge_sub_band_px` — the true per-group pixel slot width Dodge
///   computed (GH #66), used to clamp a band-axis mark's width formula
///   (which is blind to Dodge's `padding`) so adjacent dodge groups can't
///   overlap at high padding. `None` when absent — callers must treat that
///   as "no clamp", not zero.
/// - `stack_value_on_x` — whether `apply_stack` cumulated the *value* column
///   onto X rather than Y. `false` (the pre-#77 default) when the key is
///   absent.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct BatchPositionMeta {
    dodge_n_groups: usize,
    dodge_sub_band_px: Option<f64>,
    stack_value_on_x: bool,
}

impl BatchPositionMeta {
    /// Parse the Dodge/Stack schema-metadata side-channel off `batch`.
    /// Absent-or-unparseable keys fall back to the same defaults the old
    /// per-key free-function readers used (`1` / `None` / `false`), so this
    /// is byte-identical to reading each key independently.
    pub(crate) fn from_batch(batch: &RecordBatch) -> Self {
        let schema = batch.schema();
        let metadata = schema.metadata();
        let dodge_n_groups = metadata
            .get(DODGE_N_GROUPS_KEY)
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|n| *n >= 1)
            .unwrap_or(1);
        let dodge_sub_band_px = metadata.get(DODGE_SUB_BAND_PX_KEY).and_then(|s| s.parse::<f64>().ok());
        let stack_value_on_x = metadata.contains_key(STACK_VALUE_ON_X_KEY);
        Self { dodge_n_groups, dodge_sub_band_px, stack_value_on_x }
    }

    /// Number of ordinal-band Dodge groups active on the batch (see the
    /// struct-level doc). `1` means "no Dodge → no narrowing".
    pub(crate) fn dodge_n_groups(&self) -> usize {
        self.dodge_n_groups
    }

    /// Clamp a dodged band-axis mark's pixel width to its Dodge sub-band.
    ///
    /// `raw_width` is the mark's ordinary width formula
    /// (`band_extent / n_categories / dodge_n_groups * factor`, for whatever
    /// `factor` the mark uses — bar's fixed `0.8`, box/tick's `band_size`).
    /// That formula narrows by `dodge_n_groups` but is blind to Dodge's
    /// `padding`, so at high padding the factor-scaled width can exceed the
    /// true per-group slot width and adjacent dodge groups overlap (GH #66).
    ///
    /// Returns `raw_width` unchanged whenever no Dodge sub-band was stamped
    /// (no Dodge on this batch) or the sub-band isn't tighter than the raw
    /// width — so every non-dodged chart, and every dodged chart at the
    /// default padding (where the factor-scaled width already fits inside
    /// the sub-band), stays byte-for-byte identical to the pre-clamp
    /// formula.
    pub(crate) fn clamp_width(&self, raw_width: f64) -> f64 {
        match self.dodge_sub_band_px {
            Some(sub_band) if sub_band > 0.0 && sub_band < raw_width => sub_band,
            _ => raw_width,
        }
    }

    /// Whether [`apply_stack`] stamped the value-on-X marker.
    ///
    /// `false` when absent — the pre-#77 default (value on Y, or no Stack at
    /// all) — so every existing vertical-stack mark drawer path stays
    /// byte-identical.
    pub(crate) fn stack_value_on_x(&self) -> bool {
        self.stack_value_on_x
    }

    /// Stamp the Dodge group count (and, when computed, the pixel sub-band
    /// width) into schema `metadata`. The single writer [`apply_dodge_ordinal`]
    /// uses; test call sites that only need `dodge_n_groups()` to read back a
    /// non-default value pass `sub_band_px: None` to stamp that key alone,
    /// matching the pre-refactor behavior of inserting
    /// `DODGE_N_GROUPS_KEY` without `DODGE_SUB_BAND_PX_KEY`.
    pub(crate) fn stamp_dodge(metadata: &mut HashMap<String, String>, n_groups: usize, sub_band_px: Option<f64>) {
        metadata.insert(DODGE_N_GROUPS_KEY.to_string(), n_groups.to_string());
        if let Some(sub_band) = sub_band_px {
            metadata.insert(DODGE_SUB_BAND_PX_KEY.to_string(), sub_band.to_string());
        }
    }

    /// Stamp the stack value-on-X marker into schema `metadata`. The single
    /// writer [`apply_stack`] uses, in its `value_on_x` branch only — mirrors
    /// the pre-refactor byte-stable contract that the non-`value_on_x`
    /// branch stamps nothing.
    pub(crate) fn stamp_stack_value_on_x(metadata: &mut HashMap<String, String>) {
        metadata.insert(STACK_VALUE_ON_X_KEY.to_string(), "1".to_string());
    }
}
