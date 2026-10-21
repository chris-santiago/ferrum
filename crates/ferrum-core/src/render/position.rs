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

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};

use crate::render::scale_resolve::{ResolvedScales, ScaleKind};
use crate::render::RenderWarning;
use crate::spec::chart::ChartSpec;
use crate::spec::position::{PositionAdjust, StackAnchor, StackValueAxis};

/// The batches a value axis resolves against, and the span partner that
/// widens it to every stacked segment's drawn extent.
///
/// A stacked segment draws from its base to its top. `batch` carries the tops
/// (the stacked value column); `base` names the base column, so the scale
/// reads it as the value field's paired span, the same read a bound `x2`/`y2`
/// takes. In the preview batch the base column holds only the nonzero bases:
/// a 0 base is a stack's origin, not a data extent, and it follows the
/// unstacked baseline rule (0 enters the domain only through zero anchoring),
/// so zero and normalize stacks resolve exactly as before while a center
/// stack, whose runs start at a nonzero shift, covers its lowest base. The
/// draw pass runs its own `apply_stack` and reads the full base column.
///
/// `outputs` are the named outputs the axis reads beside `batch`: on a
/// faceted shared axis the all-panels batch at [`FINAL_OUTPUT_KEY`] is
/// replaced by its own stack preview, so every panel's axis spans the
/// tallest panel's stack, not the unstacked rows.
///
/// [`FINAL_OUTPUT_KEY`]: crate::transform::core::FINAL_OUTPUT_KEY
pub(crate) struct StackedAxisBatch<'a> {
    pub(crate) batch: Cow<'a, RecordBatch>,
    pub(crate) base: Option<crate::spec::encoding::EncodingSpec>,
    pub(crate) outputs: Cow<'a, HashMap<String, RecordBatch>>,
}

/// The column `apply_stack` writes each segment's base into, whichever axis
/// carries the stacked value.
pub(crate) const STACK_BASE_COLUMN: &str = "__stack_y_base__";

/// The batches an unlayered spec's value axis on `axis` resolves against,
/// accounting for the stack its one mark draws ([`StackedAxisBatch`]).
///
/// `encoding` is the encoding the mark draws (post-`CoordFlip`), and `field`
/// the axis's bound field: the preview applies only when that field is the
/// encoding's value channel on `axis` and the mark stacks its value there
/// ([`stack_preview`], which reads the explicit `Stack` and the encoding's
/// `stack` shorthand as the draw pass does). Returning the stacked tops keeps
/// stacked marks visible: `LinearScale` returns NaN for out-of-domain inputs
/// and the drawers drop every row whose top falls past the domain.
///
/// A layered spec's axis resolves against the unstacked batches: each layer
/// that stacks contributes the stack it draws, from the batch it draws, in
/// the layered domain union (`scale_resolve::domain`), so a layer reading a
/// statistic's output stacks there exactly as one reading the chart's rows,
/// and an unstacked sibling keeps reading its own values. `include_final`
/// is the facet-shared flag the axis resolves under: when set, the
/// all-panels batch the union reads is stacked too, per panel. Refuses a
/// facet field the panel split cannot read ([`stack_preview`]).
pub(crate) fn axis_batch<'a>(
    spec: &ChartSpec,
    encoding: &crate::spec::encoding::Encoding,
    axis: StackValueAxis,
    field: &str,
    primary_batch: &'a RecordBatch,
    transform_outputs: &'a HashMap<String, RecordBatch>,
    include_final: bool,
) -> Result<StackedAxisBatch<'a>, crate::render::RenderError> {
    let unstacked = StackedAxisBatch {
        batch: Cow::Borrowed(primary_batch),
        base: None,
        outputs: Cow::Borrowed(transform_outputs),
    };
    if spec.is_layered() {
        return Ok(unstacked);
    }
    let value = match axis {
        StackValueAxis::X => encoding.x.as_ref(),
        StackValueAxis::Y => encoding.y.as_ref(),
    };
    if value.map(|e| e.field.as_str()) != Some(field) {
        return Ok(unstacked);
    }
    let coord_flipped = crate::spec::coord::flips_positional_channels(spec.coord.as_ref());
    let drawn = position_encoding(spec.mark, spec.coord.as_ref(), encoding);
    let preview = |batch: &RecordBatch| {
        stack_preview(batch, spec.position.as_ref(), &drawn, coord_flipped, axis, spec.facet.as_ref())
    };
    let Some(stacked) = preview(primary_batch)? else { return Ok(unstacked) };
    let all_panels = match include_final.then(|| transform_outputs.get(crate::transform::core::FINAL_OUTPUT_KEY)).flatten() {
        Some(all_panels) => preview(all_panels)?,
        None => None,
    };
    let outputs = match all_panels {
        Some(all_panels) => {
            let mut outputs = transform_outputs.clone();
            outputs.insert(crate::transform::core::FINAL_OUTPUT_KEY.to_string(), all_panels);
            Cow::Owned(outputs)
        }
        None => Cow::Borrowed(transform_outputs),
    };
    Ok(StackedAxisBatch {
        batch: Cow::Owned(stacked),
        base: Some(crate::spec::encoding::EncodingSpec { field: STACK_BASE_COLUMN.into(), ..Default::default() }),
        outputs,
    })
}

/// The stack preview of the batch one layer of `spec` draws, on the panel's
/// `axis`: [`stack_preview`] over the position, encoding and coord the draw
/// pass hands `apply_position` for that layer
/// ([`LayerPrepared`](crate::render::prepare::LayerPrepared) in its own
/// coord). `None` when the layer stacks nothing on `axis`.
pub(crate) fn layer_stack_preview(
    spec: &ChartSpec,
    layer: &crate::spec::layer::Layer,
    axis: StackValueAxis,
    batch: &RecordBatch,
) -> Result<Option<RecordBatch>, crate::render::RenderError> {
    let drawn = crate::render::prepare::LayerPrepared::from_chart_and_layer(spec, layer).in_its_coord();
    let coord_flipped = crate::spec::coord::flips_positional_channels(drawn.coord.as_ref());
    let encoding = position_encoding(drawn.mark, drawn.coord.as_ref(), &drawn.encoding);
    stack_preview(batch, drawn.position.as_ref(), &encoding, coord_flipped, axis, spec.facet.as_ref())
}

/// The stack a mark draws over `batch` under `position` (or, with no
/// position, its encoding's `stack` shorthand: [`effective_position`]),
/// previewed for the value axis `axis`: the stacked batch with every 0 base
/// nulled ([`StackedAxisBatch`]). `None` when no stack applies, when its
/// value lies on the other axis ([`resolve_stack_value_on_x`], the triad the
/// draw pass reads), or when the stack fails; the draw pass's own stack then
/// raises that error.
///
/// The draw pass stacks each facet panel's rows apart, so rows of different
/// panels (by `facet`'s fields the batch holds) never stack together here
/// either: a batch spanning panels is stacked panel by panel. A batch
/// holding no facet field repeats in every panel and stacks whole. A facet
/// field the panel split cannot read is refused with the panel split's own
/// error ([`crate::render::facet_key_column`]), never stacked across panels.
/// A `stack` shorthand on both channels is refused as the draw pass refuses
/// it ([`stack_shorthand`]).
///
/// It stacks in first-appearance order (no color scale ranks the groups):
/// the order moves segments within a bin, never a bin's positive or negative
/// total, so the extent it previews is the drawn one. The warnings the stack
/// would raise are the draw pass's to report, once, so they are discarded.
pub(crate) fn stack_preview(
    batch: &RecordBatch,
    position: Option<&PositionAdjust>,
    encoding: &crate::spec::encoding::Encoding,
    coord_flipped: bool,
    axis: StackValueAxis,
    facet: Option<&crate::layout::facet::FacetSpec>,
) -> Result<Option<RecordBatch>, crate::render::RenderError> {
    let Some(effective) = effective_position(position, encoding, coord_flipped)? else { return Ok(None) };
    let PositionAdjust::Stack { by, offset, anchor, value_axis } = effective.as_ref() else {
        return Ok(None);
    };
    if resolve_stack_value_on_x(*value_axis, coord_flipped) != (axis == StackValueAxis::X) {
        return Ok(None);
    }
    let mut discard_warnings = Vec::new();
    let mut stack = |rows: &RecordBatch| {
        apply_stack(rows, by.as_deref(), offset, anchor, *value_axis, encoding, None, coord_flipped, &mut discard_warnings).ok()
    };
    let stacked = match facet_panels(batch, facet)? {
        None => stack(batch),
        Some(panels) => panels
            .iter()
            .map(&mut stack)
            .collect::<Option<Vec<_>>>()
            .and_then(|stacked| arrow::compute::concat_batches(&stacked.first()?.schema(), &stacked).ok()),
    };
    Ok(stacked.and_then(with_nonzero_bases))
}

/// `batch` with its stack base column's 0 entries set to null. `apply_stack`
/// always writes that column as nullable `Float64`, so the rebuild cannot
/// fail in practice; `None` if it did.
fn with_nonzero_bases(stacked: RecordBatch) -> Option<RecordBatch> {
    let idx = stacked.schema().index_of(STACK_BASE_COLUMN).ok()?;
    let bases = stacked.column(idx).as_any().downcast_ref::<Float64Array>()?;
    let nonzero: Float64Array = bases.iter().map(|b| b.filter(|v| *v != 0.0)).collect();
    let mut columns = stacked.columns().to_vec();
    columns[idx] = Arc::new(nonzero);
    RecordBatch::try_new(stacked.schema(), columns).ok()
}

/// `batch`'s rows split by facet panel, in first-appearance order of each
/// panel: one batch per distinct combination of `facet`'s fields that the
/// batch holds, each read as the panel split reads it
/// ([`crate::render::facet_key_column`]), and grouped in one pass over the
/// rows. `Ok(None)` when there is no facet, the batch holds none of its
/// fields, or every row sits in one panel, so the batch stacks whole.
fn facet_panels(
    batch: &RecordBatch,
    facet: Option<&crate::layout::facet::FacetSpec>,
) -> Result<Option<Vec<RecordBatch>>, crate::render::RenderError> {
    let Some(facet) = facet else { return Ok(None) };
    let keys = std::iter::once(facet.field.as_str())
        .chain(facet.row.as_deref())
        .filter(|field| batch.column_by_name(field).is_some())
        .map(|field| crate::render::facet_key_column(batch, field))
        .collect::<Result<Vec<_>, _>>()?;
    if keys.is_empty() {
        return Ok(None);
    }
    let mut panel_of: HashMap<Vec<Option<&str>>, usize> = HashMap::new();
    let mut panels: Vec<Vec<u32>> = Vec::new();
    for row in 0..batch.num_rows() {
        let key: Vec<Option<&str>> = keys.iter().map(|column| column.is_valid(row).then(|| column.value(row))).collect();
        let next = panels.len();
        let panel = *panel_of.entry(key).or_insert(next);
        if panel == next {
            panels.push(Vec::new());
        }
        panels[panel].push(row as u32);
    }
    if panels.len() < 2 {
        return Ok(None);
    }
    panels
        .into_iter()
        .map(|rows| {
            arrow::compute::take_record_batch(batch, &arrow::array::UInt32Array::from(rows))
                .map_err(|e| crate::render::RenderError::ScaleResolutionFailed(format!("facet split: {e}")))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

/// The channel the encoding's `stack` shorthand stacks along, in the
/// encoding's own (post-`CoordFlip`) space, and the offset it names: `x` or
/// `y`, whichever declares a stack offset ([`EncodingSpec::stack_offset`]),
/// as Vega-Lite and ggplot2 stack the value channel that declares it.
/// `Ok(None)` when neither does. A shorthand on both is refused
/// ([`RenderError::StackOnBothChannels`]): a mark stacks along one channel.
///
/// [`EncodingSpec::stack_offset`]: crate::spec::encoding::EncodingSpec::stack_offset
/// [`RenderError::StackOnBothChannels`]: crate::render::RenderError::StackOnBothChannels
pub(crate) fn stack_shorthand(
    encoding: &crate::spec::encoding::Encoding,
) -> Result<Option<(StackValueAxis, crate::spec::position::StackOffset)>, crate::render::RenderError> {
    let on = |channel: Option<&crate::spec::encoding::EncodingSpec>| channel.and_then(|e| e.stack_offset());
    match (on(encoding.x.as_ref()), on(encoding.y.as_ref())) {
        (Some(_), Some(_)) => Err(crate::render::RenderError::StackOnBothChannels),
        (Some(offset), None) => Ok(Some((StackValueAxis::X, offset))),
        (None, Some(offset)) => Ok(Some((StackValueAxis::Y, offset))),
        (None, None) => Ok(None),
    }
}

/// The position adjustment a mark draws with: `position` when set, else a
/// `Stack` along the channel its encoding's `stack` shorthand declares
/// ([`stack_shorthand`]), else none. The draw pass ([`apply_position`]) and
/// the scale-resolve preview ([`stack_preview`]) both read it, so the stack
/// the domain widens to is the one drawn.
///
/// `encoding` is post-`CoordFlip` (the prepare pass swapped it), while a
/// `Stack`'s `value_axis` is authored, pre-flip ([`resolve_stack_value_on_x`]
/// spends the flip). So the declaring channel is moved back into authored
/// space: `value_axis` is `Some(X)` when the user's stacked channel is `x`,
/// and `None` (the vertical default) when it is `y`, so a `y` shorthand,
/// flipped or not, builds exactly the `Stack` it always has.
fn effective_position<'a>(
    position: Option<&'a PositionAdjust>,
    encoding: &crate::spec::encoding::Encoding,
    coord_flipped: bool,
) -> Result<Option<Cow<'a, PositionAdjust>>, crate::render::RenderError> {
    if let Some(position) = position {
        return Ok(Some(Cow::Borrowed(position)));
    }
    let Some((channel, offset)) = stack_shorthand(encoding)? else { return Ok(None) };
    let authored_on_x = (channel == StackValueAxis::X) != coord_flipped;
    let value_axis = authored_on_x.then_some(StackValueAxis::X);
    Ok(Some(Cow::Owned(PositionAdjust::Stack { by: None, offset, anchor: StackAnchor::Top, value_axis })))
}

/// Refuse a `stack` shorthand on a channel whose resolved scale is
/// categorical ([`RenderError::StackShorthandOnCategory`]): a band or point
/// axis holds no value to accumulate, so the stack belongs on the other,
/// value channel. Asked of the scales the draw pass resolved, so a string
/// column and a number declared nominal or ordinal are refused alike.
///
/// [`RenderError::StackShorthandOnCategory`]: crate::render::RenderError::StackShorthandOnCategory
fn refuse_shorthand_on_a_category(
    encoding: &crate::spec::encoding::Encoding,
    scales: &ResolvedScales,
    coord_flipped: bool,
) -> Result<(), crate::render::RenderError> {
    let Some((channel, _)) = stack_shorthand(encoding)? else { return Ok(()) };
    let (scale, binding, channel, value_channel) = match channel {
        StackValueAxis::X => (&scales.x, encoding.x.as_ref(), "x", "y"),
        StackValueAxis::Y => (&scales.y, encoding.y.as_ref(), "y", "x"),
    };
    match (scale, binding) {
        (ScaleKind::Ordinal(_), Some(binding)) => Err(crate::render::RenderError::StackShorthandOnCategory {
            channel,
            value_channel,
            field: binding.field.clone(),
            coord_flipped,
        }),
        _ => Ok(()),
    }
}

/// The encoding a layer's position adjustment reads. Under `CoordPolar` a
/// mark that stacks its theta itself ([`Mark::stacks_polar_theta`]: the arc
/// sweeps it into a pie, a text stacks its labels into that sweep) reads its
/// theta channel's `stack` there, never as a value axis to stack, so that
/// `stack` is left out of what [`apply_position`]'s stack shorthand sees:
/// the shorthand reads both x and y, so whichever of them `CoordPolar`'s
/// `theta` names has its `stack` stripped. Every other layer's encoding,
/// unchanged.
///
/// [`Mark::stacks_polar_theta`]: crate::spec::mark::Mark::stacks_polar_theta
pub(crate) fn position_encoding<'a>(
    mark: crate::spec::mark::Mark,
    coord: Option<&crate::spec::coord::CoordKind>,
    encoding: &'a crate::spec::encoding::Encoding,
) -> Cow<'a, crate::spec::encoding::Encoding> {
    use crate::spec::coord::{CoordKind, PolarThetaChannel};
    let Some(CoordKind::Polar { theta, .. }) = coord else {
        return Cow::Borrowed(encoding);
    };
    if !mark.stacks_polar_theta() || !crate::render::marks::text::theta_stacked(encoding, *theta) {
        return Cow::Borrowed(encoding);
    }
    let mut unstacked = encoding.clone();
    let channel = match theta {
        PolarThetaChannel::X => unstacked.x.as_mut(),
        PolarThetaChannel::Y => unstacked.y.as_mut(),
    };
    if let Some(channel) = channel {
        channel.stack = None;
    }
    Cow::Owned(unstacked)
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
    // With no explicit position, the encoding's `stack` shorthand stacks
    // along the channel that declares it (`effective_position`), which must
    // be a value axis.
    if position.is_none() {
        refuse_shorthand_on_a_category(encoding, scales, coord_flipped)?;
    }
    let Some(p) = effective_position(position, encoding, coord_flipped)? else { return Ok(batch.clone()); };
    match p.as_ref() {
        PositionAdjust::Identity => Ok(batch.clone()),
        PositionAdjust::Dodge { by, padding } => {
            apply_dodge(batch, by.as_deref(), *padding, scales, encoding, coord_flipped, warnings)
        }
        PositionAdjust::Jitter { axis, width, seed } => {
            apply_jitter(batch, axis, *width, *seed, scales, encoding, warnings)
        }
        PositionAdjust::Stack { by, offset, anchor, value_axis } => {
            apply_stack(batch, by.as_deref(), offset, anchor, *value_axis, encoding, scales.color.as_ref(), coord_flipped, warnings)
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
/// The resolution target is `by_field`, else `encoding.color.field`, else
/// `encoding.group.field` (the grouping key of a color the layer inherits but
/// does not draw, a text's), else none.
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
    // Case 1: resolution target = by_field, else color, else group, else
    // nothing requested.
    let by_col_name = match by_field {
        Some(s) => s.to_string(),
        None => encoding.color.as_ref().or(encoding.group.as_ref())?.field.clone(),
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

/// A dodge's or stack's per-row group keys and, when they have one, the
/// order its groups take.
struct GroupKeys {
    cats: Vec<String>,
    order: Option<Vec<String>>,
}

/// The groups a dodge or a stack partitions rows into
/// ([`resolve_group_channel`]'s policy and warnings). Grouped by a binned
/// color (`Color(bin=...)`), or by a text's grouping-only `group` key over
/// one, the rows group by the bin its scale assigns
/// ([`crate::render::draw::binned_color_rows`]), keyed by bin index, in bin
/// order: one dodge slot or one stack layer per bin, low bin first, so a
/// label sits on the bar or segment of its bin. A row with no bin keys to
/// `""`, after every bin. `color` is the resolved color scale, `None` where
/// none is resolved yet.
fn group_keys(
    batch: &RecordBatch,
    by_field: Option<&str>,
    encoding: &crate::spec::encoding::Encoding,
    color: Option<&crate::render::scale_resolve::ColorScale>,
    adjustment_label: &str,
    warnings: &mut Vec<RenderWarning>,
) -> Option<GroupKeys> {
    let cats = resolve_group_channel(batch, by_field, encoding, adjustment_label, warnings)?;
    let binding = encoding.color.as_ref().or(encoding.group.as_ref()).filter(|color| by_field.is_none_or(|by| by == color.field));
    let Some(bins) = crate::render::draw::binned_color_rows(binding, color, batch) else {
        return Some(GroupKeys { cats, order: None });
    };
    let n_bins = match color {
        Some(crate::render::scale_resolve::ColorScale::Discretizing(buckets)) => buckets.colors().len(),
        _ => 0,
    };
    Some(GroupKeys {
        cats: bins.into_iter().map(|bin| bin.map_or_else(String::new, |i| i.to_string())).collect(),
        order: Some((0..n_bins).map(|i| i.to_string()).collect()),
    })
}

/// Each key's group rank under `order`: the groups `order` lists first, in
/// its order, then the rest in first-appearance order
/// ([`ordered_dodge_groups`]).
fn ranks_in_order(cats: &[String], order: &[String]) -> Vec<usize> {
    let rank: HashMap<String, usize> =
        ordered_dodge_groups(cats, Some(order)).into_iter().enumerate().map(|(i, g)| (g, i)).collect();
    cats.iter().map(|c| rank.get(c).copied().unwrap_or(usize::MAX)).collect()
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
/// scale is categorical. A layer grouping by its `group` key (an inherited
/// color it does not draw) reads that key as its color field here, so its
/// slots follow the legend's order as the colored marks beside it do. In
/// every other case returns `None`, which `ordered_dodge_groups` treats as
/// "keep first-appearance order".
fn dodge_slot_domain<'a>(
    by_field: Option<&str>,
    scales: &'a ResolvedScales,
    encoding: &crate::spec::encoding::Encoding,
) -> Option<&'a [String]> {
    let color_enc = encoding.color.as_ref().or(encoding.group.as_ref())?;
    let group_is_color = by_field.is_none_or(|bf| bf == color_enc.field);
    if !group_is_color {
        return None;
    }
    scales.color.as_ref()?.categorical_domain()
}

/// Write `values` into `cols[idx]` and retype `fields[idx]` to `Float64`.
///
/// Both `Dodge`'s continuous branch and `Jitter`'s data-space branch rewrite a
/// positional column with fractional coordinates, which an integer column
/// cannot hold. Before GH #184 both wrote a `Float64Array` into the *input*
/// schema's slot, so an `Int64`/`Int32` positional column made
/// `RecordBatch::try_new` fail with a raw Arrow schema message ("column types
/// must match schema types, expected Int64 but found Float64") escaping as the
/// adjustment's error. Retyping the field alongside the column is the fix, and
/// it lives here rather than twice because forgetting it on one of the two
/// sites is exactly how the defect looked.
///
/// Name, nullability, and field metadata are preserved. Nullability is safe to
/// keep as-is because both callers only ever emit a null where the value they
/// read was null, so the output's null set is a subset of the input's.
fn set_float64_column(
    fields: &mut [Field],
    cols: &mut [ArrayRef],
    idx: usize,
    values: Float64Array,
) {
    let old = &fields[idx];
    fields[idx] = Field::new(old.name(), DataType::Float64, old.is_nullable())
        .with_metadata(old.metadata().clone());
    cols[idx] = Arc::new(values);
}

/// Dense index per row: the position at which that row's category string first
/// appeared in `cats`.
///
/// `apply_stack` asks this question of both its axes, for two different reasons
/// that need the same answer. On the `by` axis the index is the within-bin
/// stacking order, so the group laid down first is the group encountered first.
/// On the category axis it is the bin identity, so two rows cumulate together
/// exactly when their category strings are equal. Written once because the two
/// loops were the same seven lines, and because the category axis's previous key
/// — a 64-bit hash of the value — made a collision a silent mis-stack where
/// string equality cannot collide at all.
fn first_appearance_indices(cats: &[String]) -> Vec<usize> {
    let mut index_of: HashMap<&str, usize> = HashMap::with_capacity(cats.len());
    let mut out: Vec<usize> = Vec::with_capacity(cats.len());
    for c in cats {
        let next = index_of.len();
        out.push(*index_of.entry(c.as_str()).or_insert(next));
    }
    out
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
    let Some(GroupKeys { cats: by_cats, order }) =
        group_keys(batch, by_field, encoding, scales.color.as_ref(), "dodge", warnings)
    else {
        return Ok(batch.clone());
    };

    // Sub-band slot order: bin order under a binned color; the legend's when
    // the dodge grouping field is the color field and the color scale is
    // categorical; otherwise first-appearance order.
    // Resolved once and threaded through both the continuous-band and
    // ordinal-band paths.
    let slot_domain = order.as_deref().or_else(|| dodge_slot_domain(by_field, scales, encoding));

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
    // GH #184: the branch is decided by the RESOLVED scale kind, never by the
    // band column's Arrow dtype. A category axis is "ordinal" because its
    // scale says so — the dtype of the column feeding it is irrelevant, and
    // the two disagree in both directions:
    //
    // - A `Float64` column the user declared `:N` resolves Ordinal but IS
    //   `Float64`, so the old dtype gate sent it down the continuous branch.
    //   That branch rewrites band values in data space (0.0 → 0.75), and the
    //   ordinal scale then looks "0.75" up in a domain of "0"/"1"/"2" and
    //   finds nothing: every bar dropped, silently.
    // - A non-`Float64` numeric column on a genuinely continuous scale
    //   (`native_scale=True` over `Int64`) IS not `Float64`, so the old gate
    //   sent it into `apply_dodge_ordinal`, which had no ordinal scale to
    //   read a bandwidth from and returned the batch unchanged — an
    //   unconditional silent no-op, with every dodge group overlapping.
    //
    // Routing on the scale kind makes both impossible, and passing the
    // `OrdinalScale` itself (rather than the `ScaleKind`) into
    // `apply_dodge_ordinal` deletes that no-op arm outright: the branch is
    // now compiler-enforced instead of re-checked and silently abandoned.
    if let ScaleKind::Ordinal(band_ordinal) = band_scale {
        return apply_dodge_ordinal(batch, &by_cats, padding, band_ordinal, band_on_y, slot_domain);
    }
    // Continuous band: read the positional values through the shared numeric
    // coercion (`arrow_cast::col_as_f64`) rather than a bare `Float64Array`
    // downcast, so every dtype the rest of the render layer treats as
    // quantitative (Int8..64, UInt8..64, Float32, Timestamp) dodges with the
    // same arithmetic `Float64` gets.
    //
    // A band column that is not numeric AT ALL on a continuous scale warns and
    // no-ops, the same answer `resolve_group_channel` gives for a grouping
    // column it cannot categorize. It is reachable — `x="cat:N"` with a numeric
    // `CoordCartesian(xlim=…)` resolves the categorical axis continuous — and
    // it is not the dodge's to refuse: that chart already draws nothing before
    // the dodge is reached, so the axis configuration is the defect and raising
    // here would replace one silent blank with a crash. What this must fix is
    // the SILENCE: pre-#184 the dtype gate sent this shape into
    // `apply_dodge_ordinal`'s unconditional `Ok(batch.clone())` with no
    // diagnostic whatsoever.
    let band_vals = match crate::render::arrow_cast::col_as_f64(batch, &band_field.field) {
        Ok(vals) => vals,
        Err(_) => {
            warnings.push(RenderWarning::PositionAdjustSkipped {
                adjustment: "dodge".into(),
                // Column name only, no channel token: `PositionAdjustSkipped`
                // carries no `coord_flipped` for `Display` to un-flip one with,
                // and the column name is already authored-space.
                reason: format!(
                    "band column '{}' has dtype {:?} on a continuous axis, which cannot carry a data-space offset",
                    band_field.field,
                    batch.schema().field(band_col_idx).data_type(),
                ),
            });
            return Ok(batch.clone());
        }
    };

    // 1. Determine group→slot order: legend order when the dodge field is the
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

    // 2. A SPAN row (`x`+`x2`, or `y`+`y2` when the band is y) carries its own
    //    band as an interval — a histogram bin — rather than a single value the
    //    band is centered on. Both endpoints must move, into the group's own
    //    sub-slot inside that interval.
    //
    //    Offsetting only the band column (which is all this branch did before,
    //    and still does for a point row) computed each rect from an OFFSET start
    //    against an UN-OFFSET end, so `multiple="dodge"` drew bins of
    //    `bin_width ∓ offset`: measured on `displot(bins=10, hue=)`, one group's
    //    bars were 61.46px and the other's 38.88px inside a 50.17px bin — unequal,
    //    overlapping, and one wider than the whole band. The shared grid was
    //    right (that is the Python half of this fix, `_multiple.py`'s
    //    `shared_extent=True` for dodge); only the widths were wrong.
    //
    //    The slot layout is `apply_dodge_ordinal`'s convention, not a new one:
    //    Dodge's `padding` is taken off each END of the band and the `n_groups`
    //    sub-slots are laid end to end from the band's leading edge. The row's
    //    OWN span is subdivided rather than the median-spacing `bandwidth`
    //    proxy, so a non-uniform final bin keeps its slots inside itself; for
    //    the uniform grid a shared-extent histogram produces, the two are equal.
    //
    //    A row whose span is null, non-finite, or non-positive (an inverted or
    //    degenerate bin) is passed through untouched on both columns: there is
    //    no interval to subdivide, and every such row is already dropped
    //    downstream (`col_as_f64`'s null skip, then `bar.rs`'s `is_zero_extent`,
    //    which names the drop).
    //
    //    This branch also needs no bandwidth proxy, which is why it is resolved
    //    BEFORE the median-spacing block below rather than after it: that
    //    block's `uniques.len() < 2` bail exists because spacing is undefined
    //    for a single band value, and applying it to a span row would leave a
    //    one-bin dodged histogram (`bins=1`) with its groups fully overlapped —
    //    the same failure this fix exists to remove, surviving in the one shape
    //    where the band interval is known from the row itself.
    //
    //    The span end is read through the same `col_as_f64` coercion as the
    //    band itself (GH #184): before, a bare `Float64Array` downcast on the
    //    `x2`/`y2` column meant an `Int64` bin-edge pair silently fell through
    //    to the point branch below and dodged as if it had no interval at all.
    //    A present-but-non-numeric span end still falls through to the point
    //    branch — that is not a dodge failure, it is a row with no interval.
    let span_end = if band_on_y { encoding.y2.as_ref() } else { encoding.x2.as_ref() }
        .map(|e| e.field.as_str())
        .and_then(|name| batch.schema().index_of(name).ok().map(|idx| (idx, name)))
        .filter(|(idx, _)| *idx != band_col_idx)
        .and_then(|(idx, name)| {
            crate::render::arrow_cast::col_as_f64(batch, name).ok().map(|vals| (idx, vals))
        });

    let mut cols: Vec<ArrayRef> = batch.columns().to_vec();
    let mut fields: Vec<Field> = batch.schema().fields().iter().map(|f| f.as_ref().clone()).collect();

    if let Some((end_idx, end_vals)) = span_end {
        let mut new_start: Vec<Option<f64>> = Vec::with_capacity(band_vals.len());
        let mut new_end: Vec<Option<f64>> = Vec::with_capacity(band_vals.len());
        for (i, g) in by_cats.iter().enumerate() {
            let group_idx = *seen.get(g).unwrap();
            let (Some(lo), Some(hi)) = (band_vals[i], end_vals[i]) else {
                // Null on either endpoint: both columns keep their exact
                // values (nulls included), so a row this batch could never
                // draw is not turned into one it draws at the wrong place.
                new_start.push(band_vals[i]);
                new_end.push(end_vals[i]);
                continue;
            };
            let span = hi - lo;
            if !span.is_finite() || span <= 0.0 {
                new_start.push(Some(lo));
                new_end.push(Some(hi));
                continue;
            }
            let slot = (span - span * padding * 2.0) / n_groups as f64;
            let slot_lo = lo + span * padding + slot * group_idx as f64;
            new_start.push(Some(slot_lo));
            new_end.push(Some(slot_lo + slot));
        }
        set_float64_column(&mut fields, &mut cols, band_col_idx, Float64Array::from(new_start));
        set_float64_column(&mut fields, &mut cols, end_idx, Float64Array::from(new_end));
    } else {
        // 3. A POINT row's band is centered on its single value, and its width
        //    is the median spacing of the unique band values (the bandwidth
        //    proxy a continuous axis has no ordinal `bandwidth()` for). Each
        //    group's mark moves to the CENTER of its sub-slot, since there is no
        //    second endpoint to carry the slot's extent. Unchanged behavior.
        let mut uniques: Vec<f64> = band_vals.iter().flatten().copied().collect();
        uniques.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        uniques.dedup();
        if uniques.len() < 2 {
            return Ok(batch.clone());
        }
        let mut diffs: Vec<f64> = uniques.windows(2).map(|w| w[1] - w[0]).collect();
        diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let bandwidth = diffs[diffs.len() / 2];
        let pad_total = bandwidth * padding * 2.0;
        let sub_band = (bandwidth - pad_total) / n_groups as f64;

        let mut new_band: Vec<Option<f64>> = Vec::with_capacity(band_vals.len());
        for (i, g) in by_cats.iter().enumerate() {
            let group_idx = *seen.get(g).unwrap();
            let offset =
                -bandwidth / 2.0 + bandwidth * padding + sub_band * (group_idx as f64 + 0.5);
            // A null band value stays null — the same rule the span branch
            // above states for a null endpoint, and for the same reason: a row
            // this batch could never draw must not become one it draws at the
            // wrong place. The pre-#184 `Float64Array::value(i)` read the raw
            // buffer slot under a null (0.0 in practice, undefined by
            // contract) and emitted a NON-null mark at `0.0 + offset`.
            new_band.push(band_vals[i].map(|v| v + offset));
        }
        set_float64_column(&mut fields, &mut cols, band_col_idx, Float64Array::from(new_band));
    }

    // Schema metadata is carried over verbatim (`BatchPositionMeta` keys and
    // anything else upstream stamped); only the rewritten positional fields'
    // dtypes can differ from the input schema, and only when the input was a
    // non-`Float64` numeric column.
    let schema = Arc::new(Schema::new(fields).with_metadata(batch.schema().metadata().clone()));
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
/// `coord_flipped` alone) and supplies the pixel bandwidth. It is an
/// [`OrdinalScale`] rather than a `ScaleKind` (GH #184): this function used to
/// re-`match` the kind and `return Ok(batch.clone())` on any other variant, an
/// unconditional silent no-op that the old dtype-based gate in
/// [`apply_dodge`] could reach with a `Linear` scale — every dodge group
/// overlapping, no warning. Taking the scale the branch already proved makes
/// that arm impossible to write. `slot_domain` is
/// the color-scale domain order threaded from [`apply_dodge`]: `Some` orders
/// sub-band slots by legend order, `None` keeps first-appearance order.
///
/// # Composition rule: scale padding vs. `Dodge` padding
///
/// The two paddings compose; they do not replace or double-count each other
/// (F-L04-03, spec §4A, decision 2). **The scale's padding defines the drawn
/// band** — `bandwidth_px` below is the padding-aware
/// [`OrdinalScale::bandwidth`](crate::scale::ordinal::OrdinalScale::bandwidth),
/// i.e. `|step|·(1 − padding_inner)`, which is the band the user actually
/// sees, not the full `step` slot. **`Dodge` then subdivides THAT band**, and
/// its own `padding` is taken out of it: `n_groups` sub-bands of
/// `(bandwidth_px·(1 − 2·padding)) / n_groups`, laid end to end from the
/// band's leading edge.
///
/// Worked example — `BandScale(padding_inner=0.2)` over `[0, 360]` with two
/// categories, `Dodge(padding=0.05)` with two groups: `step = 200`, drawn
/// band `= 160` of that 200px slot; Dodge's padding eats `160·0.05 = 8` at
/// each end, leaving two 72px sub-bands. The bar formulas then take
/// `bandwidth_px / n_groups · factor` (`160/2·0.8 = 64`) and clamp it to the
/// 72px sub-band via [`BatchPositionMeta::clamp_width`] — so both paddings
/// are visible in the result and neither is applied twice.
///
/// Before F-L04-03 `bandwidth()` ignored `padding_inner`, so Dodge subdivided
/// the whole slot: sub-bands (and the GH #66 clamp derived from them) were
/// computed against a band wider than the one drawn.
///
/// One consequence is worth stating because it is what makes the composition
/// *safe* rather than merely defined: a mark's raw width is
/// `bandwidth_px/n_groups·factor` and its clamp bound is
/// `bandwidth_px·(1 − 2·padding)/n_groups`, so now that both derive from the
/// SAME padding-aware bandwidth their ratio is `factor/(1 − 2·padding)` —
/// **independent of the scale's padding**. The GH #66 clamp therefore fires at
/// exactly the Dodge-padding thresholds it always did (`padding > 0.1` for
/// bar's 0.8 factor, `> 0.2` for box/tick's 0.6) whatever the scale does.
/// Pre-fix the ratio was `factor/((1 − padding_inner)·(1 − 2·padding))`, so
/// scale padding alone could push the clamp into firing and erase Dodge's own
/// padding — at `padding_inner = 0.25, padding = 0.05` the bar ratio was
/// `0.8/(0.75·0.9) = 1.185 > 1`, clamped, and both paddings collapsed into one.
fn apply_dodge_ordinal(
    batch: &RecordBatch,
    by_cats: &[String],
    padding: f64,
    band_scale: &crate::scale::ordinal::OrdinalScale,
    band_on_y: bool,
    slot_domain: Option<&[String]>,
) -> Result<RecordBatch, crate::render::RenderError> {
    let schema = batch.schema();
    let bandwidth_px = band_scale.bandwidth();

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
    // (`bar_width = bandwidth_px / n_groups * 0.8`, and the analogous
    // box/tick formulas) are blind to DODGE's `padding` — they narrow by
    // `n_groups` but never account for the padding eaten out of each
    // sub-band, so at `padding > ~0.1` (bar) the 0.8-factor width can exceed
    // this `sub_band` and adjacent dodge groups overlap. (They are no longer
    // blind to the SCALE's padding: since F-L04-03 they divide the same
    // padding-aware `bandwidth_px` this function does, which is what makes
    // the two paddings compose — see the composition rule above.) Stamping `sub_band`
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
    warnings: &mut Vec<RenderWarning>,
) -> Result<RecordBatch, crate::render::RenderError> {
    use crate::spec::position::JitterAxis;
    use rand::{RngCore, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use twox_hash::xxh3;

    let x_field = encoding.x.as_ref().map(|e| e.field.as_str());
    let y_field = encoding.y.as_ref().map(|e| e.field.as_str());
    let x_idx = x_field.and_then(|f| batch.schema().index_of(f).ok());
    let y_idx = y_field.and_then(|f| batch.schema().index_of(f).ok());

    let do_x = matches!(axis, JitterAxis::X | JitterAxis::Both);
    let do_y = matches!(axis, JitterAxis::Y | JitterAxis::Both);

    // Per-axis ordinality check — for ordinal axes we must NOT overwrite the
    // string column with float noise. Instead we emit a pixel-offset column
    // (`__pos_x_offset__` / `__pos_y_offset__`) that the mark renderers add
    // post-scale. The pixel offset is `(u - 0.5) * width * bandwidth_px`, so
    // `width=1.0` spans the full band; `width=0.4` (default) keeps points
    // well within their band.
    //
    // "Band" here is the DRAWN band, the same one Dodge subdivides: since
    // F-L04-03 `bandwidth()` is padding-aware, so on a padded `BandScale` the
    // spread narrows with the band instead of scattering points across the
    // full slot and into the gap the padding opened. Jitter's `width` is its
    // own fraction OF that band — the two compose exactly as scale padding and
    // `Dodge`'s padding do (see `apply_dodge_ordinal`'s composition rule).
    let x_is_ordinal = matches!(scales.x, ScaleKind::Ordinal(_));
    let y_is_ordinal = matches!(scales.y, ScaleKind::Ordinal(_));
    let x_bandwidth = if let ScaleKind::Ordinal(s) = &scales.x { s.bandwidth() } else { 1.0 };
    let y_bandwidth = if let ScaleKind::Ordinal(s) = &scales.y { s.bandwidth() } else { 1.0 };

    // Positional values are read through the shared numeric coercion, not a
    // bare `Float64Array` downcast (GH #184). The downcast failed on every
    // non-`Float64` numeric column, and the consequences split by branch:
    //
    // - On a CONTINUOUS axis it yielded `None`, so every `xv` was `NaN`, the
    //   whole column was rewritten as `NaN`, and writing a `Float64Array` into
    //   an `Int64` schema slot raised — a raw Arrow schema error surfacing from
    //   `Jitter` on a plain integer x axis.
    // - On an ORDINAL axis the column is not rewritten, so the failure only
    //   reached the `seed: None` hash key below: `format!("{xv}|{yv}")` read
    //   `NaN` and the same chart jittered *differently* depending on whether
    //   its column happened to be `Float64` or `Int64`. The seed is derived
    //   from data, so it must not vary with the dtype that data arrived in.
    //
    // Reading through `col_as_f64` closes both: an `Int64` `3` and a `Float64`
    // `3.0` both read `3.0_f64` and format as `"3"`, so the hash key — and
    // every jittered position — is dtype-invariant across the numeric family.
    // A non-numeric column (a `Utf8` category band) still reads as `None` and
    // still seeds from `NaN`, which is what keeps existing strip-plot goldens
    // byte-identical.
    let x_vals = jitter_axis_values(batch, x_idx.and(x_field), do_x && !x_is_ordinal, warnings);
    let y_vals = jitter_axis_values(batch, y_idx.and(y_field), do_y && !y_is_ordinal, warnings);

    // An axis is rewritten in data space only when it is being jittered, is
    // continuous, AND yielded readable values. The third conjunct is the
    // warn-degraded path: a non-numeric column on a continuous axis (the same
    // `x="cat:N"` + numeric `xlim` shape `apply_dodge` warns about) leaves the
    // column alone rather than replacing it with `NaN`s — which is what made
    // `RecordBatch::try_new` surface a raw Arrow schema message before #184.
    let rewrite_x = do_x && !x_is_ordinal && x_vals.is_some();
    let rewrite_y = do_y && !y_is_ordinal && y_vals.is_some();

    let n = batch.num_rows();
    let mut new_x: Vec<f64> = Vec::with_capacity(n);
    let mut new_y: Vec<f64> = Vec::with_capacity(n);
    let mut x_pixel_offsets: Vec<f64> = Vec::with_capacity(n);
    let mut y_pixel_offsets: Vec<f64> = Vec::with_capacity(n);

    for i in 0..n {
        // A null reads `NaN` exactly as the pre-#184 `is_null` check did, so a
        // nulled row keeps both its un-drawable position and its hash key.
        let xv = x_vals.as_ref().and_then(|v| v[i]).unwrap_or(f64::NAN);
        let yv = y_vals.as_ref().and_then(|v| v[i]).unwrap_or(f64::NAN);

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

        new_x.push(if rewrite_x { xv + noise_x_data } else { xv });
        new_y.push(if rewrite_y { yv + noise_y_data } else { yv });
        x_pixel_offsets.push(if do_x && x_is_ordinal { (u - 0.5) * width * x_bandwidth } else { 0.0 });
        y_pixel_offsets.push(if do_y && y_is_ordinal { (u2 - 0.5) * width * y_bandwidth } else { 0.0 });
    }

    let mut cols: Vec<ArrayRef> = batch.columns().to_vec();
    let mut fields: Vec<Field> = batch.schema().fields().iter().map(|f| f.as_ref().clone()).collect();
    // A rewritten positional column is retyped to Float64 alongside the data
    // (GH #184) — jittered coordinates are fractional and an `Int64` slot
    // cannot hold them; see [`set_float64_column`].
    if let (Some(j), true) = (x_idx, rewrite_x) {
        set_float64_column(&mut fields, &mut cols, j, Float64Array::from(new_x));
    }
    if let (Some(j), true) = (y_idx, rewrite_y) {
        set_float64_column(&mut fields, &mut cols, j, Float64Array::from(new_y));
    }

    let need_offsets = (do_x && x_is_ordinal) || (do_y && y_is_ordinal);
    if need_offsets {
        cols.push(Arc::new(Float64Array::from(x_pixel_offsets)));
        cols.push(Arc::new(Float64Array::from(y_pixel_offsets)));
        fields.push(Field::new("__pos_x_offset__", DataType::Float64, false));
        fields.push(Field::new("__pos_y_offset__", DataType::Float64, false));
    }

    // Both arms carry the input schema's metadata forward. The offset-emitting
    // arm used to build a bare `Schema::new(fields)` and drop it while the
    // other arm reused `batch.schema()` and kept it — two schema builds in one
    // function disagreeing about the same thing, which is what folding them
    // into one build removes.
    let new_schema =
        Arc::new(Schema::new(fields).with_metadata(batch.schema().metadata().clone()));
    let context = if need_offsets { "ordinal: " } else { "" };
    RecordBatch::try_new(new_schema, cols).map_err(|e| crate::render::RenderError::PositionAdjustFailed {
        adjustment: "Jitter",
        reason: crate::render::PositionAdjustReason::Message(format!("{context}{e}")),
        // No positional channel token in either message and no `coord_flipped`
        // in scope (`apply_jitter` doesn't take it) — structurally inert, see
        // the field doc on `PositionAdjustFailed`.
        coord_flipped: false,
    })
}

/// Read a `Jitter` axis column as per-row `Option<f64>`, or `None` when the
/// axis has no column this function can read.
///
/// `field` is `Some` only when the encoding names a column that is actually
/// present in the batch — an encoded-but-absent column stays the per-axis
/// no-op it has always been.
///
/// `rewrites_in_data_space` is true when this axis's values would REPLACE the
/// column (a continuous axis being jittered). Only then does an unreadable
/// column warrant a diagnostic, and it gets the same warn-and-skip
/// `apply_dodge` gives the identical shape (`x="cat:N"` with a numeric
/// `CoordCartesian(xlim=…)`): the axis configuration is the defect, the
/// adjustment just cannot service it. Pre-#184 this path let a raw Arrow schema
/// message escape as the `Jitter` failure, because the unreadable column was
/// rewritten as all-`NaN` `Float64` into an integer slot.
///
/// When the axis is ordinal (or not being jittered at all) the values only feed
/// the hash seed, so a non-numeric column reads as `None` and seeds from `NaN`
/// with no diagnostic, exactly as before — that silent fallback is what keeps
/// every `Utf8`-band strip plot byte-identical.
///
/// The warning names the COLUMN, not the channel: `PositionAdjustSkipped`
/// carries no `coord_flipped` for `Display` to un-flip a positional token with.
fn jitter_axis_values(
    batch: &RecordBatch,
    field: Option<&str>,
    rewrites_in_data_space: bool,
    warnings: &mut Vec<RenderWarning>,
) -> Option<Vec<Option<f64>>> {
    let field = field?;
    match crate::render::arrow_cast::col_as_f64(batch, field) {
        Ok(vals) => Some(vals),
        Err(_) => {
            if rewrites_in_data_space {
                let dtype = batch
                    .column_by_name(field)
                    .map_or_else(|| "unknown".to_string(), |c| format!("{:?}", c.data_type()));
                warnings.push(RenderWarning::PositionAdjustSkipped {
                    adjustment: "jitter".into(),
                    reason: format!(
                        "column '{field}' has dtype {dtype} on a continuous axis, which cannot carry a data-space offset"
                    ),
                });
            }
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Swarm
// ---------------------------------------------------------------------------

/// The transform output a layer reads, the key whose lineage records the
/// `Swarm` that fed it (`transform_lineages`).
pub(crate) fn swarm_key(layer: &crate::render::prepare::LayerPrepared) -> &str {
    layer.data_source.as_deref().unwrap_or(crate::transform::core::FINAL_OUTPUT_KEY)
}

/// Whether a layer sizes its dots itself, by a constant `size` or a size
/// field. A swarm-fed point layer that does not draws them at the swarm's own
/// `point_size` ([`SwarmSpec::dot_area`](crate::transform::swarm::SwarmSpec::dot_area)).
pub(crate) fn sizes_own_dots(layer: &crate::render::prepare::LayerPrepared) -> bool {
    layer.mark_style.as_ref().and_then(|m| m.size).is_some() || layer.encoding.size.is_some()
}

/// The packing bound (px) of one swarm-fed layer: the widest diameter that
/// layer can draw a dot at, read from its scale and style, not its data.
/// Which rows take their size from the scale and which fall back is the
/// point-mark builder's own answer ([`scaled_dot_areas`](crate::render::marks::point::scaled_dot_areas)):
/// - Rows the scale sizes bound at the size range's maximum; a row that
///   falls back (null, or mapped nowhere) widens that to the layer's
///   fallback area (its own constant, else the theme `point_size`) when the
///   fallback is wider.
/// - When every row falls back (no size scale, or a size field this layer's
///   batch cannot read as numeric), the bound is the fallback alone, never
///   widened by a range no row reaches.
/// - A constant area `a` bounds at `2·√(a/π)`.
/// - A layer that sizes nothing itself draws at the swarm's own `point_size`
///   ([`fallback_dot`](crate::render::marks::point::fallback_dot)), so bounds
///   there.
///
/// `None` for a non-point layer, which draws no dot.
fn swarm_layer_bound(
    layer: &crate::render::prepare::LayerPrepared,
    batch: &RecordBatch,
    swarm: &crate::transform::swarm::SwarmSpec,
    scales: &ResolvedScales,
    theme: &crate::layout::ThemeInputs,
) -> Option<f64> {
    use crate::render::marks::point::{fallback_dot, FallbackDot};
    if layer.mark != crate::spec::mark::Mark::Point {
        return None;
    }
    let fallback_area = match fallback_dot(layer, Some(swarm), theme) {
        FallbackDot::Area(area) => area,
        // Every row draws the swarm's own dot: no size channel sizes any.
        swarm_dot @ FallbackDot::Swarm(_) => return Some(swarm_dot.diameter()),
    };
    let scale = scales.size.as_ref();
    let bound_area = match (crate::render::marks::point::scaled_dot_areas(batch, layer.encoding.size.as_ref(), scale), scale) {
        (Some(areas), Some(scale)) => {
            let range_max = scale.min_px().max(scale.max_px());
            if areas.iter().any(Option::is_none) { range_max.max(fallback_area) } else { range_max }
        }
        _ => fallback_area,
    };
    Some(ferrum_scene::dot_diameter(bound_area))
}

/// How each of a chart's layers is placed by the swarm that feeds it,
/// indexed like `layers`: one packing per swarm for the whole chart, at the
/// largest bound ([`swarm_layer_bound`]) of any point layer it feeds, so no
/// drawn dot overlaps a neighbour and every facet panel packs at the same
/// diameter. `chart_layer_batches` are the layers' batches across every
/// panel, before any facet filter: a layer with no rows in this panel still
/// bounds the packing, and a null or unmapped size row anywhere in the chart
/// widens it everywhere. Every layer a swarm feeds (points, text, any
/// overlay) is placed by that one packing, so they share their centres; each
/// still draws at its own size, a point layer that sizes nothing itself at
/// the declared swarm's own `point_size`.
pub(crate) fn chart_swarms<'a>(
    layers: &[crate::render::prepare::LayerPrepared],
    chart_layer_batches: &[RecordBatch],
    transform_lineages: &'a crate::transform::run::OutputLineages,
    scales: &ResolvedScales,
    theme: &crate::layout::ThemeInputs,
) -> Vec<Option<crate::transform::swarm::SwarmPacking<'a>>> {
    let mut widest: HashMap<&str, f64> = HashMap::new();
    for (layer, batch) in layers.iter().zip(chart_layer_batches) {
        let key = swarm_key(layer);
        let Some(swarm) = transform_lineages.get(key).and_then(|lineage| lineage.swarm.as_ref()) else {
            continue;
        };
        if let Some(d) = swarm_layer_bound(layer, batch, swarm, scales, theme) {
            let entry = widest.entry(key).or_insert(d);
            *entry = entry.max(d);
        }
    }
    layers
        .iter()
        .map(|layer| {
            let key = swarm_key(layer);
            let spec = transform_lineages.get(key)?.swarm.as_ref()?;
            let diameter_px = widest.get(key).copied().unwrap_or(spec.point_size);
            Some(crate::transform::swarm::SwarmPacking { spec, diameter_px })
        })
        .collect()
}

/// Place a swarm layer's dots through the resolved panel scale.
///
/// The `Swarm` transform only validates its columns; the packing needs the
/// rendered value axis (plot rect, end insets, facet-shared or explicit domain,
/// scale kind), which exists only here, beside the band-pixel offsets `Jitter`
/// computes in [`apply_jitter`]. The placed offsets land in the column of the
/// axis that carries the category band: the value axis is `y` for a vertical
/// swarm, `x` for a horizontal one, exchanged when `layer_coord`, the layer's
/// coord, flips its channels.
///
/// The packing places whichever rows the layer draws. When a later transform
/// dropped the category, value or dodge column it reads, the dots are left
/// unplaced and a `PositionAdjustSkipped` warning names the column: `Ok(None)`. Dots that reach past half the category slot on a category axis
/// spill into the neighbouring categories (or past the plot edge); they are
/// drawn where they were packed, and `SwarmOverflowsBand`, voiced as `entry`,
/// says so.
pub(crate) fn apply_swarm(
    batch: &RecordBatch,
    packing: crate::transform::swarm::SwarmPacking<'_>,
    scales: &ResolvedScales,
    layer_coord: Option<&crate::spec::coord::CoordKind>,
    entry: &str,
    warnings: &mut Vec<RenderWarning>,
) -> Result<Option<RecordBatch>, crate::render::RenderError> {
    use crate::transform::swarm::{place_offsets, SwarmOrient};
    let swarm = packing.spec;
    let coord_flipped = crate::spec::coord::flips_positional_channels(layer_coord);
    let failed = |reason: String| crate::render::RenderError::PositionAdjustFailed {
        adjustment: "Swarm",
        reason: crate::render::PositionAdjustReason::Message(reason),
        coord_flipped,
    };
    let named = format!("Swarm(category='{}', value='{}')", swarm.category, swarm.value);
    let read = [Some(swarm.category.as_str()), Some(swarm.value.as_str()), swarm.dodge.as_deref()];
    if let Some(missing) = read.into_iter().flatten().find(|c| batch.column_by_name(c).is_none()) {
        let reason = format!(
            "column '{missing}' that {named} places its dots by is not in the layer's data; a later transform dropped it"
        );
        let warning = RenderWarning::PositionAdjustSkipped { adjustment: "swarm".into(), reason };
        if !warnings.contains(&warning) {
            warnings.push(warning);
        }
        return Ok(None);
    }
    let value_on_y = matches!(swarm.orient, SwarmOrient::Vertical) != coord_flipped;
    let (value_scale, cross_scale, cross_column) = if value_on_y {
        (&scales.y, &scales.x, "__pos_x_offset__")
    } else {
        (&scales.x, &scales.y, "__pos_y_offset__")
    };
    let values = crate::render::arrow_cast::col_as_f64(batch, &swarm.value)
        .map_err(|e| failed(e.to_string()))?;
    let offsets = place_offsets(packing, batch, &values, |v| value_scale.to_pixel_f64(v))
        .map_err(failed)?;

    if let ScaleKind::Ordinal(category_scale) = cross_scale {
        let reach = offsets.iter().fold(0.0_f64, |a, o| a.max(o.abs())) + packing.diameter_px / 2.0;
        let half_slot = category_scale.step() / 2.0;
        let not_yet_warned = !warnings.iter().any(|w| {
            matches!(w, RenderWarning::SwarmOverflowsBand { entry: e, category, value, .. }
                if e == entry && *category == swarm.category && *value == swarm.value)
        });
        if reach > half_slot && not_yet_warned {
            warnings.push(RenderWarning::SwarmOverflowsBand {
                entry: entry.to_string(),
                category: swarm.category.clone(),
                value: swarm.value.clone(),
                reach_px: reach,
                half_slot_px: half_slot,
            });
        }
    }

    let schema = batch.schema();
    // Replaces a column of the cross name already in the batch (an earlier
    // opposite-orient Swarm's placement on a chain under a flipped layer, or
    // a position column of that name), so `column_by_name` reads the offsets
    // pushed below.
    let (mut fields, mut cols): (Vec<Field>, Vec<ArrayRef>) = schema
        .fields()
        .iter()
        .zip(batch.columns())
        .filter(|(f, _)| f.name() != cross_column)
        .map(|(f, c)| (f.as_ref().clone(), c.clone()))
        .unzip();
    fields.push(Field::new(cross_column, DataType::Float64, false));
    cols.push(Arc::new(Float64Array::from(offsets)));
    let new_schema = Arc::new(Schema::new(fields).with_metadata(schema.metadata().clone()));
    RecordBatch::try_new(new_schema, cols).map(Some).map_err(|e| failed(e.to_string()))
}

// ---------------------------------------------------------------------------
// Stack
// ---------------------------------------------------------------------------

/// Resolve whether a Stack's cumulated *value* column lives on X **as rendered**
/// (GH #77).
///
/// Shared by [`apply_stack`] (the draw-time cumulation) and [`stack_preview`]
/// (the scale-resolve domain-widening preview) so the passes can never
/// disagree about which axis holds the value. Both read
/// `PositionAdjust::Stack::value_axis` as authored and the flip of the coord
/// the mark draws in, so this is the single place the two meet.
///
/// # The two spaces, and why this is an XOR rather than a precedence
///
/// `value_axis` is the producer's declaration about the series it AUTHORED
/// (`desugar_histogram(orientation="horizontal")` remaps its count onto `x` and
/// says `value_axis="x"`), while `coord_flipped` is a TRANSFORMATION applied to
/// that authored series afterwards. So a declared axis is not the final answer:
/// the flip moves it, exactly as it moves the channels themselves. Undeclared
/// means the authored default, vertical (value on y), which the flip likewise
/// moves onto x — so all four cells fall out of one expression, "authored on x,
/// then flipped an odd number of times".
///
/// This is the close-review F3 fix. The former shape was a PRECEDENCE
/// (`Some(X) => true, Some(Y) => false, None => coord_flipped`) — correct for
/// `mark_style.value_axis`, whose copy [`LayerPrepared::flip_coords`]
/// (crate::render::prepare) has ALREADY swapped into post-flip space before
/// `DrawCtx::series_curve_axis` reads it, and wrong here, where nothing swaps
/// `position` and the declaration arrives as authored. Mirroring that triad's
/// formula across the space boundary is what made the two disagree: measured,
/// `mark_histogram(bins=3, orientation="horizontal", multiple="stack")
/// .encode(y="v:Q", color="g:N").coord(CoordFlip())` drew nine rects, four of
/// them sharing one `y`/`height` with overlapping widths — the bin-start column
/// cumulated and the count column used as the bin key — where both its unflipped
/// spelling and its vertical twin drew the same correct three.
///
/// `position`'s declaration is therefore reconciled HERE rather than swapped in
/// `flip_coords`' eager coordinate-space group: its consumers already carry
/// the real `coord_flipped` and all reach it through this function, whereas
/// the unlayered preview ([`axis_batch`]) reads `spec.position` RAW, so an
/// eager swap would have to be applied there as well: one rule in two places
/// instead of one. See `flip_coords`' doc for the other half of that split.
///
/// Only `Some(_)` + `coord_flipped` changes answer: `None` (every chart with no
/// explicit declaration, flipped or not) and every unflipped chart resolve
/// exactly as before, which is what keeps stacked-bar output byte-identical.
pub(crate) fn resolve_stack_value_on_x(
    value_axis: Option<StackValueAxis>,
    coord_flipped: bool,
) -> bool {
    let authored_on_x = matches!(value_axis, Some(StackValueAxis::X));
    authored_on_x != coord_flipped
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
    color: Option<&crate::render::scale_resolve::ColorScale>,
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
    let Some(GroupKeys { cats: by_cats, order }) = group_keys(batch, by_field, encoding, color, "stack", warnings) else {
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
    // `None`. Shared with `stack_preview` via `resolve_stack_value_on_x` so
    // domain widening can never desync from cumulation.
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
    // The value column is read through `arrow_cast::col_as_f64`, the same
    // single source of truth `apply_dodge` and `apply_jitter` read positional
    // values through (#184). Until then this was a hand-rolled six-arm downcast
    // chain (Float64, UInt64, Int64/32/16/8) that refused everything else, so
    // one module held two answers to "which dtypes are numeric": a `Float32`,
    // `UInt8`/`UInt16`/`UInt32` or `Timestamp` column dodged and jittered but
    // raised here. `col_as_f64` delegates to `supported_numeric_dtype`, so the
    // three adjustments now widen exactly the same set and cannot drift apart
    // again — a dtype added there reaches all three at once.
    //
    // A null measure reads as 0.0, which is precisely what the chain's
    // `if a.is_null(i) { 0.0 }` arms did on every one of their six dtypes. This
    // is a stacked CUMULATION, not a position offset: the dodge point branch's
    // "a null stays null" rule is the opposite answer to a different question
    // (there a null row must not be drawn at all; here a null contributes
    // nothing to the running total but the rows above it still stack).
    let y_col = batch.column(yi);
    let ya_vals: Vec<f64> = crate::render::arrow_cast::col_as_f64(batch, &value_field.field)
        .map_err(|_| {
            // R3: `y_col` is always the VALUE column (bound from `yi` = the
            // resolved `value_field` index), whichever of x/y it actually reads
            // from — see `value_token` above, computed from the same
            // `value_on_x`/`coord_flipped` resolution this dtype check is
            // downstream of. The reason variant is unchanged, so the refusal
            // still un-flips its channel token at `Display` time; only the set
            // of dtypes that reach it has narrowed.
            crate::render::RenderError::PositionAdjustFailed {
                adjustment: "Stack",
                reason: crate::render::PositionAdjustReason::ValueDtype {
                    channel: value_token,
                    dtype: format!("{:?}", y_col.data_type()),
                },
                coord_flipped,
            }
        })?
        .into_iter()
        .map(|v| v.unwrap_or(0.0))
        .collect();
    let ya_len = ya_vals.len();

    // The category column is read through `arrow_cast::col_as_identity_key_str`,
    // the crate's one answer to "which value is this row" — the ordinal-category
    // reader the band ladder is enumerated with (`distinct_positional_categories`
    // produces exactly these strings, both delegating to
    // `col_as_ordinal_category_str`/`float_as_ordinal_str`), falling back to the
    // temporal epoch reader for `Timestamp`. Two rows therefore share a stack bin
    // if and only if they share a band wherever a band ladder exists, by
    // construction, and bin by exact instant where one does not (#184, category
    // half).
    //
    // One asymmetry, deliberate (`#247`B): the ladder can omit the
    // `NULL_CATEGORY` band when nothing is drawn in it, while the bins below are
    // a partition of ROWS and always keep their null bin. It is the safe
    // direction. A stack only rewrites the VALUE column, never the category, so
    // an extra bin cannot move a real category's cumulative tops off its tick —
    // the null bin's rows are precisely the rows no other bin contains, and each
    // of them is one whose value was null (that is why the ladder dropped the
    // band), so the segment it stacks is the zero-height one below and the
    // scale lookup then places no mark for it at all. Giving the bins the
    // ladder's occupancy rule would buy nothing and would make the stack's
    // binning depend on the axis it does not read.
    //
    // The temporal arm is what makes a stacked time series render — an ordinary
    // `mark_area(position=Stack())` over a date x, which raised for every input
    // before. It is not a second keying model bolted on: a stack over a
    // continuous quantitative axis already bins by exact value through the first
    // reader (an `Int32` x:Q stacks to its group sum), so exact-value keying over
    // a continuous axis is the established behavior and `Timestamp` was simply
    // the one dtype the reader could not spell.
    //
    // Until then this was a two-arm dtype gate — `Float64` keyed on
    // `f64::to_bits`, `Utf8` on a hash of the bytes, everything else a typed
    // `CategoryDtype` refusal — so `mark_bar(position=Stack()).encode(x=<Int64
    // as :N>, y, color=)` raised "x column must be Float64 or Utf8" while the
    // identical column dodged, jittered and banded fine. That is a pre-existing
    // hole, not a regression introduced by this wave's `#171`: the refusal is
    // byte-identical with and without it.
    //
    // Two `Float64` partitions differ from the old `to_bits` key, both toward
    // the ladder rather than away from it: `-0.0`/`0.0` (two bit patterns, one
    // band) now share a bin instead of stacking twice at one pixel, and a null
    // category row keys to its own `NULL_CATEGORY` band instead of merging into
    // the real `0.0` bin that `Float64Array::value` on a null used to return.
    // Existing `Utf8` stacks are byte-identical: their nulls used to key on the
    // `""` that `StringArray::value` returns, which partitioned them together
    // exactly as `NULL_CATEGORY` now does.
    let x_col = batch.column(xi);
    let x_cats: Vec<String> = crate::render::arrow_cast::col_as_identity_key_str(batch, &cat_field.field)
        .map_err(|_| {
            // R3: `x_col` is always the CATEGORY column (bound from `xi` = the
            // resolved `cat_field` index) — see `cat_token` above. The reason
            // variant is unchanged, so the refusal still un-flips its channel
            // token at `Display` time; it now carries the offending dtype, and
            // the set it names is the identity reader's.
            crate::render::RenderError::PositionAdjustFailed {
                adjustment: "Stack",
                reason: crate::render::PositionAdjustReason::CategoryDtype {
                    channel: cat_token,
                    dtype: format!("{:?}", x_col.data_type()),
                },
                coord_flipped,
            }
        })?
        .into_iter()
        // A null category row bins under the `NULL_CATEGORY` label, which is the
        // band `distinct_positional_categories` gives it (FA-9) — so a null
        // group stacks in its own band rather than merging into the real `0.0`
        // bin `Float64Array::value` returns for a null, and a null instant on a
        // temporal axis lands in one bin rather than several.
        .map(|c| c.unwrap_or_else(|| crate::render::arrow_cast::NULL_CATEGORY.to_string()))
        .collect();

    // Bin identity for the category axis, and stacking order for the `by`
    // groups: both are "which distinct value did this row's string first
    // appear as", so both read it off the same helper.
    let x_bins = first_appearance_indices(&x_cats);
    let group_idx = match order {
        Some(order) => ranks_in_order(&by_cats, &order),
        None => first_appearance_indices(&by_cats),
    };

    // bins: x_bin → Vec<(group_idx, row_idx, y)>
    let mut bins: BTreeMap<usize, Vec<(usize, usize, f64)>> = BTreeMap::new();
    for i in 0..ya_len {
        bins.entry(x_bins[i]).or_default().push((group_idx[i], i, ya_vals[i]));
    }

    // Each bin's magnitude `P + N`: the sum of its positive values and the
    // absolute values of its negative ones, the denominator `Normalize`
    // divides by. With no negative value it is the bin's plain sum.
    let magnitudes: HashMap<usize, f64> = bins
        .iter()
        .map(|(k, rows)| (*k, rows.iter().map(|(_, _, y)| y.abs()).sum::<f64>()))
        .collect();

    // Stacking is split by sign: within a bin, non-negative values accumulate
    // upward from 0 and negative values downward from 0, each in group order,
    // so a negative segment hangs below the axis instead of overdrawing the
    // positive segment under it. The stack spans `[-N, P]`; `Normalize`
    // scales it to `[-n, p]` with `p + n = 1`; `Center` shifts it by
    // `(N - P) / 2` so that span is centered on 0. A bin with no negative value
    // stacks exactly as a single running total would.
    //
    // new_y holds each segment's value end and new_y_base its base end (the
    // previous same-sign segment's value end, or 0 for the first). Bar / area
    // renderers draw each segment from new_y_base[i] to new_y[i], so segments
    // never overlap.
    let mut new_y = vec![0.0_f64; ya_len];
    let mut new_y_base = vec![0.0_f64; ya_len];
    for (xkey, rows) in bins.iter_mut() {
        rows.sort_by_key(|(gi, _, _)| *gi);
        let magnitude = magnitudes.get(xkey).copied().unwrap_or(0.0);
        let mut positive_acc = 0.0_f64;
        let mut negative_acc = 0.0_f64;
        for (_, row_idx, y) in rows.iter() {
            let normalized = match offset {
                StackOffset::Zero => *y,
                StackOffset::Normalize => {
                    if magnitude != 0.0 {
                        y / magnitude
                    } else {
                        0.0
                    }
                }
                StackOffset::Center => *y,
            };
            let acc = if normalized < 0.0 { &mut negative_acc } else { &mut positive_acc };
            new_y_base[*row_idx] = *acc;
            *acc += normalized;
            new_y[*row_idx] = *acc;
        }
        if matches!(offset, StackOffset::Center) {
            let mid = (positive_acc + negative_acc) / 2.0;
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
    let mut new_fields: Vec<Field> = batch
        .schema()
        .fields()
        .iter()
        .map(|f| f.as_ref().clone())
        .collect();
    // Promote the value column to Float64 (the input may have been UInt64, e.g.
    // Bin's `count`) through the SAME helper dodge and jitter retype with. This
    // was the third retype site in this file and the one `set_float64_column`
    // did not reach: it hand-built `Field::new(name, Float64, true)`, forcing
    // nullability and dropping field metadata where the helper preserves both.
    // Inert as written — `y_output` is null-free, so the forced `true` never
    // mattered — but a third hand-rolled copy of the thing the helper exists to
    // single-source is exactly the drift the helper's own doc warns about.
    set_float64_column(&mut new_fields, &mut cols, yi, Float64Array::from(y_output));

    // Append a synthetic __stack_y_base__ column so mark drawers can emit
    // per-segment rects (base → top) instead of drawing every segment from
    // y=0. Bar / area renderers look this up via `col_as_f64` when present.
    cols.push(Arc::new(Float64Array::from(new_y_base)));
    new_fields.push(Field::new(STACK_BASE_COLUMN, DataType::Float64, true));

    // GH #77 follow-up: stamp explicit schema metadata (mirroring the
    // `DODGE_N_GROUPS_KEY` / `DODGE_SUB_BAND_PX_KEY` pattern from
    // `apply_dodge_ordinal`, GH #66) recording that the stacked *value*
    // column landed on the X axis this pass, not Y. `__stack_y_base__` is
    // always the same column regardless of axis, so mark drawers (`bar.rs`,
    // `area.rs`) that consume it need an explicit signal for which scale to
    // map it through — they can't infer it from the batch shape alone.
    // Absence of the key is the pre-#77 default and keeps every existing
    // vertical-stack mark byte-identical.
    //
    // ONE schema build, carrying the input metadata forward on both paths, for
    // the reason `apply_jitter` states at its own build (close review F5): this
    // used to stamp into a clone of `batch.schema().metadata()` on the
    // `value_on_x` path and build a bare `Schema::new(new_fields)` on the other —
    // two builds in one function disagreeing about whether input metadata
    // survives, the third member of the retype/schema/metadata family `apply_dodge`
    // already gets right on every path. Byte-identical today (the only live
    // producer of batch metadata is `apply_dodge_ordinal`, and `apply_position`
    // applies exactly one adjustment per layer, so nothing can chain into here),
    // which is why it is drift rather than a defect.
    let mut metadata = batch.schema().metadata().clone();
    if value_on_x {
        BatchPositionMeta::stamp_stack_value_on_x(&mut metadata);
    }
    let new_schema = Arc::new(Schema::new(new_fields).with_metadata(metadata));
    RecordBatch::try_new(new_schema, cols).map_err(|e| crate::render::RenderError::PositionAdjustFailed {
        adjustment: "Stack",
        reason: crate::render::PositionAdjustReason::Message(format!("{e}")),
        coord_flipped,
    })
}

#[cfg(test)]
mod tests {
    use crate::scale::discrete::DiscreteLayout;
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
            fill_opacity: None,
            stroke_opacity: None,
            stroke_dash: None,
            stroke_width: None,
            x2: None,
            y2: None,
            y_slots: Default::default(),
            raster_cells: None,
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

    /// A flipped layer whose batch already carries the cross column under
    /// its own name — the shape left behind by an earlier Swarm of the
    /// opposite orientation in the same chain, whose placeholder is that
    /// name — does not end up with two fields of that name. The stale
    /// column (here `99.0`, standing in for an earlier Swarm's untouched
    /// `0.0`) must be dropped, not merely shadowed by the real offsets
    /// pushed after it.
    #[test]
    fn apply_swarm_replaces_a_stale_cross_column() {
        use crate::transform::swarm::{SwarmOrient, SwarmSide, SwarmSpec};
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, false),
            Field::new("v", DataType::Float64, false),
            // Stale: a column of THIS swarm's cross name once the layer is
            // flipped.
            Field::new("__pos_y_offset__", DataType::Float64, false),
        ]));
        let b = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["A"; 5])),
                Arc::new(Float64Array::from(vec![1.0; 5])),
                Arc::new(Float64Array::from(vec![99.0; 5])),
            ],
        )
        .unwrap();
        let swarm = SwarmSpec {
            category: "g".into(),
            value: "v".into(),
            point_size: 5.0,
            spacing: 1.0,
            side: SwarmSide::Both,
            orient: SwarmOrient::Vertical,
            dodge: None,
            name: None,
        };
        // A CoordFlip on a Vertical swarm puts the value on x and the
        // packed offsets into `__pos_y_offset__` — the stale column's name.
        let flip = crate::spec::coord::CoordKind::Flip;
        let packing = crate::transform::swarm::SwarmPacking { spec: &swarm, diameter_px: swarm.point_size };
        let out = apply_swarm(&b, packing, &dummy_scales(), Some(&flip), "mark_swarm", &mut Vec::new())
            .unwrap()
            .unwrap();
        let names: Vec<String> = out.schema().fields().iter().map(|f| f.name().clone()).collect();
        let unique: std::collections::HashSet<&String> = names.iter().collect();
        assert_eq!(names.len(), unique.len(), "duplicate field name in {names:?}");
        let offsets = out
            .column_by_name("__pos_y_offset__")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .values()
            .to_vec();
        // The real packed offsets (five identical points at one category/value
        // spread by point_size + spacing) survive; the stale 99.0 does not.
        assert_ne!(offsets, vec![99.0; 5], "stale cross column read instead of the real offsets");
        let mut sorted = offsets.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(sorted, vec![-12.0, -6.0, 0.0, 6.0, 12.0]);
    }

    /// A swarm-fed point layer whose OWN size field is not numeric in its own
    /// batch — wrong dtype, or a later transform dropped the column — is not
    /// "some row falls back" (the `Ok(values)` case, where a row's size is
    /// null or maps nowhere on an otherwise-numeric column): every row of
    /// THIS layer falls back, because `marks/point.rs`'s own per-row loader
    /// reads this exact column through `col_as_f64(...).ok()`, drawing every
    /// row at the fallback radius regardless of why the read failed. The
    /// packing bound must match that — the fallback area alone — never
    /// widened toward a size-range maximum no row of this layer actually
    /// reaches. A range max of 100px with a 36px theme fallback is the
    /// regression shape: the pre-fix `Err(_) => true` arm fed `some_row_falls_back`
    /// unconditionally, so `range_max.max(fallback_area)` picked the much
    /// wider range max.
    #[test]
    fn swarm_layer_bound_non_numeric_size_column_bounds_at_the_fallback_not_the_range() {
        use crate::render::scale_resolve::SizeScale;
        use crate::render::prepare::LayerPrepared;
        use crate::scale::linear::LinearScale;
        use crate::spec::mark::Mark;
        use crate::transform::swarm::{SwarmOrient, SwarmSide, SwarmSpec};

        let spec = ChartSpec {
            mark: Mark::Point,
            encoding: Encoding {
                size: Some(EncodingSpec { field: "size_col".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            ..Default::default()
        };
        let layer = LayerPrepared::from_chart_only(&spec);

        // `size_col` exists but is Utf8 — `col_as_f64` fails with
        // `UnsupportedDtype`, not a missing column.
        let schema = Arc::new(Schema::new(vec![Field::new("size_col", DataType::Utf8, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec!["small", "medium", "large"]))],
        )
        .unwrap();

        let mut scales = dummy_scales();
        scales.size = Some(SizeScale {
            inner: ScaleKind::Linear(LinearScale::new_internal(
                vec![0.0, 1.0],
                vec![4.0, 100.0],
                false,
                true,
            )),
        });

        let theme = crate::layout::ThemeInputs::default();
        let swarm = SwarmSpec {
            category: "g".into(),
            value: "v".into(),
            point_size: 10.0,
            spacing: 1.0,
            side: SwarmSide::Both,
            orient: SwarmOrient::Vertical,
            dodge: None,
            name: None,
        };

        let bound = swarm_layer_bound(&layer, &batch, &swarm, &scales, &theme).unwrap();
        let fallback_only = ferrum_scene::dot_diameter(theme.sizes.point_size);
        let widened_by_range = ferrum_scene::dot_diameter(100.0_f64.max(theme.sizes.point_size));
        assert_eq!(bound, fallback_only, "a non-numeric size column must bound at the fallback area alone");
        assert_ne!(bound, widened_by_range, "must not widen toward a range max no row of this layer reaches");
    }

    /// A size scale whose range runs high to low (`range=[36, 4]`) draws its
    /// widest dot at the range's FIRST end, so the packing bound reads the
    /// wider end of the range whichever way it runs, and bounds every dot
    /// the layer draws. Forward and reversed ranges bound alike.
    #[test]
    fn swarm_layer_bound_reads_the_wider_end_of_a_reversed_size_range() {
        use crate::render::prepare::LayerPrepared;
        use crate::render::scale_resolve::SizeScale;
        use crate::scale::linear::LinearScale;
        use crate::spec::mark::Mark;
        use crate::transform::swarm::{SwarmOrient, SwarmSide, SwarmSpec};

        let spec = ChartSpec {
            mark: Mark::Point,
            encoding: Encoding {
                size: Some(EncodingSpec { field: "s".into(), ..Default::default() }),
                ..Default::default()
            },
            ..Default::default()
        };
        let layer = LayerPrepared::from_chart_only(&spec);
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Float64, false)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(vec![0.0, 0.5, 1.0]))]).unwrap();
        let theme = crate::layout::ThemeInputs::default();
        let swarm = SwarmSpec {
            category: "g".into(),
            value: "v".into(),
            point_size: 3.0,
            spacing: 1.0,
            side: SwarmSide::Both,
            orient: SwarmOrient::Vertical,
            dodge: None,
            name: None,
        };
        let bound_for = |range: Vec<f64>| {
            let mut scales = dummy_scales();
            scales.size = Some(SizeScale {
                inner: ScaleKind::Linear(LinearScale::new_internal(vec![0.0, 1.0], range, false, false)),
            });
            let areas = crate::render::marks::point::scaled_dot_areas(&batch, layer.encoding.size.as_ref(), scales.size.as_ref())
                .expect("every row is sized");
            let widest_drawn = areas.into_iter().map(|a| a.expect("every row maps")).fold(0.0, f64::max);
            (swarm_layer_bound(&layer, &batch, &swarm, &scales, &theme).unwrap(), widest_drawn)
        };
        let (reversed, widest_reversed) = bound_for(vec![36.0, 4.0]);
        assert_eq!(widest_reversed, 36.0, "vacuity: the reversed range draws its widest dot at its first end");
        assert_eq!(reversed, ferrum_scene::dot_diameter(36.0));
        assert!(reversed >= ferrum_scene::dot_diameter(widest_reversed));
        let (forward, _) = bound_for(vec![4.0, 36.0]);
        assert_eq!(forward, reversed);
    }

    /// One swarm feeding a constant-sized point layer and an unsized one
    /// packs at the sized layer's diameter, while the packing still carries
    /// the swarm as declared: its `point_size`, the size the unsized layer
    /// draws at, is not overwritten by the wider packing diameter.
    #[test]
    fn chart_swarms_packs_at_the_widest_dot_and_keeps_the_declared_point_size() {
        use crate::render::prepare::LayerPrepared;
        use crate::spec::mark::Mark;
        use crate::spec::mark_style::MarkKwargsSpec;
        use crate::transform::swarm::{SwarmOrient, SwarmSide, SwarmSpec};
        let layer = |size: Option<f64>| {
            let spec = ChartSpec {
                mark: Mark::Point,
                mark_style: size.map(|size| MarkKwargsSpec { size: Some(size), ..Default::default() }),
                ..Default::default()
            };
            LayerPrepared::from_chart_only(&spec)
        };
        let layers = [layer(Some(200.0)), layer(None)];
        let batches = [batch_xyg(), batch_xyg()];
        let declared = SwarmSpec {
            category: "g".into(),
            value: "y".into(),
            point_size: 5.0,
            spacing: 1.0,
            side: SwarmSide::Both,
            orient: SwarmOrient::Vertical,
            dodge: None,
            name: None,
        };
        let lineage = crate::transform::run::OutputLineage {
            swarm: Some(declared.clone()),
            statistic_input: None,
            lowered: Vec::new(),
            source: batch_xyg().schema(),
        };
        let lineages = HashMap::from([(crate::transform::core::FINAL_OUTPUT_KEY.to_string(), lineage)]);
        let theme = crate::layout::ThemeInputs::default();
        let packings = chart_swarms(&layers, &batches, &lineages, &dummy_scales(), &theme);
        assert_eq!(packings.len(), 2);
        for packing in packings {
            let packing = packing.expect("both layers read the swarm-fed output");
            assert_eq!(packing.spec, &declared, "the declared swarm is carried unchanged");
            assert_eq!(packing.diameter_px, ferrum_scene::dot_diameter(200.0));
            assert!(packing.diameter_px > declared.point_size, "vacuity: the sized layer widens the packing");
        }
    }

    /// Placement and the overflow reach read the packing diameter, not the
    /// declared swarm's `point_size`: five dots at one value sit
    /// `diameter_px + spacing` apart.
    #[test]
    fn apply_swarm_places_at_the_packing_diameter() {
        use crate::transform::swarm::{SwarmOrient, SwarmPacking, SwarmSide, SwarmSpec};
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, false),
            Field::new("v", DataType::Float64, false),
        ]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec!["A"; 5])), Arc::new(Float64Array::from(vec![1.0; 5]))],
        )
        .unwrap();
        let swarm = SwarmSpec {
            category: "g".into(),
            value: "v".into(),
            point_size: 5.0,
            spacing: 1.0,
            side: SwarmSide::Both,
            orient: SwarmOrient::Vertical,
            dodge: None,
            name: None,
        };
        let packing = SwarmPacking { spec: &swarm, diameter_px: 9.0 };
        let out = apply_swarm(&b, packing, &dummy_scales(), None, "mark_swarm", &mut Vec::new()).unwrap().unwrap();
        let mut offsets = out
            .column_by_name("__pos_x_offset__")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .values()
            .to_vec();
        offsets.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(offsets, vec![-20.0, -10.0, 0.0, 10.0, 20.0]);
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

    /// A dodged SPAN row (`x`+`x2` — a histogram bin) must land in its group's
    /// own sub-slot INSIDE the bin: equal widths across groups, adjacent, and
    /// together spanning the bin minus Dodge's padding at each end. This is the
    /// half of `multiple="dodge"` that was missing once the Python side started
    /// sharing bin edges across hue groups (`_multiple.py`'s
    /// `shared_extent=True`): the shared grid was right, but this branch
    /// offset only `x` and left `x2` where it was, so each rect was measured
    /// from an offset start to an un-offset end.
    ///
    /// RED against that: with only `x` offset, row 0 (group a, padding 0) spans
    /// [-0.25, 1.0] — width 1.25, wider than the whole bin — and row 1 (group b)
    /// spans [0.25, 1.0], width 0.75. The two overlap for 0.75 of the bin. The
    /// exact-value assertions below fail on both counts.
    #[test]
    fn dodge_continuous_span_subdivides_the_bin_into_equal_slots() {
        // Two bins ([0,1), [1,2)) x two groups, the shape a shared-extent
        // dodged histogram produces.
        let schema = Arc::new(Schema::new(vec![
            Field::new("bin_start", DataType::Float64, false),
            Field::new("bin_end", DataType::Float64, false),
            Field::new("count", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        let b = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![0.0, 0.0, 1.0, 1.0])),
                Arc::new(Float64Array::from(vec![1.0, 1.0, 2.0, 2.0])),
                Arc::new(Float64Array::from(vec![3.0, 5.0, 7.0, 2.0])),
                Arc::new(StringArray::from(vec!["a", "b", "a", "b"])),
            ],
        )
        .unwrap();
        let enc = Encoding {
            x: Some(EncodingSpec { field: "bin_start".into(), type_: None, ..Default::default() }),
            x2: Some(EncodingSpec { field: "bin_end".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "count".into(), type_: None, ..Default::default() }),
            color: Some(EncodingSpec { field: "g".into(), type_: None, ..Default::default() }),
            ..Default::default()
        };
        let s = dummy_scales();
        let pos = PositionAdjust::Dodge { by: Some("g".into()), padding: 0.0 };
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut Vec::new()).unwrap();

        let lo = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        let hi = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        let span = |i: usize| (lo.value(i), hi.value(i));
        // padding 0, 2 groups -> each slot is half its bin, laid end to end.
        let expected = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)];
        for (i, (elo, ehi)) in expected.iter().enumerate() {
            let (glo, ghi) = span(i);
            assert!(
                (glo - elo).abs() < 1e-9 && (ghi - ehi).abs() < 1e-9,
                "row {i} slot: expected [{elo}, {ehi}], got [{glo}, {ghi}]"
            );
        }

        // The three properties, stated as properties rather than as the
        // coordinates above: equal widths, no overlap, one shared step.
        let widths: Vec<f64> = (0..4).map(|i| hi.value(i) - lo.value(i)).collect();
        assert!(
            widths.iter().all(|w| (w - widths[0]).abs() < 1e-9),
            "every group's slot must be the same width; got {widths:?}"
        );
        assert!(hi.value(0) <= lo.value(1) + 1e-9, "bin0's two slots must not overlap");
        assert!(hi.value(2) <= lo.value(3) + 1e-9, "bin1's two slots must not overlap");
        // Group a's step from bin0 to bin1 is the bin width — the shared grid
        // the Python half established, which this fix must not move.
        assert!((lo.value(2) - lo.value(0) - 1.0).abs() < 1e-9, "group a's bin step must stay 1.0");
        assert!((lo.value(3) - lo.value(1) - 1.0).abs() < 1e-9, "group b's bin step must stay 1.0");
    }

    /// Dodge's own `padding` is taken off each END of the bin and the slots are
    /// laid end to end from its leading edge — `apply_dodge_ordinal`'s
    /// convention, not a second one invented for the continuous branch.
    #[test]
    fn dodge_continuous_span_padding_insets_both_ends_of_the_bin() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("bin_start", DataType::Float64, false),
            Field::new("bin_end", DataType::Float64, false),
            Field::new("count", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        let b = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![0.0, 0.0])),
                Arc::new(Float64Array::from(vec![1.0, 1.0])),
                Arc::new(Float64Array::from(vec![3.0, 5.0])),
                Arc::new(StringArray::from(vec!["a", "b"])),
            ],
        )
        .unwrap();
        let enc = Encoding {
            x: Some(EncodingSpec { field: "bin_start".into(), type_: None, ..Default::default() }),
            x2: Some(EncodingSpec { field: "bin_end".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "count".into(), type_: None, ..Default::default() }),
            color: Some(EncodingSpec { field: "g".into(), type_: None, ..Default::default() }),
            ..Default::default()
        };
        let s = dummy_scales();
        // padding 0.1 -> 0.1 off each end of a 1.0 bin, two 0.4 slots between.
        let pos = PositionAdjust::Dodge { by: Some("g".into()), padding: 0.1 };
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut Vec::new()).unwrap();
        let lo = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        let hi = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        for (i, (elo, ehi)) in [(0.1, 0.5), (0.5, 0.9)].iter().enumerate() {
            assert!(
                (lo.value(i) - elo).abs() < 1e-9 && (hi.value(i) - ehi).abs() < 1e-9,
                "row {i} slot: expected [{elo}, {ehi}], got [{}, {}]",
                lo.value(i),
                hi.value(i)
            );
        }
    }

    /// A row with no span to subdivide passes through untouched on BOTH
    /// columns — nulls included. Without the null preservation, rebuilding the
    /// span column from a plain `Vec<f64>` would turn a null endpoint into a
    /// non-null `0.0`, and a row the drawer skips today would start drawing a
    /// rect back to the axis origin.
    #[test]
    fn dodge_continuous_span_leaves_null_and_degenerate_rows_alone() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("bin_start", DataType::Float64, true),
            Field::new("bin_end", DataType::Float64, true),
            Field::new("count", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        let b = RecordBatch::try_new(
            schema,
            vec![
                // row0: null start; row1: null end; row2: zero-width bin;
                // row3: a well-formed bin, so the dodge is not a no-op.
                Arc::new(Float64Array::from(vec![None, Some(0.0), Some(2.0), Some(4.0)])),
                Arc::new(Float64Array::from(vec![Some(1.0), None, Some(2.0), Some(5.0)])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
                Arc::new(StringArray::from(vec!["a", "b", "a", "b"])),
            ],
        )
        .unwrap();
        let enc = Encoding {
            x: Some(EncodingSpec { field: "bin_start".into(), type_: None, ..Default::default() }),
            x2: Some(EncodingSpec { field: "bin_end".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "count".into(), type_: None, ..Default::default() }),
            color: Some(EncodingSpec { field: "g".into(), type_: None, ..Default::default() }),
            ..Default::default()
        };
        let s = dummy_scales();
        let pos = PositionAdjust::Dodge { by: Some("g".into()), padding: 0.0 };
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut Vec::new()).unwrap();
        let lo = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        let hi = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();

        assert!(lo.is_null(0) && (hi.value(0) - 1.0).abs() < 1e-9, "null start row must pass through");
        assert!(hi.is_null(1) && (lo.value(1) - 0.0).abs() < 1e-9, "null end row must pass through");
        assert!(
            (lo.value(2) - 2.0).abs() < 1e-9 && (hi.value(2) - 2.0).abs() < 1e-9,
            "a zero-width bin has no interval to subdivide and must pass through"
        );
        // The well-formed row still dodges: group b (slot 1) takes the upper
        // half of [4, 5].
        assert!(
            (lo.value(3) - 4.5).abs() < 1e-9 && (hi.value(3) - 5.0).abs() < 1e-9,
            "the well-formed row must still take its slot; got [{}, {}]",
            lo.value(3),
            hi.value(3)
        );
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
            DiscreteLayout::UNPADDED,
        ));
        let lin = ScaleKind::Linear(LinearScale::new_internal(
            vec![0.0, 100.0],
            vec![0.0, 100.0],
            false,
            false,
        ));
        let (x, y) = if ordinal_on_x { (ord, lin) } else { (lin, ord) };
        ResolvedScales { x, y, color: None, size: None, shape: None, opacity: None, fill_opacity: None, stroke_opacity: None, stroke_dash: None, stroke_width: None, x2: None, y2: None, y_slots: Default::default(), raster_cells: None }
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
            DiscreteLayout::UNPADDED,
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
            fill_opacity: None,
            stroke_opacity: None,
            stroke_dash: None,
            stroke_width: None,
            x2: None,
            y2: None,
            y_slots: Default::default(),
            raster_cells: None,
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

    /// A layer that draws no inherited color (a text) holds it as its
    /// `group` key: with no `by` and no color it dodges by that key, and its
    /// slots follow the color domain (legend order) as the colored bars
    /// beside it do, not the rows' encounter order.
    #[test]
    fn dodge_by_the_group_key_follows_the_color_domain() {
        let b = batch_cat_val_model_sorted();
        let mut enc = enc_xy("cat", "val", None);
        enc.group = Some(EncodingSpec { field: "model".into(), ..Default::default() });
        let s = scales_ordinal_x_with_color(&["base", "alt"]);
        let pos = PositionAdjust::Dodge { by: None, padding: 0.0 };
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

    /// A facet field the panel split cannot read is refused with the panel
    /// split's error, not stacked across panels as one stack.
    #[test]
    fn stack_preview_refuses_a_facet_field_the_panel_split_cannot_read() {
        use arrow::array::Int64Array;
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("x", DataType::Utf8, false),
                Field::new("y", DataType::Float64, false),
                Field::new("g", DataType::Utf8, false),
                Field::new("h", DataType::Int64, false),
            ])),
            vec![
                Arc::new(StringArray::from(vec!["a", "a"])),
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
                Arc::new(StringArray::from(vec!["g1", "g2"])),
                Arc::new(Int64Array::from(vec![1, 2])),
            ],
        )
        .unwrap();
        let facet: crate::layout::facet::FacetSpec =
            serde_json::from_value(serde_json::json!({"field": "h", "mode": {"kind": "wrap", "ncols": 2}})).unwrap();
        let stack = PositionAdjust::Stack { by: None, offset: StackOffset::Zero, anchor: StackAnchor::Top, value_axis: None };
        let preview = stack_preview(&batch, Some(&stack), &enc_xy("x", "y", Some("g")), false, StackValueAxis::Y, Some(&facet));
        assert_eq!(
            preview.err().map(|e| e.to_string()),
            Some(crate::render::RenderError::ScaleResolutionFailed("facet field 'h' must be Utf8".into()).to_string()),
        );
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
            mark: Mark::Bar,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: Some(EncodingSpec { field: "g".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            position: Some(PositionAdjust::Stack {
                by: Some("g".into()),
                offset: StackOffset::Zero,
                anchor: StackAnchor::Top,
                value_axis: None,
            }),
            ..Default::default()
        };

        let no_outputs = HashMap::new();
        let result = axis_batch(&spec, &spec.encoding, StackValueAxis::Y, "y", &batch, &no_outputs, false).unwrap().batch;
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
            None,
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
            mark: Mark::Bar,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "count".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "bin_start".into(), type_: None, ..Default::default() }),
                color: Some(EncodingSpec { field: "grp".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            position: Some(PositionAdjust::Stack {
                by: Some("grp".into()),
                offset: StackOffset::Zero,
                anchor: StackAnchor::Top,
                value_axis: Some(StackValueAxis::X),
            }),
            ..Default::default()
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
            mark: crate::spec::mark::Mark::Bar,
            encoding: enc,
            position: Some(PositionAdjust::Stack {
                by: Some("g".into()),
                offset: StackOffset::Zero,
                anchor: StackAnchor::Top,
                value_axis: None,
            }),
            ..Default::default()
        };

        let no_outputs = HashMap::new();
        let x_result = axis_batch(&spec, &spec.encoding, StackValueAxis::X, "x", &b, &no_outputs, false).unwrap().batch;
        assert!(matches!(x_result, Cow::Borrowed(_)), "x-side must be a no-op for a vertical stack");

        let y_result = axis_batch(&spec, &spec.encoding, StackValueAxis::Y, "y", &b, &no_outputs, false).unwrap().batch;
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
            mark: crate::spec::mark::Mark::Bar,
            encoding: enc,
            coord: Some(crate::spec::coord::CoordKind::Flip),
            position: Some(PositionAdjust::Stack {
                by: Some("grp".into()),
                offset: StackOffset::Zero,
                anchor: StackAnchor::Top,
                value_axis: None,
            }),
            ..Default::default()
        };

        let no_outputs = HashMap::new();
        let x_result = axis_batch(&spec, &spec.encoding, StackValueAxis::X, "val", &batch, &no_outputs, false).unwrap().batch;
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

        let y_result = axis_batch(&spec, &spec.encoding, StackValueAxis::Y, "cat", &batch, &no_outputs, false).unwrap().batch;
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
            mark: crate::spec::mark::Mark::Bar,
            encoding: enc,
            coord: Some(crate::spec::coord::CoordKind::Flip),
            position: Some(PositionAdjust::Stack {
                by: Some("grp".into()),
                offset: StackOffset::Zero,
                anchor: StackAnchor::Top,
                value_axis: None,
            }),
            ..Default::default()
        };

        let no_outputs = HashMap::new();
        let y_result = axis_batch(&spec, &spec.encoding, StackValueAxis::Y, "cat", &batch, &no_outputs, false).unwrap().batch;
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

    /// The whole triad, as a table (close review F3). `value_axis` is the
    /// AUTHORED declaration and `coord_flipped` is a transformation applied
    /// after it, so the rendered answer is the XOR — not the precedence its
    /// post-flip sibling `DrawCtx::series_curve_axis` correctly uses on its own
    /// already-swapped copy.
    ///
    /// The two `Some(_)` + flip cells are the ones that changed. Every other cell
    /// is asserted here too, because they are what keeps stacked output
    /// byte-identical and a "fix" that moved them would be a regression nothing
    /// else in this module pins at this granularity.
    #[test]
    fn a_flip_moves_an_authored_stack_value_axis_and_leaves_the_rest_alone() {
        use crate::spec::position::StackValueAxis::{X, Y};
        // Unflipped: the declaration is already the rendered answer.
        assert!(!resolve_stack_value_on_x(None, false), "undeclared, unflipped → value on y");
        assert!(resolve_stack_value_on_x(Some(X), false), "declared x, unflipped → value on x");
        assert!(!resolve_stack_value_on_x(Some(Y), false), "declared y, unflipped → value on y");
        // Flipped: every authored axis moves to the other one.
        assert!(resolve_stack_value_on_x(None, true), "undeclared + flip → the default y moves to x");
        assert!(
            !resolve_stack_value_on_x(Some(X), true),
            "declared x + flip → the flip moves it BACK to y; answering `true` here is the F3 \
             defect (a horizontal stacked histogram under CoordFlip cumulating its bin column)"
        );
        assert!(
            resolve_stack_value_on_x(Some(Y), true),
            "declared y + flip → the mirror of the same defect"
        );
    }

    /// The F3 defect at the cumulation seam: an authored horizontal stack
    /// (`value_axis: X`, what `desugar_histogram(orientation="horizontal")`
    /// stamps) under a real `CoordFlip` must cumulate the POST-flip y column —
    /// the value — and group by the post-flip x — the bins.
    ///
    /// The fixture is the measured shape reduced to four rows: post-flip
    /// `x = bin_start` (the category), `y = count` (the value). Pre-fix the triad
    /// answered "value on x" and the two roles swapped, so `bin_start` was
    /// cumulated and `count` became the bin key — which is exactly what the
    /// nine-rect render showed.
    #[test]
    fn an_authored_horizontal_stack_under_a_flip_cumulates_the_post_flip_value() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("bin_start", DataType::Float64, false),
            Field::new("count", DataType::Float64, false),
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
        // Post-flip encoding, as `prepare::build_layers` hands it to the draw
        // pass: the authored y (bins) is on x, the authored x (counts) on y.
        let enc = enc_xy("bin_start", "count", Some("grp"));
        let pos = PositionAdjust::Stack {
            by: Some("grp".into()),
            offset: StackOffset::Zero,
            anchor: StackAnchor::Top,
            value_axis: Some(StackValueAxis::X),
        };
        let out =
            apply_position(&batch, Some(&pos), &dummy_scales(), &enc, true, &mut Vec::new()).unwrap();

        let counts = out.column(out.schema().index_of("count").unwrap());
        let counts = counts.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(
            (counts.value(0), counts.value(1), counts.value(2), counts.value(3)),
            (3.0, 8.0, 2.0, 6.0),
            "the COUNT column cumulates within each bin"
        );
        let bins = out.column(out.schema().index_of("bin_start").unwrap());
        let bins = bins.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(
            (bins.value(0), bins.value(1), bins.value(2), bins.value(3)),
            (0.0, 0.0, 1.0, 1.0),
            "…and the bin column is left exactly as it arrived — pre-fix it was the one cumulated"
        );
        assert!(
            !BatchPositionMeta::from_batch(&out).stack_value_on_x(),
            "the value landed on y as rendered, so the horizontal-value marker must NOT be stamped"
        );
    }

    /// Input schema metadata survives a stack, on both paths (close review F5).
    ///
    /// `apply_stack` built the output schema twice — once with the input metadata
    /// cloned (the `value_on_x` path) and once bare — so the common vertical path
    /// dropped whatever the input carried. Inert today, since no producer can
    /// chain into a stack, and asserted here so the two paths cannot drift apart
    /// again: `apply_jitter` and `apply_dodge` both carry it forward already.
    #[test]
    fn a_stack_carries_input_schema_metadata_forward_on_both_paths() {
        let fields = vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
        ];
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("__probe__".to_string(), "kept".to_string());
        let schema = Arc::new(Schema::new(fields).with_metadata(metadata));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 1.0, 2.0, 2.0])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0])),
                Arc::new(StringArray::from(vec!["a", "b", "a", "b"])),
            ],
        )
        .unwrap();
        let enc = enc_xy("x", "y", Some("g"));
        let pos = PositionAdjust::Stack {
            by: Some("g".into()),
            offset: StackOffset::Zero,
            anchor: StackAnchor::Top,
            value_axis: None,
        };
        for coord_flipped in [false, true] {
            let out =
                apply_position(&batch, Some(&pos), &dummy_scales(), &enc, coord_flipped, &mut Vec::new())
                    .unwrap();
            assert_eq!(
                out.schema().metadata().get("__probe__").map(String::as_str),
                Some("kept"),
                "coord_flipped={coord_flipped}: input metadata must survive the stack"
            );
        }
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

    /// R3: the Stack dtype-check messages ("{channel} must be a numeric
    /// dtype...", "{channel} must be a dtype that can key categories...") swap
    /// consistently with the value/category role under flip too — not just the
    /// missing-encoding pair. Both wordings moved with #184 (the value half onto
    /// `col_as_f64`'s set, the category half onto the band ladder's); the
    /// property this test pins — which channel token the message names — did
    /// not.
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
        // The load-bearing part of this assertion is the channel token `y` —
        // the post-flip x column's bad dtype must be reported as the user's own
        // y. The accepted-set clause after it moved with #184 (stack now reads
        // through `col_as_f64`, so the message names that helper's set instead
        // of the retired downcast chain's six dtypes); it is matched here only
        // so the prefix pins the whole opening of the message rather than the
        // token alone.
        assert!(
            text.starts_with("Stack: y must be a numeric dtype"),
            "post-flip x's bad dtype must be reported as the user's own y; got: {text}"
        );
    }

    /// R3 (#89 part C restructure): the Stack CATEGORY-dtype check also swaps
    /// with the value/category role under flip — sibling coverage to
    /// `stack_dtype_error_messages_name_users_channel_under_flip` above, which
    /// only pins the VALUE-dtype message. `PositionAdjustReason::CategoryDtype`
    /// was not previously exercised by any test.
    ///
    /// #184's category half RE-REASONED both category blocks rather than
    /// re-baselining them. The property under test is unchanged — a category
    /// column the stack cannot key refuses, naming the channel the USER wrote —
    /// but two things about it moved. The bad column is now `Duration`, not
    /// `Boolean`: `Boolean` was only ever refused because the retired two-arm
    /// gate admitted `Float64` and `Utf8` and nothing else, and a `Boolean`
    /// category column dodges, jitters and bands, so refusing it here was the
    /// defect. `Duration` is what is left once the stack keys through
    /// `col_as_identity_key_str` — every category-keyable dtype plus `Timestamp`
    /// by its epoch — and it is the honest boundary rather than a live user
    /// shape (`_coerce.normalize_for_rust` pre-casts `Duration`/`Date*` before
    /// they could arrive). And the text now names the accepted set and the
    /// offending dtype, the shape its VALUE-dtype sibling already had.
    ///
    /// Also pins the VALUE-dtype message's FULL rendered string (quality-review
    /// cycle 2 required fix): `ValueDtype`'s literal is a `\`-continuation join
    /// in `render/mod.rs`'s `Display` impl, and
    /// `stack_dtype_error_messages_name_users_channel_under_flip`'s
    /// `starts_with` stops before that join, leaving the tail — the exact part a
    /// lost or doubled space in the continuation would corrupt — otherwise
    /// unpinned. The two `assert_eq!` blocks at the end of this fn (unflipped,
    /// then flipped) close that gap.
    ///
    /// #184 re-reasoned the expected text, it did not re-baseline it: stack's
    /// value column moved from a hand-rolled six-arm downcast chain onto
    /// `arrow_cast::col_as_f64`, so the accepted set the message enumerates is
    /// now that helper's (`supported_numeric_dtype`) and the old wording
    /// understated it. The join is now two continuations rather than one, which
    /// is more surface for exactly the defect these two blocks exist to catch,
    /// not less.
    #[test]
    fn stack_category_dtype_error_message_names_users_channel_under_flip() {
        use arrow::array::{BooleanArray, DurationMillisecondArray};
        let dur = || DataType::Duration(arrow::datatypes::TimeUnit::Millisecond);
        let dur_col = || Arc::new(DurationMillisecondArray::from(vec![1_i64, 2])) as ArrayRef;
        const CATEGORY_REFUSAL: &str = "must be a dtype that can identify a row \
             (Utf8/LargeUtf8, Int8/Int16/Int32/Int64, UInt8/UInt16/UInt32/UInt64, \
             Float32/Float64, Boolean, or Timestamp); got Duration(Millisecond)";

        // Unflipped: the resolved category slot IS the user's own x — Duration
        // "x" is directly the bad column.
        let schema_unflipped = Arc::new(Schema::new(vec![
            Field::new("x", dur(), false),
            Field::new("y", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        let batch_unflipped = RecordBatch::try_new(
            schema_unflipped,
            vec![
                dur_col(),
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
        assert_eq!(format!("{err}"), format!("Stack: x {CATEGORY_REFUSAL}"));

        // Flipped: the resolved category slot is post-flip y — Duration "y" is
        // the bad column, but it must still be reported as the user's own x
        // (the same message text as the unflipped case above).
        let schema_flipped = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", dur(), false),
            Field::new("g", DataType::Utf8, false),
        ]));
        let batch_flipped = RecordBatch::try_new(
            schema_flipped,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
                dur_col(),
                Arc::new(StringArray::from(vec!["a", "b"])),
            ],
        )
        .unwrap();
        let err = apply_position(&batch_flipped, Some(&pos), &s, &enc, true, &mut Vec::new()).unwrap_err();
        assert_eq!(
            format!("{err}"),
            format!("Stack: x {CATEGORY_REFUSAL}"),
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
            "Stack: y must be a numeric dtype (Float64/Float32, Int8/Int16/Int32/Int64, \
             UInt8/UInt16/UInt32/UInt64, or Timestamp); got Boolean",
            "unflipped: exact VALUE-dtype message, naming the set `col_as_f64` accepts"
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
            "Stack: y must be a numeric dtype (Float64/Float32, Int8/Int16/Int32/Int64, \
             UInt8/UInt16/UInt32/UInt64, or Timestamp); got Boolean",
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

    // ---- One numeric-read policy for all three adjustments (#184 cycle 2) ---
    //
    // `apply_stack` kept a bespoke six-arm downcast chain (Float64, UInt64,
    // Int64/32/16/8) while `apply_dodge` and `apply_jitter` moved onto
    // `arrow_cast::col_as_f64`, so after the first pass at #184 a `Float32` /
    // `UInt8..32` / `Timestamp` value column DODGED and JITTERED but RAISED on
    // stack: three adjustments in one module disagreeing about which dtypes are
    // numeric. `col_as_f64` is the render layer's single source of truth for
    // that question (its doc delegates to `supported_numeric_dtype`), so stack
    // reads through it too and the disagreement is gone.

    /// The value-column dtypes `arrow_cast::col_as_f64` accepts, each carrying
    /// the same 10/20/30/40 measures. `Timestamp` is deliberately included: it
    /// is in `supported_numeric_dtype`, so it is part of the set the three
    /// adjustments must agree on, however unusual stacking one is.
    fn numeric_value_matrix() -> Vec<(&'static str, Field, ArrayRef)> {
        use arrow::array::{
            Float32Array, Int16Array, Int32Array, Int64Array, Int8Array, TimestampMillisecondArray,
            UInt16Array, UInt32Array, UInt64Array, UInt8Array,
        };
        let ts = DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None);
        vec![
            ("Float64", DataType::Float64, Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0])) as ArrayRef),
            ("Float32", DataType::Float32, Arc::new(Float32Array::from(vec![10.0_f32, 20.0, 30.0, 40.0]))),
            ("Int64", DataType::Int64, Arc::new(Int64Array::from(vec![10_i64, 20, 30, 40]))),
            ("Int32", DataType::Int32, Arc::new(Int32Array::from(vec![10_i32, 20, 30, 40]))),
            ("Int16", DataType::Int16, Arc::new(Int16Array::from(vec![10_i16, 20, 30, 40]))),
            ("Int8", DataType::Int8, Arc::new(Int8Array::from(vec![10_i8, 20, 30, 40]))),
            ("UInt64", DataType::UInt64, Arc::new(UInt64Array::from(vec![10_u64, 20, 30, 40]))),
            ("UInt32", DataType::UInt32, Arc::new(UInt32Array::from(vec![10_u32, 20, 30, 40]))),
            ("UInt16", DataType::UInt16, Arc::new(UInt16Array::from(vec![10_u16, 20, 30, 40]))),
            ("UInt8", DataType::UInt8, Arc::new(UInt8Array::from(vec![10_u8, 20, 30, 40]))),
            ("Timestamp(ms)", ts, Arc::new(TimestampMillisecondArray::from(vec![10_i64, 20, 30, 40]))),
        ]
        .into_iter()
        .map(|(name, dt, col)| (name, Field::new("y", dt, false), col))
        .collect()
    }

    /// Every dtype `col_as_f64` reads stacks to the same cumulative tops the
    /// `Float64` column produces, and comes back retyped to `Float64`.
    ///
    /// RED against the bespoke chain: `Float32`, `UInt32`, `UInt16`, `UInt8` and
    /// `Timestamp(ms)` fell through all six arms and raised
    /// `PositionAdjustReason::ValueDtype`, so `expect` panics on the first of
    /// them — while the identical columns dodged and jittered fine.
    #[test]
    fn stack_accepts_every_numeric_dtype_its_siblings_accept() {
        for (name, field, col) in numeric_value_matrix() {
            let schema = Arc::new(Schema::new(vec![
                Field::new("x", DataType::Float64, false),
                field,
                Field::new("g", DataType::Utf8, false),
            ]));
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Float64Array::from(vec![1.0, 1.0, 2.0, 2.0])),
                    col,
                    Arc::new(StringArray::from(vec!["a", "b", "a", "b"])),
                ],
            )
            .unwrap();
            let pos = PositionAdjust::Stack {
                by: Some("g".into()),
                offset: StackOffset::Zero,
                anchor: StackAnchor::Top,
                value_axis: None,
            };
            let out = apply_position(
                &batch,
                Some(&pos),
                &dummy_scales(),
                &enc_xy("x", "y", Some("g")),
                false,
                &mut Vec::new(),
            )
            .unwrap_or_else(|e| panic!("{name} must stack, as it already dodges and jitters: {e}"));
            assert_eq!(
                out.schema().field(1).data_type(),
                &DataType::Float64,
                "{name}: the stacked value column must be retyped to Float64"
            );
            let ya = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
            assert_eq!(
                (0..4).map(|i| ya.value(i)).collect::<Vec<_>>(),
                vec![10.0, 30.0, 30.0, 70.0],
                "{name}: cumulative tops must match the Float64 oracle"
            );
        }
    }

    /// The same matrix, through all THREE adjustments, which is the property
    /// that actually has to hold: one module cannot have two answers to "is this
    /// dtype numeric". Asserts acceptance, not per-adjustment geometry — the
    /// geometry oracles live in the focused tests above and beside them.
    #[test]
    fn stack_dodge_and_jitter_accept_the_same_value_dtypes() {
        for (name, field, col) in numeric_value_matrix() {
            let schema = Arc::new(Schema::new(vec![
                Field::new("x", DataType::Float64, false),
                field,
                Field::new("g", DataType::Utf8, false),
            ]));
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Float64Array::from(vec![1.0, 1.0, 2.0, 2.0])),
                    col,
                    Arc::new(StringArray::from(vec!["a", "b", "a", "b"])),
                ],
            )
            .unwrap();
            let enc = enc_xy("x", "y", Some("g"));
            for (adjustment, pos) in [
                (
                    "Stack",
                    PositionAdjust::Stack {
                        by: Some("g".into()),
                        offset: StackOffset::Zero,
                        anchor: StackAnchor::Top,
                        value_axis: None,
                    },
                ),
                ("Dodge", PositionAdjust::Dodge { by: Some("g".into()), padding: 0.0 }),
                (
                    "Jitter",
                    PositionAdjust::Jitter { axis: JitterAxis::Y, width: 0.4, seed: Some(3) },
                ),
            ] {
                let mut warnings = Vec::new();
                let out =
                    apply_position(&batch, Some(&pos), &dummy_scales(), &enc, false, &mut warnings);
                assert!(
                    out.is_ok(),
                    "{adjustment} must accept a {name} value column, like its two siblings: {:?}",
                    out.err()
                );
                assert!(
                    warnings.is_empty(),
                    "{adjustment} on a {name} value column must not warn: {warnings:?}"
                );
            }
        }
    }

    /// A non-numeric value column still refuses, and the refusal still names the
    /// accepted set — which after the migration is the set `col_as_f64` really
    /// reads, not the six dtypes the hand-rolled chain happened to list.
    #[test]
    fn stack_still_refuses_a_non_numeric_value_column_naming_the_accepted_set() {
        use arrow::array::BooleanArray;
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Boolean, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
                Arc::new(BooleanArray::from(vec![true, false])),
                Arc::new(StringArray::from(vec!["a", "b"])),
            ],
        )
        .unwrap();
        let pos = PositionAdjust::Stack {
            by: Some("g".into()),
            offset: StackOffset::Zero,
            anchor: StackAnchor::Top,
            value_axis: None,
        };
        let err = apply_position(
            &batch,
            Some(&pos),
            &dummy_scales(),
            &enc_xy("x", "y", Some("g")),
            false,
            &mut Vec::new(),
        )
        .expect_err("Boolean is not a numeric dtype and must still refuse");
        let msg = format!("{err}");
        assert!(msg.contains("Boolean"), "must name the offending dtype: {msg}");
        for named in ["Float32", "UInt8", "Timestamp"] {
            assert!(
                msg.contains(named),
                "the refusal must name {named} among the accepted dtypes, since stack now \
                 accepts it: {msg}"
            );
        }
    }

    /// Fix 2(a): stack's value field is retyped through `set_float64_column`
    /// like dodge's and jitter's, so it keeps the input field's nullability and
    /// metadata instead of hand-building `Field::new(name, Float64, true)`.
    ///
    /// RED against the hand-built field: `is_nullable()` came back `true` for a
    /// non-nullable input and the metadata map came back empty.
    #[test]
    fn stack_value_field_keeps_its_nullability_and_metadata() {
        use arrow::array::Int64Array;
        let y_field = Field::new("y", DataType::Int64, false)
            .with_metadata([("unit".to_string(), "count".to_string())].into());
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            y_field,
            Field::new("g", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 1.0])),
                Arc::new(Int64Array::from(vec![10_i64, 20])),
                Arc::new(StringArray::from(vec!["a", "b"])),
            ],
        )
        .unwrap();
        let pos = PositionAdjust::Stack {
            by: Some("g".into()),
            offset: StackOffset::Zero,
            anchor: StackAnchor::Top,
            value_axis: None,
        };
        let out = apply_position(
            &batch,
            Some(&pos),
            &dummy_scales(),
            &enc_xy("x", "y", Some("g")),
            false,
            &mut Vec::new(),
        )
        .unwrap();
        let f = out.schema().field(1).clone();
        assert_eq!(f.data_type(), &DataType::Float64, "still promoted to Float64");
        assert!(!f.is_nullable(), "a non-nullable input field must not be forced nullable");
        assert_eq!(
            f.metadata().get("unit").map(String::as_str),
            Some("count"),
            "field metadata must survive the retype, as it does for dodge and jitter"
        );
    }

    // -----------------------------------------------------------------------
    // #184, category half: the stack bins on the band ladder, not on a dtype.
    //
    // `apply_stack`'s x_keys chain was a second dtype gate, independent of the
    // value-column one above: `Float64` keyed on `f64::to_bits`, `Utf8` on a
    // hash of the bytes, and every other dtype was a typed `CategoryDtype`
    // refusal — so an `Int64` column declared `:N` raised "x column must be
    // Float64 or Utf8" while the identical column dodged, jittered and banded.
    // It now keys through `arrow_cast::col_as_positional_category_str`, which is
    // the reader the ordinal band ladder (`distinct_positional_categories`) is
    // enumerated with, so the bins and the bands agree on which value is which.
    // (Not on WHICH bands exist — see the `#247`B note in `apply_stack` for why
    // an unoccupied null band the ladder omits still gets a row bin here.)
    // -----------------------------------------------------------------------

    /// One 4-row category column per dtype the band ladder can enumerate, each
    /// spelling the SAME two categories as the `Utf8` row spells (`"1"`/`"2"`,
    /// and `"false"`/`"true"` for `Boolean` — whatever
    /// `col_as_ordinal_category_str` formats them as).
    ///
    /// Row layout is fixed across the matrix at `[c1, c1, c2, c2]` so every cell
    /// is judged against one oracle. `Boolean` is in it because it is the dtype
    /// the retired gate's refusal was *pinned* on; `LargeUtf8` because polars
    /// produces it for every string column, so it is the shape a real
    /// string-category stack arrives as.
    fn category_column_matrix() -> Vec<(&'static str, Field, ArrayRef)> {
        use arrow::array::{
            BooleanArray, Float32Array, Int16Array, Int32Array, Int64Array, Int8Array,
            LargeStringArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
        };
        vec![
            ("Utf8", DataType::Utf8, Arc::new(StringArray::from(vec!["1", "1", "2", "2"])) as ArrayRef),
            ("LargeUtf8", DataType::LargeUtf8, Arc::new(LargeStringArray::from(vec!["1", "1", "2", "2"]))),
            ("Int64", DataType::Int64, Arc::new(Int64Array::from(vec![1_i64, 1, 2, 2]))),
            ("Int32", DataType::Int32, Arc::new(Int32Array::from(vec![1_i32, 1, 2, 2]))),
            ("Int16", DataType::Int16, Arc::new(Int16Array::from(vec![1_i16, 1, 2, 2]))),
            ("Int8", DataType::Int8, Arc::new(Int8Array::from(vec![1_i8, 1, 2, 2]))),
            ("UInt64", DataType::UInt64, Arc::new(UInt64Array::from(vec![1_u64, 1, 2, 2]))),
            ("UInt32", DataType::UInt32, Arc::new(UInt32Array::from(vec![1_u32, 1, 2, 2]))),
            ("UInt16", DataType::UInt16, Arc::new(UInt16Array::from(vec![1_u16, 1, 2, 2]))),
            ("UInt8", DataType::UInt8, Arc::new(UInt8Array::from(vec![1_u8, 1, 2, 2]))),
            ("Float64", DataType::Float64, Arc::new(Float64Array::from(vec![1.0, 1.0, 2.0, 2.0]))),
            ("Float32", DataType::Float32, Arc::new(Float32Array::from(vec![1.0_f32, 1.0, 2.0, 2.0]))),
            ("Boolean", DataType::Boolean, Arc::new(BooleanArray::from(vec![false, false, true, true]))),
        ]
        .into_iter()
        .map(|(name, dt, col)| (name, Field::new("x", dt, true), col))
        .collect()
    }

    /// Stack a 2-category × 2-group batch whose category column is `(field,
    /// col)`, returning the cumulative tops and any warnings pushed.
    fn stack_by_category_column(
        field: Field,
        col: ArrayRef,
    ) -> Result<(Vec<f64>, Vec<RenderWarning>), crate::render::RenderError> {
        let schema = Arc::new(Schema::new(vec![
            field,
            Field::new("y", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                col,
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0])),
                Arc::new(StringArray::from(vec!["a", "b", "a", "b"])),
            ],
        )
        .unwrap();
        let pos = PositionAdjust::Stack {
            by: Some("g".into()),
            offset: StackOffset::Zero,
            anchor: StackAnchor::Top,
            value_axis: None,
        };
        let mut warnings = Vec::new();
        let out = apply_position(
            &batch,
            Some(&pos),
            &dummy_scales(),
            &enc_xy("x", "y", Some("g")),
            false,
            &mut warnings,
        )?;
        let ya = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        Ok(((0..4).map(|i| ya.value(i)).collect(), warnings))
    }

    /// Every category dtype the band ladder can enumerate cumulates to the same
    /// tops the `Utf8` column does — the stack's groups are the bands, not the
    /// storage.
    ///
    /// RED against the retired chain: only `Utf8` and `Float64` reached the
    /// cumulation at all; the other eleven cells raised
    /// `PositionAdjustReason::CategoryDtype`, so `expect` panics on `LargeUtf8`,
    /// the second cell — the dtype polars hands every string column over as.
    #[test]
    fn stack_keys_every_category_dtype_the_band_ladder_enumerates() {
        // Two categories, two groups, 10/20/30/40 measures: within category 1
        // group "a" takes 10 and group "b" tops out at 30; within category 2
        // group "a" takes 30 and group "b" tops out at 70.
        let oracle = vec![10.0, 30.0, 30.0, 70.0];
        for (name, field, col) in category_column_matrix() {
            let dtype = field.data_type().clone();
            let (tops, warnings) = stack_by_category_column(field, col)
                .unwrap_or_else(|e| panic!("{name} must stack, as it already dodges and bands: {e}"));
            assert_eq!(tops, oracle, "{name}: cumulative tops must match the Utf8 oracle");
            assert!(warnings.is_empty(), "{name} must stack without a warning: {warnings:?}");
            assert!(
                stack_accepts_dtype(&dtype),
                "{name}: the acceptance rule must predict the acceptance that just happened"
            );
        }
    }

    /// The relationship that holds between what `apply_stack` keys and what the
    /// band ladder enumerates: every dtype the ladder can enumerate stacks, AND
    /// `Timestamp` — which the ladder deliberately cannot enumerate — stacks
    /// too, by exact instant.
    ///
    /// This is deliberately NOT a call to `col_as_identity_key_str`, which is
    /// the function under test; it restates the contract in the terms the
    /// crate's own gate (`ensure_category_keyable`) and the documented temporal
    /// exception are written in, so a change to the reader's coverage has to
    /// disagree with the rule rather than move it along.
    ///
    /// An earlier version of this guard asserted `acceptance ==
    /// ensure_category_keyable`, which was too narrow rather than wrong in kind:
    /// it encoded the premise that a stack can only bin by a *band*. A stack
    /// over a continuous axis bins by exact value (an `Int32` x:Q stacks to its
    /// group sum through the same reader), and `Timestamp` is exactly the dtype
    /// whose exact value the category reader cannot spell — so equality with the
    /// band-keyable set would have kept every stacked time series refused.
    /// Widening it to "keyable or temporal" keeps both directions live: a
    /// narrowing back to `Float64`/`Utf8` fails on the matrix, and a widening to
    /// a dtype neither identity reader covers (`Duration`, `Date32`) fails on
    /// `stack_refuses_a_column_that_cannot_identify_a_row`.
    fn stack_accepts_dtype(dtype: &DataType) -> bool {
        crate::render::arrow_cast::ensure_category_keyable("x", dtype).is_ok()
            || matches!(dtype, DataType::Timestamp(_, _))
    }

    /// A stacked time series renders, and it stacks by the INSTANT: two rows
    /// share a bin exactly when they carry the same epoch value.
    ///
    /// RED before this landed: every `Timestamp` category column raised, because
    /// the category reader excludes temporal dtypes by documented contract (an
    /// epoch read through the `f64` widening is not injective past `2^53`) and
    /// nothing picked up the column after it declined. `mark_area(position=
    /// Stack())` over a date x — a mainstream chart — was unrenderable.
    ///
    /// The oracle is the same one the category matrix uses, which is the point:
    /// two instants behave exactly like two categories, so the cumulation is
    /// 10 → 30 in the first bin and 30 → 70 in the second.
    #[test]
    fn stack_keys_a_temporal_category_column_by_its_instant() {
        use arrow::array::TimestampMillisecondArray;
        use arrow::datatypes::TimeUnit;
        // Two instants, each carried by two rows (one per group). The second
        // pair is 2^53 + 1 ms apart from the first — a magnitude at which an
        // `f64` round-trip would collapse the two into one bin, which is why
        // this reads the raw `i64` rather than going through `col_as_f64`.
        let far_apart = 9_007_199_254_740_993_i64;
        let field = Field::new("x", DataType::Timestamp(TimeUnit::Millisecond, None), true);
        let col: ArrayRef = Arc::new(TimestampMillisecondArray::from(vec![
            1_i64, 1, far_apart, far_apart,
        ]));
        let (tops, warnings) = stack_by_category_column(field, col)
            .expect("a temporal category column must stack, not raise");
        assert_eq!(
            tops,
            vec![10.0, 30.0, 30.0, 70.0],
            "each instant must be its own bin, cumulating its two groups"
        );
        assert!(warnings.is_empty(), "a temporal stack must not warn: {warnings:?}");
        assert!(
            stack_accepts_dtype(&DataType::Timestamp(TimeUnit::Millisecond, None))
                && crate::render::arrow_cast::ensure_category_keyable(
                    "x",
                    &DataType::Timestamp(TimeUnit::Millisecond, None)
                )
                .is_err(),
            "Timestamp is the documented case where stack accepts what the band \
             ladder cannot enumerate — if that stops being true, the acceptance \
             rule above is no longer describing anything"
        );
    }

    /// The refusal is exactly "neither identity reader can key this", not a
    /// dtype list of the stack's own — and a column that genuinely cannot
    /// identify a row still refuses, rather than binning every row together or
    /// panicking inside the render pass.
    ///
    /// `Duration`/`Date32` are the honest boundary rather than a live user
    /// shape: `_coerce.normalize_for_rust` pre-casts both to `Timestamp(ms)`
    /// before any batch crosses the CDI boundary, so from Python this arm is
    /// unreachable — which is the point of testing it here, where the arm can
    /// be reached at all. `Timestamp` moved OUT of this test when the temporal
    /// reader picked it up; it is now covered by
    /// `stack_keys_a_temporal_category_column_by_its_instant`.
    #[test]
    fn stack_refuses_a_column_that_cannot_identify_a_row() {
        use arrow::array::{Date32Array, DurationMillisecondArray};
        use arrow::datatypes::TimeUnit;
        let cases: Vec<(&str, Field, ArrayRef)> = vec![
            (
                "Duration(ms)",
                Field::new("x", DataType::Duration(TimeUnit::Millisecond), true),
                Arc::new(DurationMillisecondArray::from(vec![1_i64, 1, 2, 2])),
            ),
            (
                "Date32",
                Field::new("x", DataType::Date32, true),
                Arc::new(Date32Array::from(vec![1_i32, 1, 2, 2])),
            ),
        ];
        for (name, field, col) in cases {
            let dtype = field.data_type().clone();
            let err = stack_by_category_column(field, col)
                .expect_err(&format!("{name} cannot identify a row, so it must refuse"));
            let text = format!("{err}");
            assert!(
                text.starts_with("Stack: x must be a dtype that can identify a row"),
                "{name}: the refusal must name the user's channel and the accepted set; got: {text}"
            );
            assert!(
                text.ends_with(&format!("got {dtype:?}")),
                "{name}: the refusal must name the offending dtype; got: {text}"
            );
            assert!(
                !stack_accepts_dtype(&dtype),
                "{name}: the acceptance rule must predict the refusal that just happened"
            );
        }
    }

    /// Two rows share a bin exactly when they share a BAND, which for a
    /// `Float64` category column is not the same question as sharing a bit
    /// pattern.
    ///
    /// `x = [0.0, -0.0, null, null]` is one band `"0"` (`float_as_ordinal_str`
    /// formats both signed zeros as `"0"`) plus the `NULL_CATEGORY` band the
    /// positional reader places a null row in — so `distinct_positional_categories`
    /// enumerates `["0", "null"]`, two bands, and the cumulation must be two
    /// bins: 10 → 30 in the first, 30 → 70 in the second.
    ///
    /// The premise below names `y` as the paired positional channel, which is
    /// what the axis resolver passes for an `x` ladder, so `#247`B's occupancy
    /// rule is exercised rather than sidestepped: both null-`x` rows carry a real
    /// `y` (30, 40), so the null band is occupied and stays. Drop the `y` values
    /// and the band goes — and those two rows would then draw nothing, which is
    /// why binning them separately here costs the other bin nothing.
    ///
    /// RED against the retired `to_bits` key, which answered a third thing
    /// again: `-0.0` keyed apart from `0.0` (one band, two bins — two segments
    /// stacked at one pixel), while both nulls keyed INTO the `0.0` bin, since
    /// `Float64Array::value` on a null returns `0.0`. Tops came out
    /// `[10, 20, 40, 80]`.
    #[test]
    fn stack_bins_by_band_not_by_float_bit_pattern() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("y", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        let x: ArrayRef = Arc::new(Float64Array::from(vec![Some(0.0), Some(-0.0), None, None]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::clone(&x),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0])),
                Arc::new(StringArray::from(vec!["a", "b", "a", "b"])),
            ],
        )
        .unwrap();
        assert_eq!(
            crate::render::arrow_cast::distinct_positional_categories(&batch, "x", Some("y"))
                .unwrap(),
            vec!["0".to_string(), "null".to_string()],
            "premise: the ladder this stack must agree with has two bands here"
        );

        let pos = PositionAdjust::Stack {
            by: Some("g".into()),
            offset: StackOffset::Zero,
            anchor: StackAnchor::Top,
            value_axis: None,
        };
        let out = apply_position(
            &batch,
            Some(&pos),
            &dummy_scales(),
            &enc_xy("x", "y", Some("g")),
            false,
            &mut Vec::new(),
        )
        .unwrap();
        let ya = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(
            (0..4).map(|i| ya.value(i)).collect::<Vec<_>>(),
            vec![10.0, 30.0, 30.0, 70.0],
            "the two signed zeros must cumulate together and the nulls must cumulate \
             in their own band, matching the two bands the ladder drew"
        );
        let base = out
            .column_by_name("__stack_y_base__")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(
            (0..4).map(|i| base.value(i)).collect::<Vec<_>>(),
            vec![0.0, 10.0, 0.0, 30.0],
            "each band's first segment must start at 0 — a segment whose base is the \
             other band's top is the mis-binning this pins"
        );
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
    // GH #184: Dodge and Jitter branch on the RESOLVED ScaleKind, never the
    // band column's Arrow dtype, and read positional values through
    // `arrow_cast::col_as_f64` instead of a bare `Float64Array` downcast.
    //
    // The three failure shapes the dtype gate produced, each pinned below:
    //   (a) `Float64` column + `:N` (Ordinal scale) took the CONTINUOUS branch,
    //       rewrote band values in data space, and the ordinal scale then found
    //       none of them in its domain — zero bars, no warning.
    //   (b) a non-`Float64` numeric column on a genuinely CONTINUOUS scale took
    //       `apply_dodge_ordinal`, which had no `OrdinalScale` and returned the
    //       batch unchanged — every dodge group overlapping, no warning.
    //   (c) `Jitter` on an `Int64`/`Int32` continuous axis wrote a `Float64`
    //       column into an integer schema slot and surfaced a raw Arrow schema
    //       error.
    // -----------------------------------------------------------------------

    /// `ResolvedScales` whose x is an Ordinal scale over the numeric-formatted
    /// domain a `Float64`/`Int64` category column produces (`"1"`/`"2"`, via
    /// `arrow_cast`'s `float_as_ordinal_str`), y Linear. Two bands over pixel
    /// range [0, 100] → `bandwidth() == 50`.
    fn scales_ordinal_x_numeric_domain() -> ResolvedScales {
        use crate::scale::linear::LinearScale;
        use crate::scale::ordinal::OrdinalScale;
        ResolvedScales {
            x: ScaleKind::Ordinal(OrdinalScale::new_internal(
                vec!["1".into(), "2".into()],
                vec![0.0, 100.0],
                DiscreteLayout::UNPADDED,
            )),
            y: ScaleKind::Linear(LinearScale::new_internal(
                vec![0.0, 100.0],
                vec![0.0, 100.0],
                false,
                false,
            )),
            color: None,
            size: None,
            shape: None,
            opacity: None,
            fill_opacity: None,
            stroke_opacity: None,
            stroke_dash: None,
            stroke_width: None,
            x2: None,
            y2: None,
            y_slots: Default::default(),
            raster_cells: None,
        }
    }

    /// `batch_xyg`'s shape with the x column in an arbitrary dtype: two band
    /// values (1, 2), y 10/20/30/40, and an alternating two-level Utf8 group.
    fn batch_xyg_with_x(x_field: Field, x_col: ArrayRef) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            x_field,
            Field::new("y", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                x_col,
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0])),
                Arc::new(StringArray::from(vec!["a", "b", "a", "b"])),
            ],
        )
        .unwrap()
    }

    /// The four x columns that must all dodge and jitter identically: the same
    /// band values 1, 1, 2, 2 in `Float64`, `Int64`, `Int32`, and `Float32`.
    fn x_column_dtype_matrix() -> Vec<(&'static str, Field, ArrayRef)> {
        use arrow::array::{Float32Array, Int32Array, Int64Array};
        vec![
            (
                "Float64",
                Field::new("x", DataType::Float64, false),
                Arc::new(Float64Array::from(vec![1.0, 1.0, 2.0, 2.0])) as ArrayRef,
            ),
            (
                "Int64",
                Field::new("x", DataType::Int64, false),
                Arc::new(Int64Array::from(vec![1_i64, 1, 2, 2])) as ArrayRef,
            ),
            (
                "Int32",
                Field::new("x", DataType::Int32, false),
                Arc::new(Int32Array::from(vec![1_i32, 1, 2, 2])) as ArrayRef,
            ),
            (
                "Float32",
                Field::new("x", DataType::Float32, false),
                Arc::new(Float32Array::from(vec![1.0_f32, 1.0, 2.0, 2.0])) as ArrayRef,
            ),
        ]
    }

    /// Shape (a). A `Float64` band column on an ORDINAL scale (`x="cat:N"` over
    /// integer-valued floats) must take the ordinal, pixel-space branch: the
    /// band column keeps its exact values — so the ordinal scale can still find
    /// `1`/`2` in its domain — and the sub-band offset arrives in
    /// `__pos_x_offset__`.
    ///
    /// RED against the dtype gate: `Float64` passed `!= DataType::Float64`, so
    /// this batch went down the continuous branch, which rewrote x to
    /// 0.75/1.25/1.75/2.25 and emitted NO offset columns. The ordinal scale then
    /// looked `"0.75"` up in `["1", "2"]`, missed, and every bar was dropped
    /// silently. Both assertions below fail on that: `offset_col` panics on the
    /// absent column, and the band values are not 1/1/2/2.
    #[test]
    fn dodge_float64_band_on_ordinal_scale_offsets_in_pixel_space() {
        let b = batch_xyg();
        let enc = enc_xy("x", "y", Some("g"));
        let s = scales_ordinal_x_numeric_domain();
        let pos = PositionAdjust::Dodge { by: Some("g".into()), padding: 0.0 };
        let mut warnings = Vec::new();
        let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut warnings).unwrap();
        assert!(warnings.is_empty(), "ordinal-scale dodge must not warn: {warnings:?}");

        // bandwidth 50, 2 groups, padding 0 → sub_band 25 → ∓12.5.
        assert_eq!(offset_col(&out, "__pos_x_offset__"), vec![-12.5, 12.5, -12.5, 12.5]);
        assert_eq!(offset_col(&out, "__pos_y_offset__"), vec![0.0, 0.0, 0.0, 0.0]);

        // The band column is untouched — the ordinal scale resolves categories
        // from it by string, so any data-space rewrite loses the whole layer.
        let xa = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(
            (0..4).map(|i| xa.value(i)).collect::<Vec<_>>(),
            vec![1.0, 1.0, 2.0, 2.0],
            "an ordinal band must not be rewritten in data space"
        );
    }

    /// Shape (a), mirrored: the same `Float64`-on-Ordinal batch must produce the
    /// byte-identical result its `Utf8` twin produces. The whole point of
    /// routing by scale kind is that the dtype stops being observable.
    #[test]
    fn dodge_float64_band_on_ordinal_matches_utf8_twin() {
        let numeric = apply_position(
            &batch_xyg(),
            Some(&PositionAdjust::Dodge { by: Some("g".into()), padding: 0.05 }),
            &scales_ordinal_x_numeric_domain(),
            &enc_xy("x", "y", Some("g")),
            false,
            &mut Vec::new(),
        )
        .unwrap();
        // `batch_cat_val_grp` is the same shape with a Utf8 band ("a"/"a"/"b"/"b")
        // and the group column named `grp`.
        let utf8 = apply_position(
            &batch_cat_val_grp(),
            Some(&PositionAdjust::Dodge { by: Some("grp".into()), padding: 0.05 }),
            &scales_one_ordinal(true),
            &enc_xy("cat", "val", Some("grp")),
            false,
            &mut Vec::new(),
        )
        .unwrap();
        assert_eq!(
            offset_col(&numeric, "__pos_x_offset__"),
            offset_col(&utf8, "__pos_x_offset__"),
            "a numeric ordinal band must dodge exactly like its Utf8 twin"
        );
        assert_eq!(
            numeric.schema().metadata(),
            utf8.schema().metadata(),
            "both must stamp the same dodge group count / sub-band width"
        );
    }

    /// Shape (b). A non-`Float64` numeric band on a genuinely CONTINUOUS scale
    /// must dodge in data space, to the same oracle offsets the `Float64` column
    /// produces, and the rewritten column must come back as `Float64` (an
    /// `Int64` slot cannot hold 0.75).
    ///
    /// RED against the dtype gate: `Int64 != Float64` routed this into
    /// `apply_dodge_ordinal`, whose `_ => return Ok(batch.clone())` arm no-op'd
    /// the dodge unconditionally against the `Linear` scale — x came back
    /// 1/1/2/2 with both groups fully overlapping, no warning. `assert_dodge_oracle`
    /// fails on the first row.
    #[test]
    fn dodge_non_float64_band_on_continuous_scale_actually_dodges() {
        for (name, field, col) in x_column_dtype_matrix() {
            let b = batch_xyg_with_x(field, col);
            let enc = enc_xy("x", "y", Some("g"));
            let s = dummy_scales(); // Linear x — a genuinely continuous band.
            let pos = PositionAdjust::Dodge { by: Some("g".into()), padding: 0.0 };
            let mut warnings = Vec::new();
            let out = apply_position(&b, Some(&pos), &s, &enc, false, &mut warnings).unwrap();
            assert!(warnings.is_empty(), "{name}: continuous dodge must not warn: {warnings:?}");
            assert_eq!(
                out.schema().field(0).data_type(),
                &DataType::Float64,
                "{name}: a data-space rewrite must retype the band field to Float64"
            );
            assert_dodge_oracle(&out);
            assert!(
                out.schema().index_of("__pos_x_offset__").is_err(),
                "{name}: a continuous band dodges in data space, not pixel space"
            );
        }
    }

    /// Shape (b) for a SPAN row: an `Int64` bin-edge pair (`x` + `x2`) on a
    /// continuous scale subdivides its own bin, exactly as the `Float64` pair
    /// does in `dodge_continuous_span_subdivides_the_bin_into_equal_slots`.
    ///
    /// RED twice over against the dtype gate: the band's `Int64` dtype sent the
    /// batch into `apply_dodge_ordinal`'s silent clone, and even reaching the
    /// continuous branch, the bare `Float64Array` downcast on `x2` failed, so
    /// the span was invisible and the row dodged as a point.
    #[test]
    fn dodge_int64_span_subdivides_the_bin_like_its_float64_twin() {
        use arrow::array::Int64Array;
        let schema = Arc::new(Schema::new(vec![
            Field::new("bin_start", DataType::Int64, false),
            Field::new("bin_end", DataType::Int64, false),
            Field::new("count", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        let b = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![0_i64, 0, 2, 2])),
                Arc::new(Int64Array::from(vec![2_i64, 2, 4, 4])),
                Arc::new(Float64Array::from(vec![3.0, 5.0, 7.0, 2.0])),
                Arc::new(StringArray::from(vec!["a", "b", "a", "b"])),
            ],
        )
        .unwrap();
        let enc = Encoding {
            x: Some(EncodingSpec { field: "bin_start".into(), type_: None, ..Default::default() }),
            x2: Some(EncodingSpec { field: "bin_end".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "count".into(), type_: None, ..Default::default() }),
            color: Some(EncodingSpec { field: "g".into(), type_: None, ..Default::default() }),
            ..Default::default()
        };
        let pos = PositionAdjust::Dodge { by: Some("g".into()), padding: 0.0 };
        let out = apply_position(&b, Some(&pos), &dummy_scales(), &enc, false, &mut Vec::new())
            .unwrap();

        let lo = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        let hi = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        // Bins [0,2) and [2,4), padding 0, two groups → half-bin slots laid end
        // to end from each bin's leading edge.
        let expected = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)];
        for (i, (elo, ehi)) in expected.iter().enumerate() {
            assert!(
                (lo.value(i) - elo).abs() < 1e-9 && (hi.value(i) - ehi).abs() < 1e-9,
                "row {i}: expected [{elo}, {ehi}], got [{}, {}]",
                lo.value(i),
                hi.value(i)
            );
        }
        assert_eq!(out.schema().field(1).data_type(), &DataType::Float64);
    }

    /// A band column with no numeric reading at all on a CONTINUOUS scale warns
    /// and no-ops — the same answer `resolve_group_channel` gives an
    /// un-categorizable grouping column, and not a refusal, because the shape is
    /// reachable (`x="cat:N"` with a numeric `CoordCartesian(xlim=…)` resolves
    /// the categorical axis continuous) and that chart already draws nothing
    /// before the dodge runs. The defect being fixed is the SILENCE.
    ///
    /// RED against the dtype gate: `Utf8 != Float64` reached
    /// `apply_dodge_ordinal`'s unconditional `Ok(batch.clone())` against the
    /// `Linear` scale and emitted ZERO warnings — the assertion on the warning
    /// list fails, while the unchanged-batch assertions pass either way (the
    /// no-op is the part that was always right).
    #[test]
    fn dodge_non_numeric_band_on_continuous_scale_warns_and_noops() {
        let b = batch_xyg_with_x(
            Field::new("x", DataType::Utf8, false),
            Arc::new(StringArray::from(vec!["p", "p", "q", "q"])) as ArrayRef,
        );
        let enc = enc_xy("x", "y", Some("g"));
        let pos = PositionAdjust::Dodge { by: Some("g".into()), padding: 0.0 };
        let mut warnings = Vec::new();
        let out = apply_position(&b, Some(&pos), &dummy_scales(), &enc, false, &mut warnings)
            .expect("a Utf8 band on a continuous scale degrades, it does not refuse");
        assert_eq!(out.schema(), b.schema(), "the batch must come back untouched");
        assert_eq!(warnings.len(), 1, "expected exactly one warning, got {warnings:?}");
        match &warnings[0] {
            RenderWarning::PositionAdjustSkipped { adjustment, reason } => {
                assert_eq!(adjustment, "dodge");
                assert!(reason.contains("'x'"), "must name the band column: {reason}");
                assert!(reason.contains("Utf8"), "must name the dtype: {reason}");
            }
            other => panic!("wrong warning variant: {other:?}"),
        }
    }

    /// Shape (c). `Jitter` on a non-`Float64` continuous axis renders instead of
    /// raising, and — spec §9 — produces bit-identical output across
    /// `Float64`/`Int64`/`Int32`/`Float32` for the same data, on BOTH the
    /// explicit-seed path and the `seed: None` hash path.
    ///
    /// RED against the bare downcast: `Int64`/`Int32`/`Float32` all failed it,
    /// so every `xv` was `NaN`, the rewritten column was all-`NaN`, and
    /// `RecordBatch::try_new` rejected a `Float64` array in an integer slot —
    /// `unwrap()` panics with "column types must match schema types". `Float32`
    /// reached the same error; it never even got as far as the comparison.
    #[test]
    fn jitter_continuous_axis_is_dtype_invariant() {
        for seed in [Some(42_u64), None] {
            let pos = PositionAdjust::Jitter { axis: JitterAxis::X, width: 0.5, seed };
            let enc = enc_xy("x", "y", Some("g"));
            let mut reference: Option<Vec<u64>> = None;
            for (name, field, col) in x_column_dtype_matrix() {
                let b = batch_xyg_with_x(field, col);
                let out =
                    apply_position(&b, Some(&pos), &dummy_scales(), &enc, false, &mut Vec::new())
                        .unwrap_or_else(|e| panic!("{name} (seed {seed:?}) must jitter: {e}"));
                assert_eq!(
                    out.schema().field(0).data_type(),
                    &DataType::Float64,
                    "{name}: the jittered column must be retyped to Float64"
                );
                let xa = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
                let bits: Vec<u64> = (0..4).map(|i| xa.value(i).to_bits()).collect();
                // Sanity on the values themselves, not just their agreement: each
                // lands within ±width/2 of its own band value.
                for (i, want) in [1.0, 1.0, 2.0, 2.0].iter().enumerate() {
                    assert!(
                        (xa.value(i) - want).abs() <= 0.25 + 1e-12,
                        "{name} row {i}: {} is outside ±0.25 of {want}",
                        xa.value(i)
                    );
                }
                match &reference {
                    None => reference = Some(bits),
                    Some(want) => assert_eq!(
                        &bits, want,
                        "{name} (seed {seed:?}) jittered differently from Float64 — \
                         jitter must not vary with the dtype the data arrived in"
                    ),
                }
            }
        }
    }

    /// The silent half of shape (c): on an ORDINAL band the column is never
    /// rewritten, so a failed downcast only corrupted the `seed: None` hash key
    /// (`format!("{xv}|{yv}")` read `NaN`) — identical charts jittered
    /// differently by dtype, with nothing raised and nothing warned. The pixel
    /// offsets must now agree across the numeric family.
    ///
    /// RED against the bare downcast: `Int64`/`Int32`/`Float32` seeded from
    /// `"NaN|10"` while `Float64` seeded from `"1|10"`, so the three
    /// `__pos_x_offset__` vectors differed from the `Float64` one in every row.
    #[test]
    fn jitter_ordinal_band_hash_seed_is_dtype_invariant() {
        let pos = PositionAdjust::Jitter { axis: JitterAxis::X, width: 0.4, seed: None };
        let enc = enc_xy("x", "y", Some("g"));
        let mut reference: Option<Vec<f64>> = None;
        for (name, field, col) in x_column_dtype_matrix() {
            let b = batch_xyg_with_x(field, col);
            let out = apply_position(
                &b,
                Some(&pos),
                &scales_ordinal_x_numeric_domain(),
                &enc,
                false,
                &mut Vec::new(),
            )
            .unwrap();
            let offsets = offset_col(&out, "__pos_x_offset__");
            assert!(
                offsets.iter().all(|o| o.is_finite() && o.abs() <= 0.4 * 50.0 / 2.0 + 1e-12),
                "{name}: offsets must stay inside ±width·bandwidth/2: {offsets:?}"
            );
            match &reference {
                None => reference = Some(offsets),
                Some(want) => assert_eq!(
                    &offsets, want,
                    "{name}: the hash seed must not vary with the column's dtype"
                ),
            }
        }
    }

    /// `Jitter`'s twin of the case above, and deliberately the SAME answer: a
    /// `Utf8` axis the scales call continuous warns and leaves the column alone.
    /// The two adjustments must not disagree about one input pathology.
    ///
    /// RED against the bare downcast: the column was rewritten as all-`NaN`
    /// `Float64` into the `Utf8` slot, so this raised with a raw Arrow schema
    /// message ("column types must match schema types") instead of warning —
    /// `expect` fails before any assertion below is reached.
    #[test]
    fn jitter_non_numeric_continuous_axis_warns_and_leaves_the_column_alone() {
        let b = batch_xyg_with_x(
            Field::new("x", DataType::Utf8, false),
            Arc::new(StringArray::from(vec!["p", "p", "q", "q"])) as ArrayRef,
        );
        let enc = enc_xy("x", "y", Some("g"));
        let pos = PositionAdjust::Jitter { axis: JitterAxis::X, width: 0.5, seed: Some(7) };
        let mut warnings = Vec::new();
        let out = apply_position(&b, Some(&pos), &dummy_scales(), &enc, false, &mut warnings)
            .expect("a Utf8 axis degrades, it does not raise an Arrow schema error");
        let xa = out.column(0).as_any().downcast_ref::<StringArray>().expect("still Utf8");
        assert_eq!((0..4).map(|i| xa.value(i)).collect::<Vec<_>>(), vec!["p", "p", "q", "q"]);
        assert_eq!(warnings.len(), 1, "expected exactly one warning, got {warnings:?}");
        match &warnings[0] {
            RenderWarning::PositionAdjustSkipped { adjustment, reason } => {
                assert_eq!(adjustment, "jitter");
                assert!(reason.contains("'x'"), "must name the column: {reason}");
                assert!(reason.contains("Utf8"), "must name the dtype: {reason}");
            }
            other => panic!("wrong warning variant: {other:?}"),
        }
    }

    /// A null band value on a continuous dodge stays null. The span branch has
    /// always stated this rule; the point branch read the raw buffer slot under
    /// the null (`Float64Array::value` on a null is 0.0 in practice, undefined
    /// by contract) and emitted a NON-null mark at `0.0 + offset` — a row the
    /// batch could never draw, turned into one drawn at the wrong place.
    #[test]
    fn dodge_continuous_point_keeps_a_null_band_null() {
        let b = batch_xyg_with_x(
            Field::new("x", DataType::Float64, true),
            Arc::new(Float64Array::from(vec![Some(1.0), None, Some(2.0), Some(2.0)])) as ArrayRef,
        );
        let enc = enc_xy("x", "y", Some("g"));
        let pos = PositionAdjust::Dodge { by: Some("g".into()), padding: 0.0 };
        let out = apply_position(&b, Some(&pos), &dummy_scales(), &enc, false, &mut Vec::new())
            .unwrap();
        let xa = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert!(xa.is_null(1), "a null band value must stay null, got {}", xa.value(1));
        // The non-null rows still dodge: uniques {1, 2} → bandwidth 1, two
        // groups → ∓0.25.
        assert!((xa.value(0) - 0.75).abs() < 1e-9, "row0 x={}", xa.value(0));
        assert!((xa.value(2) - 1.75).abs() < 1e-9, "row2 x={}", xa.value(2));
        assert!((xa.value(3) - 2.25).abs() < 1e-9, "row3 x={}", xa.value(3));
    }

    // ---- GH #184 end to end, through `render_svg` --------------------------
    //
    // The unit tests above pin the position arithmetic; these two pin the thing
    // the issue actually reports — bars present, and bars separated — through
    // the full pipeline, because that is the consequence a `position.rs` unit
    // test cannot see: the ordinal scale's category lookup happens downstream,
    // in the mark drawer, and it is what silently dropped every rect.

    /// Render a two-band, two-group dodged bar chart whose category column has
    /// the given dtype and declared encoding type. Returns the SVG and the
    /// sorted left edges of the bar rects.
    ///
    /// `x_scale_json` is spliced into the x channel: the continuous cases pass
    /// an explicit domain wide enough to hold the dodged positions, since a
    /// data-derived `[1, 2]` domain clips the outer sub-slots and would hide
    /// the very separation under test. The batch and the declared type are the
    /// only things that vary between twins, so any render difference is theirs.
    fn dodge_bar_svg(
        declared_type: &str,
        x_scale_json: &str,
        cat_field: Field,
        cat_col: ArrayRef,
    ) -> (String, Vec<f64>) {
        let spec: ChartSpec = serde_json::from_str(&format!(
            r#"{{"data":{{"kind":"named","name":"default"}},"mark":"bar",
                 "encoding":{{"x":{{"field":"cat","type":"{declared_type}"{x_scale_json}}},
                              "y":{{"field":"val","type":"quantitative"}},
                              "color":{{"field":"g","type":"nominal"}}}},
                 "position":{{"type":"dodge","by":"g","padding":0.0}}}}"#
        ))
        .expect("spec JSON");
        let schema = Arc::new(Schema::new(vec![
            cat_field,
            Field::new("val", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                cat_col,
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0])),
                Arc::new(StringArray::from(vec!["p", "q", "p", "q"])),
            ],
        )
        .unwrap();
        let out = crate::render::render_svg(
            &spec,
            &batch,
            &crate::render::ThemeInputs::default(),
            crate::render::Viewport { width: 640.0, height: 480.0 },
            &crate::render::config::RenderConfig::default(),
            &crate::render::chart_config::ChartConfig::default(),
        )
        .expect("render");
        let bars = bar_rect_lefts(&out.bytes, out.layout.panels[0].plot_area);
        (out.bytes, bars)
    }

    /// The left edges of the filled `<rect>`s that fall inside `plot` — the bar
    /// marks. The plot-area rect itself carries no `fill` (this repo's
    /// single-structural-property convention, see `tests/_svg_rects.py`'s
    /// `plot_rect`) and legend swatches sit outside the plot area.
    fn bar_rect_lefts(svg: &str, plot: Rect) -> Vec<f64> {
        let mut lefts = Vec::new();
        for frag in svg.split("<rect ").skip(1) {
            let frag = &frag[..frag.find('>').unwrap_or(frag.len())];
            if !frag.contains("fill=\"") {
                continue;
            }
            let attr = |name: &str| -> Option<f64> {
                let at = frag.find(&format!("{name}=\""))? + name.len() + 2;
                frag[at..].find('"').and_then(|end| frag[at..at + end].parse().ok())
            };
            let (Some(x), Some(w)) = (attr("x"), attr("width")) else { continue };
            if x >= plot.x - 1e-6 && x + w <= plot.x + plot.w + 1e-6 {
                lefts.push(x);
            }
        }
        lefts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        lefts
    }

    /// Shape (a), end to end and as the issue words it: a `Float64` category
    /// column declared `:N` with `Dodge` renders its bars — four of them, at
    /// four distinct positions — and renders byte-identically to the `Utf8`
    /// column carrying the same category labels.
    ///
    /// RED against the dtype gate: the `Float64` render emitted ZERO bar rects
    /// (`bars` empty, so the count assertion fails first) because the continuous
    /// branch moved x to 0.75/1.25/1.75/2.25 and the ordinal scale found none of
    /// those in its `["1", "2"]` domain. The `Utf8` twin was unaffected, so the
    /// byte-equality assertion failed too.
    #[test]
    fn float64_nominal_dodge_renders_the_same_bars_as_its_utf8_twin() {
        let (float_svg, float_bars) = dodge_bar_svg(
            "nominal",
            "",
            Field::new("cat", DataType::Float64, false),
            Arc::new(Float64Array::from(vec![1.0, 1.0, 2.0, 2.0])) as ArrayRef,
        );
        assert_eq!(
            float_bars.len(),
            4,
            "two bands x two dodge groups must draw four bars; got {float_bars:?}"
        );
        let mut distinct = float_bars.clone();
        distinct.dedup();
        assert_eq!(distinct.len(), 4, "each bar needs its own sub-slot; got {float_bars:?}");

        let (utf8_svg, utf8_bars) = dodge_bar_svg(
            "nominal",
            "",
            Field::new("cat", DataType::Utf8, false),
            Arc::new(StringArray::from(vec!["1", "1", "2", "2"])) as ArrayRef,
        );
        assert_eq!(float_bars, utf8_bars);
        // The category labels are identical strings, so the axis, the layout and
        // every mark must be too: the dtype is not supposed to be observable.
        assert_eq!(
            float_svg, utf8_svg,
            "a Float64 ':N' category column must render byte-identically to its Utf8 twin"
        );
    }

    /// Shape (b), end to end: an `Int64` band on a genuinely CONTINUOUS scale
    /// (the `native_scale=True` shape) separates its dodge groups, and renders
    /// byte-identically to the `Float64` column carrying the same values.
    ///
    /// RED against the dtype gate: `Int64 != Float64` routed the batch into
    /// `apply_dodge_ordinal`, which had no `OrdinalScale` and returned it
    /// unchanged — two bar positions instead of four, the two groups drawn on
    /// top of each other, and no warning. The `Float64` twin dodged correctly,
    /// so the byte-equality assertion failed as well.
    #[test]
    fn int64_continuous_dodge_separates_its_groups_like_its_float64_twin() {
        const DOMAIN: &str = r#","scale":{"type":"linear","domain":[0.0,3.0]}"#;
        let (int_svg, int_bars) = dodge_bar_svg(
            "quantitative",
            DOMAIN,
            Field::new("cat", DataType::Int64, false),
            Arc::new(arrow::array::Int64Array::from(vec![1_i64, 1, 2, 2])) as ArrayRef,
        );
        assert_eq!(
            int_bars.len(),
            4,
            "two bands x two dodge groups must draw four bars; got {int_bars:?}"
        );
        let mut distinct = int_bars.clone();
        distinct.dedup();
        assert_eq!(
            distinct.len(),
            4,
            "a continuous dodge must give each group its own position; got {int_bars:?}"
        );

        let (float_svg, float_bars) = dodge_bar_svg(
            "quantitative",
            DOMAIN,
            Field::new("cat", DataType::Float64, false),
            Arc::new(Float64Array::from(vec![1.0, 1.0, 2.0, 2.0])) as ArrayRef,
        );
        assert_eq!(int_bars, float_bars);
        assert_eq!(
            int_svg, float_svg,
            "an Int64 continuous band must render byte-identically to its Float64 twin"
        );
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
        // The `Normalize` denominator is a bin's magnitude `P + N`. Bin 1's
        // values sum to 0 (10 and -10) but its magnitude is 20, so they split
        // to [0, 0.5] and [-0.5, 0]. Bin 2's values are all 0, so its
        // magnitude is 0 and the guard must make every segment 0.0, never
        // NaN/inf from a 0/0 division.
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        let b = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 1.0, 2.0, 2.0])),
                Arc::new(Float64Array::from(vec![10.0, -10.0, 0.0, 0.0])),
                Arc::new(StringArray::from(vec!["a", "b", "a", "b"])),
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
        let bases = stack_bases(&out);
        assert!(
            tops.iter().chain(bases.iter()).all(|v| v.is_finite()),
            "normalize must not yield NaN/inf, got tops {tops:?} bases {bases:?}"
        );
        assert_eq!(&tops[..2], &[0.5, -0.5], "a cancelling bin splits by sign over its magnitude");
        assert_eq!(&bases[..2], &[0.0, 0.0], "each sign stacks from 0");
        assert_eq!(&tops[2..], &[0.0, 0.0], "all segments collapse to 0 when the magnitude is 0");
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
            mark: Mark::Bar,
            encoding: enc,
            position: Some(pos),
            ..Default::default()
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
        let mark_style = resolve_mark_style(None, &theme, &Mark::Bar).unwrap();

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
        let result = dispatch_mark_build(&spec.mark, &ctx).unwrap();
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
            DiscreteLayout::UNPADDED,
        ));
        let lin = ScaleKind::Linear(LinearScale::new_internal(
            vec![0.0, 100.0], vec![0.0, 100.0], false, false,
        ));
        ResolvedScales { x: ord, y: lin, color: None, size: None, shape: None, opacity: None, fill_opacity: None, stroke_opacity: None, stroke_dash: None, stroke_width: None, x2: None, y2: None, y_slots: Default::default(), raster_cells: None }
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

    /// The dodge seam's view of band padding, under the d3 model (F-L04-03).
    ///
    /// This test used to pin the *opposite* fact: `dodge_padding_inner_is_geometrically_inert`
    /// asserted that `OrdinalScale`'s padding perturbed neither `bandwidth()`
    /// nor any band center, because the render-side model ignored it outright
    /// (that inertness is what refuted GH #66's filed premise, and what
    /// F-L04-03 called a lie). It is no longer true, and the replacement pins
    /// what d3 actually does at this seam:
    ///
    /// - `padding_inner` narrows the drawn band — n=3 over [0, 600] with
    ///   `padding_inner = 0.9` gives denom 2.1, step 285.71…, bandwidth
    ///   28.571… — so a dodge sub-band computed from `bandwidth()` shrinks with
    ///   it, which is the composition rule (scale defines the band, `Dodge`
    ///   subdivides it).
    /// - `padding_outer` is the parameter that moves centers off the unpadded
    ///   100 / 300 / 500.
    /// - The unpadded model still reports exactly the old numbers, which is why
    ///   every default-scale dodge render is byte-identical.
    #[test]
    fn dodge_band_padding_follows_the_d3_model() {
        use crate::scale::ordinal::OrdinalScale;
        let domain = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let band = |layout| OrdinalScale::new_internal(domain.clone(), vec![0.0, 600.0], layout);

        let unpadded = band(DiscreteLayout::UNPADDED);
        assert_eq!(unpadded.bandwidth(), 200.0);
        assert_eq!(
            domain.iter().map(|c| unpadded.scale_internal(c)).collect::<Vec<_>>(),
            vec![Some(100.0), Some(300.0), Some(500.0)],
            "the unpadded model must still report the pre-F-L04-03 centers",
        );

        // padding_inner = 0.9: denom = 3 - 0.9 = 2.1 → step = 600/2.1.
        let inner = band(DiscreteLayout::band(0.9, 0.0, 0.5));
        let step = 600.0 / 2.1;
        assert!(
            (inner.bandwidth() - step * 0.1).abs() < 1e-9,
            "inner padding must narrow the band to |step|·(1 - 0.9); got {}",
            inner.bandwidth()
        );
        assert!(
            inner.bandwidth() < unpadded.bandwidth(),
            "a dodge sub-band derived from bandwidth() must shrink with inner padding"
        );
        // Each category sits at the middle of its own narrow band, which starts
        // at the slot edge (the inner gap follows the band).
        for (i, cat) in domain.iter().enumerate() {
            let expected = step * 0.1 / 2.0 + i as f64 * step;
            let got = inner.scale_internal(cat).expect("category in domain");
            assert!((got - expected).abs() < 1e-9, "center for {cat}: {got} vs {expected}");
        }

        // padding_outer = 0.5: denom = 3 + 1 = 4 → step = 150, start = 75.
        let outer = band(DiscreteLayout::band(0.0, 0.5, 0.5));
        assert_eq!(outer.bandwidth(), 150.0, "outer padding leaves bands full-slot");
        assert_eq!(
            domain.iter().map(|c| outer.scale_internal(c)).collect::<Vec<_>>(),
            vec![Some(150.0), Some(300.0), Some(450.0)],
            "outer padding must inset the centers",
        );
    }

    /// Jitter's spread is a fraction of the DRAWN band, the same band Dodge
    /// subdivides (F-L04-03, spec §4A).
    ///
    /// Two categories over `[0, 300]`: unpadded `step = bandwidth = 150`;
    /// at `padding_inner = 0.5`, `denom = 1.5`, `step = 200` and the drawn
    /// band is `100`. Every row's pixel offset is `(u − 0.5) · width ·
    /// bandwidth` off the same seeded `u`, so the padded offsets must be
    /// exactly `100/150` of their unpadded twins — and with `width = 1.0`
    /// (spread spans the whole band) each padded point must still land within
    /// `±50` of its center, i.e. inside the band the user actually sees rather
    /// than out in the gap the padding opened.
    #[test]
    fn jitter_spread_follows_the_padding_aware_band() {
        use crate::scale::linear::LinearScale;
        use crate::scale::ordinal::OrdinalScale;

        let scales = |layout| ResolvedScales {
            x: ScaleKind::Ordinal(OrdinalScale::new_internal(
                vec!["a".into(), "b".into()],
                vec![0.0, 300.0],
                layout,
            )),
            y: ScaleKind::Linear(LinearScale::new_internal(
                vec![0.0, 100.0],
                vec![0.0, 100.0],
                false,
                false,
            )),
            color: None,
            size: None,
            shape: None,
            opacity: None,
            fill_opacity: None,
            stroke_opacity: None,
            stroke_dash: None,
            stroke_width: None,
            x2: None,
            y2: None,
            y_slots: Default::default(),
            raster_cells: None,
        };
        let b = batch_cat_val_grp();
        let enc = enc_xy("cat", "val", Some("grp"));
        let pos = PositionAdjust::Jitter { axis: JitterAxis::X, width: 1.0, seed: Some(7) };
        let jitter = |layout| {
            let out = apply_position(&b, Some(&pos), &scales(layout), &enc, false, &mut Vec::new())
                .unwrap();
            offset_col(&out, "__pos_x_offset__")
        };

        let unpadded = jitter(DiscreteLayout::UNPADDED);
        let padded = jitter(DiscreteLayout::band(0.5, 0.0, 0.5));
        assert_eq!(unpadded.len(), 4);
        for (i, (p, u)) in padded.iter().zip(unpadded.iter()).enumerate() {
            assert!(u.abs() > 1e-6, "row {i}: fixture must produce a non-zero offset");
            assert!(
                (p - u * (100.0 / 150.0)).abs() < 1e-9,
                "row {i}: padded spread must be 100/150 of the unpadded one; got {p} vs {u}"
            );
            assert!(
                p.abs() <= 50.0,
                "row {i}: a jittered point must stay inside the 100px drawn band; got {p}"
            );
        }
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

    /// A two-bin, two-group batch with mixed signs in each bin: bin 1 holds
    /// a=2, b=-1 and bin 2 holds a=-2, b=3.
    fn batch_mixed_sign_xyg() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 1.0, 2.0, 2.0])),
                Arc::new(Float64Array::from(vec![2.0, -1.0, -2.0, 3.0])),
                Arc::new(StringArray::from(vec!["a", "b", "a", "b"])),
            ],
        )
        .unwrap()
    }

    /// The `(base, top)` of every segment a mixed-sign stack under `offset`
    /// produces, in row order.
    fn mixed_sign_segments(offset: StackOffset) -> Vec<(f64, f64)> {
        let pos = PositionAdjust::Stack { by: Some("g".into()), offset, anchor: StackAnchor::Top, value_axis: None };
        let out = apply_position(&batch_mixed_sign_xyg(), Some(&pos), &dummy_scales(), &enc_xy("x", "y", Some("g")), false, &mut Vec::new()).unwrap();
        stack_bases(&out).into_iter().zip(stack_tops(&out, 1)).collect()
    }

    fn assert_segments(got: &[(f64, f64)], expected: &[(f64, f64)], case: &str) {
        assert_eq!(got.len(), expected.len(), "{case}: {got:?}");
        for (g, e) in got.iter().zip(expected) {
            assert!((g.0 - e.0).abs() < 1e-12 && (g.1 - e.1).abs() < 1e-12, "{case}: expected {expected:?}, got {got:?}");
        }
    }

    /// `zero`: positives stack upward from 0 and negatives downward from 0, so
    /// each bin spans `[-N, P]` and no segment overlaps another. A single
    /// running total would draw bin 1's b from 2 down to 1, over a.
    #[test]
    fn stack_zero_splits_mixed_signs_at_zero() {
        assert_segments(
            &mixed_sign_segments(StackOffset::Zero),
            &[(0.0, 2.0), (0.0, -1.0), (0.0, -2.0), (0.0, 3.0)],
            "zero",
        );
    }

    /// `normalize`: each segment is divided by its bin's `P + N`, so
    /// positives fill `[0, p]` and negatives `[-n, 0]` with `p + n = 1`.
    #[test]
    fn stack_normalize_splits_mixed_signs_over_the_bin_magnitude() {
        assert_segments(
            &mixed_sign_segments(StackOffset::Normalize),
            &[(0.0, 2.0 / 3.0), (0.0, -1.0 / 3.0), (0.0, -0.4), (0.0, 0.6)],
            "normalize",
        );
    }

    /// `center`: the sign-split stack shifts by `(N - P) / 2`, centering its
    /// `[-N, P]` extent on 0: bin 1's [-1, 2] and bin 2's [-2, 3] both move
    /// down by 0.5.
    #[test]
    fn stack_center_centers_the_sign_split_extent_on_zero() {
        assert_segments(
            &mixed_sign_segments(StackOffset::Center),
            &[(-0.5, 1.5), (-0.5, -1.5), (-0.5, -2.5), (-0.5, 2.5)],
            "center",
        );
    }

    /// An all-negative stack: every segment hangs below 0. The batch is
    /// bin 1 a=-2, b=-1 and bin 2 a=-3, b=-4.
    fn all_negative_segments(offset: StackOffset) -> Vec<(f64, f64)> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 1.0, 2.0, 2.0])),
                Arc::new(Float64Array::from(vec![-2.0, -1.0, -3.0, -4.0])),
                Arc::new(StringArray::from(vec!["a", "b", "a", "b"])),
            ],
        )
        .unwrap();
        let pos = PositionAdjust::Stack { by: Some("g".into()), offset, anchor: StackAnchor::Top, value_axis: None };
        let out = apply_position(&batch, Some(&pos), &dummy_scales(), &enc_xy("x", "y", Some("g")), false, &mut Vec::new()).unwrap();
        stack_bases(&out).into_iter().zip(stack_tops(&out, 1)).collect()
    }

    /// All-negative stacks, the shared stack bars and areas both draw from:
    /// `zero` spans `[-N, 0]` downward from 0; `normalize` fills `[-1, 0]`
    /// (not the upside-down `[0, 1]` a single signed running total over a
    /// signed sum gives); `center` centers each bin's `[-N, 0]` on 0.
    #[test]
    fn an_all_negative_stack_hangs_below_zero_for_every_offset() {
        assert_segments(
            &all_negative_segments(StackOffset::Zero),
            &[(0.0, -2.0), (-2.0, -3.0), (0.0, -3.0), (-3.0, -7.0)],
            "zero",
        );
        assert_segments(
            &all_negative_segments(StackOffset::Normalize),
            &[(0.0, -2.0 / 3.0), (-2.0 / 3.0, -1.0), (0.0, -3.0 / 7.0), (-3.0 / 7.0, -1.0)],
            "normalize",
        );
        assert_segments(
            &all_negative_segments(StackOffset::Center),
            &[(1.5, -0.5), (-0.5, -1.5), (3.5, 0.5), (0.5, -3.5)],
            "center",
        );
    }

    /// With no negative value every offset stacks bit for bit as one running
    /// total in group order does (the all-positive fixture `batch_xyg`).
    #[test]
    fn stack_without_negatives_is_one_running_total_for_every_offset() {
        let b = batch_xyg();
        let enc = enc_xy("x", "y", Some("g"));
        let raw = [10.0_f64, 20.0, 30.0, 40.0];
        for offset in [StackOffset::Zero, StackOffset::Normalize, StackOffset::Center] {
            let pos = PositionAdjust::Stack { by: Some("g".into()), offset, anchor: StackAnchor::Top, value_axis: None };
            let out = apply_position(&b, Some(&pos), &dummy_scales(), &enc, false, &mut Vec::new()).unwrap();
            let mut expected_tops = Vec::new();
            let mut expected_bases = Vec::new();
            for bin in raw.chunks(2) {
                let total: f64 = bin.iter().sum();
                let mut acc = 0.0_f64;
                let mut segs = Vec::new();
                for v in bin {
                    let n = if matches!(offset, StackOffset::Normalize) { v / total } else { *v };
                    segs.push((acc, acc + n));
                    acc += n;
                }
                let mid = if matches!(offset, StackOffset::Center) { acc / 2.0 } else { 0.0 };
                for (b0, t0) in segs {
                    expected_bases.push(b0 - mid);
                    expected_tops.push(t0 - mid);
                }
            }
            let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&stack_tops(&out, 1)), bits(&expected_tops), "{offset:?} tops");
            assert_eq!(bits(&stack_bases(&out)), bits(&expected_bases), "{offset:?} bases");
        }
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
    /// (`scale bandwidth / dodge_n_groups * factor`, for whatever `factor` the
    /// mark uses — bar's fixed `0.8`, box/tick's `band_size`).
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
