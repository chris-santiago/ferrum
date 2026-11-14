//! The encoding impute step: each layer that writes `impute=` completes the
//! series it draws, over the rows the null-key drop kept, so no synthesised
//! row is ever counted as a caller's.

use std::collections::HashMap;

use arrow::record_batch::RecordBatch;

use super::routes::LayerRows;
use super::{
    binned_color_buckets, concat_batches_by_name, layer_entry, layer_names_unflipped, public_layer,
    user_facing_channel, FacetPartitionKey, LayerPrepared,
};
use crate::render::series_keys::{ColorSplit, SeriesKeys, SplitSource};
use crate::render::{RenderError, RenderWarning};
use arrow::array::Array;
use crate::spec::chart::ChartSpec;
use crate::transform::context::{TransformContext, TransformWarning};

// ── The impute step: each layer completes its own series ─────────────────────

/// What identifies one completion: the plan (value channel, key, series
/// fields and options) and the rows it read. Two completions with equal plans
/// over equal rows produce the same rows and the same notices. One definition
/// of "imputes identically", read by the layers of one chart that share a key
/// ([`apply_impute_after_drop`]) and by an overlay composite whose leaves
/// complete the same rows ([`ImputeReport::completes_like`]).
#[derive(Debug, Clone)]
pub(in crate::render) struct ImputeIdentity {
    plan: ImputePlan,
    rows: RecordBatch,
}

impl ImputeIdentity {
    fn new(plan: &ImputePlan, rows: &RecordBatch) -> Self {
        Self { plan: plan.clone(), rows: rows.clone() }
    }

    /// Whether `other` completes the same rows the same way as this one.
    pub(in crate::render) fn completes_like(&self, other: &Self) -> bool {
        self.plan == other.plan && self.rows == other.rows
    }
}

/// The notices one impute completion raised, with the identity of the
/// completion, so a composite that renders two completions as separate charts
/// reports an identical one once ([`ImputeReport::completes_like`]), the way one
/// chart reports layers that share a completion once.
#[derive(Debug, Clone)]
pub(in crate::render) struct ImputeReport {
    identity: ImputeIdentity,
    pub(in crate::render) warnings: Vec<RenderWarning>,
}

impl ImputeReport {
    /// Whether `other` completed the same rows the same way as this report.
    pub(in crate::render) fn completes_like(&self, other: &Self) -> bool {
        self.identity.completes_like(&other.identity)
    }
}

/// What one layer's impute completes, read off its drawn encoding: the value
/// channel's field, the key channel's field it is completed along, the
/// columns that ride with the key, the fields that name its series, and the
/// options of the `impute=` dict. Two layers that read one batch and hold equal
/// plans complete it identically.
#[derive(Debug, Clone, PartialEq)]
struct ImputePlan {
    value: String,
    key: String,
    /// The key channel's secondary (`x2` for an `x` key, a binned x's
    /// `bin_end`), copied from the row an imputed row's key is copied from.
    companions: Vec<String>,
    /// The fields a series splits by ([`SeriesKeys::levels`]), less the key and
    /// value fields, which are never a series key.
    series: Vec<String>,
    /// The series field that is a binned colour, keyed by bin.
    binned: Option<String>,
    options: crate::transform::impute::ChannelImpute,
}

/// How a layer's impute speaks when it refuses: the key channel as the caller
/// wrote it. The layer is named by the caller of the step, through
/// [`RenderError::in_layer_context`], as every layer-voiced refusal is.
#[derive(Debug, Clone, Copy)]
struct ImputeVoice {
    key: &'static str,
}

impl ImputeVoice {
    fn refusal(&self, text: String) -> RenderError {
        RenderError::Impute { reason: text }
    }

    /// The refusal of a completion, in the channels the caller wrote.
    fn of(&self, refusal: crate::transform::impute::ImputeRefusal) -> RenderError {
        use crate::transform::impute::ImputeRefusal;
        let key = self.key;
        let text = match refusal {
            ImputeRefusal::Refused(text) => text,
            ImputeRefusal::CompanionDiffers { key: at, .. } => {
                let article = if key == "x" { "an" } else { "a" };
                format!(
                    "impute: rows with {key} {at} hold different {key}2 values, so an imputed row has no \
                     {key}2; impute {article} {key} whose {key}2 follows from it"
                )
            }
            ImputeRefusal::CompanionUnheld { key: at, .. } => format!(
                "impute: keyvals {at} is held by no row, so an imputed row has no {key}2 to copy; list \
                 only {key} values some row holds"
            ),
        };
        self.refusal(text)
    }
}

/// The facts every layer's completion reads besides its own rows.
struct ImputeEnv<'a> {
    spec: &'a ChartSpec,
    /// A faceted chart's partitions, in the order the pipeline made them.
    order: &'a [FacetPartitionKey],
    /// The buckets a binned colour is cut into, when the colour scale is one.
    buckets: Option<&'a crate::render::scale_resolve::DiscretizedColors>,
}

/// One completion of a layer's rows: the completed rows (`None` when a series
/// key cannot be read, which leaves the rows as written) and how many rows it
/// left in no series, summed over facet partitions, for the one notice that
/// names them.
struct Completion {
    rows: Option<RecordBatch>,
    unseriesed: usize,
}

/// Layer `index`'s impute over `batch` (its own rows), or `None` when it
/// writes none or its columns cannot be completed (a missing column or a key
/// no reader takes: the render's own refusal speaks for those).
fn layer_impute(
    spec: &ChartSpec,
    index: usize,
    layer: &LayerPrepared,
    batch: &RecordBatch,
    buckets: Option<&crate::render::scale_resolve::DiscretizedColors>,
    color: ColorSplit,
) -> Result<Option<(ImputePlan, ImputeVoice)>, RenderError> {
    use crate::spec::encoding::EncodingSpec;
    let drawn = layer.drawn_encoding();
    let voice_for = |key: &'static str| ImputeVoice {
        key: user_facing_channel(key, layer_names_unflipped(spec, index, layer.coord.as_ref())),
    };
    fn impute_of(binding: &Option<EncodingSpec>) -> Option<&serde_json::Value> {
        binding.as_ref().and_then(|binding| binding.impute.as_ref())
    }
    let (value_wire, key_wire, wire) = match (impute_of(&drawn.x), impute_of(&drawn.y)) {
        (None, None) => return Ok(None),
        (Some(_), Some(_)) => {
            return Err(voice_for("x").refusal(
                "impute: impute= is written on both x and y; a series is completed along one axis, so \
                 write it on the value channel only"
                    .to_string(),
            ))
        }
        (Some(wire), None) => ("x", "y", wire),
        (None, Some(wire)) => ("y", "x", wire),
    };
    let voice = voice_for(key_wire);
    let said = |wire: &'static str| voice_for(wire).key;
    let (value_binding, key_binding, value_secondary, key_secondary) = if value_wire == "y" {
        (&drawn.y, &drawn.x, &drawn.y2, &drawn.x2)
    } else {
        (&drawn.x, &drawn.y, &drawn.x2, &drawn.y2)
    };
    let (Some(value_binding), Some(key_binding)) = (value_binding, key_binding) else {
        let (value, key) = (said(value_wire), said(key_wire));
        return Err(voice.refusal(format!(
            "impute: impute= on {value} completes each series along {key}, but the layer draws no {key}; \
             bind {key} or remove impute="
        )));
    };
    if let Some(secondary) = value_secondary {
        let value = said(value_wire);
        return Err(voice.refusal(format!(
            "impute: the layer also draws {value}2 '{}', which an imputed row would leave empty; remove \
             {value}2 or write the missing rows",
            secondary.field
        )));
    }
    let options = crate::transform::impute::ChannelImpute::from_wire(wire).map_err(|reason| voice.refusal(reason))?;
    let (value, key) = (value_binding.field.clone(), key_binding.field.clone());
    let key_readable = batch
        .column_by_name(&key)
        .is_some_and(|column| crate::transform::group_key::is_groupby_supported_dtype(column.data_type()));
    if !key_readable || batch.column_by_name(&value).is_none() {
        return Ok(None);
    }
    let companions: Vec<String> = key_secondary
        .iter()
        .map(|binding| binding.field.clone())
        .filter(|field| *field != key && *field != value && batch.column_by_name(field).is_some())
        .collect();
    // A binned colour keys its series by bin when its scale is cut by the bin
    // rule; otherwise it is a continuous colour, which the mark draws as one
    // series, and keys nothing.
    let color = match color {
        ColorSplit::Bins if !buckets.is_some_and(|buckets| buckets.is_cut_by_bin()) => ColorSplit::Unsplit,
        color => color,
    };
    let detail = layer.mark_style.as_ref().and_then(|m| m.detail.as_deref());
    let keys = SeriesKeys::of(layer.mark, &drawn, detail, SplitSource::Decided(color), batch.schema().as_ref());
    let series: Vec<String> =
        keys.levels().iter().map(|k| k.field().to_string()).filter(|field| *field != key && *field != value).collect();
    let binned = keys.binned_color().filter(|field| series.iter().any(|name| name == field)).map(str::to_string);
    Ok(Some((ImputePlan { value, key, companions, series, binned, options }, voice)))
}

/// Each row's series over `fields`: rows sharing every field's key are one
/// series, numbered in first-appearance order and named as a warning names a
/// group (`h=y`); a row with no value on a field belongs to none. No fields
/// make the whole frame one series. `None` when a field's key cannot be read,
/// which the mark's own refusal reports.
pub(super) fn series_of_rows(
    batch: &RecordBatch,
    fields: &[String],
    binned: Option<(&str, &crate::render::scale_resolve::DiscretizedColors)>,
) -> Option<crate::transform::impute::Series> {
    use crate::transform::impute::Series;
    let n_rows = batch.num_rows();
    if fields.is_empty() {
        return Some(Series { of_row: vec![Some(0); n_rows], labels: vec![None] });
    }
    let mut columns: Vec<Vec<Option<String>>> = Vec::with_capacity(fields.len());
    let mut bins: Vec<Option<Vec<Option<usize>>>> = Vec::with_capacity(fields.len());
    for field in fields {
        match binned.filter(|(name, _)| *name == field.as_str()) {
            Some((_, buckets)) => {
                let rows: Vec<Option<usize>> = crate::render::arrow_cast::col_as_f64(batch, field)
                    .ok()?
                    .into_iter()
                    .map(|value| value.and_then(|v| buckets.bucket(v)))
                    .collect();
                columns.push(rows.iter().map(|bin| bin.map(|bin| bin.to_string())).collect());
                bins.push(Some(rows));
            }
            None => {
                columns.push(crate::render::arrow_cast::col_as_identity_key_str(batch, field).ok()?);
                bins.push(None);
            }
        }
    }
    // A series is named as the transform door names a group: a date as a date, a
    // binned colour by its range (the legend's), never by an index or an epoch.
    let named = |index: usize, row: usize| -> String {
        let field = &fields[index];
        if let (Some(rows), Some((_, buckets))) = (&bins[index], binned) {
            if let Some(bin) = rows[row] {
                let bounds = buckets.bounds();
                let close = if bin + 2 == bounds.len() { ']' } else { ')' };
                return format!("{field}=[{}, {}{close}", bounds[bin], bounds[bin + 1]);
            }
        }
        let column = batch.column_by_name(field);
        let key = column.and_then(|column| crate::transform::group_key::groupby_key_at(column.as_ref(), column.data_type(), row));
        match key {
            Some(key) => crate::transform::context::describe_group_keys(std::slice::from_ref(field), &[key]),
            None => format!("{field}={}", columns[index][row].as_deref().unwrap_or("")),
        }
    };
    let mut ids: HashMap<Vec<&str>, usize> = HashMap::new();
    let mut labels: Vec<Option<String>> = Vec::new();
    let mut of_row = Vec::with_capacity(n_rows);
    for row in 0..n_rows {
        let key: Option<Vec<&str>> = columns.iter().map(|column| column[row].as_deref()).collect();
        of_row.push(key.map(|key| {
            *ids.entry(key).or_insert_with(|| {
                let parts: Vec<String> = (0..fields.len()).map(|index| named(index, row)).collect();
                labels.push(Some(parts.join(", ")));
                labels.len() - 1
            })
        }));
    }
    Some(Series { of_row, labels })
}

/// `batch`'s rows complete along `plan`: each series gains a row for every key
/// it lacks, and a null or `NaN` value is filled. No rows when a series key
/// cannot be read, which leaves the rows as they are. `facet_fields` are
/// constant within `batch` (a facet partition), so an imputed row copies the
/// panel's value with its series' own keys.
fn complete_rows(
    batch: &RecordBatch,
    plan: &ImputePlan,
    facet_fields: &[String],
    voice: &ImputeVoice,
    env: &ImputeEnv<'_>,
    sink: &TransformContext,
) -> Result<Completion, RenderError> {
    use crate::transform::impute::{complete_series, ImputeColumns, ImputeDoor, ImputeKey};
    if batch.num_rows() == 0 {
        return Ok(Completion { rows: Some(batch.clone()), unseriesed: 0 });
    }
    let binned = plan.binned.as_deref().zip(env.buckets);
    let Some(series) = series_of_rows(batch, &plan.series, binned) else { return Ok(Completion { rows: None, unseriesed: 0 }) };
    let unseriesed = series.of_row.iter().filter(|series| series.is_none()).count();
    let mut copied = plan.series.clone();
    for field in facet_fields {
        if *field != plan.key && *field != plan.value && !copied.contains(field) {
            copied.push(field.clone());
        }
    }
    let columns = ImputeColumns {
        value: &plan.value,
        key: Some(ImputeKey { field: &plan.key, companions: &plan.companions }),
        series: &copied,
    };
    complete_series(batch, &columns, &series, &plan.options.options(), ImputeDoor::Encoding, sink)
        .map(|rows| Completion { rows: Some(rows), unseriesed })
        .map_err(|refusal| voice.of(refusal))
}

/// `batch` split into its facet partitions, in `order`, and the rows that
/// belong to none (a null facet value).
fn split_by_facet(
    batch: &RecordBatch,
    facet: &crate::layout::FacetSpec,
    order: &[FacetPartitionKey],
) -> Result<(Vec<RecordBatch>, Option<RecordBatch>), RenderError> {
    let columns = crate::render::facet_key_column(batch, &facet.field)?;
    let rows = facet.row.as_deref().map(|field| crate::render::facet_key_column(batch, field)).transpose()?;
    let places: HashMap<&FacetPartitionKey, usize> = order.iter().enumerate().map(|(place, key)| (key, place)).collect();
    let part_of: Vec<Option<usize>> = (0..batch.num_rows())
        .map(|row| {
            let column = columns.is_valid(row).then(|| columns.value(row).to_string())?;
            let row_value = match &rows {
                Some(rows) => Some(rows.is_valid(row).then(|| rows.value(row).to_string())?),
                None => None,
            };
            places.get(&(column, row_value)).copied()
        })
        .collect();
    let select = |keep: &dyn Fn(Option<usize>) -> bool| -> Result<RecordBatch, RenderError> {
        let mask: arrow::array::BooleanArray = part_of.iter().map(|part| Some(keep(*part))).collect();
        arrow::compute::filter_record_batch(batch, &mask)
            .map_err(|e| RenderError::RowSelectionFailed { step: "impute", reason: e.to_string() })
    };
    let parts = (0..order.len()).map(|place| select(&|part| part == Some(place))).collect::<Result<Vec<_>, _>>()?;
    let unplaced = part_of.contains(&None).then(|| select(&|part| part.is_none())).transpose()?;
    Ok((parts, unplaced))
}

/// Complete `plan` over `batch`, each facet partition on its own when the chart
/// facets (every panel completes the series it holds, over the keys it holds),
/// and the partitions put back in their order.
fn complete_layer_rows(
    batch: &RecordBatch,
    plan: &ImputePlan,
    voice: &ImputeVoice,
    env: &ImputeEnv<'_>,
    sink: &TransformContext,
) -> Result<Completion, RenderError> {
    let facet = env.spec.facet.as_ref().filter(|facet| {
        batch.column_by_name(&facet.field).is_some() && facet.row.as_deref().is_none_or(|row| batch.column_by_name(row).is_some())
    });
    let Some(facet) = facet else { return complete_rows(batch, plan, &[], voice, env, sink) };
    let facet_fields: Vec<String> = std::iter::once(facet.field.clone()).chain(facet.row.clone()).collect();
    let (parts, unplaced) = split_by_facet(batch, facet, env.order)?;
    let mut completed = Vec::with_capacity(parts.len() + 1);
    let mut unseriesed = 0;
    for part in &parts {
        let done = complete_rows(part, plan, &facet_fields, voice, env, sink)?;
        unseriesed += done.unseriesed;
        match done.rows {
            Some(rows) => completed.push(rows),
            None => return Ok(Completion { rows: None, unseriesed }),
        }
    }
    completed.extend(unplaced);
    Ok(Completion { rows: Some(concat_batches_by_name("impute", completed)?), unseriesed })
}

/// `error` named as raised by layer `index`, through the one layer context
/// every layer-voiced refusal uses (a composite overlay names a member's own
/// index on it afterwards).
fn in_its_layer(spec: &ChartSpec, index: usize, layer: &LayerPrepared, error: RenderError) -> RenderError {
    error.in_layer_context(public_layer(spec, index, layer.mark), &layer_entry(spec, index, layer.mark))
}

/// The warnings a completion raised, each once, led by the one notice naming the
/// rows it left in no series (null on a series key the null-key drop kept
/// because they also place nothing): they are left as written.
fn drain_completion(sink: &TransformContext, unseriesed: usize, plan: &ImputePlan) -> Vec<TransformWarning> {
    let mut warnings = Vec::new();
    if unseriesed > 0 {
        let columns = plan.series.iter().map(|field| format!("'{field}'")).collect::<Vec<_>>().join(" or ");
        // A binned colour is also un-seried by a NaN or infinite value (it has no bin).
        let cause = if plan.binned.is_some() { "a null or non-finite value" } else { "a null" };
        warnings.push(TransformWarning::FallbackApplied {
            transform: "Impute".into(),
            group: None,
            reason: format!(
                "{unseriesed} row(s) have {cause} in {columns}, so they belong to no series; they were left as \
                 written and no key is completed for them"
            ),
        });
    }
    warnings.extend(drain_once(sink));
    warnings
}

/// The warnings a completion raised, each once: a faceted completion raises
/// the same notice in every panel that has it.
fn drain_once(sink: &TransformContext) -> Vec<TransformWarning> {
    let mut warnings: Vec<TransformWarning> = Vec::new();
    for warning in sink.warnings.borrow_mut().drain(..) {
        if !warnings.contains(&warning) {
            warnings.push(warning);
        }
    }
    warnings
}

/// The impute step, run over the rows the null-key drop kept so no
/// synthesised row is ever counted as a caller's.
///
/// A layer imputes when its drawn encoding writes `impute=` on `x` or `y`
/// (its own or inherited): that channel is the value, the other positional
/// channel the key, and the layer's series (its colour when the mark splits by
/// it, `shape`, `stroke_dash`, `detail`) are each completed over the keys the
/// layer's rows hold ([`crate::transform::impute::complete_series`]). Every layer
/// completes its own series, so layers that key different series read
/// different rows.
///
/// Layers reading one key whose impute is identical share its completed rows,
/// published in place. Where they differ, each imputing layer reads its
/// completed rows under its own key ([`LayerRows::publish_own_rows`], the route
/// the drop uses) and a layer with no impute keeps the caller's rows. Each imputing layer reports
/// its own fallback warnings once, in layer order.
pub(super) fn apply_impute_after_drop(
    spec: &ChartSpec,
    theme: &crate::layout::ThemeInputs,
    leaf_scales: Option<&crate::render::scale_resolve::LeafScaleContext>,
    order: &[FacetPartitionKey],
    layers: &[LayerPrepared],
    colors: &[ColorSplit],
    rows: &mut LayerRows<'_>,
) -> Result<Vec<ImputeReport>, RenderError> {
    // The drawn encoding is a subset of the completed one, so a layer whose
    // completed encoding writes no impute draws none.
    let writes_impute = |layer: &LayerPrepared| [&layer.encoding.x, &layer.encoding.y].into_iter().any(|binding| binding.as_ref().is_some_and(|b| b.impute.is_some()));
    if !layers.iter().any(writes_impute) {
        return Ok(Vec::new());
    }
    // Bins are resolved over the rows each layer draws now: the spec carries
    // the routes the drop wrote.
    let buckets = layers
        .iter()
        .filter(|layer| writes_impute(layer))
        .any(|layer| !layer.binned_color_bindings().is_empty())
        .then(|| binned_color_buckets(rows.spec(), rows.outputs(), rows.lineages(), theme, leaf_scales))
        .flatten();
    let mut plans: Vec<Option<(ImputePlan, ImputeVoice)>> = Vec::with_capacity(layers.len());
    for (index, layer) in layers.iter().enumerate() {
        plans.push(match rows.outputs().get(rows.rows_key(index)) {
            Some(batch) => layer_impute(spec, index, layer, batch, buckets.as_ref(), colors[index]).map_err(|e| in_its_layer(spec, index, layer, e))?,
            None => None,
        });
    }
    if plans.iter().all(Option::is_none) {
        return Ok(Vec::new());
    }
    let env = ImputeEnv { spec, order, buckets: buckets.as_ref() };
    let sink = TransformContext::default();
    let mut reports: Vec<(usize, ImputeReport)> = Vec::new();
    let mut keys: Vec<String> = Vec::new();
    for (index, _) in plans.iter().enumerate().filter(|(_, plan)| plan.is_some()) {
        let key = rows.rows_key(index).to_string();
        if !keys.contains(&key) {
            keys.push(key);
        }
    }
    for key in keys {
        let members: Vec<usize> = (0..layers.len()).filter(|&index| rows.rows_key(index) == key).collect();
        let Some(original) = rows.outputs().get(&key).cloned() else { continue };
        // Layers on one key read one batch, so they impute identically when
        // their completions are identical over it.
        let identity_of = |index: usize| plans[index].as_ref().map(|(plan, _)| ImputeIdentity::new(plan, &original));
        let first = identity_of(members[0]);
        let shared = first.as_ref().is_some_and(|first| members[1..].iter().all(|&index| identity_of(index).is_some_and(|identity| identity.completes_like(first))));
        if shared {
            let Some((plan, voice)) = &plans[members[0]] else { continue };
            let done = complete_layer_rows(&original, plan, voice, &env, &sink).map_err(|e| in_its_layer(spec, members[0], &layers[members[0]], e))?;
            // One computation ran for every layer on the key, so its notices are said once.
            reports.push((members[0], impute_report(plan, &original, drain_completion(&sink, done.unseriesed, plan))));
            if let Some(completed) = done.rows {
                rows.replace(&key, completed);
            }
            continue;
        }
        for &index in members.iter().filter(|&&index| plans[index].is_some()) {
            let Some((plan, voice)) = &plans[index] else { continue };
            let done = complete_layer_rows(&original, plan, voice, &env, &sink).map_err(|e| in_its_layer(spec, index, &layers[index], e))?;
            reports.push((index, impute_report(plan, &original, drain_completion(&sink, done.unseriesed, plan))));
            if let Some(completed) = done.rows {
                rows.publish_own_rows(index, &key, completed);
            }
        }
    }
    reports.sort_by_key(|(index, _)| *index);
    Ok(reports.into_iter().map(|(_, report)| report).filter(|report| !report.warnings.is_empty()).collect())
}

fn impute_report(plan: &ImputePlan, rows: &RecordBatch, said: Vec<TransformWarning>) -> ImputeReport {
    ImputeReport {
        identity: ImputeIdentity::new(plan, rows),
        warnings: said.into_iter().map(RenderWarning::Transform).collect(),
    }
}
