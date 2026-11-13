//! The mark-level drop of rows with a null key on a channel a layer draws.
//!
//! A row whose category (the field a mark places on a categorical `x` or `y`)
//! or whose categorical grouping value (`color`, `shape`, `stroke_dash`,
//! `detail`) is null belongs to no category. A statistic decides the nulls of
//! the fields it groups by through its own partition; this module decides
//! every other key field of the rows a layer draws, once per render, after the
//! transform pipeline and the column gate and before any scale, legend, facet
//! panel or mark build reads the rows.
//!
//! - **Keys.** A layer's key fields are the columns it draws on a categorical
//!   position or a categorical grouping channel ([`layer_key_fields`]), read
//!   from the same drawn encoding the column gate judges, so a binding the
//!   render calls ignored is never a key. A field a passed step grouped under
//!   a fit fallback keeps its null on purpose and is left alone.
//! - **Counted rows.** A null-category row counts when the layer's opposite
//!   positional channel holds a value on that row, or when the layer binds no
//!   opposite channel at all; a null-grouping row counts when every position
//!   the layer binds itself holds a value. The rows that do not count place
//!   nothing and are the census's to name.
//! - **Where the kept rows live.** Layers reading one key that count the same
//!   rows replace that key's batch in place. Where two layers on one key count
//!   different rows, the first keeps the key and a later one gets its own kept
//!   rows under [`layer_key`], recorded on the layer as `own_rows`; its
//!   `data_source` is untouched, so everything that classifies a layer by its
//!   source answers as before.
//! - **Reporting.** One [`report_skipped_null_keys`] call per layer under its
//!   entry: a count per field, or the all-null sentence when every row of the
//!   layer is null-keyed.

use std::collections::{HashMap, HashSet};

use arrow::array::{Array, BooleanArray};
use arrow::record_batch::RecordBatch;

use super::{layer_entry, FacetPartitionKey, FacetPartitions, LayerPrepared};
use crate::render::{RenderError, RenderWarning};
use crate::spec::chart::ChartSpec;
use crate::spec::encoding::EncodingSpec;
use crate::spec::layer::PositionalChannel;
use crate::transform::context::TransformContext;
use crate::transform::core::FINAL_OUTPUT_KEY;
use crate::transform::group_key::{report_skipped_null_keys, NullKeyCounts, NullKeyPolicy};
use crate::transform::run::OutputLineages;

/// The key a layer's own kept rows are published under in the transform
/// outputs, for a layer the drop gave rows its key-siblings do not share.
pub(in crate::render) fn layer_key(index: usize) -> String {
    format!("__null_keys_layer_{index}__")
}

/// The key a panel's slice of the rows published under `key` is published
/// under, in the per-panel output map the panel pass builds
/// (`scene_build::resolve_panel_scales`).
pub(in crate::render) fn panel_key(key: &str) -> String {
    format!("{key}__panel__")
}

/// What the drop did: the layers that read rows of their own now, and the
/// warnings it raised.
#[derive(Debug, Default)]
pub(super) struct NullKeyDrop {
    /// `(layer index, key of the layer's own rows)`, empty for every chart
    /// whose layers all read the key they started on.
    pub(super) routes: Vec<(usize, String)>,
    pub(super) warnings: Vec<RenderWarning>,
}

/// How a null-category row's opposite positional channel decides whether the
/// row counts.
#[derive(Debug, Clone, PartialEq)]
enum Opposite {
    /// The layer binds the opposite channel: the row counts when that column
    /// holds a value on it.
    Bound(String),
    /// The layer binds no opposite channel anywhere (a one-axis tick): every
    /// null-category row draws, so every one counts.
    Absent,
    /// The layer takes its opposite coordinate from its parent without
    /// drawing it (an injected reference column's structural nulls): none of
    /// its null-category rows places a mark, so none counts.
    Inherited,
}

/// What kind of key a field is, which decides when a null in it counts.
#[derive(Debug, Clone, PartialEq)]
enum KeyKind {
    Category(Opposite),
    Grouping,
}

/// One key field of a layer.
#[derive(Debug, Clone, PartialEq)]
struct KeyField {
    field: String,
    kind: KeyKind,
}

/// A layer's key fields in the column gate's read order (positional bindings,
/// the other bindings, then `detail`), each column once, plus the positional
/// fields the layer binds itself (what a null-grouping row must hold a value
/// on to count).
#[derive(Debug, Default)]
struct LayerKeys {
    fields: Vec<KeyField>,
    own_positions: Vec<String>,
}

/// The columns a passed step that pools a whole-batch fallback grouped by:
/// their null is the pooled fit's own, so the drop leaves them alone.
fn pooled_fields(lineage: Option<&crate::transform::run::OutputLineage>) -> HashSet<String> {
    lineage
        .map(|lineage| {
            lineage
                .steps
                .iter()
                .filter(|step| step.pools_null_keys)
                .flat_map(|step| step.groupby.iter().cloned())
                .collect()
        })
        .unwrap_or_default()
}

/// Whether `enc`'s column of `dtype` reads as categories: typed nominal or
/// ordinal, carrying a categorical scale, or a dtype the continuous reader
/// cannot place. An untyped all-null column is a quantitative position.
fn reads_categorical(enc: &EncodingSpec, dtype: &arrow::datatypes::DataType) -> bool {
    use crate::spec::encoding::DataType;
    matches!(enc.type_, Some(DataType::Nominal | DataType::Ordinal))
        || enc.scale.as_ref().is_some_and(crate::render::scale_resolve::is_categorical_scale)
        || (*dtype != arrow::datatypes::DataType::Null && crate::render::scale_resolve::dtype_reads_categorical(dtype))
}

/// Whether `mark` dashes each row by a numeric `stroke_dash` index (a tick
/// draws no `stroke_dash`, so its binding never reaches the drop).
fn dashes_each_row_by_index(mark: crate::spec::mark::Mark) -> bool {
    use crate::spec::mark::Mark;
    matches!(mark, Mark::Point | Mark::Bar | Mark::Rect | Mark::Rule)
}

/// The key fields of layer `index`, read against its own `batch`.
fn layer_key_fields(
    spec: &ChartSpec,
    index: usize,
    layer: &LayerPrepared,
    batch: &RecordBatch,
    pooled: &HashSet<String>,
) -> LayerKeys {
    let drawn = layer.drawn_encoding();
    let raw = spec.layers.as_ref().and_then(|layers| layers.get(index));
    let flips = crate::spec::coord::flips_positional_channels(layer.coord.as_ref());
    // The channel the caller wrote for a rendered (post-flip) one.
    let authored = |wire: &str| -> Option<PositionalChannel> {
        let rendered = PositionalChannel::from_wire(wire)?;
        Some(match (rendered, flips) {
            (PositionalChannel::X, true) => PositionalChannel::Y,
            (PositionalChannel::Y, true) => PositionalChannel::X,
            (PositionalChannel::X2, true) => PositionalChannel::Y2,
            (PositionalChannel::Y2, true) => PositionalChannel::X2,
            (channel, false) => channel,
        })
    };
    let binds_itself = |wire: &str| match (raw, authored(wire)) {
        (None, _) => true,
        (Some(raw), Some(channel)) => raw.own_positional().contains(&channel),
        (Some(_), None) => false,
    };
    let inherits = |wire: &str| match (raw, authored(wire)) {
        (Some(raw), Some(channel)) => {
            raw.inherited_positional.contains(&channel)
                || (channel.slot(&raw.encoding).is_none() && channel.slot(&spec.encoding).is_some())
        }
        _ => false,
    };

    let own_positions: Vec<String> = ["x", "y", "x2", "y2"]
        .into_iter()
        .filter(|wire| binds_itself(wire))
        .filter_map(|wire| {
            let binding = match wire {
                "x" => &drawn.x,
                "y" => &drawn.y,
                "x2" => &drawn.x2,
                _ => &drawn.y2,
            };
            binding.as_ref().map(|b| b.field.clone())
        })
        .collect();

    let opposite_of = |wire: &str| -> Opposite {
        let opposite = if wire == "x" { "y" } else { "x" };
        let binding = if opposite == "x" { &drawn.x } else { &drawn.y };
        match binding {
            Some(enc) => Opposite::Bound(enc.field.clone()),
            None if inherits(opposite) => Opposite::Inherited,
            None => Opposite::Absent,
        }
    };

    let mut fields: Vec<KeyField> = Vec::new();
    let mut push = |field: &str, kind: KeyKind| {
        if !pooled.contains(field) && batch.column_by_name(field).is_some() && !fields.iter().any(|k| k.field == field) {
            fields.push(KeyField { field: field.to_string(), kind });
        }
    };
    let dtype_of = |field: &str| batch.column_by_name(field).map(|c| c.data_type().clone());
    for (wire, enc) in drawn.bindings() {
        let Some(dtype) = dtype_of(&enc.field) else { continue };
        match wire {
            "x" | "y" if reads_categorical(enc, &dtype) => push(&enc.field, KeyKind::Category(opposite_of(wire))),
            "color" if reads_categorical(enc, &dtype) || enc.color_bin().is_some() => push(&enc.field, KeyKind::Grouping),
            // A shape binding draws a symbol per level whatever the field's
            // dtype (a numeric field's values are its levels).
            "shape" => push(&enc.field, KeyKind::Grouping),
            // A dash binding is a key where the mark draws levels from the
            // field: a categorical field on any mark that draws the channel,
            // a numeric index on the marks that dash each row by it. A series
            // mark (line, ribbon) splits series by categorical dashes only and
            // reads a numeric one from a group's first row, so a null there
            // draws nothing the drop would remove.
            "stroke_dash" if reads_categorical(enc, &dtype) || dashes_each_row_by_index(layer.mark) => {
                push(&enc.field, KeyKind::Grouping)
            }
            _ => {}
        }
    }
    if let Some(detail) = layer.mark_style.as_ref().and_then(|m| m.detail.as_deref()).filter(|_| layer.mark.style_field("detail").is_some()) {
        push(detail, KeyKind::Grouping);
    }
    // Positional keys first, then the grouping ones, as the gate reads them.
    fields.sort_by_key(|key| !matches!(key.kind, KeyKind::Category(_)));
    LayerKeys { fields, own_positions }
}

/// A column's null mask, `logical_nulls`-aware: a `Null`-typed column carries
/// no validity buffer and reports `null_count() == 0`.
fn null_mask(batch: &RecordBatch, field: &str) -> Option<Vec<bool>> {
    let column = batch.column_by_name(field)?;
    let nulls = column.logical_nulls()?;
    (nulls.null_count() > 0).then(|| (0..column.len()).map(|i| nulls.is_null(i)).collect())
}

/// The rows of `batch` a layer with `keys` counts as null-keyed, or `None`
/// when it counts none.
fn counted_rows(batch: &RecordBatch, keys: &LayerKeys) -> Option<Vec<bool>> {
    let n = batch.num_rows();
    let own_positions: Vec<Option<Vec<bool>>> = keys.own_positions.iter().map(|f| null_mask(batch, f)).collect();
    let position_present = |row: usize| own_positions.iter().all(|mask| mask.as_ref().is_none_or(|m| !m[row]));
    let mut counted = vec![false; n];
    for key in &keys.fields {
        let Some(nulls) = null_mask(batch, &key.field) else { continue };
        let opposite_mask = match &key.kind {
            KeyKind::Category(Opposite::Bound(field)) => null_mask(batch, field),
            _ => None,
        };
        for (row, is_null) in nulls.iter().enumerate() {
            if !is_null {
                continue;
            }
            counted[row] |= match &key.kind {
                KeyKind::Category(Opposite::Bound(_)) => opposite_mask.as_ref().is_none_or(|m| !m[row]),
                KeyKind::Category(Opposite::Absent) => true,
                KeyKind::Category(Opposite::Inherited) => false,
                KeyKind::Grouping => position_present(row),
            };
        }
    }
    counted.contains(&true).then_some(counted)
}

/// `batch` filtered to the rows `keep` selects.
fn filter_rows(batch: &RecordBatch, keep: BooleanArray) -> Result<RecordBatch, RenderError> {
    arrow::compute::filter_record_batch(batch, &keep)
        .map_err(|e| RenderError::ScaleResolutionFailed(format!("null-key drop: {e}")))
}

/// `batch` without the rows `counted` marks.
fn kept(batch: &RecordBatch, counted: &[bool]) -> Result<RecordBatch, RenderError> {
    filter_rows(batch, counted.iter().map(|c| Some(!c)).collect())
}

/// `batch`'s rows `counted` marks.
fn dropped(batch: &RecordBatch, counted: &[bool]) -> Result<RecordBatch, RenderError> {
    filter_rows(batch, counted.iter().map(|c| Some(*c)).collect())
}

/// The facet partitions of `batch` every row of which `counted` marks: their
/// panels draw empty rather than vanish.
fn emptied_partitions(spec: &ChartSpec, batch: &RecordBatch, counted: &[bool]) -> Result<Vec<FacetPartitionKey>, RenderError> {
    let Some(fspec) = &spec.facet else { return Ok(Vec::new()) };
    let col = crate::render::facet_key_column(batch, &fspec.field)?;
    let row = fspec.row.as_deref().map(|field| crate::render::facet_key_column(batch, field)).transpose()?;
    let mut totals: HashMap<FacetPartitionKey, (usize, usize)> = HashMap::new();
    let mut order: Vec<FacetPartitionKey> = Vec::new();
    for (i, &is_counted) in counted.iter().enumerate() {
        let Some(col_value) = col.is_valid(i).then(|| col.value(i).to_string()) else { continue };
        let row_value = match &row {
            Some(rows) => match rows.is_valid(i).then(|| rows.value(i).to_string()) {
                Some(value) => Some(value),
                None => continue,
            },
            None => None,
        };
        let key = (col_value, row_value);
        let entry = totals.entry(key.clone()).or_insert_with(|| {
            order.push(key);
            (0, 0)
        });
        entry.0 += 1;
        entry.1 += usize::from(is_counted);
    }
    Ok(order.into_iter().filter(|key| totals.get(key).is_some_and(|(total, counted)| total == counted)).collect())
}

/// Drop every layer's null-key rows, reporting each layer's drop once.
///
/// Each layer reports its own drop, once, in layer order, with no folding
/// across layers: two layers that drop the same rows of one frame (a
/// `Chart.layer` of two points) raise one line each, as statistic layers and
/// composite leaves do, so the number of lines never depends on what else a
/// layer keys.
///
/// `outputs` holds the pipeline's published batches (the tail at
/// [`FINAL_OUTPUT_KEY`] included) and is updated in place; a layer the drop
/// gives rows of its own has `own_rows` set and its rows published under
/// [`layer_key`], with its source's lineage copied beside them. `facet`
/// gains the partitions the drop emptied.
pub(super) fn drop_null_key_rows(
    spec: &ChartSpec,
    layers: &mut [LayerPrepared],
    outputs: &mut HashMap<String, RecordBatch>,
    lineages: &mut OutputLineages,
    facet: &mut FacetPartitions,
) -> Result<NullKeyDrop, RenderError> {
    let mut result = NullKeyDrop::default();

    // Each layer's keys and counted rows over the batch of the key it reads.
    struct Judged {
        key: String,
        fields: Vec<KeyField>,
        counted: Option<Vec<bool>>,
    }
    let mut judged: Vec<Judged> = Vec::with_capacity(layers.len());
    for (index, layer) in layers.iter().enumerate() {
        let key = layer.data_source.clone().unwrap_or_else(|| FINAL_OUTPUT_KEY.to_string());
        let Some(batch) = outputs.get(&key) else {
            judged.push(Judged { key, fields: Vec::new(), counted: None });
            continue;
        };
        let keys = layer_key_fields(spec, index, layer, batch, &pooled_fields(lineages.get(&key)));
        let has_null = keys.fields.iter().any(|k| null_mask(batch, &k.field).is_some());
        let counted = if has_null { counted_rows(batch, &keys) } else { None };
        judged.push(Judged { key, fields: keys.fields, counted });
    }
    if judged.iter().all(|j| j.counted.is_none()) {
        return Ok(result);
    }

    // The rows each key held before any layer's drop.
    let originals: HashMap<String, RecordBatch> =
        judged.iter().filter(|j| j.counted.is_some()).filter_map(|j| outputs.get(&j.key).map(|b| (j.key.clone(), b.clone()))).collect();

    // Per key, in layer order: the first layer keeps the key, and a later one
    // whose counted rows differ reads rows of its own.
    let mut seen_keys: Vec<&str> = Vec::new();
    for j in &judged {
        if !seen_keys.contains(&j.key.as_str()) {
            seen_keys.push(&j.key);
        }
    }
    for key in seen_keys {
        let members: Vec<usize> = judged.iter().enumerate().filter(|(_, j)| j.key == key).map(|(i, _)| i).collect();
        if members.iter().all(|&i| judged[i].counted.is_none()) {
            continue;
        }
        let Some(original) = originals.get(key) else { continue };
        let first = members[0];
        if let Some(counted) = &judged[first].counted {
            outputs.insert(key.to_string(), kept(original, counted)?);
        }
        for &i in &members[1..] {
            if judged[i].counted == judged[first].counted {
                continue;
            }
            let rows = match &judged[i].counted {
                Some(counted) => kept(original, counted)?,
                None => original.clone(),
            };
            let own = layer_key(i);
            outputs.insert(own.clone(), rows);
            if let Some(lineage) = lineages.get(key).cloned() {
                lineages.insert(own.clone(), lineage);
            }
            layers[i].own_rows = Some(own.clone());
            result.routes.push((i, own));
        }
    }

    // One report per layer, in layer order, beside the panels it emptied.
    let ctx = TransformContext { facet_fields: facet_fields(spec), ..TransformContext::default() };
    for (index, j) in judged.iter().enumerate() {
        let (Some(counted), Some(original)) = (&j.counted, originals.get(&j.key)) else { continue };
        let entry = layer_entry(spec, index, layers[index].mark);
        // The key fields the dropped rows are null on: what the report names.
        let dropped_rows = dropped(original, counted)?;
        let fields: Vec<String> = j
            .fields
            .iter()
            .filter(|k| null_mask(&dropped_rows, &k.field).is_some())
            .map(|k| k.field.clone())
            .collect();
        let n_counted = counted.iter().filter(|c| **c).count();
        report_skipped_null_keys(
            &dropped_rows,
            &fields,
            &entry,
            NullKeyPolicy::Skip,
            NullKeyCounts { any_null: n_counted, fully_null: 0, n_rows: original.num_rows() },
            &ctx,
        );
        facet.emptied_by_drop.extend(emptied_partitions(spec, original, counted)?);
    }
    for warning in ctx.warnings.borrow_mut().drain(..) {
        let warning = RenderWarning::Transform(warning);
        result.warnings.push(warning);
    }
    Ok(result)
}

/// Give each layer on the chart's rows that keys a grouping field the impute's
/// `series` do not carry the rows from before the impute (`pre_impute`, which
/// are the rows the drop kept): the rows the impute synthesises are null on
/// that field, so such a layer must not draw them. The layer reads them through
/// the same `own_rows` route a layer whose counted rows differ takes. Returns
/// the new routes.
pub(super) fn keep_pre_impute_rows(
    spec: &ChartSpec,
    layers: &mut [LayerPrepared],
    outputs: &mut HashMap<String, RecordBatch>,
    lineages: &mut OutputLineages,
    pre_impute: &RecordBatch,
    series: &[&str],
) -> Vec<(usize, String)> {
    let mut routes = Vec::new();
    for (index, layer) in layers.iter_mut().enumerate() {
        if layer.data_source.is_some() || layer.own_rows.is_some() {
            continue;
        }
        let keys = layer_key_fields(spec, index, layer, pre_impute, &HashSet::new());
        let keys_beyond_series =
            keys.fields.iter().any(|key| key.kind == KeyKind::Grouping && !series.contains(&key.field.as_str()));
        if !keys_beyond_series {
            continue;
        }
        let own = layer_key(index);
        outputs.insert(own.clone(), pre_impute.clone());
        if let Some(lineage) = lineages.get(FINAL_OUTPUT_KEY).cloned() {
            lineages.insert(own.clone(), lineage);
        }
        layer.own_rows = Some(own.clone());
        routes.push((index, own));
    }
    routes
}

/// The chart's facet fields, which a count never names: they are constant and
/// non-null within a panel.
fn facet_fields(spec: &ChartSpec) -> Vec<String> {
    spec.facet.as_ref().map(|f| std::iter::once(f.field.clone()).chain(f.row.clone()).collect()).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::prepare::prepare_render_inputs;
    use arrow::array::{Float64Array, NullArray, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn spec(json: &str) -> ChartSpec {
        serde_json::from_str(json).expect("the fixture spec deserializes")
    }

    fn strings(values: &[Option<&str>]) -> Arc<dyn Array> {
        Arc::new(StringArray::from(values.to_vec()))
    }

    fn floats(values: &[Option<f64>]) -> Arc<dyn Array> {
        Arc::new(Float64Array::from(values.to_vec()))
    }

    fn batch(columns: Vec<(&str, Arc<dyn Array>)>) -> RecordBatch {
        let fields: Vec<Field> = columns.iter().map(|(name, array)| Field::new(*name, array.data_type().clone(), true)).collect();
        RecordBatch::try_new(Arc::new(Schema::new(fields)), columns.into_iter().map(|(_, array)| array).collect()).unwrap()
    }

    fn prepared(spec: &ChartSpec, batch: &RecordBatch) -> crate::render::prepare::PreparedInputs {
        prepare_render_inputs(
            spec,
            batch,
            &crate::layout::ThemeInputs::default(),
            crate::layout::Viewport { width: 600.0, height: 400.0 },
            &crate::render::chart_config::ChartConfig::default(),
            None,
        )
        .expect("the fixture prepares")
    }

    /// Every drop warning a render's prepare raised, as the text a caller reads.
    fn drops(prep: &crate::render::prepare::PreparedInputs) -> Vec<String> {
        prep.warnings
            .iter()
            .filter_map(|w| match w {
                RenderWarning::Transform(t) => Some(t.to_string()),
                _ => None,
            })
            .collect()
    }

    const POINT_CAT_VAL: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"c","type":"nominal"},"y":{"field":"v"}}}"#;

    fn cat_batch(cats: &[Option<&str>], vals: &[Option<f64>]) -> RecordBatch {
        batch(vec![("c", strings(cats)), ("v", floats(vals))])
    }

    #[test]
    fn a_null_category_row_with_a_value_is_counted_and_dropped() {
        let b = cat_batch(&[Some("a"), Some("b"), None], &[Some(1.0), Some(2.0), Some(3.0)]);
        let prep = prepared(&spec(POINT_CAT_VAL), &b);
        assert_eq!(prep.final_batch().num_rows(), 2);
        assert_eq!(
            drops(&prep),
            ["mark_point: 1 row has a null 'c' and belongs to no group, so it was dropped; fill or filter the nulls in 'c'"]
        );
        assert!(prep.layer_routes.is_empty(), "one layer on one key replaces the key in place");
    }

    #[test]
    fn a_null_category_row_with_a_null_value_is_not_counted() {
        let b = cat_batch(&[Some("a"), None, Some("b")], &[Some(1.0), None, Some(3.0)]);
        let prep = prepared(&spec(POINT_CAT_VAL), &b);
        assert_eq!(prep.final_batch().num_rows(), 3, "a row null on both channels places nothing and is the census's");
        assert!(drops(&prep).is_empty(), "got {:?}", drops(&prep));
    }

    #[test]
    fn a_one_axis_layer_counts_every_null_category_row() {
        let b = batch(vec![("c", strings(&[Some("a"), None, None]))]);
        let json = r#"{"data":{"kind":"named","name":"default"},"mark":"tick","encoding":{"x":{"field":"c","type":"nominal"}}}"#;
        let prep = prepared(&spec(json), &b);
        assert_eq!(prep.final_batch().num_rows(), 1);
        assert_eq!(
            drops(&prep),
            ["mark_tick: 2 rows have a null 'c' and belong to no group, so they were dropped; fill or filter the nulls in 'c'"]
        );
    }

    #[test]
    fn a_reference_layer_with_an_inherited_coordinate_counts_nothing() {
        let b = batch(vec![
            ("c", strings(&[Some("a"), Some("b")])),
            ("v", floats(&[Some(1.0), Some(2.0)])),
            ("ref", strings(&[Some("b"), None])),
        ]);
        let json = r#"{"data":{"kind":"named","name":"default"},"mark":"bar","encoding":{"x":{"field":"c","type":"nominal"},"y":{"field":"v"}},"layers":[{"mark":"bar","encoding":{"x":{"field":"c","type":"nominal"},"y":{"field":"v"}}},{"mark":"rule","encoding":{"x":{"field":"ref","type":"nominal"},"y":{"field":"v"}},"inherited_positional":["y"]}]}"#;
        let prep = prepared(&spec(json), &b);
        assert_eq!(prep.final_batch().num_rows(), 2, "the structural null drops nothing");
        assert!(drops(&prep).is_empty(), "got {:?}", drops(&prep));
    }

    #[test]
    fn a_null_grouping_value_on_a_placed_row_is_counted() {
        let b = batch(vec![
            ("t", floats(&[Some(1.0), Some(2.0), Some(3.0)])),
            ("v", floats(&[Some(1.0), Some(2.0), Some(3.0)])),
            ("h", strings(&[Some("x"), None, Some("y")])),
        ]);
        let json = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"t"},"y":{"field":"v"},"color":{"field":"h","type":"nominal"}}}"#;
        let prep = prepared(&spec(json), &b);
        assert_eq!(prep.final_batch().num_rows(), 2);
        assert_eq!(
            drops(&prep),
            ["mark_point: 1 row has a null 'h' and belongs to no group, so it was dropped; fill or filter the nulls in 'h'"]
        );
    }

    #[test]
    fn a_null_grouping_value_on_an_unplaced_row_is_the_censuss() {
        let b = batch(vec![
            ("t", floats(&[Some(1.0), Some(2.0)])),
            ("v", floats(&[Some(1.0), None])),
            ("h", strings(&[Some("x"), None])),
        ]);
        let json = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"t"},"y":{"field":"v"},"color":{"field":"h","type":"nominal"}}}"#;
        let prep = prepared(&spec(json), &b);
        assert_eq!(prep.final_batch().num_rows(), 2);
        assert!(drops(&prep).is_empty());
    }

    #[test]
    fn an_arcs_null_colour_slice_is_counted() {
        let b = batch(vec![("n", floats(&[Some(1.0), Some(2.0), Some(3.0)])), ("h", strings(&[Some("x"), None, Some("y")]))]);
        let json = r#"{"data":{"kind":"named","name":"default"},"mark":"arc","encoding":{"x":{"field":"n"},"color":{"field":"h","type":"nominal"}},"coord":{"kind":"polar","theta":"x","start_angle":0.0,"direction":"clockwise","inner_radius":0.0,"pad_angle":0.0}}"#;
        let prep = prepared(&spec(json), &b);
        assert_eq!(prep.final_batch().num_rows(), 2);
        assert_eq!(
            drops(&prep),
            ["mark_arc: 1 row has a null 'h' and belongs to no group, so it was dropped; fill or filter the nulls in 'h'"]
        );
    }

    #[test]
    fn a_geoshapes_colour_is_not_a_key() {
        let b = batch(vec![("h", strings(&[Some("x"), None]))]);
        let json = r#"{"data":{"kind":"named","name":"default"},"mark":"geoshape","encoding":{"color":{"field":"h","type":"nominal"}}}"#;
        let layers = super::super::build_layers(&spec(json)).unwrap();
        let keys = layer_key_fields(&spec(json), 0, &layers[0], &b, &HashSet::new());
        assert!(keys.fields.is_empty(), "geoshape draws no colour, so a binding on it is no key: {:?}", keys.fields);
    }

    #[test]
    fn a_quantitative_colour_and_the_value_channels_are_not_keys() {
        let b = batch(vec![
            ("t", floats(&[Some(1.0), Some(2.0)])),
            ("v", floats(&[Some(1.0), Some(2.0)])),
            ("q", floats(&[Some(1.0), None])),
        ]);
        let json = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"t"},"y":{"field":"v"},"color":{"field":"q","type":"quantitative"},"size":{"field":"q","type":"quantitative"}}}"#;
        let prep = prepared(&spec(json), &b);
        assert_eq!(prep.final_batch().num_rows(), 2);
        assert!(drops(&prep).is_empty());
    }

    #[test]
    fn detail_is_a_key_on_a_line_and_not_on_a_point() {
        let b = batch(vec![
            ("t", floats(&[Some(1.0), Some(2.0), Some(3.0)])),
            ("v", floats(&[Some(1.0), Some(2.0), Some(3.0)])),
            ("d", strings(&[Some("p"), None, Some("p")])),
        ]);
        for (mark, dropped) in [("line", 1), ("point", 0)] {
            let json = format!(
                r#"{{"data":{{"kind":"named","name":"default"}},"mark":"{mark}","encoding":{{"x":{{"field":"t"}},"y":{{"field":"v"}}}},"mark_style":{{"detail":"d"}}}}"#
            );
            let prep = prepared(&spec(&json), &b);
            assert_eq!(prep.final_batch().num_rows(), 3 - dropped, "{mark}");
        }
    }

    #[test]
    fn a_null_typed_nominal_column_is_counted_through_logical_nulls() {
        let n = 3;
        let b = batch(vec![("v", floats(&[Some(1.0), Some(2.0), Some(3.0)])), ("c", Arc::new(NullArray::new(n)) as Arc<dyn Array>)]);
        let prep = prepared(&spec(POINT_CAT_VAL), &b);
        assert_eq!(prep.final_batch().num_rows(), 0);
        assert_eq!(
            drops(&prep),
            ["mark_point: every row has a null in its groupby key (c), so the data could not be split into groups; the grouped output is empty"]
        );
    }

    #[test]
    fn every_row_null_keyed_gives_the_all_null_sentence_naming_every_key_field() {
        let b = batch(vec![
            ("c", strings(&[None, Some("a")])),
            ("h", strings(&[Some("x"), None])),
            ("v", floats(&[Some(1.0), Some(2.0)])),
        ]);
        let json = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"c","type":"nominal"},"y":{"field":"v"},"color":{"field":"h","type":"nominal"}}}"#;
        let prep = prepared(&spec(json), &b);
        assert_eq!(prep.final_batch().num_rows(), 0);
        assert_eq!(
            drops(&prep),
            ["mark_point: every row has a null in its groupby key (c, h), so the data could not be split into groups; the grouped output is empty"]
        );
    }

    #[test]
    fn a_row_null_on_both_fields_is_counted_under_each() {
        let b = batch(vec![
            ("c", strings(&[None, None, Some("a"), Some("a")])),
            ("h", strings(&[None, Some("x"), None, Some("y")])),
            ("v", floats(&[Some(1.0), Some(2.0), Some(3.0), Some(4.0)])),
        ]);
        let json = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"c","type":"nominal"},"y":{"field":"v"},"color":{"field":"h","type":"nominal"}}}"#;
        let prep = prepared(&spec(json), &b);
        assert_eq!(prep.final_batch().num_rows(), 1);
        assert_eq!(
            drops(&prep),
            [
                "mark_point: 2 rows have a null 'c' and belong to no group, so they were dropped; fill or filter the nulls in 'c'",
                "mark_point: 2 rows have a null 'h' and belong to no group, so they were dropped; fill or filter the nulls in 'h'",
            ]
        );
    }

    #[test]
    fn a_fallback_grouped_field_is_exempt_and_a_pool_grouped_one_is_not() {
        let b = batch(vec![("c", strings(&[Some("a")])), ("v", floats(&[Some(1.0)]))]);
        let s = spec(POINT_CAT_VAL);
        let layers = super::super::build_layers(&s).unwrap();
        let exempt: HashSet<String> = HashSet::from(["c".to_string()]);
        assert!(layer_key_fields(&s, 0, &layers[0], &b, &exempt).fields.is_empty());
        assert_eq!(layer_key_fields(&s, 0, &layers[0], &b, &HashSet::new()).fields.len(), 1);

        use crate::transform::core::TransformSpec;
        let transform = |json: serde_json::Value| -> TransformSpec { serde_json::from_value(json).expect("a transform spec") };
        assert!(transform(serde_json::json!({"type": "smooth", "x": "x", "y": "y", "method": "lm", "bandwidth": 0.75, "degree": 1, "n": 4, "seed": 0, "output": "fitted", "inject_zero_ref": false, "inject_metrics": false, "groupby": ["h"]})).pools_null_keys());
        assert!(
            !transform(serde_json::json!({"type": "data_window", "ops": [{"op": "rank", "as": "r"}], "sort": ["v"], "groupby": ["h"]}))
                .pools_null_keys(),
            "a row-preserving Pool step keeps its null keys on purpose and exempts nothing"
        );
    }

    #[test]
    fn a_faceted_panel_the_drop_empties_is_recorded() {
        let b = batch(vec![
            ("c", strings(&[Some("a"), Some("a"), None, None])),
            ("v", floats(&[Some(1.0), Some(2.0), Some(3.0), Some(4.0)])),
            ("f", strings(&[Some("F1"), Some("F1"), Some("F2"), Some("F2")])),
        ]);
        let mut s = spec(POINT_CAT_VAL);
        s.facet = Some(crate::layout::FacetSpec {
            field: "f".into(),
            row: None,
            title: None,
            row_title: None,
            sort: None,
            row_sort: None,
            repeat: Vec::new(),
            mode: crate::layout::FacetMode::Wrap { ncols: 2 },
            spacing: None,
            resolve: crate::layout::facet::FacetResolve::default(),
        });
        let prep = prepared(&s, &b);
        assert!(prep.facet_emptied_by_drop.contains(&("F2".to_string(), None)));
        assert!(!prep.facet_emptied_by_drop.contains(&("F1".to_string(), None)));
        assert_eq!(prep.facet_groups.len(), 2, "the emptied panel keeps its place");
    }

    const TWO_LAYERS_SAME_KEYS: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"c","type":"nominal"},"y":{"field":"v"}},"layers":[{"mark":"point","encoding":{"x":{"field":"c","type":"nominal"},"y":{"field":"v"}}},{"mark":"line","encoding":{"x":{"field":"c","type":"nominal"},"y":{"field":"v"}}}]}"#;

    #[test]
    fn layers_keying_the_same_fields_replace_the_key_in_place_and_each_reports() {
        let b = cat_batch(&[Some("a"), Some("b"), None], &[Some(1.0), Some(2.0), Some(3.0)]);
        let prep = prepared(&spec(TWO_LAYERS_SAME_KEYS), &b);
        assert!(prep.layer_routes.is_empty());
        assert_eq!(prep.final_batch().num_rows(), 2);
        let texts = drops(&prep);
        assert_eq!(texts.len(), 2, "each layer reports its own count in its own voice: {texts:?}");
        assert!(texts[0].starts_with("mark_point: 1 row"));
        assert!(texts[1].starts_with("mark_line: 1 row"));
    }

    const TWO_LAYERS_DIFFERENT_KEYS: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"c","type":"nominal"},"y":{"field":"v"}},"layers":[{"mark":"point","encoding":{"x":{"field":"c","type":"nominal"},"y":{"field":"v"}}},{"mark":"line","encoding":{"y":{"field":"w"},"color":{"field":"h","type":"nominal"}}}]}"#;

    fn disagreeing_batch() -> RecordBatch {
        batch(vec![
            ("c", strings(&[Some("a"), Some("b"), Some("a"), Some("b")])),
            ("v", floats(&[Some(1.0), Some(2.0), Some(3.0), Some(4.0)])),
            ("w", floats(&[Some(10.0), Some(20.0), Some(30.0), Some(1000.0)])),
            ("h", strings(&[Some("x"), Some("x"), Some("y"), None])),
        ])
    }

    #[test]
    fn layers_keying_different_fields_give_the_later_one_its_own_rows() {
        let s = spec(TWO_LAYERS_DIFFERENT_KEYS);
        let prep = prepared(&s, &disagreeing_batch());
        assert_eq!(prep.layer_routes, vec![(1, layer_key(1))]);
        assert_eq!(prep.final_batch().num_rows(), 4, "the point layer keeps the key and every row it draws");
        assert_eq!(prep.transform_outputs[&layer_key(1)].num_rows(), 3, "the line's own rows lose its null-h row");
        assert_eq!(prep.layers[1].data_source, None, "the layer's source is untouched");
        assert_eq!(prep.layers[1].own_rows.as_deref(), Some(layer_key(1).as_str()));
        assert_eq!(
            drops(&prep),
            ["mark_line: 1 row has a null 'h' and belongs to no group, so it was dropped; fill or filter the nulls in 'h'"]
        );
        let routed = prep.routed_spec(&s);
        let layers = routed.layers.as_ref().unwrap();
        assert_eq!(layers[1].own_rows.as_deref(), Some(layer_key(1).as_str()));
        assert_eq!(layers[1].data_source, None);
        assert!(
            layers[1].completed_encoding(&routed.encoding).x.is_some(),
            "the moved layer still inherits the chart's x, as a chart-data layer does"
        );
    }

    #[test]
    fn the_same_disagreement_on_an_operand_key_moves_the_later_layer_and_keeps_its_source() {
        let json = TWO_LAYERS_DIFFERENT_KEYS
            .replace(r#"{"mark":"point","encoding":{"x":{"field":"c","type":"nominal"},"y":{"field":"v"}}},{"mark":"line""#, r#"{"mark":"point","encoding":{"x":{"field":"c","type":"nominal"},"y":{"field":"v"}},"data_source":"_operand_0"},{"mark":"line","data_source":"_operand_0""#);
        let s = spec(&json);
        let b = disagreeing_batch();
        // Publish the operand's rows as the pipeline would: the same rows the tail holds.
        let layers = s.layers.as_ref().unwrap();
        assert_eq!(layers[0].data_source.as_deref(), Some("_operand_0"));
        let mut outputs = HashMap::from([(FINAL_OUTPUT_KEY.to_string(), b.clone()), ("_operand_0".to_string(), b.clone())]);
        let mut prepared_layers = super::super::build_layers(&s).unwrap();
        let mut lineages = OutputLineages::new();
        let mut facet = FacetPartitions::default();
        let drop = drop_null_key_rows(&s, &mut prepared_layers, &mut outputs, &mut lineages, &mut facet).unwrap();
        assert_eq!(drop.routes, vec![(1, layer_key(1))]);
        assert_eq!(outputs["_operand_0"].num_rows(), 4);
        assert_eq!(outputs[&layer_key(1)].num_rows(), 3);
        assert_eq!(prepared_layers[1].data_source.as_deref(), Some("_operand_0"));
        assert_eq!(prepared_layers[1].rows_key(), layer_key(1));
        assert_eq!(outputs[FINAL_OUTPUT_KEY].num_rows(), 4, "an unrelated key is not touched");
    }

    #[test]
    fn a_chart_with_no_null_key_touches_no_batch_and_borrows_its_spec() {
        let s = spec(TWO_LAYERS_DIFFERENT_KEYS);
        let b = batch(vec![
            ("c", strings(&[Some("a"), Some("b")])),
            ("v", floats(&[Some(1.0), Some(2.0)])),
            ("w", floats(&[Some(1.0), Some(2.0)])),
            ("h", strings(&[Some("x"), Some("y")])),
        ]);
        let prep = prepared(&s, &b);
        assert!(prep.layer_routes.is_empty());
        assert!(drops(&prep).is_empty());
        assert!(matches!(prep.routed_spec(&s), std::borrow::Cow::Borrowed(_)));
        assert_eq!(prep.final_batch().num_rows(), 2);
    }

    #[test]
    fn a_literal_null_string_is_a_category_and_a_true_null_is_dropped_beside_it() {
        let b = cat_batch(&[Some("null"), None, Some("a")], &[Some(1.0), Some(2.0), Some(3.0)]);
        let prep = prepared(&spec(POINT_CAT_VAL), &b);
        assert_eq!(prep.final_batch().num_rows(), 2);
        let kept = prep.final_batch().column_by_name("c").unwrap().as_any().downcast_ref::<StringArray>().unwrap().iter().collect::<Vec<_>>();
        assert_eq!(kept, vec![Some("null"), Some("a")]);
    }

    #[test]
    fn a_nan_category_key_is_its_own_key_and_not_a_null() {
        let b = batch(vec![("c", floats(&[Some(1.0), Some(f64::NAN), None])), ("v", floats(&[Some(1.0), Some(2.0), Some(3.0)]))]);
        let prep = prepared(&spec(POINT_CAT_VAL), &b);
        assert_eq!(prep.final_batch().num_rows(), 2, "only the true null is dropped");
    }

    #[test]
    fn a_numeric_shape_or_dash_index_is_a_key_where_the_mark_draws_levels_from_it() {
        let b = batch(vec![
            ("t", floats(&[Some(1.0), Some(2.0), Some(3.0)])),
            ("v", floats(&[Some(1.0), Some(2.0), Some(3.0)])),
            ("k", floats(&[Some(0.0), None, Some(1.0)])),
        ]);
        let json = |mark: &str, channel: &str| {
            format!(
                r#"{{"data":{{"kind":"named","name":"default"}},"mark":"{mark}","encoding":{{"x":{{"field":"t"}},"y":{{"field":"v"}},"{channel}":{{"field":"k"}}}}}}"#
            )
        };
        for (mark, channel, kept) in [
            ("point", "shape", 2),
            ("point", "stroke_dash", 2),
            ("bar", "stroke_dash", 2),
            ("rect", "stroke_dash", 2),
            ("rule", "stroke_dash", 2),
            ("line", "stroke_dash", 3),
            ("tick", "stroke_dash", 3),
        ] {
            let prep = prepared(&spec(&json(mark, channel)), &b);
            assert_eq!(prep.final_batch().num_rows(), kept, "{mark} {channel}");
            if kept == 2 {
                assert_eq!(
                    drops(&prep),
                    [format!("mark_{mark}: 1 row has a null 'k' and belongs to no group, so it was dropped; fill or filter the nulls in 'k'")]
                );
            } else {
                assert!(drops(&prep).is_empty(), "{mark} counts nothing for a numeric dash: {:?}", drops(&prep));
            }
        }
    }

    #[test]
    fn each_layer_reports_its_own_drop_with_no_folding_across_layers() {
        let b = cat_batch(&[Some("a"), Some("b"), None], &[Some(1.0), Some(2.0), Some(3.0)]);
        let two_points = TWO_LAYERS_SAME_KEYS.replace(r#"{"mark":"line","#, r#"{"mark":"point","#);
        let identical = drops(&prepared(&spec(&two_points), &b));
        assert_eq!(identical.len(), 2, "two identical layers report twice: {identical:?}");
        assert_eq!(identical[0], identical[1]);
        assert_eq!(drops(&prepared(&spec(TWO_LAYERS_SAME_KEYS), &b)).len(), 2, "mark_point and mark_line each report");
    }

    /// The all-null sentence names the key fields the dropped rows are null
    /// on: `c` holds no null here, so it is not named.
    #[test]
    fn the_all_null_sentence_names_only_the_fields_the_dropped_rows_are_null_on() {
        let b = batch(vec![
            ("c", strings(&[Some("a"), Some("b")])),
            ("h", strings(&[None, None])),
            ("v", floats(&[Some(1.0), Some(2.0)])),
        ]);
        let json = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"c","type":"nominal"},"y":{"field":"v"},"color":{"field":"h","type":"nominal"}}}"#;
        assert_eq!(
            drops(&prepared(&spec(json), &b)),
            ["mark_point: every row has a null in its groupby key (h), so the data could not be split into groups; the grouped output is empty"]
        );
    }

    /// A field a `Smooth` step grouped under its pooled fallback keeps its null
    /// on purpose: a layer drawing the fit's output coloured by that field
    /// keeps the pooled fit's null-keyed rows, through the lineage the real run
    /// records.
    #[test]
    fn a_layer_on_a_pooled_fits_output_keeps_the_fits_null_key() {
        let smooth = r#"{"type":"smooth","x":"x","y":"y","method":"lm","bandwidth":0.75,"degree":1,"n":4,"seed":0,"output":"fitted","inject_zero_ref":false,"inject_metrics":false,"groupby":["h"]}"#;
        let json = format!(
            r#"{{"data":{{"kind":"named","name":"default"}},"mark":"line","transforms":[{smooth}],"encoding":{{"x":{{"field":"x"}},"y":{{"field":"y"}},"color":{{"field":"h","type":"nominal"}}}}}}"#
        );
        let b = batch(vec![
            ("x", floats(&[Some(1.0), Some(2.0), Some(3.0)])),
            ("y", floats(&[Some(1.0), Some(2.0), Some(4.0)])),
            ("h", strings(&[None, None, None])),
        ]);
        let prep = prepared(&spec(&json), &b);
        assert!(prep.final_batch().num_rows() > 0, "the pooled fit's rows are kept");
        assert!(
            drops(&prep).iter().all(|text| !text.starts_with("mark_line:")),
            "the mark drops nothing from a pooled fit: {:?}",
            drops(&prep)
        );
    }

    /// The lineage a moved layer's key carries is copied under its own key, so
    /// the readers that follow a layer's rows key still find the statistic's
    /// consumed rows: a data-aware sort of a moved operand layer reads them.
    #[test]
    fn a_moved_layer_keeps_the_statistic_lineage_of_its_source() {
        use crate::transform::run::OutputLineage;
        let rows = batch(vec![
            ("c", strings(&[Some("a"), Some("b"), Some("a")])),
            ("v", floats(&[Some(1.0), Some(2.0), Some(3.0)])),
            ("h", strings(&[Some("x"), None, Some("x")])),
        ]);
        let consumed = batch(vec![("c", strings(&[Some("a")])), ("raw", floats(&[Some(9.0)]))]);
        let source = OutputLineage {
            swarm: None,
            statistic_input: Some(consumed.clone()),
            lowered: Vec::new(),
            written_groupby: Vec::new(),
            source: Arc::new(Schema::empty()),
            log_input: HashMap::new(),
            steps: Vec::new(),
        };
        let json = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"c","type":"nominal"},"y":{"field":"v"}},"layers":[{"mark":"point","encoding":{"x":{"field":"c","type":"nominal"},"y":{"field":"v"}},"data_source":"K"},{"mark":"point","encoding":{"x":{"field":"c","type":"nominal"},"y":{"field":"v"},"color":{"field":"h","type":"nominal"}},"data_source":"K"}]}"#;
        let s = spec(json);
        let mut layers = super::super::build_layers(&s).unwrap();
        let mut outputs = HashMap::from([("K".to_string(), rows.clone()), (FINAL_OUTPUT_KEY.to_string(), rows.clone())]);
        let mut lineages = OutputLineages::from([("K".to_string(), source)]);
        let mut facet = FacetPartitions::default();
        let drop = drop_null_key_rows(&s, &mut layers, &mut outputs, &mut lineages, &mut facet).unwrap();
        assert_eq!(drop.routes, vec![(1, layer_key(1))]);
        let moved = lineages.get(&layer_key(1)).expect("the moved layer's rows key carries its source's lineage");
        assert_eq!(moved.statistic_input.as_ref(), Some(&consumed));
        // The reader a data-aware sort uses finds the consumed rows through it.
        let mut routed = s.clone();
        routed.layers.as_mut().unwrap()[1].own_rows = Some(layer_key(1));
        let sort_rows = crate::render::scale_resolve::layer_sort_rows(
            &routed,
            1,
            &routed.layers.as_ref().unwrap()[1],
            &outputs[FINAL_OUTPUT_KEY],
            &outputs,
            &lineages,
            false,
        )
        .unwrap();
        assert_eq!(sort_rows.batch, &consumed, "the sort reads the rows the statistic consumed");
    }

    /// A reference layer under an inherited coordinate beside a literal
    /// "null" category: its structural null rows are not counted, reach the
    /// builder, and must place nothing: a true null is never keyed into the
    /// literal's band (rule and tick), and a row null on both categorical
    /// positions draws no mark (point, text, rect).
    #[test]
    fn a_true_null_never_draws_into_a_literal_null_band_on_any_builder_that_can_receive_it() {
        let render = |spec: &ChartSpec, b: &RecordBatch| {
            crate::render::render_svg(
                spec,
                b,
                &crate::layout::ThemeInputs::default(),
                crate::layout::Viewport { width: 600.0, height: 400.0 },
                &crate::render::config::RenderConfig::default(),
                &crate::render::chart_config::ChartConfig::default(),
            )
            .unwrap()
            .bytes
        };
        let lines = |svg: &str| svg.matches("<line ").count();

        // One-axis reference marks: bar over c = [a, "null", b], reference x = [b, null, null].
        let reference = |mark: &str| {
            let json = format!(
                r#"{{"data":{{"kind":"named","name":"default"}},"mark":"bar","encoding":{{"x":{{"field":"c","type":"nominal"}},"y":{{"field":"v"}}}},"layers":[{{"mark":"bar","encoding":{{"x":{{"field":"c","type":"nominal"}},"y":{{"field":"v"}}}}}},{{"mark":"{mark}","encoding":{{"x":{{"field":"ref","type":"nominal"}},"y":{{"field":"v"}}}},"inherited_positional":["y"]}}]}}"#
            );
            spec(&json)
        };
        let with_ref = |ref_values: &[Option<&str>]| {
            batch(vec![
                ("c", strings(&[Some("a"), Some("null"), Some("b")])),
                ("v", floats(&[Some(1.0), Some(2.0), Some(3.0)])),
                ("ref", strings(ref_values)),
            ])
        };
        for mark in ["rule", "tick"] {
            let chart = reference(mark);
            let only_b = render(&chart, &with_ref(&[Some("b"), None, None]));
            let none = render(&chart, &with_ref(&[None, None, None]));
            let literal = render(&chart, &with_ref(&[Some("null"), None, None]));
            let marks = |svg: &str| lines(svg) - lines(&none);
            assert!(marks(&only_b) > 0, "{mark}: the real `b` draws");
            assert_eq!(
                marks(&literal),
                marks(&only_b),
                "{mark}: the literal 'null' reference and the real `b` each draw one mark, and the true nulls add none"
            );
        }

        // Two categorical positions, a literal "null" band on both, and a row
        // null on both: it draws nothing, so the chart equals the one without it.
        let grid = |mark: &str, extra: &str| {
            spec(&format!(
                r#"{{"data":{{"kind":"named","name":"default"}},"mark":"{mark}","encoding":{{"x":{{"field":"c","type":"nominal"}},"y":{{"field":"h","type":"nominal"}}{extra}}}}}"#
            ))
        };
        let cells = |with_null_row: bool| {
            let mut c = vec![Some("a"), Some("null"), Some("b")];
            let mut h = vec![Some("a"), Some("null"), Some("b")];
            let mut z = vec![Some(1.0), Some(2.0), Some(3.0)];
            if with_null_row {
                c.push(None);
                h.push(None);
                z.push(Some(3.0));
            }
            batch(vec![("c", strings(&c)), ("h", strings(&h)), ("z", floats(&z))])
        };
        for (mark, extra) in [
            ("point", ""),
            ("text", r#","text":{"field":"z"}"#),
            ("rect", r#","color":{"field":"z","type":"quantitative"}"#),
        ] {
            let spec = grid(mark, extra);
            assert_eq!(
                render(&spec, &cells(true)),
                render(&spec, &cells(false)),
                "{mark}: a row null on both categorical positions draws nothing"
            );
        }
    }
}
