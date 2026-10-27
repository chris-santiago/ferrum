//! A faceted chart's panels: which facet values get a panel, in what order.
//!
//! - **Membership.** A value gets a panel when some drawing layer draws rows
//!   carrying it: each layer's own rows (its named output, or the chart's
//!   rows for a layer with no `data_source`), the chart's rows for an
//!   unlayered spec. Rows no layer draws never make a panel, with one
//!   exception: a partition a statistic's `groupby` null-key drop emptied
//!   keeps its panel, drawn empty beside the drop's warning
//!   ([`FacetPartitions::emptied_by_drop`]).
//! - **Order.** Unsorted, the panels follow the order the pipeline
//!   partitioned the facet's input in, filtered to the values that get a
//!   panel; in grid mode the column and row dimensions each take their own
//!   partition order. A value a layer draws that no partition holds follows,
//!   in first appearance over the layers in layer order.
//! - **Data-aware sort.** A sort object or an `x`/`-x`/`y`/`-y` shorthand
//!   orders each value by the rows of the first layer, in layer order,
//!   that draws it, with that layer's own x and y fields for a shorthand,
//!   as the establishing layer orders an axis ([`layer_sort_rows`]). Each
//!   value reads its own pre-statistic rows or, when the sort's actual
//!   field lives only in the layer's final rows (an aggregate's own
//!   output), its final rows instead ([`batch_carrying`]) -- both the
//!   value and the category key for that value come from the SAME chosen
//!   batch, so a row never splits across the two. One batch is assembled
//!   from those rows and handed to the one sorter
//!   ([`apply_sort_to_domain`]); a value whose layer's rows cannot be read
//!   for the sort is ordered by its rule for a value with no rows, and
//!   [`RenderWarning::FacetSortPartial`] names it.

use arrow::array::Array;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use crate::layout::{FacetGroup, FacetKey, FacetSpec};
use crate::spec::chart::ChartSpec;
use crate::transform::core::FINAL_OUTPUT_KEY;
use crate::transform::run::OutputLineages;

use super::super::scale_resolve::{apply_sort_to_domain, batch_carrying, chart_sort_rows, layer_sort_rows, SortContext, SortRows};
use super::super::{RenderError, RenderWarning};
use super::FacetPartitions;

/// The assembled sort batch's columns for a shorthand's x and y values.
const SORT_X: &str = "__facet_sort_x";
const SORT_Y: &str = "__facet_sort_y";

/// One drawing layer's rows: what it draws, which decides membership, and
/// what a data-aware sort of the values it draws reads.
pub(super) struct FacetLayerRows<'a> {
    pub(super) drawn: &'a RecordBatch,
    pub(super) sort: SortRows<'a>,
}

/// Every drawing layer's rows, in layer order, and the pipeline's
/// partitions.
pub(super) struct FacetRows<'a> {
    pub(super) layers: Vec<FacetLayerRows<'a>>,
    pub(super) partitions: &'a FacetPartitions,
    pub(super) coord_flipped: bool,
}

impl<'a> FacetRows<'a> {
    /// The rows of `spec`'s layers, or of the chart for an unlayered spec,
    /// whose sort shorthand reads `chart_fields` (the rendered x and y).
    /// `layer_batches` is the batch each layer draws, as the render resolved
    /// it, in layer order.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        spec: &'a ChartSpec,
        transformed: &'a RecordBatch,
        transform_outputs: &'a HashMap<String, RecordBatch>,
        lineages: &'a OutputLineages,
        layer_batches: &[&'a RecordBatch],
        partitions: &'a FacetPartitions,
        chart_fields: (Option<&'a str>, Option<&'a str>),
        coord_flipped: bool,
    ) -> Self {
        let layers = match spec.layers.as_deref() {
            Some(layers) => layers
                .iter()
                .zip(layer_batches)
                .map(|(layer, drawn)| FacetLayerRows {
                    drawn,
                    sort: layer_sort_rows(spec, layer, transformed, transform_outputs, lineages, false),
                })
                .collect(),
            None => vec![FacetLayerRows {
                drawn: transformed,
                sort: chart_sort_rows(transformed, lineages.get(FINAL_OUTPUT_KEY), chart_fields.0, chart_fields.1),
            }],
        };
        Self { layers, partitions, coord_flipped }
    }
}

/// The facet's panels, one group per value (wrap) or per row × column pair
/// (grid), in row-major order.
pub(super) fn facet_groups(
    rows: &FacetRows<'_>,
    fspec: &FacetSpec,
    warnings: &mut Vec<RenderWarning>,
) -> Result<Vec<FacetGroup>, RenderError> {
    let partitions = &rows.partitions.order;
    let emptied = &rows.partitions.emptied_by_drop;
    let col_held = held_values(rows, &fspec.field)?;
    let mut cols = panel_order(partitions.iter().map(|(col, _)| col), emptied.iter().map(|(col, _)| col), &col_held);
    sort_panels(&mut cols, &fspec.field, fspec.sort.as_ref(), rows, &col_held, warnings)?;
    let Some(row_field) = &fspec.row else {
        let counts = value_counts(rows, &[&fspec.field])?;
        return Ok(cols
            .into_iter()
            .map(|value| FacetGroup {
                n_rows: counts.get(&vec![value.clone()]).copied().unwrap_or(0),
                key: FacetKey { field: fspec.field.clone(), value },
                row_key: None,
            })
            .collect());
    };
    let row_held = held_values(rows, row_field)?;
    let mut row_values = panel_order(
        partitions.iter().filter_map(|(_, row)| row.as_ref()),
        emptied.iter().filter_map(|(_, row)| row.as_ref()),
        &row_held,
    );
    sort_panels(&mut row_values, row_field, fspec.row_sort.as_ref(), rows, &row_held, warnings)?;
    let counts = value_counts(rows, &[&fspec.field, row_field])?;
    let mut groups = Vec::with_capacity(row_values.len() * cols.len());
    for row_value in &row_values {
        for col_value in &cols {
            groups.push(FacetGroup {
                key: FacetKey { field: fspec.field.clone(), value: col_value.clone() },
                n_rows: counts.get(&vec![col_value.clone(), row_value.clone()]).copied().unwrap_or(0),
                row_key: Some(FacetKey { field: row_field.clone(), value: row_value.clone() }),
            });
        }
    }
    Ok(groups)
}

/// The values of `field` each layer draws, per layer in layer order, and
/// every value in first appearance over the layers.
struct Held {
    per_layer: Vec<HashSet<String>>,
    appearance: Vec<String>,
}

fn held_values(rows: &FacetRows<'_>, field: &str) -> Result<Held, RenderError> {
    let mut per_layer = Vec::with_capacity(rows.layers.len());
    let mut appearance = Vec::new();
    let mut seen = HashSet::new();
    for layer in &rows.layers {
        let mut values = HashSet::new();
        for value in crate::render::facet_key_column(layer.drawn, field)?.iter().flatten() {
            values.insert(value.to_string());
            if seen.insert(value.to_string()) {
                appearance.push(value.to_string());
            }
        }
        per_layer.push(values);
    }
    Ok(Held { per_layer, appearance })
}

/// The partition order of `partition_values`, deduplicated and filtered to
/// the values a layer draws or a null-key drop emptied (`emptied`), then
/// any drawn value no partition holds.
fn panel_order<'v>(
    partition_values: impl Iterator<Item = &'v String>,
    emptied: impl Iterator<Item = &'v String>,
    held: &'v Held,
) -> Vec<String> {
    let drawn: HashSet<&String> = held.appearance.iter().chain(emptied).collect();
    let mut placed: HashSet<&String> = HashSet::new();
    let mut order: Vec<String> = Vec::new();
    for value in partition_values.filter(|value| drawn.contains(value)).chain(&held.appearance) {
        if placed.insert(value) {
            order.push(value.clone());
        }
    }
    order
}

/// Rows per combination of `fields`' values, over every layer's drawn rows.
fn value_counts(rows: &FacetRows<'_>, fields: &[&str]) -> Result<HashMap<Vec<String>, u64>, RenderError> {
    let mut counts = HashMap::new();
    for layer in &rows.layers {
        let columns = fields
            .iter()
            .map(|field| crate::render::facet_key_column(layer.drawn, field))
            .collect::<Result<Vec<_>, _>>()?;
        for row in 0..layer.drawn.num_rows() {
            let key: Option<Vec<String>> = columns
                .iter()
                .map(|column| column.is_valid(row).then(|| column.value(row).to_string()))
                .collect();
            if let Some(key) = key {
                *counts.entry(key).or_insert(0) += 1;
            }
        }
    }
    Ok(counts)
}

/// What a data-aware facet sort reads from each value's rows.
enum SortReads<'s> {
    /// Not data-aware: the sorter reads no rows.
    Nothing,
    /// A shorthand, reading the rendered `x` or `y` field of each layer.
    Channel { y: bool },
    /// A sort object reading `field`, or only counting rows.
    Field { field: &'s str, counts_only: bool },
}

fn sort_reads(sort: Option<&serde_json::Value>, coord_flipped: bool) -> SortReads<'_> {
    match sort {
        Some(serde_json::Value::String(s)) if matches!(s.as_str(), "x" | "-x" | "y" | "-y") => {
            let authored = s.trim_start_matches('-');
            SortReads::Channel { y: super::user_facing_channel(authored, coord_flipped) == "y" }
        }
        Some(serde_json::Value::Object(obj)) => match obj.get("field").and_then(|field| field.as_str()) {
            Some(field) => SortReads::Field {
                field,
                counts_only: obj.get("op").and_then(|op| op.as_str()) == Some("count"),
            },
            None => SortReads::Nothing,
        },
        _ => SortReads::Nothing,
    }
}

/// Order `domain` (the values of `field`) by `sort`, reading each value's
/// first drawing layer's sort rows, and name the values no rows could order.
fn sort_panels(
    domain: &mut Vec<String>,
    field: &str,
    sort: Option<&serde_json::Value>,
    rows: &FacetRows<'_>,
    held: &Held,
    warnings: &mut Vec<RenderWarning>,
) -> Result<(), RenderError> {
    let Some(sort) = sort else { return Ok(()) };
    let reads = sort_reads(Some(sort), rows.coord_flipped);
    let SortBatch { batch, unsortable, read_anywhere } = sort_batch(domain, field, &reads, rows, held)?;
    // Where no value's rows hold what the sort reads, the sorter itself
    // reports the missing field or channel and keeps the partition order.
    let channel_field = |name: &'static str| read_anywhere.then_some(name);
    let ctx = SortContext {
        category_field: field,
        batch: &batch,
        fallback: None,
        x_field: channel_field(SORT_X),
        y_field: channel_field(SORT_Y),
        coord_flipped: rows.coord_flipped,
    };
    let mut scratch = Vec::new();
    apply_sort_to_domain(domain, Some(sort), &ctx, &mut scratch)?;
    warnings.extend(scratch.into_iter().map(|w| match w {
        RenderWarning::SortSpecIgnored { reason } => RenderWarning::FacetSortSpecIgnored { reason },
        other => other,
    }));
    if read_anywhere && !unsortable.is_empty() {
        let source = match reads {
            SortReads::Channel { y } => format!("{} value", if y { "y" } else { "x" }),
            SortReads::Field { field, .. } => format!("'{field}' value"),
            SortReads::Nothing => String::new(),
        };
        warnings.push(RenderWarning::FacetSortPartial { unsorted: unsortable, source });
    }
    Ok(())
}

/// What a data-aware facet sort reads: one batch of every value's sort
/// rows, the values whose rows lack what the sort reads (they contribute no
/// rows), and whether any value's rows held it. The batch has no value
/// column for a sort object when none did.
struct SortBatch {
    batch: RecordBatch,
    unsortable: Vec<String>,
    read_anywhere: bool,
}

/// One batch of every value's sort rows: the value under `field`, and the
/// columns `reads` names, each read from the rows of the first layer
/// drawing that value -- its pre-statistic rows when they carry the field
/// `reads` names, else its final rows ([`batch_carrying`]).
fn sort_batch(
    domain: &[String],
    field: &str,
    reads: &SortReads<'_>,
    rows: &FacetRows<'_>,
    held: &Held,
) -> Result<SortBatch, RenderError> {
    let mut categories: Vec<String> = Vec::new();
    let mut xs: Vec<Option<f64>> = Vec::new();
    let mut ys: Vec<Option<f64>> = Vec::new();
    let mut values: Vec<Option<f64>> = Vec::new();
    let mut unsortable = Vec::new();
    let mut read_anywhere = false;
    if !matches!(reads, SortReads::Nothing) {
        for value in domain {
            // A panel a null-key drop emptied has no rows any layer draws: it
            // sorts as a value with no rows, and the drop's warning names it.
            let Some(layer) = held.per_layer.iter().position(|values| values.contains(value)) else {
                continue;
            };
            let source = &rows.layers[layer].sort;
            // The field the sort actually reads this time, so the batch
            // choice below matches the ACTUAL field (an explicit sort
            // object can name a raw field the established channel never
            // draws) rather than being decided by `x_field`/`y_field`
            // presence alone (`resolve_sort_batch`'s doc). A count-only sort
            // never reads a value column, so it stays on `source.batch`, the
            // pre-statistic grain, exactly as `resolve_sort_batch` keeps
            // `SortOp::Count` off the fallback: counting the fallback's rows
            // would count groups, not observations.
            let wanted = match reads {
                SortReads::Channel { y: true } => source.y_field,
                SortReads::Channel { y: false } => source.x_field,
                SortReads::Field { field: sort_field, counts_only: false } => Some(*sort_field),
                SortReads::Field { counts_only: true, .. } | SortReads::Nothing => None,
            };
            // Both the category key and the value column are read from this
            // SAME batch: `source.fallback`'s rows are a different grain
            // (one row per group, not per observation) from `source.batch`'s,
            // so mixing keys from one and values from the other would
            // misalign every row.
            let chosen = match wanted {
                Some(name) => batch_carrying(source.batch, source.fallback, name),
                None => source.batch,
            };
            let column = |name: Option<&str>| name.and_then(|name| crate::render::arrow_cast::col_as_f64(chosen, name).ok());
            let (x, y, field_values) = match reads {
                SortReads::Channel { y } => {
                    let read = if *y { column(source.y_field) } else { column(source.x_field) };
                    match read {
                        Some(read) if *y => (None, Some(read), None),
                        Some(read) => (Some(read), None, None),
                        None => {
                            unsortable.push(value.clone());
                            continue;
                        }
                    }
                }
                SortReads::Field { field: sort_field, counts_only } => match column(Some(sort_field)) {
                    Some(read) => (None, None, Some(read)),
                    None if *counts_only => (None, None, None),
                    None => {
                        unsortable.push(value.clone());
                        continue;
                    }
                },
                SortReads::Nothing => (None, None, None),
            };
            let Ok(keys) = crate::render::arrow_cast::col_as_str(chosen, field) else {
                unsortable.push(value.clone());
                continue;
            };
            let matching: Vec<usize> =
                keys.iter().enumerate().filter(|(_, key)| key.as_deref() == Some(value.as_str())).map(|(i, _)| i).collect();
            read_anywhere = true;
            if matching.is_empty() {
                unsortable.push(value.clone());
                continue;
            }
            for row in matching {
                categories.push(value.clone());
                xs.push(x.as_ref().and_then(|x| x[row]));
                ys.push(y.as_ref().and_then(|y| y[row]));
                values.push(field_values.as_ref().and_then(|v| v[row]));
            }
        }
    }
    // `SortReads::Nothing` (the plain `"ascending"`/`"descending"` forms)
    // reads no rows at all — `categories` stays empty, above — so this
    // batch's only job for that form is to carry `field`'s REAL dtype,
    // never a hardcoded `Utf8`, so `apply_sort_to_domain`'s numeric check
    // (the same one an ordinal axis or a categorical legend already gets
    // over this same column) can see it. A `Channel`/`Field` read still
    // reads actual category STRINGS below (`col_as_str` in the loop
    // above), so its column stays `Utf8` unchanged.
    let category_column: ArrayRef = if matches!(reads, SortReads::Nothing) {
        let real_dtype = rows
            .layers
            .iter()
            .find_map(|l| l.drawn.schema().column_with_name(field).map(|(_, f)| f.data_type().clone()))
            .unwrap_or(DataType::Utf8);
        arrow::array::new_null_array(&real_dtype, categories.len())
    } else {
        Arc::new(StringArray::from(categories))
    };
    let mut fields = vec![
        Field::new(field, category_column.data_type().clone(), true),
        Field::new(SORT_X, DataType::Float64, true),
        Field::new(SORT_Y, DataType::Float64, true),
    ];
    let mut columns: Vec<ArrayRef> = vec![
        category_column,
        Arc::new(Float64Array::from(xs)),
        Arc::new(Float64Array::from(ys)),
    ];
    if let SortReads::Field { field: sort_field, .. } = reads {
        if *sort_field != field && read_anywhere {
            fields.push(Field::new(*sort_field, DataType::Float64, true));
            columns.push(Arc::new(Float64Array::from(values)));
        }
    }
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
        .map_err(|e| RenderError::ScaleResolutionFailed(format!("facet sort rows: {e}")))?;
    Ok(SortBatch { batch, unsortable, read_anywhere })
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::FacetPartitionKey;
    use crate::layout::FacetMode;

    /// Rows over a facet field `f`, an optional row field `r`, and `y`.
    fn rows(f: &[&str], r: Option<&[&str]>, y: &[f64]) -> RecordBatch {
        let mut fields = vec![Field::new("f", DataType::Utf8, true)];
        let mut columns: Vec<ArrayRef> = vec![Arc::new(StringArray::from(f.to_vec()))];
        if let Some(r) = r {
            fields.push(Field::new("r", DataType::Utf8, true));
            columns.push(Arc::new(StringArray::from(r.to_vec())));
        }
        fields.push(Field::new("y", DataType::Float64, true));
        columns.push(Arc::new(Float64Array::from(y.to_vec())));
        RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
    }

    /// A statistic's own output rows over a facet field `f`: one row per
    /// value, carrying only `f` and `mean_y` (never `y`), the shape an
    /// aggregating layer's final rows take.
    fn agg_rows(f: &[&str], mean_y: &[f64]) -> RecordBatch {
        let fields = vec![Field::new("f", DataType::Utf8, true), Field::new("mean_y", DataType::Float64, true)];
        let columns: Vec<ArrayRef> = vec![Arc::new(StringArray::from(f.to_vec())), Arc::new(Float64Array::from(mean_y.to_vec()))];
        RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
    }

    /// Rows over an `Int64` facet field `f` and `y`.
    fn int_rows(f: &[i64], y: &[f64]) -> RecordBatch {
        let fields = vec![Field::new("f", DataType::Int64, true), Field::new("y", DataType::Float64, true)];
        let columns: Vec<ArrayRef> = vec![Arc::new(arrow::array::Int64Array::from(f.to_vec())), Arc::new(Float64Array::from(y.to_vec()))];
        RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
    }

    fn layer(batch: &RecordBatch) -> FacetLayerRows<'_> {
        FacetLayerRows { drawn: batch, sort: SortRows { batch, fallback: None, x_field: None, y_field: Some("y") } }
    }

    /// `sort="ascending"` on an `Int64` facet field orders panels by value
    /// (`-3, 1, 2, 10`), not as text (`-3, 1, 10, 2`) — the same numeric
    /// rule `apply_sort_to_domain` already gives an ordinal axis or a
    /// categorical legend over the same column. `sort_batch` builds the
    /// category column at the facet field's own dtype, so the numeric
    /// check sees it.
    #[test]
    fn ascending_sort_orders_a_numeric_facet_field_by_value() {
        let partitions = wrap_keys(&["10", "-3", "1", "2"]);
        let batch = int_rows(&[10, -3, 1, 2], &[1.0, 2.0, 3.0, 4.0]);
        let facet_rows = FacetRows { layers: vec![layer(&batch)], partitions: &partitions, coord_flipped: false };
        let groups = facet_groups(&facet_rows, &spec(false, Some(serde_json::json!("ascending")), None), &mut Vec::new()).unwrap();
        assert_eq!(values(&groups), ["-3", "1", "2", "10"]);
    }

    /// `sort="descending"` on the same `Int64` field reverses the
    /// value-ordered result (`10, 2, 1, -3`), not the text-reversed one
    /// (`2, 10, 1, -3`).
    #[test]
    fn descending_sort_orders_a_numeric_facet_field_by_value() {
        let partitions = wrap_keys(&["10", "-3", "1", "2"]);
        let batch = int_rows(&[10, -3, 1, 2], &[1.0, 2.0, 3.0, 4.0]);
        let facet_rows = FacetRows { layers: vec![layer(&batch)], partitions: &partitions, coord_flipped: false };
        let groups = facet_groups(&facet_rows, &spec(false, Some(serde_json::json!("descending")), None), &mut Vec::new()).unwrap();
        assert_eq!(values(&groups), ["10", "2", "1", "-3"]);
    }

    fn spec(row: bool, sort: Option<serde_json::Value>, row_sort: Option<serde_json::Value>) -> FacetSpec {
        FacetSpec {
            field: "f".into(),
            row: row.then(|| "r".into()),
            title: None,
            row_title: None,
            sort,
            row_sort,
            repeat: Vec::new(),
            mode: if row { FacetMode::Grid { nrows: 3, ncols: 3 } } else { FacetMode::Wrap { ncols: 3 } },
            spacing: None,
            resolve: Default::default(),
        }
    }

    fn wrap_keys(partitions: &[&str]) -> FacetPartitions {
        keys(partitions.iter().map(|value| (value.to_string(), None)).collect())
    }

    /// Partitions in `order`, none emptied by a null-key drop.
    fn keys(order: Vec<FacetPartitionKey>) -> FacetPartitions {
        FacetPartitions { order, emptied_by_drop: HashSet::new() }
    }

    fn values(groups: &[FacetGroup]) -> Vec<&str> {
        groups.iter().map(|g| g.key.value.as_str()).collect()
    }

    /// A value only a later layer draws, which comes first in the input,
    /// takes its partition place whichever operand order the layers are in.
    /// The rejected alternative orders the first layer's values first, so
    /// `(P + L)` and `(L + P)` would lay the panels out differently.
    #[test]
    fn panels_follow_the_partition_order_in_either_layer_order() {
        let partitions = wrap_keys(&["a", "b", "c"]);
        let points = rows(&["c"], None, &[1.0]);
        let line = rows(&["a", "b", "c"], None, &[1.0, 2.0, 3.0]);
        for layers in [[&points, &line], [&line, &points]] {
            let facet_rows = FacetRows {
                layers: layers.iter().map(|b| layer(b)).collect(),
                partitions: &partitions,
                coord_flipped: false,
            };
            let groups = facet_groups(&facet_rows, &spec(false, None, None), &mut Vec::new()).unwrap();
            assert_eq!(values(&groups), ["a", "b", "c"]);
        }
    }

    /// A value the input holds but no layer draws gets no panel, as on the
    /// flat chart; the rejected alternative also reads rows no layer draws.
    #[test]
    fn a_value_no_layer_draws_gets_no_panel() {
        let partitions = wrap_keys(&["a", "b", "c"]);
        let first = rows(&["a", "c"], None, &[1.0, 2.0]);
        let second = rows(&["c"], None, &[1.0]);
        let facet_rows = FacetRows {
            layers: vec![layer(&first), layer(&second)],
            partitions: &partitions,
            coord_flipped: false,
        };
        let groups = facet_groups(&facet_rows, &spec(false, None, None), &mut Vec::new()).unwrap();
        assert_eq!(values(&groups), ["a", "c"]);
        assert_eq!(groups.iter().map(|g| g.n_rows).collect::<Vec<_>>(), [1, 2]);
    }

    /// No layer draws any row: no panel at all.
    #[test]
    fn no_drawn_rows_gives_no_panel() {
        let partitions = wrap_keys(&["a"]);
        let empty = rows(&[], None, &[]);
        let facet_rows = FacetRows { layers: vec![layer(&empty)], partitions: &partitions, coord_flipped: false };
        assert!(facet_groups(&facet_rows, &spec(false, None, None), &mut Vec::new()).unwrap().is_empty());
    }

    /// A data-aware sort orders each value by the rows of the first layer
    /// that draws it, so a value only a later layer draws is sorted by that
    /// layer's rows. The rejected alternative reads the first layer's rows
    /// alone, where `c` has no rows and sorts as empty.
    #[test]
    fn a_sort_reads_each_values_first_drawing_layer() {
        let partitions = wrap_keys(&["a", "b", "c"]);
        let points = rows(&["a", "b"], None, &[1.0, 2.0]);
        let line = rows(&["a", "b", "c"], None, &[50.0, 60.0, 9.0]);
        let facet_rows = FacetRows {
            layers: vec![layer(&points), layer(&line)],
            partitions: &partitions,
            coord_flipped: false,
        };
        for sort in [
            serde_json::json!("-y"),
            serde_json::json!({"field": "y", "op": "mean", "order": "descending"}),
        ] {
            let mut warnings = Vec::new();
            let groups = facet_groups(&facet_rows, &spec(false, Some(sort.clone()), None), &mut warnings).unwrap();
            assert_eq!(values(&groups), ["c", "b", "a"], "{sort}");
            assert!(warnings.is_empty(), "{sort}: {warnings:?}");
        }
    }

    /// A faceted `X("g", sort="-y")` with `y="mean(y)")` (an aggregating
    /// layer, as a layered `bar(x, y=mean(y)) + rule(y)` produces) sorts
    /// each panel by the aggregate: the establishing layer's drawn field
    /// (`mean_y`) lives only in its final rows, not its pre-statistic ones,
    /// so the sort must read the fallback -- the same rule
    /// `resolve_sort_batch` applies to the axis sort. Rejects reading only
    /// `batch` (the pre-statistic rows, which never carry `mean_y`), which
    /// leaves every value unsortable and falls back to insertion order.
    #[test]
    fn a_facet_sort_reads_the_establishing_layers_aggregate_output() {
        let partitions = wrap_keys(&["a", "b", "c"]);
        let pre = rows(&["a", "a", "b", "b", "c", "c"], None, &[0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
        let agg = agg_rows(&["a", "b", "c"], &[1.0, 5.0, 9.0]);
        let facet_rows = FacetRows {
            layers: vec![FacetLayerRows {
                drawn: &agg,
                sort: SortRows { batch: &pre, fallback: Some(&agg), x_field: None, y_field: Some("mean_y") },
            }],
            partitions: &partitions,
            coord_flipped: false,
        };
        let mut warnings = Vec::new();
        let groups = facet_groups(&facet_rows, &spec(false, Some(serde_json::json!("-y")), None), &mut warnings).unwrap();
        assert_eq!(values(&groups), ["c", "b", "a"], "descending by the aggregate: {warnings:?}");
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    /// A raw-field sort is unaffected by the fallback: when `batch` already
    /// carries the sort's field, the sort reads `batch`, never `fallback`,
    /// even when a (deliberately unrelated) fallback is also present.
    #[test]
    fn a_facet_sort_over_a_raw_field_stays_on_the_pre_statistic_rows() {
        let partitions = wrap_keys(&["a", "b", "c"]);
        let pre = rows(&["a", "b", "c"], None, &[9.0, 5.0, 1.0]);
        // A fallback that would give the OPPOSITE order if ever read.
        let decoy = agg_rows(&["a", "b", "c"], &[1.0, 5.0, 9.0]);
        let facet_rows = FacetRows {
            layers: vec![FacetLayerRows {
                drawn: &pre,
                sort: SortRows { batch: &pre, fallback: Some(&decoy), x_field: None, y_field: Some("y") },
            }],
            partitions: &partitions,
            coord_flipped: false,
        };
        let mut warnings = Vec::new();
        let groups = facet_groups(&facet_rows, &spec(false, Some(serde_json::json!("-y")), None), &mut warnings).unwrap();
        assert_eq!(values(&groups), ["a", "b", "c"], "descending by batch's own y, not the decoy fallback: {warnings:?}");
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    /// A facet count sort stays on the pre-statistic rows even when the
    /// establishing layer's own output also names the sort's field (an
    /// aggregate whose output column is itself called `y`): count reads
    /// row presence at the pre-statistic grain (observations), never the
    /// aggregated grain (groups), the same rule `resolve_sort_batch`
    /// applies to the axis sort. The partition order starts `b` before
    /// `a` so the two reads are distinguishable by the resulting panel
    /// order alone: the aggregate's own output (one row per group) ties
    /// at 1 each and leaves the insertion order `[b, a]` unchanged under
    /// a stable sort, while the pre-statistic rows count 3 `a`s against 1
    /// `b` and sort to `[a, b]`.
    #[test]
    fn a_facet_count_sort_counts_the_pre_statistic_rows() {
        let partitions = wrap_keys(&["b", "a"]);
        let pre = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("f", DataType::Utf8, true),
                Field::new("val", DataType::Float64, true),
            ])),
            vec![
                Arc::new(StringArray::from(vec!["a", "a", "a", "b"])),
                Arc::new(Float64Array::from(vec![0.0, 0.0, 0.0, 0.0])),
            ],
        )
        .unwrap();
        let agg = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("f", DataType::Utf8, true),
                Field::new("y", DataType::Float64, true),
            ])),
            vec![
                Arc::new(StringArray::from(vec!["a", "b"])),
                Arc::new(Float64Array::from(vec![1.0, 1.0])),
            ],
        )
        .unwrap();
        let facet_rows = FacetRows {
            layers: vec![FacetLayerRows {
                drawn: &agg,
                sort: SortRows { batch: &pre, fallback: Some(&agg), x_field: None, y_field: None },
            }],
            partitions: &partitions,
            coord_flipped: false,
        };
        let mut warnings = Vec::new();
        let groups = facet_groups(
            &facet_rows,
            &spec(false, Some(serde_json::json!({"field": "y", "op": "count", "order": "descending"})), None),
            &mut warnings,
        )
        .unwrap();
        assert_eq!(
            values(&groups),
            ["a", "b"],
            "count reads the pre-statistic rows (a: 3, b: 1), not the aggregate's tied 1-each: {warnings:?}"
        );
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    /// In grid mode the column and row sorts each read each value's first
    /// drawing layer.
    #[test]
    fn grid_column_and_row_sorts_read_each_values_first_drawing_layer() {
        let partitions = keys(
            [("a", "p"), ("b", "p"), ("a", "q"), ("b", "q")]
                .iter()
                .map(|(c, r)| (c.to_string(), Some(r.to_string())))
                .collect(),
        );
        let points = rows(&["a"], Some(&["p"]), &[1.0]);
        let line = rows(&["a", "b", "a", "b"], Some(&["p", "p", "q", "q"]), &[1.0, 5.0, 7.0, 9.0]);
        let facet_rows = FacetRows {
            layers: vec![layer(&points), layer(&line)],
            partitions: &partitions,
            coord_flipped: false,
        };
        let by_y = serde_json::json!({"field": "y", "op": "sum", "order": "descending"});
        let groups =
            facet_groups(&facet_rows, &spec(true, Some(by_y.clone()), Some(by_y)), &mut Vec::new()).unwrap();
        let cells: Vec<(&str, &str)> = groups
            .iter()
            .map(|g| (g.key.value.as_str(), g.row_key.as_ref().map_or("", |k| k.value.as_str())))
            .collect();
        // Columns: b (5 + 9 from the line) before a (1 from the points);
        // rows: q (16, the line's) before p (1, the points').
        assert_eq!(cells, [("b", "q"), ("a", "q"), ("b", "p"), ("a", "p")]);
        assert!(groups.iter().all(|g| g.n_rows > 0));
    }

    /// A value whose first drawing layer's rows lack the sort field sorts
    /// as a value with no rows, and the facet sort warning names it.
    #[test]
    fn a_value_whose_rows_lack_the_sort_field_is_named() {
        let partitions = wrap_keys(&["a", "b"]);
        let points = rows(&["a"], None, &[3.0]);
        let unsortable = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("f", DataType::Utf8, true)])),
            vec![Arc::new(StringArray::from(vec!["b"])) as ArrayRef],
        )
        .unwrap();
        let facet_rows = FacetRows {
            layers: vec![layer(&points), FacetLayerRows { drawn: &unsortable, sort: SortRows { batch: &unsortable, fallback: None, x_field: None, y_field: None } }],
            partitions: &partitions,
            coord_flipped: false,
        };
        let mut warnings = Vec::new();
        let groups = facet_groups(&facet_rows, &spec(false, Some(serde_json::json!("y")), None), &mut warnings).unwrap();
        assert_eq!(values(&groups), ["b", "a"], "b has no rows, so it sorts lowest");
        assert_eq!(warnings, [RenderWarning::FacetSortPartial { unsorted: vec!["b".into()], source: "y value".into() }]);
        assert_eq!(
            warnings[0].to_string(),
            r#"facet sort left the values ["b"] unsorted: their rows in the layer that draws them carry no y value, so they sort as values with no rows; the other panels are sorted"#,
            "the warning names the values left unsorted and says the rest are sorted, never that every panel fell back"
        );
    }

    /// A grid's cells are the row × column product of the drawn values, in
    /// partition order, each counting its drawn rows; a wrap has no row key.
    #[test]
    fn grid_cells_cover_the_drawn_product_and_wrap_has_no_row_key() {
        let cells = [("c1", "r1"), ("c2", "r1"), ("c1", "r2"), ("c2", "r2")];
        let partitions = keys(cells.iter().map(|(c, r)| (c.to_string(), Some(r.to_string()))).collect());
        let f: Vec<&str> = cells.iter().flat_map(|(c, _)| [*c, *c]).collect();
        let r: Vec<&str> = cells.iter().flat_map(|(_, r)| [*r, *r]).collect();
        let batch = rows(&f, Some(&r), &[0.0; 8]);
        let facet_rows = FacetRows { layers: vec![layer(&batch)], partitions: &partitions, coord_flipped: false };
        let grid = facet_groups(&facet_rows, &spec(true, None, None), &mut Vec::new()).unwrap();
        assert_eq!(grid.len(), 4);
        assert!(grid.iter().all(|g| g.n_rows == 2 && g.row_key.is_some()));
        let wrap = facet_groups(&facet_rows, &spec(false, None, None), &mut Vec::new()).unwrap();
        assert_eq!(values(&wrap), ["c1", "c2"]);
        assert!(wrap.iter().all(|g| g.row_key.is_none()));
    }

    /// A partition a null-key drop emptied keeps its panel in partition
    /// order, drawn with no rows, while a value no layer draws for any other
    /// reason still gets none. The rejected alternative builds panels from
    /// drawn rows alone, so the emptied panel vanishes.
    #[test]
    fn a_panel_the_null_key_drop_emptied_is_drawn_empty_in_partition_order() {
        let mut partitions = wrap_keys(&["a", "b", "c"]);
        partitions.emptied_by_drop.insert(("b".to_string(), None));
        let drawn = rows(&["a", "c"], None, &[1.0, 2.0]);
        let facet_rows = FacetRows { layers: vec![layer(&drawn)], partitions: &partitions, coord_flipped: false };
        let groups = facet_groups(&facet_rows, &spec(false, None, None), &mut Vec::new()).unwrap();
        assert_eq!(values(&groups), ["a", "b", "c"]);
        assert_eq!(groups.iter().map(|g| g.n_rows).collect::<Vec<_>>(), [1, 0, 1]);

        let filtered = wrap_keys(&["a", "b", "c"]);
        let facet_rows = FacetRows { layers: vec![layer(&drawn)], partitions: &filtered, coord_flipped: false };
        let groups = facet_groups(&facet_rows, &spec(false, None, None), &mut Vec::new()).unwrap();
        assert_eq!(values(&groups), ["a", "c"], "a value emptied by anything but the drop gets no panel");
    }

    /// In grid mode an emptied cell keeps both its column and its row.
    #[test]
    fn an_emptied_grid_cell_keeps_its_column_and_row() {
        let mut partitions = keys(
            [("a", "p"), ("b", "q")].iter().map(|(c, r)| (c.to_string(), Some(r.to_string()))).collect(),
        );
        partitions.emptied_by_drop.insert(("b".to_string(), Some("q".to_string())));
        let drawn = rows(&["a"], Some(&["p"]), &[1.0]);
        let facet_rows = FacetRows { layers: vec![layer(&drawn)], partitions: &partitions, coord_flipped: false };
        let groups = facet_groups(&facet_rows, &spec(true, None, None), &mut Vec::new()).unwrap();
        let cells: Vec<(&str, &str, u64)> = groups
            .iter()
            .map(|g| (g.key.value.as_str(), g.row_key.as_ref().map_or("", |k| k.value.as_str()), g.n_rows))
            .collect();
        assert_eq!(cells, [("a", "p", 1), ("b", "p", 0), ("a", "q", 0), ("b", "q", 0)]);
    }

    /// A data-aware sort orders an emptied panel as a value with no rows
    /// and does not name it as left unsorted: it has no rows to read, and
    /// the drop's own warning names it.
    #[test]
    fn a_sort_orders_an_emptied_panel_as_one_with_no_rows_without_naming_it() {
        let mut partitions = wrap_keys(&["a", "b", "c"]);
        partitions.emptied_by_drop.insert(("a".to_string(), None));
        let drawn = rows(&["b", "c"], None, &[1.0, 5.0]);
        let facet_rows = FacetRows { layers: vec![layer(&drawn)], partitions: &partitions, coord_flipped: false };
        let mut warnings = Vec::new();
        let groups = facet_groups(&facet_rows, &spec(false, Some(serde_json::json!("-y")), None), &mut warnings).unwrap();
        assert_eq!(values(&groups), ["c", "b", "a"]);
        assert!(warnings.is_empty(), "{warnings:?}");
    }
}
