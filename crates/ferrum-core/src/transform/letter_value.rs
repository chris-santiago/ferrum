//! Letter-value transform (Phase 9b).
//!
//! Computes the letter-value statistics used by boxen plots
//! (Hofmann/Wickham/Kafadar 2017). For each group, emits one row per
//! depth k ∈ [1, K] with the lower/upper quantiles at probabilities
//! `2^-k` and `1 - 2^-k`. The number of depths K is chosen by one of
//! four strategies.
//!
//! Primary output schema:
//! `[group: <original group-column dtype>|Null, depth: Int32, lower: Float64, upper: Float64, level: Utf8]`
//! The `group` column's dtype matches `spec.group`'s ORIGINAL input dtype
//! (any dtype [`crate::transform::group_key::validate_groupby_columns`]
//! accepts — the same set every groupby-taking sibling accepts), not a
//! forced Utf8 stringification; an ungrouped call (`spec.group = None`)
//! still emits an all-null Utf8 `group` column, matching prior behavior.
//!
//! `spec.groupby` names further key columns: letter values are computed per
//! combination of `group` and every `groupby` field, and every output batch
//! (primary and secondary) appends one column per `groupby` field, named and
//! typed as its input column, after its own columns. An empty `groupby`
//! appends nothing. A row with a null on `group` or on a `groupby` field
//! belongs to no combination and is dropped, with one warning per field
//! naming the dropped-row count. Every key column partitions under
//! [`NullKeyPolicy::Skip`], so the shared partitioner reports the drop: a
//! call whose drop removed every row warns that the data could not be split
//! into groups and returns no rows, and a facet partition it empties is
//! recorded for the facet run to draw empty, as every other grouped
//! transform's drop is. A call that dropped nothing returns whatever it
//! computes, an empty batch included, exactly as the ungrouped call does.
//!
//! A group (or combination) whose rows hold no usable value, every one null,
//! NaN or infinite, keeps its key as one placeholder row: depth 1, the median's
//! letter, NaN `lower` and `upper`. Placeholders follow every group with
//! values, in input order, so the order of the groups with values (and any
//! `sort`) is what it is without them. The `depth_1` output carries the
//! placeholder; no other depth and no outlier row does.
//!
//! Secondary named outputs:
//! - `outliers` (or `<name>_outliers`): `[group: <original dtype>|Null, value: Float64, is_outlier: Bool]`,
//!   filtered to rows where the input value falls outside the outermost letter-value
//!   fence (the flag is therefore always `true` on emitted rows — it survives in the
//!   schema for wire-compatibility with the pre-filter contract).
//! - `depth_1`..`depth_12` (or `<name>_depth_K`): `[group, lower, upper, level]` rows
//!   filtered to that depth. When the actual K < 12, the unused outputs are emitted
//!   as zero-row batches so downstream renderers can safely consume them.

use arrow::array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int32Array, RecordBatch, StringArray,
};
use arrow::datatypes::{DataType, Field};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::transform::context::{describe_group_keys, TransformContext, TransformWarning};
use crate::transform::group_key::{
    batch_with_groupby_columns, deserialize_groupby_vec, group_partition_multi, groupby_key_at,
    materialize_groupby_col, refuse_groupby_output_collision, refuse_repeated_groupby_names,
    GroupKeyReader, KeyValue, NullKeyPolicy,
};

/// Maximum depth for per-depth named outputs.
pub(crate) const K_MAX_NAMED: usize = 12;

/// Letter labels by depth. depth 1 → "M" (median), 2 → "F" (fourths), ...,
/// 8 → "Z", then the conventional Tukey letter-value continuation past Z:
/// 9 → "Y", 10 → "X", 11 → "W", 12 → "V". Sized to `K_MAX_NAMED` so every
/// depth the transform can emit gets a real letter — no depth ever falls
/// back to the generic `depth_N` spelling on a user-facing axis.
const LETTERS: [&str; K_MAX_NAMED] =
    ["M", "F", "E", "D", "C", "B", "A", "Z", "Y", "X", "W", "V"];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum KDepth {
    Tukey,
    Proportion { p: f64 },
    Trustworthy,
    Full,
}

/// Aggregate operation used to order boxen groups along the categorical axis.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SortOp {
    Sum,
    Mean,
    Max,
    Min,
    Count,
    Median,
}

/// Direction of the group ordering.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SortOrder {
    Asc,
    Desc,
}

/// Optional group ordering for the letter-value transform.
///
/// When present, output groups are emitted in the order produced by ranking
/// each group's per-group aggregate of the (non-null) value column. Because
/// the categorical axis domain is built from the first-appearance order of the
/// `group` column in the primary batch, emitting groups in this order is what
/// makes the boxen categorical axis come out sorted — no separate layer sort or
/// domain step is required.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct LetterValueSort {
    pub op: SortOp,
    pub order: SortOrder,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct LetterValueSpec {
    pub value: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub group: Option<String>,
    /// Key columns beyond `group`: letter values are computed per combination
    /// of `group` and these, and every output batch carries them. A field
    /// equal to `group` is allowed, so a caller can carry the category column
    /// under its own name.
    #[serde(
        skip_serializing_if = "Vec::is_empty",
        default,
        deserialize_with = "deserialize_groupby_vec"
    )]
    pub groupby: Vec<String>,
    pub k_depth: KDepth,
    #[serde(default = "default_outlier_threshold")]
    pub outlier_threshold: f64,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub sort: Option<LetterValueSort>,
}

fn default_outlier_threshold() -> f64 { 0.0 }

/// The K depth a strategy computes for group size `n`, BEFORE the
/// [`K_MAX_NAMED`] cap. [`k_for_depth`] applies the cap; comparing this
/// against the capped result is how [`compute_depths`] detects (and reports,
/// #162) that a group's depth was silently reduced from what the strategy
/// actually wanted.
fn k_for_depth_uncapped(strategy: &KDepth, n: usize) -> usize {
    if n == 0 { return 0; }
    let n_f = n as f64;
    match strategy {
        KDepth::Tukey => {
            let raw = n_f.log2().floor() as i64 - 3;
            raw.max(1) as usize
        }
        KDepth::Proportion { p } => {
            let target = (p * n_f).ceil() as i64;
            let target = target.max(1);
            let mut k = 1usize;
            // Uncapped: this loop already stops itself once `k` PASSES
            // `K_MAX_NAMED` (one past it, at most), so it cannot run away for
            // a pathologically large `target`; `k_for_depth`'s `.min(K_MAX_NAMED)`
            // still owns the actual cap.
            while (1i64 << (k as i64 - 1)) < target {
                k += 1;
                if k > K_MAX_NAMED { break; }
            }
            k.max(1)
        }
        KDepth::Trustworthy => {
            // K = floor(log2(n / 3.32)).
            let val = (n_f / 3.32_f64).log2().floor() as i64;
            val.max(1) as usize
        }
        KDepth::Full => {
            let raw = n_f.log2().floor() as i64;
            raw.max(1) as usize
        }
    }
}

/// Compute the K depth for a given strategy and group size n, capped at
/// [`K_MAX_NAMED`] — the number of per-depth named outputs and Tukey letters
/// this transform can emit.
pub(crate) fn k_for_depth(strategy: &KDepth, n: usize) -> usize {
    k_for_depth_uncapped(strategy, n).min(K_MAX_NAMED)
}

use crate::transform::numeric_util::{coerce_to_float64, quantile_sorted, usable_at};

/// One letter-value combination: its `group` value (null when the call has
/// no `group`) and its `groupby` values, in `spec.groupby` order. Ordered by
/// `group` first, so an empty `groupby` orders exactly as `group` alone.
type ComboKey = (KeyValue, Vec<KeyValue>);

/// The output dtypes of a call's key columns, split as [`ComboKey`] is:
/// `group`'s (Utf8 for an ungrouped call's all-null column) and each
/// `groupby` field's.
type KeyDtypes = (DataType, Vec<DataType>);

/// The columns a call partitions its rows by, and how a partition key maps
/// back onto `group` and `groupby`: `group` (when set) leads, then every
/// `groupby` field other than `group`. A `groupby` field equal to `group`
/// reads `group`'s own value instead of partitioning by it twice.
struct KeyColumns {
    columns: Vec<String>,
    has_group: bool,
    /// Each `groupby` field's position in `columns`.
    by_positions: Vec<usize>,
}

impl KeyColumns {
    fn of(spec: &LetterValueSpec) -> Self {
        let mut columns: Vec<String> = spec.group.iter().cloned().collect();
        let by_positions = spec
            .groupby
            .iter()
            .map(|field| match columns.iter().position(|c| c == field) {
                Some(at) => at,
                None => {
                    columns.push(field.clone());
                    columns.len() - 1
                }
            })
            .collect();
        Self { columns, has_group: spec.group.is_some(), by_positions }
    }

    /// Split per-column values (a row's key, or the columns' dtypes) into
    /// `group`'s value and the `groupby` values, in `spec.groupby` order.
    /// `no_group` stands for `group` when the call has none: a null key, or
    /// the Utf8 dtype of an ungrouped call's all-null `group` column.
    /// `values` holds one entry per column for every key the
    /// partition or [`extract_values`] builds.
    fn split<T: Clone + std::fmt::Debug>(&self, values: &[T], no_group: T) -> PyResult<(T, Vec<T>)> {
        let at = |i: usize| {
            values.get(i).cloned().ok_or_else(|| {
                PyValueError::new_err(format!(
                    "stat_letter_value: internal error: {values:?} has no key column {i}"
                ))
            })
        };
        let group = if self.has_group { at(0)? } else { no_group };
        let by = self.by_positions.iter().map(|&i| at(i)).collect::<PyResult<Vec<_>>>()?;
        Ok((group, by))
    }

    fn combo(&self, key: &[KeyValue]) -> PyResult<ComboKey> {
        self.split(key, KeyValue::Null)
    }

    /// The `"col=value"` label naming one combination in a
    /// [`TransformWarning`]'s `group` field (`context.rs`'s documented
    /// contract), built with the same [`describe_group_keys`] formatter every
    /// sibling adopter (`smooth.rs`, `robust.rs`) uses, so the label text
    /// cannot drift from theirs.
    ///
    /// Names each distinct key column once, so `groupby=[group]` labels
    /// exactly as `group` alone does. `None` when no key column holds a
    /// value: no key column at all, or a null `group` with nothing else to
    /// identify the combination.
    fn label(&self, (group, by): &ComboKey) -> Option<String> {
        let mut values = vec![KeyValue::Null; self.columns.len()];
        if self.has_group {
            if let Some(slot) = values.first_mut() {
                *slot = group.clone();
            }
        }
        for (&at, value) in self.by_positions.iter().zip(by) {
            if let Some(slot) = values.get_mut(at) {
                *slot = value.clone();
            }
        }
        if values.iter().all(|value| matches!(value, KeyValue::Null)) {
            return None;
        }
        Some(describe_group_keys(&self.columns, &values))
    }
}

struct DepthRow {
    group: KeyValue,
    by: Vec<KeyValue>,
    depth: i32,
    lower: f64,
    upper: f64,
    level: String,
}

/// Extract `(combination key, value)` pairs in input row order (dropping
/// null, NaN or infinite values).
///
/// The key columns ([`KeyColumns`]) are resolved, dtype-checked and read
/// through the shared [`GroupKeyReader`] (the reader `group_partition_multi`
/// itself keys rows with, under the same `ctx`, so a binned color's bins key
/// these rows as they keyed the partition). A row is kept exactly when the
/// partition of [`compute_depths`] holds it: a row with a null on any key
/// column belongs to no group and is dropped here too. With no `group` and no
/// `groupby`, every row's key is `(Null, [])`.
fn extract_values(spec: &LetterValueSpec, batch: &RecordBatch, ctx: &TransformContext) -> PyResult<Vec<(ComboKey, f64)>> {
    let schema = batch.schema();
    let vi = schema.index_of(&spec.value)
        .map_err(|_| PyValueError::new_err(format!("stat_letter_value: column '{}' not found", spec.value)))?;
    let varr = coerce_to_float64(batch.column(vi), "stat_letter_value", &spec.value)?;

    let key_columns = KeyColumns::of(spec);
    let reader = GroupKeyReader::new(batch, &key_columns.columns, "stat_letter_value", ctx)?;

    let mut out = Vec::with_capacity(varr.len());
    for i in 0..varr.len() {
        let Some(v) = usable_at(&varr, i) else { continue };
        let key = (0..reader.len())
            .map(|column| {
                reader.key_at(column, i).ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "stat_letter_value: internal error extracting group key at row {i}"
                    ))
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        if key.iter().any(|value| matches!(value, KeyValue::Null)) {
            continue;
        }
        out.push((key_columns.combo(&key)?, v));
    }
    Ok(out)
}

pub(crate) fn apply(spec: &LetterValueSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    apply_with_context(spec, batch, &TransformContext::default())
}

/// Context-aware entry point (#162): identical to [`apply`], plus
/// `ctx.warnings` as the sink for [`compute_depths`]'s per-group
/// [`K_MAX_NAMED`] depth-cap warning.
pub(crate) fn apply_with_context(
    spec: &LetterValueSpec,
    batch: &RecordBatch,
    ctx: &TransformContext,
) -> PyResult<RecordBatch> {
    if !(spec.outlier_threshold.is_finite() && spec.outlier_threshold >= 0.0) {
        return Err(PyValueError::new_err(
            "stat_letter_value: outlier_threshold must be finite and >= 0",
        ));
    }
    let (depths, dtypes) = compute_depths(spec, batch, ctx)?;
    build_primary_batch(spec, &depths, &dtypes)
}

/// Determine the order in which combinations are emitted.
///
/// With `sort = None` the order is lexicographic over the combination key
/// (`group` first, then the `groupby` values) — identical to the historical
/// `BTreeMap` iteration order. With `sort = Some`, the CATEGORIES (the
/// `group` values that make the categorical axis) are ranked by the requested
/// aggregate over the usable (non-null, finite) value rows [`compute_depths`] assigns
/// to all of the category's combinations (so `count` counts only the rows
/// that boxen actually plots), and each category's combinations follow in
/// key order. Ranking each combination on its own would order the axis by a
/// single combination's aggregate instead of the category's. Ties are broken
/// lexicographically for byte-deterministic output. Without `groupby` every
/// category has exactly one combination, so this ranks groups as it always
/// has.
///
/// This ranking has no equivalent in [`group_partition_multi`] —
/// no other groupby-taking transform in the crate ranks its groups by an
/// aggregate of the value column — so it stays letter_value's own
/// post-partition step rather than something to route through the shared
/// toolkit; [`compute_depths`] applies it AFTER the shared partitioner has
/// already resolved which rows belong to which group.
fn group_order(
    sort: &Option<LetterValueSort>,
    grouped: &BTreeMap<ComboKey, Vec<f64>>,
) -> Vec<ComboKey> {
    // BTreeMap keys are already ordered by KeyValue's Ord impl (Null first,
    // then the typed variants in their natural order — lexicographic for a
    // Utf8 group column, numeric for an Int/UInt/Float/Bool one); this is
    // the `sort = None` answer and the tie-break baseline.
    let Some(sort) = sort else {
        return grouped.keys().cloned().collect();
    };

    let aggregate = |vs: &[f64]| -> f64 {
        match sort.op {
            SortOp::Count => vs.len() as f64,
            SortOp::Sum => vs.iter().sum(),
            SortOp::Mean => {
                if vs.is_empty() { f64::NAN } else { vs.iter().sum::<f64>() / vs.len() as f64 }
            }
            SortOp::Max => vs.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            SortOp::Min => vs.iter().copied().fold(f64::INFINITY, f64::min),
            SortOp::Median => {
                if vs.is_empty() {
                    f64::NAN
                } else {
                    let mut s = vs.to_vec();
                    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    quantile_sorted(&s, 0.5)
                }
            }
        }
    };

    // Each category's pooled values and its combinations, in key order.
    let mut categories: BTreeMap<&KeyValue, (Vec<f64>, Vec<ComboKey>)> = BTreeMap::new();
    for (key, vs) in grouped {
        let (pooled, combos) = categories.entry(&key.0).or_default();
        pooled.extend_from_slice(vs);
        combos.push(key.clone());
    }

    // Rank lexicographic order by aggregate; stable sort keeps lexicographic
    // order among ties, then we reverse for descending.
    let mut ranked: Vec<(f64, Vec<ComboKey>)> = categories
        .into_values()
        .map(|(pooled, combos)| (aggregate(&pooled), combos))
        .collect();
    ranked.sort_by(|(av, _), (bv, _)| {
        let cmp = av.partial_cmp(bv).unwrap_or(std::cmp::Ordering::Equal);
        match sort.order {
            SortOrder::Asc => cmp,
            SortOrder::Desc => cmp.reverse(),
        }
        // Tie-break: BTreeMap order is already lexicographic and `sort_by` is
        // stable, so equal aggregates retain ascending-lexicographic order
        // under both directions.
    });
    ranked.into_iter().flat_map(|(_, combos)| combos).collect()
}

/// Resolve the value column, partition `batch` by `spec.group` through the
/// shared [`group_partition_multi`] (the same partitioner
/// `box_stats`/`violin`/`outliers`/`smooth`/... use — existence check, dtype
/// validation, and row-to-group assignment all live there once, not
/// re-hand-rolled here), then compute each group's letter-value depths.
///
/// Every key column (`group`, then each `groupby` field not equal to it)
/// partitions under [`NullKeyPolicy::Skip`], as `BoxStats` does: a row with a
/// null on any key column belongs to no group and is dropped, and the shared
/// partitioner reports the count per field (or, when every row is null-keyed,
/// the all-null warning naming every key column). [`group_order`] reorders the
/// groups after partitioning, so `sort = None` stays lexicographic and
/// `sort = Some(...)` ranks as it always has.
fn compute_depths(
    spec: &LetterValueSpec,
    batch: &RecordBatch,
    ctx: &TransformContext,
) -> PyResult<(Vec<DepthRow>, KeyDtypes)> {
    let schema = batch.schema();
    let vi = schema.index_of(&spec.value)
        .map_err(|_| PyValueError::new_err(format!("stat_letter_value: column '{}' not found", spec.value)))?;
    let varr = coerce_to_float64(batch.column(vi), "stat_letter_value", &spec.value)?;

    refuse_repeated_groupby_names("stat_letter_value", &spec.groupby)?;
    refuse_groupby_output_collision("stat_letter_value", &spec.groupby, &output_column_names())?;
    let key_columns = KeyColumns::of(spec);
    let partition = group_partition_multi(
        batch,
        &key_columns.columns,
        "stat_letter_value",
        NullKeyPolicy::Skip,
        ctx,
    )?;
    let dtypes = key_columns.split(&partition.dtypes, DataType::Utf8)?;

    // Per-group usable (non-null, finite) sorted values. The partitioner decides
    // WHICH rows belong to which group; filtering by the VALUE column's own
    // unusable rows is this transform's job, same shape as
    // `box_stats::apply_with_context`'s per-group `filter_map` over its
    // partition's row indices.
    //
    // A group whose rows are ALL unusable filters down to an empty `vs`
    // here. Such a group must NOT be inserted into `groups`:
    // `group_order`'s `sort=Some(mean|median)` aggregate over an empty slice
    // is `NaN`, and `NaN.partial_cmp(_)` is `None`, which the comparator
    // degrades to `Ordering::Equal` against every other group — measured
    // (not just predicted) to silently INVERT the real groups' requested
    // order (`letter_value_all_nan_group_does_not_corrupt_sort_order`).
    // Every key in `groups` therefore has at least one real value. The
    // emptied group keeps its key another way: `emptied` records it, in the
    // partition's first-appearance order, and it is emitted as one
    // placeholder row after every group with values (spec's all-null rule,
    // as `Aggregate` and `BoxStats` keep an all-null group), so its band
    // slot, legend entry and facet panel stay and the render can name it.
    let mut groups: BTreeMap<ComboKey, Vec<f64>> = BTreeMap::new();
    let mut emptied: Vec<ComboKey> = Vec::new();
    for (key, rows) in &partition.groups {
        let g = key_columns.combo(key)?;
        let mut vs: Vec<f64> = rows.iter().filter_map(|&r| usable_at(&varr, r)).collect();
        if vs.is_empty() {
            emptied.push(g);
            continue;
        }
        vs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        groups.insert(g, vs);
    }
    let order = group_order(&spec.sort, &groups);

    let mut rows = Vec::new();
    for g in &order {
        let vs = match groups.get(g) {
            Some(vs) if !vs.is_empty() => vs,
            _ => continue,
        };
        let raw_k = k_for_depth_uncapped(&spec.k_depth, vs.len());
        let k = k_for_depth(&spec.k_depth, vs.len());
        if raw_k > K_MAX_NAMED {
            // #162's named case: the strategy computed a depth deeper than
            // this transform can emit (12 Tukey letters / named per-depth
            // outputs) and it was silently capped.
            ctx.warn(TransformWarning::ParameterClamped {
                transform: "LetterValue".into(),
                group: key_columns.label(g),
                param: "k_depth".into(),
                requested: raw_k.to_string(),
                applied: k.to_string(),
                panel: None,
            });
        }
        for depth in 1..=k {
            let p = 0.5_f64.powi(depth as i32);
            let lower = quantile_sorted(vs, p);
            let upper = quantile_sorted(vs, 1.0 - p);
            rows.push(DepthRow {
                group: g.0.clone(),
                by: g.1.clone(),
                depth: depth as i32,
                lower,
                upper,
                level: level_label(depth),
            });
        }
    }
    // Each emptied group's one row: its key at depth 1, the depth every group
    // with values has a row at (the median tick and the innermost band read
    // it), with NaN letter values, so the key reaches the category axis while
    // nothing draws for it.
    for g in &emptied {
        rows.push(DepthRow {
            group: g.0.clone(),
            by: g.1.clone(),
            depth: 1,
            lower: f64::NAN,
            upper: f64::NAN,
            level: level_label(1),
        });
    }
    Ok((rows, dtypes))
}

/// The Tukey letter naming `depth` (1 is the median, `"M"`).
///
/// `k_for_depth` clamps every depth this transform emits to `K_MAX_NAMED`
/// (== `LETTERS.len()`), so a `depth` past `LETTERS.len()` cannot occur here;
/// the `depth_{depth}` fallback is defensive only, kept so a future
/// divergence between the two constants degrades to a readable label instead
/// of an out-of-bounds panic.
fn level_label(depth: usize) -> String {
    if depth >= 1 && depth <= LETTERS.len() {
        LETTERS[depth - 1].to_string()
    } else {
        format!("depth_{depth}")
    }
}

/// The primary batch's own columns, before any `groupby` column; `group`
/// is declared at the ungrouped call's Utf8 and takes `group`'s input dtype
/// otherwise. Read by [`build_primary_batch`] and the `letter_value`
/// registry entry.
pub(crate) const PRIMARY_COLUMNS: &[(&str, DataType)] = &[
    ("group", DataType::Utf8),
    ("depth", DataType::Int32),
    ("lower", DataType::Float64),
    ("upper", DataType::Float64),
    ("level", DataType::Utf8),
];

/// Every column name this transform writes itself, across its primary and
/// secondary batches: the names a `groupby` field is refused against.
fn output_column_names() -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for (name, _) in PRIMARY_COLUMNS.iter().chain(OUTLIERS_COLUMNS).chain(DEPTH_COLUMNS) {
        if !names.iter().any(|n| n == name) {
            names.push((*name).to_string());
        }
    }
    names
}

/// A batch's fields from one of the column tables above: `group` takes
/// `group_dtype` and is the one nullable column.
fn own_fields(columns: &[(&str, DataType)], group_dtype: &DataType) -> Vec<Field> {
    columns
        .iter()
        .map(|(name, dt)| {
            let dt = if *name == "group" { group_dtype.clone() } else { dt.clone() };
            Field::new(*name, dt, *name == "group")
        })
        .collect()
}

fn build_primary_batch(
    spec: &LetterValueSpec,
    rows: &[DepthRow],
    (group_dtype, by_dtypes): &KeyDtypes,
) -> PyResult<RecordBatch> {
    let fields = own_fields(PRIMARY_COLUMNS, group_dtype);
    let group_keys: Vec<Vec<KeyValue>> = rows.iter().map(|r| vec![r.group.clone()]).collect();
    let group_col =
        materialize_groupby_col(&group_keys, 0, group_dtype).map_err(PyValueError::new_err)?;
    let depths: Vec<i32> = rows.iter().map(|r| r.depth).collect();
    let lowers: Vec<f64> = rows.iter().map(|r| r.lower).collect();
    let uppers: Vec<f64> = rows.iter().map(|r| r.upper).collect();
    let levels: Vec<String> = rows.iter().map(|r| r.level.clone()).collect();
    let cols: Vec<ArrayRef> = vec![
        group_col,
        Arc::new(Int32Array::from(depths)),
        Arc::new(Float64Array::from(lowers)),
        Arc::new(Float64Array::from(uppers)),
        Arc::new(StringArray::from(levels.iter().map(|s| s.as_str()).collect::<Vec<_>>())),
    ];
    let by_keys: Vec<Vec<KeyValue>> = rows.iter().map(|r| r.by.clone()).collect();
    batch_with_groupby_columns(&spec.groupby, by_dtypes, &by_keys, fields, cols, "stat_letter_value")
}

/// The outlier batch's key under default construction; a spec `name` prefixes
/// it as `"{name}_outliers"`.
pub(crate) const OUTLIERS_KEY: &str = "outliers";

/// The outlier batch's column schema — read by [`secondary_outputs`]'s schema
/// builder and by `registry::entry_for_kind("letter_value")` (R1 design
/// finding (d)).
pub(crate) const OUTLIERS_COLUMNS: &[(&str, DataType)] = &[
    ("group", DataType::Utf8),
    ("value", DataType::Float64),
    ("is_outlier", DataType::Boolean),
];

/// Every per-depth band batch's column schema. `depth` itself is implicit in
/// the output's key, so it is not a column here.
pub(crate) const DEPTH_COLUMNS: &[(&str, DataType)] = &[
    ("group", DataType::Utf8),
    ("lower", DataType::Float64),
    ("upper", DataType::Float64),
    ("level", DataType::Utf8),
];

/// The key of the depth-`k` band batch under default construction; a spec
/// `name` prefixes it as `"{name}_depth_{k}"`. One key per `k` in
/// `1..=K_MAX_NAMED` is always published, empty when that depth has no rows.
pub(crate) fn depth_key(k: usize) -> String {
    format!("depth_{k}")
}

/// Compute the secondary named outputs: outliers + depth_1..depth_12.
///
/// Reads the combinations back off `primary_batch`: its `group` column and
/// one column per `groupby` field, each at its own dtype.
///
/// `ctx` is the step's context, so the input rows key their combinations as
/// the primary's partition keyed them (a binned color's bins,
/// [`crate::transform::group_key::GroupKeyReader`]).
pub(crate) fn secondary_outputs(
    spec: &LetterValueSpec,
    input_batch: &RecordBatch,
    primary_batch: &RecordBatch,
    ctx: &TransformContext,
) -> PyResult<Vec<(String, RecordBatch)>> {
    let primary_schema = primary_batch.schema();
    let group_dtype = primary_schema.field(0).data_type().clone();
    let groups_col = primary_batch.column(0);
    let depths_col = primary_batch.column(1).as_any().downcast_ref::<Int32Array>()
        .ok_or_else(|| PyValueError::new_err("stat_letter_value: expected Int32Array for 'depth' column"))?;
    let lowers_col = primary_batch.column(2).as_any().downcast_ref::<Float64Array>()
        .ok_or_else(|| PyValueError::new_err("stat_letter_value: expected Float64Array for 'lower' column"))?;
    let uppers_col = primary_batch.column(3).as_any().downcast_ref::<Float64Array>()
        .ok_or_else(|| PyValueError::new_err("stat_letter_value: expected Float64Array for 'upper' column"))?;
    let by_cols: Vec<(ArrayRef, DataType)> = spec
        .groupby
        .iter()
        .map(|name| {
            let i = primary_schema.index_of(name).map_err(|_| {
                PyValueError::new_err(format!(
                    "stat_letter_value: expected groupby column '{name}' in the primary batch"
                ))
            })?;
            Ok((primary_batch.column(i).clone(), primary_schema.field(i).data_type().clone()))
        })
        .collect::<PyResult<_>>()?;
    let by_dtypes: Vec<DataType> = by_cols.iter().map(|(_, dt)| dt.clone()).collect();

    // Reads row `i`'s key from one of the primary batch's key columns at its
    // own (possibly non-Utf8) dtype, via the shared extractor rather than a
    // hardcoded `StringArray` downcast.
    let key_at = |arr: &ArrayRef, dt: &DataType, i: usize| -> PyResult<KeyValue> {
        groupby_key_at(arr.as_ref(), dt, i).ok_or_else(|| {
            PyValueError::new_err(format!(
                "stat_letter_value: internal error extracting group key at row {i}"
            ))
        })
    };
    let combo_at = |i: usize| -> PyResult<ComboKey> {
        let group = key_at(groups_col, &group_dtype, i)?;
        let by = by_cols.iter().map(|(arr, dt)| key_at(arr, dt, i)).collect::<PyResult<_>>()?;
        Ok((group, by))
    };

    // Per-combination max-depth (lower, upper) extents — the outermost letter value.
    let mut tail: BTreeMap<ComboKey, (i32, f64, f64)> = BTreeMap::new();
    for i in 0..primary_batch.num_rows() {
        let g = combo_at(i)?;
        let d = depths_col.value(i);
        let lo = lowers_col.value(i);
        let hi = uppers_col.value(i);
        let entry = tail.entry(g).or_insert((d, lo, hi));
        if d >= entry.0 {
            *entry = (d, lo, hi);
        }
    }

    // ---- outliers output ----
    // For each input row with a usable value: classify as outlier if value falls outside
    //   [lower_K - threshold*(upper_K - lower_K), upper_K + threshold*(upper_K - lower_K)]
    // of its own combination's outermost letter values.
    // Only rows that ARE outliers are emitted (schema unchanged: `is_outlier`
    // survives as a column but is always `true` on the rows that reach it) —
    // the batch is a filtered outlier list, not a flagged copy of every input
    // row, so a boxen chart's outlier layer draws only the points it names.
    let input_pairs = extract_values(spec, input_batch, ctx)?;
    let mut o_groups: Vec<Vec<KeyValue>> = Vec::new();
    let mut o_by: Vec<Vec<KeyValue>> = Vec::new();
    let mut o_vals: Vec<f64> = Vec::new();
    let mut o_flags: Vec<bool> = Vec::new();
    for (g, v) in &input_pairs {
        let (lo, hi) = match tail.get(g) {
            Some((_, lo, hi)) => (*lo, *hi),
            None => (f64::NAN, f64::NAN),
        };
        let span = hi - lo;
        let low_fence = lo - spec.outlier_threshold * span;
        let high_fence = hi + spec.outlier_threshold * span;
        let is_outlier = !lo.is_nan() && !hi.is_nan() && (*v < low_fence || *v > high_fence);
        if !is_outlier {
            continue;
        }
        o_groups.push(vec![g.0.clone()]);
        o_by.push(g.1.clone());
        o_vals.push(*v);
        o_flags.push(is_outlier);
    }
    // Column names/nullability come from the shared consts (no drift risk);
    // only the "group" entry's declared dtype is overridden to this call's
    // actual group dtype, since the consts describe the schema under DEFAULT
    // (ungrouped, Utf8) construction — see their doc comments and
    // `registry::letter_value_secondary_outputs`.
    let o_group_col =
        materialize_groupby_col(&o_groups, 0, &group_dtype).map_err(PyValueError::new_err)?;
    let outliers_batch = batch_with_groupby_columns(
        &spec.groupby,
        &by_dtypes,
        &o_by,
        own_fields(OUTLIERS_COLUMNS, &group_dtype),
        vec![
            o_group_col,
            Arc::new(Float64Array::from(o_vals)) as ArrayRef,
            Arc::new(BooleanArray::from(o_flags)) as ArrayRef,
        ],
        "stat_letter_value outliers",
    )?;

    // ---- per-depth outputs (depth_1..depth_12) ----
    // Each depth_K output is the primary rows filtered to depth==K, projecting
    // [group, lower, upper, level] plus the groupby columns (no depth column —
    // implicit in the output name).
    let level_col = primary_batch.column(4).as_any().downcast_ref::<StringArray>()
        .ok_or_else(|| PyValueError::new_err("stat_letter_value: expected StringArray for 'level' column"))?;
    #[derive(Clone, Default)]
    struct DepthSlice {
        groups: Vec<Vec<KeyValue>>,
        by: Vec<Vec<KeyValue>>,
        lowers: Vec<f64>,
        uppers: Vec<f64>,
        levels: Vec<String>,
    }
    let mut by_depth: BTreeMap<i32, DepthSlice> = BTreeMap::new();
    for i in 0..primary_batch.num_rows() {
        let (group, by) = combo_at(i)?;
        let slice = by_depth.entry(depths_col.value(i)).or_default();
        slice.groups.push(vec![group]);
        slice.by.push(by);
        slice.lowers.push(lowers_col.value(i));
        slice.uppers.push(uppers_col.value(i));
        slice.levels.push(level_col.value(i).to_string());
    }

    let key_outliers = crate::transform::registry::secondary_key(spec.name.as_deref(), OUTLIERS_KEY);

    let mut out: Vec<(String, RecordBatch)> = Vec::new();
    out.push((key_outliers, outliers_batch));
    for k in 1..=K_MAX_NAMED {
        let slice = by_depth.remove(&(k as i32)).unwrap_or_default();
        let group_col_v = materialize_groupby_col(&slice.groups, 0, &group_dtype)
            .map_err(PyValueError::new_err)?;
        let cols: Vec<ArrayRef> = vec![
            group_col_v,
            Arc::new(Float64Array::from(slice.lowers)) as ArrayRef,
            Arc::new(Float64Array::from(slice.uppers)) as ArrayRef,
            Arc::new(StringArray::from(slice.levels.iter().map(|s| s.as_str()).collect::<Vec<_>>())) as ArrayRef,
        ];
        let batch = batch_with_groupby_columns(
            &spec.groupby,
            &by_dtypes,
            &slice.by,
            own_fields(DEPTH_COLUMNS, &group_dtype),
            cols,
            &format!("stat_letter_value depth_{k}"),
        )?;
        out.push((crate::transform::registry::secondary_key(spec.name.as_deref(), &depth_key(k)), batch));
    }
    Ok(out)
}

// ---------- PyO3 wrapper ----------

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::transform::core::TransformSpec;

/// Parse the optional `sort={"op": ..., "order": ...}` kwarg into a
/// [`LetterValueSort`]. Returns `Ok(None)` when no dict is supplied.
fn parse_sort(sort: Option<&Bound<'_, PyDict>>) -> PyResult<Option<LetterValueSort>> {
    let Some(d) = sort else { return Ok(None) };
    let op_str: String = match d.get_item("op")? {
        Some(v) => v.extract()?,
        None => "sum".to_string(),
    };
    let order_str: String = match d.get_item("order")? {
        Some(v) => v.extract()?,
        None => "ascending".to_string(),
    };
    let op = match op_str.as_str() {
        "sum" => SortOp::Sum,
        "mean" => SortOp::Mean,
        "max" => SortOp::Max,
        "min" => SortOp::Min,
        "count" => SortOp::Count,
        "median" => SortOp::Median,
        other => {
            return Err(PyValueError::new_err(format!(
                "LetterValue: unknown sort op '{other}'; expected \
                 'sum'|'mean'|'max'|'min'|'count'|'median'"
            )))
        }
    };
    let order = match order_str.as_str() {
        "ascending" | "asc" => SortOrder::Asc,
        "descending" | "desc" => SortOrder::Desc,
        other => {
            return Err(PyValueError::new_err(format!(
                "LetterValue: unknown sort order '{other}'; expected \
                 'ascending'|'descending'"
            )))
        }
    };
    Ok(Some(LetterValueSort { op, order }))
}

/// Letter-value (boxen) summary statistics for a numeric column.
///
/// Computes a hierarchical sequence of medians and quantiles extending
/// outward from the center until the estimated uncertainty of the extreme
/// values is too large. The depth of the hierarchy is controlled by
/// ``k_depth``. Used to draw boxen plots that reveal the full distributional
/// shape for large datasets.
///
/// Output columns: ``group`` (the ``group`` column's values and type, all
/// null Utf8 when ``group`` is unset), ``depth`` (Int32, 1 at the median),
/// ``lower`` and ``upper`` (Float64 letter values at probabilities
/// ``2**-depth`` and ``1 - 2**-depth``) and ``level`` (the Tukey letter,
/// ``"M"``, ``"F"``, ``"E"``, ...), one row per depth per combination, then
/// one column per ``groupby`` field. Two further named outputs (prefixed
/// ``<name>_`` when ``name`` is set): ``outliers`` holds ``group``,
/// ``value`` (Float64) and ``is_outlier`` (Bool, always true) for each
/// input row beyond its combination's outermost letter values, and
/// ``depth_1`` through ``depth_12`` hold ``group``, ``lower``, ``upper`` and
/// ``level`` for that depth's rows (empty past the deepest depth). Each also
/// ends with the ``groupby`` columns.
///
/// A group (or combination) whose rows hold no usable value, every one null,
/// NaN or infinite, keeps its key as one placeholder row at depth 1 with NaN
/// ``lower`` and ``upper``, after every group with values, in input order.
/// The placeholder appears in ``depth_1`` and in no outlier row, so the
/// group keeps its place on a categorical axis and in a legend while nothing
/// draws for it.
///
/// Parameters
/// ----------
/// value : str
///     Numeric column to summarize (any integer, unsigned, or float dtype;
///     widened to Float64). A non-numeric column raises, naming the
///     transform, column, and dtype.
/// group : str, optional
///     Single group-key column; letter-values computed independently per
///     group.
/// groupby : list of str, default []
///     Further key columns; letter values are computed per combination of
///     ``group`` and these, and every output carries one column per field,
///     named and typed as its input column. A row with a null on a
///     ``groupby`` field is dropped, with one warning per field naming the
///     dropped-row count, and a call whose every row the drop removes warns
///     that its grouped output is empty and returns no rows. A
///     field may equal ``group``; it then keeps ``group``'s rule, where null
///     ``group`` rows form their own combination.
/// k_depth : {"tukey", "proportion", "trustworthy", "full"}, \
///           default "proportion"
///     Algorithm for choosing the number of letter-value levels:
///     ``"tukey"`` — fixed integer depth; ``"proportion"`` — fraction of
///     sample size (see ``k_proportion``); ``"trustworthy"`` — based on
///     statistical trustworthiness; ``"full"`` — all levels to median.
/// k_proportion : float, default 0.007
///     Fraction of sample size used when ``k_depth="proportion"``. Must
///     be in (0, 1).
/// outlier_threshold : float, default 0.0
///     Fence multiplier for labelling outliers: a row is flagged when its
///     value falls strictly outside ``[lo - t*span, hi + t*span]``, where
///     ``lo``/``hi`` are the group's outermost letter values and ``span =
///     hi - lo``. At the default ``0.0`` the fence sits at the outermost
///     letter values themselves, so every observation strictly beyond them
///     is flagged; a value tied with ``lo`` or ``hi`` is not. Must be ≥ 0.
/// name : str, optional
///     Named output label for sibling ``Reorder(from_=...)`` lookup.
/// sort : dict, optional
///     Group ordering for the categorical axis, as ``{"op": ..., "order":
///     ...}``. ``op`` is one of ``"sum" | "mean" | "max" | "min" | "count" |
///     "median"`` and ``order`` is ``"ascending" | "descending"`` (``"asc"``
///     / ``"desc"`` also accepted). The aggregate is computed over the value
///     column per group; ``count`` counts the usable (non-null, finite) values. Groups are then
///     emitted in ranked order (ties broken lexicographically), which sets the
///     boxen categorical axis domain directly. When omitted, groups are emitted
///     in lexicographic order.
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> fm.Chart(df).mark_boxen().encode(x="cut", y="price")
#[pyclass(eq, module = "ferrum._core", name = "LetterValue")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyLetterValue(pub(crate) TransformSpec);

#[pymethods]
impl PyLetterValue {
    #[new]
    #[pyo3(signature = (
        value, *,
        group = None,
        groupby = None,
        k_depth = "proportion",
        k_proportion = 0.007,
        outlier_threshold = 0.0,
        name = None,
        sort = None,
    ))]
    // One parameter per Python keyword argument of the constructor.
    #[allow(clippy::too_many_arguments)]
    fn new(
        value: &str,
        group: Option<String>,
        groupby: Option<&Bound<'_, PyAny>>,
        k_depth: &str,
        k_proportion: f64,
        outlier_threshold: f64,
        name: Option<String>,
        sort: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        if value.is_empty() {
            return Err(PyValueError::new_err("LetterValue: value must be non-empty"));
        }
        let groupby = crate::transform::group_key::parse_groupby_pyany("LetterValue", groupby)?;
        refuse_repeated_groupby_names("LetterValue", &groupby)?;
        if !outlier_threshold.is_finite() || outlier_threshold < 0.0 {
            return Err(PyValueError::new_err(
                "LetterValue: outlier_threshold must be finite and >= 0",
            ));
        }
        let kd = match k_depth {
            "tukey" => KDepth::Tukey,
            "proportion" => {
                if !(k_proportion.is_finite() && k_proportion > 0.0 && k_proportion < 1.0) {
                    return Err(PyValueError::new_err(
                        "LetterValue: k_proportion must be in (0, 1)",
                    ));
                }
                KDepth::Proportion { p: k_proportion }
            }
            "trustworthy" => KDepth::Trustworthy,
            "full" => KDepth::Full,
            other => return Err(PyValueError::new_err(format!(
                "LetterValue: unknown k_depth '{other}'; expected 'tukey'|'proportion'|'trustworthy'|'full'"
            ))),
        };
        let sort = parse_sort(sort)?;
        Ok(PyLetterValue(TransformSpec::LetterValue(LetterValueSpec {
            value: value.to_string(),
            group,
            groupby,
            k_depth: kd,
            outlier_threshold,
            name,
            sort,
        })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::LetterValue(s) => format!(
                "LetterValue(value='{}', group={:?}, groupby={:?}, k_depth={:?}, outlier_threshold={}, sort={:?}, name={:?})",
                s.value, s.group, s.groupby, s.k_depth, s.outlier_threshold, s.sort, s.name,
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::context::{AllNullOutcome, StepInstance};
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::Schema;

    fn batch1col(name: &str, vs: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::Float64, true)]));
        RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(vs))]).unwrap()
    }

    fn batch_grouped(groups: Vec<&str>, values: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, true),
            Field::new("v", DataType::Float64, true),
        ]));
        let g_arr: Vec<Option<&str>> = groups.iter().map(|s| Some(*s)).collect();
        RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(g_arr)),
            Arc::new(Float64Array::from(values)),
        ]).unwrap()
    }

    // ── #162: degrade-warning channel ────────────────────────────────────

    /// RED (pre-#162): fails on the pre-change code, which had no channel to
    /// report `KDepth::Full`'s computed depth silently capping at
    /// `K_MAX_NAMED` (12). `log2(10000)` floors to 13, one past the cap.
    #[test]
    fn letter_value_depth_cap_pushes_a_parameter_clamped_warning() {
        let values: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
        let batch = batch1col("x", values);
        let spec = LetterValueSpec {
            groupby: Vec::new(),
            value: "x".into(),
            group: None,
            k_depth: KDepth::Full,
            outlier_threshold: 1.5,
            name: None,
            sort: None,
        };
        let ctx = TransformContext::default();
        let _ = apply_with_context(&spec, &batch, &ctx).unwrap();
        let warnings = ctx.warnings.borrow();
        assert_eq!(warnings.len(), 1, "got {warnings:?}");
        assert!(matches!(
            &warnings[0],
            TransformWarning::ParameterClamped { transform, group, param, requested, applied, .. }
                if transform == "LetterValue" && group.is_none() && param == "k_depth"
                   && requested == "13" && applied == "12"
        ));
    }

    /// A grouped call names which group's own depth was capped.
    #[test]
    fn letter_value_grouped_depth_cap_names_the_group() {
        let groups: Vec<&str> = std::iter::repeat("big").take(10_000).collect();
        let values: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
        let batch = batch_grouped(groups, values);
        let spec = LetterValueSpec {
            groupby: Vec::new(),
            value: "v".into(),
            group: Some("g".into()),
            k_depth: KDepth::Full,
            outlier_threshold: 1.5,
            name: None,
            sort: None,
        };
        let ctx = TransformContext::default();
        let _ = apply_with_context(&spec, &batch, &ctx).unwrap();
        let warnings = ctx.warnings.borrow();
        assert_eq!(warnings.len(), 1);
        // Rust-quality-reviewer finding 2 (post-#162 close): the bare group
        // VALUE ("big") broke `TransformWarning::group`'s documented
        // "col=value" contract (`context.rs`'s doc, matched by every sibling
        // adopter); must be "g=big", naming the column too.
        assert!(matches!(
            &warnings[0],
            TransformWarning::ParameterClamped { group, .. } if group.as_deref() == Some("g=big")
        ), "expected group label \"g=big\" (col=value), got {:?}", warnings[0]);
    }

    /// A well-supported depth (Tukey on a small group) must not warn.
    #[test]
    fn letter_value_tukey_default_does_not_warn() {
        let batch = batch1col("x", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let spec = LetterValueSpec {
            groupby: Vec::new(),
            value: "x".into(),
            group: None,
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: None,
            sort: None,
        };
        let ctx = TransformContext::default();
        let _ = apply_with_context(&spec, &batch, &ctx).unwrap();
        assert!(ctx.warnings.borrow().is_empty());
    }

    #[test]
    fn test_letter_value_round_trip_json() {
        let spec = LetterValueSpec {
            groupby: Vec::new(),
            value: "x".into(),
            group: Some("g".into()),
            k_depth: KDepth::Proportion { p: 0.007 },
            outlier_threshold: 1.5,
            name: Some("lv".into()),
            sort: None,
        };
        let wrapped = TransformSpec::LetterValue(spec.clone());
        let json = serde_json::to_string(&wrapped).unwrap();
        let parsed: TransformSpec = serde_json::from_str(&json).unwrap();
        match parsed {
            TransformSpec::LetterValue(s) => assert_eq!(s, spec),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_letter_value_sort_round_trip_and_deny_unknown() {
        // A valid sort dict round-trips losslessly (byte-identity for the
        // ferrum-produced wire: LetterValueSort serializes exactly {op, order}).
        let spec = LetterValueSpec {
            groupby: Vec::new(),
            value: "v".into(),
            group: Some("g".into()),
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: None,
            sort: Some(LetterValueSort { op: SortOp::Sum, order: SortOrder::Desc }),
        };
        let wrapped = TransformSpec::LetterValue(spec.clone());
        let json = serde_json::to_string(&wrapped).unwrap();
        let parsed: TransformSpec = serde_json::from_str(&json).unwrap();
        match parsed {
            TransformSpec::LetterValue(s) => assert_eq!(s, spec),
            _ => panic!("wrong variant"),
        }

        // An EXTRA key inside the sort dict must now error (no-silent-drop:
        // matches the SEAM-04 deny_unknown_fields contract on the parent spec).
        // `TransformSpec` is internally tagged (`tag = "type"`), so the spec
        // fields — including `sort` — sit at the object root.
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        value["sort"]["bogus"] = serde_json::json!("oops");
        let polluted = serde_json::to_string(&value).unwrap();
        let err = serde_json::from_str::<TransformSpec>(&polluted);
        assert!(
            err.is_err(),
            "deny_unknown_fields must reject an unknown key in the sort dict; got: {err:?}",
        );
    }

    #[test]
    fn test_letter_value_basic_depth1_is_median() {
        let vs: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let batch = batch1col("x", vs);
        let spec = LetterValueSpec {
            groupby: Vec::new(),
            value: "x".into(), group: None,
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: None,
            sort: None,
        };
        let out = apply(&spec, &batch).unwrap();
        // First row is depth=1 (median). lower == upper == median of 1..=100 = 50.5
        let depth = out.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        let lower = out.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
        let upper = out.column(3).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(depth.value(0), 1);
        assert!((lower.value(0) - 50.5).abs() < 1e-9);
        assert!((upper.value(0) - 50.5).abs() < 1e-9);
    }

    #[test]
    fn test_letter_value_tukey_k_for_n100() {
        // Tukey: K = floor(log2(100)) - 3 = 6 - 3 = 3.
        let k = k_for_depth(&KDepth::Tukey, 100);
        assert_eq!(k, 3);
    }

    #[test]
    fn test_letter_value_proportion_k_for_p007_n200() {
        // target = ceil(0.007 * 200) = 2; smallest K with 2^(K-1) >= 2 → K=2.
        let k = k_for_depth(&KDepth::Proportion { p: 0.007 }, 200);
        assert_eq!(k, 2);
    }

    #[test]
    fn test_letter_value_full_k_for_n20() {
        // Full: K = floor(log2(20)) = 4.
        let k = k_for_depth(&KDepth::Full, 20);
        assert_eq!(k, 4);
    }

    #[test]
    fn test_letter_value_grouped_3_groups() {
        let groups: Vec<&str> = (0..30).map(|i| ["a","b","c"][i % 3]).collect();
        let values: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let batch = batch_grouped(groups, values);
        let spec = LetterValueSpec {
            groupby: Vec::new(),
            value: "v".into(), group: Some("g".into()),
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: None,
            sort: None,
        };
        let out = apply(&spec, &batch).unwrap();
        // n=10 per group → Tukey K = max(1, floor(log2(10)) - 3) = max(1, 3-3) = 1.
        // 3 groups × 1 depth = 3 rows.
        assert_eq!(out.num_rows(), 3);
        let g_arr = out.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let mut seen = std::collections::BTreeSet::new();
        for i in 0..out.num_rows() {
            seen.insert(g_arr.value(i).to_string());
        }
        assert_eq!(seen.len(), 3);
    }

    #[test]
    fn test_letter_value_outliers_classifies_extreme_values() {
        // Build n=50 normal-ish values plus 2 extremes; the extremes should be
        // flagged. Threshold 1.5 means: |value - lv_K| > 1.5 * IQR(lv_K).
        let mut vs: Vec<f64> = (1..=50).map(|i| i as f64).collect();
        vs.push(-1000.0);
        vs.push(1000.0);
        let batch = batch1col("x", vs);
        let spec = LetterValueSpec {
            groupby: Vec::new(),
            value: "x".into(), group: None,
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: None,
            sort: None,
        };
        let primary = apply(&spec, &batch).unwrap();
        let secs = secondary_outputs(&spec, &batch, &primary, &TransformContext::default()).unwrap();
        let outliers = secs.iter().find(|(k, _)| k == "outliers").unwrap();
        let sch = outliers.1.schema();
        assert_eq!(sch.field(0).name(), "group");
        assert_eq!(sch.field(1).name(), "value");
        assert_eq!(sch.field(2).name(), "is_outlier");
        // The two extreme values must be flagged is_outlier=true.
        let val_col = outliers.1.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        let flag_col = outliers.1.column(2).as_any().downcast_ref::<BooleanArray>().unwrap();
        let mut found_extreme = 0;
        for i in 0..outliers.1.num_rows() {
            let v = val_col.value(i);
            if (v - 1000.0).abs() < 1e-9 || (v + 1000.0).abs() < 1e-9 {
                assert!(flag_col.value(i), "extreme value {v} should be flagged");
                found_extreme += 1;
            }
        }
        assert_eq!(found_extreme, 2);
    }

    #[test]
    fn test_letter_value_outliers_batch_contains_only_true_flags() {
        // Regression for F-L03-02: the outliers batch must be a FILTERED list
        // of outlier rows, not every input row with a flag column. n=50
        // in-range values + 2 extremes ⇒ exactly 2 rows survive, and every
        // surviving row's flag is true (there is no false-flag row left to
        // find).
        let mut vs: Vec<f64> = (1..=50).map(|i| i as f64).collect();
        vs.push(-1000.0);
        vs.push(1000.0);
        let batch = batch1col("x", vs);
        let spec = LetterValueSpec {
            groupby: Vec::new(),
            value: "x".into(), group: None,
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: None,
            sort: None,
        };
        let primary = apply(&spec, &batch).unwrap();
        let secs = secondary_outputs(&spec, &batch, &primary, &TransformContext::default()).unwrap();
        let outliers = secs.iter().find(|(k, _)| k == "outliers").unwrap();
        assert_eq!(
            outliers.1.num_rows(),
            2,
            "outliers batch must contain only the 2 rows that are actually outliers, not all 52 input rows"
        );
        let flag_col = outliers.1.column(2).as_any().downcast_ref::<BooleanArray>().unwrap();
        for i in 0..outliers.1.num_rows() {
            assert!(flag_col.value(i), "every surviving row must be flagged true");
        }
    }

    #[test]
    fn test_letter_value_named_outputs_count_twelve() {
        // Structural check: one secondary output per declared key
        // (outliers + depth_1..depth_K_MAX_NAMED). The behavioral pin that
        // K_MAX_NAMED is actually 12 — i.e. that the raised cap lets
        // `k_for_depth` produce depths beyond the old 8 — lives in
        // `test_letter_value_k_for_depth_tukey_uses_raised_cap_at_n8192` and
        // `test_letter_value_tukey_n8192_reaches_depth_9_with_continuation_letter`
        // below; this test only pins that every declared key is published.
        let vs: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let batch = batch1col("x", vs);
        let spec = LetterValueSpec {
            groupby: Vec::new(),
            value: "x".into(), group: None,
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: None,
            sort: None,
        };
        let primary = apply(&spec, &batch).unwrap();
        let secs = secondary_outputs(&spec, &batch, &primary, &TransformContext::default()).unwrap();
        assert_eq!(secs.len(), K_MAX_NAMED + 1, "expected outliers + depth_1..depth_{K_MAX_NAMED}");
        for k in 1..=K_MAX_NAMED {
            let key = format!("depth_{k}");
            assert!(secs.iter().any(|(name, _)| name == &key), "missing {key}");
        }
    }

    #[test]
    fn test_letter_value_k_for_depth_tukey_uses_raised_cap_at_n8192() {
        // Regression for F-L03-11: at n=8192, Tukey's raw depth is
        // floor(log2(8192)) - 3 = 13 - 3 = 10, which is inert under the old
        // K_MAX_NAMED=8 clamp (k_for_depth would return 8). The raised cap
        // (12) lets the real depth through unclamped.
        assert_eq!(k_for_depth(&KDepth::Tukey, 8192), 10);
    }

    #[test]
    fn test_letter_value_tukey_n8192_reaches_depth_9_with_continuation_letter() {
        // Companion to the k_for_depth pin above: proves the raised cap
        // carries actual ROWS through `secondary_outputs`, not just keys.
        // At n=8192 (k=10), depth_9 must be a non-empty band (it was
        // unreachable — the key didn't even exist — under the old K_MAX_NAMED=8
        // cap) while depth_12 (> k=10) stays a published-but-empty band, and
        // depth_9's `level` must use the real Tukey continuation letter "Y"
        // rather than the generic `depth_9` fallback spelling.
        let vs: Vec<f64> = (0..8192).map(|i| i as f64).collect();
        let batch = batch1col("x", vs);
        let spec = LetterValueSpec {
            groupby: Vec::new(),
            value: "x".into(), group: None,
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: None,
            sort: None,
        };
        let primary = apply(&spec, &batch).unwrap();
        let secs = secondary_outputs(&spec, &batch, &primary, &TransformContext::default()).unwrap();

        let depth_9 = secs.iter().find(|(k, _)| k == "depth_9")
            .unwrap_or_else(|| panic!("depth_9 output missing — K_MAX_NAMED not raised"));
        assert_eq!(depth_9.1.num_rows(), 1, "depth_9 must carry the row a k=10 group actually produces");
        let level_col = depth_9.1.column(3).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(level_col.value(0), "Y", "depth_9's level must be the Tukey continuation letter, not a 'depth_9' fallback spelling");

        let depth_12 = secs.iter().find(|(k, _)| k == "depth_12")
            .unwrap_or_else(|| panic!("depth_12 output missing — K_MAX_NAMED not raised to 12"));
        assert_eq!(depth_12.1.num_rows(), 0, "depth_12 exceeds this group's actual depth (10) and must stay a published-but-empty band");
    }

    #[test]
    fn test_letter_value_outliers_zero_when_no_outliers() {
        // Edge case from the spec's acceptance criteria ("boxen with zero
        // outliers"): every value sits well within the outermost letter-value
        // fence, so the filtered outliers batch must be zero rows, not a
        // batch of all-false-flag rows.
        let vs: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let batch = batch1col("x", vs);
        let spec = LetterValueSpec {
            groupby: Vec::new(),
            value: "x".into(), group: None,
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: None,
            sort: None,
        };
        let primary = apply(&spec, &batch).unwrap();
        let secs = secondary_outputs(&spec, &batch, &primary, &TransformContext::default()).unwrap();
        let outliers = secs.iter().find(|(k, _)| k == "outliers").unwrap();
        assert_eq!(outliers.1.num_rows(), 0, "no input value falls outside the fence; the batch must be empty");
    }

    #[test]
    fn test_letter_value_named_outputs_use_name_prefix() {
        let vs: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let batch = batch1col("x", vs);
        let spec = LetterValueSpec {
            groupby: Vec::new(),
            value: "x".into(), group: None,
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: Some("lv".into()),
            sort: None,
        };
        let primary = apply(&spec, &batch).unwrap();
        let secs = secondary_outputs(&spec, &batch, &primary, &TransformContext::default()).unwrap();
        assert!(secs.iter().any(|(name, _)| name == "lv_outliers"));
        for k in 1..=K_MAX_NAMED {
            let key = format!("lv_depth_{k}");
            assert!(secs.iter().any(|(name, _)| name == &key), "missing {key}");
        }
    }

    /// First-appearance order of the `group` column in the primary batch.
    /// This is exactly what the categorical-axis domain is built from, so it is
    /// the order that controls the boxen axis.
    fn group_first_appearance(batch: &RecordBatch) -> Vec<String> {
        let g = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for i in 0..batch.num_rows() {
            let s = g.value(i).to_string();
            if seen.insert(s.clone()) {
                out.push(s);
            }
        }
        out
    }

    #[test]
    fn test_letter_value_sort_none_is_lexicographic() {
        // Groups appear in source as c, a, b but unsorted output is lexicographic.
        let groups: Vec<&str> = (0..30).map(|i| ["c", "a", "b"][i % 3]).collect();
        let values: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let batch = batch_grouped(groups, values);
        let spec = LetterValueSpec {
            groupby: Vec::new(),
            value: "v".into(), group: Some("g".into()),
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: None,
            sort: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(group_first_appearance(&out), vec!["a", "b", "c"]);
    }

    #[test]
    fn test_letter_value_sort_sum_desc() {
        // a: [0,1,2] sum=3 ; b: [10,11,12] sum=33 ; c: [5,6,7] sum=18.
        // Descending by sum → b, c, a.
        let groups: Vec<&str> = vec![
            "a", "a", "a", "b", "b", "b", "c", "c", "c",
        ];
        let values: Vec<f64> = vec![0.0, 1.0, 2.0, 10.0, 11.0, 12.0, 5.0, 6.0, 7.0];
        let batch = batch_grouped(groups, values);
        let spec = LetterValueSpec {
            groupby: Vec::new(),
            value: "v".into(), group: Some("g".into()),
            k_depth: KDepth::Full,
            outlier_threshold: 1.5,
            name: None,
            sort: Some(LetterValueSort { op: SortOp::Sum, order: SortOrder::Desc }),
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(group_first_appearance(&out), vec!["b", "c", "a"]);
    }

    #[test]
    fn test_letter_value_sort_tie_breaks_lexicographically() {
        // a and b both sum to 3; c sums to 30. Ascending → a, b (tie, lex), c.
        let groups: Vec<&str> = vec![
            "b", "b", "a", "a", "c", "c",
        ];
        let values: Vec<f64> = vec![1.0, 2.0, 1.0, 2.0, 15.0, 15.0];
        let batch = batch_grouped(groups, values);
        let spec = LetterValueSpec {
            groupby: Vec::new(),
            value: "v".into(), group: Some("g".into()),
            k_depth: KDepth::Full,
            outlier_threshold: 1.5,
            name: None,
            sort: Some(LetterValueSort { op: SortOp::Sum, order: SortOrder::Asc }),
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(group_first_appearance(&out), vec!["a", "b", "c"]);
    }

    #[test]
    fn test_letter_value_sort_count_uses_extracted_rows() {
        // a has 5 non-null rows, b has 2, c has 3 (one b value is NaN → dropped).
        // count descending → a, c, b.
        let groups: Vec<&str> = vec![
            "a", "a", "a", "a", "a", "b", "b", "b", "c", "c", "c",
        ];
        let values: Vec<f64> = vec![
            1.0, 1.0, 1.0, 1.0, 1.0, // a: 5
            2.0, 2.0, f64::NAN,      // b: NaN dropped → 2
            3.0, 3.0, 3.0,           // c: 3
        ];
        let batch = batch_grouped(groups, values);
        let spec = LetterValueSpec {
            groupby: Vec::new(),
            value: "v".into(), group: Some("g".into()),
            k_depth: KDepth::Full,
            outlier_threshold: 1.5,
            name: None,
            sort: Some(LetterValueSort { op: SortOp::Count, order: SortOrder::Desc }),
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(group_first_appearance(&out), vec!["a", "c", "b"]);
    }

    // ── #185: letter_value accepts every dtype its siblings accept ────────

    fn batch_value_int64_group(values: Vec<f64>, groups: Vec<i64>) -> RecordBatch {
        use arrow::array::Int64Array;
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Float64, true),
            Field::new("cyl", DataType::Int64, true),
        ]));
        RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(values)),
            Arc::new(Int64Array::from(groups)),
        ]).unwrap()
    }

    /// RED (pre-#185): a group column of any dtype other than Utf8 refused
    /// with `"stat_letter_value: group column 'cyl' must be Utf8; got Int64"`
    /// — the inverted contract every sibling groupby-taking transform (box_stats,
    /// outliers, swarm, kde_2d, smooth, ...) does NOT share. GREEN: the call
    /// renders, and the output `group` column carries the ORIGINAL Int64 dtype
    /// (not a `.to_string()` shortcut, which would silently re-widen it to Utf8
    /// one layer down — the exact drift #185 exists to close).
    #[test]
    fn letter_value_int64_group_succeeds_and_preserves_dtype() {
        use arrow::array::Int64Array;
        // Group 6: [10,20,30,40,50]; group 8: [1,2,3,4,5].
        let batch = batch_value_int64_group(
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            vec![6, 6, 6, 6, 6, 8, 8, 8, 8, 8],
        );
        let spec = LetterValueSpec {
            groupby: Vec::new(),
            value: "v".into(), group: Some("cyl".into()),
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: None,
            sort: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(
            out.schema().field(0).data_type(),
            &DataType::Int64,
            "the 'group' output column must keep the input's Int64 dtype, not widen to Utf8"
        );
        let groups = out.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let mut seen = std::collections::BTreeSet::new();
        for i in 0..out.num_rows() {
            seen.insert(groups.value(i));
        }
        assert_eq!(seen, std::collections::BTreeSet::from([6, 8]));
    }

    /// A timestamp group column renders like an integer one: the output `group`
    /// column keeps the input's unit and zone, with one block of rows per
    /// distinct instant, listed in the order LetterValue lists its groups.
    #[test]
    fn letter_value_timestamp_group_succeeds_and_preserves_unit_and_zone() {
        pyo3::Python::initialize();
        use arrow::array::TimestampMicrosecondArray;
        let dtype = DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, Some(Arc::from("Europe/Paris")));
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Float64, true),
            Field::new("d", dtype.clone(), true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0, 50.0, 1.0, 2.0, 3.0, 4.0, 5.0])),
                Arc::new(
                    TimestampMicrosecondArray::from(vec![9_i64, 9, 9, 9, 9, 3, 3, 3, 3, 3])
                        .with_timezone("Europe/Paris"),
                ),
            ],
        )
        .unwrap();
        let spec = LetterValueSpec {
            groupby: Vec::new(),
            value: "v".into(), group: Some("d".into()),
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: None,
            sort: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.schema().field(0).data_type(), &dtype);
        let groups = out.column(0).as_any().downcast_ref::<TimestampMicrosecondArray>().unwrap();
        let mut seen: Vec<i64> = Vec::new();
        for i in 0..out.num_rows() {
            if !seen.contains(&groups.value(i)) {
                seen.push(groups.value(i));
            }
        }
        assert_eq!(seen, vec![3, 9], "one block of rows per instant, in the order LetterValue lists its groups");
    }

    /// A group column of an unsupported dtype (e.g. Date32, which no groupby
    /// consumer in the crate accepts) still refuses, naming the column and
    /// dtype. Pins the WORDING word-for-word, not just a substring match:
    /// `extract_values`/`compute_depths` no longer hand-roll their own dtype
    /// gate — they delegate to `group_key::validate_groupby_columns` (via
    /// `extract_values`) and `group_key::group_partition_multi` (via
    /// `compute_depths`, which calls `validate_groupby_columns` internally
    /// too), so the refusal is that SAME shared function's exact text
    /// (`"groupby column '<col>' has unsupported dtype <dt>; supported:
    /// ..."`), not a hand-duplicated near-copy that happened to say "group
    /// column" instead of "groupby column". `apply` reaches this through
    /// `compute_depths`, so this also proves the migration to
    /// `group_partition_multi` (#185 close-review finding 2) still refuses
    /// by name.
    #[test]
    fn letter_value_unsupported_group_dtype_refuses_by_name() {
        pyo3::Python::initialize();
        use arrow::array::Date32Array;
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Float64, true),
            Field::new("d", DataType::Date32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
                Arc::new(Date32Array::from(vec![Some(1), Some(2)])),
            ],
        ).unwrap();
        let spec = LetterValueSpec {
            groupby: Vec::new(),
            value: "v".into(), group: Some("d".into()),
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: None,
            sort: None,
        };
        let err = apply(&spec, &batch).expect_err("a Date32 group column must refuse");
        assert_eq!(
            err.to_string(),
            format!(
                "ValueError: stat_letter_value: groupby column 'd' has unsupported dtype Date32; \
                 supported: {}",
                crate::transform::group_key::GROUPBY_SUPPORTED_DTYPES
            )
        );
    }

    /// Companion to the primary-batch refusal above: `extract_values` (the
    /// path `secondary_outputs` uses for its outlier classification) refuses
    /// with the SAME `validate_groupby_columns` wording, byte-for-byte —
    /// proof the two call sites did not drift onto their own copies of the
    /// message after the shared-toolkit migration.
    #[test]
    fn letter_value_extract_values_unsupported_dtype_matches_compute_depths_wording() {
        pyo3::Python::initialize();
        use arrow::array::Date32Array;
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Float64, true),
            Field::new("d", DataType::Date32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
                Arc::new(Date32Array::from(vec![Some(1), Some(2)])),
            ],
        ).unwrap();
        let spec = LetterValueSpec {
            groupby: Vec::new(),
            value: "v".into(), group: Some("d".into()),
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: None,
            sort: None,
        };
        let err = extract_values(&spec, &batch, &TransformContext::default()).expect_err("a Date32 group column must refuse");
        assert_eq!(
            err.to_string(),
            format!(
                "ValueError: stat_letter_value: groupby column 'd' has unsupported dtype Date32; \
                 supported: {}",
                crate::transform::group_key::GROUPBY_SUPPORTED_DTYPES
            )
        );
    }

    /// The Int64 group dtype survives into the SECONDARY outputs too (outliers
    /// + depth_K bands), not just the primary batch — `secondary_outputs` reads
    /// the group column back off the primary batch via the shared
    /// `groupby_key_at` extractor rather than assuming it is a `StringArray`.
    #[test]
    fn letter_value_secondary_outputs_preserve_int64_group_dtype() {
        use arrow::array::Int64Array;
        let mut vs: Vec<f64> = (1..=50).map(|i| i as f64).collect();
        vs.push(-1000.0);
        vs.push(1000.0);
        let n = vs.len();
        let groups = vec![7i64; n];
        let batch = batch_value_int64_group(vs, groups);
        let spec = LetterValueSpec {
            groupby: Vec::new(),
            value: "v".into(), group: Some("cyl".into()),
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: None,
            sort: None,
        };
        let primary = apply(&spec, &batch).unwrap();
        let secs = secondary_outputs(&spec, &batch, &primary, &TransformContext::default()).unwrap();

        let outliers = secs.iter().find(|(k, _)| k == "outliers").unwrap();
        assert_eq!(outliers.1.schema().field(0).data_type(), &DataType::Int64);
        assert_eq!(outliers.1.num_rows(), 2, "the two extreme values are still classified as outliers");
        let o_groups = outliers.1.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        for i in 0..outliers.1.num_rows() {
            assert_eq!(o_groups.value(i), 7);
        }

        let depth_1 = secs.iter().find(|(k, _)| k == "depth_1").unwrap();
        assert_eq!(depth_1.1.schema().field(0).data_type(), &DataType::Int64);
        assert_eq!(depth_1.1.num_rows(), 1);
        let d_groups = depth_1.1.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(d_groups.value(0), 7);
    }

    /// A null-valued `group` row belongs to no group: it is dropped, with one
    /// warning naming the field and the count, and never forms a group of its
    /// own. 5 rows key to `"a"`, 5 key to a null group value.
    #[test]
    fn letter_value_null_group_value_rows_are_dropped_and_counted() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, true),
            Field::new("v", DataType::Float64, true),
        ]));
        let groups: Vec<Option<&str>> = vec![
            Some("a"), Some("a"), Some("a"), Some("a"), Some("a"),
            None, None, None, None, None,
        ];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0];
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(groups)),
                Arc::new(Float64Array::from(values)),
            ],
        ).unwrap();
        let spec = LetterValueSpec {
            groupby: Vec::new(),
            value: "v".into(), group: Some("g".into()),
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: None,
            sort: None,
        };
        let ctx = TransformContext::default();
        let out = apply_with_context(&spec, &batch, &ctx).unwrap();
        let g_arr = out.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let seen: std::collections::HashSet<Option<String>> = (0..out.num_rows())
            .map(|i| if g_arr.is_null(i) { None } else { Some(g_arr.value(i).to_string()) })
            .collect();
        assert_eq!(seen, std::collections::HashSet::from([Some("a".to_string())]), "no null-keyed group is drawn");
        let warnings: Vec<String> = ctx_warnings(&ctx).iter().map(|w| w.to_string()).collect();
        assert_eq!(
            warnings,
            ["LetterValue: 5 rows have a null 'g' and belong to no group, so they were dropped; \
              fill or filter the nulls in 'g'"]
        );
        let outs = secondary_outputs(&spec, &batch, &out, &TransformContext::default()).unwrap();
        assert!(
            floats(secondary(&outs, OUTLIERS_KEY), "value").iter().all(|v| *v < 10.0),
            "a null-group row is no outlier"
        );
    }

    /// Every `group` value null: the call warns that the data could not be
    /// split into groups, naming `group`, and returns no rows.
    #[test]
    fn letter_value_every_group_value_null_warns_and_returns_no_rows() {
        let batch = keyed_batch(&[("g", vec![None, None, None])], vec![1.0, 2.0, 3.0]);
        let spec = lv_spec(serde_json::json!({
            "value": "v", "group": "g", "k_depth": {"type": "tukey"},
        }));
        let ctx = TransformContext::default();
        let out = apply_with_context(&spec, &batch, &ctx).expect("an emptied call does not refuse");
        assert_eq!(out.num_rows(), 0);
        let warnings: Vec<String> = ctx_warnings(&ctx).iter().map(|w| w.to_string()).collect();
        assert_eq!(warnings, [all_null_warning("g")]);
    }

    /// FA-7-style byte-stability guard for #185: a Utf8 group column produces
    /// the identical output it did before the KeyValue migration — the same
    /// group set, the same per-group median, and the field stays Utf8.
    #[test]
    fn letter_value_utf8_group_byte_stable_after_keyvalue_migration() {
        let groups: Vec<&str> = (0..30).map(|i| ["a", "b", "c"][i % 3]).collect();
        let values: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let batch = batch_grouped(groups, values);
        let spec = LetterValueSpec {
            groupby: Vec::new(),
            value: "v".into(), group: Some("g".into()),
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: None,
            sort: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.schema().field(0).data_type(), &DataType::Utf8);
        assert_eq!(out.num_rows(), 3);
        assert_eq!(group_first_appearance(&out), vec!["a", "b", "c"]);
    }

    /// An all-NaN group never takes part in the sort. Ranked beside the
    /// groups that have values, its `median`/`mean` aggregate is `NaN`,
    /// `NaN.partial_cmp(_)` is `None`, and the comparator degrades to
    /// `Ordering::Equal` against every other group.
    ///
    /// MEASURED, not predicted: with three real groups (a/c/e, medians
    /// 10/50/90) and three all-NaN groups (b/d/f) interleaved
    /// lexicographically between them, `sort=Some(median, Desc)` with the
    /// all-NaN groups in the ranking deterministically produced
    /// `["a", "c", "e"]`: ascending order, the exact OPPOSITE of the
    /// requested descending order, reproduced identically across three
    /// separate runs, with no panic on this toolchain. That is the real
    /// symptom this test pins: not a panic, a silently INVERTED sort. The
    /// all-NaN groups keep a placeholder row each, after the groups with
    /// values, in input order.
    #[test]
    fn letter_value_all_nan_group_does_not_corrupt_sort_order() {
        let groups: Vec<&str> = vec![
            "a", "a", "a",
            "b", "b", "b",
            "c", "c", "c",
            "d", "d", "d",
            "e", "e", "e",
            "f", "f", "f",
        ];
        let values: Vec<f64> = vec![
            9.0, 10.0, 11.0,
            f64::NAN, f64::NAN, f64::NAN,
            49.0, 50.0, 51.0,
            f64::NAN, f64::NAN, f64::NAN,
            89.0, 90.0, 91.0,
            f64::NAN, f64::NAN, f64::NAN,
        ];
        let batch = batch_grouped(groups, values);
        let spec = LetterValueSpec {
            groupby: Vec::new(),
            value: "v".into(), group: Some("g".into()),
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: None,
            sort: Some(LetterValueSort { op: SortOp::Median, order: SortOrder::Desc }),
        };
        let out = apply(&spec, &batch).unwrap();
        // a/c/e's single Tukey depth row each, then one placeholder row for
        // each all-NaN group (b/d/f) in input order.
        assert_eq!(out.num_rows(), 6, "a/c/e's depth row each, then b/d/f's placeholder row each");
        assert_eq!(
            group_first_appearance(&out),
            vec!["e", "c", "a", "b", "d", "f"],
            "descending-by-median order must be e(90), c(50), a(10) -- \
             an all-NaN group's NaN aggregate must not collapse the comparator \
             to Equal and invert the real groups' relative order -- \
             and the placeholders follow in input order"
        );
    }

    /// An all-NaN group keeps its key as one placeholder row (depth 1, the
    /// median's letter, NaN letter values) after every group with values, in
    /// input order, whether or not the groups with values are sorted; the
    /// order of those is what it is without the emptied groups.
    #[test]
    fn an_emptied_group_follows_every_group_with_values() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, true),
            Field::new("v", DataType::Float64, true),
        ]));
        // Input order: z (a null and a NaN), b, a, y (NaN), c.
        let groups: Vec<Option<&str>> = vec![
            Some("z"), Some("z"),
            Some("b"), Some("b"), Some("b"),
            Some("a"), Some("a"), Some("a"),
            Some("y"),
            Some("c"), Some("c"), Some("c"),
        ];
        let values: Vec<Option<f64>> = vec![
            None, Some(f64::NAN),
            Some(8.0), Some(9.0), Some(10.0),
            Some(1.0), Some(2.0), Some(3.0),
            Some(f64::NAN),
            Some(4.0), Some(5.0), Some(6.0),
        ];
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(groups)), Arc::new(Float64Array::from(values))],
        )
        .unwrap();
        for (sort, expected) in [
            (None, ["a", "b", "c", "z", "y"]),
            (Some(LetterValueSort { op: SortOp::Mean, order: SortOrder::Desc }), ["b", "c", "a", "z", "y"]),
            (Some(LetterValueSort { op: SortOp::Mean, order: SortOrder::Asc }), ["a", "c", "b", "z", "y"]),
        ] {
            let spec = LetterValueSpec {
                groupby: Vec::new(),
                value: "v".into(), group: Some("g".into()),
                k_depth: KDepth::Tukey,
                outlier_threshold: 1.5,
                name: None,
                sort,
            };
            let out = apply(&spec, &batch).unwrap();
            assert_eq!(group_first_appearance(&out), expected, "sort {sort:?}");
            assert_eq!(ints(&out, "depth"), vec![1; 5], "sort {sort:?}: one depth-1 row per group");
            let lower = floats(&out, "lower");
            let upper = floats(&out, "upper");
            assert!(lower[..3].iter().chain(&upper[..3]).all(|v| v.is_finite()), "sort {sort:?}");
            assert!(lower[3..].iter().chain(&upper[3..]).all(|v| v.is_nan()), "sort {sort:?}: z and y carry NaN");
            assert_eq!(strs(&out, "level"), some(&["M"; 5]), "sort {sort:?}");
        }
    }

    /// With a `sort` and no group that has a value there is nothing to rank:
    /// every group is a placeholder, in input order (neither sorted by name
    /// nor ranked by row count), whichever aggregate and direction the sort
    /// names, and the call does not refuse.
    #[test]
    fn a_sorted_letter_value_with_no_group_that_has_a_value_returns_the_placeholders_in_input_order() {
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, true),
            Field::new("v", DataType::Float64, true),
        ]));
        // Input order z (2 rows), y (1 row), x (3 rows): the name order (x, y,
        // z) and the row-count order (x, z, y) both differ from it.
        let groups: Vec<Option<&str>> =
            vec![Some("z"), Some("z"), Some("y"), Some("x"), Some("x"), Some("x")];
        let values: Vec<Option<f64>> = vec![
            None, Some(f64::NAN),
            Some(f64::NAN),
            Some(f64::NAN), None, Some(f64::NAN),
        ];
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(groups)), Arc::new(Float64Array::from(values))],
        )
        .unwrap();
        for sort in [
            LetterValueSort { op: SortOp::Count, order: SortOrder::Desc },
            LetterValueSort { op: SortOp::Mean, order: SortOrder::Asc },
        ] {
            let spec = LetterValueSpec {
                groupby: Vec::new(),
                value: "v".into(), group: Some("g".into()),
                k_depth: KDepth::Tukey,
                outlier_threshold: 1.5,
                name: None,
                sort: Some(sort),
            };
            let out = apply(&spec, &batch).unwrap_or_else(|e| panic!("sort {sort:?}: refused: {e}"));
            assert_eq!(group_first_appearance(&out), ["z", "y", "x"], "sort {sort:?}");
            assert_eq!(ints(&out, "depth"), vec![1; 3], "sort {sort:?}: one depth-1 row per group");
            let lower = floats(&out, "lower");
            let upper = floats(&out, "upper");
            assert!(lower.iter().chain(&upper).all(|v| v.is_nan()), "sort {sort:?}: every group carries NaN");
            assert_eq!(strs(&out, "level"), some(&["M"; 3]), "sort {sort:?}");
        }
    }

    /// The placeholder row reaches `depth_1` (the output the median tick and
    /// the innermost band read, where every group has a row) and no other
    /// depth, and no outlier row: its key keeps the category's slot and there
    /// is nothing else to draw.
    #[test]
    fn an_emptied_group_reaches_only_the_depth_one_output() {
        let groups: Vec<&str> = std::iter::repeat_n("a", 10).chain(std::iter::repeat_n("z", 3)).collect();
        let values: Vec<f64> = (0..10).map(f64::from).chain([f64::NAN; 3]).collect();
        let batch = batch_grouped(groups, values);
        let spec = LetterValueSpec {
            groupby: Vec::new(),
            value: "v".into(), group: Some("g".into()),
            k_depth: KDepth::Full,
            outlier_threshold: 0.0,
            name: None,
            sort: None,
        };
        let primary = apply(&spec, &batch).unwrap();
        let outs = secondary_outputs(&spec, &batch, &primary, &TransformContext::default()).unwrap();

        let depth_1 = secondary(&outs, &depth_key(1));
        assert_eq!(strs(depth_1, "group"), some(&["a", "z"]));
        assert_eq!(floats(depth_1, "lower")[0], 4.5);
        assert!(floats(depth_1, "lower")[1].is_nan() && floats(depth_1, "upper")[1].is_nan());
        for k in 2..=3 {
            assert_eq!(strs(secondary(&outs, &depth_key(k)), "group"), some(&["a"]), "depth {k}");
        }
        let outliers = secondary(&outs, OUTLIERS_KEY);
        assert_eq!(floats(outliers, "value"), vec![0.0, 1.0, 8.0, 9.0], "a's values beyond its fence");
        assert_eq!(strs(outliers, "group"), some(&["a"; 4]), "z has no value to be an outlier");
    }

    /// A null `group` value belongs to no group, so a null-keyed row whose
    /// value is unusable is dropped too and keeps no placeholder.
    #[test]
    fn a_null_keyed_row_with_no_usable_value_keeps_no_placeholder() {
        let batch = keyed_batch(
            &[("g", vec![None, None, Some("a"), Some("a"), Some("a")])],
            vec![f64::NAN, f64::NAN, 1.0, 2.0, 3.0],
        );
        let spec = lv_spec(serde_json::json!({
            "value": "v", "group": "g", "k_depth": {"type": "tukey"},
        }));
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(strs(&out, "group"), vec![Some("a".to_string())]);
        assert!(!floats(&out, "lower")[0].is_nan());
    }

    /// A `groupby` field equal to `group` is `group`'s own dimension, so the
    /// placeholder carries the one key in both columns.
    #[test]
    fn an_emptied_group_carries_its_key_in_a_groupby_field_equal_to_group() {
        let batch = keyed_batch(
            &[("g", vec![Some("z"), Some("z"), Some("a"), Some("a"), Some("a")])],
            vec![f64::NAN, f64::NAN, 1.0, 2.0, 3.0],
        );
        let spec = lv_spec(serde_json::json!({
            "value": "v", "group": "g", "groupby": ["g"], "k_depth": {"type": "tukey"},
        }));
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(strs(&out, "group"), some(&["a", "z"]));
        assert_eq!(strs(&out, "g"), some(&["a", "z"]));
        assert!(floats(&out, "lower")[1].is_nan());
    }

    /// An integer group column keeps its dtype on the placeholder row: the
    /// emptied group's key is its own integer, not a stringified one.
    #[test]
    fn an_emptied_int64_group_keeps_its_key_and_dtype() {
        use arrow::array::Int64Array;
        let batch = batch_value_int64_group(
            vec![f64::NAN, f64::NAN, 1.0, 2.0, 3.0, 4.0, 5.0],
            vec![7, 7, 6, 6, 6, 6, 6],
        );
        let spec = LetterValueSpec {
            groupby: Vec::new(),
            value: "v".into(), group: Some("cyl".into()),
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: None,
            sort: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.schema().field(0).data_type(), &DataType::Int64);
        let groups = out.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let keys: Vec<i64> = (0..out.num_rows()).map(|i| groups.value(i)).collect();
        assert_eq!(*keys.last().unwrap(), 7, "the emptied group 7 follows group 6");
        assert!(keys[..keys.len() - 1].iter().all(|&k| k == 6));
        assert!(floats(&out, "lower").last().unwrap().is_nan());
    }

    /// A null `groupby` key whose drop leaves only emptied combinations is not
    /// an every-row drop: the count warns, and the all-NaN combination keeps
    /// its placeholder row.
    #[test]
    fn a_null_key_drop_that_leaves_only_emptied_combinations_warns_the_count_and_keeps_the_placeholder() {
        let batch = keyed_batch(
            &[("g", vec![Some("a"); 3]), ("h", vec![None, Some("x"), Some("x")])],
            vec![1.0, f64::NAN, f64::NAN],
        );
        let spec = lv_spec(serde_json::json!({
            "value": "v", "group": "g", "groupby": ["h"], "k_depth": {"type": "full"},
        }));
        let ctx = TransformContext::default();
        let out = apply_with_context(&spec, &batch, &ctx).expect("a drop that leaves a placeholder does not refuse");
        assert_eq!(strs(&out, "h"), some(&["x"]));
        assert!(floats(&out, "lower")[0].is_nan(), "the emptied combination carries NaN");
        let warnings: Vec<String> = ctx_warnings(&ctx).iter().map(|w| w.to_string()).collect();
        assert_eq!(
            warnings,
            ["LetterValue: 1 row has a null 'h' and belongs to no group, so it was dropped; \
              fill or filter the nulls in 'h'"]
        );
    }

    // ── default fence (outlier_threshold = 0.0) ───────────────────────────

    #[test]
    fn letter_value_outlier_threshold_defaults_to_zero() {
        assert_eq!(default_outlier_threshold(), 0.0);
    }

    #[test]
    fn letter_value_outlier_threshold_defaults_to_zero_on_the_wire() {
        // A wire payload omitting `outlier_threshold` must resolve to the
        // default via `#[serde(default = "default_outlier_threshold")]`,
        // not require every caller to pass it explicitly.
        let json = serde_json::json!({
            "type": "letter_value",
            "value": "x",
            "k_depth": {"type": "tukey"}
        });
        let parsed: TransformSpec = serde_json::from_value(json).unwrap();
        match parsed {
            TransformSpec::LetterValue(s) => assert_eq!(s.outlier_threshold, 0.0),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn letter_value_default_threshold_flags_strictly_beyond_and_not_ties_at_outermost_letter_value() {
        // n=9, KDepth::Full -> k = floor(log2(9)) = 3 (no cap). At depth 3,
        // p = 0.125: h = 0.125*8 = 1.0 (exact) -> lower = sorted[1] = 2.0;
        // 1-p = 0.875: h = 0.875*8 = 7.0 (exact) -> upper = sorted[7] = 8.0.
        // Depth 3 is the outermost (max) depth emitted, so under the
        // default outlier_threshold the fence sits exactly at [2.0, 8.0] --
        // the outermost letter values themselves, with no widening span.
        // Only 1.0 and 9.0 fall strictly outside it; 2.0 and 8.0 are ties
        // with the fence and must not be flagged.
        let vs: Vec<f64> = (1..=9).map(|i| i as f64).collect();
        let batch = batch1col("x", vs);
        let spec = LetterValueSpec {
            groupby: Vec::new(),
            value: "x".into(), group: None,
            k_depth: KDepth::Full,
            outlier_threshold: default_outlier_threshold(),
            name: None,
            sort: None,
        };
        let primary = apply(&spec, &batch).unwrap();
        let secs = secondary_outputs(&spec, &batch, &primary, &TransformContext::default()).unwrap();
        let outliers = secs.iter().find(|(k, _)| k == "outliers").unwrap();
        let sch = outliers.1.schema();
        assert_eq!(sch.field(0).name(), "group");
        assert_eq!(sch.field(1).name(), "value");
        assert_eq!(sch.field(2).name(), "is_outlier");
        let val_col = outliers.1.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        let flag_col = outliers.1.column(2).as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(
            outliers.1.num_rows(), 2,
            "exactly the two values strictly beyond [2.0, 8.0] survive the filter; \
             a tie at the fence (2.0 or 8.0) must not"
        );
        let mut seen: Vec<f64> = (0..outliers.1.num_rows()).map(|i| val_col.value(i)).collect();
        seen.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(seen, vec![1.0, 9.0]);
        for i in 0..outliers.1.num_rows() {
            assert!(flag_col.value(i), "every surviving row must be flagged true");
        }
    }

    #[test]
    fn letter_value_default_threshold_group_with_nothing_beyond_is_empty() {
        // A constant-valued group: every letter-value quantile collapses to
        // the same constant, so the outermost fence is exactly that point
        // and every input row is tied with it, not beyond it -- the
        // outliers layer for this group must be empty.
        let vs: Vec<f64> = std::iter::repeat(5.0).take(10).collect();
        let batch = batch1col("x", vs);
        let spec = LetterValueSpec {
            groupby: Vec::new(),
            value: "x".into(), group: None,
            k_depth: KDepth::Tukey,
            outlier_threshold: default_outlier_threshold(),
            name: None,
            sort: None,
        };
        let primary = apply(&spec, &batch).unwrap();
        let secs = secondary_outputs(&spec, &batch, &primary, &TransformContext::default()).unwrap();
        let outliers = secs.iter().find(|(k, _)| k == "outliers").unwrap();
        assert_eq!(
            outliers.1.num_rows(), 0,
            "nothing lies beyond a constant group's single letter value"
        );
    }

    // ── groupby: one statistic per groupby combination ───────────────────

    fn lv_spec(json: serde_json::Value) -> LetterValueSpec {
        serde_json::from_value(json).expect("a well-formed LetterValue wire spec")
    }

    /// A batch of nullable Utf8 key columns plus a Float64 `v` column.
    fn keyed_batch(keys: &[(&str, Vec<Option<&str>>)], v: Vec<f64>) -> RecordBatch {
        let mut fields: Vec<Field> =
            keys.iter().map(|(name, _)| Field::new(*name, DataType::Utf8, true)).collect();
        fields.push(Field::new("v", DataType::Float64, true));
        let mut cols: Vec<ArrayRef> = keys
            .iter()
            .map(|(_, vals)| Arc::new(StringArray::from(vals.clone())) as ArrayRef)
            .collect();
        cols.push(Arc::new(Float64Array::from(v)));
        RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).unwrap()
    }

    fn strs(b: &RecordBatch, name: &str) -> Vec<Option<String>> {
        let a = b.column_by_name(name).unwrap_or_else(|| panic!("no column '{name}'"));
        let a = a.as_any().downcast_ref::<StringArray>().unwrap();
        (0..a.len()).map(|i| (!a.is_null(i)).then(|| a.value(i).to_string())).collect()
    }

    fn floats(b: &RecordBatch, name: &str) -> Vec<f64> {
        let a = b.column_by_name(name).unwrap_or_else(|| panic!("no column '{name}'"));
        let a = a.as_any().downcast_ref::<Float64Array>().unwrap();
        (0..a.len()).map(|i| a.value(i)).collect()
    }

    fn ints(b: &RecordBatch, name: &str) -> Vec<i32> {
        let a = b.column_by_name(name).unwrap_or_else(|| panic!("no column '{name}'"));
        let a = a.as_any().downcast_ref::<Int32Array>().unwrap();
        (0..a.len()).map(|i| a.value(i)).collect()
    }

    fn some(vals: &[&str]) -> Vec<Option<String>> {
        vals.iter().map(|s| Some(s.to_string())).collect()
    }

    fn secondary<'a>(outs: &'a [(String, RecordBatch)], key: &str) -> &'a RecordBatch {
        &outs.iter().find(|(k, _)| k == key).unwrap_or_else(|| panic!("no output '{key}'")).1
    }

    /// Two hue combinations inside category `a` get their own letter values,
    /// not `a`'s pooled ones: the pooled median of `a` is 4.5, a value
    /// neither combination has.
    #[test]
    fn letter_value_groupby_computes_per_combination_and_carries_the_key() {
        let batch = keyed_batch(
            &[
                ("g", vec![Some("a"); 5].into_iter().chain(vec![Some("a"); 3]).chain(vec![Some("b"); 3]).collect()),
                ("h", vec![Some("x"); 5].into_iter().chain(vec![Some("y"); 3]).chain(vec![Some("x"); 3]).collect()),
            ],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
        );
        let spec = lv_spec(serde_json::json!({
            "value": "v", "group": "g", "groupby": ["h"], "k_depth": {"type": "full"},
        }));
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(strs(&out, "group"), some(&["a", "a", "a", "b"]));
        assert_eq!(strs(&out, "h"), some(&["x", "x", "y", "x"]));
        assert_eq!(ints(&out, "depth"), vec![1, 2, 1, 1]);
        assert_eq!(floats(&out, "lower"), vec![3.0, 2.0, 20.0, 200.0]);
        assert_eq!(floats(&out, "upper"), vec![3.0, 4.0, 20.0, 200.0]);
    }

    /// Two groupby fields partition per combination; with no `group` the
    /// `group` column stays all-null, as an ungrouped call's does.
    #[test]
    fn letter_value_groupby_two_fields_computes_per_combination() {
        let batch = keyed_batch(
            &[
                ("h1", vec![Some("p"), Some("p"), Some("p"), Some("p"), Some("s"), Some("s")]),
                ("h2", vec![Some("q"), Some("q"), Some("q"), Some("r"), Some("q"), Some("q")]),
            ],
            vec![1.0, 2.0, 3.0, 7.0, 5.0, 6.0],
        );
        let spec = lv_spec(serde_json::json!({
            "value": "v", "groupby": ["h1", "h2"], "k_depth": {"type": "full"},
        }));
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(strs(&out, "h1"), some(&["p", "p", "s"]));
        assert_eq!(strs(&out, "h2"), some(&["q", "r", "q"]));
        assert_eq!(floats(&out, "lower"), vec![2.0, 7.0, 5.5]);
        assert_eq!(strs(&out, "group"), vec![None, None, None]);
    }

    /// An absent `groupby` and an explicit empty one are the same spec, and
    /// the empty list is not written back to the wire.
    #[test]
    fn letter_value_empty_groupby_matches_absent_groupby() {
        let batch = keyed_batch(
            &[("g", vec![Some("a"), Some("a"), Some("a"), Some("b"), Some("b")])],
            vec![1.0, 2.0, 30.0, 4.0, 5.0],
        );
        let absent = lv_spec(serde_json::json!({
            "value": "v", "group": "g", "k_depth": {"type": "full"},
        }));
        let empty = lv_spec(serde_json::json!({
            "value": "v", "group": "g", "groupby": [], "k_depth": {"type": "full"},
        }));
        assert_eq!(absent, empty);
        assert!(serde_json::to_value(&empty).unwrap().get("groupby").is_none());
        let a = apply(&absent, &batch).unwrap();
        let e = apply(&empty, &batch).unwrap();
        assert_eq!(a, e);
        let names: Vec<String> = e.schema().fields().iter().map(|f| f.name().clone()).collect();
        assert_eq!(names, ["group", "depth", "lower", "upper", "level"]);
        assert_eq!(
            secondary_outputs(&absent, &batch, &a, &TransformContext::default()).unwrap(),
            secondary_outputs(&empty, &batch, &e, &TransformContext::default()).unwrap(),
        );
    }

    /// The outlier and per-depth batches carry the groupby key too, and the
    /// outlier fence is each combination's own: pooled over `a`, 2.0 would be
    /// an outlier and 10.0 would not.
    #[test]
    fn letter_value_groupby_secondary_outputs_fence_per_combination() {
        let batch = keyed_batch(
            &[
                ("g", vec![Some("a"); 10]),
                ("h", vec![Some("x"); 5].into_iter().chain(vec![Some("y"); 5]).collect()),
            ],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0],
        );
        let spec = lv_spec(serde_json::json!({
            "value": "v", "group": "g", "groupby": ["h"], "k_depth": {"type": "full"},
        }));
        let primary = apply(&spec, &batch).unwrap();
        let outs = secondary_outputs(&spec, &batch, &primary, &TransformContext::default()).unwrap();
        let outliers = secondary(&outs, OUTLIERS_KEY);
        assert_eq!(floats(outliers, "value"), vec![1.0, 5.0, 10.0, 50.0]);
        assert_eq!(strs(outliers, "h"), some(&["x", "x", "y", "y"]));
        let depth_1 = secondary(&outs, &depth_key(1));
        assert_eq!(strs(depth_1, "h"), some(&["x", "y"]));
        assert_eq!(floats(depth_1, "lower"), vec![3.0, 30.0]);
        let depth_2 = secondary(&outs, &depth_key(2));
        assert_eq!(strs(depth_2, "h"), some(&["x", "y"]));
        assert_eq!(floats(depth_2, "upper"), vec![4.0, 40.0]);
    }

    /// A one-row combination computes its own degenerate statistic, and a
    /// combination with no plottable value keeps one placeholder row, carrying
    /// both its keys, after the combinations that have values; neither
    /// disturbs its siblings.
    #[test]
    fn letter_value_groupby_small_combinations_compute_their_own_statistic() {
        let batch = keyed_batch(
            &[
                ("g", vec![Some("a"), Some("a"), Some("a"), Some("b"), Some("b"), Some("b")]),
                ("h", vec![Some("x"), Some("y"), Some("y"), Some("x"), Some("x"), Some("x")]),
            ],
            vec![7.0, f64::NAN, f64::NAN, 1.0, 2.0, 3.0],
        );
        let spec = lv_spec(serde_json::json!({
            "value": "v", "group": "g", "groupby": ["h"], "k_depth": {"type": "full"},
        }));
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(strs(&out, "group"), some(&["a", "b", "a"]));
        assert_eq!(strs(&out, "h"), some(&["x", "x", "y"]));
        let (lower, upper) = (floats(&out, "lower"), floats(&out, "upper"));
        assert_eq!(lower[..2], [7.0, 2.0]);
        assert_eq!(upper[..2], [7.0, 2.0]);
        assert!(lower[2].is_nan() && upper[2].is_nan(), "the emptied combination carries NaN");
    }

    fn ctx_warnings(ctx: &TransformContext) -> Vec<TransformWarning> {
        ctx.warnings.borrow().clone()
    }

    /// A null `groupby` key belongs to no combination: its rows are dropped
    /// from the primary, outlier and depth batches alike, with one warning
    /// naming the field and the count. Kept as a null combination, the
    /// two null rows would draw their own box, and 100 and 200 would sit in
    /// no fence of `x`'s.
    #[test]
    fn letter_value_groupby_null_key_rows_are_dropped_with_a_warning() {
        let batch = keyed_batch(
            &[
                ("g", vec![Some("a"); 5]),
                ("h", vec![None, None, Some("x"), Some("x"), Some("x")]),
            ],
            vec![100.0, 200.0, 1.0, 2.0, 3.0],
        );
        let spec = lv_spec(serde_json::json!({
            "value": "v", "group": "g", "groupby": ["h"], "k_depth": {"type": "full"},
        }));
        let ctx = TransformContext::default();
        let out = apply_with_context(&spec, &batch, &ctx).unwrap();
        assert_eq!(strs(&out, "h"), some(&["x"]));
        assert_eq!(floats(&out, "lower"), vec![2.0]);
        let warnings = ctx_warnings(&ctx);
        assert_eq!(warnings.len(), 1, "got {warnings:?}");
        assert_eq!(
            warnings[0].to_string(),
            "LetterValue: 2 rows have a null 'h' and belong to no group, so they were dropped; \
             fill or filter the nulls in 'h'"
        );

        let outs = secondary_outputs(&spec, &batch, &out, &TransformContext::default()).unwrap();
        let outliers = secondary(&outs, OUTLIERS_KEY);
        assert_eq!(floats(outliers, "value"), vec![1.0, 3.0]);
        assert_eq!(strs(outliers, "h"), some(&["x", "x"]));
        let depth_1 = secondary(&outs, &depth_key(1));
        assert_eq!(strs(depth_1, "h"), some(&["x"]));
        assert_eq!(floats(depth_1, "lower"), vec![2.0]);
    }

    /// Every row's null `groupby` key dropped: the call warns that the data
    /// could not be split into groups and returns no rows, whole frame or
    /// facet partition, with every secondary batch published empty so each
    /// boxen layer still finds its `data_source`.
    #[test]
    fn letter_value_groupby_all_null_keys_warn_and_return_no_rows() {
        let batch = keyed_batch(
            &[("g", vec![Some("a"), Some("a")]), ("h", vec![None, None])],
            vec![1.0, 2.0],
        );
        let spec = lv_spec(serde_json::json!({
            "value": "v", "group": "g", "groupby": ["h"], "k_depth": {"type": "full"},
        }));
        let ctx = TransformContext::default();
        let out = apply_with_context(&spec, &batch, &ctx).expect("an emptied call does not refuse");
        assert_eq!(out.num_rows(), 0);
        let warnings: Vec<String> = ctx_warnings(&ctx).iter().map(|w| w.to_string()).collect();
        assert_eq!(warnings, [all_null_warning("g, h")]);
        assert!(ctx.all_null_partitions.borrow().is_empty(), "a whole frame records no partition");

        let outs = secondary_outputs(&spec, &batch, &out, &TransformContext::default()).unwrap();
        assert_eq!(secondary(&outs, OUTLIERS_KEY).num_rows(), 0);
        for depth in 1..=K_MAX_NAMED {
            assert_eq!(secondary(&outs, &depth_key(depth)).num_rows(), 0, "depth_{depth}");
        }
    }

    /// The text every all-null drop reports, naming `columns`.
    fn all_null_warning(columns: &str) -> String {
        format!(
            "LetterValue: every row has a null in its groupby key ({columns}), so the data could not be \
             split into groups; the grouped output is empty"
        )
    }

    /// A facet partition whose every row the drop removed reports the count
    /// and records the all-null case as `Emptied`, so the facet run decides
    /// once per step; it pushes no all-null warning of its own.
    #[test]
    fn a_facet_partition_the_null_key_drop_empties_records_emptied_beside_the_count() {
        let batch = keyed_batch(
            &[("g", vec![Some("a"), Some("a")]), ("h", vec![None, None])],
            vec![1.0, 2.0],
        );
        let spec = lv_spec(serde_json::json!({
            "value": "v", "group": "g", "groupby": ["h"], "k_depth": {"type": "full"},
        }));
        let ctx = TransformContext { is_partition: true, step: StepInstance::Chain(vec![1]), ..TransformContext::default() };
        let out = apply_with_context(&spec, &batch, &ctx).expect("an emptied panel does not refuse");
        assert_eq!(out.num_rows(), 0);
        let warnings: Vec<String> = ctx_warnings(&ctx).iter().map(|w| w.to_string()).collect();
        assert_eq!(
            warnings,
            ["LetterValue: 2 rows have a null 'h' and belong to no group, so they were dropped; \
              fill or filter the nulls in 'h'"]
        );
        let recorded = ctx.all_null_partitions.borrow();
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].step, StepInstance::Chain(vec![1]));
        match &recorded[0].outcome {
            AllNullOutcome::Emptied(warning) => assert_eq!(warning.to_string(), all_null_warning("g, h")),
            other => panic!("expected the emptied outcome, got {other:?}"),
        }
    }

    /// A facet partition holding a usable row drops its null rows with the
    /// count and records nothing: it is not emptied.
    #[test]
    fn a_facet_partition_the_null_key_drop_leaves_groups_in_records_nothing() {
        let batch = keyed_batch(
            &[("g", vec![Some("a"); 3]), ("h", vec![None, Some("x"), Some("x")])],
            vec![1.0, 2.0, 3.0],
        );
        let spec = lv_spec(serde_json::json!({
            "value": "v", "group": "g", "groupby": ["h"], "k_depth": {"type": "full"},
        }));
        let ctx = TransformContext { is_partition: true, ..TransformContext::default() };
        let out = apply_with_context(&spec, &batch, &ctx).unwrap();
        assert_eq!(strs(&out, "h"), some(&["x"]));
        assert!(ctx.all_null_partitions.borrow().is_empty());
        let warnings: Vec<String> = ctx_warnings(&ctx).iter().map(|w| w.to_string()).collect();
        assert_eq!(
            warnings,
            ["LetterValue: 1 row has a null 'h' and belongs to no group, so it was dropped; \
              fill or filter the nulls in 'h'"]
        );
    }

    /// Two dropping fields whose nulls together cover every row name both
    /// columns in the one all-null warning, in place of the per-field counts.
    #[test]
    fn two_dropping_fields_that_together_cover_every_row_name_both_columns() {
        let batch = keyed_batch(
            &[
                ("g", vec![Some("a"); 4]),
                ("h", vec![None, None, Some("x"), Some("x")]),
                ("d", vec![Some("p"), Some("p"), None, None]),
            ],
            vec![1.0, 2.0, 3.0, 4.0],
        );
        let spec = lv_spec(serde_json::json!({
            "value": "v", "group": "g", "groupby": ["h", "d"], "k_depth": {"type": "full"},
        }));
        let ctx = TransformContext::default();
        let out = apply_with_context(&spec, &batch, &ctx).expect("an emptied call does not refuse");
        assert_eq!(out.num_rows(), 0);
        let warnings: Vec<String> = ctx_warnings(&ctx).iter().map(|w| w.to_string()).collect();
        assert_eq!(warnings, [all_null_warning("g, h, d")]);
    }

    /// A null `group` drops its rows even beside a usable `groupby` field:
    /// every row here is null-keyed, so the call warns once, naming both key
    /// columns, and returns no rows.
    #[test]
    fn a_null_group_drops_its_rows_though_a_groupby_field_is_usable() {
        let batch = keyed_batch(
            &[("g", vec![None, None]), ("h", vec![Some("x"), Some("x")])],
            vec![1.0, 2.0],
        );
        let spec = lv_spec(serde_json::json!({
            "value": "v", "group": "g", "groupby": ["h"], "k_depth": {"type": "full"},
        }));
        let ctx = TransformContext::default();
        let out = apply_with_context(&spec, &batch, &ctx).unwrap();
        assert_eq!(out.num_rows(), 0);
        let warnings: Vec<String> = ctx_warnings(&ctx).iter().map(|w| w.to_string()).collect();
        assert_eq!(warnings, [all_null_warning("g, h")]);
    }

    /// The null-key drop reports through the shared reporter before the
    /// depth cap, so a call that drops rows and clamps a depth warns the count
    /// first, as it always has.
    #[test]
    fn the_null_key_drop_reports_before_the_depth_cap() {
        let n = 30_000_usize;
        let mut h: Vec<Option<&str>> = vec![Some("x"); n];
        h[0] = None;
        let batch = keyed_batch(
            &[("g", vec![Some("a"); n]), ("h", h)],
            (0..n).map(|i| i as f64).collect(),
        );
        let spec = lv_spec(serde_json::json!({
            "value": "v", "group": "g", "groupby": ["h"], "k_depth": {"type": "full"},
        }));
        let ctx = TransformContext::default();
        apply_with_context(&spec, &batch, &ctx).unwrap();
        let warnings = ctx_warnings(&ctx);
        assert!(matches!(warnings[0], TransformWarning::NullKeyRowsDropped { .. }), "got {warnings:?}");
        assert!(matches!(warnings[1], TransformWarning::ParameterClamped { .. }), "got {warnings:?}");
    }

    /// A grouped call that drops nothing behaves exactly like the ungrouped
    /// call: all-NaN values keep one placeholder row per combination (per
    /// group, ungrouped), and no rows at all return an empty batch; neither
    /// refuses nor warns.
    #[test]
    fn letter_value_groupby_without_null_keys_never_refuses_like_ungrouped() {
        let all_nan = keyed_batch(
            &[("g", vec![Some("a"), Some("a")]), ("h", vec![Some("x"), Some("y")])],
            vec![f64::NAN, f64::NAN],
        );
        let no_rows = keyed_batch(&[("g", vec![]), ("h", vec![])], vec![]);
        let grouped = lv_spec(serde_json::json!({
            "value": "v", "group": "g", "groupby": ["h"], "k_depth": {"type": "full"},
        }));
        let ungrouped = lv_spec(serde_json::json!({
            "value": "v", "group": "g", "k_depth": {"type": "full"},
        }));
        for (batch, grouped_rows, ungrouped_rows) in [(&all_nan, 2, 1), (&no_rows, 0, 0)] {
            let ctx = TransformContext::default();
            let out = apply_with_context(&grouped, batch, &ctx).expect("nothing was dropped");
            assert_eq!(out.num_rows(), grouped_rows);
            assert_eq!(apply(&ungrouped, batch).unwrap().num_rows(), ungrouped_rows);
            assert!(ctx_warnings(&ctx).is_empty(), "got {:?}", ctx_warnings(&ctx));
        }
    }

    /// A `groupby` field equal to `group` is `group`'s own dimension: a null
    /// `group` drops its rows, counted once under the one column.
    #[test]
    fn letter_value_groupby_equal_to_group_drops_a_null_group_once() {
        let batch = keyed_batch(&[("g", vec![None, None, Some("a")])], vec![1.0, 3.0, 9.0]);
        let grouped = lv_spec(serde_json::json!({
            "value": "v", "group": "g", "groupby": ["g"], "k_depth": {"type": "full"},
        }));
        let ctx = TransformContext::default();
        let out = apply_with_context(&grouped, &batch, &ctx).unwrap();
        assert_eq!(strs(&out, "group"), vec![Some("a".to_string())]);
        assert_eq!(strs(&out, "g"), vec![Some("a".to_string())]);
        assert_eq!(floats(&out, "lower"), vec![9.0]);
        let warnings: Vec<String> = ctx_warnings(&ctx).iter().map(|w| w.to_string()).collect();
        assert_eq!(
            warnings,
            ["LetterValue: 2 rows have a null 'g' and belong to no group, so they were dropped; \
              fill or filter the nulls in 'g'"]
        );
    }

    /// The depth-cap label names each distinct key column once, so
    /// `groupby=[group]` labels exactly as `group` alone does, for a real and
    /// for a null `group` value alike (a null `group` is dropped, so it labels nothing).
    #[test]
    fn letter_value_groupby_equal_to_group_labels_as_group_alone() {
        let clamped_labels = |spec: &LetterValueSpec, batch: &RecordBatch| -> Vec<Option<String>> {
            let ctx = TransformContext::default();
            apply_with_context(spec, batch, &ctx).unwrap();
            ctx_warnings(&ctx)
                .into_iter()
                .filter_map(|w| match w {
                    TransformWarning::ParameterClamped { group, .. } => Some(group),
                    _ => None,
                })
                .collect()
        };
        let alone = lv_spec(serde_json::json!({
            "value": "v", "group": "g", "k_depth": {"type": "full"},
        }));
        let grouped = lv_spec(serde_json::json!({
            "value": "v", "group": "g", "groupby": ["g"], "k_depth": {"type": "full"},
        }));
        let values: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
        let named = keyed_batch(&[("g", vec![Some("big"); 10_000])], values.clone());
        assert_eq!(clamped_labels(&grouped, &named), vec![Some("g=big".to_string())]);
        assert_eq!(clamped_labels(&grouped, &named), clamped_labels(&alone, &named));
        let null = keyed_batch(&[("g", vec![None; 10_000])], values);
        assert_eq!(clamped_labels(&grouped, &null), Vec::<Option<String>>::new(), "a null group is dropped, so it has no depth to clamp");
        assert_eq!(clamped_labels(&grouped, &null), clamped_labels(&alone, &null));
    }

    /// With `groupby` naming `group`, the outlier and depth batches carry the
    /// category under its own name too, equal to their `group` column.
    #[test]
    fn letter_value_groupby_equal_to_group_secondary_outputs_carry_it() {
        let batch = keyed_batch(
            &[("g", vec![Some("a"); 5].into_iter().chain(vec![Some("b"); 5]).collect())],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0],
        );
        let spec = lv_spec(serde_json::json!({
            "value": "v", "group": "g", "groupby": ["g"], "k_depth": {"type": "full"},
        }));
        let primary = apply(&spec, &batch).unwrap();
        let outs = secondary_outputs(&spec, &batch, &primary, &TransformContext::default()).unwrap();
        let outliers = secondary(&outs, OUTLIERS_KEY);
        assert_eq!(floats(outliers, "value"), vec![1.0, 5.0, 10.0, 50.0]);
        assert_eq!(strs(outliers, "g"), some(&["a", "a", "b", "b"]));
        assert_eq!(strs(outliers, "g"), strs(outliers, "group"));
        let depth_2 = secondary(&outs, &depth_key(2));
        assert_eq!(strs(depth_2, "g"), some(&["a", "b"]));
        assert_eq!(strs(depth_2, "g"), strs(depth_2, "group"));
    }

    /// With `groupby` but no `group`, the secondary batches keep their
    /// all-null `group` column and carry the `groupby` key.
    #[test]
    fn letter_value_groupby_without_group_secondary_outputs_carry_it() {
        let batch = keyed_batch(
            &[("h", vec![Some("x"); 5].into_iter().chain(vec![Some("y"); 5]).collect())],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0],
        );
        let spec = lv_spec(serde_json::json!({
            "value": "v", "groupby": ["h"], "k_depth": {"type": "full"},
        }));
        let primary = apply(&spec, &batch).unwrap();
        let outs = secondary_outputs(&spec, &batch, &primary, &TransformContext::default()).unwrap();
        let outliers = secondary(&outs, OUTLIERS_KEY);
        assert_eq!(floats(outliers, "value"), vec![1.0, 5.0, 10.0, 50.0]);
        assert_eq!(strs(outliers, "h"), some(&["x", "x", "y", "y"]));
        assert_eq!(strs(outliers, "group"), vec![None; 4]);
        let depth_1 = secondary(&outs, &depth_key(1));
        assert_eq!(strs(depth_1, "h"), some(&["x", "y"]));
        assert_eq!(strs(depth_1, "group"), vec![None, None]);
        assert_eq!(floats(depth_1, "lower"), vec![3.0, 30.0]);
    }

    /// A non-text `groupby` key keeps its input type on every batch.
    #[test]
    fn letter_value_groupby_int_key_keeps_its_type() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("v", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(arrow::array::Int64Array::from(vec![7, 7, 7, 3])) as ArrayRef,
            Arc::new(Float64Array::from(vec![1.0, 2.0, 30.0, 4.0])),
        ]).unwrap();
        let spec = lv_spec(serde_json::json!({
            "value": "v", "groupby": ["k"], "k_depth": {"type": "full"},
        }));
        let primary = apply(&spec, &batch).unwrap();
        let outs = secondary_outputs(&spec, &batch, &primary, &TransformContext::default()).unwrap();
        for (name, b) in std::iter::once(("primary", &primary)).chain(outs.iter().map(|(k, b)| (k.as_str(), b))) {
            let field = b.schema().field_with_name("k").unwrap_or_else(|_| panic!("{name} lacks 'k'")).clone();
            assert_eq!(field.data_type(), &DataType::Int64, "{name}");
        }
        let k = primary.column_by_name("k").unwrap().as_any().downcast_ref::<arrow::array::Int64Array>().unwrap();
        assert_eq!(k.values().to_vec(), vec![3, 7]);
    }

    #[test]
    fn letter_value_groupby_missing_column_refuses() {
        pyo3::Python::initialize();
        let batch = keyed_batch(&[("g", vec![Some("a")])], vec![1.0]);
        let spec = lv_spec(serde_json::json!({
            "value": "v", "group": "g", "groupby": ["ghost"], "k_depth": {"type": "full"},
        }));
        let err = apply(&spec, &batch).unwrap_err().to_string();
        assert!(
            err.contains("stat_letter_value: groupby column 'ghost' not found"),
            "{err}"
        );
    }

    /// A groupby column named like one of this transform's output columns
    /// would make every reader of that column read the wrong one.
    #[test]
    fn letter_value_groupby_output_collision_refuses() {
        pyo3::Python::initialize();
        for taken in ["group", "lower", "value", "is_outlier"] {
            let batch = keyed_batch(&[(taken, vec![Some("a")])], vec![1.0]);
            let spec = lv_spec(serde_json::json!({
                "value": "v", "groupby": [taken], "k_depth": {"type": "full"},
            }));
            let err = apply(&spec, &batch).unwrap_err().to_string();
            assert!(err.contains("collides"), "{taken}: {err}");
        }
    }

    /// `sort` ranks the categorical-axis categories by their pooled
    /// aggregate, then lists each category's combinations in key order.
    /// Ranking each combination on its own would put `a` first (its `x`
    /// combination sums to 100, the largest), though `b`'s pooled sum (120)
    /// beats `a`'s (101).
    #[test]
    fn letter_value_groupby_sort_ranks_categories_by_pooled_aggregate() {
        let batch = keyed_batch(
            &[
                ("g", vec![Some("a"), Some("a"), Some("b"), Some("b")]),
                ("h", vec![Some("x"), Some("y"), Some("x"), Some("y")]),
            ],
            vec![100.0, 1.0, 60.0, 60.0],
        );
        let spec = lv_spec(serde_json::json!({
            "value": "v", "group": "g", "groupby": ["h"], "k_depth": {"type": "full"},
            "sort": {"op": "sum", "order": "desc"},
        }));
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(strs(&out, "group"), some(&["b", "b", "a", "a"]));
        assert_eq!(strs(&out, "h"), some(&["x", "y", "x", "y"]));
    }

    /// The depth-cap warning names the whole combination whose depth was
    /// capped.
    #[test]
    fn letter_value_groupby_depth_cap_names_the_combination() {
        let batch = keyed_batch(
            &[("g", vec![Some("big"); 10_000]), ("h", vec![Some("x"); 10_000])],
            (0..10_000).map(|i| i as f64).collect(),
        );
        let spec = lv_spec(serde_json::json!({
            "value": "v", "group": "g", "groupby": ["h"], "k_depth": {"type": "full"},
        }));
        let ctx = TransformContext::default();
        apply_with_context(&spec, &batch, &ctx).unwrap();
        let warnings = ctx.warnings.borrow();
        assert_eq!(warnings.len(), 1, "got {warnings:?}");
        assert!(
            matches!(
                &warnings[0],
                TransformWarning::ParameterClamped { group: Some(g), .. } if g == "g=big, h=x"
            ),
            "got {warnings:?}"
        );
    }

    /// A `groupby` field equal to `group` carries the category under its
    /// own name without partitioning by it twice.
    #[test]
    fn letter_value_groupby_may_name_the_group_column() {
        let batch = keyed_batch(
            &[("g", vec![Some("a"), Some("a"), Some("a"), Some("b")])],
            vec![1.0, 2.0, 3.0, 9.0],
        );
        let spec = lv_spec(serde_json::json!({
            "value": "v", "group": "g", "groupby": ["g"], "k_depth": {"type": "full"},
        }));
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(strs(&out, "group"), some(&["a", "b"]));
        assert_eq!(strs(&out, "g"), some(&["a", "b"]));
        assert_eq!(floats(&out, "lower"), vec![2.0, 9.0]);
    }

    #[test]
    fn letter_value_constructor_takes_groupby_and_refuses_a_repeat() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let once = pyo3::types::PyList::new(py, ["h"]).unwrap();
            let ok = PyLetterValue::new("v", Some("g".into()), Some(once.as_any()), "tukey", 0.007, 0.0, None, None)
                .expect("a groupby list constructs");
            assert!(matches!(&ok.0, TransformSpec::LetterValue(s) if s.groupby == ["h"]));
            let twice = pyo3::types::PyList::new(py, ["h", "h"]).unwrap();
            let err = PyLetterValue::new("v", None, Some(twice.as_any()), "tukey", 0.007, 0.0, None, None)
                .err()
                .expect("a repeated groupby name refuses")
                .to_string();
            assert!(err.contains("LetterValue: groupby column 'h' is repeated"), "{err}");
        });
    }

    /// A value that is infinite is skipped as a null value is: the letter
    /// values, the outlier rows and the all-unusable group's placeholder are
    /// the null twin's, so one boxen outlier is not lost to the extra count.
    #[test]
    fn an_infinite_value_is_skipped_as_a_null_value_is() {
        pyo3::Python::initialize();
        let frame = |last: Option<f64>| {
            let schema = Arc::new(Schema::new(vec![
                Field::new("g", DataType::Utf8, true),
                Field::new("v", DataType::Float64, true),
            ]));
            let mut values: Vec<Option<f64>> = (1..=40).map(|i| Some(i as f64)).collect();
            values.extend([Some(-1000.0), Some(1000.0), last]);
            let mut groups = vec!["a"; 43];
            values.push(last);
            groups.push("b");
            RecordBatch::try_new(
                schema,
                vec![Arc::new(StringArray::from(groups)), Arc::new(Float64Array::from(values))],
            )
            .unwrap()
        };
        let spec = LetterValueSpec {
            groupby: Vec::new(),
            value: "v".into(),
            group: Some("g".into()),
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: None,
            sort: None,
        };
        let run = |batch: &RecordBatch| {
            let ctx = TransformContext::default();
            let primary = apply_with_context(&spec, batch, &ctx).unwrap();
            let secondary = secondary_outputs(&spec, batch, &primary, &ctx).unwrap();
            (primary, secondary)
        };
        let twin = run(&frame(None));
        let outliers = twin.1.iter().find(|(k, _)| k == "outliers").expect("an outliers batch");
        assert_eq!(outliers.1.num_rows(), 2, "twin sanity: the two extremes are its outliers");
        for infinite in [f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(run(&frame(Some(infinite))), twin);
        }
    }
}
