//! Letter-value transform (Phase 9b).
//!
//! Computes the letter-value statistics used by boxen plots
//! (Hofmann/Wickham/Kafadar 2017). For each group, emits one row per
//! depth k ∈ [1, K] with the lower/upper quantiles at probabilities
//! `2^-k` and `1 - 2^-k`. The number of depths K is chosen by one of
//! four strategies.
//!
//! Primary output schema:
//! `[group: Utf8|Null, depth: Int32, lower: Float64, upper: Float64, level: Utf8]`
//!
//! Secondary named outputs:
//! - `outliers` (or `<name>_outliers`): `[group: Utf8|Null, value: Float64, is_outlier: Bool]`
//! - `depth_1`..`depth_8` (or `<name>_depth_K`): `[group, lower, upper, level]` rows
//!   filtered to that depth. When the actual K < 8, the unused outputs are emitted
//!   as zero-row batches so downstream renderers can safely consume them.

use arrow::array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int32Array, RecordBatch, StringArray,
};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Maximum depth for per-depth named outputs.
pub(crate) const K_MAX_NAMED: usize = 8;

/// Letter labels by depth. depth 1 → "M" (median), 2 → "F" (fourths), etc.
const LETTERS: [&str; 8] = ["M", "F", "E", "D", "C", "B", "A", "Z"];

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
pub(crate) struct LetterValueSort {
    pub op: SortOp,
    pub order: SortOrder,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct LetterValueSpec {
    pub value: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub group: Option<String>,
    pub k_depth: KDepth,
    #[serde(default = "default_outlier_threshold")]
    pub outlier_threshold: f64,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub sort: Option<LetterValueSort>,
}

fn default_outlier_threshold() -> f64 { 1.5 }

/// Compute the K depth for a given strategy and group size n.
pub(crate) fn k_for_depth(strategy: &KDepth, n: usize) -> usize {
    if n == 0 { return 0; }
    let n_f = n as f64;
    let k = match strategy {
        KDepth::Tukey => {
            let raw = n_f.log2().floor() as i64 - 3;
            raw.max(1) as usize
        }
        KDepth::Proportion { p } => {
            let target = (p * n_f).ceil() as i64;
            let target = target.max(1);
            let mut k = 1usize;
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
    };
    k.min(K_MAX_NAMED)
}

/// numpy-style linear-interpolation quantile on already-sorted ascending values.
fn quantile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 { return f64::NAN; }
    if n == 1 { return sorted[0]; }
    let h = p * (n as f64 - 1.0);
    let lo = h.floor() as usize;
    let hi = (h.ceil() as usize).min(n - 1);
    let frac = h - h.floor();
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

struct DepthRow {
    group: Option<String>,
    depth: i32,
    lower: f64,
    upper: f64,
    level: String,
}

fn extract_values(spec: &LetterValueSpec, batch: &RecordBatch) -> PyResult<Vec<(Option<String>, f64)>> {
    let schema = batch.schema();
    let vi = schema.index_of(&spec.value)
        .map_err(|_| PyValueError::new_err(format!("stat_letter_value: column '{}' not found", spec.value)))?;
    if schema.field(vi).data_type() != &DataType::Float64 {
        return Err(PyValueError::new_err(format!(
            "stat_letter_value: column '{}' must be Float64", spec.value
        )));
    }
    let varr = batch.column(vi).as_any().downcast_ref::<Float64Array>()
        .ok_or_else(|| PyValueError::new_err(format!(
            "stat_letter_value: expected Float64Array for column '{}'", spec.value)))?;

    let g_arr_opt = if let Some(g) = &spec.group {
        let gi = schema.index_of(g.as_str())
            .map_err(|_| PyValueError::new_err(format!("stat_letter_value: group column '{g}' not found")))?;
        let dt = schema.field(gi).data_type();
        if dt != &DataType::Utf8 {
            return Err(PyValueError::new_err(format!(
                "stat_letter_value: group column '{g}' must be Utf8; got {dt:?}"
            )));
        }
        Some(batch.column(gi).as_any().downcast_ref::<StringArray>()
            .ok_or_else(|| PyValueError::new_err(format!(
                "stat_letter_value: expected StringArray for group column '{g}'")))?)
    } else {
        None
    };

    let mut out = Vec::with_capacity(varr.len());
    for i in 0..varr.len() {
        if varr.is_null(i) { continue; }
        let v = varr.value(i);
        if v.is_nan() { continue; }
        let g = match g_arr_opt {
            None => None,
            Some(ga) => {
                if ga.is_null(i) { None } else { Some(ga.value(i).to_string()) }
            }
        };
        out.push((g, v));
    }
    Ok(out)
}

pub(crate) fn apply(spec: &LetterValueSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    if !(spec.outlier_threshold.is_finite() && spec.outlier_threshold >= 0.0) {
        return Err(PyValueError::new_err(
            "stat_letter_value: outlier_threshold must be finite and >= 0",
        ));
    }
    let pairs = extract_values(spec, batch)?;
    let depths = compute_depths(spec, &pairs);
    build_primary_batch(&depths)
}

/// Determine the order in which groups are emitted.
///
/// With `sort = None` the order is lexicographic over the (optional) group key
/// — identical to the historical `BTreeMap` iteration order. With `sort =
/// Some`, groups are ranked by the requested per-group aggregate over the SAME
/// `(group, value)` pairs the transform extracts (so `count` counts only the
/// non-null value rows that boxen actually plots). Ties are broken
/// lexicographically for byte-deterministic output.
fn group_order(
    sort: &Option<LetterValueSort>,
    grouped: &BTreeMap<Option<String>, Vec<f64>>,
) -> Vec<Option<String>> {
    // BTreeMap keys are already lexicographically ordered; this is the
    // `sort = None` answer and the lexicographic tie-break baseline.
    let lexicographic: Vec<Option<String>> = grouped.keys().cloned().collect();
    let Some(sort) = sort else {
        return lexicographic;
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

    // Rank lexicographic order by aggregate; stable sort keeps lexicographic
    // order among ties, then we reverse for descending.
    let mut ordered = lexicographic;
    ordered.sort_by(|a, b| {
        let av = aggregate(grouped.get(a).map(Vec::as_slice).unwrap_or(&[]));
        let bv = aggregate(grouped.get(b).map(Vec::as_slice).unwrap_or(&[]));
        let cmp = av.partial_cmp(&bv).unwrap_or(std::cmp::Ordering::Equal);
        match sort.order {
            SortOrder::Asc => cmp,
            SortOrder::Desc => cmp.reverse(),
        }
        // Tie-break: BTreeMap order is already lexicographic and `sort_by` is
        // stable, so equal aggregates retain ascending-lexicographic order
        // under both directions.
    });
    ordered
}

fn compute_depths(spec: &LetterValueSpec, pairs: &[(Option<String>, f64)]) -> Vec<DepthRow> {
    // Group values; BTreeMap keys give a deterministic lexicographic baseline.
    let mut groups: BTreeMap<Option<String>, Vec<f64>> = BTreeMap::new();
    for (g, v) in pairs {
        groups.entry(g.clone()).or_insert_with(Vec::new).push(*v);
    }
    for vs in groups.values_mut() {
        vs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }
    let order = group_order(&spec.sort, &groups);

    let mut rows = Vec::new();
    for g in &order {
        let vs = match groups.get(g) {
            Some(vs) if !vs.is_empty() => vs,
            _ => continue,
        };
        let k = k_for_depth(&spec.k_depth, vs.len());
        for depth in 1..=k {
            let p = 0.5_f64.powi(depth as i32);
            let lower = quantile_sorted(vs, p);
            let upper = quantile_sorted(vs, 1.0 - p);
            let letter = if depth >= 1 && depth <= LETTERS.len() {
                LETTERS[depth - 1].to_string()
            } else {
                format!("depth_{depth}")
            };
            rows.push(DepthRow {
                group: g.clone(),
                depth: depth as i32,
                lower,
                upper,
                level: letter,
            });
        }
    }
    rows
}

fn build_primary_batch(rows: &[DepthRow]) -> PyResult<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("group", DataType::Utf8, true),
        Field::new("depth", DataType::Int32, false),
        Field::new("lower", DataType::Float64, false),
        Field::new("upper", DataType::Float64, false),
        Field::new("level", DataType::Utf8, false),
    ]));
    let groups: Vec<Option<String>> = rows.iter().map(|r| r.group.clone()).collect();
    let depths: Vec<i32> = rows.iter().map(|r| r.depth).collect();
    let lowers: Vec<f64> = rows.iter().map(|r| r.lower).collect();
    let uppers: Vec<f64> = rows.iter().map(|r| r.upper).collect();
    let levels: Vec<String> = rows.iter().map(|r| r.level.clone()).collect();
    let cols: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(groups.iter().map(|x| x.as_deref()).collect::<Vec<_>>())),
        Arc::new(Int32Array::from(depths)),
        Arc::new(Float64Array::from(lowers)),
        Arc::new(Float64Array::from(uppers)),
        Arc::new(StringArray::from(levels.iter().map(|s| s.as_str()).collect::<Vec<_>>())),
    ];
    RecordBatch::try_new(schema, cols)
        .map_err(|e| PyValueError::new_err(format!("stat_letter_value: {e}")))
}

/// Compute the secondary named outputs: outliers + depth_1..depth_8.
pub(crate) fn secondary_outputs(
    spec: &LetterValueSpec,
    input_batch: &RecordBatch,
    primary_batch: &RecordBatch,
) -> PyResult<Vec<(String, RecordBatch)>> {
    let groups_col = primary_batch.column(0).as_any().downcast_ref::<StringArray>()
        .ok_or_else(|| PyValueError::new_err("stat_letter_value: expected StringArray for 'group' column"))?;
    let depths_col = primary_batch.column(1).as_any().downcast_ref::<Int32Array>()
        .ok_or_else(|| PyValueError::new_err("stat_letter_value: expected Int32Array for 'depth' column"))?;
    let lowers_col = primary_batch.column(2).as_any().downcast_ref::<Float64Array>()
        .ok_or_else(|| PyValueError::new_err("stat_letter_value: expected Float64Array for 'lower' column"))?;
    let uppers_col = primary_batch.column(3).as_any().downcast_ref::<Float64Array>()
        .ok_or_else(|| PyValueError::new_err("stat_letter_value: expected Float64Array for 'upper' column"))?;

    // Per-group max-depth (lower, upper) extents — the outermost letter value.
    let mut tail: BTreeMap<Option<String>, (i32, f64, f64)> = BTreeMap::new();
    for i in 0..primary_batch.num_rows() {
        let g = if groups_col.is_null(i) { None } else { Some(groups_col.value(i).to_string()) };
        let d = depths_col.value(i);
        let lo = lowers_col.value(i);
        let hi = uppers_col.value(i);
        let entry = tail.entry(g).or_insert((d, lo, hi));
        if d >= entry.0 {
            *entry = (d, lo, hi);
        }
    }

    // ---- outliers output ----
    // For each non-null input row: classify as outlier if value falls outside
    //   [lower_K - threshold*(upper_K - lower_K), upper_K + threshold*(upper_K - lower_K)].
    let input_pairs = extract_values(spec, input_batch)?;
    let mut o_groups: Vec<Option<String>> = Vec::with_capacity(input_pairs.len());
    let mut o_vals: Vec<f64> = Vec::with_capacity(input_pairs.len());
    let mut o_flags: Vec<bool> = Vec::with_capacity(input_pairs.len());
    for (g, v) in &input_pairs {
        let (lo, hi) = match tail.get(g) {
            Some((_, lo, hi)) => (*lo, *hi),
            None => (f64::NAN, f64::NAN),
        };
        let span = hi - lo;
        let low_fence = lo - spec.outlier_threshold * span;
        let high_fence = hi + spec.outlier_threshold * span;
        let is_outlier = !lo.is_nan() && !hi.is_nan() && (*v < low_fence || *v > high_fence);
        o_groups.push(g.clone());
        o_vals.push(*v);
        o_flags.push(is_outlier);
    }
    let outliers_schema = Arc::new(Schema::new(vec![
        Field::new("group", DataType::Utf8, true),
        Field::new("value", DataType::Float64, false),
        Field::new("is_outlier", DataType::Boolean, false),
    ]));
    let outliers_batch = RecordBatch::try_new(
        outliers_schema,
        vec![
            Arc::new(StringArray::from(o_groups.iter().map(|x| x.as_deref()).collect::<Vec<_>>())) as ArrayRef,
            Arc::new(Float64Array::from(o_vals)) as ArrayRef,
            Arc::new(BooleanArray::from(o_flags)) as ArrayRef,
        ],
    ).map_err(|e| PyValueError::new_err(format!("stat_letter_value outliers: {e}")))?;

    // ---- per-depth outputs (depth_1..depth_8) ----
    // Each depth_K output is the primary rows filtered to depth==K, projecting only
    // [group, lower, upper, level] (no depth column — implicit in the output name).
    let depth_schema = Arc::new(Schema::new(vec![
        Field::new("group", DataType::Utf8, true),
        Field::new("lower", DataType::Float64, false),
        Field::new("upper", DataType::Float64, false),
        Field::new("level", DataType::Utf8, false),
    ]));
    let level_col = primary_batch.column(4).as_any().downcast_ref::<StringArray>()
        .ok_or_else(|| PyValueError::new_err("stat_letter_value: expected StringArray for 'level' column"))?;
    let mut by_depth: BTreeMap<i32, (Vec<Option<String>>, Vec<f64>, Vec<f64>, Vec<String>)> = BTreeMap::new();
    for i in 0..primary_batch.num_rows() {
        let d = depths_col.value(i);
        let g = if groups_col.is_null(i) { None } else { Some(groups_col.value(i).to_string()) };
        let entry = by_depth.entry(d).or_insert_with(|| (Vec::new(), Vec::new(), Vec::new(), Vec::new()));
        entry.0.push(g);
        entry.1.push(lowers_col.value(i));
        entry.2.push(uppers_col.value(i));
        entry.3.push(level_col.value(i).to_string());
    }

    let (key_outliers, key_depth_fmt): (String, Box<dyn Fn(usize) -> String>) = match &spec.name {
        Some(s) => {
            let s_cloned = s.clone();
            let fmt: Box<dyn Fn(usize) -> String> = Box::new(move |k: usize| format!("{s_cloned}_depth_{k}"));
            (format!("{s}_outliers"), fmt)
        }
        None => {
            let fmt: Box<dyn Fn(usize) -> String> = Box::new(|k: usize| format!("depth_{k}"));
            ("outliers".to_string(), fmt)
        }
    };

    let mut out: Vec<(String, RecordBatch)> = Vec::new();
    out.push((key_outliers, outliers_batch));
    for k in 1..=K_MAX_NAMED {
        let (groups_v, lowers_v, uppers_v, levels_v) = by_depth
            .get(&(k as i32))
            .map(|t| t.clone())
            .unwrap_or_else(|| (Vec::new(), Vec::new(), Vec::new(), Vec::new()));
        let cols: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(groups_v.iter().map(|x| x.as_deref()).collect::<Vec<_>>())) as ArrayRef,
            Arc::new(Float64Array::from(lowers_v)) as ArrayRef,
            Arc::new(Float64Array::from(uppers_v)) as ArrayRef,
            Arc::new(StringArray::from(levels_v.iter().map(|s| s.as_str()).collect::<Vec<_>>())) as ArrayRef,
        ];
        let batch = RecordBatch::try_new(depth_schema.clone(), cols)
            .map_err(|e| PyValueError::new_err(format!("stat_letter_value depth_{k}: {e}")))?;
        out.push((key_depth_fmt(k), batch));
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
/// Output columns: ``letter``, ``value_lo``, ``value_hi`` (Float64 edges
/// of each letter-value interval) plus ``is_outlier`` (Bool) for the
/// ``Outliers``-equivalent rows, optionally preceded by the ``group``
/// column.
///
/// Parameters
/// ----------
/// value : str
///     Numeric column to summarize (must be Float64).
/// group : str, optional
///     Single group-key column; letter-values computed independently per
///     group.
/// k_depth : {"tukey", "proportion", "trustworthy", "full"}, \
///           default "proportion"
///     Algorithm for choosing the number of letter-value levels:
///     ``"tukey"`` — fixed integer depth; ``"proportion"`` — fraction of
///     sample size (see ``k_proportion``); ``"trustworthy"`` — based on
///     statistical trustworthiness; ``"full"`` — all levels to median.
/// k_proportion : float, default 0.007
///     Fraction of sample size used when ``k_depth="proportion"``. Must
///     be in (0, 1).
/// outlier_threshold : float, default 1.5
///     Tukey fence multiplier for labelling outliers. Must be ≥ 0.
/// name : str, optional
///     Named output label for sibling ``Reorder(from_=...)`` lookup.
/// sort : dict, optional
///     Group ordering for the categorical axis, as ``{"op": ..., "order":
///     ...}``. ``op`` is one of ``"sum" | "mean" | "max" | "min" | "count" |
///     "median"`` and ``order`` is ``"ascending" | "descending"`` (``"asc"``
///     / ``"desc"`` also accepted). The aggregate is computed over the value
///     column per group; ``count`` counts non-null value rows. Groups are then
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
        k_depth = "proportion",
        k_proportion = 0.007,
        outlier_threshold = 1.5,
        name = None,
        sort = None,
    ))]
    fn new(
        value: &str,
        group: Option<String>,
        k_depth: &str,
        k_proportion: f64,
        outlier_threshold: f64,
        name: Option<String>,
        sort: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        if value.is_empty() {
            return Err(PyValueError::new_err("LetterValue: value must be non-empty"));
        }
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
            k_depth: kd,
            outlier_threshold,
            name,
            sort,
        })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::LetterValue(s) => format!(
                "LetterValue(value='{}', group={:?}, k_depth={:?}, outlier_threshold={}, sort={:?})",
                s.value, s.group, s.k_depth, s.outlier_threshold, s.sort,
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, StringArray};

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

    #[test]
    fn test_letter_value_round_trip_json() {
        let spec = LetterValueSpec {
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
    fn test_letter_value_basic_depth1_is_median() {
        let vs: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let batch = batch1col("x", vs);
        let spec = LetterValueSpec {
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
            value: "x".into(), group: None,
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: None,
            sort: None,
        };
        let primary = apply(&spec, &batch).unwrap();
        let secs = secondary_outputs(&spec, &batch, &primary).unwrap();
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
    fn test_letter_value_named_outputs_count_eight() {
        let vs: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let batch = batch1col("x", vs);
        let spec = LetterValueSpec {
            value: "x".into(), group: None,
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: None,
            sort: None,
        };
        let primary = apply(&spec, &batch).unwrap();
        let secs = secondary_outputs(&spec, &batch, &primary).unwrap();
        assert_eq!(secs.len(), 9);
        for k in 1..=K_MAX_NAMED {
            let key = format!("depth_{k}");
            assert!(secs.iter().any(|(name, _)| name == &key), "missing {key}");
        }
    }

    #[test]
    fn test_letter_value_named_outputs_use_name_prefix() {
        let vs: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let batch = batch1col("x", vs);
        let spec = LetterValueSpec {
            value: "x".into(), group: None,
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: Some("lv".into()),
            sort: None,
        };
        let primary = apply(&spec, &batch).unwrap();
        let secs = secondary_outputs(&spec, &batch, &primary).unwrap();
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
            value: "v".into(), group: Some("g".into()),
            k_depth: KDepth::Full,
            outlier_threshold: 1.5,
            name: None,
            sort: Some(LetterValueSort { op: SortOp::Count, order: SortOrder::Desc }),
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(group_first_appearance(&out), vec!["a", "c", "b"]);
    }
}
