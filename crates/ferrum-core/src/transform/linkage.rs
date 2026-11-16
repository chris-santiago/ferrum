//! Hierarchical clustering (Linkage) transform — kodama-backed.
//!
//! Produces four named outputs:
//! - `linkage` (also the primary): merge steps `[node_id, left, right, distance, n_obs]`.
//! - `order`: leaf display order `[original_idx, new_idx]`.
//! - `coords`: node coordinates `[node_id, x, y]`.
//! - `segments`: dendrogram arm segments `[x, y, x2, y2]` (3 per merge).
//!
//! The primary `apply()` return is the `linkage` RecordBatch. The other three
//! are returned from `secondary_outputs()`.
//!
//! Naming convention: when `name = Some(s)`, primary → `s`, secondaries →
//! `s_linkage`, `s_order`, `s_coords`, `s_segments`. When `name = None`,
//! secondaries → bare `linkage`, `order`, `coords`, `segments`. Note: using
//! two unnamed Linkage transforms in one pipeline will cause key collision on
//! the secondary names; that case should use `name`.

use arrow::array::{Array, ArrayRef, Float64Array, Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::transform::context::TransformContext;
use crate::transform::group_key::{report_skipped_null_keys, NullKeyCounts, NullKeyPolicy};
use crate::transform::numeric_util::{coerce_to_float64, numeric_column_indices};
use crate::transform::vocabulary::Vocabulary;

// ---------- Spec types ----------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct LinkageSpec {
    pub method: LinkageMethod,
    pub metric: DistanceMetric,
    pub axis: LinkageAxis,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub z_score: Option<ZScoreAxis>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub standard_scale: Option<StdScaleAxis>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
    /// The column that labels each row (a clustermap's row labels). With it
    /// set, a row whose label is null belongs to no row and is left out of
    /// the matrix on either axis (its values must not move a column order
    /// either), reported as every `Skip` statistic reports a null key. The
    /// label column itself is never a feature.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub labels: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LinkageMethod {
    Single,
    Complete,
    Average,
    Weighted,
    Centroid,
    Median,
    Ward,
}

/// `Linkage(method=)`'s vocabulary: [`PyLinkage::new`] parses through it and
/// `registry::entry_for_kind("linkage")` publishes it, so the door accepts
/// exactly the declared spellings. The spellings equal serde's snake_case
/// variant names, which the wire reads.
impl Vocabulary for LinkageMethod {
    const ALL: &'static [Self] = &[
        Self::Single,
        Self::Complete,
        Self::Average,
        Self::Weighted,
        Self::Centroid,
        Self::Median,
        Self::Ward,
    ];

    fn spellings(self) -> &'static [&'static str] {
        match self {
            Self::Single => &["single"],
            Self::Complete => &["complete"],
            Self::Average => &["average"],
            Self::Weighted => &["weighted"],
            Self::Centroid => &["centroid"],
            Self::Median => &["median"],
            Self::Ward => &["ward"],
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DistanceMetric {
    Euclidean,
    Manhattan,
    Cosine,
    Correlation,
    Chebyshev,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LinkageAxis {
    Rows,
    Columns,
}

/// The `axis=` vocabulary of `Linkage` and of the `Reorder` that applies a
/// linkage's leaf order: both name the same two axes of the wide matrix.
impl Vocabulary for LinkageAxis {
    const ALL: &'static [Self] = &[Self::Rows, Self::Columns];

    fn spellings(self) -> &'static [&'static str] {
        match self {
            Self::Rows => &["rows"],
            Self::Columns => &["columns"],
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ZScoreAxis {
    Rows,
    Columns,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum StdScaleAxis {
    Rows,
    Columns,
}

// ---------- Embed reference fixtures at compile time ----------

#[cfg(test)]
const LINKAGE_REFS: &str = include_str!("fixtures/linkage_refs.json");

// ---------- Core math helpers ----------

/// Extract a 2D matrix (rows x cols) of f64 from every numeric column in a
/// RecordBatch.
///
/// Column selection is deliberate, not a fallback: `Linkage` runs over the
/// chart's full untransformed batch — `clustermap` chains it directly on
/// `Chart(data)`, so the batch also carries the row/column label column
/// (`id_col`, typically Utf8) alongside the numeric columns. Excluding
/// non-numeric columns from the distance computation is therefore by design;
/// what was a bug is which columns counted as "numeric". Every column
/// [`numeric_column_indices`] selects (any int/uint/float dtype, per the
/// batch-D M1 policy) is coerced to Float64 and included — previously only
/// columns that were *already* exactly Float64 were selected, so an Int64
/// column silently dropped out of the distance matrix instead of widening,
/// making the clustering result depend on the columns' dtypes with no
/// warning.
///
/// Returns (n_rows, n_features, flat row-major Vec<f64>).
///
/// `rows` restricts the matrix to those rows of `batch` (the rows whose label
/// is not null), and `label` names a column that is never a feature.
fn extract_matrix(
    batch: &RecordBatch,
    rows: Option<&[usize]>,
    label: Option<&str>,
) -> PyResult<(usize, usize, Vec<f64>)> {
    let schema = batch.schema();
    let feature_cols: Vec<Float64Array> = feature_column_indices(&schema, label, "linkage")?
        .into_iter()
        .map(|i| coerce_to_float64(batch.column(i), "linkage", schema.field(i).name()))
        .collect::<PyResult<_>>()?;
    if feature_cols.is_empty() {
        return Err(PyValueError::new_err(
            "linkage: input must have at least one numeric column",
        ));
    }
    let kept: Vec<usize> = rows.map(<[usize]>::to_vec).unwrap_or_else(|| (0..batch.num_rows()).collect());
    let n_rows = kept.len();
    let n_cols = feature_cols.len();
    let mut mat = vec![0.0f64; n_rows * n_cols];
    for (feat_idx, arr) in feature_cols.iter().enumerate() {
        for (out_row, &row) in kept.iter().enumerate() {
            mat[out_row * n_cols + feat_idx] = arr.value(row);
        }
    }
    Ok((n_rows, n_cols, mat))
}

/// Transpose a row-major matrix from (n_rows x n_cols) to (n_cols x n_rows).
fn transpose(mat: &[f64], n_rows: usize, n_cols: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n_rows * n_cols];
    for r in 0..n_rows {
        for c in 0..n_cols {
            out[c * n_rows + r] = mat[r * n_cols + c];
        }
    }
    out
}

/// Apply z-score normalization along rows of a (n x feat) matrix.
/// Each row gets: (x - mean(row)) / std(row, ddof=1). Rows with std=0 remain zero.
fn z_score_rows(mat: &mut [f64], n: usize, feat: usize) {
    for i in 0..n {
        let row = &mut mat[i * feat..(i + 1) * feat];
        let mean = row.iter().sum::<f64>() / feat as f64;
        let variance = row.iter().map(|&v| (v - mean).powi(2)).sum::<f64>()
            / (feat as f64 - 1.0).max(1.0);
        let std = variance.sqrt();
        if std > 0.0 {
            for v in row.iter_mut() {
                *v = (*v - mean) / std;
            }
        }
    }
}

/// Apply standard-scale normalization along rows: (x - min) / (max - min).
/// Rows with max=min remain zero.
fn std_scale_rows(mat: &mut [f64], n: usize, feat: usize) {
    for i in 0..n {
        let row = &mut mat[i * feat..(i + 1) * feat];
        let min = row.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;
        if range > 0.0 {
            for v in row.iter_mut() {
                *v = (*v - min) / range;
            }
        }
    }
}

/// Compute condensed pairwise distance matrix for n observations × feat features.
/// Returns Vec of length n*(n-1)/2 in upper-triangle row-major order.
pub(crate) fn condensed_distances(
    mat: &[f64],
    n: usize,
    feat: usize,
    metric: DistanceMetric,
) -> PyResult<Vec<f64>> {
    let expected_len = n * (n - 1) / 2;
    let mut condensed = Vec::with_capacity(expected_len);

    let row = |i: usize| &mat[i * feat..(i + 1) * feat];

    for i in 0..n - 1 {
        for j in i + 1..n {
            let xi = row(i);
            let xj = row(j);
            let d = match metric {
                DistanceMetric::Euclidean => {
                    xi.iter()
                        .zip(xj.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt()
                }
                DistanceMetric::Manhattan => xi
                    .iter()
                    .zip(xj.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f64>(),
                DistanceMetric::Chebyshev => xi
                    .iter()
                    .zip(xj.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f64, f64::max),
                DistanceMetric::Cosine => {
                    let dot: f64 = xi.iter().zip(xj.iter()).map(|(a, b)| a * b).sum();
                    let norm_i: f64 =
                        xi.iter().map(|a| a * a).sum::<f64>().sqrt();
                    let norm_j: f64 =
                        xj.iter().map(|b| b * b).sum::<f64>().sqrt();
                    if norm_i == 0.0 || norm_j == 0.0 {
                        // Zero-norm vector: cosine distance is undefined; use 0.
                        0.0
                    } else {
                        // Clamp to avoid NaN from floating-point rounding outside [-1,1].
                        (1.0 - dot / (norm_i * norm_j)).max(0.0)
                    }
                }
                DistanceMetric::Correlation => {
                    // Pearson correlation: 1 - corrcoef(xi, xj).
                    let n_f = feat as f64;
                    let mean_i: f64 = xi.iter().sum::<f64>() / n_f;
                    let mean_j: f64 = xj.iter().sum::<f64>() / n_f;
                    let mut num = 0.0f64;
                    let mut denom_i = 0.0f64;
                    let mut denom_j = 0.0f64;
                    for (&a, &b) in xi.iter().zip(xj.iter()) {
                        let da = a - mean_i;
                        let db = b - mean_j;
                        num += da * db;
                        denom_i += da * da;
                        denom_j += db * db;
                    }
                    let denom = (denom_i * denom_j).sqrt();
                    if denom == 0.0 {
                        0.0
                    } else {
                        // Clamp to avoid NaN.
                        (1.0 - num / denom).max(0.0)
                    }
                }
            };
            condensed.push(d);
        }
    }
    Ok(condensed)
}

/// Map LinkageMethod → kodama::Method.
fn to_kodama_method(m: LinkageMethod) -> kodama::Method {
    match m {
        LinkageMethod::Single => kodama::Method::Single,
        LinkageMethod::Complete => kodama::Method::Complete,
        LinkageMethod::Average => kodama::Method::Average,
        LinkageMethod::Weighted => kodama::Method::Weighted,
        LinkageMethod::Centroid => kodama::Method::Centroid,
        LinkageMethod::Median => kodama::Method::Median,
        LinkageMethod::Ward => kodama::Method::Ward,
    }
}

/// Represents a merge step for DFS traversal.
struct MergeStep {
    cluster1: usize,
    cluster2: usize,
    dissimilarity: f64,
}

/// Build leaf display order by DFS from the top of the dendrogram.
/// The result is a Vec<usize> where result[position] = original_observation_index.
fn dfs_leaf_order(steps: &[MergeStep], n: usize) -> Vec<usize> {
    let root = n + steps.len() - 1; // label of the root cluster (last step)
    let mut order = Vec::with_capacity(n);
    dfs_recurse(steps, n, root, &mut order);
    order
}

fn dfs_recurse(
    steps: &[MergeStep],
    n: usize,
    node: usize,
    order: &mut Vec<usize>,
) {
    if node < n {
        // Leaf
        order.push(node);
    } else {
        let step = &steps[node - n];
        dfs_recurse(steps, n, step.cluster1, order);
        dfs_recurse(steps, n, step.cluster2, order);
    }
}

// ---------- Apply ----------

pub(crate) fn apply(spec: &LinkageSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    apply_with_context(spec, batch, &TransformContext::default())
}

/// The rows of `batch` whose `labels` value is not null, in input order. The
/// label is read only for its nulls, so a column of any dtype serves; a
/// missing one is refused in the keyword's own name.
fn labelled_rows(batch: &RecordBatch, labels: &str) -> PyResult<Vec<usize>> {
    let column = batch
        .column_by_name(labels)
        .ok_or_else(|| PyValueError::new_err(format!("linkage: labels column '{labels}' not found")))?;
    let nulls = column.logical_nulls();
    Ok((0..batch.num_rows()).filter(|&row| nulls.as_ref().is_none_or(|n| n.is_valid(row))).collect())
}

/// The numeric columns that are features: every column
/// [`numeric_column_indices`] selects except `label`, the column naming each
/// row, which is never clustered. `Linkage` and the column `Reorder` that
/// reads its order share this one rule.
pub(crate) fn feature_column_indices(
    schema: &Schema,
    label: Option<&str>,
    context: &str,
) -> PyResult<Vec<usize>> {
    Ok(numeric_column_indices(schema, context)?
        .into_iter()
        .filter(|&i| Some(schema.field(i).name().as_str()) != label)
        .collect())
}

/// What a `Linkage` with `labels` stamps on the schema of its `order` table so
/// a `Reorder` reading it knows the label column and the rows the linkage read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LinkageOrderMeta {
    pub(crate) label: String,
    pub(crate) rows: usize,
}

const ORDER_LABEL_KEY: &str = "__linkage_labels__";
const ORDER_ROWS_KEY: &str = "__linkage_rows__";

impl LinkageOrderMeta {
    /// The stamp on an `order` batch, or `None` when it has none (a
    /// `Linkage` without `labels`, or a batch no `Linkage` wrote).
    pub(crate) fn read(batch: &RecordBatch) -> Option<Self> {
        let schema = batch.schema();
        let meta = schema.metadata();
        Some(Self { label: meta.get(ORDER_LABEL_KEY)?.clone(), rows: meta.get(ORDER_ROWS_KEY)?.parse().ok()? })
    }

    fn schema(&self) -> Arc<Schema> {
        let metadata = std::collections::HashMap::from([
            (ORDER_LABEL_KEY.to_string(), self.label.clone()),
            (ORDER_ROWS_KEY.to_string(), self.rows.to_string()),
        ]);
        Arc::new(Schema::new_with_metadata(schema_from(ORDER_COLUMNS).fields().clone(), metadata))
    }
}

/// The empty batch of `columns`' schema.
fn empty_batch(columns: &[(&str, DataType)]) -> RecordBatch {
    RecordBatch::new_empty(schema_from(columns))
}

/// [`apply`] with a context: with `labels` set, the rows whose label is null
/// are left out of the clustering and reported through the context's sink; a
/// frame whose every label is null warns and returns no rows.
pub(crate) fn apply_with_context(
    spec: &LinkageSpec,
    batch: &RecordBatch,
    ctx: &TransformContext,
) -> PyResult<RecordBatch> {
    // Validate preprocessing flags.
    if spec.z_score.is_some() && spec.standard_scale.is_some() {
        return Err(PyValueError::new_err(
            "linkage: z_score and standard_scale are mutually exclusive",
        ));
    }

    let kept = match &spec.labels {
        Some(labels) => {
            let kept = labelled_rows(batch, labels)?;
            report_skipped_null_keys(
                batch,
                std::slice::from_ref(labels),
                "linkage",
                NullKeyPolicy::Skip,
                NullKeyCounts { any_null: batch.num_rows() - kept.len(), fully_null: 0, n_rows: batch.num_rows() },
                ctx,
            );
            Some(kept)
        }
        None => None,
    };
    if kept.as_ref().is_some_and(|rows| rows.is_empty()) && batch.num_rows() > 0 {
        return Ok(empty_batch(LINKAGE_COLUMNS));
    }

    // Extract matrix.
    let (n_rows, n_feat, mut mat) = extract_matrix(batch, kept.as_deref(), spec.labels.as_deref())?;

    // Axis transposition: if Columns, treat each column as an observation.
    let (n_obs, n_feat_for_dist) = match spec.axis {
        LinkageAxis::Rows => (n_rows, n_feat),
        LinkageAxis::Columns => {
            mat = transpose(&mat, n_rows, n_feat);
            // After transpose: n_feat rows × n_rows cols.
            (n_feat, n_rows)
        }
    };

    if n_obs < 2 {
        return Err(PyValueError::new_err(
            "linkage: need at least 2 observations to cluster",
        ));
    }

    // Pre-processing.
    if let Some(zscore_ax) = spec.z_score {
        match zscore_ax {
            ZScoreAxis::Rows => z_score_rows(&mut mat, n_obs, n_feat_for_dist),
            ZScoreAxis::Columns => {
                // z-score each column: transpose, z-score rows, transpose back.
                let mut t = transpose(&mat, n_obs, n_feat_for_dist);
                z_score_rows(&mut t, n_feat_for_dist, n_obs);
                mat = transpose(&t, n_feat_for_dist, n_obs);
            }
        }
    }
    if let Some(scale_ax) = spec.standard_scale {
        match scale_ax {
            StdScaleAxis::Rows => std_scale_rows(&mut mat, n_obs, n_feat_for_dist),
            StdScaleAxis::Columns => {
                let mut t = transpose(&mat, n_obs, n_feat_for_dist);
                std_scale_rows(&mut t, n_feat_for_dist, n_obs);
                mat = transpose(&t, n_feat_for_dist, n_obs);
            }
        }
    }

    // Compute condensed distances.
    let mut condensed = condensed_distances(&mat, n_obs, n_feat_for_dist, spec.metric)?;

    // Run kodama linkage. Note: kodama handles squaring internally for Ward/Centroid/Median.
    // We always pass raw (unsquared) euclidean distances regardless of method.
    let dend = kodama::linkage(&mut condensed, n_obs, to_kodama_method(spec.method));
    let steps = dend.steps();

    // Build the linkage RecordBatch (primary output).
    // Schema: [node_id: Int64, left: Int64, right: Int64, distance: Float64, n_obs: Int64]
    let n = n_obs;
    let mut node_ids = Vec::with_capacity(n - 1);
    let mut lefts = Vec::with_capacity(n - 1);
    let mut rights = Vec::with_capacity(n - 1);
    let mut distances = Vec::with_capacity(n - 1);
    let mut n_obss = Vec::with_capacity(n - 1);

    for (i, step) in steps.iter().enumerate() {
        node_ids.push((n + i) as i64);
        lefts.push(step.cluster1 as i64);
        rights.push(step.cluster2 as i64);
        distances.push(step.dissimilarity);
        n_obss.push(step.size as i64);
    }

    let linkage_schema = schema_from(LINKAGE_COLUMNS);
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(node_ids)),
        Arc::new(Int64Array::from(lefts)),
        Arc::new(Int64Array::from(rights)),
        Arc::new(Float64Array::from(distances)),
        Arc::new(Int64Array::from(n_obss)),
    ];
    RecordBatch::try_new(linkage_schema, cols)
        .map_err(|e| PyValueError::new_err(format!("linkage: {e}")))
}

/// Keys the four secondary batches are published under, under default
/// construction; a spec `name` prefixes each as `"{name}_{key}"`. The first
/// re-publishes the primary linkage batch itself so a named transform has a
/// stable key alongside the three derived outputs.
pub(crate) const LINKAGE_KEY: &str = "linkage";
pub(crate) const ORDER_KEY: &str = "order";
pub(crate) const COORDS_KEY: &str = "coords";
pub(crate) const SEGMENTS_KEY: &str = "segments";

/// Column schemas for this transform's batches — the single source of truth,
/// read by [`apply`]'s and [`secondary_outputs`]'s schema builders and by
/// `registry::entry_for_kind("linkage")` (R1 design finding (d): the
/// dendrogram column names were unchecked Python literals before).
///
/// [`LINKAGE_COLUMNS`] is the PRIMARY batch's schema, which the secondary
/// output under [`LINKAGE_KEY`] re-publishes unchanged.
pub(crate) const LINKAGE_COLUMNS: &[(&str, DataType)] = &[
    ("node_id", DataType::Int64),
    ("left", DataType::Int64),
    ("right", DataType::Int64),
    ("distance", DataType::Float64),
    ("n_obs", DataType::Int64),
];
pub(crate) const ORDER_COLUMNS: &[(&str, DataType)] =
    &[("original_idx", DataType::Int64), ("new_idx", DataType::Int64)];
pub(crate) const COORDS_COLUMNS: &[(&str, DataType)] = &[
    ("node_id", DataType::Int64),
    ("x", DataType::Float64),
    ("y", DataType::Float64),
];
pub(crate) const SEGMENTS_COLUMNS: &[(&str, DataType)] = &[
    ("x", DataType::Float64),
    ("y", DataType::Float64),
    ("x2", DataType::Float64),
    ("y2", DataType::Float64),
];

/// Build a non-nullable schema from one of the column tables above.
fn schema_from(columns: &[(&str, DataType)]) -> Arc<Schema> {
    Arc::new(Schema::new(
        columns.iter().map(|(name, dt)| Field::new(*name, dt.clone(), false)).collect::<Vec<_>>(),
    ))
}

/// Compute the three secondary named outputs (order, coords, segments).
///
/// `linkage_batch` is the RecordBatch returned by `apply()` — it has the
/// schema [node_id: Int64, left: Int64, right: Int64, distance: Float64, n_obs: Int64].
/// We reconstruct the dendrogram tree directly from those columns rather than
/// re-running the full distance computation.
///
/// `input` is the batch `apply` clustered: with `labels` set on the row axis,
/// the leaves are the rows whose label is not null, so `order`'s
/// `original_idx` is mapped back to the rows of `input` a reader can take.
/// A frame with no labelled row publishes every table empty.
pub(crate) fn secondary_outputs(
    spec: &LinkageSpec,
    input: &RecordBatch,
    linkage_batch: &RecordBatch,
) -> PyResult<Vec<(String, RecordBatch)>> {
    let meta = spec.labels.as_ref().map(|label| LinkageOrderMeta { label: label.clone(), rows: input.num_rows() });
    let order_schema = meta.as_ref().map_or_else(|| schema_from(ORDER_COLUMNS), LinkageOrderMeta::schema);
    if linkage_batch.num_rows() == 0 {
        return Ok(secondary_keys(spec, [
            linkage_batch.clone(),
            RecordBatch::new_empty(order_schema),
            empty_batch(COORDS_COLUMNS),
            empty_batch(SEGMENTS_COLUMNS),
        ]));
    }
    // The rows the row axis clustered, to map leaf indices back.
    let kept: Option<Vec<usize>> = match (&spec.labels, spec.axis) {
        (Some(labels), LinkageAxis::Rows) => Some(labelled_rows(input, labels)?),
        _ => None,
    };
    // Extract steps from the linkage batch.
    let n_steps = linkage_batch.num_rows();
    // n = n_obs, and n_steps = n - 1, so n = n_steps + 1.
    let n = n_steps + 1;

    let left_arr = linkage_batch
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| PyValueError::new_err("linkage secondary: expected Int64 'left' column"))?;
    let right_arr = linkage_batch
        .column(2)
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| PyValueError::new_err("linkage secondary: expected Int64 'right' column"))?;
    let dist_arr = linkage_batch
        .column(3)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| {
            PyValueError::new_err("linkage secondary: expected Float64 'distance' column")
        })?;

    // Reconstruct steps as MergeStep for DFS traversal.
    let steps: Vec<MergeStep> = (0..n_steps)
        .map(|i| MergeStep {
            cluster1: left_arr.value(i) as usize,
            cluster2: right_arr.value(i) as usize,
            dissimilarity: dist_arr.value(i),
        })
        .collect();

    // ---- ORDER output ----
    // Leaf display order via DFS from the dendrogram root.
    let leaf_order = dfs_leaf_order(&steps, n);
    // leaf_order[pos] = original_observation_index.
    // We want: original_idx = leaf_order[pos], new_idx = pos.
    let mut orig_idxs = Vec::with_capacity(n);
    let mut new_idxs = Vec::with_capacity(n);
    for (pos, &orig) in leaf_order.iter().enumerate() {
        let original = kept.as_ref().map_or(orig, |rows| rows[orig]);
        orig_idxs.push(original as i64);
        new_idxs.push(pos as i64);
    }
    let order_batch = RecordBatch::try_new(
        order_schema,
        vec![
            Arc::new(Int64Array::from(orig_idxs)) as ArrayRef,
            Arc::new(Int64Array::from(new_idxs)) as ArrayRef,
        ],
    )
    .map_err(|e| PyValueError::new_err(format!("linkage order: {e}")))?;

    // ---- COORDS output ----
    // position_of_leaf[original_idx] = display position (x coordinate).
    // leaf_order[pos] = orig → position_of_leaf[orig] = pos.
    let mut position_of_leaf = vec![0.0f64; n];
    for (pos, &orig) in leaf_order.iter().enumerate() {
        position_of_leaf[orig] = pos as f64;
    }

    // coords[node_id] = (x, y). We need 2n-1 entries.
    // Leaves (0..n): x = position, y = 0.
    // Internal (n..2n-1): x = mean of children x, y = distance.
    let mut node_x = vec![0.0f64; 2 * n - 1];
    let mut node_y = vec![0.0f64; 2 * n - 1];
    for orig in 0..n {
        node_x[orig] = position_of_leaf[orig];
        node_y[orig] = 0.0;
    }
    for (i, step) in steps.iter().enumerate() {
        let node = n + i;
        let x_left = node_x[step.cluster1];
        let x_right = node_x[step.cluster2];
        node_x[node] = (x_left + x_right) / 2.0;
        node_y[node] = step.dissimilarity;
    }

    let mut coord_node_ids = Vec::with_capacity(2 * n - 1);
    let mut coord_xs = Vec::with_capacity(2 * n - 1);
    let mut coord_ys = Vec::with_capacity(2 * n - 1);
    for node in 0..2 * n - 1 {
        coord_node_ids.push(node as i64);
        coord_xs.push(node_x[node]);
        coord_ys.push(node_y[node]);
    }
    let coords_schema = schema_from(COORDS_COLUMNS);
    let coords_batch = RecordBatch::try_new(
        coords_schema,
        vec![
            Arc::new(Int64Array::from(coord_node_ids)) as ArrayRef,
            Arc::new(Float64Array::from(coord_xs)) as ArrayRef,
            Arc::new(Float64Array::from(coord_ys)) as ArrayRef,
        ],
    )
    .map_err(|e| PyValueError::new_err(format!("linkage coords: {e}")))?;

    // ---- SEGMENTS output ----
    // 3 segments per merge: left riser, top cap, right riser.
    let n_segs = 3 * (n - 1);
    let mut seg_x = Vec::with_capacity(n_segs);
    let mut seg_y = Vec::with_capacity(n_segs);
    let mut seg_x2 = Vec::with_capacity(n_segs);
    let mut seg_y2 = Vec::with_capacity(n_segs);

    for step in steps.iter() {
        let d_node = step.dissimilarity;
        let x_left = node_x[step.cluster1];
        let y_left = node_y[step.cluster1];
        let x_right = node_x[step.cluster2];
        let y_right = node_y[step.cluster2];
        // Ensure left is to the left of right visually.
        let (x_l, y_l, x_r, y_r) = if x_left <= x_right {
            (x_left, y_left, x_right, y_right)
        } else {
            (x_right, y_right, x_left, y_left)
        };

        // 1. Left vertical riser: (x_l, y_l) → (x_l, d_node)
        seg_x.push(x_l);
        seg_y.push(y_l);
        seg_x2.push(x_l);
        seg_y2.push(d_node);

        // 2. Top horizontal cap: (x_l, d_node) → (x_r, d_node)
        seg_x.push(x_l);
        seg_y.push(d_node);
        seg_x2.push(x_r);
        seg_y2.push(d_node);

        // 3. Right vertical riser: (x_r, d_node) → (x_r, y_r)
        seg_x.push(x_r);
        seg_y.push(d_node);
        seg_x2.push(x_r);
        seg_y2.push(y_r);
    }

    let segments_schema = schema_from(SEGMENTS_COLUMNS);
    let segments_batch = RecordBatch::try_new(
        segments_schema,
        vec![
            Arc::new(Float64Array::from(seg_x)) as ArrayRef,
            Arc::new(Float64Array::from(seg_y)) as ArrayRef,
            Arc::new(Float64Array::from(seg_x2)) as ArrayRef,
            Arc::new(Float64Array::from(seg_y2)) as ArrayRef,
        ],
    )
    .map_err(|e| PyValueError::new_err(format!("linkage segments: {e}")))?;

    // We also publish the linkage batch itself under the explicit key so that named
    // transforms have a stable key alongside the three derived outputs.
    Ok(secondary_keys(spec, [linkage_batch.clone(), order_batch, coords_batch, segments_batch]))
}

/// The four secondary batches under the naming convention's keys.
fn secondary_keys(spec: &LinkageSpec, batches: [RecordBatch; 4]) -> Vec<(String, RecordBatch)> {
    let name = spec.name.as_deref();
    [LINKAGE_KEY, ORDER_KEY, COORDS_KEY, SEGMENTS_KEY]
        .into_iter()
        .zip(batches)
        .map(|(key, batch)| (crate::transform::registry::secondary_key(name, key), batch))
        .collect()
}

// ---------- PyO3 wrapper ----------

use pyo3::prelude::*;

use crate::transform::core::TransformSpec;

/// Hierarchical clustering linkage for clustermap row/column ordering.
///
/// Computes a linkage tree from a numeric matrix batch and emits a
/// leaf-order permutation index used by ``Reorder`` to sort rows or
/// columns. Optionally z-score normalises or min-max scales before
/// computing distances.
///
/// Primary output columns: ``node_id`` (Int64), ``left`` (Int64),
/// ``right`` (Int64), ``distance`` (Float64), ``n_obs`` (Int64) — one row
/// per merge step. Secondary named outputs (accessible via ``name``) are
/// ``<name>_order`` (leaf permutation index), ``<name>_coords`` (node x/y),
/// and ``<name>_segments`` (dendrogram arm segments).
///
/// Parameters
/// ----------
/// method : {"single", "complete", "average", "weighted", "centroid", \
///           "median", "ward"}, default "ward"
///     Linkage algorithm.
/// metric : {"euclidean", "manhattan", "cosine", "correlation", \
///           "chebyshev"}, default "euclidean"
///     Distance metric between observations.
/// axis : {"rows", "columns"}, default "rows"
///     Whether to cluster rows or columns of the input matrix.
/// z_score : {"rows", "columns"}, optional
///     Standardise (subtract mean, divide by std) along the given axis
///     before computing distances. Mutually exclusive with
///     ``standard_scale``.
/// standard_scale : {"rows", "columns"}, optional
///     Min-max scale to [0, 1] along the given axis before computing
///     distances. Mutually exclusive with ``z_score``.
/// name : str, optional
///     Named output label for sibling ``Reorder(from_=...)`` lookup.
/// labels : str, optional
///     The column that labels each row. A row whose label is null belongs
///     to no row: it is left out of the clustering on either axis, counted
///     and warned like every grouped statistic's null key, and ``order``'s
///     ``original_idx`` indexes the original rows. The label column is never
///     a feature. A column of any dtype; only its nulls are read. The
///     ``order`` table records the label column and the rows read, so a
///     ``Reorder`` reading it leaves the label column out of the columns it
///     permutes too. Without ``labels`` every row clusters.
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> import polars as pl
/// >>> df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.1, 2.2, 2.9, 4.1], "c": [4.0, 3.0, 2.0, 1.0]})
/// >>> chart = fm.Chart(df).transform(fm.Linkage()).mark_point().encode(x="node_id", y="distance")
/// >>> svg = chart.to_svg()
#[pyclass(eq, module = "ferrum._core", name = "Linkage")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyLinkage(pub(crate) TransformSpec);

#[pymethods]
impl PyLinkage {
    #[new]
    #[pyo3(signature = (
        *,
        method = "ward",
        metric = "euclidean",
        axis = "rows",
        z_score = None,
        standard_scale = None,
        name = None,
        labels = None,
    ))]
    fn new(
        method: &str,
        metric: &str,
        axis: &str,
        z_score: Option<&str>,
        standard_scale: Option<&str>,
        name: Option<String>,
        labels: Option<String>,
    ) -> PyResult<Self> {
        let method = LinkageMethod::parse_or_refuse("method", method)
            .map_err(|e| PyValueError::new_err(format!("Linkage: {e}")))?;
        let metric = match metric {
            "euclidean" => DistanceMetric::Euclidean,
            "manhattan" => DistanceMetric::Manhattan,
            "cosine" => DistanceMetric::Cosine,
            "correlation" => DistanceMetric::Correlation,
            "chebyshev" => DistanceMetric::Chebyshev,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Linkage: unknown metric '{metric}'; expected euclidean|manhattan|cosine|correlation|chebyshev"
                )))
            }
        };
        let axis = LinkageAxis::parse_or_refuse("axis", axis)
            .map_err(|e| PyValueError::new_err(format!("Linkage: {e}")))?;
        let z_score = match z_score {
            None => None,
            Some("rows") => Some(ZScoreAxis::Rows),
            Some("columns") => Some(ZScoreAxis::Columns),
            Some(other) => {
                return Err(PyValueError::new_err(format!(
                    "Linkage: unknown z_score axis '{other}'"
                )))
            }
        };
        let standard_scale = match standard_scale {
            None => None,
            Some("rows") => Some(StdScaleAxis::Rows),
            Some("columns") => Some(StdScaleAxis::Columns),
            Some(other) => {
                return Err(PyValueError::new_err(format!(
                    "Linkage: unknown standard_scale axis '{other}'"
                )))
            }
        };
        Ok(PyLinkage(TransformSpec::Linkage(LinkageSpec {
            method,
            metric,
            axis,
            z_score,
            standard_scale,
            name,
            labels,
        })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Linkage(s) => format!(
                "Linkage(method={:?}, metric={:?}, axis={:?}, z_score={:?}, standard_scale={:?}, name={:?}, labels={:?})",
                s.method, s.metric, s.axis, s.z_score, s.standard_scale, s.name, s.labels,
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

// ---------- Tests ----------

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    /// Build a RecordBatch from a flat row-major matrix (n_rows × n_cols).
    fn make_matrix_batch(data: &[f64], n_rows: usize, n_cols: usize) -> RecordBatch {
        let schema_fields: Vec<Field> = (0..n_cols)
            .map(|c| Field::new(format!("f{c}"), DataType::Float64, false))
            .collect();
        let schema = Arc::new(Schema::new(schema_fields));
        let arrays: Vec<ArrayRef> = (0..n_cols)
            .map(|c| {
                let col_data: Vec<f64> =
                    (0..n_rows).map(|r| data[r * n_cols + c]).collect();
                Arc::new(Float64Array::from(col_data)) as ArrayRef
            })
            .collect();
        RecordBatch::try_new(schema, arrays).unwrap()
    }

    fn col_f64(batch: &RecordBatch, name: &str) -> Vec<f64> {
        let i = batch.schema().index_of(name).unwrap();
        let a = batch
            .column(i)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        (0..a.len()).map(|j| a.value(j)).collect()
    }

    fn col_i64(batch: &RecordBatch, name: &str) -> Vec<i64> {
        let i = batch.schema().index_of(name).unwrap();
        let a = batch
            .column(i)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        (0..a.len()).map(|j| a.value(j)).collect()
    }

    /// Default spec for testing.
    fn ward_spec() -> LinkageSpec {
        LinkageSpec {
            method: LinkageMethod::Ward,
            metric: DistanceMetric::Euclidean,
            axis: LinkageAxis::Rows,
            z_score: None,
            standard_scale: None, labels: None,
            name: None,
        }
    }

    /// A decimal column holds numbers no transform reads as one, so the
    /// numeric-column split refuses it, naming the column and its dtype,
    /// instead of leaving it out of the clustering; integer and float
    /// columns are numeric, a label column is not.
    #[test]
    fn a_decimal_column_is_refused_rather_than_left_out() {
        pyo3::Python::initialize();
        use arrow::array::{Decimal128Array, StringArray};
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("a", DataType::Float64, false),
            Field::new("b", DataType::Decimal128(5, 2), false),
            Field::new("c", DataType::Int64, false),
        ]));
        let decimals = Decimal128Array::from(vec![150, 50, 250]).with_precision_and_scale(5, 2).unwrap();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["r0", "r1", "r2"])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
                Arc::new(decimals),
                Arc::new(Int64Array::from(vec![3, 1, 2])),
            ],
        )
        .unwrap();
        assert_eq!(
            apply(&ward_spec(), &batch).unwrap_err().to_string(),
            "ValueError: linkage: column 'b' is a decimal column (Decimal128(5, 2)); a numeric column \
             must be an integer or float column, so cast it to float first"
        );
        let without_decimal = batch.project(&[0, 1, 3]).unwrap();
        assert_eq!(numeric_column_indices(&without_decimal.schema(), "linkage").unwrap(), vec![1, 2]);
        assert!(apply(&ward_spec(), &without_decimal).is_ok());
    }

    #[test]
    fn test_linkage_serde_round_trip() {
        use crate::transform::core::TransformSpec;
        let original = TransformSpec::Linkage(ward_spec());
        let json = serde_json::to_string(&original).unwrap();
        assert!(json.contains(r#""type":"linkage""#), "missing tag: {json}");
        assert!(json.contains(r#""method":"ward""#), "missing method: {json}");
        assert!(!json.contains("z_score"), "z_score=None should be omitted: {json}");
        assert!(!json.contains("name"), "name=None should be omitted: {json}");
        let parsed: TransformSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn test_linkage_schema_four_outputs() {
        // Verify primary has correct schema and secondary_outputs returns 4 keys.
        let data: Vec<f64> = vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 1.0,
            2.0, 6.0,
        ];
        let batch = make_matrix_batch(&data, 4, 2);
        let spec = ward_spec();

        let primary = apply(&spec, &batch).unwrap();
        assert_eq!(primary.num_rows(), 3); // n-1 = 3
        let schema = primary.schema();
        assert_eq!(schema.field(0).name(), "node_id");
        assert_eq!(schema.field(1).name(), "left");
        assert_eq!(schema.field(2).name(), "right");
        assert_eq!(schema.field(3).name(), "distance");
        assert_eq!(schema.field(4).name(), "n_obs");
        assert_eq!(schema.field(0).data_type(), &DataType::Int64);
        assert_eq!(schema.field(3).data_type(), &DataType::Float64);

        let secondaries = secondary_outputs(&spec, &batch, &primary).unwrap();
        assert_eq!(secondaries.len(), 4, "expected 4 secondary outputs");
        let keys: Vec<&str> = secondaries.iter().map(|(k, _)| k.as_str()).collect();
        assert!(keys.contains(&"linkage"));
        assert!(keys.contains(&"order"));
        assert!(keys.contains(&"coords"));
        assert!(keys.contains(&"segments"));
    }

    #[test]
    fn test_linkage_n2_edge_case() {
        // n=2: 1 merge, 3 segments.
        let data: Vec<f64> = vec![0.0, 0.0, 3.0, 4.0];
        let batch = make_matrix_batch(&data, 2, 2);
        let spec = ward_spec();

        let primary = apply(&spec, &batch).unwrap();
        assert_eq!(primary.num_rows(), 1);
        let distances = col_f64(&primary, "distance");
        // euclidean distance between (0,0) and (3,4) = 5.0
        assert!((distances[0] - 5.0).abs() < 1e-10, "distance: {}", distances[0]);

        let secondaries = secondary_outputs(&spec, &batch, &primary).unwrap();
        let (_, seg_batch) = secondaries.iter().find(|(k, _)| k == "segments").unwrap();
        assert_eq!(seg_batch.num_rows(), 3, "n=2 should produce 3 segments");
        let (_, order_batch) = secondaries.iter().find(|(k, _)| k == "orders").unwrap_or_else(|| {
            secondaries.iter().find(|(k, _)| k == "order").unwrap()
        });
        assert_eq!(order_batch.num_rows(), 2);
        let (_, coords_batch) = secondaries.iter().find(|(k, _)| k == "coords").unwrap();
        assert_eq!(coords_batch.num_rows(), 3); // 2n-1 = 3
    }

    #[test]
    fn test_linkage_n1_error() {
        pyo3::Python::initialize();
        let data: Vec<f64> = vec![1.0, 2.0];
        let batch = make_matrix_batch(&data, 1, 2);
        let spec = ward_spec();
        let err = apply(&spec, &batch).unwrap_err();
        assert!(err.to_string().contains("at least 2"));
    }

    #[test]
    fn test_linkage_naming_convention() {
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.0];
        let batch = make_matrix_batch(&data, 3, 2);
        let spec_unnamed = ward_spec();
        let primary = apply(&spec_unnamed, &batch).unwrap();
        let secondaries = secondary_outputs(&spec_unnamed, &batch, &primary).unwrap();
        let keys: Vec<&str> = secondaries.iter().map(|(k, _)| k.as_str()).collect();
        assert!(keys.contains(&"linkage"), "unnamed should use bare 'linkage'");
        assert!(keys.contains(&"order"));
        assert!(keys.contains(&"coords"));
        assert!(keys.contains(&"segments"));

        // Named convention.
        let spec_named = LinkageSpec { name: Some("clust".to_string()), ..ward_spec() };
        let primary2 = apply(&spec_named, &batch).unwrap();
        let secondaries2 = secondary_outputs(&spec_named, &batch, &primary2).unwrap();
        let keys2: Vec<&str> = secondaries2.iter().map(|(k, _)| k.as_str()).collect();
        assert!(keys2.contains(&"clust_linkage"), "named should prefix: {keys2:?}");
        assert!(keys2.contains(&"clust_order"));
        assert!(keys2.contains(&"clust_coords"));
        assert!(keys2.contains(&"clust_segments"));
    }

    #[test]
    fn test_linkage_mutually_exclusive_preprocessing() {
        pyo3::Python::initialize();
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.0];
        let batch = make_matrix_batch(&data, 3, 2);
        let spec = LinkageSpec {
            z_score: Some(ZScoreAxis::Rows),
            standard_scale: Some(StdScaleAxis::Rows), labels: None,
            ..ward_spec()
        };
        let err = apply(&spec, &batch).unwrap_err();
        assert!(err.to_string().contains("mutually exclusive"));
    }

    #[test]
    fn test_linkage_zscore_changes_result() {
        let data: Vec<f64> = vec![
            10.0, 20.0, 30.0,
            100.0, 200.0, 300.0,
            15.0, 25.0, 35.0,
        ];
        let batch = make_matrix_batch(&data, 3, 3);
        let spec_plain = ward_spec();
        let spec_zscore = LinkageSpec {
            z_score: Some(ZScoreAxis::Rows),
            ..ward_spec()
        };
        let primary_plain = apply(&spec_plain, &batch).unwrap();
        let primary_zscore = apply(&spec_zscore, &batch).unwrap();
        let d_plain = col_f64(&primary_plain, "distance");
        let d_zscore = col_f64(&primary_zscore, "distance");
        // z_score should produce different distances.
        assert_ne!(
            d_plain[0].to_bits(),
            d_zscore[0].to_bits(),
            "z_score should change distances"
        );
    }

    #[test]
    fn test_linkage_segments_geometry() {
        // For n=3 with a simple triangle: verify segment structure (3*(3-1)=6 segments).
        let data: Vec<f64> = vec![0.0, 0.0, 1.0, 0.0, 0.5, 1.0];
        let batch = make_matrix_batch(&data, 3, 2);
        let spec = LinkageSpec {
            method: LinkageMethod::Complete,
            metric: DistanceMetric::Euclidean,
            ..ward_spec()
        };
        let primary = apply(&spec, &batch).unwrap();
        let secondaries = secondary_outputs(&spec, &batch, &primary).unwrap();
        let (_, seg_batch) = secondaries.iter().find(|(k, _)| k == "segments").unwrap();
        // 3*(n-1) = 3*2 = 6 segments.
        assert_eq!(seg_batch.num_rows(), 6);
        // Each triple of rows must have consistent y values for the cap.
        let xs = col_f64(seg_batch, "x");
        let ys = col_f64(seg_batch, "y");
        let xs2 = col_f64(seg_batch, "x2");
        let ys2 = col_f64(seg_batch, "y2");
        // For each merge (2 merges total), verify the cap segment (row 1, 4):
        // x==x2 of left riser, y==y of right riser in cap, etc.
        for merge_i in 0..2 {
            let base = merge_i * 3;
            // Left riser: x==x2 (vertical).
            assert!(
                (xs[base] - xs2[base]).abs() < 1e-10,
                "left riser x mismatch at merge {merge_i}"
            );
            // Cap: y==y2 (horizontal).
            assert!(
                (ys[base + 1] - ys2[base + 1]).abs() < 1e-10,
                "cap y mismatch at merge {merge_i}"
            );
            // Right riser: x==x2 (vertical).
            assert!(
                (xs[base + 2] - xs2[base + 2]).abs() < 1e-10,
                "right riser x mismatch at merge {merge_i}"
            );
            // The cap y2 matches left riser y2.
            assert!(
                (ys2[base] - ys[base + 1]).abs() < 1e-10,
                "cap top should match left riser top at merge {merge_i}: {} vs {}",
                ys2[base], ys[base + 1]
            );
        }
    }

    #[test]
    fn test_linkage_all_fixture_cases() {
        // Load the scipy reference fixtures and verify:
        // - distance column matches scipy within 1e-8
        // - (left, right) pairs match scipy (order-insensitive)
        // - order matches scipy leaves_list
        let refs: serde_json::Value = serde_json::from_str(LINKAGE_REFS).unwrap();
        let data_raw = refs["data"].as_array().unwrap();
        let n_rows = data_raw.len();
        let n_cols = data_raw[0].as_array().unwrap().len();
        let mut flat_data: Vec<f64> = Vec::with_capacity(n_rows * n_cols);
        for row in data_raw {
            for val in row.as_array().unwrap() {
                flat_data.push(val.as_f64().unwrap());
            }
        }
        let batch = make_matrix_batch(&flat_data, n_rows, n_cols);

        let cases = refs["cases"].as_array().unwrap();
        for case in cases {
            let case_name = case["name"].as_str().unwrap();
            let method_str = case["method"].as_str().unwrap();
            let metric_str = case["metric"].as_str().unwrap();
            let scipy_linkage = case["linkage"].as_array().unwrap();
            let scipy_order = case["order"].as_array().unwrap();

            let method = match method_str {
                "ward" => LinkageMethod::Ward,
                "complete" => LinkageMethod::Complete,
                "average" => LinkageMethod::Average,
                "single" => LinkageMethod::Single,
                "weighted" => LinkageMethod::Weighted,
                "centroid" => LinkageMethod::Centroid,
                "median" => LinkageMethod::Median,
                _ => panic!("Unknown method: {method_str}"),
            };
            // scipy uses "cityblock" for Manhattan.
            let metric = match metric_str {
                "euclidean" => DistanceMetric::Euclidean,
                "cityblock" => DistanceMetric::Manhattan,
                "cosine" => DistanceMetric::Cosine,
                "correlation" => DistanceMetric::Correlation,
                "chebyshev" => DistanceMetric::Chebyshev,
                _ => panic!("Unknown metric: {metric_str}"),
            };

            let spec = LinkageSpec {
                method,
                metric,
                axis: LinkageAxis::Rows,
                z_score: None,
                standard_scale: None, labels: None,
                name: None,
            };

            let primary = apply(&spec, &batch).unwrap_or_else(|e| {
                panic!("apply failed for {case_name}: {e}");
            });

            assert_eq!(
                primary.num_rows(),
                n_rows - 1,
                "{case_name}: linkage should have n-1 rows"
            );

            let our_dist = col_f64(&primary, "distance");
            let our_left = col_i64(&primary, "left");
            let our_right = col_i64(&primary, "right");

            for (i, scipy_row) in scipy_linkage.iter().enumerate() {
                let cols = scipy_row.as_array().unwrap();
                let scipy_left = cols[0].as_f64().unwrap() as i64;
                let scipy_right = cols[1].as_f64().unwrap() as i64;
                let scipy_dist = cols[2].as_f64().unwrap();

                // Distance within 1e-8.
                assert!(
                    (our_dist[i] - scipy_dist).abs() < 1e-8,
                    "{case_name} step {i}: distance mismatch: ours={}, scipy={scipy_dist}",
                    our_dist[i]
                );

                // (left, right) match as unordered pair.
                let our_pair = {
                    let mut p = [our_left[i], our_right[i]];
                    p.sort();
                    p
                };
                let scipy_pair = {
                    let mut p = [scipy_left, scipy_right];
                    p.sort();
                    p
                };
                assert_eq!(
                    our_pair, scipy_pair,
                    "{case_name} step {i}: cluster pair mismatch"
                );
            }

            // Order output must match scipy leaves_list.
            let secondaries = secondary_outputs(&spec, &batch, &primary).unwrap_or_else(|e| {
                panic!("secondary_outputs failed for {case_name}: {e}");
            });
            let (_, order_batch) = secondaries.iter().find(|(k, _)| k == "order").unwrap();
            let our_orig_idx = col_i64(order_batch, "original_idx");
            let scipy_leaves: Vec<i64> = scipy_order
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect();
            assert_eq!(
                our_orig_idx, scipy_leaves,
                "{case_name}: order (leaf display order) mismatch: ours={our_orig_idx:?}, scipy={scipy_leaves:?}"
            );
        }
    }

    #[test]
    fn test_linkage_coords_row_counts() {
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 7.0, 8.0];
        let batch = make_matrix_batch(&data, 4, 2);
        let spec = ward_spec();
        let primary = apply(&spec, &batch).unwrap();
        let secondaries = secondary_outputs(&spec, &batch, &primary).unwrap();
        let (_, coords_batch) = secondaries.iter().find(|(k, _)| k == "coords").unwrap();
        // 2n-1 = 7 for n=4.
        assert_eq!(coords_batch.num_rows(), 7);
        let (_, order_batch) = secondaries.iter().find(|(k, _)| k == "order").unwrap();
        assert_eq!(order_batch.num_rows(), 4);
        let (_, seg_batch) = secondaries.iter().find(|(k, _)| k == "segments").unwrap();
        assert_eq!(seg_batch.num_rows(), 9); // 3*(4-1) = 9
    }

    // ---- Batch-D dtype policy (spec §4.1) ----

    /// Build a batch shaped like `clustermap`'s: a Utf8 `id` label column
    /// followed by the numeric feature columns `extract_matrix` must select.
    fn labeled_feature_batch(id: Vec<&str>, feature_arrays: Vec<ArrayRef>, n_feat: usize) -> RecordBatch {
        let mut fields = vec![Field::new("id", DataType::Utf8, false)];
        for i in 0..n_feat {
            fields.push(Field::new(format!("f{i}"), feature_arrays[i].data_type().clone(), false));
        }
        let schema = Arc::new(Schema::new(fields));
        let mut cols: Vec<ArrayRef> = vec![Arc::new(arrow::array::StringArray::from(id))];
        cols.extend(feature_arrays);
        RecordBatch::try_new(schema, cols).unwrap()
    }

    #[test]
    fn extract_matrix_widens_int64_feature_columns_and_excludes_the_label_column() {
        // Regression: extract_matrix used to select only columns that were
        // *already* exactly Float64, so an Int64 feature column silently
        // dropped out of the distance matrix instead of widening. It must now
        // cluster identically whether the feature columns are Int64 or
        // Float64, and the Utf8 label column must still be excluded (by
        // design, not as a fallback).
        let int_batch = labeled_feature_batch(
            vec!["r0", "r1", "r2", "r3"],
            vec![
                Arc::new(Int64Array::from(vec![1_i64, 3, 5, 2])) as ArrayRef,
                Arc::new(Int64Array::from(vec![2_i64, 4, 1, 6])) as ArrayRef,
            ],
            2,
        );
        let float_batch = labeled_feature_batch(
            vec!["r0", "r1", "r2", "r3"],
            vec![
                Arc::new(Float64Array::from(vec![1.0, 3.0, 5.0, 2.0])) as ArrayRef,
                Arc::new(Float64Array::from(vec![2.0, 4.0, 1.0, 6.0])) as ArrayRef,
            ],
            2,
        );
        let spec = ward_spec();
        let from_int = apply(&spec, &int_batch).unwrap();
        let from_float = apply(&spec, &float_batch).unwrap();
        assert_eq!(col_f64(&from_int, "distance"), col_f64(&from_float, "distance"));
        assert_eq!(col_i64(&from_int, "left"), col_i64(&from_float, "left"));
        assert_eq!(col_i64(&from_int, "right"), col_i64(&from_float, "right"));
    }

    #[test]
    fn extract_matrix_all_label_columns_raises_naming_numeric() {
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Utf8, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(arrow::array::StringArray::from(vec!["a", "b"]))],
        )
        .unwrap();
        let err = apply(&ward_spec(), &batch).unwrap_err();
        assert!(err.to_string().contains("at least one numeric column"));
    }

    // ---------- labels: null-labelled rows are left out ----------

    use arrow::array::StringArray;

    /// Rows 0 to 3 labelled `a`, null, `b`, `c`; row 1 is far from the others.
    fn labelled_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("label", DataType::Utf8, true),
            Field::new("f0", DataType::Float64, false),
            Field::new("f1", DataType::Float64, false),
            Field::new("f2", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![Some("a"), None, Some("b"), Some("c")])),
                Arc::new(Float64Array::from(vec![1.0, 100.0, 1.2, 5.0])),
                Arc::new(Float64Array::from(vec![2.0, -100.0, 2.1, 9.0])),
                Arc::new(Float64Array::from(vec![3.0, 50.0, 3.3, 1.0])),
            ],
        )
        .unwrap()
    }

    fn labelled_spec(axis: LinkageAxis) -> LinkageSpec {
        LinkageSpec { axis, labels: Some("label".into()), ..ward_spec() }
    }

    fn warnings_of(ctx: &TransformContext) -> Vec<String> {
        ctx.warnings.borrow().iter().map(|w| w.to_string()).collect()
    }

    #[test]
    fn null_labelled_rows_are_skipped_counted_and_the_order_indexes_the_original_rows() {
        pyo3::Python::initialize();
        let batch = labelled_batch();
        let spec = labelled_spec(LinkageAxis::Rows);
        let ctx = TransformContext::default();
        let primary = apply_with_context(&spec, &batch, &ctx).unwrap();
        assert_eq!(primary.num_rows(), 2, "three labelled rows make two merges");
        assert_eq!(
            warnings_of(&ctx),
            ["Linkage: 1 row has a null 'label' and belongs to no group, so it was dropped; fill or filter the nulls in 'label'"]
        );
        let secondaries = secondary_outputs(&spec, &batch, &primary).unwrap();
        let order = &secondaries.iter().find(|(k, _)| k == "order").unwrap().1;
        let mut original = col_i64(order, "original_idx");
        assert_eq!(original.len(), 3);
        original.sort_unstable();
        assert_eq!(original, vec![0, 2, 3], "original_idx names rows of the original batch, never the skipped one");
        assert_eq!(col_i64(order, "new_idx"), vec![0, 1, 2]);
        let coords = &secondaries.iter().find(|(k, _)| k == "coords").unwrap().1;
        assert_eq!(coords.num_rows(), 5, "coords and segments cover the three kept leaves");
    }

    #[test]
    fn the_kept_rows_cluster_exactly_as_the_frame_without_the_null_row_does() {
        pyo3::Python::initialize();
        let batch = labelled_batch();
        let with_labels = apply(&labelled_spec(LinkageAxis::Rows), &batch).unwrap();
        let tail = RecordBatch::try_new(
            batch.schema(),
            batch.columns().iter().map(|c| arrow::compute::concat(&[&c.slice(0, 1), &c.slice(2, 2)]).unwrap()).collect(),
        )
        .unwrap();
        let plain = apply(&ward_spec(), &tail.project(&[1, 2, 3]).unwrap()).unwrap();
        assert_eq!(col_f64(&with_labels, "distance"), col_f64(&plain, "distance"));
    }

    #[test]
    fn every_label_null_warns_once_and_returns_no_rows_anywhere() {
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![
            Field::new("label", DataType::Utf8, true),
            Field::new("f0", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec![None::<&str>, None, None])), Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0]))],
        )
        .unwrap();
        let spec = labelled_spec(LinkageAxis::Rows);
        let ctx = TransformContext::default();
        let primary = apply_with_context(&spec, &batch, &ctx).unwrap();
        assert_eq!(primary.num_rows(), 0);
        assert_eq!(
            warnings_of(&ctx),
            ["Linkage: every row has a null in its groupby key (label), so the data could not be split into groups; the grouped output is empty"]
        );
        for (key, table) in secondary_outputs(&spec, &batch, &primary).unwrap() {
            assert_eq!(table.num_rows(), 0, "{key}");
        }
    }

    #[test]
    fn every_label_null_on_the_column_axis_warns_and_publishes_empty_tables() {
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![
            Field::new("label", DataType::Utf8, true),
            Field::new("f0", DataType::Float64, false),
            Field::new("f1", DataType::Float64, false),
            Field::new("f2", DataType::Float64, false),
        ]));
        let col = |v: Vec<f64>| -> Arc<dyn arrow::array::Array> { Arc::new(Float64Array::from(v)) };
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![None::<&str>, None])),
                col(vec![1.0, 2.0]),
                col(vec![3.0, 4.0]),
                col(vec![5.0, 7.0]),
            ],
        )
        .unwrap();
        let spec = labelled_spec(LinkageAxis::Columns);
        let ctx = TransformContext::default();
        let primary = apply_with_context(&spec, &batch, &ctx).unwrap();
        assert_eq!(primary.num_rows(), 0);
        assert_eq!(warnings_of(&ctx).len(), 1, "the every-row sentence, once");
        for (key, table) in secondary_outputs(&spec, &batch, &primary).unwrap() {
            assert_eq!(table.num_rows(), 0, "{key}");
        }
    }

    #[test]
    fn without_labels_the_bytes_are_unchanged_and_a_label_column_with_no_null_changes_nothing() {
        pyo3::Python::initialize();
        let batch = labelled_batch();
        let plain = apply(&ward_spec(), &batch).unwrap();
        let features_only = apply(&ward_spec(), &batch.project(&[1, 2, 3]).unwrap()).unwrap();
        assert_eq!(plain, features_only, "a string column is never a feature");
        let no_nulls = RecordBatch::try_new(
            batch.schema(),
            vec![
                Arc::new(StringArray::from(vec!["a", "z", "b", "c"])),
                batch.column(1).clone(),
                batch.column(2).clone(),
                batch.column(3).clone(),
            ],
        )
        .unwrap();
        let ctx = TransformContext::default();
        let labelled = apply_with_context(&labelled_spec(LinkageAxis::Rows), &no_nulls, &ctx).unwrap();
        assert_eq!(labelled, plain);
        assert!(warnings_of(&ctx).is_empty());
    }

    #[test]
    fn on_the_column_axis_the_null_labelled_rows_do_not_move_the_column_order() {
        pyo3::Python::initialize();
        // Columns f0 and f1 agree on the labelled rows and f2 differs; the null
        // row makes f0 and f2 look alike, so it would change the order.
        let schema = Arc::new(Schema::new(vec![
            Field::new("label", DataType::Utf8, true),
            Field::new("f0", DataType::Float64, false),
            Field::new("f1", DataType::Float64, false),
            Field::new("f2", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![Some("a"), Some("b"), None])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 100.0])),
                Arc::new(Float64Array::from(vec![1.1, 2.1, -100.0])),
                Arc::new(Float64Array::from(vec![9.0, 0.0, 100.0])),
            ],
        )
        .unwrap();
        let spec = labelled_spec(LinkageAxis::Columns);
        let ctx = TransformContext::default();
        let primary = apply_with_context(&spec, &batch, &ctx).unwrap();
        let kept = batch.slice(0, 2);
        let plain_spec = LinkageSpec { axis: LinkageAxis::Columns, ..ward_spec() };
        let expected = apply(&plain_spec, &kept).unwrap();
        assert_eq!(primary, expected, "the null-labelled row's values never reach the column clustering");
        assert_eq!(
            warnings_of(&ctx),
            ["Linkage: 1 row has a null 'label' and belongs to no group, so it was dropped; fill or filter the nulls in 'label'"]
        );
        let order = secondary_outputs(&spec, &batch, &primary).unwrap().into_iter().find(|(k, _)| k == "order").unwrap().1;
        let mut columns = col_i64(&order, "original_idx");
        columns.sort_unstable();
        assert_eq!(columns, vec![0, 1, 2], "column indices are the original columns");
    }

    #[test]
    fn a_labels_column_the_batch_lacks_is_refused_naming_labels() {
        pyo3::Python::initialize();
        let spec = LinkageSpec { labels: Some("nope".into()), ..ward_spec() };
        let err = apply(&spec, &labelled_batch()).unwrap_err().to_string();
        assert_eq!(err, "ValueError: linkage: labels column 'nope' not found");
    }

    #[test]
    fn a_label_column_of_any_dtype_is_read_for_its_nulls() {
        use arrow::array::{Date32Array, DictionaryArray, TimestampMillisecondArray};
        use arrow::datatypes::Int32Type;
        pyo3::Python::initialize();
        let base = labelled_batch();
        let expected = apply(&labelled_spec(LinkageAxis::Rows), &base).unwrap();
        let labels: [(&str, ArrayRef); 3] = [
            ("Date32", Arc::new(Date32Array::from(vec![Some(1), None, Some(3), Some(4)]))),
            ("Timestamp", Arc::new(TimestampMillisecondArray::from(vec![Some(1), None, Some(3), Some(4)]))),
            ("Dictionary", Arc::new(vec![Some("a"), None, Some("b"), Some("c")].into_iter().collect::<DictionaryArray<Int32Type>>())),
        ];
        for (dtype, label) in labels {
            let mut fields: Vec<Field> = base.schema().fields().iter().map(|f| f.as_ref().clone()).collect();
            fields[0] = Field::new("label", label.data_type().clone(), true);
            let mut cols = base.columns().to_vec();
            cols[0] = label;
            let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).unwrap();
            let ctx = TransformContext::default();
            let got = apply_with_context(&labelled_spec(LinkageAxis::Rows), &batch, &ctx).unwrap();
            assert_eq!(got, expected, "{dtype}");
            assert_eq!(
                warnings_of(&ctx),
                ["Linkage: 1 row has a null 'label' and belongs to no group, so it was dropped; fill or filter the nulls in 'label'"],
                "{dtype}"
            );
        }
    }

    #[test]
    fn a_numeric_label_column_is_never_a_feature() {
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![
            Field::new("label", DataType::Int64, true),
            Field::new("f0", DataType::Float64, false),
            Field::new("f1", DataType::Float64, false),
            Field::new("f2", DataType::Float64, false),
        ]));
        let col = |v: Vec<f64>| -> ArrayRef { Arc::new(Float64Array::from(v)) };
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![7, 3, 90, 1])), col(vec![1.0, 100.0, 1.2, 5.0]), col(vec![2.0, -100.0, 2.1, 9.0]), col(vec![3.0, 50.0, 3.3, 1.0])],
        )
        .unwrap();
        let features = batch.project(&[1, 2, 3]).unwrap();
        for axis in [LinkageAxis::Rows, LinkageAxis::Columns] {
            let labelled = apply(&labelled_spec(axis), &batch).unwrap();
            let plain = LinkageSpec { axis, ..ward_spec() };
            assert_eq!(labelled, apply(&plain, &features).unwrap(), "{axis:?}: the label is not clustered");
            assert_ne!(apply(&plain, &batch).unwrap(), labelled, "{axis:?}: control, unlabelled the id is a feature");
        }
    }

    #[test]
    fn a_labelled_order_records_its_label_and_the_rows_it_read() {
        pyo3::Python::initialize();
        let batch = labelled_batch();
        for axis in [LinkageAxis::Rows, LinkageAxis::Columns] {
            let spec = labelled_spec(axis);
            let primary = apply(&spec, &batch).unwrap();
            let order = secondary_outputs(&spec, &batch, &primary).unwrap().into_iter().find(|(k, _)| k == "order").unwrap().1;
            assert_eq!(LinkageOrderMeta::read(&order), Some(LinkageOrderMeta { label: "label".into(), rows: 4 }), "{axis:?}");
        }
        let unlabelled = ward_spec();
        let primary = apply(&unlabelled, &batch).unwrap();
        let order = secondary_outputs(&unlabelled, &batch, &primary).unwrap().into_iter().find(|(k, _)| k == "order").unwrap().1;
        assert_eq!(LinkageOrderMeta::read(&order), None);
        assert!(order.schema().metadata().is_empty());
        let every_null = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("label", DataType::Utf8, true), Field::new("f0", DataType::Float64, false)])),
            vec![Arc::new(StringArray::from(vec![None::<&str>; 3])), Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0]))],
        )
        .unwrap();
        let spec = labelled_spec(LinkageAxis::Rows);
        let primary = apply(&spec, &every_null).unwrap();
        let order = secondary_outputs(&spec, &every_null, &primary).unwrap().into_iter().find(|(k, _)| k == "order").unwrap().1;
        assert_eq!(order.num_rows(), 0);
        assert_eq!(LinkageOrderMeta::read(&order), Some(LinkageOrderMeta { label: "label".into(), rows: 3 }));
    }

    #[test]
    fn labels_round_trips_through_the_wire_and_is_omitted_when_unset() {
        let with = TransformSpec::Linkage(labelled_spec(LinkageAxis::Rows));
        let json = serde_json::to_string(&with).unwrap();
        assert!(json.contains(r#""labels":"label""#), "{json}");
        assert_eq!(serde_json::from_str::<TransformSpec>(&json).unwrap(), with);
        let without = serde_json::to_string(&TransformSpec::Linkage(ward_spec())).unwrap();
        assert!(!without.contains("labels"), "{without}");
    }
}
