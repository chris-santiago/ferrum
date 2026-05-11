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

// ---------- Spec types ----------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

/// Extract a 2D matrix (rows x cols) of f64 from all Float64 columns in a RecordBatch.
/// Returns (n_rows, n_features, flat row-major Vec<f64>).
fn extract_matrix(batch: &RecordBatch) -> PyResult<(usize, usize, Vec<f64>)> {
    let schema = batch.schema();
    let mut col_indices: Vec<usize> = Vec::new();
    for (i, field) in schema.fields().iter().enumerate() {
        if field.data_type() == &DataType::Float64 {
            col_indices.push(i);
        }
    }
    if col_indices.is_empty() {
        return Err(PyValueError::new_err(
            "linkage: input must have at least one Float64 column",
        ));
    }
    let n_rows = batch.num_rows();
    let n_cols = col_indices.len();
    let mut mat = vec![0.0f64; n_rows * n_cols];
    for (feat_idx, &col_idx) in col_indices.iter().enumerate() {
        let arr = batch
            .column(col_idx)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        for row in 0..n_rows {
            mat[row * n_cols + feat_idx] = arr.value(row);
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
fn condensed_distances(
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
    // Validate preprocessing flags.
    if spec.z_score.is_some() && spec.standard_scale.is_some() {
        return Err(PyValueError::new_err(
            "linkage: z_score and standard_scale are mutually exclusive",
        ));
    }

    // Extract matrix.
    let (n_rows, n_feat, mut mat) = extract_matrix(batch)?;

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

    let linkage_schema = Arc::new(Schema::new(vec![
        Field::new("node_id", DataType::Int64, false),
        Field::new("left", DataType::Int64, false),
        Field::new("right", DataType::Int64, false),
        Field::new("distance", DataType::Float64, false),
        Field::new("n_obs", DataType::Int64, false),
    ]));
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

/// Compute the three secondary named outputs (order, coords, segments).
///
/// `linkage_batch` is the RecordBatch returned by `apply()` — it has the
/// schema [node_id: Int64, left: Int64, right: Int64, distance: Float64, n_obs: Int64].
/// We reconstruct the dendrogram tree directly from those columns rather than
/// re-running the full distance computation.
pub(crate) fn secondary_outputs(
    spec: &LinkageSpec,
    linkage_batch: &RecordBatch,
) -> PyResult<Vec<(String, RecordBatch)>> {
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
        orig_idxs.push(orig as i64);
        new_idxs.push(pos as i64);
    }
    let order_schema = Arc::new(Schema::new(vec![
        Field::new("original_idx", DataType::Int64, false),
        Field::new("new_idx", DataType::Int64, false),
    ]));
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
    let coords_schema = Arc::new(Schema::new(vec![
        Field::new("node_id", DataType::Int64, false),
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
    ]));
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

    let segments_schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
        Field::new("x2", DataType::Float64, false),
        Field::new("y2", DataType::Float64, false),
    ]));
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

    // Build output keys based on the naming convention.
    let (key_linkage, key_order, key_coords, key_segments) = match &spec.name {
        Some(s) => (
            format!("{s}_linkage"),
            format!("{s}_order"),
            format!("{s}_coords"),
            format!("{s}_segments"),
        ),
        None => (
            "linkage".to_string(),
            "order".to_string(),
            "coords".to_string(),
            "segments".to_string(),
        ),
    };

    // We also publish the linkage batch itself under the explicit key so that named
    // transforms have a stable key alongside the three derived outputs.
    Ok(vec![
        (key_linkage, linkage_batch.clone()),
        (key_order, order_batch),
        (key_coords, coords_batch),
        (key_segments, segments_batch),
    ])
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
    ))]
    fn new(
        method: &str,
        metric: &str,
        axis: &str,
        z_score: Option<&str>,
        standard_scale: Option<&str>,
        name: Option<String>,
    ) -> PyResult<Self> {
        let method = match method {
            "single" => LinkageMethod::Single,
            "complete" => LinkageMethod::Complete,
            "average" => LinkageMethod::Average,
            "weighted" => LinkageMethod::Weighted,
            "centroid" => LinkageMethod::Centroid,
            "median" => LinkageMethod::Median,
            "ward" => LinkageMethod::Ward,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Linkage: unknown method '{method}'; expected single|complete|average|weighted|centroid|median|ward"
                )))
            }
        };
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
        let axis = match axis {
            "rows" => LinkageAxis::Rows,
            "columns" => LinkageAxis::Columns,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Linkage: unknown axis '{axis}'; expected rows|columns"
                )))
            }
        };
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
        })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Linkage(s) => format!(
                "Linkage(method={:?}, metric={:?}, axis={:?})",
                s.method, s.metric, s.axis,
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
            standard_scale: None,
            name: None,
        }
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

        let secondaries = secondary_outputs(&spec, &primary).unwrap();
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

        let secondaries = secondary_outputs(&spec, &primary).unwrap();
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
        let secondaries = secondary_outputs(&spec_unnamed, &primary).unwrap();
        let keys: Vec<&str> = secondaries.iter().map(|(k, _)| k.as_str()).collect();
        assert!(keys.contains(&"linkage"), "unnamed should use bare 'linkage'");
        assert!(keys.contains(&"order"));
        assert!(keys.contains(&"coords"));
        assert!(keys.contains(&"segments"));

        // Named convention.
        let spec_named = LinkageSpec { name: Some("clust".to_string()), ..ward_spec() };
        let primary2 = apply(&spec_named, &batch).unwrap();
        let secondaries2 = secondary_outputs(&spec_named, &primary2).unwrap();
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
            standard_scale: Some(StdScaleAxis::Rows),
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
        let secondaries = secondary_outputs(&spec, &primary).unwrap();
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
                standard_scale: None,
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
            let secondaries = secondary_outputs(&spec, &primary).unwrap_or_else(|e| {
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
        let secondaries = secondary_outputs(&spec, &primary).unwrap();
        let (_, coords_batch) = secondaries.iter().find(|(k, _)| k == "coords").unwrap();
        // 2n-1 = 7 for n=4.
        assert_eq!(coords_batch.num_rows(), 7);
        let (_, order_batch) = secondaries.iter().find(|(k, _)| k == "order").unwrap();
        assert_eq!(order_batch.num_rows(), 4);
        let (_, seg_batch) = secondaries.iter().find(|(k, _)| k == "segments").unwrap();
        assert_eq!(seg_batch.num_rows(), 9); // 3*(4-1) = 9
    }
}
