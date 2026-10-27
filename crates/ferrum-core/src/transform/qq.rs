//! QQ (quantile-quantile) transform.
//!
//! Primary output: per-row points with `theoretical: f64, sample: f64` over
//! the non-null/non-NaN values of `field`, sorted ascending by sample value.
//!
//! Secondary output (`qq_line`, when `emit_line=true`): a single-row batch with
//! columns `qq_line_x_start, qq_line_x_end, qq_line_y_start, qq_line_y_end`
//! describing a robust quartile-fit reference line.
//!
//! With a non-empty `groupby`, both are computed per group (groups in
//! first-appearance order, each group's points sorted by sample value), the
//! reference-line batch has one row per group, and both batches append one
//! column per `groupby` field, named and typed as its input column, after
//! their own columns. A row with a null key belongs to no group, the shared
//! partitioner's skip policy.

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::transform::context::TransformContext;
use crate::transform::group_key::{
    batch_with_groupby_columns, deserialize_groupby_vec, group_partition_multi,
    refuse_groupby_output_collision, refuse_repeated_groupby_names,
    KeyValue, NullKeyPolicy,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct QQSpec {
    pub field: String,
    #[serde(default = "default_distribution")]
    pub distribution: String,
    #[serde(default)]
    pub dequantize: bool,
    #[serde(default = "default_emit_line")]
    pub emit_line: bool,
    #[serde(default = "default_seed")]
    pub seed: u64,
    /// Key columns: the points and the reference line are computed per
    /// group, and both batches carry these columns.
    #[serde(
        skip_serializing_if = "Vec::is_empty",
        default,
        deserialize_with = "deserialize_groupby_vec"
    )]
    pub groupby: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

fn default_distribution() -> String {
    "normal".into()
}

/// The single source of truth for `QQ`'s free-string `distribution`
/// vocabulary. Mirrors `smooth::METHODS`/`kde::KERNELS`'s structural-guard
/// shape (batch-D T9/T11): consumed by THREE sites —
/// `registry::entry_for_kind("qq")`'s `methods` list (`registry/mod.rs`, via
/// `DISTRIBUTIONS.to_vec()`), `PyQq::new`'s pre-check below, and
/// `theoretical_quantile`'s "unknown distribution" error (built FROM this
/// slice, not a hand-kept parallel literal).
///
/// The reverse direction (`DISTRIBUTIONS ⊆ theoretical_quantile`'s match
/// arms — an entry added here without a corresponding dispatch arm) is NOT
/// compile-time-enforced, mirroring `kde::KERNELS`'s doc note on the same
/// gap: `PyQq::new`'s pre-check only proves `DISTRIBUTIONS.contains(&d)` —
/// it never calls `theoretical_quantile`, so it cannot catch a
/// `DISTRIBUTIONS` entry with no matching arm.
/// `test_all_distributions_reach_a_dispatch_arm` (below, round 2 batch-D
/// T11 cycle 2) closes this gap for real by iterating `DISTRIBUTIONS`
/// itself through `theoretical_quantile` directly.
pub(crate) const DISTRIBUTIONS: &[&str] = &["normal", "uniform", "exponential", "lognormal"];
fn default_emit_line() -> bool {
    true
}
fn default_seed() -> u64 {
    0
}

/// Inverse standard normal CDF — Beasley-Springer-Moro / Wichura AS241 hybrid.
/// Accurate to ~1e-7 across (0, 1). Caller must guarantee `p ∈ (0, 1)`.
fn inverse_normal_cdf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = q * q;
        let num = ((((-25.44106049637_f64 * r + 41.39119773534) * r - 18.61500062529) * r
            + 2.506628277459)) * q;
        let den = (((3.130829354000_f64 * r - 21.06224101826) * r + 23.08336743743) * r
            - 8.473511541334)
            * r
            + 1.0;
        return num / den;
    }
    let r = if q < 0.0 { p } else { 1.0 - p };
    let r = (-r.ln()).sqrt();
    let val = if r <= 5.0 {
        let r = r - 1.6;
        let num = ((((((0.000077454501427834140764_f64 * r + 0.0227238449892691845833) * r
            + 0.24178072517745061177)
            * r
            + 1.27045825245236838258)
            * r
            + 3.64784832476320460504)
            * r
            + 5.7694972214606914055)
            * r
            + 4.6303378461565452959)
            * r
            + 1.42343711074968357734;
        let den = ((((((0.00000000105075007164441684324_f64 * r + 0.000547593808499534494600)
            * r
            + 0.0151986665636164571966)
            * r
            + 0.14810397642748007459)
            * r
            + 0.68976733498510000455)
            * r
            + 1.6763848301838038494)
            * r
            + 2.05319162663775882187)
            * r
            + 1.0;
        num / den
    } else {
        let r = r - 5.0;
        let num = ((((((0.000000201033439929228813265_f64 * r + 0.0000271155556874348757815)
            * r
            + 0.0012426609473880784386)
            * r
            + 0.026532189526576123093)
            * r
            + 0.29656057182850489123)
            * r
            + 1.7848265399172913358)
            * r
            + 5.4637849111641143699)
            * r
            + 6.6579046435011037772;
        let den =
            ((((((0.00000000000000204426310338993978564_f64 * r + 0.00000000014215117583164458887)
                * r
                + 0.000000018463183175100546818)
                * r
                + 0.0007868691311456132591)
                * r
                + 0.0148753612908506148525)
                * r
                + 0.13692988092273580531)
                * r
                + 0.59983220655588793769)
                * r
                + 1.0;
        num / den
    };
    if q < 0.0 {
        -val
    } else {
        val
    }
}

/// Theoretical quantile at plotting position `p` for the requested reference
/// distribution.
///
/// `"lognormal"` follows the standard Q-Q convention for a distribution
/// defined as `Y = exp(X)` with `X ~ Normal(0, 1)`: the theoretical quantile
/// at `p` is the standard-normal quantile at `p` exponentiated,
/// `Q(p) = exp(Φ⁻¹(p))`. Equivalently, this is what you get by log-
/// transforming the (positive) sample, computing standard-normal theoretical
/// quantiles against the log-sample's plotting positions (unchanged by the
/// log transform, since log is monotone), and exponentiating the result back
/// onto the original scale.
fn theoretical_quantile(p: f64, distribution: &str) -> PyResult<f64> {
    match distribution {
        "normal" => Ok(inverse_normal_cdf(p)),
        "uniform" => Ok(p),
        "exponential" => Ok(-((1.0 - p).ln())),
        "lognormal" => Ok(inverse_normal_cdf(p).exp()),
        other => Err(PyValueError::new_err(format!(
            "stat_qq: unknown distribution '{other}'; expected {}",
            DISTRIBUTIONS.iter().map(|d| format!("'{d}'")).collect::<Vec<_>>().join(" | "),
        ))),
    }
}

/// Compute primary outputs (theoretical, sample) for a sorted sample.
/// `samples_sorted` must be ascending. `theoretical` length == `samples_sorted` length.
fn compute_theoretical(
    samples_sorted: &[f64],
    distribution: &str,
    dequantize: bool,
    seed: u64,
) -> PyResult<Vec<f64>> {
    let n = samples_sorted.len();
    let mut p: Vec<f64> = (1..=n).map(|i| ((i as f64) - 0.5) / (n as f64)).collect();

    if dequantize {
        // For each run of equal sample values, jitter the corresponding p values
        // (deterministically via a ChaCha8Rng seeded with `spec.seed`) so the
        // theoretical positions are spread across the run instead of being a
        // strict arithmetic sequence.
        use rand::Rng;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut i = 0;
        while i < n {
            let mut j = i + 1;
            while j < n && samples_sorted[j] == samples_sorted[i] {
                j += 1;
            }
            let run = j - i;
            if run >= 2 {
                // Replace p[i..j] with random permutation drawn uniformly within
                // the run's existing p-range. This keeps points within their
                // original quantile band but breaks the deterministic ordering
                // induced by the tie.
                let lo = p[i];
                let hi = p[j - 1];
                let mut jittered: Vec<f64> = (0..run)
                    .map(|_| lo + rng.gen::<f64>() * (hi - lo))
                    .collect();
                jittered.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                for (k, val) in jittered.into_iter().enumerate() {
                    p[i + k] = val;
                }
            }
            i = j;
        }
    }

    let mut theoretical = Vec::with_capacity(n);
    for pi in &p {
        theoretical.push(theoretical_quantile(*pi, distribution)?);
    }
    Ok(theoretical)
}

pub(crate) fn apply(spec: &QQSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    apply_with_context(spec, batch, &TransformContext::default())
}

/// The primary batch's own column names, before any `groupby` column.
const POINT_COLUMNS: [&str; 2] = ["theoretical", "sample"];

/// Every column name this transform writes itself, across the points and the
/// reference line: the names a `groupby` field is refused against.
fn output_column_names() -> Vec<String> {
    POINT_COLUMNS
        .iter()
        .copied()
        .chain(QQ_LINE_COLUMNS.iter().map(|(name, _)| *name))
        .map(str::to_string)
        .collect()
}

/// Context-aware entry point. `ctx` is the degrade-warning sink for the
/// null-key warnings: a row whose `groupby` key holds a null belongs to no
/// group and is dropped, with one warning per field naming the dropped-row
/// count.
///
/// An empty `groupby` partitions every row into one group, so the ungrouped
/// call computes exactly the one whole-column statistic it always has. A field
/// with no plottable value refuses, grouped or not; so does a whole-frame call
/// whose null-key drop removed at least one row and left no plottable value;
/// inside a facet panel that drop empties the panel instead
/// ([`TransformContext::refuse_unless_partition`]). A group with none emits
/// no points.
pub(crate) fn apply_with_context(
    spec: &QQSpec,
    batch: &RecordBatch,
    ctx: &TransformContext,
) -> PyResult<RecordBatch> {
    crate::transform::vocabulary::check_declared(
        DISTRIBUTIONS,
        "stat_qq",
        "distribution",
        &spec.distribution,
    )?;
    let schema = batch.schema();
    let v_idx = schema
        .index_of(&spec.field)
        .map_err(|_| PyValueError::new_err(format!(
            "stat_qq: column '{}' not found",
            spec.field
        )))?;
    let v_arr = coerce_to_float64(batch.column(v_idx), "stat_qq", &spec.field)?;

    refuse_groupby_output_collision("stat_qq", &spec.groupby, &output_column_names())?;
    // The partition counts the null-key rows it drops, per field; a batch
    // the drop leaves with nothing to plot is refused below, which those
    // counts explain.
    let partition =
        group_partition_multi(batch, &spec.groupby, "stat_qq", NullKeyPolicy::SkipRefusingEmpty, ctx)?;

    // Null/NaN sample values are not plotted.
    let clean = |i: usize| -> Option<f64> {
        if v_arr.is_null(i) {
            return None;
        }
        let v = v_arr.value(i);
        (!v.is_nan()).then_some(v)
    };
    if !(0..v_arr.len()).any(|i| clean(i).is_some()) {
        return Err(PyValueError::new_err("stat_qq: empty input after filter"));
    }
    let kept_rows: usize = partition.groups.iter().map(|(_, rows)| rows.len()).sum();
    let dropped_rows = batch.num_rows().saturating_sub(kept_rows);
    let plottable_kept =
        partition.groups.iter().flat_map(|(_, rows)| rows).any(|&i| clean(i).is_some());
    if dropped_rows > 0 && !plottable_kept {
        ctx.refuse_unless_partition(PyValueError::new_err("stat_qq: empty input after filter"))?;
    }

    let mut theoretical: Vec<f64> = Vec::new();
    let mut samples: Vec<f64> = Vec::new();
    let mut keys: Vec<Vec<KeyValue>> = Vec::new();
    for (key, rows) in &partition.groups {
        let mut group_samples: Vec<f64> = rows.iter().filter_map(|&i| clean(i)).collect();
        group_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let group_theoretical =
            compute_theoretical(&group_samples, &spec.distribution, spec.dequantize, spec.seed)?;
        keys.extend(std::iter::repeat_n(key.clone(), group_samples.len()));
        theoretical.extend(group_theoretical);
        samples.extend(group_samples);
    }

    let fields: Vec<Field> =
        POINT_COLUMNS.iter().map(|name| Field::new(*name, DataType::Float64, false)).collect();
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(theoretical)),
        Arc::new(Float64Array::from(samples)),
    ];
    batch_with_groupby_columns(&spec.groupby, &partition.dtypes, &keys, fields, cols, "stat_qq")
}

/// The reference-line batch's key under default construction; a spec `name`
/// prefixes it as `"{name}_qq_line"`, mirroring `letter_value::OUTLIERS_KEY`
/// and `linkage::LINKAGE_KEY`'s convention.
pub(crate) const QQ_LINE_KEY: &str = "qq_line";

/// The reference-line batch's column schema — the single source of truth,
/// read by both [`secondary_outputs`]'s schema builder below and
/// `registry::entry_for_kind("qq")`'s `secondary_outputs` list (R1 design
/// finding (d): these four names were unchecked Python literals before).
pub(crate) const QQ_LINE_COLUMNS: &[(&str, DataType)] = &[
    ("qq_line_x_start", DataType::Float64),
    ("qq_line_x_end", DataType::Float64),
    ("qq_line_y_start", DataType::Float64),
    ("qq_line_y_end", DataType::Float64),
];

/// Compute the secondary `qq_line` named output from the primary points batch.
///
/// Returns `vec![("qq_line", line_batch)]` when `emit_line=true`, else empty.
/// The line is a robust quartile fit:
///   slope     = (sample_q3 - sample_q1) / (theo_q3 - theo_q1)
///   intercept = sample_q2 - slope * theo_q2
/// Endpoints span the theoretical-x range.
///
/// With a `groupby`, the batch has one line per group (read back off the
/// primary batch's `groupby` columns, in its group order) and is published
/// even when no group has a line, so a layer reading it draws nothing rather
/// than missing its source. Without one, a primary batch with no line to fit
/// publishes no batch, as it always has.
pub(crate) fn secondary_outputs(
    spec: &QQSpec,
    primary: &RecordBatch,
) -> PyResult<Vec<(String, RecordBatch)>> {
    if !spec.emit_line {
        return Ok(Vec::new());
    }
    // Primary's first column is theoretical, second is sample (we just built it).
    let theo = primary
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| PyValueError::new_err("stat_qq: secondary expected theoretical Float64"))?;
    let samp = primary
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| PyValueError::new_err("stat_qq: secondary expected sample Float64"))?;

    // The primary batch carries no null key (the points were partitioned
    // with the skip policy), so `Pool` here only re-reads its groups.
    let partition = group_partition_multi(
        primary,
        &spec.groupby,
        "stat_qq",
        NullKeyPolicy::Pool,
        &TransformContext::default(),
    )?;
    let mut keys: Vec<Vec<KeyValue>> = Vec::new();
    let mut endpoints: [Vec<f64>; 4] = Default::default();
    for (key, rows) in &partition.groups {
        let pairs: Vec<(f64, f64)> = rows.iter().map(|&i| (theo.value(i), samp.value(i))).collect();
        if let Some(line) = quartile_fit_line(&pairs) {
            keys.push(key.clone());
            for (column, value) in endpoints.iter_mut().zip(line) {
                column.push(value);
            }
        }
    }
    if spec.groupby.is_empty() && keys.is_empty() {
        return Ok(Vec::new());
    }

    let fields: Vec<Field> = QQ_LINE_COLUMNS
        .iter()
        .map(|(name, dt)| Field::new(*name, dt.clone(), false))
        .collect();
    let cols: Vec<ArrayRef> = endpoints
        .into_iter()
        .map(|column| Arc::new(Float64Array::from(column)) as ArrayRef)
        .collect();
    let line_batch =
        batch_with_groupby_columns(&spec.groupby, &partition.dtypes, &keys, fields, cols, "stat_qq: qq_line")?;

    let key = crate::transform::registry::secondary_key(spec.name.as_deref(), QQ_LINE_KEY);
    Ok(vec![(key, line_batch)])
}

/// One group's quartile-fit reference line as `QQ_LINE_COLUMNS`' four values
/// `[x_start, x_end, y_start, y_end]`, from its `(theoretical, sample)`
/// points in sample order. `None` when no point has a finite theoretical
/// quantile.
fn quartile_fit_line(points: &[(f64, f64)]) -> Option<[f64; 4]> {
    // Filter to finite values only (Beasley-Springer-Moro maps p=0/1 → ±inf,
    // which won't occur for plotting positions but is defensive here).
    let pairs: Vec<(f64, f64)> = points.iter().copied().filter(|(t, _)| t.is_finite()).collect();
    if pairs.is_empty() {
        return None;
    }

    // The primary batch is sorted by sample (ascending). For the quartile fit
    // we need quantiles of theoretical (also ascending after sorting samples,
    // because plotting positions are increasing and the inverse CDF is
    // monotone) AND of sample values. They're already aligned.
    let theo_sorted: Vec<f64> = pairs.iter().map(|(t, _)| *t).collect();
    let samp_sorted: Vec<f64> = pairs.iter().map(|(_, s)| *s).collect();
    let theo_q1 = quantile_sorted(&theo_sorted, 0.25);
    let theo_q2 = quantile_sorted(&theo_sorted, 0.5);
    let theo_q3 = quantile_sorted(&theo_sorted, 0.75);
    let samp_q1 = quantile_sorted(&samp_sorted, 0.25);
    let samp_q2 = quantile_sorted(&samp_sorted, 0.5);
    let samp_q3 = quantile_sorted(&samp_sorted, 0.75);

    let denom = theo_q3 - theo_q1;
    let slope = if denom == 0.0 {
        0.0
    } else {
        (samp_q3 - samp_q1) / denom
    };
    let intercept = samp_q2 - slope * theo_q2;

    let theo_min = theo_sorted.iter().copied().fold(f64::INFINITY, f64::min);
    let theo_max = theo_sorted.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let y_min = slope * theo_min + intercept;
    let y_max = slope * theo_max + intercept;
    Some([theo_min, theo_max, y_min, y_max])
}

use crate::transform::numeric_util::{coerce_to_float64, quantile_sorted};

// ---------- PyO3 wrapper ----------

use pyo3::prelude::*;

use crate::transform::core::TransformSpec;

/// Quantile-quantile (Q-Q) plot point computation.
///
/// Maps the empirical quantiles of ``field`` against the theoretical
/// quantiles of a reference distribution. Optionally emits the reference
/// line (slope 1 through the first and third quartiles). When
/// ``dequantize=True``, adds a small random jitter to break ties in
/// discrete data (seeded by ``seed`` for reproducibility).
///
/// Output columns: ``theoretical`` (Float64 reference quantiles) and
/// ``sample`` (Float64 empirical quantiles), one row per plotted value,
/// sorted by ``sample``, then one column per ``groupby`` field. When
/// ``emit_line=True``, a second named output ``qq_line`` (``<name>_qq_line``
/// when ``name`` is set) holds the reference line: ``qq_line_x_start``,
/// ``qq_line_x_end``, ``qq_line_y_start``, ``qq_line_y_end`` (Float64), one
/// row per group (one row when ungrouped), then the same ``groupby`` columns.
///
/// Parameters
/// ----------
/// field : str
///     Numeric column to plot (must be Float64).
/// distribution : {"normal", "uniform", "exponential", "lognormal"}, default "normal"
///     Reference distribution for theoretical quantiles. ``"lognormal"``
///     quantiles are the standard-normal theoretical quantiles exponentiated
///     (``exp(Φ⁻¹(p))``) — the standard Q-Q convention for a variable whose
///     log is normally distributed.
/// dequantize : bool, default False
///     When True, add small seeded random jitter to break discrete ties.
/// emit_line : bool, default True
///     When True, also publish the ``qq_line`` output: a robust quartile-fit
///     reference line spanning each group's theoretical range.
/// seed : int, default 0
///     RNG seed for dequantization jitter. Seeds ``ChaCha8Rng`` for
///     byte-deterministic output across platforms.
/// groupby : list of str, default []
///     Columns to compute the points and the reference line per group of;
///     both outputs carry one column per field, named and typed as its input
///     column. A row with a null key belongs to no group and is dropped, with
///     one warning per field naming the dropped-row count; a call left with
///     no plottable value raises.
/// name : str, optional
///     Named output label for sibling ``Reorder(from_=...)`` lookup.
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> fm.Chart(df).mark_qqplot().encode(x="residuals")
#[pyclass(eq, module = "ferrum._core", name = "QQ")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyQq(pub(crate) TransformSpec);

#[pymethods]
impl PyQq {
    #[new]
    #[pyo3(signature = (
        field,
        *,
        distribution = "normal",
        dequantize = false,
        emit_line = true,
        seed = 0,
        groupby = vec![],
        name = None,
    ))]
    fn new(
        field: &str,
        distribution: &str,
        dequantize: bool,
        emit_line: bool,
        seed: u64,
        groupby: Vec<String>,
        name: Option<String>,
    ) -> PyResult<Self> {
        if field.is_empty() {
            return Err(PyValueError::new_err("QQ: field must be non-empty"));
        }
        crate::transform::vocabulary::check_declared(
            DISTRIBUTIONS,
            "QQ",
            "distribution",
            distribution,
        )?;
        refuse_repeated_groupby_names("QQ", &groupby)?;
        Ok(PyQq(TransformSpec::Qq(QQSpec {
            field: field.to_string(),
            distribution: distribution.to_string(),
            dequantize,
            emit_line,
            seed,
            groupby,
            name,
        })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Qq(s) => format!(
                "QQ(field='{}', distribution='{}', dequantize={}, emit_line={}, seed={}, groupby={:?}, name={:?})",
                s.field, s.distribution, s.dequantize, s.emit_line, s.seed, s.groupby, s.name,
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType, Field, Schema};

    fn batch_field(name: &str, values: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::Float64, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(values))]).unwrap()
    }

    fn col(b: &RecordBatch, name: &str) -> Vec<f64> {
        let i = b.schema().index_of(name).unwrap();
        let a = b.column(i).as_any().downcast_ref::<Float64Array>().unwrap();
        (0..a.len()).map(|j| a.value(j)).collect()
    }

    #[test]
    fn qq_normal_distribution_matches_scipy_reference() {
        // Sample [-2, -1, 0, 1, 2] (sorted), n=5 → plotting positions
        // 0.1, 0.3, 0.5, 0.7, 0.9. scipy.stats.norm.ppf gives:
        //   -1.2816, -0.5244, 0.0, 0.5244, 1.2816.
        let batch = batch_field("v", vec![0.0, 1.0, -1.0, 2.0, -2.0]);
        let spec = QQSpec {
            groupby: Vec::new(),
            field: "v".into(),
            distribution: "normal".into(),
            dequantize: false,
            emit_line: false,
            seed: 0,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let theo = col(&out, "theoretical");
        let samp = col(&out, "sample");
        assert_eq!(samp, vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        let refs = [-1.2816, -0.5244, 0.0, 0.5244, 1.2816];
        for (i, (got, want)) in theo.iter().zip(refs.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "i={i}: theoretical {got} not within 1e-3 of {want}"
            );
        }
    }

    // ---- T11: lognormal distribution ----

    #[test]
    fn qq_lognormal_theoretical_matches_exp_of_normal_quantile() {
        let p = [0.1, 0.3, 0.5, 0.7, 0.9];
        for &pi in &p {
            let want = inverse_normal_cdf(pi).exp();
            let got = theoretical_quantile(pi, "lognormal").unwrap();
            assert!((got - want).abs() < 1e-12, "lognormal[{pi}]: {got} vs {want}");
        }
    }

    /// RED-proof discriminator: a sample built as EXACTLY exp(normal
    /// theoretical quantile) at its own plotting positions is a perfect
    /// lognormal fit. Under `distribution="lognormal"` the (theoretical,
    /// sample) points must land on `y=x`; under `distribution="normal"` the
    /// same sample must show large, systematic deviation from `y=x` at the
    /// tails, proving the two distributions produce genuinely different
    /// shapes on identical data rather than the same computation under two
    /// names (which is what "lognormal raises today" degenerates to before
    /// this fix, and what a silent normal-alias would degenerate to after).
    #[test]
    fn qq_lognormal_discriminates_from_normal_on_lognormal_data() {
        let n = 9usize;
        let sample: Vec<f64> = (1..=n)
            .map(|i| {
                let p = (i as f64 - 0.5) / n as f64;
                inverse_normal_cdf(p).exp()
            })
            .collect();
        let batch = batch_field("v", sample);

        let spec_lognormal = QQSpec {
            groupby: Vec::new(),
            field: "v".into(),
            distribution: "lognormal".into(),
            dequantize: false,
            emit_line: false,
            seed: 0,
            name: None,
        };
        let out_ln = apply(&spec_lognormal, &batch).unwrap();
        let theo_ln = col(&out_ln, "theoretical");
        let samp_ln = col(&out_ln, "sample");
        for i in 0..n {
            assert!(
                (theo_ln[i] - samp_ln[i]).abs() < 1e-9,
                "lognormal QQ point {i} should lie on y=x: theo={} sample={}",
                theo_ln[i],
                samp_ln[i]
            );
        }

        let spec_normal = QQSpec {
            groupby: Vec::new(),
            field: "v".into(),
            distribution: "normal".into(),
            dequantize: false,
            emit_line: false,
            seed: 0,
            name: None,
        };
        let out_n = apply(&spec_normal, &batch).unwrap();
        let theo_n = col(&out_n, "theoretical");
        let samp_n = col(&out_n, "sample");
        let last = n - 1;
        assert!(
            (theo_n[last] - samp_n[last]).abs() > 1.0,
            "normal QQ on lognormal data should deviate sharply from y=x at the tail: theo={} sample={}",
            theo_n[last],
            samp_n[last]
        );
    }

    /// Determinism pin (round 2, batch-D T11 cycle 2): `lognormal` must
    /// produce byte-identical output across repeated `apply()` calls on
    /// identical input, per the spec's testing section ("determinism: two
    /// runs byte-equal"). Rust owns this pin — see the round-2 report for
    /// why it isn't duplicated Python-side.
    #[test]
    fn qq_lognormal_apply_deterministic_across_runs() {
        let batch = batch_field("v", vec![0.5, 1.2, 2.3, 0.8, 3.1, 1.9]);
        let spec = QQSpec {
            groupby: Vec::new(),
            field: "v".into(),
            distribution: "lognormal".into(),
            dequantize: false,
            emit_line: false,
            seed: 0,
            name: None,
        };
        let out1 = apply(&spec, &batch).unwrap();
        let out2 = apply(&spec, &batch).unwrap();
        let theo1 = col(&out1, "theoretical");
        let theo2 = col(&out2, "theoretical");
        let samp1 = col(&out1, "sample");
        let samp2 = col(&out2, "sample");
        for (i, (a, b)) in theo1.iter().zip(theo2.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "theoretical[{i}] not byte-identical: {a} vs {b}");
        }
        for (i, (a, b)) in samp1.iter().zip(samp2.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "sample[{i}] not byte-identical: {a} vs {b}");
        }
    }

    /// `PyQq::new` must accept `"lognormal"` and still refuse a bogus string.
    #[test]
    fn qq_pyqq_new_accepts_lognormal_and_rejects_bogus() {
        pyo3::Python::initialize();
        let ok = PyQq::new("v", "lognormal", false, true, 0, Vec::new(), None);
        assert!(ok.is_ok(), "lognormal must be accepted: {:?}", ok.err().map(|e| e.to_string()));
        let bad = PyQq::new("v", "not_a_distribution", false, true, 0, Vec::new(), None);
        assert!(bad.is_err(), "bogus distribution must be rejected");
    }

    /// Reverse guard (round 2, batch-D T11 cycle 2): every `DISTRIBUTIONS`
    /// entry must reach a real dispatch arm in `theoretical_quantile`, not
    /// just pass `PyQq::new`'s `DISTRIBUTIONS.contains` pre-check — mirrors
    /// `kde::test_all_kernels_reach_a_dispatch_arm`.
    #[test]
    fn qq_all_distributions_reach_a_dispatch_arm() {
        for &distribution in DISTRIBUTIONS {
            let result = theoretical_quantile(0.5, distribution);
            assert!(
                result.is_ok(),
                "DISTRIBUTIONS entry {distribution:?} has no matching dispatch arm in \
                 theoretical_quantile: {:?}",
                result.err().map(|e| e.to_string())
            );
        }
    }

    #[test]
    fn qq_uniform_and_exponential_distributions() {
        // Sample = [1, 2, 3, 4]; sorted; n=4 → p = 0.125, 0.375, 0.625, 0.875.
        let batch = batch_field("v", vec![1.0, 2.0, 3.0, 4.0]);
        let p = vec![0.125, 0.375, 0.625, 0.875];

        // Uniform: theoretical = p.
        let spec_u = QQSpec {
            groupby: Vec::new(),
            field: "v".into(),
            distribution: "uniform".into(),
            dequantize: false,
            emit_line: false,
            seed: 0,
            name: None,
        };
        let out_u = apply(&spec_u, &batch).unwrap();
        let theo_u = col(&out_u, "theoretical");
        for (i, (got, want)) in theo_u.iter().zip(p.iter()).enumerate() {
            assert!((got - want).abs() < 1e-12, "uniform[{i}]: {got} vs {want}");
        }

        // Exponential: theoretical = -ln(1 - p).
        let spec_e = QQSpec {
            groupby: Vec::new(),
            field: "v".into(),
            distribution: "exponential".into(),
            dequantize: false,
            emit_line: false,
            seed: 0,
            name: None,
        };
        let out_e = apply(&spec_e, &batch).unwrap();
        let theo_e = col(&out_e, "theoretical");
        for (i, pi) in p.iter().enumerate() {
            let want = -((1.0 - pi).ln());
            assert!(
                (theo_e[i] - want).abs() < 1e-12,
                "exp[{i}]: {} vs {}",
                theo_e[i],
                want
            );
        }
    }

    #[test]
    fn qq_reference_line_slope_correctness() {
        // Sample = [1, 2, 3, 4, 5] under uniform → theoretical = 0.1, 0.3, 0.5, 0.7, 0.9.
        // Quantiles by linear interp on sorted ascending:
        //   theo Q1 = 0.3, Q2 = 0.5, Q3 = 0.7  → diff = 0.4
        //   samp Q1 = 2.0, Q2 = 3.0, Q3 = 4.0  → diff = 2.0
        // slope = 2.0 / 0.4 = 5.0
        // intercept = 3.0 - 5.0 * 0.5 = 0.5
        let batch = batch_field("v", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let spec = QQSpec {
            groupby: Vec::new(),
            field: "v".into(),
            distribution: "uniform".into(),
            dequantize: false,
            emit_line: true,
            seed: 0,
            name: None,
        };
        let primary = apply(&spec, &batch).unwrap();
        let secondary = secondary_outputs(&spec, &primary).unwrap();
        assert_eq!(secondary.len(), 1);
        let (key, line) = &secondary[0];
        assert_eq!(key, "qq_line");
        assert_eq!(line.num_columns(), 4);
        assert_eq!(line.num_rows(), 1);

        let x_start = col(line, "qq_line_x_start")[0];
        let x_end = col(line, "qq_line_x_end")[0];
        let y_start = col(line, "qq_line_y_start")[0];
        let y_end = col(line, "qq_line_y_end")[0];
        let slope = 5.0;
        let intercept = 0.5;
        assert!((y_start - (slope * x_start + intercept)).abs() < 1e-9);
        assert!((y_end - (slope * x_end + intercept)).abs() < 1e-9);
        // Endpoints should span theoretical min/max.
        assert!((x_start - 0.1).abs() < 1e-12);
        assert!((x_end - 0.9).abs() < 1e-12);
    }

    /// R3 close-remediation regression: a named `QQ` must prefix the
    /// `qq_line` secondary key exactly like `letter_value`'s `_outliers`/
    /// `_depth_{k}` and `linkage`'s four dendrogram keys — mirrors
    /// `letter_value::test_letter_value_named_outputs_use_name_prefix` and
    /// `linkage`'s `named should prefix` assertion.
    #[test]
    fn qq_named_output_uses_name_prefix() {
        let batch = batch_field("v", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let spec = QQSpec {
            groupby: Vec::new(),
            field: "v".into(),
            distribution: "uniform".into(),
            dequantize: false,
            emit_line: true,
            seed: 0,
            name: Some("qq".into()),
        };
        let primary = apply(&spec, &batch).unwrap();
        let secs = secondary_outputs(&spec, &primary).unwrap();
        assert_eq!(secs.len(), 1);
        let (key, _) = &secs[0];
        assert_eq!(key, "qq_qq_line", "named QQ should prefix its qq_line key: {key}");
    }

    #[test]
    fn qq_int64_column_is_coerced() {
        // Int64 input must be coerced to Float64, not rejected. Output must
        // match a Float64 batch of the same values bit-for-bit.
        use arrow::array::Int64Array;
        pyo3::Python::initialize();
        let vals_i: Vec<i64> = vec![0, 1, -1, 2, -2];
        let vals_f: Vec<f64> = vals_i.iter().map(|&v| v as f64).collect();
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, true)]));
        let int_batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vals_i))]).unwrap();
        let float_batch = batch_field("v", vals_f);
        let spec = QQSpec {
            groupby: Vec::new(),
            field: "v".into(),
            distribution: "normal".into(),
            dequantize: false,
            emit_line: false,
            seed: 0,
            name: None,
        };
        let int_out = apply(&spec, &int_batch).expect("Int64 QQ should succeed");
        let float_out = apply(&spec, &float_batch).unwrap();
        assert_eq!(int_out.num_rows(), 5);
        let it = col(&int_out, "theoretical");
        let ft = col(&float_out, "theoretical");
        let is = col(&int_out, "sample");
        let fs = col(&float_out, "sample");
        for i in 0..5 {
            assert_eq!(it[i].to_bits(), ft[i].to_bits(), "theoretical[{i}] mismatch");
            assert_eq!(is[i].to_bits(), fs[i].to_bits(), "sample[{i}] mismatch");
        }
    }

    #[test]
    fn qq_string_column_still_errors() {
        use arrow::array::StringArray;
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Utf8, true)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec!["a", "b", "c"]))],
        )
        .unwrap();
        let spec = QQSpec {
            groupby: Vec::new(),
            field: "v".into(),
            distribution: "normal".into(),
            dequantize: false,
            emit_line: false,
            seed: 0,
            name: None,
        };
        let err = apply(&spec, &batch).unwrap_err();
        assert!(
            err.to_string().contains("must be numeric"),
            "err: {err}"
        );
    }

    #[test]
    fn qq_dequantize_jitters_only_ties() {
        // 2 ties at 5.0 + 3 unique values. With dequantize=false, theoretical
        // positions for the unique rows are exactly (i - 0.5) / n (under
        // uniform). With dequantize=true + seed=42, the 2 tied rows' theoretical
        // values shift while unique rows are unchanged. Same seed twice = bit-
        // identical output (deterministic).
        let batch = batch_field("v", vec![5.0, 5.0, 1.0, 2.0, 8.0]);
        let spec_no_jitter = QQSpec {
            groupby: Vec::new(),
            field: "v".into(),
            distribution: "uniform".into(),
            dequantize: false,
            emit_line: false,
            seed: 42,
            name: None,
        };
        let no_jitter = apply(&spec_no_jitter, &batch).unwrap();
        let theo_plain = col(&no_jitter, "theoretical");
        // sorted samples: [1, 2, 5, 5, 8] → unique rows at indices 0, 1, 4.
        // Plotting positions (no jitter): [0.1, 0.3, 0.5, 0.7, 0.9].
        assert!((theo_plain[0] - 0.1).abs() < 1e-12);
        assert!((theo_plain[1] - 0.3).abs() < 1e-12);
        assert!((theo_plain[4] - 0.9).abs() < 1e-12);

        let spec_jitter = QQSpec {
            groupby: Vec::new(),
            field: "v".into(),
            distribution: "uniform".into(),
            dequantize: true,
            emit_line: false,
            seed: 42,
            name: None,
        };
        let jitter1 = apply(&spec_jitter, &batch).unwrap();
        let jitter2 = apply(&spec_jitter, &batch).unwrap();
        let t1 = col(&jitter1, "theoretical");
        let t2 = col(&jitter2, "theoretical");

        // Determinism: same seed → bit-identical.
        for i in 0..t1.len() {
            assert_eq!(
                t1[i].to_bits(),
                t2[i].to_bits(),
                "dequantize must be deterministic at row {i}: {} vs {}",
                t1[i],
                t2[i]
            );
        }
        // Unique-value rows unchanged.
        assert!(
            (t1[0] - 0.1).abs() < 1e-12,
            "unique row 0 jittered unexpectedly: {}",
            t1[0]
        );
        assert!(
            (t1[1] - 0.3).abs() < 1e-12,
            "unique row 1 jittered unexpectedly: {}",
            t1[1]
        );
        assert!(
            (t1[4] - 0.9).abs() < 1e-12,
            "unique row 4 jittered unexpectedly: {}",
            t1[4]
        );
    }

    // ── groupby: one Q-Q statistic per groupby combination ──────────────

    use arrow::array::StringArray;

    fn qq_spec(json: serde_json::Value) -> QQSpec {
        serde_json::from_value(json).expect("a well-formed QQ wire spec")
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

    fn some(vals: &[&str]) -> Vec<Option<String>> {
        vals.iter().map(|s| Some(s.to_string())).collect()
    }

    fn assert_close(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len(), "got {got:?}, want {want:?}");
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-12, "got {got:?}, want {want:?}");
        }
    }

    fn line_of(spec: &QQSpec, primary: &RecordBatch) -> RecordBatch {
        let outs = secondary_outputs(spec, primary).unwrap();
        assert_eq!(outs.len(), 1, "one reference-line batch");
        outs.into_iter().next().unwrap().1
    }

    /// Each group's plotting positions use its own size: 4 points at
    /// (i - 0.5)/4 and 3 at (i - 0.5)/3. Pooled, all 7 would sit at
    /// (i - 0.5)/7. Under the uniform reference the theoretical quantile is
    /// the plotting position itself, and each group's samples lie exactly on
    /// a line (40p + 5 and 3p + 0.5), so each group's reference line is known.
    #[test]
    fn qq_groupby_computes_per_group_and_carries_the_key() {
        let batch = keyed_batch(
            &[("h", vec![Some("x"), Some("y"), Some("x"), Some("y"), Some("x"), Some("y"), Some("x")])],
            vec![30.0, 3.0, 10.0, 1.0, 20.0, 2.0, 40.0],
        );
        let spec = qq_spec(serde_json::json!({
            "field": "v", "distribution": "uniform", "groupby": ["h"],
        }));
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(strs(&out, "h"), some(&["x", "x", "x", "x", "y", "y", "y"]));
        assert_eq!(col(&out, "sample"), vec![10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0]);
        assert_close(
            &col(&out, "theoretical"),
            &[0.125, 0.375, 0.625, 0.875, 1.0 / 6.0, 0.5, 5.0 / 6.0],
        );

        let line = line_of(&spec, &out);
        assert_eq!(strs(&line, "h"), some(&["x", "y"]));
        assert_close(&col(&line, "qq_line_x_start"), &[0.125, 1.0 / 6.0]);
        assert_close(&col(&line, "qq_line_x_end"), &[0.875, 5.0 / 6.0]);
        assert_close(&col(&line, "qq_line_y_start"), &[10.0, 1.0]);
        assert_close(&col(&line, "qq_line_y_end"), &[40.0, 3.0]);
    }

    #[test]
    fn qq_groupby_two_fields_computes_per_combination() {
        let batch = keyed_batch(
            &[
                ("h1", vec![Some("p"), Some("p"), Some("p"), Some("s"), Some("s"), Some("s")]),
                ("h2", vec![Some("q"), Some("q"), Some("r"), Some("q"), Some("q"), Some("q")]),
            ],
            vec![2.0, 1.0, 5.0, 9.0, 7.0, 8.0],
        );
        let spec = qq_spec(serde_json::json!({
            "field": "v", "distribution": "uniform", "groupby": ["h1", "h2"],
        }));
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(strs(&out, "h1"), some(&["p", "p", "p", "s", "s", "s"]));
        assert_eq!(strs(&out, "h2"), some(&["q", "q", "r", "q", "q", "q"]));
        assert_eq!(col(&out, "sample"), vec![1.0, 2.0, 5.0, 7.0, 8.0, 9.0]);
        assert_close(
            &col(&out, "theoretical"),
            &[0.25, 0.75, 0.5, 1.0 / 6.0, 0.5, 5.0 / 6.0],
        );
    }

    /// An absent `groupby` and an explicit empty one are the same spec, and
    /// the empty list is not written back to the wire.
    #[test]
    fn qq_empty_groupby_matches_absent_groupby() {
        let batch = keyed_batch(&[("h", vec![Some("x"); 5])], vec![3.0, 1.0, 4.0, 1.0, 5.0]);
        let absent = qq_spec(serde_json::json!({"field": "v"}));
        let empty = qq_spec(serde_json::json!({"field": "v", "groupby": []}));
        assert_eq!(absent, empty);
        assert!(serde_json::to_value(&empty).unwrap().get("groupby").is_none());
        let a = apply(&absent, &batch).unwrap();
        let e = apply(&empty, &batch).unwrap();
        assert_eq!(a, e);
        let names: Vec<String> = e.schema().fields().iter().map(|f| f.name().clone()).collect();
        assert_eq!(names, ["theoretical", "sample"]);
        assert_eq!(secondary_outputs(&absent, &a).unwrap(), secondary_outputs(&empty, &e).unwrap());
    }

    /// A one-row group plots its single point at the median position with a
    /// flat reference line through it; a group with no plottable value emits
    /// no points and no line; neither disturbs its siblings.
    #[test]
    fn qq_groupby_small_groups_compute_their_own_statistic() {
        let batch = keyed_batch(
            &[("h", vec![Some("x"), Some("y"), Some("z"), Some("z"), Some("x"), Some("x")])],
            vec![1.0, 5.0, f64::NAN, f64::NAN, 2.0, 3.0],
        );
        let spec = qq_spec(serde_json::json!({
            "field": "v", "distribution": "uniform", "groupby": ["h"],
        }));
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(strs(&out, "h"), some(&["x", "x", "x", "y"]));
        assert_close(&col(&out, "theoretical"), &[1.0 / 6.0, 0.5, 5.0 / 6.0, 0.5]);
        let line = line_of(&spec, &out);
        assert_eq!(strs(&line, "h"), some(&["x", "y"]));
        assert_close(&col(&line, "qq_line_x_start"), &[1.0 / 6.0, 0.5]);
        assert_close(&col(&line, "qq_line_y_start"), &[1.0, 5.0]);
        assert_close(&col(&line, "qq_line_y_end"), &[3.0, 5.0]);
    }

    fn apply_warned(spec: &QQSpec, batch: &RecordBatch) -> (PyResult<RecordBatch>, Vec<String>) {
        let ctx = TransformContext::default();
        let out = apply_with_context(spec, batch, &ctx);
        let warnings = ctx.warnings.borrow().iter().map(|w| w.to_string()).collect();
        (out, warnings)
    }

    /// A null groupby key belongs to no group: its rows are dropped from the
    /// points and from the reference line, with one warning naming the field
    /// and the count.
    #[test]
    fn qq_groupby_null_key_rows_are_dropped_with_a_warning() {
        let batch = keyed_batch(
            &[("h", vec![Some("x"), None, Some("x"), None])],
            vec![1.0, 100.0, 2.0, 200.0],
        );
        let spec = qq_spec(serde_json::json!({
            "field": "v", "distribution": "uniform", "groupby": ["h"],
        }));
        let (out, warnings) = apply_warned(&spec, &batch);
        let out = out.unwrap();
        assert_eq!(strs(&out, "h"), some(&["x", "x"]));
        assert_eq!(col(&out, "sample"), vec![1.0, 2.0]);
        assert_close(&col(&out, "theoretical"), &[0.25, 0.75]);
        assert_eq!(
            warnings,
            ["QQ: 2 rows have a null 'h' and belong to no group, so they were dropped; \
              fill or filter the nulls in 'h'"]
        );
        let line = line_of(&spec, &out);
        assert_eq!(strs(&line, "h"), some(&["x"]));
        assert_close(&col(&line, "qq_line_y_start"), &[1.0]);
        assert_close(&col(&line, "qq_line_y_end"), &[2.0]);
    }

    /// Every key null: nothing is left to plot, so the call refuses with the
    /// transform's empty-input error.
    #[test]
    fn qq_groupby_all_null_keys_refuse() {
        pyo3::Python::initialize();
        let batch = keyed_batch(&[("h", vec![None, None])], vec![1.0, 2.0]);
        let spec = qq_spec(serde_json::json!({"field": "v", "groupby": ["h"]}));
        let (out, _) = apply_warned(&spec, &batch);
        assert_eq!(out.unwrap_err().to_string(), "ValueError: stat_qq: empty input after filter");
    }

    /// The only plottable value sits on a null-keyed row: once that row is
    /// dropped nothing is left, which refuses rather than returning no
    /// points silently.
    #[test]
    fn qq_groupby_plottable_values_only_on_null_keys_refuse() {
        pyo3::Python::initialize();
        let batch = keyed_batch(&[("h", vec![Some("x"), None])], vec![f64::NAN, 5.0]);
        let spec = qq_spec(serde_json::json!({"field": "v", "groupby": ["h"]}));
        let (out, warnings) = apply_warned(&spec, &batch);
        assert_eq!(out.unwrap_err().to_string(), "ValueError: stat_qq: empty input after filter");
        assert_eq!(warnings.len(), 1, "got {warnings:?}");
    }

    /// A grouped call that drops nothing refuses an all-NaN field exactly as
    /// the ungrouped call does, with the same error and no drop warning.
    #[test]
    fn qq_groupby_without_null_keys_refuses_all_nan_like_ungrouped() {
        pyo3::Python::initialize();
        let batch = keyed_batch(&[("h", vec![Some("x"), Some("y")])], vec![f64::NAN, f64::NAN]);
        let grouped = qq_spec(serde_json::json!({"field": "v", "groupby": ["h"]}));
        let ungrouped = qq_spec(serde_json::json!({"field": "v"}));
        let (out, warnings) = apply_warned(&grouped, &batch);
        assert_eq!(
            out.unwrap_err().to_string(),
            apply(&ungrouped, &batch).unwrap_err().to_string()
        );
        assert!(warnings.is_empty(), "got {warnings:?}");
    }

    /// A non-text groupby key keeps its input type on the points and the
    /// reference line.
    #[test]
    fn qq_groupby_int_key_keeps_its_type() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("v", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(arrow::array::Int64Array::from(vec![7, 3, 7, 3])) as ArrayRef,
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
        ]).unwrap();
        let spec = qq_spec(serde_json::json!({"field": "v", "groupby": ["k"]}));
        let out = apply(&spec, &batch).unwrap();
        let line = line_of(&spec, &out);
        for b in [&out, &line] {
            let k = b.column_by_name("k").expect("the key column is carried");
            assert_eq!(k.data_type(), &DataType::Int64);
        }
        let k = out.column_by_name("k").unwrap().as_any().downcast_ref::<arrow::array::Int64Array>().unwrap();
        assert_eq!(k.values().to_vec(), vec![7, 7, 3, 3]);
    }

    /// Grouped dequantize jitters each group exactly as a call over that
    /// group's rows alone would (each group draws from a generator seeded
    /// with `seed`), and repeats byte-for-byte.
    #[test]
    fn qq_groupby_dequantize_is_deterministic_per_group() {
        let batch = keyed_batch(
            &[("h", vec![Some("x"), Some("x"), Some("x"), Some("x"), Some("y"), Some("y"), Some("y")])],
            vec![1.0, 1.0, 1.0, 2.0, 5.0, 5.0, 6.0],
        );
        let spec = qq_spec(serde_json::json!({
            "field": "v", "distribution": "uniform", "groupby": ["h"],
            "dequantize": true, "seed": 7,
        }));
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out, apply(&spec, &batch).unwrap());
        let alone = |values: Vec<f64>| {
            let one = batch_field("v", values);
            let ungrouped = qq_spec(serde_json::json!({
                "field": "v", "distribution": "uniform", "dequantize": true, "seed": 7,
            }));
            col(&apply(&ungrouped, &one).unwrap(), "theoretical")
        };
        let theo = col(&out, "theoretical");
        assert_eq!(theo[..4], alone(vec![1.0, 1.0, 1.0, 2.0])[..]);
        assert_eq!(theo[4..], alone(vec![5.0, 5.0, 6.0])[..]);
        assert_ne!(theo[..3], [0.125, 0.375, 0.625], "the tied run is jittered");
    }

    #[test]
    fn qq_groupby_missing_column_refuses() {
        pyo3::Python::initialize();
        let batch = keyed_batch(&[("h", vec![Some("x")])], vec![1.0]);
        let spec = qq_spec(serde_json::json!({"field": "v", "groupby": ["ghost"]}));
        let err = apply(&spec, &batch).unwrap_err().to_string();
        assert!(err.contains("stat_qq: groupby column 'ghost' not found"), "{err}");
    }

    /// A groupby column named like one of this transform's output columns
    /// (points or reference line) would make every reader of that column
    /// read the wrong one.
    #[test]
    fn qq_groupby_output_collision_refuses() {
        pyo3::Python::initialize();
        for taken in ["theoretical", "sample", "qq_line_x_start"] {
            let batch = keyed_batch(&[(taken, vec![Some("x")])], vec![1.0]);
            let spec = qq_spec(serde_json::json!({"field": "v", "groupby": [taken]}));
            let err = apply(&spec, &batch).unwrap_err().to_string();
            assert!(err.contains("collides"), "{taken}: {err}");
        }
    }

    #[test]
    fn qq_constructor_takes_groupby_and_refuses_a_repeat() {
        pyo3::Python::initialize();
        let ok = PyQq::new("v", "normal", false, true, 0, vec!["h".into()], None)
            .expect("a groupby list constructs");
        assert!(matches!(&ok.0, TransformSpec::Qq(s) if s.groupby == ["h"]));
        let err = PyQq::new("v", "normal", false, true, 0, vec!["h".into(), "h".into()], None)
            .err()
            .expect("a repeated groupby name refuses")
            .to_string();
        assert!(err.contains("QQ: groupby column 'h' is repeated"), "{err}");
    }
}
