//! QQ (quantile-quantile) transform.
//!
//! Primary output: per-row points with `theoretical: f64, sample: f64` over
//! the non-null/non-NaN values of `field`, sorted ascending by sample value.
//!
//! Secondary output (`qq_line`, when `emit_line=true`): a single-row batch with
//! columns `qq_line_x_start, qq_line_x_end, qq_line_y_start, qq_line_y_end`
//! describing a robust quartile-fit reference line.

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

fn default_distribution() -> String {
    "normal".into()
}
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

fn theoretical_quantile(p: f64, distribution: &str) -> PyResult<f64> {
    match distribution {
        "normal" => Ok(inverse_normal_cdf(p)),
        "uniform" => Ok(p),
        "exponential" => Ok(-((1.0 - p).ln())),
        other => Err(PyValueError::new_err(format!(
            "stat_qq: unknown distribution '{other}'; expected normal|uniform|exponential"
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
                jittered.sort_by(|a, b| a.partial_cmp(b).unwrap());
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
    let schema = batch.schema();
    let v_idx = schema
        .index_of(&spec.field)
        .map_err(|_| PyValueError::new_err(format!(
            "stat_qq: column '{}' not found",
            spec.field
        )))?;
    if schema.field(v_idx).data_type() != &DataType::Float64 {
        return Err(PyValueError::new_err(format!(
            "stat_qq: column '{}' must be Float64",
            spec.field
        )));
    }
    let v_arr = batch
        .column(v_idx)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();

    // Filter null/NaN sample values.
    let mut samples: Vec<f64> = (0..v_arr.len())
        .filter_map(|i| {
            if v_arr.is_null(i) {
                return None;
            }
            let v = v_arr.value(i);
            if v.is_nan() {
                None
            } else {
                Some(v)
            }
        })
        .collect();

    if samples.is_empty() {
        return Err(PyValueError::new_err("stat_qq: empty input after filter"));
    }

    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let theoretical =
        compute_theoretical(&samples, &spec.distribution, spec.dequantize, spec.seed)?;

    let out_schema = Arc::new(Schema::new(vec![
        Field::new("theoretical", DataType::Float64, false),
        Field::new("sample", DataType::Float64, false),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(theoretical)),
        Arc::new(Float64Array::from(samples)),
    ];
    RecordBatch::try_new(out_schema, cols)
        .map_err(|e| PyValueError::new_err(format!("stat_qq: {e}")))
}

/// Compute the secondary `qq_line` named output from the primary points batch.
///
/// Returns `vec![("qq_line", line_batch)]` when `emit_line=true`, else empty.
/// The line is a robust quartile fit:
///   slope     = (sample_q3 - sample_q1) / (theo_q3 - theo_q1)
///   intercept = sample_q2 - slope * theo_q2
/// Endpoints span the theoretical-x range.
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

    let n = theo.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let theo_vals: Vec<f64> = (0..n).map(|i| theo.value(i)).collect();
    let samp_vals: Vec<f64> = (0..n).map(|i| samp.value(i)).collect();

    // Filter to finite values only (Beasley-Springer-Moro maps p=0/1 → ±inf,
    // which won't occur for plotting positions but is defensive here).
    let pairs: Vec<(f64, f64)> = theo_vals
        .iter()
        .zip(samp_vals.iter())
        .filter(|(t, _)| t.is_finite())
        .map(|(t, s)| (*t, *s))
        .collect();
    if pairs.is_empty() {
        return Ok(Vec::new());
    }

    // The primary batch is sorted by sample (ascending). For the quartile fit
    // we need quantiles of theoretical (also ascending after sorting samples,
    // because plotting positions are increasing and the inverse CDF is
    // monotone) AND of sample values. They're already aligned.
    let theo_q1 = quantile_sorted(&pairs.iter().map(|(t, _)| *t).collect::<Vec<_>>(), 0.25);
    let theo_q2 = quantile_sorted(&pairs.iter().map(|(t, _)| *t).collect::<Vec<_>>(), 0.5);
    let theo_q3 = quantile_sorted(&pairs.iter().map(|(t, _)| *t).collect::<Vec<_>>(), 0.75);
    let samp_q1 = quantile_sorted(&pairs.iter().map(|(_, s)| *s).collect::<Vec<_>>(), 0.25);
    let samp_q2 = quantile_sorted(&pairs.iter().map(|(_, s)| *s).collect::<Vec<_>>(), 0.5);
    let samp_q3 = quantile_sorted(&pairs.iter().map(|(_, s)| *s).collect::<Vec<_>>(), 0.75);

    let denom = theo_q3 - theo_q1;
    let slope = if denom == 0.0 {
        0.0
    } else {
        (samp_q3 - samp_q1) / denom
    };
    let intercept = samp_q2 - slope * theo_q2;

    let theo_min = pairs
        .iter()
        .map(|(t, _)| *t)
        .fold(f64::INFINITY, f64::min);
    let theo_max = pairs
        .iter()
        .map(|(t, _)| *t)
        .fold(f64::NEG_INFINITY, f64::max);
    let y_min = slope * theo_min + intercept;
    let y_max = slope * theo_max + intercept;

    let line_schema = Arc::new(Schema::new(vec![
        Field::new("qq_line_x_start", DataType::Float64, false),
        Field::new("qq_line_x_end", DataType::Float64, false),
        Field::new("qq_line_y_start", DataType::Float64, false),
        Field::new("qq_line_y_end", DataType::Float64, false),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(vec![theo_min])),
        Arc::new(Float64Array::from(vec![theo_max])),
        Arc::new(Float64Array::from(vec![y_min])),
        Arc::new(Float64Array::from(vec![y_max])),
    ];
    let line_batch = RecordBatch::try_new(line_schema, cols)
        .map_err(|e| PyValueError::new_err(format!("stat_qq: qq_line: {e}")))?;

    Ok(vec![("qq_line".to_string(), line_batch)])
}

/// Linearly-interpolated quantile over an already-sorted (ascending) sequence.
fn quantile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let h = p * (n as f64 - 1.0);
    let lo = h.floor() as usize;
    let hi = (h.ceil() as usize).min(n - 1);
    let frac = h - h.floor();
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

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
/// Output columns: ``sample`` (Float64 empirical quantiles), ``theoretical``
/// (Float64 reference quantiles). When ``emit_line=True``, also emits two
/// rows with the reference-line endpoints in additional columns
/// ``line_x`` and ``line_y``.
///
/// Parameters
/// ----------
/// field : str
///     Numeric column to plot (must be Float64).
/// distribution : {"normal", "uniform", "exponential"}, default "normal"
///     Reference distribution for theoretical quantiles.
/// dequantize : bool, default False
///     When True, add small seeded random jitter to break discrete ties.
/// emit_line : bool, default True
///     When True, append the two reference-line endpoints.
/// seed : int, default 0
///     RNG seed for dequantization jitter. Seeds ``ChaCha8Rng`` for
///     byte-deterministic output across platforms.
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
        name = None,
    ))]
    fn new(
        field: &str,
        distribution: &str,
        dequantize: bool,
        emit_line: bool,
        seed: u64,
        name: Option<String>,
    ) -> PyResult<Self> {
        if field.is_empty() {
            return Err(PyValueError::new_err("QQ: field must be non-empty"));
        }
        match distribution {
            "normal" | "uniform" | "exponential" => {}
            other => {
                return Err(PyValueError::new_err(format!(
                    "QQ: unknown distribution '{other}'; expected normal|uniform|exponential"
                )))
            }
        }
        Ok(PyQq(TransformSpec::Qq(QQSpec {
            field: field.to_string(),
            distribution: distribution.to_string(),
            dequantize,
            emit_line,
            seed,
            name,
        })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Qq(s) => format!(
                "QQ(field='{}', distribution='{}', dequantize={}, emit_line={}, seed={})",
                s.field, s.distribution, s.dequantize, s.emit_line, s.seed,
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

    #[test]
    fn qq_uniform_and_exponential_distributions() {
        // Sample = [1, 2, 3, 4]; sorted; n=4 → p = 0.125, 0.375, 0.625, 0.875.
        let batch = batch_field("v", vec![1.0, 2.0, 3.0, 4.0]);
        let p = vec![0.125, 0.375, 0.625, 0.875];

        // Uniform: theoretical = p.
        let spec_u = QQSpec {
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

    #[test]
    fn qq_dequantize_jitters_only_ties() {
        // 2 ties at 5.0 + 3 unique values. With dequantize=false, theoretical
        // positions for the unique rows are exactly (i - 0.5) / n (under
        // uniform). With dequantize=true + seed=42, the 2 tied rows' theoretical
        // values shift while unique rows are unchanged. Same seed twice = bit-
        // identical output (deterministic).
        let batch = batch_field("v", vec![5.0, 5.0, 1.0, 2.0, 8.0]);
        let spec_no_jitter = QQSpec {
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
}
