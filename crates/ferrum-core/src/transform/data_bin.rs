//! Data transform: DataBin — binning as a data transform.
//!
//! Adds a bin column to the input batch (unlike stat_bin which collapses
//! to bin counts). Output is the original batch plus a "{field}_bin" column
//! containing the bin start value for each row.

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct DataBinSpec {
    pub field: String,
    /// Output column name. Defaults to "{field}_bin".
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub as_: Option<String>,
    /// Maximum number of bins.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub maxbins: Option<usize>,
    /// Explicit bin width (overrides maxbins).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub step: Option<f64>,
    /// Whether to "nice" the bin boundaries.
    #[serde(default = "default_nice")]
    pub nice: bool,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
    /// Internal facet-extent pin. NOT a user kwarg. Set by
    /// `fix_transform_extents_for_facet` to provide the global `(data_min,
    /// data_max)` range (RAW, not niced). When set, `apply` uses it instead of
    /// the per-partition fold, so the niced bin boundaries and bin_width derive
    /// from the same range in every panel. Skipped in serialization when None
    /// so non-faceted JSON output is byte-identical to before.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub extent: Option<(f64, f64)>,
}

fn default_nice() -> bool {
    true
}

pub(crate) fn apply(spec: &DataBinSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let schema = batch.schema();

    let col_idx = schema.index_of(&spec.field).map_err(|_| {
        PyValueError::new_err(format!("data_bin: column '{}' not found", spec.field))
    })?;
    let col = batch
        .column(col_idx)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| {
            PyValueError::new_err(format!("data_bin: column '{}' must be Float64", spec.field))
        })?;

    let n_rows = batch.num_rows();

    // Compute data extent.
    // When an internal facet-extent pin is set (`spec.extent`), use it as the
    // `(data_min, data_max)` range instead of the per-partition fold, so the
    // niced bin boundaries and bin_width derive from the same global range in
    // every facet panel — identical bin edges across panels.
    let (data_min, data_max) = match spec.extent {
        Some((lo, hi)) => (lo, hi),
        None => {
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            for i in 0..n_rows {
                if col.is_null(i) { continue; }
                let v = col.value(i);
                if v.is_nan() { continue; }
                if v < lo { lo = v; }
                if v > hi { hi = v; }
            }
            (lo, hi)
        }
    };

    if data_min > data_max {
        // All null/NaN — produce NaN bin column.
        let bin_values = vec![f64::NAN; n_rows];
        return build_output(spec, batch, bin_values);
    }

    // Determine bin width.
    let bin_width = if let Some(step) = spec.step {
        step
    } else {
        let maxbins = spec.maxbins.unwrap_or(10);
        let raw_width = (data_max - data_min) / maxbins as f64;
        if spec.nice {
            nice_step(raw_width)
        } else {
            raw_width
        }
    };

    if bin_width <= 0.0 || !bin_width.is_finite() {
        let bin_values = vec![data_min; n_rows];
        return build_output(spec, batch, bin_values);
    }

    // Compute bin start for the extent.
    let bin_start = if spec.nice {
        (data_min / bin_width).floor() * bin_width
    } else {
        data_min
    };

    // Assign each row to a bin.
    let mut bin_values: Vec<f64> = Vec::with_capacity(n_rows);
    for i in 0..n_rows {
        if col.is_null(i) {
            bin_values.push(f64::NAN);
        } else {
            let v = col.value(i);
            if v.is_nan() {
                bin_values.push(f64::NAN);
            } else {
                let bin_idx = ((v - bin_start) / bin_width).floor();
                bin_values.push(bin_start + bin_idx * bin_width);
            }
        }
    }

    build_output(spec, batch, bin_values)
}

/// Compute the global `(data_min, data_max)` extent of `spec.field` over the
/// full `batch`. Used by `fix_transform_extents_for_facet` to pin every facet
/// panel to the same raw range so the niced bin boundaries and bin_width derive
/// identically in every panel.
///
/// Returns the RAW `(min, max)` — NOT niced. Nicing happens inside `apply`
/// from the pinned range, guaranteeing every panel produces the same
/// bin_width and the same niced bin_start. Returns `None` when the field is
/// missing, non-numeric, all values are null/NaN, or the range is degenerate.
///
/// When `spec.extent` is already set, it is returned unchanged.
pub(crate) fn global_extent(
    spec: &DataBinSpec,
    batch: &RecordBatch,
) -> Option<(f64, f64)> {
    // An explicit extent always wins. Otherwise return the RAW global range
    // (NICENESS CONTRACT, XFORM-08: DataBin nices inside `apply` from the pinned
    // range, so every panel derives identical edges from the raw extent — unlike
    // 1-D `Bin::global_extent`, which nices the pin itself).
    if let Some(e) = spec.extent {
        return Some(e);
    }
    crate::transform::numeric_util::column_extent(batch, &spec.field)
}

fn build_output(spec: &DataBinSpec, batch: &RecordBatch, bin_values: Vec<f64>) -> PyResult<RecordBatch> {
    let schema = batch.schema();
    let out_name = spec
        .as_
        .clone()
        .unwrap_or_else(|| format!("{}_bin", spec.field));

    let mut fields: Vec<Field> = schema.fields().iter().map(|f| f.as_ref().clone()).collect();
    fields.push(Field::new(&out_name, DataType::Float64, true));
    let out_schema = Arc::new(Schema::new(fields));

    let mut columns: Vec<ArrayRef> = (0..batch.num_columns())
        .map(|i| batch.column(i).clone())
        .collect();
    columns.push(Arc::new(Float64Array::from(bin_values)));

    RecordBatch::try_new(out_schema, columns)
        .map_err(|e| PyValueError::new_err(format!("data_bin: {e}")))
}

/// Round a raw step size to a "nice" number (1, 2, 5 × 10^k).
fn nice_step(raw: f64) -> f64 {
    if raw <= 0.0 || !raw.is_finite() {
        return 1.0;
    }
    let exp = raw.log10().floor();
    let frac = raw / 10.0_f64.powf(exp);
    let nice_frac = if frac <= 1.5 {
        1.0
    } else if frac <= 3.0 {
        2.0
    } else if frac <= 7.0 {
        5.0
    } else {
        10.0
    };
    nice_frac * 10.0_f64.powf(exp)
}

// No PyO3 wrapper: `DataBin` is constructed only via the dict-emitting
// `transform_bin` Python function and carried through the `transforms_json`
// serde path (SEAM-02). The removed `#[new]` performed no validation beyond
// serde's required `field`. `DataBinSpec` above is the serde target.

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float64Array;
    use arrow::datatypes::{Field, Schema};
    use std::sync::Arc;

    #[test]
    fn data_bin_assigns_bins() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![
                0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,
            ]))],
        )
        .unwrap();

        let spec = DataBinSpec {
            field: "x".into(),
            as_: Some("x_bin".into()),
            maxbins: Some(5),
            step: None,
            nice: true,
            name: None,
            extent: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_columns(), 2);
        assert_eq!(out.schema().field(1).name(), "x_bin");
        assert_eq!(out.num_rows(), 10);

        // All bin values should be finite.
        let bins = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        for i in 0..bins.len() {
            assert!(bins.value(i).is_finite());
        }
    }

    #[test]
    fn data_bin_explicit_step() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]))],
        )
        .unwrap();

        let spec = DataBinSpec {
            field: "x".into(),
            as_: None,
            maxbins: None,
            step: Some(2.0),
            nice: false,
            name: None,
            extent: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let bins = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        // With step=2 starting at 1.0: bins at 1.0, 1.0, 3.0, 3.0, 5.0
        assert_eq!(bins.value(0), 1.0);
        assert_eq!(bins.value(1), 1.0);
        assert_eq!(bins.value(2), 3.0);
        assert_eq!(bins.value(3), 3.0);
        assert_eq!(bins.value(4), 5.0);
    }

    // ── Regression tests for the faceted extent pin (defect class #7) ────────

    fn make_f64_batch(xs: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(xs))],
        )
        .unwrap()
    }

    #[test]
    fn data_bin_global_extent_returns_raw_range() {
        let b = make_f64_batch(vec![1.0, 2.5, 5.0, 7.5, 10.0]);
        let spec = DataBinSpec {
            field: "x".into(),
            as_: None, maxbins: Some(5), step: None, nice: true, name: None,
            extent: None,
        };
        let ext = global_extent(&spec, &b).expect("should have extent");
        assert!((ext.0 - 1.0).abs() < 1e-12, "lo={}", ext.0);
        assert!((ext.1 - 10.0).abs() < 1e-12, "hi={}", ext.1);
    }

    #[test]
    fn data_bin_global_extent_preserves_existing_extent() {
        let b = make_f64_batch(vec![1.0, 10.0]);
        let spec = DataBinSpec {
            field: "x".into(),
            as_: None, maxbins: None, step: None, nice: true, name: None,
            extent: Some((0.0, 20.0)),
        };
        let ext = global_extent(&spec, &b).expect("should have extent");
        assert!((ext.0 - 0.0).abs() < 1e-12, "lo should be user-set 0.0, got {}", ext.0);
        assert!((ext.1 - 20.0).abs() < 1e-12, "hi should be user-set 20.0, got {}", ext.1);
    }

    #[test]
    fn data_bin_pinned_extent_gives_identical_bin_boundaries_for_disjoint_partitions() {
        // Partition A: x in [0, 1]. Partition B: x in [5, 10].
        // Without pin: different bin_width → different bin edges.
        // With pin [0, 10] and maxbins=10 → raw_width=(10-0)/10=1 → nice=1 → same bins.
        let batch_a = make_f64_batch(vec![0.1, 0.5, 0.9]);
        let batch_b = make_f64_batch(vec![5.1, 7.5, 9.9]);

        let spec_unpinned = DataBinSpec {
            field: "x".into(),
            as_: Some("xb".into()), maxbins: Some(10), step: None, nice: true, name: None,
            extent: None,
        };

        let out_a_u = apply(&spec_unpinned, &batch_a).unwrap();
        let out_b_u = apply(&spec_unpinned, &batch_b).unwrap();

        let get_bins = |out: &RecordBatch| -> Vec<f64> {
            let col = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
            (0..col.len()).map(|i| col.value(i)).collect()
        };

        let bins_a_u = get_bins(&out_a_u);
        let bins_b_u = get_bins(&out_b_u);

        // Without pin: A bins are around [0,1], B bins are around [5,10] → not comparable.
        // The minimum bin value in A should be near 0, in B near 5.
        let min_a_u = bins_a_u.iter().cloned().fold(f64::INFINITY, f64::min);
        let min_b_u = bins_b_u.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            (min_b_u - min_a_u).abs() > 3.0,
            "SETUP CHECK: unpinned bin starts must differ (a_min={min_a_u}, b_min={min_b_u})"
        );

        // With pinned extent [0, 10]:
        let spec_pinned = DataBinSpec {
            field: "x".into(),
            as_: Some("xb".into()), maxbins: Some(10), step: None, nice: true, name: None,
            extent: Some((0.0, 10.0)),
        };

        let out_a_p = apply(&spec_pinned, &batch_a).unwrap();
        let out_b_p = apply(&spec_pinned, &batch_b).unwrap();
        let bins_a_p = get_bins(&out_a_p);
        let bins_b_p = get_bins(&out_b_p);

        // With the same raw range [0,10]: raw_width=1 → nice=1 → bin_start=0.
        // A: values [0.1,0.5,0.9] → all in bin starting at 0.0.
        for v in &bins_a_p {
            assert!((v - 0.0).abs() < 1e-12, "A pinned: expected bin=0.0, got {v}");
        }
        // B: values [5.1,7.5,9.9] → bins 5.0, 7.0, 9.0.
        let expected_b: Vec<f64> = vec![5.0, 7.0, 9.0];
        for (v, e) in bins_b_p.iter().zip(expected_b.iter()) {
            assert!((v - e).abs() < 1e-9, "B pinned: expected bin={e}, got {v}");
        }
        // The bin step (difference between bin values) must be identical for both:
        // both derive bin_width=1.0 (nice step from raw_width=1.0 with maxbins=10).
        // A has only 1 distinct bin, B has 3 with step 2.0 (7-5=2 → actual step per-row).
        // More precisely: raw_width = (10-0)/10 = 1.0, nice_step(1.0) = 1.0.
        // A rows all land in [0,1) → bin=0.0 (step=1).
        // B rows: 5.1→5.0, 7.5→7.0, 9.9→9.0 (each in different step-1 bins).
        let step_b = bins_b_p[1] - bins_b_p[0]; // 7.0 - 5.0 = 2.0? No, 5.1→5.0, 7.5→7.0
        // Confirm step between distinct bin values = 1.0 × floor difference.
        // The bin_width=1.0 is what matters; individual rows differ by their data gap.
        // Instead confirm: all bin values are multiples of 1.0.
        for v in &bins_b_p {
            let rem = (v / 1.0 - (v / 1.0).round()).abs();
            assert!(rem < 1e-9, "B pinned: bin {v} not a multiple of bin_width=1.0");
        }
        // And A bins too:
        for v in &bins_a_p {
            let rem = (v / 1.0 - (v / 1.0).round()).abs();
            assert!(rem < 1e-9, "A pinned: bin {v} not a multiple of bin_width=1.0");
        }
        let _ = step_b; // suppress unused warning
    }

    #[test]
    fn data_bin_non_faceted_spec_serializes_without_extent_field() {
        let spec = DataBinSpec {
            field: "x".into(),
            as_: None, maxbins: None, step: None, nice: true, name: None,
            extent: None,
        };
        let json = serde_json::to_string(&spec).unwrap();
        assert!(!json.contains("extent"), "extent must not appear when None: {json}");
    }
}
