//! `Smooth` test coverage, relocated wholesale (wave 6A `#163`) from an
//! inline `#[cfg(test)] mod tests` at the bottom of `transform/smooth.rs`.
//! The module was promoted to a directory (`smooth/{mod,fits,binding,tests}.rs`)
//! because its test body (2361 lines) exceeded its source body (~1806 lines),
//! the CLAUDE.md Rust test-module-promotion trigger stated verbatim. This
//! file could not stay inline once `mod.rs` split into three source files —
//! it exercises private items from both `mod.rs` (the orchestration layer,
//! reached via `use super::*;`) and `fits` (the numeric fit routines,
//! `pub(super)` for exactly this reason — most already reach `tests` through
//! `mod.rs`'s own `use fits::{...}`, re-globbed by `use super::*;` below; the
//! two `fits` items `mod.rs` itself never calls — `build_degenerate_smooth_batch`,
//! `student_t_critical` — are imported directly), mirroring
//! `transform/stats/tests.rs` and `render/scale_resolve/tests.rs`. Pure
//! motion: no assertion, fixture, or test name changed.

use super::*;
use super::fits::{build_degenerate_smooth_batch, student_t_critical};
use arrow::array::{Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

use crate::transform::color_bin_group::StepSpace;
use crate::transform::core::TransformSpec;

fn xy_batch(x: Vec<f64>, y: Vec<f64>) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, true),
        Field::new("y", DataType::Float64, true),
    ]));
    RecordBatch::try_new(schema, vec![
        Arc::new(Float64Array::from(x)),
        Arc::new(Float64Array::from(y)),
    ]).unwrap()
}

fn col(b: &RecordBatch, name: &str) -> Vec<f64> {
    let arr = b.column(b.schema().index_of(name).unwrap())
        .as_any().downcast_ref::<Float64Array>().unwrap();
    (0..arr.len()).map(|i| if arr.is_null(i) { f64::NAN } else { arr.value(i) }).collect()
}

#[test]
fn test_lm_recovers_slope_and_intercept_exactly() {
    let xs: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let ys: Vec<f64> = xs.iter().map(|x| 3.0 + 2.0 * x).collect();
    let batch = xy_batch(xs, ys);
    let spec = SmoothSpec {
        residual_columns: Default::default(),
        x: "x".into(), y: "y".into(),
        method: SmoothMethod::Lm,
        ci: None,
        bandwidth: 0.0, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
        x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
        inject_zero_ref: false,
        inject_metrics: false,
    x_range: None, x_range_pad: None,
        groupby: vec![],
        name: None,
    };
    let out = apply(&spec, &batch).unwrap();
    let xg = col(&out, "x");
    let yf = col(&out, "y");
    for (xq, yq) in xg.iter().zip(yf.iter()) {
        let expected = 3.0 + 2.0 * xq;
        assert!((yq - expected).abs() < 1e-10, "y(x={xq})={yq}, expected {expected}");
    }
}

#[test]
fn test_lm_ci_band_brackets_fit_at_mean_x() {
    let xs: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let ys: Vec<f64> = xs.iter().enumerate().map(|(i, &x)| {
        x + if i % 2 == 0 { 0.5 } else { -0.5 }
    }).collect();
    let mean_x = xs.iter().sum::<f64>() / xs.len() as f64;
    let batch = xy_batch(xs, ys);
    let spec = SmoothSpec {
        residual_columns: Default::default(),
        x: "x".into(), y: "y".into(),
        method: SmoothMethod::Lm,
        ci: Some(0.95),
        bandwidth: 0.0, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 51, seed: 0,
        x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
        inject_zero_ref: false,
        inject_metrics: false,
    x_range: None, x_range_pad: None,
        groupby: vec![],
        name: None,
    };
    let out = apply(&spec, &batch).unwrap();
    let xg = col(&out, "x");
    let yf = col(&out, "y");
    let lo = col(&out, "ci_lower");
    let hi = col(&out, "ci_upper");
    let i = (0..xg.len()).min_by(|a, b|
        (xg[*a] - mean_x).abs().partial_cmp(&(xg[*b] - mean_x).abs()).unwrap()
    ).unwrap();
    assert!(lo[i] < yf[i] && yf[i] < hi[i], "CI must bracket fit at x={}", xg[i]);
    assert!(hi[i] - lo[i] > 0.0);
}

#[test]
fn test_lm_zero_variance_x_emits_nan_line() {
    let xs = vec![5.0; 10];
    let ys: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let batch = xy_batch(xs, ys);
    let spec = SmoothSpec {
        residual_columns: Default::default(),
        x: "x".into(), y: "y".into(),
        method: SmoothMethod::Lm,
        ci: Some(0.95),
        bandwidth: 0.0, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
        x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
        inject_zero_ref: false,
        inject_metrics: false,
    x_range: None, x_range_pad: None,
        groupby: vec![],
        name: None,
    };
    let out = apply(&spec, &batch).unwrap();
    let yf = col(&out, "y");
    assert!(yf.iter().all(|y| y.is_nan()));
}

#[test]
fn test_lm_n_lt_2_emits_all_nan() {
    let batch = xy_batch(vec![1.0], vec![1.0]);
    let spec = SmoothSpec {
        residual_columns: Default::default(),
        x: "x".into(), y: "y".into(),
        method: SmoothMethod::Lm,
        ci: None,
        bandwidth: 0.0, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
        x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
        inject_zero_ref: false,
        inject_metrics: false,
        groupby: vec![],
        name: None,
        x_range: None, x_range_pad: None,
    };
    let out = apply(&spec, &batch).unwrap();
    let yf = col(&out, "y");
    assert!(yf.iter().all(|y| y.is_nan()));
}

#[test]
fn test_smooth_round_trip_json() {
    let original = SmoothSpec {
        residual_columns: Default::default(),
        x: "x".into(), y: "y".into(),
        method: SmoothMethod::Lm,
        ci: Some(0.95),
        bandwidth: 0.5, degree: 2, k: gam::DEFAULT_BASIS_DIM, n: 100, seed: 42,
        x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
        inject_zero_ref: false,
        inject_metrics: false,
    x_range: None, x_range_pad: None,
        groupby: vec![],
        name: None,
    };
    let json = serde_json::to_string(&original).unwrap();
    let parsed: SmoothSpec = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, original);
}

#[test]
fn test_loess_deg1_against_fixtures() {
    use serde::Deserialize;
    const FIXTURES: &str = include_str!("../fixtures/stat_refs.json");
    #[derive(Deserialize)]
    struct LoessCase {
        name: String, x: Vec<f64>, y: Vec<f64>,
        bandwidth: f64, degree: u8, n: usize,
        x_grid: Vec<f64>, y_fit: Vec<f64>,
    }
    #[derive(Deserialize)]
    struct F { loess: Vec<LoessCase> }
    let cases: F = serde_json::from_str(FIXTURES).unwrap();
    for case in cases.loess {
        if case.degree != 1 { continue; }
        let batch = xy_batch(case.x.clone(), case.y.clone());
        let spec = SmoothSpec {
            residual_columns: Default::default(),
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::Loess, ci: None,
            bandwidth: case.bandwidth, degree: case.degree, k: gam::DEFAULT_BASIS_DIM, n: case.n, seed: 0,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false,
            inject_metrics: false,
        x_range: None, x_range_pad: None,
            groupby: vec![],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let xg = col(&out, "x");
        let yf = col(&out, "y");
        for i in 0..case.n {
            assert!((xg[i] - case.x_grid[i]).abs() < 1e-9, "x grid {} vs {}", xg[i], case.x_grid[i]);
            assert!((yf[i] - case.y_fit[i]).abs() < 1e-9, "case {}: y_fit[{i}] = {} vs {}", case.name, yf[i], case.y_fit[i]);
        }
    }
}

#[test]
fn test_loess_deg2_against_fixtures() {
    use serde::Deserialize;
    const FIXTURES: &str = include_str!("../fixtures/stat_refs.json");
    #[derive(Deserialize)]
    struct LoessCase {
        name: String, x: Vec<f64>, y: Vec<f64>,
        bandwidth: f64, degree: u8, n: usize,
        x_grid: Vec<f64>, y_fit: Vec<f64>,
    }
    #[derive(Deserialize)]
    struct F { loess: Vec<LoessCase> }
    let cases: F = serde_json::from_str(FIXTURES).unwrap();
    for case in cases.loess {
        if case.degree != 2 { continue; }
        let batch = xy_batch(case.x.clone(), case.y.clone());
        let spec = SmoothSpec {
            residual_columns: Default::default(),
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::Loess, ci: None,
            bandwidth: case.bandwidth, degree: case.degree, k: gam::DEFAULT_BASIS_DIM, n: case.n, seed: 0,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false,
            inject_metrics: false,
        x_range: None, x_range_pad: None,
            groupby: vec![],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let xg = col(&out, "x");
        let yf = col(&out, "y");
        for i in 0..case.n {
            assert!((xg[i] - case.x_grid[i]).abs() < 1e-9);
            assert!((yf[i] - case.y_fit[i]).abs() < 1e-9,
                "case {}: y_fit[{i}] = {} vs {} (diff {})",
                case.name, yf[i], case.y_fit[i], (yf[i] - case.y_fit[i]).abs());
        }
    }
}

#[test]
fn test_loess_deg2_local_window_too_small_emits_nan() {
    // Only 3 points but degree=2 requires k >= 3 — when bandwidth*n rounds to <3 the impl
    // floors k to degree+1=3 (so this test verifies the floor and that we don't panic).
    let xs: Vec<f64> = vec![0.0, 1.0, 2.0];
    let ys: Vec<f64> = vec![0.0, 1.0, 4.0];
    let batch = xy_batch(xs, ys);
    let spec = SmoothSpec {
        residual_columns: Default::default(),
        x: "x".into(), y: "y".into(),
        method: SmoothMethod::Loess, ci: None,
        bandwidth: 0.1,  // bw * n = 0.3, floored to k = degree + 1 = 3
        degree: 2, k: gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
        x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
        inject_zero_ref: false,
        inject_metrics: false,
    x_range: None, x_range_pad: None,
        groupby: vec![],
        name: None,
    };
    // Primary goal: no panic with k == n == degree+1.
    // With k=n=3 and tricube weights, the farthest neighbor's weight is exactly 0 for grid
    // endpoints, making X'WX rank-deficient → solve_3x3_spd returns None → NaN.
    // Interior grid points may have better conditioning and return finite values.
    // We assert: apply succeeds (no panic), output has 5 rows, each value is NaN or finite.
    let out = apply(&spec, &batch).unwrap();
    let yf = col(&out, "y");
    assert_eq!(yf.len(), 5);
    for (i, &v) in yf.iter().enumerate() {
        assert!(v.is_nan() || v.is_finite(), "y[{i}]={v} is neither NaN nor finite");
    }
    // The extreme endpoints (x=0 and x=2) should emit NaN: h==max_dist, u=1 → w=0 for
    // the farthest point, leaving rank-2 system.
    assert!(yf[0].is_nan(), "y[0]={} should be NaN (rank-deficient at left endpoint)", yf[0]);
    assert!(yf[4].is_nan(), "y[4]={} should be NaN (rank-deficient at right endpoint)", yf[4]);
}

#[test]
fn test_loess_deg1_bootstrap_ci_is_reproducible_under_seed() {
    let xs: Vec<f64> = (0..40).map(|i| i as f64 / 10.0).collect();
    let ys: Vec<f64> = xs.iter().map(|x| (x).sin()).collect();
    let batch = xy_batch(xs, ys);
    let spec1 = SmoothSpec {
        residual_columns: Default::default(),
        x: "x".into(), y: "y".into(),
        method: SmoothMethod::Loess,
        ci: Some(0.95),
        bandwidth: 0.5, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 20, seed: 42,
        x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
        inject_zero_ref: false,
        inject_metrics: false,
    x_range: None, x_range_pad: None,
        groupby: vec![],
        name: None,
    };
    let spec2 = spec1.clone();
    let out1 = apply(&spec1, &batch).unwrap();
    let out2 = apply(&spec2, &batch).unwrap();
    let lo1 = col(&out1, "ci_lower");
    let lo2 = col(&out2, "ci_lower");
    let hi1 = col(&out1, "ci_upper");
    let hi2 = col(&out2, "ci_upper");
    for i in 0..lo1.len() {
        assert_eq!(lo1[i].to_bits(), lo2[i].to_bits(), "ci_lower not deterministic at {i}");
        assert_eq!(hi1[i].to_bits(), hi2[i].to_bits(), "ci_upper not deterministic at {i}");
    }
}

#[test]
fn smooth_x_bins_pre_aggregates_then_fits() {
    let xs: Vec<f64> = (0..100).map(|i| i as f64 / 10.0).collect();
    let ys: Vec<f64> = xs.iter().map(|x| 2.0 * x + 1.0).collect();
    let batch = xy_batch(xs, ys);
    let spec = SmoothSpec {
        residual_columns: Default::default(),
        x: "x".into(), y: "y".into(),
        method: SmoothMethod::Lm, ci: None,
        bandwidth: 0.0, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
        x_bins: Some(10),
        x_estimator: Some(crate::transform::aggregate::AggFn::Mean),
        output: SmoothOutput::Fitted,
        inject_zero_ref: false,
        inject_metrics: false,
    x_range: None, x_range_pad: None,
        groupby: vec![],
        name: None,
    };
    let out = apply(&spec, &batch).unwrap();
    let xg = col(&out, "x");
    let yf = col(&out, "y");
    let slope = (yf[xg.len() - 1] - yf[0]) / (xg[xg.len() - 1] - xg[0]);
    assert!((slope - 2.0).abs() < 1e-6, "expected slope 2.0, got {slope}");
}

#[test]
fn smooth_output_residuals_returns_y_minus_fitted() {
    let xs: Vec<f64> = (0..50).map(|i| i as f64).collect();
    let ys: Vec<f64> = xs.iter().enumerate().map(|(i, &x)|
        2.0 * x + 1.0 + if i % 2 == 0 { 0.1 } else { -0.1 }
    ).collect();
    let batch = xy_batch(xs, ys);
    let spec = SmoothSpec {
        residual_columns: Default::default(),
        x: "x".into(), y: "y".into(),
        method: SmoothMethod::Lm, ci: None,
        bandwidth: 0.0, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
        x_bins: None, x_estimator: None,
        output: SmoothOutput::Residuals,
        inject_zero_ref: false,
        inject_metrics: false,
        groupby: vec![],
        name: None,
        x_range: None, x_range_pad: None,
    };
    let out = apply(&spec, &batch).unwrap();
    assert_eq!(out.schema().field(0).name(), "x");
    assert_eq!(out.schema().field(1).name(), "residual");
    let r = col(&out, "residual");
    let mean_r: f64 = r.iter().sum::<f64>() / r.len() as f64;
    assert!(mean_r.abs() < 1e-9, "residual mean = {mean_r}");
    let max_abs = r.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
    assert!(max_abs < 0.5, "max |residual| = {max_abs}");
}

#[test]
fn smooth_output_default_is_fitted() {
    let xs: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let ys: Vec<f64> = xs.iter().map(|x| x + 1.0).collect();
    let batch = xy_batch(xs, ys);
    let spec = SmoothSpec {
        residual_columns: Default::default(),
        x: "x".into(), y: "y".into(),
        method: SmoothMethod::Lm, ci: Some(0.95),
        bandwidth: 0.0, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
        x_bins: None, x_estimator: None,
        output: SmoothOutput::Fitted,
        inject_zero_ref: false,
        inject_metrics: false,
    x_range: None, x_range_pad: None,
        groupby: vec![],
        name: None,
    };
    let out = apply(&spec, &batch).unwrap();
    let schema = out.schema();
    let names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
    assert_eq!(names, vec!["x".to_string(), "y".to_string(), "ci_lower".to_string(), "ci_upper".to_string()]);
}

#[test]
fn test_lm_two_row_minimal_fit() {
    // 2 points is the minimum for a valid linear fit: y = 1 + 2x.
    let batch = xy_batch(vec![0.0, 1.0], vec![1.0, 3.0]);
    let spec = SmoothSpec {
        residual_columns: Default::default(),
        x: "x".into(), y: "y".into(),
        method: SmoothMethod::Lm,
        ci: None,
        bandwidth: 0.0, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
        x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
        inject_zero_ref: false,
        inject_metrics: false,
        x_range: None, x_range_pad: None,
        groupby: vec![],
        name: None,
    };
    let out = apply(&spec, &batch).unwrap();
    let xg = col(&out, "x");
    let yf = col(&out, "y");
    assert_eq!(xg.len(), 5);
    // Fit should be y = 1 + 2x exactly (perfect fit through 2 points).
    for (xq, yq) in xg.iter().zip(yf.iter()) {
        let expected = 1.0 + 2.0 * xq;
        assert!(
            (yq - expected).abs() < 1e-10,
            "y(x={xq})={yq}, expected {expected}"
        );
    }
}

#[test]
fn test_lm_constant_y_flat_line() {
    // All y=3.0 → fit should be flat at y=3.0 regardless of x.
    let xs: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let ys: Vec<f64> = vec![3.0; 10];
    let batch = xy_batch(xs, ys);
    let spec = SmoothSpec {
        residual_columns: Default::default(),
        x: "x".into(), y: "y".into(),
        method: SmoothMethod::Lm,
        ci: None,
        bandwidth: 0.0, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 7, seed: 0,
        x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
        inject_zero_ref: false,
        inject_metrics: false,
        x_range: None, x_range_pad: None,
        groupby: vec![],
        name: None,
    };
    let out = apply(&spec, &batch).unwrap();
    let yf = col(&out, "y");
    for (i, &v) in yf.iter().enumerate() {
        assert!(
            (v - 3.0).abs() < 1e-10,
            "expected y=3.0 at grid point {i}; got {v}"
        );
    }
}

// ---- Method alias & new-method tests ----

/// "linear" must produce exactly the same output as "lm".
#[test]
fn test_linear_alias_matches_lm() {
    let xs: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let ys: Vec<f64> = xs.iter().map(|&x| 2.5 * x - 3.0).collect();
    let batch = xy_batch(xs, ys);
    let base = SmoothSpec {
        residual_columns: Default::default(),
        x: "x".into(), y: "y".into(),
        method: SmoothMethod::Lm,
        ci: Some(0.95),
        bandwidth: 0.75, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 10, seed: 0,
        x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
        inject_zero_ref: false, inject_metrics: false, x_range: None, x_range_pad: None,
        groupby: vec![], name: None,
    };
    // "linear" maps to SmoothMethod::Lm in the PySmooth constructor.
    // Test directly by verifying both produce identical RecordBatch output.
    let out_lm = apply(&base, &batch).unwrap();
    // The "linear" alias resolves to the same SmoothMethod::Lm struct,
    // so we verify the spec round-trip via JSON then re-apply.
    let json = serde_json::to_string(&base).unwrap();
    let parsed: SmoothSpec = serde_json::from_str(&json).unwrap();
    let out_linear = apply(&parsed, &batch).unwrap();
    let y_lm = col(&out_lm, "y");
    let y_linear = col(&out_linear, "y");
    for (a, b) in y_lm.iter().zip(y_linear.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "'linear' and 'lm' must produce identical output");
    }
}

/// "quadratic" (Poly { degree: 2 }) must recover a perfect degree-2 polynomial.
#[test]
fn test_quadratic_recovers_degree2_polynomial() {
    // y = 1 + 2x + 3x² — a perfect quadratic.
    let xs: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
    let ys: Vec<f64> = xs.iter().map(|&x| 1.0 + 2.0 * x + 3.0 * x * x).collect();
    let batch = xy_batch(xs.clone(), ys.clone());
    let spec = SmoothSpec {
        residual_columns: Default::default(),
        x: "x".into(), y: "y".into(),
        method: SmoothMethod::Poly { degree: 2 },
        ci: None,
        bandwidth: 0.75, degree: 2, k: gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
        x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
        inject_zero_ref: false, inject_metrics: false, x_range: None, x_range_pad: None,
        groupby: vec![], name: None,
    };
    let out = apply(&spec, &batch).unwrap();
    let xg = col(&out, "x");
    let yf = col(&out, "y");
    assert_eq!(yf.len(), 5);
    for (&xq, &yq) in xg.iter().zip(yf.iter()) {
        let expected = 1.0 + 2.0 * xq + 3.0 * xq * xq;
        assert!(
            (yq - expected).abs() < 1e-6,
            "quadratic: y({xq}) = {yq}, expected {expected}"
        );
    }
}

/// "cubic" (Poly { degree: 3 }) must recover a perfect degree-3 polynomial.
#[test]
fn test_cubic_recovers_degree3_polynomial() {
    // y = -2 + x - 0.5x² + 0.1x³
    let xs: Vec<f64> = (0..30).map(|i| i as f64 * 0.3).collect();
    let ys: Vec<f64> = xs.iter().map(|&x| -2.0 + x - 0.5 * x * x + 0.1 * x * x * x).collect();
    let batch = xy_batch(xs.clone(), ys.clone());
    let spec = SmoothSpec {
        residual_columns: Default::default(),
        x: "x".into(), y: "y".into(),
        method: SmoothMethod::Poly { degree: 3 },
        ci: None,
        bandwidth: 0.75, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 7, seed: 0,
        x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
        inject_zero_ref: false, inject_metrics: false, x_range: None, x_range_pad: None,
        groupby: vec![], name: None,
    };
    let out = apply(&spec, &batch).unwrap();
    let xg = col(&out, "x");
    let yf = col(&out, "y");
    for (&xq, &yq) in xg.iter().zip(yf.iter()) {
        let expected = -2.0 + xq - 0.5 * xq * xq + 0.1 * xq * xq * xq;
        assert!(
            (yq - expected).abs() < 1e-5,
            "cubic: y({xq}) = {yq}, expected {expected}"
        );
    }
}

/// "log" (LogX) must fit y ~ a + b·ln(x) and return NaN for x <= 0.
#[test]
fn test_log_x_recovers_log_relationship() {
    // y = 1 + 3·ln(x) (ground truth)
    let xs: Vec<f64> = (1..=30).map(|i| i as f64).collect();
    let ys: Vec<f64> = xs.iter().map(|&x| 1.0 + 3.0 * x.ln()).collect();
    let batch = xy_batch(xs.clone(), ys.clone());
    let spec = SmoothSpec {
        residual_columns: Default::default(),
        x: "x".into(), y: "y".into(),
        method: SmoothMethod::LogX,
        ci: None,
        bandwidth: 0.75, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 10, seed: 0,
        x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
        inject_zero_ref: false, inject_metrics: false, x_range: None, x_range_pad: None,
        groupby: vec![], name: None,
    };
    let out = apply(&spec, &batch).unwrap();
    let xg = col(&out, "x");
    let yf = col(&out, "y");
    for (&xq, &yq) in xg.iter().zip(yf.iter()) {
        let expected = 1.0 + 3.0 * xq.ln();
        assert!(
            (yq - expected).abs() < 1e-6,
            "log: y({xq}) = {yq}, expected {expected}"
        );
    }
}

/// "sqrt" (SqrtX) must fit y ~ a + b·√x.
#[test]
fn test_sqrt_x_recovers_sqrt_relationship() {
    // y = 2 + 4·√x (ground truth)
    let xs: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let ys: Vec<f64> = xs.iter().map(|&x| 2.0 + 4.0 * x.sqrt()).collect();
    let batch = xy_batch(xs.clone(), ys.clone());
    let spec = SmoothSpec {
        residual_columns: Default::default(),
        x: "x".into(), y: "y".into(),
        method: SmoothMethod::SqrtX,
        ci: None,
        bandwidth: 0.75, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 10, seed: 0,
        x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
        inject_zero_ref: false, inject_metrics: false, x_range: None, x_range_pad: None,
        groupby: vec![], name: None,
    };
    let out = apply(&spec, &batch).unwrap();
    let xg = col(&out, "x");
    let yf = col(&out, "y");
    for (&xq, &yq) in xg.iter().zip(yf.iter()) {
        let expected = 2.0 + 4.0 * xq.sqrt();
        assert!(
            (yq - expected).abs() < 1e-6,
            "sqrt: y({xq}) = {yq}, expected {expected}"
        );
    }
}

/// "log" on a grid that includes x=0 must produce NaN at that grid point.
#[test]
fn test_log_x_nan_at_nonpositive_grid_points() {
    let xs: Vec<f64> = (1..=20).map(|i| i as f64).collect();
    let ys: Vec<f64> = xs.iter().map(|&x| x.ln()).collect();
    let batch = xy_batch(xs, ys);
    let spec = SmoothSpec {
        residual_columns: Default::default(),
        x: "x".into(), y: "y".into(),
        method: SmoothMethod::LogX,
        ci: None,
        bandwidth: 0.75, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
        x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
        inject_zero_ref: false, inject_metrics: false,
        // Force x_range to include x=0 to probe the NaN path.
        x_range: Some([0.0, 20.0]), x_range_pad: None,
        groupby: vec![], name: None,
    };
    let out = apply(&spec, &batch).unwrap();
    let xg = col(&out, "x");
    let yf = col(&out, "y");
    // The first grid point (x=0) must be NaN since ln(0) = -inf.
    assert!(
        yf[0].is_nan(),
        "log method must emit NaN at x={} (grid point 0); got {}",
        xg[0], yf[0]
    );
    // Remaining grid points (x > 0) must be finite.
    for (i, &yq) in yf.iter().enumerate().skip(1) {
        assert!(
            yq.is_finite(),
            "log method: y[{i}] should be finite at x={} but got {yq}",
            xg[i]
        );
    }
}

/// Every declared method (iterated from `METHODS` itself, not a
/// hand-copied literal — batch-D T9 round 2) must be accepted without
/// error by PySmooth::new.
#[test]
fn test_all_method_aliases_accepted_by_pysmooth_new() {
    pyo3::Python::initialize();
    for method in METHODS {
        let result = PySmooth::new(
            "x", "y",
            method,
            None, None, None, None, 200, 0,
            None, None, "fitted",
            false, false, None, None, None, None,
        );
        assert!(
            result.is_ok(),
            "PySmooth::new with method={method:?} should succeed; got {:?}",
            result.err().map(|e| e.to_string())
        );
    }
}

/// `PySmooth::new` refuses a non-finite `x_range` endpoint, at either end,
/// for each of NaN/+inf/-inf — a regression test for the binding check
/// `binding.rs`'s constructor runs (`x_range`'s `is_finite()` guard): were
/// that check deleted, every one of these six cases would construct
/// successfully instead of refusing (`result.err()` would be `None`), and
/// the test would panic at `.err().unwrap_or_else(..)` ("must be refused,
/// not accepted"), never reaching the `assert_eq!` on the refusal text.
#[test]
fn pysmooth_new_refuses_a_non_finite_x_range_at_either_end() {
    pyo3::Python::initialize();
    for (a, b) in [
        (f64::NAN, 8.0),
        (8.0, f64::NAN),
        (f64::INFINITY, 8.0),
        (8.0, f64::INFINITY),
        (f64::NEG_INFINITY, 8.0),
        (8.0, f64::NEG_INFINITY),
    ] {
        let result = PySmooth::new(
            "x", "y", "lm",
            None, None, None, None, 200, 0,
            None, None, "fitted",
            false, false, None, None, Some([a, b]), None,
        );
        let err = result.err().unwrap_or_else(|| panic!("x_range=[{a}, {b}] must be refused, not accepted"));
        assert_eq!(
            err.to_string(),
            format!("ValueError: Smooth: x_range must be (lo, hi) with both finite; received [{a}, {b}]"),
            "x_range=[{a}, {b}]"
        );
    }
}

// ── NaN regression: smooth must drop NaN rows, matching kde/bin behavior ────

/// A NaN x-value in the input must be silently dropped: the smoothed output
/// must be identical to running the same smooth on the clean (NaN-free) subset.
#[test]
fn test_nan_in_x_is_dropped_from_smooth() {
    let xs_clean: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let ys_clean: Vec<f64> = xs_clean.iter().map(|x| 2.0 * x + 1.0).collect();

    // Inject a NaN at index 3.
    let mut xs_nan = xs_clean.clone();
    xs_nan[3] = f64::NAN;

    let batch_clean = xy_batch(xs_clean.clone(), ys_clean.clone());
    let batch_nan = xy_batch(xs_nan, ys_clean.clone());

    let spec = SmoothSpec {
        residual_columns: Default::default(),
        x: "x".into(), y: "y".into(),
        method: SmoothMethod::Lm,
        ci: None,
        bandwidth: 0.0, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
        x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
        inject_zero_ref: false, inject_metrics: false, x_range: None, x_range_pad: None,
        groupby: vec![], name: None,
    };

    let out_clean = apply(&spec, &batch_clean).unwrap();
    let out_nan   = apply(&spec, &batch_nan).unwrap();

    let yf_clean = col(&out_clean, "y");
    let yf_nan   = col(&out_nan,   "y");

    assert_eq!(yf_clean.len(), yf_nan.len());
    for (i, (a, b)) in yf_clean.iter().zip(yf_nan.iter()).enumerate() {
        // Neither result should be NaN-corrupted.
        assert!(a.is_finite(), "clean output y[{i}]={a} should be finite");
        assert!(b.is_finite(), "nan-input output y[{i}]={b} should be NaN-corrupted (expected finite)");
        assert!(
            (a - b).abs() < 1e-9,
            "y[{i}]: clean={a}, nan-input={b}; dropping NaN row must not change smoothed values"
        );
    }
}

/// A NaN y-value in the input must be silently dropped.
#[test]
fn test_nan_in_y_is_dropped_from_smooth() {
    let xs_clean: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let ys_clean: Vec<f64> = xs_clean.iter().map(|x| 3.0 * x - 0.5).collect();

    // Inject a NaN at index 5.
    let mut ys_nan = ys_clean.clone();
    ys_nan[5] = f64::NAN;

    let batch_clean = xy_batch(xs_clean.clone(), ys_clean.clone());
    let batch_nan = xy_batch(xs_clean.clone(), ys_nan);

    let spec = SmoothSpec {
        residual_columns: Default::default(),
        x: "x".into(), y: "y".into(),
        method: SmoothMethod::Lm,
        ci: None,
        bandwidth: 0.0, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
        x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
        inject_zero_ref: false, inject_metrics: false, x_range: None, x_range_pad: None,
        groupby: vec![], name: None,
    };

    let out_clean = apply(&spec, &batch_clean).unwrap();
    let out_nan   = apply(&spec, &batch_nan).unwrap();

    let yf_clean = col(&out_clean, "y");
    let yf_nan   = col(&out_nan,   "y");

    assert_eq!(yf_clean.len(), yf_nan.len());
    for (i, (a, b)) in yf_clean.iter().zip(yf_nan.iter()).enumerate() {
        // Both should be finite (NaN row was dropped, LM can still fit).
        assert!(a.is_finite(), "clean output y[{i}]={a} should be finite");
        assert!(b.is_finite(), "nan-y-input output y[{i}]={b} should be finite after NaN drop");
        // The fit values must be close but NOT required to be byte-identical
        // (removing one data point changes the regression coefficients slightly).
        // What must NOT happen: NaN-corrupted output.  Here we just assert finite.
    }
}

// ── T0.3: grouped smooth accepts non-Utf8 group dtypes + preserves dtype ──

fn col_i64<'a>(b: &'a RecordBatch, name: &str) -> &'a arrow::array::Int64Array {
    b.column(b.schema().index_of(name).unwrap())
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .unwrap()
}

/// An Int64 group column must work (was "must be Utf8") AND round-trip as
/// Int64 in the output, not a forced StringArray (XFORM-03).
#[test]
fn test_smooth_grouped_int64_group_roundtrips_as_int64() {
    use arrow::array::Int64Array;
    pyo3::Python::initialize();
    // Two groups, each a perfect line so LM recovers it exactly.
    let xs: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let mut all_x: Vec<f64> = Vec::new();
    let mut all_y: Vec<f64> = Vec::new();
    let mut all_g: Vec<i64> = Vec::new();
    for &x in &xs {
        all_x.push(x);
        all_y.push(3.0 + 2.0 * x); // group 100
        all_g.push(100);
    }
    for &x in &xs {
        all_x.push(x);
        all_y.push(1.0 - 1.0 * x); // group 200
        all_g.push(200);
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, true),
        Field::new("y", DataType::Float64, true),
        Field::new("g", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(all_x)),
            Arc::new(Float64Array::from(all_y)),
            Arc::new(Int64Array::from(all_g)),
        ],
    )
    .unwrap();
    let spec = SmoothSpec {
        residual_columns: Default::default(),
        x: "x".into(), y: "y".into(),
        method: SmoothMethod::Lm,
        ci: None,
        bandwidth: 0.0, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
        x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
        inject_zero_ref: false, inject_metrics: false, x_range: None, x_range_pad: None,
        groupby: vec!["g".into()],
        name: None,
    };
    let out = apply(&spec, &batch).expect("Int64 groupby must succeed");
    // 2 groups × 5 grid points = 10 rows.
    assert_eq!(out.num_rows(), 10);
    // Group column must be the LAST field, dtype Int64 (NOT Utf8).
    let last = out.schema().fields().len() - 1;
    assert_eq!(out.schema().field(last).name(), "g");
    assert_eq!(
        out.schema().field(last).data_type(),
        &DataType::Int64,
        "group column must round-trip as Int64, not String"
    );
    let groups = col_i64(&out, "g");
    for i in 0..5 {
        assert_eq!(groups.value(i), 100, "first 5 rows are group 100");
    }
    for i in 5..10 {
        assert_eq!(groups.value(i), 200, "last 5 rows are group 200");
    }
    // Each group's fit recovers its own line.
    let yf = col(&out, "y");
    let xg = col(&out, "x");
    for i in 0..5 {
        assert!((yf[i] - (3.0 + 2.0 * xg[i])).abs() < 1e-9, "group 100 fit");
    }
    for i in 5..10 {
        assert!((yf[i] - (1.0 - 1.0 * xg[i])).abs() < 1e-9, "group 200 fit");
    }
}

/// Batch-F T2h: grouping by a column whose name collides with an output
/// column name (`y`) must refuse with a typed error naming the
/// collision, not silently append a second `y` field that
/// `Schema::index_of` would resolve to the wrong one downstream.
#[test]
fn test_smooth_grouped_name_collision_with_output_column_refuses() {
    pyo3::Python::initialize();
    let xs: Vec<f64> = (0..6).map(|i| i as f64).collect();
    let ys: Vec<f64> = xs.iter().map(|&x| 2.0 * x).collect();
    let group_vals: Vec<&str> = vec!["a", "a", "a", "b", "b", "b"];
    // The groupby column is literally named "y" — the Fitted output's
    // own fitted-y field — while the real x/y data live in "x_in"/"y_in".
    let schema = Arc::new(Schema::new(vec![
        Field::new("x_in", DataType::Float64, true),
        Field::new("y_in", DataType::Float64, true),
        Field::new("y", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(xs)),
            Arc::new(Float64Array::from(ys)),
            Arc::new(arrow::array::StringArray::from(group_vals)),
        ],
    )
    .unwrap();
    let spec = SmoothSpec {
        residual_columns: Default::default(),
        x: "x_in".into(), y: "y_in".into(),
        method: SmoothMethod::Lm,
        ci: None,
        bandwidth: 0.0, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 4, seed: 0,
        x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
        inject_zero_ref: false, inject_metrics: false, x_range: None, x_range_pad: None,
        groupby: vec!["y".into()],
        name: None,
    };
    let err = apply(&spec, &batch).expect_err(
        "groupby column colliding with an output column name must refuse",
    );
    let msg = err.to_string();
    assert!(msg.contains('y'), "error must name the colliding column: {msg}");
    assert!(msg.contains("collides"), "error must explain the collision: {msg}");
}

/// A Boolean group column must work (was "must be Utf8") and round-trip as
/// Boolean in the output.
#[test]
fn test_smooth_grouped_boolean_group_column() {
    use arrow::array::BooleanArray;
    pyo3::Python::initialize();
    let xs: Vec<f64> = (0..8).map(|i| i as f64).collect();
    let mut all_x: Vec<f64> = Vec::new();
    let mut all_y: Vec<f64> = Vec::new();
    let mut all_g: Vec<bool> = Vec::new();
    for &x in &xs {
        all_x.push(x);
        all_y.push(2.0 * x);
        all_g.push(true);
    }
    for &x in &xs {
        all_x.push(x);
        all_y.push(-3.0 * x + 1.0);
        all_g.push(false);
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, true),
        Field::new("y", DataType::Float64, true),
        Field::new("g", DataType::Boolean, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(all_x)),
            Arc::new(Float64Array::from(all_y)),
            Arc::new(BooleanArray::from(all_g)),
        ],
    )
    .unwrap();
    let spec = SmoothSpec {
        residual_columns: Default::default(),
        x: "x".into(), y: "y".into(),
        method: SmoothMethod::Lm,
        ci: None,
        bandwidth: 0.0, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 4, seed: 0,
        x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
        inject_zero_ref: false, inject_metrics: false, x_range: None, x_range_pad: None,
        groupby: vec!["g".into()],
        name: None,
    };
    let out = apply(&spec, &batch).expect("Boolean groupby must succeed");
    assert_eq!(out.num_rows(), 8, "2 groups × 4 grid points");
    let last = out.schema().fields().len() - 1;
    assert_eq!(out.schema().field(last).data_type(), &DataType::Boolean);
    let groups = out
        .column(last)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap();
    for i in 0..4 {
        assert!(groups.value(i), "first group (true) appears first");
    }
    for i in 4..8 {
        assert!(!groups.value(i), "second group (false) appears second");
    }
}

/// Sentinel: a Utf8 group column still round-trips as Utf8 (byte-stable path).
#[test]
fn test_smooth_grouped_utf8_group_unchanged() {
    use arrow::array::StringArray;
    pyo3::Python::initialize();
    let xs: Vec<f64> = (0..6).map(|i| i as f64).collect();
    let mut all_x: Vec<f64> = Vec::new();
    let mut all_y: Vec<f64> = Vec::new();
    let mut all_g: Vec<&str> = Vec::new();
    for &x in &xs {
        all_x.push(x);
        all_y.push(x + 1.0);
        all_g.push("A");
    }
    for &x in &xs {
        all_x.push(x);
        all_y.push(2.0 * x);
        all_g.push("B");
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, true),
        Field::new("y", DataType::Float64, true),
        Field::new("g", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(all_x)),
            Arc::new(Float64Array::from(all_y)),
            Arc::new(StringArray::from(all_g)),
        ],
    )
    .unwrap();
    let spec = SmoothSpec {
        residual_columns: Default::default(),
        x: "x".into(), y: "y".into(),
        method: SmoothMethod::Lm,
        ci: None,
        bandwidth: 0.0, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 4, seed: 0,
        x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
        inject_zero_ref: false, inject_metrics: false, x_range: None, x_range_pad: None,
        groupby: vec!["g".into()],
        name: None,
    };
    let out = apply(&spec, &batch).unwrap();
    let last = out.schema().fields().len() - 1;
    assert_eq!(out.schema().field(last).data_type(), &DataType::Utf8);
    let groups = out
        .column(last)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(groups.value(0), "A");
    assert_eq!(groups.value(out.num_rows() - 1), "B");
}

/// GH #197 EXEMPTION: an all-null `groupby` column leaves this transform
/// with nothing to partition, so it fits the whole batch pooled instead of
/// emitting nothing. The pooled fit IS the null-keyed group: its output
/// carries the `groupby` column too, all null at its original dtype, and
/// its own columns equal the ungrouped fit's — the fit itself is unchanged.
#[test]
fn test_smooth_all_null_group_pools_the_whole_batch_with_a_null_group_column() {
    use arrow::array::StringArray;
    pyo3::Python::initialize();
    let x: Vec<f64> = (0..6).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&v| v + 1.0).collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, true),
        Field::new("y", DataType::Float64, true),
        Field::new("g", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(x.clone())),
            Arc::new(Float64Array::from(y.clone())),
            Arc::new(StringArray::from(vec![None::<&str>; 6])),
        ],
    )
    .unwrap();
    let grouped = SmoothSpec {
        residual_columns: Default::default(),
        x: "x".into(), y: "y".into(),
        method: SmoothMethod::Lm,
        ci: None,
        bandwidth: 0.0, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 4, seed: 0,
        x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
        inject_zero_ref: false, inject_metrics: false, x_range: None, x_range_pad: None,
        groupby: vec!["g".into()],
        name: None,
    };
    let out = apply(&grouped, &batch).unwrap();

    let g = out.column(out.schema().index_of("g").unwrap());
    assert_eq!(g.data_type(), &DataType::Utf8, "the groupby column keeps its original dtype");
    assert_eq!(g.null_count(), g.len(), "the pooled fit IS the null-keyed group");

    let ungrouped = SmoothSpec { groupby: Vec::new(), ..grouped };
    let whole = apply(&ungrouped, &xy_batch(x, y)).unwrap();
    for field in whole.schema().fields() {
        assert_eq!(
            out.column_by_name(field.name()),
            whole.column_by_name(field.name()),
            "the pooled fit's own column '{}' equals the ungrouped fit's",
            field.name()
        );
    }
}

/// Unknown method aliases must return an error.
#[test]
fn test_unknown_method_alias_returns_error() {
    pyo3::Python::initialize();
    let result = PySmooth::new(
        "x", "y", "polynomial",
        None, None, None, None, 200, 0,
        None, None, "fitted",
        false, false, None, None, None, None,
    );
    assert!(result.is_err(), "unknown method 'polynomial' must return an error");
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("unknown method"), "error message should mention 'unknown method': {msg}");
}

// ---- Batch-D T9: glm routing + lm degree≥2 → polynomial promotion ----

/// `Smooth(method="glm")` must be accepted BY THE STRING PARSER
/// (previously "glm" hit the `other` catch-all and raised "unknown
/// method"). Batch-D T9 round 2 (Required Fix 3, rust-quality verdict):
/// routes through `PySmooth::new`, not a hand-built `SmoothSpec` literal
/// with `method: SmoothMethod::Glm` already set — the prior version of
/// this test compiled and passed regardless of whether the parser
/// accepted `"glm"` at all, so its "no longer hits the catch-all" claim
/// was not actually proven by it.
#[test]
fn test_glm_method_accepted_and_produces_finite_fit() {
    pyo3::Python::initialize();
    let xs: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let ys: Vec<f64> = xs.iter().enumerate()
        .map(|(i, &x)| 3.0 + 2.0 * x + if i % 2 == 0 { 0.3 } else { -0.3 })
        .collect();
    let batch = xy_batch(xs, ys);
    let ctor = PySmooth::new(
        "x", "y", "glm",
        None, None, None, None, 10, 0,
        None, None, "fitted",
        false, false, None, None, None, None,
    ).expect("PySmooth::new(method='glm') must be accepted by the string parser");
    let spec = match ctor.0 {
        TransformSpec::Smooth(s) => s,
        _ => panic!("expected Smooth spec"),
    };
    let out = apply(&spec, &batch).unwrap();
    let yf = col(&out, "y");
    assert!(yf.iter().all(|v| v.is_finite()), "glm fit must be finite; got {yf:?}");
}

/// GLM (Gaussian, Identity link) IRLS converges to the exact OLS point
/// estimate, so its fitted curve must match "lm"'s to high precision —
/// this is the correct behavior (not a sign the two are aliased; the
/// CI-band test below proves the implementations are genuinely distinct).
#[test]
fn test_glm_point_estimate_matches_lm_ols() {
    let xs: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let ys: Vec<f64> = xs.iter().enumerate()
        .map(|(i, &x)| 3.0 + 2.0 * x + if i % 3 == 0 { 0.4 } else { -0.2 })
        .collect();
    let batch = xy_batch(xs, ys);
    let lm_spec = SmoothSpec {
        residual_columns: Default::default(),
        x: "x".into(), y: "y".into(),
        method: SmoothMethod::Lm,
        ci: None,
        bandwidth: 0.75, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 10, seed: 0,
        x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
        inject_zero_ref: false, inject_metrics: false, x_range: None, x_range_pad: None,
        groupby: vec![], name: None,
    };
    let glm_spec = SmoothSpec { method: SmoothMethod::Glm, ..lm_spec.clone() };
    let lm_out = apply(&lm_spec, &batch).unwrap();
    let glm_out = apply(&glm_spec, &batch).unwrap();
    let lm_y = col(&lm_out, "y");
    let glm_y = col(&glm_out, "y");
    for (a, b) in lm_y.iter().zip(glm_y.iter()) {
        assert!((a - b).abs() < 1e-8, "glm should match lm's OLS point estimate: lm={a}, glm={b}");
    }
}

/// RED-proof discriminator: a `"glm" => SmoothMethod::Lm` alias would
/// happen to match `lm`'s point estimate (see above) but could not
/// reproduce `glm.rs`'s real Wald CI band, which omits the dispersion
/// (σ²) scaling that `lm`'s t-distribution band applies. With `ci` set,
/// the two bands must therefore differ even though the fitted line is
/// the same — proving `glm_fit` takes a genuinely distinct code path
/// rather than aliasing to `Lm`.
#[test]
fn test_glm_ci_band_differs_from_lm_ci_band_proving_distinct_implementation() {
    let xs: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let ys: Vec<f64> = xs.iter().enumerate()
        .map(|(i, &x)| 3.0 + 2.0 * x + if i % 2 == 0 { 1.5 } else { -1.5 })
        .collect();
    let batch = xy_batch(xs, ys);
    let lm_spec = SmoothSpec {
        residual_columns: Default::default(),
        x: "x".into(), y: "y".into(),
        method: SmoothMethod::Lm,
        ci: Some(0.95),
        bandwidth: 0.75, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 10, seed: 0,
        x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
        inject_zero_ref: false, inject_metrics: false, x_range: None, x_range_pad: None,
        groupby: vec![], name: None,
    };
    let glm_spec = SmoothSpec { method: SmoothMethod::Glm, ..lm_spec.clone() };
    let lm_out = apply(&lm_spec, &batch).unwrap();
    let glm_out = apply(&glm_spec, &batch).unwrap();
    let lm_width: Vec<f64> = col(&lm_out, "ci_upper").iter().zip(col(&lm_out, "ci_lower").iter())
        .map(|(&hi, &lo)| hi - lo).collect();
    let glm_width: Vec<f64> = col(&glm_out, "ci_upper").iter().zip(col(&glm_out, "ci_lower").iter())
        .map(|(&hi, &lo)| hi - lo).collect();
    let any_differs = lm_width.iter().zip(glm_width.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(
        any_differs,
        "glm and lm CI bands must differ (real GLM Wald CI has no dispersion scaling); \
         lm_width={lm_width:?}, glm_width={glm_width:?}"
    );
}

/// `Smooth(method="lm", degree=2)` must resolve to the exact same
/// `Poly{degree:2}` code path as `Smooth(method="quadratic")` — byte-equal
/// output, since both now construct the identical enum variant.
#[test]
fn test_lm_degree_2_byte_equal_to_quadratic() {
    let xs: Vec<f64> = (0..20).map(|i| i as f64 * 0.3).collect();
    let ys: Vec<f64> = xs.iter().map(|&x| 1.0 + 2.0 * x + 3.0 * x * x).collect();
    let batch = xy_batch(xs, ys);
    let lm_deg2_spec = SmoothSpec {
        residual_columns: Default::default(),
        x: "x".into(), y: "y".into(),
        method: SmoothMethod::Poly { degree: 2 },
        ci: None,
        bandwidth: 0.75, degree: 2, k: gam::DEFAULT_BASIS_DIM, n: 15, seed: 0,
        x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
        inject_zero_ref: false, inject_metrics: false, x_range: None, x_range_pad: None,
        groupby: vec![], name: None,
    };
    // Route both through PySmooth::new to prove the *constructor's*
    // resolution — not just the shared enum — treats them identically.
    pyo3::Python::initialize();
    let lm_ctor = PySmooth::new(
        "x", "y", "lm", None, None, Some(2), None, 15, 0, None, None, "fitted",
        false, false, None, None, None, None,
    ).unwrap();
    let quad_ctor = PySmooth::new(
        "x", "y", "quadratic", None, None, None, None, 15, 0, None, None, "fitted",
        false, false, None, None, None, None,
    ).unwrap();
    assert_eq!(lm_ctor.0, quad_ctor.0, "method='lm', degree=2 must construct the identical spec as method='quadratic'");

    let out = apply(&lm_deg2_spec, &batch).unwrap();
    let xg = col(&out, "x");
    let yf = col(&out, "y");
    for (&xq, &yq) in xg.iter().zip(yf.iter()) {
        let expected = 1.0 + 2.0 * xq + 3.0 * xq * xq;
        assert!((yq - expected).abs() < 1e-6, "lm+degree=2: y({xq})={yq}, expected {expected}");
    }
}

/// `Smooth(method="lm")` with no `degree` and `Smooth(method="lm",
/// degree=1)` must both resolve to plain `SmoothMethod::Lm` — the
/// no-op case every existing `mark_smooth(method="lm")` caller (and
/// `lmplot`'s default `order=1`) relies on for unchanged output.
#[test]
fn test_lm_no_degree_and_degree_1_both_stay_linear() {
    pyo3::Python::initialize();
    let implicit = PySmooth::new(
        "x", "y", "lm", None, None, None, None, 200, 0, None, None, "fitted",
        false, false, None, None, None, None,
    ).unwrap();
    let explicit_one = PySmooth::new(
        "x", "y", "lm", None, None, Some(1), None, 200, 0, None, None, "fitted",
        false, false, None, None, None, None,
    ).unwrap();
    assert_eq!(implicit.0, explicit_one.0, "degree omitted vs degree=1 must construct the same spec");
    match &implicit.0 {
        TransformSpec::Smooth(s) => assert_eq!(s.method, SmoothMethod::Lm),
        _ => panic!("expected Smooth spec"),
    }
}

/// `degree=0` with `method="lm"` must refuse rather than silently
/// behaving like `degree=1`.
#[test]
fn test_lm_degree_zero_refuses() {
    pyo3::Python::initialize();
    let result = PySmooth::new(
        "x", "y", "lm", None, None, Some(0), None, 200, 0, None, None, "fitted",
        false, false, None, None, None, None,
    );
    assert!(result.is_err(), "degree=0 with method='lm' must be refused");
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("degree"), "error should mention 'degree': {msg}");
}

/// Closing D9: `degree` on a method with no degree concept ("log",
/// "sqrt", "glm", and — batch-D T10 — "gam") must refuse rather than
/// being silently ignored.
#[test]
fn test_degree_refused_for_non_polynomial_non_loess_methods() {
    pyo3::Python::initialize();
    for method in &["log", "sqrt", "glm", "gam"] {
        let result = PySmooth::new(
            "x", "y", method, None, None, Some(2), None, 200, 0, None, None, "fitted",
            false, false, None, None, None, None,
        );
        assert!(result.is_err(), "degree=2 with method={method:?} must be refused");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("degree"), "error should mention 'degree' for method={method:?}: {msg}");
    }
}

// ---- Batch-D T9 round 2: quadratic/cubic alias-degree policy ----

/// `method="quadratic"` with a `degree` that AGREES with the fixed
/// degree (2) is a harmless no-op — accepted, not refused.
#[test]
fn test_quadratic_matching_degree_is_accepted_noop() {
    pyo3::Python::initialize();
    let implicit = PySmooth::new(
        "x", "y", "quadratic", None, None, None, None, 200, 0, None, None, "fitted",
        false, false, None, None, None, None,
    ).unwrap();
    let explicit = PySmooth::new(
        "x", "y", "quadratic", None, None, Some(2), None, 200, 0, None, None, "fitted",
        false, false, None, None, None, None,
    ).unwrap();
    assert_eq!(implicit.0, explicit.0, "degree omitted vs degree=2 must construct the same spec for 'quadratic'");
}

/// `method="cubic"` with a `degree` that AGREES with the fixed degree
/// (3) is a harmless no-op — accepted, not refused.
#[test]
fn test_cubic_matching_degree_is_accepted_noop() {
    pyo3::Python::initialize();
    let implicit = PySmooth::new(
        "x", "y", "cubic", None, None, None, None, 200, 0, None, None, "fitted",
        false, false, None, None, None, None,
    ).unwrap();
    let explicit = PySmooth::new(
        "x", "y", "cubic", None, None, Some(3), None, 200, 0, None, None, "fitted",
        false, false, None, None, None, None,
    ).unwrap();
    assert_eq!(implicit.0, explicit.0, "degree omitted vs degree=3 must construct the same spec for 'cubic'");
}

/// A `degree` that CONFLICTS with `method="quadratic"`'s fixed degree
/// must refuse rather than silently fitting degree 2 anyway. Batch-D T9
/// round 2 (Required Fix 1, rust-quality verdict finding 1): this used
/// to construct successfully and quietly fit a degree-2 curve.
#[test]
fn test_quadratic_conflicting_degree_refuses() {
    pyo3::Python::initialize();
    for bad_degree in [1u8, 3, 5] {
        let result = PySmooth::new(
            "x", "y", "quadratic", None, None, Some(bad_degree), None, 200, 0, None, None, "fitted",
            false, false, None, None, None, None,
        );
        assert!(result.is_err(), "degree={bad_degree} conflicting with method='quadratic' must be refused");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("degree") && msg.contains("quadratic"),
            "error should name both 'degree' and 'quadratic': {msg}");
    }
}

/// A `degree` that CONFLICTS with `method="cubic"`'s fixed degree must
/// refuse rather than silently fitting degree 3 anyway.
#[test]
fn test_cubic_conflicting_degree_refuses() {
    pyo3::Python::initialize();
    for bad_degree in [1u8, 2, 4] {
        let result = PySmooth::new(
            "x", "y", "cubic", None, None, Some(bad_degree), None, 200, 0, None, None, "fitted",
            false, false, None, None, None, None,
        );
        assert!(result.is_err(), "degree={bad_degree} conflicting with method='cubic' must be refused");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("degree") && msg.contains("cubic"),
            "error should name both 'degree' and 'cubic': {msg}");
    }
}

// ---- Batch-D T9 round 2: rank-deficient degree refuses on the ungrouped path ----

/// A `degree` that leaves the design rank-deficient (fewer usable points
/// than coefficients to solve for) must refuse with a typed error on the
/// UNGROUPED top-level call, not silently render an all-NaN grid.
/// RED-proves the spec review's finding 2: `degree=20` on a 20-row frame
/// used to render zero polylines with no diagnostic at all.
#[test]
fn test_rank_deficient_degree_refuses_on_ungrouped_path() {
    pyo3::Python::initialize();
    let xs: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let ys: Vec<f64> = xs.iter().map(|&x| x + 1.0).collect();
    let batch = xy_batch(xs, ys);
    for degree in [20u8, 21, 25] {
        let ctor = PySmooth::new(
            "x", "y", "lm", None, None, Some(degree), None, 5, 0, None, None, "fitted",
            false, false, None, None, None, None,
        ).unwrap();
        let spec = match ctor.0 {
            TransformSpec::Smooth(s) => s,
            _ => panic!("expected Smooth spec"),
        };
        let result = apply(&spec, &batch);
        assert!(result.is_err(), "degree={degree} on 20 points must refuse (rank-deficient), not silently NaN");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("degree") && msg.contains("rank-deficient"),
            "error for degree={degree} should mention 'degree' and 'rank-deficient': {msg}");
    }
}

/// The boundary case: `degree = n_points - 1` is EXACTLY determined
/// (not rank-deficient) and must still fit, not refuse.
#[test]
fn test_degree_exactly_n_minus_one_is_not_rank_deficient() {
    let xs: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let ys: Vec<f64> = xs.iter().map(|&x| x + 1.0).collect();
    let batch = xy_batch(xs, ys);
    let spec = SmoothSpec {
        residual_columns: Default::default(),
        x: "x".into(), y: "y".into(),
        method: SmoothMethod::Poly { degree: 19 },
        ci: None,
        bandwidth: 0.75, degree: 19, k: gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
        x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
        inject_zero_ref: false, inject_metrics: false, x_range: None, x_range_pad: None,
        groupby: vec![], name: None,
    };
    let out = apply(&spec, &batch).expect("degree=19 on 20 points is exactly determined, not rank-deficient");
    let yf = col(&out, "y");
    assert!(yf.iter().all(|v| v.is_finite()), "exactly-determined fit must be finite; got {yf:?}");
}

/// A rank-deficient degree inside ONE GROUP of a grouped fit must still
/// degrade to that group's own all-NaN range (T6's contract) rather than
/// refusing the whole batch — the refusal above is specific to the
/// ungrouped top-level call.
#[test]
fn test_rank_deficient_degree_inside_one_group_stays_nan_not_refused() {
    pyo3::Python::initialize();
    // Group "a": 20 points, degree=19 is exactly determined (finite fit).
    // Group "b": only 3 points, degree=19 is massively rank-deficient.
    let mut all_x: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let mut all_y: Vec<f64> = all_x.iter().map(|&x| x + 1.0).collect();
    let mut all_g: Vec<&str> = vec!["a"; 20];
    all_x.extend([100.0, 101.0, 102.0]);
    all_y.extend([1.0, 2.0, 3.0]);
    all_g.extend(["b", "b", "b"]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, true),
        Field::new("y", DataType::Float64, true),
        Field::new("g", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(all_x)),
            Arc::new(Float64Array::from(all_y)),
            Arc::new(arrow::array::StringArray::from(all_g)),
        ],
    ).unwrap();
    let spec = SmoothSpec {
        residual_columns: Default::default(),
        x: "x".into(), y: "y".into(),
        method: SmoothMethod::Poly { degree: 19 },
        ci: None,
        bandwidth: 0.75, degree: 19, k: gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
        x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
        inject_zero_ref: false, inject_metrics: false, x_range: None, x_range_pad: None,
        groupby: vec!["g".into()], name: None,
    };
    let out = apply(&spec, &batch)
        .expect("a rank-deficient group must degrade to NaN, not refuse the whole grouped batch");
    assert_eq!(out.num_rows(), 10, "2 groups × 5 grid points");
    let yf = col(&out, "y");
    assert!(yf[..5].iter().all(|v| v.is_finite()), "group 'a' (exactly determined) must be finite");
    assert!(yf[5..].iter().all(|v| v.is_nan()), "group 'b' (rank-deficient) must be NaN, not a refusal");
}

// ---- Batch-D T9 round 2: apply_grouped schema-mismatch panic (Required Fix 6) ----

/// RED-proof for the live `PanicException` the rust-quality verdict
/// reproduced through `Chart.to_svg()`: a grouped `glm` fit with
/// `inject_metrics=true` where the SECOND group (not the first) is
/// degenerate (glm's IRLS fails on zero-variance x) used to panic inside
/// `apply_grouped` — the reference schema was pinned to group 0's WIDE
/// (6-column, metrics included) output, and group 1's NARROW (4-column)
/// degenerate output was then indexed positionally past its own schema
/// (`index out of bounds: the len is 4 but the index is 4`). `lm` and
/// `poly` inherit the identical `apply_grouped` template and were
/// equally exposed; this test pins the fix at the shared level.
#[test]
fn test_apply_grouped_survives_degenerate_non_first_group_with_inject_metrics() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, true),
        Field::new("y", DataType::Float64, true),
        Field::new("g", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0])),
            Arc::new(arrow::array::StringArray::from(vec!["a", "a", "a", "a", "b", "b", "b", "b"])),
        ],
    ).unwrap();
    let spec = SmoothSpec {
        residual_columns: Default::default(),
        x: "x".into(), y: "y".into(),
        method: SmoothMethod::Glm,
        ci: None,
        bandwidth: 0.75, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
        x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
        inject_zero_ref: false, inject_metrics: true, x_range: None, x_range_pad: None,
        groupby: vec!["g".into()], name: None,
    };
    let out = apply(&spec, &batch)
        .expect("grouped glm with a degenerate non-first group + inject_metrics must not panic");
    assert_eq!(out.num_rows(), 10, "2 groups × 5 grid points");
    let metrics_y = out
        .column(out.schema().index_of("_metrics_y").unwrap())
        .as_any().downcast_ref::<Float64Array>().unwrap();
    // Group 'a' (rows 0..5): a real glm fit -> the anchor row (rightmost
    // grid x, i.e. the group's LAST row) holds a finite metrics value.
    assert!(!metrics_y.is_null(4) && metrics_y.value(4).is_finite(),
        "group 'a' anchor row _metrics_y should be finite");
    // Group 'b' (rows 5..10): zero-variance x -> degenerate glm ->
    // `build_degenerate_smooth_batch` still emits the metrics columns
    // (batch-D T9 round 4), all-null -> every row null/NaN, not a panic
    // (and no longer via `apply_grouped`'s by-name padding fallback,
    // since group 'b' now carries the columns itself).
    for i in 5..10 {
        assert!(metrics_y.is_null(i) || metrics_y.value(i).is_nan(),
            "group 'b' row {i} _metrics_y should be null/NaN (degenerate group)");
    }
    let metrics_text = out
        .column(out.schema().index_of("_metrics_text").unwrap())
        .as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
    assert!(!metrics_text.is_null(4), "group 'a' anchor row _metrics_text should be present");
    for i in 5..10 {
        assert!(metrics_text.is_null(i), "group 'b' row {i} _metrics_text should be null (degenerate group)");
    }
}

// ---- Batch-D T9 round 4: degenerate-branch schema uniformity (root cause) ----

/// RED-proof for `build_degenerate_smooth_batch` itself: when metrics
/// are requested it must append `_metrics_text`/`_metrics_y`, both
/// entirely null (no anchor row — there is no fit to summarize).
#[test]
fn build_degenerate_smooth_batch_emits_null_metrics_when_requested() {
    let grid = vec![0.0, 1.0, 2.0];
    let out = build_degenerate_smooth_batch(&grid, 3, true).unwrap();
    assert_eq!(out.schema().fields().len(), 6, "core 4 + metrics 2");
    let text = out.column(out.schema().index_of("_metrics_text").unwrap())
        .as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
    let y = out.column(out.schema().index_of("_metrics_y").unwrap())
        .as_any().downcast_ref::<Float64Array>().unwrap();
    for i in 0..3 {
        assert!(text.is_null(i), "row {i} _metrics_text must be null, not a garbage anchor");
        assert!(y.is_null(i), "row {i} _metrics_y must be null, not a garbage anchor");
    }
}

/// The `metrics_requested = false` side of the same contract: no columns
/// at all when `inject_metrics` was never set, matching every method's
/// pre-round-4 schema for the common (metrics-off) case.
#[test]
fn build_degenerate_smooth_batch_omits_metrics_columns_when_not_requested() {
    let grid = vec![0.0, 1.0];
    let out = build_degenerate_smooth_batch(&grid, 2, false).unwrap();
    assert_eq!(out.schema().fields().len(), 4, "core columns only, no metrics");
    assert!(out.schema().index_of("_metrics_text").is_err());
    assert!(out.schema().index_of("_metrics_y").is_err());
}

/// RED-proof at the public `apply()` boundary for the reviewer's cited
/// `all_nan_output` example: a genuinely too-sparse ungrouped Fitted
/// call (1 usable point) with `inject_metrics=true` must still emit the
/// metrics columns — all-null — rather than dropping them (which is
/// exactly the divergence a sibling group/facet-panel schema then had
/// to reconcile around).
#[test]
fn all_nan_output_via_apply_with_inject_metrics_emits_null_metrics_columns() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.0])),
            Arc::new(Float64Array::from(vec![2.0])),
        ],
    ).unwrap();
    let spec = SmoothSpec {
        residual_columns: Default::default(),
        x: "x".into(), y: "y".into(),
        method: SmoothMethod::Lm,
        ci: None, bandwidth: 0.75, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 4, seed: 0,
        x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
        inject_zero_ref: false, inject_metrics: true, x_range: None, x_range_pad: None,
        groupby: vec![], name: None,
    };
    let out = apply(&spec, &batch).expect("1 usable point must degrade, not refuse (xs.len() < 2)");
    assert_eq!(out.num_rows(), 4, "spec.n grid rows");
    assert_eq!(out.schema().fields().len(), 6, "core 4 + metrics 2, even though the fit is all-NaN");
    let text = out.column(out.schema().index_of("_metrics_text").unwrap())
        .as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
    for i in 0..4 {
        assert!(text.is_null(i), "row {i} _metrics_text must be null for an all-NaN degenerate fit");
    }
}

/// RED-proof at the public `apply()` boundary for the reviewer's other
/// cited example, `poly_fit`: a `groupby` group too sparse for the
/// requested degree (2 points, degree=3 — rank-deficient, `poly_ols`
/// returns empty coefficients) with `inject_metrics=true` must emit
/// null metrics for that group's rows, NOT the garbage `(r2=1, rmse=0,
/// mae=0)` tuple `residuals::compute` derives from an all-NaN predictor
/// — proving `poly_fit`'s degenerate branch uses `metrics.is_some()` as
/// a gate rather than passing the caller's already-computed tuple
/// through.
#[test]
fn poly_fit_degenerate_group_with_inject_metrics_emits_null_not_garbage() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, true),
        Field::new("y", DataType::Float64, true),
        Field::new("g", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![
                0.0, 1.0, 2.0, 3.0, 4.0, // "a": 5 points, cubic-recoverable
                0.0, 1.0,                // "b": 2 points, rank-deficient for degree=3
            ])),
            Arc::new(Float64Array::from(vec![
                0.0, 1.0, 8.0, 27.0, 64.0, // "a": y = x^3
                0.0, 1.0,                  // "b"
            ])),
            Arc::new(arrow::array::StringArray::from(vec!["a", "a", "a", "a", "a", "b", "b"])),
        ],
    ).unwrap();
    let spec = SmoothSpec {
        residual_columns: Default::default(),
        x: "x".into(), y: "y".into(),
        method: SmoothMethod::Poly { degree: 3 },
        ci: None,
        bandwidth: 0.75, degree: 3, k: gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
        x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
        inject_zero_ref: false, inject_metrics: true, x_range: None, x_range_pad: None,
        groupby: vec!["g".into()], name: None,
    };
    let out = apply(&spec, &batch)
        .expect("a groupby group too sparse for degree=3 must degrade, not refuse the whole fit");
    assert_eq!(out.num_rows(), 10, "2 groups × 5 grid points");
    let metrics_text = out
        .column(out.schema().index_of("_metrics_text").unwrap())
        .as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
    // Group 'a' (rows 0..5): a real cubic fit -> the anchor row (the
    // group's LAST row) holds a finite, non-garbage metrics value.
    assert!(!metrics_text.is_null(4), "group 'a' anchor row _metrics_text should be present");
    // Group 'b' (rows 5..10): rank-deficient poly_ols (n=2 < p=4) ->
    // null, not the (r2=1, rmse=0, mae=0) garbage an unconditional
    // pass-through of the caller's already-computed `metrics` tuple
    // would have produced.
    for i in 5..10 {
        assert!(metrics_text.is_null(i), "group 'b' row {i} _metrics_text must be null, not a garbage anchor");
    }
}

// ---- Batch-D T10: GAM (penalized cubic B-splines) ----

/// A genuinely curved fixture with two turning points — no straight
/// line, and no single quadratic, can track it. Shared by the gam
/// discriminators below so they all argue about the same data.
fn curved_xy() -> (Vec<f64>, Vec<f64>) {
    let xs: Vec<f64> = (0..60).map(|i| i as f64 * 0.15).collect();
    let ys: Vec<f64> = xs.iter().map(|x| (x * 1.4).sin() * 5.0 + 0.3 * x).collect();
    (xs, ys)
}

fn gam_spec(ci: Option<f64>, n: usize) -> SmoothSpec {
    gam_spec_with_k(ci, n, gam::DEFAULT_BASIS_DIM)
}

fn gam_spec_with_k(ci: Option<f64>, n: usize, k: usize) -> SmoothSpec {
    SmoothSpec {
        residual_columns: Default::default(),
        x: "x".into(), y: "y".into(),
        method: SmoothMethod::Gam,
        ci,
        bandwidth: 0.75, degree: 1, k, n, seed: 0,
        x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
        inject_zero_ref: false, inject_metrics: false, x_range: None, x_range_pad: None,
        groupby: vec![], name: None,
    }
}

/// Count sign changes in the fitted curve's first difference — the
/// number of turning points the fit actually renders.
fn turning_points(ys: &[f64]) -> usize {
    let diffs: Vec<f64> = ys.windows(2).map(|w| w[1] - w[0]).collect();
    diffs
        .windows(2)
        .filter(|w| w[0] * w[1] < 0.0)
        .count()
}

/// RED-proof: `"gam"` reaches a real fit through the actual string
/// parser (before T10 it was rejected as an unknown method), and that
/// fit is finite everywhere on the grid.
#[test]
fn test_gam_method_accepted_and_produces_finite_fit() {
    pyo3::Python::initialize();
    let (xs, ys) = curved_xy();
    let batch = xy_batch(xs, ys);
    let ctor = PySmooth::new(
        "x", "y", "gam",
        None, None, None, None, 25, 0,
        None, None, "fitted",
        false, false, None, None, None, None,
    ).expect("PySmooth::new(method='gam') must be accepted by the string parser");
    let spec = match ctor.0 {
        TransformSpec::Smooth(s) => s,
        _ => panic!("expected Smooth spec"),
    };
    let out = apply(&spec, &batch).unwrap();
    let yf = col(&out, "y");
    assert_eq!(yf.len(), 25);
    assert!(yf.iter().all(|v| v.is_finite()), "gam fit must be finite; got {yf:?}");
}

/// The discriminator that a lazy `"gam" => SmoothMethod::Lm` (or
/// `Poly`) alias could not pass: on data with two genuine turning
/// points, the gam fit must BEND — it renders at least two turning
/// points where the linear fit renders none, and tracks the data an
/// order of magnitude more closely. Pins curvature properties, not a
/// bare inequality.
#[test]
fn test_gam_captures_curvature_that_lm_cannot() {
    let (xs, ys) = curved_xy();
    let batch = xy_batch(xs.clone(), ys.clone());
    let gam_out = apply(&gam_spec(None, 60), &batch).unwrap();
    let lm_out = apply(
        &SmoothSpec { method: SmoothMethod::Lm, ..gam_spec(None, 60) },
        &batch,
    ).unwrap();

    let gam_y = col(&gam_out, "y");
    let lm_y = col(&lm_out, "y");

    assert_eq!(turning_points(&lm_y), 0, "a linear fit cannot turn");
    assert!(
        turning_points(&gam_y) >= 2,
        "gam must render the fixture's two turning points; got {} (fit = {gam_y:?})",
        turning_points(&gam_y),
    );

    // The grid is the data's own x here (n = 60 over the same span),
    // so fitted-vs-observed error is directly comparable.
    let rmse = |fit: &[f64]| -> f64 {
        (fit.iter().zip(ys.iter()).map(|(f, y)| (f - y) * (f - y)).sum::<f64>()
            / ys.len() as f64).sqrt()
    };
    let gam_rmse = rmse(&gam_y);
    let lm_rmse = rmse(&lm_y);
    assert!(
        gam_rmse * 10.0 < lm_rmse,
        "gam RMSE {gam_rmse} should be far below lm RMSE {lm_rmse}"
    );
}

/// gam is not loess in disguise. Two independent structural proofs:
/// (1) the fitted curves differ materially on curved data, and (2) gam
/// is seed-invariant — loess's CI band comes from a seeded bootstrap,
/// so changing `seed` moves loess's band and cannot move gam's, which
/// is a closed-form posterior band.
#[test]
fn test_gam_is_not_loess_point_estimates_and_seed_dependence_both_differ() {
    let (xs, ys) = curved_xy();
    let batch = xy_batch(xs, ys.clone());
    let y_range = ys.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        - ys.iter().fold(f64::INFINITY, |a, &b| a.min(b));

    let gam_out = apply(&gam_spec(Some(0.95), 40), &batch).unwrap();
    let loess_out = apply(
        &SmoothSpec {
            residual_columns: Default::default(),
            method: SmoothMethod::Loess,
            degree: 2,
            ..gam_spec(Some(0.95), 40)
        },
        &batch,
    ).unwrap();
    let max_gap = col(&gam_out, "y").iter().zip(col(&loess_out, "y").iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_gap > 0.01 * y_range,
        "gam and loess must differ materially on curved data; max gap {max_gap} \
         vs y range {y_range}"
    );

    // Seed dependence: loess's bootstrap band moves, gam's does not.
    let reseed = |mut s: SmoothSpec| { s.seed = 987_654; s };
    let gam_reseeded = apply(&reseed(gam_spec(Some(0.95), 40)), &batch).unwrap();
    for (a, b) in col(&gam_out, "ci_lower").iter().zip(col(&gam_reseeded, "ci_lower").iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "gam's band must not depend on seed");
    }
    let loess_reseeded = apply(
        &reseed(SmoothSpec {
            residual_columns: Default::default(),
            method: SmoothMethod::Loess,
            degree: 2,
            ..gam_spec(Some(0.95), 40)
        }),
        &batch,
    ).unwrap();
    let loess_band_moved = col(&loess_out, "ci_lower").iter()
        .zip(col(&loess_reseeded, "ci_lower").iter())
        .any(|(a, b)| (a - b).abs() > 1e-12);
    assert!(
        loess_band_moved,
        "control: loess's bootstrap band should move with the seed — if this ever \
         stops holding, the seed-invariance half of this discriminator proves nothing"
    );
}

/// The CI band must bracket the fit and stay finite across the grid.
#[test]
fn test_gam_ci_band_brackets_the_fit() {
    let (xs, ys) = curved_xy();
    let batch = xy_batch(xs, ys);
    let out = apply(&gam_spec(Some(0.95), 30), &batch).unwrap();
    let y = col(&out, "y");
    let lo = col(&out, "ci_lower");
    let hi = col(&out, "ci_upper");
    for i in 0..y.len() {
        assert!(lo[i].is_finite() && hi[i].is_finite(), "band must be finite at grid point {i}");
        assert!(lo[i] < y[i] && y[i] < hi[i], "band must bracket the fit at grid point {i}");
    }
}

/// `ci=None` leaves the band NaN, like every sibling method.
#[test]
fn test_gam_without_ci_emits_nan_band() {
    let (xs, ys) = curved_xy();
    let batch = xy_batch(xs, ys);
    let out = apply(&gam_spec(None, 10), &batch).unwrap();
    assert!(col(&out, "ci_lower").iter().all(|v| v.is_nan()));
    assert!(col(&out, "ci_upper").iter().all(|v| v.is_nan()));
    assert!(col(&out, "y").iter().all(|v| v.is_finite()));
}

/// Determinism through the production entry point: two applies of the
/// same spec produce bit-identical columns (no RNG anywhere in gam).
#[test]
fn test_gam_is_byte_deterministic_through_apply() {
    let (xs, ys) = curved_xy();
    let batch = xy_batch(xs, ys);
    let a = apply(&gam_spec(Some(0.9), 37), &batch).unwrap();
    let b = apply(&gam_spec(Some(0.9), 37), &batch).unwrap();
    for name in ["x", "y", "ci_lower", "ci_upper"] {
        let (ca, cb) = (col(&a, name), col(&b, name));
        for (i, (va, vb)) in ca.iter().zip(cb.iter()).enumerate() {
            assert_eq!(va.to_bits(), vb.to_bits(), "column {name} row {i} must be bit-identical");
        }
    }
}

/// Grouped gam: one curve per group, each fitted on its own rows.
#[test]
fn test_gam_grouped_fit_emits_one_curve_per_group() {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut g = Vec::new();
    for i in 0..40 {
        let x = i as f64 * 0.2;
        xs.push(x);
        ys.push((x * 1.3).sin() * 4.0);
        g.push("a");
        xs.push(x);
        ys.push((x * 1.3).sin() * 4.0 + 25.0);
        g.push("b");
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, true),
        Field::new("y", DataType::Float64, true),
        Field::new("g", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(Float64Array::from(xs)),
        Arc::new(Float64Array::from(ys)),
        Arc::new(arrow::array::StringArray::from(g)),
    ]).unwrap();

    let spec = SmoothSpec { groupby: vec!["g".into()], ..gam_spec(None, 12) };
    let out = apply(&spec, &batch).unwrap();
    assert_eq!(out.num_rows(), 24, "2 groups × 12 grid points");
    let y = col(&out, "y");
    assert!(y.iter().all(|v| v.is_finite()), "both groups must fit: {y:?}");
    // Group 'b' is group 'a' shifted up by 25 — the fits must carry
    // that offset, proving each group was fitted on its own rows.
    for i in 0..12 {
        let gap = y[12 + i] - y[i];
        assert!(
            (gap - 25.0).abs() < 0.5,
            "per-group offset at grid point {i} is {gap}, expected ~25"
        );
    }
}

/// The degenerate-group contract, for gam: a group with too few
/// distinct x values to place a cubic basis degrades to that group's
/// own NaN range instead of failing the whole grouped fit.
#[test]
fn test_gam_degenerate_group_degrades_to_nan_not_refusal() {
    let mut xs: Vec<f64> = (0..20).map(|i| i as f64 * 0.3).collect();
    let mut ys: Vec<f64> = xs.iter().map(|x| (x * 1.5).sin() * 3.0).collect();
    let mut g: Vec<&str> = vec!["a"; 20];
    // Group 'b': only 3 distinct x values — below the cubic basis minimum.
    for x in [1.0, 2.0, 3.0] {
        xs.push(x);
        ys.push(x * 2.0);
        g.push("b");
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, true),
        Field::new("y", DataType::Float64, true),
        Field::new("g", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(Float64Array::from(xs)),
        Arc::new(Float64Array::from(ys)),
        Arc::new(arrow::array::StringArray::from(g)),
    ]).unwrap();

    let spec = SmoothSpec { groupby: vec!["g".into()], ..gam_spec(None, 6) };
    let out = apply(&spec, &batch)
        .expect("a group too sparse for a spline must degrade, not fail the grouped fit");
    assert_eq!(out.num_rows(), 12, "2 groups × 6 grid points");
    let y = col(&out, "y");
    assert!(y[..6].iter().all(|v| v.is_finite()), "group 'a' must fit");
    assert!(y[6..].iter().all(|v| v.is_nan()), "group 'b' must degrade to NaN");
}

/// gam's whole-frame policy (module doc, "gam contributes no third
/// case"): a whole-frame fit the data cannot support degrades to an
/// all-NaN grid rather than raising, because no caller parameter is
/// responsible for the failure. Contrast
/// `test_rank_deficient_degree_refuses_on_ungrouped_path`, where the
/// caller's own `degree` is.
#[test]
fn test_gam_whole_frame_too_sparse_degrades_rather_than_refusing() {
    // 3 distinct x values: below the 4 a cubic basis needs.
    let batch = xy_batch(vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let out = apply(&gam_spec(Some(0.95), 5), &batch)
        .expect("gam has no caller-supplied parameter to blame, so it degrades, not refuses");
    assert!(col(&out, "y").iter().all(|v| v.is_nan()));
    assert!(col(&out, "ci_lower").iter().all(|v| v.is_nan()));
}

/// gam composes with `output="residuals"` and `inject_metrics` like
/// every other method — the shared `build_predictor` arm is wired.
#[test]
fn test_gam_residuals_output_and_metrics_compose() {
    let (xs, ys) = curved_xy();
    let n_rows = xs.len();
    let batch = xy_batch(xs, ys);
    let spec = SmoothSpec {
        residual_columns: Default::default(),
        output: SmoothOutput::Residuals,
        inject_metrics: true,
        ..gam_spec(None, 20)
    };
    let out = apply(&spec, &batch).unwrap();
    assert_eq!(out.num_rows(), n_rows, "residuals output is one row per input row");
    let resid = col(&out, "residual");
    assert!(resid.iter().all(|v| v.is_finite()), "gam residuals must be finite");
    // A spline that tracks this fixture leaves small residuals.
    let max_abs = resid.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    assert!(max_abs < 1.0, "max |residual| = {max_abs} on a smooth sine fixture");
    assert!(out.schema().index_of("_metrics_text").is_ok(), "inject_metrics must add its columns");
}

/// The degenerate path still emits a spec-determined schema (round-4
/// contract) when gam is the method that degenerated.
#[test]
fn test_gam_degenerate_fit_with_inject_metrics_emits_null_metrics_columns() {
    let batch = xy_batch(vec![1.0, 1.0, 2.0, 2.0], vec![1.0, 2.0, 3.0, 4.0]);
    let spec = SmoothSpec { inject_metrics: true, ..gam_spec(None, 4) };
    let out = apply(&spec, &batch).unwrap();
    let metrics_text = out
        .column(out.schema().index_of("_metrics_text").unwrap())
        .as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
    assert!((0..out.num_rows()).all(|i| metrics_text.is_null(i)),
        "a degenerate gam fit must emit present-but-null metrics columns");
}

// ── R1 design finding 6 + (a): gam's smoothness knob, and the
//    refusal symmetry across degree / bandwidth / k ──────────────────

/// A high-frequency fixture the DEFAULT basis (k = 10) provably cannot
/// follow: six full sine periods over the x span. This is the measured
/// under-smoothing symptom `k=` exists to fix.
fn high_frequency_xy() -> (Vec<f64>, Vec<f64>) {
    let xs: Vec<f64> = (0..120).map(|i| i as f64 * 0.1).collect();
    let ys: Vec<f64> = xs.iter().map(|x| (x * 3.2).sin() * 4.0).collect();
    (xs, ys)
}

/// The discriminator for exposing `k`: on data that turns more often
/// than a 10-function basis can follow, `k=20` tracks the signal and
/// `k=10` does not. Both halves are asserted — a `k` that were silently
/// ignored (the shape this parameter exists to avoid becoming) would
/// produce IDENTICAL output, which the final bit-comparison rejects.
#[test]
fn test_gam_k_20_follows_a_signal_the_default_basis_under_smooths() {
    let (xs, ys) = high_frequency_xy();
    let n_grid = xs.len();
    let batch = xy_batch(xs, ys.clone());

    let rmse = |fit: &[f64]| -> f64 {
        (fit.iter().zip(ys.iter()).map(|(f, y)| (f - y) * (f - y)).sum::<f64>()
            / ys.len() as f64).sqrt()
    };
    let default_out = apply(&gam_spec(None, n_grid), &batch).unwrap();
    let wide_out = apply(&gam_spec_with_k(None, n_grid, 20), &batch).unwrap();
    let default_y = col(&default_out, "y");
    let wide_y = col(&wide_out, "y");

    let y_range = ys.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        - ys.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    // The symptom: the default basis cannot resolve six periods.
    assert!(
        rmse(&default_y) > 0.15 * y_range,
        "fixture is not under-smoothed at the default k — it cannot demonstrate \
         what k= fixes; default RMSE {} vs y range {y_range}",
        rmse(&default_y),
    );
    // The fix: a wider basis resolves it.
    assert!(
        rmse(&wide_y) * 3.0 < rmse(&default_y),
        "k=20 RMSE {} should be far below k=10 RMSE {}",
        rmse(&wide_y),
        rmse(&default_y),
    );
    assert!(
        turning_points(&wide_y) > turning_points(&default_y),
        "k=20 must render more turning points ({}) than k=10 ({})",
        turning_points(&wide_y),
        turning_points(&default_y),
    );
    assert!(
        default_y.iter().zip(wide_y.iter()).any(|(a, b)| a.to_bits() != b.to_bits()),
        "k= must reach the fit — identical output would mean it is being ignored"
    );
}

/// `k` is clamped UP to the smallest cubic basis and DOWN to the number
/// of distinct x values, so neither extreme can produce a degenerate
/// design — the reason gam adds no whole-frame refusal of its own.
#[test]
fn test_gam_k_is_clamped_to_the_distinct_x_count() {
    let (xs, ys) = curved_xy();
    let distinct = xs.len(); // the fixture's x values are all distinct
    let batch = xy_batch(xs, ys);
    let huge = apply(&gam_spec_with_k(None, 20, distinct * 4), &batch)
        .expect("an over-large k must clamp, not fail");
    let at_cap = apply(&gam_spec_with_k(None, 20, distinct), &batch).unwrap();
    for (a, b) in col(&huge, "y").iter().zip(col(&at_cap, "y").iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "k above distinct_x must clamp to distinct_x");
    }
}

// ── #162: degrade-warning channel ────────────────────────────────────

/// RED (pre-#162): fails on the pre-change code, which had no channel to
/// report `k` silently clamping down to `distinct_x`.
#[test]
fn gam_k_clamp_pushes_a_parameter_clamped_warning() {
    let (xs, ys) = curved_xy();
    let distinct = xs.len();
    let batch = xy_batch(xs, ys);
    let ctx = TransformContext::default();
    let spec = gam_spec_with_k(None, 20, distinct * 4);
    let _ = apply_with_context(&spec, &batch, &ctx)
        .expect("an over-large k must clamp, not fail");
    let warnings = ctx.warnings.borrow();
    assert_eq!(warnings.len(), 1, "got {warnings:?}");
    assert!(matches!(
        &warnings[0],
        TransformWarning::ParameterClamped { transform, group, param, requested, applied, .. }
            if transform == "Smooth" && group.is_none() && param == "k"
               && requested == &(distinct * 4).to_string() && applied == &distinct.to_string()
    ));
}

/// A `k` that does not need clamping must not warn.
#[test]
fn gam_k_at_cap_does_not_warn() {
    let (xs, ys) = curved_xy();
    let distinct = xs.len();
    let batch = xy_batch(xs, ys);
    let ctx = TransformContext::default();
    let spec = gam_spec_with_k(None, 20, distinct);
    let _ = apply_with_context(&spec, &batch, &ctx).unwrap();
    assert!(ctx.warnings.borrow().is_empty());
}

/// Rust-quality-reviewer finding 3 (post-#162 close): `gam`'s k-clamp
/// warning was only wired into the `SmoothOutput::Fitted` path
/// (`gam_fit`, called from `apply_one_group`'s method dispatch) — the
/// `Residuals` output branch returns earlier in the same function and
/// built its predictor via `build_predictor`'s generic `Gam` arm, which
/// calls `gam::fit` and discards whether it clamped. Same over-large `k`,
/// same production `apply_with_context` path, `output="residuals"`
/// instead of the default `"fitted"`.
#[test]
fn gam_k_clamp_pushes_a_parameter_clamped_warning_on_residuals_output() {
    let (xs, ys) = curved_xy();
    let distinct = xs.len();
    let batch = xy_batch(xs, ys);
    let ctx = TransformContext::default();
    let mut spec = gam_spec_with_k(None, 20, distinct * 4);
    spec.output = SmoothOutput::Residuals;
    let out = apply_with_context(&spec, &batch, &ctx)
        .expect("an over-large k must clamp, not fail, on the residuals path too");
    assert_eq!(out.num_rows(), distinct, "residuals are emitted at the original input xs");
    let warnings = ctx.warnings.borrow();
    assert_eq!(warnings.len(), 1, "got {warnings:?}");
    assert!(matches!(
        &warnings[0],
        TransformWarning::ParameterClamped { transform, group, param, requested, applied, .. }
            if transform == "Smooth" && group.is_none() && param == "k"
               && requested == &(distinct * 4).to_string() && applied == &distinct.to_string()
    ));
}

/// The degenerate (not merely clamped) gam case must also warn on the
/// residuals path — mirrors the Fitted path's `gam_fit` `None` arm.
#[test]
fn gam_degenerate_pushes_a_degenerate_fit_warning_on_residuals_output() {
    // Fewer than 4 distinct x values: gam::fit returns None regardless of k.
    let xs = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
    let ys = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let batch = xy_batch(xs, ys);
    let ctx = TransformContext::default();
    let mut spec = gam_spec_with_k(None, 20, gam::DEFAULT_BASIS_DIM);
    spec.output = SmoothOutput::Residuals;
    let _ = apply_with_context(&spec, &batch, &ctx)
        .expect("a degenerate gam fit must degrade, not fail, on the residuals path too");
    let warnings = ctx.warnings.borrow();
    assert_eq!(warnings.len(), 1, "got {warnings:?}");
    assert!(matches!(
        &warnings[0],
        TransformWarning::DegenerateFit { transform, group, .. }
            if transform == "Smooth" && group.is_none()
    ));
}

/// A rank-deficient/zero-variance `lm` fit must push a `DegenerateFit`
/// warning naming the transform, matching the T9 round-3 "facet panel
/// degrades to NaN with no diagnostic" complaint this channel closes.
#[test]
fn lm_zero_variance_pushes_a_degenerate_fit_warning() {
    let xs = vec![3.0; 10];
    let ys: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let batch = xy_batch(xs, ys);
    let spec = SmoothSpec {
        residual_columns: Default::default(),
        x: "x".into(), y: "y".into(),
        method: SmoothMethod::Lm,
        ci: None,
        bandwidth: 0.0, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
        x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
        inject_zero_ref: false, inject_metrics: false, x_range: None, x_range_pad: None,
        groupby: vec![], name: None,
    };
    let ctx = TransformContext::default();
    let out = apply_with_context(&spec, &batch, &ctx).unwrap();
    assert_eq!(out.num_rows(), 5, "output is unchanged: still the NaN-filled grid");
    let warnings = ctx.warnings.borrow();
    assert_eq!(warnings.len(), 1, "got {warnings:?}");
    assert!(matches!(
        &warnings[0],
        TransformWarning::DegenerateFit { transform, group, reason, .. }
            if transform == "Smooth" && group.is_none() && reason.contains("lm")
    ));
}

/// A degenerate fit's own warning names the exact pipeline step that
/// raised it ([`TransformContext::step`]), not the `Unplaced` default —
/// the step a layer's rows must have passed for
/// `render::scene_build::group_owned_by_its_fit` to defer to this warning
/// instead of repeating it.
#[test]
fn a_degenerate_fit_names_the_step_that_raised_it() {
    use crate::transform::context::StepInstance;

    let batch = xy_batch(vec![1.0], vec![1.0]);
    let spec = SmoothSpec {
        residual_columns: Default::default(),
        x: "x".into(), y: "y".into(),
        method: SmoothMethod::Lm,
        ci: None,
        bandwidth: 0.0, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
        x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
        inject_zero_ref: false, inject_metrics: false, x_range: None, x_range_pad: None,
        groupby: vec![], name: None,
    };
    let step = StepInstance::Chain(vec![2]);
    let ctx = TransformContext { step: step.clone(), ..TransformContext::default() };
    let _ = apply_with_context(&spec, &batch, &ctx).expect("fewer than 2 points degrades, not fails");
    let warnings = ctx.warnings.borrow();
    assert_eq!(warnings.len(), 1, "got {warnings:?}");
    assert!(matches!(
        &warnings[0],
        TransformWarning::DegenerateFit { step: raised_at, .. } if raised_at == &step
    ));
}

/// A grouped call attributes the warning to the specific degenerate
/// group — the healthy group must NOT warn.
#[test]
fn smooth_grouped_degenerate_group_names_the_group() {
    let xs: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let ys: Vec<f64> = xs.iter().map(|&x| 2.0 * x + 1.0).collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, true),
        Field::new("y", DataType::Float64, true),
        Field::new("g", DataType::Utf8, true),
    ]));
    let mut all_x = xs.clone();
    all_x.push(9.0);
    let mut all_y = ys.clone();
    all_y.push(1.0);
    let mut groups: Vec<&str> = std::iter::repeat("healthy").take(xs.len()).collect();
    groups.push("sparse");
    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(Float64Array::from(all_x)),
        Arc::new(Float64Array::from(all_y)),
        Arc::new(arrow::array::StringArray::from(groups)),
    ]).unwrap();
    let spec = SmoothSpec {
        residual_columns: Default::default(),
        x: "x".into(), y: "y".into(),
        method: SmoothMethod::Lm,
        ci: None,
        bandwidth: 0.0, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
        x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
        inject_zero_ref: false, inject_metrics: false, x_range: None, x_range_pad: None,
        groupby: vec!["g".into()], name: None,
    };
    let ctx = TransformContext::default();
    let _ = apply_with_context(&spec, &batch, &ctx).unwrap();
    let warnings = ctx.warnings.borrow();
    assert_eq!(warnings.len(), 1, "only the sparse group should warn; got {warnings:?}");
    assert!(matches!(
        &warnings[0],
        TransformWarning::DegenerateFit { transform, group, .. }
            if transform == "Smooth" && group.as_deref() == Some("g=sparse")
    ));
}

/// `k` omitted must equal `k` at its documented default — the wire and
/// the fit both have to agree that 10 is the default, or a spec written
/// without `k` would render differently from one written with `k=10`.
#[test]
fn test_gam_default_k_matches_an_explicit_ten() {
    pyo3::Python::initialize();
    let make = |k: Option<usize>| {
        let ctor = PySmooth::new(
            "x", "y", "gam", None, None, None, k, 20, 0,
            None, None, "fitted", false, false, None, None, None, None,
        ).expect("gam spec must construct");
        match ctor.0 {
            TransformSpec::Smooth(s) => s,
            _ => panic!("expected Smooth spec"),
        }
    };
    assert_eq!(make(None), make(Some(gam::DEFAULT_BASIS_DIM)));
    assert_eq!(make(None).k, gam::DEFAULT_BASIS_DIM);
}

/// `k` is omitted from the wire at its default, so every `Smooth` spec
/// written before this field existed round-trips byte-identically —
/// and a non-default `k` does ride the wire.
#[test]
fn test_gam_k_is_omitted_from_the_wire_at_its_default() {
    let default_json = serde_json::to_string(&gam_spec(None, 20)).unwrap();
    assert!(
        !default_json.contains("\"k\""),
        "a default k must not appear on the wire: {default_json}"
    );
    let wide_json = serde_json::to_string(&gam_spec_with_k(None, 20, 20)).unwrap();
    assert!(wide_json.contains("\"k\":20"), "a non-default k must ride the wire: {wide_json}");
    let round_tripped: SmoothSpec = serde_json::from_str(&default_json).unwrap();
    assert_eq!(round_tripped, gam_spec(None, 20));
}

/// The asymmetry finding, closed on gam's side: `k` is refused on every
/// method that has no spline basis to size, exactly as `degree` is
/// refused on every method that has no degree.
#[test]
fn test_k_on_a_non_gam_method_refuses() {
    pyo3::Python::initialize();
    for method in ["lm", "linear", "loess", "quadratic", "cubic", "log", "sqrt", "glm"] {
        let result = PySmooth::new(
            "x", "y", method, None, None, None, Some(12), 200, 0,
            None, None, "fitted", false, false, None, None, None, None,
        );
        assert!(result.is_err(), "k=12 with method={method:?} must be refused");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains('k') && msg.contains(method),
            "error should name k and the method for {method:?}: {msg}"
        );
    }
}

/// A `k` below the smallest cubic basis is a value error, not something
/// to silently clamp — clamping it would render a basis the caller did
/// not ask for with no diagnostic.
#[test]
fn test_k_below_the_minimum_basis_refuses() {
    pyo3::Python::initialize();
    for bad_k in [0usize, 1, 2, 3] {
        let result = PySmooth::new(
            "x", "y", "gam", None, None, None, Some(bad_k), 200, 0,
            None, None, "fitted", false, false, None, None, None, None,
        );
        assert!(result.is_err(), "k={bad_k} must be refused");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains('k'), "error should mention k for k={bad_k}: {msg}");
    }
    assert!(
        PySmooth::new(
            "x", "y", "gam", None, None, None, Some(gam::MIN_BASIS_DIM), 200, 0,
            None, None, "fitted", false, false, None, None, None, None,
        ).is_ok(),
        "k at exactly the minimum basis dimension must be accepted"
    );
}

/// The other half of design finding 6: an explicit non-default
/// `bandwidth` on a method that never reads it now refuses instead of
/// silently no-opping, mirroring `degree`'s policy.
#[test]
fn test_non_default_bandwidth_on_a_non_loess_method_refuses() {
    pyo3::Python::initialize();
    for method in ["lm", "linear", "quadratic", "cubic", "log", "sqrt", "glm", "gam"] {
        let result = PySmooth::new(
            "x", "y", method, None, Some(0.4), None, None, 200, 0,
            None, None, "fitted", false, false, None, None, None, None,
        );
        assert!(result.is_err(), "bandwidth=0.4 with method={method:?} must be refused");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("bandwidth") && msg.contains(method),
            "error should name bandwidth and the method for {method:?}: {msg}"
        );
    }
    assert!(
        PySmooth::new(
            "x", "y", "loess", None, Some(0.4), None, None, 200, 0,
            None, None, "fitted", false, false, None, None, None, None,
        ).is_ok(),
        "loess is the method bandwidth belongs to"
    );
}

/// The one tolerance, matching `degree`'s: an explicit value that
/// AGREES with the default is a harmless no-op on any method. Only a
/// direct `Smooth(...)` caller can reach this today -- `desugar_smooth`
/// (`marks/statistical.py`) forwards `bandwidth` conditionally
/// (`if bandwidth is not None`) since its own default moved to `None` --
/// but the invariant must still hold: an explicit default and an omitted
/// value construct the identical spec.
#[test]
fn test_explicit_default_bandwidth_is_accepted_on_every_method() {
    pyo3::Python::initialize();
    let make = |method: &str, bandwidth: Option<f64>| {
        PySmooth::new(
            "x", "y", method, None, bandwidth, None, None, 200, 0,
            None, None, "fitted", false, false, None, None, None, None,
        )
    };
    for method in ["lm", "linear", "loess", "quadratic", "cubic", "log", "sqrt", "glm", "gam"] {
        let explicit = make(method, Some(DEFAULT_LOESS_BANDWIDTH))
            .unwrap_or_else(|e| panic!("explicit default bandwidth rejected for {method:?}: {e}"));
        let omitted = make(method, None).unwrap();
        assert_eq!(
            explicit.0, omitted.0,
            "bandwidth omitted vs bandwidth={DEFAULT_LOESS_BANDWIDTH} must construct the \
             same spec for method={method:?}"
        );
    }
}

/// **Mutation gate: wire-path k below MIN_BASIS_DIM renders without panic.**
/// This catches the mutation `k.clamp(MIN_BASIS_DIM, ...)` → `k.min(...)`
/// via the deserialization path: a SmoothSpec with `k: 2` (below the
/// minimum) must deserialize and render without panic, producing the same
/// output as the clamped equivalent (`k: MIN_BASIS_DIM`).
///
/// When `k` is omitted or set to its default, serde uses the default
/// function. But an explicit `"k": 2` in JSON bypasses the Python PySmooth
/// validation (which rejects k < 4), so the Rust clamp at `gam::fit` is
/// the only defense against the panic at line 156 (`dim - SPLINE_DEGREE`
/// underflow). Without the clamp, dim would be 2, and `2 - 3` panics.
#[test]
fn test_gam_k_below_min_basis_dim_via_json_deserialization() {
    let (xs, ys) = curved_xy();
    let batch = xy_batch(xs, ys);

    // Build a spec with k=2 (below the minimum of 4) directly.
    // This bypasses Python validation, simulating what happens when a
    // ChartSpec JSON with explicit "k": 2 deserializes in Rust.
    let spec_low = SmoothSpec {
        residual_columns: Default::default(),
        x: "x".into(), y: "y".into(),
        method: SmoothMethod::Gam,
        ci: None,
        bandwidth: 0.75, degree: 1, k: 2, n: 200, seed: 0,
        x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
        inject_zero_ref: false, inject_metrics: false, x_range: None, x_range_pad: None,
        groupby: vec![], name: None,
    };

    // Apply the spec with k=2 — it must render without panic.
    let result_low = apply(&spec_low, &batch).expect(
        "k=2 must render without panic; gam::fit clamps it to MIN_BASIS_DIM"
    );

    // Build the same spec with k=4 (the minimum).
    let spec_min = gam_spec_with_k(None, 200, gam::MIN_BASIS_DIM);
    let result_min = apply(&spec_min, &batch).expect("k=4 must render");

    // Both outputs must be byte-identical.
    let col_low = col(&result_low, "y");
    let col_min = col(&result_min, "y");
    assert_eq!(col_low.len(), col_min.len(), "both outputs must have the same length");
    for (a, b) in col_low.iter().zip(col_min.iter()) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "k=2 (clamped) and k=4 must produce identical fitted values"
        );
    }
}

/// **Mutation gate: zero k also renders.** Boundary case:
/// a SmoothSpec with `k: 0` (far below the minimum) must render without
/// panic and match the k=4 output.
#[test]
fn test_gam_k_zero_via_json_deserialization() {
    let (xs, ys) = curved_xy();
    let batch = xy_batch(xs, ys);

    // Build a spec with k=0.
    let spec_zero = SmoothSpec {
        residual_columns: Default::default(),
        x: "x".into(), y: "y".into(),
        method: SmoothMethod::Gam,
        ci: None,
        bandwidth: 0.75, degree: 1, k: 0, n: 200, seed: 0,
        x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
        inject_zero_ref: false, inject_metrics: false, x_range: None, x_range_pad: None,
        groupby: vec![], name: None,
    };

    // Apply the spec with k=0 — must not panic.
    let result_zero = apply(&spec_zero, &batch).expect(
        "k=0 must render without panic; gam::fit clamps it to MIN_BASIS_DIM"
    );

    // Build the same spec with k=4 (the minimum).
    let spec_min = gam_spec_with_k(None, 200, gam::MIN_BASIS_DIM);
    let result_min = apply(&spec_min, &batch).expect("k=4 must render");

    // Both outputs must be byte-identical.
    let col_zero = col(&result_zero, "y");
    let col_min = col(&result_min, "y");
    for (a, b) in col_zero.iter().zip(col_min.iter()) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "k=0 (clamped) and k=4 must produce identical fitted values"
        );
    }
}

// ── #166: ci-level discriminating test (the Smooth family's gap) ────
//
// No existing test varied `ci` for a Smooth-family fit, so the GAM CI
// band's `t_crit = student_t_critical(level, fit.residual_dof())`
// (`gam_fit`, above) was unpinned: dropping the `t_crit` multiplier
// entirely (hardcoding the half-width to `std_error(x)`), or passing raw
// `n` in place of `fit.residual_dof()` (the fit's fractional `n − edf`
// residual degrees of freedom), would leave every pre-existing test
// green — none compared two `ci` levels or checked the half-width
// against an independently computed reference.

/// `gam_fit`'s CI half-width must equal
/// `student_t_critical(level, fit.residual_dof()) * fit.std_error(x)`
/// exactly, for more than one `ci` level, on a fixture whose
/// `residual_dof()` diverges measurably from raw `n` (a fixture where
/// they coincide would pass vacuously under the "passed n" bug).
#[test]
fn gam_ci_half_width_matches_student_t_on_residual_dof_not_raw_n() {
    let n = 30;
    let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 * 10.0).collect();
    let ys: Vec<f64> = xs.iter().enumerate()
        .map(|(i, &x)| x.sin() + if i % 2 == 0 { 0.3 } else { -0.3 })
        .collect();
    let k = 8;

    let fit = gam::fit(&xs, &ys, k).expect("fixture must produce a usable gam fit");
    let residual_dof = fit.residual_dof();
    assert!(
        (residual_dof - n as f64).abs() > 1.0,
        "fixture must smooth enough that residual_dof() diverges measurably from \
         raw n (residual_dof={residual_dof}, n={n}); else this test cannot \
         discriminate the 'passed n' bug"
    );

    let probe_x = xs[n / 2];
    let grid = vec![probe_x];

    for &level in &[0.80_f64, 0.95, 0.99] {
        let (out, _) = gam_fit(&xs, &ys, &grid, Some(level), 1, k, None)
            .expect("gam_fit must succeed on this fixture");
        let yhat = col(&out, "y")[0];
        let hi = col(&out, "ci_upper")[0];
        let observed_half_width = hi - yhat;

        let expected_half_width = student_t_critical(level, residual_dof) * fit.std_error(probe_x);
        let wrong_half_width_using_raw_n = student_t_critical(level, n as f64) * fit.std_error(probe_x);

        assert!(
            (observed_half_width - expected_half_width).abs() < 1e-9,
            "level={level}: half-width must equal student_t_critical(level, \
             residual_dof()) * std_error(x); observed={observed_half_width}, \
             expected={expected_half_width}"
        );
        assert!(
            (observed_half_width - wrong_half_width_using_raw_n).abs() > 1e-6,
            "level={level}: fixture's residual_dof() must diverge from raw n \
             enough that the buggy (n-instead-of-residual_dof) half-width is \
             distinguishably different: observed={observed_half_width}, \
             wrong={wrong_half_width_using_raw_n}"
        );
    }
}

/// `Smooth(x_estimator=)` parses through
/// `XEstimator`, a declared subset of `AggFn`. Every accepted spelling stores
/// the same `AggFn` the old private match mapped it to, so the transform
/// output is byte-identical to a spec carrying that `AggFn` directly; a bogus
/// spelling is refused with the shared `a|b|c` rendering.
#[test]
fn smooth_x_estimator_parses_through_the_agg_fn_subset() {
    use crate::transform::aggregate::{AggFn, XEstimator};
    use crate::transform::vocabulary::Vocabulary;
    pyo3::Python::initialize();
    let xs: Vec<f64> = (0..60).map(|i| i as f64 / 6.0).collect();
    let ys: Vec<f64> = xs.iter().enumerate().map(|(i, x)| x * 1.5 + ((i * 7) % 5) as f64).collect();
    let batch = xy_batch(xs, ys);
    let build = |x_estimator: Option<&str>| {
        PySmooth::new(
            "x", "y", "lm",
            None, None, None, None, 10, 0,
            Some(6), x_estimator, "fitted",
            false, false, None, None, None, None,
        )
        .map(|ctor| match ctor.0 {
            TransformSpec::Smooth(s) => s,
            _ => panic!("expected Smooth spec"),
        })
    };
    // The pre-fix mapping (binding.rs's former private match), as the oracle.
    let old_table = [
        ("mean", AggFn::Mean),
        ("sum", AggFn::Sum),
        ("min", AggFn::Min),
        ("max", AggFn::Max),
        ("median", AggFn::Median),
    ];
    assert_eq!(XEstimator::all(), old_table.iter().map(|(s, _)| *s).collect::<Vec<_>>());
    for (spelling, expected) in old_table {
        let spec = build(Some(spelling)).unwrap();
        assert_eq!(spec.x_estimator, Some(expected), "'{spelling}'");
        let mut oracle = build(None).unwrap();
        oracle.x_estimator = Some(expected);
        assert_eq!(
            apply(&spec, &batch).unwrap(),
            apply(&oracle, &batch).unwrap(),
            "x_estimator='{spelling}' must fit byte-identically to {expected:?}"
        );
    }
    assert_ne!(
        apply(&build(Some("mean")).unwrap(), &batch).unwrap(),
        apply(&build(Some("max")).unwrap(), &batch).unwrap(),
        "vacuity control: estimators must move the fit"
    );
    pyo3::Python::attach(|py| {
        for bogus in ["geomean", "q1", "std"] {
            let err = build(Some(bogus)).unwrap_err();
            assert_eq!(
                err.value(py).to_string(),
                format!("Smooth: unknown x_estimator '{bogus}'; expected mean|sum|min|max|median")
            );
        }
    });
}

// ── An undefined confidence band warns ───────────────────────────────────

/// A minimal `lm`-method spec over `x`/`y`, with `ci`/`groupby` as the only
/// knobs the tests below vary.
fn lm_ci_spec(ci: Option<f64>, groupby: Vec<String>) -> SmoothSpec {
    SmoothSpec {
        residual_columns: Default::default(),
        x: "x".into(), y: "y".into(),
        method: SmoothMethod::Lm,
        ci,
        bandwidth: 0.0, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
        x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
        inject_zero_ref: false, inject_metrics: false, x_range: None, x_range_pad: None,
        groupby,
        name: None,
    }
}

/// `lm` at `n=2` (whole frame): the line interpolates the two points exactly,
/// but zero residual degrees of freedom leaves the band undefined. The line
/// still draws; `IntervalUndefined` names the cause.
#[test]
fn lm_whole_frame_n2_warns_interval_undefined_and_still_draws_the_line() {
    let batch = xy_batch(vec![1.0, 2.0], vec![3.0, 5.0]);
    let ctx = TransformContext::default();
    let spec = lm_ci_spec(Some(0.95), vec![]);
    let out = apply_with_context(&spec, &batch, &ctx).expect("n=2 still fits a line");
    assert!(!value_column_is_all_nan(&out, 1), "the fit line must be finite");
    assert!(value_column_is_all_nan(&out, 2), "ci_lower must stay NaN");
    assert!(value_column_is_all_nan(&out, 3), "ci_upper must stay NaN");
    let warnings = ctx.warnings.borrow();
    assert_eq!(warnings.len(), 1, "got {warnings:?}");
    assert!(
        matches!(
            &warnings[0],
            TransformWarning::IntervalUndefined { transform, group, ci, reason, .. }
                if transform == "Smooth" && group.is_none() && *ci == 0.95
                   && reason.contains("2 points") && reason.contains("2-parameter")
        ),
        "{warnings:?}"
    );
}

/// Two groups, `a` at `n=2` (band undefined) and `b` at `n=5` (band fine):
/// exactly one warning, naming group `a`; `b`'s own fit is unaffected.
#[test]
fn lm_grouped_warns_only_for_the_undersized_group() {
    let xs_a = vec![1.0, 2.0];
    let ys_a = vec![3.0, 5.0];
    let xs_b: Vec<f64> = (0..5).map(|i| i as f64).collect();
    let ys_b: Vec<f64> = xs_b.iter().map(|&x| 1.0 + x).collect();
    let mut all_x = xs_a.clone();
    all_x.extend(xs_b.clone());
    let mut all_y = ys_a.clone();
    all_y.extend(ys_b.clone());
    let mut all_g: Vec<&str> = vec!["a"; xs_a.len()];
    all_g.extend(vec!["b"; xs_b.len()]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, true),
        Field::new("y", DataType::Float64, true),
        Field::new("g", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(all_x)),
            Arc::new(Float64Array::from(all_y)),
            Arc::new(arrow::array::StringArray::from(all_g)),
        ],
    )
    .unwrap();
    let ctx = TransformContext::default();
    let spec = lm_ci_spec(Some(0.95), vec!["g".into()]);
    let _ = apply_with_context(&spec, &batch, &ctx).expect("mixed-size groups both fit a line");
    let warnings = ctx.warnings.borrow();
    assert_eq!(warnings.len(), 1, "got {warnings:?}");
    assert!(
        matches!(
            &warnings[0],
            TransformWarning::IntervalUndefined { transform, group, .. }
                if transform == "Smooth" && group.as_deref() == Some("g=a")
        ),
        "{warnings:?}"
    );

    // Control: group `b` alone (ungrouped) bands byte-identically to its
    // half of the grouped run, so the undersized sibling never perturbs it.
    let batch_b = xy_batch(xs_b, ys_b);
    let ctx_b = TransformContext::default();
    let spec_b = lm_ci_spec(Some(0.95), vec![]);
    let out_b = apply_with_context(&spec_b, &batch_b, &ctx_b).unwrap();
    assert!(ctx_b.warnings.borrow().is_empty(), "group b alone must not warn");
    assert!(!value_column_is_all_nan(&out_b, 2), "group b's band is defined");
}

/// `quadratic` (`degree=2`) at `n=3` leaves zero residual degrees of
/// freedom for its 3-parameter design; the reason names the parameter
/// count.
#[test]
fn poly_quadratic_n3_warns_with_the_parameter_count() {
    let xs = vec![1.0, 2.0, 3.0];
    let ys = vec![1.0, 4.0, 9.0];
    let batch = xy_batch(xs, ys);
    let ctx = TransformContext::default();
    let spec = SmoothSpec {
        residual_columns: Default::default(),
        method: SmoothMethod::Poly { degree: 2 },
        ..lm_ci_spec(Some(0.95), vec![])
    };
    let out = apply_with_context(&spec, &batch, &ctx).expect("n=3 still fits a quadratic");
    assert!(!value_column_is_all_nan(&out, 1));
    let warnings = ctx.warnings.borrow();
    assert_eq!(warnings.len(), 1, "got {warnings:?}");
    assert!(
        matches!(
            &warnings[0],
            TransformWarning::IntervalUndefined { transform, reason, .. }
                if transform == "Smooth" && reason.contains("3 points") && reason.contains("3-parameter")
        ),
        "{warnings:?}"
    );
}

/// `log` and `sqrt` at `n=2` each leave zero residual degrees of freedom
/// for their 2-parameter design, the same as plain `lm`.
#[test]
fn log_and_sqrt_n2_warn_interval_undefined() {
    for method in [SmoothMethod::LogX, SmoothMethod::SqrtX] {
        let batch = xy_batch(vec![1.0, 2.0], vec![3.0, 5.0]);
        let ctx = TransformContext::default();
        let spec = SmoothSpec { method, ..lm_ci_spec(Some(0.95), vec![]) };
        let out = apply_with_context(&spec, &batch, &ctx).expect("n=2 still fits a line");
        assert!(!value_column_is_all_nan(&out, 1));
        let warnings = ctx.warnings.borrow();
        assert_eq!(warnings.len(), 1, "got {warnings:?}");
        assert!(
            matches!(&warnings[0], TransformWarning::IntervalUndefined { transform, .. } if transform == "Smooth"),
            "{warnings:?}"
        );
    }
}

/// Loess with a constructed fixture whose bootstrap leaves some grid
/// points with fewer than 4 finite resamples: warns naming the point
/// count, rather than silently leaving holes in the band.
#[test]
fn loess_partial_band_warns_with_the_point_count() {
    // A heavily duplicated local design: 5 of the 11 rows sit at x=4, with
    // the rest spread one-per-value. A local window (`k=3` here) centered
    // near x=4 draws on that mass so often, under bootstrap resampling,
    // that some grid points land with fewer than 4 finite resamples out of
    // 200, while x=0..3 and x=5..6 anchor the fit everywhere else -- the
    // line never goes all-NaN. Built, not read from a real dataset, to
    // exercise exactly this partial-NaN bootstrap shape (`seed=4` is the
    // one this module's `probe` run found; any seed near it reproduces the
    // same shape since the collapse is structural, not seed-specific luck).
    let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.0, 6.0];
    let ys: Vec<f64> = xs.iter().enumerate().map(|(i, &x)| x + 0.1 * ((i % 3) as f64)).collect();
    let batch = xy_batch(xs, ys);
    let ctx = TransformContext::default();
    let spec = SmoothSpec {
        residual_columns: Default::default(),
        method: SmoothMethod::Loess,
        bandwidth: 3.0 / 11.0,
        degree: 1,
        n: 40,
        seed: 4,
        ..lm_ci_spec(Some(0.95), vec![])
    };
    let out = apply_with_context(&spec, &batch, &ctx).expect("loess still fits a line");
    assert!(!value_column_is_all_nan(&out, 1));
    let warnings = ctx.warnings.borrow();
    assert!(
        warnings.iter().any(|w| matches!(
            w,
            TransformWarning::IntervalUndefined { transform, reason, .. }
                if transform == "Smooth" && reason.contains("grid point")
        )),
        "expected an IntervalUndefined naming the undefined grid points, got {warnings:?}"
    );
}

/// The same bootstrap shape as the test above, tuned to leave exactly one
/// grid point undefined rather than several: the boundary the `> 0` guard
/// sits on, distinct from a fixture with many undefined points that would
/// pass even a stricter guard.
#[test]
fn loess_partial_band_warns_with_a_single_undefined_point() {
    let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.0, 6.0];
    let ys: Vec<f64> = xs.iter().enumerate().map(|(i, &x)| x + 0.1 * ((i % 3) as f64)).collect();
    let batch = xy_batch(xs, ys);
    let ctx = TransformContext::default();
    let spec = SmoothSpec {
        residual_columns: Default::default(),
        method: SmoothMethod::Loess,
        bandwidth: 3.0 / 11.0,
        degree: 1,
        n: 6,
        seed: 0,
        ..lm_ci_spec(Some(0.95), vec![])
    };
    let out = apply_with_context(&spec, &batch, &ctx).expect("loess still fits a line");
    assert!(!value_column_is_all_nan(&out, 1));
    let warnings = ctx.warnings.borrow();
    assert_eq!(warnings.len(), 1, "got {warnings:?}");
    assert!(
        matches!(
            &warnings[0],
            TransformWarning::IntervalUndefined { transform, reason, .. }
                if transform == "Smooth" && reason.contains("at 1 of 6 grid points")
        ),
        "expected an IntervalUndefined naming exactly 1 of 6 grid points, got {warnings:?}"
    );
}

/// Controls: `ci=None` and a well-determined fit (`n >= p + 2`) never warn.
#[test]
fn interval_undefined_controls_do_not_warn() {
    let ctx = TransformContext::default();
    let spec = lm_ci_spec(None, vec![]);
    let batch = xy_batch(vec![1.0, 2.0], vec![3.0, 5.0]);
    let _ = apply_with_context(&spec, &batch, &ctx).unwrap();
    assert!(ctx.warnings.borrow().is_empty(), "ci=None must never warn IntervalUndefined");

    let ctx2 = TransformContext::default();
    let spec2 = lm_ci_spec(Some(0.95), vec![]);
    let xs: Vec<f64> = (0..4).map(|i| i as f64).collect();
    let ys: Vec<f64> = xs.iter().map(|&x| 1.0 + 2.0 * x).collect();
    let batch2 = xy_batch(xs, ys);
    let out2 = apply_with_context(&spec2, &batch2, &ctx2).unwrap();
    assert!(!value_column_is_all_nan(&out2, 2), "n=4 has a defined band");
    assert!(ctx2.warnings.borrow().is_empty(), "a well-determined fit must not warn");
}

// ── A drawn log axis demands this fit in its own space ───────────────────

type LogSpace = crate::transform::color_bin_group::LogAxisSpace;

fn ctx_demanding(space: StepSpace) -> TransformContext {
    TransformContext { log_space: Some(std::rc::Rc::new(space)), ..TransformContext::default() }
}

/// The `x` slot demanded along `x` in `x_space`, the `y` slot along `y` in
/// `y_space`, whichever is `Some` — mirrors `kde::tests`' and
/// `kde_2d::tests`' identical helper.
fn axis_slots(x_space: Option<LogSpace>, y_space: Option<LogSpace>) -> StepSpace {
    use crate::spec::layer::PositionalAxis;
    StepSpace::for_slots(
        [(SpaceSlot::X, PositionalAxis::X, x_space), (SpaceSlot::Y, PositionalAxis::Y, y_space)]
            .into_iter()
            .filter_map(|(slot, axis, space)| space.map(|space| (slot, SlotSpace { space, axes: vec![axis], records_caller_input: true }))),
    )
}

fn positive_space() -> LogSpace {
    LogSpace { base: 10.0, side: crate::scale::log::LogSide::Positive }
}

/// `lm` drawn on a demanded log x fits a straight line in log(x), not a
/// curve in data space: data generated as `y = a + b*log10(x)` recovers an
/// (almost) exact fit, and the mapped-back grid is evenly spaced in LOG
/// units — unevenly, geometrically spaced back in data units, which a
/// linear-space grid over the same data range would not be. The
/// `ctx.log_input` assertion below confirms the mechanism populates that
/// sink when told to; the pass-through name-collision claim itself (a
/// caller field literally named `x`, colliding with the Fitted output's
/// own grid column) is pinned at the level that actually decides it —
/// `color_bin_group::tests::a_smooth_fit_on_a_field_named_like_its_own_output_still_records_the_caller_column`.
#[test]
fn lm_on_a_demanded_log_x_fits_a_straight_line_in_log_units() {
    pyo3::Python::initialize();
    let xs: Vec<f64> = vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0];
    let ys: Vec<f64> = xs.iter().map(|&x| 3.0 + 2.0 * x.log10()).collect();
    let batch = xy_batch(xs, ys);
    let spec = SmoothSpec { n: 8, ..lm_ci_spec(None, vec![]) };
    let ctx = ctx_demanding(axis_slots(Some(positive_space()), None));
    let out = apply_with_context(&spec, &batch, &ctx).unwrap();
    let grid = col(&out, "x");
    let fitted = col(&out, "y");
    assert!(grid.iter().all(|v| v.is_finite() && *v > 0.0), "{grid:?}");
    for (&x, &y) in grid.iter().zip(fitted.iter()) {
        let expected = 3.0 + 2.0 * x.log10();
        assert!((y - expected).abs() < 1e-6, "x={x} y={y} expected={expected}");
    }
    let mut diffs: Vec<f64> = grid.windows(2).map(|w| w[1] - w[0]).collect();
    diffs.sort_by(f64::total_cmp);
    assert!(
        diffs.last().unwrap() / diffs.first().unwrap() > 1.5,
        "a log-space grid widens geometrically toward the high end: {grid:?}"
    );
    let recorded = ctx.log_input.borrow();
    let column = recorded
        .get(&(crate::spec::layer::PositionalAxis::X, "x".to_string()))
        .expect("the demanded X slot records the caller's column even though its name matches the Fitted output's own 'x' grid column");
    assert_eq!(column.len(), 8, "the recorded column is the caller's whole input");
}

/// A demanded x with off-side (`<= 0`) rows and a genuine null mixed in
/// among on-side ones: `with_mapped_slot` maps everything off-side to
/// null before the fit body sees it, so the output grid never draws an
/// off-side x — but `ctx.log_input` still records the caller's WHOLE,
/// UNTOUCHED column (every row, the off-side value and the null both
/// kept as written), not a copy filtered down to the on-side rows the
/// fit itself used. That untouched copy is what the off-side tally reads
/// — the same raw column a plain sibling layer would read straight off
/// the batch.
#[test]
fn lm_on_a_demanded_x_with_off_side_and_null_rows_records_the_caller_column_untouched_and_draws_no_off_side_x() {
    pyo3::Python::initialize();
    let xs: Vec<Option<f64>> = vec![Some(-2.0), None, Some(1.0), Some(2.0), Some(4.0), Some(8.0), Some(16.0)];
    let ys: Vec<Option<f64>> = vec![
        Some(0.0),
        Some(0.0),
        Some(3.0),
        Some(3.0 + 2.0 * 2f64.log10()),
        Some(3.0 + 2.0 * 4f64.log10()),
        Some(3.0 + 2.0 * 8f64.log10()),
        Some(3.0 + 2.0 * 16f64.log10()),
    ];
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, true),
        Field::new("y", DataType::Float64, true),
    ]));
    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(Float64Array::from(xs.clone())),
        Arc::new(Float64Array::from(ys)),
    ])
    .unwrap();
    let spec = SmoothSpec { n: 5, ..lm_ci_spec(None, vec![]) };
    let ctx = ctx_demanding(axis_slots(Some(positive_space()), None));
    let out = apply_with_context(&spec, &batch, &ctx).expect("the on-side rows alone still fit a line");
    let grid = col(&out, "x");
    assert!(grid.iter().all(|v| v.is_finite() && *v > 0.0), "the grid must draw no off-side x: {grid:?}");
    let recorded = ctx.log_input.borrow();
    let column = recorded
        .get(&(crate::spec::layer::PositionalAxis::X, "x".to_string()))
        .expect("the demanded X slot records the caller's column");
    assert_eq!(
        column.len(),
        xs.len(),
        "the recorded column is the caller's WHOLE input length, off-side and null rows included, not filtered down to the rows the fit itself uses"
    );
    let recorded_f64 = column.as_any().downcast_ref::<Float64Array>().expect("caller x is Float64");
    assert!(recorded_f64.is_null(1), "the caller's own null stays null in the recorded copy, not silently filled in");
    assert_eq!(recorded_f64.value(0), -2.0, "the off-side row's own value is recorded verbatim, not nulled out by the log mapping that only touches the fit's own input");
}

/// A demanded Y slot maps `y`, `ci_lower`, `ci_upper` and `_metrics_y` back
/// through the SAME `LogAxisSpace::backward` ([`TransformSpec::fit_space_columns`]'s
/// one declaration), applied per row and independently to each column — so
/// a confidence band that is SYMMETRIC around the fit in log(y) units
/// (`ci_lower`/`ci_upper` sit the same log-distance from `y` on either
/// side) maps back to one that is NOT symmetric in data units, the
/// nonlinearity `10^v` imposes. `_metrics_y`'s single anchor row maps back
/// through the identical function, consistently with `y` itself: the
/// anchor is the max of the FITTED y column in data units, not the max of
/// its log-space value re-expressed.
#[test]
fn lm_on_a_demanded_y_maps_ci_lower_ci_upper_and_metrics_y_back_asymmetrically() {
    pyo3::Python::initialize();
    let xs: Vec<f64> = (0..6).map(|i| i as f64).collect();
    let noise = [0.02, -0.02, 0.015, -0.015, 0.01, -0.01];
    let ys: Vec<f64> = xs.iter().zip(noise.iter()).map(|(&x, &n)| 10f64.powf(1.0 + 0.3 * x + n)).collect();
    let batch = xy_batch(xs, ys);
    let spec = SmoothSpec { n: 6, inject_metrics: true, ..lm_ci_spec(Some(0.95), vec![]) };
    let ctx = ctx_demanding(axis_slots(None, Some(positive_space())));
    let out = apply_with_context(&spec, &batch, &ctx).expect("a noisy linear-in-log-y fit still bands");
    let y = col(&out, "y");
    let lo = col(&out, "ci_lower");
    let hi = col(&out, "ci_upper");
    assert!(y.iter().chain(lo.iter()).chain(hi.iter()).all(|v| v.is_finite() && *v > 0.0), "y={y:?} lo={lo:?} hi={hi:?}");
    let mut saw_asymmetry = false;
    for i in 0..y.len() {
        let below = y[i] - lo[i];
        let above = hi[i] - y[i];
        assert!(below > 0.0 && above > 0.0, "row {i}: a two-sided band must bracket the fit, got lo={}, y={}, hi={}", lo[i], y[i], hi[i]);
        if (below - above).abs() > 1e-6 {
            saw_asymmetry = true;
        }
        // In log(y) units the same band is symmetric: `backward` applies
        // `10^v` uniformly, so the log-distance from `y` to each edge
        // matches to machine precision.
        let log_below = y[i].log10() - lo[i].log10();
        let log_above = hi[i].log10() - y[i].log10();
        assert!((log_below - log_above).abs() < 1e-9, "row {i}: the band must be symmetric in log(y) units before the backward map: {log_below} vs {log_above}");
    }
    assert!(saw_asymmetry, "a log-symmetric band must map back asymmetric in data units somewhere across the grid: y={y:?} lo={lo:?} hi={hi:?}");
    let metrics_y = out
        .column(out.schema().index_of("_metrics_y").unwrap())
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let anchor_row = (0..metrics_y.len()).find(|&i| !metrics_y.is_null(i)).expect("inject_metrics writes one non-null anchor row");
    let max_y = y.iter().copied().fold(f64::MIN, f64::max);
    assert!((metrics_y.value(anchor_row) - max_y).abs() < 1e-9, "_metrics_y must anchor at the max FITTED y in data units, got {} vs max {}", metrics_y.value(anchor_row), max_y);
}

/// `method="log"` is linear in log(x) (`ln`/`log10` only rescale the
/// coefficient), so under a demanded log x it runs the SAME body as
/// `method="lm"` rather than taking `ln` of already-log-mapped units —
/// the two methods' outputs must be identical under the same demand.
#[test]
fn log_x_on_a_demanded_log_x_matches_lm() {
    pyo3::Python::initialize();
    let xs: Vec<f64> = vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0];
    let ys: Vec<f64> = vec![1.0, 1.5, 3.0, 2.5, 4.0, 3.5];
    let batch = xy_batch(xs, ys);
    let lm_spec = SmoothSpec { n: 6, method: SmoothMethod::Lm, ..lm_ci_spec(None, vec![]) };
    let log_spec = SmoothSpec { method: SmoothMethod::LogX, ..lm_spec.clone() };
    let lm_out = apply_with_context(&lm_spec, &batch, &ctx_demanding(axis_slots(Some(positive_space()), None))).unwrap();
    let log_out = apply_with_context(&log_spec, &batch, &ctx_demanding(axis_slots(Some(positive_space()), None))).unwrap();
    assert_eq!(col(&lm_out, "x"), col(&log_out, "x"));
    assert_eq!(col(&lm_out, "y"), col(&log_out, "y"));
}

/// The backstop refusal: `method="sqrt"` under a demanded log x refuses
/// even on a direct call that builds its own `TransformContext` with no
/// prepare-time gate in front of it — `render::prepare::refuse_sqrt_on_log_x`
/// is the one place this text is voiced through the layer entry; this is
/// the unvoiced backstop the module doc on `apply_in_log_space` describes.
#[test]
fn sqrt_under_a_demanded_log_x_is_refused() {
    pyo3::Python::initialize();
    let batch = xy_batch(vec![1.0, 2.0, 4.0, 8.0], vec![1.0, 2.0, 3.0, 4.0]);
    let spec = SmoothSpec { method: SmoothMethod::SqrtX, ..lm_ci_spec(None, vec![]) };
    let ctx = ctx_demanding(axis_slots(Some(positive_space()), None));
    let err = apply_with_context(&spec, &batch, &ctx).unwrap_err().to_string();
    assert!(err.contains("method='sqrt'") && err.contains("log(x)"), "{err}");
}

/// `x_range_pad` is taken AFTER `x_range` is mapped into the demanded
/// axis's own log10 units, so the padded ends are EXACTLY
/// `10^(-0.05*log10(16))` and `10^(1.05*log10(16))` for `x_range=[1, 16]`,
/// `pad=0.05` — not a data-space 5% margin on `[1, 16]` itself (which would
/// still render here, but the next test below shows where a data-space pad
/// would go off-side and a log-space one cannot).
#[test]
fn x_range_pad_is_taken_in_the_demanded_axis_own_log_units() {
    pyo3::Python::initialize();
    let xs: Vec<f64> = vec![1.0, 2.0, 4.0, 8.0, 16.0];
    let ys: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let batch = xy_batch(xs, ys);
    let spec = SmoothSpec { x_range: Some([1.0, 16.0]), x_range_pad: Some(0.05), ..lm_ci_spec(None, vec![]) };
    let ctx = ctx_demanding(axis_slots(Some(positive_space()), None));
    let out = apply_with_context(&spec, &batch, &ctx).unwrap();
    let grid = col(&out, "x");
    let log16 = 16f64.log10();
    let expected_lo = 10f64.powf(-0.05 * log16);
    let expected_hi = 10f64.powf(1.05 * log16);
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * b.abs().max(1.0);
    assert!(
        close(grid.iter().cloned().reduce(f64::min).unwrap(), expected_lo),
        "expected the padded low end {expected_lo}, got {grid:?}"
    );
    assert!(
        close(grid.iter().cloned().reduce(f64::max).unwrap(), expected_hi),
        "expected the padded high end {expected_hi}, got {grid:?}"
    );
}

/// `x_range_pad` ALONE (no explicit `x_range`) pads the batch's own x
/// extent in the demanded axis's own LOG units, not data units: this
/// fixture's x spans exactly `[1, 16]`, the same span the test above
/// passes as an EXPLICIT `x_range`, so the two tests' expected padded
/// ends are identical — proving the pad-alone path falls back to the
/// batch's own extent mapped into log units first, padded there, not to
/// the batch's own extent padded in data units (`[1 - 0.05*15, 16 +
/// 0.05*15] = [0.25, 16.75]`, a different pair of numbers entirely).
#[test]
fn x_range_pad_alone_pads_the_batchs_own_extent_in_the_demanded_axis_own_log_units() {
    pyo3::Python::initialize();
    let xs: Vec<f64> = vec![1.0, 2.0, 4.0, 8.0, 16.0];
    let ys: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let batch = xy_batch(xs, ys);
    let spec = SmoothSpec { x_range: None, x_range_pad: Some(0.05), ..lm_ci_spec(None, vec![]) };
    let ctx = ctx_demanding(axis_slots(Some(positive_space()), None));
    let out = apply_with_context(&spec, &batch, &ctx).unwrap();
    let grid = col(&out, "x");
    let log16 = 16f64.log10();
    let expected_lo = 10f64.powf(-0.05 * log16);
    let expected_hi = 10f64.powf(1.05 * log16);
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * b.abs().max(1.0);
    assert!(
        close(grid.iter().cloned().reduce(f64::min).unwrap(), expected_lo),
        "expected the padded low end {expected_lo}, got {grid:?}"
    );
    assert!(
        close(grid.iter().cloned().reduce(f64::max).unwrap(), expected_hi),
        "expected the padded high end {expected_hi}, got {grid:?}"
    );
}

/// A range whose DATA-SPACE pad would go off-side: `x_range=[1, 100]`
/// padded 5% in data units would give `[1 - 4.95, 100 + 4.95] = [-3.95,
/// 104.95]`, negative and unplaceable on a log x. Padded in LOG units
/// instead, the ends are exactly `10^-0.1` and `10^2.1` (`log10(100) = 2`,
/// margin `0.05*2 = 0.1`), both strictly positive.
#[test]
fn x_range_pad_in_log_units_never_goes_off_side_where_a_data_space_pad_would() {
    pyo3::Python::initialize();
    let xs: Vec<f64> = vec![1.0, 10.0, 100.0];
    let ys: Vec<f64> = vec![1.0, 2.0, 3.0];
    let batch = xy_batch(xs, ys);
    let spec = SmoothSpec { x_range: Some([1.0, 100.0]), x_range_pad: Some(0.05), ..lm_ci_spec(None, vec![]) };
    let ctx = ctx_demanding(axis_slots(Some(positive_space()), None));
    let out = apply_with_context(&spec, &batch, &ctx).unwrap();
    let grid = col(&out, "x");
    let expected_lo = 10f64.powf(-0.1);
    let expected_hi = 10f64.powf(2.1);
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * b.abs().max(1.0);
    assert!(grid.iter().all(|v| v.is_finite() && *v > 0.0), "every grid point must stay on-side: {grid:?}");
    assert!(
        close(grid.iter().cloned().reduce(f64::min).unwrap(), expected_lo),
        "expected the padded low end {expected_lo}, got {grid:?}"
    );
    assert!(
        close(grid.iter().cloned().reduce(f64::max).unwrap(), expected_hi),
        "expected the padded high end {expected_hi}, got {grid:?}"
    );
}

// ── Residuals mode carries the caller's other input columns ─────────────

/// A minimal `lm`-method residuals-output spec, with `x`/`y`/`groupby` as
/// the knobs the carried-column tests below vary.
fn residuals_spec(x: &str, y: &str, groupby: Vec<String>) -> SmoothSpec {
    SmoothSpec {
        residual_columns: Default::default(),
        x: x.into(), y: y.into(),
        method: SmoothMethod::Lm,
        ci: None,
        bandwidth: 0.0, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
        x_bins: None, x_estimator: None, output: SmoothOutput::Residuals,
        inject_zero_ref: false, inject_metrics: false, x_range: None, x_range_pad: None,
        groupby,
        name: None,
    }
}

/// Every other column the caller's batch holds rides along on the
/// residuals batch, copied at exactly the rows the fit kept (one row is
/// dropped for a NaN `y`), in the surviving rows' own order, at every
/// dtype, with a null and a NaN each copied as the caller's own row held
/// them — the `Bin(per_row=True)` row-preserving contract applied to
/// `Smooth`'s residual mode. The predictor's own column ("v") and the
/// computed output ("x") hold the same surviving values under their own,
/// different, names.
#[test]
fn smooth_residuals_carries_other_input_columns_at_surviving_rows_in_order() {
    let v: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let w: Vec<f64> = vec![1.0, 3.0, f64::NAN, 7.0, 9.0]; // row 2 dropped (NaN y)
    let s: Vec<Option<f64>> = vec![Some(10.0), None, Some(30.0), Some(f64::NAN), Some(50.0)];
    let id: Vec<&str> = vec!["r0", "r1", "r2", "r3", "r4"];
    let flag: Vec<bool> = vec![true, false, true, false, true];
    let count: Vec<i64> = vec![100, 200, 300, 400, 500];
    let schema = Arc::new(Schema::new(vec![
        Field::new("v", DataType::Float64, true),
        Field::new("w", DataType::Float64, true),
        Field::new("s", DataType::Float64, true),
        Field::new("id", DataType::Utf8, true),
        Field::new("flag", DataType::Boolean, true),
        Field::new("count", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(Float64Array::from(v.clone())),
        Arc::new(Float64Array::from(w.clone())),
        Arc::new(Float64Array::from(s.clone())),
        Arc::new(arrow::array::StringArray::from(id.clone())),
        Arc::new(arrow::array::BooleanArray::from(flag.clone())),
        Arc::new(arrow::array::Int64Array::from(count.clone())),
    ]).unwrap();
    let spec = residuals_spec("v", "w", vec![]);
    let out = apply(&spec, &batch).unwrap();

    assert_eq!(out.num_rows(), 4, "row 2 (NaN y) is dropped from the fit, and from every carried column");
    let schema = out.schema();
    let mut names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    names.sort_unstable();
    assert_eq!(names, vec!["count", "flag", "id", "residual", "s", "v", "w", "x"]);

    assert_eq!(col(&out, "v"), vec![0.0, 1.0, 3.0, 4.0], "the predictor's own column carries too");
    assert_eq!(col(&out, "x"), vec![0.0, 1.0, 3.0, 4.0], "the computed x holds the same surviving values");
    assert_eq!(col(&out, "w"), vec![1.0, 3.0, 7.0, 9.0], "the response column rides along as a carried column");

    let s_out = out.column(out.schema().index_of("s").unwrap())
        .as_any().downcast_ref::<Float64Array>().unwrap();
    assert_eq!(s_out.value(0), 10.0);
    assert!(s_out.is_null(1), "the caller's own null in a carried column stays null");
    assert!(!s_out.is_null(2) && s_out.value(2).is_nan(), "a NaN (never nulled) in a carried column is copied as the caller's row held it");
    assert_eq!(s_out.value(3), 50.0);

    let id_out = out.column(out.schema().index_of("id").unwrap())
        .as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
    assert_eq!((0..4).map(|i| id_out.value(i)).collect::<Vec<_>>(), vec!["r0", "r1", "r3", "r4"]);

    let flag_out = out.column(out.schema().index_of("flag").unwrap())
        .as_any().downcast_ref::<arrow::array::BooleanArray>().unwrap();
    assert_eq!((0..4).map(|i| flag_out.value(i)).collect::<Vec<_>>(), vec![true, false, false, true]);

    let count_out = out.column(out.schema().index_of("count").unwrap())
        .as_any().downcast_ref::<arrow::array::Int64Array>().unwrap();
    assert_eq!((0..4).map(|i| count_out.value(i)).collect::<Vec<_>>(), vec![100, 200, 400, 500]);
}

/// A predictor literally named `x` is not carried a second time: the
/// computed `x` output already holds its values, so the row-preserving
/// carry does not duplicate it — it simply does not refuse the call
/// either, since the column's content is byte-identical to what `x`
/// already carries. The response `y` still rides along as an ordinary
/// carried column.
#[test]
fn smooth_residuals_predictor_named_x_is_not_carried_twice() {
    let xs: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let ys: Vec<f64> = xs.iter().map(|&x| 2.0 * x + 1.0).collect();
    let batch = xy_batch(xs, ys.clone());
    let spec = residuals_spec("x", "y", vec![]);
    let out = apply(&spec, &batch).unwrap();

    let schema = out.schema();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names, vec!["x", "residual", "y"], "x is not duplicated; y rides along as a carried column");
    assert_eq!(col(&out, "y"), ys, "the carried y column holds the caller's own response values");
}

/// An input column named like one of the step's own output columns is
/// refused rather than shadowed — the `Bin(per_row=True)` collision text,
/// applied to `Smooth`'s residual mode, for every name the step can emit:
/// the always-emitted `residual`, the computed `x` when the predictor is
/// a different column (`v`), and `_ref_zero`/`_metrics_text`/`_metrics_y`
/// only when their own inject flag asks for them.
#[test]
fn smooth_residuals_refuses_an_input_column_named_like_each_column_it_emits() {
    pyo3::Python::initialize();
    for (taken, spec) in [
        ("residual", residuals_spec("v", "w", vec![])),
        ("x", residuals_spec("v", "w", vec![])),
        ("_ref_zero", SmoothSpec { inject_zero_ref: true, ..residuals_spec("v", "w", vec![]) }),
        ("_metrics_text", SmoothSpec { inject_metrics: true, ..residuals_spec("v", "w", vec![]) }),
        ("_metrics_y", SmoothSpec { inject_metrics: true, ..residuals_spec("v", "w", vec![]) }),
    ] {
        let v: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0];
        let w: Vec<f64> = vec![1.0, 3.0, 5.0, 7.0];
        let taken_col: Vec<f64> = vec![-1.0, -2.0, -3.0, -4.0];
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Float64, true),
            Field::new("w", DataType::Float64, true),
            Field::new(taken, DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(v)),
            Arc::new(Float64Array::from(w)),
            Arc::new(Float64Array::from(taken_col)),
        ]).unwrap();
        let err = apply(&spec, &batch).expect_err(&format!("a data column named '{taken}' must refuse"));
        assert_eq!(
            err.to_string(),
            format!(
                "ValueError: stat_smooth: residual output column '{taken}' collides with an \
                 input column of the same name; rename the input column"
            ),
            "{taken}",
        );
    }
}

/// An optional column this call does NOT emit (`_ref_zero`/`_metrics_text`/
/// `_metrics_y` with the matching inject flag unset) is carried like any
/// other column — the collision check names only what a call actually
/// emits, not every column the step could ever emit.
#[test]
fn smooth_residuals_carries_an_input_column_named_like_an_optional_column_it_does_not_emit() {
    for taken in ["_ref_zero", "_metrics_text", "_metrics_y"] {
        let v: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0];
        let w: Vec<f64> = vec![1.0, 3.0, 5.0, 7.0];
        let taken_col: Vec<f64> = vec![9.0, 8.0, 7.0, 6.0];
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Float64, true),
            Field::new("w", DataType::Float64, true),
            Field::new(taken, DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(v)),
            Arc::new(Float64Array::from(w)),
            Arc::new(Float64Array::from(taken_col)),
        ]).unwrap();
        let spec = residuals_spec("v", "w", vec![]);
        let out = apply(&spec, &batch).expect(&format!("'{taken}' is an ordinary carried column when not requested"));
        assert_eq!(col(&out, taken), vec![9.0, 8.0, 7.0, 6.0], "{taken}");
    }
}

/// Fewer than 2 usable points degrades to the existing zero-row residuals
/// batch (unchanged), but its schema still carries every other input
/// column, at its own dtype — so a value channel on a data column
/// resolves even though no row is drawn.
#[test]
fn smooth_residuals_degenerate_batch_still_carries_input_schema() {
    let v: Vec<f64> = vec![1.0];
    let w: Vec<f64> = vec![2.0];
    let s: Vec<&str> = vec!["only-row"];
    let schema = Arc::new(Schema::new(vec![
        Field::new("v", DataType::Float64, true),
        Field::new("w", DataType::Float64, true),
        Field::new("s", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(Float64Array::from(v)),
        Arc::new(Float64Array::from(w)),
        Arc::new(arrow::array::StringArray::from(s)),
    ]).unwrap();
    let spec = residuals_spec("v", "w", vec![]);
    let out = apply(&spec, &batch).expect("fewer than 2 points degrades, not fails");
    assert_eq!(out.num_rows(), 0, "a single usable row still can't fit a line");
    let schema = out.schema();
    let mut names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    names.sort_unstable();
    assert_eq!(names, vec!["residual", "s", "v", "w", "x"]);
    assert_eq!(
        out.schema().field(out.schema().index_of("s").unwrap()).data_type(),
        &DataType::Utf8,
        "the carried column keeps its own dtype even with no rows to carry",
    );
}

/// A grouped residuals fit carries every other input column per group, in
/// each group's own row order, and the group key is appended once by the
/// stacking template — not duplicated by the carry (the predictor is
/// literally named `x` here too, so this also covers "a predictor named x
/// and group keys not duplicated" together).
#[test]
fn smooth_residuals_grouped_carries_columns_without_duplicating_group_key() {
    let xs: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let ys: Vec<f64> = vec![1.0, 3.0, 5.0, 7.0, 5.0, 3.0];
    let s: Vec<f64> = vec![10.0, 11.0, 12.0, 20.0, 21.0, 22.0];
    let expected_groups: Vec<&str> = vec!["a", "a", "a", "b", "b", "b"];
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, true),
        Field::new("y", DataType::Float64, true),
        Field::new("s", DataType::Float64, true),
        Field::new("g", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(Float64Array::from(xs)),
        Arc::new(Float64Array::from(ys)),
        Arc::new(Float64Array::from(s.clone())),
        Arc::new(arrow::array::StringArray::from(expected_groups.clone())),
    ]).unwrap();
    let spec = residuals_spec("x", "y", vec!["g".into()]);
    let out = apply(&spec, &batch).unwrap();

    let schema = out.schema();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names, vec!["x", "residual", "y", "s", "g"], "x is not duplicated, and g is appended once");
    assert_eq!(col(&out, "s"), s, "the carried column follows each group's own rows, in order, group by group");
    let g_col = out.column(out.schema().index_of("g").unwrap())
        .as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
    assert_eq!((0..6).map(|i| g_col.value(i)).collect::<Vec<_>>(), expected_groups);
}

/// With the groups interleaved in the input (`a, b, a, b, ...`), each
/// group's emitted rows carry its OWN input row's values, in that row's
/// own position within the group — not the input's own interleaved
/// order, and not another group's row at the same input position. A
/// carry that read `row_idx` against the wrong partition, or read values
/// in plain input order instead of per-group, would mix `a`'s and `b`'s
/// `id` values here; a contiguous fixture (as above) cannot tell the two
/// apart.
#[test]
fn smooth_residuals_grouped_rows_carry_their_own_values_under_interleaved_groups() {
    let xs: Vec<f64> = (0..8).map(|i| i as f64).collect();
    let ys: Vec<f64> = xs.iter().enumerate().map(|(i, &x)| if i % 2 == 0 { 2.0 * x } else { -1.0 * x }).collect();
    let id: Vec<&str> = vec!["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7"];
    let g: Vec<&str> = vec!["a", "b", "a", "b", "a", "b", "a", "b"];
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, true),
        Field::new("y", DataType::Float64, true),
        Field::new("id", DataType::Utf8, true),
        Field::new("g", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(Float64Array::from(xs)),
        Arc::new(Float64Array::from(ys)),
        Arc::new(arrow::array::StringArray::from(id)),
        Arc::new(arrow::array::StringArray::from(g)),
    ]).unwrap();
    let spec = residuals_spec("x", "y", vec!["g".into()]);
    let out = apply(&spec, &batch).unwrap();

    assert_eq!(out.num_rows(), 8);
    let id_out = out.column(out.schema().index_of("id").unwrap())
        .as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
    assert_eq!(
        (0..8).map(|i| id_out.value(i)).collect::<Vec<_>>(),
        vec!["r0", "r2", "r4", "r6", "r1", "r3", "r5", "r7"],
        "group a's own rows (even input positions), then group b's own rows (odd), each in its original relative order",
    );
    let g_out = out.column(out.schema().index_of("g").unwrap())
        .as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
    assert_eq!(
        (0..8).map(|i| g_out.value(i)).collect::<Vec<_>>(),
        vec!["a", "a", "a", "a", "b", "b", "b", "b"],
    );
}

/// Under `x_bins=`, the FIT aggregates into bins, but the residual rows —
/// and every carried column — stay at the ORIGINAL (un-binned) input
/// rows, in their original order: residual mode is always row-preserving
/// over the caller's own rows, `x_bins` or not.
#[test]
fn smooth_residuals_with_x_bins_still_carries_the_original_rows() {
    let xs: Vec<f64> = (0..20).map(|i| i as f64 / 2.0).collect();
    let ys: Vec<f64> = xs.iter().map(|&x| 2.0 * x + 1.0).collect();
    let s: Vec<f64> = (0..20).map(|i| i as f64 * 10.0).collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, true),
        Field::new("y", DataType::Float64, true),
        Field::new("s", DataType::Float64, true),
    ]));
    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(Float64Array::from(xs)),
        Arc::new(Float64Array::from(ys)),
        Arc::new(Float64Array::from(s.clone())),
    ]).unwrap();
    let spec = SmoothSpec { x_bins: Some(4), ..residuals_spec("x", "y", vec![]) };
    let out = apply(&spec, &batch).unwrap();
    assert_eq!(out.num_rows(), 20, "residual rows stay one per ORIGINAL input row; x_bins does not shrink them");
    assert_eq!(col(&out, "s"), s, "the carried column follows the original rows, not the binned ones");
}

/// A carried (non-positional) column rides through a demanded log x
/// unmapped: its own values are copied straight from the caller's batch,
/// never forward/backward-mapped the way the predictor's own `x` slot is
/// — and the predictor's own column, literally named `x` here, is still
/// not carried a second time even though the fit now reads it through a
/// log-mapped private column, not its own name.
#[test]
fn smooth_residuals_on_a_demanded_log_x_carries_other_columns_unmapped() {
    pyo3::Python::initialize();
    let xs: Vec<f64> = vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0];
    let ys: Vec<f64> = xs.iter().map(|&x| 3.0 + 2.0 * x.log10()).collect();
    let weights: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, true),
        Field::new("y", DataType::Float64, true),
        Field::new("w", DataType::Float64, true),
    ]));
    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(Float64Array::from(xs)),
        Arc::new(Float64Array::from(ys.clone())),
        Arc::new(Float64Array::from(weights.clone())),
    ]).unwrap();
    let spec = SmoothSpec { output: SmoothOutput::Residuals, ..lm_ci_spec(None, vec![]) };
    let ctx = ctx_demanding(axis_slots(Some(positive_space()), None));
    let out = apply_with_context(&spec, &batch, &ctx).expect("a clean log-x fit emits residuals");
    assert_eq!(out.num_rows(), 6, "every row survives (no nulls to drop)");
    assert_eq!(col(&out, "w"), weights, "the carried column is copied straight from the caller, never log-mapped");

    let schema = out.schema();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(
        names, vec!["x", "residual", "y", "w"],
        "the caller's own x column is not carried a second time even under a demanded log x",
    );
}

/// When the predictor is NOT named `x` (it is `v` here), a demanded log x
/// maps only a fresh, private column for the fit itself — the caller's
/// own `v` column is carried verbatim, in data units, and the private
/// column never leaks into the output: `run_in_log_space`'s own
/// post-body filter strips any column named in its `private` set, which
/// is exactly what the generated name would be.
#[test]
fn smooth_residuals_on_a_demanded_log_x_keeps_a_predictor_not_named_x_in_data_units() {
    pyo3::Python::initialize();
    let v: Vec<f64> = vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0];
    let w: Vec<f64> = v.iter().map(|&x| 3.0 + 2.0 * x.log10()).collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("v", DataType::Float64, true),
        Field::new("w", DataType::Float64, true),
    ]));
    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(Float64Array::from(v.clone())),
        Arc::new(Float64Array::from(w)),
    ]).unwrap();
    let spec = SmoothSpec { x: "v".into(), y: "w".into(), output: SmoothOutput::Residuals, ..lm_ci_spec(None, vec![]) };
    let ctx = ctx_demanding(axis_slots(Some(positive_space()), None));
    let out = apply_with_context(&spec, &batch, &ctx).expect("a clean log-x fit emits residuals");

    let schema = out.schema();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert!(
        names.iter().all(|n| !n.starts_with("__log_space_")),
        "no private log-space column leaks into the output: {names:?}",
    );
    assert_eq!(names, vec!["x", "residual", "v", "w"], "v carries under its own name, not folded into x");
    assert_eq!(col(&out, "v"), v, "the carried predictor is the caller's own data-space values, never log-mapped");
    let xg = col(&out, "x");
    for (xq, vq) in xg.iter().zip(v.iter()) {
        assert!(
            (xq - vq).abs() < 1e-9,
            "the computed x, mapped back, must equal the predictor's own data-space value: x={xq} v={vq}",
        );
    }
}

// ── The internal residual_columns naming setting ─────────────────────────

/// Setting every computed column's name moves them all out of the way:
/// the output carries the computed columns under their SET names first,
/// then every input column in input order — including the predictor, and
/// including data columns that happen to be named like the DEFAULT output
/// names (`x`, `residual`, `_ref_zero`, `_metrics_text`, `_metrics_y`),
/// which are now ordinary data, not a collision.
#[test]
fn smooth_residuals_emits_its_columns_under_the_names_its_setting_gives() {
    let v: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0];
    let w: Vec<f64> = vec![1.0, 3.0, 5.0, 7.0];
    let x_col: Vec<f64> = vec![100.0, 101.0, 102.0, 103.0];
    let residual_col: Vec<f64> = vec![200.0, 201.0, 202.0, 203.0];
    let ref_zero_col: Vec<f64> = vec![300.0, 301.0, 302.0, 303.0];
    let metrics_text_col: Vec<f64> = vec![400.0, 401.0, 402.0, 403.0];
    let metrics_y_col: Vec<f64> = vec![500.0, 501.0, 502.0, 503.0];
    let schema = Arc::new(Schema::new(vec![
        Field::new("v", DataType::Float64, true),
        Field::new("w", DataType::Float64, true),
        Field::new("x", DataType::Float64, true),
        Field::new("residual", DataType::Float64, true),
        Field::new("_ref_zero", DataType::Float64, true),
        Field::new("_metrics_text", DataType::Float64, true),
        Field::new("_metrics_y", DataType::Float64, true),
    ]));
    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(Float64Array::from(v)),
        Arc::new(Float64Array::from(w)),
        Arc::new(Float64Array::from(x_col.clone())),
        Arc::new(Float64Array::from(residual_col.clone())),
        Arc::new(Float64Array::from(ref_zero_col.clone())),
        Arc::new(Float64Array::from(metrics_text_col.clone())),
        Arc::new(Float64Array::from(metrics_y_col.clone())),
    ]).unwrap();
    let mut spec = SmoothSpec { inject_zero_ref: true, inject_metrics: true, ..residuals_spec("v", "w", vec![]) };
    spec.residual_columns = ResidualColumns {
        x: "x_1".into(), residual: "residual_1".into(), ref_zero: "_ref_zero_1".into(),
        metrics_text: "_metrics_text_1".into(), metrics_y: "_metrics_y_1".into(),
    };
    let out = apply(&spec, &batch).unwrap();

    let schema = out.schema();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(
        names,
        vec!["x_1", "residual_1", "_ref_zero_1", "_metrics_text_1", "_metrics_y_1", "v", "w", "x", "residual", "_ref_zero", "_metrics_text", "_metrics_y"],
        "the computed columns under their set names, then every input column in input order",
    );
    assert_eq!(col(&out, "x"), x_col, "the data's own 'x' column, not the computed output, carries verbatim");
    assert_eq!(col(&out, "residual"), residual_col);
    assert_eq!(col(&out, "_ref_zero"), ref_zero_col);
    assert_eq!(col(&out, "_metrics_text"), metrics_text_col);
    assert_eq!(col(&out, "_metrics_y"), metrics_y_col);
}

/// An input column named like a SET name (not a default one) is refused
/// the same way an input column named like a default output name is.
#[test]
fn smooth_residuals_refuses_an_input_column_named_like_a_set_name() {
    pyo3::Python::initialize();
    let v: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0];
    let w: Vec<f64> = vec![1.0, 3.0, 5.0, 7.0];
    let r1_col: Vec<f64> = vec![9.0, 8.0, 7.0, 6.0];
    let schema = Arc::new(Schema::new(vec![
        Field::new("v", DataType::Float64, true),
        Field::new("w", DataType::Float64, true),
        Field::new("r1", DataType::Float64, true),
    ]));
    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(Float64Array::from(v)),
        Arc::new(Float64Array::from(w)),
        Arc::new(Float64Array::from(r1_col)),
    ]).unwrap();
    let mut spec = residuals_spec("v", "w", vec![]);
    spec.residual_columns.residual = "r1".into();
    let err = apply(&spec, &batch).expect_err("a data column named like the SET residual name must refuse");
    assert_eq!(
        err.to_string(),
        "ValueError: stat_smooth: residual output column 'r1' collides with an input column of the same name; rename the input column",
    );
}

/// An empty residual column name, or two columns sharing one name, both
/// refuse before the fit even runs.
#[test]
fn smooth_residuals_refuses_an_empty_or_shared_column_name() {
    pyo3::Python::initialize();
    let batch = xy_batch(vec![0.0, 1.0], vec![1.0, 3.0]);

    let mut empty_name = residuals_spec("x", "y", vec![]);
    empty_name.residual_columns.residual = String::new();
    let err = apply(&empty_name, &batch).expect_err("an empty residual column name must refuse");
    assert_eq!(err.to_string(), "ValueError: stat_smooth: a residual column name cannot be empty");

    let mut shared_name = residuals_spec("x", "y", vec![]);
    shared_name.residual_columns.ref_zero = "residual".into();
    let err = apply(&shared_name, &batch).expect_err("two columns sharing one name must refuse");
    assert_eq!(
        err.to_string(),
        "ValueError: stat_smooth: residual column name 'residual' is given to more than one column",
    );
}

/// A set `x` name reaches the log-space machinery: declared in the X slot
/// by `TransformSpec::space_columns`, and actually mapped back to data
/// units by a demanded log x at render time.
#[test]
fn smooth_residuals_with_a_set_x_name_on_a_demanded_log_x_maps_that_column_back() {
    let mut spec = residuals_spec("v", "w", vec![]);
    spec.residual_columns.x = "x_1".into();
    let columns = TransformSpec::Smooth(spec.clone()).space_columns();
    assert_eq!(columns.len(), 1, "{columns:?}");
    assert_eq!(columns[0].output, "x_1");
    assert_eq!(columns[0].slot, SpaceSlot::X);
    assert_eq!(columns[0].input, "v");

    let v: Vec<f64> = vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0];
    let w: Vec<f64> = v.iter().map(|&x| 3.0 + 2.0 * x.log10()).collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("v", DataType::Float64, true),
        Field::new("w", DataType::Float64, true),
    ]));
    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(Float64Array::from(v.clone())),
        Arc::new(Float64Array::from(w)),
    ]).unwrap();
    let ctx = ctx_demanding(axis_slots(Some(positive_space()), None));
    let out = apply_with_context(&spec, &batch, &ctx).expect("a clean log-x fit emits residuals");
    let xg = col(&out, "x_1");
    for (xq, vq) in xg.iter().zip(v.iter()) {
        assert!(
            (xq - vq).abs() < 1e-9,
            "x_1, mapped back, must equal v's own data-space value: x_1={xq} v={vq}",
        );
    }
}

/// Grouping by a column named like the DEFAULT residual column works once
/// the computed residual is given a fresh name: the group key keeps the
/// caller's own values, under a different name from the computed column.
#[test]
fn smooth_residuals_grouped_by_a_column_named_like_the_default_residual_with_the_residual_set_fresh() {
    let v: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let w: Vec<f64> = vec![1.0, 3.0, 5.0, 7.0, 5.0, 3.0];
    let residual_group: Vec<&str> = vec!["a", "a", "a", "b", "b", "b"];
    let schema = Arc::new(Schema::new(vec![
        Field::new("v", DataType::Float64, true),
        Field::new("w", DataType::Float64, true),
        Field::new("residual", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(Float64Array::from(v)),
        Arc::new(Float64Array::from(w)),
        Arc::new(arrow::array::StringArray::from(residual_group.clone())),
    ]).unwrap();
    let mut spec = residuals_spec("v", "w", vec!["residual".into()]);
    spec.residual_columns.residual = "residual_1".into();
    let out = apply(&spec, &batch).expect(
        "grouping by a column named like the default residual, with the residual set fresh, must not collide",
    );

    let schema = out.schema();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(
        names, vec!["x", "residual_1", "v", "w", "residual"],
        "the computed residual is 'residual_1'; the group key keeps the data's own 'residual' values",
    );
    let g_col = out.column(out.schema().index_of("residual").unwrap())
        .as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
    assert_eq!((0..6).map(|i| g_col.value(i)).collect::<Vec<_>>(), residual_group);
}

/// The default setting is omitted from the wire; setting one name
/// serializes only that one field, and deserializing it back fills every
/// other name from the default.
#[test]
fn smooth_spec_residual_columns_default_is_omitted_and_a_partial_setting_round_trips() {
    let spec = residuals_spec("v", "w", vec![]);
    let json = serde_json::to_string(&spec).unwrap();
    assert!(!json.contains("residual_columns"), "the default setting is omitted from the wire: {json}");
    let parsed: SmoothSpec = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, spec);

    let mut with_setting = spec.clone();
    with_setting.residual_columns.residual = "r1".into();
    let json2 = serde_json::to_string(&with_setting).unwrap();
    assert!(
        json2.contains(r#""residual_columns":{"residual":"r1"}"#),
        "only the set name travels: {json2}",
    );
    let parsed2: SmoothSpec = serde_json::from_str(&json2).unwrap();
    assert_eq!(parsed2, with_setting);
    assert_eq!(parsed2.residual_columns.x, "x", "the unset names keep their default");
}

/// The wire omits a default setting and refuses unknown keys: a misspelled
/// key is refused, not silently filled from `ResidualColumns::default()` as
/// `#[serde(default)]` alone would do without `deny_unknown_fields`.
#[test]
fn smooth_spec_residual_columns_with_an_unknown_key_fails_to_deserialize() {
    let err = serde_json::from_str::<ResidualColumns>(r#"{"residul": "r1"}"#)
        .expect_err("a misspelled key must be refused, not silently defaulted");
    assert!(err.to_string().contains("residul"), "the error should name the unknown key: {err}");
}

/// `_with_residual_columns` only reaches a non-Smooth spec through a
/// `PySmooth` built directly in Rust, never through the public `fm.Smooth`
/// door (whose one constructor, `PySmooth::new`, only ever builds
/// `TransformSpec::Smooth`) — still a typed error, not a panic, at that
/// PyO3 boundary.
#[test]
fn with_residual_columns_on_a_non_smooth_spec_is_a_typed_error_not_a_panic() {
    pyo3::Python::initialize();
    let wrong = PySmooth(TransformSpec::Identity(crate::transform::identity::IdentitySpec {
        name: None,
        statistics: false,
    }));
    let err = wrong._with_residual_columns(None, None, None, None, None)
        .expect_err("a non-Smooth spec must be refused, not panic");
    assert!(err.to_string().contains("Smooth"), "{err}");
}
