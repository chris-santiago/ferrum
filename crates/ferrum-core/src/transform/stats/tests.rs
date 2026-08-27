//! Statistics test coverage.
//!
//! This module carries two kinds of tests:
//! - the original small in-src correctness suite (`test_*`), and
//! - the bulk of edge-case coverage relocated from `tests/bug_hunt_stats_transforms.rs`
//!   (R1, 2026-08-27), which previously re-derived every formula (studentized
//!   residuals, Cook's distance, rank-averaging, variance/covariance ranking,
//!   the `phi_inv`/Shapiro-Wilk polynomial approximation, silhouette) in a
//!   standalone integration-test crate and asserted against the *copy* rather
//!   than the real functions. Every test below calls the real functions in
//!   `super::*` — private helpers directly; the two pyfunction-gated ones
//!   (`calinski_harabasz_score`, `mds_classical`) through a `Python<'_>` token
//!   via `pyo3::Python::initialize(); Python::attach(|py| …)`, the crate's
//!   established idiom for exercising pyo3-gated functions from an in-src
//!   unit test (see `spec::chart::tests::test_parse_json_field_error_is_name_prefixed`,
//!   `spec::composite::tests::composite_tree_from_py_reads_legend_override`) —
//!   which is why this coverage can only live here rather than in `tests/`.
//!
//! `silhouette_samples_vec` gained a `metric: DistanceMetric` parameter and a
//! `PyResult` return since the mirror was written (it now delegates to
//! `linkage::condensed_distances` instead of an inlined Euclidean loop); every
//! ported silhouette test passes `DistanceMetric::Euclidean` (the mirror's
//! implicit metric) and unwraps.
//!
//! One genuine tautology from the mirror file was dropped outright:
//! `bug_hunt_calinski_zero_within_scatter` asserted `within == 0.0` against a
//! local `let within = 0.0;` it never touched — it called no function under
//! test. Six near-duplicate mirror tests were dropped in favor of a stronger
//! or identical sibling that already exists (either in the original suite
//! below or elsewhere in the mirror's own Round-2 pass): the pre-fix,
//! panic-style `bug_hunt_variance_rank_n_zero` / `bug_hunt_cooks_negative_leverage`
//! / `bug_hunt_cooks_p_eff_zero` / `bug_hunt_shapiro_n3_minimum` in favor of
//! their exact-value Round-2 regression counterparts (`b21_variance_rank_n_zero`,
//! `b18_negative_leverage`, `b19_peff_zero`, `b20_shapiro_clamped_n3_skewed`);
//! `bug_hunt_covariance_rank_negative_correlation_absolute` (weak `> 0.0`
//! check) in favor of the Round-2 exact-value `covariance_rank_anticorrelated`;
//! and `bug_hunt_r2_kendall_all_joint_ties` in favor of the equivalent
//! `kendall_all_tied_both` (both assert an all-tied-both-columns group
//! produces `tau = NaN` with matching tie counts). Kendall-tau-b coverage
//! lives in `crate::diagnostics`, where the real `kendall_tau_b` is defined,
//! not here.
//!
//! The HEAD-era `test_calinski_well_vs_poorly_separated` (predating this
//! batch, not from the mirror file) was itself a same-shape local
//! re-implementation of the Calinski-Harabasz formula that called no crate
//! code. It asserted the identical contract as
//! `calinski_monotonicity_well_separated_beats_overlapping` below
//! (well-separated CH > poorly-separated CH) with noisier but not more
//! discriminating data, so it was deleted rather than kept as a second
//! mirror of the same property (spec review, 2026-08-27).

use super::*;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

// ─────────────────────────────────────────────────────────────────────────
// Original correctness suite
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_studentized_no_x() {
    let yt = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let yp = vec![1.1, 1.9, 3.2, 3.8, 5.1];
    let stud = studentized_residual_vec(&yt, &yp, None);
    assert_eq!(stud.len(), 5);
    let mean: f64 = stud.iter().sum::<f64>() / 5.0;
    assert!(approx_eq(mean, 0.0, 0.3));
}

#[test]
fn test_rankdata_average_no_ties() {
    let x = [3.0, 1.0, 2.0];
    let r = rankdata_average_vec(&x);
    assert!(approx_eq(r[0], 3.0, 1e-12));
    assert!(approx_eq(r[1], 1.0, 1e-12));
    assert!(approx_eq(r[2], 2.0, 1e-12));
}

#[test]
fn test_rankdata_average_with_ties() {
    let x = [1.0, 2.0, 2.0, 4.0];
    let r = rankdata_average_vec(&x);
    assert!(approx_eq(r[0], 1.0, 1e-12));
    assert!(approx_eq(r[1], 2.5, 1e-12));
    assert!(approx_eq(r[2], 2.5, 1e-12));
    assert!(approx_eq(r[3], 4.0, 1e-12));
}

#[test]
fn test_shapiro_w_normal_like() {
    let x: Vec<f64> = (-50..=50).map(|i| i as f64 * 0.1).collect();
    let w = shapiro_w_scalar(&x);
    assert!(w > 0.95, "W = {w} should be close to 1 for uniform-ish data");
}

#[test]
fn test_shapiro_w_small() {
    let x = [1.0, 2.0, 3.0];
    let w = shapiro_w_scalar(&x);
    assert!(w > 0.0, "W = {w} should be positive");
}

#[test]
fn test_variance_rank_basic() {
    // 2 cols: [1,2,3] and [10,10,10]
    let flat = [1.0, 10.0, 2.0, 10.0, 3.0, 10.0];
    let v = variance_rank_vec(&flat, 3, 2);
    assert!(v[0] > 0.0);
    assert!(approx_eq(v[1], 0.0, 1e-12));
}

// ---- PCA ----

fn make_batch(data: &[f64], n: usize, p: usize) -> RecordBatch {
    let fields: Vec<Field> = (0..p)
        .map(|j| Field::new(format!("f{j}"), DataType::Float64, false))
        .collect();
    let schema = Arc::new(Schema::new(fields));
    let cols: Vec<ArrayRef> = (0..p)
        .map(|j| {
            let vals: Vec<f64> = (0..n).map(|i| data[i * p + j]).collect();
            Arc::new(Float64Array::from(vals)) as ArrayRef
        })
        .collect();
    RecordBatch::try_new(schema, cols).unwrap()
}

#[test]
fn test_pca_identity_eigenvectors() {
    // 3x3 identity → SVD returns identity-like structure
    let data = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let batch = make_batch(&data, 3, 3);
    let (_scores, evr) = pca_svd(&batch, Some(3)).unwrap();
    // After centering identity, variance is the same across components
    let total: f64 = evr.iter().sum();
    assert!(approx_eq(total, 1.0, 1e-10), "total EVR should be 1.0, got {total}");
}

#[test]
fn test_pca_correlated_columns() {
    // col1 = [1,2,3,4,5], col2 = [2,4,6,8,10] (perfectly correlated)
    let data = [1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0, 5.0, 10.0];
    let batch = make_batch(&data, 5, 2);
    let (_scores, evr) = pca_svd(&batch, Some(2)).unwrap();
    // First component should capture ~100% of variance
    assert!(evr[0] > 0.999, "first EVR should be ~1.0, got {}", evr[0]);
}

// ---- Silhouette ----

#[test]
fn test_silhouette_perfect_separation() {
    // Two well-separated clusters
    let flat = [
        0.0, 0.0, 0.1, 0.0, 0.0, 0.1,
        10.0, 10.0, 10.1, 10.0, 10.0, 10.1,
    ];
    let labels = [0i64, 0, 0, 1, 1, 1];
    let sv = silhouette_samples_vec(&flat, 6, 2, &labels, DistanceMetric::Euclidean).unwrap();
    for &s in &sv {
        assert!(s > 0.9, "perfectly separated clusters should have silhouette > 0.9, got {s}");
    }
}

// ---- Calinski-Harabasz ----

/// Calls the real `calinski_harabasz_score` pyfunction (defined above in this
/// module) through a `Python<'_>` token, per the crate's established idiom
/// for exercising pyo3-gated functions from an in-src unit test — see
/// `spec::chart::tests::test_parse_json_field_error_is_name_prefixed` and
/// `spec::composite::tests::composite_tree_from_py_reads_legend_override`.
/// Replaces an earlier hand-computed `compute_ch` duplicate of the CH
/// formula that called no crate code (R1 mirror anti-pattern, corrected in
/// spec review).
#[test]
fn calinski_monotonicity_well_separated_beats_overlapping() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let labels = [0i64, 0, 1, 1];

        // Well-separated: two tight clusters far apart.
        let flat_good = [0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0];
        let ch_good = calinski_harabasz_score(
            py,
            PyRecordBatch::new(make_batch(&flat_good, 4, 2)),
            PyArray::from_array_ref(Arc::new(Int64Array::from(labels.to_vec()))),
        )
        .unwrap();

        // Overlapping: same two labels, points close together.
        let flat_bad = [0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 1.5, 1.5];
        let ch_bad = calinski_harabasz_score(
            py,
            PyRecordBatch::new(make_batch(&flat_bad, 4, 2)),
            PyArray::from_array_ref(Arc::new(Int64Array::from(labels.to_vec()))),
        )
        .unwrap();

        assert!(
            ch_good > ch_bad,
            "well-separated CH ({ch_good}) should be > overlapping CH ({ch_bad})"
        );
        // ch_good's identical-within-cluster points make within_scatter exactly
        // 0.0, which short-circuits to `Ok(f64::INFINITY)` before the
        // between/within division runs — so the ordering check above alone does
        // not exercise that division. ch_bad does reach it; hand-computed:
        // x_bar=(0.75,0.75), between=0.5, within=2.0, ch = (0.5/1)/(2.0/2) = 0.5.
        assert!(
            approx_eq(ch_bad, 0.5, 1e-10),
            "overlapping-cluster CH should be exactly 0.5, got {ch_bad}"
        );
    });
}

// ---- MDS ----

/// Calls the real `mds_classical` pyfunction (defined above in this module)
/// through the same `Python::attach` idiom used for `calinski_harabasz_score`
/// above. Replaces an earlier hand-implementation of classical MDS (squared-
/// distance matrix, row/col/grand means, double-centering, eigendecomposition)
/// that asserted against its own local computation and called no crate code
/// (R1 mirror anti-pattern, corrected in quality review).
#[test]
fn mds_triangle_geometry_recovers_original_distances() {
    // 3 points: (0,0), (3,0), (0,4) → distances: 3, 4, 5
    let flat = [0.0, 0.0, 3.0, 0.0, 0.0, 4.0];
    // pairwise Euclidean: d(0,1)=3, d(0,2)=4, d(1,2)=5
    let dm = DistanceMetric::Euclidean;
    let condensed = crate::transform::linkage::condensed_distances(&flat, 3, 2, dm).unwrap();
    assert!(approx_eq(condensed[0], 3.0, 1e-10));
    assert!(approx_eq(condensed[1], 4.0, 1e-10));
    assert!(approx_eq(condensed[2], 5.0, 1e-10));

    pyo3::Python::initialize();
    Python::attach(|py| {
        let out: RecordBatch = mds_classical(
            py,
            PyRecordBatch::new(make_batch(&flat, 3, 2)),
            2,
            "features",
            "euclidean",
        )
        .unwrap()
        .into();

        assert_eq!(out.num_columns(), 2, "n_components=2 should yield 2 embedding columns");
        let dim0 = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        let dim1 = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();

        // MDS recovers geometry up to rotation/reflection, so verify the
        // *inter-point* distances in the embedded space match the originals,
        // not the coordinates themselves.
        let full_dist_orig = [
            [0.0, 3.0, 4.0],
            [3.0, 0.0, 5.0],
            [4.0, 5.0, 0.0],
        ];
        for i in 0..3 {
            for j in (i + 1)..3 {
                let dx = dim0.value(i) - dim0.value(j);
                let dy = dim1.value(i) - dim1.value(j);
                let d_embed = (dx * dx + dy * dy).sqrt();
                let d_orig = full_dist_orig[i][j];
                assert!(
                    approx_eq(d_embed, d_orig, 1e-8),
                    "MDS distance ({i},{j}): embed={d_embed}, orig={d_orig}"
                );
            }
        }
    });
}

// ─────────────────────────────────────────────────────────────────────────
// R1-relocated coverage (tests/bug_hunt_stats_transforms.rs, 2026-08-27)
// ─────────────────────────────────────────────────────────────────────────

// ---- NaN propagation ----

#[test]
fn rankdata_nan_no_panic() {
    let x = [1.0, f64::NAN, 3.0, f64::NAN, 2.0];
    let r = rankdata_average_vec(&x);
    assert_eq!(r.len(), 5);
    for &ri in &r {
        assert!(ri.is_finite(), "rank should be finite even with NaN input, got {ri}");
    }
}

#[test]
fn rankdata_nan_sum_invariant() {
    let x = [f64::NAN, 1.0, f64::NAN, 2.0];
    let r = rankdata_average_vec(&x);
    let sum: f64 = r.iter().sum();
    let expected = 4.0 * 5.0 / 2.0; // 10.0
    assert!(
        approx_eq(sum, expected, 1e-10),
        "rank sum with NaN should still be n*(n+1)/2 = {expected}, got {sum}"
    );
}

#[test]
fn studentized_nan_in_ytrue_no_panic() {
    let yt = [1.0, f64::NAN, 3.0, 4.0];
    let yp = [1.0, 2.0, 3.0, 4.0];
    let stud = studentized_residual_vec(&yt, &yp, None);
    assert_eq!(stud.len(), 4);
}

#[test]
fn studentized_nan_in_hat_no_panic() {
    let yt = [1.0, 2.0, 3.0, 4.0, 5.0];
    let yp = [1.1, 1.9, 3.2, 3.8, 5.1];
    let h = [0.2, f64::NAN, 0.2, 0.2, 0.2];
    let stud = studentized_residual_vec(&yt, &yp, Some(&h));
    assert_eq!(stud.len(), 5);
}

#[test]
fn cooks_nan_in_hdiag_no_panic() {
    let yt = [1.0, 2.0, 3.0, 4.0, 5.0];
    let yp = [1.1, 1.9, 3.2, 3.8, 5.1];
    let h = [0.2, f64::NAN, 0.2, 0.2, 0.2];
    let cooks = cooks_distance_vec(&yt, &yp, &h);
    assert_eq!(cooks.len(), 5);
}

#[test]
fn shapiro_nan_input_no_panic() {
    let x = [1.0, f64::NAN, 3.0, 4.0, 5.0, 6.0, 7.0];
    let w = shapiro_w_scalar(&x);
    assert!(!w.is_infinite(), "shapiro W with NaN input should not be Inf, got {w}");
}

// ---- Infinity in input data ----

#[test]
fn studentized_infinity_in_ytrue_no_panic() {
    let yt = [1.0, f64::INFINITY, 3.0, 4.0];
    let yp = [1.0, 2.0, 3.0, 4.0];
    let stud = studentized_residual_vec(&yt, &yp, None);
    assert_eq!(stud.len(), 4);
}

#[test]
fn cooks_infinity_in_residual_no_panic() {
    let yt = [f64::INFINITY, 2.0, 3.0, 4.0, 5.0];
    let yp = [1.0, 2.0, 3.0, 4.0, 5.0];
    let h = [0.2, 0.2, 0.2, 0.2, 0.2]; // p_eff=1
    let cooks = cooks_distance_vec(&yt, &yp, &h);
    assert_eq!(cooks.len(), 5);
}

#[test]
fn rankdata_infinity_ordering() {
    let x = [1.0, f64::INFINITY, 2.0, f64::NEG_INFINITY];
    let r = rankdata_average_vec(&x);
    // NEG_INFINITY < 1.0 < 2.0 < INFINITY
    assert!(approx_eq(r[3], 1.0, 1e-12), "-Inf should be rank 1, got {}", r[3]);
    assert!(approx_eq(r[0], 2.0, 1e-12), "1.0 should be rank 2, got {}", r[0]);
    assert!(approx_eq(r[2], 3.0, 1e-12), "2.0 should be rank 3, got {}", r[2]);
    assert!(approx_eq(r[1], 4.0, 1e-12), "+Inf should be rank 4, got {}", r[1]);
}

#[test]
fn shapiro_infinity_no_panic() {
    let x = [1.0, 2.0, 3.0, f64::INFINITY, 5.0, 6.0, 7.0];
    let w = shapiro_w_scalar(&x);
    assert!(!w.is_infinite() || w == 0.0 || w.is_nan(),
        "shapiro W with Inf should be 0, NaN, or finite, got {w}");
}

// ---- Empty / degenerate inputs ----

#[test]
fn rankdata_empty() {
    let r = rankdata_average_vec(&[]);
    assert!(r.is_empty(), "empty input should produce empty output");
}

#[test]
fn studentized_empty_no_hat() {
    let stud = studentized_residual_vec(&[], &[], None);
    assert!(stud.is_empty(), "empty input should produce empty output");
}

#[test]
fn studentized_empty_with_hat() {
    let stud = studentized_residual_vec(&[], &[], Some(&[]));
    assert!(stud.is_empty(), "empty input should produce empty output");
}

#[test]
fn cooks_empty() {
    let cooks = cooks_distance_vec(&[], &[], &[]);
    assert!(cooks.is_empty(), "empty input should produce empty output");
}

#[test]
fn variance_rank_p_zero() {
    let v = variance_rank_vec(&[], 5, 0);
    assert!(v.is_empty(), "p=0 should produce empty variance vector");
}

#[test]
fn covariance_rank_n_zero() {
    let v = covariance_rank_vec(&[], 0, 2, &[]);
    assert_eq!(v.len(), 2);
    for &vi in &v {
        assert!(approx_eq(vi, 0.0, 1e-12), "n=0 cov should be 0, got {vi}");
    }
}

#[test]
fn silhouette_empty() {
    let sv = silhouette_samples_vec(&[], 0, 2, &[], DistanceMetric::Euclidean).unwrap();
    assert!(sv.is_empty(), "empty input should produce empty silhouette");
}

// ---- Extreme numeric values ----

#[test]
fn studentized_large_values_stay_finite() {
    let yt = [1e300, 2e300, 3e300];
    let yp = [1.1e300, 1.9e300, 3.2e300];
    let stud = studentized_residual_vec(&yt, &yp, None);
    for &s in &stud {
        assert!(s.is_finite(), "studentized residual with large values should be finite, got {s}");
    }
}

#[test]
fn cooks_tiny_residuals_stay_finite() {
    let yt = [1.0, 2.0, 3.0, 4.0, 5.0];
    let yp = [1.0 + 1e-300, 2.0 + 1e-300, 3.0 + 1e-300, 4.0 + 1e-300, 5.0 + 1e-300];
    let h = [0.2, 0.2, 0.2, 0.2, 0.2];
    let cooks = cooks_distance_vec(&yt, &yp, &h);
    for &c in &cooks {
        assert!(c.is_finite(), "Cook's D with tiny residuals should be finite, got {c}");
        assert!(c >= 0.0, "Cook's D should be non-negative, got {c}");
    }
}

#[test]
fn variance_rank_large_values_no_panic() {
    // col values ~1e154 → squared differences ~1e308 which is near f64::MAX
    let flat = [1e154, 1e100, 2e154, 1e100]; // n=2, p=2
    let v = variance_rank_vec(&flat, 2, 2);
    assert_eq!(v.len(), 2);
}

// ---- Hat-matrix leverage boundary values ----

#[test]
fn studentized_leverage_one_no_inf() {
    let yt = [1.0, 2.0, 3.0, 4.0, 5.0];
    let yp = [1.1, 1.9, 3.2, 3.8, 5.1];
    let h = [1.0, 0.0, 0.0, 0.0, 0.0]; // p_eff=1
    let stud = studentized_residual_vec(&yt, &yp, Some(&h));
    assert_eq!(stud.len(), 5);
    assert!(stud[0].is_finite(), "leverage=1.0 should not produce Inf, got {}", stud[0]);
}

#[test]
fn studentized_leverage_greater_than_one_no_panic() {
    let yt = [1.0, 2.0, 3.0, 4.0, 5.0];
    let yp = [1.1, 1.9, 3.2, 3.8, 5.1];
    let h = [1.5, 0.1, 0.1, 0.1, 0.1]; // p_eff=2, h[0]>1
    let stud = studentized_residual_vec(&yt, &yp, Some(&h));
    assert_eq!(stud.len(), 5);
    assert!(stud[0].is_finite(), "leverage>1.0 should still produce finite result, got {}", stud[0]);
}

#[test]
fn cooks_leverage_exactly_one_stays_finite() {
    let yt = [1.0, 2.0, 3.0, 4.0, 5.0];
    let yp = [1.1, 1.9, 3.2, 3.8, 5.1];
    let h = [1.0, 0.0, 0.0, 0.0, 0.0]; // p_eff=1
    let cooks = cooks_distance_vec(&yt, &yp, &h);
    assert_eq!(cooks.len(), 5);
    assert!(cooks[0].is_finite(), "Cook's D with h=1.0 should be finite, got {}", cooks[0]);
    assert!(cooks[0] > 0.0, "Cook's D with h=1.0 and non-zero residual should be positive");
}

#[test]
fn cooks_leverage_zero_is_zero() {
    let yt = [1.0, 2.0, 3.0, 4.0, 5.0];
    let yp = [1.1, 1.9, 3.2, 3.8, 5.1];
    let h = [0.0, 0.5, 0.5, 0.5, 0.5]; // p_eff=2
    let cooks = cooks_distance_vec(&yt, &yp, &h);
    assert!(
        approx_eq(cooks[0], 0.0, 1e-12),
        "Cook's D with h=0 should be 0, got {}", cooks[0]
    );
}

// ---- Studentized / Cook's hand-computed correctness ----

#[test]
fn studentized_no_hat_hand_computed() {
    // residuals = [1, -1, 2, -2], mean_r = 0, var = (1+1+4+4)/3 = 10/3,
    // sigma = sqrt(10/3), stud_i = r_i / sigma.
    let yt = [3.0, 1.0, 5.0, 0.0];
    let yp = [2.0, 2.0, 3.0, 2.0]; // residuals: [1, -1, 2, -2]
    let stud = studentized_residual_vec(&yt, &yp, None);

    let sigma = (10.0_f64 / 3.0).sqrt();
    assert!(approx_eq(stud[0], 1.0 / sigma, 1e-10),
        "stud[0] should be {}, got {}", 1.0 / sigma, stud[0]);
    assert!(approx_eq(stud[1], -1.0 / sigma, 1e-10),
        "stud[1] should be {}, got {}", -1.0 / sigma, stud[1]);
    assert!(approx_eq(stud[2], 2.0 / sigma, 1e-10),
        "stud[2] should be {}, got {}", 2.0 / sigma, stud[2]);
    assert!(approx_eq(stud[3], -2.0 / sigma, 1e-10),
        "stud[3] should be {}, got {}", -2.0 / sigma, stud[3]);
}

#[test]
fn studentized_with_hat_hand_computed() {
    // residuals = [0.5, -0.5], h = [0.5, 0.5], p_eff=1,
    // sse = 0.25+0.25 = 0.5, sigma_sq = 0.5/(2-1) = 0.5,
    // sigma = sqrt(0.5), stud_i = r_i / (sigma * sqrt(max(1-h_i, 1e-12)))
    // = r_i / (sqrt(0.5) * sqrt(0.5)) = r_i / 0.5
    let yt = [1.5, 1.5];
    let yp = [1.0, 2.0]; // residuals: [0.5, -0.5]
    let h = [0.5, 0.5]; // p_eff = 1
    let stud = studentized_residual_vec(&yt, &yp, Some(&h));

    assert!(approx_eq(stud[0], 1.0, 1e-10),
        "stud[0] should be 1.0, got {}", stud[0]);
    assert!(approx_eq(stud[1], -1.0, 1e-10),
        "stud[1] should be -1.0, got {}", stud[1]);
}

#[test]
fn cooks_hand_computed() {
    // residuals = [1, -1, 0], h = [0.5, 0.25, 0.25], p_eff=1.
    // sse = 1 + 1 + 0 = 2, sigma_sq = 2 / (3-1) = 1.
    // D_0 = (1^2 / (1*1)) * (0.5 / (1-0.5)^2) = 1 * (0.5/0.25) = 2.0
    // D_1 = (1^2 / (1*1)) * (0.25 / (1-0.25)^2) = 1 * (0.25/0.5625) = 4/9
    // D_2 = (0^2 / (1*1)) * anything = 0
    let yt = [2.0, 1.0, 3.0];
    let yp = [1.0, 2.0, 3.0]; // residuals: [1, -1, 0]
    let h = [0.5, 0.25, 0.25]; // p_eff=1
    let cooks = cooks_distance_vec(&yt, &yp, &h);

    assert!(approx_eq(cooks[0], 2.0, 1e-10),
        "Cook's D[0] should be 2.0, got {}", cooks[0]);
    assert!(approx_eq(cooks[1], 4.0 / 9.0, 1e-10),
        "Cook's D[1] should be 4/9, got {}", cooks[1]);
    assert!(approx_eq(cooks[2], 0.0, 1e-10),
        "Cook's D[2] should be 0.0, got {}", cooks[2]);
}

// ---- phi_inv edge cases ----

#[test]
fn phi_inv_negative_saturates() {
    assert_eq!(phi_inv(-1.0), -8.0);
    assert_eq!(phi_inv(-0.001), -8.0);
}

#[test]
fn phi_inv_greater_than_one_saturates() {
    assert_eq!(phi_inv(1.5), 8.0);
    assert_eq!(phi_inv(100.0), 8.0);
}

#[test]
fn phi_inv_nan_stays_nan() {
    let v = phi_inv(f64::NAN);
    assert!(v.is_nan(), "phi_inv(NaN) should be NaN, got {v}");
}

#[test]
fn phi_inv_at_plow_boundary() {
    let v = phi_inv(0.02425);
    assert!(v.is_finite(), "phi_inv at plow boundary should be finite, got {v}");
    assert!(v < 0.0, "phi_inv(0.02425) should be negative, got {v}");
}

#[test]
fn phi_inv_very_small_p() {
    let v = phi_inv(1e-10);
    assert!(v.is_finite(), "phi_inv(1e-10) should be finite, got {v}");
    assert!(v < -4.0, "phi_inv(1e-10) should be very negative, got {v}");
}

#[test]
fn phi_inv_monotonicity() {
    let ps = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999];
    for i in 0..ps.len() - 1 {
        let a = phi_inv(ps[i]);
        let b = phi_inv(ps[i + 1]);
        assert!(
            a < b,
            "phi_inv should be monotonically increasing: phi_inv({}) = {} >= phi_inv({}) = {}",
            ps[i], a, ps[i + 1], b
        );
    }
}

// ---- Shapiro-Wilk adversarial distributions ----

#[test]
fn shapiro_bimodal_has_low_w() {
    let mut x = Vec::new();
    for _ in 0..20 { x.push(0.0); }
    for _ in 0..20 { x.push(100.0); }
    let w = shapiro_w_scalar(&x);
    assert!(w.is_finite(), "bimodal W should be finite, got {w}");
    assert!(w < 0.9, "bimodal W should be low (<0.9), got {w}");
}

#[test]
fn shapiro_all_negative_matches_all_positive() {
    // W is shift/reflection invariant.
    let x_neg: Vec<f64> = (1..=10).map(|i| -(i as f64)).collect();
    let x_pos: Vec<f64> = (1..=10).map(|i| i as f64).collect();
    let w_neg = shapiro_w_scalar(&x_neg);
    let w_pos = shapiro_w_scalar(&x_pos);
    assert!(
        approx_eq(w_neg, w_pos, 1e-10),
        "W should be shift/reflection invariant: W_neg={w_neg}, W_pos={w_pos}"
    );
}

#[test]
fn shapiro_scale_invariance() {
    let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();
    let x_scaled: Vec<f64> = x.iter().map(|v| v * 1000.0).collect();
    let w = shapiro_w_scalar(&x);
    let w_scaled = shapiro_w_scalar(&x_scaled);
    assert!(
        approx_eq(w, w_scaled, 1e-10),
        "W should be scale-invariant: W={w}, W_scaled={w_scaled}"
    );
}

#[test]
fn shapiro_shift_invariance() {
    let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();
    let x_shifted: Vec<f64> = x.iter().map(|v| v + 1e6).collect();
    let w = shapiro_w_scalar(&x);
    let w_shifted = shapiro_w_scalar(&x_shifted);
    assert!(
        approx_eq(w, w_shifted, 1e-6),
        "W should be shift-invariant: W={w}, W_shifted={w_shifted}"
    );
}

// ---- rankdata_average_vec correctness ----

#[test]
fn rankdata_three_way_tie_beginning() {
    let x = [5.0, 5.0, 5.0, 10.0, 20.0];
    let r = rankdata_average_vec(&x);
    assert!(approx_eq(r[0], 2.0, 1e-12), "first of 3-way tie should be 2.0, got {}", r[0]);
    assert!(approx_eq(r[1], 2.0, 1e-12), "second of 3-way tie should be 2.0, got {}", r[1]);
    assert!(approx_eq(r[2], 2.0, 1e-12), "third of 3-way tie should be 2.0, got {}", r[2]);
    assert!(approx_eq(r[3], 4.0, 1e-12), "after tie should be 4.0, got {}", r[3]);
    assert!(approx_eq(r[4], 5.0, 1e-12), "last should be 5.0, got {}", r[4]);
}

#[test]
fn rankdata_multiple_tie_groups() {
    let x = [10.0, 20.0, 10.0, 30.0, 20.0];
    let r = rankdata_average_vec(&x);
    // Sorted order: 10,10,20,20,30 → ranks (1,2),(3,4),(5)
    assert!(approx_eq(r[0], 1.5, 1e-12), "10 should have rank 1.5, got {}", r[0]);
    assert!(approx_eq(r[1], 3.5, 1e-12), "20 should have rank 3.5, got {}", r[1]);
    assert!(approx_eq(r[2], 1.5, 1e-12), "10 should have rank 1.5, got {}", r[2]);
    assert!(approx_eq(r[3], 5.0, 1e-12), "30 should have rank 5.0, got {}", r[3]);
    assert!(approx_eq(r[4], 3.5, 1e-12), "20 should have rank 3.5, got {}", r[4]);
}

#[test]
fn rankdata_descending() {
    let x = [5.0, 4.0, 3.0, 2.0, 1.0];
    let r = rankdata_average_vec(&x);
    assert!(approx_eq(r[0], 5.0, 1e-12));
    assert!(approx_eq(r[1], 4.0, 1e-12));
    assert!(approx_eq(r[2], 3.0, 1e-12));
    assert!(approx_eq(r[3], 2.0, 1e-12));
    assert!(approx_eq(r[4], 1.0, 1e-12));
}

// ---- Silhouette boundary cases ----

#[test]
fn silhouette_single_cluster_is_zero() {
    let flat = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // n=3, feat=2
    let labels = [0i64, 0, 0];
    let sv = silhouette_samples_vec(&flat, 3, 2, &labels, DistanceMetric::Euclidean).unwrap();
    for &s in &sv {
        assert!(approx_eq(s, 0.0, 1e-12),
            "single cluster silhouette should be 0, got {s}");
    }
}

#[test]
fn silhouette_identical_points_is_zero() {
    let flat = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]; // n=3, feat=2, all same point
    let labels = [0i64, 0, 1];
    let sv = silhouette_samples_vec(&flat, 3, 2, &labels, DistanceMetric::Euclidean).unwrap();
    for &s in &sv {
        assert!(
            approx_eq(s, 0.0, 1e-12),
            "identical points silhouette should be 0, got {s}"
        );
    }
}

#[test]
fn silhouette_singleton_clusters_is_one() {
    // a(i) = 0 (singleton cluster), b(i) = dist to other point
    // s(i) = (b - 0) / max(0, b) = 1.0 for non-zero distance
    let flat = [0.0, 0.0, 10.0, 10.0]; // n=2, feat=2
    let labels = [0i64, 1];
    let sv = silhouette_samples_vec(&flat, 2, 2, &labels, DistanceMetric::Euclidean).unwrap();
    for &s in &sv {
        assert!(
            approx_eq(s, 1.0, 1e-10),
            "singleton clusters with distance>0 should have silhouette=1, got {s}"
        );
    }
}

#[test]
fn silhouette_bounds_poorly_clustered() {
    // Poorly clustered: point 2 (label 0) is closer to cluster 1 points
    let flat = [
        0.0, 0.0,   // point 0: cluster 0
        10.0, 10.0, // point 1: cluster 0
        9.0, 9.0,   // point 2: cluster 0 (but close to cluster 1)
        8.0, 8.0,   // point 3: cluster 1
        11.0, 11.0, // point 4: cluster 1
    ];
    let labels = [0i64, 0, 0, 1, 1];
    let sv = silhouette_samples_vec(&flat, 5, 2, &labels, DistanceMetric::Euclidean).unwrap();
    for (i, &s) in sv.iter().enumerate() {
        assert!(
            s >= -1.0 && s <= 1.0,
            "silhouette[{i}] should be in [-1,1], got {s}"
        );
    }
}

#[test]
fn silhouette_many_singleton_clusters_stay_in_range() {
    let flat = [
        0.0, 0.0,
        10.0, 0.0,
        20.0, 0.0,
        30.0, 0.0,
        0.5, 0.0, // close to point 0
    ];
    let labels = [0i64, 1, 2, 3, 0]; // cluster 0 has 2 points, rest are singletons
    let sv = silhouette_samples_vec(&flat, 5, 2, &labels, DistanceMetric::Euclidean).unwrap();
    assert_eq!(sv.len(), 5);
    for (i, &s) in sv.iter().enumerate() {
        assert!(
            s.is_finite() && s >= -1.0 && s <= 1.0,
            "silhouette[{i}] should be finite and in [-1,1], got {s}"
        );
    }
}

// ---- Variance/covariance rank correctness ----

#[test]
fn variance_rank_hand_computed() {
    // col=[1,2,3], mean=2, var=(1+0+1)/3=2/3.
    let flat = [1.0, 2.0, 3.0]; // n=3, p=1
    let v = variance_rank_vec(&flat, 3, 1);
    let expected = 2.0 / 3.0;
    assert!(
        approx_eq(v[0], expected, 1e-12),
        "variance should be 2/3, got {}", v[0]
    );
}

#[test]
fn variance_rank_uses_population_not_sample_variance() {
    let flat = [0.0, 10.0]; // n=2, p=1
    let v = variance_rank_vec(&flat, 2, 1);
    // Population variance: mean=5, var = ((0-5)^2 + (10-5)^2) / 2 = 50/2 = 25
    // Sample variance would be: 50 / 1 = 50
    assert!(
        approx_eq(v[0], 25.0, 1e-12),
        "variance_rank uses population variance (n not n-1): expected 25.0, got {}", v[0]
    );
}

#[test]
fn covariance_rank_hand_computed() {
    // x=[1,2,3], y=[2,4,6]. x_mean=2, y_mean=4,
    // cov = ((1-2)(2-4) + (2-2)(4-4) + (3-2)(6-4)) / (3-1) = (2 + 0 + 2) / 2 = 2.0.
    let flat = [1.0, 2.0, 3.0]; // n=3, p=1
    let y = [2.0, 4.0, 6.0];
    let result = covariance_rank_vec(&flat, 3, 1, &y);
    assert!(
        approx_eq(result[0], 2.0, 1e-12),
        "cov(x,y) should be 2.0, got {}", result[0]
    );
}

#[test]
fn covariance_rank_constant_x_is_zero() {
    let flat = [5.0, 5.0, 5.0, 5.0]; // n=4, p=1
    let y = [1.0, 2.0, 3.0, 4.0];
    let result = covariance_rank_vec(&flat, 4, 1, &y);
    assert!(
        approx_eq(result[0], 0.0, 1e-12),
        "constant x should have cov=0, got {}", result[0]
    );
}

#[test]
fn covariance_rank_constant_y_is_zero() {
    let flat = [1.0, 2.0, 3.0, 4.0]; // n=4, p=1
    let y = [7.0, 7.0, 7.0, 7.0];
    let result = covariance_rank_vec(&flat, 4, 1, &y);
    assert!(
        approx_eq(result[0], 0.0, 1e-12),
        "constant y should have cov=0, got {}", result[0]
    );
}

#[test]
fn variance_rank_n1_is_zero() {
    let flat = [42.0]; // n=1, p=1
    let v = variance_rank_vec(&flat, 1, 1);
    assert!(
        approx_eq(v[0], 0.0, 1e-12),
        "single observation variance should be 0, got {}", v[0]
    );
}

// ---- Off-by-one in rankdata tied-rank averaging ----

#[test]
fn rankdata_four_way_tie_averaging() {
    let x = [7.0, 7.0, 7.0, 7.0, 100.0];
    let r = rankdata_average_vec(&x);
    // Tie group occupies ranks 1,2,3,4. Average = 2.5.
    for i in 0..4 {
        assert!(
            approx_eq(r[i], 2.5, 1e-12),
            "4-way tie rank[{i}] should be 2.5, got {}", r[i]
        );
    }
    assert!(approx_eq(r[4], 5.0, 1e-12), "non-tied rank should be 5.0, got {}", r[4]);
}

#[test]
fn rankdata_two_separate_ties() {
    let x = [3.0, 1.0, 3.0, 1.0];
    let r = rankdata_average_vec(&x);
    // Sorted: 1,1,3,3 → ranks (1,2) and (3,4)
    assert!(approx_eq(r[0], 3.5, 1e-12));
    assert!(approx_eq(r[1], 1.5, 1e-12));
    assert!(approx_eq(r[2], 3.5, 1e-12));
    assert!(approx_eq(r[3], 1.5, 1e-12));
}

// ---- studentized_residual_vec: n=2 no hat ----

#[test]
fn studentized_n2_no_hat_hand_computed() {
    let yt = [2.0, 0.0];
    let yp = [1.0, 1.0]; // residuals: [1, -1]
    let stud = studentized_residual_vec(&yt, &yp, None);
    let sigma = 2.0_f64.sqrt();
    assert!(approx_eq(stud[0], 1.0 / sigma, 1e-10),
        "stud[0] should be 1/sqrt(2), got {}", stud[0]);
    assert!(approx_eq(stud[1], -1.0 / sigma, 1e-10),
        "stud[1] should be -1/sqrt(2), got {}", stud[1]);
}

// ---- p_eff rounding edge case in hat-path ----

#[test]
fn studentized_peff_rounding_up() {
    // h_diag sums to 1.5, p_eff rounds to 2.
    let yt = [1.0, 2.0, 3.0, 4.0, 5.0];
    let yp = [1.5, 2.5, 2.5, 3.5, 5.5]; // residuals: [-0.5, -0.5, 0.5, 0.5, -0.5]
    let h = [0.3, 0.3, 0.3, 0.3, 0.3]; // sum=1.5, rounds to 2
    let stud = studentized_residual_vec(&yt, &yp, Some(&h));
    assert_eq!(stud.len(), 5);
    let p_eff = (h.iter().sum::<f64>()).round() as usize;
    assert_eq!(p_eff, 2, "p_eff should round 1.5 to 2");
    for &s in &stud {
        assert!(s.is_finite(), "studentized residual should be finite, got {s}");
    }
}

#[test]
fn studentized_peff_rounds_to_zero_stays_finite() {
    let yt = [1.0, 2.0, 3.0, 4.0, 5.0];
    let yp = [1.1, 1.9, 3.2, 3.8, 5.1];
    let h = [0.098, 0.098, 0.098, 0.098, 0.098]; // sum=0.49, rounds to 0
    let stud = studentized_residual_vec(&yt, &yp, Some(&h));
    assert_eq!(stud.len(), 5);
    for &s in &stud {
        assert!(s.is_finite(), "studentized with p_eff=0 should be finite, got {s}");
    }
}

// ---- Variance rank with extreme value range ----

#[test]
fn variance_rank_tiny_vs_large_column() {
    let eps = 1e-150;
    let flat = [
        1.0, 100.0,
        1.0 + eps, 200.0,
        1.0 - eps, 150.0,
    ]; // n=3, p=2
    let v = variance_rank_vec(&flat, 3, 2);
    assert!(v[0] >= 0.0, "tiny-variance column should be non-negative, got {}", v[0]);
    assert!(v[1] > v[0], "large-variance column should have higher var, got {} vs {}", v[1], v[0]);
}

#[test]
fn covariance_rank_discriminates_columns() {
    // col0 = y (perfectly correlated), col1 = constant (zero cov)
    let flat = [
        1.0, 5.0,
        2.0, 5.0,
        3.0, 5.0,
        4.0, 5.0,
    ]; // n=4, p=2
    let y = [1.0, 2.0, 3.0, 4.0];
    let result = covariance_rank_vec(&flat, 4, 2, &y);
    assert!(
        result[0] > result[1],
        "correlated column cov ({}) should be > constant column cov ({})",
        result[0], result[1]
    );
    assert!(approx_eq(result[1], 0.0, 1e-12),
        "constant column cov should be 0, got {}", result[1]);
}

// ---- Shapiro-Wilk: n=3 boundary ----

#[test]
fn shapiro_n3_two_identical_values() {
    let x = [1.0, 1.0, 100.0];
    let w = shapiro_w_scalar(&x);
    assert!(
        w.is_finite() && w >= 0.0 && w <= 1.0,
        "n=3 with 2 identical values: W should be in [0,1], got {w}"
    );
}

#[test]
fn shapiro_n3_extreme_magnitude_differences() {
    let x = [1e-100, 1.0, 1e100];
    let w = shapiro_w_scalar(&x);
    assert!(
        w.is_finite() && w >= 0.0 && w <= 1.0,
        "n=3 extreme magnitudes: W should be in [0,1], got {w}"
    );
}

#[test]
fn shapiro_w_never_exceeds_one_across_sizes() {
    for n in 3..=30 {
        let x: Vec<f64> = (0..n).map(|i| (i as f64).powi(2) + 0.1 * (i as f64)).collect();
        let w = shapiro_w_scalar(&x);
        if w.is_finite() {
            assert!(w <= 1.0 + 1e-10, "W should never exceed 1.0: n={n}, W={w}");
            assert!(w >= 0.0, "W should never be negative: n={n}, W={w}");
        }
    }
}

#[test]
fn silhouette_nan_features_no_panic() {
    let flat = [
        0.0, 0.0,
        f64::NAN, 1.0,
        10.0, 10.0,
        11.0, 11.0,
    ]; // n=4, feat=2
    let labels = [0i64, 0, 1, 1];
    let sv = silhouette_samples_vec(&flat, 4, 2, &labels, DistanceMetric::Euclidean).unwrap();
    assert_eq!(sv.len(), 4);
}

// ---- studentized_residual_vec: mismatched lengths (zip-truncation contract) ----

#[test]
fn studentized_mismatched_lengths_output_len_is_ytrue_len() {
    // y_true and y_pred have different lengths; zip truncates residuals to
    // the shorter one, but sigma/output length still come from y_true.len().
    let yt = [1.0, 2.0, 3.0, 4.0, 5.0]; // len=5
    let yp = [1.0, 2.0, 3.0]; // len=3
    let stud = studentized_residual_vec(&yt, &yp, None);
    assert_eq!(
        stud.len(), 5,
        "output length is y_true.len() (5), not min(5,3), which is a documented contract quirk"
    );
}

#[test]
fn cooks_hdiag_longer_than_data_output_len_is_data_len() {
    // h_diag longer than y_true/y_pred: p_eff sums ALL of h_diag, but the
    // per-row zip still truncates to the shorter y arrays.
    let yt = [1.0, 2.0, 3.0]; // len=3
    let yp = [1.0, 2.0, 3.0]; // len=3
    let h = [0.25, 0.25, 0.25, 0.25, 0.25]; // len=5, sum=1.25, p_eff=1
    let cooks = cooks_distance_vec(&yt, &yp, &h);
    assert_eq!(cooks.len(), 3, "output length should match y arrays, not h_diag");
}

// ─────────────────────────────────────────────────────────────────────────
// Round-2 relocated coverage (regression fixes B18-B21, Kendall excluded —
// see crate::diagnostics; perfect-fit/boundary/property tests)
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn b18_negative_leverage_clamped_nonnegative() {
    let yt = [1.0, 2.0, 3.0, 4.0, 5.0];
    let yp = [1.1, 1.9, 3.2, 3.8, 5.1];
    let h = [-0.5, 0.5, 0.5, 0.5, 0.5]; // p_eff=2, h[0]=-0.5
    let cooks = cooks_distance_vec(&yt, &yp, &h);
    for (i, &c) in cooks.iter().enumerate() {
        assert!(c >= 0.0, "B18 regression: Cook's D[{i}] must be >= 0, got {c}");
    }
}

#[test]
fn b19_peff_zero_returns_zeros_not_nan() {
    let yt = [1.0, 2.0, 3.0, 4.0, 5.0];
    let yp = [1.1, 1.9, 3.2, 3.8, 5.1];
    let h = [0.0, 0.0, 0.0, 0.0, 0.0]; // sum=0, p_eff=0
    let cooks = cooks_distance_vec(&yt, &yp, &h);
    for (i, &c) in cooks.iter().enumerate() {
        assert!(c == 0.0, "B19 regression: Cook's D[{i}] with p_eff=0 must be 0.0, got {c}");
    }
}

#[test]
fn b20_shapiro_clamped_n3_skewed() {
    // The original bug was n=3, skewed data producing W > 1.0.
    let inputs: Vec<Vec<f64>> = vec![
        vec![1.0, 100.0, 200.0],
        vec![0.001, 0.002, 1000.0],
        vec![-100.0, 0.0, 0.001],
        vec![1.0, 1.0, 100.0],
    ];
    for (idx, x) in inputs.iter().enumerate() {
        let w = shapiro_w_scalar(x);
        assert!(w <= 1.0, "B20 regression: shapiro W (input {idx}) must be <= 1.0, got {w}");
        assert!(w >= 0.0, "B20 regression: shapiro W (input {idx}) must be >= 0.0, got {w}");
    }
}

#[test]
fn b21_variance_rank_n_zero_returns_zeros_not_nan() {
    let v = variance_rank_vec(&[], 0, 3);
    assert_eq!(v.len(), 3);
    for (i, &vi) in v.iter().enumerate() {
        assert!(vi == 0.0, "B21 regression: variance_rank[{i}] with n=0 must be 0.0, got {vi}");
    }
}

// ---- Perfectly collinear / perfect fit edge cases ----

#[test]
fn studentized_perfect_fit_no_hat_is_zero() {
    let yt = [1.0, 2.0, 3.0, 4.0, 5.0];
    let yp = [1.0, 2.0, 3.0, 4.0, 5.0]; // perfect fit
    let stud = studentized_residual_vec(&yt, &yp, None);
    for (i, &s) in stud.iter().enumerate() {
        assert!(s == 0.0, "perfect fit stud[{i}] should be 0.0, got {s}");
    }
}

#[test]
fn studentized_perfect_fit_with_hat_is_zero() {
    let yt = [1.0, 2.0, 3.0, 4.0, 5.0];
    let yp = [1.0, 2.0, 3.0, 4.0, 5.0];
    let h = [0.4, 0.2, 0.1, 0.2, 0.1]; // p_eff=1
    let stud = studentized_residual_vec(&yt, &yp, Some(&h));
    for (i, &s) in stud.iter().enumerate() {
        assert!(s == 0.0, "perfect fit with hat stud[{i}] should be 0.0, got {s}");
    }
}

#[test]
fn cooks_perfect_fit_is_zero() {
    let yt = [1.0, 2.0, 3.0, 4.0, 5.0];
    let yp = [1.0, 2.0, 3.0, 4.0, 5.0];
    let h = [0.2, 0.2, 0.2, 0.2, 0.2]; // p_eff=1
    let cooks = cooks_distance_vec(&yt, &yp, &h);
    for (i, &c) in cooks.iter().enumerate() {
        assert!(c == 0.0, "perfect fit Cook's D[{i}] should be 0.0, got {c}");
    }
}

#[test]
fn cooks_n_equals_peff_is_zero() {
    let yt = [1.0, 2.0, 3.0];
    let yp = [1.5, 2.5, 2.5];
    let h = [1.0, 1.0, 1.0]; // sum=3 → p_eff=3, n=3 → n <= p_eff → zeros
    let cooks = cooks_distance_vec(&yt, &yp, &h);
    for (i, &c) in cooks.iter().enumerate() {
        assert!(c == 0.0, "n == p_eff: Cook's D[{i}] should be 0.0, got {c}");
    }
}

#[test]
fn cooks_n_less_than_peff_is_zero() {
    let yt = [1.0, 2.0];
    let yp = [1.5, 2.5];
    let h = [2.0, 2.0]; // sum=4 → p_eff=4, n=2 → n <= p_eff → zeros
    let cooks = cooks_distance_vec(&yt, &yp, &h);
    for (i, &c) in cooks.iter().enumerate() {
        assert!(c == 0.0, "n < p_eff: Cook's D[{i}] should be 0.0, got {c}");
    }
}

// ---- Variance/covariance with NaN in columns ----

#[test]
fn variance_rank_nan_in_one_column_poisons_only_that_column() {
    let flat = [
        1.0, f64::NAN,
        2.0, 2.0,
        3.0, 3.0,
    ]; // n=3, p=2
    let v = variance_rank_vec(&flat, 3, 2);
    assert_eq!(v.len(), 2);
    assert!(v[0].is_finite(), "non-NaN column variance should be finite, got {}", v[0]);
    assert!(v[1].is_nan(), "NaN column variance should be NaN, got {}", v[1]);
}

#[test]
fn covariance_rank_nan_in_x_poisons_that_column_only() {
    let flat = [
        f64::NAN, 1.0,
        2.0, 2.0,
        3.0, 3.0,
    ]; // n=3, p=2
    let y = [1.0, 2.0, 3.0];
    let result = covariance_rank_vec(&flat, 3, 2, &y);
    assert_eq!(result.len(), 2);
    assert!(result[0].is_nan(), "NaN x column cov should be NaN, got {}", result[0]);
    assert!(result[1].is_finite(), "clean column cov should be finite, got {}", result[1]);
}

#[test]
fn covariance_rank_nan_in_y_poisons_all_columns() {
    let flat = [1.0, 2.0, 3.0]; // n=3, p=1
    let y = [1.0, f64::NAN, 3.0];
    let result = covariance_rank_vec(&flat, 3, 1, &y);
    assert!(result[0].is_nan(), "NaN y should make all covs NaN, got {}", result[0]);
}

#[test]
fn variance_rank_infinity_in_column_produces_nan() {
    let flat = [
        1.0, f64::INFINITY,
        2.0, 1.0,
        3.0, 2.0,
    ]; // n=3, p=2
    let v = variance_rank_vec(&flat, 3, 2);
    assert!(v[0].is_finite(), "normal column var should be finite, got {}", v[0]);
    // Column 1: [Inf, 1, 2] → mean = Inf, (Inf - Inf)^2 = NaN → var = NaN
    assert!(v[1].is_nan(), "Inf column variance should be NaN (Inf-Inf=NaN), got {}", v[1]);
}

// ---- Shapiro n=4/n=5 boundary (inner a[2..n-2] loop starts at n>4) ----

#[test]
fn shapiro_n4_boundary_linear_data_high_w() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let w = shapiro_w_scalar(&x);
    assert!(w.is_finite() && w >= 0.0 && w <= 1.0, "n=4 Shapiro W should be in [0,1], got {w}");
    assert!(w > 0.9, "linear n=4 data should have W > 0.9, got {w}");
}

#[test]
fn shapiro_n5_first_inner_loop_iteration() {
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let w = shapiro_w_scalar(&x);
    assert!(w.is_finite() && w >= 0.0 && w <= 1.0, "n=5 Shapiro W should be in [0,1], got {w}");
}

#[test]
fn shapiro_all_identical_is_exactly_one() {
    // denom = 0 → W = 1.0 (via denom <= 0 guard).
    let x = [5.0, 5.0, 5.0];
    let w = shapiro_w_scalar(&x);
    assert!(w == 1.0, "all-identical values: W should be 1.0, got {w}");
}

#[test]
fn shapiro_n100_various_distributions() {
    let uniform: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let w = shapiro_w_scalar(&uniform);
    assert!(w >= 0.0 && w <= 1.0, "n=100 uniform W should be in [0,1], got {w}");

    let exponential: Vec<f64> = (0..100).map(|i| (i as f64 * 0.05).exp()).collect();
    let w = shapiro_w_scalar(&exponential);
    assert!(w >= 0.0 && w <= 1.0, "n=100 exponential W should be in [0,1], got {w}");
    assert!(w < 0.95, "exponential data should have lower W than 0.95, got {w}");
}

#[test]
fn shapiro_n5_skewed_negative_eps_no_panic() {
    let x = [0.0, 0.0001, 0.0002, 0.0003, 10000.0];
    let w = shapiro_w_scalar(&x);
    assert!(w.is_finite() && w >= 0.0 && w <= 1.0, "skewed n=5 W should be in [0,1], got {w}");
}

// ---- Studentized residual single-element (n=1) ----

#[test]
fn studentized_n1_no_hat_uses_sigma_one() {
    // n<=1 branch: sigma defaults to 1.0, so stud = r / 1.0.
    let yt = [5.0];
    let yp = [3.0]; // residual = 2.0
    let stud = studentized_residual_vec(&yt, &yp, None);
    assert_eq!(stud.len(), 1);
    assert!(approx_eq(stud[0], 2.0, 1e-10), "n=1 stud should be 2.0 (residual/1.0), got {}", stud[0]);
}

#[test]
fn studentized_n1_with_hat_hand_computed() {
    // p_eff=round(0.8)=1, n=1. n.saturating_sub(1).max(1) = 1.
    // sse=4, sigma_sq=4, sigma=2. stud = 2/(2*sqrt(max(0.2,1e-12))) = 1/sqrt(0.2).
    let yt = [5.0];
    let yp = [3.0]; // r = 2.0
    let h = [0.8];
    let stud = studentized_residual_vec(&yt, &yp, Some(&h));
    assert_eq!(stud.len(), 1);
    let expected = 1.0 / (0.2_f64).sqrt();
    assert!(approx_eq(stud[0], expected, 1e-10),
        "n=1 with hat stud should be {expected}, got {}", stud[0]);
}

// ---- rankdata with all-equal / single / negative-zero / subnormal values ----

#[test]
fn rankdata_all_equal_gets_midpoint_rank() {
    let x = [7.0, 7.0, 7.0, 7.0, 7.0];
    let r = rankdata_average_vec(&x);
    let expected = 3.0; // (1+2+3+4+5)/5 = 3.0
    for (i, &ri) in r.iter().enumerate() {
        assert!(approx_eq(ri, expected, 1e-12), "all-equal rank[{i}] should be {expected}, got {ri}");
    }
}

#[test]
fn rankdata_single_element_is_one() {
    let x = [42.0];
    let r = rankdata_average_vec(&x);
    assert_eq!(r.len(), 1);
    assert!(approx_eq(r[0], 1.0, 1e-12), "single element rank should be 1.0, got {}", r[0]);
}

#[test]
fn rankdata_negative_zero_ties_with_positive_zero() {
    // -0.0 == 0.0 in IEEE 754, so they should be treated as tied.
    let x = [0.0, -0.0, 1.0];
    let r = rankdata_average_vec(&x);
    assert!(approx_eq(r[0], 1.5, 1e-12), "0.0 should tie with -0.0: rank should be 1.5, got {}", r[0]);
    assert!(approx_eq(r[1], 1.5, 1e-12), "-0.0 should tie with 0.0: rank should be 1.5, got {}", r[1]);
    assert!(approx_eq(r[2], 3.0, 1e-12), "1.0 rank should be 3.0, got {}", r[2]);
}

#[test]
fn rankdata_subnormals_ordered_correctly() {
    let x = [f64::MIN_POSITIVE, f64::MIN_POSITIVE / 2.0, 0.0];
    let r = rankdata_average_vec(&x);
    // 0 < MIN_POSITIVE/2 < MIN_POSITIVE (subnormal < normal)
    assert!(approx_eq(r[2], 1.0, 1e-12), "0.0 should be rank 1, got {}", r[2]);
    assert!(approx_eq(r[1], 2.0, 1e-12), "MIN_POSITIVE/2 should be rank 2, got {}", r[1]);
    assert!(approx_eq(r[0], 3.0, 1e-12), "MIN_POSITIVE should be rank 3, got {}", r[0]);
}

// ---- Silhouette with 3+ clusters and adversarial configurations ----

#[test]
fn silhouette_three_well_separated_clusters() {
    let flat = [
        0.0, 0.0,   // cluster 0
        0.1, 0.0,   // cluster 0
        10.0, 0.0,  // cluster 1
        10.1, 0.0,  // cluster 1
        0.0, 10.0,  // cluster 2
        0.1, 10.0,  // cluster 2
    ];
    let labels = [0i64, 0, 1, 1, 2, 2];
    let sv = silhouette_samples_vec(&flat, 6, 2, &labels, DistanceMetric::Euclidean).unwrap();
    for (i, &s) in sv.iter().enumerate() {
        assert!(s >= -1.0 && s <= 1.0 && s.is_finite(), "3-cluster silhouette[{i}] should be in [-1,1], got {s}");
        assert!(s > 0.5, "well-separated 3-cluster point {i} should have s > 0.5, got {s}");
    }
}

#[test]
fn silhouette_negative_label_values_no_panic() {
    let flat = [
        0.0, 0.0,
        0.1, 0.0,
        10.0, 10.0,
        10.1, 10.0,
    ];
    let labels = [-1i64, -1, -2, -2];
    let sv = silhouette_samples_vec(&flat, 4, 2, &labels, DistanceMetric::Euclidean).unwrap();
    for (i, &s) in sv.iter().enumerate() {
        assert!(s.is_finite() && s >= -1.0 && s <= 1.0, "negative-label silhouette[{i}] should be in [-1,1], got {s}");
    }
}

#[test]
fn silhouette_misclassified_point_is_negative() {
    let flat = [
        0.0, 0.0,     // cluster 0 center
        100.0, 100.0, // cluster 0 but close to cluster 1
        99.0, 99.0,   // cluster 1
        100.0, 100.0, // cluster 1 center
    ];
    let labels = [0i64, 0, 1, 1];
    let sv = silhouette_samples_vec(&flat, 4, 2, &labels, DistanceMetric::Euclidean).unwrap();
    assert!(sv[1] < 0.0, "misclassified point should have negative silhouette, got {}", sv[1]);
}

// ---- Cook's distance with fractional h_diag summing near .5 ----

#[test]
fn cooks_peff_rounding_boundary_at_half() {
    // f64::round() uses "round half away from zero": 0.5→1, 1.5→2.
    let yt = [1.0, 2.0, 3.0, 4.0];
    let yp = [1.5, 2.5, 2.5, 3.5]; // residuals: [-0.5, -0.5, 0.5, 0.5]

    // Sum = 0.49 → rounds to 0 → p_eff=0 → early return zeros
    let h_below = [0.1225, 0.1225, 0.1225, 0.1225]; // sum = 0.49
    let cooks_below = cooks_distance_vec(&yt, &yp, &h_below);
    for &c in &cooks_below {
        assert!(c == 0.0, "sum=0.49 rounds to p_eff=0, Cook's D should be 0, got {c}");
    }

    // Sum = 0.50 → rounds to 1 (round half away from zero) → normal computation
    let h_half = [0.125, 0.125, 0.125, 0.125]; // sum = 0.5
    let cooks_half = cooks_distance_vec(&yt, &yp, &h_half);
    assert!(cooks_half.iter().any(|&c| c > 0.0), "sum=0.5 rounds to p_eff=1, should have non-zero Cook's D");
    for (i, &c) in cooks_half.iter().enumerate() {
        assert!(c.is_finite() && c >= 0.0, "Cook's D[{i}] should be finite and non-negative, got {c}");
    }

    // Sum = 0.51 → rounds to 1 → normal computation (same as 0.5)
    let h_just_above = [0.1275, 0.1275, 0.1275, 0.1275]; // sum = 0.51
    let cooks_above = cooks_distance_vec(&yt, &yp, &h_just_above);
    assert!(cooks_above.iter().any(|&c| c > 0.0), "sum=0.51 rounds to p_eff=1, should have non-zero Cook's D");
    for (i, &c) in cooks_above.iter().enumerate() {
        assert!(c.is_finite() && c >= 0.0, "Cook's D[{i}] should be finite and non-negative, got {c}");
    }
}

// ---- Covariance rank with n=1 and n=2 (boundary for division) ----

#[test]
fn covariance_rank_n1_is_zero() {
    let flat = [42.0]; // n=1, p=1
    let y = [7.0];
    let result = covariance_rank_vec(&flat, 1, 1, &y);
    assert_eq!(result.len(), 1);
    assert!(result[0] == 0.0, "n=1 cov should be 0.0, got {}", result[0]);
}

#[test]
fn covariance_rank_n2_hand_computed() {
    // x=[0, 10], y=[0, 20]. x_mean=5, y_mean=10.
    // cov = ((0-5)(0-10) + (10-5)(20-10)) / (2-1) = (50 + 50) / 1 = 100.
    let flat = [0.0, 10.0]; // n=2, p=1
    let y = [0.0, 20.0];
    let result = covariance_rank_vec(&flat, 2, 1, &y);
    assert!(approx_eq(result[0], 100.0, 1e-10), "n=2 cov should be 100.0, got {}", result[0]);
}

// ---- Studentized residual with NaN propagation through sigma ----

#[test]
fn studentized_all_nan_residuals_falls_back_to_zero() {
    let yt = [f64::NAN, f64::NAN, f64::NAN];
    let yp = [1.0, 2.0, 3.0];
    let stud = studentized_residual_vec(&yt, &yp, None);
    assert_eq!(stud.len(), 3);
    for (i, &s) in stud.iter().enumerate() {
        assert!(s == 0.0, "all-NaN residuals: stud[{i}] should be 0.0 (NaN sigma fallback), got {s}");
    }
}

#[test]
fn studentized_nan_hat_with_zero_residuals_falls_back_to_zero() {
    let yt = [1.0, 2.0, 3.0, 4.0];
    let yp = [1.0, 2.0, 3.0, 4.0]; // zero residuals
    let h = [0.5, f64::NAN, 0.0, 0.0];
    let stud = studentized_residual_vec(&yt, &yp, Some(&h));
    for (i, &s) in stud.iter().enumerate() {
        assert!(s == 0.0, "zero residuals + NaN hat: stud[{i}] should be 0.0, got {s}");
    }
}

// ---- Large-scale rankdata correctness ----

#[test]
fn rankdata_large_descending_matches_sum_invariant() {
    let n = 1000;
    let x: Vec<f64> = (0..n).rev().map(|i| i as f64).collect();
    let r = rankdata_average_vec(&x);
    assert_eq!(r.len(), n);
    let sum: f64 = r.iter().sum();
    let expected_sum = n as f64 * (n as f64 + 1.0) / 2.0;
    assert!(approx_eq(sum, expected_sum, 1e-6), "rank sum should be {expected_sum}, got {sum}");
    assert!(approx_eq(r[0], 1000.0, 1e-12), "x[0]=999 should have rank 1000, got {}", r[0]);
    assert!(approx_eq(r[n - 1], 1.0, 1e-12), "x[999]=0 should have rank 1, got {}", r[n - 1]);
}

#[test]
fn rankdata_large_many_ties_matches_sum_invariant() {
    let n = 1000;
    // Only 10 distinct values, so lots of ties
    let x: Vec<f64> = (0..n).map(|i| (i % 10) as f64).collect();
    let r = rankdata_average_vec(&x);
    let sum: f64 = r.iter().sum();
    let expected_sum = n as f64 * (n as f64 + 1.0) / 2.0;
    assert!(approx_eq(sum, expected_sum, 1e-6), "rank sum with ties should be {expected_sum}, got {sum}");
    // Value 0 occupies sorted positions 1..100 → average rank 50.5.
    assert!(approx_eq(r[0], 50.5, 1e-10), "value 0 should have rank 50.5, got {}", r[0]);
}

// ---- Cook's distance numerical precision ----

#[test]
fn cooks_leverage_near_one_larger_than_low_leverage() {
    let yt = [1.0, 2.0, 3.0, 4.0, 5.0];
    let yp = [1.5, 1.5, 3.5, 3.5, 5.5];
    let h = [0.9999, 0.0001, 0.0, 0.0, 0.0]; // p_eff=1
    let cooks = cooks_distance_vec(&yt, &yp, &h);
    assert!(cooks[0].is_finite() && cooks[0] > 0.0,
        "leverage near 1 should give large but finite Cook's D, got {}", cooks[0]);
    assert!(cooks[0] > cooks[1],
        "high-leverage point should have larger Cook's D: {} vs {}", cooks[0], cooks[1]);
}

#[test]
fn cooks_nonnegative_brute_force() {
    let test_cases: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = vec![
        (vec![1.0, 2.0, 3.0, 4.0], vec![1.1, 2.1, 2.9, 3.9], vec![0.3, 0.2, 0.2, 0.3]),
        (vec![0.0, 0.0, 100.0, 100.0], vec![50.0, 50.0, 50.0, 50.0], vec![0.25, 0.25, 0.25, 0.25]),
        (vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![0.8, 0.05, 0.05, 0.05, 0.05]),
    ];
    for (idx, (yt, yp, h)) in test_cases.iter().enumerate() {
        let cooks = cooks_distance_vec(yt, yp, h);
        for (j, &c) in cooks.iter().enumerate() {
            assert!(c >= 0.0 && c.is_finite(), "test case {idx}, Cook's D[{j}] must be finite and non-negative, got {c}");
        }
    }
}

// ---- Variance rank with Inf propagation in multi-column ----

#[test]
fn variance_rank_two_inf_in_one_column_is_nan() {
    let flat = [
        f64::INFINITY, 1.0,
        f64::INFINITY, 2.0,
    ]; // n=2, p=2
    let v = variance_rank_vec(&flat, 2, 2);
    assert!(v[0].is_nan(), "two-Inf column variance should be NaN, got {}", v[0]);
    assert!(v[1].is_finite(), "normal column variance should be finite, got {}", v[1]);
}

#[test]
fn variance_rank_mixed_inf_is_nan() {
    // Inf + -Inf = NaN mean → var = NaN.
    let flat = [f64::INFINITY, f64::NEG_INFINITY]; // n=2, p=1
    let v = variance_rank_vec(&flat, 2, 1);
    assert!(v[0].is_nan(), "Inf + -Inf mean produces NaN → var should be NaN, got {}", v[0]);
}

// ---- Studentized residual with very large p_eff ----

#[test]
fn studentized_peff_much_larger_than_n_stays_finite() {
    let yt = [1.0, 2.0, 3.0];
    let yp = [1.5, 2.5, 2.5]; // residuals: [-0.5, -0.5, 0.5]
    let h = [100.0, 100.0, 100.0]; // sum = 300, p_eff = 300, n = 3
    let stud = studentized_residual_vec(&yt, &yp, Some(&h));
    assert_eq!(stud.len(), 3);
    for &s in &stud {
        assert!(s.is_finite(), "p_eff >> n: studentized should still be finite, got {s}");
    }
}

// ---- Covariance rank with perfectly anti-correlated data ----

#[test]
fn covariance_rank_anticorrelated_hand_computed() {
    // x_mean=3, y_mean=6.
    // cov = ((1-3)(10-6) + (2-3)(8-6) + (3-3)(6-6) + (4-3)(4-6) + (5-3)(2-6)) / 4
    //     = (-8 + -2 + 0 + -2 + -8) / 4 = -20/4 = -5. abs(-5) = 5.
    let flat = [1.0, 2.0, 3.0, 4.0, 5.0]; // n=5, p=1
    let y = [10.0, 8.0, 6.0, 4.0, 2.0]; // perfectly anti-correlated
    let result = covariance_rank_vec(&flat, 5, 1, &y);
    assert!(
        approx_eq(result[0], 5.0, 1e-10),
        "anti-correlated abs(cov) should be 5.0, got {}", result[0]
    );
}

// ---- Shapiro with repeated values / outliers ----

#[test]
fn shapiro_many_repeats_has_low_w() {
    let mut x = vec![0.0; 50];
    x.extend(vec![1.0; 50]);
    let w = shapiro_w_scalar(&x);
    assert!(w.is_finite() && w >= 0.0 && w <= 1.0, "binary data W should be in [0,1], got {w}");
    assert!(w < 0.8, "binary data should have low W (non-normal), got {w}");
}

#[test]
fn shapiro_one_extreme_outlier_no_panic() {
    let mut x: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
    x[19] = 1e10; // extreme outlier
    let w = shapiro_w_scalar(&x);
    assert!(w.is_finite() && w >= 0.0 && w <= 1.0, "one-outlier data W should be in [0,1], got {w}");
}

#[test]
fn silhouette_infinity_features_no_panic() {
    let flat = [
        0.0, 0.0,
        f64::INFINITY, 0.0,
        10.0, 10.0,
        11.0, 10.0,
    ];
    let labels = [0i64, 0, 1, 1];
    let sv = silhouette_samples_vec(&flat, 4, 2, &labels, DistanceMetric::Euclidean).unwrap();
    assert_eq!(sv.len(), 4);
}

// ---- Cook's distance with single observation ----

#[test]
fn cooks_n1_peff1_is_zero() {
    let yt = [5.0];
    let yp = [3.0];
    let h = [1.0]; // p_eff = 1, n = 1 → n <= p_eff → zeros
    let cooks = cooks_distance_vec(&yt, &yp, &h);
    assert_eq!(cooks.len(), 1);
    assert!(cooks[0] == 0.0, "n=1, p_eff=1: Cook's D should be 0, got {}", cooks[0]);
}

#[test]
fn cooks_n1_peff0_is_zero() {
    let yt = [5.0];
    let yp = [3.0];
    let h = [0.0]; // p_eff = 0
    let cooks = cooks_distance_vec(&yt, &yp, &h);
    assert_eq!(cooks.len(), 1);
    assert!(cooks[0] == 0.0, "n=1, p_eff=0: Cook's D should be 0, got {}", cooks[0]);
}

#[test]
fn cooks_nan_hdiag_peff_stays_finite() {
    // NaN sum → NaN.round() → `as usize` saturates to 0 on Rust → p_eff=0.
    let yt = [1.0, 2.0, 3.0, 4.0, 5.0];
    let yp = [1.1, 1.9, 3.2, 3.8, 5.1];
    let h = [0.2, f64::NAN, 0.2, 0.2, 0.2];
    let cooks = cooks_distance_vec(&yt, &yp, &h);
    assert_eq!(cooks.len(), 5);
    for (i, &c) in cooks.iter().enumerate() {
        assert!(c.is_finite(), "NaN in h_diag: Cook's D[{i}] should be finite, got {c}");
    }
}

// ---- Variance rank with single column having constant value ----

#[test]
fn variance_rank_constant_column_is_exactly_zero() {
    let flat = [42.0, 42.0, 42.0, 42.0, 42.0]; // n=5, p=1
    let v = variance_rank_vec(&flat, 5, 1);
    assert!(v[0] == 0.0, "constant column variance should be exactly 0.0, got {}", v[0]);
}

#[test]
fn variance_rank_constant_vs_varying_column() {
    let flat = [
        1.0, 5.0,
        2.0, 5.0,
        3.0, 5.0,
        4.0, 5.0,
        5.0, 5.0,
    ]; // n=5, p=2
    let v = variance_rank_vec(&flat, 5, 2);
    assert!(v[0] > 0.0, "varying column should have positive variance");
    assert!(v[1] == 0.0, "constant column should have zero variance");
}
