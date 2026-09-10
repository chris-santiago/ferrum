use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::transform::bin::resolve_bin_extent;
use crate::transform::bin_mode::{parse_bin_axis, BinMode};
use crate::transform::group_key::{
    deserialize_groupby_vec, group_partition_multi, groupby_field_nullable,
    materialize_groupby_col, parse_groupby_pyany, KeyValue,
};
use crate::transform::numeric_util::resolve_shared_extent;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum BandwidthSpec {
    Scott,
    Silverman,
    Fixed { value: f64 },
}

pub(crate) fn default_kernel() -> String { "gaussian".to_string() }

/// The single source of truth for `Kde`'s (and `DensityData`'s, which shares
/// this module's [`kernel_density_at_grid`] dispatch) free-string `kernel`
/// vocabulary. Mirrors `smooth::METHODS`'s structural-guard shape (batch-D
/// T9, extended to `kde`/`density_data` by T11): consumed by THREE sites —
/// `registry::entry_for_kind("kde")`'s `methods` list (`registry/mod.rs`, via
/// `KERNELS.to_vec()`), `PyKde::new`'s pre-check below, and
/// `kernel_density_at_grid`'s "unknown kernel" error (built FROM this slice,
/// not a hand-kept parallel literal).
///
/// The pre-check in `PyKde::new` makes the guarantee structural for `Kde`:
/// a dispatch arm added to `kernel_density_at_grid` for a new kernel without
/// adding it here is unreachable at construction time. `DensityData` has no
/// constructor-time kernel parameter at all (`PyDensityData::new` doesn't
/// expose one — `DensityDataSpec` is built directly from wire JSON), so for
/// that caller `kernel_density_at_grid`'s own `other => Err(...)` arm is the
/// *only* enforcement point; this is precisely why that arm returns a typed
/// error instead of silently defaulting to Gaussian (the prior behavior this
/// const and its guard replace).
///
/// The reverse direction (`KERNELS ⊆ kernel_density_at_grid`'s match arms —
/// an entry added here without a corresponding dispatch arm below) is NOT
/// compile-time-enforced, the same gap `smooth::METHODS`'s doc names for
/// itself: nothing in Rust's exhaustiveness checking ties a `&str` slice to
/// a `match` on a different `&str`. Unlike `Smooth` (where `PySmooth::new`'s
/// own match arm IS the dispatch, so the constructor test doubles as the
/// reverse guard), `PyKde::new`'s pre-check only proves
/// `KERNELS.contains(&kernel)` — it never calls `kernel_density_at_grid`, so
/// it cannot catch a `KERNELS` entry with no matching arm.
/// `test_all_kernels_reach_a_dispatch_arm` (below, round 2 batch-D T11 cycle
/// 2) closes this gap for real by iterating `KERNELS` itself through
/// `kernel_density_at_grid` directly.
pub(crate) const KERNELS: &[&str] = &[
    "gaussian", "epanechnikov", "epan", "tophat", "uniform", "cosine", "exponential", "linear",
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct KdeSpec {
    pub field: String,
    pub bandwidth: BandwidthSpec,
    /// Multiplier applied to the resolved bandwidth after rule evaluation.
    /// `bw_adjust = 1.0` is the identity. Works with all BandwidthSpec variants.
    #[serde(default = "default_bw_adjust")]
    pub bw_adjust: f64,
    pub n: usize,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub extent: Option<(f64, f64)>,
    #[serde(default)]
    pub cumulative: bool,
    /// When true, all groups share the same x-grid (required for stack/fill).
    #[serde(default)]
    pub shared_extent: bool,
    /// Kernel function for density estimation.
    /// Supported: "gaussian" (default), "epanechnikov" / "epan",
    /// "tophat" / "uniform", "cosine", "exponential", "linear".
    #[serde(default = "default_kernel")]
    pub kernel: String,
    /// When non-empty, partition input by these columns and emit per-(value, group)
    /// rows. Output schema gains one column per groupby field after the 2nd field.
    /// Wire accepts: `null` → `[]`, `"col"` → `["col"]`, `["a","b"]` → `["a","b"]`.
    #[serde(
        skip_serializing_if = "Vec::is_empty",
        default,
        deserialize_with = "deserialize_groupby_vec"
    )]
    pub groupby: Vec<String>,
    /// Opt-in histogram count-scaling (batch F, F-L09-10). When `Some`, the
    /// output `density` is rescaled per group to match the counts a
    /// histogram binned the same way would show: multiplied by
    /// `n_group × bin_width` (non-cumulative) or the CDF by `n_group`
    /// (cumulative). `bin_width` is derived PER GROUP via the exact same
    /// extent-fold/nicing/bin-count routine [`crate::transform::bin::Bin`]
    /// uses (see [`resolve_bin_extent`] and [`resolve_group_bin_width`]),
    /// mirroring `Bin::apply_grouped`'s own `shared_extent` gate (reusing
    /// this spec's existing `shared_extent` field, which Python sets from
    /// the same `multiple in ("stack","fill")` rule for both
    /// `mark_histogram` and `mark_density`) AND [`count_scale_extent`]'s
    /// facet pin (mirroring `Bin::apply`'s own `spec.extent` check — see
    /// that field's doc). The one remaining unmirrored input: `nice` is
    /// always assumed `true` (`Bin`'s own default, and the only value
    /// `mark_histogram` currently passes) — a `Bin` with `nice=False` would
    /// diverge silently, since `count_scale` carries no wire slot for it
    /// (quality-review cycle-4 finding B; no current caller can trigger this,
    /// so no slot was added speculatively). `None` (the default) reproduces
    /// today's output byte-for-byte.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub count_scale: Option<BinMode>,
    /// Facet-pinned extent for [`count_scale`]'s bin-width derivation —
    /// independent of [`extent`](Self::extent), which is the KDE's own grid
    /// domain. Set ONLY by `fix_transform_extents_for_facet`
    /// (`render/prepare/extent.rs`), mirroring the niced-global pin it gives
    /// `BinSpec::extent` (via `bin::global_extent` on a synthetic `BinSpec`
    /// built from this spec's `field`/`count_scale` mode) — never set by
    /// `Kde`'s own constructor, since a user has no direct way to express
    /// "the histogram's facet-shared bin range." When `Some`, takes absolute
    /// precedence over `shared_extent` in [`resolve_group_bin_width`]
    /// (mirroring `Bin::apply`'s own explicit-extent-wins precedence) and is
    /// used verbatim with no re-nicing. `None` (the default, and always true
    /// outside a faceted chart) falls back to `shared_extent`'s own gate.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub count_scale_extent: Option<(f64, f64)>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

pub(crate) fn default_bw_adjust() -> f64 { 1.0 }

/// Compute the global `(lo, hi)` extent of `spec.field` over the full `batch`.
///
/// Returns `None` when the field is missing, non-numeric, or all values are
/// null/NaN. This is the pre-facet extent that `render::prepare` uses to pin the
/// value axis before partitioning, so every facet panel shares the same KDE grid
/// range (see `fix_transform_extents_for_facet`).
///
/// NICENESS CONTRACT (XFORM-08): returns the RAW `(lo, hi)` (no nicing), unlike
/// the `Bin` sibling. KDE controls a continuous grid start/end, not discrete bin
/// edges, so the pinned extent is the raw data range. Uses the shared
/// `column_extent` helper, which coerces integer-typed fields just as
/// `apply_one_group` does.
pub(crate) fn global_extent(spec: &KdeSpec, batch: &RecordBatch) -> Option<(f64, f64)> {
    crate::transform::numeric_util::column_extent(batch, &spec.field)
}

/// Compute the global extent [`KdeSpec::count_scale_extent`] should be
/// pinned to for `spec` — i.e. the SAME niced-global extent
/// `fix_transform_extents_for_facet` gives `BinSpec::extent` for the
/// equivalent histogram (quality-review cycle-4 finding B). Returns `None`
/// when `spec.count_scale` is absent (nothing to pin) or the field yields no
/// extent (missing/non-numeric/all-null).
///
/// Builds a synthetic [`crate::transform::bin::BinSpec`] from `spec.field`
/// and `spec.count_scale`'s mode and delegates to
/// [`crate::transform::bin::global_extent`] — the SAME routine `Bin`'s own
/// facet-pin arm calls — rather than re-deriving the nice/target logic here.
/// `nice: true` matches `Bin`'s own default and the only value
/// `mark_histogram` currently passes (see `KdeSpec::count_scale`'s doc for
/// the one input this does NOT mirror: an explicit `Bin(nice=False)`).
pub(crate) fn count_scale_global_extent(spec: &KdeSpec, batch: &RecordBatch) -> Option<(f64, f64)> {
    let mode = spec.count_scale.as_ref()?;
    let synthetic_bin_spec = crate::transform::bin::BinSpec {
        field: spec.field.clone(),
        mode: mode.clone(),
        extent: None,
        nice: true,
        cumulative: false,
        shared_extent: false,
        groupby: vec![],
        agg_field: None,
        agg: None,
        name: None,
    };
    crate::transform::bin::global_extent(&synthetic_bin_spec, batch)
}

pub(crate) fn apply(spec: &KdeSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    if !spec.groupby.is_empty() {
        // `count_scale`'s bin-edge sharing mirrors `Bin::apply_grouped`'s own
        // `resolve_shared_extent(spec.extent, spec.shared_extent, ...)` call
        // exactly, with `count_scale_extent` standing in for `Bin`'s own
        // `spec.extent` role (an explicit/facet-pinned extent always wins,
        // regardless of `shared_extent`'s value — see `count_scale_extent`'s
        // doc): a real histogram only shares one DATA-DERIVED extent across
        // hue levels when `shared_extent` is set (Python sets it from the
        // SAME `multiple in ("stack","fill")` rule for both
        // `mark_histogram` and `mark_density` — see `KdeSpec::count_scale`'s
        // doc). The default `multiple="layer"` leaves both `None`, so each
        // group derives its own extent, exactly like an un-shared `Bin`.
        // `shared_extent` is otherwise inert when ungrouped (real `Bin`'s
        // ungrouped `apply()` never consults it either), so it is only
        // folded in here on the grouped branch.
        let count_scale_pinned_extent = spec.count_scale.as_ref().and_then(|_| {
            resolve_shared_extent(spec.count_scale_extent, spec.shared_extent, batch, &spec.field)
        });
        return apply_grouped(spec, batch, &spec.groupby, count_scale_pinned_extent);
    }
    // `count_scale_extent`'s facet pin mirrors `Bin::apply`'s own
    // `spec.extent` check, which is NOT gated on grouping (unlike
    // `shared_extent`) — an ungrouped, faceted `Kde` must still honor it.
    let count_scale_pinned_extent = spec.count_scale.as_ref().and(spec.count_scale_extent);
    apply_one_group(spec, batch, None, count_scale_pinned_extent)
}

/// Resolve the histogram bin width `Bin` would compute for THIS group's own
/// clean (non-null/non-NaN) values, given `mode` and an optional pinned
/// extent (either a facet-pinned `count_scale_extent` or a `shared_extent`
/// data-derived extent — see `apply()`'s call site for which). Mirrors
/// `Bin::apply_one_group`'s extent-fold, degenerate-collapse, nice-rounding,
/// and bin-count decision via [`resolve_bin_extent`] — the SAME per-group
/// derivation `Bin::apply_grouped` uses (batch F, F-L09-10 fix): a `None`
/// pinned extent means this group derives its own `(lo, hi)` AND its own
/// nice target from its own `clean.len()`, exactly like an un-shared `Bin`;
/// a `Some((lo, hi))` pinned extent is used verbatim with no nicing
/// (mirroring `Bin::apply_grouped`'s `Cow`-patched explicit-extent path,
/// which skips `apply_one_group`'s nicing branch since `nice` only applies
/// when `extent.is_none()`) — bin-count resolution (and hence width) still
/// depends on THIS group's own `clean.len()` for Sturges/FreedmanDiaconis, so
/// width can still differ per group even when the extent is pinned. Returns
/// `Ok(None)` — "nothing to scale by" — for an empty `clean`, rather than a
/// NaN sentinel; the caller already leaves `n_group=0`'s factor unapplied in
/// that case since the group's own density is already all-NaN.
fn resolve_group_bin_width(
    clean: &[f64],
    mode: &BinMode,
    pinned_extent: Option<(f64, f64)>,
    ctx: &str,
) -> PyResult<Option<f64>> {
    if clean.is_empty() {
        return Ok(None);
    }
    let (lo, hi, should_nice) = match pinned_extent {
        Some((lo, hi)) => (lo, hi, false),
        None => {
            let (lo, hi) = clean
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
            if lo == hi {
                // Matches `Bin`'s degenerate all-equal collapse to a single
                // unit-width bin (`single_unit_bin`).
                return Ok(Some(1.0));
            }
            (lo, hi, true)
        }
    };
    let (lo, hi, n_bins) = resolve_bin_extent(clean, mode, lo, hi, should_nice, ctx)?;
    Ok(Some((hi - lo) / n_bins as f64))
}

fn apply_one_group(
    spec: &KdeSpec,
    batch: &RecordBatch,
    only_indices: Option<&[usize]>,
    count_scale_pinned_extent: Option<(f64, f64)>,
) -> PyResult<RecordBatch> {
    let schema = batch.schema();
    let idx = schema.index_of(&spec.field).map_err(|_| {
        PyValueError::new_err(format!("stat_kde: column '{}' not found", spec.field))
    })?;
    let arr = crate::transform::numeric_util::coerce_to_float64(
        batch.column(idx),
        "stat_kde",
        &spec.field,
    )?;
    let clean = crate::transform::numeric_util::clean_float64_values(&arr, only_indices);

    let (lo, hi) = match spec.extent {
        Some((a, b)) => (a, b),
        None => {
            if clean.is_empty() {
                (0.0, 0.0)
            } else {
                clean.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                    (a.min(v), b.max(v))
                })
            }
        }
    };

    let grid: Vec<f64> = (0..spec.n)
        .map(|i| {
            if spec.n <= 1 {
                lo
            } else {
                lo + (hi - lo) * (i as f64) / ((spec.n - 1) as f64)
            }
        })
        .collect();

    let density: Vec<f64> = if clean.len() < 2 {
        vec![f64::NAN; spec.n]
    } else {
        let h = bandwidth(&clean, &spec.bandwidth)? * spec.bw_adjust;
        if h <= 0.0 || !h.is_finite() {
            vec![f64::NAN; spec.n]
        } else {
            kernel_density_at_grid(&clean, h, &grid, &spec.kernel)?
        }
    };

    let mut density = if spec.cumulative {
        trapezoidal_cumulative(&grid, &density)
    } else {
        density
    };

    // Batch-F count-scale (F-L09-10): non-cumulative density scales by
    // n_group × bin_width; cumulative CDF scales by n_group alone (see
    // `KdeSpec::count_scale`'s doc for the normative semantics). `n_group`
    // is this group's own clean (non-null/non-NaN) count — the same
    // null-drop policy `Bin` applies — so it matches the bar-height total a
    // per-hue histogram would show for this group. `bin_width` is THIS
    // group's own width (see `resolve_group_bin_width`), not a single
    // scalar shared across every group.
    if let Some(mode) = &spec.count_scale {
        if let Some(bw) = resolve_group_bin_width(&clean, mode, count_scale_pinned_extent, "stat_kde")? {
            let n_group = clean.len() as f64;
            let factor = if spec.cumulative { n_group } else { n_group * bw };
            for d in density.iter_mut() {
                *d *= factor;
            }
        }
    }

    let out_schema = Arc::new(Schema::new(vec![
        Field::new("value", DataType::Float64, false),
        Field::new("density", DataType::Float64, true),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(grid)),
        Arc::new(Float64Array::from(density)),
    ];
    RecordBatch::try_new(out_schema, cols)
        .map_err(|e| PyValueError::new_err(format!("stat_kde: {e}")))
}

/// Partition input batch by the groupby columns, call apply_one_group per
/// partition, then stack the results into a single batch with the group
/// columns appended after the 2nd field.
fn apply_grouped(
    spec: &KdeSpec,
    batch: &RecordBatch,
    group_cols: &[String],
    count_scale_pinned_extent: Option<(f64, f64)>,
) -> PyResult<RecordBatch> {
    // Multi-column partition path (first-appearance order, null-key skip, dtype-preserve).
    let (group_order, group_idx_map, gtypes) =
        group_partition_multi(batch, group_cols, "stat_kde")?;

    // Shared vs per-group extent (stack/fill aligns to one global extent;
    // layer mode keeps per-group extents). resolve_shared_extent returns the
    // explicit extent unchanged, else the global extent when shared, else None.
    let global_extent = resolve_shared_extent(spec.extent, spec.shared_extent, batch, &spec.field);
    let shared_spec = KdeSpec {
        extent: global_extent.or(spec.extent),
        ..spec.clone()
    };

    let mut all_values: Vec<f64> = Vec::new();
    let mut all_density: Vec<f64> = Vec::new();
    let mut group_keys_out: Vec<Vec<KeyValue>> = Vec::new();
    for g in &group_order {
        let ixs = group_idx_map.get(g)
            .ok_or_else(|| PyValueError::new_err(format!(
                "stat_kde: missing group key {g:?} in index map")))?;
        let out = apply_one_group(&shared_spec, batch, Some(ixs), count_scale_pinned_extent)?;
        let n = out.num_rows();
        let values = out.column(0).as_any().downcast_ref::<Float64Array>()
            .ok_or_else(|| PyValueError::new_err("stat_kde: expected Float64Array for 'value' output column"))?;
        let density = out.column(1).as_any().downcast_ref::<Float64Array>()
            .ok_or_else(|| PyValueError::new_err("stat_kde: expected Float64Array for 'density' output column"))?;
        for i in 0..n {
            all_values.push(values.value(i));
            all_density.push(if density.is_null(i) { f64::NAN } else { density.value(i) });
            group_keys_out.push(g.clone());
        }
    }

    let mut fields = vec![
        Field::new("value", DataType::Float64, false),
        Field::new("density", DataType::Float64, true),
    ];
    // FA-7: preserve the original groupby dtypes (nullable per FA-9; null keys
    // are skipped upstream in group_partition_multi).
    for (name, dtype) in group_cols.iter().zip(gtypes.iter()) {
        fields.push(Field::new(name, dtype.clone(), groupby_field_nullable()));
    }
    let out_schema = Arc::new(Schema::new(fields));
    let mut cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(all_values)),
        Arc::new(Float64Array::from(all_density)),
    ];
    for (gi, dtype) in gtypes.iter().enumerate() {
        let group_col_arr = materialize_groupby_col(&group_keys_out, gi, dtype)
            .map_err(PyValueError::new_err)?;
        cols.push(group_col_arr);
    }
    RecordBatch::try_new(out_schema, cols)
        .map_err(|e| PyValueError::new_err(format!("stat_kde: {e}")))
}

pub(crate) fn bandwidth(x: &[f64], spec: &BandwidthSpec) -> PyResult<f64> {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let sigma = var.sqrt();
    Ok(match spec {
        BandwidthSpec::Scott => sigma * n.powf(-0.2),
        BandwidthSpec::Silverman => {
            let mut sorted = x.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let q25 = percentile(&sorted, 0.25);
            let q75 = percentile(&sorted, 0.75);
            let iqr = q75 - q25;
            0.9 * sigma.min(iqr / 1.34) * n.powf(-0.2)
        }
        BandwidthSpec::Fixed { value } => *value,
    })
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    // numpy linear-interpolation quantile.
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    let h = p * (n as f64 - 1.0);
    let lo = h.floor() as usize;
    let hi = (h.ceil() as usize).min(n - 1);
    let frac = h - h.floor();
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Dispatch to the requested kernel's density estimator. Shared by `Kde`
/// (`apply_one_group` above) and `DensityData`
/// (`density_data::apply_one_group`) so the crate has exactly one
/// kernel-math table. Returns a typed error for any kernel string not in
/// [`KERNELS`] — no silent fallback to Gaussian; see `KERNELS`'s doc comment
/// for why this is the only enforcement point for `DensityData`, which has
/// no constructor-time kernel validation of its own.
pub(crate) fn kernel_density_at_grid(
    x: &[f64],
    h: f64,
    grid: &[f64],
    kernel: &str,
) -> PyResult<Vec<f64>> {
    Ok(match kernel {
        "gaussian" => gaussian_density_at_grid(x, h, grid),
        "epanechnikov" | "epan" => epanechnikov_density_at_grid(x, h, grid),
        "tophat" | "uniform" => tophat_density_at_grid(x, h, grid),
        "cosine" => cosine_density_at_grid(x, h, grid),
        "exponential" => exponential_density_at_grid(x, h, grid),
        "linear" => linear_density_at_grid(x, h, grid),
        other => {
            return Err(PyValueError::new_err(format!(
                "kernel_density_at_grid: unknown kernel '{other}'; expected {}",
                KERNELS.iter().map(|k| format!("'{k}'")).collect::<Vec<_>>().join(" | "),
            )))
        }
    })
}

pub(crate) fn gaussian_density_at_grid(x: &[f64], h: f64, grid: &[f64]) -> Vec<f64> {
    let n = x.len() as f64;
    let norm = 1.0 / (n * h * (2.0 * std::f64::consts::PI).sqrt());
    grid.iter()
        .map(|&g| {
            let s: f64 = x
                .iter()
                .map(|&xi| {
                    let z = (g - xi) / h;
                    (-0.5 * z * z).exp()
                })
                .sum();
            norm * s
        })
        .collect()
}

/// Epanechnikov kernel: K(u) = (3/4)(1 - u²) for |u| ≤ 1, else 0.
/// Integrates to 1 over [-1, 1] and is optimal in MSE sense.
fn epanechnikov_density_at_grid(x: &[f64], h: f64, grid: &[f64]) -> Vec<f64> {
    let n = x.len() as f64;
    // The kernel integrates to 1 over [-h, h], so no extra h factor in norm.
    let norm = 1.0 / (n * h);
    grid.iter()
        .map(|&g| {
            let s: f64 = x
                .iter()
                .map(|&xi| {
                    let u = (g - xi) / h;
                    if u.abs() <= 1.0 {
                        0.75 * (1.0 - u * u)
                    } else {
                        0.0
                    }
                })
                .sum();
            norm * s
        })
        .collect()
}

/// Tophat (uniform) kernel: K(u) = 0.5 for |u| ≤ 1, else 0.
fn tophat_density_at_grid(x: &[f64], h: f64, grid: &[f64]) -> Vec<f64> {
    let n = x.len() as f64;
    let norm = 1.0 / (n * h);
    grid.iter()
        .map(|&g| {
            let s: f64 = x
                .iter()
                .map(|&xi| {
                    let u = (g - xi) / h;
                    if u.abs() <= 1.0 { 0.5 } else { 0.0 }
                })
                .sum();
            norm * s
        })
        .collect()
}

/// Cosine kernel: K(u) = (π/4)·cos(π·u/2) for |u| ≤ 1, else 0.
fn cosine_density_at_grid(x: &[f64], h: f64, grid: &[f64]) -> Vec<f64> {
    use std::f64::consts::PI;
    let n = x.len() as f64;
    let norm = 1.0 / (n * h);
    grid.iter()
        .map(|&g| {
            let s: f64 = x
                .iter()
                .map(|&xi| {
                    let u = (g - xi) / h;
                    if u.abs() <= 1.0 {
                        (PI / 4.0) * ((PI / 2.0) * u).cos()
                    } else {
                        0.0
                    }
                })
                .sum();
            norm * s
        })
        .collect()
}

/// Exponential (double-exponential / Laplace) kernel: K(u) = (1/2)·exp(-|u|).
/// Unlike the compact-support kernels above, this decays exponentially with
/// no hard cutoff (support is all of ℝ) — it has a cusp (non-differentiable
/// peak) at u=0, unlike Gaussian's smooth peak. Integrates to 1 over
/// (-∞, ∞), so no truncation-dependent normalization is needed.
fn exponential_density_at_grid(x: &[f64], h: f64, grid: &[f64]) -> Vec<f64> {
    let n = x.len() as f64;
    let norm = 1.0 / (n * h);
    grid.iter()
        .map(|&g| {
            let s: f64 = x
                .iter()
                .map(|&xi| {
                    let u = (g - xi) / h;
                    0.5 * (-u.abs()).exp()
                })
                .sum();
            norm * s
        })
        .collect()
}

/// Linear (triangular) kernel: K(u) = 1 - |u| for |u| ≤ 1, else 0. Compact
/// support with hard edges at ±h, tapering linearly to 0 (unlike
/// Epanechnikov's parabolic taper). Integrates to 1 over [-1, 1].
fn linear_density_at_grid(x: &[f64], h: f64, grid: &[f64]) -> Vec<f64> {
    let n = x.len() as f64;
    let norm = 1.0 / (n * h);
    grid.iter()
        .map(|&g| {
            let s: f64 = x
                .iter()
                .map(|&xi| {
                    let u = (g - xi) / h;
                    if u.abs() <= 1.0 {
                        1.0 - u.abs()
                    } else {
                        0.0
                    }
                })
                .sum();
            norm * s
        })
        .collect()
}

fn trapezoidal_cumulative(grid: &[f64], y: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(grid.len());
    out.push(0.0);
    for i in 1..grid.len() {
        let dx = grid[i] - grid[i - 1];
        let avg = 0.5 * (y[i] + y[i - 1]);
        out.push(out[i - 1] + avg * dx);
    }
    out
}

use pyo3::prelude::*;

use crate::transform::core::TransformSpec;

/// One-dimensional kernel density estimate (KDE).
///
/// Fits a Gaussian kernel to ``field`` and evaluates the density on an
/// evenly-spaced grid of ``n`` points spanning (or clipped to) ``extent``.
/// The result is used to draw smooth density curves in violin plots and
/// ridge plots.
///
/// Output columns: ``value`` (Float64 grid points) and ``density``
/// (Float64 kernel density values, integrates to 1 over the grid range).
///
/// Parameters
/// ----------
/// field : str
///     Numeric column to estimate (must be Float64).
/// bandwidth : float or {"scott", "silverman"}, optional
///     Kernel bandwidth. A float sets a fixed bandwidth; ``"scott"`` and
///     ``"silverman"`` use the corresponding automatic rules. Default is
///     ``"scott"``.
/// n : int, default 512
///     Number of evaluation grid points. Must be > 0.
/// extent : (float, float), optional
///     ``(lo, hi)`` range to evaluate over; defaults to the data min/max.
///     Both values must be finite and ``lo < hi``.
/// cumulative : bool, default False
///     When True, output is the cumulative distribution function (CDF)
///     rather than the PDF.
/// kernel : str, default "gaussian"
///     Kernel function. One of ``"gaussian"``, ``"epanechnikov"`` /
///     ``"epan"``, ``"tophat"`` / ``"uniform"``, ``"cosine"``,
///     ``"exponential"``, ``"linear"``.
/// groupby : list[str] or str or None, optional
///     One or more group-key columns; KDE computed independently per group.
///     Output schema gains one column per groupby field after the 2nd field.
/// count_scale : str, int, float, or None, optional
///     Opt-in histogram count-scaling. When set, rescales ``density`` per
///     group to match the counts a histogram of ``field`` binned the same
///     way would show: ``density × n_group × bin_width`` (or the CDF ×
///     ``n_group`` when ``cumulative=True``). Accepts the same binning-rule
///     vocabulary as ``Bin2D``'s ``bins_x``/``bins_y``: a bin-count rule
///     string (``"sturges"`` or ``"fd"``), an ``int`` (fixed bin count), or
///     a ``float`` (fixed bin width). ``bin_width`` is derived PER GROUP
///     (mirroring ``Bin``'s own grouped derivation and this spec's
///     ``shared_extent`` gate), using the same routine ``Bin`` uses — bit
///     parity with a histogram binned the same way, including per-hue-level
///     widths when ``shared_extent=False`` (the ``displot`` default).
///     Default ``None`` leaves ``density`` unscaled (today's behavior).
/// name : str, optional
///     Named output label for sibling ``Reorder(from_=...)`` lookup.
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> fm.Chart(df).mark_density().encode(x="val")
///
/// Direct construction (custom bandwidth):
///
/// >>> fm.Chart(df).mark_area().encode(
/// ...     x="x_kde", y="density",
/// ...     transform=fm.Kde("val", bandwidth=0.3),
/// ... )
#[pyclass(eq, module = "ferrum._core", name = "Kde")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyKde(pub(crate) TransformSpec);

#[pymethods]
impl PyKde {
    #[new]
    #[pyo3(signature = (field, *, bandwidth = None, bw_adjust = 1.0, n = 512, extent = None, cumulative = false, shared_extent = false, kernel = "gaussian", groupby = None, count_scale = None, name = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        field: &str,
        bandwidth: Option<&Bound<'_, PyAny>>,
        bw_adjust: f64,
        n: usize,
        extent: Option<(f64, f64)>,
        cumulative: bool,
        shared_extent: bool,
        kernel: &str,
        groupby: Option<&Bound<'_, PyAny>>,
        count_scale: Option<&Bound<'_, PyAny>>,
        name: Option<String>,
    ) -> PyResult<Self> {
        if !bw_adjust.is_finite() || bw_adjust <= 0.0 {
            return Err(PyValueError::new_err("Kde: bw_adjust must be a positive finite number"));
        }
        if field.is_empty() {
            return Err(PyValueError::new_err("Kde: field must be non-empty"));
        }
        if n == 0 {
            return Err(PyValueError::new_err("Kde: n must be > 0"));
        }
        let bw = match bandwidth {
            None => BandwidthSpec::Scott,
            Some(obj) => {
                if let Ok(s) = obj.extract::<String>() {
                    match s.as_str() {
                        "scott" => BandwidthSpec::Scott,
                        "silverman" => BandwidthSpec::Silverman,
                        other => return Err(PyValueError::new_err(format!(
                            "Kde: unknown bandwidth '{other}'; expected 'scott' | 'silverman' | float"
                        ))),
                    }
                } else if let Ok(v) = obj.extract::<f64>() {
                    if !v.is_finite() || v <= 0.0 {
                        return Err(PyValueError::new_err(
                            "Kde: numeric bandwidth must be a positive finite number",
                        ));
                    }
                    BandwidthSpec::Fixed { value: v }
                } else {
                    return Err(PyValueError::new_err(
                        "Kde: bandwidth must be 'scott', 'silverman', or a positive float",
                    ));
                }
            }
        };
        if let Some((a, b)) = extent {
            if !a.is_finite() || !b.is_finite() || a >= b {
                return Err(PyValueError::new_err(
                    "Kde: extent must be (lo, hi) with lo < hi and both finite",
                ));
            }
        }
        if !KERNELS.contains(&kernel) {
            return Err(PyValueError::new_err(format!(
                "Kde: unknown kernel '{kernel}'; expected {}",
                KERNELS.iter().map(|k| format!("'{k}'")).collect::<Vec<_>>().join(" | "),
            )));
        }
        let kernel_str = kernel.to_string();
        let groupby = parse_groupby_pyany(groupby)?;
        let count_scale = count_scale.map(|obj| parse_bin_axis(obj, "stat_kde")).transpose()?;
        Ok(PyKde(TransformSpec::Kde(KdeSpec {
            field: field.to_string(),
            bandwidth: bw,
            bw_adjust,
            n,
            extent,
            cumulative,
            shared_extent,
            kernel: kernel_str,
            groupby,
            count_scale,
            // `count_scale_extent` has no constructor parameter — it is set
            // ONLY by `fix_transform_extents_for_facet`'s facet-pinning
            // pass (see its doc). A user-constructed `Kde` always starts
            // `None`.
            count_scale_extent: None,
            name,
        })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Kde(s) => format!(
                "Kde(field='{}', bandwidth={:?}, n={}, extent={:?}, cumulative={}, kernel='{}')",
                s.field, s.bandwidth, s.n, s.extent,
                if s.cumulative { "True" } else { "False" },
                s.kernel,
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Float64Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use serde::Deserialize;
    use std::sync::Arc;

    const FIXTURES: &str = include_str!("fixtures/stat_refs.json");

    #[derive(Deserialize)]
    struct KdeCase {
        name: String,
        input: Vec<f64>,
        bandwidth: String,
        #[serde(default)]
        fixed_bandwidth: Option<f64>,
        n: usize,
        extent: [f64; 2],
        cumulative: bool,
        expected_bandwidth: f64,
        value: Vec<f64>,
        density: Vec<f64>,
    }

    #[derive(Deserialize)]
    struct Fixtures {
        kde: Vec<KdeCase>,
    }

    fn load_kde() -> Vec<KdeCase> {
        let f: Fixtures = serde_json::from_str(FIXTURES).unwrap();
        f.kde
    }

    fn batch_with(name: &str, values: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::Float64, true)]));
        RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(values))]).unwrap()
    }

    fn col(b: &RecordBatch, name: &str) -> Vec<f64> {
        let arr = b
            .column(b.schema().index_of(name).unwrap())
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        (0..arr.len())
            .map(|i| if arr.is_null(i) { f64::NAN } else { arr.value(i) })
            .collect()
    }

    #[test]
    fn test_kde_against_fixtures_within_tolerance() {
        for case in load_kde() {
            let bw = match (case.bandwidth.as_str(), case.fixed_bandwidth) {
                ("scott", _) => BandwidthSpec::Scott,
                ("silverman", _) => BandwidthSpec::Silverman,
                ("fixed", Some(v)) => BandwidthSpec::Fixed { value: v },
                other => panic!("unknown bandwidth spec: {other:?}"),
            };
            let spec = KdeSpec {
                field: "x".into(),
                bandwidth: bw,
                bw_adjust: 1.0,
                shared_extent: false,
                n: case.n,
                extent: Some((case.extent[0], case.extent[1])),
                cumulative: case.cumulative,
                kernel: default_kernel(),
                groupby: vec![],
                count_scale: None,
                count_scale_extent: None,
                name: None,
            };
            let batch = batch_with("x", case.input.clone());
            let out = apply(&spec, &batch).unwrap();
            let got_value = col(&out, "value");
            let got_density = col(&out, "density");
            for i in 0..case.n {
                assert!(
                    (got_value[i] - case.value[i]).abs() < 1e-9,
                    "case {} value[{i}]: got {} vs expected {}",
                    case.name,
                    got_value[i],
                    case.value[i]
                );
                assert!(
                    (got_density[i] - case.density[i]).abs() < 1e-6,
                    "case {} density[{i}]: got {} vs expected {} (diff {})",
                    case.name,
                    got_density[i],
                    case.density[i],
                    (got_density[i] - case.density[i]).abs()
                );
            }
            // Suppress unused field warning for expected_bandwidth
            let _ = case.expected_bandwidth;
        }
    }

    #[test]
    fn test_kde_zero_variance_emits_nan_densities() {
        let batch = batch_with("x", vec![3.0, 3.0, 3.0]);
        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Scott,
                bw_adjust: 1.0,
                shared_extent: false,
            n: 16,
            extent: Some((0.0, 6.0)),
            cumulative: false,
            kernel: default_kernel(),
            groupby: vec![],
            count_scale: None,
            count_scale_extent: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let density = col(&out, "density");
        assert!(density.iter().all(|d| d.is_nan()), "expected all-NaN densities");
    }

    #[test]
    fn test_kde_n_lt_2_emits_nan_densities() {
        let batch = batch_with("x", vec![1.0]);
        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Scott,
                bw_adjust: 1.0,
                shared_extent: false,
            n: 8,
            extent: Some((0.0, 2.0)),
            cumulative: false,
            kernel: default_kernel(),
            groupby: vec![],
            count_scale: None,
            count_scale_extent: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let density = col(&out, "density");
        assert!(density.iter().all(|d| d.is_nan()));
    }

    #[test]
    fn test_kde_grouped_preserves_group_column_and_per_group_density() {
        use arrow::array::StringArray;
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("g", DataType::Utf8, false),
        ]));
        // Two groups, well-separated means so per-group bandwidth differs from
        // global bandwidth.
        let xs = Float64Array::from(vec![
            0.0, 0.5, 1.0, 1.5, 2.0,        // group A
            10.0, 10.5, 11.0, 11.5, 12.0,    // group B
        ]);
        let gs = StringArray::from(vec!["A", "A", "A", "A", "A",
                                          "B", "B", "B", "B", "B"]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(xs), Arc::new(gs)]).unwrap();
        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Scott,
                bw_adjust: 1.0,
                shared_extent: false,
            n: 8,
            extent: None,
            cumulative: false,
            kernel: default_kernel(),
            groupby: vec!["g".into()],
            count_scale: None,
            count_scale_extent: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        // 2 groups × 8 grid points = 16 rows
        assert_eq!(out.num_rows(), 16);
        // Schema must include the group column as the 3rd field.
        let out_schema = out.schema();
        assert_eq!(out_schema.field(0).name(), "value");
        assert_eq!(out_schema.field(1).name(), "density");
        assert_eq!(out_schema.field(2).name(), "g");
        // Per-group grids span per-group extents.
        let values = col(&out, "value");
        let groups = out.column(2).as_any().downcast_ref::<StringArray>().unwrap();
        // Rows 0..8 are group A (values 0.0..2.0); rows 8..16 are group B (10.0..12.0).
        for i in 0..8 { assert_eq!(groups.value(i), "A"); }
        for i in 8..16 { assert_eq!(groups.value(i), "B"); }
        assert!((values[0] - 0.0).abs() < 1e-9, "group A grid starts at min");
        assert!((values[7] - 2.0).abs() < 1e-9, "group A grid ends at max");
        assert!((values[8] - 10.0).abs() < 1e-9, "group B grid starts at min");
        assert!((values[15] - 12.0).abs() < 1e-9, "group B grid ends at max");
    }

    #[test]
    fn test_kde_grouped_int64_group_column() {
        // T0.3 regression: a non-Utf8 (Int64) group column must work, not raise
        // "must be Utf8". Pre-fix this errored in kde::apply_grouped.
        use arrow::array::Int64Array;
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("g", DataType::Int64, false),
        ]));
        let xs = Float64Array::from(vec![
            0.0, 0.5, 1.0, 1.5, 2.0, // group 1
            10.0, 10.5, 11.0, 11.5, 12.0, // group 2
        ]);
        let gs = Int64Array::from(vec![1_i64, 1, 1, 1, 1, 2, 2, 2, 2, 2]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(xs), Arc::new(gs)]).unwrap();
        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            shared_extent: false,
            n: 8,
            extent: None,
            cumulative: false,
            kernel: default_kernel(),
            groupby: vec!["g".into()],
            count_scale: None,
            count_scale_extent: None,
            name: None,
        };
        let out = apply(&spec, &batch).expect("Int64 groupby must succeed");
        assert_eq!(out.num_rows(), 16, "2 groups × 8 grid points");
        // Group column must round-trip as Int64, NOT String.
        assert_eq!(out.schema().field(2).name(), "g");
        assert_eq!(out.schema().field(2).data_type(), &DataType::Int64);
        let groups = out.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
        for i in 0..8 {
            assert_eq!(groups.value(i), 1, "first 8 rows are group 1");
        }
        for i in 8..16 {
            assert_eq!(groups.value(i), 2, "last 8 rows are group 2");
        }
    }

    #[test]
    fn test_kde_grouped_boolean_group_column() {
        // T0.3 regression: a Boolean group column must work (was "must be Utf8").
        use arrow::array::BooleanArray;
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("g", DataType::Boolean, false),
        ]));
        let xs = Float64Array::from(vec![
            0.0, 0.5, 1.0, 1.5, 2.0, // false group
            10.0, 10.5, 11.0, 11.5, 12.0, // true group
        ]);
        let gs = BooleanArray::from(vec![
            false, false, false, false, false, true, true, true, true, true,
        ]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(xs), Arc::new(gs)]).unwrap();
        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            shared_extent: false,
            n: 8,
            extent: None,
            cumulative: false,
            kernel: default_kernel(),
            groupby: vec!["g".into()],
            count_scale: None,
            count_scale_extent: None,
            name: None,
        };
        let out = apply(&spec, &batch).expect("Boolean groupby must succeed");
        assert_eq!(out.num_rows(), 16);
        assert_eq!(out.schema().field(2).data_type(), &DataType::Boolean);
        let groups = out.column(2).as_any().downcast_ref::<BooleanArray>().unwrap();
        // First-appearance order → false group first, then true.
        for i in 0..8 {
            assert!(!groups.value(i), "first 8 rows are the false group");
        }
        for i in 8..16 {
            assert!(groups.value(i), "last 8 rows are the true group");
        }
    }

    #[test]
    fn test_kde_ungrouped_output_schema_unchanged() {
        // Sentinel: ungrouped output schema MUST stay [value, density] (2 cols)
        // so existing goldens stay byte-identical.
        let batch = batch_with("x", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Scott,
                bw_adjust: 1.0,
                shared_extent: false,
            n: 4,
            extent: Some((0.0, 6.0)),
            cumulative: false,
            kernel: default_kernel(),
            groupby: vec![],
            count_scale: None,
            count_scale_extent: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_columns(), 2);
        assert_eq!(out.schema().field(0).name(), "value");
        assert_eq!(out.schema().field(1).name(), "density");
    }

    #[test]
    fn test_kde_grouped_missing_column_errors() {
        pyo3::Python::initialize();
        let batch = batch_with("x", vec![1.0, 2.0, 3.0]);
        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Scott,
                bw_adjust: 1.0,
                shared_extent: false,
            n: 4,
            extent: None,
            cumulative: false,
            kernel: default_kernel(),
            groupby: vec!["ghost".into()],
            count_scale: None,
            count_scale_extent: None,
            name: None,
        };
        let err = apply(&spec, &batch).unwrap_err();
        assert!(err.to_string().contains("ghost"), "err: {err}");
    }

    #[test]
    fn test_kde_int64_column_is_coerced() {
        // Int64 input must be coerced to Float64, not rejected. The density
        // output must match a Float64 batch of the same values bit-for-bit.
        use arrow::array::Int64Array;
        pyo3::Python::initialize();
        let vals_i: Vec<i64> = (0..10).collect();
        let vals_f: Vec<f64> = vals_i.iter().map(|&v| v as f64).collect();
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
        let int_batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vals_i))]).unwrap();
        let float_batch = batch_with("x", vals_f);
        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            shared_extent: false,
            n: 16,
            extent: None,
            cumulative: false,
            kernel: default_kernel(),
            groupby: vec![],
            count_scale: None,
            count_scale_extent: None,
            name: None,
        };
        let int_out = apply(&spec, &int_batch).expect("Int64 KDE should succeed");
        let float_out = apply(&spec, &float_batch).unwrap();
        assert_eq!(int_out.num_rows(), 16);
        let id = col(&int_out, "density");
        let fd = col(&float_out, "density");
        let iv = col(&int_out, "value");
        let fv = col(&float_out, "value");
        for i in 0..16 {
            assert_eq!(iv[i].to_bits(), fv[i].to_bits(), "value[{i}] mismatch");
            assert_eq!(id[i].to_bits(), fd[i].to_bits(), "density[{i}] mismatch");
        }
    }

    #[test]
    fn test_kde_int32_column_is_coerced() {
        // Prove general numeric coercion (not just Int64).
        use arrow::array::Int32Array;
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, true)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![0_i32, 1, 2, 3, 4]))],
        )
        .unwrap();
        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            shared_extent: false,
            n: 8,
            extent: None,
            cumulative: false,
            kernel: default_kernel(),
            groupby: vec![],
            count_scale: None,
            count_scale_extent: None,
            name: None,
        };
        let out = apply(&spec, &batch).expect("Int32 KDE should succeed");
        assert_eq!(out.num_rows(), 8);
    }

    #[test]
    fn test_kde_string_column_still_errors() {
        // Negative test: a genuinely non-numeric column must STILL error.
        use arrow::array::StringArray;
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Utf8, true)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec!["a", "b", "c"]))],
        )
        .unwrap();
        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            shared_extent: false,
            n: 8,
            extent: None,
            cumulative: false,
            kernel: default_kernel(),
            groupby: vec![],
            count_scale: None,
            count_scale_extent: None,
            name: None,
        };
        let err = apply(&spec, &batch).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("must be numeric"), "err: {msg}");
    }

    #[test]
    fn test_kde_round_trip_json() {
        let original = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Fixed { value: 0.5 },
            bw_adjust: 1.0,
            n: 32,
            extent: Some((-1.0, 5.0)),
            cumulative: true,
            shared_extent: false,
            kernel: "gaussian".to_string(),
            groupby: vec![],
            count_scale: None,
            count_scale_extent: None,
            name: None,
        };
        let json = serde_json::to_string(&original).unwrap();
        let parsed: KdeSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn test_kde_single_row_emits_nan_densities() {
        // A single data point → n < 2, so Scott/Silverman bandwidth is undefined.
        // The transform should handle gracefully by emitting NaN densities.
        let batch = batch_with("x", vec![7.0]);
        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Silverman,
            bw_adjust: 1.0,
            shared_extent: false,
            n: 10,
            extent: Some((5.0, 9.0)),
            cumulative: false,
            kernel: default_kernel(),
            groupby: vec![],
            count_scale: None,
            count_scale_extent: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 10);
        let density = col(&out, "density");
        assert!(
            density.iter().all(|d| d.is_nan()),
            "single-row KDE should emit all-NaN densities"
        );
    }

    #[test]
    fn test_kde_all_identical_values_emits_nan_densities() {
        // All values identical → std=0 → bandwidth=0 (both Scott and Silverman).
        // This is effectively the zero-variance case.
        let batch = batch_with("x", vec![4.0; 20]);
        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Silverman,
            bw_adjust: 1.0,
            shared_extent: false,
            n: 16,
            extent: Some((2.0, 6.0)),
            cumulative: false,
            kernel: default_kernel(),
            groupby: vec![],
            count_scale: None,
            count_scale_extent: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let density = col(&out, "density");
        assert!(
            density.iter().all(|d| d.is_nan()),
            "zero-variance KDE should emit all-NaN densities"
        );
    }

    // ---- Kernel-specific tests ----

    fn kde_spec_with_kernel(kernel: &str, n: usize) -> KdeSpec {
        KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Fixed { value: 1.0 },
            bw_adjust: 1.0,
            shared_extent: false,
            n,
            extent: Some((-3.0, 3.0)),
            cumulative: false,
            kernel: kernel.to_string(),
            groupby: vec![],
            count_scale: None,
            count_scale_extent: None,
            name: None,
        }
    }

    fn simple_batch() -> RecordBatch {
        batch_with("x", vec![0.0, 0.5, -0.5, 1.0, -1.0])
    }

    /// Gaussian kernel (default) must produce non-zero density everywhere in the grid.
    #[test]
    fn test_kde_gaussian_produces_nonzero_density() {
        let batch = simple_batch();
        let spec = kde_spec_with_kernel("gaussian", 20);
        let out = apply(&spec, &batch).unwrap();
        let density = col(&out, "density");
        assert!(
            density.iter().all(|&d| d > 0.0),
            "Gaussian KDE: all grid densities should be positive"
        );
    }

    /// Epanechnikov kernel must produce non-zero density on simple data.
    #[test]
    fn test_kde_epanechnikov_produces_nonzero_density() {
        let batch = simple_batch();
        let spec = kde_spec_with_kernel("epanechnikov", 20);
        let out = apply(&spec, &batch).unwrap();
        let density = col(&out, "density");
        // At least some grid points must be non-zero (those inside bandwidth support).
        let nonzero_count = density.iter().filter(|&&d| d > 0.0).count();
        assert!(
            nonzero_count > 0,
            "Epanechnikov KDE: expected some non-zero densities, got all zeros"
        );
    }

    /// Epanechnikov kernel must produce exactly 0 for grid points further than
    /// bandwidth h from all data points.
    #[test]
    fn test_kde_epanechnikov_zero_outside_bandwidth() {
        // Single data point at x=0, bandwidth=0.5.
        // Grid points beyond |g - 0| > 0.5 must be exactly 0.
        let batch = batch_with("x", vec![0.0, 0.01]); // need ≥ 2 points
        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Fixed { value: 0.5 },
            bw_adjust: 1.0,
            shared_extent: false,
            n: 21,
            extent: Some((-2.0, 2.0)),
            cumulative: false,
            kernel: "epanechnikov".to_string(),
            groupby: vec![],
            count_scale: None,
            count_scale_extent: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let values = col(&out, "value");
        let density = col(&out, "density");
        // All grid points where |g| > 0.5 AND |g - 0.01| > 0.5 must have density = 0.
        for (i, (&v, &d)) in values.iter().zip(density.iter()).enumerate() {
            let in_support = v.abs() <= 0.5 || (v - 0.01).abs() <= 0.5;
            if !in_support {
                assert_eq!(
                    d, 0.0,
                    "Epanechnikov: expected density=0 at grid point {i} (x={v}), got {d}"
                );
            }
        }
    }

    /// "epan" alias must produce the same output as "epanechnikov".
    #[test]
    fn test_kde_epan_alias_matches_epanechnikov() {
        let batch = simple_batch();
        let spec1 = kde_spec_with_kernel("epanechnikov", 15);
        let spec2 = kde_spec_with_kernel("epan", 15);
        let out1 = apply(&spec1, &batch).unwrap();
        let out2 = apply(&spec2, &batch).unwrap();
        let d1 = col(&out1, "density");
        let d2 = col(&out2, "density");
        for (a, b) in d1.iter().zip(d2.iter()) {
            assert_eq!(
                a.to_bits(), b.to_bits(),
                "'epan' and 'epanechnikov' must produce identical output"
            );
        }
    }

    /// Tophat kernel must produce non-zero density on simple data.
    #[test]
    fn test_kde_tophat_produces_nonzero_density() {
        let batch = simple_batch();
        let spec = kde_spec_with_kernel("tophat", 20);
        let out = apply(&spec, &batch).unwrap();
        let density = col(&out, "density");
        let nonzero_count = density.iter().filter(|&&d| d > 0.0).count();
        assert!(
            nonzero_count > 0,
            "Tophat KDE: expected some non-zero densities"
        );
    }

    /// "uniform" alias must produce the same output as "tophat".
    #[test]
    fn test_kde_uniform_alias_matches_tophat() {
        let batch = simple_batch();
        let spec1 = kde_spec_with_kernel("tophat", 15);
        let spec2 = kde_spec_with_kernel("uniform", 15);
        let out1 = apply(&spec1, &batch).unwrap();
        let out2 = apply(&spec2, &batch).unwrap();
        let d1 = col(&out1, "density");
        let d2 = col(&out2, "density");
        for (a, b) in d1.iter().zip(d2.iter()) {
            assert_eq!(
                a.to_bits(), b.to_bits(),
                "'uniform' and 'tophat' must produce identical output"
            );
        }
    }

    /// Cosine kernel must produce non-zero density on simple data.
    #[test]
    fn test_kde_cosine_produces_nonzero_density() {
        let batch = simple_batch();
        let spec = kde_spec_with_kernel("cosine", 20);
        let out = apply(&spec, &batch).unwrap();
        let density = col(&out, "density");
        let nonzero_count = density.iter().filter(|&&d| d > 0.0).count();
        assert!(
            nonzero_count > 0,
            "Cosine KDE: expected some non-zero densities"
        );
    }

    // ---- T11: exponential + linear kernels ----

    /// Exponential kernel must produce non-zero density on simple data.
    #[test]
    fn test_kde_exponential_produces_nonzero_density() {
        let batch = simple_batch();
        let spec = kde_spec_with_kernel("exponential", 20);
        let out = apply(&spec, &batch).unwrap();
        let density = col(&out, "density");
        assert!(
            density.iter().all(|&d| d > 0.0),
            "Exponential KDE: unbounded support means every grid point should be non-zero"
        );
    }

    /// Exponential kernel discriminator: K(u) = exp(-|u|)/2 evaluated exactly
    /// against a single data point at 0, h=1. This pins the dedicated formula
    /// (not a silent Gaussian fallback, which would give ~0.3989 at u=0
    /// instead of exactly 0.5) and its "cusp" property — the value decays by
    /// the SAME multiplicative factor e^-1 per unit step on both sides of the
    /// peak, unlike Gaussian's super-exponential (accelerating) falloff.
    #[test]
    fn test_kde_exponential_matches_exact_formula_and_has_cusp() {
        let x = vec![0.0];
        let grid = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let h = 1.0;
        let got = exponential_density_at_grid(&x, h, &grid);
        let want: Vec<f64> = grid.iter().map(|&u| 0.5 * (-(u.abs())).exp()).collect();
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-12, "exponential[{i}]: got {g}, want {w}");
        }
        // Gaussian's value at u=0 is 1/sqrt(2*pi) ≈ 0.3989, not 0.5 — proves
        // this isn't the Gaussian formula under a different name.
        assert!((got[2] - 0.5).abs() < 1e-12, "exponential K(0) must be exactly 0.5");
        // Cusp: constant ratio e across the peak on both sides (linear decay
        // in log-space), which a smooth (Gaussian) peak does not exhibit.
        assert!((got[2] / got[1] - std::f64::consts::E).abs() < 1e-9);
        assert!((got[2] / got[3] - std::f64::consts::E).abs() < 1e-9);
    }

    /// Linear kernel must produce non-zero density on simple data.
    #[test]
    fn test_kde_linear_produces_nonzero_density() {
        let batch = simple_batch();
        let spec = kde_spec_with_kernel("linear", 20);
        let out = apply(&spec, &batch).unwrap();
        let density = col(&out, "density");
        let nonzero_count = density.iter().filter(|&&d| d > 0.0).count();
        assert!(nonzero_count > 0, "Linear KDE: expected some non-zero densities");
    }

    /// Linear kernel discriminator: triangular K(u) = 1 - |u| for |u| <= 1,
    /// with a HARD edge (exact 0) at and beyond |u| = 1 — unlike
    /// Epanechnikov's parabolic taper (which is also compactly supported but
    /// non-linear) and unlike exponential/Gaussian's unbounded tails.
    #[test]
    fn test_kde_linear_produces_hard_edged_triangular_shape() {
        let x = vec![0.0];
        let grid = vec![-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5];
        let h = 1.0;
        let got = linear_density_at_grid(&x, h, &grid);
        let want = [0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0];
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-12, "linear[{i}]: got {g}, want {w}");
        }
        // Discriminator vs Epanechnikov: at u=0.5, Epanechnikov gives
        // 0.75*(1-0.25) = 0.5625, not the triangular kernel's exact 0.5.
        let epan_at_half = epanechnikov_density_at_grid(&x, h, &[0.5]);
        assert!(
            (epan_at_half[0] - 0.5625).abs() < 1e-12,
            "sanity: epanechnikov(0.5) should be 0.5625, got {}",
            epan_at_half[0]
        );
        assert!(
            (got[4] - epan_at_half[0]).abs() > 1e-9,
            "linear and epanechnikov must diverge in shape away from the peak"
        );
    }

    /// Before this task, `kernel_density_at_grid`'s catch-all silently
    /// substituted Gaussian for any unrecognised kernel string. A `KdeSpec`
    /// built directly (bypassing `PyKde::new`'s validation — e.g. a spec
    /// deserialized from wire JSON with a typo'd kernel) must now error at
    /// `apply()` time instead of silently rendering the wrong shape.
    #[test]
    fn test_kde_apply_with_unvalidated_bogus_kernel_now_errors() {
        pyo3::Python::initialize();
        let batch = batch_with("x", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Fixed { value: 1.0 },
            bw_adjust: 1.0,
            shared_extent: false,
            n: 8,
            extent: Some((0.0, 6.0)),
            cumulative: false,
            kernel: "not_a_real_kernel".into(),
            groupby: vec![],
            count_scale: None,
            count_scale_extent: None,
            name: None,
        };
        let err = apply(&spec, &batch).unwrap_err();
        assert!(err.to_string().contains("unknown kernel"), "err: {err}");
    }

    /// Reverse guard (round 2, batch-D T11 cycle 2): every `KERNELS` entry
    /// must reach a real dispatch arm in `kernel_density_at_grid`, not just
    /// pass `PyKde::new`'s `KERNELS.contains` pre-check — that pre-check
    /// only proves the FORWARD direction (a dispatch arm without a `KERNELS`
    /// entry is unreachable at construction time), and it never calls
    /// `kernel_density_at_grid` itself, so a name added to `KERNELS` with no
    /// matching arm would otherwise pass both the pre-check and this test
    /// suite and only fail at render time in a user's chart. Mirrors
    /// `smooth::test_all_method_aliases_accepted_by_pysmooth_new`, adapted to
    /// call the actual compute site directly since (unlike `Smooth`) `Kde`'s
    /// constructor pre-check is not itself the dispatch.
    #[test]
    fn test_all_kernels_reach_a_dispatch_arm() {
        for &kernel in KERNELS {
            let result = kernel_density_at_grid(&[0.0, 1.0], 1.0, &[0.0], kernel);
            assert!(
                result.is_ok(),
                "KERNELS entry {kernel:?} has no matching dispatch arm in \
                 kernel_density_at_grid: {:?}",
                result.err().map(|e| e.to_string())
            );
        }
    }

    /// Determinism pin (round 2, batch-D T11 cycle 2): the two new kernels
    /// must produce byte-identical output across repeated `apply()` calls on
    /// identical input, per the spec's testing section ("determinism: two
    /// runs byte-equal"). Rust owns this pin — see the round-2 report for
    /// why it isn't duplicated Python-side.
    #[test]
    fn test_kde_exponential_and_linear_apply_deterministic_across_runs() {
        for kernel in ["exponential", "linear"] {
            let batch = simple_batch();
            let spec = kde_spec_with_kernel(kernel, 50);
            let out1 = apply(&spec, &batch).unwrap();
            let out2 = apply(&spec, &batch).unwrap();
            let d1 = col(&out1, "density");
            let d2 = col(&out2, "density");
            assert_eq!(d1.len(), d2.len());
            for (i, (a, b)) in d1.iter().zip(d2.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(), b.to_bits(),
                    "kernel={kernel} density[{i}] not byte-identical across runs: {a} vs {b}"
                );
            }
        }
    }

    /// Gaussian must produce the same output as before (regression against the
    /// hardcoded path we replaced with the dispatch).
    #[test]
    fn test_kde_gaussian_regression_matches_direct_call() {
        let data = vec![-1.0, 0.0, 1.0, 2.0, 3.0];
        let h = 0.8_f64;
        let grid: Vec<f64> = (0..10).map(|i| -2.0 + i as f64 * 0.5).collect();
        let expected = gaussian_density_at_grid(&data, h, &grid);
        let via_dispatch = kernel_density_at_grid(&data, h, &grid, "gaussian").unwrap();
        for (a, b) in expected.iter().zip(via_dispatch.iter()) {
            assert_eq!(
                a.to_bits(), b.to_bits(),
                "gaussian dispatch must be byte-identical to direct call"
            );
        }
    }

    /// Unknown kernel name in PyKde::new must return an error.
    #[test]
    fn test_kde_unknown_kernel_returns_error() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            use pyo3::types::PyString;
            let bw_obj = PyString::new(py, "scott").into_any();
            let result = PyKde::new(
                "x",
                Some(&bw_obj),
                1.0,
                512,
                None,
                false,
                false,
                "laplacian",
                None,
                None,
                None,
            );
            assert!(result.is_err(), "unknown kernel 'laplacian' must return an error");
            let msg = result.unwrap_err().to_string();
            assert!(
                msg.contains("unknown kernel"),
                "error message should mention 'unknown kernel': {msg}"
            );
        });
    }

    /// Gaussian kernel must round-trip through JSON including the kernel field.
    #[test]
    fn test_kde_kernel_field_round_trips_json() {
        let original = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 64,
            extent: None,
            cumulative: false,
            shared_extent: false,
            kernel: "epanechnikov".to_string(),
            groupby: vec![],
            count_scale: None,
            count_scale_extent: None,
            name: None,
        };
        let json = serde_json::to_string(&original).unwrap();
        let parsed: KdeSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.kernel, "epanechnikov");
        assert_eq!(parsed, original);
    }

    /// Old JSON without a "kernel" field must deserialize to gaussian (backward compat).
    #[test]
    fn test_kde_kernel_field_defaults_to_gaussian_on_old_json() {
        // Simulate a serialized KdeSpec without the "kernel" field.
        let json = r#"{"field":"x","bandwidth":{"kind":"scott"},"bw_adjust":1.0,"n":32}"#;
        let parsed: KdeSpec = serde_json::from_str(json).unwrap();
        assert_eq!(
            parsed.kernel, "gaussian",
            "missing 'kernel' field must default to 'gaussian' for backward compat"
        );
    }

    // ── Task 8: global_extent helper ─────────────────────────────────────────

    /// `global_extent` returns the full-data range over the KDE field.
    #[test]
    fn test_kde_global_extent_returns_full_range() {
        let batch = batch_with("x", vec![3.0, 1.0, 5.0, 2.0, 4.0]);
        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 32,
            extent: None,
            cumulative: false,
            shared_extent: false,
            kernel: default_kernel(),
            groupby: vec![],
            count_scale: None,
            count_scale_extent: None,
            name: None,
        };
        let ext = super::global_extent(&spec, &batch);
        assert_eq!(ext, Some((1.0, 5.0)), "global_extent should be (min, max) = (1.0, 5.0)");
    }

    /// `global_extent` handles integer columns via coerce_to_float64.
    #[test]
    fn test_kde_global_extent_integer_column() {
        use arrow::array::Int64Array;
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![10_i64, 20, 30]))],
        ).unwrap();
        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 16,
            extent: None,
            cumulative: false,
            shared_extent: false,
            kernel: default_kernel(),
            groupby: vec![],
            count_scale: None,
            count_scale_extent: None,
            name: None,
        };
        let ext = super::global_extent(&spec, &batch);
        assert_eq!(ext, Some((10.0, 30.0)), "global_extent on Int64 column should return (10.0, 30.0)");
    }

    /// `global_extent` returns None on missing field.
    #[test]
    fn test_kde_global_extent_missing_field_returns_none() {
        let batch = batch_with("x", vec![1.0, 2.0, 3.0]);
        let spec = KdeSpec {
            field: "nonexistent".into(),
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 16,
            extent: None,
            cumulative: false,
            shared_extent: false,
            kernel: default_kernel(),
            groupby: vec![],
            count_scale: None,
            count_scale_extent: None,
            name: None,
        };
        let ext = super::global_extent(&spec, &batch);
        assert_eq!(ext, None, "global_extent on missing field must return None");
    }

    /// `global_extent` returns None when all values are null or NaN.
    #[test]
    fn test_kde_global_extent_all_null_returns_none() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, true)]));
        let arr = Float64Array::from(vec![None::<f64>, None, None]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 16,
            extent: None,
            cumulative: false,
            shared_extent: false,
            kernel: default_kernel(),
            groupby: vec![],
            count_scale: None,
            count_scale_extent: None,
            name: None,
        };
        let ext = super::global_extent(&spec, &batch);
        assert_eq!(ext, None, "global_extent on all-null field must return None");
    }

    // ── T3.8a regression tests ────────────────────────────────────────────────

    /// D-GROUPBY-1 regression (T3.8a): a 2-column groupby on `Kde` must produce
    /// `n` density grid points per distinct `(g1, g2)` combination, with both
    /// group columns appended after the standard 2 kde output columns.
    #[test]
    fn kde_two_column_groupby_produces_per_combination_rows() {
        use arrow::array::StringArray;
        pyo3::Python::initialize();
        // 8 rows: two g1 × two g2 × 2 x-values each.
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("g1", DataType::Utf8, false),
            Field::new("g2", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![
                    1.0, 9.0, 1.0, 9.0, 1.0, 9.0, 1.0, 9.0,
                ])) as arrow::array::ArrayRef,
                Arc::new(StringArray::from(vec!["A", "A", "A", "A", "B", "B", "B", "B"]))
                    as arrow::array::ArrayRef,
                Arc::new(StringArray::from(vec!["x", "x", "y", "y", "x", "x", "y", "y"]))
                    as arrow::array::ArrayRef,
            ],
        )
        .unwrap();

        let n = 8_usize;
        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n,
            extent: None,
            cumulative: false,
            shared_extent: false,
            kernel: default_kernel(),
            groupby: vec!["g1".into(), "g2".into()],
            count_scale: None,
            count_scale_extent: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();

        // Schema: value, density, g1, g2.
        let s = out.schema();
        assert!(s.index_of("value").is_ok(), "missing value");
        assert!(s.index_of("density").is_ok(), "missing density");
        assert!(s.index_of("g1").is_ok(), "missing g1");
        assert!(s.index_of("g2").is_ok(), "missing g2");

        // 4 distinct (g1,g2) groups × n grid points each.
        assert_eq!(
            out.num_rows(),
            4 * n,
            "expected 4 groups × {n} grid points = {} rows, got {}",
            4 * n,
            out.num_rows()
        );

        // Collect distinct (g1,g2) combos in the output.
        let g1_col = out.column(s.index_of("g1").unwrap())
            .as_any().downcast_ref::<StringArray>().unwrap();
        let g2_col = out.column(s.index_of("g2").unwrap())
            .as_any().downcast_ref::<StringArray>().unwrap();
        let combos: std::collections::BTreeSet<(String, String)> = (0..out.num_rows())
            .map(|i| (g1_col.value(i).to_string(), g2_col.value(i).to_string()))
            .collect();
        assert_eq!(combos.len(), 4, "must have 4 distinct (g1,g2) pairs");
        assert!(combos.contains(&("A".into(), "x".into())));
        assert!(combos.contains(&("A".into(), "y".into())));
        assert!(combos.contains(&("B".into(), "x".into())));
        assert!(combos.contains(&("B".into(), "y".into())));
    }

    /// D-GROUPBY-1 regression (T3.8a): the KdeSpec serde round-trip preserves
    /// the canonical array wire form; deserializing the old bare-string form
    /// yields a one-element vec identical to `vec!["col"]`.
    #[test]
    fn kde_groupby_serde_round_trip_and_back_compat() {
        // Minimal valid KdeSpec JSON (bandwidth and n are required fields).
        // "bandwidth":{"kind":"scott"} is the Scott variant serialized by serde.
        let base_json =
            r#"{"field":"x","bandwidth":{"kind":"scott"},"n":64,"groupby":"col"}"#;

        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 64,
            extent: None,
            cumulative: false,
            shared_extent: false,
            kernel: default_kernel(),
            groupby: vec!["col".into()],
            count_scale: None,
            count_scale_extent: None,
            name: None,
        };

        // Serialize: groupby=["col"] must use array form.
        let json = serde_json::to_string(&spec).unwrap();
        assert!(json.contains(r#""groupby":["col"]"#), "must serialize as array, got: {json}");

        // Legacy bare-string deserializes to vec!["col"].
        let from_legacy: KdeSpec = serde_json::from_str(base_json).unwrap();
        assert_eq!(from_legacy.groupby, vec!["col"], "bare-string must parse to [\"col\"]");

        // Legacy null deserializes to [].
        let null_json =
            r#"{"field":"x","bandwidth":{"kind":"scott"},"n":64,"groupby":null}"#;
        let from_null: KdeSpec = serde_json::from_str(null_json).unwrap();
        assert!(from_null.groupby.is_empty(), "null groupby must parse to []");

        // Empty groupby is omitted from serialization.
        let empty = KdeSpec { groupby: vec![], ..spec.clone() };
        let empty_json = serde_json::to_string(&empty).unwrap();
        assert!(!empty_json.contains("groupby"), "empty groupby must be omitted: {empty_json}");
    }

    // ── Batch F (F-L09-10): count-scale ─────────────────────────────────────

    fn count_scale_spec(field: &str, n: usize, extent: (f64, f64), count_scale: Option<BinMode>) -> KdeSpec {
        KdeSpec {
            field: field.into(),
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n,
            extent: Some(extent),
            cumulative: false,
            shared_extent: false,
            kernel: default_kernel(),
            groupby: vec![],
            count_scale,
            count_scale_extent: None,
            name: None,
        }
    }

    /// Wire-default round-trip: `count_scale` is omitted from serialized JSON
    /// when absent (matching every other `Option` field on `KdeSpec`), old
    /// JSON with no `count_scale` key deserializes to `None` (today's wire
    /// specs stay byte-identical), and a `Some` value round-trips through its
    /// tagged `BinMode` shape unchanged.
    #[test]
    fn test_kde_count_scale_wire_default_round_trip() {
        let unscaled = count_scale_spec("x", 8, (0.0, 10.0), None);
        let json = serde_json::to_string(&unscaled).unwrap();
        assert!(!json.contains("count_scale"), "absent count_scale must be omitted: {json}");
        let parsed: KdeSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, unscaled);

        // Old JSON (pre-batch-F) with no count_scale key must deserialize to None.
        let old_json = r#"{"field":"x","bandwidth":{"kind":"scott"},"n":8,"extent":[0.0,10.0]}"#;
        let from_old: KdeSpec = serde_json::from_str(old_json).unwrap();
        assert_eq!(from_old.count_scale, None, "missing count_scale key must default to None");

        // A `Some` value round-trips through its tagged BinMode wire shape.
        let scaled = count_scale_spec("x", 8, (0.0, 10.0), Some(BinMode::Fixed { n: 5 }));
        let scaled_json = serde_json::to_string(&scaled).unwrap();
        assert!(
            scaled_json.contains(r#""count_scale":{"kind":"fixed","n":5}"#),
            "expected tagged BinMode shape, got: {scaled_json}"
        );
        let scaled_parsed: KdeSpec = serde_json::from_str(&scaled_json).unwrap();
        assert_eq!(scaled_parsed, scaled);
    }

    /// Absent `count_scale` (the default) must leave `density` EXACTLY the
    /// raw kernel-density output — a real byte-identity pin against an
    /// independently-derived oracle, not a determinism check. Cross-checks
    /// `apply()`'s output against a direct call to `gaussian_density_at_grid`
    /// (the same primitive `apply_one_group` dispatches to for a fixed
    /// bandwidth) computed OUTSIDE `apply()`, so a regression that
    /// spuriously applies ANY scaling factor when `count_scale=None` would
    /// be caught — unlike the prior `apply(&spec)` vs `apply(&spec)`
    /// self-comparison, which could never fail regardless of what `apply()`
    /// computed.
    #[test]
    fn test_kde_count_scale_absent_matches_raw_kernel_density_exactly() {
        let x = vec![0.0, 1.0, 2.0, 4.0];
        let h = 1.0;
        let batch = batch_with("x", x.clone());
        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Fixed { value: h },
            bw_adjust: 1.0,
            n: 6,
            extent: Some((0.0, 4.0)),
            cumulative: false,
            shared_extent: false,
            kernel: default_kernel(),
            groupby: vec![],
            count_scale: None,
            count_scale_extent: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let got_value = col(&out, "value");
        let got_density = col(&out, "density");

        // Independently-computed oracle: the raw kernel density on the
        // ACTUAL returned grid, called directly (bypassing `apply()`'s own
        // count-scale dispatch entirely).
        let expected_density = gaussian_density_at_grid(&x, h, &got_value);
        for (i, (g, e)) in got_density.iter().zip(expected_density.iter()).enumerate() {
            assert_eq!(
                g.to_bits(), e.to_bits(),
                "point {i}: count_scale=None must be the RAW kernel density, got {g} vs expected {e}"
            );
        }
    }

    /// Ungrouped count-scale: `density × n × bin_width` must have bit-parity
    /// with `Bin`'s OWN bin width for the same field + bin mode — not a
    /// hand-derived approximation. Runs the real `Bin` transform on the same
    /// batch/field/mode and cross-checks the scaling factor against its
    /// actual `bin_end - bin_start`.
    #[test]
    fn test_kde_count_scale_ungrouped_scales_by_n_times_bin_width() {
        let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let batch = batch_with("x", values.clone());

        // The real Bin transform, same field + mode, to get the ACTUAL bin width.
        let bin_spec = crate::transform::bin::BinSpec {
            field: "x".into(),
            mode: BinMode::Fixed { n: 5 },
            extent: None,
            nice: true,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
            agg_field: None,
            agg: None,
            name: None,
        };
        let bin_out = crate::transform::bin::apply(&bin_spec, &batch).unwrap();
        let bin_starts = bin_out
            .column(bin_out.schema().index_of("bin_start").unwrap())
            .as_any().downcast_ref::<Float64Array>().unwrap();
        let bin_ends = bin_out
            .column(bin_out.schema().index_of("bin_end").unwrap())
            .as_any().downcast_ref::<Float64Array>().unwrap();
        let expected_bin_width = bin_ends.value(0) - bin_starts.value(0);

        let unscaled = count_scale_spec("x", 12, (0.0, 19.0), None);
        let scaled = count_scale_spec("x", 12, (0.0, 19.0), Some(BinMode::Fixed { n: 5 }));
        let out_unscaled = apply(&unscaled, &batch).unwrap();
        let out_scaled = apply(&scaled, &batch).unwrap();
        let d_unscaled = col(&out_unscaled, "density");
        let d_scaled = col(&out_scaled, "density");

        let n = values.len() as f64;
        for (i, (u, s)) in d_unscaled.iter().zip(d_scaled.iter()).enumerate() {
            let expected = u * n * expected_bin_width;
            assert!(
                (s - expected).abs() < 1e-9,
                "point {i}: scaled {s} != unscaled {u} * n {n} * bin_width {expected_bin_width} = {expected}"
            );
        }
    }

    /// Runs the REAL grouped `Bin` transform on `batch`/`field`/`group_cols`
    /// under `shared_extent` and returns the bin width (`bin_end -
    /// bin_start`) actually rendered for each group. The oracle every
    /// grouped count-scale test below cross-checks against — never
    /// `resolve_bin_extent` called with the implementation's own arguments,
    /// which would be tautological on the exact global-vs-per-group
    /// dimension these tests exist to pin (quality-review cycle-3
    /// finding 1/3). Parameterized on `shared_extent` (quality-review
    /// cycle-4 finding A) so both `Bin::apply_grouped` branches — an
    /// un-shared per-group extent (`shared_extent=false`, the `displot`
    /// default) and a shared extent with still-per-group `n_bins`
    /// (`shared_extent=true`) — get a real-`Bin` oracle, not just a
    /// hand-traced claim.
    fn actual_bin_widths_per_group(
        batch: &RecordBatch,
        field: &str,
        group_cols: &[String],
        mode: BinMode,
        shared_extent: bool,
    ) -> std::collections::HashMap<String, f64> {
        let bin_spec = crate::transform::bin::BinSpec {
            field: field.into(),
            mode,
            extent: None,
            nice: true,
            cumulative: false,
            shared_extent,
            groupby: group_cols.to_vec(),
            agg_field: None,
            agg: None,
            name: None,
        };
        let bin_out = crate::transform::bin::apply(&bin_spec, batch).unwrap();
        widths_by_group_from_bin_output(&bin_out, &group_cols[0])
    }

    /// Read `bin_end - bin_start` per group off an ALREADY-COMPUTED `Bin`
    /// output batch, keyed by `group_col`'s value. Shared by
    /// `actual_bin_widths_per_group` (which builds its own `BinSpec`) and
    /// the faceted test below (which needs the width `Bin` renders for a
    /// spec it does NOT control the construction of — the facet-pinned
    /// spec `fix_transform_extents_for_facet` produces).
    fn widths_by_group_from_bin_output(
        bin_out: &RecordBatch,
        group_col: &str,
    ) -> std::collections::HashMap<String, f64> {
        use arrow::array::StringArray;
        let s = bin_out.schema();
        let starts = bin_out.column(s.index_of("bin_start").unwrap())
            .as_any().downcast_ref::<Float64Array>().unwrap();
        let ends = bin_out.column(s.index_of("bin_end").unwrap())
            .as_any().downcast_ref::<Float64Array>().unwrap();
        let groups = bin_out.column(s.index_of(group_col).unwrap())
            .as_any().downcast_ref::<StringArray>().unwrap();
        let mut widths = std::collections::HashMap::new();
        for i in 0..bin_out.num_rows() {
            widths.entry(groups.value(i).to_string())
                .or_insert_with(|| ends.value(i) - starts.value(i));
        }
        widths
    }

    /// Grouped count-scale, `Fixed` mode: `bin_width` must be derived PER
    /// GROUP, mirroring `Bin::apply_grouped`'s `shared_extent=false` default
    /// (the `displot` default, `multiple="layer"`) — NOT one scalar shared
    /// across every hue level. Reproduces the quality reviewer's own
    /// hand-computed numbers: group A `[0,2,4,6,8]` and group B
    /// `[10,14,18]` under `Fixed{n:4}` each render width **2** from `Bin`
    /// (`nice_step(0,8,4)=2`, `nice_step(10,18,4)=2`), NOT the global width
    /// **5** a single-scalar derivation over the full `[0..18]` range would
    /// give (`nice_step(0,18,4)=5`) — confirmed live: this test fails
    /// (asserts 5.0 where `Bin` renders 2.0) against the pre-fix
    /// implementation and passes against the per-group fix.
    #[test]
    fn test_kde_count_scale_grouped_fixed_mirrors_bin_per_group_width() {
        use arrow::array::StringArray;
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("g", DataType::Utf8, false),
        ]));
        let xs = Float64Array::from(vec![
            0.0, 2.0, 4.0, 6.0, 8.0, // group A (5 points)
            10.0, 14.0, 18.0,        // group B (3 points)
        ]);
        let gs = StringArray::from(vec!["A", "A", "A", "A", "A", "B", "B", "B"]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(xs), Arc::new(gs)]).unwrap();
        let group_cols = vec!["g".to_string()];
        let mode = BinMode::Fixed { n: 4 };

        let widths = actual_bin_widths_per_group(&batch, "x", &group_cols, mode.clone(), false);
        assert!((widths["A"] - 2.0).abs() < 1e-9, "group A: Bin renders width 2.0, got {}", widths["A"]);
        assert!((widths["B"] - 2.0).abs() < 1e-9, "group B: Bin renders width 2.0, got {}", widths["B"]);

        let unscaled = KdeSpec { groupby: group_cols.clone(), ..count_scale_spec("x", 10, (0.0, 18.0), None) };
        let scaled = KdeSpec { groupby: group_cols, ..count_scale_spec("x", 10, (0.0, 18.0), Some(mode)) };
        let out_unscaled = apply(&unscaled, &batch).unwrap();
        let out_scaled = apply(&scaled, &batch).unwrap();
        let d_unscaled = col(&out_unscaled, "density");
        let d_scaled = col(&out_scaled, "density");
        let groups = out_unscaled.column(2).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(out_scaled.num_rows(), out_unscaled.num_rows());

        for i in 0..out_unscaled.num_rows() {
            let g = groups.value(i).to_string();
            let n_group = if g == "A" { 5.0 } else { 3.0 };
            let expected = d_unscaled[i] * n_group * widths[&g];
            assert!(
                (d_scaled[i] - expected).abs() < 1e-9,
                "row {i} (group {g}): scaled {} != unscaled {} * n_group {n_group} * bin_width {} = {expected}",
                d_scaled[i], d_unscaled[i], widths[&g],
            );
        }
    }

    /// Grouped count-scale, `Sturges` mode (the `displot` default rule):
    /// `resolve_bin_count`'s Sturges arm depends on `vals.len()`, so two
    /// groups of DIFFERENT sizes must render DIFFERENT bin counts (hence
    /// widths) even under the same field range — exactly what `Bin` itself
    /// does and what a single shared-scalar derivation could never
    /// reproduce. Discriminates the exact bug finding 1 describes for the
    /// most common real-world `count_scale` mode (the pre-batch-F default
    /// `bins="sturges"` displot forwards).
    #[test]
    fn test_kde_count_scale_grouped_sturges_mirrors_bin_per_group_n_bins() {
        use arrow::array::StringArray;
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("g", DataType::Utf8, false),
        ]));
        let a: Vec<f64> = (0..12).map(|i| i as f64).collect(); // 12 points
        let b: Vec<f64> = vec![0.0, 4.0, 8.0, 12.0];            // 4 points
        let mut xs_vals = a.clone();
        xs_vals.extend(b.iter().copied());
        let mut gs_vals = vec!["A"; a.len()];
        gs_vals.extend(vec!["B"; b.len()]);
        let xs = Float64Array::from(xs_vals);
        let gs = StringArray::from(gs_vals);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(xs), Arc::new(gs)]).unwrap();
        let group_cols = vec!["g".to_string()];

        let widths = actual_bin_widths_per_group(&batch, "x", &group_cols, BinMode::Sturges, false);
        assert!(
            (widths["A"] - widths["B"]).abs() > 1e-9,
            "expected different Sturges widths per group (A={}, B={}) — otherwise this \
             test can't discriminate per-group from a shared-scalar derivation",
            widths["A"], widths["B"],
        );

        let unscaled = KdeSpec { groupby: group_cols.clone(), ..count_scale_spec("x", 8, (0.0, 12.0), None) };
        let scaled = KdeSpec { groupby: group_cols, ..count_scale_spec("x", 8, (0.0, 12.0), Some(BinMode::Sturges)) };
        let out_unscaled = apply(&unscaled, &batch).unwrap();
        let out_scaled = apply(&scaled, &batch).unwrap();
        let d_unscaled = col(&out_unscaled, "density");
        let d_scaled = col(&out_scaled, "density");
        let groups = out_unscaled.column(2).as_any().downcast_ref::<StringArray>().unwrap();

        for i in 0..out_unscaled.num_rows() {
            let g = groups.value(i).to_string();
            let n_group = if g == "A" { a.len() as f64 } else { b.len() as f64 };
            let expected = d_unscaled[i] * n_group * widths[&g];
            assert!(
                (d_scaled[i] - expected).abs() < 1e-9,
                "row {i} (group {g}): scaled {} != unscaled {} * n_group {n_group} * bin_width {} = {expected}",
                d_scaled[i], d_unscaled[i], widths[&g],
            );
        }
    }

    /// Grouped count-scale, `shared_extent=true` + `Sturges` mode
    /// (quality-review cycle-4 finding A) — the one branch the two tests
    /// above never exercise: `resolve_group_bin_width`'s `Some((lo, hi))`
    /// arm, reached when `multiple in ("stack","fill")` sets `shared_extent`
    /// on both `mark_histogram` and `mark_density`. Group B's own 3 points
    /// span only `[5,7]` — if the extent were (wrongly) still derived
    /// per-group, group B would bin over `[5,7]`, not the shared `[0,9]`;
    /// this test would then still pass by accident unless it also proves
    /// the shared bound is actually reaching `Bin` and `Kde` identically,
    /// which the real-`Bin` oracle does by construction (both read the same
    /// `shared_extent=true` semantics). Sturges still varies `n_bins` (and
    /// hence width) per group SIZE even though the extent is shared — group
    /// A (10 points) and group B (3 points) get different bin counts over
    /// the same `[0,9]` range, pinning both halves of the branch at once.
    #[test]
    fn test_kde_count_scale_grouped_shared_extent_sturges_mirrors_bin_shared_extent_per_group_n_bins() {
        use arrow::array::StringArray;
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("g", DataType::Utf8, false),
        ]));
        let a: Vec<f64> = (0..10).map(|i| i as f64).collect(); // 10 points, spans [0,9]
        let b: Vec<f64> = vec![5.0, 6.0, 7.0];                  // 3 points, spans [5,7] alone
        let mut xs_vals = a.clone();
        xs_vals.extend(b.iter().copied());
        let mut gs_vals = vec!["A"; a.len()];
        gs_vals.extend(vec!["B"; b.len()]);
        let xs = Float64Array::from(xs_vals);
        let gs = StringArray::from(gs_vals);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(xs), Arc::new(gs)]).unwrap();
        let group_cols = vec!["g".to_string()];

        let widths = actual_bin_widths_per_group(&batch, "x", &group_cols, BinMode::Sturges, true);
        assert!(
            (widths["A"] - widths["B"]).abs() > 1e-9,
            "expected different Sturges n_bins per group even under a shared extent \
             (A={}, B={}) — otherwise this test can't discriminate the shared-extent, \
             still-per-group-n_bins shape from either extreme",
            widths["A"], widths["B"],
        );

        let unscaled = KdeSpec {
            groupby: group_cols.clone(),
            shared_extent: true,
            ..count_scale_spec("x", 8, (0.0, 9.0), None)
        };
        let scaled = KdeSpec {
            groupby: group_cols,
            shared_extent: true,
            ..count_scale_spec("x", 8, (0.0, 9.0), Some(BinMode::Sturges))
        };
        let out_unscaled = apply(&unscaled, &batch).unwrap();
        let out_scaled = apply(&scaled, &batch).unwrap();
        let d_unscaled = col(&out_unscaled, "density");
        let d_scaled = col(&out_scaled, "density");
        let groups = out_unscaled.column(2).as_any().downcast_ref::<StringArray>().unwrap();

        for i in 0..out_unscaled.num_rows() {
            let g = groups.value(i).to_string();
            let n_group = if g == "A" { a.len() as f64 } else { b.len() as f64 };
            let expected = d_unscaled[i] * n_group * widths[&g];
            assert!(
                (d_scaled[i] - expected).abs() < 1e-9,
                "row {i} (group {g}): scaled {} != unscaled {} * n_group {n_group} * bin_width {} = {expected}",
                d_scaled[i], d_unscaled[i], widths[&g],
            );
        }
    }

    /// Faceted grouped count-scale (quality-review cycle-4 finding B): a
    /// faceted `displot(col=, kind="hist", kde=True, stat="count")` runs
    /// each panel's `Bin` with its extent PRE-PINNED to the niced GLOBAL
    /// span (`fix_transform_extents_for_facet`, `render/prepare/extent.rs`),
    /// so a narrow panel's histogram still bins over the WHOLE-dataset
    /// range, not its own local span. `count_scale_extent` must be pinned
    /// the SAME way (`kde::count_scale_global_extent`) so a faceted KDE
    /// overlay derives the SAME per-panel bin width the real `Bin` renders
    /// in that panel — not the panel's own local (narrower) span, which
    /// would under-scale the overlay (the reviewer's ~25%,
    /// unbounded-for-narrow-panels divergence). Simulates faceting
    /// directly: pins BOTH a `Bin` and a `Kde` spec against the FULL
    /// pre-facet batch (mirroring `fix_transform_extents_for_facet`'s
    /// actual call in `render::prepare`), then applies BOTH to a NARROW
    /// panel sub-batch — reproducing the divergence's exact shape.
    #[test]
    fn test_kde_count_scale_faceted_grouped_matches_bin_actual_panel_width() {
        use crate::render::prepare::fix_transform_extents_for_facet;
        use crate::transform::core::TransformSpec;
        use arrow::array::StringArray;

        // Full pre-facet dataset: spans [0, 20], two hue groups.
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("g", DataType::Utf8, false),
        ]));
        let full_xs: Vec<f64> = (0..=20).map(|i| i as f64).collect();
        let full_gs: Vec<&str> = (0..=20).map(|i| if i % 2 == 0 { "A" } else { "B" }).collect();
        let full_batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Float64Array::from(full_xs)) as ArrayRef,
                Arc::new(StringArray::from(full_gs)) as ArrayRef,
            ],
        ).unwrap();

        let mode = BinMode::Fixed { n: 5 };
        let group_cols = vec!["g".to_string()];
        let bin_spec = crate::transform::bin::BinSpec {
            field: "x".into(),
            mode: mode.clone(),
            extent: None,
            nice: true,
            cumulative: false,
            shared_extent: false,
            groupby: group_cols.clone(),
            agg_field: None,
            agg: None,
            name: None,
        };
        let kde_spec = KdeSpec {
            groupby: group_cols.clone(),
            ..count_scale_spec("x", 8, (2.0, 9.0), Some(mode.clone()))
        };

        // Mirror `render::prepare`'s actual facet-pin call: pin BOTH specs
        // against the FULL pre-facet batch.
        let pinned = fix_transform_extents_for_facet(
            &[TransformSpec::Bin(bin_spec), TransformSpec::Kde(kde_spec)],
            &full_batch,
        );
        let TransformSpec::Bin(pinned_bin) = &pinned[0] else { panic!("expected a pinned Bin") };
        let TransformSpec::Kde(pinned_kde) = &pinned[1] else { panic!("expected a pinned Kde") };

        // A NARROW panel: only rows with x in [2, 9] (8 points) — a fraction
        // of the full [0, 20] dataset span, the shape that diverges under a
        // per-panel-local derivation.
        let panel_xs: Vec<f64> = (2..=9).map(|i| i as f64).collect();
        let panel_gs: Vec<&str> = (2..=9).map(|i| if i % 2 == 0 { "A" } else { "B" }).collect();
        let panel_batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(panel_xs)) as ArrayRef,
                Arc::new(StringArray::from(panel_gs.clone())) as ArrayRef,
            ],
        ).unwrap();

        // The REAL Bin transform's ACTUAL per-group width IN THIS PANEL
        // (using the facet-pinned spec, exactly as `render::prepare` would
        // apply it per panel).
        let bin_out = crate::transform::bin::apply(pinned_bin, &panel_batch).unwrap();
        let panel_widths = widths_by_group_from_bin_output(&bin_out, "g");

        // Sanity: the panel's OWN local span [2,9] must give a DIFFERENT
        // (narrower) width than the global-pinned one Bin actually
        // rendered — otherwise this test can't discriminate the facet-pin
        // fix from a false pass.
        let local_widths = actual_bin_widths_per_group(&panel_batch, "x", &group_cols, mode, false);
        assert!(
            (panel_widths["A"] - local_widths["A"]).abs() > 1e-9,
            "test setup must produce a real global-vs-local divergence to discriminate the fix \
             (global-pinned width {} == local-only width {})",
            panel_widths["A"], local_widths["A"],
        );

        // `Kde`'s own facet-pinned `count_scale_extent` must equal `Bin`'s
        // own facet-pinned `extent` — both derived via the SAME
        // niced-global routine over the SAME field.
        assert_eq!(
            pinned_kde.count_scale_extent, pinned_bin.extent,
            "Kde's facet-pinned count_scale_extent must equal Bin's facet-pinned extent \
             (both niced-global over the same field)"
        );

        // Apply Kde (scaled vs unscaled) to the SAME narrow panel and
        // confirm the applied factor equals Bin's ACTUAL per-panel-group
        // width — not the panel's own local span.
        let unscaled_kde = KdeSpec {
            count_scale: None,
            count_scale_extent: None,
            ..pinned_kde.clone()
        };
        let out_unscaled = apply(&unscaled_kde, &panel_batch).unwrap();
        let out_scaled = apply(pinned_kde, &panel_batch).unwrap();
        let d_unscaled = col(&out_unscaled, "density");
        let d_scaled = col(&out_scaled, "density");
        let groups = out_unscaled.column(2).as_any().downcast_ref::<StringArray>().unwrap();

        for i in 0..out_unscaled.num_rows() {
            let g = groups.value(i).to_string();
            let n_group = panel_gs.iter().filter(|&&v| v == g).count() as f64;
            let expected = d_unscaled[i] * n_group * panel_widths[&g];
            assert!(
                (d_scaled[i] - expected).abs() < 1e-9,
                "row {i} (group {g}): faceted scaled {} != unscaled {} * n_group {n_group} * \
                 Bin's ACTUAL panel width {} = {expected}",
                d_scaled[i], d_unscaled[i], panel_widths[&g],
            );
        }
    }

    /// Direct unit tests for `resolve_group_bin_width` (quality-review
    /// cycle-3 findings 4/5): the empty-clean case returns `Ok(None)` — "no
    /// scaling factor" — rather than a NaN sentinel, and the degenerate
    /// all-equal case collapses to `Bin`'s unit-width bin (`1.0`) exactly.
    #[test]
    fn test_resolve_group_bin_width_empty_returns_none() {
        let width = resolve_group_bin_width(&[], &BinMode::Sturges, None, "stat_kde").unwrap();
        assert_eq!(width, None, "empty clean column must return None, not a NaN sentinel");
    }

    #[test]
    fn test_resolve_group_bin_width_degenerate_all_equal_matches_bin_unit_width() {
        let clean = vec![5.0, 5.0, 5.0, 5.0];
        let width = resolve_group_bin_width(&clean, &BinMode::Fixed { n: 3 }, None, "stat_kde").unwrap();
        assert_eq!(width, Some(1.0), "degenerate all-equal field must collapse to Bin's unit-width bin");
    }

    /// Cumulative count-scale: the CDF scales by `n_group` alone — NOT
    /// `n_group × bin_width` — per the normative spec semantics.
    #[test]
    fn test_kde_count_scale_cumulative_scales_by_n_group_only() {
        let values: Vec<f64> = (0..15).map(|i| i as f64).collect();
        let batch = batch_with("x", values.clone());

        let mut unscaled = count_scale_spec("x", 10, (0.0, 14.0), None);
        unscaled.cumulative = true;
        let mut scaled = count_scale_spec("x", 10, (0.0, 14.0), Some(BinMode::Fixed { n: 5 }));
        scaled.cumulative = true;

        let out_unscaled = apply(&unscaled, &batch).unwrap();
        let out_scaled = apply(&scaled, &batch).unwrap();
        let d_unscaled = col(&out_unscaled, "density");
        let d_scaled = col(&out_scaled, "density");

        let n = values.len() as f64;
        for (i, (u, s)) in d_unscaled.iter().zip(d_scaled.iter()).enumerate() {
            let expected = u * n;
            assert!(
                (s - expected).abs() < 1e-9,
                "point {i}: cumulative scaled {s} != unscaled {u} * n {n} = {expected} (bin_width must NOT apply to the CDF)"
            );
        }
    }
}
