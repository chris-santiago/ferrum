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
    /// this spec's existing `shared_extent` field) AND
    /// [`count_scale_extent`]'s facet pin (mirroring `Bin::apply`'s own
    /// `spec.extent` check — see that field's doc).
    ///
    /// **`shared_extent` reuse is reserved, not exercised.** No wired
    /// producer sets `shared_extent` together with `count_scale` today: the
    /// only producer is `plots/distribution.py::_kde_overlay`, which
    /// hard-codes `shared_extent=False` *deliberately* — `_displot_build`
    /// never forwards `multiple=` into `mark_histogram`, so the real `Bin`
    /// on this path always runs an ungrouped extent, and the overlay must
    /// TRACK that value rather than re-derive `multiple in
    /// ("stack","fill")`. `mark_density` never sets `count_scale` at all.
    /// So the `Some((lo, hi))` arm of the `shared_extent` branch is reachable
    /// only from a hand-composed spec or a future producer; it is
    /// implemented and tested against a real grouped `Bin`
    /// (`test_kde_count_scale_grouped_shared_extent_*`) so that a producer
    /// which does set both gets `Bin`'s semantics rather than a surprise
    /// (batch-F close review, F2 — the earlier wording here justified the
    /// field overload with a claim its own producer's docstring contradicts).
    ///
    /// The unmirrored inputs are enumerated in `PyKde::new`'s public
    /// `count_scale` parameter doc, since they are only reachable by
    /// hand-composing a `Bin` and a `Kde`: `nice` is always assumed `true`,
    /// an explicit `Bin(extent=)` is not mirrored, and `Bin`'s exclusion of
    /// out-of-range rows from its counts is not mirrored. `None` (the
    /// default) reproduces today's output byte-for-byte.
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

/// Refuse a malformed [`KdeSpec::count_scale_extent`] at its consumption
/// boundary, phrased like its `Bin` sibling (`stat_bin: extent must satisfy
/// lo < hi; got (…)`) so the two read as one rule with two slots.
/// Unlike [`KdeSpec::extent`], this slot has no constructor parameter,
/// so `PyKde::new`'s validation never sees it — the wire is its only route,
/// and `deny_unknown_fields` happily accepts the key from a hand-authored or
/// round-tripped `ChartSpec.from_json`. Consumed unvalidated, a reversed
/// `(10, 0)` yields a negative bin width and silently sign-flips every
/// density; a non-finite bound yields a NaN width that quietly NaNs the whole
/// output (batch-F close review, F3 — the one extent slot with no
/// user-facing constructor was also the one with no check, while both
/// siblings validate). Finiteness is checked too because `Kde`'s own
/// constructor checks it for `extent`, and this slot has no constructor to
/// check it anywhere else.
fn validate_count_scale_extent(spec: &KdeSpec) -> PyResult<()> {
    if let Some((lo, hi)) = spec.count_scale_extent {
        if !lo.is_finite() || !hi.is_finite() || lo >= hi {
            return Err(PyValueError::new_err(format!(
                "stat_kde: count_scale_extent must satisfy lo < hi with both finite; \
                 got ({lo}, {hi})"
            )));
        }
    }
    Ok(())
}

pub(crate) fn apply(spec: &KdeSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    validate_count_scale_extent(spec)?;
    if !spec.groupby.is_empty() {
        // `count_scale`'s bin-edge sharing mirrors `Bin::apply_grouped`'s own
        // `resolve_shared_extent(spec.extent, spec.shared_extent, ...)` call
        // exactly, with `count_scale_extent` standing in for `Bin`'s own
        // `spec.extent` role (an explicit/facet-pinned extent always wins,
        // regardless of `shared_extent`'s value — see `count_scale_extent`'s
        // doc): a real histogram only shares one DATA-DERIVED extent across
        // hue levels when `shared_extent` is set. No wired producer sets it
        // together with `count_scale` today (`_kde_overlay` hard-codes
        // `shared_extent=False` to track the real `Bin` on its path — see
        // `KdeSpec::count_scale`'s doc), so in practice both stay `None` and
        // each group derives its own extent, exactly like an un-shared `Bin`.
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
///
///     Parity holds against the histogram ``displot``/``mark_histogram``
///     builds. Hand-composing your own ``Bin`` alongside this ``Kde`` can
///     break it, silently — three ``Bin`` inputs have no ``count_scale``
///     counterpart and are therefore NOT mirrored:
///
///     * ``Bin(nice=...)`` — ``count_scale`` always assumes ``nice=True``
///       (``Bin``'s own default), so a ``Bin(nice=False)`` bins over the raw
///       range while the overlay scales by a niced-range width.
///     * ``Bin(extent=(lo, hi))`` — an explicitly pinned bin range is not
///       carried over; ``count_scale`` derives its own extent per group
///       (or from the facet pin). Only the facet-pinned extent is shared.
///     * ``Bin``'s exclusion of rows outside ``[lo, hi]`` from its counts —
///       ``count_scale`` scales by the group's full non-null count, so a
///       ``Bin(extent=)`` that clips rows shows fewer counts than the
///       overlay assumes.
///
///     In each case the curve is mis-scaled relative to the bars rather
///     than refused, since ``Kde`` cannot see the sibling ``Bin``'s spec.
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
            // Every constructor parameter appears, matching the completeness
            // `Robust::__repr__` gained in this same batch (batch-F close
            // review, F7 — the two siblings had made opposite decisions about
            // repr completeness). `count_scale_extent` is deliberately absent:
            // it has no constructor parameter and is set only by the
            // facet-pinning pass, so it is never part of the value a user
            // wrote.
            TransformSpec::Kde(s) => format!(
                "Kde(field='{}', bandwidth={:?}, bw_adjust={}, n={}, extent={:?}, \
                 cumulative={}, shared_extent={}, kernel='{}', groupby={:?}, \
                 count_scale={:?}, name={:?})",
                s.field, s.bandwidth, s.bw_adjust, s.n, s.extent,
                if s.cumulative { "True" } else { "False" },
                if s.shared_extent { "True" } else { "False" },
                s.kernel, s.groupby, s.count_scale, s.name,
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

// Test coverage lives in `tests.rs` (precedent: `projection/tests.rs`,
// `transform/stats/tests.rs`) — batch F pushed this module past the point where
// an inline `mod tests { ... }` block belongs: 1660 test lines against 775
// source lines, 2.1x. See `tests.rs`'s own header for what it carries.
#[cfg(test)]
mod tests;
