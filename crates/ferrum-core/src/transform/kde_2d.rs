//! Kde2D: 2D Gaussian kernel-density estimate evaluated on a uniform n×n grid.
//!
//! Output is a SINGLE-ROW batch with list-typed columns:
//!   grid_x:  List<Float64>   length n
//!   grid_y:  List<Float64>   length n
//!   density: List<Float64>   length n*n, row-major: density[gy * n + gx]
//!   nx:      UInt32          = n
//!   ny:      UInt32          = n
//!   extent:  List<Float64>   length 4: [xmin, xmax, ymin, ymax]
//!
//! Downstream consumers (Contour) read the grid back from this shape.

use arrow::array::{
    Array, ArrayRef, Float64Array, Float64Builder, ListBuilder, RecordBatch, UInt32Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::transform::color_bin_group::{SlotSpace, StepSpace};
use crate::transform::context::TransformContext;
use crate::transform::core::SpaceSlot;
use crate::transform::log_space::{run_in_log_space, with_mapped_slot, ColumnShape, DemandedField};
use crate::transform::group_key::{
    deserialize_groupby_vec, group_partition_multi, groupby_field_nullable,
    materialize_groupby_col, parse_groupby_pyany, refuse_groupby_output_collision, KeyValue,
    NullKeyPolicy,
};
use crate::transform::kde::{self, BandwidthSpec};

/// The surface columns every `stat_kde_2d` output batch carries, in schema
/// order. Read by the `groupby` output-collision guard (GH #179); pinned
/// against the batch this module actually emits by
/// `output_field_names_match_the_emitted_schema` in the tests below, so the
/// guard cannot fall behind a schema change.
const OUTPUT_FIELDS: [&str; 6] = ["grid_x", "grid_y", "density", "nx", "ny", "extent"];

/// This transform's own voice prefix, passed to every shared
/// `crate::transform::log_space` call so a `Kde2D` refusal never borrows
/// `Bin`'s or `Kde`'s.
const LABEL: &str = "stat_kde_2d";

fn default_bandwidth() -> BandwidthSpec {
    BandwidthSpec::Scott
}

fn default_kde2d_n() -> usize {
    128
}

/// Compute the global 2-D extent `(x_lo, x_hi, y_lo, y_hi)` of `spec.x`/`spec.y`
/// over the full `batch`.
///
/// Returns the RAW per-axis `(min, max)` (no nicing — a KDE surface has no bin
/// edges to align), matching the data-derived extent `apply_one_group` computes
/// when `spec.extent` is `None`. Returns `None` when either field is missing,
/// non-numeric, all values are null/NaN, or an axis range is degenerate
/// (`lo >= hi` or non-finite).
///
/// When `spec.extent` is already set (user-provided), it is returned unchanged so
/// the faceted pin never clobbers an explicit extent.
///
/// NICENESS CONTRACT (XFORM-08): returns the RAW per-axis `(min, max)` (no
/// nicing), mirroring the 1-D `kde::global_extent` and unlike `Bin`. This is the
/// pre-facet extent that `render::prepare` uses to pin the 2-D grid range before
/// partitioning, so every facet panel shares the same KDE grid axes (see
/// `fix_transform_extents_for_facet`). Each axis uses the shared extent helper,
/// which coerces integer-typed fields just as `apply_one_group` does.
///
/// GROUPED-ROWS CONTRACT (GH #197): both axes fold only the rows whose
/// `groupby` key survives the null-key policy, via
/// [`crate::transform::numeric_util::grouped_column_extent`] — a null-keyed row
/// contributes no surface, so letting it widen the grid would spread every
/// surviving group's surface over a region none of them occupies. Identical to
/// the pre-#197 whole-column fold whenever the request is ungrouped or its keys
/// hold no nulls.
pub(crate) fn global_extent(
    spec: &Kde2DSpec,
    batch: &RecordBatch,
) -> Option<(f64, f64, f64, f64)> {
    // When the caller has already pinned an explicit extent, use it directly.
    if let Some(e) = spec.extent {
        return Some(e);
    }
    let (x_lo, x_hi) =
        crate::transform::numeric_util::grouped_column_extent(batch, &spec.x, &spec.groupby)?;
    let (y_lo, y_hi) =
        crate::transform::numeric_util::grouped_column_extent(batch, &spec.y, &spec.groupby)?;
    Some((x_lo, x_hi, y_lo, y_hi))
}

/// [`global_extent`] with `space`'s `x`/`y` slots (whichever is demanded)
/// mapping that slot's column into log units first ([`with_mapped_slot`],
/// each slot into its own column, so `x` and `y` reading one field map it
/// once each) and the extent back after — the facet pin a log-demanded
/// `Kde2D` needs on either or both axes independently, mirroring
/// [`crate::transform::bin::global_extent_in_space`]. A spec stating its
/// own extent pins it as written, as `global_extent` does. `None` when
/// `global_extent` over the mapped columns is, and when a demanded field is
/// missing or non-numeric, or the batch already holds the step-private
/// column name: the pin is skipped, and the step refuses that input in its
/// own voice when it runs.
/// [`crate::transform::log_space::LogAxisSpace::backward`] is monotone
/// increasing on either side, so each mapped-back pair keeps its order
/// with no reordering step.
pub(crate) fn global_extent_in_space(spec: &Kde2DSpec, batch: &RecordBatch, space: &StepSpace) -> Option<(f64, f64, f64, f64)> {
    if spec.extent.is_some() {
        return global_extent(spec, batch);
    }
    let (x_slot, y_slot) = (space.of(SpaceSlot::X), space.of(SpaceSlot::Y));
    let mut mapped_batch = batch.clone();
    let mut mapped_spec = spec.clone();
    if let Some(slot) = x_slot {
        let demand = DemandedField { slot: SpaceSlot::X, field: &spec.x, space: slot, shape: ColumnShape::Scalar };
        (mapped_batch, mapped_spec.x) = with_mapped_slot(&mapped_batch, &demand, LABEL).ok()?;
    }
    if let Some(slot) = y_slot {
        let demand = DemandedField { slot: SpaceSlot::Y, field: &spec.y, space: slot, shape: ColumnShape::Scalar };
        (mapped_batch, mapped_spec.y) = with_mapped_slot(&mapped_batch, &demand, LABEL).ok()?;
    }
    let (x_lo, x_hi, y_lo, y_hi) = global_extent(&mapped_spec, &mapped_batch)?;
    let back = |slot: Option<&SlotSpace>, (lo, hi): (f64, f64)| slot.map_or((lo, hi), |s| (s.space.backward(lo), s.space.backward(hi)));
    let (x_lo, x_hi) = back(x_slot, (x_lo, x_hi));
    let (y_lo, y_hi) = back(y_slot, (y_lo, y_hi));
    Some((x_lo, x_hi, y_lo, y_hi))
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Kde2DSpec {
    pub x: String,
    pub y: String,
    #[serde(default = "default_bandwidth")]
    pub bandwidth: BandwidthSpec,
    #[serde(default = "default_kde2d_n")]
    pub n: usize,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub extent: Option<(f64, f64, f64, f64)>,
    /// When non-empty, partition input by these columns and emit one row per
    /// group. Output schema gains one column per groupby field after the 6th field.
    /// Wire accepts: `null` → `[]`, `"col"` → `["col"]`, `["a","b"]` → `["a","b"]`.
    #[serde(
        skip_serializing_if = "Vec::is_empty",
        default,
        deserialize_with = "deserialize_groupby_vec"
    )]
    pub groupby: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

pub(crate) fn apply(spec: &Kde2DSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    apply_with_context(spec, batch, &TransformContext::default())
}

/// Context-aware entry point. `ctx` is used as the GH #162 degrade-warning
/// sink, for the one degrade this transform can report: an all-null `groupby`
/// key column (GH #197), raised inside the shared partitioner; and, when a
/// drawn log axis demands this `Kde2D` on `x`, `y`, or both
/// (`ctx.log_space`), as the caller-column sink
/// ([`TransformContext::note_log_input`]).
pub(crate) fn apply_with_context(
    spec: &Kde2DSpec,
    batch: &RecordBatch,
    ctx: &TransformContext,
) -> PyResult<RecordBatch> {
    if spec.n == 0 {
        return Err(PyValueError::new_err("stat_kde_2d: n must be > 0"));
    }
    let x_slot = ctx.log_space.as_deref().and_then(|s| s.of(SpaceSlot::X));
    let y_slot = ctx.log_space.as_deref().and_then(|s| s.of(SpaceSlot::Y));
    if x_slot.is_none() && y_slot.is_none() {
        return apply_in_data_space(spec, batch, ctx);
    }
    apply_in_log_space(spec, batch, ctx, x_slot, y_slot)
}

/// [`apply_with_context`]'s ordinary (data-space) body: every `Kde2D`
/// whose axes are not demanded into a log space runs only this, and a
/// demanded one runs it over its mapped columns ([`apply_in_log_space`]).
fn apply_in_data_space(spec: &Kde2DSpec, batch: &RecordBatch, ctx: &TransformContext) -> PyResult<RecordBatch> {
    if !spec.groupby.is_empty() {
        return apply_grouped(spec, batch, &spec.groupby, ctx);
    }
    apply_one_group(spec, batch, None, None)
}

/// `apply_with_context` under a drawn log axis's demand on the `x` slot,
/// the `y` slot, or both independently (`x_slot`/`y_slot`, since a
/// bivariate fit's two axes can each resolve a different `LogScale`, or
/// none, even when both read one column): each demanded axis's half of an
/// explicit `extent` is refused off-side and mapped in that axis's space
/// ([`crate::transform::log_space::LogAxisSpace::extent_in_space`]), the
/// shared [`run_in_log_space`] runner maps each demanded slot's column into
/// its own step-private column, and the ordinary body runs unchanged with
/// only the demanded slots renamed — the fit and the grid for that axis
/// compute there. Every output in an axis's units then maps back on that
/// axis: its grid (`grid_x`/`grid_y`) and its two bounds in `extent`
/// (positions 0..2 for `x`, 2..4 for `y`). `density` stays per log unit on
/// every demanded axis, as a 1-D `Kde` does.
fn apply_in_log_space(
    spec: &Kde2DSpec,
    batch: &RecordBatch,
    ctx: &TransformContext,
    x_slot: Option<&SlotSpace>,
    y_slot: Option<&SlotSpace>,
) -> PyResult<RecordBatch> {
    let in_space = |slot: Option<&SlotSpace>, extent: (f64, f64)| match slot {
        Some(slot) => slot.space.extent_in_space(LABEL, Some(extent)),
        None => Ok(Some(extent)),
    };
    let mapped_extent = match spec.extent {
        None => None,
        Some((x_lo, x_hi, y_lo, y_hi)) => {
            let x = in_space(x_slot, (x_lo, x_hi))?;
            let y = in_space(y_slot, (y_lo, y_hi))?;
            x.zip(y).map(|((x_lo, x_hi), (y_lo, y_hi))| (x_lo, x_hi, y_lo, y_hi))
        }
    };
    let mapped_spec = Kde2DSpec { extent: mapped_extent, ..spec.clone() };
    let demand: Vec<DemandedField> = [(SpaceSlot::X, &spec.x, x_slot), (SpaceSlot::Y, &spec.y, y_slot)]
        .into_iter()
        .filter_map(|(slot, field, space)| space.map(|space| DemandedField { slot, field, space, shape: ColumnShape::Scalar }))
        .collect();
    run_in_log_space(
        ctx,
        LABEL,
        batch,
        &demand,
        |mapped_batch, columns, ctx| {
            let x = columns.column(SpaceSlot::X, &spec.x);
            let y = columns.column(SpaceSlot::Y, &spec.y);
            apply_in_data_space(&Kde2DSpec { x, y, ..mapped_spec }, mapped_batch, ctx)
        },
        &[
            ("grid_x", SpaceSlot::X, ColumnShape::List),
            ("grid_y", SpaceSlot::Y, ColumnShape::List),
            ("extent", SpaceSlot::X, ColumnShape::ListPositions(0, 2)),
            ("extent", SpaceSlot::Y, ColumnShape::ListPositions(2, 4)),
        ],
    )
}

/// Compute a single 2-D KDE surface from a subset of rows (or all rows when
/// `only_indices` is None). The `extent_override` lets the caller supply a
/// pre-computed global extent so all groups share the same grid axes.
fn apply_one_group(
    spec: &Kde2DSpec,
    batch: &RecordBatch,
    only_indices: Option<&[usize]>,
    extent_override: Option<(f64, f64, f64, f64)>,
) -> PyResult<RecordBatch> {
    let schema = batch.schema();
    let xi = schema.index_of(&spec.x).map_err(|_| {
        PyValueError::new_err(format!("stat_kde_2d: column '{}' not found", spec.x))
    })?;
    let yi = schema.index_of(&spec.y).map_err(|_| {
        PyValueError::new_err(format!("stat_kde_2d: column '{}' not found", spec.y))
    })?;

    let xa = crate::transform::numeric_util::coerce_to_float64(
        batch.column(xi),
        "stat_kde_2d",
        &spec.x,
    )?;
    let ya = crate::transform::numeric_util::coerce_to_float64(
        batch.column(yi),
        "stat_kde_2d",
        &spec.y,
    )?;

    let mut xs: Vec<f64> = Vec::with_capacity(xa.len());
    let mut ys: Vec<f64> = Vec::with_capacity(ya.len());
    let push = |i: usize, xs: &mut Vec<f64>, ys: &mut Vec<f64>| {
        if xa.is_null(i) || ya.is_null(i) {
            return;
        }
        let xv = xa.value(i);
        let yv = ya.value(i);
        if !xv.is_nan() && !yv.is_nan() {
            xs.push(xv);
            ys.push(yv);
        }
    };
    match only_indices {
        Some(ixs) => {
            for &i in ixs {
                push(i, &mut xs, &mut ys);
            }
        }
        None => {
            for i in 0..xa.len() {
                push(i, &mut xs, &mut ys);
            }
        }
    }

    let n = spec.n;

    // Compute extent (xmin, xmax, ymin, ymax). Caller-supplied override takes
    // priority (used by apply_grouped for per-group extent control).
    let (xmin, xmax, ymin, ymax) = extent_override.or(spec.extent).unwrap_or_else(|| {
        if xs.is_empty() {
            (0.0, 0.0, 0.0, 0.0)
        } else {
            let (mut xlo, mut xhi) = (f64::INFINITY, f64::NEG_INFINITY);
            let (mut ylo, mut yhi) = (f64::INFINITY, f64::NEG_INFINITY);
            for i in 0..xs.len() {
                if xs[i] < xlo { xlo = xs[i]; }
                if xs[i] > xhi { xhi = xs[i]; }
                if ys[i] < ylo { ylo = ys[i]; }
                if ys[i] > yhi { yhi = ys[i]; }
            }
            (xlo, xhi, ylo, yhi)
        }
    });

    let grid_x: Vec<f64> = (0..n)
        .map(|i| {
            if n == 1 {
                xmin
            } else {
                xmin + (xmax - xmin) * (i as f64) / ((n - 1) as f64)
            }
        })
        .collect();
    let grid_y: Vec<f64> = (0..n)
        .map(|i| {
            if n == 1 {
                ymin
            } else {
                ymin + (ymax - ymin) * (i as f64) / ((n - 1) as f64)
            }
        })
        .collect();

    // Density.
    let density: Vec<f64> = if xs.len() < 2 {
        vec![0.0; n * n]
    } else {
        let hx = kde::bandwidth(&xs, &spec.bandwidth)?;
        let hy = kde::bandwidth(&ys, &spec.bandwidth)?;
        if hx <= 0.0 || hy <= 0.0 || !hx.is_finite() || !hy.is_finite() {
            vec![0.0; n * n]
        } else {
            let n_pts = xs.len() as f64;
            let norm = n_pts * hx * hy * 2.0 * std::f64::consts::PI;
            let mut d = vec![0.0_f64; n * n];
            for gy in 0..n {
                for gx in 0..n {
                    let mut s = 0.0_f64;
                    for i in 0..xs.len() {
                        let dx = (grid_x[gx] - xs[i]) / hx;
                        let dy = (grid_y[gy] - ys[i]) / hy;
                        s += (-0.5 * (dx * dx + dy * dy)).exp();
                    }
                    d[gy * n + gx] = s / norm;
                }
            }
            d
        }
    };

    build_surface_batch(&grid_x, &grid_y, &density, n, (xmin, xmax, ymin, ymax))
}

/// Assemble the 6-column single-row RecordBatch for one 2-D KDE surface.
fn build_surface_batch(
    grid_x: &[f64],
    grid_y: &[f64],
    density: &[f64],
    n: usize,
    extent: (f64, f64, f64, f64),
) -> PyResult<RecordBatch> {
    let (xmin, xmax, ymin, ymax) = extent;
    let mut grid_x_b = ListBuilder::new(Float64Builder::new());
    grid_x_b.values().append_slice(grid_x);
    grid_x_b.append(true);
    let grid_x_arr = grid_x_b.finish();

    let mut grid_y_b = ListBuilder::new(Float64Builder::new());
    grid_y_b.values().append_slice(grid_y);
    grid_y_b.append(true);
    let grid_y_arr = grid_y_b.finish();

    let mut density_b = ListBuilder::new(Float64Builder::new());
    density_b.values().append_slice(density);
    density_b.append(true);
    let density_arr = density_b.finish();

    let mut extent_b = ListBuilder::new(Float64Builder::new());
    extent_b.values().append_slice(&[xmin, xmax, ymin, ymax]);
    extent_b.append(true);
    let extent_arr = extent_b.finish();

    let nx_arr = UInt32Array::from(vec![n as u32]);
    let ny_arr = UInt32Array::from(vec![n as u32]);

    let list_f64 = DataType::List(Arc::new(Field::new("item", DataType::Float64, true)));
    let out_schema = Arc::new(Schema::new(vec![
        Field::new("grid_x", list_f64.clone(), false),
        Field::new("grid_y", list_f64.clone(), false),
        Field::new("density", list_f64.clone(), false),
        Field::new("nx", DataType::UInt32, false),
        Field::new("ny", DataType::UInt32, false),
        Field::new("extent", list_f64, false),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(grid_x_arr),
        Arc::new(grid_y_arr),
        Arc::new(density_arr),
        Arc::new(nx_arr),
        Arc::new(ny_arr),
        Arc::new(extent_arr),
    ];
    RecordBatch::try_new(out_schema, cols)
        .map_err(|e| PyValueError::new_err(format!("stat_kde_2d: {e}")))
}

/// Partition input batch by the groupby columns, compute one 2-D KDE surface
/// per group, then stack the results. Each group yields one output row; the
/// group-key columns are appended after the 6th field so downstream marks can
/// color by group.
///
/// Per-group extents are used by default. The caller-specified `spec.extent`,
/// when present, is applied as a shared override across all groups.
fn apply_grouped(
    spec: &Kde2DSpec,
    batch: &RecordBatch,
    group_cols: &[String],
    warn_ctx: &TransformContext,
) -> PyResult<RecordBatch> {
    // Multi-column partition path (first-appearance order, null-key skip, dtype-preserve).
    let output_fields: Vec<String> = OUTPUT_FIELDS.iter().map(|f| f.to_string()).collect();
    refuse_groupby_output_collision("stat_kde_2d", group_cols, &output_fields)?;
    let partition = group_partition_multi(
        batch,
        group_cols,
        "stat_kde_2d",
        NullKeyPolicy::Skip,
        warn_ctx,
    )?;

    // Per-group extents by default; spec.extent (when set) is a shared override.
    let shared_extent: Option<(f64, f64, f64, f64)> = spec.extent;

    // Collect per-group surface data into parallel flat vectors so we can build
    // the multi-row output batch in a single pass.
    let n = spec.n;
    let mut all_grid_x: ListBuilder<Float64Builder> = ListBuilder::new(Float64Builder::new());
    let mut all_grid_y: ListBuilder<Float64Builder> = ListBuilder::new(Float64Builder::new());
    let mut all_density: ListBuilder<Float64Builder> = ListBuilder::new(Float64Builder::new());
    let mut all_nx: Vec<u32> = Vec::with_capacity(partition.groups.len());
    let mut all_ny: Vec<u32> = Vec::with_capacity(partition.groups.len());
    let mut all_extent: ListBuilder<Float64Builder> = ListBuilder::new(Float64Builder::new());
    let mut group_keys_out: Vec<Vec<KeyValue>> = Vec::with_capacity(partition.groups.len());

    for (g, ixs) in &partition.groups {
        let surface = apply_one_group(spec, batch, Some(ixs), shared_extent)?;

        // Unpack the single-row surface batch back into its list values.
        let gx = list_values_at(&surface, "grid_x")?;
        let gy = list_values_at(&surface, "grid_y")?;
        let dens = list_values_at(&surface, "density")?;
        let ext = list_values_at(&surface, "extent")?;

        all_grid_x.values().append_slice(&gx);
        all_grid_x.append(true);
        all_grid_y.values().append_slice(&gy);
        all_grid_y.append(true);
        all_density.values().append_slice(&dens);
        all_density.append(true);
        all_extent.values().append_slice(&ext);
        all_extent.append(true);
        all_nx.push(n as u32);
        all_ny.push(n as u32);
        group_keys_out.push(g.clone());
    }

    let list_f64 = DataType::List(Arc::new(Field::new("item", DataType::Float64, true)));
    let mut fields = vec![
        Field::new("grid_x", list_f64.clone(), false),
        Field::new("grid_y", list_f64.clone(), false),
        Field::new("density", list_f64.clone(), false),
        Field::new("nx", DataType::UInt32, false),
        Field::new("ny", DataType::UInt32, false),
        Field::new("extent", list_f64, false),
    ];
    // FA-7: preserve the original groupby dtypes (nullable per FA-9; null keys
    // are skipped upstream in group_partition_multi).
    for (name, dtype) in group_cols.iter().zip(partition.dtypes.iter()) {
        fields.push(Field::new(name, dtype.clone(), groupby_field_nullable()));
    }
    let out_schema = Arc::new(Schema::new(fields));
    let mut cols: Vec<ArrayRef> = vec![
        Arc::new(all_grid_x.finish()),
        Arc::new(all_grid_y.finish()),
        Arc::new(all_density.finish()),
        Arc::new(UInt32Array::from(all_nx)),
        Arc::new(UInt32Array::from(all_ny)),
        Arc::new(all_extent.finish()),
    ];
    for (gi, dtype) in partition.dtypes.iter().enumerate() {
        let group_col_arr = materialize_groupby_col(&group_keys_out, gi, dtype)
            .map_err(PyValueError::new_err)?;
        cols.push(group_col_arr);
    }
    RecordBatch::try_new(out_schema, cols)
        .map_err(|e| PyValueError::new_err(format!("stat_kde_2d: {e}")))
}

/// Extract the Float64 values from the first (and only) list entry of a named
/// column in a single-row RecordBatch.
fn list_values_at(batch: &RecordBatch, name: &str) -> PyResult<Vec<f64>> {
    use arrow::array::ListArray;
    let idx = batch.schema().index_of(name).map_err(|_| {
        PyValueError::new_err(format!("stat_kde_2d: column '{}' not found in surface batch", name))
    })?;
    let la = batch
        .column(idx)
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "stat_kde_2d: expected ListArray for column '{}'", name
            ))
        })?;
    let inner = la.value(0);
    let arr = inner.as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
        PyValueError::new_err(format!(
            "stat_kde_2d: expected Float64Array inside list column '{}'", name
        ))
    })?;
    Ok((0..arr.len()).map(|i| arr.value(i)).collect())
}

use pyo3::prelude::*;

use crate::transform::core::TransformSpec;

/// Two-dimensional kernel density estimate (KDE2D).
///
/// Fits a bivariate Gaussian kernel to the (``x``, ``y``) point cloud and
/// evaluates the density on a square grid of ``n × n`` points. The result
/// is the primary input for the ``Contour`` transform.
///
/// Output is a single-row batch with columns ``grid_x`` (List<Float64>,
/// length ``n``), ``grid_y`` (List<Float64>, length ``n``), ``density``
/// (List<Float64>, length ``n*n``, row-major), ``nx`` (UInt32), ``ny``
/// (UInt32), and ``extent`` (List<Float64>, ``[xmin, xmax, ymin, ymax]``).
///
/// When ``groupby`` is set, one row is emitted per distinct group and a
/// ``<groupby>`` column (string, integer, floating-point, or boolean dtype —
/// preserved from the input column, not coerced to Utf8) is appended so
/// downstream marks can color by group. Supported dtypes: Utf8, LargeUtf8,
/// Int8/Int16/Int32/Int64, UInt8/UInt16/UInt32/UInt64, Float32, Float64,
/// Boolean.
///
/// Parameters
/// ----------
/// x : str
///     Numeric column for the horizontal axis (must be Float64).
/// y : str
///     Numeric column for the vertical axis (must be Float64).
/// bandwidth : float or {"scott", "silverman"}, optional
///     Kernel bandwidth applied to both axes. Default is ``"scott"``.
/// n : int, default 128
///     Grid resolution (``n × n`` cells). Must be > 0.
/// extent : (float, float, float, float), optional
///     ``(xmin, xmax, ymin, ymax)`` clipping box; all values must be finite
///     and satisfy ``xmin < xmax``, ``ymin < ymax``.
/// groupby : list[str] or str or None, optional
///     One or more group-key columns. When set, one 2-D KDE surface is
///     computed per distinct group combination. Output schema gains one
///     column per groupby field after the 6th field.
/// name : str, optional
///     Named output label for sibling ``Reorder(from_=...)`` lookup.
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> fm.Chart(df).mark_contour().encode(x="x", y="y")
#[pyclass(eq, module = "ferrum._core", name = "Kde2D")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyKde2D(pub(crate) TransformSpec);

#[pymethods]
impl PyKde2D {
    #[new]
    #[pyo3(signature = (x, y, *, bandwidth = None, n = 128, extent = None, groupby = None, name = None))]
    fn new(
        x: &str,
        y: &str,
        bandwidth: Option<&Bound<'_, PyAny>>,
        n: usize,
        extent: Option<(f64, f64, f64, f64)>,
        groupby: Option<&Bound<'_, PyAny>>,
        name: Option<String>,
    ) -> PyResult<Self> {
        if x.is_empty() {
            return Err(PyValueError::new_err("Kde2D: x must be non-empty"));
        }
        if y.is_empty() {
            return Err(PyValueError::new_err("Kde2D: y must be non-empty"));
        }
        if n == 0 {
            return Err(PyValueError::new_err("Kde2D: n must be > 0"));
        }
        let bw = match bandwidth {
            None => BandwidthSpec::Scott,
            Some(obj) => {
                if let Ok(s) = obj.extract::<String>() {
                    match s.as_str() {
                        "scott" => BandwidthSpec::Scott,
                        "silverman" => BandwidthSpec::Silverman,
                        other => {
                            return Err(PyValueError::new_err(format!(
                                "Kde2D: unknown bandwidth '{other}'; expected 'scott' | 'silverman' | float"
                            )))
                        }
                    }
                } else if let Ok(v) = obj.extract::<f64>() {
                    if !v.is_finite() || v <= 0.0 {
                        return Err(PyValueError::new_err(
                            "Kde2D: numeric bandwidth must be a positive finite number",
                        ));
                    }
                    BandwidthSpec::Fixed { value: v }
                } else {
                    return Err(PyValueError::new_err(
                        "Kde2D: bandwidth must be 'scott', 'silverman', or a positive float",
                    ));
                }
            }
        };
        if let Some((xmin, xmax, ymin, ymax)) = extent {
            if !xmin.is_finite() || !xmax.is_finite() || !ymin.is_finite() || !ymax.is_finite() {
                return Err(PyValueError::new_err(
                    "Kde2D: extent values must all be finite",
                ));
            }
            if xmin >= xmax || ymin >= ymax {
                return Err(PyValueError::new_err(
                    "Kde2D: extent must satisfy xmin < xmax and ymin < ymax",
                ));
            }
        }
        let groupby = parse_groupby_pyany(groupby)?;
        Ok(PyKde2D(TransformSpec::Kde2D(Kde2DSpec {
            x: x.to_string(),
            y: y.to_string(),
            bandwidth: bw,
            n,
            extent,
            groupby,
            name,
        })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Kde2D(s) => format!(
                "Kde2D(x='{}', y='{}', bandwidth={:?}, n={}, extent={:?}, groupby={:?}, name={:?})",
                s.x, s.y, s.bandwidth, s.n, s.extent, s.groupby, s.name
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use arrow::array::{Float64Array, ListArray, RecordBatch, StringArray, UInt32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn batch_xy(xs: Vec<f64>, ys: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("y", DataType::Float64, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(xs)),
                Arc::new(Float64Array::from(ys)),
            ],
        )
        .unwrap()
    }

    fn list_f64_at(batch: &RecordBatch, name: &str) -> Vec<f64> {
        let idx = batch.schema().index_of(name).unwrap();
        let la = batch
            .column(idx)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        assert_eq!(la.len(), 1, "{name} should be single-row list column");
        let inner = la.value(0);
        let arr = inner.as_any().downcast_ref::<Float64Array>().unwrap();
        (0..arr.len()).map(|i| arr.value(i)).collect()
    }

    fn u32_at(batch: &RecordBatch, name: &str) -> u32 {
        let idx = batch.schema().index_of(name).unwrap();
        let arr = batch
            .column(idx)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        assert_eq!(arr.len(), 1);
        arr.value(0)
    }

    #[test]
    fn kde_2d_density_integrates_to_approximately_one() {
        pyo3::Python::initialize();
        // Use a small set of points and a generous extent so the kernel mass is
        // contained within the grid (Σ density · dx · dy ≈ 1).
        let xs = vec![0.0, 1.0, 0.0, 1.0, 0.5];
        let ys = vec![0.0, 0.0, 1.0, 1.0, 0.5];
        let n = 64;
        let spec = Kde2DSpec {
            x: "x".into(),
            y: "y".into(),
            bandwidth: BandwidthSpec::Scott,
            n,
            extent: Some((-3.0, 4.0, -3.0, 4.0)),
            groupby: vec![],
            name: None,
        };
        let b = batch_xy(xs, ys);
        let out = apply(&spec, &b).unwrap();
        assert_eq!(out.num_rows(), 1);

        let grid_x = list_f64_at(&out, "grid_x");
        let grid_y = list_f64_at(&out, "grid_y");
        let density = list_f64_at(&out, "density");
        assert_eq!(grid_x.len(), n);
        assert_eq!(grid_y.len(), n);
        assert_eq!(density.len(), n * n);

        let dx = (grid_x[n - 1] - grid_x[0]) / ((n - 1) as f64);
        let dy = (grid_y[n - 1] - grid_y[0]) / ((n - 1) as f64);
        let sum: f64 = density.iter().sum::<f64>() * dx * dy;
        assert!(
            (sum - 1.0).abs() < 0.2,
            "expected Σ density · dx · dy ≈ 1.0; got {sum}"
        );
    }

    #[test]
    fn kde_2d_extent_explicit_overrides_data_range() {
        pyo3::Python::initialize();
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![0.0, 1.0, 2.0];
        let n = 16;
        let spec = Kde2DSpec {
            x: "x".into(),
            y: "y".into(),
            bandwidth: BandwidthSpec::Scott,
            n,
            extent: Some((-10.0, 10.0, -10.0, 10.0)),
            groupby: vec![],
            name: None,
        };
        let b = batch_xy(xs, ys);
        let out = apply(&spec, &b).unwrap();

        let extent = list_f64_at(&out, "extent");
        assert_eq!(extent, vec![-10.0, 10.0, -10.0, 10.0]);

        let grid_x = list_f64_at(&out, "grid_x");
        let grid_y = list_f64_at(&out, "grid_y");
        assert!((grid_x[0] - (-10.0)).abs() < 1e-12);
        assert!((grid_x[n - 1] - 10.0).abs() < 1e-12);
        assert!((grid_y[0] - (-10.0)).abs() < 1e-12);
        assert!((grid_y[n - 1] - 10.0).abs() < 1e-12);

        assert_eq!(u32_at(&out, "nx"), n as u32);
        assert_eq!(u32_at(&out, "ny"), n as u32);
    }

    #[test]
    fn kde_2d_int64_columns_are_coerced() {
        // Int64 x and y must be coerced to Float64, not rejected.
        use arrow::array::Int64Array;
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int64, true),
            Field::new("y", DataType::Int64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![0_i64, 1, 0, 1, 2])),
                Arc::new(Int64Array::from(vec![0_i64, 0, 1, 1, 2])),
            ],
        )
        .unwrap();
        let n = 8;
        let spec = Kde2DSpec {
            x: "x".into(),
            y: "y".into(),
            bandwidth: BandwidthSpec::Scott,
            n,
            extent: None,
            groupby: vec![],
            name: None,
        };
        let out = apply(&spec, &batch).expect("Int64 KDE2D should succeed");
        assert_eq!(out.num_rows(), 1);
        let density = list_f64_at(&out, "density");
        assert_eq!(density.len(), n * n);
    }

    #[test]
    fn kde_2d_string_column_still_errors() {
        // Negative test: a non-numeric x column must STILL error.
        use arrow::array::StringArray;
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Utf8, true),
            Field::new("y", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            ],
        )
        .unwrap();
        let spec = Kde2DSpec {
            x: "x".into(),
            y: "y".into(),
            bandwidth: BandwidthSpec::Scott,
            n: 8,
            extent: None,
            groupby: vec![],
            name: None,
        };
        let err = apply(&spec, &batch).unwrap_err();
        assert!(err.to_string().contains("must be numeric"), "err: {err}");
    }

    #[test]
    fn kde_2d_round_trip() {
        let with_extent = Kde2DSpec {
            x: "px".into(),
            y: "py".into(),
            bandwidth: BandwidthSpec::Fixed { value: 0.7 },
            n: 64,
            extent: Some((-1.0, 1.0, -2.0, 2.0)),
            groupby: vec![],
            name: Some("k2d".into()),
        };
        let json = serde_json::to_string(&with_extent).unwrap();
        let parsed: Kde2DSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, with_extent);

        let no_extent = Kde2DSpec {
            x: "px".into(),
            y: "py".into(),
            bandwidth: BandwidthSpec::Silverman,
            n: 32,
            extent: None,
            groupby: vec![],
            name: None,
        };
        let json2 = serde_json::to_string(&no_extent).unwrap();
        let parsed2: Kde2DSpec = serde_json::from_str(&json2).unwrap();
        assert_eq!(parsed2, no_extent);
        // Sanity: extent and groupby omitted in serialized form when empty.
        assert!(!json2.contains("extent"), "extent=None must be omitted: {json2}");
        assert!(!json2.contains("groupby"), "groupby=[] must be omitted: {json2}");

        // groupby round-trips through JSON (array form).
        let with_groupby = Kde2DSpec {
            x: "px".into(),
            y: "py".into(),
            bandwidth: BandwidthSpec::Scott,
            n: 16,
            extent: None,
            groupby: vec!["species".into()],
            name: None,
        };
        let json3 = serde_json::to_string(&with_groupby).unwrap();
        assert!(json3.contains("groupby"), "groupby non-empty must appear in JSON: {json3}");
        let parsed3: Kde2DSpec = serde_json::from_str(&json3).unwrap();
        assert_eq!(parsed3, with_groupby);
    }

    // Helper: build a batch with x, y, and a string group column.
    fn batch_xy_g(
        xs: Vec<f64>,
        ys: Vec<f64>,
        gs: Vec<&str>,
    ) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("y", DataType::Float64, true),
            Field::new("g", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(xs)),
                Arc::new(Float64Array::from(ys)),
                Arc::new(StringArray::from(gs)),
            ],
        )
        .unwrap()
    }

    // Extract a Vec<f64> from row `row` of a list column.
    fn list_f64_row(batch: &RecordBatch, name: &str, row: usize) -> Vec<f64> {
        let idx = batch.schema().index_of(name).unwrap();
        let la = batch
            .column(idx)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let inner = la.value(row);
        let arr = inner.as_any().downcast_ref::<Float64Array>().unwrap();
        (0..arr.len()).map(|i| arr.value(i)).collect()
    }

    #[test]
    fn kde_2d_no_groupby_schema_unchanged() {
        // Sentinel: ungrouped output must have exactly 6 columns and 1 row,
        // byte-stable with the pre-groupby implementation.
        pyo3::Python::initialize();
        let xs = vec![0.0, 1.0, 0.0, 1.0];
        let ys = vec![0.0, 0.0, 1.0, 1.0];
        let spec = Kde2DSpec {
            x: "x".into(),
            y: "y".into(),
            bandwidth: BandwidthSpec::Scott,
            n: 8,
            extent: None,
            groupby: vec![],
            name: None,
        };
        let b = batch_xy(xs, ys);
        let out = apply(&spec, &b).unwrap();
        assert_eq!(out.num_rows(), 1, "ungrouped: expected 1 output row");
        assert_eq!(out.num_columns(), 6, "ungrouped: expected 6 output columns");
        let schema = out.schema();
        assert_eq!(schema.field(0).name(), "grid_x");
        assert_eq!(schema.field(1).name(), "grid_y");
        assert_eq!(schema.field(2).name(), "density");
        assert_eq!(schema.field(3).name(), "nx");
        assert_eq!(schema.field(4).name(), "ny");
        assert_eq!(schema.field(5).name(), "extent");
    }

    #[test]
    fn kde_2d_groupby_two_groups_emits_one_row_per_group() {
        // Core groupby test: two well-separated groups → 2 rows in output,
        // distinct surfaces, group column present with correct values.
        pyo3::Python::initialize();

        // Group A: cluster near (0, 0)
        // Group B: cluster near (10, 10)
        // Separation >> bandwidth → surfaces are disjoint; density at each
        // group's grid centre is higher for that group.
        let xs: Vec<f64> = vec![
            0.0, 0.2, 0.1, 0.3, 0.0, // A
            10.0, 10.2, 10.1, 10.3, 10.0, // B
        ];
        let ys: Vec<f64> = vec![
            0.0, 0.2, 0.1, 0.3, 0.0, // A
            10.0, 10.2, 10.1, 10.3, 10.0, // B
        ];
        let gs: Vec<&str> = vec!["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"];
        let n = 8usize;
        let spec = Kde2DSpec {
            x: "x".into(),
            y: "y".into(),
            bandwidth: BandwidthSpec::Scott,
            n,
            extent: None,
            groupby: vec!["g".into()],
            name: None,
        };
        let b = batch_xy_g(xs, ys, gs);
        let out = apply(&spec, &b).unwrap();

        // 2 groups → 2 rows.
        assert_eq!(out.num_rows(), 2, "expected 2 output rows (one per group)");
        // Schema: 6 surface columns + 1 group column = 7.
        assert_eq!(out.num_columns(), 7, "expected 7 output columns with groupby");
        let schema = out.schema();
        assert_eq!(schema.field(6).name(), "g", "7th column must be the group key");
        assert_eq!(
            schema.field(6).data_type(),
            &DataType::Utf8,
            "group column must be Utf8"
        );

        // Group column values must be ["A", "B"] (first-appearance order).
        let garr = out
            .column(6)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(garr.value(0), "A");
        assert_eq!(garr.value(1), "B");

        // Each row must have correctly-sized list columns.
        let gx0 = list_f64_row(&out, "grid_x", 0);
        let gx1 = list_f64_row(&out, "grid_x", 1);
        let gy0 = list_f64_row(&out, "grid_y", 0);
        let gy1 = list_f64_row(&out, "grid_y", 1);
        let d0 = list_f64_row(&out, "density", 0);
        let d1 = list_f64_row(&out, "density", 1);
        assert_eq!(gx0.len(), n);
        assert_eq!(gx1.len(), n);
        assert_eq!(gy0.len(), n);
        assert_eq!(gy1.len(), n);
        assert_eq!(d0.len(), n * n);
        assert_eq!(d1.len(), n * n);

        // Per-group extents: group A grid spans near [0, 0.3], group B near [10, 10.3].
        assert!(
            gx0[0] < 1.0,
            "group A grid_x should start near 0; got {}",
            gx0[0]
        );
        assert!(
            gx1[0] > 9.0,
            "group B grid_x should start near 10; got {}",
            gx1[0]
        );

        // Surfaces are distinct: sum of density differs between groups
        // (both integrate to ≈1 but on very different extents, so when
        // compared at the same absolute coordinates they would be near 0 for
        // the other group's grid, but here we just confirm they're different).
        let sum0: f64 = d0.iter().sum();
        let sum1: f64 = d1.iter().sum();
        // Both should be positive (non-trivial surfaces).
        assert!(sum0 > 0.0, "group A density surface should be non-zero");
        assert!(sum1 > 0.0, "group B density surface should be non-zero");

        // nx/ny columns must equal n for each row.
        let nx_arr = out
            .column(3)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let ny_arr = out
            .column(4)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        assert_eq!(nx_arr.value(0), n as u32);
        assert_eq!(nx_arr.value(1), n as u32);
        assert_eq!(ny_arr.value(0), n as u32);
        assert_eq!(ny_arr.value(1), n as u32);
    }

    #[test]
    fn kde_2d_groupby_int64_group_column() {
        // T0.3 regression: a non-Utf8 (Int64) group column must work, not raise
        // "must be Utf8". Pre-fix this errored in kde_2d::apply_grouped.
        use arrow::array::Int64Array;
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("y", DataType::Float64, true),
            Field::new("g", DataType::Int64, false),
        ]));
        let xs = Float64Array::from(vec![0.0, 0.2, 0.1, 10.0, 10.2, 10.1]);
        let ys = Float64Array::from(vec![0.0, 0.2, 0.1, 10.0, 10.2, 10.1]);
        let gs = Int64Array::from(vec![7_i64, 7, 7, 9, 9, 9]);
        let b = RecordBatch::try_new(schema, vec![Arc::new(xs), Arc::new(ys), Arc::new(gs)]).unwrap();
        let n = 8usize;
        let spec = Kde2DSpec {
            x: "x".into(),
            y: "y".into(),
            bandwidth: BandwidthSpec::Scott,
            n,
            extent: None,
            groupby: vec!["g".into()],
            name: None,
        };
        let out = apply(&spec, &b).expect("Int64 groupby must succeed");
        assert_eq!(out.num_rows(), 2, "one row per group");
        assert_eq!(out.num_columns(), 7);
        // Group column must round-trip as Int64, in first-appearance order.
        assert_eq!(out.schema().field(6).data_type(), &DataType::Int64);
        let garr = out.column(6).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(garr.value(0), 7);
        assert_eq!(garr.value(1), 9);
    }

    #[test]
    fn kde_2d_groupby_boolean_group_column() {
        // T0.3 regression: a Boolean group column must work (was "must be Utf8").
        use arrow::array::BooleanArray;
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("y", DataType::Float64, true),
            Field::new("g", DataType::Boolean, false),
        ]));
        let xs = Float64Array::from(vec![0.0, 0.2, 0.1, 10.0, 10.2, 10.1]);
        let ys = Float64Array::from(vec![0.0, 0.2, 0.1, 10.0, 10.2, 10.1]);
        let gs = BooleanArray::from(vec![true, true, true, false, false, false]);
        let b = RecordBatch::try_new(schema, vec![Arc::new(xs), Arc::new(ys), Arc::new(gs)]).unwrap();
        let spec = Kde2DSpec {
            x: "x".into(),
            y: "y".into(),
            bandwidth: BandwidthSpec::Scott,
            n: 8,
            extent: None,
            groupby: vec!["g".into()],
            name: None,
        };
        let out = apply(&spec, &b).expect("Boolean groupby must succeed");
        assert_eq!(out.num_rows(), 2);
        assert_eq!(out.schema().field(6).data_type(), &DataType::Boolean);
        let garr = out.column(6).as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(garr.value(0), "true group appears first");
        assert!(!garr.value(1), "false group appears second");
    }

    #[test]
    fn kde_2d_groupby_missing_column_errors() {
        pyo3::Python::initialize();
        let b = batch_xy(vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]);
        let spec = Kde2DSpec {
            x: "x".into(),
            y: "y".into(),
            bandwidth: BandwidthSpec::Scott,
            n: 4,
            extent: None,
            groupby: vec!["ghost".into()],
            name: None,
        };
        let err = apply(&spec, &b).unwrap_err();
        assert!(
            err.to_string().contains("ghost"),
            "error message must mention the missing column: {err}"
        );
    }

    // ── R5: global_extent helper ─────────────────────────────────────────────

    /// `global_extent` returns the RAW per-axis (min, max) over both fields,
    /// ignoring any groupby and applying no nicing.
    #[test]
    fn kde_2d_global_extent_returns_raw_per_axis_range() {
        pyo3::Python::initialize();
        // x spans 0.3..9.7, y spans -2.0..5.0; no nicing should be applied.
        let xs = vec![0.3, 0.5, 1.2, 4.8, 9.7];
        let ys = vec![-2.0, 0.0, 1.0, 3.0, 5.0];
        let spec = Kde2DSpec {
            x: "x".into(),
            y: "y".into(),
            bandwidth: BandwidthSpec::Scott,
            n: 16,
            extent: None,
            groupby: vec![],
            name: None,
        };
        let b = batch_xy(xs, ys);
        let ext = super::global_extent(&spec, &b);
        assert_eq!(
            ext,
            Some((0.3, 9.7, -2.0, 5.0)),
            "Kde2D global_extent must be the raw per-axis (xmin, xmax, ymin, ymax)"
        );
    }

    /// `global_extent` returns the explicit extent unchanged when `spec.extent`
    /// is already set, so the faceted pin never clobbers a user value.
    #[test]
    fn kde_2d_global_extent_explicit_passthrough() {
        pyo3::Python::initialize();
        let spec = Kde2DSpec {
            x: "x".into(),
            y: "y".into(),
            bandwidth: BandwidthSpec::Scott,
            n: 16,
            extent: Some((-100.0, 100.0, -50.0, 50.0)),
            groupby: vec![],
            name: None,
        };
        let b = batch_xy(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 2.0]);
        assert_eq!(
            super::global_extent(&spec, &b),
            Some((-100.0, 100.0, -50.0, 50.0)),
            "global_extent must return the explicit extent unchanged"
        );
    }

    /// `global_extent` returns None when an axis field is missing.
    #[test]
    fn kde_2d_global_extent_missing_field_returns_none() {
        pyo3::Python::initialize();
        let spec = Kde2DSpec {
            x: "ghost".into(),
            y: "y".into(),
            bandwidth: BandwidthSpec::Scott,
            n: 16,
            extent: None,
            groupby: vec![],
            name: None,
        };
        let b = batch_xy(vec![0.0, 1.0], vec![0.0, 1.0]);
        assert_eq!(super::global_extent(&spec, &b), None);
    }

    #[test]
    fn kde_2d_groupby_shared_extent_overrides_per_group() {
        // When spec.extent is set, all groups must share it.
        pyo3::Python::initialize();
        let xs: Vec<f64> = vec![0.0, 0.5, 1.0, 5.0, 5.5, 6.0];
        let ys: Vec<f64> = vec![0.0, 0.5, 1.0, 5.0, 5.5, 6.0];
        let gs: Vec<&str> = vec!["A", "A", "A", "B", "B", "B"];
        let shared = (-1.0_f64, 10.0_f64, -1.0_f64, 10.0_f64);
        let n = 4usize;
        let spec = Kde2DSpec {
            x: "x".into(),
            y: "y".into(),
            bandwidth: BandwidthSpec::Scott,
            n,
            extent: Some(shared),
            groupby: vec!["g".into()],
            name: None,
        };
        let b = batch_xy_g(xs, ys, gs);
        let out = apply(&spec, &b).unwrap();
        assert_eq!(out.num_rows(), 2);

        // Both groups must use the shared extent for their grids.
        let gx0 = list_f64_row(&out, "grid_x", 0);
        let gx1 = list_f64_row(&out, "grid_x", 1);
        assert!(
            (gx0[0] - (-1.0)).abs() < 1e-12,
            "group A grid_x[0] should be -1; got {}",
            gx0[0]
        );
        assert!(
            (gx1[0] - (-1.0)).abs() < 1e-12,
            "group B grid_x[0] should be -1; got {}",
            gx1[0]
        );
        assert!(
            (gx0[n - 1] - 10.0).abs() < 1e-12,
            "group A grid_x[n-1] should be 10; got {}",
            gx0[n - 1]
        );
        assert!(
            (gx1[n - 1] - 10.0).abs() < 1e-12,
            "group B grid_x[n-1] should be 10; got {}",
            gx1[n - 1]
        );

        // Extent column must reflect the shared extent for both rows.
        let ext0 = list_f64_row(&out, "extent", 0);
        let ext1 = list_f64_row(&out, "extent", 1);
        assert_eq!(ext0, vec![-1.0, 10.0, -1.0, 10.0]);
        assert_eq!(ext1, vec![-1.0, 10.0, -1.0, 10.0]);
    }

    // ── GH #179: output-collision guard at this append site ─────────────────

    /// The names the collision guard checks must be the names this module
    /// actually emits; a schema change that skipped [`OUTPUT_FIELDS`] would
    /// leave the guard checking a stale list.
    #[test]
    fn output_field_names_match_the_emitted_schema() {
        pyo3::Python::initialize();
        let batch = batch_xy(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 2.0]);
        let spec = Kde2DSpec {
            x: "x".into(),
            y: "y".into(),
            bandwidth: default_bandwidth(),
            n: 4,
            extent: None,
            groupby: vec![],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let schema = out.schema();
        let emitted: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(emitted, OUTPUT_FIELDS.to_vec());
    }

    /// Grouping by a column named like one of the surface columns refuses
    /// (GH #179 residual 1: this append site had no guard).
    #[test]
    fn kde_2d_groupby_colliding_with_an_output_column_refuses() {
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("y", DataType::Float64, true),
            Field::new("density", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![0.0, 1.0])),
                Arc::new(Float64Array::from(vec![0.0, 1.0])),
                Arc::new(StringArray::from(vec!["a", "b"])),
            ],
        )
        .unwrap();
        let spec = Kde2DSpec {
            x: "x".into(),
            y: "y".into(),
            bandwidth: default_bandwidth(),
            n: 4,
            extent: None,
            groupby: vec!["density".into()],
            name: None,
        };
        let err = apply(&spec, &batch)
            .expect_err("grouping by a column named 'density' must refuse");
        assert!(err.to_string().contains("collides"), "{err}");
    }

    // ── Log-space fit (a drawn log axis's demand, independently per axis) ──

    type Space = crate::transform::color_bin_group::LogAxisSpace;

    fn ctx_demanding(space: StepSpace) -> TransformContext {
        TransformContext { log_space: Some(std::rc::Rc::new(space)), ..TransformContext::default() }
    }

    /// The `x` slot demanded along `x` in `x`, the `y` slot along `y` in
    /// `y`, whichever is `Some`.
    fn axis_slots(x: Option<Space>, y: Option<Space>) -> StepSpace {
        use crate::spec::layer::PositionalAxis;
        StepSpace::for_slots(
            [(SpaceSlot::X, PositionalAxis::X, x), (SpaceSlot::Y, PositionalAxis::Y, y)]
                .into_iter()
                .filter_map(|(slot, axis, space)| space.map(|space| (slot, SlotSpace { space, axes: vec![axis], records_caller_input: true }))),
        )
    }

    /// Both `x` and `y` demanded independently, each with its own space.
    fn both_fields(x_space: Space, y_space: Space) -> StepSpace {
        axis_slots(Some(x_space), Some(y_space))
    }

    fn positive_space() -> Space {
        Space { base: 10.0, side: crate::scale::log::LogSide::Positive }
    }

    /// Both axes demanded into log space (a bivariate `kind="kde"` with
    /// `log_scale=True` on both `x` and `y`) grids and fits each axis in
    /// its own log units, and backward-maps `grid_x`/`grid_y` back to
    /// strictly positive data values independently — renders without
    /// error on all-positive, log-separated data, with nothing off-side.
    #[test]
    fn kde_2d_in_log_space_grids_both_axes_independently() {
        pyo3::Python::initialize();
        let xs = vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0];
        let ys = vec![0.5, 1.0, 2.0, 4.0, 8.0, 16.0];
        let n = 8;
        let spec = Kde2DSpec { x: "x".into(), y: "y".into(), bandwidth: default_bandwidth(), n, extent: None, groupby: vec![], name: None };
        let batch = batch_xy(xs, ys);
        let ctx = ctx_demanding(both_fields(positive_space(), positive_space()));
        let out = apply_with_context(&spec, &batch, &ctx).unwrap();
        let grid_x = list_f64_at(&out, "grid_x");
        let grid_y = list_f64_at(&out, "grid_y");
        assert!(grid_x.iter().all(|v| v.is_finite() && *v > 0.0), "{grid_x:?}");
        assert!(grid_y.iter().all(|v| v.is_finite() && *v > 0.0), "{grid_y:?}");
        let recorded = ctx.log_input.borrow();
        assert_eq!(recorded.len(), 2, "both axes' caller columns are recorded: {recorded:?}");
    }

    /// Only `x` demanded into log space: `grid_x` is fit in log units
    /// (strictly positive back in data units) while `grid_y` keeps its
    /// ordinary linear-space grid, which can legitimately include
    /// negative values.
    #[test]
    fn kde_2d_in_log_space_on_one_axis_only_maps_that_axis() {
        pyo3::Python::initialize();
        let xs = vec![1.0, 2.0, 4.0, 8.0, 16.0];
        let ys = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let n = 8;
        let spec = Kde2DSpec { x: "x".into(), y: "y".into(), bandwidth: default_bandwidth(), n, extent: None, groupby: vec![], name: None };
        let batch = batch_xy(xs, ys);
        let ctx = ctx_demanding(axis_slots(Some(positive_space()), None));
        let out = apply_with_context(&spec, &batch, &ctx).unwrap();
        let grid_x = list_f64_at(&out, "grid_x");
        let grid_y = list_f64_at(&out, "grid_y");
        assert!(grid_x.iter().all(|v| v.is_finite() && *v > 0.0), "{grid_x:?}");
        assert!(grid_y.iter().any(|v| *v < 0.0), "y keeps its ordinary linear-space grid: {grid_y:?}");
    }

    /// Every row off-side on the one demanded axis (an all-negative `x` on
    /// the positive side): the body runs on the zero-row mapped slice
    /// rather than the real one, but the recorded caller column still
    /// holds every `x` row — the axis resolver's own
    /// `LogScaleNoValueOnSide` refusal reads it, not this transform's own
    /// (degenerate) surface.
    #[test]
    fn kde_2d_in_log_space_all_off_side_on_the_demanded_axis_still_records_the_caller_column() {
        pyo3::Python::initialize();
        let xs = vec![-1.0, -2.0, -4.0];
        let ys = vec![1.0, 2.0, 3.0];
        let spec = Kde2DSpec { x: "x".into(), y: "y".into(), bandwidth: default_bandwidth(), n: 8, extent: None, groupby: vec![], name: None };
        let batch = batch_xy(xs, ys);
        let ctx = ctx_demanding(axis_slots(Some(positive_space()), None));
        let _ = apply_with_context(&spec, &batch, &ctx).unwrap();
        let recorded = ctx.log_input.borrow();
        let column = recorded.get(&(crate::spec::layer::PositionalAxis::X, "x".to_string())).expect("the walk's root records the caller's column");
        assert_eq!(column.len(), 3, "the recorded column keeps every caller row, even when none are on-side");
    }

    /// A context demanding the `x` slot in `x` and the `y` slot in `y`,
    /// whichever is `Some`.
    fn ctx_demanding_axes(x: Option<Space>, y: Option<Space>) -> TransformContext {
        ctx_demanding(axis_slots(x, y))
    }

    /// The facet pin a `Kde2D` takes with its `x`/`y` slots demanded in `x`/`y`.
    fn log_pin(spec: &Kde2DSpec, batch: &RecordBatch, x: Option<Space>, y: Option<Space>) -> Option<(f64, f64, f64, f64)> {
        global_extent_in_space(spec, batch, &axis_slots(x, y))
    }

    fn base_two_space() -> Space {
        Space { base: 2.0, side: crate::scale::log::LogSide::Positive }
    }

    /// `name`'s one list in `batch`, every entry (null as NaN). `pub(crate)`
    /// so a `Kde2D` facet-pin test outside this module (`render::prepare::extent`)
    /// reads a `Kde2D` output batch's list columns the same way, rather than
    /// keeping a second copy.
    pub(crate) fn list_values(batch: &RecordBatch, name: &str) -> Vec<f64> {
        let list = batch.column(batch.schema().index_of(name).unwrap()).as_any().downcast_ref::<ListArray>().unwrap().value(0);
        let values = list.as_any().downcast_ref::<Float64Array>().unwrap();
        (0..values.len()).map(|i| if values.is_null(i) { f64::NAN } else { values.value(i) }).collect()
    }

    /// `pub(crate)` for the same reason as [`list_values`].
    pub(crate) fn assert_close(name: &str, actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len(), "{name}");
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() <= 1e-9 * e.abs().max(1.0), "{name}[{i}]: {a} vs {e}");
        }
    }

    /// The fit a `Kde2D` demanded in `x`/`y` must equal: the data-space fit
    /// over each demanded axis's column mapped forward by hand into columns
    /// of their own names, with each demanded axis's grid and extent bounds
    /// mapped back.
    fn assert_matches_the_hand_mapped_fit(spec: &Kde2DSpec, batch: &RecordBatch, x: Option<Space>, y: Option<Space>) {
        let out = apply_with_context(spec, batch, &ctx_demanding_axes(x, y)).unwrap();
        let column = |field: &str, space: Option<Space>| -> Vec<f64> {
            let values = crate::transform::numeric_util::coerce_to_float64(batch.column(batch.schema().index_of(field).unwrap()), "test", field).unwrap();
            values.iter().map(|v| v.map_or(f64::NAN, |v| space.map_or(v, |s| s.forward(v).unwrap_or(f64::NAN)))).collect()
        };
        let hand_mapped = batch_xy(column(&spec.x, x), column(&spec.y, y));
        let oracle = apply(&Kde2DSpec { x: "x".into(), y: "y".into(), ..spec.clone() }, &hand_mapped).unwrap();
        let back = |values: Vec<f64>, space: Option<Space>| -> Vec<f64> { values.into_iter().map(|v| space.map_or(v, |s| s.backward(v))).collect() };
        assert_close("grid_x", &list_values(&out, "grid_x"), &back(list_values(&oracle, "grid_x"), x));
        assert_close("grid_y", &list_values(&out, "grid_y"), &back(list_values(&oracle, "grid_y"), y));
        assert_close("density", &list_values(&out, "density"), &list_values(&oracle, "density"));
        let extent = list_values(&oracle, "extent");
        let expected_extent = [back(extent[0..2].to_vec(), x), back(extent[2..4].to_vec(), y)].concat();
        assert_close("extent", &list_values(&out, "extent"), &expected_extent);
    }

    fn one_column(values: &[f64]) -> RecordBatch {
        RecordBatch::try_new(Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, true)])), vec![Arc::new(Float64Array::from(values.to_vec()))]).unwrap()
    }

    const SPREAD: [f64; 10] = [1.5, 2.0, 3.0, 4.5, 7.0, 10.0, 15.0, 22.0, 35.0, 50.0];

    /// A `Kde2D` reading one column on both axes computes each axis's slot
    /// from the caller's column mapped once into that slot's own space: a
    /// log `x` beside a linear `y` fits `(log10 v, v)`, two log axes fit
    /// `(log10 v, log10 v)`, and a base-10 `x` beside a base-2 `y` fits
    /// `(log10 v, log2 v)`. Rejects keying the demand by the column name,
    /// which maps the one column for both slots, or maps it twice.
    #[test]
    fn kde_2d_reading_one_column_on_both_axes_maps_each_slot_once_in_its_own_space() {
        pyo3::Python::initialize();
        let spec = Kde2DSpec { x: "v".into(), y: "v".into(), bandwidth: default_bandwidth(), n: 6, extent: None, groupby: vec![], name: None };
        let batch = one_column(&SPREAD);
        assert_matches_the_hand_mapped_fit(&spec, &batch, Some(positive_space()), None);
        assert_matches_the_hand_mapped_fit(&spec, &batch, None, Some(positive_space()));
        assert_matches_the_hand_mapped_fit(&spec, &batch, Some(positive_space()), Some(positive_space()));
        assert_matches_the_hand_mapped_fit(&spec, &batch, Some(positive_space()), Some(base_two_space()));
    }

    /// A log-demanded `Kde2D` maps every output it emits in an axis's units
    /// back on that axis, the `extent` list included: positions 0..2 hold
    /// `x`'s bounds and 2..4 `y`'s, so an `x`-only demand maps only the
    /// first two. Rejects leaving `extent` in log units beside data-unit
    /// grids.
    #[test]
    fn kde_2d_maps_each_axis_extent_bounds_back_on_its_own_axis() {
        pyo3::Python::initialize();
        let spec = Kde2DSpec { x: "x".into(), y: "y".into(), bandwidth: default_bandwidth(), n: 6, extent: None, groupby: vec![], name: None };
        let batch = batch_xy(SPREAD.to_vec(), SPREAD.iter().rev().map(|v| v * 3.0).collect());
        assert_matches_the_hand_mapped_fit(&spec, &batch, Some(positive_space()), None);
        assert_matches_the_hand_mapped_fit(&spec, &batch, None, Some(base_two_space()));
        assert_matches_the_hand_mapped_fit(&spec, &batch, Some(positive_space()), Some(base_two_space()));
    }

    /// A `Kde2D` whose demanded `x` is an integer, `Float32` or non-nullable
    /// column holding an off-side 0 fits exactly as over the same values in
    /// a nullable Float64 column, and so does its facet pin. Rejects writing
    /// the mapped Float64 values back under the caller's field.
    #[test]
    fn kde_2d_in_log_space_fits_every_numeric_dtype_like_float64() {
        pyo3::Python::initialize();
        let spec = Kde2DSpec { x: "x".into(), y: "y".into(), bandwidth: default_bandwidth(), n: 6, extent: None, groupby: vec![], name: None };
        let values: Vec<i64> = (0..=20).collect();
        let ys: Vec<f64> = values.iter().map(|&v| ((v * 7) % 11) as f64).collect();
        let float = batch_xy(values.iter().map(|&v| v as f64).collect(), ys.clone());
        let expected = apply_with_context(&spec, &float, &ctx_demanding_axes(Some(positive_space()), None)).unwrap();
        let expected_pin = log_pin(&spec, &float, Some(positive_space()), None);
        assert!(expected_pin.is_some());
        for (dtype, field, column) in crate::transform::log_space::numeric_dtype_columns("x", &values) {
            let batch = RecordBatch::try_new(
                Arc::new(Schema::new(vec![field, Field::new("y", DataType::Float64, true)])),
                vec![column, Arc::new(Float64Array::from(ys.clone()))],
            )
            .unwrap();
            let out = apply_with_context(&spec, &batch, &ctx_demanding_axes(Some(positive_space()), None)).unwrap_or_else(|e| panic!("{dtype}: {e}"));
            assert_eq!(out, expected, "{dtype}");
            assert_eq!(log_pin(&spec, &batch, Some(positive_space()), None), expected_pin, "{dtype}");
        }
    }

    /// A log-demanded `Kde2D` over a column the batch lacks refuses in its
    /// own `stat_kde_2d:` voice, naming the column.
    #[test]
    fn a_log_kde_2d_over_a_missing_field_refuses_in_its_own_voice() {
        pyo3::Python::initialize();
        let spec = Kde2DSpec { x: "absent".into(), y: "y".into(), bandwidth: default_bandwidth(), n: 6, extent: None, groupby: vec![], name: None };
        let batch = batch_xy(SPREAD.to_vec(), SPREAD.to_vec());
        let err = apply_with_context(&spec, &batch, &ctx_demanding_axes(Some(positive_space()), None)).unwrap_err().to_string();
        assert!(err.contains("stat_kde_2d: column 'absent' not found"), "{err}");
    }

}
