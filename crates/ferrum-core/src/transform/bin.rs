use arrow::array::{ArrayRef, Float64Array, RecordBatch, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::scale::ticks::sturges_floor;
use crate::transform::aggregate::{AggColumn, AggFn};
use crate::transform::vocabulary::Vocabulary;
use crate::transform::bin_mode::{resolve_bin_count, BinMode};
use crate::transform::context::TransformContext;
use crate::transform::group_key::{
    deserialize_groupby_vec, group_partition_multi, groupby_field_nullable,
    materialize_groupby_col, parse_groupby_pyany, refuse_groupby_output_collision, KeyValue,
    NullKeyPolicy,
};
use crate::transform::numeric_util::{
    clean_float64_values, clean_float64_values_indexed, coerce_to_float64, grouped_column_extent,
    resolve_shared_extent,
};

// ── Shared private helpers ────────────────────────────────────────────────────

/// Apply nice rounding to a raw `(lo, hi)` extent for a target bin count.
///
/// Computes the nice step size via `nice_step`, then floors `lo` and ceils
/// `hi` to the nearest multiple of that step.  When the step is non-positive
/// or non-finite (e.g. degenerate extent), the original values are returned
/// unchanged.
///
/// This is the single source for the `nice_step` + floor/ceil block that
/// previously appeared separately in `global_extent` and `apply_one_group`.
fn nice_extent(lo: f64, hi: f64, target: usize) -> (f64, f64) {
    let step = crate::scale::ticks::nice_step(lo, hi, target);
    if step.is_finite() && step > 0.0 {
        ((lo / step).floor() * step, (hi / step).ceil() * step)
    } else {
        (lo, hi)
    }
}

/// Resolve the `(lo, hi, n_bins)` triple `apply_one_group` derives from an
/// already-known non-degenerate `(lo, hi)` extent, given `mode` and whether
/// nice-rounding applies. Extracted from `apply_one_group`'s inline
/// nice-rounding + bin-count block (batch F) so `Kde`'s opt-in count-scale
/// feature (`KdeSpec::count_scale`) can derive `Bin`'s exact bin width via
/// the same primitives (`nice_extent`, `resolve_bin_count`) instead of
/// re-implementing the decision — see `crate::transform::kde`'s
/// `resolve_group_bin_width`. `should_nice` mirrors `apply_one_group`'s own
/// `spec.nice && spec.extent.is_none()` gate: nicing only ever applies to an
/// auto-derived extent, never a caller-pinned one. `ctx` is threaded through
/// to `resolve_bin_count`'s error messages, same as the inline call it
/// replaces. The nicing target comes from [`BinMode::nice_target_count`] — see
/// its NICING-TARGET CONTRACT doc for why every rule-derived mode (Sturges,
/// Width, and since #193 Freedman-Diaconis) nices against the Sturges count.
pub(crate) fn resolve_bin_extent(
    clean: &[f64],
    mode: &BinMode,
    lo: f64,
    hi: f64,
    should_nice: bool,
    ctx: &str,
) -> PyResult<(f64, f64, usize)> {
    let (lo, hi) = if should_nice {
        let target = mode
            .nice_target_count()
            .unwrap_or_else(|| sturges_floor(clean.len()))
            .max(1);
        nice_extent(lo, hi, target)
    } else {
        (lo, hi)
    };
    let n_bins = resolve_bin_count(mode, clean, lo, hi, ctx)?;
    Ok((lo, hi, n_bins))
}

/// 1-D equal-width binning spec.
///
/// The bin-count decision is a typed [`BinMode`] (no overlapping
/// `bin_count`/`bin_width` `Option`s). The flat `bin_count`/`bin_width` JSON wire
/// keys are preserved by the [`BinWire`] serde shim (see
/// `#[serde(into/try_from)]`), so existing wire payloads round-trip
/// byte-for-byte and the precedence (`bin_count` wins over `bin_width` wins over
/// Sturges) is resolved once at the (de)serialization boundary rather than at
/// apply time. A rule the numeric keys cannot spell rides the additive
/// `bin_rule` key (#193), so every mode survives `spec → wire → spec`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(into = "BinWire", try_from = "BinWire")]
pub(crate) struct BinSpec {
    pub field: String,
    pub mode: BinMode,
    pub extent: Option<(f64, f64)>,
    pub nice: bool,
    pub cumulative: bool,
    /// When ``true`` and ``groupby`` is set, compute a single global extent
    /// from all rows before per-group binning so every group shares the same
    /// bin edges. Required for ``multiple="stack"`` / ``"fill"`` to produce
    /// aligned bars that can be stacked correctly.
    pub shared_extent: bool,
    /// When non-empty, partition input by these columns and emit per-(bin, group)
    /// rows. Output schema gains one column per groupby field after the 4th
    /// field (5th when `agg` is set — see `agg`/`agg_field`).
    pub groupby: Vec<String>,
    /// Column to aggregate per bin (batch-E T7). Must be set together with
    /// [`agg`](Self::agg) — either both are `Some` or both are `None`.
    pub agg_field: Option<String>,
    /// Per-bin aggregate method, typed at the door (an alias is stored as its
    /// canonical fn, as `AggregateOp.fn_` is). When set (together
    /// with `agg_field`), the output batch gains an `agg_value: Float64`
    /// column: the aggregate of `agg_field`'s values over the rows in each
    /// bin, using the same null/NaN-drop policy `Aggregate` uses. A
    /// structurally empty bin (no rows fall in its range) yields a null
    /// `agg_value` — not `NaN`, not `0` — distinct from a non-empty bin whose
    /// `agg_field` values are all null/NaN, which yields `Aggregate`'s own
    /// all-null convention (`NaN`, or `0` for `count`/`distinct`).
    /// Unaffected by `cumulative`: a per-bin aggregate like `median` has no
    /// well-defined running-total form, so `agg_value` is always the
    /// non-cumulative per-bin statistic even when `cumulative=true`.
    pub agg: Option<AggFn>,
    pub name: Option<String>,
}

/// Flat serde projection of [`BinSpec`] preserving the legacy wire shape.
///
/// `BinSpec` serializes/deserializes through this struct so the wire keeps the
/// flat `bin_count`/`bin_width`/`groupby` keys that Python and the test suite
/// assert on. `BinMode` is never serialized directly for the 1-D `Bin`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BinWire {
    field: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    bin_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    bin_width: Option<f64>,
    /// The rule-derived mode carrier (#193). The flat numeric keys can only
    /// spell `Fixed` and `Width`, so before this key existed
    /// `BinMode::FreedmanDiaconis` serialized as "neither" and read back as
    /// Sturges — a chart's `to_dict()`/`to_json()` stated binning it did not
    /// render. `BinWire` is `deny_unknown_fields`, so the carrier has to be an
    /// additive, explicitly-named key rather than a widened numeric one.
    ///
    /// Written only for Freedman-Diaconis: Sturges remains the wire's *default*
    /// (the absence of all three keys), so every pre-#193 payload keeps its
    /// exact bytes. The name itself comes from [`BinMode::rule_name`], the same
    /// table `parse_bin_axis` accepts, so a parseable rule is always writable.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    bin_rule: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    extent: Option<(f64, f64)>,
    #[serde(default = "default_true")]
    nice: bool,
    #[serde(default)]
    cumulative: bool,
    #[serde(default)]
    shared_extent: bool,
    /// Wire accepts: `null` → `[]`, `"col"` → `["col"]`, `["a","b"]` → `["a","b"]`.
    #[serde(
        skip_serializing_if = "Vec::is_empty",
        default,
        deserialize_with = "deserialize_groupby_vec"
    )]
    groupby: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    agg_field: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    agg: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    name: Option<String>,
}

impl TryFrom<BinWire> for BinSpec {
    type Error = String;

    fn try_from(w: BinWire) -> Result<Self, String> {
        // Resolve the wire's mode keys once at the boundary. `bin_rule` names a
        // mode outright and so cannot be combined with a numeric key — that
        // pair has no single reading, and the wire refuses rather than picking
        // one. Among the two numeric keys the pre-#193 precedence is unchanged:
        // bin_count wins, then bin_width, else Sturges.
        let mode = match (&w.bin_rule, w.bin_count, w.bin_width) {
            (Some(rule), None, None) => BinMode::from_rule_name(rule).ok_or_else(|| {
                format!("stat_bin: unknown bin_rule '{rule}'; expected 'sturges'|'fd'|'freedman_diaconis'")
            })?,
            (Some(rule), _, _) => {
                return Err(format!(
                    "stat_bin: bin_rule '{rule}' cannot be combined with bin_count/bin_width"
                ))
            }
            (None, Some(n), _) => BinMode::Fixed { n },
            (None, None, Some(width)) => BinMode::Width { w: width },
            (None, None, None) => BinMode::Sturges,
        };
        let agg = w
            .agg
            .map(|m| AggFn::parse_or_refuse("agg", &m))
            .transpose()
            .map_err(|e| format!("stat_bin: {e}"))?;
        Ok(BinSpec {
            field: w.field,
            mode,
            extent: w.extent,
            nice: w.nice,
            cumulative: w.cumulative,
            shared_extent: w.shared_extent,
            groupby: w.groupby,
            agg_field: w.agg_field,
            agg,
            name: w.name,
        })
    }
}

impl From<BinSpec> for BinWire {
    fn from(s: BinSpec) -> Self {
        // Exactly one of the three mode keys is ever written, and Sturges writes
        // none of them (see `BinWire::bin_rule`). `explicit_rule_name` decides
        // which modes the numeric keys cannot express, and supplies a spelling
        // `try_from` reads back.
        let (bin_count, bin_width) = match &s.mode {
            BinMode::Fixed { n } => (Some(*n), None),
            BinMode::Width { w } => (None, Some(*w)),
            BinMode::Sturges | BinMode::FreedmanDiaconis => (None, None),
        };
        let bin_rule = s.mode.explicit_rule_name().map(str::to_string);
        BinWire {
            field: s.field,
            bin_count,
            bin_width,
            bin_rule,
            extent: s.extent,
            nice: s.nice,
            cumulative: s.cumulative,
            shared_extent: s.shared_extent,
            groupby: s.groupby,
            agg_field: s.agg_field,
            agg: s.agg.map(|f| f.as_str().to_string()),
            name: s.name,
        }
    }
}

fn default_true() -> bool { true }

/// Compute the global (optionally niced) `(lo, hi)` extent of `spec.field`
/// over the full `batch`.
///
/// Returns `None` when the field is missing, non-numeric, or all values are
/// null/NaN. When `spec.nice` is true and `spec.extent` is `None`, applies the
/// same nicing logic as `apply_one_group` so that panels aligned to this extent
/// will produce the same bin edges as the grouped path would compute.
///
/// This is the pre-facet extent that `render::prepare` uses to pin the value
/// axis before partitioning, so every facet panel shares the same bin edges
/// (see `fix_transform_extents_for_facet`).
pub(crate) fn global_extent(spec: &BinSpec, batch: &RecordBatch) -> Option<(f64, f64)> {
    // When the caller has already pinned an explicit extent, use it directly.
    if let Some(e) = spec.extent {
        return Some(e);
    }
    // GROUPED-ROWS CONTRACT (GH #197): fold only the rows whose `groupby` key
    // survives the null-key policy — the same fold `resolve_shared_extent`
    // applies on the in-transform path, so the faceted pin and the
    // `shared_extent` resolution cannot disagree about which rows the shared bin
    // edges are derived from. A null-keyed row is binned into no group, so
    // letting it widen the edges would spread every surviving group's histogram
    // over a range none of them occupies. Identical to the pre-#197
    // whole-column fold whenever the request is ungrouped or its keys hold no
    // nulls.
    let (lo, hi) = grouped_column_extent(batch, &spec.field, &spec.groupby)?;
    // NICENESS CONTRACT (XFORM-08): `Bin` nices its faceted-pin extent so the
    // pinned range reproduces the same bin edges every per-group/per-panel
    // partition would compute. This is the one extent-deriving transform that
    // nices — `Kde`/`Violin`/`DensityData`/`Bin2D`/`Hex`/`Raster`/`Kde2D` all
    // return the raw `column_extent`. The divergence lives here, at the call
    // site, not in the shared fold. Only nice when `nice=true` and no explicit
    // extent was set.
    if spec.nice {
        // Same nicing target `resolve_bin_extent` uses per group, asked through
        // the same [`BinMode::nice_target_count`] — Width/Sturges/FD all fall
        // back to the Sturges count, so the faceted pin and the per-panel
        // partition cannot disagree about the edges. Only the fallback arm
        // scans the column for `clean.len()`, so a `Fixed{n}` spec still pays
        // no second column pass.
        let target = match spec.mode.nice_target_count() {
            Some(n) => n.max(1),
            None => {
                let schema = batch.schema();
                let idx = schema.index_of(&spec.field).ok()?;
                let arr = coerce_to_float64(batch.column(idx), "stat_bin", &spec.field).ok()?;
                let n = clean_float64_values(&arr, None).len();
                sturges_floor(n).max(1)
            }
        };
        Some(nice_extent(lo, hi, target))
    } else {
        Some((lo, hi))
    }
}

pub(crate) fn apply(spec: &BinSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    apply_with_context(spec, batch, &TransformContext::default())
}

/// Context-aware entry point. `ctx` is used only as the GH #162 degrade-warning
/// sink, for the one degrade `Bin` can report: an all-null `groupby` key column
/// (GH #197), raised inside the shared partitioner.
pub(crate) fn apply_with_context(
    spec: &BinSpec,
    batch: &RecordBatch,
    ctx: &TransformContext,
) -> PyResult<RecordBatch> {
    // Resolved ONCE regardless of groupby, so agg_field/agg pairing and
    // vocabulary validation happen exactly once (not once per group, and not
    // skipped when a grouped batch happens to produce zero groups).
    let agg_col = resolve_bin_agg(spec, batch)?;
    // D-GROUPBY-1: groupby is now Vec<String>. Non-empty → grouped path.
    if !spec.groupby.is_empty() {
        return apply_grouped(spec, batch, &spec.groupby, agg_col.as_ref(), ctx);
    }
    apply_one_group(spec, batch, None, agg_col.as_ref())
}

/// Resolve `spec`'s optional per-bin aggregate column (batch-E T7).
///
/// `agg_field`/`agg` must be set together ([`check_agg_pairing`]; a wire-built
/// spec meets that check only here). When both are set, `agg_field` is
/// resolved via [`AggColumn::resolve`], the same numeric-widening/null-drop policy
/// `stat_aggregate` uses — so a bin's `agg_value` matches what `Aggregate`
/// would compute over the same rows.
fn resolve_bin_agg(spec: &BinSpec, batch: &RecordBatch) -> PyResult<Option<AggColumn>> {
    check_agg_pairing(spec.agg_field.is_some(), spec.agg.is_some())
        .map_err(|e| PyValueError::new_err(format!("stat_bin: {e}")))?;
    let (Some(field), Some(fn_)) = (&spec.agg_field, spec.agg) else {
        return Ok(None);
    };
    let schema = batch.schema();
    let idx = schema.index_of(field).map_err(|_| {
        PyValueError::new_err(format!("stat_bin: agg_field column '{field}' not found"))
    })?;
    Ok(Some(AggColumn::resolve(batch.column(idx), field, fn_, "stat_bin")?))
}

/// `Bin`'s field requirement: `agg_field` and `agg` are set together or not
/// at all. The one refusal text; each door prefixes itself.
fn check_agg_pairing(has_field: bool, has_agg: bool) -> Result<(), &'static str> {
    if has_field == has_agg {
        Ok(())
    } else {
        Err("agg_field and agg must both be set, or both omitted")
    }
}

fn apply_one_group(
    spec: &BinSpec,
    batch: &RecordBatch,
    only_indices: Option<&[usize]>,
    agg_col: Option<&AggColumn>,
) -> PyResult<RecordBatch> {
    build_bin_batch(compute_one_group(spec, batch, only_indices, agg_col)?)
}

/// This transform's per-group compute: resolve the column, clean/bin the
/// values, and return the four (or five, with `agg`) output columns as plain
/// `Vec`s — no `RecordBatch` built here. [`apply_one_group`] wraps this with
/// [`build_bin_batch`] for the ungrouped entry point; [`apply_grouped`] calls
/// this directly per partition and extends its own accumulator `Vec`s from
/// the typed result.
fn compute_one_group(
    spec: &BinSpec,
    batch: &RecordBatch,
    only_indices: Option<&[usize]>,
    agg_col: Option<&AggColumn>,
) -> PyResult<BinColumns> {
    let schema = batch.schema();
    let idx = schema.index_of(&spec.field).map_err(|_| {
        PyValueError::new_err(format!(
            "stat_bin: column '{}' not found; available: {:?}",
            spec.field,
            schema.fields().iter().map(|f| f.name()).collect::<Vec<_>>()
        ))
    })?;
    // Auto-cast integer types to Float64 so that Int32/Int64 columns work without
    // requiring the caller to pre-cast (e.g. JointChart marginal histograms).
    // `coerce_to_float64` is the single source for this cast (XFORM-05).
    let arr = coerce_to_float64(batch.column(idx), "stat_bin", &spec.field)?;

    // Drop nulls and NaN; optionally restrict to a subset of row indices (for
    // groupby). `row_idx[i]` is `clean[i]`'s ORIGINAL row index, needed to
    // look up `agg_field`'s value for exactly the same rows a bin counts.
    let (clean, row_idx) = clean_float64_values_indexed(&arr, only_indices);

    // Empty input → empty output (per spec §6: stat_bin is the exception that allows empty)
    if clean.is_empty() {
        return Ok(empty_bin_columns(agg_col.is_some()));
    }

    let (lo, hi) = match spec.extent {
        Some((a, b)) if a < b => (a, b),
        Some((a, b)) => return Err(PyValueError::new_err(format!(
            "stat_bin: extent must satisfy lo < hi; got ({a}, {b})"
        ))),
        None => {
            let (lo, hi) = clean.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                (a.min(v), b.max(v))
            });
            if lo == hi {
                // Spec §4.1 numeric edge: all-equal → single unit bin. Every
                // clean row falls in this one bin, so its agg_value (when
                // requested) aggregates the full `row_idx` set.
                let agg_value = agg_col.map(|ac| ac.aggregate_rows(&row_idx));
                return Ok(single_unit_bin_columns(lo, clean.len() as u64, agg_value));
            }
            (lo, hi)
        }
    };

    // Optional "nice" rounding of the extent, plus bin-count resolution.
    // Only nices when extent is auto-derived (not when the caller explicitly
    // set extent). See `resolve_bin_extent`'s doc for the target-selection
    // rule (Fixed{n} vs Sturges fallback).
    let should_nice = spec.nice && spec.extent.is_none();
    let (lo, hi, n_bins) = resolve_bin_extent(&clean, &spec.mode, lo, hi, should_nice, "stat_bin")?;

    let edges: Vec<f64> = (0..=n_bins)
        .map(|i| lo + (hi - lo) * (i as f64) / (n_bins as f64))
        .collect();

    let mut counts = vec![0u64; n_bins];
    // Per-bin ORIGINAL row indices, collected only when an aggregate was
    // requested (avoids the allocation entirely on the byte-stable default
    // path).
    let mut bin_rows: Vec<Vec<usize>> = if agg_col.is_some() { vec![Vec::new(); n_bins] } else { Vec::new() };
    for (i, v) in clean.iter().enumerate() {
        if *v < lo || *v > hi { continue; }
        // Last edge is inclusive; otherwise [lo, hi) per bin.
        let pos = if *v == hi {
            n_bins - 1
        } else {
            let raw = ((*v - lo) / (hi - lo) * (n_bins as f64)).floor() as usize;
            raw.min(n_bins - 1)
        };
        counts[pos] += 1;
        if agg_col.is_some() {
            bin_rows[pos].push(row_idx[i]);
        }
    }

    // Per-bin aggregate (batch-E T7), independent of `cumulative` — see the
    // `agg` field's doc comment. A structurally empty bin (no rows landed in
    // its range) yields `None` (Arrow null), never `Aggregate`'s own
    // all-null convention (`NaN`/`0`), which only applies to a NON-empty bin
    // whose `agg_field` values are all null/NaN.
    let agg_values: Option<Vec<Option<f64>>> = agg_col.map(|ac| {
        bin_rows
            .iter()
            .map(|rows| if rows.is_empty() { None } else { Some(ac.aggregate_rows(rows)) })
            .collect()
    });

    let total = clean.len() as f64;
    let bin_starts: Vec<f64> = (0..n_bins).map(|i| edges[i]).collect();
    let bin_ends:   Vec<f64> = (0..n_bins).map(|i| edges[i + 1]).collect();
    let densities:  Vec<f64> = counts
        .iter()
        .zip(bin_starts.iter().zip(bin_ends.iter()))
        .map(|(c, (s, e))| (*c as f64) / (total * (e - s)))
        .collect();

    let (final_counts, final_densities) = if spec.cumulative {
        let mut acc_count: u64 = 0;
        let cum_counts: Vec<u64> = counts.iter().map(|c| {
            acc_count = acc_count.saturating_add(*c);
            acc_count
        }).collect();
        let mut acc_density: f64 = 0.0;
        let cum_densities: Vec<f64> = densities
            .iter()
            .zip(bin_starts.iter().zip(bin_ends.iter()))
            .map(|(d, (s, e))| {
                acc_density += d * (e - s);
                acc_density
            })
            .collect();
        (cum_counts, cum_densities)
    } else {
        (counts, densities)
    };

    Ok(BinColumns {
        starts: bin_starts,
        ends: bin_ends,
        counts: final_counts,
        densities: final_densities,
        agg_values,
    })
}

/// Partition input batch by the groupby columns, call `compute_one_group` per
/// partition, and concatenate the typed columns into a single batch with the
/// group columns appended after the 4th field.
///
/// When `spec.shared_extent` is `true` and `spec.extent` is `None`, a global
/// extent is computed across all rows of the field before per-group binning so
/// every group produces the same bin edges — required for `multiple="stack"` /
/// `"fill"` stacking to find matching x-key pairs.
fn apply_grouped(
    spec: &BinSpec,
    batch: &RecordBatch,
    group_cols: &[String],
    agg_col: Option<&AggColumn>,
    warn_ctx: &TransformContext,
) -> PyResult<RecordBatch> {
    // Multi-column partition via the shared helper (first-appearance order,
    // null-key skip, dtype-preservation — same policy as single-column path).
    refuse_groupby_output_collision(
        "stat_bin",
        group_cols,
        &bin_output_field_names(agg_col.is_some()),
    )?;
    let partition =
        group_partition_multi(batch, group_cols, "stat_bin", NullKeyPolicy::Skip, warn_ctx)?;

    // When shared_extent=true, inject the global raw extent so every group gets
    // the same bin edges (required for stack/fill). resolve_shared_extent returns
    // the explicit extent unchanged when set, else None unless shared is on.
    let effective_spec: std::borrow::Cow<BinSpec> =
        match resolve_shared_extent(spec.extent, spec.shared_extent, batch, &spec.field, group_cols) {
            Some(e) if spec.extent.is_none() => {
                let mut patched = spec.clone();
                patched.extent = Some(e);
                std::borrow::Cow::Owned(patched)
            }
            _ => std::borrow::Cow::Borrowed(spec),
        };
    let spec_ref: &BinSpec = &effective_spec;

    // Per-group typed columns are extended directly into these accumulators.
    let mut all_starts: Vec<f64> = Vec::new();
    let mut all_ends: Vec<f64> = Vec::new();
    let mut all_counts: Vec<u64> = Vec::new();
    let mut all_densities: Vec<f64> = Vec::new();
    let mut all_agg_values: Vec<Option<f64>> = Vec::new();
    let mut group_keys_out: Vec<Vec<KeyValue>> = Vec::new();
    for (g, ixs) in &partition.groups {
        let cols = compute_one_group(spec_ref, batch, Some(ixs), agg_col)?;
        let n = cols.starts.len();
        all_starts.extend(cols.starts);
        all_ends.extend(cols.ends);
        all_counts.extend(cols.counts);
        all_densities.extend(cols.densities);
        if let Some(agg_values) = cols.agg_values {
            all_agg_values.extend(agg_values);
        }
        group_keys_out.extend(std::iter::repeat_n(g.clone(), n));
    }

    build_bin_batch_grouped(
        BinColumns {
            starts: all_starts,
            ends: all_ends,
            counts: all_counts,
            densities: all_densities,
            agg_values: agg_col.is_some().then_some(all_agg_values),
        },
        group_keys_out, group_cols, &partition.dtypes,
    )
}

/// The four (or five, with `agg`) per-row output columns shared by
/// [`build_bin_batch`] and [`build_bin_batch_grouped`]. Bundled into one
/// struct — rather than 4-5 positional `Vec` params on each builder —
/// because `build_bin_batch_grouped` already carries 3 groupby params of its
/// own; adding `agg_values` as a 5th positional param would push it past
/// clippy's `too_many_arguments` threshold.
struct BinColumns {
    starts: Vec<f64>,
    ends: Vec<f64>,
    counts: Vec<u64>,
    densities: Vec<f64>,
    /// `Some`, adds a nullable `agg_value: Float64` column after `density`
    /// (batch-E T7). `None` reproduces the pre-T7 4-column schema
    /// byte-for-byte — every existing caller that never sets `agg`.
    agg_values: Option<Vec<Option<f64>>>,
}

impl BinColumns {
    /// The `bin_start`/`bin_end`/`count`/`density` fields/arrays every bin
    /// output batch carries, plus the optional `agg_value` field/array when
    /// `agg_values` is `Some`. Shared by both builders below so the 4-vs-5
    /// column decision is made in exactly one place.
    fn into_fields_and_cols(self) -> (Vec<Field>, Vec<ArrayRef>) {
        let mut fields = vec![
            Field::new("bin_start", DataType::Float64, false),
            Field::new("bin_end",   DataType::Float64, false),
            Field::new("count",     DataType::UInt64,  false),
            Field::new("density",   DataType::Float64, false),
        ];
        let mut cols: Vec<ArrayRef> = vec![
            Arc::new(Float64Array::from(self.starts)),
            Arc::new(Float64Array::from(self.ends)),
            Arc::new(UInt64Array::from(self.counts)),
            Arc::new(Float64Array::from(self.densities)),
        ];
        if let Some(agg) = self.agg_values {
            fields.push(Field::new("agg_value", DataType::Float64, true));
            cols.push(Arc::new(Float64Array::from(agg)));
        }
        (fields, cols)
    }
}

/// The output field names both bin batch builders emit, in schema order — the
/// list a `groupby` request is checked against by
/// [`refuse_groupby_output_collision`]. Derived from
/// [`BinColumns::into_fields_and_cols`] itself rather than re-spelled, so the
/// collision guard cannot fall behind the 4-vs-5 column decision.
fn bin_output_field_names(with_agg: bool) -> Vec<String> {
    let (fields, _) = BinColumns {
        starts: Vec::new(),
        ends: Vec::new(),
        counts: Vec::new(),
        densities: Vec::new(),
        agg_values: with_agg.then_some(Vec::new()),
    }
    .into_fields_and_cols();
    fields.into_iter().map(|f| f.name().clone()).collect()
}

fn build_bin_batch(cols: BinColumns) -> PyResult<RecordBatch> {
    let (fields, cols) = cols.into_fields_and_cols();
    let schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(schema, cols)
        .map_err(|e| PyValueError::new_err(format!("stat_bin: {e}")))
}

/// See [`BinColumns::agg_values`]'s doc — same optional 5th column, inserted
/// before the groupby columns so `agg_value` (when present) always sits
/// right after `density` regardless of groupby arity.
fn build_bin_batch_grouped(
    cols: BinColumns,
    group_keys: Vec<Vec<KeyValue>>,
    group_col_names: &[String],
    group_dtypes: &[DataType],
) -> PyResult<RecordBatch> {
    let (mut fields, mut cols) = cols.into_fields_and_cols();
    // FA-7: preserve original groupby dtypes (nullable for FA-9 null keys,
    // though null keys are skipped upstream for stat_bin).
    for (name, dtype) in group_col_names.iter().zip(group_dtypes.iter()) {
        fields.push(Field::new(name, dtype.clone(), groupby_field_nullable()));
    }
    for (gi, dtype) in group_dtypes.iter().enumerate() {
        let group_col = materialize_groupby_col(&group_keys, gi, dtype)
            .map_err(PyValueError::new_err)?;
        cols.push(group_col);
    }
    let schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(schema, cols)
        .map_err(|e| PyValueError::new_err(format!("stat_bin: {e}")))
}

fn empty_bin_columns(with_agg: bool) -> BinColumns {
    BinColumns {
        starts: Vec::new(),
        ends: Vec::new(),
        counts: Vec::new(),
        densities: Vec::new(),
        agg_values: with_agg.then_some(Vec::new()),
    }
}

fn single_unit_bin_columns(v: f64, count: u64, agg_value: Option<f64>) -> BinColumns {
    let start = v - 0.5;
    let end   = v + 0.5;
    let density = (count as f64) / ((count as f64) * (end - start));
    BinColumns {
        starts: vec![start],
        ends: vec![end],
        counts: vec![count],
        densities: vec![density],
        agg_values: agg_value.map(|v| vec![Some(v)]),
    }
}

use pyo3::prelude::*;

use crate::transform::core::TransformSpec;

/// Equal-width or quantile binning of a numeric column.
///
/// Discretizes a continuous field into intervals and counts values per bin.
/// Output columns: ``bin_start``, ``bin_end`` (Float64 bin edges), ``count``
/// (UInt64), and ``density`` (Float64, integrates to 1 over the range).
/// When ``cumulative=True``, ``count`` and ``density`` become running totals.
/// When ``agg_field``/``agg`` are set, the output additionally gains
/// ``agg_value`` (Float64): the per-bin aggregate of ``agg_field``.
///
/// Parameters
/// ----------
/// field : str
///     Column to bin (must be Float64).
/// bins : str, int or float, optional
///     The whole bin-count decision in one value, with the vocabulary
///     ``Bin2D(bins_x=)`` and ``Kde(count_scale=)`` already accept:
///     ``"sturges"``, ``"fd"``/``"freedman_diaconis"``, an ``int`` bin count
///     (``bool`` refused; ``numpy`` integers accepted), or a ``float`` bin
///     width. Anything else is refused by name. Mutually exclusive with
///     ``bin_count``/``bin_width`` — pass either this or one of those, not both.
/// bin_count : int, optional
///     Number of equal-width bins. Mutually exclusive with ``bin_width``.
///     When both are omitted, Sturges' rule determines bin count.
/// bin_width : float, optional
///     Fixed bin width. Mutually exclusive with ``bin_count``.
/// extent : (float, float), optional
///     ``(lo, hi)`` range to cover; data outside is clipped. Both must be
///     finite and ``lo < hi``.
/// nice : bool, default True
///     Round bin edges to visually clean numbers.
/// cumulative : bool, default False
///     When True, append a ``cumulative`` count column.
/// groupby : list[str] or str or None, optional
///     One or more group-key columns; bins computed independently per group.
///     Pass a single string or a one-element list for single-column groupby
///     (byte-identical to the old ``Option<String>`` behavior).
/// agg_field : str, optional
///     Column to aggregate per bin. Must be set together with ``agg``.
/// agg : str, optional
///     Per-bin aggregate method: one of ``"mean"``, ``"sum"``, ``"count"``,
///     ``"min"``, ``"max"``, ``"median"``, ``"variance"``, ``"stdev"``,
///     ``"q1"``, ``"q3"``, ``"distinct"`` — the same vocabulary
///     ``Aggregate`` accepts. When set, the output gains ``agg_value``:
///     the aggregate of ``agg_field``'s values over each bin's rows, with
///     ``Aggregate``'s own null/NaN-drop policy. A bin with no rows in its
///     range yields a null ``agg_value`` (not ``NaN``, not ``0``).
/// name : str, optional
///     Named output label for sibling ``Reorder(from_=...)`` lookup.
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> fm.Chart(df).mark_histogram(bin_count=10).encode(x="mpg")
///
/// Direct construction:
///
/// >>> fm.Chart(df).mark_bar().encode(
/// ...     x=fm.X("mpg_bin_start"), y="count",
/// ...     transform=fm.Bin("mpg", bin_count=10),
/// ... )
#[pyclass(eq, module = "ferrum._core", name = "Bin")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyBin(pub(crate) TransformSpec);

#[pymethods]
impl PyBin {
    #[new]
    #[pyo3(signature = (field, *, bins = None, bin_count = None, bin_width = None, extent = None, nice = true, cumulative = false, shared_extent = false, groupby = None, agg_field = None, agg = None, name = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        field: &str,
        bins: Option<&Bound<'_, PyAny>>,
        bin_count: Option<usize>,
        bin_width: Option<f64>,
        extent: Option<(f64, f64)>,
        nice: bool,
        cumulative: bool,
        shared_extent: bool,
        groupby: Option<&Bound<'_, PyAny>>,
        agg_field: Option<String>,
        agg: Option<String>,
        name: Option<String>,
    ) -> PyResult<Self> {
        if field.is_empty() {
            return Err(PyValueError::new_err("Bin: field must be non-empty"));
        }
        if let Some(c) = bin_count {
            if c == 0 {
                return Err(PyValueError::new_err("Bin: bin_count must be > 0"));
            }
        }
        if let Some(w) = bin_width {
            if !w.is_finite() || w <= 0.0 {
                return Err(PyValueError::new_err(
                    "Bin: bin_width must be a positive finite number",
                ));
            }
        }
        if let Some((a, b)) = extent {
            if !a.is_finite() || !b.is_finite() || a >= b {
                return Err(PyValueError::new_err(
                    "Bin: extent must be (lo, hi) with lo < hi and both finite",
                ));
            }
        }
        let bin_err = |e: String| PyValueError::new_err(format!("Bin: {e}"));
        check_agg_pairing(agg_field.is_some(), agg.is_some()).map_err(|e| bin_err(e.into()))?;
        let agg = agg.map(|m| AggFn::parse_or_refuse("agg", &m)).transpose().map_err(bin_err)?;
        let groupby = parse_groupby_pyany(groupby)?;
        // `bins` names the whole decision in one value — the rule-bearing
        // spelling `Bin2D(bins_x=)` and `Kde(count_scale=)` already accept,
        // parsed by the same shared owner (#193). It is mutually exclusive with
        // the two numeric params rather than layered on top of them: with a
        // precedence rule instead, `Bin(bins="fd", bin_count=10)` would silently
        // discard one of the caller's two answers.
        let mode = match bins {
            Some(obj) => {
                if bin_count.is_some() || bin_width.is_some() {
                    return Err(PyValueError::new_err(
                        "Bin: bins= is mutually exclusive with bin_count=/bin_width=; pass only one",
                    ));
                }
                crate::transform::bin_mode::parse_bin_axis(obj, "Bin")?
            }
            // Flat-param precedence, unchanged: bin_count wins → Fixed; else
            // bin_width → Width; else Sturges.
            None => {
                if let Some(n) = bin_count {
                    BinMode::Fixed { n }
                } else if let Some(w) = bin_width {
                    BinMode::Width { w }
                } else {
                    BinMode::Sturges
                }
            }
        };
        Ok(PyBin(TransformSpec::Bin(BinSpec {
            field: field.to_string(),
            mode,
            extent,
            nice,
            cumulative,
            shared_extent,
            groupby,
            agg_field,
            agg,
            name,
        })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Bin(s) => {
                // Project the typed mode back to the public (bin_count, bin_width)
                // spelling for continuity with the constructor signature.
                let (bin_count, bin_width): (Option<usize>, Option<f64>) = match &s.mode {
                    BinMode::Fixed { n } => (Some(*n), None),
                    BinMode::Width { w } => (None, Some(*w)),
                    BinMode::Sturges | BinMode::FreedmanDiaconis => (None, None),
                };
                // #193: a rule mode the numeric projection cannot spell gains a
                // trailing `bins=` term rather than printing as `bin_count=None,
                // bin_width=None`, which would repr-lie about the binning the
                // same way the wire used to. Sturges keeps printing as the
                // all-None default it has always printed as, so every
                // pre-#193 repr stays byte-identical.
                let bins_term = s
                    .mode
                    .explicit_rule_name()
                    .map(|rule| format!(", bins='{rule}'"))
                    .unwrap_or_default();
                format!(
                    "Bin(field='{}', bin_count={:?}, bin_width={:?}, extent={:?}, nice={}, cumulative={}, \
                     shared_extent={}, groupby={:?}, agg_field={:?}, agg={:?}, name={:?}{})",
                    s.field, bin_count, bin_width, s.extent,
                    if s.nice { "True" } else { "False" },
                    if s.cumulative { "True" } else { "False" },
                    if s.shared_extent { "True" } else { "False" },
                    s.groupby, s.agg_field, s.agg.map(AggFn::as_str), s.name,
                    bins_term,
                )
            }
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Float64Array, UInt64Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn batch_with(values: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
        ]));
        RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(values))]).unwrap()
    }

    fn col_f64<'a>(b: &'a RecordBatch, name: &str) -> &'a Float64Array {
        b.column(b.schema().index_of(name).unwrap())
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
    }

    fn col_u64<'a>(b: &'a RecordBatch, name: &str) -> &'a UInt64Array {
        b.column(b.schema().index_of(name).unwrap())
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
    }

    #[test]
    fn test_bin_basic_counts_match_numpy_histogram() {
        // numpy.histogram([1,2,3,4,5,6,7,8,9,10], bins=5, range=(1,10))
        // edges: [1.0, 2.8, 4.6, 6.4, 8.2, 10.0]
        // counts: [2, 2, 2, 2, 2]   (10 inclusive captured by upper-edge convention)
        let batch = batch_with(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let spec = BinSpec {
            field: "x".into(),
            mode: BinMode::Fixed { n: 5 },
            extent: Some((1.0, 10.0)),
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
            agg_field: None,
            agg: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 5);
        let counts = col_u64(&out, "count");
        for i in 0..5 {
            assert_eq!(counts.value(i), 2, "bin {i} count: got {}", counts.value(i));
        }
        let starts = col_f64(&out, "bin_start");
        let ends = col_f64(&out, "bin_end");
        for i in 0..5 {
            let expected_start = 1.0 + 1.8 * i as f64;
            let expected_end = expected_start + 1.8;
            assert!((starts.value(i) - expected_start).abs() < 1e-9);
            assert!((ends.value(i) - expected_end).abs() < 1e-9);
        }
    }

    #[test]
    fn test_bin_density_normalizes_to_one() {
        let batch = batch_with(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let spec = BinSpec {
            field: "x".into(),
            mode: BinMode::Fixed { n: 5 },
            extent: Some((1.0, 10.0)),
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
            agg_field: None,
            agg: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let densities = col_f64(&out, "density");
        let starts = col_f64(&out, "bin_start");
        let ends = col_f64(&out, "bin_end");
        let mut total: f64 = 0.0;
        for i in 0..5 {
            total += densities.value(i) * (ends.value(i) - starts.value(i));
        }
        assert!((total - 1.0).abs() < 1e-12, "density integrates to {total}");
    }

    #[test]
    fn test_bin_default_count_uses_sturges_floor() {
        // sturges_floor(8) = 4 per scale::ticks::sturges_floor
        let batch = batch_with(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let spec = BinSpec {
            field: "x".into(),
            mode: BinMode::Sturges,
            extent: None,
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
            agg_field: None,
            agg: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 4);
    }

    #[test]
    fn test_bin_all_equal_data_emits_single_unit_bin() {
        let batch = batch_with(vec![3.0, 3.0, 3.0]);
        let spec = BinSpec {
            field: "x".into(),
            mode: BinMode::Sturges,
            extent: None,
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
            agg_field: None,
            agg: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 1);
        let starts = col_f64(&out, "bin_start");
        let ends = col_f64(&out, "bin_end");
        let counts = col_u64(&out, "count");
        assert!((starts.value(0) - 2.5).abs() < 1e-12);
        assert!((ends.value(0)   - 3.5).abs() < 1e-12);
        assert_eq!(counts.value(0), 3);
    }

    #[test]
    fn test_bin_drops_nulls_and_nans() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
        ]));
        let arr = Float64Array::from(vec![Some(1.0), None, Some(2.0), Some(f64::NAN), Some(3.0)]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        let spec = BinSpec {
            field: "x".into(),
            mode: BinMode::Fixed { n: 2 },
            extent: Some((1.0, 3.0)),
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
            agg_field: None,
            agg: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let counts = col_u64(&out, "count");
        let total: u64 = (0..out.num_rows()).map(|i| counts.value(i)).sum();
        assert_eq!(total, 3, "expected 3 non-null/non-nan values");
    }

    #[test]
    fn test_bin_missing_field_errors() {
        pyo3::Python::initialize();
        let batch = batch_with(vec![1.0, 2.0, 3.0]);
        let spec = BinSpec {
            field: "ghost".into(),
            mode: BinMode::Fixed { n: 5 },
            extent: None,
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
            agg_field: None,
            agg: None,
            name: None,
        };
        let err = apply(&spec, &batch).unwrap_err();
        assert!(err.to_string().contains("ghost"), "err: {err}");
    }

    #[test]
    fn test_bin_wrong_dtype_errors() {
        pyo3::Python::initialize();
        use arrow::array::StringArray;
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec!["a", "b", "c"]))],
        ).unwrap();
        let spec = BinSpec {
            field: "x".into(),
            mode: BinMode::Fixed { n: 2 },
            extent: Some((1.0, 3.0)),
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
            agg_field: None,
            agg: None,
            name: None,
        };
        let err = apply(&spec, &batch).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("numeric") || msg.contains("Utf8"), "err: {msg}");
    }

    #[test]
    fn test_bin_int64_auto_casts_to_float64() {
        pyo3::Python::initialize();
        use arrow::array::Int64Array;
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5]))],
        ).unwrap();
        let spec = BinSpec {
            field: "x".into(),
            mode: BinMode::Fixed { n: 2 },
            extent: None,
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
            agg_field: None,
            agg: None,
            name: None,
        };
        let result = apply(&spec, &batch);
        assert!(result.is_ok(), "Int64 should auto-cast to Float64: {:?}", result.err());
    }

    #[test]
    fn test_bin_nice_extent_rounds_outward() {
        // x in [0.13, 9.7], 10 bins, nice=true → extent should round to a "nice"
        // outer bound (e.g. [0, 10] for step=1.0). The exact result depends on
        // nice_step's algorithm but lo ≤ 0.13 and hi ≥ 9.7 must hold, and
        // (hi - lo) must be a clean multiple of step.
        let batch = batch_with(vec![0.13, 1.5, 4.5, 7.7, 9.7]);
        let spec = BinSpec {
            field: "x".into(),
            mode: BinMode::Fixed { n: 10 },
            extent: None,
            nice: true,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
            agg_field: None,
            agg: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let starts = col_f64(&out, "bin_start");
        let ends   = col_f64(&out, "bin_end");
        let lo = starts.value(0);
        let hi = ends.value(out.num_rows() - 1);
        // nice_step(0.13, 9.7, 10) = 1.0 → floor(0.13/1.0)*1.0 = 0.0, ceil(9.7/1.0)*1.0 = 10.0
        assert_eq!(lo, 0.0, "nice lo should be exactly 0.0 (step=1.0 rounds 0.13 down)");
        assert_eq!(hi, 10.0, "nice hi should be exactly 10.0 (step=1.0 rounds 9.7 up)");
        // bin width = (10.0 - 0.0) / 10 bins = 1.0 exactly
        let bin_width = ends.value(0) - starts.value(0);
        assert!((bin_width - 1.0).abs() < 1e-12, "bin width should be exactly 1.0, got {bin_width}");
        // total count must equal n=5
        let counts = col_u64(&out, "count");
        let total: u64 = (0..out.num_rows()).map(|i| counts.value(i)).sum();
        assert_eq!(total, 5);
    }

    #[test]
    fn bin_cumulative_count_is_monotonic() {
        let batch = batch_with(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let spec = BinSpec {
            field: "x".into(),
            mode: BinMode::Fixed { n: 5 },
            extent: Some((1.0, 10.0)),
            nice: false,
            cumulative: true,
            shared_extent: false,
            groupby: vec![],
            agg_field: None,
            agg: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let counts = col_u64(&out, "count");
        let n = counts.len();
        for i in 1..n {
            assert!(counts.value(i) >= counts.value(i - 1),
                "cumulative count not monotonic at i={i}: {} < {}",
                counts.value(i), counts.value(i - 1));
        }
        assert_eq!(counts.value(n - 1), 10);
        // Cumulative density at the end should equal 1.0 (full CDF).
        let densities = col_f64(&out, "density");
        let last = densities.value(n - 1);
        assert!((last - 1.0).abs() < 1e-12, "cumulative density should reach 1.0, got {last}");
    }

    #[test]
    fn test_bin_single_row_produces_one_bin() {
        let batch = batch_with(vec![42.0]);
        let spec = BinSpec {
            field: "x".into(),
            mode: BinMode::Sturges,
            extent: None,
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
            agg_field: None,
            agg: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 1);
        let counts = col_u64(&out, "count");
        assert_eq!(counts.value(0), 1);
    }

    #[test]
    fn test_bin_all_identical_values_single_bin() {
        // 10 rows all with value=5.0 → should produce exactly 1 bin containing all 10.
        let batch = batch_with(vec![5.0; 10]);
        let spec = BinSpec {
            field: "x".into(),
            mode: BinMode::Fixed { n: 5 },
            extent: None,
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
            agg_field: None,
            agg: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 1);
        let counts = col_u64(&out, "count");
        assert_eq!(counts.value(0), 10);
    }

    #[test]
    fn test_bin_count_one_single_bin_spanning_full_extent() {
        let batch = batch_with(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let spec = BinSpec {
            field: "x".into(),
            mode: BinMode::Fixed { n: 1 },
            extent: Some((1.0, 5.0)),
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
            agg_field: None,
            agg: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 1);
        let counts = col_u64(&out, "count");
        assert_eq!(counts.value(0), 5);
        let starts = col_f64(&out, "bin_start");
        let ends = col_f64(&out, "bin_end");
        assert!((starts.value(0) - 1.0).abs() < 1e-12);
        assert!((ends.value(0) - 5.0).abs() < 1e-12);
    }

    // ── Task 8: global_extent helper ─────────────────────────────────────────

    /// `global_extent` returns the raw (not niced) extent when `nice=false`.
    #[test]
    fn test_bin_global_extent_no_nice() {
        let batch = batch_with(vec![1.5, 3.7, 2.2, 8.1]);
        let spec = BinSpec {
            field: "x".into(),
            mode: BinMode::Fixed { n: 4 },
            extent: None,
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
            agg_field: None,
            agg: None,
            name: None,
        };
        let ext = super::global_extent(&spec, &batch);
        let Some((lo, hi)) = ext else {
            panic!("global_extent must return Some for valid data");
        };
        assert!((lo - 1.5).abs() < 1e-12, "lo should be 1.5, got {lo}");
        assert!((hi - 8.1).abs() < 1e-12, "hi should be 8.1, got {hi}");
    }

    /// `global_extent` with `nice=true` applies the same nicing as `apply_one_group`,
    /// so the extent matches what a grouped-path call would produce for this field.
    #[test]
    fn test_bin_global_extent_niced_matches_apply_one_group() {
        // x in [0.13, 9.7], 10 bins, nice=true. The apply path produces [0.0, 10.0].
        let batch = batch_with(vec![0.13, 1.5, 4.5, 7.7, 9.7]);
        let spec = BinSpec {
            field: "x".into(),
            mode: BinMode::Fixed { n: 10 },
            extent: None,
            nice: true,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
            agg_field: None,
            agg: None,
            name: None,
        };
        let ext = super::global_extent(&spec, &batch);
        let Some((lo, hi)) = ext else {
            panic!("global_extent must return Some for valid data");
        };
        // nice_step(0.13, 9.7, 10) = 1.0 → floor(0.13/1.0)*1.0 = 0.0, ceil(9.7/1.0)*1.0 = 10.0
        assert_eq!(lo, 0.0, "niced lo should be 0.0, got {lo}");
        assert_eq!(hi, 10.0, "niced hi should be 10.0, got {hi}");

        // Also verify that running apply with this pinned extent produces the same
        // bin edges as running apply without a pinned extent (but with nice=true),
        // i.e. the niced global extent is consistent with the apply-side nicing.
        let spec_pinned = BinSpec { extent: Some((lo, hi)), nice: false, ..spec.clone() };
        let spec_auto = BinSpec { extent: None, nice: true, ..spec.clone() };
        let out_pinned = apply(&spec_pinned, &batch).unwrap();
        let out_auto = apply(&spec_auto, &batch).unwrap();
        assert_eq!(out_pinned.num_rows(), out_auto.num_rows(), "bin count must match");
        let pinned_starts = col_f64(&out_pinned, "bin_start");
        let auto_starts = col_f64(&out_auto, "bin_start");
        for i in 0..out_pinned.num_rows() {
            assert!(
                (pinned_starts.value(i) - auto_starts.value(i)).abs() < 1e-12,
                "bin_start[{i}] mismatch: pinned={}, auto={}",
                pinned_starts.value(i),
                auto_starts.value(i)
            );
        }
    }

    /// NICING-TARGET CONTRACT (#193), the faceted half: under Freedman-Diaconis
    /// the facet-wide pin `global_extent` computes and the edges a panel would
    /// compute for itself must be the SAME edges. Both ask
    /// `BinMode::nice_target_count`, which answers `None` for every
    /// rule-derived mode, so both nice against `sturges_floor(n)` and only then
    /// count via FD. Were either side to nice against its own FD count instead,
    /// the pinned range and the per-panel range would diverge and faceted
    /// panels would show bars at different positions than an unfaceted one.
    #[test]
    fn fd_niced_global_extent_matches_the_apply_path_edges() {
        pyo3::Python::initialize();
        // A spread wide enough for a non-degenerate IQR, so the FD arm actually
        // fires instead of falling back to Sturges.
        let values: Vec<f64> = (0..64).map(|i| 0.13 + (i as f64) * 0.15).collect();
        let spec = BinSpec {
            field: "x".into(),
            mode: BinMode::FreedmanDiaconis,
            extent: None,
            nice: true,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
            agg_field: None,
            agg: None,
            name: None,
        };
        let batch = batch_with(values);
        let Some((lo, hi)) = super::global_extent(&spec, &batch) else {
            panic!("global_extent must return Some for valid data");
        };
        let out_pinned = apply(&BinSpec { extent: Some((lo, hi)), nice: false, ..spec.clone() }, &batch).unwrap();
        let out_auto = apply(&spec, &batch).unwrap();
        assert_eq!(
            out_pinned.num_rows(),
            out_auto.num_rows(),
            "FD bin count must agree between the pinned and auto extents"
        );
        let pinned_starts = col_f64(&out_pinned, "bin_start");
        let auto_starts = col_f64(&out_auto, "bin_start");
        for i in 0..out_pinned.num_rows() {
            assert!(
                (pinned_starts.value(i) - auto_starts.value(i)).abs() < 1e-12,
                "FD bin_start[{i}] mismatch: pinned={}, auto={}",
                pinned_starts.value(i),
                auto_starts.value(i)
            );
        }
        // And the FD count is genuinely FD's, not the Sturges count the nicing
        // target happens to use — otherwise this test would pass on a spec that
        // never reached the FD arm at all.
        let sturges_n = crate::scale::ticks::sturges_floor(64);
        assert_ne!(
            out_auto.num_rows(),
            sturges_n,
            "fixture must discriminate: FD resolved to the Sturges count ({sturges_n})"
        );
    }

    /// `global_extent` returns the explicitly-set extent when `spec.extent` is `Some`.
    #[test]
    fn test_bin_global_extent_explicit_extent_passthrough() {
        let batch = batch_with(vec![1.0, 5.0, 10.0]);
        let spec = BinSpec {
            field: "x".into(),
            mode: BinMode::Fixed { n: 5 },
            extent: Some((0.0, 20.0)),
            nice: true,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
            agg_field: None,
            agg: None,
            name: None,
        };
        let ext = super::global_extent(&spec, &batch);
        // When extent is already set, global_extent returns it directly.
        assert_eq!(ext, Some((0.0, 20.0)), "global_extent must return the explicit extent unchanged");
    }

    /// `global_extent` returns None on missing field.
    #[test]
    fn test_bin_global_extent_missing_field_returns_none() {
        pyo3::Python::initialize();
        let batch = batch_with(vec![1.0, 2.0, 3.0]);
        let spec = BinSpec {
            field: "nonexistent".into(),
            mode: BinMode::Fixed { n: 5 },
            extent: None,
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
            agg_field: None,
            agg: None,
            name: None,
        };
        let ext = super::global_extent(&spec, &batch);
        assert_eq!(ext, None, "global_extent on missing field must return None");
    }

    /// `global_extent` handles integer columns the same way as `apply_one_group`.
    #[test]
    fn test_bin_global_extent_integer_column() {
        pyo3::Python::initialize();
        use arrow::array::Int64Array;
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![1_i64, 5, 3]))],
        ).unwrap();
        let spec = BinSpec {
            field: "x".into(),
            mode: BinMode::Fixed { n: 3 },
            extent: None,
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
            agg_field: None,
            agg: None,
            name: None,
        };
        let ext = super::global_extent(&spec, &batch);
        let Some((lo, hi)) = ext else {
            panic!("global_extent must return Some for valid Int64 data");
        };
        assert!((lo - 1.0).abs() < 1e-12, "lo should be 1.0, got {lo}");
        assert!((hi - 5.0).abs() < 1e-12, "hi should be 5.0, got {hi}");
    }

    // ── Unit tests for shared private helpers (R4 dedup) ─────────────────────

    /// `nice_extent` rounds (0.3, 9.7) outward to (0.0, 10.0) at bin_count 10.
    ///
    /// This pins the single source of the nice algorithm. The same inputs are
    /// exercised through `apply_one_group` and `global_extent` in the tests
    /// above; this test targets the helper directly so a future change to the
    /// algorithm cannot be made in one call site and silently missed in others.
    #[test]
    fn test_nice_extent_rounds_outward() {
        // nice_step(0.3, 9.7, 10) = 1.0 → floor(0.3/1.0)*1.0 = 0.0, ceil(9.7/1.0)*1.0 = 10.0
        let (lo, hi) = super::nice_extent(0.3, 9.7, 10);
        assert_eq!(lo, 0.0, "nice lo: got {lo}");
        assert_eq!(hi, 10.0, "nice hi: got {hi}");
    }

    /// `nice_extent` is a no-op when the step is degenerate (lo == hi would not
    /// reach here in practice; guard against zero step).
    #[test]
    fn test_nice_extent_degenerate_step_passthrough() {
        // Passing target=0 forces nice_step to return 0 or non-positive → passthrough.
        // We avoid calling with target=0 in production (max(1) guards it), but
        // the helper itself must not panic; it just returns the original values.
        let (lo, hi) = super::nice_extent(1.0, 2.0, 1);
        // Any finite result where lo <= 1.0 and hi >= 2.0 is acceptable.
        assert!(lo <= 1.0, "niced lo must not exceed raw lo");
        assert!(hi >= 2.0, "niced hi must not be below raw hi");
    }

    // `Bin`'s extent now delegates to `numeric_util::column_extent`; these tests
    // exercise that shared helper through the path `global_extent` takes.
    use crate::transform::numeric_util::column_extent;

    /// `column_extent` returns the min/max of a clean Float64 batch.
    #[test]
    fn test_column_extent_basic() {
        let batch = batch_with(vec![3.0, 1.0, 5.0, 2.0]);
        let ext = column_extent(&batch, "x");
        assert_eq!(ext, Some((1.0, 5.0)));
    }

    /// `column_extent` drops nulls and NaN values.
    #[test]
    fn test_column_extent_drops_nulls_and_nan() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, true)]));
        let arr = Float64Array::from(vec![Some(2.0), None, Some(f64::NAN), Some(8.0)]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        let ext = column_extent(&batch, "x");
        assert_eq!(ext, Some((2.0, 8.0)));
    }

    /// `column_extent` returns None for a missing field.
    #[test]
    fn test_column_extent_missing_field() {
        pyo3::Python::initialize();
        let batch = batch_with(vec![1.0, 2.0]);
        let ext = column_extent(&batch, "ghost");
        assert_eq!(ext, None);
    }

    /// `column_extent` returns None when lo == hi (degenerate extent).
    #[test]
    fn test_column_extent_degenerate_all_equal() {
        let batch = batch_with(vec![5.0, 5.0, 5.0]);
        let ext = column_extent(&batch, "x");
        assert_eq!(ext, None, "degenerate extent (lo==hi) must return None");
    }

    // ── T3.8a regression tests ────────────────────────────────────────────────

    /// Helper: build a batch with an `x` float column plus two string group columns.
    fn batch_with_two_groups(
        xs: Vec<f64>,
        g1s: Vec<&str>,
        g2s: Vec<&str>,
    ) -> RecordBatch {
        use arrow::array::StringArray;
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("g1", DataType::Utf8, false),
            Field::new("g2", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(xs)) as arrow::array::ArrayRef,
                Arc::new(StringArray::from(g1s)) as arrow::array::ArrayRef,
                Arc::new(StringArray::from(g2s)) as arrow::array::ArrayRef,
            ],
        )
        .unwrap()
    }

    fn col_str<'a>(b: &'a RecordBatch, name: &str) -> &'a arrow::array::StringArray {
        b.column(b.schema().index_of(name).unwrap())
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .unwrap()
    }

    /// D-GROUPBY-1 regression (T3.8a): a 2-column groupby on `Bin` must produce one
    /// set of bins per distinct `(g1, g2)` combination, with the two group columns
    /// appended after the standard 4 bin output columns.
    #[test]
    fn bin_two_column_groupby_produces_per_combination_rows() {
        // 8 rows: two g1 values × two g2 values × 2 x values each.
        let batch = batch_with_two_groups(
            vec![1.0, 9.0, 1.0, 9.0, 1.0, 9.0, 1.0, 9.0],
            vec!["A", "A", "A", "A", "B", "B", "B", "B"],
            vec!["x", "x", "y", "y", "x", "x", "y", "y"],
        );
        let spec = BinSpec {
            field: "x".into(),
            mode: BinMode::Fixed { n: 2 },
            extent: Some((0.0, 10.0)),
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: vec!["g1".into(), "g2".into()],
            agg_field: None,
            agg: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();

        // Output must have columns: bin_start, bin_end, count, density, g1, g2.
        let schema = out.schema();
        assert!(schema.index_of("bin_start").is_ok(), "missing bin_start");
        assert!(schema.index_of("bin_end").is_ok(), "missing bin_end");
        assert!(schema.index_of("count").is_ok(), "missing count");
        assert!(schema.index_of("density").is_ok(), "missing density");
        assert!(schema.index_of("g1").is_ok(), "missing g1 column");
        assert!(schema.index_of("g2").is_ok(), "missing g2 column");

        // 4 distinct groups × 2 bins each = 8 output rows.
        assert_eq!(out.num_rows(), 8, "expected 4 groups × 2 bins = 8 rows, got {}", out.num_rows());

        // Each group contributes 2 bins with count=2 (2 x-values per group).
        let counts = col_u64(&out, "count");
        for i in 0..8 {
            assert_eq!(counts.value(i), 1, "each bin should have count=1, row {i} has {}", counts.value(i));
        }

        // Verify both group columns are present and contain the right values.
        let g1 = col_str(&out, "g1");
        let g2 = col_str(&out, "g2");
        // Collect all (g1, g2) pairs seen in the output.
        let mut combos: std::collections::BTreeSet<(String, String)> =
            (0..out.num_rows())
            .map(|i| (g1.value(i).to_string(), g2.value(i).to_string()))
            .collect();
        assert_eq!(combos.len(), 4, "must have 4 distinct (g1,g2) groups in output");
        assert!(combos.contains(&("A".into(), "x".into())));
        assert!(combos.contains(&("A".into(), "y".into())));
        assert!(combos.contains(&("B".into(), "x".into())));
        assert!(combos.contains(&("B".into(), "y".into())));
    }

    /// D-GROUPBY-1 regression (T3.8a): a single-column groupby via `vec!["g"]`
    /// must produce the same output schema and row values as the old `Some("g")`
    /// path did before the migration.
    #[test]
    fn bin_single_column_groupby_matches_pre_migration_shape() {
        use arrow::array::StringArray;
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 9.0, 1.0, 9.0])) as arrow::array::ArrayRef,
                Arc::new(StringArray::from(vec!["A", "A", "B", "B"])) as arrow::array::ArrayRef,
            ],
        )
        .unwrap();

        let spec = BinSpec {
            field: "x".into(),
            mode: BinMode::Fixed { n: 2 },
            extent: Some((0.0, 10.0)),
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: vec!["g".into()],
            agg_field: None,
            agg: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();

        // Schema: bin_start, bin_end, count, density, g
        let s = out.schema();
        assert!(s.index_of("bin_start").is_ok());
        assert!(s.index_of("bin_end").is_ok());
        assert!(s.index_of("count").is_ok());
        assert!(s.index_of("density").is_ok());
        assert!(s.index_of("g").is_ok());
        assert_eq!(s.fields().len(), 5, "single-groupby output has exactly 5 columns");

        // 2 groups × 2 bins = 4 rows.
        assert_eq!(out.num_rows(), 4);

        // Each bin should have count=1.
        let counts = col_u64(&out, "count");
        for i in 0..4 {
            assert_eq!(counts.value(i), 1);
        }
    }

    /// D-GROUPBY-1 regression (T3.8a): the BinSpec serde round-trip preserves
    /// the canonical array wire form; deserializing the old bare-string form
    /// yields a one-element vec identical to the direct vec!["col"] construction.
    #[test]
    fn bin_groupby_serde_round_trip_and_back_compat() {
        // Serialize a BinSpec with groupby=["col"] — must emit array form.
        let spec = BinSpec {
            field: "x".into(),
            mode: BinMode::Sturges,
            extent: None,
            nice: true,
            cumulative: false,
            shared_extent: false,
            groupby: vec!["col".into()],
            agg_field: None,
            agg: None,
            name: None,
        };
        let json = serde_json::to_string(&spec).unwrap();
        assert!(json.contains(r#""groupby":["col"]"#), "must serialize as array, got: {json}");

        // Deserialize a legacy JSON that has a bare string for groupby.
        let legacy = r#"{"field":"x","groupby":"col"}"#;
        let from_legacy: BinSpec = serde_json::from_str(legacy).unwrap();
        assert_eq!(from_legacy.groupby, vec!["col"], "bare-string legacy must parse to [\"col\"]");

        // Deserialize null → [].
        let null_json = r#"{"field":"x","groupby":null}"#;
        let from_null: BinSpec = serde_json::from_str(null_json).unwrap();
        assert!(from_null.groupby.is_empty(), "null groupby must parse to []");

        // Empty groupby must be omitted from serialization (skip_serializing_if).
        let empty_spec = BinSpec { groupby: vec![], ..spec.clone() };
        let empty_json = serde_json::to_string(&empty_spec).unwrap();
        assert!(!empty_json.contains("groupby"), "empty groupby must be omitted: {empty_json}");
    }

    // ── T3.8b regression tests: BinWire flat-wire shim + precedence ────────────

    fn sturges_spec() -> BinSpec {
        BinSpec {
            field: "x".into(),
            mode: BinMode::Sturges,
            extent: None,
            nice: true,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
            agg_field: None,
            agg: None,
            name: None,
        }
    }

    /// `Fixed{n}` round-trips through serde as a flat `"bin_count":n` wire key and
    /// parses back to `Fixed{n}` — never as a tagged `mode` object.
    #[test]
    fn bin_fixed_mode_serializes_flat_bin_count() {
        let spec = BinSpec { mode: BinMode::Fixed { n: 5 }, ..sturges_spec() };
        let json = serde_json::to_string(&spec).unwrap();
        assert!(json.contains(r#""bin_count":5"#), "flat bin_count expected, got: {json}");
        assert!(!json.contains("mode"), "mode must not appear on the wire: {json}");
        assert!(!json.contains("kind"), "no tagged enum on the wire: {json}");
        let back: BinSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back.mode, BinMode::Fixed { n: 5 });
        assert_eq!(back, spec);
    }

    /// `Width{w}` round-trips through serde as a flat `"bin_width":w` wire key and
    /// parses back to `Width{w}`.
    #[test]
    fn bin_width_mode_serializes_flat_bin_width() {
        let spec = BinSpec { mode: BinMode::Width { w: 2.0 }, ..sturges_spec() };
        let json = serde_json::to_string(&spec).unwrap();
        assert!(json.contains(r#""bin_width":2.0"#), "flat bin_width expected, got: {json}");
        assert!(!json.contains("bin_count"), "bin_count must be omitted for Width: {json}");
        let back: BinSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back.mode, BinMode::Width { w: 2.0 });
        assert_eq!(back, spec);
    }

    /// `Sturges` (neither count nor width) emits no `bin_count`/`bin_width` keys
    /// and parses back to `Sturges`.
    #[test]
    fn bin_sturges_mode_omits_both_flat_keys() {
        let spec = sturges_spec();
        let json = serde_json::to_string(&spec).unwrap();
        assert!(!json.contains("bin_count"), "Sturges must omit bin_count: {json}");
        assert!(!json.contains("bin_width"), "Sturges must omit bin_width: {json}");
        let back: BinSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back.mode, BinMode::Sturges);
        assert_eq!(back, spec);
    }

    /// A legacy wire that sets BOTH `bin_count` and `bin_width` resolves to
    /// `Fixed` (bin_count wins) — pins the precedence-at-boundary contract.
    #[test]
    fn bin_legacy_both_keys_resolves_to_fixed() {
        let legacy = r#"{"field":"x","bin_count":7,"bin_width":3.5}"#;
        let parsed: BinSpec = serde_json::from_str(legacy).unwrap();
        assert_eq!(
            parsed.mode,
            BinMode::Fixed { n: 7 },
            "bin_count must win over bin_width at the deserialization boundary"
        );
    }

    // ── BUG-HUNT (step4): BinWire shim + degenerate wires + apply edges ─────────

    /// Round-trip every constructable 1-D BinMode through BinWire and assert the
    /// `BinSpec` is byte-identical after one serialize→deserialize cycle. The
    /// constructor only yields Sturges/Fixed/Width, so all three MUST round-trip.
    #[test]
    fn bughunt_binwire_round_trip_every_constructable_mode_is_stable() {
        for mode in [
            BinMode::Sturges,
            BinMode::Fixed { n: 1 },
            BinMode::Fixed { n: 42 },
            BinMode::Width { w: 0.5 },
            BinMode::Width { w: 100.0 },
            // #193 made FD constructable from `Bin(bins="fd")`, so it joins the
            // set this test sweeps rather than staying a documented wire hole.
            BinMode::FreedmanDiaconis,
        ] {
            let spec = BinSpec { mode: mode.clone(), ..sturges_spec() };
            let json = serde_json::to_string(&spec).unwrap();
            let back: BinSpec = serde_json::from_str(&json).unwrap();
            assert_eq!(back, spec, "BinMode {mode:?} did not round-trip byte-stably via BinWire");
            // Second cycle must be byte-identical to the first (idempotent wire).
            let json2 = serde_json::to_string(&back).unwrap();
            assert_eq!(json, json2, "second serialization differs for {mode:?}");
        }
    }

    /// Freedman-Diaconis survives the 1-D wire (#193).
    ///
    /// FLIPPED FROM ITS OPPOSITE. This test used to pin the collapse — FD
    /// serialized as neither flat numeric key, exactly like Sturges, so a
    /// serialize→deserialize cycle silently downgraded it, and its own docstring
    /// said "if a future change makes the 1-D Bin emit FD, this test will fail
    /// and force the wire to gain an `fd` flag". `Bin(bins="fd")` is that
    /// change, and `BinWire::bin_rule` is that flag. What is pinned now is the
    /// other half of that sentence: the rule reaches the wire under its own
    /// name, and reads back as itself rather than as the fallback it would have
    /// resolved to.
    #[test]
    fn binwire_freedman_diaconis_round_trips_under_its_own_name() {
        let spec = BinSpec { mode: BinMode::FreedmanDiaconis, ..sturges_spec() };
        let json = serde_json::to_string(&spec).unwrap();
        assert!(
            json.contains(r#""bin_rule":"freedman_diaconis""#),
            "FD must be spelled out on the wire, got: {json}"
        );
        assert!(!json.contains("bin_count"), "FD must not emit bin_count: {json}");
        assert!(!json.contains("bin_width"), "FD must not emit bin_width: {json}");
        let back: BinSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(
            back.mode,
            BinMode::FreedmanDiaconis,
            "FD must read back as FD, not as the Sturges it used to collapse into"
        );
    }

    /// `bin_rule` accepts the same spellings `parse_bin_axis` does (one table,
    /// `BinMode::from_rule_name`), and refuses anything else by name instead of
    /// falling back to Sturges.
    #[test]
    fn binwire_bin_rule_accepts_every_parser_spelling_and_refuses_others() {
        for (wire_name, expected) in [
            ("sturges", BinMode::Sturges),
            ("fd", BinMode::FreedmanDiaconis),
            ("freedman_diaconis", BinMode::FreedmanDiaconis),
        ] {
            let json = format!(r#"{{"field":"x","bin_rule":"{wire_name}"}}"#);
            let parsed: BinSpec = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed.mode, expected, "bin_rule '{wire_name}' misread");
        }
        let err = serde_json::from_str::<BinSpec>(r#"{"field":"x","bin_rule":"auto"}"#)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("unknown bin_rule 'auto'"),
            "an unknown rule must be refused by name, got: {err}"
        );
    }

    /// `bin_rule` names a mode outright, so pairing it with a numeric key has no
    /// single reading: the wire refuses rather than silently dropping one of the
    /// two answers (unlike the legacy bin_count/bin_width pair, whose
    /// precedence predates this key and is preserved above).
    #[test]
    fn binwire_bin_rule_with_a_numeric_key_refuses() {
        for payload in [
            r#"{"field":"x","bin_rule":"fd","bin_count":7}"#,
            r#"{"field":"x","bin_rule":"fd","bin_width":2.5}"#,
        ] {
            let err = serde_json::from_str::<BinSpec>(payload).unwrap_err().to_string();
            assert!(
                err.contains("cannot be combined with bin_count/bin_width"),
                "conflicting mode keys must refuse, got: {err} for {payload}"
            );
        }
    }

    /// A degenerate hand-authored wire with `bin_count:0` deserializes WITHOUT
    /// error (the From shim does no validation), so the zero-count must be caught
    /// at apply-time by `resolve_bin_count`, not silently coerced to a usable count.
    #[test]
    fn bughunt_wire_bin_count_zero_deserializes_then_errors_at_apply() {
        pyo3::Python::initialize();
        let spec: BinSpec = serde_json::from_str(r#"{"field":"x","bin_count":0}"#).unwrap();
        assert_eq!(spec.mode, BinMode::Fixed { n: 0 }, "wire bin_count:0 → Fixed{{0}}");
        let batch = batch_with(vec![1.0, 2.0, 3.0]);
        let err = apply(&spec, &batch).unwrap_err();
        assert!(
            err.to_string().contains("must be >= 1"),
            "Fixed{{0}} must error at apply, got: {err}"
        );
    }

    /// A degenerate hand-authored wire with `bin_width:0` must error at apply.
    #[test]
    fn bughunt_wire_bin_width_zero_errors_at_apply() {
        pyo3::Python::initialize();
        let spec: BinSpec = serde_json::from_str(r#"{"field":"x","bin_width":0}"#).unwrap();
        assert_eq!(spec.mode, BinMode::Width { w: 0.0 });
        let batch = batch_with(vec![1.0, 2.0, 3.0]);
        let err = apply(&spec, &batch).unwrap_err();
        assert!(err.to_string().contains("positive finite"), "got: {err}");
    }

    /// A degenerate hand-authored wire with a negative `bin_width` must error at
    /// apply, not produce reversed/negative-stride bin edges.
    #[test]
    fn bughunt_wire_bin_width_negative_errors_at_apply() {
        pyo3::Python::initialize();
        let spec: BinSpec = serde_json::from_str(r#"{"field":"x","bin_width":-2.5}"#).unwrap();
        assert_eq!(spec.mode, BinMode::Width { w: -2.5 });
        let batch = batch_with(vec![1.0, 2.0, 3.0]);
        let err = apply(&spec, &batch).unwrap_err();
        assert!(err.to_string().contains("positive finite"), "got: {err}");
    }

    /// A wire `bin_width:1e400` (out of f64 range) is REJECTED by serde_json at
    /// the deserialization boundary ("number out of range"), so an overflowing
    /// width can never reach `apply` as `Width{inf}`. Pins that the JSON layer is
    /// the first line of defense for out-of-range float widths. (The `inf`-reaching
    /// path is instead exercised directly in bin_mode's `resolve_width_infinity`.)
    #[test]
    fn bughunt_wire_bin_width_overflow_is_rejected_at_deserialize() {
        let r: Result<BinSpec, _> = serde_json::from_str(r#"{"field":"x","bin_width":1e400}"#);
        assert!(
            r.is_err(),
            "an out-of-range (1e400) bin_width must be rejected at deserialize, not coerced to inf"
        );
        assert!(
            r.unwrap_err().to_string().contains("out of range"),
            "the rejection must name the out-of-range cause"
        );
    }

    /// BinWire uses `deny_unknown_fields`; a wire with the OLD tagged-enum `mode`
    /// key must be rejected, proving the flat-wire contract is enforced and a
    /// stray `mode` cannot silently slip through and be ignored.
    #[test]
    fn bughunt_wire_rejects_tagged_mode_key() {
        let r: Result<BinSpec, _> =
            serde_json::from_str(r#"{"field":"x","mode":{"kind":"sturges"}}"#);
        assert!(r.is_err(), "deny_unknown_fields must reject a stray 'mode' key");
    }

    /// A bin field that is entirely null produces empty output (zero rows) with
    /// the correct 4-column schema, not a panic on the min/max fold or a div-by-0.
    #[test]
    fn bughunt_bin_all_null_column_emits_empty_output() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, true)]));
        let arr = Float64Array::from(vec![None, None, None]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        let spec = BinSpec { field: "x".into(), mode: BinMode::Fixed { n: 5 }, ..sturges_spec() };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 0, "all-null column must produce zero bins");
        assert_eq!(out.num_columns(), 4, "schema must still be the 4 bin columns");
        assert_eq!(out.schema().field(0).name(), "bin_start");
    }

    /// A zero-row column (columns exist, no rows) produces empty output, not a
    /// panic indexing `clean[0]` or folding an empty iterator into a bad extent.
    #[test]
    fn bughunt_bin_zero_row_column_emits_empty_output() {
        let batch = batch_with(vec![]);
        let spec = BinSpec { field: "x".into(), mode: BinMode::Sturges, ..sturges_spec() };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 0);
        assert_eq!(out.num_columns(), 4);
    }

    /// All-NaN column behaves like all-null: every value dropped → empty output.
    #[test]
    fn bughunt_bin_all_nan_column_emits_empty_output() {
        let batch = batch_with(vec![f64::NAN, f64::NAN]);
        let spec = BinSpec { field: "x".into(), mode: BinMode::Fixed { n: 3 }, ..sturges_spec() };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 0, "all-NaN column must produce zero bins");
    }

    /// An explicit extent with `lo >= hi` must error at apply with a legible
    /// message — the bin-edge stride `(hi-lo)/n` would otherwise be <= 0.
    #[test]
    fn bughunt_bin_reversed_explicit_extent_errors() {
        pyo3::Python::initialize();
        let batch = batch_with(vec![1.0, 2.0, 3.0]);
        let spec = BinSpec {
            field: "x".into(),
            mode: BinMode::Fixed { n: 4 },
            extent: Some((10.0, 1.0)),
            ..sturges_spec()
        };
        let err = apply(&spec, &batch).unwrap_err();
        assert!(err.to_string().contains("lo < hi"), "reversed extent must error: {err}");
    }

    /// An explicit extent where `lo == hi` must also error (not slip into the
    /// all-equal single-unit-bin branch, which is only reached when extent is None).
    #[test]
    fn bughunt_bin_equal_explicit_extent_errors() {
        pyo3::Python::initialize();
        let batch = batch_with(vec![5.0, 5.0, 5.0]);
        let spec = BinSpec {
            field: "x".into(),
            mode: BinMode::Fixed { n: 4 },
            extent: Some((5.0, 5.0)),
            ..sturges_spec()
        };
        let err = apply(&spec, &batch).unwrap_err();
        assert!(err.to_string().contains("lo < hi"), "lo==hi extent must error: {err}");
    }

    /// Off-by-one: a value sitting exactly on an interior bin edge must land in
    /// the UPPER bin (half-open `[edge, next)`), and the value exactly at `hi`
    /// must land in the LAST bin (the only inclusive edge). 2 bins over [0,10]:
    /// edge at 5.0 → bin 1; value 10.0 → bin 1; value 0.0 → bin 0.
    #[test]
    fn bughunt_bin_interior_edge_value_lands_in_upper_bin() {
        let batch = batch_with(vec![0.0, 5.0, 10.0]);
        let spec = BinSpec {
            field: "x".into(),
            mode: BinMode::Fixed { n: 2 },
            extent: Some((0.0, 10.0)),
            nice: false,
            ..sturges_spec()
        };
        let out = apply(&spec, &batch).unwrap();
        let counts = col_u64(&out, "count");
        assert_eq!(counts.value(0), 1, "bin 0 [0,5): only 0.0");
        assert_eq!(counts.value(1), 2, "bin 1 [5,10]: 5.0 (interior edge → upper) and 10.0 (hi)");
    }

    /// `i64::MAX`-magnitude data must not panic and must produce a finite,
    /// NaN-free, single-bin output (range collapses under f64 rounding) or a
    /// valid multi-bin output. Asserts no NaN/Inf leaks into bin edges.
    #[test]
    fn bughunt_bin_extreme_large_values_no_nan_in_edges() {
        let batch = batch_with(vec![1e300, 2e300, 3e300]);
        let spec = BinSpec {
            field: "x".into(),
            mode: BinMode::Fixed { n: 3 },
            extent: None,
            nice: false,
            ..sturges_spec()
        };
        let out = apply(&spec, &batch).unwrap();
        let starts = col_f64(&out, "bin_start");
        let ends = col_f64(&out, "bin_end");
        for i in 0..out.num_rows() {
            assert!(starts.value(i).is_finite(), "bin_start[{i}] not finite");
            assert!(ends.value(i).is_finite(), "bin_end[{i}] not finite");
        }
    }

    // ── BUG-HUNT (step4): groupby cardinality / mixed-dtype / non-existent col ──

    /// A groupby on a column that does not exist must error (via the shared
    /// `group_partition_multi`), naming the missing column.
    #[test]
    fn bughunt_bin_groupby_missing_column_errors() {
        pyo3::Python::initialize();
        let batch = batch_with(vec![1.0, 2.0, 3.0]);
        let spec = BinSpec {
            field: "x".into(),
            mode: BinMode::Fixed { n: 2 },
            extent: Some((0.0, 10.0)),
            groupby: vec!["ghost".into()],
            ..sturges_spec()
        };
        let err = apply(&spec, &batch).unwrap_err();
        assert!(err.to_string().contains("ghost"), "missing groupby col must error: {err}");
    }

    /// A row whose groupby key is NULL must be skipped (not assigned to any
    /// group) — internally consistent with kde/smooth via `group_partition_multi`.
    /// Build g=[A, null, A]; only the two A rows are binned, the null row dropped.
    #[test]
    fn bughunt_bin_null_group_key_row_is_skipped() {
        pyo3::Python::initialize();
        use arrow::array::StringArray;
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("g", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 9.0, 2.0])) as arrow::array::ArrayRef,
                Arc::new(StringArray::from(vec![Some("A"), None, Some("A")])) as arrow::array::ArrayRef,
            ],
        )
        .unwrap();
        let spec = BinSpec {
            field: "x".into(),
            mode: BinMode::Fixed { n: 2 },
            extent: Some((0.0, 10.0)),
            nice: false,
            groupby: vec!["g".into()],
            ..sturges_spec()
        };
        let out = apply(&spec, &batch).unwrap();
        // Only group "A" survives → 2 bins, total count 2 (rows 0 and 2; row 1's
        // null-keyed value 9.0 is dropped along with its null key).
        let counts = col_u64(&out, "count");
        let total: u64 = (0..out.num_rows()).map(|i| counts.value(i)).sum();
        assert_eq!(total, 2, "null-keyed row must be excluded from all groups, got total {total}");
        let g = col_str(&out, "g");
        for i in 0..out.num_rows() {
            assert!(!g.is_null(i), "no null-keyed group should appear, row {i}");
            assert_eq!(g.value(i), "A");
        }
    }

    /// A 3-column groupby with three DIFFERENT dtypes (Utf8 + Int64 + Boolean)
    /// must partition jointly, preserve each column's dtype in the output, and
    /// yield exactly the number of distinct (s, i, b) combinations present.
    #[test]
    fn bughunt_bin_three_mixed_dtype_groupby_cardinality_and_dtypes() {
        pyo3::Python::initialize();
        use arrow::array::{BooleanArray, Int64Array, StringArray};
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("s", DataType::Utf8, false),
            Field::new("i", DataType::Int64, false),
            Field::new("b", DataType::Boolean, false),
        ]));
        // 4 rows: (A,1,true), (A,1,true) [dup], (A,2,true), (B,1,false) → 3 distinct combos.
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 1.0, 1.0, 1.0])) as arrow::array::ArrayRef,
                Arc::new(StringArray::from(vec!["A", "A", "A", "B"])) as arrow::array::ArrayRef,
                Arc::new(Int64Array::from(vec![1_i64, 1, 2, 1])) as arrow::array::ArrayRef,
                Arc::new(BooleanArray::from(vec![true, true, true, false])) as arrow::array::ArrayRef,
            ],
        )
        .unwrap();
        let spec = BinSpec {
            field: "x".into(),
            mode: BinMode::Fixed { n: 1 },
            extent: Some((0.0, 2.0)),
            nice: false,
            groupby: vec!["s".into(), "i".into(), "b".into()],
            ..sturges_spec()
        };
        let out = apply(&spec, &batch).unwrap();
        // 3 distinct combos × 1 bin each = 3 rows.
        assert_eq!(out.num_rows(), 3, "expected 3 distinct (s,i,b) combos, got {}", out.num_rows());
        // Output dtypes preserved per column.
        let s = out.schema();
        assert_eq!(s.field(s.index_of("s").unwrap()).data_type(), &DataType::Utf8);
        assert_eq!(s.field(s.index_of("i").unwrap()).data_type(), &DataType::Int64);
        assert_eq!(s.field(s.index_of("b").unwrap()).data_type(), &DataType::Boolean);
        // The duplicate (A,1,true) combo aggregates to a single bin with count 2.
        let counts = col_u64(&out, "count");
        let total: u64 = (0..out.num_rows()).map(|i| counts.value(i)).sum();
        assert_eq!(total, 4, "all 4 rows must be counted across the 3 groups, got {total}");
    }

    /// `group_partition_multi` called with an EMPTY group_cols slice (the shape
    /// `apply` guards against but the helper itself must define): a non-empty
    /// batch collapses to ONE group with an empty composite key holding all rows.
    /// This pins the degenerate contract so a future caller bypassing the
    /// `is_empty()` guard gets a defined (not panicking) result.
    #[test]
    fn bughunt_group_partition_multi_empty_cols_single_all_rows_group() {
        pyo3::Python::initialize();
        let batch = batch_with(vec![1.0, 2.0, 3.0]);
        let p = crate::transform::group_key::group_partition_multi(
            &batch,
            &[],
            "stat_bin",
            crate::transform::group_key::NullKeyPolicy::Skip,
            &TransformContext::default(),
        )
        .unwrap();
        assert_eq!(p.dtypes.len(), 0, "no group columns → no dtypes");
        assert_eq!(p.groups.len(), 1, "non-empty batch with [] cols → one empty-key group");
        assert!(p.groups[0].0.is_empty(), "the single group's composite key is empty");
        assert_eq!(p.groups[0].1, vec![0, 1, 2], "the empty-key group holds every row");
    }

    /// `group_partition_multi` with empty cols on an EMPTY batch yields zero
    /// groups (no rows to form the empty-key group), not a phantom group.
    #[test]
    fn bughunt_group_partition_multi_empty_cols_empty_batch_zero_groups() {
        pyo3::Python::initialize();
        let batch = batch_with(vec![]);
        let p = crate::transform::group_key::group_partition_multi(
            &batch,
            &[],
            "stat_bin",
            crate::transform::group_key::NullKeyPolicy::Skip,
            &TransformContext::default(),
        )
        .unwrap();
        assert!(p.groups.is_empty(), "empty batch + [] cols → zero groups");
    }

    // ── Task 7: per-bin aggregation (agg_field/agg) ──────────────────────────

    /// Build a 2-column batch: `x` (the binned field, never null in these
    /// fixtures) and `w` (the field aggregated per bin, nullable).
    fn batch_with_agg(xs: Vec<f64>, ws: Vec<Option<f64>>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("w", DataType::Float64, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(xs)) as arrow::array::ArrayRef,
                Arc::new(Float64Array::from(ws)) as arrow::array::ArrayRef,
            ],
        )
        .unwrap()
    }

    /// `col_f64` as `Option<f64>` per row (`None` for an Arrow null), used to
    /// tell "structurally empty bin" (`None`) apart from a genuine `NaN`
    /// float value (`Some(NaN)`).
    fn col_f64_opt(b: &RecordBatch, name: &str) -> Vec<Option<f64>> {
        let arr = col_f64(b, name);
        (0..arr.len()).map(|i| if arr.is_null(i) { None } else { Some(arr.value(i)) }).collect()
    }

    fn agg_spec(field: &str, mode: BinMode, extent: Option<(f64, f64)>, agg_field: &str, agg: &str) -> BinSpec {
        BinSpec {
            field: field.into(),
            mode,
            extent,
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
            agg_field: Some(agg_field.into()),
            agg: Some(AggFn::parse(agg).expect("a declared AggFn spelling")),
            name: None,
        }
    }

    /// `mean` per bin matches hand computation: bin 0 = [1,2,3] → mean(w)=20,
    /// bin 1 = [4,5,6] → mean(w)=50.
    #[test]
    fn bin_agg_mean_matches_hand_computed_per_bin_values() {
        let batch = batch_with_agg(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![Some(10.0), Some(20.0), Some(30.0), Some(40.0), Some(50.0), Some(60.0)],
        );
        let spec = agg_spec("x", BinMode::Fixed { n: 2 }, Some((1.0, 6.0)), "w", "mean");
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 2);
        let agg = col_f64_opt(&out, "agg_value");
        assert_eq!(agg, vec![Some(20.0), Some(50.0)]);
    }

    /// `sum` per bin matches hand computation over the same partition.
    #[test]
    fn bin_agg_sum_matches_hand_computed_per_bin_values() {
        let batch = batch_with_agg(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0), Some(6.0)],
        );
        let spec = agg_spec("x", BinMode::Fixed { n: 2 }, Some((1.0, 6.0)), "w", "sum");
        let out = apply(&spec, &batch).unwrap();
        let agg = col_f64_opt(&out, "agg_value");
        assert_eq!(agg, vec![Some(6.0), Some(15.0)], "bin0=1+2+3=6, bin1=4+5+6=15");
    }

    /// `min`/`max`/`median` also match hand computation (rounds out the
    /// "mean/sum/etc." acceptance criterion beyond the two additive ops).
    #[test]
    fn bin_agg_min_max_median_match_hand_computed_values() {
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ws = vec![Some(7.0), Some(1.0), Some(4.0), Some(9.0), Some(2.0)];
        let batch = batch_with_agg(xs, ws);
        // Single bin spanning everything → min=1, max=9, median=4.
        let min_out = apply(&agg_spec("x", BinMode::Fixed { n: 1 }, Some((1.0, 5.0)), "w", "min"), &batch).unwrap();
        assert_eq!(col_f64_opt(&min_out, "agg_value"), vec![Some(1.0)]);
        let max_out = apply(&agg_spec("x", BinMode::Fixed { n: 1 }, Some((1.0, 5.0)), "w", "max"), &batch).unwrap();
        assert_eq!(col_f64_opt(&max_out, "agg_value"), vec![Some(9.0)]);
        let med_out = apply(&agg_spec("x", BinMode::Fixed { n: 1 }, Some((1.0, 5.0)), "w", "median"), &batch).unwrap();
        assert_eq!(col_f64_opt(&med_out, "agg_value"), vec![Some(4.0)]);
    }

    /// `agg_field` set without `agg` must refuse (the pairing contract).
    #[test]
    fn bin_agg_field_without_agg_errors() {
        pyo3::Python::initialize();
        let batch = batch_with_agg(vec![1.0, 2.0], vec![Some(1.0), Some(2.0)]);
        let spec = BinSpec {
            agg_field: Some("w".into()),
            agg: None,
            ..agg_spec("x", BinMode::Fixed { n: 1 }, Some((1.0, 2.0)), "w", "mean")
        };
        let err = apply(&spec, &batch).unwrap_err();
        assert!(err.to_string().contains("must both be set"), "err: {err}");
    }

    /// `agg` set without `agg_field` must refuse (the pairing contract, other
    /// direction).
    #[test]
    fn bin_agg_without_agg_field_errors() {
        pyo3::Python::initialize();
        let batch = batch_with_agg(vec![1.0, 2.0], vec![Some(1.0), Some(2.0)]);
        let spec = BinSpec {
            agg_field: None,
            agg: Some(AggFn::Mean),
            ..agg_spec("x", BinMode::Fixed { n: 1 }, Some((1.0, 2.0)), "w", "mean")
        };
        let err = apply(&spec, &batch).unwrap_err();
        assert!(err.to_string().contains("must both be set"), "err: {err}");
    }

    /// Null values in `agg_field` are dropped before aggregating, matching
    /// `Aggregate`'s own null-drop policy — a bin's mean ignores its null rows
    /// rather than treating them as 0.
    #[test]
    fn bin_agg_null_values_in_agg_field_are_dropped_like_aggregate() {
        // Bin 0 (x in [1,3)): w = [10.0, null, 30.0] → mean of {10,30} = 20, not 40/3.
        let batch = batch_with_agg(vec![1.0, 2.0, 3.0], vec![Some(10.0), None, Some(30.0)]);
        let spec = agg_spec("x", BinMode::Fixed { n: 1 }, Some((1.0, 3.0)), "w", "mean");
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(col_f64_opt(&out, "agg_value"), vec![Some(20.0)]);
    }

    /// A NON-empty bin whose `agg_field` values are ALL null yields `Aggregate`'s
    /// own all-null convention: `NaN` for `mean` (a real float, not an Arrow
    /// null) — distinct from a structurally empty bin, which yields an Arrow
    /// null (see the next test).
    #[test]
    fn bin_agg_nonempty_bin_all_null_agg_field_yields_nan_not_arrow_null() {
        let batch = batch_with_agg(vec![1.0, 2.0], vec![None, None]);
        let spec = agg_spec("x", BinMode::Fixed { n: 1 }, Some((1.0, 2.0)), "w", "mean");
        let out = apply(&spec, &batch).unwrap();
        let arr = col_f64(&out, "agg_value");
        assert!(!arr.is_null(0), "a non-empty bin's agg_value is a real (NaN) float, not an Arrow null");
        assert!(arr.value(0).is_nan(), "all-null agg_field in a non-empty bin must yield NaN");
    }

    /// A structurally empty bin (no rows fall in its range at all) yields a
    /// null `agg_value` — not `NaN`, not `0` — per the pinned edge case.
    #[test]
    fn bin_agg_structurally_empty_bin_yields_arrow_null_not_nan_or_zero() {
        // 2 fixed bins over [0, 10]: bin0=[0,5), bin1=[5,10]. All data (x=1,2)
        // falls in bin0, so bin1 has count=0 and must carry a null agg_value.
        let batch = batch_with_agg(vec![1.0, 2.0], vec![Some(100.0), Some(200.0)]);
        let spec = agg_spec("x", BinMode::Fixed { n: 2 }, Some((0.0, 10.0)), "w", "mean");
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 2);
        let counts = col_u64(&out, "count");
        assert_eq!(counts.value(1), 0, "bin1 must be structurally empty for this fixture");
        let agg = col_f64_opt(&out, "agg_value");
        assert_eq!(agg[0], Some(150.0), "bin0 mean(100,200)=150");
        assert_eq!(agg[1], None, "structurally empty bin1 must be an Arrow null, not NaN/0");
    }

    /// Grouped bins compute a per-(group, bin) aggregate, not a global one:
    /// group "A"'s bin0 mean must differ from group "B"'s bin0 mean.
    #[test]
    fn bin_agg_grouped_computes_per_group_bin_cell() {
        use arrow::array::StringArray;
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("w", DataType::Float64, true),
            Field::new("g", DataType::Utf8, false),
        ]));
        // Group A: x=[1,2], w=[10,20] → bin0 mean = 15.
        // Group B: x=[1,2], w=[100,200] → bin0 mean = 150.
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 1.0, 2.0])) as arrow::array::ArrayRef,
                Arc::new(Float64Array::from(vec![10.0, 20.0, 100.0, 200.0])) as arrow::array::ArrayRef,
                Arc::new(StringArray::from(vec!["A", "A", "B", "B"])) as arrow::array::ArrayRef,
            ],
        )
        .unwrap();
        let spec = BinSpec {
            groupby: vec!["g".into()],
            ..agg_spec("x", BinMode::Fixed { n: 1 }, Some((1.0, 2.0)), "w", "mean")
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 2, "one bin per group = 2 rows");
        let groups = col_str(&out, "g");
        let agg = col_f64_opt(&out, "agg_value");
        let a_idx = (0..out.num_rows()).find(|&i| groups.value(i) == "A").unwrap();
        let b_idx = (0..out.num_rows()).find(|&i| groups.value(i) == "B").unwrap();
        assert_eq!(agg[a_idx], Some(15.0), "group A's own bin0 mean, not global");
        assert_eq!(agg[b_idx], Some(150.0), "group B's own bin0 mean, not global");
    }

    /// Null-preservation proof: a structurally empty group-bin's `agg_value`
    /// must reach the final output as a real Arrow null, never `0.0` (the
    /// `Float64Array` null-slot placeholder) and never `Aggregate`'s own
    /// all-null convention. `bin_agg_grouped_computes_per_group_bin_cell`
    /// above uses a single bin per group, so every group-bin cell is
    /// non-empty and never exercises a structurally-empty bin. This fixture
    /// gives group "A" a structurally empty second bin (group "B" does not),
    /// so the `None` `compute_one_group` emits for that bin must survive
    /// `apply_grouped`'s typed `Vec` accumulation into the row `build_bin_batch_grouped`
    /// places it at.
    #[test]
    fn bin_agg_grouped_preserves_null_for_structurally_empty_group_bin() {
        use arrow::array::StringArray;
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("w", DataType::Float64, true),
            Field::new("g", DataType::Utf8, false),
        ]));
        // 2 bins over [0, 10]: bin0=[0,5), bin1=[5,10].
        // Group A: x=[1,2] → both land in bin0; bin1 is structurally empty.
        // Group B: x=[1,9] → one row per bin; both bins non-empty.
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 1.0, 9.0])) as arrow::array::ArrayRef,
                Arc::new(Float64Array::from(vec![10.0, 20.0, 100.0, 900.0])) as arrow::array::ArrayRef,
                Arc::new(StringArray::from(vec!["A", "A", "B", "B"])) as arrow::array::ArrayRef,
            ],
        )
        .unwrap();
        let spec = BinSpec {
            groupby: vec!["g".into()],
            ..agg_spec("x", BinMode::Fixed { n: 2 }, Some((0.0, 10.0)), "w", "mean")
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 4, "2 groups x 2 bins = 4 rows");
        let groups = col_str(&out, "g");
        let starts = col_f64(&out, "bin_start");
        let counts = col_u64(&out, "count");
        let agg = col_f64_opt(&out, "agg_value");

        let a_bin1 = (0..out.num_rows())
            .find(|&i| groups.value(i) == "A" && starts.value(i) >= 5.0)
            .expect("group A's bin1 row must exist");
        assert_eq!(counts.value(a_bin1), 0, "group A's bin1 must be structurally empty for this fixture");
        assert_eq!(
            agg[a_bin1], None,
            "group A's empty bin1 must round-trip through apply_grouped as a null, not 0.0/NaN"
        );

        let b_bin1 = (0..out.num_rows())
            .find(|&i| groups.value(i) == "B" && starts.value(i) >= 5.0)
            .expect("group B's bin1 row must exist");
        assert_eq!(counts.value(b_bin1), 1, "group B's bin1 has one row for this fixture");
        assert_eq!(agg[b_bin1], Some(900.0), "group B's non-empty bin1 must carry its real mean");
    }

    /// `agg_value` is unaffected by `cumulative` (pinned per cycle-2 review):
    /// `count`/`density` become running totals, but a per-bin aggregate like
    /// `mean` has no well-defined cumulative form, so `agg_value` stays each
    /// bin's OWN (non-cumulative) statistic even when `cumulative=true` —
    /// the contract documented on `BinSpec::agg`'s doc comment.
    #[test]
    fn bin_agg_value_ignores_cumulative_while_count_and_density_still_cumulate() {
        let batch = batch_with_agg(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![Some(10.0), Some(20.0), Some(30.0), Some(40.0), Some(50.0), Some(60.0)],
        );
        let mut spec = agg_spec("x", BinMode::Fixed { n: 2 }, Some((1.0, 6.0)), "w", "mean");
        spec.cumulative = true;
        let out = apply(&spec, &batch).unwrap();
        let counts = col_u64(&out, "count");
        // count must still be a running total under cumulative=true.
        assert_eq!(counts.value(1), 6, "cumulative count must reach the full row total by the last bin");
        assert!(counts.value(1) >= counts.value(0), "count must be monotonic under cumulative=true");
        // agg_value must NOT cumulate: bin1's mean is its own [4,5,6] mean
        // (50), not a mean folded over bin0's rows too (which would be 35).
        let agg = col_f64_opt(&out, "agg_value");
        assert_eq!(agg, vec![Some(20.0), Some(50.0)], "agg_value must stay per-bin under cumulative=true");
    }

    /// A `BinSpec` without `agg` (the default) serializes byte-identically to
    /// before T7 — no `agg_field`/`agg` keys on the wire, and the applied
    /// schema stays the 4-column shape (no `agg_value`).
    #[test]
    fn bin_agg_unset_is_byte_stable_on_wire_and_schema() {
        let spec = sturges_spec();
        let json = serde_json::to_string(&spec).unwrap();
        assert!(!json.contains("agg_field") && !json.contains("\"agg\""), "unset agg must be omitted: {json}");
        let batch = batch_with(vec![1.0, 2.0, 3.0]);
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_columns(), 4, "no agg_value column when agg is unset");
    }

    /// When `agg` is set, the wire round-trips `agg_field`/`agg` byte-for-byte.
    #[test]
    fn bin_agg_set_round_trips_on_wire() {
        let spec = agg_spec("x", BinMode::Sturges, None, "w", "sum");
        let json = serde_json::to_string(&spec).unwrap();
        assert!(json.contains(r#""agg_field":"w""#), "got: {json}");
        assert!(json.contains(r#""agg":"sum""#), "got: {json}");
        let back: BinSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back, spec);
    }

    // ── GH #179/#197: output-collision guard and the all-null warning ───────

    /// Grouping by a column named like one of `Bin`'s own output columns
    /// refuses rather than emitting two same-named fields (GH #179 residual 1:
    /// `build_bin_batch_grouped` had no guard).
    #[test]
    fn bin_groupby_colliding_with_an_output_column_refuses() {
        pyo3::Python::initialize();
        use arrow::array::StringArray;
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("count", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
                Arc::new(StringArray::from(vec!["a", "a", "b"])),
            ],
        )
        .unwrap();
        let spec = BinSpec {
            field: "x".into(),
            mode: BinMode::Fixed { n: 2 },
            extent: None,
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: vec!["count".into()],
            agg_field: None,
            agg: None,
            name: None,
        };
        let err = apply(&spec, &batch)
            .expect_err("grouping by a column named 'count' must refuse");
        assert!(err.to_string().contains("collides"), "{err}");
        assert!(err.to_string().contains("stat_bin"), "{err}");
    }

    /// GH #197: an all-null groupby column produces the empty group set and
    /// says so once through the degrade channel.
    #[test]
    fn bin_all_null_group_keys_yield_no_rows_and_warn() {
        pyo3::Python::initialize();
        use arrow::array::StringArray;
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("g", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
                Arc::new(StringArray::from(vec![None, None] as Vec<Option<&str>>)),
            ],
        )
        .unwrap();
        let spec = BinSpec {
            field: "x".into(),
            mode: BinMode::Fixed { n: 2 },
            extent: None,
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: vec!["g".into()],
            agg_field: None,
            agg: None,
            name: None,
        };
        let ctx = TransformContext::default();
        let out = apply_with_context(&spec, &batch, &ctx).unwrap();
        assert_eq!(out.num_rows(), 0);
        let warnings = ctx.warnings.borrow();
        assert_eq!(warnings.len(), 1, "got {warnings:?}");
        // "Bin", not the wire tag "stat_bin" — wave-1 close remediation
        // (`display_name_for_label`).
        assert!(warnings[0].to_string().contains("Bin"), "{:?}", warnings[0]);
    }


    /// User ruling Q4: `std`/`var` are declared aliases on `Bin(agg=)`: the
    /// alias is stored as its canonical fn (so it bins identically and writes
    /// the canonical wire spelling), and a bogus agg is refused with the
    /// `AggFn::accepted_values()` rendering at the constructor and the wire.
    #[test]
    fn bin_agg_alias_matches_canonical_and_bogus_is_refused() {
        pyo3::Python::initialize();
        let batch = batch_with_agg(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![Some(1.0), Some(4.0), Some(2.0), Some(8.0), Some(3.0), Some(9.0)],
        );
        let mode = || BinMode::Fixed { n: 2 };
        for (alias, canonical) in [("std", "stdev"), ("var", "variance")] {
            let a = agg_spec("x", mode(), Some((1.0, 6.0)), "w", alias);
            assert_eq!(a, agg_spec("x", mode(), Some((1.0, 6.0)), "w", canonical));
            let json = serde_json::to_string(&a).unwrap();
            assert!(json.contains(&format!(r#""agg":"{canonical}""#)), "{json}");
        }
        let std = apply(&agg_spec("x", mode(), Some((1.0, 6.0)), "w", "std"), &batch).unwrap();
        let var = apply(&agg_spec("x", mode(), Some((1.0, 6.0)), "w", "var"), &batch).unwrap();
        assert_ne!(col_f64_opt(&std, "agg_value"), col_f64_opt(&var, "agg_value"), "vacuity control");
        let expected = AggFn::accepted_values();
        let wire = serde_json::from_str::<BinSpec>(r#"{"field":"x","agg_field":"w","agg":"geomean"}"#)
            .unwrap_err()
            .to_string();
        assert!(wire.starts_with(&format!("stat_bin: unknown agg 'geomean'; expected {expected}")), "{wire}");
        let unpaired: BinSpec = serde_json::from_str(r#"{"field":"x","agg":"mean"}"#).unwrap();
        pyo3::Python::attach(|py| {
            let err = PyBin::new(
                "x", None, None, None, None, true, false, false, None,
                Some("w".into()), Some("geomean".into()), None,
            )
            .unwrap_err();
            assert_eq!(
                err.value(py).to_string(),
                format!("Bin: unknown agg 'geomean'; expected {expected}")
            );
            let err = apply(&unpaired, &batch).unwrap_err();
            assert_eq!(
                err.value(py).to_string(),
                "stat_bin: agg_field and agg must both be set, or both omitted"
            );
        });
    }
}
