use std::collections::HashMap;

use arrow::array::RecordBatch;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};

use crate::transform::context::TransformContext;

use crate::transform::aggregate::AggregateSpec;
use crate::transform::bin::BinSpec;
use crate::transform::bin_2d::Bin2DSpec;
use crate::transform::contour::ContourSpec;
use crate::transform::error_extent::ErrorExtentSpec;
use crate::transform::box_stats::BoxStatsSpec;
use crate::transform::kde::KdeSpec;
use crate::transform::kde_2d::Kde2DSpec;
use crate::transform::outliers::OutliersSpec;
use crate::transform::qq::QQSpec;
use crate::transform::raster::RasterSpec;
use crate::transform::hex::HexSpec;
use crate::transform::swarm::SwarmSpec;
use crate::transform::unpivot::UnpivotSpec;
use crate::transform::reorder::ReorderSpec;
use crate::transform::reference_line::ReferenceLineSpec;
use crate::transform::linkage::LinkageSpec;
use crate::transform::letter_value::LetterValueSpec;
use crate::transform::logistic::LogisticSpec;
use crate::transform::glm::GlmSpec;
use crate::transform::identity::IdentitySpec;
use crate::transform::robust::RobustSpec;
use crate::transform::smooth::SmoothSpec;
use crate::transform::summary::SummarySpec;
use crate::transform::violin::{self, ViolinSpec};

// Phase 12 data transforms
use crate::transform::filter::FilterSpec;
use crate::transform::calculate::CalculateSpec;
use crate::transform::fold::FoldSpec;
use crate::transform::pivot::PivotSpec;
use crate::transform::join_aggregate::JoinAggregateSpec;
use crate::transform::data_window::DataWindowSpec;
use crate::transform::density_data::DensityDataSpec;
use crate::transform::regression_data::RegressionDataSpec;
use crate::transform::loess_data::LoessDataSpec;
use crate::transform::impute::ImputeSpec;
use crate::transform::flatten::FlattenSpec;
use crate::transform::sample::SampleSpec;
use crate::transform::top_k::TopKSpec;
use crate::transform::data_stack::DataStackSpec;
use crate::transform::timeunit::TimeUnitSpec;
use crate::transform::data_bin::DataBinSpec;
use crate::transform::data_aggregate::DataAggregateSpec;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum TransformSpec {
    Bin(BinSpec),
    #[serde(rename = "bin_2d")]
    Bin2D(Bin2DSpec),
    Kde(KdeSpec),
    Smooth(SmoothSpec),
    Aggregate(AggregateSpec),
    Summary(SummarySpec),
    Outliers(OutliersSpec),
    ErrorExtent(ErrorExtentSpec),
    BoxStats(BoxStatsSpec),
    Violin(ViolinSpec),
    Kde2D(Kde2DSpec),
    Contour(ContourSpec),
    Qq(QQSpec),
    Linkage(LinkageSpec),
    Raster(RasterSpec),
    Hex(HexSpec),
    Swarm(SwarmSpec),
    Unpivot(UnpivotSpec),
    Reorder(ReorderSpec),
    ReferenceLine(ReferenceLineSpec),
    LetterValue(LetterValueSpec),
    Logistic(LogisticSpec),
    Glm(GlmSpec),
    Robust(RobustSpec),
    Identity(IdentitySpec),
    // Phase 12 data transforms
    Filter(FilterSpec),
    Calculate(CalculateSpec),
    Fold(FoldSpec),
    Pivot(PivotSpec),
    JoinAggregate(JoinAggregateSpec),
    DataWindow(DataWindowSpec),
    DensityData(DensityDataSpec),
    RegressionData(RegressionDataSpec),
    LoessData(LoessDataSpec),
    Impute(ImputeSpec),
    Flatten(FlattenSpec),
    Sample(SampleSpec),
    TopK(TopKSpec),
    DataStack(DataStackSpec),
    TimeUnit(TimeUnitSpec),
    DataBin(DataBinSpec),
    DataAggregate(DataAggregateSpec),
}

/// Single source of truth for the 41 `TransformSpec` variants. Dispatchers
/// across this module, `spec/chart.rs`, and `lib.rs` define a local `arm!`
/// macro and invoke `for_each_transform!(arm)`; the macro expands the table
/// into per-variant arms. Adding a transform means one edit here plus the
/// variant in the enum below — no more 5-site fan-out.
///
/// Table entries are `(Variant, module_ident, PyWrapper)` triples:
///   - `Variant`    — `TransformSpec` enum variant ident.
///   - `module_ident` — name of the per-variant submodule under
///     `crate::transform::`. An `ident` (not a `path`) so it can be
///     followed by `::apply` at the call site.
///   - `PyWrapper`  — name of the PyO3 newtype struct in that submodule
///     (e.g. `PyBin` for `Bin`). Used by `chart.rs` accessors and the
///     `lib.rs::_core` registration to project to/from Python.
///
/// Naming convention: wrapper is `Py<Variant>` for every entry. The
/// historic `PyQQ` exception was renamed to `PyQq` to match `Qq`.
///
/// **Py-wrapper coverage (SEAM-02):** the `apply`/`spec_name` dispatchers use
/// every entry's `Variant`/`module_ident`, but the `PyWrapper` column is only
/// meaningful for transforms that expose a Python class. The Phase-12 data
/// transforms are constructed exclusively through the dict-emitting
/// `transform_*` functions (`src/ferrum/transforms.py`); their typed `Data*`
/// pyclasses were never imported or exported, so they were removed. The eight
/// `Data*` entries below therefore keep their `PyWrapper` name for layout
/// symmetry but carry no live `#[pyclass]`; the Python-facing sites
/// (registration, the `ChartSpec.transforms` getter, and `coerce_transforms`)
/// drive off [`for_each_py_transform!`] instead, which lists only the variants
/// that still have a Python wrapper class.
macro_rules! for_each_transform {
    ($mac:ident) => {
        $mac! {
            Bin           => bin            : PyBin,
            Bin2D         => bin_2d         : PyBin2D,
            Kde           => kde            : PyKde,
            Smooth        => smooth         : PySmooth,
            Aggregate     => aggregate      : PyAggregate,
            Summary       => summary        : PySummary,
            Outliers      => outliers       : PyOutliers,
            ErrorExtent   => error_extent   : PyErrorExtent,
            BoxStats      => box_stats      : PyBoxStats,
            Violin        => violin         : PyViolin,
            Kde2D         => kde_2d         : PyKde2D,
            Contour       => contour        : PyContour,
            Qq            => qq             : PyQq,
            Linkage       => linkage        : PyLinkage,
            Raster        => raster         : PyRaster,
            Hex           => hex            : PyHex,
            Swarm         => swarm          : PySwarm,
            Unpivot       => unpivot        : PyUnpivot,
            Reorder       => reorder        : PyReorder,
            ReferenceLine => reference_line : PyReferenceLine,
            LetterValue   => letter_value   : PyLetterValue,
            Logistic      => logistic       : PyLogistic,
            Glm           => glm            : PyGlm,
            Robust        => robust         : PyRobust,
            Identity      => identity       : PyIdentity,
            Filter        => filter         : PyDataFilter,
            Calculate     => calculate      : PyDataCalculate,
            Fold          => fold           : PyDataFold,
            Pivot         => pivot          : PyDataPivot,
            JoinAggregate => join_aggregate : PyJoinAggregate,
            DataWindow    => data_window    : PyDataWindow,
            DensityData   => density_data   : PyDensityData,
            RegressionData => regression_data : PyRegressionData,
            LoessData     => loess_data     : PyLoessData,
            Impute        => impute         : PyImpute,
            Flatten       => flatten        : PyFlatten,
            Sample        => sample         : PySample,
            TopK          => top_k          : PyTopK,
            DataStack     => data_stack     : PyDataStack,
            TimeUnit      => timeunit       : PyTimeUnit,
            DataBin       => data_bin       : PyDataBin,
            DataAggregate => data_aggregate : PyDataAggregate,
        }
    };
}
// `for_each_transform!` is the full dispatch table (`apply`, `spec_name` use
// it directly within this module). It is also re-exported for
// `transform::registry`, which drives its per-kind metadata coverage test off
// the same table (batch-D task 1, spec §4.6) — a new variant lands here and
// in the enum together, so the registry's exhaustiveness test fails the
// moment a kind is missing a metadata decision.
pub(crate) use for_each_transform;

/// The subset of [`for_each_transform!`] whose variants still expose a Python
/// wrapper class (`Py<Variant>`). This is the table that drives the three
/// Python-facing sites: module registration (`lib.rs`), the
/// `ChartSpec.transforms` getter, and `coerce_transforms` (both `spec/chart.rs`).
///
/// The eight Phase-12 `Data*` transforms (`Filter`, `Calculate`, `Fold`,
/// `Pivot`, `DataWindow`, `DataStack`, `DataBin`, `DataAggregate`) are absent
/// here: they are constructed only via the dict-emitting `transform_*` Python
/// functions and have no typed pyclass (SEAM-02). They still round-trip through
/// the `transforms_json` serde path; the getter renders them as plain dicts via
/// the `_ =>` fallback arm, mirroring the `layers`/`coord`/`position` getters.
macro_rules! for_each_py_transform {
    ($mac:ident) => {
        $mac! {
            Bin           => bin            : PyBin,
            Bin2D         => bin_2d         : PyBin2D,
            Kde           => kde            : PyKde,
            Smooth        => smooth         : PySmooth,
            Aggregate     => aggregate      : PyAggregate,
            Summary       => summary        : PySummary,
            Outliers      => outliers       : PyOutliers,
            ErrorExtent   => error_extent   : PyErrorExtent,
            BoxStats      => box_stats      : PyBoxStats,
            Violin        => violin         : PyViolin,
            Kde2D         => kde_2d         : PyKde2D,
            Contour       => contour        : PyContour,
            Qq            => qq             : PyQq,
            Linkage       => linkage        : PyLinkage,
            Raster        => raster         : PyRaster,
            Hex           => hex            : PyHex,
            Swarm         => swarm          : PySwarm,
            Unpivot       => unpivot        : PyUnpivot,
            Reorder       => reorder        : PyReorder,
            ReferenceLine => reference_line : PyReferenceLine,
            LetterValue   => letter_value   : PyLetterValue,
            Logistic      => logistic       : PyLogistic,
            Glm           => glm            : PyGlm,
            Robust        => robust         : PyRobust,
            Identity      => identity       : PyIdentity,
            JoinAggregate => join_aggregate : PyJoinAggregate,
            DensityData   => density_data   : PyDensityData,
            RegressionData => regression_data : PyRegressionData,
            LoessData     => loess_data     : PyLoessData,
            Impute        => impute         : PyImpute,
            Flatten       => flatten        : PyFlatten,
            Sample        => sample         : PySample,
            TopK          => top_k          : PyTopK,
            TimeUnit      => timeunit       : PyTimeUnit,
        }
    };
}
pub(crate) use for_each_py_transform;

impl TransformSpec {
    pub(crate) fn apply(&self, batch: &RecordBatch) -> PyResult<RecordBatch> {
        macro_rules! arm {
            ($($V:ident => $m:ident : $py:ident,)*) => {
                match self {
                    $( Self::$V(s) => crate::transform::$m::apply(s, batch), )*
                }
            };
        }
        for_each_transform!(arm)
    }
}

impl TransformSpec {
    pub(crate) fn apply_with_context(
        &self,
        batch: &RecordBatch,
        ctx: &TransformContext,
    ) -> PyResult<RecordBatch> {
        // The macro arm enumerates the variants whose context-override has the
        // standard `apply_with_context(s, batch, ctx)` signature. Reorder has a
        // different signature (`apply_with_outputs(s, batch, &named_outputs)`),
        // so it stays explicit alongside them.
        //
        // `Smooth` was added in batch-D T9 round 3: its rank-deficiency
        // refusal needs `ctx.is_partition` to tell a genuine whole-frame call
        // apart from one facet panel of a larger frame (see
        // `smooth::apply_one_group`'s degenerate-input policy doc).
        //
        // `Robust`/`RegressionData`/`LoessData`/`Kde`/`LetterValue`/`Impute`
        // were added for the #162 degrade-warning channel: each needs
        // `ctx.warnings` (the `Rc<RefCell<Vec<TransformWarning>>>` sink) to
        // report its own degrade sites (NaN-panel fits, the gam `k` clamp,
        // `LetterValue`'s depth cap, `Impute`'s fallbacks) — see each
        // module's `apply_with_context` doc. None of them read
        // `ctx.is_partition`/`ctx.panel_pixel_size`, so adding them here does
        // not change any existing behavior; it only gives them somewhere to
        // push a warning.
        //
        // The remaining grouped transforms (`Aggregate` through
        // `JoinAggregate` below) joined for GH #197's all-null-group-key
        // warning, which the shared partitioner pushes on their behalf. Each
        // one's `apply` is now a forwarder to its own `apply_with_context`, so
        // omitting an arm here would silently drop that transform's warnings
        // and nothing else — which is exactly the silence this wave removes.
        //
        // Wave-1 close remediation (S3-2): the remaining variants have no
        // context-dependent behavior at all (no groupby, no per-group fit, no
        // named-output lookup), so they forward to `apply(batch)` unchanged —
        // but each is now spelled out explicitly instead of hiding behind a
        // `_` wildcard. A `_` arm silently drops warnings for any FUTURE
        // transform added to `TransformSpec` that needs `ctx` but whose arm a
        // human forgot to add above; spelling every remaining variant here
        // makes that omission a compile error instead — the day a new variant
        // is added to the enum, this match stops being exhaustive until an
        // arm decides, explicitly, whether it needs `ctx` or not.
        macro_rules! arm {
            ($($V:ident => $m:ident,)*) => {
                match self {
                    $( Self::$V(s) => crate::transform::$m::apply_with_context(s, batch, ctx), )*
                    Self::Reorder(s) => crate::transform::reorder::apply_with_outputs(
                        s, batch, Some(&ctx.named_outputs)),
                    Self::Bin2D(_)
                    | Self::Contour(_)
                    | Self::Qq(_)
                    | Self::Linkage(_)
                    | Self::Hex(_)
                    | Self::Unpivot(_)
                    | Self::ReferenceLine(_)
                    | Self::Logistic(_)
                    | Self::Glm(_)
                    | Self::Identity(_)
                    | Self::Filter(_)
                    | Self::Calculate(_)
                    | Self::Fold(_)
                    | Self::Flatten(_)
                    | Self::Sample(_)
                    | Self::TimeUnit(_)
                    | Self::DataBin(_) => self.apply(batch),
                }
            };
        }
        arm! {
            Raster => raster,
            Swarm  => swarm,
            Violin => violin,
            Smooth => smooth,
            Robust => robust,
            RegressionData => regression_data,
            LoessData => loess_data,
            Kde => kde,
            LetterValue => letter_value,
            Impute => impute,
            Aggregate => aggregate,
            Summary => summary,
            BoxStats => box_stats,
            ErrorExtent => error_extent,
            Bin => bin,
            Kde2D => kde_2d,
            Outliers => outliers,
            TopK => top_k,
            Pivot => pivot,
            DataAggregate => data_aggregate,
            DensityData => density_data,
            DataStack => data_stack,
            DataWindow => data_window,
            JoinAggregate => join_aggregate,
        }
    }

    /// Additional named outputs produced by this transform's `apply` invocation,
    /// alongside its primary RecordBatch output. Default: empty.
    /// Implementing this method lets a transform publish multiple named outputs
    /// (e.g. QQ publishes both points + "qq_line"; LetterValue publishes outliers
    /// and per-depth bands).
    ///
    /// `input` is the batch fed INTO this transform's `apply`; `primary` is the
    /// batch returned BY `apply`. Most transforms ignore `input`, but transforms
    /// like `LetterValue` need to classify each original row.
    pub(crate) fn secondary_outputs(
        &self,
        input: &RecordBatch,
        primary: &RecordBatch,
    ) -> PyResult<Vec<(String, RecordBatch)>> {
        // The macro arm enumerates variants whose secondary-output has the
        // standard `secondary_outputs(s, primary)` signature. LetterValue takes
        // both input and primary, so it stays explicit.
        macro_rules! arm {
            ($($V:ident => $m:ident,)*) => {
                match self {
                    $( Self::$V(s) => crate::transform::$m::secondary_outputs(s, primary), )*
                    Self::LetterValue(s) =>
                        crate::transform::letter_value::secondary_outputs(s, input, primary),
                    _ => {
                        let _ = input;
                        Ok(Vec::new())
                    }
                }
            };
        }
        arm! {
            Qq      => qq,
            Linkage => linkage,
        }
    }
}

/// Sentinel key under which the final pipeline output is always published in
/// the map returned by [`apply_transforms_named`]. Layers with `data_source: None`
/// resolve to this key.
pub(crate) const FINAL_OUTPUT_KEY: &str = "__final__";

/// Apply each transform; record named outputs with fan-out semantics.
///
/// - **Named transforms** (`name = Some(...)`) run on the CURRENT chained
///   batch (the prior unnamed output, or the original input if none). They
///   publish their output under their name but do NOT advance the chained
///   pipeline pointer — subsequent unnamed transforms see the same chained
///   input the named transform did.
/// - **Unnamed transforms** chain: each consumes the prior unnamed output.
///
/// [`FINAL_OUTPUT_KEY`] always points at the final UNNAMED-chain tail, or at
/// the original input when no unnamed transforms ran.
///
/// **Semantics note (Phase 9 finalize):** Earlier the named branch ran on the
/// ORIGINAL batch (true fan-out from the input). Phase 9's compound desugars
/// (mark_contour's Kde2D→Contour, clustermap's Linkage→Reorder→Unpivot,
/// residplot's residuals overlay) need a named transform to consume the
/// previous unnamed transform's output. Switching the named branch to use
/// `current` instead of `batch` enables those patterns without breaking
/// existing first-transform-named usages (where `current == batch` anyway).
pub(crate) fn apply_transforms_named(
    specs: &[TransformSpec],
    batch: &RecordBatch,
    ctx: &TransformContext,
) -> PyResult<HashMap<String, RecordBatch>> {
    let mut outputs: HashMap<String, RecordBatch> = HashMap::new();
    let mut current = batch.clone();
    for spec in specs {
        // Each transform sees the named outputs accumulated so far via the
        // context (Phase 9 finalize: enables Reorder(from='<named_output>')).
        let mut step_ctx = ctx.clone();
        step_ctx.named_outputs = outputs.clone();
        if let Some(name) = spec_name(spec) {
            // Named: run on the CURRENT chained batch (prior unnamed tail).
            // Does not advance the chained pipeline pointer.
            let result = spec.apply_with_context(&current, &step_ctx)?;
            // Secondary outputs first — explicit `name` (registered below)
            // wins on key collision.
            for (key, b) in spec.secondary_outputs(&current, &result)? {
                outputs.insert(key, b);
            }
            outputs.insert(name.to_string(), result);
        } else {
            // Unnamed: chained.
            let input = current.clone();
            current = spec.apply_with_context(&current, &step_ctx)?;
            for (key, b) in spec.secondary_outputs(&input, &current)? {
                outputs.insert(key, b);
            }
        }
    }
    outputs.insert(FINAL_OUTPUT_KEY.to_string(), current);
    Ok(outputs)
}

fn spec_name(spec: &TransformSpec) -> Option<&str> {
    // The module + wrapper columns of the table are unused here; we keep
    // them in the pattern so spec_name reads off the same source of truth.
    macro_rules! arm {
        ($($V:ident => $m:ident : $py:ident,)*) => {
            match spec { $( TransformSpec::$V(s) => s.name.as_deref(), )* }
        };
    }
    for_each_transform!(arm)
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::array::{Float64Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    use crate::transform::aggregate::{AggregateSpec, AggregateOp, AggFn};
    use crate::transform::bin::BinSpec;
    use crate::transform::bin_mode::BinMode;

    fn make_one_col_batch(name: &str, values: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new(name, DataType::Float64, false),
        ]));
        RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(values))]).unwrap()
    }

    #[test]
    fn test_transform_spec_linkage_round_trip() {
        use crate::transform::linkage::{LinkageSpec, LinkageMethod, DistanceMetric, LinkageAxis};
        let original = TransformSpec::Linkage(LinkageSpec {
            method: LinkageMethod::Ward, metric: DistanceMetric::Euclidean,
            axis: LinkageAxis::Rows, z_score: None, standard_scale: None, name: None,
        });
        let json = serde_json::to_string(&original).unwrap();
        assert!(json.contains(r#""type":"linkage""#));
        let parsed: TransformSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn test_transform_spec_bin_round_trip() {
        let original = TransformSpec::Bin(BinSpec {
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
        });
        let json = serde_json::to_string(&original).unwrap();
        assert!(json.contains(r#""type":"bin""#), "missing tag: {json}");
        let parsed: TransformSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn test_apply_transforms_empty_returns_input_unchanged() {
        let batch = make_one_col_batch("x", vec![1.0, 2.0, 3.0]);
        let ctx = TransformContext::default();
        let outputs = apply_transforms_named(&[], &batch, &ctx).unwrap();
        let out = &outputs[FINAL_OUTPUT_KEY];
        assert_eq!(out.num_rows(), 3);
        assert_eq!(out.num_columns(), 1);
        assert_eq!(out.schema().field(0).name(), "x");
    }

    #[test]
    fn test_pipeline_bin_then_aggregate() {
        pyo3::Python::initialize();
        // Bin produces {bin_start, bin_end, count, density}; aggregate over count by bin_start.
        let batch = make_one_col_batch("x", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let pipeline = vec![
            TransformSpec::Bin(BinSpec {
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
            }),
            TransformSpec::Aggregate(AggregateSpec {
                ops: vec![AggregateOp {
                    field: "count".into(),
                    fn_: AggFn::Sum,
                    as_: "total_count".into(),
                }],
                groupby: vec![],
                name: None,
            }),
        ];

        // Bin produces a UInt64 `count` column. Before the batch-D dtype policy
        // (spec §4.1) stat_aggregate hard-rejected anything but Float64, so this
        // pipeline failed with a schema-mismatch error; it now coerces the count
        // column and sums it. Ten input rows → total_count 10.
        let ctx = TransformContext::default();
        let outputs = apply_transforms_named(&pipeline, &batch, &ctx).unwrap();
        let out = &outputs[FINAL_OUTPUT_KEY];
        assert_eq!(out.schema().field(0).name(), "total_count");
        let totals = out
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .expect("total_count is Float64");
        assert_eq!(totals.len(), 1);
        assert_eq!(totals.value(0), 10.0);
    }

    #[test]
    fn test_pipeline_schema_mismatch_after_bin() {
        pyo3::Python::initialize();
        // After stat_bin, the input column "x" no longer exists. A follow-up aggregate
        // referring to "x" must raise PyValueError mentioning the missing column.
        let batch = make_one_col_batch("x", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let pipeline = vec![
            TransformSpec::Bin(BinSpec {
                field: "x".into(),
                mode: BinMode::Fixed { n: 3 },
                extent: Some((1.0, 5.0)),
                nice: false,
                cumulative: false,
                shared_extent: false,
                groupby: vec![],
                agg_field: None,
                agg: None,
                name: None,
            }),
            TransformSpec::Aggregate(AggregateSpec {
                ops: vec![AggregateOp {
                    field: "x".into(),
                    fn_: AggFn::Mean,
                    as_: "m".into(),
                }],
                groupby: vec![],
                name: None,
            }),
        ];
        let ctx = TransformContext::default();
        let err = apply_transforms_named(&pipeline, &batch, &ctx).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("'x'") && (msg.contains("not found") || msg.contains("missing")),
            "expected missing-column error; got: {msg}");
    }

    #[test]
    fn transform_spec_json_byte_identical_when_name_none() {
        let s = TransformSpec::Bin(BinSpec {
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
        });
        let json = serde_json::to_string(&s).unwrap();
        assert!(!json.contains("name"), "name=None must be omitted: {json}");
        assert!(json.contains(r#""type":"bin""#));
    }

    #[test]
    fn test_transform_spec_bin_2d_round_trip() {
        use crate::transform::bin_2d::Bin2DSpec;
        let original = TransformSpec::Bin2D(Bin2DSpec {
            x: "x".into(), y: "y".into(),
            bins_x: BinMode::Fixed { n: 10 },
            bins_y: BinMode::Sturges,
            extent_x: None, extent_y: None,
            cumulative: false, name: None,
        });
        let json = serde_json::to_string(&original).unwrap();
        assert!(json.contains(r#""type":"bin_2d""#), "missing tag: {json}");
        let parsed: TransformSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn test_transform_spec_reorder_round_trip() {
        use crate::transform::reorder::ReorderSpec;
        let original = TransformSpec::Reorder(ReorderSpec {
            by: "new_idx".into(), drop_index: true, from: None, name: None,
        });
        let json = serde_json::to_string(&original).unwrap();
        assert!(json.contains(r#""type":"reorder""#));
        let parsed: TransformSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn test_transform_spec_unpivot_round_trip() {
        use crate::transform::unpivot::UnpivotSpec;
        let original = TransformSpec::Unpivot(UnpivotSpec {
            id_vars: vec!["row_id".into()],
            value_vars: Some(vec!["a".into(), "b".into()]),
            var_name: "variable".into(),
            value_name: "value".into(),
            name: None,
        });
        let json = serde_json::to_string(&original).unwrap();
        assert!(json.contains(r#""type":"unpivot""#), "missing tag: {json}");
        let parsed: TransformSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn apply_with_context_default_falls_back_to_apply() {
        pyo3::Python::initialize();
        let batch = make_one_col_batch("x", vec![1.0, 2.0, 3.0]);
        let spec = TransformSpec::Bin(BinSpec {
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
        });
        let ctx = TransformContext::default();
        let with_ctx = spec.apply_with_context(&batch, &ctx).unwrap();
        let without = spec.apply(&batch).unwrap();
        assert_eq!(with_ctx.num_columns(), without.num_columns());
        assert_eq!(with_ctx.num_rows(), without.num_rows());
    }

    // ── Wave-1 close mutation-pin: dispatch-integrity test ────────────────
    //
    // The mutation-close finding (`.sdd/close-mutation-tester.md`) moved
    // `Robust` OUT of `apply_with_context`'s context-aware `arm!` invocation
    // (above) and INTO its plain-`apply(batch)` fallback list, and the whole
    // suite still passed: every per-transform test that exercises this
    // degrade calls that transform's OWN module-level `apply_with_context`
    // directly (`robust::apply_with_context(&spec, &batch, &ctx)`), never
    // `TransformSpec::apply_with_context` — so none of them notice that the
    // ENUM-LEVEL dispatch routed the real `ctx` to an ephemeral, discarded
    // `TransformContext::default()` instead (`robust::apply`'s own
    // fallback). The exhaustive match only catches an OMITTED variant, not
    // one MISFILED into the wrong bucket.
    //
    // This test closes the class, not the instance: it drives every
    // context-aware variant through `TransformSpec::apply_with_context`
    // itself (not the module function each fixture's own home-module test
    // calls), with a fixture that provokes that transform's own
    // `ctx.warnings`-reporting degrade path, and asserts the warning landed
    // in THE CALLER's context. A variant misfiled into the fallback bucket
    // cannot do that: the fallback silently substitutes its own throwaway
    // context, so the warning never reaches the context this test holds.
    #[test]
    fn context_aware_variants_route_their_own_warnings_through_enum_dispatch() {
        use crate::transform::box_stats::WhiskerExtent;
        use crate::transform::data_window::WindowOp;
        use crate::transform::error_extent::ErrorMethod;
        use crate::transform::join_aggregate::AggSpec as JoinAggSpec;
        use crate::transform::kde::{default_kernel, BandwidthSpec};
        use crate::transform::letter_value::KDepth;
        use crate::transform::smooth::{SmoothMethod, SmoothOutput};
        use crate::transform::summary::ErrorFn;
        use arrow::array::{ArrayRef, StringArray};

        pyo3::Python::initialize();

        /// A batch of same-length Float64 columns, all nullable — the shape
        /// every below-threshold single/pair-point degenerate fixture below
        /// needs.
        fn f64_batch(cols: &[(&str, Vec<f64>)]) -> RecordBatch {
            let fields: Vec<Field> = cols
                .iter()
                .map(|(name, _)| Field::new(*name, DataType::Float64, true))
                .collect();
            let arrays: Vec<ArrayRef> = cols
                .iter()
                .map(|(_, vals)| Arc::new(Float64Array::from(vals.clone())) as ArrayRef)
                .collect();
            RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap()
        }

        /// A Float64 value column paired with an all-Utf8-null groupby
        /// column `g` — the shared fixture shape for every "group-key
        /// partitioner family" case below: `NullKeyPolicy::Skip`'s
        /// `GroupKeyAllNull` degrade fires whenever every row's groupby key
        /// is null (GH #197), pushed from the one shared
        /// `group_partition_multi` call every one of these routes through.
        fn value_and_all_null_group(value_name: &str, vals: Vec<f64>) -> RecordBatch {
            let n = vals.len();
            let schema = Arc::new(Schema::new(vec![
                Field::new(value_name, DataType::Float64, true),
                Field::new("g", DataType::Utf8, true),
            ]));
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Float64Array::from(vals)),
                    Arc::new(StringArray::from(vec![None; n] as Vec<Option<&str>>)),
                ],
            )
            .unwrap()
        }

        // Canonical enumeration of the `arm!` invocation inside
        // `TransformSpec::apply_with_context` (above) — 24 variants routed
        // through the uniform `apply_with_context(s, batch, ctx)` macro
        // shape. `Reorder` is also context-aware (an explicit arm with a
        // different signature, `apply_with_outputs`) but is not named here:
        // its ctx use is `named_outputs`, an orthogonal channel from the
        // warnings-based check this test performs.
        //
        // Kept as a literal, not derived from the macro (a local, unexported
        // item) — if a variant is ever added to or removed from the `arm!`
        // invocation above, update this list too, and either add a fixture
        // below or add it to `EXCLUDED` with a reason.
        const CONTEXT_AWARE_VARIANTS: &[&str] = &[
            "Raster", "Swarm", "Violin", "Smooth", "Robust", "RegressionData", "LoessData",
            "Kde", "LetterValue", "Impute", "Aggregate", "Summary", "BoxStats", "ErrorExtent",
            "Bin", "Kde2D", "Outliers", "TopK", "Pivot", "DataAggregate", "DensityData",
            "DataStack", "DataWindow", "JoinAggregate",
        ];
        // Raster/Swarm/Violin read `ctx.panel_pixel_size` (see
        // `context.rs`'s module doc: they are its only readers), never
        // `ctx.warnings` — there is no warning-emitting degrade fixture to
        // build for them, so a "non-empty warnings vec" assertion cannot
        // discriminate a misfile of these three. Genuinely impractical for
        // THIS test's shape (a fixture built around a different ctx field
        // would need a different assertion); not silently skipped.
        const EXCLUDED: &[&str] = &["Raster", "Swarm", "Violin"];

        let mut cases: Vec<(&'static str, TransformSpec, RecordBatch)> = Vec::new();

        // ── #162 degrade-warning channel: fit/parameter/fallback degrades ──

        cases.push((
            "Robust",
            TransformSpec::Robust(RobustSpec {
                x: "x".into(), y: "y".into(),
                n_grid: 8, ci: None, huber_c: 1.345, max_iter: 50, tol: 1e-10,
                output: SmoothOutput::Fitted,
                inject_zero_ref: false, inject_metrics: false,
                groupby: vec![], x_range: None, name: None,
            }),
            f64_batch(&[("x", vec![1.0]), ("y", vec![1.0])]),
        ));

        {
            let xs: Vec<f64> = (0..60).map(|i| i as f64 * 0.15).collect();
            let ys: Vec<f64> = xs.iter().map(|x| (x * 1.4).sin() * 5.0 + 0.3 * x).collect();
            let k = xs.len() * 4;
            cases.push((
                "Smooth",
                TransformSpec::Smooth(SmoothSpec {
                    x: "x".into(), y: "y".into(),
                    method: SmoothMethod::Gam,
                    ci: None, bandwidth: 0.75, degree: 1, k, n: 20, seed: 0,
                    x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
                    inject_zero_ref: false, inject_metrics: false, x_range: None,
                    groupby: vec![], name: None,
                }),
                f64_batch(&[("x", xs), ("y", ys)]),
            ));
        }

        cases.push((
            "RegressionData",
            TransformSpec::RegressionData(RegressionDataSpec {
                x: "x".into(), y: "y".into(), method: "linear".into(), order: 1,
                groupby: None, as_: ("rx".into(), "ry".into()), name: None,
            }),
            f64_batch(&[("x", vec![5.0]), ("y", vec![9.0])]),
        ));

        cases.push((
            "LoessData",
            TransformSpec::LoessData(LoessDataSpec {
                x: "x".into(), y: "y".into(), bandwidth: 0.5,
                groupby: None, as_: ("lx".into(), "ly".into()), name: None,
            }),
            f64_batch(&[("x", vec![0.0, 1.0]), ("y", vec![5.0, 7.0])]),
        ));

        cases.push((
            "Kde",
            TransformSpec::Kde(KdeSpec {
                field: "x".into(), bandwidth: BandwidthSpec::Silverman, bw_adjust: 1.0,
                shared_extent: false, n: 5, extent: None, cumulative: false,
                kernel: default_kernel(), groupby: vec![],
                count_scale: None, count_scale_extent: None, name: None,
            }),
            f64_batch(&[("x", vec![1.0])]),
        ));

        cases.push((
            "LetterValue",
            TransformSpec::LetterValue(LetterValueSpec {
                value: "x".into(), group: None, k_depth: KDepth::Full,
                outlier_threshold: 1.5, name: None, sort: None,
            }),
            f64_batch(&[("x", (0..10_000).map(|i| i as f64).collect())]),
        ));

        {
            let schema = Arc::new(Schema::new(vec![
                Field::new("g", DataType::Utf8, false),
                Field::new("sales", DataType::Float64, true),
            ]));
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec!["a", "a", "b", "b"])),
                    Arc::new(Float64Array::from(vec![Some(100.0), Some(120.0), None, None])),
                ],
            )
            .unwrap();
            cases.push((
                "Impute",
                TransformSpec::Impute(ImputeSpec {
                    field: "sales".into(), method: "mean".into(), value: None,
                    groupby: Some(vec!["g".into()]), key: None, keyvals: None, name: None,
                }),
                batch,
            ));
        }

        // ── GH #197 group-key partitioner family: an all-null groupby
        // column degrades to "no grouping happened" and warns once, through
        // the one shared `group_partition_multi` call every one of these
        // routes through — see `group_key/mod.rs`'s module doc.

        cases.push((
            "Aggregate",
            TransformSpec::Aggregate(AggregateSpec {
                ops: vec![AggregateOp { field: "v".into(), fn_: AggFn::Sum, as_: "total".into() }],
                groupby: vec!["g".into()], name: None,
            }),
            value_and_all_null_group("v", vec![1.0, 2.0]),
        ));

        cases.push((
            "Summary",
            TransformSpec::Summary(SummarySpec {
                field: "v".into(), groupby: vec!["g".into()], error_fn: ErrorFn::Ci,
                ci: 0.95, n_boot: 100, seed: 0, name: None,
            }),
            value_and_all_null_group("v", vec![1.0, 2.0]),
        ));

        cases.push((
            "BoxStats",
            TransformSpec::BoxStats(BoxStatsSpec {
                field: "v".into(), groupby: vec!["g".into()],
                whisker_extent: WhiskerExtent::IqrMultiplier(1.5), name: None,
            }),
            value_and_all_null_group("v", vec![1.0, 2.0]),
        ));

        cases.push((
            "ErrorExtent",
            TransformSpec::ErrorExtent(ErrorExtentSpec {
                field: "v".into(), groupby: vec!["g".into()], method: ErrorMethod::Ci,
                seed: 0, n_boot: 100, name: None,
            }),
            value_and_all_null_group("v", vec![1.0, 2.0]),
        ));

        cases.push((
            "Bin",
            TransformSpec::Bin(BinSpec {
                field: "v".into(), mode: BinMode::Fixed { n: 2 }, extent: None,
                nice: false, cumulative: false, shared_extent: false,
                groupby: vec!["g".into()], agg_field: None, agg: None, name: None,
            }),
            value_and_all_null_group("v", vec![1.0, 2.0]),
        ));

        {
            let schema = Arc::new(Schema::new(vec![
                Field::new("x", DataType::Float64, true),
                Field::new("y", DataType::Float64, true),
                Field::new("g", DataType::Utf8, true),
            ]));
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Float64Array::from(vec![1.0, 2.0])),
                    Arc::new(Float64Array::from(vec![1.0, 2.0])),
                    Arc::new(StringArray::from(vec![None, None] as Vec<Option<&str>>)),
                ],
            )
            .unwrap();
            cases.push((
                "Kde2D",
                TransformSpec::Kde2D(Kde2DSpec {
                    x: "x".into(), y: "y".into(), bandwidth: BandwidthSpec::Silverman,
                    n: 3, extent: None, groupby: vec!["g".into()], name: None,
                }),
                batch,
            ));
        }

        cases.push((
            "Outliers",
            TransformSpec::Outliers(OutliersSpec {
                field: "v".into(), groupby: vec!["g".into()], extent: 1.5, name: None,
            }),
            value_and_all_null_group("v", vec![1.0, 2.0]),
        ));

        {
            let schema = Arc::new(Schema::new(vec![
                Field::new("product", DataType::Utf8, true),
                Field::new("revenue", DataType::Float64, false),
            ]));
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec![None, None] as Vec<Option<&str>>)),
                    Arc::new(Float64Array::from(vec![50.0, 80.0])),
                ],
            )
            .unwrap();
            cases.push((
                "TopK",
                TransformSpec::TopK(TopKSpec {
                    n: 3, field: "revenue".into(), op: "sum".into(), sort: "descending".into(),
                    groupby: Some(vec!["product".into()]), name: None,
                }),
                batch,
            ));
        }

        {
            let schema = Arc::new(Schema::new(vec![
                Field::new("g", DataType::Utf8, true),
                Field::new("key", DataType::Utf8, false),
                Field::new("val", DataType::Float64, false),
            ]));
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec![None, None] as Vec<Option<&str>>)),
                    Arc::new(StringArray::from(vec!["x", "x"])),
                    Arc::new(Float64Array::from(vec![1.0, 2.0])),
                ],
            )
            .unwrap();
            cases.push((
                "Pivot",
                TransformSpec::Pivot(PivotSpec {
                    field: "key".into(), value: "val".into(),
                    groupby: Some(vec!["g".into()]), limit: None, op: "sum".into(), name: None,
                }),
                batch,
            ));
        }

        cases.push((
            "DataAggregate",
            TransformSpec::DataAggregate(DataAggregateSpec {
                aggregates: vec![
                    JoinAggSpec { field: "v".into(), fn_: AggFn::Mean, as_: "m".into() },
                ],
                groupby: Some(vec!["g".into()]), name: None,
            }),
            value_and_all_null_group("v", vec![1.0, 2.0]),
        ));

        cases.push((
            "DensityData",
            TransformSpec::DensityData(DensityDataSpec {
                field: "x".into(), bandwidth: None, groupby: Some(vec!["g".into()]),
                extent: None, steps: None, cumulative: false, kernel: default_kernel(),
                as_: ("value".into(), "density".into()), name: None,
            }),
            value_and_all_null_group("x", vec![1.0, 2.0]),
        ));

        {
            let schema = Arc::new(Schema::new(vec![
                Field::new("g", DataType::Utf8, true),
                Field::new("v", DataType::Float64, false),
            ]));
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec![None, None] as Vec<Option<&str>>)),
                    Arc::new(Float64Array::from(vec![1.0, 2.0])),
                ],
            )
            .unwrap();
            cases.push((
                "DataStack",
                TransformSpec::DataStack(DataStackSpec {
                    field: "v".into(), groupby: vec!["g".into()], sort: None,
                    as_: ("y0".into(), "y1".into()), offset: "zero".into(), name: None,
                }),
                batch,
            ));
        }

        {
            let schema = Arc::new(Schema::new(vec![
                Field::new("g", DataType::Utf8, true),
                Field::new("v", DataType::Float64, false),
            ]));
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec![None, None, None] as Vec<Option<&str>>)),
                    Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
                ],
            )
            .unwrap();
            cases.push((
                "DataWindow",
                TransformSpec::DataWindow(DataWindowSpec {
                    ops: vec![WindowOp {
                        op: "row_number".into(), field: None, as_: "rn".into(), param: None,
                    }],
                    sort: None, groupby: Some(vec!["g".into()]), frame: None,
                    ignore_peers: false, name: None,
                }),
                batch,
            ));
        }

        {
            let schema = Arc::new(Schema::new(vec![
                Field::new("g", DataType::Utf8, true),
                Field::new("val", DataType::Float64, false),
            ]));
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec![None, None] as Vec<Option<&str>>)),
                    Arc::new(Float64Array::from(vec![1.0, 3.0])),
                ],
            )
            .unwrap();
            cases.push((
                "JoinAggregate",
                TransformSpec::JoinAggregate(JoinAggregateSpec {
                    aggregates: vec![
                        JoinAggSpec { field: "val".into(), fn_: AggFn::Mean, as_: "m".into() },
                    ],
                    groupby: Some(vec!["g".into()]), name: None,
                }),
                batch,
            ));
        }

        // Drift guard: every canonical context-aware variant must be either
        // fixture-covered here or explicitly excluded above with a reason —
        // catches a typo in either list, and catches a future addition to
        // the `arm!` invocation this test mirrors that nobody updated this
        // list for.
        let covered: std::collections::BTreeSet<&str> =
            cases.iter().map(|(n, _, _)| *n).collect();
        let excluded: std::collections::BTreeSet<&str> = EXCLUDED.iter().copied().collect();
        let canonical: std::collections::BTreeSet<&str> =
            CONTEXT_AWARE_VARIANTS.iter().copied().collect();
        assert!(
            covered.is_disjoint(&excluded),
            "a variant cannot be both fixture-covered and excluded: {:?}",
            covered.intersection(&excluded).collect::<Vec<_>>()
        );
        let accounted_for: std::collections::BTreeSet<&str> =
            covered.union(&excluded).copied().collect();
        assert_eq!(
            accounted_for, canonical,
            "every context-aware variant must be fixture-covered or explicitly excluded"
        );

        // The actual dispatch-integrity check: route every case through the
        // ENUM-LEVEL `TransformSpec::apply_with_context` (not the module
        // function each fixture's own home-module test calls) and confirm
        // its warning landed in THIS caller's context. A variant misfiled
        // into the plain-`apply(batch)` fallback bucket silently substitutes
        // an ephemeral, discarded context, so its warning never reaches
        // here — this assertion fails directly for exactly that variant.
        for (name, spec, batch) in &cases {
            let ctx = TransformContext::default();
            spec.apply_with_context(batch, &ctx)
                .unwrap_or_else(|e| panic!("{name}: fixture must apply cleanly, got {e}"));
            assert!(
                !ctx.warnings.borrow().is_empty(),
                "{name}: expected a degrade warning in the context passed to \
                 TransformSpec::apply_with_context — got none. Either its fixture \
                 stopped triggering the degrade, or {name} was misfiled into the \
                 plain-apply fallback bucket in apply_with_context's arm! invocation."
            );
        }
    }
}
