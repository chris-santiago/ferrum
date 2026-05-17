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
pub(crate) use for_each_transform;

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
        // Default: ignore context and forward to existing apply().
        // The macro arm enumerates the variants whose context-override has the
        // standard `apply_with_context(s, batch, ctx)` signature. Reorder has a
        // different signature (`apply_with_outputs(s, batch, &named_outputs)`),
        // so it stays explicit alongside the default fallback.
        macro_rules! arm {
            ($($V:ident => $m:ident,)*) => {
                match self {
                    $( Self::$V(s) => crate::transform::$m::apply_with_context(s, batch, ctx), )*
                    Self::Reorder(s) => crate::transform::reorder::apply_with_outputs(
                        s, batch, Some(&ctx.named_outputs)),
                    _ => self.apply(batch),
                }
            };
        }
        arm! {
            Raster => raster,
            Swarm  => swarm,
            Violin => violin,
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
            bin_count: Some(10),
            bin_width: None,
            extent: None,
            nice: true,
            cumulative: false,
            shared_extent: false,
            groupby: None,
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
                bin_count: Some(5),
                bin_width: None,
                extent: Some((1.0, 10.0)),
                nice: false,
                cumulative: false,
                shared_extent: false,
                groupby: None,
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

        // bin produces UInt64 count, but stat_aggregate requires Float64 for op fields.
        // The pipeline is expected to fail with a clear schema-mismatch error from stat_aggregate.
        let ctx = TransformContext::default();
        let err = apply_transforms_named(&pipeline, &batch, &ctx).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Float64") || msg.contains("dtype"),
            "expected dtype error from stat_aggregate; got: {msg}");
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
                bin_count: Some(3),
                bin_width: None,
                extent: Some((1.0, 5.0)),
                nice: false,
                cumulative: false,
                shared_extent: false,
                groupby: None,
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
            bin_count: Some(10),
            bin_width: None,
            extent: None,
            nice: true,
            cumulative: false,
            shared_extent: false,
            groupby: None,
            name: None,
        });
        let json = serde_json::to_string(&s).unwrap();
        assert!(!json.contains("name"), "name=None must be omitted: {json}");
        assert!(json.contains(r#""type":"bin""#));
    }

    #[test]
    fn test_transform_spec_bin_2d_round_trip() {
        use crate::transform::bin_2d::{Bin2DSpec, BinSpec2DAxis};
        let original = TransformSpec::Bin2D(Bin2DSpec {
            x: "x".into(), y: "y".into(),
            bins_x: BinSpec2DAxis::Fixed { n: 10 },
            bins_y: BinSpec2DAxis::Sturges,
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
            bin_count: Some(2),
            bin_width: None,
            extent: Some((1.0, 3.0)),
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: None,
            name: None,
        });
        let ctx = TransformContext::default();
        let with_ctx = spec.apply_with_context(&batch, &ctx).unwrap();
        let without = spec.apply(&batch).unwrap();
        assert_eq!(with_ctx.num_columns(), without.num_columns());
        assert_eq!(with_ctx.num_rows(), without.num_rows());
    }
}
