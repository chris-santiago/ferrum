use std::collections::HashMap;

use arrow::array::RecordBatch;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};

use crate::transform::context::TransformContext;

use crate::transform::aggregate::AggregateSpec;
use crate::transform::bin::{self, BinSpec};
use crate::transform::bin_2d::{self, Bin2DSpec};
use crate::transform::contour::{self, ContourSpec};
use crate::transform::error_extent::{self, ErrorExtentSpec};
use crate::transform::box_stats::{self, BoxStatsSpec};
use crate::transform::kde::KdeSpec;
use crate::transform::kde_2d::Kde2DSpec;
use crate::transform::outliers::{self, OutliersSpec};
use crate::transform::qq::{self, QQSpec};
use crate::transform::raster::{self, RasterSpec};
use crate::transform::hex::{self, HexSpec};
use crate::transform::swarm::{self, SwarmSpec};
use crate::transform::unpivot::{self, UnpivotSpec};
use crate::transform::reorder::{self, ReorderSpec};
use crate::transform::linkage::{self, LinkageSpec};
use crate::transform::letter_value::{self, LetterValueSpec};
use crate::transform::logistic::{self, LogisticSpec};
use crate::transform::glm::{self, GlmSpec};
use crate::transform::robust::{self, RobustSpec};
use crate::transform::smooth::SmoothSpec;
use crate::transform::summary::SummarySpec;
use crate::transform::violin::{self, ViolinSpec};

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
    LetterValue(LetterValueSpec),
    Logistic(LogisticSpec),
    Glm(GlmSpec),
    Robust(RobustSpec),
}

impl TransformSpec {
    pub(crate) fn apply(&self, batch: &RecordBatch) -> PyResult<RecordBatch> {
        match self {
            Self::Bin(s)       => bin::apply(s, batch),
            Self::Bin2D(s)     => bin_2d::apply(s, batch),
            Self::Kde(s)       => crate::transform::kde::apply(s, batch),
            Self::Smooth(s)    => crate::transform::smooth::apply(s, batch),
            Self::Aggregate(s) => crate::transform::aggregate::apply(s, batch),
            Self::Summary(s)   => crate::transform::summary::apply(s, batch),
            Self::Outliers(s)  => outliers::apply(s, batch),
            Self::ErrorExtent(s) => error_extent::apply(s, batch),
            Self::BoxStats(s)  => box_stats::apply(s, batch),
            Self::Violin(s)    => violin::apply(s, batch),
            Self::Kde2D(s)     => crate::transform::kde_2d::apply(s, batch),
            Self::Contour(s)   => contour::apply(s, batch),
            Self::Qq(s)        => qq::apply(s, batch),
            Self::Linkage(s)   => linkage::apply(s, batch),
            Self::Raster(s)    => raster::apply(s, batch),
            Self::Hex(s)       => hex::apply(s, batch),
            Self::Swarm(s)     => swarm::apply(s, batch),
            Self::Unpivot(s)   => unpivot::apply(s, batch),
            Self::Reorder(s)   => reorder::apply(s, batch),
            Self::LetterValue(s) => letter_value::apply(s, batch),
            Self::Logistic(s) => logistic::apply(s, batch),
            Self::Glm(s)      => glm::apply(s, batch),
            Self::Robust(s)   => robust::apply(s, batch),
        }
    }
}

pub(crate) fn apply_transforms(
    specs: &[TransformSpec],
    batch: &RecordBatch,
) -> PyResult<RecordBatch> {
    let mut current = batch.clone(); // Arrow Arc-clones; cheap
    for spec in specs {
        current = spec.apply(&current)?;
    }
    Ok(current)
}

impl TransformSpec {
    pub(crate) fn apply_with_context(
        &self,
        batch: &RecordBatch,
        ctx: &TransformContext,
    ) -> PyResult<RecordBatch> {
        // Default: ignore context and forward to existing apply().
        // Phase 8b transforms that NEED context (Raster, Swarm) override here.
        // Phase 9 finalize: Reorder optionally reads its index column from a
        // sibling named output via ctx.named_outputs.
        match self {
            Self::Raster(s) => crate::transform::raster::apply_with_context(s, batch, ctx),
            Self::Swarm(s) => crate::transform::swarm::apply_with_context(s, batch, ctx),
            Self::Violin(s) => crate::transform::violin::apply_with_context(s, batch, ctx),
            Self::Reorder(s) =>
                crate::transform::reorder::apply_with_outputs(s, batch, Some(&ctx.named_outputs)),
            _ => self.apply(batch),
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
        match self {
            Self::Qq(s) => crate::transform::qq::secondary_outputs(s, primary),
            Self::Linkage(s) => crate::transform::linkage::secondary_outputs(s, primary),
            Self::LetterValue(s) => crate::transform::letter_value::secondary_outputs(s, input, primary),
            _ => {
                let _ = input;
                Ok(Vec::new())
            }
        }
    }
}

pub(crate) fn apply_transforms_with_context(
    specs: &[TransformSpec],
    batch: &RecordBatch,
    ctx: &TransformContext,
) -> PyResult<RecordBatch> {
    let mut current = batch.clone();
    for spec in specs {
        current = spec.apply_with_context(&current, ctx)?;
    }
    Ok(current)
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
    match spec {
        TransformSpec::Bin(s) => s.name.as_deref(),
        TransformSpec::Bin2D(s) => s.name.as_deref(),
        TransformSpec::Kde(s) => s.name.as_deref(),
        TransformSpec::Smooth(s) => s.name.as_deref(),
        TransformSpec::Aggregate(s) => s.name.as_deref(),
        TransformSpec::Summary(s) => s.name.as_deref(),
        TransformSpec::Outliers(s) => s.name.as_deref(),
        TransformSpec::ErrorExtent(s) => s.name.as_deref(),
        TransformSpec::BoxStats(s) => s.name.as_deref(),
        TransformSpec::Violin(s) => s.name.as_deref(),
        TransformSpec::Kde2D(s) => s.name.as_deref(),
        TransformSpec::Contour(s) => s.name.as_deref(),
        TransformSpec::Qq(s) => s.name.as_deref(),
        TransformSpec::Linkage(s) => s.name.as_deref(),
        TransformSpec::Raster(s) => s.name.as_deref(),
        TransformSpec::Hex(s) => s.name.as_deref(),
        TransformSpec::Swarm(s) => s.name.as_deref(),
        TransformSpec::Unpivot(s) => s.name.as_deref(),
        TransformSpec::Reorder(s) => s.name.as_deref(),
        TransformSpec::LetterValue(s) => s.name.as_deref(),
        TransformSpec::Logistic(s) => s.name.as_deref(),
        TransformSpec::Glm(s) => s.name.as_deref(),
        TransformSpec::Robust(s) => s.name.as_deref(),
    }
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
        let out = apply_transforms(&[], &batch).unwrap();
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
        let err = apply_transforms(&pipeline, &batch).unwrap_err();
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
        let err = apply_transforms(&pipeline, &batch).unwrap_err();
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
