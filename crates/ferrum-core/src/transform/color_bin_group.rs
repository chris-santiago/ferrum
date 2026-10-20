//! Statistics under a binned color group by its bins.
//!
//! A binned color (`Color(bin=...)`) leaves its column unchanged and maps it
//! through a bin-ordinal scale. A statistic lowered from a layer's encoding
//! or mark that groups by that column would otherwise make one group per raw
//! value, so several marks of one bin would stack, dodge or overlap as
//! separate groups. Under a [`ColorBinPlan`] the steps a binned-color layer
//! reads group that column by bin: the shared partitioner
//! ([`crate::transform::group_key::group_partition_multi`]) keys each finite
//! value by its bin's start under the color scale's edges, and the step
//! writes that start under the column's own name. The step reads the raw
//! values as written, so a histogram of `v` colored by `v`'s bins still
//! counts each row at its own position.
//!
//! **Which steps.** The lowered steps (those after their chain's statistics
//! marker, [`is_statistics_marker`]: the steps a layer's encoding or mark
//! lowered to, never the user's own `.transform()` steps, which run as
//! written) that group by the binned column, among the steps the rows a
//! binned-color layer reads passed: the run records them in each output's
//! lineage ([`crate::transform::run::OutputLineage`]), through every branch
//! that read those rows by `input` ([`BinDemand`]).
//!
//! **Edges.** The edges are cut from the rows the binned statistics consume,
//! so the render runs the pipeline twice. The first run groups by raw values
//! and records, in each output's lineage, the steps its rows passed and the
//! rows its statistics consumed; the color scale cuts its edges from those
//! rows ([`crate::transform::run::OutputLineage::binned_rows`]), and the
//! second run groups the demanded steps by them. The consumed rows precede
//! every lowered step, so both runs record the same ones.

use std::collections::HashMap;
use std::rc::Rc;

use crate::transform::core::TransformSpec;
use crate::transform::run::{OutputLineages, StepId};

/// Whether `spec` is a chain's statistics marker: an identity step with
/// `statistics: true` (wire `{"type": "identity", "statistics": true}`).
/// The steps after the first marker in a chain are lowered from layers'
/// encodings and marks; the steps before it are the user's own.
pub(crate) fn is_statistics_marker(spec: &TransformSpec) -> bool {
    matches!(spec, TransformSpec::Identity(identity) if identity.statistics)
}

/// The binned columns each lowered step groups by bin, by step id.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct BinDemand {
    steps: HashMap<StepId, Vec<String>>,
}

impl BinDemand {
    /// The demand `reads` (one `(key, field)` per binned binding of a
    /// reading layer: the key it draws, [`crate::transform::run::FINAL_OUTPUT_KEY`]
    /// for a layer with none) place on a run whose outputs' lineages are
    /// `lineages`: every statistic of the key's own that groups by the
    /// field ([`crate::transform::run::OutputLineage::grouping`]), through
    /// every branch that read its rows by `input`, never a step whose rows
    /// a named or secondary output only shares with its chain's tail. A key
    /// no output carries demands nothing: the render refuses the layer
    /// reading it.
    pub(crate) fn of(lineages: &OutputLineages, reads: &[(&str, &str)]) -> Self {
        let mut demand = Self::default();
        for (key, field) in reads {
            let Some(lineage) = lineages.get(*key) else { continue };
            for step in lineage.grouping(field) {
                let fields = demand.steps.entry(step.id.clone()).or_default();
                if !fields.iter().any(|existing| existing == field) {
                    fields.push(field.to_string());
                }
            }
        }
        demand
    }

    /// Whether any step bins a column.
    pub(crate) fn any(&self) -> bool {
        !self.steps.is_empty()
    }

    /// The columns step `id` bins.
    pub(crate) fn fields(&self, id: &[usize]) -> &[String] {
        self.steps.get(id).map_or(&[], Vec::as_slice)
    }
}

/// A run's binned-color grouping: the demanded steps and the color scale's
/// edges they group by.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ColorBinPlan {
    pub(crate) demand: BinDemand,
    pub(crate) edges: Option<Rc<[f64]>>,
}

/// The binned columns one step groups by bin and the edges they are cut by,
/// carried by that step's
/// [`crate::transform::context::TransformContext::color_bins`].
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StepBins {
    pub(crate) fields: Vec<String>,
    pub(crate) edges: Option<Rc<[f64]>>,
}

impl StepBins {
    /// The bin start a finite `value` of groupby column `field` keys by;
    /// `None` when this step does not bin `field` (or has no edges yet) or
    /// `value` is not finite, which keeps its raw key.
    pub(crate) fn bin_start(&self, field: &str, value: f64) -> Option<f64> {
        if !self.bins(field) {
            return None;
        }
        let edges = self.edges.as_deref()?;
        crate::transform::bin::edge_bucket(edges, value).and_then(|bin| edges.get(bin).copied())
    }

    /// Whether this step groups `field` by bin.
    pub(crate) fn bins(&self, field: &str) -> bool {
        self.edges.is_some() && self.fields.iter().any(|binned| binned == field)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::run::{apply_transforms_named, FINAL_OUTPUT_KEY};

    fn steps(json: serde_json::Value) -> Vec<TransformSpec> {
        serde_json::from_value(json).expect("transforms deserialize")
    }

    /// The mean of `y` by `v`, kept as `y`, so steps of it chain.
    fn agg(name: Option<&str>) -> serde_json::Value {
        let mut step = serde_json::json!({"type": "aggregate", "ops": [{"field": "y", "fn": "mean", "as": "y"}], "groupby": ["v"]});
        if let Some(name) = name {
            step["name"] = serde_json::json!(name);
        }
        step
    }

    fn marker() -> serde_json::Value {
        serde_json::json!({"type": "identity", "statistics": true})
    }

    /// A filter, unnamed, as a user writes it.
    fn keep_all() -> serde_json::Value {
        serde_json::json!({"type": "filter", "predicate": "true"})
    }

    /// A branch named `name` over `input` (the original rows when `None`).
    fn branch(name: &str, input: Option<&str>, transforms: serde_json::Value) -> serde_json::Value {
        serde_json::json!({"type": "branch", "name": name, "input": input, "transforms": transforms})
    }

    /// The demand `reads` place on `chain` once it ran over
    /// [`non_finite_rows`].
    fn demand(chain: serde_json::Value, reads: &[(&str, &str)]) -> BinDemand {
        pyo3::Python::initialize();
        let (_, lineages) = apply_transforms_named(&steps(chain), &non_finite_rows(), &Default::default()).expect("the chain runs");
        BinDemand::of(&lineages, reads)
    }

    fn v() -> Vec<String> {
        vec!["v".to_string()]
    }

    /// With no marker every step is the user's, so no demand reaches any.
    #[test]
    fn a_chain_without_a_marker_bins_nothing() {
        assert!(!demand(serde_json::json!([agg(None), agg(Some("s"))]), &[(FINAL_OUTPUT_KEY, "v"), ("s", "v")]).any());
    }

    /// The unnamed steps after the marker bin for the chart rows' readers;
    /// a user step before the marker runs as written.
    #[test]
    fn the_chart_rows_bin_only_the_lowered_unnamed_steps() {
        let demand = demand(serde_json::json!([agg(None), marker(), agg(None), agg(Some("s"))]), &[(FINAL_OUTPUT_KEY, "v")]);
        assert_eq!((demand.fields(&[0]), demand.fields(&[2]), demand.fields(&[3])), (&[][..], &v()[..], &[][..]));
    }

    /// A named step bins for its own readers only, and a branch's readers
    /// bin that branch's own lowered steps, its scoped keys resolving inside.
    #[test]
    fn a_named_step_and_a_branch_bin_for_their_own_readers() {
        let b = branch("b", None, serde_json::json!([{"type": "identity", "name": "rows"}, marker(), agg(None), agg(Some("inner"))]));
        let demand = demand(
            serde_json::json!([marker(), agg(Some("s")), agg(Some("t")), b]),
            &[("s", "v"), ("b", "v"), ("b_inner", "v"), ("b_inner", "w")],
        );
        assert_eq!(demand.fields(&[1]), v());
        assert!(demand.fields(&[2]).is_empty(), "a sibling's step is untouched");
        assert_eq!((demand.fields(&[3, 2]), demand.fields(&[3, 3])), (&v()[..], &v()[..]), "a field the step does not group by is not demanded");
        assert!(demand.fields(&[3, 0]).is_empty());
    }

    /// A layer reading a branch whose input is a statistic's branch bins
    /// that statistic, through as many branches as it takes, while the
    /// user's own steps in the reading branches stay unbinned. Rejects
    /// claiming only the branch the layer reads.
    #[test]
    fn a_read_through_a_branch_input_bins_the_statistic_it_reads() {
        let chain = serde_json::json!([
            branch("stat", None, serde_json::json!([marker(), agg(None)])),
            {"type": "identity", "name": "rows"},
            keep_all(),
            branch("post", Some("stat"), serde_json::json!([keep_all()])),
            branch("again", Some("post"), serde_json::json!([keep_all()])),
        ]);
        for reader in ["post", "again"] {
            let demand = demand(chain.clone(), &[(reader, "v")]);
            assert_eq!(demand.fields(&[0, 1]), v(), "{reader}");
            assert_eq!(demand.steps.len(), 1, "{reader}: only the statistic bins");
        }
    }

    /// A branch reading a named output, a statistic's own or one a branch
    /// scopes, bins that statistic for the branch's readers. Rejects
    /// stopping at the reading branch.
    #[test]
    fn a_branch_reading_a_named_output_bins_its_statistic() {
        let read = demand(serde_json::json!([marker(), agg(Some("s")), branch("post", Some("s"), serde_json::json!([keep_all()]))]), &[("post", "v")]);
        assert_eq!(read.fields(&[1]), v());
        let read = demand(
            serde_json::json!([
                branch("b", None, serde_json::json!([marker(), agg(Some("s"))])),
                branch("post", Some("b_s"), serde_json::json!([keep_all()])),
            ]),
            &[("post", "v")],
        );
        assert_eq!(read.fields(&[0, 1]), v());
        let read = demand(
            serde_json::json!([branch("b", None, serde_json::json!([marker(), agg(Some("s")), branch("post", Some("s"), serde_json::json!([keep_all()]))]))]),
            &[("b_post", "v")],
        );
        assert_eq!(read.fields(&[0, 1]), v(), "an inner key resolves before an outer one");
    }

    /// Demand reaches only the statistics the binned reader's rows came
    /// from: a sibling statistic of the same shape that an unbinned layer
    /// reads keeps its raw grain, and so do the shared rows before a named
    /// statistic. Rejects binning every lowered step.
    #[test]
    fn a_transitive_read_leaves_its_siblings_unbinned() {
        let read = demand(
            serde_json::json!([
                branch("stat", None, serde_json::json!([marker(), agg(None)])),
                branch("sibling", None, serde_json::json!([marker(), agg(None)])),
                branch("post", Some("stat"), serde_json::json!([keep_all()])),
            ]),
            &[("post", "v")],
        );
        assert_eq!(read.fields(&[0, 1]), v());
        assert!(read.fields(&[1, 1]).is_empty(), "the sibling statistic keeps its grain");
        let read = demand(
            serde_json::json!([marker(), agg(None), agg(Some("s")), branch("post", Some("s"), serde_json::json!([keep_all()]))]),
            &[("post", "v")],
        );
        assert_eq!((read.fields(&[1]), read.fields(&[2])), (&[][..], &v()[..]), "the chain rows other layers read stay raw");
    }

    /// A user step before a statistic's marker, and a user branch with no
    /// marker, run as written under a transitive read; only the lowered
    /// statistic bins. Rejects binning the user's own grouping steps.
    #[test]
    fn a_users_steps_stay_raw_under_a_transitive_read() {
        let read = demand(
            serde_json::json!([
                branch("stat", None, serde_json::json!([agg(None), marker(), agg(None)])),
                branch("post", Some("stat"), serde_json::json!([agg(None)])),
            ]),
            &[("post", "v")],
        );
        assert_eq!((read.fields(&[0, 0]), read.fields(&[0, 2])), (&[][..], &v()[..]));
        assert!(read.fields(&[1, 0]).is_empty(), "the user's aggregate in the reading branch");
        let read = demand(
            serde_json::json!([
                branch("user", None, serde_json::json!([agg(None)])),
                branch("post", Some("user"), serde_json::json!([keep_all()])),
            ]),
            &[("post", "v")],
        );
        assert!(!read.any(), "a user branch with no marker");
    }

    /// A context whose step groups `v` by the bins of `edges`.
    fn binned_ctx(edges: &[f64]) -> crate::transform::context::TransformContext {
        crate::transform::context::TransformContext {
            color_bins: Some(Rc::new(StepBins { fields: vec!["v".into()], edges: Some(Rc::from(edges.to_vec())) })),
            ..Default::default()
        }
    }

    /// Rows with `v` finite in two bins of 1, 5, 9, and three rows each of
    /// NaN, +inf, -inf and null; `x` and `y` rise with the row.
    fn non_finite_rows() -> arrow::array::RecordBatch {
        use arrow::array::{Float64Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        let v: Vec<Option<f64>> = [1.5, 2.5, 3.5, 6.0, 7.0, 8.0, f64::NAN, f64::NAN, f64::NAN, f64::INFINITY, f64::INFINITY, f64::INFINITY,
            f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY]
            .into_iter()
            .map(Some)
            .chain([None, None, None])
            .collect();
        let n = v.len();
        arrow::array::RecordBatch::try_new(
            std::sync::Arc::new(Schema::new(vec![
                Field::new("g", DataType::Utf8, false),
                Field::new("x", DataType::Float64, false),
                Field::new("y", DataType::Float64, false),
                Field::new("v", DataType::Float64, true),
            ])),
            vec![
                std::sync::Arc::new(StringArray::from(vec!["a"; n])),
                std::sync::Arc::new(Float64Array::from((0..n).map(|i| i as f64).collect::<Vec<_>>())),
                std::sync::Arc::new(Float64Array::from((0..n).map(|i| (i * i % 7) as f64 + 1.0).collect::<Vec<_>>())),
                std::sync::Arc::new(Float64Array::from(v)),
            ],
        )
        .unwrap()
    }

    /// Each value of `name` in `batch`, as f64.
    fn values(batch: &arrow::array::RecordBatch, name: &str) -> Vec<f64> {
        crate::render::arrow_cast::col_as_f64(batch, name).unwrap().into_iter().flatten().collect()
    }

    /// Every grouping statistic under a binned `v` groups its finite rows by
    /// bin start and keeps NaN, +inf and -inf each as their own group, as
    /// grouping by the raw value keeps them; a null row is dropped as the
    /// statistic's null policy always dropped it. Rejects dropping the
    /// non-finite rows (keyed as null) and grouping finite rows by raw value.
    #[test]
    fn every_statistic_keeps_non_finite_rows_as_their_own_groups() {
        let statistics = [
            serde_json::json!({"type": "aggregate", "ops": [{"field": "y", "fn": "mean", "as": "m"}], "groupby": ["v"]}),
            serde_json::json!({"type": "bin", "field": "x", "bin_count": 2, "nice": false, "groupby": ["v"]}),
            serde_json::json!({"type": "box_stats", "field": "y", "groupby": ["v"]}),
            serde_json::json!({"type": "kde", "field": "y", "bandwidth": {"kind": "scott"}, "n": 4, "groupby": ["v"]}),
            serde_json::json!({"type": "violin", "field": "y", "groupby": ["v"]}),
            serde_json::json!({"type": "smooth", "x": "x", "y": "y", "method": "lm", "bandwidth": 0.75, "degree": 1, "n": 4, "groupby": ["v"]}),
            serde_json::json!({"type": "summary", "field": "y", "error_fn": "stderr", "ci": 0.95, "n_boot": 10, "groupby": ["v"]}),
            serde_json::json!({"type": "error_extent", "field": "y", "method": "stdev", "groupby": ["v"]}),
        ];
        let rows = non_finite_rows();
        for statistic in statistics {
            let kind = statistic["type"].clone();
            let spec: TransformSpec = serde_json::from_value(statistic).unwrap_or_else(|e| panic!("{kind}: {e}"));
            let out = spec.apply_with_context(&rows, &binned_ctx(&[1.0, 5.0, 9.0])).unwrap_or_else(|e| panic!("{kind}: {e}"));
            let mut groups = values(&out, "v");
            groups.sort_by(f64::total_cmp);
            groups.dedup_by(|a, b| a.total_cmp(b).is_eq());
            assert_eq!(groups.len(), 5, "{kind}: two bins and three non-finite groups: {groups:?}");
            assert_eq!(&groups[1..3], [1.0, 5.0], "{kind}: finite rows key by bin start");
            assert!(groups.contains(&f64::INFINITY) && groups.contains(&f64::NEG_INFINITY) && groups.iter().any(|v| v.is_nan()), "{kind}: {groups:?}");
        }
    }

    /// A null `v` under a binned color still warns in the user's words:
    /// LetterValue and QQ name the field `v`, as they do grouping by raw
    /// value.
    #[test]
    fn a_null_binned_key_warns_naming_the_users_field() {
        let rows = non_finite_rows();
        for statistic in [
            serde_json::json!({"type": "letter_value", "value": "y", "groupby": ["v"], "k_depth": {"type": "full"}}),
            serde_json::json!({"type": "qq", "field": "y", "groupby": ["v"]}),
        ] {
            let kind = statistic["type"].clone();
            let spec: TransformSpec = serde_json::from_value(statistic).unwrap_or_else(|e| panic!("{kind}: {e}"));
            let ctx = binned_ctx(&[1.0, 5.0, 9.0]);
            spec.apply_with_context(&rows, &ctx).unwrap_or_else(|e| panic!("{kind}: {e}"));
            let warnings = ctx.warnings.borrow();
            let named: Vec<&str> = warnings
                .iter()
                .filter_map(|w| match w {
                    crate::transform::context::TransformWarning::NullKeyRowsDropped { field, .. } => Some(field.as_str()),
                    _ => None,
                })
                .collect();
            assert_eq!(named, ["v"], "{kind}: {warnings:?}");
        }
    }

    /// A step whose own output column collides with a binned groupby column
    /// is refused with the same message as when it groups by raw value.
    #[test]
    fn a_binned_groupby_keeps_the_output_collision_refusal() {
        pyo3::Python::initialize();
        let spec: TransformSpec = serde_json::from_value(serde_json::json!(
            {"type": "aggregate", "ops": [{"field": "y", "fn": "mean", "as": "v"}], "groupby": ["g", "v"]}
        ))
        .unwrap();
        let rows = non_finite_rows();
        let raw = spec.apply_with_context(&rows, &Default::default()).expect_err("refused as written").to_string();
        let binned = spec.apply_with_context(&rows, &binned_ctx(&[1.0, 5.0, 9.0])).expect_err("refused under bins").to_string();
        assert_eq!(binned, raw);
        assert!(binned.contains("'v'"), "{binned}");
    }

    /// A degenerate group's warning labels it by the user's field and its
    /// bin start.
    #[test]
    fn a_degenerate_binned_group_is_labelled_by_the_users_field() {
        use arrow::array::{Float64Array, RecordBatch};
        use arrow::datatypes::{DataType, Field, Schema};
        let rows = RecordBatch::try_new(
            std::sync::Arc::new(Schema::new(vec![
                Field::new("x", DataType::Float64, false),
                Field::new("y", DataType::Float64, false),
                Field::new("v", DataType::Float64, false),
            ])),
            vec![
                std::sync::Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0, 3.0])),
                std::sync::Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
                std::sync::Arc::new(Float64Array::from(vec![1.5, 2.5, 3.5, 6.0])),
            ],
        )
        .unwrap();
        let spec: TransformSpec = serde_json::from_value(serde_json::json!(
            {"type": "smooth", "x": "x", "y": "y", "method": "lm", "bandwidth": 0.75, "degree": 1, "n": 4, "groupby": ["v"]}
        ))
        .unwrap();
        let ctx = binned_ctx(&[1.0, 5.0, 9.0]);
        spec.apply_with_context(&rows, &ctx).expect("renders");
        let groups: Vec<String> = ctx
            .warnings
            .borrow()
            .iter()
            .filter_map(|w| match w {
                crate::transform::context::TransformWarning::DegenerateFit { group, .. } => group.clone(),
                _ => None,
            })
            .collect();
        assert_eq!(groups, ["v=5"], "the one-row bin is labelled by v and its start");
    }

    /// A letter-value step grouped by `v`, named `name` (unnamed when `None`).
    fn letter_value(name: Option<&str>) -> serde_json::Value {
        let mut step = serde_json::json!({"type": "letter_value", "value": "y", "groupby": ["v"], "k_depth": {"type": "full"}});
        if let Some(name) = name {
            step["name"] = serde_json::json!(name);
        }
        step
    }

    /// A key resolves to the step that published exactly it, not to the
    /// first step whose name and `_` prefix it: `s_t_outliers` is the
    /// outliers of the step named `s_t`, not of `s`. Rejects resolving by the
    /// first name prefix.
    #[test]
    fn a_key_resolves_to_the_step_that_publishes_it_exactly() {
        let read = demand(serde_json::json!([marker(), agg(Some("s")), letter_value(Some("s_t"))]), &[("s_t_outliers", "v")]);
        assert_eq!((read.fields(&[1]), read.fields(&[2])), (&[][..], &v()[..]));
    }

    /// An unnamed step publishes its secondary outputs under their bare
    /// keys, and a reader of one bins that step. Rejects leaving a bare key
    /// unresolved.
    #[test]
    fn a_bare_secondary_key_resolves_to_its_unnamed_step() {
        assert_eq!(demand(serde_json::json!([marker(), letter_value(None)]), &[("outliers", "v")]).fields(&[1]), v());
    }

    /// Two steps publishing one key: the later one's batch is the one under
    /// the key, so it is the one that bins. Rejects claiming the first
    /// producer.
    #[test]
    fn the_last_step_publishing_a_key_is_its_producer() {
        let read = demand(serde_json::json!([marker(), agg(Some("s")), agg(Some("s"))]), &[("s", "v")]);
        assert_eq!((read.fields(&[1]), read.fields(&[2])), (&[][..], &v()[..]));
        let read = demand(
            serde_json::json!([marker(), agg(Some("b_s")), {"type": "branch", "name": "b", "transforms": [marker(), agg(Some("s"))]}]),
            &[("b_s", "v")],
        );
        assert!(read.fields(&[1]).is_empty(), "the branch re-publishes b_s after the named step");
        assert_eq!(read.fields(&[2, 1]), v());
    }

    /// A step bins only its own fields, and only once edges exist.
    #[test]
    fn step_bins_key_finite_values_by_bin_start() {
        let edges: Rc<[f64]> = Rc::from(vec![1.0, 5.0, 9.0]);
        let bins = StepBins { fields: vec!["v".into()], edges: Some(edges) };
        assert_eq!(bins.bin_start("v", 6.0), Some(5.0));
        assert_eq!(bins.bin_start("v", f64::NAN), None);
        assert_eq!(bins.bin_start("w", 6.0), None);
        assert_eq!(StepBins { fields: vec!["v".into()], edges: None }.bin_start("v", 6.0), None);
    }
}
