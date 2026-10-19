//! Statistics under a binned color group by its bins.
//!
//! A binned color (`Color(bin=...)`) leaves its column unchanged and maps it
//! through a bin-ordinal scale. A statistic lowered from a layer's encoding
//! or mark that groups by that column would otherwise make one group per raw
//! value, so several marks of one bin would stack, dodge or overlap as
//! separate groups. Under a [`ColorBinPlan`] the steps a binned-color layer
//! reads directly group that column by bin: the shared partitioner
//! ([`crate::transform::group_key::group_partition_multi`]) keys each finite
//! value by its bin's start under the color scale's edges, and the step
//! writes that start under the column's own name. The step reads the raw
//! values as written, so a histogram of `v` colored by `v`'s bins still
//! counts each row at its own position.
//!
//! **Which steps.** A step bins a column only when (1) a layer whose drawn
//! color, or whose text grouping-only `group` key, carries a `bin` rule on
//! that column reads the step's output directly (its `data_source`, or the
//! chart rows for a layer with none), and (2) the step sits after its
//! chain's statistics marker ([`is_statistics_marker`]): the steps a layer's
//! encoding or mark lowered to, never the user's own `.transform()` steps,
//! which run as written. A chain with no marker holds only user steps, so
//! nothing in it bins.
//!
//! **Edges.** The edges are cut from the rows the binned statistics consume,
//! so the render runs the pipeline twice. Both runs record, beside each
//! output a bin-grouped lineage produced, the rows that lineage read before
//! its first bin-grouped step ([`BinnedOrigin`]); the color scale cuts its
//! edges from those origins, and the second run groups by them.
//! Origins are read before any grouping, so they are the same on both runs.

use std::rc::Rc;

use crate::transform::core::{spec_name, TransformSpec};
use crate::transform::registry::secondary_key;

/// The rows an output's lineage read before its first step that grouped a
/// binned color by bin, and the columns its bin-grouped steps grouped: what
/// a binned color cuts its edges from. The pipeline run records one per
/// output whose lineage passed such a step, keyed like the output
/// ([`BinnedOrigins`]), as it records each output's `Swarm`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BinnedOrigin {
    pub(crate) rows: arrow::array::RecordBatch,
    pub(crate) fields: Vec<String>,
}

impl BinnedOrigin {
    /// This origin with `fields` grouped too; its rows stay the first
    /// grouped step's input.
    pub(crate) fn with_fields(&self, fields: &[String]) -> Self {
        let mut origin = self.clone();
        for field in fields {
            if !origin.fields.contains(field) {
                origin.fields.push(field.clone());
            }
        }
        origin
    }

    /// Its rows, when one of its bin-grouped steps grouped `field`.
    pub(crate) fn rows_of(&self, field: &str) -> Option<&arrow::array::RecordBatch> {
        self.fields.iter().any(|grouped| grouped == field).then_some(&self.rows)
    }
}

/// Each published output's [`BinnedOrigin`], keyed by the output's key.
pub(crate) type BinnedOrigins = std::collections::HashMap<String, BinnedOrigin>;

/// Whether `spec` is a chain's statistics marker: an identity step with
/// `statistics: true` (wire `{"type": "identity", "statistics": true}`).
/// The steps after the first marker in a chain are lowered from layers'
/// encodings and marks; the steps before it are the user's own.
pub(crate) fn is_statistics_marker(spec: &TransformSpec) -> bool {
    matches!(spec, TransformSpec::Identity(identity) if identity.statistics)
}

/// The binned columns each step of one chain groups by bin, and, for a
/// `Branch` step, its own chain's.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct ChainDemand {
    steps: Vec<StepDemand>,
}

#[derive(Debug, Clone, Default, PartialEq)]
struct StepDemand {
    fields: Vec<String>,
    branch: ChainDemand,
}

impl ChainDemand {
    /// The demand `demands` (one `(data_source, field)` per binned binding of
    /// a reading layer) places on `transforms`: each pair names the step that
    /// produces `data_source` directly ([`Self::claim`]).
    pub(crate) fn of(transforms: &[TransformSpec], demands: &[(Option<&str>, &str)]) -> Self {
        let mut demand = Self::skeleton(transforms);
        for (source, field) in demands {
            demand.claim(transforms, *source, field);
        }
        demand
    }

    fn skeleton(transforms: &[TransformSpec]) -> Self {
        let steps = transforms
            .iter()
            .map(|spec| StepDemand {
                fields: Vec::new(),
                branch: match spec {
                    TransformSpec::Branch(branch) => Self::skeleton(&branch.transforms),
                    _ => Self::default(),
                },
            })
            .collect();
        Self { steps }
    }

    /// Whether any step of this chain or a branch inside it bins a column.
    pub(crate) fn any(&self) -> bool {
        self.steps.iter().any(|step| !step.fields.is_empty() || step.branch.any())
    }

    /// The columns step `index` of this chain bins.
    pub(crate) fn fields(&self, index: usize) -> &[String] {
        self.steps.get(index).map_or(&[], |step| step.fields.as_slice())
    }

    /// The demand on the chain of the `Branch` at step `index`.
    pub(crate) fn branch(&self, index: usize) -> Option<&ChainDemand> {
        self.steps.get(index).map(|step| &step.branch)
    }

    /// Mark the steps of `chain` that produce `source` directly as binning
    /// `field`, among the lowered steps (after the chain's marker):
    /// - `None`: the chain's unnamed steps, which advance to its tail;
    /// - a key: the last step in chain order that publishes exactly that key
    ///   ([`publication`]), as the run's last insert under a key is the batch
    ///   a reader draws. A named or unnamed step's own key is that step; a
    ///   `Branch`'s name is its unnamed steps; a key a branch scopes is
    ///   resolved inside it.
    ///
    /// A key no step publishes resolves to nothing: the render refuses a
    /// layer reading it before any scale reads its rows.
    fn claim(&mut self, chain: &[TransformSpec], source: Option<&str>, field: &str) {
        let lowered_from = chain.iter().position(is_statistics_marker).map(|marker| marker + 1);
        let lowered = |index: usize| lowered_from.is_some_and(|from| index >= from);
        let Some(key) = source else {
            for (index, spec) in chain.iter().enumerate() {
                if lowered(index) && spec_name(spec).is_none() {
                    self.push(chain, index, field);
                }
            }
            return;
        };
        let producer = chain.iter().enumerate().rev().find_map(|(index, spec)| publication(spec, key).map(|p| (index, spec, p)));
        match producer {
            Some((index, _, Publication::Own)) if lowered(index) => self.push(chain, index, field),
            Some((index, TransformSpec::Branch(branch), Publication::BranchTail)) => {
                self.steps[index].branch.claim(&branch.transforms, None, field)
            }
            Some((index, TransformSpec::Branch(branch), Publication::Scoped(inner))) => {
                self.steps[index].branch.claim(&branch.transforms, Some(inner), field)
            }
            _ => {}
        }
    }

    /// Mark step `index` of `chain` as binning `field`, when it groups by it.
    fn push(&mut self, chain: &[TransformSpec], index: usize, field: &str) {
        if !chain[index].groupby().is_some_and(|groupby| groupby.iter().any(|g| g == field)) {
            return;
        }
        let fields = &mut self.steps[index].fields;
        if !fields.iter().any(|existing| existing == field) {
            fields.push(field.to_string());
        }
    }
}

/// How one step publishes a key ([`publication`]).
enum Publication<'k> {
    /// The step's own output: its name, or a secondary key it publishes
    /// ([`TransformSpec::secondary_keys`], bare when it is unnamed).
    Own,
    /// A `Branch`'s unnamed tail, published under the branch's name.
    BranchTail,
    /// A key a `Branch` re-publishes scoped under its name: the inner key.
    Scoped(&'k str),
}

/// How `spec` publishes `key` when it runs, mirroring the pipeline runner's
/// inserts: a step's name and its secondary keys; a `Branch`'s name for its
/// tail, and `"{branch}_{inner}"` for every key its own chain publishes.
fn publication<'k>(spec: &TransformSpec, key: &'k str) -> Option<Publication<'k>> {
    match spec {
        TransformSpec::Branch(branch) if branch.name == key => Some(Publication::BranchTail),
        TransformSpec::Branch(branch) => {
            let inner = key.strip_prefix(&secondary_key(Some(&branch.name), ""))?;
            branch.transforms.iter().any(|step| publication(step, inner).is_some()).then_some(Publication::Scoped(inner))
        }
        step if spec_name(step) == Some(key) || step.secondary_keys().iter().any(|published| published == key) => {
            Some(Publication::Own)
        }
        _ => None,
    }
}

/// A run's binned-color grouping: the demand on the pipeline's own chain and
/// the color scale's edges, `None` on the run that publishes the origins
/// they are cut from, which groups every column by its raw values.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ColorBinPlan {
    pub(crate) demand: ChainDemand,
    pub(crate) edges: Option<Rc<[f64]>>,
}

/// The binned columns one step groups by bin and the edges they are cut by
/// (`None` on the origins run), carried by that step's
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

    fn steps(json: serde_json::Value) -> Vec<TransformSpec> {
        serde_json::from_value(json).expect("transforms deserialize")
    }

    fn agg(name: Option<&str>) -> serde_json::Value {
        let mut step = serde_json::json!({"type": "aggregate", "ops": [{"field": "y", "fn": "mean", "as": "m"}], "groupby": ["v"]});
        if let Some(name) = name {
            step["name"] = serde_json::json!(name);
        }
        step
    }

    fn marker() -> serde_json::Value {
        serde_json::json!({"type": "identity", "statistics": true})
    }

    /// With no marker every step is the user's, so no demand reaches any.
    #[test]
    fn a_chain_without_a_marker_bins_nothing() {
        let chain = steps(serde_json::json!([agg(None), agg(Some("s"))]));
        assert!(!ChainDemand::of(&chain, &[(None, "v"), (Some("s"), "v")]).any());
    }

    /// The unnamed steps after the marker bin for the chart rows' readers;
    /// a user step before the marker runs as written.
    #[test]
    fn the_chart_rows_bin_only_the_lowered_unnamed_steps() {
        let chain = steps(serde_json::json!([agg(None), marker(), agg(None), agg(Some("s"))]));
        let demand = ChainDemand::of(&chain, &[(None, "v")]);
        assert_eq!((demand.fields(0), demand.fields(2), demand.fields(3)), (&[][..], &["v".to_string()][..], &[][..]));
    }

    /// A named step bins for its own readers only, and a branch's readers
    /// bin that branch's own lowered steps, its scoped keys resolving inside.
    #[test]
    fn a_named_step_and_a_branch_bin_for_their_own_readers() {
        let branch = serde_json::json!({"type": "branch", "name": "b", "input": "rows", "transforms": [
            {"type": "identity", "name": "rows"}, marker(), agg(None), agg(Some("inner"))]});
        let chain = steps(serde_json::json!([marker(), agg(Some("s")), agg(Some("t")), branch]));
        let demand = ChainDemand::of(&chain, &[(Some("s"), "v"), (Some("b"), "v"), (Some("b_inner"), "v"), (Some("b_inner"), "w")]);
        assert_eq!(demand.fields(1), ["v"]);
        assert!(demand.fields(2).is_empty(), "a sibling's step is untouched");
        let inner = demand.branch(3).expect("the branch's demand");
        assert_eq!((inner.fields(2), inner.fields(3)), (&["v".to_string()][..], &["v".to_string()][..]), "a field the step does not group by is not demanded");
        assert!(inner.fields(0).is_empty());
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

    /// A key resolves to the step that publishes exactly it, not to the
    /// first step whose name and `_` prefix it: `s_t_outliers` is the
    /// outliers of the step named `s_t`, not of `s`. Rejects resolving by the
    /// first name prefix.
    #[test]
    fn a_key_resolves_to_the_step_that_publishes_it_exactly() {
        let chain = steps(serde_json::json!([marker(), agg(Some("s")), letter_value(Some("s_t"))]));
        let demand = ChainDemand::of(&chain, &[(Some("s_t_outliers"), "v")]);
        assert_eq!((demand.fields(1), demand.fields(2)), (&[][..], &["v".to_string()][..]));
    }

    /// An unnamed step publishes its secondary outputs under their bare
    /// keys, and a reader of one bins that step. Rejects leaving a bare key
    /// unresolved.
    #[test]
    fn a_bare_secondary_key_resolves_to_its_unnamed_step() {
        let chain = steps(serde_json::json!([marker(), letter_value(None)]));
        for key in ["outliers", "depth_3"] {
            assert_eq!(ChainDemand::of(&chain, &[(Some(key), "v")]).fields(1), ["v"], "{key}");
        }
    }

    /// Two steps publishing one key: the later one's batch is the one under
    /// the key, so it is the one that bins. Rejects claiming the first
    /// producer.
    #[test]
    fn the_last_step_publishing_a_key_is_its_producer() {
        let chain = steps(serde_json::json!([marker(), agg(Some("s")), agg(Some("s"))]));
        let demand = ChainDemand::of(&chain, &[(Some("s"), "v")]);
        assert_eq!((demand.fields(1), demand.fields(2)), (&[][..], &["v".to_string()][..]));
        let branch = |inner: &str| serde_json::json!({"type": "branch", "name": "b", "transforms": [marker(), agg(Some(inner))]});
        let chain = steps(serde_json::json!([marker(), agg(Some("b_s")), branch("s")]));
        let demand = ChainDemand::of(&chain, &[(Some("b_s"), "v")]);
        assert!(demand.fields(1).is_empty(), "the branch re-publishes b_s after the named step");
        assert_eq!(demand.branch(2).expect("the branch").fields(1), ["v"]);
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
