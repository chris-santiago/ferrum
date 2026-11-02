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
//!
//! **The positional twin.** A log-resolved positional channel (`x`/`y`)
//! places the same two-pass demand on the lowered statistics it draws
//! ([`SpaceDemand`], [`LogAxisSpace`], [`StepSpace`]): the first run's
//! lineages tell which steps a log-resolved channel reads, and the second
//! run carries each demanded step the axis's log space
//! (`TransformContext::log_space`), so `Bin`/`Kde`/`Kde2D`/the ecdf's
//! `DataWindow` compute in that space instead of data space. [`RunPlan`]
//! generalizes [`ColorBinPlan`] to carry both demands, so a chart needing
//! either reruns once, through the same branch.

use std::collections::HashMap;
use std::rc::Rc;

use crate::render::RenderError;
use crate::spec::layer::PositionalAxis;
use crate::transform::core::{SpaceSlot, TransformSpec};
use crate::transform::run::{OutputLineages, SpaceWalkStep, StepId};

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

/// The generalized run plan `run_pipeline` reruns a chart for: the existing
/// binned-color demand ([`ColorBinPlan`]) and the positional-space demand a
/// log axis places on the lowered statistics it draws ([`SpaceDemand`]). A
/// chart needing either reruns once, through the same [`run_transforms_with_facet`]
/// branch; a chart needing neither keeps its first run, byte-identical.
///
/// [`run_transforms_with_facet`]: crate::render::prepare::run_transforms_with_facet
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RunPlan {
    pub(crate) color: ColorBinPlan,
    pub(crate) space: SpaceDemand,
}

pub(crate) use crate::transform::log_space::LogAxisSpace;

/// One input slot a step is demanded on: the space to compute it in, the
/// axes whose drawn columns reach it, and whether the step should record
/// the caller's own input column for the off-side tally
/// ([`crate::transform::context::TransformContext::note_log_input`]),
/// once per axis in `axes`. `records_caller_input` is true exactly when the
/// slot reads the caller's own column (no earlier lowered step computes it,
/// [`SpaceDemand::of`]'s walk root) AND the step's output column differs
/// from it (a pass-through, like the ecdf's `DataWindow` sort field, never
/// needs a second reader to find it: the axis already reads that column
/// directly).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SlotSpace {
    pub(crate) space: LogAxisSpace,
    pub(crate) axes: Vec<PositionalAxis>,
    pub(crate) records_caller_input: bool,
}

/// The log space(s) one lowered step is demanded in, keyed by the step's
/// own input slot ([`SpaceSlot`]), never by the column name the slot reads:
/// a `Kde2D` whose `x` and `y` read one column is demanded on each slot
/// independently, and each can resolve an independent axis.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct StepSpace {
    slots: HashMap<SpaceSlot, SlotSpace>,
}

impl StepSpace {
    pub(crate) fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// The space demanded on this step's `slot`, `None` if `slot` is not
    /// demanded.
    pub(crate) fn of(&self, slot: SpaceSlot) -> Option<&SlotSpace> {
        self.slots.get(&slot)
    }

    /// Demand the slot `step` of a space walk reaches in `space`, for the
    /// read drawn on `axis`. The slot records the caller's column when the
    /// step computes from it at the walk's root. A slot already demanded
    /// takes `axis` too when the two reads compute it alike (the same base
    /// and side for a computed slot, the same side for a pass-through,
    /// which only leaves out off-side rows), and is refused when they would
    /// compute it differently: one statistic computes each input in one
    /// space.
    fn demand(&mut self, step: &SpaceWalkStep, axis: PositionalAxis, space: LogAxisSpace) -> Result<(), RenderError> {
        let (slot, input) = (step.slot, step.input);
        let Some(existing) = self.slots.get_mut(&slot) else {
            let records_caller_input = step.is_root && step.computes();
            self.slots.insert(slot, SlotSpace { space, axes: vec![axis], records_caller_input });
            return Ok(());
        };
        let agrees = if step.computes() { existing.space == space } else { existing.space.side == space.side };
        if !agrees {
            let first = existing.axes.first().copied().unwrap_or(axis);
            let (first, axis) = first.pairwise_voice(axis);
            return Err(RenderError::ScaleResolutionFailed(format!(
                "the {first} and {axis} scales both draw a column one statistic computes from '{input}', in different log \
                 spaces; a statistic computes each input in one space, so give both axes the same LogScale or draw that \
                 column on one axis"
            )));
        }
        if !existing.axes.contains(&axis) {
            existing.axes.push(axis);
        }
        existing.records_caller_input |= step.is_root && step.computes();
        Ok(())
    }

    /// A space demanded on exactly the given slots, for a caller that needs
    /// to hand a `StepSpace` to a transform's `apply_with_context` without
    /// first building a whole pipeline run (`bin`/`kde`/`kde_2d`/`contour`/
    /// `data_window`'s own log-space unit tests).
    #[cfg(test)]
    pub(crate) fn for_slots(entries: impl IntoIterator<Item = (SpaceSlot, SlotSpace)>) -> Self {
        Self { slots: entries.into_iter().collect() }
    }
}

/// Which lowered steps a log-resolved positional channel demands, and on
/// which of their own input slots — the positional twin of [`BinDemand`].
/// A layer's drawn channel names an OUTPUT column (`bin_start`, `value`,
/// `grid_x`, a `DataWindow` sort field passed through unchanged); the
/// column itself carries no fact about which step computed it or from
/// which of ITS OWN slots, so [`TransformSpec::space_columns`] is the one
/// per-variant map from a step's output columns to the input slot each
/// holds the units of, and [`of`](Self::of) walks a key's lineage backward
/// through it ([`crate::transform::run::OutputLineage::space_walk`]): a
/// `Contour`'s `contour_x` demand reaches the `Kde2D` it reads through its
/// own `grid_x`. A read whose drawn column no lowered step computes (a
/// histogram's count axis, a kde's density axis) demands nothing, so a log
/// count/density/aggregate axis never moves the statistic's own binned or
/// fitted axis.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct SpaceDemand {
    steps: HashMap<StepId, StepSpace>,
}

impl SpaceDemand {
    /// The demand `reads` (one `(key, axis, drawn_column, space)` per
    /// layer drawing a log-resolved channel's own positional column) place
    /// on a run whose outputs' lineages are `lineages`. A key no output
    /// carries demands nothing. Refused when two reads reach one slot of
    /// one step in spaces that would compute it differently
    /// ([`StepSpace::demand`]).
    pub(crate) fn of(lineages: &OutputLineages, reads: &[(String, PositionalAxis, String, LogAxisSpace)]) -> Result<Self, RenderError> {
        let mut demand = Self::default();
        for (key, axis, drawn_column, space) in reads {
            let Some(lineage) = lineages.get(key) else { continue };
            for step in lineage.space_walk(drawn_column) {
                demand.steps.entry(lineage.lowered[step.index].id.clone()).or_default().demand(&step, *axis, *space)?;
            }
        }
        Ok(demand)
    }

    /// Whether any step is demanded in any space.
    pub(crate) fn any(&self) -> bool {
        !self.steps.is_empty()
    }

    /// The space(s) step `id` is demanded in, empty when none.
    pub(crate) fn space(&self, id: &[usize]) -> StepSpace {
        self.steps.get(id).cloned().unwrap_or_default()
    }

    /// A demand naming exactly one step, for a caller that needs to hand a
    /// `SpaceDemand` to a function under test without first building a
    /// whole pipeline run (`render::prepare::extent`'s own facet-pin
    /// tests). Test-only: outside `#[cfg(test)]`, every demand is built
    /// from a real run's lineages via [`Self::of`].
    #[cfg(test)]
    pub(crate) fn single(id: StepId, space: StepSpace) -> Self {
        let mut demand = Self::default();
        demand.steps.insert(id, space);
        demand
    }
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

    // ── SpaceDemand (a drawn log axis's demand on lowered statistics) ──────

    fn space() -> LogAxisSpace {
        LogAxisSpace { base: 10.0, side: crate::scale::log::LogSide::Positive }
    }

    /// The `SpaceDemand` `reads` place on `chain`, run over [`non_finite_rows`]
    /// — the positional twin of [`demand`]. Each read is `(key, axis,
    /// drawn_column, space)`.
    fn space_demand_in(chain: serde_json::Value, reads: &[(&str, PositionalAxis, &str, LogAxisSpace)]) -> Result<SpaceDemand, crate::render::RenderError> {
        pyo3::Python::initialize();
        let (_, lineages) = apply_transforms_named(&steps(chain), &non_finite_rows(), &Default::default()).expect("the chain runs");
        let reads: Vec<(String, PositionalAxis, String, LogAxisSpace)> =
            reads.iter().map(|(k, a, c, s)| (k.to_string(), *a, c.to_string(), *s)).collect();
        SpaceDemand::of(&lineages, &reads)
    }

    /// [`space_demand_in`] with every read in the base-10 positive [`space`].
    fn space_demand(chain: serde_json::Value, reads: &[(&str, PositionalAxis, &str)]) -> SpaceDemand {
        let reads: Vec<(&str, PositionalAxis, &str, LogAxisSpace)> = reads.iter().map(|(k, a, c)| (*k, *a, *c, space())).collect();
        space_demand_in(chain, &reads).expect("one space per slot")
    }

    fn bin_step(field: &str, name: Option<&str>) -> serde_json::Value {
        let mut step = serde_json::json!({"type": "bin", "field": field, "bin_count": 4, "nice": false});
        if let Some(name) = name {
            step["name"] = serde_json::json!(name);
        }
        step
    }

    fn kde2d_step(x: &str, y: &str, name: Option<&str>) -> serde_json::Value {
        let mut step = serde_json::json!({"type": "kde2_d", "x": x, "y": y});
        if let Some(name) = name {
            step["name"] = serde_json::json!(name);
        }
        step
    }

    fn contour_step(name: Option<&str>) -> serde_json::Value {
        let mut step = serde_json::json!({"type": "contour"});
        if let Some(name) = name {
            step["name"] = serde_json::json!(name);
        }
        step
    }

    fn data_window_step(sort_field: &str, name: Option<&str>) -> serde_json::Value {
        let mut step = serde_json::json!({
            "type": "data_window",
            "ops": [{"op": "count", "field": sort_field, "as": "c"}],
            "sort": [sort_field],
        });
        if let Some(name) = name {
            step["name"] = serde_json::json!(name);
        }
        step
    }

    /// With no marker every step is the user's, so no space demand reaches
    /// any — the same rule [`a_chain_without_a_marker_bins_nothing`] pins
    /// for the color demand.
    #[test]
    fn a_chain_without_a_marker_demands_no_space() {
        let demand = space_demand(serde_json::json!([bin_step("v", None)]), &[(FINAL_OUTPUT_KEY, PositionalAxis::X, "bin_start")]);
        assert!(!demand.any());
    }

    /// An `x` read drawing `bin_start` demands the `Bin` on its `field`
    /// slot, recording the caller's own column (root, and the output column
    /// differs from it).
    #[test]
    fn an_x_read_drawing_bin_start_demands_the_bin_on_its_field_slot() {
        let demand = space_demand(serde_json::json!([marker(), bin_step("v", None)]), &[(FINAL_OUTPUT_KEY, PositionalAxis::X, "bin_start")]);
        let slot = demand.space(&[1]).of(SpaceSlot::Field).cloned().expect("the Bin is demanded on its field slot");
        assert_eq!(slot, SlotSpace { space: space(), axes: vec![PositionalAxis::X], records_caller_input: true });
    }

    /// A `y` read drawing `count` demands nothing: `Bin`'s own
    /// `space_columns` names only `bin_start`/`bin_end`, so a log count
    /// axis never moves the binned axis. Rejects a field-blind demand that
    /// marks every lowered step in the lineage regardless of which column
    /// the channel actually draws.
    #[test]
    fn a_y_read_drawing_count_demands_nothing() {
        let demand = space_demand(serde_json::json!([marker(), bin_step("v", None)]), &[(FINAL_OUTPUT_KEY, PositionalAxis::Y, "count")]);
        assert!(!demand.any());
    }

    /// A `contour_x` read reaches the shared `Kde2D` through its own
    /// `grid_x`: the named `Contour` output's lineage records `Kde2D` as a
    /// shared (not lowered-for-this-reader) step, and the walk still finds
    /// it because shared steps are searched too. `Contour` itself is
    /// demanded on its `grid_x` slot but never records a caller input (its
    /// own input is a statistic's output, not the caller's); `Kde2D` is
    /// demanded on its `x` slot and does record one.
    #[test]
    fn a_contour_x_read_reaches_the_shared_kde2d_through_grid_x() {
        let chain = serde_json::json!([marker(), kde2d_step("x", "y", None), contour_step(Some("contour"))]);
        let demand = space_demand(chain, &[("contour", PositionalAxis::X, "contour_x")]);
        let contour_space = demand.space(&[2]).of(SpaceSlot::GridX).cloned().expect("Contour demanded on grid_x");
        assert!(!contour_space.records_caller_input, "grid_x is Kde2D's output, not the caller's");
        let kde2d_space = demand.space(&[1]).of(SpaceSlot::X).cloned().expect("Kde2D demanded on x");
        assert!(kde2d_space.records_caller_input, "x is the caller's own field, and grid_x differs from it");
        assert!(demand.space(&[1]).of(SpaceSlot::Y).is_none(), "an x read leaves the y slot alone");
    }

    /// A `DataWindow` read demands only its own sort slot, as a
    /// pass-through column: its drawn output equals its input, so it never
    /// records a caller input itself (the axis already reads that column
    /// directly).
    #[test]
    fn a_data_window_read_demands_its_sort_slot_as_a_pass_through() {
        let demand = space_demand(serde_json::json!([marker(), data_window_step("v", None)]), &[(FINAL_OUTPUT_KEY, PositionalAxis::X, "v")]);
        let slot = demand.space(&[1]).of(SpaceSlot::Sort(0)).cloned().expect("DataWindow demanded on its sort slot");
        assert!(!slot.records_caller_input, "v passes through unchanged, so the axis already reads it directly");
    }

    fn smooth_step(x: &str, y: &str, name: Option<&str>) -> serde_json::Value {
        let mut step = serde_json::json!({"type": "smooth", "x": x, "y": y, "method": "lm", "bandwidth": 0.75, "degree": 1, "n": 5});
        if let Some(name) = name {
            step["name"] = serde_json::json!(name);
        }
        step
    }

    /// `Smooth`'s Fitted output literally names its grid column `x`: when
    /// the caller's own field is ALSO named `x` (as here), a name-based
    /// `computes()` (`output != input`) would misread the collision as a
    /// pass-through — like `DataWindow`'s sort field above, which really
    /// does carry its input through unchanged — and drop the off-side
    /// record. [`SpaceColumn::pass_through`] is declared explicitly
    /// instead, so the record survives the name coincidence.
    #[test]
    fn a_smooth_fit_on_a_field_named_like_its_own_output_still_records_the_caller_column() {
        let demand = space_demand(serde_json::json!([marker(), smooth_step("x", "y", None)]), &[(FINAL_OUTPUT_KEY, PositionalAxis::X, "x")]);
        let slot = demand.space(&[1]).of(SpaceSlot::X).cloned().expect("Smooth demanded on its X slot");
        assert!(
            slot.records_caller_input,
            "x is the caller's own field, genuinely computed (not passed through), even though it shares the Fitted output's own column name"
        );
    }

    /// The same name-collision shape as the `Smooth` case above, for
    /// `Kde`'s grid column (always named `value`): a caller field literally
    /// named `value` must still have its off-side record kept.
    #[test]
    fn a_kde_on_a_field_named_like_its_own_output_still_records_the_caller_column() {
        pyo3::Python::initialize();
        let batch = arrow::array::RecordBatch::try_new(
            std::sync::Arc::new(arrow::datatypes::Schema::new(vec![arrow::datatypes::Field::new(
                "value",
                arrow::datatypes::DataType::Float64,
                false,
            )])),
            vec![std::sync::Arc::new(arrow::array::Float64Array::from(vec![1.0, 2.0, 4.0]))],
        )
        .unwrap();
        let chain = serde_json::json!([marker(), {"type": "kde", "field": "value", "bandwidth": {"kind": "scott"}, "n": 4}]);
        let (_, lineages) = apply_transforms_named(&steps(chain), &batch, &Default::default()).expect("the chain runs");
        let reads = vec![(FINAL_OUTPUT_KEY.to_string(), PositionalAxis::X, "value".to_string(), space())];
        let demand = SpaceDemand::of(&lineages, &reads).expect("one space per slot");
        let slot = demand.space(&[1]).of(SpaceSlot::Field).cloned().expect("Kde demanded on its field slot");
        assert!(
            slot.records_caller_input,
            "value is the caller's own field, genuinely computed (not passed through), even though it shares the grid's own output name"
        );
    }

    /// A read through a branch input reaches the statistic it reads,
    /// exactly as the color demand's [`a_read_through_a_branch_input_bins_the_statistic_it_reads`]
    /// does — the space demand reuses the same lineage-walking mechanism.
    #[test]
    fn space_reaches_through_a_branch_input_like_the_color_demand() {
        let chain = serde_json::json!([
            branch("stat", None, serde_json::json!([marker(), bin_step("v", None)])),
            branch("post", Some("stat"), serde_json::json!([keep_all()])),
        ]);
        let demand = space_demand(chain, &[("post", PositionalAxis::X, "bin_start")]);
        assert!(demand.space(&[0, 1]).of(SpaceSlot::Field).is_some());
    }

    /// A `Kde2D` whose `x` and `y` read one column is demanded per slot: an
    /// `x` read drawing `grid_x` demands only the `x` slot, and reads on
    /// both axes demand each slot in its own axis's space, even when the
    /// two spaces differ. Rejects keying the demand by the column name,
    /// which gives both slots one entry.
    #[test]
    fn a_kde2d_reading_one_column_on_both_axes_is_demanded_per_slot() {
        let chain = || serde_json::json!([marker(), kde2d_step("v", "v", None)]);
        let x_only = space_demand(chain(), &[(FINAL_OUTPUT_KEY, PositionalAxis::X, "grid_x")]);
        assert!(x_only.space(&[1]).of(SpaceSlot::X).is_some());
        assert!(x_only.space(&[1]).of(SpaceSlot::Y).is_none(), "the y slot reads v in data units");
        let base_two = LogAxisSpace { base: 2.0, ..space() };
        let both = space_demand_in(chain(), &[(FINAL_OUTPUT_KEY, PositionalAxis::X, "grid_x", space()), (FINAL_OUTPUT_KEY, PositionalAxis::Y, "grid_y", base_two)])
            .expect("two slots, one space each");
        let step = both.space(&[1]);
        assert_eq!(step.of(SpaceSlot::X).map(|s| (s.space, s.axes.clone())), Some((space(), vec![PositionalAxis::X])));
        assert_eq!(step.of(SpaceSlot::Y).map(|s| (s.space, s.axes.clone())), Some((base_two, vec![PositionalAxis::Y])));
    }

    /// One computed slot drawn on `x` and `y` in different log spaces is
    /// refused naming both scales: a statistic computes each input in one
    /// space. In one space the slot is demanded once, on both axes, so the
    /// caller's rows count on each. Rejects the later read silently
    /// overwriting the first.
    #[test]
    fn one_computed_slot_on_two_axes_needs_one_space() {
        let chain = || serde_json::json!([marker(), bin_step("v", None)]);
        let base_two = LogAxisSpace { base: 2.0, ..space() };
        let refused = space_demand_in(chain(), &[(FINAL_OUTPUT_KEY, PositionalAxis::X, "bin_start", space()), (FINAL_OUTPUT_KEY, PositionalAxis::Y, "bin_start", base_two)])
            .expect_err("two spaces for one slot");
        assert_eq!(
            refused.to_string(),
            crate::render::RenderError::ScaleResolutionFailed(
                "the x and y scales both draw a column one statistic computes from 'v', in different log spaces; a statistic \
                 computes each input in one space, so give both axes the same LogScale or draw that column on one axis"
                    .to_string()
            )
            .to_string()
        );
        let shared = space_demand_in(chain(), &[(FINAL_OUTPUT_KEY, PositionalAxis::X, "bin_start", space()), (FINAL_OUTPUT_KEY, PositionalAxis::Y, "bin_end", space())])
            .expect("one space");
        let slot = shared.space(&[1]).of(SpaceSlot::Field).cloned().expect("demanded");
        assert_eq!(slot, SlotSpace { space: space(), axes: vec![PositionalAxis::X, PositionalAxis::Y], records_caller_input: true });
    }

    /// The same two-spaces refusal, with one read on the primary `y` slot
    /// and the other on an independent-y secondary slot: the message names
    /// the secondary "secondary y", distinct from the bare "y" the primary
    /// keeps — not "the y and y scales" twice over.
    #[test]
    fn one_computed_slot_on_the_primary_and_a_secondary_y_needs_one_space() {
        let chain = || serde_json::json!([marker(), bin_step("v", None)]);
        let base_two = LogAxisSpace { base: 2.0, ..space() };
        let refused = space_demand_in(
            chain(),
            &[
                (FINAL_OUTPUT_KEY, PositionalAxis::Y, "bin_start", space()),
                (FINAL_OUTPUT_KEY, PositionalAxis::secondary_y(1), "bin_start", base_two),
            ],
        )
        .expect_err("two spaces for one slot");
        assert_eq!(
            refused.to_string(),
            crate::render::RenderError::ScaleResolutionFailed(
                "the y and secondary y scales both draw a column one statistic computes from 'v', in different log spaces; a \
                 statistic computes each input in one space, so give both axes the same LogScale or draw that column on one axis"
                    .to_string()
            )
            .to_string()
        );
    }

    /// Two DIFFERENT independent-y secondary slots drawing one statistic's
    /// field in different log spaces are distinguished by slot number: "the
    /// secondary y and secondary y scales" could not tell which secondary
    /// slot is which, so each names its own.
    #[test]
    fn one_computed_slot_on_two_distinct_secondary_ys_names_each_slot() {
        let chain = || serde_json::json!([marker(), bin_step("v", None)]);
        let base_two = LogAxisSpace { base: 2.0, ..space() };
        let refused = space_demand_in(
            chain(),
            &[
                (FINAL_OUTPUT_KEY, PositionalAxis::secondary_y(1), "bin_start", space()),
                (FINAL_OUTPUT_KEY, PositionalAxis::secondary_y(2), "bin_start", base_two),
            ],
        )
        .expect_err("two spaces for one slot");
        assert_eq!(
            refused.to_string(),
            crate::render::RenderError::ScaleResolutionFailed(
                "the secondary y slot 1 and secondary y slot 2 scales both draw a column one statistic computes from 'v', in \
                 different log spaces; a statistic computes each input in one space, so give both axes the same LogScale or draw \
                 that column on one axis"
                    .to_string()
            )
            .to_string()
        );
    }

    /// A pass-through slot (a `DataWindow` sort field) drawn on two axes
    /// whose spaces differ only in base agrees: it only leaves out
    /// off-side rows, which depend on the side alone. Opposite sides are
    /// refused.
    #[test]
    fn a_pass_through_slot_on_two_axes_needs_only_one_side() {
        let chain = || serde_json::json!([marker(), data_window_step("v", None)]);
        let base_two = LogAxisSpace { base: 2.0, ..space() };
        let agreed = space_demand_in(chain(), &[(FINAL_OUTPUT_KEY, PositionalAxis::X, "v", space()), (FINAL_OUTPUT_KEY, PositionalAxis::Y, "v", base_two)]).expect("one side");
        assert_eq!(agreed.space(&[1]).of(SpaceSlot::Sort(0)).map(|s| s.axes.clone()), Some(vec![PositionalAxis::X, PositionalAxis::Y]));
        let negative = LogAxisSpace { side: crate::scale::log::LogSide::Negative, ..space() };
        assert!(space_demand_in(chain(), &[(FINAL_OUTPUT_KEY, PositionalAxis::X, "v", space()), (FINAL_OUTPUT_KEY, PositionalAxis::Y, "v", negative)]).is_err());
    }

    /// `LogAxisSpace::forward`/`backward` round-trip an on-side value, and
    /// `forward` excludes an off-side (zero or wrong-signed) one.
    #[test]
    fn log_axis_space_forward_and_backward_round_trip_on_side_values() {
        let positive = space();
        assert_eq!(positive.forward(100.0), Some(2.0));
        assert_eq!(positive.backward(2.0), 100.0);
        assert_eq!(positive.forward(0.0), None);
        assert_eq!(positive.forward(-5.0), None);
        assert!(positive.off_side(0.0));
        assert!(positive.off_side(-5.0));
        assert!(!positive.off_side(5.0));
        assert!(!positive.off_side(f64::NAN));

        // Signed: `forward` is `s * log_b|v|` (`s = -1` on the negative
        // side), so it stays monotone INCREASING in `v` here too — a
        // smaller magnitude (closer to zero) maps to a LARGER signed
        // result than a more negative one, matching data order (see
        // `log_space::LogAxisSpace::forward`'s doc).
        let negative = LogAxisSpace { base: 10.0, side: crate::scale::log::LogSide::Negative };
        assert_eq!(negative.forward(-100.0), Some(-2.0));
        assert_eq!(negative.backward(-2.0), -100.0);
        assert_eq!(negative.forward(0.0), None);
        assert_eq!(negative.forward(5.0), None);
    }
}
