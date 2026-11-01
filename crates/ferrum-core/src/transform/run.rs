//! The pipeline runner: one chain of steps over one batch, publishing named
//! outputs, with what follows each batch's rows recorded beside it.
//!
//! Every published batch carries a lineage: the record of what its rows
//! went through that a later reader still needs. A step's outputs inherit
//! its input's lineage, and a `Branch` over a published key starts from
//! that key's lineage, so a property recorded once follows the rows into
//! every step, named output, secondary output and branch that reads them,
//! and into nothing else. A branch over the source starts a fresh lineage.
//!
//! What a lineage records:
//! - the `Swarm` whose rows these still are;
//! - the lowered steps the rows passed (the steps after a chain's
//!   statistics marker, [`crate::transform::color_bin_group::is_statistics_marker`]),
//!   and the rows the first of them consumed: a batch whose lineage passed
//!   one is a statistic's output, and those rows are what a data-aware sort
//!   and a binned color's edges read in its place;
//! - the rows the lineage started from: the pipeline's source, or an
//!   operand's framed rows ([`crate::transform::branch::OperandFrame`]);
//! - the composition steps the rows met ([`crate::transform::composed`]).

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use pyo3::exceptions::PyValueError;
use pyo3::{PyErr, PyResult, Python};

use crate::transform::branch::{self, BranchSpec};
use crate::transform::color_bin_group::{is_statistics_marker, BinDemand, SpaceDemand, StepBins, StepSpace};
use crate::transform::composed::{ComposedRuns, ComposedSpec, ComposedState};
use crate::transform::context::{RowsOrigin, StepInstance, TransformContext};
use crate::transform::core::{spec_name, SpaceColumn, SpaceSlot, TransformSpec};
use crate::transform::swarm::SwarmSpec;

/// Sentinel key under which the final pipeline output is always published in
/// the map returned by [`apply_transforms_named`]. Layers with `data_source: None`
/// resolve to this key.
pub(crate) const FINAL_OUTPUT_KEY: &str = "__final__";

/// A step's place in a pipeline: the `Branch` indices from the pipeline's
/// own chain down to its chain, then its index there. Every run of one
/// pipeline (each facet partition, each binned-color pass) gives a step
/// the same id.
pub(crate) type StepId = Vec<usize>;

/// A lowered step a batch's rows passed, and the columns it groups by.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LoweredStep {
    pub(crate) id: StepId,
    pub(crate) groupby: Vec<String>,
    /// Whether the batch reached this step's rows through a named or
    /// secondary output made from its chain's tail: those rows are the
    /// tail's, which every reader of the chain shares, so they are not the
    /// batch's own statistic.
    pub(crate) shared: bool,
    /// The output columns this step computes in one input slot's units
    /// ([`TransformSpec::space_columns`]), read by [`OutputLineage::space_walk`]
    /// to tell which of a channel's drawn columns this step computed, and
    /// from which of its own input slots.
    pub(crate) space_columns: Vec<SpaceColumn>,
}

/// One step of [`OutputLineage::space_walk`]: the lowered step at `index`
/// in [`OutputLineage::lowered`] computes `output` from its `slot`, which
/// reads column `input`. `is_root` when no earlier lowered step outputs
/// `input`, so `input` is the caller's own column rather than a statistic's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SpaceWalkStep<'a> {
    pub(crate) index: usize,
    pub(crate) slot: SpaceSlot,
    pub(crate) output: &'a str,
    pub(crate) input: &'a str,
    pub(crate) is_root: bool,
}

impl SpaceWalkStep<'_> {
    /// Whether the step computes `output` from `input` rather than passing
    /// the input through under its own name (a `DataWindow` sort field).
    pub(crate) fn computes(&self) -> bool {
        self.output != self.input
    }
}

/// What a run records beside one published batch ([`apply_transforms_named`]).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct OutputLineage {
    /// The `Swarm` whose rows the batch still is: a step after it keeps it,
    /// a secondary output descends from none. Read by
    /// `render::position::apply_swarm`, so the one routing rule decides both
    /// which batch a layer reads and which `Swarm` places it.
    pub(crate) swarm: Option<SwarmSpec>,
    /// The rows the batch's own statistics consumed: the input of the first
    /// lowered step it passed that is not shared
    /// ([`LoweredStep::shared`]).
    pub(crate) statistic_input: Option<RecordBatch>,
    /// The lowered steps the lineage passed, in order.
    pub(crate) lowered: Vec<LoweredStep>,
    /// The `groupby` fields of the user's own steps the lineage passed (the
    /// steps a chart's statistics marker does not lower, and a
    /// composition's steps), and the fields a lowered step's caller wrote on
    /// it ([`caller_groupby`]), in order, each once. A line, area or ribbon
    /// splits by the grouping of the statistic that produced its rows: a
    /// field an earlier step grouped by stays here past a lowered statistic
    /// only when that statistic partitioned by it too
    /// ([`TransformSpec::partition`]), so an ungrouped ECDF window over a
    /// caller's grouped rows draws one line, not one per earlier group. A
    /// user's own steps, and a lowered step that computes each row alone,
    /// keep every field written before them.
    pub(crate) written_groupby: Vec<String>,
    /// The schema of the rows the lineage started from: the pipeline's
    /// source, or an operand's framed rows.
    pub(crate) source: SchemaRef,
    /// The caller's own input column a demanded lowered statistic in this
    /// lineage read, by `(axis, field)` ("x"/"y", the caller's column name)
    /// — recorded only for a slot `SpaceDemand::of` marks as the walk's
    /// root with an output column that differs from its input
    /// (`SlotSpace::records_caller_input`), once per axis that draws it, so
    /// a pass-through statistic (the ecdf's `DataWindow`) never duplicates
    /// what the axis already reads directly from its own drawn column. Read
    /// by the off-side tally
    /// (`render::scale_resolve::domain::fold_log_input_into_tally`), which
    /// counts it the same content-deduplicated way a primitive layer's own
    /// column read is counted. Empty for a lineage no space demand ever
    /// reached.
    pub(crate) log_input: HashMap<(String, String), arrow::array::ArrayRef>,
}

impl OutputLineage {
    /// Whether the batch is a statistic's output: its rows passed a lowered
    /// step. Every other batch holds the user's rows.
    pub(crate) fn is_statistic(&self) -> bool {
        !self.lowered.is_empty()
    }

    /// The rows a data-aware sort reads for this batch, `batch`: the rows
    /// its statistics consumed, or the batch itself when it holds the
    /// user's rows.
    pub(crate) fn pre_statistic_rows<'b>(&'b self, batch: &'b RecordBatch) -> &'b RecordBatch {
        self.statistic_input.as_ref().unwrap_or(batch)
    }

    /// The batch's own statistics that group by `field`: what a binned
    /// color on `field` read from this batch groups by bin.
    pub(crate) fn grouping<'a>(&'a self, field: &'a str) -> impl Iterator<Item = &'a LoweredStep> {
        self.lowered.iter().filter(move |step| !step.shared && step.groupby.iter().any(|g| g == field))
    }

    /// The rows a binned color on `field` cuts its edges from: the rows the
    /// batch's own statistics consumed, when one of them groups by `field`.
    pub(crate) fn binned_rows(&self, field: &str) -> Option<&RecordBatch> {
        self.grouping(field).next().and(self.statistic_input.as_ref())
    }

    /// The lowered steps whose computed columns `drawn_column` descends
    /// from, latest first: the latest lowered step (shared steps included,
    /// since a named output's drawn column can descend from its chain's
    /// tail, as a `Contour` polygon does from the `Kde2D` surface it reads)
    /// whose [`LoweredStep::space_columns`] name the column, then, with that
    /// step's input as the next column to find, the latest earlier step
    /// that outputs it, until no earlier step does. Empty for a column no
    /// lowered step computes (a histogram's count, a kde's density).
    pub(crate) fn space_walk<'a>(&'a self, drawn_column: &'a str) -> Vec<SpaceWalkStep<'a>> {
        let outputs = |step: &LoweredStep, column: &str| step.space_columns.iter().any(|c| c.output == column);
        let mut walk = Vec::new();
        let mut target = drawn_column;
        let mut search_upto = self.lowered.len();
        while let Some((index, column)) = self.lowered[..search_upto]
            .iter()
            .enumerate()
            .rev()
            .find_map(|(i, step)| step.space_columns.iter().find(|c| c.output == target).map(|c| (i, c)))
        {
            let is_root = !self.lowered[..index].iter().any(|step| outputs(step, &column.input));
            walk.push(SpaceWalkStep { index, slot: column.slot, output: &column.output, input: &column.input, is_root });
            target = &column.input;
            search_upto = index;
        }
        walk
    }
}

/// Each published key's [`OutputLineage`].
pub(crate) type OutputLineages = HashMap<String, OutputLineage>;

/// The batch every transform output key publishes, and each key's lineage
/// ([`apply_transforms_named`]).
pub(crate) type NamedOutputs = (HashMap<String, RecordBatch>, OutputLineages);

/// Apply each transform; record named outputs with fan-out semantics.
///
/// - **Named transforms** (`name = Some(...)`) run on the CURRENT chained
///   batch (the prior unnamed output, or the original input if none). They
///   publish their output under their name but do NOT advance the chained
///   pipeline pointer — subsequent unnamed transforms see the same chained
///   input the named transform did.
/// - **Unnamed transforms** chain: each consumes the prior unnamed output.
/// - **`Branch`** runs its own sub-chain, with these same rules, over the
///   original batch (`input = None`) or the batch published under `input`.
///   Every named and secondary output inside it — a branch's outputs are
///   its secondary outputs — is published
///   SCOPED under [`crate::transform::registry::secondary_key`]`(Some(branch.name), key)`,
///   then the sub-chain tail under the branch's own `name`, unscoped, as
///   always. It does not advance the chained pipeline pointer.
/// - **`Composed`** (a composition's step) is placed by row lineage
///   ([`crate::transform::composed`]).
/// - A chain's **statistics marker** passes its rows through and marks the
///   steps after it as lowered.
///
/// [`FINAL_OUTPUT_KEY`] always points at the final UNNAMED-chain tail, or at
/// the original input when no unnamed transforms ran.
///
/// Under a binned color's grouping ([`crate::transform::color_bin_group`]),
/// each step the run's [`BinDemand`] names groups those columns by bin.
///
/// A named step reads the chained batch rather than the original input so
/// compound desugars (mark_contour's Kde2D→Contour, clustermap's
/// Linkage→Reorder→Unpivot, residplot's residuals overlay) can consume the
/// previous unnamed transform's output; `Branch` is the route to the
/// original input.
pub(crate) fn apply_transforms_named(
    specs: &[TransformSpec],
    batch: &RecordBatch,
    ctx: &TransformContext,
) -> PyResult<NamedOutputs> {
    let plan = ctx.run_plan.as_deref();
    let mut run = Run {
        ctx,
        demand: plan.map(|plan| &plan.color.demand),
        edges: plan.and_then(|plan| plan.color.edges.clone()),
        space: plan.map(|plan| &plan.space),
        composed: ComposedRuns::default(),
    };
    let source = Rows::over(batch.clone());
    let mut published = Published::default();
    let tail = run.chain(specs, &[], source.clone(), &source, &mut published)?;
    published.insert(FINAL_OUTPUT_KEY.to_string(), tail);
    run.composed.refuse_unrun(ctx)?;
    Ok(published.into_outputs())
}

/// A branch applied on its own, outside a pipeline run: its sub-chain's
/// tail over `batch` (or over the named output its `input` names, read from
/// `ctx`).
pub(crate) fn apply_branch(spec: &BranchSpec, batch: &RecordBatch, ctx: &TransformContext) -> PyResult<RecordBatch> {
    let mut run = Run { ctx, demand: None, edges: None, space: None, composed: ComposedRuns::default() };
    let mut published = Published { outputs: ctx.named_outputs.clone(), ..Published::default() };
    run.branch(spec, &[], &Rows::over(batch.clone()), &mut published).map(|tail| tail.batch)
}

/// The run-time lineage of one batch's rows ([`OutputLineage`], plus the
/// composition steps the rows met).
#[derive(Debug, Clone)]
struct Lineage<'s> {
    swarm: Option<SwarmSpec>,
    statistic_input: Option<RecordBatch>,
    lowered: Vec<LoweredStep>,
    written_groupby: Vec<String>,
    source: SchemaRef,
    /// The identity of these rows, read into [`TransformContext::rows`] by
    /// every step or copy that reads them next: the run's source and a
    /// framed branch's own rows carry [`RowsOrigin::output`] `None` with
    /// [`StepInstance::Unplaced`] and [`StepInstance::Chain`]`(path)`
    /// respectively; an unnamed step's tail is `{Chain(id), None}`; a named
    /// step's published output is `{Chain(id), Some(name)}`; a secondary
    /// output is `{Chain(id), Some(key)}`; an unnamed composition copy's
    /// output is `{Composed(group), None}`; a named copy's published
    /// output is `{Composed(group), Some(name)}`. The `output` slot is what
    /// tells a named step's own tail apart from its secondary output, or a
    /// named copy's published output apart from the rows it ran over — both
    /// pairs come from the SAME step, so [`StepInstance`] alone cannot name
    /// both. Two calls sharing one `origin` read the identical rows, which
    /// is how [`crate::transform::context::merge_null_key_reports`] tells a
    /// composite's own several lowered steps over one tail (reporting the
    /// same rows twice) from two lowerings over genuinely different rows.
    origin: RowsOrigin,
    /// Which composition steps ran on these rows and which still wait for a
    /// column ([`ComposedState`]).
    composed: ComposedState<'s>,
    /// See [`OutputLineage::log_input`].
    log_input: HashMap<(String, String), arrow::array::ArrayRef>,
}

impl Lineage<'_> {
    /// The lineage of rows nothing has run on yet.
    fn over(batch: &RecordBatch) -> Self {
        Self {
            swarm: None,
            statistic_input: None,
            lowered: Vec::new(),
            written_groupby: Vec::new(),
            source: batch.schema(),
            origin: RowsOrigin::default(),
            composed: ComposedState::default(),
            log_input: HashMap::new(),
        }
    }

    fn recorded(self) -> OutputLineage {
        OutputLineage {
            swarm: self.swarm,
            statistic_input: self.statistic_input,
            lowered: self.lowered,
            written_groupby: self.written_groupby,
            source: self.source,
            log_input: self.log_input,
        }
    }

    /// This lineage after a step whose `fields` were authored by the
    /// caller rather than derived by the chart's own statistic planner:
    /// a user's own step's whole `groupby`, or a lowered step's
    /// [`caller_groupby`]. Each field joins [`OutputLineage::written_groupby`]
    /// once.
    fn written(mut self, fields: &[String]) -> Self {
        for field in fields {
            if !self.written_groupby.contains(field) {
                self.written_groupby.push(field.clone());
            }
        }
        self
    }

    /// This lineage after a lowered statistic that computed each output row
    /// across `partition` ([`TransformSpec::partition`]): a field an earlier
    /// step grouped by stays in [`OutputLineage::written_groupby`] only when
    /// the statistic partitioned by it too, in its order: a field it
    /// computed across does not separate the rows it produced.
    fn regrouped(mut self, partition: &[String]) -> Self {
        self.written_groupby.retain(|field| partition.contains(field));
        self
    }
}

/// A batch and its lineage.
#[derive(Debug, Clone)]
struct Rows<'s> {
    batch: RecordBatch,
    lineage: Lineage<'s>,
}

impl Rows<'_> {
    /// Rows nothing has run on yet.
    fn over(batch: RecordBatch) -> Self {
        Self { lineage: Lineage::over(&batch), batch }
    }
}

/// Every output published so far in one pipeline run, with its lineage.
#[derive(Default, Clone)]
struct Published<'s> {
    outputs: HashMap<String, RecordBatch>,
    lineages: HashMap<String, Lineage<'s>>,
    /// Keys [`Self::insert`] has recorded since the last [`Self::clear_log`]
    /// (construction counts as one), in insertion order, not deduplicated —
    /// a key an inner step overwrites is logged once per insert. A branch's
    /// sub-chain runs against a CHILD `Published` cloned from the parent;
    /// this log is how [`Run::branch`] tells "genuinely new in this run" from
    /// "inherited by the clone" without diffing the two maps.
    inserted: Vec<String>,
}

impl<'s> Published<'s> {
    /// Publish `rows` under `key`, replacing any earlier batch and lineage
    /// there.
    fn insert(&mut self, key: String, rows: Rows<'s>) {
        self.inserted.push(key.clone());
        self.lineages.insert(key.clone(), rows.lineage);
        self.outputs.insert(key, rows.batch);
    }

    /// The rows published under `key`, with their lineage (a fresh one for
    /// a batch handed in from outside the run).
    fn rows(&self, key: &str) -> Option<Rows<'s>> {
        let batch = self.outputs.get(key)?.clone();
        let lineage = self.lineages.get(key).cloned().unwrap_or_else(|| Lineage::over(&batch));
        Some(Rows { batch, lineage })
    }

    /// Reset the insertion log without touching what is published — called
    /// on a branch's child `Published` right after it is cloned from the
    /// parent, so the log that follows counts only this branch's own run.
    fn clear_log(&mut self) {
        self.inserted.clear();
    }

    fn into_outputs(self) -> NamedOutputs {
        let lineages = self.lineages.into_iter().map(|(key, lineage)| (key, lineage.recorded())).collect();
        (self.outputs, lineages)
    }
}

/// One pipeline run: its context, the steps that group a binned color by
/// bin and the edges they group by, and the composition steps it met.
struct Run<'s, 'c> {
    ctx: &'c TransformContext,
    demand: Option<&'c BinDemand>,
    edges: Option<Rc<[f64]>>,
    space: Option<&'c SpaceDemand>,
    composed: ComposedRuns<'s>,
}

impl<'s> Run<'s, '_> {
    /// Run `specs`, the chain at `path`, starting from `start`, publishing
    /// named, secondary and branch outputs into `published`; returns the
    /// chain tail. `source` is the rows a `Branch` with no `input` reads.
    ///
    /// `lowering` tracks the id of the statistics marker this chain last
    /// met: each marker opens a new lowering, overwriting whatever the
    /// chain's own PRIOR marker (if any) had opened, so a chain carrying
    /// several markers — one per lowering, the wire shape a chart with
    /// several statistics (two `Chart.layer` statistics, a figure statistic
    /// beside its mark's) writes — gives each marker's own steps their own
    /// lowering id rather than all sharing the chain's first one. Threaded
    /// into [`Self::step`] only for a step the marker actually lowers (the
    /// same condition `own` already gates on); a step before any marker, a
    /// branch's own steps with no marker of its own, and every
    /// [`Self::run_copy`] call carry `None`.
    fn chain(
        &mut self,
        specs: &'s [TransformSpec],
        path: &[usize],
        start: Rows<'s>,
        source: &Rows<'s>,
        published: &mut Published<'s>,
    ) -> PyResult<Rows<'s>> {
        let mut current = start;
        let mut lowered = false;
        let mut lowering: Option<StepId> = None;
        let statistic_columns = statistic_columns(specs);
        for (index, spec) in specs.iter().enumerate() {
            let id: StepId = [path, &[index]].concat();
            current = match spec {
                TransformSpec::Branch(branch) => {
                    let tail = self.branch(branch, &id, source, published)?;
                    published.insert(branch.name.clone(), tail);
                    current
                }
                TransformSpec::Composed(copy) => {
                    self.composed.met(copy, &current.batch, &statistic_columns);
                    self.place(copy, current, published)?
                }
                marker if is_statistics_marker(marker) => {
                    lowered = true;
                    lowering = Some(id.clone());
                    current
                }
                step => {
                    let missing = if lowered || statistic_columns.is_empty() {
                        Vec::new()
                    } else {
                        unread_fields(step, &current)
                    };
                    if !lowered {
                        refuse_a_statistic_read(step, &missing, &statistic_columns)?;
                    }
                    refuse_a_dropped_read(step, &current)?;
                    let own = (lowered && !matches!(step, TransformSpec::Identity(_))).then(|| id.clone());
                    let step_lowering = own.is_some().then(|| lowering.clone()).flatten();
                    let hint_columns: &[(String, &'static str)] = if missing.is_empty() { &[] } else { &statistic_columns };
                    self.step(step, &id, own, step_lowering, current, published)
                        .map_err(|err| with_statistics_hint(err, hint_columns, StatisticsRemedy::ComputeFirst))?
                }
            };
        }
        Ok(current)
    }

    /// Run one row step over `current`: publish its secondary outputs and,
    /// when it is named, its output; return the chain's new tail. `id` is
    /// the step's position, `lowered` is that id when it is a lowered step,
    /// and `step_lowering` is the id of the marker that opened its lowering
    /// (always `None` when `lowered` is `None`; see [`Self::chain`]'s doc).
    /// [`Self::chain`]'s step arm refuses a composition-dropped read and an
    /// exact statistic read before calling this. Below, `secondary_outputs`,
    /// `fork` and `settle` can also raise, but only once `spec`'s own
    /// `apply_with_context` has already succeeded; when `spec` reads a
    /// column its rows lack, that `apply_with_context` call is the one that
    /// refuses, so it is the only error the statistics hint ever wraps (the
    /// hint is appended by [`Self::chain`]'s step arm around this call).
    fn step(
        &mut self,
        spec: &'s TransformSpec,
        id: &StepId,
        lowered: Option<StepId>,
        step_lowering: Option<StepId>,
        mut current: Rows<'s>,
        published: &mut Published<'s>,
    ) -> PyResult<Rows<'s>> {
        let fields = match (&lowered, self.demand) {
            (Some(id), Some(demand)) => demand.fields(id),
            _ => &[],
        };
        let space = match (&lowered, self.space) {
            (Some(id), Some(space)) => space.space(id),
            _ => StepSpace::default(),
        };
        // Each transform sees the named outputs accumulated so far via the
        // context, which `Reorder(from=...)` reads.
        let mut step_ctx = self.ctx.clone();
        step_ctx.named_outputs = published.outputs.clone();
        step_ctx.step = StepInstance::Chain(id.clone());
        step_ctx.rows = current.lineage.origin.clone();
        step_ctx.lowering = step_lowering;
        step_ctx.color_bins = (!fields.is_empty())
            .then(|| Rc::new(StepBins { fields: fields.to_vec(), edges: self.edges.clone() }));
        step_ctx.log_space = (!space.is_empty()).then(|| Rc::new(space));
        // A fresh sink per step call (never `self.ctx.log_input.clone()`,
        // which would share one `Rc` across the whole run): the column a
        // demanded step records here belongs to THIS step's own output,
        // read back right after the call below.
        step_ctx.log_input = Rc::new(RefCell::new(HashMap::new()));
        let result = spec.apply_with_context(&current.batch, &step_ctx)?;
        let step_log_input = step_ctx.log_input.borrow().clone();
        let own = lowered.map(|id| LoweredStep {
            id,
            groupby: spec.groupby().map(<[String]>::to_vec).unwrap_or_default(),
            shared: false,
            space_columns: spec.space_columns(),
        });
        // A user's own step records its WHOLE grouping on every output it
        // makes, since there is no planner involved before the marker at
        // all. A lowered step records only the fields the caller wrote
        // directly on it ([`caller_groupby`]'s doc), which is everything
        // for most kinds but a narrower subset for `Aggregate`/`Bin`.
        let written_fields: &[String] = if own.is_some() { caller_groupby(spec) } else { spec.groupby().unwrap_or(&[]) };
        // A lowered statistic that computes across rows first drops every
        // earlier written field it did not partition by (`Lineage::regrouped`);
        // a user's own step, and a lowered step that computes each row alone,
        // keep every field written before them.
        let partition = if own.is_some() { spec.partition() } else { None };
        let after = |lineage: Lineage<'s>| match partition {
            Some(partition) => lineage.regrouped(partition).written(written_fields),
            None => lineage.written(written_fields),
        };
        // Secondary outputs first, so a named step's explicit `name` wins a
        // key collision. A secondary output descends from no Swarm.
        for (key, batch) in spec.secondary_outputs(&current.batch, &result, &step_ctx)? {
            let mut lineage = after(Lineage { swarm: None, ..forked(&current, own.clone(), &step_log_input) });
            lineage.origin = RowsOrigin { step: StepInstance::Chain(id.clone()), output: Some(key.clone()) };
            let rows = self.fork(&mut current, Rows { batch, lineage }, published)?;
            published.insert(key, rows);
        }
        match spec_name(spec) {
            Some(name) => {
                let mut lineage = after(with_swarm(forked(&current, own, &step_log_input), spec));
                lineage.origin = RowsOrigin { step: StepInstance::Chain(id.clone()), output: Some(name.to_string()) };
                let rows = self.fork(&mut current, Rows { batch: result, lineage }, published)?;
                published.insert(name.to_string(), rows);
                Ok(current)
            }
            None => {
                let mut lineage = after(current.lineage.clone());
                if let Some(own) = own {
                    lineage.statistic_input.get_or_insert_with(|| current.batch.clone());
                    lineage.lowered.push(own);
                    for (key, column) in &step_log_input {
                        lineage.log_input.insert(key.clone(), column.clone());
                    }
                }
                lineage.origin = RowsOrigin { step: StepInstance::Chain(id.clone()), output: None };
                self.settle(Rows { batch: result, lineage: with_swarm(lineage, spec) }, published)
            }
        }
    }

    /// Place a composition step's copy on `rows`: nothing when a copy of its
    /// group already ran on, or waits in, this lineage; run it when the
    /// rows carry every column it reads; else leave it waiting.
    fn place(&mut self, copy: &'s ComposedSpec, mut rows: Rows<'s>, published: &mut Published<'s>) -> PyResult<Rows<'s>> {
        if rows.lineage.composed.holds(&copy.group) {
            return Ok(rows);
        }
        if !copy.readable_in(&rows.batch.schema()) {
            rows.lineage.composed.wait(copy);
            return Ok(rows);
        }
        let rows = self.run_copy(copy, rows, published)?;
        self.settle(rows, published)
    }

    /// `rows` with every waiting copy they can now read run over them, in
    /// the order the lineage met them.
    fn settle(&mut self, mut rows: Rows<'s>, published: &mut Published<'s>) -> PyResult<Rows<'s>> {
        for copy in rows.lineage.composed.take_waiting() {
            if copy.readable_in(&rows.batch.schema()) {
                rows = self.run_copy(copy, rows, published)?;
            } else {
                rows.lineage.composed.wait(copy);
            }
        }
        Ok(rows)
    }

    /// Settle `out`, an output made from `current` that does not advance
    /// the chain. A named copy that ran on it has published its one output
    /// for this chain, so `current` stops waiting on it.
    fn fork(&mut self, current: &mut Rows<'s>, out: Rows<'s>, published: &mut Published<'s>) -> PyResult<Rows<'s>> {
        let out = self.settle(out, published)?;
        current.lineage.composed.adopt_named_runs(&out.lineage.composed);
        Ok(out)
    }

    /// Run `copy` over `rows`. An unnamed copy returns its output with the
    /// group recorded as run on it; a named copy publishes its output under
    /// its name and returns `rows` unchanged, the group recorded as run.
    fn run_copy(&mut self, copy: &'s ComposedSpec, rows: Rows<'s>, published: &mut Published<'s>) -> PyResult<Rows<'s>> {
        let mut step_ctx = self.ctx.clone();
        step_ctx.named_outputs = published.outputs.clone();
        step_ctx.step = StepInstance::Composed(copy.group.clone());
        step_ctx.rows = rows.lineage.origin.clone();
        let out = copy.step.apply_with_context(&rows.batch, &step_ctx)?;
        self.composed.ran(&copy.group);
        let Rows { batch, mut lineage } = rows;
        match spec_name(&copy.step) {
            Some(name) => {
                lineage.composed.ran(copy, None);
                let mut published_lineage = lineage.clone().written(copy.step.groupby().unwrap_or(&[]));
                published_lineage.origin =
                    RowsOrigin { step: StepInstance::Composed(copy.group.clone()), output: Some(name.to_string()) };
                published.insert(name.to_string(), Rows { batch: out, lineage: published_lineage });
                Ok(Rows { batch, lineage })
            }
            None => {
                lineage.composed.ran(copy, Some((&batch.schema(), &out.schema())));
                let mut lineage = lineage.written(copy.step.groupby().unwrap_or(&[]));
                lineage.origin = RowsOrigin { step: StepInstance::Composed(copy.group.clone()), output: None };
                Ok(Rows { batch: out, lineage })
            }
        }
    }

    /// Run a branch's sub-chain over its input, publishing every output
    /// inside it into `published` SCOPED under the branch's own name — a
    /// branch's inner outputs are its secondary outputs. Returns the
    /// sub-chain tail, which [`Self::chain`] publishes UNSCOPED under the
    /// branch's `name`. `path` is the branch's own step id.
    ///
    /// The branch starts from its input's rows and lineage: `source` when it
    /// has no `input`, else the rows published under `input`. A framed
    /// branch starts from its frame's rows of that input
    /// ([`crate::transform::branch::OperandFrame`]), which are the rows its
    /// lineage starts from and the rows a branch nested in it with no
    /// `input` reads.
    ///
    /// The sub-chain runs against a CHILD `Published` cloned from
    /// `published` (batches are Arc-backed, so the clone is cheap): every
    /// inner step sees every outer key plus its own unscoped inner keys,
    /// exactly as if run at top level — `Reorder(from=)` on an inner
    /// `Linkage`, and a nested `Branch` whose `input` names an inner or
    /// outer key, both resolve correctly. The child's insertion log is
    /// cleared right after cloning ([`Published::clear_log`]), so only keys
    /// this run genuinely adds — the branch's own named/secondary outputs,
    /// and (recursively) a nested branch's tail plus its own already-scoped
    /// keys — are logged. Each logged key, deduplicated and read at its
    /// final value, is re-published into the PARENT under
    /// [`crate::transform::registry::secondary_key`]`(Some(&spec.name), key)`:
    /// a scoped key can never equal the branch's own bare name (the rule's
    /// `"{name}_{key}"` shape always has the separator), so two sibling
    /// branches that reuse the same inner name publish two DIFFERENT scoped
    /// keys, each carrying its own rows.
    fn branch(
        &mut self,
        spec: &'s BranchSpec,
        path: &[usize],
        source: &Rows<'s>,
        published: &mut Published<'s>,
    ) -> PyResult<Rows<'s>> {
        let input = branch::resolve_input(spec, &source.batch, &published.outputs)?;
        let start = match spec.input.as_deref() {
            Some(key) => published.rows(key).unwrap_or_else(|| Rows::over(input.clone())),
            None => source.clone(),
        };
        let mut child = published.clone();
        child.clear_log();
        let tail = match &spec.frame {
            None => self.chain(&spec.transforms, path, start, source, &mut child)?,
            Some(frame) => {
                let batch = frame.rows(&spec.name, &start.batch, &self.ctx.facet_fields)?;
                let lineage = Lineage {
                    source: batch.schema(),
                    origin: RowsOrigin { step: StepInstance::Chain(path.to_vec()), output: None },
                    ..start.lineage
                };
                let framed = Rows { batch, lineage };
                self.chain(&spec.transforms, path, framed.clone(), &framed, &mut child)?
            }
        };

        let mut seen = std::collections::HashSet::new();
        for key in &child.inserted {
            if !seen.insert(key.as_str()) {
                continue;
            }
            // `insert` always logs and inserts a key together, so
            // `child.inserted` never names a key `child.outputs` lacks.
            let Some(rows) = child.rows(key) else { continue };
            published.insert(crate::transform::registry::secondary_key(Some(&spec.name), key), rows);
        }
        Ok(tail)
    }
}

/// Whether `spec`'s `groupby` is the chart's own statistic planner's
/// derivation from the layer's effective encoding (every bound channel,
/// value channels included) rather than a keyword the caller wrote on the
/// step itself. Only `Aggregate` and `Bin` are ever lowered this way: the
/// planner never builds any other kind (the one statistic plan).
///
/// `Aggregate` and `Bin` differ in how much of their `groupby` the planner
/// can own. An `Aggregate`'s whole `groupby` is always planner-derived,
/// spanning every encoded channel including ones that must never split a
/// series (`opacity`, `size`), so none of it joins [`OutputLineage::written_groupby`]
/// and splitting on it stays gated on `shape`/`stroke_dash` binding
/// instead. A `Bin`'s `groupby` can MIX a planner-derived field (its own
/// planning, when the chart groups a positional Bin by a drawn channel)
/// with a field the caller wrote directly on the `Bin` itself (a composite
/// mark's `bin=Bin(groupby=[...])`), which the wire alone cannot tell
/// apart -- Python records the caller's own subset under `BinSpec::written_groupby`
/// at the point it knows which fields it wrote, and this reads only that
/// subset. Every other lowered kind with a `groupby` -- `Smooth`, `Kde`,
/// `Robust`, `Logistic`, `Glm`, and the rest -- carries a value the caller
/// passed directly to the transform, exactly as a user's own pre-marker
/// step's `groupby` does, so its WHOLE `groupby` joins
/// [`OutputLineage::written_groupby`], read by a line/area/ribbon's series
/// split regardless of whether any channel also names the field.
fn caller_groupby(spec: &TransformSpec) -> &[String] {
    match spec {
        TransformSpec::Aggregate(_) => &[],
        TransformSpec::Bin(bin) => &bin.written_groupby,
        other => other.groupby().unwrap_or(&[]),
    }
}

/// The lineage of an output a step makes from `current` beside the chain
/// (a named or secondary output), `own` the step itself when it is lowered:
/// the tail's lowered steps reach it as shared, and its own statistic, when
/// it is one, consumed the tail's rows and (if demanded) recorded
/// `log_input`'s caller column.
fn forked<'s>(current: &Rows<'s>, own: Option<LoweredStep>, log_input: &HashMap<(String, String), arrow::array::ArrayRef>) -> Lineage<'s> {
    let mut lineage = current.lineage.clone();
    for step in &mut lineage.lowered {
        step.shared = true;
    }
    if let Some(own) = own {
        lineage.statistic_input = Some(current.batch.clone());
        lineage.lowered.push(own);
        for (key, column) in log_input {
            lineage.log_input.insert(key.clone(), column.clone());
        }
    }
    lineage
}

/// `lineage` after `spec`: a `Swarm` step's output is that Swarm's rows; any
/// other step's keeps the Swarm its input carried.
fn with_swarm<'s>(mut lineage: Lineage<'s>, spec: &TransformSpec) -> Lineage<'s> {
    if let TransformSpec::Swarm(swarm) = spec {
        lineage.swarm = Some(swarm.clone());
    }
    lineage
}

/// Each column a lowered step of `chain` (one after its statistics marker)
/// computes, with the computing statistic's label: what the chain's own
/// steps before the marker cannot read, since they run first.
fn statistic_columns(chain: &[TransformSpec]) -> Vec<(String, &'static str)> {
    let Some(marker) = chain.iter().position(is_statistics_marker) else { return Vec::new() };
    chain[marker + 1..]
        .iter()
        .flat_map(|step| step.output_columns().into_iter().map(move |column| (column, step.label())))
        .collect()
}

/// The fields `spec` reads that `rows` lack: the ones its chain's own steps,
/// which run before any statistic, cannot supply.
fn unread_fields(spec: &TransformSpec, rows: &Rows<'_>) -> Vec<String> {
    let schema = rows.batch.schema();
    spec.fields_read().into_iter().filter(|field| schema.index_of(field).is_err()).collect()
}

/// Refuse `spec`, one of the user's own steps, when `missing` (computed by
/// [`unread_fields`]) names a column that one of `statistic_columns` names
/// exactly: the user's steps run before the chart's statistics, so the step
/// would otherwise refuse in its own voice with no word of why the column is
/// missing. A `missing` field that matches no statistic's output reaches
/// [`with_statistics_hint`] instead, over the step's own refusal.
fn refuse_a_statistic_read(spec: &TransformSpec, missing: &[String], statistic_columns: &[(String, &'static str)]) -> PyResult<()> {
    let exact = missing
        .iter()
        .find_map(|field| statistic_columns.iter().find(|(column, _)| column == field).map(|(_, label)| (field, *label)));
    match exact {
        None => Ok(()),
        Some((column, label)) => Err(PyValueError::new_err(format!(
            "{}: column '{column}' is computed by the chart's statistics ({label}), which run after the chart's \
             .transform() steps: .transform() reads the chart's rows before its statistics. Compute '{column}' \
             in .transform() before this step (e.g. with fm.transform_aggregate) and bind the encoding to it",
            spec.label(),
        ))),
    }
}

/// The remedy [`with_statistics_hint`] names, by which door's chain the
/// statistics it appends belong to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StatisticsRemedy {
    /// The flat chart and a static `LayerChart` member
    /// ([`Run::chain`]'s step arm): compute the value in `.transform()`
    /// before this step.
    ComputeFirst,
    /// The `+` composition door
    /// ([`crate::transform::composed::ComposedRuns::refuse_unrun`]): name
    /// one of the statistics' own output columns directly, since a
    /// statistic's output is readable there, unlike on the flat door.
    NameAnOutput,
}

/// `err`, the step's own refusal of a column its rows lack ([`Run::chain`]'s
/// step arm refuses a composition-dropped read and an exact statistic read
/// before calling the step, and `each_steps_label_is_the_prefix_its_own_refusal_carries`
/// guards that every other step reading a missing column refuses in its own
/// `apply_with_context`, so this is the only error either door's call site
/// ever wraps), with the statistics clause appended: which statistics,
/// naming them and the columns they compute, after `.transform()` already
/// ran, plus `remedy`'s one sentence. `err` passes through unchanged when
/// `statistic_columns` is empty — no statistics after the marker, or (at the
/// flat/`LayerChart` call site) no column actually missing.
pub(crate) fn with_statistics_hint(err: PyErr, statistic_columns: &[(String, &'static str)], remedy: StatisticsRemedy) -> PyErr {
    if statistic_columns.is_empty() {
        return err;
    }
    let message = Python::attach(|py| err.value(py).to_string());
    let (labels, outputs) = distinct_statistics(statistic_columns);
    let outputs = python_list(&outputs);
    let remedy = match remedy {
        StatisticsRemedy::ComputeFirst => "compute the value in .transform() before this step (e.g. with \
             fm.transform_aggregate) and bind the encoding to it"
            .to_string(),
        StatisticsRemedy::NameAnOutput => format!(
            "a step written on the composition runs after an operand's statistics only where it names one \
             of their columns: name one of {outputs}"
        ),
    };
    PyValueError::new_err(format!(
        "{message}. The chart's statistics ({}) compute {outputs} after its .transform() steps, so .transform() \
         cannot read them: {remedy}",
        labels.join(", "),
    ))
}

/// `items` as a Python list literal (`['a', 'b']`): the spelling the
/// statistics hint borrows from Python's own `repr`, not Rust's
/// double-quoted `Debug`.
fn python_list(items: &[&str]) -> String {
    format!("[{}]", items.iter().map(|item| format!("'{item}'")).collect::<Vec<_>>().join(", "))
}

/// The distinct statistic labels and the distinct columns they compute, read
/// off `statistic_columns` in the order it lists them.
fn distinct_statistics<'a>(statistic_columns: &'a [(String, &'static str)]) -> (Vec<&'static str>, Vec<&'a str>) {
    let mut labels: Vec<&'static str> = Vec::new();
    let mut outputs: Vec<&str> = Vec::new();
    for (column, label) in statistic_columns {
        if !labels.contains(label) {
            labels.push(*label);
        }
        if !outputs.contains(&column.as_str()) {
            outputs.push(column.as_str());
        }
    }
    (labels, outputs)
}

/// Refuse `spec` when it reads a column its rows lack because a
/// composition step that ran on them dropped it: the statistic would
/// otherwise refuse in its own voice, naming neither the composition's step
/// nor what to do about it.
fn refuse_a_dropped_read(spec: &TransformSpec, rows: &Rows<'_>) -> PyResult<()> {
    let schema = rows.batch.schema();
    let dropped = spec
        .fields_read()
        .into_iter()
        .filter(|field| schema.index_of(field).is_err())
        .find_map(|field| rows.lineage.composed.dropped(&field).map(|copy| (field, copy)));
    match dropped {
        None => Ok(()),
        Some((column, copy)) => Err(PyValueError::new_err(format!(
            "{}: column '{column}' was dropped by the step written on the composition's .transform() ({}), \
             which runs on this operand's rows before its statistics; keep '{column}' in that step's output, \
             or write the step on the operands it should change instead of on the composition",
            spec.label(),
            copy.step.label(),
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    use crate::transform::color_bin_group::BinDemand;

    /// `g` (a, a, b, b, b) and `v` (1..=5).
    fn rows() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, false),
            Field::new("v", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "a", "b", "b", "b"])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0])),
            ],
        )
        .unwrap()
    }

    fn chain(json: serde_json::Value) -> Vec<TransformSpec> {
        serde_json::from_value(json).expect("the chain deserializes")
    }

    fn composed(predicate: &str) -> serde_json::Value {
        serde_json::json!({"type": "composed", "group": "f", "step": {"type": "filter", "predicate": predicate}})
    }

    /// A copy of a step that adds 100 to `count`: running it twice on one
    /// rows set shows, where a filter would not.
    fn add_100() -> serde_json::Value {
        serde_json::json!({"type": "composed", "group": "c", "step": {"type": "calculate", "as_field": "count", "expr": "datum.count + 100"}})
    }

    fn count_by_g(name: Option<&str>) -> serde_json::Value {
        let mut step = serde_json::json!({"type": "aggregate", "ops": [{"field": "", "fn": "count", "as": "count"}], "groupby": ["g"]});
        if let Some(name) = name {
            step["name"] = serde_json::json!(name);
        }
        step
    }

    fn branch(name: &str, input: Option<&str>, transforms: serde_json::Value) -> serde_json::Value {
        serde_json::json!({"type": "branch", "name": name, "input": input, "transforms": transforms})
    }

    fn run(specs: &[TransformSpec]) -> PyResult<HashMap<String, RecordBatch>> {
        pyo3::Python::initialize();
        apply_transforms_named(specs, &rows(), &TransformContext::default()).map(|(outputs, ..)| outputs)
    }

    fn counts(batch: &RecordBatch) -> Vec<f64> {
        let column = batch.column_by_name("count").expect("a count column");
        let column = arrow::compute::cast(column, &DataType::Float64).unwrap();
        column.as_any().downcast_ref::<Float64Array>().unwrap().values().to_vec()
    }

    /// The wire round-trips, a named step is a copy like any other, and a
    /// branch or nested step is refused.
    #[test]
    fn a_composed_step_round_trips_and_refuses_a_step_that_is_not_one_row_step() {
        let wire = composed("datum.v > 2");
        let spec: TransformSpec = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(serde_json::to_value(&spec).unwrap(), wire);
        let named = serde_json::json!({"type": "composed", "group": "f", "step": {"type": "filter", "predicate": "true", "name": "n"}});
        assert!(serde_json::from_value::<TransformSpec>(named).is_ok(), "a named step is one row step");
        for (step, reason) in [
            (serde_json::json!({"type": "branch", "name": "b", "transforms": [{"type": "identity"}]}), "is a branch"),
            (composed("true"), "is itself composed"),
        ] {
            let err = serde_json::from_value::<TransformSpec>(serde_json::json!({"type": "composed", "group": "f", "step": step}))
                .unwrap_err()
                .to_string();
            assert!(err.contains(&format!("composed: the step of group 'f' {reason}")), "{err}");
        }
    }

    /// Where the rows at the copy carry its columns, it runs there, before
    /// the statistic. The rejected alternative runs it after the statistic,
    /// where `v` is gone and the counts are unfiltered.
    #[test]
    fn a_composed_step_runs_where_its_rows_carry_its_columns() {
        let outputs = run(&chain(serde_json::json!([composed("datum.v > 2"), count_by_g(None)]))).unwrap();
        assert_eq!(counts(&outputs[FINAL_OUTPUT_KEY]), [3.0], "only b's three rows pass v > 2");
    }

    /// Where they lack them, it runs on the first later tail that carries
    /// them: after the count, keeping the groups above one.
    #[test]
    fn a_composed_step_waits_for_the_first_tail_that_carries_its_columns() {
        let outputs = run(&chain(serde_json::json!([composed("datum.count > 2"), count_by_g(None)]))).unwrap();
        assert_eq!(counts(&outputs[FINAL_OUTPUT_KEY]), [3.0]);
    }

    /// A waiting copy runs over a named output and a secondary output made
    /// from its rows, and leaves the tail, which never carries its column,
    /// as written.
    #[test]
    fn a_waiting_composed_step_runs_over_named_and_secondary_outputs() {
        let outputs = run(&chain(serde_json::json!([composed("datum.count > 2"), count_by_g(Some("agg"))]))).unwrap();
        assert_eq!(counts(&outputs["agg"]), [3.0], "the named output is filtered");
        assert_eq!(outputs[FINAL_OUTPUT_KEY].num_rows(), 5, "the tail never carries count");

        let qq = serde_json::json!({"type": "qq", "field": "v", "emit_line": true});
        let filter = serde_json::json!({"type": "composed", "group": "f", "step": {"type": "filter", "predicate": "datum.qq_line_x_start > 1e9"}});
        let outputs = run(&chain(serde_json::json!([filter, qq]))).unwrap();
        assert_eq!(outputs[crate::transform::qq::QQ_LINE_KEY].num_rows(), 0, "the secondary output is filtered");
        assert!(outputs[FINAL_OUTPUT_KEY].num_rows() > 0, "the points never carry the line's columns");
    }

    /// A copy waiting in a published key's rows follows them into the branch
    /// that reads that key, and runs there once its statistic made the
    /// column. Rejects a wait scoped to the chain that met the copy, which
    /// leaves the count unfiltered and refuses the group as run nowhere.
    #[test]
    fn a_waiting_composed_step_follows_its_rows_into_a_reading_branch() {
        let outputs = run(&chain(serde_json::json!([
            branch("rows", None, serde_json::json!([composed("datum.count > 2")])),
            branch("stat", Some("rows"), serde_json::json!([count_by_g(None)])),
        ])))
        .unwrap();
        assert_eq!(counts(&outputs["stat"]), [3.0]);
        assert_eq!(outputs["rows"].num_rows(), 5);
    }

    /// A copy runs once per lineage: a branch reading rows a copy already
    /// ran on takes no second run, from its own chain or from a copy still
    /// waiting in the chain beside it. Rejects running every waiting copy
    /// over every later branch, which adds 100 twice to the operand that
    /// carries its own copy.
    #[test]
    fn a_composed_step_runs_at_most_once_per_lineage() {
        let outputs = run(&chain(serde_json::json!([
            add_100(),
            branch("hist", None, serde_json::json!([add_100(), count_by_g(None)])),
            branch("again", Some("hist"), serde_json::json!([add_100(), {"type": "identity"}])),
        ])))
        .unwrap();
        assert_eq!(counts(&outputs["hist"]), [102.0, 103.0]);
        assert_eq!(counts(&outputs["again"]), [102.0, 103.0], "the reading branch's copy is the same group");
    }

    /// A branch over the source starts a fresh lineage: a copy waiting in
    /// the chain beside it never reaches its rows, so an operand over other
    /// rows is never filtered by another's step. Rejects running a waiting
    /// copy over every later branch's tail.
    #[test]
    fn a_branch_over_the_source_takes_no_waiting_copy() {
        let outputs = run(&chain(serde_json::json!([
            composed("datum.count > 2"),
            branch("other", None, serde_json::json!([count_by_g(None)])),
            count_by_g(None),
        ])))
        .unwrap();
        assert_eq!(counts(&outputs[FINAL_OUTPUT_KEY]), [3.0]);
        assert_eq!(counts(&outputs["other"]), [2.0, 3.0]);
    }

    /// A named copy publishes its output under its own name, scoped in its
    /// branch, and leaves the rows the chain carries on as they were.
    #[test]
    fn a_named_composed_step_publishes_beside_the_chain() {
        let named = serde_json::json!({"type": "composed", "group": "f", "step": {"type": "filter", "predicate": "datum.v > 2", "name": "kept"}});
        let outputs = run(&chain(serde_json::json!([branch("op", None, serde_json::json!([named, count_by_g(None)]))]))).unwrap();
        assert_eq!(outputs["op_kept"].num_rows(), 3);
        assert_eq!(counts(&outputs["op"]), [2.0, 3.0], "the tail is unfiltered");
    }

    /// Two copies of one group: the chain that never carries the column is
    /// left as written, the other filtered, and nothing is refused.
    #[test]
    fn a_chain_that_never_carries_the_columns_is_left_as_written() {
        let outputs = run(&chain(serde_json::json!([
            branch("counted", None, serde_json::json!([composed("datum.count > 2"), count_by_g(None)])),
            branch("rows", None, serde_json::json!([composed("datum.count > 2"), {"type": "identity"}])),
        ])))
        .unwrap();
        assert_eq!(counts(&outputs["counted"]), [3.0]);
        assert_eq!(outputs["rows"].num_rows(), 5);
    }

    /// A group no copy ran is refused in the step's own voice over the first
    /// copy's rows. The rejected alternative leaves every chain as written
    /// and blanks nothing visibly, so a typo would go unnoticed.
    #[test]
    fn a_composed_group_no_copy_ran_is_refused() {
        let err = run(&chain(serde_json::json!([composed("datum.count > 2"), {"type": "identity"}]))).unwrap_err().to_string();
        assert_eq!(err, r#"ValueError: data_filter: column 'count' not found; available: ["g", "v"]"#);
    }

    /// A statistic reading a column a composition step dropped from its rows
    /// is refused naming that step, the column and the remedy. Rejects the
    /// statistic's own missing-column text, which names neither.
    #[test]
    fn a_statistic_reading_a_column_a_composed_step_dropped_is_refused_naming_the_step() {
        let mean = serde_json::json!({"type": "composed", "group": "m", "step": {"type": "aggregate", "ops": [{"field": "v", "fn": "mean", "as": "m"}], "groupby": ["g"]}});
        let sum = serde_json::json!({"type": "aggregate", "ops": [{"field": "v", "fn": "sum", "as": "s"}], "groupby": ["g"]});
        let err = run(&chain(serde_json::json!([mean, {"type": "identity", "statistics": true}, sum]))).unwrap_err().to_string();
        assert_eq!(
            err,
            "ValueError: stat_aggregate: column 'v' was dropped by the step written on the composition's .transform() \
             (stat_aggregate), which runs on this operand's rows before its statistics; keep 'v' in that step's \
             output, or write the step on the operands it should change instead of on the composition"
        );
    }

    /// A user step reading a column a statistic after the chain's marker
    /// computes is refused naming the step, the statistic and the remedy:
    /// the user's steps run before the statistics. Rejects the step's own
    /// missing-column text, which says nothing of why. A column no statistic
    /// computes keeps that text.
    #[test]
    fn a_user_step_reading_a_statistics_output_is_refused_with_the_remedy() {
        let count_all = serde_json::json!({"type": "aggregate", "ops": [{"field": "", "fn": "count", "as": "count_all"}], "groupby": ["g"]});
        let marker = serde_json::json!({"type": "identity", "statistics": true});
        let filter = |column: &str| serde_json::json!({"type": "filter", "predicate": format!("datum.{column} > 1")});
        let err = run(&chain(serde_json::json!([filter("count_all"), marker, count_all]))).unwrap_err().to_string();
        assert_eq!(
            err,
            "ValueError: data_filter: column 'count_all' is computed by the chart's statistics (stat_aggregate), which \
             run after the chart's .transform() steps: .transform() reads the chart's rows before its statistics. \
             Compute 'count_all' in .transform() before this step (e.g. with fm.transform_aggregate) and bind the \
             encoding to it"
        );
        let bin = serde_json::json!({"type": "bin", "field": "v", "bin_count": 2, "nice": false});
        let err = run(&chain(serde_json::json!([filter("bin_start"), marker, bin]))).unwrap_err().to_string();
        assert!(err.contains("data_filter: column 'bin_start' is computed by the chart's statistics (stat_bin)"), "{err}");
        let err = run(&chain(serde_json::json!([filter("nope"), marker, count_all]))).unwrap_err().to_string();
        assert_eq!(
            err,
            "ValueError: data_filter: column 'nope' not found; available: [\"g\", \"v\"]. The chart's statistics \
             (stat_aggregate) compute ['count_all'] after its .transform() steps, so .transform() cannot read them: \
             compute the value in .transform() before this step (e.g. with fm.transform_aggregate) and bind the \
             encoding to it",
            "a typo matching none of the statistics' own outputs still names them, with the remedy"
        );
        let outputs = run(&chain(serde_json::json!([filter("v"), marker, count_all]))).unwrap();
        assert_eq!(counts_of(&outputs[FINAL_OUTPUT_KEY], "count_all"), [1.0, 3.0], "a readable step runs before the statistic");
    }

    /// A column no statistic computes, in a chain with no statistics at all,
    /// keeps the step's own plain text: there is nothing to hint at. Rejects
    /// a fix that appends the statistics clause whenever a column is
    /// missing, regardless of whether the chain has any statistics.
    #[test]
    fn a_missing_column_in_a_chain_with_no_statistics_keeps_the_plain_text() {
        let err = run(&chain(serde_json::json!([
            {"type": "filter", "predicate": "datum.nope > 1"},
            {"type": "identity"},
        ])))
        .unwrap_err()
        .to_string();
        assert_eq!(err, r#"ValueError: data_filter: column 'nope' not found; available: ["g", "v"]"#);
    }

    /// A composed step naming a column only every met chain's statistics
    /// compute (their union) is refused over those statistics, with the
    /// composition's own remedy: name one of their columns directly, since
    /// a statistic's output is readable there. Rejects the bare not-found
    /// text, which says nothing of why `bar`'s `count` never appears.
    #[test]
    fn a_composed_group_no_copy_ran_names_every_met_chains_statistics() {
        let count_all = serde_json::json!({"type": "aggregate", "ops": [{"field": "", "fn": "count", "as": "count_all"}], "groupby": ["g"]});
        let marker = serde_json::json!({"type": "identity", "statistics": true});
        let err = run(&chain(serde_json::json!([
            branch("bar", None, serde_json::json!([composed("datum.count > 1"), marker, count_all])),
            branch("pt", None, serde_json::json!([composed("datum.count > 1")])),
        ])))
        .unwrap_err()
        .to_string();
        assert_eq!(
            err,
            "ValueError: data_filter: column 'count' not found; available: [\"g\", \"v\"]. The chart's statistics \
             (stat_aggregate) compute ['count_all'] after its .transform() steps, so .transform() cannot read them: \
             a step written on the composition runs after an operand's statistics only where it names one of their \
             columns: name one of ['count_all']"
        );
    }

    /// The union is order-independent: naming `bar`'s statistics does not
    /// depend on whether `bar` or `pt` is the first operand the group
    /// meets. The `available` half still reads the FIRST met chain's
    /// columns (here `pt`'s), since `first_rows` never changes. Rejects a
    /// fix that keeps only the first met chain's own statistics, which
    /// would leave this bare.
    #[test]
    fn a_composed_group_no_copy_ran_names_statistics_whichever_operand_meets_it_first() {
        let count_all = serde_json::json!({"type": "aggregate", "ops": [{"field": "", "fn": "count", "as": "count_all"}], "groupby": ["g"]});
        let marker = serde_json::json!({"type": "identity", "statistics": true});
        let err = run(&chain(serde_json::json!([
            branch("pt", None, serde_json::json!([composed("datum.count > 1")])),
            branch("bar", None, serde_json::json!([composed("datum.count > 1"), marker, count_all])),
        ])))
        .unwrap_err()
        .to_string();
        assert_eq!(
            err,
            "ValueError: data_filter: column 'count' not found; available: [\"g\", \"v\"]. The chart's statistics \
             (stat_aggregate) compute ['count_all'] after its .transform() steps, so .transform() cannot read them: \
             a step written on the composition runs after an operand's statistics only where it names one of their \
             columns: name one of ['count_all']"
        );
    }

    /// `with_statistics_hint` lists every distinct label once, in
    /// first-appearance order, and every distinct column once, dropping a
    /// repeat: a broken dedup or join would pass every other test here,
    /// since each uses one label and one column.
    #[test]
    fn with_statistics_hint_names_every_label_once_and_every_column_once() {
        pyo3::Python::initialize();
        let statistic_columns: Vec<(String, &'static str)> = vec![
            ("bin_start".to_string(), "stat_bin"),
            ("count".to_string(), "stat_bin"),
            ("count_all".to_string(), "stat_aggregate"),
            ("count".to_string(), "stat_bin"),
        ];
        let err = PyValueError::new_err("data_filter: column 'zzz' not found; available: [\"g\", \"v\"]");
        let hinted = with_statistics_hint(err, &statistic_columns, StatisticsRemedy::ComputeFirst);
        let text = pyo3::Python::attach(|py| hinted.value(py).to_string());
        assert_eq!(
            text,
            "data_filter: column 'zzz' not found; available: [\"g\", \"v\"]. The chart's statistics (stat_bin, \
             stat_aggregate) compute ['bin_start', 'count', 'count_all'] after its .transform() steps, so \
             .transform() cannot read them: compute the value in .transform() before this step (e.g. with \
             fm.transform_aggregate) and bind the encoding to it"
        );
    }

    /// A column a composed step dropped is refused in the reading step's own
    /// voice naming the dropping step, with no statistics hint: the chain
    /// carries a statistic after the marker (`stat_aggregate`'s `s`), but the
    /// missing `v` has nothing to do with it. Rejects a fix that wraps every
    /// error `Run::step` can raise, which would append the hint here too and
    /// blame `.transform()` for a column the composition's own step dropped.
    #[test]
    fn a_composed_dropped_read_is_refused_in_its_own_voice_with_no_statistics_hint() {
        let mean = serde_json::json!({"type": "composed", "group": "m", "step": {"type": "aggregate", "ops": [{"field": "v", "fn": "mean", "as": "m"}], "groupby": ["g"]}});
        let sum = serde_json::json!({"type": "aggregate", "ops": [{"field": "v", "fn": "sum", "as": "s"}], "groupby": ["g"]});
        let marker = serde_json::json!({"type": "identity", "statistics": true});
        let filter = serde_json::json!({"type": "filter", "predicate": "datum.v > 1"});
        let err = run(&chain(serde_json::json!([mean, filter, marker, sum]))).unwrap_err().to_string();
        assert_eq!(
            err,
            "ValueError: data_filter: column 'v' was dropped by the step written on the composition's \
             .transform() (stat_aggregate), which runs on this operand's rows before its statistics; keep 'v' \
             in that step's output, or write the step on the operands it should change instead of on the \
             composition"
        );
    }

    /// Every step refuses rows lacking the columns it names under its own
    /// label: its module's refusal and the refusals raised outside its
    /// module share one prefix. Runs one instance of every variant (the
    /// registry's coverage-guarded list) over rows holding none of the
    /// columns they name; a variant that does not refuse there is named
    /// with why. Rejects a label that drifts from its module's literal
    /// (`filter` beside `data_filter`, `Linkage` beside `linkage`).
    #[test]
    fn each_steps_label_is_the_prefix_its_own_refusal_carries() {
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![Field::new("unrelated", DataType::Utf8, false)]));
        let rows = RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(vec!["a", "b"]))]).unwrap();
        let mut refused = Vec::new();
        let mut ran = Vec::new();
        for (kind, step) in crate::transform::registry::tests::all_transform_spec_instances() {
            match step.apply_with_context(&rows, &TransformContext::default()) {
                Ok(_) => ran.push(kind),
                Err(err) => {
                    let text = pyo3::Python::attach(|py| err.value(py).to_string());
                    assert!(text.starts_with(&format!("{}:", step.label())), "{kind}: label {:?}, refusal {text}", step.label());
                    refused.push(kind);
                }
            }
        }
        assert_eq!(ran, RUN_WITHOUT_A_REFUSAL, "every other step refuses these rows, under its label");
        // `unpivot`'s registry instance melts every column; naming an id
        // column the rows lack makes it refuse.
        let unpivot: TransformSpec = serde_json::from_value(serde_json::json!(
            {"type": "unpivot", "id_vars": ["q"], "var_name": "variable", "value_name": "value"}
        ))
        .unwrap();
        let err = unpivot.apply_with_context(&rows, &TransformContext::default()).expect_err("q is missing");
        let text = pyo3::Python::attach(|py| err.value(py).to_string());
        assert!(text.starts_with(&format!("{}:", unpivot.label())), "unpivot: {text}");
    }

    /// The variants whose registry instance reads no column of its rows:
    /// `unpivot` melts whatever columns there are (checked above with an id
    /// column the rows lack), `reference_line` emits its two endpoints from
    /// its own values, and `identity` and `sample` pass rows through.
    const RUN_WITHOUT_A_REFUSAL: &[&str] = &["unpivot", "reference_line", "identity", "sample"];

    fn counts_of(batch: &RecordBatch, column: &str) -> Vec<f64> {
        let column = batch.column_by_name(column).expect("the column");
        let column = arrow::compute::cast(column, &DataType::Float64).unwrap();
        column.as_any().downcast_ref::<Float64Array>().unwrap().values().to_vec()
    }

    /// Each pipeline run judges its own copies: a context reused across
    /// runs, as the facet partition loop reuses one, does not carry a group
    /// that ran in one partition into the next.
    #[test]
    fn each_run_judges_its_own_composed_copies() {
        pyo3::Python::initialize();
        let ctx = TransformContext::default();
        let specs = chain(serde_json::json!([composed("datum.v > 2")]));
        apply_transforms_named(&specs, &rows(), &ctx).expect("v is there");
        let schema = Arc::new(Schema::new(vec![Field::new("w", DataType::Float64, false)]));
        let without_v = RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(vec![1.0]))]).unwrap();
        let err = apply_transforms_named(&specs, &without_v, &ctx).unwrap_err().to_string();
        assert!(err.contains("data_filter: column 'v' not found"), "{err}");
    }

    /// A composed step groups nothing a binned color demands, so a demand
    /// on its column never reaches it, even after the statistics marker.
    #[test]
    fn a_composed_step_takes_no_binned_color_demand() {
        let specs = chain(serde_json::json!([
            {"type": "identity", "statistics": true},
            {"type": "composed", "group": "f", "step": {"type": "aggregate", "ops": [{"field": "", "fn": "count", "as": "count"}], "groupby": ["g"]}},
        ]));
        pyo3::Python::initialize();
        let (_, lineages) = apply_transforms_named(&specs, &rows(), &TransformContext::default()).unwrap();
        assert!(!BinDemand::of(&lineages, &[(FINAL_OUTPUT_KEY, "g")]).any());
        assert!(!lineages[FINAL_OUTPUT_KEY].is_statistic(), "a composed step is the user's, not a statistic");
    }

    /// `fields_read` names each step's input columns, and a composed copy
    /// reads its step's.
    #[test]
    fn fields_read_names_each_steps_input_columns() {
        let read = |json: serde_json::Value| -> Vec<String> {
            serde_json::from_value::<TransformSpec>(json).unwrap().fields_read().into_iter().collect()
        };
        assert_eq!(read(serde_json::json!({"type": "filter", "predicate": "datum.b > 1 && datum.a"})), ["a", "b"]);
        assert_eq!(read(serde_json::json!({"type": "calculate", "as_field": "z", "expr": "datum.x * 2"})), ["x"]);
        assert_eq!(read(count_by_g(None)), ["g"]);
        assert_eq!(
            read(serde_json::json!({"type": "aggregate", "ops": [{"field": "y", "fn": "mean", "as": "m"}], "groupby": ["g"]})),
            ["g", "y"]
        );
        assert_eq!(read(serde_json::json!({"type": "qq", "field": "v", "groupby": ["h"]})), ["h", "v"]);
        assert_eq!(read(composed("datum.count > 1")), ["count"]);
        assert!(read(serde_json::json!({"type": "identity"})).is_empty());
        assert!(read(serde_json::json!({"type": "filter", "predicate": "true"})).is_empty());
    }

    /// A lowered `Bin`'s own `written_groupby` is the only part of its
    /// `groupby` [`caller_groupby`] names: a planner-built `Bin` (no
    /// `written_groupby`) names none of it, even though its `groupby` is
    /// non-empty, while a lowered `Aggregate`'s `groupby` names none of
    /// it either, and every other lowered kind (`Smooth` here) names all
    /// of it.
    #[test]
    fn a_lowered_bins_written_groupby_is_the_only_caller_written_grouping() {
        pyo3::Python::initialize();
        let marker = || serde_json::json!({"type": "identity", "statistics": true});

        let written = chain(serde_json::json!([marker(), {
            "type": "bin", "field": "v", "groupby": ["g"], "written_groupby": ["g"], "name": "b",
        }]));
        let (_, lineages) = apply_transforms_named(&written, &rows(), &TransformContext::default()).unwrap();
        assert_eq!(lineages["b"].written_groupby, vec!["g".to_string()]);

        let planned = chain(serde_json::json!([marker(), {
            "type": "bin", "field": "v", "groupby": ["g"], "name": "b",
        }]));
        let (_, lineages) = apply_transforms_named(&planned, &rows(), &TransformContext::default()).unwrap();
        assert_eq!(lineages["b"].written_groupby, Vec::<String>::new());

        let aggregate = chain(serde_json::json!([marker(), {
            "type": "aggregate", "ops": [{"field": "v", "fn": "mean", "as": "mean_v"}], "groupby": ["g"], "name": "a",
        }]));
        let (_, lineages) = apply_transforms_named(&aggregate, &rows(), &TransformContext::default()).unwrap();
        assert_eq!(lineages["a"].written_groupby, Vec::<String>::new());

        let smooth = chain(serde_json::json!([marker(), {
            "type": "smooth", "x": "v", "y": "v", "method": "loess", "bandwidth": 0.75, "degree": 2, "n": 10,
            "seed": 0, "output": "fitted", "inject_zero_ref": false, "inject_metrics": false, "groupby": ["g"], "name": "s",
        }]));
        let (_, lineages) = apply_transforms_named(&smooth, &rows(), &TransformContext::default()).unwrap();
        assert_eq!(lineages["s"].written_groupby, vec!["g".to_string()]);
    }

    /// A field an earlier step grouped by stays in `written_groupby` past a
    /// lowered statistic only when that statistic partitions by it too: an
    /// ungrouped lowered window counts across every row and drops it, the
    /// same window grouped by it keeps it, and a lowered `Smooth` keeps only
    /// its own grouping. A lowered per-row `Bin` computes each row alone and
    /// keeps every field, and a chain of the user's own steps keeps every
    /// field each of them grouped by, in order.
    #[test]
    fn a_lowered_statistic_keeps_only_the_written_grouping_it_partitions_by() {
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, false),
            Field::new("t", DataType::Utf8, false),
            Field::new("c", DataType::Utf8, false),
            Field::new("v", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "a", "b", "b", "b", "a"])),
                Arc::new(StringArray::from(vec!["x", "y", "x", "y", "x", "y"])),
                Arc::new(StringArray::from(vec!["p", "p", "p", "q", "q", "q"])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
            ],
        )
        .unwrap();
        let written = |steps: serde_json::Value| -> Vec<String> {
            let (_, lineages) = apply_transforms_named(&chain(steps), &batch, &TransformContext::default()).unwrap();
            lineages[FINAL_OUTPUT_KEY].written_groupby.clone()
        };
        let marker = || serde_json::json!({"type": "identity", "statistics": true});
        let mean_by_g = || serde_json::json!({"type": "aggregate", "ops": [{"field": "v", "fn": "mean", "as": "mean_v"}], "groupby": ["g"]});
        let rank_by = |field: &str| serde_json::json!({"type": "data_window", "ops": [{"op": "rank", "as": format!("r_{field}")}], "sort": ["v"], "groupby": [field]});
        let ecdf = |groupby: Option<Vec<&str>>| {
            let mut step = serde_json::json!({"type": "data_window", "ops": [{"op": "count", "field": "mean_v", "as": "count"}], "sort": ["mean_v"], "frame": [null, 0]});
            if let Some(groupby) = groupby {
                step["groupby"] = serde_json::json!(groupby);
            }
            step
        };

        assert_eq!(written(serde_json::json!([mean_by_g(), marker(), ecdf(None)])), Vec::<String>::new());
        assert_eq!(written(serde_json::json!([mean_by_g(), marker(), ecdf(Some(vec!["g"]))])), vec!["g".to_string()]);
        assert_eq!(
            written(serde_json::json!([rank_by("g"), marker(), {"type": "bin", "field": "v", "per_row": true}])),
            vec!["g".to_string()],
        );
        assert_eq!(written(serde_json::json!([rank_by("g"), rank_by("t")])), vec!["g".to_string(), "t".to_string()]);
        let smooth_by_c = serde_json::json!({
            "type": "smooth", "x": "v", "y": "v", "method": "loess", "bandwidth": 0.75, "degree": 2, "n": 10,
            "seed": 0, "output": "fitted", "inject_zero_ref": false, "inject_metrics": false, "groupby": ["c"],
        });
        assert_eq!(written(serde_json::json!([rank_by("g"), marker(), smooth_by_c])), vec!["c".to_string()]);
    }
}
