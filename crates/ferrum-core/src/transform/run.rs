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

use std::collections::HashMap;
use std::rc::Rc;

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use crate::transform::branch::{self, BranchSpec};
use crate::transform::color_bin_group::{is_statistics_marker, BinDemand, StepBins};
use crate::transform::composed::{ComposedRuns, ComposedSpec, ComposedState};
use crate::transform::context::TransformContext;
use crate::transform::core::{spec_name, TransformSpec};
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
    /// The schema of the rows the lineage started from: the pipeline's
    /// source, or an operand's framed rows.
    pub(crate) source: SchemaRef,
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
    let plan = ctx.color_bin_plan.as_deref();
    let mut run = Run {
        ctx,
        demand: plan.map(|plan| &plan.demand),
        edges: plan.and_then(|plan| plan.edges.clone()),
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
    let mut run = Run { ctx, demand: None, edges: None, composed: ComposedRuns::default() };
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
    source: SchemaRef,
    /// Which composition steps ran on these rows and which still wait for a
    /// column ([`ComposedState`]).
    composed: ComposedState<'s>,
}

impl Lineage<'_> {
    /// The lineage of rows nothing has run on yet.
    fn over(batch: &RecordBatch) -> Self {
        Self { swarm: None, statistic_input: None, lowered: Vec::new(), source: batch.schema(), composed: ComposedState::default() }
    }

    fn recorded(self) -> OutputLineage {
        OutputLineage { swarm: self.swarm, statistic_input: self.statistic_input, lowered: self.lowered, source: self.source }
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
    composed: ComposedRuns<'s>,
}

impl<'s> Run<'s, '_> {
    /// Run `specs`, the chain at `path`, starting from `start`, publishing
    /// named, secondary and branch outputs into `published`; returns the
    /// chain tail. `source` is the rows a `Branch` with no `input` reads.
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
                    self.composed.met(copy, &current.batch);
                    self.place(copy, current, published)?
                }
                marker if is_statistics_marker(marker) => {
                    lowered = true;
                    current
                }
                step => {
                    if !lowered {
                        refuse_a_statistic_read(step, &current, &statistic_columns)?;
                    }
                    let lowered = (lowered && !matches!(step, TransformSpec::Identity(_))).then_some(id);
                    self.step(step, lowered, current, published)?
                }
            };
        }
        Ok(current)
    }

    /// Run one row step over `current`: publish its secondary outputs and,
    /// when it is named, its output; return the chain's new tail. `lowered`
    /// is the step's id when it is a lowered step.
    fn step(
        &mut self,
        spec: &'s TransformSpec,
        lowered: Option<StepId>,
        mut current: Rows<'s>,
        published: &mut Published<'s>,
    ) -> PyResult<Rows<'s>> {
        refuse_a_dropped_read(spec, &current)?;
        let fields = match (&lowered, self.demand) {
            (Some(id), Some(demand)) => demand.fields(id),
            _ => &[],
        };
        // Each transform sees the named outputs accumulated so far via the
        // context, which `Reorder(from=...)` reads.
        let mut step_ctx = self.ctx.clone();
        step_ctx.named_outputs = published.outputs.clone();
        step_ctx.color_bins = (!fields.is_empty())
            .then(|| Rc::new(StepBins { fields: fields.to_vec(), edges: self.edges.clone() }));
        let result = spec.apply_with_context(&current.batch, &step_ctx)?;
        let own = lowered.map(|id| LoweredStep { id, groupby: spec.groupby().map(<[String]>::to_vec).unwrap_or_default(), shared: false });
        // Secondary outputs first, so a named step's explicit `name` wins a
        // key collision. A secondary output descends from no Swarm.
        for (key, batch) in spec.secondary_outputs(&current.batch, &result, &step_ctx)? {
            let lineage = Lineage { swarm: None, ..forked(&current, own.clone()) };
            let rows = self.fork(&mut current, Rows { batch, lineage }, published)?;
            published.insert(key, rows);
        }
        match spec_name(spec) {
            Some(name) => {
                let lineage = with_swarm(forked(&current, own), spec);
                let rows = self.fork(&mut current, Rows { batch: result, lineage }, published)?;
                published.insert(name.to_string(), rows);
                Ok(current)
            }
            None => {
                let mut lineage = current.lineage.clone();
                if let Some(own) = own {
                    lineage.statistic_input.get_or_insert_with(|| current.batch.clone());
                    lineage.lowered.push(own);
                }
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
        let out = copy.step.apply_with_context(&rows.batch, &step_ctx)?;
        self.composed.ran(&copy.group);
        let Rows { batch, mut lineage } = rows;
        match spec_name(&copy.step) {
            Some(name) => {
                lineage.composed.ran(copy, None);
                published.insert(name.to_string(), Rows { batch: out, lineage: lineage.clone() });
                Ok(Rows { batch, lineage })
            }
            None => {
                lineage.composed.ran(copy, Some((&batch.schema(), &out.schema())));
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
                let lineage = Lineage { source: batch.schema(), ..start.lineage };
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

/// The lineage of an output a step makes from `current` beside the chain
/// (a named or secondary output), `own` the step itself when it is lowered:
/// the tail's lowered steps reach it as shared, and its own statistic, when
/// it is one, consumed the tail's rows.
fn forked<'s>(current: &Rows<'s>, own: Option<LoweredStep>) -> Lineage<'s> {
    let mut lineage = current.lineage.clone();
    for step in &mut lineage.lowered {
        step.shared = true;
    }
    if let Some(own) = own {
        lineage.statistic_input = Some(current.batch.clone());
        lineage.lowered.push(own);
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
/// computes, with the step: what the chain's own steps before the marker
/// cannot read, since they run first.
fn statistic_columns(chain: &[TransformSpec]) -> Vec<(String, &TransformSpec)> {
    let Some(marker) = chain.iter().position(is_statistics_marker) else { return Vec::new() };
    chain[marker + 1..].iter().flat_map(|step| step.output_columns().into_iter().map(move |column| (column, step))).collect()
}

/// Refuse `spec`, one of the user's own steps, when it reads a column its
/// rows lack that a statistic later in its chain computes: the user's steps
/// run before the chart's statistics, so the step would otherwise refuse in
/// its own voice with no word of why the column is missing.
fn refuse_a_statistic_read(spec: &TransformSpec, rows: &Rows<'_>, statistic_columns: &[(String, &TransformSpec)]) -> PyResult<()> {
    if statistic_columns.is_empty() {
        return Ok(());
    }
    let schema = rows.batch.schema();
    let read = spec.fields_read().into_iter().filter(|field| schema.index_of(field).is_err()).find_map(|field| {
        statistic_columns.iter().find(|(column, _)| *column == field).map(|(_, statistic)| (field, *statistic))
    });
    match read {
        None => Ok(()),
        Some((column, statistic)) => Err(PyValueError::new_err(format!(
            "{}: column '{column}' is computed by the chart's statistics ({}), which run after the chart's \
             .transform() steps: .transform() reads the chart's rows before its statistics. Compute '{column}' \
             in .transform() before this step (e.g. with fm.transform_aggregate) and bind the encoding to it",
            spec.label(),
            statistic.label(),
        ))),
    }
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
        assert!(err.contains("data_filter: column 'nope' not found"), "{err}");
        let outputs = run(&chain(serde_json::json!([filter("v"), marker, count_all]))).unwrap();
        assert_eq!(counts_of(&outputs[FINAL_OUTPUT_KEY], "count_all"), [1.0, 3.0], "a readable step runs before the statistic");
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
}
