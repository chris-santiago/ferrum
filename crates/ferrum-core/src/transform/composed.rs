//! Composed — a step written on a `+` composition, placed by row lineage.
//!
//! A composition's `.transform(step)` reaches each operand it covers as one
//! copy in that operand's chain, all copies sharing the step's `group`.
//! Python places each copy after the operand's own steps, before its
//! statistics; Python cannot see a statistic's output columns, so the
//! runner ([`crate::transform::run`]) decides where the copy runs from the
//! rows it meets, by the rows' lineage:
//!
//! - when the rows at the copy carry every column the step reads
//!   ([`TransformSpec::fields_read`]), it runs there;
//! - otherwise it waits in the rows' lineage, and runs on the first later
//!   rows of that lineage that carry them: the chain's tail, a named or
//!   secondary output made from those rows, or the rows of a branch that
//!   reads them through its `input`;
//! - a group runs at most once per lineage: rows that a copy of the group
//!   ran on, or that a copy waits on, take no other copy of it. A branch
//!   over the source starts a fresh lineage, so nothing waiting in one
//!   operand's rows reaches another's.
//!
//! An unnamed step changes the rows it runs on. A named step publishes its
//! output under its name and leaves the rows as they were, once per chain.
//!
//! A group whose every copy ran nowhere is malformed: the run refuses it in
//! the step's own voice over the rows its first copy met ([`ComposedRuns`]).
//!
//! The step is never a `Branch` and never another `Composed`: it is one of
//! the user's row steps. It takes no binned-color demand, since it groups
//! nothing the demand reads ([`TransformSpec::groupby`] is `None`).

use arrow::array::RecordBatch;
use arrow::datatypes::Schema;
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};

use crate::transform::context::TransformContext;
use crate::transform::core::{spec_name, TransformSpec};

/// One copy of a composition's step. Wire:
/// `{"type": "composed", "group": "<id>", "step": {<the step's own wire>}}`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(try_from = "ComposedWire", into = "ComposedWire")]
pub(crate) struct ComposedSpec {
    /// The written step's identity, shared by every copy of it.
    pub group: String,
    /// The step itself: never a `Branch` or a `Composed`.
    pub step: Box<TransformSpec>,
}

/// The serde shape of [`ComposedSpec`], checked on the way in.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ComposedWire {
    group: String,
    step: Box<TransformSpec>,
}

impl TryFrom<ComposedWire> for ComposedSpec {
    type Error = String;

    fn try_from(wire: ComposedWire) -> Result<Self, Self::Error> {
        let refusal = match &*wire.step {
            TransformSpec::Branch(_) => Some("is a branch"),
            TransformSpec::Composed(_) => Some("is itself composed"),
            _ => None,
        };
        match refusal {
            Some(reason) => Err(format!(
                "composed: the step of group '{}' {reason}; a composition's step is one row step",
                wire.group
            )),
            None => Ok(Self { group: wire.group, step: wire.step }),
        }
    }
}

impl From<ComposedSpec> for ComposedWire {
    fn from(spec: ComposedSpec) -> Self {
        Self { group: spec.group, step: spec.step }
    }
}

impl ComposedSpec {
    /// Whether rows of `schema` carry every column the step reads.
    pub(crate) fn readable_in(&self, schema: &Schema) -> bool {
        self.step.fields_read().iter().all(|field| schema.index_of(field).is_ok())
    }
}

/// The composition steps one lineage met ([`crate::transform::run`]'s
/// `Lineage`): the groups that ran on its rows, the copies still waiting
/// for a column, and the columns an unnamed copy dropped from its rows.
#[derive(Debug, Clone, Default)]
pub(crate) struct ComposedState<'s> {
    ran: Vec<&'s str>,
    waiting: Vec<&'s ComposedSpec>,
    dropped: Vec<(String, &'s ComposedSpec)>,
}

impl<'s> ComposedState<'s> {
    /// Whether a copy of `group` ran on these rows or waits on them.
    pub(crate) fn holds(&self, group: &str) -> bool {
        self.ran.contains(&group) || self.waiting.iter().any(|copy| copy.group == group)
    }

    /// Leave `copy` waiting for a column these rows lack.
    pub(crate) fn wait(&mut self, copy: &'s ComposedSpec) {
        self.waiting.push(copy);
    }

    /// The waiting copies, in the order the lineage met them, taken out to
    /// be run or put back.
    pub(crate) fn take_waiting(&mut self) -> Vec<&'s ComposedSpec> {
        std::mem::take(&mut self.waiting)
    }

    /// Record that `copy` ran on these rows. `schemas` is the input and
    /// output schema of an unnamed copy, whose dropped columns are recorded;
    /// `None` for a named one, which left the rows as they were.
    pub(crate) fn ran(&mut self, copy: &'s ComposedSpec, schemas: Option<(&Schema, &Schema)>) {
        self.waiting.retain(|waiting| waiting.group != copy.group);
        if !self.ran.contains(&copy.group.as_str()) {
            self.ran.push(copy.group.as_str());
        }
        if let Some((input, output)) = schemas {
            for field in input.fields() {
                if output.index_of(field.name()).is_err() {
                    self.dropped.push((field.name().clone(), copy));
                }
            }
        }
    }

    /// Record as run here every named copy that ran on `out`, an output made
    /// from these rows: a named copy publishes once per chain.
    pub(crate) fn adopt_named_runs(&mut self, out: &ComposedState<'s>) {
        let named: Vec<&'s ComposedSpec> = self
            .waiting
            .iter()
            .copied()
            .filter(|copy| spec_name(&copy.step).is_some() && out.ran.contains(&copy.group.as_str()))
            .collect();
        for copy in named {
            self.ran(copy, None);
        }
    }

    /// The copy that dropped `column` from these rows, if one did.
    pub(crate) fn dropped(&self, column: &str) -> Option<&'s ComposedSpec> {
        self.dropped.iter().find(|(dropped, _)| dropped == column).map(|(_, copy)| *copy)
    }
}

/// Per pipeline run, every composed group in the order the run met it: its
/// first copy, the rows that copy met, and whether any copy ran. The first
/// copy a run meets is its first operand's, so a refusal reads that
/// operand's rows.
#[derive(Debug, Default)]
pub(crate) struct ComposedRuns<'s> {
    groups: Vec<GroupRun<'s>>,
}

#[derive(Debug)]
struct GroupRun<'s> {
    copy: &'s ComposedSpec,
    first_rows: RecordBatch,
    ran: bool,
}

impl<'s> ComposedRuns<'s> {
    /// Record that `copy` met `rows`; the first copy's rows stand.
    pub(crate) fn met(&mut self, copy: &'s ComposedSpec, rows: &RecordBatch) {
        if self.groups.iter().all(|run| run.copy.group != copy.group) {
            self.groups.push(GroupRun { copy, first_rows: rows.clone(), ran: false });
        }
    }

    /// Record that a copy of `group` ran.
    pub(crate) fn ran(&mut self, group: &str) {
        if let Some(run) = self.groups.iter_mut().find(|run| run.copy.group == group) {
            run.ran = true;
        }
    }

    /// Refuse the first group no copy ran, in the step's own voice over its
    /// first copy's rows: running it there raises the refusal a missing
    /// column draws (`data_filter: column 'count' not found; available:
    /// [...]`). That run's warnings go to a sink of its own, since its
    /// output is thrown away. A step that would not refuse there is named
    /// by the first column those rows lack.
    pub(crate) fn refuse_unrun(&self, ctx: &TransformContext) -> PyResult<()> {
        let Some(run) = self.groups.iter().find(|run| !run.ran) else {
            return Ok(());
        };
        let quiet = TransformContext { warnings: Default::default(), ..ctx.clone() };
        run.copy.step.apply_with_context(&run.first_rows, &quiet)?;
        let schema = run.first_rows.schema();
        let missing = run.copy.step.fields_read().into_iter().find(|field| schema.index_of(field).is_err());
        let available: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        Err(PyValueError::new_err(format!(
            "{}: column '{}' not found; available: {available:?}",
            run.copy.step.label(),
            missing.unwrap_or_default()
        )))
    }
}
