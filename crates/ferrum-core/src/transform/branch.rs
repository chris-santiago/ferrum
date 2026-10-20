//! Branch — a named sub-pipeline over a chosen input.
//!
//! A `Branch` runs its own `transforms` chain, with the same named, unnamed
//! and secondary-output semantics as the top-level pipeline, over one of two
//! inputs:
//!
//! - `input = None`: the original batch the pipeline was given, wherever the
//!   branch sits in the chain (a nested branch reads that same batch);
//! - `input = Some(key)`: the batch already published under `key`.
//!
//! A plain named step reads the current unnamed-chain tail instead; the two
//! rules coexist so a layer can derive its data from the source rather than
//! from another layer's chained output.
//!
//! A branch may carry an [`OperandFrame`]: a `+` operand over a different
//! frame than its composition's reads the combined batch, and the frame
//! picks that operand's own rows and columns back out of it, as the
//! branch's first act. Inside a framed branch, a nested branch with no
//! `input` reads those framed rows, not the combined batch.
//!
//! The sub-chain tail is published under `name`, unscoped. Every named or
//! secondary output inside it — a branch's inner outputs are its secondary
//! outputs — is published SCOPED under
//! [`secondary_key(Some(name), key)`](crate::transform::registry::secondary_key),
//! so two branches that reuse the same inner name (two composites of the
//! same kind over different data, e.g. `boxplot(df) + boxplot(df2)`) publish
//! two distinct keys, each carrying its own rows. A scoped
//! key can never equal the branch's own bare `name` (the rule's
//! `"{name}_{key}"` shape always has the separator), so there is no
//! collision between a branch's own name and anything it scopes. A
//! branch never advances the outer unnamed chain. Execution lives in
//! [`crate::transform::run::apply_transforms_named`].

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use arrow::array::{Array, BooleanArray, RecordBatch};
use arrow::datatypes::{Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};

use crate::transform::core::TransformSpec;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct BranchSpec {
    /// Key the sub-chain tail is published under.
    pub name: String,
    /// Published key to read; `None` reads the pipeline's original batch, or,
    /// inside a framed branch, that branch's framed rows.
    #[serde(default)]
    pub input: Option<String>,
    /// The operand's own rows and columns, picked out of the input before
    /// the sub-chain runs ([`OperandFrame`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frame: Option<OperandFrame>,
    /// The sub-chain; never empty on an unframed branch.
    pub transforms: Vec<TransformSpec>,
}

/// One `+` operand's own frame inside a combined batch. Wire:
/// `{"marker": "<column>", "renames": {"<original>": "<renamed>"}, "columns": ["..."]}`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct OperandFrame {
    /// The boolean column that is true on the rows this operand contributed.
    pub marker: String,
    /// Each column the merge renamed to keep it apart from another
    /// operand's, by its original name.
    #[serde(default)]
    pub renames: BTreeMap<String, String>,
    /// The operand frame's own columns, in order.
    pub columns: Vec<String>,
}

impl OperandFrame {
    /// The operand's own rows of `batch`, for branch `branch`: the rows its
    /// marker is true on (a false or null marker drops the row), each
    /// renamed column back under its original name (dtype and nulls kept),
    /// cut to its own columns in order, followed by each of `keep` (the
    /// facet fields of a partition run) the rows hold.
    pub(crate) fn rows(&self, branch: &str, batch: &RecordBatch, keep: &[String]) -> PyResult<RecordBatch> {
        let refuse = |what: String| PyValueError::new_err(format!("branch: '{branch}' frame {what}"));
        let columns_of = |rows: &RecordBatch| -> Vec<String> { rows.schema().fields().iter().map(|f| f.name().clone()).collect() };
        if self.columns.is_empty() {
            return Err(refuse("lists no columns; an operand's frame keeps at least one".into()));
        }
        let marker = batch
            .column_by_name(&self.marker)
            .ok_or_else(|| refuse(format!("reads marker column '{}', which the rows lack; their columns are {:?}", self.marker, columns_of(batch))))?;
        let mask = marker.as_any().downcast_ref::<BooleanArray>().ok_or_else(|| {
            refuse(format!("marker column '{}' is {}, not boolean", self.marker, marker.data_type()))
        })?;
        let mut rows = arrow::compute::filter_record_batch(batch, mask).map_err(|e| refuse(format!("cannot keep its rows: {e}")))?;
        for (original, renamed) in &self.renames {
            let column = rows.column_by_name(renamed).cloned().ok_or_else(|| {
                refuse(format!("restores '{original}' from '{renamed}', which the rows lack; their columns are {:?}", columns_of(&rows)))
            })?;
            rows = with_column(&rows, original, column).map_err(|e| refuse(format!("cannot restore '{original}': {e}")))?;
        }
        if let Some(missing) = self.columns.iter().find(|column| rows.column_by_name(column).is_none()) {
            return Err(refuse(format!("lists column '{missing}', which the operand's rows lack; their columns are {:?}", columns_of(&rows))));
        }
        let own = crate::transform::project::ProjectSpec { fields: self.columns.clone(), name: None };
        crate::transform::project::project(&own, &rows, keep)
    }
}

/// `rows` with `column` under `name`: in place of a column of that name, or
/// appended. The field is nullable, as a calculated column's is.
fn with_column(rows: &RecordBatch, name: &str, column: arrow::array::ArrayRef) -> Result<RecordBatch, arrow::error::ArrowError> {
    let schema = rows.schema();
    let mut fields: Vec<Field> = schema.fields().iter().map(|f| f.as_ref().clone()).collect();
    let mut columns = rows.columns().to_vec();
    let field = Field::new(name, column.data_type().clone(), true);
    match schema.index_of(name) {
        Ok(index) => {
            fields[index] = field;
            columns[index] = column;
        }
        Err(_) => {
            fields.push(field);
            columns.push(column);
        }
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
}

/// The batch this branch reads, before its frame: `source` with no
/// `input`, else the batch published under `input`. Refuses an unframed
/// branch with an empty sub-chain and an `input` that names no published
/// key.
pub(crate) fn resolve_input<'b>(
    spec: &BranchSpec,
    source: &'b RecordBatch,
    published: &'b HashMap<String, RecordBatch>,
) -> PyResult<&'b RecordBatch> {
    if spec.transforms.is_empty() && spec.frame.is_none() {
        return Err(PyValueError::new_err(format!(
            "branch: '{}' has no transforms; a branch needs at least one step or a frame",
            spec.name
        )));
    }
    let Some(key) = spec.input.as_deref() else {
        return Ok(source);
    };
    published.get(key).ok_or_else(|| {
        let mut available: Vec<&str> = published.keys().map(String::as_str).collect();
        available.sort_unstable();
        PyValueError::new_err(format!(
            "branch: '{}' reads input='{key}', which no earlier transform published; \
             available: {available:?}",
            spec.name
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::transform::calculate::CalculateSpec;
    use crate::transform::context::TransformContext;
    use crate::transform::core::{apply_transforms_named, FINAL_OUTPUT_KEY};
    use crate::transform::filter::FilterSpec;
    use crate::transform::swarm::SwarmSpec;
    use arrow::array::{Array, Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    /// Each key's `Swarm` record.
    fn swarms_of(lineages: &crate::transform::run::OutputLineages) -> HashMap<String, crate::transform::swarm::SwarmSpec> {
        lineages.iter().filter_map(|(key, lineage)| Some((key.clone(), lineage.swarm.clone()?))).collect()
    }

    fn batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, false),
            Field::new("v", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "a", "b", "b"])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
            ],
        )
        .unwrap()
    }

    fn keep(above: f64, name: Option<&str>) -> TransformSpec {
        TransformSpec::Filter(FilterSpec {
            predicate: format!("datum.v > {above}"),
            name: name.map(Into::into),
            param: None,
        })
    }

    fn swarm(name: Option<&str>) -> TransformSpec {
        TransformSpec::Swarm(SwarmSpec {
            category: "g".into(),
            value: "v".into(),
            point_size: 5.0,
            spacing: 1.0,
            side: Default::default(),
            orient: Default::default(),
            dodge: None,
            name: name.map(Into::into),
        })
    }

    fn branch(name: &str, input: Option<&str>, transforms: Vec<TransformSpec>) -> TransformSpec {
        TransformSpec::Branch(BranchSpec {
            name: name.into(),
            input: input.map(Into::into),
            frame: None,
            transforms,
        })
    }

    fn values(batch: &RecordBatch) -> Vec<f64> {
        let idx = batch.schema().index_of("v").unwrap();
        let col = batch.column(idx).as_any().downcast_ref::<Float64Array>().unwrap();
        (0..col.len()).map(|i| col.value(i)).collect()
    }

    fn run(specs: &[TransformSpec]) -> PyResult<crate::transform::run::NamedOutputs> {
        pyo3::Python::initialize();
        apply_transforms_named(specs, &batch(), &TransformContext::default())
    }

    fn error_text(specs: &[TransformSpec]) -> String {
        let err = run(specs).unwrap_err();
        pyo3::Python::attach(|py| err.value(py).to_string())
    }

    /// `input = None` reads the source batch even after an unnamed step has
    /// narrowed the chain, while a plain named step at the same position reads
    /// the narrowed tail. A branch that read the tail would publish two rows.
    #[test]
    fn unset_input_reads_the_original_batch_and_a_plain_named_step_reads_the_tail() {
        let specs = vec![
            keep(2.0, None),
            branch("b", None, vec![keep(0.0, None)]),
            keep(0.0, Some("plain")),
        ];
        let (outputs, _) = run(&specs).unwrap();
        assert_eq!(values(&outputs["b"]), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(values(&outputs["plain"]), [3.0, 4.0]);
        assert_eq!(values(&outputs[FINAL_OUTPUT_KEY]), [3.0, 4.0]);
    }

    /// `input = Some(k)` starts from the batch published under `k`, not from
    /// the source or the tail.
    #[test]
    fn set_input_reads_the_published_key() {
        let double = TransformSpec::Calculate(CalculateSpec {
            as_field: "w".into(),
            expr: "datum.v * 2".into(),
            name: None,
        });
        let specs = vec![keep(1.0, None), keep(3.0, Some("high")), branch("b", Some("high"), vec![double])];
        let (outputs, _) = run(&specs).unwrap();
        let out = &outputs["b"];
        assert_eq!(values(out), [4.0]);
        let w = out.column(out.schema().index_of("w").unwrap());
        assert_eq!(w.as_any().downcast_ref::<Float64Array>().unwrap().value(0), 8.0);
    }

    /// Inside a branch, unnamed steps chain, a named step reads the inner
    /// tail and publishes under its own (inner-scope) key, and a nested
    /// branch with no input reads the pipeline's original batch, not the
    /// outer branch's. Every one of "outer"'s inner keys — the named step,
    /// and the nested branches' own (already bare-published-then-rescoped)
    /// tails — reaches the TOP level scoped under "outer"; "outer" itself,
    /// the one key this chain's enclosing `run_chain` publishes, stays bare.
    #[test]
    fn the_sub_chain_has_top_level_semantics_recursively() {
        let specs = vec![
            keep(3.0, None),
            branch(
                "outer",
                None,
                vec![
                    keep(1.0, None),
                    keep(2.0, Some("inner")),
                    branch("nested", None, vec![keep(0.0, None)]),
                    branch("nested_on_inner", Some("inner"), vec![keep(3.0, None)]),
                ],
            ),
        ];
        let (outputs, _) = run(&specs).unwrap();
        let scoped = |key: &str| crate::transform::registry::secondary_key(Some("outer"), key);
        assert_eq!(values(&outputs["outer"]), [2.0, 3.0, 4.0]);
        assert_eq!(values(&outputs[&scoped("inner")]), [3.0, 4.0]);
        assert_eq!(values(&outputs[&scoped("nested")]), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(values(&outputs[&scoped("nested_on_inner")]), [4.0]);
        assert_eq!(values(&outputs[FINAL_OUTPUT_KEY]), [4.0]);
    }

    /// Two sibling branches that reuse the same inner name publish two
    /// DIFFERENT scoped keys, each with its own rows — the exact collision
    /// Branch-scoped publication removes (e.g. `boxplot(df) + boxplot(df2)`,
    /// two composites of the same kind over different data, each publishing
    /// an inner "box").
    #[test]
    fn two_sibling_branches_with_the_same_inner_name_publish_distinct_scoped_keys() {
        let specs = vec![
            branch("b1", None, vec![keep(1.0, Some("x"))]),
            branch("b2", None, vec![keep(2.0, Some("x"))]),
        ];
        let (outputs, _) = run(&specs).unwrap();
        let x1 = crate::transform::registry::secondary_key(Some("b1"), "x");
        let x2 = crate::transform::registry::secondary_key(Some("b2"), "x");
        assert_ne!(x1, x2);
        assert_eq!(values(&outputs[&x1]), [2.0, 3.0, 4.0]);
        assert_eq!(values(&outputs[&x2]), [3.0, 4.0]);
    }

    /// A branch nested inside a branch: the inner branch's own secondary
    /// output reaches the top level double-scoped (`b1_b2_k`), not single
    /// ("b2_k") or bare ("k") — one more level of `secondary_key` prefix per
    /// level of nesting the output crosses.
    #[test]
    fn a_doubly_nested_branch_publishes_its_secondary_output_double_scoped() {
        let qq: TransformSpec =
            serde_json::from_value(serde_json::json!({"type": "qq", "field": "v"})).unwrap();
        let specs = vec![branch("b1", None, vec![branch("b2", None, vec![qq])])];
        let (outputs, _) = run(&specs).unwrap();
        // b2's own tail, inserted bare into b1's child by run_chain's Branch
        // arm, then rescoped once (by b1) on the way to the top level.
        let b1_b2 = crate::transform::registry::secondary_key(Some("b1"), "b2");
        assert!(outputs.contains_key(&b1_b2), "got keys: {:?}", outputs.keys().collect::<Vec<_>>());
        // b2's own secondary output, already scoped to "b2_<key>" by b2's
        // OWN run_branch before b1 ever sees it, then rescoped once more
        // (by b1) on the way to the top level.
        let b2_qq = crate::transform::registry::secondary_key(Some("b2"), crate::transform::qq::QQ_LINE_KEY);
        let b1_b2_qq = crate::transform::registry::secondary_key(Some("b1"), &b2_qq);
        assert!(outputs.contains_key(&b1_b2_qq), "got keys: {:?}", outputs.keys().collect::<Vec<_>>());
    }

    /// An inner `Reorder(from=...)` resolves against an inner (SCOPE-LOCAL)
    /// `Linkage`'s own secondary output, exactly as the pair does at top
    /// level (`reorder.rs`'s `clustermap_chain` fixture) — both run inside
    /// the SAME child `Published`, before either's keys are rescoped on the
    /// way out, so the inner lookup sees the unscoped inner key.
    #[test]
    fn an_inner_reorder_reads_its_inner_linkages_order_output() {
        use crate::transform::linkage::{DistanceMetric, LinkageAxis, LinkageMethod, LinkageSpec};
        use crate::transform::reorder::ReorderSpec;
        use arrow::array::Float64Array as F64;

        // Two well-separated row pairs so Ward's leaf order is not the
        // identity permutation — a bug that dropped the branch-local scope
        // and fed Reorder nothing would leave rows in schema order instead.
        let schema = Arc::new(Schema::new(vec![
            Field::new("row", DataType::Utf8, false),
            Field::new("a", DataType::Float64, false),
            Field::new("b", DataType::Float64, false),
        ]));
        let clustered_batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["r0", "r1", "r2", "r3"])),
                Arc::new(F64::from(vec![0.0, 0.1, 10.0, 10.1])),
                Arc::new(F64::from(vec![0.0, 0.1, 10.0, 10.1])),
            ],
        )
        .unwrap();

        let linkage = TransformSpec::Linkage(LinkageSpec {
            method: LinkageMethod::Ward,
            metric: DistanceMetric::Euclidean,
            axis: LinkageAxis::Rows,
            z_score: None,
            standard_scale: None,
            name: Some("link".into()),
        });
        let reorder = TransformSpec::Reorder(ReorderSpec {
            by: "original_idx".into(),
            drop_index: false,
            from: Some("link_order".into()),
            axis: LinkageAxis::Rows,
            name: None,
        });
        let specs = vec![branch("clustered", None, vec![linkage, reorder])];

        pyo3::Python::initialize();
        let (outputs, _) =
            apply_transforms_named(&specs, &clustered_batch, &TransformContext::default()).unwrap();
        // The real assertion is that this ran at all: `Reorder::apply_with_outputs`
        // raises a typed "from=... not found in named outputs" error (not a
        // silent identity fallback) when the inner Linkage's "link_order"
        // isn't visible — which it would not be if the child `Published` a
        // branch's sub-chain runs against failed to expose a sibling inner
        // step's own key to a later step in the SAME branch. Every row must
        // also survive the round trip.
        let tail = &outputs["clustered"];
        assert_eq!(tail.num_rows(), 4);
        let rows = tail.column(tail.schema().index_of("row").unwrap());
        let mut rows: Vec<&str> = rows
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|v| v.unwrap())
            .collect();
        rows.sort_unstable();
        assert_eq!(rows, ["r0", "r1", "r2", "r3"]);
    }

    /// A secondary output produced inside a branch is published SCOPED under
    /// the branch's name — a branch's inner outputs are its secondary
    /// outputs — beside the branch's own (unscoped) tail.
    #[test]
    fn inner_secondary_outputs_are_published_under_their_own_keys() {
        let qq: TransformSpec =
            serde_json::from_value(serde_json::json!({"type": "qq", "field": "v"})).unwrap();
        let (outputs, _) = run(&[branch("b", None, vec![qq])]).unwrap();
        assert!(outputs.contains_key("b"));
        let scoped = crate::transform::registry::secondary_key(Some("b"), crate::transform::qq::QQ_LINE_KEY);
        assert!(outputs.contains_key(&scoped), "got keys: {:?}", outputs.keys().collect::<Vec<_>>());
    }

    /// An inner step named exactly like the branch publishes under its own
    /// SCOPED key (`"b_b"`), never the branch's bare name — which the tail
    /// alone occupies.
    #[test]
    fn an_inner_step_named_like_the_branch_publishes_under_its_scoped_key_and_the_tail_under_the_bare_name() {
        let specs = vec![branch("b", None, vec![swarm(Some("b")), keep(3.0, None)])];
        let (outputs, lineages) = run(&specs).unwrap();
        let swarms = swarms_of(&lineages);
        assert_eq!(values(&outputs["b"]), [4.0], "the tail, under the bare name");
        assert!(!swarms.contains_key("b"), "the tail descends from no Swarm");
        let scoped = crate::transform::registry::secondary_key(Some("b"), "b");
        assert_eq!(
            values(&outputs[&scoped]),
            [1.0, 2.0, 3.0, 4.0],
            "the inner named Swarm step's own output, under its scoped key"
        );
        assert!(swarms.contains_key(&scoped), "the inner step's own Swarm record follows it");
    }

    /// The swarm record follows the sub-chain: a tail after an inner unnamed
    /// Swarm carries it, a branch over a Swarm-carrying key inherits it, and
    /// neither touches the outer chain's record.
    #[test]
    fn the_swarm_record_merges_by_the_publication_rule() {
        let specs = vec![
            swarm(Some("sw")),
            branch("after_swarm", None, vec![swarm(None), keep(0.0, None)]),
            branch("from_sw", Some("sw"), vec![keep(0.0, None)]),
            keep(0.0, None),
        ];
        let (_, lineages) = run(&specs).unwrap();
        let swarms = swarms_of(&lineages);
        assert!(swarms.contains_key("after_swarm"));
        assert!(swarms.contains_key("from_sw"));
        assert!(swarms.contains_key("sw"));
        assert!(!swarms.contains_key(FINAL_OUTPUT_KEY), "a branch never advances the outer chain");
    }

    /// The outer chain's tail is the same with and without a branch in it.
    #[test]
    fn a_branch_never_advances_the_outer_chain() {
        let with = run(&[keep(1.0, None), branch("b", None, vec![keep(3.0, None)]), keep(2.0, None)]).unwrap();
        let without = run(&[keep(1.0, None), keep(2.0, None)]).unwrap();
        assert_eq!(with.0[FINAL_OUTPUT_KEY], without.0[FINAL_OUTPUT_KEY]);
    }

    #[test]
    fn an_unpublished_input_is_refused_naming_the_key_and_the_available_keys() {
        let text = error_text(&[keep(0.0, Some("z")), keep(0.0, Some("a")), branch("b", Some("missing"), vec![keep(0.0, None)])]);
        assert!(text.contains("input='missing'"), "{text}");
        assert!(text.contains(r#"available: ["a", "z"]"#), "{text}");
    }

    /// A key published later in the chain is not yet available.
    #[test]
    fn an_input_published_after_the_branch_is_refused() {
        let text = error_text(&[branch("b", Some("late"), vec![keep(0.0, None)]), keep(0.0, Some("late"))]);
        assert!(text.contains("input='late'"), "{text}");
    }

    #[test]
    fn an_empty_branch_is_refused() {
        let text = error_text(&[branch("b", None, vec![])]);
        assert!(text.contains("'b' has no transforms"), "{text}");
        let nested = error_text(&[branch("outer", None, vec![branch("inner", None, vec![])])]);
        assert!(nested.contains("'inner' has no transforms"), "{nested}");
    }

    #[test]
    fn the_wire_form_round_trips_and_refuses_unknown_fields() {
        let wire = serde_json::json!({
            "type": "branch",
            "name": "b",
            "input": null,
            "transforms": [{"type": "filter", "predicate": "datum.v > 1"}],
        });
        let spec: TransformSpec = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(spec, branch("b", None, vec![keep(1.0, None)]));
        assert_eq!(serde_json::to_value(&spec).unwrap(), wire);

        let mut unknown = wire;
        unknown["inputs"] = serde_json::json!("b");
        let err = serde_json::from_value::<TransformSpec>(unknown).unwrap_err();
        assert!(err.to_string().contains("unknown field `inputs`"), "{err}");
    }

    /// A bare `apply` runs the sub-chain over the batch it is given.
    #[test]
    fn a_bare_apply_returns_the_sub_chain_tail() {
        pyo3::Python::initialize();
        let out = branch("b", None, vec![keep(2.0, None)]).apply(&batch()).unwrap();
        assert_eq!(values(&out), [3.0, 4.0]);
    }

    // ── Operand frames ──

    /// A combined batch of two operands: the left's `g`/`v` rows, then the
    /// right's, whose `v` the merge renamed to `v__r` (the left's `v` is
    /// null there), with `right` marking the right's rows.
    fn combined() -> RecordBatch {
        use arrow::array::BooleanArray;
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, false),
            Field::new("v", DataType::Float64, true),
            Field::new("v__r", DataType::Float64, true),
            Field::new("w", DataType::Float64, true),
            Field::new("right", DataType::Boolean, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "b", "c", "c", "d"])),
                Arc::new(Float64Array::from(vec![Some(1.0), Some(2.0), None, None, None])),
                Arc::new(Float64Array::from(vec![None, None, Some(100.0), Some(101.0), Some(102.0)])),
                Arc::new(Float64Array::from(vec![None, None, Some(7.0), Some(8.0), Some(9.0)])),
                Arc::new(BooleanArray::from(vec![Some(false), Some(false), Some(true), Some(true), None])),
            ],
        )
        .unwrap()
    }

    fn framed(name: &str, transforms: Vec<TransformSpec>) -> TransformSpec {
        TransformSpec::Branch(BranchSpec {
            name: name.into(),
            input: None,
            frame: Some(right_frame()),
            transforms,
        })
    }

    fn right_frame() -> OperandFrame {
        OperandFrame {
            marker: "right".into(),
            renames: BTreeMap::from([("v".to_string(), "v__r".to_string())]),
            columns: vec!["g".into(), "v".into(), "w".into()],
        }
    }

    fn run_combined(specs: &[TransformSpec]) -> PyResult<crate::transform::run::NamedOutputs> {
        pyo3::Python::initialize();
        apply_transforms_named(specs, &combined(), &TransformContext::default())
    }

    fn column_names(batch: &RecordBatch) -> Vec<String> {
        batch.schema().fields().iter().map(|f| f.name().clone()).collect()
    }

    /// A framed branch starts from its operand's own rows: the rows its
    /// marker is true on (a null marker drops the row), `v` restored from
    /// the renamed copy, and only its own columns, in order. Rejects running
    /// the chain over the combined batch, where `v` is the left's and null.
    #[test]
    fn a_framed_branch_starts_from_its_operands_own_rows_and_columns() {
        let (outputs, _) = run_combined(&[framed("op", vec![keep(100.5, None)])]).unwrap();
        let out = &outputs["op"];
        assert_eq!(column_names(out), ["g", "v", "w"]);
        assert_eq!(values(out), [101.0]);
        assert_eq!(out.schema().field_with_name("v").unwrap().data_type(), &DataType::Float64, "the restored column keeps its dtype");
    }

    /// Inside a framed branch, a nested branch with no input reads the
    /// framed rows, before the operand's own steps. Rejects reading the
    /// combined batch, whose `v` is null on these rows.
    #[test]
    fn a_nested_branch_with_no_input_reads_the_framed_rows() {
        let nested = branch("nested", None, vec![keep(0.0, None)]);
        let (outputs, _) = run_combined(&[framed("op", vec![keep(100.5, None), nested])]).unwrap();
        assert_eq!(values(&outputs["op_nested"]), [100.0, 101.0]);
        assert_eq!(column_names(&outputs["op_nested"]), ["g", "v", "w"]);
    }

    /// A framed branch with no steps publishes its framed rows; an unframed
    /// one is still refused.
    #[test]
    fn a_framed_branch_needs_no_step() {
        let (outputs, _) = run_combined(&[framed("op", vec![])]).unwrap();
        assert_eq!(values(&outputs["op"]), [100.0, 101.0]);
    }

    /// A frame its rows cannot satisfy is refused naming the branch and what
    /// is missing.
    #[test]
    fn a_frame_the_rows_cannot_satisfy_is_refused() {
        let refusal = |frame: OperandFrame| {
            let spec = TransformSpec::Branch(BranchSpec { name: "op".into(), input: None, frame: Some(frame), transforms: vec![] });
            let err = run_combined(&[spec]).unwrap_err();
            pyo3::Python::attach(|py| err.value(py).to_string())
        };
        let text = refusal(OperandFrame { marker: "nope".into(), ..right_frame() });
        assert!(text.starts_with("branch: 'op' frame reads marker column 'nope', which the rows lack"), "{text}");
        let text = refusal(OperandFrame { marker: "w".into(), ..right_frame() });
        assert_eq!(text, "branch: 'op' frame marker column 'w' is Float64, not boolean");
        let text = refusal(OperandFrame { renames: BTreeMap::from([("v".to_string(), "v__x".to_string())]), ..right_frame() });
        assert!(text.starts_with("branch: 'op' frame restores 'v' from 'v__x', which the rows lack"), "{text}");
        let text = refusal(OperandFrame { columns: vec!["g".into(), "z".into()], ..right_frame() });
        assert!(text.starts_with("branch: 'op' frame lists column 'z', which the operand's rows lack"), "{text}");
        let text = refusal(OperandFrame { columns: vec![], ..right_frame() });
        assert_eq!(text, "branch: 'op' frame lists no columns; an operand's frame keeps at least one");
    }

    /// The frame's wire round-trips, `renames` defaults to none, and an
    /// unknown frame field is refused.
    #[test]
    fn the_frame_wire_round_trips_and_refuses_unknown_fields() {
        let wire = serde_json::json!({
            "type": "branch",
            "name": "op",
            "input": null,
            "frame": {"marker": "right", "renames": {"v": "v__r"}, "columns": ["g", "v", "w"]},
            "transforms": [],
        });
        let spec: TransformSpec = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(spec, framed("op", vec![]));
        assert_eq!(serde_json::to_value(&spec).unwrap(), wire);
        let bare: OperandFrame = serde_json::from_value(serde_json::json!({"marker": "m", "columns": ["g"]})).unwrap();
        assert!(bare.renames.is_empty());
        let err = serde_json::from_value::<OperandFrame>(serde_json::json!({"marker": "m", "columns": ["g"], "rename": {}})).unwrap_err();
        assert!(err.to_string().contains("unknown field `rename`"), "{err}");
    }
}
