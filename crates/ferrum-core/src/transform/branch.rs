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
//! The sub-chain tail is published under `name`, and every named or
//! secondary output inside it under its own key; on a key collision `name`
//! wins. A branch never advances the outer unnamed chain. Execution lives in
//! [`crate::transform::core::apply_transforms_named`].

use std::collections::HashMap;

use arrow::array::RecordBatch;
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};

use crate::transform::core::TransformSpec;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct BranchSpec {
    /// Key the sub-chain tail is published under.
    pub name: String,
    /// Published key to read; `None` reads the pipeline's original batch.
    #[serde(default)]
    pub input: Option<String>,
    /// The sub-chain; never empty.
    pub transforms: Vec<TransformSpec>,
}

/// The batch this branch's sub-chain starts from, refusing an empty
/// sub-chain and an `input` that names no published key.
pub(crate) fn resolve_input<'b>(
    spec: &BranchSpec,
    source: &'b RecordBatch,
    published: &'b HashMap<String, RecordBatch>,
) -> PyResult<&'b RecordBatch> {
    if spec.transforms.is_empty() {
        return Err(PyValueError::new_err(format!(
            "branch: '{}' has no transforms; a branch needs at least one step",
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
            transforms,
        })
    }

    fn values(batch: &RecordBatch) -> Vec<f64> {
        let idx = batch.schema().index_of("v").unwrap();
        let col = batch.column(idx).as_any().downcast_ref::<Float64Array>().unwrap();
        (0..col.len()).map(|i| col.value(i)).collect()
    }

    fn run(specs: &[TransformSpec]) -> PyResult<crate::transform::core::NamedOutputs> {
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
    /// tail and publishes under its own key, and a nested branch with no
    /// input reads the pipeline's original batch, not the outer branch's.
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
        assert_eq!(values(&outputs["outer"]), [2.0, 3.0, 4.0]);
        assert_eq!(values(&outputs["inner"]), [3.0, 4.0]);
        assert_eq!(values(&outputs["nested"]), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(values(&outputs["nested_on_inner"]), [4.0]);
        assert_eq!(values(&outputs[FINAL_OUTPUT_KEY]), [4.0]);
    }

    /// A secondary output produced inside a branch is published under its own
    /// key beside the branch tail.
    #[test]
    fn inner_secondary_outputs_are_published_under_their_own_keys() {
        let qq: TransformSpec =
            serde_json::from_value(serde_json::json!({"type": "qq", "field": "v"})).unwrap();
        let (outputs, _) = run(&[branch("b", None, vec![qq])]).unwrap();
        assert!(outputs.contains_key("b"));
        assert!(outputs.contains_key(crate::transform::qq::QQ_LINE_KEY));
    }

    /// On a key collision the branch's own `name` wins over an inner named
    /// step, for both the batch and the swarm record.
    #[test]
    fn the_branch_name_wins_a_key_collision() {
        let specs = vec![branch("b", None, vec![swarm(Some("b")), keep(3.0, None)])];
        let (outputs, swarms) = run(&specs).unwrap();
        assert_eq!(values(&outputs["b"]), [4.0]);
        assert!(!swarms.contains_key("b"), "the tail descends from no Swarm");
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
        let (_, swarms) = run(&specs).unwrap();
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
}
