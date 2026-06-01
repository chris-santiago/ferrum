//! Reactive parameter specs (D6).
//!
//! A `ParameterSpec` is the lean Rust mirror of a Python `Parameter`
//! (`fm.param` variable, or a `Selection`). The static SVG renderer only needs
//! a parameter's *initial* value to resolve `domainParam` references on scales;
//! full selection projection lives in the `selections` section read by WASM.
//!
//! See `design-docs/superpowers/specs/2026-06-01-d6-reactive-params-wire-contract.md`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ParamKind {
    Variable,
    Point,
    Interval,
}

/// A single declared parameter. `value`/`bind`/`select` are opaque JSON to the
/// static renderer; only a variable's `value` is consulted (for `domainParam`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ParameterSpec {
    pub name: String,
    pub kind: ParamKind,
    /// Variable initial value (numbers, arrays, strings, ...). `None` for
    /// selections, whose static value is always the empty selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bind: Option<serde_json::Value>,
    /// Opaque selection projection; WASM-bound, inert statically.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub select: Option<serde_json::Value>,
}

/// Name-indexed view over a chart's declared parameters, used by the static
/// scale resolver to turn `domainParam` references into concrete domains.
pub(crate) struct ParamStore<'a> {
    by_name: HashMap<&'a str, &'a ParameterSpec>,
}

impl<'a> ParamStore<'a> {
    pub(crate) fn new(params: &'a [ParameterSpec]) -> Self {
        let by_name = params.iter().map(|p| (p.name.as_str(), p)).collect();
        ParamStore { by_name }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }

    /// Static numeric domain for a parameter reference.
    ///
    /// Returns `Some(vec)` only when the named parameter is a variable whose
    /// `value` is a JSON array of at least two numbers. Selections (point /
    /// interval) and non-array / too-short / non-numeric values yield `None`,
    /// which the resolver treats as "auto-infer the domain from data" — the
    /// correct static semantics for an empty selection.
    pub(crate) fn numeric_domain(&self, name: &str) -> Option<Vec<f64>> {
        let param = self.by_name.get(name)?;
        if param.kind != ParamKind::Variable {
            return None;
        }
        let arr = param.value.as_ref()?.as_array()?;
        if arr.len() < 2 {
            return None;
        }
        arr.iter().map(|v| v.as_f64()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variable_param_round_trips() {
        let spec = ParameterSpec {
            name: "thresh".into(),
            kind: ParamKind::Variable,
            value: Some(serde_json::json!([0.0, 100.0])),
            bind: None,
            select: None,
        };
        let json = serde_json::to_string(&spec).unwrap();
        let parsed: ParameterSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, spec);
    }

    #[test]
    fn interval_param_round_trips() {
        let spec = ParameterSpec {
            name: "brush".into(),
            kind: ParamKind::Interval,
            value: None,
            bind: None,
            select: Some(serde_json::json!({"encodings": ["x"]})),
        };
        let json = serde_json::to_string(&spec).unwrap();
        let parsed: ParameterSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, spec);
        assert_eq!(json, r#"{"name":"brush","kind":"interval","select":{"encodings":["x"]}}"#);
    }

    #[test]
    fn numeric_domain_variable_array() {
        let params = vec![ParameterSpec {
            name: "d".into(),
            kind: ParamKind::Variable,
            value: Some(serde_json::json!([0, 100])),
            bind: None,
            select: None,
        }];
        let store = ParamStore::new(&params);
        assert_eq!(store.numeric_domain("d"), Some(vec![0.0, 100.0]));
    }

    #[test]
    fn numeric_domain_variable_scalar_is_none() {
        let params = vec![ParameterSpec {
            name: "d".into(),
            kind: ParamKind::Variable,
            value: Some(serde_json::json!(50)),
            bind: None,
            select: None,
        }];
        let store = ParamStore::new(&params);
        assert_eq!(store.numeric_domain("d"), None);
    }

    #[test]
    fn numeric_domain_selection_is_none() {
        let params = vec![
            ParameterSpec {
                name: "p".into(),
                kind: ParamKind::Point,
                value: None,
                bind: None,
                select: None,
            },
            ParameterSpec {
                name: "i".into(),
                kind: ParamKind::Interval,
                value: Some(serde_json::json!([0, 100])),
                bind: None,
                select: None,
            },
        ];
        let store = ParamStore::new(&params);
        assert_eq!(store.numeric_domain("p"), None);
        // interval with an array value is still a selection → None
        assert_eq!(store.numeric_domain("i"), None);
    }

    #[test]
    fn numeric_domain_missing_name_is_none() {
        let store = ParamStore::new(&[]);
        assert!(store.is_empty());
        assert_eq!(store.numeric_domain("nope"), None);
    }

    #[test]
    fn numeric_domain_too_short_is_none() {
        let params = vec![ParameterSpec {
            name: "d".into(),
            kind: ParamKind::Variable,
            value: Some(serde_json::json!([42])),
            bind: None,
            select: None,
        }];
        let store = ParamStore::new(&params);
        assert_eq!(store.numeric_domain("d"), None);
    }
}
