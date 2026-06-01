//! Reactive parameter specs (D6).
//!
//! A `ParameterSpec` is the lean Rust mirror of a Python `Parameter`
//! (`fm.param` variable, or a `Selection`). The static SVG renderer only needs
//! a parameter's *initial* value to resolve `domainParam` references on scales;
//! full selection projection lives in the `selections` section read by WASM.
//!
//! These types live in `ferrum-scene` (alongside `SelectionSpec` and
//! `InteractionConfig`) so the interactive scene graph can carry the declared
//! parameters across the WASM boundary. `ferrum-core` re-exports them for use
//! in `ChartSpec` and the static scale resolver.
//!
//! See `design-docs/superpowers/specs/2026-06-01-d6-reactive-params-wire-contract.md`.

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

    /// The serde shape is unchanged from the pre-move ferrum-core definition:
    /// a variable with a numeric-array `value` parses from snake_case `kind`.
    #[test]
    fn variable_param_parses_canonical_json() {
        let json = r#"{"name":"d","kind":"variable","value":[0,100]}"#;
        let parsed: ParameterSpec = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.name, "d");
        assert_eq!(parsed.kind, ParamKind::Variable);
        assert_eq!(parsed.value, Some(serde_json::json!([0, 100])));
        assert_eq!(parsed.bind, None);
        assert_eq!(parsed.select, None);
    }
}
