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

/// What aspect of the scene a parameter drives. Read by the WASM runtime to
/// route a parameter's live updates to the right reactive machinery.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum BindingRole {
    /// Reactive rescale: the param feeds a scale's domain on the target panel.
    Domain,
    /// Crossfilter: the param drives a `filter` transform on the target panel.
    Filter,
    /// Legend toggle: a `bind="legend"` point selection.
    Legend,
}

/// A single param→scene binding, emitted into `InteractionConfig` so the WASM
/// runtime knows which panel/scale a declared parameter drives. The static
/// resolver substitutes `domainParam` into a concrete domain (and clears the
/// reference) before the scene exists, so this is the only record the runtime
/// has of the connection.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ParamBinding {
    pub param: String,
    pub role: BindingRole,
    /// Target panel for `Domain`/`Filter` bindings; `None` for `Legend`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub panel: Option<usize>,
    /// Channel wire name ("x"/"y"/"color"/…) for `Domain` bindings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channel: Option<String>,
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

    #[test]
    fn domain_binding_round_trips() {
        let binding = ParamBinding {
            param: "d".into(),
            role: BindingRole::Domain,
            panel: Some(0),
            channel: Some("x".into()),
        };
        let json = serde_json::to_string(&binding).unwrap();
        assert_eq!(
            json,
            r#"{"param":"d","role":"domain","panel":0,"channel":"x"}"#
        );
        let parsed: ParamBinding = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, binding);
    }

    #[test]
    fn filter_binding_omits_channel() {
        let binding = ParamBinding {
            param: "brush".into(),
            role: BindingRole::Filter,
            panel: Some(2),
            channel: None,
        };
        let json = serde_json::to_string(&binding).unwrap();
        assert_eq!(json, r#"{"param":"brush","role":"filter","panel":2}"#);
        let parsed: ParamBinding = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, binding);
    }

    #[test]
    fn legend_binding_omits_panel_and_channel() {
        let binding = ParamBinding {
            param: "sel".into(),
            role: BindingRole::Legend,
            panel: None,
            channel: None,
        };
        let json = serde_json::to_string(&binding).unwrap();
        assert_eq!(json, r#"{"param":"sel","role":"legend"}"#);
        let parsed: ParamBinding = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, binding);
    }
}
