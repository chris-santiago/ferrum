//! Static parameter resolution helpers (D6).
//!
//! The `ParameterSpec`/`ParamKind` *types* live in `ferrum-scene` (alongside
//! `SelectionSpec` and `InteractionConfig`) so the interactive scene graph can
//! carry the declared parameters across the WASM boundary. This module hosts
//! the `ferrum-core`-only static resolver (`ParamStore`) that turns
//! `domainParam` references on scales into concrete numeric domains.
//!
//! See `design-docs/superpowers/specs/2026-06-01-d6-reactive-params-wire-contract.md`.

use std::collections::HashMap;

use ferrum_scene::{ParamKind, ParameterSpec};

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
