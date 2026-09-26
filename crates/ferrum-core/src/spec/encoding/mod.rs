//! Encoding-channel spec: `DataType`, the 16-variant `ScaleSpec` enum, and
//! `EncodingSpec`/`Encoding` (the `Chart.encode(...)` IR). The scale
//! wire-key gate — `ScaleSpec`'s own `Serialize`/`Deserialize` impls, the
//! accepted-key tables, `scale_accepted_keys`/`validate_scale_dict`, and the
//! raw-dict temporal-domain conversion — lives in `gate.rs` instead: wave 6A
//! `#158` split this module past the CLAUDE.md test-vs-source promotion
//! trigger (source 1454 lines / tests 1626 lines), and the gate cluster is a
//! single cohesive chokepoint with its own tests, documented in `gate.rs`'s
//! own doc comment. `gate.rs` marks `deserialize_scale_value` `pub(super)`
//! so this file's `EncodingSpec::new` below can call it directly; every
//! other gate item stays private to `gate.rs`. `DataType`/`EncodingSpec`/
//! `Encoding` test coverage (everything except the gate cluster) lives in
//! `tests.rs`, mirroring `render/scale_resolve/` and `transform/stats/`.

mod gate;
pub(crate) mod sort;
#[cfg(test)]
mod tests;

pub use gate::{
    scale_accepted_keys, scale_temporal_type_tags, scale_type_tags, sort_accepted_vocabulary,
    validate_scale_dict,
};
use gate::deserialize_scale_value;

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DataType {
    Quantitative,
    Nominal,
    Ordinal,
    Temporal,
}

impl DataType {
    pub fn as_str(&self) -> &'static str {
        match self {
            DataType::Quantitative => "quantitative",
            DataType::Nominal => "nominal",
            DataType::Ordinal => "ordinal",
            DataType::Temporal => "temporal",
        }
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug)]
pub struct ParseDataTypeError(pub String);

impl fmt::Display for ParseDataTypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unknown data type '{}'; expected one of [Q, N, O, T, quantitative, nominal, ordinal, temporal]",
            self.0
        )
    }
}

impl std::error::Error for ParseDataTypeError {}

impl FromStr for DataType {
    type Err = ParseDataTypeError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Q" | "quantitative" => Ok(DataType::Quantitative),
            "N" | "nominal" => Ok(DataType::Nominal),
            "O" | "ordinal" => Ok(DataType::Ordinal),
            "T" | "temporal" => Ok(DataType::Temporal),
            other => Err(ParseDataTypeError(other.to_string())),
        }
    }
}

/// Shared fields across all 7 continuous `ScaleSpec` variants (Linear, Log, Time,
/// Symlog, Pow, Sqrt, Utc).  Embedded via `#[serde(flatten)]` so these fields
/// appear at the same JSON level as the variant-specific fields (no `"common"` key).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContinuousScaleCommon {
    /// Explicit data domain [min, max].  Auto-inferred from the column when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domain: Option<Vec<f64>>,
    /// Pixel range [lo, hi].  Defaults to the plot extent when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range: Option<Vec<f64>>,
    /// Clamp output to the range bounds.
    #[serde(default)]
    pub clamp: bool,
    /// Fractional inward pixel padding (0.0 = no padding).  Themes-T4
    /// quantitative default is 0.05, applied at the renderer when
    /// `padding.is_none()` and `domain.is_none()`.  User-specified
    /// `domain` suppresses the default to 0.0 unless `padding` is also set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub padding: Option<f64>,
    /// Color scheme name for continuous color scales (e.g. `"viridis"`, `"blues"`,
    /// `"rdbu"`).  Honored by `build_color_scale` when the encoding's color field
    /// is quantitative.  Takes precedence over the theme's sequential scheme.
    ///
    /// This field lives on `ContinuousScaleCommon` (not on each variant separately)
    /// so that `{"type": "linear", "scheme": "blues"}` round-trips through serde
    /// without the scheme being silently dropped (D4 fix, 2026-05-31).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scheme: Option<String>,
    /// Reactive-rescale reference (D6): names a parameter whose static value
    /// (a numeric array) supplies this scale's domain. A sibling of `domain`
    /// rather than a retyped `domain`, so every scale struct stays byte-stable
    /// when no parameter is referenced. The static resolver substitutes the
    /// parameter's value into `domain` (and clears this field) before scale
    /// resolution; WASM reads it to rescale live.
    #[serde(rename = "domainParam", default, skip_serializing_if = "Option::is_none")]
    pub domain_param: Option<String>,
    /// Domain-swap sugar (F-L04-07): after domain resolution, `reverse=true`
    /// swaps the resolved domain pair. For an explicit `domain=[a, b]` with
    /// `zero=false`, this is exactly equivalent to writing `domain=[b, a]`
    /// by hand; it is NOT exactly equivalent in general — an auto-inferred
    /// domain keeps the default padding inset an explicit `domain=` would
    /// suppress, and `Linear`'s `zero=true` extension runs before the swap
    /// (see `apply_domain_reverse`'s doc in `render::scale_resolve::positional`
    /// for why reversing first would collapse the domain instead). Range
    /// orientation, the structural y-inversion predicate, and tick
    /// label/fraction pairing are untouched; this only flips which end of
    /// the domain lands at which end of the range.
    /// `#[serde(default, skip_serializing_if = "std::ops::Not::not")]` (the
    /// `Layer::independent_y` idiom) so every pre-existing wire form
    /// serializes byte-identically when `reverse` is unset (spec §6, decision
    /// 5: domain-swap over range-swap since descending domains are already
    /// tolerated, tested, and normalized everywhere downstream).
    ///
    /// **Positional resolution only.** `ContinuousScaleCommon` is also read
    /// by non-positional channels that only pull `domain`/`range` from it
    /// and never consult this field: color (`build_color_scale`, which
    /// honors `reverse` solely on the separate `ScaleSpec::Sequential`
    /// variant — `ScaleSpec::Diverging` has no `reverse` field at all,
    /// documented at `scale_spec_is_reversed`'s doc in
    /// `scale_resolve::color`) and size/opacity (`linear_overrides` in
    /// `scale_resolve::auxiliary`). A raw-dict color- or size-channel scale
    /// like `{"type": "linear", "reverse": true}` deserializes this field
    /// (the batch-C task 4 scale-key gate accepts `reverse` on `Linear`/
    /// `Log`/etc. unconditionally — it is keyed off scale *type*, not
    /// encoding *channel*, and cannot see which channel a scale is attached
    /// to) and it is silently inert there today — the same silent-no-op
    /// class F-L04-07 closes for typos, still present here because it is an
    /// inertness, not an unrecognized key. **Known, documented limit of the
    /// gate, not a gap it was scoped to close**: see
    /// `accepted_keys_for_scale_type`'s doc below.
    ///
    /// `{"type": "diverging", "reverse": true}` is a different, THIRD case,
    /// and the gate DOES close it: `Diverging` has no `reverse` field to
    /// deserialize into, so `accepted_keys_for_scale_type("diverging")`
    /// deliberately omits it, and the gate refuses the dict outright
    /// (`unknown key 'reverse' for type 'diverging'; accepted: domain,
    /// domainMid, scheme`) before serde's flatten/internal-tag machinery
    /// would otherwise have dropped the key without even a round-trip.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub reverse: bool,
}

/// Scale override on an encoding channel. Honored by scale_resolve.rs in Phase 8a.
///
/// **`ScaleSpec` is the canonical wire + render contract for scales.** It is what
/// serializes across the Arrow/JSON boundary and what the render path resolves into
/// compute scales (`ScaleKind`) in `render::scale_resolve::positional`. The resolution
/// mapping is:
///
/// | `ScaleSpec` variant | `ScaleKind` produced |
/// |---|---|
/// | `Linear` | `ScaleKind::Linear` |
/// | `Log` | `ScaleKind::Log` |
/// | `Time`, `Utc` | `ScaleKind::Time` |
/// | `Symlog` | `ScaleKind::Symlog` |
/// | `Pow` | `ScaleKind::Pow` (stored exponent) |
/// | `Sqrt` | `ScaleKind::Pow(0.5)` |
/// | `Ordinal` | `ScaleKind::Ordinal` |
/// | `Band` | `ScaleKind::Ordinal` under the band model: inner = `padding_inner.unwrap_or(padding)`, outer = `padding_outer.unwrap_or(padding)` (d3's `padding` shorthand sets BOTH sides), `align` threaded |
/// | `Point` | `ScaleKind::Ordinal` under the point model: `padding` (an end padding) and `align` per the point formula; `reverse` reverses the resolved domain order post-sort, GH #65 |
/// | `Sequential`, `Diverging`, `Quantize` | `ScaleKind::Linear` (positional fallback) |
/// | `BinOrdinal` | `ScaleKind::Linear` |
///
/// **Dual-representation link, now single-sourced (SPEC-04).** The user-facing
/// PyO3 `*Scale` classes in `crate::scale` (BandScale, PointScale, QuantileScale,
/// etc.) remain a **separate construction surface** — thin builders with their
/// own field sets, defaults, and validation, used for direct Python compute
/// (`.bandwidth()`, `.scale()`, `.ticks()`) rather than for rendering. But each
/// pyclass now exposes an inherent `to_scale_spec(&self) -> ScaleSpec`, so the
/// wire form of a `*Scale` instance is emitted from one place next to this enum
/// instead of being hand-copied in Python. Extending a `ScaleSpec` variant breaks
/// its `to_scale_spec` builder until updated (a compile-time drift guard), and a
/// parity test enumerates every pyclass → variant mapping (a test-time guard).
/// See `crate::scale` module docs for the full picture.
///
/// Uses `tag = "type"` (NOT the spec-module convention `tag = "kind"`) for Vega-Lite
/// wire-format alignment — see design spec §11 row 16 ("Vega-Lite interop stays open
/// without translation"). This is the only tagged enum in this module that uses
/// `"type"`; the choice is intentional.
///
/// **No `deny_unknown_fields` (documented exception).** Unlike `EncodingSpec` and
/// `ChartSpec` (which deny unknown keys so typo'd channel/top-level keys fail loud),
/// `ScaleSpec` is an internally-tagged enum whose every continuous variant embeds
/// `ContinuousScaleCommon` via `#[serde(flatten)]`. serde cannot enforce
/// `deny_unknown_fields` through a flattened or internally-tagged shape, so a typo'd
/// scale key (e.g. `clammp` for `clamp`) inside a `scale` sub-dict is tolerated and
/// silently dropped. This is a structural serde limitation, not an oversight — adding
/// the attribute would be a no-op (or a compile error) given the flatten.
///
/// **Gate closes this at the wire boundary, not here (F-L04-07, batch-C
/// task 4).** The structural serde limitation above is permanent —
/// `deny_unknown_fields` still cannot see through the flatten — but a
/// scale-key validation gate now runs at every point a `ScaleSpec` is
/// deserialized (accepted-key sets derived from this schema, beside the
/// existing `validate_chart_config_keys` precedent), refusing an unknown
/// raw-dict scale key before it can be silently dropped. It lives in this
/// enum's own `Deserialize` impl (`remote = "Self"`, immediately below),
/// which is the one place every JSON- and PyO3-dict-sourced `ScaleSpec`
/// passes through regardless of caller — see
/// `accepted_keys_for_scale_type`/`validate_scale_spec_keys`. `reverse` on
/// `ContinuousScaleCommon` above was the reason this needed closing: a
/// typo'd `"reverse"` (e.g. `"reveres"`) now refuses naming the real key
/// among the accepted set, instead of vanishing with no error and no
/// visible effect.
///
/// **What a new 17th variant still grows by hand (`#159`).** `ScaleSpec::tag()`
/// below is compiler-enforced against this enum's own shape (its match has
/// no wildcard arm, so a new variant fails to compile there).
/// `ScaleSpec::TAGS` is weaker: a plain array, verified against `tag()` only
/// by a runtime test (`gate.rs`'s `tags_equals_the_image_of_tag_over_maximal_instances`),
/// not by the compiler — a variant missing from both `TAGS` and
/// `maximal_instances()` would pass that test undetected. Three further
/// things a new variant needs are enforced by neither: a `maximal_instances()`
/// entry and an `accepted_keys_for_scale_type` match arm (both `gate.rs`),
/// and a `ScaleKind` resolution row in the table above. A temporal variant
/// additionally needs an entry in
/// `ScaleSpec::TEMPORAL_TAGS`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase", remote = "Self")]
pub enum ScaleSpec {
    Linear {
        #[serde(flatten)]
        common: ContinuousScaleCommon,
        #[serde(default)]
        nice: bool,
        #[serde(default)]
        zero: bool,
        /// **Provenance (#247A, spec §6): ferrum wrote this scale, the caller
        /// did not.** Its ONE producer is the bar zero-anchor injection in
        /// `_spec_build.py::_build_encoding_specs`, and only in the case where
        /// that injection authored the whole scale (the channel carried no
        /// `scale=` of its own). An injection that merely ADDS `zero` on top of
        /// a scale the caller wrote leaves this `false` — the caller's
        /// `padding`/`nice`/`range` are still their intent, and the
        /// explicit-scale contract must keep honoring them.
        ///
        /// Its two consumers both used to read a bare `enc.scale.is_some()` and
        /// so could not tell ferrum's own injection from a user's scale:
        /// `render::composite::leaf_positional_domain` (which excluded any
        /// explicit-scale leaf from cross-leaf domain sharing) and
        /// `render::scale_resolve::positional::build_axis_scale` (whose
        /// explicit-scale bypass returns before the shared domain is
        /// consulted). Together those locked a zero-anchored `mark_bar` leaf out
        /// of an `fm.layer` union in BOTH directions, so the shared axis showed
        /// leaf 0's own domain and every other layer drew against a scale the
        /// ticks did not describe (#247A: `fm.layer(bar(1..3), point(40..80))`
        /// rendered a 0..3 axis). The anchor itself is unaffected — see
        /// [`ScaleSpec::anchors_zero`], which carries `zero` into the shared
        /// domain so the union spans `0..80` rather than dropping to `1..80`.
        ///
        /// Skipped when `false`, so every scale that does not set it (every
        /// user-authored scale, every non-bar chart) serializes to the bytes it
        /// did before this field existed.
        ///
        /// **User-reachable by hand, deliberately.** The name says "the system
        /// wrote this", but nothing can stop a caller from writing
        /// `scale={"type": "linear", "systemInjected": True}` in a raw dict: the
        /// Python producer crosses the same wire boundary every user dict
        /// crosses, so the key gate (`accepted_keys_for_scale_type`) has to
        /// accept it on `linear`, and a gate that could tell the two apart would
        /// need a notion of caller identity the wire does not carry. What that
        /// spelling buys is exactly what the flag means and nothing more: the
        /// scale stops short-circuiting a cross-mark domain (a composite shared
        /// domain, or a layered spec's cross-layer union) and resolves on the
        /// auto path instead, which for a hand-written scale means its
        /// `padding`/`nice`/`range` stop being honored. A coherent, if pointless,
        /// request — not a corruption, and not a way to reach any behavior a
        /// caller could not already reach by omitting the scale. It is NOT
        /// offered as an override path (`_override_apply._SCALE_LEAF_SYSTEM_ONLY`
        /// excludes it from the auto-derived `<channel>_scale_<leaf>` surface),
        /// so the hand-written raw dict is the only spelling that reaches it.
        ///
        /// **Not the north star.** Rust already owns an auto-path bar
        /// zero-anchor for layered specs
        /// (`render::scale_resolve::domain::layer_wants_bar_zero_anchor`), and
        /// deleting Python's injection in favour of it would leave one
        /// mechanism instead of two. That unification is a logged follow-up, not
        /// this field: the auto-inferred path applies padding/`nice` treatment
        /// the explicit-scale bypass skips, so it would move every bar-chart
        /// golden (spec §8).
        #[serde(rename = "systemInjected", default, skip_serializing_if = "std::ops::Not::not")]
        system_injected: bool,
    },
    Log {
        #[serde(default = "default_log_base")]
        base: f64,
        #[serde(flatten)]
        common: ContinuousScaleCommon,
        #[serde(default)]
        nice: bool,
    },
    Time {
        #[serde(flatten)]
        common: ContinuousScaleCommon,
        #[serde(default)]
        nice: bool,
    },
    Symlog {
        #[serde(default = "default_symlog_constant")]
        constant: f64,
        #[serde(flatten)]
        common: ContinuousScaleCommon,
        #[serde(default)]
        nice: bool,
    },
    Ordinal {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        domain: Option<Vec<String>>,
        /// Polymorphic range: either pixel coordinates (numbers, for positional
        /// use) or color strings (for color-channel use), or a mix.
        ///
        /// Typed as `Vec<OrdinalRangeValue>` (an `untagged` enum over
        /// `Number` | `Str`) so both `[0, 300]` and `["#ccc", "#e4572e"]`
        /// round-trip through the JSON wire format as a plain array — the exact
        /// shape Python emits for `scale.range`. The positional resolver pulls
        /// the `Number` arms; `build_color_scale` pulls the `Str` arms (F2
        /// typed-range fix, 2026-05-31, replacing the prior `serde_json::Value`
        /// + JSON-sniffing approach from the D1 fix).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        range: Option<Vec<crate::scale::ordinal::OrdinalRangeValue>>,
        #[serde(default)]
        padding: f64,
    },
    Pow {
        #[serde(default = "default_pow_exponent")]
        exponent: f64,
        #[serde(flatten)]
        common: ContinuousScaleCommon,
    },
    Sqrt {
        #[serde(flatten)]
        common: ContinuousScaleCommon,
    },
    Utc {
        #[serde(flatten)]
        common: ContinuousScaleCommon,
        #[serde(default)]
        nice: bool,
    },
    Band {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        domain: Option<Vec<String>>,
        #[serde(default = "default_band_padding")]
        padding: f64,
        #[serde(default, skip_serializing_if = "Option::is_none", rename = "paddingInner")]
        padding_inner: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none", rename = "paddingOuter")]
        padding_outer: Option<f64>,
        #[serde(default = "default_band_align")]
        align: f64,
        /// Explicit pixel range `[lo, hi]`. Defaults to the plot extent when
        /// absent (issue #39 fix — previously always dropped at the wire
        /// boundary regardless of what the user passed).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        range: Option<Vec<f64>>,
    },
    Point {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        domain: Option<Vec<String>>,
        #[serde(default = "default_point_padding")]
        padding: f64,
        #[serde(default = "default_band_align")]
        align: f64,
        #[serde(default)]
        reverse: bool,
        /// Explicit pixel range `[lo, hi]`. Defaults to the plot extent when
        /// absent (issue #39 fix — previously always dropped at the wire
        /// boundary regardless of what the user passed).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        range: Option<Vec<f64>>,
    },
    Sequential {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        scheme: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        domain: Option<Vec<f64>>,
        #[serde(default)]
        reverse: bool,
        /// Explicit gradient color stops as `(t, hex)` pairs, carried when
        /// the scale is backed by a `Gradient`-constructed `ContinuousScheme`
        /// rather than a named colormap (F-L04-02 second revision, spec
        /// §4.2 amended 2026-08-28; re-shaped from a colors-only `Vec<String>`
        /// in the spec reviewer's cycle-3 pass — the field was uncommitted,
        /// so its shape was still free to change, and a colors-only wire
        /// form silently discarded the `t` position `fm.Gradient([(t, color),
        /// ...])` documents and validates).
        ///
        /// `(f64, String)` tuples (not a parallel `positions: Vec<f64>` array)
        /// so the wire shape mirrors the `Gradient(stops: Vec<(f64, String)>)`
        /// pyfunction's own parameter type exactly — one field, no
        /// length-must-match-the-other-field invariant to maintain across
        /// (de)serialization. Serde's default tuple encoding makes each pair
        /// a 2-element JSON array (`[[0.0,"#ff0000"],[0.9,"#00ff00"],...]`),
        /// which round-trips a `(f64, String)` losslessly and reads as
        /// "stop points" at a glance, unlike two same-length arrays a reader
        /// has to zip mentally.
        ///
        /// Mutually exclusive with `scheme` in practice — a `ContinuousScheme`
        /// tree is built from either a named map or a `Gradient`, never both
        /// — but not enforced at the type level since the two fields
        /// serialize independently. Absent for every scheme-name-backed
        /// spec, so this is a pure addition: the `skip_serializing_if` guard
        /// keeps every pre-existing wire form byte-identical.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        stops: Option<Vec<(f64, String)>>,
    },
    Diverging {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        scheme: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        domain: Option<Vec<f64>>,
        #[serde(default, skip_serializing_if = "Option::is_none", rename = "domainMid")]
        domain_mid: Option<f64>,
    },
    Quantize {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        domain: Option<Vec<f64>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        range: Option<Vec<String>>,
    },
    Quantile {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        domain: Option<Vec<f64>>,   // sorted sample values
        #[serde(default, skip_serializing_if = "Option::is_none")]
        range: Option<Vec<f64>>,    // discrete numeric outputs
    },
    Threshold {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        domain: Option<Vec<f64>>,   // threshold boundaries
        #[serde(default, skip_serializing_if = "Option::is_none")]
        range: Option<Vec<f64>>,    // discrete numeric outputs; len == domain.len() + 1
    },
    #[serde(rename = "bin-ordinal")]
    BinOrdinal {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bins: Option<Vec<f64>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        scheme: Option<String>,
    },
}

impl ScaleSpec {
    /// The `domainParam` reference on a continuous scale, if any.
    ///
    /// Only the 7 continuous variants carry `ContinuousScaleCommon` (and thus a
    /// `domain_param`); categorical / sequential / diverging variants always
    /// return `None` (their reactive rescale is a recorded D6 follow-up).
    pub(crate) fn domain_param(&self) -> Option<&str> {
        let common = match self {
            ScaleSpec::Linear { common, .. }
            | ScaleSpec::Log { common, .. }
            | ScaleSpec::Time { common, .. }
            | ScaleSpec::Symlog { common, .. }
            | ScaleSpec::Pow { common, .. }
            | ScaleSpec::Sqrt { common, .. }
            | ScaleSpec::Utc { common, .. } => common,
            _ => return None,
        };
        common.domain_param.as_deref()
    }

    /// Set the numeric `domain` on a continuous scale and clear any
    /// `domainParam` reference (so downstream scale resolution sees a clean,
    /// fully-resolved domain). No-op for non-continuous variants.
    pub(crate) fn set_domain(&mut self, domain: Vec<f64>) {
        let common = match self {
            ScaleSpec::Linear { common, .. }
            | ScaleSpec::Log { common, .. }
            | ScaleSpec::Time { common, .. }
            | ScaleSpec::Symlog { common, .. }
            | ScaleSpec::Pow { common, .. }
            | ScaleSpec::Sqrt { common, .. }
            | ScaleSpec::Utc { common, .. } => common,
            _ => return,
        };
        common.domain = Some(domain);
        common.domain_param = None;
    }

    /// The explicit `[min, max]` extent this spec's `domain` implies for a
    /// *positional* Linear axis, or `None` to derive the axis from the data
    /// column instead.
    ///
    /// This match is exhaustive **on purpose, with no wildcard arm** — unlike
    /// the sibling `domain_param`/`set_domain` above, where `None`/no-op is a
    /// safe default for every variant, a wrong default here silently drops
    /// marks. Do not "fix" the asymmetry by adding a wildcard: a new
    /// `ScaleSpec` variant must explicitly decide here whether its `domain`
    /// is a positional extent or a discrete-binning artifact (a sample list,
    /// a boundary list, or bin edges). Omitting a variant is a compile error,
    /// not a silent fallback. See issue #40 (a `Diverging` domain's 3-element
    /// `[lo, mid, hi]` was truncated to `[lo, mid]` by the domain-as-extent
    /// path) and the #38 Quantile/Threshold precedent this mirrors.
    pub(crate) fn positional_extent(&self) -> Option<Vec<f64>> {
        match self {
            // Sequential/Diverging/Quantize domains are outer bounds: 2
            // elements `[min, max]` (Sequential/Quantize) or 3 elements
            // `[lo, mid, hi]` (Diverging). Either way the positional extent
            // is the first and last element — mirrors
            // `color.rs::scale_explicit_domain`.
            ScaleSpec::Sequential { domain, .. }
            | ScaleSpec::Diverging { domain, .. }
            | ScaleSpec::Quantize { domain, .. } => match domain {
                Some(d) if d.len() >= 2 => Some(vec![d[0], d[d.len() - 1]]),
                _ => None,
            },

            // Quantile/Threshold/BinOrdinal domains are binning artifacts
            // (a sorted sample list, threshold boundaries, or bin edges),
            // not a positional extent (#38 semantics). Derive the axis from
            // the data column instead.
            ScaleSpec::Quantile { .. }
            | ScaleSpec::Threshold { .. }
            | ScaleSpec::BinOrdinal { .. } => None,

            // The 7 continuous variants already carry an explicit extent in
            // `common.domain`.
            ScaleSpec::Linear { common, .. }
            | ScaleSpec::Log { common, .. }
            | ScaleSpec::Time { common, .. }
            | ScaleSpec::Symlog { common, .. }
            | ScaleSpec::Pow { common, .. }
            | ScaleSpec::Sqrt { common }
            | ScaleSpec::Utc { common, .. } => common.domain.clone(),

            // Categorical variants have no continuous extent.
            ScaleSpec::Ordinal { .. } | ScaleSpec::Band { .. } | ScaleSpec::Point { .. } => None,
        }
    }

    /// Whether ferrum's own spec build authored this scale rather than the
    /// caller — see [`ScaleSpec::Linear`]'s `system_injected` field for the full
    /// contract. Only `Linear` can carry the flag (it is the only variant the
    /// bar zero-anchor injection ever produces), so every other variant answers
    /// `false`: a scale ferrum did not write is, by definition, the caller's.
    pub(crate) fn is_system_injected(&self) -> bool {
        matches!(self, ScaleSpec::Linear { system_injected: true, .. })
    }

    /// Whether this scale asks its resolved domain to include zero.
    ///
    /// `zero` is `Linear`'s own field — the only continuous variant that carries
    /// one, as `resolve_continuous_domain_and_range`'s doc records — so every
    /// other variant answers `false`. Read by
    /// `render::composite::leaf_positional_domain` so an injected scale's anchor
    /// reaches the cross-leaf union instead of being lost with the bypass it no
    /// longer takes; the explicit-scale path reads the field directly from its
    /// own `Linear` arm.
    pub(crate) fn anchors_zero(&self) -> bool {
        matches!(self, ScaleSpec::Linear { zero: true, .. })
    }

    /// The wire `"type"` tag this instance serializes under. Exhaustive —
    /// no wildcard arm — so a 17th variant fails to compile here before any
    /// consumer (`ScaleSpec::TAGS`, `gate.rs`'s drift tests) could silently
    /// under-cover it. Hand-spells this enum's own `#[serde(tag = "type",
    /// rename_all = "lowercase")]` mapping because serde does not expose a
    /// variant's resolved tag string at runtime.
    pub(crate) fn tag(&self) -> &'static str {
        match self {
            ScaleSpec::Linear { .. } => "linear",
            ScaleSpec::Log { .. } => "log",
            ScaleSpec::Time { .. } => "time",
            ScaleSpec::Symlog { .. } => "symlog",
            ScaleSpec::Ordinal { .. } => "ordinal",
            ScaleSpec::Pow { .. } => "pow",
            ScaleSpec::Sqrt { .. } => "sqrt",
            ScaleSpec::Utc { .. } => "utc",
            ScaleSpec::Band { .. } => "band",
            ScaleSpec::Point { .. } => "point",
            ScaleSpec::Sequential { .. } => "sequential",
            ScaleSpec::Diverging { .. } => "diverging",
            ScaleSpec::Quantize { .. } => "quantize",
            ScaleSpec::Quantile { .. } => "quantile",
            ScaleSpec::Threshold { .. } => "threshold",
            ScaleSpec::BinOrdinal { .. } => "bin-ordinal",
            // No `_` arm: a new variant must be added here before this
            // module compiles at all.
        }
    }

    /// The complete, wildcard-free set of 16 wire `"type"` tags, one per
    /// variant — the single published home for the scale-tag vocabulary
    /// (`#159`), read by `gate.rs`'s `scale_type_tags` pyfunction and by
    /// Python's `_SCALE_TYPE_TAGS` through that binding, replacing what used
    /// to be three independently hand-maintained copies (a test-only
    /// `ALL_TAGS` list here, and two Python literals). Declaration order,
    /// not sorted: `scale_type_tags()`'s documented return order.
    pub(crate) const TAGS: [&'static str; 16] = [
        "linear",
        "log",
        "time",
        "symlog",
        "ordinal",
        "pow",
        "sqrt",
        "utc",
        "band",
        "point",
        "sequential",
        "diverging",
        "quantize",
        "quantile",
        "threshold",
        "bin-ordinal",
    ];

    /// The temporal subset of [`ScaleSpec::TAGS`]: the two variants
    /// (`Time`, `Utc`) whose `domain` accepts date/datetime elements
    /// (`convert_raw_dict_temporal_domain`, this module's `gate.rs`
    /// sibling) and whose PyO3 siblings (`TimeScale`, `UtcScale`) both parse
    /// date/datetime constructor arguments. Read by `gate.rs`'s
    /// `scale_temporal_type_tags` pyfunction, published so Python's
    /// `_TEMPORAL_SCALE_TYPES` reads this set instead of holding its own
    /// `{"time", "utc"}` literal.
    pub(crate) const TEMPORAL_TAGS: [&'static str; 2] = ["time", "utc"];
}

fn default_log_base() -> f64 {
    10.0
}
fn default_symlog_constant() -> f64 {
    1.0
}
fn default_pow_exponent() -> f64 {
    2.0
}
pub(crate) fn default_band_padding() -> f64 {
    0.1
}
fn default_point_padding() -> f64 {
    0.5
}
fn default_band_align() -> f64 {
    0.5
}

/// Encoding channel specification — maps a data field to a visual variable.
///
/// Created implicitly by Python's encoding channel classes (``X``, ``Y``,
/// ``Color``, ...). Carries the field name, optional inferred data type,
/// and optional scale/title overrides.
///
/// Parameters
/// ----------
/// field : str
///     Column name in the input DataFrame.
/// type_ : {"Q", "N", "O", "T", "quantitative", "nominal", "ordinal", \
///          "temporal"}, optional
///     Data type. Inferred from the column dtype when omitted.
/// scale : dict, optional
///     Scale override (e.g. ``{"type": "log"}``). Honored by the renderer.
/// title : str, optional
///     Axis or legend title. Overrides the auto-generated field name.
/// axis : dict, optional
///     Per-channel axis style overrides, typed against the shared
///     ``AxisStyleSpec``. Every advertised ``fm.Axis`` field that renders at
///     chart level (grid color/dash/width, label color/font-size, domain
///     color/width, title styling, tick options, …) also renders per-channel.
///     Unknown keys fail loud (a serde error surfaced as ``ValueError``).
/// legend : dict, optional
///     Per-channel legend style overrides, typed against the shared
///     ``LegendStyleSpec`` (orient, title, symbol/gradient geometry, columns,
///     …). Unknown keys fail loud.
/// sort : dict or str, optional
///     Sort order for ordinal/nominal scales. Accepts ``"ascending"``,
///     ``"descending"``, an explicit array of domain values, or a sort-field
///     object (``{"field": ..., "op": ..., "order": ...}``). Honored by the
///     renderer. A sort-field object missing a string ``"field"``, or naming
///     an unrecognized ``"op"``/``"order"``, raises ``ValueError`` here.
/// stack : str, optional
///     Stack method for bar/area marks. Accepts ``"zero"``,
///     ``"normalize"``, or ``"center"``. Honored by the renderer.
/// impute : dict, optional
///     Imputation strategy. Accepts ``{"value": N}`` to fill missing
///     group×x combinations with constant *N*. Honored by the renderer.
/// scheme : str, optional
///     Color scheme name for quantitative color encodings (e.g. ``"viridis"``).
///     Honored by the renderer via ``scale_resolve``.
/// format : str, optional
///     Tick/label format string. Applied to axis tick labels for x/y
///     encodings and to text mark labels. Honored by the renderer.
/// format_type : str, optional
///     Format type (e.g. ``"time"``). When set to ``"time"``, the
///     ``format`` string is interpreted as a date/time pattern. Honored
///     by the renderer for text mark labels.
///
/// Notes
/// -----
/// Users typically work with the higher-level encoding channel classes
/// from ``ferrum.encoding`` (``X``, ``Y``, ``Color``, ...);
/// ``EncodingSpec`` is the internal IR that ``Chart.encode(...)`` builds.
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> enc = fm.EncodingSpec(x="sepal_length", y="sepal_width", color="species")
#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EncodingSpec {
    pub field: String,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none", default)]
    pub type_: Option<DataType>,

    // NEW honored fields (Phase 8a):
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<ScaleSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,

    // Honored renderer fields (D7–D13 — consumed by prepare.rs / position.rs / scale_resolve.rs):
    // Per-channel axis/legend styling. Typed against the shared style structs
    // (B5 fix) so every advertised `fm.Axis`/`fm.Legend` field that renders at
    // chart level also renders per-channel, and unknown keys on the `axis`/
    // `legend` sub-dicts fail loud (`AxisStyleSpec`/`LegendStyleSpec` carry
    // `deny_unknown_fields`) instead of silently dropping.
    //
    // No-silent-drop scope (S4FIX2): `EncodingSpec` itself, `ChartSpec`
    // (chart.rs), the per-transform `*Spec` structs, and the `AxisStyleSpec`/
    // `LegendStyleSpec` style structs all carry `#[serde(deny_unknown_fields)]`,
    // so a typo'd channel / top-level / transform / axis-style key raises a serde
    // error rather than being silently dropped. The one documented exception is
    // `ScaleSpec` (the `scale:` field below): it is an internally-tagged enum
    // whose every variant uses `#[serde(flatten)]` for `ContinuousScaleCommon`,
    // and serde cannot enforce `deny_unknown_fields` through a flattened /
    // internally-tagged shape — so unknown keys inside a `scale` sub-dict (e.g.
    // a mistyped `clammp`) are tolerated and dropped. This is a structural serde
    // constraint, not an oversight; see the `ScaleSpec` doc comment.
    //
    // Boxed so the (wide, ~30-field) style structs do not bloat `EncodingSpec`'s
    // monomorphized serde deserialize frame — the unboxed form overflowed the
    // default 2MB test-thread stack when deserializing a deep `ChartSpec` graph.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub axis: Option<Box<crate::render::chart_config::AxisStyleSpec>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub legend: Option<Box<crate::render::chart_config::LegendStyleSpec>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sort: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stack: Option<String>,
    // Retained, tolerated, and consumer-less (#237, 2026-09-20): the Python
    // per-channel `condition=` serializer (`_emit_condition`) was deleted --
    // it had no reader here or anywhere else in the crate (`grep -rn
    // "\.condition" crates/ferrum-core/src crates/ferrum-wasm/src` outside
    // this file matches only unrelated `.conditionals` hits, the real,
    // chart-level mechanism `ChartSpec.conditionals` uses). Dropping this
    // field too would not make an old saved spec's `"condition"` key
    // *ignored* -- `EncodingSpec` carries `#[serde(deny_unknown_fields)]`, so
    // `ChartSpec::from_json` (the live render path, `_render.py`) would
    // reject any spec persisted before this fix with "unknown field
    // `condition`". Kept solely to preserve that round-trip; the field is
    // still accepted and read back verbatim by the PyO3 constructor/getter
    // below and pinned by `tests/test_core_stub_parity.py`, but nothing in
    // the render path consults it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub impute: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scheme: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    #[serde(rename = "formatType", default, skip_serializing_if = "Option::is_none")]
    pub format_type: Option<String>,
}

impl EncodingSpec {
    pub(crate) fn repr_string(&self) -> String {
        match &self.type_ {
            None => format!("EncodingSpec(field='{}')", self.field),
            Some(t) => format!("EncodingSpec(field='{}', type_='{}')", self.field, t.as_str()),
        }
    }
}

/// Serialize an optional Rust value to a Python object via JSON round-trip.
///
/// `None` maps to `Ok(None)`; `Some(v)` serializes `v` to a JSON string and
/// deserializes it back into a Python object via `json.loads`, matching the
/// behavior previously repeated inline in each `EncodingSpec` getter.
/// The spec interface (C8) names this function for the `Option<serde_json::Value>`
/// case; the generic `T: Serialize` bound covers that case and the typed-struct
/// getters (scale/axis/legend) with one implementation rather than two.
///
/// `pub(crate)` so the `crate::scale` `*Scale` pyclasses can reuse it to emit
/// their canonical `ScaleSpec` wire dict (`_to_scale_spec_dict`) without a second
/// serialization helper (SPEC-04).
pub(crate) fn encode_serde_value_for_py<T: serde::Serialize>(
    py: Python,
    v: &Option<T>,
) -> PyResult<Option<Py<PyAny>>> {
    match v {
        None => Ok(None),
        Some(val) => {
            let json = serde_json::to_string(val)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let json_module = py.import("json")?;
            Ok(Some(json_module.call_method1("loads", (json,))?.unbind()))
        }
    }
}

#[pymethods]
impl EncodingSpec {
    #[new]
    #[pyo3(signature = (
        field, type_ = None, *,
        scale = None, title = None,
        axis = None, legend = None, sort = None, stack = None,
        condition = None, impute = None, scheme = None, format = None, format_type = None,
    ))]
    fn new(
        py: Python,
        field: &str,
        type_: Option<&str>,
        scale: Option<&Bound<'_, PyAny>>,
        title: Option<String>,
        axis: Option<&Bound<'_, PyAny>>,
        legend: Option<&Bound<'_, PyAny>>,
        sort: Option<&Bound<'_, PyAny>>,
        stack: Option<String>,
        condition: Option<&Bound<'_, PyAny>>,
        impute: Option<&Bound<'_, PyAny>>,
        scheme: Option<String>,
        format: Option<String>,
        format_type: Option<String>,
    ) -> PyResult<Self> {
        if field.is_empty() {
            return Err(PyValueError::new_err("field must be non-empty"));
        }
        let type_ = match type_ {
            Some(s) => Some(
                s.parse::<DataType>()
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            ),
            None => None,
        };

        fn json_round<T: for<'de> serde::Deserialize<'de>>(
            py: Python,
            obj: Option<&Bound<'_, PyAny>>,
            name: &str,
        ) -> PyResult<Option<T>> {
            let Some(o) = obj else { return Ok(None) };
            let json_module = py.import("json")?;
            let s: String = json_module.call_method1("dumps", (o,))?.extract()?;
            Ok(Some(
                serde_json::from_str(&s)
                    .map_err(|e| PyValueError::new_err(format!("{name}: {e}")))?,
            ))
        }

        let sort: Option<serde_json::Value> = json_round(py, sort, "sort")?;
        if let Some(serde_json::Value::Object(obj)) = &sort {
            sort::parse_sort_field_object(obj).map_err(
                |reason| {
                    PyValueError::new_err(format!(
                        "EncodingSpec(sort=...): {reason}; received {}",
                        serde_json::Value::Object(obj.clone())
                    ))
                },
            )?;
        }

        Ok(EncodingSpec {
            field: field.to_string(),
            type_,
            scale: deserialize_scale_value(py, scale)?,
            title,
            axis: json_round(py, axis, "axis")?,
            legend: json_round(py, legend, "legend")?,
            sort,
            stack,
            condition: json_round(py, condition, "condition")?,
            impute: json_round(py, impute, "impute")?,
            scheme,
            format,
            format_type,
        })
    }

    /// Column name in the input DataFrame.
    #[getter]
    fn field(&self) -> &str {
        &self.field
    }

    /// Data type string (``"quantitative"``, ``"nominal"``, ``"ordinal"``,
    /// ``"temporal"``), or ``None`` when inferred.
    #[getter]
    fn type_(&self) -> Option<&'static str> {
        self.type_.as_ref().map(|t| t.as_str())
    }

    /// Scale override dict, or ``None``.
    #[getter]
    fn scale(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        encode_serde_value_for_py(py, &self.scale)
    }

    /// Axis or legend title override, or ``None``.
    #[getter]
    fn title(&self) -> Option<&str> {
        self.title.as_deref()
    }

    /// Axis style overrides. Typed against the shared `AxisStyleSpec`: every
    /// advertised `fm.Axis` field that renders at chart level also renders
    /// per-channel; unknown keys fail loud at construction.
    #[getter]
    fn axis(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        encode_serde_value_for_py(py, &self.axis)
    }

    /// Legend style overrides. Typed against the shared `LegendStyleSpec`: every
    /// advertised `fm.Legend` field that renders at chart level also renders
    /// per-channel; unknown keys fail loud at construction.
    #[getter]
    fn legend(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        encode_serde_value_for_py(py, &self.legend)
    }

    /// Conditional encoding rules (selection-driven); returns what was passed at construction.
    #[getter]
    fn condition(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        encode_serde_value_for_py(py, &self.condition)
    }

    /// Sort order for ordinal/nominal scales ("ascending", "descending", or explicit array).
    #[getter]
    fn sort(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        encode_serde_value_for_py(py, &self.sort)
    }

    /// Stack method for bar/area marks ("zero", "normalize", "center").
    #[getter]
    fn stack(&self) -> Option<&str> {
        self.stack.as_deref()
    }

    /// Imputation strategy. {"value": N} fills missing group×x combinations with N.
    #[getter]
    fn impute(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        encode_serde_value_for_py(py, &self.impute)
    }

    /// Color scheme name for quantitative encodings (e.g. ``"viridis"``).
    #[getter]
    fn scheme(&self) -> Option<&str> {
        self.scheme.as_deref()
    }

    /// Tick/label format string. Applied to axis tick labels (x/y) and text mark labels.
    #[getter]
    fn format(&self) -> Option<&str> {
        self.format.as_deref()
    }

    /// Format type string. When "time", format is applied as a date/time pattern.
    #[getter]
    fn format_type(&self) -> Option<&str> {
        self.format_type.as_deref()
    }

    /// Return a string representation of this encoding spec.
    fn __repr__(&self) -> String {
        self.repr_string()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Encoding {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub x: Option<EncodingSpec>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub y: Option<EncodingSpec>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub color: Option<EncodingSpec>,
    // NEW Phase 8a:
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub size: Option<EncodingSpec>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub shape: Option<EncodingSpec>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub opacity: Option<EncodingSpec>,
    // NEW Phase 8b Task 22 (ribbon mark): paired-channel endpoints. x2 reserved for
    // future scale_resolve work in Task 36; ribbon drawer reads y2 directly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub x2: Option<EncodingSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub y2: Option<EncodingSpec>,
    // Phase 10c: text channel for mark_text label content. When set, mark_text
    // reads this column for the rendered label; otherwise it falls back to
    // formatting the y value (legacy Phase 7 behavior).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<EncodingSpec>,
    // Phase 10 gallery-defaults: tooltip field emitted as SVG <title> on each mark.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tooltip: Option<EncodingSpec>,
    // Multi-field tooltip support: when set, takes precedence over `tooltip`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tooltip_fields: Option<Vec<EncodingSpec>>,
    // Phase 10 gallery-defaults: href field wraps marks in SVG <a xlink:href=...>.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub href: Option<EncodingSpec>,
    // Phase 10 gallery-defaults: description field emits SVG <desc> for accessibility.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<EncodingSpec>,
    // Phase 11c: key channel for animated transitions — identifies marks
    // across data updates so the WASM renderer can lerp between old/new.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<EncodingSpec>,
    // mark_image URL-tile path: each row holds a base64 data URL
    // (data:image/png;base64,… or data:image/jpeg;base64,…).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<EncodingSpec>,
    // ── Stroke/angle channels (silent-drop remediation) ───────────────
    // These are data-driven per-row constants, not scale-transformed channels.
    // Values flow directly from the batch column into mark style / FillStroke.
    // stroke_width: per-row stroke line width in pixels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stroke_width: Option<EncodingSpec>,
    // stroke_opacity: per-row stroke opacity in [0, 1].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stroke_opacity: Option<EncodingSpec>,
    // stroke_dash: per-row palette index (0=solid, 1=dashed, 2=dotted, 3=dash-dot).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stroke_dash: Option<EncodingSpec>,
    // angle: per-row rotation in degrees around the mark anchor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub angle: Option<EncodingSpec>,
    // fill_opacity: per-row fill opacity in [0, 1]. Emitted as SVG `fill-opacity`
    // attribute — distinct from `opacity` which bakes into the fill RGBA alpha.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fill_opacity: Option<EncodingSpec>,
}

/// Merge a single `parent` `EncodingSpec` into a `child` slot.
///
/// - If `child` is `None`, adopt the entire parent value.
/// - If `child` and `parent` share the same `field`, propagate any parent
///   metadata that the child has not set (scale, title, scheme, type_,
///   axis, legend, format, format_type).
/// - If the fields differ, do nothing.
fn inherit_encoding_spec(child: &mut Option<EncodingSpec>, parent: &Option<EncodingSpec>) {
    match (child.as_mut(), parent.as_ref()) {
        (None, Some(_)) => {
            *child = parent.clone();
        }
        (Some(c), Some(p)) if c.field == p.field => {
            if c.scale.is_none() && p.scale.is_some() {
                c.scale = p.scale.clone();
            }
            if c.title.is_none() && p.title.is_some() {
                c.title = p.title.clone();
            }
            if c.scheme.is_none() && p.scheme.is_some() {
                c.scheme = p.scheme.clone();
            }
            if c.type_.is_none() && p.type_.is_some() {
                c.type_ = p.type_;
            }
            if c.axis.is_none() && p.axis.is_some() {
                c.axis = p.axis.clone();
            }
            if c.legend.is_none() && p.legend.is_some() {
                c.legend = p.legend.clone();
            }
            if c.format.is_none() && p.format.is_some() {
                c.format = p.format.clone();
            }
            if c.format_type.is_none() && p.format_type.is_some() {
                c.format_type = p.format_type.clone();
            }
        }
        _ => {}
    }
}

impl Encoding {
    /// Inherit unset encoding channels from `parent`.
    ///
    /// For each of the 9 channels (x, y, color, size, shape, opacity, x2, y2, text):
    ///   - If this layer's channel is unset (`None`), adopt the parent's value.
    ///   - If this layer's channel is set with the same `field` as the parent
    ///     and has no `scale` of its own, inherit the parent's scale spec.
    ///     This lets a chart-level explicit scale (domain/range/padding)
    ///     apply to every layer that references the same field, while
    ///     leaving layer-supplied scales untouched.
    ///
    /// Phase 10f: pre-F7 only x/y/color/size received the scale merge;
    /// shape/opacity/x2/y2/text fell through with no merge. F7 applies the
    /// merge uniformly so the policy is symmetric and predictable — the
    /// per-channel asymmetry was an undocumented accident.
    ///
    /// `encoding.color` (like every other channel here) is always fully
    /// inherited — a layer's per-row color READ is instead gated downstream,
    /// at `scene_build.rs`'s per-layer `DrawCtx` construction, using
    /// [`LayerPrepared::color_is_own`](crate::render::prepare::LayerPrepared::color_is_own)
    /// (spec §4.4, 2026-08-28 T4 amendment, cycle-4 finding; widened batch-A
    /// T5d, 2026-08-28 from `Mark::Text`-only, unconditionally, to any OTHER
    /// mark — but only when that layer ALSO carries its own literal
    /// `stroke=`/`fill=` override, i.e.
    /// either paint's `PaintSource` being `UserColor`
    /// (`scene_build.rs`'s `build_panel_mark_batches`). A layer with no color
    /// of its own and no literal paint override still inherits — a blanket,
    /// unconditional widening regressed `catplot(kind="box", hue=x)`, whose
    /// box/tick/point layers rely on exactly that inheritance to paint each
    /// box in its own category's color). An earlier revision gated the
    /// exemption HERE instead,
    /// by deleting `color` from the merged `Encoding` for kwarg-less Text
    /// layers — but `encoding`
    /// (via [`crate::render::prepare::LayerPrepared`]) also feeds the legend
    /// (`resolve_legend_color_scale`, which reads layer 0's `encoding.color`
    /// directly) and dodge/stack position grouping
    /// (`position::resolve_group_channel`, same read), so deleting the
    /// channel here silently broke both for any Text layer with an
    /// inherited-only color: the legend vanished when text was layer 0, and
    /// dodged/stacked value labels collapsed to the group center instead of
    /// tracking their bar. Keeping this merge mark-agnostic (matching `main`)
    /// and gating only the mark-specific CONSUMER fixes both regressions.
    pub(crate) fn inherit_from(&mut self, parent: &Encoding) {
        inherit_encoding_spec(&mut self.x, &parent.x);
        inherit_encoding_spec(&mut self.y, &parent.y);
        inherit_encoding_spec(&mut self.color, &parent.color);
        inherit_encoding_spec(&mut self.size, &parent.size);
        inherit_encoding_spec(&mut self.shape, &parent.shape);
        inherit_encoding_spec(&mut self.opacity, &parent.opacity);
        inherit_encoding_spec(&mut self.x2, &parent.x2);
        inherit_encoding_spec(&mut self.y2, &parent.y2);
        inherit_encoding_spec(&mut self.text, &parent.text);
        inherit_encoding_spec(&mut self.tooltip, &parent.tooltip);
        if self.tooltip_fields.is_none() && parent.tooltip_fields.is_some() {
            self.tooltip_fields = parent.tooltip_fields.clone();
        }
        inherit_encoding_spec(&mut self.href, &parent.href);
        inherit_encoding_spec(&mut self.description, &parent.description);
        inherit_encoding_spec(&mut self.key, &parent.key);
        inherit_encoding_spec(&mut self.url, &parent.url);
        inherit_encoding_spec(&mut self.stroke_width, &parent.stroke_width);
        inherit_encoding_spec(&mut self.stroke_opacity, &parent.stroke_opacity);
        inherit_encoding_spec(&mut self.stroke_dash, &parent.stroke_dash);
        inherit_encoding_spec(&mut self.angle, &parent.angle);
        inherit_encoding_spec(&mut self.fill_opacity, &parent.fill_opacity);
    }

    /// Like `inherit_from` but skips positional channels (x, y, x2, y2).
    /// Used for layers routed to their own data via `data_source` — they
    /// should not inherit the primary batch's positional fields. `color`
    /// inheritance is unconditional here too — see `inherit_from`'s doc
    /// comment for why a Text-mark exemption does not belong in this merge.
    pub(crate) fn inherit_non_positional(&mut self, parent: &Encoding) {
        // Skip x, y, x2, y2 — positional channels belong to the layer's own data.
        inherit_encoding_spec(&mut self.color, &parent.color);
        inherit_encoding_spec(&mut self.size, &parent.size);
        inherit_encoding_spec(&mut self.shape, &parent.shape);
        inherit_encoding_spec(&mut self.opacity, &parent.opacity);
        inherit_encoding_spec(&mut self.text, &parent.text);
        inherit_encoding_spec(&mut self.tooltip, &parent.tooltip);
        if self.tooltip_fields.is_none() && parent.tooltip_fields.is_some() {
            self.tooltip_fields = parent.tooltip_fields.clone();
        }
        inherit_encoding_spec(&mut self.href, &parent.href);
        inherit_encoding_spec(&mut self.description, &parent.description);
        inherit_encoding_spec(&mut self.key, &parent.key);
        inherit_encoding_spec(&mut self.url, &parent.url);
        inherit_encoding_spec(&mut self.stroke_width, &parent.stroke_width);
        inherit_encoding_spec(&mut self.stroke_opacity, &parent.stroke_opacity);
        inherit_encoding_spec(&mut self.stroke_dash, &parent.stroke_dash);
        inherit_encoding_spec(&mut self.angle, &parent.angle);
        inherit_encoding_spec(&mut self.fill_opacity, &parent.fill_opacity);
    }

    /// Overlay channels from `overlay` onto `self`.
    ///
    /// For each of the 12 channels: if `overlay.{channel}.is_some()`,
    /// replace `self.{channel}` with `overlay`'s value. Channels where
    /// `overlay` is `None` are left untouched.
    ///
    /// This is the semantic inverse of [`inherit_from`](Self::inherit_from):
    /// `inherit_from` fills gaps (child inherits absent channels from
    /// parent), while `overlay_from` replaces present channels (overlay
    /// wins when `Some`).
    pub fn overlay_from(&mut self, overlay: &Encoding) {
        macro_rules! ov {
            ($($ch:ident),*) => {
                $( if overlay.$ch.is_some() { self.$ch = overlay.$ch.clone(); } )*
            };
        }
        ov!(x, y, color, size, shape, opacity, x2, y2, text, tooltip, tooltip_fields, href, description, key, url,
            stroke_width, stroke_opacity, stroke_dash, angle, fill_opacity);
    }

    /// Every populated channel as `(wire name, &EncodingSpec)`, in
    /// declaration order. `tooltip_fields` (the one list-valued channel) is
    /// expanded to `tooltip_fields[i]` per entry rather than a single
    /// `(name, EncodingSpec)` pair, since its `EncodingSpec`s are indexed,
    /// not singular.
    ///
    /// Destructures `Encoding` exhaustively (no `..`), so a field added to
    /// the struct is a compile error here until this function names it, and
    /// clippy's `-D warnings` (which includes `unused_variables`) then
    /// forces that named binding to actually reach `push!` or the
    /// `tooltip_fields` loop below -- coverage is compiler- and
    /// lint-enforced, not a second hand-typed list kept in sync by
    /// convention.
    ///
    /// Used by the render-time sort-object gate
    /// ([`crate::render::RenderError::SortSpecMalformed`]) to walk every
    /// channel a hand-built spec could carry a malformed `sort` on.
    pub(crate) fn channels(&self) -> Vec<(String, &EncodingSpec)> {
        let Encoding {
            x,
            y,
            color,
            size,
            shape,
            opacity,
            x2,
            y2,
            text,
            tooltip,
            tooltip_fields,
            href,
            description,
            key,
            url,
            stroke_width,
            stroke_opacity,
            stroke_dash,
            angle,
            fill_opacity,
        } = self;
        let mut out = Vec::new();
        macro_rules! push {
            ($($name:literal => $ch:expr),* $(,)?) => {
                $( if let Some(e) = $ch { out.push(($name.to_string(), e)); } )*
            };
        }
        push!(
            "x" => x, "y" => y, "color" => color, "size" => size, "shape" => shape,
            "opacity" => opacity, "x2" => x2, "y2" => y2, "text" => text,
            "tooltip" => tooltip, "href" => href, "description" => description,
            "key" => key, "url" => url, "stroke_width" => stroke_width,
            "stroke_opacity" => stroke_opacity, "stroke_dash" => stroke_dash,
            "angle" => angle, "fill_opacity" => fill_opacity,
        );
        if let Some(fields) = tooltip_fields {
            for (i, e) in fields.iter().enumerate() {
                out.push((format!("tooltip_fields[{i}]"), e));
            }
        }
        out
    }

    /// Swap the positional channels between the x and y roles (`CoordFlip`).
    ///
    /// The ONE expression of the flip mapping in this crate: `x`↔`y` and
    /// `x2`↔`y2` always travel together, so paired endpoints (segment, ribbon,
    /// ranged rule) stay self-consistent under the flip. Every stage that has
    /// to move an encoding between the pre-flip (authored) and post-flip
    /// (rendered) coordinate spaces calls this rather than re-deriving the
    /// mapping locally — spec §4.4, "Extended 2026-09-02":
    ///
    /// - [`crate::render::prepare::LayerPrepared::flip_coords`] applies it to
    ///   each layer's rendering encoding (and swaps the `x_is_own`/`y_is_own`
    ///   provenance flags with it, since those describe these same slots).
    /// - [`crate::render::scale_resolve::numeric_domain_union`] applies it to
    ///   read a still-authored `spec.layers` encoding in the post-flip space
    ///   its `channel` argument names.
    ///
    /// A second, hand-rolled swap at either site would be a second flip
    /// convention to keep in sync; there is deliberately only this one.
    pub(crate) fn flip_positional(&mut self) {
        std::mem::swap(&mut self.x, &mut self.y);
        std::mem::swap(&mut self.x2, &mut self.y2);
    }
}

