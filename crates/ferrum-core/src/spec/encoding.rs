use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

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

// ── Wire-key gate: schema-derived accepted-key sets (F-L04-07, batch-C task 4) ──
//
// `#[serde(remote = "Self")]` above turns the `#[derive(Serialize,
// Deserialize)]` output into INHERENT `ScaleSpec::serialize`/
// `ScaleSpec::deserialize` associated functions rather than trait impls
// (serde's documented "remote derive for Self" idiom — see
// https://serde.rs/remote-derive.html — used here to wrap the derived
// deserialize with a pre-check, without hand-copying all 16 variants into a
// second shadow type). The two trait impls below restore `ScaleSpec:
// Serialize + Deserialize` for every existing caller (`serde_json::to_string`,
// `EncodingSpec`'s own field, etc.): `Serialize` delegates straight through;
// `Deserialize` captures the incoming payload as a generic `serde_json::Value`
// first, walks its keys against `accepted_keys_for_scale_type`, and only then
// hands the (unmodified) `Value` to the inherent derived `deserialize` to do
// the real variant parsing — mirroring `binding.rs::validate_chart_config_keys`'s
// "gate before serde" shape, but at the type's own `Deserialize` boundary
// instead of a hand-walked `PyDict`.
//
// This is the single "every user scale dict passes exactly once" chokepoint
// FOR THE KEY GATE (spec §5) — verified (spec review cycle 1/2, rs-quality
// review) to cover all three routes a scale dict can take into Rust:
// `EncodingSpec::new`'s `json_round::<ScaleSpec>` (chart-level channels,
// calling `serde_json::from_str::<ScaleSpec>` directly); `ChartSpec::from_json`
// and the composite leaf path (`composite_node_from_py`'s
// `dict.get_item("spec")?.extract::<ChartSpec>()`, itself built by the same
// `EncodingSpec::new` constructor); and the LAYER path
// (`spec::chart::coerce_layers` → `pyo3_serde::from_py`, for
// `chart.layer(...)` channels and every composite-mark layer expansion),
// which never constructs an `EncodingSpec` in Rust at all but still
// deserializes each layer's `scale` sub-value through the identical
// `ScaleSpec::deserialize` call. All three refuse the same `clammp` typo
// with the same message shape. This claim is scoped to the KEY gate only —
// see `convert_raw_dict_temporal_domain`'s doc for why the separate
// TEMPORAL-DOMAIN CONVERSION does not have the same single-chokepoint
// property (it runs at a PyO3 constructor the layer path never enters, and
// is closed by a different mechanism on the Python side instead).
impl Serialize for ScaleSpec {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        ScaleSpec::serialize(self, serializer)
    }
}

impl<'de> Deserialize<'de> for ScaleSpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        validate_scale_spec_keys(&value).map_err(serde::de::Error::custom)?;
        ScaleSpec::deserialize(value).map_err(serde::de::Error::custom)
    }
}

/// `ContinuousScaleCommon`'s wire field names, shared by all 7 continuous
/// `ScaleSpec` variants via `#[serde(flatten)]`. Drift-tested against a
/// maximally-populated instance's serialized key set
/// (`accepted_keys_for_scale_type_matches_every_variants_serialized_keys`).
const CONTINUOUS_COMMON_SCALE_KEYS: &[&str] =
    &["domain", "range", "clamp", "padding", "scheme", "domainParam", "reverse"];

/// The accepted wire-key set for a `ScaleSpec` `"type"` tag, or `None` when
/// `scale_type` names no known variant (in which case `validate_scale_spec_keys`
/// leaves the dict unvalidated and lets the derived deserialize's own
/// "unknown variant" error surface instead). `"type"` itself is never listed
/// here — the caller (`validate_scale_spec_keys`) skips it explicitly, since
/// every variant accepts it by construction (the enum's internal tag).
///
/// Each arm's non-common fields are that variant's own struct fields, using
/// their WIRE spelling (`#[serde(rename = ...)]` where present:
/// `paddingInner`/`paddingOuter` for Band, `domainMid` for Diverging) — the
/// same convention `chart_config.rs::accepted_keys_for_section` uses.
///
/// **Known limit (T1 cross-task note): type-keyed, not channel-keyed.** This
/// gate accepts `reverse` on every continuous type regardless of which
/// encoding channel the scale is attached to, even though `reverse` is
/// currently inert on a non-positional channel (color/size/opacity read
/// `domain`/`range` off `ContinuousScaleCommon` but never consult `reverse`
/// — see `ContinuousScaleCommon::reverse`'s doc). Closing that would need the
/// gate to know which channel it is validating, which the `ScaleSpec::deserialize`
/// chokepoint does not see. `Diverging` deliberately does NOT list `reverse`
/// (it has no such field — T1 adjudicated this the more-silent third case,
/// since without the gate the key would not even round-trip); every other
/// non-goal is unaffected.
fn accepted_keys_for_scale_type(scale_type: &str) -> Option<Vec<&'static str>> {
    let keys: Vec<&'static str> = match scale_type {
        "linear" => CONTINUOUS_COMMON_SCALE_KEYS.iter().chain(["nice", "zero"].iter()).copied().collect(),
        "log" => CONTINUOUS_COMMON_SCALE_KEYS.iter().chain(["base", "nice"].iter()).copied().collect(),
        "time" => CONTINUOUS_COMMON_SCALE_KEYS.iter().chain(["nice"].iter()).copied().collect(),
        "symlog" => CONTINUOUS_COMMON_SCALE_KEYS.iter().chain(["constant", "nice"].iter()).copied().collect(),
        "pow" => CONTINUOUS_COMMON_SCALE_KEYS.iter().chain(["exponent"].iter()).copied().collect(),
        "sqrt" => CONTINUOUS_COMMON_SCALE_KEYS.to_vec(),
        "utc" => CONTINUOUS_COMMON_SCALE_KEYS.iter().chain(["nice"].iter()).copied().collect(),
        "ordinal" => vec!["domain", "range", "padding"],
        "band" => vec!["domain", "padding", "paddingInner", "paddingOuter", "align", "range"],
        "point" => vec!["domain", "padding", "align", "reverse", "range"],
        "sequential" => vec!["scheme", "domain", "reverse", "stops"],
        "diverging" => vec!["scheme", "domain", "domainMid"],
        "quantize" => vec!["domain", "range"],
        "quantile" => vec!["domain", "range"],
        "threshold" => vec!["domain", "range"],
        "bin-ordinal" => vec!["bins", "scheme"],
        _ => return None,
    };
    Some(keys)
}

/// Python-visible mirror of [`accepted_keys_for_scale_type`] — the single
/// source of truth for a `ScaleSpec` `"type"` tag's accepted wire-key set
/// (batch-C task 4, round 4). Exists so `_spec_build.py`'s override-scale
/// merge can filter stale keys against the SAME table the Rust key gate
/// enforces, instead of hand-mirroring a second, drift-prone copy of it in
/// Python (the recurring defect two task reviewers independently found:
/// the Python mirror was both too narrow — dropping `nice` on a
/// linear→log override — and blind to continuous↔non-continuous type
/// switches).
///
/// Returns `Err(ValueError)` for a `scale_type` naming no known `ScaleSpec`
/// variant, echoing [`validate_scale_spec_keys`]'s own notion of "known
/// type" (it delegates to the identical `accepted_keys_for_scale_type` call)
/// rather than maintaining a second known-types list.
///
/// This diverges, deliberately, from the crate's other keyed-registry
/// lookups (`palette_kind`/`palette_colors`/`palette_sample`,
/// `render/color/palette.rs`) which return `Option<T>` — `None` in Python —
/// for an unknown name. `scale_type` here is not an internal lookup key;
/// it arrives from user input via `Chart.override(<channel>_scale_type=...)`,
/// and its Python consumer (`_spec_build.py`) wants exactly one thing on an
/// unknown tag: let it fall through, unfiltered, to `ScaleSpec`'s own
/// deserialize gate, whose "unknown variant" message names the accepted
/// tag set — a strictly richer answer than this function could construct.
/// An `Option` return would force the caller to synthesize that same
/// fall-through decision from `None`; an exception the caller already
/// catches (`except ValueError`) *is* the fall-through, with no extra
/// branch at the call site. The exception's TYPE is therefore load-bearing,
/// not incidental — narrowing it to a different variant would silently
/// break the caller's fallback — and it is pinned by
/// `scale_accepted_keys_refuses_unknown_type`'s `is_instance_of::<PyValueError>`
/// assertion, not left to a message-substring check.
#[pyfunction]
pub fn scale_accepted_keys(scale_type: &str) -> PyResult<Vec<String>> {
    accepted_keys_for_scale_type(scale_type)
        .map(|keys| keys.into_iter().map(str::to_string).collect())
        .ok_or_else(|| PyValueError::new_err(format!("unknown scale type '{scale_type}'")))
}

/// Wire refusal for the scale-key gate (F-L04-07, spec §6). Pinned shape,
/// mirroring `binding.rs::chart_config_unknown_key_err`: names the unknown
/// key, the scale type, and the sorted accepted keys.
///
/// Deliberately carries NO `"scale: "` literal prefix (rs-quality review,
/// S3+1): this message is produced inside `ScaleSpec`'s own type-level
/// `Deserialize` impl, which is reached from more than one wire-boundary
/// context, each of which already supplies its own contextual prefix —
/// `EncodingSpec::new`'s `json_round` wraps every field error as
/// `"{name}: {e}"` (`"scale: {e}"` for this field), `coerce_layers`' error
/// path wraps as `"layers[{i}]: {e}"`, and `ChartSpec::from_json`'s bare
/// `serde_json::Error::to_string()` has no wrapper at all. A hard-coded
/// `"scale: "` here duplicated `json_round`'s own prefix on that one path
/// (`"scale: scale: unknown key …"`) while adding nothing on the other two.
/// `for type '{scale_type}'` already makes the message self-describing on
/// the unwrapped `from_json` path without it.
fn scale_gate_unknown_key_err(key: &str, scale_type: &str, accepted: &[&str]) -> String {
    let mut sorted: Vec<&str> = accepted.to_vec();
    sorted.sort_unstable();
    format!("unknown key '{key}' for type '{scale_type}'; accepted: {}", sorted.join(", "))
}

/// The wire-key gate itself (F-L04-07, spec §5/§6): walks a raw scale JSON
/// object's keys against `accepted_keys_for_scale_type`, called from
/// `ScaleSpec`'s `Deserialize` impl before the derived variant parsing runs.
///
/// Deliberately permissive on shapes it cannot judge — a non-object `value`
/// (malformed scale JSON), a missing/non-string `"type"`, or an unrecognized
/// `"type"` all return `Ok(())` and fall through to the derived deserialize,
/// whose own type-mismatch / missing-tag / "unknown variant" error is a
/// better, more specific message than anything this gate could produce for
/// those cases — this gate's only job is the *known-type, unknown-key*
/// case serde's flatten cannot see through on its own.
fn validate_scale_spec_keys(value: &serde_json::Value) -> Result<(), String> {
    let Some(obj) = value.as_object() else { return Ok(()) };
    let Some(scale_type) = obj.get("type").and_then(|v| v.as_str()) else { return Ok(()) };
    let Some(accepted) = accepted_keys_for_scale_type(scale_type) else { return Ok(()) };
    for key in obj.keys() {
        if key == "type" {
            continue;
        }
        if !accepted.contains(&key.as_str()) {
            return Err(scale_gate_unknown_key_err(key, scale_type, &accepted));
        }
    }
    Ok(())
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
///     ``"descending"``, or an explicit array of domain values. Honored
///     by the renderer.
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

/// Raw-dict temporal domain conversion (F-L04-10, spec §4D): converts every
/// element of a `{"type": "time"/"utc", "domain": [...]}` raw-dict scale's
/// `domain` list to epoch-ms via
/// [`temporal_value_to_epoch_ms`](crate::scale::time::temporal_value_to_epoch_ms)
/// — the exact rule `TimeScale(domain=...)`'s own PyO3 constructor applies —
/// BEFORE `EncodingSpec::new`'s `json_round` ever calls `json.dumps` on the
/// dict. This has to happen here, at the raw `Bound<PyDict>`, rather than
/// downstream in serde: a Python `datetime.datetime`/`datetime.date` object
/// cannot survive `json.dumps` at all (`TypeError: Object of type datetime is
/// not JSON serializable`), so by the time any JSON string exists, the
/// conversion opportunity is already gone.
///
/// **Adopted division of labor with `ferrum.encoding._scale._scale_to_dict`
/// (batch-C task 4, cycle 2).** `_scale_to_dict`'s Python-side dict branch
/// now ALSO converts every valid `date`/`datetime`/ISO-string domain element
/// to epoch-ms, at the ONE Python seam both wire routes share
/// (`ChannelBase.to_encoding_spec_dict()`) — covering the chart-level
/// channel path AND the layer/composite-mark path (`coerce_layers` /
/// `pyo3_serde::from_py`, which this Rust hook cannot reach, since it is
/// never routed through `EncodingSpec::new`). Because of that, by the time
/// THIS function runs on the chart-level path, a domain naming only valid
/// elements has typically already arrived pre-converted to floats — this
/// function is then a pass-through no-op for it, not a second conversion of
/// the same value. What this function remains the SOLE source of: the
/// clean, accepted-forms-naming `TypeError` for a genuinely INVALID domain
/// element (e.g. `object()`) on the chart-level path — `_scale_to_dict`
/// deliberately leaves a non-convertible element untouched rather than
/// raising, so as not to duplicate `temporal_value_to_epoch_ms`'s
/// accepted-forms taxonomy in Python. **Recorded residual, not fixed this
/// task:** the layer/composite path has no equivalent of this refusal — an
/// invalid domain element there still surfaces as `json.dumps`'s generic,
/// ferrum-silent `TypeError: Object of type ... is not JSON serializable`
/// once `coerce_layers` reaches it, rather than this function's message.
/// The BLOCKING half of this gap (valid dates crashing on the layer path)
/// is closed by the Python seam; only this narrower error-message-quality
/// asymmetry on an already-invalid value remains, logged for the batch
/// close rather than duplicating the refusal logic into Python here.
///
/// A `*Scale` pyclass instance's `_to_scale_spec_dict()` already returns a
/// domain of plain floats (its own constructor applied this same rule at
/// its own PyO3 boundary), and `_scale_to_dict`'s dict branch never invents
/// a `domain` this function would need to look past — so every dict this
/// sees, if it names a `domain`, is either already-numeric (pass-through, no
/// error) or genuinely needs conversion. Returns the ORIGINAL object
/// unchanged whenever there is nothing to convert (not a dict, not
/// time/utc-typed, no `domain` key, `domain` is `None`, or `domain` is not a
/// list/tuple — e.g. a reactive `Parameter`, which `_scale_to_dict` would already
/// have rewritten to a sibling `domainParam` key before `domain` ever
/// reached here) — the caller's own dict object is never mutated; a
/// converted domain always lands on a fresh `.copy()`.
fn convert_raw_dict_temporal_domain<'py>(
    scale: Option<&Bound<'py, PyAny>>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let Some(obj) = scale else { return Ok(None) };
    let Ok(d) = obj.cast::<PyDict>() else {
        return Ok(Some(obj.clone()));
    };
    let is_temporal = matches!(
        d.get_item("type")?.and_then(|t| t.extract::<String>().ok()).as_deref(),
        Some("time") | Some("utc")
    );
    if !is_temporal {
        return Ok(Some(obj.clone()));
    }
    let Some(domain_obj) = d.get_item("domain")? else {
        return Ok(Some(obj.clone()));
    };
    if domain_obj.is_none() {
        return Ok(Some(obj.clone()));
    }
    // A `list` (the documented `domain=[...]` shape) or a `tuple` (accepted
    // everywhere else a domain flows through `json.dumps`, which serializes
    // either as a JSON array) — anything else (e.g. a malformed scalar) is
    // left for the eventual serde type-mismatch error to name.
    let elements: Vec<Bound<'py, PyAny>> = if let Ok(list) = domain_obj.cast::<PyList>() {
        list.iter().collect()
    } else if let Ok(tuple) = domain_obj.cast::<pyo3::types::PyTuple>() {
        tuple.iter().collect()
    } else {
        return Ok(Some(obj.clone()));
    };
    let mut converted: Vec<f64> = Vec::with_capacity(elements.len());
    for item in elements {
        converted.push(crate::scale::time::temporal_value_to_epoch_ms(&item)?);
    }
    let out = d.copy()?;
    out.set_item("domain", converted)?;
    Ok(Some(out.into_any()))
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

        let scale_converted = convert_raw_dict_temporal_domain(scale)?;

        Ok(EncodingSpec {
            field: field.to_string(),
            type_,
            scale: json_round(py, scale_converted.as_ref(), "scale")?,
            title,
            axis: json_round(py, axis, "axis")?,
            legend: json_round(py, legend, "legend")?,
            sort: json_round(py, sort, "sort")?,
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
    /// `mark_style.paint.stroke_is_user_set || fill_is_user_set`
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_scale_domain_param_round_trips() {
        // `domainParam` deserializes from the compact wire form Python emits...
        let scale: ScaleSpec =
            serde_json::from_str(r#"{"type":"linear","domainParam":"d"}"#).unwrap();
        assert_eq!(scale.domain_param(), Some("d"));
        // ...and survives a serialize → deserialize round-trip.
        let back = serde_json::to_string(&scale).unwrap();
        assert!(back.contains(r#""domainParam":"d""#), "missing domainParam: {back}");
        let reparsed: ScaleSpec = serde_json::from_str(&back).unwrap();
        assert_eq!(reparsed, scale);
    }

    // ── Scale wire-key gate (F-L04-07, batch-C task 4) ──────────────────────

    fn maximal_common() -> ContinuousScaleCommon {
        ContinuousScaleCommon {
            domain: Some(vec![0.0, 1.0]),
            range: Some(vec![0.0, 1.0]),
            clamp: true,
            padding: Some(0.1),
            scheme: Some("blues".to_string()),
            domain_param: Some("p".to_string()),
            reverse: true,
        }
    }

    /// One maximally-populated instance per `ScaleSpec` variant (every
    /// optional field `Some`, every bool `true`) — every wire key that
    /// variant can ever emit is present in the serialized JSON. Used by the
    /// drift test below AND doubles as documentation of exactly which keys
    /// each type accepts.
    ///
    /// Deliberately returns bare instances, not `(tag, instance)` pairs: the
    /// tag comes from `scale_spec_tag_exhaustive` below, the ONE place a
    /// variant maps to its wire tag under compiler-enforced exhaustiveness,
    /// rather than a second, hand-typed label here that could silently
    /// mislabel an entry (spec review cycle-2 ❌3 / rs-quality S3+2).
    fn maximal_instances() -> Vec<ScaleSpec> {
        vec![
            ScaleSpec::Linear { common: maximal_common(), nice: true, zero: true },
            ScaleSpec::Log { base: 2.0, common: maximal_common(), nice: true },
            ScaleSpec::Time { common: maximal_common(), nice: true },
            ScaleSpec::Symlog { constant: 2.0, common: maximal_common(), nice: true },
            ScaleSpec::Ordinal {
                domain: Some(vec!["a".to_string()]),
                range: Some(vec![crate::scale::ordinal::OrdinalRangeValue::Number(1.0)]),
                padding: 0.1,
            },
            ScaleSpec::Pow { exponent: 3.0, common: maximal_common() },
            ScaleSpec::Sqrt { common: maximal_common() },
            ScaleSpec::Utc { common: maximal_common(), nice: true },
            ScaleSpec::Band {
                domain: Some(vec!["a".to_string()]),
                padding: 0.1,
                padding_inner: Some(0.1),
                padding_outer: Some(0.2),
                align: 0.5,
                range: Some(vec![0.0, 1.0]),
            },
            ScaleSpec::Point {
                domain: Some(vec!["a".to_string()]),
                padding: 0.5,
                align: 0.5,
                reverse: true,
                range: Some(vec![0.0, 1.0]),
            },
            ScaleSpec::Sequential {
                scheme: Some("viridis".to_string()),
                domain: Some(vec![0.0, 1.0]),
                reverse: true,
                stops: Some(vec![(0.0, "#fff".to_string())]),
            },
            ScaleSpec::Diverging {
                scheme: Some("rdbu".to_string()),
                domain: Some(vec![0.0, 1.0, 2.0]),
                domain_mid: Some(1.0),
            },
            ScaleSpec::Quantize {
                domain: Some(vec![0.0, 1.0]),
                range: Some(vec!["#fff".to_string()]),
            },
            ScaleSpec::Quantile { domain: Some(vec![0.0, 1.0]), range: Some(vec![0.0, 1.0]) },
            ScaleSpec::Threshold { domain: Some(vec![0.0]), range: Some(vec![0.0, 1.0]) },
            ScaleSpec::BinOrdinal {
                bins: Some(vec![0.0, 1.0]),
                scheme: Some("blues".to_string()),
            },
        ]
    }

    /// Variant-coverage guard (spec review cycle-2 ❌3; rs-quality S3+2):
    /// an EXHAUSTIVE match over `&ScaleSpec` — no wildcard arm — mapping
    /// every variant to its wire `"type"` tag. This is the compiler-enforced
    /// equivalent, for a closed Rust enum, of `chart_config.rs`'s
    /// `accepted_keys_for_section_covers_every_gated_section` (which has to
    /// iterate a string list, `CHART_CONFIG_SECTIONS`, because
    /// `ChartConfig`'s "sections" aren't a Rust enum the compiler can check
    /// this way): a future 17th `ScaleSpec` variant added anywhere in this
    /// crate makes THIS match fail to compile (E0004, non-exhaustive
    /// patterns) the moment `spec/encoding.rs`'s tests are built — before
    /// `accepted_keys_for_scale_type` or `maximal_instances()` even get a
    /// chance to silently under-cover it. RED-proofed by commenting out the
    /// `Threshold` arm and confirming `cargo build --tests -p ferrum-core`
    /// fails with "non-exhaustive patterns: `ScaleSpec::Threshold { .. }`
    /// not covered", then restoring it.
    fn scale_spec_tag_exhaustive(spec: &ScaleSpec) -> &'static str {
        match spec {
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

    /// `maximal_instances()` itself must stay exhaustive too — the tag match
    /// above only proves the ENUM-to-tag mapping is complete; it says
    /// nothing about whether `maximal_instances()`'s hand-built `Vec`
    /// actually contains one of every variant. Assert the two enumerations
    /// agree in length and (via `scale_spec_tag_exhaustive`) in tag identity
    /// — 16 variants today (corrected from the brief/reports' "15": `Band`
    /// was omitted from that count everywhere it appears, though it was
    /// always covered).
    #[test]
    fn maximal_instances_covers_every_scale_spec_variant() {
        use std::collections::HashSet;
        let tags: HashSet<&str> =
            maximal_instances().iter().map(scale_spec_tag_exhaustive).collect();
        const ALL_TAGS: &[&str] = &[
            "linear", "log", "time", "symlog", "ordinal", "pow", "sqrt", "utc", "band", "point",
            "sequential", "diverging", "quantize", "quantile", "threshold", "bin-ordinal",
        ];
        assert_eq!(tags.len(), ALL_TAGS.len(), "maximal_instances() has a duplicate or missing tag");
        for tag in ALL_TAGS {
            assert!(tags.contains(tag), "maximal_instances() is missing a '{tag}' instance");
        }
    }

    /// Drift guard: `accepted_keys_for_scale_type`'s hand-maintained set must
    /// equal the REAL key set a maximally-populated instance of that variant
    /// serializes — the same "derive from the schema, verify against the
    /// real struct" discipline `chart_config.rs`'s
    /// `*_keys_match_serde` tests use. A field added to `ContinuousScaleCommon`
    /// or any variant without updating `accepted_keys_for_scale_type` fails
    /// this test, not silently under-covers the gate.
    #[test]
    fn accepted_keys_for_scale_type_matches_every_variants_serialized_keys() {
        use std::collections::HashSet;
        for instance in maximal_instances() {
            let tag = scale_spec_tag_exhaustive(&instance);
            let value = serde_json::to_value(&instance).unwrap();
            let obj = value.as_object().unwrap();
            assert_eq!(
                obj.get("type").and_then(|v| v.as_str()),
                Some(tag),
                "scale_spec_tag_exhaustive disagrees with the real serialized \"type\" tag"
            );
            let actual: HashSet<&str> =
                obj.keys().map(|k| k.as_str()).filter(|k| *k != "type").collect();
            let expected: HashSet<&str> = accepted_keys_for_scale_type(tag)
                .unwrap_or_else(|| panic!("accepted_keys_for_scale_type must cover '{tag}'"))
                .into_iter()
                .collect();
            assert_eq!(actual, expected, "accepted-key set drifted for scale type '{tag}'");
        }
    }

    #[test]
    fn accepted_keys_for_scale_type_returns_none_for_unknown_type() {
        assert!(accepted_keys_for_scale_type("bogus").is_none());
    }

    /// The typo repro from spec §9: `{"type":"linear","clammp":true}` refuses,
    /// naming the real key (`clamp`) among the accepted set.
    ///
    // RED-proof (mutate-and-revert, run manually — not committed as a test):
    // temporarily replace `validate_scale_spec_keys`'s body with `Ok(())` and
    // re-run THIS test — it fails (`clammp` round-trips silently, matching
    // the pre-fix carve-out `tests/test_bug_hunt_encoding_step4.py::
    // test_scale_dict_typo_key_is_rejected` pins as its own RED state via the
    // opposite assertion). Restore before committing.
    #[test]
    fn scale_gate_typo_key_refused_names_real_key_among_accepted() {
        let err = serde_json::from_str::<ScaleSpec>(r#"{"type":"linear","clammp":true}"#)
            .expect_err("typo'd key must be refused");
        let msg = err.to_string();
        assert!(msg.contains("unknown key 'clammp' for type 'linear'"), "msg: {msg}");
        assert!(msg.contains("clamp"), "accepted list must name the real key: {msg}");
    }

    /// The finding's own motivating example: `"reveres"` (typo of `reverse`).
    #[test]
    fn scale_gate_reveres_typo_of_reverse_refused() {
        let err = serde_json::from_str::<ScaleSpec>(r#"{"type":"linear","reveres":true}"#)
            .expect_err("typo'd 'reveres' must be refused");
        let msg = err.to_string();
        assert!(msg.contains("unknown key 'reveres' for type 'linear'"), "msg: {msg}");
        assert!(msg.contains("reverse"), "accepted list must name the real key: {msg}");
    }

    /// T1's third silent-no-op case (encoding.rs's `ContinuousScaleCommon::reverse`
    /// doc, cycle-4 note): `Diverging` has no `reverse` field at all, so the
    /// gate must refuse it explicitly rather than silently dropping it.
    #[test]
    fn scale_gate_diverging_reverse_key_refused() {
        let err = serde_json::from_str::<ScaleSpec>(r#"{"type":"diverging","reverse":true}"#)
            .expect_err("'reverse' is not an accepted Diverging key");
        let msg = err.to_string();
        assert!(msg.contains("unknown key 'reverse' for type 'diverging'"), "msg: {msg}");
    }

    /// An unrecognized `"type"` is left to the derived deserialize's own
    /// "unknown variant" error — the gate does not shadow it with a less
    /// specific message.
    #[test]
    fn scale_gate_unknown_type_falls_through_to_variant_error() {
        let err = serde_json::from_str::<ScaleSpec>(r#"{"type":"bogus"}"#)
            .expect_err("unknown type must be refused");
        assert!(err.to_string().contains("unknown variant"), "msg: {err}");
    }

    /// Every currently-legal raw-dict scale shape used across `tests/*.py`
    /// (enumerated via a repo-wide grep before writing the gate, per the
    /// brief) must still parse cleanly — the gate must not over-refuse.
    #[test]
    fn scale_gate_does_not_over_refuse_currently_valid_shapes() {
        let valid = [
            r#"{"type":"linear","domain":[5.0,25.0]}"#,
            r#"{"type":"linear","zero":false}"#,
            r#"{"type":"log"}"#,
            r#"{"type":"linear","reverse":true}"#,
            r#"{"type":"linear","domain":[0.0,10.0],"nice":false,"reverse":true}"#,
            r#"{"type":"time"}"#,
            r#"{"type":"time","domain":[0.0,1000.0]}"#,
            r##"{"type":"ordinal","domain":["A","B","C"],"range":["#ccc","#ccc","#e45"]}"##,
            r#"{"type":"linear","domainParam":"d"}"#,
            r#"{"type":"diverging","domainMid":2.5}"#,
            r#"{"type":"band","paddingInner":0.1,"paddingOuter":0.4}"#,
            r#"{"type":"bin-ordinal","bins":[0.0,10.0,20.0,30.0]}"#,
        ];
        for json in valid {
            serde_json::from_str::<ScaleSpec>(json)
                .unwrap_or_else(|e| panic!("must still parse ({json}): {e}"));
        }
    }

    /// `ScaleSpec` must still serialize normally through the wrapped
    /// `Serialize` impl (the `remote = "Self"` idiom restores the trait impl
    /// rather than leaving only the derive-generated inherent method).
    #[test]
    fn scale_spec_serialize_still_works_through_the_trait() {
        let scale = ScaleSpec::Linear { common: maximal_common(), nice: true, zero: false };
        let json = serde_json::to_string(&scale).unwrap();
        assert!(json.contains(r#""type":"linear""#), "json: {json}");
        // Round-trips through the same gated Deserialize.
        let back: ScaleSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back, scale);
    }

    // ── Python-visible accepted-key table (batch-C task 4, round 4) ─────────

    /// `scale_accepted_keys` must agree with `accepted_keys_for_scale_type`
    /// — the function it delegates to — for every one of the 16 `ScaleSpec`
    /// tags, exhaustively (reuses `maximal_instances()` +
    /// `scale_spec_tag_exhaustive` rather than a second hand-typed tag list,
    /// so this test tracks the same compiler-enforced coverage as
    /// `maximal_instances_covers_every_scale_spec_variant` above).
    #[test]
    fn scale_accepted_keys_matches_accepted_keys_for_scale_type_for_every_tag() {
        for instance in maximal_instances() {
            let tag = scale_spec_tag_exhaustive(&instance);
            let mut expected: Vec<&str> = accepted_keys_for_scale_type(tag).unwrap();
            expected.sort_unstable();
            let mut actual: Vec<String> = scale_accepted_keys(tag).unwrap();
            actual.sort_unstable();
            assert_eq!(actual, expected, "mismatch for tag '{tag}'");
        }
    }

    /// An unrecognized `scale_type` refuses with a `ValueError`, echoing
    /// `validate_scale_spec_keys`'s own "unknown type falls through" notion
    /// of known-vs-unknown rather than silently returning an empty list.
    #[test]
    fn scale_accepted_keys_refuses_unknown_type() {
        attach(|py| {
            let err = scale_accepted_keys("not-a-real-type").unwrap_err();
            let msg = err.value(py).to_string();
            assert!(msg.contains("not-a-real-type"), "message should name the unknown type: {msg}");
            // The exception TYPE is the load-bearing half of the contract: the
            // Python consumer's fallback (`_spec_build.py`) catches `ValueError`
            // specifically to let an unknown tag fall through to the
            // deserialize gate's own richer message.
            assert!(err.is_instance_of::<PyValueError>(py), "expected ValueError, got {err:?}");
        });
    }

    // ── Raw-dict temporal domain conversion (F-L04-10, spec §4D) ────────────

    fn attach<F, T>(build: F) -> T
    where
        F: for<'py> FnOnce(Python<'py>) -> T,
    {
        pyo3::Python::initialize();
        Python::attach(build)
    }

    #[test]
    fn convert_raw_dict_temporal_domain_converts_date_and_datetime_elements() {
        attach(|py| {
            let d = PyDict::new(py);
            d.set_item("type", "time").unwrap();
            let date = pyo3::types::PyDate::new(py, 2020, 6, 1).unwrap();
            let datetime =
                pyo3::types::PyDateTime::new(py, 2020, 6, 1, 12, 30, 0, 0, None).unwrap();
            d.set_item("domain", (date, datetime)).unwrap();
            let scale_obj = d.into_any();
            let converted = convert_raw_dict_temporal_domain(Some(&scale_obj)).unwrap().unwrap();
            let out_dict = converted.cast::<PyDict>().unwrap();
            let domain: Vec<f64> = out_dict.get_item("domain").unwrap().unwrap().extract().unwrap();
            // date_epoch_ms(2020-06-01) = 1590969600000.0 (midnight UTC);
            // + 12:30:00 = + 45000 * 1000 ms.
            assert_eq!(domain, vec![1_590_969_600_000.0, 1_590_969_600_000.0 + 45_000_000.0]);
            // The caller's own dict must be untouched (no in-place mutation):
            // `scale_obj` still holds the original `date`/`datetime` pair, not
            // the converted floats — `convert_raw_dict_temporal_domain`
            // returned a fresh `.copy()`, distinct from `scale_obj` itself.
            assert!(!converted.is(&scale_obj));
            let original_dict = scale_obj.cast::<PyDict>().unwrap();
            let original_domain = original_dict.get_item("domain").unwrap().unwrap();
            let original_first = original_domain.get_item(0).unwrap();
            assert!(
                original_first.cast::<pyo3::types::PyDate>().is_ok(),
                "original domain element must still be a date object, not converted in place"
            );
        });
    }

    #[test]
    fn convert_raw_dict_temporal_domain_leaves_utc_type_domain_the_same_way() {
        attach(|py| {
            let d = PyDict::new(py);
            d.set_item("type", "utc").unwrap();
            let date = pyo3::types::PyDate::new(py, 2020, 6, 1).unwrap();
            d.set_item("domain", (date.clone(), date)).unwrap();
            let scale_obj = d.into_any();
            let converted = convert_raw_dict_temporal_domain(Some(&scale_obj)).unwrap().unwrap();
            let out_dict = converted.cast::<PyDict>().unwrap();
            let domain: Vec<f64> = out_dict.get_item("domain").unwrap().unwrap().extract().unwrap();
            assert_eq!(domain, vec![1_590_969_600_000.0, 1_590_969_600_000.0]);
        });
    }

    #[test]
    fn convert_raw_dict_temporal_domain_passes_through_non_temporal_type_unchanged() {
        attach(|py| {
            let d = PyDict::new(py);
            d.set_item("type", "linear").unwrap();
            d.set_item("domain", (0.0, 10.0)).unwrap();
            let scale_obj = d.into_any();
            let converted = convert_raw_dict_temporal_domain(Some(&scale_obj)).unwrap().unwrap();
            // Unchanged: the SAME dict object comes back (identity, not a copy).
            assert!(converted.is(&scale_obj));
        });
    }

    #[test]
    fn convert_raw_dict_temporal_domain_passes_through_missing_or_none_domain() {
        attach(|py| {
            let d = PyDict::new(py);
            d.set_item("type", "time").unwrap();
            let scale_obj = d.into_any();
            let converted = convert_raw_dict_temporal_domain(Some(&scale_obj)).unwrap().unwrap();
            assert!(converted.is(&scale_obj));
        });
    }

    #[test]
    fn convert_raw_dict_temporal_domain_refuses_bool_naming_accepted_forms() {
        attach(|py| {
            let d = PyDict::new(py);
            d.set_item("type", "time").unwrap();
            d.set_item("domain", (true, false)).unwrap();
            let scale_obj = d.into_any();
            let err = convert_raw_dict_temporal_domain(Some(&scale_obj)).unwrap_err();
            let msg = err.to_string();
            assert!(msg.contains("bool"), "msg: {msg}");
        });
    }

    /// End to end through the real `EncodingSpec::new` PyO3 constructor
    /// (not a hand-built mirror): a raw-dict `scale={"type": "time", "domain":
    /// [datetime.date(...), datetime.datetime(...)]}` produces an
    /// `EncodingSpec` whose `ScaleSpec::Time` domain is the converted
    /// epoch-ms pair, exactly like `TimeScale(domain=[...])` would.
    #[test]
    fn encoding_spec_new_converts_raw_dict_temporal_domain_end_to_end() {
        attach(|py| {
            let d = PyDict::new(py);
            d.set_item("type", "time").unwrap();
            let date_lo = pyo3::types::PyDate::new(py, 2020, 1, 1).unwrap();
            let date_hi = pyo3::types::PyDate::new(py, 2020, 12, 31).unwrap();
            d.set_item("domain", (date_lo, date_hi)).unwrap();
            let scale_obj = d.into_any();
            let spec = EncodingSpec::new(
                py, "date", None, Some(&scale_obj), None, None, None, None, None, None, None,
                None, None, None,
            )
            .unwrap();
            match spec.scale {
                Some(ScaleSpec::Time { common, .. }) => {
                    let domain = common.domain.expect("domain must be Some");
                    // Exact epoch-ms (rs-quality S2): a seconds-vs-ms error, a
                    // local-vs-UTC offset, or an off-by-a-day epoch would all
                    // pass a loose `len == 2 && ascending` check but not this
                    // one. Values independently computed via Python's
                    // `calendar.timegm(date.timetuple()) * 1000.0` (midnight
                    // UTC for a naive date), matching T3's UTC contract —
                    // not re-derived from ferrum's own converter.
                    assert_eq!(domain, vec![1_577_836_800_000.0, 1_609_372_800_000.0]);
                }
                other => panic!("expected ScaleSpec::Time, got {other:?}"),
            }
        });
    }

    #[test]
    fn test_scale_set_domain_clears_domain_param() {
        let mut scale: ScaleSpec =
            serde_json::from_str(r#"{"type":"linear","domainParam":"d"}"#).unwrap();
        scale.set_domain(vec![10.0, 20.0]);
        assert_eq!(scale.domain_param(), None);
        match scale {
            ScaleSpec::Linear { common, .. } => assert_eq!(common.domain, Some(vec![10.0, 20.0])),
            _ => panic!("expected Linear"),
        }
    }

    #[test]
    fn test_data_type_short_and_long_forms() {
        assert_eq!(DataType::from_str("Q").unwrap(), DataType::Quantitative);
        assert_eq!(DataType::from_str("quantitative").unwrap(), DataType::Quantitative);
        assert_eq!(DataType::from_str("N").unwrap(), DataType::Nominal);
        assert_eq!(DataType::from_str("nominal").unwrap(), DataType::Nominal);
    }

    #[test]
    fn test_data_type_unknown() {
        let err = DataType::from_str("Z").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("'Z'"), "msg: {msg}");
        assert!(msg.contains("quantitative"), "msg: {msg}");
    }

    #[test]
    fn test_data_type_serde_long_form() {
        assert_eq!(serde_json::to_string(&DataType::Quantitative).unwrap(), "\"quantitative\"");
    }

    #[test]
    fn test_encoding_spec_round_trip_no_type() {
        let original = EncodingSpec { field: "price".into(), type_: None, ..Default::default() };
        let json = serde_json::to_string(&original).unwrap();
        assert_eq!(json, r#"{"field":"price"}"#);
        let parsed: EncodingSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn test_encoding_spec_round_trip_with_type() {
        let original = EncodingSpec {
            field: "weight".into(),
            type_: Some(DataType::Quantitative),
            ..Default::default()
        };
        let json = serde_json::to_string(&original).unwrap();
        assert_eq!(json, r#"{"field":"weight","type":"quantitative"}"#);
        let parsed: EncodingSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn encoding_spec_unknown_field_is_rejected() {
        // No-silent-drop seam (S4FIX2): a typo'd channel key (e.g. `typ` for
        // `type`) on the EncodingSpec itself must fail loud via
        // `deny_unknown_fields`, not be silently dropped.
        let json = r#"{"field":"a","typ":"quantitative"}"#;
        let err = serde_json::from_str::<EncodingSpec>(json).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("unknown field") && msg.contains("typ"),
            "expected an unknown-field error, got: {msg}"
        );
    }

    #[test]
    fn encoding_spec_all_known_fields_round_trip_under_deny() {
        // Every advertised EncodingSpec field must still deserialize after the
        // deny was added — a known key must never be mistaken for an unknown one.
        let json = r##"{"field":"a","type":"quantitative","scale":{"type":"linear"},"title":"T","axis":{"grid":false},"legend":{"disabled":true},"sort":"ascending","stack":"zero","condition":{},"impute":{"value":0},"scheme":"viridis","format":".2f","formatType":"number"}"##;
        let parsed: EncodingSpec = serde_json::from_str(json).expect("known fields must deserialize");
        assert_eq!(parsed.field, "a");
        assert_eq!(parsed.type_, Some(DataType::Quantitative));
        assert!(parsed.scale.is_some());
    }

    #[test]
    fn test_encoding_round_trip_both_axes() {
        let e = Encoding {
            x: Some(EncodingSpec { field: "price".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec {
                field: "weight".into(),
                type_: Some(DataType::Quantitative),
                ..Default::default()
            }),
            color: None,
            ..Default::default()
        };
        let json = serde_json::to_string(&e).unwrap();
        assert_eq!(
            json,
            r#"{"x":{"field":"price"},"y":{"field":"weight","type":"quantitative"}}"#,
        );
        let parsed: Encoding = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, e);
    }

    #[test]
    fn test_encoding_omits_none_fields() {
        let e = Encoding::default();
        let json = serde_json::to_string(&e).unwrap();
        assert_eq!(json, "{}");
    }

    #[test]
    fn test_encoding_round_trip_with_color() {
        let e = Encoding {
            x: Some(EncodingSpec { field: "price".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "weight".into(), type_: None, ..Default::default() }),
            color: Some(EncodingSpec {
                field: "species".into(),
                type_: Some(DataType::Nominal),
                ..Default::default()
            }),
            ..Default::default()
        };
        let json = serde_json::to_string(&e).unwrap();
        assert_eq!(
            json,
            r#"{"x":{"field":"price"},"y":{"field":"weight"},"color":{"field":"species","type":"nominal"}}"#,
        );
        let parsed: Encoding = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, e);
    }

    #[test]
    fn test_encoding_omits_color_when_none() {
        let e = Encoding {
            x: Some(EncodingSpec { field: "a".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "b".into(), type_: None, ..Default::default() }),
            color: None,
            ..Default::default()
        };
        let json = serde_json::to_string(&e).unwrap();
        assert_eq!(json, r#"{"x":{"field":"a"},"y":{"field":"b"}}"#);
    }

    // --- Phase 8a new tests ---

    #[test]
    fn encoding_spec_round_trips_with_scale() {
        let e = EncodingSpec {
            field: "price".into(),
            type_: Some(DataType::Quantitative),
            scale: Some(ScaleSpec::Log {
                base: 10.0,
                common: ContinuousScaleCommon {
                    domain: None,
                    range: None,
                    clamp: false,
                    padding: None,
                    scheme: None,
                    domain_param: None,
                    reverse: false,
                },
                nice: true,
            }),
            ..Default::default()
        };
        let json = serde_json::to_string(&e).unwrap();
        assert!(json.contains(r#""scale":{"type":"log""#));
        let parsed: EncodingSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, e);
    }

    #[test]
    fn encoding_spec_round_trips_with_title() {
        let e = EncodingSpec {
            field: "x".into(),
            type_: None,
            title: Some("My X Axis".into()),
            ..Default::default()
        };
        let json = serde_json::to_string(&e).unwrap();
        assert!(json.contains(r#""title":"My X Axis""#));
        let parsed: EncodingSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, e);
    }

    #[test]
    fn encoding_spec_round_trips_with_typed_axis() {
        // A per-channel axis carrying both an honored styling key (`grid` toggle)
        // and an orphan positioning key (`orient`) round-trips through the typed
        // `AxisStyleSpec`.
        use crate::render::chart_config::AxisStyleSpec;
        let e = EncodingSpec {
            field: "x".into(),
            type_: None,
            axis: Some(Box::new(AxisStyleSpec {
                grid: Some(false),
                orient: Some("bottom".into()),
                ..Default::default()
            })),
            ..Default::default()
        };
        let json = serde_json::to_string(&e).unwrap();
        let parsed: EncodingSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, e);
    }

    #[test]
    fn encoding_spec_round_trips_with_typed_axis_styling_and_orphan() {
        // Honored styling key `grid_color` + orphan `orient` both survive the
        // round-trip (B5).
        use crate::render::chart_config::AxisStyleSpec;
        let e = EncodingSpec {
            field: "x".into(),
            axis: Some(Box::new(AxisStyleSpec {
                grid_color: Some("#cccccc".into()),
                orient: Some("bottom".into()),
                ..Default::default()
            })),
            ..Default::default()
        };
        let json = serde_json::to_string(&e).unwrap();
        let parsed: EncodingSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, e);
        assert_eq!(parsed.axis.as_ref().unwrap().grid_color.as_deref(), Some("#cccccc"));
        assert_eq!(parsed.axis.as_ref().unwrap().orient.as_deref(), Some("bottom"));
    }

    #[test]
    fn encoding_spec_per_channel_axis_unknown_key_fails_loud() {
        // A misspelled per-channel axis key must error (deny_unknown_fields),
        // not silently drop (B5 fail-loud).
        let json = r##"{"field":"x","axis":{"grid_colr":"#f00"}}"##;
        let parsed: Result<EncodingSpec, _> = serde_json::from_str(json);
        assert!(parsed.is_err(), "unknown per-channel axis key must fail to deserialize");
    }

    #[test]
    fn encoding_spec_per_channel_legend_unknown_key_fails_loud() {
        let json = r##"{"field":"c","legend":{"symbol_sze":10}}"##;
        let parsed: Result<EncodingSpec, _> = serde_json::from_str(json);
        assert!(parsed.is_err(), "unknown per-channel legend key must fail to deserialize");
    }

    #[test]
    fn encoding_spec_axis_camel_case_alias_deserializes() {
        // Raw-dict back-compat: camelCase `labelAngle` maps to `label_angle`.
        let json = r##"{"field":"x","axis":{"labelAngle":-30}}"##;
        let parsed: EncodingSpec = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.axis.as_ref().unwrap().label_angle, Some(-30.0));
    }

    #[test]
    fn encoding_spec_axis_false_suppression_round_trips() {
        // `axis=False` serializes (Python side) to the suppression form; the typed
        // struct must accept it and round-trip.
        let json = r##"{"field":"x","axis":{"domain":false,"ticks":false,"labels":false,"title":"","grid":false}}"##;
        let parsed: EncodingSpec = serde_json::from_str(json).unwrap();
        let axis = parsed.axis.as_ref().unwrap();
        assert_eq!(axis.domain, Some(false));
        assert_eq!(axis.ticks, Some(false));
        assert_eq!(axis.labels, Some(false));
        assert_eq!(axis.grid, Some(false));
        assert_eq!(axis.title.as_deref(), Some("")); // empty-string suppress sentinel
        // Re-serialize → re-parse stays equal.
        let back = serde_json::to_string(&parsed).unwrap();
        let reparsed: EncodingSpec = serde_json::from_str(&back).unwrap();
        assert_eq!(reparsed, parsed);
    }

    #[test]
    fn encoding_spec_legend_disabled_suppression_round_trips() {
        // `legend=None`/`False` serializes to `{"disabled": true}`.
        let json = r##"{"field":"c","legend":{"disabled":true}}"##;
        let parsed: EncodingSpec = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.legend.as_ref().unwrap().disabled, Some(true));
    }

    #[test]
    fn encoding_spec_phase_7_canonical_json_byte_identical_when_no_new_fields() {
        let e = EncodingSpec { field: "x".into(), type_: None, ..Default::default() };
        assert_eq!(serde_json::to_string(&e).unwrap(), r#"{"field":"x"}"#);

        let e2 = EncodingSpec {
            field: "y".into(),
            type_: Some(DataType::Quantitative),
            ..Default::default()
        };
        assert_eq!(
            serde_json::to_string(&e2).unwrap(),
            r#"{"field":"y","type":"quantitative"}"#,
        );
    }

    #[test]
    fn encoding_spec_round_trips_pre_phase_8_json() {
        // Existing JSON without any new fields must deserialize.
        let json = r#"{"field":"price","type":"quantitative"}"#;
        let parsed: EncodingSpec = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.field, "price");
        assert_eq!(parsed.type_, Some(DataType::Quantitative));
        assert_eq!(parsed.scale, None);
        assert_eq!(parsed.title, None);
    }

    // --- Phase 8b Task 22: x2 / y2 channels (ribbon support) ---

    #[test]
    fn encoding_round_trips_with_y2() {
        let e = Encoding {
            x: Some(EncodingSpec { field: "t".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "lo".into(), type_: None, ..Default::default() }),
            y2: Some(EncodingSpec { field: "hi".into(), type_: None, ..Default::default() }),
            ..Default::default()
        };
        let json = serde_json::to_string(&e).unwrap();
        assert!(json.contains(r#""y2":{"field":"hi"}"#), "json: {json}");
        let parsed: Encoding = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, e);
    }

    #[test]
    fn encoding_omits_x2_y2_when_none() {
        // Existing 8a JSON without x2/y2 must remain byte-identical.
        let e = Encoding {
            x: Some(EncodingSpec { field: "a".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "b".into(), type_: None, ..Default::default() }),
            ..Default::default()
        };
        let json = serde_json::to_string(&e).unwrap();
        assert_eq!(json, r#"{"x":{"field":"a"},"y":{"field":"b"}}"#);
        assert!(!json.contains("x2"));
        assert!(!json.contains("y2"));
    }

    #[test]
    fn scale_spec_log_default_base_is_10() {
        let json = r#"{"type":"log"}"#;
        let parsed: ScaleSpec = serde_json::from_str(json).unwrap();
        match parsed {
            ScaleSpec::Log { base, .. } => assert_eq!(base, 10.0),
            _ => panic!("expected Log variant"),
        }
    }

    #[test]
    fn scale_spec_pow_defaults() {
        let json = r#"{"type":"pow"}"#;
        let parsed: ScaleSpec = serde_json::from_str(json).unwrap();
        match parsed {
            ScaleSpec::Pow { exponent, common } => {
                assert_eq!(exponent, 2.0);
                assert!(!common.clamp);
                assert_eq!(common.padding, None);
            }
            _ => panic!("expected Pow variant"),
        }
    }

    #[test]
    fn scale_spec_sqrt_round_trip() {
        let json = r#"{"type":"sqrt","clamp":true}"#;
        let parsed: ScaleSpec = serde_json::from_str(json).unwrap();
        match &parsed {
            ScaleSpec::Sqrt { common } => assert!(common.clamp),
            _ => panic!("expected Sqrt variant"),
        }
        let re = serde_json::to_string(&parsed).unwrap();
        assert!(re.contains(r#""type":"sqrt""#));
    }

    #[test]
    fn scale_spec_utc_round_trip() {
        let json = r#"{"type":"utc","nice":true}"#;
        let parsed: ScaleSpec = serde_json::from_str(json).unwrap();
        match &parsed {
            ScaleSpec::Utc { nice, .. } => assert!(nice),
            _ => panic!("expected Utc variant"),
        }
    }

    #[test]
    fn scale_spec_band_defaults() {
        let json = r#"{"type":"band"}"#;
        let parsed: ScaleSpec = serde_json::from_str(json).unwrap();
        match parsed {
            ScaleSpec::Band { padding, align, padding_inner, padding_outer, range, .. } => {
                assert_eq!(padding, 0.1);
                assert_eq!(align, 0.5);
                assert_eq!(padding_inner, None);
                assert_eq!(padding_outer, None);
                assert_eq!(range, None);
            }
            _ => panic!("expected Band variant"),
        }
    }

    #[test]
    fn scale_spec_point_defaults() {
        let json = r#"{"type":"point"}"#;
        let parsed: ScaleSpec = serde_json::from_str(json).unwrap();
        match parsed {
            ScaleSpec::Point { padding, align, reverse, range, .. } => {
                assert_eq!(padding, 0.5);
                assert_eq!(align, 0.5);
                assert!(!reverse);
                assert_eq!(range, None);
            }
            _ => panic!("expected Point variant"),
        }
    }

    /// Issue #39: `ScaleSpec::Band { range }` round-trips through JSON when
    /// present, and the wire form carries the explicit pixel range key.
    #[test]
    fn scale_spec_band_range_round_trip() {
        let spec = ScaleSpec::Band {
            domain: None,
            padding: default_band_padding(),
            padding_inner: None,
            padding_outer: None,
            align: default_band_align(),
            range: Some(vec![40.0, 260.0]),
        };
        let json = serde_json::to_string(&spec).unwrap();
        assert!(json.contains(r#""range":[40.0,260.0]"#), "json={json}");
        let re_parsed: ScaleSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(re_parsed, spec);
    }

    /// Issue #39: with `range: None`, the JSON contains no `"range"` key
    /// (byte-identity guard — absent range must not perturb existing wire output).
    #[test]
    fn scale_spec_band_range_absent_omits_key() {
        let spec = ScaleSpec::Band {
            domain: None,
            padding: default_band_padding(),
            padding_inner: None,
            padding_outer: None,
            align: default_band_align(),
            range: None,
        };
        let json = serde_json::to_string(&spec).unwrap();
        assert!(!json.contains("\"range\""), "json={json}");
    }

    /// Issue #39: `ScaleSpec::Point { range }` round-trips through JSON when
    /// present, and the wire form carries the explicit pixel range key.
    #[test]
    fn scale_spec_point_range_round_trip() {
        let spec = ScaleSpec::Point {
            domain: None,
            padding: default_point_padding(),
            align: default_band_align(),
            reverse: false,
            range: Some(vec![40.0, 260.0]),
        };
        let json = serde_json::to_string(&spec).unwrap();
        assert!(json.contains(r#""range":[40.0,260.0]"#), "json={json}");
        let re_parsed: ScaleSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(re_parsed, spec);
    }

    /// Issue #39: with `range: None`, the JSON contains no `"range"` key
    /// (byte-identity guard — absent range must not perturb existing wire output).
    #[test]
    fn scale_spec_point_range_absent_omits_key() {
        let spec = ScaleSpec::Point {
            domain: None,
            padding: default_point_padding(),
            align: default_band_align(),
            reverse: false,
            range: None,
        };
        let json = serde_json::to_string(&spec).unwrap();
        assert!(!json.contains("\"range\""), "json={json}");
    }

    #[test]
    fn scale_spec_sequential_round_trip() {
        let json = r#"{"type":"sequential","scheme":"viridis","reverse":true}"#;
        let parsed: ScaleSpec = serde_json::from_str(json).unwrap();
        match &parsed {
            ScaleSpec::Sequential { scheme, reverse, .. } => {
                assert_eq!(scheme.as_deref(), Some("viridis"));
                assert!(reverse);
            }
            _ => panic!("expected Sequential variant"),
        }
    }

    /// F-L04-02 second revision (spec §4.2, amended 2026-08-28): with
    /// `stops: None`, the JSON contains no `"stops"` key — byte-identity
    /// guard for every pre-existing Sequential wire form (scheme-name-backed
    /// specs never carry stops, so this must not perturb them).
    #[test]
    fn scale_spec_sequential_stops_absent_omits_key() {
        let spec = ScaleSpec::Sequential {
            scheme: Some("viridis".to_string()),
            domain: None,
            reverse: false,
            stops: None,
        };
        let json = serde_json::to_string(&spec).unwrap();
        assert!(!json.contains("\"stops\""), "json={json}");
        let re_parsed: ScaleSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(re_parsed, spec);
    }

    /// F-L04-02 second revision (spec-reviewer cycle 3): a `Gradient`-backed
    /// scheme round-trips its `(t, hex)` stop pairs through the wire form
    /// byte-identically — carrying real, possibly non-uniform `t` positions,
    /// not just an ordered color list.
    #[test]
    fn scale_spec_sequential_stops_round_trip() {
        let spec = ScaleSpec::Sequential {
            scheme: None,
            domain: None,
            reverse: false,
            stops: Some(vec![
                (0.0, "#ff0000".to_string()),
                (0.9, "#00ff00".to_string()),
                (1.0, "#0000ff".to_string()),
            ]),
        };
        let json = serde_json::to_string(&spec).unwrap();
        assert!(
            json.contains("\"stops\":[[0.0,\"#ff0000\"],[0.9,\"#00ff00\"],[1.0,\"#0000ff\"]]"),
            "json={json}"
        );
        let re_parsed: ScaleSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(re_parsed, spec);
        match &re_parsed {
            ScaleSpec::Sequential { scheme, stops, .. } => {
                assert_eq!(scheme, &None);
                assert_eq!(
                    stops.as_deref(),
                    Some(
                        &[
                            (0.0, "#ff0000".to_string()),
                            (0.9, "#00ff00".to_string()),
                            (1.0, "#0000ff".to_string()),
                        ][..]
                    )
                );
            }
            _ => panic!("expected Sequential variant"),
        }
    }

    #[test]
    fn scale_spec_diverging_round_trip() {
        let json = r#"{"type":"diverging","scheme":"rdbu","domainMid":0.5}"#;
        let parsed: ScaleSpec = serde_json::from_str(json).unwrap();
        match &parsed {
            ScaleSpec::Diverging { scheme, domain_mid, .. } => {
                assert_eq!(scheme.as_deref(), Some("rdbu"));
                assert_eq!(*domain_mid, Some(0.5));
            }
            _ => panic!("expected Diverging variant"),
        }
    }

    #[test]
    fn scale_spec_quantize_round_trip() {
        let json = r##"{"type":"quantize","domain":[0,100],"range":["#f00","#0f0","#00f"]}"##;
        let parsed: ScaleSpec = serde_json::from_str(json).unwrap();
        match &parsed {
            ScaleSpec::Quantize { domain, range } => {
                assert_eq!(domain.as_ref().unwrap(), &vec![0.0, 100.0]);
                assert_eq!(range.as_ref().unwrap().len(), 3);
            }
            _ => panic!("expected Quantize variant"),
        }
    }

    #[test]
    fn scale_spec_quantile_round_trip() {
        let json = r#"{"type":"quantile","domain":[0,25,50,75,100],"range":[0,1,2,3]}"#;
        let parsed: ScaleSpec = serde_json::from_str(json).unwrap();
        match &parsed {
            ScaleSpec::Quantile { domain, range } => {
                assert_eq!(domain.as_ref().unwrap(), &vec![0.0, 25.0, 50.0, 75.0, 100.0]);
                assert_eq!(range.as_ref().unwrap().len(), 4);
            }
            _ => panic!("expected Quantile variant"),
        }
        let re = serde_json::to_string(&parsed).unwrap();
        assert!(re.contains(r#""type":"quantile""#), "json: {re}");
    }

    #[test]
    fn scale_spec_threshold_round_trip() {
        let json = r#"{"type":"threshold","domain":[0,50,100],"range":[0,1,2,3]}"#;
        let parsed: ScaleSpec = serde_json::from_str(json).unwrap();
        match &parsed {
            ScaleSpec::Threshold { domain, range } => {
                assert_eq!(domain.as_ref().unwrap(), &vec![0.0, 50.0, 100.0]);
                // range.len() == domain.len() + 1
                assert_eq!(range.as_ref().unwrap().len(), 4);
            }
            _ => panic!("expected Threshold variant"),
        }
        let re = serde_json::to_string(&parsed).unwrap();
        assert!(re.contains(r#""type":"threshold""#), "json: {re}");
    }

    #[test]
    fn scale_spec_bin_ordinal_round_trip() {
        let json = r#"{"type":"bin-ordinal","bins":[0,10,20,30],"scheme":"blues"}"#;
        let parsed: ScaleSpec = serde_json::from_str(json).unwrap();
        match &parsed {
            ScaleSpec::BinOrdinal { bins, scheme } => {
                assert_eq!(bins.as_ref().unwrap(), &vec![0.0, 10.0, 20.0, 30.0]);
                assert_eq!(scheme.as_deref(), Some("blues"));
            }
            _ => panic!("expected BinOrdinal variant"),
        }
        // Verify serialization uses "bin-ordinal" as the type tag
        let re = serde_json::to_string(&parsed).unwrap();
        assert!(re.contains(r#""type":"bin-ordinal""#), "json: {re}");
    }

    #[test]
    fn scale_spec_band_with_padding_inner_outer() {
        let json = r#"{"type":"band","paddingInner":0.2,"paddingOuter":0.1,"align":0.3}"#;
        let parsed: ScaleSpec = serde_json::from_str(json).unwrap();
        match parsed {
            ScaleSpec::Band { padding_inner, padding_outer, align, .. } => {
                assert_eq!(padding_inner, Some(0.2));
                assert_eq!(padding_outer, Some(0.1));
                assert_eq!(align, 0.3);
            }
            _ => panic!("expected Band variant"),
        }
    }

    #[test]
    fn inherit_from_propagates_title_on_same_field() {
        let parent = Encoding {
            x: Some(EncodingSpec {
                field: "fpr".into(),
                title: Some("False Positive Rate".into()),
                ..Default::default()
            }),
            y: Some(EncodingSpec {
                field: "tpr".into(),
                title: Some("True Positive Rate".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut child = Encoding {
            x: Some(EncodingSpec { field: "fpr".into(), ..Default::default() }),
            y: Some(EncodingSpec { field: "tpr".into(), ..Default::default() }),
            ..Default::default()
        };
        child.inherit_from(&parent);
        assert_eq!(child.x.as_ref().unwrap().title.as_deref(), Some("False Positive Rate"));
        assert_eq!(child.y.as_ref().unwrap().title.as_deref(), Some("True Positive Rate"));
    }

    #[test]
    fn inherit_from_propagates_scheme_on_same_field() {
        let parent = Encoding {
            color: Some(EncodingSpec {
                field: "species".into(),
                scheme: Some("paper_ink".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut child = Encoding {
            color: Some(EncodingSpec { field: "species".into(), ..Default::default() }),
            ..Default::default()
        };
        child.inherit_from(&parent);
        assert_eq!(child.color.as_ref().unwrap().scheme.as_deref(), Some("paper_ink"));
    }

    #[test]
    fn inherit_from_does_not_overwrite_child_title() {
        let parent = Encoding {
            x: Some(EncodingSpec {
                field: "fpr".into(),
                title: Some("Parent Title".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut child = Encoding {
            x: Some(EncodingSpec {
                field: "fpr".into(),
                title: Some("Child Title".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        child.inherit_from(&parent);
        assert_eq!(child.x.as_ref().unwrap().title.as_deref(), Some("Child Title"));
    }

    #[test]
    fn inherit_from_does_not_cross_pollinate_different_fields() {
        let parent = Encoding {
            x: Some(EncodingSpec {
                field: "fpr".into(),
                title: Some("False Positive Rate".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut child = Encoding {
            x: Some(EncodingSpec { field: "other_field".into(), ..Default::default() }),
            ..Default::default()
        };
        child.inherit_from(&parent);
        assert_eq!(child.x.as_ref().unwrap().title, None);
    }

    // --- Encoding inheritance edge-case tests ---

    #[test]
    fn inherit_from_partial_encoding_only_fills_absent_channels() {
        // Parent has x, y, color, size. Child only has x, color.
        // Expected: child.y and child.size adopt parent values;
        // child.x and child.color remain untouched.
        let parent = Encoding {
            x: Some(EncodingSpec { field: "px".into(), title: Some("Parent X".into()), ..Default::default() }),
            y: Some(EncodingSpec { field: "py".into(), title: Some("Parent Y".into()), ..Default::default() }),
            color: Some(EncodingSpec { field: "pc".into(), scheme: Some("blues".into()), ..Default::default() }),
            size: Some(EncodingSpec { field: "ps".into(), ..Default::default() }),
            ..Default::default()
        };
        let mut child = Encoding {
            x: Some(EncodingSpec { field: "cx".into(), ..Default::default() }),
            color: Some(EncodingSpec { field: "cc".into(), ..Default::default() }),
            ..Default::default()
        };
        child.inherit_from(&parent);
        // x field differs → no metadata propagation
        assert_eq!(child.x.as_ref().unwrap().field, "cx");
        assert_eq!(child.x.as_ref().unwrap().title, None);
        // y was None → adopted from parent
        assert_eq!(child.y.as_ref().unwrap().field, "py");
        assert_eq!(child.y.as_ref().unwrap().title.as_deref(), Some("Parent Y"));
        // color field differs → no metadata propagation
        assert_eq!(child.color.as_ref().unwrap().field, "cc");
        assert_eq!(child.color.as_ref().unwrap().scheme, None);
        // size was None → adopted from parent
        assert_eq!(child.size.as_ref().unwrap().field, "ps");
    }

    #[test]
    fn inherit_from_propagates_all_metadata_on_same_field() {
        // When field matches, scale + title + scheme + type_ + axis + legend + format + format_type
        // all propagate if child has None for that slot.
        let parent = Encoding {
            x: Some(EncodingSpec {
                field: "val".into(),
                type_: Some(DataType::Quantitative),
                scale: Some(ScaleSpec::Log { base: 2.0, common: ContinuousScaleCommon { domain: None, range: None, clamp: false, padding: None, scheme: None, domain_param: None, reverse: false }, nice: true }),
                title: Some("Value (log2)".into()),
                axis: Some(Box::new(crate::render::chart_config::AxisStyleSpec {
                    grid: Some(false),
                    ..Default::default()
                })),
                legend: Some(Box::new(crate::render::chart_config::LegendStyleSpec::default())),
                format: Some(".2f".into()),
                format_type: Some("number".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut child = Encoding {
            x: Some(EncodingSpec { field: "val".into(), ..Default::default() }),
            ..Default::default()
        };
        child.inherit_from(&parent);
        let cx = child.x.as_ref().unwrap();
        assert!(cx.scale.is_some(), "scale should be inherited");
        assert_eq!(cx.title.as_deref(), Some("Value (log2)"));
        assert_eq!(cx.type_, Some(DataType::Quantitative));
        assert!(cx.axis.is_some(), "axis should be inherited");
        assert!(cx.legend.is_some(), "legend should be inherited");
        assert_eq!(cx.format.as_deref(), Some(".2f"));
        assert_eq!(cx.format_type.as_deref(), Some("number"));
    }

    #[test]
    fn inherit_from_does_not_overwrite_any_child_metadata() {
        // Child has all metadata set; parent has different values for each.
        // None of the child's values should be replaced.
        let parent = Encoding {
            x: Some(EncodingSpec {
                field: "val".into(),
                type_: Some(DataType::Temporal),
                scale: Some(ScaleSpec::Linear { common: ContinuousScaleCommon { domain: None, range: None, clamp: false, padding: None, scheme: None, domain_param: None, reverse: false }, nice: true, zero: false }),
                title: Some("Parent Title".into()),
                scheme: Some("parent_scheme".into()),
                format: Some("parent_fmt".into()),
                format_type: Some("parent_ft".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut child = Encoding {
            x: Some(EncodingSpec {
                field: "val".into(),
                type_: Some(DataType::Quantitative),
                scale: Some(ScaleSpec::Log { base: 10.0, common: ContinuousScaleCommon { domain: None, range: None, clamp: false, padding: None, scheme: None, domain_param: None, reverse: false }, nice: false }),
                title: Some("Child Title".into()),
                scheme: Some("child_scheme".into()),
                format: Some("child_fmt".into()),
                format_type: Some("child_ft".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        child.inherit_from(&parent);
        let cx = child.x.as_ref().unwrap();
        assert_eq!(cx.type_, Some(DataType::Quantitative));
        assert!(matches!(cx.scale, Some(ScaleSpec::Log { .. })));
        assert_eq!(cx.title.as_deref(), Some("Child Title"));
        assert_eq!(cx.scheme.as_deref(), Some("child_scheme"));
        assert_eq!(cx.format.as_deref(), Some("child_fmt"));
        assert_eq!(cx.format_type.as_deref(), Some("child_ft"));
    }

    #[test]
    fn inherit_non_positional_skips_x_y_x2_y2() {
        // Parent has x, y, x2, y2, color, size. Child has nothing.
        // inherit_non_positional must adopt color and size but NOT x/y/x2/y2.
        let parent = Encoding {
            x: Some(EncodingSpec { field: "px".into(), ..Default::default() }),
            y: Some(EncodingSpec { field: "py".into(), ..Default::default() }),
            x2: Some(EncodingSpec { field: "px2".into(), ..Default::default() }),
            y2: Some(EncodingSpec { field: "py2".into(), ..Default::default() }),
            color: Some(EncodingSpec { field: "pc".into(), scheme: Some("reds".into()), ..Default::default() }),
            size: Some(EncodingSpec { field: "ps".into(), ..Default::default() }),
            opacity: Some(EncodingSpec { field: "po".into(), ..Default::default() }),
            ..Default::default()
        };
        let mut child = Encoding::default();
        child.inherit_non_positional(&parent);
        // Positional channels must remain None
        assert!(child.x.is_none(), "x should not be inherited");
        assert!(child.y.is_none(), "y should not be inherited");
        assert!(child.x2.is_none(), "x2 should not be inherited");
        assert!(child.y2.is_none(), "y2 should not be inherited");
        // Non-positional channels must be inherited
        assert_eq!(child.color.as_ref().unwrap().field, "pc");
        assert_eq!(child.color.as_ref().unwrap().scheme.as_deref(), Some("reds"));
        assert_eq!(child.size.as_ref().unwrap().field, "ps");
        assert_eq!(child.opacity.as_ref().unwrap().field, "po");
    }

    #[test]
    fn inherit_non_positional_propagates_metadata_on_same_field() {
        // Child has color with same field but no scheme; parent has scheme.
        let parent = Encoding {
            color: Some(EncodingSpec {
                field: "species".into(),
                scheme: Some("dark2".into()),
                title: Some("Species Group".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut child = Encoding {
            color: Some(EncodingSpec { field: "species".into(), ..Default::default() }),
            ..Default::default()
        };
        child.inherit_non_positional(&parent);
        assert_eq!(child.color.as_ref().unwrap().scheme.as_deref(), Some("dark2"));
        assert_eq!(child.color.as_ref().unwrap().title.as_deref(), Some("Species Group"));
    }

    #[test]
    fn inherit_from_propagates_stroke_and_angle_channels() {
        // Verify that the late-addition stroke_width/stroke_opacity/stroke_dash/angle/fill_opacity
        // channels are properly inherited when child is None.
        let parent = Encoding {
            stroke_width: Some(EncodingSpec { field: "sw".into(), ..Default::default() }),
            stroke_opacity: Some(EncodingSpec { field: "so".into(), ..Default::default() }),
            stroke_dash: Some(EncodingSpec { field: "sd".into(), ..Default::default() }),
            angle: Some(EncodingSpec { field: "a".into(), ..Default::default() }),
            fill_opacity: Some(EncodingSpec { field: "fo".into(), ..Default::default() }),
            ..Default::default()
        };
        let mut child = Encoding::default();
        child.inherit_from(&parent);
        assert_eq!(child.stroke_width.as_ref().unwrap().field, "sw");
        assert_eq!(child.stroke_opacity.as_ref().unwrap().field, "so");
        assert_eq!(child.stroke_dash.as_ref().unwrap().field, "sd");
        assert_eq!(child.angle.as_ref().unwrap().field, "a");
        assert_eq!(child.fill_opacity.as_ref().unwrap().field, "fo");
    }

    #[test]
    fn inherit_from_tooltip_fields_only_fills_when_child_is_none() {
        // tooltip_fields inheritance: child None → adopt parent; child Some → keep child.
        let parent = Encoding {
            tooltip_fields: Some(vec![
                EncodingSpec { field: "a".into(), ..Default::default() },
                EncodingSpec { field: "b".into(), ..Default::default() },
            ]),
            ..Default::default()
        };
        // Case 1: child is None → inherits
        let mut child1 = Encoding::default();
        child1.inherit_from(&parent);
        assert_eq!(child1.tooltip_fields.as_ref().unwrap().len(), 2);

        // Case 2: child already has tooltip_fields → keeps own
        let mut child2 = Encoding {
            tooltip_fields: Some(vec![
                EncodingSpec { field: "only_mine".into(), ..Default::default() },
            ]),
            ..Default::default()
        };
        child2.inherit_from(&parent);
        assert_eq!(child2.tooltip_fields.as_ref().unwrap().len(), 1);
        assert_eq!(child2.tooltip_fields.as_ref().unwrap()[0].field, "only_mine");
    }

    #[test]
    fn inherit_from_completely_empty_parent_is_noop() {
        let parent = Encoding::default();
        let mut child = Encoding {
            x: Some(EncodingSpec { field: "cx".into(), title: Some("My X".into()), ..Default::default() }),
            color: Some(EncodingSpec { field: "cc".into(), ..Default::default() }),
            ..Default::default()
        };
        let child_before = child.clone();
        child.inherit_from(&parent);
        assert_eq!(child, child_before);
    }

    #[test]
    fn inherit_from_completely_empty_child_adopts_everything() {
        let parent = Encoding {
            x: Some(EncodingSpec { field: "px".into(), ..Default::default() }),
            y: Some(EncodingSpec { field: "py".into(), ..Default::default() }),
            color: Some(EncodingSpec { field: "pc".into(), ..Default::default() }),
            size: Some(EncodingSpec { field: "ps".into(), ..Default::default() }),
            shape: Some(EncodingSpec { field: "psh".into(), ..Default::default() }),
            opacity: Some(EncodingSpec { field: "po".into(), ..Default::default() }),
            text: Some(EncodingSpec { field: "pt".into(), ..Default::default() }),
            ..Default::default()
        };
        let mut child = Encoding::default();
        child.inherit_from(&parent);
        assert_eq!(child.x.as_ref().unwrap().field, "px");
        assert_eq!(child.y.as_ref().unwrap().field, "py");
        assert_eq!(child.color.as_ref().unwrap().field, "pc");
        assert_eq!(child.size.as_ref().unwrap().field, "ps");
        assert_eq!(child.shape.as_ref().unwrap().field, "psh");
        assert_eq!(child.opacity.as_ref().unwrap().field, "po");
        assert_eq!(child.text.as_ref().unwrap().field, "pt");
    }

    #[test]
    fn overlay_from_replaces_present_channels() {
        let mut base = Encoding::default();
        base.x = Some(EncodingSpec { field: "base_x".into(), ..Default::default() });
        base.y = Some(EncodingSpec { field: "base_y".into(), ..Default::default() });

        let mut overlay = Encoding::default();
        overlay.x = Some(EncodingSpec { field: "overlay_x".into(), ..Default::default() });
        // overlay.y is None — should NOT replace base.y

        base.overlay_from(&overlay);
        assert_eq!(base.x.as_ref().unwrap().field, "overlay_x");
        assert_eq!(base.y.as_ref().unwrap().field, "base_y");
    }

    #[test]
    fn overlay_from_covers_all_12_channels() {
        let mut base = Encoding {
            x: Some(EncodingSpec { field: "bx".into(), ..Default::default() }),
            y: Some(EncodingSpec { field: "by".into(), ..Default::default() }),
            color: Some(EncodingSpec { field: "bc".into(), ..Default::default() }),
            size: Some(EncodingSpec { field: "bs".into(), ..Default::default() }),
            shape: Some(EncodingSpec { field: "bsh".into(), ..Default::default() }),
            opacity: Some(EncodingSpec { field: "bo".into(), ..Default::default() }),
            x2: Some(EncodingSpec { field: "bx2".into(), ..Default::default() }),
            y2: Some(EncodingSpec { field: "by2".into(), ..Default::default() }),
            text: Some(EncodingSpec { field: "bt".into(), ..Default::default() }),
            tooltip: Some(EncodingSpec { field: "btt".into(), ..Default::default() }),
            tooltip_fields: None,
            href: Some(EncodingSpec { field: "bh".into(), ..Default::default() }),
            description: Some(EncodingSpec { field: "bd".into(), ..Default::default() }),
            key: Some(EncodingSpec { field: "bk".into(), ..Default::default() }),
            url: None,
            stroke_width: None,
            stroke_opacity: None,
            stroke_dash: None,
            angle: None,
            fill_opacity: None,
        };
        // Overlay only tooltip, href, description (the three that were missed
        // by the old inline merge).
        let overlay = Encoding {
            tooltip: Some(EncodingSpec { field: "ott".into(), ..Default::default() }),
            href: Some(EncodingSpec { field: "oh".into(), ..Default::default() }),
            description: Some(EncodingSpec { field: "od".into(), ..Default::default() }),
            ..Default::default()
        };
        base.overlay_from(&overlay);
        // Replaced channels:
        assert_eq!(base.tooltip.as_ref().unwrap().field, "ott");
        assert_eq!(base.href.as_ref().unwrap().field, "oh");
        assert_eq!(base.description.as_ref().unwrap().field, "od");
        // Untouched channels:
        assert_eq!(base.x.as_ref().unwrap().field, "bx");
        assert_eq!(base.y.as_ref().unwrap().field, "by");
        assert_eq!(base.color.as_ref().unwrap().field, "bc");
    }
}
