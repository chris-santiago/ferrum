//! The scale wire-key gate: `ScaleSpec`'s own `Serialize`/`Deserialize`
//! impls (the `remote = "Self"` wrapper that runs `validate_scale_spec_keys`
//! before the derived deserialize), the accepted-key tables,
//! `scale_accepted_keys`/`validate_scale_dict` (the two Python-visible gate
//! pyfunctions), and the raw-dict temporal-domain conversion
//! (`convert_raw_dict_temporal_domain`) that `deserialize_scale_value` runs
//! ahead of it.
//!
//! Wave 6A `#158`: split out of `spec/encoding.rs` past the CLAUDE.md
//! test-vs-source promotion trigger (the file's test body, 1626 lines,
//! exceeded its 1454-line source body). This cluster is the crate's single
//! "every user scale dict passes exactly once" chokepoint, documented at the
//! `Deserialize` impl below and pinned by its own drift test
//! (`accepted_keys_for_scale_type_matches_every_variants_serialized_keys`),
//! so it moves whole rather than scattering production code away from the
//! tests that pin it. `deserialize_scale_value` is `pub(super)` — `mod.rs`'s
//! `EncodingSpec::new` calls it directly, the one widening this split needs;
//! every other item here has no caller outside this file.
//!
//! Mirrors the `transform/smooth/fits.rs` / `transform/smooth/binding.rs`
//! split (wave 6A `#163`, landed `66526d33`) and the `render/scale_resolve/`
//! / `transform/stats/` promoted-directory precedent, with one difference:
//! this file keeps its own inline `#[cfg(test)] mod tests` rather than
//! handing its tests to a shared sibling `tests.rs`, because those tests
//! exercise several helpers here (`accepted_keys_for_scale_type`,
//! `convert_raw_dict_temporal_domain`) that stay private to this file —
//! a sibling test module could only reach them by widening past this file's
//! own tree, which the rest of the tests never need. See
//! `spec/encoding/mod.rs`'s doc comment for where the remaining
//! `DataType`/`EncodingSpec`/`Encoding` round-trip tests (unrelated to the
//! gate) landed instead.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde::{Deserialize, Serialize};

use super::ScaleSpec;

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
        "linear" => CONTINUOUS_COMMON_SCALE_KEYS
            .iter()
            .chain(["nice", "zero", "systemInjected"].iter())
            .copied()
            .collect(),
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

/// Convert + deserialize a scale value into a `ScaleSpec`, exactly as
/// `EncodingSpec::new`'s `scale=` parameter does: raw-dict temporal-domain
/// conversion first (`convert_raw_dict_temporal_domain`), then a JSON
/// round-trip through `ScaleSpec`'s own gated `Deserialize` impl
/// (`validate_scale_spec_keys`, run at that impl's own boundary).
///
/// **The single chokepoint both `EncodingSpec::new` and [`validate_scale_dict`]
/// call** — batch-C close remediation, converging the rust- and
/// python-design reviews' independent findings on the same fix: neither
/// caller re-implements the dict→JSON→`ScaleSpec` sequence, so there is no
/// second validation path that could drift from this one. Error messages are
/// therefore byte-identical between the two callers by construction, not by
/// coincidence — pinned by
/// `validate_scale_dict_message_matches_encoding_spec_route` below.
pub(super) fn deserialize_scale_value(
    py: Python,
    scale: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<ScaleSpec>> {
    let scale_converted = convert_raw_dict_temporal_domain(scale)?;
    let Some(obj) = scale_converted else { return Ok(None) };
    let json_module = py.import("json")?;
    let s: String = json_module.call_method1("dumps", (obj,))?.extract()?;
    Ok(Some(
        serde_json::from_str(&s).map_err(|e| PyValueError::new_err(format!("scale: {e}")))?,
    ))
}

/// Validate a raw scale dict against the SAME gate `EncodingSpec(field,
/// scale={...})` enforces, without constructing an `EncodingSpec` for the
/// side effect of its exception (rust-design review recommendation 5 /
/// python-design review's S2 on the probe idiom, converged): raises the
/// gate's own `ValueError` — byte-identical to the `EncodingSpec` route's
/// message, since both call [`deserialize_scale_value`] — on an invalid
/// dict, returns `None` on a valid one.
///
/// Exists so `_override_consume.py`'s override-scale merge can ask "is this scale
/// dict valid?" directly instead of via
/// `_EncodingSpec("<override-scale-probe>", scale={...})`, a throwaway
/// production object built purely to catch its constructor's exception. A
/// `{"type": "time"/"utc", "domain": [...]}` dict carrying `datetime.date`/
/// `datetime.datetime` elements validates rather than exploding on
/// `json.dumps`, exactly like the `EncodingSpec` route: `deserialize_scale_value`
/// runs `convert_raw_dict_temporal_domain` first, before any JSON encoding
/// is attempted.
#[pyfunction]
pub fn validate_scale_dict(py: Python, scale: &Bound<'_, PyDict>) -> PyResult<()> {
    deserialize_scale_value(py, Some(scale.as_any())).map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{ContinuousScaleCommon, EncodingSpec};

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
            ScaleSpec::Linear { common: maximal_common(), nice: true, zero: true, system_injected: true },
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
        let scale = ScaleSpec::Linear { common: maximal_common(), nice: true, zero: false, system_injected: false };
        let json = serde_json::to_string(&scale).unwrap();
        assert!(json.contains(r#""type":"linear""#), "json: {json}");
        // Round-trips through the same gated Deserialize.
        let back: ScaleSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back, scale);
    }

    // ── #247A scale provenance (wire contract, spec §6) ──────────────────────

    /// The wire payload a bar chart's injected y scale serialized to BEFORE the
    /// provenance field existed, captured verbatim: it must still deserialize,
    /// report no provenance, and re-serialize to those same bytes. This is the
    /// byte-stability proof asked for as a round-trip of a pre-change payload
    /// rather than as a reading of the serde attributes — a `skip_serializing_if`
    /// typo'd onto the wrong predicate would pass an attribute inspection and
    /// fail here.
    #[test]
    fn pre_provenance_payload_round_trips_to_identical_bytes() {
        const CAPTURED: &str = r#"{"type":"linear","clamp":false,"nice":false,"zero":true}"#;
        let scale: ScaleSpec = serde_json::from_str(CAPTURED).unwrap();
        assert!(!scale.is_system_injected(), "an absent flag must not read as injected");
        assert!(scale.anchors_zero(), "the captured payload's own zero anchor");
        assert_eq!(serde_json::to_string(&scale).unwrap(), CAPTURED);
    }

    /// The flag travels: set on the wire, it survives into
    /// `is_system_injected()` and back out to the same key. `anchors_zero` is
    /// asserted alongside it because the two are read independently by
    /// `composite::leaf_positional_domain` — one decides whether the leaf
    /// shares, the other what it contributes.
    #[test]
    fn system_injected_flag_round_trips_and_is_readable() {
        const WIRE: &str =
            r#"{"type":"linear","clamp":false,"nice":false,"zero":true,"systemInjected":true}"#;
        let scale: ScaleSpec = serde_json::from_str(WIRE).unwrap();
        assert!(scale.is_system_injected());
        assert!(scale.anchors_zero());
        assert_eq!(serde_json::to_string(&scale).unwrap(), WIRE);
    }

    /// Provenance is `Linear`-only, which the key gate enforces rather than
    /// leaving serde's flatten to swallow: the injection has no other variant to
    /// target (`zero` is Linear's own field), so the same key on a `log` scale is
    /// a typo or a misunderstanding, refused naming the accepted set.
    #[test]
    fn system_injected_is_refused_on_a_non_linear_scale_type() {
        let err = serde_json::from_str::<ScaleSpec>(r#"{"type":"log","systemInjected":true}"#)
            .unwrap_err()
            .to_string();
        assert!(err.contains("unknown key 'systemInjected' for type 'log'"), "err: {err}");
        assert!(!ScaleSpec::Log { base: 10.0, common: maximal_common(), nice: false }
            .is_system_injected());
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

    // ── `validate_scale_dict` (batch-C close remediation) ───────────────────

    /// A valid dict for a representative sample of scale types returns
    /// `None` (no error) rather than raising.
    #[test]
    fn validate_scale_dict_accepts_representative_valid_dicts() {
        attach(|py| {
            let linear = PyDict::new(py);
            linear.set_item("type", "linear").unwrap();
            linear.set_item("domain", (0.0, 1.0)).unwrap();
            linear.set_item("clamp", true).unwrap();

            let band = PyDict::new(py);
            band.set_item("type", "band").unwrap();
            band.set_item("padding", 0.2).unwrap();
            band.set_item("align", 0.5).unwrap();

            let point = PyDict::new(py);
            point.set_item("type", "point").unwrap();
            point.set_item("reverse", true).unwrap();

            let diverging = PyDict::new(py);
            diverging.set_item("type", "diverging").unwrap();
            diverging.set_item("domainMid", 0.0).unwrap();

            let quantile = PyDict::new(py);
            quantile.set_item("type", "quantile").unwrap();
            quantile.set_item("domain", (1.0, 2.0, 3.0)).unwrap();

            for dict in [linear, band, point, diverging, quantile] {
                let scale_type: String = dict.get_item("type").unwrap().unwrap().extract().unwrap();
                validate_scale_dict(py, &dict)
                    .unwrap_or_else(|e| panic!("expected type '{scale_type}' to validate, got {e}"));
            }
        });
    }

    /// An unknown key refuses with a `ValueError` naming the key, the type,
    /// and the accepted set — the same shape `accepted_keys_for_scale_type`'s
    /// gate produces everywhere else.
    #[test]
    fn validate_scale_dict_refuses_unknown_key() {
        attach(|py| {
            let d = PyDict::new(py);
            d.set_item("type", "linear").unwrap();
            d.set_item("clammp", true).unwrap();
            let err = validate_scale_dict(py, &d).unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py));
            let msg = err.value(py).to_string();
            assert!(msg.contains("clammp") && msg.contains("linear"), "msg: {msg}");
        });
    }

    /// An unknown `"type"` tag falls through to the derived deserialize's own
    /// "unknown variant" error, exactly like the `EncodingSpec` route.
    #[test]
    fn validate_scale_dict_refuses_unknown_tag() {
        attach(|py| {
            let d = PyDict::new(py);
            d.set_item("type", "not-a-real-type").unwrap();
            let err = validate_scale_dict(py, &d).unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py));
        });
    }

    /// A non-string `"type"` also refuses (the gate itself is permissive on
    /// this shape and defers to the derived deserialize's type-mismatch
    /// error, which still raises).
    #[test]
    fn validate_scale_dict_refuses_non_str_tag() {
        attach(|py| {
            let d = PyDict::new(py);
            d.set_item("type", 123).unwrap();
            let err = validate_scale_dict(py, &d).unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py));
        });
    }

    /// The load-bearing claim: `validate_scale_dict`'s error message is
    /// BYTE-IDENTICAL to constructing an `EncodingSpec` with the same
    /// offending dict — both call `deserialize_scale_value`, so this is a
    /// property of the shared chokepoint, not a coincidence to maintain by
    /// hand.
    #[test]
    fn validate_scale_dict_message_matches_encoding_spec_route() {
        attach(|py| {
            let d = PyDict::new(py);
            d.set_item("type", "linear").unwrap();
            d.set_item("clammp", true).unwrap();
            let direct_err = validate_scale_dict(py, &d).unwrap_err();

            let encoding_err = EncodingSpec::new(
                py, "x", None, Some(d.as_any()), None, None, None, None, None, None, None,
                None, None, None,
            )
            .unwrap_err();

            assert_eq!(direct_err.value(py).to_string(), encoding_err.value(py).to_string());
        });
    }

    /// A datetime-bearing `{"type": "time", "domain": [...]}` dict validates
    /// rather than exploding on `json.dumps` — the raw-dict temporal-domain
    /// conversion runs before any JSON encoding is attempted, exactly like
    /// the `EncodingSpec` route.
    #[test]
    fn validate_scale_dict_accepts_datetime_bearing_temporal_domain() {
        attach(|py| {
            let d = PyDict::new(py);
            d.set_item("type", "time").unwrap();
            let date_lo = pyo3::types::PyDate::new(py, 2020, 1, 1).unwrap();
            let date_hi = pyo3::types::PyDate::new(py, 2020, 12, 31).unwrap();
            d.set_item("domain", (date_lo, date_hi)).unwrap();
            validate_scale_dict(py, &d).unwrap_or_else(|e| {
                panic!("expected a datetime-bearing temporal domain to validate, got {e}")
            });
        });
    }
}
