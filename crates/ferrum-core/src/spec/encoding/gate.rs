//! The scale wire-key gate: `ScaleSpec`'s own `Serialize`/`Deserialize`
//! impls (the `remote = "Self"` wrapper that runs `validate_scale_spec_keys`
//! before the derived deserialize), the accepted-key tables,
//! `scale_accepted_keys`/`validate_scale_dict` (the two Python-visible gate
//! pyfunctions), `translate_scale_switch` (what a scale-type switch keeps,
//! read off the same accepted-key table), and the raw-dict temporal-domain conversion
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
//!
//! **Known blind spot: `#[serde(alias = ...)]` (`#159`).**
//! `accepted_keys_for_scale_type` is derived from each variant's WIRE-spelled
//! field names (the `#[serde(rename = ...)]` destination, where present) —
//! it has no notion of a field also carrying `#[serde(alias = ...)]`. If a
//! future field grows one, `validate_scale_spec_keys` would refuse the alias
//! spelling as an "unknown key" even though the derived `Deserialize` this
//! gate wraps would happily accept it: the gate and the type it guards would
//! silently disagree on what a valid dict looks like. No `ScaleSpec` field
//! carries `#[serde(alias)]` today, and `no_scale_spec_field_carries_a_serde_alias`
//! below scans this module's source files to keep it that way.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde::{Deserialize, Serialize};

use super::sort::{SortOp, SortOrder};
use super::ScaleSpec;
use crate::scale::discrete::{check_align, check_ordinal_padding, check_point_padding, PaddingMeaning};

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
        "bin-ordinal" => vec!["bins", "scheme", "bin"],
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
    Ok(known_scale_type_keys(scale_type)?.into_iter().map(str::to_string).collect())
}

/// [`accepted_keys_for_scale_type`], refusing an unknown tag with the
/// `ValueError` both Python-visible readers of the table raise.
fn known_scale_type_keys(scale_type: &str) -> PyResult<Vec<&'static str>> {
    accepted_keys_for_scale_type(scale_type).ok_or_else(|| unknown_scale_type(scale_type))
}

fn unknown_scale_type(scale_type: &str) -> PyErr {
    PyValueError::new_err(format!("unknown scale type '{scale_type}'"))
}

/// The padding-family keys (`padding`, `paddingInner`, `paddingOuter`,
/// `align`) a `ScaleSpec` `"type"` tag accepts: the keys
/// [`translate_scale_switch`] translates by meaning rather than by name.
/// Published so a caller selecting a scale's padding family reads the set
/// instead of restating it. Raises the same `ValueError` as
/// [`scale_accepted_keys`] for an unknown tag.
#[pyfunction]
pub fn scale_padding_keys(scale_type: &str) -> PyResult<Vec<&'static str>> {
    let (_, meaning) = switch_side(scale_type)?;
    Ok(meaning.keys().to_vec())
}

/// One side of a scale-type switch: the tag's accepted keys and what its
/// padding-family keys mean, refusing an unknown tag.
fn switch_side(scale_type: &str) -> PyResult<(Vec<&'static str>, PaddingMeaning)> {
    let keys = known_scale_type_keys(scale_type)?;
    let meaning = padding_meaning_for_scale_type(scale_type).ok_or_else(|| unknown_scale_type(scale_type))?;
    Ok((keys, meaning))
}

/// What a `ScaleSpec` `"type"` tag's padding-family keys mean, or `None` for
/// an unknown tag. Beside [`accepted_keys_for_scale_type`] because the two
/// must agree: a meaning's [`keys`](PaddingMeaning::keys) are exactly the
/// padding-family keys the tag accepts (pinned by
/// `padding_meaning_keys_match_each_types_accepted_keys`).
fn padding_meaning_for_scale_type(scale_type: &str) -> Option<PaddingMeaning> {
    let meaning = match scale_type {
        "linear" | "log" | "time" | "symlog" | "pow" | "sqrt" | "utc" => PaddingMeaning::SpanFraction,
        "ordinal" => PaddingMeaning::OrdinalInner,
        "band" => PaddingMeaning::Band,
        "point" => PaddingMeaning::Point,
        "sequential" | "diverging" | "quantize" | "quantile" | "threshold" | "bin-ordinal" => {
            PaddingMeaning::Unpadded
        }
        _ => return None,
    };
    Some(meaning)
}

/// A channel's scale dict, carried across a switch from `old_type` to
/// `new_type`, and the keys the switch drops: the one owner of what a
/// scale-type switch keeps.
///
/// - The padding-family keys the old type accepts (`padding`,
///   `paddingInner`, `paddingOuter`, `align`) are translated by meaning
///   ([`PaddingMeaning::switch_carries`]): an ordinal inner gap becomes a
///   band inner gap, a point's end padding becomes band outer padding, and a
///   value with no meaning under the new type is dropped. A key whose value
///   is `None` is unset, so it is carried by nothing.
/// - Every other key the old type accepts survives when the new type also
///   accepts it, and is dropped otherwise.
/// - A key the old type does not accept (including `"type"`) passes through
///   untouched, for the scale gate to judge, unless the switch writes a key
///   of the same name: then the old scale states that key in a type that
///   rejects it, and this refuses with a `ValueError` naming the key rather
///   than let the translation overwrite it.
///
/// The second element, `dropped`, lists in `base` order every key of `base`
/// whose value the switch does not write onto the new scale: the keys the
/// new type does not accept, the padding-family keys no carry reads (a
/// continuous span fraction switched to a discrete type, an ordinal gap
/// switched to point, a band's outer side and `align` switched to ordinal),
/// and a band source that loses on precedence or is `None`. A caller that
/// must validate every dropped value probes exactly these under the old
/// type, without restating the translation rule.
///
/// Carried values are never checked or coerced here: a value malformed under
/// the new type reaches [`validate_scale_values`] and the scale gate as the
/// user wrote it.
///
/// Every carry preserves bounds: the key a carry writes accepts every value
/// the key it reads accepts under the old type. Ordinal `padding`, band
/// `paddingInner` and band `padding` are all in `[0, 1]`; point `padding` and
/// band `paddingOuter` are both finite and `>= 0`, and band `padding`'s
/// `[0, 1]` lies inside that; `align` is in `[0, 1]` on band and point alike.
/// So a carried value that passes the old type's bound check also passes the
/// new key's, which is why checking a switched scale's padding values under
/// the old type is enough to catch every out-of-bounds one, carried or
/// dropped. A carry added to [`PaddingMeaning::switch_carries`] must keep
/// this, or the new key's own check must run on what it carries. Returns a new dict and never mutates `base`; the same type
/// returns a copy of `base` and drops nothing. Raises the same `ValueError`
/// as [`scale_accepted_keys`] for an unknown tag on either side.
#[pyfunction]
pub fn translate_scale_switch<'py>(
    py: Python<'py>,
    old_type: &str,
    new_type: &str,
    base: &Bound<'py, PyDict>,
) -> PyResult<(Bound<'py, PyDict>, Vec<String>)> {
    let (accepted_old, old_meaning) = switch_side(old_type)?;
    let (accepted_new, new_meaning) = switch_side(new_type)?;
    if old_type == new_type {
        return Ok((base.copy()?, Vec::new()));
    }
    let carries = old_meaning.switch_carries(new_meaning);
    let out = PyDict::new(py);
    let mut removed = Vec::new();
    for (key, value) in base.iter() {
        let Ok(name) = key.extract::<String>() else {
            out.set_item(key, value)?;
            continue;
        };
        if !accepted_old.contains(&name.as_str()) {
            if carries.iter().any(|carry| carry.to == name) {
                return Err(PyValueError::new_err(format!(
                    "a '{old_type}' scale does not accept '{name}', which its switch to '{new_type}' writes"
                )));
            }
            out.set_item(key, value)?;
        } else if old_meaning.keys().contains(&name.as_str()) || !accepted_new.contains(&name.as_str()) {
            removed.push(name);
        } else {
            out.set_item(key, value)?;
        }
    }
    let mut carried_from = Vec::new();
    for carry in carries {
        for source in carry.from {
            if let Some(value) = base.get_item(source)?.filter(|v| !v.is_none()) {
                out.set_item(carry.to, value)?;
                carried_from.push(source);
                break;
            }
        }
    }
    let dropped = removed.into_iter().filter(|name| !carried_from.contains(&name.as_str())).collect();
    Ok((out, dropped))
}

/// Python-visible mirror of [`ScaleSpec::TAGS`] (`#159`): the complete,
/// wildcard-free set of 16 wire `"type"` tags, in `TAGS` declaration order —
/// the single published home for the scale-tag vocabulary. Exists so
/// `_override_apply.py`'s `_SCALE_TYPE_TAGS` reads this binding instead of
/// holding its own literal tuple a new `ScaleSpec` variant would not touch,
/// on the `scale_accepted_keys`/`sort_accepted_vocabulary` precedent.
#[pyfunction]
pub fn scale_type_tags() -> Vec<&'static str> {
    ScaleSpec::TAGS.to_vec()
}

/// Python-visible mirror of [`ScaleSpec::TEMPORAL_TAGS`]: the subset of
/// `scale_type_tags()` (`time`, `utc`) whose `domain` accepts date/datetime
/// elements. Replaces `_scale.py`'s hand-maintained `{"time", "utc"}`
/// literal.
#[pyfunction]
pub fn scale_temporal_type_tags() -> Vec<&'static str> {
    ScaleSpec::TEMPORAL_TAGS.to_vec()
}

/// The `sort`-field-object vocabulary: keys `"op"` and `"order"`, each the
/// accepted string spellings for that key, sourced from
/// [`SortOp::ACCEPTED`]/[`SortOrder::ACCEPTED`] -- the same values
/// `parse_sort_field_object`/`EncodingSpec::new` accept. Published on the
/// `scale_accepted_keys` precedent so Python's sort-accepting channels
/// (`X`/`Y`/`Facet`/... and every other honored-`sort` door) validate
/// against these instead of holding a second copy of the vocabulary.
#[pyfunction]
pub fn sort_accepted_vocabulary() -> std::collections::BTreeMap<&'static str, Vec<String>> {
    std::collections::BTreeMap::from([
        ("op", SortOp::ACCEPTED.iter().map(|s| s.to_string()).collect()),
        ("order", SortOrder::ACCEPTED.iter().map(|s| s.to_string()).collect()),
    ])
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
    let type_str = d.get_item("type")?.and_then(|t| t.extract::<String>().ok());
    let is_temporal =
        type_str.as_deref().is_some_and(|t| ScaleSpec::TEMPORAL_TAGS.contains(&t));
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

/// Shared first half of the chokepoint both [`deserialize_scale_value`] and
/// [`deserialize_scale_value_bare`] build on: raw-dict temporal-domain
/// conversion (`convert_raw_dict_temporal_domain`), then `json.dumps` to a
/// JSON string. `None` only when `scale` itself is `None`; every other error
/// (an unconvertible temporal element, or an unencodable object reaching
/// `json.dumps`) propagates as the raw `PyErr` it already is — neither this
/// function nor either of its two callers below adds a prefix to it, so an
/// `object()` domain element still raises the same unprefixed `TypeError`
/// `EncodingSpec::new`'s `scale=` parameter always has.
fn scale_value_to_json(py: Python, scale: Option<&Bound<'_, PyAny>>) -> PyResult<Option<String>> {
    let scale_converted = convert_raw_dict_temporal_domain(scale)?;
    let Some(obj) = scale_converted else { return Ok(None) };
    let json_module = py.import("json")?;
    Ok(Some(json_module.call_method1("dumps", (obj,))?.extract()?))
}

/// Convert + deserialize a scale value into a `ScaleSpec`, exactly as
/// `EncodingSpec::new`'s `scale=` parameter does: [`scale_value_to_json`]
/// first, then a JSON round-trip through `ScaleSpec`'s own gated
/// `Deserialize` impl (`validate_scale_spec_keys`, run at that impl's own
/// boundary), with the serde error prefixed by `prefix` — `"scale: "` for
/// [`deserialize_scale_value`], empty for [`deserialize_scale_value_bare`],
/// which [`validate_scale_values`] uses to build its own override-door voice
/// around the bare clause instead.
///
/// **The single chokepoint both `EncodingSpec::new` and [`validate_scale_dict`]
/// call, through [`deserialize_scale_value`]** — batch-C close remediation,
/// converging the rust- and python-design reviews' independent findings on
/// the same fix: neither caller re-implements the dict→JSON→`ScaleSpec`
/// sequence, so there is no second validation path that could drift from
/// this one. Error messages are therefore byte-identical between the two
/// callers by construction, not by coincidence — pinned by
/// `validate_scale_dict_message_matches_encoding_spec_route` below.
fn deserialize_scale_value_with_prefix(
    py: Python,
    scale: Option<&Bound<'_, PyAny>>,
    prefix: &str,
) -> PyResult<Option<ScaleSpec>> {
    let Some(s) = scale_value_to_json(py, scale)? else { return Ok(None) };
    Ok(Some(
        serde_json::from_str(&s).map_err(|e| PyValueError::new_err(format!("{prefix}{e}")))?,
    ))
}

pub(crate) fn deserialize_scale_value(
    py: Python,
    scale: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<ScaleSpec>> {
    deserialize_scale_value_with_prefix(py, scale, "scale: ")
}

fn deserialize_scale_value_bare(
    py: Python,
    scale: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<ScaleSpec>> {
    deserialize_scale_value_with_prefix(py, scale, "")
}

/// Validate one override-door scale value (#157): the same conversion +
/// deserialize chokepoint [`deserialize_scale_value`] runs, via its
/// un-prefixed twin [`deserialize_scale_value_bare`], followed by the
/// data-free bounds the scale classes enforce at construction, through the
/// same predicates and so the same clause text:
///
/// - `band`: padding through [`BandPadding::layout`](crate::scale::discrete::BandPadding::layout),
///   then `align` through [`check_align`];
/// - `point`: a set `padding` through [`check_point_padding`], then `align`
///   through [`check_align`];
/// - `ordinal`: `padding` through [`check_ordinal_padding`].
///
/// Every other type has no value bound checked here.
///
/// Raises a `ValueError` whose message is the refusal's **clause alone** —
/// naming no call, channel or key, since this function has none of those to
/// give: its caller wraps the clause in the override's own voice,
/// `Chart.override(<path>=...) <clause>`. Three clause shapes:
///
/// - an out-of-bounds value gives the class door's clause after its argument
///   name: [`reason`](crate::scale::discrete::BandPaddingError::reason) for
///   band padding, e.g. `must be a number in [0, 1]; received 1.5`, and the
///   [`check_align`]/[`check_point_padding`]/[`check_ordinal_padding`]
///   clause otherwise (ordinal: `must be in [0, 1]; got 1.5`). The clause
///   does not name the key, so a caller naming the key validates one key at
///   a time;
/// - a serde error (an unknown key, or a value of the wrong wire type) gives
///   serde's own text, with no `"scale: "` prefix, e.g.
///   `invalid type: string "a", expected f64`;
/// - an object `json.dumps` cannot encode gives a `ValueError` — never the
///   raw `TypeError` `json.dumps` itself raises — worded
///   `cannot be encoded as a scale value: <python error text>`.
#[pyfunction]
pub fn validate_scale_values(py: Python, scale: &Bound<'_, PyDict>) -> PyResult<()> {
    let spec = deserialize_scale_value_bare(py, Some(scale.as_any())).map_err(|e| {
        if e.is_instance_of::<PyTypeError>(py) {
            PyValueError::new_err(format!("cannot be encoded as a scale value: {}", e.value(py)))
        } else {
            e
        }
    })?;
    match spec {
        Some(ScaleSpec::Band { padding, align, .. }) => {
            padding.layout(align).map_err(|err| PyValueError::new_err(err.reason()))?;
            check_align(align).map_err(PyValueError::new_err)?;
        }
        Some(ScaleSpec::Point { padding, align, .. }) => {
            if let Some(padding) = padding {
                check_point_padding(padding).map_err(PyValueError::new_err)?;
            }
            check_align(align).map_err(PyValueError::new_err)?;
        }
        Some(ScaleSpec::Ordinal { padding, .. }) => {
            check_ordinal_padding(padding).map_err(PyValueError::new_err)?;
        }
        _ => {}
    }
    Ok(())
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
    /// tag comes from `ScaleSpec::tag()`, the ONE place (production code,
    /// not a test-local duplicate) a variant maps to its wire tag under
    /// compiler-enforced exhaustiveness, rather than a second, hand-typed
    /// label here that could silently mislabel an entry.
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
                padding: crate::scale::discrete::BandPadding::new(Some(0.1), Some(0.1), Some(0.2))
                    .expect("in-bounds band padding"),
                align: 0.5,
                range: Some(vec![0.0, 1.0]),
            },
            ScaleSpec::Point {
                domain: Some(vec!["a".to_string()]),
                padding: Some(0.5),
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
                bin: Some(crate::spec::encoding::ScaleBin {
                    bin_count: Some(4),
                    bin_width: Some(1.0),
                    extent: Some((0.0, 1.0)),
                    nice: false,
                }),
            },
        ]
    }

    /// `ScaleSpec::TAGS` = the image of `ScaleSpec::tag()` over
    /// `maximal_instances()` (`#159`): `ScaleSpec::tag()` is a
    /// compiler-enforced EXHAUSTIVE match (no wildcard arm — a future 17th
    /// variant fails to compile there before this test would even run), so
    /// this test proves the two OTHER hand-maintained enumerations —
    /// `ScaleSpec::TAGS` and `maximal_instances()`'s hand-built `Vec` — both
    /// still agree with it: in length (no duplicate or missing tag) and in
    /// identity (every tag `tag()` produces is one `TAGS` lists, and vice
    /// versa). 16 variants today.
    #[test]
    fn tags_equals_the_image_of_tag_over_maximal_instances() {
        use std::collections::HashSet;
        let produced: HashSet<&str> = maximal_instances().iter().map(ScaleSpec::tag).collect();
        let published: HashSet<&str> = ScaleSpec::TAGS.iter().copied().collect();
        assert_eq!(
            produced.len(),
            maximal_instances().len(),
            "maximal_instances() has a duplicate tag"
        );
        assert_eq!(published.len(), ScaleSpec::TAGS.len(), "ScaleSpec::TAGS has a duplicate tag");
        assert_eq!(
            produced, published,
            "ScaleSpec::TAGS has drifted from the tags maximal_instances() actually produces"
        );
    }

    /// Every published tag has a non-`None` accepted-key set (`#159`): a tag
    /// `ScaleSpec::TAGS` lists but `accepted_keys_for_scale_type` does not
    /// recognize would silently fall through the wire-key gate
    /// (`validate_scale_spec_keys`'s "unrecognized type, leave unvalidated"
    /// branch) for every scale of that type.
    #[test]
    fn every_published_tag_has_an_accepted_key_set() {
        for tag in ScaleSpec::TAGS {
            assert!(
                accepted_keys_for_scale_type(tag).is_some(),
                "ScaleSpec::TAGS lists '{tag}' but accepted_keys_for_scale_type does not cover it"
            );
        }
    }

    // Kept as two constants so no line of this file pairs them itself.
    const SERDE_ATTR: &str = "serde(";
    const ALIAS_KEY: &str = "alias";

    /// 1-based numbers of the non-comment lines in `text` carrying a serde
    /// attribute with an `alias` key.
    fn serde_alias_lines(text: &str) -> Vec<usize> {
        text.lines()
            .enumerate()
            .filter(|(_, line)| !line.trim_start().starts_with("//"))
            .filter(|(_, line)| {
                line.find(SERDE_ATTR).is_some_and(|at| line[at..].contains(ALIAS_KEY))
            })
            .map(|(i, _)| i + 1)
            .collect()
    }

    #[test]
    fn serde_alias_rule_flags_a_synthetic_attribute() {
        let synthetic = format!(
            "    /// #[{SERDE_ATTR}{ALIAS_KEY})] in a doc\n    #[{SERDE_ATTR}{ALIAS_KEY} = \"lo\")]\n    pub low: f64,\n"
        );
        assert_eq!(serde_alias_lines(&synthetic), vec![2]);
    }

    /// The gate's serde-alias blind spot (this file's module doc) stays
    /// theoretical: no source file of this module, where `ScaleSpec` and the
    /// `ContinuousScaleCommon` it flattens are declared, and not
    /// `scale/discrete.rs`, where the `BandPadding` that `ScaleSpec::Band`
    /// flattens is declared, carries a `#[serde(alias = ...)]` attribute.
    #[test]
    fn no_scale_spec_field_carries_a_serde_alias() {
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let dir = src.join("spec/encoding");
        let mut scanned = Vec::new();
        let mut hits = Vec::new();
        let module_files =
            std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display()));
        let flattened_elsewhere = std::iter::once(src.join("scale/discrete.rs"));
        for path in module_files.map(|entry| entry.unwrap().path()).chain(flattened_elsewhere) {
            if path.extension().is_some_and(|ext| ext == "rs") {
                let text = std::fs::read_to_string(&path).unwrap();
                hits.extend(
                    serde_alias_lines(&text)
                        .into_iter()
                        .map(|n| format!("{}:{n}", path.display())),
                );
                scanned.push(path.file_name().unwrap().to_string_lossy().into_owned());
            }
        }
        assert!(
            scanned.iter().any(|f| f == "mod.rs"),
            "the scan must reach ScaleSpec's file: {scanned:?}"
        );
        assert!(
            scanned.iter().any(|f| f == "discrete.rs"),
            "the scan must reach BandPadding's file: {scanned:?}"
        );
        assert!(
            hits.is_empty(),
            "serde alias attributes the scale-key gate cannot see: {hits:?}"
        );
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
            let tag = instance.tag();
            let value = serde_json::to_value(&instance).unwrap();
            let obj = value.as_object().unwrap();
            assert_eq!(
                obj.get("type").and_then(|v| v.as_str()),
                Some(tag),
                "ScaleSpec::tag() disagrees with the real serialized \"type\" tag"
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
    /// tags, exhaustively (reuses `maximal_instances()` + `ScaleSpec::tag()`
    /// rather than a second hand-typed tag list, so this test tracks the
    /// same compiler-enforced coverage as
    /// `tags_equals_the_image_of_tag_over_maximal_instances` above).
    #[test]
    fn scale_accepted_keys_matches_accepted_keys_for_scale_type_for_every_tag() {
        for instance in maximal_instances() {
            let tag = instance.tag();
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

    /// `scale_type_tags()`/`scale_temporal_type_tags()` return exactly the
    /// published Rust constants they mirror, in the same order — the
    /// binding contract `_override_apply.py`/`_scale.py` read against.
    #[test]
    fn scale_type_tags_and_temporal_tags_mirror_the_published_constants() {
        assert_eq!(scale_type_tags(), ScaleSpec::TAGS.to_vec());
        assert_eq!(scale_temporal_type_tags(), ScaleSpec::TEMPORAL_TAGS.to_vec());
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
                None, None, None, None,
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
                None, None, None, None,
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

    // ── `validate_scale_values` (#157) ──────────────────────────────────────

    /// A malformed `padding` masked by in-bounds explicit `paddingInner`/
    /// `paddingOuter` is still refused — `BandPadding::validate` checks every
    /// set side regardless of masking — with the clause alone, exactly
    /// `BandPaddingError::reason`'s text and nothing else: no `"scale: "`
    /// prefix, no mention of the channel, the call or `band`.
    #[test]
    fn validate_scale_values_refuses_band_padding_bounds_naming_the_clause_alone() {
        attach(|py| {
            let d = PyDict::new(py);
            d.set_item("type", "band").unwrap();
            d.set_item("padding", 1.5).unwrap();
            d.set_item("paddingInner", 0.2).unwrap();
            d.set_item("paddingOuter", 0.2).unwrap();
            let err = validate_scale_values(py, &d).unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py));
            let msg = err.value(py).to_string();
            assert_eq!(msg, "must be a number in [0, 1]; received 1.5");
        });
    }

    /// A serde type-mismatch clause carries no `"scale: "` prefix — the
    /// un-prefixed twin of `deserialize_scale_value`, not the prefixed one
    /// `validate_scale_dict` uses.
    #[test]
    fn validate_scale_values_serde_clause_has_no_scale_prefix() {
        attach(|py| {
            let d = PyDict::new(py);
            d.set_item("type", "linear").unwrap();
            d.set_item("padding", "a").unwrap();
            let err = validate_scale_values(py, &d).unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py));
            let msg = err.value(py).to_string();
            assert_eq!(msg, "invalid type: string \"a\", expected f64");
        });
    }

    /// An unknown key (here, `paddingInner` on `point`, which does not have
    /// one) refuses with the gate's own "unknown key ... accepted: ..."
    /// clause, un-prefixed, exactly like `scale_gate_typo_key_refused_names_real_key_among_accepted`
    /// but through `validate_scale_values` and without `"scale: "`.
    #[test]
    fn validate_scale_values_refuses_unknown_key_naming_accepted_set() {
        attach(|py| {
            let d = PyDict::new(py);
            d.set_item("type", "point").unwrap();
            d.set_item("paddingInner", 1.5).unwrap();
            let err = validate_scale_values(py, &d).unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py));
            let msg = err.value(py).to_string();
            assert_eq!(
                msg,
                "unknown key 'paddingInner' for type 'point'; accepted: align, domain, padding, range, reverse"
            );
        });
    }

    /// An object `json.dumps` cannot encode raises a `ValueError` naming the
    /// Python encoder's own text — never the raw `TypeError` `json.dumps`
    /// itself produces (REFUSE MALFORMED: no raw `TypeError` escapes a
    /// public call).
    #[test]
    fn validate_scale_values_dumps_failure_raises_value_error_not_type_error() {
        attach(|py| {
            let d = PyDict::new(py);
            d.set_item("type", "linear").unwrap();
            let unencodable = py.import("builtins").unwrap().getattr("object").unwrap().call0().unwrap();
            d.set_item("domain", unencodable).unwrap();
            let err = validate_scale_values(py, &d).unwrap_err();
            assert!(
                err.is_instance_of::<PyValueError>(py),
                "expected ValueError, got {err:?}"
            );
            assert!(!err.is_instance_of::<PyTypeError>(py));
            let msg = err.value(py).to_string();
            assert_eq!(
                msg,
                "cannot be encoded as a scale value: Object of type object is not JSON serializable"
            );
        });
    }

    /// A valid band fragment (in-bounds `padding`/`align`) passes with no
    /// error.
    #[test]
    fn validate_scale_values_accepts_a_valid_band_fragment() {
        attach(|py| {
            let d = PyDict::new(py);
            d.set_item("type", "band").unwrap();
            d.set_item("padding", 0.2).unwrap();
            d.set_item("align", 0.5).unwrap();
            validate_scale_values(py, &d)
                .unwrap_or_else(|e| panic!("expected a valid band fragment to pass, got {e}"));
        });
    }

    // ── translate_scale_switch: the type-switch owner ───────────────────────

    /// A value in a test scale dict.
    #[derive(Clone, Copy)]
    enum V {
        Num(f64),
        Str(&'static str),
        Bool(bool),
        None,
    }

    fn scale_dict<'py>(py: Python<'py>, scale_type: &str, items: &[(&str, V)]) -> Bound<'py, PyDict> {
        let d = PyDict::new(py);
        d.set_item("type", scale_type).unwrap();
        for &(key, value) in items {
            match value {
                V::Num(n) => d.set_item(key, n),
                V::Str(text) => d.set_item(key, text),
                V::Bool(b) => d.set_item(key, b),
                V::None => d.set_item(key, py.None()),
            }
            .unwrap();
        }
        d
    }

    /// The dict a switch returns, as sorted `(key, repr(value))` pairs.
    fn switched(old: &str, new: &str, items: &[(&str, V)]) -> Vec<(String, String)> {
        attach(|py| {
            let (out, _) = translate_scale_switch(py, old, new, &scale_dict(py, old, items)).unwrap();
            let mut pairs: Vec<(String, String)> = out
                .iter()
                .map(|(k, v)| (k.extract().unwrap(), v.repr().unwrap().to_string()))
                .collect();
            pairs.sort();
            pairs
        })
    }

    fn pairs(scale_type: &str, items: &[(&str, &str)]) -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = items.iter().map(|&(k, v)| (k.to_string(), v.to_string())).collect();
        out.push(("type".to_string(), format!("'{scale_type}'")));
        out.sort();
        out
    }

    const PADDING_FAMILY: [&str; 4] = ["padding", "paddingInner", "paddingOuter", "align"];

    /// Each padding meaning's keys are exactly the padding-family keys its
    /// tag accepts, for every tag: the meaning table and the accepted-key
    /// table cannot disagree about which keys a switch translates.
    #[test]
    fn padding_meaning_keys_match_each_types_accepted_keys() {
        for tag in ScaleSpec::TAGS {
            let meaning = padding_meaning_for_scale_type(tag).unwrap_or_else(|| panic!("no meaning for '{tag}'"));
            let accepted = accepted_keys_for_scale_type(tag).unwrap();
            let mut expected: Vec<&str> = PADDING_FAMILY.into_iter().filter(|k| accepted.contains(k)).collect();
            let mut actual = meaning.keys().to_vec();
            expected.sort_unstable();
            actual.sort_unstable();
            assert_eq!(actual, expected, "tag '{tag}'");
        }
    }

    /// ordinal → band: the ordinal inner gap becomes the band inner gap, and
    /// the band's outer padding stays unset.
    #[test]
    fn switch_ordinal_to_band_carries_padding_to_padding_inner() {
        assert_eq!(
            switched("ordinal", "band", &[("padding", V::Num(0.3)), ("domain", V::Str("d"))]),
            pairs("ordinal", &[("paddingInner", "0.3"), ("domain", "'d'")])
        );
        assert_eq!(switched("ordinal", "band", &[("padding", V::Num(0.0))]), pairs("ordinal", &[("paddingInner", "0.0")]));
    }

    /// ordinal → point: ordinal has no end padding, so nothing is carried and
    /// the point default applies.
    #[test]
    fn switch_ordinal_to_point_drops_padding() {
        assert_eq!(switched("ordinal", "point", &[("padding", V::Num(0.3))]), pairs("ordinal", &[]));
    }

    /// point → band: the end padding becomes band outer padding, `align`
    /// carries, and `reverse` (which band does not accept) is dropped as any
    /// non-family key is.
    #[test]
    fn switch_point_to_band_carries_padding_to_padding_outer_and_align() {
        assert_eq!(
            switched("point", "band", &[("padding", V::Num(2.0)), ("align", V::Num(0.3)), ("reverse", V::Bool(true))]),
            pairs("point", &[("paddingOuter", "2.0"), ("align", "0.3")])
        );
    }

    /// point → ordinal: neither the end padding nor `align` means anything
    /// on an ordinal scale.
    #[test]
    fn switch_point_to_ordinal_drops_padding_and_align() {
        assert_eq!(
            switched("point", "ordinal", &[("padding", V::Num(2.0)), ("align", V::Num(0.3))]),
            pairs("point", &[])
        );
    }

    /// band → ordinal: the band's inner side (`paddingInner`, else `padding`)
    /// becomes the ordinal gap; the outer side and `align` are dropped.
    #[test]
    fn switch_band_to_ordinal_carries_the_inner_side() {
        let all = [
            ("padding", V::Num(0.4)),
            ("paddingInner", V::Num(0.2)),
            ("paddingOuter", V::Num(0.3)),
            ("align", V::Num(0.1)),
        ];
        assert_eq!(switched("band", "ordinal", &all), pairs("band", &[("padding", "0.2")]));
        assert_eq!(switched("band", "ordinal", &[("padding", V::Num(0.4))]), pairs("band", &[("padding", "0.4")]));
        assert_eq!(switched("band", "ordinal", &[("paddingOuter", V::Num(0.3))]), pairs("band", &[]));
    }

    /// band → point: the band's outer side (`paddingOuter`, else `padding`)
    /// becomes the point end padding, and `align` carries.
    #[test]
    fn switch_band_to_point_carries_the_outer_side_and_align() {
        let all = [
            ("padding", V::Num(0.4)),
            ("paddingInner", V::Num(0.2)),
            ("paddingOuter", V::Num(0.3)),
            ("align", V::Num(0.1)),
        ];
        assert_eq!(switched("band", "point", &all), pairs("band", &[("padding", "0.3"), ("align", "0.1")]));
        assert_eq!(
            switched("band", "point", &[("padding", V::Num(0.4)), ("paddingInner", V::Num(0.2))]),
            pairs("band", &[("padding", "0.4")])
        );
    }

    /// continuous ↔ discrete: a span fraction and a step fraction share a
    /// key name but no meaning, so the padding family is dropped both ways;
    /// every other key keeps the accepted-key rule (`domain` is accepted by
    /// both sides and kept, `nice` and `reverse` are dropped where the new
    /// type has no such key).
    #[test]
    fn switch_between_continuous_and_discrete_drops_padding() {
        let continuous = ["linear", "log", "time", "symlog", "pow", "sqrt", "utc"];
        let discrete = ["ordinal", "band", "point"];
        for c in continuous {
            for d in discrete {
                let out = switched(c, d, &[("padding", V::Num(0.05)), ("domain", V::Str("x"))]);
                assert_eq!(out, pairs(c, &[("domain", "'x'")]), "{c} -> {d}");
                let family: Vec<(&str, V)> = PADDING_FAMILY.iter().map(|&k| (k, V::Num(0.2))).collect();
                let accepted = accepted_keys_for_scale_type(d).unwrap();
                let items: Vec<(&str, V)> = family.into_iter().filter(|(k, _)| accepted.contains(k)).collect();
                assert_eq!(switched(d, c, &items), pairs(d, &[]), "{d} -> {c}");
            }
        }
        assert_eq!(switched("linear", "band", &[("nice", V::Bool(true))]), pairs("linear", &[]));
        assert_eq!(switched("point", "linear", &[("reverse", V::Bool(true))]), pairs("point", &[("reverse", "True")]));
    }

    /// continuous → continuous: the span fraction means the same thing, so
    /// `padding` carries unchanged; a key the new type lacks (`zero` on log)
    /// is still dropped.
    #[test]
    fn switch_within_continuous_keeps_padding() {
        assert_eq!(
            switched("linear", "log", &[("padding", V::Num(0.05)), ("nice", V::Bool(true)), ("zero", V::Bool(true))]),
            pairs("linear", &[("padding", "0.05"), ("nice", "True")])
        );
    }

    /// The same type returns the dict unchanged, `None` values included.
    #[test]
    fn switch_to_the_same_type_is_unchanged() {
        let items = [("padding", V::Num(0.4)), ("paddingInner", V::None), ("align", V::Num(0.1))];
        assert_eq!(
            switched("band", "band", &items),
            pairs("band", &[("padding", "0.4"), ("paddingInner", "None"), ("align", "0.1")])
        );
    }

    /// An absent or `None` key is unset: it writes no key, and a `None`
    /// higher in the precedence falls through to the next source.
    #[test]
    fn switch_writes_no_key_for_an_unset_value() {
        assert_eq!(switched("ordinal", "band", &[("padding", V::None)]), pairs("ordinal", &[]));
        assert_eq!(switched("point", "band", &[]), pairs("point", &[]));
        assert_eq!(
            switched("band", "ordinal", &[("paddingInner", V::None), ("padding", V::Num(0.4))]),
            pairs("band", &[("padding", "0.4")])
        );
        assert_eq!(switched("band", "point", &[("paddingOuter", V::None), ("align", V::None)]), pairs("band", &[]));
    }

    /// Every carry preserves bounds: across every switch between the
    /// discrete types, a padding-family value the old type accepts is, once
    /// carried, accepted by the new type. Each key is sampled at and beyond
    /// its bounds, so the population holds both carried values and values
    /// the old type refuses.
    #[test]
    fn every_carry_keeps_a_value_the_old_type_accepts_acceptable() {
        let discrete = ["ordinal", "band", "point"];
        let samples = [-0.5, 0.0, 0.5, 1.0, 1.5, 4.0];
        let (mut carried, mut refused_by_old) = (0, 0);
        attach(|py| {
            for old in discrete {
                for new in discrete.into_iter().filter(|&t| t != old) {
                    let keys = padding_meaning_for_scale_type(old).unwrap().keys();
                    for &key in keys {
                        for value in samples {
                            let base = scale_dict(py, old, &[(key, V::Num(value))]);
                            if validate_scale_values(py, &base).is_err() {
                                refused_by_old += 1;
                                continue;
                            }
                            let (out, dropped) = translate_scale_switch(py, old, new, &base).unwrap();
                            if dropped.iter().any(|k| k == key) {
                                continue;
                            }
                            carried += 1;
                            out.set_item("type", new).unwrap();
                            if let Err(err) = validate_scale_values(py, &out) {
                                panic!(
                                    "{old} {key}={value} is accepted, but carried to {new} it is refused: {}",
                                    err.value(py)
                                );
                            }
                        }
                    }
                }
            }
        });
        assert!(carried > 0 && refused_by_old > 0, "carried {carried}, refused by the old type {refused_by_old}");
    }

    /// A carried value is neither checked nor coerced: one out of bounds for
    /// the new type arrives as written, and `validate_scale_values` refuses it
    /// under the new type.
    #[test]
    fn switch_leaves_a_malformed_carried_value_for_validation_to_refuse() {
        attach(|py| {
            let base = scale_dict(py, "ordinal", &[("padding", V::Num(2.0))]);
            let (out, _) = translate_scale_switch(py, "ordinal", "band", &base).unwrap();
            assert_eq!(out.get_item("paddingInner").unwrap().unwrap().extract::<f64>().unwrap(), 2.0);
            out.set_item("type", "band").unwrap();
            let err = validate_scale_values(py, &out).expect_err("paddingInner 2 is outside [0, 1]");
            assert_eq!(err.value(py).to_string(), "must be a number in [0, 1]; received 2");
        });
        assert_eq!(
            switched("point", "band", &[("padding", V::Str("wide"))]),
            pairs("point", &[("paddingOuter", "'wide'")])
        );
    }

    /// A key the old type does not accept passes through untouched, for the
    /// scale gate to judge.
    #[test]
    fn switch_passes_a_key_the_old_type_does_not_accept_through() {
        assert_eq!(
            switched("ordinal", "band", &[("clammp", V::Bool(true))]),
            pairs("ordinal", &[("clammp", "True")])
        );
    }

    /// An unknown tag on either side refuses with the same `ValueError`
    /// `scale_accepted_keys` raises.
    #[test]
    fn switch_refuses_an_unknown_type() {
        attach(|py| {
            let base = scale_dict(py, "band", &[]);
            for (old, new) in [("bogus", "band"), ("band", "bogus"), ("bogus", "bogus")] {
                let err = translate_scale_switch(py, old, new, &base).unwrap_err();
                assert!(err.is_instance_of::<PyValueError>(py), "{err:?}");
                assert_eq!(err.value(py).to_string(), "unknown scale type 'bogus'");
            }
        });
    }

    /// The keys a switch reports dropped, in `base` order.
    fn dropped(old: &str, new: &str, items: &[(&str, V)]) -> Vec<String> {
        attach(|py| translate_scale_switch(py, old, new, &scale_dict(py, old, items)).unwrap().1)
    }

    /// `dropped` names every key whose value the switch does not write: the
    /// padding family no carry reads, a band source that loses on precedence
    /// or is `None`, and every key the new type does not accept. A carried
    /// source, a surviving key and `"type"` are never listed.
    #[test]
    fn switch_reports_every_key_whose_value_it_drops() {
        let wide = V::Str("wide");
        for new in ["band", "point", "ordinal"] {
            assert_eq!(dropped("linear", new, &[("padding", wide), ("domain", V::Str("x"))]), ["padding"], "linear -> {new}");
        }
        assert_eq!(dropped("ordinal", "point", &[("padding", wide)]), ["padding"]);
        assert_eq!(dropped("point", "ordinal", &[("padding", wide), ("align", wide), ("reverse", V::Bool(true))]), [
            "padding", "align", "reverse"
        ]);
        let band = [("padding", V::Num(0.4)), ("paddingInner", V::Num(0.2)), ("paddingOuter", V::Num(0.3)), ("align", wide)];
        assert_eq!(dropped("band", "ordinal", &band), ["padding", "paddingOuter", "align"]);
        assert_eq!(dropped("band", "point", &band), ["padding", "paddingInner"]);
        assert_eq!(dropped("band", "ordinal", &[("padding", V::Num(0.4))]), Vec::<String>::new());
        assert_eq!(dropped("band", "ordinal", &[("paddingInner", V::None), ("padding", V::Num(0.4))]), ["paddingInner"]);
        assert_eq!(dropped("ordinal", "band", &[("padding", V::Num(0.3)), ("domain", V::Str("d"))]), Vec::<String>::new());
        assert_eq!(dropped("log", "band", &[("base", V::Num(2.0)), ("nice", V::Bool(true)), ("domain", V::Str("d"))]), [
            "base", "nice"
        ]);
        assert_eq!(dropped("linear", "log", &[("padding", V::Num(0.05)), ("zero", V::Bool(true))]), ["zero"]);
        assert_eq!(dropped("band", "band", &band), Vec::<String>::new());
        assert_eq!(dropped("ordinal", "band", &[("clammp", V::Bool(true))]), Vec::<String>::new());
    }

    /// A key the old type does not accept passes through, unless the switch
    /// writes a key of that name: the old scale is malformed, and the switch
    /// refuses naming the key instead of overwriting it.
    #[test]
    fn switch_refuses_an_unaccepted_key_its_translation_would_overwrite() {
        attach(|py| {
            for (old, new, key) in [("ordinal", "band", "paddingInner"), ("point", "band", "paddingOuter")] {
                let base = scale_dict(py, old, &[("padding", V::Num(0.3)), (key, V::Num(0.9))]);
                let err = translate_scale_switch(py, old, new, &base).unwrap_err();
                assert!(err.is_instance_of::<PyValueError>(py), "{err:?}");
                assert_eq!(
                    err.value(py).to_string(),
                    format!("a '{old}' scale does not accept '{key}', which its switch to '{new}' writes")
                );
            }
            // The refusal is about the key's name, not whether a value was carried.
            let base = scale_dict(py, "ordinal", &[("paddingInner", V::Num(0.9))]);
            assert!(translate_scale_switch(py, "ordinal", "band", &base).is_err());
            // A non-colliding unaccepted key still passes through.
            let base = scale_dict(py, "ordinal", &[("paddingOuter", V::Num(0.9))]);
            let (out, dropped) = translate_scale_switch(py, "ordinal", "band", &base).unwrap();
            assert_eq!(out.get_item("paddingOuter").unwrap().unwrap().extract::<f64>().unwrap(), 0.9);
            assert!(dropped.is_empty());
        });
    }

    // ── `validate_scale_values`: point, ordinal and band align bounds ───────

    /// The clause `validate_scale_values` refuses `{"type": scale_type, key:
    /// value}` with, or `None` when it accepts it.
    fn values_clause(scale_type: &str, key: &str, value: f64) -> Option<String> {
        attach(|py| {
            let d = scale_dict(py, scale_type, &[(key, V::Num(value))]);
            validate_scale_values(py, &d).err().map(|e| e.value(py).to_string())
        })
    }

    /// The message the scale class's constructor refuses `argument=value`
    /// with, called through the Python type as a user would.
    fn class_door_message<T: pyo3::PyTypeInfo>(argument: &str, value: f64) -> String {
        attach(|py| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("domain", vec!["a", "b"]).unwrap();
            kwargs.set_item(argument, value).unwrap();
            let err = py.get_type::<T>().call((), Some(&kwargs)).expect_err("out-of-bounds argument");
            assert!(err.is_instance_of::<PyValueError>(py), "{err:?}");
            err.value(py).to_string()
        })
    }

    /// Every bound `validate_scale_values` checks refuses with exactly the
    /// clause the class door words around its own argument name, because both
    /// run the same predicate.
    #[test]
    fn validate_scale_values_clause_equals_the_class_doors() {
        use crate::scale::band::BandScale;
        use crate::scale::ordinal::OrdinalScale;
        use crate::scale::point::PointScale;
        let cases: [(&str, &str, &str, f64, String); 7] = [
            ("point", "padding", "padding", -1.0, class_door_message::<PointScale>("padding", -1.0)),
            ("point", "align", "align", 1.5, class_door_message::<PointScale>("align", 1.5)),
            ("point", "align", "align", -0.1, class_door_message::<PointScale>("align", -0.1)),
            ("band", "align", "align", 5.0, class_door_message::<BandScale>("align", 5.0)),
            ("band", "paddingInner", "padding_inner", 1.5, class_door_message::<BandScale>("padding_inner", 1.5)),
            ("ordinal", "padding", "padding", 1.5, class_door_message::<OrdinalScale>("padding", 1.5)),
            ("ordinal", "padding", "padding", -0.1, class_door_message::<OrdinalScale>("padding", -0.1)),
        ];
        for (scale_type, key, argument, value, class_message) in cases {
            let clause = values_clause(scale_type, key, value).unwrap_or_else(|| panic!("{scale_type} {key}={value}"));
            let class_prefix = match scale_type {
                "point" => format!("PointScale({argument}=...) "),
                "band" => format!("BandScale({argument}=...) "),
                _ => format!("{argument} "),
            };
            assert_eq!(class_message, format!("{class_prefix}{clause}"), "{scale_type} {key}={value}");
        }
    }

    /// The exact clauses, so a reworded predicate shows up here.
    #[test]
    fn validate_scale_values_refuses_point_ordinal_and_band_align_bounds() {
        assert_eq!(values_clause("point", "padding", -1.0).unwrap(), "must be a finite number >= 0; received -1");
        assert_eq!(values_clause("point", "align", 1.5).unwrap(), "must be a number in [0, 1]; received 1.5");
        assert_eq!(values_clause("band", "align", 5.0).unwrap(), "must be a number in [0, 1]; received 5");
        assert_eq!(values_clause("band", "align", -0.5).unwrap(), "must be a number in [0, 1]; received -0.5");
        assert_eq!(values_clause("ordinal", "padding", 1.5).unwrap(), "must be in [0, 1]; got 1.5");
        assert_eq!(values_clause("ordinal", "padding", -0.1).unwrap(), "must be in [0, 1]; got -0.1");
    }

    /// The bounds' valid edges pass: 0 and 1 for `align` and ordinal
    /// `padding`, 0 and a large value for point `padding` and band
    /// `paddingOuter`, and a point scale with no `padding` at all.
    #[test]
    fn validate_scale_values_accepts_the_valid_edges() {
        let valid = [
            ("point", "padding", 0.0),
            ("point", "padding", 1.0e6),
            ("point", "align", 0.0),
            ("point", "align", 1.0),
            ("band", "align", 0.0),
            ("band", "align", 1.0),
            ("band", "paddingOuter", 1.0e6),
            ("band", "paddingInner", 1.0),
            ("ordinal", "padding", 0.0),
            ("ordinal", "padding", 1.0),
        ];
        for (scale_type, key, value) in valid {
            assert_eq!(values_clause(scale_type, key, value), None, "{scale_type} {key}={value}");
        }
        attach(|py| validate_scale_values(py, &scale_dict(py, "point", &[])).unwrap());
    }

    /// `scale_padding_keys` publishes exactly each type's translated keys,
    /// and refuses an unknown tag like `scale_accepted_keys`.
    #[test]
    fn scale_padding_keys_publishes_each_types_padding_family() {
        assert_eq!(scale_padding_keys("band").unwrap(), ["padding", "paddingInner", "paddingOuter", "align"]);
        assert_eq!(scale_padding_keys("point").unwrap(), ["padding", "align"]);
        assert_eq!(scale_padding_keys("ordinal").unwrap(), ["padding"]);
        assert_eq!(scale_padding_keys("linear").unwrap(), ["padding"]);
        assert!(scale_padding_keys("sequential").unwrap().is_empty());
        attach(|py| {
            let err = scale_padding_keys("bogus").unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py));
            assert_eq!(err.value(py).to_string(), "unknown scale type 'bogus'");
        });
    }
}
