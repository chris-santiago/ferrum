//! `TimeScale` test coverage.
//!
//! Promoted from an inline `#[cfg(test)] mod tests { ... }` (per this repo's
//! Rust test-module convention, `CLAUDE.md`) when the F-L04-10/F-L04-06
//! (batch-C task 3) additions — optional-domain construction, the
//! PyO3-boundary temporal extraction, and the UTC round-trip/byte-identity
//! pins — would have pushed `time.rs` past readable length as a single file.
//! Carries the pre-existing scale-arithmetic/wire-shape suite plus the new
//! `temporal_extraction_*` tests, which construct real Python
//! `datetime.date`/`datetime.datetime` objects via `Python::attach` to
//! exercise `temporal_value_to_epoch_ms` through the actual PyO3 boundary
//! rather than a hand-built mirror of it.

use super::*;

/// Wrap a plain `Vec<f64>` as the `Some(Vec<TemporalDomainValue>)` shape
/// `TimeScale::new`'s domain parameter now takes (F-L04-10) — every
/// existing epoch-ms-domain test below constructs `TemporalDomainValue`
/// directly (its `.0` field is `pub(crate)`) rather than round-tripping
/// through Python, since the float arm of the PyO3-boundary extraction
/// (`temporal_value_to_epoch_ms`) is exercised separately, per input
/// kind, by the `temporal_extraction_*` tests further down.
fn td(values: Vec<f64>) -> Option<Vec<TemporalDomainValue>> {
    Some(values.into_iter().map(TemporalDomainValue).collect())
}

#[test]
fn test_time_scale_round_trip_ms() {
    // 2026-01-01 00:00:00 UTC = 1767225600000.0 ms
    // 2026-12-31 23:59:59 UTC ≈ 1798761599000.0 ms
    let t = TimeScale::new(
        td(vec![1_767_225_600_000.0, 1_798_761_599_000.0]),
        Some(vec![0.0, 1000.0]),
        false,
        false,
        None,
        false,
        false,
    ).unwrap();
    let mid = (1_767_225_600_000.0 + 1_798_761_599_000.0) / 2.0;
    let y = t.scale(mid);
    let back = t.invert(y);
    assert!((back - mid).abs() < 1e-3, "round-trip failed: got {back}");
}

/// #99/#104 residue: `TimeScale` reuses `LinearScaleData` verbatim (see
/// the struct doc above), so the shared degenerate-domain guard covers
/// it for free — pinned directly here so the coverage isn't only
/// inferable from `linear.rs`'s own test. A single-instant domain (all
/// rows share one timestamp) must scale to a finite pixel, not NaN.
#[test]
fn test_time_scale_degenerate_single_instant_domain_returns_range_midpoint() {
    let instant = 1_767_225_600_000.0; // 2026-01-01 00:00:00 UTC
    let t = TimeScale::new_internal(vec![instant, instant], vec![0.0, 1000.0], false, false);
    let px = t.scale_internal(instant);
    assert!(px.is_finite(), "degenerate time domain must scale to a finite pixel, got NaN");
    assert_eq!(px, 500.0, "degenerate time domain must map to the range midpoint");
}

#[test]
fn test_time_ticks_returns_some_ticks_for_year_span() {
    let t = TimeScale::new(
        td(vec![1_767_225_600_000.0, 1_798_761_599_000.0]),
        Some(vec![0.0, 1000.0]),
        false,
        false,
        None,
        false,
        false,
    ).unwrap();
    let ticks = t.ticks(10);
    assert!(!ticks.is_empty(), "expected non-empty ticks");
}

/// SPEC-01 regression: `utc` and `domain_user_set` are independent fields.
///
/// Under the old positional tuple `(Data, Option<f64>, bool, bool)` the
/// `utc` flag occupied slot `.3` — the same slot Linear/Log/... used for
/// `domain_user_set`. Setting `utc=true` therefore also flipped the slot
/// that the family treats as `domain_user_set`. With named fields the two
/// are distinct: `utc` can be true while the domain is still user-set, and
/// `domain()` must keep returning the list regardless of `utc`. This test
/// would not compile/pass against the conflated tuple representation.
#[test]
fn test_time_utc_independent_of_domain_user_set() {
    let domain = vec![1_767_225_600_000.0, 1_798_761_599_000.0];
    let utc_scale = TimeScale::new(
        td(domain.clone()), Some(vec![0.0, 1000.0]), false, false, None, true, false,
    ).unwrap();
    let local_scale = TimeScale::new(
        td(domain.clone()), Some(vec![0.0, 1000.0]), false, false, None, false, false,
    ).unwrap();

    assert!(utc_scale.utc(), "utc flag must be honoured");
    assert!(!local_scale.utc(), "non-utc flag must be honoured");
    // utc must not bleed into the domain getter: both report the domain.
    assert_eq!(utc_scale.domain(), Some(domain.clone()));
    assert_eq!(local_scale.domain(), Some(domain));
}

/// SPEC-06 regression: `domain()` returns `Option<Vec<f64>>` for parity
/// with the other continuous scales — `Some` when the caller passed an
/// explicit domain.
#[test]
fn test_time_domain_returns_option() {
    let t = TimeScale::new(
        td(vec![0.0, 1000.0]), Some(vec![0.0, 1.0]), false, false, None, false, false,
    ).unwrap();
    let domain: Option<Vec<f64>> = t.domain();
    assert_eq!(domain, Some(vec![0.0, 1000.0]));
}

/// F-L04-10: `TimeScale(domain=None)` now constructs — matching the five
/// continuous siblings (SPEC-06) — instead of requiring an explicit
/// domain. `domain()` reports `None` (data-derived), and the internal
/// sentinel matches `LinearScale`'s own `[0.0, 1.0]` placeholder that
/// render-time inference replaces (mirrors
/// `linear_named_fields_round_trip`'s no-domain half).
#[test]
fn time_domain_none_constructs_and_infers() {
    let t = TimeScale::new(None, None, false, false, None, false, false).unwrap();
    assert_eq!(t.domain(), None);
    assert_eq!(t.range(), None);
    assert_eq!(t.domain_pair(), [0.0, 1.0]);
}

/// `nice=True` with no explicit domain must not calendar-round the `[0,
/// 1]` sentinel — mirrors `LinearScale::new`'s `nice && domain_user_set`
/// guard. Before this guard, `TimeScale(nice=True)` (now legal, since
/// `domain` is optional) would silently apply calendar-nice rounding to
/// the epoch-ms interval `[0, 1]` (1970-01-01T00:00:00.000Z ..
/// .001Z) — a meaningless domain no caller asked to see rounded.
#[test]
fn time_domain_none_with_nice_leaves_sentinel_untouched() {
    let t = TimeScale::new(None, None, false, true, None, false, false).unwrap();
    assert_eq!(t.domain_pair(), [0.0, 1.0]);
}

// ── `reverse` kwarg (F-L04-07, batch-C task 2) ──────────────────────────

#[test]
fn time_reverse_round_trips_through_to_scale_spec() {
    let t = TimeScale::new(
        td(vec![1_767_225_600_000.0, 1_798_761_599_000.0]),
        None, false, false, None, false, true,
    ).unwrap();
    assert!(t.reverse());
    match t.to_scale_spec() {
        ScaleSpec::Time { common, .. } => {
            assert!(common.reverse, "reverse=True must survive to the wire spec");
            assert_eq!(common.domain, Some(vec![1_767_225_600_000.0, 1_798_761_599_000.0]));
        }
        other => panic!("expected ScaleSpec::Time, got {other:?}"),
    }
}

/// `utc=True` composed with `reverse=True` must select `ScaleSpec::Utc`
/// (not `Time`) while still carrying the reverse bit — the two flags are
/// independent fields (see `test_time_utc_independent_of_domain_user_set`
/// for the same independence guarantee against `domain_user_set`).
#[test]
fn time_utc_and_reverse_compose_independently() {
    let t = TimeScale::new(
        td(vec![1_767_225_600_000.0, 1_798_761_599_000.0]),
        None, false, false, None, true, true,
    ).unwrap();
    assert!(t.utc());
    assert!(t.reverse());
    match t.to_scale_spec() {
        ScaleSpec::Utc { common, .. } => assert!(common.reverse),
        other => panic!("expected ScaleSpec::Utc, got {other:?}"),
    }
}

/// Default (`reverse` unset) omits the wire key entirely, matching every
/// pre-existing baseline in `test_scale_spec_parity`.
#[test]
fn time_reverse_default_emits_no_reverse_key() {
    let t = TimeScale::new(
        td(vec![1_767_225_600_000.0, 1_798_761_599_000.0]),
        None, false, false, None, false, false,
    ).unwrap();
    assert!(!t.reverse());
    let json = serde_json::to_string(&t.to_scale_spec()).unwrap();
    assert!(!json.contains("reverse"), "default reverse must not appear on the wire: {json}");
}

/// `repr_string()` pinned in both directions (quality-review F2): the
/// default-shaped repr is byte-identical to before this change, and
/// `reverse=True` appends the exact `, reverse=True)` suffix. `TimeScale`
/// is the useful second pin (alongside `linear_repr_pins_both_reverse_branches`)
/// because `reverse` composes with the pre-existing `utc=True` prefix.
#[test]
fn time_repr_pins_both_reverse_branches() {
    let default_scale = TimeScale::new(
        td(vec![0.0, 1000.0]), None, false, false, None, false, false,
    ).unwrap();
    assert_eq!(
        default_scale.repr_string(),
        "TimeScale(domain=[0, 1000], range=[0, 1], clamp=False)",
    );

    let reversed_scale = TimeScale::new(
        td(vec![0.0, 1000.0]), None, false, false, None, false, true,
    ).unwrap();
    assert_eq!(
        reversed_scale.repr_string(),
        "TimeScale(domain=[0, 1000], range=[0, 1], clamp=False, reverse=True)",
    );

    let utc_reversed_scale = TimeScale::new(
        td(vec![0.0, 1000.0]), None, false, false, None, true, true,
    ).unwrap();
    assert_eq!(
        utc_reversed_scale.repr_string(),
        "TimeScale(utc=True, domain=[0, 1000], range=[0, 1], clamp=False, reverse=True)",
    );
}

/// F-L04-10: the `domain_user_set` gap named at `repr_string`'s edit —
/// before this fix, an unset domain would have printed the internal
/// `[0, 1]` sentinel as though it were user-supplied. Mirrors
/// `linear_repr_pins_both_reverse_branches`'s `domain=None` pin.
#[test]
fn time_repr_pins_domain_none() {
    let t = TimeScale::new(None, None, false, false, None, false, false).unwrap();
    assert_eq!(t.repr_string(), "TimeScale(domain=None, range=[0, 1], clamp=False)");
}

// ── Minor tick tests ─────────────────────────────────────────────────────

/// Regression: time scale major positions are unchanged after adding minor support.
///
/// Both `ticks_internal(10)` and `minor_ticks_internal()` use count=10 for
/// the major tick generation, so the comparison is apples-to-apples.
#[test]
fn test_time_major_positions_unchanged() {
    // 2026-01-01 to 2026-12-31 (month-level major ticks).
    let t = TimeScale::new_internal(
        vec![1_767_225_600_000.0, 1_798_761_599_000.0],
        vec![0.0, 1000.0],
        false,
        false,
    );
    let before = t.ticks_internal(10);
    let _ = t.minor_ticks_internal();
    let after = t.ticks_internal(10);
    assert_eq!(before, after, "major ticks changed after calling minor_ticks_internal");
}

/// Time minor ticks are present and lie between major ticks.
///
/// `minor_ticks_internal()` uses the fixed major count of 10 internally.
#[test]
fn test_time_minor_ticks_exist_between_majors() {
    // Use a span where major ticks fall at month boundaries.
    // 2026-01-01 = 1767225600000 ms, 2026-12-31 ≈ 1798761599000 ms.
    let t = TimeScale::new_internal(
        vec![1_767_225_600_000.0, 1_798_761_599_000.0],
        vec![0.0, 1000.0],
        false,
        false,
    );
    // minor_ticks_internal uses count=10 for majors internally.
    let majors = t.ticks_internal(10);
    let minors = t.minor_ticks_internal();

    // Minors must be non-empty when there are at least 2 major ticks.
    if majors.len() >= 2 {
        assert!(!minors.is_empty(), "expected minor ticks between month boundaries");
    }

    // Minors must lie within the domain and not coincide with majors.
    let lo = 1_767_225_600_000.0_f64;
    let hi = 1_798_761_599_000.0_f64;
    let major_set: std::collections::HashSet<u64> =
        majors.iter().map(|&v| v.to_bits()).collect();
    for m in &minors {
        assert!(!major_set.contains(&m.position.to_bits()),
            "time minor at {} coincides with major", m.position);
        assert!(m.position >= lo && m.position <= hi,
            "time minor {} outside domain [{lo},{hi}]", m.position);
        assert!(!m.is_major);
    }
}

/// Minor count per major interval = 4 (DEFAULT_MINOR_SUBDIVISIONS - 1).
/// Use a small fixed-interval domain to make the count deterministic.
///
/// `minor_ticks_internal()` uses the fixed major count of 10; the 10-second
/// span with count=10 produces major ticks at 1-second intervals, giving a
/// predictable 4 minors per interval.
#[test]
fn test_time_minor_count_per_interval() {
    // Span of exactly 10 seconds — major ticks at 1-second intervals.
    let lo = 0.0_f64;
    let hi = 10_000.0; // 10 seconds in ms
    let t = TimeScale::new_internal(vec![lo, hi], vec![0.0, 1000.0], false, false);
    // minor_ticks_internal uses count=10 for majors; ticks_internal(10) gives same list.
    let majors = t.ticks_internal(10);
    let minors = t.minor_ticks_internal();
    if majors.len() >= 2 {
        // (DEFAULT_MINOR_SUBDIVISIONS - 1) = 4 interior minors per interval.
        let expected = (majors.len() - 1) * 4;
        assert_eq!(minors.len(), expected,
            "expected {expected} minors for {}-interval time scale, got {}: {minors:?}",
            majors.len() - 1,
            minors.len(),
        );
    }
}

// ── PyO3-boundary temporal extraction (F-L04-10) ────────────────────────
//
// Each test constructs a REAL Python object (via `Python::attach` +
// `pyo3::types::{PyDate, PyDateTime, PyTzInfo, PyDelta}`) and feeds it
// through `temporal_value_to_epoch_ms`, the exact function
// `TemporalDomainValue::extract` calls — not a hand-built mirror of it, per
// the "prove RED against reality" rule. Expected values are hand-derived
// from `temporal_coord_to_epoch_ms`'s own documented formula
// (`ferrum/annotation/coords.py`); the Python-side parity test
// (`tests/test_timescale_domain.py`) additionally proves the two converters
// agree by calling both, so any future drift is caught from either side.

fn attach_and_extract<F>(build: F) -> PyResult<f64>
where
    F: for<'py> FnOnce(Python<'py>) -> PyResult<Bound<'py, PyAny>>,
{
    pyo3::Python::initialize();
    Python::attach(|py| {
        let obj = build(py)?;
        temporal_value_to_epoch_ms(&obj)
    })
}

#[test]
fn temporal_extraction_accepts_float_epoch_ms_unchanged() {
    let ms = attach_and_extract(|py| Ok(1_767_225_600_123.5f64.into_pyobject(py)?.into_any())).unwrap();
    assert_eq!(ms, 1_767_225_600_123.5);
}

/// A plain Python `int` is accepted as a numeric epoch-ms value, widened to
/// `f64` — sibling parity with `LinearScale`/`LogScale`/etc, all of which
/// accept `int` domain elements through pyo3's ordinary numeric conversion.
/// This is a DELIBERATE delta from `temporal_coord_to_epoch_ms` (which
/// accepts no numeric type at all); see `temporal_value_to_epoch_ms`'s doc.
#[test]
fn temporal_extraction_accepts_int_epoch_ms_as_numeric() {
    let ms = attach_and_extract(|py| Ok(1_767_225_600_123i64.into_pyobject(py)?.into_any())).unwrap();
    assert_eq!(ms, 1_767_225_600_123.0);
}

/// `datetime.date(2020, 6, 1)` → midnight UTC. Value taken from
/// `temporal_coord_to_epoch_ms`'s own doctest example.
#[test]
fn temporal_extraction_accepts_date_as_midnight_utc() {
    let ms = attach_and_extract(|py| {
        Ok(pyo3::types::PyDate::new(py, 2020, 6, 1)?.into_any())
    }).unwrap();
    assert_eq!(ms, 1_590_969_600_000.0);
}

/// A naive `datetime.datetime` means UTC: 2020-06-01T12:30:00 is
/// 1590969600000 (midnight) + 45000 seconds * 1000.
#[test]
fn temporal_extraction_accepts_naive_datetime_as_utc() {
    let ms = attach_and_extract(|py| {
        Ok(pyo3::types::PyDateTime::new(py, 2020, 6, 1, 12, 30, 0, 0, None)?.into_any())
    }).unwrap();
    assert_eq!(ms, 1_591_014_600_000.0);
}

/// An aware `datetime.datetime` converts to UTC: 2020-06-01T12:00:00+05:00
/// is 2020-06-01T07:00:00 UTC.
#[test]
fn temporal_extraction_converts_aware_datetime_to_utc() {
    let ms = attach_and_extract(|py| {
        let offset = pyo3::types::PyDelta::new(py, 0, 5 * 3600, 0, true)?;
        let tz = pyo3::types::PyTzInfo::fixed_offset(py, offset)?;
        Ok(pyo3::types::PyDateTime::new(py, 2020, 6, 1, 12, 0, 0, 0, Some(&tz))?.into_any())
    }).unwrap();
    assert_eq!(ms, 1_590_994_800_000.0);
}

/// `datetime.timezone.utc` is a fixed offset of zero — the aware branch must
/// reduce to the same instant a naive UTC-meaning datetime would give.
#[test]
fn temporal_extraction_converts_utc_aware_datetime() {
    let ms = attach_and_extract(|py| {
        let tz: Bound<'_, pyo3::types::PyTzInfo> = pyo3::types::PyTzInfo::utc(py)?.to_owned();
        Ok(pyo3::types::PyDateTime::new(py, 2020, 6, 1, 12, 30, 0, 0, Some(&tz))?.into_any())
    }).unwrap();
    assert_eq!(ms, 1_591_014_600_000.0);
}

/// ISO-8601 date-only string: same value as the `datetime.date` test above.
#[test]
fn temporal_extraction_accepts_iso_date_string() {
    let ms = attach_and_extract(|py| Ok("2020-06-01".into_pyobject(py)?.into_any())).unwrap();
    assert_eq!(ms, 1_590_969_600_000.0);
}

/// ISO-8601 naive datetime string: same value as the naive-datetime test.
#[test]
fn temporal_extraction_accepts_iso_naive_datetime_string() {
    let ms = attach_and_extract(|py| Ok("2020-06-01T12:30:00".into_pyobject(py)?.into_any())).unwrap();
    assert_eq!(ms, 1_591_014_600_000.0);
}

/// ISO-8601 datetime string with a `Z` (UTC) offset. `Z`-suffix support in
/// `datetime.fromisoformat` is Python-version-dependent (added in 3.11 —
/// verified directly against this project's pinned dev interpreter,
/// Python 3.10.14, which rejects it; see `iso8601_string_epoch_ms`'s doc).
/// Rather than hardcode one outcome (which would make this test's pass/fail
/// a function of which Python happens to be embedded in `cargo test`, not
/// of Rust's correctness), this asks the SAME running interpreter whether
/// IT accepts the string and asserts `temporal_value_to_epoch_ms` agrees —
/// the version-safe form of the "Z" pin, proven against reality either way.
#[test]
fn temporal_extraction_iso_datetime_string_with_z_offset_matches_runtime_fromisoformat() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let s = "2020-06-01T12:30:00Z";
        let datetime_cls = py.import("datetime").unwrap().getattr("datetime").unwrap();
        let python_accepts = datetime_cls.call_method1("fromisoformat", (s,)).is_ok();

        let obj = s.into_pyobject(py).unwrap().into_any();
        let rust_result = temporal_value_to_epoch_ms(&obj);
        assert_eq!(
            rust_result.is_ok(),
            python_accepts,
            "Rust's Z-suffix acceptance must match this interpreter's datetime.fromisoformat",
        );
        if python_accepts {
            assert_eq!(rust_result.unwrap(), 1_591_014_600_000.0);
        }
    });
}

/// ISO-8601 datetime string with a numeric `+HH:MM` offset: same instant as
/// the `+05:00` aware-datetime test.
#[test]
fn temporal_extraction_accepts_iso_datetime_string_with_numeric_offset() {
    let ms = attach_and_extract(|py| Ok("2020-06-01T12:00:00+05:00".into_pyobject(py)?.into_any())).unwrap();
    assert_eq!(ms, 1_590_994_800_000.0);
}

/// A non-temporal, non-numeric, non-string value refuses naming every
/// accepted form.
#[test]
fn temporal_extraction_refuses_non_temporal_value_naming_accepted_forms() {
    let err = attach_and_extract(|py| Ok(pyo3::types::PyList::empty(py).into_any())).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("float"), "{msg}");
    assert!(msg.contains("datetime.date"), "{msg}");
    assert!(msg.contains("datetime.datetime"), "{msg}");
    assert!(msg.contains("ISO-8601"), "{msg}");
}

/// A string that is not valid ISO-8601 refuses naming the accepted string
/// forms (distinct message from the type-refusal case above — this is a
/// `str`, just an unparseable one).
#[test]
fn temporal_extraction_refuses_unparseable_string() {
    let err = attach_and_extract(|py| Ok("not-a-date".into_pyobject(py)?.into_any())).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("not-a-date"), "{msg}");
    assert!(msg.contains("ISO-8601"), "{msg}");
}

/// `bool` is refused explicitly, not silently accepted as `1.0`/`0.0`.
/// Python's `bool` is an `int` subclass, so without the explicit check in
/// `temporal_value_to_epoch_ms` (adjudicated finding, cycle 3), `ob.extract::<f64>()`
/// would have accepted `True`/`False` as epoch-ms — the exact footgun
/// `temporal_coord_to_epoch_ms` also refuses (its accepted-type union has no
/// numeric type at all). RED-proven: reverting the `is_instance_of::<PyBool>()`
/// guard makes this test fail (`True` extracts to `1.0` instead of erroring).
#[test]
fn temporal_extraction_refuses_bool() {
    // `bool::into_pyobject` returns a `Borrowed` (Python interns True/False
    // as singletons) rather than an owned `Bound`; `.to_owned()` converts
    // before `.into_any()`, which takes ownership.
    let err = attach_and_extract(|py| Ok(true.into_pyobject(py)?.to_owned().into_any())).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("bool"), "{msg}");

    let err_false =
        attach_and_extract(|py| Ok(false.into_pyobject(py)?.to_owned().into_any())).unwrap_err();
    assert!(err_false.to_string().contains("bool"));
}
