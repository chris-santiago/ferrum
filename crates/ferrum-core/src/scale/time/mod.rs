use chrono::NaiveDate;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDate, PyDateTime, PyTzInfoAccess};

use super::core::{continuous_common, resolve_continuous, scale_spec_to_py_dict};
use super::linear::LinearScaleData;
use super::ticks::{calendar_ticks, minor_ticks_default, nice_calendar_interval, nice_time_interval_ms, CalendarInterval, Tick};
use crate::spec::encoding::ScaleSpec;

/// Continuous temporal scale backed by Unix epoch milliseconds.
///
/// Maps an epoch-millisecond domain to a numeric range. Tick generation
/// uses time-aware "nice" intervals (seconds, minutes, hours, days, months,
/// years) rather than purely numeric rounding. Domain values are
/// floating-point epoch milliseconds (UTC).
///
/// **UTC by contract (F-L04-06):** every temporal rendering path (tick
/// values, calendar `nice`, the default and explicit time formatters) always
/// operates in UTC — there is no local-time rendering, ever (barred by the
/// byte-determinism hard constraint). `utc=True` and `utc=False` therefore
/// render byte-identical SVG; the flag exists only so the wire form can
/// distinguish `fr.TimeScale` from `fr.UtcScale`-shaped input (the `Time` vs
/// `Utc` `ScaleSpec` tag) without changing what gets drawn. Naive `datetime`
/// domain values mean UTC for the same reason: there is no other timezone
/// the renderer could honor.
///
/// Parameters
/// ----------
/// domain : Sequence[float | datetime.date | datetime.datetime | str] | None, default None
///     Input domain as ``[t_min, t_max]``. Each element may be a ``float``
///     (epoch milliseconds, unchanged), a ``datetime.date`` (midnight UTC),
///     a ``datetime.datetime`` (naive means UTC; aware converts to UTC), or
///     an ISO-8601 date/datetime string. ``None`` (the default) infers the
///     domain from the encoded column's data, like every other continuous
///     scale. Conversion follows the same rule as
///     ``ferrum.annotation.coords.temporal_coord_to_epoch_ms``, the
///     annotation layer's canonical converter.
/// range : tuple[float, float]
///     Output range as ``[lo, hi]`` pixel coordinates.
/// clamp : bool, default False
///     Clamp out-of-domain inputs to the range endpoints.
/// nice : bool, default False
///     Extend domain endpoints to the nearest calendar interval boundary —
///     but only once there IS a domain to extend. With an explicit
///     ``domain=``, nicing applies immediately, here, to that domain (this
///     object's own ``domain``/``ticks()`` reflect it). With ``domain=None``
///     (data-derived), nicing is deferred rather than applied to the
///     placeholder sentinel: this object's own ``domain``/``ticks()`` stay
///     un-niced (``nice=True`` is a no-op on the standalone object), but
///     using it as an encoding's scale (``x=fr.X("date",
///     scale=fr.TimeScale(nice=True))``) DOES nice — the chart-render path
///     infers the domain from the column first, then applies the same
///     calendar rounding to that inferred domain before drawing.
/// reverse : bool, default False
///     Swap the resolved domain endpoints when this scale resolves inside a
///     chart render, producing a descending (most recent-first) axis —
///     equivalent, AT RENDER TIME, to writing ``domain=[hi, lo]`` for an
///     explicit domain (an auto-inferred domain keeps its usual padding
///     before the swap). The swap applies only at render resolution: this
///     object's own ``scale()``/``invert()``/``ticks()`` and its ``domain``
///     getter keep reporting the constructor's domain unchanged. This
///     diverges from ``PointScale``'s identically-named ``reverse``, which
///     DOES apply inside ``PointScale.scale()``.
///
/// Examples
/// --------
/// Ferrum converts datetime columns automatically; a ``TimeScale`` is
/// constructed implicitly when the channel data type is temporal::
///
///     import ferrum as fr
///     chart = fr.Chart(df).encode(x=fr.X("date:T"))
// Named-field PyO3 facade for the temporal scale.
//
// `utc` and `domain_user_set` are **separate** fields. They previously shared a
// single positional tuple slot across the continuous-scale family (Linear/Log/...
// used slot `.3` for `domain_user_set` while `TimeScale` silently repurposed it
// for `utc`); naming them makes that conflation impossible (SPEC-01).
#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, PartialEq)]
pub struct TimeScale {
    data: LinearScaleData,
    padding: Option<f64>,
    range_user_set: bool,
    utc: bool,
    domain_user_set: bool,
    reverse: bool,
}

impl TimeScale {
    /// Crate-internal constructor (no PyO3, no validation), for render-side use.
    /// `TimeScale` reuses [`LinearScaleData`]: the domain-to-range mapping is
    /// affine over epoch milliseconds, only tick/nice behavior is time-aware.
    /// `utc` defaults to `false`; chain [`with_utc`](Self::with_utc) to set it
    /// (mirrors [`OrdinalScale::with_explicit_range`](super::ordinal::OrdinalScale::with_explicit_range)'s
    /// pattern for an extra provenance flag on an otherwise-stable constructor).
    pub(crate) fn new_internal(domain: Vec<f64>, range: Vec<f64>, clamp: bool, nice: bool) -> Self {
        let inner = LinearScaleData {
            domain: [domain[0], domain[1]],
            range:  [range[0],  range[1]],
            clamp,
        };
        let s = TimeScale {
            data: inner,
            padding: None,
            range_user_set: true,
            utc: false,
            domain_user_set: true,
            reverse: false,
        };
        if nice { s.time_nice() } else { s }
    }

    /// Set the `utc` wire tag (F-L04-06): the resolver's `ScaleSpec::Utc` arm
    /// chains `.with_utc(true)` onto its `new_internal` result;
    /// `ScaleSpec::Time` and every auto-inferred construction leave the
    /// `new_internal` default (`false`) untouched. No rendering branch reads
    /// this bit — see the struct doc's "UTC by contract" note — so it affects
    /// nothing but round-trip fidelity.
    pub(crate) fn with_utc(mut self, utc: bool) -> Self {
        self.utc = utc;
        self
    }

    /// Crate-internal scale call (no PyO3 boundary).
    pub(crate) fn scale_internal(&self, x: f64) -> f64 {
        self.data.scale(x)
    }

    /// Crate-internal tick call (uses time-aware nice intervals).
    pub(crate) fn ticks_internal(&self, count: usize) -> Vec<f64> {
        self.time_ticks(count)
    }

    /// Return minor ticks subdivided uniformly between major calendar ticks.
    ///
    /// Time is already in a linear (epoch-ms) space, so the default
    /// subdivision algorithm applies directly: each major interval is divided
    /// into `DEFAULT_MINOR_SUBDIVISIONS` sub-intervals in epoch-ms, yielding
    /// 4 interior minor ticks per major gap.
    ///
    /// The major tick count is fixed at 10 (the conventional default).  Minor
    /// tick density is always `DEFAULT_MINOR_SUBDIVISIONS` (5 sub-intervals →
    /// 4 interior minors per gap); there is no per-call override.
    // Wired to the render layer via `ScaleKind::minor_tick_fractions`
    // (`render/scale_resolve/mod.rs`, dispatched through `dispatch_continuous!`).
    pub(crate) fn minor_ticks_internal(&self) -> Vec<Tick> {
        let majors = self.time_ticks(10);
        // Time domain is already linear (epoch-ms), so the identity transform
        // gives visually-uniform minor ticks.
        minor_ticks_default(&majors, |x| x)
    }

    pub(crate) fn range_pair(&self) -> [f64; 2] {
        self.data.range
    }

    pub(crate) fn domain_pair(&self) -> [f64; 2] {
        self.data.domain
    }

    /// Crate-internal accessor for the `utc` wire tag (F-L04-06), so
    /// resolver-side tests (`render::scale_resolve::positional`) can assert
    /// the tag survived resolution without a cross-module `pub(crate)` on
    /// the `#[pymethods]` `utc()` getter itself — mirrors `range_pair`/
    /// `domain_pair`'s pattern of a plain inherent accessor the pymethod
    /// getter is free to keep thin. `#[cfg(test)]`: no production caller
    /// reads this outside the crate's own test suite — `_to_scale_spec_dict`/
    /// `to_scale_spec` read `self.utc` directly, and the only Python-visible
    /// reader is the `#[pymethods] utc()` getter, which does the same. Adding
    /// this back to production would need a real production consumer, not a
    /// speculative one.
    #[cfg(test)]
    pub(crate) fn utc_flag(&self) -> bool {
        self.utc
    }

    /// This scale's domain endpoints rounded outward to the nearest calendar
    /// interval — the exact rounding `TimeScale(nice=True)` applies at
    /// construction (`time_nice`) — without mutating `self`. Unlike the other
    /// four continuous kinds, this is CALENDAR-aware (month/year boundaries
    /// via `chrono`), not a raw epoch-ms `nice_step` round: before
    /// `ScaleKind::niced_domain` existed, the chart-level `configure_axis(
    /// nice=True)` cascade rounded every kind with the same linear
    /// `nice_step`, silently landing a time axis on the wrong bounds. See
    /// [`LinearScale::nice_domain_pair`](super::linear::LinearScale::nice_domain_pair).
    pub(crate) fn nice_domain_pair(&self) -> [f64; 2] {
        self.time_nice().domain_pair()
    }

    /// Replace this scale's data-space domain in place, keeping its range and
    /// every kind-specific parameter.
    ///
    /// The sibling of [`domain_pair`](Self::domain_pair), added for the
    /// chart-level scale-domain config (D3, spec §4.2), which adjusts a
    /// RESOLVED domain rather than building a new scale — reconstructing via
    /// `new_internal` would have to re-supply parameters the caller cannot
    /// see. Because it is a second way into the domain field, it must apply
    /// whatever validation this kind's own constructor applies, or a config
    /// domain could store a value construction would have refused. Backed by `LinearScaleData`, which has no sanitizer — `new_internal` writes the pair straight through — so a raw write here matches construction exactly.
    ///
    /// `domain_user_set` flips to `true`: the domain now IS explicitly set (by
    /// the chart config), so `repr_string`/the `domain` getter must stop
    /// reporting it as data-derived.
    pub(crate) fn set_domain_pair(&mut self, domain: [f64; 2]) {
        self.data.domain = domain;
        self.domain_user_set = true;
    }

    pub(crate) fn repr_string(&self) -> String {
        let LinearScaleData { domain, range, clamp } = &self.data;
        let prefix = if self.utc { "TimeScale(utc=True, " } else { "TimeScale(" };
        // Mirrors `LinearScale::repr_string`'s `domain_user_set` guard
        // (F-L04-10): `domain` is no longer always user-set now that
        // `TimeScale(domain=None)` constructs, so this must stop reporting
        // the `[0, 1]` sentinel as if it were the caller's own value — the
        // exact gap the `set_domain_pair`/`domain()` doc comments already
        // promised this method would honor.
        let domain_s = if self.domain_user_set {
            format!("[{}, {}]", domain[0], domain[1])
        } else {
            "None".to_string()
        };
        // `reverse` only appears when non-default, matching the `utc` prefix's own
        // convention, so the default-shaped repr stays byte-identical to before.
        let reverse_s = if self.reverse { ", reverse=True" } else { "" };
        format!(
            "{}domain={}, range=[{}, {}], clamp={}{})",
            prefix, domain_s, range[0], range[1], if *clamp { "True" } else { "False" }, reverse_s
        )
    }

    /// Canonical `ScaleSpec` for this scale (SPEC-04 single-source bridge).
    ///
    /// `utc == true` maps to `ScaleSpec::Utc`, else `ScaleSpec::Time` — the
    /// `"utc"`/`"time"` wire tag the legacy `_scale_to_dict` emitted. `nice` is
    /// baked into the domain at construction, so it is always `false` here.
    pub(crate) fn to_scale_spec(&self) -> ScaleSpec {
        let common = continuous_common(
            self.data.domain,
            self.domain_user_set,
            self.data.range,
            self.range_user_set,
            self.data.clamp,
            self.padding,
            self.reverse,
        );
        if self.utc {
            ScaleSpec::Utc { common, nice: false }
        } else {
            ScaleSpec::Time { common, nice: false }
        }
    }

    fn time_ticks(&self, count: usize) -> Vec<f64> {
        let [d0, d1] = self.data.domain;
        // Pass domain values directly; calendar_ticks handles reversal when d0 > d1.
        calendar_ticks(d0, d1, count)
    }

    fn time_nice(&self) -> Self {
        let [d0, d1] = self.data.domain;
        let lo = d0.min(d1);
        let hi = d0.max(d1);
        let span = hi - lo;
        let cal = nice_calendar_interval(span, 10);
        let (new_lo, new_hi) = match cal {
            CalendarInterval::Month | CalendarInterval::Year => {
                use chrono::{Datelike, TimeZone, Utc};
                // Safe conversion: fall back to approximate arithmetic on out-of-range timestamps.
                let Some(dt_lo) = Utc.timestamp_millis_opt(lo as i64).single() else {
                    let iv = nice_time_interval_ms(span, 10);
                    if !iv.is_finite() || iv <= 0.0 { return self.clone(); }
                    return TimeScale {
                        data: LinearScaleData {
                            domain: [(lo / iv).floor() * iv, (hi / iv).ceil() * iv],
                            range: self.data.range,
                            clamp: self.data.clamp,
                        },
                        padding: self.padding,
                        range_user_set: self.range_user_set,
                        utc: self.utc,
                        domain_user_set: self.domain_user_set,
                        reverse: self.reverse,
                    };
                };
                let Some(dt_hi) = Utc.timestamp_millis_opt(hi as i64).single() else {
                    return self.clone();
                };
                let snapped_lo = if cal == CalendarInterval::Year {
                    Utc.with_ymd_and_hms(dt_lo.year(), 1, 1, 0, 0, 0)
                        .single().map(|t| t.timestamp_millis() as f64)
                        .unwrap_or(lo)
                } else {
                    Utc.with_ymd_and_hms(dt_lo.year(), dt_lo.month(), 1, 0, 0, 0)
                        .single().map(|t| t.timestamp_millis() as f64)
                        .unwrap_or(lo)
                };
                let (ny, nm) = if cal == CalendarInterval::Year {
                    (dt_hi.year() + 1, 1u32)
                } else {
                    let m = dt_hi.month() + 1;
                    if m > 12 { (dt_hi.year() + 1, 1u32) } else { (dt_hi.year(), m) }
                };
                let snapped_hi = Utc.with_ymd_and_hms(ny, nm, 1, 0, 0, 0)
                    .single().map(|t| t.timestamp_millis() as f64)
                    .unwrap_or(hi);
                (snapped_lo, snapped_hi)
            }
            _ => {
                let interval = nice_time_interval_ms(span, 10);
                if !interval.is_finite() || interval <= 0.0 {
                    return self.clone();
                }
                ((lo / interval).floor() * interval, (hi / interval).ceil() * interval)
            }
        };
        let new_domain = if d0 <= d1 { [new_lo, new_hi] } else { [new_hi, new_lo] };
        TimeScale {
            data: LinearScaleData { domain: new_domain, range: self.data.range, clamp: self.data.clamp },
            padding: self.padding,
            range_user_set: self.range_user_set,
            utc: self.utc,
            domain_user_set: self.domain_user_set,
            reverse: self.reverse,
        }
    }
}

// ── PyO3-boundary temporal domain extraction (F-L04-10) ─────────────────────
//
// One element of a `TimeScale(domain=[...])` list. Mirrors
// `ferrum.annotation.coords.temporal_coord_to_epoch_ms` — the Python
// canonical rule — element for element: a `float` or `int` (NOT `bool`,
// explicitly refused — see `temporal_value_to_epoch_ms`'s doc) is an
// epoch-ms value, unchanged/widened; a naive `datetime.datetime` or a
// `datetime.date` means UTC; an aware `datetime.datetime` converts to UTC;
// an ISO-8601 date or datetime string parses under the same rule.
// `int`/`bool` acceptance is a deliberate, adjudicated delta from
// `temporal_coord_to_epoch_ms` (which accepts no numeric type at all —
// see `temporal_value_to_epoch_ms`'s doc for why that's not a drift risk).
// `tests/test_timescale_domain.py`'s cross-language parity test proves the
// date/datetime/string rules agree across the full input taxonomy, so THOSE
// cannot silently drift from the Python one. Kept alongside `TimeScale` (not
// a separate scale-wide module) so `TimeScale` stays a single-source raw
// pyclass like its five continuous siblings.
struct TemporalDomainValue(f64);

impl FromPyObject<'_, '_> for TemporalDomainValue {
    type Error = PyErr;

    fn extract(ob: pyo3::Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        temporal_value_to_epoch_ms(&ob).map(TemporalDomainValue)
    }
}

/// Convert one Python domain value to epoch-milliseconds (UTC), per
/// [`TemporalDomainValue`]'s accepted-forms contract.
///
/// `datetime.datetime` is checked before `datetime.date` — a `datetime` IS-A
/// `date` (subclass), so casting to `PyDate` would silently succeed on a
/// full datetime and drop its time-of-day component if tried first.
///
/// `bool` is refused explicitly, ahead of the numeric branch: Python's
/// `bool` is an `int` subclass, so `ob.extract::<f64>()` would otherwise
/// silently accept `True`/`False` as `1.0`/`0.0` epoch-ms — a footgun
/// `temporal_coord_to_epoch_ms` also closes (its own signature has no
/// numeric type in its accepted union at all). Plain `int` (not `bool`)
/// keeps working as an epoch-ms value via the numeric branch below — a
/// DELIBERATE, adjudicated parity carve-out from `temporal_coord_to_epoch_ms`
/// (which never accepts numbers — annotation coordinates route numeric
/// values through a different path before that function ever sees them, an
/// unrelated call-site choice at the annotation layer). Every one of
/// `TimeScale`'s five continuous siblings accepts `int` domain values
/// through pyo3's ordinary numeric conversion (`Vec<f64>` from a Python list
/// containing `int`s), so `TimeScale` keeps that sibling-parity behavior
/// rather than narrowing to float-only to chase parity with a function that
/// was never in the numeric-acceptance business to begin with.
///
/// `pub(crate)` (batch-C task 4, F-L04-10): reused verbatim by
/// `spec::encoding`'s raw-dict scale gate to convert a
/// `{"type": "time"/"utc", "domain": [...]}` raw-dict element BEFORE the
/// dict is JSON-stringified for serde (a Python `datetime` object cannot
/// survive `json.dumps`, so this conversion has to happen at the PyO3
/// boundary, not downstream in serde) — see
/// `spec::encoding::convert_raw_dict_temporal_domain`. Not duplicated there;
/// that function calls this one directly rather than re-implementing any
/// part of the accepted-forms taxonomy.
pub(crate) fn temporal_value_to_epoch_ms(ob: &Bound<'_, PyAny>) -> PyResult<f64> {
    if let Ok(dt) = ob.cast::<PyDateTime>() {
        return datetime_epoch_ms(dt);
    }
    if let Ok(date) = ob.cast::<PyDate>() {
        return date_epoch_ms(date);
    }
    if ob.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(format!(
            "TimeScale domain values must be float (epoch-ms), datetime.date, datetime.datetime, \
             or an ISO-8601 date/datetime string; got bool ({ob}), which is not accepted as a \
             numeric epoch-ms value"
        )));
    }
    if let Ok(f) = ob.extract::<f64>() {
        return Ok(f);
    }
    if let Ok(s) = ob.extract::<String>() {
        return iso8601_string_epoch_ms(ob.py(), &s);
    }
    Err(PyTypeError::new_err(format!(
        "TimeScale domain values must be float (epoch-ms), datetime.date, datetime.datetime, \
         or an ISO-8601 date/datetime string; got {}",
        ob.get_type().name()?
    )))
}

/// A `datetime.datetime`: naive means UTC (matches `_coerce.py`'s handling of
/// naive polars `Datetime` columns); an aware value converts via its own
/// `.timestamp()` method. Python defines `.timestamp()` for an AWARE
/// datetime as `(dt - datetime(1970, 1, 1, tzinfo=timezone.utc))
/// .total_seconds()` — the exact definition `temporal_coord_to_epoch_ms`
/// uses — so this matches for any `tzinfo` implementation (fixed offset,
/// `zoneinfo`, or otherwise), not only `datetime.timezone.utc`. Naive
/// component access uses `getattr` rather than PyO3's `PyDateAccess` trait,
/// which requires direct C-struct field access unavailable under this
/// crate's `abi3-py310` limited-API build.
fn datetime_epoch_ms(dt: &Bound<'_, PyDateTime>) -> PyResult<f64> {
    if dt.get_tzinfo().is_some() {
        let secs: f64 = dt.call_method0("timestamp")?.extract()?;
        return Ok(secs * 1000.0);
    }
    let year: i32 = dt.getattr("year")?.extract()?;
    let month: u32 = dt.getattr("month")?.extract()?;
    let day: u32 = dt.getattr("day")?.extract()?;
    let hour: u32 = dt.getattr("hour")?.extract()?;
    let minute: u32 = dt.getattr("minute")?.extract()?;
    let second: u32 = dt.getattr("second")?.extract()?;
    let microsecond: u32 = dt.getattr("microsecond")?.extract()?;
    naive_epoch_ms(year, month, day, hour, minute, second, microsecond)
}

/// A `datetime.date`: midnight UTC on that calendar date.
fn date_epoch_ms(date: &Bound<'_, PyDate>) -> PyResult<f64> {
    let year: i32 = date.getattr("year")?.extract()?;
    let month: u32 = date.getattr("month")?.extract()?;
    let day: u32 = date.getattr("day")?.extract()?;
    naive_epoch_ms(year, month, day, 0, 0, 0, 0)
}

/// Days-and-time-of-day since the Unix epoch (UTC), mirroring CPython
/// `timedelta.total_seconds()`'s operation order —
/// `((days*86400 + seconds) * 1e6 + microseconds) / 1e6`, then `* 1000.0` —
/// to match `temporal_coord_to_epoch_ms`'s naive branch (`(value -
/// datetime(1970, 1, 1)).total_seconds() * 1000.0`).
///
/// **Bit-for-bit for any date within ~285 years of the epoch (1685–2255) —
/// not universally.** CPython's `total_seconds()` divides an *exact,
/// arbitrary-precision* integer numerator (`days*86400*1e6 + …`) by `1e6` in
/// one correctly-rounded step. This function instead casts that same
/// numerator to `f64` *before* dividing (`f64` has no arbitrary-precision
/// integer type to stage the numerator in); once the numerator's magnitude
/// exceeds `2^53` — i.e. once total seconds-since-epoch exceeds `2^53/1e6 ≈
/// 9.0072e9 s ≈ 285.4 years`, so a date before ~1685 or after ~2255 — the
/// intermediate cast itself rounds, and CPython's own true result may differ
/// by a sub-millisecond delta (verified empirically: `datetime(2300, 1, 1,
/// 0, 0, 0, 1)` gives `10413792000000.0` here vs. Python's
/// `10413792000000.002`). Every realistic chart date sits well inside the
/// exact window; replicating CPython's arbitrary-precision division for the
/// sliver of dates outside it would need a big-integer division routine this
/// function does not have, for a delta no rendered pixel can resolve.
fn naive_epoch_ms(
    year: i32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
    microsecond: u32,
) -> PyResult<f64> {
    let date = NaiveDate::from_ymd_opt(year, month, day).ok_or_else(|| {
        PyValueError::new_err(format!("invalid calendar date: {year:04}-{month:02}-{day:02}"))
    })?;
    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).expect("1970-01-01 is a valid date");
    let days = (date - epoch).num_days();
    let seconds_of_day = hour as i64 * 3600 + minute as i64 * 60 + second as i64;
    let total_seconds =
        ((days * 86_400 + seconds_of_day) as f64 * 1_000_000.0 + microsecond as f64) / 1_000_000.0;
    Ok(total_seconds * 1000.0)
}

/// ISO-8601 string → epoch-ms, mirroring `temporal_coord_to_epoch_ms`'s exact
/// fallback order — `date.fromisoformat` first (so a full datetime string,
/// which fails the date-only parse, falls through to the datetime branch),
/// then `datetime.fromisoformat`.
///
/// This calls the REAL `datetime.date.fromisoformat`/
/// `datetime.datetime.fromisoformat` classmethods rather than
/// re-implementing their grammar with `chrono`'s string parser: CPython's
/// `fromisoformat` accepted forms differ across Python versions (3.11 added
/// `Z`-suffix and looser fractional-second-digit-count support that 3.10
/// lacks — verified directly against this project's pinned interpreter,
/// which rejects `Z` and requires exactly 3 or 6 fractional digits), so a
/// hand-rolled Rust re-implementation would either under- or over-accept
/// relative to whatever `temporal_coord_to_epoch_ms` actually does at
/// runtime. Delegating to the same stdlib call the Python canonical
/// function makes eliminates that drift entirely rather than merely testing
/// for it. Once parsed, the resulting `date`/`datetime` object flows through
/// the exact same [`date_epoch_ms`]/[`datetime_epoch_ms`] Rust arithmetic
/// the object-typed input paths use — no second epoch-ms formula to keep in
/// sync.
fn iso8601_string_epoch_ms(py: Python<'_>, s: &str) -> PyResult<f64> {
    let datetime_mod = py.import("datetime")?;
    let date_cls = datetime_mod.getattr("date")?;
    let datetime_cls = datetime_mod.getattr("datetime")?;

    if let Ok(date_obj) = date_cls.call_method1("fromisoformat", (s,)) {
        return date_epoch_ms(date_obj.cast::<PyDate>()?);
    }
    if let Ok(dt_obj) = datetime_cls.call_method1("fromisoformat", (s,)) {
        return datetime_epoch_ms(dt_obj.cast::<PyDateTime>()?);
    }
    Err(PyValueError::new_err(format!(
        "Cannot parse TimeScale domain value {s:?} as an ISO-8601 date or datetime. \
         Use 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SS[.ffffff][±HH:MM]'."
    )))
}

#[pymethods]
impl TimeScale {
    #[new]
    #[pyo3(signature = (*, domain = None, range = None, clamp = false, nice = false, padding = None, utc = false, reverse = false))]
    fn new(
        domain: Option<Vec<TemporalDomainValue>>,
        range: Option<Vec<f64>>,
        clamp: bool,
        nice: bool,
        padding: Option<f64>,
        utc: bool,
        reverse: bool,
    ) -> PyResult<Self> {
        // Each domain element converts to epoch-ms at the PyO3 boundary
        // (`TemporalDomainValue`'s `FromPyObject`, F-L04-10) before reaching
        // the same `resolve_continuous` prelude every other continuous scale
        // uses — a `None` domain now infers from data like its five siblings
        // (SPEC-06), instead of TimeScale alone demanding an explicit one.
        let domain_ms = domain.map(|d| d.into_iter().map(|v| v.0).collect::<Vec<f64>>());
        let resolved = resolve_continuous(domain_ms, range, [0.0, 1.0])?;
        let inner = LinearScaleData {
            domain: resolved.domain,
            range: resolved.range,
            clamp,
        };
        let s = TimeScale {
            data: inner,
            padding,
            range_user_set: resolved.range_user_set,
            utc,
            domain_user_set: resolved.domain_user_set,
            reverse,
        };
        if nice && resolved.domain_user_set {
            Ok(s.time_nice())
        } else {
            Ok(s)
        }
    }

    /// Map an epoch-millisecond value ``x`` to its output range coordinate.
    fn scale(&self, x: f64) -> f64 { self.data.scale(x) }
    /// Invert a range coordinate ``y`` back to an epoch-millisecond value.
    fn invert(&self, y: f64) -> f64 { self.data.invert(y) }

    /// Return approximately ``count`` time-aligned tick values within the domain.
    ///
    /// Tick granularity snaps to calendar intervals (seconds, minutes, hours,
    /// days, months, or years) based on the domain span.
    #[pyo3(signature = (count = 10))]
    fn ticks(&self, count: usize) -> Vec<f64> { self.time_ticks(count) }

    /// Return a copy of this scale with domain endpoints rounded to the nearest calendar interval.
    fn nice(&self) -> Self { self.time_nice() }

    /// Input domain as ``[t_min, t_max]`` in epoch milliseconds, or ``None``
    /// when data-derived.
    ///
    /// Every element is stored already converted to epoch-ms (F-L04-10):
    /// whatever mix of `float`/`datetime.date`/`datetime.datetime`/ISO
    /// string a caller passed to `domain=`, this getter always returns
    /// floats — the same shape the five sibling continuous scales'
    /// `domain()` getters return.
    #[getter]
    fn domain(&self) -> Option<Vec<f64>> {
        if self.domain_user_set { Some(self.data.domain.to_vec()) } else { None }
    }

    /// Output range as ``[lo, hi]`` pixel coordinates, or ``None`` when
    /// the renderer should auto-fill from the plot-area dimensions.
    #[getter]
    fn range(&self) -> Option<Vec<f64>> {
        if self.range_user_set { Some(self.data.range.to_vec()) } else { None }
    }

    /// Whether out-of-domain inputs are clamped to the range endpoints.
    #[getter]
    fn clamp(&self) -> bool { self.data.clamp }

    /// Fractional inward pixel padding (themes-T4). ``None`` lets the renderer
    /// apply the 5% default when ``domain`` is unset.
    #[getter]
    fn padding(&self) -> Option<f64> { self.padding }

    /// Whether this is a UTC time scale (affects type serialization).
    #[getter]
    fn utc(&self) -> bool { self.utc }

    /// Whether this scale's domain is swapped when it resolves inside a
    /// chart render (descending axis). Does not affect this object's own
    /// `scale`/`invert`/`ticks`/`domain` — unlike `PointScale::reverse`,
    /// which DOES apply inside `PointScale::scale`.
    #[getter]
    fn reverse(&self) -> bool { self.reverse }

    /// Emit this scale's canonical `ScaleSpec` as a wire dict (SPEC-04 bridge).
    fn _to_scale_spec_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        scale_spec_to_py_dict(py, self.to_scale_spec())
    }

    fn __repr__(&self) -> String { self.repr_string() }
}

#[cfg(test)]
mod tests;
