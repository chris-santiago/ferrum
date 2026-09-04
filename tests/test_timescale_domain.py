"""Feature tests for ``TimeScale(domain=None)`` + datetime/date/ISO-string domain
acceptance and the UTC-by-contract behavior (F-L04-10 + F-L04-06).

Batch-C task 3's Python half. The Rust half (task 3's rust-coder) made
``TimeScale``'s ``domain`` optional (default ``None``, inferred like its five
continuous siblings), added a custom PyO3-boundary extraction accepting
``float | datetime.date | datetime.datetime | str`` domain elements under one
naive-means-UTC rule, and threaded ``utc`` through resolution so
``ScaleSpec::Utc`` resolves with ``utc=true`` (previously hardcoded
``false``). See ``.sdd/task-3-report.md`` for the Rust-side detail — notably
that the ISO-string path delegates to the real ``datetime.date.fromisoformat``
/ ``datetime.datetime.fromisoformat`` classmethods rather than a hand-rolled
parser, which is what makes the cross-language parity test below a real proof
rather than two independent reimplementations that happen to agree today.

Covers:
  1. ``TimeScale()`` constructs with no domain; on an encoding it renders with
     an inferred domain identical, byte-for-byte, to the no-explicit-scale
     render (the direct sibling-parity statement — ``TimeScale()`` infers
     like its five continuous siblings) and with deterministic tick labels;
     the raw-dict ``{"type": "time"}`` spelling matches the class spelling.
  2. Datetime taxonomy: naive datetime, aware datetime (multiple non-UTC
     fixed offsets plus one ``zoneinfo.ZoneInfo`` case spanning a real DST
     transition), date, ISO string, float, and mixed-element domains all
     render byte-identical SVG to the equivalent explicit epoch-float
     domain — guarded by a standalone sentinel proving the suite would
     actually catch a regression that ignores the explicit domain entirely.
  3. Refusal: a non-temporal domain value raises, naming the accepted forms.
  4. Cross-language parity (the mandatory test): for the full input taxonomy,
     ``TimeScale(domain=[v, v2]).domain`` equals
     ``[temporal_coord_to_epoch_ms(v), temporal_coord_to_epoch_ms(v2)]`` from
     ``ferrum.annotation.coords`` — the canonical naive-means-UTC converter —
     and both sides reject the same invalid inputs with the same exception
     type. Two numeric-tower carve-outs are pinned explicitly rather than
     left as silent gaps: ``int`` is a *deliberate* documented delta (Rust
     accepts it as epoch-ms for sibling parity with ``float``;
     ``temporal_coord_to_epoch_ms`` refuses bare numbers), while ``bool``
     joins the refusal-parity list on *both* sides (Rust rejects it — a
     bool must not silently pass as an epoch-ms float — and the canonical
     converter already refuses it).
  5. ``utc=True``/``utc=False`` render byte-identical SVG through
     ``to_svg()`` — the Python-visible pin of the UTC-by-contract rule.

RED-proof note (discriminating by construction, not a toggled runtime check):
the cross-language parity assertions in section 4 only prove something if the
two conversions (Rust's ``TemporalDomainValue`` extraction vs. Python's
``temporal_coord_to_epoch_ms``) could actually disagree. Proven by simulated
divergence during development (not committed as a test, since it would
permanently fail): asserting
``scale.domain == [temporal_coord_to_epoch_ms(lo) + 3_600_000.0, ...]`` (an
off-by-one-hour expectation) against the real ``TimeScale(domain=[lo, hi])``
fails with a clear mismatch, confirming the real assertion
(``scale.domain == [temporal_coord_to_epoch_ms(lo), ...]``) is non-vacuous —
it is not comparing a value against itself by construction, and a future
divergence between the Rust extraction and the Python converter (e.g. a
timezone-arithmetic bug on one side only) would be caught here.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import polars as pl
import pytest

import ferrum as fm
import ferrum._core as fc
from ferrum.annotation.coords import temporal_coord_to_epoch_ms
from tests._svg_extents import axis_tick_labels

# ---------------------------------------------------------------------------
# Fixed epoch-ms pair: 2020-06-01T00:00:00 UTC / 2020-06-02T00:00:00 UTC.
# Every taxonomy case below is a differently-spelled representation of this
# exact instant pair, so every render/parity comparison has one fixed target.
# ---------------------------------------------------------------------------

_LO_MS = 1_590_969_600_000.0
_HI_MS = 1_591_056_000_000.0
_FLOAT_DOMAIN = [_LO_MS, _HI_MS]


# ---------------------------------------------------------------------------
# Chart-building helper
# ---------------------------------------------------------------------------


def _svg_for_domain(domain: list[object], *, utc: bool = False) -> str:
    """Render a fixed 2-point chart with an explicit ``TimeScale`` domain on ``x``.

    The data column is always the epoch-ms float pair ``[_LO_MS, _HI_MS]`` so
    every domain spelling (float, date, datetime, ISO string, mixed) plots
    the identical two marks — only the scale's domain representation varies
    between call sites, isolating the domain-conversion behavior under test
    from any data/rendering concern.
    """
    df = pl.DataFrame({"x": [_LO_MS, _HI_MS], "y": [0.0, 100.0]})
    scale = fc.TimeScale(domain=list(domain), nice=False, utc=utc)
    chart = fm.Chart(df).mark_point().encode(x=fm.X("x", scale=scale), y="y")
    return chart.to_svg()


# ---------------------------------------------------------------------------
# 1. TimeScale() constructs; inferred-domain render; raw-dict equivalence
# ---------------------------------------------------------------------------


def test_timescale_constructs_with_no_domain() -> None:
    """``TimeScale()`` constructs like its five continuous siblings — no required kwarg."""
    scale = fc.TimeScale()
    assert scale.domain is None


def test_timescale_renders_with_inferred_domain_on_encoding() -> None:
    """A domain-less ``TimeScale`` on a temporal encoding infers its domain from the data.

    Two discriminating assertions, not just "a render happened": the tick
    labels are the deterministic monthly sequence the inferred 2020 domain
    produces (a wildly wrong inferred domain, e.g. one anchored at the 1970
    epoch, would produce different labels), and the render is byte-identical
    to the no-explicit-scale render (``encode(x="date")``) — the direct
    sibling-parity statement that ``TimeScale()`` infers exactly like its
    five continuous siblings, stated as an equality rather than implied.
    """
    df = pl.DataFrame(
        {
            "date": [dt.date(2020, 1, 1), dt.date(2020, 6, 1), dt.date(2020, 12, 1)],
            "y": [1.0, 2.0, 3.0],
        }
    )
    chart_explicit = fm.Chart(df).mark_point().encode(x=fm.X("date", scale=fm.TimeScale()), y="y")
    chart_no_scale = fm.Chart(df).mark_point().encode(x="date", y="y")
    svg_explicit = chart_explicit.to_svg()
    svg_no_scale = chart_no_scale.to_svg()

    assert svg_explicit == svg_no_scale, (
        "TimeScale() must infer its domain identically to the no-explicit-scale sibling path"
    )
    labels = axis_tick_labels(svg_explicit, axis="x")
    assert labels == ["Jan 2020", "Mar 2020", "May 2020", "Jul 2020", "Sep 2020", "Nov 2020"]


def test_raw_dict_time_spelling_matches_class_spelling() -> None:
    """``scale={"type": "time"}`` renders byte-identical to ``scale=fm.TimeScale()``.

    Byte-equality alone would be vacuous if both paths silently produced a
    blank render, so this also pins that the shared SVG actually has a
    rendered tick axis (mirrors ``test_scale_reverse.py``'s dict-parity
    pattern).
    """
    df = pl.DataFrame(
        {
            "date": [dt.date(2020, 1, 1), dt.date(2020, 6, 1), dt.date(2020, 12, 1)],
            "y": [1.0, 2.0, 3.0],
        }
    )
    chart_class = fm.Chart(df).mark_point().encode(x=fm.X("date", scale=fm.TimeScale()), y="y")
    chart_dict = fm.Chart(df).mark_point().encode(x=fm.X("date", scale={"type": "time"}), y="y")

    svg_class = chart_class.to_svg()
    svg_dict = chart_dict.to_svg()
    assert svg_class == svg_dict

    labels = axis_tick_labels(svg_class, axis="x")
    assert len(labels) >= 2


# ---------------------------------------------------------------------------
# 2. Datetime taxonomy: every accepted spelling renders identically to its
#    epoch-float equivalent
# ---------------------------------------------------------------------------

# A DST-crossing zoneinfo pair cannot land on the shared _LO_MS/_HI_MS
# instants (New York's spring-forward transition is in March, not the June
# dates every other taxonomy case represents), so each case carries its own
# float-equivalent baseline rather than always comparing against the shared
# _FLOAT_DOMAIN. Every non-DST case's baseline is still _FLOAT_DOMAIN.
_DST_LO = dt.datetime(2020, 3, 8, 1, 30, 0, tzinfo=ZoneInfo("America/New_York"))  # EST, UTC-5
_DST_HI = dt.datetime(2020, 3, 8, 3, 30, 0, tzinfo=ZoneInfo("America/New_York"))  # EDT, UTC-4
_DST_FLOAT_DOMAIN = [
    temporal_coord_to_epoch_ms(_DST_LO),
    temporal_coord_to_epoch_ms(_DST_HI),
]

_TAXONOMY_CASES: list[tuple[str, list[object], list[float]]] = [
    ("float", [_LO_MS, _HI_MS], _FLOAT_DOMAIN),
    ("date", [dt.date(2020, 6, 1), dt.date(2020, 6, 2)], _FLOAT_DOMAIN),
    ("naive_datetime", [dt.datetime(2020, 6, 1), dt.datetime(2020, 6, 2)], _FLOAT_DOMAIN),
    (
        "aware_datetime_utc",
        [
            dt.datetime(2020, 6, 1, tzinfo=dt.timezone.utc),
            dt.datetime(2020, 6, 2, tzinfo=dt.timezone.utc),
        ],
        _FLOAT_DOMAIN,
    ),
    (
        "aware_datetime_plus5",
        [
            dt.datetime(2020, 6, 1, 5, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=5))),
            dt.datetime(2020, 6, 2, 5, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=5))),
        ],
        _FLOAT_DOMAIN,
    ),
    (
        "aware_datetime_minus8",
        [
            dt.datetime(2020, 5, 31, 16, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=-8))),
            dt.datetime(2020, 6, 1, 16, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=-8))),
        ],
        _FLOAT_DOMAIN,
    ),
    (
        "aware_datetime_zoneinfo_dst",
        [_DST_LO, _DST_HI],
        _DST_FLOAT_DOMAIN,
    ),
    ("iso_date", ["2020-06-01", "2020-06-02"], _FLOAT_DOMAIN),
    ("iso_datetime", ["2020-06-01T00:00:00", "2020-06-02T00:00:00"], _FLOAT_DOMAIN),
    (
        "iso_datetime_offset",
        ["2020-06-01T05:00:00+05:00", "2020-06-02T05:00:00+05:00"],
        _FLOAT_DOMAIN,
    ),
    ("mixed_date_and_iso_string", [dt.date(2020, 6, 1), "2020-06-02T00:00:00"], _FLOAT_DOMAIN),
    (
        "mixed_datetime_and_float",
        [dt.datetime(2020, 6, 1, tzinfo=dt.timezone.utc), _HI_MS],
        _FLOAT_DOMAIN,
    ),
]
_TAXONOMY_IDS = [name for name, _domain, _float_domain in _TAXONOMY_CASES]


@pytest.mark.parametrize("name, domain, float_domain", _TAXONOMY_CASES, ids=_TAXONOMY_IDS)
def test_datetime_taxonomy_renders_identically_to_epoch_float_equivalent(
    name: str, domain: list[object], float_domain: list[float]
) -> None:
    """Every accepted temporal domain spelling renders byte-identical SVG to
    the equivalent explicit float (epoch-ms) domain — domain conversion is a
    pure input-normalization step with zero effect on what gets drawn.
    """
    svg_temporal = _svg_for_domain(domain)
    svg_float = _svg_for_domain(float_domain)
    assert svg_temporal == svg_float, (
        f"{name}: temporal domain must render identically to its epoch-float equivalent"
    )


def test_taxonomy_suite_is_discriminating_against_ignored_explicit_domain() -> None:
    """Sentinel proving the byte-identity taxonomy above is not vacuously green.

    Every case in the taxonomy compares two renders over the *same* fixed
    data column (``_LO_MS``/``_HI_MS``) — only the scale's domain
    representation varies. If a regression made an explicit ``TimeScale``
    domain be silently ignored (falling back to inferring from the data
    column instead), every taxonomy case would still pass, since the data
    column never changes between cases. This test pins that a real domain
    shift actually changes the rendered SVG, so the taxonomy suite would
    catch that regression.
    """
    svg_base = _svg_for_domain(_FLOAT_DOMAIN)
    svg_shifted = _svg_for_domain([_LO_MS + 3_600_000.0, _HI_MS + 3_600_000.0])
    assert svg_base != svg_shifted, (
        "an explicit domain shift must change the render — otherwise the domain is being ignored"
    )


# ---------------------------------------------------------------------------
# 3. Refusal: a non-temporal value raises, naming the accepted forms
# ---------------------------------------------------------------------------


def test_refuses_non_temporal_value_naming_accepted_forms() -> None:
    """A domain element of an unsupported type raises ``TypeError`` naming the accepted forms."""
    with pytest.raises(TypeError, match="datetime.date, datetime.datetime, or an ISO-8601"):
        fc.TimeScale(domain=[object(), 1.0])


def test_refuses_unparseable_iso_string() -> None:
    """A string that is not valid ISO-8601 date/datetime raises ``ValueError``."""
    with pytest.raises(ValueError, match="ISO-8601"):
        fc.TimeScale(domain=["not-a-date", "2020-06-02"])


# ---------------------------------------------------------------------------
# 4. Cross-language parity: TimeScale's Rust extraction vs.
#    ferrum.annotation.coords.temporal_coord_to_epoch_ms
# ---------------------------------------------------------------------------

_PARITY_CASES: list[tuple[str, object, object]] = [
    ("date", dt.date(2020, 6, 1), dt.date(2020, 6, 2)),
    ("naive_datetime", dt.datetime(2020, 6, 1), dt.datetime(2020, 6, 2)),
    (
        "aware_datetime_utc",
        dt.datetime(2020, 6, 1, tzinfo=dt.timezone.utc),
        dt.datetime(2020, 6, 2, tzinfo=dt.timezone.utc),
    ),
    (
        "aware_datetime_plus5",
        dt.datetime(2020, 6, 1, 5, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=5))),
        dt.datetime(2020, 6, 2, 5, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=5))),
    ),
    (
        "aware_datetime_minus8",
        dt.datetime(2020, 5, 31, 16, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=-8))),
        dt.datetime(2020, 6, 1, 16, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=-8))),
    ),
    ("aware_datetime_zoneinfo_dst", _DST_LO, _DST_HI),
    ("iso_date", "2020-06-01", "2020-06-02"),
    ("iso_datetime", "2020-06-01T00:00:00", "2020-06-02T00:00:00"),
    ("iso_datetime_offset", "2020-06-01T05:00:00+05:00", "2020-06-02T05:00:00+05:00"),
    ("iso_datetime_microseconds", "2020-06-01T00:00:00.123456", "2020-06-02T00:00:00.654321"),
    ("iso_datetime_negative_offset", "2020-06-01T00:00:00-05:00", "2020-06-02T00:00:00-05:00"),
    ("mixed_date_and_iso_string", dt.date(2020, 6, 1), "2020-06-02T00:00:00"),
]
_PARITY_IDS = [name for name, _lo, _hi in _PARITY_CASES]


@pytest.mark.parametrize("name, lo, hi", _PARITY_CASES, ids=_PARITY_IDS)
def test_cross_language_parity_matches_temporal_coord_to_epoch_ms(
    name: str, lo: object, hi: object
) -> None:
    """``TimeScale``'s Rust-side domain conversion agrees with
    ``temporal_coord_to_epoch_ms`` — the annotation layer's canonical
    converter — for every accepted temporal input kind, across the full
    naive/aware/date/ISO-string taxonomy.
    """
    scale = fc.TimeScale(domain=[lo, hi], nice=False)
    expected = [temporal_coord_to_epoch_ms(lo), temporal_coord_to_epoch_ms(hi)]
    assert scale.domain == expected, (
        f"{name}: Rust conversion diverged from temporal_coord_to_epoch_ms"
    )


def test_cross_language_parity_float_passthrough_is_identity() -> None:
    """Float domain elements pass through unchanged on both sides.

    ``temporal_coord_to_epoch_ms`` is typed for ``date | datetime | str`` and
    does not accept ``float`` (it isn't a coordinate that needs converting),
    so float is the one taxonomy member this test asserts identity for
    directly rather than routing through the Python converter.
    """
    scale = fc.TimeScale(domain=[_LO_MS, _HI_MS], nice=False)
    assert scale.domain == [_LO_MS, _HI_MS]


def test_cross_language_parity_int_is_a_deliberate_documented_delta() -> None:
    """``int`` is the one taxonomy member where Rust and Python deliberately disagree.

    ``TimeScale(domain=[1, 2])`` accepts ``int`` the same way it accepts
    ``float`` — PyO3's ``extract::<f64>()`` coerces any Python number,
    matching the sibling continuous scales (``LinearScale(domain=[1, 2])``
    is unremarkable). ``temporal_coord_to_epoch_ms``, by contrast, is typed
    ``date | datetime | str`` and refuses a bare number outright — it is not
    a coordinate that needs converting. This is a real, permanent asymmetry
    (not a gap to close): pinned here per the batch-C ledger adjudication so
    a future reader sees it as intentional rather than an untested edge.
    """
    scale = fc.TimeScale(domain=[1, 2], nice=False)
    assert scale.domain == [1.0, 2.0]

    with pytest.raises(TypeError):
        temporal_coord_to_epoch_ms(1)  # type: ignore[arg-type]


_REFUSAL_PARITY_CASES: list[tuple[str, object, type[Exception]]] = [
    ("unparseable_string", "not-a-date", ValueError),
    ("invalid_calendar_string", "2020-13-45", ValueError),
    ("non_temporal_object", object(), TypeError),
    ("nested_list", [1, 2, 3], TypeError),
    ("bool_true", True, TypeError),
    ("bool_false", False, TypeError),
]
_REFUSAL_PARITY_IDS = [name for name, _bad, _exc in _REFUSAL_PARITY_CASES]


@pytest.mark.parametrize(
    "name, bad_value, expected_exc", _REFUSAL_PARITY_CASES, ids=_REFUSAL_PARITY_IDS
)
def test_refusal_parity_rejects_same_invalid_inputs(
    name: str, bad_value: object, expected_exc: type[Exception]
) -> None:
    """``TimeScale``'s Rust extraction and ``temporal_coord_to_epoch_ms`` reject
    the same invalid inputs with the same exception type.

    Paired with ``2.0`` rather than ``1.0`` so a bad value that happens to
    coerce to ``1.0`` (e.g. ``True``, before it was rejected outright) can
    never collide into a degenerate ``[1.0, 1.0]`` domain and raise the
    domain-length ``ValueError`` instead of the type-rejection error this
    test is actually probing.
    """
    with pytest.raises(expected_exc):
        fc.TimeScale(domain=[bad_value, 2.0])
    with pytest.raises(expected_exc):
        temporal_coord_to_epoch_ms(bad_value)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 5. utc=True/False byte-identity through to_svg — the Python-visible pin
# ---------------------------------------------------------------------------


def test_utc_true_and_false_render_byte_identical_through_to_svg() -> None:
    """``utc=True`` and ``utc=False`` render byte-identical SVG — UTC-by-contract.

    The ``utc`` flag only distinguishes the ``Time`` vs. ``Utc`` wire tag; it
    never changes what gets drawn, since every temporal rendering path is
    always UTC.
    """
    svg_false = _svg_for_domain(_FLOAT_DOMAIN, utc=False)
    svg_true = _svg_for_domain(_FLOAT_DOMAIN, utc=True)
    assert svg_false == svg_true

    labels = axis_tick_labels(svg_false, axis="x")
    assert len(labels) >= 2, "byte-identity must not be vacuous over a blank render"
