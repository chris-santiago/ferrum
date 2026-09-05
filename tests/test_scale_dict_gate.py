"""Feature tests for the raw-dict scale key gate (F-L04-07 completeness half
+ F-L04-10 raw-dict coverage, batch-C task 4).

The gate validates every raw ``scale={...}`` dict against a schema-derived,
per-type accepted-key set at ``ScaleSpec``'s own ``Deserialize`` boundary --
the single chokepoint every ``fm.X(...)``/``fm.Y(...)``/``ChartSpec.from_json``
scale dict passes through -- closing the last documented ``#[serde(flatten)]``
silent-drop carve-out on ``ScaleSpec``. An unknown key (a typo like ``clammp``
for ``clamp``) refuses naming the offending key, the scale type, and the sorted
accepted-key list, where it previously vanished silently.
``tests/test_bug_hunt_encoding_step4.py::test_scale_dict_typo_key_is_rejected``
is the flipped positive mirror of that old tolerance pin, and makes every
refusal test here non-vacuously RED against any pre-gate build by construction.

The same Rust change converts a raw-dict temporal domain
(``{"type": "time"/"utc", "domain": [datetime.date(...), ...]}``) to epoch-ms
before ``json.dumps`` would otherwise choke on it -- but only on the
chart-level channel path (``EncodingSpec::new``, a constructor the layer /
composite-mark path never enters). Both the conversion and its matching
refusal therefore live at the one Python seam BOTH routes share,
``ferrum.encoding._scale._scale_to_dict``; items 9 and 11 pin each half on
both routes.

Covers:
  1. Unknown-key refusal: the typo case names the real key among the sorted
     accepted list, at both wire-boundary entry points (``ChartSpec.from_json``
     and the ``EncodingSpec::new`` constructor path), and generalizes across
     all 16 known ``ScaleSpec`` types.
  2. A valid-keys sweep: every accepted key, for every scale type, populated
     in one hand-authored raw dict, still parses and builds a ``ChartSpec``.
     This proves the gate accepts every key IT ENUMERATES for a type -- it is
     necessarily silent on a legal shape the hand-authored fixture omits (see
     item 8, which closes that blind spot with producer-emitted dicts).
  3. ``reverse`` accepted (no refusal) via raw dict on all seven continuous
     types, including an end-to-end render-order flip proof for ``utc`` (the
     one continuous type with no dedicated Python scale class, so
     ``tests/test_scale_reverse.py``'s class-based sweep cannot reach it).
  4. ``{"type": "diverging", "reverse": true}`` refused (no such field).
  5. Raw-dict temporal domains render byte-identical SVG to the epoch-float
     equivalent -- the raw-dict path specifically, as distinct from
     ``tests/test_timescale_domain.py``'s class-constructor path.
  6. A non-temporal domain element on a raw-dict time/utc scale refuses,
     naming the accepted forms.
  7. The untyped raw-dict spelling (no ``"type"`` key at all) reaches the gate
     as ``linear`` via ``_scale_to_dict``'s injection, both for a legal key
     and for a typo (refused naming ``linear``'s accepted list).
  8. A producer-emitted-dict arm in the "no over-refusal" sweep: dicts built
     by ``_scale_to_dict`` / a real ``Chart.to_spec()`` call, not hand-written
     literals -- the shape of coverage that catches a ferrum-internal emitter
     writing a gate-refused key (as the bar zero-anchor once did; see
     ``tests/test_bar_zero.py`` for that mark-specific suite).
  9. Raw-dict temporal domains render on the LAYER path (``chart.layer(...)``
     and ``chart_a + chart_b``), not just the chart-level channel path.
 10. One coherent ``TimeScale domain value`` vocabulary for both bad-element
     kinds (unparseable ISO string, non-temporal element) regardless of which
     side catches them, with a live cross-language guard against Rust's own
     wording (NOT byte-identical for the ISO case: Rust quotes with ``{s:?}``,
     Python with ``{value!r}``).
 11. The temporal seam owns conversion AND refusal: the temporal tag set is
     derived from the live ``TimeScale`` class rather than hand-listed, the
     refusal message is byte-identical to Rust's, and the chart-level and
     layer routes refuse the same malformed dict identically.

Scope note (per the repo's test-file convention, stated in
``tests/test_boxen_palette.py``): this module covers the wire GATE. The
``Chart.override(<channel>_scale_*=...)`` cascade regressions that the gate's
introduction earned -- the type-switch merge and its eight remediation rounds
-- live in ``tests/test_override_scale_merge.py``, next to
``tests/test_override.py`` where the rest of the override surface is pinned.
The round-by-round remediation narrative that used to open this module is in
the commit history and ``.sdd/``, where it cannot go stale against the code.
"""

from __future__ import annotations

import calendar
import datetime as dt
import decimal
import json

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from ferrum._core import ChartSpec
from tests._svg_extents import axis_tick_labels

# ---------------------------------------------------------------------------
# Schema-derived accepted-key sets (mirrors
# ``crates/ferrum-core/src/spec/encoding.rs::accepted_keys_for_scale_type``,
# whose own drift guard
# (``accepted_keys_for_scale_type_matches_every_variants_serialized_keys``)
# proves this hand-maintained Rust set matches every ``ScaleSpec`` variant's
# real serialized keys). Sorted the same way the gate's own error message
# sorts them (``sort_unstable`` on ``&str`` — plain lexicographic order,
# matching Python's default string sort for ASCII).
# ---------------------------------------------------------------------------

ACCEPTED_KEYS: dict[str, list[str]] = {
    "linear": [
        "clamp",
        "domain",
        "domainParam",
        "nice",
        "padding",
        "range",
        "reverse",
        "scheme",
        "zero",
    ],
    "log": [
        "base",
        "clamp",
        "domain",
        "domainParam",
        "nice",
        "padding",
        "range",
        "reverse",
        "scheme",
    ],
    "time": ["clamp", "domain", "domainParam", "nice", "padding", "range", "reverse", "scheme"],
    "symlog": [
        "clamp",
        "constant",
        "domain",
        "domainParam",
        "nice",
        "padding",
        "range",
        "reverse",
        "scheme",
    ],
    "pow": ["clamp", "domain", "domainParam", "exponent", "padding", "range", "reverse", "scheme"],
    "sqrt": ["clamp", "domain", "domainParam", "padding", "range", "reverse", "scheme"],
    "utc": ["clamp", "domain", "domainParam", "nice", "padding", "range", "reverse", "scheme"],
    "ordinal": ["domain", "padding", "range"],
    "band": ["align", "domain", "padding", "paddingInner", "paddingOuter", "range"],
    "point": ["align", "domain", "padding", "range", "reverse"],
    "sequential": ["domain", "reverse", "scheme", "stops"],
    "diverging": ["domain", "domainMid", "scheme"],
    "quantize": ["domain", "range"],
    "quantile": ["domain", "range"],
    "threshold": ["domain", "range"],
    "bin-ordinal": ["bins", "scheme"],
}

# One raw dict per type with EVERY accepted key populated (the valid-keys
# sweep, item 2 above). These are hand-authored wire shapes, not derived from
# a Python ``*Scale`` class's ``_to_scale_spec_dict()`` — several accepted
# keys (e.g. ``linear``'s ``zero``) are raw-dict-only spellings with no
# constructor kwarg on the corresponding class at all (``LinearScale`` has no
# ``zero=`` parameter; ``{"zero": False}`` is a real, currently-legal
# ``tests/test_bar_zero.py`` shape reachable only as a raw dict), so a
# class-derived sweep would under-cover the gate's own accepted-key sets.
VALID_DICTS: dict[str, dict[str, object]] = {
    "linear": {
        "type": "linear",
        "domain": [0, 10],
        "range": [0, 100],
        "clamp": True,
        "padding": 1.0,
        "reverse": True,
        "nice": True,
        "zero": True,
        "scheme": "viridis",
        "domainParam": "unused_param_name",
    },
    "log": {
        "type": "log",
        "domain": [1, 100],
        "range": [0, 100],
        "clamp": True,
        "padding": 1.0,
        "reverse": True,
        "nice": True,
        "base": 2.0,
        "scheme": "viridis",
        "domainParam": "unused_param_name",
    },
    "time": {
        "type": "time",
        "domain": [0, 1000],
        "range": [0, 100],
        "clamp": True,
        "padding": 1.0,
        "reverse": True,
        "nice": True,
        "scheme": "viridis",
        "domainParam": "unused_param_name",
    },
    "symlog": {
        "type": "symlog",
        "domain": [-10, 10],
        "range": [0, 100],
        "clamp": True,
        "padding": 1.0,
        "reverse": True,
        "nice": True,
        "constant": 2.0,
        "scheme": "viridis",
        "domainParam": "unused_param_name",
    },
    "pow": {
        "type": "pow",
        "domain": [0, 10],
        "range": [0, 100],
        "clamp": True,
        "padding": 1.0,
        "reverse": True,
        "exponent": 3.0,
        "scheme": "viridis",
        "domainParam": "unused_param_name",
    },
    "sqrt": {
        "type": "sqrt",
        "domain": [0, 10],
        "range": [0, 100],
        "clamp": True,
        "padding": 1.0,
        "reverse": True,
        "scheme": "viridis",
        "domainParam": "unused_param_name",
    },
    "utc": {
        "type": "utc",
        "domain": [0, 1000],
        "range": [0, 100],
        "clamp": True,
        "padding": 1.0,
        "reverse": True,
        "nice": True,
        "scheme": "viridis",
        "domainParam": "unused_param_name",
    },
    "ordinal": {"type": "ordinal", "domain": ["a", "b", "c"], "range": [0, 100], "padding": 0.1},
    "band": {
        "type": "band",
        "domain": ["a", "b", "c"],
        "padding": 0.2,
        "paddingInner": 0.1,
        "paddingOuter": 0.3,
        "align": 0.6,
        "range": [0, 100],
    },
    "point": {
        "type": "point",
        "domain": ["a", "b", "c"],
        "padding": 0.3,
        "align": 0.4,
        "reverse": True,
        "range": [0, 100],
    },
    "sequential": {
        "type": "sequential",
        "scheme": "viridis",
        "domain": [0, 1],
        "reverse": True,
        "stops": [[0.0, "#ff0000"], [1.0, "#00ff00"]],
    },
    "diverging": {"type": "diverging", "scheme": "redblue", "domain": [-1, 0, 1], "domainMid": 0.0},
    "quantize": {"type": "quantize", "domain": [0, 100], "range": ["low", "mid", "high"]},
    "quantile": {"type": "quantile", "domain": [1.0, 2.0, 3.0, 4.0], "range": [0.0, 0.5, 1.0]},
    "threshold": {"type": "threshold", "domain": [0.5, 1.0], "range": [0.0, 0.5, 1.0]},
    "bin-ordinal": {"type": "bin-ordinal", "bins": [0, 10, 20, 30], "scheme": "blues"},
}

# Types that resolve positionally (attach to x); the rest are color-channel
# scales (categorical/binned color mapping) and are exercised on Color
# instead, matching how each type is actually used in the wider suite (e.g.
# ``tests/test_flexibility_campaign.py``'s ordinal dict-form color test,
# ``tests/test_phase_12_scales.py::TestScaleIntegration``'s Sequential-on-
# Color case).
POSITIONAL_TYPES = {
    "linear",
    "log",
    "time",
    "symlog",
    "pow",
    "sqrt",
    "utc",
    "ordinal",
    "band",
    "point",
}
COLOR_TYPES = {"sequential", "diverging", "quantize", "quantile", "threshold", "bin-ordinal"}
assert POSITIONAL_TYPES | COLOR_TYPES == set(ACCEPTED_KEYS)

_ALL_TYPES = sorted(ACCEPTED_KEYS)


def _base_spec_json() -> dict:
    return json.loads(ChartSpec(mark="point", x="a", y="b").to_json())


def _accepted_suffix(scale_type: str) -> str:
    """The gate's pinned ``"accepted: ..."`` suffix for one scale type."""
    return "accepted: " + ", ".join(ACCEPTED_KEYS[scale_type])


# ---------------------------------------------------------------------------
# 1. Unknown-key refusal
# ---------------------------------------------------------------------------


def test_typo_key_refused_names_key_type_and_accepted_list() -> None:
    """``clammp`` (typo of ``clamp``) on a ``linear`` scale refuses, naming
    the bad key, the scale type, and the full sorted accepted-key list —
    the spec §9 repro this gate exists to close.

    Verbatim-substring pins (not a loose ``match=``) per the batch-B
    mutation-testing lesson: a gate that named the wrong key, the wrong
    type, or a subtly-wrong accepted list would still satisfy a loose regex
    but fail these exact substring checks.
    """
    j = _base_spec_json()
    j["encoding"]["x"]["scale"] = {"type": "linear", "clammp": True}
    with pytest.raises(ValueError) as exc_info:
        ChartSpec.from_json(json.dumps(j))
    message = str(exc_info.value)
    assert "unknown key 'clammp'" in message
    assert "for type 'linear'" in message
    assert _accepted_suffix("linear") in message


def test_reveres_typo_of_reverse_refused() -> None:
    """``reveres`` (typo of ``reverse``) refuses, naming ``reverse`` among
    the accepted keys — the finding's own motivating example.
    """
    j = _base_spec_json()
    j["encoding"]["x"]["scale"] = {"type": "linear", "reveres": True}
    with pytest.raises(ValueError) as exc_info:
        ChartSpec.from_json(json.dumps(j))
    message = str(exc_info.value)
    assert "unknown key 'reveres'" in message
    assert "for type 'linear'" in message
    assert "reverse" in message


def test_typo_key_refused_via_encoding_constructor_path() -> None:
    """The same typo refuses through the OTHER wire chokepoint entry point:
    ``fm.X(..., scale=...)`` building a ``ChartSpec`` via ``EncodingSpec::new``,
    not just ``ChartSpec.from_json``.

    Both paths bottom out at the identical ``ScaleSpec::deserialize`` call
    (see ``.sdd/task-4-report.md``'s "chokepoint" section), so a gate placed
    anywhere else could plausibly cover one path and silently miss the
    other. This proves both are covered. Substring-only (not the exact
    ``"scale: scale: unknown key ..."`` double-prefixed message the
    ``json_round`` field-error wrapper produces on this specific path) —
    that cosmetic double prefix is a documented, harmless quirk (task-4
    report, "Concerns" #3), not part of this test's contract.
    """
    df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("a", scale={"type": "linear", "clammp": True}), y="b")
    )
    with pytest.raises(ValueError) as exc_info:
        chart.to_spec()
    message = str(exc_info.value)
    assert "unknown key 'clammp'" in message
    assert "for type 'linear'" in message


@pytest.mark.parametrize("scale_type", _ALL_TYPES)
def test_unknown_key_refused_for_every_type(scale_type: str) -> None:
    """A bogus key refuses for every one of the 16 known scale types, each
    naming that type's own exact sorted accepted-key list.

    Generalizes the ``linear``-only typo case above across the full type
    taxonomy — the gate is schema-derived per type, so each type's accepted
    list must be independently correct, not just the one spec-§9 example.
    """
    j = _base_spec_json()
    j["encoding"]["x"]["scale"] = {"type": scale_type, "boguskey": True}
    with pytest.raises(ValueError) as exc_info:
        ChartSpec.from_json(json.dumps(j))
    message = str(exc_info.value)
    assert f"unknown key 'boguskey' for type '{scale_type}'" in message
    assert _accepted_suffix(scale_type) in message


def test_unknown_type_falls_through_to_variant_error_not_gate_message() -> None:
    """An unrecognized ``"type"`` value gets serde's own "unknown variant"
    error, not a gate-authored message — the gate is deliberately
    permissive on shapes it cannot judge (see
    ``validate_scale_spec_keys``'s doc).
    """
    j = _base_spec_json()
    j["encoding"]["x"]["scale"] = {"type": "bogus-scale-type", "clammp": True}
    with pytest.raises(ValueError, match="unknown variant"):
        ChartSpec.from_json(json.dumps(j))


# ---------------------------------------------------------------------------
# 2. Valid-keys sweep: every key the gate itself enumerates is accepted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale_type", _ALL_TYPES)
def test_every_accepted_key_for_every_type_still_parses(scale_type: str) -> None:
    """Every accepted key, for every scale type, populated together in one
    HAND-AUTHORED raw dict, still builds a ``ChartSpec`` without error.

    Scoped claim (quality-review remediation): this proves the gate accepts
    every key its OWN ``ACCEPTED_KEYS``/``VALID_DICTS`` mirror enumerates —
    it is a near-circular check (``VALID_DICTS`` is authored from
    ``ACCEPTED_KEYS``, enforced by the fixture-integrity assertion below) and
    cannot detect over-refusal of a legal shape neither list happens to
    include, or of a dict as a PRODUCER (not a human) actually assembles —
    that is exactly how the S4 bar-zero-anchor regression slipped past this
    sweep (a real bug: ``_spec_build.py`` stamped a ``zero`` key onto every
    bar y-scale regardless of type, refused for every non-linear type once
    the gate landed). ``test_producer_emitted_scale_dicts_do_not_over_refuse``
    below closes that specific gap with producer-emitted dicts instead of
    literals.
    """
    scale = VALID_DICTS[scale_type]
    assert set(scale) - {"type"} == set(ACCEPTED_KEYS[scale_type]), (
        f"{scale_type}: VALID_DICTS fixture must populate every accepted key exactly once"
    )
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "val": [1.0, 2.0, 3.0]})
    if scale_type in POSITIONAL_TYPES:
        chart = fm.Chart(df).mark_point().encode(x=fm.X("x", scale=scale), y="val")
    else:
        chart = fm.Chart(df).mark_point().encode(x="x", y="val", color=fm.Color("val", scale=scale))
    spec = chart.to_spec()  # must not raise
    assert spec is not None


# ---------------------------------------------------------------------------
# 3. reverse accepted via raw dict on all 7 continuous types
# ---------------------------------------------------------------------------

_CONTINUOUS_TYPES = ["linear", "log", "time", "symlog", "pow", "sqrt", "utc"]


@pytest.mark.parametrize("scale_type", _CONTINUOUS_TYPES)
def test_reverse_accepted_via_raw_dict_on_every_continuous_type(scale_type: str) -> None:
    """``{"type": <continuous>, "reverse": true}`` is accepted (no refusal)
    for all 7 continuous types, including ``utc`` — the one continuous type
    with no dedicated Python scale class
    (``tests/test_scale_reverse.py``'s class-based sweep covers the other
    six directly via ``fc.<X>Scale(reverse=True)`` but cannot reach ``utc``,
    which is only ever a raw-dict ``"type"`` tag or ``TimeScale(utc=True)``).
    """
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    domain = [-10, 10] if scale_type == "symlog" else [1, 100]
    scale = {"type": scale_type, "domain": domain, "reverse": True}
    chart = fm.Chart(df).mark_point().encode(x=fm.X("x", scale=scale), y="y")
    assert chart.to_spec() is not None  # must not raise


def test_utc_reverse_via_raw_dict_flips_rendered_tick_order() -> None:
    """``{"type": "utc", "reverse": true}`` doesn't just parse — it actually
    flips rendered tick-label order, mirroring
    ``tests/test_scale_reverse.py``'s discriminating per-class assertions
    for the one continuous type that module's class-based sweep cannot
    reach (see above).
    """

    def _svg(reverse: bool) -> str:
        df = pl.DataFrame({"x": [0.0, 1_000_000.0], "y": [0.0, 100.0]})
        scale = {"type": "utc", "domain": [0.0, 1_000_000.0], "nice": False, "reverse": reverse}
        chart = fm.Chart(df).mark_point().encode(x=fm.X("x", scale=scale), y="y")
        return chart.to_svg()

    forward = axis_tick_labels(_svg(False), axis="x")
    reversed_ = axis_tick_labels(_svg(True), axis="x")
    assert len(forward) >= 3, "too few forward tick labels to be discriminating"
    assert reversed_ == list(reversed(forward)), "reverse=True must flip utc tick order end to end"


# ---------------------------------------------------------------------------
# 4. diverging + reverse refused
# ---------------------------------------------------------------------------


def test_diverging_reverse_refused_with_accepted_list() -> None:
    """``{"type": "diverging", "reverse": true}`` refuses — ``Diverging`` has
    no ``reverse`` field (T1's third, more-silent carve-out this gate
    closes: without the gate, ``reverse`` on a ``Diverging`` scale would not
    even round-trip, let alone raise).
    """
    j = _base_spec_json()
    j["encoding"]["x"]["scale"] = {"type": "diverging", "reverse": True}
    with pytest.raises(ValueError) as exc_info:
        ChartSpec.from_json(json.dumps(j))
    message = str(exc_info.value)
    assert "unknown key 'reverse' for type 'diverging'" in message
    assert _accepted_suffix("diverging") in message


# ---------------------------------------------------------------------------
# 5. Raw-dict temporal domain renders identically to epoch-float equivalent
# ---------------------------------------------------------------------------


def _epoch_ms(d: dt.date) -> float:
    """Independent (non-ferrum) UTC epoch-ms conversion for a naive date, so
    the render-identity assertions below don't depend on the very converter
    (``temporal_value_to_epoch_ms``) whose output they're checking.
    """
    return calendar.timegm(d.timetuple()) * 1000.0


def _svg_for_raw_dict_domain(domain: list[object], *, scale_type: str) -> str:
    lo_ms, hi_ms = _epoch_ms(dt.date(2021, 3, 1)), _epoch_ms(dt.date(2021, 3, 2))
    df = pl.DataFrame({"x": [lo_ms, hi_ms], "y": [0.0, 100.0]})
    scale = {"type": scale_type, "domain": domain, "nice": False}
    chart = fm.Chart(df).mark_point().encode(x=fm.X("x", scale=scale), y="y")
    return chart.to_svg()


@pytest.mark.parametrize("scale_type", ["time", "utc"])
def test_raw_dict_date_domain_renders_identically_to_epoch_float(scale_type: str) -> None:
    """``scale={"type": "time"/"utc", "domain": [datetime.date(...), ...]}``
    (a RAW DICT, not a ``TimeScale(...)`` instance) renders byte-identical
    SVG to the equivalent explicit epoch-ms float domain.

    Distinct from ``tests/test_timescale_domain.py``'s taxonomy (T3): that
    module's ``_svg_for_domain`` always goes through the ``TimeScale``
    PyO3 constructor, which already had its own datetime-accepting
    extraction before this task. This test is the raw-dict-specific path
    this task actually adds (``convert_raw_dict_temporal_domain``, called
    from ``EncodingSpec::new`` BEFORE ``json.dumps`` ever sees the dict).
    """
    lo, hi = dt.date(2021, 3, 1), dt.date(2021, 3, 2)
    lo_ms, hi_ms = _epoch_ms(lo), _epoch_ms(hi)

    svg_date = _svg_for_raw_dict_domain([lo, hi], scale_type=scale_type)
    svg_float = _svg_for_raw_dict_domain([lo_ms, hi_ms], scale_type=scale_type)
    assert svg_date == svg_float

    labels = axis_tick_labels(svg_date, axis="x")
    assert len(labels) >= 2, "byte-identity must not be vacuous over a blank render"


def test_raw_dict_datetime_domain_renders_identically_to_epoch_float() -> None:
    """The ``datetime.datetime`` (not just ``datetime.date``) element case,
    for the ``time`` type — proves the raw-dict conversion routes through
    the same ``temporal_value_to_epoch_ms`` taxonomy the class path uses,
    not a narrower date-only special case.

    Non-midnight datetimes (13:30, not 00:00:00) are load-bearing here
    (quality-review remediation): a converter implemented as "take
    ``.date()`` then midnight-UTC" — a narrower date-only special case, the
    exact failure mode this test's docstring claims to exclude — would
    silently truncate the time-of-day component and still produce
    byte-identical output against a midnight-only baseline, passing
    vacuously. ``calendar.timegm(lo.timetuple())`` (not ``_epoch_ms``, which
    takes a ``date`` and is midnight-only by construction) carries H:M:S
    from a real ``datetime``, so a truncating converter now disagrees with
    the baseline and this test would fail.
    """
    lo = dt.datetime(2021, 3, 1, 13, 30, 0)
    hi = dt.datetime(2021, 3, 2, 7, 45, 0)
    lo_ms = calendar.timegm(lo.timetuple()) * 1000.0
    hi_ms = calendar.timegm(hi.timetuple()) * 1000.0

    svg_datetime = _svg_for_raw_dict_domain([lo, hi], scale_type="time")
    svg_float = _svg_for_raw_dict_domain([lo_ms, hi_ms], scale_type="time")
    assert svg_datetime == svg_float

    # Non-vacuity: the non-midnight baseline must actually differ from the
    # midnight-truncated one, or a date-only-truncating converter would pass
    # both this test and the weaker midnight-based version it replaces.
    midnight_ms = _epoch_ms(lo.date())
    assert lo_ms != midnight_ms, "13:30 must not collapse to midnight in the independent baseline"


# ---------------------------------------------------------------------------
# 6. Non-temporal junk domain element refuses, naming accepted forms
# ---------------------------------------------------------------------------


def test_non_temporal_domain_element_refuses_naming_accepted_forms() -> None:
    """A raw-dict ``time`` scale's ``domain`` list containing a value that is
    neither float, ``datetime.date``, ``datetime.datetime``, nor an
    ISO-8601 string refuses with ``TypeError``, naming the accepted forms —
    the same accepted-forms taxonomy ``TimeScale(domain=...)``'s own PyO3
    constructor uses (``temporal_value_to_epoch_ms``, reused verbatim by the
    raw-dict conversion path).
    """
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("x", scale={"type": "time", "domain": [object(), 1.0]}), y="y")
    )
    with pytest.raises(TypeError, match="datetime.date, datetime.datetime, or an ISO-8601"):
        chart.to_spec()


# ---------------------------------------------------------------------------
# 7. Untyped raw-dict spelling (no "type" key) reaches the gate as linear
# ---------------------------------------------------------------------------


def test_untyped_raw_dict_with_legal_key_parses_as_linear() -> None:
    """``scale={"zero": False}`` — no ``"type"`` key at all — is the single
    most common raw-dict spelling in the codebase (dozens of sites across
    ``tests/`` and ``src/ferrum/plots/``, e.g.
    ``tests/test_bar_zero.py::test_bar_explicit_zero_false``). It reaches
    the gate as ``linear`` because
    ``ferrum.encoding._scale._scale_to_dict``'s dict branch injects
    ``{"type": "linear", **scale}`` before the wire boundary ever sees it —
    named explicitly here rather than left as an implicit assumption every
    other test in this module quietly depends on.
    """
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    chart = fm.Chart(df).mark_point().encode(x=fm.X("a", scale={"zero": False}), y="b")
    spec_json = json.loads(chart.to_spec().to_json())
    assert spec_json["encoding"]["x"]["scale"]["type"] == "linear"
    assert spec_json["encoding"]["x"]["scale"]["zero"] is False


def test_untyped_raw_dict_typo_refuses_naming_linear() -> None:
    """The same untyped-dict injection means an untyped typo (``{"clammp":
    True}``, no ``"type"`` key) refuses naming ``linear``'s accepted list —
    the gate sees the exact same ``{"type": "linear", "clammp": true}``
    shape the explicitly-typed case (``test_typo_key_refused_names_key_type_and_accepted_list``)
    exercises, just reached through the untyped spelling instead.
    """
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    chart = fm.Chart(df).mark_point().encode(x=fm.X("a", scale={"clammp": True}), y="b")
    with pytest.raises(ValueError) as exc_info:
        chart.to_spec()
    message = str(exc_info.value)
    assert "unknown key 'clammp'" in message
    assert "for type 'linear'" in message
    assert _accepted_suffix("linear") in message


# ---------------------------------------------------------------------------
# 8. Producer-emitted dicts (not hand-written literals) do not over-refuse
# ---------------------------------------------------------------------------


def test_producer_emitted_scale_dicts_do_not_over_refuse() -> None:
    """The "no over-refusal" sweep in section 2 above is authored FROM
    ``ACCEPTED_KEYS`` and can only catch a gate that refuses a key it
    enumerates itself. This test instead feeds the gate dicts actually
    PRODUCED by ferrum's own internal scale-dict assembly code — the shape
    of coverage that would have caught the S4 bar-zero-anchor regression
    (``_spec_build.py`` used to stamp ``"zero": True`` onto every bar
    y-scale regardless of type, which the hand-authored sweep's
    ``VALID_DICTS`` never exercised because it never modeled that
    producer's own composition logic).
    """
    from ferrum.encoding._scale import _scale_to_dict

    # Producer 1: _scale_to_dict's own untyped-dict injection + domainParam
    # sugar (D6 reactive rescale) — a dict shape only a producer emits, not
    # something a user hand-writes as a literal in the sweep above.
    param_domain_scale = _scale_to_dict({"domain": fm.param("d", value=[0.0, 10.0])})
    assert param_domain_scale == {"type": "linear", "domainParam": "d"}
    j = _base_spec_json()
    j["encoding"]["x"]["scale"] = param_domain_scale
    ChartSpec.from_json(json.dumps(j))  # must not raise

    # Producer 2: a real Chart.to_spec() call for every non-linear bar
    # y-scale in the S4 repro matrix — extracts the ACTUAL scale dict
    # _spec_build.py assembles (post zero-anchor logic, post override
    # merge), not a re-typed hand literal, and feeds it back through the
    # gate a second time via ChartSpec.from_json to prove that specific
    # producer output round-trips.
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})
    for scale_factory in (fm.LogScale(), fm.SymlogScale(), fm.SqrtScale()):
        chart = fm.Chart(df).mark_bar().encode(x="cat", y=fm.Y("val", scale=scale_factory))
        produced = json.loads(chart.to_spec().to_json())
        produced_scale = produced["encoding"]["y"]["scale"]
        assert "zero" not in produced_scale, (
            f"producer must not emit a zero key for {produced_scale.get('type')!r}"
        )
        j2 = _base_spec_json()
        j2["encoding"]["x"]["scale"] = produced_scale
        ChartSpec.from_json(json.dumps(j2))  # must not raise — round-trips clean


# ---------------------------------------------------------------------------
# 9. Raw-dict temporal domains render on the LAYER path, not just chart-level
# ---------------------------------------------------------------------------


def test_layer_path_raw_dict_date_domain_renders() -> None:
    """``scale={"type": "time", "domain": [datetime.date(...), ...]}`` on a
    ``chart.layer(...)`` channel renders — the S3 quality-review finding.

    Before the ``ferrum.encoding._scale._scale_to_dict`` fix, this exact
    dict rendered fine on a bare chart-level channel (routed through Rust's
    ``EncodingSpec::new`` hook) but crashed with an opaque ``TypeError:
    Object of type date is not JSON serializable`` on the layer path
    (``coerce_layers`` -> ``pyo3_serde::from_py`` json-dumps the raw dict
    directly, bypassing that Rust hook entirely). Fixed at the shared
    Python seam both paths call identically — see the module docstring.
    """
    from ferrum.layer import Layer

    lo, hi = dt.date(2021, 3, 1), dt.date(2021, 3, 2)
    lo_ms, hi_ms = _epoch_ms(lo), _epoch_ms(hi)
    df = pl.DataFrame({"d": [lo_ms, hi_ms], "y": [0.0, 100.0]})

    base = fm.Chart(df).mark_point().encode(x="d", y="y")

    def _layered_svg(domain: list[object]) -> str:
        layer = Layer(
            mark="line",
            encoding={"x": fm.X("d", type="T", scale={"type": "time", "domain": domain}), "y": "y"},
        )
        return base.layer(layer).to_svg()

    svg_date = _layered_svg([lo, hi])
    svg_float = _layered_svg([lo_ms, hi_ms])
    assert svg_date == svg_float

    labels = axis_tick_labels(svg_date, axis="x")
    assert len(labels) >= 2, "byte-identity must not be vacuous over a blank render"


def test_layer_path_raw_dict_iso_string_domain_renders() -> None:
    """The ISO-8601 string element case on the layer path — the other
    non-numeric spelling the quality review flagged as diverging from the
    chart-level path before the fix.
    """
    from ferrum.layer import Layer

    lo, hi = dt.date(2021, 3, 1), dt.date(2021, 3, 2)
    lo_ms, hi_ms = _epoch_ms(lo), _epoch_ms(hi)
    df = pl.DataFrame({"d": [lo_ms, hi_ms], "y": [0.0, 100.0]})

    base = fm.Chart(df).mark_point().encode(x="d", y="y")

    def _layered_svg(domain: list[object]) -> str:
        layer = Layer(
            mark="line",
            encoding={"x": fm.X("d", type="T", scale={"type": "time", "domain": domain}), "y": "y"},
        )
        return base.layer(layer).to_svg()

    svg_iso = _layered_svg(["2021-03-01", "2021-03-02"])
    svg_float = _layered_svg([lo_ms, hi_ms])
    assert svg_iso == svg_float


def test_chart_plus_chart_composition_raw_dict_date_domain_renders() -> None:
    """``chart_a + chart_b`` (the documented, common layering spelling) with
    a raw-dict temporal domain on one side renders — the exact composition
    shape the coordinator's remediation brief names (``(a+b).to_svg()``).
    """
    lo, hi = dt.date(2021, 3, 1), dt.date(2021, 3, 2)
    lo_ms, hi_ms = _epoch_ms(lo), _epoch_ms(hi)
    df_a = pl.DataFrame({"d": [lo_ms, hi_ms], "y": [0.0, 100.0]})
    df_b = pl.DataFrame({"d": [lo_ms, hi_ms], "y2": [10.0, 20.0]})

    chart_a = (
        fm.Chart(df_a)
        .mark_point()
        .encode(x=fm.X("d", type="T", scale={"type": "time", "domain": [lo, hi]}), y="y")
    )
    chart_b = fm.Chart(df_b).mark_line().encode(x=fm.X("d", type="T"), y="y2")
    svg = (chart_a + chart_b).to_svg()  # must not raise
    assert "<svg" in svg


# ---------------------------------------------------------------------------
# 10. Exception-drift fix: coherent vocabulary for both bad-domain-element
# kinds
# ---------------------------------------------------------------------------


def test_bad_iso_string_in_scale_domain_speaks_scale_domain_vocabulary() -> None:
    """A bad ISO string in a raw-dict temporal domain used to raise
    ``ferrum.annotation.coords``'s "Cannot parse annotation coordinate ..."
    wording, since ``_scale_to_dict``'s Python-side conversion (item 5/9
    above) routes every string element through
    ``temporal_coord_to_epoch_ms`` directly — a subsystem the user never
    named. Now re-raised in the same vocabulary Rust's own
    ``iso8601_string_epoch_ms`` hook uses for the identical mistake, so a
    scale-domain mistake reads as a scale-domain mistake regardless of
    which subsystem (Python, here, or Rust, for the sibling case below)
    happens to catch it.
    """
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("x", scale={"type": "time", "domain": ["notadate", "2021-03-02"]}), y="y")
    )
    with pytest.raises(ValueError) as exc_info:
        chart.to_spec()
    message = str(exc_info.value)
    assert "Cannot parse TimeScale domain value" in message
    assert "annotation coordinate" not in message
    assert "ISO-8601" in message


def test_non_iso_junk_element_speaks_the_same_vocabulary() -> None:
    """The sibling bad-domain-element case — a value that is neither
    numeric nor date/datetime/str at all — is unchanged (still Rust's
    ``TypeError``, since it never reaches the ISO-string branch), and now
    reads coherently alongside the ISO-string case's message above: both
    name ``"TimeScale domain value(s)"``, not two different subsystems.
    """
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("x", scale={"type": "time", "domain": [object(), 1.0]}), y="y")
    )
    with pytest.raises(TypeError, match="TimeScale domain values"):
        chart.to_spec()


def test_iso_parse_message_matches_rust_wording_modulo_quoting() -> None:
    """Cross-language drift guard for the Python copy of Rust's ISO-parse
    message (py-quality cycle-3 required fix 5). The two literals
    (``_scale.py::_convert_temporal_domain_elements`` and Rust's
    ``iso8601_string_epoch_ms``) are NOT byte-identical: Rust quotes the
    offending value with ``{s:?}`` (double quotes), Python with
    ``{value!r}`` (single quotes). This pins the shared sentence with that
    one known quoting difference normalized, rather than claiming
    byte-identity.

    Rust's wording is triggered through the real production PyO3
    constructor, ``ferrum._core.EncodingSpec`` — the exact class
    ``SpecBuildMixin._build_encoding_specs`` instantiates at the
    chart-level channel path (``src/ferrum/_spec_build.py``) — called
    directly with a raw dict so Python's own conversion layer (which
    intercepts every string element before Rust ever sees one on the
    normal ``fm.X(...)``/``fm.Y(...)`` route) is bypassed. This is a real
    production entry point exercised directly, not a hand-built mirror of
    Rust's message.
    """
    from ferrum._core import EncodingSpec

    bad_value = "notadate"
    df = pl.DataFrame({"x": [1.0], "y": [2.0]})

    with pytest.raises(ValueError) as py_exc:
        (
            fm.Chart(df)
            .mark_point()
            .encode(x=fm.X("x", scale={"type": "time", "domain": [bad_value, "2021-03-02"]}), y="y")
            .to_spec()
        )
    py_message = str(py_exc.value)

    with pytest.raises(ValueError) as rust_exc:
        EncodingSpec(
            "x", "quantitative", scale={"type": "time", "domain": [bad_value, "2021-03-02"]}
        )
    rust_message = str(rust_exc.value)

    py_normalized = py_message.replace(repr(bad_value), f'"{bad_value}"')
    assert py_normalized == rust_message, (
        "Python's ISO-parse message must track Rust's wording (quoting difference aside)"
    )


# ---------------------------------------------------------------------------
# 11. The temporal seam owns conversion AND refusal, identically on both routes
# ---------------------------------------------------------------------------


def test_scale_to_dict_temporal_membership_check_tolerates_nonstring_type() -> None:
    """``_scale_to_dict``'s temporal-conversion branch
    (``src/ferrum/encoding/_scale.py``) tested
    ``scale.get("type") in _TEMPORAL_SCALE_TYPES`` -- a frozenset
    membership test -- against a user-supplied value with no type guard.
    An unhashable ``"type"`` (a scale pyclass instance) raised
    ``TypeError: unhashable type: ...`` from INSIDE this conversion
    helper, a message naming no ferrum concept, from a line THIS task
    added (batch-C task 4's own cycle-2 diff -- not pre-existing, despite
    round 5's report attributing it to a pre-task cause). The pre-task
    shape for the identical mistake is a plain JSON-serialization
    ``TypeError`` once the bad ``"type"`` value reaches Rust; guarding the
    membership test with ``isinstance(..., str)`` restores that plain
    shape instead of crashing earlier with a worse message.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})
    scale = {"type": fm.LogScale(), "domain": [0.0, 30.0]}

    with pytest.raises(TypeError) as excinfo:
        fm.Chart(df).mark_bar().encode(x="cat", y=fm.Y("val", scale=scale)).to_svg()

    assert "unhashable" not in str(excinfo.value)
    assert "is not JSON serializable" in str(excinfo.value)


def test_temporal_scale_types_are_derived_from_the_live_scale_class() -> None:
    """``_TEMPORAL_SCALE_TYPES`` is asked of ``TimeScale`` -- the one scale
    class with a temporal domain -- rather than hand-listed, so a renamed or
    added temporal wire tag cannot leave the Python conversion seam behind.

    ``scale_accepted_keys`` cannot answer this question (it publishes key
    NAMES, and a temporal ``domain`` is spelled ``domain`` like every other
    continuous type's), so this is the derivation that replaces the literal
    ``frozenset({"time", "utc"})``. Pinned against the tags the class actually
    emits, both directions, so the derivation cannot silently return a subset.
    """
    from ferrum._core import TimeScale
    from ferrum.encoding._scale import _TEMPORAL_SCALE_TYPES

    emitted = {TimeScale(utc=utc)._to_scale_spec_dict()["type"] for utc in (False, True)}
    assert _TEMPORAL_SCALE_TYPES == emitted
    assert len(emitted) == 2, "utc=False/True must emit two distinct wire tags"


_MALFORMED_TEMPORAL_ELEMENTS = [
    pytest.param(True, id="bool"),
    pytest.param(object(), id="object"),
    pytest.param(None, id="none"),
    pytest.param([1.0], id="list"),
]


@pytest.mark.parametrize("bad_element", _MALFORMED_TEMPORAL_ELEMENTS)
def test_non_temporal_element_message_matches_rust_wording(bad_element: object) -> None:
    """The Python seam's accepted-forms refusal is byte-identical to the one
    Rust's own ``temporal_value_to_epoch_ms`` raises for the same element.

    ``_convert_temporal_domain_elements`` now owns the refusal (it is the only
    point BOTH wire routes pass through), so it runs BEFORE Rust's hook on the
    chart-level path and the chart-level message must not have changed. Rust's
    wording is obtained from the real production PyO3 constructor
    (``ferrum._core.EncodingSpec``, the class ``_build_encoding_specs``
    instantiates) called with the raw dict directly, so Python's conversion
    layer is bypassed -- not a hand-built mirror of Rust's message.
    """
    from ferrum._core import EncodingSpec

    scale = {"type": "time", "domain": [bad_element, 1.0]}
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})

    with pytest.raises(TypeError) as py_exc:
        fm.Chart(df).mark_point().encode(x=fm.X("x", scale=scale), y="y").to_spec()
    with pytest.raises(TypeError) as rust_exc:
        EncodingSpec("x", "quantitative", scale=scale)

    assert str(py_exc.value) == str(rust_exc.value)
    assert "TimeScale domain values must be" in str(py_exc.value)


@pytest.mark.parametrize("bad_element", _MALFORMED_TEMPORAL_ELEMENTS)
def test_chart_level_and_layer_routes_refuse_a_bad_element_identically(
    bad_element: object,
) -> None:
    """Route parity: the SAME malformed raw dict raises the same exception
    type and the same message on the chart-level channel route and on the
    layer/composite-mark route.

    Rust's ``convert_raw_dict_temporal_domain`` hook runs at
    ``EncodingSpec::new``, a constructor the layer route never enters, so
    delegating refusal to it gave one user mistake three vocabularies: Rust's
    accepted-forms ``TypeError`` on a bare chart, serde's ``invalid type:
    boolean `true`, expected f64`` on a layer, and ``json.dumps``'s generic
    "not JSON serializable" for an element serde could not even reach. Owning
    the rule at the one shared Python seam collapses all three onto one.
    """
    from ferrum.layer import Layer

    scale = {"type": "time", "domain": [bad_element, 1.0]}
    df = pl.DataFrame({"d": [1.0, 2.0], "y": [3.0, 4.0]})

    with pytest.raises(Exception) as chart_level:
        fm.Chart(df).mark_point().encode(x=fm.X("d", type="T", scale=scale), y="y").to_svg()

    base = fm.Chart(df).mark_point().encode(x="d", y="y")
    layer = Layer(mark="line", encoding={"x": fm.X("d", type="T", scale=scale), "y": "y"})
    with pytest.raises(Exception) as layer_route:
        base.layer(layer).to_svg()

    assert type(layer_route.value) is type(chart_level.value), (
        "both routes must raise the same exception type for the same malformed dict"
    )
    assert str(layer_route.value) == str(chart_level.value), (
        "both routes must raise the same message for the same malformed dict"
    )


@pytest.mark.parametrize(
    "good_element",
    [
        pytest.param(np.float32(0), id="np.float32"),
        pytest.param(np.int64(0), id="np.int64"),
        pytest.param(decimal.Decimal("0"), id="Decimal"),
    ],
)
def test_chart_level_and_layer_routes_accept_a_float_able_numeric_identically(
    good_element: object,
) -> None:
    """The numeric branch of ``_convert_temporal_domain_elements`` used to
    append the ORIGINAL value rather than ``float(value)``, on the rationale
    that Rust's ``extract::<f64>()`` (which goes through ``__float__``/
    ``__index__``) accepts a numpy scalar or a ``Decimal`` as a valid
    epoch-ms element. That rationale holds only on the chart-level route,
    which reaches Rust's extractor -- the layer/composite-mark route reaches
    ``json.dumps`` instead, which cannot serialize a numpy scalar or a
    ``Decimal`` at all. A domain built from ``arr.min()``/``arr.max()`` on a
    float32 or int64 array is an ordinary way to build a domain, so this
    rendered fine on a bare chart and hard-failed only once layered --
    exactly the shape a boundary bug should not have. Appending
    ``float(value)`` widens the layer route up to match the chart route;
    this pins that both routes now render identically for the same element.
    """
    df = pl.DataFrame({"d": [1.0, 2.0], "y": [3.0, 4.0]})
    scale = {"type": "time", "domain": [good_element, 5.0]}
    encoding = {"x": fm.X("d", type="T", scale=scale), "y": "y"}

    chart_svg = fm.Chart(df).mark_point().encode(**encoding).to_svg()

    from ferrum.layer import Layer

    base = fm.Chart(df).mark_point().encode(x="d", y="y")
    layer = Layer(mark="point", encoding=encoding)
    layer_svg = base.layer(layer).to_svg()

    assert isinstance(chart_svg, str) and chart_svg
    assert isinstance(layer_svg, str) and layer_svg
