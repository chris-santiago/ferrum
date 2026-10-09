"""Regression tests for the ``Chart.override(<channel>_scale_*=...)`` cascade.

``tests/test_scale_dict_gate.py`` covers the raw-dict scale key GATE, a
wire-boundary feature; this module covers ``Chart.override``'s scale-leaf
merge, which is override-cascade behavior that merely *interacts* with the
gate. Per the repo's test-file convention (stated in
``tests/marks/test_boxen_palette.py``'s module docstring), the two belong in separate
modules -- and this one sits next to ``tests/test_override.py``, where the rest
of the override surface, including the per-leaf render sweep that guards the
leaf registry itself, is pinned.

The subject is ``ferrum._override_consume.merge_encoding_scale`` (formerly
``_spec_build._merge_override_scale``), reached through the production render
path only -- every test here goes through ``Chart.to_spec()`` / ``Chart.to_svg()``,
never the helper directly. Its contract, in the order the sections below pin it:

  1. A type-changing override drops the base scale's keys the NEW type does not
     accept, in both directions and on both positional and color channels, and
     a same-type override drops nothing (sections 1-2).
  2. A key BOTH types accept survives the switch (section 2).
  3. ``mark_bar().override(y_scale_domain=...)`` suppresses the zero-anchor
     exactly as the equivalent explicit ``scale={"domain": ...}`` does -- a
     deliberate, disclosed behavior change (section 3).
  4. A base-scale key that was already invalid under its OWN type refuses
     through the real wire gate, with the identical message the same base scale
     raises standalone: never laundered into silence by the switch, never
     promoted into effect because the new type declares a same-named field
     (sections 4, 6, 8).
  5. A malformed ``<channel>_scale_type=`` value (non-``str``, or an unknown
     tag) surfaces the gate's own refusal, not a PyO3 argument-coercion error
     naming an internal parameter (section 5).
  6. A key both types accept is NOT value-validated under the type the switch is
     replacing: a string ``domain`` illegal under ``linear`` and legal under
     ``band`` must survive (section 7).
"""

from __future__ import annotations

import json
import re

import polars as pl
import pytest

import ferrum as fm

# ---------------------------------------------------------------------------
# 1. Recurring S4: override-scale-merge type switch over an explicit base
# ---------------------------------------------------------------------------

_TYPE_SWITCH_BASE_SCALES: list[tuple[str, object]] = [
    ("class:LinearScale", fm.LinearScale()),
    ("dict:typed-linear-zero-false", {"type": "linear", "zero": False}),
    ("dict:untyped-zero-false", {"zero": False}),
]
_TYPE_SWITCH_BASE_IDS = [name for name, _scale in _TYPE_SWITCH_BASE_SCALES]

_TYPE_SWITCH_MARKS: list[tuple[str, str]] = [("bar", "cat"), ("point", "cat")]
_TYPE_SWITCH_MARK_IDS = [name for name, _x in _TYPE_SWITCH_MARKS]


def _chart_with_y_scale(mark_name: str, scale: object) -> "fm.Chart":
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})
    chart = fm.Chart(df).encode(x="cat", y=fm.Y("val", scale=scale))
    return chart.mark_bar() if mark_name == "bar" else chart.mark_point()


@pytest.mark.parametrize("mark_name, _x", _TYPE_SWITCH_MARKS, ids=_TYPE_SWITCH_MARK_IDS)
@pytest.mark.parametrize(
    "scale_name, base_scale", _TYPE_SWITCH_BASE_SCALES, ids=_TYPE_SWITCH_BASE_IDS
)
def test_override_type_switch_linear_to_log_drops_stale_zero_key(
    mark_name: str, _x: str, scale_name: str, base_scale: object
) -> None:
    """``.override(y_scale_type="log")`` over an explicit linear-ish base
    scale (class, typed dict, or untyped dict — all three carry/normalize
    to a real ``"zero"`` key) must not carry ``zero`` onto ``log``, on both
    ``bar`` and ``point`` (the bug is not bar-specific: the zero-anchor
    guard only ever protects the bar-injected key, not a user-authored
    ``LinearScale()``'s own ``zero`` field).
    """
    chart = _chart_with_y_scale(mark_name, base_scale).override(y_scale_type="log")
    scale = json.loads(chart.to_spec().to_json())["encoding"]["y"]["scale"]
    assert scale["type"] == "log"
    assert "zero" not in scale
    assert "<svg" in chart.to_svg()


@pytest.mark.parametrize("mark_name, _x", _TYPE_SWITCH_MARKS, ids=_TYPE_SWITCH_MARK_IDS)
def test_override_type_switch_log_to_linear_drops_stale_base_key(mark_name: str, _x: str) -> None:
    """The reverse-direction switch: ``fm.LogScale()`` (carries ``base``)
    overridden to ``linear`` must drop ``base`` — the sibling case to the
    previous test, proving the drop is not accidentally one-directional.
    """
    chart = _chart_with_y_scale(mark_name, fm.LogScale()).override(y_scale_type="linear")
    scale = json.loads(chart.to_spec().to_json())["encoding"]["y"]["scale"]
    assert scale["type"] == "linear"
    assert "base" not in scale
    assert "<svg" in chart.to_svg()


def test_override_same_type_control_keeps_variant_specific_key() -> None:
    """Control: when the override does NOT change the effective type (only
    ``y_scale_domain`` is overridden, ``fm.LogScale(base=2.0)``'s own
    ``type`` is untouched), the merge must NOT filter — ``base`` must
    survive. Proves the type-aware filter fires only on an actual type
    change, not on every override.
    """
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "val": [10.0, 20.0, 15.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x", y=fm.Y("val", scale=fm.LogScale(base=2.0)))
        .override(y_scale_domain=[1, 100])
    )
    scale = json.loads(chart.to_spec().to_json())["encoding"]["y"]["scale"]
    assert scale["type"] == "log"
    assert scale["base"] == 2.0
    assert scale["domain"] == [1.0, 100.0]
    assert "<svg" in chart.to_svg()


# ---------------------------------------------------------------------------
# 2. Round 4: filter against the TARGET type's own accepted-key set
# (ferrum._core.scale_accepted_keys), not a hand-mirrored intersection —
# survival of a shared key, and mixed continuous/non-continuous switches
# ---------------------------------------------------------------------------

_SURVIVAL_TYPE_SWITCHES: list[tuple[str, str]] = [
    ("linear", "log"),
    ("time", "utc"),
]
_SURVIVAL_IDS = [f"{old}->{new}" for old, new in _SURVIVAL_TYPE_SWITCHES]


@pytest.mark.parametrize("old_type, new_type", _SURVIVAL_TYPE_SWITCHES, ids=_SURVIVAL_IDS)
def test_override_type_switch_keeps_a_key_the_new_type_also_accepts(
    old_type: str, new_type: str
) -> None:
    """A key both the old and new type accept (``nice``, on every continuous
    type except ``pow``/``sqrt``) must SURVIVE a type-changing
    ``Chart.override(<channel>_scale_type=...)``, not just a variant-specific
    key be dropped. An earlier revision filtered against a Python mirror of
    the INTERSECTION of the continuous types' keys, which dropped ``nice`` on
    every one of these switches (it is per-variant, not shared by all seven),
    silently moving the axis relative to the explicit spelling — this is the
    equality/inequality
    shape already used at
    ``test_bar_override_y_scale_domain_matches_explicit_zero_false_not_zero_true``
    (section 3 below), applied to ``nice`` instead of ``domain``/``zero``.
    """
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})

    def _svg(scale: dict) -> str:
        return fm.Chart(df).mark_point().encode(x=fm.X("x", scale=scale), y="y").to_svg()

    svg_override = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("x", scale={"type": old_type, "nice": True}), y="y")
        .override(x_scale_type=new_type)
        .to_svg()
    )
    svg_explicit_nice_true = _svg({"type": new_type, "nice": True})
    svg_explicit_nice_omitted = _svg({"type": new_type})
    assert svg_override == svg_explicit_nice_true, (
        f"nice=True must survive the {old_type}->{new_type} override switch"
    )
    assert svg_override != svg_explicit_nice_omitted, (
        "the two explicit spellings must render differently, or this pair is vacuous"
    )


_MIXED_FAMILY_SWITCHES: list[tuple[str, object, str, str]] = [
    # (id, base_scale, override_type, stale_key_expected_absent)
    ("point_padding->band", fm.PointScale(padding=0.5), "band", "reverse"),
    ("band_paddingInner->point", fm.BandScale(padding_inner=0.4), "point", "paddingInner"),
    ("band_paddingInner->log", fm.BandScale(padding_inner=0.4), "log", "paddingInner"),
    ("log_base->band", fm.LogScale(base=2.0), "band", "base"),
    ("point_align->linear", fm.PointScale(align=0.0), "linear", "align"),
    ("linear->band", fm.LinearScale(), "band", "zero"),
]
_MIXED_FAMILY_IDS = [case_id for case_id, *_rest in _MIXED_FAMILY_SWITCHES]


@pytest.mark.parametrize(
    "base_scale, override_type, stale_key",
    [c[1:] for c in _MIXED_FAMILY_SWITCHES],
    ids=_MIXED_FAMILY_IDS,
)
def test_override_mixed_family_type_switch_renders_without_stale_key(
    base_scale: object, override_type: str, stale_key: str
) -> None:
    """A type-changing override across the continuous/non-continuous
    boundary (``point``/``band``/``log``/``linear`` mixed pairwise) used to
    hard-raise, since round 3's filter only fired when BOTH the old and new
    type were in the 7-member continuous family. Every one of these shapes
    rendered pre-gate (the carried key is not a field on the target
    variant, so serde's flatten silently dropped it), so filtering by the
    target type's own accepted-key set — which fires on any type change,
    not just a continuous-to-continuous one — must restore that.
    """
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "val": [10.0, 20.0, 15.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("x", scale=base_scale), y="val")
        .override(x_scale_type=override_type)
    )
    scale = json.loads(chart.to_spec().to_json())["encoding"]["x"]["scale"]
    assert scale["type"] == override_type
    assert stale_key not in scale
    assert "<svg" in chart.to_svg()


def test_override_mixed_family_type_switch_on_color_channel_renders() -> None:
    """The non-positional-channel sibling of the mixed-family switches
    above: ``Color(scale=fm.LinearScale())`` (carries ``zero``/``clamp``/
    ``nice``, none of which ``sequential`` has) overridden to
    ``color_scale_type="sequential"`` used to raise
    ``unknown key 'clamp' for type 'sequential'``.
    """
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "val": [10.0, 20.0, 15.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x", y="val", color=fm.Color("val", scale=fm.LinearScale()))
        .override(color_scale_type="sequential")
    )
    scale = json.loads(chart.to_spec().to_json())["encoding"]["color"]["scale"]
    assert scale["type"] == "sequential"
    assert "clamp" not in scale
    assert "zero" not in scale
    assert "<svg" in chart.to_svg()


# ---------------------------------------------------------------------------
# 3. Adjudicated-kept: bar + override(y_scale_domain=...) no longer widens
# ---------------------------------------------------------------------------


def test_bar_override_y_scale_domain_matches_explicit_zero_false_not_zero_true() -> None:
    """DELIBERATE BEHAVIOR CHANGE (round 2's override-merge/zero-anchor
    reorder): before the reorder, a bar's
    ``.override(y_scale_domain=[...])`` got the zero-anchor injected FIRST
    onto an empty scale (``{"type": "linear", "zero": True}``), then had
    ``domain`` merged on top, producing ``{"type": "linear", "zero": True,
    "domain": [lo, hi]}`` — Rust's positional resolver would then widen the
    resolved axis extent to include zero even though the user explicitly
    supplied a domain. Now the override merge runs first, so ``domain`` is
    present by the time the zero-anchor check runs and the injection is
    correctly suppressed — matching the pre-existing, documented rule that
    a channel-level ``scale={"domain": ...}`` already suppresses the
    zero-anchor (``tests/test_bar_zero.py::test_bar_explicit_domain_no_zero``).
    Asserts the sharper pair, not just a tick-label check: equality with
    the explicit ``zero: False`` spelling AND inequality with the
    explicit ``zero: True`` spelling, over the exact repro pair below.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [50.0, 120.0, 90.0]})
    svg_override = (
        fm.Chart(df).mark_bar().encode(x="cat", y="val").override(y_scale_domain=[50, 200]).to_svg()
    )
    svg_zero_false = (
        fm.Chart(df)
        .mark_bar()
        .encode(x="cat", y=fm.Y("val", scale={"domain": [50, 200], "zero": False}))
        .to_svg()
    )
    svg_zero_true = (
        fm.Chart(df)
        .mark_bar()
        .encode(x="cat", y=fm.Y("val", scale={"domain": [50, 200], "zero": True}))
        .to_svg()
    )
    assert svg_override == svg_zero_false, (
        "override(y_scale_domain=...) must match the explicit zero=False spelling"
    )
    assert svg_override != svg_zero_true, (
        "override(y_scale_domain=...) must NOT widen to the old zero=True behavior"
    )


# ---------------------------------------------------------------------------
# 4. Round 5: old-type validity — a key invalid for the OLD type must not
# be laundered through a type-changing override
# ---------------------------------------------------------------------------


def test_override_type_switch_typo_base_key_still_refuses() -> None:
    """Case B: ``clammp`` (a typo of
    ``clamp``) is not a real field of ``linear`` OR ``log``, so a
    type-changing override must refuse it exactly as the no-override
    control (case A) does, rather than letting it vanish silently the way
    it did before round 5's old-type-validity check — round 4's filter only
    dropped keys the target type rejects that the SOURCE type accepted, so
    a source-invalid key it happened to also reject was passed through
    unfiltered and then silently dropped by the merge's own ``**`` spread
    once it reached a type with no such field.

    Round 6 replaced the hand-synthesized refusal message with a probe of
    the real gate (``ferrum._core.EncodingSpec``), so the override spelling
    and the no-override control are now literally the same call on the
    same dict — asserted here as exact message equality, not a shared
    substring.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})
    scale = {"type": "linear", "clammp": True}

    with pytest.raises(ValueError, match="unknown key 'clammp' for type 'linear'") as no_override:
        fm.Chart(df).mark_bar().encode(x="cat", y=fm.Y("val", scale=scale)).to_svg()

    with pytest.raises(ValueError) as with_override:
        fm.Chart(df).mark_bar().encode(x="cat", y=fm.Y("val", scale=scale)).override(
            y_scale_type="log"
        ).to_svg()

    assert str(with_override.value) == str(no_override.value), (
        "the override spelling must raise the identical message as the no-override control"
    )


def test_override_type_switch_key_invalid_for_old_type_is_not_promoted() -> None:
    """Case C: ``nice`` is not
    a ``band`` field, so ``{"type": "band", "nice": True}`` refuses with no
    override at all (case A control, asserted first). ``nice`` IS a real
    ``log`` field, though, so before round 5's old-type-validity check, a
    ``.override(x_scale_type="log")`` switch let it pass through unfiltered
    (round 4's filter only drops keys against the SOURCE type's own accepted
    set for keys the source type actually has) and ``log``'s own field
    silently absorbed it — worse than the pre-gate baseline, where
    ``band``'s flatten just dropped an inapplicable key. Both the
    no-override and the override spelling must refuse identically.

    Round 6 note: this is now a probe of the real gate rather than a
    synthesized message (see previous test's docstring), so equality is
    asserted directly rather than a shared substring.
    """
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "val": [10.0, 20.0, 15.0]})
    scale = {"type": "band", "nice": True}

    with pytest.raises(ValueError, match="unknown key 'nice' for type 'band'") as no_override:
        fm.Chart(df).mark_point().encode(x=fm.X("x", scale=scale), y="val").to_svg()

    with pytest.raises(ValueError) as with_override:
        fm.Chart(df).mark_point().encode(x=fm.X("x", scale=scale), y="val").override(
            x_scale_type="log"
        ).to_svg()

    assert str(with_override.value) == str(no_override.value), (
        "the override spelling must raise the identical message as the no-override control"
    )


# ---------------------------------------------------------------------------
# 5. Round 5: non-string override-type value must not leak a PyO3
# argument-coercion error through an explicit base scale
#
# Shared with section 6's old-type parametrization below (round 7): one
# list of malformed "type" values feeds both the override slot (this
# section) and the base slot (section 6), so the two positions cannot
# drift apart the way round 6's two
# independently hand-typed lists did (``float:1.5`` was override-only-
# absent, ``none:None`` was missing from both).
# ---------------------------------------------------------------------------

_NONSTRING_TYPE_VALUES: list[tuple[str, object]] = [
    ("int:5", 5),
    ("float:1.5", 1.5),
    ("bytes:log", b"log"),
    ("class:LogScale", fm.LogScale()),
    ("none:None", None),
]
_NONSTRING_IDS = [name for name, _value in _NONSTRING_TYPE_VALUES]

#: The two OVERRIDE-value tests directly below exercise genuinely malformed
#: ``type`` override values -- ones that must still refuse somehow. ``None``
#: moved out of that subset under #157: a ``None`` override leaf is no
#: override (the ``ENCODING_SCALE`` payload branch drops it before the merge
#: ever sees it), so ``.override(y_scale_type=None)`` is a deliberate no-op,
#: not a malformed value, and must NOT raise -- pinned separately below by
#: ``test_override_none_type_is_no_op_keeps_the_base_scale_type``. It stays
#: in the shared ``_NONSTRING_TYPE_VALUES`` list itself: section 6 reuses the
#: full list for the BASE scale's own ``"type"`` key (a value embedded in
#: ``scale={"type": ..., ...}`` via ``.encode()``, not an override leaf),
#: which the None-override-is-no-op rule does not touch.
_MALFORMED_TYPE_VALUES = [
    (name, value) for name, value in _NONSTRING_TYPE_VALUES if value is not None
]
_MALFORMED_TYPE_IDS = [name for name, _value in _MALFORMED_TYPE_VALUES]


@pytest.mark.parametrize(
    "new_type_value", [value for _name, value in _MALFORMED_TYPE_VALUES], ids=_MALFORMED_TYPE_IDS
)
def test_override_nonstring_type_over_explicit_base_does_not_leak_pyo3_argument_error(
    new_type_value: object,
) -> None:
    """A non-``str`` ``<channel>_scale_type=`` override value (a user
    reaching for ``y_scale=`` and mistyping ``y_scale_type=``) used to
    surface PyO3's own argument-coercion error naming an internal
    parameter the user never wrote (``argument 'scale_type': 'LogScale'
    object is not an instance of 'str'``) whenever the channel carried an
    explicit base scale — because ``merge_encoding_scale`` handed the raw
    value straight to ``ferrum._core.scale_accepted_keys`` before any gate
    ever saw it. It must not, regardless of which downstream error the
    malformed value ultimately produces once it reaches the real gate.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})
    with pytest.raises(Exception) as excinfo:
        fm.Chart(df).mark_bar().encode(x="cat", y=fm.Y("val", scale=fm.LinearScale())).override(
            y_scale_type=new_type_value
        ).to_svg()
    assert "argument 'scale_type'" not in str(excinfo.value)


@pytest.mark.parametrize(
    "new_type_value", [value for _name, value in _MALFORMED_TYPE_VALUES], ids=_MALFORMED_TYPE_IDS
)
def test_override_nonstring_type_same_error_class_with_and_without_base_scale(
    new_type_value: object,
) -> None:
    """The same malformed ``<channel>_scale_type=`` value, with vs. without
    an explicit base scale on the channel, must land in the SAME downstream
    error path — not diverge into the PyO3 argument-coercion error on one
    side (explicit base) and the real gate's own message on the other
    (no base), which is exactly the class/no-base inconsistency this
    round's ``isinstance`` guard closes.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})

    with pytest.raises(Exception) as with_base:
        fm.Chart(df).mark_bar().encode(x="cat", y=fm.Y("val", scale=fm.LinearScale())).override(
            y_scale_type=new_type_value
        ).to_svg()
    with pytest.raises(Exception) as without_base:
        fm.Chart(df).mark_bar().encode(x="cat", y="val").override(
            y_scale_type=new_type_value
        ).to_svg()

    assert type(with_base.value) is type(without_base.value), (
        "with-base and no-base spellings must raise the same exception type"
    )
    assert str(with_base.value) == str(without_base.value), (
        "with-base and no-base spellings must raise the same message"
    )


def test_override_none_type_is_no_op_keeps_the_base_scale_type() -> None:
    """``.override(y_scale_type=None)`` is no override (#157): a None
    scale-override leaf is dropped before the merge ever sees it, so the
    channel's own scale type stands -- unlike every other non-string type
    value above (section 5), which still refuses.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})
    base = (
        fm.Chart(df)
        .mark_bar()
        .encode(x="cat", y=fm.Y("val", scale=fm.LogScale(domain=[1.0, 100.0])))
    )
    overridden = base.override(y_scale_type=None)
    assert overridden.to_svg() == base.to_svg()


def test_override_bogus_type_over_explicit_base_surfaces_gate_unknown_variant() -> None:
    """Round 4's ``except ValueError`` fallback branch around
    ``scale_accepted_keys`` — reached when the override's type string names
    no known ``ScaleSpec`` variant — was executed by zero tests before this
    round. An unknown override-type string over an EXPLICIT base scale must
    still fall through to the real gate's own "unknown variant" refusal,
    not get swallowed or mis-shaped by the merge helper.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})
    with pytest.raises(ValueError, match=r"unknown variant `bogus`"):
        fm.Chart(df).mark_bar().encode(x="cat", y=fm.Y("val", scale=fm.LogScale())).override(
            y_scale_type="bogus"
        ).to_svg()


# ---------------------------------------------------------------------------
# 6. Round 6: old-type validity is now a probe of the REAL gate (not a
# hand-synthesized message) -- closes the old-type non-string leak (round
# 5's isinstance guard only protected the NEW-type call one branch below)
# and the unknown-old-type-tag fall-through (round 5's docstring claimed
# it was safe; it wasn't, since a type-changing override replaces the tag
# before the gate could ever see the base's own claim). Round 7 (below)
# folded ``None`` into this parametrization -- round 6's entry guard
# (``old_type is None``) skipped the probe entirely for an EXPLICIT
# ``{"type": None}`` base, which is the one shape this section's own
# universality claim did not yet cover; see section 7's narrative note.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "old_type_value", [value for _name, value in _NONSTRING_TYPE_VALUES], ids=_NONSTRING_IDS
)
def test_override_type_switch_nonstring_old_type_same_error_as_no_override_control(
    old_type_value: object,
) -> None:
    """Round 5's ``isinstance(new_type, str)`` guard (section 5) covered only
    the OVERRIDE's own type value. The sibling call one branch earlier --
    validating ``base_scale``'s OWN ``"type"`` -- had no such guard, so a
    non-string base ``"type"`` behind a type-changing override raised
    PyO3's raw ``TypeError: argument 'scale_type': ...`` (naming an
    internal parameter the user never wrote) instead of whatever the real
    gate raises for that exact base scale standalone.

    Round 6 replaced the whole old-type-validity check with a probe of the
    real gate (``ferrum._core.EncodingSpec``) that never hands an
    unvalidated value to the ``&str``-typed ``scale_accepted_keys`` at
    all, so the override spelling and the no-override control are now the
    identical call on the identical dict for every case the probe actually
    reached. The claim was NOT yet true for ``old_type_value is None``,
    though: round 6's entry guard (``old_type is None``) short-circuited
    BEFORE the probe for that one value, skipping it entirely rather than
    validating it -- round 7 replaced the guard with ``"type" not in
    base_scale`` (a base scale with an explicit ``"type": None`` key IS a
    real, present claim, distinct from a channel with no ``scale=`` at
    all), so the probe now runs unconditionally on every malformed
    ``old_type_value`` including ``None`` and the claim is true: this can
    no longer diverge by construction, for ANY malformed ``"type"`` value,
    not just the ones enumerated here.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})

    def render(*, override: bool):
        scale = {"type": old_type_value, "domain": [0.0, 30.0]}
        c = fm.Chart(df).mark_bar().encode(x="cat", y=fm.Y("val", scale=scale))
        if override:
            c = c.override(y_scale_type="log")
        return c.to_svg()

    with pytest.raises(Exception) as no_override:
        render(override=False)
    with pytest.raises(Exception) as with_override:
        render(override=True)

    assert "argument 'scale_type'" not in str(with_override.value)
    assert type(with_override.value) is type(no_override.value), (
        "with-override and no-override spellings must raise the same exception type"
    )
    assert str(with_override.value) == str(no_override.value), (
        "with-override and no-override spellings must raise the same message"
    )


def test_override_type_switch_unknown_old_type_tag_refuses_matching_control() -> None:
    """A base scale whose OWN ``"type"`` names no known ``ScaleSpec``
    variant (a typo like ``"linearr"``) must refuse a type-changing
    override exactly as it refuses standalone -- NOT fall through
    unfiltered. Round 5's docstring claimed the fall-through was safe
    "since the gate will refuse the base type's own tag downstream
    regardless"; that was false, because the override supplies its OWN
    ``"type"`` and the base's invalid tag never reaches the wire on that
    spelling -- the user's actual mistake (the typo'd base type) would
    never be named.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})
    scale = {"type": "linearr", "zero": True}

    with pytest.raises(ValueError, match=r"unknown variant `linearr`") as no_override:
        fm.Chart(df).mark_bar().encode(x="cat", y=fm.Y("val", scale=scale)).to_svg()
    with pytest.raises(ValueError) as with_override:
        fm.Chart(df).mark_bar().encode(x="cat", y=fm.Y("val", scale=scale)).override(
            y_scale_type="log"
        ).to_svg()

    assert str(with_override.value) == str(no_override.value), (
        "an unknown old-type tag must refuse identically with and without the override, "
        "not be silently replaced by the override before the gate ever sees it"
    )


# ---------------------------------------------------------------------------
# 7. Round 7: producer-composition regression coverage for the narrowed
# old-type-validity probe -- a key both the old and new type accept, whose
# VALUE only legal-typechecks under the NEW type, must survive a
# type-changing override rather than being value-refused under the type
# the switch is actively replacing
# ---------------------------------------------------------------------------

_UNTYPED_STRING_DOMAIN_OVERRIDE_TARGETS = ["band", "point", "ordinal"]


@pytest.mark.parametrize("new_type", _UNTYPED_STRING_DOMAIN_OVERRIDE_TARGETS)
def test_override_type_switch_untyped_string_domain_survives_switch_to_categorical(
    new_type: str,
) -> None:
    """An UNTYPED raw-dict scale (``_scale_to_dict``'s ``"linear"`` default,
    stamped by ``_emit_scale`` before ``merge_encoding_scale`` ever sees
    it) whose ``domain`` is a string list -- only legal under a categorical
    scale, not the stamped-default ``linear`` -- used to refuse under round
    6's whole-dict old-type probe (``invalid type: string "a", expected
    f64``) even though the final, override-merged scale is a perfectly
    legal ``band``/``point``/``ordinal`` domain. ``domain`` is a key both
    ``linear`` and the target type accept, so round 7's narrower probe
    (tag-only, plus a key-membership probe over only the keys ``linear``
    does NOT recognize) never value-validates it under ``linear`` at all --
    its value reaches the target type's own downstream validation on the
    final merged dict, unchanged.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("cat", scale={"domain": ["a", "b", "c"]}), y="val")
        .override(x_scale_type=new_type)
    )
    scale = json.loads(chart.to_spec().to_json())["encoding"]["x"]["scale"]
    assert scale["type"] == new_type
    assert scale["domain"] == ["a", "b", "c"]
    assert "<svg" in chart.to_svg()


def test_override_type_switch_untyped_iso_domain_survives_switch_to_time() -> None:
    """The temporal sibling of the case above: an UNTYPED raw-dict scale
    (stamped ``"linear"``) whose ``domain`` is a list of ISO date strings
    used to refuse the identical way under round 6's whole-dict probe (a
    string is not a valid ``linear`` domain element). The override-merged
    scale is a legal ``time`` scale, and asserting the domain converted to
    the exact epoch-ms pair also proves ``_scale_to_dict``'s temporal
    conversion runs on the POST-override effective type -- the switch is
    not laundering the ISO strings past the conversion the way it used to
    launder them past validation.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(
            x="cat",
            y=fm.Y("val", scale={"domain": ["2021-01-01", "2021-12-31"]}),
        )
        .override(y_scale_type="time")
    )
    scale = json.loads(chart.to_spec().to_json())["encoding"]["y"]["scale"]
    assert scale["type"] == "time"
    assert scale["domain"] == [1_609_459_200_000.0, 1_640_908_800_000.0]
    assert "<svg" in chart.to_svg()


def test_override_type_switch_explicit_linear_string_domain_survives_switch_to_band() -> None:
    """The explicit-``"type"`` sibling of the untyped case above:
    ``{"type": "linear", "domain": ["a", "b"]}`` -- the user's own claimed
    type, not a ``_scale_to_dict`` default -- switched to ``band`` must
    also survive; the fix is not conditioned on the type having been
    stamped by ``_scale_to_dict`` rather than authored directly.
    """
    df = pl.DataFrame({"cat": ["a", "b"], "val": [10.0, 20.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("cat", scale={"type": "linear", "domain": ["a", "b"]}), y="val")
        .override(x_scale_type="band")
    )
    scale = json.loads(chart.to_spec().to_json())["encoding"]["x"]["scale"]
    assert scale["type"] == "band"
    assert scale["domain"] == ["a", "b"]
    assert "<svg" in chart.to_svg()


# ---------------------------------------------------------------------------
# 8. Round 8: the DROPPED-key bucket -- a key accepted_old recognizes but
# accepted_new doesn't, about to be silently dropped by the type-switch
# filter -- must be validated under old_type before it's dropped, not
# laundered into a silent render. This is the third bucket of the
# partition section 7's docstring names: unknown_under_old (refused),
# survivors (validated downstream under new_type), and now dropped
# (validated here, under old_type, the same shape as unknown_under_old).
# ---------------------------------------------------------------------------

_DROPPED_BUCKET_CASES = [
    pytest.param({"type": "linear", "zero": "yes"}, "log", id="linear_zero-to-log"),
    pytest.param({"type": "linear", "zero": "yes"}, "band", id="linear_zero-to-band"),
    pytest.param({"type": "log", "base": "ten"}, "band", id="log_base-to-band"),
    pytest.param({"type": "band", "align": "left"}, "linear", id="band_align-to-linear"),
    # A padding-family value with no meaning under the new type is dropped
    # the same way as any other dropped key, and must still refuse under
    # the OLD type's value bounds before it is discarded -- a continuous
    # span fraction switched to a discrete type, and a discrete inner gap
    # switched to a type with no
    # end-padding slot (ordinal has none; point's own end padding, named
    # ``padding`` too, is a different field the switch does not read from
    # here).
    pytest.param({"type": "linear", "padding": "wide"}, "band", id="linear_padding-to-band"),
    pytest.param({"type": "ordinal", "padding": "wide"}, "point", id="ordinal_padding-to-point"),
]


@pytest.mark.parametrize("base_scale,new_type", _DROPPED_BUCKET_CASES)
def test_override_type_switch_dropped_key_bad_value_refuses_matching_control(
    base_scale: dict, new_type: str
) -> None:
    """A key ``old_type`` accepts but ``new_type`` doesn't is dropped by
    the type-switch filter before ``new_type``'s own downstream gate ever
    sees it -- and ``unknown_under_old``'s probe never checks it either,
    since it IS a member of ``old_type``'s accepted-key set. Before this
    round, nothing validated its VALUE at all: ``{"type": "linear", "zero":
    "yes"}`` (an invalid ``bool``) + ``.override(y_scale_type="log")``
    rendered silently, even though the identical base scale refuses
    standalone. The dropped-bucket probe closes this: the override
    spelling must refuse with the exact same exception type and message as
    the no-override control.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})

    def render(*, override: bool):
        c = fm.Chart(df).mark_bar().encode(x="cat", y=fm.Y("val", scale=base_scale))
        if override:
            c = c.override(y_scale_type=new_type)
        return c.to_svg()

    with pytest.raises(Exception) as no_override:
        render(override=False)
    with pytest.raises(Exception) as with_override:
        render(override=True)

    assert type(with_override.value) is type(no_override.value), (
        "a dropped key's bad value must refuse with the same exception type as the "
        "no-override control, not render silently once the type switch drops it"
    )
    assert str(with_override.value) == str(no_override.value), (
        "a dropped key's bad value must refuse with the same message as the no-override control"
    )


def test_override_type_switch_surviving_key_untouched_by_dropped_bucket_probe() -> None:
    """Control for the fix above: a key BOTH the old and new type accept
    (``domain``, legal on both ``linear`` and ``band``) is a survivor, not
    a member of ``drop = accepted_old - accepted_new`` -- the new
    dropped-bucket probe must never see it, so it keeps reaching
    ``new_type``'s own downstream validation untouched, exactly as section 7
    already pins. A regression that widened the dropped-bucket probe to
    cover survivors too would refuse this case; it must still render.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("cat", scale={"type": "linear", "domain": ["a", "b", "c"]}), y="val")
        .override(x_scale_type="band")
    )
    scale = json.loads(chart.to_spec().to_json())["encoding"]["x"]["scale"]
    assert scale["type"] == "band"
    assert scale["domain"] == ["a", "b", "c"]
    assert "<svg" in chart.to_svg()


# ---------------------------------------------------------------------------
# 9. The padding-family meaning table a type-changing override reads
# (``ferrum._core.translate_scale_switch``), on every pair that table
# covers. ``merge_encoding_scale`` is only the consumer -- the pairs
# themselves are owned in Rust
# (``scale/discrete.rs``'s ``PaddingMeaning``) and probed directly in the
# Rust test suite; these assert the Python-visible effect end to end,
# through ``Chart.to_spec()``.
# ---------------------------------------------------------------------------


def _scale_dict(chart: "fm.Chart", channel: str = "x") -> dict:
    return json.loads(chart.to_spec().to_json())["encoding"][channel]["scale"]


_PADDING_PAIR_MATRIX = [
    # (id, field, base_scale dict, new_type, present {key: value}, absent [keys])
    # `field` is the numeric column "val" when `new_type` is continuous
    # (needs a numeric domain to render at all) and the categorical column
    # "cat" otherwise.
    pytest.param(
        "cat",
        {"type": "ordinal", "padding": 0.3},
        "band",
        {"paddingInner": 0.3},
        ["padding", "paddingOuter"],
        id="ordinal_padding-to-band",
    ),
    pytest.param(
        "cat",
        {"type": "ordinal", "padding": 0.3},
        "point",
        {},
        ["padding"],
        id="ordinal_padding-to-point",
    ),
    pytest.param(
        "cat",
        {"type": "point", "padding": 0.4, "align": 0.0},
        "band",
        {"paddingOuter": 0.4, "align": 0.0},
        ["padding", "paddingInner"],
        id="point_padding_align-to-band",
    ),
    pytest.param(
        "cat",
        {"type": "band", "paddingInner": 0.3, "paddingOuter": 0.2, "align": 0.1},
        "ordinal",
        # BandPaddingKey::INNER_PRECEDENCE: paddingInner wins over padding.
        {"padding": 0.3},
        ["paddingInner", "paddingOuter", "align"],
        id="band_inner_outer_align-to-ordinal",
    ),
    pytest.param(
        "cat",
        {"type": "band", "paddingInner": 0.3, "paddingOuter": 0.2, "align": 0.1},
        "point",
        # BandPaddingKey::OUTER_PRECEDENCE: paddingOuter wins over padding.
        {"padding": 0.2, "align": 0.1},
        ["paddingInner", "paddingOuter"],
        id="band_inner_outer_align-to-point",
    ),
    pytest.param(
        "cat",
        {"type": "band", "padding": 0.4},
        "ordinal",
        {"padding": 0.4},
        [],
        id="band_shorthand_padding-to-ordinal",
    ),
    pytest.param(
        "cat",
        {"type": "band", "padding": 0.4},
        "point",
        {"padding": 0.4},
        ["paddingInner", "paddingOuter"],
        id="band_shorthand_padding-to-point",
    ),
    pytest.param(
        "val",
        {"type": "linear", "padding": 0.1},
        "band",
        {},
        ["padding"],
        id="linear_padding-to-band_continuous_to_discrete_drops",
    ),
    pytest.param(
        "val",
        {"type": "band", "paddingInner": 0.3, "paddingOuter": 0.2},
        "linear",
        {},
        ["padding", "paddingInner", "paddingOuter"],
        id="band-to-linear_discrete_to_continuous_drops",
    ),
]


@pytest.mark.parametrize("field, base_scale, new_type, present, absent", _PADDING_PAIR_MATRIX)
def test_padding_pair_matrix_translates_by_meaning(
    field: str, base_scale: dict, new_type: str, present: dict, absent: list
) -> None:
    """``.override(x_scale_type=...)`` over an explicit padding-family base
    carries each pair exactly as the padding-family meaning table states,
    through the real render path (not a direct call to
    ``translate_scale_switch``), AND
    renders the identical SVG as the equivalent TARGET scale declared
    directly with ``present`` as its own padding-family kwargs.

    The comparison chart declares the SAME keys (``present``) directly, so
    this proves the MERGE produced the right translated values -- a merge
    bug that drops or rekeys a value cannot pass it, since the comparison
    chart's dict would then differ from the switched chart's translated
    one. It does not independently prove the RENDERER honors those keys: a
    renderer that ignored ``paddingInner``/``paddingOuter``/``align``
    entirely would ignore them on both sides of this comparison alike and
    still match. Renderer coverage of those keys is assumed to live in
    other test files (e.g. ``tests/test_band_padding_inherit.py``'s
    measured bar-width assertions).
    """
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X(field, scale=base_scale), y="val")
        .override(x_scale_type=new_type)
    )
    scale = _scale_dict(chart)
    assert scale["type"] == new_type
    for key, value in present.items():
        assert scale[key] == value, f"expected {key}={value}, got {scale.get(key)!r}"
    for key in absent:
        assert key not in scale, f"expected {key} to be dropped, found {scale[key]!r}"
    svg = chart.to_svg()
    assert "<svg" in svg
    equivalent = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X(field, scale={"type": new_type, **present}), y="val")
    )
    assert svg == equivalent.to_svg(), (
        "the switched chart's rendered geometry must match the target scale "
        "declared directly with the SAME padding-family values"
    )


def test_ordinal_scale_plain_to_band_byte_identical_to_unpadded_band() -> None:
    """``fm.OrdinalScale(domain=[...])``'s wire-default ``padding`` of 0.0
    (the class door always serializes it, unlike the bare ``{"type":
    "ordinal"}`` dict door, which omits a key nobody set) carries to
    ``paddingInner`` 0.0, which renders byte-identical to an explicit plain
    (unpadded) band. Uses the class, not the dict door: the dict door has
    no ``padding`` key at all, so it exercises nothing to carry.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})
    switched = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("cat", scale=fm.OrdinalScale(domain=["a", "b", "c"])), y="val")
        .override(x_scale_type="band")
    )
    scale = _scale_dict(switched)
    assert scale["paddingInner"] == 0.0
    plain_band = fm.Chart(df).mark_point().encode(x=fm.X("cat", scale={"type": "band"}), y="val")
    assert switched.to_svg() == plain_band.to_svg()


def test_point_scale_unbounded_padding_carries_to_band_outer_and_renders() -> None:
    """``PointScale(padding=2)``: point's own bound is "finite >= 0", so 2
    is legal there. Carried to band's ``paddingOuter`` (also unbounded
    above), it is legal there too and must render, not refuse it. A merge
    that only dropped keys the new type doesn't accept would instead
    carry this value onto band's own ``padding`` field (the same key
    name, but a different meaning: band's both-sides shorthand, bounded
    to ``[0, 1]``, not point's unbounded outer-only inset) and refuse it
    there as an out-of-range band padding the user never set directly.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("cat", scale=fm.PointScale(padding=2.0)), y="val")
        .override(x_scale_type="band")
    )
    scale = _scale_dict(chart)
    assert scale["paddingOuter"] == 2.0
    assert "padding" not in scale
    svg = chart.to_svg()
    assert "<svg" in svg
    equivalent = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("cat", scale=fm.BandScale(padding_outer=2.0)), y="val")
    )
    assert svg == equivalent.to_svg(), (
        "must render identically to BandScale(padding_outer=2.0) declared directly"
    )


def test_point_scale_unbounded_padding_drops_on_switch_to_ordinal() -> None:
    """The sibling of the case above: ``PointScale(padding=2)`` switched to
    ``ordinal`` instead drops the value (ordinal has no end-padding slot),
    rather than carrying an out-of-[0,1]-range value into ordinal's own
    ``padding`` field.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("cat", scale=fm.PointScale(padding=2.0)), y="val")
        .override(x_scale_type="ordinal")
    )
    scale = _scale_dict(chart)
    assert scale["padding"] == 0.0
    svg = chart.to_svg()
    assert "<svg" in svg
    equivalent = fm.Chart(df).mark_point().encode(x=fm.X("cat", scale={"type": "ordinal"}), y="val")
    assert svg == equivalent.to_svg(), "must render identically to plain {'type': 'ordinal'}"


def test_ordinal_scale_plain_to_point_renders_at_the_default() -> None:
    """``fm.OrdinalScale(domain=[...])``'s own ``padding`` (0.0, always
    serialized by the class door -- unlike the bare ``{"type": "ordinal"}``
    dict door, which has no ``padding`` key to carry or drop in the first
    place) has no meaning under ``point`` and is dropped, rather than
    reinterpreted as point's end padding -- so a switch to ``point`` renders
    at point's own 0.5 default, not at ordinal's 0.0 (which would put
    points flush with the range ends).

    A merge that only drops keys the new type doesn't accept would
    instead keep ``padding`` by coincidence of key-name equality (both
    ordinal and point have a field literally named ``padding``), carrying
    0.0 unchanged and rendering points flush with the range ends, NOT at
    the 0.5 default -- a real behavior difference the bare-dict version of
    this test (with no ``padding`` key present at all) could not exercise.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})
    switched = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("cat", scale=fm.OrdinalScale(domain=["a", "b", "c"])), y="val")
        .override(x_scale_type="point")
    )
    explicit_default = (
        fm.Chart(df).mark_point().encode(x=fm.X("cat", scale=fm.PointScale()), y="val")
    )
    assert "padding" not in _scale_dict(switched)
    assert switched.to_svg() == explicit_default.to_svg()


# ---------------------------------------------------------------------------
# 10. The merge is one function reached from three call sites: the
# chart-level channel build, the per-layer channel-object build, and the
# per-layer bare-string-channel build. A single ``fm.Chart(...).layer(
# fm.Layer(...))`` + chart-level ``.override(x_scale_type=...)`` exercises
# the chart-level channel (this chart's own bare-string ``x="cat"``) and the
# layer-channel-object door (the appended ``Layer``'s ``fm.X(...,
# scale=...)``) in one render; the override cascades to both (``
# _build_layers_list``'s own contract).
# ---------------------------------------------------------------------------


def test_padding_translation_reaches_layer_channel_object_door() -> None:
    """The appended layer's own ``fm.X(..., scale=PointScale(padding=0.5))``
    translates through the SAME ``.override(x_scale_type="band")`` as the
    host chart's bare-string ``x`` channel (which carries no scale at all,
    and so is left untouched by the translation, consistently with the
    plain chart-channel door)."""
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})
    host = (
        fm.Chart(df)
        .mark_bar()
        .encode(x="cat", y="val")
        .layer(
            fm.Layer(
                mark="point",
                encoding={"x": fm.X("cat", scale=fm.PointScale(padding=0.5)), "y": "val"},
            )
        )
        .override(x_scale_type="band")
    )
    spec = json.loads(host.to_spec().to_json())
    layer_scales = [layer["encoding"]["x"]["scale"] for layer in spec["layers"]]
    assert layer_scales[0] == {"type": "band", "align": 0.5}
    assert layer_scales[1]["paddingOuter"] == 0.5
    assert "padding" not in layer_scales[1]
    equivalent = (
        fm.Chart(df)
        .mark_bar()
        .encode(x=fm.X("cat", scale=fm.BandScale()), y="val")
        .layer(
            fm.Layer(
                mark="point",
                encoding={"x": fm.X("cat", scale=fm.BandScale(padding_outer=0.5)), "y": "val"},
            )
        )
    )
    # The two layers' x scales legitimately differ (unpadded band vs.
    # paddingOuter=0.5), which is exactly what this test is pinning, so the
    # layer-1-off-axis warning that difference triggers is expected here,
    # not a signal of a merge bug.
    with pytest.warns(UserWarning, match="its x scale differs"):
        svg = host.to_svg()
    with pytest.warns(UserWarning, match="its x scale differs"):
        equivalent_svg = equivalent.to_svg()
    assert svg == equivalent_svg, (
        "must render identically to the two layers' target scales declared directly"
    )


def test_padding_translation_reaches_layer_string_channel_door() -> None:
    """A layer whose own channel is the bare-string shorthand
    (``encoding={"x": "cat"}``, the per-layer bare-string-channel build)
    carries NO scale of its own -- its merge reaches ``merge_encoding_scale`` as
    ``base_scale={"field": "cat"}``, with no ``"type"`` key to switch FROM,
    so it short-circuits exactly like the chart-level no-scale door
    (section 1's ``test_override_type_switch_on_auto_axis``). The override's
    OWN leaves (``x_scale_type`` and the carried-looking ``x_scale_padding``,
    which is really just an override leaf here, not a translated value)
    still apply, unaffected by the padding-family translation having
    nothing to translate.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})
    host = (
        fm.Chart(df)
        .mark_bar()
        .encode(x="cat", y="val")
        .layer(fm.Layer(mark="line", encoding={"x": "cat", "y": "val"}))
        .override(x_scale_type="band", x_scale_padding_outer=0.5)
    )
    spec = json.loads(host.to_spec().to_json())
    layer_scales = [layer["encoding"]["x"]["scale"] for layer in spec["layers"]]
    assert layer_scales[0]["type"] == "band"
    assert layer_scales[0]["paddingOuter"] == 0.5
    assert layer_scales[1] == layer_scales[0]
    equivalent = (
        fm.Chart(df)
        .mark_bar()
        .encode(x=fm.X("cat", scale=fm.BandScale(padding_outer=0.5)), y="val")
        .layer(
            fm.Layer(
                mark="line",
                encoding={"x": fm.X("cat", scale=fm.BandScale(padding_outer=0.5)), "y": "val"},
            )
        )
    )
    assert host.to_svg() == equivalent.to_svg(), (
        "must render identically to both layers' target scale declared directly"
    )


# ---------------------------------------------------------------------------
# 11. A malformed padding-family value must refuse at ``to_spec()``,
# BEFORE render, whether the switch carries it under a new key name
# (nothing else ever checks a carried value -- it reaches the new type's
# own field under a key the user never wrote) or drops it with only a
# shape check. The translation preserves bounds by construction, so
# checking under the OLD type, in the key the user wrote, covers both.
#
# Two refusal voices exist, by design. A BOUND failure names the scale
# and the key in one place, ``scale: <key> <clause>``. A wrong-type value
# goes through the existing shape probe first and keeps THAT probe's own
# ``"scale: invalid type ..."`` text, which names no key at all. Messages
# are pinned in full, anchored, not matched as a loose substring.
# ---------------------------------------------------------------------------

_MALFORMED_PADDING_TO_SPEC_REFUSALS = [
    pytest.param(
        {"type": "ordinal", "padding": 1.5},
        "band",
        "scale: padding must be in [0, 1]; got 1.5",
        id="ordinal_padding_1.5-to-band_carried_bounds",
    ),
    pytest.param(
        {"type": "point", "padding": -1.0},
        "band",
        "scale: padding must be a finite number >= 0; received -1",
        id="point_padding_neg1-to-band_carried_bounds",
    ),
    pytest.param(
        {"type": "band", "paddingInner": 1.5},
        "ordinal",
        "scale: paddingInner must be a number in [0, 1]; received 1.5",
        id="band_paddingInner_1.5-to-ordinal_carried_bounds",
    ),
    pytest.param(
        {"type": "band", "paddingOuter": -1.0},
        "point",
        "scale: paddingOuter must be a finite number >= 0; received -1",
        id="band_paddingOuter_neg1-to-point_carried_bounds",
    ),
    pytest.param(
        {"type": "band", "align": 5.0},
        "ordinal",
        "scale: align must be a number in [0, 1]; received 5",
        id="band_align_5-to-ordinal_dropped_bounds",
    ),
    pytest.param(
        {"type": "ordinal", "padding": "wide"},
        "band",
        'scale: invalid type: string "wide", expected f64',
        id="ordinal_padding_wide-to-band_carried_wrong_type",
    ),
]


@pytest.mark.parametrize("base_scale, new_type, message", _MALFORMED_PADDING_TO_SPEC_REFUSALS)
def test_malformed_padding_value_refuses_at_to_spec_not_render(
    base_scale: dict, new_type: str, message: str
) -> None:
    """Each malformed value here is caught at ``to_spec()``, before render.
    The five BOUND rows refuse naming both the scale and the key the user
    wrote (``scale: <key> <clause>``); the sixth, wrong-type row refuses
    through the existing shape probe's own ``scale: invalid type ...``
    text, which names no key.

    A merge that never value-checks a padding-family key under its own
    type splits the five bound rows two ways:

    - ``ordinal_padding_1.5-to-band`` and ``point_padding_neg1-to-band``
      (both carried onto band's own ``paddingInner``/``paddingOuter``)
      would still refuse, but only at ``to_svg()``, under BAND's own
      render-time bound check, naming the CARRIED key (``band scale on x:
      paddingInner must be a number in [0, 1]; received 1.5``) -- a key
      the user never wrote.
    - ``band_paddingInner_1.5-to-ordinal``, ``band_paddingOuter_neg1-to-
      point`` and ``band_align_5-to-ordinal`` would render completely
      silently (``to_svg()`` succeeding): ordinal and point have no
      render-time bound check on their own padding dict door.

    ``ordinal_padding_wide-to-band`` is a different kind of row: a
    shape-malformed value like this one is already caught by the
    existing shape probe standalone, with the identical ``scale: invalid
    type: string "wide", expected f64`` text, so no bound check is needed
    to refuse it. This row instead pins that the bound check runs AFTER
    that shape probe, not in its place: if the shape probe were skipped,
    the wrong-type value would go through the bound check's wrapper
    instead and come out KEYED, as ``scale: padding invalid type: string
    "wide", expected f64`` -- which differs from the shape gate's own
    un-keyed text, so the same value would read differently depending on
    whether the switch carried it or dropped it.

    Every row here refuses at ``to_spec()``; the five bound rows name the
    key the user actually wrote, and the wrong-type row keeps the shape
    probe's own un-keyed text.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("cat", scale=base_scale), y="val")
        .override(x_scale_type=new_type)
    )
    # Refused at to_spec() itself -- before render, not merely before
    # to_svg() returns a string. The full message is pinned, anchored, so
    # a stray prefix or suffix (e.g. a missing/extra "scale: " or key name)
    # fails the test instead of slipping past a loose substring match.
    with pytest.raises(ValueError, match=rf"^{re.escape(message)}$"):
        chart.to_spec()
