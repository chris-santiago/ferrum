"""Matrix test: one aggregate vocabulary, one refusal rendering, one wire.

Wave 6B task 9 (`#137` + routed `#176` item + `#165`(3)(4)). Phase 1 (this
file): written and run *before* any source change, against `HEAD` at
`a8f6a2df`. It encodes the target (post-fix) contract from the decision doc
(`.claude/output/decisions/2026-09-23-wave6b-dedup-decision.md` section 5)
and user ruling Q4 (stdev/variance canonical, std/var declared aliases on
every door) -- most cells are RED today by design; rust-coder (phase 2) then
python-coder (phase 3) turn them green.

Populations are derived, never hand-listed:

- ``_BIN_METHODS``/``_HEX_METHODS``/``_RASTER_METHODS`` are read live from
  ``transform_registry(kind)["methods"]`` -- the actual wire, not a Python
  mirror of it.
- ``SPELLING_POPULATION`` is the union of those three live sets, per the
  brief's "union of every accepted set the research names"
  (``registry-vocab.md`` sections 1.4 and 2.3).
- ``_ALIASES`` is the one population this file cannot read from a live
  source: `std`/`var` becoming declared aliases of `stdev`/`variance` is a
  design ruling (Q4) the fix has not shipped yet, not a duplicated Rust
  vocabulary -- see ``registry-vocab.md`` section 1.4 Drift 1.
- ``INTERPOLATE_VALUES`` likewise has no live Python source pre-fix (no
  ``interpolate_accepted_values()`` binding exists yet); it is the value set
  research names four times over in prose docstrings
  (``registry-vocab.md`` section 1.4).

Refusal-text pinning is exact everywhere, including the four PyO3
constructors (Bin, AggregateOp, Hex, Raster). Phase 1 could only pin their
``accepted_values()``-based lists as an order-independent token set, because
where rust-coder would insert the two new aliases inside that
declaration-ordered list was not yet decided. Phase 2 landed the order --
each alias sits immediately after its canonical spelling
(``mean|sum|count|min|max|median|variance|var|stdev|std|q1|q3|distinct`` for
``AggFn``, and ``HexAggregate``, a declared ``AggFn`` subset, lists in the
same order: ``mean|sum|count|min|max|median|variance|var|stdev|std``) -- so phase 3 tightened all four assertions to full
ordered-string equality.

Cycle 3 (quality-review remediation): every field-requirement population
(``_BIN_NEEDS_FIELD``/``_HEX_NEEDS_FIELD``/``_RASTER_NEEDS_FIELD``/
``_AGGREGATE_NEEDS_FIELD``) and ``INTERPOLATE_VALUES`` are now read live from
``transform_registry(kind)["methods_needing_field"]`` /
``interpolate_accepted_values()`` -- both bindings exist as of phase 2 cycles
2-4, so nothing in this file needs to be hand-typed pending a future wire
anymore. ``_ALIASES`` stays hand-typed (Q4's alias-canonical mapping is a
design ruling, not a value any single wire call publishes) but is now
asserted equal to what the alias/canonical pairs the wire *does* publish
imply, rather than trusted blind.

Cycle 6 (spec-review remediation): ``Smooth(x_estimator=)`` joins the matrix.
At the time, no live Python binding published its accepted set, so
``_X_ESTIMATOR_METHODS`` was hand-typed from the Rust source and pinned by an
exact refusal-text assertion. Cycle 8 (below) replaced that with the live
binding once one existed.

Cycle 7 (quality-review remediation): ``mark_smooth``'s x_estimator refusal
cells now assert refusal at declaration (``_smooth_x_estimator_declare``,
never renders), matching ``Smooth``/``lmplot``'s own timing instead of
letting a render-time refusal hide the difference.

Cycle 8 (quality-review remediation): ``x_estimator_accepted_values()`` now
exists, so ``_X_ESTIMATOR_METHODS`` reads it live like every other
population in this file. ``mark_smooth``'s throwaway-``Smooth``-construction
gate is replaced by ``validate_optional_choice``, shared with ``lmplot``.
``fm.Smooth`` is the raw PyO3 pyclass with no Python wrapper, so it is not
one of that gate's callers -- its own constructor is the third door's gate,
unreachable from Python source. A non-string ``x_estimator`` now refuses by
name on both Python doors (previously a raw PyO3 ``TypeError`` reached the
caller); ``fm.Smooth`` keeps its own PyO3 ``TypeError`` for a non-string,
documented as the one gap this task's gate cannot close without editing
``crates/``.

Cycle 9 (orchestrator ruling reversed): ``validate_optional_choice`` no
longer hand-rebuilds ``fm.Smooth``'s own Rust text for a bad string. The
wave constraint ("every refusal names the public entry the caller invoked")
outranks the three-doors-agree-on-text invariant cycle 8 kept, so
``mark_smooth``/``lmplot`` now render ``validate_choice``'s usual door-named
wording for any non-member value, string or not -- the same shape
``validate_optional_choice`` already used. ``fm.Smooth``'s own Rust text
is unaffected (it has no Python wrapper to gate through). The matrix's
invariant is now "each door names itself, and all three render the same
accepted set" rather than "identical text across all three."
"""

from __future__ import annotations

import hashlib
import json

import polars as pl
import pytest

import ferrum as fm
from ferrum._core import (
    interpolate_accepted_values,
    transform_registry,
    x_estimator_accepted_values,
)
from ferrum.transforms import transform_aggregate, transform_join_aggregate, transform_window

# ---------------------------------------------------------------------------
# Derived populations -- every one read live from the wire.
# ---------------------------------------------------------------------------

_BIN_ENTRY = transform_registry("bin")
_HEX_ENTRY = transform_registry("hex")
_RASTER_ENTRY = transform_registry("raster")
_AGGREGATE_ENTRY = transform_registry("aggregate")
_JOIN_AGGREGATE_ENTRY = transform_registry("join_aggregate")
_DATA_AGGREGATE_ENTRY = transform_registry("data_aggregate")
_DATA_WINDOW_ENTRY = transform_registry("data_window")

_BIN_METHODS = frozenset(_BIN_ENTRY["methods"])
_HEX_METHODS = frozenset(_HEX_ENTRY["methods"])
_RASTER_METHODS = frozenset(_RASTER_ENTRY["methods"])
_AGGREGATE_METHODS = frozenset(_AGGREGATE_ENTRY["methods"])

_BIN_NEEDS_FIELD = frozenset(_BIN_ENTRY["methods_needing_field"])
_HEX_NEEDS_FIELD = frozenset(_HEX_ENTRY["methods_needing_field"])
_RASTER_NEEDS_FIELD = frozenset(_RASTER_ENTRY["methods_needing_field"])
_AGGREGATE_NEEDS_FIELD = frozenset(_AGGREGATE_ENTRY["methods_needing_field"])

# Q4 (decision doc section 5): stdev/variance canonical, std/var declared
# aliases on every door that reads AggFn's vocabulary. This mapping cannot be
# read from any single wire call (no binding publishes "these two spellings
# are aliases of each other") -- it is asserted below against what the wire
# *does* publish: every alias's canonical counterpart is itself a declared
# `bin`/`aggregate` spelling, and both members of each pair carry the same
# field requirement.
_ALIASES = {"std": "stdev", "var": "variance"}
for _alias, _canonical in _ALIASES.items():
    assert _canonical in _BIN_METHODS and _alias in _BIN_METHODS
    assert (_canonical in _AGGREGATE_NEEDS_FIELD) == (_alias in _AGGREGATE_NEEDS_FIELD)
del _alias, _canonical

# The union of every accepted set research names, read live.
SPELLING_POPULATION = frozenset(_BIN_METHODS | _HEX_METHODS | _RASTER_METHODS)
BOGUS = "geomean"
ALL_SPELLINGS = sorted(SPELLING_POPULATION | {BOGUS})

# Target (post-fix) accepted sets, one per door family. bin/aggregate now
# both publish the full 13-spelling AggFn vocabulary (incl. aliases) on the
# wire, so this equality is itself a live cross-check, not an assumption.
_STAT_ALL = _BIN_METHODS
assert _STAT_ALL == _AGGREGATE_METHODS
_HEX_TARGET = _HEX_METHODS
_RASTER_TARGET = _RASTER_METHODS  # unaffected: raster has no std/var overlap

# Phase 2's landed accepted_values() ordering (report, "Alias ordering"):
# each alias sits immediately after its canonical spelling. Asserted below
# against the derived target sets, so a future re-ordering fails this file
# loudly instead of silently going stale.
_STAT_JOINED = "mean|sum|count|min|max|median|variance|var|stdev|std|q1|q3|distinct"
_HEX_JOINED = "mean|sum|count|min|max|median|variance|var|stdev|std"
_RASTER_JOINED = "count|density|mean|sum|any"
assert frozenset(_STAT_JOINED.split("|")) == _STAT_ALL
assert frozenset(_HEX_JOINED.split("|")) == _HEX_TARGET
assert frozenset(_RASTER_JOINED.split("|")) == _RASTER_TARGET

INTERPOLATE_VALUES = frozenset(interpolate_accepted_values())


# ---------------------------------------------------------------------------
# Target accept/refuse decisions, mirroring each Rust constructor's own
# check order (bin.rs's pairing check gates before spelling; hex.rs/raster.rs/
# aggregate.rs check spelling before the needs-a-field check).
# ---------------------------------------------------------------------------


def _bin_target(spelling: str, field_present: bool) -> str:
    # bin's pairing check (bin.rs) structurally refuses whenever agg_field is
    # absent and agg is set, for EVERY declared spelling including "count" --
    # confirmed live below by _BIN_NEEDS_FIELD == _BIN_METHODS, so this
    # unconditional short-circuit is provably equivalent to "declared as
    # needing a field", not a hand-guessed rule.
    if not field_present:
        return "refuse_field"
    if spelling not in _STAT_ALL:
        return "refuse_spelling"
    return "accept"


assert _BIN_NEEDS_FIELD == _BIN_METHODS


def _aggregateop_target(spelling: str, field_present: bool) -> str:
    if spelling not in _STAT_ALL:
        return "refuse_spelling"
    if spelling in _AGGREGATE_NEEDS_FIELD and not field_present:
        return "refuse_field"
    return "accept"


def _hex_target(spelling: str, field_present: bool) -> str:
    if spelling not in _HEX_TARGET:
        return "refuse_spelling"
    if spelling in _HEX_NEEDS_FIELD and not field_present:
        return "refuse_field"
    return "accept"


def _raster_target(spelling: str, field_present: bool) -> str:
    if spelling not in _RASTER_TARGET:
        return "refuse_spelling"
    if spelling in _RASTER_NEEDS_FIELD and not field_present:
        return "refuse_field"
    return "accept"


def _validate_choice_text(door: str, param: str, choices, bad: str) -> str:
    """Mirror ``ferrum._validate.validate_choice``'s exact message algorithm."""
    return f"{door}: {param} must be one of {sorted(str(c) for c in choices)}; got {bad!r}"


def _vocabulary_refusal_text(door: str, param: str, accepted: list[str], bad: object) -> str:
    """The Rust vocabulary refusal shape, ``{door}: unknown {param} '{bad}';
    expected a|b|c``, in the order Rust publishes *accepted*."""
    return f"{door}: unknown {param} '{bad}'; expected {'|'.join(accepted)}"


# ---------------------------------------------------------------------------
# Bin(agg=)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field_present", [True, False])
@pytest.mark.parametrize("spelling", ALL_SPELLINGS)
def test_bin_door(spelling, field_present):
    kwargs: dict = {"agg": spelling}
    if field_present:
        kwargs["agg_field"] = "w"
    target = _bin_target(spelling, field_present)
    if target == "accept":
        fm.Bin("v", **kwargs)
    elif target == "refuse_field":
        with pytest.raises(ValueError) as exc:
            fm.Bin("v", **kwargs)
        assert str(exc.value) == "Bin: agg_field and agg must both be set, or both omitted"
    else:
        with pytest.raises(ValueError) as exc:
            fm.Bin("v", **kwargs)
        assert str(exc.value) == f"Bin: unknown agg '{spelling}'; expected {_STAT_JOINED}"


# ---------------------------------------------------------------------------
# AggregateOp(fn_=) and Aggregate(ops=[AggregateOp(...)])
# ---------------------------------------------------------------------------


def _aggregateop_refusal_text(target: str, spelling: str) -> str:
    """The exact message ``AggregateOp``'s two non-accept branches raise.

    Shared by the ``AggregateOp`` door test and the ``Aggregate`` wrapper
    door test below, which routes through the identical constructor call --
    one message table, not two copies that could silently diverge.
    """
    if target == "refuse_field":
        return 'AggregateOp: field must be non-empty (use "*" for count of all rows)'
    assert target == "refuse_spelling"
    return f"AggregateOp: unknown fn '{spelling}'; expected {_STAT_JOINED}"


@pytest.mark.parametrize("field_present", [True, False])
@pytest.mark.parametrize("spelling", ALL_SPELLINGS)
def test_aggregateop_door(spelling, field_present):
    field = "v" if field_present else ""
    target = _aggregateop_target(spelling, field_present)
    if target == "accept":
        fm.AggregateOp(field, spelling, "out")
    else:
        with pytest.raises(ValueError) as exc:
            fm.AggregateOp(field, spelling, "out")
        assert str(exc.value) == _aggregateop_refusal_text(target, spelling)


@pytest.mark.parametrize("field_present", [True, False])
@pytest.mark.parametrize("spelling", ALL_SPELLINGS)
def test_aggregate_wrapper_door(spelling, field_present):
    """Aggregate(ops=[...]) routes through AggregateOp.__init__ unchanged."""
    field = "v" if field_present else ""
    target = _aggregateop_target(spelling, field_present)
    if target == "accept":
        fm.Aggregate(ops=[fm.AggregateOp(field, spelling, "out")])
    else:
        with pytest.raises(ValueError) as exc:
            fm.Aggregate(ops=[fm.AggregateOp(field, spelling, "out")])
        assert str(exc.value) == _aggregateop_refusal_text(target, spelling)


# ---------------------------------------------------------------------------
# Hex(aggregate=)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field_present", [True, False])
@pytest.mark.parametrize("spelling", ALL_SPELLINGS)
def test_hex_door(spelling, field_present):
    kwargs: dict = {"aggregate": spelling}
    if field_present:
        kwargs["field"] = "v"
    target = _hex_target(spelling, field_present)
    if target == "accept":
        fm.Hex("x", "y", **kwargs)
    elif target == "refuse_field":
        with pytest.raises(ValueError) as exc:
            fm.Hex("x", "y", **kwargs)
        assert str(exc.value) == f"Hex: aggregate='{spelling}' requires field"
    else:
        with pytest.raises(ValueError) as exc:
            fm.Hex("x", "y", **kwargs)
        assert str(exc.value) == f"Hex: unknown aggregate '{spelling}'; expected {_HEX_JOINED}"


# ---------------------------------------------------------------------------
# Raster(aggregate=)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field_present", [True, False])
@pytest.mark.parametrize("spelling", ALL_SPELLINGS)
def test_raster_door(spelling, field_present):
    kwargs: dict = {"aggregate": spelling}
    if field_present:
        kwargs["field"] = "v"
    target = _raster_target(spelling, field_present)
    if target == "accept":
        fm.Raster("x", "y", **kwargs)
    elif target == "refuse_field":
        with pytest.raises(ValueError) as exc:
            fm.Raster("x", "y", **kwargs)
        assert str(exc.value) == f"Raster: aggregate='{spelling}' requires field"
    else:
        with pytest.raises(ValueError) as exc:
            fm.Raster("x", "y", **kwargs)
        assert str(exc.value) == (
            f"Raster: unknown aggregate '{spelling}'; expected {_RASTER_JOINED}"
        )


# ---------------------------------------------------------------------------
# mark_hex(aggregate=) -- Python door, validate_choice wording. The vocabulary
# is checked eagerly at call time like mark_raster; the needs-field rule at
# resolution, since a color encoding may supply the field.
# ---------------------------------------------------------------------------


_HEX_NO_FIELD = "|".join(
    f"'{m}'"
    for m in transform_registry("hex")["methods"]
    if m not in transform_registry("hex")["methods_needing_field"]
)


@pytest.fixture()
def hex_df() -> pl.DataFrame:
    n = 40
    return pl.DataFrame(
        {
            "x": [float(i % 7) for i in range(n)],
            "y": [float(i % 5) for i in range(n)],
            "v": [float(i) for i in range(n)],
        }
    )


def _hex_render_svg(df: pl.DataFrame, aggregate: str, field: str | None = "v") -> str:
    return fm.Chart(df).mark_hex(aggregate=aggregate, field=field).encode(x="x", y="y").to_svg()


@pytest.mark.parametrize("field_present", [True, False])
@pytest.mark.parametrize("spelling", ALL_SPELLINGS)
def test_mark_hex_door(hex_df, spelling, field_present):
    field = "v" if field_present else None
    target = _hex_target(spelling, field_present)
    if target == "accept":
        fm.Chart(hex_df).mark_hex(aggregate=spelling, field=field).encode(x="x", y="y").to_svg()
    elif target == "refuse_field":
        # Declared, then refused at resolution: `.encode(color=...)` could
        # still have supplied the field.
        chart = fm.Chart(hex_df).mark_hex(aggregate=spelling, field=field).encode(x="x", y="y")
        with pytest.raises(ValueError) as exc:
            chart.to_svg()
        assert str(exc.value) == (
            f"mark_hex: aggregate={spelling!r} requires field=... (the "
            f"column to aggregate); pass field=... or use "
            f"aggregate={_HEX_NO_FIELD}, which need no field."
        )
    else:
        with pytest.raises(ValueError) as exc:
            fm.Chart(hex_df).mark_hex(aggregate=spelling, field=field)
        assert str(exc.value) == _validate_choice_text(
            "mark_hex", "aggregate", _HEX_TARGET, spelling
        )


@pytest.mark.parametrize(("alias", "canonical"), sorted(_ALIASES.items()))
def test_mark_hex_alias_renders_byte_identical_to_canonical(hex_df, alias, canonical):
    svg_alias = _hex_render_svg(hex_df, alias)
    svg_canonical = _hex_render_svg(hex_df, canonical)
    assert (
        hashlib.sha256(svg_alias.encode()).hexdigest()
        == hashlib.sha256(svg_canonical.encode()).hexdigest()
    )


def test_mark_hex_std_and_var_render_differently_control(hex_df):
    """Vacuity control: the hash-equality check above is not trivially true."""
    svg_std = _hex_render_svg(hex_df, "std")
    svg_var = _hex_render_svg(hex_df, "var")
    assert svg_std != svg_var


# ---------------------------------------------------------------------------
# mark_raster(aggregate=) -- Python door, validate_choice wording, checked
# eagerly at call time (#136).
# ---------------------------------------------------------------------------


@pytest.fixture()
def raster_df() -> pl.DataFrame:
    n = 40
    return pl.DataFrame(
        {
            "x": [float(i % 7) for i in range(n)],
            "y": [float(i % 5) for i in range(n)],
            "v": [float(i) for i in range(n)],
        }
    )


@pytest.mark.parametrize("field_present", [True, False])
@pytest.mark.parametrize("spelling", ALL_SPELLINGS)
def test_mark_raster_door(raster_df, spelling, field_present):
    field = "v" if field_present else None
    target = _raster_target(spelling, field_present)
    kwargs = {"aggregate": spelling, "field": field}
    if target == "accept":
        fm.Chart(raster_df).mark_raster(**kwargs)
    elif target == "refuse_field":
        with pytest.raises(ValueError) as exc:
            fm.Chart(raster_df).mark_raster(**kwargs)
        assert str(exc.value) == (
            f"mark_raster: aggregate={spelling!r} requires field=... (the "
            f"column to aggregate); pass field=... or use "
            f"aggregate='count'|'density'|'any', which need no field."
        )
    else:
        with pytest.raises(ValueError) as exc:
            fm.Chart(raster_df).mark_raster(**kwargs)
        assert str(exc.value) == _validate_choice_text(
            "mark_raster", "aggregate", _RASTER_TARGET, spelling
        )


# ---------------------------------------------------------------------------
# RenderConfig(raster_aggregate=) -- Python door, validate_choice wording,
# checked eagerly at construction.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field_present", [True, False])
@pytest.mark.parametrize("spelling", ALL_SPELLINGS)
def test_render_config_door(spelling, field_present):
    kwargs: dict = {"raster_aggregate": spelling}
    if field_present:
        kwargs["raster_field"] = "v"
    target = _raster_target(spelling, field_present)
    if target == "accept":
        fm.RenderConfig(**kwargs)
    elif target == "refuse_field":
        with pytest.raises(ValueError) as exc:
            fm.RenderConfig(**kwargs)
        assert str(exc.value) == (
            f"RenderConfig: raster_aggregate={spelling!r} requires raster_field=... (the "
            f"column to aggregate); pass raster_field=... or use "
            f"raster_aggregate='count'|'density'|'any', which need no field."
        )
    else:
        with pytest.raises(ValueError) as exc:
            fm.RenderConfig(**kwargs)
        assert str(exc.value) == _validate_choice_text(
            "RenderConfig", "raster_aggregate", _RASTER_TARGET, spelling
        )


# ---------------------------------------------------------------------------
# transform_registry(kind) -- the wire itself.
# ---------------------------------------------------------------------------


def test_transform_registry_raster_and_hex_declare_methods_needing_field():
    expected_needs = {"raster": {"mean", "sum"}, "hex": _HEX_TARGET - {"count"}}
    for kind, expected in expected_needs.items():
        entry = transform_registry(kind)
        assert "methods_needing_field" in entry, sorted(entry)
        needing = frozenset(entry["methods_needing_field"])
        assert needing <= frozenset(entry["methods"])
        assert needing == frozenset(expected)


def test_transform_registry_aggregate_and_join_aggregate_publish_methods():
    for kind in ("aggregate", "join_aggregate"):
        entry = transform_registry(kind)
        assert entry["methods"], f"{kind} methods must be non-empty"
        assert frozenset(entry["methods"]) == _STAT_ALL


# ---------------------------------------------------------------------------
# methods_needing_field: bin / join_aggregate / data_aggregate / data_window,
# each probed through its own real Python door -- mirroring the shape of the
# Rust registry guard (`methods_needing_field_is_a_subset_of_declared_methods`,
# phase 2 cycle 3): with a field, the call must be accepted (a control,
# proving a refusal below comes from the field, not a broken fixture);
# without one, refused if and only if the method is declared as needing it.
# ---------------------------------------------------------------------------

_TRANSFORM_DF = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "g": ["a", "a", "b", "b"]})


def _render_wire_transform(spec: dict) -> str:
    return fm.Chart(_TRANSFORM_DF).transform(spec).encode(x="g", y="w").to_svg()


@pytest.mark.parametrize("spelling", sorted(_BIN_METHODS))
def test_bin_methods_needing_field_matches_its_own_refusal(spelling):
    fm.Bin("v", agg_field="w", agg=spelling)  # control: accepted with agg_field
    with pytest.raises(ValueError) as exc:
        fm.Bin("v", agg=spelling)  # no agg_field
    assert str(exc.value) == "Bin: agg_field and agg must both be set, or both omitted"


@pytest.mark.parametrize("spelling", sorted(_JOIN_AGGREGATE_ENTRY["methods"]))
def test_join_aggregate_methods_needing_field_matches_its_own_refusal(spelling):
    # join_aggregate declares every spelling as needing a field (no count()
    # shorthand: every op resolves a real column) -- asserted against the
    # wire, not assumed, before the door is probed.
    assert spelling in _JOIN_AGGREGATE_ENTRY["methods_needing_field"]
    _render_wire_transform(transform_join_aggregate({"field": "x", "fn": spelling, "as": "w"}))
    with pytest.raises(ValueError, match=r"missing field `field`"):
        _render_wire_transform(transform_join_aggregate({"fn": spelling, "as": "w"}))


@pytest.mark.parametrize("spelling", sorted(_DATA_AGGREGATE_ENTRY["methods"]))
def test_data_aggregate_methods_needing_field_matches_its_own_refusal(spelling):
    # data_aggregate (ferrum.transform_aggregate's wire kind) likewise
    # declares every spelling as needing a field.
    assert spelling in _DATA_AGGREGATE_ENTRY["methods_needing_field"]
    _render_wire_transform(
        transform_aggregate({"field": "x", "fn": spelling, "as": "w"}, groupby=["g"])
    )
    with pytest.raises(ValueError, match=r"missing field `field`"):
        _render_wire_transform(transform_aggregate({"fn": spelling, "as": "w"}, groupby=["g"]))


@pytest.mark.parametrize("spelling", sorted(_DATA_WINDOW_ENTRY["methods"]))
def test_data_window_methods_needing_field_matches_its_own_refusal(spelling):
    needs_field = spelling in _DATA_WINDOW_ENTRY["methods_needing_field"]
    with_field = transform_window({"op": spelling, "as": "w", "field": "x"}, sort=["x"])
    _render_wire_transform(with_field)  # control: accepted with field regardless
    without_field = transform_window({"op": spelling, "as": "w"}, sort=["x"])
    if needs_field:
        with pytest.raises(ValueError, match=r"requires 'field'"):
            _render_wire_transform(without_field)
    else:
        _render_wire_transform(without_field)


# ---------------------------------------------------------------------------
# Interpolate: declaration-time gate + byte-identical render corpus.
# ---------------------------------------------------------------------------

_INTERPOLATE_LINE_DF = pl.DataFrame(
    {"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.0, 4.0, 1.0, 5.0, 3.0]}
)

# Hashes captured 2026-09-23 at HEAD (a8f6a2df), before any source change in
# this task -- the required "small hash corpus now" pin. Phase 3 must
# reproduce every one of these exactly (byte-stable for SVG, per the decision
# doc's declared contract for every valid interpolate=).
_INTERPOLATE_SVG_SHA256 = {
    "linear": "e3722fdd24cc3edaa40803630df01d87d5e149df3f37bf05a77a895965a348c9",
    "step": "3cdc110bad0e4535c503cd8c5f72912161ff16fa7e17175ab00d95cc049ef10c",
    "step-before": "d1259447796addde3a34792166b320f64164a35d3ba3174067488b899d9e97ba",
    "step-after": "baca45a2913ea599f1bbcc01043e1fff593f7b65a5dd8e0162fb010b09f34fa7",
    "monotone": "2e93c04ccfa6acac1f29aee06ccbb116b15b6acd2c3090e1f8ba873ae13a132c",
    "basis": "1abdfe981925f7927c287c3086ad2c02d91e5eaa307aac10a1ffd24c8ab8a3ef",
    "cardinal": "0f1efc5d2e822359316a1d12fe77666cdc8237154a3b6e51b0388f3d8e4ed680",
}


def _mark_line_svg(interpolate: str) -> str:
    return (
        fm.Chart(_INTERPOLATE_LINE_DF)
        .encode(x="x", y="y")
        .mark_line(interpolate=interpolate)
        .to_svg()
    )


@pytest.mark.parametrize("value", sorted(INTERPOLATE_VALUES))
def test_valid_interpolate_renders_byte_identical_to_pre_fix(value):
    svg = _mark_line_svg(value)
    assert hashlib.sha256(svg.encode()).hexdigest() == _INTERPOLATE_SVG_SHA256[value]


def test_interpolate_hash_corpus_is_not_vacuous():
    """Vacuity control: the pinned hashes are not all the same value."""
    assert len(set(_INTERPOLATE_SVG_SHA256.values())) == len(_INTERPOLATE_SVG_SHA256)


# Every door that accepts interpolate= (grepped across src/ferrum/marks's
# accepted-kwarg tables and mark methods): mark_line/mark_area route through
# MarkBase's _MARK_KWARG_HONORED table (marks/base.py); mark_ribbon is a
# composite-mark constructor parameter with its own gate
# (marks/_chart_methods_statistical.py), not routed through MarkBase.
INTERPOLATE_DOORS = ("mark_line", "mark_area", "mark_ribbon")

_INTERPOLATE_RIBBON_DF = pl.DataFrame(
    {"x": [1.0, 2.0, 3.0], "y_lo": [0.5, 1.5, 2.5], "y_hi": [1.5, 2.5, 3.5]}
)


def _interpolate_door_chart(door: str, interpolate):
    """Build (never renders) a chart through *door* with the given interpolate=.

    Every door here validates interpolate= eagerly, at this call, so none
    need .encode()/.to_svg() to raise.
    """
    if door == "mark_ribbon":
        return fm.Chart(_INTERPOLATE_RIBBON_DF).mark_ribbon(interpolate=interpolate)
    chart = fm.Chart(_INTERPOLATE_LINE_DF).encode(x="x", y="y")
    return getattr(chart, door)(interpolate=interpolate)


def _interpolate_door_svg(door: str, interpolate) -> str:
    chart = _interpolate_door_chart(door, interpolate)
    if door == "mark_ribbon":
        chart = chart.encode(x="x", y="y_lo", y2="y_hi")
    return chart.to_svg()


def _interpolate_door_svg_default(door: str) -> str:
    """Render *door*'s output with interpolate= never passed at all."""
    if door == "mark_ribbon":
        return (
            fm.Chart(_INTERPOLATE_RIBBON_DF)
            .mark_ribbon()
            .encode(x="x", y="y_lo", y2="y_hi")
            .to_svg()
        )
    chart = fm.Chart(_INTERPOLATE_LINE_DF).encode(x="x", y="y")
    return getattr(chart, door)().to_svg()


@pytest.mark.parametrize("door", INTERPOLATE_DOORS)
def test_valid_interpolate_accepted_at_every_door(door):
    for value in sorted(INTERPOLATE_VALUES):
        _interpolate_door_chart(door, value)


@pytest.mark.parametrize("bad", ["cubic", 5])
@pytest.mark.parametrize("door", INTERPOLATE_DOORS)
def test_bogus_or_nonstring_interpolate_refused_at_declaration_on_every_door(door, bad):
    """Non-member strings and non-strings both refuse, at declaration, on
    every door -- no isinstance(str) guard skips a non-string value into a
    Rust serde message that never names the door the caller called."""
    expected = _vocabulary_refusal_text(door, "interpolate", interpolate_accepted_values(), bad)
    with pytest.raises(ValueError) as exc:
        _interpolate_door_chart(door, bad)
    assert str(exc.value) == expected


@pytest.mark.parametrize("door", INTERPOLATE_DOORS)
def test_interpolate_none_is_unset_and_renders_like_the_default_on_every_door(door):
    """interpolate=None means "unset" on every door -- it must not refuse,
    and must render identically to the door's own default (interpolate=
    never passed at all). Pinned across all three doors together, not as
    three independent assertions, so the doors cannot silently disagree
    with each other again."""
    _interpolate_door_chart(door, None)  # must not raise
    assert _interpolate_door_svg(door, None) == _interpolate_door_svg_default(door)


# ---------------------------------------------------------------------------
# Smooth(x_estimator=) -- a declared AggFn subset (spec-reviewer finding).
# fm.Smooth is the raw PyO3 pyclass with no Python wrapper -- its own
# constructor is the direct door and this file's gate function cannot reach
# it, so it keeps its own Rust text ("Smooth: unknown x_estimator '...';
# expected a|b|c") for any bad string, and its own raw PyO3 TypeError for a
# non-string. mark_smooth and lmplot both refuse at declaration through
# validate_optional_choice, naming themselves for ANY non-member value,
# string or not (the wave constraint -- every new refusal names the public
# entry the caller invoked -- outranks matching fm.Smooth's own wording).
# The cross-door invariant is no longer "identical text": it is "each door
# names itself, and all three render the same accepted SET" -- true by
# construction here, since every door's expected text below is built from
# the one live x_estimator_accepted_values() call, not three independent
# copies.
# ---------------------------------------------------------------------------

_X_ESTIMATOR_METHODS = frozenset(x_estimator_accepted_values())
_X_ESTIMATOR_JOINED = "|".join(x_estimator_accepted_values())
assert frozenset(_X_ESTIMATOR_JOINED.split("|")) == _X_ESTIMATOR_METHODS
assert _X_ESTIMATOR_METHODS <= _STAT_ALL  # a declared subset of AggFn

SMOOTH_X_ESTIMATOR_DOORS = ("Smooth", "mark_smooth", "lmplot")

_SMOOTH_DF = pl.DataFrame(
    {"x": [float(i) for i in range(1, 21)], "y": [float(i) + (-1) ** i for i in range(20)]}
)


def _smooth_x_estimator_declare(door: str, x_estimator):
    """Construct *door* with x_estimator= -- no rendering.

    All three doors now refuse a bad x_estimator at this point: Smooth and
    lmplot by construction; mark_smooth through its own eager gate (the same
    sibling-timing fix as mark_ribbon's interpolate= gate, #137/#176) rather
    than deferring to .to_svg() the way it used to (cycle 7, quality-review
    finding).
    """
    if door == "Smooth":
        return fm.Smooth("x", "y", x_bins=4, x_estimator=x_estimator)
    if door == "mark_smooth":
        return (
            fm.Chart(_SMOOTH_DF).encode(x="x", y="y").mark_smooth(x_bins=4, x_estimator=x_estimator)
        )
    if door == "lmplot":
        return fm.lmplot(_SMOOTH_DF, x="x", y="y", x_bins=4, x_estimator=x_estimator)
    raise AssertionError(door)


def _smooth_x_estimator_attempt(door: str, x_estimator) -> None:
    """Construct *door* with x_estimator=, and render it too for mark_smooth.

    The accept-cell coverage: not just that construction didn't raise, but
    that the full pipeline completes. Smooth is not a Chart (nothing to
    render); lmplot's own accept coverage stays construction-only, unchanged
    from before this cycle.
    """
    chart = _smooth_x_estimator_declare(door, x_estimator)
    if door == "mark_smooth":
        chart.to_svg()


def _smooth_wire_x_estimator(chart) -> str:
    """The Smooth transform's x_estimator field straight off *chart*'s wire spec."""
    spec = json.loads(chart._build_spec().to_json())
    for t in spec["transforms"]:
        if t.get("type") == "smooth":
            return t["x_estimator"]
    raise AssertionError("no smooth transform in the built spec")


@pytest.mark.parametrize("spelling", sorted(_X_ESTIMATOR_METHODS))
@pytest.mark.parametrize("door", SMOOTH_X_ESTIMATOR_DOORS)
def test_smooth_x_estimator_every_declared_spelling_accepted(door, spelling):
    _smooth_x_estimator_attempt(door, spelling)  # must not raise


@pytest.mark.parametrize("bad", ["std", "q1", BOGUS])
def test_smooth_refused_with_its_own_rust_text(bad):
    """Refused at declaration, in fm.Smooth's own Rust constructor text --
    unaffected by the wave-constraint ruling, since fm.Smooth has no Python
    wrapper for validate_optional_choice to gate through."""
    expected = f"Smooth: unknown x_estimator {bad!r}; expected {_X_ESTIMATOR_JOINED}"
    with pytest.raises(ValueError) as exc:
        _smooth_x_estimator_declare("Smooth", bad)
    assert str(exc.value) == expected


@pytest.mark.parametrize("bad", ["std", "q1", BOGUS, 5])
@pytest.mark.parametrize("door", ("mark_smooth", "lmplot"))
def test_smooth_python_doors_refuse_by_name(door, bad):
    """mark_smooth/lmplot name themselves for any non-member value, string
    or not (the wave constraint -- every new refusal names the public entry
    the caller invoked -- outranks matching fm.Smooth's own wording),
    refused at declaration: _smooth_x_estimator_declare never renders, so a
    pass here proves the refusal happened at construction, not merely that
    it happened somewhere before an unreached .to_svg()."""
    expected = _validate_choice_text(door, "x_estimator", _X_ESTIMATOR_METHODS, bad)
    with pytest.raises(ValueError) as exc:
        _smooth_x_estimator_declare(door, bad)
    assert str(exc.value) == expected


def test_smooth_x_estimator_nonstring_on_the_raw_pyclass_is_a_pyo3_typeerror():
    """fm.Smooth is the raw PyO3 pyclass with no Python wrapper -- a
    non-string x_estimator raises PyO3's own TypeError here, not the
    ValueError the two Python doors raise. Ruled crate-wide and
    pre-existing (every pyclass constructor door has this gap, not just
    Smooth's), filed as F16; documented here rather than silently
    uncovered, not a contract this matrix asks fm.Smooth to meet."""
    with pytest.raises(TypeError):
        fm.Smooth("x", "y", x_bins=4, x_estimator=5)


@pytest.mark.parametrize("spelling", sorted(_X_ESTIMATOR_METHODS))
@pytest.mark.parametrize("door", ("mark_smooth", "lmplot"))
def test_smooth_x_estimator_reaches_the_wire_byte_identical_to_the_direct_spelling(door, spelling):
    """The wire's x_estimator field, after mark_smooth/lmplot's Python
    forwarding, equals the literal spelling the caller typed -- proving the
    sugar doors do not silently rewrite it on the way to Rust, the same
    guarantee fm.Smooth(x_estimator=...) gives by construction (it IS the
    wire, so it has no separate forwarding step to drift in)."""
    if door == "mark_smooth":
        chart = (
            fm.Chart(_SMOOTH_DF).encode(x="x", y="y").mark_smooth(x_bins=4, x_estimator=spelling)
        )
    else:
        chart = fm.lmplot(_SMOOTH_DF, x="x", y="y", x_bins=4, x_estimator=spelling)
    assert _smooth_wire_x_estimator(chart) == spelling


def test_smooth_x_estimator_min_and_max_render_differently_control():
    """Vacuity control: two different x_estimator spellings must not
    silently collapse to the same rendered output."""
    svg_min = (
        fm.Chart(_SMOOTH_DF).encode(x="x", y="y").mark_smooth(x_bins=4, x_estimator="min").to_svg()
    )
    svg_max = (
        fm.Chart(_SMOOTH_DF).encode(x="x", y="y").mark_smooth(x_bins=4, x_estimator="max").to_svg()
    )
    assert svg_min != svg_max


# ---------------------------------------------------------------------------
# Population sanity / vacuity controls.
# ---------------------------------------------------------------------------


def test_hex_target_is_a_declared_subset_of_stat_all():
    assert _HEX_TARGET < _STAT_ALL


def test_bogus_spelling_outside_every_target_set():
    assert BOGUS not in _STAT_ALL
    assert BOGUS not in _HEX_TARGET
    assert BOGUS not in _RASTER_TARGET
    assert BOGUS not in INTERPOLATE_VALUES
    assert BOGUS not in _X_ESTIMATOR_METHODS


def test_aliases_map_onto_declared_canonical_stat_names():
    assert frozenset(_ALIASES.values()) <= _BIN_METHODS
