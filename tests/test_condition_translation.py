"""Batch H (F-L13-03/F-L15-01): channel ``condition=`` translation + when()
field shorthand.

``encode(<channel>=fm.<Channel>(field, condition=<ConditionalSpec>))`` used to
serialize the conditional into the per-channel ``EncodingSpec.condition``
field -- an inert nested blob the Rust wire accepts but no renderer consumes.
This module pins the fix: ``Chart.encode()`` translates a channel-carried
``condition=`` into the chart's real conditional pipeline (``_conditionals``,
the same list ``encode(<channel>=<ConditionalSpec>)`` already populated),
stamping the wire channel from the encode key and auto-registering the
source selection -- the base field encoding still applies, minus the
``condition`` kwarg. A ``condition=`` value that isn't a ``ConditionalSpec``
raises ``TypeError`` naming the two working forms.

It also pins H10: a bare ``str`` branch value in ``sel.when(...)``/
``.otherwise(...)`` is field shorthand (parsed via the same grammar
``encode()`` uses) -- a bare string means a field everywhere in ferrum;
literals require ``fm.value(...)``. This is what makes the Altair migration
example (``docs/site/comparison/altair.md``) work as printed.

See ``tests/test_sweep_selection_conditional.py`` for the broader
selection x conditional x composition combinatorial sweep (the
``encode(<channel>=cond)`` form was already covered there); this module is
scoped to the two net-new translation paths landed in batch H task 8.
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

import ferrum as fm
from ferrum._interactive import _render_scene
from ferrum._warn import reset_warnings
from ferrum.encoding import _channel_class_map
from ferrum.encoding._channel_policy import (
    _CONDITIONAL_CHANNELS,
    _CONDITIONAL_REFUSALS,
    conditional_wire_channel,
)
from ferrum.selection import _resolve_encoding_value, selection_interval, selection_point, value

#: The Rust enum the Python conditional vocabulary mirrors. Repo-relative so
#: the mirror test fails loudly (not silently skips) if the file moves.
_RUST_SELECTION_RS = (
    Path(__file__).resolve().parents[1] / "crates" / "ferrum-scene" / "src" / "selection.rs"
)


@pytest.fixture
def df():
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [2.0, 4.0, 1.0, 5.0, 3.0],
            "category": ["a", "b", "a", "b", "a"],
        }
    )


# ── H9: channel-carried condition= translation ──────────────────────────────


def test_channel_condition_translates_to_conditionals(df):
    """ChannelClass(field=..., condition=cond) reaches the chart's
    top-level conditionals, not the per-channel EncodingSpec.condition."""
    sel = selection_point(fields=["category"], name="sel_cc")
    cond = sel.when(fm.Color("category")).otherwise(value("#cccccc"))
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", color=fm.Color("category", condition=cond))
        .add_selection(sel)
        .properties(width=200, height=200)
    )
    kw = json.loads(chart.to_json())

    color_enc = kw["encoding"]["color"]
    assert "condition" not in color_enc, "condition must not reach the per-channel wire dict"
    assert color_enc["field"] == "category", "base field encoding must still apply"

    conditionals = kw["conditionals"]
    assert len(conditionals) == 1
    assert conditionals[0]["selection_name"] == "sel_cc"
    assert conditionals[0]["channel"] == "color"


def test_channel_condition_auto_registers_selection(df):
    """The ConditionalSpec's source selection is auto-registered even
    without an explicit .add_selection() call, mirroring encode(<channel>=cond)."""
    sel = selection_point(fields=["category"], name="sel_auto")
    cond = sel.when(value(1.0)).otherwise(value(0.2))
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", opacity=fm.Opacity("x", condition=cond))
        .properties(width=200, height=200)
    )
    kw = json.loads(chart.to_json())
    selections = kw.get("selections", [])
    assert any(s["name"] == "sel_auto" for s in selections)


def test_channel_condition_non_conditionalspec_raises_typeerror(df):
    """A condition= value that isn't a ConditionalSpec raises TypeError
    naming both working forms."""
    with pytest.raises(TypeError, match=r"encode\(color=<ConditionalSpec>\)"):
        fm.Chart(df).mark_point().encode(
            x="x:Q", y="y:Q", color=fm.Color("category", condition={"not": "a spec"})
        )


def test_channel_condition_reaches_real_interactive_pipeline(df):
    """The translated conditional renders through the real interactive scene
    pipeline -- proving this is not just a spec-dict shape but the live path."""
    sel = selection_point(fields=["category"], name="sel_live")
    cond = sel.when(fm.Color("category")).otherwise(value("#cccccc"))
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", color=fm.Color("category", condition=cond))
        .add_selection(sel)
        .properties(width=200, height=200)
    )
    scene_json, packed = _render_scene(chart)
    assert isinstance(packed, bytes)
    scene = json.loads(scene_json)
    conditionals = scene["interaction"]["conditionals"]
    assert len(conditionals) == 1
    assert conditionals[0]["selection_name"] == "sel_live"
    assert conditionals[0]["channel"] == "color"


# ── Channel vocabulary: which channels a conditional may target ─────────────
#
# Batch-H close remediation (2026-09-11). The H9 translation stamped the raw
# encode key as the wire channel with no mapping onto the conditional
# vocabulary, so `fill`/`stroke` died with a raw serde
# `unknown variant 'fill'` at to_spec(), and every non-conditional channel
# (text, tooltip, facet, ...) turned from silently inert into the same
# unmapped serde failure. One vocabulary now governs both spellings:
# `ferrum.encoding._channel_policy.conditional_wire_channel`.
#
# Close re-review (same day): `stroke` came back OUT of the alias table. The
# only conditional colour value the renderer has (`EncodingValue::Color`) is
# applied to `fill_color` by ferrum-wasm's `apply_value_common`, so aliasing a
# stroke conditional onto `color` repainted the mark's interior -- accepted
# input, wrong visual property. It is refused by name until Rust grows an
# `EncodingValue::StrokeColor`. `fill -> color` is sound and stays.


def _cond(sel):
    return sel.when(fm.Color("category")).otherwise(value("#cccccc"))


def _conditionals_of(chart) -> list[dict]:
    return json.loads(chart.to_json())["conditionals"]


def test_fill_condition_lands_on_color(df):
    """fill carries a conditional onto `color`, the channel it aliases to for
    its base encoding -- and `Color` genuinely IS the fill, so the wire value
    changes the property the user named."""
    sel = selection_point(fields=["category"], name="sel_fill")
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", fill=fm.Fill("category", condition=_cond(sel)))
        .properties(width=200, height=200)
    )
    conditionals = _conditionals_of(chart)
    assert len(conditionals) == 1
    assert conditionals[0]["channel"] == "color"


def test_fill_condition_matches_the_encode_cond_spelling(df):
    """`encode(fill=<Fill>(field, condition=cond))` and `encode(fill=cond)`
    produce the identical wire conditional -- one vocabulary, both forms."""
    sel = selection_point(fields=["category"], name="sel_fill_2")
    base = fm.Chart(df).mark_point().properties(width=200, height=200)
    via_channel = base.encode(x="x:Q", y="y:Q", fill=fm.Fill("category", condition=_cond(sel)))
    via_cond = base.encode(x="x:Q", y="y:Q", fill=_cond(sel))
    assert _conditionals_of(via_channel) == _conditionals_of(via_cond)


@pytest.mark.parametrize("spelling", ["channel", "bare"])
def test_stroke_condition_is_refused_not_aliased(df, spelling):
    """A stroke conditional is refused by name, both spellings.

    Aliasing it onto `color` would emit a wire conditional the runtime
    applies to `fill_color` (ferrum-wasm `apply_value_common`; `FillStrokeView`
    has no stroke-colour member), so the mark's interior would change on
    selection while the user asked for its outline. A typed refusal beats
    changing the wrong property.
    """
    sel = selection_point(fields=["category"], name=f"sel_stroke_{spelling}")
    declared = fm.Stroke("category", condition=_cond(sel)) if spelling == "channel" else _cond(sel)
    with pytest.raises(ValueError, match="condition= is not supported"):
        fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q", stroke=declared)


def test_stroke_refusal_explains_why_fill_works_and_stroke_does_not(df):
    """The one refusal a user could read as arbitrary says why."""
    sel = selection_point(fields=["category"], name="sel_stroke_msg")
    with pytest.raises(ValueError) as excinfo:
        fm.Chart(df).mark_point().encode(
            x="x:Q", y="y:Q", stroke=fm.Stroke("category", condition=_cond(sel))
        )
    msg = str(excinfo.value)
    assert "fill (-> color)" in msg, f"the working alias must still be listed: {msg!r}"
    assert "FILL" in msg, f"the refusal must name the wrong-property reason: {msg!r}"


def test_stroke_does_not_declare_condition_honored():
    """The honored-kwarg table agrees with the refusal -- `Stroke` must not
    advertise a kwarg `encode()` rejects (`Fill` still does, correctly)."""
    assert "condition" not in fm.Stroke._honored_kwargs
    assert "condition" in fm.Fill._honored_kwargs


@pytest.mark.parametrize("spelling", ["channel", "bare"])
def test_shape_condition_is_refused(df, spelling):
    """A shape conditional is refused by name, both spellings.

    `ChannelName::Shape` exists on the wire, so the spec serialized and
    deserialized happily -- but `EncodingValue` has no `Shape` variant, so the
    conditional could only ever carry a `Field` value, which
    `apply_value_common` returns `false` for: accepted, serialized, and a
    runtime no-op. Accepted-and-inert is the silent-drop shape this batch
    exists to remove, so it refuses like `stroke` does (adjudicated
    2026-09-11, batch-H close re-review; the cycle-1 docstring that called it
    honored was written on the assumption the wire worked).
    """
    sel = selection_point(fields=["category"], name=f"sel_shape_{spelling}")
    declared = fm.Shape("category", condition=_cond(sel)) if spelling == "channel" else _cond(sel)
    with pytest.raises(ValueError, match="condition= is not supported"):
        fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q", shape=declared)


def test_shape_refusal_names_the_missing_wire_value(df):
    """The refusal says *why*, like stroke's does -- an inert channel is not
    obviously different from a working one from the call site."""
    sel = selection_point(fields=["category"], name="sel_shape_msg")
    with pytest.raises(ValueError) as excinfo:
        fm.Chart(df).mark_point().encode(
            x="x:Q", y="y:Q", shape=fm.Shape("category", condition=_cond(sel))
        )
    msg = str(excinfo.value)
    assert "no effect" in msg or "no-op" in msg, f"refusal must name the inertness: {msg!r}"
    assert "shape" not in msg.split("can target only:")[1].split(".")[0], (
        f"shape must no longer be advertised as a target: {msg!r}"
    )


def test_shape_does_not_declare_condition_honored():
    """Honored table agrees with the refusal (mirrors the Stroke case)."""
    assert "condition" not in fm.Shape._honored_kwargs
    assert "sort" in fm.Shape._honored_kwargs, "the rest of Shape's contract is unchanged"
    assert "condition" in fm.StrokeDash._honored_kwargs, (
        "StrokeDash shares Shape's sort role but HAS an EncodingValue -- it keeps condition"
    )


def test_shape_conditional_refused_through_the_inferred_channel_path(df):
    """`Chart.conditional(sel.when(fm.Shape(...)))` carries no stamped channel
    -- the channel is inferred from the branch value at serialization -- so the
    refusal has to resolve the inferred channel too, or this entry point would
    still emit the inert conditional."""
    sel = selection_point(fields=["category"], name="sel_shape_inferred")
    spec = sel.when(fm.Shape("category")).otherwise(fm.Shape("category"))
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").add_selection(sel)
    with pytest.raises(ValueError, match="condition= is not supported"):
        chart.conditional(spec)


def test_conditional_aliases_derive_from_the_encoding_alias_table():
    """The conditional alias map is derived from `_ENCODING_ALIASES`, not a
    hand-copy of its source->target pairs, so the two cannot drift."""
    from ferrum.encoding._aliases import _ENCODING_ALIASES
    from ferrum.encoding._channel_policy import _CONDITIONAL_CHANNEL_ALIASES

    expected = {
        rule.source: rule.target
        for rule in _ENCODING_ALIASES
        if rule.target in _CONDITIONAL_CHANNELS and rule.source not in _CONDITIONAL_REFUSALS
    }
    assert _CONDITIONAL_CHANNEL_ALIASES == expected
    # `stroke` is the alias rule the refusal registry removes; `fill` is the one
    # that survives. Pin both, so a silent re-admission of stroke fails here.
    assert set(_CONDITIONAL_CHANNEL_ALIASES) == {"fill"}
    assert "stroke" in _CONDITIONAL_REFUSALS


@pytest.mark.parametrize(
    "channel_name, channel_cls",
    [
        ("stroke_width", fm.StrokeWidth),
        ("stroke_opacity", fm.StrokeOpacity),
        ("fill_opacity", fm.FillOpacity),
        ("angle", fm.Angle),
    ],
)
def test_numeric_appearance_channels_honor_condition_without_warning(df, channel_name, channel_cls):
    """The four channels whose `condition=` demonstrably works now declare it
    honored -- the warn guard must not tell the user it was ignored on a call
    that changes the render."""
    sel = selection_point(fields=["category"], name=f"sel_{channel_name}")
    cond = sel.when(value(4.0)).otherwise(value(1.0))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chart = (
            fm.Chart(df)
            .mark_point()
            .encode(**{"x": "x:Q", "y": "y:Q", channel_name: channel_cls("x", condition=cond)})
            .properties(width=200, height=200)
        )
    assert not [w for w in caught if "condition" in str(w.message)], (
        f"{channel_name}(condition=...) must not warn: {[str(w.message) for w in caught]}"
    )
    conditionals = _conditionals_of(chart)
    assert len(conditionals) == 1
    assert conditionals[0]["channel"] == channel_name


@pytest.mark.parametrize(
    "channel_name, channel_cls",
    [
        ("text", fm.Text),
        ("tooltip", fm.Tooltip),
        ("detail", fm.Detail),
        ("href", fm.Href),
        ("facet", fm.Facet),
        ("x", fm.X),
    ],
)
def test_unsupported_channel_condition_refused_at_encode(df, channel_name, channel_cls):
    """A conditional on a channel the pipeline cannot carry raises a typed
    ValueError naming the supported set -- at encode() time, not as a raw
    serde `unknown variant` from inside to_spec()."""
    sel = selection_point(fields=["category"], name=f"sel_bad_{channel_name}")
    with pytest.raises(ValueError, match="condition= is not supported"):
        fm.Chart(df).mark_point().encode(
            **{"x": "x:Q", "y": "y:Q", channel_name: channel_cls("category", condition=_cond(sel))}
        )


def test_unsupported_channel_refusal_names_the_supported_channels(df):
    """The refusal message lists what WOULD work, including where the
    fill/stroke aliases land."""
    sel = selection_point(fields=["category"], name="sel_msg")
    with pytest.raises(ValueError) as excinfo:
        fm.Chart(df).mark_point().encode(
            x="x:Q", y="y:Q", text=fm.Text("category", condition=_cond(sel))
        )
    msg = str(excinfo.value)
    for supported in ("color", "opacity", "size", "stroke_dash", "angle"):
        assert supported in msg, f"refusal must name {supported!r}: {msg!r}"
    assert "fill (-> color)" in msg
    advertised = msg.split("can target only:")[1].split(".")[0]
    for refused in ("stroke (-> color)", "shape"):
        assert refused not in advertised, (
            f"{refused!r} is no longer a conditional target and must not be advertised: {msg!r}"
        )


def test_unsupported_channel_refused_for_the_bare_conditionalspec_form(df):
    """The `encode(<channel>=cond)` spelling refuses through the same
    vocabulary -- the two forms cannot disagree about what is supported."""
    sel = selection_point(fields=["category"], name="sel_bad_bare")
    with pytest.raises(ValueError, match="condition= is not supported"):
        fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q", tooltip=_cond(sel))


def test_chart_conditional_refuses_a_stamped_unsupported_channel(df):
    """`Chart.conditional()` routes a pre-stamped channel through the same
    vocabulary, so no entry point can smuggle one past the typed refusal."""
    sel = selection_point(fields=["category"], name="sel_stamped")
    bad = replace(_cond(sel), channel="text")
    with pytest.raises(ValueError, match="condition= is not supported"):
        fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").add_selection(sel).conditional(bad)


def test_chart_conditional_aliases_a_stamped_fill_channel(df):
    """A pre-stamped `fill` is re-pointed at `color` by the same rule."""
    sel = selection_point(fields=["category"], name="sel_stamped_fill")
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .add_selection(sel)
        .conditional(replace(_cond(sel), channel="fill"))
        .properties(width=200, height=200)
    )
    assert _conditionals_of(chart)[0]["channel"] == "color"


def test_every_condition_honoring_channel_produces_a_valid_wire_channel(df):
    """Table-driven vocabulary check over the whole channel map: a channel
    that declares `condition` honored must translate to a wire channel the
    Rust `ChannelName` enum accepts (proven by building the real spec), and
    every other channel must refuse with the typed error. This is the single
    test that keeps the honored-kwarg table, the docstrings, and the Rust
    enum from drifting apart again."""
    sel = selection_point(fields=["category"], name="sel_table")
    for name, cls in sorted(_channel_class_map().items()):
        honors = "condition" in getattr(cls, "_honored_kwargs", frozenset())
        try:
            chart = (
                fm.Chart(df)
                .mark_point()
                .encode(**{"x": "x:Q", "y": "y:Q", name: cls("category", condition=_cond(sel))})
                .properties(width=200, height=200)
            )
        except ValueError as exc:
            assert "condition= is not supported" in str(exc), (
                f"{name}: refusal must be the typed vocabulary error, got {exc!r}"
            )
            assert not honors, f"{name} declares condition honored but the vocabulary refuses it"
            continue
        except TypeError:
            # Channels whose constructor rejects a plain field string (e.g.
            # Tooltip's multi-field form) are out of this test's reach.
            continue
        assert honors, f"{name} translated a condition= but does not declare it honored"
        # to_json() builds the real ChartSpec: a channel the Rust enum
        # rejects raises `unknown variant` here.
        assert _conditionals_of(chart)[0]["channel"] in _CONDITIONAL_CHANNELS


def test_conditional_channels_mirror_rust_encoding_values():
    """The mirror direction: `_CONDITIONAL_CHANNELS` must stay in step with
    Rust's `EncodingValue` variants, parsed from the crate source.

    The table-driven test above proves every channel Python accepts is one
    Rust accepts. Nothing proved the converse, so the "mirrored from Rust"
    comment on `_CONDITIONAL_CHANNELS` was aspirational: if Rust gained a
    conditional value (a stroke colour, say -- the filed follow-up), Python
    would lag silently and no test would fail. This parses
    `crates/ferrum-scene/src/selection.rs`'s `EncodingValue` enum and asserts
    both directions, with the two documented exceptions spelled out.

    **Cycle-3 hardening (H3 S1 cluster, 2026-09-11):** the value-variant ->
    channel mapping is parsed from `EncodingValue::channel()`'s own match
    arms, not assumed from `snake(variant_name)` -- that assumption happened
    to hold for every variant except the deliberately-excluded `Field`, but
    `channel()` is where the mapping is actually declared, so parsing it
    directly can't drift from a future variant whose name doesn't equal its
    channel. `variants_of`'s line regex also no longer requires a trailing
    `{`/`(`/`,` -- a unit variant written without one (the enum's last
    member, say) would previously vanish from the parse silently, shrinking
    the gap set in direction 2 rather than failing loudly.
    """
    src = _RUST_SELECTION_RS.read_text()

    def variants_of(enum_name: str) -> list[str]:
        body = re.search(rf"pub enum {enum_name} \{{(.*?)\n\}}", src, re.DOTALL)
        assert body, f"could not locate the {enum_name} enum -- has the file moved?"
        # A 4-space-indented identifier at line start is a variant line
        # regardless of what follows it (trailing `{`/`(`/`,`/`}`/nothing) --
        # attribute lines (`#[serde(...)]`) and comments (`///`/`//`) start
        # with a non-word character and never match `\w`.
        return re.findall(r"^\s{4}(\w+)\b", body.group(1), re.MULTILINE)

    def snake(name: str) -> str:
        return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()

    def value_to_channel_map() -> dict[str, str]:
        """Parse `EncodingValue::channel()`'s match arms directly, rather than
        assuming a value variant's channel by guessing from its own name."""
        body = re.search(
            r"pub fn channel\(&self\) -> ChannelName \{\s*match self \{(.*?)\n\s*\}\s*\n\s*\}",
            src,
            re.DOTALL,
        )
        assert body, "could not locate EncodingValue::channel() -- has it moved or been renamed?"
        pairs = re.findall(
            r"EncodingValue::(\w+)\s*\{[^}]*\}\s*=>\s*ChannelName::(\w+)", body.group(1)
        )
        assert pairs, f"could not parse any match arms from channel(): {body.group(1)!r}"
        return dict(pairs)

    value_variants = variants_of("EncodingValue")
    channel_variants = variants_of("ChannelName")
    assert "Color" in value_variants and "Angle" in value_variants, (
        f"suspicious EncodingValue parse: {value_variants}"
    )
    assert "Shape" in channel_variants and "X" in channel_variants, (
        f"suspicious ChannelName parse: {channel_variants}"
    )

    value_channel_map = value_to_channel_map()
    assert set(value_channel_map) == set(value_variants), (
        "EncodingValue::channel() must have exactly one match arm per variant "
        f"(Rust's own exhaustiveness check enforces this at compile time):\n"
        f"  Missing an arm : {sorted(set(value_variants) - set(value_channel_map))}\n"
        f"  Unknown variant: {sorted(set(value_channel_map) - set(value_variants))}"
    )

    # `Field` names a data column rather than a literal visual value; it has no
    # channel of its own (Rust maps it to Color as the conventional default).
    rust_value_channels = {snake(chan) for val, chan in value_channel_map.items() if val != "Field"}
    rust_channel_names = {snake(v) for v in channel_variants}

    # Direction 1 — exact mirror, no exceptions in either sense. Adding a value
    # variant in Rust without adding it here fails; keeping a channel here that
    # Rust cannot express (the `shape` case) fails too.
    assert _CONDITIONAL_CHANNELS == rust_value_channels, (
        "_CONDITIONAL_CHANNELS must equal Rust's EncodingValue-backed channels.\n"
        f"  Missing from Python : {sorted(rust_value_channels - _CONDITIONAL_CHANNELS)}\n"
        f"  Extra in Python     : {sorted(_CONDITIONAL_CHANNELS - rust_value_channels)}\n"
        "A new Rust value variant should be added here (and may un-refuse an entry in "
        "_CONDITIONAL_REFUSALS -- EncodingValue::StrokeColor/Shape are the two filed "
        "follow-ups). A Python-only channel is an accepted-but-inert conditional."
    )

    # Direction 2 — every wire channel name with NO value variant must be an
    # explicit, documented refusal. This is what makes the next such gap a test
    # failure instead of a discovery: both `shape` and `stroke` were discoveries.
    gap = rust_channel_names - rust_value_channels
    assert gap <= set(_CONDITIONAL_REFUSALS), (
        "these wire channel names have no EncodingValue counterpart and are not "
        f"documented refusals: {sorted(gap - set(_CONDITIONAL_REFUSALS))}. A conditional "
        "declared on one would serialize and then do nothing -- add it to "
        "_CONDITIONAL_REFUSALS with the reason, or give it a Rust value variant."
    )

    # And every documented refusal must actually refuse, with its reason.
    for name in _CONDITIONAL_REFUSALS:
        with pytest.raises(ValueError, match="condition= is not supported"):
            conditional_wire_channel(name)


def test_channelless_shorthand_branch_warns_as_color():
    """`_resolve_encoding_value(<shorthand>)` with no channel builds its branch
    object as `Color`, and that choice is observable in the channel's own
    honored-kwarg warning.

    Pins the stand-in class the mutation gate flagged: a `Shape` (or any other)
    substitution would warn `shape(aggregate=...)`. The serialized wire dict is
    identical either way, so the warning is the only discriminator -- which is
    exactly why this test exists rather than a comment claiming there is none.
    """
    reset_warnings()
    with pytest.warns(UserWarning, match=r"color\(aggregate"):
        _resolve_encoding_value("mean(x)")


# ── One conditional per wire channel ────────────────────────────────────────
#
# Close re-review (2026-09-11): two declarations resolving to the same wire
# channel -- `encode(color=condA, fill=condB)`, since fill aliases to color --
# produced two wire conditionals with the same `channel`, no warning, and no
# defined winner; whichever the runtime applied last silently discarded the
# other. The declaration layer refuses instead.


def test_two_conditionals_on_one_wire_channel_refused_same_call(df):
    """`encode(color=condA, fill=condB)` refuses, naming both spellings."""
    sel = selection_point(fields=["category"], name="sel_collide")
    with pytest.raises(ValueError) as excinfo:
        fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q", color=_cond(sel), fill=_cond(sel))
    msg = str(excinfo.value)
    assert "encode(color=...)" in msg and "encode(fill=...)" in msg, msg
    assert "'color' wire channel" in msg, msg


def test_two_conditionals_on_one_wire_channel_refused_across_calls(df):
    """The rule holds across chained `encode()` calls, not just within one."""
    sel = selection_point(fields=["category"], name="sel_collide_2")
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q", color=_cond(sel))
    with pytest.raises(ValueError, match="already carries one on this chart"):
        chart.encode(fill=_cond(sel))


def test_collision_rule_covers_the_channel_carried_spelling(df):
    """A channel-carried `condition=` collides with a bare one the same way --
    the check lives at the single append site both spellings go through."""
    sel = selection_point(fields=["category"], name="sel_collide_3")
    with pytest.raises(ValueError, match="both declare a conditional"):
        fm.Chart(df).mark_point().encode(
            x="x:Q", y="y:Q", color=_cond(sel), fill=fm.Fill("category", condition=_cond(sel))
        )


def test_chart_conditional_refuses_a_second_conditional_on_one_channel(df):
    """`Chart.conditional()` enforces the same rule, resolving the channel it
    will serialize under even when the spec carries none."""
    sel = selection_point(fields=["category"], name="sel_collide_4")
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q", color=_cond(sel))
    with pytest.raises(ValueError, match="already carries a conditional"):
        chart.conditional(_cond(sel))


def test_conditionals_on_different_wire_channels_still_compose(df):
    """The rule is per channel, not per chart: distinct channels compose."""
    sel = selection_point(fields=["category"], name="sel_multi")
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(
            x="x:Q",
            y="y:Q",
            color=_cond(sel),
            opacity=sel.when(value(1.0)).otherwise(value(0.2)),
            size=sel.when(value(80.0)).otherwise(value(20.0)),
        )
        .properties(width=200, height=200)
    )
    assert sorted(c["channel"] for c in _conditionals_of(chart)) == ["color", "opacity", "size"]


def test_layer_merge_refuses_two_conditionals_on_one_wire_channel(df):
    """`Chart + Chart` now refuses the same collision `encode()`/`.layer()` do
    (2026-09-11, batch-H close cycle 4). Cycle 3 exempted this merge on the
    premise that "each layer's conditional belongs to that layer's marks" --
    false: the WASM runtime's `conditionals` list is flat and scope-free,
    applying every conditional to every mark batch in every panel
    (`resolve_conditionals_with_packed`), so `a + b` with each side
    conditioning `color` used to build two wire conditionals stamped the
    same channel with no defined winner, the exact shape this rule exists
    to refuse."""
    sel = selection_point(fields=["category"], name="sel_layers")
    left = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q", color=_cond(sel))
    right = fm.Chart(df).mark_line().encode(x="x:Q", y="y:Q", color=_cond(sel))
    with pytest.raises(ValueError, match="chart \\+ chart"):
        left + right


def test_layer_merge_still_allows_distinct_channel_conditionals(df):
    """Two layers conditioning DIFFERENT wire channels still compose --
    the rule is per channel, matching `encode()`'s equivalent."""
    sel = selection_point(fields=["category"], name="sel_layers_distinct")
    left = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q", color=_cond(sel))
    right = (
        fm.Chart(df)
        .mark_line()
        .encode(x="x:Q", y="y:Q", opacity=sel.when(value(1.0)).otherwise(value(0.2)))
    )
    merged = (left + right).properties(width=200, height=200)
    assert sorted(c["channel"] for c in _conditionals_of(merged)) == ["color", "opacity"]


def test_chained_layer_merge_checks_against_everything_merged_so_far(df):
    """`a + b + c` checks a third chart's conditional against BOTH prior
    charts' merged conditionals, not just the immediately-preceding one."""
    sel = selection_point(fields=["category"], name="sel_layers_chain")
    a = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
    b = fm.Chart(df).mark_line().encode(x="x:Q", y="y:Q", color=_cond(sel))
    c = fm.Chart(df).mark_area().encode(x="x:Q", y="y:Q", color=_cond(sel))
    with pytest.raises(ValueError, match="chart \\+ chart"):
        (a + b) + c


def test_stroke_dropped_by_color_warning_no_longer_fires_with_a_kept_conditional(df):
    """The warn/keep incoherence is gone: `encode(color=..., stroke=<Stroke
    with condition>)` used to warn "stroke is ignored" while silently keeping
    and applying the stroke channel's conditional half. With stroke refused as
    a conditional target, the call raises before any spec is built, so the
    contradictory warning cannot be emitted."""
    sel = selection_point(fields=["category"], name="sel_incoherent")
    reset_warnings()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="condition= is not supported"):
            fm.Chart(df).mark_point().encode(
                x="x:Q",
                y="y:Q",
                color=fm.Color("category"),
                stroke=fm.Stroke("category", condition=_cond(sel)),
            )
    assert not [w for w in caught if "is ignored when color=" in str(w.message)], (
        f"the drop-warning must not fire on a refused call: {[str(w.message) for w in caught]}"
    )


# ── H10: bare-string when() field shorthand ─────────────────────────────────


def test_when_bare_string_is_field_shorthand(df):
    """A bare str branch value parses as field shorthand and resolves as
    the implied channel's field, not a literal."""
    sel = selection_point(fields=["category"], name="sel_shorthand")
    cond = sel.when("category:N").otherwise(value("#cccccc"))
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", color=cond)
        .properties(width=200, height=200)
    )
    kw = json.loads(chart.to_json())
    conditionals = kw["conditionals"]
    assert len(conditionals) == 1
    assert conditionals[0]["if_selected"] == {"kind": "field", "name": "category"}


def test_when_bare_string_color_lookalike_resolves_as_field_not_literal():
    """A bare string that LOOKS like a color name (e.g. 'red') still resolves
    as a field reference, not a color literal -- consistent with encode()'s
    rule that literals require fm.value(). This pins the convention against
    a future "helpful" auto-literal heuristic that special-cases parseable
    color names."""
    sel = selection_point(name="sel_lookalike")
    cond = sel.when(fm.Color("category")).otherwise("red")
    d = cond.to_spec_dict(params={})
    assert d["if_not"] == {"kind": "field", "name": "red"}


def test_when_bare_string_altair_migration_example_renders(df):
    """The Altair migration doc example (docs/site/comparison/altair.md)
    renders end to end, not just resolves to a spec dict."""
    brush = selection_interval()
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", color=brush.when("category:N").otherwise(value("lightgray")))
        .add_selection(brush)
        .properties(width=200, height=200)
    )
    scene_json, packed = _render_scene(chart)
    assert isinstance(packed, bytes)
    scene = json.loads(scene_json)
    conditionals = scene["interaction"]["conditionals"]
    assert len(conditionals) == 1
    assert conditionals[0]["if_selected"] == {"kind": "field", "name": "category"}


def test_when_literal_value_wrapped_string_stays_literal(df):
    """fm.value(<string>) keeps literal semantics -- it is not reinterpreted
    as field shorthand even though the wrapped payload is a str."""
    sel = selection_point(fields=["category"], name="sel_literal")
    cond = sel.when(fm.Color("category")).otherwise(value("#cccccc"))
    d = cond.to_spec_dict(params={})
    assert d["if_not"] == {"kind": "color", "value": {"r": 204, "g": 204, "b": 204, "a": 255}}

    # A non-color string wrapped in value() must still raise -- it is never
    # silently treated as field shorthand.
    bad = sel.when(fm.Color("category")).otherwise(value("category:N"))
    with pytest.raises(ValueError, match="not a valid color"):
        bad.to_spec_dict(params={})


# ── H3-1: the fourth entry point -- Chart.layer(Layer(encoding=)) ───────────
#
# Cycle-3 finding: `encode()` (two spellings) and `Chart.conditional()` route
# through the conditional vocabulary, but `Chart.layer(Layer(encoding=...))`
# never passes through `encode()` at all -- a layer's encoding dict is
# converted straight to `_Layer.encoding`, so a channel-carried `condition=`
# used to serialize into the dead per-channel `EncodingSpec.condition` blob
# (nothing on the Rust side reads it) instead of the chart's real
# `conditionals` pipeline. Adjudicated: implement, don't refuse -- route it
# through `_append_conditional` at `Chart.layer()` time, gated by the same
# `_CONDITIONAL_REFUSALS` vocabulary.


def test_layer_encoding_condition_reaches_chart_level_conditionals(df):
    """A `Layer(encoding={"color": fm.Color(field, condition=cond)})` channel
    lands in the chart's top-level `conditionals`, not the per-layer
    `EncodingSpec.condition` blob -- the same translation `encode()` does."""
    sel = selection_point(fields=["category"], name="sel_layer_h31")
    cond = _cond(sel)
    chart = (
        fm.Chart(df)
        .mark_bar()
        .encode(x="x:Q", y="y:Q")
        .layer(fm.Layer(mark="point", encoding={"color": fm.Color("category", condition=cond)}))
        .add_selection(sel)
        .properties(width=200, height=200)
    )
    kw = json.loads(chart.to_json())
    # layers[0] is the chart's own primary mark (synthesized bar layer);
    # layers[1] is the layer attached via .layer(Layer(...)).
    layer_color_enc = kw["layers"][1]["encoding"]["color"]
    assert "condition" not in layer_color_enc, (
        "condition= must not serialize into the per-layer EncodingSpec.condition blob"
    )
    assert layer_color_enc["field"] == "category", "the base field encoding must still apply"
    conditionals = kw["conditionals"]
    assert len(conditionals) == 1
    assert conditionals[0]["selection_name"] == "sel_layer_h31"
    assert conditionals[0]["channel"] == "color"


def test_layer_encoding_condition_auto_registers_selection(df):
    """The layer-declared conditional's source selection is auto-registered,
    mirroring `encode(<channel>=cond)` and the channel-carried spelling."""
    sel = selection_point(fields=["category"], name="sel_layer_auto")
    cond = sel.when(value(1.0)).otherwise(value(0.2))
    chart = (
        fm.Chart(df)
        .mark_bar()
        .encode(x="x:Q", y="y:Q")
        .layer(fm.Layer(mark="point", encoding={"opacity": fm.Opacity("x", condition=cond)}))
        .properties(width=200, height=200)
    )
    kw = json.loads(chart.to_json())
    assert any(s["name"] == "sel_layer_auto" for s in kw.get("selections", []))


@pytest.mark.parametrize("channel_name, channel_cls", [("stroke", fm.Stroke), ("shape", fm.Shape)])
def test_layer_encoding_condition_on_refused_channel_raises(df, channel_name, channel_cls):
    """The refusal vocabulary gates the layer path too -- a `stroke`/`shape`
    conditional declared inside `Layer(encoding=...)` raises the same typed
    error `encode()` raises, instead of silently reaching the dead per-layer
    `EncodingSpec.condition` blob."""
    sel = selection_point(fields=["category"], name=f"sel_layer_refuse_{channel_name}")
    chart = fm.Chart(df).mark_bar().encode(x="x:Q", y="y:Q")
    with pytest.raises(ValueError, match="condition= is not supported"):
        chart.layer(
            fm.Layer(
                mark="point", encoding={channel_name: channel_cls("category", condition=_cond(sel))}
            )
        )


def test_layer_encoding_condition_collides_with_existing_chart_conditional(df):
    """A layer-declared conditional colliding with an already-claimed wire
    channel refuses, the same as two `encode()` calls colliding."""
    sel = selection_point(fields=["category"], name="sel_layer_collide")
    chart = fm.Chart(df).mark_bar().encode(x="x:Q", y="y:Q", color=_cond(sel))
    with pytest.raises(ValueError, match="already carries one on this chart"):
        chart.layer(
            fm.Layer(mark="point", encoding={"fill": fm.Fill("category", condition=_cond(sel))})
        )


# ── H3-1's bare-spelling half: Layer(encoding={<channel>: <bare ConditionalSpec>}) ──
#
# Cycle-5 finding: the entry-point table (below) only pins that this
# spelling REFUSES a bad channel -- nothing pinned the happy path, and a
# mutant that keeps the isinstance(channel, ConditionalSpec) check and its
# vocabulary gate but writes `stripped[name] = channel` instead of
# `continue` (leaking the ConditionalSpec into the layer's own encoding
# dict) passed the whole suite. These three tests mirror the
# channel-carried spelling's three above, one property per test, matching
# `_strip_layer_conditionals`'s docstring claim that both branches mirror
# `encode()`'s two spellings.


def test_layer_encoding_bare_condition_reaches_chart_level_conditionals(df):
    """A bare `Layer(encoding={"color": cond})` -- matching
    `encode(color=cond)` -- lands in the chart's top-level `conditionals`,
    and the layer's OWN encoding dict has the key stripped entirely (not
    merely absent from the wire, which a leaked `ConditionalSpec` would
    ALSO produce -- `_build_layers_list` silently skips any encoding value
    with no `to_encoding_spec_dict()`, so the wire alone cannot distinguish
    `continue` from the reviewer-named mutant `stripped[name] = channel`;
    the `_layers[-1].encoding == {}` assertion below is what actually
    kills it)."""
    sel = selection_point(fields=["category"], name="sel_layer_bare")
    cond = _cond(sel)
    chart = (
        fm.Chart(df)
        .mark_bar()
        .encode(x="x:Q", y="y:Q")
        .layer(fm.Layer(mark="point", encoding={"color": cond}))
        .add_selection(sel)
        .properties(width=200, height=200)
    )
    assert chart._layers[-1].encoding == {}, chart._layers[-1].encoding
    kw = json.loads(chart.to_json())
    assert "color" not in kw["layers"][1]["encoding"]
    conditionals = kw["conditionals"]
    assert len(conditionals) == 1
    assert conditionals[0]["selection_name"] == "sel_layer_bare"
    assert conditionals[0]["channel"] == "color"


def test_layer_encoding_bare_condition_auto_registers_selection(df):
    """The layer-declared bare conditional's source selection is
    auto-registered, mirroring `encode(<channel>=cond)` and the
    channel-carried layer spelling."""
    sel = selection_point(fields=["category"], name="sel_layer_bare_auto")
    cond = sel.when(value(1.0)).otherwise(value(0.2))
    chart = (
        fm.Chart(df)
        .mark_bar()
        .encode(x="x:Q", y="y:Q")
        .layer(fm.Layer(mark="point", encoding={"opacity": cond}))
        .properties(width=200, height=200)
    )
    kw = json.loads(chart.to_json())
    assert any(s["name"] == "sel_layer_bare_auto" for s in kw.get("selections", []))


def test_layer_encoding_bare_condition_collides_with_existing_chart_conditional(df):
    """A bare layer-declared conditional colliding with an already-claimed
    wire channel refuses, the same as the channel-carried layer spelling."""
    sel = selection_point(fields=["category"], name="sel_layer_bare_collide")
    chart = fm.Chart(df).mark_bar().encode(x="x:Q", y="y:Q", color=_cond(sel))
    with pytest.raises(ValueError, match="already carries one on this chart"):
        chart.layer(fm.Layer(mark="point", encoding={"fill": _cond(sel)}))


def test_layer_encoding_fieldless_alias_conditional_beside_base_color_still_builds(df):
    """The fieldless-alias fix (H4-3) applies on the layer path too:
    `Layer(encoding={"fill": fm.Fill(condition=cond)})` beside the chart's
    own `color=` encoding builds -- a fieldless `fill` has no field for the
    alias to drop."""
    sel = selection_point(fields=["category"], name="sel_layer_fieldless")
    cond = _cond(sel)
    chart = (
        fm.Chart(df)
        .mark_bar()
        .encode(x="x:Q", y="y:Q", color=fm.Color("category"))
        .layer(fm.Layer(mark="point", encoding={"fill": fm.Fill(condition=cond)}))
        .properties(width=200, height=200)
    )
    kw = json.loads(chart.to_json())
    assert kw["encoding"]["color"] == {"field": "category"}
    conditionals = kw["conditionals"]
    assert len(conditionals) == 1
    assert conditionals[0]["channel"] == "color"


#: Every public way to attach a conditional, table-driven over entry point x
#: spelling, not over one representative case per entry point (2026-09-11,
#: batch-H close cycle 4: the pre-cycle-4 version of this test covered only
#: the channel-carried spelling for the `Chart.layer()` entry point, which
#: is exactly how the bare-`ConditionalSpec`-in-`Layer(encoding=)`
#: silent-drop (cycle-4 S2) survived cycle 3 undetected -- the reviewer's
#: own framing: "write the tests over the enumeration... not over the one
#: case the finding named"). `Chart.conditional()` has one spelling in this
#: sense (it always takes a bare `ConditionalSpec`; its stamped-vs-inferred
#: axis is a separate dimension pinned by
#: `test_shape_conditional_refused_through_the_inferred_channel_path`).
_ENTRY_POINT_LABELS = (
    "encode/bare",
    "encode/channel-carried",
    "conditional/inferred",
    "layer/bare",
    "layer/channel-carried",
)


def _entry_point_refusal_calls(base, sel) -> dict:
    """Build the label -> thunk table for :data:`_ENTRY_POINT_LABELS`, each
    of which must refuse the same unsupported channel (`shape`) with the
    same typed error."""
    bad_channel = fm.Shape("category", condition=_cond(sel))
    bad_channel_bare = _cond(sel)
    return {
        "encode/bare": lambda: base.encode(shape=bad_channel_bare),
        "encode/channel-carried": lambda: base.encode(shape=bad_channel),
        "conditional/inferred": lambda: base.conditional(
            sel.when(fm.Shape("category")).otherwise(fm.Shape("category"))
        ),
        "layer/bare": lambda: base.layer(
            fm.Layer(mark="point", encoding={"shape": bad_channel_bare})
        ),
        "layer/channel-carried": lambda: base.layer(
            fm.Layer(mark="point", encoding={"shape": bad_channel})
        ),
    }


@pytest.mark.parametrize("label", _ENTRY_POINT_LABELS)
def test_all_conditional_entry_points_route_through_one_vocabulary(df, label):
    """Every (entry point, spelling) combination refuses the SAME
    unsupported channel with the SAME typed error, proving they share one
    resolver rather than N independent checks."""
    sel = selection_point(fields=["category"], name=f"sel_entry_{label.replace('/', '_')}")
    base = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").add_selection(sel)
    calls = _entry_point_refusal_calls(base, sel)
    with pytest.raises(ValueError, match="condition= is not supported"):
        calls[label]()


# ── H3-2: alias base-vs-conditional collision ────────────────────────────────
#
# Cycle-3 finding: the collision rule refused two conditionals on one wire
# channel, but not a base encoding colliding with an alias source's OWN
# conditional -- `encode(color=fm.Color("d"), fill=fm.Fill("c",
# condition=cond))` silently dropped fill's field (keep_target, since color
# is present) while keeping fill's conditional, which then overrides color
# at selection time. Adjudicated: refuse the pair, same message shape as the
# existing same-call collision.


def test_alias_base_field_vs_own_conditional_refused(df):
    """`encode(color=..., fill=Fill(..., condition=cond))` refuses -- fill's
    base field would be silently dropped by the fill->color alias while its
    conditional survives and overrides color."""
    sel = selection_point(fields=["category"], name="sel_alias_collide")
    with pytest.raises(ValueError) as excinfo:
        fm.Chart(df).mark_point().encode(
            x="x:Q",
            y="y:Q",
            color=fm.Color("d"),
            fill=fm.Fill("c", condition=_cond(sel)),
        )
    msg = str(excinfo.value)
    assert "encode(color=...)" in msg and "encode(fill=...)" in msg, msg


def test_alias_base_field_vs_own_conditional_refused_reverse_order(df):
    """The same refusal fires regardless of kwarg order in the call."""
    sel = selection_point(fields=["category"], name="sel_alias_collide_rev")
    with pytest.raises(ValueError) as excinfo:
        fm.Chart(df).mark_point().encode(
            x="x:Q",
            y="y:Q",
            fill=fm.Fill("c", condition=_cond(sel)),
            color=fm.Color("d"),
        )
    msg = str(excinfo.value)
    assert "encode(color=...)" in msg and "encode(fill=...)" in msg, msg


def test_fieldless_alias_conditional_beside_base_color_still_builds(df):
    """`encode(color=fm.Color("g"), fill=fm.Fill(condition=cond))` builds --
    a FIELDLESS `fill` has no base field for the alias to drop, so this is
    not the H3-2 collision (2026-09-11, batch-H close cycle 4, S3: the
    cycle-3 guard raised here unconditionally, a false refusal on a call
    that built before cycle 3 and that the equivalent bare-ConditionalSpec
    spelling `encode(color=..., fill=cond)` still accepts today)."""
    sel = selection_point(fields=["category"], name="sel_alias_fieldless")
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", color=fm.Color("g"), fill=fm.Fill(condition=_cond(sel)))
        .properties(width=200, height=200)
    )
    assert json.loads(chart.to_json())["encoding"]["color"] == {"field": "g"}
    conditionals = _conditionals_of(chart)
    assert len(conditionals) == 1
    assert conditionals[0]["channel"] == "color"


def test_fill_condition_without_a_colliding_base_color_still_works(df):
    """The refusal is scoped to the actual collision -- a fill conditional
    with no separately-encoded color still lands on color as usual."""
    sel = selection_point(fields=["category"], name="sel_alias_ok")
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", fill=fm.Fill("c", condition=_cond(sel)))
        .properties(width=200, height=200)
    )
    conditionals = _conditionals_of(chart)
    assert len(conditionals) == 1
    assert conditionals[0]["channel"] == "color"


# ── H3-3: the honored-kwarg `condition` bit is derived, not hand-placed ─────


def test_condition_honored_bit_is_derived_from_the_vocabulary():
    """A fresh ChannelBase subclass gets its `condition` honored-kwarg bit
    from `_channel_policy.channel_honors_condition`, keyed on `_channel_name`
    -- not from whatever its class body happened to declare. This is the
    executable proof `ChannelBase.__init_subclass__` does the deriving: a
    subclass naming a real conditional channel gains the bit even though its
    own `_honored_kwargs` literal never mentions it, and one naming a refused
    channel never gains it even if its literal DOES mention it."""
    from ferrum.encoding.base import ChannelBase

    class _FakeConditionalChannel(ChannelBase):
        _channel_name = "size"  # a real _CONDITIONAL_CHANNELS member
        _honored_kwargs = frozenset({"type"})  # deliberately omits "condition"

    class _FakeRefusedChannel(ChannelBase):
        _channel_name = "stroke"  # a _CONDITIONAL_REFUSALS member
        _honored_kwargs = frozenset({"type", "condition"})  # deliberately claims it

    assert "condition" in _FakeConditionalChannel._honored_kwargs
    assert "condition" not in _FakeRefusedChannel._honored_kwargs
