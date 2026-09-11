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

import polars as pl
import pytest

import ferrum as fm
from ferrum._interactive import _render_scene
from ferrum.selection import selection_interval, selection_point, value


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
