"""Channel-level ``title=`` reaches every legend surface (#90, wave 4 task 2).

Before this fix, ``fm.Color(field, title="Group")`` reached the wire
(``encoding.color.title``) and the axis analogue honored an equivalent
``title=`` kwarg, but every legend-drawing consumer in Rust ignored
``encoding.color.title`` and fell straight back to the field name. Only the
separate ``Legend(title=...)`` override worked. Root cause:
``crates/ferrum-core/src/render/prepare/legend.rs``'s ``legend_title``
computation (color legend/colorbar) and ``aux_legend_title`` (size/shape/dash)
both skipped the channel's own ``title`` field.

The fix resolves the channel title three-state, identically to axis titles,
through the shared ``resolve_axis_title`` helper (``layout/axis.rs``):
absent -> field name; present + non-blank -> that text verbatim; present +
blank (the ``title=""`` sentinel Python emits for an explicit ``title=None``)
-> suppressed, with no fallback to the field name. Precedence is
``legend.title`` > channel ``title`` > field name, and this is now true on
every surface fed by the one resolved ``legend_title``/``aux_legend_title``
producer: the nominal swatch legend, the continuous colorbar, the
discretizing colorbar, the conditional-color legend (unaffected: a
conditional branch carries no ``title`` of its own, so it still falls back to
the field name), the facet figure-legend, the composite shared-legend band,
and the size/shape/stroke-dash aux blocks.

See also: ``tests/test_v0151_title_suppression.py``'s
``test_channel_title_kwarg_controls_legend_title`` (the flipped v0.15.1 pin,
single-surface) and this module's surface sweep.
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm
from ferrum.selection import selection_point, value
from tests._snapshots import legend_texts as _legend_texts


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 3.0],
            "g": ["a", "b", "a"],
            "v": [1.0, 2.0, 3.0],
            "sz": [1.0, 2.0, 3.0],
            "sh": ["a", "b", "a"],
            "dh": ["a", "b", "a"],
        }
    )


# ---------------------------------------------------------------------------
# Color legend / colorbar surfaces
# ---------------------------------------------------------------------------


def test_nominal_swatch_legend_honors_channel_title(df):
    svg = (
        fm.Chart(df).mark_point().encode(x="x", y="y", color=fm.Color("g", title="Group")).to_svg()
    )
    texts = _legend_texts(svg)
    assert "Group" in texts, f"channel title must appear on the swatch legend; texts={texts}"
    assert "g" not in texts, f"field name must not appear once title overrides it; texts={texts}"


def test_nominal_swatch_legend_suppresses_on_blank_title(df):
    """``Color('g', title=None)`` explicitly suppresses the legend title —
    no fallback to the field name."""
    svg = fm.Chart(df).mark_point().encode(x="x", y="y", color=fm.Color("g", title=None)).to_svg()
    texts = _legend_texts(svg)
    assert "g" not in texts, f"blank channel title must suppress, not fall back; texts={texts}"


def test_continuous_colorbar_honors_channel_title(df):
    svg = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x", y="y", color=fm.Color("v", type="quantitative", title="Value"))
        .to_svg()
    )
    texts = _legend_texts(svg)
    assert "Value" in texts, f"channel title must appear on the colorbar; texts={texts}"
    assert "v" not in texts, f"field name must not appear once title overrides it; texts={texts}"


def test_discretizing_colorbar_honors_channel_title(df):
    svg = (
        fm.Chart(df)
        .mark_point()
        .encode(
            x="x",
            y="y",
            color=fm.Color(
                "v",
                type="quantitative",
                title="Bucketed",
                scale=fm.QuantizeScale(domain=[1.0, 3.0], range=["#f00", "#0f0", "#00f"]),
            ),
        )
        .to_svg()
    )
    texts = _legend_texts(svg)
    assert "Bucketed" in texts, (
        f"channel title must appear on the discretizing colorbar; texts={texts}"
    )
    assert "v" not in texts, f"field name must not appear once title overrides it; texts={texts}"


def test_conditional_color_legend_bare_spelling_unaffected_by_channel_title(df):
    """``encode(color=sel.when(...).otherwise(...))`` — the BARE conditional
    spelling, where ``spec.encoding.color`` stays ``None`` (confirmed via
    ``ConditionalSpec.to_spec_dict``, which never serializes a title from
    ``if_selected``/``if_not``). This spelling is structurally incapable of
    carrying a channel title, so it is unaffected by #90 by construction, not
    by design choice: the conditional-color legend keeps falling back to the
    field name. Pinned so a future change to the conditional path can't
    silently regress this surface without a discriminating test noticing.

    This is NOT the representative case for "conditional-color legend
    honors title="; see
    ``test_conditional_color_legend_h9_spelling_honors_channel_title`` below
    for the spelling that actually is affected."""
    sel = selection_point(fields=["g"], name="sel_title_cc")
    svg = (
        fm.Chart(df)
        .mark_point()
        .add_params(sel)
        .encode(x="x", y="y", color=sel.when(fm.Color("g")).otherwise(value("#cccccc")))
        .to_svg()
    )
    texts = _legend_texts(svg)
    assert "g" in texts, (
        f"conditional-color legend must still fall back to the field name; texts={texts}"
    )


def test_conditional_color_legend_h9_spelling_honors_channel_title(df):
    """``Color(field, title=X, condition=cond)`` — the H9 sibling spelling
    (``chart.py``'s ``encode()``, the ``isinstance(value, ChannelBase)`` +
    ``cond = channel._kwargs.get("condition")`` branch). Unlike the bare
    ``sel.when(...).otherwise(...)`` spelling above, this form populates
    ``spec.encoding.color`` with BOTH ``field`` and ``title`` (the
    conditional is split off into ``_conditionals`` separately, but the base
    field encoding — including its ``title`` — survives), so it flows
    through the fixed ``Some(c) => resolve_axis_title(c.title.as_deref(),
    ...)`` branch in ``legend.rs`` exactly like the non-conditional color
    legend does. This is the surface #90 actually fixes for conditional
    color; the bare spelling above never carried a title to lose."""
    sel = selection_point(fields=["g"], name="sel_title_cc_h9")
    cond = sel.when(fm.Color("g")).otherwise(value("#cccccc"))
    svg = (
        fm.Chart(df)
        .mark_point()
        .add_params(sel)
        .encode(x="x", y="y", color=fm.Color("g", title="CondTitle", condition=cond))
        .to_svg()
    )
    texts = _legend_texts(svg)
    assert "CondTitle" in texts, (
        f"H9 conditional-color spelling must honor the channel title; texts={texts}"
    )
    assert "g" not in texts, f"field name must not appear once title overrides it; texts={texts}"


# ---------------------------------------------------------------------------
# Figure-level surfaces: facet figure-legend, composite shared-legend band
# ---------------------------------------------------------------------------


def test_facet_figure_legend_honors_channel_title(df):
    df2 = df.with_columns(pl.Series("fc", ["p", "q", "p"]))
    svg = (
        fm.Chart(df2)
        .mark_point()
        .encode(x="x", y="y", color=fm.Color("g", title="FacetGroup"))
        .facet(col="fc")
        .to_svg()
    )
    texts = _legend_texts(svg)
    assert texts.count("FacetGroup") == 1, (
        f"one figure-level legend must show the channel title once; texts={texts}"
    )
    assert "g" not in texts, f"field name must not appear once title overrides it; texts={texts}"


def test_composite_shared_legend_band_honors_channel_title(df):
    df_b = df.with_columns((pl.col("y") + 1).alias("y"))
    chart_a = fm.Chart(df).mark_point().encode(x="x", y="y", color=fm.Color("g", title="BandGroup"))
    chart_b = (
        fm.Chart(df_b).mark_point().encode(x="x", y="y", color=fm.Color("g", title="BandGroup"))
    )
    svg = fm.hconcat(chart_a, chart_b, resolve={"color": "shared"}).to_svg()
    texts = _legend_texts(svg)
    assert texts.count("BandGroup") == 1, (
        f"one shared-legend band must show the channel title once; texts={texts}"
    )
    assert "g" not in texts, f"field name must not appear once title overrides it; texts={texts}"


# ---------------------------------------------------------------------------
# Aux (size/shape/stroke-dash) legend blocks
# ---------------------------------------------------------------------------


def test_size_aux_legend_honors_channel_title(df):
    svg = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x", y="y", size=fm.Size("sz", title="SizeTitle"))
        .to_svg()
    )
    texts = _legend_texts(svg)
    assert "SizeTitle" in texts, f"channel title must appear on the size aux legend; texts={texts}"
    assert "sz" not in texts, f"field name must not appear once title overrides it; texts={texts}"


def test_shape_aux_legend_honors_channel_title(df):
    svg = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x", y="y", shape=fm.Shape("sh", title="ShapeTitle"))
        .to_svg()
    )
    texts = _legend_texts(svg)
    assert "ShapeTitle" in texts, (
        f"channel title must appear on the shape aux legend; texts={texts}"
    )
    assert "sh" not in texts, f"field name must not appear once title overrides it; texts={texts}"


def test_stroke_dash_aux_legend_honors_channel_title(df):
    svg = (
        fm.Chart(df)
        .mark_line()
        .encode(x="x", y="y", stroke_dash=fm.StrokeDash("dh", title="DashTitle"))
        .to_svg()
    )
    texts = _legend_texts(svg)
    assert "DashTitle" in texts, (
        f"channel title must appear on the stroke-dash aux legend; texts={texts}"
    )
    assert "dh" not in texts, f"field name must not appear once title overrides it; texts={texts}"


# ---------------------------------------------------------------------------
# Precedence + byte-identity
# ---------------------------------------------------------------------------


def test_legend_title_override_still_outranks_channel_title(df):
    svg = (
        fm.Chart(df)
        .mark_point()
        .encode(
            x="x",
            y="y",
            color=fm.Color("g", title="Channel", legend=fm.Legend(title="Legend Wins")),
        )
        .to_svg()
    )
    texts = _legend_texts(svg)
    assert "Legend Wins" in texts, f"Legend(title=) must outrank the channel title=; texts={texts}"
    assert "Channel" not in texts


def test_no_channel_title_is_byte_identical(df):
    """A chart naming no channel-level ``title=`` renders exactly as before
    #90 — the field name still appears as the legend title."""
    with_no_title = fm.Chart(df).mark_point().encode(x="x", y="y", color="g").to_svg()
    with_default_title = (
        fm.Chart(df).mark_point().encode(x="x", y="y", color=fm.Color("g")).to_svg()
    )
    assert with_no_title == with_default_title
    texts = _legend_texts(with_no_title)
    assert "g" in texts, (
        f"absent channel title must still fall back to the field name; texts={texts}"
    )
