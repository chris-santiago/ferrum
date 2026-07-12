"""Tests for ``fm.Resolve(scale=, legend=)`` — GH #16 Task 4.

Ships the user-facing control for composite figure-level shared legends: a
new ``Resolve`` value class accepted everywhere ``resolve=`` is accepted
today, plus the ``"legend"`` sub-object emitted on the composite node's wire
``resolve`` field (spec §6). Legend resolution defaults to following scale
resolution and can be forced back to per-panel with
``legend={"color": "independent"}``.

The actual figure-level legend RENDERING (one legend drawn instead of N
per-panel legends) is Rust-side work landing in later tasks of this phase —
this file tests only what Task 4 owns: ``Resolve`` normalization, the wire
shape ``_lower_composite`` produces, the validation matrix's typed
``ValueError``s, and back-compat with the flat-dict ``resolve=`` form. See
``design-docs/superpowers/specs/2026-07-12-composite-shared-legend-design.md``
§4/§6 for the semantic rule this file locks in.
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm
from ferrum.composition import (
    HConcatChart,
    LayerChart,
    Resolve,
    _composite_resolve_field,
    _lower_composite,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df_a():
    return pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "g": ["a", "b", "a"]})


@pytest.fixture
def df_b():
    return pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "g": ["a", "b", "b"]})


@pytest.fixture
def chart_a(df_a):
    return fm.Chart(df_a).mark_point().encode(x="x", y="y", color="g")


@pytest.fixture
def chart_b(df_b):
    return fm.Chart(df_b).mark_point().encode(x="x", y="y", color="g")


# ---------------------------------------------------------------------------
# fm.Resolve export + normalization
# ---------------------------------------------------------------------------


def test_resolve_is_exported_from_top_level():
    assert fm.Resolve is Resolve


def test_resolve_default_fields_are_none():
    r = Resolve()
    assert r.scale is None
    assert r.legend is None


def test_resolve_is_a_value_object_equality():
    assert Resolve(scale={"color": "shared"}) == Resolve(scale={"color": "shared"})
    assert Resolve(scale={"color": "shared"}) != Resolve(scale={"color": "independent"})


# ---------------------------------------------------------------------------
# Accepted everywhere resolve= is accepted today.
# ---------------------------------------------------------------------------


def test_hconcat_accepts_resolve_value_class(chart_a, chart_b):
    combo = HConcatChart([chart_a, chart_b], resolve=Resolve(scale={"color": "shared"}))
    assert combo._resolve == Resolve(scale={"color": "shared"})


def test_layerchart_accepts_resolve_value_class(chart_a, chart_b):
    combo = LayerChart(chart_a, chart_b, resolve=Resolve(scale={"color": "independent"}))
    assert combo._resolve == Resolve(scale={"color": "independent"})


def test_repeatchart_accepts_resolve_value_class():
    template = (
        fm.Chart(pl.DataFrame({"m1": [1.0, 2.0], "m2": [3.0, 4.0]}))
        .mark_point()
        .encode(x=fm.Repeat.column, y="m1")
    )
    grid = fm.RepeatChart(template, column=["m1", "m2"], resolve=Resolve(scale={"x": "shared"}))
    assert grid.resolve == Resolve(scale={"x": "shared"})


def test_concatchart_accepts_resolve_value_class(chart_a, chart_b):
    combo = fm.ConcatChart(chart_a, chart_b, resolve=Resolve(scale={"y": "shared"}))
    assert combo._resolve == Resolve(scale={"y": "shared"})


def test_hconcat_sugar_accepts_resolve_value_class(chart_a, chart_b):
    combo = fm.hconcat(chart_a, chart_b, resolve=Resolve(scale={"color": "shared"}))
    assert isinstance(combo, HConcatChart)


def test_layer_sugar_accepts_resolve_value_class(chart_a, chart_b):
    combo = fm.layer(chart_a, chart_b, resolve=Resolve(scale={"color": "independent"}))
    assert isinstance(combo, LayerChart)


# ---------------------------------------------------------------------------
# Flat dict remains valid and means Resolve(scale=dict) — back-compat.
# ---------------------------------------------------------------------------


def test_flat_dict_and_resolve_scale_lower_identically(chart_a, chart_b):
    via_dict = HConcatChart([chart_a, chart_b], resolve={"color": "shared"})
    via_resolve = HConcatChart([chart_a, chart_b], resolve=Resolve(scale={"color": "shared"}))
    lowered_dict = _lower_composite(via_dict, auto_tooltips=False)
    lowered_resolve = _lower_composite(via_resolve, auto_tooltips=False)
    assert lowered_dict.tree["resolve"] == lowered_resolve.tree["resolve"]
    assert lowered_dict.tree["resolve"] == {"color": "shared"}


def test_flat_dict_renders_byte_identical_to_resolve_scale_equivalent(chart_a, chart_b):
    """Byte-stability guard: no legend= means the wire is scale-only, unchanged."""
    via_dict = HConcatChart([chart_a, chart_b], resolve={"color": "shared"}).to_svg()
    via_resolve = HConcatChart(
        [chart_a, chart_b], resolve=Resolve(scale={"color": "shared"})
    ).to_svg()
    assert via_dict == via_resolve


def test_no_resolve_still_renders_byte_identical_to_main(chart_a, chart_b):
    """Independent-resolve (no resolve= at all) composites are untouched."""
    plain = HConcatChart([chart_a, chart_b]).to_svg()
    explicit_independent = HConcatChart(
        [chart_a, chart_b], resolve=Resolve(scale={"color": "independent"})
    ).to_svg()
    assert "<svg" in plain
    assert plain == explicit_independent


# ---------------------------------------------------------------------------
# Wire emission: the "legend" sub-object (spec §6 wire contract).
# ---------------------------------------------------------------------------


def test_wire_omits_legend_key_when_resolve_has_no_legend(chart_a, chart_b):
    combo = HConcatChart([chart_a, chart_b], resolve={"color": "shared"})
    lowered = _lower_composite(combo, auto_tooltips=False)
    assert "legend" not in lowered.tree["resolve"]


def test_wire_emits_legend_independent_over_shared_scale(chart_a, chart_b):
    combo = HConcatChart(
        [chart_a, chart_b],
        resolve=Resolve(scale={"color": "shared"}, legend={"color": "independent"}),
    )
    lowered = _lower_composite(combo, auto_tooltips=False)
    assert lowered.tree["resolve"] == {
        "color": "shared",
        "legend": {"color": "independent"},
    }


def test_wire_emits_legend_shared_matching_shared_scale(chart_a, chart_b):
    combo = HConcatChart(
        [chart_a, chart_b],
        resolve=Resolve(scale={"color": "shared"}, legend={"color": "shared"}),
    )
    lowered = _lower_composite(combo, auto_tooltips=False)
    assert lowered.tree["resolve"] == {
        "color": "shared",
        "legend": {"color": "shared"},
    }


def test_composite_resolve_field_directly_scale_and_legend(chart_a, chart_b):
    field = _composite_resolve_field(
        Resolve(scale={"color": "shared", "size": "shared"}, legend={"size": "independent"}),
        kind="TestKind",
    )
    assert field == {
        "color": "shared",
        "size": "shared",
        "legend": {"size": "independent"},
    }


def test_composite_resolve_field_accepts_flat_dict():
    field = _composite_resolve_field({"x": "shared"}, kind="TestKind")
    assert field == {"x": "shared"}


def test_composite_resolve_field_accepts_none():
    assert _composite_resolve_field(None, kind="TestKind") == {}


# ---------------------------------------------------------------------------
# Validation matrix — typed ValueError, never a silent fallback.
# ---------------------------------------------------------------------------


def test_legend_shared_without_scale_shared_raises_at_lowering(chart_a, chart_b):
    """Acceptance criterion 6: legend='shared' with no shared scale is an error."""
    combo = HConcatChart([chart_a, chart_b], resolve=Resolve(legend={"color": "shared"}))
    with pytest.raises(ValueError, match="color"):
        _lower_composite(combo, auto_tooltips=False)


def test_legend_shared_over_independent_scale_raises_at_lowering(chart_a, chart_b):
    combo = HConcatChart(
        [chart_a, chart_b],
        resolve=Resolve(scale={"color": "independent"}, legend={"color": "shared"}),
    )
    with pytest.raises(ValueError, match="color"):
        _lower_composite(combo, auto_tooltips=False)


def test_legend_shared_error_names_both_modes(chart_a, chart_b):
    combo = HConcatChart(
        [chart_a, chart_b],
        resolve=Resolve(scale={"color": "independent"}, legend={"color": "shared"}),
    )
    with pytest.raises(ValueError) as excinfo:
        _lower_composite(combo, auto_tooltips=False)
    message = str(excinfo.value)
    assert "shared" in message
    assert "independent" in message


def test_legend_shared_over_shared_other_channel_scale_raises(chart_a, chart_b):
    """legend={'color': 'shared'} with only size shared (not color) still errors."""
    combo = HConcatChart(
        [chart_a, chart_b],
        resolve=Resolve(scale={"size": "shared"}, legend={"color": "shared"}),
    )
    with pytest.raises(ValueError, match="color"):
        _lower_composite(combo, auto_tooltips=False)


def test_legend_unsupported_channel_raises_at_construction(chart_a, chart_b):
    with pytest.raises(ValueError, match="legend"):
        HConcatChart([chart_a, chart_b], resolve=Resolve(legend={"x": "shared"}))


def test_legend_unsupported_channel_shape_raises(chart_a, chart_b):
    with pytest.raises(ValueError, match="legend"):
        HConcatChart([chart_a, chart_b], resolve=Resolve(legend={"shape": "shared"}))


def test_legend_invalid_mode_string_raises_at_construction(chart_a, chart_b):
    with pytest.raises(ValueError, match="'shared' or 'independent'"):
        HConcatChart([chart_a, chart_b], resolve=Resolve(legend={"color": "loud"}))


def test_resolve_invalid_type_raises():
    chart = fm.Chart(pl.DataFrame({"x": [1]})).mark_point().encode(x="x")
    with pytest.raises(ValueError, match="dict, Resolve, or None"):
        fm.HConcatChart([chart, chart], resolve=42)


def test_resolve_legend_wrong_type_raises(chart_a, chart_b):
    with pytest.raises(ValueError, match="resolve.legend"):
        HConcatChart([chart_a, chart_b], resolve=Resolve(legend="color"))


# ---------------------------------------------------------------------------
# Explicit legend=independent over shared scale: today's rendering (spec §4).
# ---------------------------------------------------------------------------


def test_legend_independent_over_shared_scale_lowers_without_error(chart_a, chart_b):
    combo = HConcatChart(
        [chart_a, chart_b],
        resolve=Resolve(scale={"color": "shared"}, legend={"color": "independent"}),
    )
    lowered = _lower_composite(combo, auto_tooltips=False)
    assert lowered.tree["resolve"]["color"] == "shared"
    assert lowered.tree["resolve"]["legend"] == {"color": "independent"}


def test_legend_independent_over_shared_scale_renders(chart_a, chart_b):
    """Acceptance criterion 5 (partial — wire only, rendering lands separately).

    An end-to-end ``to_svg()`` smoke call: the Rust side already round-trips
    an accepted legend override (Task 1), so this must not raise and must
    still produce valid SVG with both panels.
    """
    svg = HConcatChart(
        [chart_a, chart_b],
        resolve=Resolve(scale={"color": "shared"}, legend={"color": "independent"}),
    ).to_svg()
    assert "<svg" in svg


# ---------------------------------------------------------------------------
# share_scale() sugar: scale-only merge, existing legend carried through.
# ---------------------------------------------------------------------------


def test_share_scale_preserves_existing_legend_field(chart_a, chart_b):
    base = HConcatChart(
        [chart_a, chart_b],
        resolve=Resolve(scale={"color": "shared"}, legend={"color": "independent"}),
    )
    merged = base.share_scale(size="shared")
    assert isinstance(merged._resolve, Resolve)
    assert merged._resolve.scale == {"color": "shared", "size": "shared"}
    assert merged._resolve.legend == {"color": "independent"}


def test_share_scale_without_legend_stays_flat_dict(chart_a, chart_b):
    """No legend ever set: share_scale keeps returning the plain scale dict."""
    base = HConcatChart([chart_a, chart_b], resolve={"color": "shared"})
    merged = base.share_scale(size="shared")
    assert merged._resolve == {"color": "shared", "size": "shared"}


def test_share_scale_still_raises_for_jointchart():
    joint = fm.jointplot(pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]}), x="x", y="y")
    with pytest.raises(ValueError, match="resolve="):
        joint.share_scale(x="shared")


# ---------------------------------------------------------------------------
# Interaction with LayerChart's x-always-shared / y-independent invariants.
# ---------------------------------------------------------------------------


def test_layerchart_resolve_value_class_x_independent_still_raises(chart_a, chart_b):
    with pytest.raises(ValueError, match="GH #55"):
        LayerChart(chart_a, chart_b, resolve=Resolve(scale={"x": "independent"}))


def test_layerchart_resolve_value_class_y_independent_still_supported(chart_a, chart_b):
    layered = LayerChart(chart_a, chart_b, resolve=Resolve(scale={"y": "independent"}))
    svg = layered.to_svg()
    assert "<svg" in svg


# ---------------------------------------------------------------------------
# RepeatChart.spec introspection stays JSON-serializable with Resolve inputs.
# ---------------------------------------------------------------------------


def test_repeatchart_spec_serializable_with_resolve_value_class(df_a):
    import json

    template = fm.Chart(df_a).mark_point().encode(x="x", y="y", color="g")
    rc = fm.RepeatChart(
        template,
        row=["x", "y"],
        column=["x", "y"],
        resolve=Resolve(scale={"color": "shared"}, legend={"color": "independent"}),
    )
    spec = rc.spec
    json.dumps(spec)  # must not raise: spec is a serialization surface
    assert spec["resolve"] == {"color": "shared", "legend": {"color": "independent"}}


def test_repeatchart_spec_flat_dict_resolve_unchanged(df_a):
    template = fm.Chart(df_a).mark_point().encode(x="x", y="y", color="g")
    flat = {"color": "shared"}
    rc = fm.RepeatChart(template, row=["x", "y"], column=["x", "y"], resolve=flat)
    assert rc.spec["resolve"] == {"color": "shared"}
    assert rc.spec["resolve"] is flat  # back-compat: pass-through, not a copy
