"""Regression guards for the scene-production consolidation (MOD-10, REND-04, REND-11, REND-12).

These tests pin the behavior-sensitive half of the render-surface consolidation:

- MOD-10: ``_render_scene`` relocated into the neutral ``ferrum._scene`` module
  and re-exported from ``ferrum._interactive``; ``display._render_scene_json``
  delegates to the single producer (no drift).
- REND-11: the duck-typed chart-vs-composite fork (``hasattr('_render_interactive')``)
  and the zoom-rebuild probe (``hasattr('_clone')``) are replaced by typed,
  ``runtime_checkable`` Protocols (``SceneRenderable`` / ``ZoomRebuildable``) that
  route identically for every real type.
- REND-04: the inline zero-row scene literal is extracted into the named
  ``_empty_scene_json`` helper, byte-identical to the prior literal.
- REND-12: ``Chart.interactive()`` returns an ``InteractiveChart`` (matching the
  corrected return annotation).

Everything here is behavior-preserving: the scene-json output is unchanged.
"""

from __future__ import annotations

import json

import polars as pl
import ferrum as fm
from ferrum._scene import (
    SceneRenderable,
    SupportsRender,
    ZoomRebuildable,
    _empty_scene_json,
    _render_scene,
)


def _point_chart() -> fm.Chart:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    return fm.Chart(df).mark_point().encode(x="x", y="y")


# ---------------------------------------------------------------------------
# REND-11 — typed routing parity (isinstance == old hasattr)
# ---------------------------------------------------------------------------


def test_scene_renderable_routing_matches_old_hasattr_fork():
    """Regression: REND-11 — isinstance(SceneRenderable) routes like hasattr('_render_interactive').

    A plain ``Chart`` is NOT a ``SceneRenderable`` (so ``_render_scene`` falls
    through to the Rust path) and a composite IS (so it routes to
    ``_render_interactive``).  The isinstance verdict must match the historical
    ``hasattr`` check for each real type, and ``_render_scene`` must produce a
    valid ``(scene_json, packed_data)`` pair for both.
    """
    chart = _point_chart()
    composite = fm.hconcat(chart, chart)

    # Plain Chart: not a SceneRenderable, lacks _render_interactive → Rust path.
    assert not isinstance(chart, SceneRenderable)
    assert not hasattr(chart, "_render_interactive")

    # Composite: is a SceneRenderable, has _render_interactive → composite path.
    assert isinstance(composite, SceneRenderable)
    assert hasattr(composite, "_render_interactive")

    # Both produce a valid (str, bytes) scene pair.
    for obj in (chart, composite):
        scene_json, packed = _render_scene(obj)
        assert isinstance(scene_json, str)
        assert isinstance(packed, bytes)
        json.loads(scene_json)  # must parse


def test_zoom_rebuildable_routing_matches_old_hasattr_clone():
    """Regression: REND-11 — isinstance(ZoomRebuildable) routes like hasattr('_clone').

    Plain ``Chart`` supports zoom rebuild (exposes ``_clone``); composites do
    not.  The typed ``ZoomRebuildable`` capability must agree with the prior
    ``hasattr('_clone')`` sniff for each real type.
    """
    chart = _point_chart()
    composite = fm.hconcat(chart, chart)

    # Plain Chart: rebuildable on zoom (has _clone).
    assert isinstance(chart, ZoomRebuildable) is True
    assert hasattr(chart, "_clone") is True

    # Composite: not rebuildable on zoom (no _clone).
    assert isinstance(composite, ZoomRebuildable) is False
    assert hasattr(composite, "_clone") is False


def test_supports_render_covers_chart_and_composite():
    """Regression: REND-11 — SupportsRender structurally matches every chart-like.

    The broad chart-like surface (``to_svg`` / ``to_png``) used to type the
    save/HTML/scene-producer entry points must accept both a plain ``Chart`` and
    a composite, so those call sites are meaningfully typed (not ``object``).
    """
    chart = _point_chart()
    composite = fm.hconcat(chart, chart)
    assert isinstance(chart, SupportsRender)
    assert isinstance(composite, SupportsRender)


# ---------------------------------------------------------------------------
# REND-04 — extracted empty-scene helper (byte-identical literal)
# ---------------------------------------------------------------------------


def test_empty_scene_json_has_expected_shape():
    """Regression: REND-04 — _empty_scene_json pins the hand-mirrored empty-scene shape.

    The zero-row scene dict must carry the documented top-level keys and the
    fully-spelled interaction block, and must round-trip as JSON.  A future
    schema drift in the hand-maintained mirror is caught here.
    """
    scene_json, packed = _empty_scene_json((400.0, 300.0))
    assert packed == b""

    parsed = json.loads(scene_json)
    assert set(parsed) == {
        "panels",
        "width",
        "height",
        "background",
        "title",
        "legend",
        "decorations",
        "selections",
        "interaction",
    }
    assert parsed["panels"] == []
    assert parsed["width"] == 400.0
    assert parsed["height"] == 300.0
    assert set(parsed["interaction"]) == {
        "zoom_enabled",
        "pan_enabled",
        "toolbar",
        "conditionals",
        "linked_panels",
        "tick_levels",
        "params",
        "param_bindings",
    }


def test_render_scene_zero_row_uses_empty_scene_helper():
    """Regression: REND-04 — _render_scene on a zero-row Chart emits the empty-scene literal.

    A Chart whose data resolves to zero rows must bypass the Rust renderer and
    return exactly what ``_empty_scene_json`` produces for the chart's viewport,
    byte-for-byte.
    """
    empty_df = pl.DataFrame(
        {"x": pl.Series("x", [], dtype=pl.Float64), "y": pl.Series("y", [], dtype=pl.Float64)}
    )
    chart = fm.Chart(empty_df).mark_point().encode(x="x", y="y")

    scene_json, packed = _render_scene(chart)
    assert packed == b""

    parsed = json.loads(scene_json)
    assert parsed["panels"] == []
    # The literal must match the helper for this chart's resolved viewport.
    expected_json, _ = _empty_scene_json((parsed["width"], parsed["height"]))
    assert scene_json == expected_json


# ---------------------------------------------------------------------------
# REND-12 — Chart.interactive() return type
# ---------------------------------------------------------------------------


def test_chart_interactive_returns_interactive_chart():
    """Regression: REND-12 — Chart.interactive() returns an InteractiveChart.

    The corrected return annotation must match runtime behavior: the call no
    longer returns a plain ``Chart`` clone but an ``InteractiveChart`` widget.
    """
    from ferrum._interactive import InteractiveChart

    widget = _point_chart().interactive()
    assert isinstance(widget, InteractiveChart)
    assert type(widget).__name__ == "InteractiveChart"


# ---------------------------------------------------------------------------
# MOD-10 — single producer, no drift between entry points
# ---------------------------------------------------------------------------


def test_single_scene_producer_no_drift():
    """Regression: MOD-10 — display._render_scene_json and _scene._render_scene agree.

    Both the public HTML/JSON entry point (``display._render_scene_json``) and
    the canonical producer (``_scene._render_scene``) must yield identical output
    for the same chart — there is exactly one scene producer, no parallel copy.
    """
    from ferrum.display import _render_scene_json

    chart = _point_chart()
    json_a, packed_a = _render_scene_json(chart)
    json_b, packed_b = _render_scene(chart)
    assert json_a == json_b
    assert packed_a == packed_b


def test_interactive_module_reexports_canonical_producer():
    """Regression: MOD-10 — ferrum._interactive re-exports the one _scene._render_scene.

    ``_render_scene`` was relocated into ``ferrum._scene``; importing it from
    ``ferrum._interactive`` (the historical home that existing tests and
    composition.py use) must resolve to the very same function object, proving
    there is a single producer rather than a re-implemented copy.
    """
    from ferrum import _interactive, _scene

    assert _interactive._render_scene is _scene._render_scene
