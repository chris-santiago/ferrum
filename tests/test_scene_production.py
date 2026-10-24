"""Regression guards for the scene-production consolidation (MOD-10, REND-04, REND-11, REND-12).

These tests pin the behavior-sensitive half of the render-surface consolidation:

- MOD-10: ``_render_scene`` relocated into the neutral ``ferrum._scene`` module
  and re-exported from ``ferrum._interactive``; ``display._render_scene_json``
  delegates to the single producer (no drift).
- REND-11: the duck-typed chart-vs-composite fork (``hasattr('_render_interactive')``)
  and the zoom-rebuild probe (``hasattr('_clone')``) are replaced by typed,
  ``runtime_checkable`` Protocols (``SceneRenderable`` / ``ZoomRebuildable``) that
  route identically for every real type.
- REND-12: ``Chart.interactive()`` returns an ``InteractiveChart`` (matching the
  corrected return annotation).
- A zero-row chart's scene is produced by the real Rust ``render_interactive``
  entry, like any other chart: one panel with no mark nodes, warning once,
  rather than a hand-mirrored empty-scene literal.

Everything here is behavior-preserving: the scene-json output is unchanged.
"""

from __future__ import annotations

import json

import polars as pl
import pytest
import ferrum as fm
from ferrum._scene import (
    SceneRenderable,
    SupportsRender,
    ZoomRebuildable,
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
# Zero-row scene: the real Rust render, not a hand-mirrored literal
# ---------------------------------------------------------------------------


def test_render_scene_zero_row_draws_its_frame():
    """A Chart whose data resolves to zero rows renders through the real Rust entry.

    It draws one panel with no mark nodes and warns once, the same way a
    filter-emptied chart already does, rather than bypassing the renderer
    for a hand-mirrored empty scene.
    """
    empty_df = pl.DataFrame(
        {"x": pl.Series("x", [], dtype=pl.Float64), "y": pl.Series("y", [], dtype=pl.Float64)}
    )
    chart = fm.Chart(empty_df).mark_point().encode(x="x", y="y")

    with pytest.warns(UserWarning, match="the chart has no rows to draw"):
        scene_json, packed = _render_scene(chart)

    parsed = json.loads(scene_json)
    assert len(parsed["panels"]) == 1
    assert sum(len(b["nodes"]) for b in parsed["panels"][0]["marks"]) == 0
    assert isinstance(packed, bytes)


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
