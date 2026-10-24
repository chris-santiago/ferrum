"""Chart → ``(scene_json, packed_data)`` producer for every interactive/HTML export path.

This module is the single, neutral home for turning any chart-like object into
the SceneGraph JSON + packed-instance buffer the WASM renderer consumes.  The
sole producer is :func:`_render_scene`; the interactive widget
(``ferrum._interactive``), the HTML/JSON export router (``ferrum.display``), and
the composite wrappers (``ferrum.composition`` — most composites route through
the Rust composite-render entry directly via their own ``_render_interactive``,
while ``LayerChart``'s merged single-panel path calls back into
:func:`_render_scene`) all route through it, so there is exactly one
chart→scene path with no parallel copies.

The render-contract Protocols (:class:`SupportsRender`, :class:`SceneRenderable`,
:class:`ZoomRebuildable`) are co-located here so the capability checks that drive
scene production are declared once, typed, and runtime-checkable rather than
duck-typed at each call site.

``_scene`` sits below ``_interactive``/``display``/``composition`` in the import
graph: it depends only on the Rust ``render_interactive`` binding (imported
function-locally) and the chart's own methods, never on the modules that import
it — so there is no cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ferrum.chart import Chart


@runtime_checkable
class SupportsRender(Protocol):
    """The chart-like rendering surface every export path consumes.

    Both ``Chart`` and every composition wrapper (``HConcatChart`` /
    ``VConcatChart`` / ``JointChart`` / ``LayerChart`` / ...) satisfy this
    structurally: they expose ``to_svg()`` and ``to_png(scale=...)``.  It types
    the chart-like parameters of the save/HTML/scene-producer entry points so
    those call sites are no longer typed ``object``.  The scene-json producers
    additionally dispatch on :class:`SceneRenderable` to reach a composite's
    ``_render_interactive`` (which is not part of this base surface because plain
    ``Chart`` does not implement it).

    Marked ``runtime_checkable`` for parity with the sibling protocols; the
    export code relies on it as a static contract first.
    """

    def to_svg(self) -> str: ...

    def to_png(self, *, scale: float = ...) -> bytes: ...


@runtime_checkable
class SceneRenderable(Protocol):
    """Composite-chart capability: builds its own interactive scene.

    Composition types (``HConcatChart`` etc.) implement
    ``_render_interactive()`` to merge their child scenes; plain ``Chart`` does
    not and instead drives the Rust ``render_interactive`` through
    :func:`_render_scene`.  Branching :func:`_render_scene` on
    ``isinstance(chart, SceneRenderable)`` is behavior-identical to the prior
    ``hasattr(chart, "_render_interactive")`` sniff: ``isinstance`` against a
    ``runtime_checkable`` Protocol checks the same attribute presence, so
    composites match and plain ``Chart`` falls through to the Rust path exactly
    as before.
    """

    def _render_interactive(self) -> tuple[str, bytes]: ...


@runtime_checkable
class ZoomRebuildable(Protocol):
    """Chart capability: can be cloned for a domain-overridden zoom rebuild.

    Plain ``Chart`` exposes ``_clone()`` (used to apply per-panel xlim/ylim
    overrides on a fresh copy); compositions do not, so they cannot be rebuilt
    in place on zoom.  ``isinstance(chart, ZoomRebuildable)`` is
    behavior-identical to the prior ``hasattr(chart, "_clone")`` check.
    """

    def _clone(self) -> Any: ...


def _render_scene(chart: "Chart | SceneRenderable") -> tuple[str, bytes]:
    """Return (scene_json, packed_bytes) for the interactive renderer.

    Dispatches to ``_render_interactive()`` when the object is a
    composition type (HConcat, VConcat, Layer, etc.), and falls through
    to the standard Rust ``render_interactive`` for plain ``Chart``. Zero-row
    data renders through this same Rust call like any other chart: it draws
    its frame and axes and warns once, the same way a filter-emptied chart
    already does, rather than a hand-mirrored empty scene.
    """
    # Composition types implement _render_interactive(); plain Charts do not.
    if isinstance(chart, SceneRenderable):
        return chart._render_interactive()

    from ferrum._core import render_interactive

    spec, data, viewport, theme_dict, chart_config_dict = chart._render_inputs(_auto_tooltips=True)
    return render_interactive(
        spec,
        data,
        viewport=viewport,
        theme=theme_dict,
        chart_config=chart_config_dict or None,
    )
