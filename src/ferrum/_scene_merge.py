"""Scene-graph merge layer (composition → WASM renderer).

Extracted from ``composition.py``: the declarative composite-chart classes
(``HConcatChart`` etc.) live there, while this module owns the low-level
scene-graph merge / offset / packed-byte manipulation that composes several
rendered child scenes into one interactive scene + packed-instance buffer.

The composite classes import the public ``_merge_child_scenes*`` /
``_render_single_with_figure_chrome`` entry points from here; this module never
imports ``composition`` at module level.  The only cross-module dependency is
the renderer entry point (``ferrum._interactive._render_scene``) and the Rust
``figure_title_nodes`` helper, both imported function-locally to avoid an import
cycle.

The four grid variants (linear, wrapping-grid, sparse-grid, nonuniform-grid)
share one placement pipeline: each computes a list of :class:`_PlacedChild`
plus the composed ``(width, height)`` via its own placement strategy, then
:func:`_assemble_placed_children` merges them into the final scene + packed
buffer.  The per-strategy placement math is intentionally kept byte-for-byte
identical to the pre-split implementation.
"""

from __future__ import annotations

import copy
import json as _json
import struct
from dataclasses import dataclass
from typing import Optional
from typing import TypedDict

# ---------------------------------------------------------------------------
# Shared offset key-set constants
# ---------------------------------------------------------------------------

# Keys for the two area sub-dicts in a panel node (x/y are offset directly).
_PANEL_AREA_KEYS: tuple[str, ...] = ("plot_area", "clip")

# Keys for per-node lists inside a panel (each element passed to _offset_node).
_PANEL_NODE_LIST_KEYS: tuple[str, ...] = ("axes", "grid", "annotations", "strip_title")

# Keys for per-node lists in the outer scene (title / legend / decorations).
_OUTER_NODE_LIST_KEYS: tuple[str, ...] = ("title", "legend", "decorations")


class _FigureChrome(TypedDict):
    """Figure-level chrome payload threaded through scene-merge helpers.

    Produced by :meth:`_CompositeBase._figure_chrome_kwargs` and consumed by
    :func:`_inject_figure_chrome` (via every ``_merge_child_scenes*`` function
    and :func:`_render_single_with_figure_chrome`).  Using a ``TypedDict`` makes
    the key contract explicit and prevents the dict from silently drifting.
    """

    title: Optional[str]
    subtitle: Optional[str]
    caption: Optional[str]
    chrome: dict


# ---------------------------------------------------------------------------
# Placement records + the shared assemble / render prelude
# ---------------------------------------------------------------------------


@dataclass
class _PlacedChild:
    """One child's rendered scene + packed bytes at its computed placement.

    Carries the per-child ``(panel_id_offset, dx, dy, scene, packed)`` placement
    as a single record so the packed-instance offset cannot drift from the
    scene-node offset (both read ``dx``/``dy`` from the same field).  Built by
    each ``_merge_child_scenes*`` variant's placement loop and consumed by
    :func:`_assemble_placed_children`.
    """

    scene: dict  # parsed child scene JSON
    packed: bytes  # child packed GPU-instance bytes
    dx: float  # lateral placement offset
    dy: float  # vertical placement offset
    panel_id_offset: int  # cumulative panel-id base for this child


def _assemble_placed_children(
    placed: list["_PlacedChild"],
    width: float,
    height: float,
    figure_chrome: Optional["_FigureChrome"],
) -> tuple[str, bytes]:
    """Merge placed children into one scene + packed buffer.

    Builds a fresh :func:`_empty_scene`, sets *width*/*height*, runs
    :func:`_merge_one_child` for each placement in order, injects figure chrome
    (only when *figure_chrome* is not ``None``, after the merge loop), then
    merges packed bytes with ``y_offset = chrome_top_h``.  Returns
    ``(scene_json, packed)``.

    Only ever called on the non-empty path; each variant keeps its own
    empty-children early-return guard.
    """
    merged = _empty_scene()
    merged["width"] = width
    merged["height"] = height

    for pc in placed:
        _merge_one_child(merged, pc.scene, pc.dx, pc.dy, pc.panel_id_offset)

    # chrome_top_h = title+subtitle band height; chrome_bottom_h = caption band height;
    # together they are the figure chrome.
    chrome_top_h = 0.0
    if figure_chrome is not None:
        chrome_top_h = _inject_figure_chrome(merged, **figure_chrome)

    merged_packed = _merge_packed_data(
        [pc.packed for pc in placed],
        [pc.panel_id_offset for pc in placed],
        [(pc.dx, pc.dy) for pc in placed],
        y_offset=chrome_top_h,
    )
    return _json.dumps(merged), merged_packed


def _render_charts(charts: list) -> list[tuple[dict, bytes]]:
    """Render each chart and return a list of ``(scene_dict, packed)`` pairs.

    Shared by the four ``_merge_child_scenes*`` helpers so the
    ``_render_scene`` + ``_json.loads`` prelude is not copy-pasted.
    The caller is responsible for its own empty-input guard (COMP-03).
    """
    from ferrum._interactive import _render_scene

    return [
        (_json.loads(scene_json), packed)
        for scene_json, packed in (_render_scene(c) for c in charts)
    ]


# ---------------------------------------------------------------------------
# Placement strategies
#
# Each strategy turns a list of rendered children into a (placed, width, height)
# triple; the public _merge_child_scenes* wrappers funnel that through the shared
# empty-guard + _assemble_placed_children.  Each strategy's math is preserved
# byte-for-byte from the pre-split implementation.
# ---------------------------------------------------------------------------


def _place_linear(
    rendered: list[tuple[dict, bytes]],
    spacing: float,
    layout: str,
) -> tuple[list["_PlacedChild"], float, float]:
    """Linear horizontal/vertical placement (HConcat / VConcat / Concat row)."""
    x_offset = 0.0
    y_offset = 0.0
    panel_id_offset = 0
    width = 0
    height = 0
    placed: list[_PlacedChild] = []

    for scene, packed in rendered:
        dx = x_offset if layout == "horizontal" else 0.0
        dy = y_offset if layout == "vertical" else 0.0
        placed.append(_PlacedChild(scene, packed, dx, dy, panel_id_offset))
        panel_id_offset += len(scene.get("panels", []))

        w = scene.get("width", 0)
        h = scene.get("height", 0)
        if layout == "horizontal":
            x_offset += w + spacing
            width = x_offset - spacing
            height = max(height, h)
        else:
            y_offset += h + spacing
            height = y_offset - spacing
            width = max(width, w)

    return placed, width, height


def _place_wrapping_grid(
    rendered: list[tuple[dict, bytes]],
    spacing: float,
    columns: int,
) -> tuple[list["_PlacedChild"], float, float]:
    """Wrapping-grid placement: rows of *columns* charts, stacked vertically."""
    columns = max(1, columns)

    # Partition into rows.
    rows: list[list[tuple[dict, bytes]]] = []
    for i in range(0, len(rendered), columns):
        rows.append(rendered[i : i + columns])

    # Place each row horizontally, then stack rows vertically.
    y_offset = 0.0
    panel_id_offset = 0
    width = 0
    placed: list[_PlacedChild] = []

    for row in rows:
        row_width = 0.0
        row_height = 0.0
        x_offset = 0.0

        for scene, packed in row:
            placed.append(_PlacedChild(scene, packed, x_offset, y_offset, panel_id_offset))
            panel_id_offset += len(scene.get("panels", []))

            w = scene.get("width", 0)
            h = scene.get("height", 0)
            x_offset += w + spacing
            row_width = x_offset - spacing
            row_height = max(row_height, h)

        width = max(width, row_width)
        y_offset += row_height + spacing

    height = y_offset - spacing

    return placed, width, height


def _place_sparse_grid(
    rendered: list[tuple[int, int, dict, bytes]],
    spacing: float,
) -> tuple[list["_PlacedChild"], float, float]:
    """Sparse-grid placement: explicit (row, col), uniform panel size."""
    # Uniform panel dimensions (max across all children).
    panel_w = max(s.get("width", 0) for _, _, s, _ in rendered)
    panel_h = max(s.get("height", 0) for _, _, s, _ in rendered)

    # Grid extents.
    max_row = max(r for r, _, _, _ in rendered)
    max_col = max(c for _, c, _, _ in rendered)
    n_rows = max_row + 1
    n_cols = max_col + 1

    # Place each panel at its (row, col) position.
    panel_id_offset = 0
    placed: list[_PlacedChild] = []

    for row_idx, col_idx, scene, packed in rendered:
        dx = col_idx * (panel_w + spacing)
        dy = row_idx * (panel_h + spacing)
        placed.append(_PlacedChild(scene, packed, dx, dy, panel_id_offset))
        panel_id_offset += len(scene.get("panels", []))

    width = n_cols * panel_w + (n_cols - 1) * spacing
    height = n_rows * panel_h + (n_rows - 1) * spacing

    return placed, width, height


def _place_nonuniform_grid(
    rendered: list[tuple[int, int, dict, bytes]],
    spacing: float,
) -> tuple[list["_PlacedChild"], float, float]:
    """Nonuniform-grid placement: explicit (row, col), per-row/per-col native size."""
    # Compute per-row heights and per-column widths.
    row_heights: dict[int, float] = {}
    col_widths: dict[int, float] = {}
    for row_idx, col_idx, scene, _packed in rendered:
        h = scene.get("height", 0)
        w = scene.get("width", 0)
        row_heights[row_idx] = max(row_heights.get(row_idx, 0), h)
        col_widths[col_idx] = max(col_widths.get(col_idx, 0), w)

    # Compute cumulative offsets for each row/col.
    sorted_rows = sorted(row_heights)
    sorted_cols = sorted(col_widths)
    row_y: dict[int, float] = {}
    y = 0.0
    for i, r in enumerate(sorted_rows):
        row_y[r] = y
        y += row_heights[r] + (spacing if i < len(sorted_rows) - 1 else 0)
    total_height = y

    col_x: dict[int, float] = {}
    x = 0.0
    for i, c in enumerate(sorted_cols):
        col_x[c] = x
        x += col_widths[c] + (spacing if i < len(sorted_cols) - 1 else 0)
    total_width = x

    # Place each panel at its computed position.
    panel_id_offset = 0
    placed: list[_PlacedChild] = []

    for row_idx, col_idx, scene, packed in rendered:
        dx = col_x[col_idx]
        dy = row_y[row_idx]
        placed.append(_PlacedChild(scene, packed, dx, dy, panel_id_offset))
        panel_id_offset += len(scene.get("panels", []))

    return placed, total_width, total_height


def _render_grid_panels(
    panels: list[tuple[int, int, object]],
) -> list[tuple[int, int, dict, bytes]]:
    """Render ``(row, col, chart)`` triples, preserving their grid coordinates."""
    row_cols = [(r, c) for r, c, _ in panels]
    charts = [chart for _, _, chart in panels]
    scenes_packed = _render_charts(charts)
    return [(r, c, scene, packed) for (r, c), (scene, packed) in zip(row_cols, scenes_packed)]


# ---------------------------------------------------------------------------
# Public scene-merge entry points (thin wrappers over the strategies above)
# ---------------------------------------------------------------------------


def _merge_child_scenes(
    charts: list,
    spacing: float,
    layout: str = "horizontal",
    *,
    figure_chrome: Optional["_FigureChrome"] = None,
) -> tuple[str, bytes]:
    """Render each child chart and merge their scene JSONs.

    Parameters
    ----------
    charts : list
        Child charts to render.
    spacing : float
        Pixel gap between charts.
    layout : ``"horizontal"`` or ``"vertical"``
    figure_chrome : dict, optional
        Figure-level chrome to inject as an on-canvas title band, with keys
        ``title`` / ``subtitle`` / ``caption`` and a ``chrome`` positioning
        sub-dict (see :func:`_inject_figure_chrome`).  ``None`` injects no band.

    Returns
    -------
    tuple[str, bytes]
        ``(merged_scene_json, merged_packed_data)``
    """
    rendered = _render_charts(charts)

    if not rendered:
        return _EMPTY_SCENE_JSON, b""

    placed, width, height = _place_linear(rendered, spacing, layout)
    return _assemble_placed_children(placed, width, height, figure_chrome)


def _merge_child_scenes_grid(
    charts: list,
    spacing: float,
    columns: int,
    *,
    figure_chrome: Optional["_FigureChrome"] = None,
) -> tuple[str, bytes]:
    """Render child charts in a wrapping grid layout.

    Arranges charts left-to-right, wrapping to the next row after
    *columns* charts.  Each row is merged horizontally, then rows
    are merged vertically.

    Parameters
    ----------
    charts : list
        Child charts to render.
    spacing : float
        Pixel gap between charts.
    columns : int
        Number of columns before wrapping.
    figure_chrome : dict, optional
        Figure-level chrome band to inject (see :func:`_inject_figure_chrome`).

    Returns
    -------
    tuple[str, bytes]
        ``(merged_scene_json, merged_packed_data)``
    """
    if not charts:
        return _EMPTY_SCENE_JSON, b""

    rendered = _render_charts(charts)
    placed, width, height = _place_wrapping_grid(rendered, spacing, columns)
    return _assemble_placed_children(placed, width, height, figure_chrome)


def _merge_child_scenes_sparse_grid(
    panels: list[tuple[int, int, object]],
    spacing: float,
    *,
    figure_chrome: Optional["_FigureChrome"] = None,
) -> tuple[str, bytes]:
    """Render child charts in a sparse grid layout (for corner-mode repeat).

    Each panel carries explicit ``(row, col)`` grid coordinates.  Panels are
    positioned at ``(col * panel_w + col * spacing, row * panel_h + row * spacing)``
    using uniform panel dimensions (the max width/height across all children).
    Grid positions without a panel (upper triangle in corner mode) are left empty.

    Parameters
    ----------
    panels : list of (row, col, chart)
        Each element is a ``(row_index, col_index, chart)`` triple.
    spacing : float
        Pixel gap between adjacent panels.
    figure_chrome : dict, optional
        Figure-level chrome band to inject (see :func:`_inject_figure_chrome`).

    Returns
    -------
    tuple[str, bytes]
        ``(merged_scene_json, merged_packed_data)``
    """
    if not panels:
        return _EMPTY_SCENE_JSON, b""

    rendered = _render_grid_panels(panels)
    placed, width, height = _place_sparse_grid(rendered, spacing)
    return _assemble_placed_children(placed, width, height, figure_chrome)


def _merge_child_scenes_nonuniform_grid(
    panels: list[tuple[int, int, object]],
    spacing: float,
    *,
    figure_chrome: Optional["_FigureChrome"] = None,
) -> tuple[str, bytes]:
    """Render child charts in a sparse grid with per-row/per-column sizing.

    Unlike ``_merge_child_scenes_sparse_grid`` which uses uniform panel
    dimensions, this variant computes the maximum width per column and
    maximum height per row, so that differently-sized children (e.g.
    marginal plots next to a center plot) occupy only as much space as
    they need.

    Parameters
    ----------
    panels : list of (row, col, chart)
        Each element is a ``(row_index, col_index, chart)`` triple.
    spacing : float
        Pixel gap between adjacent panels.
    figure_chrome : dict, optional
        Figure-level chrome band to inject (see :func:`_inject_figure_chrome`).

    Returns
    -------
    tuple[str, bytes]
        ``(merged_scene_json, merged_packed_data)``
    """
    if not panels:
        return _EMPTY_SCENE_JSON, b""

    rendered = _render_grid_panels(panels)
    placed, total_width, total_height = _place_nonuniform_grid(rendered, spacing)
    return _assemble_placed_children(placed, total_width, total_height, figure_chrome)


# ---------------------------------------------------------------------------
# Per-child merge + figure chrome
# ---------------------------------------------------------------------------


def _merge_one_child(
    merged: dict,
    scene: dict,
    dx: float,
    dy: float,
    panel_id_offset: int,
) -> int:
    """Merge a single child scene into *merged* at the given offset.

    Handles panels, selections, interaction conditionals, tick_levels,
    linked_panels, zoom_enabled/pan_enabled propagation,
    background, and title/legend/decoration nodes.

    Returns
    -------
    int
        The number of panels merged (so the caller can update
        ``panel_id_offset``).
    """
    _merge_scene_panels(merged, scene, dx, dy, panel_id_offset)
    n_panels = len(scene.get("panels", []))

    merged["selections"].extend(scene.get("selections", []))
    child_interaction = scene.get("interaction", {})
    merged["interaction"]["conditionals"].extend(child_interaction.get("conditionals", []))
    for tl in child_interaction.get("tick_levels", []):
        tl_copy = dict(tl)
        tl_copy["panel_id"] = tl_copy.get("panel_id", 0) + panel_id_offset
        merged["interaction"]["tick_levels"].append(tl_copy)
    for lp in child_interaction.get("linked_panels", []):
        merged["interaction"]["linked_panels"].append([p + panel_id_offset for p in lp])
    if not child_interaction.get("zoom_enabled", True):
        merged["interaction"]["zoom_enabled"] = False
    if not child_interaction.get("pan_enabled", True):
        merged["interaction"]["pan_enabled"] = False
    existing_param_names = {p["name"] for p in merged["interaction"]["params"]}
    for param in child_interaction.get("params", []):
        if param["name"] not in existing_param_names:
            merged["interaction"]["params"].append(param)
            existing_param_names.add(param["name"])
    existing_binding_keys = {
        (b["param"], b["role"], b.get("panel"), b.get("channel"))
        for b in merged["interaction"]["param_bindings"]
    }
    for binding in child_interaction.get("param_bindings", []):
        b = dict(binding)
        if b.get("panel") is not None:
            b["panel"] = b["panel"] + panel_id_offset
        key = (b["param"], b["role"], b.get("panel"), b.get("channel"))
        if key not in existing_binding_keys:
            merged["interaction"]["param_bindings"].append(b)
            existing_binding_keys.add(key)
    if merged["background"] is None and scene.get("background"):
        merged["background"] = scene["background"]

    # Offset and merge outer-level nodes (title, legend, decorations).
    for key in _OUTER_NODE_LIST_KEYS:
        for node in scene.get(key, []):
            n = copy.deepcopy(node)
            _offset_node(n, dx, dy)
            merged[key].append(n)

    return n_panels


def _inject_figure_chrome(
    merged: dict,
    *,
    title: Optional[str],
    subtitle: Optional[str],
    caption: Optional[str],
    chrome: dict,
) -> float:  # called via **figure_chrome (_FigureChrome unpacked)
    """Inject a figure-level title / subtitle / caption band into *merged*.

    This is the **single** shared implementation of the interactive on-canvas
    figure-chrome band, called by every composite scene-merge function so the
    band renders identically for HConcat / VConcat / Concat / Joint /
    ClusterMap / Repeat.  It is the interactive counterpart of the SVG band the
    Rust ``compose_svg_*`` compositors emit: the layout math (band heights,
    node x / y, anchor) lives in the Rust ``figure_title_nodes`` helper; this
    function only injects the returned nodes and offsets / grows the merged
    scene by the returned band heights.

    When no chrome text is present this is a no-op, so a composite with no
    figure title produces a byte-identical merged scene to before.

    The ``width`` / ``height`` passed to ``figure_title_nodes`` are the merged
    panels' pre-chrome bounding box (``merged["width"]`` / ``merged["height"]``
    as set by the caller before this runs).  The title/subtitle chrome-top band is
    positioned identically to SVG for all composites.  The caption (chrome-bottom)
    absolute y matches SVG for the concat family (HConcat / VConcat / Concat /
    Repeat), where the interactive body height equals the SVG body height.  For
    JointChart and ClusterMapChart the interactive body is native panel size
    rather than the ratio-scaled viewBox the SVG path uses, so the caption y
    will differ from the SVG (pre-existing W5 limitation — interactive
    nonuniform-grid layout is flat horizontal, not a 2×2 grid).

    Parameters
    ----------
    merged : dict
        The merged scene dict, with ``width`` / ``height`` already set to the
        composed panels' pre-chrome bounding box.  Mutated in place.
    title, subtitle, caption : str or None
        Figure-level chrome text.  All ``None`` -> no-op.
    chrome : dict
        Positioning kwargs (``left_inset`` / ``right_inset`` / ``anchor``)
        resolved from the composite's configure layers, matching the SVG path.

    Returns
    -------
    float
        The chrome-top band height (``chrome_top_h``, title+subtitle) applied to
        shift every panel's scene nodes down.  Returned so the caller can apply
        the **same** downward shift to the panels' packed GPU-instance bytes
        (which live in a separate binary sidecar, not in *merged*), keeping packed
        marks aligned with the shifted plot_area / axes.  ``0.0`` when no chrome
        text is present or the band has no top (caption-only), in which case no
        offset is applied anywhere.
    """
    if title is None and subtitle is None and caption is None:
        return 0.0

    from ferrum._core import figure_title_nodes

    panel_w = merged.get("width", 0) or 0.0
    panel_h = merged.get("height", 0) or 0.0
    # chrome_top_h = title+subtitle band height; chrome_bottom_h = caption band height;
    # together they are the figure chrome.
    nodes_json, chrome_top_h, chrome_bottom_h = figure_title_nodes(
        width=float(panel_w),
        height=float(panel_h),
        title=title,
        subtitle=subtitle,
        caption=caption,
        **chrome,
    )

    # Offset every child scene node DOWN by the chrome-top band height so the
    # panels sit below the title/subtitle.  The chrome nodes themselves are
    # already in outer-canvas space (caption y already includes chrome_top_h +
    # panel_h) and must NOT be offset.
    if chrome_top_h:
        for panel in merged.get("panels", []):
            for area_key in _PANEL_AREA_KEYS:
                area = panel.get(area_key)
                if area is not None:
                    area["y"] = area.get("y", 0) + chrome_top_h
            for batch in panel.get("marks", []):
                for node in batch.get("nodes", []):
                    _offset_node(node, 0.0, chrome_top_h)
            for key in _PANEL_NODE_LIST_KEYS:
                for node in panel.get(key, []):
                    _offset_node(node, 0.0, chrome_top_h)
        for key in _OUTER_NODE_LIST_KEYS:
            for node in merged.get(key, []):
                _offset_node(node, 0.0, chrome_top_h)

    # Inject the chrome nodes (already absolute) into the merged title list.
    # Note: for JointChart/ClusterMapChart the caption y is relative to the
    # interactive body height, which differs from the SVG ratio-viewBox body
    # (W5 limitation — interactive nonuniform-grid layout).
    merged.setdefault("title", []).extend(_json.loads(nodes_json))

    # Grow the merged canvas to fit the chrome-top + chrome-bottom bands
    # (width unchanged).
    merged["height"] = panel_h + chrome_top_h + chrome_bottom_h

    # Report the chrome-top shift so the caller can apply the identical downward
    # offset to the packed GPU-instance bytes (see _merge_packed_data).
    return float(chrome_top_h)


def _render_single_with_figure_chrome(chart, figure_chrome: "_FigureChrome") -> tuple[str, bytes]:
    """Render one chart's scene and wrap it in the figure-chrome band.

    Used by the asymmetric composites (Joint / ClusterMap) on their
    single-panel fast path (no marginals / dendrograms), where the body is a
    lone child scene rather than a merged grid.  When *figure_chrome* carries
    no title text this returns the child scene unchanged (byte-identical to
    ``_render_scene``), preserving backward compatibility.
    """
    from ferrum._interactive import _render_scene

    scene_json, packed = _render_scene(chart)
    if (
        figure_chrome.get("title") is None
        and figure_chrome.get("subtitle") is None
        and figure_chrome.get("caption") is None
    ):
        return scene_json, packed

    scene = _json.loads(scene_json)
    chrome_top_h = _inject_figure_chrome(scene, **figure_chrome)
    # Shift the packed GPU instances down by the chrome-top band so they stay
    # aligned with the scene nodes _inject_figure_chrome just shifted.  No
    # panel-id remap here (single child) and no lateral placement offset,
    # so child_xy is [(0.0, 0.0)].
    if chrome_top_h:
        packed = _merge_packed_data([packed], [0], [(0.0, 0.0)], y_offset=chrome_top_h)
    return _json.dumps(scene), packed


# ---------------------------------------------------------------------------
# Empty-scene skeleton + canonical empty-merge literal
# ---------------------------------------------------------------------------


def _empty_scene() -> dict:
    """Return a skeleton scene dict for merging."""
    return {
        "width": 0,
        "height": 0,
        "background": None,
        "title": [],
        "panels": [],
        "legend": [],
        "decorations": [],
        "selections": [],
        "interaction": {
            "zoom_enabled": True,
            "pan_enabled": True,
            "conditionals": [],
            "linked_panels": [],
            "tick_levels": [],
            "params": [],
            "param_bindings": [],
        },
    }


# Canonical empty-merge return value: (scene_json, packed_bytes).
# Used by all four _merge_child_scenes* helpers on their empty-input early-return
# so the empty case shares the full scene schema with _empty_scene().
_EMPTY_SCENE_JSON: str = _json.dumps(_empty_scene())


# ---------------------------------------------------------------------------
# Scene-node + panel offsetting
# ---------------------------------------------------------------------------


def _merge_scene_panels(
    merged: dict,
    scene: dict,
    dx: float,
    dy: float,
    panel_id_offset: int,
) -> None:
    """Offset and append panels from *scene* into *merged*.

    Each panel is deep-copied before mutation so the original *scene*
    dict is not modified in place — callers may re-read it (e.g.
    ``_merge_one_child`` counts ``scene.get("panels", [])`` after this
    call returns).
    """
    for panel in scene.get("panels", []):
        panel = copy.deepcopy(panel)
        panel["id"] = panel.get("id", 0) + panel_id_offset

        for area_key in _PANEL_AREA_KEYS:
            area = panel.get(area_key, {})
            area["x"] = area.get("x", 0) + dx
            area["y"] = area.get("y", 0) + dy

        for batch in panel.get("marks", []):
            for node in batch.get("nodes", []):
                _offset_node(node, dx, dy)
        for key in _PANEL_NODE_LIST_KEYS:
            for node in panel.get(key, []):
                _offset_node(node, dx, dy)

        merged["panels"].append(panel)


def _offset_node(node: dict, dx: float, dy: float) -> None:
    """Offset a scene node's position by ``(dx, dy)``."""
    if dx == 0.0 and dy == 0.0:
        return
    t = node.get("type")
    if t == "circle":
        node["cx"] = node.get("cx", 0) + dx
        node["cy"] = node.get("cy", 0) + dy
    elif t == "rect":
        node["x"] = node.get("x", 0) + dx
        node["y"] = node.get("y", 0) + dy
    elif t == "line":
        node["x1"] = node.get("x1", 0) + dx
        node["y1"] = node.get("y1", 0) + dy
        node["x2"] = node.get("x2", 0) + dx
        node["y2"] = node.get("y2", 0) + dy
    elif t == "text":
        node["x"] = node.get("x", 0) + dx
        node["y"] = node.get("y", 0) + dy
    elif t == "path":
        for cmd in node.get("commands", []):
            for xkey in ("x", "cx", "c1x", "c2x"):
                if xkey in cmd:
                    cmd[xkey] = cmd[xkey] + dx
            for ykey in ("y", "cy", "c1y", "c2y"):
                if ykey in cmd:
                    cmd[ykey] = cmd[ykey] + dy
    elif t == "image":
        node["x"] = node.get("x", 0) + dx
        node["y"] = node.get("y", 0) + dy
    elif t == "polygon":
        for ring in node.get("rings", []):
            for pt in ring:
                pt[0] += dx
                pt[1] += dy
    elif t == "polyline":
        for pt in node.get("points", []):
            pt[0] += dx
            pt[1] += dy
    elif t == "group":
        for child in node.get("children", []):
            _offset_node(child, dx, dy)


# ---------------------------------------------------------------------------
# Packed GPU-instance merging
# ---------------------------------------------------------------------------

# Packed GPU-instance layout (mirrors ferrum-core pack_instances.rs and the
# CircleInstance / RectInstance structs in ferrum-wasm scene_load.rs).
#   kind 0 (CircleInstance): 16 f32 = 64-byte stride; cx is f32[0] (byte 0),
#                             cy is f32[1] (byte 4).
#   kind 1 (RectInstance):   18 f32 = 72-byte stride; position_x is f32[0]
#                             (byte 0), position_y is f32[1] (byte 4).
# X is the first f32 (byte 0) and Y is the second f32 (byte 4) in both
# records.  Composition translates both fields by the per-panel (dx, dy)
# placement offset so packed instances land in the same absolute frame as
# their plot_area and scene-graph nodes.
_PACKED_INSTANCE_SIZES = {0: 64, 1: 72}  # kind -> sizeof(Instance)
_PACKED_X_FIELD_OFFSET = 0  # byte offset of the X f32 within each instance
_PACKED_Y_FIELD_OFFSET = 4  # byte offset of the Y f32 within each instance


def _offset_packed_batch_xy(
    buf: bytearray, instance_start: int, kind: int, count: int, dx: float, dy: float
) -> None:
    """Add *(dx, dy)* to the X and Y f32 of every instance in one packed batch.

    *buf* is the assembled output buffer; *instance_start* is the byte index in
    *buf* where this batch's instance data begins (just past its 20-byte
    header).  For both ``CircleInstance`` and ``RectInstance``, X is the first
    f32 (``_PACKED_X_FIELD_OFFSET`` = 0) and Y is the second f32
    (``_PACKED_Y_FIELD_OFFSET`` = 4).  When *dx* or *dy* is zero, that axis is
    skipped so callers that only shift one axis incur no extra writes.
    """
    stride = _PACKED_INSTANCE_SIZES[kind]
    for i in range(count):
        base = instance_start + i * stride
        if dx != 0.0:
            x_pos = base + _PACKED_X_FIELD_OFFSET
            (x,) = struct.unpack_from("<f", buf, x_pos)
            struct.pack_into("<f", buf, x_pos, x + dx)
        if dy != 0.0:
            y_pos = base + _PACKED_Y_FIELD_OFFSET
            (y,) = struct.unpack_from("<f", buf, y_pos)
            struct.pack_into("<f", buf, y_pos, y + dy)


def _merge_packed_data(
    packed_list: list[bytes],
    panel_id_offsets: list[int],
    child_xy_offsets: list[tuple[float, float]],
    *,
    y_offset: float = 0.0,
) -> bytes:
    """Merge packed binary data from multiple child scenes.

    Rewrites the ``panel_idx`` field in each batch's 20-byte header to
    account for the panel-id offset of each child in the composed layout.
    Also translates every packed GPU instance by the per-child composition
    offset *(child_dx, child_dy)* plus the global figure-title band
    *y_offset*, so packed (>1000-mark) circle/rect batches stay aligned with
    the scene-graph nodes and ``plot_area`` rects that
    :func:`_merge_scene_panels` and :func:`_inject_figure_chrome` place in
    the same absolute composite frame.

    When both the per-child dx/dy and the global *y_offset* are zero the
    instance bytes are left byte-identical to the unmerged child (only
    ``panel_idx`` in the header changes).

    Binary format per batch::

        [header 20B][instance_data][data_indices?][tooltips?]

    Header layout (20 bytes, all u32 little-endian)::

        [panel_idx][batch_idx][kind][count][flags]

    After the header:
    - Instance data: ``count * 64`` bytes for kind=0 (CircleInstance),
      ``count * 72`` bytes for kind=1 (RectInstance).
    - If ``flags & 0x2``: ``count * 4`` bytes of u32 data indices.
    - If ``flags & 0x1``: tooltip string table in the format produced by
      ``pack_instances.rs::pack_tooltips``:

      .. code-block:: text

          [num_fields: u32]
          num_fields   × [name_len: u32][name UTF-8 bytes]
          count×num_fields × [value_len: u32][value UTF-8 bytes]

      There is NO overall byte-length prefix.  ``count`` is the instance
      count from the batch header (field index 3).

    Parameters
    ----------
    packed_list : list[bytes]
        Packed binary data from each child scene.
    panel_id_offsets : list[int]
        Cumulative panel-id offset for each child (same length as
        *packed_list*).
    child_xy_offsets : list[tuple[float, float]]
        Per-child ``(dx, dy)`` panel-grid placement offsets (same length as
        *packed_list*).  These are the same offsets passed to
        :func:`_merge_scene_panels` for the corresponding child.  A child at
        ``(0.0, 0.0)`` with *y_offset* == 0.0 stays byte-identical.
    y_offset : float, default 0.0
        Global pixels to add to every packed instance's Y coordinate.  This
        is the ``chrome_top_h`` returned by :func:`_inject_figure_chrome` for a
        titled composite; ``0.0`` for a non-titled composite.
    """
    result = bytearray()
    for packed, offset, (child_dx, child_dy) in zip(
        packed_list, panel_id_offsets, child_xy_offsets
    ):
        if not packed:
            continue
        pos = 0
        while pos + 20 <= len(packed):
            panel_idx, batch_idx, kind, count, flags = struct.unpack_from("<5I", packed, pos)
            if kind not in _PACKED_INSTANCE_SIZES:
                break  # unknown kind — stop parsing this child

            # Rewrite panel_idx with the composition offset
            header = struct.pack("<5I", panel_idx + offset, batch_idx, kind, count, flags)

            batch_end = pos + 20 + count * _PACKED_INSTANCE_SIZES[kind]

            if flags & 0x2:
                batch_end += count * 4  # u32 data indices

            if flags & 0x1:
                # Scan the tooltip string table field-by-field, mirroring
                # scene_load.rs::unpack_binary_instances (lines ~708-735).
                # Format: [num_fields:u32] + num_fields×[len:u32][name] +
                #         count×num_fields×[len:u32][value].  No overall
                # length prefix — must walk the table to find its end.
                if batch_end + 4 > len(packed):
                    break  # truncated — stop parsing this child
                num_fields = struct.unpack_from("<I", packed, batch_end)[0]
                batch_end += 4
                # Skip field names.
                for _ in range(num_fields):
                    if batch_end + 4 > len(packed):
                        break
                    slen = struct.unpack_from("<I", packed, batch_end)[0]
                    batch_end += 4 + slen
                    if batch_end > len(packed):
                        break
                # Skip per-row values: count rows × num_fields entries.
                for _ in range(count * num_fields):
                    if batch_end + 4 > len(packed):
                        break
                    slen = struct.unpack_from("<I", packed, batch_end)[0]
                    batch_end += 4 + slen
                    if batch_end > len(packed):
                        break

            if batch_end > len(packed):
                break  # truncated data — stop parsing this child

            # Append the rewritten header + the batch body (instances + any
            # trailing data_indices / tooltips), then translate the X and Y
            # fields of every instance by the combined per-panel placement
            # offset and the global figure-title band.
            instance_start = len(result) + 20
            result.extend(header)
            result.extend(packed[pos + 20 : batch_end])
            total_dx = child_dx
            total_dy = child_dy + y_offset
            if total_dx != 0.0 or total_dy != 0.0:
                _offset_packed_batch_xy(result, instance_start, kind, count, total_dx, total_dy)

            pos = batch_end

    return bytes(result)
