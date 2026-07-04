"""Multi-chart composition primitives (HConcat, VConcat, Layer, Concat, Joint, Repeat, ClusterMap)."""

from __future__ import annotations

import copy
import json as _json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ferrum._chrome import chrome_kwargs, merge_configure_layers
from ferrum._configure_mixin import ConfigureMixin
from ferrum._overrides import _FIGURE_CHROME_KEYS

# Scene-graph merge layer lives in _scene_merge.py.  The composite chart classes
# below call these entry points; they are also re-exported here so existing
# ``from ferrum.composition import _merge_*`` / ``_offset_node`` / etc. sites
# (notably the scene-composition + html-export regression tests) keep resolving.
from ferrum._scene_merge import (  # noqa: F401  (re-exported for external importers)
    _EMPTY_SCENE_JSON,
    _OUTER_NODE_LIST_KEYS,
    _PACKED_INSTANCE_SIZES,
    _PANEL_AREA_KEYS,
    _PANEL_NODE_LIST_KEYS,
    _FigureChrome,
    _empty_scene,
    _inject_figure_chrome,
    _merge_child_scenes,
    _merge_child_scenes_grid,
    _merge_child_scenes_nonuniform_grid,
    _merge_child_scenes_sparse_grid,
    _merge_one_child,
    _merge_packed_data,
    _merge_scene_panels,
    _offset_node,
    _render_single_with_figure_chrome,
)


def _embed_chart_spec(c) -> Optional[dict]:
    """Convert a Chart's ``.to_spec()`` output to an embedded JSON dict."""
    if c is None or not hasattr(c, "to_spec"):
        return None
    return _json.loads(c.to_spec().to_json())


def _copy_configure_layers(src: "_ChartLike", dst: "_ChartLike") -> None:
    """Copy ``_configure_layers`` from *src* to *dst* if present.

    Used by ``theme()``, ``properties()``, and ``share_scale()`` after
    ``_rebuild_with_charts`` to ensure composition-level configure
    settings survive rebuild operations.
    """
    config = getattr(src, "_configure_layers", None)
    if config:
        dst._configure_layers = list(config)


def _shallow_copy_composite(src) -> object:
    """Shallow copy for ``_ChartLike`` subclasses that mix ``__slots__`` and ``__dict__``.

    Shared by ``_CompositeBase.__copy__`` and ``LayerChart.__copy__`` so the
    ``__dict__`` + MRO-slot copy logic lives in one place.  Returns a new
    instance with all ``__dict__`` keys and all ``__slots__`` attributes from
    the full MRO copied to the new object.  The caller is responsible for
    making any mutable slot attributes (e.g. ``charts``, ``_charts``) into
    fresh copies after this returns.
    """
    new = object.__new__(type(src))
    if hasattr(src, "__dict__"):
        new.__dict__.update(src.__dict__)
    for cls in type(src).__mro__:
        for slot in getattr(cls, "__slots__", ()):
            if slot == "__dict__":
                continue
            try:
                setattr(new, slot, getattr(src, slot))
            except AttributeError:
                pass
    return new


def _validate_resolve(resolve: Optional[Dict[str, str]], label: str) -> None:
    """Raise ``ValueError`` when *resolve* is not a valid channel-mode dict.

    Parameters
    ----------
    resolve : dict or None
        Must be a dict mapping channel names to ``"shared"`` or
        ``"independent"`` when not ``None``.
    label : str
        Class or function name used in the error message.
    """
    if resolve is None:
        return
    if not isinstance(resolve, dict):
        raise ValueError(
            f"{label}: resolve must be a dict mapping channel names "
            f"to 'shared' or 'independent'; got {type(resolve).__name__}"
        )
    for ch, mode in resolve.items():
        if mode not in ("shared", "independent"):
            raise ValueError(
                f"{label}: resolve[{ch!r}]={mode!r}; expected 'shared' or 'independent'"
            )


def _validate_share_modes(channels: Dict[str, str]) -> None:
    """Raise ``ValueError`` when any ``share_scale`` value is not a valid mode.

    Shared by :meth:`_ChartLike.share_scale` and
    :meth:`RepeatChart.share_scale` so the ``"shared"``/``"independent"``
    vocabulary is validated in one place.

    Parameters
    ----------
    channels : dict
        Channel name → mode mapping from a ``share_scale`` call.
    """
    for ch, mode in channels.items():
        if mode not in ("shared", "independent"):
            raise ValueError(f"share_scale: {ch}={mode!r}; expected 'shared' or 'independent'")


def _apply_resolve(charts: list, resolve: Optional[Dict[str, str]]) -> list:
    """Return charts with shared-scale injection applied per *resolve*.

    For each channel whose mode is ``"shared"``, compute the union domain
    across all charts and inject an explicit scale on every chart that
    binds that channel.  Charts are returned unchanged when *resolve* is
    ``None`` or empty, or when no channel qualifies.

    Parameters
    ----------
    charts : list of Chart
        Source charts.
    resolve : dict or None
        Per-channel scale-sharing spec, e.g. ``{"color": "shared"}``.

    Returns
    -------
    list of Chart
        Original list (same object) when nothing changes, otherwise a
        new list with scale-injected clones.
    """
    if not resolve:
        return charts
    from ferrum._scale_share import compute_union_domain, inject_scale

    shared = [ch for ch, mode in resolve.items() if mode == "shared"]
    if not shared:
        return charts
    result = list(charts)
    for channel in shared:
        sd = compute_union_domain(result, channel)
        if sd is None:
            continue
        result = [inject_scale(c, channel, sd) for c in result]
    return result


@dataclass
class _LoweredTree:
    """An HConcat/VConcat lowered to the one-call Rust composite render path.

    ``tree`` + ``payloads`` are the positional arguments to
    ``render_composite_svg`` / ``render_composite_interactive``; ``viewport``,
    ``theme``, and ``chart_config`` are the shared per-call kwargs the entry
    applies to every leaf.  Built by :func:`_lower_composite`.
    """

    tree: dict
    payloads: list
    viewport: tuple
    theme: dict
    chart_config: Optional[dict]


def _is_leaf_chart(node) -> bool:
    """Return True when *node* is a single ``Chart`` that lowers to a tree leaf.

    A ``Chart`` (plain or layered via ``+``) exposes ``_render_inputs`` and is
    not a composition wrapper, so it compiles to one ``ChartSpec`` + payload.
    Composition wrappers (:class:`_ChartLike`) are never leaves.
    """
    return not isinstance(node, _ChartLike) and hasattr(node, "_render_inputs")


def _composite_resolve_field(resolve: Optional[Dict[str, str]]) -> Optional[dict]:
    """Map a composition ``resolve=`` dict onto a composite node's resolve field.

    The Rust composite resolve pass spans only the positional ``x``/``y``
    channels, so a ``"shared"`` request on any other channel (``color``,
    ``size``, …) is not representable there.  Returns:

    - ``{}`` when there is nothing to share,
    - ``{"x": mode, ...}`` restricted to x/y when shareable,
    - ``None`` when a non-x/y channel is marked ``"shared"`` (the caller then
      keeps the old ``_scale_share`` injection path).
    """
    if not resolve:
        return {}
    out: dict = {}
    for channel, mode in resolve.items():
        if channel in ("x", "y"):
            out[channel] = mode
        elif mode == "shared":
            return None
    return out


def _lower_composite(composite, *, auto_tooltips: bool) -> Optional[_LoweredTree]:
    """Lower a composition (recursively) to a one-call composite render-tree.

    Handles every composite whose class declares a ``_composite_layout`` wire
    kind: the linear forms (HConcat/VConcat) and the wrapping-grid ``ConcatChart``
    (``wrap`` layout).  Layout-specific tree fields (e.g. ``ncols`` for wrap) come
    from each node's :meth:`_CompositeBase._composite_node_fields` hook, so the
    lowering body stays layout-agnostic rather than branching per class.

    Returns ``None`` — signalling the caller to keep the existing
    string-compositor / scene-merge path — when the composition cannot be
    rendered faithfully by the uniform composite entry.  The new path is taken
    when:

    - every descendant is a ``Chart`` leaf or a nested composite that declares a
      ``_composite_layout`` (HConcat/VConcat/ConcatChart),
    - no composite node carries composition-level configure layers (the
      composite path uses default figure-band chrome positioning),
    - every ``resolve=`` shared channel is positional x/y, and
    - no nested composite carries a figure *subtitle* or *caption* (those remain
      root-only on the composite path).

    A nested composite carrying only a figure *title* is no longer gated: its
    title lowers to a per-child ``"label"`` on the tree node (Task 5d wire), so
    ``compare=`` diagnostics whose per-model panels are titled *composites* — e.g.
    ``residuals`` (a titled ``VConcat``) — now ride the new path and share axes
    position-wise (GH #45).  Per-leaf ``viewport``/``theme``/``chart_config`` are
    no longer required to be uniform: when the leaves differ, each leaf node
    carries its own binding override (Task 5d wire; absent key = inherit the
    call-level value).  Homogeneous trees keep the compact call-level form so
    Task 6/7 output stays byte-identical.

    ``auto_tooltips`` mirrors ``Chart._render_scene``: the interactive path
    prepares leaves with auto-tooltips injected, the SVG path does not.  The
    helper is deliberately generic (leaf + nested-composite lowering) so the
    ratio/overlay cutovers (Tasks 8-9) can reuse it.
    """
    payloads: list = []
    # (viewport, theme, chart_config) per leaf; when the leaves differ, each
    # leaf node (parallel list below) carries its own binding override.
    leaf_inputs: list = []
    leaf_nodes: list = []

    def lower(node, is_root: bool) -> Optional[dict]:
        if _is_leaf_chart(node):
            return _lower_leaf_node(
                node,
                auto_tooltips=auto_tooltips,
                payloads=payloads,
                leaf_inputs=leaf_inputs,
                leaf_nodes=leaf_nodes,
            )

        layout = getattr(node, "_composite_layout", None)
        if layout is not None:
            if getattr(node, "_configure_layers", None):
                return None
            if not is_root and (
                node._figure_subtitle is not None or node._figure_caption is not None
            ):
                return None  # non-root subtitle/caption stay root-only (labels are title-only)
            resolve = _composite_resolve_field(getattr(node, "_resolve", None))
            if resolve is None:
                return None
            children: list = []
            for child in node.charts:
                child_node = lower(child, is_root=False)
                if child_node is None:
                    return None
                children.append(child_node)
            comp: dict = {
                "kind": "composite",
                "layout": layout,
                "children": children,
                "spacing": node.spacing,
                **node._composite_node_fields(),
            }
            if resolve:
                comp["resolve"] = resolve
            if is_root:
                if node._figure_title is not None:
                    comp["title"] = node._figure_title
                if node._figure_subtitle is not None:
                    comp["subtitle"] = node._figure_subtitle
                if node._figure_caption is not None:
                    comp["caption"] = node._figure_caption
            elif node._figure_title is not None:
                # A non-root composite's figure title lowers to a per-child panel
                # label (Task 5d wire), so titled composite compare= panels share
                # axes position-wise instead of gating to the old path (GH #45).
                comp["label"] = node._figure_title
            return comp

        # LayerChart / Joint / Repeat / ClusterMap declare no _composite_layout.
        return None

    root = lower(composite, is_root=True)
    if root is None or not leaf_inputs:
        return None

    viewport, theme, chart_config = _apply_leaf_binding_overrides(leaf_nodes, leaf_inputs)
    return _LoweredTree(
        tree=root,
        payloads=payloads,
        viewport=viewport,
        theme=theme,
        chart_config=chart_config or None,
    )


def _apply_leaf_binding_overrides(leaf_nodes: list, leaf_inputs: list) -> tuple:
    """Attach a per-leaf binding override to each node when leaf inputs differ.

    Shared by :func:`_lower_composite` and :func:`_build_grid_tree`: the first
    leaf's ``(viewport, theme, chart_config)`` becomes the call-level default
    passed to ``render_composite_svg``/``render_composite_interactive``; when any
    other leaf differs, every leaf node gets its own ``viewport``/``theme``/
    ``chart_config`` key (Task 5d wire) so no leaf silently inherits a sibling's
    binding.  Homogeneous trees keep the compact call-level form.

    Parameters
    ----------
    leaf_nodes : list of dict
        The tree's leaf node dicts, in the same order as *leaf_inputs*.
    leaf_inputs : list of tuple
        Each leaf's ``(viewport, theme, chart_config)`` from ``_render_inputs``.

    Returns
    -------
    tuple
        The call-level default ``(viewport, theme, chart_config)`` — the first
        leaf's inputs.
    """
    first = leaf_inputs[0]
    if any(other != first for other in leaf_inputs[1:]):
        for node, (viewport, theme, chart_config) in zip(leaf_nodes, leaf_inputs):
            node["viewport"] = viewport
            node["theme"] = theme
            # Written unconditionally: an empty dict is a valid "no configure"
            # override (matches _resolve_chart_config's empty-dict return for a
            # leaf with no configure/annotations/structural).  Gating on
            # truthiness here would leave the key absent, which means "inherit
            # the call-level default" -- silently reusing leaf 0's chart_config
            # on every unconfigured sibling (annotation bleed, mis-rotated axis
            # labels on the wrong panel).
            node["chart_config"] = chart_config
    return first


def _lower_leaf_node(
    chart,
    *,
    auto_tooltips: bool,
    payloads: list,
    leaf_inputs: list,
    leaf_nodes: list,
) -> Optional[dict]:
    """Lower one leaf chart to its wire node, appending to the parallel lists.

    The single source of truth for leaf lowering, shared by
    :func:`_lower_composite` and :func:`_build_grid_tree` — the empty-data
    guard here (``None`` = defer the whole tree to the legacy path) previously
    lived in two hand-copied blocks, and its omission from one of them was a
    real regression (Task 8b gap 1). Returns the ``{"kind": "leaf", ...}``
    node, or ``None`` when the leaf's data is empty.
    """
    spec, data, viewport, theme, chart_config = chart._render_inputs(
        _auto_tooltips=auto_tooltips
    )
    if data.num_rows == 0:
        return None  # per-child empty-data handling stays on the legacy path
    index = len(payloads)
    payloads.append(data)
    leaf_inputs.append((viewport, theme, chart_config))
    leaf_node = {"kind": "leaf", "spec": spec, "data": index}
    leaf_nodes.append(leaf_node)
    return leaf_node


def _build_grid_tree(
    cells: List[Optional[object]],
    *,
    nrows: int,
    ncols: int,
    row_ratios: Optional[List[float]],
    col_ratios: Optional[List[float]],
    spacing: float,
    auto_tooltips: bool,
    title: Optional[str],
    subtitle: Optional[str],
    caption: Optional[str],
    resolve: Optional[dict] = None,
) -> Optional[_LoweredTree]:
    """Lower a row-major grid of leaf charts (with optional holes) to a tree.

    Used by :class:`JointChart`, :class:`ClusterMapChart`, and
    :class:`RepeatChart` — grid composites whose fixed panel slots (and, for
    Joint/ClusterMap's single-marginal corner or Repeat's ``corner=True``
    upper triangle / wrapped trailing cells, one or more unused cells) don't
    fit :func:`_lower_composite`'s generic ``node.charts`` walk. Every entry in
    *cells* is either a plain leaf ``Chart`` or ``None`` (lowered to a
    ``{"kind": "hole"}`` placeholder cell, which the Rust grid layout reserves a
    slot for but draws nothing into — see the Task 8a hole wire). Holes are
    valid at any grid position, not only the 2×2 corner.
    ``title``/``subtitle``/``caption`` are always attached at the tree root: a
    1x1 grid wrapping a single chart is a valid composite tree (spec §6), so
    this same builder — and the same
    ``render_composite_svg``/``render_composite_interactive`` entries — cover
    every marginal-count / grid-shape case uniformly, with no separate
    single-chart bypass.

    Parameters
    ----------
    cells : list of Chart or None
        Row-major grid cells; ``len(cells)`` must equal ``nrows * ncols``.
    nrows, ncols : int
        Grid dimensions.
    row_ratios, col_ratios : list of float, optional
        Relative row/column sizes (``None`` for a uniform single row/column).
    spacing : float
        Pixel gap between adjacent cells.
    auto_tooltips : bool
        Forwarded to each leaf's ``_render_inputs`` (interactive vs. static).
    title, subtitle, caption : str, optional
        Root-only figure chrome.
    resolve : dict, optional
        Composite resolve field (e.g. ``{"x": "shared"}``) attached to the grid
        node so the Rust resolve pass unions the shared channel's domain across
        every cell. ``None`` or empty leaves each cell with independent scales.

    Returns
    -------
    _LoweredTree or None
        ``None`` when any non-hole cell's data is empty (``num_rows == 0``),
        mirroring :func:`_lower_composite`'s ``lower()`` leaf guard — the
        Rust composite-render entry cannot lay out a zero-row leaf, so the
        whole tree defers to the caller's legacy per-child render path
        instead (which already handles empty-data charts leaf-by-leaf).
    """
    payloads: list = []
    leaf_inputs: list = []
    leaf_nodes: list = []
    children: list = []
    for cell in cells:
        if cell is None:
            children.append({"kind": "hole"})
            continue
        node = _lower_leaf_node(
            cell,
            auto_tooltips=auto_tooltips,
            payloads=payloads,
            leaf_inputs=leaf_inputs,
            leaf_nodes=leaf_nodes,
        )
        if node is None:
            return None
        children.append(node)

    tree: dict = {
        "kind": "composite",
        "layout": "grid",
        "children": children,
        "nrows": nrows,
        "ncols": ncols,
        "spacing": spacing,
    }
    if row_ratios is not None:
        tree["row_ratios"] = row_ratios
    if col_ratios is not None:
        tree["col_ratios"] = col_ratios
    if resolve:
        tree["resolve"] = resolve
    if title is not None:
        tree["title"] = title
    if subtitle is not None:
        tree["subtitle"] = subtitle
    if caption is not None:
        tree["caption"] = caption

    viewport, theme, chart_config = _apply_leaf_binding_overrides(leaf_nodes, leaf_inputs)
    return _LoweredTree(
        tree=tree,
        payloads=payloads,
        viewport=viewport,
        theme=theme,
        chart_config=chart_config or None,
    )


class _ChartLike(ConfigureMixin):
    """Common rendering plumbing shared by every composition wrapper.

    Concrete subclasses must implement :meth:`to_svg`, :attr:`charts`,
    :meth:`theme`, :meth:`properties`, and :meth:`__repr__`.  This base
    centralizes the save / show / Jupyter-display / PNG-stub boilerplate
    that previously drifted across five copies (K2 / K3 / K11 / K15).

    **Composition-level configure** (``configure_axis``, ``configure_grid``,
    etc.) is accumulated in ``_configure_layers`` and injected into each
    child chart at render time.  Child-level config always wins over
    composition-level config because the composition layers are prepended
    (earlier entries in ``_configure`` are overridden by later ones).
    """

    def to_svg(self) -> str:  # pragma: no cover - abstract
        raise NotImplementedError(f"{type(self).__name__} must implement to_svg")

    def interactive(self, *, toolbar: bool = True):
        """Return an interactive rendering of this composition.

        Parameters
        ----------
        toolbar : bool, default True
            Whether to show the interactive toolbar (zoom/pan controls, export
            button). Set to ``False`` to render without the toolbar.

        Returns
        -------
        InteractiveChart
            An interactive widget/container backed by the WASM renderer.
        """
        from ferrum._interactive import InteractiveChart

        return InteractiveChart(self, toolbar=toolbar)

    # Subclasses provide ``charts`` as either an instance attribute
    # (symmetric containers — HConcat / VConcat) or as a ``@property``
    # (asymmetric containers — Joint / Repeat / ClusterMap, where the
    # shape is fixed and a derived list is the natural accessor).  We
    # do not declare it on the base because Python's data-descriptor
    # rules would block the attribute form on ``_CompositeBase``.

    def show(self) -> None:
        """Print the SVG markup to stdout."""
        print(self.to_svg())

    def _repr_svg_(self) -> str:
        """Return SVG for Jupyter inline display."""
        return self.to_svg()

    def _repr_mimebundle_(self, include=None, exclude=None) -> dict:
        """Return a Jupyter MIME bundle for rich display.

        Jupyter prefers ``_repr_mimebundle_`` over per-type ``_repr_*_``
        methods when both exist, so providing it lets front-ends negotiate
        formats without falling back to text repr.
        """
        return {"image/svg+xml": self.to_svg()}

    def to_png(self, *, scale: float = 2.0) -> bytes:
        """Return the composition rendered as PNG bytes.

        This **returns** the PNG-encoded image data; it does not display the
        composition.  Rasterises the SVG produced by :meth:`to_svg` through
        the Rust resvg pipeline — the same rasteriser ``Chart.to_png()`` uses.

        Parameters
        ----------
        scale : float, default 2.0
            Pixel-density multiplier applied to the SVG's intrinsic dimensions.
            ``2.0`` (the default) produces a retina-quality image.  ``1.0``
            renders at 1:1 resolution.

        Returns
        -------
        bytes
            PNG image as raw bytes suitable for ``IPython.display.Image``
            or writing directly to disk.
        """
        from ferrum._core import rasterize_svg

        return bytes(rasterize_svg(self.to_svg(), scale=scale))

    def _figure_title_text(self) -> str:
        """Return the resolved figure title text for the document ``<title>``.

        This is the canonical title accessor shared by every chart-like.
        The base implementation resolves a single chart's ``_title`` (a
        ``Title`` dataclass or plain string); :class:`_CompositeBase`
        overrides it to resolve the composite's ``_figure_title``.  Both
        fall back to ``"Ferrum chart"`` when no title is set.
        """
        from ferrum.display import _extract_title_text

        return _extract_title_text(getattr(self, "_title", None))

    def to_html(
        self,
        *,
        embed_wasm: bool = True,
        toolbar: bool = True,
        csp_nonce: str | None = None,
    ) -> str:
        """Return the composition as a self-contained interactive HTML document.

        This **returns** the HTML markup; it does not display the composition
        or write it to disk.  The returned string is byte-identical to what
        ``save(path)`` writes for an ``.html`` destination — it embeds the
        WASM-backed interactive renderer rather than a static SVG snapshot.
        Because it bundles that renderer, the document is substantially larger
        than a static export; for a lightweight static image use
        :meth:`to_svg` / :meth:`to_png`.

        Routes through the shared :func:`ferrum.display.html_string` helper, so
        the HTML assembly and tab-title resolution are identical to a plain
        ``Chart``.  It does **not** construct a live ``InteractiveChart``
        widget, so headless HTML export works without ``anywidget`` installed.

        Parameters
        ----------
        embed_wasm : bool, default True
            When True, the WASM binary is base64-inlined for single-file
            distribution.  When False, the document references an adjacent
            ``ferrum_wasm_bg.wasm`` sidecar that must be served alongside it.
        toolbar : bool, default True
            When False, the interactive toolbar (zoom / pan controls, export
            button) is hidden in the rendered HTML.
        csp_nonce : str, optional
            When provided, both the ``<style>`` and ``<script type="module">``
            tags receive a ``nonce="..."`` attribute so they pass strict
            Content-Security-Policy headers.

        Returns
        -------
        str
            A complete, self-contained interactive HTML document.
        """
        from ferrum.display import html_string

        return html_string(
            self,
            embed_wasm=embed_wasm,
            toolbar=toolbar,
            csp_nonce=csp_nonce,
        )

    def show_svg(self) -> str:
        """Render the composition to an SVG string.

        .. deprecated:: 0.16.0
            Use :meth:`to_svg` instead.  ``show_svg`` will be removed in a
            future release.  It now forwards to :meth:`to_svg`.

        Returns
        -------
        str
            SVG markup for the composition.
        """
        warnings.warn(
            f"{type(self).__name__}.show_svg() is deprecated; use .to_svg() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.to_svg()

    def show_png(self, *, scale: float = 2.0) -> bytes:
        """Render the composition to PNG bytes.

        .. deprecated:: 0.16.0
            Use :meth:`to_png` instead.  ``show_png`` will be removed in a
            future release.  It now forwards to :meth:`to_png`.

        Parameters
        ----------
        scale : float, default 2.0
            Pixel-density multiplier applied to the SVG's intrinsic dimensions.

        Returns
        -------
        bytes
            PNG image as raw bytes.
        """
        warnings.warn(
            f"{type(self).__name__}.show_png() is deprecated; use .to_png() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.to_png(scale=scale)

    def save(
        self,
        path: str,
        *,
        format=None,
        scale: float = 2.0,
        toolbar: bool = True,
        embed_wasm: bool = True,
        csp_nonce: str | None = None,
    ) -> None:
        """Save the composition to a file.

        Routes through :func:`ferrum.display.save_chart` — the single
        save-format router shared with ``Chart.save`` — so the supported
        format table (``svg`` / ``png`` / ``html`` / ``json`` / ``pdf``) and
        the HTML / title resolution are identical across chart and composite.

        Parameters
        ----------
        path : str
            Destination file path.  The extension determines the format when
            *format* is omitted.
        format : {"svg", "png", "html", "json", "pdf"}, optional
            Explicit format override.  Other formats raise ``ValueError``.
        scale : float, default 2.0
            Pixel-density multiplier for PNG and PDF output.  Has no effect
            on SVG, HTML, or JSON exports.
        toolbar : bool, default True
            Whether to include the interactive toolbar (zoom/pan controls,
            export button) when saving as HTML.  Has no effect on SVG, PNG,
            JSON, or PDF exports.
        embed_wasm : bool, default True
            For ``"html"`` format only.  When True, the WASM binary is
            base64-inlined for single-file distribution.  When False, an
            adjacent ``ferrum_wasm_bg.wasm`` sidecar is written alongside.
        csp_nonce : str, optional
            For ``"html"`` format only.  When provided, both the ``<style>``
            and ``<script type="module">`` tags receive a ``nonce="..."``
            attribute so they pass strict Content-Security-Policy headers.

        Raises
        ------
        ValueError
            If *format* (or the path extension) is not a recognised export
            format.
        """
        from ferrum.display import save_chart

        save_chart(
            self,
            path,
            format=format,
            scale=scale,
            toolbar=toolbar,
            embed_wasm=embed_wasm,
            csp_nonce=csp_nonce,
        )

    def share_scale(self, **channels):
        """Share scales across this composition's member charts.

        Computes the union domain for each channel marked ``"shared"``
        and re-emits every member chart with an explicit ``scale=`` dict
        on that channel, so the participating axes lock to the same
        ticks.  Channels marked ``"independent"`` (the default for any
        channel not listed) keep their per-chart domains.

        Parameters
        ----------
        **channels : str
            Channel name → ``"shared"`` | ``"independent"``.  Common
            channels: ``x``, ``y``, ``color``, ``size``.

        Returns
        -------
        _ChartLike
            A new composition of the same type with the shared scales
            injected.  No-op (returns ``self``) when no channel is
            ``"shared"`` or none of the requested channels are bound on
            any member chart.

        Raises
        ------
        ValueError
            If any value is not ``"shared"`` or ``"independent"``.

        Examples
        --------
        >>> import ferrum as fm
        >>> combined = (chart_a | chart_b).share_scale(x="shared")
        >>> grid = fm.JointChart(center, top=hist_x, right=hist_y).share_scale(y="shared")
        """
        _validate_share_modes(channels)
        shared = [ch for ch, mode in channels.items() if mode == "shared"]
        if not shared:
            return self
        from ferrum._scale_share import compute_union_domain, inject_scale

        member_charts = self.charts
        scale_dicts = {}
        for channel in shared:
            sd = compute_union_domain(member_charts, channel)
            if sd is not None:
                scale_dicts[channel] = sd
        if not scale_dicts:
            return self

        def _apply(chart):
            out = chart
            for ch, sd in scale_dicts.items():
                out = inject_scale(out, ch, sd)
            return out

        result = self._rebuild_with_charts(_apply)
        _copy_configure_layers(self, result)
        return result

    def theme(self, t):
        """Apply a theme to every sub-chart and return a new composition.

        Parameters
        ----------
        t : Theme
            Theme value to apply.

        Returns
        -------
        _ChartLike
            A new instance of the same composition class with *t* applied
            to each sub-chart.
        """
        result = self._rebuild_with_charts(lambda c: c.theme(t))
        _copy_configure_layers(self, result)
        return result

    def properties(self, **kwargs):
        """Forward ``properties(**kwargs)`` to every sub-chart.

        This base implementation is used by ``LayerChart`` and by plain
        single-chart wrappers that do not override it.
        ``_CompositeBase`` overrides this with a version that intercepts
        figure-level chrome (``title``, ``subtitle``, ``caption``) and stores
        it at the composition level rather than fanning it to every child.

        Parameters
        ----------
        **kwargs
            Keyword arguments accepted by ``Chart.properties`` (e.g.
            ``width``, ``height``, ``title``).

        Returns
        -------
        _ChartLike
            A new instance of the same composition class with updated
            sub-chart properties.
        """
        result = self._rebuild_with_charts(lambda c: c.properties(**kwargs))
        _copy_configure_layers(self, result)
        return result

    # ---- Declarative configuration surface ----

    def _inject_parent_config(self, chart):
        """Prepend composition-level configure layers onto a child chart.

        For ``Chart`` children, composition-level layers are prepended to the
        chart's ``_configure`` list so that per-chart layers (which appear
        later) override them — ``_resolve_chart_config`` processes
        ``_configure`` in order with later entries winning.

        For nested composition children (``_ChartLike`` subclasses), the
        layers are merged into the child's ``_configure_layers`` so they
        propagate further down at render time.
        """
        config = getattr(self, "_configure_layers", None)
        if not config:
            return chart
        if isinstance(chart, _ChartLike):
            # Nested composition: merge into child's own _configure_layers.
            new = copy.copy(chart)
            existing = getattr(chart, "_configure_layers", [])
            new._configure_layers = list(config) + list(existing)
            return new
        # Plain Chart: prepend to _configure list.
        new = chart._clone()
        new._configure = list(config) + list(new._configure)
        return new

    def _append_configure(self, config) -> "_ChartLike":
        """Clone self, append *config* to ``_configure_layers``, return new instance."""
        new = copy.copy(self)
        new._configure_layers = list(getattr(self, "_configure_layers", [])) + [config]
        return new

    # ---- Declarative configuration surface (provided by ConfigureMixin) ----

    def _rebuild_with_charts(self, fn):  # pragma: no cover - abstract
        """Return a new composition with each member chart transformed by *fn*.

        Subclasses must implement this — it's the seam between the
        generic ``share_scale`` / ``theme`` / ``properties`` plumbing on
        the base and each composition's constructor signature.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement _rebuild_with_charts")


class _CompositeBase(_ChartLike):
    """Shared base for every composite chart with figure-level chrome.

    This is the single home for figure-level title / subtitle / caption
    across all composites — the symmetric list containers (HConcat /
    VConcat / Concat) *and* the asymmetric panel layouts (Joint / Repeat /
    ClusterMap).  Figure chrome is stored at the composition level
    (``_figure_title`` / ``_figure_subtitle`` / ``_figure_caption``),
    intercepted in :meth:`properties` so it never reaches an inner panel,
    and surfaced for the HTML document title via :meth:`_figure_title_text`.

    The symmetric containers also use this class's ``__init__`` to hold an
    ordered ``charts`` list and a pixel ``spacing`` between panels, plus
    ``__or__`` / ``__and__`` to chain further compositions.  The asymmetric
    layouts keep their own slot-based ``__init__`` and ``charts`` property;
    they call :meth:`_init_figure_chrome` to wire the chrome fields.

    **Symmetric-concat layout strategy.**  ``HConcatChart`` and
    ``VConcatChart`` differ only in their layout axis, so their
    ``_resolved_charts`` / ``_rebuild_with_charts`` / ``_render_interactive``
    / ``to_svg`` / ``__repr__`` bodies live here once, parameterized by three
    class attributes a subclass overrides: :attr:`_layout` (the
    ``_merge_child_scenes`` layout key), :attr:`_svg_compose_name` (the
    ``ferrum._core`` SVG compositor to import), and :attr:`_svg_align` (the
    cross-axis alignment passed to that compositor).  These default to
    ``None`` on the base; the asymmetric layouts (Joint / Repeat /
    ClusterMap) and the wrapping-grid ``ConcatChart`` override the symmetric
    methods wholesale, so the ``None`` defaults are never reached for them.
    """

    # Symmetric-concat strategy hooks (overridden by HConcat / VConcat).
    _layout: Optional[str] = None
    _svg_compose_name: Optional[str] = None
    _svg_align: Optional[str] = None
    # Composite-tree layout kind for the one-call Rust composite render path
    # (``render_composite_svg`` / ``render_composite_interactive``).
    _composite_layout: Optional[str] = None

    def __init__(
        self,
        charts: List,
        *,
        spacing: float = 10.0,
        resolve: Optional[Dict[str, str]] = None,
    ) -> None:
        _validate_resolve(resolve, type(self).__name__)
        self.charts = list(charts)
        self.spacing = spacing
        self._resolve = resolve
        self._init_figure_chrome()

    def _init_figure_chrome(self) -> None:
        """Initialize the figure-chrome fields to their empty state.

        Called from every composite's ``__init__`` (directly for the
        asymmetric layouts, via :meth:`__init__` for the symmetric ones)
        so the chrome attributes always exist before :meth:`properties`
        runs.
        """
        self._figure_title: Optional[str] = None
        self._figure_subtitle: Optional[str] = None
        self._figure_caption: Optional[str] = None

    def _carry_figure_chrome(self, dst: "_CompositeBase") -> None:
        """Copy this composite's figure chrome onto *dst* (a new instance)."""
        dst._figure_title = self._figure_title
        dst._figure_subtitle = self._figure_subtitle
        dst._figure_caption = self._figure_caption

    def _figure_title_text(self) -> str:
        """Resolve the composite's figure title text for the document ``<title>``."""
        from ferrum.display import _extract_title_text

        return _extract_title_text(self._figure_title)

    def _figure_chrome_kwargs(self) -> "_FigureChrome":
        """Bundle figure chrome for the interactive scene-merge functions.

        Returns the ``figure_chrome`` payload consumed by every
        ``_merge_child_scenes*`` helper: the title / subtitle / caption text
        plus the positioning ``chrome`` sub-dict resolved from this
        composite's configure layers (the same kwargs the SVG path passes to
        ``compose_svg_*``).  This keeps the interactive on-canvas band in step
        with the SVG band from a single source of chrome values.
        """
        return _FigureChrome(
            title=self._figure_title,
            subtitle=self._figure_subtitle,
            caption=self._figure_caption,
            chrome=chrome_kwargs(merge_configure_layers(getattr(self, "_configure_layers", None))),
        )

    def __copy__(self):
        """Shallow copy that duplicates mutable list attributes."""
        new = _shallow_copy_composite(self)
        # Ensure the mutable charts list is a fresh copy.  Asymmetric
        # layouts expose ``charts`` as a read-only property (derived from
        # their panels), so only refresh it when it is a writable attribute.
        if not isinstance(getattr(type(self), "charts", None), property):
            new.charts = list(self.charts)
        return new

    def __or__(self, other):
        return HConcatChart([self, other])

    def __and__(self, other):
        return VConcatChart([self, other])

    def properties(self, **kwargs):
        """Set figure-level or per-child chart properties.

        The keyword arguments ``title``, ``subtitle``, and ``caption`` are
        intercepted and stored at the figure level — they render once around
        the whole composed figure and are never fanned to individual child
        panels.  All other keyword arguments (e.g. ``width``, ``height``)
        are forwarded to each child via ``Chart.properties``.

        Parameters
        ----------
        title : str, optional
            Figure-level title rendered above all panels.
        subtitle : str, optional
            Figure-level subtitle rendered below the title, above all panels.
        caption : str, optional
            Figure-level caption rendered below all panels.
        **kwargs
            Additional keyword arguments forwarded to ``Chart.properties``
            for every child chart (e.g. ``width``, ``height``).

        Returns
        -------
        _CompositeBase
            A new instance of the same composition class with the figure
            chrome stored and / or child properties updated.
        """
        # Separate figure-level chrome from per-child kwargs.
        # Use the shared _FIGURE_CHROME_KEYS constant so the key set stays
        # in sync with the factory-dict split in _overrides._apply_overrides.
        chrome_vals = {k: kwargs.pop(k, None) for k in _FIGURE_CHROME_KEYS}
        figure_title = chrome_vals["title"]
        figure_subtitle = chrome_vals["subtitle"]
        figure_caption = chrome_vals["caption"]

        if kwargs:
            # Forward remaining (non-chrome) kwargs to the appropriate panel(s).
            result = self._forward_child_properties(kwargs)
        else:
            # Nothing to fan — rebuild preserving charts unchanged.
            result = self._rebuild_with_charts(lambda c: c)

        _copy_configure_layers(self, result)

        # ``_rebuild_with_charts`` / ``_forward_child_properties`` already
        # carried this composite's existing chrome onto ``result``; only
        # override the fields a value was given for.
        if figure_title is not None:
            result._figure_title = figure_title
        if figure_subtitle is not None:
            result._figure_subtitle = figure_subtitle
        if figure_caption is not None:
            result._figure_caption = figure_caption

        return result

    def _forward_child_properties(self, kwargs: dict) -> "_CompositeBase":
        """Apply non-chrome ``properties`` kwargs to the relevant child panel(s).

        The default fans the kwargs to every member chart, which is correct
        for the symmetric containers (HConcat / VConcat / Concat) and for the
        repeat-grid template.  Asymmetric layouts whose marginal panels derive
        their size from a primary panel (Joint → center, ClusterMap → heatmap)
        override this to route the kwargs to that primary panel only.
        """
        return self._rebuild_with_charts(lambda c: c.properties(**kwargs))

    # ------------------------------------------------------------------
    # Symmetric-concat layout strategy.
    #
    # These five methods are the shared bodies of HConcat / VConcat,
    # parameterized by ``_layout`` / ``_svg_compose_name`` / ``_svg_align``.
    # The asymmetric layouts (Joint / Repeat / ClusterMap) and the
    # wrapping-grid ConcatChart override all five, so the ``None`` hook
    # defaults are never reached for them.
    # ------------------------------------------------------------------

    def _composite_node_fields(self) -> dict:
        """Layout-specific tree fields for this node's composite-render entry.

        The linear forms (HConcat/VConcat) contribute nothing beyond
        ``layout``/``children``/``spacing``, so the base returns an empty dict.
        :class:`ConcatChart` overrides this to emit the ``wrap`` layout's
        ``ncols``.  Keeping the layout-specific keys behind this hook lets
        :func:`_lower_composite` stay layout-agnostic instead of branching per
        composite class.
        """
        return {}

    def _resolved_charts(self) -> list:
        """Return charts with shared scales injected per ``resolve``.

        Uses ``getattr`` for ``_resolve`` so this shared base method is safe on
        any ``_CompositeBase`` subclass: the symmetric concat classes set
        ``_resolve`` in ``__init__``, while asymmetric subclasses
        (Joint/Repeat/ClusterMap) never set it and override the render path, so
        they would otherwise hit an ``AttributeError`` if this base method were
        ever called on them.
        """
        return _apply_resolve(self.charts, getattr(self, "_resolve", None))

    def _rebuild_with_charts(self, fn):
        new = type(self)(
            [fn(c) for c in self.charts],
            spacing=self.spacing,
            resolve=getattr(self, "_resolve", None),
        )
        self._carry_figure_chrome(new)
        return new

    def _render_interactive(self) -> tuple[str, bytes]:
        """Render to (scene_json, packed_data) for the interactive renderer.

        Routes HConcat/VConcat/ConcatChart through the one-call Rust composite
        entry (``render_composite_interactive``) when the composition lowers
        cleanly (see :func:`_lower_composite`), otherwise falls back to the
        per-child scene-merge path (each subclass supplies its own
        ``_render_interactive_scene_merge``).
        """
        lowered = _lower_composite(self, auto_tooltips=True)
        if lowered is not None:
            from ferrum._core import render_composite_interactive

            return render_composite_interactive(
                lowered.tree,
                lowered.payloads,
                viewport=lowered.viewport,
                theme=lowered.theme,
                chart_config=lowered.chart_config,
            )
        return self._render_interactive_scene_merge()

    def _render_interactive_scene_merge(self) -> tuple[str, bytes]:
        """Merge per-child scenes along ``_layout`` (string-merge path)."""
        charts = [self._inject_parent_config(c) for c in self._resolved_charts()]
        return _merge_child_scenes(
            charts,
            self.spacing,
            layout=self._layout,
            figure_chrome=self._figure_chrome_kwargs(),
        )

    def to_svg(self) -> str:
        """Render the concatenated charts to an SVG string.

        Routes HConcat/VConcat/ConcatChart through the one-call Rust composite
        entry (``render_composite_svg``) when the composition lowers cleanly
        (see :func:`_lower_composite`), otherwise falls back to the
        string-compositor path (each subclass supplies its own
        ``_to_svg_string_compositor``).
        """
        lowered = _lower_composite(self, auto_tooltips=False)
        if lowered is not None:
            from ferrum._core import render_composite_svg

            return render_composite_svg(
                lowered.tree,
                lowered.payloads,
                viewport=lowered.viewport,
                theme=lowered.theme,
                chart_config=lowered.chart_config,
            )
        return self._to_svg_string_compositor()

    def _to_svg_string_compositor(self) -> str:
        """Compose per-child SVGs along ``_layout`` via the Rust string compositor."""
        from ferrum import _core

        compose = getattr(_core, self._svg_compose_name)
        charts = [self._inject_parent_config(c) for c in self._resolved_charts()]
        svgs = [c.to_svg() for c in charts]
        chrome = chrome_kwargs(merge_configure_layers(getattr(self, "_configure_layers", None)))
        return compose(
            svgs,
            spacing=self.spacing,
            align=self._svg_align,
            title=self._figure_title,
            subtitle=self._figure_subtitle,
            caption=self._figure_caption,
            **chrome,
        )

    def __repr__(self) -> str:
        """Return a short developer-readable description."""
        return f"{type(self).__name__}([{', '.join(repr(c) for c in self.charts)}])"


class HConcatChart(_CompositeBase):
    """Horizontal concatenation of two or more charts.

    Each sub-chart retains its own scales, axes, and legend by default.
    Pass ``resolve=`` to unify one or more channels across panels.
    Construct via the ``|`` operator on ``Chart`` instances or directly
    with a list.

    Parameters
    ----------
    charts : list of Chart
        Sub-charts to concatenate left-to-right.
    spacing : float, default 10.0
        Horizontal pixel gap between adjacent charts.
    resolve : dict, optional
        Per-channel scale-sharing overrides, e.g.
        ``{"color": "shared"}``.  Accepts the same keys and values as
        ``ConcatChart(resolve=...)``.

    Examples
    --------
    >>> import ferrum as fm
    >>> combined = fm.Chart(df).encode(x="hp", y="mpg").mark_point() | fm.Chart(df).encode(x="hp").mark_histogram()
    >>> combined.save("side_by_side.svg")
    """

    # Layout-strategy hooks consumed by _CompositeBase's symmetric-concat
    # methods (_render_interactive / to_svg).  Construction, resolve, rebuild,
    # and __repr__ are all inherited unchanged.
    _layout = "horizontal"
    _svg_compose_name = "compose_svg_horizontal"
    _svg_align = "top"
    _composite_layout = "hconcat"


class VConcatChart(_CompositeBase):
    """Vertical concatenation of two or more charts.

    Each sub-chart retains its own scales, axes, and legend by default.
    Pass ``resolve=`` to unify one or more channels across panels.
    Construct via the ``&`` operator on ``Chart`` instances or directly
    with a list.

    Parameters
    ----------
    charts : list of Chart
        Sub-charts to stack top-to-bottom.
    spacing : float, default 10.0
        Vertical pixel gap between adjacent charts.
    resolve : dict, optional
        Per-channel scale-sharing overrides, e.g.
        ``{"color": "shared"}``.  Accepts the same keys and values as
        ``ConcatChart(resolve=...)``.

    Examples
    --------
    >>> import ferrum as fm
    >>> stacked = fm.Chart(df).encode(x="hp", y="mpg").mark_point() & fm.Chart(df).encode(x="hp").mark_histogram()
    >>> stacked.save("stacked.svg")
    """

    # Layout-strategy hooks consumed by _CompositeBase's symmetric-concat
    # methods (_render_interactive / to_svg).  Construction, resolve, rebuild,
    # and __repr__ are all inherited unchanged.
    _layout = "vertical"
    _svg_compose_name = "compose_svg_vertical"
    _svg_align = "left"
    _composite_layout = "vconcat"


# --------------------------------------------------------------------------
# Phase 9 compound views: JointChart, RepeatChart, ClusterMapChart
# --------------------------------------------------------------------------


class JointChart(_CompositeBase):
    """Joint distribution view: center chart plus optional top and right marginals.

    Lays out a 2 × 2 grid: center chart occupies the bottom-left panel,
    *top* marginal goes top-left, *right* marginal goes bottom-right, and the
    top-right corner is empty.  The x-axis is shared between the center and
    top charts; the y-axis is shared between the center and right charts.

    The panel size ratio between the center and each marginal is controlled by
    ``ratio``.  A ratio of 5 gives the center 5/(5+1) of each dimension and
    each marginal 1/(5+1).

    Most users obtain a ``JointChart`` from `ferrum.jointplot` rather than
    constructing one directly.

    Parameters
    ----------
    center : Chart
        Primary scatter / distribution chart occupying the main panel.
    top : Chart, optional
        Marginal chart drawn above the center (e.g. a histogram of the x
        variable).
    right : Chart, optional
        Marginal chart drawn to the right of the center (e.g. a histogram
        of the y variable).
    ratio : int, default 5
        Size ratio of the center panel to each marginal panel.  Must be > 0.
    spacing : float, default 10.0
        Pixel gap between adjacent panels.

    Raises
    ------
    ValueError
        If *ratio* is not > 0.

    Examples
    --------
    >>> import ferrum as fm
    >>> joint = fm.jointplot(df, x="hp", y="mpg")
    >>> joint.save("joint.svg")
    """

    __slots__ = ("center", "top", "right", "ratio", "spacing")

    def __init__(
        self,
        center,
        *,
        top=None,
        right=None,
        ratio: int = 5,
        spacing: float = 10.0,
    ) -> None:
        if ratio <= 0:
            raise ValueError(f"ratio must be > 0; got {ratio}")
        self.center = center
        self.top = top
        self.right = right
        self.ratio = ratio
        self.spacing = spacing
        self._init_figure_chrome()

    @property
    def charts(self) -> list:
        """List of Chart : All non-None sub-charts (center, top, right)."""
        return [c for c in (self.center, self.top, self.right) if c is not None]

    @property
    def spec(self) -> dict:
        """Dict : Serializable layout spec consumed by the SVG compositor."""
        share_x = ["center"]
        if self.top is not None:
            share_x.append("top")
        share_y = ["center"]
        if self.right is not None:
            share_y.append("right")
        return {
            "kind": "joint",
            "center": _embed_chart_spec(self.center),
            "top": _embed_chart_spec(self.top),
            "right": _embed_chart_spec(self.right),
            "ratio": self.ratio,
            "spacing": self.spacing,
            "share": {"x": share_x, "y": share_y},
        }

    def _forward_child_properties(self, kwargs: dict) -> "JointChart":
        """Route non-chrome ``properties`` kwargs to the center chart only.

        The marginals (top, right) are kept unchanged because their width /
        height is derived from the center plus ``ratio`` at render time.
        Figure-level chrome (``title`` / ``subtitle`` / ``caption``) is
        intercepted by :meth:`_CompositeBase.properties` before this hook
        runs, so it never reaches the center panel.
        """
        result = JointChart(
            self.center.properties(**kwargs),
            top=self.top,
            right=self.right,
            ratio=self.ratio,
            spacing=self.spacing,
        )
        self._carry_figure_chrome(result)
        return result

    def _rebuild_with_charts(self, fn):
        new = JointChart(
            fn(self.center),
            top=(fn(self.top) if self.top is not None else None),
            right=(fn(self.right) if self.right is not None else None),
            ratio=self.ratio,
            spacing=self.spacing,
        )
        self._carry_figure_chrome(new)
        return new

    def _composite_tree(self, *, auto_tooltips: bool) -> Optional[_LoweredTree]:
        """Lower this JointChart to a 2×2 ratio/hole composite grid tree.

        Row-major cell layout — mirrors the pre-cutover ``to_svg``/
        ``_render_interactive`` panel positions exactly:

        - both marginals: ``[top, HOLE, center, right]`` on a 2×2 grid, with
          ``row_ratios=[marginal_share, center_share]`` and
          ``col_ratios=[center_share, marginal_share]`` (the empty top-right
          corner becomes a ``{"kind": "hole"}`` cell rather than a wasted,
          unconditionally-reserved column — see Task 8a).
        - one marginal: a dense 2×1 or 1×2 grid (no hole needed).
        - no marginals: a dense 1×1 grid — a single-cell composite tree is
          valid (spec §6), so this one builder and the shared
          ``render_composite_svg``/``render_composite_interactive`` entries
          cover every marginal-count case, carrying figure chrome at the root
          uniformly instead of a separate single-chart bypass.

        Marginals suppress their own axis decoration via ``axis(show=False)``
        before lowering (the data axis is redundant against the centre panel;
        the marginal-only axis is illegible at marginal size) — applied here so
        both the static and interactive paths share the exact same behavior
        (pre-cutover, only ``to_svg`` hid marginal axes; the interactive path
        did not).

        Returns
        -------
        _LoweredTree or None
            ``None`` when center/top/right is not a plain leaf ``Chart``
            (e.g. a nested composition), signalling the caller to fall back to
            the legacy per-child render path.
        """
        center = self._inject_parent_config(self.center)
        top = self._inject_parent_config(self.top) if self.top is not None else None
        right = self._inject_parent_config(self.right) if self.right is not None else None
        if top is not None:
            top = top.axis(show=False)
        if right is not None:
            right = right.axis(show=False)

        if any(not _is_leaf_chart(c) for c in (center, top, right) if c is not None):
            return None

        marginal_share = 1.0 / (self.ratio + 1)
        center_share = self.ratio / (self.ratio + 1)

        cells: List[Optional[object]]
        if top is not None and right is not None:
            cells = [top, None, center, right]
            nrows, ncols = 2, 2
            row_ratios: Optional[List[float]] = [marginal_share, center_share]
            col_ratios: Optional[List[float]] = [center_share, marginal_share]
        elif top is not None:
            cells = [top, center]
            nrows, ncols = 2, 1
            row_ratios = [marginal_share, center_share]
            col_ratios = None
        elif right is not None:
            cells = [center, right]
            nrows, ncols = 1, 2
            row_ratios = None
            col_ratios = [center_share, marginal_share]
        else:
            cells = [center]
            nrows, ncols = 1, 1
            row_ratios = None
            col_ratios = None

        return _build_grid_tree(
            cells,
            nrows=nrows,
            ncols=ncols,
            row_ratios=row_ratios,
            col_ratios=col_ratios,
            spacing=self.spacing,
            auto_tooltips=auto_tooltips,
            title=self._figure_title,
            subtitle=self._figure_subtitle,
            caption=self._figure_caption,
        )

    def _render_interactive(self) -> tuple[str, bytes]:
        """Render to (scene_json, packed_data) via the composite grid entry.

        Routes through ``render_composite_interactive`` when every panel is a
        plain leaf ``Chart`` (see :meth:`_composite_tree`), otherwise falls
        back to :meth:`_render_interactive_legacy`.
        """
        lowered = self._composite_tree(auto_tooltips=True)
        if lowered is not None:
            from ferrum._core import render_composite_interactive

            return render_composite_interactive(
                lowered.tree,
                lowered.payloads,
                viewport=lowered.viewport,
                theme=lowered.theme,
                chart_config=lowered.chart_config,
            )
        return self._render_interactive_legacy()

    def _render_interactive_legacy(self) -> tuple[str, bytes]:
        """Merge per-child scenes in a 2x2 grid (fallback for non-leaf children).

        Grid layout mirrors the SVG path (``_to_svg_legacy``):
          - top marginal  at (row=0, col=0)
          - center        at (row=1, col=0)
          - right marginal at (row=1, col=1)
        """
        center = self._inject_parent_config(self.center)
        top = self._inject_parent_config(self.top) if self.top is not None else None
        right = self._inject_parent_config(self.right) if self.right is not None else None

        has_top = top is not None
        has_right = right is not None

        # No marginals: render center directly (still wrap the figure band).
        if not has_top and not has_right:
            return _render_single_with_figure_chrome(center, self._figure_chrome_kwargs())

        # Build grid panels matching the SVG path layout.
        panels: list[tuple[int, int, object]] = []
        if has_top and has_right:
            # Full 2x2 grid: top at (0,0), center at (1,0), right at (1,1).
            panels = [(0, 0, top), (1, 0, center), (1, 1, right)]
        elif has_top:
            # Vertical stack: top at (0,0), center at (1,0).
            panels = [(0, 0, top), (1, 0, center)]
        else:
            # Horizontal stack: center at (0,0), right at (0,1).
            panels = [(0, 0, center), (0, 1, right)]

        return _merge_child_scenes_nonuniform_grid(
            panels,
            self.spacing,
            figure_chrome=self._figure_chrome_kwargs(),
        )

    def to_svg(self) -> str:
        """Render the joint chart to an SVG string.

        Routes through ``render_composite_svg`` when every panel is a plain
        leaf ``Chart`` (see :meth:`_composite_tree`), otherwise falls back to
        :meth:`_to_svg_legacy`.

        Returns
        -------
        str
            SVG markup with the 2 × 2 grid layout.
        """
        lowered = self._composite_tree(auto_tooltips=False)
        if lowered is not None:
            from ferrum._core import render_composite_svg

            return render_composite_svg(
                lowered.tree,
                lowered.payloads,
                viewport=lowered.viewport,
                theme=lowered.theme,
                chart_config=lowered.chart_config,
            )
        return self._to_svg_legacy()

    def _to_svg_legacy(self) -> str:
        """Compose per-child SVGs via the string compositor (fallback path)."""
        from ferrum._core import compose_svg_grid

        center = self._inject_parent_config(self.center)
        top = self._inject_parent_config(self.top) if self.top is not None else None
        right = self._inject_parent_config(self.right) if self.right is not None else None
        top_chart = top.axis(show=False) if top is not None else None
        right_chart = right.axis(show=False) if right is not None else None
        top_svg = top_chart.to_svg() if top_chart is not None else None
        right_svg = right_chart.to_svg() if right_chart is not None else None
        panels = [top_svg, None, center.to_svg(), right_svg]
        marginal_share = 1.0 / (self.ratio + 1)
        center_share = self.ratio / (self.ratio + 1)
        chrome = chrome_kwargs(merge_configure_layers(getattr(self, "_configure_layers", None)))
        return compose_svg_grid(
            panels,
            rows=2,
            cols=2,
            row_ratios=[marginal_share, center_share],
            col_ratios=[center_share, marginal_share],
            spacing=self.spacing,
            title=self._figure_title,
            subtitle=self._figure_subtitle,
            caption=self._figure_caption,
            **chrome,
        )

    def __repr__(self) -> str:
        """Return a short developer-readable description."""
        return (
            f"JointChart(center={self.center!r}, top={self.top!r}, "
            f"right={self.right!r}, ratio={self.ratio})"
        )


class RepeatChart(_CompositeBase):
    """Repeat a template chart over a grid of row / column field combinations.

    Use ``Repeat.column``, ``Repeat.row``, or ``Repeat.layer`` typed sentinels
    in the template's ``.encode(...)`` call to mark which encoding channel
    receives the per-panel field substitution.  ``RepeatChart.expand()``
    materializes the grid into fully-resolved ``(row_field, col_field, Chart)``
    tuples.

    ``diagonal=`` provides an alternate template for panels where
    ``row_field == col_field`` (symmetric n × n repeat).  ``corner=True``
    filters the expanded grid to the lower triangle including the diagonal.

    Most users obtain a ``RepeatChart`` through ``Chart.repeat()`` or
    ``ferrum.pairplot``.

    Parameters
    ----------
    template : Chart
        Template chart whose ``Repeat.*`` placeholders are substituted per
        panel.
    row : list of str, optional
        Field names assigned to the row axis.
    column : list of str, optional
        Field names assigned to the column axis.
    layer : list of str, optional
        Field names assigned to the layer axis (for non-grid repeat layouts).
    diagonal : Chart, optional
        Alternate template used when ``row_field == col_field``.  Requires
        both *row* and *column* to be set.
    corner : bool, default False
        When ``True``, only the lower-triangle panels (``ri >= ci``) are
        rendered, giving a half-matrix layout.
    spacing : float, default 10.0
        Pixel gap between adjacent panels.
    columns : int, optional
        Maximum number of columns for a wrapped 1-D repeat layout (no-op
        for 2-D row/column repeat).
    resolve : dict, optional
        Per-channel scale-sharing overrides — e.g.
        ``resolve={"x": "shared", "y": "independent"}``.  ``"shared"``
        computes the union domain across all panels (and across every
        layer of layered panels) and injects an explicit scale on every
        participating chart so the axis ticks match.  ``"independent"``
        (the default for unlisted channels) keeps per-panel domains.

    Raises
    ------
    ValueError
        If *diagonal* is set but *row* or *column* is not.

    Examples
    --------
    >>> import ferrum as fm
    >>> base = fm.Chart(df).encode(x=fm.Repeat.column, y=fm.Repeat.row).mark_point()
    >>> grid = fm.RepeatChart(base, row=["mpg", "hp"], column=["mpg", "hp"])
    >>> grid.save("pair_grid.svg")
    """

    __slots__ = (
        "template",
        "row",
        "column",
        "layer",
        "diagonal",
        "corner",
        "spacing",
        "columns",
        "resolve",
    )

    def __init__(
        self,
        template,
        *,
        row=None,
        column=None,
        layer=None,
        diagonal=None,
        corner: bool = False,
        spacing: float = 10.0,
        columns: Optional[int] = None,
        resolve=None,
    ) -> None:
        if diagonal is not None and (row is None or column is None):
            raise ValueError("RepeatChart: diagonal= requires both row= and column= to be set")
        if corner and (row is None or column is None):
            raise ValueError("RepeatChart: corner=True requires both row= and column= to be set")
        if row is None and column is None and layer is None:
            raise ValueError("RepeatChart: at least one of row=, column=, or layer= must be set")
        if columns is not None and columns <= 0:
            raise ValueError(f"RepeatChart: columns must be > 0; got {columns}")
        _validate_resolve(resolve, "RepeatChart")
        self.template = template
        self.row = list(row) if row is not None else None
        self.column = list(column) if column is not None else None
        self.layer = list(layer) if layer is not None else None
        self.diagonal = diagonal
        self.corner = corner
        self.spacing = spacing
        self.columns = columns
        self.resolve = resolve
        self._init_figure_chrome()

    @property
    def charts(self) -> list:
        """List of Chart : Template plus diagonal (when set), in init order."""
        return [c for c in (self.template, self.diagonal) if c is not None]

    @property
    def spec(self) -> dict:
        """Dict : Serializable layout spec consumed by the SVG compositor."""
        return {
            "kind": "repeat",
            "template": _embed_chart_spec(self.template),
            "row": self.row,
            "column": self.column,
            "layer": self.layer,
            "diagonal": _embed_chart_spec(self.diagonal),
            "corner": self.corner,
            "columns": self.columns,
            "resolve": self.resolve,
            "spacing": self.spacing,
        }

    def expand(self) -> list:
        """Materialize the template into fully-resolved chart panels.

        Panel iteration shape:

        - 2-D grid (both *row* and *column* set): ``len(row) × len(column)``
          panels, optionally filtered by *corner*; *diagonal* substitutes
          the template on ``row_field == col_field`` panels.
        - 1-D wrap (only one of *row* or *column* set): the populated
          field list, paired with ``None`` on the missing axis.  Geometry
          is applied by :meth:`to_svg` driven by ``columns``.
        - Layer-only (``layer=`` set, *row* and *column* both ``None``):
          a single panel containing all layers.

        When ``layer=`` is set, each panel becomes a layered ``Chart``
        with one layer per element in ``self.layer`` (substituted into
        every ``Repeat.layer`` placeholder).  Diagonal panels skip
        layering — the diagonal template already defines that panel.

        Returns
        -------
        list of tuple
            Each element is ``(row_field, col_field, Chart)`` with all
            ``Repeat.*`` placeholders replaced.  For 1-D and layer-only
            layouts the unused axis is ``None``.

        Raises
        ------
        ValueError
            If *diagonal* is set but ``row != column`` (asymmetric
            repeat), or if the template references a ``Repeat.*``
            placeholder for an axis that was not populated.
        """
        panels = [
            (row_field, col_field, self._make_panel(row_field, col_field))
            for row_field, col_field in self._panel_coordinates()
        ]
        return self._apply_resolve(panels)

    def _apply_resolve(self, panels: list) -> list:
        """Inject shared scales onto every panel per ``self.resolve``.

        For each channel marked ``"shared"``, walks every panel (and every
        layer of layered panels), computes the union domain, and re-emits
        each panel with an explicit ``scale=`` dict on that channel.
        ``"independent"`` channels are no-ops.  When no panel binds a
        shared channel the channel is silently skipped — sharing a
        channel that nothing uses is harmless.
        """
        if not self.resolve:
            return panels
        from ferrum._scale_share import compute_union_domain, inject_scale

        shared = [ch for ch, mode in self.resolve.items() if mode == "shared"]
        if not shared:
            return panels
        result = list(panels)
        for channel in shared:
            charts = [chart for _, _, chart in result]
            scale_dict = compute_union_domain(charts, channel)
            if scale_dict is None:
                continue
            result = [
                (row_field, col_field, inject_scale(chart, channel, scale_dict))
                for row_field, col_field, chart in result
            ]
        return result

    def _panel_coordinates(self) -> list:
        """Compute ``(row_field, col_field)`` pairs for every panel.

        Either entry is ``None`` when the corresponding axis is unset
        (1-D wrap) or both are ``None`` (layer-only).
        """
        if self.row is not None and self.column is not None:
            if self.diagonal is not None and self.row != self.column:
                raise ValueError(
                    "RepeatChart: diagonal= requires row == column "
                    "(diagonal panels only exist on a symmetric grid); "
                    f"got row={self.row!r}, column={self.column!r}"
                )
            coords = []
            for ri, row_field in enumerate(self.row):
                for ci, col_field in enumerate(self.column):
                    if self.corner and ri < ci:
                        continue
                    coords.append((row_field, col_field))
            return coords
        if self.column is not None:
            return [(None, f) for f in self.column]
        if self.row is not None:
            return [(f, None) for f in self.row]
        # layer-only: __init__ already ruled out the all-None axes case.
        return [(None, None)]

    def _make_panel(self, row_field: Optional[str], col_field: Optional[str]):
        """Build the chart for one panel, layering across ``self.layer`` if set."""
        use_diagonal = (
            self.diagonal is not None
            and self.row is not None
            and self.column is not None
            and row_field == col_field
        )
        if use_diagonal:
            # Diagonal panels are intentional overrides; skip layering.
            return self._resolve_template(self.diagonal, row_field, col_field)
        if self.layer is not None:
            layers = [
                self._resolve_template(self.template, row_field, col_field, layer_field=lf)
                for lf in self.layer
            ]
            result = layers[0]
            for nxt in layers[1:]:
                result = result + nxt
            return result
        return self._resolve_template(self.template, row_field, col_field)

    def _resolve_template(
        self,
        source,
        row_field: Optional[str],
        col_field: Optional[str],
        layer_field: Optional[str] = None,
    ):
        """Clone source (a Chart) and substitute Repeat placeholders in encoding.

        Any of the field arguments may be ``None`` when the corresponding
        axis is unset; ``_concrete_field`` raises if the template
        actually references the missing axis.
        """
        from ferrum.repeat import _RepeatPlaceholder
        from ferrum.encoding.base import ChannelBase

        new = source._clone()
        for axis, ch in list(new._encoding.items()):
            if isinstance(ch, _RepeatPlaceholder):
                concrete = self._concrete_field(ch.field, row_field, col_field, layer_field)
                from ferrum.chart import _channel_class_for

                cls = _channel_class_for(axis) or _channel_class_for("x")
                new._encoding[axis] = cls(concrete)
            elif isinstance(ch, ChannelBase) and isinstance(ch.field, _RepeatPlaceholder):
                concrete = self._concrete_field(ch.field.field, row_field, col_field, layer_field)
                new._encoding[axis] = ch.__class__(concrete)
        return new

    @staticmethod
    def _concrete_field(
        placeholder_axis: str,
        row_field: Optional[str],
        col_field: Optional[str],
        layer_field: Optional[str] = None,
    ) -> str:
        """Map a Repeat placeholder axis name to the concrete field string.

        Raises ``ValueError`` if the template references a placeholder for
        an axis that was not populated on the ``RepeatChart`` (e.g.
        ``Repeat.row`` in a column-only 1-D repeat, or ``Repeat.layer``
        without ``layer=``).
        """
        if placeholder_axis == "column":
            if col_field is None:
                raise ValueError("RepeatChart: template uses Repeat.column but column= was not set")
            return col_field
        if placeholder_axis == "row":
            if row_field is None:
                raise ValueError("RepeatChart: template uses Repeat.row but row= was not set")
            return row_field
        if placeholder_axis == "layer":
            if layer_field is None:
                raise ValueError("RepeatChart: template uses Repeat.layer but layer= was not set")
            return layer_field
        raise ValueError(f"unknown Repeat placeholder axis '{placeholder_axis}'")

    def _rebuild_with_charts(self, fn):
        new = RepeatChart(
            fn(self.template),
            row=self.row,
            column=self.column,
            layer=self.layer,
            diagonal=(fn(self.diagonal) if self.diagonal is not None else None),
            corner=self.corner,
            spacing=self.spacing,
            columns=self.columns,
            resolve=self.resolve,
        )
        self._carry_figure_chrome(new)
        return new

    def share_scale(self, **channels):
        """Share scales across this repeat's panels by merging into ``resolve=``.

        Equivalent to constructing the chart with ``resolve={...}`` set
        — both paths run through :meth:`_apply_resolve` at ``expand()``
        time, so the union-domain computation sees every panel (including
        each layer of layered panels) exactly once.  Passing the same
        channel twice with different modes takes the call's value.

        Parameters
        ----------
        **channels : str
            Channel name → ``"shared"`` | ``"independent"``.

        Returns
        -------
        RepeatChart
            A new ``RepeatChart`` with the merged ``resolve=`` config.
        """
        _validate_share_modes(channels)
        merged = dict(self.resolve or {})
        merged.update(channels)
        result = RepeatChart(
            self.template,
            row=self.row,
            column=self.column,
            layer=self.layer,
            diagonal=self.diagonal,
            corner=self.corner,
            spacing=self.spacing,
            columns=self.columns,
            resolve=merged or None,
        )
        _copy_configure_layers(self, result)
        self._carry_figure_chrome(result)
        return result

    def _composite_tree(self, *, auto_tooltips: bool) -> Optional[_LoweredTree]:
        """Lower this repeat grid to a composite grid/hole tree.

        The materialized panels form a row-major grid: a 2-D repeat is a dense
        ``len(row) × len(column)`` grid (with ``corner=True`` filling the upper
        triangle with ``{"kind": "hole"}`` cells); a 1-D repeat wraps by
        ``columns`` into ``nrows × ncols`` with trailing holes. Every present
        cell lowers to a leaf; the shared :func:`_build_grid_tree` builder emits
        the tree consumed by ``render_composite_svg`` /
        ``render_composite_interactive``.

        ``resolve=`` sharing rides the tree's resolve field (the Rust resolve
        pass unions x/y domains across cells) rather than the per-panel
        ``_apply_resolve`` injection the legacy path uses. Panels are therefore
        materialized *without* ``_apply_resolve``.

        Returns
        -------
        _LoweredTree or None
            ``None`` — deferring to the legacy per-child render path — when
            composition-level configure layers are set (their figure-chrome
            positioning stays on the legacy compositor, mirroring
            :func:`_lower_composite`'s ``_configure_layers`` gate), when
            ``resolve=`` marks a non-x/y channel ``"shared"`` (the Rust resolve
            pass spans only x/y), when any materialized panel is not a plain
            leaf ``Chart``, or when any panel's data is empty.
        """
        if getattr(self, "_configure_layers", None):
            return None
        resolve_field = _composite_resolve_field(self.resolve)
        if resolve_field is None:
            return None

        # Materialize raw panels (no _apply_resolve: the tree resolve field
        # drives x/y sharing via the Rust resolve pass).
        panels = [
            (row_field, col_field, self._make_panel(row_field, col_field))
            for row_field, col_field in self._panel_coordinates()
        ]
        charts = [self._inject_parent_config(chart) for _, _, chart in panels]
        if any(not _is_leaf_chart(c) for c in charts):
            return None

        cells: List[Optional[object]]
        if self.row is not None and self.column is not None:
            nrows, ncols = len(self.row), len(self.column)
            cells = [None] * (nrows * ncols)
            for (row_field, col_field, _), chart in zip(panels, charts):
                ri = self.row.index(row_field)
                ci = self.column.index(col_field)
                cells[ri * ncols + ci] = chart
        else:
            ncols, nrows = self._wrap_dimensions(len(charts))
            cells = [None] * (nrows * ncols)
            for idx, chart in enumerate(charts):
                cells[idx] = chart

        return _build_grid_tree(
            cells,
            nrows=nrows,
            ncols=ncols,
            row_ratios=None,
            col_ratios=None,
            spacing=self.spacing,
            auto_tooltips=auto_tooltips,
            title=self._figure_title,
            subtitle=self._figure_subtitle,
            caption=self._figure_caption,
            resolve=resolve_field or None,
        )

    def _render_interactive(self) -> tuple[str, bytes]:
        """Render to (scene_json, packed_data) via the composite grid entry.

        Routes through ``render_composite_interactive`` when the grid lowers
        cleanly (see :meth:`_composite_tree`), otherwise falls back to
        :meth:`_render_interactive_legacy`.
        """
        lowered = self._composite_tree(auto_tooltips=True)
        if lowered is not None:
            from ferrum._core import render_composite_interactive

            return render_composite_interactive(
                lowered.tree,
                lowered.payloads,
                viewport=lowered.viewport,
                theme=lowered.theme,
                chart_config=lowered.chart_config,
            )
        return self._render_interactive_legacy()

    def _render_interactive_legacy(self) -> tuple[str, bytes]:
        """Render to (scene_json, packed_data) by expanding panels and merging scenes."""
        panels = [(r, c, self._inject_parent_config(chart)) for r, c, chart in self.expand()]

        if self.corner and self.row is not None and self.column is not None:
            # Corner mode: panels must be placed at their true (row, col) grid
            # coordinates with gaps for the upper triangle.  Map field names
            # back to integer indices for the sparse grid merge.
            row_index = {v: i for i, v in enumerate(self.row)}
            col_index = {v: i for i, v in enumerate(self.column)}
            indexed = [(row_index[r], col_index[c], chart) for r, c, chart in panels]
            return _merge_child_scenes_sparse_grid(
                indexed,
                self.spacing,
                figure_chrome=self._figure_chrome_kwargs(),
            )

        expanded_charts = [chart for _, _, chart in panels]
        if self.row is not None and self.column is not None:
            n_cols = len(self.column)
        else:
            n_cols, _ = self._wrap_dimensions(len(expanded_charts))
        return _merge_child_scenes_grid(
            expanded_charts,
            self.spacing,
            columns=n_cols,
            figure_chrome=self._figure_chrome_kwargs(),
        )

    def to_svg(self) -> str:
        """Render the repeated grid to an SVG string.

        Routes through ``render_composite_svg`` when the grid lowers cleanly
        (see :meth:`_composite_tree`), otherwise falls back to
        :meth:`_to_svg_legacy`.

        Returns
        -------
        str
            SVG markup containing all materialized panel charts in a grid.

        Notes
        -----
        2-D grids (both ``row`` and ``column`` set) lay out as
        ``len(row) × len(column)``.  1-D layouts (only one axis set) wrap
        by ``columns`` — column-only spreads left-to-right and wraps
        downward; row-only spreads top-to-bottom in a single column unless
        ``columns`` opens additional columns.  When ``columns`` is unset
        the 1-D layout is a single row (column-only) or column (row-only).
        """
        lowered = self._composite_tree(auto_tooltips=False)
        if lowered is not None:
            from ferrum._core import render_composite_svg

            return render_composite_svg(
                lowered.tree,
                lowered.payloads,
                viewport=lowered.viewport,
                theme=lowered.theme,
                chart_config=lowered.chart_config,
            )
        return self._to_svg_legacy()

    def _to_svg_legacy(self) -> str:
        """Compose per-panel SVGs via the Rust string grid compositor (fallback)."""
        from ferrum._core import compose_svg_grid

        panels = [(r, c, self._inject_parent_config(chart)) for r, c, chart in self.expand()]
        if self.row is not None and self.column is not None:
            n_rows = len(self.row)
            n_cols = len(self.column)
            grid: list = [None] * (n_rows * n_cols)
            for row_field, col_field, chart in panels:
                ri = self.row.index(row_field)
                ci = self.column.index(col_field)
                grid[ri * n_cols + ci] = chart.to_svg()
        else:
            n_panels = len(panels)
            n_cols, n_rows = self._wrap_dimensions(n_panels)
            grid = [None] * (n_rows * n_cols)
            for idx, (_, _, chart) in enumerate(panels):
                grid[idx] = chart.to_svg()
        chrome = chrome_kwargs(merge_configure_layers(getattr(self, "_configure_layers", None)))
        return compose_svg_grid(
            grid,
            rows=n_rows,
            cols=n_cols,
            row_ratios=[1.0] * n_rows,
            col_ratios=[1.0] * n_cols,
            spacing=self.spacing,
            title=self._figure_title,
            subtitle=self._figure_subtitle,
            caption=self._figure_caption,
            **chrome,
        )

    def _wrap_dimensions(self, n_panels: int) -> tuple:
        """Compute ``(n_cols, n_rows)`` for a 1-D wrapped layout.

        ``columns=`` is honored when set; otherwise column-only repeats
        produce a single row and row-only repeats produce a single column.
        """
        if self.columns is not None:
            n_cols = min(self.columns, n_panels)
        elif self.column is not None:
            n_cols = n_panels  # horizontal default: one row
        else:
            n_cols = 1  # vertical default: one column
        n_cols = max(1, n_cols)
        n_rows = (n_panels + n_cols - 1) // n_cols
        return n_cols, n_rows

    def __repr__(self) -> str:
        """Return a short developer-readable description."""
        return (
            f"RepeatChart(row={self.row}, column={self.column}, "
            f"diagonal={'set' if self.diagonal is not None else 'None'}, corner={self.corner})"
        )


class ClusterMapChart(_CompositeBase):
    """Clustered heatmap with optional row and column dendrograms.

    Lays out a 2 × 2 grid: the heatmap occupies the bottom-right panel,
    the column dendrogram goes top-right, the row dendrogram (rotated 90°)
    goes bottom-left, and the top-left corner is empty.  Dendrogram value
    axes are hidden; categorical axes align with the heatmap row/column labels.

    Panel size is split by ``dendrogram_ratio``: dendrograms receive that
    fraction of the total width/height, the heatmap receives the remainder.

    Most users obtain a ``ClusterMapChart`` from `ferrum.clustermap` rather
    than constructing one directly.

    Parameters
    ----------
    heatmap : Chart
        The central heatmap chart.
    row_dendrogram : Chart, optional
        Dendrogram chart for the row axis.  Displayed to the left of the
        heatmap, rotated 90°.
    col_dendrogram : Chart, optional
        Dendrogram chart for the column axis.  Displayed above the heatmap.
    dendrogram_ratio : float, default 0.2
        Fraction of the total width/height allocated to each dendrogram panel.
        Must be in the open interval (0, 1).
    spacing : float, default 10.0
        Pixel gap between adjacent panels.

    Raises
    ------
    ValueError
        If *dendrogram_ratio* is not in the open interval (0, 1).

    Examples
    --------
    >>> import ferrum as fm
    >>> cm = fm.clustermap(df, method="ward", cmap="rdbu")
    >>> cm.save("clustermap.svg")
    """

    __slots__ = (
        "heatmap",
        "row_dendrogram",
        "col_dendrogram",
        "dendrogram_ratio",
        "spacing",
    )

    def __init__(
        self,
        heatmap,
        *,
        row_dendrogram=None,
        col_dendrogram=None,
        dendrogram_ratio: float = 0.2,
        spacing: float = 10.0,
    ) -> None:
        if not (0.0 < dendrogram_ratio < 1.0):
            raise ValueError(f"dendrogram_ratio must be in (0, 1); got {dendrogram_ratio}")
        self.heatmap = heatmap
        self.row_dendrogram = row_dendrogram
        self.col_dendrogram = col_dendrogram
        self.dendrogram_ratio = dendrogram_ratio
        self.spacing = spacing
        self._init_figure_chrome()

    @property
    def charts(self) -> list:
        """List of Chart : All non-None sub-charts in ``__init__`` order
        (heatmap, row_dendrogram, col_dendrogram).
        """
        return [
            c for c in (self.heatmap, self.row_dendrogram, self.col_dendrogram) if c is not None
        ]

    @property
    def spec(self) -> dict:
        """Dict : Serializable layout spec consumed by the SVG compositor."""
        return {
            "kind": "cluster_map",
            "heatmap": _embed_chart_spec(self.heatmap),
            "row_dendrogram": _embed_chart_spec(self.row_dendrogram),
            "col_dendrogram": _embed_chart_spec(self.col_dendrogram),
            "dendrogram_ratio": self.dendrogram_ratio,
            "spacing": self.spacing,
        }

    def _forward_child_properties(self, kwargs: dict) -> "ClusterMapChart":
        """Route non-chrome ``properties`` kwargs to the heatmap chart only.

        The dendrogram panels are kept unchanged because their width / height
        is derived from the heatmap plus ``dendrogram_ratio`` at render time.
        Figure-level chrome (``title`` / ``subtitle`` / ``caption``) is
        intercepted by :meth:`_CompositeBase.properties` before this hook
        runs, so it never reaches the heatmap panel.
        """
        result = ClusterMapChart(
            self.heatmap.properties(**kwargs),
            row_dendrogram=self.row_dendrogram,
            col_dendrogram=self.col_dendrogram,
            dendrogram_ratio=self.dendrogram_ratio,
            spacing=self.spacing,
        )
        self._carry_figure_chrome(result)
        return result

    def _rebuild_with_charts(self, fn):
        new = ClusterMapChart(
            fn(self.heatmap),
            row_dendrogram=(fn(self.row_dendrogram) if self.row_dendrogram is not None else None),
            col_dendrogram=(fn(self.col_dendrogram) if self.col_dendrogram is not None else None),
            dendrogram_ratio=self.dendrogram_ratio,
            spacing=self.spacing,
        )
        self._carry_figure_chrome(new)
        return new

    def _pre_resized_dendrograms(self) -> tuple[object, Optional[object], Optional[object]]:
        """Return ``(heatmap, col_dendro, row_dendro)`` with dendrograms pre-sized.

        Dendrograms are resized to their final target dimensions — derived from
        the heatmap's own declared size and ``dendrogram_ratio`` — *before*
        lowering to a leaf spec, rather than relying on the composite grid's
        post-hoc ``layout_scale`` fit. A dendrogram's branch positions are
        computed against its own declared viewport at spec-compile time, so a
        non-uniform post-render stretch would distort branch geometry and
        squash tick labels/strokes non-uniformly; JointChart's marginals (plain
        histograms/KDE/etc.) have no such topology constraint and so
        legitimately rely on the grid's genuine ratio scaling instead.

        With this pre-sizing, each column/row's native extent already exactly
        equals its ratio-derived slot size, so the composite grid's fit-factor
        computation degenerates to identity scale (pure translation) for every
        cell — matching the pre-cutover ``to_svg`` behavior exactly.
        """
        heatmap = self._inject_parent_config(self.heatmap)
        d = self.dendrogram_ratio
        h = 1.0 - d
        hm_w = heatmap._width or 600.0
        hm_h = heatmap._height or 400.0
        dendro_w = hm_w * d / h
        dendro_h = hm_h * d / h
        # Dendrograms have no meaningful axes (only the tree structure
        # matters). clustermap() already calls .axis(show=False) on each
        # dendrogram chart at construction time, so no axis-hiding call is
        # needed here (unlike JointChart's marginals).
        col_dendro = (
            self._inject_parent_config(self.col_dendrogram).properties(width=hm_w, height=dendro_h)
            if self.col_dendrogram is not None
            else None
        )
        row_dendro = (
            self._inject_parent_config(self.row_dendrogram).properties(width=dendro_w, height=hm_h)
            if self.row_dendrogram is not None
            else None
        )
        return heatmap, col_dendro, row_dendro

    def _composite_tree(self, *, auto_tooltips: bool) -> Optional[_LoweredTree]:
        """Lower this ClusterMapChart to a 2×2 ratio/hole composite grid tree.

        Row-major cell layout mirrors the pre-cutover panel positions exactly:

        - both dendrograms: ``[HOLE, col_dendro, row_dendro, heatmap]`` on a
          2×2 grid, ``row_ratios=col_ratios=[d, h]`` where ``d`` is
          ``dendrogram_ratio`` and ``h = 1 - d`` — the empty top-left corner
          becomes a ``{"kind": "hole"}`` cell.
        - one dendrogram: a dense 2×1 or 1×2 grid (no hole).
        - no dendrogram: a dense 1×1 grid — see :meth:`JointChart._composite_tree`
          for why a single-cell tree needs no separate bypass.

        Returns
        -------
        _LoweredTree or None
            ``None`` when heatmap/row_dendrogram/col_dendrogram is not a plain
            leaf ``Chart``, signalling the caller to fall back to the legacy
            per-child render path.
        """
        heatmap, col_dendro, row_dendro = self._pre_resized_dendrograms()

        if any(not _is_leaf_chart(c) for c in (heatmap, col_dendro, row_dendro) if c is not None):
            return None

        d = self.dendrogram_ratio
        h = 1.0 - d

        cells: List[Optional[object]]
        if col_dendro is not None and row_dendro is not None:
            cells = [None, col_dendro, row_dendro, heatmap]
            nrows, ncols = 2, 2
            row_ratios: Optional[List[float]] = [d, h]
            col_ratios: Optional[List[float]] = [d, h]
        elif col_dendro is not None:
            cells = [col_dendro, heatmap]
            nrows, ncols = 2, 1
            row_ratios = [d, h]
            col_ratios = None
        elif row_dendro is not None:
            cells = [row_dendro, heatmap]
            nrows, ncols = 1, 2
            row_ratios = None
            col_ratios = [d, h]
        else:
            cells = [heatmap]
            nrows, ncols = 1, 1
            row_ratios = None
            col_ratios = None

        return _build_grid_tree(
            cells,
            nrows=nrows,
            ncols=ncols,
            row_ratios=row_ratios,
            col_ratios=col_ratios,
            spacing=self.spacing,
            auto_tooltips=auto_tooltips,
            title=self._figure_title,
            subtitle=self._figure_subtitle,
            caption=self._figure_caption,
        )

    def _render_interactive(self) -> tuple[str, bytes]:
        """Render to (scene_json, packed_data) via the composite grid entry.

        Routes through ``render_composite_interactive`` when every panel is a
        plain leaf ``Chart`` (see :meth:`_composite_tree`), otherwise falls
        back to :meth:`_render_interactive_legacy`.
        """
        lowered = self._composite_tree(auto_tooltips=True)
        if lowered is not None:
            from ferrum._core import render_composite_interactive

            return render_composite_interactive(
                lowered.tree,
                lowered.payloads,
                viewport=lowered.viewport,
                theme=lowered.theme,
                chart_config=lowered.chart_config,
            )
        return self._render_interactive_legacy()

    def _render_interactive_legacy(self) -> tuple[str, bytes]:
        """Merge per-child scenes in a 2x2 grid (fallback for non-leaf children).

        Grid layout mirrors the SVG path (``_to_svg_legacy``):
          - col_dendrogram at (row=0, col=1) -- above heatmap
          - row_dendrogram at (row=1, col=0) -- left of heatmap
          - heatmap        at (row=1, col=1) -- main content
        """
        heatmap = self._inject_parent_config(self.heatmap)
        row_dendro = (
            self._inject_parent_config(self.row_dendrogram)
            if self.row_dendrogram is not None
            else None
        )
        col_dendro = (
            self._inject_parent_config(self.col_dendrogram)
            if self.col_dendrogram is not None
            else None
        )
        has_row = row_dendro is not None
        has_col = col_dendro is not None

        # No dendrograms: render heatmap directly (still wrap the figure band).
        if not has_row and not has_col:
            return _render_single_with_figure_chrome(heatmap, self._figure_chrome_kwargs())

        # Build grid panels matching the SVG path layout.
        panels: list[tuple[int, int, object]] = []
        if has_row and has_col:
            # Full 2x2: col_dendro at (0,1), row_dendro at (1,0), heatmap at (1,1).
            panels = [
                (0, 1, col_dendro),
                (1, 0, row_dendro),
                (1, 1, heatmap),
            ]
        elif has_row:
            # Horizontal: row_dendro at (0,0), heatmap at (0,1).
            panels = [(0, 0, row_dendro), (0, 1, heatmap)]
        else:
            # Vertical: col_dendro at (0,0), heatmap at (1,0).
            panels = [(0, 0, col_dendro), (1, 0, heatmap)]

        return _merge_child_scenes_nonuniform_grid(
            panels,
            self.spacing,
            figure_chrome=self._figure_chrome_kwargs(),
        )

    def to_svg(self) -> str:
        """Render the cluster map to an SVG string.

        Routes through ``render_composite_svg`` when every panel is a plain
        leaf ``Chart`` (see :meth:`_composite_tree`), otherwise falls back to
        :meth:`_to_svg_legacy`.

        Returns
        -------
        str
            SVG markup with the 2 × 2 grid layout.
        """
        lowered = self._composite_tree(auto_tooltips=False)
        if lowered is not None:
            from ferrum._core import render_composite_svg

            return render_composite_svg(
                lowered.tree,
                lowered.payloads,
                viewport=lowered.viewport,
                theme=lowered.theme,
                chart_config=lowered.chart_config,
            )
        return self._to_svg_legacy()

    def _to_svg_legacy(self) -> str:
        """Compose per-child SVGs via the string compositor (fallback path)."""
        from ferrum._core import compose_svg_grid

        heatmap, col_dendro, row_dendro = self._pre_resized_dendrograms()
        d = self.dendrogram_ratio
        h = 1.0 - d
        col_svg = col_dendro.to_svg() if col_dendro is not None else None
        row_svg = row_dendro.to_svg() if row_dendro is not None else None
        panels = [None, col_svg, row_svg, heatmap.to_svg()]
        chrome = chrome_kwargs(merge_configure_layers(getattr(self, "_configure_layers", None)))
        return compose_svg_grid(
            panels,
            rows=2,
            cols=2,
            row_ratios=[d, h],
            col_ratios=[d, h],
            spacing=self.spacing,
            title=self._figure_title,
            subtitle=self._figure_subtitle,
            caption=self._figure_caption,
            **chrome,
        )

    def __repr__(self) -> str:
        """Return a short developer-readable description."""
        return (
            f"ClusterMapChart(heatmap=set, row_dendrogram={'set' if self.row_dendrogram else 'None'}, "
            f"col_dendrogram={'set' if self.col_dendrogram else 'None'}, "
            f"ratio={self.dendrogram_ratio})"
        )


# ---------------------------------------------------------------------------
# Phase 12: LayerChart and ConcatChart
# ---------------------------------------------------------------------------


class LayerChart(_ChartLike):
    """Overlay multiple charts on shared axes (same coordinate space).

    All layers share x/y scales by default (union domain).  The charts
    are merged using the same ``Chart + Chart`` layer-merge logic that
    the ``+`` operator provides — domain union, null-padded diagonal
    concat for heterogeneous data, named-transform routing for per-layer
    transforms.

    Use ``LayerChart`` when you have pre-built ``Chart`` objects and want
    a composition-level overlay without constructing the ``+`` chain
    inline.  The resulting SVG is rendered as a single plot area with
    all layers stacked.

    Parameters
    ----------
    *charts : Chart
        Two or more charts to overlay.  At least one chart is required.
    resolve : dict, optional
        Per-channel scale-sharing overrides — e.g.
        ``resolve={"color": "independent"}``.  By default all positional
        channels (x, y) are shared (union domain); non-positional
        channels follow the same inheritance rules as ``Chart + Chart``.
    title : str, optional
        Title applied to the combined chart via ``.properties(title=...)``.

    Raises
    ------
    ValueError
        If fewer than one chart is provided, or if ``resolve`` contains
        invalid values.

    Examples
    --------
    >>> import ferrum as fm
    >>> scatter = fm.Chart(df).mark_point().encode(x="x", y="y")
    >>> line = fm.Chart(df).mark_line().encode(x="x", y="y")
    >>> fm.LayerChart(scatter, line).save("overlay.svg")
    """

    __slots__ = ("_charts", "_resolve", "_title")

    def __init__(
        self,
        *charts,
        resolve: Optional[Dict[str, str]] = None,
        title: Optional[str] = None,
    ) -> None:
        if len(charts) < 1:
            raise ValueError("LayerChart requires at least one chart")
        _validate_resolve(resolve, "LayerChart")
        self._charts = list(charts)
        self._resolve = resolve
        self._title = title

    def __copy__(self):
        """Shallow copy that duplicates the mutable _charts list."""
        new = _shallow_copy_composite(self)
        # _shallow_copy_composite copies _charts as the same list reference;
        # make it a fresh copy so mutations to the original don't affect the copy.
        new._charts = list(self._charts)
        return new

    @property
    def charts(self) -> list:
        """List of Chart : All member charts in layer order (bottom to top)."""
        return list(self._charts)

    def _composite_tree(self, *, auto_tooltips: bool) -> Optional[_LoweredTree]:
        """Lower this overlay to a composite overlay tree.

        Every layer becomes a leaf sharing one panel rect (the Rust overlay
        layout from Task 5b); z-order is layer order — the first chart is drawn
        at the bottom, the last on top. x/y are always shared (union domain),
        matching the legacy ``+``-merge which unconditionally unions the
        positional scales; the overlay is therefore meaningless without it.

        The ``title`` becomes a composite *figure* title (root chrome, the same
        treatment every other composite form gives titles), rather than the
        chart-level title the legacy ``_build_merged`` path applies via
        ``.properties(title=...)``.

        Returns
        -------
        _LoweredTree or None
            ``None`` — deferring to :meth:`_build_merged` (legacy) — when
            composition-level configure layers are set, when ``resolve=`` marks
            a non-x/y channel ``"shared"`` (the Rust resolve pass spans only
            x/y; other channel sharing stays on the Python ``_scale_share``
            path), when any layer is not a plain leaf ``Chart``, or when any
            layer's data is empty.
        """
        if getattr(self, "_configure_layers", None):
            return None
        if self._resolve:
            for channel, mode in self._resolve.items():
                if channel not in ("x", "y") and mode == "shared":
                    return None

        layers = [self._inject_parent_config(c) for c in self._charts]
        if any(not _is_leaf_chart(c) for c in layers):
            return None

        payloads: list = []
        leaf_inputs: list = []
        leaf_nodes: list = []
        children: list = []
        for layer in layers:
            node = _lower_leaf_node(
                layer,
                auto_tooltips=auto_tooltips,
                payloads=payloads,
                leaf_inputs=leaf_inputs,
                leaf_nodes=leaf_nodes,
            )
            if node is None:
                return None
            children.append(node)

        tree: dict = {
            "kind": "composite",
            "layout": "overlay",
            "children": children,
            "spacing": 0.0,
            "resolve": {"x": "shared", "y": "shared"},
        }
        if self._title is not None:
            tree["title"] = self._title

        viewport, theme, chart_config = _apply_leaf_binding_overrides(leaf_nodes, leaf_inputs)
        return _LoweredTree(
            tree=tree,
            payloads=payloads,
            viewport=viewport,
            theme=theme,
            chart_config=chart_config or None,
        )

    def _render_interactive(self) -> tuple[str, bytes]:
        """Render to (scene_json, packed_data) via the merged single-panel Chart.

        Unlike :meth:`to_svg`, this ALWAYS routes through
        :meth:`_render_interactive_legacy` and never through the composite
        overlay tree (see :meth:`_composite_tree`). The interactive contract
        requires LayerChart to produce EXACTLY ONE scene panel: selections,
        hit-testing, and the WASM interaction runtime all assume every layer
        of a ``LayerChart`` shares a single panel. The overlay tree gives each
        layer its own panel that merely shares one *rect* — visually
        identical to the merged single-panel chart in static SVG (no
        panel-identity concept there), but a distinct panel in scene JSON,
        which breaks the one-panel contract. So the static path (``to_svg``)
        keeps the Task 9 overlay-tree cutover; the interactive path stays on
        the legacy merged-chart route.
        """
        return self._render_interactive_legacy()

    def _render_interactive_legacy(self) -> tuple[str, bytes]:
        """Render to (scene_json, packed_data) via the merged multi-layer Chart."""
        from ferrum._scene import _render_scene

        merged = self._build_merged()
        return _render_scene(merged)

    def to_svg(self) -> str:
        """Render the layered charts to an SVG string.

        Routes through ``render_composite_svg`` when the overlay lowers cleanly
        (see :meth:`_composite_tree`), otherwise falls back to
        :meth:`_to_svg_legacy`.

        Returns
        -------
        str
            SVG markup with all layers rendered in a single plot area.
        """
        lowered = self._composite_tree(auto_tooltips=False)
        if lowered is not None:
            from ferrum._core import render_composite_svg

            return render_composite_svg(
                lowered.tree,
                lowered.payloads,
                viewport=lowered.viewport,
                theme=lowered.theme,
                chart_config=lowered.chart_config,
            )
        return self._to_svg_legacy()

    def _to_svg_legacy(self) -> str:
        """Merge layers via the ``Chart + Chart`` operator and render (fallback).

        Merges all layers using the ``Chart + Chart`` operator which handles
        domain union, data merging, and transform routing, then renders the
        resulting multi-layer chart to SVG.
        """
        merged = self._build_merged()
        return merged.to_svg()

    def _build_merged(self):
        """Merge member charts into a single multi-layer Chart via ``+``.

        Applies ``resolve=`` scale sharing, ``title=``, and composition-level
        configure layers when set.
        """
        result = self._charts[0]
        for chart in self._charts[1:]:
            result = result + chart
        if self._resolve:
            shared = [ch for ch, mode in self._resolve.items() if mode == "shared"]
            if shared:
                from ferrum._scale_share import compute_union_domain, inject_scale

                for channel in shared:
                    sd = compute_union_domain(self._charts, channel)
                    if sd is not None:
                        result = inject_scale(result, channel, sd)
        if self._title is not None:
            result = result.properties(title=self._title)
        # Prepend composition-level configure layers so per-chart config wins.
        result = self._inject_parent_config(result)
        return result

    def properties(self, **kwargs):
        """Forward non-chrome ``properties(**kwargs)`` to every layer; store ``title`` locally.

        ``LayerChart`` is a single-plot overlay: it merges its layers into one
        ``Chart`` at render time via :meth:`_build_merged`, which already applies
        ``self._title`` to the merged chart.  Because of that, ``title`` must be
        stored on the ``LayerChart`` itself (not fanned to the inner charts), so that:

        - :meth:`_figure_title_text` (→ ``_title``) returns the correct text for
          the HTML document ``<title>``.
        - :meth:`_build_merged` applies the title to the merged chart's on-plot
          chrome exactly once — inner layers carry no stray title.

        Non-chrome kwargs (``width``, ``height``, ...) are fanned to every layer as
        usual via the base :meth:`_ChartLike.properties` implementation.

        Parameters
        ----------
        **kwargs
            Same keyword arguments accepted by ``Chart.properties``.  ``title``
            is intercepted here; all other kwargs are forwarded to each layer.

        Returns
        -------
        LayerChart
            A new ``LayerChart`` instance with ``_title`` updated and / or
            per-layer properties applied.
        """
        title = kwargs.pop("title", None)

        if kwargs:
            # Fan non-chrome kwargs to every layer.
            result = self._rebuild_with_charts(lambda c: c.properties(**kwargs))
            _copy_configure_layers(self, result)
        else:
            # Nothing to fan — preserve layers unchanged.
            result = copy.copy(self)
            _copy_configure_layers(self, result)

        if title is not None:
            result._title = title

        return result

    def _rebuild_with_charts(self, fn):
        return LayerChart(
            *[fn(c) for c in self._charts],
            resolve=self._resolve,
            title=self._title,
        )

    def __repr__(self) -> str:
        """Return a short developer-readable description."""
        n = len(self._charts)
        return f"LayerChart({n} layer{'s' if n != 1 else ''})"


class ConcatChart(_CompositeBase):
    """General wrapping concatenation of charts in a grid.

    Arranges charts left-to-right, wrapping to the next row after
    ``columns`` charts.  When ``columns`` is ``None``, all charts are
    placed in a single row.

    Parameters
    ----------
    *charts : Chart
        Two or more charts to arrange.
    columns : int, optional
        Maximum number of columns before wrapping.  Defaults to
        ``len(charts)`` (single row, no wrapping).
    spacing : float, default 10.0
        Pixel gap between adjacent panels.
    resolve : dict, optional
        Per-channel scale-sharing overrides — e.g.
        ``resolve={"x": "shared", "y": "shared"}``.

    Raises
    ------
    ValueError
        If fewer than one chart is provided, if ``columns`` is not > 0,
        or if ``resolve`` contains invalid values.

    Examples
    --------
    >>> import ferrum as fm
    >>> charts = [fm.Chart(df).mark_point().encode(x=col, y="y") for col in cols]
    >>> fm.ConcatChart(*charts, columns=2).save("grid.svg")
    """

    __slots__ = ("_columns", "_resolve")

    # ``wrap`` layout on the one-call Rust composite path: children flow
    # left-to-right into rows of ``ncols``, the last row may be partial (no
    # empty-cell concept — sparse holes stay on the old path).  Static +
    # interactive dispatch and the lowering gate are inherited from
    # ``_CompositeBase``; only the ``ncols`` field and the string/scene-merge
    # fallbacks (used when a child cannot lower) are specialised here.
    _composite_layout = "wrap"

    def __init__(
        self,
        *charts,
        columns: Optional[int] = None,
        spacing: float = 10.0,
        resolve: Optional[Dict[str, str]] = None,
    ) -> None:
        if len(charts) < 1:
            raise ValueError("ConcatChart requires at least one chart")
        if columns is not None and columns <= 0:
            raise ValueError(f"ConcatChart: columns must be > 0; got {columns}")
        _validate_resolve(resolve, "ConcatChart")
        super().__init__(list(charts), spacing=spacing)
        self._columns = columns
        self._resolve = resolve

    @property
    def columns(self) -> Optional[int]:
        """Number of columns in the wrapping grid, or None for single-row."""
        return self._columns

    def _wrap_ncols(self) -> int:
        """Resolve the effective column count for the wrapping grid (>= 1)."""
        n_panels = len(self.charts)
        n_cols = self._columns if self._columns is not None else n_panels
        return max(1, min(n_cols, n_panels))

    def _composite_node_fields(self) -> dict:
        """Emit the ``wrap`` layout's ``ncols`` for the composite render-tree."""
        return {"ncols": self._wrap_ncols()}

    def _render_interactive_scene_merge(self) -> tuple[str, bytes]:
        """Merge per-child scenes into a grid (string-merge fallback path)."""
        render_charts = [self._inject_parent_config(c) for c in self._resolved_charts()]
        n_cols = min(self._wrap_ncols(), len(render_charts))
        return _merge_child_scenes_grid(
            render_charts,
            self.spacing,
            columns=n_cols,
            figure_chrome=self._figure_chrome_kwargs(),
        )

    def _to_svg_string_compositor(self) -> str:
        """Compose per-child SVGs into a wrapping grid (string-compositor fallback)."""
        from ferrum._core import compose_svg_grid

        n_panels = len(self.charts)
        n_cols = self._wrap_ncols()
        n_rows = (n_panels + n_cols - 1) // n_cols

        # Apply resolve (shared scales) and composition-level config before rendering
        render_charts = [self._inject_parent_config(c) for c in self._resolved_charts()]

        grid: list = [None] * (n_rows * n_cols)
        for idx, chart in enumerate(render_charts):
            grid[idx] = chart.to_svg()

        chrome = chrome_kwargs(merge_configure_layers(getattr(self, "_configure_layers", None)))
        return compose_svg_grid(
            grid,
            rows=n_rows,
            cols=n_cols,
            row_ratios=[1.0] * n_rows,
            col_ratios=[1.0] * n_cols,
            spacing=self.spacing,
            title=self._figure_title,
            subtitle=self._figure_subtitle,
            caption=self._figure_caption,
            **chrome,
        )

    def _resolved_charts(self) -> list:
        """Return charts with shared scales injected per ``resolve``."""
        return _apply_resolve(self.charts, self._resolve)

    def _rebuild_with_charts(self, fn):
        new = ConcatChart(
            *[fn(c) for c in self.charts],
            columns=self._columns,
            spacing=self.spacing,
            resolve=self._resolve,
        )
        # Carry figure-level chrome through rebuilds.
        self._carry_figure_chrome(new)
        return new

    def __repr__(self) -> str:
        """Return a short developer-readable description."""
        n = len(self.charts)
        return f"ConcatChart({n} chart{'s' if n != 1 else ''}, columns={self._columns})"


# ---------------------------------------------------------------------------
# Layer-composition helpers (extracted from chart.py)
# ---------------------------------------------------------------------------


def _expand_layers(c) -> tuple[list, list]:
    """Return ``(layers, top_level_transforms)`` for one side of ``Chart + Chart``.

    Composite-mark charts arrive pre-layered (``_layers`` is set, ``_mark`` is
    ``None``) -- splat their layers as-is and carry their top-level transforms
    across.  Plain single-mark charts wrap into a one-element ``_Layer`` list.

    Transforms are returned as plain PyO3 objects.  The named-transform path
    (routing a layer's output to a specific ``data_source``) is handled in
    ``__add__`` when the LHS chart has no transforms and the RHS does.

    Encoding-implicit ``_PendingAggregate`` sentinels (added to ``c._transforms``
    by ``encode()`` for channels like ``Y("v", aggregate="mean")``) are excluded
    from the returned top-level transforms.  In a layered chart each layer
    aggregates its own data independently; ``Chart.to_spec`` rebuilds these
    aggregates per-layer from each layer's encoding via
    ``_resolve_layer_aggregates``.  Leaving them at the chart top level would
    aggregate the merged batch once with the wrong (single-layer) groupby.
    """
    from ferrum._layer import _Layer
    from ferrum.encoding.base import _PendingAggregate, _PendingBin

    def _top_transforms(chart) -> list:
        return [
            t
            for t in (chart._transforms or [])
            if not isinstance(t, (_PendingAggregate, _PendingBin))
        ]

    if c._layers is not None:
        return list(c._layers), _top_transforms(c)
    return [
        _Layer(
            mark=c._mark,
            encoding=dict(c._encoding),
            transforms=[],
            mark_kwargs=dict(c._mark_kwargs) if c._mark_kwargs else None,
            position=c._position,
        )
    ], _top_transforms(c)


def _merge_top_transforms(new, rhs_top_xforms: list) -> None:
    """Merge RHS top-level transforms into the combined chart's pipeline.

    Deduplicates by identity first (fast), then by value equality
    (PyO3 transform classes implement ``__eq__`` via ``#[pyclass(eq, ...)]``;
    ``_NamedTransform`` defers to its inner transform for equality checks).
    Value deduplication prevents the same logical transform from running
    twice when both sides of ``+`` use an identical transform object.
    """
    from ferrum._layer_transforms import _NamedTransform

    existing = list(new._transforms or [])
    existing_ids = {id(t) for t in existing}
    for t in rhs_top_xforms:
        if id(t) in existing_ids:
            continue
        # Value dedup: unwrap _NamedTransform for the equality check.
        inner_t = t.transform if isinstance(t, _NamedTransform) else t
        if any(inner_t == (e.transform if isinstance(e, _NamedTransform) else e) for e in existing):
            continue
        existing.append(t)
        existing_ids.add(id(t))
    new._transforms = existing


def _warn_on_layer_conflicts(lhs, rhs) -> None:
    """Warn when layered chart ``+`` would silently discard RHS theme/facet/coord."""
    if (
        (rhs._theme is not None and rhs._theme != lhs._theme)
        or rhs._facet != lhs._facet
        or rhs._coord != lhs._coord
    ):
        import warnings

        warnings.warn(
            "Layered chart `+`: secondary layer's theme/facet/coord is ignored; "
            "primary layer wins.",
            UserWarning,
            stacklevel=3,
        )


def _promote_layer_color(new) -> None:
    """Promote the first layer's ``ChannelBase`` color encoding to chart level when absent.

    When the LHS of ``Chart + Chart`` has no color encoding, the merged chart
    inherits ``_encoding["color"] = None`` from the LHS clone.  The Rust
    renderer builds the chart-level color scale from ``spec.encoding.color``
    only, so a ``None`` chart-level color means no color scale is created —
    every layer that carries a layer-level color encoding then falls back to
    the theme default color, collapsing all categories to one.

    This function scans the merged layers in order and promotes the first
    layer that carries a ``ChannelBase`` color encoding (not a plain string
    shorthand) to the chart level, so ``build_color_scale`` can see the
    scheme and build the correct domain.

    Plain string-valued color encodings (e.g. ``"class"`` from composite-mark
    desugars like ``mark_roc``) are intentionally skipped — they are
    layer-internal shorthands and must not be promoted to chart level because
    ``_build_encoding_specs`` expects ``ChannelBase`` objects there.

    This is a no-op when:
    - the chart-level color encoding is already set (first encoding-bearing
      layer already won via the LHS clone), or
    - no layer carries a ``ChannelBase`` color encoding.
    """
    from ferrum.encoding.base import ChannelBase

    if new._encoding.get("color") is not None:
        return
    for layer in new._layers or []:
        color_ch = layer.encoding.get("color")
        if isinstance(color_ch, ChannelBase):
            new._encoding["color"] = copy.copy(color_ch)
            return
