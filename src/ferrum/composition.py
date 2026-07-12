"""Multi-chart composition primitives (HConcat, VConcat, Layer, Concat, Joint, Repeat, ClusterMap)."""

from __future__ import annotations

import copy
import json as _json
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Union

from ferrum._chrome import chrome_kwargs, merge_configure_layers
from ferrum._configure_mixin import ConfigureMixin
from ferrum._overrides import _FIGURE_CHROME_KEYS


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


def _composite_chrome_kwargs(node) -> dict:
    """Return *node*'s figure-chrome positioning overrides (padding/anchor).

    Wraps ``chrome_kwargs(merge_configure_layers(node._configure_layers))`` --
    resolves ``configure_padding(left=/right=)`` / ``configure_title(anchor=)``
    into the ``{"left_inset": ..., "right_inset": ..., "anchor": ...}`` shape
    the composite render entry's root-only ``config`` wire field accepts
    (Task 10-rust's ``RootChromeConfig``). Only meaningful at the tree root --
    a non-root composite node has no chrome band of its own to position, so
    callers only attach this to the wire ``config`` key when lowering the true
    root (see :func:`_lower_any` and :func:`_build_grid_tree`).
    """
    return chrome_kwargs(merge_configure_layers(getattr(node, "_configure_layers", None)))


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


@dataclass(frozen=True)
class Resolve:
    """Per-channel scale and legend resolution for a composition's ``resolve=``.

    Accepted everywhere a composition takes ``resolve=`` (``HConcatChart``,
    ``VConcatChart``, ``ConcatChart``, ``RepeatChart``, ``LayerChart``, and
    the ``hconcat``/``vconcat``/``concat``/``layer`` sugar) in place of a
    flat ``{channel: mode}`` dict — the flat-dict form remains valid and is
    equivalent to ``Resolve(scale=that_dict)``; both keep meaning *scale*
    resolution.

    Parameters
    ----------
    scale : dict, optional
        Channel name (``"x"``, ``"y"``, ``"color"``, ``"size"``) ->
        ``"shared"`` | ``"independent"``.  Whether panels share a unioned
        domain for that channel — the same axis the flat-dict form controls.
    legend : dict, optional
        Channel name (``"color"`` or ``"size"`` only) ->
        ``"shared"`` | ``"independent"``.  Whether a composite that shares
        *scale* for that channel renders one figure-level legend
        (``"shared"``) or keeps each participating panel's own legend
        (``"independent"``).  Absent from *legend* means "follow the scale
        mode" for that channel — the default, matching Vega-Lite. A
        ``"shared"`` legend mode requires a ``"shared"`` scale mode for the
        same channel; an unsatisfiable combination raises ``ValueError`` at
        render time rather than silently falling back to per-panel legends.
        This shared-legend-requires-shared-scale matrix is enforced at
        render (lowering), not at ``Resolve``/composite construction time —
        constructing ``Resolve(scale={"color": "independent"}, legend={"color": "shared"})``
        succeeds; the mismatch only raises once something renders it.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.HConcatChart([a, b], resolve=fm.Resolve(scale={"color": "shared"}))
    >>> # Force per-panel legends even though the color scale is shared:
    >>> fm.HConcatChart(
    ...     [a, b],
    ...     resolve=fm.Resolve(scale={"color": "shared"}, legend={"color": "independent"}),
    ... )
    """

    scale: Optional[Dict[str, str]] = None
    legend: Optional[Dict[str, str]] = None


# Type accepted wherever a composition takes ``resolve=``: the legacy flat
# scale-mode dict (back-compat) or a Resolve(scale=, legend=) value class.
ResolveArg = Union[Dict[str, str], Resolve, None]

_LEGEND_RESOLVE_CHANNELS = ("color", "size")


def _validate_scale_modes(modes: Optional[Dict[str, str]], label: str, *, field: str) -> None:
    """Raise ``ValueError`` when *modes* is not a valid channel-mode dict.

    Shared by :func:`_validate_resolve`'s flat-dict and ``Resolve.scale``
    branches so both forms enforce the same ``"shared"``/``"independent"``
    vocabulary. *field* names which part of ``resolve=`` is being validated
    (``"resolve"`` for the flat-dict form, ``"scale"`` for ``Resolve.scale``)
    so the error text points at what the caller actually wrote.
    """
    if modes is None:
        return
    if not isinstance(modes, dict):
        raise ValueError(
            f"{label}: {field} must be a dict mapping channel names "
            f"to 'shared' or 'independent'; got {type(modes).__name__}"
        )
    for ch, mode in modes.items():
        if mode not in ("shared", "independent"):
            raise ValueError(
                f"{label}: {field}[{ch!r}]={mode!r}; expected 'shared' or 'independent'"
            )


def _legend_channel_unsupported_error(label: str, channel: str) -> ValueError:
    """Return the ``ValueError`` for a ``resolve.legend`` channel outside color/size.

    Single source of this message, called from both :func:`_validate_legend_modes`
    (construction-time validation, reached for every ``Resolve``-bearing
    composite's ``__init__`` via :func:`_validate_resolve`) and
    :func:`_composite_resolve_field` (render-time lowering) so a caller sees
    identical wording regardless of which validation pass catches the
    unsupported channel first.
    """
    return ValueError(
        f"{label}: resolve.legend[{channel!r}] is not a legend-resolvable channel "
        f"(supported: {_LEGEND_RESOLVE_CHANNELS}); legend resolution only applies to color/size"
    )


def _validate_legend_modes(legend: Optional[Dict[str, str]], label: str) -> None:
    """Raise ``ValueError`` when ``Resolve.legend`` is not a valid legend dict.

    Legend resolution is restricted to ``color``/``size`` (spec §6) — it has
    no meaning for positional channels or any channel the composite resolve
    pass doesn't carry a scale for, so an unsupported channel name is
    rejected here unconditionally (unlike the scale dict, where an
    unsupported channel is only an error when explicitly marked
    ``"shared"`` — see :func:`_composite_resolve_field`).
    """
    if legend is None:
        return
    if not isinstance(legend, dict):
        raise ValueError(
            f"{label}: resolve.legend must be a dict mapping 'color'/'size' to "
            f"'shared' or 'independent'; got {type(legend).__name__}"
        )
    for ch, mode in legend.items():
        if ch not in _LEGEND_RESOLVE_CHANNELS:
            raise _legend_channel_unsupported_error(label, ch)
        if mode not in ("shared", "independent"):
            raise ValueError(
                f"{label}: resolve.legend[{ch!r}]={mode!r}; expected 'shared' or 'independent'"
            )


def _validate_resolve(resolve: ResolveArg, label: str) -> None:
    """Raise ``ValueError`` when *resolve* is not a valid ``resolve=`` value.

    Parameters
    ----------
    resolve : dict, Resolve, or None
        A flat dict is validated as a scale channel-mode mapping (back-compat
        — equivalent to ``Resolve(scale=resolve)``). A :class:`Resolve`
        validates ``.scale`` the same way plus ``.legend`` (restricted to
        ``color``/``size`` — see :func:`_validate_legend_modes`).
    label : str
        Class or function name used in the error message.
    """
    if resolve is None:
        return
    if isinstance(resolve, Resolve):
        _validate_scale_modes(resolve.scale, label, field="resolve.scale")
        _validate_legend_modes(resolve.legend, label)
        return
    if not isinstance(resolve, dict):
        raise ValueError(
            f"{label}: resolve must be a dict, Resolve, or None; got {type(resolve).__name__}"
        )
    _validate_scale_modes(resolve, label, field="resolve")


def _resolve_scale_modes(resolve: ResolveArg) -> Dict[str, str]:
    """Return the scale channel->mode mapping carried by a ``resolve=`` value.

    Used by composition-internal code that needs to inspect scale modes
    directly (``share_scale`` merging, ``LayerChart``'s x/y-independence
    checks) rather than the composite wire field assembled by
    :func:`_composite_resolve_field`. Returns ``{}`` for ``None``, the dict
    itself for the flat-dict form, and ``.scale or {}`` for a
    :class:`Resolve`.
    """
    if resolve is None:
        return {}
    if isinstance(resolve, Resolve):
        return resolve.scale or {}
    return resolve


def _resolve_wire_dict(resolve: ResolveArg) -> Optional[dict]:
    """Return a ``resolve=`` value in its JSON-serializable wire-dict shape.

    The non-validating half of :func:`_assemble_resolve_wire`'s two entry
    points, used by introspection surfaces (``RepeatChart.spec``). For a
    :class:`Resolve` value, this runs the same channel-restriction loop as
    :func:`_composite_resolve_field` (the render-time lowering step), minus
    the raising, so the two can never emit different shapes for the same
    :class:`Resolve` value — see the divergence regression test in
    ``tests/test_composite_shared_legend.py``.

    **Flat-dict raw-view exception (2026-07-12, #74):** ``None`` and the
    flat-dict form do **not** go through that restriction loop at all —
    they pass through unchanged (back-compat: byte-identical, including
    object identity for the flat-dict form). :func:`_composite_resolve_field`
    has no such bypass: it runs a flat dict through the same restriction
    loop as a :class:`Resolve`, with ``validate=True``. So for an
    already-invalid flat dict (one carrying an unsupported channel key —
    reachable only by constructing ``_resolve`` outside the validated
    ``Resolve``/``resolve=`` surface), this function returns it raw while
    :func:`_composite_resolve_field` raises at lowering; the two are *not*
    guaranteed to agree in that case. In practice every flat dict reaching
    either function has already passed :func:`_validate_resolve`, so this
    divergence is unobservable from the public API — it is a deliberate
    identity/back-compat carve-out, not an equivalence guarantee.
    """
    if resolve is None or isinstance(resolve, dict):
        return resolve
    return _assemble_resolve_wire(resolve.scale or {}, resolve.legend or {}, validate=False)


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


def _unsupported_resolve_error(kind: str) -> ValueError:
    """Return the ``ValueError`` for a form with no ``resolve=`` field to share.

    Shared by :meth:`_ChartLike.share_scale`'s ``_supports_user_resolve``
    gate and :class:`JointChart`/:class:`ClusterMapChart`'s
    ``_rebuild_with_charts`` (reached only if a caller passes an explicit
    ``resolve=`` override directly rather than through ``share_scale``, which
    already raises via the ``_supports_user_resolve`` gate before getting
    there) so the message text is identical from both call sites.
    """
    return ValueError(
        f"{kind}: share_scale requires a resolve= field, which {kind} does not carry "
        "(its panel alignment is fixed layout geometry, not a resolve= channel); "
        "construct a composition that supports resolve= (HConcat/VConcat/Concat/"
        "Repeat/Layer) instead"
    )


# Sentinel distinguishing "no override requested" from an explicit
# ``resolve=None`` in ``_rebuild_with_charts(fn, resolve=...)`` overrides --
# see ``_ChartLike.share_scale``, the only caller that passes ``resolve=``.
_RESOLVE_UNCHANGED = object()


def compute_union_domain(charts, channel: str) -> Optional[dict]:
    """Compute a ferrum scale dict spanning *channel* across *charts*.

    Walks every layer of every chart, collects ``(field, data)`` pairs,
    detects the scale type from the first binding's dtype, then either
    unions numeric min/max (linear) or unique values (ordinal).  Time
    domains use the same numeric union path but emit ``type="time"``.

    :meth:`LayerChart._build_merged` (the interactive one-panel merged
    path) is the ONLY production caller left. Every other composition's
    scale sharing -- ``_ChartLike.share_scale``, ``RepeatChart.expand()``
    /``resolve=``, and HConcat/VConcat/ConcatChart/JointChart/ClusterMapChart's
    own ``resolve=`` -- rides the Rust composite resolve pass instead
    (:func:`_composite_resolve_field`), which unions *transform-aware*
    chart extents (e.g. a box mark's whisker reach, a KDE's density
    support) rather than this function's raw column min/max. That
    divergence is real and intentional for now: ``_build_merged`` renders
    through the flat single-Chart entry (the interactive one-panel
    contract), which has no composite tree to carry a resolve field, so
    this raw-column union is the only mechanism available there. This is
    unrelated to the per-layer independent-y mechanism GH #52 shipped
    (2026-07-11, secondary y-axis) -- that work left x/color/size shared
    resolution on this raw-column path unchanged -- and remains open; see
    the "python overlay" S2 row in the code archaeology followups doc.

    Parameters
    ----------
    charts : iterable of Chart
        Charts whose channel will share a domain.
    channel : str
        Encoding channel name (``"x"``, ``"y"``, ``"color"``, ...).

    Returns
    -------
    dict or None
        ``{"type": "linear" | "ordinal" | "time", "domain": [...]}``
        suitable for passing as ``scale=`` on an encoding channel.
        Returns ``None`` when no chart binds the channel, no data is
        available, or the dtype is unsupported.
    """
    from ferrum._render_prepare import (
        _chart_bindings,
        _classify_field,
        _column_minmax,
        _column_unique,
    )

    bindings: list = []
    for chart in charts:
        data = getattr(chart, "_data", None)
        if data is None:
            continue
        for field in _chart_bindings(chart, channel):
            if field is not None:
                bindings.append((field, data))
    if not bindings:
        return None

    first_field, first_data = bindings[0]
    scale_type = _classify_field(first_data, first_field)
    if scale_type is None:
        return None

    if scale_type in ("linear", "time"):
        lo, hi = float("inf"), float("-inf")
        for field, data in bindings:
            extent = _column_minmax(data, field)
            if extent is None:
                continue
            lo = min(lo, extent[0])
            hi = max(hi, extent[1])
        if lo == float("inf"):
            return None
        return {"type": scale_type, "domain": [lo, hi]}

    # ordinal: union of unique values, preserving first-appearance order
    seen: list = []
    seen_set: set = set()
    for field, data in bindings:
        for v in _column_unique(data, field):
            if v not in seen_set:
                seen_set.add(v)
                seen.append(v)
    if not seen:
        return None
    return {"type": "ordinal", "domain": seen}


def inject_scale(chart, channel: str, scale_dict: dict):
    """Return a clone of *chart* with ``scale=scale_dict`` set on *channel*.

    For layered charts each layer's encoding is updated independently.
    Channels not currently bound on the chart (or on a particular layer)
    are left untouched — no implicit binding is added.

    Paired exclusively with :func:`compute_union_domain` at
    :meth:`LayerChart._build_merged` -- the single remaining raw-column
    scale-injection seam (see that function's docstring for why).
    """
    from ferrum._layer import _Layer
    from ferrum.encoding.base import ChannelBase
    from ferrum.chart import _channel_class_for

    def _set_on(value):
        if isinstance(value, ChannelBase):
            new_kwargs = dict(value._kwargs)
            new_kwargs["scale"] = scale_dict
            return type(value)(value.field, **new_kwargs)
        cls = _channel_class_for(channel)
        if cls is None:
            return value
        return cls(value, scale=scale_dict)

    new = chart._clone()
    if new._layers:
        new._layers = [
            _Layer(
                mark=layer.mark,
                encoding={
                    k: (_set_on(v) if k == channel else v) for k, v in layer.encoding.items()
                },
                transforms=layer.transforms,
                mark_kwargs=layer.mark_kwargs,
                data_source=layer.data_source,
                position=layer.position,
            )
            for layer in new._layers
        ]
    else:
        if channel in new._encoding:
            new._encoding[channel] = _set_on(new._encoding[channel])
    return new


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

    def render_svg(self) -> str:
        """Render this lowered tree to an SVG string via the composite entry."""
        from ferrum._core import render_composite_svg

        return render_composite_svg(
            self.tree,
            self.payloads,
            viewport=self.viewport,
            theme=self.theme,
            chart_config=self.chart_config,
        )

    def render_interactive(self) -> tuple[str, bytes]:
        """Render this lowered tree to (scene_json, packed_data) via the composite entry."""
        from ferrum._core import render_composite_interactive

        return render_composite_interactive(
            self.tree,
            self.payloads,
            viewport=self.viewport,
            theme=self.theme,
            chart_config=self.chart_config,
        )


def _is_leaf_chart(node) -> bool:
    """Return True when *node* is a single ``Chart`` that lowers to a tree leaf.

    A ``Chart`` (plain or layered via ``+``) exposes ``_render_inputs`` and is
    not a composition wrapper, so it compiles to one ``ChartSpec`` + payload.
    Composition wrappers (:class:`_ChartLike`) are never leaves.
    """
    return not isinstance(node, _ChartLike) and hasattr(node, "_render_inputs")


_COMPOSITE_RESOLVE_CHANNELS = ("x", "y", "color", "size")


def _assemble_resolve_wire(
    scale: Dict[str, str], legend: Dict[str, str], *, validate: bool, kind: Optional[str] = None
) -> dict:
    """Flatten a ``(scale, legend)`` channel-mode pair to the resolve wire shape.

    The single wire-assembly core shared by :func:`_resolve_wire_dict`
    (introspection, ``validate=False``) and :func:`_composite_resolve_field`
    (render-time lowering, ``validate=True``) for a :class:`Resolve` value
    — both restrict *scale* to :data:`_COMPOSITE_RESOLVE_CHANNELS` and
    *legend* to :data:`_LEGEND_RESOLVE_CHANNELS` through this exact same
    loop, so the two surfaces can never drift apart on *which channels
    survive* for that value shape; the only difference between the two
    modes is the treatment of invalid input: ``validate=True`` raises,
    while ``validate=False`` excludes an *unsupported channel* from the
    result and passes a *mode-matrix violation* (``"shared"`` legend over a
    non-shared scale) through verbatim — introspection is a raw view of
    what was constructed, and the violation still raises at lowering. In
    practice introspection never sees either case anyway, since
    :func:`_validate_resolve`/:func:`_validate_legend_modes` already reject
    an unsupported legend channel at ``Resolve``/composite construction.

    **This equivalence covers only the** :class:`Resolve` **path.** A flat
    dict never reaches this function from :func:`_resolve_wire_dict` (it
    has its own raw-view early return, 2026-07-12 #74) but always reaches
    it from :func:`_composite_resolve_field` (``scale=resolve, legend={}``)
    — so the two callers' overall outputs are not equivalent for the
    flat-dict form the way they are for :class:`Resolve`; see
    :func:`_resolve_wire_dict`'s docstring for that exception.

    **Legend mode-matrix (spec §4/§6).** A channel's effective legend mode is
    the explicit ``legend[channel]`` when given, else that channel's scale
    mode (the default: legend resolution follows scale resolution).
    ``"shared"`` legend resolution requires ``"shared"`` scale resolution for
    the same channel — deduping legends whose domains differ would fabricate
    a mapping no panel uses (spec §4 "Explicit legend resolution", key
    decision 5) — so that combination raises when ``validate=True``, rather
    than silently falling back to per-panel legends.

    Parameters
    ----------
    scale, legend : dict
        Already-flattened channel -> mode maps (a flat-dict ``resolve=`` is
        *scale*, ``{}`` *legend*; a :class:`Resolve` is ``.scale or {}``,
        ``.legend or {}``).
    validate : bool
        ``True`` raises ``ValueError`` on an unsupported ``"shared"`` scale
        channel, an unsupported legend channel, or a ``"shared"`` legend
        mode without a ``"shared"`` effective scale mode. ``False`` excludes
        unsupported channels from the result without raising and passes a
        mode-matrix violation through verbatim (see above).
    kind : str, optional
        Composition class name used in the raised messages. Required when
        ``validate=True``.

    Returns
    -------
    dict
        ``{"x": mode, ...}`` restricted to the supported scale channels,
        plus an optional ``"legend"`` sub-object (spec §6 wire contract)
        when *legend* is non-empty.

    Raises
    ------
    ValueError
        See *validate* above. Only raised when ``validate=True``.
    """
    out: dict = {}
    for channel, mode in scale.items():
        if channel in _COMPOSITE_RESOLVE_CHANNELS:
            out[channel] = mode
        elif mode == "shared" and validate:
            raise ValueError(
                f"{kind}: resolve= marks {channel!r} 'shared', which the composite "
                f"resolve pass does not support (supported: {_COMPOSITE_RESOLVE_CHANNELS}); "
                "set it 'independent' or drop it from resolve="
            )

    if legend:
        legend_out: dict = {}
        for channel, mode in legend.items():
            if channel not in _LEGEND_RESOLVE_CHANNELS:
                if validate:
                    raise _legend_channel_unsupported_error(kind, channel)
                continue
            effective_scale_mode = out.get(channel, "independent")
            if mode == "shared" and effective_scale_mode != "shared":
                if validate:
                    raise ValueError(
                        f"{kind}: resolve.legend[{channel!r}]='shared' requires "
                        f"resolve.scale[{channel!r}]='shared' (got "
                        f"scale={effective_scale_mode!r}, legend='shared'); a shared "
                        "legend needs a unioned domain to dedup from"
                    )
            legend_out[channel] = mode
        out["legend"] = legend_out
    return out


def _composite_resolve_field(resolve: ResolveArg, *, kind: str) -> dict:
    """Map a composition ``resolve=`` value onto a composite node's resolve field.

    The validating half of :func:`_assemble_resolve_wire`'s two entry
    points: the Rust composite resolve pass spans the positional ``x``/``y``
    channels plus ``color``/``size`` (10-pre-b), so a ``"shared"`` request on
    any other channel (``shape``, ``opacity``, …) is not representable there
    and raises rather than silently rendering something other than what the
    caller asked for.

    Parameters
    ----------
    resolve : dict, Resolve, or None
        A flat dict is scale-only (back-compat, equivalent to
        ``Resolve(scale=resolve)``); a :class:`Resolve` additionally supplies
        ``legend``.
    kind : str
        Composition class name, used in error messages.

    Raises
    ------
    ValueError
        When an unsupported channel is marked ``"shared"`` in ``scale``, when
        ``legend`` names a channel other than ``color``/``size``, or when
        ``legend`` requests ``"shared"`` for a channel whose effective scale
        mode is not ``"shared"``. See :func:`_assemble_resolve_wire`.
    """
    if resolve is None:
        return {}
    if isinstance(resolve, Resolve):
        scale, legend = resolve.scale or {}, resolve.legend or {}
    else:
        scale, legend = resolve, {}
    return _assemble_resolve_wire(scale, legend, validate=True, kind=kind)


@dataclass(frozen=True)
class _RootChrome:
    """Root-only figure chrome bundled for one composite-tree node.

    Groups the ``title``/``subtitle``/``caption``/``config`` figure-chrome
    values together with ``is_root`` (is *this* node the tree root?) and
    ``kind`` (the composition class name, for error messages) because the
    five are always constructed together from the same ``self`` at each call
    site and only ever consumed together by :func:`_composite_node`: a nested
    (non-root) composite rejects an explicit subtitle/caption/config and
    lowers its title to a per-child ``"label"`` instead of the root-only
    ``"title"`` key. Bundling them into one small, frozen value object
    collapses :func:`_build_grid_tree`'s former 12-parameter signature (used
    identically by :class:`JointChart`, :class:`RepeatChart`, and
    :class:`ClusterMapChart`) down to one ``chrome`` argument.
    """

    kind: str
    is_root: bool = True
    title: Optional[str] = None
    subtitle: Optional[str] = None
    caption: Optional[str] = None
    config: Optional[dict] = None


def _composite_node(
    layout: str,
    children: list,
    *,
    spacing: float,
    is_root: bool,
    resolve: Optional[dict] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    caption: Optional[str] = None,
    config: Optional[dict] = None,
    **extra_fields,
) -> dict:
    """Build one ``{"kind": "composite", ...}`` wire node.

    The single constructor for every composite-tree node, shared by
    :func:`_lower_any` (hconcat/vconcat/wrap), :func:`_build_grid_tree`
    (grid), and :meth:`LayerChart._composite_tree` (overlay) — previously
    each site hand-assembled this dict independently and re-implemented the
    same root-chrome-vs-label rule.

    *layout* is the wire ``layout`` kind (``"hconcat"``, ``"grid"``,
    ``"overlay"``, ...); ``**extra_fields`` are layout-specific keys merged
    onto the node (e.g. the wrap layout's ``ncols``, the grid layout's
    ``nrows``/``ncols``/``row_ratios``/``col_ratios``).

    The chrome rule: at the tree root (``is_root``), *title*/*subtitle*/
    *caption*/*config* attach directly when given. When nested
    (``is_root=False``), *subtitle*/*caption*/*config* are simply not
    offered here (callers already reject a non-root subtitle/caption before
    calling this), and a non-``None`` *title* lowers to a per-child
    ``"label"`` instead of the root-only ``"title"`` key.

    Returns
    -------
    dict
        The assembled composite node, ready to nest as a child or become the
        tree root.
    """
    node: dict = {"kind": "composite", "layout": layout, "children": children, "spacing": spacing}
    node.update(extra_fields)
    if resolve:
        node["resolve"] = resolve
    if is_root:
        if title is not None:
            node["title"] = title
        if subtitle is not None:
            node["subtitle"] = subtitle
        if caption is not None:
            node["caption"] = caption
        if config:
            node["config"] = config
    elif title is not None:
        node["label"] = title
    return node


def _lower_composite(composite, *, auto_tooltips: bool) -> _LoweredTree:
    """Lower a composition (recursively) to a one-call composite render-tree.

    Entry point for every composite whose class declares a ``_composite_layout``
    wire kind: the linear forms (HConcat/VConcat) and the wrapping-grid
    ``ConcatChart`` (``wrap`` layout). The recursive walk itself lives in
    :func:`_lower_any`, shared with :func:`_build_grid_tree`'s grid cells, so a
    JointChart/RepeatChart/ClusterMapChart/LayerChart nested anywhere in the
    tree — as an HConcat/VConcat child, a grid cell, or arbitrarily deeper —
    lowers to a nested ``CompositeNode::Composite`` rather than a separate
    render path.

    ``auto_tooltips`` mirrors ``Chart._render_scene``: the interactive path
    prepares leaves with auto-tooltips injected, the SVG path does not.

    Raises
    ------
    ValueError
        When the composition cannot be rendered faithfully (see
        :func:`_lower_any` for the specific cases: non-root subtitle/caption,
        an unsupported shared-resolve channel, or an unrecognized node kind),
        or when every child's data is empty (sized holes leave nothing left
        to size cells from — mirrors :func:`_build_grid_tree`'s all-empty
        case for grid composites).
    """
    payloads: list = []
    # (viewport, theme, chart_config) per leaf; when the leaves differ, each
    # leaf node (parallel list below) carries its own binding override.
    leaf_inputs: list = []
    leaf_nodes: list = []

    root = _lower_any(
        composite,
        is_root=True,
        auto_tooltips=auto_tooltips,
        payloads=payloads,
        leaf_inputs=leaf_inputs,
        leaf_nodes=leaf_nodes,
    )
    if not leaf_inputs:
        raise ValueError(
            f"{type(composite).__name__}: every child's data is empty; nothing to render"
        )

    viewport, theme, chart_config = _apply_leaf_binding_overrides(leaf_nodes, leaf_inputs)
    return _LoweredTree(
        tree=root,
        payloads=payloads,
        viewport=viewport,
        theme=theme,
        chart_config=chart_config or None,
    )


def _contains_independent_y_layer(node) -> bool:
    """Return whether *node* is, or nests, a dual-axis (independent-y) chart.

    Used by :func:`_lower_any` to detect a parent composite's explicit
    ``resolve={"y": "shared"}`` colliding with a dual-axis chart anywhere in
    its subtree (GH #52 spec §4 "Nesting"). Dual-axis has two disjoint
    spellings that both flag one or more layers ``independent_y=True`` (GH
    #71): a ``LayerChart(resolve={"y": "independent"})`` (checked via
    ``_y_independent()``) and a plain ``Chart`` produced by
    ``chart + SecondaryY(...)`` (checked via
    :meth:`ferrum.chart.Chart._has_independent_y_layer`, the capability
    predicate mirroring ``_y_independent()`` for that spelling). Either
    lowers to one leaf whose per-layer y-scale slots are resolved
    leaf-locally in Rust -- it does not participate in cross-panel y
    sharing, so an explicit ask to share y across a subtree containing one
    is contradictory and must raise rather than silently drop the caller's
    request. Recurses through every composite form's ``.charts``
    (HConcat/VConcat/wrap children, JointChart/RepeatChart/ClusterMapChart
    cells, LayerChart layers) to catch the conflict at any nesting depth,
    not just an immediate child.
    """
    if isinstance(node, LayerChart) and node._y_independent():
        return True
    if isinstance(node, _ChartLike):
        return any(_contains_independent_y_layer(child) for child in node.charts)
    if _is_leaf_chart(node):
        return node._has_independent_y_layer()
    return False


def _lower_any(
    node,
    *,
    is_root: bool,
    auto_tooltips: bool,
    payloads: list,
    leaf_inputs: list,
    leaf_nodes: list,
) -> dict:
    """Lower one node — a leaf ``Chart`` or any composite class — to its wire dict.

    The single recursive entry point shared by every composite form's tree
    lowering: :func:`_lower_composite` (HConcat/VConcat/ConcatChart children)
    and :func:`_build_grid_tree` (JointChart/ClusterMapChart/RepeatChart grid
    cells). Every form appends into the SAME caller-owned accumulator lists
    (``payloads``/``leaf_inputs``/``leaf_nodes``) so payload indices stay
    globally unique across arbitrary nesting depth — a ``LayerChart`` inside
    an ``HConcatChart`` inside a ``JointChart`` marginal, for instance — and
    only the true tree root finalizes the call-level ``viewport``/``theme``/
    ``chart_config`` default via :func:`_apply_leaf_binding_overrides`.

    ``allow_hole`` mirrors the parent layout's hole support: every non-overlay
    layout (hconcat/vconcat/wrap/grid) accepts a ``{"kind": "hole"}`` cell for
    an empty-data leaf (sized under hconcat/vconcat per Task 10-rust; ignored
    by grid/wrap cell math per spec/Task 8a). Only ``LayerChart``'s overlay
    lowering (which does not route through this generic composite branch)
    excludes empty layers instead of holing them, since a hole is illegal
    under overlay.

    Raises
    ------
    ValueError
        When *node* cannot lower faithfully: a non-root figure *subtitle*/
        *caption* (those stay root-only chrome), a ``resolve=`` request for a
        channel the Rust composite resolve pass doesn't support (see
        :func:`_composite_resolve_field`), or a node type this function does
        not recognize. Every message names the composition's *class*
        (``type(node).__name__``) rather than its wire ``layout`` string, so
        the text always matches what a caller typed (``HConcatChart``, not
        ``hconcat``) per the no-fallback contract (Task 10; no legacy render
        path remains to defer to).
    """
    if _is_leaf_chart(node):
        leaf = _lower_leaf_node(
            node,
            auto_tooltips=auto_tooltips,
            payloads=payloads,
            leaf_inputs=leaf_inputs,
            leaf_nodes=leaf_nodes,
            allow_hole=True,
        )
        assert leaf is not None  # allow_hole=True: empty data lowers to a sized hole
        return leaf

    if isinstance(node, LayerChart) and node._y_independent():
        # An independent-y (dual-axis) LayerChart nested inside a composite
        # does not fit the overlay composite tree (a composite panel carries
        # no per-layer y-scale-slot concept -- see _composite_tree's
        # docstring), so it lowers through the SAME merged flat leaf path
        # to_svg()/._render_interactive() use at the tree root (GH #52 spec
        # §4 "Nesting"): one leaf ChartSpec whose layers carry the
        # independent_y flags, resolved leaf-locally by Rust. The leaf does
        # not participate in cross-panel y sharing -- see the resolve="y":
        # "shared" conflict check below.
        leaf = _lower_leaf_node(
            node._build_merged(),
            auto_tooltips=auto_tooltips,
            payloads=payloads,
            leaf_inputs=leaf_inputs,
            leaf_nodes=leaf_nodes,
            allow_hole=True,
        )
        assert leaf is not None  # allow_hole=True: empty data lowers to a sized hole
        return leaf

    if isinstance(node, (JointChart, ClusterMapChart, RepeatChart, LayerChart)):
        sub = node._composite_tree(auto_tooltips=auto_tooltips, is_root=is_root)
        return _splice_lowered_subtree(
            sub, payloads=payloads, leaf_inputs=leaf_inputs, leaf_nodes=leaf_nodes
        )

    layout = getattr(node, "_composite_layout", None)
    if layout is None:
        raise ValueError(
            f"composition: unrecognized node kind {type(node).__name__!r} in a composite tree"
        )
    kind = type(node).__name__

    if not is_root and (node._figure_subtitle is not None or node._figure_caption is not None):
        raise ValueError(
            f"{kind}: figure subtitle/caption are root-only chrome and cannot be set on "
            "a composite nested inside another composition"
        )
    resolve = _composite_resolve_field(getattr(node, "_resolve", None), kind=kind)
    if resolve.get("y") == "shared" and any(
        _contains_independent_y_layer(child) for child in node.charts
    ):
        raise ValueError(
            f"{kind}: resolve={{'y': 'shared'}} conflicts with a nested independent-y "
            "LayerChart (resolve={'y': 'independent'}) in this subtree -- a dual-axis "
            "LayerChart's per-layer y-scale slots do not participate in cross-panel y "
            "sharing (GH #52 spec §4 'Nesting'); drop the nested LayerChart's "
            "independent-y resolve or remove this composite's explicit y sharing"
        )
    children: list = []
    for child in node.charts:
        child = node._inject_parent_config(child)
        child_node = _lower_any(
            child,
            is_root=False,
            auto_tooltips=auto_tooltips,
            payloads=payloads,
            leaf_inputs=leaf_inputs,
            leaf_nodes=leaf_nodes,
        )
        children.append(child_node)
    # A non-root composite's figure title lowers to a per-child panel label
    # (Task 5d wire), so titled composite compare= panels share axes
    # position-wise instead of gating to the old path (GH #45); see
    # _composite_node's chrome rule.
    return _composite_node(
        layout,
        children,
        spacing=node.spacing,
        resolve=resolve,
        is_root=is_root,
        title=node._figure_title,
        subtitle=node._figure_subtitle,
        caption=node._figure_caption,
        config=(_composite_chrome_kwargs(node) if is_root else None),
        **node._composite_node_fields(),
    )


def _splice_lowered_subtree(
    sub: "_LoweredTree",
    *,
    payloads: list,
    leaf_inputs: list,
    leaf_nodes: list,
) -> dict:
    """Merge an already-lowered nested composite's tree into the parent's lists.

    ``sub`` was produced by a form's own ``_composite_tree()`` (JointChart,
    ClusterMapChart, RepeatChart, or LayerChart) called as a non-root node —
    it only knows its own local payload indices and its own call-level
    ``(viewport, theme, chart_config)`` default. Splicing it into an outer
    tree means every one of its leaves needs an EXPLICIT per-leaf binding
    override (the outer root's eventual default may differ from the
    sub-tree's), so every leaf here has its override written unconditionally
    — mirroring :func:`_apply_leaf_binding_overrides`'s "no silent inherit"
    rule. Returns *sub*'s (mutated in place) tree dict for the caller to use
    as the child node.
    """
    offset = len(payloads)
    payloads.extend(sub.payloads)

    def walk(n: dict) -> None:
        kind = n.get("kind")
        if kind == "leaf":
            n["data"] += offset
            viewport = n.get("viewport", sub.viewport)
            theme = n.get("theme", sub.theme)
            chart_config = n.get("chart_config", sub.chart_config or {})
            n["viewport"] = viewport
            n["theme"] = theme
            n["chart_config"] = chart_config
            leaf_inputs.append((viewport, theme, chart_config))
            leaf_nodes.append(n)
        elif kind == "composite":
            for child in n["children"]:
                walk(child)
        # "hole": no leaf data to splice.

    walk(sub.tree)
    return sub.tree


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
    allow_hole: bool = True,
) -> Optional[dict]:
    """Lower one leaf chart to its wire node, appending to the parallel lists.

    The single source of truth for leaf lowering, shared (via :func:`_lower_any`)
    by every composite form. An empty-data leaf (``num_rows == 0``) lowers to a
    ``{"kind": "hole"}`` placeholder when *allow_hole* is set, sized from the
    viewport the leaf would otherwise have rendered at (matching the space its
    flat ``Chart.to_svg()`` empty-dataset placeholder would have occupied) --
    required under hconcat/vconcat (Task 10-rust's ``HoleSizeRequired``) and
    harmlessly ignored by grid/wrap cell math.  ``allow_hole=False`` is used
    only by :meth:`LayerChart._composite_tree`, whose overlay layout has no
    hole placeholder at all: a ``None`` return there signals the caller to
    SKIP this layer (an empty layer draws no marks either way — matches the
    legacy ``Chart + Chart`` merge behavior), not an error.
    """
    spec, data, viewport, theme, chart_config = chart._render_inputs(_auto_tooltips=auto_tooltips)
    if data.num_rows == 0:
        if allow_hole:
            return {"kind": "hole", "width": viewport[0], "height": viewport[1]}
        return None  # overlay: caller skips this empty-data layer
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
    row_ratios: Optional[List[float]] = None,
    col_ratios: Optional[List[float]] = None,
    spacing: float,
    auto_tooltips: bool,
    resolve: Optional[dict] = None,
    chrome: _RootChrome,
) -> _LoweredTree:
    """Lower a row-major grid of chart cells (with optional holes) to a tree.

    Used by :class:`JointChart`, :class:`ClusterMapChart`, and
    :class:`RepeatChart` — grid composites whose fixed panel slots (and, for
    Joint/ClusterMap's single-marginal corner or Repeat's ``corner=True``
    upper triangle / wrapped trailing cells, one or more unused cells) don't
    fit :func:`_lower_composite`'s generic ``node.charts`` walk. Every entry in
    *cells* is either a leaf ``Chart``, a nested composite (lowered recursively
    via :func:`_lower_any` — a ``LayerChart`` or ``HConcatChart`` used as a
    JointChart marginal, for instance), or ``None`` (a ``{"kind": "hole"}``
    placeholder cell, which the Rust grid layout reserves a slot for but draws
    nothing into — see the Task 8a hole wire). Holes are valid at any grid
    position, not only the 2×2 corner; an empty-data cell also lowers to a hole
    (grid layouts support them) rather than raising.

    *chrome* (see :class:`_RootChrome`) bundles the root-only
    title/subtitle/caption/config values with ``is_root`` and the calling
    class's name (for error messages) into one value object — a 1x1 grid
    wrapping a single chart is a valid composite tree (spec §6), so this same
    builder covers every marginal-count / grid-shape case uniformly, with no
    separate single-chart bypass. When this grid is itself a nested cell
    (``chrome.is_root`` is ``False``), *title* lowers to a per-child
    ``"label"`` instead, and a *subtitle*/*caption*/*config* on a nested grid
    is rejected (those stay root-only chrome).

    Parameters
    ----------
    cells : list of Chart, composite, or None
        Row-major grid cells; ``len(cells)`` must equal ``nrows * ncols``.
    nrows, ncols : int
        Grid dimensions.
    row_ratios, col_ratios : list of float, optional
        Relative row/column sizes (``None`` for a uniform single row/column).
    spacing : float
        Pixel gap between adjacent cells.
    auto_tooltips : bool
        Forwarded to each leaf's ``_render_inputs`` (interactive vs. static).
    resolve : dict, optional
        Composite resolve field (e.g. ``{"x": "shared"}``) attached to the grid
        node so the Rust resolve pass unions the shared channel's domain across
        every cell. ``None`` or empty leaves each cell with independent scales.
    chrome : _RootChrome
        Root-only figure chrome (title/subtitle/caption/config), whether this
        grid is the tree root, and the calling class's name for error text.

    Returns
    -------
    _LoweredTree

    Raises
    ------
    ValueError
        When a nested (non-root) *subtitle*/*caption* is set, or when every
        cell is empty (an all-holes grid has no leaf to size cells from —
        see Task 10-python sub-task 3). Both messages name ``chrome.kind``
        (the calling class, e.g. ``"JointChart"``).
    """
    if not chrome.is_root and (chrome.subtitle is not None or chrome.caption is not None):
        raise ValueError(
            f"{chrome.kind}: figure subtitle/caption are root-only chrome and cannot be set on "
            "a composite nested inside another composition"
        )

    payloads: list = []
    leaf_inputs: list = []
    leaf_nodes: list = []
    children: list = []
    for cell in cells:
        if cell is None:
            children.append({"kind": "hole"})
            continue
        node = _lower_any(
            cell,
            is_root=False,
            auto_tooltips=auto_tooltips,
            payloads=payloads,
            leaf_inputs=leaf_inputs,
            leaf_nodes=leaf_nodes,
        )
        children.append(node)

    if not leaf_nodes:
        # Every cell is empty (e.g. pairplot/jointplot on a zero-row
        # DataFrame): there is no leaf viewport left to size cells from, and
        # grid/wrap holes' width/height are ignored by cell math (Task
        # 10-rust) -- a leafless grid cannot be faithfully sized without a
        # Rust change, so this is a loud, typed failure rather than a
        # silently wrong-size render.
        raise ValueError(
            f"{chrome.kind} ({nrows}x{ncols}): every panel's data is empty; nothing to render"
        )

    extra_fields: dict = {"nrows": nrows, "ncols": ncols}
    if row_ratios is not None:
        extra_fields["row_ratios"] = row_ratios
    if col_ratios is not None:
        extra_fields["col_ratios"] = col_ratios
    tree = _composite_node(
        "grid",
        children,
        spacing=spacing,
        resolve=resolve,
        is_root=chrome.is_root,
        title=chrome.title,
        subtitle=chrome.subtitle,
        caption=chrome.caption,
        config=chrome.config,
        **extra_fields,
    )

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

    # Whether this composition accepts a user-facing resolve= field and
    # therefore supports :meth:`share_scale`. True for the concat/repeat/
    # layer forms (HConcatChart, VConcatChart, ConcatChart, RepeatChart,
    # LayerChart), each of which sets this to True and carries a real
    # ``_resolve`` the composite resolve pass reads. False (the default
    # here) for JointChart/ClusterMapChart, whose panel alignment is fixed
    # layout geometry with no resolve= channel to share into -- share_scale
    # gates on this explicit predicate rather than probing for a private
    # attribute (JointChart also has a ``_resolve`` field internally, for
    # ``jointplot(hue=...)``'s figure-legend wiring, so an attribute probe
    # can no longer double as the "does this support user resolve=" check).
    _supports_user_resolve: bool = False

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
        """Merge ``channels`` into this composition's ``resolve=`` scale field.

        Pure sugar for constructing the same composition with the merged
        scale-mode dict — resolution happens at render time through the
        composite tree, the same Rust resolve pass
        (:func:`_composite_resolve_field`) that ``resolve=`` at construction
        already uses. No ``scale=`` domain dict is computed or injected here,
        and a "shared" union runs over transform-aware chart extents (e.g. a
        box mark's whisker reach, a KDE's density support), not raw column
        min/max — see :func:`compute_union_domain` for the one remaining
        raw-column injection seam (:meth:`LayerChart._build_merged`, the
        interactive one-panel path; GH #52). ``**channels`` only ever
        touches scale resolution; when the existing ``resolve=`` is a
        :class:`Resolve` with a ``legend`` field set, that legend field
        carries through unchanged onto the rebuilt composition. The merged
        result is always stored as a :class:`Resolve` (never the flat-dict
        form), even when no legend override is present -- a stable
        ``_resolve`` type after every ``share_scale`` call. This has no
        effect on rendering: the flat-dict form and ``Resolve(scale=that_dict)``
        lower to identical wire output (byte-identical SVG).

        A child whose channel carries an explicit ``scale=`` (e.g.
        ``fm.Y("y", scale={"domain": [0, 200]})``) is EXCLUDED from the
        shared union and keeps its pinned domain -- an explicit per-chart
        scale always wins over composition-level sharing (spec §6). The
        remaining children still share among themselves; with only one
        unpinned child left, its "union" is simply its own domain.

        Parameters
        ----------
        **channels : str
            Channel name → ``"shared"`` | ``"independent"``.  Common
            channels: ``x``, ``y``, ``color``, ``size``.

        Returns
        -------
        _ChartLike
            A new composition of the same type with the merged
            ``resolve=`` field.  No-op (returns ``self``) when no
            channel is given.

        Raises
        ------
        ValueError
            If any value is not ``"shared"`` or ``"independent"``, or if
            this composition has no ``resolve=`` field to merge into
            (``JointChart``/``ClusterMapChart``: their marginal/dendrogram
            alignment is fixed layout geometry, not a resolve= channel).

        Examples
        --------
        >>> import ferrum as fm
        >>> combined = (chart_a | chart_b).share_scale(x="shared")
        >>> grid = fm.HConcatChart([chart_a, chart_b], resolve={"x": "shared"})
        >>> combined.to_svg() == grid.to_svg()
        True
        """
        _validate_share_modes(channels)
        if not channels:
            return self
        if not self._supports_user_resolve:
            raise _unsupported_resolve_error(type(self).__name__)
        existing = self._resolve
        merged_scale = {**_resolve_scale_modes(existing), **channels}
        existing_legend = existing.legend if isinstance(existing, Resolve) else None
        merged = Resolve(scale=merged_scale, legend=existing_legend)
        result = self._rebuild_with_charts(lambda c: c, resolve=merged)
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

    def _rebuild_with_charts(
        self, fn, *, resolve=_RESOLVE_UNCHANGED
    ):  # pragma: no cover - abstract
        """Return a new composition with each member chart transformed by *fn*.

        Subclasses must implement this — it's the seam between the
        generic ``theme`` / ``properties`` plumbing on the base and each
        composition's constructor signature. Subclasses with
        ``_supports_user_resolve = True`` (and therefore reachable through
        the base :meth:`share_scale`) additionally accept a ``resolve=``
        override — when given, it replaces the rebuilt instance's
        ``resolve=`` instead of preserving the original.
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
    ``_rebuild_with_charts`` / ``_render_interactive`` / ``to_svg`` /
    ``__repr__`` bodies live here once, parameterized by :attr:`_composite_layout`
    (the wire ``layout`` kind the composite render entry uses:
    ``"hconcat"``/``"vconcat"``/``"wrap"``).  This defaults to ``None`` on the
    base; the asymmetric layouts (Joint / Repeat / ClusterMap) and the
    wrapping-grid ``ConcatChart`` override the symmetric methods wholesale, so
    the ``None`` default is never reached for them.
    """

    # Composite-tree layout kind for the one-call Rust composite render path
    # (``render_composite_svg`` / ``render_composite_interactive``); overridden
    # by HConcat/VConcat/ConcatChart.
    _composite_layout: Optional[str] = None

    def __init__(
        self,
        charts: List,
        *,
        spacing: float = 10.0,
        resolve: ResolveArg = None,
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
    # These four methods are the shared bodies of HConcat / VConcat,
    # parameterized by ``_composite_layout``. The asymmetric layouts
    # (Joint / Repeat / ClusterMap) and the wrapping-grid ConcatChart
    # override all four, so the ``None`` hook default is never reached
    # for them.
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

    def _rebuild_with_charts(self, fn, *, resolve=_RESOLVE_UNCHANGED):
        new = type(self)(
            [fn(c) for c in self.charts],
            spacing=self.spacing,
            resolve=(getattr(self, "_resolve", None) if resolve is _RESOLVE_UNCHANGED else resolve),
        )
        self._carry_figure_chrome(new)
        return new

    def _render_interactive(self) -> tuple[str, bytes]:
        """Render to (scene_json, packed_data) for the interactive renderer.

        Routes HConcat/VConcat/ConcatChart through the one-call Rust composite
        entry (``render_composite_interactive``); see :func:`_lower_composite`.
        """
        lowered = _lower_composite(self, auto_tooltips=True)
        return lowered.render_interactive()

    def to_svg(self) -> str:
        """Render the concatenated charts to an SVG string.

        Routes HConcat/VConcat/ConcatChart through the one-call Rust composite
        entry (``render_composite_svg``); see :func:`_lower_composite`.
        """
        lowered = _lower_composite(self, auto_tooltips=False)
        return lowered.render_svg()

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
    resolve : dict or Resolve, optional
        Per-channel scale-sharing overrides, e.g. ``{"color": "shared"}``
        (equivalent to ``Resolve(scale={"color": "shared"})``). Pass a
        :class:`Resolve` to also control figure-level legend resolution,
        e.g. ``Resolve(scale={"color": "shared"}, legend={"color": "independent"})``
        to keep per-panel legends over a shared color scale.  Accepts the
        same keys and values as ``ConcatChart(resolve=...)``.

    Examples
    --------
    >>> import ferrum as fm
    >>> combined = fm.Chart(df).encode(x="hp", y="mpg").mark_point() | fm.Chart(df).encode(x="hp").mark_histogram()
    >>> combined.save("side_by_side.svg")
    """

    # Layout-strategy hook consumed by _CompositeBase's symmetric-concat
    # methods (_render_interactive / to_svg).  Construction, resolve, rebuild,
    # and __repr__ are all inherited unchanged.
    _composite_layout = "hconcat"
    _supports_user_resolve = True


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
    resolve : dict or Resolve, optional
        Per-channel scale-sharing overrides, e.g. ``{"color": "shared"}``
        (equivalent to ``Resolve(scale={"color": "shared"})``). Pass a
        :class:`Resolve` to also control figure-level legend resolution,
        e.g. ``Resolve(scale={"color": "shared"}, legend={"color": "independent"})``
        to keep per-panel legends over a shared color scale.  Accepts the
        same keys and values as ``ConcatChart(resolve=...)``.

    Examples
    --------
    >>> import ferrum as fm
    >>> stacked = fm.Chart(df).encode(x="hp", y="mpg").mark_point() & fm.Chart(df).encode(x="hp").mark_histogram()
    >>> stacked.save("stacked.svg")
    """

    # Layout-strategy hook consumed by _CompositeBase's symmetric-concat
    # methods (_render_interactive / to_svg).  Construction, resolve, rebuild,
    # and __repr__ are all inherited unchanged.
    _composite_layout = "vconcat"
    _supports_user_resolve = True


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
        _resolve: ResolveArg = None,
    ) -> None:
        if ratio <= 0:
            raise ValueError(f"ratio must be > 0; got {ratio}")
        self.center = center
        self.top = top
        self.right = right
        self.ratio = ratio
        self.spacing = spacing
        # Not a public parameter: JointChart has no user-facing resolve=
        # (its panel alignment is fixed layout geometry -- see
        # _unsupported_resolve_error), but ``jointplot(hue=...)`` needs a
        # way to opt the grid it builds into the shared-color legend band
        # (spec §8.6) without exposing share_scale()/resolve= to callers.
        # Named ``_resolve`` honestly (same field name and lowering path
        # every other composition uses) -- ``share_scale()`` gates on the
        # explicit ``_supports_user_resolve`` class attribute rather than
        # probing for this attribute, so JointChart can carry a real
        # ``_resolve`` internally and still correctly fail that gate (its
        # class-level ``_supports_user_resolve`` stays False, inherited
        # unchanged from ``_ChartLike``).
        self._resolve = _resolve
        self._init_figure_chrome()

    @property
    def charts(self) -> list:
        """List of Chart : All non-None sub-charts (center, top, right)."""
        return [c for c in (self.center, self.top, self.right) if c is not None]

    @property
    def spec(self) -> dict:
        """Dict : Serializable layout introspection (ferrum-spec §3.12 contract).

        Embedded charts round-trip through ``ChartSpec.from_json``. Purely an
        introspection/serialization surface — rendering goes through the
        composite spec tree (:meth:`_composite_tree`), not this dict.
        """
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
            _resolve=self._resolve,
        )
        self._carry_figure_chrome(result)
        return result

    def _rebuild_with_charts(self, fn, *, resolve=_RESOLVE_UNCHANGED):
        if resolve is not _RESOLVE_UNCHANGED:
            # Unreachable via the public share_scale() sugar (its
            # _supports_user_resolve gate raises the same error before ever
            # reaching here, since JointChart's class-level
            # _supports_user_resolve is False) -- kept as a defensive typed
            # error for any direct caller, for signature uniformity with the
            # other _rebuild_with_charts forms.
            raise _unsupported_resolve_error(type(self).__name__)
        new = JointChart(
            fn(self.center),
            top=(fn(self.top) if self.top is not None else None),
            right=(fn(self.right) if self.right is not None else None),
            ratio=self.ratio,
            spacing=self.spacing,
            _resolve=self._resolve,
        )
        self._carry_figure_chrome(new)
        return new

    def _composite_tree(self, *, auto_tooltips: bool, is_root: bool = True) -> _LoweredTree:
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
        did not). ``.axis()`` is a plain-``Chart`` method, so a marginal stays
        gated to a leaf chart; the *center* panel carries no such constraint
        and lowers through :func:`_build_grid_tree`'s generic cell handling
        (:func:`_lower_any`), so a nested composite (e.g. a ``LayerChart``
        overlaying a trend line on the center scatter) lowers recursively
        instead of forcing the whole tree to the legacy path.

        ``is_root`` is ``False`` when this JointChart is itself a cell nested
        inside another composite: the figure title then lowers to a per-child
        ``"label"`` and a subtitle/caption declines (root-only chrome).

        ``self._resolve`` (set by ``jointplot(hue=...)`` via the private
        ``_resolve=`` constructor argument, never a public constructor
        argument) lowers onto this grid node's resolve field via
        :func:`_composite_resolve_field`, exactly like :class:`RepeatChart`'s
        public ``resolve=`` -- the Rust resolve pass then unions the
        center/top/right color domains and, when the effective legend mode
        is ``"shared"`` (the default once scale is shared), the compositor
        renders one figure-level legend instead of one per panel (spec §8.6).

        Returns
        -------
        _LoweredTree

        Raises
        ------
        ValueError
            When a marginal (*top*/*right*) is not a plain leaf ``Chart``
            (``axis(show=False)`` requires one).
        """
        center = self._inject_parent_config(self.center)
        top = self._inject_parent_config(self.top) if self.top is not None else None
        right = self._inject_parent_config(self.right) if self.right is not None else None
        if top is not None:
            if not _is_leaf_chart(top):
                raise ValueError(
                    "JointChart: the 'top' marginal must be a plain Chart "
                    "(marginal axis-hiding via axis(show=False) requires one), "
                    f"got {type(top).__name__}"
                )
            top = top.axis(show=False)
        if right is not None:
            if not _is_leaf_chart(right):
                raise ValueError(
                    "JointChart: the 'right' marginal must be a plain Chart "
                    "(marginal axis-hiding via axis(show=False) requires one), "
                    f"got {type(right).__name__}"
                )
            right = right.axis(show=False)

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

        resolve_field = _composite_resolve_field(self._resolve, kind=type(self).__name__)

        return _build_grid_tree(
            cells,
            nrows=nrows,
            ncols=ncols,
            row_ratios=row_ratios,
            col_ratios=col_ratios,
            spacing=self.spacing,
            auto_tooltips=auto_tooltips,
            resolve=resolve_field or None,
            chrome=_RootChrome(
                kind=type(self).__name__,
                is_root=is_root,
                title=self._figure_title,
                subtitle=self._figure_subtitle,
                caption=self._figure_caption,
                config=_composite_chrome_kwargs(self),
            ),
        )

    def _render_interactive(self) -> tuple[str, bytes]:
        """Render to (scene_json, packed_data) via the composite grid entry."""
        lowered = self._composite_tree(auto_tooltips=True)
        return lowered.render_interactive()

    def to_svg(self) -> str:
        """Render the joint chart to an SVG string.

        Returns
        -------
        str
            SVG markup with the 2 × 2 grid layout.
        """
        lowered = self._composite_tree(auto_tooltips=False)
        return lowered.render_svg()

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
    resolve : dict or Resolve, optional
        Per-channel scale-sharing overrides — e.g.
        ``resolve={"x": "shared", "y": "independent"}``.  ``"shared"``
        computes the union domain across all panels (and across every
        layer of layered panels) and injects an explicit scale on every
        participating chart so the axis ticks match.  ``"independent"``
        (the default for unlisted channels) keeps per-panel domains.  Pass
        a :class:`Resolve` to also control figure-level legend resolution
        for a shared ``color``/``size`` scale, e.g.
        ``Resolve(scale={"color": "shared"}, legend={"color": "independent"})``.

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

    _supports_user_resolve = True

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
        resolve: ResolveArg = None,
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
    def _resolve(self) -> ResolveArg:
        """Alias for ``self.resolve`` (the public constructor attribute).

        ``RepeatChart`` exposes its resolve field as the public ``resolve``
        attribute (unlike the other forms' private ``_resolve``) so it can
        appear in :attr:`spec`. This read-only alias lets the base
        :meth:`_ChartLike.share_scale` (gated by ``_supports_user_resolve``)
        read ``self._resolve`` and run its merge logic
        (:func:`_resolve_scale_modes`) for ``RepeatChart`` unchanged, so
        :meth:`share_scale` needs no bespoke override.
        """
        return self.resolve

    @property
    def spec(self) -> dict:
        """Dict : Serializable layout introspection (ferrum-spec §3.12 contract).

        Embedded charts round-trip through ``ChartSpec.from_json``. Purely an
        introspection/serialization surface — rendering goes through the
        composite spec tree (:meth:`_composite_tree`), not this dict.
        """
        return {
            "kind": "repeat",
            "template": _embed_chart_spec(self.template),
            "row": self.row,
            "column": self.column,
            "layer": self.layer,
            "diagonal": _embed_chart_spec(self.diagonal),
            "corner": self.corner,
            "columns": self.columns,
            "resolve": _resolve_wire_dict(self.resolve),
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
            ``Repeat.*`` placeholders replaced. For 1-D and layer-only
            layouts the unused axis is ``None``. Panels are returned
            exactly as materialized — ``resolve=`` is NOT applied here.
            Scale sharing is a render-time concern: :meth:`to_svg` /
            :meth:`_render_interactive` (via :meth:`_composite_tree`) lower
            ``self.resolve`` onto the composite tree's resolve field, which
            the Rust resolve pass unions across cells (see
            :func:`_composite_resolve_field`). A caller that wants shared
            panels should render the ``RepeatChart`` directly rather than
            reading scale dicts off ``expand()``'s output.

        Raises
        ------
        ValueError
            If *diagonal* is set but ``row != column`` (asymmetric
            repeat), or if the template references a ``Repeat.*``
            placeholder for an axis that was not populated.
        """
        return [
            (row_field, col_field, self._make_panel(row_field, col_field))
            for row_field, col_field in self._panel_coordinates()
        ]

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

    def _rebuild_with_charts(self, fn, *, resolve=_RESOLVE_UNCHANGED):
        new = RepeatChart(
            fn(self.template),
            row=self.row,
            column=self.column,
            layer=self.layer,
            diagonal=(fn(self.diagonal) if self.diagonal is not None else None),
            corner=self.corner,
            spacing=self.spacing,
            columns=self.columns,
            resolve=(self.resolve if resolve is _RESOLVE_UNCHANGED else resolve),
        )
        self._carry_figure_chrome(new)
        return new

    def share_scale(self, **channels):
        """Share scales across this repeat's panels by merging into ``resolve=``.

        Pure sugar for :meth:`_ChartLike.share_scale` — ``RepeatChart``
        exposes its resolve field as the public ``resolve`` attribute (see
        the ``_resolve`` alias property), so the base implementation's merge
        logic and ``_rebuild_with_charts(lambda c: c, resolve=merged)`` call
        work unchanged.  Both paths store the identical ``resolve`` dict,
        which :meth:`_composite_tree` lowers onto the composite tree's
        resolve field at render time (:meth:`to_svg` /
        :meth:`_render_interactive`); the Rust resolve pass then unions the
        shared channel's domain across every panel (including each layer of
        layered panels). :meth:`expand` does NOT apply ``resolve=`` — it
        returns panels un-injected regardless of this setting.  Passing the
        same channel twice with different modes takes the call's value.

        Parameters
        ----------
        **channels : str
            Channel name → ``"shared"`` | ``"independent"``.

        Returns
        -------
        RepeatChart
            A new ``RepeatChart`` with the merged ``resolve=`` config.
        """
        return super().share_scale(**channels)

    def _composite_tree(self, *, auto_tooltips: bool, is_root: bool = True) -> _LoweredTree:
        """Lower this repeat grid to a composite grid/hole tree.

        The materialized panels form a row-major grid: a 2-D repeat is a dense
        ``len(row) × len(column)`` grid (with ``corner=True`` filling the upper
        triangle with ``{"kind": "hole"}`` cells); a 1-D repeat wraps by
        ``columns`` into ``nrows × ncols`` with trailing holes. Every present
        cell lowers via the shared :func:`_build_grid_tree` builder (its
        :func:`_lower_any` cell dispatch), which emits the tree consumed by
        ``render_composite_svg`` / ``render_composite_interactive``. Composition-
        level configure layers are pushed onto each panel via
        :meth:`_ChartLike._inject_parent_config` before lowering, so
        composite-level configure (axis/grid/legend/color) needs no separate
        gate here (mirrors :class:`JointChart`/:class:`ClusterMapChart`);
        figure-chrome positioning (``configure_padding``/``configure_title(anchor=)``)
        rides the tree root's ``config`` slot (see :func:`_composite_chrome_kwargs`).

        ``resolve=`` sharing rides the tree's resolve field (the Rust resolve
        pass unions domains across cells for the supported channels — see
        :func:`_composite_resolve_field`, which raises on unsupported shared
        channels). :meth:`expand` already returns panels un-injected, so this
        reuses it directly rather than re-materializing the grid inline.

        Returns
        -------
        _LoweredTree
        """
        resolve_field = _composite_resolve_field(self.resolve, kind="RepeatChart")

        panels = self.expand()
        charts = [self._inject_parent_config(chart) for _, _, chart in panels]

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
            resolve=resolve_field or None,
            chrome=_RootChrome(
                kind=type(self).__name__,
                is_root=is_root,
                title=self._figure_title,
                subtitle=self._figure_subtitle,
                caption=self._figure_caption,
                config=_composite_chrome_kwargs(self),
            ),
        )

    def _render_interactive(self) -> tuple[str, bytes]:
        """Render to (scene_json, packed_data) via the composite grid entry."""
        lowered = self._composite_tree(auto_tooltips=True)
        return lowered.render_interactive()

    def to_svg(self) -> str:
        """Render the repeated grid to an SVG string.

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
        return lowered.render_svg()

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
        """Dict : Serializable layout introspection (ferrum-spec §3.12 contract).

        Embedded charts round-trip through ``ChartSpec.from_json``. Purely an
        introspection/serialization surface — rendering goes through the
        composite spec tree (:meth:`_composite_tree`), not this dict.
        """
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

    def _rebuild_with_charts(self, fn, *, resolve=_RESOLVE_UNCHANGED):
        if resolve is not _RESOLVE_UNCHANGED:
            # Unreachable via the public share_scale() sugar (its
            # _supports_user_resolve gate raises the same error before ever
            # reaching here, since ClusterMapChart keeps it False) -- kept as
            # a defensive typed error for any direct caller, for signature
            # uniformity with the other _rebuild_with_charts forms.
            raise _unsupported_resolve_error(type(self).__name__)
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

    def _composite_tree(self, *, auto_tooltips: bool, is_root: bool = True) -> _LoweredTree:
        """Lower this ClusterMapChart to a 2×2 ratio/hole composite grid tree.

        Row-major cell layout mirrors the pre-cutover panel positions exactly:

        - both dendrograms: ``[HOLE, col_dendro, row_dendro, heatmap]`` on a
          2×2 grid, ``row_ratios=col_ratios=[d, h]`` where ``d`` is
          ``dendrogram_ratio`` and ``h = 1 - d`` — the empty top-left corner
          becomes a ``{"kind": "hole"}`` cell.
        - one dendrogram: a dense 2×1 or 1×2 grid (no hole).
        - no dendrogram: a dense 1×1 grid — see :meth:`JointChart._composite_tree`
          for why a single-cell tree needs no separate bypass.

        Each cell (``heatmap``/``row_dendrogram``/``col_dendrogram``) lowers via
        :func:`_build_grid_tree`'s generic cell dispatch (:func:`_lower_any`),
        so a nested composite dendrogram (e.g. a ``LayerChart`` combining the
        dendrogram with a threshold annotation) lowers recursively rather than
        forcing the whole tree to the legacy path. ``heatmap`` itself must
        still be a plain ``Chart`` in practice — :meth:`_pre_resized_dendrograms`
        reads its declared ``_width``/``_height`` before this method runs, so a
        composite heatmap fails there regardless of this method's own gates.

        Returns
        -------
        _LoweredTree
        """
        heatmap, col_dendro, row_dendro = self._pre_resized_dendrograms()

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
            chrome=_RootChrome(
                kind=type(self).__name__,
                is_root=is_root,
                title=self._figure_title,
                subtitle=self._figure_subtitle,
                caption=self._figure_caption,
                config=_composite_chrome_kwargs(self),
            ),
        )

    def _render_interactive(self) -> tuple[str, bytes]:
        """Render to (scene_json, packed_data) via the composite grid entry."""
        lowered = self._composite_tree(auto_tooltips=True)
        return lowered.render_interactive()

    def to_svg(self) -> str:
        """Render the cluster map to an SVG string.

        Returns
        -------
        str
            SVG markup with the 2 × 2 grid layout.
        """
        lowered = self._composite_tree(auto_tooltips=False)
        return lowered.render_svg()

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


def _validate_layer_resolve(resolve: ResolveArg) -> None:
    """Raise ``ValueError`` when *resolve* marks ``x`` ``"independent"``.

    ``LayerChart`` overlays share one coordinate space along x by design
    (the overlay contract): a dual-x-axis layered chart is not supported
    (see GH #55). Accepting an explicit ``"independent"`` request here
    without raising would mean the rendered axes silently diverge from what
    the caller asked for — the same drift class this closes for
    ``share_scale``.

    ``y: "independent"`` is a supported secondary-axis request (GH #52):
    :meth:`LayerChart.to_svg` and :meth:`LayerChart._render_interactive`
    both route through :meth:`LayerChart._build_merged` (the merged flat
    single-panel path) when *resolve* marks ``y`` ``"independent"``, so no
    validation gate is needed here for that channel.

    Parameters
    ----------
    resolve : dict, Resolve, or None
        The ``resolve=`` value passed to :class:`LayerChart` (already
        validated for mode vocabulary by :func:`_validate_resolve`).
    """
    if _resolve_scale_modes(resolve).get("x") == "independent":
        raise ValueError(
            "LayerChart: layers share one coordinate space (overlay contract); "
            "per-layer independent x scales are not supported "
            "(see GH #55 dual-x-axis)"
        )


class LayerChart(_ChartLike):
    """Overlay multiple charts on shared axes (same coordinate space).

    All layers share x scale by default (union domain) and this cannot be
    turned off — the overlay only makes sense with a single shared x
    coordinate space; see :func:`_validate_layer_resolve`.  ``y`` shares by
    default too, but ``resolve={"y": "independent"}`` renders a dual-axis
    chart instead: layer 0's y-axis on the left, each subsequent layer's own
    y-axis stacked on the right (GH #52) — see :meth:`_build_merged` and
    :meth:`to_svg`.  The charts are merged using the same ``Chart + Chart``
    layer-merge logic that the ``+`` operator provides — domain union,
    null-padded diagonal concat for heterogeneous data, named-transform
    routing for per-layer transforms.

    Render routing (three routes): the ``_y_independent()`` predicate
    selects the static path — (1) static shared/default y → the Phase B
    overlay composite tree (:meth:`_composite_tree`); (2) static
    independent y → the merged flat single-panel chart
    (:meth:`_build_merged` → ``to_svg``) — while (3) the interactive
    entry point ALWAYS uses the merged flat chart regardless of resolve
    (:meth:`_render_interactive_merged`) because selections/hit-testing
    require overlays to be ONE scene panel.  Nested lowering follows the
    same predicate in ``_lower_any``.

    Use ``LayerChart`` when you have pre-built ``Chart`` objects and want
    a composition-level overlay without constructing the ``+`` chain
    inline.  The resulting SVG is rendered as a single plot area with
    all layers stacked.

    Parameters
    ----------
    *charts : Chart
        Two or more charts to overlay.  At least one chart is required.
    resolve : dict or Resolve, optional
        Per-channel scale-sharing overrides — e.g.
        ``resolve={"color": "independent"}``.  ``x`` is always shared (the
        overlay contract) and marking it ``"independent"`` raises (see GH
        #55 dual-x-axis).  ``y: "independent"`` renders a secondary axis
        per non-primary layer (GH #52).  Non-positional channels follow
        the same inheritance rules as ``Chart + Chart``.  Pass a
        :class:`Resolve` to also control figure-level legend resolution for
        a shared ``color``/``size`` scale.
    title : str, optional
        Title applied to the combined chart via ``.properties(title=...)``.

    Raises
    ------
    ValueError
        If fewer than one chart is provided, if ``resolve`` contains
        invalid values, or if ``resolve`` marks ``x`` ``"independent"``
        (see GH #55 dual-x-axis).

    Examples
    --------
    >>> import ferrum as fm
    >>> scatter = fm.Chart(df).mark_point().encode(x="x", y="y")
    >>> line = fm.Chart(df).mark_line().encode(x="x", y="y")
    >>> fm.LayerChart(scatter, line).save("overlay.svg")
    >>> fm.LayerChart(scatter, line, resolve={"y": "independent"}).save("dual.svg")
    """

    __slots__ = ("_charts", "_resolve", "_title")

    _supports_user_resolve = True

    def __init__(
        self,
        *charts,
        resolve: ResolveArg = None,
        title: Optional[str] = None,
    ) -> None:
        if len(charts) < 1:
            raise ValueError("LayerChart requires at least one chart")
        _validate_resolve(resolve, "LayerChart")
        _validate_layer_resolve(resolve)
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

    def _composite_tree(self, *, auto_tooltips: bool, is_root: bool = True) -> _LoweredTree:
        """Lower this overlay to a composite overlay tree.

        Every layer becomes a leaf sharing one panel rect (the Rust overlay
        layout from Task 5b); z-order is layer order — the first chart is drawn
        at the bottom, the last on top. x/y are always shared (union domain),
        matching the legacy ``+``-merge which unconditionally unions the
        positional scales; the overlay is therefore meaningless without it.
        ``resolve=`` on other supported channels (``color``/``size``) rides the
        same tree resolve field (see :func:`_composite_resolve_field`).

        This is exclusively the shared-y path: neither :meth:`to_svg` nor
        :func:`_lower_any` calls this method when ``resolve`` marks ``y``
        ``"independent"`` -- both route through :meth:`_build_merged`
        instead (GH #52), because a composite panel carries no per-layer
        y-scale-slot concept. A y-independent ``LayerChart`` nested as a
        *child* of another composite lowers via :func:`_lower_any`'s
        dedicated branch (one merged flat leaf, per-layer slots resolved
        leaf-locally), not through this overlay-tree method at all.

        Composition-level configure layers are pushed onto each layer via
        :meth:`_ChartLike._inject_parent_config` before lowering, so they need
        no separate gate here (mirrors :class:`JointChart`/:class:`ClusterMapChart`).

        The ``title`` becomes a composite *figure* title (root chrome) when
        this ``LayerChart`` is the tree root, or a per-child ``"label"`` when
        it is nested inside another composite — the same treatment every other
        composite form's title gets, rather than the chart-level title the
        legacy ``_build_merged`` path applies via ``.properties(title=...)``.

        An empty-data layer is SKIPPED (an overlay has no hole placeholder;
        an empty layer draws no marks in the merged ``Chart + Chart`` render
        either). Every layer empty is a typed error.

        Returns
        -------
        _LoweredTree

        Raises
        ------
        ValueError
            When ``resolve=`` marks an unsupported channel ``"shared"``, when
            a layer is not a plain leaf ``Chart``, when a layer carries its
            own ``independent_y=True`` flag (a ``chart + SecondaryY(...)``
            member reaching this shared-y overlay path -- see below), or
            when every layer's data is empty.
        """
        resolve_field = _composite_resolve_field(self._resolve, kind="LayerChart")
        # x is always forced "shared" here regardless of self._resolve --
        # __init__'s _validate_layer_resolve already rejects an explicit
        # "independent" for x (GH #55), so this can never silently override
        # an x request the caller actually made; it only fills in the
        # default when self._resolve left x unset. y is forced "shared"
        # too because neither to_svg() nor _lower_any reaches this method
        # when y is "independent" (both route via _build_merged; see this
        # method's docstring) -- callers that DO reach here always want a
        # shared y.
        resolve_field["x"] = "shared"
        resolve_field["y"] = "shared"

        layers = [self._inject_parent_config(c) for c in self._charts]
        for c in layers:
            if not _is_leaf_chart(c):
                raise ValueError(
                    f"LayerChart: every layer must be a plain Chart, got {type(c).__name__}"
                )
            # A member produced by `chart + SecondaryY(...)` (GH #71) already
            # carries its own flagged independent-y layer -- this shared-y
            # overlay path forces x/y "shared" across every member above, so
            # nesting one here is the same GH #52 spec §4 "Nesting" conflict
            # _lower_any raises for a LayerChart(resolve={"y": "independent"})
            # nested under an explicit parent resolve={"y": "shared"}; raise
            # here instead of silently overlaying the flagged layer onto the
            # shared y-scale it was asked to opt out of.
            if c._has_independent_y_layer():
                raise ValueError(
                    "LayerChart: resolve={'y': 'shared'} (the default when unset) "
                    "conflicts with a member chart produced by `chart + "
                    "SecondaryY(...)` -- its flagged layer's y-scale slot does not "
                    "participate in cross-panel y sharing (GH #52 spec §4 "
                    "'Nesting'); wrap this LayerChart with resolve={'y': "
                    "'independent'} instead, or drop the SecondaryY member"
                )

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
                allow_hole=False,
            )
            if node is None:
                continue  # empty-data layer: draws no marks; skip it
            children.append(node)
        if not children:
            raise ValueError("LayerChart: every layer's data is empty; nothing to render")

        tree = _composite_node(
            "overlay",
            children,
            spacing=0.0,
            resolve=resolve_field,
            is_root=is_root,
            title=self._title,
        )

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

        Unlike :meth:`to_svg`'s default/shared-y path, this ALWAYS routes
        through :meth:`_render_interactive_merged` and never through the
        composite overlay tree (see :meth:`_composite_tree`). The
        interactive contract requires LayerChart to produce EXACTLY ONE
        scene panel: selections, hit-testing, and the WASM interaction
        runtime all assume every layer of a ``LayerChart`` shares a single
        panel. The overlay tree gives each layer its own panel that merely
        shares one *rect* — visually identical to the merged single-panel
        chart in static SVG (no panel-identity concept there), but a
        distinct panel in scene JSON, which breaks the one-panel contract.
        So the default/shared-y static path (``to_svg``) keeps the Task 9
        overlay-tree cutover; the interactive path renders the merged
        single-panel Chart (the FLAT path, not a composition fallback) --
        the same flat path ``to_svg`` also uses for independent y (GH #52).
        """
        return self._render_interactive_merged()

    def _render_interactive_merged(self) -> tuple[str, bytes]:
        """Render to (scene_json, packed_data) via the merged multi-layer Chart.

        The permanent LayerChart interactive path (one-panel contract — see
        :meth:`_render_interactive`): the :meth:`_build_merged` Chart renders
        through the flat single-chart scene entry.
        """
        from ferrum._scene import _render_scene

        merged = self._build_merged()
        return _render_scene(merged)

    def _y_independent(self) -> bool:
        """Return whether ``resolve={"y": "independent"}`` was requested (GH #52)."""
        return _resolve_scale_modes(self._resolve).get("y") == "independent"

    def to_svg(self) -> str:
        """Render the layered charts to an SVG string.

        Default/shared-y renders through the composite overlay tree
        (:meth:`_composite_tree`). ``resolve={"y": "independent"}`` renders
        a dual-axis chart instead: the overlay tree has no per-layer
        y-scale-slot concept, so it routes through the same merged flat
        single-panel path (:meth:`_build_merged`) the interactive output
        already uses (GH #52) -- one implementation serves both output
        kinds for independent y.

        Returns
        -------
        str
            SVG markup with all layers rendered in a single plot area.
        """
        if self._y_independent():
            return self._build_merged().to_svg()
        lowered = self._composite_tree(auto_tooltips=False)
        return lowered.render_svg()

    def _build_merged(self):
        """Merge member charts into a single multi-layer Chart via ``+``.

        Applies ``resolve=`` scale sharing, ``title=``, and composition-level
        configure layers when set.

        This is the single remaining production call site for
        :func:`compute_union_domain`/:func:`inject_scale`'s raw-column scale
        injection (the one-panel interactive-render contract has no
        composite tree to carry a resolve field — see those functions'
        docstrings). Its union semantics for ``color``/``size`` may
        therefore diverge from the static overlay tree's transform-aware
        unions (:meth:`_composite_tree`); see GH #52.

        Secondary y-axis (GH #52): when ``resolve={"y": "independent"}``,
        every merged wire layer contributed by a non-primary member chart
        (``self._charts[1:]``) that carries its own ``y`` encoding is
        marked ``independent_y=True`` -- Rust resolves that layer's y-scale
        independently and renders it as a stacked right axis (spec §6 slot
        contract: layer 0 is always the primary/left axis). A non-primary
        layer with NO ``y`` encoding of its own (e.g. a vertical rule keyed
        only on ``x``) is left unmarked so it joins the primary scale --
        Rust cannot resolve a y-scale for a layer with no y encoding, so
        flagging it would error or produce a phantom axis (spec §4
        "Degenerate cases"). ``y`` is therefore never a member of the
        ``shared`` union-domain list below when independent (its mode is
        ``"independent"``, not ``"shared"``), so no union domain is
        injected for it -- each independent layer resolves natively in
        Rust.

        A non-primary member chart that itself decomposes into MORE THAN
        ONE y-bearing layer (e.g. a composite-mark boxplot member, whose
        box/whisker/outlier layers each encode ``y``) raises instead of
        silently flagging every one of those layers ``independent_y=True``:
        the per-layer boolean wire has no way to group a member's internal
        layers into a single right-axis slot, so it would render one right
        axis per internal layer instead of one grouped axis (a tracked
        follow-up, not this task's scope). The primary (first) member chart
        is exempt -- it always owns the single left axis regardless of how
        many layers it contributes.

        Composite-mark shorthands applied before ``.encode()`` (e.g.
        ``mark_line(point=True)``, which does ``line_chart + point_chart``
        *inside* :meth:`~ferrum.chart.Chart.mark_line` before the caller's
        ``.encode(y=...)`` runs) arrive here with ``chart._layers`` already
        set but every one of those layers' OWN ``encoding`` snapshot empty --
        the subsequent ``.encode(y=...)`` only ever writes ``chart._encoding``
        at the chart level. Each such layer still renders using that
        chart-level ``y`` (an empty per-layer encoding falls back to the
        chart-level encoding at draw time -- confirmed by the non-independent
        default render, where both the line and point layers correctly track
        one shared y-scale despite carrying no per-layer ``y`` of their own).
        So for y-bearing-layer counting purposes each layer without its own
        ``y`` still counts as y-bearing when the member chart carries a
        chart-level ``y`` -- otherwise this composite-mark shape would
        silently join the primary scale (GH #52 Task 10f bug #1) instead of
        hitting the same multi-y-layer guard a pre-merge ``a + b`` (each side
        encoded before ``+``) already hits.
        """
        y_independent = self._y_independent()
        result = self._charts[0]
        n_before = len(result._layers) if result._layers is not None else 1
        for member_index, chart in enumerate(self._charts[1:], start=1):
            # A pre-merged composite-mark member (``chart._layers is not
            # None``) inherits its chart-level ``y`` onto every layer that
            # carries no ``y`` of its own -- see docstring above.
            inherited_y = chart._encoding.get("y") if chart._layers is not None else None
            result = result + chart
            if y_independent:
                layers = list(result._layers)
                y_bearing = [
                    i
                    for i in range(n_before, len(layers))
                    if layers[i].encoding.get("y") is not None
                    or (inherited_y is not None and layers[i].encoding.get("y") is None)
                ]
                if len(y_bearing) > 1:
                    raise ValueError(
                        f"LayerChart: member chart at position {member_index} contributes "
                        f"{len(y_bearing)} y-bearing layers under resolve={{'y': 'independent'}} "
                        "-- member charts under independent y must be a single y-layer chart "
                        "(this includes composite-mark shorthands like mark_line(point=True), "
                        "whose merged line+point layers both inherit the chart-level y; "
                        "grouping a member's internal layers into one right-axis slot is a "
                        "tracked follow-up); only the primary (first) member chart may be "
                        "multi-layer"
                    )
                for i in y_bearing:
                    layers[i] = replace(layers[i], independent_y=True)
                result._layers = layers
            n_before = len(result._layers)
        shared = [
            ch for ch, mode in _resolve_scale_modes(self._resolve).items() if mode == "shared"
        ]
        if shared:
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

    def _rebuild_with_charts(self, fn, *, resolve=_RESOLVE_UNCHANGED):
        return LayerChart(
            *[fn(c) for c in self._charts],
            resolve=(self._resolve if resolve is _RESOLVE_UNCHANGED else resolve),
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
    resolve : dict or Resolve, optional
        Per-channel scale-sharing overrides — e.g.
        ``resolve={"x": "shared", "y": "shared"}``.  Pass a :class:`Resolve`
        to also control figure-level legend resolution for a shared
        ``color``/``size`` scale.

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
    # left-to-right into rows of ``ncols``, the last row may be partial.
    # Static + interactive dispatch are inherited from ``_CompositeBase``;
    # only the ``ncols`` field is specialised here.
    _composite_layout = "wrap"
    _supports_user_resolve = True

    def __init__(
        self,
        *charts,
        columns: Optional[int] = None,
        spacing: float = 10.0,
        resolve: ResolveArg = None,
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

    def _rebuild_with_charts(self, fn, *, resolve=_RESOLVE_UNCHANGED):
        new = ConcatChart(
            *[fn(c) for c in self.charts],
            columns=self._columns,
            spacing=self.spacing,
            resolve=(self._resolve if resolve is _RESOLVE_UNCHANGED else resolve),
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
