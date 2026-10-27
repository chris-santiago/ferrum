"""Figure-level override passthrough utility.

``_apply_overrides`` is the single shared entry point called by every figure-
function builder (``_*_chart_from_source``) to merge user overrides into the
assembled chart just before ``.theme()`` is applied.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable

from ferrum._entry import EntryRefusal
from ferrum._layer import drop_orphaned_conditionals, merge_style
from ferrum.encoding._aliases import _prose_list
from ferrum.marks.base import layer_honors, refuse_untaken_layer_style, resolve_layer_style

if TYPE_CHECKING:
    from ferrum.chart import Chart

# Full catalog of possible sub-layer names for each composite mark type.
# Keyed by the ``kind`` string passed to ``_set_composite_mark``. A name in
# the catalog that this call did not draw (boxplot's outlier under
# outliers=False) may be suppressed, since it is already absent, but style
# aimed at it refuses: nothing would draw it.
LAYER_NAME_CATALOG: dict[str, frozenset[str]] = {}

# Figure-level chrome keys that ``_CompositeBase.properties`` intercepts and
# stores on the composed figure rather than fanning to inner panels.  The
# factory ``properties={...}`` dict path must route these the same way the
# chained ``.properties(title=...)`` call does, so they don't leak into a
# sub-panel's SVG.
_FIGURE_CHROME_KEYS = ("title", "subtitle", "caption")


def register_layer_names(kind: str, names: frozenset[str]) -> None:
    """Register possible sub-layer names for *kind* (additive)."""
    existing = LAYER_NAME_CATALOG.get(kind, frozenset())
    LAYER_NAME_CATALOG[kind] = existing | names


def _apply_overrides(
    chart: Any,
    *,
    entry: str,
    mark: dict[str, Any] | None = None,
    encode: dict[str, Any] | None = None,
    properties: dict[str, Any] | None = None,
    layers: list | None = None,
    _landed: set[str] | None = None,
) -> Any:
    """Merge user overrides into *chart* and return the modified copy.

    *entry* is the public call that received the overrides (the figure
    function, or the visualizer whose ``fit`` builds the chart); every
    refusal is an :class:`EntryRefusal` voiced as it.

    Application order: mark → encode → properties → layers.
    For compound views (VConcatChart, HConcatChart, etc.), fans out to
    each child chart via ``_rebuild_with_charts``: each ``mark=`` key
    applies to every child that draws the sub-layer it names, and a key no
    child takes refuses. *_landed* collects, across that fan-out, the keys
    some child took.
    """
    from ferrum.chart import Chart
    from ferrum.composition import LayerChart, _CompositeBase

    if not isinstance(chart, Chart):
        # Split figure-level chrome out of the properties dict for any chart-like
        # that intercepts chrome in its .properties() override.  Currently that is
        # _CompositeBase (all composite figures) and LayerChart (single-plot
        # overlay whose .properties(title=) stores the title on _title rather
        # than fanning it to inner layers).
        #
        # Chrome (title/subtitle/caption) is routed through
        # ``chart.properties(**chrome)`` AFTER the child rebuild, so each type's
        # overridden .properties() handles it correctly:
        #   - _CompositeBase: stores title/subtitle/caption at the figure level.
        #   - LayerChart: stores title on _title; fans subtitle/caption to the
        #     merged chart (LayerChart has no figure band; subtitle/caption
        #     behaviour on LayerChart is intentionally identical to the chained
        #     .properties() path).
        #
        # Only the NON-chrome remainder is fanned to children via _rebuild.
        figure_chrome: dict[str, Any] = {}
        child_properties = properties
        if properties is not None and isinstance(chart, (_CompositeBase, LayerChart)):
            figure_chrome = {k: properties[k] for k in _FIGURE_CHROME_KEYS if k in properties}
            child_properties = {
                k: v for k, v in properties.items() if k not in _FIGURE_CHROME_KEYS
            } or None

        landed: set[str] = set() if _landed is None else _landed

        def _apply(c: Any) -> Any:
            return _apply_overrides(
                c,
                entry=entry,
                mark=mark,
                encode=encode,
                properties=child_properties,
                layers=layers,
                _landed=landed,
            )

        rebuild = getattr(chart, "_rebuild_with_charts", None)
        if rebuild is None:
            # Chart type has no _rebuild_with_charts method at all — fall back
            # to applying properties directly on the top-level chart.
            if child_properties is not None and hasattr(chart, "properties"):
                chart = chart.properties(**child_properties)
        else:
            try:
                chart = rebuild(_apply)
            except NotImplementedError:
                # Chart type declares _rebuild_with_charts (abstract slot on
                # _ChartLike) but has not implemented it — fall back to direct
                # properties application.  AttributeError from a child's
                # .properties() call during _apply now propagates uncaught.
                if child_properties is not None and hasattr(chart, "properties"):
                    chart = chart.properties(**child_properties)

        if figure_chrome:
            chart = chart.properties(**figure_chrome)
        if mark is not None and _landed is None:
            _refuse_unlanded(entry, mark, landed)
        return chart

    if mark is not None:
        chart = _apply_mark_overrides(chart, mark, entry=entry, _landed=_landed)
    if encode is not None:
        chart = chart.encode(**encode)
    if properties is not None:
        chart = chart.properties(**properties)
    if layers is not None:
        chart = chart.layer(*layers)
    return chart


def refuse_malformed_mark(entry: str, mark: Any) -> None:
    """Refuse a ``mark=`` override that is neither ``None`` nor a dict, voiced as *entry*.

    The shape is the caller's input alone, so every public entry that takes
    ``mark=`` (a figure function, a visualizer's ``fit``) runs this before it
    computes anything from the model: the refusal cannot then depend on what
    the computation would have refused first, nor on process state that
    computation reads. :class:`EntryRefusal`.
    """
    if mark is None or isinstance(mark, dict):
        return
    kind = type(mark).__name__
    raise EntryRefusal(
        lambda voice: (
            f"{voice}(mark=...) must be a dict, of mark kwargs or of sub-layer "
            f"names to their mark kwargs; got {kind}"
        ),
        entry,
    )


def _apply_mark_overrides(
    chart: "Chart",
    overrides: Any,
    *,
    entry: str,
    _landed: set[str] | None = None,
) -> "Chart":
    """Apply a figure's ``mark=`` dict to *chart*, voiced as *entry*.

    A chart with named sub-layers takes ``{name: {kwarg: value}}`` (style
    for that sub-layer, held to its mark's honored kwargs) or
    ``{name: False}`` (drop it). Any other chart takes the dict as style for
    its mark, or, layered, each key on every layer whose mark honors it.
    Refusals are :class:`EntryRefusal`, voiced as *entry*. Inside a
    compound fan-out (*_landed* given) a child takes only what applies to
    it and records the keys it took; the caller refuses the rest.
    """
    refuse_malformed_mark(entry, overrides)
    resolved = chart._resolve_pending()
    names = resolved.layer_names

    if not names:
        return _apply_unnamed_style(entry, resolved, overrides, _landed)

    valid = _valid_layer_names(resolved)
    drawn = set(names)
    for key, val in overrides.items():
        if key not in valid:
            if _landed is not None:
                continue
            raise EntryRefusal(unknown_sub_layer_message("mark", key, valid), entry)
        if val is not False and val is not None and not isinstance(val, dict):
            raise EntryRefusal(_non_dict_override_message(key, type(val).__name__), entry)
        # Dropping a layer that is already absent, or styling it with nothing,
        # leaves the chart as asked.
        if key in drawn or not val:
            if _landed is not None:
                _landed.add(key)
            continue
        if _landed is None:
            raise EntryRefusal(undrawn_sub_layer_message("mark", [key], drawn), entry)

    new = resolved._clone()
    new_layers: list = []
    dropped: list = []
    for ly in new._layers or []:
        val = overrides.get(ly.name) if ly.name is not None else None
        if ly.name not in overrides:
            new_layers.append(ly)
        elif val is False or val is None:
            dropped.append(ly)
        else:
            styled = merge_style(ly.mark_kwargs, resolve_layer_style(entry, ly.mark, val))
            new_layers.append(replace(ly, mark_kwargs=styled))
    if not new_layers:
        raise EntryRefusal(
            lambda voice: (
                f"{voice}(mark=...) drops every sub-layer this call draws "
                f"({sorted(drawn)}), so the chart would draw nothing; keep at least one"
            ),
            entry,
        )
    new._layers = new_layers
    new._conditionals = drop_orphaned_conditionals(new._conditionals, new_layers, dropped)
    return new


def _apply_unnamed_style(
    entry: str, chart: "Chart", overrides: dict[str, Any], landed: set[str] | None
) -> "Chart":
    """A ``mark=`` style dict on a chart with no named sub-layers, voiced as *entry*.

    A flat chart takes it as its mark's style. A layered chart's style is
    its own per layer (spec §4.8), so each key reaches every layer whose
    mark honors it, the way a composite's catch-all ``size`` reaches only
    the layers that draw it. A key no layer honors refuses, unless this
    chart is one child of a compound fan-out (*landed* given), where it
    records the keys it took and leaves the rest to its siblings. A layer
    with no mark is held to the point's fields (:func:`ferrum._layer.layer_mark`).
    """
    new = chart._clone()
    marks = [ly.mark for ly in new._layers] if new._layers else [new._mark]
    parts = [{k: v for k, v in overrides.items() if layer_honors(m, k)} for m in marks]
    taken = set().union(*parts)
    if landed is not None:
        landed |= taken
    else:
        refuse_untaken_layer_style(
            entry, marks, {k: v for k, v in overrides.items() if k not in taken}
        )
    styles = [resolve_layer_style(entry, m, part) for m, part in zip(marks, parts)]
    if new._layers:
        new._layers = [
            replace(ly, mark_kwargs=merge_style(ly.mark_kwargs, style))
            for ly, style in zip(new._layers, styles)
        ]
    elif styles[0]:
        new._mark_kwargs = merge_style(new._mark_kwargs, styles[0])
    return new


def _sub_layer_call(voice: str, door: str, names: Sequence[str]) -> str:
    """``voice(door={'a': ..., 'b': ...})``: the call that aimed style at the sub-layers *names*."""
    aimed = ", ".join(f"{name!r}: ..." for name in names)
    return f"{voice}({door}={{{aimed}}})"


def unknown_sub_layer_message(door: str, name: Any, valid: Collection[str]) -> Callable[[str], str]:
    """The refusal text, for any voice, of style aimed through *door* at *name*, no sub-layer.

    *door* is the argument that aims style at a sub-layer by name (a
    figure's ``mark``, a composite mark's ``layer_kwargs``); *valid* is
    every sub-layer name the call could draw. Both doors refuse with it.
    """

    def message_for(voice: str) -> str:
        names = (
            f"this chart's sub-layers are {sorted(valid)}"
            if valid
            else "this chart has no named sub-layers to aim at"
        )
        return f"{_sub_layer_call(voice, door, [name])}: unknown sub-layer {name!r}; {names}"

    return message_for


def undrawn_sub_layer_message(
    door: str, names: Sequence[str], drawn: Collection[str]
) -> Callable[[str], str]:
    """The refusal text, for any voice, of style aimed through *door* at sub-layers not drawn.

    *names* are sub-layers the call could draw but did not this time (a
    boxplot's outliers under ``outliers=False``); *drawn* are those it drew.
    Both doors that aim style at a sub-layer by name refuse with it.
    """
    undrawn = _prose_list([repr(name) for name in names], conjunction="or")
    return lambda voice: (
        f"{_sub_layer_call(voice, door, names)}: this call draws no {undrawn} sub-layer, so "
        f"nothing would take the style; it draws {sorted(drawn)}"
    )


def _non_dict_override_message(key: str, kind: str) -> Callable[[str], str]:
    return lambda entry: (
        f"{entry}(mark={{{key!r}: ...}}) must be a dict of mark kwargs, or False to "
        f"drop the sub-layer; got {kind}"
    )


def _refuse_unlanded(entry: str, overrides: Any, landed: set[str]) -> None:
    """Refuse the ``mark=`` keys no child of a compound figure took, voiced as *entry*."""
    if not isinstance(overrides, dict):
        return
    unlanded = sorted(k for k in overrides if k not in landed)
    if unlanded:
        raise EntryRefusal(
            lambda voice: (
                f"{voice}(mark=...): no panel of this chart draws a sub-layer named, or "
                f"a mark that honors, {unlanded!r}, so nothing would take them"
            ),
            entry,
        )


def _valid_layer_names(chart: "Chart") -> set[str]:
    """Return all *possible* sub-layer names for this chart's mark type.

    Checks the ``LAYER_NAME_CATALOG`` registry first (covers conditional
    layers). Falls back to the currently-active layer names.
    """
    kind = getattr(chart, "_composite_kind", None)
    if kind and kind in LAYER_NAME_CATALOG:
        return set(LAYER_NAME_CATALOG[kind])
    return {ly.name for ly in (chart._layers or []) if ly.name}


def _populate_catalog() -> None:
    """Eagerly import mark modules that register layer names at module scope."""
    import ferrum.marks.composite  # noqa: F401
    import ferrum.marks.heavy_stat  # noqa: F401
    import ferrum.marks.diagnostic._classification  # noqa: F401
    import ferrum.marks.diagnostic._regression  # noqa: F401
    import ferrum.marks.diagnostic._explanation  # noqa: F401
    import ferrum.marks.diagnostic._selection  # noqa: F401


_populate_catalog()
