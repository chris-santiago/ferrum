"""The operands of a ``+`` composition: one record each, lowered once.

A chart drawn from layers is a list of operands. A flat chart, a composite
mark and a host with layers attached are one operand; ``a + b`` (and
``Chart.layer(fm.Layer(data=...))``, which composes the same way) is the
operands of ``a`` followed by those of ``b``, at any nesting depth. Each
operand reads one frame of the chart's frame registry and carries its own
chain in three parts: the caller's steps, the composition steps that reach
it, and the statistics its mark lowered. Each ``_Layer`` names the operand it
belongs to (``_Layer.operand``) and addresses keys inside that operand's
chain.

``Chart.to_spec`` lowers the operands once (:func:`lower_operands`): a chart
of one operand at the pipeline's top level, and each operand of a
composition as one branch whose chain is ``[own steps][composition
steps][statistics marker][its statistics]``, its layers reading keys scoped
inside it. Every operand lowers to exactly one branch, and operands over one
frame whose chains are equal read the same branch: one branch per distinct
(frame, chain). A composition step reaches the left operand and every operand over
the composition's frame (the left operand's), and never an operand over
another frame, as a Vega-Lite parent transform reaches the children that
share its data.

This module never imports ``Chart`` at module level (``chart.py`` imports
it).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ferrum._encode_lowering import _encoding_statistics, _expand_grouping, _statistic_plan
from ferrum._layer import _Layer
from ferrum._layer_transforms import (
    _Branch,
    _OperandFrame,
    _advances_chain,
    _effective_encoding,
    _source_identity,
    _split_chain,
    _statistic_columns,
    _transforms_to_json_list_named,
    _with_statistics_marker,
)
from ferrum.encoding.base import ChannelBase

#: The key an operand's rows are published under, after its own steps and the
#: composition steps that reach it: the rows a layer attached to the chart
#: reads (``Chart.layer``, ``SecondaryY``).
ROWS_KEY = "_rows"

#: A pass-through step: the chain of an operand branch over the composition's
#: own frame with no step of its own.
_PASS_THROUGH = {"type": "identity"}


@dataclass(frozen=True)
class _Operand:
    """One operand of a chart drawn from layers. Never serialized.

    Attributes
    ----------
    frame :
        Index of the frame it reads in the chart's frame registry
        (:func:`chart_frames`); ``0`` is the composition's frame, its left
        operand's.
    own :
        The caller's steps, written with ``.transform(...)`` on the chart
        the operand came from.
    statistics :
        The statistics its mark lowered when it resolved (a composite mark's
        bins, density grid or fitted curve), without the statistics marker.
    composed :
        The steps written on a composition this operand is reached by, each
        the :class:`~ferrum._layer_transforms._ComposedStep` every reached
        operand shares, innermost composition first.
    """

    frame: int
    own: tuple
    statistics: tuple
    composed: tuple = ()


def chart_operands(chart: Any) -> tuple[_Operand, ...]:
    """*chart*'s operands: a composition's own, else the one its chain makes."""
    if chart._operands:
        return chart._operands
    own, statistics = _split_chain(list(chart._transforms or []))
    return (_Operand(0, own, statistics),)


def chart_frames(chart: Any) -> tuple:
    """*chart*'s frame registry: a composition's, else its own data."""
    return chart._frames or (chart._data,)


def _same_frame(a: Any, b: Any) -> bool:
    """Whether two operand frames are one frame.

    The same object is one frame. Otherwise two frames are one frame when
    they hold the same schema (column names, order and dtypes) and equal
    values, NaN equal to NaN and null equal to null, so a clone, a frame
    rebuilt from the same values and a ``LazyFrame`` collected again by a
    nested ``+`` are the frame they copy.
    """
    if a is b:
        return True
    if a is None or b is None:
        return False
    from ferrum._coerce import to_polars

    a, b = to_polars(a), to_polars(b)
    return a.schema == b.schema and a.equals(b, null_equal=True)


def merged_frames(lhs: tuple, rhs: tuple) -> tuple[tuple, list[int]]:
    """The registry of ``a + b`` from the operands' registries, and where each *rhs* frame went.

    Each right frame equal to one already registered (:func:`_same_frame`)
    is that frame; any other is appended. The left registry keeps its
    order, so frame ``0`` stays the composition's frame.
    """
    frames = list(lhs)
    at: list[int] = []
    for frame in rhs:
        index = next((i for i, known in enumerate(frames) if _same_frame(known, frame)), None)
        if index is None:
            index = len(frames)
            frames.append(frame)
        at.append(index)
    return tuple(frames), at


def combined_frames(frames: tuple) -> tuple[Any, tuple[_OperandFrame, ...]]:
    """The one batch a composition over *frames* sends, and each frame's selector in it.

    One frame is sent as it is, and needs no selector. Several are
    diagonally concatenated: each frame's rows carry a marker column, true
    on its own rows, and each column whose name an earlier frame already
    holds is renamed (``<column>__rhs_<frame>``), so no frame's rows read
    another's values. Generated names never collide with any frame's
    column. Data that is no supported frame is refused by ``to_polars``.
    """
    if len(frames) == 1:
        return frames[0], ()
    import polars as pl

    from ferrum._coerce import to_polars

    data = [to_polars(frame) for frame in frames]
    taken: set[str] = set().union(*(set(frame.columns) for frame in data))

    def fresh(base: str) -> str:
        name, n = base, 0
        while name in taken:
            n += 1
            name = f"{base}_{n}"
        taken.add(name)
        return name

    held: set[str] = set()
    parts = []
    selectors = []
    for i, frame in enumerate(data):
        renames = {c: fresh(f"{c}__rhs_{i:04d}") for c in frame.columns if c in held}
        marker = fresh(f"__operand_row_{i:04d}")
        part = frame.rename(renames).with_columns(pl.lit(True).alias(marker))
        held |= set(part.columns)
        parts.append(part)
        selectors.append(_OperandFrame(marker, tuple(renames.items()), tuple(frame.columns)))
    return pl.concat(parts, how="diagonal"), tuple(selectors)


def reached_by_composition(operands: tuple[_Operand, ...], steps: tuple) -> tuple[_Operand, ...]:
    """*operands* with *steps*, written on their composition, added to every operand over its frame."""
    from dataclasses import replace

    return tuple(
        replace(op, composed=op.composed + steps) if op.frame == 0 else op for op in operands
    )


# ---------------------------------------------------------------------------
# The columns of the rows a layer reads, where Python knows them
# ---------------------------------------------------------------------------


def _frame_columns_after(chart: Any, op: _Operand, steps: tuple) -> list[str] | None:
    """The columns of *op*'s frame after *steps*, when no step of them advances; else ``None``."""
    data = chart_frames(chart)[op.frame]
    if data is None or any(_advances_chain(t) for t in steps):
        return None
    return list(data.columns)


def known_chart_data_columns(chart: Any) -> list[str] | None:
    """The columns of *chart*'s data, when Python knows them; else ``None``.

    A chart's data is its left operand's rows after that operand's own
    steps and the composition steps that reach it: on a ``+`` composition
    the left operand's frame, never the combined batch. Rows a step
    produced are the engine's to know.
    """
    op = chart_operands(chart)[0]
    return _frame_columns_after(chart, op, op.own + op.composed)


def known_columns(layer: _Layer, chart: Any) -> list[str] | None:
    """The columns of the rows *layer* reads, when Python knows them; else ``None``.

    A layer reading its chart's data (``chart_data`` / ``reads_chart_data``)
    reads its operand's rows; a layer reading its operand's tail reads them
    after the operand's statistics too. A layer reading a named output is
    the engine's to know.
    """
    op = chart_operands(chart)[layer.operand]
    if layer.chart_data or layer.reads_chart_data:
        return _frame_columns_after(chart, op, op.own + op.composed)
    if layer.data_source is not None:
        return None
    return _frame_columns_after(chart, op, op.own + op.composed + op.statistics)


def refuse_missing_statistic_inputs(layers: list, chart: Any) -> None:
    """Refuse a per-layer statistic whose input or groupby column the layer's rows lack.

    Each layer's statistic is read off its effective encoding over *chart*'s
    (:func:`~ferrum._layer_transforms._effective_encoding`), so a statistic
    it inherits from its parent is checked over the rows it reads too.
    Checked where Python knows those rows (:func:`known_columns`). A missing
    column is voiced by the render gate's own message
    (``ferrum._core.layer_unknown_column_message``), naming a layer a figure
    or composite mark built by that call (``_Layer.entry``) as the gate
    does: the statistic remedy when it is another layer's statistic output
    or a column the chart's composite mark binds (one its layers reading a
    named output draw), else the plain message. Any other layer is left to
    the transform stage.
    """
    from ferrum._core import layer_unknown_column_message

    effective = [_effective_encoding(layer, chart._encoding) for layer in layers]
    statistics: set = set()
    for layer, encoding in zip(layers, effective):
        statistics |= _statistic_columns(encoding)
        if (
            chart._composite_kind is not None
            and layer.data_source is not None
            and not layer.reads_chart_data
        ):
            statistics |= {
                ch.field for ch in layer.encoding.values() if isinstance(ch, ChannelBase)
            }
    for i, (layer, encoding) in enumerate(zip(layers, effective)):
        aggregates, bins = _encoding_statistics(encoding)
        if not aggregates and not bins:
            continue
        columns = known_columns(layer, chart)
        if columns is None:
            continue
        inputs = {a.field for a in aggregates if a.field} | {b.field for b in bins}
        inputs |= set(_statistic_plan(encoding, chart._facet).groupby)
        missing = inputs - set(columns)
        # A grouping-only color is refused under the spelling it was bound as.
        for channel, value in _expand_grouping(encoding).items():
            field = value.field if isinstance(value, ChannelBase) else value
            if field in missing:
                remedy = "statistic" if field in statistics else "plain"
                raise ValueError(
                    layer_unknown_column_message(
                        i, channel, field, remedy, layer.independent_y, entry=layer.entry
                    )
                )


# ---------------------------------------------------------------------------
# Lowering
# ---------------------------------------------------------------------------


def attach_to_rows(layers: list, chart: Any) -> list:
    """Point each layer attached to *chart* (``_Layer.chart_data``) at its operand's rows.

    On a chart of one operand whose statistics leave the tail where its own
    steps end, the layer reads that tail as it is. Otherwise it reads
    :data:`ROWS_KEY` inside its operand's chain, and takes each x/y/x2/y2 it
    does not bind from the bindings its chart's callers wrote
    (``positional_source="written"``).
    """
    from dataclasses import replace

    if not any(layer.chart_data for layer in layers):
        return layers
    at_tail = not chart._operands and not any(
        _advances_chain(step) for step in chart_operands(chart)[0].statistics
    )
    out = []
    for layer in layers:
        if layer.chart_data and at_tail:
            layer = replace(layer, chart_data=False)
        elif layer.chart_data:
            layer = replace(
                layer,
                chart_data=False,
                data_source=ROWS_KEY,
                reads_chart_data=True,
                positional_source=layer.positional_source or "written",
            )
        out.append(layer)
    return out


def _operand_chain(op: _Operand, layers: list, statistics: list) -> list:
    """*op*'s chain as its *layers* read it: ``[own][composed][rows][marker][statistics]``.

    *statistics* holds each layer's ``_LayerStatistic`` (or ``None``): every
    aggregate, then every bin, after the operand's mark statistics. The rows
    snapshot is published only when a layer, or a layer's statistic, reads
    it.
    """
    computed = [s for s in statistics if s is not None]
    lowered = [
        *op.statistics,
        *(s.step for s in computed if not s.bins),
        *(s.step for s in computed if s.bins),
    ]
    reads_rows = any(layer.data_source == ROWS_KEY for layer in layers) or any(
        isinstance(s.step, _Branch) and s.step.input == ROWS_KEY for s in computed
    )
    rows = [_source_identity(ROWS_KEY)] if reads_rows else []
    return [*op.own, *op.composed, *rows, *_with_statistics_marker(lowered)]


def _chain_key(op: _Operand) -> str:
    """One string per distinct operand chain over one frame, so equal operands share a branch."""
    steps = [*op.own, *op.composed, None, *op.statistics]
    wire = ["|" if step is None else _transforms_to_json_list_named([step])[0] for step in steps]
    return json.dumps([op.frame, wire], sort_keys=True)


def lower_operands(chart: Any, layers: list, statistics: list) -> tuple[list, list]:
    """Lower *chart*'s operands to the pipeline its *layers* read.

    *layers* address keys relative to their operand's chain (``None`` its
    tail), and *statistics* holds each layer's encoding statistic
    (``_resolve_layer_statistics``). A chart of one operand runs its chain
    at the pipeline's top level and its layers keep their keys. Each
    operand of a composition runs as one branch, ``_operand_<k>``, over the
    composition's batch, its frame selecting its own rows when the batch
    holds several frames; each of its layers reads the branch's tail or a key
    scoped inside it (``secondary_key``). Operands over one frame with one
    chain share a branch, so a chart composed with itself computes once.
    An operand no layer reads is not lowered.

    Returns the layers with their final keys and the pipeline's steps.
    """
    from dataclasses import replace

    from ferrum._core import secondary_key

    operands = chart_operands(chart)
    if not chart._operands:
        return layers, _operand_chain(operands[0], layers, statistics)
    selectors = chart._frame_selectors
    branch_of: list[int] = []
    shared: dict[str, int] = {}
    for op in operands:
        branch_of.append(shared.setdefault(_chain_key(op), len(shared)))
    members: dict[int, list[int]] = {}
    for i, layer in enumerate(layers):
        members.setdefault(branch_of[layer.operand], []).append(i)
    out = list(layers)
    transforms: list = []
    for k in sorted(members):
        op = operands[branch_of.index(k)]
        indices = members[k]
        name = f"_operand_{k}"
        frame = selectors[op.frame] if selectors else None
        chain = _operand_chain(op, [layers[i] for i in indices], [statistics[i] for i in indices])
        if not chain and frame is None:
            chain = [dict(_PASS_THROUGH)]
        transforms.append(_Branch(name, None, chain, frame=frame))
        for i in indices:
            key = layers[i].data_source
            out[i] = replace(
                layers[i], data_source=name if key is None else secondary_key(name, key)
            )
    return out, transforms
