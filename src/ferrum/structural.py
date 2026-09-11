"""Structural feature dataclasses for advanced chart layout options."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ferrum._validate import validate_choice, validate_fraction_value, validate_pixel_value

_VALID_BREAK_STYLES = frozenset({"slash", "zigzag", "wave", "gap"})
_VALID_CONNECT_STYLES = frozenset({"bracket", "lines", "none"})
_INSET_BOUNDS_NAMES = ("left", "top", "right", "bottom")


@dataclass(frozen=True)
class SecondaryY:
    """A secondary y-axis encoding overlaid on a chart.

    ``chart + SecondaryY(...)`` desugars to an appended layer on *chart*: mark
    ``mark``, ``y`` encoding on ``field`` (carrying ``axis``/``scale``), ``x``
    inherited from the base chart, color literal ``color``, opacity
    ``opacity`` — flagged as an independent-y layer (GH #52). The base
    chart's own layer(s) are unchanged, so ``layered_chart + SecondaryY(...)``
    keeps the base layers sharing the left axis while only the appended
    layer gets its own right axis; adding multiple ``SecondaryY`` instances
    stacks multiple right axes outward. The base chart must carry an ``x``
    encoding for the secondary layer to inherit — adding ``SecondaryY`` to
    a chart with no ``x`` raises ``ValueError``. ``field`` is read from the
    base chart's own table (the desugar performs no data merge), so a
    ``field`` that is not a column of the base data also raises
    ``ValueError`` at ``+`` time. ``mark`` must name a primitive mark
    (``point``, ``line``, ``bar``, ``area``, ``rule``, ``text``, ``tick``,
    ``rect``) -- a composite mark name (e.g. ``"boxplot"``) would otherwise
    bypass the ``mark_*()`` desugar pipeline and reach the renderer as an
    unknown primitive, so it raises ``ValueError`` at ``+`` time; use
    ``LayerChart(chart, other_chart, resolve={"y": "independent"})`` for a
    composite overlay on a secondary axis instead.

    This re-bases the feature onto ferrum's per-layer independent-y
    subsystem: unlike the original overlay-only renderer, the secondary
    series now reserves its own right-side margin band (so the plot area
    narrows to make room for it, rather than the axis overdrawing the plot),
    gets a real axis (ticks, labels, per-encoding ``Axis(...)`` config), and
    is fully interactive (tooltips, zoom/pan, hit-testing) like any other
    layer.

    Parameters
    ----------
    field : str
        Data field to encode on the secondary y axis.
    mark : str, default "line"
        Mark type for the secondary series.
    axis : Axis, optional
        Per-axis configuration for the secondary y axis.
    color : str, optional
        Color for the secondary mark.
    opacity : float, optional
        Opacity for the secondary mark.
    scale : Scale, optional
        Scale configuration for the secondary y axis.
    """

    field: str
    mark: str = "line"
    axis: Any = None  # Axis instance or None
    color: str | None = None
    opacity: float | None = None
    scale: Any = None  # Scale instance or None


@dataclass(frozen=True)
class BreakAxis:
    """An axis break that omits a region of the scale to skip outlier values.

    Parameters
    ----------
    axis : str
        Which axis to break: ``"x"`` or ``"y"``.
    gap : tuple or list
        A single ``(start, end)`` break region or a list of ``(start, end)``
        tuples for multiple breaks.
    break_size : float, default 12
        Visual size of the break indicator in pixels.
    break_style : str, default "slash"
        Break indicator style: ``"slash"``, ``"zigzag"``, ``"wave"``, or ``"gap"``.
    """

    axis: str
    gap: tuple | list
    break_size: float = 12
    break_style: str = "slash"

    def __post_init__(self) -> None:
        validate_choice("BreakAxis.axis", "axis", self.axis, ("x", "y"))
        validate_choice(
            "BreakAxis.break_style", "break_style", self.break_style, _VALID_BREAK_STYLES
        )


def _inset_bound_kind_and_value(entry: Any) -> tuple[str, float]:
    """Classify one ``Inset.bounds`` entry as ``("norm", value)`` or ``("px", value)``.

    Mirrors the wire contract emitted by ``ferrum._render._inset_coord``:
    [NormCoord][ferrum.NormCoord] and bare numbers are normalized-fraction
    bounds (a bare number is *never* data-space, per the 2026-05-24
    inset-panels design); [PixelCoord][ferrum.PixelCoord] is a pixel-offset
    bound. See the batch-H design spec §4 H6 / §6.
    """
    from ferrum.annotation.coords import NormCoord, PixelCoord

    if isinstance(entry, NormCoord):
        return "norm", entry.value
    if isinstance(entry, PixelCoord):
        return "px", entry.value
    return "norm", float(entry)


def _validate_inset_bounds(bounds: tuple) -> None:
    """Raise ``ValueError`` for an out-of-range, negative-px, or empty inset box.

    Each of the four ``bounds`` entries (``left``, ``top``, ``right``,
    ``bottom``) is classified via :func:`_inset_bound_kind_and_value`, then
    checked against its own kind's contract: a normalized entry must be a
    finite fraction in ``[0, 1]`` (:func:`ferrum._validate.validate_fraction_value`);
    a pixel entry must be a finite, non-negative pixel value
    (:func:`ferrum._validate.validate_pixel_value`). The box is additionally
    refused as empty when a coordinate pair is of the *same* kind and has
    zero or negative extent (``left >= right`` or ``top >= bottom``) — a
    mixed-kind pair (one normalized, one pixel) is not directly comparable
    on a shared scale, so it is not checked for emptiness (spec §4 H6:
    "compared within like kinds").
    """
    if len(bounds) != 4:
        raise ValueError(
            "Inset.bounds must have exactly 4 entries (left, top, right, bottom); "
            f"got {len(bounds)}"
        )
    kinds: list[str] = []
    values: list[float] = []
    for i, entry in enumerate(bounds):
        kind, value = _inset_bound_kind_and_value(entry)
        label = f"Inset.bounds[{i}] ({_INSET_BOUNDS_NAMES[i]})"
        if kind == "norm":
            validate_fraction_value(label, value)
        else:
            validate_pixel_value(label, value)
        kinds.append(kind)
        values.append(value)

    left, top, right, bottom = values
    if kinds[0] == kinds[2] and left >= right:
        raise ValueError(
            f"Inset.bounds: empty box — left ({left!r}) must be less than "
            f"right ({right!r}); got bounds={bounds!r}"
        )
    if kinds[1] == kinds[3] and top >= bottom:
        raise ValueError(
            f"Inset.bounds: empty box — top ({top!r}) must be less than "
            f"bottom ({bottom!r}); got bounds={bounds!r}"
        )


@dataclass(frozen=True)
class Inset:
    """An inset chart embedded within a parent chart's plot area.

    Parameters
    ----------
    chart : Chart
        The chart to embed as an inset.
    bounds : tuple
        ``(left, top, right, bottom)`` boundary coordinates for the inset
        within the parent plot area.  Each coordinate may be a bare
        ``float`` (a normalized ``[0, 1]`` fraction of the plot area — the
        same as [NormCoord][ferrum.NormCoord]), [NormCoord][ferrum.NormCoord]
        itself, or [PixelCoord][ferrum.PixelCoord] (an absolute pixel offset
        from the plot area's origin, resolved in Rust against the computed
        plot rect at layout time). Mixed kinds within one ``bounds`` tuple
        are allowed; each entry resolves independently. A normalized entry
        outside ``[0, 1]``, a negative pixel entry, or an empty box (``left
        >= right`` or ``top >= bottom``, compared only within like kinds)
        raises ``ValueError`` at construction time.

        .. note::
           **Dated note (2026-09-11, batch H, F-L08-08):** earlier revisions
           of this docstring described a bare ``float`` as a *data-space*
           coordinate and stated that [PixelCoord][ferrum.PixelCoord] was
           unsupported for inset bounds. Neither was ever true of the
           design this feature shipped from (the 2026-05-24 inset-panels
           design fixed bounds as normalized-or-pixel, never data-space) —
           the docstring claim was same-commit authorship drift, and
           passing ``fm.px(...)`` previously raised ``TypeError`` as a
           stopgap rather than being resolved. Both are corrected here:
           bare floats and [NormCoord][ferrum.NormCoord] are normalized
           fractions, and [PixelCoord][ferrum.PixelCoord] now resolves
           correctly.
    border : bool, default True
        Draw a border around the inset.
    border_color : str, default "#999"
        Border color.
    border_dash : list[float], optional
        Border dash pattern.
    background : str or None, default "#fff"
        Inset background color; ``None`` for transparent.
    shadow : bool, default False
        Apply a drop shadow to the inset.
    connect_to : tuple, optional
        Data coordinates ``(x, y)`` of the source region in the parent chart
        that this inset zooms into.  Draws a connector from the parent region
        to the inset when provided.
    connect_style : str, default "lines"
        Connector style: ``"bracket"``, ``"lines"``, or ``"none"``.
    """

    chart: Any  # Chart instance
    bounds: tuple
    border: bool = True
    border_color: str = "#999"
    border_dash: list[float] | None = None
    background: str | None = "#fff"
    shadow: bool = False
    connect_to: tuple | None = None
    connect_style: str = "lines"

    def __post_init__(self) -> None:
        validate_choice(
            "Inset.connect_style", "connect_style", self.connect_style, _VALID_CONNECT_STYLES
        )
        _validate_inset_bounds(self.bounds)
