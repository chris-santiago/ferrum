"""Internal layer wrapper used by composite-mark expansion."""

from __future__ import annotations

from ferrum.marks.base import MARK_HONORED_KWARGS, check_constant_size

_LAYER_TRANSFORMS_REFUSAL = (
    "Layer(transforms=[...]): a layer does not run its own transforms. Give the layer "
    "its own chart with .transform(...) and compose it with +, which routes the "
    "transform to that layer alone."
)


class Layer:
    """Single rendering layer inside a multi-layer ``Chart``.

    Most multi-layer charts are built via the ``+`` operator on two ``Chart``
    instances (``chart1 + chart2``).  Direct construction of ``Layer`` objects
    is rarely needed outside of ferrum internals and composite-mark expansion
    code.

    Parameters
    ----------
    data : object, optional
        DataFrame or Arrow-compatible object for this layer.  When ``None``
        the layer inherits data from the parent ``ChartSpec``.
    mark : str or MarkSpec, optional
        Mark type string (e.g. ``"point"``, ``"line"``) or a ``MarkSpec``
        object.
    name : str, optional
        Identifier for this layer, used by composite-mark expansion and
        layer-targeted mark overrides. ``None`` for unnamed layers.
    encoding : dict, optional
        Mapping of channel names to ``ChannelBase`` objects.  Defaults to an
        empty dict when omitted.  A channel's own ``condition=`` kwarg (e.g.
        ``fm.Color(field, condition=sel.when(...).otherwise(...))``) is
        honored: ``Chart.layer()`` lifts it into the chart's real
        conditional pipeline, gated by the same vocabulary ``encode()``
        uses -- a conditional on a channel the wire cannot carry
        (``stroke``, ``shape``, ``x``, ``y``) raises the same typed
        ``ValueError`` (2026-09-11, batch-H close cycle 3).
    transforms : list, optional
        Must be omitted, ``None``, ``[]`` or ``()``. A layer does not run
        its own transforms: it reads either the chart's final batch or a
        *named* chart-level transform's output through its data source. Any
        other value raises ``ValueError``. To transform one layer's data, give
        that layer its own chart with ``.transform(...)`` and compose it with
        ``+``, which routes the transform to that layer alone.
    mark_kwargs : dict, optional
        Extra keyword arguments forwarded to the mark renderer (e.g.
        ``{"dx": 5, "opacity": 0.7}``). A constant ``size`` on a primitive
        ``mark`` meets the same rule as ``mark_<mark>(size=...)``: it raises
        ``ValueError`` here, voiced as ``mark_<mark>``, when the mark never
        draws a constant size or the value is not a finite number above zero.

    Examples
    --------
    >>> from ferrum.layer import Layer
    >>> layer = Layer(mark="text", encoding={"x": "x", "y": "y"}, mark_kwargs={"dx": 5})
    """

    def __init__(
        self, data=None, mark=None, *, name=None, encoding=None, transforms=None, mark_kwargs=None
    ):
        # Only "no transforms" is accepted; any other value (a non-empty
        # sequence, a falsy non-sequence, an array whose truth is ambiguous)
        # gets the refusal rather than being dropped or tripping a raw error.
        if transforms is not None and not (
            isinstance(transforms, (list, tuple)) and len(transforms) == 0
        ):
            raise ValueError(_LAYER_TRANSFORMS_REFUSAL)
        self.data = data
        self.mark = mark
        self.name = name
        self.encoding = encoding or {}
        self.transforms = []
        self.mark_kwargs = dict(mark_kwargs or {})
        size = self.mark_kwargs.get("size")
        if size is not None and isinstance(mark, str) and mark in MARK_HONORED_KWARGS:
            self.mark_kwargs["size"] = check_constant_size(f"mark_{mark}", mark, size)

    def __repr__(self) -> str:
        """Return a short developer-readable description."""
        return f"Layer(mark={self.mark!r})"
