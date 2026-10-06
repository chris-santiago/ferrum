"""Internal layer wrapper used by composite-mark expansion."""

from __future__ import annotations

from ferrum._layer import layer_mark
from ferrum.marks.base import resolve_layer_style

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
    mark : str, optional
        Mark name (e.g. ``"point"``, ``"line"``). ``None`` draws a point.
        Any other non-string value raises ``ValueError``.
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
        Style keyword arguments for the layer's mark (e.g.
        ``{"dx": 5, "opacity": 0.7}``), held to the fields Rust's style
        table says the mark draws: aliases (``color``, ``alpha``) resolve,
        and a field the mark never draws, a constant ``size`` the mark never
        draws or that is not a finite number above zero, or a malformed
        value raises ``ValueError`` here, voiced as ``mark_<mark>``. So does
        a kwarg only the mark's own method takes (``orient`` on a bar), and
        a field only ferrum's own code writes (``detail``, ``clip``,
        ``value_axis``), each naming the call that sets it.

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
        # The wire carries a mark by name only, so anything else is refused
        # here rather than at serialization.
        if mark is not None and not isinstance(mark, str):
            raise ValueError(
                f"Layer(mark=...): a layer's mark is a mark name such as 'point' or "
                f"'line'; got {mark!r}"
            )
        self.data = data
        self.mark = mark
        self.name = name
        self.encoding = encoding or {}
        self.transforms = []
        self.mark_kwargs = resolve_layer_style(f"mark_{layer_mark(mark)}", mark, mark_kwargs or {})

    def __repr__(self) -> str:
        """Return a short developer-readable description."""
        return f"Layer(mark={self.mark!r})"
