"""Internal layer wrapper used by composite-mark expansion."""
from __future__ import annotations


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
    encoding : dict, optional
        Mapping of channel names to ``ChannelBase`` objects.  Defaults to an
        empty dict when omitted.
    transforms : list, optional
        Sequence of transform objects applied to this layer's data before
        rendering.  Defaults to an empty list when omitted.

    Examples
    --------
    >>> from ferrum.layer import Layer
    >>> layer = Layer(mark="point", encoding={"x": x_ch, "y": y_ch})
    """

    def __init__(self, data=None, mark=None, *, encoding=None, transforms=None):
        self.data = data
        self.mark = mark
        self.encoding = encoding or {}
        self.transforms = transforms or []

    def __repr__(self) -> str:
        """Return a short developer-readable description."""
        return f"Layer(mark={self.mark!r})"
