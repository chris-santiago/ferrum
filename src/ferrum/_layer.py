"""Internal _Layer value type — frozen dataclass shared between Chart and the
mark-desugar modules (composite, heavy_stat, statistical, diagnostic).

The wire format emitted to Rust by ``Chart._build_layers_list`` still uses
``mark_style`` for backward compat with ``coerce_layers``; ``_Layer``
canonicalises on ``mark_kwargs``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# Primitive (renderer-native) mark names — those that do not desugar into a
# composite of other marks.  Single-sourced here (a stdlib-only leaf imported by
# the desugar pipeline) so chart.py, _desugar.py, and the statistical
# chart-method mixin cannot drift across the T4.1 split seam.
_PRIMITIVE_MARKS = frozenset(["point", "line", "bar", "area", "rule", "text", "tick", "rect"])


@dataclass(frozen=True)
class _Layer:
    """Internal layer descriptor consumed by ``Chart._build_layers_list``."""

    name: Optional[str] = None
    mark: Optional[str] = None
    encoding: dict = field(default_factory=dict)
    transforms: list = field(default_factory=list)
    mark_kwargs: Optional[dict] = None
    data_source: Optional[str] = None
    position: Any = None
    blend: Optional[str] = None
    # Secondary-y-axis wire flag (GH #52). Set only by
    # LayerChart._build_merged (ferrum.composition) on non-primary layers
    # that carry their own `y` encoding; every other producer of `_Layer`
    # leaves this at its default False (shares the primary y-scale slot).
    # Chart._build_layers_list serializes it onto the wire layer dict only
    # when True, so absent/false round-trips byte-identically with
    # pre-#52 specs.
    independent_y: bool = False


@dataclass(frozen=True)
class MarkDesugarResult:
    """Typed return from a desugar function consumed by ``_resolve_pending``.

    Replaces the legacy tuple protocol that used 3-tuple, 4-tuple, and 5-tuple
    shapes to signal different modes.  The mode is selected by whether
    ``layers`` is set, not by the result's arity.

    Modes
    -----
    **Layered** (``layers`` is not ``None``): a multi-layer mark such as a
    boxplot, violin, or CI smooth.  ``layers`` is a list of ``_Layer``
    descriptors and ``transforms`` apply at the chart level.  ``mark``,
    ``remap``, ``position``, and ``data`` are ignored in this mode.

    **Single-mark** (``layers`` is ``None``): one primitive mark.  ``mark``
    names it (e.g. ``"bar"``, ``"line"``), ``remap`` rewrites the chart's
    encoding channels onto the transform output columns, ``transforms`` is
    the per-mark transform chain, ``position`` is an optional position
    adjustment, and ``data`` carries a synthetic table when the desugar
    materialises its own data (only ``desugar_function`` does so today).

    Attributes
    ----------
    mark : str or None
        Primitive mark name in single-mark mode.  Ignored in layered mode.
    transforms : list
        Transform chain.  Applied at the chart level in layered mode; the
        per-mark chain in single-mark mode.
    remap : dict
        Encoding-channel remap (e.g. ``{"x": "value", "y": "density"}``).
        Single-mark mode only.
    position : Any
        Optional position adjustment (e.g. ``Stack``, ``Dodge``).
        Single-mark mode only.
    layers : list or None
        ``_Layer`` descriptors.  Non-``None`` selects layered mode.
    data : Any
        Synthetic data table.  Single-mark mode only; ``None`` unless the
        desugar materialises its own data.

    Examples
    --------
    Single-mark mode (a histogram bar):

    >>> single = MarkDesugarResult(
    ...     mark="bar", remap={"x": "bin_start", "x2": "bin_end", "y": "count"}
    ... )
    >>> single.mark
    'bar'
    >>> single.layers is None
    True

    Layered mode (two stacked layers):

    >>> layered = MarkDesugarResult(
    ...     layers=[_Layer(mark="ribbon"), _Layer(mark="line")]
    ... )
    >>> [layer.mark for layer in layered.layers]
    ['ribbon', 'line']
    >>> layered.mark is None
    True

    An empty layered result is still layered mode (``layers`` present, even if
    no layers are built for this configuration):

    >>> MarkDesugarResult(layers=[]).layers
    []

    The two modes are mutually exclusive.  Setting both ``mark`` and ``layers``
    (or neither) is rejected at construction:

    >>> MarkDesugarResult(mark="bar", layers=[])  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: MarkDesugarResult must be either layered ...
    >>> MarkDesugarResult()  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: MarkDesugarResult must be either layered ...
    """

    mark: Optional[str] = None
    transforms: list = field(default_factory=list)
    remap: dict = field(default_factory=dict)
    position: Any = None
    layers: Optional[list] = None
    data: Any = None  # synthetic data (e.g. desugar_function)

    def __post_init__(self) -> None:
        """Enforce the layered-XOR-single-mark invariant.

        Exactly one mode must be populated, discriminated by presence
        (``is not None``), not truthiness: an empty ``layers=[]`` is still a
        valid layered result.  A result that sets both ``mark`` and ``layers``,
        or neither, is contradictory and rejected.
        """
        if (self.layers is not None) == (self.mark is not None):
            raise ValueError(
                "MarkDesugarResult must be either layered (layers set, mark "
                "None) or single-mark (mark set, layers None), not both or "
                f"neither; got mark={self.mark!r}, "
                f"layers={'set' if self.layers is not None else 'None'}."
            )


@dataclass(frozen=True)
class _PendingMark:
    """Sentinel stored on ``Chart._pending_stat_mark`` when a composite or
    diagnostic ``mark_*()`` is called before ``.encode()``.

    ``Chart._resolve_pending`` calls ``desugar_fn(x_field, y_field, **kwargs)``
    once the encoding is known and threads the result back into the chart's
    transforms / layers / encoding.
    """

    kind: str
    kwargs: dict
    desugar_fn: Any  # Callable[[Optional[str], Optional[str], **Any], MarkDesugarResult]
    prior_mark: str | None = None  # Existing primitive mark to preserve as a layer
