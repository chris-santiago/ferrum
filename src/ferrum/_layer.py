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
    # Secondary-y-axis wire flag (GH #52). Two producers set it:
    # LayerChart._build_merged (ferrum.composition), on non-primary layers
    # that carry their own `y` encoding, and chart._desugar_secondary_y,
    # on the layer appended for `chart + SecondaryY(...)`; every other
    # producer of `_Layer` leaves this at its default False (shares the
    # primary y-scale slot).
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
    adjustment, ``data`` carries a synthetic table when the desugar
    materialises its own data (only ``desugar_function`` does so today), and
    ``mark_kwargs`` carries a constant mark-style override the desugar itself
    computes (as opposed to the caller's own style kwargs, which the pending
    dispatcher handles separately via ``_split_style_kwargs``) -- only
    ``desugar_function`` sets it today, to thread ``clip=False`` onto the
    synthesized line layer. ``__post_init__`` rejects ``mark_kwargs`` set
    together with ``layers`` (single-mark-only, mechanically enforced, not
    just documented).

    Both ``mark_kwargs`` and ``data`` are read by every single-mark consumer:
    ``Chart.mark_function`` (``marks/_chart_methods_statistical.py``), which
    calls ``desugar_function`` directly, and ``_desugar._resolve_pending_impl``
    -- the dispatcher every OTHER composite/diagnostic ``mark_*()`` goes
    through via ``Chart._set_composite_mark``. Both merge ``mark_kwargs``
    onto the chart's own inherited ``_mark_kwargs`` (caller's style kwargs
    win over the desugar-computed value, which in turn wins over whatever
    the chart already carried -- e.g. a prior ``.mark_line(stroke_width=7)``
    survives a later dispatcher-routed desugar's own ``mark_kwargs``/style),
    and the dispatcher applies ``data`` as the resolved chart's replacement
    data in the plain branch; the prior-layer (``smooth_layer``) branch
    raises rather than silently dropping ``data``, since a ``_Layer`` has no
    per-layer raw-data slot to attach a synthetic table to without
    corrupting the prior layer's own data (see ``_resolve_pending_impl``'s
    comment).

    **The "caller's style wins over the desugar-computed value" precedence
    is documented but currently inert in production** (mutation-tester
    finding, close-remediation cycle 2): exactly one desugar sets
    ``mark_kwargs`` today (``desugar_function``), and it never reaches the
    dispatcher -- ``Chart.mark_function`` always calls it directly, bypassing
    ``_resolve_pending_impl`` entirely (see that function's own module for
    why: its signature is incompatible with the dispatcher's calling
    convention). So in shipped code the dispatcher's merge branch always
    has ``result.mark_kwargs is None`` on one side; the precedence is
    exercised only by a synthetic test adapter
    (``tests/marks/test_heavy_stats.py``'s
    ``_function_desugar_via_dispatcher``), not by any real caller. It stays
    documented -- and mechanically correct when reached -- for the day a
    second dispatcher-routed desugar sets ``mark_kwargs``.

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
    mark_kwargs : dict or None
        Constant mark-style override computed by the desugar itself.
        Single-mark mode only; ``None`` unless the desugar has one to set.

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

    ``mark_kwargs`` is single-mark-only; setting it alongside ``layers`` is
    rejected too, rather than silently doing nothing:

    >>> MarkDesugarResult(layers=[], mark_kwargs={"clip": False})  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: MarkDesugarResult.mark_kwargs is single-mark-only ...
    """

    mark: Optional[str] = None
    transforms: list = field(default_factory=list)
    remap: dict = field(default_factory=dict)
    position: Any = None
    layers: Optional[list] = None
    data: Any = None  # synthetic data (e.g. desugar_function)
    mark_kwargs: Optional[dict] = None  # desugar-computed style (e.g. desugar_function's clip)

    def __post_init__(self) -> None:
        """Enforce the layered-XOR-single-mark invariant, plus the
        single-mark-only fields.

        Exactly one mode must be populated, discriminated by presence
        (``is not None``), not truthiness: an empty ``layers=[]`` is still a
        valid layered result.  A result that sets both ``mark`` and ``layers``,
        or neither, is contradictory and rejected.

        ``mark_kwargs`` is documented single-mark-only (layered mode has its
        own per-layer ``_Layer.mark_kwargs``); setting both ``layers`` and
        ``mark_kwargs`` is rejected here rather than left to silently do
        nothing, since layered mode ignores the top-level field entirely.
        """
        if (self.layers is not None) == (self.mark is not None):
            raise ValueError(
                "MarkDesugarResult must be either layered (layers set, mark "
                "None) or single-mark (mark set, layers None), not both or "
                f"neither; got mark={self.mark!r}, "
                f"layers={'set' if self.layers is not None else 'None'}."
            )
        if self.layers is not None and self.mark_kwargs is not None:
            raise ValueError(
                "MarkDesugarResult.mark_kwargs is single-mark-only; a layered "
                "result (layers set) must put per-layer style on each "
                "_Layer.mark_kwargs instead, not on the top-level field, "
                f"got mark_kwargs={self.mark_kwargs!r}."
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
