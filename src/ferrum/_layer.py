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
    #: Keys of ``mark_kwargs`` that came from a caller parameter aimed at THIS
    #: layer specifically -- ``mark_smooth(line_kwargs=...)`` / ``lmplot(
    #: line_kws=...)`` on the fit line today (#199). The chart-level catch-all
    #: ``**mark_kwargs``, which the pending dispatcher spreads across every
    #: emitted layer, does not override them: a value the caller aimed at one
    #: layer is more specific than one they aimed at the mark as a whole.
    #: Read by ``_desugar._apply_style_to_layer`` and by nothing else, and
    #: never serialized (``_spec_build._build_layers_list`` reads named
    #: fields). A layer that never passes through that function -- one
    #: assembled outside the pending dispatcher, e.g.
    #: ``plots/regression._fit_chart_from_named_transform`` -- is therefore
    #: unaffected by this field either way; it is NOT a claim that no other
    #: merger exists (``marks/_mark_kwargs.apply_user_mark_kwargs`` is one,
    #: and ignores this field -- see ``_desugar._catch_all_over``). Carries
    #: KEYS, not values: the values live in ``mark_kwargs`` with every other
    #: style, so nothing has to be folded in later and no layer can lose its
    #: styling by skipping a fold step.
    caller_style_keys: frozenset = frozenset()
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
    ``mark_kwargs`` carries the style this desugar resolved for the mark it
    emits, whatever its source: a constant the desugar computes for itself
    (``desugar_function``'s ``clip=False``) or a value the caller aimed at
    that mark through one of the desugar's own named parameters
    (``desugar_smooth``'s ``line_kwargs=``, i.e. ``lmplot``'s ``line_kws``,
    #199). What it is NOT is the caller's chart-level catch-all
    ``**mark_kwargs`` -- those are partitioned out by ``_split_style_kwargs``
    and applied by the pending dispatcher, which is a different slot with a
    different reach (every emitted layer, not this one mark).
    ``caller_style_keys`` names the subset of these keys that came from the
    caller rather than from the desugar, and is what decides precedence
    against that catch-all; see below. ``__post_init__`` rejects
    ``mark_kwargs`` set together with ``layers`` (single-mark-only,
    mechanically enforced, not just documented).

    Both ``mark_kwargs`` and ``data`` are read by every single-mark consumer:
    ``Chart.mark_function`` (``marks/_chart_methods_statistical.py``), which
    calls ``desugar_function`` directly, and ``_desugar._resolve_pending_impl``
    -- the dispatcher every OTHER composite/diagnostic ``mark_*()`` goes
    through via ``Chart._set_composite_mark``. Both merge ``mark_kwargs``
    onto the chart's own inherited ``_mark_kwargs`` rather than replacing it,
    so a key the chart already carried and this desugar does not name
    survives, and the dispatcher applies ``data`` as the resolved chart's replacement
    data in the plain branch; the prior-layer (``smooth_layer``) branch
    raises rather than silently dropping ``data``, since a ``_Layer`` has no
    per-layer raw-data slot to attach a synthetic table to without
    corrupting the prior layer's own data (see ``_resolve_pending_impl``'s
    comment).

    **Precedence against the chart-level catch-all is decided by
    ``caller_style_keys``, not by prose.** When both slots name the same
    style key, the more specific instruction wins, and specificity is read
    off the slot rather than guessed at:

    - a key NOT in ``caller_style_keys`` is the desugar's own computed
      default, so the caller's catch-all ``**mark_kwargs`` overrides it --
      this is what lets ``mark_boxplot(opacity=0.8)`` beat a ribbon's
      built-in ``opacity=0.3``;
    - a key IN ``caller_style_keys`` is something the caller aimed at this
      one mark through a named parameter, so the catch-all does NOT override
      it -- ``mark_smooth(stroke="#0000ff", line_kwargs={"stroke":
      "#ff0000"})`` draws the fit line ``#ff0000`` and the CI band
      ``#0000ff``, since only the line was named.

    Both merge sites read the same field, so the two shapes cannot drift:
    ``_apply_style_to_layer`` for a layered result (per ``_Layer``) and
    ``_merge_desugar_mark_kwargs`` for a single-mark one. The ordering is
    therefore a property of where a value came from, which the code records,
    rather than of which desugar happened to set it.

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
        The style this desugar resolved for the mark it emits -- its own
        computed constants and any value the caller aimed at that mark
        through one of its named parameters. Single-mark mode only; ``None``
        unless the desugar has one to set.
    caller_style_keys : frozenset of str
        Which ``mark_kwargs`` keys came from the caller rather than from the
        desugar. Decides precedence against the chart-level catch-all (see
        above); empty means every key is a desugar default. Single-mark mode
        only -- a layered result records the same thing per layer, on
        ``_Layer.caller_style_keys``.

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
    mark_kwargs: Optional[dict] = None  # style resolved for the emitted mark
    caller_style_keys: frozenset = frozenset()  # which of those keys the caller named

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
        ``caller_style_keys`` rides the same rule for the same reason -- it
        annotates ``mark_kwargs``, so it is meaningless without one, and a
        layered result records it per layer instead.
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
        if self.layers is not None and self.caller_style_keys:
            raise ValueError(
                "MarkDesugarResult.caller_style_keys is single-mark-only; it "
                "annotates the top-level mark_kwargs, so a layered result "
                "(layers set) must record it per layer on "
                "_Layer.caller_style_keys instead, got "
                f"caller_style_keys={set(self.caller_style_keys)!r}."
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
