"""Internal _Layer value type — frozen dataclass shared between Chart and the
mark-desugar modules (composite, heavy_stat, statistical, diagnostic).

The wire format emitted to Rust by ``Chart._build_layers_list`` still uses
``mark_style`` for backward compat with ``coerce_layers``; ``_Layer``
canonicalises on ``mark_kwargs``.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Optional

from ferrum._core import mark_style_fields, marks_drawing_constant_size, style_table_marks

# The marks whose ``mark_*`` method sets a flat mark directly, with no
# desugar into a layered composite. A narrower set than the marks with a
# row in Rust's style table (``style_table_marks()``), which also holds the
# primitives only a composite emits (ribbon, segment, arc, ...). Kept here,
# a leaf of the desugar pipeline, so chart.py, _desugar.py and the
# statistical chart-method mixin read one set.
_FLAT_MARK_METHODS = frozenset(["point", "line", "bar", "area", "rule", "text", "tick", "rect"])


@dataclass(frozen=True)
class _Layer:
    """Internal layer descriptor consumed by ``Chart._build_layers_list``."""

    name: Optional[str] = None
    mark: Optional[str] = None
    encoding: dict = field(default_factory=dict)
    mark_kwargs: Optional[dict] = None
    #: Keys of ``mark_kwargs`` that came from a caller instruction aimed at
    #: THIS layer specifically -- ``mark_smooth(layer_kwargs={"line": ...})``
    #: and ``lmplot(line_kws=...)`` on the fit line today (#199). The chart-level catch-all
    #: ``**mark_kwargs``, which the pending dispatcher spreads across every
    #: emitted layer, does not override them: a value the caller aimed at one
    #: layer is more specific than one they aimed at the mark as a whole.
    #: Read by :func:`apply_catch_all_style` below -- the ONE merger of a
    #: catch-all style over a layer, which both the pending dispatcher
    #: (``_desugar._resolve_pending_impl``) and the desugar-side spreader
    #: (``marks/_mark_kwargs.apply_user_mark_kwargs``) call -- and never
    #: serialized (``_spec_build._build_layers_list`` reads named fields). A
    #: layer that never passes through either -- one assembled outside both,
    #: e.g. ``plots/regression._fit_chart_from_named_transform`` -- is
    #: unaffected by this field either way. Carries
    #: KEYS, not values: the values live in ``mark_kwargs`` with every other
    #: style, so nothing has to be folded in later and no layer can lose its
    #: styling by skipping a fold step.
    caller_style_keys: frozenset = frozenset()
    data_source: Optional[str] = None
    position: Any = None
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
    #: This layer's OWN coordinate system (`#172`(1)) -- a coord value class
    #: (``CoordFlip``/``CoordPolar``/...) or the ``"flip"`` string
    #: ``mark_*(orient="horizontal")`` stamps on a chart. ``None`` means
    #: "inherit the chart's", which is every layer any other producer builds.
    #:
    #: The one producer is the ``Chart + Chart`` merge
    #: (:func:`ferrum.composition._carry_layer_coord`), which is where a member
    #: chart's coord used to be dropped: the merge keeps the LHS's coord at
    #: chart level, so the RHS's had nowhere to go and the merged render placed
    #: that member's marks in a coordinate system the caller never asked for --
    #: silently, for a ``CoordFlip``. ``LayerChart`` reaches it through the same
    #: merge (``_build_merged``), so the interactive and independent-y routes
    #: are covered by the one stamp.
    #:
    #: ``_spec_build._build_layers_list`` serializes it onto the wire layer dict
    #: only when set, so a layer that declares none round-trips byte-identically
    #: with a pre-`#172` spec (see ``Layer::coord`` in Rust ``spec/layer.rs``).
    coord: Any = None
    #: The public entry the caller invoked to produce this layer
    #: (``"mark_smooth"`` for every layer ``mark_smooth(...)`` emits,
    #: ``"lmplot"`` for every layer of an ``lmplot`` chart), so a render-time
    #: refusal about this layer names the call the caller wrote rather than
    #: the primitive the layer draws. ``None`` for a layer the caller declared
    #: as that primitive itself. Serialized onto the wire
    #: layer dict only when set (``Layer::entry`` in Rust ``spec/layer.rs``).
    entry: Optional[str] = None
    #: The ``ConditionalSpec`` objects this layer declared, held by identity.
    #: A conditional is owned by a layer when it was declared on that layer's
    #: own ``Layer(encoding=...)``, or on the chart the layer came from when
    #: that chart became one operand of ``chart + chart`` (see
    #: :func:`ferrum.composition._bind_operand_conditionals`). A conditional
    #: no layer owns was declared on the chart as a whole and reaches every
    #: layer. ``Chart.to_spec()`` turns ownership into wire layer indices once,
    #: against the final layer list (:func:`ferrum.composition._conditional_layer_scopes`).
    #: Never serialized and never compared.
    conditionals: tuple = field(default=(), compare=False, repr=False)


def owned_conditional_ids(layers: list) -> set[int]:
    """The identities of the conditionals some layer of *layers* owns (``_Layer.conditionals``)."""
    return {id(c) for layer in layers for c in layer.conditionals}


def drop_orphaned_conditionals(conditionals: list, kept: list, dropped: list) -> list:
    """Return *conditionals* without the ones only the *dropped* layers owned.

    A conditional owned by layers that are all gone must go with them: left
    behind, it would be owned by no layer, which reads as declared on the
    chart as a whole, and would restyle every remaining layer. A conditional
    a *kept* layer also owns keeps its place, and one no layer ever owned
    (declared on the chart as a whole) is untouched.
    """
    orphaned = owned_conditional_ids(dropped) - owned_conditional_ids(kept)
    return [c for c in conditionals if id(c) not in orphaned]


def is_unset(value: Any) -> bool:
    """Whether a style *value* leaves its key unset: ``None`` is unset at every style door."""
    return value is None


def set_style(style: Mapping[str, Any]) -> dict[str, Any]:
    """The keys of *style* the caller actually set, without the unset ones (:func:`is_unset`)."""
    return {key: value for key, value in style.items() if not is_unset(value)}


def merge_style(existing: Mapping[str, Any] | None, style: Mapping[str, Any]) -> dict[str, Any]:
    """Merge a caller's *style* over an *existing* style and return the result as a new dict.

    A key *style* sets wins. A key *style* leaves unset (:func:`is_unset`)
    sets nothing, so it never erases a value *existing* already holds. Every
    door that merges a caller's style over a style already set merges through
    this.
    """
    return {**(existing or {}), **set_style(style)}


def catch_all_over(resolved: dict | None, style_dict: dict, caller_style_keys: frozenset) -> dict:
    """Merge a chart-level catch-all *style_dict* over an already-*resolved* style.

    The one precedence rule for that question, homed here beside the
    ``caller_style_keys`` field it reads:

    - a resolved key the caller did NOT name is the desugar's own default, so
      the catch-all overrides it (``mark_boxplot(opacity=0.8)`` beats a
      ribbon's built-in ``opacity=0.3``);
    - a resolved key the caller DID name, through one of the mark's own
      per-layer parameters, is the more specific instruction, so the catch-all
      leaves it alone (``mark_smooth(layer_kwargs={"line": {"stroke": ...}})``
      beats a bare ``stroke=`` on the fit line, while the band still takes the
      bare one);
    - a ``None`` value in the catch-all is unset (:func:`merge_style`), so
      it never overrides a resolved default (``mark_smooth(opacity=None)``
      still draws the ribbon at its built-in ``0.3``, not fully opaque).

    Both callers -- :func:`apply_catch_all_style` for a ``_Layer`` and
    ``_desugar._merge_desugar_mark_kwargs`` for a single-mark result -- read
    the rule from here rather than restating it, so the two shapes of a
    desugar result cannot answer it differently.
    """
    return merge_style(
        resolved, {k: v for k, v in style_dict.items() if k not in caller_style_keys}
    )


_MARKS_DRAWING_CONSTANT_SIZE = frozenset(marks_drawing_constant_size())
_STYLE_TABLE_MARKS = frozenset(style_table_marks())


def layer_mark(mark: str | None) -> str:
    """The mark a layer draws: its own, or a point when it names none, as on the wire."""
    return mark or "point"


@functools.cache
def drawn_style_fields(mark: str) -> frozenset[str] | None:
    """The ``mark_style`` fields *mark* draws, from Rust's one style table.

    ``None`` when *mark* has no row in the table (a composite name such as
    ``"boxplot"``): only the primitive marks a layer can draw have a row.
    """
    if mark not in _STYLE_TABLE_MARKS:
        return None
    return frozenset(mark_style_fields(mark))


def layer_draws(mark: str | None, field_name: str) -> bool:
    """Whether a layer drawing *mark* draws the style field *field_name*.

    A layer with no mark draws a point (:func:`layer_mark`). A composite
    name has no row in the table, so the field is left for the renderer to
    judge.
    """
    drawn = drawn_style_fields(layer_mark(mark))
    return drawn is None or field_name in drawn


@functools.cache
def marks_drawing(field_name: str) -> tuple[str, ...]:
    """The primitive marks, in :func:`ferrum._core.style_table_marks` order,
    whose row draws *field_name*.

    Read by a caller that names which marks draw a field in a warning or
    error -- e.g. ``encoding/_aliases.py``'s "ignored" warning for
    ``detail`` -- instead of hand-listing the marks as a Python literal.
    """
    return tuple(mark for mark in style_table_marks() if layer_draws(mark, field_name))


def apply_catch_all_style(layer: _Layer, style_dict: dict) -> _Layer:
    """Merge a catch-all *style_dict* into *layer*'s ``mark_kwargs``.

    Precedence is :func:`catch_all_over`'s, keyed on the layer's own
    ``caller_style_keys``.  When *style_dict* is empty the layer is returned
    unchanged, so output stays byte-identical to a render that passed no
    style at all.

    This is the package's ONE catch-all-over-a-layer merger. Both spreaders
    call it: ``_desugar._resolve_pending_impl``, which spreads the chart-level
    ``**mark_kwargs`` over a layered desugar's emitted layers, and
    ``marks/_mark_kwargs.apply_user_mark_kwargs``, which spreads a desugar's
    own validated ``**mark_kwargs`` over the layers it just built.

    A catch-all field reaches only a layer whose mark draws it
    (:func:`layer_draws`): the caller aimed the style at the marks that draw
    it, not at, say, a smooth's confidence band, and render refuses a field
    on a mark that never draws it. Each spreader refuses a field that no
    layer it emitted draws (``ferrum.marks.base.refuse_style_no_layer_draws``).
    """
    style_dict = {k: v for k, v in style_dict.items() if layer_draws(layer.mark, k)}
    if not style_dict:
        return layer
    merged = catch_all_over(layer.mark_kwargs, style_dict, layer.caller_style_keys)
    return replace(layer, mark_kwargs=merged)


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
    the catalogued layer this mark IS, through the per-layer door
    (``mark_smooth(layer_kwargs={"line": ...})``, i.e. ``lmplot``'s
    ``line_kws``, #199 -- see ``layer_name`` below). What it is NOT is the caller's chart-level catch-all
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
      one layer through the per-layer door, so the catch-all does NOT
      override it -- ``mark_smooth(stroke="#0000ff", layer_kwargs={"line":
      {"stroke": "#ff0000"}})`` draws the fit line ``#ff0000`` and the CI
      band ``#0000ff``, since only the line was named.

    Both merge sites read the same field, so the two shapes cannot drift:
    ``apply_catch_all_style`` for a layered result (per ``_Layer``) and
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
    degraded : tuple of str
        Sentences saying what this desugar had to drop or approximate, with
        no speaker. The desugar does not warn itself: the dispatcher warns
        each one as ``"{entry}: {note}"``, opened by the public call the
        caller made (``_PendingMark.entry``), so one note speaks as
        ``mark_violin`` or as ``catplot`` depending on who called. Both
        modes.
    layer_name : str or None
        Which catalogued sub-layer of the composite mark this single mark IS
        (``_overrides.LAYER_NAME_CATALOG[kind]``) -- e.g. ``desugar_smooth``'s
        no-CI shortcut emits one ``line``, the same layer the CI shape emits
        beside its ``ribbon``. It is what lets the per-layer door
        ``mark_smooth(layer_kwargs={"line": ...})`` reach a shape that has no
        ``_Layer`` to carry the name, without the dispatcher having to guess
        from the primitive mark name (``"line"`` the mark and ``"line"`` the
        layer coincide here and would not in general). Single-mark mode only;
        ``None`` when the mark is not a catalogued layer of anything.

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
    layer_name: Optional[str] = None  # the catalogued sub-layer this mark is
    degraded: tuple = ()  # what the desugar dropped, voiced by the dispatcher

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
        layered result records it per layer instead.  So does ``layer_name``:
        it names the single mark's own catalogued layer, and a layered result
        already carries a name per ``_Layer``.
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
        if self.layers is not None and self.layer_name is not None:
            raise ValueError(
                "MarkDesugarResult.layer_name is single-mark-only; it names "
                "the catalogued sub-layer the single mark is, so a layered "
                "result (layers set) must name each layer on _Layer.name "
                f"instead, got layer_name={self.layer_name!r}."
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
    #: The public entry the caller invoked, when a figure function or
    #: visualizer called the ``mark_*`` method on the caller's behalf (e.g.
    #: ``lmplot``); ``None`` means the ``mark_*`` method itself. Every layer
    #: this mark emits names it (``_Layer.entry``), and so does every warning
    #: its desugar reports (``MarkDesugarResult.degraded``).
    entry: str | None = None
    #: ``(channel, title)`` pairs a figure titled this mark's legends with
    #: (``ferrum.plots._helpers._title_legend``), applied to the bindings
    #: the desugar emits, whichever columns it chose (:func:`titled_encoding`).
    legend_titles: tuple = ()


def titled_encoding(encoding: dict, legend_titles: tuple) -> dict:
    """Return *encoding* with each ``(channel, title)`` in *legend_titles* titling its binding.

    Only a binding to a column takes the title: a channel the encoding does
    not bind stays unbound, and a constant stays a constant. *encoding* is
    not mutated.
    """
    from ferrum.encoding import _channel_class_for
    from ferrum.encoding.base import ChannelBase

    titled = dict(encoding)
    for channel, title in legend_titles:
        binding = titled.get(channel)
        if isinstance(binding, ChannelBase) and isinstance(binding.field, str):
            titled[channel] = binding._with_options(title=title)
        elif isinstance(binding, str):
            titled[channel] = _channel_class_for(channel)(binding, title=title)
    return titled
