"""MarkBase — kwarg validation + storage for mark style overrides.

Phase 8a: only constant overrides are supported (e.g. mark_point(size=100)).
Encoding-driven overrides come through .encode(size=Size("col")).
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

from ferrum._core import (
    blend_accepted_values,
    check_mark_size,
    interpolate_accepted_values,
    mark_bool_kwargs,
    mark_integer_kwargs,
    mark_kwargs_fields,
    mark_numeric_kwargs,
    mark_string_kwargs,
    stroke_cap_accepted_values,
    stroke_join_accepted_values,
)
from ferrum._entry import EntryRefusal
from ferrum._layer import (
    _MARKS_DRAWING_CONSTANT_SIZE,
    drawn_style_fields,
    layer_draws,
    layer_mark,
)
from ferrum._validate import (
    as_finite_real,
    coerce_bool,
    coerce_finite_real,
    coerce_non_negative_int,
    is_none_color_sentinel,
    is_real_number,
    validate_choice,
    validate_color_value,
    validate_optional_vocabulary,
    validate_str,
)

# Which style fields each mark draws is one Rust table
# (`ferrum._core.mark_style_fields`), and which constant `size=` values are
# legal is one Rust rule (`check_mark_size`); every layer meets both at
# render. Python runs them early, at each door that builds a mark's kwargs,
# so a refusal surfaces at construction. A composite mark takes every field
# some mark draws (`layer_style_fields`) and is held to the marks it emitted
# once its desugar has run (`refuse_style_no_layer_draws`). The table is
# read through ``_layer`` (`drawn_style_fields`), so the two modules cannot
# drift.

# Marks whose primary visual channel is stroke rather than fill.
# For these marks, the user-facing ``color=`` alias resolves to ``stroke``
# instead of ``fill``.  Fill-primary marks (bar, area, point, rect, arc,
# tick, …) keep the ``color`` → ``fill`` mapping.
_STROKE_PRIMARY_MARKS: frozenset[str] = frozenset(["line", "rule", "segment", "trail", "function"])

# Aliases map user-friendly names to their canonical renderer-level keys.
# ``color`` is handled separately in ``_canonical_mark_kwarg`` because the
# canonical target depends on whether the mark is stroke-primary or not.
# All other aliases are mark-type-independent.
_MARK_KWARG_ALIASES: dict[str, str] = {
    "alpha": "opacity",
    "linetype": "stroke_dash",
    "line_type": "stroke_dash",
}

_LINETYPE_MAP: dict[str, list[float]] = {
    "solid": [],
    "dashed": [4.0, 2.0],
    "dotted": [1.0, 3.0],
    "dashdot": [4.0, 2.0, 1.0, 2.0],
    "longdash": [8.0, 4.0],
}


_Checked = TypeVar("_Checked")


def _voiced(entry: str, check: Callable[[str], _Checked]) -> _Checked:
    """Return ``check(entry)``, its refusal an :class:`EntryRefusal` voiced as *entry*.

    *check* is one call of a pure validator, voiced as its argument (a
    :mod:`ferrum._validate` check, :func:`ferrum._core.check_mark_size`).
    The refusal's text for another voice re-calls that same validator with
    the other voice (:func:`_check_refusal_text`), so a figure function or
    visualizer that wrote the style re-voices it as itself
    (:func:`ferrum._entry.voiced_as`) and a mark method called directly
    keeps its own. Only *check*'s own ``ValueError`` becomes the refusal.
    """
    try:
        return check(entry)
    except ValueError as err:
        raise _validator_refusal(entry, check, str(err)) from None


def _refusal_of(entry: str, check: Callable[[str], Any]) -> EntryRefusal | None:
    """The refusal :func:`_voiced` raises for *check*, returned rather than raised.

    ``None`` when *check* accepts *entry*.
    """
    try:
        check(entry)
    except ValueError as err:
        return _validator_refusal(entry, check, str(err))
    return None


def _validator_refusal(entry: str, check: Callable[[str], Any], text: str) -> EntryRefusal:
    """*check*'s refusal *text*, voiced as *entry* and re-voiceable by re-calling *check*."""
    return EntryRefusal(lambda voice: _check_refusal_text(check, voice, text), entry)


def _check_refusal_text(check: Callable[[str], Any], voice: str, frozen: str) -> str:
    """The text *check*, a pure validator that refused one voice, refuses *voice* with.

    A pure validator judges the value alone, so it refuses every voice
    alike. Should *check* accept *voice*, the refusal keeps *frozen*, the
    text it was raised with, so re-voicing never turns a refusal into a
    different error.
    """
    try:
        check(voice)
    except ValueError as err:
        return str(err)
    return frozen


def _stroke_dash_error(context: str, value: Any) -> str:
    """Build the refusal message for a *value* :func:`_normalize_stroke_dash` rejects."""
    return (
        f"{context}={value!r} is not a valid stroke_dash value. Expected a "
        "numeric list (e.g. [4.0, 2.0]), a comma-separated numeric string "
        f'(e.g. "4,2"), or a named linetype (one of {sorted(_LINETYPE_MAP)}).'
    )


def _dash_lengths(lengths: Any, *, context: str, original: Any) -> list[float]:
    """Return *lengths* as floats, refusing any that is not a finite, non-negative real number.

    A length must be what :func:`ferrum._validate.as_finite_real` accepts (so
    never ``bool``, a numeric-looking string, or a value a float cannot hold;
    numpy scalars and ``Decimal`` are fine) and be non-negative. The refusal names
    *context* and *original*, the value the caller passed, via
    :func:`_stroke_dash_error`.
    """
    out = []
    for item in lengths:
        fv = as_finite_real(item)
        if fv is None or fv < 0:
            raise ValueError(_stroke_dash_error(context, original))
        out.append(fv)
    return out


def _normalize_stroke_dash(value: Any, *, context: str) -> list[float] | None:
    """Normalize a user-supplied ``stroke_dash=`` value to its wire form, a list of floats.

    The one dash vocabulary for every surface that accepts ``stroke_dash=``:
    :class:`MarkBase` (including the ``linetype=``/``line_type=`` aliases),
    ``annotate_hline``/``annotate_vline``, ``fm.SelectionMark``, and the
    ``stroke_dash`` conditional literal (``selection._literal_stroke_dash``).

    Accepts a closed set of forms:

    - ``None``: unset (solid / mark default); returned as ``None``.
    - a named linetype, one of ``sorted(_LINETYPE_MAP)``.
    - a comma-separated numeric string, spaces tolerated (``"4,2"``,
      ``"4, 2"``); ``""`` is the empty (solid) dash.
    - a real number (one dash length, ``[value]``), or a list/tuple of them.

    Every length must be a finite, non-negative real number (see
    :func:`_dash_lengths`). Anything else, including a ``bool``, a dict, or an
    array, raises ``ValueError`` naming *context* (the public entry and kwarg
    the caller invoked, without the value; this function appends
    ``={value!r}``) and the value.
    """
    if value is None:
        return None
    if isinstance(value, str):
        if value in _LINETYPE_MAP:
            lengths: Any = _LINETYPE_MAP[value]
        else:
            try:
                lengths = [float(x) for x in value.split(",") if x.strip()]
            except ValueError as exc:
                raise ValueError(_stroke_dash_error(context, value)) from exc
    elif is_real_number(value):
        lengths = [value]
    elif isinstance(value, (list, tuple)):
        lengths = value
    else:
        raise ValueError(_stroke_dash_error(context, value))
    return _dash_lengths(lengths, context=context, original=value)


def _is_paint_sentinel(value: str) -> bool:
    """Return ``True`` if *value* is a non-color paint sentinel, not a color.

    Mirrors ``resolve_paint_color`` in
    ``crates/ferrum-core/src/render/draw.rs`` exactly, so this Python gate
    accepts/rejects the identical set of strings as the Rust authority:

    - ``"theme:label"`` — exact string match (draw.rs matches it exactly,
      not trimmed, not case-folded). An internal theme-lookup token
      composite marks pass as ``stroke=``/``fill=`` (see
      ``marks/composite.py``).
    - ``"none"``/``"transparent"`` — trimmed and case-insensitive (draw.rs:
      ``trimmed.eq_ignore_ascii_case("none") || trimmed.eq_ignore_ascii_case("transparent")``),
      so ``"None"``, ``"NONE"``, ``" none "``, ``"Transparent"``,
      ``"TRANSPARENT"``, and ``" transparent "`` all count. Both spellings
      are an explicit paint-clear (spec §4.1; ``"transparent"`` joined
      ``"none"`` as a clearing spelling in the 2026-09-01 T8 quality-review
      supersession — refusing a real CSS Color 4 keyword with a message
      that promises CSS-name support recreated the same divergence class
      this batch remediates).

    All three sentinel spellings must be checked BEFORE
    ``ferrum.color.to_hex`` and never reach the parser — all raise inside
    ``to_hex`` by design, see
    ``tests/test_color_vocabulary.py::TestSentinelsAreNotColors``. The
    ``"none"``/``"transparent"`` arm is shared with
    ``ferrum._validate.is_none_color_sentinel`` (``selection.py`` composes
    from the same predicate for its own, differently-handled refusal).
    """
    return value == "theme:label" or is_none_color_sentinel(value)


# Canonical set of valid constant shape names for mark_point(shape=...).
# Must stay in sync with shape_from_str() in crates/ferrum-core/src/render/marks/point.rs.
_VALID_POINT_SHAPES: frozenset[str] = frozenset(
    [
        "circle",
        "square",
        "cross",
        "diamond",
        "triangle-up",
        "triangle_up",
        "triangle-down",
        "triangle_down",
        "|",
        "vline",
        "-",
        "hline",
    ]
)


#: Every ``mark_style`` wire field, in ``MarkKwargsSpec`` declaration order.
#: Each is drawn by at least one primitive mark (Rust's style table,
#: :func:`ferrum._core.mark_style_fields`).
_STYLE_FIELDS: tuple[str, ...] = tuple(mark_kwargs_fields())

#: The style fields only ferrum's own code writes, each with the public call
#: a caller sets it through. Rust reads them on the wire (a desugar or the
#: channel aliasing writes them), but no style door takes them.
_INTERNAL_STYLE_DOORS: dict[str, str] = {
    "detail": "encode(detail=...)",
    "clip": "mark_function(clip=...)",
    "value_axis": "the orient= of the statistical mark that draws it",
}

#: The kwargs a ``mark_<name>()`` method consumes itself rather than
#: forwarding as style: ``orient="horizontal"`` flips the chart's coordinates
#: (:meth:`MarkBase.orient_coord_flip`). Python-owned, because it describes
#: Python's methods; every other door refuses it, naming the method.
METHOD_ONLY_KWARGS: dict[str, frozenset[str]] = {
    "bar": frozenset({"orient"}),
    "tick": frozenset({"orient"}),
}

#: The ``orient=`` values every mark method, desugar and figure function takes
#: (``orientation=`` on ``mark_density``/``mark_histogram``).
ORIENT_CHOICES = ("horizontal", "vertical")

#: The style kwargs whose wire field is an unsigned integer: the names Rust
#: publishes as ``MarkKwargsSpec``'s ``usize`` fields. Each is stored as
#: :func:`ferrum._validate.coerce_non_negative_int` returns it, so
#: ``np.int64(3)`` reaches the wire as ``3`` and ``2.5``, ``-1`` or ``True``
#: refuses at the door.
_INTEGER_MARK_KWARGS: frozenset[str] = frozenset(mark_integer_kwargs())

#: The style kwargs whose wire field is a real number: the rest of the names
#: Rust publishes as ``MarkKwargsSpec``'s numeric fields. Each is stored as
#: :func:`ferrum._validate.coerce_finite_real` returns it, so a numpy scalar
#: or ``Decimal`` reaches the wire as a builtin number and a non-number or
#: non-finite value refuses at the door.
_REAL_MARK_KWARGS: frozenset[str] = frozenset(mark_numeric_kwargs()) - _INTEGER_MARK_KWARGS

#: The style kwargs whose wire field is a truth value: the names Rust
#: publishes as ``MarkKwargsSpec``'s ``bool`` fields. Each is stored as
#: :func:`ferrum._validate.coerce_bool` returns it, so ``np.False_`` reaches
#: the wire as ``false`` and ``"no"`` or ``0`` refuses at the door rather
#: than being read for its truthiness.
_BOOL_MARK_KWARGS: frozenset[str] = frozenset(mark_bool_kwargs())

#: The style kwargs whose wire field is a string: the names Rust publishes as
#: ``MarkKwargsSpec``'s string-valued fields (``shape``, ``fill`` and
#: ``stroke`` among them). A non-string refuses at the door, ahead of the
#: color, shape and vocabulary checks that judge a string's content.
_STRING_MARK_KWARGS: frozenset[str] = frozenset(mark_string_kwargs())

#: The style fields a Rust vocabulary governs, each with its accepted values
#: in the order Rust publishes them.
_STYLE_VOCABULARIES: dict[str, list[str]] = {
    "interpolate": interpolate_accepted_values(),
    "stroke_cap": stroke_cap_accepted_values(),
    "stroke_join": stroke_join_accepted_values(),
    "blend": blend_accepted_values(),
}


@functools.cache
def layer_style_fields(mark_name: str) -> frozenset[str]:
    """The style kwargs a layer drawing *mark_name* takes at a public door.

    A mark with a row in Rust's style table takes the fields that row draws
    (:func:`ferrum._layer.drawn_style_fields`). Any other mark name (a
    composite, statistical or diagnostic mark) takes every field some mark
    draws: it spreads its style over the marks it emits, and a field none of
    them draws refuses once they are known
    (:func:`refuse_style_no_layer_draws`). The internal fields
    (:data:`_INTERNAL_STYLE_DOORS`) are never taken.
    """
    drawn = drawn_style_fields(mark_name)
    taken = drawn if drawn is not None else frozenset(_STYLE_FIELDS)
    return taken - frozenset(_INTERNAL_STYLE_DOORS)


def honored_kwargs(mark_name: str) -> frozenset[str]:
    """The kwargs ``mark_<mark_name>(...)`` honors (aliases already resolved).

    :func:`layer_style_fields` plus the kwargs the method consumes itself
    (:data:`METHOD_ONLY_KWARGS`).

    Examples
    --------
    >>> sorted(honored_kwargs("bar") - layer_style_fields("bar"))
    ['orient']
    >>> "font_size" in honored_kwargs("point")
    False
    """
    return layer_style_fields(mark_name) | METHOD_ONLY_KWARGS.get(mark_name, frozenset())


def mark_honors(mark_name: str, kwarg: str) -> bool:
    """Return ``True`` if ``mark_<mark_name>(...)`` honors the (already alias-resolved) *kwarg*.

    Examples
    --------
    >>> mark_honors("bar", "zero")
    True
    >>> mark_honors("bar", "arrow")
    False
    >>> mark_honors("bar", "size")
    False
    >>> mark_honors("boxplot", "opacity")
    True
    """
    return kwarg in honored_kwargs(mark_name)


def _internal_field_error(key: str, *, entry: str) -> str:
    """The refusal of an internal style field *key*, naming the call that sets it."""
    return (
        f"{entry}: keyword argument {key!r} is not a style kwarg; set it through "
        f"{_INTERNAL_STYLE_DOORS[key]}"
    )


def _mark_kwarg_error(mark_name: str, key: str, honored: frozenset[str], *, entry: str) -> str:
    """Build the refusal message for one kwarg *key* that *mark_name* does not take.

    Voiced as *entry*, the public call that received the kwarg, which is
    ``mark_<mark_name>`` unless another door (a figure function, a
    visualizer, ``Chart.override``) wrote it into the mark. *honored* is
    the set the door took, named in the message.

    Three phrasings:

    - *key* is a field only ferrum's own code writes
      (:data:`_INTERNAL_STYLE_DOORS`): it names the public call that sets it.
    - *key* names no style field on any mark at all (a genuine typo, e.g.
      ``mark_point(squiggly=True)``): "unknown keyword argument".
    - *key* is a real field, just not one *mark_name* draws (e.g.
      ``mark_bar(arrow=True)``): "is not honored by".
    """
    if key in _INTERNAL_STYLE_DOORS:
        return _internal_field_error(key, entry=entry)
    if key not in _VALID_MARK_KWARGS:
        return f"{entry}: unknown keyword argument {key!r}. Honored: {sorted(honored)}"
    return (
        f"{entry}: keyword argument {key!r} is not honored by mark_{mark_name}. "
        f"Honored: {sorted(honored)}"
    )


def refuse_size_no_layer_draws(entry: str, size: float, drawn_marks: list[str | None]) -> None:
    """Hold a composite mark's constant `size=` to the size rule of the marks it emits.

    A composite, statistical or diagnostic mark spreads its catch-all
    `size=` over the marks it emits that draw it
    (:data:`_MARKS_DRAWING_CONSTANT_SIZE`). *drawn_marks* are the primitive
    marks the desugar emitted. The value meets the first drawing mark's rule
    or, when none of them draws `size`, refuses as inert, both through
    :func:`ferrum._core.check_mark_size` and voiced as *entry*, the public
    call the caller invoked. Called by the composite dispatcher once the
    desugar has run, which is the first point the emitted marks are known.
    """
    if not drawn_marks:
        # The desugar emitted no mark at all, so no mark's size rule applies.
        raise EntryRefusal(
            lambda voice: (
                f"{voice}(size=...): {voice} draws no marks for this call, so size= has "
                f"nothing to size; received {size!r}."
            ),
            entry,
        )
    drawing = [mark for mark in drawn_marks if mark in _MARKS_DRAWING_CONSTANT_SIZE]
    mark = layer_mark((drawing or drawn_marks)[0])
    _voiced(entry, lambda voice: check_mark_size(voice, mark, size))


def refuse_style_no_layer_draws(
    entry: str, style: Mapping[str, Any], drawn_marks: list[str | None]
) -> None:
    """Hold a composite mark's catch-all *style* to the marks it emitted.

    *style* is the resolved catch-all the composite spreads over the layers
    its desugar emitted, whose marks are *drawn_marks*; each field reaches
    only the layers whose mark draws it
    (:func:`ferrum._layer.apply_catch_all_style`). A constant ``size`` meets
    :func:`refuse_size_no_layer_draws`. Any other field that no emitted mark
    draws refuses, voiced as *entry*, the public call the caller invoked,
    since nothing would take it. Called once the desugar has run, the first
    point the emitted marks are known.
    """
    if "size" in style:
        refuse_size_no_layer_draws(entry, style["size"], drawn_marks)
    untaken = [
        key
        for key in style
        if key != "size" and not any(layer_draws(mark, key) for mark in drawn_marks)
    ]
    if untaken:
        marks = sorted({layer_mark(mark) for mark in drawn_marks})
        raise EntryRefusal(
            lambda voice: (
                f"{voice}: keyword argument(s) {untaken!r} are drawn by no mark this call "
                f"draws ({marks}), so nothing would take them"
            ),
            entry,
        )


def check_constant_size(entry: str, mark: str, size: Any) -> float | None:
    """Return *size* as a number once it meets the one constant-size rule for *mark*.

    ``None`` means no size was set and passes through unchanged.

    *mark* is a primitive mark name. A non-number or non-finite value
    refuses as :func:`ferrum._validate.coerce_finite_real` does; a mark that
    never draws a constant size, or a value that is not above zero, refuses
    with the render refusal's own text (:func:`ferrum._core.check_mark_size`).
    Every door that writes a constant ``size`` into a primitive mark's
    kwargs calls this, voiced as the public call the caller made (*entry*).
    Refusals are :class:`ferrum._entry.EntryRefusal`.
    """
    value = _voiced(entry, lambda voice: coerce_finite_real(f"{voice}(size=...)", size))
    if value is not None:
        _voiced(entry, lambda voice: check_mark_size(voice, mark, value))
    return value


def _mark_kwargs_error_plural(mark_name: str, keys: list[str], *, entry: str) -> str:
    """Plural counterpart of :func:`_mark_kwarg_error` for a batch of *keys*
    (used by ``validate_user_mark_kwargs``, which reports every offending
    key from one call rather than failing on the first), voiced as *entry*."""
    honored = sorted(honored_kwargs(mark_name))
    if all(key not in _VALID_MARK_KWARGS for key in keys):
        return f"{entry}: unknown keyword argument(s) {keys!r}. Honored: {honored}"
    return (
        f"{entry}: keyword argument(s) {keys!r} not honored by mark_{mark_name}. Honored: {honored}"
    )


#: Every kwarg some style door can name: the wire fields and the method-only
#: kwargs. A key outside it is a typo ("unknown keyword argument"); a key in
#: it that a door does not take is refused as not honored there. Also the
#: leaf vocabulary of the ``Chart.override(mark_<leaf>=...)`` registry
#: (``_override_apply.py``), which is built with no per-chart mark context;
#: the per-mark check runs where the style is applied.
_VALID_MARK_KWARGS: frozenset[str] = frozenset(_STYLE_FIELDS).union(*METHOD_ONLY_KWARGS.values())


def _validate_literal_color(entry: str, key: str, value: str) -> None:
    """Raise ``ValueError`` if *value* is not a color ``ferrum.color.to_hex`` accepts.

    A thin, label-composing wrapper over
    :func:`ferrum._validate.validate_color_value` (close-remediation, batch
    F) — the shared gate ``configure._validate_config_color`` also composes
    over, collapsing what were two independently hand-copied ``try:
    to_hex(...) except ValueError: raise ValueError(...)`` bodies into one.
    Not a second color vocabulary — the raised message wraps to_hex's own
    accepted-forms text unchanged. Callers must short-circuit the non-color
    sentinels (``"none"``, ``"transparent"``, ``"theme:label"``) before
    calling this function — each raises here by design (``"none"``/
    ``"transparent"`` with to_hex's sentinel-aware clearing-paint message,
    ``"theme:label"`` with the generic accepted-forms text), matching
    to_hex's behavior. This mark-kwarg call site already restricts *value*
    to ``str`` before calling (``_resolve_style``'s string-field gate), so it
    leaves ``validate_color_value``'s ``require_str`` at its default
    ``False`` -- unlike ``configure.py``'s config-surface fields, which
    request it to refuse a structurally-valid-to-``to_hex`` non-``str``
    (e.g. an RGB(A) tuple) at construction.

    The ``ferrum.color`` import is lazy (inside this function, not at
    module load time) to avoid a module-init-order dependency: ``ferrum.marks.base``
    cannot be reached without ``ferrum._core`` already having loaded
    (``ferrum/__init__.py`` imports it unconditionally before anything
    else), so ``ferrum.color`` — whose only import is ``ferrum._core`` — is
    always importable by the time this runs.
    """
    from ferrum.color import to_hex

    validate_color_value(f"{entry}: {key}", value, to_hex)


def _canonical_mark_kwarg(mark_name: str, key: str) -> str:
    """The style field *key* names on *mark_name*, its aliases resolved.

    ``color`` is mark-type-aware: stroke-primary marks (line, rule, segment,
    trail, function) map it to ``stroke``, every other mark to ``fill``. The
    other aliases (``alpha``, ``linetype``) are mark-independent; a canonical
    key maps to itself.
    """
    if key == "color":
        return "stroke" if mark_name in _STROKE_PRIMARY_MARKS else "fill"
    return _MARK_KWARG_ALIASES.get(key, key)


def _untaken_key_refusal(
    entry: str, mark_name: str, key: str, value: Any, honored: frozenset[str]
) -> EntryRefusal:
    """The refusal a door holding *mark_name* to *honored* gives *key* = *value*, a key it does not take.

    *key* (aliases resolved) is outside *honored*; every such key is
    refused, so a caller asks :func:`layer_honors` or *honored* whether a
    key is taken and this function only how it is refused. The one order
    every door refuses a key it does not take in, except :func:`_resolve_style`
    treats ``size=None`` as unset before checking if size is taken:

    - an argument only ``mark_<mark_name>()`` takes
      (:data:`METHOD_ONLY_KWARGS`) names that method;
    - a constant ``size`` on a primitive mark that the one size rule
      (:func:`check_constant_size`) refuses gets that rule's text, which
      names the mark's own width or font kwarg where it has one;
    - any other key, a ``size`` the size rule lets through among them, is
      refused as unknown, internal or not honored by ``mark_<mark_name>``
      (:func:`_mark_kwarg_error`).

    The refusal is voiced as *entry*, and returned rather than raised so a
    door that spreads one style dict over several layers can name every
    layer's refusal.
    """
    canonical = _canonical_mark_kwarg(mark_name, key)
    if canonical in METHOD_ONLY_KWARGS.get(mark_name, frozenset()):
        return EntryRefusal(_method_only_message(mark_name, canonical), entry)
    if canonical == "size" and drawn_style_fields(mark_name) is not None:
        size_refusal = _refusal_of(
            entry, lambda voice: check_constant_size(voice, mark_name, value)
        )
        if size_refusal is not None:
            return size_refusal
    return EntryRefusal(
        lambda voice: _mark_kwarg_error(mark_name, canonical, honored, entry=voice), entry
    )


def _resolve_style(
    entry: str, mark_name: str, kwargs: Mapping[str, Any], honored: frozenset[str]
) -> dict[str, Any]:
    """Return *kwargs* with aliases resolved and values checked, each key held to *honored*.

    The body of both style rules, :func:`resolve_mark_kwargs` (a mark
    method's kwargs) and :func:`resolve_layer_style` (a layer's), which
    differ only in the set they take. Every refusal is an
    :class:`ferrum._entry.EntryRefusal` built at its raise site: a message
    template, or one pure validator re-called in the new voice
    (:func:`_voiced`). Each ``lambda`` below is called before the loop
    moves on, or re-called once the refusal has ended it, so it always
    reads the key and value it refused.
    """
    resolved: dict[str, Any] = {}
    for k, v in kwargs.items():
        canonical = _canonical_mark_kwarg(mark_name, k)
        if canonical == "size" and drawn_style_fields(mark_name) is not None:
            # A constant `size=` on a primitive meets the one size rule,
            # taken or not: ``None`` is unset and passes on any primitive,
            # and any other value the mark never draws is refused by it.
            resolved[canonical] = check_constant_size(entry, mark_name, v)
            continue
        # A kwarg the mark does not take is refused as such before its
        # value is judged.
        if canonical not in honored:
            raise _untaken_key_refusal(entry, mark_name, k, v, honored)
        if canonical == "stroke_dash":
            # Named forms ("dashed", "dotted", ...), the comma-split
            # form, and list/tuple passthrough are all resolved by the
            # one shared helper -- reached identically whether the user
            # wrote the canonical ``stroke_dash=`` kwarg or either alias,
            # so a named string works either way.
            v = _voiced(entry, lambda voice: _normalize_stroke_dash(v, context=f"{voice}: {k}"))
        elif canonical in _INTEGER_MARK_KWARGS:
            v = _voiced(entry, lambda voice: coerce_non_negative_int(f"{voice}({k}=...)", v))
        elif canonical in _REAL_MARK_KWARGS:
            v = _voiced(entry, lambda voice: coerce_finite_real(f"{voice}({k}=...)", v))
        # ``None`` is the unset value of every optional field, as it is for
        # the numeric fields above. A field a Rust vocabulary governs is
        # left to that vocabulary's check below, which refuses a non-string
        # as it refuses any other value outside the vocabulary.
        elif canonical in _BOOL_MARK_KWARGS and v is not None:
            v = _voiced(entry, lambda voice: coerce_bool(voice, k, v))
        elif (
            canonical in _STRING_MARK_KWARGS
            and canonical not in _STYLE_VOCABULARIES
            and v is not None
        ):
            _voiced(entry, lambda voice: validate_str(voice, k, v))
        elif canonical == "orient":
            _voiced(entry, lambda voice: validate_choice(voice, "orient", v, ORIENT_CHOICES))
        resolved[canonical] = v
    # Validate shape= value for point marks. The constant shape is a fixed
    # string that must name one of the glyphs supported by the Rust renderer;
    # an unknown name would silently default to circle at render time.
    shape_val = resolved.get("shape")
    if shape_val is not None:
        _voiced(
            entry, lambda voice: validate_choice(voice, "shape", shape_val, _VALID_POINT_SHAPES)
        )
    # A value outside a vocabulary Rust publishes is refused at declaration,
    # in the text the same value is refused with at render.
    for field_name, accepted in _STYLE_VOCABULARIES.items():
        _voiced(
            entry,
            lambda voice: validate_optional_vocabulary(
                voice, field_name, resolved.get(field_name), accepted
            ),
        )
    # Validate literal fill/stroke color strings at construction time via
    # ferrum's single Rust color parser (ferrum.color.to_hex) so bad
    # colors fail immediately rather than silently at render time. The
    # ``color=`` alias is already folded into ``fill``/``stroke`` above,
    # so checking those two canonical keys covers all three user-facing
    # spellings. Paint sentinels (see ``_is_paint_sentinel``) are
    # short-circuited ahead of the parser — ``"theme:label"``,
    # ``"none"`` and ``"transparent"`` all raise in to_hex by design and
    # are not colors. The stored value stays the user's
    # original string; to_hex is consulted only for validation.
    for paint_key in ("fill", "stroke"):
        paint_val = resolved.get(paint_key)
        if paint_val is not None and not _is_paint_sentinel(paint_val):
            _voiced(entry, lambda voice: _validate_literal_color(voice, paint_key, paint_val))
    return resolved


def resolve_mark_kwargs(entry: str, mark_name: str, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Return *kwargs* as ``mark_<mark_name>(...)`` honors them: aliases resolved, values checked.

    The honored-kwarg rule for a mark method's kwargs (through
    :class:`MarkBase`), and for a composite's catch-all style. Every refusal
    is an :class:`ferrum._entry.EntryRefusal` voiced as *entry*, the public
    call that received the kwarg: a kwarg the method does not honor
    (:func:`honored_kwargs`), a constant ``size`` that breaks the size rule
    (:func:`check_constant_size`), an unknown point ``shape``, an
    ``orient`` or a value of a Rust vocabulary (``interpolate``,
    ``stroke_cap``, ``stroke_join``, ``blend``) outside its vocabulary, a
    non-numeric value for a numeric field, a value that is not ``True`` or
    ``False`` for a truth-valued field, a non-string value for a
    string-valued field (``shape``, ``fill`` and ``stroke`` among them), or
    a literal ``fill``/``stroke`` that is not a color.
    """
    return _resolve_style(entry, mark_name, kwargs, honored_kwargs(mark_name))


def resolve_layer_style(entry: str, mark: str | None, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Return a caller's style *kwargs* for one layer drawing *mark*, as *mark* honors them.

    The rule of every door that styles a layer rather than calling a mark
    method: ``fm.Layer(mark_kwargs=)``, a figure function's ``mark=`` dict
    or ``*_kws``, a composite's ``layer_kwargs=`` and
    ``Chart.override(mark_<leaf>=)``. A layer takes the fields its mark
    draws (:func:`layer_style_fields`); a kwarg its mark's method consumes
    itself (:data:`METHOD_ONLY_KWARGS`) refuses, naming that method. A
    layer with no *mark* draws a point (:func:`ferrum._layer.layer_mark`)
    and is held to the point's fields. Refusals are
    :class:`ferrum._entry.EntryRefusal`, voiced as *entry*.
    """
    mark = layer_mark(mark)
    return _resolve_style(entry, mark, kwargs, layer_style_fields(mark))


def _method_only_message(mark: str, key: str) -> Callable[[str], str]:
    """The refusal text, for any voice, of *key*, an argument only ``mark_<mark>()`` takes."""
    return lambda voice: (
        f"{voice}: keyword argument {key!r} is an argument of mark_{mark}(), not a "
        f"layer style; set it through mark_{mark}({key}=...)"
    )


def refuse_untaken_layer_style(
    entry: str, marks: Sequence[str | None], untaken: Mapping[str, Any]
) -> None:
    """Refuse style *untaken* that no layer of a chart drawing *marks* takes.

    A door that spreads one style dict over layers of different marks calls
    this with the keys no layer took. Each layer refuses each key
    (:func:`_untaken_key_refusal`): a kwarg only the mark's method takes
    names that method, a constant ``size`` the mark never draws that the
    size rule refuses gets the rule's text, and any other key, an unset
    ``size=None`` among them, is refused as not honored by the mark. One key on a chart of one mark
    refuses with that layer's text; otherwise the refusal names every
    untaken key and every layer's mark, then gives each layer's refusal of
    each key. Voiced as *entry*. No keys, no refusal.
    """
    if not untaken:
        return
    layer_marks = list(dict.fromkeys(layer_mark(mark) for mark in marks))
    refusals = [
        _untaken_key_refusal(entry, mark, key, value, layer_style_fields(mark))
        for key, value in untaken.items()
        for mark in layer_marks
    ]
    if len(refusals) == 1:
        raise refusals[0]
    keys = list(untaken)
    drawn = ", ".join(f"mark_{mark}" for mark in layer_marks)
    raise EntryRefusal(
        lambda voice: (
            f"{voice}: no layer of this chart ({drawn}) takes {keys!r}: "
            + "; ".join(refusal.message_for(voice) for refusal in refusals)
        ),
        entry,
    )


def resolve_method_bool(entry: str, param: str, value: Any, default: bool) -> bool:
    """Return a truth-valued argument a mark method consumes itself, as a ``bool``.

    ``mark_errorband(borders=)``, ``mark_qq(line=, dequantize=)``,
    ``mark_contour(fill=, smooth=)``, ``mark_function(clip=)`` and
    ``mark_density(fill=, cumulative=, smooth=)`` read their argument to
    decide what the composite draws or computes instead of forwarding it as
    style, so the style gate never sees it. This holds it to the same rule :func:`_resolve_style`
    holds the field to on a layer: ``None`` is unset (*default*), and
    anything else must be a truth value (:func:`ferrum._validate.coerce_bool`),
    never read for its truthiness. Refusals are
    :class:`ferrum._entry.EntryRefusal`, voiced as *entry*.
    """
    if value is None:
        return default
    return _voiced(entry, lambda voice: coerce_bool(voice, param, value))


def layer_honors(mark: str | None, key: str) -> bool:
    """Whether a layer drawing *mark* honors the style kwarg *key* (aliases resolved).

    The membership half of :func:`resolve_layer_style`, for a door that
    spreads one style dict over layers of different marks. A layer with no
    *mark* draws a point.
    """
    mark = layer_mark(mark)
    return _canonical_mark_kwarg(mark, key) in layer_style_fields(mark)


class MarkBase:
    """Validate and store mark-level keyword arguments for primitive marks.

    Used by ``mark_*()`` builder functions to validate kwargs before
    serializing them into ``ChartSpec.mark_style``. Raises ``ValueError``
    on the first keyword argument *mark_name* does not honor — whether
    because it names no known style field at all, or because it names a
    field the mark never draws (Rust's style table) — so typos and
    doesn't-apply-here kwargs both surface immediately rather than
    silently disappearing into the renderer.

    Parameters
    ----------
    mark_name : str
        Name of the mark (e.g. ``"point"``, ``"bar"``). Used in error
        messages only.
    **kwargs : Any
        Style overrides forwarded to the renderer (e.g. ``size=100``,
        ``fill="red"``). Every key must be honored by *mark_name* per
        :func:`mark_honors` (aliases resolve first).

    Raises
    ------
    ValueError
        If any keyword argument is not honored by *mark_name*, or if a
        constant ``size=`` on a primitive mark breaks the one size rule
        (:func:`check_constant_size`): the mark never draws a constant
        size, or the value is not a finite number above zero.

    Examples
    --------
    >>> from ferrum.marks.base import MarkBase
    >>> mb = MarkBase("point", size=80, fill="steelblue")
    >>> mb.to_mark_kwargs_dict()
    {'size': 80, 'fill': 'steelblue'}
    """

    def __init__(self, mark_name: str, **kwargs: Any) -> None:
        self.mark_name = mark_name
        self._kwargs = resolve_mark_kwargs(f"mark_{mark_name}", mark_name, kwargs)

    @property
    def kwargs(self) -> dict[str, Any]:
        """Read-only view of the resolved (alias-expanded) kwargs dict."""
        return dict(self._kwargs)

    def to_mark_kwargs_dict(self) -> dict:
        """Return the subset of stored kwargs that map to ``MarkKwargsSpec`` fields.

        Statistical mark kwargs — the "forwarded to the transform" group in
        *Mark style kwargs* in the marks & encodings guide
        (``docs/site/guide/marks-encodings.md``), e.g.
        ``bandwidth``, ``method``, ``ci`` — are not included here; they are
        consumed directly by the desugar functions (``desugar_density``,
        ``desugar_smooth``, …) which build the transform objects before
        this dict is ever inspected.

        ``orient`` is consumed Python-side (sets ``_coord = "flip"`` on the
        chart) and is never forwarded to the Rust renderer. ``zero`` is
        included: a composite spreads it to the bar layers it emits, and
        ``Chart._set_mark`` consumes it on a single bar chart
        (:meth:`zero_anchor`).

        Returns
        -------
        dict
            Mapping of renderer-level style keys to their values, in
            ``MarkKwargsSpec`` declaration order.  Only keys present in the
            stored kwargs are included; absent keys do not appear (no
            ``None`` defaults).

        Examples
        --------
        >>> mb = MarkBase("point", size=40, opacity=0.8)
        >>> mb.to_mark_kwargs_dict()
        {'size': 40, 'opacity': 0.8}
        """
        return {k: self._kwargs[k] for k in _STYLE_FIELDS if k in self._kwargs}

    def zero_anchor(self) -> bool:
        """Return the effective value of the ``zero=`` parameter (default ``True``).

        Used by ``Chart._set_mark`` to store the zero-anchor preference on the
        chart so ``_build_encoding_specs`` can suppress the ``scale.zero=True``
        injection when the caller passes ``zero=False``.

        Returns
        -------
        bool
            ``False`` when ``zero=False`` was explicitly passed; ``True`` otherwise.
            The stored value is already a ``bool`` (or ``None``, unset), since
            the door refused any other.

        Examples
        --------
        >>> MarkBase("bar", zero=False).zero_anchor()
        False
        >>> MarkBase("bar").zero_anchor()
        True
        """
        zero = self._kwargs.get("zero")
        return True if zero is None else zero

    def orient_coord_flip(self) -> bool:
        """Return True if ``orient="horizontal"`` was passed, indicating coord flip.

        Used by ``Chart._set_mark`` to set ``_coord = "flip"`` without
        forwarding ``orient`` to the Rust renderer.

        Returns
        -------
        bool
            True when ``orient="horizontal"`` is in the stored kwargs.

        Examples
        --------
        >>> mb = MarkBase("bar", orient="horizontal")
        >>> mb.orient_coord_flip()
        True
        >>> MarkBase("bar").orient_coord_flip()
        False
        """
        return self._kwargs.get("orient") == "horizontal"
