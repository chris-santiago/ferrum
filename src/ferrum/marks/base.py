"""MarkBase — kwarg validation + storage for mark style overrides.

Phase 8a: only constant overrides are supported (e.g. mark_point(size=100)).
Encoding-driven overrides come through .encode(size=Size("col")).
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from ferrum._core import (
    blend_accepted_values,
    check_mark_size,
    interpolate_accepted_values,
    mark_integer_kwargs,
    mark_kwargs_fields,
    mark_numeric_kwargs,
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
    coerce_finite_real,
    coerce_non_negative_int,
    is_none_color_sentinel,
    is_real_number,
    validate_choice,
    validate_color_value,
    validate_optional_vocabulary,
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


_Resolver = TypeVar("_Resolver", bound=Callable[..., Any])


def _refuses_in_any_voice(resolve: _Resolver) -> _Resolver:
    """Make every refusal of ``resolve(entry, ...)`` an :class:`EntryRefusal`.

    *resolve* judges a caller's style and raises ``ValueError`` voiced as
    its first argument, the public call that received the style. The
    refusal it raises instead carries the same judgment for any voice, so a
    figure function or visualizer that wrote the style re-voices it as
    itself (:func:`ferrum._entry.voiced_as`) and a mark method called
    directly keeps its own. An :class:`EntryRefusal` raised inside
    *resolve* already has that form and propagates unchanged.
    """

    @functools.wraps(resolve)
    def wrapper(entry: str, *args: Any) -> Any:
        try:
            return resolve(entry, *args)
        except EntryRefusal:
            raise
        except ValueError as err:
            first = str(err)

        def message_for(voice: str) -> str:
            try:
                resolve(voice, *args)
            except ValueError as err:
                return str(err)
            return first

        raise EntryRefusal(message_for, entry) from None

    return wrapper  # type: ignore[return-value]


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

    A primitive mark takes the fields Rust's style table says it draws
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


@_refuses_in_any_voice
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
        raise ValueError(
            f"{entry}(size=...): {entry} draws no marks for this call, so size= has "
            f"nothing to size; received {size!r}."
        )
    drawing = [mark for mark in drawn_marks if mark in _MARKS_DRAWING_CONSTANT_SIZE]
    check_mark_size(entry, layer_mark((drawing or drawn_marks)[0]), size)


@_refuses_in_any_voice
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
        raise ValueError(
            f"{entry}: keyword argument(s) {untaken!r} are drawn by no mark this call draws "
            f"({marks}), so nothing would take them"
        )


@_refuses_in_any_voice
def check_constant_size(entry: str, mark: str, size: Any) -> float | None:
    """Return *size* as a number once it meets the one constant-size rule for *mark*.

    ``None`` means no size was set and passes through unchanged.

    *mark* is a primitive mark name. A non-number or non-finite value
    refuses as :func:`ferrum._validate.coerce_finite_real` does; a mark that
    never draws a constant size, or a value that is not above zero, refuses
    with the render refusal's own text (:func:`ferrum._core.check_mark_size`).
    Every door that writes a constant ``size`` into a primitive mark's
    kwargs calls this, voiced as the public call the caller made (*entry*).
    """
    value = coerce_finite_real(f"{entry}(size=...)", size)
    if value is not None:
        check_mark_size(entry, mark, value)
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
    to ``str`` before calling (see its own ``isinstance`` guard), so it
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
    trail) map it to ``stroke``, every other mark to ``fill``. The other
    aliases (``alpha``, ``linetype``) are mark-independent; a canonical key
    maps to itself.
    """
    if key == "color":
        return "stroke" if mark_name in _STROKE_PRIMARY_MARKS else "fill"
    return _MARK_KWARG_ALIASES.get(key, key)


def _resolve_style(
    entry: str, mark_name: str, kwargs: Mapping[str, Any], honored: frozenset[str]
) -> dict[str, Any]:
    """Return *kwargs* with aliases resolved and values checked, each key held to *honored*.

    The body of both style rules, :func:`resolve_mark_kwargs` (a mark
    method's kwargs) and :func:`resolve_layer_style` (a layer's), which
    differ only in the set they take.
    """
    resolved: dict[str, Any] = {}
    for k, v in kwargs.items():
        canonical = _canonical_mark_kwarg(mark_name, k)
        # A constant `size=` on a primitive meets the one size rule,
        # which names the mark's own width/font kwarg where one exists,
        # ahead of the generic not-honored door below.
        if canonical == "size" and drawn_style_fields(mark_name) is not None:
            resolved[canonical] = check_constant_size(entry, mark_name, v)
            continue
        # A kwarg the mark does not take is refused as such before its
        # value is judged.
        if canonical not in honored:
            raise ValueError(_mark_kwarg_error(mark_name, canonical, honored, entry=entry))
        if canonical == "stroke_dash":
            # Named forms ("dashed", "dotted", ...), the comma-split
            # form, and list/tuple passthrough are all resolved by the
            # one shared helper -- reached identically whether the user
            # wrote the canonical ``stroke_dash=`` kwarg or either alias,
            # so a named string works either way.
            v = _normalize_stroke_dash(v, context=f"{entry}: {k}")
        elif canonical in _INTEGER_MARK_KWARGS:
            v = coerce_non_negative_int(f"{entry}({k}=...)", v)
        elif canonical in _REAL_MARK_KWARGS:
            v = coerce_finite_real(f"{entry}({k}=...)", v)
        elif canonical == "orient":
            validate_choice(entry, "orient", v, ORIENT_CHOICES)
        resolved[canonical] = v
    # Validate shape= value for point marks. The constant shape is a fixed
    # string that must name one of the glyphs supported by the Rust renderer;
    # an unknown name would silently default to circle at render time.
    shape_val = resolved.get("shape")
    if shape_val is not None and isinstance(shape_val, str):
        validate_choice(entry, "shape", shape_val, _VALID_POINT_SHAPES)
    # A value outside a vocabulary Rust publishes is refused at declaration,
    # in the text the same value is refused with at render.
    for field_name, accepted in _STYLE_VOCABULARIES.items():
        validate_optional_vocabulary(entry, field_name, resolved.get(field_name), accepted)
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
        if isinstance(paint_val, str) and not _is_paint_sentinel(paint_val):
            _validate_literal_color(entry, paint_key, paint_val)
    return resolved


@_refuses_in_any_voice
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
    non-numeric value for a numeric field, or a literal ``fill``/``stroke``
    that is not a color.
    """
    return _resolve_style(entry, mark_name, kwargs, honored_kwargs(mark_name))


@_refuses_in_any_voice
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
    for key in kwargs:
        if key in METHOD_ONLY_KWARGS.get(mark, frozenset()):
            raise ValueError(
                f"{entry}: keyword argument {key!r} is an argument of mark_{mark}(), not a "
                f"layer style; set it through mark_{mark}({key}=...)"
            )
    return _resolve_style(entry, mark, kwargs, layer_style_fields(mark))


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

        Examples
        --------
        >>> MarkBase("bar", zero=False).zero_anchor()
        False
        >>> MarkBase("bar").zero_anchor()
        True
        """
        return bool(self._kwargs.get("zero", True))

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
