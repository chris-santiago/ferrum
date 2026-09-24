"""Shared zero-import leaf-validation predicates for the rest of ``ferrum``.

Independent predicates live here:

- ``is_real_number`` is the one answer to "is this a real number" for every
  numeric parameter: any ``numbers.Real`` or ``decimal.Decimal``, never
  ``bool``. The pixel/fraction validators, ``Inset.bounds``, dash lengths and
  the transform literal serializers all ask it.

- ``validate_choice`` is the canonical way to reject a value that is not a
  member of a documented closed vocabulary (a ``kind=``/``method=``/``order=``
  style keyword, or a dataclass field with a fixed set of legal values). It
  replaces the inconsistent idioms that predate it — different message
  templates, different punctuation, unsorted set reprs — with one message
  shape across the whole package.

  ``func_name`` convention: pass the public callable name the caller invoked
  (e.g. ``"pairplot"``), or ``"Class.attr"`` for a dataclass field validated
  in ``__post_init__`` (e.g. ``"BreakAxis.axis"``).

- ``is_none_color_sentinel`` is the single Python definition of the
  paint-clear sentinel test, shared by ``marks/base.py``'s
  ``_is_paint_sentinel`` and ``selection.py``'s ``_hex_to_color_dict`` so
  the two color-parsing surfaces test the identical predicate even though
  they handle a match differently. It matches two invisible spellings —
  ``"none"`` and ``"transparent"`` — kept in sync with Rust's
  ``resolve_paint_color`` (``crates/ferrum-core/src/render/draw.rs``); see
  the function docstring below.

- ``validate_pixel_value`` is the single definition of the "pixel contract"
  (spec §4.7): a value must be numeric (``bool`` excluded), finite, and
  non-negative. It is shared by ``configure.PaddingConfig.__post_init__``
  (construction-time validation) and ``_override_apply.validate``'s padding
  guard (the ``Chart.override`` entry point), so both boundaries refuse the
  same values with the same reasoning instead of maintaining two hand-copied
  isinstance/range checks that can drift.

- ``validate_positive_int`` (#238) is the "wrap-count contract": a value
  must be a genuine ``int`` (``bool`` excluded, same reasoning as
  ``validate_pixel_value``) and ``>= 1``. ``None`` passes as the caller's
  "unset, let inference decide" sentinel, following ``validate_pixel_value``'s
  convention. Shared by ``_Facet.__post_init__`` for both ``ncols`` and
  ``nrows``, which in turn covers both of #238's entries
  (``Chart.facet(ncols=)`` and ``encode(facet=fm.Facet(columns=))``) from
  the single sink they both write.

- ``validate_finite_ordered_pair`` (#92) is the "break-gap contract": both
  ``start`` and ``end`` must be numeric (``bool`` excluded) and finite
  (no NaN/±inf), and ``start`` must be strictly less than ``end``. Shared by
  ``BreakAxis.__post_init__``, which validates every ``(start, end)`` pair in
  ``gap`` (a single pair or a list of pairs) — the numeric-pair analog of
  ``validate_fraction_value``'s single-value contract.

- ``validate_color_value`` (close-remediation, batch F) is the "literal
  color contract": a color-bearing value must be parseable by the caller's
  color parser (``ferrum.color.to_hex`` in every current caller). It
  replaces two independently hand-copied gates that were the same six-line
  ``try: to_hex(value) except ValueError: raise ValueError(<label> +
  str(exc))`` body — ``marks.base._validate_literal_color`` and
  ``configure._validate_config_color`` — differing only in how each
  composes its own label prefix and whether the surface additionally
  requires ``str`` before parsing (``configure.py``'s five config
  dataclasses are typed ``str | None`` and must refuse a non-``str``, e.g.
  an RGB(A) tuple, at construction rather than let it reach the wire and
  fail there naming neither the class nor the field; ``marks/base.py``'s
  caller already restricts to ``str`` before calling, so it leaves
  ``require_str`` at its default). Both callers now compose their own
  label (``f"{owner}.{field}"`` / ``f"mark_{mark_name}: {key}"``) and pass
  it through unchanged, so each surface's exact historical message shape is
  preserved byte-for-byte. The parser itself is passed in by the caller
  (``parse_color=to_hex``) rather than imported here — this module's
  zero-import leaf invariant (below) is not "no import of ``ferrum.color``",
  it is "no import of anything under ``ferrum`` at all", pinned unconditionally
  by ``tests/test_finding_p7.py::test_validate_module_is_a_leaf`` (an AST
  walk that catches a lazy, function-body import exactly as readily as a
  module-level one) — so this function stays a pure predicate over
  caller-supplied inputs instead of reaching for ``ferrum.color`` itself.

Each is a small, self-contained predicate with no shared machinery between
them — this module is a landing spot for that shape of helper, not a
general utility dump; a new addition should be a comparably minimal,
zero-import validation predicate, not accreted business logic.

This is a leaf module: it imports nothing from ``ferrum``, matching the
``_warn``/``_metrics_fmt`` convention, so any part of the package can depend
on it without inverting the dependency graph. ``validate_color_value``
needs a color parser to do its job but keeps the invariant intact by taking
one as a parameter (``parse_color``) instead of importing ``ferrum.color``
itself, even lazily.
"""

from __future__ import annotations

import decimal
import math
import numbers
from typing import Any, Callable


def is_real_number(value: Any) -> bool:
    """Return whether *value* is a real number a numeric parameter accepts.

    Accepts every :class:`numbers.Real` (``int``, ``float``,
    ``fractions.Fraction``, numpy integer and floating scalars) and
    :class:`decimal.Decimal`. Rejects ``bool`` (a truth value is not a
    quantity, though ``bool`` subclasses ``int``) and ``numpy.timedelta64``
    (registered as a numpy integer, but ``float()`` on it raises). ``np.bool_``
    and ``np.datetime64`` are not ``numbers.Real`` and are rejected with
    every other non-number.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, decimal.Decimal):
        return True
    if not isinstance(value, numbers.Real):
        return False
    import numpy as np

    return not isinstance(value, np.timedelta64)


def is_none_color_sentinel(value: str) -> bool:
    """Return ``True`` if *value* is a canonical paint-clear sentinel.

    Matches two invisible spellings — ``"none"`` and ``"transparent"`` —
    trimmed and case-insensitive (``"None"``, ``"NONE"``, ``" none "``,
    ``"Transparent"``, ``" transparent "`` all count), mirroring Rust's
    ``trimmed.eq_ignore_ascii_case("none") || trimmed.eq_ignore_ascii_case("transparent")``
    (``crates/ferrum-core/src/render/draw.rs``'s ``resolve_paint_color``).
    ``"transparent"`` joins ``"none"`` as a clearing spelling (spec §4.1,
    superseded 2026-09-01 T8 quality review): ``transparent`` is a real CSS
    Color 4 keyword, so refusing it while accepting ``"none"`` recreated the
    exact divergence class this batch remediates. Neither spelling enters
    the parser's vocabulary — both are matched here, before
    ``ferrum.color.to_hex`` is ever called.

    This is the single Python definition of the sentinel predicate:
    ``marks/base.py``'s ``_is_paint_sentinel`` (which treats a match as an
    explicit paint-clear) and ``selection.py``'s ``_hex_to_color_dict``
    (which treats a match as a refusal — the selection wire has no
    cleared-paint representation) both compose from it, so the two surfaces
    differ only in how they *handle* a match, never in what counts as one.

    The unrelated ``"theme:label"`` sentinel is not part of this predicate;
    callers that need it check it separately with an exact (untrimmed,
    case-sensitive) string comparison.
    """
    trimmed = value.strip().lower()
    return trimmed in ("none", "transparent")


def validate_choice(
    func_name: str,
    param: str,
    value: Any,
    choices: "frozenset[Any] | set[Any] | tuple[Any, ...] | list[Any]",
) -> None:
    """Raise ``ValueError`` when ``value`` is not in ``choices``.

    Produces a canonical error message across all figure functions and
    dataclass fields::

        {func_name}: {param} must be one of {sorted(choices)}; got {value!r}
    """
    if value not in choices:
        raise ValueError(
            f"{func_name}: {param} must be one of {sorted(str(c) for c in choices)}; got {value!r}"
        )


def validate_optional_choice(
    func_name: str,
    param: str,
    value: Any,
    choices: "frozenset[Any] | set[Any] | tuple[Any, ...] | list[Any]",
) -> None:
    """:func:`validate_choice` for an optional keyword: ``None`` (unset) always passes.

    Any other value, string or not, must be in *choices*, so a malformed value
    is refused at the declaring door (*func_name*) instead of reaching a Rust
    serde type error that names no door. Used with a Rust-published
    vocabulary the caller passes in (``interpolate=``, ``x_estimator=``).
    """
    if value is not None:
        validate_choice(func_name, param, value, choices)


def validate_pixel_value(label: str, value: Any, *, required: bool = False) -> float | None:
    """Raise ``ValueError`` when *value* violates the pixel contract (spec §4.7).

    A valid pixel value is a real number (:func:`is_real_number`: ``bool``
    excluded, numpy scalars accepted), finite (not NaN/±inf), and non-negative.
    ``None`` is the caller's "unset" sentinel and always passes — unless
    *required* is set, for a caller with no optional-unset case for this
    field (e.g. one entry of a fixed-length tuple every entry of which is
    mandatory). ``required=True`` does not add a second message: it simply
    stops the early return, so ``None`` falls into the same
    ``not {type(value).__name__}`` branch every other wrong type already
    raises from, reported as ``not NoneType``.

    *label* is a pre-composed description of the field being checked (e.g.
    ``"padding.top"`` or ``"override 'padding_top'"``) so each call site keeps
    its own message convention; this function owns only the check itself,
    which two independent entry points — ``PaddingConfig.__post_init__`` and
    the ``Chart.override`` padding guard — otherwise had to keep in sync by
    hand.

    Returns
    -------
    float or None
        *value* as a ``float`` (so a numpy scalar reaches the JSON wire as a
        plain number), or ``None`` when unset.

    Raises
    ------
    ValueError
        If *value* is not numeric, is NaN/±inf, or is negative — or, when
        *required* is set, if *value* is ``None``.
    """
    if value is None and not required:
        return
    if not is_real_number(value):
        raise ValueError(
            f"{label} must be a numeric pixel value (int or float), "
            f"not {type(value).__name__}; received {value!r}"
        )
    if not math.isfinite(value):
        raise ValueError(f"{label} must be a finite numeric pixel value; received {value}")
    if value < 0:
        raise ValueError(f"{label} must be non-negative; received {value}")
    return float(value)


def validate_fraction_value(label: str, value: Any, *, required: bool = False) -> float | None:
    """Raise ``ValueError`` when *value* is not a fraction in ``[0, 1]``.

    The bounded-fraction sibling of :func:`validate_pixel_value`, and it lives
    here for the same reason that one does: the numeric-and-finite half of the
    two contracts is identical, so expressing it twice is how the two drift.
    (It was briefly expressed twice — ``configure._normalize_grid_style``
    called ``validate_pixel_value`` for ``width`` and hand-rolled the same
    numeric/finite check for ``opacity`` one line below.)

    A valid fraction is a real number (:func:`is_real_number`), finite, and within ``[0, 1]`` inclusive. ``None`` is the
    caller's "unset" sentinel and always passes — unless *required* is set,
    matching :func:`validate_pixel_value`'s convention: it stops the early
    return rather than adding a second message, so ``None`` is reported by
    the same ``not {type(value).__name__}`` branch every other wrong type
    already raises from.

    *label* is a pre-composed field description, matching
    :func:`validate_pixel_value`'s convention.

    Returns
    -------
    float or None
        *value* as a ``float``, or ``None`` when unset (see
        :func:`validate_pixel_value`).

    Raises
    ------
    ValueError
        If *value* is not numeric, is NaN/±inf, or falls outside ``[0, 1]`` —
        or, when *required* is set, if *value* is ``None``.
    """
    if value is None and not required:
        return
    if not is_real_number(value):
        raise ValueError(
            f"{label} must be a number between 0 and 1, "
            f"not {type(value).__name__}; received {value!r}"
        )
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1; received {value!r}")
    return float(value)


def validate_positive_int(label: str, value: Any) -> None:
    """Raise ``ValueError`` when *value* is not a positive ``int`` (``>= 1``).

    A valid value is a genuine ``int`` (``bool`` excluded, since ``bool`` is
    an ``int`` subclass) and ``>= 1``. ``None`` is the caller's "unset, let
    inference decide" sentinel and always passes, matching
    :func:`validate_pixel_value`'s convention.

    *label* is a pre-composed field description (e.g. ``"_Facet.ncols"``),
    matching :func:`validate_pixel_value`'s convention.

    Raises
    ------
    ValueError
        If *value* is not an ``int``, or is less than 1.
    """
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(
            f"{label} must be a positive int, not {type(value).__name__}; received {value!r}"
        )
    if value < 1:
        raise ValueError(f"{label} must be >= 1; received {value!r}")


def validate_finite_ordered_pair(label: str, start: Any, end: Any) -> None:
    """Raise ``ValueError`` unless ``(start, end)`` is a finite, ordered numeric pair.

    Each of *start*/*end* must be numeric (``int``/``float``, ``bool``
    excluded) and finite (not NaN/±inf); *start* must be strictly less than
    *end*.

    *label* is a pre-composed description of the pair being checked (e.g.
    ``"BreakAxis.gap[0]"``), matching :func:`validate_pixel_value`'s
    convention.

    Raises
    ------
    ValueError
        If either endpoint is not numeric, is NaN/±inf, or if
        ``start >= end``.
    """
    for name, value in (("start", start), ("end", end)):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(
                f"{label}: {name} must be a numeric value (int or float), "
                f"not {type(value).__name__}; received {value!r}"
            )
        if not math.isfinite(value):
            raise ValueError(f"{label}: {name} must be finite; received {value}")
    if start >= end:
        raise ValueError(
            f"{label}: start ({start!r}) must be less than end ({end!r}); got ({start!r}, {end!r})"
        )


def validate_color_value(
    label: str,
    value: Any,
    parse_color: Callable[[Any], Any],
    *,
    require_str: bool = False,
) -> None:
    """Raise ``ValueError`` if *value* is not a color *parse_color* accepts.

    *parse_color* is the caller-supplied color parser -- every current
    caller passes ``ferrum.color.to_hex`` (ferrum's single Rust color
    parser). It is a parameter rather than an import performed here because
    this module is a zero-import leaf (see the module docstring); the
    caller already needs ``to_hex`` for other reasons and simply forwards
    it, so no caller pays a new import to use this function.

    This function is a thin construction-time gate around *parse_color*,
    shared by every literal-color-bearing surface in the package instead of
    each hand-copying the same ``try/except`` body. *label* is a
    pre-composed description of the field being checked -- e.g.
    ``f"{owner}.{field}"`` (``configure.py``'s five config dataclasses) or
    ``f"mark_{mark_name}: {key}"`` (``marks/base.py``'s literal fill/stroke
    gate) -- so each caller keeps its own historical message shape; this
    function owns only the check itself.

    ``None`` always passes (the caller's "unset" sentinel, matching this
    module's other predicates). Non-color sentinels ("none"/"transparent"/
    "theme:label") are NOT special-cased here: each existing caller already
    short-circuits its own sentinel vocabulary before calling this function
    (``marks/base.py``'s literal-color call site checks ``_is_paint_sentinel``
    first; ``configure.py``'s config surfaces have no sentinel vocabulary of
    their own), so a sentinel that reaches this function raises exactly as
    *parse_color* itself would.

    ``require_str=True`` additionally refuses any non-``str`` *value* before
    ever calling *parse_color* -- ``configure.py``'s config-surface fields
    are typed ``str | None``, and without this gate a value ``to_hex``
    happens to accept structurally (e.g. an RGB(A) tuple) would pass
    construction, serialize a tuple onto the wire, and fail there with a
    serde message naming neither the class nor the field (#107 residual).
    ``marks/base.py``'s caller already restricts to ``str`` before calling
    (see its own ``isinstance`` guard), so it leaves this at the default
    ``False``.

    Raises
    ------
    ValueError
        If *value* is not a ``str`` when ``require_str=True``, or if
        ``parse_color(value)`` itself raises ``ValueError``.
    """
    if value is None:
        return
    if require_str and not isinstance(value, str):
        raise ValueError(f"{label}={value!r} must be a color string, not {type(value).__name__}")
    try:
        parse_color(value)
    except ValueError as exc:
        raise ValueError(f"{label}={value!r} is not a valid color ({exc})") from exc
