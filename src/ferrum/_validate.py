"""Shared zero-import leaf-validation predicates for the rest of ``ferrum``.

Two independent predicates live here today:

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

Both are small, self-contained predicates with no shared machinery between
them — this module is a landing spot for that shape of helper, not a
general utility dump; a new addition should be a comparably minimal,
zero-import validation predicate, not accreted business logic.

This is a leaf module: it imports nothing from ``ferrum``, matching the
``_warn``/``_metrics_fmt`` convention, so any part of the package can depend
on it without inverting the dependency graph.
"""

from __future__ import annotations

from typing import Any


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
