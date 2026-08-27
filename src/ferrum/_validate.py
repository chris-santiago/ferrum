"""Shared closed-choice validation for public callables and dataclass fields.

``validate_choice`` is the canonical way to reject a value that is not a
member of a documented closed vocabulary (a ``kind=``/``method=``/``order=``
style keyword, or a dataclass field with a fixed set of legal values). It
replaces the inconsistent idioms that predate it — different message
templates, different punctuation, unsorted set reprs — with one message
shape across the whole package.

``func_name`` convention: pass the public callable name the caller invoked
(e.g. ``"pairplot"``), or ``"Class.attr"`` for a dataclass field validated in
``__post_init__`` (e.g. ``"BreakAxis.axis"``).

This is a leaf module: it imports nothing from ``ferrum``, matching the
``_warn``/``_metrics_fmt`` convention, so any part of the package can depend
on it without inverting the dependency graph.
"""

from __future__ import annotations

from typing import Any


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
