"""Encoding-string shorthand parser.

Supports (per spec §3.2 Channel shorthand strings):
- "fieldname"            → (fieldname, None, None)
- "fieldname:Q"          → (fieldname, "Q", None)
- "agg(fieldname)"       → (fieldname, None, "agg")
- "agg()"                → (None, None, "agg")  (e.g. count())
- "agg(fieldname):Q"     → (fieldname, "Q", "agg")
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

_VALID_TYPES = frozenset(["Q", "N", "O", "T"])
_PATTERN = re.compile(
    r"""
    ^                                       # start
    (?:                                     # optional aggregate prefix:
        (?P<agg>[a-z][a-z0-9_]*)            #   agg name (lowercase identifier)
        \(                                  #   open paren
        (?P<aggfield>[a-zA-Z_][a-zA-Z0-9_]*)?  # optional field inside parens
        \)                                  #   close paren
    )?
    (?(agg)|(?P<field>[a-zA-Z_][a-zA-Z0-9_]*))  # if no agg, require bare field
    (?::(?P<type>[A-Z]))?                   # optional type suffix
    $                                       # end
    """,
    re.VERBOSE,
)


def parse_shorthand(s: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse a shorthand encoding string into (field, type, aggregate).

    Returns (None, type, agg) for "count()".
    Raises ValueError for malformed input or unknown type letters.
    """
    if "(" in s and ")" not in s:
        raise ValueError(f"unbalanced parens in shorthand: {s!r}")
    if ")" in s and "(" not in s:
        raise ValueError(f"unbalanced parens in shorthand: {s!r}")

    m = _PATTERN.match(s)
    if not m:
        raise ValueError(f"could not parse shorthand: {s!r}")

    type_ = m.group("type")
    if type_ is not None and type_ not in _VALID_TYPES:
        raise ValueError(
            f"unknown type {type_!r} in {s!r}; expected one of Q, N, O, T"
        )

    agg = m.group("agg")
    if agg is not None:
        return (m.group("aggfield"), type_, agg)
    return (m.group("field"), type_, None)
