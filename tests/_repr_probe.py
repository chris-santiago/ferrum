"""Argument-boundary matching for repr probes (#195).

A bare substring probe on a repr is vacuous: ``"g"`` is inside the word
``groupby``, and ``Some("g")`` also matches ``name=Some("g")``.
"""

from __future__ import annotations

import re


def has_arg(text: str, frag: str) -> bool:
    """True when ``frag`` appears in ``text`` as one whole argument: preceded by
    ``(`` or ``, `` and followed by ``, `` or ``)``, regardless of argument order."""
    return re.search(r"(?:\(|, )" + re.escape(frag) + r"(?=, |\))", text) is not None
