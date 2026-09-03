"""Shared "%b %Y"-style date-tick shape parsing for rendered-chart assertions.

Extracted from ``test_flexibility_campaign.py`` so sibling test modules can
pin the wrap-shape contract of "%b %Y"-formatted x-axis ticks without
importing (or re-implementing) another test module's private names -- the
``tests/_svg_extents.py`` precedent for cross-cutting test helpers.

``axis.rs``'s label-collision cascade may legitimately wrap a single
"Mon YYYY" tick onto two rendered ``<text>`` lines (``"Jan"`` / ``"2020"``)
once the axis needs it (spec §4.6); it must never wrap SOME sibling ticks
of the identical format while leaving others combined (spec-review cycle
3's "the cascade degrades uniformly" requirement -- a ragged one-line/
two-line mix is wrong regardless of which form the axis as a whole
resolves to). ``month_year_tick_shapes``/``reconstruct_month_year_labels``
are the single, canonical parser for that contract; quality-review cycle 1
found two independently-drifted copies (one matching a month token via
membership in ``MONTH_ABBREVS``, the other via a looser
``re.fullmatch(r"[A-Z][a-z]{2}", t)`` that also matches non-month
three-letter capitalized words like ``"Val"``/``"Sum"``) and asked for one
definition.
"""

from __future__ import annotations

import re

MONTH_ABBREVS = {
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
}


def month_year_tick_shapes(tick_labels: list[str]) -> list[int]:
    """For a sequence of "%b %Y"-formatted x-axis tick ``<text>`` contents,
    return the number of physical text nodes each LOGICAL date tick consumed
    -- 1 for a combined ``"Jan 2020"`` node, 2 for a wrapped ``"Jan"`` +
    ``"2020"`` pair.
    """
    shapes: list[int] = []
    i = 0
    while i < len(tick_labels):
        t = tick_labels[i]
        if re.fullmatch(r"[A-Z][a-z]{2} 20\d{2}", t):
            shapes.append(1)
            i += 1
        elif (
            t in MONTH_ABBREVS
            and i + 1 < len(tick_labels)
            and re.fullmatch(r"20\d{2}", tick_labels[i + 1])
        ):
            shapes.append(2)
            i += 2
        else:
            i += 1
    return shapes


def reconstruct_month_year_labels(tick_labels: list[str]) -> list[str]:
    """Recombine "%b %Y"-formatted x-axis ticks into one ``"Mon YYYY"``
    string per date tick, regardless of whether it rendered as a single
    combined node or a wrapped ``"Mon"`` / ``"YYYY"`` pair (see
    ``month_year_tick_shapes``'s doc).
    """
    out: list[str] = []
    i = 0
    while i < len(tick_labels):
        t = tick_labels[i]
        if re.fullmatch(r"[A-Z][a-z]{2} 20\d{2}", t):
            out.append(t)
            i += 1
        elif (
            t in MONTH_ABBREVS
            and i + 1 < len(tick_labels)
            and re.fullmatch(r"20\d{2}", tick_labels[i + 1])
        ):
            out.append(f"{t} {tick_labels[i + 1]}")
            i += 2
        else:
            i += 1
    return out
