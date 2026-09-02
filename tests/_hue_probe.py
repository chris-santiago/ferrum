"""Shared SVG probes for the color-channel typing tests.

Three test modules assert the same underlying invariant — that a group
discriminator bound to the ``color`` channel renders a categorical swatch
legend and a discrete palette, never a fabricated 0..1 colorbar — at three
different layers of the stack:

- ``tests/test_diagnostic_class_column_typing.py`` (diagnostic desugars),
- ``tests/marks/test_nominal_color_sweep.py`` (composite-mark desugars),
- ``tests/test_figure_hue_typing.py`` (public figure functions' ``hue=``).

They had independently grown the same swatch-matching regex and the same
render-and-capture-warnings dance. This module is the one definition, so a
change to how ferrum emits a legend swatch updates one regex rather than
three that can silently drift apart — which is the same class of failure the
tests themselves exist to catch.

Named with a leading underscore so pytest does not collect it as a test
module, matching ``tests/_snapshots.py``.
"""

from __future__ import annotations

import re
import warnings
from typing import Any

#: A categorical-legend swatch plus its label. Anchored on the swatch circle
#: (``r="4"``) specifically so it cannot be satisfied by axis tick labels,
#: which would make an "N legend entries" assertion pass on a chart that
#: renders no legend at all.
LEGEND_ENTRY_RE = re.compile(r'<circle[^>]*\br="4"[^>]*/><text[^>]*>([^<]+)</text>')

#: Any explicit ``fill=``/``stroke=`` hex literal in the rendered SVG.
_HEX_PAINT_RE = re.compile(r'(?:fill|stroke)="(#[0-9a-fA-F]{6})"')


def legend_labels(svg: str) -> list[str]:
    """Return the text of every categorical legend entry in *svg*."""
    return LEGEND_ENTRY_RE.findall(svg)


def has_colorbar(svg: str) -> bool:
    """True when *svg* contains a continuous colorbar legend.

    A colorbar over a group discriminator is the headline symptom of an
    untyped color binding: the renderer invents a continuous 0..1 domain for
    a column that names two groups.
    """
    return "linearGradient" in svg or "Gradient" in svg


def paint_colors(svg: str) -> set[str]:
    """Return every hex color painted in *svg*, lowercased.

    Used for the discriminating half of a dtype-parity assertion: two renders
    that agree on legend entry *count* can still disagree on the actual
    palette, which is exactly what a continuous ramp does (it emits two
    colors, they are just the wrong two).
    """
    return {c.lower() for c in _HEX_PAINT_RE.findall(svg)}


def render(chart: Any) -> tuple[str, list[str]]:
    """Render *chart* to SVG, returning ``(svg, user_warning_messages)``.

    Warnings are part of the assertion surface here, not noise: for
    line/ribbon marks an untyped numeric color field trips
    ``UnsupportedColorScaleOnMark``, so "renders correctly AND says nothing"
    is the property under test. On the silent marks (bar, rect, polygon,
    rule, segment, tick) there is no warning either way, which is why the
    palette assertions cannot be replaced by a warning assertion.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = chart.to_svg()
    return svg, [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
