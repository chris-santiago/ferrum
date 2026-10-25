"""Independent WCAG 2.x contrast oracle for rendered-SVG assertions.

Not ferrum's formula: the public W3C standard, implemented here once and
shared by every test module that checks a drawn ink's contrast against the
cell or background it sits on (``tests/test_contrast_sentinel.py``,
``tests/test_annotation_contrast.py``), so a contrast assertion is never a
re-derivation of ferrum's own branching logic on the raw input data -- a
test built that way would just restate the implementation as its own
oracle.

Named with a leading underscore so pytest does not collect it as a test
module, matching ``tests/_snapshots.py`` and ``tests/_hue_probe.py``.
"""

from __future__ import annotations

import re

_HEX6 = re.compile(r"^#([0-9a-fA-F]{6})$")
_HEX8 = re.compile(r"^#([0-9a-fA-F]{8})$")
_RGBA = re.compile(r"rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*(?:,\s*([\d.]+)\s*)?\)")


def _srgb_to_linear(channel_0_1: float) -> float:
    return channel_0_1 / 12.92 if channel_0_1 <= 0.04045 else ((channel_0_1 + 0.055) / 1.055) ** 2.4


def relative_luminance(rgb: tuple[int, int, int]) -> float:
    r, g, b = (_srgb_to_linear(c / 255.0) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(rgb1: tuple[int, int, int], rgb2: tuple[int, int, int]) -> float:
    """WCAG 2.x contrast ratio between two opaque sRGB colors."""
    l1, l2 = relative_luminance(rgb1), relative_luminance(rgb2)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def composite_over(
    fg: tuple[int, int, int, float], bg: tuple[int, int, int]
) -> tuple[int, int, int]:
    """Alpha-composite an (r, g, b, alpha 0..1) foreground over an opaque background."""
    r, g, b, a = fg
    br, bg_g, bb = bg
    return (
        round(r * a + br * (1 - a)),
        round(g * a + bg_g * (1 - a)),
        round(b * a + bb * (1 - a)),
    )


def parse_paint(value: str) -> tuple[int, int, int, float]:
    """Parse an SVG fill attribute (hex6/hex8/`rgba()`) into `(r, g, b, alpha 0..1)`."""
    if m := _HEX8.match(value):
        r, g, b, a = (int(m.group(1)[i : i + 2], 16) for i in (0, 2, 4, 6))
        return r, g, b, a / 255.0
    if m := _HEX6.match(value):
        r, g, b = (int(m.group(1)[i : i + 2], 16) for i in (0, 2, 4))
        return r, g, b, 1.0
    if m := _RGBA.match(value):
        r, g, b = (round(float(m.group(i))) for i in (1, 2, 3))
        a = float(m.group(4)) if m.group(4) else 1.0
        return r, g, b, a
    raise ValueError(f"not a recognized SVG paint: {value!r}")


def opaque_rgb(fill: str, background: tuple[int, int, int]) -> tuple[int, int, int]:
    """The color *fill* actually displays as, composited over *background* if translucent."""
    r, g, b, a = parse_paint(fill)
    return (r, g, b) if a >= 1.0 else composite_over((r, g, b, a), background)
