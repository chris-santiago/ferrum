"""Named built-in themes shipped with ferrum.

Each theme overrides only the keys that differ from
``ThemeInputs::default()``; the Rust side fills in the rest. Themes-T4
(2026-05-11) flipped the default identity to an Observable Plot aesthetic:
tableau blue marks, faint ``#DDD`` grid, Inter typography, left-aligned
semibold titles, 5% scale padding on quantitative axes. See
``ferrum-spec.md`` §3.13 and
``docs/superpowers/specs/2026-05-11-themes-overhaul-design.md`` §3 for the
authoritative definitions.

Eight pre-built ``Theme`` instances are exported here and re-exported at
``ferrum.themes.<name>``:

``default``
    All properties at Rust renderer defaults (equivalent to ``Theme()``).
    Tableau blue marks, faint visible grid, Inter typography, semibold
    left-aligned title, 5% inset on quantitative axes.

``minimal``
    No grid lines, no axis lines, generous padding (24 px).

``dark``
    Dark navy background (``#1a1a2e``), light text, dark2 color scheme.

``publication``
    Print-ready: white background, no grid, black axis strokes, Tableau10
    palette, bold middle-aligned title, Inter typeface.

``economist``
    Light blue background (``#d3e0e6``), red title accents, no axis lines,
    Set1 palette.

``fivethirtyeight``
    Grey background (``#f0f0f0``), Set1 palette, no axis lines, bold
    start-anchored title.

``solarized_light``
    Warm cream background (``#fdf6e3``), muted teal text, Set2 palette.

``solarized_dark``
    Dark teal background (``#002b36``), warm-light text, Set2 palette.
"""
from __future__ import annotations

from ferrum.themes import Theme


default = Theme()

minimal = Theme(
    grid=False,
    axis_line=False,
    tick_size=0,
    padding=24,
    label_color="#888888",
)

dark = Theme(
    background="#1a1a2e",
    font_color="#e6e6e6",
    label_color="#b8b8c8",
    title_color="#ffffff",
    axis_line_color="#555566",
    tick_color="#555566",
    grid_color="#2a2a3e",
    grid_width=0.5,
    mark_color="#7fb3d5",
    color_scheme="dark2",
    strip_background_color="#252540",
)

# Publication: print-ready monochrome. Plan's design spec used DejaVu Serif
# but only Inter is bundled (`crates/ferrum-core/src/render/embed_font.rs`);
# unbundled fonts resolve via system fallback and diverge across CI hosts.
# The Inter + bold-title + black-stroke identity is the publication signal.
publication = Theme(
    background="#ffffff",
    grid=False,
    axis_line_color="#000000",
    axis_line_width=1.0,
    tick_color="#000000",
    font_color="#000000",
    label_color="#000000",
    title_color="#000000",
    title_font_weight="bold",
    title_anchor="middle",
    mark_color="#000000",
    color_scheme="tableau10",
    point_size=24,
)

economist = Theme(
    background="#d3e0e6",
    font_color="#1a1a1a",
    title_color="#c00000",
    title_font_weight="bold",
    title_anchor="start",
    axis_line=False,
    grid_color="#b0c4cc",
    grid_width=0.6,
    mark_color="#005a8c",
    color_scheme="set1",
    strip_background_color="#bfd4dc",
)

fivethirtyeight = Theme(
    background="#f0f0f0",
    font_color="#333333",
    label_color="#555555",
    axis_line=False,
    tick_color="#999999",
    grid_color="#cbcbcb",
    grid_width=1.0,
    mark_color="#fc4f30",
    color_scheme="set1",
    title_font_weight="bold",
    title_anchor="start",
)

solarized_light = Theme(
    background="#fdf6e3",
    font_color="#586e75",
    label_color="#657b83",
    title_color="#073642",
    title_font_weight="bold",
    grid_color="#eee8d5",
    grid_width=0.6,
    axis_line_color="#93a1a1",
    tick_color="#93a1a1",
    mark_color="#268bd2",
    color_scheme="set2",
)

solarized_dark = Theme(
    background="#002b36",
    font_color="#93a1a1",
    label_color="#839496",
    title_color="#fdf6e3",
    title_font_weight="bold",
    grid_color="#073642",
    grid_width=0.6,
    axis_line_color="#586e75",
    tick_color="#586e75",
    mark_color="#268bd2",
    color_scheme="set2",
    strip_background_color="#073642",
)
