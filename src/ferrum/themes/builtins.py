"""Named built-in themes shipped with ferrum.

Eight pre-built ``Theme`` instances are exported here and re-exported at
``ferrum.themes.<name>``:

``default``
    All properties at Rust renderer defaults (equivalent to ``Theme()``).

``minimal``
    No grid lines, no axis lines, generous padding (20 px).

``dark``
    Dark navy background (``#1a1a2e``), light text, dark2 color scheme.

``publication``
    Print-ready: no background, no grid, Tableau10 palette, bold titles,
    Inter typeface.

``economist``
    Light blue background (``#d3e0e6``), red title accents, no axis lines,
    Set1 palette.

``fivethirtyeight``
    Grey background (``#f0f0f0``), Set1 palette, no axis lines, Inter
    typeface.

``solarized_light``
    Warm cream background (``#fdf6e3``), muted teal text, Set2 palette.

``solarized_dark``
    Dark teal background (``#002b36``), warm-light text, Set2 palette.
"""
from __future__ import annotations

from ferrum.themes import Theme


# Ferrum defaults (all None → Rust ThemeInputs::default())
default = Theme()

# Minimal: no grid, no axis lines, generous padding
minimal = Theme(
    grid=False,
    axis_line=False,
    padding=20,
)

# Dark: low-contrast dark background, light text, dark-friendly palette
dark = Theme(
    background="#1a1a2e",
    font_color="#e6e6e6",
    title_color="#ffffff",
    axis_line_color="#666666",
    grid_color="#333333",
    color_scheme="dark2",
)

# Publication: print-ready, no background, high contrast, Tableau10
publication = Theme(
    background=None,
    grid=False,
    color_scheme="tableau10",
    font_family="Inter",
    title_font_weight="bold",
    axis_line_color="#000000",
    font_color="#000000",
)

# Economist: red accents, light blue background, no axis lines
economist = Theme(
    background="#d3e0e6",
    font_family="Inter",
    title_color="#c00000",
    grid_color="#b0c4cc",
    axis_line=False,
    color_scheme="set1",
)

# FiveThirtyEight-style: grey bg, divergent palette, no axis lines
fivethirtyeight = Theme(
    background="#f0f0f0",
    color_scheme="set1",
    grid_color="#cccccc",
    axis_line=False,
    font_family="Inter",
)

# Solarized light: warm cream bg
solarized_light = Theme(
    background="#fdf6e3",
    font_color="#586e75",
    title_color="#073642",
    grid_color="#eee8d5",
    axis_line_color="#93a1a1",
    color_scheme="set2",
)

# Solarized dark
solarized_dark = Theme(
    background="#002b36",
    font_color="#93a1a1",
    title_color="#fdf6e3",
    grid_color="#073642",
    axis_line_color="#586e75",
    color_scheme="set2",
)
