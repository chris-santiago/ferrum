"""Named built-in themes shipped with ferrum.

Each theme overrides only the keys that differ from
``ThemeInputs::default()``; the Rust side fills in the rest. The default
identity is Paper Ink: warm cream background (#FAF7F2), blue lead marks,
warm-tinted grid, Inter typography. See
``docs/superpowers/specs/2026-05-12-custom-themes-design.md`` for the
authoritative definitions.

Twelve pre-built ``Theme`` instances are exported here and re-exported at
``ferrum.themes.<name>``:

``default``
    All properties at Rust renderer defaults (equivalent to ``Theme()``).
    Paper Ink identity: warm cream bg, blue marks, warm grid.

``paper_ink``
    Explicit Paper Ink — identical to ``default`` but with all properties
    set, useful as a derivation base.

``slate_citrus``
    Dark navy background (``#111827``), vibrant neon accents, lime/cyan
    categorical cycle.

``arctic_signal``
    Cool white background (``#F8FAFC``), precise signal colors, sky blue
    lead mark.

``observable``
    Preserved pre-2026-05-12 default: white background, tableau blue
    marks, neutral gray grid, tableau10 palette.

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
    sequential_scheme="viridis",
    diverging_scheme="rdbu",
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
    sequential_scheme="plasma",
    diverging_scheme="rdbu",
    strip_background_color="#252540",
    reference_line_color="#666666",
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
    sequential_scheme="viridis",
    diverging_scheme="rdbu",
    point_size=24,
    reference_line_color="#999999",
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
    sequential_scheme="blues",
    diverging_scheme="rdbu",
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
    sequential_scheme="viridis",
    diverging_scheme="rdbu",
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
    sequential_scheme="viridis",
    diverging_scheme="rdbu",
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
    sequential_scheme="plasma",
    diverging_scheme="rdbu",
    strip_background_color="#073642",
)

paper_ink = Theme(
    background="#FAF7F2",
    font_color="#1F2937",
    label_color="#6B7280",
    title_color="#1F2937",
    grid_color="#D6D3D1",
    axis_line_color="#6B7280",
    tick_color="#6B7280",
    mark_color="#2563EB",
    color_scheme="paper_ink",
    sequential_scheme="cool_blue",
    diverging_scheme="blue_to_red",
    strip_background_color="#EDE9E3",
    reference_line_color="#9CA3AF",
)

slate_citrus = Theme(
    background="#111827",
    font_color="#E5E7EB",
    label_color="#9CA3AF",
    title_color="#E5E7EB",
    grid_color="#374151",
    axis_line_color="#9CA3AF",
    tick_color="#9CA3AF",
    mark_color="#60A5FA",
    color_scheme="slate_citrus",
    sequential_scheme="night_blue",
    diverging_scheme="cyan_to_amber",
    strip_background_color="#1E293B",
    reference_line_color="#6B7280",
)

arctic_signal = Theme(
    background="#F8FAFC",
    font_color="#0F172A",
    label_color="#64748B",
    title_color="#0F172A",
    grid_color="#CBD5E1",
    axis_line_color="#64748B",
    tick_color="#64748B",
    mark_color="#0284C7",
    color_scheme="arctic_signal",
    sequential_scheme="signal_blue",
    diverging_scheme="blue_to_violet",
    strip_background_color="#E2E8F0",
    reference_line_color="#94A3B8",
)

observable = Theme(
    background="#ffffff",
    font_color="#222222",
    label_color="#555555",
    title_color="#222222",
    grid_color="#DDDDDD",
    axis_line_color="#888888",
    tick_color="#888888",
    mark_color="#4C78A8",
    color_scheme="tableau10",
    sequential_scheme="blues",
    diverging_scheme="rdbu",
    strip_background_color="#F0F0F0",
    reference_line_color="#AAAAAA",
)
