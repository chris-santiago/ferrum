#!/usr/bin/env python
"""Generate the split API reference pages under docs/site/api/.

Each page renders ``::: ferrum`` with an explicit ``members:`` list so the
anchors are the canonical top-level ``ferrum.X`` ids (e.g. ``ferrum.hconcat``)
that the docs' ``[ferrum.X]`` autorefs reference. This replaces the old
paradigm where one monolithic ``ferrum.md`` (``::: ferrum``) owned every
``ferrum.X`` anchor and every cross-reference resolved to that single
(too-large-to-render) page.

Symbols are partitioned across pages by their *defining* module
(``obj.__module__``), so an incidental import (e.g. ``Chart`` imported into the
``annotations`` module) is assigned to its real home, not wherever it is
imported. Run this whenever the public API changes:

    uv run --no-sync python scripts/gen_api_pages.py        # write pages
    uv run --no-sync python scripts/gen_api_pages.py --check # report only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import ferrum as fm

API_DIR = Path(__file__).resolve().parent.parent / "docs" / "site" / "api"

# ── Page metadata: page-slug -> (title, one-line description) ────────────────
# Order here is the nav order for the visible API section.
PAGES: dict[str, tuple[str, str]] = {
    "chart": ("Chart", "The `Chart` class — declaration, encoding, theming, and rendering."),
    "marks": ("Marks", "All `.mark_*()` methods on `Chart` — the visual primitives."),
    "encoding": ("Encoding", "Typed encoding channels — `X`, `Y`, `Color`, `Size`, `Tooltip`, and more."),
    "plots": ("Plots", "Figure-level helpers — `displot`, `catplot`, `lmplot`, `pairplot`, and the `*_chart` diagnostics."),
    "visualizers": ("Visualizers", "sklearn-protocol `.fit().show()` diagnostic visualizer classes."),
    "model_sources": ("Model Sources", "`ModelSource` / `ComparedModelSource` — fitted-model adapters that feed the diagnostics."),
    "themes": ("Themes", "Built-in themes and `Theme` / `set_default_theme` / `theme_context`."),
    "selection": ("Selection", "`selection_point`, `selection_interval`, conditional encodings (`when` / `value`)."),
    "parameters": ("Parameters", "Reactive parameters — `param`, `Parameter`, `VariableParameter`."),
    "composition": ("Composition", "`HConcatChart`, `VConcatChart`, `LayerChart`, `hconcat`, `vconcat`, `concat`."),
    "annotations": ("Annotations", "`annotate_*` helpers, label classes, and annotation coordinates."),
    "transforms": ("Transforms", "`transform_filter`, `transform_aggregate`, `transform_calculate`, and friends."),
    "statistics": ("Statistics", "Statistical transform value-objects — `Kde`, `Smooth`, `Bin`, `Violin`, `Summary`, etc."),
    "scales": ("Scales", "`Scale` and its subclasses — `LinearScale`, `LogScale`, `OrdinalScale`, etc."),
    "schemes": ("Schemes", "Color scheme helpers — `Gradient`, `continuous_palette`."),
    "coord": ("Coord", "Coordinate systems — Cartesian, flip, polar, geo, fixed."),
    "position": ("Position", "Position adjustments — `Dodge`, `Stack`, `Jitter`, `Identity`."),
    "layer": ("Layer", "`Layer` and the `layer()` helper for explicit layering."),
    "repeat": ("Repeat", "`Repeat` for faceted repeat patterns."),
    "structural": ("Structural", "Structural view modifiers — `BreakAxis`, `Inset`, `SecondaryY`."),
    "axis": ("Axis", "`Axis` value class for axis configuration."),
    "grid": ("Grid", "`Grid` value class for gridline configuration."),
    "legend": ("Legend", "`Legend` value class for legend configuration."),
    "title": ("Title", "`Title` value class for figure titles."),
    "render_config": ("Render Config", "`RenderConfig` for auto-raster, scale, and output tuning."),
    "configure": ("Configure", "Chart-level configuration dataclasses — Configure, AxisConfig, ColorConfig, GridConfig, LegendConfig, PaddingConfig, TitleConfig — plus format-preset resolution (resolve_format) and the Chart.override error (FerrumOverrideError)."),
    "specs": ("Specs", "Low-level serialized specs — `ChartSpec`, `EncodingSpec`."),
    "rendering": ("Rendering", "Low-level rendering entry points — `render_svg`, `render_png`, layout/compose helpers."),
}

# Pages kept out of the visible nav (low-level; retained so autorefs resolve).
HIDDEN_PAGES = {"specs", "rendering"}

# ── Module-prefix -> page rules (longest prefix wins) ────────────────────────
MODULE_RULES: list[tuple[str, str]] = [
    ("ferrum.diagnostics.visualizers", "visualizers"),
    ("ferrum.diagnostics.source", "model_sources"),
    ("ferrum.diagnostics.sources", "model_sources"),
    ("ferrum.encoding", "encoding"),
    ("ferrum.plots", "plots"),
    ("ferrum.transforms", "transforms"),
    ("ferrum.themes", "themes"),
    ("ferrum.selection", "selection"),
    ("ferrum.parameter", "parameters"),
    ("ferrum.composition", "composition"),
    ("ferrum.annotations", "annotations"),
    ("ferrum.annotation", "annotations"),
    ("ferrum._metric_labels", "annotations"),
    ("ferrum.coord", "coord"),
    ("ferrum.position", "position"),
    ("ferrum.axis", "axis"),
    ("ferrum.legend", "legend"),
    ("ferrum.title", "title"),
    ("ferrum.render_config", "render_config"),
    ("ferrum.repeat", "repeat"),
    ("ferrum.layer", "layer"),
    ("ferrum.schemes", "schemes"),
    # ferrum.configure.* (Configure/AxisConfig/…), ferrum.exceptions
    # (FerrumOverrideError), and ferrum.format_presets (resolve_format) are
    # homed on the "configure" page below. This is distinct from the
    # `ferrum.config` namespace (the contextvars runtime-defaults store),
    # which keeps its own hand-written config.md.
    ("ferrum.configure", "configure"),
    ("ferrum.exceptions", "configure"),
    ("ferrum.format_presets", "configure"),
    ("ferrum.structural", "structural"),
    ("ferrum.grid", "grid"),
    ("ferrum.chart", "chart"),
]

# Explicit per-name overrides (edge cases / re-exports under unhelpful modules).
NAME_OVERRIDES: dict[str, str] = {
    "hconcat": "composition", "vconcat": "composition", "concat": "composition",
    "layer": "layer",
    "selection_single": "selection", "selection_multi": "selection",
    "Scale": "scales",
}

# _core is a grab-bag; split it by name.
_CORE_SCALES = {
    "BandScale", "BinOrdinalScale", "ContinuousScheme", "DivergingScale", "LinearScale",
    "LogScale", "OrdinalScale", "PointScale", "PowScale", "QuantileScale", "QuantizeScale",
    "SequentialScale", "SqrtScale", "SymlogScale", "ThresholdScale", "TimeScale", "Scale",
}
_CORE_SCHEMES = {"Gradient"}
_CORE_SPECS = {"ChartSpec", "EncodingSpec"}
_CORE_RENDERING = {
    "compose_svg_grid", "compose_svg_horizontal", "compose_svg_vertical",
    "compute_layout", "process_batch", "render_png", "render_svg",
}


def page_for(name: str) -> str:
    if name in NAME_OVERRIDES:
        return NAME_OVERRIDES[name]
    obj = getattr(fm, name, None)
    mod = getattr(obj, "__module__", None) or getattr(type(obj), "__module__", "")
    if mod == "ferrum._core":
        if name in _CORE_SCALES:
            return "scales"
        if name in _CORE_SCHEMES:
            return "schemes"
        if name in _CORE_SPECS:
            return "specs"
        if name in _CORE_RENDERING:
            return "rendering"
        return "statistics"
    for prefix, page in MODULE_RULES:
        if mod.startswith(prefix):
            return page
    return "UNHOMED"


def build_partition() -> tuple[dict[str, list[str]], list[str]]:
    names = list(getattr(fm, "__all__", None) or [n for n in dir(fm) if not n.startswith("_")])
    by_page: dict[str, list[str]] = {}
    unhomed: list[str] = []
    for n in sorted(names):
        obj = getattr(fm, n, None)
        # Skip module-namespace re-exports (e.g. ferrum.color, ferrum.config) — those
        # keep their own submodule-rendered pages and are handled separately.
        if type(obj).__name__ == "module":
            continue
        p = page_for(n)
        if p == "UNHOMED":
            unhomed.append(n)
            continue
        by_page.setdefault(p, []).append(n)
    return by_page, unhomed


def render_block(members: list[str], filters: list[str]) -> str:
    lines = ["::: ferrum", "    options:", "      members:"]
    lines += [f"        - {m}" for m in members]
    lines += [
        "      show_root_heading: false",
        "      show_root_toc_entry: false",
        f"      filters: {filters!r}".replace("'", '"'),
        "      members_order: source",
    ]
    return "\n".join(lines) + "\n"


def marks_index() -> str:
    """Prose + autoref links to Chart.mark_* methods (which live on chart.md).

    marks.md does NOT render the Chart class itself — that would duplicate the
    ``ferrum.Chart`` root anchor with chart.md and make ``[ferrum.Chart]``
    resolve ambiguously. Instead it links into chart.md's rendered methods.
    """
    marks = sorted(n for n in dir(fm.Chart) if n.startswith("mark_") and not n.startswith("__"))
    rows = "\n".join(f"| [`{m}`][ferrum.Chart.{m}] |" for m in marks)
    return (
        "# Marks\n\n"
        "Marks are the visual primitives of a chart — each is a `.mark_*()` method on "
        "`Chart`. Full signatures are on the [Chart](chart.md) page; this is a quick index.\n\n"
        "| Mark |\n|---|\n" + rows + "\n"
    )


def page_body(slug: str, members: list[str]) -> str:
    title, desc = PAGES[slug]
    if slug == "marks":
        return marks_index()
    if slug == "chart":
        # Full Chart (incl. mark_* methods) so chart.md is the sole owner of
        # every ferrum.Chart.* anchor.
        block = render_block(["Chart"], ["!^_"])
    else:
        block = render_block(members, ["!^_"])
    return f"# {title}\n\n{desc}\n\n{block}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report partition, write nothing")
    args = ap.parse_args()

    by_page, unhomed = build_partition()
    # chart + marks both draw from Chart; ensure both slugs exist.
    by_page.setdefault("chart", ["Chart"])
    by_page.setdefault("marks", ["Chart"])

    unknown = set(by_page) - set(PAGES) - {"model_sources"}
    if unknown:
        print(f"WARNING: pages with no PAGES metadata: {sorted(unknown)}", file=sys.stderr)

    for slug in PAGES:
        members = sorted(by_page.get(slug, []))
        if not members and slug not in {"chart", "marks"}:
            print(f"WARNING: page '{slug}' has no members", file=sys.stderr)
        if args.check:
            print(f"{slug} ({len(members)}): {members}")
            continue
        (API_DIR / f"{slug}.md").write_text(page_body(slug, members))

    if unhomed:
        print(f"UNHOMED public symbols (no page): {unhomed}", file=sys.stderr)
    if not args.check:
        print(f"Wrote {len(PAGES)} API pages to {API_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
