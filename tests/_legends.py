"""Shared legend readers and SVG chrome primitives for rendered-chart assertions.

Owns the one static-SVG legend reader (``legend_set``), its interactive-scene
twin (``interactive_legend_set`` and the scene-dict form it delegates to,
``scene_legend_set``, for a caller that already holds a rendered scene), and
the primitives both need: ``tag``, ``svg_root``, ``clip_groups`` (the
plot-area clip-group walk), and ``chrome_texts`` (every text element drawn
outside a plot-area clip group). Any test module that reads a rendered
chart's legend, or needs to tell a chart's own marks apart from its
axis/legend chrome, imports from here instead of writing its own copy;
``tests/test_layer_encoding_model.py`` and ``tests/test_drawn_channel_rule.py``
both do.

Named with a leading underscore so pytest does not collect it as a test
module, matching ``tests/_wcag.py`` and ``tests/_snapshots.py``.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from typing import Any


def tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def svg_root(svg: str) -> ET.Element:
    return ET.fromstring(svg)


def clip_groups(root: ET.Element) -> list[ET.Element]:
    """Each outermost plot-area clip group, in document order."""
    out: list[ET.Element] = []

    def walk(element: ET.Element) -> None:
        if tag(element) == "g" and element.get("clip-path", "").startswith("url(#ferrum-clip-"):
            out.append(element)
            return
        for child in element:
            walk(child)

    walk(root)
    return out


def chrome_texts(svg: str) -> list[ET.Element]:
    """Every text element drawn outside a plot-area clip group, in document order."""
    root = svg_root(svg)
    inside = {id(e) for group in clip_groups(root) for e in group.iter()}
    return [e for e in root.iter() if tag(e) == "text" and id(e) not in inside]


def legend_set(svg: str) -> dict[str, tuple[str, ...]]:
    """Each legend's title and entries, read off the chrome: a bold start-anchored title, then its entries."""
    legends: dict[str, list[str]] = {}
    title: str | None = None
    title_x = 0.0
    for e in chrome_texts(svg):
        text = "".join(e.itertext()).strip()
        if e.get("text-anchor") != "start" or "transform" in e.attrib:
            continue
        if e.get("font-weight") == "600" and e.get("font-size") != "13":
            title, title_x = text, float(e.get("x", "0"))
            legends.setdefault(title, [])
        elif title is not None and float(e.get("x", "0")) >= title_x:
            legends[title].append(text)
    return {t: tuple(entries) for t, entries in legends.items()}


def scene_legend_set(scene: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    """Each interactive legend's title and its entry labels, in order, from an already-built scene.

    Takes the parsed ``scene_json`` dict directly rather than a chart, so a
    caller that already rendered ``.interactive()`` once (for example inside
    a warnings capture) reads the legend off that same render instead of
    building the scene a second time.
    """
    legends: dict[str, list[str]] = {}
    title: str | None = None
    for node in scene["legend"]:
        if node["type"] != "text":
            continue
        if (node.get("style") or {}).get("font_weight") == {"custom": "600"}:
            content: str = node["content"]
            title = content
            legends[title] = []
        elif title is not None:
            legends[title].append(node["content"])
    return {t: tuple(entries) for t, entries in legends.items()}


def interactive_legend_set(chart: Any) -> dict[str, tuple[str, ...]]:
    """Each interactive legend's title and its entry labels, in order."""
    return scene_legend_set(json.loads(chart.interactive().scene_json))
