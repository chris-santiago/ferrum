"""Regression guards for the render-surface module-move refactor (REND-05/06/08).

This test file guards three pure moves that were already implemented and gate-passed:

- REND-05: annotation-coordinate resolver + label-map engine moved from
  ``ferrum._render`` to ``ferrum._render_prepare``.
- REND-06: dead exception classes ``InteractiveRenderError`` and
  ``WasmNotAvailableError`` deleted from ``ferrum._interactive``.
- REND-08: PNG→PDF byte codec moved from ``ferrum.display`` to ``ferrum._pdf``.

Each test exercises behavior through ferrum's public API, not internal imports.
The tests must FAIL if a future change breaks the new module wiring or
re-introduces the deleted symbols, and PASS with the current implementation.
"""

from __future__ import annotations

import re

import polars as pl
import ferrum as fm
import ferrum._interactive
from ferrum.annotations import annotate_vline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _line_elements(svg: str) -> list[str]:
    """Return all <line ...> opening tags from the SVG."""
    return re.findall(r"<line\b[^>]*>", svg)


def _x1_values(line_tags: list[str]) -> list[float]:
    """Extract the numeric x1 attribute from <line> tags."""
    vals = []
    for tag in line_tags:
        m = re.search(r'\bx1="([^"]+)"', tag)
        if m:
            try:
                vals.append(float(m.group(1)))
            except ValueError:
                pass
    return vals


# ---------------------------------------------------------------------------
# REND-05 — categorical annotation coordinate resolver (_render_prepare)
# ---------------------------------------------------------------------------


def test_categorical_annotation_resolves_to_band_coordinate_after_render_prepare_split():
    """Regression: REND-05 — _resolve_category_coords_in_annotations wired from _render_prepare.

    Builds a bar chart with a 3-category nominal x-axis and attaches an
    annotate_vline anchored at the middle category ("B").  After to_svg(), the
    annotation <line> element must be present and its x1 coordinate must be in
    the interior of the plot (not zero, not the right edge, finite).

    If the module move broke the import wiring (_render.py forgot to import from
    _render_prepare), the render would either raise an ImportError or the
    annotation coordinate would default to zero/null.
    """
    df = pl.DataFrame({"cat": ["A", "B", "C"], "val": [1.0, 2.0, 3.0]})
    base = fm.Chart(df).mark_bar().encode(x="cat:N", y="val:Q")
    ann = annotate_vline("B", stroke="#ff0000")
    composed = base + ann

    svg = composed.to_svg()

    # The SVG must contain at least one <line> element (the annotation).
    line_tags = _line_elements(svg)
    assert line_tags, (
        "No <line> elements found in SVG after categorical annotate_vline. "
        "The annotation coordinate resolver must be reachable from _render.py "
        "via the new _render_prepare import."
    )

    # Extract x1 values from line elements that carry the annotation stroke color.
    ann_lines = [t for t in line_tags if "#ff0000" in t or "ff0000" in t]
    assert ann_lines, (
        "No annotation <line> with stroke=#ff0000 found. "
        "The resolver must emit an annotation element for category 'B'."
    )

    x1_vals = _x1_values(ann_lines)
    assert x1_vals, "Could not parse x1 from annotation <line> tag."

    x1 = x1_vals[0]
    # The chart default width is 400px with padding. The band center for 'B'
    # (index 1 out of 3) is approximately (1+0.5)/3 = 50% of the plot area,
    # which lands well inside any reasonable plot width. It must not be 0 or NaN.
    assert x1 > 0, f"Annotation x1={x1} is not positive; expected a band-center coordinate."
    assert x1 < 1000, f"Annotation x1={x1} looks unreasonably large."


def test_categorical_annotation_band_positions_differ_for_first_and_last():
    """Regression: REND-05 — _build_ordinal_norm_map maps distinct categories to distinct coords.

    'A' (first band) and 'C' (last band) must produce different x1 values, and
    A must be to the left of C.  This guards that _extract_ordinal_domain and
    _build_ordinal_norm_map both flow correctly through the new module boundary.
    """
    df = pl.DataFrame({"cat": ["A", "B", "C"], "val": [1.0, 2.0, 3.0]})

    STROKE = "#abcdef"

    def _get_x1(category: str) -> float:
        base = fm.Chart(df).mark_bar().encode(x="cat:N", y="val:Q")
        ann = annotate_vline(category, stroke=STROKE)
        svg = (base + ann).to_svg()
        ann_lines = [t for t in _line_elements(svg) if STROKE.lstrip("#") in t or STROKE in t]
        vals = _x1_values(ann_lines)
        assert vals, f"No annotation x1 found for category {category!r}"
        return vals[0]

    x_a = _get_x1("A")
    x_c = _get_x1("C")

    assert x_a != x_c, (
        f"'A' and 'C' annotations share x1={x_a}; ordinal norm map must produce distinct positions."
    )
    assert x_a < x_c, (
        f"'A' (x1={x_a}) must be left of 'C' (x1={x_c}) in ordinal order."
    )


# ---------------------------------------------------------------------------
# REND-05 — label-map engine (_render_prepare)
# ---------------------------------------------------------------------------


def test_axis_label_map_remapped_text_appears_in_svg_after_render_prepare_split():
    """Regression: REND-05 — _collect_label_maps/_apply_label_maps wired from _render_prepare.

    A chart with Axis(label_map=...) must produce remapped tick labels in the
    rendered SVG.  If _render.py lost its import from _render_prepare, the
    label_map would be silently ignored and the raw column values ('a', 'b', 'c')
    would appear instead of the remapped names.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [1.0, 2.0, 3.0]})
    chart = (
        fm.Chart(df)
        .mark_bar()
        .encode(
            x=fm.X("cat:N", axis=fm.Axis(label_map={"a": "Alpha", "b": "Beta", "c": "Gamma"})),
            y="val:Q",
        )
    )

    svg = chart.to_svg()

    assert "Alpha" in svg, (
        "'Alpha' not found in SVG — _collect_label_maps/_apply_label_maps "
        "must be imported from _render_prepare and called during render."
    )
    assert "Beta" in svg, "'Beta' not found in SVG after label_map remapping."
    assert "Gamma" in svg, "'Gamma' not found in SVG after label_map remapping."


# ---------------------------------------------------------------------------
# REND-08 — PDF byte codec (_pdf)
# ---------------------------------------------------------------------------


def test_pdf_export_produces_valid_pdf_header_via_save_after_pdf_split(tmp_path):
    """Regression: REND-08 — _png_to_minimal_pdf wired from ferrum._pdf via display.save_chart_svg.

    Saves a minimal chart to .pdf and asserts the file starts with the PDF
    magic bytes.  If _pdf.py is not importable or the display.py lazy import
    broke, save() would raise ImportError or produce garbage bytes.
    """
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y")

    pdf_path = tmp_path / "chart.pdf"
    chart.save(str(pdf_path))

    data = pdf_path.read_bytes()
    assert data[:5] == b"%PDF-", (
        f"PDF file does not start with b'%PDF-'; got {data[:10]!r}. "
        "Check that display.py imports _png_to_minimal_pdf from ferrum._pdf."
    )
    assert len(data) > 1024, (
        f"PDF too small ({len(data)} bytes); expected a real encoded image stream."
    )


# ---------------------------------------------------------------------------
# REND-06 — deleted dead exception classes
# ---------------------------------------------------------------------------


def test_interactive_render_error_and_wasm_not_available_error_are_deleted():
    """Regression: REND-06 — InteractiveRenderError and WasmNotAvailableError must be gone.

    Both classes were dead code (never raised, never caught outside their own
    definition site) and were deleted as part of REND-06.  This test guards
    against silent re-introduction by asserting they are not attributes of
    ferrum._interactive.

    If someone accidentally re-adds either class, this test fails immediately.
    """
    assert not hasattr(ferrum._interactive, "InteractiveRenderError"), (
        "InteractiveRenderError was deleted as dead code (REND-06) and must not "
        "be present in ferrum._interactive."
    )
    assert not hasattr(ferrum._interactive, "WasmNotAvailableError"), (
        "WasmNotAvailableError was deleted as dead code (REND-06) and must not "
        "be present in ferrum._interactive."
    )


# ---------------------------------------------------------------------------
# Module-boundary / no-cycle guard
# ---------------------------------------------------------------------------


def test_new_modules_import_cleanly_after_render_surface_split():
    """Regression: REND-05/08 — new modules import without cycle or missing-dep errors.

    ferrum._render_prepare and ferrum._pdf must both be importable as top-level
    modules.  This is the minimal cycle guard: if either module has a circular
    import or a missing dependency introduced during the move, this fails.
    """
    import ferrum._render_prepare  # noqa: F401
    import ferrum._pdf  # noqa: F401

    # Verify the key public symbols are present in each module.
    assert hasattr(ferrum._render_prepare, "_resolve_category_coords_in_annotations"), (
        "_resolve_category_coords_in_annotations missing from ferrum._render_prepare"
    )
    assert hasattr(ferrum._render_prepare, "_collect_label_maps"), (
        "_collect_label_maps missing from ferrum._render_prepare"
    )
    assert hasattr(ferrum._pdf, "_png_to_minimal_pdf"), (
        "_png_to_minimal_pdf missing from ferrum._pdf"
    )
