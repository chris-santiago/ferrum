"""#121: ``pr_chart(iso_lines=True)`` keeps the iso-F curves' declared grey.

Pre-fix, ``desugar_pr``'s iso-line layer bound its grouping key
(``_iso_f``) to the ``color`` *encoding* channel. ``build_color_scale``
resolves one color scale per ``ChartSpec`` from the chart-level encoding,
which for ``pr_chart`` is keyed on ``class`` -- so every ``_iso_f`` lookup
missed that scale's domain, and the line mark's stroke-resolution fallback
on a miss painted the layer with ``paint.fill`` (the theme fill at the
layer's ``opacity=0.6``) instead of the layer's own literal
``mark_kwargs={"stroke": "#AAAAAA", ...}``. All four iso curves rendered
identically in theme blue at 60% opacity.

The fix (``desugar_pr`` in
``ferrum.marks.diagnostic._classification``) groups the iso-line layer by
``mark_kwargs.detail`` instead of the ``color`` channel -- the same
pattern ``desugar_parallel_coordinates`` uses to group polylines without
claiming a paint channel -- so the layer's literal ``stroke`` survives.

These assertions extract the rendered ``stroke`` VALUES from the SVG
(never just element counts or SVG-was-produced), so they fail on the
pre-fix code and pass only once the grey is actually painted.
"""

from __future__ import annotations

import re

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

import ferrum as fm

_ISO_STROKE_RGBA = "rgba(170,170,170,0.600)"  # #AAAAAA at the layer's opacity=0.6
_PR_LINE_STROKE = "#2563eb"  # theme fill, the color the iso curves wrongly inherited


def _fit_binary_model():
    X, y = make_classification(
        n_samples=300,
        n_classes=2,
        n_informative=4,
        random_state=0,
    )
    model = LogisticRegression(max_iter=300).fit(X, y)
    return model, X, y


def test_pr_chart_iso_lines_render_declared_grey_dashed():
    model, X, y = _fit_binary_model()
    svg = fm.pr_chart(model, X, y, iso_lines=True).to_svg()

    iso_polylines = re.findall(r'<polyline[^>]*stroke-dasharray="4,4"[^>]*/>', svg)
    assert len(iso_polylines) == 4, f"expected 4 dashed iso-F polylines, got {len(iso_polylines)}"
    strokes = [re.search(r'stroke="([^"]+)"', p).group(1) for p in iso_polylines]
    assert strokes == [_ISO_STROKE_RGBA] * 4, (
        f"expected all four iso curves to render the declared grey "
        f"({_ISO_STROKE_RGBA}), got {strokes}"
    )

    for f in ("F=0.2", "F=0.4", "F=0.6", "F=0.8"):
        assert f in svg, f"iso label {f!r} missing from SVG"


def test_pr_chart_iso_lines_leave_curve_baseline_and_legend_unchanged():
    model, X, y = _fit_binary_model()
    svg = fm.pr_chart(model, X, y, iso_lines=True).to_svg()

    # The PR curve itself: one solid (non-dashed) polyline in the theme's
    # data color, untouched by the iso-line fix.
    solid_polylines = [
        p for p in re.findall(r"<polyline[^>]*/>", svg) if "stroke-dasharray" not in p
    ]
    assert len(solid_polylines) == 1, (
        f"expected exactly one solid PR-curve polyline, got {len(solid_polylines)}"
    )
    assert f'stroke="{_PR_LINE_STROKE}"' in solid_polylines[0]

    # The binary-prevalence baseline rule: one grey dashed <line>, unaffected
    # by the iso-line layer's own grouping change.
    baseline_lines = re.findall(r'<line[^>]*stroke-dasharray="4,4"[^>]*/>', svg)
    assert len(baseline_lines) == 1, (
        f"expected exactly one dashed baseline <line>, got {len(baseline_lines)}"
    )
    assert 'stroke="#aaaaaa"' in baseline_lines[0]

    # Legend: the class-keyed color scale still produces a legend titled
    # "class" -- the iso-line layer no longer participates in that scale
    # at all, so the legend content is unaffected by the fix.
    assert ">class<" in svg


def test_pr_chart_iso_lines_and_ap_annotation_compose():
    model, X, y = _fit_binary_model()
    svg = fm.pr_chart(model, X, y, annotate_ap=True, iso_lines=True).to_svg()

    iso_polylines = re.findall(r'<polyline[^>]*stroke-dasharray="4,4"[^>]*/>', svg)
    strokes = [re.search(r'stroke="([^"]+)"', p).group(1) for p in iso_polylines]
    assert strokes == [_ISO_STROKE_RGBA] * 4
    assert "AP = 0." in svg
    assert "F=0.6" in svg
