"""Mechanical oracle queries run over the committed golden SVG corpus.

Three kinds of test live here:

* **Corpus queries** — every oracle in ``_svg_corpus.QUERIES`` run over every
  committed golden. A golden byte-diff says output MOVED; these say output is
  WRONG, and they name the file and the defect.
* **Detector self-tests** — every detector needs a test that it can detect.
  Each query gets a synthetic positive it must flag and a synthetic negative it
  must not, so a query that has quietly gone blind (a renamed attribute, a
  regex that stopped matching, an empty corpus) fails loudly instead of
  reporting a clean zero.
* **Geometry derivations** — the transform, viewport and path-data arithmetic
  the oracles stand on, derived here from SVG 1.1 (§7 coordinate systems, §8
  path data, §7.8 preserveAspectRatio) with hand-computed expected values.
  ``tests/_svg_corpus.py`` was adopted from an untrusted branch; these are the
  independent check on its geometry, written against the specification rather
  than against the adopted implementation.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from tests import _svg_corpus as corpus


def _doc(body: str, *, tmp_path, width: float = 100.0, height: float = 80.0, view_box=None):
    box = view_box if view_box is not None else f"0 0 {width} {height}"
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="{box}">{body}</svg>'
    )
    path = tmp_path / "synthetic.svg"
    path.write_text(svg)
    return corpus.parse_golden(path)


# ---------------------------------------------------------------------------
# the corpus itself
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus_documents() -> list[corpus.Document]:
    return [corpus.parse_golden(p) for p in corpus.golden_paths()]


@pytest.fixture(scope="module")
def corpus_findings(corpus_documents) -> dict[str, list[corpus.Finding]]:
    out: dict[str, list[corpus.Finding]] = {name: [] for name in corpus.QUERIES}
    for doc in corpus_documents:
        for name, fn in corpus.QUERIES.items():
            out[name].extend(fn(doc))
    return out


def test_run_all_keys_every_registered_query_even_with_no_hits():
    """``run_all``'s contract: an empty list means "ran and found nothing", not
    "never ran". Exercised on a one-file subset so the whole corpus is parsed
    exactly once, by the ``corpus_documents`` fixture."""
    result = corpus.run_all(corpus.golden_paths()[:1])
    assert set(result) == set(corpus.QUERIES)


def test_golden_corpus_is_not_empty():
    """A query over an empty corpus reports a clean zero for every property.

    ``golden_paths()`` walks two hard-coded roots; if either moves, every
    corpus assertion below would pass by having nothing to look at. This is the
    "the subject becomes EMPTY" failure, and it is silent without this.
    """
    assert len(corpus.golden_paths()) >= 100


def test_every_corpus_query_ran(corpus_findings):
    """A zero is only meaningful if the run proves it executed."""
    assert set(corpus_findings) == set(corpus.QUERIES)
    assert len(corpus.QUERIES) == 7


def test_corpus_gives_every_oracle_something_to_look_at(corpus_documents):
    """Non-vacuity: each oracle's subject population is non-empty in the corpus.

    ``test_every_corpus_query_ran`` proves the functions were called; this
    proves they were called on documents that actually contain the element
    kinds they judge, so a clean result is a measurement and not an accident of
    the corpus holding no such elements.

    Asserted over the WHOLE corpus with non-emptiness thresholds. A sampled
    slice and population-size thresholds would both be hostage to alphabetical
    path order in a growing corpus, turning this guard into a spurious red.
    """
    nodes = [n for doc in corpus_documents for n in doc.nodes]
    assert any(n.tag in corpus._DRAWABLE and not n.in_defs for n in nodes)
    assert any(n.tag in ("rect", "circle") and not n.in_defs for n in nodes)
    assert any(n.tag in corpus._CHROME_TAGS and not n.in_defs for n in nodes)
    assert any(n.clip_ref is not None for n in nodes)
    assert any(doc.clip_rects for doc in corpus_documents)
    assert any(doc.id_counts for doc in corpus_documents)
    assert any("transform" in n.attrib for n in nodes)
    assert any(n.tag == "path" and n.attrib.get("d") for n in nodes)
    assert any(n.tag == "text" and n.text for n in nodes)


def test_corpus_has_no_negative_geometry(corpus_findings):
    """No golden carries a negative ``width``/``height``/``r``.

    SVG 1.1 makes a negative value an error; renderers disagree on whether to
    drop the element or clamp it, so the same golden renders differently in two
    viewers while byte-comparing equal in both.
    """
    assert corpus.unexplained(corpus_findings["negative_size"]) == []


def test_corpus_has_no_nonfinite_attribute_values(corpus_findings):
    """No golden carries NaN/inf in a numeric attribute value."""
    assert corpus.unexplained(corpus_findings["nan_inf_attr"]) == []


def test_corpus_has_no_zero_size_filled_marks(corpus_findings):
    """No golden emits a filled rect/circle of zero extent.

    A computed, serialized, invisible mark: it cost a scale lookup, a layout
    slot and bytes on the wire, and renders as nothing.
    """
    assert corpus.unexplained(corpus_findings["zero_size_filled"]) == []


def test_corpus_has_no_duplicate_ids(corpus_findings):
    """No golden defines one ``id`` twice.

    ``url(#id)`` resolves to the FIRST match in document order, so a duplicate
    silently cross-wires clip paths, colorbar gradients and legend clips. The
    document stays well-formed and renders; it is just wrong. The single
    authority for keeping these disjoint is ``render/svg.rs::uniquify_clip_ids``
    (Rust layer); a producer that embeds a pre-rendered body without routing
    through it reappears here.
    """
    assert corpus.unexplained(corpus_findings["duplicate_id"]) == []


def test_corpus_off_viewport_is_only_the_break_sentinel(corpus_findings):
    """Nothing is drawn outside the viewBox except at ``BREAK_HIDDEN``.

    Stated as a PROPERTY, not as a per-file allow-list: a break-axis chart
    hides a mark inside the gap by parking it at
    ``render/scene_build.rs::BREAK_HIDDEN`` (Rust layer), and that is the only
    sanctioned reason for an off-canvas coordinate. Any other one is an element
    the user paid bytes for and cannot see.
    """
    assert corpus.unexplained(corpus_findings["off_viewport"]) == []


def test_corpus_has_nothing_clipped_out_of_existence(corpus_findings):
    """Nothing references a clip that removes it entirely, except at the sentinel.

    A mark clipped out of existence is present in the DOM and invisible, so a
    census of silent drops cannot count it — at the point of the drop, nothing
    dropped.
    """
    assert corpus.unexplained(corpus_findings["outside_clip"]) == []


def test_corpus_paints_no_element_twice(corpus_findings):
    """No golden paints one ``line``/``text`` twice at the same resolved place.

    This is the permanent pin on the shared-axis chrome dedup (finding P2): a
    composition that lays panels over one plot rect and paints each panel's
    axes emits coincident chrome, which is pixel-identical to the correct
    render and therefore invisible to every byte-equality golden. Data marks
    are excluded by the oracle — coincident data is legal overplot.
    """
    assert corpus.unexplained(corpus_findings["duplicate_drawable"]) == []


# ---------------------------------------------------------------------------
# detector self-tests: each oracle flags a planted defect and clears a clean doc
# ---------------------------------------------------------------------------


def test_off_viewport_flags_a_planted_defect_and_clears_a_clean_doc(tmp_path):
    doc = _doc(
        '<text x="10" y="200">below</text><text x="10" y="20">inside</text>', tmp_path=tmp_path
    )
    hits = corpus.query_off_viewport(doc)
    assert [h.tag for h in hits] == ["text"]
    assert "y=[200.0,200.0]" in hits[0].detail
    clean = _doc('<text x="10" y="20">inside</text>', tmp_path=tmp_path)
    assert corpus.query_off_viewport(clean) == []


def test_break_sentinel_matches_the_rust_constant_it_mirrors():
    """``BREAK_HIDDEN`` is a cross-language contract, so pin the literal.

    The Python value is only an allow-list-free explanation of an off-canvas
    coordinate as long as it is the value the renderer actually parks marks at.
    """
    rust = (
        Path(__file__).resolve().parent.parent
        / "crates"
        / "ferrum-core"
        / "src"
        / "render"
        / "scene_build.rs"
    ).read_text()
    assert corpus.BREAK_HIDDEN == -99999.0
    assert "const BREAK_HIDDEN: f64 = -99999.0;" in rust


def test_off_viewport_respects_the_break_sentinel(tmp_path):
    doc = _doc('<text x="10" y="-99999">hidden</text>', tmp_path=tmp_path)
    hits = corpus.query_off_viewport(doc)
    assert len(hits) == 1 and hits[0].sentinel
    assert corpus.unexplained(hits) == []
    # A different off-canvas coordinate is NOT explained away.
    other = _doc('<text x="10" y="-4242">stray</text>', tmp_path=tmp_path)
    assert len(corpus.unexplained(corpus.query_off_viewport(other))) == 1


def test_off_viewport_honors_a_non_zero_root_viewbox_origin(tmp_path):
    """SVG 1.1 §7.7: ``viewBox="min-x min-y w h"`` — the visible window starts
    at the origin, it is not ``0,0..w,h``.

    Dropping the origin is wrong in BOTH directions, so both are asserted: the
    element at (250,250) is plainly inside ``100 100 200 200`` and must stay
    clean, and the one at (50,50) is genuinely outside and must be flagged.
    Mirrors ``test_nested_viewport_offsets_a_viewbox_that_does_not_start_at_the_origin``,
    which already modelled the origin for INSET viewports.
    """
    doc = _doc(
        '<circle cx="250" cy="250" r="2"/><circle cx="50" cy="50" r="2"/>',
        tmp_path=tmp_path,
        view_box="100 100 200 200",
    )
    assert doc.viewport == (100.0, 100.0, 300.0, 300.0)
    hits = corpus.query_off_viewport(doc)
    assert len(hits) == 1
    assert "x=[48.0,52.0]" in hits[0].detail
    assert "outside 100,100..300,300" in hits[0].detail


def test_off_viewport_falls_back_to_width_height_without_a_viewbox(tmp_path):
    path = tmp_path / "no_viewbox.svg"
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="60" height="40">'
        '<circle cx="10" cy="10" r="2"/><circle cx="200" cy="10" r="2"/></svg>'
    )
    doc = corpus.parse_golden(path)
    assert doc.viewport == (0.0, 0.0, 60.0, 40.0)
    assert len(corpus.query_off_viewport(doc)) == 1


def test_off_viewport_ignores_elements_inside_defs(tmp_path):
    """Template geometry in ``<defs>`` is never painted where it is written."""
    doc = _doc(
        '<defs><rect x="900" y="900" width="4" height="4"/></defs>',
        tmp_path=tmp_path,
    )
    assert corpus.query_off_viewport(doc) == []


def test_negative_size_flags_a_planted_defect_and_clears_a_clean_doc(tmp_path):
    doc = _doc(
        '<rect x="1" y="1" width="-5" height="4"/><rect x="1" y="1" width="5" height="4"/>',
        tmp_path=tmp_path,
    )
    hits = corpus.query_negative_size(doc)
    assert len(hits) == 1 and hits[0].detail == "width=-5"
    clean = _doc(
        '<rect x="1" y="1" width="5" height="4"/><circle cx="2" cy="2" r="1"/>', tmp_path=tmp_path
    )
    assert corpus.query_negative_size(clean) == []


def test_nan_inf_flags_a_planted_defect_and_clears_a_clean_doc(tmp_path):
    doc = _doc(
        '<circle cx="NaN" cy="10" r="2"/><line x1="0" y1="0" x2="-inf" y2="1"/>'
        '<path d="M0 0 L Infinity 3"/>',
        tmp_path=tmp_path,
    )
    keys = {h.detail.split("=", 1)[0] for h in corpus.query_nan_inf(doc)}
    assert keys == {"cx", "x2", "d"}
    clean = _doc('<circle cx="1" cy="10" r="2"/><path d="M0 0 L1 3"/>', tmp_path=tmp_path)
    assert corpus.query_nan_inf(clean) == []


def test_nan_inf_is_anchored_on_attribute_values_not_a_file_scan():
    """The anchor is the ATTRIBUTE VALUE, never a substring scan of the file.

    Every golden embeds a base64 font body containing the letters ``nan`` and
    ``inf``; a substring scan of the file reports a false positive on every
    file in the corpus. This test fails if the query is ever loosened back to a
    file-level scan.
    """
    paths = corpus.golden_paths()
    naive = [p for p in paths if "nan" in p.read_text().lower()]
    assert len(naive) == len(paths), "expected every golden's font body to contain 'nan'"
    for path in paths[:5]:
        assert corpus.query_nan_inf(corpus.parse_golden(path)) == []


def test_nonfinite_token_regex_ignores_nan_inside_an_alphanumeric_run():
    """The second half of the anti-false-positive defence, after the attribute
    anchor: ``nan``/``inf`` count only as whole tokens.

    The string below is lifted from a golden's base64 font body. Without the
    boundary assertions the regex matches it, and every attribute in the corpus
    that ever carried a base64 payload would report a defect.
    """
    blob = "AdQcuak4aAwDhAmgAhQDnanAAhQDuAnUAhQDbAns"
    assert corpus._NONFINITE_RE.search(blob) is None
    assert corpus._NONFINITE_RE.search("M 0 NaN") is not None
    assert corpus._NONFINITE_RE.search("-inf") is not None
    assert corpus._NONFINITE_RE.search("1 Infinity") is not None


def test_zero_size_filled_flags_a_planted_defect_and_clears_a_clean_doc(tmp_path):
    doc = _doc(
        '<rect x="1" y="1" width="0" height="4" fill="#123456"/>'
        '<rect x="1" y="1" width="0" height="4" fill="none"/>'
        '<circle cx="5" cy="5" r="0" fill="#abcdef"/>',
        tmp_path=tmp_path,
    )
    hits = corpus.query_zero_size_filled(doc)
    assert sorted(h.tag for h in hits) == ["circle", "rect"]
    clean = _doc(
        '<rect x="1" y="1" width="3" height="4" fill="#123456"/>'
        '<circle cx="5" cy="5" r="2" fill="#abcdef"/>',
        tmp_path=tmp_path,
    )
    assert corpus.query_zero_size_filled(clean) == []


def test_duplicate_ids_flags_a_planted_defect_and_clears_a_clean_doc(tmp_path):
    doc = _doc(
        '<defs><clipPath id="c"><rect x="0" y="0" width="10" height="10"/></clipPath>'
        '<clipPath id="c"><rect x="50" y="50" width="10" height="10"/></clipPath></defs>',
        tmp_path=tmp_path,
    )
    hits = corpus.query_duplicate_ids(doc)
    assert len(hits) == 1 and "'c' appears 2 times" in hits[0].detail
    clean = _doc(
        '<defs><clipPath id="c"><rect x="0" y="0" width="10" height="10"/></clipPath>'
        '<clipPath id="d"><rect x="50" y="50" width="10" height="10"/></clipPath></defs>',
        tmp_path=tmp_path,
    )
    assert corpus.query_duplicate_ids(clean) == []


def test_outside_clip_flags_a_planted_defect_and_clears_a_clean_doc(tmp_path):
    doc = _doc(
        '<defs><clipPath id="c"><rect x="0" y="0" width="20" height="20"/></clipPath></defs>'
        '<g clip-path="url(#c)"><circle cx="60" cy="60" r="2" fill="#000"/>'
        '<circle cx="5" cy="5" r="2" fill="#000"/></g>',
        tmp_path=tmp_path,
    )
    hits = corpus.query_outside_clip(doc)
    assert len(hits) == 1
    assert "outside clip #c" in hits[0].detail
    clean = _doc(
        '<defs><clipPath id="c"><rect x="0" y="0" width="20" height="20"/></clipPath></defs>'
        '<g clip-path="url(#c)"><circle cx="5" cy="5" r="2" fill="#000"/></g>',
        tmp_path=tmp_path,
    )
    assert corpus.query_outside_clip(clean) == []


def test_outside_clip_maps_the_clip_rect_through_the_referencing_transform(tmp_path):
    """``clipPathUnits`` defaults to ``userSpaceOnUse``.

    The clip rect is written in the user space of the element that references
    it, so it moves with that element's transform. A detector comparing the raw
    rect against root-space geometry reports the translated-but-clipped case as
    a defect.
    """
    body = (
        '<defs><clipPath id="c"><rect x="0" y="0" width="20" height="20"/></clipPath></defs>'
        '<g transform="translate(60 40)" clip-path="url(#c)">'
        '<circle cx="5" cy="5" r="2" fill="#000"/></g>'
    )
    assert corpus.query_outside_clip(_doc(body, tmp_path=tmp_path)) == []


def test_outside_clip_models_the_renderers_first_match_for_a_duplicate_id(tmp_path):
    """The query must model the RENDERER, which takes the first definition.

    A detector that took the last definition would report the cross-wired
    element as correctly clipped — passing for exactly the wrong reason.
    """
    doc = _doc(
        '<defs><clipPath id="c"><rect x="0" y="0" width="10" height="10"/></clipPath>'
        '<clipPath id="c"><rect x="40" y="40" width="30" height="30"/></clipPath></defs>'
        '<g clip-path="url(#c)"><circle cx="50" cy="50" r="2" fill="#000"/></g>',
        tmp_path=tmp_path,
    )
    assert doc.clip_rects["c"] == (0.0, 0.0, 10.0, 10.0)
    hits = corpus.query_outside_clip(doc)
    assert len(hits) == 1 and hits[0].tag == "circle"


def test_outside_clip_flags_a_dangling_clip_reference(tmp_path):
    doc = _doc('<g clip-path="url(#missing)"><circle cx="5" cy="5" r="2"/></g>', tmp_path=tmp_path)
    hits = corpus.query_outside_clip(doc)
    assert len(hits) == 1 and "undefined" in hits[0].detail


def test_outside_clip_skips_a_clip_shape_it_cannot_model(tmp_path):
    """A non-rect clip is unmodelled, and unmodelled is not "undefined".

    Reporting a path-shaped clipPath as a dangling reference would be a false
    positive wearing a real finding's clothes.
    """
    doc = _doc(
        '<defs><clipPath id="c"><path d="M0 0 L20 0 L20 20 Z"/></clipPath></defs>'
        '<g clip-path="url(#c)"><circle cx="5" cy="5" r="2"/></g>',
        tmp_path=tmp_path,
    )
    assert "c" in doc.clip_path_ids
    assert "c" not in doc.clip_rects
    assert corpus.query_outside_clip(doc) == []


@pytest.mark.parametrize(
    "clip_body",
    [
        pytest.param(
            '<clipPath id="c"><rect x="0" y="0" width="20" height="20" '
            'transform="translate(60 60)"/></clipPath>',
            id="transform-on-the-rect",
        ),
        pytest.param(
            '<clipPath id="c"><g transform="translate(60 60)">'
            '<rect x="0" y="0" width="20" height="20"/></g></clipPath>',
            id="transform-on-an-intervening-g",
        ),
        pytest.param(
            '<clipPath id="c" transform="translate(60 60)">'
            '<rect x="0" y="0" width="20" height="20"/></clipPath>',
            id="transform-on-the-clipPath",
        ),
        pytest.param(
            '<clipPath id="c" clipPathUnits="objectBoundingBox">'
            '<rect x="0" y="0" width="1" height="1"/></clipPath>',
            id="objectBoundingBox-units",
        ),
        pytest.param(
            '<clipPath id="c"><rect x="0" y="0" width="20" height="20"/>'
            '<rect x="40" y="40" width="20" height="20"/></clipPath>',
            id="two-rects",
        ),
    ],
)
def test_outside_clip_declines_to_model_a_clip_it_would_have_to_guess_at(clip_body, tmp_path):
    """Unmodelled, not approximated — the rule R6 established for clip shapes.

    Each clip below would resolve to a DIFFERENT window than its raw ``rect``
    attributes suggest. Reading the raw attributes anyway puts the circle
    outside a window that does not exist, i.e. a false positive shaped exactly
    like a real finding.
    """
    doc = _doc(
        f"<defs>{clip_body}</defs>"
        '<g clip-path="url(#c)"><circle cx="70" cy="70" r="2" fill="#000"/></g>',
        tmp_path=tmp_path,
    )
    assert "c" in doc.clip_path_ids
    assert "c" not in doc.clip_rects
    assert corpus.query_outside_clip(doc) == []


def test_outside_clip_still_models_an_untransformed_rect_under_a_transformed_ancestor(tmp_path):
    """The disqualifier is a transform INSIDE the clipPath, not one above it.

    Nested-viewport insets put the whole ``<defs>`` under a scaling ancestor;
    disqualifying on the accumulated root-space ctm would silently drop clip
    coverage for every inset in the corpus.
    """
    doc = _doc(
        '<g transform="translate(5 5)">'
        '<defs><clipPath id="c"><rect x="0" y="0" width="20" height="20"/></clipPath></defs>'
        '<g clip-path="url(#c)"><circle cx="70" cy="70" r="2" fill="#000"/></g></g>',
        tmp_path=tmp_path,
    )
    assert doc.clip_rects["c"] == (0.0, 0.0, 20.0, 20.0)
    assert len(corpus.query_outside_clip(doc)) == 1


def test_outside_clip_judges_geometry_only_for_painted_content(tmp_path):
    """A clipped template inside ``<defs>`` is judged where it is used.

    Both scopes asserted together: the geometry judgment skips defs (matching
    ``off_viewport``/``zero_size_filled``), while a dangling reference is a
    document-level defect reported wherever it appears.
    """
    out_of_clip = _doc(
        '<defs><clipPath id="c"><rect x="0" y="0" width="20" height="20"/></clipPath>'
        '<g clip-path="url(#c)"><circle cx="70" cy="70" r="2" fill="#000"/></g>'
        '<circle clip-path="url(#c)" cx="70" cy="70" r="2" fill="#000"/></defs>',
        tmp_path=tmp_path,
    )
    assert corpus.query_outside_clip(out_of_clip) == []
    dangling = _doc(
        '<defs><g clip-path="url(#missing)"><circle cx="5" cy="5" r="2"/></g></defs>',
        tmp_path=tmp_path,
    )
    hits = corpus.query_outside_clip(dangling)
    assert len(hits) == 1 and "undefined" in hits[0].detail


def test_duplicate_drawable_flags_a_planted_defect_and_clears_a_clean_doc(tmp_path):
    doc = _doc(
        '<line x1="5" y1="5" x2="90" y2="5" stroke="#111"/>'
        '<line x1="5" y1="5" x2="90" y2="5" stroke="#111"/>'
        '<line x1="5" y1="9" x2="90" y2="9" stroke="#111"/>',
        tmp_path=tmp_path,
    )
    hits = corpus.query_duplicate_drawable(doc)
    assert len(hits) == 1 and hits[0].tag == "line"
    assert "2 coincident <line>" in hits[0].detail
    clean = _doc(
        '<line x1="5" y1="5" x2="90" y2="5" stroke="#111"/>'
        '<line x1="5" y1="9" x2="90" y2="9" stroke="#111"/>',
        tmp_path=tmp_path,
    )
    assert corpus.query_duplicate_drawable(clean) == []


def test_duplicate_drawable_catches_coincidence_only_visible_after_transforms(tmp_path):
    """Two panels painting one axis rarely agree on raw attributes.

    The pair below writes different ``x1``/``x2`` and a different transform,
    and lands on the same resolved coordinates. A detector keyed on the raw
    attribute text sees two distinct elements and reports clean.
    """
    doc = _doc(
        '<line x1="5" y1="5" x2="45" y2="5" stroke="#111"/>'
        '<g transform="translate(-5 0)">'
        '<line x1="10" y1="5" x2="50" y2="5" stroke="#111"/></g>',
        tmp_path=tmp_path,
    )
    hits = corpus.query_duplicate_drawable(doc)
    assert len(hits) == 1 and hits[0].tag == "line"


def test_duplicate_drawable_separates_text_by_content_fill_and_stroke(tmp_path):
    """Same place, different identity: three legal pairs, none a duplicate."""
    doc = _doc(
        '<text x="5" y="5" fill="#111">a</text>'
        '<text x="5" y="5" fill="#111">b</text>'
        '<text x="5" y="5" fill="#222">a</text>'
        '<text x="5" y="5" fill="#111" stroke="#333">a</text>',
        tmp_path=tmp_path,
    )
    assert corpus.query_duplicate_drawable(doc) == []


def test_duplicate_drawable_ignores_data_marks_and_defs_and_the_sentinel(tmp_path):
    """Coincident data is legal; template and parked geometry are not painted."""
    doc = _doc(
        '<circle cx="5" cy="5" r="2" fill="#111"/><circle cx="5" cy="5" r="2" fill="#111"/>'
        '<rect x="1" y="1" width="2" height="2" fill="#111"/>'
        '<rect x="1" y="1" width="2" height="2" fill="#111"/>'
        '<defs><line x1="5" y1="5" x2="9" y2="5" stroke="#111"/>'
        '<line x1="5" y1="5" x2="9" y2="5" stroke="#111"/></defs>'
        f'<line x1="{corpus.BREAK_HIDDEN:g}" y1="5" x2="9" y2="5" stroke="#111"/>'
        f'<line x1="{corpus.BREAK_HIDDEN:g}" y1="5" x2="9" y2="5" stroke="#111"/>',
        tmp_path=tmp_path,
    )
    assert corpus.query_duplicate_drawable(doc) == []


# ---------------------------------------------------------------------------
# geometry derivations (SVG 1.1, hand-computed)
# ---------------------------------------------------------------------------


def _close(pt, expected, tol=1e-9):
    return math.isclose(pt[0], expected[0], abs_tol=tol) and math.isclose(
        pt[1], expected[1], abs_tol=tol
    )


def test_matrix_apply_uses_svgs_a_c_e_over_b_d_f_column_order():
    """SVG 1.1 §7.4: ``matrix(a b c d e f)`` maps ``(ax+cy+e, bx+dy+f)``."""
    m = corpus.parse_transform("matrix(2 3 4 5 6 7)")
    assert _close(m.apply(1.0, 1.0), (2 * 1 + 4 * 1 + 6, 3 * 1 + 5 * 1 + 7))
    assert _close(m.apply(0.0, 0.0), (6.0, 7.0))


def test_transform_list_applies_right_to_left():
    """SVG 1.1 §7.4: a transform list post-multiplies, so the LAST listed
    transform is applied to the point FIRST.

    ``translate(10 0) scale(2)`` sends (1,1) to (12,2). The reversed reading
    sends it to (22,2), so this discriminates the composition order.
    """
    m = corpus.parse_transform("translate(10 0) scale(2)")
    assert _close(m.apply(1.0, 1.0), (12.0, 2.0))
    reversed_order = corpus.parse_transform("scale(2) translate(10 0)")
    assert _close(reversed_order.apply(1.0, 1.0), (22.0, 2.0))


def test_rotate_about_a_centre_moves_points_that_are_not_the_centre():
    """SVG 1.1 §7.4: ``rotate(a cx cy)`` == translate(cx,cy) rotate(a) translate(-cx,-cy).

    With y down, ``rotate(90)`` sends (x,y) to (-y,x). Rotating (20,10) about
    (10,10): offset (10,0) -> (0,10) -> (10,20). A detector that rotated about
    the origin would report (-10,20).
    """
    m = corpus.parse_transform("rotate(90 10 10)")
    assert _close(m.apply(20.0, 10.0), (10.0, 20.0))
    assert _close(m.apply(10.0, 10.0), (10.0, 10.0))


def test_rotate_minus_90_matches_the_corpus_y_axis_title_convention():
    """The corpus writes ``rotate(-90 cx cy)`` on y-axis titles.

    ``rotate(-90)`` sends (x,y) to (y,-x), so a point ``d`` to the right of the
    centre lands ``d`` above it.
    """
    m = corpus.parse_transform("rotate(-90 23.865 229.345)")
    assert _close(m.apply(23.865 + 7.0, 229.345), (23.865, 229.345 - 7.0))


def test_rotate_without_a_centre_rotates_about_the_origin():
    m = corpus.parse_transform("rotate(90)")
    assert _close(m.apply(20.0, 10.0), (-10.0, 20.0))


def test_skew_transforms_are_modelled_not_silently_dropped():
    """``skewX(a)`` is ``matrix(1 0 tan(a) 1 0 0)``, ``skewY(a)`` ``matrix(1 tan(a) 0 1 0 0)``."""
    assert _close(corpus.parse_transform("skewX(45)").apply(1.0, 1.0), (2.0, 1.0))
    assert _close(corpus.parse_transform("skewY(45)").apply(1.0, 1.0), (1.0, 2.0))


def test_an_unmodelled_transform_function_raises_rather_than_becoming_identity():
    """Silently ignoring a transform turns a real finding into a clean zero."""
    with pytest.raises(corpus.SvgCorpusError, match="unmodelled transform"):
        corpus.parse_transform("shear(3)")
    with pytest.raises(corpus.SvgCorpusError, match="matrix"):
        corpus.parse_transform("matrix(1 2 3)")


def test_nested_viewport_fits_a_non_uniform_box_with_xmidymid_meet():
    """SVG 1.1 §7.8: ``preserveAspectRatio`` defaults to ``xMidYMid meet``.

    Numbers taken from the ``configure/structural_inset`` golden, which insets
    a 640x480 viewBox into a 197.560x141.736 box at (398.218, 56.496).
    ``meet`` uses the SMALLER scale and centres the slack:

        s  = min(197.560/640, 141.736/480) = min(0.308688, 0.295283) = 0.2952833
        tx = (197.560 - 640*s)/2 = (197.560 - 188.98133)/2 = 4.2893333
        ty = (141.736 - 480*s)/2 = 0.0

    A per-axis stretch would put the viewBox corner at x = 398.218 + 197.560 =
    595.778 instead of 591.489, so this discriminates ``meet`` from a stretch.
    """
    m = corpus._nested_viewport_matrix(
        {
            "x": "398.218",
            "y": "56.496",
            "width": "197.560",
            "height": "141.736",
            "viewBox": "0 0 640 480",
        }
    )
    s = min(197.560 / 640, 141.736 / 480)
    tx = (197.560 - 640 * s) / 2.0
    assert math.isclose(s, 0.2952833333, abs_tol=1e-9)
    assert math.isclose(tx, 4.2893333333, abs_tol=1e-9)
    assert _close(m.apply(0.0, 0.0), (398.218 + tx, 56.496), tol=1e-6)
    assert _close(m.apply(640.0, 480.0), (398.218 + tx + 640 * s, 56.496 + 480 * s), tol=1e-6)
    assert not _close(m.apply(640.0, 480.0), (398.218 + 197.560, 56.496 + 141.736), tol=1e-3)


def test_nested_viewport_offsets_a_viewbox_that_does_not_start_at_the_origin():
    m = corpus._nested_viewport_matrix(
        {"x": "10", "y": "20", "width": "50", "height": "50", "viewBox": "5 5 50 50"}
    )
    assert _close(m.apply(5.0, 5.0), (10.0, 20.0))
    assert _close(m.apply(55.0, 55.0), (60.0, 70.0))


def test_nested_viewport_without_a_viewbox_is_a_plain_translate():
    m = corpus._nested_viewport_matrix({"x": "10", "y": "20", "width": "50", "height": "50"})
    assert _close(m.apply(3.0, 4.0), (13.0, 24.0))


def test_nested_viewport_composes_under_the_ancestor_transform(tmp_path):
    """A nested viewport's scale must survive the walk to root space."""
    doc = _doc(
        '<g transform="translate(5 5)">'
        '<svg x="10" y="10" width="50" height="40" viewBox="0 0 100 80">'
        '<circle cx="100" cy="80" r="1"/></svg></g>',
        tmp_path=tmp_path,
    )
    circle = next(n for n in doc.nodes if n.tag == "circle")
    # s = min(50/100, 40/80) = 0.5, no slack; (100,80) -> (50,40) -> +10,+10 -> +5,+5
    assert _close(circle.ctm.apply(100.0, 80.0), (65.0, 55.0))


def test_path_absolute_moveto_lineto_horizontal_vertical_and_close():
    """SVG 1.1 §8.3. ``Z`` returns the current point to the last moveto."""
    pts = corpus.path_points("M10 10 L20 20 H40 V50 Z L5 5")
    assert pts == [(10.0, 10.0), (20.0, 20.0), (40.0, 20.0), (40.0, 50.0), (5.0, 5.0)]


def test_path_relative_commands_offset_from_the_current_point():
    pts = corpus.path_points("M10 10 l5 5 h10 v-3")
    assert pts == [(10.0, 10.0), (15.0, 15.0), (25.0, 15.0), (25.0, 12.0)]


def test_path_close_restores_the_subpath_start_for_the_next_relative_command():
    pts = corpus.path_points("M10 10 l5 5 Z l1 1")
    assert pts == [(10.0, 10.0), (15.0, 15.0), (11.0, 11.0)]


def test_path_extra_moveto_pairs_are_an_implicit_lineto():
    """SVG 1.1 §8.3.2: additional coordinate pairs after ``M`` are treated as
    ``L``; after ``m`` they are treated as ``l``.

    The trailing ``Z l1 1`` is what makes this discriminating. Treating the
    extra pairs as repeated MOVETOs yields the same point list but moves the
    subpath start, so only the closepath reveals the difference.
    """
    assert corpus.path_points("M10 10 20 20 30 30") == [
        (10.0, 10.0),
        (20.0, 20.0),
        (30.0, 30.0),
    ]
    assert corpus.path_points("M10 10 20 20 Z l1 1") == [
        (10.0, 10.0),
        (20.0, 20.0),
        (11.0, 11.0),
    ]
    assert corpus.path_points("m10 10 5 5 5 5") == [(10.0, 10.0), (15.0, 15.0), (20.0, 20.0)]
    assert corpus.path_points("m10 10 5 5 Z l1 1") == [(10.0, 10.0), (15.0, 15.0), (11.0, 11.0)]


def test_path_curve_control_points_are_all_collected_and_repeat_implicitly():
    """A relative cubic measures every control point from the point the
    command started at, and a bare second argument set repeats the command."""
    pts = corpus.path_points("M0 0 c1 2 3 4 5 6 1 2 3 4 5 6")
    assert pts == [
        (0.0, 0.0),
        (1.0, 2.0),
        (3.0, 4.0),
        (5.0, 6.0),
        (6.0, 8.0),
        (8.0, 10.0),
        (10.0, 12.0),
    ]


def test_path_quadratic_and_smooth_commands_take_two_pairs():
    assert corpus.path_points("M0 0 Q1 2 3 4") == [(0.0, 0.0), (1.0, 2.0), (3.0, 4.0)]
    assert corpus.path_points("M0 0 S1 2 3 4") == [(0.0, 0.0), (1.0, 2.0), (3.0, 4.0)]
    assert corpus.path_points("M0 0 T5 6") == [(0.0, 0.0), (5.0, 6.0)]


def test_path_numbers_may_be_separated_by_a_sign_or_a_second_decimal_point():
    """SVG 1.1 §8.3.1 allows ``M10-5`` and ``M1.5.5`` — the sign and the second
    decimal point are themselves the separator."""
    assert corpus.path_points("M10-5L20-10") == [(10.0, -5.0), (20.0, -10.0)]
    assert corpus.path_points("M1.5.5") == [(1.5, 0.5)]
    assert corpus.path_points("M1e2 2e-1") == [(100.0, 0.2)]


def test_path_arc_flags_are_single_digits_not_part_of_the_next_number():
    """SVG 1.1 §8.3.8: ``large-arc-flag`` and ``sweep-flag`` are one character
    each and need no separator before the endpoint.

    ``a5 5 0 1110 0`` is flags 1,1 then endpoint (10,0). A tokenizer that lexed
    numbers greedily would read ``1110`` as a single value and land the
    endpoint somewhere else entirely.
    """
    assert corpus.path_points("M0 0 a5 5 0 1 1 10 0") == [(0.0, 0.0), (10.0, 0.0)]
    assert corpus.path_points("M0 0 a5 5 0 1110 0") == [(0.0, 0.0), (10.0, 0.0)]
    assert corpus.path_points("M0 0 A5 5 0 0 1 10 20") == [(0.0, 0.0), (10.0, 20.0)]


def test_malformed_path_data_raises_rather_than_truncating_silently():
    """A truncated parse reports a shorter, still-inside bounding set — a clean
    zero for a defective path."""
    with pytest.raises(corpus.SvgCorpusError):
        corpus.path_points("M10 10 L20")
    with pytest.raises(corpus.SvgCorpusError, match="begins with a number"):
        corpus.path_points("10 10 L20 20")
    with pytest.raises(corpus.SvgCorpusError, match="arc flag"):
        corpus.path_points("M0 0 a5 5 0 3 1 10 0")


def test_a_number_after_closepath_raises_instead_of_looping_forever():
    """SVG 1.1 §8.3.3: ``Z``/``z`` takes no arguments.

    ``Z`` is the one command that consumes no coordinates, so a number
    following it leaves the scan with nothing to consume. Reporting it is the
    only terminating answer: a silent truncation is a false clean zero, and a
    spin is worse than either — inside ``pytest -n auto`` a hang produces no
    message, no traceback and no failing test name.
    """
    with pytest.raises(corpus.SvgCorpusError, match="closepath"):
        corpus.path_points("M0 0 Z 5 5")
    with pytest.raises(corpus.SvgCorpusError, match="closepath"):
        corpus.path_points("M0 0 z5 5")


def test_every_closepath_shaped_malformation_terminates():
    """Termination stated as a property, not as one known-bad input.

    Each of these puts the scan in the state that used to spin. They need not
    agree on a message; they must all TERMINATE, and terminate by reporting.
    (``path_points`` also carries a cursor-progress guard as a backstop for a
    future non-consuming branch; it is unreachable given the branches that
    exist today, so no test can observe it.)
    """
    for d in ("M0 0 Z 5 5", "M0 0 z 1", "M0 0 Z Z 5", "M0 0 L1 1 Z 2 2 Z"):
        with pytest.raises(corpus.SvgCorpusError):
            corpus.path_points(d)


def test_local_points_bound_each_drawable_shape():
    rect = corpus.local_points("rect", {"x": "1", "y": "2", "width": "3", "height": "4"})
    assert min(p[0] for p in rect) == 1.0 and max(p[0] for p in rect) == 4.0
    assert min(p[1] for p in rect) == 2.0 and max(p[1] for p in rect) == 6.0
    assert corpus.local_points("circle", {"cx": "5", "cy": "5", "r": "2"}) == [
        (3.0, 3.0),
        (7.0, 7.0),
    ]
    assert corpus.local_points("ellipse", {"cx": "5", "cy": "5", "rx": "2", "ry": "4"}) == [
        (3.0, 1.0),
        (7.0, 9.0),
    ]
    assert corpus.local_points("polyline", {"points": "1,2 3,4"}) == [(1.0, 2.0), (3.0, 4.0)]
    assert corpus.local_points("line", {"x1": "1", "y1": "2", "x2": "3", "y2": "4"}) == [
        (1.0, 2.0),
        (3.0, 4.0),
    ]


def test_node_text_covers_tspan_children_but_not_the_style_font_blob(tmp_path):
    """``Node.text`` is identity for the duplicate-drawable key, so it must see
    a ``<text>``'s whole content — and must never absorb the embedded font."""
    doc = _doc(
        '<style>.f{}</style><text x="1" y="2">a<tspan>b</tspan>c</text>',
        tmp_path=tmp_path,
    )
    text = next(n for n in doc.nodes if n.tag == "text")
    assert text.text == "abc"
    assert next(n for n in doc.nodes if n.tag == "style").text == ""
    assert next(n for n in doc.nodes if n.tag == "svg").text == ""


def test_nodes_are_flattened_in_document_order_with_matching_indices(tmp_path):
    doc = _doc(
        '<g><rect x="1" y="1" width="1" height="1"/></g><circle cx="1" cy="1" r="1"/>',
        tmp_path=tmp_path,
    )
    assert [n.tag for n in doc.nodes] == ["svg", "g", "rect", "circle"]
    assert [n.index for n in doc.nodes] == [0, 1, 2, 3]


def test_subtree_lookup_is_by_index_not_by_value(tmp_path):
    """Two byte-identical clipped groups must resolve to their OWN children.

    A clip governs its subtree, not the rest of the document. The adopted
    module carries ``Node.index`` so the subtree slice is an O(1) field read;
    the archive recovered the position with ``doc.nodes.index(node)``, which
    compares Nodes by value and returned the FIRST group for both, hiding the
    second group's out-of-clip child.
    """
    doc = _doc(
        '<defs><clipPath id="c"><rect x="0" y="0" width="20" height="20"/></clipPath></defs>'
        '<g clip-path="url(#c)"><circle cx="5" cy="5" r="1" fill="#000"/></g>'
        '<g clip-path="url(#c)"><circle cx="70" cy="70" r="1" fill="#000"/></g>',
        tmp_path=tmp_path,
    )
    groups = [n for n in doc.nodes if n.tag == "g"]
    assert [c.attrib["cx"] for c in corpus._subtree_drawables(doc, groups[0])] == ["5"]
    assert [c.attrib["cx"] for c in corpus._subtree_drawables(doc, groups[1])] == ["70"]
    assert len(corpus.query_outside_clip(doc)) == 1
