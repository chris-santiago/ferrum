"""Full ``stroke_dash=`` form matrix x four normalizing surfaces (#124).

`#124` reverses `f8148bdc` (batch A T9, 2026-09-02): the canonical
``stroke_dash=`` kwarg becomes the superset of ``linetype=``/``line_type=``'s
accepted forms, not a strict subset. This module is the discriminating pin
for that reversal, at the level `tests/test_mark_construction_color_validation.py`
does not reach: it asserts the *rendered* ``stroke-dasharray`` (or the wire
``stroke_dash`` list, for `fm.SelectionMark`, which has no static-SVG render
path), not just the intermediate normalized value, across all four surfaces
that route through the shared ``ferrum.marks.base._normalize_stroke_dash``
helper:

- ``MarkBase`` mark kwargs (``mark_line(stroke_dash=...)``) -- the reported
  surface.
- ``annotate_hline``/``annotate_vline``/``annotate_abline``, each both
  standalone (``.to_svg()``, which draws through the mark) and layered with
  ``+`` (which draws through the ``AnnotationLine`` primitive) -- these two
  representations are derived from one normalized capture
  (`annotations.py`'s ``_capture_line_style``, shared by all three
  constructors), so a form that renders one dasharray on the mark path must
  render the identical dasharray on the primitive path. All three are
  pinned identically rather than two of three: fixing only ``hline``/
  ``vline`` and leaving ``abline`` on its own raw-forwarding path would have
  left a third of the "canonical is the superset" claim unreached by any
  spelling, the same argument `#124` makes for why the restriction should
  never have shipped in the first place.
- ``fm.SelectionMark(stroke_dash=...)``, which used to pass its value
  through completely unvalidated and constructed silently on ``"wiggly"``;
  now it refuses at construction, through the same helper.

Every form accepted before this module's helper extraction still renders
the identical ``stroke-dasharray`` it always did (golden risk is nil, per
the wave-5 design spec's `#124` acceptance criteria) -- the *only* measured
delta is that the five named linetypes now also resolve on the canonical
``stroke_dash=`` kwarg (and, transitively, on the three annotation
constructors and ``SelectionMark``, which had no named-form path at all
before this module's fix). Junk still refuses on every surface.
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm
from ferrum.marks.base import MarkBase
from tests._svg_rects import svg_elements

# ---------------------------------------------------------------------------
# Shared form matrix: (label, raw value, expected rendered stroke-dasharray
# attribute -- None means the attribute is absent, i.e. a solid line).
# ---------------------------------------------------------------------------

_FORM_MATRIX: list[tuple[str, object, str | None]] = [
    # Named linetypes -- the #124 delta: unreachable via stroke_dash= before
    # this module's fix on every one of the three surfaces below.
    ("named_solid", "solid", None),
    ("named_dashed", "dashed", "4,2"),
    ("named_dotted", "dotted", "1,3"),
    ("named_dashdot", "dashdot", "4,2,1,2"),
    ("named_longdash", "longdash", "8,4"),
    # Comma-separated numeric strings, with and without spaces -- already
    # symmetric before this module's fix; pinned here so the helper
    # extraction did not regress them.
    ("comma_string_no_spaces", "4,2", "4,2"),
    ("comma_string_with_spaces", "4, 2", "4,2"),
    ("comma_string_single_value", "6", "6"),
    ("comma_string_four_values", "4,2,1,2", "4,2,1,2"),
    # Both empty spellings -- "" and [] -- are the same solid line.
    ("comma_string_empty", "", None),
    ("empty_list", [], None),
    # Numeric list/tuple forms -- already symmetric; int vs. float makes no
    # rendered difference (the renderer's dasharray text is the same either
    # way).
    ("float_list", [4.0, 2.0], "4,2"),
    ("int_list", [4, 2], "4,2"),
    ("tuple", (4.0, 2.0), "4,2"),
]

_FORM_IDS = [t[0] for t in _FORM_MATRIX]

# Junk: neither a named linetype nor a valid comma-split numeric string.
# Must still refuse on every surface -- #124 widens the vocabulary, it does
# not remove the refusal.
_JUNK_VALUES = ["wiggly", "squiggly", "4,x", "not-a-dash-pattern"]


def _df() -> pl.DataFrame:
    return pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 1.5]})


def _dasharray_by_stroke(svg: str, tag: str, stroke: str) -> str | None:
    """Return the single *tag* element's ``stroke-dasharray`` (or ``None``).

    Scoped by ``stroke`` color rather than by position in the document --
    the repo's established discipline (see ``tests/_svg_rects.py``'s module
    docstring on why ``test_bug_hunt_stack_value_axis.py``'s positional read
    is a known-fragile pattern, not one to add to).
    """
    matches = svg_elements(svg, tag, lambda a: a.get("stroke") == stroke)
    assert len(matches) == 1, f"expected exactly one <{tag} stroke={stroke!r}>, got {matches}"
    return matches[0].get("stroke-dasharray")


# ---------------------------------------------------------------------------
# Surface 1: MarkBase mark kwargs, rendered through mark_line().
# ---------------------------------------------------------------------------


class TestMarkBaseRenderedDasharray:
    @pytest.mark.parametrize("label,value,expected", _FORM_MATRIX, ids=_FORM_IDS)
    def test_form_renders_expected_dasharray(self, label, value, expected):
        svg = (
            fm.Chart(_df())
            .mark_line(stroke="#111111", stroke_dash=value)
            .encode(x="x", y="y")
            .to_svg()
        )
        assert _dasharray_by_stroke(svg, "polyline", "#111111") == expected

    @pytest.mark.parametrize("junk", _JUNK_VALUES)
    def test_junk_refuses(self, junk):
        with pytest.raises(ValueError, match="stroke_dash"):
            MarkBase("line", stroke_dash=junk)

    def test_named_and_alias_forms_render_identically(self):
        # Discriminating: the canonical stroke_dash= kwarg and the linetype=
        # alias must render the *same* dasharray for the same named form --
        # not merely that both construct without raising.
        svg_canonical = (
            fm.Chart(_df())
            .mark_line(stroke="#111111", stroke_dash="dashdot")
            .encode(x="x", y="y")
            .to_svg()
        )
        svg_alias = (
            fm.Chart(_df())
            .mark_line(stroke="#111111", linetype="dashdot")
            .encode(x="x", y="y")
            .to_svg()
        )
        canonical_dash = _dasharray_by_stroke(svg_canonical, "polyline", "#111111")
        alias_dash = _dasharray_by_stroke(svg_alias, "polyline", "#111111")
        assert canonical_dash == "4,2,1,2"
        assert canonical_dash == alias_dash


# ---------------------------------------------------------------------------
# Surface 2: annotate_hline / annotate_vline / annotate_abline, standalone
# AND layered with +.
# ---------------------------------------------------------------------------


class TestAnnotateLineRenderedDasharray:
    @pytest.mark.parametrize("label,value,expected", _FORM_MATRIX, ids=_FORM_IDS)
    def test_hline_standalone_renders_expected_dasharray(self, label, value, expected):
        svg = fm.annotate_hline(y=2.0, stroke="#111111", stroke_dash=value).to_svg()
        assert _dasharray_by_stroke(svg, "line", "#111111") == expected

    @pytest.mark.parametrize("label,value,expected", _FORM_MATRIX, ids=_FORM_IDS)
    def test_hline_overlay_renders_expected_dasharray(self, label, value, expected):
        base = fm.Chart(_df()).mark_line().encode(x="x", y="y")
        svg = (base + fm.annotate_hline(y=1.5, stroke="#111111", stroke_dash=value)).to_svg()
        assert _dasharray_by_stroke(svg, "line", "#111111") == expected

    @pytest.mark.parametrize("label,value,expected", _FORM_MATRIX, ids=_FORM_IDS)
    def test_vline_standalone_renders_expected_dasharray(self, label, value, expected):
        svg = fm.annotate_vline(x=2.0, stroke="#111111", stroke_dash=value).to_svg()
        assert _dasharray_by_stroke(svg, "line", "#111111") == expected

    @pytest.mark.parametrize("label,value,expected", _FORM_MATRIX, ids=_FORM_IDS)
    def test_vline_overlay_renders_expected_dasharray(self, label, value, expected):
        base = fm.Chart(_df()).mark_line().encode(x="x", y="y")
        svg = (base + fm.annotate_vline(x=1.5, stroke="#111111", stroke_dash=value)).to_svg()
        assert _dasharray_by_stroke(svg, "line", "#111111") == expected

    def test_standalone_and_overlay_agree_for_a_named_form(self):
        # Discriminating: the mark path (standalone) and the AnnotationLine
        # primitive path (+ overlay) must resolve the SAME named form to the
        # SAME dasharray -- the bug this module's fix closes is exactly that
        # _capture_line_style fed one raw (unnormalized) value into the
        # primitive while MarkBase separately normalized the mark's copy.
        standalone_svg = fm.annotate_hline(y=2.0, stroke="#111111", stroke_dash="dotted").to_svg()
        base = fm.Chart(_df()).mark_line().encode(x="x", y="y")
        overlay_svg = (
            base + fm.annotate_hline(y=1.5, stroke="#111111", stroke_dash="dotted")
        ).to_svg()
        standalone_dash = _dasharray_by_stroke(standalone_svg, "line", "#111111")
        overlay_dash = _dasharray_by_stroke(overlay_svg, "line", "#111111")
        assert standalone_dash == "1,3"
        assert standalone_dash == overlay_dash

    @pytest.mark.parametrize("junk", _JUNK_VALUES)
    def test_hline_junk_refuses(self, junk):
        # Was unreachable-by-any-spelling before #124: annotate_hline had no
        # linetype= alias, so a named form always raised; now the message
        # must name the actual public entry invoked, not the internal
        # mark_rule construction it used to surface from.
        with pytest.raises(ValueError, match="annotate_hline"):
            fm.annotate_hline(y=2.0, stroke_dash=junk)

    @pytest.mark.parametrize("junk", _JUNK_VALUES)
    def test_vline_junk_refuses(self, junk):
        with pytest.raises(ValueError, match="annotate_vline"):
            fm.annotate_vline(x=2.0, stroke_dash=junk)


class TestAnnotateAblineRenderedDasharray:
    """``annotate_abline`` is the third annotate_* line constructor sharing
    ``_capture_line_style`` -- unlike ``hline``/``vline``, its standalone
    render draws a two-point ``mark_line`` (``<polyline>``), not a
    ``mark_rule`` (``<line>``); the ``+`` overlay path still draws through
    the same ``AnnotationLine`` primitive (``<line>``) as its siblings.
    """

    @pytest.mark.parametrize("label,value,expected", _FORM_MATRIX, ids=_FORM_IDS)
    def test_abline_standalone_renders_expected_dasharray(self, label, value, expected):
        svg = fm.annotate_abline(
            slope=1.0, intercept=0.0, stroke="#111111", stroke_dash=value
        ).to_svg()
        assert _dasharray_by_stroke(svg, "polyline", "#111111") == expected

    @pytest.mark.parametrize("label,value,expected", _FORM_MATRIX, ids=_FORM_IDS)
    def test_abline_overlay_renders_expected_dasharray(self, label, value, expected):
        base = fm.Chart(_df()).mark_line().encode(x="x", y="y")
        svg = (
            base + fm.annotate_abline(slope=1.0, intercept=0.0, stroke="#111111", stroke_dash=value)
        ).to_svg()
        assert _dasharray_by_stroke(svg, "line", "#111111") == expected

    def test_standalone_and_overlay_agree_for_a_named_form(self):
        # Discriminating, mirroring TestAnnotateLineRenderedDasharray's hline
        # pin: this is the exact bug abline shared with hline/vline before it
        # was routed through _capture_line_style -- the mark's copy of
        # stroke_dash was normalized via MarkBase, but AnnotationLine.dash
        # was built from the raw, unnormalized parameter.
        standalone_svg = fm.annotate_abline(
            slope=1.0, intercept=0.0, stroke="#111111", stroke_dash="dotted"
        ).to_svg()
        base = fm.Chart(_df()).mark_line().encode(x="x", y="y")
        overlay_svg = (
            base
            + fm.annotate_abline(slope=1.0, intercept=0.0, stroke="#111111", stroke_dash="dotted")
        ).to_svg()
        standalone_dash = _dasharray_by_stroke(standalone_svg, "polyline", "#111111")
        overlay_dash = _dasharray_by_stroke(overlay_svg, "line", "#111111")
        assert standalone_dash == "1,3"
        assert standalone_dash == overlay_dash

    @pytest.mark.parametrize("junk", _JUNK_VALUES)
    def test_abline_junk_refuses(self, junk):
        # Was unreachable-by-any-spelling before #124, same as hline/vline:
        # annotate_abline had no linetype= alias either.
        with pytest.raises(ValueError, match="annotate_abline"):
            fm.annotate_abline(slope=1.0, intercept=0.0, stroke_dash=junk)

    def test_default_stroke_is_unaffected_by_the_normalization_route(self):
        # abline's default stroke is "black" (unlike hline/vline's "#333"
        # fallback, which only applies when stroke=None) -- routing through
        # _capture_line_style must not silently change it. The renderer
        # normalizes the literal "black" to its hex form ("#000000") in the
        # rendered SVG, which is what this asserts against.
        svg = fm.annotate_abline(slope=1.0, intercept=0.0).to_svg()
        assert _dasharray_by_stroke(svg, "polyline", "#000000") is None


# ---------------------------------------------------------------------------
# Surface 3: fm.SelectionMark -- no static-SVG render path, so pinned at the
# to_spec_dict() wire boundary (the same surface test_bug_hunt_selection_state.py
# and test_selection_value_color_routing.py already use for this class).
# ---------------------------------------------------------------------------


def _expected_wire_stroke_dash(dasharray: str | None) -> list[float]:
    """Translate an expected SVG dasharray string into the wire float list.

    ``None`` (solid, no attribute) corresponds to an explicit empty list on
    the wire -- ``_normalize_stroke_dash`` maps both "solid" and the empty
    comma-string spelling to ``[]``, not to ``None`` (``None`` is reserved
    for "the caller passed no stroke_dash= at all").
    """
    if dasharray is None:
        return []
    return [float(x) for x in dasharray.split(",")]


class TestSelectionMarkStrokeDash:
    @pytest.mark.parametrize("label,value,expected_dasharray", _FORM_MATRIX, ids=_FORM_IDS)
    def test_form_normalizes_to_expected_wire_list(self, label, value, expected_dasharray):
        mark = fm.SelectionMark(stroke_dash=value)
        # list(...) tolerates the tuple form, which passes through
        # unmodified (pinned as its own contract by MarkBase's
        # test_stroke_dash_tuple_passthrough_unchanged) -- content equality
        # is what this matrix is pinning, not container type.
        assert list(mark.to_spec_dict()["stroke_dash"]) == _expected_wire_stroke_dash(
            expected_dasharray
        )

    def test_none_omits_stroke_dash_from_wire_dict(self):
        mark = fm.SelectionMark()
        assert "stroke_dash" not in mark.to_spec_dict()

    @pytest.mark.parametrize("junk", _JUNK_VALUES)
    def test_junk_refuses_at_construction(self, junk):
        # #124: SelectionMark(stroke_dash="wiggly") used to construct
        # silently and pass the raw string through to_spec_dict()
        # unvalidated -- REFUSE MALFORMED requires it to raise, and it must
        # raise at construction (not deferred to to_spec_dict()), matching
        # every other SelectionMark field's construction-time validation
        # (fill/stroke color parsing).
        with pytest.raises(ValueError, match="SelectionMark: stroke_dash"):
            fm.SelectionMark(stroke_dash=junk)

    def test_named_form_matches_marks_base_normalization(self):
        # Discriminating: the same named form resolves to the same wire
        # list on both surfaces -- not merely that both accept it.
        mark = fm.SelectionMark(stroke_dash="dashed")
        mb = MarkBase("line", stroke_dash="dashed")
        assert mark.to_spec_dict()["stroke_dash"] == mb.kwargs["stroke_dash"]
