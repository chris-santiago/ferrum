"""Phase 8b Task 37: continuous_palette() lookup + Gradient factory.

F-L04-02 (Batch A appearance-resolution remediation, 2026-08-28): render-level
parity tests pinning that ``Color(scale=fm.continuous_palette("viridis"))``,
``Color(scale=fm.ContinuousScheme("viridis"))``, and
``Color(scale=fm.SequentialScale(scheme="viridis"))`` are three equivalent
spellings of the same wire form and render byte-identical SVG — see
``TestContinuousSchemeColorScaleParity`` below.

F-L04-02 second revision (spec §4.2, amended 2026-08-28 — supersedes the
earlier refusal amendment, whose "Gradient already renders via
``Color(scheme=...)``" premise was verified false):
``Color(scale=fm.Gradient([...]))`` now renders instead of raising —
``TestContinuousSchemeColorScaleParity`` also pins that a Gradient-backed
scheme actually paints distinct colors (discriminating on stop list and on
``.reversed()``), not merely that it no longer errors.
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fe


def test_viridis_lookup():
    s = fe.continuous_palette("viridis")
    assert s is not None


def test_plasma_lookup():
    fe.continuous_palette("plasma")


def test_magma_lookup():
    fe.continuous_palette("magma")


def test_inferno_lookup():
    fe.continuous_palette("inferno")


def test_cividis_lookup():
    fe.continuous_palette("cividis")


def test_unknown_palette_raises():
    with pytest.raises(ValueError, match="Unknown colormap"):
        fe.continuous_palette("notacolor")


def test_continuous_palette_list():
    # The list is derived from the Rust palette registry (all sequential +
    # diverging schemes), so it covers the classic colormaps and the custom
    # continuous palettes without drifting from what from_name() accepts.
    from ferrum._core import list_palettes, palette_kind

    names = set(fe.continuous_palette.list())
    expected = {n for n in list_palettes() if palette_kind(n) in ("sequential", "diverging")}
    assert names == expected
    # The classic colormaps are always present.
    assert {"viridis", "plasma", "magma", "inferno", "cividis"} <= names
    # Every listed name must be constructible.
    for n in names:
        assert fe.continuous_palette(n) is not None


def test_reversed_returns_new_scheme():
    s = fe.continuous_palette("viridis")
    rev = s.reversed()
    assert rev is not s


def test_gradient_two_stops():
    g = fe.Gradient([(0.0, "red"), (1.0, "blue")])
    assert g is not None


# ---------------------------------------------------------------------------
# Spec reviewer cycle 3, finding 2: Gradient(...) rejects too few stops at
# construction instead of silently constructing a degenerate scheme that
# later falls through to the no-scale default with zero diagnostics.
# ---------------------------------------------------------------------------


def test_gradient_rejects_zero_stops():
    with pytest.raises(ValueError, match="need at least 2 stops"):
        fe.Gradient([])


def test_gradient_rejects_one_stop():
    with pytest.raises(ValueError, match="need at least 2 stops"):
        fe.Gradient([(0.0, "red")])


def test_gradient_rejects_t_out_of_range():
    with pytest.raises(ValueError, match=r"t must be within \[0, 1\]"):
        fe.Gradient([(-0.1, "red"), (1.0, "blue")])
    with pytest.raises(ValueError, match=r"t must be within \[0, 1\]"):
        fe.Gradient([(0.0, "red"), (1.1, "blue")])


def test_gradient_rejects_non_ascending_t():
    with pytest.raises(ValueError, match="strictly sorted ascending"):
        fe.Gradient([(0.5, "red"), (0.2, "blue")])


def test_gradient_rejects_duplicate_t():
    # Strictly ascending, not merely non-decreasing.
    with pytest.raises(ValueError, match="strictly sorted ascending"):
        fe.Gradient([(0.5, "red"), (0.5, "blue")])


# ---------------------------------------------------------------------------
# F-L04-02: Color(scale=continuous_palette(...)) renders identically to
# Color(scale=SequentialScale(scheme=...)) instead of raising TypeError.
# ---------------------------------------------------------------------------


def _color_chart(scale) -> str:
    df = pl.DataFrame({"x": [1, 2, 3, 4], "y": [4, 3, 2, 1], "val": [0.0, 1.0, 2.0, 3.0]})
    return (
        fe.Chart(df).mark_point().encode(x="x", y="y", color=fe.Color("val", scale=scale)).to_svg()
    )


class TestContinuousSchemeColorScaleParity:
    """`continuous_palette`, the direct `ContinuousScheme` constructor, and
    `SequentialScale` must serialize to the same wire dict and therefore
    render byte-identical SVG when passed to `Color(scale=...)`.
    """

    def test_continuous_palette_matches_sequential_scale(self):
        via_palette = _color_chart(fe.continuous_palette("viridis"))
        via_sequential = _color_chart(fe.SequentialScale(scheme="viridis"))
        assert via_palette == via_sequential

    def test_direct_constructor_matches_sequential_scale(self):
        # fm.ContinuousScheme("viridis") is the constructor form the
        # docstrings advertise (appearance.py, ContinuousScheme's own
        # docstring) — previously TypeError'd under Color(scale=...). Use
        # the public re-export (src/ferrum/__init__.py), not the private
        # extension module, so this pins the documented path.
        via_ctor = _color_chart(fe.ContinuousScheme("viridis"))
        via_sequential = _color_chart(fe.SequentialScale(scheme="viridis"))
        assert via_ctor == via_sequential

    def test_continuous_palette_matches_direct_constructor(self):
        via_palette = _color_chart(fe.continuous_palette("viridis"))
        via_ctor = _color_chart(fe.ContinuousScheme("viridis"))
        assert via_palette == via_ctor

    def test_direct_constructor_rejects_unknown_name(self):
        # Pins the rejection path across the PyO3 __new__ boundary
        # (Rust-side coverage lives in continuous.rs's
        # py_continuous_scheme_new_rejects_unknown_name).
        with pytest.raises(ValueError, match="Unknown colormap"):
            fe.ContinuousScheme("notacolor")

    def test_reversed_matches_sequential_scale_reverse_true(self):
        via_reversed = _color_chart(fe.continuous_palette("viridis").reversed())
        via_sequential = _color_chart(fe.SequentialScale(scheme="viridis", reverse=True))
        assert via_reversed == via_sequential

    def test_reversed_differs_from_forward(self):
        # Discriminating check (spec §7): the reversed scheme must actually
        # change rendered output, not just fail to error.
        forward = _color_chart(fe.continuous_palette("viridis"))
        reversed_ = _color_chart(fe.continuous_palette("viridis").reversed())
        assert forward != reversed_

    def test_double_reversed_matches_forward(self):
        forward = _color_chart(fe.continuous_palette("viridis"))
        double_reversed = _color_chart(fe.continuous_palette("viridis").reversed().reversed())
        assert forward == double_reversed

    def test_different_named_scheme_discriminates(self):
        # Discriminating check: a different colormap name must actually
        # change rendered output (guards against a fallback that ignores
        # the name and always renders one hardcoded scheme).
        viridis = _color_chart(fe.continuous_palette("viridis"))
        plasma = _color_chart(fe.continuous_palette("plasma"))
        assert viridis != plasma

    def test_gradient_backed_scheme_differs_from_named_scheme(self):
        # F-L04-02 second revision: Color(scale=fm.Gradient([...])) renders
        # instead of raising — the earlier typed-refusal contract is
        # superseded (spec §4.2, amended 2026-08-28 second revision).
        # Discriminating check: a Gradient's own stops must actually paint
        # colors, not silently fall back to a named colormap (which would
        # also happen to "render" without erroring, but for the wrong
        # reason).
        gradient_svg = _color_chart(fe.Gradient([(0.0, "red"), (1.0, "blue")]))
        viridis_svg = _color_chart(fe.continuous_palette("viridis"))
        assert gradient_svg != viridis_svg

    def test_different_gradient_stops_discriminate(self):
        # Discriminating check (spec §7): two different stop lists must
        # render different SVG bytes, not the same fallback gradient.
        red_to_blue = _color_chart(fe.Gradient([(0.0, "red"), (1.0, "blue")]))
        yellow_to_purple = _color_chart(fe.Gradient([(0.0, "yellow"), (1.0, "purple")]))
        assert red_to_blue != yellow_to_purple

    def test_gradient_reversed_differs_from_forward(self):
        # Discriminating check: .reversed() on a Gradient-backed scheme must
        # actually change rendered output.
        gradient = fe.Gradient([(0.0, "red"), (1.0, "blue")])
        forward = _color_chart(gradient)
        reversed_ = _color_chart(gradient.reversed())
        assert forward != reversed_

    def test_gradient_double_reversed_matches_forward(self):
        gradient = fe.Gradient([(0.0, "red"), (0.5, "green"), (1.0, "blue")])
        forward = _color_chart(gradient)
        double_reversed = _color_chart(gradient.reversed().reversed())
        assert forward == double_reversed

    def test_gradient_named_css_colors_in_stops_render(self):
        # Gradient stops accept the full CSS color vocabulary (named colors,
        # not just hex) — pins that scale= resolution parses stops via the
        # full-CSS parser, not a hex-only path.
        named = fe.Gradient([(0.0, "cornflowerblue"), (1.0, "tomato")])
        hex_equivalent = fe.Gradient([(0.0, "#6495ed"), (1.0, "#ff6347")])
        assert _color_chart(named) == _color_chart(hex_equivalent)

    def test_gradient_non_uniform_t_position_discriminates(self):
        # Spec reviewer cycle 3, finding 1: a Gradient's t positions must
        # actually reach the render, not get silently re-spaced to
        # i / (n - 1). The middle stop at t=0.9 (skewed toward "blue") must
        # render differently from the same three colors evenly spaced at
        # t=0.5 (the pre-fix behavior, which discarded t entirely).
        skewed = _color_chart(fe.Gradient([(0.0, "red"), (0.9, "green"), (1.0, "blue")]))
        even = _color_chart(fe.Gradient([(0.0, "red"), (0.5, "green"), (1.0, "blue")]))
        assert skewed != even

    def test_named_scheme_wire_dict_omits_stops_key(self):
        # Byte-identity guard (spec §7), Python mirror of Rust's
        # scale_spec_sequential_stops_absent_omits_key: a named-colormap
        # scheme's wire dict carries no "stops" key at all — checked at the
        # wire level (not by comparing two post-change spellings against
        # each other, which stays green even if both sides regressed to
        # emitting an empty stops list identically).
        #
        # _to_scale_spec_dict is deliberately absent from every *Scale stub
        # in _core.pyi (test_scale_spec_parity.py's family-wide policy: the
        # stubs expose the public surface, not this internal SPEC-04
        # delegation hook), so pyright cannot see the attribute here.
        assert (
            "stops" not in fe.continuous_palette("viridis")._to_scale_spec_dict()  # pyright: ignore[reportAttributeAccessIssue]
        )
        assert (
            "stops" not in fe.SequentialScale(scheme="viridis")._to_scale_spec_dict()  # pyright: ignore[reportAttributeAccessIssue]
        )
