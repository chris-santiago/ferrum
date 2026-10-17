"""Every ``RenderWarning`` auxiliary-scale resolution pushes is recorded
exactly once per ``to_svg()`` call, and a non-linear opacity-family scale
refuses rather than warning.

The population under test is derived from a grep of every ``warnings.push``
site in the two source files ``build_auxiliary_scales`` calls into, rather
than hand-listed: ``grep -oE
'warnings\\.push\\(crate::render::RenderWarning::[A-Za-z]+'
crates/ferrum-core/src/render/scale_resolve/{auxiliary,color}.rs``, variant
names deduplicated. At this tree that grep returns five variants:
``ShapePaletteOverflowed``,
``StrokeDashPaletteOverflowed`` (``auxiliary.rs``), and
``UnknownColorScheme``, ``GradientTooFewStops``, ``ColorPaletteOverflowed``
(``color.rs``). ``test_population_is_covered`` fails if a future push site
changes that set without a matching case here or an explicit, justified
exclusion (see ``_UNREACHABLE_VIA_PUBLIC_API`` below).

Two independent facts are pinned:

- **Once per render.** ``ShapePaletteOverflowed`` and ``StrokeDashPaletteOverflowed``
  are whole-chart facts — every panel of a
  faceted chart re-derives byte-identical text — so they are recorded exactly
  once per ``to_svg()``, faceted or not. ``ColorPaletteOverflowed`` is
  different by ruling: an independently-resolved color scale can legitimately
  overflow its palette differently per panel, so its provisional pre-pass
  copy (whose scale is never rendered, only used for layout/legend sizing) is
  dropped, but each panel's own copy survives — a faceted chart with N panels
  records it N times, not once. ``UnknownColorScheme`` is recorded once on an
  unfaceted chart; on a faceted chart it still records once per panel rather
  than collapsing like its siblings above — a known gap (F36) this file does
  not assert on (see the note on ``_FACETED_CASES`` below).
- **A non-linear opacity scale refuses.** A ``fill_opacity`` column under a
  non-``Linear`` ``scale=`` refuses with the numeric auxiliary scale gate's
  text instead of warning and falling back.
"""

import re
import warnings
from pathlib import Path

import polars as pl
import pytest

import ferrum as fm

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCALE_RESOLVE_DIR = _REPO_ROOT / "crates" / "ferrum-core" / "src" / "render" / "scale_resolve"
_PUSH_PATTERN = re.compile(r"warnings\.push\(crate::render::RenderWarning::(\w+)")


def _warning_producers(rust_file: str) -> frozenset[str]:
    return frozenset(_PUSH_PATTERN.findall((_SCALE_RESOLVE_DIR / rust_file).read_text()))


def _render_and_record(chart) -> tuple[str, list[str]]:
    """Render ``chart`` and return its SVG and the ``UserWarning`` messages recorded."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = chart.to_svg()
    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    return svg, messages


def _record_warnings(chart) -> list[str]:
    """Render ``chart`` and return the ``UserWarning`` messages recorded."""
    return _render_and_record(chart)[1]


# GradientTooFewStops is reachable only from a hand-built spec that bypasses
# the fm.Gradient(...) constructor (color.rs's own doc comment: the public
# constructor already refuses fewer than 2 stops, so this arm exists for a
# hand-written ChartSpec.from_json payload, which this file does not build).
# Every other producer below has a real Python door.
_UNREACHABLE_VIA_PUBLIC_API = frozenset({"GradientTooFewStops"})

_N_PANELS = 3
_GROUPS = ("a", "b", "c")
_OPACITY_ROWS = 20

# The default categorical palette's length, read through the public door
# (ferrum-core publishes it; fm.color.palette wraps that, on the
# scale_accepted_keys precedent), not hand-picked: one more category than
# this overflows it. SHAPE_PALETTE and DASH_PALETTE (used below) have no
# equivalent Python-reachable binding, so their overflow counts stay
# hand-picked against the values read from the Rust source at write time.
_COLOR_PALETTE_LEN = len(fm.color.palette("tableau10"))
_COLOR_OVERFLOW_N = _COLOR_PALETTE_LEN + 1


def _categorical_df(n_cat: int, *, groups: tuple[str, ...] | None = None) -> pl.DataFrame:
    """``n_cat`` distinct string categories in ``cat``.

    With ``groups`` given, the SAME ``n_cat`` categories repeat identically
    inside every group (column ``g``), so a categorical-domain warning's text
    is byte-identical across panels.
    """
    if groups is None:
        return pl.DataFrame(
            {
                "x": list(range(n_cat)),
                "y": list(range(n_cat)),
                "cat": [f"c{i}" for i in range(n_cat)],
            }
        )
    rows = [(i, i, f"c{i}", g) for g in groups for i in range(n_cat)]
    return pl.DataFrame(rows, schema=["x", "y", "cat", "g"], orient="row")


def _stroke_dash_df(n_cat: int, *, groups: tuple[str, ...] | None = None) -> pl.DataFrame:
    """``n_cat`` distinct dash categories, each with 2 rows.

    A 1-point line group draws nothing and would add an unrelated
    ``MarkBuildEmpty`` warning to the recorded list, so every category gets a
    real 2-point segment.
    """

    def _one_group():
        return [(i, i, f"s{i % n_cat}") for i in range(n_cat * 2)]

    if groups is None:
        return pl.DataFrame(_one_group(), schema=["x", "y", "series"], orient="row")
    rows = [(x, y, series, g) for g in groups for x, y, series in _one_group()]
    return pl.DataFrame(rows, schema=["x", "y", "series", "g"], orient="row")


def _opacity_scale_channel(field: str):
    return fm.FillOpacity(field, scale=fm.LogScale())


def _shape_overflow_chart():
    # SHAPE_PALETTE has 8 entries (scale_resolve/mod.rs); 9 categories overflows it.
    return fm.Chart(_categorical_df(9)).mark_point().encode(x="x:Q", y="y:Q", shape="cat:N")


def _shape_overflow_facet_chart():
    df = _categorical_df(9, groups=_GROUPS)
    return fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q", shape="cat:N", facet=fm.Facet("g"))


def _stroke_dash_overflow_chart():
    # DASH_PALETTE has 3 entries plus the solid slot = 4 slots; 5 categories overflows it.
    return fm.Chart(_stroke_dash_df(5)).mark_line().encode(x="x:Q", y="y:Q", stroke_dash="series:N")


def _stroke_dash_overflow_facet_chart():
    df = _stroke_dash_df(5, groups=_GROUPS)
    return (
        fm.Chart(df)
        .mark_line()
        .encode(x="x:Q", y="y:Q", stroke_dash="series:N", facet=fm.Facet("g"))
    )


def _opacity_constant_column_chart():
    df = pl.DataFrame(
        {
            "x": list(range(_OPACITY_ROWS)),
            "y": [i * 2.0 + 1.0 for i in range(_OPACITY_ROWS)],
            "fo": [0.3] * _OPACITY_ROWS,
        }
    )
    return (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", fill_opacity=_opacity_scale_channel("fo"))
    )


def _color_overflow_chart():
    df = _categorical_df(_COLOR_OVERFLOW_N)
    return fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q", color="cat:N")


def _color_overflow_facet_chart():
    df = _categorical_df(_COLOR_OVERFLOW_N, groups=_GROUPS)
    return fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q", color="cat:N", facet=fm.Facet("g"))


def _unknown_color_scheme_chart():
    # Color(scheme=...) validates the name eagerly at construction (raises);
    # scale=SequentialScale(scheme=...) does not, and is the render-reachable door.
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "v": [1.0, 2.0, 3.0]})
    return (
        fm.Chart(df)
        .mark_point()
        .encode(
            x="x:Q",
            y="y:Q",
            color=fm.Color("v", scale=fm.SequentialScale(scheme="not-a-scheme")),
        )
    )


_REFUSAL_TEXT = (
    "fill_opacity scale: type 'log' is not supported; "
    "pass fm.LinearScale(domain=[...], range=[...]) or drop scale="
)
_SHAPE_TEXT = "shape palette has fewer entries than the 9 categories; shapes were recycled"
_DASH_TEXT = "stroke dash palette has fewer entries than the 5 categories; dashes were recycled"
_COLOR_TEXT = (
    f"color palette has fewer entries than the {_COLOR_OVERFLOW_N} categories; colors were recycled"
)
_UNKNOWN_SCHEME_TEXT = (
    "color scheme 'not-a-scheme' is not recognized; the theme's default scheme was used instead"
)

# One representative unfaceted chart + its exact expected message per
# grep-derived, publicly-reachable producer.
_UNFACETED_CASES: dict[str, tuple] = {
    "ShapePaletteOverflowed": (_shape_overflow_chart, _SHAPE_TEXT),
    "StrokeDashPaletteOverflowed": (_stroke_dash_overflow_chart, _DASH_TEXT),
    "ColorPaletteOverflowed": (_color_overflow_chart, _COLOR_TEXT),
    "UnknownColorScheme": (_unknown_color_scheme_chart, _UNKNOWN_SCHEME_TEXT),
}

# Faceted (3-panel, shared resolution) cell per producer whose count this
# file pins: the two whole-chart facts collapse to exactly one warning
# across the panel loop; ColorPaletteOverflowed keeps one copy per panel,
# since an independently-resolved color scale can legitimately overflow
# differently per panel, so its expected count is _N_PANELS, not 1.
# UnknownColorScheme has no entry here: on a faceted chart it still records
# once per panel (3, not 1) rather than collapsing like its siblings above —
# a known gap (F36) this file does not assert on.
_FACETED_CASES: dict[str, tuple] = {
    "ShapePaletteOverflowed": (_shape_overflow_facet_chart, _SHAPE_TEXT, 1),
    "StrokeDashPaletteOverflowed": (_stroke_dash_overflow_facet_chart, _DASH_TEXT, 1),
    "ColorPaletteOverflowed": (_color_overflow_facet_chart, _COLOR_TEXT, _N_PANELS),
}


def test_population_is_covered():
    """The grep-derived producer set has an unfaceted case for every publicly
    reachable member."""
    producers = _warning_producers("auxiliary.rs") | _warning_producers("color.rs")
    assert producers, "grep found no RenderWarning producer"
    assert _UNREACHABLE_VIA_PUBLIC_API <= producers, (
        "a producer named in _UNREACHABLE_VIA_PUBLIC_API is no longer pushed; drop its exclusion"
    )
    reachable = producers - _UNREACHABLE_VIA_PUBLIC_API
    assert reachable == set(_UNFACETED_CASES), (
        f"reachable producers are {sorted(reachable)} but this test covers "
        f"{sorted(_UNFACETED_CASES)}; add or remove a case for the difference"
    )


@pytest.mark.parametrize("variant", sorted(_UNFACETED_CASES))
def test_producer_warns_exactly_once(variant):
    build_chart, expected_text = _UNFACETED_CASES[variant]
    messages = _record_warnings(build_chart())
    assert messages == [expected_text], (
        f"{variant}: expected exactly one warning with text {expected_text!r}, got {messages!r}"
    )


@pytest.mark.parametrize("variant", sorted(_FACETED_CASES))
def test_producer_warns_correctly_when_faceted(variant):
    build_chart, expected_text, expected_count = _FACETED_CASES[variant]
    messages = _record_warnings(build_chart())
    assert messages == [expected_text] * expected_count, (
        f"{variant}: expected exactly {expected_count} warning(s) with text "
        f"{expected_text!r}, got {messages!r}"
    )


def test_no_auxiliary_warning_is_vacuity_control():
    """A chart with no auxiliary-scale channel records zero warnings."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
    assert _record_warnings(chart) == []


def test_a_non_linear_opacity_scale_refuses():
    """A ``fill_opacity`` column under a non-linear ``scale=`` refuses with the
    gate's text, whether the channel class or the render refuses it."""
    with pytest.raises(ValueError, match=re.escape(_REFUSAL_TEXT)):
        _opacity_constant_column_chart().to_svg()
