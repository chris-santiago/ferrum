"""Pins the ``"theme:contrast"`` text-fill sentinel: a text layer whose
``fill`` is the literal string ``"theme:contrast"`` paints each row's label
in whichever ink (the theme's font color, its background, or black/white)
reads best against the color drawn under that row, per the WCAG 2.x
contrast formula.

The sentinel is engine-internal, not public color vocabulary: the public
``mark_text(fill=...)`` door still refuses it exactly as it refuses any
other unrecognized paint string (see ``marks/base.py``'s
``_is_paint_sentinel``, which lists only ``"theme:label"``, ``"none"`` and
``"transparent"``). A producer reaches it the same way existing composite
marks reach the kindred ``"theme:label"`` sentinel
(``marks/composite.py``'s box-plot whisker/cap layers set
``mark_kwargs={"stroke": "theme:label", ...}`` on a hand-built ``_Layer``,
bypassing the public mark door entirely): by constructing ``ferrum._layer._Layer``
directly and attaching it with ``Chart.layer(...)``. Every chart this module
builds does the same, mirroring the one mechanism a figure generator would
use to emit the sentinel.

Every contrast assertion here checks the OUTCOME property (the WCAG 2.x
contrast ratio between the drawn ink and the actual rendered cell color,
read back out of the SVG/scene) rather than re-deriving the ink by replaying
Rust's own branching logic on the raw input data -- a test built that way
would just restate the implementation as its own oracle. ``tests/_wcag.py``
is the public W3C formula, not a copy of any ferrum source, shared with
``tests/test_annotation_contrast.py``.
"""

from __future__ import annotations

import json
import re
import warnings

import polars as pl
import pytest

import ferrum as fm
from ferrum._layer import _Layer
from tests._snapshots import legend_texts
from tests._svg_rects import figure_elements
from tests._wcag import (
    contrast_ratio as _contrast_ratio,
    opaque_rgb as _opaque_rgb,
    parse_paint as _parse_paint,
)

# ---------------------------------------------------------------------------
# Public vocabulary boundary: the sentinel is reachable only through the
# internal `_Layer` route every other test in this module uses.
# ---------------------------------------------------------------------------


def test_the_public_mark_text_door_still_refuses_the_sentinel_as_a_color() -> None:
    """``"theme:contrast"`` is not public color vocabulary.

    It reaches the renderer only through an internal `_Layer`, the same way
    `"theme:label"` reaches it through `marks/composite.py`'s hand-built
    layers; a caller spelling it through the public `mark_text(fill=...)`
    door gets the same refusal any other unrecognized color string does.
    """
    with pytest.raises(ValueError, match="theme:contrast"):
        fm.Chart(pl.DataFrame({"x": [1]})).mark_text(fill="theme:contrast")


_WCAG_AA = 4.5

# ---------------------------------------------------------------------------
# Chart construction
# ---------------------------------------------------------------------------

#: A fixed theme for every test below, so expected fallback inks are values
#: this module chose, not ferrum's current built-in theme defaults.
_FONT = "#1a1a1a"
_BACKGROUND = "#f2f2f2"
_FONT_RGB = (0x1A, 0x1A, 0x1A)
_BACKGROUND_RGB = (0xF2, 0xF2, 0xF2)


def _expected_ink(
    cell_rgb: tuple[int, int, int],
    *,
    font_rgb: tuple[int, int, int] = _FONT_RGB,
    background_contrast_ratio: float | None = None,
) -> str:
    """The ink spec 6.5's own rule picks for *cell_rgb*, independent of Rust's code.

    First tier: whichever of the theme font color or background has the
    higher contrast against the cell, when that higher ratio clears
    `_WCAG_AA`. Second tier: whichever of black/white contrasts better,
    when neither theme ink clears AA. This is the acceptance rule itself
    (spec 6.5), not a copy of `contrast_ink`'s Rust body -- it is built
    entirely from `_contrast_ratio`, this module's own WCAG implementation.

    `background_contrast_ratio` lets a caller whose theme background is
    FULLY transparent (alpha 0) supply the background candidate's ratio
    directly: at alpha 0 the background paints the cell completely
    unchanged when composited over it, so its contrast against the cell is
    always exactly 1.0 (it can never win) -- computing that from a
    module-level opaque background constant would be the wrong color
    entirely, not just a wrong ratio. This does NOT generalize to a
    partially translucent background: at any alpha between 0 and 1 the
    background blends with the cell and is judged by that blend, whose
    contrast is neither 1.0 nor fixed, so it can win. A caller testing
    that case must compute the real blended ratio itself, not pass an
    override here.
    """
    font_ratio = _contrast_ratio(font_rgb, cell_rgb)
    bg_ratio = (
        background_contrast_ratio
        if background_contrast_ratio is not None
        else _contrast_ratio(_BACKGROUND_RGB, cell_rgb)
    )
    if max(font_ratio, bg_ratio) >= _WCAG_AA:
        return _FONT if font_ratio >= bg_ratio else _BACKGROUND
    black_ratio = _contrast_ratio((0, 0, 0), cell_rgb)
    white_ratio = _contrast_ratio((255, 255, 255), cell_rgb)
    return "#000000" if black_ratio >= white_ratio else "#ffffff"


def _test_theme() -> fm.Theme:
    return fm.Theme(font_color=_FONT, background_color=_BACKGROUND)


def _contrast_text_layer(*, x: str, y: str, color: str, text: str) -> _Layer:
    """A hand-built text layer carrying the sentinel on its own `mark_kwargs`."""
    return _Layer(
        mark="text",
        encoding={"x": x, "y": y, "color": color, "text": text},
        mark_kwargs={"fill": "theme:contrast"},
        data_source=None,
    )


def _plain_text_layer(*, x: str, y: str, color: str, text: str) -> _Layer:
    """The same layer with no fill override: its own color draws the raw mapped hue."""
    return _Layer(
        mark="text",
        encoding={"x": x, "y": y, "color": color, "text": text},
        data_source=None,
    )


def _rect_text_rgb_pairs(svg: str) -> list[tuple[tuple[int, int, int], str, str]]:
    """Pair each data `<rect>`'s displayed (composited) color with the `<text>` at its center.

    Matched purely by geometry (`figure_elements` resolves every coordinate
    into one shared figure space), since the rect and text are drawn by two
    separate layers, not necessarily adjacent in document order.
    """
    elements = figure_elements(svg)
    rects = [
        e
        for e in elements
        if e["tag"] == "rect"
        and e.get("fill")
        and e["fill"] != _BACKGROUND
        and not e["fill"].startswith("url(")
    ]
    texts = [e for e in elements if e["tag"] == "text"]
    pairs: list[tuple[tuple[int, int, int], str, str]] = []
    for r in rects:
        cx, cy = r["x"] + r["w"] / 2, r["y"] + r["h"] / 2
        match = next(
            (t for t in texts if abs(t["x"] - cx) < 0.5 and abs(t["y"] - cy) < 0.5),
            None,
        )
        if match is not None:
            pairs.append((_opaque_rgb(r["fill"], _BACKGROUND_RGB), match["fill"], match["text"]))
    return pairs


# ---------------------------------------------------------------------------
# Continuous lookup + null fallback
# ---------------------------------------------------------------------------


def _continuous_chart() -> fm.Chart:
    df = pl.DataFrame(
        {
            "row": ["p0", "p1", "p2", "p3"],
            "col": ["q0", "q0", "q1", "q1"],
            "value": [0.0, 10.0, None, 5.0],
            "label": ["L0", "L1", "L2", "L3"],
        }
    )
    base = fm.Chart(df).mark_rect().encode(x="col", y="row", color="value").theme(_test_theme())
    return base.layer(_contrast_text_layer(x="col", y="row", color="value", text="label"))


def test_continuous_lookup_contrast_meets_wcag_aa_against_the_rendered_cell() -> None:
    svg = _continuous_chart().to_svg()
    pairs = _rect_text_rgb_pairs(svg)
    non_null = [(cell, ink, text) for cell, ink, text in pairs if text != "L2"]
    assert len(non_null) == 3, "expected the three finite-value rows to pair up"
    for cell_rgb, ink, text in non_null:
        assert ink.lower() == _expected_ink(cell_rgb), (
            f"{text}: unexpected ink {ink!r} against cell {cell_rgb}"
        )


def test_continuous_null_value_is_judged_against_the_theme_mark_color() -> None:
    """A null color value is not a plain constant fallback: the rect still paints a real
    fallback cell (the theme's mark color), and the label is judged against that cell by
    the same WCAG rule every resolved row uses, read back from the actual rendered cell.
    """
    svg = _continuous_chart().to_svg()
    pairs = {text: cell for cell, _, text in _rect_text_rgb_pairs(svg)}
    inks = {text: ink for _, ink, text in _rect_text_rgb_pairs(svg)}
    assert inks["L2"].lower() == _expected_ink(pairs["L2"])


def test_continuous_lookup_discriminates_against_a_plain_fixed_fill() -> None:
    """A single constant ink fails the dark cell the sentinel reads correctly."""
    svg = _continuous_chart().to_svg()
    pairs = {text: cell for cell, _, text in _rect_text_rgb_pairs(svg)}
    inks = {text: ink for _, ink, text in _rect_text_rgb_pairs(svg)}
    dark_cell = pairs["L1"]  # value=10.0, the darkest cell in the domain
    fixed_ratio = _contrast_ratio(_FONT_RGB, dark_cell)
    assert fixed_ratio < _WCAG_AA, (
        "test setup must make a fixed font-color fill fail at least one cell"
    )
    drawn_ink_rgb = _parse_paint(inks["L1"])[:3]
    assert inks["L1"].lower() != _FONT, (
        "the sentinel must not draw the same constant a fixed fill would"
    )
    assert _contrast_ratio(drawn_ink_rgb, dark_cell) >= _WCAG_AA, (
        "the sentinel's own ink must clear AA where the fixed fill fails"
    )


# ---------------------------------------------------------------------------
# Categorical lookup + out-of-domain fallback
# ---------------------------------------------------------------------------


def _categorical_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "row": ["p0", "p1", "p2"],
            "col": ["q0", "q0", "q0"],
            "cat": ["Alpha", "Beta", "Gamma"],
            "label": ["L0", "L1", "L2"],
        }
    )


def _categorical_color() -> fm.Color:
    # "Gamma" is deliberately left out of the domain: an out-of-domain lookup.
    return fm.Color(
        "cat", scale=fm.OrdinalScale(domain=["Alpha", "Beta"], range=["#ffffff", "#000000"])
    )


def _categorical_chart(text_layer_builder) -> fm.Chart:
    df = _categorical_df()
    base = (
        fm.Chart(df)
        .mark_rect()
        .encode(x="col", y="row", color=_categorical_color())
        .theme(_test_theme())
    )
    return base.layer(text_layer_builder(x="col", y="row", color="cat", text="label"))


def test_categorical_lookup_contrast_meets_wcag_aa_for_each_mapped_category() -> None:
    svg = _categorical_chart(_contrast_text_layer).to_svg()
    pairs = {text: (cell, ink) for cell, ink, text in _rect_text_rgb_pairs(svg)}
    cell, ink = pairs["L0"]  # Alpha -> white
    assert ink.lower() == _expected_ink(cell)
    cell, ink = pairs["L1"]  # Beta -> black
    assert ink.lower() == _expected_ink(cell)
    assert pairs["L0"][1].lower() != pairs["L1"][1].lower(), (
        "the two categories must not resolve to the same ink"
    )


def test_categorical_out_of_domain_is_judged_against_the_theme_mark_color() -> None:
    """An out-of-domain category is not a plain constant fallback: the rect still paints a
    real fallback cell (the theme's mark color), and the label is judged against that cell
    by the same WCAG rule every resolved category uses, read back from the actual rendered
    cell.
    """
    svg = _categorical_chart(_contrast_text_layer).to_svg()
    pairs = {text: cell for cell, _, text in _rect_text_rgb_pairs(svg)}
    inks = {text: ink for _, ink, text in _rect_text_rgb_pairs(svg)}
    assert inks["L2"].lower() == _expected_ink(pairs["L2"])


def test_categorical_lookup_discriminates_against_a_plain_fixed_fill() -> None:
    svg = _categorical_chart(_contrast_text_layer).to_svg()
    pairs = {text: cell for cell, _, text in _rect_text_rgb_pairs(svg)}
    inks = {text: ink for _, ink, text in _rect_text_rgb_pairs(svg)}
    black_cell = pairs["L1"]  # Beta's cell is pure black
    fixed_ratio = _contrast_ratio(_FONT_RGB, black_cell)
    assert fixed_ratio < _WCAG_AA, "a plain fixed dark fill must fail against a black cell"
    drawn_ink_rgb = _parse_paint(inks["L1"])[:3]
    assert inks["L1"].lower() != _FONT, (
        "the sentinel must not draw the same constant a fixed fill would"
    )
    assert _contrast_ratio(drawn_ink_rgb, black_cell) >= _WCAG_AA, (
        "the sentinel's own ink must clear AA where the fixed fill fails"
    )


def test_categorical_own_color_binding_still_varies_per_row_unlike_a_constant_fallback() -> None:
    """The sentinel reads the layer's own per-row color (not one constant ink)."""
    svg = _categorical_chart(_contrast_text_layer).to_svg()
    texts = {e["text"]: e["fill"] for e in figure_elements(svg) if e["tag"] == "text"}
    assert len({texts["L0"], texts["L1"], texts["L2"]}) > 1, (
        "ink must vary by row, not be a single constant"
    )


def test_categorical_plain_color_draws_the_raw_mapped_hue_the_sentinel_does_not() -> None:
    """Without the sentinel, the layer's own color draws the category's literal mapped color."""
    svg = _categorical_chart(_plain_text_layer).to_svg()
    texts = {e["text"]: e["fill"] for e in figure_elements(svg) if e["tag"] == "text"}
    assert texts["L0"].lower() == "#ffffff"  # Alpha's own mapped color
    assert texts["L1"].lower() == "#000000"  # Beta's own mapped color


# ---------------------------------------------------------------------------
# Below-AA fallback to the better of black/white
# ---------------------------------------------------------------------------


def test_below_aa_falls_back_to_whichever_of_black_or_white_contrasts_better() -> None:
    df = pl.DataFrame({"row": ["p0"], "col": ["q0"], "cat": ["mid"], "label": ["L0"]})
    low_contrast_theme = fm.Theme(font_color="#a0a0a0", background_color="#606060")
    color = fm.Color("cat", scale=fm.OrdinalScale(domain=["mid"], range=["#808080"]))
    chart = (
        fm.Chart(df)
        .mark_rect()
        .encode(x="col", y="row", color=color)
        .theme(low_contrast_theme)
        .layer(_contrast_text_layer(x="col", y="row", color="cat", text="label"))
    )
    svg = chart.to_svg()
    pairs = _rect_text_rgb_pairs(svg)
    assert len(pairs) == 1
    cell_rgb, ink, _ = pairs[0]

    font_rgb, bg_rgb = (0xA0, 0xA0, 0xA0), (0x60, 0x60, 0x60)
    font_ratio = _contrast_ratio(font_rgb, cell_rgb)
    bg_ratio = _contrast_ratio(bg_rgb, cell_rgb)
    assert font_ratio < _WCAG_AA and bg_ratio < _WCAG_AA, (
        "test setup must make both theme inks fail AA"
    )

    black_ratio = _contrast_ratio((0, 0, 0), cell_rgb)
    white_ratio = _contrast_ratio((255, 255, 255), cell_rgb)
    best_fallback = "#000000" if black_ratio >= white_ratio else "#ffffff"
    assert ink.lower() == best_fallback


# ---------------------------------------------------------------------------
# Alpha compositing: the ink must respond to the composited color, not the
# raw (pre-composite) cell hue.
# ---------------------------------------------------------------------------


def test_translucent_cell_is_composited_before_contrast_is_judged() -> None:
    df = pl.DataFrame({"row": ["p0"], "col": ["q0"], "cat": ["half"], "label": ["L0"]})
    color = fm.Color("cat", scale=fm.OrdinalScale(domain=["half"], range=["rgba(0,0,0,0.5)"]))
    chart = (
        fm.Chart(df)
        .mark_rect()
        .encode(x="col", y="row", color=color)
        .theme(_test_theme())
        .layer(_contrast_text_layer(x="col", y="row", color="cat", text="label"))
    )
    svg = chart.to_svg()
    pairs = _rect_text_rgb_pairs(svg)
    assert len(pairs) == 1
    composited_rgb, ink, _ = pairs[0]

    raw_black = (0, 0, 0)
    assert composited_rgb != raw_black, (
        "test setup must make the composited color visibly lighter than the raw cell hue"
    )

    # Had the engine judged contrast against the raw, un-composited black
    # cell instead of the displayed composited color, white would win by a
    # wide margin; against the true composited (lighter) color, black wins.
    naive_ink = (
        "#000000"
        if _contrast_ratio((0, 0, 0), raw_black) >= _contrast_ratio((255, 255, 255), raw_black)
        else "#ffffff"
    )
    composited_ink = (
        "#000000"
        if _contrast_ratio((0, 0, 0), composited_rgb)
        >= _contrast_ratio((255, 255, 255), composited_rgb)
        else "#ffffff"
    )
    assert naive_ink != composited_ink, "test setup must make compositing change the preferred ink"
    assert ink.lower() == composited_ink


# ---------------------------------------------------------------------------
# No second scale, no second legend
# ---------------------------------------------------------------------------


def test_continuous_own_color_binding_adds_no_second_legend() -> None:
    svg = _continuous_chart().to_svg()
    colorbar_ids = set(re.findall(r"ferrum-colorbar-\d+", svg))
    assert len(colorbar_ids) == 1, (
        "a second drawing layer on the same color field must not add a second colorbar"
    )
    assert svg.count("<linearGradient") == 1


def test_categorical_own_color_binding_adds_no_second_legend() -> None:
    svg = _categorical_chart(_contrast_text_layer).to_svg()
    texts = legend_texts(svg)
    assert texts.count("Alpha") == 1
    assert texts.count("Beta") == 1
    assert texts.count("cat") == 1


# ---------------------------------------------------------------------------
# Charts without the sentinel are unaffected
# ---------------------------------------------------------------------------


def test_a_literal_fill_on_mark_text_is_unaffected_by_the_sentinel_code_path() -> None:
    df = pl.DataFrame({"row": ["p0", "p1"], "col": ["q0", "q0"], "v": [0.0, 1.0]})
    chart = fm.Chart(df).mark_text(fill="#123456").encode(x="col", y="row", text="v")
    svg = chart.to_svg()
    texts = [e for e in figure_elements(svg) if e["tag"] == "text" and e["text"] in ("0", "1")]
    assert len(texts) == 2
    assert {e["fill"].lower() for e in texts} == {"#123456"}


# ---------------------------------------------------------------------------
# Interactive output
# ---------------------------------------------------------------------------


def _interactive_text_colors(scene: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for panel in scene["panels"]:
        for batch in panel["marks"]:
            if batch["kind"] != "text":
                continue
            for node in batch["nodes"]:
                c = node["style"]["color"]
                out[node["content"]] = "#{:02x}{:02x}{:02x}".format(c["r"], c["g"], c["b"])
    return out


def test_interactive_output_carries_the_resolved_per_node_color() -> None:
    chart = _categorical_chart(_contrast_text_layer)
    static_svg = chart.to_svg()
    static_by_text = {
        e["text"]: e["fill"].lower() for e in figure_elements(static_svg) if e["tag"] == "text"
    }
    scene_json = chart.interactive().scene_json
    interactive_by_text = _interactive_text_colors(json.loads(scene_json))

    assert interactive_by_text["L0"] == static_by_text["L0"]
    assert interactive_by_text["L1"] == static_by_text["L1"]
    assert interactive_by_text["L2"] == static_by_text["L2"]
    # Never the literal sentinel string leaking through unresolved: checked
    # against the raw wire payload, not a dict of already-resolved hex
    # strings this module itself formatted (which could never contain it).
    assert "theme:contrast" not in scene_json


# ---------------------------------------------------------------------------
# A contrast text that is the only reader of a color field binds it to pick
# an ink but does not draw it: no scale, no legend, one warning.
# ---------------------------------------------------------------------------


def _only_reader_continuous() -> tuple[pl.DataFrame, str]:
    df = pl.DataFrame(
        {
            "row": ["p0", "p1"],
            "col": ["q0", "q0"],
            "value": [0.0, 10.0],
            "label": ["L0", "L1"],
        }
    )
    return df, "value"


def _only_reader_categorical() -> tuple[pl.DataFrame, str]:
    df = pl.DataFrame(
        {
            "row": ["p0", "p1"],
            "col": ["q0", "q0"],
            "cat": ["Alpha", "Beta"],
            "label": ["L0", "L1"],
        }
    )
    return df, "cat"


@pytest.mark.parametrize(
    "build",
    [_only_reader_continuous, _only_reader_categorical],
    ids=["continuous", "categorical"],
)
def test_a_contrast_text_as_the_only_color_reader_draws_no_scale_and_warns_once(build) -> None:
    df, field = build()
    base = fm.Chart(df).mark_rect().encode(x="col", y="row").theme(_test_theme())
    chart = base.layer(_contrast_text_layer(x="col", y="row", color=field, text="label"))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = chart.to_svg()
    messages = [str(w.message) for w in caught]
    assert len(messages) == 1, messages
    assert "no layer draws that color" in messages[0], messages
    assert "theme mark color" in messages[0], messages

    assert not re.findall(r"ferrum-colorbar-\d+", svg), (
        "an orphan color binding must build no colorbar"
    )
    assert svg.count("<linearGradient") == 0
    legend = legend_texts(svg)
    assert field not in legend, "an orphan color binding must add no legend"
    for category in df[field].to_list():
        if isinstance(category, str):
            assert category not in legend, (
                f"{category}: an orphan color binding must add no legend entry"
            )

    pairs = {text: (cell, ink) for cell, ink, text in _rect_text_rgb_pairs(svg)}
    for text, (cell_rgb, ink) in pairs.items():
        assert ink.lower() == _expected_ink(cell_rgb), (
            f"{text}: unexpected ink {ink!r} against cell {cell_rgb}"
        )


# ---------------------------------------------------------------------------
# The sentinel is honored in exactly one slot: a text mark's own `fill`.
# ---------------------------------------------------------------------------


def test_sentinel_fill_is_refused_as_an_invalid_color_on_a_non_text_mark() -> None:
    df = pl.DataFrame({"row": ["p0"], "col": ["q0"]})
    rect_layer = _Layer(
        mark="rect",
        encoding={"x": "col", "y": "row"},
        mark_kwargs={"fill": "theme:contrast"},
        data_source=None,
    )
    chart = fm.Chart(df).mark_point().encode(x="col", y="row").layer(rect_layer)
    with pytest.raises(ValueError, match="theme:contrast"):
        chart.to_svg()


def test_sentinel_stroke_is_refused_as_an_invalid_color_on_any_mark() -> None:
    df = pl.DataFrame({"row": ["p0"], "col": ["q0"]})
    rect_layer = _Layer(
        mark="rect",
        encoding={"x": "col", "y": "row"},
        mark_kwargs={"stroke": "theme:contrast"},
        data_source=None,
    )
    chart = fm.Chart(df).mark_point().encode(x="col", y="row").layer(rect_layer)
    with pytest.raises(ValueError, match="theme:contrast"):
        chart.to_svg()


# ---------------------------------------------------------------------------
# A translucent theme background: the cell is judged at its own opaque
# color, and every label stays fully visible (never painted at a reduced
# alpha by inheriting the background's).
# ---------------------------------------------------------------------------


def test_a_transparent_theme_background_still_renders_every_label_visibly() -> None:
    df = pl.DataFrame(
        {"row": ["p0", "p1"], "col": ["q0", "q0"], "cat": ["Alpha", "Beta"], "label": ["L0", "L1"]}
    )
    color = fm.Color(
        "cat", scale=fm.OrdinalScale(domain=["Alpha", "Beta"], range=["#ffffff", "#000000"])
    )
    chart = (
        fm.Chart(df)
        .mark_rect()
        .encode(x="col", y="row", color=color)
        .theme(fm.Theme(font_color=_FONT, background_color="rgba(0,0,0,0)"))
        .layer(_contrast_text_layer(x="col", y="row", color="cat", text="label"))
    )
    svg = chart.to_svg()
    texts = {e["text"]: e["fill"] for e in figure_elements(svg) if e["tag"] == "text"}
    cells = {"L0": (0xFF, 0xFF, 0xFF), "L1": (0x00, 0x00, 0x00)}
    for label, cell_rgb in cells.items():
        r, g, b, alpha = _parse_paint(texts[label])
        assert alpha == 1.0, (
            f"{label}: a label must stay fully visible, not fade with a transparent background"
        )
        # The background is fully transparent, so it composites to the cell
        # itself (contrast 1.0) and can never win the first tier.
        expected = _expected_ink(cell_rgb, background_contrast_ratio=1.0)
        assert (r, g, b) == _parse_paint(expected)[:3], (
            f"{label}: ink must be judged against the cell's own opaque color, "
            "not a background whose underlay the chart cannot know"
        )
