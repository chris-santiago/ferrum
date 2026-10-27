"""``heatmap(square=True)`` draws square cells at the chart's normal or
caller-set size, for 0, 1 and n rows.

The cell grid's aspect (columns shown / rows shown) is fixed in the plot
area via ``CoordFixed``, the layout's one aspect owner (see
``ferrum-spec.md``'s "Masking and shape" section) -- no viewport is set,
so a square heatmap sizes like a non-square one. ``square=False`` never
touches this path at all.
"""

from __future__ import annotations

import json
import re
import warnings

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from ferrum.coord import CoordCartesian, CoordFixed
from tests._svg_rects import data_bar_rects, mark_layer, plot_rect, svg_elements

_WIDE = pl.DataFrame(
    {
        "row": ["r1", "r2", "r3"],
        "A": [1.0, 2.0, 3.0],
        "B": [4.0, 5.0, 6.0],
        "C": [7.0, 8.0, 9.0],
    }
)


def _cell_rects(svg: str) -> list[tuple[float, float, float, float]]:
    """Return ``(x, y, w, h)`` of every drawn heatmap cell rect."""
    return [
        (float(a["x"]), float(a["y"]), float(a["width"]), float(a["height"]))
        for a in data_bar_rects(svg)
    ]


def _assert_cells_square(
    svg: str, *, count: int | None = None
) -> list[tuple[float, float, float, float]]:
    rects = _cell_rects(svg)
    if count is not None:
        assert len(rects) == count, rects
    for x, y, w, h in rects:
        assert abs(w - h) < 1e-3, (w, h)
    return rects


class TestHeatmapSquareSizing:
    def test_1_col_0_rows_renders_an_empty_square_panel(self):
        """`test_viewport_doors.py`'s `test_heatmap_square_on_an_empty_table_draws_its_panel_empty` pins this case too."""
        df = pl.DataFrame({"a": pl.Series([], dtype=pl.Float64)})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svg = fm.heatmap(df, square=True).to_svg()
        messages = [str(w.message) for w in caught]
        assert any("its panel is drawn empty" in m for m in messages), messages
        assert re.search(r'width="640(\.0)?"', svg), svg
        assert re.search(r'height="480(\.0)?"', svg), svg
        x, y, w, h = plot_rect(svg)
        assert abs(w - h) < 1e-3, (w, h)

    def test_1x1_one_square_cell(self):
        df = pl.DataFrame({"row": ["r1"], "a": [1.0]})
        svg = fm.heatmap(df, annot=False, square=True).to_svg()
        _assert_cells_square(svg, count=1)

    def test_3x3_nine_square_cells_no_collapse_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svg = fm.heatmap(_WIDE, annot=False, square=True).to_svg()
        messages = [str(w.message) for w in caught]
        assert not any("collaps" in m or "too small" in m for m in messages), messages
        _assert_cells_square(svg, count=9)

    @pytest.mark.parametrize(
        "rows, cols",
        [(2, 5), (20, 3), (10, 10)],
        ids=["2rows_5cols", "20rows_3cols", "10rows_10cols"],
    )
    def test_every_shape_renders_square_cells(self, rows, cols):
        data: dict[str, list[str] | list[float]] = {"row": [f"r{i}" for i in range(rows)]}
        data.update({f"c{j}": [float(i * cols + j) for i in range(rows)] for j in range(cols)})
        df = pl.DataFrame(data)
        svg = fm.heatmap(df, annot=False, square=True).to_svg()
        _assert_cells_square(svg, count=rows * cols)

    def test_annot_true_cells_square_one_label_per_cell(self):
        svg = fm.heatmap(_WIDE, annot=True, square=True).to_svg()
        _assert_cells_square(svg, count=9)
        px, py, pw, ph = plot_rect(svg)
        # The label layer draws outside the cells' clip group, so scope to
        # <text> elements whose anchor sits inside the plot rect -- the
        # axis/title/colorbar chrome text sits outside it.
        labels = [
            attrs
            for attrs in re.finditer(r"<text([^>]*)>", svg)
            for attrs in [dict(re.findall(r'([\w-]+)="([^"]+)"', attrs.group(1)))]
            if px <= float(attrs["x"]) <= px + pw and py <= float(attrs["y"]) <= py + ph
        ]
        assert len(labels) == 9, labels

    @pytest.mark.parametrize("mask", ["upper", "lower"])
    def test_triangle_mask_kept_cells_square(self, mask):
        svg = fm.heatmap(_WIDE, annot=False, square=True, mask=mask).to_svg()
        _assert_cells_square(svg, count=6)

    def test_boolean_mask_hiding_a_whole_row_and_column_kept_cells_square(self):
        mask_arr = np.zeros((3, 3), dtype=bool)
        mask_arr[0, :] = True
        mask_arr[:, 0] = True
        svg = fm.heatmap(_WIDE, annot=False, square=True, mask=mask_arr).to_svg()
        _assert_cells_square(svg, count=4)

    def test_duplicate_row_labels_square_over_distinct_bands(self):
        df = pl.DataFrame({"row": ["r1", "r1", "r2"], "a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        svg = fm.heatmap(df, annot=False, square=True).to_svg()
        # The two "r1" rows draw into the same band and overlap; only that
        # every drawn cell is square is asserted here, not the overdraw.
        _assert_cells_square(svg, count=6)

    def test_null_row_label_nan_and_null_cells_square(self):
        df = pl.DataFrame(
            {"row": ["r1", None, "r3"], "a": [1.0, float("nan"), None], "b": [4.0, 5.0, 6.0]}
        )
        svg = fm.heatmap(df, annot=False, square=True).to_svg()
        _assert_cells_square(svg, count=6)

    def test_properties_dict_width_height_honored_and_square(self):
        svg = fm.heatmap(
            _WIDE, annot=False, square=True, properties={"width": 300, "height": 300}
        ).to_svg()
        assert re.search(r'width="300(\.0)?"', svg), svg
        assert re.search(r'height="300(\.0)?"', svg), svg
        _assert_cells_square(svg, count=9)

    def test_properties_call_width_height_honored_and_square(self):
        chart = fm.heatmap(_WIDE, annot=False, square=True).properties(width=300, height=300)
        svg = chart.to_svg()
        assert re.search(r'width="300(\.0)?"', svg), svg
        assert re.search(r'height="300(\.0)?"', svg), svg
        _assert_cells_square(svg, count=9)

    def test_90x90_caller_size_gives_the_same_collapse_warnings_as_square_false(self):
        def _warn(square: bool) -> list[str]:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fm.heatmap(
                    _WIDE,
                    annot=False,
                    square=square,
                    properties={"width": 90, "height": 90},
                ).to_svg()
            return sorted(str(w.message) for w in caught)

        assert _warn(True) == _warn(False)

    def test_caller_coord_after_square_wins(self):
        chart = fm.heatmap(_WIDE, annot=False, square=True).coord(CoordCartesian())
        assert isinstance(chart._coord, CoordCartesian)

    def test_square_false_never_touches_coord(self, monkeypatch):
        """square=False takes the exact pre-existing code path: `.coord()` is
        never called, so its output is unaffected by this change -- the
        strongest available pin short of a snapshot against an ephemeral
        baseline render."""
        from ferrum.chart import Chart

        def _boom(self, *_a, **_k):
            raise AssertionError("square=False must not set a coord")

        monkeypatch.setattr(Chart, "coord", _boom)
        fm.heatmap(_WIDE, annot=False, square=False).to_svg()

    def test_interactive_cells_square_in_the_scene(self):
        chart = fm.heatmap(_WIDE, annot=False, square=True)
        scene = json.loads(chart.interactive().scene_json)
        rects = [
            n
            for p in scene["panels"]
            for b in p["marks"]
            if b["kind"] == "rect"
            for n in b["nodes"]
        ]
        assert len(rects) == 9, rects
        for n in rects:
            assert abs(n["w"] - n["h"]) < 1e-3, n

    def test_zero_numeric_columns_still_refuses(self):
        df = pl.DataFrame({"a": ["x", "y", "z"]})
        with pytest.raises(ValueError, match="numeric"):
            fm.heatmap(df, square=True)


_EMPTY_WIDE = pl.DataFrame(
    {
        "row": pl.Series([], dtype=pl.Utf8),
        "A": pl.Series([], dtype=pl.Float64),
        "B": pl.Series([], dtype=pl.Float64),
        "C": pl.Series([], dtype=pl.Float64),
    }
)

_NO_ROWS_WARNING = "the chart has no rows to draw after its transforms; its panel is drawn empty"


def _warnings_of(fn) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    return [str(w.message) for w in caught]


class TestHeatmapMaskKeepsNoCellDrawsEmptyPanel:
    """A mask that keeps no cell -- every row of a zero-row table, or a
    boolean mask hiding every cell -- takes the ordinary zero-row
    draw-and-warn path on both `square=True` and `square=False`: the masked
    long frame carries its "column"/id columns even when empty, so
    `square=True` can read them rather than raising an untyped
    `polars.exceptions.ColumnNotFoundError` at `heatmap()` construction, and
    `square=False` reaches the same path rather than raising a typed
    `ValueError: unknown column 'column' ...` at render."""

    @pytest.mark.parametrize("square", [True, False])
    def test_all_true_boolean_mask_draws_an_empty_panel(self, square):
        mask = np.ones((3, 3), dtype=bool)
        messages = _warnings_of(
            lambda: fm.heatmap(_WIDE, annot=False, square=square, mask=mask).to_svg()
        )
        assert messages == [_NO_ROWS_WARNING], messages

    def test_all_true_boolean_mask_square_true_gives_a_square_empty_panel(self):
        mask = np.ones((3, 3), dtype=bool)
        svg = fm.heatmap(_WIDE, annot=False, square=True, mask=mask).to_svg()
        x, y, w, h = plot_rect(svg)
        assert abs(w - h) < 1e-3, (w, h)

    @pytest.mark.parametrize("mask", ["upper", "lower"])
    @pytest.mark.parametrize("square", [True, False])
    def test_zero_row_table_with_triangle_mask_draws_an_empty_panel(self, mask, square):
        messages = _warnings_of(
            lambda: fm.heatmap(_EMPTY_WIDE, annot=False, square=square, mask=mask).to_svg()
        )
        assert messages == [_NO_ROWS_WARNING], messages

    def test_zero_row_table_with_array_mask_draws_a_square_empty_panel(self):
        mask = np.zeros((0, 3), dtype=bool)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svg = fm.heatmap(_EMPTY_WIDE, annot=False, square=True, mask=mask).to_svg()
        messages = [str(w.message) for w in caught]
        assert messages == [_NO_ROWS_WARNING], messages
        x, y, w, h = plot_rect(svg)
        assert abs(w - h) < 1e-3, (w, h)

    def test_all_true_mask_with_annot_draws_an_empty_panel_one_warning(self):
        messages = _warnings_of(
            lambda: fm.heatmap(
                _WIDE, annot=True, square=True, mask=np.ones((3, 3), dtype=bool)
            ).to_svg()
        )
        assert messages == [_NO_ROWS_WARNING], messages

    def test_all_true_mask_with_center_keeps_its_own_warning_too(self):
        messages = _warnings_of(
            lambda: fm.heatmap(
                _WIDE, annot=False, square=True, mask=np.ones((3, 3), dtype=bool), center=0
            ).to_svg()
        )
        assert _NO_ROWS_WARNING in messages, messages
        assert any("center=0" in m for m in messages), messages
        assert len(messages) == 2, messages

    def test_all_true_mask_interactive_matches_the_unmasked_zero_row_warning_count(self):
        def _interactive_warning_count(build) -> int:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                chart = build()
                json.loads(chart.interactive().scene_json)
            return len(caught)

        masked = _interactive_warning_count(
            lambda: fm.heatmap(_WIDE, annot=False, square=True, mask=np.ones((3, 3), dtype=bool))
        )
        unmasked_empty = _interactive_warning_count(
            lambda: fm.heatmap(_EMPTY_WIDE, annot=False, square=True)
        )
        assert masked == unmasked_empty, (masked, unmasked_empty)

    def test_unmasked_zero_row_multi_column_square_rect(self):
        """`n_cols_drawn` reads 0 at zero rows (no band is drawn), so
        `max(..., 1)` gives ratio 1 for a 3-column table too, rather than
        `len(value_cols)` unconditionally drawing a 3:1 panel."""
        svg = fm.heatmap(_EMPTY_WIDE, annot=False, square=True).to_svg()
        x, y, w, h = plot_rect(svg)
        assert abs(w - h) < 1e-3, (w, h)


class TestHeatmapMaskAspectReadsDrawnBandsNotRawShape:
    """Every mask case in `TestHeatmapSquareSizing` keeps the drawn grid's
    aspect equal to the raw table's (the triangle masks keep every band;
    the row+column mask in that class hides one of each from a 3x3, so
    2/2 == 3/3): a raw-shape ratio (`len(value_cols) / tbl.num_rows`,
    ignoring the mask) would pass every one of them. These cells hide a
    whole row or column so the drawn and raw ratios differ, which a
    raw-shape count gets wrong -- confirmed by patching `_heatmap_build`'s
    ratio computation to `len(value_cols) / tbl.num_rows` and reading the
    drawn geometry: each case below draws non-square cells
    (127.744 x 191.616 or 191.616 x 127.744) or a non-square empty panel
    (514.928 x 171.643) under that mutation, where they are square here."""

    def test_hiding_one_whole_row_of_a_3x3_still_draws_square_cells(self):
        mask = np.zeros((3, 3), dtype=bool)
        mask[0, :] = True
        svg = fm.heatmap(_WIDE, annot=False, square=True, mask=mask).to_svg()
        _assert_cells_square(svg, count=6)

    def test_upper_mask_on_a_3row_x_2col_table_drops_a_whole_row(self):
        df = pl.DataFrame({"row": ["r1", "r2", "r3"], "A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
        svg = fm.heatmap(df, annot=False, square=True, mask="upper").to_svg()
        _assert_cells_square(svg, count=3)

    def test_hiding_one_whole_column_of_a_3x3_still_draws_square_cells(self):
        mask = np.zeros((3, 3), dtype=bool)
        mask[:, 0] = True
        svg = fm.heatmap(_WIDE, annot=False, square=True, mask=mask).to_svg()
        _assert_cells_square(svg, count=6)


# ---------------------------------------------------------------------------
# Layering doors: a square heatmap's own `CoordFixed` follows the one rule
# every per-layer coord follows -- the first member owns the panel's aspect,
# and a later member's own coord refuses in the voice of the call that built
# it (`heatmap`, not an internal layer index). The first half below pins
# alignment: `fm.layer(...)`'s own static composite route carries the
# leader's `CoordFixed` to the overlay's shared plot rect, so every other
# member's marks land on the leader's square cells rather than a plain,
# unaspected rect intersection. The second half pins the refusal: the same
# route's panel-property gate refuses a non-first member's own `CoordFixed`.
# ---------------------------------------------------------------------------

#: One point per heatmap cell, naming the exact categories `_heatmap_build`
#: binds internally (`"column"` for the value-column name, the frame's own
#: id column -- `"row"` on `_WIDE` -- for the row label), so a point chart
#: sharing those two field names shares the heatmap's own x/y scales.
_CELL_POINTS = pl.DataFrame(
    {
        "column": [c for c in ("A", "B", "C") for _ in range(3)],
        "row": ["r1", "r2", "r3"] * 3,
    }
)


def _points_on_every_cell():
    return fm.Chart(_CELL_POINTS).mark_point().encode(x="column", y="row")


def _circle_centers(svg: str) -> list[tuple[float, float]]:
    """Every drawn point's `(cx, cy)`, scoped to the plot area like `mark_layer`."""
    return [
        (float(a["cx"]), float(a["cy"]))
        for a in svg_elements(mark_layer(svg), "circle", lambda a: "cx" in a and "cy" in a)
    ]


def _cell_centers(svg: str) -> list[tuple[float, float]]:
    return [(x + w / 2, y + h / 2) for x, y, w, h in _cell_rects(svg)]


def _assert_points_on_cell_centers(
    points: list[tuple[float, float]], centers: list[tuple[float, float]]
) -> None:
    """Every point matches some remaining centre within 1e-3, one to one.

    Nearest-neighbour matching rather than a fixed pairing: emission order
    between a point batch and a rect batch is not a contract either side
    makes, so the ASSIGNMENT a correct render would draw is what is checked,
    not one particular ordering of it.
    """
    assert len(points) == len(centers), (points, centers)
    remaining = list(centers)
    for px, py in points:
        distances = [((px - cx) ** 2 + (py - cy) ** 2) ** 0.5 for cx, cy in remaining]
        i = distances.index(min(distances))
        assert distances[i] < 1e-3, (
            f"point ({px}, {py}) is not within 1e-3 of any remaining cell centre; "
            f"closest is {remaining[i]} at distance {distances[i]}"
        )
        remaining.pop(i)


class TestHeatmapFirstInAnOverlayAlignsOtherMembers:
    """A square heatmap placed FIRST draws its cells, and the overlay's
    shared plot area carries its fixed aspect to every other member, so a
    point chart naming the heatmap's own `column`/`row` categories lands
    exactly on the cells it names."""

    def test_plus_static(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            svg = (fm.heatmap(_WIDE, annot=False, square=True) + _points_on_every_cell()).to_svg()
        _assert_points_on_cell_centers(_circle_centers(svg), _cell_centers(svg))

    def test_layer_static(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            svg = fm.layer(
                fm.heatmap(_WIDE, annot=False, square=True), _points_on_every_cell()
            ).to_svg()
        _assert_points_on_cell_centers(_circle_centers(svg), _cell_centers(svg))

    def test_layer_interactive(self):
        layered = fm.layer(fm.heatmap(_WIDE, annot=False, square=True), _points_on_every_cell())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scene_json, _packed = layered._render_interactive()
        panel = json.loads(scene_json)["panels"][0]
        by_kind = {batch["kind"]: batch["nodes"] for batch in panel["marks"]}
        points = [(n["cx"], n["cy"]) for n in by_kind["point"]]
        centers = [(n["x"] + n["w"] / 2, n["y"] + n["h"] / 2) for n in by_kind["rect"]]
        _assert_points_on_cell_centers(points, centers)


def _assert_refuses_naming_heatmap_as_the_first_member(message: str) -> None:
    for token in ("heatmap", "fixed aspect ratio", "first member"):
        assert token in message, message
    assert "layer 1: fm.CoordFixed" not in message, (
        f"a figure-built layer must be named by its call, not an internal index; got {message!r}"
    )


class TestHeatmapNotFirstInAnOverlayRefuses:
    """A square heatmap placed AFTER another member cannot own the panel's
    aspect twice; it refuses in its own voice -- the call that built it, not
    an internal layer index -- on every layering door."""

    def test_plus_static(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError) as err:
                (_points_on_every_cell() + fm.heatmap(_WIDE, annot=False, square=True)).to_svg()
        _assert_refuses_naming_heatmap_as_the_first_member(str(err.value))

    def test_layer_static(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError) as err:
                fm.layer(
                    _points_on_every_cell(), fm.heatmap(_WIDE, annot=False, square=True)
                ).to_svg()
        _assert_refuses_naming_heatmap_as_the_first_member(str(err.value))

    def test_layer_interactive(self):
        layered = fm.layer(_points_on_every_cell(), fm.heatmap(_WIDE, annot=False, square=True))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError) as err:
                layered._render_interactive()
        _assert_refuses_naming_heatmap_as_the_first_member(str(err.value))


class TestOverlayPanelScopedCoordGeneral:
    """The mechanism `heatmap(square=True)` exercises, over a plain
    `fm.CoordFixed` rather than a figure call: it is the panel's own aspect,
    not one member's, on the static composite route."""

    def test_a_callers_fixed_aspect_leader_aligns_a_coordless_member(self):
        """`fm.layer(a.coord(CoordFixed(ratio=1)), b)`: the leader's fixed
        aspect shapes the group's one shared plot rect, which the coordless
        member draws against too, rather than each leaf correcting only its
        OWN drawing rect inside an unaspected shared intersection while the
        coordless member draws across the whole, wider intersection -- the
        same shape of misalignment `heatmap(square=True) + points` would
        have. Both members here bind the identical two data points, so a
        correct render draws one circle per point, not two apart.
        """
        df = pl.DataFrame({"x": [0.0, 10.0], "y": [0.0, 10.0]})
        a = fm.Chart(df).mark_point().encode(x="x", y="y").coord(CoordFixed(ratio=1.0))
        b = fm.Chart(df).mark_point().encode(x="x", y="y")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            svg = fm.layer(a, b).to_svg()
        # Each of the two leaves emits its own unfilled plot-area rect, so
        # `plot_rect` (which expects exactly one) does not apply here; both
        # must be the SAME aspect-corrected square, imposed once and shared.
        plot_rects = svg_elements(svg, "rect", lambda at: "fill" not in at)
        assert len(plot_rects) == 2, plot_rects
        widths_heights = {(float(r["width"]), float(r["height"])) for r in plot_rects}
        assert len(widths_heights) == 1, (
            f"both members must share one imposed plot rect; got {widths_heights}"
        )
        ((pw, ph),) = widths_heights
        assert abs(pw - ph) < 1e-3, (
            f"the leader's fixed aspect must shape the shared plot area; got {pw}x{ph}"
        )
        circles = {(round(cx, 3), round(cy, 3)) for cx, cy in _circle_centers(svg)}
        assert circles == {
            (round(x, 3), round(y, 3)) for x, y in _circle_centers((a + b).to_svg())
        }, (
            "a and b draw the same two points; aligned on the leader's aspect-"
            f"corrected shared rect they must land on the same two circles, got {circles}"
        )

    @pytest.mark.parametrize(
        "coord,name",
        [
            (CoordFixed(ratio=1.0), "CoordFixed"),
            (CoordCartesian(xlim=(0.0, 10.0)), "CoordCartesian"),
            (CoordCartesian(expand=False), "CoordCartesian"),
        ],
    )
    def test_a_callers_non_first_panel_scoped_coord_refuses_on_fm_layer_static(self, coord, name):
        df = pl.DataFrame({"x": [0.0, 10.0], "y": [0.0, 10.0]})
        lead = fm.Chart(df).mark_point().encode(x="x", y="y")
        member = fm.Chart(df).mark_point().encode(x="x", y="y").coord(coord)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError) as err:
                fm.layer(lead, member).to_svg()
        message = str(err.value)
        assert name in message, message
        assert "layer 1" in message, message
