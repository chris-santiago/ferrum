"""``Chart.__add__``'s per-layer ``data_source`` routing (#198).

Two rewrite sites inside ``__add__`` reassign an RHS layer's ``data_source``
so it reads the right batch after composition -- the column-collision
identity-transform site, and the "promote RHS's top-level transforms to a
named chart-level transform" site. Both used to do this UNCONDITIONALLY,
clobbering a layer's own pre-existing ``data_source`` whenever one was
already set.

That is harmless for a plain single-mark chart (its one layer's
``data_source`` starts ``None``), but a multi-output composite mark
(``mark_qq``, ``mark_violin``, ``mark_boxplot``, ...) ALWAYS gives its own
top-level transform(s) an explicit name and points every consuming layer's
``data_source`` at it, precisely so those layers can be read standalone --
e.g. ``mark_qq``'s reference-line layer reads
``"qq_main_qq_line"`` (the transform's SECONDARY output), not
``"qq_main"`` (its primary one).

**RED, measured on the pre-fix tree**: ``fm.Chart(df).mark_point().encode(x=
"x", y="y") + fm.Chart(df).mark_qq().encode(x="y")`` raised ``ValueError:
layer 2: encoding 'x' references unknown column 'qq_line_x_start' (not
present in that layer's data)`` -- the reference layer's ``data_source`` got
clobbered to a fresh ``_auto_...`` name that published only the QQ
transform's PRIMARY output, so its own encoding columns (which belong to the
secondary output) were never in that batch. ``qq + point`` (LHS already
layered, so ``__add__`` takes the untouched ``else`` branch) rendered fine --
only the RHS-promotion direction was broken, which is why the fix must
preserve each layer's OWN ``data_source`` at both rewrite sites rather than
special-casing ``qq``.

The two orders are NOT expected to produce byte-identical SVG: ``new._layers
= lhs_layers + rhs_layers`` always paints the LHS's marks first, so swapping
operand order swaps PAINT ORDER (verified: the two renders' `<circle>`
elements are the same 50 positions FROM EACH SIDE, just emitted in the
opposite 50-then-50 block order). What must match between the two orders --
and is asserted here as the discriminating claim -- is the GEOMETRY each
side contributes: the same circle positions and the same reference-line
endpoints, regardless of which side is LHS.

A second, closely related defect surfaced during this fix and is pinned
here too: two DIFFERENT composite-mark charts that happen to share a
transform name (every ``mark_qq()`` names its transform ``"qq_main"``
regardless of field) used to compose WITHOUT raising, but silently: Rust's
named-output map has no duplicate-name guard, so the second transform
processed overwrites the first's entry and every layer reading that name --
from EITHER side -- resolves to the transform that ran last. Composition now
refuses this case at ``+``-time instead of rendering corrupted output.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import ferrum as fm

from tests._svg_rects import svg_elements as _svg_elements

#: The QQ reference line is the one non-grid, non-axis ``<line>`` in a QQ
#: chart's SVG -- grid/axis lines use the theme's ``#d6d3d1``/``#6b7280``
#: strokes, the reference line is dashed with ``#9ca3af`` (see
#: ``marks/heavy_stat.py``'s ``desugar_qq`` reference-line mark style).
_QQ_LINE_STROKE = "#9ca3af"


def _circle_centers(svg: str) -> list[tuple[float, float]]:
    """``(cx, cy)`` for every rendered ``<circle>``, rounded for stability."""
    return sorted(
        (round(float(a["cx"]), 2), round(float(a["cy"]), 2))
        for a in _svg_elements(svg, "circle", lambda a: "cx" in a and "cy" in a)
    )


def _qq_reference_lines(svg: str) -> list[tuple[float, float, float, float]]:
    """Endpoints of every QQ reference-line segment, rounded for stability."""
    return sorted(
        (
            round(float(a["x1"]), 2),
            round(float(a["y1"]), 2),
            round(float(a["x2"]), 2),
            round(float(a["y2"]), 2),
        )
        for a in _svg_elements(svg, "line", lambda a: a.get("stroke") == _QQ_LINE_STROKE)
    )


@pytest.fixture
def qq_df() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    return pl.DataFrame({"x": rng.normal(size=50), "y": rng.normal(size=50)})


class TestQqComposesInEitherOrder:
    """The primary #198 regression: a plain chart plus a ``mark_qq`` chart."""

    def test_point_plus_qq_does_not_raise(self, qq_df):
        point = fm.Chart(qq_df).mark_point().encode(x="x", y="y")
        qq = fm.Chart(qq_df).mark_qq().encode(x="y")
        svg = (point + qq).to_svg()
        assert "<svg" in svg

    def test_qq_plus_point_still_renders(self, qq_df):
        """The direction that already worked pre-fix must keep working."""
        point = fm.Chart(qq_df).mark_point().encode(x="x", y="y")
        qq = fm.Chart(qq_df).mark_qq().encode(x="y")
        svg = (qq + point).to_svg()
        assert "<svg" in svg

    def test_both_orders_draw_the_same_qq_geometry(self, qq_df):
        """The QQ scatter + reference line are identical regardless of side.

        See the module docstring for why full-SVG byte equality is not the
        right claim (paint order swaps with operand order): the geometry
        each side contributes must match, not the document order.
        """
        point = fm.Chart(qq_df).mark_point().encode(x="x", y="y")
        qq = fm.Chart(qq_df).mark_qq().encode(x="y")
        svg_point_first = (point + qq).to_svg()
        svg_qq_first = (qq + point).to_svg()

        assert _circle_centers(svg_point_first) == _circle_centers(svg_qq_first)
        ref_point_first = _qq_reference_lines(svg_point_first)
        ref_qq_first = _qq_reference_lines(svg_qq_first)
        assert ref_point_first and ref_point_first == ref_qq_first

    def test_point_first_does_not_corrupt_the_qq_scatter(self, qq_df):
        """The pre-fix defect specifically dropped the QQ point layer's own
        columns; assert its 50 points are actually present (not the 50
        `point` circles duplicated, and not fewer than 50 after a partial
        failure).
        """
        point = fm.Chart(qq_df).mark_point().encode(x="x", y="y")
        qq = fm.Chart(qq_df).mark_qq().encode(x="y")
        svg = (point + qq).to_svg()
        centers = _circle_centers(svg)
        assert len(centers) == 100
        assert len(set(centers)) == 100, (
            "the plain point layer and the QQ point layer must draw distinct positions"
        )


class TestSelfComposition:
    """Composing a chart with itself must not trip the collision refusal."""

    def test_qq_plus_itself_still_renders(self, qq_df):
        q = fm.Chart(qq_df).mark_qq().encode(x="y")
        svg = (q + q).to_svg()
        assert "<svg" in svg
        # Two overlapping copies of the same 50-point scatter.
        assert len(_circle_centers(svg)) == 100


class TestUnroutableCollisionRefusal:
    """Two different composite marks sharing a transform name must refuse,
    not silently let one side's data overwrite the other's.
    """

    def test_two_different_qq_charts_refuse_instead_of_corrupting(self):
        rng = np.random.default_rng(0)
        df = pl.DataFrame({"a": rng.normal(size=50), "b": rng.normal(loc=5.0, size=50)})
        qq_a = fm.Chart(df).mark_qq().encode(x="a")
        qq_b = fm.Chart(df).mark_qq().encode(x="b")
        with pytest.raises(ValueError, match="qq_main"):
            qq_a + qq_b

    def test_refusal_names_the_layer_mark_and_output(self):
        rng = np.random.default_rng(0)
        df = pl.DataFrame({"a": rng.normal(size=50), "b": rng.normal(loc=5.0, size=50)})
        qq_a = fm.Chart(df).mark_qq().encode(x="a")
        qq_b = fm.Chart(df).mark_qq().encode(x="b")
        with pytest.raises(ValueError) as excinfo:
            qq_a + qq_b
        message = str(excinfo.value)
        assert "point" in message  # layer name
        assert "mark='point'" in message  # mark
        assert "qq_main" in message  # named output

    def test_two_qq_charts_on_the_same_field_do_not_refuse(self, qq_df):
        """Structurally-identical (dedup-safe) transforms are not a collision."""
        qq_1 = fm.Chart(qq_df).mark_qq().encode(x="y")
        qq_2 = fm.Chart(qq_df).mark_qq().encode(x="y")
        svg = (qq_1 + qq_2).to_svg()
        assert "<svg" in svg


class TestViolinInnerPointStillRoutesCorrectly:
    """A layer with NO ``data_source`` (violin's ``inner="point"`` overlay
    reads the raw, unsplit data) must keep reading the default batch, not
    get swept into the named-output preservation meant for its siblings.
    """

    @pytest.fixture
    def cat_df(self) -> pl.DataFrame:
        rng = np.random.default_rng(0)
        return pl.DataFrame(
            {
                "cat": ["a"] * 20 + ["b"] * 20,
                "val": np.concatenate([rng.normal(0, 1, 20), rng.normal(3, 1, 20)]),
            }
        )

    def test_violin_point_composes_with_a_plain_chart_either_order(self, cat_df):
        violin = fm.Chart(cat_df).mark_violin(inner="point").encode(x="cat:N", y="val:Q")
        rule = fm.Chart(cat_df).mark_rule().encode(x="cat:N", y="val:Q")

        svg_violin_first = (violin + rule).to_svg()
        svg_rule_first = (rule + violin).to_svg()
        assert "<svg" in svg_violin_first
        assert "<svg" in svg_rule_first
        # The violin's own per-row point overlay (20 categorical positions
        # worth of raw rows) renders in both orders.
        assert len(_circle_centers(svg_violin_first)) == len(cat_df)
        assert len(_circle_centers(svg_rule_first)) == len(cat_df)


class TestColumnOverlapRenameRoutesCorrectly:
    """The sibling rewrite site (diagonal-concat column-collision path):
    an RHS multi-output composite mark's already-routed layers must survive
    the collision-rename, not get swept into the Identity-transform
    redirection meant for a layer with no ``data_source`` of its own.
    """

    def test_qq_with_overlapping_column_name_still_renders_both_layers(self):
        rng = np.random.default_rng(0)
        lhs_df = pl.DataFrame({"x": rng.normal(size=30), "y": rng.normal(size=30)})
        # "y" collides with lhs_df's own column, but the qq chart itself
        # reads "z" -- forcing the overlap-rename branch to fire for a
        # column the QQ transform doesn't even touch.
        rhs_df = pl.DataFrame({"y": rng.normal(size=30), "z": rng.normal(size=30)})
        point = fm.Chart(lhs_df).mark_point().encode(x="x", y="y")
        qq = fm.Chart(rhs_df).mark_qq().encode(x="z")

        svg = (point + qq).to_svg()
        assert "<svg" in svg
        centers = _circle_centers(svg)
        assert len(centers) == 60  # 30 from `point`, 30 from qq's point layer
        assert _qq_reference_lines(svg), "the QQ reference line must still render"
