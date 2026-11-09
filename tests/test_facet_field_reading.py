"""``Chart.facet``'s ``field=``/``row=``/``col=`` read exactly as ``encode(facet_col=...)`` reads them.

Reading the argument raw, instead of through the same parse ``encode()``
uses, would reach the facet spec un-split: a shorthand string
(``"g:N"``) would fail with ``unknown column 'g:N'``, and a channel
object (``fm.FacetCol(...)``) would fail outright, since the facet spec
serializer cannot JSON-encode a channel. This module pins that
``.facet()`` reads through the parse instead, so shorthand and a channel
object both work, and that it draws byte-identically to
``encode(facet_col=...)`` when the two declare the same facet.
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm
from ferrum.encoding import FacetCol, FacetRow

_DF = pl.DataFrame(
    {
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "y": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "g": ["a", "a", "b", "b", "c", "c"],
        "h": ["p", "q", "p", "q", "p", "q"],
    }
)


def _plain() -> fm.Chart:
    return fm.Chart(_DF).mark_point().encode(x="x", y="y")


def test_a_shorthand_facet_field_is_read_as_encode_reads_it():
    """``facet(col="g:N")`` draws byte-identical to ``facet(col="g")`` --
    an unparsed shorthand string reaching the facet spec raw would instead
    fail with ``unknown column 'g:N' referenced by an encoding``.
    """
    bare = _plain().facet(col="g").to_svg()
    typed = _plain().facet(col="g:N").to_svg()
    assert bare == typed

    # The same rule on field= (wrap mode) and row= (lowered to wrap).
    assert _plain().facet(field="g").to_svg() == _plain().facet(field="g:O").to_svg()
    assert _plain().facet(row="g").to_svg() == _plain().facet(row="g:N").to_svg()


#: Columns whose own names the shorthand grammar cannot read as a plain
#: field (``"grp (k)"``), whose prefix names no aggregate (``"f(h)"``), or
#: whose suffix names no type (``"h:x"``).
_NON_SHORTHAND_DF = pl.DataFrame(
    {
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "y": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "grp (k)": ["a", "a", "b", "b", "c", "c"],
        "f(h)": ["p", "q", "p", "q", "p", "q"],
        "h:x": ["m", "n", "m", "n", "m", "n"],
    }
)


def _non_shorthand_plain() -> fm.Chart:
    return fm.Chart(_NON_SHORTHAND_DF).mark_point().encode(x="x", y="y")


@pytest.mark.parametrize("column", ["grp (k)", "f(h)", "h:x"])
def test_a_facet_field_naming_a_non_shorthand_column_draws_as_its_channel_object(column: str):
    """``Chart.facet(col=...)`` binds a column name the shorthand grammar
    cannot read (``"grp (k)"``), whose prefix names no aggregate
    (``"f(h)"``), or whose suffix names no type (``"h:x"``), as that
    exact column -- byte-identical to the same call with
    ``col=fm.FacetCol(<name>)``, and to
    ``encode(facet_col=fm.FacetCol(<name>))``.

    The ``col=`` string spelling alone does not discriminate against the
    pre-task tree here: that tree stores ``.facet()``'s raw string
    argument unparsed, which happens to land on the same field name
    either way. The channel-object spelling is what the pre-task tree
    cannot handle at all: it raises ``TypeError: Object of type FacetCol
    is not JSON serializable``, since ``.facet()`` there never learned to
    read a channel object, so comparing the string form against the
    object form through the SAME door is what a regression here breaks.
    """
    via_string = _non_shorthand_plain().facet(col=column).to_svg()
    via_facet_object = _non_shorthand_plain().facet(col=FacetCol(column)).to_svg()
    via_encode_object = _non_shorthand_plain().encode(facet_col=FacetCol(column)).to_svg()
    assert via_string == via_facet_object == via_encode_object


def test_lmplot_displot_catplot_col_name_the_same_non_shorthand_column():
    """``lmplot``/``displot``/``catplot``'s own ``col=`` keyword forwards a
    non-shorthand column name into ``Chart.facet`` the same way: each
    renders byte-identical to the same call with ``col=fm.FacetCol(name)``
    -- the comparison that discriminates against the pre-task tree (see
    the parametrized test above for why the bare string alone does not).
    """
    df = _NON_SHORTHAND_DF.rename({"x": "v"})
    bare_lmplot = fm.lmplot(df, x="v", y="v", ci=None, col="grp (k)").to_svg()
    object_lmplot = fm.lmplot(df, x="v", y="v", ci=None, col=FacetCol("grp (k)")).to_svg()
    assert bare_lmplot == object_lmplot

    bare_displot = fm.displot(df, x="v", kind="hist", col="grp (k)").to_svg()
    object_displot = fm.displot(df, x="v", kind="hist", col=FacetCol("grp (k)")).to_svg()
    assert bare_displot == object_displot

    cat_df = pl.DataFrame(
        {
            "cat": ["p", "q"] * 3,
            "v": _NON_SHORTHAND_DF["x"].to_list(),
            "grp (k)": _NON_SHORTHAND_DF["grp (k)"].to_list(),
        }
    )
    bare_catplot = fm.catplot(cat_df, x="cat", y="v", kind="bar", col="grp (k)").to_svg()
    object_catplot = fm.catplot(
        cat_df, x="cat", y="v", kind="bar", col=FacetCol("grp (k)")
    ).to_svg()
    assert bare_catplot == object_catplot


def test_a_shorthand_facet_col_renders_as_encode_facet_col_does():
    """``facet(col="g:N")`` and ``encode(facet_col="g:N")`` draw the same panels."""
    via_facet = _plain().facet(col="g:N").to_svg()
    via_encode = _plain().encode(facet_col="g:N").to_svg()
    assert via_facet == via_encode


def test_facet_col_and_encode_facet_col_are_byte_identical_on_a_plain_field():
    """Negative control: the untyped spelling already agreed; this must
    keep holding once both doors read through the same parse.
    """
    via_facet = _plain().facet(col="g").to_svg()
    via_encode = _plain().encode(facet_col="g").to_svg()
    assert via_facet == via_encode


def test_a_facet_col_channel_object_renders_with_its_title():
    """``facet(col=fm.FacetCol("g", title="G"))`` renders, with the strip
    titled ``"G"`` -- a channel object reaching the facet spec raw would
    instead raise ``TypeError: Object of type FacetCol is not JSON serializable``.
    """
    svg = _plain().facet(col=FacetCol("g", title="G")).to_svg()
    assert "<svg" in svg
    assert "G: a" in svg or "G:" in svg


def test_a_facet_row_channel_object_renders_with_its_title():
    svg = _plain().facet(row=FacetRow("g", title="G")).to_svg()
    assert "<svg" in svg
    assert "G:" in svg


def test_a_facet_col_shorthand_aggregate_is_refused_as_encode_facet_col_refuses_it():
    """``facet(col="mean(g)")`` refuses the same way ``encode(facet_col=...)``
    already does, rather than reaching the facet spec as a raw, unparsed
    aggregate string.
    """
    with pytest.raises(ValueError) as encode_err:
        _plain().encode(facet_col="mean(g)").to_svg()
    with pytest.raises(ValueError) as facet_err:
        _plain().facet(col="mean(g)").to_svg()
    assert str(facet_err.value) == str(encode_err.value)
    assert "FacetCol(aggregate='mean') is not supported" in str(facet_err.value)


@pytest.mark.parametrize(
    "kwarg,cls_name",
    [("field", "Facet"), ("row", "FacetRow"), ("col", "FacetCol")],
)
def test_a_non_binding_value_is_the_parses_typeerror(kwarg, cls_name):
    """``facet(col=5)`` (and ``field=``/``row=``) raises the parse's own
    ``TypeError`` naming the facet door, rather than leaking the facet
    spec serializer's own raw ``facet: invalid type: integer 5, expected
    a string``.
    """
    with pytest.raises(TypeError) as excinfo:
        _plain().facet(**{kwarg: 5}).to_svg()
    assert str(excinfo.value) == (
        f"Chart.facet({kwarg}=...) expects str, {cls_name} instance, or Repeat placeholder; got int"
    )


def test_facet_with_no_field_still_requires_one():
    """``facet(col=None)`` with no other field keeps the unchanged refusal."""
    with pytest.raises(ValueError, match=r"facet\(\) requires either `field=`, or `row=`/`col=`"):
        _plain().facet(col=None).to_svg()


def test_a_facet_channels_columns_sets_the_wrap_count():
    """``facet(field=fm.Facet("g", columns=2))`` sets the wrap count from
    the channel's own ``columns=``, byte-identical to ``facet("g", ncols=2)``
    and to ``encode(facet=fm.Facet("g", columns=2))``.
    """
    via_channel = _plain().facet(field=fm.Facet("g", columns=2)).to_svg()
    via_ncols = _plain().facet("g", ncols=2).to_svg()
    via_encode = _plain().encode(facet=fm.Facet("g", columns=2)).to_svg()
    assert via_channel == via_ncols == via_encode


def test_the_calls_own_ncols_wins_over_a_channels_columns():
    """The call's own ``ncols=`` outranks a ``fm.Facet``'s ``columns=``,
    the same precedence its own ``title=``/``sort=`` already have.
    """
    via_call_ncols = _plain().facet(fm.Facet("g", columns=2), ncols=3).to_svg()
    bare_ncols = _plain().facet("g", ncols=3).to_svg()
    assert via_call_ncols == bare_ncols


def test_the_calls_own_title_and_sort_win_over_the_channels():
    """The call's own ``title=``/``row_title=``/``sort=`` win over a
    channel object's, on every facet declaration shape.
    """
    wrap_title = _plain().facet(col=FacetCol("g", title="G"), title="H").to_svg()
    assert "H:" in wrap_title
    assert "G:" not in wrap_title

    grid_row_title = _plain().facet(row=FacetRow("h", title="G"), col="g", row_title="H").to_svg()
    assert "H:" in grid_row_title
    assert "G:" not in grid_row_title

    via_call_sort = (
        _plain().facet(col=FacetCol("g", sort=["c", "b", "a"]), sort=["a", "b", "c"]).to_svg()
    )
    bare_sort = _plain().facet(col="g", sort=["a", "b", "c"]).to_svg()
    assert via_call_sort == bare_sort


@pytest.mark.parametrize("kwarg", ["col", "row"])
def test_lmplot_col_and_row_are_read_the_same_way(kwarg):
    """``lmplot``'s figure-level ``col=``/``row=`` forward into ``Chart.facet``,
    so they share this exact reading rule.
    """
    df = _DF.rename({"x": "v", "y": "w"})
    bare = fm.lmplot(df, x="v", y="w", ci=None, **{kwarg: "g"}).to_svg()
    typed = fm.lmplot(df, x="v", y="w", ci=None, **{kwarg: "g:N"}).to_svg()
    assert bare == typed


def test_displot_and_catplot_col_are_read_the_same_way():
    """``displot``'s and ``catplot``'s own ``col=`` facet keyword forwards
    into ``Chart.facet`` the same way ``lmplot``'s does.
    """
    hist_df = _DF.rename({"x": "v"})
    bare_displot = fm.displot(hist_df, x="v", col="g", kind="hist").to_svg()
    typed_displot = fm.displot(hist_df, x="v", col="g:N", kind="hist").to_svg()
    assert bare_displot == typed_displot

    cat_df = pl.DataFrame({"cat": ["p", "q"] * 3, "v": _DF["x"].to_list(), "g": _DF["g"].to_list()})
    bare_catplot = fm.catplot(cat_df, x="cat", y="v", col="g", kind="bar").to_svg()
    typed_catplot = fm.catplot(cat_df, x="cat", y="v", col="g:N", kind="bar").to_svg()
    assert bare_catplot == typed_catplot


def test_a_grid_facet_keeps_each_dimensions_own_title():
    """In grid mode, ``col=``'s own channel title drives the column strip
    and ``row=``'s own channel title drives the row strip, independently.
    """
    svg = _plain().facet(row=FacetRow("h", title="Row"), col=FacetCol("g", title="Col")).to_svg()
    assert "Row:" in svg
    assert "Col:" in svg
