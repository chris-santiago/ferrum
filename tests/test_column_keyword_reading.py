"""A column-naming keyword (a figure's ``hue=``, ``residplot``'s ``color=`` column
test) reads shorthand exactly as ``encode()`` reads it, and a string the shorthand
grammar cannot read, or whose prefix names no aggregate, is the column it names --
the same reading :func:`ferrum.encoding._resolve.channel_for_column` gives every
such keyword (``ferrum-spec.md``'s "Column-keyword reading" paragraph). This is
deliberately LENIENT where ``encode()``, ``Layer(encoding=)`` and every figure's
``x=``/``y=`` stay STRICT: those doors refuse a string the shorthand grammar
cannot read, since there a non-shorthand string is never a valid binding.

A resolved column name an internal site already holds (a composite's own
``color_field``, a diagnostic mark's discriminator) is never re-parsed: it binds
as written, the baseline reading :func:`ferrum.marks._desugar_helpers.nominal_color_channel`
restores for its ~30 internal callers. Only the keyword doors themselves --
``regplot``/``lmplot``/``residplot`` (``tests/test_regression_keyword_reading.py``),
``displot``, ``catplot``, ``relplot(kind="line")``, ``jointplot``, ``pairplot``
and ``Chart.facet`` (``tests/test_facet_field_reading.py``) -- parse the caller's
own value, once, at the door.
"""

from __future__ import annotations

import hashlib
from typing import Callable

import polars as pl
import pytest

import ferrum as fm
from ferrum.encoding import Color


def _digest(svg: str) -> str:
    return hashlib.sha256(svg.encode()).hexdigest()


# ---------------------------------------------------------------------------
# A resolved column name an internal site already holds binds as written.
# ---------------------------------------------------------------------------

_CAT_DF = pl.DataFrame(
    {
        "temp (C)": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "f(h)": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "h:x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "cat": ["a", "a", "b", "b", "c", "c"],
        "v": [10.0, 20.0, 15.0, 25.0, 30.0, 12.0],
    }
)

#: One digest per composite mark, built with a chart-level `color=` naming a
#: resolved column whose own name happens to be valid shorthand -- a
#: `nominal_color_channel` caller hands it the resolved name, never the
#: raw keyword, so it must bind as written rather than being re-parsed.
_RESOLVED_COLOR_DIGESTS: dict[str, str] = {
    "boxplot": "cf298d2fad3c52d4ef6cc3a40b8fc750deab470fb7bd842d29bd9c119aaa6d93",
    "violin": "2861c3aac9c1b70b9e6e677311b883fd75f410da6ffa4bb7cd1f425cc13edec0",
    "boxen": "23263211634772704d676ca0774b5c369204af563474460ec8910cb64c444e74",
    "errorbar": "f684d4132192a7a22e1500239d09d7a922019b076ea55de2d7ee5e0bca3ed580",
    "errorband": "f684d4132192a7a22e1500239d09d7a922019b076ea55de2d7ee5e0bca3ed580",
    "swarm": "a18b101b1d286d0fccce150f239b38ea886c6d75b9e07ba7795944cea63498c2",
    "catplot_box_hue": "cf298d2fad3c52d4ef6cc3a40b8fc750deab470fb7bd842d29bd9c119aaa6d93",
    "fh": "024bf89044480d3ba31f58c605fc914ee148233039f661769a41651343121dbe",
    "hx": "b03f3dfd8a30e0def0fc04ca24d1f4156cac7763657e661300c660b2cda9052e",
}


def test_a_resolved_color_column_is_bound_as_written():
    """A composite mark's ``color=`` channel object, built from an already
    -resolved column name that happens to parse as shorthand (``"temp (C)"``,
    ``"f(h)"``, ``"h:x"``), binds that column as written: a resolved name is
    never re-read as a keyword's own shorthand, the ~30 internal callers'
    baseline reading. ``mark_violin``'s quartile/point layers re-pass the
    same resolved field through the same helper. Each render's bytes are
    pinned by digest, computed on the baseline tree.
    """
    builds: dict[str, fm.Chart] = {
        "boxplot": fm.Chart(_CAT_DF).mark_boxplot().encode(x="cat", y="v", color=Color("temp (C)")),
        "boxen": fm.Chart(_CAT_DF).mark_boxen().encode(x="cat", y="v", color=Color("temp (C)")),
        "errorbar": fm.Chart(_CAT_DF).mark_errorbar().encode(x="cat", y="v", color=Color("temp (C)")),
        "errorband": fm.Chart(_CAT_DF).mark_errorband().encode(x="cat", y="v", color=Color("temp (C)")),
        "swarm": fm.Chart(_CAT_DF).mark_swarm().encode(x="cat", y="v", color=Color("temp (C)")),
        "fh": fm.Chart(_CAT_DF).mark_boxplot().encode(x="cat", y="v", color=Color("f(h)")),
        "hx": fm.Chart(_CAT_DF).mark_boxplot().encode(x="cat", y="v", color=Color("h:x")),
    }
    for name, chart in builds.items():
        assert _digest(chart.to_svg()) == _RESOLVED_COLOR_DIGESTS[name], name

    # mark_violin warns about its undodged inner mark under a hue split --
    # expected here, unrelated to the column-reading rule under test.
    with pytest.warns(UserWarning, match="inner marks are drawn"):
        violin_svg = fm.Chart(_CAT_DF).mark_violin().encode(x="cat", y="v", color=Color("temp (C)")).to_svg()
    assert _digest(violin_svg) == _RESOLVED_COLOR_DIGESTS["violin"]

    catplot_svg = fm.catplot(_CAT_DF, x="cat", y="v", kind="box", hue=Color("temp (C)")).to_svg()
    assert _digest(catplot_svg) == _RESOLVED_COLOR_DIGESTS["catplot_box_hue"]


# ---------------------------------------------------------------------------
# A figure's hue= keyword reads a column-naming string the way encode()
# reads a column-keyword value, not the way it reads x=/y=.
# ---------------------------------------------------------------------------

_HUE_DF = pl.DataFrame(
    {
        "grp (k)": ["a", "a", "b", "b", "c", "c"],
        "f(h)": ["a", "a", "b", "b", "c", "c"],
        "log(price)": ["a", "a", "b", "b", "c", "c"],
        "h:x": ["a", "a", "b", "b", "c", "c"],
        "g": ["a", "a", "b", "b", "c", "c"],
        "cat": ["x", "y", "x", "y", "x", "y"],
        "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "w": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
        "k": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0],
    }
)

#: One builder per figure door that reads a grouping ``hue=``, each taking
#: the ``hue=`` value alone so every case below can run it unchanged.
_HUE_DOOR_BUILDS: dict[str, Callable[[object], fm.Chart]] = {
    "regplot": lambda hue: fm.regplot(_HUE_DF, x="v", y="w", hue=hue, ci=None),
    "lmplot": lambda hue: fm.lmplot(_HUE_DF, x="v", y="w", hue=hue, ci=None),
    "catplot": lambda hue: fm.catplot(_HUE_DF, x="cat", y="v", hue=hue, kind="strip"),
    "displot": lambda hue: fm.displot(_HUE_DF, x="v", hue=hue, kind="hist"),
    "relplot_line": lambda hue: fm.relplot(_HUE_DF, x="v", y="w", hue=hue, kind="line"),
    "jointplot": lambda hue: fm.jointplot(_HUE_DF, x="v", y="w", hue=hue),
    "pairplot": lambda hue: fm.pairplot(_HUE_DF, vars=["v", "w"], hue=hue),
}

#: The non-shorthand-spelling test's own builders: every _HUE_DOOR_BUILDS
#: door, plus the decision case's second kind for catplot (box, beside
#: strip) and displot (kde, beside hist) -- each kind's hue consumer is
#: its own code path, so both need the same coverage.
_NON_SHORTHAND_HUE_BUILDS: dict[str, Callable[[object], fm.Chart]] = {
    **_HUE_DOOR_BUILDS,
    "catplot_box": lambda hue: fm.catplot(_HUE_DF, x="cat", y="v", hue=hue, kind="box"),
    "displot_kde": lambda hue: fm.displot(_HUE_DF, x="v", hue=hue, kind="kde"),
}


@pytest.mark.parametrize("door", list(_NON_SHORTHAND_HUE_BUILDS))
@pytest.mark.parametrize("spelling", ["grp (k)", "f(h)", "log(price)", "h:x"])
def test_a_hue_naming_a_non_shorthand_column_draws_as_its_channel_object(door: str, spelling: str):
    """A column name the shorthand grammar cannot read (``"grp (k)"``), or
    whose prefix names no aggregate (``"f(h)"``, ``"log(price)"``), or
    whose suffix names no type (``"h:x"``), draws byte-identical to
    ``hue=fm.Color(<name>)`` on every door that reads a grouping ``hue=``,
    rather than failing the shorthand parse raw (``could not parse
    shorthand: '...'``). ``"f(h)"`` is the one spelling that also
    exercises ``accepts_aggregate`` at the door: its prefix reads as a
    call, but ``"f"`` names no aggregate, so it still binds as a column.
    """
    build = _NON_SHORTHAND_HUE_BUILDS[door]
    via_string = build(spelling).to_svg()
    via_object = build(Color(spelling)).to_svg()
    assert via_string == via_object


def test_a_shorthand_hue_reaches_every_consumer_as_its_channel():
    """``hue="g:N"`` reaches every downstream consumer of the parsed hue
    (a groupby, a ``Dodge(by=)``) as the SAME channel the door parsed, not
    the raw, unparsed keyword string a groupby lookup would otherwise
    fail on (``groupby column 'g:N' not found``).
    """
    kde_shorthand = fm.displot(_HUE_DF, x="v", hue="g:N", kind="kde").to_svg()
    kde_bare = fm.displot(_HUE_DF, x="v", hue="g", kind="kde").to_svg()
    kde_object = fm.displot(_HUE_DF, x="v", hue=Color("g", type="N"), kind="kde").to_svg()
    assert kde_shorthand == kde_bare == kde_object

    hist_kde_overlay_shorthand = fm.displot(_HUE_DF, x="v", hue="g:N", kde=True, kind="hist").to_svg()
    hist_kde_overlay_bare = fm.displot(_HUE_DF, x="v", hue="g", kde=True, kind="hist").to_svg()
    assert hist_kde_overlay_shorthand == hist_kde_overlay_bare

    for kind in ("kde", "reg", "hist"):
        shorthand = fm.jointplot(_HUE_DF, x="v", y="w", hue="g:N", kind=kind).to_svg()
        bare = fm.jointplot(_HUE_DF, x="v", y="w", hue="g", kind=kind).to_svg()
        assert shorthand == bare, kind

    pairplot_shorthand = fm.pairplot(_HUE_DF, vars=["v", "w"], hue="g:N", kind="reg").to_svg()
    pairplot_bare = fm.pairplot(_HUE_DF, vars=["v", "w"], hue="g", kind="reg").to_svg()
    assert pairplot_shorthand == pairplot_bare

    catplot_shorthand = fm.catplot(
        _HUE_DF, x="cat", y="v", hue="g:N", kind="bar", dodge=True
    ).to_svg()
    catplot_bare = fm.catplot(_HUE_DF, x="cat", y="v", hue="g", kind="bar", dodge=True).to_svg()
    assert catplot_shorthand == catplot_bare


@pytest.mark.parametrize(
    "door,build",
    [
        ("catplot_strip", lambda hue: fm.catplot(_HUE_DF, x="cat", y="v", hue=hue, kind="strip")),
        ("displot_hist", lambda hue: fm.displot(_HUE_DF, x="v", hue=hue, kind="hist")),
        ("relplot_line", lambda hue: fm.relplot(_HUE_DF, x="v", y="w", hue=hue, kind="line")),
        (
            "jointplot_scatter",
            lambda hue: fm.jointplot(_HUE_DF, x="v", y="w", hue=hue, kind="scatter"),
        ),
        (
            "pairplot_scatter",
            lambda hue: fm.pairplot(
                _HUE_DF, vars=["v", "w"], hue=hue, kind="scatter", diag_kind="hist"
            ),
        ),
    ],
)
def test_a_typed_shorthand_hue_draws_as_its_channel_object(
    door: str, build: Callable[[object], fm.Chart]
) -> None:
    """``hue="k:Q"`` keeps its declared type (a continuous colorbar), drawn
    byte-identical to ``hue=fm.Color("k", type="Q")`` -- not forced
    nominal the way an undeclared-type ``hue=`` is.
    """
    via_shorthand = build("k:Q").to_svg()
    via_object = build(Color("k", type="Q")).to_svg()
    assert via_shorthand == via_object


def test_catplot_typed_shorthand_hue_meets_the_hue_order_refusal():
    """``catplot(hue="k:Q", hue_order=...)`` meets the SAME refusal
    ``hue=fm.Color("k", type="Q")`` does -- the declared ``Q`` reaches the
    ``hue_order=`` check through the shorthand parse now, rather than
    being silently dropped and drawn nominal.
    """
    with pytest.raises(ValueError) as excinfo:
        fm.catplot(_HUE_DF, x="cat", y="v", hue="k:Q", hue_order=["1.0", "2.0"], kind="strip").to_svg()
    assert str(excinfo.value) == (
        "catplot(hue=fm.Color('k', type_=...), hue_order=...): hue_order= cannot order a "
        "quantitative hue channel -- there is no discrete legend order for a continuous "
        "scale to apply. Drop hue_order=, or type the channel nominal/ordinal (e.g. "
        "type_='nominal') so hue_order= has levels to order."
    )


@pytest.mark.parametrize("door", list(_HUE_DOOR_BUILDS))
@pytest.mark.parametrize("hue_value", ["mean(k)", "count()", "distinct(g)"])
def test_an_aggregate_on_a_grouping_hue_is_refused(door: str, hue_value: str):
    """An aggregate shorthand hue (and an aggregating channel object) is
    refused on every grouping-hue door: ``hue=`` names the column the
    chart groups its rows by, not a value to compute, so it is never
    drawn as a 24-category legend of computed values or left to fail raw
    in the statistic transform.
    """
    build = _HUE_DOOR_BUILDS[door]
    with pytest.raises(ValueError) as excinfo:
        build(hue_value).to_svg()
    written_field, written_agg = {
        "mean(k)": ("'k'", "'mean'"),
        "count()": (None, "'count'"),
        "distinct(g)": ("'g'", "'distinct'"),
    }[hue_value]
    if written_field is None:
        remedy = "Pass the column to group by as hue=."
        written = f"Color(aggregate={written_agg})"
    else:
        remedy = f"Drop the aggregate (hue={written_field})."
        written = f"Color({written_field}, aggregate={written_agg})"
    assert str(excinfo.value) == (
        f"{door.replace('_line', '')}(hue={written}) is not supported: hue= names the "
        f"column the chart groups its rows by, so aggregate={written_agg} would group "
        f"them by a computed value instead. {remedy}"
    )


@pytest.mark.parametrize("door", list(_HUE_DOOR_BUILDS))
def test_an_aggregating_hue_channel_object_is_refused_the_same_way(door: str):
    """The same refusal reaches a pre-built ``fm.Color(..., aggregate=...)``
    object, not only an aggregate shorthand string.
    """
    build = _HUE_DOOR_BUILDS[door]
    with pytest.raises(ValueError) as excinfo:
        build(Color("k", aggregate="mean")).to_svg()
    assert str(excinfo.value) == (
        f"{door.replace('_line', '')}(hue=Color('k', aggregate='mean')) is not supported: "
        "hue= names the column the chart groups its rows by, so aggregate='mean' would "
        "group them by a computed value instead. Drop the aggregate (hue='k')."
    )


def test_relplot_scatter_hue_stays_an_encode_binding():
    """``relplot(kind="scatter")`` is the one deliberate carve-out: its
    ``hue=`` binds straight to ``color=``, unparsed through the grouping
    -hue door, keeping its seaborn-parity dtype-driven inference. An
    aggregate shorthand renders (it is read by ``encode()``'s own binding,
    not refused as a grouping hue would be), pinned by digest; a
    non-shorthand column name is refused by ``encode()``'s strict grammar.
    """
    svg = fm.relplot(_HUE_DF, x="v", y="w", hue="mean(k)", kind="scatter").to_svg()
    assert _digest(svg) == "62e650d629da18b55e51ee31bf4a556cd7caf50c96a16fd142ca42743d2ed12d"

    with pytest.raises(ValueError) as excinfo:
        fm.relplot(_HUE_DF, x="v", y="w", hue="grp (k)", kind="scatter").to_svg()
    assert str(excinfo.value) == "could not parse shorthand: 'grp (k)'"


@pytest.mark.parametrize("door", list(_HUE_DOOR_BUILDS))
def test_a_non_binding_hue_names_its_door(door: str):
    """``hue=5`` raises the parse's own ``TypeError`` naming the door that
    received it, rather than a raw ``AttributeError`` or another door's
    own name.
    """
    build = _HUE_DOOR_BUILDS[door]
    with pytest.raises(TypeError) as excinfo:
        build(5).to_svg()
    assert str(excinfo.value) == (
        f"{door.replace('_line', '')}(hue=...) expects str, Color instance, or Repeat "
        "placeholder; got int"
    )


# ---------------------------------------------------------------------------
# residplot's color= reads a column-naming string the same lenient way,
# before falling through to its own column-or-literal-color dispatch.
# ---------------------------------------------------------------------------

_RESIDPLOT_DF = pl.DataFrame(
    {
        # Two rows per "temp (C)" value: a one-row group degrades the fit
        # (an unrelated warning this module does not test).
        "temp (C)": [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0],
        "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "w": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "g": ["a", "b", "a", "b", "a", "b", "a", "b"],
        "red": ["x", "y", "x", "y", "x", "y", "x", "y"],
        "k": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
    }
)


def test_residplot_color_column_is_bound_as_written():
    """``color="temp (C)"`` -- a column name the shorthand grammar cannot
    read -- groups by that column, byte-identical to
    ``color=fm.Color("temp (C)")``.
    """
    via_string = fm.residplot(_RESIDPLOT_DF, x="v", y="w", color="temp (C)").to_svg()
    via_object = fm.residplot(_RESIDPLOT_DF, x="v", y="w", color=Color("temp (C)")).to_svg()
    assert via_string == via_object


@pytest.mark.parametrize("color_value", ["mean(k)", "distinct(g)", "mean(g)", "grp (kk)"])
def test_residplot_color_that_is_neither_a_column_nor_a_color_is_refused_in_its_voice(
    color_value: str,
):
    """An aggregate shorthand, or any other string that is neither a
    column nor a parseable color, is refused with ``residplot``'s own
    baseline "not a column ... and not a parseable color" text, rather
    than drawing with a stray internal warning or failing raw inside the
    statistic transform.
    """
    with pytest.raises(ValueError) as excinfo:
        fm.residplot(_RESIDPLOT_DF, x="v", y="w", color=color_value).to_svg()
    assert str(excinfo.value) == (
        f"residplot(color='{color_value}'): not a column of the data and not a "
        f"parseable color. As a column name it would group the data, but no column "
        f"'{color_value}' exists; as a literal color it would paint the marks, but the "
        f"color parser refused it (invalid color '{color_value}': expected a CSS color "
        "name, #rrggbb[aa]/#rgb[a] hex, or rgb()/rgba())."
    )


def test_residplot_literal_colors_still_paint():
    """A literal color string or tuple still paints the points --
    unaffected by the shorthand-reading fix. (A column literally named
    ``"red"`` still groups rather than paints, column membership winning
    over the literal-color reading; that case is already pinned by
    ``tests/test_residplot_color.py::TestColorDispatch::test_colour_named_column_groups_rather_than_paints``.)
    """
    no_red_column = _RESIDPLOT_DF.drop("red")
    by_name = fm.residplot(no_red_column, x="v", y="w", color="red").to_svg()
    by_rgb_fn = fm.residplot(no_red_column, x="v", y="w", color="rgb(255,0,0)").to_svg()
    by_hex = fm.residplot(no_red_column, x="v", y="w", color="#ff0000").to_svg()
    by_tuple = fm.residplot(no_red_column, x="v", y="w", color=(255, 0, 0)).to_svg()
    assert by_name == by_rgb_fn == by_hex == by_tuple


# ---------------------------------------------------------------------------
# pairplot's markers= Shape reads the parsed hue channel's field.
# ---------------------------------------------------------------------------


def test_pairplot_markers_read_the_hue_channels_field():
    """``pairplot(hue=..., markers=[...])`` reads the hue channel's field
    (and, when it declares one, its type) to build the ``Shape`` channel,
    for a bare string, a shorthand string and a channel object alike --
    all three render the same panels, the bare string's own bytes pinned
    against the baseline (the one spelling the markers Shape already
    read correctly from the raw, unparsed hue; a channel object there
    raises ``Shape: field must be str ... got Color``).
    """
    df = _HUE_DF
    via_string = fm.pairplot(df, vars=["v", "w"], hue="g", markers=["circle", "square"]).to_svg()
    via_shorthand = fm.pairplot(
        df, vars=["v", "w"], hue="g:N", markers=["circle", "square"]
    ).to_svg()
    via_object = fm.pairplot(
        df, vars=["v", "w"], hue=Color("g"), markers=["circle", "square"]
    ).to_svg()
    assert via_string == via_shorthand == via_object
    assert _digest(via_string) == "e76bb20a55793e5879f262c6d0d8791489a1ba629afbc0aa94bcb82f2bc0f9e2"


# ---------------------------------------------------------------------------
# encode(), Layer(encoding=) and every figure's x=/y= keep the strict
# shorthand grammar: a non-shorthand string is refused, not bound literally.
# ---------------------------------------------------------------------------


def test_encode_doors_keep_the_strict_grammar():
    """``encode()``, ``Layer(encoding=)`` and a figure's own ``x=``/``y=``
    (including through ``**encode_kwargs``) refuse a column name the
    shorthand grammar cannot read: only the column-naming keyword doors
    above read such a string leniently.
    """
    df = _RESIDPLOT_DF

    with pytest.raises(ValueError) as excinfo:
        fm.Chart(df).mark_point().encode(x="v", y="w", color="grp (k)").to_svg()
    assert str(excinfo.value) == "could not parse shorthand: 'grp (k)'"

    with pytest.raises(ValueError) as excinfo:
        fm.Layer(encoding={"color": "grp (k)"})
    assert str(excinfo.value) == "could not parse shorthand: 'grp (k)'"

    with pytest.raises(ValueError) as excinfo:
        fm.regplot(df, x="Mass (g)", y="w", ci=None).to_svg()
    assert str(excinfo.value) == "could not parse shorthand: 'Mass (g)'"

    with pytest.raises(ValueError) as excinfo:
        fm.regplot(df, x="v", y="w", ci=None, color="grp (k)").to_svg()
    assert str(excinfo.value) == "could not parse shorthand: 'grp (k)'"


# ---------------------------------------------------------------------------
# The reading predicate: a string is shorthand only when its aggregate,
# when it has one, is in the published aggregate vocabulary.
# ---------------------------------------------------------------------------


def test_a_string_is_shorthand_only_when_its_aggregate_is_known():
    """``channel_for_column`` reads a fieldless-prefix string as a plain
    column name when the grammar cannot read it as a call at all, and
    reads a recognized aggregate call into its field and aggregate --
    the one predicate every column-keyword door shares.

    Imports ``channel_for_column`` locally, not at module level: a tree
    that does not yet define this symbol would fail to import the whole
    module, blocking every other test in it from even collecting, let
    alone running.
    """
    from ferrum.encoding._resolve import channel_for_column

    plain = channel_for_column("color", "log(price)", declare="test(color=...)")
    assert plain.field == "log(price)"
    assert plain._kwargs.get("aggregate") is None

    with_field = channel_for_column("color", "mean(k)", declare="test(color=...)")
    assert with_field.field == "k"
    assert with_field._kwargs.get("aggregate") == "mean"

    fieldless = channel_for_column("color", "count()", declare="test(color=...)")
    assert fieldless.field is None
    assert fieldless._kwargs.get("aggregate") == "count"
