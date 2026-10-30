"""A statistic layer's explicit-domain warning names the axis its chart shows.

A statistic layer draws columns a statistic computed (a box's quartiles, a
density grid, a fitted line), never the caller's own, so each of its
positional bindings carries the name the chart's axis shows: the caller's
``title=``, else the field they wrote, else the statistic's own displayed
name. Rust tells the two kinds of layer apart: under an explicit domain a
statistic layer warns in its entry's voice (``"<entry>: x scale: ..."``) and
a primitive layer in the row voice, which names its own field and is not
compared here.

The oracle is that predicate. For every axis whose scale is continuous, the
case renders under a domain that leaves out every row, and every
entry-voiced message on that axis must quote the axis title the unlimited
render draws. Where the build binds an axis to a column of its own data, the
rendered axis title is the chart-level binding's title, else layer 0's (a
statistic's own axis name), else that column.

Population: every figure and composite-mark build of
``tests/test_figure_regroup_population.py`` (whose completeness tests fail on
a figure or mark with no build), plus variants for the kwargs that add or
change a statistic layer, plus a caller-title door: each build whose
chart-level x/y binds a column of its data, and each figure variant whose
``x=``/``y=`` takes a channel object, is rebuilt with that channel titled,
and every entry-voiced message and the axis title must name the caller's
title.

Self-check: a case whose wire spec carries a statistics marker must compare
at least one message, unless :data:`_NO_CONTINUOUS_STATISTIC_AXIS` states why
every statistic layer it builds draws only on categorical axes (each listed
case is asserted to still compare none, so a stale entry fails); a case with
no statistics marker must compare none.
"""

from __future__ import annotations

import json
import re
import warnings
from typing import Any, Callable, NamedTuple

import pytest

import ferrum as fm
from ferrum._layer_transforms import _STATISTICS_MARKER_WIRE
from ferrum.encoding import X, Y
from ferrum.encoding.base import ChannelBase, drawn_title
from tests._svg_extents import x_axis_titles, y_axis_titles
from tests.test_figure_regroup_population import (
    _DIAGNOSTIC_MARK_BUILDS,
    _FIGURE_BUILDS,
    _MARK_CASES,
    _as_polars,
    _df,
    _established_field,
)

CALLER_X = "Caller x"
CALLER_Y = "Caller y"

#: A domain no fixture value reaches, so every placed row is left out.
_FAR_DOMAIN = (1e12, 1e12 + 1)

#: A domain-left-out message, in either voice: the statistic voice prefixes
#: the entry and counts marks, the row voice counts rows.
_DOMAIN_MESSAGE = re.compile(
    r"^(?:(?P<entry>[^:]+): )?(?P<channel>[a-z0-9_]+) scale: \d+ (?:row|mark)\(s\) of "
    r"'(?P<name>.*?)' (?:fall outside|cross|name a category|reach a category)"
)


class _Case(NamedTuple):
    build: Callable[[], Any]
    #: The ``title=`` the caller wrote on each axis this case titles.
    caller_titles: dict[str, str]


class _Channels:
    """The x/y spellings a figure variant is built with, recording which it used.

    Plain spellings bind the bare field; titled ones bind ``fm.X``/``fm.Y``
    carrying the caller's title, so a variant written once yields both the
    plain case and its caller-title door.
    """

    def __init__(self, *, titled: bool) -> None:
        self.titled = titled
        self.used: dict[str, str] = {}

    def x(self, field: str) -> Any:
        if not self.titled:
            return field
        self.used["x"] = CALLER_X
        return X(field, title=CALLER_X)

    def y(self, field: str) -> Any:
        if not self.titled:
            return field
        self.used["y"] = CALLER_Y
        return Y(field, title=CALLER_Y)


#: Figure builds for the kwargs that add or change a statistic layer, each
#: written over the x/y spellings so the caller-title door rebuilds it titled.
_TITLE_VARIANTS: dict[str, Callable[[_Channels], Any]] = {
    **{
        f"displot[hist,hue={multiple},stat={stat},kde]": (
            lambda ch, multiple=multiple, stat=stat: fm.displot(
                _df(), x=ch.x("v"), hue="h", multiple=multiple, stat=stat, kde=True
            )
        )
        for multiple in ("layer", "dodge", "stack", "fill")
        for stat in ("count", "density")
    },
    "displot[ecdf,kde]": lambda ch: fm.displot(_df(), x=ch.x("v"), kind="ecdf", kde=True),
    "displot[rug,kde]": lambda ch: fm.displot(_df(), x=ch.x("v"), kind="rug", kde=True),
    **{
        f"catplot[{kind}]": (
            lambda ch, kind=kind: fm.catplot(_df(), x=ch.x("g"), y=ch.y("v"), kind=kind)
        )
        for kind in ("box", "violin", "boxen", "bar", "point", "strip")
    },
    "regplot[metrics]": lambda ch: fm.regplot(_df(), x=ch.x("v"), y=ch.y("w"), show_metrics=True),
    "lmplot": lambda ch: fm.lmplot(_df(), x=ch.x("v"), y=ch.y("w")),
    "residplot": lambda ch: fm.residplot(_df(), x=ch.x("v"), y=ch.y("w")),
    "residplot[lowess]": lambda ch: fm.residplot(_df(), x=ch.x("v"), y=ch.y("w"), lowess=True),
}

#: Figure variants with no x=/y= spelling to title.
_FIXED_FIGURE_VARIANTS: dict[str, Callable[[], Any]] = {
    "displot[titled x,kde]": lambda: fm.displot(_df(), x=X("v", title="Value USD"), kde=True),
    "heatmap[annot]": lambda: fm.heatmap(_df().select("v", "w"), annot=True),
}

#: Composite-mark builds beside the regroup module's, for the inner variants,
#: the band-plus-metrics fit and the band borders that add statistic layers.
_MARK_TITLE_VARIANTS: dict[str, Callable[[], Any]] = {
    "mark_errorband[borders]": lambda: (
        fm.Chart(_df()).mark_errorband(borders=True).encode(x="g", y="v")
    ),
    "mark_violin[inner=box]": lambda: fm.Chart(_df()).mark_violin(inner="box").encode(x="g", y="v"),
    "mark_violin[inner=quartile]": lambda: (
        fm.Chart(_df()).mark_violin(inner="quartile").encode(x="g", y="v")
    ),
    "mark_smooth[ci,metrics]": lambda: (
        fm.Chart(_df()).mark_smooth(ci=0.95, show_metrics=True).encode(x="v", y="w")
    ),
}

#: Axes that show a statistic's or a figure's own axis name in place of the
#: column the build binds there (``mark_qq``'s x binds the sample column but
#: draws theoretical quantiles). Each entry is asserted to still differ from
#: that column, so a stale entry fails.
_FIXED_AXIS_TITLES: dict[tuple[str, str], str] = {
    ("mark_alpha_selection", "y"): "score",
    ("mark_learning_curve", "x"): "training samples",
    ("mark_learning_curve", "y"): "score",
    ("mark_pca_scree", "x"): "Component",
    ("mark_pca_scree", "y"): "Explained variance",
    ("mark_qq", "x"): "Theoretical Quantiles",
    ("mark_rank2d", "x"): "Feature",
    ("mark_rank2d", "y"): "Feature",
    ("mark_shap_bar", "x"): "Mean |SHAP value|",
    ("mark_shap_waterfall", "x"): "SHAP value",
    ("mark_silhouette", "x"): "silhouette coefficient",
    ("mark_silhouette", "y"): "sample",
    ("mark_validation_curve", "y"): "score",
    ("pca_scree_chart", "x"): "Component",
    ("pca_scree_chart", "y"): "Explained variance",
    ("rank2d_chart", "x"): "Feature",
    ("rank2d_chart", "y"): "Feature",
    ("rank_chart", "x"): "Feature",
    ("rank_chart", "y"): "Feature",
    ("shap_bar_chart", "x"): "Mean |SHAP value|",
    ("shap_waterfall_chart", "x"): "SHAP value",
    ("silhouette_chart", "x"): "silhouette coefficient",
    ("silhouette_chart", "y"): "sample",
    ("validation_curve_chart", "x"): "alpha",
}

#: Cases with a statistics marker whose every statistic layer draws only on
#: categorical axes, so nothing on a continuous axis speaks the entry voice.
_NO_CONTINUOUS_STATISTIC_AXIS: dict[str, str] = {
    "heatmap": "the cells and their axes are the matrix's row and column labels",
    "heatmap[annot]": (
        "the cells, the annotation text and their axes are the matrix's row and column labels"
    ),
}


def _caller_titled_channels(chart: Any) -> dict[str, ChannelBase]:
    """*chart*'s chart-level x/y bindings to a column of its own data, each
    rebuilt carrying the caller's title."""
    columns = set(_as_polars(chart._data).columns)
    channels: dict[str, ChannelBase] = {}
    for axis, cls, title in (("x", X, CALLER_X), ("y", Y, CALLER_Y)):
        binding = chart._encoding.get(axis)
        field = binding.field if isinstance(binding, ChannelBase) else binding
        if isinstance(field, str) and field in columns:
            channels[axis] = (
                binding._with_options(title=title)
                if isinstance(binding, ChannelBase)
                else cls(field, title=title)
            )
    return channels


def _caller_titled_mark_case(build: Callable[[], Any]) -> _Case:
    """*build*'s caller-title door: the same chart with every chart-level
    x/y binding to its data titled through ``.encode()`` before it resolves."""

    def titled() -> Any:
        chart = build()
        return chart.encode(**_caller_titled_channels(chart))

    titles = {axis: ch.option("title") for axis, ch in _caller_titled_channels(build()).items()}
    assert titles, "a caller-title door whose build binds no chart-level x/y to its data"
    return _Case(titled, titles)


def _caller_titled_figure_case(variant: Callable[[_Channels], Any]) -> _Case:
    channels = _Channels(titled=True)
    variant(channels)
    return _Case(lambda: variant(_Channels(titled=True)), channels.used)


def _cases() -> dict[str, Callable[[], _Case]]:
    """Every case, each built lazily so a broken build fails by its own name."""
    cases: dict[str, Callable[[], _Case]] = {}
    plain = {**_FIGURE_BUILDS, **_FIXED_FIGURE_VARIANTS}
    for label, build in plain.items():
        cases[label] = lambda build=build: _Case(build, {})
    mark_builds = {label: build for label, (_, build) in _MARK_CASES.items()}
    mark_builds.update(_MARK_TITLE_VARIANTS)
    for label, build in mark_builds.items():
        cases[label] = lambda build=build: _Case(build, {})
    # A diagnostic mark draws its own domain's columns with no x/y the
    # caller binds, so it has no caller-title door (pinned below).
    for label, (mark_name, build) in _MARK_CASES.items():
        if mark_name not in _DIAGNOSTIC_MARK_BUILDS:
            cases[f"{label}+caller-title"] = lambda build=build: _caller_titled_mark_case(build)
    for label, build in _MARK_TITLE_VARIANTS.items():
        cases[f"{label}+caller-title"] = lambda build=build: _caller_titled_mark_case(build)
    for label, variant in _TITLE_VARIANTS.items():
        # lmplot and residplot's untitled builds are the regroup module's own.
        if label not in _FIGURE_BUILDS:
            cases[label] = lambda variant=variant: _Case(
                lambda: variant(_Channels(titled=False)), {}
            )
        cases[f"{label}+caller-title"] = lambda variant=variant: _caller_titled_figure_case(variant)
    return cases


_CASES = _cases()


def _quietly(render: Callable[[], Any]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return render()


def _domain_messages(render: Callable[[], Any]) -> list[re.Match]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        render()
    matches = (_DOMAIN_MESSAGE.match(str(w.message)) for w in caught)
    return [m for m in matches if m is not None]


def _carries_statistics_marker(wire: Any) -> bool:
    """Whether any transform chain in *wire* (top level, a branch's, a
    layer's) holds the statistics marker."""
    if isinstance(wire, dict):
        return wire == _STATISTICS_MARKER_WIRE or any(
            _carries_statistics_marker(value) for value in wire.values()
        )
    if isinstance(wire, list):
        return any(_carries_statistics_marker(item) for item in wire)
    return False


def _continuous_axes(chart: Any) -> list[str]:
    """The axes Rust resolves on a continuous scale: the interactive scene's
    panel coord carries a numeric domain for those and none for a
    categorical axis."""
    scene = json.loads(_quietly(lambda: chart.interactive().scene_json))
    coord = scene["panels"][0]["coord"]
    return [axis for axis in ("x", "y") if coord[f"{axis}_domain"] is not None]


def _axis_title(svg: str, axis: str) -> str | None:
    """The one name *axis* shows on every panel that draws it, or ``None``."""
    titles = set(x_axis_titles(svg) if axis == "x" else y_axis_titles(svg))
    assert len(titles) <= 1, f"the {axis} axes of one chart show different titles: {titles}"
    return titles.pop() if titles else None


def _established_name(chart: Any, axis: str) -> str | None:
    """The name the axis of resolved *chart* shows when the build binds it to
    a column of its own data: the title= the chart-level binding declares,
    else the column. A layer's own title never counts: a statistic or figure
    axis that shows its own name is listed in :data:`_FIXED_AXIS_TITLES`."""
    field = _established_field(chart, axis)
    if field is None:
        return None
    binding = chart._encoding.get(axis)
    if isinstance(binding, ChannelBase) and binding.option("title"):
        return binding.option("title")
    return field


def _compared_messages(case: str, built: _Case) -> int:
    """Check every entry-voiced message under the far domain on each
    continuous axis against that axis's rendered title; return how many."""
    svg = _quietly(lambda: built.build().to_svg())
    compared = 0
    for axis in _continuous_axes(built.build()):
        shown = _axis_title(svg, axis)
        limit = "xlim" if axis == "x" else "ylim"
        far = getattr(built.build(), limit)(*_FAR_DOMAIN)
        for message in _domain_messages(far.to_svg):
            if message["entry"] is None or message["channel"] != axis:
                continue
            assert message["name"] == shown, (
                f"{case}: {axis}: {message['entry']}'s statistic layer warns "
                f"{message['name']!r}; the axis shows {shown!r}"
            )
            compared += 1
    return compared


def _check_axis_titles(case: str, built: _Case) -> None:
    svg = _quietly(lambda: built.build().to_svg())
    resolved = _quietly(lambda: built.build().layer())
    for axis in ("x", "y"):
        established = _established_name(resolved, axis)
        fixed = _FIXED_AXIS_TITLES.get((case, axis))
        if fixed is not None:
            assert established is not None and fixed != established, (
                f"{case}: {axis}: listed as showing {fixed!r} in place of its column, "
                f"but the axis binds {established!r}"
            )
        expected = built.caller_titles.get(axis) or fixed or established
        if expected is None:
            continue
        shown = _axis_title(svg, axis)
        assert shown == expected, f"{case}: {axis}: the axis shows {shown!r}, not {expected!r}"


@pytest.mark.parametrize("case", sorted(_CASES))
def test_a_statistic_layer_warns_in_the_name_its_axis_shows(case: str) -> None:
    built = _CASES[case]()
    compared = _compared_messages(case, built)
    _check_axis_titles(case, built)
    computes = _carries_statistics_marker(json.loads(built.build().to_spec().to_json()))
    if not computes:
        assert compared == 0, (
            f"{case}: {compared} entry-voiced message(s) from a lowering with no statistic"
        )
    elif case in _NO_CONTINUOUS_STATISTIC_AXIS:
        assert compared == 0, (
            f"{case}: listed as drawing statistics only on categorical axes, "
            f"but {compared} message(s) compared"
        )
    else:
        assert compared > 0, f"{case}: a statistic lowering compared no entry-voiced message"


def test_every_listed_exception_names_a_case() -> None:
    listed = set(_NO_CONTINUOUS_STATISTIC_AXIS) | {case for case, _ in _FIXED_AXIS_TITLES}
    stale = sorted(listed - set(_CASES))
    assert not stale, f"listed case(s) not in the population: {stale}"


def test_the_population_covers_every_figure_and_mark_build() -> None:
    """The regroup module's figure and mark builds are all here, and every
    mark build but a diagnostic one has its caller-title door."""
    assert set(_FIGURE_BUILDS) <= set(_CASES)
    for label, (mark_name, _) in _MARK_CASES.items():
        assert label in _CASES
        if mark_name not in _DIAGNOSTIC_MARK_BUILDS:
            assert f"{label}+caller-title" in _CASES


@pytest.mark.parametrize("mark_name", sorted(_DIAGNOSTIC_MARK_BUILDS))
def test_a_diagnostic_mark_binds_no_chart_level_axis_a_caller_could_title(
    mark_name: str,
) -> None:
    """The diagnostic marks are left out of the caller-title door because
    none binds a chart-level x/y to a column of its data; one that does
    fails here and joins the door."""
    chart = _quietly(_DIAGNOSTIC_MARK_BUILDS[mark_name])
    assert not _caller_titled_channels(chart), mark_name


#: The ``title=`` spellings a channel object takes: ``None``, ``""`` and a
#: blank string suppress the axis title, a string replaces it.
_TITLE_SPELLINGS: dict[str, Any] = {
    "None": None,
    "empty": "",
    "blank": "  ",
    "string": "Caller title",
}

#: Doors that rebind the caller's x/y channel on their own layers: the axis
#: and the column the channel object binds, and the build over it.
_TITLE_DOORS: dict[str, tuple[str, str, Callable[[Any], Any]]] = {
    "residplot x": ("x", "v", lambda x: fm.residplot(_df(), x=x, y="w")),
    "residplot y": ("y", "w", lambda y: fm.residplot(_df(), x="v", y=y)),
    "displot ecdf x": ("x", "v", lambda x: fm.displot(_df(), x=x, kind="ecdf")),
    "mark_boxplot y": ("y", "v", lambda y: fm.Chart(_df()).mark_boxplot().encode(x="g", y=y)),
    "catplot box y": ("y", "v", lambda y: fm.catplot(_df(), x="g", y=y, kind="box")),
    "catplot bar y": ("y", "v", lambda y: fm.catplot(_df(), x="g", y=y, kind="bar")),
    "lmplot x": ("x", "v", lambda x: fm.lmplot(_df(), x=x, y="w")),
}


@pytest.mark.parametrize("spelling", sorted(_TITLE_SPELLINGS))
@pytest.mark.parametrize("door", sorted(_TITLE_DOORS))
def test_a_figure_axis_takes_title_as_a_plain_chart_axis_does(door: str, spelling: str) -> None:
    """The same channel object titles a figure's axis exactly as it titles a
    plain point chart's: suppressed by ``title=None`` and ``title=""``,
    replaced by a string. Rejects a reader that treats a suppression as
    unset and draws the column name instead."""
    axis, field, build = _TITLE_DOORS[door]
    channel = (X if axis == "x" else Y)(field, title=_TITLE_SPELLINGS[spelling])
    plain = (
        fm.Chart(_df()).mark_point().encode(x=channel, y="w")
        if axis == "x"
        else fm.Chart(_df()).mark_point().encode(x="v", y=channel)
    )
    titles = x_axis_titles if axis == "x" else y_axis_titles
    assert titles(_quietly(build(channel).to_svg)) == titles(_quietly(plain.to_svg)), door


@pytest.mark.parametrize(
    "title",
    [None, "", "  ", "\t\n", "Caller title", " padded "],
    ids=["None", "empty", "blank", "whitespace", "string", "padded"],
)
def test_drawn_title_reads_a_title_as_the_renderer_draws_it(title: Any) -> None:
    """``drawn_title`` restates the renderer's rule for which titles draw:
    it is ``None`` exactly when a plain chart's axis draws no title, else
    the title the axis draws. Pins the Python reader to the renderer so the
    two cannot drift."""
    channel = X("v", title=title)
    shown = x_axis_titles(_quietly(fm.Chart(_df()).mark_point().encode(x=channel, y="w").to_svg))
    drawn = drawn_title(channel)
    assert shown == ([] if drawn is None else [drawn.strip()])
