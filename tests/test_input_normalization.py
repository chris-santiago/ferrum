"""Every render route and ``to_spec`` normalize chart data to polars first.

``Chart(data)`` accepts any input ``to_polars`` supports. Readers that run
before the Arrow conversion (the auto-raster gate, type inference, the ordinal
domain for category-coordinate annotations, the shared-resolve union domain)
answer from that one polars frame, so:

- a ``polars.LazyFrame`` renders identically to its collected frame, and is
  collected once per render and once per ``+``;
- a ``Path`` or ``str`` is refused with ``TypeError: Unsupported data type``
  on every route, ahead of desugar-time validation;
- pandas and PyArrow data get the same answers polars data gets;
- polars data passes through as the same object, and the user's chart is
  never mutated.
"""

from __future__ import annotations

import pathlib
import re
import warnings

import polars as pl
import pytest

import ferrum as fm
from ferrum.render_config import RenderConfig

_UNSUPPORTED = r"^Unsupported data type: (PosixPath|WindowsPath|str)\."


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "y": [4.0, 5.0, 6.0],
            "g": ["a", "b", "a"],
            "t": pl.Series([1, 2, 3]).cast(pl.Datetime("ms")),
        }
    )


def _point(d):
    return fm.Chart(d).mark_point().encode(x="x", y="y")


def _line(d):
    return fm.Chart(d).mark_line().encode(x="x", y="y")


# Each route renders a chart built from one data object to a string.
_ROUTES = {
    "point": lambda d: _point(d).to_svg(),
    "point color": lambda d: fm.Chart(d).mark_point().encode(x="x", y="y", color="g").to_svg(),
    "temporal x": lambda d: fm.Chart(d).mark_line().encode(x="t", y="y").to_svg(),
    "histogram": lambda d: fm.Chart(d).mark_histogram().encode(x="x").to_svg(),
    "facet": lambda d: _point(d).facet("g").to_svg(),
    "interactive": lambda d: _point(d).interactive().to_html(),
    "plus": lambda d: (_point(d) + _line(d)).to_svg(),
    "LayerChart": lambda d: fm.LayerChart(_point(d), _line(d)).to_svg(),
    "LayerChart shared interactive": lambda d: (
        fm.LayerChart(_point(d), _line(d), resolve={"x": "shared", "y": "shared"})
        .interactive()
        .to_html()
    ),
    "hconcat": lambda d: (_point(d) | _line(d)).to_svg(),
    "SecondaryY": lambda d: (_line(d) + fm.SecondaryY("y", mark="bar")).to_svg(),
    "label_map": lambda d: (
        fm.Chart(d)
        .mark_bar()
        .encode(x=fm.X("g", axis=fm.Axis(label_map={"a": "A"})), y="y")
        .to_svg()
    ),
    "displot": lambda d: fm.displot(d, x="x").to_svg(),
    "mark_raster": lambda d: fm.Chart(d).mark_raster().encode(x="x", y="y").to_svg(),
}


def _quiet(fn, d):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(d)


@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_lazyframe_renders_like_its_collected_frame(route):
    df = _frame()
    fn = _ROUTES[route]
    assert _quiet(fn, df.lazy()) == _quiet(fn, df)


def test_lazyframe_spec_keeps_the_temporal_annotation():
    df = _frame()
    lazy = fm.Chart(df.lazy()).mark_line().encode(x="t", y="y").to_spec()
    eager = fm.Chart(df).mark_line().encode(x="t", y="y").to_spec()
    assert lazy.to_json() == eager.to_json()
    assert '"type":"temporal"' in lazy.to_json().replace(" ", "")


class _CollectCounter:
    """Count ``collect()`` calls on the given LazyFrames only.

    polars itself builds and collects internal lazy frames (``DataFrame.rename``
    does), so only collects of the caller's own frames are counted.
    """

    def __init__(self, monkeypatch, *frames: pl.LazyFrame):
        self.calls = 0
        watched = {id(f) for f in frames}
        original = pl.LazyFrame.collect
        counter = self

        def counting_collect(frame, *args, **kwargs):
            if id(frame) in watched:
                counter.calls += 1
            return original(frame, *args, **kwargs)

        monkeypatch.setattr(pl.LazyFrame, "collect", counting_collect)


@pytest.mark.parametrize(
    "render",
    [
        lambda c: c.to_svg(),
        lambda c: c.to_spec(),
    ],
    ids=["to_svg", "to_spec"],
)
def test_lazyframe_is_collected_once_per_render(monkeypatch, render):
    lazy = _frame().lazy()
    chart = fm.Chart(lazy).mark_point().encode(x="x", y="y", color="g")
    counter = _CollectCounter(monkeypatch, lazy)
    _quiet(render, chart)
    assert counter.calls == 1


@pytest.mark.parametrize("shared", [True, False], ids=["same LazyFrame", "two LazyFrames"])
def test_lazyframe_is_collected_once_per_plus(monkeypatch, shared):
    # Each distinct LazyFrame operand is collected exactly once; a LazyFrame
    # both operands share is collected once for the pair.
    lazy = _frame().lazy()
    other = lazy if shared else _frame().with_columns(pl.col("y") * 2).lazy()
    lhs, rhs = _point(lazy), _line(other)
    counter = _CollectCounter(monkeypatch, lazy, other)
    lhs + rhs
    assert counter.calls == (1 if shared else 2)


def test_lazyframe_is_collected_once_per_interactive_render(monkeypatch):
    # `.interactive()` renders the scene once on construction and `to_html`
    # renders it again; each render collects once.
    lazy = _frame().lazy()
    chart = fm.Chart(lazy).mark_point().encode(x="x", y="y", color="g")
    view = _quiet(lambda c: c.interactive(), chart)
    counter = _CollectCounter(monkeypatch, lazy)
    view.to_html()
    assert counter.calls == 1


_REFUSED_ROUTES = {
    **_ROUTES,
    "to_spec": lambda d: _point(d).to_spec(),
    "to_json": lambda d: _point(d).to_json(),
    "plus mixed": lambda d: (_point(_frame()) + _line(d)).to_svg(),
}


@pytest.mark.parametrize("kind", ["Path", "str"])
@pytest.mark.parametrize("route", sorted(_REFUSED_ROUTES))
def test_path_and_str_get_the_unsupported_type_refusal(kind, route):
    data = pathlib.Path("/tmp/data.csv") if kind == "Path" else "/tmp/data.csv"
    with pytest.raises(TypeError, match=_UNSUPPORTED):
        _quiet(_REFUSED_ROUTES[route], data)


@pytest.mark.parametrize(
    "build",
    [
        lambda d: fm.Chart(d).mark_histogram(),
        lambda d: fm.Chart(d).mark_boxplot(),
        lambda d: fm.Chart(d).mark_density(),
        # `+ SecondaryY` desugars the base chart when the operator runs.
        lambda d: fm.Chart(d).mark_histogram() + fm.SecondaryY("x", mark="line"),
    ],
    ids=["histogram", "boxplot", "density", "histogram + SecondaryY"],
)
@pytest.mark.parametrize("entry", ["to_svg", "to_spec"])
@pytest.mark.parametrize("kind", ["Path", "str"])
def test_unsupported_data_is_refused_before_desugar_validation(build, entry, kind):
    # Each chart is also missing the encoding its desugar requires; the data
    # refusal is reported first.
    data = pathlib.Path("/tmp/data.csv") if kind == "Path" else "/tmp/data.csv"
    with pytest.raises(TypeError, match=_UNSUPPORTED):
        getattr(build(data), entry)()


def _foreign(kind: str, df: pl.DataFrame):
    return df.to_pandas() if kind == "pandas" else df.to_arrow()


@pytest.mark.parametrize("kind", ["pandas", "pyarrow"])
def test_category_annotation_lands_on_its_category(kind):
    df = pl.DataFrame({"c": ["a", "b", "c", "d"], "v": [1.0, 2.0, 3.0, 4.0]})

    def annotation_x(d) -> str:
        chart = fm.Chart(d).mark_bar().encode(x="c", y="v") + fm.annotate_text("d", 3.5, "HERE")
        match = re.search(r'<text x="([\d.]+)"[^>]*>HERE</text>', chart.to_svg())
        assert match is not None
        return match.group(1)

    polars_x = annotation_x(df)
    assert polars_x == "553.795"
    assert annotation_x(_foreign(kind, df)) == polars_x


@pytest.mark.parametrize("kind", ["pandas", "pyarrow"])
@pytest.mark.parametrize("output", ["interactive", "static"])
def test_shared_resolve_layer_chart_renders(kind, output):
    df = _frame().select("x", "y")

    def render(d):
        lc = fm.LayerChart(_point(d), _line(d), resolve={"x": "shared", "y": "shared"})
        return _quiet(
            lambda c: c.interactive().to_html() if output == "interactive" else c.to_svg(), lc
        )

    assert render(_foreign(kind, df)) == render(df)


@pytest.mark.parametrize("kind", ["pandas", "pyarrow"])
def test_untyped_numeric_color_resolves_by_dtype_in_the_auto_raster_gate(kind):
    df = pl.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0], "y": [4.0, 5.0, 6.0, 7.0], "v": [0.1, 0.5, 0.2, 0.9]}
    )
    cfg = RenderConfig(raster_threshold=0, raster_behavior="warn")

    def render(d):
        chart = (
            fm.Chart(d)
            .mark_point()
            .encode(x="x:Q", y="y:Q", color="v")
            .properties(render_config=cfg)
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svg = chart.to_svg()
        return svg, [str(w.message) for w in caught]

    polars_svg, polars_warnings = render(df)
    assert any("substituted mark_raster" in m for m in polars_warnings)
    svg, messages = render(_foreign(kind, df))
    assert messages == polars_warnings
    assert svg == polars_svg


@pytest.mark.parametrize(
    "render",
    [lambda c: c.to_svg(), lambda c: c.interactive().to_html(), lambda c: c.to_spec()],
    ids=["to_svg", "interactive", "to_spec"],
)
def test_polars_data_is_not_converted(monkeypatch, render):
    # polars data reaches every reader as the caller's own frame: nothing on
    # the render path rebuilds it from Arrow.
    chart = fm.Chart(_frame()).mark_point().encode(x="x", y="y", color="g")
    rebuilt = []
    original = pl.from_arrow
    monkeypatch.setattr(pl, "from_arrow", lambda *a, **k: rebuilt.append(1) or original(*a, **k))
    _quiet(render, chart)
    assert rebuilt == []


def test_chart_without_data_is_unchanged():
    chart = fm.Chart().mark_point().encode(x="x", y="y")
    with pytest.raises(ValueError, match=r"Chart\(data=None\) requires per-layer data"):
        chart.to_svg()


@pytest.mark.parametrize("kind", ["LazyFrame", "pandas"])
def test_the_users_chart_is_never_mutated(kind):
    df = _frame()
    data = df.lazy() if kind == "LazyFrame" else df.to_pandas()
    chart = fm.Chart(data).mark_point().encode(x="x", y="y")
    other = _line(data)
    _quiet(lambda c: c.to_svg(), chart)
    _quiet(lambda c: c.interactive().to_html(), chart)
    chart.to_spec()
    chart + other
    assert chart._data is data
    assert other._data is data
    collected = data.collect() if kind == "LazyFrame" else pl.from_pandas(data)
    assert collected.equals(df)
