"""A figure call reads its ``encode=`` x/y as its own ``x=``/``y=``.

``regplot``, ``lmplot``, ``residplot``, ``catplot``, ``displot`` and
``jointplot(kind="hist")`` compute a statistic from the columns their
``x=``/``y=`` name. ``encode={"y": "w2"}`` is the same binding as ``y="w2"``,
so the call merges the two once, before any refusal, frame read, kind
dispatch or layer build, and the chart it draws is the chart the keyword call
draws: the fit, the summary, the histogram or the 2-D bins refit on the
column ``encode=`` names. A ``None`` leaves the keyword binding in place, and
every refusal reads the merged binding.

The twin of each case is the keyword call, compared byte for byte. The rows
that must not change (the ``jointplot`` centres that read the encoding
directly, the displot and catplot kinds with no statistic on the channel, a
call with no ``encode=``) are pinned by digests measured on the build before
the change.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

import polars as pl
import pytest

import ferrum as fm
from ferrum._entry import EntryRefusal
from ferrum.encoding import X, Y
from tests.test_null_group_warning import _render_capturing


def _frame() -> pl.DataFrame:
    n = 16
    return pl.DataFrame(
        {
            "v": [float(i) + 0.3 * (i % 3) for i in range(n)],
            "w": [2.0 + (i * 7 % 5) + 0.4 * i for i in range(n)],
            "w2": [1.0 + (i * 3 % 7) + 0.2 * i for i in range(n)],
            "u": [30.0 - i * 1.3 + (i % 4) for i in range(n)],
            "b": [0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
            "b2": [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0],
            "g": ["a", "b"] * 8,
            "f": ["p"] * 8 + ["q"] * 8,
            "c": ["x", "y", "z", "x"] * 4,
            "yr": [2000 + (i % 4) for i in range(n)],
        }
    )


def _svg(chart: Any) -> str:
    return chart.to_svg()


def _digest(chart: Any) -> str:
    return hashlib.sha256(_svg(chart).encode()).hexdigest()


def _refusal(build: Callable[[], Any]) -> tuple[type, str]:
    """The exception type and text a build (rendered) raises."""
    with pytest.raises(Exception) as caught:  # noqa: PT011 -- the twin pins the text
        _svg(build())
    return type(caught.value), str(caught.value)


_METHODS = [
    ("lm", "w", "w2"),
    ("robust", "w", "w2"),
    ("logistic", "b", "b2"),
    ("glm", "w", "w2"),
]


# ---------------------------------------------------------------------------
# regplot / lmplot
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scatter", [True, False], ids=["scatter", "no_scatter"])
@pytest.mark.parametrize(("method", "y", "y2"), _METHODS, ids=[m[0] for m in _METHODS])
def test_regplot_encode_y_equals_the_keyword_call(method: str, y: str, y2: str, scatter: bool):
    df = _frame()
    via_encode = fm.regplot(df, x="v", y=y, method=method, scatter=scatter, encode={"y": y2})
    keyword = fm.regplot(df, x="v", y=y2, method=method, scatter=scatter)
    assert _svg(via_encode) == _svg(keyword)


@pytest.mark.parametrize("scatter", [True, False], ids=["scatter", "no_scatter"])
@pytest.mark.parametrize(("method", "y", "y2"), _METHODS, ids=[m[0] for m in _METHODS])
def test_regplot_encode_x_equals_the_keyword_call(method: str, y: str, y2: str, scatter: bool):
    df = _frame()
    via_encode = fm.regplot(df, x="v", y=y, method=method, scatter=scatter, encode={"x": "u"})
    keyword = fm.regplot(df, x="u", y=y, method=method, scatter=scatter)
    assert _svg(via_encode) == _svg(keyword)


def test_encode_changes_the_regplot_drawing():
    """Negative control: the encode call differs from the call without it."""
    df = _frame()
    assert _svg(fm.regplot(df, x="v", y="w", encode={"y": "w2"})) != _svg(
        fm.regplot(df, x="v", y="w")
    )


def test_lmplot_encode_x_equals_the_keyword_call():
    df = _frame()
    assert _svg(fm.lmplot(df, x="v", y="w", encode={"x": "u"})) == _svg(fm.lmplot(df, x="u", y="w"))


def test_lmplot_encode_x_with_hue_equals_the_keyword_call():
    df = _frame()
    assert _svg(fm.lmplot(df, x="v", y="w", hue="g", encode={"x": "u"})) == _svg(
        fm.lmplot(df, x="u", y="w", hue="g")
    )


def test_lmplot_encode_x_with_col_equals_the_keyword_call():
    df = _frame()
    assert _svg(fm.lmplot(df, x="v", y="w", col="f", encode={"x": "u"})) == _svg(
        fm.lmplot(df, x="u", y="w", col="f")
    )


def test_lmplot_encode_y_with_hue_and_col_equals_the_keyword_call():
    df = _frame()
    assert _svg(fm.lmplot(df, x="v", y="w", hue="g", col="f", encode={"y": "w2"})) == _svg(
        fm.lmplot(df, x="v", y="w2", hue="g", col="f")
    )


def test_regplot_encode_channel_with_title_and_scale_equals_the_keyword_call():
    df = _frame()
    channel = Y("w2", title="T", scale=fm.LogScale())
    assert _svg(fm.regplot(df, x="v", y="w", encode={"y": channel})) == _svg(
        fm.regplot(df, x="v", y=Y("w2", title="T", scale=fm.LogScale()))
    )


def test_lmplot_encode_x_channel_with_title_equals_the_keyword_call():
    df = _frame()
    assert _svg(fm.lmplot(df, x="v", y="w", encode={"x": X("u", title="U axis")})) == _svg(
        fm.lmplot(df, x=X("u", title="U axis"), y="w")
    )


def test_regplot_encode_none_keeps_the_keyword_binding():
    df = _frame()
    assert _svg(fm.regplot(df, x="v", y="w", encode={"y": None})) == _svg(
        fm.regplot(df, x="v", y="w")
    )


def test_regplot_encode_missing_column_is_the_keyword_calls_refusal():
    df = _frame()
    via_encode = _refusal(lambda: fm.regplot(df, x="v", y="w", encode={"y": "nope"}))
    keyword = _refusal(lambda: fm.regplot(df, x="v", y="nope"))
    assert via_encode == keyword
    assert "nope" in via_encode[1]


def test_regplot_encode_binned_y_is_the_keyword_calls_refusal():
    df = _frame()
    via_encode = _refusal(lambda: fm.regplot(df, x="v", y="w", encode={"y": Y("w", bin=True)}))
    keyword = _refusal(lambda: fm.regplot(df, x="v", y=Y("w", bin=True)))
    assert via_encode == keyword


def test_lmplot_encode_nominal_x_is_the_keyword_calls_refusal():
    df = _frame()
    via_encode = _refusal(lambda: fm.lmplot(df, x="v", y="w", encode={"x": "v:N"}))
    keyword = _refusal(lambda: fm.lmplot(df, x="v:N", y="w"))
    assert via_encode == keyword


# ---------------------------------------------------------------------------
# residplot
# ---------------------------------------------------------------------------


def test_residplot_layered_encode_y_equals_the_keyword_call():
    df = _frame()
    assert _svg(fm.residplot(df, x="v", y="w", encode={"y": "w2"})) == _svg(
        fm.residplot(df, x="v", y="w2")
    )


def test_residplot_layered_encode_x_equals_the_keyword_call():
    df = _frame()
    assert _svg(fm.residplot(df, x="v", y="w", encode={"x": "u"})) == _svg(
        fm.residplot(df, x="u", y="w")
    )


def test_residplot_flat_encode_y_equals_the_keyword_call():
    df = _frame()
    flat = {"zero_line": False, "show_metrics": False}
    assert _svg(fm.residplot(df, x="v", y="w", encode={"y": "w2"}, **flat)) == _svg(
        fm.residplot(df, x="v", y="w2", **flat)
    )


def test_residplot_flat_encode_x_equals_the_keyword_call():
    df = _frame()
    flat = {"zero_line": False, "show_metrics": False}
    assert _svg(fm.residplot(df, x="v", y="w", encode={"x": "u"}, **flat)) == _svg(
        fm.residplot(df, x="u", y="w", **flat)
    )


def test_residplot_encode_none_keeps_the_keyword_binding():
    df = _frame()
    assert _svg(fm.residplot(df, x="v", y="w", encode={"y": None})) == _svg(
        fm.residplot(df, x="v", y="w")
    )


def test_residplot_encode_missing_column_is_the_keyword_calls_refusal():
    df = _frame()
    via_encode = _refusal(lambda: fm.residplot(df, x="v", y="w", encode={"y": "nope"}))
    keyword = _refusal(lambda: fm.residplot(df, x="v", y="nope"))
    assert via_encode == keyword


def test_residplot_encode_binned_x_is_the_keyword_calls_refusal():
    df = _frame()
    via_encode = _refusal(lambda: fm.residplot(df, x="v", y="w", encode={"x": X("v", bin=True)}))
    keyword = _refusal(lambda: fm.residplot(df, x=X("v", bin=True), y="w"))
    assert via_encode == keyword


# ---------------------------------------------------------------------------
# catplot
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["point", "bar"])
def test_catplot_summary_encode_y_equals_the_keyword_call(kind: str):
    df = _frame()
    assert _svg(fm.catplot(df, x="c", y="w", kind=kind, encode={"y": "w2"})) == _svg(
        fm.catplot(df, x="c", y="w2", kind=kind)
    )


@pytest.mark.parametrize("kind", ["point", "bar"])
def test_catplot_summary_encode_category_equals_the_keyword_call(kind: str):
    df = _frame()
    assert _svg(fm.catplot(df, x="c", y="w", kind=kind, encode={"x": "g"})) == _svg(
        fm.catplot(df, x="g", y="w", kind=kind)
    )


@pytest.mark.parametrize("kind", ["point", "bar"])
def test_catplot_summary_native_scale_encode_y_equals_the_keyword_call(kind: str):
    df = _frame()
    assert _svg(
        fm.catplot(df, x="yr", y="w", kind=kind, native_scale=True, encode={"y": "w2"})
    ) == _svg(fm.catplot(df, x="yr", y="w2", kind=kind, native_scale=True))


@pytest.mark.parametrize("kind", ["point", "bar"])
def test_catplot_summary_horizontal_encode_value_equals_the_keyword_call(kind: str):
    df = _frame()
    assert _svg(fm.catplot(df, x="w", y="c", kind=kind, orient="h", encode={"y": "w2"})) == _svg(
        fm.catplot(df, x="w2", y="c", kind=kind, orient="h")
    )


@pytest.mark.parametrize("kind", ["point", "bar"])
def test_catplot_summary_encode_channel_equals_the_keyword_call(kind: str):
    df = _frame()
    channel = Y("w2", title="Value")
    assert _svg(fm.catplot(df, x="c", y="w", kind=kind, encode={"y": channel})) == _svg(
        fm.catplot(df, x="c", y=Y("w2", title="Value"), kind=kind)
    )


def test_catplot_encode_value_without_keyword_equals_the_keyword_call():
    df = _frame()
    assert _svg(fm.catplot(df, x="c", kind="point", encode={"y": "w2"})) == _svg(
        fm.catplot(df, x="c", y="w2", kind="point")
    )


@pytest.mark.parametrize("kind", ["point", "bar"])
def test_catplot_encode_none_keeps_the_keyword_binding(kind: str):
    df = _frame()
    assert _svg(fm.catplot(df, x="c", y="w", kind=kind, encode={"y": None})) == _svg(
        fm.catplot(df, x="c", y="w", kind=kind)
    )


@pytest.mark.parametrize("kind", ["point", "bar"])
def test_catplot_encode_missing_column_is_the_keyword_calls_refusal(kind: str):
    df = _frame()
    via_encode = _refusal(lambda: fm.catplot(df, x="c", y="w", kind=kind, encode={"y": "nope"}))
    keyword = _refusal(lambda: fm.catplot(df, x="c", y="nope", kind=kind))
    assert via_encode == keyword


@pytest.mark.parametrize("kind", ["point", "bar"])
def test_catplot_encode_binned_value_equals_the_keyword_call(kind: str):
    df = _frame()
    assert _svg(fm.catplot(df, x="c", y="w", kind=kind, encode={"y": Y("w", bin=True)})) == _svg(
        fm.catplot(df, x="c", y=Y("w", bin=True), kind=kind)
    )


def test_catplot_horizontal_encode_category_equals_the_keyword_call():
    df = _frame()
    assert _svg(fm.catplot(df, x="w", y="c", kind="point", orient="h", encode={"x": "g"})) == _svg(
        fm.catplot(df, x="w", y="g", kind="point", orient="h")
    )


@pytest.mark.parametrize("kind", ["box", "point"])
def test_catplot_encode_category_channel_with_order_equals_the_keyword_call(kind: str):
    df = _frame()
    channel = X("c", sort=["z", "y", "x"])
    assert _svg(fm.catplot(df, x="g", y="w", kind=kind, encode={"x": channel})) == _svg(
        fm.catplot(df, x=X("c", sort=["z", "y", "x"]), y="w", kind=kind)
    )


# ---------------------------------------------------------------------------
# displot
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("overlay", ["kde", "rug"])
def test_displot_overlay_encode_x_equals_the_keyword_call(overlay: str):
    df = _frame()
    assert _svg(fm.displot(df, x="v", encode={"x": "u"}, **{overlay: True})) == _svg(
        fm.displot(df, x="u", **{overlay: True})
    )


def test_displot_kde_with_rug_encode_x_equals_the_keyword_call():
    df = _frame()
    assert _svg(fm.displot(df, x="v", kind="kde", rug=True, encode={"x": "u"})) == _svg(
        fm.displot(df, x="u", kind="kde", rug=True)
    )


def test_displot_hist_mark_override_encode_x_equals_the_keyword_call():
    df = _frame()
    mark = {"opacity": 0.5}
    assert _svg(fm.displot(df, x="v", kind="hist", mark=mark, encode={"x": "u"})) == _svg(
        fm.displot(df, x="u", kind="hist", mark=mark)
    )


@pytest.mark.parametrize("kind", ["hist", "kde", "ecdf", "rug"])
def test_displot_encode_x_without_keyword_equals_the_keyword_call(kind: str):
    df = _frame()
    assert _svg(fm.displot(df, kind=kind, encode={"x": "u"})) == _svg(
        fm.displot(df, x="u", kind=kind)
    )


def test_displot_overlay_encode_channel_with_title_equals_the_keyword_call():
    df = _frame()
    assert _svg(fm.displot(df, x="v", kde=True, encode={"x": X("u", title="U axis")})) == _svg(
        fm.displot(df, x=X("u", title="U axis"), kde=True)
    )


def test_displot_overlay_encode_none_keeps_the_keyword_binding():
    df = _frame()
    assert _svg(fm.displot(df, x="v", kde=True, encode={"x": None})) == _svg(
        fm.displot(df, x="v", kde=True)
    )


def test_displot_overlay_encode_missing_column_is_the_keyword_calls_refusal():
    df = _frame()
    via_encode = _refusal(lambda: fm.displot(df, x="v", kde=True, encode={"x": "nope"}))
    keyword = _refusal(lambda: fm.displot(df, x="nope", kde=True))
    assert via_encode == keyword


def test_displot_encode_binned_x_is_the_keyword_calls_refusal():
    df = _frame()
    via_encode = _refusal(lambda: fm.displot(df, x="v", kind="kde", encode={"x": X("u", bin=True)}))
    keyword = _refusal(lambda: fm.displot(df, x=X("u", bin=True), kind="kde"))
    assert via_encode == keyword


# ---------------------------------------------------------------------------
# jointplot
# ---------------------------------------------------------------------------


def test_jointplot_hist_encode_y_equals_the_keyword_call():
    df = _frame()
    assert _svg(fm.jointplot(df, x="v", y="w", kind="hist", encode={"y": "w2"})) == _svg(
        fm.jointplot(df, x="v", y="w2", kind="hist")
    )


def test_jointplot_hist_encode_x_equals_the_keyword_call():
    df = _frame()
    assert _svg(fm.jointplot(df, x="v", y="w", kind="hist", encode={"x": "u"})) == _svg(
        fm.jointplot(df, x="u", y="w", kind="hist")
    )


def test_jointplot_hist_encode_none_keeps_the_keyword_binding():
    df = _frame()
    assert _svg(fm.jointplot(df, x="v", y="w", kind="hist", encode={"y": None})) == _svg(
        fm.jointplot(df, x="v", y="w", kind="hist")
    )


def test_jointplot_hist_encode_missing_column_is_the_keyword_calls_refusal():
    df = _frame()
    via_encode = _refusal(lambda: fm.jointplot(df, x="v", y="w", kind="hist", encode={"y": "nope"}))
    keyword = _refusal(lambda: fm.jointplot(df, x="v", y="nope", kind="hist"))
    assert via_encode == keyword


def test_jointplot_hist_restating_the_column_equals_the_keyword_call():
    df = _frame()
    assert _svg(fm.jointplot(df, x="v", y="w", kind="hist", encode={"y": "w"})) == _svg(
        fm.jointplot(df, x="v", y="w", kind="hist")
    )


# ---------------------------------------------------------------------------
# Boundaries: zero rows, null and NaN values, every dtype, both output kinds
# ---------------------------------------------------------------------------


def _with_gaps(df: pl.DataFrame) -> pl.DataFrame:
    """``w2`` and ``u`` with a null every fifth row and a NaN at row 3."""
    gapped = [
        pl.when(pl.int_range(pl.len()) % 5 == 0).then(None).otherwise(pl.col(name)).alias(name)
        for name in ("w2", "u")
    ]
    return df.with_columns(gapped).with_columns(
        pl.when(pl.int_range(pl.len()) == 3).then(float("nan")).otherwise(pl.col("w2")).alias("w2")
    )


#: Each figure, built once with the column in ``encode=`` and once with it as
#: the keyword: (through encode=, through the keyword).
_TWINS: dict[str, tuple[Callable[[pl.DataFrame], Any], Callable[[pl.DataFrame], Any]]] = {
    "regplot": (
        lambda d: fm.regplot(d, x="v", y="w", encode={"y": "w2"}),
        lambda d: fm.regplot(d, x="v", y="w2"),
    ),
    "lmplot_hue": (
        lambda d: fm.lmplot(d, x="v", y="w", hue="g", encode={"x": "u"}),
        lambda d: fm.lmplot(d, x="u", y="w", hue="g"),
    ),
    "residplot": (
        lambda d: fm.residplot(d, x="v", y="w", encode={"y": "w2"}),
        lambda d: fm.residplot(d, x="v", y="w2"),
    ),
    "residplot_flat": (
        lambda d: fm.residplot(
            d, x="v", y="w", zero_line=False, show_metrics=False, encode={"x": "u"}
        ),
        lambda d: fm.residplot(d, x="u", y="w", zero_line=False, show_metrics=False),
    ),
    "catplot_point": (
        lambda d: fm.catplot(d, x="c", y="w", kind="point", encode={"y": "w2"}),
        lambda d: fm.catplot(d, x="c", y="w2", kind="point"),
    ),
    "catplot_bar_horizontal": (
        lambda d: fm.catplot(d, x="w", y="c", kind="bar", orient="h", encode={"y": "w2"}),
        lambda d: fm.catplot(d, x="w2", y="c", kind="bar", orient="h"),
    ),
    "displot_kde_overlay": (
        lambda d: fm.displot(d, x="v", kde=True, encode={"x": "u"}),
        lambda d: fm.displot(d, x="u", kde=True),
    ),
    "jointplot_hist": (
        lambda d: fm.jointplot(d, x="v", y="w", kind="hist", encode={"y": "w2"}),
        lambda d: fm.jointplot(d, x="v", y="w2", kind="hist"),
    ),
}


def _outcome(build: Callable[[], Any]) -> tuple[str, list[str]]:
    """What a build renders to (its svg, or the exception text) and every warning raised."""

    def render() -> str:
        try:
            return _svg(build())
        except ValueError as refusal:
            return f"{type(refusal).__name__}: {refusal}"

    return _render_capturing(render)


@pytest.mark.parametrize("name", sorted(_TWINS))
def test_a_zero_row_frame_draws_what_the_keyword_call_draws(name: str):
    via_encode, keyword = _TWINS[name]
    frame = _frame().head(0)
    drawn_by_encode = _outcome(lambda: via_encode(frame))
    drawn_by_keyword = _outcome(lambda: keyword(frame))
    assert drawn_by_encode == drawn_by_keyword


@pytest.mark.parametrize("name", sorted(_TWINS))
def test_null_and_nan_values_in_the_encode_named_column_draw_what_the_keyword_call_draws(
    name: str,
):
    via_encode, keyword = _TWINS[name]
    frame = _with_gaps(_frame())
    assert _outcome(lambda: via_encode(frame)) == _outcome(lambda: keyword(frame))
    assert _outcome(lambda: via_encode(frame))[0].startswith("<svg")


@pytest.mark.parametrize("dtype", [pl.Float32, pl.Int64, pl.Decimal(10, 2)], ids=str)
@pytest.mark.parametrize("name", sorted(_TWINS))
def test_every_dtype_of_the_encode_named_column_draws_what_the_keyword_call_draws(
    name: str, dtype: Any
):
    via_encode, keyword = _TWINS[name]
    frame = _frame().with_columns(
        pl.col("w2").round(0).cast(dtype), pl.col("u").round(0).cast(dtype)
    )
    assert _outcome(lambda: via_encode(frame)) == _outcome(lambda: keyword(frame))


@pytest.mark.parametrize("name", sorted(_TWINS))
def test_the_interactive_page_of_the_encode_call_is_the_keyword_calls(name: str):
    via_encode, keyword = _TWINS[name]
    frame = _frame()
    page, _ = _render_capturing(lambda: via_encode(frame).interactive().to_html())
    expected, _ = _render_capturing(lambda: keyword(frame).interactive().to_html())
    assert page == expected


# ---------------------------------------------------------------------------
# jointplot(kind="hist") reads a channel object by its field
# ---------------------------------------------------------------------------


_HIST_REFUSALS = {
    "x_bin": (
        {"x": X("v", bin=True)},
        "jointplot(x=X('v', bin=True)) is not supported: the histogram computes x itself, "
        "so bin= would be ignored. Count rows over both fields with "
        "fm.Chart(data).mark_hex().encode(y='w', x='v'), or drop x=.",
    ),
    "y_bin": (
        {"y": Y("w", bin=True)},
        "jointplot(y=Y('w', bin=True)) is not supported: the histogram computes y itself, "
        "so bin= would be ignored. Count rows over both fields with "
        "fm.Chart(data).mark_hex().encode(x='v', y='w'), or drop y=.",
    ),
    "y_aggregate": (
        {"y": Y("w", aggregate="mean")},
        "jointplot(y=Y('w', aggregate='mean')) is not supported: the histogram computes y "
        "itself, so aggregate= would be ignored. Draw the aggregate per bin with a bar mark, "
        "fm.Chart(data).mark_bar().encode(x=fm.X('v', bin=True), y=fm.Y('w', aggregate='mean')), "
        "or drop y=.",
    ),
}


@pytest.mark.parametrize("name", sorted(_HIST_REFUSALS))
def test_jointplot_hist_binned_channel_keeps_the_typed_refusal(name: str):
    binding, text = _HIST_REFUSALS[name]
    df = _frame()
    with pytest.raises(EntryRefusal) as through_encode:
        _svg(fm.jointplot(df, x="v", y="w", kind="hist", encode=binding))
    assert str(through_encode.value) == text
    with pytest.raises(EntryRefusal) as through_keyword:
        _svg(fm.jointplot(df, **{"x": "v", "y": "w", **binding}, kind="hist"))
    assert str(through_keyword.value) == text


def test_jointplot_hist_encode_channel_with_title_draws_the_keyword_calls_chart():
    df = _frame()
    via_encode = _svg(fm.jointplot(df, x="v", y="w", kind="hist", encode={"y": Y("w2", title="T")}))
    keyword = _svg(fm.jointplot(df, x="v", y=Y("w2", title="T"), kind="hist"))
    assert via_encode == keyword
    assert ">T</text>" in via_encode


_HIST_SPELLINGS = {
    "x_aggregate": ("x", "mean(v)", X("v", aggregate="mean")),
    "y_aggregate": ("y", "mean(w2)", Y("w2", aggregate="mean")),
}


def _hist_jointplot(binding: Any, *, axis: str, door: str) -> Any:
    keywords = {"x": "v", "y": "w"}
    if door == "keyword":
        return fm.jointplot(_frame(), **{**keywords, axis: binding}, kind="hist")
    return fm.jointplot(_frame(), **keywords, kind="hist", encode={axis: binding})


@pytest.mark.parametrize("door", ["keyword", "encode"])
@pytest.mark.parametrize("name", sorted(_HIST_SPELLINGS))
def test_jointplot_hist_refuses_a_shorthand_as_it_refuses_the_channel_it_spells(
    name: str, door: str
):
    axis, shorthand, channel = _HIST_SPELLINGS[name]
    by_channel = _refusal(lambda: _hist_jointplot(channel, axis=axis, door=door))
    by_shorthand = _refusal(lambda: _hist_jointplot(shorthand, axis=axis, door=door))
    assert by_channel[0] is EntryRefusal
    assert "computes" in by_channel[1]
    assert by_shorthand == by_channel


@pytest.mark.parametrize("door", ["keyword", "encode"])
@pytest.mark.parametrize("axis", ["x", "y"])
def test_jointplot_hist_reads_a_typed_shorthand_as_the_channel_it_spells(axis: str, door: str):
    column, channel_cls = ("v", X) if axis == "x" else ("w2", Y)
    reference = _svg(_hist_jointplot(channel_cls(column, type="Q"), axis=axis, door="keyword"))
    assert _svg(_hist_jointplot(f"{column}:Q", axis=axis, door=door)) == reference
    assert _svg(_hist_jointplot(channel_cls(column, type="Q"), axis=axis, door=door)) == reference


def test_jointplot_hist_limit_on_a_channel_keeps_its_other_options():
    df = _frame()
    via_encode = _svg(
        fm.jointplot(df, x="v", y="w", kind="hist", xlim=(0, 20), encode={"x": X("u", title="T")})
    )
    keyword = _svg(fm.jointplot(df, x=X("u", title="T"), y="w", kind="hist", xlim=(0, 20)))
    assert via_encode == keyword
    assert ">T</text>" in via_encode


# ---------------------------------------------------------------------------
# jointplot: a limit sets the domain of the caller's own scale
# ---------------------------------------------------------------------------


def _positive_frame() -> pl.DataFrame:
    return _frame().with_columns((pl.col("v") + 1).alias("p"), (pl.col("w") + 1).alias("q"))


_OWN_SCALES = {
    "log": (lambda: fm.LogScale(), lambda dom: fm.LogScale(domain=dom)),
    "symlog": (lambda: fm.SymlogScale(), lambda dom: fm.SymlogScale(domain=dom)),
    "sqrt": (lambda: fm.SqrtScale(), lambda dom: fm.SqrtScale(domain=dom)),
    "own_domain_replaced": (
        lambda: fm.LinearScale(domain=[2, 3]),
        lambda dom: fm.LinearScale(domain=dom),
    ),
}


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("kind", ["scatter", "reg", "kde", "hist", "hex"])
@pytest.mark.parametrize("scale_name", sorted(_OWN_SCALES))
def test_a_jointplot_limit_is_the_domain_of_the_callers_own_scale(
    scale_name: str, kind: str, axis: str
):
    bare, with_domain = _OWN_SCALES[scale_name]
    channel_cls, column = (X, "p") if axis == "x" else (Y, "q")
    other = {"x": "q", "y": "p"}[axis]
    limited = {f"{axis}lim": (1, 20)}
    via_limit = fm.jointplot(
        _positive_frame(),
        **{axis: channel_cls(column, scale=bare()), {"x": "y", "y": "x"}[axis]: other},
        kind=kind,
        **limited,
    )
    spelled = fm.jointplot(
        _positive_frame(),
        **{
            axis: channel_cls(column, scale=with_domain([1, 20])),
            {"x": "y", "y": "x"}[axis]: other,
        },
        kind=kind,
    )
    assert _outcome(lambda: via_limit) == _outcome(lambda: spelled)
    assert _outcome(lambda: via_limit)[0].startswith("<svg")


@pytest.mark.parametrize("scale_name", ["log", "symlog"])
def test_a_jointplot_hist_limit_through_encode_is_the_domain_of_the_scale(scale_name: str):
    bare, with_domain = _OWN_SCALES[scale_name]
    df = _positive_frame()
    via_encode = fm.jointplot(
        df, x="v", y="q", kind="hist", xlim=(1, 20), encode={"x": X("p", scale=bare())}
    )
    spelled = fm.jointplot(df, x=X("p", scale=with_domain([1, 20])), y="q", kind="hist")
    assert _outcome(lambda: via_encode) == _outcome(lambda: spelled)
    page, _ = _render_capturing(lambda: via_encode.interactive().to_html())
    expected, _ = _render_capturing(lambda: spelled.interactive().to_html())
    assert page == expected


def test_a_jointplot_limit_the_scale_cannot_take_gets_the_scales_refusal():
    df = _positive_frame()
    by_limit = _refusal(
        lambda: fm.jointplot(df, x=X("p", scale=fm.LogScale()), y="q", xlim=(0, 20))
    )
    by_domain = _refusal(
        lambda: fm.jointplot(df, x=X("p", scale=fm.LogScale(domain=[0, 20])), y="q")
    )
    assert by_limit[0] is by_domain[0]
    assert "log scale domain must not contain 0" in by_limit[1]
    assert "log scale domain must not contain 0" in by_domain[1]


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("kind", ["scatter", "reg", "kde", "hist", "hex"])
def test_a_jointplot_limit_leaves_a_scale_value_it_cannot_merge_to_the_scale_refusal(
    kind: str, axis: str
):
    channel_cls, column = (X, "p") if axis == "x" else (Y, "q")
    other = {"x": "q", "y": "p"}[axis]
    keywords = {axis: channel_cls(column, scale="log"), {"x": "y", "y": "x"}[axis]: other}
    without_limit = _refusal(lambda: fm.jointplot(_positive_frame(), **keywords, kind=kind))
    with_limit = _refusal(
        lambda: fm.jointplot(_positive_frame(), **keywords, kind=kind, **{f"{axis}lim": (1, 20)})
    )
    assert "invalid type: string" in without_limit[1]
    assert with_limit == without_limit


# ---------------------------------------------------------------------------
# jointplot(kind="hist"): the centre's axes are titled by the field
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "binding",
    ["w2", "w2:Q", Y("w2"), Y("w2", scale=fm.SymlogScale())],
    ids=["str", "shorthand", "channel", "channel_with_scale"],
)
def test_the_hist_centre_titles_its_axes_by_the_field(binding: Any):
    svg = _svg(fm.jointplot(_frame(), x="v", y=binding, kind="hist"))
    assert "x_lo" not in svg and "y_lo" not in svg
    assert ">w2</text>" in svg and ">v</text>" in svg


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("spelling", ["shorthand", "channel"])
def test_the_hist_centre_refuses_a_declared_time_type(axis: str, spelling: str):
    column = "v" if axis == "x" else "w2"
    channel_cls = X if axis == "x" else Y
    binding = f"{column}:T" if spelling == "shorthand" else channel_cls(column, type="T")
    with pytest.raises(EntryRefusal) as caught:
        _svg(fm.jointplot(_frame(), **{"x": "v", "y": "w", axis: binding}, kind="hist"))
    assert str(caught.value) == (
        f"jointplot({axis}={channel_cls.__name__}({column!r}, type='T')) is not supported: "
        f"the histogram bins {axis} as numbers into 2-D cells, so type='T' would be read as "
        f"numbers, not as times. Drop the type ({axis}={column!r})."
    )


# ---------------------------------------------------------------------------
# residplot: an encode= naming the column the figure draws stays a later binding
# ---------------------------------------------------------------------------


def _clashing_frame() -> pl.DataFrame:
    """A frame with its own ``residual`` and ``x`` columns, so the computed columns take fresh names."""
    return _frame().with_columns(pl.col("u").alias("residual"), pl.col("w2").alias("x"))


_DRAWN_COLUMN_BUILDS: dict[str, Callable[[], Any]] = {
    "residual_1_layered": lambda: fm.residplot(
        _clashing_frame(), x="v", y="w", encode={"y": Y("residual_1", title="Deviation")}
    ),
    "residual_1_flat": lambda: fm.residplot(
        _clashing_frame(),
        x="v",
        y="w",
        zero_line=False,
        show_metrics=False,
        encode={"y": Y("residual_1", title="Deviation")},
    ),
    "x_1_layered": lambda: fm.residplot(
        _clashing_frame(), x="v", y="w", encode={"x": X("x_1", title="pred")}
    ),
    "x_1_flat": lambda: fm.residplot(
        _clashing_frame(),
        x="v",
        y="w",
        zero_line=False,
        show_metrics=False,
        encode={"x": X("x_1", title="pred")},
    ),
}

_DRAWN_COLUMN_DIGESTS: dict[str, str] = {
    "residual_1_flat": "efe68039251d2b596b8eadca24f5a502ba0d62a09a8f5bedf3b80a0b213a8488",
    "residual_1_layered": "0cf90ffc1ce72f909388d031374712b2a78ad0438b048efb2eb6a2be61d2db02",
    "x_1_flat": "90863dd9939321253ebfd55f5d65f3586388439545b6412128bf539b9b09b87c",
    "x_1_layered": "06606493d86a263aacb3db9e2e0a30d4fb9554ce48fbd312b2b0eb1628ef6923",
}


@pytest.mark.parametrize("name", sorted(_DRAWN_COLUMN_BUILDS))
def test_an_encode_naming_the_drawn_column_renders_as_it_did(name: str):
    assert _digest(_DRAWN_COLUMN_BUILDS[name]()) == _DRAWN_COLUMN_DIGESTS[name]


#: Horizontal catplots read ``encode=`` in the lowering's space (x the
#: category, y the value) under both orientations: a restatement keeps
#: landing on the chart as it did, byte for byte.
_HORIZONTAL_UNCHANGED = {
    f"{kind}_{form}": (kind, form)
    for kind in ("strip", "swarm", "box", "violin", "boxen", "point", "bar")
    for form in ("category_sort", "value_scale")
}


def _horizontal_restating(kind: str, form: str) -> Any:
    encode = (
        {"x": X("c", sort=["z", "y", "x"])}
        if form == "category_sort"
        else {"y": Y("w", scale=fm.LinearScale(domain=[0, 40]))}
    )
    return fm.catplot(_frame(), x="w", y="c", kind=kind, orient="h", encode=encode)


_HORIZONTAL_DIGESTS: dict[str, str] = {
    "bar_category_sort": "ed3b40cbee8b04d5602f3992c20968cafec84f685b3c952b9c72c7a340400096",
    "bar_value_scale": "626efc3f8e67664d6c2cd49172ca28eb628faee0d1dfb7d6f19ec1c812e98be5",
    "box_category_sort": "7aa31abd8c049b51cfbe2cba56affab4c915e258573bee0fa1451aea08d13612",
    "box_value_scale": "8c103f37f053d65030adce4544e9d8a76eea6a2efcdb8fbdddfa2adbcab7a4f4",
    "boxen_category_sort": "681899804f534c954e018dce606ac4f2c9eb703e606e686ac69c23b583ddc0b5",
    "boxen_value_scale": "b98d523afa859736c7d18ca66736239a4e26cab79c95fae2bced0ad3987bebbb",
    "point_category_sort": "c0d9902e73a8fbedb587d74f103076d5288e5960843873f2582e2552e2959bac",
    "point_value_scale": "92982b2084bf994f65f8f9c4f5e6788bfd7c5fad7d239a4ad9b235dbefbc532c",
    "strip_category_sort": "de749cf8c2a9e39e5fd8de4b59ac65b78e85a9027a8190b384246480a6a67554",
    "strip_value_scale": "cdbd4493255a83375c15fbb5f6f145c1d983fda5844d159986e2ebd36f7a9338",
    "swarm_category_sort": "dd29ca0375f2eec40e393a0ecf70eae328f007adff3bf3e8c271814f06b4f22c",
    "swarm_value_scale": "98c7a457835458feb765f4af39715ba57fb63648f2603719f5bcfcbac227cea4",
    "violin_category_sort": "cb46cc1b934e10a3728acb145023b375334361c0356cac434f20c94a7cb52a9b",
    "violin_value_scale": "09767748a76dbb45b8213027110af2bb9d9fbab62de9feeef947b67d0bd9c450",
}


@pytest.mark.parametrize("name", sorted(_HORIZONTAL_UNCHANGED))
def test_horizontal_catplot_restatements_render_as_they_did(name: str):
    assert _digest(_horizontal_restating(*_HORIZONTAL_UNCHANGED[name])) == _HORIZONTAL_DIGESTS[name]


# ---------------------------------------------------------------------------
# Rows that stay as they were: digests measured before the change
# ---------------------------------------------------------------------------

_UNCHANGED: dict[str, Callable[[], Any]] = {
    "jointplot_scatter_encode_y": lambda: fm.jointplot(
        _frame(), x="v", y="w", kind="scatter", encode={"y": "w2"}
    ),
    "jointplot_reg_encode_y": lambda: fm.jointplot(
        _frame(), x="v", y="w", kind="reg", encode={"y": "w2"}
    ),
    "jointplot_kde_encode_x": lambda: fm.jointplot(
        _frame(), x="v", y="w", kind="kde", encode={"x": "u"}
    ),
    "jointplot_hex_encode_y": lambda: fm.jointplot(
        _frame(), x="v", y="w", kind="hex", encode={"y": "w2"}
    ),
    "displot_hist_encode_x": lambda: fm.displot(_frame(), x="v", kind="hist", encode={"x": "u"}),
    "displot_kde_encode_x": lambda: fm.displot(_frame(), x="v", kind="kde", encode={"x": "u"}),
    "displot_ecdf_encode_x": lambda: fm.displot(_frame(), x="v", kind="ecdf", encode={"x": "u"}),
    "displot_rug_encode_x": lambda: fm.displot(_frame(), x="v", kind="rug", encode={"x": "u"}),
    "catplot_strip_encode_y": lambda: fm.catplot(
        _frame(), x="c", y="w", kind="strip", encode={"y": "w2"}
    ),
    "catplot_box_encode_y": lambda: fm.catplot(
        _frame(), x="c", y="w", kind="box", encode={"y": "w2"}
    ),
    "catplot_violin_encode_y": lambda: fm.catplot(
        _frame(), x="c", y="w", kind="violin", encode={"y": "w2"}
    ),
    "catplot_swarm_encode_y": lambda: fm.catplot(
        _frame(), x="c", y="w", kind="swarm", encode={"y": "w2"}
    ),
    "catplot_boxen_encode_x": lambda: fm.catplot(
        _frame(), x="c", y="w", kind="boxen", encode={"x": "g"}
    ),
    "displot_overlay_same_column_title": lambda: fm.displot(
        _frame(), x="v", kde=True, encode={"x": X("v", title="V title")}
    ),
    "displot_overlay_same_column_scale": lambda: fm.displot(
        _frame(), x="v", kde=True, encode={"x": X("v", scale=fm.LogScale())}
    ),
    "catplot_point_same_column_title": lambda: fm.catplot(
        _frame(), x="c", y="w", kind="point", encode={"y": Y("w", title="Value")}
    ),
    "regplot_same_column_title": lambda: fm.regplot(
        _frame(), x="v", y="w", encode={"y": Y("w", title="T")}
    ),
    "lmplot_same_column_scale": lambda: fm.lmplot(
        _frame(), x="v", y="w", encode={"y": Y("w", scale=fm.LogScale())}
    ),
    "residplot_drawn_column_title": lambda: fm.residplot(
        _frame(), x="v", y="w", encode={"y": Y("residual", title="Deviation")}
    ),
    "residplot_fieldless_title": lambda: fm.residplot(
        _frame(), x="v", y="w", encode={"y": Y(title="Deviation")}
    ),
    "catplot_horizontal_count_channel": lambda: fm.catplot(
        _frame(), y="c", kind="count", orient="h", encode={"y": Y(aggregate="count", title="n2")}
    ),
    "regplot_no_encode": lambda: fm.regplot(_frame(), x="v", y="w"),
    "lmplot_no_encode": lambda: fm.lmplot(_frame(), x="v", y="w", hue="g", col="f"),
    "residplot_no_encode": lambda: fm.residplot(_frame(), x="v", y="w"),
    "catplot_point_no_encode": lambda: fm.catplot(_frame(), x="c", y="w", kind="point"),
    "catplot_bar_native_no_encode": lambda: fm.catplot(
        _frame(), x="yr", y="w", kind="bar", native_scale=True
    ),
    "displot_kde_overlay_no_encode": lambda: fm.displot(_frame(), x="v", kde=True),
    "jointplot_hist_no_encode": lambda: fm.jointplot(_frame(), x="v", y="w", kind="hist"),
}

_UNCHANGED_DIGESTS: dict[str, str] = {
    "catplot_horizontal_count_channel": "da334b6500a043fc6f55c12b6ea6b4897d3d9c7642760476e6168660575a9c9b",
    "catplot_point_same_column_title": "a05513a1c26e8bba576e5ea349d8e2accf19e8adc7146bd3e2abcb9c51098d5e",
    "displot_overlay_same_column_scale": "7714733600ab3c657cb14e6ff67f70209083a7d10947641eabd4b29397f24b0e",
    "displot_overlay_same_column_title": "a89c29c29852d125c0b2db56abb618262941c86f1c79b9834429cb6b2923c44c",
    "lmplot_same_column_scale": "0f7d7dcb08ea2522191867f38c3777d49f8a939bd991372fa377b39b4011d6a2",
    "regplot_same_column_title": "d208a470de8b7ebd6652071b8f4986d0d4c39d79fa71d92e5eaf2d18e1ba2cf8",
    "residplot_drawn_column_title": "0cf90ffc1ce72f909388d031374712b2a78ad0438b048efb2eb6a2be61d2db02",
    "residplot_fieldless_title": "0cf90ffc1ce72f909388d031374712b2a78ad0438b048efb2eb6a2be61d2db02",
    "catplot_bar_native_no_encode": "1604d370145c5211fcd1fd19ab043014ac2964c15f031dd6fee692bddbd2a378",
    "catplot_box_encode_y": "bf854b27d7ac9e8a2faacb2605f499b50643a13588a6c59010599da946101d17",
    "catplot_boxen_encode_x": "92e5debb7b7d770929ef3174f37a6d059a8199523f701ef244aca16161be75c3",
    "catplot_point_no_encode": "2217afe490666c65e39e0f786ab37f5d14c2f7593cf2e4373f5bc32e974c55b8",
    "catplot_strip_encode_y": "fa922b4c682a6b8b13c79a783489ccbd49e70d338329a15c7067d913ef9be23b",
    "catplot_swarm_encode_y": "d514478e89246db8a62046ce1554c8be2cc124ed7157412e5c563cc0fe647e0d",
    "catplot_violin_encode_y": "b606264e7b559d7c08baa2e5e9fcdddff4a58a7b8246163f4954db7bbd60b350",
    "displot_ecdf_encode_x": "6df28e8e79e83cc06468210428087ba4dd5f25a4191d33dd37578d87c08a6624",
    "displot_hist_encode_x": "d0db1a74f5a711b87ccb6a510309a89c15df467169e9912515014c44b7f59d4b",
    "displot_kde_encode_x": "baedc76e38e8b9d0d5b0719d3cc848e333c5bfcd80e924dd21cb3942dcb923c7",
    "displot_kde_overlay_no_encode": "7714733600ab3c657cb14e6ff67f70209083a7d10947641eabd4b29397f24b0e",
    "displot_rug_encode_x": "6206a64e9d9a131600b777213b982e8210a2a7edeb2a68a684bb740319d5eb66",
    "jointplot_hex_encode_y": "36089152ec40bbf33eb9781fd7d6d9cc4b040e86d4f2580b80a536a65bda974e",
    "jointplot_hist_no_encode": "a83ff37ad1e12ceba4d546969babd30322082af3300eb0321a2b44074ae73369",
    "jointplot_kde_encode_x": "a5ba90da5dcff94e4b6857223ee174245a11f6426dad4d1896ecebbeb9174077",
    "jointplot_reg_encode_y": "4b46584da38a66cb42aeade181b77a26c5b73ffa778c693125e49b29726d6ff5",
    "jointplot_scatter_encode_y": "1e30a2bf287d0a7e0b87bfb77a0ec025b798cf250a1a3af72d576a581c5d484c",
    "lmplot_no_encode": "5e47c76ac943510c6791f7206a7b847996c1b8d6908769a25bf8caf8253dbf85",
    "regplot_no_encode": "3bdbaec47137ac3c63b88318dee35406805bf6b572494e3167df051eb0a69850",
    "residplot_no_encode": "52e76cf3103c470adf22724adcb6595e2188a26c7a6ffa0d71d812c711cb1f6a",
}


@pytest.mark.parametrize("name", sorted(_UNCHANGED))
def test_unchanged_rows_render_byte_identically(name: str):
    assert _digest(_UNCHANGED[name]()) == _UNCHANGED_DIGESTS[name]
