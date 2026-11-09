"""``regplot``/``lmplot``/``residplot`` read every column-naming keyword the way
``Chart.encode()`` reads it: shorthand (``"v:Q"``) parses into the field and
type it spells, a declared ``N``/``O`` on an axis the fit or the residual
axis reads as numbers is refused, a declared ``T`` sets a time axis, and a
typed ``hue=``/``color=`` keeps its type instead of being forced nominal.
With ``scatter=False`` the call's keyword bindings (beyond ``x``/``y``) bind
on the fit chart itself, through ``Chart.encode``, exactly as ``encode=``
already does on every form.

Every ``_VALID_METHODS`` member is swept at least once across the
byte-identical and the keyword-binding tests, over both ``scatter=`` forms,
so a method that reads its predictor/response through a route this module
does not cover fails here rather than only in production.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
import warnings

import polars as pl
import pytest

import ferrum as fm
from ferrum._entry import EntryRefusal
from ferrum.encoding import Color, X, Y
from ferrum.plots.regression import _SMOOTH_ROUTED_METHODS, _VALID_METHODS

from _svg_extents import axis_tick_labels

_METHODS = sorted(_VALID_METHODS)


#: Enough rows that a per-group refit (hue="g") still has a usable sample
#: on each side, and that the logistic response below overlaps rather than
#: cleanly separating.
_N_ROWS = 40


@pytest.fixture
def xy_df() -> pl.DataFrame:
    """A noisy monotone relationship, a binary response, and a grouping column.

    ``b`` is drawn from a logistic model (not thresholded), so the two
    classes overlap -- a clean split makes ``Logistic`` refuse outright
    ("singular design"), the same shape ``test_lmplot_scatter_data_source.py``'s
    ``obs_df`` fixture avoids for the same reason.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    predictor = np.linspace(1.0, 20.0, _N_ROWS)
    probs = 1.0 / (1.0 + np.exp(-(predictor - 10.5) / 3.0))
    return pl.DataFrame(
        {
            "v": predictor.tolist(),
            "w": (predictor * 1.5 + 2.0 + rng.normal(0, 0.3, _N_ROWS)).tolist(),
            "b": (rng.random(_N_ROWS) < probs).astype(float).tolist(),
            "g": (["a", "b"] * ((_N_ROWS + 1) // 2))[:_N_ROWS],
            # Few distinct values on purpose: forcing this column nominal
            # (the comparison `test_a_quantitative_hue_keeps_its_type` makes)
            # groups it into one fit per value, and a one-row-per-value
            # column would degenerate every one of those fits.
            "k": [float(i % 4) for i in range(_N_ROWS)],
        }
    )


def _response_column(method: str) -> str:
    return "b" if method == "logistic" else "w"


def _figure_call(figure, df: pl.DataFrame, *, method: str, scatter: bool, **kw) -> fm.Chart:
    return figure(df, x="v", y=_response_column(method), method=method, scatter=scatter, **kw)


# ---------------------------------------------------------------------------
# A shorthand keyword is read as encode() reads it.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("figure", [fm.regplot, fm.lmplot], ids=["regplot", "lmplot"])
@pytest.mark.parametrize("scatter", [True, False])
@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("ci", [95, None])
def test_a_shorthand_keyword_is_read_as_encode_reads_it(xy_df, figure, scatter, method, ci):
    """``x="v:Q"``/``y="w:Q"`` draws byte-identical to the bare field name,
    for every accepted method, both ``scatter=`` forms and both ``ci=``
    forms (a drawn CI band, and none) -- reaching the fit unparsed, the
    shorthand would fail raw instead (a column literally named ``"v:Q"``
    does not exist).
    """
    y_col = _response_column(method)
    bare = _figure_call(figure, xy_df, method=method, scatter=scatter, ci=ci).to_svg()
    typed = figure(xy_df, x="v:Q", y=f"{y_col}:Q", method=method, scatter=scatter, ci=ci).to_svg()
    assert bare == typed


@pytest.mark.parametrize("figure", [fm.regplot, fm.lmplot], ids=["regplot", "lmplot"])
@pytest.mark.parametrize("method", sorted(_SMOOTH_ROUTED_METHODS) + ["robust"])
def test_a_shorthand_keyword_survives_every_smooth_option(xy_df, figure, method):
    """``x_bins=``, ``truncate=False``, ``logx=True`` and ``x_jitter=`` all
    still forward correctly once ``x=``/``y=`` are read as shorthand --
    including ``method="robust"``, whose dedicated transform has no
    ``x_bins=``/``x_estimator=`` of its own to forward.
    """
    if method == "robust":
        # No x_bins/x_estimator: the Robust transform has none to
        # pre-aggregate the fit input with (regplot/lmplot's own
        # x_bins= refusal for this method).
        kwargs = dict(truncate=False, x_jitter=0.1)
    elif method == "sqrt":
        # sqrt refuses logx=True (a second predictor transform on the log
        # axis), so it is exercised without logx/x_jitter and with the
        # other two options alone.
        kwargs = dict(x_bins=5, truncate=False)
    else:
        kwargs = dict(x_bins=5, truncate=False, logx=True, x_jitter=0.1)
    bare = figure(xy_df, x="v", y="w", method=method, ci=None, **kwargs).to_svg()
    typed = figure(xy_df, x="v:Q", y="w:Q", method=method, ci=None, **kwargs).to_svg()
    assert bare == typed


@pytest.mark.parametrize("lowess", [False, True])
@pytest.mark.parametrize("robust", [False, True])
def test_residplot_shorthand_keyword_is_read_as_encode_reads_it(xy_df, lowess, robust):
    """residplot's x/y shorthand, layered and flat -- an un-split
    ``"v:Q"``/``"w:Q"`` reaching ``drop_nulls`` as a column name would
    instead raise polars' own ``ColumnNotFoundError``.
    """
    bare = fm.residplot(xy_df, x="v", y="w", lowess=lowess, robust=robust).to_svg()
    typed = fm.residplot(xy_df, x="v:Q", y="w:Q", lowess=lowess, robust=robust).to_svg()
    assert bare == typed


def test_a_shorthand_aggregate_still_refuses_with_the_call_time_text(xy_df):
    """``x="mean(v)"`` keeps refusing with the call-time text on every door --
    this is the negative control that the shorthand parse added here did not
    widen into honoring an aggregate shorthand on the predictor.
    """
    for figure, kwargs in (
        (fm.regplot, {}),
        (fm.lmplot, {}),
        (fm.residplot, {}),
    ):
        with pytest.raises(EntryRefusal) as excinfo:
            figure(xy_df, x="mean(v)", y="w", **kwargs).to_svg()
        assert "x=X('v', aggregate='mean')" in str(excinfo.value)


# ---------------------------------------------------------------------------
# A categorical type (N/O) on a fit axis is refused.
# ---------------------------------------------------------------------------


def _fit_categorical_message(voice: str, channel: str, cls: str, field: str, type_: str) -> str:
    return (
        f"{voice}({channel}={cls}({field!r}, type={type_!r})) is not supported: the fit reads "
        f"{channel} as numbers and draws its curve along a continuous axis, so type={type_!r} "
        f"would place the fitted values as categories. Drop the type ({channel}={field!r})."
    )


@pytest.mark.parametrize("figure", [fm.regplot, fm.lmplot], ids=["regplot", "lmplot"])
@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("scatter", [True, False])
@pytest.mark.parametrize("channel", ["x", "y"])
@pytest.mark.parametrize("type_", ["N", "O"])
@pytest.mark.parametrize("as_object", [False, True])
def test_a_categorical_type_on_a_fit_axis_is_refused(
    xy_df, figure, method, scatter, channel, type_, as_object
):
    """``x="v:N"``/``x=X("v", type="N")`` (and ``"O"``) refuse in the
    figure's own voice, for a shorthand string or an explicit channel
    object alike, on either axis, every accepted method and both
    ``scatter=`` forms -- reaching the fit unrefused, a shorthand type
    would fail raw and a channel object would draw a broken chart of
    hundreds of categories.
    """
    y_col = _response_column(method)
    field = "v" if channel == "x" else y_col
    cls, cls_name = (X, "X") if channel == "x" else (Y, "Y")
    value = cls(field, type=type_) if as_object else f"{field}:{type_}"
    kwargs = {
        "x": "v",
        "y": y_col,
        "method": method,
        "scatter": scatter,
        "ci": None,
        channel: value,
    }
    with pytest.raises(EntryRefusal) as excinfo:
        figure(xy_df, **kwargs).to_svg()
    assert str(excinfo.value) == _fit_categorical_message(
        figure.__name__, channel, cls_name, field, type_
    )


@pytest.mark.parametrize("figure", [fm.regplot, fm.lmplot], ids=["regplot", "lmplot"])
def test_a_categorical_type_via_encode_is_refused_too(xy_df, figure):
    """The same refusal reaches through ``encode={"x": "v:N"}``, not only
    the ``x=``/``y=`` keyword door.
    """
    with pytest.raises(EntryRefusal) as excinfo:
        figure(xy_df, x="v", y="w", ci=None, encode={"x": "v:N"}).to_svg()
    assert str(excinfo.value) == _fit_categorical_message(figure.__name__, "x", "X", "v", "N")


@pytest.mark.parametrize("figure", [fm.regplot, fm.lmplot], ids=["regplot", "lmplot"])
@pytest.mark.parametrize("type_", ["N", "O"])
def test_a_categorical_type_on_the_response_axis_is_refused(xy_df, figure, type_):
    """Same rule on ``y=``, including ``method="logistic"`` on its binary
    response.
    """
    with pytest.raises(EntryRefusal) as excinfo:
        figure(xy_df, x="v", y=f"w:{type_}", ci=None).to_svg()
    assert str(excinfo.value) == _fit_categorical_message(figure.__name__, "y", "Y", "w", type_)

    with pytest.raises(EntryRefusal) as excinfo:
        figure(xy_df, x="v", y=f"b:{type_}", method="logistic", ci=None).to_svg()
    assert str(excinfo.value) == _fit_categorical_message(figure.__name__, "y", "Y", "b", type_)


@pytest.mark.parametrize("type_", ["N", "O"])
def test_residplot_categorical_predictor_is_refused(xy_df, type_):
    """residplot's x refuses the same way regplot's does, for a shorthand
    string or a channel object.
    """
    with pytest.raises(EntryRefusal) as excinfo:
        fm.residplot(xy_df, x=f"v:{type_}", y="w").to_svg()
    assert str(excinfo.value) == _fit_categorical_message("residplot", "x", "X", "v", type_)

    with pytest.raises(EntryRefusal) as excinfo:
        fm.residplot(xy_df, x=X("v", type=type_), y="w").to_svg()
    assert str(excinfo.value) == _fit_categorical_message("residplot", "x", "X", "v", type_)


@pytest.mark.parametrize("type_", ["N", "O", "T"])
def test_residplot_categorical_or_temporal_response_is_refused(xy_df, type_):
    """residplot's y also refuses a declared ``T``: the residual axis draws
    signed differences around zero, never a category or a time.
    """
    placement = "timestamps" if type_ == "T" else "categories"
    with pytest.raises(EntryRefusal) as excinfo:
        fm.residplot(xy_df, x="v", y=f"w:{type_}").to_svg()
    assert str(excinfo.value) == (
        f"residplot(y=Y('w', type={type_!r})) is not supported: the residual axis "
        f"draws numbers around 0, so type={type_!r} would place the residuals as "
        f"{placement}. Drop the type (y='w')."
    )


def test_a_quantitative_or_unset_type_is_unaffected(xy_df):
    """Negative control: no type and an explicit ``:Q`` are both honored,
    never refused, on every door this module checks above.
    """
    bare = fm.regplot(xy_df, x="v", y="w", ci=None).to_svg()
    assert fm.regplot(xy_df, x="v:Q", y="w:Q", ci=None).to_svg() == bare
    bare_r = fm.residplot(xy_df, x="v", y="w").to_svg()
    assert fm.residplot(xy_df, x="v:Q", y="w:Q").to_svg() == bare_r


def test_a_categorical_type_on_a_plain_mark_smooth_is_refused_in_its_own_voice(xy_df):
    """The pending ``scatter=False`` smooth form IS ``mark_smooth``, so the
    same rule reaches it, voiced as ``mark_smooth`` (case where no figure
    wraps the call).
    """
    with pytest.raises(EntryRefusal) as excinfo:
        fm.Chart(xy_df).mark_smooth().encode(x="v:N", y="w").to_svg()
    assert str(excinfo.value) == _fit_categorical_message("mark_smooth", "x", "X", "v", "N")

    with pytest.raises(EntryRefusal) as excinfo:
        fm.Chart(xy_df).mark_point().mark_smooth().encode(x="v:N", y="w").to_svg()
    assert str(excinfo.value) == _fit_categorical_message("mark_smooth", "x", "X", "v", "N")


# ---------------------------------------------------------------------------
# A declared T type sets a time axis.
# ---------------------------------------------------------------------------


def _x_ticks(svg: str) -> list[str]:
    return axis_tick_labels(svg, axis="x")


@pytest.mark.parametrize("figure", [fm.regplot, fm.lmplot], ids=["regplot", "lmplot"])
def test_a_temporal_type_sets_the_axis_with_a_scatter_drawn(xy_df, figure):
    """``x="v:T"``/``y="w:T"`` draw a time axis (date-formatted ticks), not a
    plain numeric one, with ``scatter=True`` (the default): the scatter's
    own channel establishes the axis, and the fit still computes on the
    raw numbers -- rendering at all proves the point, since an unparsed
    ``"v:T"`` reaching the fit column lookup raw would fail instead.
    """
    plain = figure(xy_df, x="v", y="w", ci=None).to_svg()
    timed = figure(xy_df, x="v:T", y="w", ci=None).to_svg()
    assert plain != timed
    plain_ticks = _x_ticks(plain)
    timed_ticks = _x_ticks(timed)
    assert plain_ticks and all(re.fullmatch(r"-?\d+(\.\d+)?", t) for t in plain_ticks)
    assert not (timed_ticks and all(re.fullmatch(r"-?\d+(\.\d+)?", t) for t in timed_ticks))

    # Draws the exact channel object `X("v", type="T")` already draws.
    via_object = figure(xy_df, x=X("v", type="T"), y="w", ci=None).to_svg()
    assert via_object == timed


@pytest.mark.parametrize("figure", [fm.regplot, fm.lmplot], ids=["regplot", "lmplot"])
def test_a_temporal_shorthand_keeps_its_type_under_logx(xy_df, figure):
    """``x="v:T"`` with ``logx=True`` keeps the declared ``T``, the exact
    spec ``X("v", type="T")`` with the same ``logx=True`` builds -- the
    one keyword parse (``_required_channel``) feeds ``_log_scaled_x`` a
    channel either way, so neither spelling loses the type the other
    keeps.

    Compares ``to_spec()``, not rendered bytes: a log scale draws the
    axis the same way whether or not the channel also declares ``T``, so
    a spelling that dropped the declared type under ``logx`` (building
    ``X("v", scale=log)`` instead of ``X("v", type="T", scale=log)``)
    would still render byte-identical -- only the spec itself tells the
    two apart.
    """
    shorthand = figure(xy_df, x="v:T", y="w", logx=True, ci=None)
    via_object = figure(xy_df, x=X("v", type="T"), y="w", logx=True, ci=None)
    assert shorthand.to_spec() == via_object.to_spec()
    assert shorthand.to_svg() == via_object.to_svg()


@pytest.mark.parametrize("figure", [fm.regplot, fm.lmplot], ids=["regplot", "lmplot"])
@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("scatter", [True, False])
@pytest.mark.parametrize("channel", ["x", "y"])
def test_a_temporal_shorthand_renders_the_same_as_the_channel_object(
    xy_df, figure, method, scatter, channel
):
    """``"v:T"``/``"w:T"`` draws byte-identical to ``X(..., type="T")``/
    ``Y(..., type="T")`` on every method and both ``scatter=`` forms --
    whatever a declared ``T`` does to the render (applied with a scatter
    drawn, not applied with ``scatter=False``), shorthand and the channel
    object do the same thing.
    """
    y_col = _response_column(method)
    field = "v" if channel == "x" else y_col
    cls = X if channel == "x" else Y
    kwargs = {"x": "v", "y": y_col, "method": method, "scatter": scatter, "ci": None}
    kwargs[channel] = f"{field}:T"
    shorthand = figure(xy_df, **kwargs).to_svg()
    kwargs[channel] = cls(field, type="T")
    via_object = figure(xy_df, **kwargs).to_svg()
    assert shorthand == via_object


def test_residplot_temporal_predictor_sets_the_axis(xy_df):
    """residplot's x keeps a declared ``T`` through its re-point onto the
    residual batch's own predictor column, drawing the same time axis a
    channel object ``X("v", type="T")`` draws.
    """
    plain = fm.residplot(xy_df, x="v", y="w").to_svg()
    timed = fm.residplot(xy_df, x="v:T", y="w").to_svg()
    assert plain != timed
    plain_ticks = _x_ticks(plain)
    assert plain_ticks and all(re.fullmatch(r"-?\d+(\.\d+)?", t) for t in plain_ticks)

    via_object = fm.residplot(xy_df, x=X("v", type="T"), y="w").to_svg()
    assert via_object == timed


def test_residplot_shorthand_color_groups_by_its_column(xy_df):
    """``color="g:N"`` groups by column ``"g"``, byte-identical to
    ``color="g"``: the shorthand parse runs on ``color=`` the same way it
    runs on ``x=``/``y=``, before the column-or-literal-color dispatch
    ever sees the raw string.
    """
    via_shorthand = fm.residplot(xy_df, x="v", y="w", color="g:N").to_svg()
    bare = fm.residplot(xy_df, x="v", y="w", color="g").to_svg()
    assert via_shorthand == bare


# ---------------------------------------------------------------------------
# A genuinely temporal (datetime) predictor is refused alike.
# ---------------------------------------------------------------------------


@pytest.fixture
def datetime_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "t": [dt.datetime(2020, 1, i) for i in range(1, 11)],
            "w": [float(i) * 2.0 + 1.0 for i in range(1, 11)],
        }
    )


@pytest.mark.parametrize("spelling", ["t", "t:T"])
@pytest.mark.parametrize("figure", [fm.regplot, fm.lmplot], ids=["regplot", "lmplot"])
def test_a_datetime_predictor_is_refused_alike(datetime_df, figure, spelling):
    """A real datetime column fails the same way with or without ``:T`` --
    the transform itself refuses a non-numeric predictor, so declaring the
    type changes nothing about the outcome, only confirming the shorthand
    reaches the same column either way.
    """
    with pytest.raises(ValueError) as excinfo:
        figure(datetime_df, x=spelling, y="w", ci=None).to_svg()
    assert "stat_smooth: column 't' must be numeric" in str(excinfo.value)


# ---------------------------------------------------------------------------
# A typed hue keeps its type.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("figure", [fm.regplot, fm.lmplot], ids=["regplot", "lmplot"])
@pytest.mark.parametrize("type_", ["N", "O"])
def test_a_typed_hue_keeps_its_type(xy_df, figure, type_):
    """``hue="g:N"``/``"g:O"`` draw byte-identical to an explicit
    ``Color("g", type=...)`` channel object (negative control: a
    categorical hue reads correctly either way).
    """
    via_shorthand = figure(xy_df, x="v", y="w", hue=f"g:{type_}", ci=None).to_svg()
    via_object = figure(xy_df, x="v", y="w", hue=Color("g", type=type_), ci=None).to_svg()
    assert via_shorthand == via_object


@pytest.mark.parametrize("figure", [fm.regplot, fm.lmplot], ids=["regplot", "lmplot"])
def test_a_quantitative_hue_keeps_its_type(xy_df, figure):
    """``hue="k:Q"`` draws what ``hue=fm.Color("k", type="Q")`` draws (a
    continuous colorbar), a different render than forcing the field
    nominal gives it.
    """
    via_shorthand = figure(xy_df, x="v", y="w", hue="k:Q", ci=None).to_svg()
    via_object = figure(xy_df, x="v", y="w", hue=Color("k", type="Q"), ci=None).to_svg()
    forced_nominal = figure(xy_df, x="v", y="w", hue=Color("k", type="N"), ci=None).to_svg()
    assert via_shorthand == via_object
    assert via_shorthand != forced_nominal


def test_hue_none_leaves_no_grouping(xy_df):
    """``hue=None`` is unchanged: byte-identical to leaving ``hue=`` unset."""
    svg = fm.regplot(xy_df, x="v", y="w", hue=None, ci=None).to_svg()
    bare = fm.regplot(xy_df, x="v", y="w", ci=None).to_svg()
    assert svg == bare


@pytest.mark.parametrize("figure", [fm.regplot, fm.lmplot], ids=["regplot", "lmplot"])
def test_hue_a_non_binding_value_is_the_parses_typeerror(xy_df, figure):
    """``hue=5`` raises the parse's own ``TypeError`` naming the hue door,
    rather than leaking a raw ``AttributeError: 'int' object has no
    attribute 'option'`` from the unguarded typing helper.
    """
    with pytest.raises(TypeError) as excinfo:
        figure(xy_df, x="v", y="w", hue=5, ci=None).to_svg()
    assert str(excinfo.value) == (
        f"{figure.__name__}(hue=...) expects str, Color instance, or Repeat placeholder; got int"
    )


# ---------------------------------------------------------------------------
# `x=None`/`x=5`/`X(type=None)`: the parse's own doors, unchanged or fixed.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("figure", [fm.regplot, fm.lmplot], ids=["regplot", "lmplot"])
def test_x_none_refuses_before_the_parse(xy_df, figure):
    with pytest.raises(ValueError) as excinfo:
        figure(xy_df, x=None, y="w", ci=None).to_svg()
    assert str(excinfo.value) == (
        f"{figure.__name__}(x=None): x names the column to fit and has no default. "
        "Pass a column name, or an encoding object wrapping one."
    )


@pytest.mark.parametrize("figure", [fm.regplot, fm.lmplot], ids=["regplot", "lmplot"])
def test_x_a_non_binding_value_is_the_parses_typeerror(xy_df, figure):
    with pytest.raises(TypeError) as excinfo:
        figure(xy_df, x=5, y="w", ci=None).to_svg()
    assert str(excinfo.value) == (
        f"{figure.__name__}(x=...) expects str, X instance, or Repeat placeholder; got int"
    )


def test_x_with_an_explicit_none_type_is_unset(xy_df):
    """``X("v", type=None)`` reads exactly as ``x="v"``: ``None`` is unset,
    not a declared type the categorical rule could see.
    """
    via_none_type = fm.regplot(xy_df, x=X("v", type=None), y="w", ci=None).to_svg()
    bare = fm.regplot(xy_df, x="v", y="w", ci=None).to_svg()
    assert via_none_type == bare


# ---------------------------------------------------------------------------
# With scatter=False, the call's keyword bindings reach the fit.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scatter", [True, False])
@pytest.mark.parametrize("method", _METHODS)
def test_a_keyword_binding_reaches_the_fit_when_no_scatter_is_drawn(xy_df, method, scatter):
    """A misspelled flag (``robust=True``, not a real ``regplot``/``lmplot``
    keyword -- it is ``residplot``'s own parameter name) refuses on EVERY
    accepted method, under BOTH ``scatter=`` forms -- reaching the fit
    chart unvalidated, it would render a plain fit instead, with
    ``scatter=False`` silently dropping it.
    """
    y_col = _response_column(method)
    with pytest.raises(ValueError, match=r"unknown encoding channel: 'robust'"):
        fm.regplot(
            xy_df, x="v", y=y_col, method=method, scatter=scatter, ci=None, robust=True
        ).to_svg()


@pytest.mark.parametrize("method", ["lm", "loess", "robust", "logistic", "glm"])
@pytest.mark.parametrize("channel", ["size", "opacity", "tooltip"])
def test_a_value_channel_on_the_fit_warns_undrawn_when_no_scatter_is_drawn(xy_df, method, channel):
    """``size=``/``opacity=``/``tooltip="k"`` on a no-scatter fit, with
    ``ci=`` left at its default (a drawn CI band), warn in the figure's
    own voice, the same bytes and the same sorted warnings the sibling
    ``encode=`` door gives for a binding the statistic's own marks do not
    carry, rather than vanishing from the fit chart with no word at all.
    """
    y_col = _response_column(method)
    with pytest.warns(UserWarning) as kw_record:
        kw_svg = fm.regplot(
            xy_df, x="v", y=y_col, method=method, scatter=False, **{channel: "k"}
        ).to_svg()
    kw_messages = sorted(str(w.message) for w in kw_record)
    assert any(
        message
        == (
            f"regplot: {channel} 'k' was dropped: the statistic this line mark draws "
            f"does not carry 'k', so it draws without {channel}"
        )
        for message in kw_messages
    )

    with pytest.warns(UserWarning) as enc_record:
        enc_svg = fm.regplot(
            xy_df, x="v", y=y_col, method=method, scatter=False, encode={channel: "k"}
        ).to_svg()
    enc_messages = sorted(str(w.message) for w in enc_record)
    assert kw_svg == enc_svg
    assert kw_messages == enc_messages


@pytest.mark.parametrize("channel", ["size", "opacity"])
def test_a_value_channel_is_refused_on_the_single_line_shortcut(xy_df, channel):
    """``ci=None, show_metrics=False`` collapses the fit to a single-mark
    chart, whose own value-channel rule is ``encode=``'s single-mark-statistic
    treatment: refused, not warned -- the same outcome a direct
    ``mark_smooth(ci=None, show_metrics=False).encode(size=...)`` gives.
    """
    with pytest.raises(ValueError) as excinfo:
        fm.regplot(
            xy_df,
            x="v",
            y="w",
            method="lm",
            scatter=False,
            ci=None,
            show_metrics=False,
            **{channel: "k"},
        ).to_svg()
    assert str(excinfo.value) == "unknown column 'k' referenced by an encoding"


def test_a_tooltip_is_dropped_on_the_single_line_shortcut(xy_df):
    """``tooltip=`` on the same single-line shortcut is dropped, not
    refused: binding it changes nothing about the rendered chart, the
    same way it changes nothing on a direct
    ``mark_smooth(method="lm", ci=None, show_metrics=False).encode(tooltip=...)``
    call (a single-mark statistic's own treatment of an undrawn binding,
    left as is here) -- the sibling call names ``method="lm"`` explicitly,
    matching ``regplot``'s own ``method="lm"`` rather than ``mark_smooth``'s
    default ``method="loess"``, so the two routes compare the same fit.
    """
    without = fm.regplot(
        xy_df, x="v", y="w", method="lm", scatter=False, ci=None, show_metrics=False
    ).to_svg()
    with_tooltip = fm.regplot(
        xy_df, x="v", y="w", method="lm", scatter=False, ci=None, show_metrics=False, tooltip="g"
    ).to_svg()
    assert without == with_tooltip

    direct_without = (
        fm.Chart(xy_df)
        .mark_smooth(method="lm", ci=None, show_metrics=False)
        .encode(x="v", y="w")
        .to_svg()
    )
    direct_with_tooltip = (
        fm.Chart(xy_df)
        .mark_smooth(method="lm", ci=None, show_metrics=False)
        .encode(x="v", y="w", tooltip="g")
        .to_svg()
    )
    assert direct_without == direct_with_tooltip

    # The two routes compare the same fit end to end: the keyword form's
    # own render equals the direct mark_smooth() sibling's.
    assert with_tooltip == direct_with_tooltip


@pytest.mark.parametrize("method", ["lm", "loess", "robust", "logistic", "glm"])
@pytest.mark.parametrize("channel,value", [("text", "g"), ("x2", "v")])
def test_an_undrawn_text_or_x2_binding_matches_the_encode_door(xy_df, method, channel, value):
    """``text="g"``/``x2="v"`` on a no-scatter fit bind the same way through
    the keyword door and through ``encode=``: the same bytes and the same
    (empty) set of warnings, on every fit family.
    """
    y_col = _response_column(method)
    with warnings.catch_warnings(record=True) as kw_record:
        warnings.simplefilter("always")
        kw_svg = fm.regplot(
            xy_df, x="v", y=y_col, method=method, scatter=False, **{channel: value}
        ).to_svg()
    with warnings.catch_warnings(record=True) as enc_record:
        warnings.simplefilter("always")
        enc_svg = fm.regplot(
            xy_df, x="v", y=y_col, method=method, scatter=False, encode={channel: value}
        ).to_svg()
    assert kw_svg == enc_svg
    assert sorted(str(w.message) for w in kw_record) == sorted(str(w.message) for w in enc_record)


#: One digest per (method, grouping channel), computed on the baseline
#: tree -- ``hue=``/``color=`` group and color the fit identically
#: (``color=`` is ``hue=``'s own alias), ``detail=``/``stroke_dash=`` group
#: without a color split, and ``shape=`` draws no shape at all on a line
#: mark (its own digest matches ``detail=``'s, the ungrouped-looking
#: render a warned, undrawn channel gives).
_GROUPING_KEYWORD_DIGESTS: dict[tuple[str, str], str] = {
    ("lm", "hue"): "46b927c1277a21af69b9617c6f8fcaeec4fcc6b6759391922cf486daa3fcef51",
    ("lm", "color"): "46b927c1277a21af69b9617c6f8fcaeec4fcc6b6759391922cf486daa3fcef51",
    ("lm", "detail"): "83c75869133e7d8f6a0ccda91a0b42dfcefbc3400c15230d8c67e3505ca41032",
    ("lm", "stroke_dash"): "b57500f0b19903c2fc4c4c78edf83816ce1883754c892d5af447c936a5d5cdd8",
    ("lm", "shape"): "83c75869133e7d8f6a0ccda91a0b42dfcefbc3400c15230d8c67e3505ca41032",
    ("loess", "hue"): "e18d3da671f5309ee34b93ada72bc1b861b283ad4d811bd896707b83204aadcc",
    ("loess", "color"): "e18d3da671f5309ee34b93ada72bc1b861b283ad4d811bd896707b83204aadcc",
    ("loess", "detail"): "e01c0182bd306ff71873e775b0778ac90642a7fcd3a9130b6c8102176fa5239b",
    ("loess", "stroke_dash"): "04a3730069db3f835414956bcdba5905d212d5bf9177eea82dfc64b32cc94e3f",
    ("loess", "shape"): "e01c0182bd306ff71873e775b0778ac90642a7fcd3a9130b6c8102176fa5239b",
    ("robust", "hue"): "29f41de8fec726d99d30aaa128547604d7219c1bf625b9facf89149791ebd6a2",
    ("robust", "color"): "29f41de8fec726d99d30aaa128547604d7219c1bf625b9facf89149791ebd6a2",
    ("robust", "detail"): "69865b1c6fa9eb30531b96172fbac9ce408d334d82cb9eb2cbd3bb9fbf3af37d",
    ("robust", "stroke_dash"): "2d29c30b1565c4137614d1a67d665202419d66f1b6fc5fec13815cd5112372ab",
    ("robust", "shape"): "69865b1c6fa9eb30531b96172fbac9ce408d334d82cb9eb2cbd3bb9fbf3af37d",
    ("logistic", "hue"): "ab1895814c87deef0e8e3eac31095040b3aab4fdcc0cea3cffd21357ef8aea73",
    ("logistic", "color"): "ab1895814c87deef0e8e3eac31095040b3aab4fdcc0cea3cffd21357ef8aea73",
    ("logistic", "detail"): "15bdf768b6a66867bbd7aafc005e3021ac7b9a671159198fe1c9dea48a20a0af",
    ("logistic", "stroke_dash"): "7bc9e92b72b1439eccbea92b99c396f6c331223a0794b93cdcf9ba5ec2f42c69",
    ("logistic", "shape"): "15bdf768b6a66867bbd7aafc005e3021ac7b9a671159198fe1c9dea48a20a0af",
    ("glm", "hue"): "46b927c1277a21af69b9617c6f8fcaeec4fcc6b6759391922cf486daa3fcef51",
    ("glm", "color"): "46b927c1277a21af69b9617c6f8fcaeec4fcc6b6759391922cf486daa3fcef51",
    ("glm", "detail"): "83c75869133e7d8f6a0ccda91a0b42dfcefbc3400c15230d8c67e3505ca41032",
    ("glm", "stroke_dash"): "b57500f0b19903c2fc4c4c78edf83816ce1883754c892d5af447c936a5d5cdd8",
    ("glm", "shape"): "83c75869133e7d8f6a0ccda91a0b42dfcefbc3400c15230d8c67e3505ca41032",
}


def _digest(svg: str) -> str:
    return hashlib.sha256(svg.encode()).hexdigest()


@pytest.mark.parametrize("method", ["lm", "loess", "robust", "logistic", "glm"])
@pytest.mark.parametrize("channel", ["hue", "color", "detail", "stroke_dash", "shape"])
def test_grouping_keywords_still_fit_one_curve_per_group_with_no_scatter(xy_df, method, channel):
    """``hue=``/``color=``/``detail=``/``stroke_dash=``/``shape=`` keep
    grouping the fit: the grouping bindings are a subset of the keyword-
    binding dict the fit now binds with no scatter drawn, so one curve per
    group still renders in its own group's color. Each render's bytes are
    pinned by digest; ``shape=``, which a line mark never draws, warns the
    same undrawn-channel text ``encode=`` gives for any channel a drawn
    mark does not carry.
    """
    y_col = _response_column(method)
    kwargs = dict(x="v", y=y_col, method=method, scatter=False, ci=None)
    if channel == "shape":
        with pytest.warns(UserWarning) as record:
            svg = fm.regplot(xy_df, shape="g", **kwargs).to_svg()
        messages = [str(w.message) for w in record]
        assert any(
            message
            == (
                "regplot: shape 'g' groups the statistic by 'g', one per value, but no "
                "layer's mark draws shape (this line mark does not), so the groups are "
                "drawn alike with no scale and no legend; bind color= to tell them apart"
            )
            for message in messages
        )
    else:
        svg = fm.regplot(xy_df, **{channel: "g"}, **kwargs).to_svg()
    assert _digest(svg) == _GROUPING_KEYWORD_DIGESTS[(method, channel)]


def test_lmplot_scatter_false_with_facet_still_warns_a_dropped_value_channel(xy_df):
    """``col=``/``row=`` facets after the fit is built, so a dropped-value
    warning raised while binding the fit survives faceting unaffected.
    """
    with pytest.warns(UserWarning, match=r"size 'k' was dropped"):
        fm.lmplot(xy_df, x="v", y="w", scatter=False, ci=None, col="g", size="k").to_svg()


def test_encode_kwarg_on_every_form_is_unaffected(xy_df):
    """``encode=`` keeps landing through ``_finalize_chart`` on every form,
    scatter present or not, independently of the keyword-binding door
    above -- each form's own render pinned by digest.
    """
    with pytest.warns(UserWarning):
        with_scatter = fm.regplot(xy_df, x="v", y="w", ci=None, encode={"size": "k"}).to_svg()
    with pytest.warns(UserWarning):
        no_scatter = fm.regplot(
            xy_df, x="v", y="w", scatter=False, ci=None, encode={"size": "k"}
        ).to_svg()
    assert (
        _digest(with_scatter) == "6efee6f305670f238425fcf29dffc5396d443ef5e7ef8e9cb47b917fefc9b261"
    )
    assert _digest(no_scatter) == "92cd391cbd7960be92340d48199ed6c4a16ef193bc6b3a2a79ed8edb05bc00ce"


@pytest.mark.parametrize("scatter", [True, False])
def test_an_unbound_size_is_byte_identical_when_none(xy_df, scatter):
    """``size=None`` is unset on every form, scatter present or not."""
    bare = fm.regplot(xy_df, x="v", y="w", scatter=scatter, ci=None).to_svg()
    unset = fm.regplot(xy_df, x="v", y="w", scatter=scatter, ci=None, size=None).to_svg()
    assert bare == unset


def test_a_non_binding_size_is_chart_encodes_typeerror(xy_df):
    """``size=30`` raises ``Chart.encode``'s own ``TypeError``, on both
    ``scatter=`` forms, the same as it already does with ``scatter=True``.
    """
    for scatter in (True, False):
        with pytest.raises(TypeError, match=r"encode\(size=\.\.\.\) expects"):
            fm.regplot(xy_df, x="v", y="w", scatter=scatter, ci=None, size=30).to_svg()


# ---------------------------------------------------------------------------
# Zero-row and one-row data.
# ---------------------------------------------------------------------------


def test_degenerate_data_behaves_the_same_with_shorthand(xy_df):
    """Zero-row and one-row data with ``x="v:Q"`` renders the same output
    and degenerate-fit warning (by text, not just ``pytest.warns``) as
    ``x="v"``; ``x="v:N"`` is refused before any row is read.
    """
    empty = xy_df.filter(pl.col("v") > 1000.0)
    one_row = xy_df.head(1)
    for frame in (empty, one_row):
        with pytest.warns(UserWarning) as bare_record:
            bare = fm.regplot(frame, x="v", y="w", ci=None).to_svg()
        with pytest.warns(UserWarning) as typed_record:
            typed = fm.regplot(frame, x="v:Q", y="w", ci=None).to_svg()
        assert bare == typed
        assert sorted(str(w.message) for w in bare_record) == sorted(
            str(w.message) for w in typed_record
        )

        with pytest.raises(EntryRefusal):
            fm.regplot(frame, x="v:N", y="w", ci=None).to_svg()


def test_degenerate_data_behaves_the_same_with_a_shorthand_facet_column(xy_df):
    """``col="g:N"`` behaves the same as ``col="g"`` on degenerate data:
    zero rows refuses with the same ``ValueError`` text (the facet layout
    sees no column either way), and one row renders byte-identical, with
    the same degraded-fit warning.
    """
    empty = xy_df.filter(pl.col("v") > 1000.0)
    one_row = xy_df.head(1)

    with pytest.raises(ValueError) as bare_excinfo:
        fm.lmplot(empty, x="v", y="w", ci=None, col="g").to_svg()
    with pytest.raises(ValueError) as typed_excinfo:
        fm.lmplot(empty, x="v", y="w", ci=None, col="g:N").to_svg()
    assert str(bare_excinfo.value) == str(typed_excinfo.value)

    with pytest.warns(UserWarning) as bare_record:
        bare = fm.lmplot(one_row, x="v", y="w", ci=None, col="g").to_svg()
    with pytest.warns(UserWarning) as typed_record:
        typed = fm.lmplot(one_row, x="v", y="w", ci=None, col="g:N").to_svg()
    assert bare == typed
    assert sorted(str(w.message) for w in bare_record) == sorted(
        str(w.message) for w in typed_record
    )
