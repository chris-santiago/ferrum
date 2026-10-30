"""A figure that re-points a caller's own x/y binding at a statistic's
output column keeps every option the caller wrote on that channel, not only
its title: ``displot(kind="hist")``, ``displot(kind="ecdf")``,
``residplot``'s primary bindings and ``displot(log_scale=True)`` all run
through the same re-pointing rule (``_rebind_channel``/``_apply_remap``), so
a `scale=` (with its rows-left-out warning), `axis=`, `format=` and
`title=` all survive on every door -- the same thing `.xlim()` already gets
on any chart.
"""

from __future__ import annotations

import re
import warnings
from typing import Any, Callable

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from tests._svg_extents import axis_tick_labels, x_axis_extents, x_axis_title, y_axis_title


def _warnings_of(render: Callable[[], Any]) -> tuple[list[str], Any]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = render()
    return [str(w.message) for w in caught], result


def _domain_messages(msgs: list[str]) -> list[str]:
    return [m for m in msgs if "explicit domain" in m]


#: 20 rows, `w = 2v`, shared by every x/y-channel-option door below.
_DF = pl.DataFrame({"v": [float(i) for i in range(20)], "w": [float(i) * 2 for i in range(20)]})

#: `_DF` plus a two-level grouping column, for the hue/facet doors.
_GROUPED_DF = _DF.with_columns(g=pl.Series(["a", "b"] * 10))


# ---------------------------------------------------------------------------
# A caller's x `scale=` governs the axis exactly as `.xlim()` does, on every
# door that re-points a positional channel at a statistic's output: ecdf's
# bin start, residplot's residual batch, and the already-honored hist bin
# start as a control. The hue and column-facet doors extend this to ecdf's
# grouped and per-panel renders, and the overlay doors extend it to a
# layered kde=/rug= chart.
# ---------------------------------------------------------------------------

#: Each door's builder, keyed by name, taking the x channel to bind.
_DOMAIN_DOORS: dict[str, Callable[[Any], Any]] = {
    "displot ecdf": lambda x: fm.displot(_DF, x=x, kind="ecdf"),
    "displot ecdf+kde": lambda x: fm.displot(_DF, x=x, kind="ecdf", kde=True),
    "residplot": lambda x: fm.residplot(_DF, x=x, y="w"),
    "displot hist": lambda x: fm.displot(_DF, x=x, kind="hist"),
}


@pytest.mark.parametrize("door", sorted(_DOMAIN_DOORS))
def test_a_callers_x_domain_warns_the_same_as_xlim(door):
    """An explicit `scale=` domain on the x channel warns about the rows it
    leaves out, identically to the same bounds passed through `.xlim()`; a
    plain call (no domain at all) warns nothing."""
    build = _DOMAIN_DOORS[door]

    def via_scale():
        return build(fm.X("v", scale=fm.LinearScale(domain=[5.0, 6.0]))).to_svg()

    def via_xlim():
        return build("v").xlim(5, 6).to_svg()

    def plain():
        return build("v").to_svg()

    scale_msgs, _ = _warnings_of(via_scale)
    xlim_msgs, _ = _warnings_of(via_xlim)
    plain_msgs, _ = _warnings_of(plain)
    scale_msgs = _domain_messages(scale_msgs)
    xlim_msgs = _domain_messages(xlim_msgs)
    plain_msgs = _domain_messages(plain_msgs)

    assert scale_msgs == xlim_msgs, door
    assert scale_msgs != [], f"{door}: scale= produced no domain warning"
    assert plain_msgs == [], f"{door}: a plain call warned with no domain set"


def test_ecdf_hue_domain_warns_the_same_as_xlim():
    """An ecdf `scale=` domain warns on a `hue=`-grouped chart the same way
    `.xlim()` does, over every group's line together."""

    def via_scale():
        return fm.displot(
            _GROUPED_DF, x=fm.X("v", scale=fm.LinearScale(domain=[5.0, 6.0])), hue="g", kind="ecdf"
        ).to_svg()

    def via_xlim():
        return fm.displot(_GROUPED_DF, x="v", hue="g", kind="ecdf").xlim(5, 6).to_svg()

    def plain():
        return fm.displot(_GROUPED_DF, x="v", hue="g", kind="ecdf").to_svg()

    scale_msgs, _ = _warnings_of(via_scale)
    xlim_msgs, _ = _warnings_of(via_xlim)
    plain_msgs, _ = _warnings_of(plain)
    assert _domain_messages(scale_msgs) == _domain_messages(xlim_msgs)
    assert _domain_messages(scale_msgs) != []
    assert _domain_messages(plain_msgs) == []


def test_ecdf_column_facet_shares_one_x_domain_across_panels():
    """An ecdf `scale=` domain governs every `col=` panel's x axis equally,
    one domain across panels rather than each panel inferring its own."""
    svg = fm.displot(
        _GROUPED_DF, x=fm.X("v", scale=fm.LinearScale(domain=[5.0, 6.0])), col="g", kind="ecdf"
    ).to_svg()
    extents = x_axis_extents(svg)
    assert len(extents) == 2
    assert extents[0] == extents[1] == (5.0, 6.0)

    plain_extents = x_axis_extents(fm.displot(_GROUPED_DF, x="v", col="g", kind="ecdf").to_svg())
    assert len(plain_extents) == 2
    assert plain_extents[0] == plain_extents[1]
    assert plain_extents[0] != (5.0, 6.0)


# ---------------------------------------------------------------------------
# `axis=None` and `format=` on a caller's x channel, on every door that
# re-points a positional channel: ecdf (bare and with a kde=/rug= overlay)
# and residplot.
# ---------------------------------------------------------------------------

_AXIS_DOORS: dict[str, Callable[[Any], Any]] = {
    "displot ecdf": lambda x: fm.displot(_DF, x=x, kind="ecdf"),
    "displot ecdf+kde": lambda x: fm.displot(_DF, x=x, kind="ecdf", kde=True),
    "displot ecdf+rug": lambda x: fm.displot(_DF, x=x, kind="ecdf", rug=True),
    "residplot": lambda x: fm.residplot(_DF, x=x, y="w"),
}


@pytest.mark.parametrize("door", sorted(_AXIS_DOORS))
def test_axis_none_hides_the_x_axis(door):
    """`axis=None` on the x channel removes its title and ticks; the same
    call with no `axis=` draws both."""
    build = _AXIS_DOORS[door]
    svg = build(fm.X("v", axis=None)).to_svg()
    assert x_axis_title(svg) is None, door
    assert axis_tick_labels(svg, axis="x") == [], door

    plain_svg = build("v").to_svg()
    assert x_axis_title(plain_svg) is not None, door
    assert axis_tick_labels(plain_svg, axis="x") != [], door


@pytest.mark.parametrize("door", sorted(_AXIS_DOORS))
def test_format_applies_to_the_x_ticks(door):
    """`format=` on the x channel formats every rendered tick label; the
    same call with no `format=` renders plain ticks."""
    build = _AXIS_DOORS[door]
    svg = build(fm.X("v", format="+.2f")).to_svg()
    labels = axis_tick_labels(svg, axis="x")
    assert labels, door
    assert all(label.startswith("+") for label in labels), (door, labels)

    plain_svg = build("v").to_svg()
    plain_labels = axis_tick_labels(plain_svg, axis="x")
    assert plain_labels, door
    assert not any(label.startswith("+") for label in plain_labels), (door, plain_labels)


# ---------------------------------------------------------------------------
# `residplot`'s y channel (the residual axis): `scale=`, `axis=`, `format=`
# and the default title.
# ---------------------------------------------------------------------------


def test_residplot_y_domain_warns_the_same_as_ylim():
    """An explicit `scale=` domain on residplot's y channel warns about the
    residuals it leaves out, identically to the same bounds through
    `.ylim()`; a plain call warns nothing."""

    def via_scale():
        return fm.residplot(
            _DF, x="v", y=fm.Y("w", scale=fm.LinearScale(domain=[5.0, 6.0]))
        ).to_svg()

    def via_ylim():
        return fm.residplot(_DF, x="v", y="w").ylim(5, 6).to_svg()

    def plain():
        return fm.residplot(_DF, x="v", y="w").to_svg()

    scale_msgs, _ = _warnings_of(via_scale)
    ylim_msgs, _ = _warnings_of(via_ylim)
    plain_msgs, _ = _warnings_of(plain)
    assert _domain_messages(scale_msgs) == _domain_messages(ylim_msgs)
    assert _domain_messages(scale_msgs) != []
    assert _domain_messages(plain_msgs) == []


def test_residplot_y_axis_none_hides_the_y_axis():
    """`axis=None` on residplot's y channel removes its title and ticks; the
    same call with no `axis=` draws both."""
    svg = fm.residplot(_DF, x="v", y=fm.Y("w", axis=None)).to_svg()
    assert y_axis_title(svg) is None
    assert axis_tick_labels(svg, axis="y") == []

    plain_svg = fm.residplot(_DF, x="v", y="w").to_svg()
    assert y_axis_title(plain_svg) is not None
    assert axis_tick_labels(plain_svg, axis="y") != []


def test_residplot_y_format_applies_to_the_y_ticks():
    """`format=` on residplot's y channel formats every rendered tick label
    (two decimals, an explicit sign); the same call with no `format=`
    renders plain ticks."""
    svg = fm.residplot(_DF, x="v", y=fm.Y("w", format="+.2f")).to_svg()
    labels = axis_tick_labels(svg, axis="y")
    assert labels
    assert all(re.fullmatch(r"[+-]\d+\.\d{2}", label) for label in labels), labels
    assert any(label.startswith("+") for label in labels), labels

    plain_svg = fm.residplot(_DF, x="v", y="w").to_svg()
    plain_labels = axis_tick_labels(plain_svg, axis="y")
    assert plain_labels
    assert not any(label.startswith("+") for label in plain_labels), plain_labels


def test_residplot_y_without_an_explicit_title_still_titles_w_residual():
    """A y channel carrying `scale=` but no `title=` still falls back to
    the default `"<y> residual"` title, the same as a bare string `y=`."""
    svg = fm.residplot(_DF, x="v", y=fm.Y("w", scale=fm.LinearScale())).to_svg()
    assert y_axis_title(svg) == "w residual"


# ---------------------------------------------------------------------------
# ecdf's fieldless `y=` styles the count axis, exactly as `kind="hist"` does.
# ---------------------------------------------------------------------------

#: 20 rows engineered so BOTH kinds' value axis has a mark outside [0, 5]:
#: ten rows share one bin (hist's count there exceeds 5), and ecdf's
#: cumulative count climbs well past 5 too.
_COUNT_DF = pl.DataFrame({"v": [0.0] * 10 + [float(i) for i in range(1, 11)]})


@pytest.mark.parametrize("kind", ["hist", "ecdf"])
def test_ecdf_fieldless_y_title_and_domain_honored_like_hist(kind):
    """A fieldless `fm.Y(title=, scale=)` on the count axis titles and warns
    identically for `kind="ecdf"` and `kind="hist"`."""
    y = fm.Y(title="Cum", scale=fm.LinearScale(domain=[0.0, 5.0]))
    msgs, svg = _warnings_of(lambda: fm.displot(_COUNT_DF, x="v", y=y, kind=kind).to_svg())
    msgs = _domain_messages(msgs)
    assert msgs != [], kind
    assert all("'Cum'" in m for m in msgs), (kind, msgs)
    assert y_axis_title(svg) == "Cum", kind


def test_ecdf_fieldless_y_axis_none_hides_the_y_axis():
    """`axis=None` on ecdf's fieldless `y=` removes the count axis's title
    and ticks; the same call with no `axis=` draws both."""
    svg = fm.displot(_DF, x="v", y=fm.Y(axis=None), kind="ecdf").to_svg()
    assert y_axis_title(svg) is None
    assert axis_tick_labels(svg, axis="y") == []

    plain_svg = fm.displot(_DF, x="v", kind="ecdf").to_svg()
    assert y_axis_title(plain_svg) is not None
    assert axis_tick_labels(plain_svg, axis="y") != []


def test_ecdf_fieldless_y_format_applies_to_the_count_ticks():
    """`format=` on ecdf's fieldless `y=` formats every rendered count tick
    label; the same call with no `format=` renders plain ticks."""
    svg = fm.displot(_DF, x="v", y=fm.Y(format="+.1f"), kind="ecdf").to_svg()
    labels = axis_tick_labels(svg, axis="y")
    assert labels
    assert all(label.startswith("+") for label in labels), labels

    plain_svg = fm.displot(_DF, x="v", kind="ecdf").to_svg()
    plain_labels = axis_tick_labels(plain_svg, axis="y")
    assert plain_labels
    assert not any(label.startswith("+") for label in plain_labels), plain_labels


# ---------------------------------------------------------------------------
# ecdf refuses a `y=` that names a field.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("y", ["w", lambda: fm.Y("w")], ids=["bare string", "channel"])
def test_ecdf_refuses_a_field_bearing_y(y):
    """`kind="ecdf"` with a `y=` that names a field (a string or a channel
    with one) raises `ValueError` naming the call and the remedy, rather
    than silently dropping the binding or re-pointing it at the count
    column under the caller's field title."""
    y_value = y() if callable(y) else y
    with pytest.raises(ValueError) as excinfo:
        fm.displot(_DF, x="v", y=y_value, kind="ecdf")
    message = str(excinfo.value)
    assert "displot(kind='ecdf', y='w')" in message
    assert "fm.Y(title=..., scale=...)" in message


# ---------------------------------------------------------------------------
# `log_scale=True` reaches a channel `x=`, ecdf, and every kde=/rug=
# overlay, on both the already-binning (hist) and the eagerly-bound (ecdf)
# primary kinds.
# ---------------------------------------------------------------------------

#: A positive-skewed sample (seed fixed) whose Sturges bin edges put at
#: least one bin_start at or below 0 -- the one value a `LogScale` cannot
#: place, so its own typed warning is proof the resolved scale is
#: genuinely `log`, not merely accepted-and-ignored.
_rng = np.random.default_rng(0)
_LOG_DF = pl.DataFrame({"v": (np.exp(_rng.normal(2, 2.5, 300)) + 0.5).tolist()})

_LOG_SCALE_WARNING = "which a LogScale cannot place"


def test_log_scale_with_a_channel_x_renders():
    """A channel `x=fm.X("v", title="Value")` with `log_scale=True` renders:
    the channel's own `title=` survives the rebuild, and the resolved scale
    is genuinely `log` (the typed `LogScale` warning is the proof), not
    silently accepted-and-linear."""
    msgs, svg = _warnings_of(
        lambda: fm.displot(
            _LOG_DF, x=fm.X("v", title="Value"), kind="hist", log_scale=True
        ).to_svg()
    )
    assert x_axis_title(svg) == "Value"
    assert any(_LOG_SCALE_WARNING in m for m in msgs)


def test_log_scale_on_ecdf_renders():
    """`kind="ecdf"` with `log_scale=True` renders, with the `LogScale`
    warning as proof the scale is real; the same call with no `log_scale`
    never emits it."""
    msgs, _ = _warnings_of(lambda: fm.displot(_LOG_DF, x="v", kind="ecdf", log_scale=True).to_svg())
    assert any(_LOG_SCALE_WARNING in m for m in msgs)

    control_msgs, _ = _warnings_of(lambda: fm.displot(_LOG_DF, x="v", kind="ecdf").to_svg())
    assert not any(_LOG_SCALE_WARNING in m for m in control_msgs)


#: (primary kind, overlay flag) -- every kind/overlay combination a figure
#: can layer, so `log_scale=True` is proven to reach each one.
_OVERLAY_LOG_CASES = [("hist", "kde"), ("hist", "rug"), ("ecdf", "kde"), ("ecdf", "rug")]


@pytest.mark.parametrize(
    "kind,overlay", _OVERLAY_LOG_CASES, ids=[f"{k}+{o}" for k, o in _OVERLAY_LOG_CASES]
)
def test_log_scale_reaches_a_layered_overlay(kind, overlay):
    """A layered `kde=`/`rug=` overlay inherits `log_scale=True` from the
    chart-level x binding its panel shares; the same call with no
    `log_scale` never emits the `LogScale` warning."""
    kwargs = {overlay: True}

    msgs, _ = _warnings_of(
        lambda: fm.displot(_LOG_DF, x="v", kind=kind, log_scale=True, **kwargs).to_svg()
    )
    assert any(_LOG_SCALE_WARNING in m for m in msgs), (kind, overlay)

    control_msgs, _ = _warnings_of(lambda: fm.displot(_LOG_DF, x="v", kind=kind, **kwargs).to_svg())
    assert not any(_LOG_SCALE_WARNING in m for m in control_msgs), (kind, overlay)


_LOG_SCALE_REFUSAL = (
    "displot(log_scale=True, x=<channel with its own scale=>): two instructions "
    "for one x scale. Keep your scale= and drop log_scale=True, or pass x as a "
    "column name and let log_scale set the log scale."
)


def test_log_scale_refuses_a_channel_that_already_carries_its_own_scale():
    """A channel `x=` that already carries its own `scale=` raises the
    exact two-instructions refusal, voiced `displot(log_scale=True, ...)` --
    the same shape `lmplot(logx=True)` gives, not silently overruled in
    either direction."""
    with pytest.raises(ValueError) as excinfo:
        fm.displot(_DF, x=fm.X("v", scale=fm.LinearScale()), kind="hist", log_scale=True)
    assert str(excinfo.value) == _LOG_SCALE_REFUSAL


_LMPLOT_LOGX_REFUSAL = (
    "lmplot(logx=True, x=<channel with its own scale=>): two instructions "
    "for one x scale. Keep your scale= and drop logx=True, or pass x as a "
    "column name and let logx set the log scale."
)


def test_lmplot_logx_refusal_names_logx_not_log_scale():
    """`lmplot(logx=True)`'s refusal names `logx`, not `log_scale`, even
    though both figures share the same underlying helper."""
    with pytest.raises(ValueError) as excinfo:
        fm.lmplot(_DF, x=fm.X("v", scale=fm.LinearScale()), y="w", logx=True)
    assert str(excinfo.value) == _LMPLOT_LOGX_REFUSAL


# ---------------------------------------------------------------------------
# Sibling-route byte identity: a bare field name and the equivalent channel
# object with no extra options render identically, on every door that
# re-points a caller channel at a statistic's output column.
# ---------------------------------------------------------------------------


def test_ecdf_bare_x_and_channel_x_render_identically():
    bare = fm.displot(_DF, x="v", kind="ecdf").to_svg()
    channel = fm.displot(_DF, x=fm.X("v"), kind="ecdf").to_svg()
    assert bare == channel


def test_residplot_bare_x_and_channel_x_render_identically():
    bare = fm.residplot(_DF, x="v", y="w").to_svg()
    channel = fm.residplot(_DF, x=fm.X("v"), y="w").to_svg()
    assert bare == channel


def test_residplot_bare_y_and_channel_y_render_identically():
    bare = fm.residplot(_DF, x="v", y="w").to_svg()
    channel = fm.residplot(_DF, x="v", y=fm.Y("w")).to_svg()
    assert bare == channel


@pytest.mark.parametrize("kind", ["hist", "kde", "ecdf"])
def test_log_scale_bare_x_and_channel_x_render_identically(kind):
    df = pl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0] * 10})
    bare = fm.displot(df, x="v", kind=kind, log_scale=True).to_svg()
    channel = fm.displot(df, x=fm.X("v"), kind=kind, log_scale=True).to_svg()
    assert bare == channel, kind
